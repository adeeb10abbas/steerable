#!/usr/bin/env python3
"""Write a schema-valid Cosmos v3 infrastructure attempt outside denominators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.v3.cosmos_droid.compile_pair import _file_record, _load_object
from experiments.v3.cosmos_droid.contract import (
    MODEL_CONTRACTS,
    AuthorizedPair,
    ContractError,
    load_authorized_pair,
    verify_runtime_identity,
)
from tools.vla_wam_v3_episode_schema import (
    INFRASTRUCTURE_SCHEMA_VERSION,
    MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID,
    validate_infrastructure_record,
    write_jsonl,
)


def build_infrastructure_record(
    *, pair: AuthorizedPair, relation: str, attempt_path: Path,
    runtime: dict[str, Any], output_jsonl: Path,
) -> dict[str, Any]:
    """Build a technical/partial record without fabricating missing artifacts."""

    cell = pair.cell(relation)
    attempt = _load_object(attempt_path)
    if attempt.get("schema_version") != "vla-wam-shared-v3-cosmos-infrastructure-export-v1":
        raise ContractError("unexpected infrastructure export schema")
    for key, expected in {
        "study_id": cell["study_id"],
        "registered_cell_id": cell["cell_id"],
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "requested_relation": relation,
        "environment_seed": pair.seed,
        "sampling_seed": pair.seed,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }.items():
        if attempt.get(key) != expected:
            raise ContractError(f"infrastructure export mismatch for {key}")
    log = _file_record(attempt.get("log_path"), "infrastructure log")
    artifacts: dict[str, Any] = {
        "raw_result_jsonl": {
            "path": str(output_jsonl.resolve()),
            "integrity_scope": "batch_manifest_after_close",
        },
    }
    for key, source_key, label in (
        ("viewport_video", "viewport_video_path", "partial viewport video"),
        (
            "executed_action_trace",
            "executed_action_trace_path",
            "partial executed action trace",
        ),
    ):
        value = attempt.get(source_key)
        if value is not None:
            artifacts[key] = _file_record(value, label)
    record = {
        "schema_version": INFRASTRUCTURE_SCHEMA_VERSION,
        "record_type": "infrastructure_attempt",
        "behavioral_result_valid": False,
        "classification": attempt.get("classification"),
        "study_id": cell["study_id"],
        "registered_cell_id": cell["cell_id"],
        "attempt_id": attempt.get("attempt_id"),
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "arena": cell["arena"],
        "environment_seed": pair.seed,
        "policy_seed": pair.seed,
        "requested_relation": relation,
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": cell["success_predicate_id"],
        "reset_id": cell["reset_identity"],
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": {
            "id": MODEL_CONTRACTS[pair.model_id]["checkpoint_id"],
            "revision": MODEL_CONTRACTS[pair.model_id]["checkpoint_revision"],
        },
        "runtime_identity": {
            "id": f"{pair.model_id}:{runtime['runtime_identity_sha256'][:16]}",
            "sha256": runtime["runtime_identity_sha256"],
        },
        "artifacts": artifacts,
        "stage": attempt.get("stage"),
        "error": attempt.get("error"),
        "log_hash": log["sha256"],
        "log": log,
        "runtime_intervention": attempt.get("runtime_intervention"),
        "repair_attempt_id": attempt.get("repair_attempt_id"),
        "event_timeline": attempt.get("event_timeline"),
        "queue_sha256": pair.queue_sha256,
        "denominator_policy": "excluded_from_behavioral_denominator",
    }
    return validate_infrastructure_record(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--relation", choices=["left", "right"], required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    pair = load_authorized_pair(args.study_root, args.model_id, args.seed)
    runtime = verify_runtime_identity(args.study_root, args.model_id, args.runtime_manifest)
    record = build_infrastructure_record(
        pair=pair, relation=args.relation, attempt_path=args.attempt,
        runtime=runtime, output_jsonl=args.output_jsonl,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_jsonl(args.output_jsonl, [record])
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
