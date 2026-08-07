#!/usr/bin/env python3
"""Build one hash-bound V3-C001 model release after every gate passes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import (
    EXPERIMENT_ID,
    MODEL_CONTRACTS,
    PROMPT_FORMS,
    SCHEMA_PREFIX,
    canonical_json_bytes,
    load_json,
    load_jsonl,
    prompt_sha256,
    sha256_file,
    validate_release_manifest,
)


class ReleaseBuildError(ValueError):
    """Raised when a model remains unreleased."""


def _proof(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseBuildError(f"proof is missing: {path}")
    return {"proof_path": str(path), "proof_sha256": sha256_file(path)}


def _validate_phase_a(model_id: str, path: Path, runtime_sha: str) -> None:
    value = load_json(path)
    if value.get("model_id") != model_id:
        raise ReleaseBuildError("Phase-A proof model_id mismatch")
    if model_id == "groot_n17_droid_vla":
        required = (
            "behavioral_release",
            "model_blind_neutral_reset_fixture_passed",
            "raw_video_action_jsonl_write_passed",
            "fixed_observation_exact_repeat_passed",
            "fixed_observation_left_right_prompt_sensitivity_passed",
        )
        if not all(value.get(key) is True for key in required):
            raise ReleaseBuildError("GR00T Phase-A release proof did not pass")
        if value.get("runtime_identity_sha256") != runtime_sha:
            raise ReleaseBuildError("GR00T Phase-A runtime binding mismatch")
    else:
        if value.get("status") != "passed" or value.get("runtime_identity_sha256") != runtime_sha:
            raise ReleaseBuildError("Cosmos Phase-A release proof did not pass")


def _validate_write_proof(model_id: str, path: Path) -> None:
    value = load_json(path)
    if value.get("model_id") != model_id:
        raise ReleaseBuildError("write proof model_id mismatch")
    if value.get("passed") is not True:
        raise ReleaseBuildError("raw write proof did not pass")
    required = {
        "simulator_viewport_video",
        "executed_actions",
        "state_trace",
        "behavioral_jsonl",
    }
    if set(value.get("outputs", [])) != required:
        raise ReleaseBuildError("raw write proof does not cover all registered outputs")
    if value.get("behavioral_episode_count") != 0 or value.get("model_request_count") != 0:
        raise ReleaseBuildError("raw write proof must be model-blind")


def _prompt_report(requests: Path, model_id: str, output: Path) -> dict[str, Any]:
    rows = [row for row in load_jsonl(requests) if row.get("model_id") == model_id]
    if len(rows) != 12:
        raise ReleaseBuildError("prompt registry is incomplete")
    for row in rows:
        if row.get("prompt_sha256") != prompt_sha256(row.get("prompt", "")):
            raise ReleaseBuildError("prompt byte hash mismatch")
    report = {
        "schema_version": f"{SCHEMA_PREFIX}-prompt-byte-gate-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "passed": True,
        "request_count": 12,
        "prompt_forms": list(PROMPT_FORMS),
        "requests": _proof(requests),
        "prompt_sha256": sorted({row["prompt_sha256"] for row in rows}),
        "behavioral_episode_count": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(report))
    return report


def build(
    *,
    model_id: str,
    registration_manifest: Path,
    requests: Path,
    runtime_identity: Path,
    phase_a_proof: Path,
    fixed_observation_report: Path,
    raw_write_proof: Path,
    prompt_report: Path,
    output: Path,
) -> dict[str, Any]:
    contract = MODEL_CONTRACTS[model_id]
    runtime_file_sha = sha256_file(runtime_identity)
    runtime_value = load_json(runtime_identity)
    semantic_sha = runtime_value.get("runtime_identity_sha256", runtime_file_sha)
    if semantic_sha != contract["phase_a_runtime_identity_sha256"]:
        raise ReleaseBuildError("runtime does not match the registered Phase-A identity")
    checkpoint = runtime_value.get("checkpoint_identifier")
    revision = runtime_value.get("checkpoint_revision")
    if checkpoint != contract["checkpoint"] or revision != contract["checkpoint_revision"]:
        raise ReleaseBuildError("checkpoint identity changed")
    _validate_phase_a(model_id, phase_a_proof, semantic_sha)
    fixed = load_json(fixed_observation_report)
    if (
        fixed.get("model_id") != model_id
        or fixed.get("passed") is not True
        or fixed.get("exact_repeat_passed") is not True
        or fixed.get("prompt_only_sensitivity_passed") is not True
        or set(fixed.get("prompt_forms", {})) != set(PROMPT_FORMS)
        or fixed.get("behavioral_episode_count") != 0
    ):
        raise ReleaseBuildError("fixed-observation report did not release all four prompt forms")
    _validate_write_proof(model_id, raw_write_proof)
    _prompt_report(requests, model_id, prompt_report)

    release = {
        "schema_version": f"{SCHEMA_PREFIX}-release-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": model_id,
        "registration_manifest_sha256": sha256_file(registration_manifest),
        "runtime_identity": {
            "semantic_sha256": semantic_sha,
            "checkpoint": contract["checkpoint"],
            "checkpoint_revision": contract["checkpoint_revision"],
            "path": str(runtime_identity),
            "file_sha256": runtime_file_sha,
        },
        "phase_a_direct_release": {
            "passed": True,
            "runtime_identity_match": True,
            **_proof(phase_a_proof),
        },
        "gates": {
            "prompt_byte_hash": {"passed": True, **_proof(prompt_report)},
            "fixed_observation_exact_repeat": {"passed": True, **_proof(fixed_observation_report)},
            "fixed_observation_prompt_only_sensitivity": {
                "passed": True,
                "prompt_forms": list(PROMPT_FORMS),
                **_proof(fixed_observation_report),
            },
            "raw_video_action_jsonl_state_write": {"passed": True, **_proof(raw_write_proof)},
        },
        "behavioral_release": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(release))
    validate_release_manifest(
        output,
        model_id=model_id,
        registration_manifest_sha256=sha256_file(registration_manifest),
    )
    return release


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=tuple(MODEL_CONTRACTS), required=True)
    parser.add_argument("--registration-manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--phase-a-proof", type=Path, required=True)
    parser.add_argument("--fixed-observation-report", type=Path, required=True)
    parser.add_argument("--raw-write-proof", type=Path, required=True)
    parser.add_argument("--prompt-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = build(**vars(args))
    print(json.dumps(release, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
