#!/usr/bin/env python3
"""Compile one Cosmos v3 matched pair into validated raw episode JSONL.

The simulator-side adapter must export one JSON object per direction.  This
compiler is intentionally strict: it will not infer detached release, contact,
or missing future evidence from actions or success.  A failed compile is an
infrastructure attempt and must be recorded by the orchestration ledger before
repairing the identical registered pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v3.cosmos_droid.contract import (
    MODEL_CONTRACTS,
    AuthorizedPair,
    ContractError,
    load_authorized_pair,
    sha256_file,
    verify_release_gate,
    verify_runtime_identity,
)
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION,
    MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID,
    derive_failure_taxonomy,
    derive_initial_state_sha256,
    derive_measurements,
    validate_behavioral_record,
    write_jsonl,
)


SUSTAINED_STEPS = 3
PICKUP_LIFT_M = 0.03


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"cannot read cell export {path}: {error}") from error
    if not isinstance(value, dict):
        raise ContractError(f"cell export must be an object: {path}")
    return value


def _file_record(path_value: Any, name: str) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        raise ContractError(f"{name} path is required")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise ContractError(f"missing {name}: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _first_sustained(values: list[bool]) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run == SUSTAINED_STEPS:
            return index - SUSTAINED_STEPS + 1
    return None


def _in_cone(step: dict[str, Any], relation: str) -> bool:
    obj = np.asarray(step["object_xyz"], dtype=np.float64)
    ref = np.asarray(step["reference_xyz"], dtype=np.float64)
    if obj.shape != (3,) or ref.shape != (3,) or not np.isfinite(obj).all() or not np.isfinite(ref).all():
        raise ContractError("each raw state sample requires finite object/reference XYZ")
    delta = obj - ref
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    requested = float(delta[1]) if relation == "left" else -float(delta[1])
    return horizontal > 1e-8 and requested / horizontal >= math.cos(math.radians(45.0))


def _event_timeline(export: dict[str, Any], relation: str) -> list[dict[str, Any]]:
    steps = export["steps"]
    requested = [_in_cone(step, relation) for step in steps]
    opposite = [_in_cone(step, "right" if relation == "left" else "left") for step in steps]
    initial_z = float(steps[0]["object_xyz"][2])
    pickup = _first_sustained([
        float(step["object_xyz"][2]) - initial_z >= PICKUP_LIFT_M for step in steps
    ])
    contact = None
    if all("contact_detected" in step for step in steps):
        contact = next((i for i, step in enumerate(steps) if step["contact_detected"]), None)
    events: list[tuple[str, int]] = [("episode_start", 0)]
    for name, step in (
        ("first_contact", contact),
        ("verified_pickup", pickup),
        ("requested_region_entry", next((i for i, v in enumerate(requested) if v), None)),
        ("opposite_region_entry", next((i for i, v in enumerate(opposite) if v), None)),
    ):
        if step is not None:
            events.append((name, step))
    final_step = len(steps) - 1
    events.append(("episode_end", final_step))
    return [
        {"event": name, "action_step": step}
        for name, step in sorted(events, key=lambda item: (item[1], item[0] == "episode_end"))
    ]


def _validate_raw_steps(export: dict[str, Any]) -> list[dict[str, Any]]:
    raw = export.get("steps")
    if not isinstance(raw, list) or len(raw) < 2:
        raise ContractError("steps must retain the initial state plus at least one post-action state")
    contact_presence = {"contact_detected" in step for step in raw if isinstance(step, dict)}
    if len(contact_presence) != 1:
        raise ContractError("contact_detected must be present on every state sample or none")
    normalized = []
    for index, step in enumerate(raw):
        if not isinstance(step, dict) or step.get("action_step") != index:
            raise ContractError("state samples must use contiguous action_step values from zero")
        row = {
            "action_step": index,
            "object_xyz": step.get("object_xyz"),
            "reference_xyz": step.get("reference_xyz"),
            "grippers_open": step.get("grippers_open"),
        }
        _in_cone(row, "left")
        if type(row["grippers_open"]) is not bool:
            raise ContractError("grippers_open must be a raw boolean on every state sample")
        if "contact_detected" in step:
            if type(step["contact_detected"]) is not bool:
                raise ContractError("contact_detected must be boolean")
            row["contact_detected"] = step["contact_detected"]
        normalized.append(row)
    return normalized


def _future_evidence(
    export: dict[str, Any], actions_executed: int, expected_seed: int
) -> list[dict[str, Any]]:
    requests = export.get("policy_requests")
    if not isinstance(requests, list) or len(requests) != math.ceil(actions_executed / 32):
        raise ContractError("one action/future record is required for every 32-action policy request")
    result = []
    for expected_index, request in enumerate(requests):
        if not isinstance(request, dict) or request.get("request_index") != expected_index:
            raise ContractError("policy request indices must be contiguous from zero")
        if request.get("sampling_seed") != expected_seed:
            raise ContractError("policy request does not retain the matched pair sampling seed")
        action = _file_record(request.get("returned_action_path"), "returned action chunk")
        future = _file_record(request.get("decoded_future_path"), "decoded future")
        action_array = np.load(action["path"], allow_pickle=False)
        future_array = np.load(future["path"], allow_pickle=False)
        if action_array.shape != (32, 8):
            raise ContractError("returned action chunk must have shape [32,8]")
        if future_array.ndim != 4 or future_array.shape[0] != 33 or future_array.shape[-1] != 3:
            raise ContractError("decoded future must be a retained 33-frame RGB array")
        result.append({
            "request_index": expected_index,
            "sampling_seed": request.get("sampling_seed"),
            "returned_action": action,
            "returned_action_shape": list(action_array.shape),
            "decoded_future": future,
            "decoded_future_shape": list(future_array.shape),
            "future_evidence_status": "exposed_and_retained",
        })
    return result


def build_behavioral_record(
    *, pair: AuthorizedPair, relation: str, export_path: Path,
    runtime: dict[str, Any], raw_jsonl_path: Path,
) -> dict[str, Any]:
    """Build and validate exactly one behavioral episode record."""

    cell = pair.cell(relation)
    export = _load_object(export_path)
    expected_values = {
        "schema_version": "vla-wam-shared-v3-cosmos-simulator-export-v1",
        "study_id": cell["study_id"],
        "registered_cell_id": cell["cell_id"],
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "requested_relation": relation,
        "prompt": cell["prompt"],
        "environment_seed": pair.seed,
        "sampling_seed": pair.seed,
        "reset_id": cell["reset_identity"],
        "predicate_id": cell["success_predicate_id"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    for key, expected in expected_values.items():
        if export.get(key) != expected:
            raise ContractError(f"{relation} simulator export mismatch for {key}")
    steps = _validate_raw_steps(export)
    actions_executed = len(steps) - 1
    if export.get("actions_executed") != actions_executed:
        raise ContractError("actions_executed must equal retained state samples minus one")
    action_artifact = _file_record(export.get("executed_action_trace_path"), "executed action trace")
    actions = np.load(action_artifact["path"], allow_pickle=False)
    if actions.shape != (actions_executed, 8):
        raise ContractError("executed action trace must have shape [actions_executed,8]")
    future_requests = _future_evidence(export, actions_executed, pair.seed)
    video_artifact = _file_record(export.get("viewport_video_path"), "viewport video")
    source_artifacts = {
        "simulator_export": _file_record(str(export_path), "simulator export"),
    }
    for name, path_value in export.get("source_artifacts", {}).items():
        source_artifacts[name] = _file_record(path_value, f"source artifact {name}")
    requested_success = export.get("requested_success")
    final_detached_release = export.get("final_detached_release")
    if type(requested_success) is not bool or type(final_detached_release) is not bool:
        raise ContractError("success and detached release must be separate raw booleans")
    first_contact_step = export.get("first_contact_step")
    contact_reason = export.get("first_contact_unavailable_reason")
    if all("contact_detected" in step for step in steps):
        derived_contact = next((i for i, step in enumerate(steps) if step["contact_detected"]), None)
        if first_contact_step != derived_contact or contact_reason is not None:
            raise ContractError("contact summary must exactly match the retained contact stream")
    elif first_contact_step is not None or not isinstance(contact_reason, str) or not contact_reason.strip():
        raise ContractError("missing contact stream requires null step and a non-empty reason")

    record: dict[str, Any] = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": cell["study_id"],
        "registered_cell_id": cell["cell_id"],
        "attempt_id": export.get("attempt_id"),
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
        "artifacts": {
            "viewport_video": video_artifact,
            "executed_action_trace": action_artifact,
            "raw_result_jsonl": {
                "path": str(raw_jsonl_path.resolve()),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "source_artifacts": source_artifacts,
        "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "future_requests": future_requests,
        "missing_future_policy": "infrastructure_invalid_never_zero",
        "requested_success": requested_success,
        "failure_stage": export.get("frozen_failure_stage"),
        "frozen_failure_stage": export.get("frozen_failure_stage"),
        "failure_taxonomy": "transport_failed",
        "steps": steps,
        "actions_executed": actions_executed,
        "action_cap": export.get("action_cap"),
        "right_censored": export.get("right_censored"),
        "wall_time_s": export.get("wall_time_s"),
        "operational_wall_time_valid": export.get("operational_wall_time_valid"),
        "final_detached_release": final_detached_release,
        "first_contact_step": first_contact_step,
        "first_contact_unavailable_reason": contact_reason,
        "event_timeline": _event_timeline({"steps": steps}, relation),
        "queue_sha256": pair.queue_sha256,
    }
    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    if export.get("initial_state_sha256") != record["initial_state_sha256"]:
        raise ContractError(
            f"{relation} initial-state fingerprint does not match the retained state"
        )
    measurements = derive_measurements(record)
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    return validate_behavioral_record(record)


def compile_pair(
    *, study_root: Path, model_id: str, seed: int, runtime_manifest: Path,
    release_manifest: Path, left_export: Path, right_export: Path, output_jsonl: Path,
) -> dict[str, Any]:
    pair = load_authorized_pair(study_root, model_id, seed)
    runtime = verify_runtime_identity(study_root, model_id, runtime_manifest)
    verify_release_gate(
        release_manifest, pair=pair,
        runtime_identity_sha256=runtime["runtime_identity_sha256"],
    )
    records = [
        build_behavioral_record(
            pair=pair, relation="left", export_path=left_export,
            runtime=runtime, raw_jsonl_path=output_jsonl,
        ),
        build_behavioral_record(
            pair=pair, relation="right", export_path=right_export,
            runtime=runtime, raw_jsonl_path=output_jsonl,
        ),
    ]
    initial_hashes = {record.get("initial_state_sha256") for record in records}
    initial_hash = next(iter(initial_hashes), None)
    if (
        len(initial_hashes) != 1 or not isinstance(initial_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", initial_hash)
    ):
        raise ContractError("LEFT/RIGHT do not share one non-empty initial-state fingerprint")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(output_jsonl, records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--left-export", type=Path, required=True)
    parser.add_argument("--right-export", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    manifest = compile_pair(**vars(args))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
