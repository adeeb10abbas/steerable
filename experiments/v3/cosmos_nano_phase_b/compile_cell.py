#!/usr/bin/env python3
"""Compile one released V3-B001 Nano cell into retained raw episode JSONL.

The compiler accepts behavioral evidence only when every policy request carries
the same cell-specific release and live-reset fingerprints checked before
inference.  Missing futures, short failures, prompt changes, and partial cells
are infrastructure-invalid and are never converted into behavioral failures.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.v3.cosmos_nano_phase_b.live_support import verify_live_runtime_identity
from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    ACTION_SPACE,
    AMENDMENT_ID,
    CHECKPOINT_REVISION,
    MODEL_ID,
    MODEL_REPOSITORY,
    PHASE,
    STUDY_ID,
    AuthorizedCell,
    ReleaseBundle,
    RuntimeContractError,
    canonical_json_bytes,
    load_json,
    load_release_bundle,
    sha256_bytes,
    sha256_file,
    validate_reset_attestation,
)
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION,
    MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID,
    derive_failure_taxonomy,
    derive_frozen_failure_stage,
    derive_initial_state_sha256,
    derive_measurements,
    validate_behavioral_record,
    write_jsonl,
)


EXPORT_SCHEMA = "vla-wam-shared-v3b-nano-simulator-export-v1"
SUSTAINED_STEPS = 3
PICKUP_LIFT_M = 0.03


def _fail(message: str) -> None:
    raise RuntimeContractError(message)


def _file_record(path_value: Any, label: str) -> dict[str, Any]:
    if not isinstance(path_value, str) or not path_value:
        _fail(f"{label} path is required")
    path = Path(path_value).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        _fail(f"missing or empty {label}: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _first_sustained(mask: list[bool]) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == SUSTAINED_STEPS:
            return index - SUSTAINED_STEPS + 1
    return None


def _in_cone(step: Mapping[str, Any], relation: str) -> bool:
    obj = np.asarray(step.get("object_xyz"), dtype=np.float64)
    ref = np.asarray(step.get("reference_xyz"), dtype=np.float64)
    if obj.shape != (3,) or ref.shape != (3,) or not np.isfinite(obj).all() or not np.isfinite(ref).all():
        _fail("each state sample requires finite robot-base object/reference XYZ")
    delta = obj - ref
    horizontal = math.hypot(float(delta[0]), float(delta[1]))
    requested = float(delta[1]) if relation == "left" else -float(delta[1])
    return horizontal > 1e-8 and requested / horizontal >= math.cos(math.radians(45.0))


def _validate_steps(export: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = export.get("steps")
    if not isinstance(raw, list) or len(raw) < 2:
        _fail("steps must retain the initial state plus every post-action state")
    contact_presence = {"contact_detected" in step for step in raw if isinstance(step, dict)}
    if len(contact_presence) != 1:
        _fail("contact_detected must be present on every state sample or none")
    normalized: list[dict[str, Any]] = []
    for index, step in enumerate(raw):
        if not isinstance(step, dict) or step.get("action_step") != index:
            _fail("state action_step values must be contiguous from zero")
        row: dict[str, Any] = {
            "action_step": index,
            "object_xyz": step.get("object_xyz"),
            "reference_xyz": step.get("reference_xyz"),
            "grippers_open": step.get("grippers_open"),
        }
        _in_cone(row, "left")
        if type(row["grippers_open"]) is not bool:
            _fail("grippers_open must be a retained raw boolean")
        if "contact_detected" in step:
            if type(step["contact_detected"]) is not bool:
                _fail("contact_detected must be boolean")
            row["contact_detected"] = step["contact_detected"]
        normalized.append(row)
    return normalized


def _event_timeline(steps: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    requested = [_in_cone(step, relation) for step in steps]
    opposite_relation = "right" if relation == "left" else "left"
    opposite = [_in_cone(step, opposite_relation) for step in steps]
    initial_z = float(steps[0]["object_xyz"][2])
    pickup = _first_sustained(
        [float(step["object_xyz"][2]) - initial_z >= PICKUP_LIFT_M for step in steps]
    )
    contact = None
    if all("contact_detected" in step for step in steps):
        contact = next((index for index, step in enumerate(steps) if step["contact_detected"]), None)
    events: list[tuple[str, int]] = [("episode_start", 0)]
    for name, step in (
        ("first_contact", contact),
        ("verified_pickup", pickup),
        ("requested_region_entry", next((i for i, value in enumerate(requested) if value), None)),
        ("opposite_region_entry", next((i for i, value in enumerate(opposite) if value), None)),
    ):
        if step is not None:
            events.append((name, step))
    events.append(("episode_end", len(steps) - 1))
    events.sort(key=lambda item: (item[1], item[0] == "episode_end"))
    return [{"event": name, "action_step": step} for name, step in events]


def _validate_policy_requests(
    export: Mapping[str, Any],
    *,
    cell: AuthorizedCell,
    release_fingerprint: str,
    reset_fingerprint: str,
    actions_executed: int,
) -> list[dict[str, Any]]:
    requests = export.get("policy_requests")
    expected_count = math.ceil(actions_executed / ACTION_CHUNK_STEPS)
    if not isinstance(requests, list) or len(requests) != expected_count:
        _fail("one retained action/future record is required per 32-action policy request")
    retained: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if not isinstance(request, dict):
            _fail("policy request records must be objects")
        expected = {
            "request_index": index,
            "action_step_start": index * ACTION_CHUNK_STEPS,
            "sampling_seed": cell.seed,
            "prompt": cell.row["prompt"],
            "release_fingerprint_sha256": release_fingerprint,
            "reset_fingerprint_sha256": reset_fingerprint,
        }
        for key, value in expected.items():
            if request.get(key) != value:
                _fail(f"policy request {index} fingerprint/static contract mismatch for {key}")
        action_record = _file_record(request.get("returned_action_path"), "returned action chunk")
        future_record = _file_record(request.get("decoded_future_path"), "decoded future")
        action = np.load(action_record["path"], allow_pickle=False)
        future = np.load(future_record["path"], allow_pickle=False)
        if action.shape != (ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(action).all():
            _fail("returned action chunk must be finite shape [32,8]")
        if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
            _fail("decoded future must retain all 33 RGB frames")
        retained.append(
            {
                "request_index": index,
                "action_step_start": index * ACTION_CHUNK_STEPS,
                "sampling_seed": cell.seed,
                "release_fingerprint_sha256": release_fingerprint,
                "reset_fingerprint_sha256": reset_fingerprint,
                "returned_action": action_record,
                "returned_action_shape": list(action.shape),
                "decoded_future": future_record,
                "decoded_future_shape": list(future.shape),
                "future_evidence_status": "exposed_and_retained",
            }
        )
    return retained


def build_behavioral_record(
    *,
    export_path: Path,
    output_jsonl: Path,
    cell: AuthorizedCell,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    reset_attestation: Mapping[str, Any],
    reset_fingerprint: str,
) -> dict[str, Any]:
    export = load_json(export_path, "Phase-B simulator export")
    release_fingerprint = release.release_fingerprint(cell)
    expected = {
        "schema_version": EXPORT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "phase": PHASE,
        "registered_cell_id": cell.cell_id,
        "matched_block_id": cell.row["matched_block_id"],
        "model_id": MODEL_ID,
        "arm": cell.arm,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "environment_seed": cell.seed,
        "sampling_seed": cell.seed,
        "fixture_id": cell.row["fixture_id"],
        "fixture_sha256": cell.row["fixture_sha256"],
        "release_fingerprint_sha256": release_fingerprint,
        "reset_fingerprint_sha256": reset_fingerprint,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "action_space": ACTION_SPACE,
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
    }
    for key, value in expected.items():
        if export.get(key) != value:
            _fail(f"simulator export mismatch for {key}")
    if export.get("initial_state_sha256") != reset_attestation["initial_state_sha256"]:
        _fail("simulator export does not use the request-authorized reset fingerprint")

    steps = _validate_steps(export)
    actions_executed = len(steps) - 1
    if export.get("actions_executed") != actions_executed or not 1 <= actions_executed <= ACTION_CAP:
        _fail("actions_executed must equal retained post-action states and be within 1..450")
    requested_success = export.get("requested_success")
    right_censored = export.get("right_censored")
    if type(requested_success) is not bool or type(right_censored) is not bool:
        _fail("requested_success and right_censored must be separate raw booleans")
    if requested_success:
        if right_censored:
            _fail("a successful cell cannot be right-censored")
    elif not right_censored or actions_executed != ACTION_CAP:
        _fail("every valid failure must run to and be censored at the 450-action cap")

    action_record = _file_record(export.get("executed_action_trace_path"), "executed action trace")
    actions = np.load(action_record["path"], allow_pickle=False)
    if actions.shape != (actions_executed, ACTION_DIM) or not np.isfinite(actions).all():
        _fail("executed joint-position trace must be finite shape [actions_executed,8]")
    future_requests = _validate_policy_requests(
        export,
        cell=cell,
        release_fingerprint=release_fingerprint,
        reset_fingerprint=reset_fingerprint,
        actions_executed=actions_executed,
    )
    video_record = _file_record(export.get("viewport_video_path"), "viewport video")
    export_reset_path = export.get("reset_attestation_path")
    export_reset = load_json(Path(export_reset_path), "export reset attestation") if isinstance(export_reset_path, str) else None
    if export_reset is None or sha256_bytes(canonical_json_bytes(export_reset)) != reset_fingerprint:
        _fail("export reset-attestation file does not match the request-authorized fingerprint")
    source_artifacts = {
        "simulator_export": _file_record(str(export_path), "simulator export"),
        "reset_attestation": _file_record(export_reset_path, "reset attestation"),
    }
    additional_sources = export.get("source_artifacts", {})
    if not isinstance(additional_sources, dict):
        _fail("source_artifacts must be an object")
    for name, path_value in additional_sources.items():
        if not isinstance(name, str) or not name:
            _fail("source_artifact names must be non-empty strings")
        source_artifacts[name] = _file_record(path_value, f"source artifact {name}")

    final_detached_release = export.get("final_detached_release")
    if type(final_detached_release) is not bool:
        _fail("final_detached_release must be a raw scorer boolean")
    first_contact_step = export.get("first_contact_step")
    contact_reason = export.get("first_contact_unavailable_reason")
    if all("contact_detected" in step for step in steps):
        derived_contact = next((index for index, step in enumerate(steps) if step["contact_detected"]), None)
        if first_contact_step != derived_contact or contact_reason is not None:
            _fail("contact summary must exactly match the retained contact stream")
    elif first_contact_step is not None or not isinstance(contact_reason, str) or not contact_reason.strip():
        _fail("missing contact stream requires null contact step and a non-empty reason")

    record: dict[str, Any] = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": STUDY_ID,
        "registered_cell_id": cell.cell_id,
        "attempt_id": export.get("attempt_id"),
        "model_id": MODEL_ID,
        "pair_id": cell.row["matched_block_id"],
        "arena": "droid_robolab",
        "environment_seed": cell.seed,
        "policy_seed": cell.seed,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_family": "direct_command",
        "predicate_id": cell.row["success_predicate_id"],
        "reset_id": reset_fingerprint,
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": {"id": MODEL_REPOSITORY, "revision": CHECKPOINT_REVISION},
        "runtime_identity": {
            "id": f"{MODEL_ID}:{runtime['runtime_identity_sha256'][:16]}",
            "sha256": runtime["runtime_identity_sha256"],
        },
        "artifacts": {
            "viewport_video": video_record,
            "executed_action_trace": action_record,
            "raw_result_jsonl": {
                "path": str(Path(output_jsonl).resolve()),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "source_artifacts": source_artifacts,
        "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "future_requests": future_requests,
        "missing_future_policy": "infrastructure_invalid_never_zero",
        "requested_success": requested_success,
        "failure_stage": "pending_frozen_stage_derivation",
        "frozen_failure_stage": "pending_frozen_stage_derivation",
        "failure_taxonomy": "transport_failed",
        "steps": steps,
        "actions_executed": actions_executed,
        "action_cap": ACTION_CAP,
        "right_censored": right_censored,
        "wall_time_s": export.get("wall_time_s"),
        "operational_wall_time_valid": export.get("operational_wall_time_valid"),
        "final_detached_release": final_detached_release,
        "first_contact_step": first_contact_step,
        "first_contact_unavailable_reason": contact_reason,
        "event_timeline": _event_timeline(steps, cell.relation),
        "initial_state_sha256": derive_initial_state_sha256(
            {"measurement_frame": MEASUREMENT_FRAME_ID, "steps": steps}
        ),
        "amendment_id": AMENDMENT_ID,
        "phase_b_arm": cell.arm,
        "release_manifest_sha256": release.manifest_sha256,
        "release_fingerprint_sha256": release_fingerprint,
        "reset_fingerprint_sha256": reset_fingerprint,
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
    }
    if record["initial_state_sha256"] != reset_attestation["initial_state_sha256"]:
        _fail("retained first state does not match the request-authorized reset")
    # The common schema derives continuous measurements from raw states.  Set
    # the two classifications only after those measurements exist, then run the
    # complete validator once more before serialization.
    measurements = derive_measurements(record)
    frozen_stage = derive_frozen_failure_stage(record, measurements)
    record["failure_stage"] = frozen_stage
    record["frozen_failure_stage"] = frozen_stage
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    return validate_behavioral_record(record)


def compile_cell(
    *,
    study_root: Path,
    release_manifest: Path,
    release_manifest_sha256: str,
    runtime_manifest: Path,
    reset_attestation: Path,
    cell_id: str,
    export: Path,
    output_jsonl: Path,
) -> dict[str, Any]:
    release = load_release_bundle(
        release_manifest,
        expected_manifest_sha256=release_manifest_sha256,
    )
    cell = release.cell(cell_id)
    runtime = verify_live_runtime_identity(
        runtime_manifest,
        study_root=study_root,
        release=release,
    )
    reset, reset_fingerprint = validate_reset_attestation(
        reset_attestation,
        cell=cell,
        release=release,
        runtime=runtime,
    )
    record = build_behavioral_record(
        export_path=export,
        output_jsonl=output_jsonl,
        cell=cell,
        release=release,
        runtime=runtime,
        reset_attestation=reset,
        reset_fingerprint=reset_fingerprint,
    )
    Path(output_jsonl).parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(output_jsonl, [record])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--reset-attestation", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    result = compile_cell(**vars(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
