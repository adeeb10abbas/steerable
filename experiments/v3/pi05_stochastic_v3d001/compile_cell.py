#!/usr/bin/env python3
"""Compile one retained V3-D001 π0.5 simulator export, fail closed."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.v3.pi05_stochastic_v3d001.contract import (
    ACTION_CAP, ACTION_CHUNK_STEPS, ACTION_DIM, ACTION_SPACE, ARENA,
    FROZEN_CHECKPOINT, MODEL_ID, PHASE, QUEUE_SHA256, REGISTRATION_ID,
    RELEASE_MANIFEST_SHA256, STUDY_ID, SUCCESS_PREDICATE_ID, AuthorizedCell,
    ContractError, Release, load_release, sha256_file, validate_runtime,
)
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION, MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID, derive_failure_taxonomy, derive_frozen_failure_stage,
    derive_initial_state_sha256, derive_measurements, validate_behavioral_record,
    write_jsonl,
)


EXPORT_SCHEMA = "vla-wam-shared-v3d001-pi05-simulator-export-v1"
CONTACT_UNAVAILABLE = (
    "The pinned RoboLab runtime exposes grasp but no verified contact stream; "
    "grasp is not substituted for contact."
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _file(value: Any, label: str) -> dict[str, Any]:
    path = Path(str(value)).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"missing or empty {label}: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _bound_file(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a hash-bound file record")
    record = _file(value.get("path"), label)
    if value.get("sha256") != record["sha256"] or value.get("bytes") != record["bytes"]:
        raise ContractError(f"{label} hash/size binding changed")
    return {**value, **record}


def _steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ContractError("V3-D001 requires initial plus post-action states")
    output: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, dict) or row.get("action_step") != index:
            raise ContractError("V3-D001 action_step must be contiguous from zero")
        for key in ("object_xyz", "reference_xyz"):
            xyz = row.get(key)
            if not isinstance(xyz, list) or len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ContractError(f"V3-D001 requires finite {key}")
        if type(row.get("grippers_open")) is not bool or type(row.get("object_grabbed")) is not bool:
            raise ContractError("V3-D001 requires raw gripper and object_grabbed booleans")
        if "contact_detected" in row:
            raise ContractError("contact is unavailable and cannot be inferred from grasp")
        output.append(dict(row))
    return output


def _cone(step: Mapping[str, Any], relation: str) -> bool:
    obj = np.asarray(step["object_xyz"], dtype=float)
    ref = np.asarray(step["reference_xyz"], dtype=float)
    delta = obj - ref
    radius = math.hypot(float(delta[0]), float(delta[1]))
    margin = float(delta[1]) if relation == "left" else -float(delta[1])
    return radius > 1e-8 and margin / radius >= math.cos(math.radians(45.0))


def _first_sustained(mask: list[bool], width: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == width:
            return index - width + 1
    return None


def _timeline(steps: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    initial_z = float(steps[0]["object_xyz"][2])
    pickup = _first_sustained([float(row["object_xyz"][2]) - initial_z >= .03 for row in steps])
    requested = next((i for i, row in enumerate(steps) if _cone(row, relation)), None)
    opposite = next((i for i, row in enumerate(steps) if _cone(row, "right" if relation == "left" else "left")), None)
    events = [("episode_start", 0)]
    for name, index in (("verified_pickup", pickup), ("requested_region_entry", requested), ("opposite_region_entry", opposite)):
        if index is not None:
            events.append((name, index))
    events.append(("episode_end", len(steps) - 1))
    events.sort(key=lambda item: (item[1], item[0] == "episode_end"))
    return [{"event": name, "action_step": index} for name, index in events]


def _diagnostics(record: Mapping[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    measurements = record["measurements"]
    lateral = [float(row["object_xyz"][1]) - float(row["reference_xyz"][1]) for row in steps]
    return {
        "schema_version": "vla-wam-shared-v3d001-pi05-episode-diagnostics-v1",
        "success": record["requested_success"],
        "failure_category": record["failure_taxonomy"],
        "signed_final_lateral_offset_m": measurements["signed_final_lateral_offset_m"],
        "requested_side_depth_m": measurements["final_requested_signed_margin_m"],
        "cone_entry_step": measurements["first_requested_entry_step"],
        "cone_entry_sustained": measurements["first_sustained_requested_entry_step"] is not None,
        "episode_length_steps": record["actions_executed"],
        "time_to_first_contact_steps": None,
        "first_contact_status": "instrumentation_unavailable",
        "first_contact_unavailable_reason": CONTACT_UNAVAILABLE,
        "grasp_step": next((i for i, row in enumerate(steps) if row["object_grabbed"]), None),
        "grasp_source": "retained_object_grabbed_boolean_stream",
        "cumulative_lateral_path_m": sum(abs(b - a) for a, b in zip(lateral, lateral[1:])),
        "peak_lateral_excursion_m": max(abs(value - lateral[0]) for value in lateral),
        "endpoint_shift_m": None,
        "action_distinct": None,
        "pair_fields_status": "derived_only_after_both_cells_in_matched_stochastic_block",
    }


def build_record(*, export: Mapping[str, Any], output_jsonl: Path, cell: AuthorizedCell,
                 release: Release, runtime: Mapping[str, Any], runtime_path: Path) -> dict[str, Any]:
    expected = {
        "schema_version": EXPORT_SCHEMA, "study_id": STUDY_ID,
        "registration_id": REGISTRATION_ID, "phase": PHASE, "arena": ARENA,
        "model_id": MODEL_ID, "registered_cell_id": cell.cell_id,
        "matched_stochastic_block_id": cell.block_id,
        "nested_condition_id": cell.row["nested_condition_id"],
        "environment_seed": cell.environment_seed,
        "shared_policy_sampling_seed_index": cell.sampling_index,
        "policy_sampling_seed_base": cell.sampling_seed_base,
        "per_request_sampling_seed_rule": cell.row["per_request_sampling_seed_rule"],
        "requested_relation": cell.relation, "prompt": cell.row["prompt"],
        "prompt_mode": "static_episode_prompt", "success_predicate_id": SUCCESS_PREDICATE_ID,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256, "queue_sha256": QUEUE_SHA256,
        "cell_sha256": cell.row["cell_sha256"],
        "release_fingerprint_sha256": release.fingerprint(cell),
        "source_phase_a_runtime_identity_sha256": cell.row["source_phase_a_runtime_identity_sha256"],
        "source_phase_a_initial_state_sha256": cell.row["source_phase_a_initial_state_sha256"],
        "runtime_identity_sha256": sha256_file(runtime_path),
        "action_space": ACTION_SPACE, "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP, "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
    }
    for key, wanted in expected.items():
        if export.get(key) != wanted:
            raise ContractError(f"V3-D001 simulator export mismatch for {key}")
    for key in ("lane_pod_uid", "lane_gpu_uuid"):
        if not isinstance(export.get(key), str) or not export[key].strip():
            raise ContractError(f"V3-D001 requires explicit {key}")
    steps = _steps(export.get("steps"))
    actions_executed = len(steps) - 1
    success, censored = export.get("requested_success"), export.get("right_censored")
    if type(success) is not bool or type(censored) is not bool:
        raise ContractError("V3-D001 success/right_censored must be raw booleans")
    if export.get("actions_executed") != actions_executed or not 1 <= actions_executed <= ACTION_CAP:
        raise ContractError("V3-D001 action count disagrees with retained states")
    if (success and censored) or (not success and (not censored or actions_executed != ACTION_CAP)):
        raise ContractError("valid failures run to 450; successes cannot be censored")
    actions_record = _bound_file(export.get("executed_action_trace"), "executed actions")
    actions = np.load(actions_record["path"], allow_pickle=False)
    if actions.shape != (actions_executed, ACTION_DIM) or not np.isfinite(actions).all():
        raise ContractError("executed actions must be finite [actions_executed,8]")
    chunks_record = _bound_file(export.get("returned_action_chunks"), "returned chunks")
    chunks = np.load(chunks_record["path"], allow_pickle=False)
    request_count = math.ceil(actions_executed / ACTION_CHUNK_STEPS)
    if chunks.shape != (request_count, ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(chunks).all():
        raise ContractError("returned chunks must be complete finite [requests,15,8]")
    if export.get("request_sampling_seeds") != [cell.sampling_seed_base + i for i in range(request_count)]:
        raise ContractError("V3-D001 per-request seed sequence changed")
    video = _file(export.get("viewport_video_path"), "viewport video")
    capture = _file(export.get("state_capture_path"), "state capture")
    trace_metadata = _file(export.get("action_trace_metadata_path"), "action trace metadata")
    if export.get("first_contact_step") is not None or export.get("first_contact_unavailable_reason") != CONTACT_UNAVAILABLE:
        raise ContractError("V3-D001 contact must remain explicitly unavailable")
    record: dict[str, Any] = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION, "record_type": "behavioral_episode",
        "behavioral_result_valid": True, "study_id": STUDY_ID,
        "registered_cell_id": cell.cell_id,
        "attempt_id": f"{cell.cell_id}:attempt01", "model_id": MODEL_ID,
        "pair_id": cell.block_id, "arena": ARENA,
        "environment_seed": cell.environment_seed, "policy_seed": cell.sampling_seed_base,
        "requested_relation": cell.relation, "prompt": cell.row["prompt"],
        "prompt_family": "direct_command", "predicate_id": SUCCESS_PREDICATE_ID,
        "reset_id": cell.row["source_phase_a_initial_state_sha256"],
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": dict(FROZEN_CHECKPOINT),
        "runtime_identity": {"id": runtime["runtime_id"], "sha256": sha256_file(runtime_path)},
        "artifacts": {"viewport_video": video, "executed_action_trace": actions_record,
                      "raw_result_jsonl": {"path": str(output_jsonl.resolve()), "integrity_scope": "batch_manifest_after_close"}},
        "source_artifacts": {"state_capture": capture, "action_trace_metadata": trace_metadata,
                             "returned_action_chunks": chunks_record},
        "future_interface": "actions_only", "future_evidence_status": "not_exposed_by_action_only_interface",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
        "requested_success": success, "failure_stage": "pending", "frozen_failure_stage": "pending",
        "failure_taxonomy": "transport_failed", "steps": steps,
        "actions_executed": actions_executed, "action_cap": ACTION_CAP,
        "right_censored": censored, "final_detached_release": export.get("final_detached_release"),
        "first_contact_step": None, "first_contact_unavailable_reason": CONTACT_UNAVAILABLE,
        "wall_time_s": export.get("wall_time_s"),
        "operational_wall_time_valid": export.get("operational_wall_time_valid"),
        "event_timeline": _timeline(steps, cell.relation),
        "initial_state_sha256": derive_initial_state_sha256({"measurement_frame": MEASUREMENT_FRAME_ID, "steps": steps}),
        "registration_id": REGISTRATION_ID, "phase": PHASE,
        "shared_policy_sampling_seed_index": cell.sampling_index,
        "nested_condition_id": cell.row["nested_condition_id"],
        "analysis_unit": cell.row["analysis_unit"],
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "queue_sha256": QUEUE_SHA256, "cell_sha256": cell.row["cell_sha256"],
        "release_fingerprint_sha256": release.fingerprint(cell),
        "lane_pod_uid": export["lane_pod_uid"], "lane_gpu_uuid": export["lane_gpu_uuid"],
        "action_space": ACTION_SPACE, "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "request_sampling_seeds": export["request_sampling_seeds"],
    }
    if record["initial_state_sha256"] != cell.row["source_phase_a_initial_state_sha256"]:
        raise ContractError("V3-D001 first retained state differs from its Phase-A reset")
    measurements = derive_measurements(record)
    stage = derive_frozen_failure_stage(record, measurements)
    record["failure_stage"] = stage
    record["frozen_failure_stage"] = stage
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    normalized = validate_behavioral_record(record)
    normalized["v3d001_episode_diagnostics"] = _diagnostics(normalized, steps)
    return normalized


def compile_cell(*, repo_root: Path, release_manifest: Path, runtime_identity: Path,
                 phase_a_release_gate: Path, cell_id: str, export: Path,
                 output_jsonl: Path) -> dict[str, Any]:
    release = load_release(repo_root, release_manifest)
    runtime = validate_runtime(repo_root, runtime_identity, phase_a_release_gate)
    record = build_record(export=_object(export), output_jsonl=output_jsonl,
                          cell=release.cell(cell_id), release=release,
                          runtime=runtime, runtime_path=runtime_identity.resolve())
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(output_jsonl, [record])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--phase-a-release-gate", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_cell(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
