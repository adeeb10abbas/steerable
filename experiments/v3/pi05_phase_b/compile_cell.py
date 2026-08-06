#!/usr/bin/env python3
"""Compile one V3-B002 simulator export into hash-manifested raw JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.v3.pi05_phase_b.contract import (
    ACTION_CAP, ACTION_CHUNK_STEPS, ACTION_DIM, ACTION_SPACE, AMENDMENT_ID,
    MODEL_ID, OPENPI_COMMIT, STUDY_ID, AuthorizedCell, ContractError,
    ReleaseBundle, canonical_json_bytes, load_release_bundle, sha256_bytes,
    sha256_file,
)
from experiments.v3.pi05_phase_b.runtime import validate_runtime_identity
from experiments.v3.pi05_phase_b.diagnostics import attach_episode_diagnostics
from tools.vla_wam_v3_episode_schema import (
    BEHAVIORAL_SCHEMA_VERSION, MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID, derive_failure_taxonomy, derive_frozen_failure_stage,
    derive_initial_state_sha256, derive_measurements, validate_behavioral_record,
    write_jsonl,
)


EXPORT_SCHEMA = "vla-wam-shared-v3b-pi05-simulator-export-v1"
CONTACT_UNAVAILABLE = (
    "The pinned RoboLab integration exposes object_grabbed and detached release "
    "but no verified physical contact stream; grasp is not substituted for contact."
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected JSON object: {path}")
    return value


def _file_record(value: Any, label: str) -> dict[str, Any]:
    path = Path(str(value)).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ContractError(f"missing or empty {label}: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _bound_file_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a hash-bound file record")
    record = _file_record(value.get("path"), label)
    if value.get("sha256") != record["sha256"] or value.get("bytes") != record["bytes"]:
        raise ContractError(f"{label} hash/size binding changed")
    return {**record, **{key: child for key, child in value.items() if key not in record}}


def _cone(step: Mapping[str, Any], relation: str) -> bool:
    obj = np.asarray(step["object_xyz"], dtype=float)
    ref = np.asarray(step["reference_xyz"], dtype=float)
    delta = obj-ref
    radius = math.hypot(float(delta[0]), float(delta[1]))
    margin = float(delta[1]) if relation == "left" else -float(delta[1])
    return radius > 1e-8 and margin/radius >= math.cos(math.radians(45))


def _first_sustained(mask: list[bool], width: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run+1 if value else 0
        if run == width:
            return index-width+1
    return None


def _normalize_steps(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ContractError("steps require initial plus post-action states")
    output = []
    for index, raw in enumerate(value):
        if not isinstance(raw, dict) or raw.get("action_step") != index:
            raise ContractError("action_step must be contiguous from zero")
        for name in ("object_xyz", "reference_xyz"):
            xyz = raw.get(name)
            if not isinstance(xyz, list) or len(xyz) != 3 or not np.isfinite(xyz).all():
                raise ContractError(f"each state requires finite {name}")
        if type(raw.get("grippers_open")) is not bool:
            raise ContractError("each state requires raw grippers_open")
        if type(raw.get("object_grabbed")) is not bool:
            raise ContractError("each state requires the true RoboLab object_grabbed conditional")
        if "contact_detected" in raw:
            raise ContractError("contact is unavailable; do not substitute a grasp stream")
        output.append({
            "action_step": index, "object_xyz": raw["object_xyz"],
            "reference_xyz": raw["reference_xyz"],
            "grippers_open": raw["grippers_open"], "object_grabbed": raw["object_grabbed"],
        })
    return output


def _timeline(steps: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    initial_z = float(steps[0]["object_xyz"][2])
    pickup = _first_sustained([float(step["object_xyz"][2])-initial_z >= .03 for step in steps])
    requested = next((i for i, step in enumerate(steps) if _cone(step, relation)), None)
    opposite = next((i for i, step in enumerate(steps) if _cone(step, "right" if relation == "left" else "left")), None)
    events = [("episode_start", 0)]
    for name, index in (("verified_pickup", pickup), ("requested_region_entry", requested), ("opposite_region_entry", opposite)):
        if index is not None:
            events.append((name, index))
    events.append(("episode_end", len(steps)-1))
    events.sort(key=lambda pair: (pair[1], pair[0] == "episode_end"))
    return [{"event": name, "action_step": index} for name, index in events]


def _ablation_diagnostics(record: Mapping[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    relation = str(record["requested_relation"])
    lateral = [float(step["object_xyz"][1])-float(step["reference_xyz"][1]) for step in steps]
    requested = [_cone(step, relation) for step in steps]
    grasp = next((i for i, step in enumerate(steps) if step["object_grabbed"]), None)
    final = lateral[-1]
    return {
        "success": bool(record["requested_success"]),
        "failure_category": record["failure_taxonomy"],
        "signed_final_lateral_offset_m": final,
        "requested_side_depth_m": final if relation == "left" else -final,
        "cone_entry_step": next((i for i, value in enumerate(requested) if value), None),
        "cone_entry_sustained": _first_sustained(requested) is not None,
        "episode_length_steps": int(record["actions_executed"]),
        "time_to_first_contact_steps": None,
        "first_contact_status": "instrumentation_unavailable",
        "first_contact_unavailable_reason": CONTACT_UNAVAILABLE,
        "grasp_step": grasp,
        "grasp_source": "raw_robolab_object_grabbed_conditional",
        "cumulative_lateral_path_m": sum(abs(current-previous) for previous, current in zip(lateral, lateral[1:])),
        "peak_lateral_excursion_m": max(abs(value-lateral[0]) for value in lateral),
        "endpoint_shift_m": None,
        "action_distinct": None,
        "pair_fields_status": "pair_derived_in_separate_pair_jsonl",
    }


def build_record(*, export: Mapping[str, Any], output_jsonl: Path, cell: AuthorizedCell,
                 release: ReleaseBundle, runtime: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": EXPORT_SCHEMA, "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID, "registered_cell_id": cell.cell_id,
        "matched_block_id": cell.row["matched_block_id"], "model_id": MODEL_ID,
        "arm": cell.arm, "requested_relation": cell.relation,
        "prompt": cell.row["prompt"], "prompt_sha256": cell.row["prompt_sha256"],
        "environment_seed": cell.seed, "sampling_seed": cell.seed,
        "fixture_id": cell.row["fixture_id"], "fixture_sha256": cell.row["fixture_sha256"],
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "action_space": ACTION_SPACE, "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP, "instruction_controller": "static",
        "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
    }
    for key, wanted in expected.items():
        if export.get(key) != wanted:
            raise ContractError(f"simulator export mismatch for {key}")
    steps = _normalize_steps(export.get("steps"))
    actions_executed = len(steps)-1
    success, censored = export.get("requested_success"), export.get("right_censored")
    if type(success) is not bool or type(censored) is not bool:
        raise ContractError("success/right_censored must be raw booleans")
    if export.get("actions_executed") != actions_executed or not 1 <= actions_executed <= ACTION_CAP:
        raise ContractError("action count does not match retained states")
    if (success and censored) or (not success and (not censored or actions_executed != ACTION_CAP)):
        raise ContractError("valid failures run to 450; successful cells are not censored")
    action_record = _bound_file_record(export.get("executed_action_trace"), "executed actions")
    actions = np.load(action_record["path"], allow_pickle=False)
    if actions.shape != (actions_executed, ACTION_DIM) or not np.isfinite(actions).all():
        raise ContractError("executed actions must be finite [actions_executed,8]")
    chunk_record = _bound_file_record(export.get("returned_action_chunks"), "returned chunks")
    chunks = np.load(chunk_record["path"], allow_pickle=False)
    expected_requests = math.ceil(actions_executed/ACTION_CHUNK_STEPS)
    if chunks.shape != (expected_requests, ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(chunks).all():
        raise ContractError("returned chunks must be complete finite [requests,15,8]")
    seeds = export.get("request_sampling_seeds")
    if seeds != [cell.seed*1000+index for index in range(expected_requests)]:
        raise ContractError("request seed sequence changed")
    reset_path = Path(str(export.get("reset_attestation_path"))).resolve()
    reset = _object(reset_path)
    claimed = reset.get("reset_fingerprint_sha256")
    reset_body = {key: value for key, value in reset.items() if key != "reset_fingerprint_sha256"}
    if claimed != sha256_bytes(canonical_json_bytes(reset_body)) or export.get("reset_fingerprint_sha256") != claimed:
        raise ContractError("reset fingerprint binding changed")
    video = _file_record(export.get("viewport_video_path"), "viewport video")
    trace_meta = _file_record(export.get("action_trace_metadata_path"), "action trace metadata")
    contact_reason = export.get("first_contact_unavailable_reason")
    if export.get("first_contact_step") is not None or contact_reason != CONTACT_UNAVAILABLE:
        raise ContractError("contact must remain explicitly instrumentation-unavailable")
    record: dict[str, Any] = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION, "record_type": "behavioral_episode",
        "behavioral_result_valid": True, "study_id": STUDY_ID,
        "registered_cell_id": cell.cell_id, "attempt_id": export.get("attempt_id"),
        "model_id": MODEL_ID, "pair_id": cell.row["matched_block_id"],
        "arena": "droid_robolab", "environment_seed": cell.seed, "policy_seed": cell.seed,
        "requested_relation": cell.relation, "prompt": cell.row["prompt"],
        "prompt_family": "direct_command", "predicate_id": cell.row["success_predicate_id"],
        "reset_id": claimed, "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": {"id": "pi05_droid_jointpos_polaris", "revision": "v2a010-manifest-" + cell.row["runtime_identity_requirement"]["checkpoint_manifest_sha256"]},
        "runtime_identity": {"id": f"{MODEL_ID}:{runtime['runtime_identity_sha256'][:16]}", "sha256": runtime["runtime_identity_sha256"]},
        "artifacts": {"viewport_video": video, "executed_action_trace": action_record,
                      "raw_result_jsonl": {"path": str(output_jsonl.resolve()), "integrity_scope": "batch_manifest_after_close"}},
        "source_artifacts": {"reset_attestation": _file_record(str(reset_path), "reset attestation"),
                             "action_trace_metadata": trace_meta, "returned_action_chunks": chunk_record},
        "future_interface": "actions_only", "future_evidence_status": "not_exposed_by_action_only_interface",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
        "requested_success": success, "failure_stage": "pending", "frozen_failure_stage": "pending",
        "failure_taxonomy": "transport_failed", "steps": steps,
        "actions_executed": actions_executed, "action_cap": ACTION_CAP,
        "right_censored": censored, "wall_time_s": export.get("wall_time_s"),
        "operational_wall_time_valid": export.get("operational_wall_time_valid"),
        "final_detached_release": export.get("final_detached_release"),
        "first_contact_step": None, "first_contact_unavailable_reason": CONTACT_UNAVAILABLE,
        "event_timeline": _timeline(steps, cell.relation),
        "initial_state_sha256": derive_initial_state_sha256({"measurement_frame": MEASUREMENT_FRAME_ID, "steps": steps}),
        "amendment_id": AMENDMENT_ID, "phase_b_arm": cell.arm,
        "release_manifest_sha256": release.manifest_sha256,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "reset_fingerprint_sha256": claimed, "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
    }
    if record["initial_state_sha256"] != reset.get("initial_state_sha256"):
        raise ContractError("first retained state does not match reset attestation")
    measurements = derive_measurements(record)
    stage = derive_frozen_failure_stage(record, measurements)
    record["failure_stage"] = stage
    record["frozen_failure_stage"] = stage
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    normalized = validate_behavioral_record(record)
    return attach_episode_diagnostics(normalized)


def compile_cell(*, repo_root: Path, release_manifest: Path, release_manifest_sha256: str,
                 runtime_manifest: Path, cell_id: str, export: Path, output_jsonl: Path) -> dict[str, Any]:
    release = load_release_bundle(repo_root, release_manifest, expected_manifest_sha256=release_manifest_sha256)
    runtime = validate_runtime_identity(_object(runtime_manifest), repo_root=repo_root, release=release)
    record = build_record(export=_object(export), output_jsonl=output_jsonl,
                          cell=release.cell(cell_id), release=release, runtime=runtime)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    return write_jsonl(output_jsonl, [record])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_cell(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
