"""Compile canonical V3-E004 DROID exports and matched-pair diagnostics.

Model-specific bridges may expose different action horizons and optional
decoded futures, but they emit one canonical simulator export.  This compiler
reapplies the frozen DROID predicate/taxonomy from raw states, verifies the
per-cell live scene gate, and writes immutable per-episode JSONL.  Pair-only
measurements are emitted separately after both directions exist; unavailable
pair fields are never encoded as zero in an episode row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .live_snapshot_adapter import BOUND_GATE_SCHEMA
from .runtime_contract import E004Cell, E004RuntimeBundle, RuntimeContractError, load_runtime_bundle, sha256_file


EXPORT_SCHEMA = "vla-wam-shared-v3e004-droid-simulator-export-v1"
EPISODE_SCHEMA = "vla-wam-shared-v3e004-droid-behavioral-episode-v1"
PAIR_SCHEMA = "vla-wam-shared-v3e004-droid-matched-pair-v1"
CONTACT_UNAVAILABLE = (
    "The pinned RoboLab integration exposes a verified object-grabbed conditional "
    "and detached release but no verified physical contact stream; grasp is not "
    "substituted for contact."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def _finite_json(path: Path) -> Any:
    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeContractError(f"not finite UTF-8 JSON: {path}: {exc}") from exc


def _file_record(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        path = Path(str(value.get("path"))).resolve()
    else:
        path = Path(str(value)).resolve()
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty {label}: {path}")
    output = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if isinstance(value, Mapping):
        _require(value.get("bytes") == output["bytes"] and value.get("sha256") == output["sha256"], f"{label} binding changed")
        output.update({key: child for key, child in value.items() if key not in output})
    return output


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _vec3(value: Any, label: str) -> list[float]:
    _require(isinstance(value, list) and len(value) == 3, f"{label} must be a 3-vector")
    result = [float(item) for item in value]
    _require(all(math.isfinite(item) for item in result), f"{label} must be finite")
    return result


def _cone(step: Mapping[str, Any], relation: str) -> bool:
    obj, ref = _vec3(step.get("object_xyz"), "object_xyz"), _vec3(step.get("reference_xyz"), "reference_xyz")
    forward = obj[0] - ref[0]
    lateral = obj[1] - ref[1]
    radius = math.hypot(forward, lateral)
    margin = lateral if relation == "left" else -lateral
    return radius > 1e-8 and margin / radius >= math.cos(math.radians(45.0))


def _first_sustained(mask: list[bool], width: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == width:
            return index - width + 1
    return None


def _normalize_steps(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) >= 2, "state capture requires initial plus post-action states")
    output: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        _require(isinstance(raw, dict) and raw.get("action_step") == index, "state action_step must be contiguous from zero")
        row = {
            "action_step": index,
            "object_xyz": _vec3(raw.get("object_xyz"), f"steps[{index}].object_xyz"),
            "reference_xyz": _vec3(raw.get("reference_xyz"), f"steps[{index}].reference_xyz"),
        }
        for key in ("grippers_open", "object_grabbed"):
            _require(type(raw.get(key)) is bool, f"steps[{index}].{key} must be a raw boolean")
            row[key] = raw[key]
        if "contact_detected" in raw:
            _require(type(raw["contact_detected"]) is bool, f"steps[{index}].contact_detected must be boolean")
            row["contact_detected"] = raw["contact_detected"]
        output.append(row)
    has_contact = ["contact_detected" in row for row in output]
    _require(all(has_contact) or not any(has_contact), "contact stream must be present for every step or none")
    return output


def _failure_category(*, success: bool, steps: list[dict[str, Any]], relation: str, detached_release: bool) -> str:
    if success:
        return "correct"
    if not any(bool(step["object_grabbed"]) for step in steps):
        return "pick_failed"
    opposite = "right" if relation == "left" else "left"
    if _first_sustained([_cone(step, opposite) for step in steps]) is not None and all(_cone(step, opposite) for step in steps[-3:]):
        return "wrong_side"
    if _first_sustained([_cone(step, relation) for step in steps]) is not None and all(_cone(step, relation) for step in steps[-3:]) and not detached_release:
        return "release_failed"
    return "transport_failed"


def _validate_live_gate(record: Any, *, bundle: E004RuntimeBundle, cell: E004Cell) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _file_record(record, "bound live scene gate")
    value = _finite_json(Path(artifact["path"]))
    _require(isinstance(value, dict) and value.get("schema_version") == BOUND_GATE_SCHEMA, "live scene gate schema changed")
    expected = {
        "status": "passed_and_released_for_exact_cell_request_zero",
        "passed": True,
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
    }
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"live scene gate differs for {key}")
    compiled = value.get("compiled_gate")
    _require(isinstance(compiled, dict) and compiled.get("passed") is True, "compiled live scene gate did not pass")
    scene = compiled.get("scene")
    _require(isinstance(scene, dict), "live scene gate has no scene record")
    _require(math.isclose(float(scene.get("symmetry_level_s")), cell.symmetry_level_s, abs_tol=1e-12), "live gate symmetry level differs from queue")
    _require(all(value is False for value in scene.get("occlusion_check", {}).values()), "live gate retained an occluded camera")
    _require(all(value is True for value in scene.get("target_visible", {}).values()), "live gate retained an invisible target camera")
    return artifact, scene


def build_episode_record(*, export: Mapping[str, Any], bundle: E004RuntimeBundle, cell: E004Cell, output_path: Path) -> dict[str, Any]:
    expected = {
        "schema_version": EXPORT_SCHEMA,
        "study_id": cell.row["study_id"],
        "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "model_id": cell.model_id,
        "arena": cell.row["arena"],
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "matched_pair_id": cell.matched_pair_id,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "symmetry_level_s": cell.symmetry_level_s,
        "success_predicate_id": cell.row["success_predicate_id"],
        "runtime_identity_requirement": cell.row["runtime_identity_requirement"],
        "instruction_controller": "static_episode_prompt",
    }
    for key, wanted in expected.items():
        _require(export.get(key) == wanted, f"simulator export differs for {key}")
    live_gate_artifact, scene = _validate_live_gate(export.get("live_scene_gate"), bundle=bundle, cell=cell)
    steps = _normalize_steps(export.get("steps"))
    actions_executed = len(steps) - 1
    action_cap = int(cell.row["runtime_identity_requirement"]["action_cap"])
    _require(export.get("actions_executed") == actions_executed, "state/action count mismatch")
    _require(1 <= actions_executed <= action_cap, "executed action count is outside the registered cap")
    success, right_censored = export.get("requested_success"), export.get("right_censored")
    detached = export.get("final_detached_release")
    _require(type(success) is bool and type(right_censored) is bool and type(detached) is bool, "scorer booleans are invalid")
    requested = [_cone(step, cell.relation) for step in steps]
    final_sustained = len(requested) >= 3 and all(requested[-3:])
    _require(success == (final_sustained and detached), "requested_success differs from the frozen sustained-cone plus detached-release predicate")
    _require(not success or not right_censored, "successful episode cannot be right-censored")
    _require(not right_censored or (not success and actions_executed == action_cap), "right-censored failure must reach action cap")
    actions_artifact = _file_record(export.get("executed_action_trace"), "executed action trace")
    actions = np.load(actions_artifact["path"], allow_pickle=False)
    _require(actions.ndim == 2 and actions.shape[0] == actions_executed and actions.shape[1] > 0 and np.isfinite(actions).all(), "executed action trace is not finite [actions,dim]")
    video_artifact = _file_record(export.get("viewport_video"), "viewport video")
    _require(Path(video_artifact["path"]).suffix.lower() == ".mp4", "viewport evidence must be MP4")
    runtime_artifact = _file_record(export.get("runtime_identity"), "runtime identity")
    runtime = _finite_json(Path(runtime_artifact["path"]))
    _require(isinstance(runtime, dict) and runtime.get("model_id") == cell.model_id, "runtime identity model differs")
    _require(runtime.get("runtime_identity_requirement") == cell.row["runtime_identity_requirement"], "runtime identity does not bind registered checkpoint settings")
    lateral = [step["object_xyz"][1] - step["reference_xyz"][1] for step in steps]
    initial_state = {
        "object_xyz": steps[0]["object_xyz"],
        "reference_xyz": steps[0]["reference_xyz"],
        "grippers_open": steps[0]["grippers_open"],
        "realised_object_poses": scene["realised_object_poses"],
        "arm_reset_pose": scene["arm_reset_pose"],
    }
    first_contact: int | None
    contact_reason: str | None
    if "contact_detected" in steps[0]:
        first_contact = next((index for index, step in enumerate(steps) if step["contact_detected"]), None)
        contact_reason = None
    else:
        first_contact = None
        contact_reason = CONTACT_UNAVAILABLE
    grasp = next((index for index, step in enumerate(steps) if step["object_grabbed"]), None)
    final = float(lateral[-1])
    failure = _failure_category(success=success, steps=steps, relation=cell.relation, detached_release=detached)
    output_path = Path(output_path).resolve()
    record = {
        "schema_version": EPISODE_SCHEMA,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": cell.row["study_id"],
        "amendment_id": cell.row["amendment_id"],
        "registered_cell_id": cell.cell_id,
        "registered_cell_sha256": cell.row_sha256,
        "matched_pair_id": cell.matched_pair_id,
        "model_id": cell.model_id,
        "arena": cell.row["arena"],
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_sha256": cell.row["prompt_sha256"],
        "success": success,
        "requested_success": success,
        "failure_category": failure,
        "failure_taxonomy": failure,
        "signed_final_lateral_offset": final,
        "signed_final_lateral_offset_m": final,
        "requested_side_depth": final if cell.relation == "left" else -final,
        "requested_side_depth_m": final if cell.relation == "left" else -final,
        "cone_entry_step": next((index for index, value in enumerate(requested) if value), None),
        "cone_entry_sustained": _first_sustained(requested) is not None,
        "endpoint_shift": None,
        "endpoint_shift_m": None,
        "action_distinct": None,
        "pair_fields_status": "derived_only_after_both_hash-bound_directions_exist",
        "episode_length": actions_executed,
        "episode_length_steps": actions_executed,
        "time_to_first_contact": first_contact,
        "time_to_first_contact_steps": first_contact,
        "first_contact_unavailable_reason": contact_reason,
        "grasp_step": grasp,
        "cumulative_lateral_path": float(sum(abs(current - previous) for previous, current in zip(lateral, lateral[1:]))),
        "cumulative_lateral_path_m": float(sum(abs(current - previous) for previous, current in zip(lateral, lateral[1:]))),
        "peak_lateral_excursion": float(max(abs(value - lateral[0]) for value in lateral)),
        "peak_lateral_excursion_m": float(max(abs(value - lateral[0]) for value in lateral)),
        "symmetry_level_s": cell.symmetry_level_s,
        "asymmetry_metric_A": float(scene["asymmetry_metric_A"]),
        "position_residual": float(scene["position_residual"]),
        "orientation_residual": float(scene["orientation_residual"]),
        "midline_residual": float(scene["midline_residual"]),
        "occlusion_check": scene["occlusion_check"],
        "target_visible": scene["target_visible"],
        "realised_object_poses": scene["realised_object_poses"],
        "arm_reset_pose": scene["arm_reset_pose"],
        "object_layout_symmetric_not_embodiment": True,
        "initial_state_sha256": _sha256_json(initial_state),
        "final_detached_release": detached,
        "right_censored": right_censored,
        "actions_executed": actions_executed,
        "action_cap": action_cap,
        "steps": steps,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": bundle.candidate_sha256,
        "artifacts": {
            "viewport_video": video_artifact,
            "executed_action_trace": actions_artifact,
            "live_scene_gate": live_gate_artifact,
            "runtime_identity": runtime_artifact,
            "raw_episode_jsonl": {"path": str(output_path), "integrity_scope": "post_close_manifest"},
        },
        "future_evidence": export.get("future_evidence"),
        "future_evidence_status": export.get("future_evidence_status", "not_exposed_by_action_only_interface"),
        "missing_measurement_policy": "NR remains null and is never converted to zero",
    }
    return record


def write_episode(*, record: Mapping[str, Any], output: Path) -> dict[str, Any]:
    output = Path(output).resolve()
    manifest = output.with_name(output.name + ".manifest.json")
    _require(not output.exists() and not manifest.exists(), f"refusing to overwrite retained episode: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    value = {
        "schema_version": "vla-wam-shared-v3e004-jsonl-manifest-v1",
        "registered_cell_id": record["registered_cell_id"],
        "row_count": 1,
        "jsonl_path": str(output),
        "jsonl_sha256": sha256_file(output),
        "jsonl_bytes": output.stat().st_size,
    }
    manifest.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def compile_episode(*, bundle: E004RuntimeBundle, export_path: Path, export_sha256: str, output: Path) -> dict[str, Any]:
    export_path = Path(export_path).resolve()
    _require(export_path.is_file() and sha256_file(export_path) == export_sha256, "simulator export digest mismatch")
    export = _finite_json(export_path)
    _require(isinstance(export, dict), "simulator export must be an object")
    cell = bundle.cell(str(export.get("registered_cell_id")))
    _require(cell.row["execution_mode"] == "new_behavioral_episode", "compiler cannot relabel preserved evidence")
    record = build_episode_record(export=export, bundle=bundle, cell=cell, output_path=output)
    manifest = write_episode(record=record, output=output)
    return {"cell_id": cell.cell_id, "success": record["success"], "failure_category": record["failure_category"], "manifest": manifest}


def _one_episode(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    _require(len(lines) == 1, f"expected one episode row: {path}")
    value = json.loads(lines[0], parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    _require(value.get("schema_version") == EPISODE_SCHEMA and value.get("behavioral_result_valid") is True, "pair input is not a valid E004 episode")
    manifest_path = Path(path).with_name(Path(path).name + ".manifest.json")
    manifest = _finite_json(manifest_path)
    _require(manifest.get("row_count") == 1 and manifest.get("jsonl_sha256") == sha256_file(path), "episode manifest changed")
    return value, manifest


def compile_pair(*, left_jsonl: Path, right_jsonl: Path, output: Path) -> dict[str, Any]:
    left, _ = _one_episode(Path(left_jsonl).resolve())
    right, _ = _one_episode(Path(right_jsonl).resolve())
    _require(left["requested_relation"] == "left" and right["requested_relation"] == "right", "pair directions are not LEFT/RIGHT")
    for key in ("matched_pair_id", "model_id", "arena", "environment_seed", "sampling_seed", "symmetry_level_s", "registration_sha256", "queue_sha256", "candidate_sha256"):
        _require(left[key] == right[key], f"matched pair differs for {key}")
    _require(left["initial_state_sha256"] == right["initial_state_sha256"], "matched directions do not share an identical reset")
    left_actions = np.load(left["artifacts"]["executed_action_trace"]["path"], allow_pickle=False)
    right_actions = np.load(right["artifacts"]["executed_action_trace"]["path"], allow_pickle=False)
    _require(left_actions.ndim == 2 and right_actions.ndim == 2 and left_actions.shape[1:] == right_actions.shape[1:], "matched action dimensions differ")
    prefix = min(10, len(left_actions), len(right_actions))
    _require(prefix > 0, "matched pair has no common executed-action prefix")
    delta = left_actions[:prefix].astype(np.float64) - right_actions[:prefix].astype(np.float64)
    left_offset = float(left["signed_final_lateral_offset_m"])
    right_offset = float(right["signed_final_lateral_offset_m"])
    row = {
        "schema_version": PAIR_SCHEMA,
        "study_id": left["study_id"],
        "amendment_id": left["amendment_id"],
        "matched_pair_id": left["matched_pair_id"],
        "model_id": left["model_id"],
        "arena": left["arena"],
        "environment_seed": left["environment_seed"],
        "sampling_seed": left["sampling_seed"],
        "symmetry_level_s": left["symmetry_level_s"],
        "asymmetry_metric_A_left": left["asymmetry_metric_A"],
        "asymmetry_metric_A_right": right["asymmetry_metric_A"],
        "identical_reset": True,
        "initial_state_sha256": left["initial_state_sha256"],
        "left_registered_cell_id": left["registered_cell_id"],
        "right_registered_cell_id": right["registered_cell_id"],
        "left_success": left["success"],
        "right_success": right["success"],
        "left_failure_category": left["failure_category"],
        "right_failure_category": right["failure_category"],
        "left_signed_final_lateral_offset_m": left_offset,
        "right_signed_final_lateral_offset_m": right_offset,
        "endpoint_shift": right_offset - left_offset,
        "endpoint_shift_right_minus_left_m": right_offset - left_offset,
        "endpoint_redirection_left_minus_right_m": left_offset - right_offset,
        "endpoint_ordering_aligned": left_offset > right_offset,
        "action_distinct": bool(np.any(delta != 0.0)),
        "action_distinct_prefix_steps": prefix,
        "action_prefix_l2": float(np.linalg.norm(delta)),
        "action_prefix_max_abs": float(np.max(np.abs(delta))),
        "left_episode": _file_record(left_jsonl, "left episode"),
        "right_episode": _file_record(right_jsonl, "right episode"),
        "registration_sha256": left["registration_sha256"],
        "queue_sha256": left["queue_sha256"],
        "candidate_sha256": left["candidate_sha256"],
    }
    output = Path(output).resolve()
    manifest = output.with_name(output.name + ".manifest.json")
    _require(not output.exists() and not manifest.exists(), f"refusing to overwrite retained pair: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(row, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    manifest_value = {
        "schema_version": "vla-wam-shared-v3e004-pair-manifest-v1",
        "matched_pair_id": row["matched_pair_id"],
        "row_count": 1,
        "jsonl_sha256": sha256_file(output),
        "jsonl_bytes": output.stat().st_size,
    }
    manifest.write_text(json.dumps(manifest_value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return row


def _bundle_from_args(args: argparse.Namespace) -> E004RuntimeBundle:
    return load_runtime_bundle(
        registration_path=args.registration,
        registration_sha256=args.registration_sha256,
        queue_path=args.queue,
        queue_sha256=args.queue_sha256,
        candidate_path=args.candidate,
        candidate_sha256=args.candidate_sha256,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    episode = sub.add_parser("episode")
    for name in ("registration", "queue", "candidate"):
        episode.add_argument(f"--{name}", type=Path, required=True)
        episode.add_argument(f"--{name}-sha256", required=True)
    episode.add_argument("--export", type=Path, required=True)
    episode.add_argument("--export-sha256", required=True)
    episode.add_argument("--output", type=Path, required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--left-jsonl", type=Path, required=True)
    pair.add_argument("--right-jsonl", type=Path, required=True)
    pair.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "episode":
        result = compile_episode(bundle=_bundle_from_args(args), export_path=args.export, export_sha256=args.export_sha256, output=args.output)
    else:
        result = compile_pair(left_jsonl=args.left_jsonl, right_jsonl=args.right_jsonl, output=args.output)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
