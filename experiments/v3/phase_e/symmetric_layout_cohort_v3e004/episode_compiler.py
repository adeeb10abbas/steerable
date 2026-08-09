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
from .request0_replay import (
    AMENDMENT_SCHEMA,
    CACHE_MANIFEST_SCHEMA,
    CAPTURE_ATTESTATION_SCHEMA,
    EVIDENCE_ENVELOPE_SCHEMA,
    REPLAY_ATTESTATION_SCHEMA,
    RESET_CONTRACT_SCHEMA,
    canonical_json_sha256,
)
from .runtime_contract import E004Cell, E004RuntimeBundle, RuntimeContractError, load_runtime_bundle, sha256_file


EXPORT_SCHEMA = "vla-wam-shared-v3e004-droid-simulator-export-v1"
EPISODE_SCHEMA = "vla-wam-shared-v3e004-droid-behavioral-episode-v1"
PAIR_SCHEMA = "vla-wam-shared-v3e004-droid-matched-pair-v1"
CONTACT_UNAVAILABLE = (
    "The pinned RoboLab integration exposes a verified object-grabbed conditional "
    "and detached release but no verified physical contact stream; grasp is not "
    "substituted for contact."
)
REQUEST0_ENVELOPE_SCHEMA = EVIDENCE_ENVELOPE_SCHEMA


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


def frozen_requested_success(
    steps: list[Mapping[str, Any]], relation: str, detached_release: bool
) -> bool:
    """Apply the frozen B001 behavioral success predicate.

    B001 terminates on a detached release in the requested 45-degree cone.
    Three-step sustained cone entry is retained as a separate trajectory
    diagnostic; it is not an additional binary-success requirement.
    """
    return bool(steps and _cone(steps[-1], relation) and detached_release)


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


def _validate_live_gate(
    record: Any, *, bundle: E004RuntimeBundle, cell: E004Cell
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
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
    snapshot_artifact = _file_record(value.get("snapshot"), "live scene snapshot")
    snapshot = _finite_json(Path(snapshot_artifact["path"]))
    _require(
        isinstance(snapshot, dict)
        and snapshot.get("schema_version") == "vla-wam-shared-v3e004-live-scene-snapshot-v1",
        "live scene snapshot schema changed",
    )
    for key, wanted in (
        ("registered_cell_id", cell.cell_id),
        ("registered_cell_sha256", cell.row_sha256),
        ("registration_sha256", bundle.registration_sha256),
        ("queue_sha256", bundle.queue_sha256),
        ("candidate_sha256", bundle.candidate_sha256),
    ):
        _require(snapshot.get(key) == wanted, f"live scene snapshot differs for {key}")
    cameras = snapshot.get("cameras")
    _require(isinstance(cameras, dict) and cameras, "live scene snapshot has no cameras")
    camera_identity: dict[str, dict[str, Any]] = {}
    for name, camera in sorted(cameras.items()):
        _require(isinstance(camera, dict), f"camera record is invalid: {name}")
        rgb_sha = camera.get("rgb_source_sha256")
        rgb_shape = camera.get("rgb_source_shape")
        rgb_dtype = camera.get("rgb_source_dtype")
        _require(isinstance(rgb_sha, str) and len(rgb_sha) == 64, f"camera RGB hash is missing: {name}")
        _require(
            isinstance(rgb_shape, list)
            and len(rgb_shape) == 3
            and all(type(value) is int and value > 0 for value in rgb_shape),
            f"camera RGB shape is invalid: {name}",
        )
        _require(isinstance(rgb_dtype, str) and rgb_dtype, f"camera RGB dtype is missing: {name}")
        camera_identity[str(name)] = {
            "rgb_source_sha256": rgb_sha,
            "rgb_source_shape": rgb_shape,
            "rgb_source_dtype": rgb_dtype,
        }
    return artifact, scene, camera_identity


def _digest(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _same_artifact(binding: Any, artifact: Mapping[str, Any], label: str) -> None:
    _require(isinstance(binding, Mapping), f"{label} binding is missing")
    for key in ("path", "bytes", "sha256"):
        _require(binding.get(key) == artifact.get(key), f"{label} binding differs for {key}")


def _validated_reset_contract(artifact: Mapping[str, Any], label: str) -> tuple[dict[str, Any], str]:
    value = _finite_json(Path(str(artifact["path"])))
    _require(isinstance(value, dict) and value.get("schema_version") == RESET_CONTRACT_SCHEMA, f"{label} schema changed")
    payload_sha = _digest(value.get("reset_contract_sha256"), f"{label} payload")
    unsigned = {key: child for key, child in value.items() if key != "reset_contract_sha256"}
    _require(payload_sha == canonical_json_sha256(unsigned), f"{label} self-digest changed")
    return value, payload_sha


def _validate_request0_replay(
    record: Any,
    *,
    bundle: E004RuntimeBundle,
    cell: E004Cell,
) -> dict[str, Any]:
    """Require prospective R001 evidence for every matched E004 DROID row."""

    _require(isinstance(record, Mapping), "R001 request-zero evidence is missing")
    _require(record.get("schema_version") == REQUEST0_ENVELOPE_SCHEMA, "R001 evidence envelope schema changed")
    expected_mode = "capture_left" if cell.relation == "left" else "replay_right"
    _require(record.get("mode") == expected_mode, "R001 evidence mode differs from registered relation")
    _require(
        record.get("closed_loop_observation_policy") == "native_after_first_executed_action",
        "R001 closed-loop observation policy changed",
    )
    artifacts = {
        name: _file_record(record.get(name), f"R001 {name.replace('_', ' ')}")
        for name in (
            "amendment",
            "cache_manifest",
            "observation_cache",
            "reset_contract",
            "native_reset_contract",
            "attestation",
        )
    }
    amendment = _finite_json(Path(artifacts["amendment"]["path"]))
    _require(isinstance(amendment, dict) and amendment.get("schema_version") == AMENDMENT_SCHEMA, "R001 amendment schema changed")
    _require(amendment.get("registered_before_new_request") is True, "R001 amendment was not prospective")
    for key, wanted in (
        ("study_id", cell.row["study_id"]),
        ("registration_sha256", bundle.registration_sha256),
        ("queue_sha256", bundle.queue_sha256),
        ("candidate_sha256", bundle.candidate_sha256),
    ):
        _require(amendment.get(key) == wanted, f"R001 amendment differs for {key}")
    manifest = _finite_json(Path(artifacts["cache_manifest"]["path"]))
    _require(isinstance(manifest, dict) and manifest.get("schema_version") == CACHE_MANIFEST_SCHEMA, "R001 cache manifest schema changed")
    _require(manifest.get("source_relation") == "left", "R001 cache was not captured from LEFT")
    _require(manifest.get("source_cell_id") == f"{cell.matched_pair_id}:left", "R001 cache source cell differs from matched LEFT")
    _require(manifest.get("matched_pair_id") == cell.matched_pair_id, "R001 cache matched-pair identity changed")
    _require(manifest.get("model_request_count_at_capture") == 0, "R001 cache was captured after a model request")
    _require(manifest.get("behavioral_action_count_at_capture") == 0, "R001 cache was captured after a behavioral action")
    _same_artifact(manifest.get("amendment"), artifacts["amendment"], "R001 amendment")
    _same_artifact(manifest.get("observation_cache"), artifacts["observation_cache"], "R001 observation cache")
    _same_artifact(manifest.get("reset_contract"), artifacts["reset_contract"], "R001 LEFT reset contract")
    left_contract, left_reset_sha = _validated_reset_contract(artifacts["reset_contract"], "R001 LEFT reset contract")
    _require(manifest.get("reset_contract", {}).get("payload_sha256") == left_reset_sha, "R001 reset payload binding changed")
    observation_sha = _digest(manifest.get("observation_payload_sha256"), "R001 observation payload")
    _require(record.get("observation_payload_sha256") == observation_sha, "R001 observation payload differs from cache")
    _require(record.get("reset_contract_payload_sha256") == left_reset_sha, "R001 reset payload differs from LEFT cache")
    native_contract, native_reset_sha = _validated_reset_contract(
        artifacts["native_reset_contract"], "R001 native reset contract"
    )
    _require(native_contract == left_contract and native_reset_sha == left_reset_sha, "R001 native reset contract differs from LEFT")
    identity = canonical_json_sha256(
        {
            "observation_payload_sha256": observation_sha,
            "reset_contract_sha256": left_reset_sha,
        }
    )
    _require(record.get("pair_identity_sha256") == identity, "R001 request-zero pair identity changed")
    attestation = _finite_json(Path(artifacts["attestation"]["path"]))
    _require(isinstance(attestation, dict), "R001 attestation is not an object")
    _require(attestation.get("matched_pair_id") == cell.matched_pair_id, "R001 attestation pair changed")
    _require(attestation.get("model_request_count_at_attestation") == 0, "R001 attestation followed a model request")
    _require(attestation.get("behavioral_action_count_at_attestation") == 0, "R001 attestation followed a behavioral action")
    _same_artifact(attestation.get("amendment"), artifacts["amendment"], "R001 attested amendment")
    _same_artifact(attestation.get("cache_manifest"), artifacts["cache_manifest"], "R001 attested cache manifest")
    _same_artifact(attestation.get("observation_cache"), artifacts["observation_cache"], "R001 attested observation cache")
    if expected_mode == "capture_left":
        _require(attestation.get("schema_version") == CAPTURE_ATTESTATION_SCHEMA, "R001 LEFT capture attestation schema changed")
        _require(attestation.get("registered_cell_id") == cell.cell_id, "R001 LEFT capture cell changed")
        _require(attestation.get("mode") == "capture_left", "R001 LEFT attestation mode changed")
        _same_artifact(attestation.get("reset_contract"), artifacts["reset_contract"], "R001 attested LEFT reset contract")
        _require(attestation.get("observation_payload_sha256") == observation_sha, "R001 captured observation payload changed")
        _require(attestation.get("reset_contract_payload_sha256") == left_reset_sha, "R001 captured reset payload changed")
    else:
        _require(attestation.get("schema_version") == REPLAY_ATTESTATION_SCHEMA, "R001 RIGHT replay attestation schema changed")
        _require(attestation.get("target_cell_id") == cell.cell_id, "R001 RIGHT replay cell changed")
        _require(attestation.get("target_relation") == "right", "R001 replay relation changed")
        _require(attestation.get("physical_state_and_camera_contract_bit_identical") is True, "R001 physical reset gate did not pass")
        _require(attestation.get("request0_non_language_bytes_bit_identical") is True, "R001 request-zero bytes were not identical")
        _require(attestation.get("request0_observation_payload_sha256") == observation_sha, "R001 replayed payload changed")
        _require(attestation.get("right_reset_contract_sha256") == left_reset_sha, "R001 RIGHT reset payload changed")
        _require(attestation.get("closed_loop_observation_policy") == "native_after_first_executed_action", "R001 replay scope changed")
        _same_artifact(attestation.get("left_reset_contract"), artifacts["reset_contract"], "R001 attested LEFT reset contract")
        right_binding = attestation.get("right_native_reset_contract")
        _same_artifact(right_binding, artifacts["native_reset_contract"], "R001 attested RIGHT reset contract")
        _require(right_binding.get("payload_sha256") == left_reset_sha, "R001 attested RIGHT reset payload changed")
    return {
        "schema_version": REQUEST0_ENVELOPE_SCHEMA,
        "mode": expected_mode,
        "pair_identity_sha256": identity,
        "observation_payload_sha256": observation_sha,
        "reset_contract_payload_sha256": left_reset_sha,
        "closed_loop_observation_policy": "native_after_first_executed_action",
        "artifacts": artifacts,
    }


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
    live_gate_artifact, scene, camera_identity = _validate_live_gate(
        export.get("live_scene_gate"), bundle=bundle, cell=cell
    )
    request0 = _validate_request0_replay(export.get("request0_replay"), bundle=bundle, cell=cell)
    steps = _normalize_steps(export.get("steps"))
    actions_executed = len(steps) - 1
    action_cap = int(cell.row["runtime_identity_requirement"]["action_cap"])
    _require(export.get("actions_executed") == actions_executed, "state/action count mismatch")
    _require(1 <= actions_executed <= action_cap, "executed action count is outside the registered cap")
    success, right_censored = export.get("requested_success"), export.get("right_censored")
    detached = export.get("final_detached_release")
    _require(type(success) is bool and type(right_censored) is bool and type(detached) is bool, "scorer booleans are invalid")
    _require(
        success == frozen_requested_success(steps, cell.relation, detached),
        "requested_success differs from the frozen B001 detached-release-in-cone predicate",
    )
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
        "initial_rgb_views": camera_identity,
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
    requested = [_cone(step, cell.relation) for step in steps]
    failure = _failure_category(success=success, steps=steps, relation=cell.relation, detached_release=detached)
    output_path = Path(output_path).resolve()
    native_initial_state_sha256 = _sha256_json(initial_state)
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
        "initial_state_sha256": native_initial_state_sha256,
        "native_initial_state_sha256": native_initial_state_sha256,
        "native_initial_rgb_views": camera_identity,
        "request0_pair_identity_sha256": request0["pair_identity_sha256"],
        "request0_observation_payload_sha256": request0["observation_payload_sha256"],
        "request0_reset_contract_sha256": request0["reset_contract_payload_sha256"],
        "request0_replay_mode": request0["mode"],
        "request0_replay": request0,
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
            "request0_replay": request0["artifacts"],
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
    for key in (
        "request0_pair_identity_sha256",
        "request0_observation_payload_sha256",
        "request0_reset_contract_sha256",
    ):
        _require(left.get(key) == right.get(key), f"matched directions differ for {key}")
    _require(left.get("request0_replay_mode") == "capture_left", "matched LEFT row lacks R001 capture attestation")
    _require(right.get("request0_replay_mode") == "replay_right", "matched RIGHT row lacks R001 replay attestation")
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
        "r001_request0_identity_verified": True,
        "r001_physical_state_and_camera_contract_identical": True,
        "identical_policy_request0_non_language_bytes": True,
        "request0_pair_identity_sha256": left["request0_pair_identity_sha256"],
        "request0_observation_payload_sha256": left["request0_observation_payload_sha256"],
        "request0_reset_contract_sha256": left["request0_reset_contract_sha256"],
        "native_initial_rgb_bytes_identical": left["native_initial_rgb_views"] == right["native_initial_rgb_views"],
        "left_native_initial_state_sha256": left["native_initial_state_sha256"],
        "right_native_initial_state_sha256": right["native_initial_state_sha256"],
        "left_native_initial_rgb_views": left["native_initial_rgb_views"],
        "right_native_initial_rgb_views": right["native_initial_rgb_views"],
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
