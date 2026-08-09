"""Compile native LingBot/RoboTwin outputs into E005 evidence.

The scorer below re-evaluates only RoboTwin's frozen relation-aware region and
detached-release contract.  It never imports the DROID cone predicate, and
unavailable contact/future measurements remain null rather than becoming zero.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from .runtime_contract import (
    AMENDMENT_ID,
    ARENA,
    MODEL_ID,
    STUDY_ID,
    E005ContractError,
    RegisteredBundle,
    RegisteredCell,
    canonical_json_bytes,
    canonical_sha256,
    file_record,
    require,
    sha256_file,
)


ACTION_CAP = 400
PICKUP_LIFT_M = 0.03
SUSTAINED_SAMPLES = 3
FAILURE_CATEGORIES = {
    "correct",
    "pick_failed",
    "transport_failed",
    "wrong_side",
    "release_failed",
}
EPISODE_SCHEMA = "vla-wam-shared-v3e005-lingbot-robotwin-episode-v1"
PAIR_SCHEMA = "vla-wam-shared-v3e005-lingbot-robotwin-pair-v1"
INFRASTRUCTURE_SCHEMA = "vla-wam-shared-v3e005-lingbot-infrastructure-attempt-v1"


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise E005ContractError(f"cannot read native trajectory {path}: {error}") from error
    require(isinstance(value, list) and bool(value), "native trajectory must be a non-empty list")
    require(all(isinstance(step, dict) for step in value), "native trajectory contains a non-object step")
    return value


def native_relation_region(step: Mapping[str, Any], relation: str) -> bool:
    """Exact Phase-A RoboTwin relation region in native SAPIEN coordinates."""

    require(relation in {"left", "right"}, "RoboTwin relation must be left/right")
    try:
        dx = float(step["object_minus_target_x"])
        dy = float(step["object_minus_target_y"])
    except (KeyError, TypeError, ValueError) as error:
        raise E005ContractError(f"malformed native RoboTwin relation state: {error}") from error
    require(math.isfinite(dx) and math.isfinite(dy), "native relation state is non-finite")
    distance = math.hypot(dx, dy)
    side = dx < 0.0 if relation == "left" else dx > 0.0
    return bool(0.08 < distance < 0.20 and side and abs(dy) < 0.05)


def first_sustained(values: Sequence[bool], length: int = SUSTAINED_SAMPLES) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run == length:
            return index - length + 1
    return None


def final_sustained(values: Sequence[bool], length: int = SUSTAINED_SAMPLES) -> bool:
    return len(values) >= length and all(values[-length:])


def validate_video(path: Path, *, decode: bool) -> dict[str, Any]:
    record = file_record(path)
    require(record["bytes"] > 0, "simulator viewport video is empty")
    if decode:
        completed = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(Path(path).resolve()),
                "-map",
                "0:v:0",
                "-f",
                "null",
                "-",
            ],
            check=False,
            capture_output=True,
        )
        require(
            completed.returncode == 0,
            "simulator video failed decode validation: "
            + completed.stderr.decode(errors="replace").strip(),
        )
    return record


def validate_action_trace(path: Path, result: Mapping[str, Any], actions: int) -> dict[str, Any]:
    declared = result.get("action_trace")
    require(isinstance(declared, dict), "native result lacks action_trace provenance")
    resolved = Path(path).expanduser().resolve()
    require(
        Path(str(declared.get("path", ""))).expanduser().resolve() == resolved,
        "native action trace path leaves the exact cell directory",
    )
    record = file_record(resolved)
    require(declared.get("sha256") == record["sha256"], "native action trace hash claim is false")
    require(declared.get("count") == actions, "native action trace count differs from episode length")
    shape = declared.get("shape")
    require(
        isinstance(shape, list) and bool(shape) and shape[0] == actions,
        "native action trace shape differs from episode length",
    )
    try:
        import numpy as np

        with np.load(resolved) as payload:
            require("executed" in payload, "native action trace lacks executed array")
            observed_shape = list(payload["executed"].shape)
    except (OSError, ValueError) as error:
        raise E005ContractError(f"cannot read native action trace {resolved}: {error}") from error
    require(observed_shape == shape, "executed action array shape differs from native provenance")
    record["shape"] = observed_shape
    record["count"] = actions
    return record


def _float_xyz(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 3, f"{label} must be XYZ")
    output = [float(component) for component in value]
    require(all(math.isfinite(component) for component in output), f"{label} must be finite")
    return output


def validate_native_trajectory(
    trajectory: list[dict[str, Any]], relation: str, actions: int
) -> None:
    require(len(trajectory) == actions + 1, "trajectory must retain initial state plus one state per action")
    for index, step in enumerate(trajectory):
        require(step.get("action_step") == index, "native trajectory action_step is not contiguous")
        _float_xyz(step.get("object_xyz"), f"trajectory[{index}].object_xyz")
        _float_xyz(step.get("target_xyz"), f"trajectory[{index}].target_xyz")
        require(type(step.get("grippers_open")) is bool, "native trajectory lacks grippers_open")
        expected_region = native_relation_region(step, relation)
        require(
            step.get("relation_region") == expected_region,
            "native relation_region differs from the frozen RoboTwin predicate",
        )
        expected_success = bool(expected_region and step["grippers_open"])
        require(
            step.get("success") == expected_success,
            "native success differs from relation region plus detached release",
        )


def episode_measurements(
    result: Mapping[str, Any], trajectory: list[dict[str, Any]]
) -> dict[str, Any]:
    relation = str(result["requested_relation"])
    opposite = "right" if relation == "left" else "left"
    initial_z = float(trajectory[0]["object_xyz"][2])
    pickup = [
        float(step["object_xyz"][2]) >= initial_z + PICKUP_LIFT_M
        and not bool(step["grippers_open"])
        for step in trajectory
    ]
    requested = [native_relation_region(step, relation) for step in trajectory]
    opposite_mask = [native_relation_region(step, opposite) for step in trajectory]
    grasp_step = first_sustained(pickup)
    entry_step = first_sustained(requested)
    success = bool(result["requested_success"])
    if success:
        category = "correct"
    elif grasp_step is None:
        category = "pick_failed"
    elif final_sustained(opposite_mask):
        category = "wrong_side"
    elif final_sustained(requested) and not bool(trajectory[-1]["grippers_open"]):
        category = "release_failed"
    else:
        category = "transport_failed"
    require(category in FAILURE_CATEGORIES, "unknown failure category")
    native_x = [float(step["object_xyz"][0]) for step in trajectory]
    signed_lateral = [-value for value in native_x]
    final_dx = float(trajectory[-1]["object_minus_target_x"])
    signed_final = -final_dx
    return {
        "success": success,
        "failure_category": category,
        # RoboTwin native LEFT is -source-X.  Positive is therefore robot-left.
        "signed_final_lateral_offset": signed_final,
        "requested_side_depth": signed_final * (1.0 if relation == "left" else -1.0),
        # The field name is fixed by registration; the event is RoboTwin's
        # native relation-region entry, not the DROID 45-degree predicate.
        "cone_entry_step": entry_step,
        "cone_entry_sustained": entry_step is not None,
        "episode_length": int(result["actions_executed"]),
        "time_to_first_contact": None,
        "time_to_first_contact_unavailable_reason": (
            "The frozen LingBot RoboTwin runner does not retain a contact-force stream."
        ),
        "grasp_step": grasp_step,
        "cumulative_lateral_path": sum(
            abs(current - previous)
            for previous, current in zip(signed_lateral, signed_lateral[1:])
        ),
        "peak_lateral_excursion": max(
            (abs(value - signed_lateral[0]) for value in signed_lateral), default=0.0
        ),
    }


def _require_snapshot(snapshot: Mapping[str, Any], cell: RegisteredCell) -> dict[str, Any]:
    required = {
        "realised_object_poses",
        "arm_reset_pose",
        "asymmetry_metric_A",
        "position_residual_m",
        "orientation_residual_rad",
        "midline_residual_m",
        "occlusion_check",
        "all_camera_occlusion_checks",
        "mirrored_asset_identity_verified",
        "mirrored_yaw_verified",
    }
    require(required <= set(snapshot), "live reset snapshot lacks registered geometry fields")
    require(isinstance(snapshot["realised_object_poses"], dict), "realised_object_poses is not an object")
    for key in (
        "asymmetry_metric_A",
        "position_residual_m",
        "orientation_residual_rad",
        "midline_residual_m",
    ):
        value = snapshot[key]
        require(type(value) in {int, float} and math.isfinite(float(value)), f"snapshot {key} is non-finite")
    require(type(snapshot["occlusion_check"]) is bool, "occlusion_check must be boolean")
    if math.isclose(cell.symmetry_level, 1.0, abs_tol=1e-12):
        require(float(snapshot["position_residual_m"]) < 0.001, "s1 position residual failed")
        require(
            float(snapshot["orientation_residual_rad"]) < math.radians(0.5),
            "s1 orientation residual failed",
        )
        require(float(snapshot["midline_residual_m"]) < 0.001, "s1 midline residual failed")
        require(snapshot["occlusion_check"] is False, "s1 target is occluded")
        require(snapshot["mirrored_asset_identity_verified"] is True, "s1 asset mirror failed")
        require(snapshot["mirrored_yaw_verified"] is True, "s1 yaw mirror failed")
    return dict(snapshot)


def physical_snapshot_fingerprint(snapshot: Mapping[str, Any]) -> str:
    """Hash only physical reset fields, excluding timestamps/cell annotations."""

    payload = {
        "realised_object_poses": snapshot["realised_object_poses"],
        "arm_reset_pose": snapshot["arm_reset_pose"],
    }
    return canonical_sha256(payload)


def build_provisional_episode(
    *,
    bundle: RegisteredBundle,
    cell: RegisteredCell,
    result_path: Path,
    snapshot_path: Path,
    runtime: Mapping[str, Any],
    candidate_sha256: str,
    model_blind_gate_sha256: str,
    expected_study_commit: str,
    attempt_id: str,
    verify_video_decode: bool,
) -> dict[str, Any]:
    """Validate one native cell; pair-only fields are added after its mate."""

    result_file = Path(result_path).expanduser().resolve()
    result = json.loads(result_file.read_text())
    require(isinstance(result, dict), "native result is not an object")
    expected = {
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "requested_relation": cell.relation,
        "prompt": cell.prompt,
        "prompt_family": "direct_command",
    }
    for key, value in expected.items():
        require(result.get(key) == value, f"{cell.cell_id}: native result mismatch for {key}")
    require(
        result.get("condition") == f"{cell.level_code}__{cell.relation}",
        f"{cell.cell_id}: native condition label drift",
    )
    require(type(result.get("requested_success")) is bool, "native requested_success is not boolean")
    actions = result.get("actions_executed")
    require(type(actions) is int and 0 <= actions <= ACTION_CAP, "native action count exceeds frozen cap")
    condition_dir = result_file.parent
    trajectory_path = condition_dir / "trajectory.json"
    require(
        Path(str(result.get("trajectory_path", ""))).expanduser().resolve()
        == trajectory_path.resolve(),
        "native trajectory path leaves the exact cell directory",
    )
    trajectory = load_trajectory(trajectory_path)
    validate_native_trajectory(trajectory, cell.relation, actions)
    require(
        result["requested_success"] == trajectory[-1]["success"],
        "native requested_success differs from frozen final scorer",
    )
    video_path = condition_dir / "simulator.mp4"
    require(
        Path(str(result.get("simulator_video", ""))).expanduser().resolve() == video_path.resolve(),
        "native video path leaves the exact cell directory",
    )
    video_record = validate_video(video_path, decode=verify_video_decode)
    action_record = validate_action_trace(condition_dir / "action_trace.npz", result, actions)
    latent_path = condition_dir / "first_predicted_latent.pt"
    require(
        Path(str(result.get("first_predicted_latent_path", ""))).expanduser().resolve()
        == latent_path.resolve(),
        "LingBot latent path leaves the exact cell directory",
    )
    latent_record = file_record(latent_path)
    require(latent_record["bytes"] > 0, "LingBot exposed latent tensor is empty")
    snapshot = _require_snapshot(json.loads(Path(snapshot_path).read_text()), cell)
    measures = episode_measurements(result, trajectory)
    queue_row_sha = hashlib.sha256(canonical_json_bytes(cell.row)).hexdigest()
    return {
        "schema_version": EPISODE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "cell_id": cell.cell_id,
        "attempt_id": attempt_id,
        "matched_seed_id": cell.row["matched_seed_id"],
        "matched_layout_pair_id": cell.matched_layout_pair_id,
        "scene_id": cell.scene_id,
        "scene_cluster_id": cell.row["scene_cluster_id"],
        "anchor_task": cell.anchor_task,
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "relation": cell.relation,
        "prompt": cell.prompt,
        "prompt_sha256": cell.row["prompt_sha256"],
        "static_episode_prompt": True,
        "success_predicate_id": "frozen_v3_robotwin_relation_aware_success",
        "outcome_coordinate_contract": "frozen_robotwin_native_lateral_axis_and_region",
        **measures,
        "endpoint_shift": None,
        "action_distinct": None,
        "action_pair": None,
        "symmetry_level_s": cell.symmetry_level,
        "asymmetry_metric_A": float(snapshot["asymmetry_metric_A"]),
        "position_residual": float(snapshot["position_residual_m"]),
        "orientation_residual": float(snapshot["orientation_residual_rad"]),
        "midline_residual": float(snapshot["midline_residual_m"]),
        "occlusion_check": bool(snapshot["occlusion_check"]),
        "all_camera_occlusion_checks": snapshot["all_camera_occlusion_checks"],
        "realised_object_poses": snapshot["realised_object_poses"],
        "arm_reset_pose": snapshot["arm_reset_pose"],
        "mirrored_asset_identity_verified": bool(snapshot["mirrored_asset_identity_verified"]),
        "mirrored_yaw_verified": bool(snapshot["mirrored_yaw_verified"]),
        "initial_physical_fingerprint_sha256": physical_snapshot_fingerprint(snapshot),
        "future_interface": "latent_only_future_not_decodable",
        "future_evidence": [{"kind": "latent_tensor_not_decoded", **latent_record}],
        "source_artifacts": {
            "result": file_record(result_file),
            "trajectory": file_record(trajectory_path),
            "simulator_viewport_video": video_record,
            "executed_action_trace": action_record,
            "live_reset_snapshot": file_record(snapshot_path),
        },
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "queue_row_sha256": queue_row_sha,
        "layout_candidate_sha256": candidate_sha256,
        "model_blind_gate_sha256": model_blind_gate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "checkpoint_revision": runtime["checkpoint"]["revision"],
        "checkpoint_manifest_sha256": runtime["checkpoint"]["hash_manifest_artifact"]["sha256"],
        "external_repository_commit": runtime["external_repository"]["commit"],
        "simulator_repository_commit": runtime["simulator_repository"]["commit"],
        "study_commit": expected_study_commit,
    }


def action_pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    try:
        import numpy as np

        with np.load(left_path) as payload:
            left = np.asarray(payload["executed"], dtype=np.float64)
        with np.load(right_path) as payload:
            right = np.asarray(payload["executed"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as error:
        raise E005ContractError(f"cannot compare executed action traces: {error}") from error
    count = min(10, len(left), len(right))
    rms = float(np.sqrt(np.mean(np.square(left[:count] - right[:count])))) if count else None
    return {
        "actions_compared": count,
        "first_10_action_rms": rms,
        "action_distinct": bool(rms is not None and rms > 0.0),
    }


def close_pair(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    require(
        left.get("relation") == "left" and right.get("relation") == "right",
        "pair relation order is not LEFT/RIGHT",
    )
    for key in (
        "matched_layout_pair_id",
        "environment_seed",
        "sampling_seed",
        "symmetry_level_s",
        "scene_id",
        "registration_sha256",
        "queue_sha256",
        "layout_candidate_sha256",
        "model_blind_gate_sha256",
        "runtime_identity_sha256",
        "study_commit",
        "attempt_id",
    ):
        require(left.get(key) == right.get(key), f"matched pair differs for {key}")
    if "execution_lane" in left or "execution_lane" in right:
        require(
            left.get("execution_lane") == right.get("execution_lane"),
            "matched pair differs for execution_lane",
        )
    require(
        left.get("initial_physical_fingerprint_sha256")
        == right.get("initial_physical_fingerprint_sha256"),
        "LEFT/RIGHT initial physical snapshots are not identical",
    )
    action = action_pair(
        Path(left["source_artifacts"]["executed_action_trace"]["path"]),
        Path(right["source_artifacts"]["executed_action_trace"]["path"]),
    )
    # Positive means the LEFT prompt ended farther robot-left than RIGHT.
    endpoint_shift = float(left["signed_final_lateral_offset"]) - float(
        right["signed_final_lateral_offset"]
    )
    output: list[dict[str, Any]] = []
    for source in (left, right):
        row = dict(source)
        row["endpoint_shift"] = endpoint_shift
        row["action_distinct"] = action["action_distinct"]
        row["action_pair"] = action
        output.append(row)
    pair = {
        "schema_version": PAIR_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "matched_layout_pair_id": left["matched_layout_pair_id"],
        "environment_seed": left["environment_seed"],
        "sampling_seed": left["sampling_seed"],
        "scene_id": left["scene_id"],
        "symmetry_level_s": left["symmetry_level_s"],
        "left_cell_id": left["cell_id"],
        "right_cell_id": right["cell_id"],
        "endpoint_shift": endpoint_shift,
        "action_pair": action,
        "initial_physical_fingerprint_sha256": left[
            "initial_physical_fingerprint_sha256"
        ],
        "registration_sha256": left["registration_sha256"],
        "queue_sha256": left["queue_sha256"],
        "layout_candidate_sha256": left["layout_candidate_sha256"],
        "model_blind_gate_sha256": left["model_blind_gate_sha256"],
        "runtime_identity_sha256": left["runtime_identity_sha256"],
        "study_commit": left["study_commit"],
        "attempt_id": left["attempt_id"],
        "execution_lane": left.get("execution_lane"),
    }
    pair["pair_sha256"] = canonical_sha256(pair)
    return output[0], output[1], pair


def infrastructure_record(
    *,
    cell: RegisteredCell,
    attempt_id: str,
    error: BaseException | str,
    stage: str,
    retained_paths: Sequence[Path],
    bundle: RegisteredBundle,
    candidate_sha256: str,
    model_blind_gate_sha256: str,
) -> dict[str, Any]:
    message = str(error)
    retained = [file_record(path) for path in retained_paths if Path(path).is_file()]
    partial = bool(retained)
    row = {
        "schema_version": INFRASTRUCTURE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "cell_id": cell.cell_id,
        "matched_seed_id": cell.row["matched_seed_id"],
        "matched_layout_pair_id": cell.matched_layout_pair_id,
        "environment_seed": cell.environment_seed,
        "sampling_seed": cell.sampling_seed,
        "scene_id": cell.scene_id,
        "symmetry_level_s": cell.symmetry_level,
        "relation": cell.relation,
        "prompt": cell.prompt,
        "attempt_id": attempt_id,
        "classification": "partial" if partial else "technical_invalid",
        "behavioral_result_valid": False,
        "stage": stage,
        "error": message,
        "error_sha256": hashlib.sha256(message.encode()).hexdigest(),
        "retained_artifacts": retained,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "layout_candidate_sha256": candidate_sha256,
        "model_blind_gate_sha256": model_blind_gate_sha256,
        "denominator_policy": "excluded from RoboTwin behavioral denominator; never encoded as zero",
    }
    row["record_sha256"] = canonical_sha256(row)
    return row
