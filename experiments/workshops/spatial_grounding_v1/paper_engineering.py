"""One disclosed LAT scene realization and six recorded zero-model goal trials."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any, Mapping

import numpy as np

from .family_campaign_executor import CALIBRATION_SHA256, _fsync_json, _run_child
from .fixtures import FixtureCandidate, Pose, pose_error
from .grasp_calibration import SCHEMA as CALIBRATION_SCHEMA
from .lat_candidate_generator import workspace_digest
from .lat_proposals import _bounds, _overlap, _waypoints
from .paper_informed_geometry import translation_preview
from .prospective_family_capture import _manifest, verify_capture_artifacts
from .prospective_family_scene import _digest, _usda, _validate_base_receipt
from .qualification_batch_verifier import Registration, verify_candidate

SCHEMA = "sgw-01-paper-engineering-registration-v1"
ATTEMPT = "SGW-ENG-008-LAT-057"
OBJECT_NAMES = ["rubiks_cube", "banana", "bowl", "table"]
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"


def record(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def bound_file(value: Mapping[str, Any]) -> Path:
    path = Path(value["path"])
    observed = record(path)
    if observed["sha256"] != value["sha256"] or observed["bytes"] != value["bytes"]:
        raise ValueError(f"engineering input bytes differ: {path}")
    return path


def load_registration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if (value.get("schema_version") != SCHEMA or value.get("attempt_id") != ATTEMPT
            or value.get("source_candidate_id") != "LAT-CANDIDATE-057"
            or value.get("robolab_commit") != ROBOLAB_COMMIT
            or value.get("scripted_trials") != 6 or value.get("actions_per_trial") != 450
            or value.get("model_requests") != 0 or value.get("release_permitted") is not False
            or value.get("maximum_new_engineering_layouts") != 1):
        raise ValueError("registration is not the bounded zero-model paper-informed engineering attempt")
    for key in ("workspace", "proposals", "base_scene", "assets", "renderer", "calibration"):
        bound_file(value[key])
    if value["calibration"]["sha256"] != CALIBRATION_SHA256:
        raise ValueError("engineering attempt must retain the measured controller calibration")
    return value


def write_overlay(
    *, root: Path, base_scene: Path, workspace_path: Path,
    poses: Mapping[str, Mapping[str, Any]], phase: str,
) -> Path:
    """Sublayer without authoring any robot, camera, table or orientation change."""
    if not set(poses).issubset(set(OBJECT_NAMES) - {"table"}) or not {"rubiks_cube", "bowl"}.issubset(poses):
        raise ValueError("engineering overlay may move only measured non-table task objects")
    workspace = json.loads(workspace_path.read_bytes())
    _validate_base_receipt(workspace, workspace_path)
    root.mkdir(parents=True, exist_ok=False)
    usd = root / "scene.usda"
    usd.write_text(_usda(
        base_scene.resolve(), [],
        {name: pose["position_m"] for name, pose in poses.items()},
        {name: pose["quaternion_wxyz"] for name, pose in poses.items()},
    ), encoding="utf-8")
    value = {
        "schema_version": "sgw-01-paper-engineering-overlay-v1",
        "status": "prospective_scene_design_not_measured_or_qualified",
        "family": "LAT", "engineering_attempt_id": ATTEMPT, "engineering_phase": phase,
        "overlay_usda": record(usd), "base_scene": record(base_scene),
        "base_workspace_receipt": {
            **record(workspace_path), "receipt_sha256": workspace["receipt_sha256"],
            "asset_manifest_sha256": workspace["asset_manifest_sha256"],
            "robolab_commit": workspace["robolab_commit"],
        },
        "native_import_contract": {
            "robolab_utils_sha256": "6562517740be7c60e24e964afced5eac63bd00b4c5de0b6fee92295b87c81110",
            "objects_of_interest": list(OBJECT_NAMES), "dynamic_bodies": [],
            "kinematic_or_static_bodies": [], "kinematic_bodies": [],
            "banana_contact_bodies": ["table"],
        },
        "authored_actor_poses": dict(poses),
        "model_request_count": 0, "behavioral_episode_count": 0,
        "release_permitted": False,
    }
    value["manifest_sha256"] = _digest(value)
    path = root / "manifest.json"
    _fsync_json(path, value)
    _manifest(path)
    return path


def validate_native_scene(
    scene: Mapping[str, Any], poses: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the opt-in LAT task to the exact fresh capture and immutable overlay."""
    if scene.get("engineering_attempt_id") != ATTEMPT or scene.get("object_names") != OBJECT_NAMES:
        raise ValueError("LAT engineering scene identity/inventory differs")
    capture_path = bound_file(scene["capture"])
    manifest_path = bound_file(scene["manifest"])
    capture = json.loads(capture_path.read_bytes())
    manifest = _manifest(manifest_path)
    if (capture.get("family") != "LAT" or capture.get("receipt_sha256") != workspace_digest(capture)
            or capture.get("model_request_count") != 0 or capture.get("behavioral_episode_count") != 0
            or capture.get("overlay_manifest_sha256") != manifest["manifest_sha256"]
            or capture.get("asset_manifest_sha256") != manifest["base_workspace_receipt"]["asset_manifest_sha256"]
            or capture.get("robolab_commit") != ROBOLAB_COMMIT
            or manifest.get("engineering_phase") != "translated"
            or manifest.get("engineering_attempt_id") != ATTEMPT
            or scene.get("asset") != manifest["overlay_usda"]["path"]
            or scene.get("asset_sha256") != manifest["overlay_usda"]["sha256"]):
        raise ValueError("LAT engineering scene lacks its exact fresh translated capture")
    if capture["overlay_manifest"] != record(manifest_path):
        raise ValueError("capture does not bind the selected engineering manifest bytes")
    if poses is not None:
        expected = {name: {
            "position_m": capture["objects"][name]["root_position_env_local_xyz_m"],
            "quaternion_wxyz": capture["objects"][name]["root_quaternion_world_wxyz"],
        } for name in ("rubiks_cube", "bowl")}
        if poses != expected:
            raise ValueError("LAT engineering task poses differ from the fresh measured layout")
    return dict(scene)


def materialize_lat_candidate(
    *, capture_path: Path, manifest_path: Path, source_candidate: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    capture = json.loads(capture_path.read_bytes())
    manifest = _manifest(manifest_path)
    objects = capture["objects"]
    if set(objects) != set(OBJECT_NAMES):
        raise ValueError("LAT capture must retain the cube, bowl, banana and table")
    source = FixtureCandidate.from_json(source_candidate)
    cube_center = np.asarray(objects["rubiks_cube"]["geometric_center_env_local_xyz_m"])
    origin = np.asarray(capture["environment_origin_world_xyz_m"])
    goals = {}
    for label in ("positive", "negative"):
        old = source.metadata["abs_ik_waypoints"][label]
        displacement = np.asarray(old[-1]["position_world_xyz_m"])[:2] - np.asarray(old[0]["position_world_xyz_m"])[:2]
        target = cube_center.copy()
        target[:2] += displacement
        goals[label] = target
    table_min = np.asarray(objects["table"]["bbox_env_local_min_xyz_m"])
    table_max = np.asarray(objects["table"]["bbox_env_local_max_xyz_m"])
    cube_bounds = [_bounds(objects["rubiks_cube"], point) for point in (cube_center, *goals.values())]
    bowl = _bounds(objects["bowl"], np.asarray(objects["bowl"]["geometric_center_env_local_xyz_m"]))
    banana = _bounds(objects["banana"], np.asarray(objects["banana"]["geometric_center_env_local_xyz_m"]))
    rejections = []
    if any(np.any(low[:2] < table_min[:2] + .01) or np.any(high[:2] > table_max[:2] - .01)
           for low, high in (*cube_bounds, bowl, banana)):
        rejections.append("initial_or_goal_footprint_outside_measured_table_clearance")
    corridor = (np.minimum(cube_bounds[1][0], cube_bounds[2][0]),
                np.maximum(cube_bounds[1][1], cube_bounds[2][1]))
    if _overlap(corridor, bowl) or _overlap(corridor, banana):
        rejections.append("measured_cube_transport_corridor_overlaps_reference_or_distractor")
    if _overlap(bowl, banana):
        rejections.append("reference_distractor_footprints_overlap")
    scene = {
        "asset": manifest["overlay_usda"]["path"], "asset_sha256": manifest["overlay_usda"]["sha256"],
        "object_names": list(OBJECT_NAMES), "engineering_attempt_id": ATTEMPT,
        "capture": record(capture_path), "manifest": record(manifest_path),
    }
    validate_native_scene(scene)
    candidate = {
        "candidate_id": ATTEMPT, "family": "LAT", "seed": source.seed,
        "asset_manifest_sha256": source.asset_manifest_sha256, "task_asset": scene["asset"],
        "object_poses": {name: {
            "position_m": objects[name]["root_position_env_local_xyz_m"],
            "quaternion_wxyz": objects[name]["root_quaternion_world_wxyz"],
        } for name in ("rubiks_cube", "bowl")},
        "metadata": {
            "status": "separate_engineering_layout_not_frozen_pool_member",
            "engineering_attempt_id": ATTEMPT, "native_scene": scene,
            "engineering_geometry_measurements": True,
            "collision_geometry_local_bounds": capture.get("collision_geometry_local_bounds", {
                "available": False, "reason": "native capture has no collision geometry inventory",
            }),
            "scoring_center_offsets_root_local_m": {
                name: objects[name]["geometric_center_offset_root_local_xyz_m"] for name in ("rubiks_cube", "bowl")
            },
            "abs_ik_waypoints": {label: _waypoints(cube_center, target, origin) for label, target in goals.items()},
            "geometric_screen_status": "rejected" if rejections else "passed",
            "geometric_rejection_reasons": rejections, "historical_layout_fingerprint": None,
        },
    }
    FixtureCandidate.from_json(candidate)
    return candidate, {
        "status": "measured_footprint_rejection" if rejections else "measured_object_footprint_screen_passed",
        "reasons": rejections, "goal_centers_env_local_xyz_m": {label: value.tolist() for label, value in goals.items()},
        "scope": "Object footprints/corridor only; not arm/gripper reachability or a fixture release.",
        "release_permitted": False,
    }


def _capture(
    *, manifest: Path, registration: Mapping[str, Any], study_root: Path, robolab: Path,
) -> Path:
    root = manifest.parent
    output = root / "capture.json"
    command = [
        sys.executable, "-m", "experiments.workshops.spatial_grounding_v1.prospective_family_capture",
        "--study-root", str(study_root), "--robolab-root", str(robolab),
        "--assets-manifest", registration["assets"]["path"], "--renderer-receipt", registration["renderer"]["path"],
        "--overlay-manifest", str(manifest), "--output", str(output), "--environment-seed", "20260922",
        "--headless", "--num-envs", "1", "--device", "cuda:0", "--renderer", "realtime",
        "--rendering-type", "balanced", "--rendering_mode", "balanced",
        f"--kit_args=--portable-root={root}/kit --/rtx/verifyDriverVersion/enabled=false",
    ]
    _run_child(command, label="capture", root=root, values={}, timeout_seconds=1200)
    verified = verify_capture_artifacts(output)
    _fsync_json(root / "capture-verification.json", verified)
    return output


def compare_native_configuration(reference: Mapping[str, Any], translated: Mapping[str, Any]) -> dict[str, Any]:
    """Record native differences without inventing bit-identical physics."""
    before, after = reference["robot_snapshot"], translated["robot_snapshot"]
    if before["asset_usd"].get("available") is not True or before["asset_usd"] != after["asset_usd"]:
        raise ValueError("native robot asset identity is unavailable or changed")
    if before["joint_names"] != after["joint_names"]:
        raise ValueError("native robot joint inventory changed")
    joint_delta = np.asarray(after["joint_position_rad"]) - np.asarray(before["joint_position_rad"])
    if not np.isfinite(joint_delta).all() or np.max(np.abs(joint_delta)) > math.radians(2):
        raise ValueError("native robot initial joints differ beyond the retained angular tolerance")
    comparisons = {
        "robot": pose_error(
            Pose.from_json({"position_m": before["base_position_env_local_xyz_m"],
                            "quaternion_wxyz": before["base_quaternion_world_wxyz"]}),
            Pose.from_json({"position_m": after["base_position_env_local_xyz_m"],
                            "quaternion_wxyz": after["base_quaternion_world_wxyz"]}),
        ),
    }
    names = {"over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"}
    previous = reference["camera_extrinsics"]["cameras"]
    current = translated["camera_extrinsics"]["cameras"]
    if set(previous) != names or set(current) != names:
        raise ValueError("native camera inventory differs")
    for name in sorted(names):
        if previous[name].get("available") is not True or current[name].get("available") is not True:
            raise ValueError(f"required native camera extrinsics unavailable: {name}")
        comparisons[name] = pose_error(
            Pose.from_json({"position_m": previous[name]["position_world_xyz_m"],
                            "quaternion_wxyz": previous[name]["quaternion_world_wxyz"]}),
            Pose.from_json({"position_m": current[name]["position_world_xyz_m"],
                            "quaternion_wxyz": current[name]["quaternion_world_wxyz"]}),
        )
    if any(position > .003 or angle > 2 for position, angle in comparisons.values()):
        raise ValueError("native robot/camera poses differ beyond the retained reset tolerances")
    return {
        "pose_errors": {name: {"max_component_position_error_m": pair[0], "angle_error_degrees": pair[1]}
                        for name, pair in comparisons.items()},
        "max_joint_position_difference_rad": float(np.max(np.abs(joint_delta))),
        "authored_robot_and_cameras_unchanged": True,
        "native_consistency_scope": "Existing 3mm/2degree reset tolerances, not bit-identical physics or a new success gate.",
    }


def run(*, registration_path: Path, output: Path) -> dict[str, Any]:
    registration = load_registration(registration_path)
    if output.exists():
        raise FileExistsError(f"refusing to reuse engineering output {output}")
    output.mkdir(parents=True)
    _fsync_json(output / "registration-binding.json", record(registration_path))
    try:
        study = Path(__file__).resolve().parents[3]
        robolab = Path(registration["robolab_root"])
        workspace_path = bound_file(registration["workspace"])
        proposals = json.loads(bound_file(registration["proposals"]).read_bytes())
        source = next(row for row in proposals["candidates"] if row["candidate_id"] == registration["source_candidate_id"])
        workspace = json.loads(workspace_path.read_bytes())
        if source["metadata"]["workspace_receipt_sha256"] != workspace["receipt_sha256"]:
            raise ValueError("source LAT layout does not bind the selected measured workspace")
        reference_manifest = write_overlay(
            root=output / "reference", base_scene=bound_file(registration["base_scene"]),
            workspace_path=workspace_path, poses=source["object_poses"], phase="reference_zero_action_only",
        )
        reference_capture = _capture(manifest=reference_manifest, registration=registration, study_root=study, robolab=robolab)
        reference = json.loads(reference_capture.read_bytes())
        robot = reference["robot_snapshot"]
        preview = translation_preview(reference["objects"], robot["base_position_env_local_xyz_m"])
        preview["fresh_reference_capture"] = record(reference_capture)
        _fsync_json(output / "translation-calculation.json", preview)
        manifest_path = write_overlay(
            root=output / "translated", base_scene=bound_file(registration["base_scene"]),
            workspace_path=workspace_path, poses=preview["translated_actor_poses"], phase="translated",
        )
        capture_path = _capture(manifest=manifest_path, registration=registration, study_root=study, robolab=robolab)
        capture = json.loads(capture_path.read_bytes())
        actual_robot = capture["robot_snapshot"]
        configuration = compare_native_configuration(reference, capture)
        _fsync_json(output / "robot-camera-comparison.json", configuration)
        realized = {}
        for name, expected in preview["translated_actor_poses"].items():
            row = capture["objects"][name]
            errors = pose_error(Pose.from_json({
                "position_m": row["root_position_env_local_xyz_m"],
                "quaternion_wxyz": row["root_quaternion_world_wxyz"],
            }), Pose.from_json(expected))
            if errors[0] > .003 or errors[1] > 2:
                raise ValueError(f"translated {name} realization exceeds existing reset pose tolerances")
            realized[name] = {"max_component_position_error_m": errors[0], "angle_error_degrees": errors[1]}
        realized["cube_center_robot_distance_m"] = math.dist(
            capture["objects"]["rubiks_cube"]["geometric_center_env_local_xyz_m"],
            actual_robot["base_position_env_local_xyz_m"],
        )
        _fsync_json(output / "realization.json", {
            "status": "new_scene_natively_materialized_not_qualified", "capture": record(capture_path),
            "robot_camera_comparison": record(output / "robot-camera-comparison.json"),
            "measurements": realized, "release_permitted": False,
        })
        candidate, footprints = materialize_lat_candidate(
            capture_path=capture_path, manifest_path=manifest_path, source_candidate=source,
        )
        _fsync_json(output / "footprint-screen.json", footprints)
        proposal_path = output / "engineering-proposal.json"
        _fsync_json(proposal_path, {
            "status": "proposed_unqualified", "model_request_count": 0, "behavioral_episode_count": 0,
            "engineering_attempt_id": ATTEMPT, "outside_all_frozen_candidate_pools": True, "candidates": [candidate],
        })
        if footprints["reasons"]:
            result = {"status": "new_scene_materialized_geometrically_rejected", "reason": footprints["reasons"],
                      "new_scripted_trials": 0, "model_requests": 0, "release_permitted": False}
            _fsync_json(output / "result.json", result)
            return result
        trial_root = output / f"0-{ATTEMPT}"
        trial_root.mkdir()
        free = shutil.disk_usage(output).free
        if free < 3860 * 1024**3:
            raise RuntimeError("engineering recording would violate the concurrent frozen queue's storage reserve")
        _fsync_json(trial_root / "storage_preflight.json", {
            "plan_sha256": record(registration_path)["sha256"], "index": 0, "candidate_id": ATTEMPT,
            "status": "passed_conservative_batch_space_check", "model_requests": 0, "behavioral_episodes": 0,
            "free_bytes": free, "required_bytes": 3860 * 1024**3,
        })
        command = [
            sys.executable, "-m", "experiments.workshops.spatial_grounding_v1.model_blind_qualification",
            "--family", "LAT", "--proposal-file", str(proposal_path), "--candidate-id", ATTEMPT,
            "--output-root", str(trial_root / "result"),
            "--bridge-factory", "experiments.workshops.spatial_grounding_v1.robolab_lat_qualification:create_bridge",
            "--controller-factory", "experiments.workshops.spatial_grounding_v1.robolab_lat_qualification:create_controller",
            "--controller-calibration", registration["calibration"]["path"],
            "--robolab-root", str(robolab), "--assets-manifest", registration["assets"]["path"],
            "--headless", "--num-envs", "1", "--device", "cuda:0", "--renderer", "realtime",
            "--rendering-type", "balanced", "--rendering_mode", "balanced",
            f"--kit_args=--portable-root={trial_root}/kit --/rtx/verifyDriverVersion/enabled=false",
        ]
        _run_child(command, label="qualification", root=trial_root, values={}, timeout_seconds=2400)
        calibration = json.loads(bound_file(registration["calibration"]).read_bytes())
        verification = verify_candidate(Registration(
            {"candidate_ids_in_frozen_hash_order": [ATTEMPT]},
            record(registration_path)["sha256"], {ATTEMPT: FixtureCandidate.from_json(candidate)},
            {"recipe": CALIBRATION_SCHEMA, "calibration_sha256": CALIBRATION_SHA256, "calibration": calibration},
        ), index=0, root=trial_root)
        _fsync_json(output / "qualification-verification.json", verification)
        result = {
            "status": "both_goal_paths_recorded_and_independently_verified",
            "physical_outcome": verification["status"], "passed_checks": verification["passed_checks"],
            "new_scripted_trials": 6, "model_requests": 0, "release_permitted": False,
            "outside_all_frozen_candidate_pools": True,
            "next_step": "Inspect actual native geometry/clearance and both-goal outcomes before any new campaign.",
        }
        _fsync_json(output / "result.json", result)
        return result
    except Exception:
        _fsync_json(output / "infrastructure-failure.json", {
            "status": "engineering_infrastructure_failure_not_model_or_physical_failure",
            "traceback": traceback.format_exc(), "model_requests": 0, "release_permitted": False,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(registration_path=args.registration, output=args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
