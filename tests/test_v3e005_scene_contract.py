from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.cross_arena_geometry_v3e005.model_blind_gate import (
    reference_occludes_target,
    target_visibility_pixels,
)
from experiments.v3.phase_e.cross_arena_geometry_v3e005.runtime_contract import (
    E005ContractError,
    QUEUE_SHA256,
    REGISTRATION_SHA256,
    load_registered_bundle,
)
from experiments.v3.phase_e.cross_arena_geometry_v3e005.scene_contract import (
    CAMERAS,
    ORIENTATION_TOLERANCE_RAD,
    POSITION_TOLERANCE_M,
    SCENE_IDS,
    ActorPose,
    actor_pose,
    asymmetry_A,
    candidate_payload,
    candidate_sha256,
    canonical_json_bytes,
    layout_for,
    load_candidate,
    quaternion_yaw,
    residuals,
    symmetric_layout,
    validate_live_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"


def _resolved_scene(candidate: dict, scene_id: str) -> dict:
    source = candidate["scenes"][scene_id]
    target_identity = tuple(source["assets"]["target"]["asset_identity"])
    reference_identity = tuple(source["assets"]["reference"]["asset_identity"])
    centers = source["control_source"]["historical_settled_centers_native_xyz_m"]
    control = {
        "target": ActorPose(tuple(centers["target"]), (0.5, 0.5, 0.5, 0.5), target_identity),
        "reference": ActorPose(tuple(centers["reference"]), (0.5, 0.5, 0.5, 0.5), reference_identity),
    }
    symmetric = symmetric_layout(control)
    return {
        "scene_id": scene_id,
        "resolved_layouts": {
            "0.00": {role: pose.to_json() for role, pose in control.items()},
            "1.00": {role: pose.to_json() for role, pose in symmetric.items()},
        },
    }


def _snapshot(candidate: dict, scene_id: str, level: float, gate_scene: dict) -> dict:
    layout = layout_for(candidate, scene_id, level, gate_scene)
    return {
        "realised_object_poses": layout,
        "asset_contract": candidate["scenes"][scene_id]["assets"],
        "occlusion_check": {camera: False for camera in CAMERAS},
        "views": {
            camera: {
                "shape": [480, 640, 3],
                "dtype": "uint8",
                "pixel_range": 255,
                "target_visible_pixels": 100,
            }
            for camera in CAMERAS
        },
        "arm_reset_pose": {
            "status": "available",
            "robots": {"robot.left": {"joint_positions_rad": [0.0]}},
        },
    }


def test_candidate_binds_exact_registration_queue_and_seven_scene_inventory():
    candidate = candidate_payload()
    assert candidate["registration_sha256"] == REGISTRATION_SHA256
    assert candidate["queue_sha256"] == QUEUE_SHA256
    assert tuple(candidate["scenes"]) == SCENE_IDS
    assert candidate["model_request_count"] == 0
    assert candidate["behavioral_episode_count"] == 0
    expected = {
        "robotwin_pair_03": (("086_woodenblock", 1), ("081_playingcards", 1)),
        "robotwin_pair_04": (("047_mouse", 0), ("048_stapler", 2)),
        "robotwin_pair_05": (("081_playingcards", 0), ("073_rubikscube", 1)),
        "robotwin_pair_06": (("113_coffee-box", 3), ("081_playingcards", 1)),
        "robotwin_pair_07": (("075_bread", 6), ("048_stapler", 0)),
        "robotwin_pair_08": (("081_playingcards", 2), ("077_phone", 4)),
        "robotwin_pair_09": (("073_rubikscube", 0), ("086_woodenblock", 0)),
    }
    lingbot_manifest = json.loads(
        (
            ROOT
            / "artifacts/vla_wam_shared_v3/results/lingbot_va_robotwin_phase_a_evidence_hash_manifest.json"
        ).read_text()
    )
    source_hashes = {
        row["sha256"]
        for row in lingbot_manifest.get("files", lingbot_manifest.get("entries", []))
    }
    for scene_id, (target, reference) in expected.items():
        scene = candidate["scenes"][scene_id]
        assert tuple(scene["assets"]["target"]["asset_identity"]) == target
        assert tuple(scene["assets"]["reference"]["asset_identity"]) == reference
        assert scene["inventory"] == {
            "target_count": 1,
            "reference_count": 1,
            "no_duplicated_reference": True,
            "mirrored_clutter_pairs": [],
            "source_task_has_no_other_movable_actors": True,
        }
        assert scene["assets"]["target"]["material"]["contract"] == "exact_visual_mesh_bytes"
        assert scene["assets"]["reference"]["material"]["contract"] == "exact_visual_mesh_bytes"
        assert scene["control_source"]["historical_result_evidence"]["sha256"] in source_hashes
        assert "lingbot_va" in scene["control_source"]["historical_result_evidence"]["manifest"]


def test_scene_contract_matches_all_108_registered_cells_without_modifying_queue():
    bundle = load_registered_bundle(ROOT)
    candidate = candidate_payload()
    assert len(bundle.cells) == 108
    for cell in bundle.cells:
        scene = candidate["scenes"][cell.scene_id]
        assert scene["anchor_task"] == cell.anchor_task
        assert scene["prompts"][cell.relation] == cell.prompt
        assert cell.symmetry_level in {0.0, 1.0}
    assert hashlib.sha256((BASE / "registration.json").read_bytes()).hexdigest() == REGISTRATION_SHA256
    assert hashlib.sha256((BASE / "queue.jsonl").read_bytes()).hexdigest() == QUEUE_SHA256


def test_candidate_loader_fails_closed_on_byte_or_binding_drift(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_bytes(canonical_json_bytes(candidate_payload()))
    digest = candidate_sha256()
    assert load_candidate(path, digest) == candidate_payload()
    path.write_bytes(path.read_bytes().replace(b'"target_count":1', b'"target_count":2', 1))
    with pytest.raises(E005ContractError, match="SHA-256"):
        load_candidate(path, digest)
    path.write_bytes(canonical_json_bytes(candidate_payload()))
    with pytest.raises(E005ContractError, match="registration"):
        load_candidate(path, digest, "0" * 64, QUEUE_SHA256)


def test_symmetric_transform_centers_both_singletons_and_sets_self_mirrored_yaw():
    source = {
        "target": ActorPose((0.17, -0.1, 0.741), (0.5, 0.5, 0.5, 0.5), ("a", 1)),
        "reference": ActorPose((-0.11, -0.02, 0.742), (0.35, 0.36, 0.61, 0.61), ("b", 2)),
    }
    transformed = symmetric_layout(source)
    assert {pose.position_xyz_m[0] for pose in transformed.values()} == {0.0}
    assert transformed["target"].position_xyz_m[1:] == source["target"].position_xyz_m[1:]
    assert transformed["reference"].position_xyz_m[1:] == source["reference"].position_xyz_m[1:]
    assert quaternion_yaw(transformed["target"].quaternion_wxyz) == pytest.approx(0.0, abs=1e-12)
    assert quaternion_yaw(transformed["reference"].quaternion_wxyz) == pytest.approx(0.0, abs=1e-12)
    assert residuals(transformed)["midline_residual_m"] < POSITION_TOLERANCE_M
    assert residuals(transformed)["orientation_residual_rad"] < ORIENTATION_TOLERANCE_RAD
    assert asymmetry_A(transformed) == pytest.approx(0.0, abs=1e-10)


def test_layout_for_requires_gate_scene_only_for_concrete_pose_resolution():
    candidate = candidate_payload()
    recipe = layout_for(candidate, "robotwin_pair_04", 1.0)
    assert recipe["native_x_m"] == 0.0
    assert recipe["source_pose_required"] is True
    gate_scene = _resolved_scene(candidate, "robotwin_pair_04")
    concrete = layout_for(candidate, "robotwin_pair_04", 1.0, gate_scene)
    assert actor_pose(concrete, "target").position_xyz_m[0] == pytest.approx(0.0)
    assert actor_pose(concrete, "reference").position_xyz_m[0] == pytest.approx(0.0)


def test_live_snapshot_validates_assets_views_arm_pose_occlusion_and_strict_s1():
    candidate = candidate_payload()
    scene_id = "robotwin_pair_07"
    gate_scene = _resolved_scene(candidate, scene_id)
    snapshot = _snapshot(candidate, scene_id, 1.0, gate_scene)
    result = validate_live_snapshot(candidate, scene_id, 1.0, snapshot, gate_scene)
    assert result["passed"] is True
    assert result["midline_residual_m"] == pytest.approx(0.0)
    broken = deepcopy(snapshot)
    broken["occlusion_check"]["head_camera"] = True
    with pytest.raises(E005ContractError, match="occluded"):
        validate_live_snapshot(candidate, scene_id, 1.0, broken, gate_scene)
    broken = deepcopy(snapshot)
    broken["asset_contract"]["target"]["scale_xyz"] = [9.0, 9.0, 9.0]
    with pytest.raises(E005ContractError, match="scale/material"):
        validate_live_snapshot(candidate, scene_id, 1.0, broken, gate_scene)
    broken = deepcopy(snapshot)
    broken["arm_reset_pose"]["status"] = "unavailable"
    with pytest.raises(E005ContractError, match="arm reset"):
        validate_live_snapshot(candidate, scene_id, 1.0, broken, gate_scene)


def test_conservative_camera_segment_occlusion_geometry():
    camera = (0.0, 0.0, 1.0)
    target = (0.0, 0.0, 0.0)
    assert reference_occludes_target(camera, target, (0.0, 0.0, 0.5), 0.05) is True
    assert reference_occludes_target(camera, target, (0.2, 0.0, 0.5), 0.05) is False
    assert reference_occludes_target(camera, target, (0.0, 0.0, 1.2), 0.5) is False


def test_target_visibility_requires_actor_pixels_in_every_registered_camera():
    import numpy as np

    class Entity:
        per_scene_id = 7

    class Actor:
        actor = Entity()

    class Camera:
        def __init__(self, visible: bool = True):
            self.visible = visible

        def get_picture(self, name: str):
            assert name == "Segmentation"
            value = np.zeros((4, 5, 4), dtype=np.uint32)
            if self.visible:
                value[1:3, 2:4, 1] = 7
            return value

    class Rig:
        left_camera = Camera()
        right_camera = Camera()
        static_camera_name = ["head_camera"]
        static_camera_list = [Camera()]

    class Env:
        cameras = Rig()

    assert target_visibility_pixels(Env(), Actor()) == {
        "head_camera": 4,
        "left_camera": 4,
        "right_camera": 4,
    }
    Env.cameras.right_camera = Camera(visible=False)
    assert target_visibility_pixels(Env(), Actor()) == {
        "head_camera": 4,
        "left_camera": 4,
        "right_camera": 0,
    }
