from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.fastwam_model_blind_gate import (
    arm_reset_pose,
    camera_centers,
    collision_bounding_radius,
    reference_occludes_target,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.fastwam_robotwin import (
    ARENA,
    CORE_SEEDS,
    EXPECTED_OBJECT,
    EXPECTED_REFERENCE,
    LEVELS,
    MODEL_ID,
    PROMPTS,
    RELATIONS,
    SOURCE_RELEASE_QUEUE_SHA256,
    FastWAME004Error,
    asymmetry_A,
    candidate_payload,
    canonicalize_candidate_floats,
    canonical_json_bytes,
    layout_for_level,
    load_candidate,
    quaternion_yaw,
    residuals,
    validate_registered_queue,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.fastwam_runtime import (
    episode_measures,
)
from tools.build_v3e004_fastwam_cohort_manifest import Invalid, validate_progress_closed


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


def registered_rows() -> list[dict]:
    return [json.loads(line) for line in (BASE / "queue.jsonl").read_text().splitlines() if line.strip()]


def test_fastwam_s0_is_exact_v3b007_fixture_and_s1_is_full_pose_symmetric():
    control = layout_for_level(0.0)
    symmetric = layout_for_level(1.0)
    assert set(control) == set(symmetric) == {"target", "reference"}
    assert control["target"].asset_identity == symmetric["target"].asset_identity == EXPECTED_OBJECT
    assert control["reference"].asset_identity == symmetric["reference"].asset_identity == EXPECTED_REFERENCE
    assert control["target"].position_xyz_m == pytest.approx(
        (-0.047076620161533356, -0.030880313366651535, 0.7405446767807007)
    )
    assert control["reference"].position_xyz_m == pytest.approx(
        (-0.21130692958831787, -0.1640346497297287, 0.7408550977706909)
    )
    # RoboTwin native source-x is the lateral axis.  Each singleton is on its
    # reflection plane, and its world-Z yaw is self-mirrored.
    assert all(pose.position_xyz_m[0] == pytest.approx(0.0) for pose in symmetric.values())
    assert all(quaternion_yaw(pose.quaternion_wxyz) == pytest.approx(0.0, abs=1e-12) for pose in symmetric.values())
    assert asymmetry_A(control) > 0.0
    assert asymmetry_A(symmetric) == pytest.approx(0.0, abs=1e-12)
    assert residuals(symmetric)["midline_residual_m"] == pytest.approx(0.0)
    assert residuals(symmetric)["orientation_residual_rad"] == pytest.approx(0.0, abs=1e-12)


def test_fastwam_candidate_is_strict_hash_bound_and_preserves_arena_boundary(tmp_path):
    payload = candidate_payload()
    assert payload["arena"] == ARENA == "robotwin"
    assert payload["inventory"]["target_count"] == 1
    assert payload["inventory"]["reference_count"] == 1
    assert payload["inventory"]["no_duplicated_reference"] is True
    assert payload["coordinate_contract"]["native_lateral_axis"] == "source_x"
    assert payload["source_release"]["source_queue_sha256"] == SOURCE_RELEASE_QUEUE_SHA256
    assert payload["matched_seeds"] == list(CORE_SEEDS)
    path = tmp_path / "candidate.json"
    path.write_bytes(canonical_json_bytes(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_candidate(path, digest) == payload
    path.write_bytes(path.read_bytes().replace(b'"target_count":1', b'"target_count":2'))
    with pytest.raises(FastWAME004Error, match="SHA-256"):
        load_candidate(path, digest)


def test_fastwam_candidate_float_hash_is_cross_libm_stable():
    variants = [0.00022248008525882487, 0.0002224800852588249]
    assert len({canonicalize_candidate_floats(value) for value in variants}) == 1
    variants = [-0.0002222756212954535, -0.00022227562129545353]
    assert len({canonicalize_candidate_floats(value) for value in variants}) == 1


def test_registered_fastwam_queue_is_exact_108_cell_9400_series():
    selected = validate_registered_queue(registered_rows())
    assert len(selected) == 108
    assert {int(row["environment_seed"]) for row in selected} == set(CORE_SEEDS)
    assert {float(row["symmetry_level_s"]) for row in selected} == set(LEVELS)
    assert {str(row["relation"]) for row in selected} == set(RELATIONS)
    assert {str(row["prompt"]) for row in selected} == set(PROMPTS.values())
    assert all(row["model_id"] == MODEL_ID and row["arena"] == "robotwin" for row in selected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"environment_seed": 9900, "sampling_seed": 9900}, "non-E004 seed"),
        ({"prompt": "Put it left."}, "prompt bytes"),
        ({"arena": "droid"}, "arena boundary"),
    ],
)
def test_registered_fastwam_queue_fails_closed_on_old_seed_prompt_or_arena(mutation, message):
    rows = registered_rows()
    target = next(row for row in rows if row.get("model_id") == MODEL_ID)
    target.update(mutation)
    with pytest.raises(FastWAME004Error, match=message):
        validate_registered_queue(rows)


def test_conservative_camera_segment_occlusion_geometry():
    camera = (0.0, 0.0, 1.0)
    target = (0.0, 0.0, 0.0)
    assert reference_occludes_target(camera, target, (0.0, 0.0, 0.5), 0.05) is True
    assert reference_occludes_target(camera, target, (0.2, 0.0, 0.5), 0.05) is False
    assert reference_occludes_target(camera, target, (0.0, 0.0, 1.2), 0.5) is False


def test_robtwin_camera_rig_and_actor_wrapper_introspection():
    class Pose:
        def __init__(self, p):
            self.p = p

    class Entity:
        def __init__(self, p):
            self._pose = Pose(p)

        def get_pose(self):
            return self._pose

    class Camera:
        def __init__(self, p):
            self.entity = Entity(p)

    class Shape:
        vertices = [[-0.03, -0.04, 0.0], [0.03, 0.04, 0.0]]
        scale = [1.0, 1.0, 1.0]

    class Component:
        collision_shapes = [Shape()]

    class ActorEntity:
        def get_components(self):
            return [Component()]

    class ActorWrapper:
        actor = ActorEntity()

    class Articulation:
        def __init__(self, values):
            self.values = values

        def get_qpos(self):
            return self.values

    class Robot:
        left_entity = Articulation([1.0, 2.0])
        right_entity = Articulation([3.0, 4.0])

    class Rig:
        left_camera = Camera((1.0, 0.0, 0.0))
        right_camera = Camera((-1.0, 0.0, 0.0))
        static_camera_name = ["head_camera"]
        static_camera_list = [Camera((0.0, 1.0, 1.0))]

    class Env:
        cameras = Rig()
        robot = Robot()
        scene = None

    assert camera_centers(Env()) == {
        "head_camera": (0.0, 1.0, 1.0),
        "left_camera": (1.0, 0.0, 0.0),
        "right_camera": (-1.0, 0.0, 0.0),
    }
    assert collision_bounding_radius(ActorWrapper()) == pytest.approx(0.05)
    reset = arm_reset_pose(Env())
    assert reset["status"] == "available"
    assert reset["robots"]["robot.left"]["joint_positions_rad"] == [1.0, 2.0]


def test_fastwam_episode_measurement_keeps_unavailable_contact_null_and_uses_normalized_sign():
    trajectory = [
        {
            "object_xyz": [-0.01, 0.0, 0.74],
            "object_minus_target_x": -0.01,
            "object_minus_target_y": 0.0,
            "grippers_open": False,
        },
        {
            "object_xyz": [-0.12, 0.0, 0.79],
            "object_minus_target_x": -0.12,
            "object_minus_target_y": 0.0,
            "grippers_open": True,
        },
        {
            "object_xyz": [-0.13, 0.0, 0.79],
            "object_minus_target_x": -0.13,
            "object_minus_target_y": 0.0,
            "grippers_open": True,
        },
        {
            "object_xyz": [-0.14, 0.0, 0.79],
            "object_minus_target_x": -0.14,
            "object_minus_target_y": 0.0,
            "grippers_open": True,
        },
    ]
    row = episode_measures(
        {"requested_relation": "left", "requested_success": True, "actions_executed": 3},
        trajectory,
    )
    assert row["failure_category"] == "correct"
    assert row["signed_final_lateral_offset"] == pytest.approx(0.14)
    assert row["requested_side_depth"] == pytest.approx(0.14)
    assert row["time_to_first_contact"] is None
    assert row["cone_entry_step"] == 1
    assert row["cone_entry_sustained"] is True


def test_fastwam_cohort_closer_accepts_bounded_resume_tail_but_not_partial_progress(tmp_path):
    validate_progress_closed(
        [{"event": "seed_complete", "seed": 9413}, {"event": "seed_reused", "seed": 9414}],
        list(range(9401, 9415)),
        root=tmp_path,
    )
    with pytest.raises(Invalid, match="not closed"):
        validate_progress_closed(
            [{"event": "seed_started", "seed": 9414}],
            list(range(9401, 9415)),
            root=tmp_path,
        )
    with pytest.raises(Invalid, match="wrong seed"):
        validate_progress_closed(
            [{"event": "seed_reused", "seed": 9413}],
            list(range(9401, 9415)),
            root=tmp_path,
        )
