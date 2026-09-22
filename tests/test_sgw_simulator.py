from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate, Pose
from experiments.workshops.spatial_grounding_v1.model_blind_qualification import _load_selected_candidate, qualify_candidate
from experiments.workshops.spatial_grounding_v1.lat_proposals import propose_lat_layouts
from experiments.workshops.spatial_grounding_v1.simulator_bridge import ObjectState, ResetResult, SimulatorSnapshot


def fixture() -> FixtureCandidate:
    return FixtureCandidate.from_json({
        "candidate_id": "lat-001", "family": "LAT", "seed": 1, "task_asset": "measured.usda",
        "asset_manifest_sha256": "b" * 64,
        "metadata": {"scoring_center_offsets_root_local_m": {"rubiks_cube": [0, 0, 0], "bowl": [0, 0, 0]}},
        "object_poses": {
            "rubiks_cube": {"position_m": [0.4, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
            "bowl": {"position_m": [0.5, 0.0, 0.1], "quaternion_wxyz": [1, 0, 0, 0]},
        },
    })


def state(y: float, z: float, *, supported: bool = True, attached: bool = False) -> dict[str, ObjectState]:
    return {
        "rubiks_cube": ObjectState(Pose((0.4, y, z), (1, 0, 0, 0)), 0.0, 0.0, supported, attached),
        "bowl": ObjectState(Pose((0.5, 0.0, 0.1), (1, 0, 0, 0)), 0.0, 0.0, True, False),
    }


class FakeEnvironment:
    def __init__(self, candidate=None, *, rejected=False, interrupt=None) -> None:
        self.goal = 1
        self.steps = 0
        self.candidate = candidate or fixture()
        self.rejected = rejected
        self.interrupt = interrupt
        self.closed = False

    def objects(self, y=0.0, lift=0.0, *, supported=True, attached=False):
        centers = self.candidate.scoring_poses()
        cube = centers["rubiks_cube"].position_m
        return {
            "rubiks_cube": ObjectState(Pose((cube[0], cube[1] + y, cube[2] + lift), (1, 0, 0, 0)),
                                      0.0, 0.0, supported, attached),
            "bowl": ObjectState(centers["bowl"], 0.0, 0.0, True, False),
        }

    def reset(self) -> ResetResult:
        self.steps = 0
        initial = self.objects()
        return ResetResult(
            SimulatorSnapshot(initial, 0.0, reset_root_poses=self.candidate.object_poses),
            {
                "reset_id": f"reset-{id(self)}",
                "camera_id": "head-v1",
                "camera_name": "head_camera",
                "fingerprint": "a" * 64,
                "temporal_cache_reset": True,
                "control_step_dt_s": 0.2,
            },
        )

    def step(self, _action: list[float]) -> SimulatorSnapshot:
        self.steps += 1
        if self.steps == self.interrupt:
            raise RuntimeError("injected mid-step infrastructure failure")
        if self.rejected:
            return SimulatorSnapshot(self.objects(), self.steps * 0.2)
        if self.steps <= 3:
            return SimulatorSnapshot(self.objects(lift=0.04, supported=False, attached=True), self.steps * 0.2)
        return SimulatorSnapshot(self.objects(y=0.04 * self.goal), self.steps * 0.2)

    def close(self) -> None:
        self.closed = True

    def snapshot(self) -> SimulatorSnapshot:
        return SimulatorSnapshot(state(0.0, 0.1), 0.0)

    def render_viewport(self):
        return np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)


class FakeBridge:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def create_environment(self, _task, _seed: int) -> FakeEnvironment:
        self.environment = FakeEnvironment(**self.kwargs)
        return self.environment


class FakeController:
    def actions_for_goal(self, environment: FakeEnvironment, _candidate, goal_sign: int):
        environment.goal = goal_sign
        return [np.zeros((1, 8)) for _ in range(450)]


def test_qualification_runs_exactly_six_model_blind_checks(tmp_path) -> None:
    candidate = fixture()
    candidate.metadata["scoring_center_offsets_root_local_m"]["rubiks_cube"] = [0, 0.029, 0]
    candidate.object_poses["rubiks_cube"] = Pose((0.4, -0.029, 0.1), (1, 0, 0, 0))
    receipt = qualify_candidate(candidate, FakeBridge(candidate=candidate), FakeController(), seed=4, evidence_root=tmp_path)
    assert receipt["status"] == "accepted_model_blind_fixture_candidate"
    assert receipt["model_request_count"] == 0
    assert len(receipt["checks"]) == 6
    assert {check["goal_sign"] for check in receipt["checks"]} == {-1, 1}
    assert all(len(check["per_step_states"]) == check["actions_executed"] + 1 for check in receipt["checks"])
    assert all(check["viewport_video"]["frame_count"] == 451 for check in receipt["checks"])
    assert all(check["requested_margin_m"] == pytest.approx(0.04) for check in receipt["checks"])
    assert sum(1 for _ in iio.imiter(receipt["checks"][0]["viewport_video"]["path"])) == 451


def test_rejected_trials_keep_all_six_videos_and_states(tmp_path):
    receipt = qualify_candidate(fixture(), FakeBridge(rejected=True), FakeController(), seed=4, evidence_root=tmp_path)
    assert receipt["status"] == "rejected_model_blind_fixture_candidate"
    assert len(receipt["checks"]) == 6
    assert all(check["viewport_video"]["frame_count"] == 451 for check in receipt["checks"])
    assert all(not check["passed"] for check in receipt["checks"])


def test_interrupted_trial_preserves_issued_action_partial_video_and_error(tmp_path):
    bridge = FakeBridge(interrupt=3)
    with pytest.raises(RuntimeError, match="injected"):
        qualify_candidate(fixture(), bridge, FakeController(), seed=4, evidence_root=tmp_path)
    path = tmp_path / "goal-+1/reset-0"
    receipt = json.loads((path / "trial.json").read_text())
    assert receipt["status"] == "infrastructure_invalid_qualification"
    assert receipt["observed_actions"] == 2
    assert receipt["viewport_video"]["frame_count"] == 3
    assert (path / "action-0003.npy").is_file()
    assert not (path / "state-0003.json").exists()
    assert bridge.environment.closed


def test_measured_proposal_flows_through_synthetic_qualification_and_real_video(tmp_path):
    source = Path(__file__).parents[1] / "artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json"
    workspace = json.loads(source.read_text())
    rows = propose_lat_layouts(workspace, seed=20260922)
    selected = next(row for row in rows if row["metadata"]["geometric_screen_status"] == "passed")
    proposal_file = tmp_path / "proposal.json"
    proposal_file.write_text(json.dumps({
        "status": "proposed_unqualified", "model_request_count": 0, "behavioral_episode_count": 0, "candidates": rows,
    }))
    candidate = _load_selected_candidate(SimpleNamespace(
        proposal_file=proposal_file, candidate_id=selected["candidate_id"],
    ))
    receipt = qualify_candidate(
        candidate, FakeBridge(candidate=candidate), FakeController(),
        seed=20260922, evidence_root=tmp_path / "synthetic-not-physical-proof",
    )
    assert receipt["status"] == "accepted_model_blind_fixture_candidate"
    assert receipt["model_request_count"] == receipt["behavioral_episode_count"] == 0
    assert len(receipt["checks"]) == 6
    for check in receipt["checks"]:
        assert check["viewport_video"]["frame_count"] == 451
        assert check["files"]["action-0450.npy"]["bytes"] > 0
        assert check["files"]["state-0450.json"]["bytes"] > 0


def test_task_definition_cannot_enable_goal_termination() -> None:
    from experiments.workshops.spatial_grounding_v1.task_definitions import RoboLabTaskDefinition

    try:
        RoboLabTaskDefinition(fixture(), goal_termination=True)
    except ValueError as error:
        assert "goal-independent" in str(error)
    else:
        raise AssertionError("task must always run the fixed 450-action cap")


def test_lat_runtime_requires_measured_table_contact() -> None:
    source = (Path(__file__).parents[1] / "experiments/workshops/spatial_grounding_v1/robolab_lat_qualification.py").read_text(encoding="utf-8")
    assert '"rubiks_cube__table"' in source
    assert "force_matrix_w" in source
    assert ">= 1.0" in source


def test_lat_task_registry_registers_only_scoped_overlay(tmp_path: Path) -> None:
    from experiments.workshops.spatial_grounding_v1.robolab_lat_qualification import register_lat_task

    task = tmp_path / "lat_qualification_task.py"
    task.write_text("# scoped task overlay\n", encoding="utf-8")
    calls = []
    register_lat_task(lambda **kwargs: calls.append(kwargs), task, cameras=("camera",))
    assert calls == [{"task": [str(task)], "cameras": ("camera",)}]


def test_physical_center_velocity_uses_geometric_center_and_euclidean_norms() -> None:
    from experiments.workshops.spatial_grounding_v1.robolab_measurements import geometric_center_state

    center, linear_speed, angular_speed = geometric_center_state(
        com_position_env_local_xyz_m=(0.4, 0.0, 0.1),
        geometric_center_env_local_xyz_m=(0.4, 0.1, 0.1),
        com_velocity_world=(0.0, 0.0, 0.0, 0.0, 0.0, 2.0),
    )

    assert center == (0.4, 0.1, 0.1)
    assert linear_speed == 0.2
    assert angular_speed == 2.0


def test_physical_center_rejects_malformed_rigid_body_measurements() -> None:
    from experiments.workshops.spatial_grounding_v1.robolab_measurements import geometric_center_state

    try:
        geometric_center_state(
            com_position_env_local_xyz_m=(0.0, 0.0, 0.0),
            geometric_center_env_local_xyz_m=(0.0, 0.0, 0.0),
            com_velocity_world=(0.0,) * 5,
        )
    except ValueError as error:
        assert "six-vector" in str(error)
    else:
        raise AssertionError("malformed rigid-body velocity must fail closed")
