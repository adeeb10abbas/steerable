from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.workshops.spatial_grounding_v1.fixtures import FixtureCandidate, Pose
from experiments.workshops.spatial_grounding_v1.model_blind_qualification import qualify_candidate
from experiments.workshops.spatial_grounding_v1.simulator_bridge import ObjectState, ResetResult, SimulatorSnapshot


def fixture() -> FixtureCandidate:
    return FixtureCandidate.from_json({
        "candidate_id": "lat-001", "family": "LAT", "seed": 1, "task_asset": "measured.usda",
        "asset_manifest_sha256": "b" * 64,
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
    def __init__(self) -> None:
        self.goal = 1
        self.steps = 0

    def reset(self) -> ResetResult:
        self.steps = 0
        return ResetResult(
            SimulatorSnapshot(state(0.0, 0.1)),
            {
                "reset_id": f"reset-{id(self)}",
                "camera_id": "head-v1",
                "camera_name": "head_camera",
                "fingerprint": "a" * 64,
                "temporal_cache_reset": True,
            },
        )

    def step(self, _action: list[float]) -> SimulatorSnapshot:
        self.steps += 1
        if self.steps <= 3:
            return SimulatorSnapshot(state(0.0, 0.14, supported=False, attached=True))
        return SimulatorSnapshot(state(0.04 * self.goal, 0.1))

    def close(self) -> None:
        pass

    def snapshot(self) -> SimulatorSnapshot:
        return SimulatorSnapshot(state(0.0, 0.1))

    def render_viewport(self) -> bytes:
        return b"fake-viewport"


class FakeBridge:
    def create_environment(self, _task, _seed: int) -> FakeEnvironment:
        return FakeEnvironment()


class FakeController:
    def actions_for_goal(self, environment: FakeEnvironment, _candidate, goal_sign: int):
        environment.goal = goal_sign
        return [[0.0] for _ in range(4)]


def test_qualification_runs_exactly_six_model_blind_checks() -> None:
    receipt = qualify_candidate(fixture(), FakeBridge(), FakeController(), seed=4)
    assert receipt["status"] == "accepted_model_blind_fixture_candidate"
    assert receipt["model_request_count"] == 0
    assert len(receipt["checks"]) == 6
    assert {check["goal_sign"] for check in receipt["checks"]} == {-1, 1}


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
