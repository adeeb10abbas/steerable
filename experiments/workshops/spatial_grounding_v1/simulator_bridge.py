"""Narrow model-blind RoboLab/Isaac bridge used only for fixture qualification."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .fixtures import FixtureCandidate, Pose, ResetSnapshot
from .task_definitions import RoboLabTaskDefinition


class SimulatorBridgeError(RuntimeError):
    """The assigned simulator lane cannot prove a model-blind qualification."""


@dataclass(frozen=True)
class ObjectState:
    pose: Pose
    linear_speed_m_s: float
    angular_speed_rad_s: float
    supported: bool
    attached_to_gripper: bool


@dataclass(frozen=True)
class SimulatorSnapshot:
    objects: Mapping[str, ObjectState]
    simulated_time_s: float = 0.0
    reset_root_poses: Mapping[str, Pose] | None = None
    termination_reason: str | None = None
    robot_body_frames: Mapping[str, Any] | None = None
    context_measurements: Mapping[str, Any] | None = None

    def reset_snapshot(self) -> ResetSnapshot:
        if self.reset_root_poses is None:
            raise SimulatorBridgeError("snapshot lacks explicit root reset poses")
        return ResetSnapshot(self.reset_root_poses)

    def scoring_state(self, action_step: int) -> dict[str, Any]:
        cube = self.objects["rubiks_cube"]
        return {
            "action_step": action_step, "sim_time_s": self.simulated_time_s,
            "cube_xyz_m": cube.pose.position_m,
            "bowl_xyz_m": self.objects["bowl"].pose.position_m,
            "plate_xyz_m": self.objects["plate"].pose.position_m if "plate" in self.objects else None,
            "supported": cube.supported,
            "final_detached_release": not cube.attached_to_gripper,
            "gripper_holding": cube.attached_to_gripper,
            "linear_speed_m_s": cube.linear_speed_m_s,
            "angular_speed_rad_s": cube.angular_speed_rad_s,
        }


@dataclass(frozen=True)
class ResetResult:
    """One physical reset and the evidence required before a policy request."""

    snapshot: SimulatorSnapshot
    receipt: Mapping[str, Any]

    def validate_for_policy(self) -> None:
        required = {"reset_id", "camera_id", "camera_name", "fingerprint", "temporal_cache_reset"}
        missing = required.difference(self.receipt)
        if missing:
            raise SimulatorBridgeError(f"reset receipt is missing {sorted(missing)}")
        if not self.receipt["temporal_cache_reset"]:
            raise SimulatorBridgeError("reset receipt does not attest a full temporal-cache reset")
        if any(not str(self.receipt[key]) for key in ("reset_id", "camera_id", "camera_name", "fingerprint")):
            raise SimulatorBridgeError("reset receipt contains an empty stable identity")


class Environment(Protocol):
    def reset(self) -> ResetResult: ...
    def step(self, action: Sequence[float]) -> SimulatorSnapshot: ...
    def snapshot(self) -> SimulatorSnapshot: ...
    def render_viewport(self) -> Any: ...
    def close(self) -> None: ...


class ScriptedController(Protocol):
    """A deterministic controller supplied by the pinned runtime, never a model."""

    def actions_for_goal(self, environment: Environment, candidate: FixtureCandidate, goal_sign: int) -> Sequence[Sequence[float]]: ...


class SimulatorBridge(Protocol):
    def create_environment(self, task: RoboLabTaskDefinition, seed: int) -> Environment: ...


def load_factory(reference: str) -> Callable[..., Any]:
    """Load a coordinator-pinned bridge/controller factory without shelling out."""

    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise SimulatorBridgeError("factory must use module:attribute syntax")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise SimulatorBridgeError(f"factory {reference!r} is not callable")
    return value


class RoboLabIsaacBridge:
    """Runtime implementation backed by the assigned pinned RoboLab checkout.

    This deliberately demands an existing task factory.  SGW-01 cannot assume
    a scene name, object prim, controller action dimensionality, or workspace
    pose before the coordinator supplies the verified asset/runtime binding.
    """

    def __init__(self, *, task_factory: str, device: str, renderer: str, rendering_type: str) -> None:
        if renderer != "realtime" or rendering_type != "balanced":
            raise SimulatorBridgeError("qualification requires the verified realtime/balanced RTX renderer")
        self._task_factory = load_factory(task_factory)
        self._device = device
        self._renderer = renderer
        self._rendering_type = rendering_type

    def create_environment(self, task: RoboLabTaskDefinition, seed: int) -> Environment:
        value = self._task_factory(
            task=task.bridge_config(),
            seed=seed,
            device=self._device,
            renderer=self._renderer,
            rendering_type=self._rendering_type,
            model_request_count=0,
        )
        if not all(hasattr(value, name) for name in ("reset", "step", "close")):
            raise SimulatorBridgeError("pinned task factory did not return an Environment-compatible object")
        return value


def load_candidate_file(path: Path) -> FixtureCandidate:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SimulatorBridgeError(f"cannot read fixture candidate {path}") from error
    return FixtureCandidate.from_json(value)
