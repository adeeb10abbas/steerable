"""Initial-pose-anchored kinematic reference motion for V4 interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from experiments.online_correction_v4.motion import ReferenceMotionController


class RootKinematicWriter(Protocol):
    """Minimal IsaacLab asset surface used for privileged reference translation."""

    def write_root_pose_to_sim(self, pose: Sequence[float]) -> None:
        ...

    def write_root_velocity_to_sim(self, velocity: Sequence[float]) -> None:
        ...


@dataclass
class KinematicReferenceMotion:
    """Apply reference offsets from a frozen initial root pose without cumulative drift."""

    writer: RootKinematicWriter
    motion_controller: ReferenceMotionController
    direction: tuple[float, float]
    initial_pose: tuple[float, ...] | None = None
    write_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = field(default_factory=list)

    def anchor_initial_pose(self, pose: Sequence[float]) -> None:
        if len(pose) < 7:
            raise ValueError("initial pose must contain xyz and quaternion wxyz")
        if self.initial_pose is not None:
            raise RuntimeError("initial pose is already anchored")
        self.initial_pose = tuple(float(item) for item in pose)

    @property
    def is_anchored(self) -> bool:
        return self.initial_pose is not None

    def displacement_at(self, sim_time_s: float) -> tuple[float, float, float]:
        dx, dy = self.motion_controller.world_offset(sim_time_s, direction=self.direction)
        return (dx, dy, 0.0)

    def pose_at(self, sim_time_s: float) -> tuple[float, ...]:
        if self.initial_pose is None:
            raise RuntimeError("initial pose must be anchored before applying motion")
        x0, y0, z0 = self.initial_pose[:3]
        qw, qx, qy, qz = self.initial_pose[3:7]
        dx, dy, dz = self.displacement_at(sim_time_s)
        return (x0 + dx, y0 + dy, z0 + dz, qw, qx, qy, qz)

    def apply_at(self, sim_time_s: float) -> tuple[float, ...]:
        pose = self.pose_at(sim_time_s)
        velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.writer.write_root_pose_to_sim(pose)
        self.writer.write_root_velocity_to_sim(velocity)
        self.write_calls.append((pose, velocity))
        return pose

    def freeze_at(self, sim_time_s: float, *, reason: str) -> tuple[float, ...]:
        self.motion_controller.freeze_at(sim_time_s, reason=reason)
        return self.apply_at(sim_time_s)
