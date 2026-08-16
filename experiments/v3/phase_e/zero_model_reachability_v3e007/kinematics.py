"""Exact CPU kinematics for the RoboLab Franka/Robotiq USD articulation.

The module deliberately imports neither Isaac Sim nor torch.  It reads the
joint frames from the frozen USD, reconstructs the seven-joint chain, and uses
SciPy for deterministic bounded pose IK.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc


ARM_JOINT_PATHS = tuple(
    f"/panda/panda_link{index}/panda_joint{index + 1}" for index in range(7)
)
FIXED_JOINT_PATHS = (
    "/panda/panda_link7/panda_joint8",
    "/panda/panda_link8/panda_hand_joint",
)


@dataclass(frozen=True)
class JointFrame:
    path: str
    axis_xyz: tuple[float, float, float]
    lower_rad: float
    upper_rad: float
    parent_to_joint: tuple[tuple[float, ...], ...]
    child_to_joint: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class KinematicChain:
    arm_joints: tuple[JointFrame, ...]
    fixed_joints: tuple[JointFrame, ...]

    @property
    def lower(self) -> np.ndarray:
        return np.asarray([joint.lower_rad for joint in self.arm_joints], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray([joint.upper_rad for joint in self.arm_joints], dtype=np.float64)


def _matrix_tuple(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(cell) for cell in row) for row in value)


def _usd_frame(position: object, quaternion: object) -> np.ndarray:
    imaginary = quaternion.GetImaginary()
    wxyz = np.asarray(
        [quaternion.GetReal(), imaginary[0], imaginary[1], imaginary[2]],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_quat(
        [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    ).as_matrix()
    matrix[:3, 3] = np.asarray(position, dtype=np.float64)
    return matrix


def load_chain(robot_usd: Path) -> KinematicChain:
    """Load the exact articulation frames from the frozen flattened USD."""

    from pxr import Usd, UsdPhysics  # CPU-only OpenUSD bindings in the target env.

    stage = Usd.Stage.Open(str(robot_usd))
    if stage is None:
        raise RuntimeError(f"could not open robot USD: {robot_usd}")

    def parse(path: str, *, fixed: bool) -> JointFrame:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"missing registered joint prim: {path}")
        joint = UsdPhysics.Joint(prim)
        if len(joint.GetBody0Rel().GetTargets()) != 1 or len(joint.GetBody1Rel().GetTargets()) != 1:
            raise RuntimeError(f"joint body relation is not unique: {path}")
        if fixed:
            axis = (1.0, 0.0, 0.0)
            lower = upper = 0.0
        else:
            revolute = UsdPhysics.RevoluteJoint(prim)
            axis_token = str(revolute.GetAxisAttr().Get())
            axis = {
                "X": (1.0, 0.0, 0.0),
                "Y": (0.0, 1.0, 0.0),
                "Z": (0.0, 0.0, 1.0),
            }.get(axis_token)
            if axis is None:
                raise RuntimeError(f"unsupported joint axis {axis_token}: {path}")
            lower = np.deg2rad(float(revolute.GetLowerLimitAttr().Get()))
            upper = np.deg2rad(float(revolute.GetUpperLimitAttr().Get()))
        return JointFrame(
            path=path,
            axis_xyz=axis,
            lower_rad=float(lower),
            upper_rad=float(upper),
            parent_to_joint=_matrix_tuple(
                _usd_frame(joint.GetLocalPos0Attr().Get(), joint.GetLocalRot0Attr().Get())
            ),
            child_to_joint=_matrix_tuple(
                _usd_frame(joint.GetLocalPos1Attr().Get(), joint.GetLocalRot1Attr().Get())
            ),
        )

    return KinematicChain(
        arm_joints=tuple(parse(path, fixed=False) for path in ARM_JOINT_PATHS),
        fixed_joints=tuple(parse(path, fixed=True) for path in FIXED_JOINT_PATHS),
    )


def _axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = Rotation.from_rotvec(np.asarray(axis, dtype=np.float64) * angle).as_matrix()
    return matrix


def forward(chain: KinematicChain, joint_position_rad: Sequence[float]) -> np.ndarray:
    """Return the exact robot-root to Robotiq base-link transform."""

    q = np.asarray(joint_position_rad, dtype=np.float64)
    if q.shape != (7,) or not np.all(np.isfinite(q)):
        raise ValueError("joint position must be one finite seven-vector")
    transform = np.eye(4, dtype=np.float64)
    for joint, angle in zip(chain.arm_joints, q, strict=True):
        transform = (
            transform
            @ np.asarray(joint.parent_to_joint, dtype=np.float64)
            @ _axis_rotation(joint.axis_xyz, float(angle))
            @ np.linalg.inv(np.asarray(joint.child_to_joint, dtype=np.float64))
        )
    for joint in chain.fixed_joints:
        transform = (
            transform
            @ np.asarray(joint.parent_to_joint, dtype=np.float64)
            @ np.linalg.inv(np.asarray(joint.child_to_joint, dtype=np.float64))
        )
    return transform


def pose_components(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(transform[:3, 3], dtype=np.float64)
    xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return position, np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def pose_errors(
    transform: np.ndarray, target_position: Sequence[float], target_wxyz: Sequence[float]
) -> tuple[float, float]:
    target_rotation = Rotation.from_quat(
        [target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]]
    )
    actual_rotation = Rotation.from_matrix(transform[:3, :3])
    position_error = float(
        np.linalg.norm(transform[:3, 3] - np.asarray(target_position, dtype=np.float64))
    )
    orientation_error_deg = float(
        np.rad2deg(np.linalg.norm((target_rotation.inv() * actual_rotation).as_rotvec()))
    )
    return position_error, orientation_error_deg


def deterministic_starts(chain: KinematicChain, reset_q: Sequence[float], count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("start count must include reset plus at least one coverage start")
    reset = np.asarray(reset_q, dtype=np.float64)
    sampler = qmc.Halton(d=7, scramble=False)
    coverage = sampler.random(count)[1:]
    # Keep starts away from exact limits while covering the full joint box.
    coverage = 0.05 + 0.90 * coverage
    starts = chain.lower + coverage * (chain.upper - chain.lower)
    return np.vstack((reset, starts))


def _residual(
    q: np.ndarray,
    chain: KinematicChain,
    target_position: np.ndarray,
    target_rotation: Rotation,
    position_scale_m: float,
    orientation_scale_rad: float,
) -> np.ndarray:
    transform = forward(chain, q)
    actual_rotation = Rotation.from_matrix(transform[:3, :3])
    return np.concatenate(
        (
            (transform[:3, 3] - target_position) / position_scale_m,
            (target_rotation.inv() * actual_rotation).as_rotvec() / orientation_scale_rad,
        )
    )


def translational_jacobian(chain: KinematicChain, q: Sequence[float], epsilon: float = 1e-6) -> np.ndarray:
    q_array = np.asarray(q, dtype=np.float64)
    jacobian = np.empty((3, 7), dtype=np.float64)
    for index in range(7):
        plus = q_array.copy()
        minus = q_array.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        jacobian[:, index] = (
            forward(chain, plus)[:3, 3] - forward(chain, minus)[:3, 3]
        ) / (2.0 * epsilon)
    return jacobian


def solve_pose(
    chain: KinematicChain,
    *,
    target_position: Sequence[float],
    target_wxyz: Sequence[float],
    starts: Iterable[Sequence[float]],
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    max_function_evaluations: int,
) -> dict[str, object]:
    """Solve one bounded pose target and retain the best deterministic attempt."""

    target_position_array = np.asarray(target_position, dtype=np.float64)
    target_wxyz_array = np.asarray(target_wxyz, dtype=np.float64)
    target_wxyz_array /= np.linalg.norm(target_wxyz_array)
    target_rotation = Rotation.from_quat(
        [target_wxyz_array[1], target_wxyz_array[2], target_wxyz_array[3], target_wxyz_array[0]]
    )
    lower = chain.lower + 1e-9
    upper = chain.upper - 1e-9
    best: dict[str, object] | None = None
    for start_index, start in enumerate(starts):
        initial = np.clip(np.asarray(start, dtype=np.float64), lower, upper)
        solution = least_squares(
            _residual,
            initial,
            bounds=(lower, upper),
            args=(
                chain,
                target_position_array,
                target_rotation,
                position_tolerance_m,
                np.deg2rad(orientation_tolerance_deg),
            ),
            method="trf",
            max_nfev=max_function_evaluations,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        transform = forward(chain, solution.x)
        position_error, orientation_error = pose_errors(
            transform, target_position_array, target_wxyz_array
        )
        score = max(
            position_error / position_tolerance_m,
            orientation_error / orientation_tolerance_deg,
        )
        record: dict[str, object] = {
            "start_index": start_index,
            "joint_position_rad": [float(value) for value in solution.x],
            "position_error_m": position_error,
            "orientation_error_deg": orientation_error,
            "normalized_max_error": float(score),
            "function_evaluations": int(solution.nfev),
        }
        if best is None or float(record["normalized_max_error"]) < float(best["normalized_max_error"]):
            best = record
        if position_error <= position_tolerance_m and orientation_error <= orientation_tolerance_deg:
            break
    if best is None:
        raise RuntimeError("IK received no deterministic starts")
    feasible = bool(
        float(best["position_error_m"]) <= position_tolerance_m
        and float(best["orientation_error_deg"]) <= orientation_tolerance_deg
    )
    best["feasible"] = feasible
    if feasible:
        q = np.asarray(best["joint_position_rad"], dtype=np.float64)
        distance_to_limits = np.minimum(q - chain.lower, chain.upper - q)
        spans = chain.upper - chain.lower
        jacobian = translational_jacobian(chain, q)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        best["minimum_normalized_joint_limit_margin"] = float(
            np.min(distance_to_limits / spans)
        )
        best["minimum_translational_jacobian_singular_value_m_per_rad"] = float(
            np.min(singular_values)
        )
    else:
        best["minimum_normalized_joint_limit_margin"] = None
        best["minimum_translational_jacobian_singular_value_m_per_rad"] = None
    return best

