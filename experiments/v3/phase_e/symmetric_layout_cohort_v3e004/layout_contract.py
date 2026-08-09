"""Pure, fail-closed layout contract for V3-E004.

No Isaac, RoboLab, policy, or checkpoint package is imported here.  This
module defines the prospective object-pose intervention and validates live
settled-state evidence supplied by a simulator gate.

The asymmetry metric is a registered, dimensionless root-sum-of-squares over
the declared symmetry constraints.  For a pair ``(a, b)`` the constraint is
``pose(a) == mirror(pose(b))``; for a midline object it is
``pose == mirror(pose)``.  Position and yaw use explicit registered inverse-
unit weights.  This definition is deliberately independent of the nominal
level ``s``: analyses must use the measured ``A`` rather than substitute
``1-s``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "vla-wam-shared-v3e004-layout-candidate-v1"
STATUS = "model_blind_candidate_not_released_for_inference"
ASYMMETRY_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
MIDLINE_TOLERANCE_M = 0.001
DEFAULT_REALISATION_POSITION_TOLERANCE_M = 0.003
DEFAULT_REALISATION_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
MINIMUM_MIRROR_PAIR_CENTER_SEPARATION_M = 0.30


class LayoutContractError(ValueError):
    """A registration or live fixture does not satisfy the E004 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayoutContractError(message)


def _finite(value: Any, label: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def wrap_angle(angle_rad: float) -> float:
    """Wrap radians to ``[-pi, pi)`` deterministically."""

    angle = _finite(angle_rad, "angle_rad")
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class PoseSE2:
    """An object's registered planar pose plus fixed height and asset identity."""

    x_m: float
    y_m: float
    z_m: float
    yaw_rad: float
    asset_identity: str

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m", "yaw_rad"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "yaw_rad", wrap_angle(self.yaw_rad))
        _require(isinstance(self.asset_identity, str) and self.asset_identity, "asset_identity must be nonempty")

    @classmethod
    def from_json(cls, value: Mapping[str, Any], label: str = "pose") -> "PoseSE2":
        _require(isinstance(value, Mapping), f"{label} must be an object")
        expected = {"x_m", "y_m", "z_m", "yaw_rad", "asset_identity"}
        _require(set(value) == expected, f"{label} keys differ: expected {sorted(expected)}")
        return cls(
            x_m=_finite(value["x_m"], f"{label}.x_m"),
            y_m=_finite(value["y_m"], f"{label}.y_m"),
            z_m=_finite(value["z_m"], f"{label}.z_m"),
            yaw_rad=_finite(value["yaw_rad"], f"{label}.yaw_rad"),
            asset_identity=str(value["asset_identity"]),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "x_m": self.x_m,
            "y_m": self.y_m,
            "z_m": self.z_m,
            "yaw_rad": self.yaw_rad,
            "asset_identity": self.asset_identity,
        }

    def mirrored(self) -> "PoseSE2":
        return PoseSE2(self.x_m, -self.y_m, self.z_m, -self.yaw_rad, self.asset_identity)


def pose_map_sha256(poses: Mapping[str, PoseSE2]) -> str:
    """Digest convention used by the s=0 control-pose attestation."""

    return hashlib.sha256(
        json.dumps(
            {name: pose.to_json() for name, pose in sorted(poses.items())},
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SymmetryWeights:
    """Registered weights that make the asymmetry metric dimensionless."""

    position_inverse_m: float
    orientation_inverse_rad: float

    def __post_init__(self) -> None:
        _require(_finite(self.position_inverse_m, "position_inverse_m") > 0.0, "position weight must be positive")
        _require(_finite(self.orientation_inverse_rad, "orientation_inverse_rad") > 0.0, "orientation weight must be positive")

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "SymmetryWeights":
        _require(isinstance(value, Mapping), "asymmetry weights must be an object")
        _require(
            set(value) == {"position_inverse_m", "orientation_inverse_rad"},
            "asymmetry weights require position_inverse_m and orientation_inverse_rad",
        )
        return cls(
            position_inverse_m=_finite(value["position_inverse_m"], "position_inverse_m"),
            orientation_inverse_rad=_finite(value["orientation_inverse_rad"], "orientation_inverse_rad"),
        )

    def to_json(self) -> dict[str, float]:
        return {
            "position_inverse_m": self.position_inverse_m,
            "orientation_inverse_rad": self.orientation_inverse_rad,
        }


def interpolate_pose(control: PoseSE2, symmetric: PoseSE2, symmetry_level_s: float) -> PoseSE2:
    """Interpolate translation and shortest-arc SO(2) orientation.

    Shortest-arc yaw interpolation is the planar equivalent of quaternion
    SLERP.  Asset identity and height must remain fixed; E004 does not
    authorize asset swaps, scale changes, or appearance interpolation.
    """

    s = _finite(symmetry_level_s, "symmetry_level_s")
    _require(0.0 <= s <= 1.0, "symmetry_level_s must be in [0, 1]")
    _require(control.asset_identity == symmetric.asset_identity, "interpolation cannot change asset identity")
    _require(math.isclose(control.z_m, symmetric.z_m, abs_tol=1e-12), "interpolation cannot change object height")
    delta_yaw = wrap_angle(symmetric.yaw_rad - control.yaw_rad)
    return PoseSE2(
        x_m=control.x_m + s * (symmetric.x_m - control.x_m),
        y_m=control.y_m + s * (symmetric.y_m - control.y_m),
        z_m=control.z_m,
        yaw_rad=wrap_angle(control.yaw_rad + s * delta_yaw),
        asset_identity=control.asset_identity,
    )


def interpolate_layout(
    control_poses: Mapping[str, PoseSE2],
    symmetric_poses: Mapping[str, PoseSE2],
    symmetry_level_s: float,
) -> dict[str, PoseSE2]:
    _require(set(control_poses) == set(symmetric_poses), "control and symmetric movable-object inventories differ")
    return {
        name: interpolate_pose(control_poses[name], symmetric_poses[name], symmetry_level_s)
        for name in sorted(control_poses)
    }


def _constraint_components(
    poses: Mapping[str, PoseSE2],
    mirror_pairs: Sequence[tuple[str, str]],
    midline_objects: Sequence[str],
) -> tuple[list[float], list[float]]:
    position: list[float] = []
    orientation: list[float] = []
    consumed: set[str] = set()
    for left, right in mirror_pairs:
        _require(left != right, "a mirror pair must contain two distinct object names")
        _require(left in poses and right in poses, f"missing mirrored pair member: {left}, {right}")
        _require(left not in consumed and right not in consumed, "an object belongs to multiple symmetry constraints")
        a, b = poses[left], poses[right]
        _require(a.asset_identity == b.asset_identity, f"mirrored pair {left}/{right} uses different assets")
        position.extend((a.x_m - b.x_m, a.y_m + b.y_m, a.z_m - b.z_m))
        orientation.append(wrap_angle(a.yaw_rad + b.yaw_rad))
        consumed.update((left, right))
    for name in midline_objects:
        _require(name in poses, f"missing midline object: {name}")
        _require(name not in consumed, f"{name} belongs to multiple symmetry constraints")
        pose = poses[name]
        # A midline object is compared to its own reflection.  X/Z cancel;
        # the factor of two is the actual pose-to-mirror distance.
        position.append(2.0 * pose.y_m)
        orientation.append(wrap_angle(2.0 * pose.yaw_rad))
        consumed.add(name)
    _require(consumed == set(poses), f"objects missing symmetry constraints: {sorted(set(poses) - consumed)}")
    return position, orientation


def asymmetry_metric(
    poses: Mapping[str, PoseSE2],
    mirror_pairs: Sequence[tuple[str, str]],
    midline_objects: Sequence[str],
    weights: SymmetryWeights,
) -> float:
    position, orientation = _constraint_components(poses, mirror_pairs, midline_objects)
    terms = [weights.position_inverse_m * value for value in position]
    terms.extend(weights.orientation_inverse_rad * value for value in orientation)
    return math.sqrt(sum(value * value for value in terms))


def symmetry_residuals(
    poses: Mapping[str, PoseSE2],
    mirror_pairs: Sequence[tuple[str, str]],
    midline_objects: Sequence[str],
) -> dict[str, float]:
    pair_position: list[float] = []
    pair_orientation: list[float] = []
    for left, right in mirror_pairs:
        _require(left in poses and right in poses, f"missing mirrored pair member: {left}, {right}")
        a, b = poses[left], poses[right]
        _require(a.asset_identity == b.asset_identity, f"mirrored pair {left}/{right} uses different assets")
        pair_position.extend((abs(a.x_m - b.x_m), abs(a.y_m + b.y_m), abs(a.z_m - b.z_m)))
        pair_orientation.append(abs(wrap_angle(a.yaw_rad + b.yaw_rad)))
    _require(midline_objects, "at least one midline object is required")
    for name in midline_objects:
        _require(name in poses, f"missing midline object: {name}")
    return {
        "position_residual_m": max(pair_position, default=0.0),
        "orientation_residual_rad": max(pair_orientation, default=0.0),
        "midline_residual_m": max(abs(poses[name].y_m) for name in midline_objects),
    }


@dataclass(frozen=True)
class E004Candidate:
    """Hashable, model-blind layout definition used by every DROID runner."""

    control_poses: Mapping[str, PoseSE2]
    symmetric_poses: Mapping[str, PoseSE2]
    mirror_pairs: tuple[tuple[str, str], ...]
    midline_objects: tuple[str, ...]
    target_object: str
    reference_object: str
    expected_cameras: tuple[str, ...]
    robot_base_xy_m: tuple[float, float]
    weights: SymmetryWeights
    s0_frozen_control_attestation: Mapping[str, Any]
    companion_counterfactual_s0_poses: Mapping[str, PoseSE2]
    orientation_invariant_objects: tuple[str, ...] = ()
    realisation_position_tolerance_m: float = DEFAULT_REALISATION_POSITION_TOLERANCE_M
    realisation_orientation_tolerance_rad: float = DEFAULT_REALISATION_ORIENTATION_TOLERANCE_RAD

    def __post_init__(self) -> None:
        control_inventory = set(self.control_poses)
        symmetric_inventory = set(self.symmetric_poses)
        companion_inventory = set(self.companion_counterfactual_s0_poses)
        orientation_invariant = set(self.orientation_invariant_objects)
        _require(control_inventory < symmetric_inventory, "E004 requires an explicit s>0 companion inventory transition")
        _require(
            symmetric_inventory - control_inventory == companion_inventory,
            "every s>0-only object needs one registered counterfactual s=0 pose",
        )
        _require(
            len(orientation_invariant) == len(self.orientation_invariant_objects)
            and orientation_invariant <= control_inventory,
            "orientation-invariant objects must be unique members of the frozen control inventory",
        )
        _require(
            _finite(self.realisation_position_tolerance_m, "realisation_position_tolerance_m") > 0.0,
            "realisation position tolerance must be positive",
        )
        _require(
            _finite(self.realisation_orientation_tolerance_rad, "realisation_orientation_tolerance_rad") > 0.0,
            "realisation orientation tolerance must be positive",
        )
        _require(self.target_object in self.midline_objects, "target must be a declared midline object")
        _require(self.reference_object in self.midline_objects, "reference must be a declared midline object")
        _require(self.target_object != self.reference_object, "target and reference must be distinct")
        _require(len(set(self.expected_cameras)) == len(self.expected_cameras) and self.expected_cameras, "expected cameras must be unique and nonempty")
        _require(len(self.robot_base_xy_m) == 2 and all(math.isfinite(v) for v in self.robot_base_xy_m), "robot base XY is invalid")
        _require(isinstance(self.s0_frozen_control_attestation, Mapping), "s0 control attestation is required")
        _require(
            self.s0_frozen_control_attestation.get("inventory_policy") == "exact_b001_inventory_and_poses",
            "s0 must be the exact hash-bound B001 inventory and poses",
        )
        _require(self.s0_frozen_control_attestation.get("inventory_transition") is True, "inventory transition must be disclosed")
        _require(
            self.s0_frozen_control_attestation.get("source_fixture_id") == "V3-B001/control",
            "s0 source fixture must be V3-B001/control",
        )
        for field in ("source_fixture_sha256", "source_queue_sha256", "control_poses_sha256"):
            digest = self.s0_frozen_control_attestation.get(field)
            _require(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest),
                f"s0 attestation {field} is invalid",
            )
        _require(
            self.s0_frozen_control_attestation.get("source_inventory") == sorted(control_inventory),
            "s0 attested inventory differs from control poses",
        )
        computed_control_digest = pose_map_sha256(self.control_poses)
        _require(
            self.s0_frozen_control_attestation.get("control_poses_sha256") == computed_control_digest,
            "s0 control pose digest does not bind the supplied poses",
        )
        _require(
            self.s0_frozen_control_attestation.get("dose_response_primary_levels") == [0.25, 0.5, 0.75, 1.0],
            "inventory-matched H3 levels must be registered",
        )
        _require(
            self.s0_frozen_control_attestation.get("s0_analysis_role") == "anchored_reference_not_in_primary_H3_slope",
            "s0 analysis role must disclose the inventory transition",
        )
        for name, pose in self.companion_counterfactual_s0_poses.items():
            _require(
                pose.asset_identity == self.symmetric_poses[name].asset_identity,
                f"companion {name} changes asset identity",
            )
        for left, right in self.mirror_pairs:
            if right in companion_inventory:
                primary = left
            elif left in companion_inventory:
                primary = right
            else:
                continue
            _require(primary in control_inventory, f"companion pair {left}/{right} has no s=0 primary")
            _require(
                self.symmetric_poses[left].asset_identity == self.symmetric_poses[right].asset_identity,
                f"companion pair {left}/{right} is not the same asset",
            )
        for level in ASYMMETRY_LEVELS[1:]:
            poses = self.layout(level)
            for left, right in self.mirror_pairs:
                distance = math.sqrt(
                    (poses[left].x_m - poses[right].x_m) ** 2
                    + (poses[left].y_m - poses[right].y_m) ** 2
                    + (poses[left].z_m - poses[right].z_m) ** 2
                )
                _require(
                    distance >= MINIMUM_MIRROR_PAIR_CENTER_SEPARATION_M,
                    f"active companion pair overlaps its registered clearance at s={level}",
                )

        # Validate the full symmetric endpoint before any candidate can be
        # serialized.  Visibility remains a live camera gate.
        residual = symmetry_residuals(self.symmetric_poses, self.mirror_pairs, self.midline_objects)
        _require(residual["position_residual_m"] < POSITION_TOLERANCE_M, "s=1 position symmetry fails")
        _require(residual["orientation_residual_rad"] < ORIENTATION_TOLERANCE_RAD, "s=1 orientation symmetry fails")
        _require(residual["midline_residual_m"] < MIDLINE_TOLERANCE_M, "s=1 midline symmetry fails")
        _require(self._target_is_closer_to_base(), "target must be in front of (closer to robot base than) reference")

    def _target_is_closer_to_base(self) -> bool:
        base_x, base_y = self.robot_base_xy_m
        cube = self.symmetric_poses[self.target_object]
        bowl = self.symmetric_poses[self.reference_object]
        target_distance = math.hypot(cube.x_m - base_x, cube.y_m - base_y)
        reference_distance = math.hypot(bowl.x_m - base_x, bowl.y_m - base_y)
        return target_distance < reference_distance

    def layout(self, symmetry_level_s: float) -> dict[str, PoseSE2]:
        s = _finite(symmetry_level_s, "symmetry_level_s")
        _require(0.0 <= s <= 1.0, "symmetry_level_s must be in [0, 1]")
        if math.isclose(s, 0.0, abs_tol=1e-12):
            # Exact preserved B001 control: the companion is absent.
            return dict(self.control_poses)
        augmented_control = dict(self.control_poses)
        augmented_control.update(self.companion_counterfactual_s0_poses)
        return interpolate_layout(augmented_control, self.symmetric_poses, s)

    def _active_constraints(
        self, poses: Mapping[str, PoseSE2]
    ) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
        active_pairs: list[tuple[str, str]] = []
        singleton_constraints = list(self.midline_objects)
        for left, right in self.mirror_pairs:
            if left in poses and right in poses:
                active_pairs.append((left, right))
            elif left in poses and right in self.companion_counterfactual_s0_poses:
                # The setwise pose-to-mirror distance for an unmatched s=0
                # asset compares the sole primary to its own reflection.
                singleton_constraints.append(left)
            elif right in poses and left in self.companion_counterfactual_s0_poses:
                singleton_constraints.append(right)
            else:
                raise LayoutContractError(f"undeclared active inventory for pair {left}/{right}")
        return tuple(active_pairs), tuple(singleton_constraints)

    def asymmetry_A(self, poses: Mapping[str, PoseSE2]) -> float:
        pairs, singletons = self._active_constraints(poses)
        return asymmetry_metric(poses, pairs, singletons, self.weights)

    def residuals(self, poses: Mapping[str, PoseSE2]) -> dict[str, float]:
        pairs, singletons = self._active_constraints(poses)
        return symmetry_residuals(poses, pairs, singletons)

    def to_json(self) -> dict[str, Any]:
        levels = {
            f"{s:.2f}": {
                "symmetry_level_s": s,
                "realised_object_poses": {name: pose.to_json() for name, pose in self.layout(s).items()},
                "asymmetry_metric_A": self.asymmetry_A(self.layout(s)),
                "inventory_transition": not math.isclose(s, 0.0, abs_tol=1e-12),
            }
            for s in ASYMMETRY_LEVELS
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "symmetry_levels": list(ASYMMETRY_LEVELS),
            "control_poses": {name: pose.to_json() for name, pose in sorted(self.control_poses.items())},
            "symmetric_poses": {name: pose.to_json() for name, pose in sorted(self.symmetric_poses.items())},
            "companion_counterfactual_s0_poses": {
                name: pose.to_json() for name, pose in sorted(self.companion_counterfactual_s0_poses.items())
            },
            "orientation_invariant_objects": list(self.orientation_invariant_objects),
            "mirror_pairs": [list(pair) for pair in self.mirror_pairs],
            "midline_objects": list(self.midline_objects),
            "target_object": self.target_object,
            "reference_object": self.reference_object,
            "expected_cameras": list(self.expected_cameras),
            "robot_base_xy_m": list(self.robot_base_xy_m),
            "asymmetry_metric": {
                "definition": "dimensionless RSS of registered pose-to-mirror constraints over all movable objects",
                "weights": self.weights.to_json(),
                "use_realised_A_not_one_minus_s": True,
            },
            "s0_frozen_control_attestation": dict(self.s0_frozen_control_attestation),
            "companion_activation_policy": {
                "inventory_transition": True,
                "s0": "companions absent; exact hash-bound B001 control inventory",
                "s_gt_0": "same-asset companions active at registered interpolated SE(2) poses",
                "counterfactual_anchor": "absent companion is anchored at its collision-free s=1 pose",
                "minimum_pair_center_separation_m": MINIMUM_MIRROR_PAIR_CENTER_SEPARATION_M,
                "undeclared_inventory_change": "fail_closed",
                "H1_s0_to_s1": "confirmatory_with_inventory-transition limitation",
                "H3_primary_levels": [0.25, 0.5, 0.75, 1.0],
            },
            "levels": levels,
            "tolerances": {
                "position_residual_m_strict_upper": POSITION_TOLERANCE_M,
                "orientation_residual_rad_strict_upper": ORIENTATION_TOLERANCE_RAD,
                "orientation_residual_deg_strict_upper": 0.5,
                "midline_residual_m_strict_upper": MIDLINE_TOLERANCE_M,
                "occlusion_check_required_false_all_cameras": True,
                "realisation_position_m": self.realisation_position_tolerance_m,
                "realisation_orientation_rad": self.realisation_orientation_tolerance_rad,
            },
            "scope_caveat": "Symmetric object layout relative to the robot midline; robot joint configuration, arm reset pose, and wrist-camera mounting are not asserted symmetric.",
        }


def build_candidate(
    *,
    control_poses: Mapping[str, PoseSE2],
    symmetric_poses: Mapping[str, PoseSE2],
    mirror_pairs: Iterable[tuple[str, str]],
    midline_objects: Iterable[str],
    target_object: str,
    reference_object: str,
    expected_cameras: Iterable[str],
    robot_base_xy_m: Sequence[float],
    weights: SymmetryWeights,
    s0_frozen_control_attestation: Mapping[str, Any],
    companion_counterfactual_s0_poses: Mapping[str, PoseSE2],
    orientation_invariant_objects: Iterable[str] = (),
    realisation_position_tolerance_m: float = DEFAULT_REALISATION_POSITION_TOLERANCE_M,
    realisation_orientation_tolerance_rad: float = DEFAULT_REALISATION_ORIENTATION_TOLERANCE_RAD,
) -> E004Candidate:
    _require(len(robot_base_xy_m) == 2, "robot_base_xy_m must contain exactly two values")
    return E004Candidate(
        control_poses=dict(control_poses),
        symmetric_poses=dict(symmetric_poses),
        mirror_pairs=tuple(tuple(pair) for pair in mirror_pairs),
        midline_objects=tuple(midline_objects),
        target_object=target_object,
        reference_object=reference_object,
        expected_cameras=tuple(expected_cameras),
        robot_base_xy_m=(_finite(robot_base_xy_m[0], "robot_base_x"), _finite(robot_base_xy_m[1], "robot_base_y")),
        weights=weights,
        s0_frozen_control_attestation=dict(s0_frozen_control_attestation),
        companion_counterfactual_s0_poses=dict(companion_counterfactual_s0_poses),
        orientation_invariant_objects=tuple(orientation_invariant_objects),
        realisation_position_tolerance_m=_finite(
            realisation_position_tolerance_m, "realisation_position_tolerance_m"
        ),
        realisation_orientation_tolerance_rad=_finite(
            realisation_orientation_tolerance_rad, "realisation_orientation_tolerance_rad"
        ),
    )


def candidate_from_json(value: Mapping[str, Any]) -> E004Candidate:
    _require(isinstance(value, Mapping), "candidate must be an object")
    _require(value.get("schema_version") == SCHEMA_VERSION, "candidate schema changed")
    _require(value.get("status") == STATUS, "candidate is not model-blind and unreleased")
    _require(value.get("model_request_count") == 0, "candidate includes model requests")
    _require(value.get("behavioral_episode_count") == 0, "candidate includes behavioral episodes")
    _require(value.get("symmetry_levels") == list(ASYMMETRY_LEVELS), "candidate symmetry levels changed")
    control = {
        str(name): PoseSE2.from_json(pose, f"control_poses.{name}")
        for name, pose in value.get("control_poses", {}).items()
    }
    symmetric = {
        str(name): PoseSE2.from_json(pose, f"symmetric_poses.{name}")
        for name, pose in value.get("symmetric_poses", {}).items()
    }
    companions = {
        str(name): PoseSE2.from_json(pose, f"companion_counterfactual_s0_poses.{name}")
        for name, pose in value.get("companion_counterfactual_s0_poses", {}).items()
    }
    metric = value.get("asymmetry_metric")
    _require(isinstance(metric, Mapping), "candidate asymmetry metric is missing")
    candidate = build_candidate(
        control_poses=control,
        symmetric_poses=symmetric,
        mirror_pairs=[tuple(pair) for pair in value.get("mirror_pairs", [])],
        midline_objects=value.get("midline_objects", []),
        target_object=str(value.get("target_object", "")),
        reference_object=str(value.get("reference_object", "")),
        expected_cameras=value.get("expected_cameras", []),
        robot_base_xy_m=value.get("robot_base_xy_m", []),
        weights=SymmetryWeights.from_json(metric.get("weights", {})),
        s0_frozen_control_attestation=value.get("s0_frozen_control_attestation", {}),
        companion_counterfactual_s0_poses=companions,
        orientation_invariant_objects=value.get("orientation_invariant_objects", []),
        realisation_position_tolerance_m=value.get("tolerances", {}).get("realisation_position_m"),
        realisation_orientation_tolerance_rad=value.get("tolerances", {}).get("realisation_orientation_rad"),
    )
    # The serialized derived layouts/A values are part of the byte-hashed
    # contract.  Recalculation on the pod may differ by one floating-point ULP
    # across Python/libm builds, so semantic verification uses a far tighter
    # tolerance than any registered physical gate rather than demanding that
    # two independent arithmetic implementations serialize identical last
    # bits.  The candidate file itself remains SHA-256 bound by load_candidate.
    expected = candidate.to_json()
    for field in (
        "levels",
        "tolerances",
        "asymmetry_metric",
        "scope_caveat",
        "s0_frozen_control_attestation",
        "companion_activation_policy",
    ):
        _require(
            _json_semantically_equal(value.get(field), expected[field]),
            f"candidate derived field changed: {field}",
        )
    return candidate


def _json_semantically_equal(left: Any, right: Any) -> bool:
    """Compare derived finite JSON, allowing only cross-libm roundoff."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=1e-14, abs_tol=1e-14)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _json_semantically_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _json_semantically_equal(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidate(path: Path, expected_sha256: str) -> E004Candidate:
    path = Path(path).resolve()
    _require(path.is_file(), f"candidate file does not exist: {path}")
    _require(
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(char in "0123456789abcdef" for char in expected_sha256),
        "expected candidate SHA-256 is invalid",
    )
    _require(sha256_file(path) == expected_sha256, "candidate SHA-256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise LayoutContractError(f"candidate is not finite UTF-8 JSON: {exc}") from exc
    return candidate_from_json(value)


def validate_arm_reset_pose(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum DROID arm-pose evidence required by E004."""

    _require(isinstance(value, Mapping), "arm reset pose must be an object")
    required = {"arm_joint_positions_rad", "gripper_position", "measurement_source_sha256"}
    _require(required <= set(value), f"arm reset pose is missing {sorted(required - set(value))}")
    joints = value["arm_joint_positions_rad"]
    _require(isinstance(joints, list) and len(joints) == 7, "arm reset pose requires seven arm joints")
    joints = [_finite(item, f"arm_joint_positions_rad[{index}]") for index, item in enumerate(joints)]
    gripper = value["gripper_position"]
    if isinstance(gripper, list):
        _require(gripper, "gripper position list cannot be empty")
        gripper_value: float | list[float] = [
            _finite(item, f"gripper_position[{index}]") for index, item in enumerate(gripper)
        ]
    else:
        gripper_value = _finite(gripper, "gripper_position")
    digest = value["measurement_source_sha256"]
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        "arm reset measurement source digest is invalid",
    )
    output = dict(value)
    output["arm_joint_positions_rad"] = joints
    output["gripper_position"] = gripper_value
    if "eef_position_robot_xyz_m" in output:
        eef = output["eef_position_robot_xyz_m"]
        _require(isinstance(eef, list) and len(eef) == 3, "EEF position must be a 3-vector")
        output["eef_position_robot_xyz_m"] = [
            _finite(item, f"eef_position_robot_xyz_m[{index}]") for index, item in enumerate(eef)
        ]
    return output


def evaluate_layout(
    candidate: E004Candidate,
    *,
    symmetry_level_s: float,
    realised_object_poses: Mapping[str, PoseSE2],
    occlusion_check_by_camera: Mapping[str, bool],
    target_visible_by_camera: Mapping[str, bool],
    arm_reset_pose: Mapping[str, Any],
    realisation_orientation_tolerance_rad: float | None = None,
) -> dict[str, Any]:
    """Validate one settled live scene and return its per-scene gate record."""

    s = _finite(symmetry_level_s, "symmetry_level_s")
    _require(any(math.isclose(s, level, abs_tol=1e-12) for level in ASYMMETRY_LEVELS), "unregistered symmetry level")
    expected = candidate.layout(s)
    orientation_tolerance = (
        candidate.realisation_orientation_tolerance_rad
        if realisation_orientation_tolerance_rad is None
        else _finite(realisation_orientation_tolerance_rad, "realisation_orientation_tolerance_rad")
    )
    _require(orientation_tolerance > 0.0, "realisation orientation tolerance must be positive")
    _require(set(realised_object_poses) == set(expected), "live movable-object inventory changed")
    for name, pose in realised_object_poses.items():
        _require(pose.asset_identity == expected[name].asset_identity, f"live asset identity changed for {name}")
    expected_cameras = set(candidate.expected_cameras)
    _require(set(occlusion_check_by_camera) == expected_cameras, "occlusion evidence camera set is incomplete")
    _require(set(target_visible_by_camera) == expected_cameras, "visibility evidence camera set is incomplete")
    _require(all(value is False for value in occlusion_check_by_camera.values()), "reference occludes target from at least one camera")
    _require(all(value is True for value in target_visible_by_camera.values()), "target is not visibly resolved in every camera")
    arm_pose = validate_arm_reset_pose(arm_reset_pose)
    for name, pose in realised_object_poses.items():
        target = expected[name]
        position_error = math.sqrt(
            (pose.x_m - target.x_m) ** 2 + (pose.y_m - target.y_m) ** 2 + (pose.z_m - target.z_m) ** 2
        )
        orientation_error = abs(wrap_angle(pose.yaw_rad - target.yaw_rad))
        _require(
            position_error <= candidate.realisation_position_tolerance_m,
            f"live pose position differs from requested s for {name}: "
            f"observed={position_error:.9g} m, "
            f"limit={candidate.realisation_position_tolerance_m:.9g} m",
        )
        if name not in candidate.orientation_invariant_objects:
            _require(
                orientation_error <= orientation_tolerance,
                f"live pose orientation differs from requested s for {name}: "
                f"observed={orientation_error:.9g} rad, "
                f"limit={orientation_tolerance:.9g} rad",
            )
    residual = candidate.residuals(realised_object_poses)
    if math.isclose(s, 1.0, abs_tol=1e-12):
        _require(residual["position_residual_m"] < POSITION_TOLERANCE_M, "live s=1 position residual fails")
        _require(residual["orientation_residual_rad"] < ORIENTATION_TOLERANCE_RAD, "live s=1 orientation residual fails")
        _require(residual["midline_residual_m"] < MIDLINE_TOLERANCE_M, "live s=1 midline residual fails")
    return {
        "symmetry_level_s": s,
        "live_orientation_realisation_tolerance_rad": orientation_tolerance,
        "asymmetry_metric_A": candidate.asymmetry_A(realised_object_poses),
        "inventory_transition": not math.isclose(s, 0.0, abs_tol=1e-12),
        **residual,
        # Exact raw-schema aliases; units are defined by the adjacent
        # explicit fields and the hash-bound candidate tolerances.
        "position_residual": residual["position_residual_m"],
        "orientation_residual": residual["orientation_residual_rad"],
        "midline_residual": residual["midline_residual_m"],
        "occlusion_check": {name: bool(occlusion_check_by_camera[name]) for name in candidate.expected_cameras},
        "target_visible": {name: bool(target_visible_by_camera[name]) for name in candidate.expected_cameras},
        "realised_object_poses": {
            name: realised_object_poses[name].to_json() for name in sorted(realised_object_poses)
        },
        "arm_reset_pose": arm_pose,
        "object_layout_symmetric_not_embodiment": True,
    }
