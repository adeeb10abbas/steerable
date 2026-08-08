"""FastWAM RoboTwin stretch contract for V3-E004.

This module is intentionally simulator- and model-free.  It binds the exact
V3-B007 control fixture, constructs a two-object layout symmetric about
RoboTwin's native lateral ``x=0`` plane, and validates the separately
registered 108-cell E004 queue.  Native RoboTwin ``-x`` is robot-left; compact
E004 evidence therefore records ``signed_lateral = -native_x``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


FASTWAM_COMMIT = "068d3fd70c89df3726c09893f47b75a624b20c02"
CHECKPOINT_SHA256 = "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"
DATASET_STATS_SHA256 = "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
SOURCE_RELEASE_QUEUE_SHA256 = "2ffe2f99e4d6c4b3d80c24fab7276b21bb83de86d92b8a3438ce38a7ba9e1ae3"
MODEL_ID = "fastwam_robotwin"
ARENA = "robotwin"
CORE_SEEDS = tuple(range(9400, 9427))
LEVELS = (0.0, 1.0)
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
EXPECTED_OBJECT = ("086_woodenblock", 1)
EXPECTED_REFERENCE = ("081_playingcards", 1)
SOURCE_FIXTURE_ENVIRONMENT_SEED = 4_300_003
CAMERAS = ("head_camera", "left_camera", "right_camera")
POSITION_TOLERANCE_M = 0.001
ORIENTATION_TOLERANCE_RAD = math.radians(0.5)
LIVE_POSITION_TOLERANCE_M = 0.003
LIVE_ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
POSITION_INVERSE_M = 10.0
ORIENTATION_INVERSE_RAD = 1.0


class FastWAME004Error(ValueError):
    """A FastWAM E004 fixture, queue, or result failed closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FastWAME004Error(message)


def wrap_angle(value: float) -> float:
    _require(math.isfinite(float(value)), "angle must be finite")
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def normalize_quaternion(value: Sequence[float]) -> tuple[float, float, float, float]:
    _require(len(value) == 4, "quaternion must be wxyz length four")
    row = tuple(float(item) for item in value)
    _require(all(math.isfinite(item) for item in row), "quaternion must be finite")
    norm = math.sqrt(sum(item * item for item in row))
    _require(norm > 0.0, "quaternion norm is zero")
    return tuple(item / norm for item in row)  # type: ignore[return-value]


def quaternion_multiply(
    a: Sequence[float], b: Sequence[float]
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = normalize_quaternion(a)
    bw, bx, by, bz = normalize_quaternion(b)
    return normalize_quaternion(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quaternion_yaw(value: Sequence[float]) -> float:
    w, x, y, z = normalize_quaternion(value)
    return wrap_angle(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def with_world_yaw(value: Sequence[float], yaw_rad: float) -> tuple[float, float, float, float]:
    """Set world-Z yaw while preserving the source pose's roll/pitch tilt."""

    source_yaw = quaternion_yaw(value)
    delta = wrap_angle(float(yaw_rad) - source_yaw)
    rotation = (math.cos(delta / 2.0), 0.0, 0.0, math.sin(delta / 2.0))
    result = quaternion_multiply(rotation, value)
    _require(abs(wrap_angle(quaternion_yaw(result) - yaw_rad)) < 1e-9, "yaw replacement failed")
    return result


@dataclass(frozen=True)
class ActorPose:
    position_xyz_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    asset_identity: tuple[str, int]

    def __post_init__(self) -> None:
        _require(len(self.position_xyz_m) == 3, "position must be xyz")
        _require(all(math.isfinite(float(item)) for item in self.position_xyz_m), "position must be finite")
        object.__setattr__(self, "position_xyz_m", tuple(float(item) for item in self.position_xyz_m))
        object.__setattr__(self, "quaternion_wxyz", normalize_quaternion(self.quaternion_wxyz))
        _require(
            isinstance(self.asset_identity, tuple)
            and len(self.asset_identity) == 2
            and isinstance(self.asset_identity[0], str)
            and self.asset_identity[0]
            and type(self.asset_identity[1]) is int,
            "asset identity must be (name, integer id)",
        )

    @property
    def yaw_rad(self) -> float:
        return quaternion_yaw(self.quaternion_wxyz)

    def to_json(self) -> dict[str, Any]:
        return {
            "position_xyz_m": list(self.position_xyz_m),
            "quaternion_wxyz": list(self.quaternion_wxyz),
            "yaw_rad": self.yaw_rad,
            "asset_identity": list(self.asset_identity),
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ActorPose":
        return cls(
            position_xyz_m=tuple(value["position_xyz_m"]),
            quaternion_wxyz=tuple(value["quaternion_wxyz"]),
            asset_identity=(str(value["asset_identity"][0]), int(value["asset_identity"][1])),
        )


CONTROL_FIXTURE = {
    "target": ActorPose(
        (-0.047076620161533356, -0.030880313366651535, 0.7405446767807007),
        (0.3491150736808777, 0.3625031113624573, 0.609960675239563, 0.6120932698249817),
        EXPECTED_OBJECT,
    ),
    "reference": ActorPose(
        (-0.21130692958831787, -0.1640346497297287, 0.7408550977706909),
        (-0.34580183029174805, -0.3450961410999298, 0.6168054938316345, 0.6171554923057556),
        EXPECTED_REFERENCE,
    ),
}


def symmetric_fixture() -> dict[str, ActorPose]:
    """Center both singletons on native x=0 and make each yaw self-mirrored."""

    output: dict[str, ActorPose] = {}
    for name, pose in CONTROL_FIXTURE.items():
        _, depth_y, height_z = pose.position_xyz_m
        output[name] = ActorPose(
            (0.0, depth_y, height_z),
            with_world_yaw(pose.quaternion_wxyz, 0.0),
            pose.asset_identity,
        )
    return output


SYMMETRIC_FIXTURE = symmetric_fixture()


def residuals(layout: Mapping[str, ActorPose]) -> dict[str, float]:
    _require(set(layout) == {"target", "reference"}, "fixture must contain exactly one target and one reference")
    _require(layout["target"].asset_identity == EXPECTED_OBJECT, "target asset drift")
    _require(layout["reference"].asset_identity == EXPECTED_REFERENCE, "reference asset drift")
    return {
        # There are no paired clutter actors in pair03.  Both movable actors
        # are self-mirrored singletons on the source-x midline.
        "position_residual_m": 0.0,
        "orientation_residual_rad": max(abs(wrap_angle(2.0 * pose.yaw_rad)) for pose in layout.values()),
        "midline_residual_m": max(abs(pose.position_xyz_m[0]) for pose in layout.values()),
    }


def asymmetry_A(layout: Mapping[str, ActorPose]) -> float:
    _require(set(layout) == {"target", "reference"}, "fixture inventory drift")
    terms: list[float] = []
    for pose in layout.values():
        terms.append(POSITION_INVERSE_M * (2.0 * pose.position_xyz_m[0]))
        terms.append(ORIENTATION_INVERSE_RAD * wrap_angle(2.0 * pose.yaw_rad))
    return math.sqrt(sum(value * value for value in terms))


def layout_for_level(level: float) -> dict[str, ActorPose]:
    if math.isclose(float(level), 0.0, abs_tol=1e-12):
        return dict(CONTROL_FIXTURE)
    if math.isclose(float(level), 1.0, abs_tol=1e-12):
        return dict(SYMMETRIC_FIXTURE)
    raise FastWAME004Error("FastWAM stretch authorizes only s=0 and s=1")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def canonicalize_candidate_floats(value: Any) -> Any:
    """Remove cross-libm last-bit drift before hashing the FastWAM fixture."""

    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        _require(math.isfinite(value), "FastWAM candidate contains a non-finite float")
        return float(format(value, ".15g"))
    if isinstance(value, dict):
        return {key: canonicalize_candidate_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonicalize_candidate_floats(item) for item in value]
    raise FastWAME004Error(f"unsupported FastWAM candidate value: {type(value).__name__}")


def candidate_payload() -> dict[str, Any]:
    payload = {
        "schema_version": "vla-wam-shared-v3e004-fastwam-robotwin-layout-candidate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "model_blind_candidate_not_released_for_inference",
        "source_release": {
            "amendment_id": "V3-B007",
            "fastwam_commit": FASTWAM_COMMIT,
            "source_queue_sha256": SOURCE_RELEASE_QUEUE_SHA256,
            "source_fixture_environment_seed": SOURCE_FIXTURE_ENVIRONMENT_SEED,
        },
        "coordinate_contract": {
            "native_lateral_axis": "source_x",
            "native_left": "negative_x",
            "compact_signed_lateral": "negative_source_x",
            "sagittal_plane": "source_x_equals_zero",
            "source_y_and_z_held_fixed": True,
        },
        "inventory": {
            "target_count": 1,
            "reference_count": 1,
            "no_duplicated_reference": True,
            "target_identity": list(EXPECTED_OBJECT),
            "reference_identity": list(EXPECTED_REFERENCE),
            "mirrored_clutter_pairs": [],
        },
        "layouts": {
            "0.00": {name: pose.to_json() for name, pose in sorted(CONTROL_FIXTURE.items())},
            "1.00": {name: pose.to_json() for name, pose in sorted(SYMMETRIC_FIXTURE.items())},
        },
        "derived": {
            level: {
                "symmetry_level_s": float(level),
                "asymmetry_metric_A": asymmetry_A(layout_for_level(float(level))),
                **residuals(layout_for_level(float(level))),
            }
            for level in ("0.00", "1.00")
        },
        "asymmetry_metric": {
            "definition": "dimensionless RSS of each singleton pose minus its source-x reflection",
            "position_inverse_m": POSITION_INVERSE_M,
            "orientation_inverse_rad": ORIENTATION_INVERSE_RAD,
            "use_realised_A": True,
        },
        "s1_tolerances": {
            "position_residual_m_strict_upper": POSITION_TOLERANCE_M,
            "orientation_residual_rad_strict_upper": ORIENTATION_TOLERANCE_RAD,
            "midline_residual_m_strict_upper": POSITION_TOLERANCE_M,
            "occlusion_false_for_all_cameras": list(CAMERAS),
        },
        "matched_seeds": list(CORE_SEEDS),
        "exact_prompts": PROMPTS,
        "scope_caveat": "This is symmetric object placement about RoboTwin source x=0, not a symmetric robot, reset configuration, camera rig, or embodiment.",
    }
    full = residuals(SYMMETRIC_FIXTURE)
    _require(full["midline_residual_m"] < POSITION_TOLERANCE_M, "s1 midline residual fails")
    _require(full["orientation_residual_rad"] < ORIENTATION_TOLERANCE_RAD, "s1 yaw residual fails")
    return canonicalize_candidate_floats(payload)


def candidate_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(candidate_payload())).hexdigest()


def load_candidate(path: Path, expected_sha256: str) -> dict[str, Any]:
    payload = Path(path).read_bytes()
    _require(hashlib.sha256(payload).hexdigest() == expected_sha256, "FastWAM candidate SHA-256 mismatch")
    value = json.loads(payload, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    _require(value == candidate_payload(), "FastWAM candidate semantic reconstruction changed")
    return value


def validate_registered_queue(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if row.get("model_id") == MODEL_ID]
    _require(len(selected) == 108, "FastWAM E004 queue must contain 108 cells")
    _require(len({row.get("cell_id") for row in selected}) == 108, "FastWAM queue cell ids are not unique")
    observed: set[tuple[int, float, str]] = set()
    for row in selected:
        seed = int(row.get("environment_seed"))
        level = float(row.get("symmetry_level_s"))
        relation = str(row.get("relation"))
        _require(seed in CORE_SEEDS, "FastWAM queue uses a non-E004 seed")
        _require(level in LEVELS, "FastWAM queue uses an unauthorized symmetry level")
        _require(relation in RELATIONS, "FastWAM queue relation changed")
        _require(row.get("sampling_seed") == seed, "FastWAM environment/sampling seed mismatch")
        _require(row.get("prompt") == PROMPTS[relation], "FastWAM exact prompt bytes changed")
        _require(row.get("arena") == ARENA, "FastWAM arena boundary changed")
        _require(row.get("success_predicate_id") == "frozen_v3b007_robotwin_relation_aware_success", "FastWAM predicate changed")
        observed.add((seed, level, relation))
    expected = {(seed, level, relation) for seed in CORE_SEEDS for level in LEVELS for relation in RELATIONS}
    _require(observed == expected, "FastWAM queue coverage is incomplete")
    return selected
