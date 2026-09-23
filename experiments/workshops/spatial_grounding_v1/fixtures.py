"""Model-blind SGW-01 fixture contracts and deterministic selection.

Candidate geometry is supplied by a measured RoboLab asset manifest.  This
module intentionally does not synthesize poses: inventing a workspace pose
would make a candidate look qualified before it has ever existed in Isaac.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


FAMILIES = frozenset(("LAT", "HEIGHT", "DIST"))
STAGE_LAYOUTS = {"P": 1, "D": 4, "C": 24}
MAX_CANDIDATES_PER_FAMILY = 100
ACTION_CAP = 450
NEUTRAL_TOLERANCE_M = 0.005
GOAL_MARGIN_M = 0.03
RESET_POSITION_TOLERANCE_M = 0.003
RESET_ANGLE_TOLERANCE_DEGREES = 2.0
REFERENCE_MOTION_LIMIT_M = 0.005


class FixtureError(ValueError):
    """A candidate or qualification receipt is scientifically invalid."""


@dataclass(frozen=True)
class Pose:
    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "Pose":
        position = tuple(float(item) for item in value["position_m"])
        quaternion = tuple(float(item) for item in value["quaternion_wxyz"])
        if len(position) != 3 or len(quaternion) != 4:
            raise FixtureError("poses require three position and four quaternion components")
        if not all(math.isfinite(item) for item in position + quaternion):
            raise FixtureError("poses must be finite")
        norm = math.sqrt(sum(item * item for item in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-3):
            raise FixtureError("pose quaternion is not normalized")
        return cls(position, quaternion)


@dataclass(frozen=True)
class FixtureCandidate:
    candidate_id: str
    family: str
    seed: int
    asset_manifest_sha256: str
    task_asset: str
    object_poses: Mapping[str, Pose]
    metadata: Mapping[str, Any]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "FixtureCandidate":
        family = str(value["family"])
        if family not in FAMILIES:
            raise FixtureError(f"unsupported relation family: {family}")
        poses = {str(name): Pose.from_json(pose) for name, pose in value["object_poses"].items()}
        required = {"rubiks_cube", "bowl"}
        if family == "DIST":
            required.add("plate")
        if set(poses) != required:
            raise FixtureError(f"{family} candidate must contain exactly {sorted(required)}")
        asset_hash = str(value["asset_manifest_sha256"])
        if len(asset_hash) != 64 or any(char not in "0123456789abcdef" for char in asset_hash):
            raise FixtureError("candidate must bind an immutable asset-manifest SHA-256")
        candidate = cls(
            candidate_id=str(value["candidate_id"]),
            family=family,
            seed=int(value["seed"]),
            asset_manifest_sha256=asset_hash,
            task_asset=str(value["task_asset"]),
            object_poses=poses,
            metadata=dict(value.get("metadata", {})),
        )
        if not candidate.candidate_id or not candidate.task_asset:
            raise FixtureError("candidate ID and actual task asset are required")
        candidate.validate_neutral_start()
        return candidate

    def scoring_poses(self, roots: Mapping[str, Pose] | None = None) -> Mapping[str, Pose]:
        roots = self.object_poses if roots is None else roots
        offsets = self.metadata.get("scoring_center_offsets_root_local_m")
        if not isinstance(offsets, Mapping) or set(offsets) != set(roots):
            raise FixtureError("candidate lacks measured scoring-center offsets for every object")
        result = {}
        for name, root in roots.items():
            offset = tuple(float(value) for value in offsets[name])
            if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
                raise FixtureError("scoring-center offset must be a finite three-vector")
            result[name] = Pose(
                _add(root.position_m, _rotate(root.quaternion_wxyz, offset)),
                root.quaternion_wxyz,
            )
        return result

    def relation_m(self, poses: Mapping[str, Pose] | None = None) -> float:
        positions = self.scoring_poses(poses)
        cube = positions["rubiks_cube"].position_m
        bowl = positions["bowl"].position_m
        if self.family == "LAT":
            return cube[1] - bowl[1]
        if self.family == "HEIGHT":
            return cube[2] - bowl[2]
        plate = positions["plate"].position_m
        return _distance(cube, plate) - _distance(cube, bowl)

    def validate_neutral_start(self) -> None:
        if abs(self.relation_m()) > NEUTRAL_TOLERANCE_M:
            raise FixtureError("candidate is not neutral within 5 mm")
        if self.metadata.get("historical_layout_fingerprint"):
            raise FixtureError("candidate duplicates a historical layout")

    def fingerprint(self) -> str:
        value = {
            "family": self.family,
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "task_asset": self.task_asset,
            "object_poses": {name: asdict(pose) for name, pose in sorted(self.object_poses.items())},
            "scoring_center_offsets_root_local_m": self.metadata.get("scoring_center_offsets_root_local_m"),
        }
        return sha256(_canonical_json(value)).hexdigest()

    def task_payload(self) -> dict[str, Any]:
        payload = {
            "schema_version": "sgw-01-task-definition-v1",
            "candidate_id": self.candidate_id,
            "family": self.family,
            "task_asset": self.task_asset,
            "asset_manifest_sha256": self.asset_manifest_sha256,
            "object_poses": {name: asdict(pose) for name, pose in self.object_poses.items()},
            "action_cap": ACTION_CAP,
            "goal_termination": False,
            "model_request_count": 0,
        }
        native_scene = self.metadata.get("native_scene")
        if native_scene is not None:
            payload["native_scene"] = native_scene
        return payload


@dataclass(frozen=True)
class ResetSnapshot:
    poses: Mapping[str, Pose]

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> "ResetSnapshot":
        return cls({str(name): Pose.from_json(pose) for name, pose in value["poses"].items()})


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _distance(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


def _add(left: tuple[float, float, float], right: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _rotate(quaternion: tuple[float, float, float, float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    vx, vy, vz = vector
    return (
        (1 - 2 * (y*y + z*z))*vx + 2*(x*y - z*w)*vy + 2*(x*z + y*w)*vz,
        2*(x*y + z*w)*vx + (1 - 2*(x*x + z*z))*vy + 2*(y*z - x*w)*vz,
        2*(x*z - y*w)*vx + 2*(y*z + x*w)*vy + (1 - 2*(x*x + y*y))*vz,
    )


def pose_error(observed: Pose, expected: Pose) -> tuple[float, float]:
    position_error = max(abs(a - b) for a, b in zip(observed.position_m, expected.position_m, strict=True))
    dot = abs(sum(a * b for a, b in zip(observed.quaternion_wxyz, expected.quaternion_wxyz, strict=True)))
    angle_degrees = math.degrees(2.0 * math.acos(min(1.0, dot)))
    return position_error, angle_degrees


def validate_reset(candidate: FixtureCandidate, snapshots: Iterable[ResetSnapshot]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    snapshots = list(snapshots)
    if len(snapshots) != 3:
        raise FixtureError("each physical goal requires exactly three deterministic reset checks")
    for index, snapshot in enumerate(snapshots):
        if set(snapshot.poses) != set(candidate.object_poses):
            raise FixtureError("reset snapshot object inventory differs from candidate")
        for name, expected in candidate.object_poses.items():
            position_error, angle_error = pose_error(snapshot.poses[name], expected)
            if position_error > RESET_POSITION_TOLERANCE_M or angle_error > RESET_ANGLE_TOLERANCE_DEGREES:
                raise FixtureError(f"reset {index} for {name} exceeds 3 mm / 2 degree tolerance")
            rows.append({"repeat": index, "object": name, "position_error_m": position_error, "angle_error_degrees": angle_error})
        if abs(candidate.relation_m(snapshot.poses)) > NEUTRAL_TOLERANCE_M:
            raise FixtureError(f"reset {index} is not neutral")
    return rows


def candidate_order(seed: int, candidates: Iterable[FixtureCandidate]) -> list[FixtureCandidate]:
    values = list(candidates)
    if len(values) > MAX_CANDIDATES_PER_FAMILY:
        raise FixtureError("candidate generator exceeded the frozen 100-candidate family maximum")
    if len({candidate.candidate_id for candidate in values}) != len(values):
        raise FixtureError("candidate IDs must be unique")
    for index, first in enumerate(values):
        for second in values[index + 1:]:
            if _duplicate_layout(first, second):
                raise FixtureError("candidates duplicate a layout within the reset tolerance")
    return sorted(values, key=lambda item: sha256(f"{seed}|{item.fingerprint()}".encode()).hexdigest())


def _duplicate_layout(first: FixtureCandidate, second: FixtureCandidate) -> bool:
    if first.family != second.family or set(first.object_poses) != set(second.object_poses):
        return False
    return all(
        pose_error(first.object_poses[name], second.object_poses[name])[0] <= RESET_POSITION_TOLERANCE_M
        and pose_error(first.object_poses[name], second.object_poses[name])[1] <= RESET_ANGLE_TOLERANCE_DEGREES
        for name in first.object_poses
    )


def select_qualified_layouts(
    family: str, seed: int, candidates: Iterable[FixtureCandidate], accepted_candidate_ids: set[str]
) -> dict[str, str]:
    """Assign P01, D01–D04, and C01–C24 by frozen hash order.

    HEIGHT and DIST must carry a binary counterbalance label supplied from the
    live fixture construction.  The pilot uses its seeded side; development
    uses two per side and confirmation uses twelve per side.
    """

    ordered = [
        candidate for candidate in candidate_order(seed, candidates)
        if candidate.family == family and candidate.candidate_id in accepted_candidate_ids
    ]
    if family in {"HEIGHT", "DIST"}:
        key = "upper_support_side" if family == "HEIGHT" else "bowl_side"
        sides = {"left", "right"}
        if any(candidate.metadata.get(key) not in sides for candidate in ordered):
            raise FixtureError(f"{family} candidates require measured {key} values")
        pilot_side = ("left", "right")[seed % 2]
        groups = {side: [candidate for candidate in ordered if candidate.metadata[key] == side] for side in sides}
        required_per_side = 14
        if any(len(groups[side]) < required_per_side for side in sides):
            raise FixtureError(f"{family} lacks enough accepted candidates for the frozen counterbalance")
        selected = [groups[pilot_side].pop(0)]
        for stage, per_side in (("D", 2), ("C", 12)):
            for side in ("left", "right"):
                selected.extend(groups[side][:per_side])
                del groups[side][:per_side]
    else:
        if len(ordered) < 29:
            raise FixtureError(f"{family} has only {len(ordered)} accepted candidates; 29 are required")
        selected = ordered[:29]
    labels = ["P01"] + [f"D{index:02d}" for index in range(1, 5)] + [f"C{index:02d}" for index in range(1, 25)]
    return {f"{family}-{label}": candidate.candidate_id for label, candidate in zip(labels, selected, strict=True)}


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(value))
