"""Freeze the V3-E006 out-of-distribution reference before state search.

This module reads only already-completed, successful V3-E004 pi0.5 ``s=1``
rollouts.  It makes no model request and executes no simulator action.  The
resulting robust feature centres, scales, and numeric thresholds are frozen
before either canonical-state candidate is constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np


SCHEMA = "vla-wam-shared-v3e006-ood-reference-freeze-v1"
AMENDMENT_ID = "V3-E006"
MODEL_ID = "pi05_current_stack_droid"
SYMMETRY_LEVEL = 1.0
FEATURE_NAMES = (
    "arm_q0_rad",
    "arm_q1_rad",
    "arm_q2_rad",
    "arm_q3_rad",
    "arm_q4_rad",
    "arm_q5_rad",
    "arm_q6_rad",
    "cube_in_eef_x_m",
    "cube_in_eef_y_m",
    "cube_in_eef_z_m",
    "cube_in_eef_rotvec_x_rad",
    "cube_in_eef_rotvec_y_rad",
    "cube_in_eef_rotvec_z_rad",
)
SCALE_FLOORS = np.asarray([0.05] * 7 + [0.005] * 3 + [0.05] * 3, dtype=np.float64)
SOURCE_DISTANCE_QUANTILE = 0.99
MIN_REFERENCE_ROWS_PER_DIRECTION = 100


class OODReferenceError(RuntimeError):
    """The historical reference could not be frozen without ambiguity."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OODReferenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _quat_normalize_wxyz(value: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    _require(quaternion.shape == (4,) and np.isfinite(quaternion).all(), "quaternion must be finite wxyz")
    norm = float(np.linalg.norm(quaternion))
    _require(norm > 1e-12, "quaternion norm is zero")
    quaternion = quaternion / norm
    sign_key = next((float(item) for item in quaternion if abs(float(item)) > 1e-12), 1.0)
    return quaternion if sign_key > 0 else -quaternion


def _quat_inverse_wxyz(value: Sequence[float]) -> np.ndarray:
    w, x, y, z = _quat_normalize_wxyz(value)
    return np.asarray([w, -x, -y, -z], dtype=np.float64)


def _quat_multiply_wxyz(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = _quat_normalize_wxyz(left)
    w2, x2, y2, z2 = _quat_normalize_wxyz(right)
    return _quat_normalize_wxyz(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_rotate_inverse_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = _quat_normalize_wxyz(quaternion)
    w, xyz = float(q[0]), q[1:]
    inverse = -xyz
    v = np.asarray(vector, dtype=np.float64)
    return 2 * np.dot(inverse, v) * inverse + (w * w - np.dot(inverse, inverse)) * v + 2 * w * np.cross(inverse, v)


def _quat_to_rotvec_wxyz(value: Sequence[float]) -> np.ndarray:
    q = _quat_normalize_wxyz(value)
    vector = q[1:]
    magnitude = float(np.linalg.norm(vector))
    if magnitude <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(magnitude, float(q[0]))
    return vector / magnitude * angle


def state_feature(
    *,
    arm_joint_positions_rad: Sequence[float],
    eef_position_world_m: Sequence[float],
    eef_quaternion_world_wxyz: Sequence[float],
    cube_position_world_m: Sequence[float],
    cube_quaternion_world_wxyz: Sequence[float],
) -> np.ndarray:
    """Return the frozen 13-dimensional joint/relative-transform feature."""

    arm = np.asarray(arm_joint_positions_rad, dtype=np.float64)
    eef_position = np.asarray(eef_position_world_m, dtype=np.float64)
    cube_position = np.asarray(cube_position_world_m, dtype=np.float64)
    _require(arm.shape == (7,), "arm joint feature must contain exactly seven joints")
    _require(eef_position.shape == (3,) and cube_position.shape == (3,), "world positions must be 3-vectors")
    translation = _quat_rotate_inverse_wxyz(
        eef_quaternion_world_wxyz,
        cube_position - eef_position,
    )
    relative_quaternion = _quat_multiply_wxyz(
        _quat_inverse_wxyz(eef_quaternion_world_wxyz),
        cube_quaternion_world_wxyz,
    )
    feature = np.concatenate((arm, translation, _quat_to_rotvec_wxyz(relative_quaternion)))
    _require(feature.shape == (len(FEATURE_NAMES),) and np.isfinite(feature).all(), "state feature is invalid")
    return feature


def normalized_distance(feature: Sequence[float], *, center: Sequence[float], scale: Sequence[float]) -> float:
    vector = np.asarray(feature, dtype=np.float64)
    location = np.asarray(center, dtype=np.float64)
    denominator = np.asarray(scale, dtype=np.float64)
    _require(vector.shape == location.shape == denominator.shape == (len(FEATURE_NAMES),), "OOD vectors differ in shape")
    _require(np.isfinite(vector).all() and np.isfinite(location).all(), "OOD vectors must be finite")
    _require(np.isfinite(denominator).all() and np.all(denominator > 0), "OOD scales must be finite and positive")
    return float(np.sqrt(np.mean(np.square((vector - location) / denominator))))


def _load_json(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def _successful_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (
            row.get("model_id") == MODEL_ID
            and math.isclose(float(row.get("symmetry_level_s")), SYMMETRY_LEVEL, abs_tol=1e-12)
            and row.get("success") is True
            and row.get("failure_category") == "correct"
        ):
            rows.append(row)
    _require(rows, "no successful E004 pi0.5 s=1 source rows exist")
    _require(len({str(row.get("cell_id")) for row in rows}) == len(rows), "successful source cell ids are not unique")
    return sorted(rows, key=lambda row: str(row["cell_id"]))


def _first_three_grabbed_near_midline(
    steps: Sequence[Mapping[str, Any]], cube_positions: np.ndarray, offset: int
) -> int | None:
    grabbed = [bool(step.get("object_grabbed")) for step in steps]
    initial_z = float(cube_positions[offset, 2])
    for start in range(len(grabbed) - 2):
        index = start + 2
        if (
            all(grabbed[start : start + 3])
            and abs(float(cube_positions[offset + index, 1])) < 0.04
            and float(cube_positions[offset + index, 2]) - initial_z < 0.04
        ):
            return index
    return None


def _first_carry_near_midline(
    steps: Sequence[Mapping[str, Any]], cube_positions: np.ndarray, offset: int, *, after: int
) -> int | None:
    initial_z = float(cube_positions[offset, 2])
    for index in range(after + 1, len(steps)):
        if (
            bool(steps[index].get("object_grabbed"))
            and float(cube_positions[offset + index, 2]) - initial_z >= 0.04
            and abs(float(cube_positions[offset + index, 1])) < 0.04
        ):
            return index
    return None


def _hdf5_path(attempt_root: Path) -> Path:
    matches = sorted(Path(attempt_root).glob("simulator/*/run_0.hdf5"))
    _require(len(matches) == 1, f"expected one E004 HDF5 trace under {attempt_root}, found {len(matches)}")
    return matches[0].resolve()


def _source_stage_features(row: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    raw_binding = row.get("source_raw_episode")
    _require(isinstance(raw_binding, Mapping), "compact source row lacks raw-episode binding")
    raw_path = Path(str(raw_binding.get("path"))).resolve()
    _require(raw_path.is_file(), f"raw E004 episode is unavailable: {raw_path}")
    _require(raw_binding.get("bytes") == raw_path.stat().st_size, "raw E004 episode byte count changed")
    _require(raw_binding.get("sha256") == sha256_file(raw_path), "raw E004 episode hash changed")
    attempt_root = raw_path.parent
    state_capture_path = attempt_root / "state_capture" / "state_capture.json"
    _require(state_capture_path.is_file(), f"E004 state capture is unavailable: {state_capture_path}")
    state_capture = _load_json(state_capture_path)
    steps = state_capture.get("steps")
    _require(isinstance(steps, list) and len(steps) >= 4, "E004 state capture has no usable steps")
    hdf5_path = _hdf5_path(attempt_root)
    with h5py.File(hdf5_path, "r") as handle:
        root = handle["data/demo_0"]
        joint = np.asarray(root["states/articulation/robot/joint_position"], dtype=np.float64)
        cube_pose = np.asarray(root["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64)
        eef_position = np.asarray(root["ee_pose/position"], dtype=np.float64)
        eef_quaternion = np.asarray(root["ee_pose/orientation"], dtype=np.float64)
    _require(joint.ndim == 2 and joint.shape[1] >= 7, "E004 joint trace shape changed")
    _require(cube_pose.shape == (len(joint), 7), "E004 cube trace shape changed")
    _require(eef_position.shape == (len(joint), 3) and eef_quaternion.shape == (len(joint), 4), "E004 EEF trace shape changed")
    offset = len(joint) - len(steps)
    _require(offset == 74, f"E004 HDF5/state alignment changed: expected offset 74, got {offset}")
    grasp = _first_three_grabbed_near_midline(steps, cube_pose[:, :3], offset)
    stages: dict[str, np.ndarray] = {}
    source_indices: dict[str, int] = {}
    if grasp is not None:
        carry = _first_carry_near_midline(steps, cube_pose[:, :3], offset, after=grasp)
        for name, state_index in (("canonical_grasp", grasp), ("canonical_carry", carry)):
            if state_index is None:
                continue
            hdf5_index = offset + state_index
            stages[name] = state_feature(
                arm_joint_positions_rad=joint[hdf5_index, :7],
                eef_position_world_m=eef_position[hdf5_index],
                eef_quaternion_world_wxyz=eef_quaternion[hdf5_index],
                cube_position_world_m=cube_pose[hdf5_index, :3],
                cube_quaternion_world_wxyz=cube_pose[hdf5_index, 3:],
            )
            source_indices[name] = int(state_index)
    provenance = {
        "cell_id": str(row["cell_id"]),
        "requested_relation": str(row["requested_relation"]),
        "environment_seed": int(row["environment_seed"]),
        "raw_episode": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": sha256_file(raw_path),
        },
        "state_capture": {
            "path": str(state_capture_path.resolve()),
            "bytes": state_capture_path.stat().st_size,
            "sha256": sha256_file(state_capture_path),
        },
        "hdf5_trace": {
            "path": str(hdf5_path),
            "bytes": hdf5_path.stat().st_size,
            "sha256": sha256_file(hdf5_path),
        },
        "hdf5_to_state_capture_offset": offset,
        "selected_state_indices": source_indices,
    }
    return stages, provenance


def _direction_balanced_center(matrix: np.ndarray, directions: Sequence[str]) -> np.ndarray:
    by_direction = {
        relation: np.median(matrix[np.asarray([value == relation for value in directions])], axis=0)
        for relation in ("left", "right")
    }
    return (by_direction["left"] + by_direction["right"]) / 2.0


def _stage_reference(name: str, rows: list[tuple[np.ndarray, str, str]]) -> dict[str, Any]:
    _require(rows, f"no historical rows qualify for {name}")
    matrix = np.stack([row[0] for row in rows]).astype(np.float64, copy=False)
    directions = [row[1] for row in rows]
    cell_ids = [row[2] for row in rows]
    counts = {relation: directions.count(relation) for relation in ("left", "right")}
    _require(all(count >= MIN_REFERENCE_ROWS_PER_DIRECTION for count in counts.values()), f"insufficient {name} references by direction: {counts}")
    center = _direction_balanced_center(matrix, directions)
    median_absolute_deviation = np.median(np.abs(matrix - np.median(matrix, axis=0)), axis=0)
    robust_scale = np.maximum(1.4826 * median_absolute_deviation, SCALE_FLOORS)
    distances = np.asarray(
        [normalized_distance(row, center=center, scale=robust_scale) for row in matrix],
        dtype=np.float64,
    )
    threshold = float(np.quantile(distances, SOURCE_DISTANCE_QUANTILE, method="linear"))
    _require(math.isfinite(threshold) and threshold > 0, f"invalid {name} OOD threshold")
    return {
        "stage": name,
        "reference_count": len(rows),
        "reference_count_by_direction": counts,
        "reference_cell_ids": cell_ids,
        "feature_names": list(FEATURE_NAMES),
        "direction_balanced_center": center.tolist(),
        "robust_scale": robust_scale.tolist(),
        "scale_floor": SCALE_FLOORS.tolist(),
        "source_distance_summary": {
            "minimum": float(np.min(distances)),
            "median": float(np.median(distances)),
            "quantile_0_95": float(np.quantile(distances, 0.95, method="linear")),
            "quantile_0_99": threshold,
            "maximum": float(np.max(distances)),
        },
        "acceptance": {
            "metric": "sqrt(mean(square((candidate_feature - direction_balanced_center) / robust_scale)))",
            "threshold_rule": "candidate_distance <= empirical successful-E004 source-distance 0.99 quantile",
            "source_distance_quantile": SOURCE_DISTANCE_QUANTILE,
            "maximum_distance_inclusive": threshold,
            "no_post_candidate_relaxation": True,
        },
    }


def build_reference(*, e004_episodes: Path, output: Path) -> dict[str, Any]:
    e004_episodes = Path(e004_episodes).resolve()
    output = Path(output).resolve()
    _require(e004_episodes.is_file(), f"E004 compact episode file is missing: {e004_episodes}")
    _require(not output.exists(), f"refusing to overwrite OOD freeze: {output}")
    source_rows = _successful_rows(e004_episodes)
    stage_rows: dict[str, list[tuple[np.ndarray, str, str]]] = {
        "canonical_grasp": [],
        "canonical_carry": [],
    }
    provenance: list[dict[str, Any]] = []
    for row in source_rows:
        stages, source = _source_stage_features(row)
        provenance.append(source)
        for name, feature in stages.items():
            stage_rows[name].append((feature, str(row["requested_relation"]), str(row["cell_id"])))
    value: dict[str, Any] = {
        "schema_version": SCHEMA,
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": AMENDMENT_ID,
        "status": "frozen_before_any_v3e006_candidate_evaluation_or_model_request",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_scope": {
            "model_id": MODEL_ID,
            "arena": "droid_robolab",
            "symmetry_level_s": SYMMETRY_LEVEL,
            "successful_source_episode_count": len(source_rows),
            "e004_compact_episodes": {
                "path": str(e004_episodes),
                "bytes": e004_episodes.stat().st_size,
                "sha256": sha256_file(e004_episodes),
            },
        },
        "stage_selection": {
            "hdf5_to_state_capture_alignment": "len(hdf5 states)-len(state_capture steps) must equal 74",
            "canonical_grasp": "first end index of three consecutive object_grabbed samples while |cube world y|<0.04 m and cube rise<0.04 m",
            "canonical_carry": "first later object_grabbed sample with cube rise>=0.04 m while |cube world y|<0.04 m",
            "purpose": "historical comparison only; candidate construction remains deterministic simulator state/IK without a pi0.5 request",
        },
        "feature_definition": {
            "names": list(FEATURE_NAMES),
            "joint_component": "first seven Franka arm joints in radians",
            "transform_component": "cube pose expressed in the recorded EEF frame as translation metres plus shortest wxyz-quaternion rotation vector radians",
            "direction_balancing": "centre is the arithmetic midpoint of the LEFT and RIGHT coordinatewise medians",
            "scale": "max(1.4826*pooled coordinatewise MAD, preregistered coordinate floor)",
        },
        "stages": {
            name: _stage_reference(name, stage_rows[name])
            for name in ("canonical_grasp", "canonical_carry")
        },
        "source_provenance": provenance,
        "candidate_evaluation_prohibited_during_this_step": True,
    }
    unsigned = dict(value)
    value["normalized_reference_sha256"] = canonical_json_sha256(unsigned)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e004-episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_reference(e004_episodes=args.e004_episodes, output=args.output)
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(args.output.resolve()),
                "output_sha256": sha256_file(args.output),
                "successful_source_episode_count": value["source_scope"]["successful_source_episode_count"],
                "stage_reference_counts": {
                    name: stage["reference_count"] for name, stage in value["stages"].items()
                },
                "stage_thresholds": {
                    name: stage["acceptance"]["maximum_distance_inclusive"]
                    for name, stage in value["stages"].items()
                },
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
