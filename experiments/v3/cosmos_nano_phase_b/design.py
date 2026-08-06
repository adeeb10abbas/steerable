"""Fail-closed builder for the Cosmos3 Nano position-reflection release.

This module deliberately imports neither Isaac/RoboLab nor a model package.  It
turns a *completed model-blind calibration report* into a prospective amendment
and matched-cell registry.  A malformed, dirty, behavioral, or incompletely
gated calibration report releases nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


STUDY_ID = "vla_wam_language_steerability_v3"
PHASE = "B_confound_ablation"
AMENDMENT_ID = "V3-B001"
MODEL_ID = "cosmos3_nano_policy_droid"
MODEL_REPOSITORY = "nvidia/Cosmos3-Nano-Policy-DROID"
MODEL_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
COSMOS_REPOSITORY_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
ROBOLAB_REPOSITORY_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
CALIBRATION_STUDY_COMMIT = "eb18135ec86a848167c54ee0c01f267d89a8f423"
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
SCENE_ASSET = "rubiks_cube_banana_bowl.usda"
SUCCESS_PREDICATE_ID = (
    "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
)
CALIBRATION_SCHEMA = (
    "vla-wam-shared-v3b-nano-position-mirror-model-blind-calibration-v1"
)
CANDIDATE_SCHEMA = "vla-wam-shared-v3b-nano-position-mirror-candidate-v1"
AMENDMENT_SCHEMA = "vla-wam-shared-v3b-nano-mirror-amendment-v1"
CELL_SCHEMA = "vla-wam-shared-v3b-nano-mirror-cell-v1"
MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-mirror-manifest-v1"
RANDOMIZATION_NAMESPACE = (
    "vla_wam_language_steerability_v3:V3-B001:"
    "cosmos3_nano:movable_object_center_position_reflection:"
    "block_hash_randomization_v1"
)
FACTOR_NAME = "movable_object_center_position_reflection_about_robot_sagittal_plane"

# These are the exact neutral DROID/RoboLab scene coordinates inherited from
# the frozen task.  The intervention reflects every movable object across the
# robot sagittal plane y=0 and changes no non-position scene state.
CONTROL_POSITIONS_WORLD_XYZ: dict[str, tuple[float, float, float]] = {
    "banana": (0.538878858089447, -0.07555567473173141, 0.0684281587600708),
    "bowl": (0.44258353114128113, 0.12658219039440155, 0.07732785493135452),
    "rubiks_cube": (
        0.303364634513855,
        0.12396888434886932,
        0.08113233000040054,
    ),
}

SOURCE_BINDINGS = {
    "artifacts/vla_wam_shared_v3/confound_fixture_calibration_registry.json": (
        "47fe840bb0cfc343a09657c25132bd4da1dfe4bf6f2a5bad639bd0424189ffda"
    ),
    "artifacts/vla_wam_shared_v3/measurement_coverage_audit.json": (
        "86b3ded424bba2d697b406ea7b1c8bfc147dba8413363ae0190b4d151e76d4ea"
    ),
    "artifacts/vla_wam_shared_v3/protocol.json": (
        "0e1a6465c96178e0c768c9398fe003c6617456b5101cfe8ce068283a8a7572d2"
    ),
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py": (
        "9c4d90be770266bac3ba5242b743098348c565ee622179b5e88fa2af0c4891bc"
    ),
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py": (
        "ba0eb879590960c57976dd1b749c4ebbd3e86054e152ca5af014ac1bc2b6d02a"
    ),
    "experiments/v3/cosmos_nano_phase_b/fixture_candidate.py": (
        "a5523d4a9ac72843ee5f28357a5025bf604c1a06c639b1791cb836b7790b54ca"
    ),
    "experiments/v3/cosmos_nano_phase_b/fixture_tasks.py": (
        "c2a9c10ffad7c51910448f70e7b4957f6ede75f27eb761e54224a2144d456148"
    ),
    "experiments/v3/cosmos_nano_phase_b/model_blind_fixture_gate.py": (
        "c533f04116dbed356e3c87c0329bea0043c4c4c9df8aa1bcd0dd4983e34c816f"
    ),
    "experiments/v3/cosmos_nano_phase_b/task_files/control_left.py": (
        "13caf30e102c2e342d8c6d8eb4d9f5f5fe5145f5509316f5c437715e401551ef"
    ),
    "experiments/v3/cosmos_nano_phase_b/task_files/control_right.py": (
        "3293ec9f7ea46b5d81d859d7e0628199273e77acec89bc9e99c3577fd23ea53f"
    ),
    "experiments/v3/cosmos_nano_phase_b/task_files/position_mirrored_left.py": (
        "94547e8d7c979199aad30f2659f495685fda77ba0a544546ab9dd3ccac0b3a32"
    ),
    "experiments/v3/cosmos_nano_phase_b/task_files/position_mirrored_right.py": (
        "f340db0f4585bede492b1cecc69efd7c5490ef24b20f16581562317155f3b55d"
    ),
}

OUTPUT_FILENAMES = {
    "amendment": "post_result_nano_mirror_v3b001_amendment.json",
    "cells": "nano_mirror_v3b001_cells.jsonl",
    "manifest": "nano_mirror_v3b001_manifest.json",
}

REPORT_ARMS = ARMS
TASK_LABELS = tuple(
    f"{arm}_{relation}" for arm in REPORT_ARMS for relation in RELATIONS
)
TASK_WRAPPER_PATHS = {
    label: f"experiments/v3/cosmos_nano_phase_b/task_files/{label}.py"
    for label in TASK_LABELS
}


class ReleaseError(ValueError):
    """Raised before any output is written when a release gate fails."""


@dataclass(frozen=True)
class ReleasePayloads:
    amendment: dict[str, Any]
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    amendment_bytes: bytes
    cells_bytes: bytes
    manifest_bytes: bytes


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReleaseError(f"value is not canonical finite JSON: {exc}") from exc


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    try:
        return b"".join(
            (
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
    except (TypeError, ValueError) as exc:
        raise ReleaseError(f"row is not canonical finite JSON: {exc}") from exc


def _reject_constant(value: str) -> None:
    raise ReleaseError(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseError(f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def load_calibration_report(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ReleaseError(f"cannot read calibration report {path}: {exc}") from exc
    try:
        report = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"calibration report is not valid UTF-8 JSON: {exc}") from exc
    require(isinstance(report, dict), "calibration report must be a JSON object")
    return report, payload


def _keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} must be an object")
    observed = set(value)
    require(
        observed == expected,
        f"{label} keys differ: missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )
    return value


def _sha(value: Any, label: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase SHA-256 digest",
    )
    return value


def _number(value: Any, label: str) -> float:
    require(
        type(value) in (int, float) and math.isfinite(float(value)),
        f"{label} must be finite",
    )
    return float(value)


def _positive(value: Any, label: str) -> float:
    result = _number(value, label)
    require(result > 0.0, f"{label} must be positive")
    return result


def _timestamp(value: Any, label: str) -> str:
    require(isinstance(value, str) and value.endswith("Z"), f"{label} must end in Z")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseError(f"{label} must be an RFC3339 timestamp") from exc
    return value


def mirrored_positions() -> dict[str, list[float]]:
    return {
        name: [position[0], -position[1], position[2]]
        for name, position in CONTROL_POSITIONS_WORLD_XYZ.items()
    }


def control_positions() -> dict[str, list[float]]:
    return {name: list(position) for name, position in CONTROL_POSITIONS_WORLD_XYZ.items()}


def fixture_sha256(fixture: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(fixture))


def verify_source_bindings(repo_root: Path, reported: Any) -> dict[str, str]:
    reported = _keys(reported, set(SOURCE_BINDINGS), "source_bindings")
    require(reported == SOURCE_BINDINGS, "calibration source bindings changed")
    observed: dict[str, str] = {}
    for relative, expected in SOURCE_BINDINGS.items():
        path = Path(repo_root) / relative
        require(path.is_file(), f"missing release-bound source: {relative}")
        observed[relative] = sha256_file(path)
        require(observed[relative] == expected, f"release-bound source changed: {relative}")
    return observed


def _vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} must have length {length}")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _file_record(value: Any, label: str, *, extras: set[str] | None = None) -> Path:
    expected = {"path", "sha256", "bytes"} | (extras or set())
    record = _keys(value, expected, label)
    require(isinstance(record["path"], str) and record["path"], f"{label}.path is empty")
    path = Path(record["path"])
    require(path.is_absolute(), f"{label}.path must be absolute")
    require(path.is_file(), f"{label}.path does not persist after calibration: {path}")
    expected_hash = _sha(record["sha256"], f"{label}.sha256")
    require(type(record["bytes"]) is int and record["bytes"] > 0, f"{label}.bytes must be positive")
    require(path.stat().st_size == record["bytes"], f"{label} byte count changed")
    require(sha256_file(path) == expected_hash, f"{label} hash changed after calibration")
    return path


def _validate_repo_source_record(
    value: Any,
    label: str,
    repo_root: Path,
    expected_relative: str,
) -> Path:
    recorded = _file_record(value, label)
    expected = repo_root / expected_relative
    require(expected.is_file(), f"missing committed calibration source: {expected_relative}")
    require(
        sha256_file(recorded) == sha256_file(expected) == SOURCE_BINDINGS[expected_relative],
        f"{label} differs from the release-bound committed source",
    )
    return recorded


def _validate_candidate(candidate: Any) -> dict[str, Any]:
    candidate = _keys(
        candidate,
        {
            "analytic_neutrality_precheck",
            "behavioral_episode_count",
            "exact_prompts",
            "factor",
            "layouts",
            "model_id",
            "model_request_count",
            "phase",
            "release_boundary",
            "schema_version",
            "source_identity",
            "status",
            "study_id",
        },
        "fixture candidate",
    )
    expected = {
        "schema_version": CANDIDATE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "phase": PHASE,
        "status": "model_blind_candidate_not_released_for_inference",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "exact_prompts": PROMPTS,
    }
    for key, value in expected.items():
        require(candidate[key] == value, f"fixture candidate {key} changed")
    source = _keys(
        candidate["source_identity"],
        {"neutral_left_task", "neutral_right_task", "robolab_commit", "scene", "scene_metadata"},
        "fixture candidate.source_identity",
    )
    require(source["robolab_commit"] == ROBOLAB_REPOSITORY_COMMIT, "candidate RoboLab commit changed")
    require(source["scene"] == SCENE_ASSET, "candidate scene changed")
    _file_record(source["scene_metadata"], "fixture candidate source scene_metadata")
    for key, relative in (
        ("neutral_left_task", "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"),
        ("neutral_right_task", "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py"),
    ):
        path = _file_record(source[key], f"fixture candidate source {key}")
        require(
            sha256_file(path) == SOURCE_BINDINGS[relative],
            f"fixture candidate {key} is not the frozen neutral task",
        )

    factor = _keys(
        candidate["factor"],
        {"changed", "claim_boundary", "held_fixed", "name", "robot_base_plane_y_m", "transform"},
        "fixture candidate.factor",
    )
    require(
        factor["name"] == FACTOR_NAME
        and factor["transform"]
        == "for rubiks_cube, bowl, and banana only: (x,y,z) -> (x,-y,z)"
        and _number(factor["robot_base_plane_y_m"], "candidate sagittal plane") == 0.0,
        "candidate one-factor transform changed",
    )
    require(
        factor["changed"] == "initial center positions of the three movable objects"
        and isinstance(factor["held_fixed"], list)
        and "object quaternions" in factor["held_fixed"]
        and "prompt bytes and scorer axes" in factor["held_fixed"],
        "candidate held-fixed contract changed",
    )

    layouts = _keys(candidate["layouts"], set(REPORT_ARMS), "fixture candidate.layouts")
    expected_positions = {
        "control": control_positions(),
        "position_mirrored": mirrored_positions(),
    }
    rotations: dict[str, dict[str, list[float]]] = {}
    for arm in REPORT_ARMS:
        layout = _keys(
            layouts[arm],
            {"positions_robot_base_m", "quaternions_wxyz_unchanged"},
            f"fixture candidate.layouts.{arm}",
        )
        positions = _keys(
            layout["positions_robot_base_m"],
            set(CONTROL_POSITIONS_WORLD_XYZ),
            f"fixture candidate {arm} positions",
        )
        for name, expected_position in expected_positions[arm].items():
            require(
                _vector(positions[name], 3, f"candidate {arm}.{name}") == expected_position,
                f"candidate {arm}.{name} position changed",
            )
        rotations[arm] = _keys(
            layout["quaternions_wxyz_unchanged"],
            set(CONTROL_POSITIONS_WORLD_XYZ),
            f"fixture candidate {arm} quaternions",
        )
        for name, quaternion in rotations[arm].items():
            values = _vector(quaternion, 4, f"candidate {arm}.{name} quaternion")
            require(
                math.isclose(sum(item * item for item in values), 1.0, rel_tol=0.0, abs_tol=1e-4),
                f"candidate {arm}.{name} quaternion is not normalized",
            )
    require(rotations["control"] == rotations["position_mirrored"], "candidate changed object quaternions")

    analytic = _keys(
        candidate["analytic_neutrality_precheck"],
        {
            "control_cube_minus_bowl_xy_m",
            "live_simulator_reset_check_still_required",
            "outside_left_and_right_45deg_cones",
            "position_mirrored_cube_minus_bowl_xy_m",
        },
        "fixture candidate.analytic_neutrality_precheck",
    )
    control_delta = [
        control_positions()["rubiks_cube"][index] - control_positions()["bowl"][index]
        for index in range(2)
    ]
    mirror_delta = [
        mirrored_positions()["rubiks_cube"][index] - mirrored_positions()["bowl"][index]
        for index in range(2)
    ]
    require(
        _vector(analytic["control_cube_minus_bowl_xy_m"], 2, "control analytic delta") == control_delta
        and _vector(analytic["position_mirrored_cube_minus_bowl_xy_m"], 2, "mirror analytic delta") == mirror_delta
        and analytic["outside_left_and_right_45deg_cones"] is True
        and analytic["live_simulator_reset_check_still_required"] is True,
        "candidate neutrality precheck changed",
    )
    return candidate


def _load_candidate_from_report(report: Mapping[str, Any]) -> dict[str, Any]:
    path = _file_record(report["candidate"], "calibration candidate")
    candidate, payload = load_calibration_report(path)
    require(
        sha256_bytes(payload) == report["candidate"]["sha256"],
        "candidate payload does not match calibration record",
    )
    return _validate_candidate(candidate)


def _validate_live_tasks(report: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    reset_gate = _keys(
        report["reset_gate"],
        {
            "angular_speed_tolerance_basis",
            "angular_speed_tolerance_rad_s",
            "left_right_physical_fingerprints_equal_within_each_arm",
            "linear_speed_tolerance_m_s",
            "live_position_reflection_passed_at_every_repeat",
            "neither_predicate_true_at_every_reset",
            "initial_quaternion_sources_identical_across_layouts",
            "position_tolerance_m",
            "post_settle_quaternion_differences_recorded_not_gated",
            "repeat_count_per_task",
            "settle_steps",
            "settle_steps_basis",
            "stable_window_steps",
        },
        "calibration reset_gate",
    )
    require(
        type(reset_gate["repeat_count_per_task"]) is int
        and reset_gate["repeat_count_per_task"] == 3,
        "calibration requires exactly three resets per task",
    )
    require(
        type(reset_gate["settle_steps"]) is int and reset_gate["settle_steps"] == 60,
        "settle window must be exactly 60 steps",
    )
    require(
        reset_gate["settle_steps_basis"]
        == (
            "The model-blind 60-step probe reduced movable-object translation below "
            "0.004 m/s while preserving a neutral reset; a longer 180-step probe was "
            "rejected after free settling crossed a task termination boundary"
        ),
        "settle-window basis changed",
    )
    position_tolerance = _number(reset_gate["position_tolerance_m"], "position tolerance")
    linear_tolerance = _number(reset_gate["linear_speed_tolerance_m_s"], "linear speed tolerance")
    angular_tolerance = _number(reset_gate["angular_speed_tolerance_rad_s"], "angular speed tolerance")
    require(
        position_tolerance == 0.003
        and linear_tolerance == 0.02
        and angular_tolerance == 0.20,
        "calibration tolerances changed",
    )
    require(
        reset_gate["angular_speed_tolerance_basis"]
        == (
            "0.02 m/s linear tolerance divided by a conservative 0.10 m object "
            "radius gives 0.20 rad/s, bounding rotational surface speed at the "
            "same scale as translation"
        ),
        "angular stability tolerance basis changed",
    )
    require(
        type(reset_gate["stable_window_steps"]) is int
        and reset_gate["stable_window_steps"] == 15,
        "stability window must be exactly 15 consecutive steps",
    )
    for key in (
        "left_right_physical_fingerprints_equal_within_each_arm",
        "neither_predicate_true_at_every_reset",
        "live_position_reflection_passed_at_every_repeat",
        "initial_quaternion_sources_identical_across_layouts",
        "post_settle_quaternion_differences_recorded_not_gated",
    ):
        require(reset_gate[key] is True, f"reset gate did not pass: {key}")

    tasks = report["tasks"]
    require(isinstance(tasks, list) and len(tasks) == 4, "calibration must contain four task records")
    by_label: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        task = _keys(task, {"arm", "label", "prompt", "relation", "repeat_resets", "task_name"}, "calibration task")
        label = task["label"]
        require(label in TASK_LABELS and label not in by_label, f"unexpected or duplicate task label: {label}")
        arm, relation = label.rsplit("_", 1)
        require(task["arm"] == arm and task["relation"] == relation, f"task identity changed for {label}")
        require(task["prompt"] == PROMPTS[relation], f"prompt changed for {label}")
        expected_task_name = "V3BNano" + "".join(part.title() for part in label.split("_")) + "CalibrationTask"
        require(task["task_name"] == expected_task_name, f"task class changed for {label}")
        repeats = task["repeat_resets"]
        require(
            isinstance(repeats, list) and len(repeats) == reset_gate["repeat_count_per_task"],
            f"reset count changed for {label}",
        )
        expected_layout = candidate["layouts"][arm]
        for repeat_index, row in enumerate(repeats):
            row = _keys(
                row,
                {
                    "input_views",
                    "left_predicate_at_reset",
                    "positions_robot_base_m",
                    "quaternions_wxyz",
                    "repeat",
                    "right_predicate_at_reset",
                    "stability_window",
                    "velocities",
                },
                f"{label} reset {repeat_index}",
            )
            require(row["repeat"] == repeat_index, f"repeat index changed for {label}")
            require(row["left_predicate_at_reset"] is False, f"{label} reset starts LEFT")
            require(row["right_predicate_at_reset"] is False, f"{label} reset starts RIGHT")
            positions = _keys(row["positions_robot_base_m"], set(CONTROL_POSITIONS_WORLD_XYZ), f"{label} positions")
            quaternions = _keys(row["quaternions_wxyz"], set(CONTROL_POSITIONS_WORLD_XYZ), f"{label} quaternions")
            velocities = _keys(row["velocities"], set(CONTROL_POSITIONS_WORLD_XYZ), f"{label} velocities")
            stability = _keys(row["stability_window"], set(CONTROL_POSITIONS_WORLD_XYZ), f"{label} stability window")
            for name in CONTROL_POSITIONS_WORLD_XYZ:
                live_position = _vector(positions[name], 3, f"{label}.{name} position")
                intended_position = expected_layout["positions_robot_base_m"][name]
                require(
                    max(abs(left - right) for left, right in zip(live_position, intended_position))
                    <= position_tolerance,
                    f"{label}.{name} live position missed candidate tolerance",
                )
                live_quaternion = _vector(quaternions[name], 4, f"{label}.{name} quaternion")
                require(
                    math.isclose(
                        sum(item * item for item in live_quaternion),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=1e-4,
                    ),
                    f"{label}.{name} settled quaternion is not normalized",
                )
                _vector(velocities[name], 6, f"{label}.{name} final velocity")
                maxima = _keys(
                    stability[name],
                    {"max_angular_speed_rad_s", "max_linear_speed_m_s"},
                    f"{label}.{name} stability maxima",
                )
                require(
                    0.0
                    <= _number(maxima["max_linear_speed_m_s"], f"{label}.{name} max linear speed")
                    <= linear_tolerance,
                    f"{label}.{name} sustained linear stability failed",
                )
                require(
                    0.0
                    <= _number(maxima["max_angular_speed_rad_s"], f"{label}.{name} max angular speed")
                    <= angular_tolerance,
                    f"{label}.{name} sustained angular stability failed",
                )
            views = _keys(
                row["input_views"],
                {"head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"},
                f"{label} RGB views",
            )
            for view_name, view in views.items():
                view = _keys(view, {"dtype", "pixel_range", "shape"}, f"{label}.{view_name}")
                shape = view["shape"]
                require(
                    isinstance(shape, list)
                    and len(shape) == 3
                    and all(type(item) is int and item > 0 for item in shape)
                    and shape[-1] == 3,
                    f"{label}.{view_name} RGB shape is invalid",
                )
                require(view["dtype"] == "uint8" and type(view["pixel_range"]) is int and view["pixel_range"] > 0, f"{label}.{view_name} is blank")
        by_label[label] = task
    require(set(by_label) == set(TASK_LABELS), "calibration task inventory is incomplete")

    for arm in REPORT_ARMS:
        left = by_label[f"{arm}_left"]["repeat_resets"]
        right = by_label[f"{arm}_right"]["repeat_resets"]
        for left_row, right_row in zip(left, right):
            require(
                left_row["positions_robot_base_m"] == right_row["positions_robot_base_m"]
                and left_row["quaternions_wxyz"] == right_row["quaternions_wxyz"],
                f"{arm} LEFT/RIGHT reset fingerprints differ",
            )
    control_repeats = by_label["control_left"]["repeat_resets"]
    mirrored_repeats = by_label["position_mirrored_left"]["repeat_resets"]
    for repeat_index, (control_row, mirrored_row) in enumerate(
        zip(control_repeats, mirrored_repeats)
    ):
        for name in CONTROL_POSITIONS_WORLD_XYZ:
            control_position = control_row["positions_robot_base_m"][name]
            mirrored_position = mirrored_row["positions_robot_base_m"][name]
            require(
                max(
                    abs(left - right)
                    for left, right in zip(
                        [control_position[0], -control_position[1], control_position[2]],
                        mirrored_position,
                    )
                )
                <= position_tolerance,
                f"repeat {repeat_index} live position reflection failed for {name}",
            )
    diagnostics = report["post_settle_cross_layout_quaternion_differences"]
    require(
        isinstance(diagnostics, list) and len(diagnostics) == len(control_repeats),
        "post-settle quaternion diagnostics are incomplete",
    )
    for repeat_index, diagnostic in enumerate(diagnostics):
        diagnostic = _keys(
            diagnostic,
            {"objects", "repeat"},
            f"post-settle quaternion diagnostic {repeat_index}",
        )
        require(diagnostic["repeat"] == repeat_index, "quaternion diagnostic repeat changed")
        objects = _keys(
            diagnostic["objects"],
            set(CONTROL_POSITIONS_WORLD_XYZ),
            f"quaternion diagnostic {repeat_index}.objects",
        )
        for name, recorded in objects.items():
            recorded = _keys(
                recorded,
                {
                    "absolute_quaternion_dot",
                    "angular_distance_rad",
                    "max_abs_component_difference",
                },
                f"quaternion diagnostic {repeat_index}.{name}",
            )
            control_quaternion = control_repeats[repeat_index]["quaternions_wxyz"][name]
            mirrored_quaternion = mirrored_repeats[repeat_index]["quaternions_wxyz"][name]
            dot = min(
                1.0,
                abs(
                    sum(
                        left * right
                        for left, right in zip(control_quaternion, mirrored_quaternion)
                    )
                ),
            )
            expected_component_difference = max(
                abs(left - right)
                for left, right in zip(control_quaternion, mirrored_quaternion)
            )
            observed_component_difference = _number(
                recorded["max_abs_component_difference"],
                "quaternion diagnostic max_abs_component_difference",
            )
            require(
                math.isclose(
                    observed_component_difference,
                    expected_component_difference,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ),
                "post-settle quaternion diagnostic mismatch: "
                f"{repeat_index}.{name}.max_abs_component_difference",
            )

            observed_dot = _number(
                recorded["absolute_quaternion_dot"],
                "quaternion diagnostic absolute_quaternion_dot",
            )
            require(
                0.0 <= observed_dot <= 1.0
                and math.isclose(observed_dot, dot, rel_tol=0.0, abs_tol=1e-15),
                "post-settle quaternion diagnostic mismatch: "
                f"{repeat_index}.{name}.absolute_quaternion_dot",
            )

            observed_angle = _number(
                recorded["angular_distance_rad"],
                "quaternion diagnostic angular_distance_rad",
            )
            # A one-ULP dot-product difference near |dot|=1 is valid serializer/runtime
            # roundoff, but inverse cosine amplifies it.  Bind the recorded dot to the
            # source quaternions above, then bind the angle to that recorded dot.
            expected_angle = 2.0 * math.acos(observed_dot)
            require(
                math.isclose(
                    observed_angle,
                    expected_angle,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
                "post-settle quaternion diagnostic mismatch: "
                f"{repeat_index}.{name}.angular_distance_rad",
            )


def _validate_report_shape(report: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    report = _keys(
        report,
        {
            "behavioral_episode_count",
            "calibration_driver_source",
            "candidate",
            "claim_boundary",
            "environment_seed",
            "factor_task_source",
            "factor_task_wrappers",
            "gpu_query",
            "gpu_uuid",
            "model_id",
            "model_request_count",
            "passed",
            "phase",
            "pod",
            "pod_uid",
            "post_settle_cross_layout_quaternion_differences",
            "renderer",
            "reset_gate",
            "robolab",
            "schema_version",
            "status",
            "study_checkout",
            "study_id",
            "tasks",
            "viewport_write_gate",
        },
        "calibration report",
    )
    expected = {
        "schema_version": CALIBRATION_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "phase": PHASE,
        "status": "complete_model_blind_calibration_not_yet_released",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "environment_seed": 9400,
    }
    for key, value in expected.items():
        require(report[key] == value, f"calibration {key} changed")
    require(isinstance(report["pod"], str) and "ali" in report["pod"], "calibration pod is not ali-owned")
    require(isinstance(report["pod_uid"], str) and report["pod_uid"], "calibration pod UID is missing")
    require(isinstance(report["gpu_uuid"], str) and report["gpu_uuid"], "calibration GPU UUID is missing")
    require(
        isinstance(report["gpu_query"], str) and report["gpu_uuid"] in report["gpu_query"],
        "calibration GPU query does not bind the assigned UUID",
    )
    verify_source_bindings(repo_root, SOURCE_BINDINGS)
    _validate_repo_source_record(
        report["calibration_driver_source"],
        "calibration driver",
        repo_root,
        "experiments/v3/cosmos_nano_phase_b/model_blind_fixture_gate.py",
    )
    _validate_repo_source_record(
        report["factor_task_source"],
        "factor task helper",
        repo_root,
        "experiments/v3/cosmos_nano_phase_b/fixture_tasks.py",
    )
    wrappers = _keys(report["factor_task_wrappers"], set(TASK_LABELS), "factor_task_wrappers")
    for label, relative in TASK_WRAPPER_PATHS.items():
        _validate_repo_source_record(wrappers[label], f"factor wrapper {label}", repo_root, relative)

    study_checkout = _keys(report["study_checkout"], {"commit", "tracked_diff_empty"}, "study_checkout")
    require(
        study_checkout["commit"] == CALIBRATION_STUDY_COMMIT
        and study_checkout["tracked_diff_empty"] is True,
        "calibration study checkout was not the clean source-identity commit",
    )
    robolab = _keys(report["robolab"], {"commit", "effective_import", "tracked_diff_empty", "versions"}, "robolab")
    require(
        robolab["commit"] == ROBOLAB_REPOSITORY_COMMIT
        and robolab["tracked_diff_empty"] is True,
        "calibration did not use the clean exact RoboLab commit",
    )
    _file_record(robolab["effective_import"], "effective RoboLab import")
    versions = _keys(robolab["versions"], {"isaaclab", "isaacsim", "robolab"}, "RoboLab versions")
    require(all(isinstance(value, str) and value for value in versions.values()), "RoboLab versions are incomplete")

    renderer = _keys(
        report["renderer"],
        {"all_required_rgb_views_nonblank", "backend", "nvidia_icd", "quality"},
        "renderer",
    )
    require(
        renderer["backend"] == "realtime RTX Vulkan"
        and renderer["quality"] == "balanced"
        and renderer["all_required_rgb_views_nonblank"] is True,
        "RTX/Vulkan renderer gate did not pass",
    )
    _file_record(renderer["nvidia_icd"], "NVIDIA Vulkan ICD")

    candidate = _load_candidate_from_report(report)
    _validate_live_tasks(report, candidate)
    videos = _keys(report["viewport_write_gate"], set(TASK_LABELS), "viewport_write_gate")
    repeats = report["reset_gate"]["repeat_count_per_task"]
    for label, record in videos.items():
        _file_record(record, f"viewport writer {label}", extras={"decoded_frame_count"})
        require(
            type(record["decoded_frame_count"]) is int
            and record["decoded_frame_count"] == repeats,
            f"viewport writer {label} did not retain every reset frame",
        )
    require(
        report["claim_boundary"]
        == (
            "Model-blind calibration of a positions-only movable-object reflection. "
            "Initial quaternion sources are identical, while any recorded post-settle "
            "orientation difference is a downstream physical consequence of the position "
            "intervention. It is not behavioral evidence, a full scene mirror, or a "
            "reachability claim."
        ),
        "calibration claim boundary changed",
    )
    return report


def validate_calibration_report(report: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Validate every persisted model-blind release gate."""

    return _validate_report_shape(report, Path(repo_root).resolve())


def _seed_values(value: Any, parent_key: str = "") -> set[int]:
    result: set[int] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            result.update(_seed_values(child, key))
    elif isinstance(value, list):
        for child in value:
            result.update(_seed_values(child, parent_key))
    elif "seed" in parent_key.lower() and type(value) is int:
        result.add(value)
    return result


def verify_seed_range_unused(repo_root: Path) -> dict[str, list[int]]:
    """Reject overlap with every frozen v3 behavioral seed registry."""

    root = Path(repo_root)
    targets = set(SEEDS)
    observed: dict[str, list[int]] = {}
    queue_path = root / "artifacts/vla_wam_shared_v3/phase_a_cells.jsonl"
    require(queue_path.is_file(), "missing Phase-A seed registry")
    phase_a: set[int] = set()
    for line_number, line in enumerate(queue_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReleaseError(f"invalid Phase-A queue line {line_number}: {exc}") from exc
        phase_a.update(
            value
            for key, value in row.items()
            if "seed" in key and type(value) is int
        )
    observed[str(queue_path.relative_to(root))] = sorted(phase_a & targets)

    for relative in (
        "artifacts/vla_wam_shared_v3/four_phrasings_registry.json",
        "artifacts/vla_wam_shared_v3/stochastic_rollout_registry.json",
    ):
        path = root / relative
        require(path.is_file(), f"missing seed registry: {relative}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ReleaseError(f"invalid seed registry {relative}: {exc}") from exc
        observed[relative] = sorted(_seed_values(payload) & targets)
    collisions = {path: values for path, values in observed.items() if values}
    require(not collisions, f"Phase-B seeds are not unused: {collisions}")
    return observed


def _rank(*parts: object) -> str:
    joined = "\x1f".join(str(part) for part in (RANDOMIZATION_NAMESPACE, *parts))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def randomized_orders() -> dict[int, tuple[tuple[str, str], ...]]:
    """Return outcome-independent, position-balanced orders for all 27 seeds.

    Six complete four-seed blocks use all cyclic rotations of a hash-ranked
    base order.  The final three-seed block uses three rotations.  Consequently
    every condition occurs six or seven times at every ordinal position.
    """

    conditions = tuple((arm, relation) for arm in ARMS for relation in RELATIONS)
    orders: dict[int, tuple[tuple[str, str], ...]] = {}
    for block_index, start in enumerate(range(0, len(SEEDS), 4)):
        block_seeds = SEEDS[start : start + 4]
        base = tuple(
            sorted(
                conditions,
                key=lambda item: _rank(block_index, "base", item[0], item[1]),
            )
        )
        rotations = tuple(base[offset:] + base[:offset] for offset in range(4))
        rotations = tuple(
            sorted(
                rotations,
                key=lambda order: _rank(
                    block_index,
                    "rotation",
                    *(f"{arm}:{relation}" for arm, relation in order),
                ),
            )
        )
        assigned_seeds = tuple(sorted(block_seeds, key=lambda seed: _rank(block_index, "seed", seed)))
        for seed, order in zip(assigned_seeds, rotations):
            orders[seed] = order
    require(set(orders) == set(SEEDS), "randomization did not assign every seed")
    position_counts: Counter[tuple[int, tuple[str, str]]] = Counter()
    for order in orders.values():
        require(set(order) == set(conditions) and len(order) == 4, "randomized block is incomplete")
        for position, condition in enumerate(order, 1):
            position_counts[(position, condition)] += 1
    require(
        set(position_counts.values()) <= {6, 7},
        "randomized execution order is not position-balanced",
    )
    return orders


def _analysis_plan() -> dict[str, Any]:
    return {
        "coordinate": {
            "symbol": "s",
            "definition": "signed_final_lateral_offset_m; positive is robot LEFT",
            "availability": "required for every valid episode, including failures",
        },
        "full_sample_primary": {
            "population": "all seeds with four valid behavioral cells; infrastructure-invalid attempts are repaired at the identical cell and remain outside the denominator",
            "per_arm_steering_separation": "D[a,i] = s[a,i,left] - s[a,i,right]",
            "interpretation": "positive D means the prompt change ordered endpoints LEFT-to-RIGHT",
            "directional_bias_contrast": "B[a,i] = (-s[a,i,right]) - s[a,i,left]",
            "interpretation_of_bias": "positive B means greater requested-side depth for RIGHT than LEFT",
            "position_reflection_interaction": "I[i] = B[position_mirrored,i] - B[control,i]",
            "secondary_redirection_interaction": "J[i] = D[position_mirrored,i] - D[control,i]",
            "missingness": "no imputation; valid behavioral failures remain included",
        },
        "success_conditional_secondary": {
            "field": "final_requested_signed_margin_m",
            "complete_case_subset_id": "nano_v3b001_all_four_cells_correct",
            "inclusion_rule": "a seed is included only when all four cells—control LEFT, control RIGHT, position-mirrored LEFT, and position-mirrored RIGHT—satisfy the frozen requested-success predicate",
            "per_arm_margin_gap": "G[a,i] = margin[a,i,right] - margin[a,i,left] = (-s[a,i,right]) - s[a,i,left]",
            "position_reflection_interaction": "G[position_mirrored,i] - G[control,i]",
            "reporting": "name the subset and its realized n; do not use failures as zero margins or mix unmatched successful cells",
        },
        "claim_boundary": "The position-reflection interaction tests whether directional bias changes under the registered movable-object center-position reflection. Robot, cameras, and nonmovable geometry remain fixed, so this is neither a full-scene symmetry test nor causal attribution to training data.",
    }


def _release_fixtures(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    arm_map = {"control": "control", "position_mirrored": "position_mirrored"}
    fixtures: dict[str, dict[str, Any]] = {}
    for release_arm, candidate_arm in arm_map.items():
        layout = candidate["layouts"][candidate_arm]
        nonposition = {
            "scene_asset": SCENE_ASSET,
            "quaternions_wxyz": layout["quaternions_wxyz_unchanged"],
            "held_fixed": candidate["factor"]["held_fixed"],
        }
        fixtures[release_arm] = {
            "fixture_id": f"v3b001_nano_{release_arm}",
            "candidate_layout": candidate_arm,
            "scene_asset": SCENE_ASSET,
            "frame_of_reference": "robot",
            "sagittal_plane_y_m": 0.0,
            "positions_world_xyz": layout["positions_robot_base_m"],
            "quaternions_wxyz": layout["quaternions_wxyz_unchanged"],
            "nonposition_state_sha256": sha256_bytes(canonical_json_bytes(nonposition)),
        }
    require(
        fixtures["control"]["nonposition_state_sha256"]
        == fixtures["position_mirrored"]["nonposition_state_sha256"],
        "release fixtures changed a non-position factor",
    )
    return fixtures


def _relation_geometry(candidate: Mapping[str, Any]) -> dict[str, Any]:
    analytic = candidate["analytic_neutrality_precheck"]
    clearances: dict[str, float] = {}
    for arm, key in (
        ("control", "control_cube_minus_bowl_xy_m"),
        ("position_mirrored", "position_mirrored_cube_minus_bowl_xy_m"),
    ):
        delta_x, delta_y = analytic[key]
        clearances[arm] = abs(delta_x) - abs(delta_y)
        require(clearances[arm] > 0.0, f"{arm} reset is not neutral to both 45-degree cones")
    return {
        "success_predicate_id": SUCCESS_PREDICATE_ID,
        "cone_half_angle_degrees": 45.0,
        "sustained_samples": 3,
        "requested_side_margin_definition": "+signed_final_lateral_offset_m for LEFT; -signed_final_lateral_offset_m for RIGHT",
        "opposite_side_margin_definition": "the negative of requested-side margin",
        "neutral_reset_45deg_cone_exclusion_clearance_m": clearances,
        "calibration_selection_rule": "exact analytic fixture geometry plus repeated live model-blind neutrality checks before model inference",
    }


def _build_rows(fixtures: Mapping[str, Any], amendment_sha256: str) -> tuple[dict[str, Any], ...]:
    orders = randomized_orders()
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        block_id = f"v3b001:nano:seed{seed}"
        for order_index, (arm, relation) in enumerate(orders[seed], 1):
            fixture = fixtures[arm]
            cell_id = f"{block_id}:{arm}:{relation}"
            rows.append(
                {
                    "schema_version": CELL_SCHEMA,
                    "study_id": STUDY_ID,
                    "amendment_id": AMENDMENT_ID,
                    "amendment_sha256": amendment_sha256,
                    "phase": PHASE,
                    "arena": "droid_robolab",
                    "model_id": MODEL_ID,
                    "cell_id": cell_id,
                    "matched_block_id": block_id,
                    "arm": arm,
                    "relation": relation,
                    "environment_seed": seed,
                    "sampling_seed": seed,
                    "execution_order_index_within_seed": order_index,
                    "randomization_key_sha256": _rank(seed, arm, relation, order_index),
                    "factor": FACTOR_NAME,
                    "fixture_id": fixture["fixture_id"],
                    "fixture_sha256": fixture_sha256(fixture),
                    "prompt_family": "direct_command",
                    "prompt": PROMPTS[relation],
                    "prompt_sha256": sha256_bytes(PROMPTS[relation].encode("utf-8")),
                    "success_predicate_id": SUCCESS_PREDICATE_ID,
                    "runtime_identity_requirement": {
                        "model_repository": MODEL_REPOSITORY,
                        "checkpoint_revision": MODEL_REVISION,
                        "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
                        "robolab_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
                        "clean_external_repositories_required": True,
                    },
                    "required_raw_outputs": [
                        "viewport_video",
                        "executed_action_trace",
                        "raw_result_jsonl",
                        "every_exposed_decoded_future",
                    ],
                    "required_episode_fields": {
                        "signed_final_lateral_offset_m": "required_finite_for_every_valid_episode_including_failures",
                        "final_requested_signed_margin_m": "required_finite_for_every_valid_episode; +s for LEFT and -s for RIGHT; analyzed only on the named success-complete subset",
                        "requested_success": "required_boolean_from_frozen_scorer",
                        "failure_class": "correct|pick_failed|transport_failed|wrong_side|release_failed",
                    },
                    "valid_failure_policy": "retain every valid failure in the full-sample signed-offset analysis",
                    "technical_invalidity_policy": "retain separately and repair only this identical registered cell",
                    "execution_status": "authorized_after_v3b001_calibration_with_live_identity_and_output_gate_recheck",
                }
            )
    require(len(rows) == 108, "Nano position-reflection registry must contain exactly 108 cells")
    return tuple(rows)


def _position_balance(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    counts: Counter[tuple[str, int]] = Counter()
    for row in rows:
        condition = f"{row['arm']}:{row['relation']}"
        counts[(condition, row["execution_order_index_within_seed"])] += 1
    return {
        condition: {str(position): counts[(condition, position)] for position in range(1, 5)}
        for condition in (f"{arm}:{relation}" for arm in ARMS for relation in RELATIONS)
    }


def build_release(
    repo_root: Path,
    calibration_report_path: Path,
    *,
    recorded_at_utc: str,
) -> ReleasePayloads:
    """Build, but do not write, a released amendment, queue, and manifest."""

    root = Path(repo_root).resolve()
    recorded_at_utc = _timestamp(recorded_at_utc, "release recorded_at_utc")
    report, calibration_bytes = load_calibration_report(calibration_report_path)
    validate_calibration_report(report, root)
    candidate = _load_candidate_from_report(report)
    fixtures = _release_fixtures(candidate)
    relation_geometry = _relation_geometry(candidate)
    seed_audit = verify_seed_range_unused(root)
    calibration_record = {
        "path": Path(calibration_report_path).name,
        "bytes": len(calibration_bytes),
        "sha256": sha256_bytes(calibration_bytes),
    }
    amendment = {
        "schema_version": AMENDMENT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "phase": PHASE,
        "status": "released_after_model_blind_calibration_before_any_phase_b_model_request",
        "recorded_at_utc": recorded_at_utc,
        "post_result_disclosure": "Phase-A Nano outcomes and the 982-episode measurement audit were known before this ablation was selected.",
        "scientific_question": "Does Nano's directional margin asymmetry change when only movable-object center positions are reflected about the robot sagittal plane?",
        "model_identity": {
            "model_id": MODEL_ID,
            "model_repository": MODEL_REPOSITORY,
            "checkpoint_revision": MODEL_REVISION,
            "server_repository_commit": COSMOS_REPOSITORY_COMMIT,
            "robolab_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        },
        "exact_prompts": PROMPTS,
        "design": {
            "factor": FACTOR_NAME,
            "arms": list(ARMS),
            "directions": list(RELATIONS),
            "seeds": list(SEEDS),
            "matched_seed_count": 27,
            "cells_per_seed": 4,
            "behavioral_cell_count": 108,
            "pairing": "Each seed contains control/position-mirrored × LEFT/RIGHT; sampling and environment seed are identical across all four cells.",
            "execution_order": "outcome-independent SHA-256 block randomization, position-balanced to 6 or 7 appearances per condition per ordinal position",
            "randomization_namespace": RANDOMIZATION_NAMESPACE,
            "post_settle_orientation_policy": "Initial quaternion sources are identical. Cross-layout post-settle orientation differences are recorded as downstream physical mediators of the positions-only intervention, not used as a release threshold, and not silently described as held fixed.",
        },
        "fixtures": fixtures,
        "relation_geometry": relation_geometry,
        "calibration_report": calibration_record,
        "calibration_evidence": {
            "study_checkout": report["study_checkout"],
            "robolab_commit": report["robolab"]["commit"],
            "candidate": report["candidate"],
            "calibration_driver_source": report["calibration_driver_source"],
            "factor_task_source": report["factor_task_source"],
            "factor_task_wrappers": report["factor_task_wrappers"],
            "renderer": report["renderer"],
            "reset_gate": report["reset_gate"],
            "post_settle_cross_layout_quaternion_differences": report[
                "post_settle_cross_layout_quaternion_differences"
            ],
            "viewport_write_gate": report["viewport_write_gate"],
            "model_request_count": report["model_request_count"],
            "behavioral_episode_count": report["behavioral_episode_count"],
        },
        "source_bindings": SOURCE_BINDINGS,
        "unused_seed_audit": seed_audit,
        "analysis_plan": _analysis_plan(),
        "logging_contract": {
            "signed_final_lateral_offset_m": "finite and explicit for every valid episode, failures included",
            "final_requested_signed_margin_m": "finite and explicit for every valid episode; analyzed only on the named success-complete subset",
            "margin_derivation_check": "+signed offset for LEFT; -signed offset for RIGHT",
            "valid_failures": "preserved in full-sample offset denominators",
            "infrastructure_failures": "separate stream; never model failures or zeros",
        },
        "prohibited": [
            "reuse Phase-A episodes as contemporaneous control",
            "change prompts, success predicate, model revision, or fixture after release",
            "infer missing margin or offset from requested_success",
            "analyze requested-side margin outside the named success-complete subset",
            "pool DROID and RoboTwin",
        ],
    }
    amendment_bytes = canonical_json_bytes(amendment)
    amendment_hash = sha256_bytes(amendment_bytes)
    rows = _build_rows(fixtures, amendment_hash)
    cells_bytes = canonical_jsonl_bytes(rows)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "hash_bound_release_ready",
        "recorded_at_utc": recorded_at_utc,
        "calibration_report": calibration_record,
        "files": {
            "amendment": {
                "path": OUTPUT_FILENAMES["amendment"],
                "bytes": len(amendment_bytes),
                "sha256": amendment_hash,
            },
            "cells": {
                "path": OUTPUT_FILENAMES["cells"],
                "bytes": len(cells_bytes),
                "sha256": sha256_bytes(cells_bytes),
                "row_count": len(rows),
            },
        },
        "counts": {
            "matched_seeds": len(SEEDS),
            "control_cells": sum(row["arm"] == "control" for row in rows),
            "position_mirrored_cells": sum(
                row["arm"] == "position_mirrored" for row in rows
            ),
            "left_cells": sum(row["relation"] == "left" for row in rows),
            "right_cells": sum(row["relation"] == "right" for row in rows),
            "behavioral_cells": len(rows),
        },
        "execution_order_position_counts": _position_balance(rows),
        "release_rule": "Launch only rows in this exact queue after rechecking the hash-bound amendment, exact live runtime identity, viewport writer, and raw-output path; any failed calibration gate yields no queue.",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    return ReleasePayloads(
        amendment=amendment,
        rows=rows,
        manifest=manifest,
        amendment_bytes=amendment_bytes,
        cells_bytes=cells_bytes,
        manifest_bytes=manifest_bytes,
    )


def write_release(output_dir: Path, payloads: ReleasePayloads) -> dict[str, Path]:
    """Atomically write a validated release, refusing to overwrite history."""

    output_dir = Path(output_dir)
    paths = {name: output_dir / filename for name, filename in OUTPUT_FILENAMES.items()}
    existing = [str(path) for path in paths.values() if path.exists()]
    require(not existing, f"refusing to overwrite existing release artifacts: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = {
        "amendment": payloads.amendment_bytes,
        "cells": payloads.cells_bytes,
        "manifest": payloads.manifest_bytes,
    }
    temporary: list[tuple[Path, Path]] = []
    try:
        for name, target in paths.items():
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output_dir,
                prefix=f".{target.name}.",
                delete=False,
            ) as handle:
                handle.write(contents[name])
                handle.flush()
                os.fsync(handle.fileno())
                temporary.append((Path(handle.name), target))
        for source, target in temporary:
            source.replace(target)
        return paths
    finally:
        for source, _ in temporary:
            if source.exists():
                source.unlink()
