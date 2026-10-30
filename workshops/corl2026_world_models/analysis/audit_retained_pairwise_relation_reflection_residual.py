#!/usr/bin/env python3
"""Audit retained pairwise relation reflection residuals only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
TURN2849_PACKAGE_COMMIT = "711eac7c0253ddd10784d17b205b7da4397b4fb9"
TURN2849_PACKAGE_TREE = "57ca5e11185631b0fc6f8a0da716b0af3ff630ed"
TURN2849_PACKAGE_PARENT = "d1b6724cc5e28b5a2dc78d4edff86d026ba0a81a"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2849_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_center_representation_residual_difference.py"
    ),
    "lines": 1061,
    "bytes": 42188,
    "sha256": "e153680f042aefa61938a7ba9b576eace5a7e0ed9104993a500adf9647b665fa",
}
TURN2849_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_center_representation_residual_difference_diagnostic.json"
    ),
    "lines": 2999,
    "bytes": 105930,
    "sha256": "d7b1d6c0e35b93298db8af1b7659885cd088f61a4be46e75f05b0ad172006021",
}
OUTPUT_PATH = REPOSITORY / (
    "workshops/corl2026_world_models/results/"
    "retained_pairwise_relation_reflection_residual_diagnostic.json"
)
OBJECT_NAMES = ("banana", "bowl", "rubiks_cube")
OBJECT_PAIRS = tuple(combinations(OBJECT_NAMES, 2))
CAMERA_NAMES = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
COMPARISON_KEYS = (
    ("development", "D01", "D1"),
    ("development", "D02", "D1"),
    ("pilot", "P00", "N3"),
)
WORLD_AXES = ("x", "y", "z")
PROJECTED_AXES = ("u", "v")
WORLD_FAMILIES = (
    "center_relation_reflection_residual",
    "aabb_midpoint_relation_reflection_residual",
    "center_minus_aabb_midpoint_relation_residual_difference",
)
PROJECTED_FAMILIES = (
    "point_center_relation_reflection_residual",
    "envelope_center_relation_reflection_residual",
    "point_minus_envelope_relation_residual_difference",
)
EXPECTED_WORLD_SIGNS = {
    "center_relation_reflection_residual": {
        "x": {"negative": 5, "zero": 0, "positive": 4},
        "y": {"negative": 3, "zero": 0, "positive": 6},
        "z": {"negative": 5, "zero": 0, "positive": 4},
    },
    "aabb_midpoint_relation_reflection_residual": {
        "x": {"negative": 6, "zero": 0, "positive": 3},
        "y": {"negative": 3, "zero": 0, "positive": 6},
        "z": {"negative": 5, "zero": 0, "positive": 4},
    },
    "center_minus_aabb_midpoint_relation_residual_difference": {
        "x": {"negative": 3, "zero": 1, "positive": 5},
        "y": {"negative": 3, "zero": 0, "positive": 6},
        "z": {"negative": 5, "zero": 1, "positive": 3},
    },
}
EXPECTED_PROJECTED_SIGNS = {
    "point_center_relation_reflection_residual": {
        "u": {"negative": 19, "zero": 0, "positive": 17},
        "v": {"negative": 18, "zero": 0, "positive": 18},
    },
    "envelope_center_relation_reflection_residual": {
        "u": {"negative": 18, "zero": 0, "positive": 18},
        "v": {"negative": 18, "zero": 0, "positive": 18},
    },
    "point_minus_envelope_relation_residual_difference": {
        "u": {"negative": 16, "zero": 0, "positive": 20},
        "v": {"negative": 19, "zero": 0, "positive": 17},
    },
}
EXPECTED_ZERO_COUNTS = {
    "center_relation_reflection_residual": (0, 0),
    "aabb_midpoint_relation_reflection_residual": (0, 0),
    "center_minus_aabb_midpoint_relation_residual_difference": (2, 0),
    "point_center_relation_reflection_residual": (0, 0),
    "envelope_center_relation_reflection_residual": (0, 0),
    "point_minus_envelope_relation_residual_difference": (0, 0),
}
EXPECTED_GLOBAL_EXTREMA = {
    "center_relation_reflection_residual": {
        "component": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "axis": "y",
            "signed_mm": 2.3232661187648773,
            "absolute_mm": 2.3232661187648773,
            "tie_count": 1,
        },
        "row": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "signed_mm": [
                -0.9375512599945068,
                2.3232661187648773,
                2.2351741790771484e-05,
            ],
            "l2_mm": 2.505307930019323,
            "tie_count": 1,
        },
    },
    "aabb_midpoint_relation_reflection_residual": {
        "component": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "axis": "y",
            "signed_mm": 2.43547186255455,
            "absolute_mm": 2.43547186255455,
            "tie_count": 1,
        },
        "row": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "signed_mm": [
                -0.6215423345565796,
                2.43547186255455,
                6.51925802230835e-05,
            ],
            "l2_mm": 2.5135309966640644,
            "tie_count": 1,
        },
    },
    "center_minus_aabb_midpoint_relation_residual_difference": {
        "component": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "axis": "x",
            "signed_mm": -0.31600892543792725,
            "absolute_mm": 0.31600892543792725,
            "tie_count": 2,
        },
        "row": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "first_object_name": "banana",
            "second_object_name": "rubiks_cube",
            "signed_mm": [
                -0.31600892543792725,
                -0.11220946907997131,
                -4.0978193283081055e-05,
            ],
            "l2_mm": 0.3353395407446829,
            "tie_count": 1,
        },
    },
    "point_center_relation_reflection_residual": {
        "component": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "camera_name": "wrist_cam",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "axis": "u",
            "signed_px": 4.068751020790614,
            "absolute_px": 4.068751020790614,
            "tie_count": 1,
        },
        "row": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "camera_name": "wrist_cam",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "signed_px": [4.068751020790614, -1.7411999817632022],
            "l2_px": 4.425665175504903,
            "tie_count": 1,
        },
    },
    "envelope_center_relation_reflection_residual": {
        "component": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "camera_name": "wrist_cam",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "axis": "u",
            "signed_px": 4.420630524573312,
            "absolute_px": 4.420630524573312,
            "tie_count": 1,
        },
        "row": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "camera_name": "wrist_cam",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "signed_px": [4.420630524573312, -1.1941083346341657],
            "l2_px": 4.5790685679111744,
            "tie_count": 1,
        },
    },
    "point_minus_envelope_relation_residual_difference": {
        "component": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "camera_name": "wrist_cam",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "axis": "v",
            "signed_px": -0.5470916471290366,
            "absolute_px": 0.5470916471290366,
            "tie_count": 1,
        },
        "row": {
            "phase": "development",
            "layout_pair_id": "D02",
            "model_id": "D1",
            "camera_name": "wrist_cam",
            "first_object_name": "bowl",
            "second_object_name": "rubiks_cube",
            "signed_px": [-0.3518795037826976, -0.5470916471290366],
            "l2_px": 0.6504832477018295,
            "tie_count": 1,
        },
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def strict_json_loads(payload: bytes, source: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant in {source}: {value}")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key in {source}: {key}")
            result[key] = value
        return result

    parsed = json.loads(
        payload.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_pairs,
    )
    require(isinstance(parsed, dict), f"Top-level JSON object required: {source}")
    return parsed


def git_bytes(commit: str, path: str) -> bytes:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(REPOSITORY),
            "--no-replace-objects",
            "show",
            f"{commit}:{path}",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def git_line(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(REPOSITORY), "--no-replace-objects", *arguments],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def verify_payload(payload: bytes, descriptor: dict, label: str) -> None:
    require(len(payload) == descriptor["bytes"], f"{label} byte count changed")
    require(
        payload.count(b"\n") == descriptor["lines"],
        f"{label} line count changed",
    )
    require(
        hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
        f"{label} SHA-256 changed",
    )


def import_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_turn2849_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2849_PACKAGE_COMMIT}^{{commit}}")
        == TURN2849_PACKAGE_COMMIT,
        "Pinned Turn2849 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2849_PACKAGE_COMMIT)
        == TURN2849_PACKAGE_TREE,
        "Pinned Turn2849 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2849_PACKAGE_COMMIT)
        == TURN2849_PACKAGE_PARENT,
        "Pinned Turn2849 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2849_PACKAGE_COMMIT, TURN2849_AUDIT["path"])
    verify_payload(committed_audit, TURN2849_AUDIT, "Pinned Turn2849 audit")
    local_audit_path = REPOSITORY / TURN2849_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2849_AUDIT, "Local Turn2849 audit")
    require(local_audit == committed_audit, "Local Turn2849 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2849_center_difference")

    committed_report = git_bytes(TURN2849_PACKAGE_COMMIT, TURN2849_REPORT["path"])
    verify_payload(committed_report, TURN2849_REPORT, "Pinned Turn2849 report")
    local_report = (REPOSITORY / TURN2849_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2849_REPORT, "Local Turn2849 report")
    require(local_report == committed_report, "Local Turn2849 report differs")
    parsed = strict_json_loads(committed_report, TURN2849_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2849 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2849 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-center-representation-residual-difference-diagnostic-v1",
        "Turn2849 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_object_rows"] == 9
        and selection["world_difference_components"] == 27
        and selection["mapping_pair_camera_rows"] == 12
        and selection["projected_camera_object_rows"] == 36
        and selection["projected_difference_components"] == 72
        and selection["availability_selected"] is True
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection[
            "P00_shared_left_command_with_extra_reflected_right_member"
        ]
        is True
        and selection["isolated_layout_effect_estimated"] is False,
        "Turn2849 selection boundary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 31
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2849 evidence boundary changed",
    )
    require(
        parsed["new_behavioral_cells"] == 0
        and parsed["new_model_requests"] == 0
        and parsed["new_actions"] == 0
        and parsed["new_observations"] == 0
        and parsed["new_request_selections"] == 0
        and parsed["labels_created"] == 0
        and parsed["forecast_accuracy_estimated"] is False
        and parsed["resource_budget_qualified"] is False
        and parsed["confirmation_released"] is False,
        "Turn2849 zero-activity boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2849_PACKAGE_COMMIT,
        "tree": TURN2849_PACKAGE_TREE,
        "parent": TURN2849_PACKAGE_PARENT,
        "audit": TURN2849_AUDIT,
        "report": TURN2849_REPORT,
        "canonical_regeneration_verified": True,
    }


def finite_number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape")
    return [finite_number(child, label) for child in value]


def subtract(first: list[float], second: list[float]) -> list[float]:
    require(len(first) == len(second), "Vector shape changed")
    return [first[index] - second[index] for index in range(len(first))]


def vector_l1(values: list[float]) -> float:
    return math.fsum(abs(value) for value in values)


def vector_l2(values: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def sign(value: float) -> str:
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def vector_record(
    values: list[float], axes: tuple[str, ...], unit: str
) -> dict:
    require(len(values) == len(axes), "Vector-record shape changed")
    absolute = [abs(value) for value in values]
    result = {
        f"signed_{unit}": values,
        f"absolute_{unit}": absolute,
        "component_signs": {
            axis: sign(values[index]) for index, axis in enumerate(axes)
        },
        f"l1_{unit}": vector_l1(values),
        f"l2_{unit}": vector_l2(values),
        f"linf_{unit}": max(absolute),
    }
    if unit == "m":
        result.update(
            {
                "signed_mm": [1000.0 * value for value in values],
                "absolute_mm": [1000.0 * value for value in absolute],
                "l1_mm": 1000.0 * result["l1_m"],
                "l2_mm": 1000.0 * result["l2_m"],
                "linf_mm": 1000.0 * result["linf_m"],
            }
        )
    return result


def provenance_key(provenance: dict, projected: bool) -> tuple[str, ...]:
    keys = ["phase", "layout_pair_id", "model_id"]
    if projected:
        keys.append("camera_name")
    keys.append("object_name")
    return tuple(str(provenance[key]) for key in keys)


def validate_source_provenance(provenance: object, projected: bool) -> dict:
    require(isinstance(provenance, dict), "Source provenance missing")
    required = {
        "phase",
        "layout_pair_id",
        "model_id",
        "object_name",
        "original_member_cell_ids",
        "original_member_commands",
        "original_representative_cell_id",
        "reflected_member_cell_ids",
        "reflected_member_commands",
        "reflected_representative_cell_id",
        "command_membership_classification",
        "source_package_commit",
    }
    if projected:
        required |= {"camera_name", "camera_configuration_sha256"}
    require(set(provenance) == required, "Source provenance keys changed")
    require(
        provenance["source_package_commit"]
        == "717136db3c518a3e59a01d33a86f95839ee83b23",
        "Source provenance package pin changed",
    )
    return provenance


def pair_provenance(first: dict, second: dict, projected: bool) -> dict:
    first = validate_source_provenance(first, projected)
    second = validate_source_provenance(second, projected)
    first_object = first["object_name"]
    second_object = second["object_name"]
    require((first_object, second_object) in OBJECT_PAIRS, "Object pair changed")
    shared_first = {key: value for key, value in first.items() if key != "object_name"}
    shared_second = {
        key: value for key, value in second.items() if key != "object_name"
    }
    require(shared_first == shared_second, "Pair provenance differs")
    return {
        **shared_first,
        "first_object_name": first_object,
        "second_object_name": second_object,
        "pair_direction": "first_minus_second",
    }


def source_vector(row: dict, key: str, length: int, label: str) -> list[float]:
    return finite_vector(row.get(key), length, label)


def build_world_row(first: dict, second: dict) -> dict:
    provenance = pair_provenance(
        first["provenance"], second["provenance"], projected=False
    )
    first_center = source_vector(
        first, "actual_minus_constructed_center_residual_m", 3, "First center"
    )
    second_center = source_vector(
        second, "actual_minus_constructed_center_residual_m", 3, "Second center"
    )
    first_midpoint = source_vector(
        first,
        "actual_minus_constructed_aabb_midpoint_residual_m",
        3,
        "First midpoint",
    )
    second_midpoint = source_vector(
        second,
        "actual_minus_constructed_aabb_midpoint_residual_m",
        3,
        "Second midpoint",
    )
    first_difference = source_vector(
        first,
        "center_minus_aabb_midpoint_residual_difference_m",
        3,
        "First representation difference",
    )
    second_difference = source_vector(
        second,
        "center_minus_aabb_midpoint_residual_difference_m",
        3,
        "Second representation difference",
    )
    center = subtract(first_center, second_center)
    midpoint = subtract(first_midpoint, second_midpoint)
    direct_difference = subtract(center, midpoint)
    inherited_difference = subtract(first_difference, second_difference)
    require(
        direct_difference == inherited_difference,
        "World pair representation identity changed",
    )
    return {
        "provenance": provenance,
        "center_relation_reflection_residual": vector_record(
            center, WORLD_AXES, "m"
        ),
        "aabb_midpoint_relation_reflection_residual": vector_record(
            midpoint, WORLD_AXES, "m"
        ),
        "center_minus_aabb_midpoint_relation_residual_difference": (
            vector_record(inherited_difference, WORLD_AXES, "m")
        ),
        "published_per_object_residual_then_pair_subtraction": True,
        "representation_difference_pair_identity_exact": True,
    }


def build_projected_row(first: dict, second: dict) -> dict:
    provenance = pair_provenance(
        first["provenance"], second["provenance"], projected=True
    )
    first_point = source_vector(
        first,
        "actual_minus_constructed_projected_point_center_residual_px",
        2,
        "First projected point",
    )
    second_point = source_vector(
        second,
        "actual_minus_constructed_projected_point_center_residual_px",
        2,
        "Second projected point",
    )
    first_envelope = source_vector(
        first,
        "actual_minus_constructed_projected_envelope_center_residual_px",
        2,
        "First projected envelope",
    )
    second_envelope = source_vector(
        second,
        "actual_minus_constructed_projected_envelope_center_residual_px",
        2,
        "Second projected envelope",
    )
    first_difference = source_vector(
        first,
        "point_center_minus_envelope_center_residual_difference_px",
        2,
        "First projected representation difference",
    )
    second_difference = source_vector(
        second,
        "point_center_minus_envelope_center_residual_difference_px",
        2,
        "Second projected representation difference",
    )
    point = subtract(first_point, second_point)
    envelope = subtract(first_envelope, second_envelope)
    direct_difference = subtract(point, envelope)
    inherited_difference = subtract(first_difference, second_difference)
    require(
        direct_difference == inherited_difference,
        "Projected pair representation identity changed",
    )
    return {
        "provenance": provenance,
        "point_center_relation_reflection_residual": vector_record(
            point, PROJECTED_AXES, "px"
        ),
        "envelope_center_relation_reflection_residual": vector_record(
            envelope, PROJECTED_AXES, "px"
        ),
        "point_minus_envelope_relation_residual_difference": vector_record(
            inherited_difference, PROJECTED_AXES, "px"
        ),
        "published_per_object_residual_then_pair_subtraction": True,
        "representation_difference_pair_identity_exact": True,
    }


def sign_census(rows: list[dict], family: str, axes: tuple[str, ...]) -> dict:
    census = {
        axis: {"negative": 0, "zero": 0, "positive": 0} for axis in axes
    }
    for row in rows:
        signs = row[family]["component_signs"]
        require(tuple(signs) == axes, "Sign axis order changed")
        for axis in axes:
            census[axis][signs[axis]] += 1
    return census


def compact_provenance(provenance: dict) -> dict:
    keys = [
        "phase",
        "layout_pair_id",
        "model_id",
        "first_object_name",
        "second_object_name",
    ]
    if "camera_name" in provenance:
        keys.append("camera_name")
        keys.append("camera_configuration_sha256")
    return {key: provenance[key] for key in keys}


def component_extremum(
    rows: list[dict], family: str, axes: tuple[str, ...], unit: str
) -> dict:
    candidates = []
    for row in rows:
        values = row[family][f"signed_{unit}"]
        for index, axis in enumerate(axes):
            candidates.append((abs(values[index]), row, index, axis))
    maximum = max(candidate[0] for candidate in candidates)
    ties = []
    for absolute, row, index, axis in candidates:
        if absolute != maximum:
            continue
        tie = {
            **compact_provenance(row["provenance"]),
            "axis": axis,
            f"signed_{unit}": row[family][f"signed_{unit}"][index],
            f"absolute_{unit}": absolute,
        }
        if unit == "m":
            tie.update(
                {
                    "signed_mm": row[family]["signed_mm"][index],
                    "absolute_mm": row[family]["absolute_mm"][index],
                }
            )
        ties.append(tie)
    return {f"maximum_absolute_{unit}": maximum, "tie_count": len(ties), "ties": ties}


def row_extremum(rows: list[dict], family: str, unit: str) -> dict:
    maximum = max(row[family][f"l2_{unit}"] for row in rows)
    ties = []
    for row in rows:
        if row[family][f"l2_{unit}"] != maximum:
            continue
        tie = {
            **compact_provenance(row["provenance"]),
            f"signed_{unit}": row[family][f"signed_{unit}"],
            f"l2_{unit}": maximum,
        }
        if unit == "m":
            tie.update(
                {
                    "signed_mm": row[family]["signed_mm"],
                    "l2_mm": row[family]["l2_mm"],
                }
            )
        ties.append(tie)
    return {f"maximum_l2_{unit}": maximum, "tie_count": len(ties), "ties": ties}


def family_summary(
    rows: list[dict], family: str, axes: tuple[str, ...], unit: str
) -> dict:
    vectors = [row[family][f"signed_{unit}"] for row in rows]
    return {
        "rows": len(rows),
        "components": len(rows) * len(axes),
        "exact_zero_components": sum(
            value == 0.0 for vector in vectors for value in vector
        ),
        "exact_zero_rows": sum(
            all(value == 0.0 for value in vector) for vector in vectors
        ),
        "sign_census": sign_census(rows, family, axes),
        "maximum_absolute_component": component_extremum(
            rows, family, axes, unit
        ),
        "maximum_l2_row": row_extremum(rows, family, unit),
    }


def group_extrema(
    rows: list[dict],
    families: tuple[str, ...],
    axes: tuple[str, ...],
    unit: str,
    group_keys: tuple[str, ...],
) -> list[dict]:
    groups = {}
    for row in rows:
        group = tuple(row["provenance"][key] for key in group_keys)
        groups.setdefault(group, []).append(row)
    results = []
    for group in sorted(groups):
        selected = groups[group]
        for family in families:
            results.append(
                {
                    **dict(zip(group_keys, group)),
                    "family": family,
                    "rows": len(selected),
                    "maximum_absolute_component": component_extremum(
                        selected, family, axes, unit
                    ),
                    "maximum_l2_row": row_extremum(selected, family, unit),
                }
            )
    return results


def validate_expected_extremum(family: str, summary: dict, unit: str) -> None:
    expected = EXPECTED_GLOBAL_EXTREMA[family]
    component = summary["maximum_absolute_component"]
    row = summary["maximum_l2_row"]
    require(component["tie_count"] == expected["component"]["tie_count"], f"{family} component tie count")
    require(row["tie_count"] == expected["row"]["tie_count"], f"{family} row tie count")
    component_ties = component["ties"]
    for key, value in expected["component"].items():
        if key == "tie_count":
            continue
        if key in ("first_object_name", "second_object_name") and component["tie_count"] > 1:
            continue
        require(component_ties[0][key] == value, f"{family} component {key}")
    row_tie = row["ties"][0]
    for key, value in expected["row"].items():
        if key == "tie_count":
            continue
        require(row_tie[key] == value, f"{family} row {key}")
    if family == "center_minus_aabb_midpoint_relation_residual_difference":
        require(
            [(tie["first_object_name"], tie["second_object_name"]) for tie in component_ties]
            == [("banana", "rubiks_cube"), ("bowl", "rubiks_cube")],
            "World representation-difference component ties changed",
        )


def analyze(source: dict) -> tuple[list[dict], list[dict], dict]:
    source_world = {}
    for row in source.get("world_rows", []):
        provenance = validate_source_provenance(row.get("provenance"), False)
        key = provenance_key(provenance, False)
        require(key not in source_world, f"Duplicate source world row: {key}")
        source_world[key] = row
    source_projected = {}
    for row in source.get("projected_rows", []):
        provenance = validate_source_provenance(row.get("provenance"), True)
        key = provenance_key(provenance, True)
        require(key not in source_projected, f"Duplicate source projected row: {key}")
        source_projected[key] = row
    require(len(source_world) == 9, "Source world row count changed")
    require(len(source_projected) == 36, "Source projected row count changed")

    world_rows = []
    projected_rows = []
    for comparison in COMPARISON_KEYS:
        for first_object, second_object in OBJECT_PAIRS:
            world_rows.append(
                build_world_row(
                    source_world[(*comparison, first_object)],
                    source_world[(*comparison, second_object)],
                )
            )
        for camera_name in CAMERA_NAMES:
            for first_object, second_object in OBJECT_PAIRS:
                projected_rows.append(
                    build_projected_row(
                        source_projected[
                            (*comparison, camera_name, first_object)
                        ],
                        source_projected[
                            (*comparison, camera_name, second_object)
                        ],
                    )
                )
    require(len(world_rows) == 9, "World relation row count changed")
    require(len(projected_rows) == 36, "Projected relation row count changed")

    world_summaries = {
        family: family_summary(world_rows, family, WORLD_AXES, "m")
        for family in WORLD_FAMILIES
    }
    projected_summaries = {
        family: family_summary(projected_rows, family, PROJECTED_AXES, "px")
        for family in PROJECTED_FAMILIES
    }
    for family in WORLD_FAMILIES:
        summary = world_summaries[family]
        require(summary["sign_census"] == EXPECTED_WORLD_SIGNS[family], f"{family} signs")
        require(
            (summary["exact_zero_components"], summary["exact_zero_rows"])
            == EXPECTED_ZERO_COUNTS[family],
            f"{family} zeros",
        )
        validate_expected_extremum(family, summary, "m")
    for family in PROJECTED_FAMILIES:
        summary = projected_summaries[family]
        require(
            summary["sign_census"] == EXPECTED_PROJECTED_SIGNS[family],
            f"{family} signs",
        )
        require(
            (summary["exact_zero_components"], summary["exact_zero_rows"])
            == EXPECTED_ZERO_COUNTS[family],
            f"{family} zeros",
        )
        validate_expected_extremum(family, summary, "px")

    summary = {
        "original_reflected_mapping_pairs": 3,
        "source_reset_mappings": 6,
        "exhaustive_object_pairs": len(OBJECT_PAIRS),
        "object_pair_order": [list(pair) for pair in OBJECT_PAIRS],
        "pair_direction": "first_minus_second",
        "world_relation_rows": len(world_rows),
        "world_family_components": 27,
        "world_total_vector_components": 81,
        "projected_mapping_camera_rows": 12,
        "projected_relation_rows": len(projected_rows),
        "projected_family_components": 72,
        "projected_total_vector_components": 216,
        "world_families": world_summaries,
        "projected_families": projected_summaries,
        "world_extrema_by_object_pair": group_extrema(
            world_rows,
            WORLD_FAMILIES,
            WORLD_AXES,
            "m",
            ("first_object_name", "second_object_name"),
        ),
        "projected_extrema_by_object_pair": group_extrema(
            projected_rows,
            PROJECTED_FAMILIES,
            PROJECTED_AXES,
            "px",
            ("first_object_name", "second_object_name"),
        ),
        "projected_extrema_by_camera": group_extrema(
            projected_rows,
            PROJECTED_FAMILIES,
            PROJECTED_AXES,
            "px",
            ("camera_name",),
        ),
        "world_representation_identity_components_verified": 27,
        "projected_representation_identity_components_verified": 72,
    }
    return world_rows, projected_rows, summary


def audit() -> dict:
    _, source, source_pin = load_turn2849_package()
    world_rows, projected_rows, summary = analyze(source)
    return {
        "schema_version": (
            "wmf-retained-pairwise-relation-reflection-residual-diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2849_center_representation_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "exhaustive_object_pairs_per_mapping": 3,
            "world_relation_rows": 9,
            "world_vector_families": 3,
            "world_total_vector_components": 81,
            "mapping_pair_camera_rows": 12,
            "projected_relation_rows": 36,
            "projected_vector_families": 3,
            "projected_total_vector_components": 216,
            "availability_selected": True,
            "outcome_dependent_selection": False,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
            "D01_D02_command_confounded": True,
            "P00_shared_left_command_with_extra_reflected_right_member": True,
            "isolated_layout_or_command_effect_estimated": False,
        },
        "relation_semantics": {
            "direction": (
                "Object pairs are the exhaustive combinations of the frozen order "
                "banana,bowl,rubiks_cube. Every vector is first object minus second "
                "object; direction is fixed before inspecting values."
            ),
            "operation_order": (
                "For each family, subtract the published per-object actual-minus-"
                "constructed residual of the second object from that of the first. "
                "The published per-object center-minus-midpoint or point-minus-envelope "
                "difference is likewise subtracted first-minus-second. Its exact "
                "identity with pairwise center minus pairwise midpoint or pairwise point "
                "minus pairwise envelope is verified, but that regrouped expression is "
                "not serialized as the primary value and no tolerance is used."
            ),
            "shared_vector_invariance": (
                "Pair subtraction is algebraically invariant to adding an identical "
                "residual vector to both objects. This mathematical property is not "
                "evidence that any observed residual has a shared physical cause."
            ),
            "units": (
                "World vectors use robot-base x/y/z in metres with exact 1000x "
                "millimetre copies. Projected vectors use stored u/v pixels. L1,L2 "
                "and Linf norms are unnormalized descriptive magnitudes."
            ),
        },
        "summary": summary,
        "world_relation_rows": world_rows,
        "projected_relation_rows": projected_rows,
        "algebraically_derived_from_turn2849": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "a7_result": False,
        "r0_rhat_r_endpoint": False,
        "motion_or_temporal_transition_inferred": False,
        "generated_or_executed_correspondence": False,
        "normalization_applied": False,
        "included_in_existing_six_report_annotation_closure": False,
        "finding": (
            "Across three availability-selected original/reflected mapping pairs and "
            "all three object pairs, nine world relation rows and 36 camera-object-"
            "pair rows describe exact first-minus-second reflection residuals. The "
            "largest absolute center-relation component is +2.3232661187648773 mm on "
            "D02 bowl-minus-rubiks_cube y; the largest absolute projected point-center "
            "relation component is +4.068751020790614 px on D02 wrist_cam bowl-minus-"
            "rubiks_cube u. The largest absolute center-minus-AABB-midpoint relation "
            "residual-difference component is -0.31600892543792725 mm, tied on D02 "
            "banana-minus-rubiks_cube x and D02 bowl-minus-rubiks_cube x; the largest "
            "absolute projected point-minus-envelope relation residual-difference "
            "component is -0.5470916471290366 px on D02 wrist_cam bowl-minus-"
            "rubiks_cube v. These reset-stage geometric relation descriptions are not "
            "r0,rhat,r, A7, an effect or a scientific gate result."
        ),
        "claim_boundaries": {
            "frozen_relational_geometry_only": (
                "This report subtracts deterministic per-object residual vectors from "
                "the pinned Turn2849 package. It adds no observation, image, scene, "
                "model request, robot episode, label or scientific sample."
            ),
            "constructed_reference_only": (
                "The constructed pure-reflection correspondent remains a deterministic "
                "geometric reference, not recovered fixture truth, an acceptance target, "
                "an observed image, a counterfactual rollout or an independent measure."
            ),
            "not_primary_relation_endpoint": (
                "Pairwise reset-stage point and AABB-center residuals are not human-"
                "marked cube-minus-bowl image centroids and have no request-start, "
                "generated-horizon or executed-time correspondence. They are not r0, "
                "rhat or r and do not answer A7 or any A1-A9 question."
            ),
            "aabb_and_projection_boundary": (
                "AABB midpoints and projected AABB-envelope centers are retained "
                "geometric proxies, not OBBs, meshes, silhouettes, masks, image "
                "footprints, object pixels or observed centroids."
            ),
            "selection_and_confounding": (
                "All three object pairs are exhausted without outcome selection, but "
                "the three mapping pairs remain availability-selected. D01/D02 are "
                "command-confounded and P00 has unequal command membership. No layout, "
                "command, model, task-success, population or camera effect is estimated."
            ),
            "descriptive_extrema_only": (
                "Signs,norms and extrema are exact descriptions, not tolerances, "
                "quality scores, pass/fail rules or object-pair,camera,model or "
                "representation rankings."
            ),
            "science_and_release": (
                "No threshold, fitting, registration, normalization, forecast metric, "
                "current-state inference, qualification, gate advance, dispatch "
                "authority or confirmation release follows."
            ),
        },
        "new_behavioral_cells": 0,
        "new_model_requests": 0,
        "new_actions": 0,
        "new_observations": 0,
        "new_request_selections": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "resource_budget_qualified": False,
        "confirmation_released": False,
        "evidence_boundaries": {
            "raw_camera_bytes_read": False,
            "raw_pvc_bytes_read": False,
            "rgb_or_image_content_analyzed": False,
            "image_footprint_or_object_pixels_inferred": False,
            "silhouette_or_segmentation_analyzed": False,
            "obb_or_exact_mesh_analyzed": False,
            "human_image_centroids_used": False,
            "projected_point_center_validated_as_human_centroid": False,
            "aabb_envelope_center_validated_as_human_centroid": False,
            "visibility_inferred": False,
            "general_occlusion_analyzed": False,
            "robot_or_static_scene_occlusion_analyzed": False,
            "camera_crop_or_time_alignment_authenticated": False,
            "labelability_estimated": False,
            "measurement_feasibility_qualified": False,
            "fixture_reference_recovered": False,
            "fixture_pass_fail_classification_created": False,
            "scientific_threshold_applied": False,
            "fitting_or_registration_performed": False,
            "normalization_applied": False,
            "cross_time_field_join_performed": False,
            "counterfactual_or_temporal_transition_inferred": False,
            "causal_effect_estimated": False,
            "isolated_layout_or_command_effect_estimated": False,
            "task_success_or_model_rate": False,
            "camera_ranking_estimated": False,
            "representation_ranking_estimated": False,
            "forecast_or_persistence_metric": False,
            "a1_a9_result": False,
            "a7_result": False,
            "current_state_inferred": False,
            "gate_advanced": False,
            "release_authority": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Describe retained pairwise relation reflection residuals without "
            "advancing science gates."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        type=Path,
        help="Verify the fixed canonical diagnostic without modifying it",
    )
    group.add_argument(
        "--write",
        action="store_true",
        help="Atomically write the canonical diagnostic to its fixed result path",
    )
    arguments = parser.parse_args()
    encoded = canonical_bytes(audit())
    if arguments.check is not None:
        require(arguments.check.resolve() == OUTPUT_PATH.resolve(), "Unexpected check path")
        require(arguments.check.read_bytes() == encoded, "Saved diagnostic differs")
        print(
            "Verified 9 world relation rows / 81 vector components and 36 projected "
            "relation rows / 216 vector components; retained-only and no stage advance."
        )
    elif arguments.write:
        temporary = OUTPUT_PATH.with_suffix(".json.tmp")
        temporary.write_bytes(encoded)
        os.replace(temporary, OUTPUT_PATH)
        print(f"Wrote {OUTPUT_PATH.relative_to(REPOSITORY)} ({len(encoded)} bytes)")
    else:
        sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
