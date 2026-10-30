#!/usr/bin/env python3
"""Audit retained center-representation residual differences only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
TURN2848_PACKAGE_COMMIT = "dc026ae59781bd7d75d51714cd65ef1939262fce"
TURN2848_PACKAGE_TREE = "b6f7f8c01db2e79337cf5cbde918550315897a7e"
TURN2848_PACKAGE_PARENT = "6e717598b222db9f4432f048e15b93f37f9c051b"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2848_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_aabb_reflection_residual_decomposition.py"
    ),
    "lines": 771,
    "bytes": 29471,
    "sha256": "d0968108eaee8c0118536c766dc1ac15a0752e6247dded1a59e41f8f21dbfe7f",
}
TURN2848_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_aabb_reflection_residual_decomposition_diagnostic.json"
    ),
    "lines": 17758,
    "bytes": 769389,
    "sha256": "598db9917bab4e75ff13fbef14d8ffe0c203827baacf1a66d0dffa88e6f99f38",
}
OUTPUT_PATH = REPOSITORY / (
    "workshops/corl2026_world_models/results/"
    "retained_center_representation_residual_difference_diagnostic.json"
)
WORLD_AXES = ("x", "y", "z")
PROJECTED_AXES = ("u", "v")
EXPECTED_WORLD_SIGN_CENSUS = {
    "x": {"negative": 3, "zero": 4, "positive": 2},
    "y": {"negative": 6, "zero": 2, "positive": 1},
    "z": {"negative": 2, "zero": 4, "positive": 3},
}
EXPECTED_PROJECTED_SIGN_CENSUS = {
    "u": {"negative": 19, "zero": 0, "positive": 17},
    "v": {"negative": 15, "zero": 0, "positive": 21},
}
EXPECTED_WORLD_MAX_COMPONENT = {
    "phase": "development",
    "layout_pair_id": "D02",
    "model_id": "D1",
    "object_name": "rubiks_cube",
    "axis": "x",
    "signed_m": 0.00031600892543792725,
    "signed_mm": 0.31600892543792725,
}
EXPECTED_WORLD_MAX_ROW = {
    "phase": "development",
    "layout_pair_id": "D02",
    "model_id": "D1",
    "object_name": "rubiks_cube",
    "difference_m": [
        0.00031600892543792725,
        0.00011220201849937439,
        4.284083843231201e-08,
    ],
    "l2_mm": 0.3353370479787535,
}
EXPECTED_PROJECTED_MAX_COMPONENT = {
    "phase": "development",
    "layout_pair_id": "D02",
    "model_id": "D1",
    "camera_name": "wrist_cam",
    "object_name": "rubiks_cube",
    "axis": "v",
    "signed_px": 0.5471362857456938,
}
EXPECTED_PROJECTED_MAX_ROW = {
    "phase": "development",
    "layout_pair_id": "D02",
    "model_id": "D1",
    "camera_name": "wrist_cam",
    "object_name": "rubiks_cube",
    "difference_px": [0.20384773985642823, 0.5471362857456938],
    "l2_px": 0.5838767132059366,
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


def load_turn2848_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2848_PACKAGE_COMMIT}^{{commit}}")
        == TURN2848_PACKAGE_COMMIT,
        "Pinned Turn2848 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2848_PACKAGE_COMMIT)
        == TURN2848_PACKAGE_TREE,
        "Pinned Turn2848 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2848_PACKAGE_COMMIT)
        == TURN2848_PACKAGE_PARENT,
        "Pinned Turn2848 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2848_PACKAGE_COMMIT, TURN2848_AUDIT["path"])
    verify_payload(committed_audit, TURN2848_AUDIT, "Pinned Turn2848 audit")
    local_audit_path = REPOSITORY / TURN2848_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2848_AUDIT, "Local Turn2848 audit")
    require(local_audit == committed_audit, "Local Turn2848 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2848_residual_decomposition")

    committed_report = git_bytes(TURN2848_PACKAGE_COMMIT, TURN2848_REPORT["path"])
    verify_payload(committed_report, TURN2848_REPORT, "Pinned Turn2848 report")
    local_report = (REPOSITORY / TURN2848_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2848_REPORT, "Local Turn2848 report")
    require(local_report == committed_report, "Local Turn2848 report differs")
    parsed = strict_json_loads(committed_report, TURN2848_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2848 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2848 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-aabb-reflection-residual-decomposition-diagnostic-v1",
        "Turn2848 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_object_rows"] == 9
        and selection["world_signed_absolute_component_rows"] == 135
        and selection["projected_camera_object_rows"] == 36
        and selection["projected_signed_absolute_component_rows"] == 360
        and selection["availability_selected"] is True
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection[
            "P00_shared_left_command_with_extra_reflected_right_member"
        ]
        is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2848 selection boundary changed",
    )
    require(
        all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2848 evidence boundary changed",
    )
    require(
        parsed["new_behavioral_cells"] == 0
        and parsed["new_model_requests"] == 0
        and parsed["labels_created"] == 0
        and parsed["forecast_accuracy_estimated"] is False
        and parsed["resource_budget_qualified"] is False
        and parsed["confirmation_released"] is False,
        "Turn2848 zero-science boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2848_PACKAGE_COMMIT,
        "tree": TURN2848_PACKAGE_TREE,
        "parent": TURN2848_PACKAGE_PARENT,
        "audit": TURN2848_AUDIT,
        "report": TURN2848_REPORT,
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


def signed_vector(first: list[float], second: list[float]) -> list[float]:
    require(len(first) == len(second), "Signed-vector shape changed")
    return [first[index] - second[index] for index in range(len(first))]


def midpoint(bounds: dict, label: str) -> list[float]:
    require(isinstance(bounds, dict) and set(bounds) == {"lower", "upper"}, label)
    lower = finite_vector(bounds["lower"], 3, f"{label} lower")
    upper = finite_vector(bounds["upper"], 3, f"{label} upper")
    require(
        all(lower[index] < upper[index] for index in range(3)),
        f"{label} non-positive extent",
    )
    return [(lower[index] + upper[index]) / 2.0 for index in range(3)]


def l2(values: list[float]) -> float:
    return math.sqrt(math.fsum(value * value for value in values))


def sign(value: float) -> str:
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def component_vector(
    source_row: dict,
    family: str,
    components: tuple[str, ...],
    unit: str,
) -> list[float]:
    rows = source_row.get("signed_absolute_component_rows")
    require(isinstance(rows, list), "Component rows missing")
    selected = [row for row in rows if row.get("family") == family]
    require(
        [row.get("component") for row in selected] == list(components),
        f"{family} component order changed",
    )
    values = []
    for row in selected:
        value = finite_number(row.get(f"signed_{unit}"), f"{family} signed value")
        absolute = finite_number(
            row.get(f"absolute_{unit}"), f"{family} absolute value"
        )
        require(absolute == abs(value), f"{family} absolute value changed")
        if unit == "m":
            require(
                finite_number(row.get("signed_mm"), f"{family} signed mm")
                == 1000.0 * value,
                f"{family} signed mm changed",
            )
            require(
                finite_number(row.get("absolute_mm"), f"{family} absolute mm")
                == 1000.0 * absolute,
                f"{family} absolute mm changed",
            )
        values.append(value)
    return values


def provenance_key(provenance: dict, projected: bool) -> tuple[str, ...]:
    keys = [
        "phase",
        "layout_pair_id",
        "model_id",
    ]
    if projected:
        keys.append("camera_name")
    keys.append("object_name")
    return tuple(str(provenance[key]) for key in keys)


def validate_provenance(provenance: object, projected: bool) -> dict:
    require(isinstance(provenance, dict), "Row provenance missing")
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
    require(set(provenance) == required, "Row provenance keys changed")
    require(
        provenance["source_package_commit"]
        == "717136db3c518a3e59a01d33a86f95839ee83b23",
        "Row source package pin changed",
    )
    return provenance


def row_record(
    provenance: dict,
    first_name: str,
    first: list[float],
    second_name: str,
    second: list[float],
    difference_name: str,
    difference: list[float],
    axes: tuple[str, ...],
    unit: str,
) -> dict:
    require(
        len(first) == len(second) == len(difference) == len(axes),
        "Representation row shape changed",
    )
    expected = signed_vector(first, second)
    require(expected == difference, "Representation difference arithmetic changed")
    maximum_index = max(range(len(axes)), key=lambda index: abs(difference[index]))
    result = {
        "provenance": provenance,
        first_name: first,
        second_name: second,
        difference_name: difference,
        "component_signs": {
            axis: sign(difference[index]) for index, axis in enumerate(axes)
        },
        f"difference_l2_{unit}": l2(difference),
        "maximum_absolute_component": {
            "axis": axes[maximum_index],
            f"signed_{unit}": difference[maximum_index],
            f"absolute_{unit}": abs(difference[maximum_index]),
        },
    }
    if unit == "m":
        result["difference_mm"] = [1000.0 * value for value in difference]
        result["difference_l2_mm"] = 1000.0 * result["difference_l2_m"]
        result["maximum_absolute_component"].update(
            {
                "signed_mm": 1000.0 * difference[maximum_index],
                "absolute_mm": 1000.0 * abs(difference[maximum_index]),
            }
        )
    return result


def sign_census(rows: list[dict], axes: tuple[str, ...]) -> dict:
    census = {
        axis: {"negative": 0, "zero": 0, "positive": 0} for axis in axes
    }
    for row in rows:
        signs = row["component_signs"]
        require(tuple(signs) == axes, "Sign axis order changed")
        for axis in axes:
            census[axis][signs[axis]] += 1
    return census


def component_maximum(
    rows: list[dict], axes: tuple[str, ...], difference_name: str, unit: str
) -> dict:
    candidates = []
    for row in rows:
        for index, axis in enumerate(axes):
            candidates.append((abs(row[difference_name][index]), row, index, axis))
    _, row, index, axis = max(candidates, key=lambda candidate: candidate[0])
    result = {
        **row["provenance"],
        "axis": axis,
        f"signed_{unit}": row[difference_name][index],
        f"absolute_{unit}": abs(row[difference_name][index]),
    }
    if unit == "m":
        result["signed_mm"] = 1000.0 * row[difference_name][index]
        result["absolute_mm"] = 1000.0 * abs(row[difference_name][index])
    return result


def row_maximum(rows: list[dict], difference_name: str, unit: str) -> dict:
    row = max(rows, key=lambda candidate: candidate[f"difference_l2_{unit}"])
    result = {
        **row["provenance"],
        f"difference_{unit}": row[difference_name],
        f"l2_{unit}": row[f"difference_l2_{unit}"],
    }
    if unit == "m":
        result["difference_mm"] = row["difference_mm"]
        result["l2_mm"] = row["difference_l2_mm"]
    return result


def comparison_provenance(comparison: dict, object_name: str) -> dict:
    return {
        "phase": comparison["phase"],
        "layout_pair_id": comparison["layout_pair_id"],
        "model_id": comparison["model_id"],
        "object_name": object_name,
        "original_member_cell_ids": comparison["original_member_cell_ids"],
        "original_member_commands": comparison["original_member_commands"],
        "original_representative_cell_id": comparison[
            "original_representative_cell_id"
        ],
        "reflected_member_cell_ids": comparison["reflected_member_cell_ids"],
        "reflected_member_commands": comparison["reflected_member_commands"],
        "reflected_representative_cell_id": comparison[
            "reflected_representative_cell_id"
        ],
        "command_membership_classification": comparison[
            "command_membership_classification"
        ],
        "source_package_commit": "717136db3c518a3e59a01d33a86f95839ee83b23",
    }


def analyze(
    turn2848: ModuleType, decomposition: dict
) -> tuple[list[dict], list[dict], dict]:
    _, correspondence_pin = turn2848.load_turn2847_package()
    turn2847_path = REPOSITORY / turn2848.TURN2847_AUDIT["path"]
    turn2847 = import_module(turn2847_path, "wmf_turn2847_representation_source")
    turn2846, _, _ = turn2847.load_turn2846_package()
    pair, projection, _, mappings = turn2847.index_retained_mappings(turn2846)
    correspondence = strict_json_loads(
        git_bytes(correspondence_pin["commit"], correspondence_pin["report"]["path"]),
        correspondence_pin["report"]["path"],
    )

    source_world = {}
    for row in decomposition["world_rows"]:
        provenance = validate_provenance(row.get("provenance"), projected=False)
        key = provenance_key(provenance, projected=False)
        require(key not in source_world, f"Duplicate world source row: {key}")
        source_world[key] = row
    source_projected = {}
    for row in decomposition["projected_rows"]:
        provenance = validate_provenance(row.get("provenance"), projected=True)
        key = provenance_key(provenance, projected=True)
        require(key not in source_projected, f"Duplicate projected source row: {key}")
        source_projected[key] = row
    require(len(source_world) == 9, "World source row count changed")
    require(len(source_projected) == 36, "Projected source row count changed")

    world_rows = []
    projected_rows = []
    direct_world_component_matches = 0
    direct_projected_component_matches = 0
    stored_projection_matches = 0
    camera_pair_matches = 0

    comparisons = correspondence.get("comparisons")
    require(isinstance(comparisons, list) and len(comparisons) == 3, "Comparisons")
    for comparison in comparisons:
        comparison_key = (
            comparison["phase"],
            comparison["layout_pair_id"],
            comparison["model_id"],
        )
        original = mappings[(*comparison_key, "original")]
        reflected = mappings[(*comparison_key, "reflected")]
        original_geometry, original_configs = turn2847.extract_mapping_geometry(original)
        reflected_geometry, reflected_configs = turn2847.extract_mapping_geometry(
            reflected
        )
        require(original_configs == reflected_configs, "Camera configuration changed")
        camera_pair_matches += len(turn2847.CAMERA_NAMES)
        source_world_by_object = {
            row["object_name"]: row for row in comparison["world_object_rows"]
        }
        source_projected_by_camera_object = {
            (camera["camera_name"], row["object_name"]): row
            for camera in comparison["camera_rows"]
            for row in camera["object_rows"]
        }
        camera_hashes = {
            camera["camera_name"]: camera["camera_configuration_sha256"]
            for camera in comparison["camera_rows"]
        }
        ideal_bounds_by_object = {}

        for object_name in turn2847.OBJECT_NAMES:
            provenance = comparison_provenance(comparison, object_name)
            key = (
                comparison["phase"],
                comparison["layout_pair_id"],
                comparison["model_id"],
                object_name,
            )
            source = source_world[key]
            require(source["provenance"] == provenance, "World provenance changed")
            center_residual = component_vector(
                source, "center", WORLD_AXES, "m"
            )
            midpoint_residual = component_vector(
                source, "aabb_midpoint", WORLD_AXES, "m"
            )
            difference = signed_vector(center_residual, midpoint_residual)
            correspondence_row = source_world_by_object[object_name]
            require(
                center_residual
                == finite_vector(
                    correspondence_row["actual_minus_constructed_center_target_m"],
                    3,
                    "Center residual",
                ),
                "World center residual changed",
            )
            require(
                midpoint_residual
                == finite_vector(
                    correspondence_row[
                        "actual_minus_constructed_aabb_midpoint_m"
                    ],
                    3,
                    "Midpoint residual",
                ),
                "World midpoint residual changed",
            )

            actual_center = finite_vector(
                correspondence_row["actual_reflected_center_robot_base_m"],
                3,
                "Actual center",
            )
            constructed_center = finite_vector(
                correspondence_row[
                    "constructed_center_reflection_target_robot_base_m"
                ],
                3,
                "Constructed center",
            )
            actual_midpoint = midpoint(
                correspondence_row["actual_reflected_aabb_robot_base_m"],
                "Actual AABB",
            )
            constructed_midpoint = midpoint(
                correspondence_row[
                    "constructed_center_reflected_aabb_robot_base_m"
                ],
                "Constructed AABB",
            )
            actual_offset = signed_vector(actual_center, actual_midpoint)
            constructed_offset = signed_vector(
                constructed_center, constructed_midpoint
            )
            independent_difference = signed_vector(actual_offset, constructed_offset)
            require(
                difference == independent_difference,
                "World direct and geometry-derived differences disagree",
            )
            direct_world_component_matches += len(WORLD_AXES)
            row = row_record(
                provenance,
                "actual_minus_constructed_center_residual_m",
                center_residual,
                "actual_minus_constructed_aabb_midpoint_residual_m",
                midpoint_residual,
                "center_minus_aabb_midpoint_residual_difference_m",
                difference,
                WORLD_AXES,
                "m",
            )
            row.update(
                {
                    "actual_center_minus_actual_aabb_midpoint_m": actual_offset,
                    "constructed_center_minus_constructed_aabb_midpoint_m": (
                        constructed_offset
                    ),
                    "direct_component_and_geometry_recomputation_exact": True,
                }
            )
            world_rows.append(row)
            ideal_bounds_by_object[object_name] = correspondence_row[
                "constructed_center_reflected_aabb_robot_base_m"
            ]

        for camera_name in turn2847.CAMERA_NAMES:
            configuration = original_configs[camera_name]
            original_hash = original["cameras"][camera_name][
                "camera_configuration_sha256"
            ]
            reflected_hash = reflected["cameras"][camera_name][
                "camera_configuration_sha256"
            ]
            require(
                original_hash == reflected_hash == camera_hashes[camera_name],
                "Camera configuration hash changed",
            )
            for object_name in turn2847.OBJECT_NAMES:
                base_provenance = comparison_provenance(comparison, object_name)
                provenance = {
                    **base_provenance,
                    "camera_name": camera_name,
                    "camera_configuration_sha256": original_hash,
                }
                key = (
                    comparison["phase"],
                    comparison["layout_pair_id"],
                    comparison["model_id"],
                    camera_name,
                    object_name,
                )
                source = source_projected[key]
                require(
                    source["provenance"] == provenance,
                    "Projected provenance changed",
                )
                envelope_residual = component_vector(
                    source, "envelope_center", PROJECTED_AXES, "px"
                )
                correspondence_row = source_projected_by_camera_object[
                    (camera_name, object_name)
                ]
                require(
                    envelope_residual
                    == finite_vector(
                        correspondence_row[
                            "actual_minus_constructed_envelope_center_uv_px"
                        ],
                        2,
                        "Envelope-center residual",
                    ),
                    "Projected envelope-center residual changed",
                )
                world_source = source_world_by_object[object_name]
                actual_center = finite_vector(
                    world_source["actual_reflected_center_robot_base_m"],
                    3,
                    "Actual center",
                )
                constructed_center = finite_vector(
                    world_source[
                        "constructed_center_reflection_target_robot_base_m"
                    ],
                    3,
                    "Constructed center",
                )
                actual_uv = projection.project(configuration, actual_center)
                constructed_uv = projection.project(configuration, constructed_center)
                stored_actual_uv = finite_vector(
                    reflected["cameras"][camera_name]["geometry"][
                        "projected_object_centers_uv"
                    ][object_name],
                    2,
                    "Stored actual center projection",
                )
                require(
                    actual_uv == stored_actual_uv,
                    "Stored actual center projection changed",
                )
                stored_projection_matches += 1
                point_residual = signed_vector(actual_uv, constructed_uv)
                difference = signed_vector(point_residual, envelope_residual)

                recomputed_envelope = turn2847.projected_row(
                    pair,
                    projection,
                    object_name,
                    configuration,
                    ideal_bounds_by_object[object_name],
                    reflected_geometry["object_aabbs_robot_base_m"][object_name],
                )
                require(
                    recomputed_envelope[
                        "actual_minus_constructed_envelope_center_uv_px"
                    ]
                    == envelope_residual,
                    "Recomputed envelope-center residual changed",
                )
                actual_envelope_center = turn2847.envelope_center(
                    recomputed_envelope["actual_reflected_envelope_uv_px"]
                )
                constructed_envelope_center = turn2847.envelope_center(
                    recomputed_envelope[
                        "constructed_center_reflected_envelope_uv_px"
                    ]
                )
                independent_difference = signed_vector(
                    signed_vector(actual_uv, actual_envelope_center),
                    signed_vector(constructed_uv, constructed_envelope_center),
                )
                require(
                    difference == independent_difference,
                    "Projected direct and geometry-derived differences disagree",
                )
                direct_projected_component_matches += len(PROJECTED_AXES)
                row = row_record(
                    provenance,
                    "actual_minus_constructed_projected_point_center_residual_px",
                    point_residual,
                    "actual_minus_constructed_projected_envelope_center_residual_px",
                    envelope_residual,
                    "point_center_minus_envelope_center_residual_difference_px",
                    difference,
                    PROJECTED_AXES,
                    "px",
                )
                row.update(
                    {
                        "constructed_center_point_uv_px": constructed_uv,
                        "actual_center_point_uv_px": actual_uv,
                        "stored_actual_center_projection_formula_exact": True,
                        "direct_component_and_geometry_recomputation_exact": True,
                    }
                )
                projected_rows.append(row)

    require(len(world_rows) == 9, "World output row count changed")
    require(len(projected_rows) == 36, "Projected output row count changed")
    require(len(source_world) == len(world_rows), "Unused world source row")
    require(len(source_projected) == len(projected_rows), "Unused projected source row")
    require(camera_pair_matches == 12, "Camera-pair match count changed")
    require(stored_projection_matches == 36, "Stored projection match count changed")
    require(direct_world_component_matches == 27, "World match count changed")
    require(
        direct_projected_component_matches == 72,
        "Projected match count changed",
    )

    world_signs = sign_census(world_rows, WORLD_AXES)
    projected_signs = sign_census(projected_rows, PROJECTED_AXES)
    require(world_signs == EXPECTED_WORLD_SIGN_CENSUS, "World sign census changed")
    require(
        projected_signs == EXPECTED_PROJECTED_SIGN_CENSUS,
        "Projected sign census changed",
    )
    world_difference_name = "center_minus_aabb_midpoint_residual_difference_m"
    projected_difference_name = (
        "point_center_minus_envelope_center_residual_difference_px"
    )
    world_max_component = component_maximum(
        world_rows, WORLD_AXES, world_difference_name, "m"
    )
    world_max_row = row_maximum(world_rows, world_difference_name, "m")
    projected_max_component = component_maximum(
        projected_rows, PROJECTED_AXES, projected_difference_name, "px"
    )
    projected_max_row = row_maximum(
        projected_rows, projected_difference_name, "px"
    )
    for key, value in EXPECTED_WORLD_MAX_COMPONENT.items():
        require(world_max_component[key] == value, f"World max component {key}")
    for key, value in EXPECTED_WORLD_MAX_ROW.items():
        require(world_max_row[key] == value, f"World max row {key}")
    for key, value in EXPECTED_PROJECTED_MAX_COMPONENT.items():
        require(
            projected_max_component[key] == value,
            f"Projected max component {key}",
        )
    for key, value in EXPECTED_PROJECTED_MAX_ROW.items():
        require(projected_max_row[key] == value, f"Projected max row {key}")

    summary = {
        "original_reflected_mapping_pairs": 3,
        "source_reset_mappings": 6,
        "camera_configuration_pairs_verified_exactly_identical": (
            camera_pair_matches
        ),
        "world_object_rows": len(world_rows),
        "world_difference_components": len(world_rows) * len(WORLD_AXES),
        "world_exact_zero_components": sum(
            value == 0.0
            for row in world_rows
            for value in row[world_difference_name]
        ),
        "world_exact_zero_rows": sum(
            all(value == 0.0 for value in row[world_difference_name])
            for row in world_rows
        ),
        "world_sign_census": world_signs,
        "world_direct_and_independent_component_matches": (
            direct_world_component_matches
        ),
        "maximum_absolute_world_component_difference": world_max_component,
        "maximum_world_row_l2_difference": world_max_row,
        "projected_camera_object_rows": len(projected_rows),
        "projected_difference_components": (
            len(projected_rows) * len(PROJECTED_AXES)
        ),
        "projected_exact_zero_components": sum(
            value == 0.0
            for row in projected_rows
            for value in row[projected_difference_name]
        ),
        "projected_exact_zero_rows": sum(
            all(value == 0.0 for value in row[projected_difference_name])
            for row in projected_rows
        ),
        "projected_sign_census": projected_signs,
        "stored_actual_center_projection_formula_matches": stored_projection_matches,
        "projected_direct_and_independent_component_matches": (
            direct_projected_component_matches
        ),
        "maximum_absolute_projected_component_difference": (
            projected_max_component
        ),
        "maximum_projected_row_l2_difference": projected_max_row,
    }
    require(
        summary["world_difference_components"] == 27
        and summary["world_exact_zero_components"] == 10
        and summary["world_exact_zero_rows"] == 1
        and summary["projected_difference_components"] == 72
        and summary["projected_exact_zero_components"] == 0
        and summary["projected_exact_zero_rows"] == 0,
        "Representation-difference accounting changed",
    )
    return world_rows, projected_rows, summary


def audit() -> dict:
    turn2848, decomposition, source_pin = load_turn2848_package()
    world_rows, projected_rows, summary = analyze(turn2848, decomposition)
    return {
        "schema_version": (
            "wmf-retained-center-representation-residual-difference-"
            "diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2848_residual_decomposition_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_object_rows": 9,
            "world_difference_components": 27,
            "mapping_pair_camera_rows": 12,
            "projected_camera_object_rows": 36,
            "projected_difference_components": 72,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
            "D01_D02_command_confounded": True,
            "P00_shared_left_command_with_extra_reflected_right_member": True,
            "isolated_layout_effect_estimated": False,
        },
        "representation_semantics": {
            "world": (
                "Subtract the frozen actual-minus-constructed AABB-midpoint residual "
                "from the frozen actual-minus-constructed retained object-center "
                "residual. The independent identity is the reflected-arm center-minus-"
                "AABB-midpoint offset minus the constructed-reference equivalent."
            ),
            "projected": (
                "Project the retained actual object center and constructed reflected "
                "center point through the unchanged frozen camera configuration, form "
                "their residual, then subtract the frozen projected AABB-envelope-center "
                "residual. The independent identity compares point-minus-envelope-center "
                "offsets in the actual and constructed representations."
            ),
            "signs_and_units": (
                "World signs follow robot-base x/y/z and are stored in metres with exact "
                "1000x millimetre copies. Projected signs follow stored u/v pixel "
                "coordinates. Zero means exact float equality; no epsilon, threshold, "
                "normalization, fitting or registration is applied."
            ),
            "proxy_boundary": (
                "Retained object centers, AABB midpoints, projected center points and "
                "projected AABB-envelope centers are geometric representations. Neither "
                "projected representation is an observed or human-labeled image centroid, "
                "and this diagnostic does not decide which representation is preferable."
            ),
        },
        "summary": summary,
        "world_rows": world_rows,
        "projected_rows": projected_rows,
        "finding": (
            "Across three availability-selected original/reflected mapping pairs, nine "
            "world-object rows yield 27 exact center-versus-AABB-midpoint residual "
            "differences and 36 camera-object rows yield 72 projected point-center-versus-"
            "AABB-envelope-center residual differences. The largest absolute world "
            "component difference is +0.31600892543792725 mm on D02 cube x; the largest "
            "projected component difference is +0.5471362857456938 px on D02 cube wrist "
            "v. These representation-sensitivity descriptions do not validate an image "
            "centroid, estimate an effect or advance any scientific gate."
        ),
        "claim_boundaries": {
            "frozen_representation_sensitivity_only": (
                "This report compares deterministic representations inside the pinned "
                "Turn2848 package and adds no independent observation, model request, "
                "robot episode, image, label or scientific sample."
            ),
            "not_human_centroids": (
                "Projected point centers and projected AABB-envelope centers are not "
                "object-pixel measurements, masks or human image centroids."
            ),
            "descriptive_extrema_only": (
                "Sign counts and extrema are exact descriptions, not tolerances, quality "
                "scores, camera rankings, representation rankings or fixture decisions."
            ),
            "selection_confounding": (
                "All three pairs are availability-selected; D01/D02 are command-confounded "
                "and P00 has unequal command membership."
            ),
            "science_and_dispatch": (
                "No A1-A9 result, measurement qualification, stage advance, dispatch "
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
            "counterfactual_or_temporal_transition_inferred": False,
            "causal_effect_estimated": False,
            "isolated_layout_or_command_effect_estimated": False,
            "task_success_or_model_rate": False,
            "camera_ranking_estimated": False,
            "representation_ranking_estimated": False,
            "forecast_or_persistence_metric": False,
            "a1_a9_result": False,
            "current_state_inferred": False,
            "gate_advanced": False,
            "release_authority": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare retained point-center and AABB-center reflection residual "
            "representations without advancing science gates."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--check",
        type=Path,
        help="Verify an existing canonical diagnostic without modifying it",
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
            "Verified 9 world rows / 27 components and 36 projected rows / "
            "72 components; retained-only representation sensitivity and no stage advance."
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
