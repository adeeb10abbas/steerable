#!/usr/bin/env python3
"""Audit exact centered-scatter identities for retained residual triangles."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
TURN2853_PACKAGE_COMMIT = "8b07203927aebbec82fb730c1a7aeac2af46e088"
TURN2853_PACKAGE_TREE = "e5fdfd9f28c12beb7f95fb2ba884b8983c4b8fd9"
TURN2853_PACKAGE_PARENT = "29a9501b1c95e1d1b7a593ae5d33a8e055ae4a30"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2853_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_residual_triangle_edge_polarization.py"
    ),
    "lines": 840,
    "bytes": 34723,
    "sha256": "7c51bf3099db1bb25852ed773718eb68c2581e3202d747a0d7f3046c4acc1e2d",
}
TURN2853_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_residual_triangle_edge_polarization_diagnostic.json"
    ),
    "lines": 5301,
    "bytes": 188378,
    "sha256": "3ddb13226e877819629a4a560fe6e039d22ad425b86fac46a4e4848ce22538da",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_centered_scatter_diagnostic.json"
)

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
ORDERED_CONTEXTS = (
    ("development", "D01", "D1"),
    ("development", "D02", "D1"),
    ("pilot", "P00", "N3"),
)
ORDERED_CAMERAS = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
WORLD_PROVENANCE_DESCRIPTOR = {
    "bytes": 9573,
    "sha256": "5b5e133fce4deafd022f20200f44b38ca44c3fd7be3b8594d689f34bbf7a281c",
}
PROJECTED_PROVENANCE_DESCRIPTOR = {
    "bytes": 43458,
    "sha256": "4fe80cf34376cc28e9489f02e8bbf6eafe61855c440f5be6653122978c74fb21",
}
WORLD_PROVENANCE_KEYS = {
    "phase",
    "layout_pair_id",
    "model_id",
    "original_member_cell_ids",
    "original_member_commands",
    "original_representative_cell_id",
    "reflected_member_cell_ids",
    "reflected_member_commands",
    "reflected_representative_cell_id",
    "command_membership_classification",
    "source_package_commit",
    "object_order",
    "edge_ab",
    "edge_ac",
    "edge_bc",
    "source_pair_direction",
    "source_turn2850_package_commit",
}
PROJECTED_PROVENANCE_KEYS = WORLD_PROVENANCE_KEYS | {
    "camera_name",
    "camera_configuration_sha256",
}
SENSITIVE_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "private_key",
    "secret",
    "token",
)
SENSITIVE_VALUE_PATTERNS = (
    "-----BEGIN PRIVATE KEY-----",
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "Bearer ",
    "github_pat_",
    "ghp_",
    "AKIA",
)


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
    require(payload.count(b"\n") == descriptor["lines"], f"{label} line count changed")
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


def load_turn2853_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2853_PACKAGE_COMMIT}^{{commit}}")
        == TURN2853_PACKAGE_COMMIT,
        "Pinned Turn2853 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2853_PACKAGE_COMMIT)
        == TURN2853_PACKAGE_TREE,
        "Pinned Turn2853 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2853_PACKAGE_COMMIT)
        == TURN2853_PACKAGE_PARENT,
        "Pinned Turn2853 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2853_PACKAGE_COMMIT, TURN2853_AUDIT["path"])
    verify_payload(committed_audit, TURN2853_AUDIT, "Pinned Turn2853 audit")
    local_audit_path = REPOSITORY / TURN2853_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2853_AUDIT, "Local Turn2853 audit")
    require(local_audit == committed_audit, "Local Turn2853 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2853_edge_polarization")

    committed_report = git_bytes(TURN2853_PACKAGE_COMMIT, TURN2853_REPORT["path"])
    verify_payload(committed_report, TURN2853_REPORT, "Pinned Turn2853 report")
    local_report = (REPOSITORY / TURN2853_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2853_REPORT, "Local Turn2853 report")
    require(local_report == committed_report, "Local Turn2853 report differs")
    parsed = strict_json_loads(committed_report, TURN2853_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2853 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2853 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-residual-triangle-edge-polarization-diagnostic-v1",
        "Turn2853 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_source_triangle_rows"] == 9
        and selection["projected_source_triangle_rows"] == 36
        and selection["world_edge_polarization_rows"] == 9
        and selection["projected_edge_polarization_rows"] == 36
        and selection["availability_selected"] is True
        and selection["outcome_dependent_selection"] is False
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"]
        is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2853 selection boundary changed",
    )
    summary = parsed["summary"]
    require(
        summary["world_identity_rows"] == 9
        and summary["projected_identity_rows"] == 36
        and summary["total_exact_edge_cycle_and_polarization_rows"] == 45
        and summary["total_exact_cycle_components"] == 99
        and summary["total_exact_polarization_identities"] == 135
        and summary["turn2852_exact_rational_scalar_records_revalidated"] == 180
        and summary["total_serialized_exact_rational_records"] == 504
        and summary["all_exact_edge_cycles_zero"] is True
        and summary["all_exact_polarization_residuals_zero"] is True,
        "Turn2853 exact summary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 46
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2853 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2852"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["physical_distance_or_length_measured"] is False
        and parsed["physical_scene_triangle_area_measured"] is False
        and parsed["image_footprint_or_silhouette_area_measured"] is False
        and parsed["physical_plane_normal_inferred"] is False
        and parsed["orientation_or_chirality_inferred"] is False
        and parsed["collinearity_or_degeneracy_classified"] is False
        and parsed["triangle_inequality_or_metric_classified"] is False
        and parsed["angle_or_similarity_inferred"] is False
        and parsed["coordinate_or_frame_invariance_claimed"] is False
        and parsed["normalization_applied"] is False
        and parsed["scientific_threshold_applied"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False,
        "Turn2853 direct boundary changed",
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
        "Turn2853 zero-activity boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2853_PACKAGE_COMMIT,
        "tree": TURN2853_PACKAGE_TREE,
        "parent": TURN2853_PACKAGE_PARENT,
        "audit": TURN2853_AUDIT,
        "report": TURN2853_REPORT,
        "canonical_regeneration_verified": True,
        "inherited_turn2852_package": parsed["turn2852_gram_identity_package"],
    }


def finite_number(value: object, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} is not numeric",
    )
    result = float(value)
    require(math.isfinite(result), f"{label} is not finite")
    return result


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape changed")
    return [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def exact_vector(values: list[float]) -> list[Fraction]:
    return [Fraction.from_float(value) for value in values]


def vector_add(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    require(len(first) == len(second), "Vector-add shape changed")
    return [left + right for left, right in zip(first, second)]


def vector_subtract(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    require(len(first) == len(second), "Vector-subtract shape changed")
    return [left - right for left, right in zip(first, second)]


def vector_scale(value: Fraction, vector: list[Fraction]) -> list[Fraction]:
    return [value * component for component in vector]


def dot(first: list[Fraction], second: list[Fraction]) -> Fraction:
    require(len(first) == len(second), "Dot-product shape changed")
    return sum((left * right for left, right in zip(first, second)), Fraction(0))


def exact_record(value: Fraction, approximate_key: str) -> dict:
    approximate = float(value)
    require(math.isfinite(approximate), "Exact-record display approximation is non-finite")
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        approximate_key: approximate,
    }


def rational_from_record(record: object, approximate_key: str, label: str) -> Fraction:
    require(isinstance(record, dict), f"{label} record missing")
    require(
        set(record) == {"numerator", "denominator", approximate_key},
        f"{label} record keys changed",
    )
    numerator = record.get("numerator")
    denominator = record.get("denominator")
    require(isinstance(numerator, str), f"{label} numerator changed")
    require(isinstance(denominator, str), f"{label} denominator changed")
    approximate = finite_number(record.get(approximate_key), f"{label} approximation")
    value = Fraction(int(numerator), int(denominator))
    require(approximate == float(value), f"{label} approximation changed")
    return value


def validate_provenance(value: object, projected: bool) -> dict:
    require(isinstance(value, dict), "Provenance missing")
    expected = PROJECTED_PROVENANCE_KEYS if projected else WORLD_PROVENANCE_KEYS
    require(set(value) == expected, "Provenance keys changed")
    require(
        value["object_order"] == ["banana", "bowl", "rubiks_cube"],
        "Object order changed",
    )
    require(
        value["edge_ab"] == "banana_minus_bowl"
        and value["edge_ac"] == "banana_minus_rubiks_cube"
        and value["edge_bc"] == "bowl_minus_rubiks_cube",
        "Edge definitions changed",
    )
    require(value["source_pair_direction"] == "first_minus_second", "Pair direction changed")
    require(
        value["source_turn2850_package_commit"]
        == "6cec04b0ff7bc98cec039b456b647390aa7b4acd",
        "Turn2850 provenance pin changed",
    )
    return value


def row_key(row: dict, projected: bool) -> tuple:
    provenance = validate_provenance(row.get("provenance"), projected)
    values = [
        provenance["phase"],
        provenance["layout_pair_id"],
        provenance["model_id"],
    ]
    if projected:
        values.extend((provenance["camera_name"], provenance["camera_configuration_sha256"]))
    values.append(row.get("family"))
    return tuple(values)


def row_order_key(row: dict, projected: bool) -> tuple:
    provenance = validate_provenance(row.get("provenance"), projected)
    values = [
        provenance["phase"],
        provenance["layout_pair_id"],
        provenance["model_id"],
    ]
    if projected:
        values.append(provenance["camera_name"])
    values.append(row.get("family"))
    return tuple(values)


def expected_row_order(projected: bool) -> list[tuple]:
    if projected:
        return [
            (*context, camera, family)
            for context in ORDERED_CONTEXTS
            for camera in ORDERED_CAMERAS
            for family in PROJECTED_FAMILIES
        ]
    return [
        (*context, family)
        for context in ORDERED_CONTEXTS
        for family in WORLD_FAMILIES
    ]


def validate_predecessor_exact_records(
    row: dict,
    edge_ab: list[Fraction],
    edge_ac: list[Fraction],
    edge_bc: list[Fraction],
    unit: str,
) -> int:
    approximate_coordinate = f"approximate_{unit}"
    cycle = row.get(f"exact_cycle_ab_plus_bc_minus_ac_{unit}")
    require(isinstance(cycle, list) and len(cycle) == len(edge_ab), "Predecessor cycle shape changed")
    expected_cycle = vector_subtract(vector_add(edge_ab, edge_bc), edge_ac)
    for index, (record, expected) in enumerate(zip(cycle, expected_cycle)):
        require(
            rational_from_record(record, approximate_coordinate, f"Predecessor cycle {index}")
            == expected,
            "Predecessor exact cycle record changed",
        )
    require(expected_cycle == [Fraction(0)] * len(edge_ab), "Predecessor exact cycle is nonzero")

    ab2 = dot(edge_ab, edge_ab)
    ac2 = dot(edge_ac, edge_ac)
    bc2 = dot(edge_bc, edge_bc)
    ab_ac = dot(edge_ab, edge_ac)
    ab_bc = dot(edge_ab, edge_bc)
    ac_bc = dot(edge_ac, edge_bc)
    expected_scalars = {
        f"edge_ab_squared_norm_{unit}2": ab2,
        f"edge_ac_squared_norm_{unit}2": ac2,
        f"edge_bc_squared_norm_{unit}2": bc2,
        f"edge_ab_dot_edge_ac_{unit}2": ab_ac,
        f"edge_ab_dot_edge_bc_{unit}2": ab_bc,
        f"edge_ac_dot_edge_bc_{unit}2": ac_bc,
        f"polarization_identity_ab_ac_residual_{unit}2": 2 * ab_ac - (ab2 + ac2 - bc2),
        f"polarization_identity_ab_bc_residual_{unit}2": 2 * ab_bc - (ac2 - ab2 - bc2),
        f"polarization_identity_ac_bc_residual_{unit}2": 2 * ac_bc - (ac2 + bc2 - ab2),
    }
    scalars = row.get("exact_stored_binary_float_rational_scalars")
    require(isinstance(scalars, dict), "Predecessor exact scalar map missing")
    require(set(scalars) == set(expected_scalars), "Predecessor exact scalar keys changed")
    for key, expected in expected_scalars.items():
        require(
            rational_from_record(scalars[key], f"approximate_{unit}2", f"Predecessor {key}")
            == expected,
            f"Predecessor {key} changed",
        )
    require(
        all(expected_scalars[key] == 0 for key in expected_scalars if "residual" in key),
        "Predecessor polarization residual is nonzero",
    )
    require(
        row.get("exact_edge_cycle_identity") is True
        and row.get("all_three_exact_polarization_identities") is True
        and row.get("fixed_object_and_operand_order") is True
        and row.get("normalization_applied") is False
        and row.get("scientific_threshold_applied") is False,
        "Predecessor row boundary flag changed",
    )
    return len(cycle) + len(expected_scalars)


def centered_scatter_row(row: dict, projected: bool) -> tuple[dict, int]:
    unit = "px" if projected else "m"
    length = 2 if projected else 3
    families = PROJECTED_FAMILIES if projected else WORLD_FAMILIES
    provenance = validate_provenance(row.get("provenance"), projected)
    family = row.get("family")
    require(family in families, "Family changed")

    edge_ab_float = finite_vector(row.get(f"edge_ab_signed_{unit}"), length, "Edge AB")
    edge_ac_float = finite_vector(row.get(f"edge_ac_signed_{unit}"), length, "Edge AC")
    edge_bc_float = finite_vector(row.get(f"edge_bc_signed_{unit}"), length, "Edge BC")
    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_bc = exact_vector(edge_bc_float)
    predecessor_records = validate_predecessor_exact_records(
        row, edge_ab, edge_ac, edge_bc, unit
    )

    one_third = Fraction(1, 3)
    a_centered = vector_scale(one_third, vector_add(edge_ab, edge_ac))
    b_centered = vector_scale(
        one_third,
        vector_add(vector_scale(Fraction(-2), edge_ab), edge_ac),
    )
    c_centered = vector_scale(
        one_third,
        vector_add(edge_ab, vector_scale(Fraction(-2), edge_ac)),
    )
    centered_sum = vector_add(vector_add(a_centered, b_centered), c_centered)
    require(centered_sum == [Fraction(0)] * length, "Exact centered sum changed")

    reconstructed = {
        "edge_ab": vector_subtract(a_centered, b_centered),
        "edge_ac": vector_subtract(a_centered, c_centered),
        "edge_bc": vector_subtract(b_centered, c_centered),
    }
    source_edges = {"edge_ab": edge_ab, "edge_ac": edge_ac, "edge_bc": edge_bc}
    edge_residuals = {
        name: vector_subtract(reconstructed[name], source_edges[name])
        for name in ("edge_ab", "edge_ac", "edge_bc")
    }
    require(
        all(residual == [Fraction(0)] * length for residual in edge_residuals.values()),
        "Exact pairwise edge reconstruction changed",
    )

    a_scatter = dot(a_centered, a_centered)
    b_scatter = dot(b_centered, b_centered)
    c_scatter = dot(c_centered, c_centered)
    centered_scatter_sum = a_scatter + b_scatter + c_scatter
    edge_squared_norm_sum = (
        dot(edge_ab, edge_ab) + dot(edge_ac, edge_ac) + dot(edge_bc, edge_bc)
    )
    one_third_edge_squared_norm_sum = one_third * edge_squared_norm_sum
    huygens_residual = centered_scatter_sum - one_third_edge_squared_norm_sum
    require(huygens_residual == 0, "Exact Huygens identity changed")

    coordinate_key = f"approximate_{unit}"
    squared_key = f"approximate_{unit}2"
    centered_coordinates = {
        "a_centered": [exact_record(value, coordinate_key) for value in a_centered],
        "b_centered": [exact_record(value, coordinate_key) for value in b_centered],
        "c_centered": [exact_record(value, coordinate_key) for value in c_centered],
    }
    exact_scalars = {
        f"a_centered_squared_norm_{unit}2": exact_record(a_scatter, squared_key),
        f"b_centered_squared_norm_{unit}2": exact_record(b_scatter, squared_key),
        f"c_centered_squared_norm_{unit}2": exact_record(c_scatter, squared_key),
        f"centered_scatter_sum_{unit}2": exact_record(centered_scatter_sum, squared_key),
        f"edge_squared_norm_sum_{unit}2": exact_record(edge_squared_norm_sum, squared_key),
        f"one_third_edge_squared_norm_sum_{unit}2": exact_record(
            one_third_edge_squared_norm_sum, squared_key
        ),
        f"huygens_identity_residual_{unit}2": exact_record(huygens_residual, squared_key),
    }
    output = {
        "provenance": provenance,
        "family": family,
        f"source_edge_ab_signed_{unit}": edge_ab_float,
        f"source_edge_ac_signed_{unit}": edge_ac_float,
        f"source_edge_bc_signed_{unit}": edge_bc_float,
        f"exact_centered_coordinates_{unit}": centered_coordinates,
        f"exact_centered_sum_residual_{unit}": [
            exact_record(value, coordinate_key) for value in centered_sum
        ],
        f"exact_pairwise_edge_reconstruction_residuals_{unit}": {
            name: [exact_record(value, coordinate_key) for value in edge_residuals[name]]
            for name in ("edge_ab", "edge_ac", "edge_bc")
        },
        "exact_stored_binary_float_rational_scalars": exact_scalars,
        "exact_centered_sum_identity": True,
        "exact_pairwise_edge_reconstruction": True,
        "exact_huygens_identity": True,
        "fixed_object_and_operand_order": True,
        "normalization_applied": False,
        "square_root_applied": False,
        "eigendecomposition_applied": False,
        "scientific_threshold_applied": False,
    }
    return output, predecessor_records


def analyze(source: dict) -> tuple[list[dict], list[dict], dict]:
    world_source = source.get("world_edge_polarization_rows")
    projected_source = source.get("projected_edge_polarization_rows")
    require(isinstance(world_source, list) and len(world_source) == 9, "World source rows changed")
    require(
        isinstance(projected_source, list) and len(projected_source) == 36,
        "Projected source rows changed",
    )
    world_provenance = canonical_bytes([row.get("provenance") for row in world_source])
    projected_provenance = canonical_bytes(
        [row.get("provenance") for row in projected_source]
    )
    require(
        len(world_provenance) == WORLD_PROVENANCE_DESCRIPTOR["bytes"]
        and hashlib.sha256(world_provenance).hexdigest()
        == WORLD_PROVENANCE_DESCRIPTOR["sha256"],
        "World provenance payload changed",
    )
    require(
        len(projected_provenance) == PROJECTED_PROVENANCE_DESCRIPTOR["bytes"]
        and hashlib.sha256(projected_provenance).hexdigest()
        == PROJECTED_PROVENANCE_DESCRIPTOR["sha256"],
        "Projected provenance payload changed",
    )
    world_keys = [row_key(row, False) for row in world_source]
    projected_keys = [row_key(row, True) for row in projected_source]
    require(
        [row_order_key(row, False) for row in world_source] == expected_row_order(False),
        "World source row order changed",
    )
    require(
        [row_order_key(row, True) for row in projected_source]
        == expected_row_order(True),
        "Projected source row order changed",
    )
    require(len(set(world_keys)) == 9, "Duplicate world source row")
    require(len(set(projected_keys)) == 36, "Duplicate projected source row")

    world_rows = []
    projected_rows = []
    predecessor_records = 0
    for row in world_source:
        output, records = centered_scatter_row(row, False)
        world_rows.append(output)
        predecessor_records += records
    for row in projected_source:
        output, records = centered_scatter_row(row, True)
        projected_rows.append(output)
        predecessor_records += records
    require(predecessor_records == 504, "Predecessor exact record count changed")

    world_records = 9 * (9 + 3 + 9 + 7)
    projected_records = 36 * (6 + 2 + 6 + 7)
    summary = {
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_identity_rows": 9,
        "projected_identity_rows": 36,
        "total_centered_scatter_identity_rows": 45,
        "turn2853_exact_rational_records_revalidated": predecessor_records,
        "exact_centered_coordinate_components": 297,
        "exact_centered_sum_residual_components": 99,
        "exact_pairwise_edge_reconstruction_residual_components": 297,
        "exact_huygens_identities": 45,
        "world_serialized_exact_rational_records": world_records,
        "projected_serialized_exact_rational_records": projected_records,
        "total_serialized_exact_rational_records": world_records + projected_records,
        "all_exact_centered_sums_zero": True,
        "all_exact_pairwise_edge_reconstruction_residuals_zero": True,
        "all_exact_huygens_residuals_zero": True,
    }
    require(world_records == 252, "World exact record count changed")
    require(projected_records == 756, "Projected exact record count changed")
    return world_rows, projected_rows, summary


def assert_output_hygiene(value: object) -> None:
    def visit(item: object, location: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
                require(
                    not any(part in normalized for part in SENSITIVE_KEY_PARTS),
                    f"Sensitive key forbidden at {location}.{key}",
                )
                visit(child, f"{location}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{location}[{index}]")
        elif isinstance(item, str):
            require("\x00" not in item, f"NUL forbidden at {location}")
            require(
                not item.startswith(("/", "~", "file://", "ssh://")),
                f"Absolute/raw path forbidden at {location}",
            )
            require(not re.match(r"^[A-Za-z]:[\\/]", item), f"Drive path forbidden at {location}")
            require("/raw/" not in item and "\\raw\\" not in item, f"Raw path forbidden at {location}")
            require(
                not any(pattern in item for pattern in SENSITIVE_VALUE_PATTERNS),
                f"Secret-like value forbidden at {location}",
            )
            require(
                re.search(r"(?:^|[^A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}", item) is None,
                f"Secret-like value forbidden at {location}",
            )

    visit(value, "$report")


def audit() -> dict:
    _, source, source_pin = load_turn2853_package()
    world_rows, projected_rows, summary = analyze(source)
    report = {
        "schema_version": "wmf-retained-residual-triangle-centered-scatter-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "turn2853_edge_polarization_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_source_edge_rows": 9,
            "projected_source_edge_rows": 36,
            "world_centered_scatter_rows": 9,
            "projected_centered_scatter_rows": 36,
            "availability_selected": True,
            "outcome_dependent_selection": False,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
            "D01_D02_command_confounded": True,
            "P00_shared_left_command_with_extra_reflected_right_member": True,
            "isolated_layout_or_command_effect_estimated": False,
        },
        "construction_semantics": {
            "stored_coordinate_domain": (
                "Each finite source edge coordinate is interpreted as its exact binary "
                "floating-point rational value. Integer numerator/denominator strings "
                "control equality; approximate numbers are display aids only."
            ),
            "fixed_order": (
                "A=banana,B=bowl,C=rubiks_cube with eAB=A-B,eAC=A-C,eBC=B-C. "
                "No object, operand, coordinate, camera or source-row reordering occurs."
            ),
            "centered_coordinate_formulas": (
                "The exact algebraic centroid origin gives a_c=(eAB+eAC)/3, "
                "b_c=(-2*eAB+eAC)/3 and c_c=(eAB-2*eAC)/3."
            ),
            "edge_reconstruction": (
                "The audit verifies a_c-b_c=eAB,a_c-c_c=eAC and b_c-c_c=eBC "
                "component by component over exact rationals."
            ),
            "huygens_formula": (
                "The Huygens residual is sum(||a_c||^2,||b_c||^2,||c_c||^2) "
                "- (||eAB||^2+||eAC||^2+||eBC||^2)/3."
            ),
            "units_and_scope": (
                "World coordinate/scalar records use stored m/m^2 units; projected "
                "records use stored px/px^2 units. These are residual-vector-space "
                "coordinate identities, not physical or image-object measurements."
            ),
            "excluded_operations": (
                "No square root, eigendecomposition, normalization, tolerance, threshold, "
                "classification, fitting, registration or ranking is performed."
            ),
        },
        "summary": summary,
        "world_centered_scatter_rows": world_rows,
        "projected_centered_scatter_rows": projected_rows,
        "algebraically_derived_from_turn2853": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "observed_centroid_or_center_estimated": False,
        "physical_scatter_or_variance_measured": False,
        "statistical_variance_or_population_moment_estimated": False,
        "translation_invariance_claimed": False,
        "physical_distance_or_length_measured": False,
        "physical_scene_triangle_area_measured": False,
        "image_footprint_or_silhouette_area_measured": False,
        "principal_axes_or_eigenvalues_computed": False,
        "physical_plane_normal_inferred": False,
        "orientation_or_chirality_inferred": False,
        "collinearity_or_degeneracy_classified": False,
        "triangle_inequality_or_metric_classified": False,
        "angle_or_similarity_inferred": False,
        "coordinate_or_frame_invariance_claimed": False,
        "normalization_applied": False,
        "square_root_applied": False,
        "eigendecomposition_applied": False,
        "scientific_threshold_applied": False,
        "a7_result": False,
        "r0_rhat_r_endpoint": False,
        "motion_or_temporal_transition_inferred": False,
        "generated_or_executed_correspondence": False,
        "included_in_existing_six_report_annotation_closure": False,
        "finding": (
            "All 9 world and 36 projected retained residual-triangle rows have exact "
            "zero centered-coordinate sums, exact pairwise edge reconstruction and zero "
            "Huygens residual. The package serializes 1,008 exact rational records and "
            "revalidates all 504 Turn2853 exact records. This is a deterministic stored-"
            "coordinate algebra check only, not physical or scientific evidence."
        ),
        "claim_boundaries": {
            "exactness_domain": (
                "Exact means equality over rational interpretations of pinned finite "
                "binary floating-point values. It does not authenticate source precision, "
                "physical truth, uncertainty or measurement accuracy."
            ),
            "constructed_center_not_observed_centroid": (
                "The centered coordinates use only an algebraic origin reconstructed from "
                "edge vectors. They are not observed or human-labeled object centroids, "
                "scene centers, camera centers or recovered physical positions."
            ),
            "scatter_not_physical_variance_or_inertia": (
                "Squared coordinate norms and their sum describe frozen residual vectors. "
                "They are not physical scatter, variance, inertia, distance, length, image "
                "footprint, silhouette, visible area or metric quality; no statistical "
                "variance or population moment is estimated."
            ),
            "no_invariance_or_spectral_claim": (
                "No square root or eigendecomposition is performed. No coordinate/frame "
                "or translation invariance, principal direction, eigenvalue, conditioning, "
                "degeneracy, angle, similarity, acceptance or representation ranking is "
                "claimed."
            ),
            "constructed_reference_only": (
                "The inherited constructed pure-reflection correspondent remains a "
                "deterministic reference, not recovered fixture truth, an observed image, "
                "a counterfactual rollout or an independent measurement."
            ),
            "not_primary_endpoint": (
                "Rows have no request-start, generated-horizon or executed-time "
                "correspondence. They are not r0,rhat,r and answer neither A7 nor any "
                "A1-A9 question."
            ),
            "selection_and_confounding": (
                "The three mapping pairs remain availability-selected. D01/D02 are "
                "command-confounded and P00 has unequal command membership. No layout, "
                "command, model, task-success, population, representation or camera "
                "effect is estimated."
            ),
            "science_annotation_and_release": (
                "No raw bytes, labels, manifest, annotation eligibility, inference, "
                "forecast metric, current-state claim, stage gate, dispatch authority or "
                "confirmation release follows."
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
        "output_hygiene": {
            "absolute_or_raw_paths_included": False,
            "sensitive_material_included": False,
        },
        "evidence_boundaries": {
            "raw_camera_bytes_read": False,
            "raw_pvc_bytes_read": False,
            "rgb_or_image_content_analyzed": False,
            "observed_centroid_or_center_estimated": False,
            "physical_scatter_or_variance_measured": False,
            "statistical_variance_or_population_moment_estimated": False,
            "translation_invariance_claimed": False,
            "principal_axes_or_eigenvalues_computed": False,
            "physical_distance_or_length_measured": False,
            "physical_scene_triangle_area_measured": False,
            "image_footprint_or_object_pixels_inferred": False,
            "image_footprint_or_silhouette_area_measured": False,
            "silhouette_or_segmentation_analyzed": False,
            "obb_or_exact_mesh_analyzed": False,
            "physical_plane_normal_inferred": False,
            "orientation_or_chirality_inferred": False,
            "collinearity_or_degeneracy_classified": False,
            "triangle_inequality_or_metric_classified": False,
            "angle_or_similarity_inferred": False,
            "coordinate_or_frame_invariance_claimed": False,
            "human_image_centroids_used": False,
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
            "square_root_applied": False,
            "eigendecomposition_applied": False,
            "cross_time_field_join_performed": False,
            "counterfactual_or_temporal_transition_inferred": False,
            "motion_or_temporal_transition_inferred": False,
            "generated_or_executed_correspondence": False,
            "causal_effect_estimated": False,
            "isolated_layout_or_command_effect_estimated": False,
            "task_success_or_model_rate": False,
            "camera_ranking_estimated": False,
            "representation_ranking_estimated": False,
            "edge_or_metric_ranking_estimated": False,
            "forecast_or_persistence_metric": False,
            "r0_rhat_r_endpoint": False,
            "a1_a9_result": False,
            "a7_result": False,
            "independent_observation": False,
            "new_scientific_sample": False,
            "current_state_inferred": False,
            "gate_advanced": False,
            "release_authority": False,
        },
    }
    require(
        len(report["evidence_boundaries"]) == 53
        and all(value is False for value in report["evidence_boundaries"].values()),
        "Output evidence boundary changed",
    )
    assert_output_hygiene(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact centered-scatter identities for retained residual triangles "
            "without advancing science gates."
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
    report = audit()
    encoded = canonical_bytes(report)
    if arguments.check is not None:
        require(arguments.check.resolve() == OUTPUT_PATH.resolve(), "Unexpected check path")
        require(arguments.check.read_bytes() == encoded, "Saved diagnostic differs")
        print(
            "Verified exact centered sums, pairwise edge reconstruction and Huygens "
            "identities for 9 world and 36 projected retained residual-triangle rows; "
            "1,008 exact records and no stage advance."
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
