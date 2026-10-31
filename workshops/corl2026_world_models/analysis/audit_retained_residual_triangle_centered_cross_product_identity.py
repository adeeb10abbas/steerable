#!/usr/bin/env python3
"""Audit exact centered cross-product identities for retained residual triangles."""

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
TURN2854_PACKAGE_COMMIT = "be873e2f680b387639a0546e34c98c90cd0a3232"
TURN2854_PACKAGE_TREE = "a8499f3f5fa4658e2c12b8d9836b655c9afdf3a3"
TURN2854_PACKAGE_PARENT = "38fffe509d4ed4e26bfef2f5111556dc43cfebd3"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2854_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_residual_triangle_centered_scatter.py"
    ),
    "lines": 975,
    "bytes": 40713,
    "sha256": "7cfd34eeec39dad99cc4c8971a9cd2b5befb249597dababa6ca0c2de5ea7c782",
}
TURN2854_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_residual_triangle_centered_scatter_diagnostic.json"
    ),
    "lines": 8709,
    "bytes": 279933,
    "sha256": "2fd1cbf76108e311d7b462fe8e709e8203ff650848fd94c5ce9e9e74d1d2b942",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_centered_cross_product_identity_diagnostic.json"
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
WORLD_COMPONENTS = (
    ("xx", 0, 0),
    ("xy", 0, 1),
    ("xz", 0, 2),
    ("yy", 1, 1),
    ("yz", 1, 2),
    ("zz", 2, 2),
)
PROJECTED_COMPONENTS = (
    ("xx", 0, 0),
    ("xy", 0, 1),
    ("yy", 1, 1),
)
WORLD_PROVENANCE_DESCRIPTOR = {
    "bytes": 9573,
    "sha256": "5b5e133fce4deafd022f20200f44b38ca44c3fd7be3b8594d689f34bbf7a281c",
}
PROJECTED_PROVENANCE_DESCRIPTOR = {
    "bytes": 43458,
    "sha256": "4fe80cf34376cc28e9489f02e8bbf6eafe61855c440f5be6653122978c74fb21",
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


def load_turn2854_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2854_PACKAGE_COMMIT}^{{commit}}")
        == TURN2854_PACKAGE_COMMIT,
        "Pinned Turn2854 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2854_PACKAGE_COMMIT)
        == TURN2854_PACKAGE_TREE,
        "Pinned Turn2854 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2854_PACKAGE_COMMIT)
        == TURN2854_PACKAGE_PARENT,
        "Pinned Turn2854 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2854_PACKAGE_COMMIT, TURN2854_AUDIT["path"])
    verify_payload(committed_audit, TURN2854_AUDIT, "Pinned Turn2854 audit")
    local_audit_path = REPOSITORY / TURN2854_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2854_AUDIT, "Local Turn2854 audit")
    require(local_audit == committed_audit, "Local Turn2854 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2854_centered_scatter")

    committed_report = git_bytes(TURN2854_PACKAGE_COMMIT, TURN2854_REPORT["path"])
    verify_payload(committed_report, TURN2854_REPORT, "Pinned Turn2854 report")
    local_report = (REPOSITORY / TURN2854_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2854_REPORT, "Local Turn2854 report")
    require(local_report == committed_report, "Local Turn2854 report differs")
    parsed = strict_json_loads(committed_report, TURN2854_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2854 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2854 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-residual-triangle-centered-scatter-diagnostic-v1",
        "Turn2854 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_source_edge_rows"] == 9
        and selection["projected_source_edge_rows"] == 36
        and selection["world_centered_scatter_rows"] == 9
        and selection["projected_centered_scatter_rows"] == 36
        and selection["availability_selected"] is True
        and selection["outcome_dependent_selection"] is False
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"]
        is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2854 selection boundary changed",
    )
    summary = parsed["summary"]
    require(
        summary["world_identity_rows"] == 9
        and summary["projected_identity_rows"] == 36
        and summary["total_centered_scatter_identity_rows"] == 45
        and summary["turn2853_exact_rational_records_revalidated"] == 504
        and summary["exact_centered_coordinate_components"] == 297
        and summary["exact_centered_sum_residual_components"] == 99
        and summary["exact_pairwise_edge_reconstruction_residual_components"] == 297
        and summary["exact_huygens_identities"] == 45
        and summary["total_serialized_exact_rational_records"] == 1008
        and summary["all_exact_centered_sums_zero"] is True
        and summary["all_exact_pairwise_edge_reconstruction_residuals_zero"] is True
        and summary["all_exact_huygens_residuals_zero"] is True,
        "Turn2854 exact summary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 53
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2854 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2853"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["observed_centroid_or_center_estimated"] is False
        and parsed["physical_scatter_or_variance_measured"] is False
        and parsed["statistical_variance_or_population_moment_estimated"] is False
        and parsed["translation_invariance_claimed"] is False
        and parsed["principal_axes_or_eigenvalues_computed"] is False
        and parsed["coordinate_or_frame_invariance_claimed"] is False
        and parsed["normalization_applied"] is False
        and parsed["square_root_applied"] is False
        and parsed["eigendecomposition_applied"] is False
        and parsed["scientific_threshold_applied"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False,
        "Turn2854 direct boundary changed",
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
        "Turn2854 zero-activity boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2854_PACKAGE_COMMIT,
        "tree": TURN2854_PACKAGE_TREE,
        "parent": TURN2854_PACKAGE_PARENT,
        "audit": TURN2854_AUDIT,
        "report": TURN2854_REPORT,
        "canonical_regeneration_verified": True,
        "inherited_turn2853_package": parsed["turn2853_edge_polarization_package"],
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
    try:
        value = Fraction(int(numerator), int(denominator))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError(f"{label} rational encoding changed") from error
    require(approximate == float(value), f"{label} approximation changed")
    return value


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


def row_key(prior: ModuleType, row: dict, projected: bool) -> tuple:
    provenance = prior.validate_provenance(row.get("provenance"), projected)
    values = [
        provenance["phase"],
        provenance["layout_pair_id"],
        provenance["model_id"],
    ]
    if projected:
        values.extend((provenance["camera_name"], provenance["camera_configuration_sha256"]))
    values.append(row.get("family"))
    return tuple(values)


def row_order_key(prior: ModuleType, row: dict, projected: bool) -> tuple:
    provenance = prior.validate_provenance(row.get("provenance"), projected)
    values = [
        provenance["phase"],
        provenance["layout_pair_id"],
        provenance["model_id"],
    ]
    if projected:
        values.append(provenance["camera_name"])
    values.append(row.get("family"))
    return tuple(values)


def validate_vector_records(
    records: object,
    expected: list[Fraction],
    approximate_key: str,
    label: str,
) -> int:
    require(isinstance(records, list) and len(records) == len(expected), f"{label} shape changed")
    for index, (record, value) in enumerate(zip(records, expected)):
        require(
            rational_from_record(record, approximate_key, f"{label}[{index}]") == value,
            f"{label}[{index}] changed",
        )
    return len(expected)


def validate_turn2854_row(
    prior: ModuleType,
    row: dict,
    projected: bool,
) -> tuple[dict, list[list[Fraction]], dict[str, list[Fraction]], int]:
    unit = "px" if projected else "m"
    length = 2 if projected else 3
    families = PROJECTED_FAMILIES if projected else WORLD_FAMILIES
    provenance = prior.validate_provenance(row.get("provenance"), projected)
    family = row.get("family")
    require(family in families, "Family changed")

    edge_ab_float = finite_vector(row.get(f"source_edge_ab_signed_{unit}"), length, "Edge AB")
    edge_ac_float = finite_vector(row.get(f"source_edge_ac_signed_{unit}"), length, "Edge AC")
    edge_bc_float = finite_vector(row.get(f"source_edge_bc_signed_{unit}"), length, "Edge BC")
    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_bc = exact_vector(edge_bc_float)
    require(vector_subtract(edge_ac, edge_ab) == edge_bc, "Source exact edge cycle changed")

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
    centered = [a_centered, b_centered, c_centered]
    centered_sum = vector_add(vector_add(a_centered, b_centered), c_centered)
    require(centered_sum == [Fraction(0)] * length, "Exact centered sum changed")
    edges = {"edge_ab": edge_ab, "edge_ac": edge_ac, "edge_bc": edge_bc}
    edge_residuals = {
        "edge_ab": vector_subtract(vector_subtract(a_centered, b_centered), edge_ab),
        "edge_ac": vector_subtract(vector_subtract(a_centered, c_centered), edge_ac),
        "edge_bc": vector_subtract(vector_subtract(b_centered, c_centered), edge_bc),
    }
    require(
        all(value == [Fraction(0)] * length for value in edge_residuals.values()),
        "Exact edge reconstruction changed",
    )

    coordinate_key = f"approximate_{unit}"
    squared_key = f"approximate_{unit}2"
    records = 0
    centered_records = row.get(f"exact_centered_coordinates_{unit}")
    require(
        isinstance(centered_records, dict)
        and set(centered_records) == {"a_centered", "b_centered", "c_centered"},
        "Predecessor centered-coordinate keys changed",
    )
    for name, expected in zip(("a_centered", "b_centered", "c_centered"), centered):
        records += validate_vector_records(
            centered_records[name], expected, coordinate_key, f"Predecessor {name}"
        )
    records += validate_vector_records(
        row.get(f"exact_centered_sum_residual_{unit}"),
        centered_sum,
        coordinate_key,
        "Predecessor centered sum",
    )
    residual_records = row.get(f"exact_pairwise_edge_reconstruction_residuals_{unit}")
    require(
        isinstance(residual_records, dict) and set(residual_records) == set(edge_residuals),
        "Predecessor edge-residual keys changed",
    )
    for name in ("edge_ab", "edge_ac", "edge_bc"):
        records += validate_vector_records(
            residual_records[name],
            edge_residuals[name],
            coordinate_key,
            f"Predecessor {name} residual",
        )

    centered_squared = [dot(value, value) for value in centered]
    centered_scatter_sum = sum(centered_squared, Fraction(0))
    edge_squared_norm_sum = sum((dot(value, value) for value in edges.values()), Fraction(0))
    one_third_edge_squared_norm_sum = one_third * edge_squared_norm_sum
    expected_scalars = {
        f"a_centered_squared_norm_{unit}2": centered_squared[0],
        f"b_centered_squared_norm_{unit}2": centered_squared[1],
        f"c_centered_squared_norm_{unit}2": centered_squared[2],
        f"centered_scatter_sum_{unit}2": centered_scatter_sum,
        f"edge_squared_norm_sum_{unit}2": edge_squared_norm_sum,
        f"one_third_edge_squared_norm_sum_{unit}2": one_third_edge_squared_norm_sum,
        f"huygens_identity_residual_{unit}2": (
            centered_scatter_sum - one_third_edge_squared_norm_sum
        ),
    }
    scalar_records = row.get("exact_stored_binary_float_rational_scalars")
    require(isinstance(scalar_records, dict), "Predecessor scalar map missing")
    require(set(scalar_records) == set(expected_scalars), "Predecessor scalar keys changed")
    for name, expected in expected_scalars.items():
        require(
            rational_from_record(scalar_records[name], squared_key, f"Predecessor {name}")
            == expected,
            f"Predecessor {name} changed",
        )
        records += 1
    require(expected_scalars[f"huygens_identity_residual_{unit}2"] == 0, "Huygens residual changed")
    require(
        row.get("exact_centered_sum_identity") is True
        and row.get("exact_pairwise_edge_reconstruction") is True
        and row.get("exact_huygens_identity") is True
        and row.get("fixed_object_and_operand_order") is True
        and row.get("normalization_applied") is False
        and row.get("square_root_applied") is False
        and row.get("eigendecomposition_applied") is False
        and row.get("scientific_threshold_applied") is False,
        "Predecessor row boundary flag changed",
    )
    require(records == 7 * length + 7, "Predecessor row record count changed")
    source_edges = {
        f"source_edge_ab_signed_{unit}": edge_ab_float,
        f"source_edge_ac_signed_{unit}": edge_ac_float,
        f"source_edge_bc_signed_{unit}": edge_bc_float,
    }
    return source_edges, centered, edges, records


def cross_product_row(prior: ModuleType, row: dict, projected: bool) -> tuple[dict, int, int]:
    unit = "px" if projected else "m"
    components = PROJECTED_COMPONENTS if projected else WORLD_COMPONENTS
    provenance = prior.validate_provenance(row.get("provenance"), projected)
    family = row.get("family")
    source_edges, centered, edges, predecessor_records = validate_turn2854_row(
        prior, row, projected
    )
    one_third = Fraction(1, 3)
    component_rows = []
    for name, first, second in components:
        centered_sum = sum(
            (value[first] * value[second] for value in centered), Fraction(0)
        )
        edge_sum = sum(
            (value[first] * value[second] for value in edges.values()), Fraction(0)
        )
        one_third_edge_sum = one_third * edge_sum
        residual = centered_sum - one_third_edge_sum
        require(residual == 0, f"Exact component identity changed: {name}")
        approximate_key = f"approximate_{unit}2"
        component_rows.append(
            {
                "component": name,
                "first_coordinate_index": first,
                "second_coordinate_index": second,
                f"centered_cross_product_sum_{unit}2": exact_record(
                    centered_sum, approximate_key
                ),
                f"edge_cross_product_sum_{unit}2": exact_record(edge_sum, approximate_key),
                f"one_third_edge_cross_product_sum_{unit}2": exact_record(
                    one_third_edge_sum, approximate_key
                ),
                f"cross_product_identity_residual_{unit}2": exact_record(
                    residual, approximate_key
                ),
                "exact_identity": True,
            }
        )
    require(
        [value["component"] for value in component_rows]
        == [name for name, _, _ in components],
        "Component order changed",
    )
    output = {
        "provenance": provenance,
        "family": family,
        **source_edges,
        "fixed_symmetric_component_order": [name for name, _, _ in components],
        f"exact_centered_cross_product_components_{unit}2": component_rows,
        "all_exact_component_identities": True,
        "fixed_object_operand_coordinate_and_component_order": True,
        "normalization_applied": False,
        "square_root_applied": False,
        "eigendecomposition_applied": False,
        "scientific_threshold_applied": False,
    }
    return output, predecessor_records, 4 * len(components)


def analyze(prior: ModuleType, source: dict) -> tuple[list[dict], list[dict], dict]:
    world_source = source.get("world_centered_scatter_rows")
    projected_source = source.get("projected_centered_scatter_rows")
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
    require(
        [row_order_key(prior, row, False) for row in world_source] == expected_row_order(False),
        "World source row order changed",
    )
    require(
        [row_order_key(prior, row, True) for row in projected_source]
        == expected_row_order(True),
        "Projected source row order changed",
    )
    require(
        len({row_key(prior, row, False) for row in world_source}) == 9,
        "Duplicate world source row",
    )
    require(
        len({row_key(prior, row, True) for row in projected_source}) == 36,
        "Duplicate projected source row",
    )

    world_rows = []
    projected_rows = []
    predecessor_records = 0
    serialized_records = 0
    for row in world_source:
        output, predecessor, serialized = cross_product_row(prior, row, False)
        world_rows.append(output)
        predecessor_records += predecessor
        serialized_records += serialized
    for row in projected_source:
        output, predecessor, serialized = cross_product_row(prior, row, True)
        projected_rows.append(output)
        predecessor_records += predecessor
        serialized_records += serialized
    require(predecessor_records == 1008, "Predecessor exact record count changed")

    world_identities = 9 * len(WORLD_COMPONENTS)
    projected_identities = 36 * len(PROJECTED_COMPONENTS)
    require(world_identities == 54, "World component identity count changed")
    require(projected_identities == 108, "Projected component identity count changed")
    require(serialized_records == 648, "Serialized exact record count changed")
    summary = {
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_identity_rows": 9,
        "projected_identity_rows": 36,
        "total_cross_product_identity_rows": 45,
        "turn2854_exact_rational_records_revalidated": predecessor_records,
        "world_component_identities": world_identities,
        "projected_component_identities": projected_identities,
        "total_exact_component_identities": world_identities + projected_identities,
        "exact_rational_records_per_component_identity": 4,
        "world_serialized_exact_rational_records": world_identities * 4,
        "projected_serialized_exact_rational_records": projected_identities * 4,
        "total_serialized_exact_rational_records": serialized_records,
        "all_exact_cross_product_identity_residuals_zero": True,
    }
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
    prior, source, source_pin = load_turn2854_package()
    world_rows, projected_rows, summary = analyze(prior, source)
    report = {
        "schema_version": (
            "wmf-retained-residual-triangle-centered-cross-product-identity-diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2854_centered_scatter_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_source_centered_rows": 9,
            "projected_source_centered_rows": 36,
            "world_cross_product_identity_rows": 9,
            "projected_cross_product_identity_rows": 36,
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
                "Each finite stored edge coordinate is interpreted as its exact binary "
                "floating-point rational value. Integer numerator/denominator strings "
                "control equality; approximate numbers are display aids only."
            ),
            "fixed_order": (
                "A=banana,B=bowl,C=rubiks_cube with eAB=A-B,eAC=A-C,eBC=B-C. "
                "World component order is xx,xy,xz,yy,yz,zz; projected order is "
                "xx,xy,yy. No source row, object, operand, coordinate or camera "
                "reordering occurs."
            ),
            "centered_coordinate_formulas": (
                "The inherited exact algebraic origin uses a_c=(eAB+eAC)/3, "
                "b_c=(-2*eAB+eAC)/3 and c_c=(eAB-2*eAC)/3."
            ),
            "component_identity": (
                "For each fixed coordinate pair p,q, the audit verifies exactly that "
                "sum_i c_i[p]*c_i[q] equals one third of the sum over AB,AC,BC of "
                "edge[p]*edge[q]. Each term is an ordinary scalar product of two fixed "
                "coordinate components; no vector cross product is computed."
            ),
            "units_and_scope": (
                "World component records use stored m^2 units and projected records use "
                "stored px^2 units. These are frozen residual-vector coordinate products, "
                "not physical or image-object measurements."
            ),
            "excluded_operations": (
                "No square root, eigendecomposition, normalization, tolerance, threshold, "
                "classification, fitting, registration or ranking is performed."
            ),
        },
        "summary": summary,
        "world_centered_cross_product_identity_rows": world_rows,
        "projected_centered_cross_product_identity_rows": projected_rows,
        "algebraically_derived_from_turn2854": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "observed_centroid_or_center_estimated": False,
        "physical_scatter_or_variance_measured": False,
        "statistical_variance_or_population_moment_estimated": False,
        "covariance_or_statistical_moment_estimated": False,
        "tensor_or_physical_inertia_interpreted": False,
        "vector_cross_product_computed": False,
        "translation_invariance_claimed": False,
        "physical_distance_or_length_measured": False,
        "physical_scene_triangle_area_measured": False,
        "image_footprint_or_silhouette_area_measured": False,
        "principal_axes_or_eigenvalues_computed": False,
        "spectral_property_inferred": False,
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
            "All 54 world and 108 projected fixed components have zero exact identity "
            "residual. The package serializes 648 exact rational records and revalidates "
            "all 1,008 Turn2854 exact records. This is a deterministic stored-coordinate "
            "algebra check only, not physical, statistical or scientific evidence."
        ),
        "claim_boundaries": {
            "exactness_domain": (
                "Exact means equality over rational interpretations of pinned finite "
                "binary floating-point values. It does not authenticate source precision, "
                "physical truth, uncertainty or measurement accuracy."
            ),
            "constructed_center_only": (
                "The inherited centered coordinates use an algebraic origin reconstructed "
                "from edge vectors. They are not observed or labeled object, scene, camera "
                "or physical centers."
            ),
            "products_not_physical_or_statistical_quantities": (
                "Coordinate products are retained solely to check the stated identity. "
                "They are not covariance, variance, a population moment, physical scatter, "
                "inertia, distance, area, image footprint or a metric-quality estimate."
            ),
            "no_tensor_invariance_or_spectral_claim": (
                "The component list is not interpreted as a tensor or physical inertia "
                "object. No coordinate, frame or translation invariance, spectral property, "
                "principal direction, eigenvalue, conditioning or ranking is claimed."
            ),
            "no_vector_cross_product": (
                "Cross-product identity names refer only to scalar products between fixed "
                "coordinate components. No 3D vector cross product, normal or orientation "
                "is computed or inferred."
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
            "covariance_or_statistical_moment_estimated": False,
            "tensor_or_physical_inertia_interpreted": False,
            "vector_cross_product_computed": False,
            "translation_invariance_claimed": False,
            "principal_axes_or_eigenvalues_computed": False,
            "spectral_property_inferred": False,
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
        len(report["evidence_boundaries"]) == 57
        and all(value is False for value in report["evidence_boundaries"].values()),
        "Output evidence boundary changed",
    )
    assert_output_hygiene(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact centered cross-product component identities for retained "
            "residual triangles without advancing science gates."
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
            "Verified 54 world and 108 projected exact centered cross-product component "
            "identities; 648 exact records, 1,008 predecessor records revalidated and "
            "no stage advance."
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
