#!/usr/bin/env python3
"""Audit exact centered-vertex complement identities for retained residual triangles."""

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
TURN2856_PACKAGE_COMMIT = "376cc7832ee6537ca90b3776408d7f2f8e140cfe"
TURN2856_PACKAGE_TREE = "34e1bc050166552db34577ad7359adb4386ca16b"
TURN2856_PACKAGE_PARENT = "9a3e24cb2cf26a18636147020e2217abdd1f450c"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2856_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_residual_triangle_centered_product_expansion.py"
    ),
    "lines": 996,
    "bytes": 42006,
    "sha256": "a9d015bb544fa77f6e891cdee770a204ddcb74d6c3d04fe6f1d61d34e32ca625",
}
TURN2856_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_residual_triangle_centered_product_expansion_diagnostic.json"
    ),
    "lines": 7280,
    "bytes": 263506,
    "sha256": "8b1f7101d10fcd33821f11964c8f76101f6e017c34ddfdc51fbe9af00ea782d7",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_centered_vertex_complement_diagnostic.json"
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
VERTEX_ORDER = ("a_centered", "b_centered", "c_centered")
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


def load_turn2856_package() -> tuple[ModuleType, ModuleType, ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2856_PACKAGE_COMMIT}^{{commit}}")
        == TURN2856_PACKAGE_COMMIT,
        "Pinned Turn2856 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2856_PACKAGE_COMMIT)
        == TURN2856_PACKAGE_TREE,
        "Pinned Turn2856 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2856_PACKAGE_COMMIT)
        == TURN2856_PACKAGE_PARENT,
        "Pinned Turn2856 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2856_PACKAGE_COMMIT, TURN2856_AUDIT["path"])
    verify_payload(committed_audit, TURN2856_AUDIT, "Pinned Turn2856 audit")
    local_audit_path = REPOSITORY / TURN2856_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2856_AUDIT, "Local Turn2856 audit")
    require(local_audit == committed_audit, "Local Turn2856 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2856_product_expansion")

    committed_report = git_bytes(TURN2856_PACKAGE_COMMIT, TURN2856_REPORT["path"])
    verify_payload(committed_report, TURN2856_REPORT, "Pinned Turn2856 report")
    local_report = (REPOSITORY / TURN2856_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2856_REPORT, "Local Turn2856 report")
    require(local_report == committed_report, "Local Turn2856 report differs")
    parsed = strict_json_loads(committed_report, TURN2856_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2856 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2856 reports differ")

    require(
        parsed["schema_version"]
        == "wmf-retained-residual-triangle-centered-product-expansion-diagnostic-v1",
        "Turn2856 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_source_component_rows"] == 9
        and selection["projected_source_component_rows"] == 36
        and selection["world_product_expansion_identity_rows"] == 9
        and selection["projected_product_expansion_identity_rows"] == 36
        and selection["availability_selected"] is True
        and selection["outcome_dependent_selection"] is False
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"]
        is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2856 selection boundary changed",
    )
    summary = parsed["summary"]
    require(
        summary["world_identity_rows"] == 9
        and summary["projected_identity_rows"] == 36
        and summary["total_product_expansion_identity_rows"] == 45
        and summary["turn2855_exact_rational_records_revalidated"] == 648
        and summary["world_component_identities"] == 54
        and summary["projected_component_identities"] == 108
        and summary["total_exact_component_identities"] == 162
        and summary["exact_rational_records_per_component_identity"] == 4
        and summary["total_serialized_exact_rational_records"] == 648
        and summary["all_exact_centered_coordinate_sums_zero"] is True
        and summary["all_exact_product_expansion_identity_residuals_zero"] is True,
        "Turn2856 exact summary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 59
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2856 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2855"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["observed_centroid_or_center_estimated"] is False
        and parsed["physical_scatter_or_variance_measured"] is False
        and parsed["statistical_variance_or_population_moment_estimated"] is False
        and parsed["covariance_or_statistical_moment_estimated"] is False
        and parsed["matrix_or_tensor_quantity_constructed"] is False
        and parsed["tensor_or_physical_inertia_interpreted"] is False
        and parsed["vector_cross_product_computed"] is False
        and parsed["physical_pairwise_interaction_inferred"] is False
        and parsed["coordinate_or_frame_invariance_claimed"] is False
        and parsed["normalization_applied"] is False
        and parsed["square_root_applied"] is False
        and parsed["eigendecomposition_applied"] is False
        and parsed["scientific_threshold_applied"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False,
        "Turn2856 direct boundary changed",
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
        "Turn2856 zero-activity boundary changed",
    )

    # prior.audit() authenticated and regenerated the complete predecessor chain.
    # Import those now-authenticated modules without repeating that work.
    turn2855 = import_module(
        REPOSITORY / prior.TURN2855_AUDIT["path"],
        "wmf_authenticated_turn2855_centered_components",
    )
    turn2854 = import_module(
        REPOSITORY / turn2855.TURN2854_AUDIT["path"],
        "wmf_authenticated_turn2854_centered_scatter",
    )
    turn2855_pin = parsed["turn2855_centered_component_package"]
    return prior, turn2855, turn2854, parsed, {
        "commit": TURN2856_PACKAGE_COMMIT,
        "tree": TURN2856_PACKAGE_TREE,
        "parent": TURN2856_PACKAGE_PARENT,
        "audit": TURN2856_AUDIT,
        "report": TURN2856_REPORT,
        "canonical_regeneration_verified": True,
        "inherited_turn2855_package": turn2855_pin,
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


def validate_exact_record(
    record: object,
    expected: Fraction,
    approximate_key: str,
    label: str,
) -> None:
    require(
        rational_from_record(record, approximate_key, label) == expected,
        f"{label} exact value changed",
    )


def expected_source_row_keys(unit: str) -> set[str]:
    return {
        "provenance",
        "family",
        f"source_edge_ab_signed_{unit}",
        f"source_edge_ac_signed_{unit}",
        f"source_edge_bc_signed_{unit}",
        "fixed_symmetric_component_order",
        f"exact_centered_product_expansion_components_{unit}2",
        "all_exact_product_expansion_identities",
        "fixed_vertex_operand_coordinate_and_component_order",
        "normalization_applied",
        "square_root_applied",
        "eigendecomposition_applied",
        "scientific_threshold_applied",
    }


def validate_turn2856_row(
    prior: ModuleType,
    turn2855: ModuleType,
    turn2854: ModuleType,
    row: dict,
    projected: bool,
) -> tuple[dict, list[list[Fraction]], int]:
    require(isinstance(row, dict), "Turn2856 source row missing")
    unit = "px" if projected else "m"
    length = 2 if projected else 3
    components = PROJECTED_COMPONENTS if projected else WORLD_COMPONENTS
    families = PROJECTED_FAMILIES if projected else WORLD_FAMILIES
    require(set(row) == expected_source_row_keys(unit), "Turn2856 source row keys changed")
    provenance = turn2854.validate_provenance(row.get("provenance"), projected)
    family = row.get("family")
    require(family in families, "Turn2856 family changed")

    edge_ab_float = finite_vector(
        row.get(f"source_edge_ab_signed_{unit}"), length, "Turn2856 edge AB"
    )
    edge_ac_float = finite_vector(
        row.get(f"source_edge_ac_signed_{unit}"), length, "Turn2856 edge AC"
    )
    edge_bc_float = finite_vector(
        row.get(f"source_edge_bc_signed_{unit}"), length, "Turn2856 edge BC"
    )
    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_bc = exact_vector(edge_bc_float)
    require(vector_subtract(edge_ac, edge_ab) == edge_bc, "Exact source edge cycle changed")
    edges = (edge_ab, edge_ac, edge_bc)

    one_third = Fraction(1, 3)
    centered = [
        vector_scale(one_third, vector_add(edge_ab, edge_ac)),
        vector_scale(
            one_third,
            vector_add(vector_scale(Fraction(-2), edge_ab), edge_ac),
        ),
        vector_scale(
            one_third,
            vector_add(edge_ab, vector_scale(Fraction(-2), edge_ac)),
        ),
    ]
    centered_sum = vector_add(vector_add(centered[0], centered[1]), centered[2])
    require(centered_sum == [Fraction(0)] * length, "Exact centered sum changed")

    require(
        row.get("fixed_symmetric_component_order")
        == [name for name, _, _ in components],
        "Turn2856 component order changed",
    )
    source_components = row.get(f"exact_centered_product_expansion_components_{unit}2")
    require(
        isinstance(source_components, list) and len(source_components) == len(components),
        "Turn2856 component rows changed",
    )
    approximate_key = f"approximate_{unit}2"
    predecessor_records = 0
    expected_component_keys = {
        "component",
        "first_coordinate_index",
        "second_coordinate_index",
        f"same_vertex_product_sum_{unit}2",
        f"ordered_distinct_vertex_product_sum_{unit}2",
        f"factorized_centered_coordinate_sum_product_{unit}2",
        f"product_expansion_identity_residual_{unit}2",
        "exact_identity",
    }
    for source_component, (name, first, second) in zip(source_components, components):
        require(
            isinstance(source_component, dict)
            and set(source_component) == expected_component_keys,
            "Turn2856 component record keys changed",
        )
        require(
            source_component.get("component") == name
            and source_component.get("first_coordinate_index") == first
            and source_component.get("second_coordinate_index") == second,
            "Turn2856 component identity changed",
        )
        same_vertex_sum = sum(
            (value[first] * value[second] for value in centered), Fraction(0)
        )
        ordered_distinct_vertex_sum = sum(
            (
                first_vertex[first] * second_vertex[second]
                for first_index, first_vertex in enumerate(centered)
                for second_index, second_vertex in enumerate(centered)
                if first_index != second_index
            ),
            Fraction(0),
        )
        first_coordinate_sum = sum((value[first] for value in centered), Fraction(0))
        second_coordinate_sum = sum((value[second] for value in centered), Fraction(0))
        factorized_sum_product = first_coordinate_sum * second_coordinate_sum
        residual = same_vertex_sum + ordered_distinct_vertex_sum - factorized_sum_product
        require(
            first_coordinate_sum == 0
            and second_coordinate_sum == 0
            and factorized_sum_product == 0
            and residual == 0,
            "Turn2856 product expansion changed",
        )
        expectations = {
            f"same_vertex_product_sum_{unit}2": same_vertex_sum,
            f"ordered_distinct_vertex_product_sum_{unit}2": ordered_distinct_vertex_sum,
            f"factorized_centered_coordinate_sum_product_{unit}2": factorized_sum_product,
            f"product_expansion_identity_residual_{unit}2": residual,
        }
        for key, expected in expectations.items():
            validate_exact_record(
                source_component.get(key),
                expected,
                approximate_key,
                f"Turn2856 {name} {key}",
            )
            predecessor_records += 1
        require(source_component.get("exact_identity") is True, "Turn2856 identity flag changed")

    require(
        row.get("all_exact_product_expansion_identities") is True
        and row.get("fixed_vertex_operand_coordinate_and_component_order") is True
        and row.get("normalization_applied") is False
        and row.get("square_root_applied") is False
        and row.get("eigendecomposition_applied") is False
        and row.get("scientific_threshold_applied") is False,
        "Turn2856 row boundary flag changed",
    )
    require(predecessor_records == 4 * len(components), "Turn2856 record count changed")
    source_edges = {
        f"source_edge_ab_signed_{unit}": edge_ab_float,
        f"source_edge_ac_signed_{unit}": edge_ac_float,
        f"source_edge_bc_signed_{unit}": edge_bc_float,
    }
    return {"provenance": provenance, "family": family, **source_edges}, centered, predecessor_records


def vertex_complement_row(
    prior: ModuleType,
    turn2855: ModuleType,
    turn2854: ModuleType,
    row: dict,
    projected: bool,
) -> tuple[dict, int, int]:
    unit = "px" if projected else "m"
    components = PROJECTED_COMPONENTS if projected else WORLD_COMPONENTS
    source, centered, predecessor_records = validate_turn2856_row(
        prior, turn2855, turn2854, row, projected
    )
    component_rows = []
    approximate_key = f"approximate_{unit}2"
    for name, first, second in components:
        vertex_identities = []
        for vertex_index, (vertex_name, focal_vertex) in enumerate(
            zip(VERTEX_ORDER, centered)
        ):
            distinct_second_coordinate_sum = sum(
                (
                    other_vertex[second]
                    for other_index, other_vertex in enumerate(centered)
                    if other_index != vertex_index
                ),
                Fraction(0),
            )
            full_second_coordinate_sum = (
                focal_vertex[second] + distinct_second_coordinate_sum
            )
            focal_product = focal_vertex[first] * focal_vertex[second]
            focal_times_distinct_sum = (
                focal_vertex[first] * distinct_second_coordinate_sum
            )
            focal_times_full_sum = focal_vertex[first] * full_second_coordinate_sum
            residual = focal_product + focal_times_distinct_sum - focal_times_full_sum
            require(
                distinct_second_coordinate_sum == -focal_vertex[second],
                f"Exact vertex complement changed: {name}/{vertex_name}",
            )
            require(
                full_second_coordinate_sum == 0
                and focal_times_full_sum == 0
                and residual == 0,
                f"Exact vertex cancellation changed: {name}/{vertex_name}",
            )
            vertex_identities.append(
                {
                    "vertex": vertex_name,
                    "vertex_index": vertex_index,
                    f"focal_vertex_product_{unit}2": exact_record(
                        focal_product, approximate_key
                    ),
                    f"focal_times_distinct_vertex_coordinate_sum_{unit}2": exact_record(
                        focal_times_distinct_sum, approximate_key
                    ),
                    f"focal_times_full_centered_coordinate_sum_{unit}2": exact_record(
                        focal_times_full_sum, approximate_key
                    ),
                    f"vertex_complement_cancellation_residual_{unit}2": exact_record(
                        residual, approximate_key
                    ),
                    "exact_identity": True,
                }
            )
        component_rows.append(
            {
                "component": name,
                "first_coordinate_index": first,
                "second_coordinate_index": second,
                "fixed_vertex_order": list(VERTEX_ORDER),
                "vertex_complement_identities": vertex_identities,
                "all_exact_vertex_complement_identities": True,
            }
        )
    require(
        [value["component"] for value in component_rows]
        == [name for name, _, _ in components],
        "Output component order changed",
    )
    output = {
        **source,
        "fixed_symmetric_component_order": [name for name, _, _ in components],
        f"exact_centered_vertex_complement_components_{unit}2": component_rows,
        "all_exact_vertex_complement_identities": True,
        "fixed_vertex_operand_coordinate_and_component_order": True,
        "normalization_applied": False,
        "square_root_applied": False,
        "eigendecomposition_applied": False,
        "scientific_threshold_applied": False,
    }
    return output, predecessor_records, 4 * len(VERTEX_ORDER) * len(components)


def analyze(
    prior: ModuleType,
    turn2855: ModuleType,
    turn2854: ModuleType,
    source: dict,
) -> tuple[list[dict], list[dict], dict]:
    world_source = source.get("world_centered_product_expansion_rows")
    projected_source = source.get("projected_centered_product_expansion_rows")
    require(isinstance(world_source, list) and len(world_source) == 9, "World rows changed")
    require(
        isinstance(projected_source, list) and len(projected_source) == 36,
        "Projected rows changed",
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
        [turn2855.row_order_key(turn2854, row, False) for row in world_source]
        == turn2855.expected_row_order(False),
        "World source row order changed",
    )
    require(
        [turn2855.row_order_key(turn2854, row, True) for row in projected_source]
        == turn2855.expected_row_order(True),
        "Projected source row order changed",
    )
    require(
        len({turn2855.row_key(turn2854, row, False) for row in world_source}) == 9,
        "Duplicate world source row",
    )
    require(
        len({turn2855.row_key(turn2854, row, True) for row in projected_source}) == 36,
        "Duplicate projected source row",
    )

    world_rows = []
    projected_rows = []
    predecessor_records = 0
    serialized_records = 0
    for row in world_source:
        output, predecessor, serialized = vertex_complement_row(
            prior, turn2855, turn2854, row, False
        )
        world_rows.append(output)
        predecessor_records += predecessor
        serialized_records += serialized
    for row in projected_source:
        output, predecessor, serialized = vertex_complement_row(
            prior, turn2855, turn2854, row, True
        )
        projected_rows.append(output)
        predecessor_records += predecessor
        serialized_records += serialized

    world_component_rows = 9 * len(WORLD_COMPONENTS)
    projected_component_rows = 36 * len(PROJECTED_COMPONENTS)
    world_identities = world_component_rows * len(VERTEX_ORDER)
    projected_identities = projected_component_rows * len(VERTEX_ORDER)
    require(predecessor_records == 648, "Turn2856 exact record count changed")
    require(world_component_rows == 54, "World component-row count changed")
    require(projected_component_rows == 108, "Projected component-row count changed")
    require(world_identities == 162, "World identity count changed")
    require(projected_identities == 324, "Projected identity count changed")
    require(serialized_records == 1944, "Serialized exact record count changed")
    summary = {
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_identity_rows": 9,
        "projected_identity_rows": 36,
        "total_vertex_complement_identity_rows": 45,
        "turn2856_exact_rational_records_revalidated": predecessor_records,
        "world_component_rows": world_component_rows,
        "projected_component_rows": projected_component_rows,
        "total_component_rows": world_component_rows + projected_component_rows,
        "vertex_identities_per_component": len(VERTEX_ORDER),
        "world_vertex_complement_identities": world_identities,
        "projected_vertex_complement_identities": projected_identities,
        "total_exact_vertex_complement_identities": world_identities + projected_identities,
        "exact_rational_records_per_vertex_identity": 4,
        "world_serialized_exact_rational_records": world_identities * 4,
        "projected_serialized_exact_rational_records": projected_identities * 4,
        "total_serialized_exact_rational_records": serialized_records,
        "all_exact_centered_coordinate_sums_zero": True,
        "all_exact_focal_plus_vertex_complement_coordinate_sums_zero": True,
        "all_exact_vertex_complement_cancellation_residuals_zero": True,
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
    prior, turn2855, turn2854, source, source_pin = load_turn2856_package()
    world_rows, projected_rows, summary = analyze(
        prior, turn2855, turn2854, source
    )
    report = {
        "schema_version": (
            "wmf-retained-residual-triangle-centered-vertex-complement-diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2856_centered_product_expansion_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_source_product_expansion_rows": 9,
            "projected_source_product_expansion_rows": 36,
            "world_vertex_complement_identity_rows": 9,
            "projected_vertex_complement_identity_rows": 36,
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
                "Each finite retained edge coordinate is interpreted as its exact binary "
                "floating-point rational value. Integer numerator/denominator strings "
                "control equality; approximate numbers are display aids only."
            ),
            "fixed_order": (
                "A=banana,B=bowl,C=rubiks_cube with eAB=A-B,eAC=A-C,eBC=B-C. "
                "World component order is xx,xy,xz,yy,yz,zz; projected order is "
                "xx,xy,yy. The three algebraic vertices and all coordinate operands "
                "retain their inherited order."
            ),
            "centered_coordinate_formulas": (
                "The inherited exact algebraic origin uses a_c=(eAB+eAC)/3, "
                "b_c=(-2*eAB+eAC)/3 and c_c=(eAB-2*eAC)/3; each fixed coordinate "
                "sum is therefore checked as exactly zero."
            ),
            "vertex_complement_identity": (
                "For each fixed coordinate pair p,q and each focal algebraic vertex, "
                "the q-coordinate sum of the other two vertices is checked as the exact "
                "negative of the focal q coordinate. The focal p coordinate times its "
                "own q coordinate therefore cancels the focal p coordinate times that "
                "two-vertex complement sum; the localized residual is checked exactly."
            ),
            "units_and_scope": (
                "World product records use stored m^2 units and projected records use "
                "stored px^2 units. These are frozen residual-vector coordinate products, "
                "not physical or image-object measurements."
            ),
            "excluded_operations": (
                "No physical or statistical vertex relation, pairwise interaction, vector "
                "cross product, covariance, population moment, matrix or tensor "
                "interpretation, square root, eigendecomposition, normalization, tolerance, "
                "threshold, fitting, registration, classification or ranking is performed."
            ),
        },
        "summary": summary,
        "world_centered_vertex_complement_rows": world_rows,
        "projected_centered_vertex_complement_rows": projected_rows,
        "algebraically_derived_from_turn2856": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "observed_centroid_or_center_estimated": False,
        "physical_scatter_or_variance_measured": False,
        "statistical_variance_or_population_moment_estimated": False,
        "covariance_or_statistical_moment_estimated": False,
        "matrix_or_tensor_quantity_constructed": False,
        "tensor_or_physical_inertia_interpreted": False,
        "vector_cross_product_computed": False,
        "physical_pairwise_interaction_inferred": False,
        "vertexwise_physical_relation_inferred": False,
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
            "All 162 world and 324 projected fixed-vertex complement identities have zero "
            "exact residual. The package serializes 1,944 exact rational records and "
            "revalidates all 648 Turn2856 exact records. This is a deterministic localized "
            "stored-coordinate algebra check only, not physical, statistical or scientific "
            "evidence."
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
                "Focal and complement coordinate products are retained only to localize a "
                "finite algebraic cancellation. Vertex names are fixed operand labels, not "
                "evidence of a physical or statistical vertex relation. The products are "
                "not covariance, variance, a population moment, interaction, association, "
                "physical scatter, inertia, distance, area, image footprint or metric evidence."
            ),
            "no_matrix_tensor_invariance_or_spectral_claim": (
                "The component list is not assembled or interpreted as a matrix, tensor or "
                "physical inertia object. No coordinate, frame or translation invariance, "
                "spectral property, principal direction, eigenvalue or ranking is claimed."
            ),
            "no_vector_cross_product": (
                "All operations are ordinary products of two fixed scalar coordinates and "
                "finite sums. No 3D vector cross product, normal or orientation is computed."
            ),
            "localized_not_independent": (
                "The three focal-vertex identities per component are deterministic "
                "decompositions of the same inherited zero-sum coordinates. They are not "
                "independent samples, repeated measurements or evidence from three scenes."
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
            "matrix_or_tensor_quantity_constructed": False,
            "tensor_or_physical_inertia_interpreted": False,
            "vector_cross_product_computed": False,
            "physical_pairwise_interaction_inferred": False,
            "vertexwise_physical_relation_inferred": False,
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
        len(report["evidence_boundaries"]) == 60
        and all(value is False for value in report["evidence_boundaries"].values()),
        "Output evidence boundary changed",
    )
    assert_output_hygiene(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact centered-vertex complement identities for retained residual "
            "triangles without advancing science gates."
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
            "Verified 162 world and 324 projected exact centered-vertex complement "
            "identities; 1,944 exact records, 648 predecessor records revalidated and "
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
