#!/usr/bin/env python3
"""Audit exact two-sided centered bipartition products for retained triangles."""

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
TURN2857_PACKAGE_COMMIT = "8bdbeda83479cf65b552a3a78aa6f15dfa2c95c0"
TURN2857_PACKAGE_TREE = "747b65513fab24706ff1dcaa269de14e5f861dce"
TURN2857_PACKAGE_PARENT = "35c89d54975a655be52d0070062000612e6b2f75"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2857_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_residual_triangle_centered_vertex_complement.py"
    ),
    "lines": 1063,
    "bytes": 45344,
    "sha256": "4b2a79a6c9fe59fba2dadfc5e216516ba0ad74692177e8463d95b681d9bfc075",
}
TURN2857_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_residual_triangle_centered_vertex_complement_diagnostic.json"
    ),
    "lines": 17350,
    "bytes": 656145,
    "sha256": "f27c83446e7131a0bef780d9474c714a83fb3608fd329d3ce102d8bf679b5f50",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_centered_bipartition_product_diagnostic.json"
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


def load_turn2857_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2857_PACKAGE_COMMIT}^{{commit}}")
        == TURN2857_PACKAGE_COMMIT,
        "Pinned Turn2857 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2857_PACKAGE_COMMIT)
        == TURN2857_PACKAGE_TREE,
        "Pinned Turn2857 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2857_PACKAGE_COMMIT)
        == TURN2857_PACKAGE_PARENT,
        "Pinned Turn2857 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2857_PACKAGE_COMMIT, TURN2857_AUDIT["path"])
    verify_payload(committed_audit, TURN2857_AUDIT, "Pinned Turn2857 audit")
    local_audit_path = REPOSITORY / TURN2857_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2857_AUDIT, "Local Turn2857 audit")
    require(local_audit == committed_audit, "Local Turn2857 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2857_vertex_complement")

    committed_report = git_bytes(TURN2857_PACKAGE_COMMIT, TURN2857_REPORT["path"])
    verify_payload(committed_report, TURN2857_REPORT, "Pinned Turn2857 report")
    local_report = (REPOSITORY / TURN2857_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2857_REPORT, "Local Turn2857 report")
    require(local_report == committed_report, "Local Turn2857 report differs")
    parsed = strict_json_loads(committed_report, TURN2857_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2857 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2857 reports differ")

    require(
        parsed["schema_version"]
        == "wmf-retained-residual-triangle-centered-vertex-complement-diagnostic-v1",
        "Turn2857 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_source_product_expansion_rows"] == 9
        and selection["projected_source_product_expansion_rows"] == 36
        and selection["world_vertex_complement_identity_rows"] == 9
        and selection["projected_vertex_complement_identity_rows"] == 36
        and selection["availability_selected"] is True
        and selection["outcome_dependent_selection"] is False
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"] is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2857 selection boundary changed",
    )
    summary = parsed["summary"]
    require(
        summary["world_identity_rows"] == 9
        and summary["projected_identity_rows"] == 36
        and summary["total_vertex_complement_identity_rows"] == 45
        and summary["turn2856_exact_rational_records_revalidated"] == 648
        and summary["world_component_rows"] == 54
        and summary["projected_component_rows"] == 108
        and summary["total_component_rows"] == 162
        and summary["vertex_identities_per_component"] == 3
        and summary["world_vertex_complement_identities"] == 162
        and summary["projected_vertex_complement_identities"] == 324
        and summary["total_exact_vertex_complement_identities"] == 486
        and summary["exact_rational_records_per_vertex_identity"] == 4
        and summary["total_serialized_exact_rational_records"] == 1944
        and summary["all_exact_centered_coordinate_sums_zero"] is True
        and summary["all_exact_focal_plus_vertex_complement_coordinate_sums_zero"] is True
        and summary["all_exact_vertex_complement_cancellation_residuals_zero"] is True,
        "Turn2857 exact summary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 60
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2857 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2856"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["matrix_or_tensor_quantity_constructed"] is False
        and parsed["physical_pairwise_interaction_inferred"] is False
        and parsed["vertexwise_physical_relation_inferred"] is False
        and parsed["coordinate_or_frame_invariance_claimed"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False
        and parsed["new_behavioral_cells"] == 0
        and parsed["new_model_requests"] == 0
        and parsed["new_actions"] == 0
        and parsed["new_observations"] == 0
        and parsed["new_request_selections"] == 0
        and parsed["labels_created"] == 0
        and parsed["forecast_accuracy_estimated"] is False
        and parsed["resource_budget_qualified"] is False
        and parsed["confirmation_released"] is False,
        "Turn2857 activity or claim boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2857_PACKAGE_COMMIT,
        "tree": TURN2857_PACKAGE_TREE,
        "parent": TURN2857_PACKAGE_PARENT,
        "audit": TURN2857_AUDIT,
        "report": TURN2857_REPORT,
        "canonical_regeneration_verified": True,
        "inherited_turn2856_package": parsed["turn2856_centered_product_expansion_package"],
    }


def finite_number(value: object, label: str) -> float:
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"Finite number required: {label}",
    )
    return float(value)


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"Vector changed: {label}")
    return [finite_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def exact_vector(values: list[float]) -> list[Fraction]:
    return [Fraction.from_float(value) for value in values]


def vector_add(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    require(len(first) == len(second), "Vector length mismatch")
    return [left + right for left, right in zip(first, second)]


def vector_subtract(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    require(len(first) == len(second), "Vector length mismatch")
    return [left - right for left, right in zip(first, second)]


def vector_scale(value: Fraction, vector: list[Fraction]) -> list[Fraction]:
    return [value * coordinate for coordinate in vector]


def exact_record(value: Fraction, approximate_key: str) -> dict:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        approximate_key: float(value),
    }


def exact_record_fingerprints(value: object) -> set[str]:
    fingerprints: set[str] = set()

    def visit(item: object) -> None:
        if isinstance(item, dict):
            keys = set(item)
            if (
                "numerator" in keys
                and "denominator" in keys
                and len(keys) == 3
                and sum(key.startswith("approximate_") for key in keys) == 1
            ):
                fingerprints.add(
                    json.dumps(
                        item,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return fingerprints


def rational_from_record(record: object, approximate_key: str, label: str) -> Fraction:
    require(isinstance(record, dict), f"Exact record missing: {label}")
    require(
        set(record) == {"numerator", "denominator", approximate_key},
        f"Exact record keys changed: {label}",
    )
    numerator = record["numerator"]
    denominator = record["denominator"]
    require(
        isinstance(numerator, str) and re.fullmatch(r"0|-?[1-9][0-9]*", numerator),
        f"Noncanonical numerator: {label}",
    )
    require(
        isinstance(denominator, str) and re.fullmatch(r"[1-9][0-9]*", denominator),
        f"Noncanonical denominator: {label}",
    )
    result = Fraction(int(numerator), int(denominator))
    require(
        numerator == str(result.numerator) and denominator == str(result.denominator),
        f"Unreduced exact record: {label}",
    )
    approximate = finite_number(record[approximate_key], f"{label}.{approximate_key}")
    require(approximate == float(result), f"Approximation changed: {label}")
    return result


def validate_exact_record(
    record: object,
    expected: Fraction,
    approximate_key: str,
    label: str,
) -> None:
    require(
        rational_from_record(record, approximate_key, label) == expected,
        f"Exact value changed: {label}",
    )


def expected_source_row_keys(unit: str) -> set[str]:
    return {
        "provenance",
        "family",
        f"source_edge_ab_signed_{unit}",
        f"source_edge_ac_signed_{unit}",
        f"source_edge_bc_signed_{unit}",
        "fixed_symmetric_component_order",
        f"exact_centered_vertex_complement_components_{unit}2",
        "all_exact_vertex_complement_identities",
        "fixed_vertex_operand_coordinate_and_component_order",
        "normalization_applied",
        "square_root_applied",
        "eigendecomposition_applied",
        "scientific_threshold_applied",
    }


def bipartition_product_row(
    prior: ModuleType,
    row: dict,
    projected: bool,
) -> tuple[dict, int, int]:
    require(isinstance(row, dict), "Turn2857 source row missing")
    unit = "px" if projected else "m"
    length = 2 if projected else 3
    components = PROJECTED_COMPONENTS if projected else WORLD_COMPONENTS
    families = PROJECTED_FAMILIES if projected else WORLD_FAMILIES
    require(set(row) == expected_source_row_keys(unit), "Turn2857 source row keys changed")
    provenance = row.get("provenance")
    require(isinstance(provenance, dict), "Turn2857 provenance missing")
    family = row.get("family")
    require(family in families, "Turn2857 family changed")

    edge_ab_float = finite_vector(
        row.get(f"source_edge_ab_signed_{unit}"), length, "Turn2857 edge AB"
    )
    edge_ac_float = finite_vector(
        row.get(f"source_edge_ac_signed_{unit}"), length, "Turn2857 edge AC"
    )
    edge_bc_float = finite_vector(
        row.get(f"source_edge_bc_signed_{unit}"), length, "Turn2857 edge BC"
    )
    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_bc = exact_vector(edge_bc_float)
    require(vector_subtract(edge_ac, edge_ab) == edge_bc, "Exact source edge cycle changed")
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
    require(
        vector_add(vector_add(centered[0], centered[1]), centered[2])
        == [Fraction(0)] * length,
        "Exact centered sum changed",
    )

    require(
        row.get("fixed_symmetric_component_order")
        == [name for name, _, _ in components],
        "Turn2857 component order changed",
    )
    source_components = row.get(f"exact_centered_vertex_complement_components_{unit}2")
    require(
        isinstance(source_components, list) and len(source_components) == len(components),
        "Turn2857 component rows changed",
    )
    approximate_key = f"approximate_{unit}2"
    predecessor_records = 0
    output_components = []
    for source_component, (name, first, second) in zip(source_components, components):
        require(isinstance(source_component, dict), "Turn2857 component missing")
        require(
            source_component.get("component") == name
            and source_component.get("first_coordinate_index") == first
            and source_component.get("second_coordinate_index") == second
            and source_component.get("fixed_vertex_order") == list(VERTEX_ORDER)
            and source_component.get("all_exact_vertex_complement_identities") is True,
            "Turn2857 component identity changed",
        )
        source_identities = source_component.get("vertex_complement_identities")
        require(
            isinstance(source_identities, list) and len(source_identities) == len(VERTEX_ORDER),
            "Turn2857 vertex identities changed",
        )
        output_identities = []
        for vertex_index, (vertex_name, focal, source_identity) in enumerate(
            zip(VERTEX_ORDER, centered, source_identities)
        ):
            require(
                isinstance(source_identity, dict)
                and source_identity.get("vertex") == vertex_name
                and source_identity.get("vertex_index") == vertex_index
                and source_identity.get("exact_identity") is True,
                "Turn2857 vertex identity changed",
            )
            complement_first = sum(
                (
                    other[first]
                    for other_index, other in enumerate(centered)
                    if other_index != vertex_index
                ),
                Fraction(0),
            )
            complement_second = sum(
                (
                    other[second]
                    for other_index, other in enumerate(centered)
                    if other_index != vertex_index
                ),
                Fraction(0),
            )
            focal_product = focal[first] * focal[second]
            focal_times_complement = focal[first] * complement_second
            require(
                complement_first == -focal[first]
                and complement_second == -focal[second],
                f"Exact two-sided complement changed: {name}/{vertex_name}",
            )
            predecessor_expectations = {
                f"focal_vertex_product_{unit}2": focal_product,
                f"focal_times_distinct_vertex_coordinate_sum_{unit}2": focal_times_complement,
                f"focal_times_full_centered_coordinate_sum_{unit}2": Fraction(0),
                f"vertex_complement_cancellation_residual_{unit}2": Fraction(0),
            }
            for key, expected in predecessor_expectations.items():
                validate_exact_record(
                    source_identity.get(key),
                    expected,
                    approximate_key,
                    f"Turn2857 {name}/{vertex_name} {key}",
                )
                predecessor_records += 1

            complement_product = complement_first * complement_second
            residual = complement_product - focal_product
            require(
                complement_product == focal_product and residual == 0,
                f"Exact bipartition product identity changed: {name}/{vertex_name}",
            )
            output_identities.append(
                {
                    "focal_vertex": vertex_name,
                    "focal_vertex_index": vertex_index,
                    f"focal_vertex_product_{unit}2": exact_record(
                        focal_product, approximate_key
                    ),
                    f"two_vertex_complement_sum_product_{unit}2": exact_record(
                        complement_product, approximate_key
                    ),
                    f"bipartition_product_identity_residual_{unit}2": exact_record(
                        residual, approximate_key
                    ),
                    "exact_identity": True,
                }
            )
        output_components.append(
            {
                "component": name,
                "first_coordinate_index": first,
                "second_coordinate_index": second,
                "fixed_focal_vertex_order": list(VERTEX_ORDER),
                "bipartition_product_identities": output_identities,
                "all_exact_bipartition_product_identities": True,
            }
        )

    require(predecessor_records == 4 * len(VERTEX_ORDER) * len(components), "Turn2857 record count changed")
    source_edges = {
        f"source_edge_ab_signed_{unit}": edge_ab_float,
        f"source_edge_ac_signed_{unit}": edge_ac_float,
        f"source_edge_bc_signed_{unit}": edge_bc_float,
    }
    output = {
        "provenance": provenance,
        "family": family,
        **source_edges,
        "fixed_symmetric_component_order": [name for name, _, _ in components],
        f"exact_centered_bipartition_product_components_{unit}2": output_components,
        "all_exact_bipartition_product_identities": True,
        "fixed_vertex_operand_coordinate_and_component_order": True,
        "normalization_applied": False,
        "square_root_applied": False,
        "eigendecomposition_applied": False,
        "scientific_threshold_applied": False,
    }
    return output, predecessor_records, 3 * len(VERTEX_ORDER) * len(components)


def row_key(row: dict, projected: bool) -> tuple:
    provenance = row["provenance"]
    return (
        provenance["phase"],
        provenance["layout_pair_id"],
        provenance["model_id"],
        provenance.get("camera_name") if projected else None,
        row["family"],
    )


def analyze(prior: ModuleType, source: dict) -> tuple[list[dict], list[dict], dict]:
    world_source = source.get("world_centered_vertex_complement_rows")
    projected_source = source.get("projected_centered_vertex_complement_rows")
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
    require(len({row_key(row, False) for row in world_source}) == 9, "Duplicate world row")
    require(
        len({row_key(row, True) for row in projected_source}) == 36,
        "Duplicate projected row",
    )

    world_rows = []
    projected_rows = []
    predecessor_records = 0
    serialized_records = 0
    for row in world_source:
        output, predecessor, serialized = bipartition_product_row(prior, row, False)
        world_rows.append(output)
        predecessor_records += predecessor
        serialized_records += serialized
    for row in projected_source:
        output, predecessor, serialized = bipartition_product_row(prior, row, True)
        projected_rows.append(output)
        predecessor_records += predecessor
        serialized_records += serialized

    world_component_rows = 9 * len(WORLD_COMPONENTS)
    projected_component_rows = 36 * len(PROJECTED_COMPONENTS)
    world_identities = world_component_rows * len(VERTEX_ORDER)
    projected_identities = projected_component_rows * len(VERTEX_ORDER)
    require(predecessor_records == 1944, "Turn2857 exact record count changed")
    require(world_component_rows == 54, "World component-row count changed")
    require(projected_component_rows == 108, "Projected component-row count changed")
    require(world_identities == 162, "World identity count changed")
    require(projected_identities == 324, "Projected identity count changed")
    require(serialized_records == 1458, "Serialized exact record count changed")
    predecessor_unique_records = exact_record_fingerprints(source)
    candidate_unique_records = exact_record_fingerprints(
        {"world": world_rows, "projected": projected_rows}
    )
    require(len(predecessor_unique_records) == 970, "Turn2857 unique exact values changed")
    require(len(candidate_unique_records) == 486, "Candidate unique exact values changed")
    require(
        candidate_unique_records <= predecessor_unique_records,
        "Candidate unexpectedly introduced a new exact value",
    )
    summary = {
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_identity_rows": 9,
        "projected_identity_rows": 36,
        "total_bipartition_product_identity_rows": 45,
        "turn2857_exact_rational_records_revalidated": predecessor_records,
        "world_component_rows": world_component_rows,
        "projected_component_rows": projected_component_rows,
        "total_component_rows": world_component_rows + projected_component_rows,
        "focal_vertex_bipartitions_per_component": len(VERTEX_ORDER),
        "world_bipartition_product_identities": world_identities,
        "projected_bipartition_product_identities": projected_identities,
        "total_exact_bipartition_product_identities": world_identities + projected_identities,
        "exact_rational_records_per_bipartition_identity": 3,
        "world_serialized_exact_rational_records": world_identities * 3,
        "projected_serialized_exact_rational_records": projected_identities * 3,
        "total_serialized_exact_rational_records": serialized_records,
        "inherited_focal_product_record_slots": world_identities + projected_identities,
        "new_bipartition_product_and_residual_record_roles": 2
        * (world_identities + projected_identities),
        "turn2857_unique_exact_rational_record_objects": len(predecessor_unique_records),
        "candidate_unique_exact_rational_record_objects": len(candidate_unique_records),
        "candidate_unique_record_objects_already_present_in_turn2857": len(
            candidate_unique_records & predecessor_unique_records
        ),
        "new_unique_exact_rational_record_objects_vs_turn2857": len(
            candidate_unique_records - predecessor_unique_records
        ),
        "value_level_novelty": False,
        "all_exact_centered_coordinate_sums_zero": True,
        "all_exact_two_sided_vertex_complement_sums_negate_focal_coordinates": True,
        "all_exact_bipartition_product_identity_residuals_zero": True,
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
    prior, source, source_pin = load_turn2857_package()
    world_rows, projected_rows, summary = analyze(prior, source)
    report = {
        "schema_version": (
            "wmf-retained-residual-triangle-centered-bipartition-product-diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2857_centered_vertex_complement_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_source_vertex_complement_rows": 9,
            "projected_source_vertex_complement_rows": 36,
            "world_bipartition_product_identity_rows": 9,
            "projected_bipartition_product_identity_rows": 36,
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
                "xx,xy,yy. Focal vertices retain A,B,C order."
            ),
            "bipartition_product_identity": (
                "For each fixed coordinate pair p,q and focal algebraic vertex, the "
                "p-coordinate sum and q-coordinate sum over the other two vertices are "
                "each checked as the exact negative of the corresponding focal coordinate. "
                "Their product is therefore checked as exactly equal to the focal p-times-q "
                "product, with a zero exact residual."
            ),
            "difference_from_turn2857": (
                "Turn2857 multiplied the focal p coordinate by the other-two q-coordinate "
                "sum and checked cancellation. This package instead multiplies the other-two "
                "p-coordinate sum by the other-two q-coordinate sum and compares that "
                "two-sided complement product with the focal product."
            ),
            "value_level_novelty": (
                "The expression and 972 bipartition-product/residual field roles are new, but "
                "the 1,458 serialized slots contain 486 unique exact rational record objects, "
                "all already present among Turn2857's 970 unique objects. Thus this package "
                "adds no new exact numerical value or numerical evidence."
            ),
            "units_and_scope": (
                "World products use stored m^2 units and projected products use stored px^2 "
                "units. These are frozen residual-vector coordinate products, not physical "
                "or image-object measurements."
            ),
            "excluded_operations": (
                "No physical or statistical partition, pairwise interaction, covariance, "
                "population moment, matrix or tensor interpretation, square root, "
                "eigendecomposition, normalization, threshold, fitting, classification or "
                "ranking is performed."
            ),
        },
        "summary": summary,
        "world_centered_bipartition_product_rows": world_rows,
        "projected_centered_bipartition_product_rows": projected_rows,
        "algebraically_derived_from_turn2857": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "observed_centroid_or_center_estimated": False,
        "physical_scatter_or_variance_measured": False,
        "statistical_variance_or_population_moment_estimated": False,
        "covariance_or_statistical_moment_estimated": False,
        "matrix_or_tensor_quantity_constructed": False,
        "tensor_or_physical_inertia_interpreted": False,
        "physical_pairwise_interaction_inferred": False,
        "physical_partition_or_set_relation_inferred": False,
        "vertexwise_physical_relation_inferred": False,
        "translation_invariance_claimed": False,
        "physical_distance_or_length_measured": False,
        "physical_scene_triangle_area_measured": False,
        "image_footprint_or_silhouette_area_measured": False,
        "principal_axes_or_eigenvalues_computed": False,
        "spectral_property_inferred": False,
        "orientation_or_chirality_inferred": False,
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
            "All 162 world and 324 projected fixed-vertex two-sided complement products "
            "equal their focal products with zero exact residual. The package serializes "
            "1,458 exact rational record slots and revalidates all 1,944 Turn2857 record slots. "
            "All 486 unique candidate record objects already occur in Turn2857, so value-level "
            "novelty is zero. This is a dependent stored-coordinate algebra check only, not "
            "physical, statistical or scientific evidence."
        ),
        "claim_boundaries": {
            "exactness_domain": (
                "Exact means equality over rational interpretations of pinned finite binary "
                "floating-point values. It does not authenticate source precision, physical "
                "truth, uncertainty or measurement accuracy."
            ),
            "dependent_bipartitions": (
                "The three focal-versus-other-two bipartitions per component are deterministic "
                "re-expressions of the same inherited zero-sum coordinates. They are not "
                "independent samples, repeated measurements, sets of objects or scenes."
            ),
            "no_new_numerical_values": (
                "The new expression reuses only exact rational record objects already present "
                "in Turn2857. New semantic field roles are not new observations, measurements, "
                "samples or numerical evidence."
            ),
            "not_a_physical_or_statistical_partition": (
                "Bipartition and complement refer only to fixed algebraic operand indices. "
                "No physical grouping, interaction, covariance, population moment, association, "
                "scatter, inertia, distance, area or image footprint is inferred."
            ),
            "constructed_reference_only": (
                "The inherited pure-reflection correspondent remains a deterministic reference, "
                "not recovered fixture truth, an observed image, a counterfactual rollout or an "
                "independent measurement."
            ),
            "not_primary_endpoint": (
                "Rows have no request-start, generated-horizon or executed-time correspondence. "
                "They are not r0,rhat,r and answer neither A7 nor any A1-A9 question."
            ),
            "selection_and_confounding": (
                "The three mapping pairs remain availability-selected. D01/D02 are command-"
                "confounded and P00 has unequal command membership. No layout, command, model, "
                "task-success, population, representation or camera effect is estimated."
            ),
            "science_annotation_and_release": (
                "No raw bytes, labels, manifest, annotation eligibility, inference, forecast "
                "metric, current-state claim, stage gate, dispatch authority or confirmation "
                "release follows."
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
            "physical_pairwise_interaction_inferred": False,
            "physical_partition_or_set_relation_inferred": False,
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
            "orientation_or_chirality_inferred": False,
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
            "novel_numeric_evidence_created": False,
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
            "Verify exact two-sided centered bipartition products for retained residual "
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
            "Verified 162 world and 324 projected exact centered bipartition-product "
            "identities; 1,458 exact record slots, 1,944 predecessor slots revalidated and "
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
