#!/usr/bin/env python3
"""Audit exact Gram identities for retained residual-space triangles only."""

from __future__ import annotations

import argparse
from fractions import Fraction
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
TURN2851_PACKAGE_COMMIT = "0f5312a304e119c0f5660da99b216cab5fe26831"
TURN2851_PACKAGE_TREE = "f0a3f2efba7c29653d01445df686e4e1bff7f036"
TURN2851_PACKAGE_PARENT = "0fbeee46221150f85d8bea7e1e53dbcac15e58f6"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2851_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_residual_triangle_exterior_product.py"
    ),
    "lines": 1020,
    "bytes": 41348,
    "sha256": "78aa64d769e1c7529cf0424e6dd70e2b7b20cb4d50c68b00f298ef24209f6605",
}
TURN2851_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_residual_triangle_exterior_product_diagnostic.json"
    ),
    "lines": 3968,
    "bytes": 138932,
    "sha256": "80f6a6783e2e30466bfa20df264b9f2846110638a54fb093d603c80a7fcffbfa",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_gram_identity_diagnostic.json"
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
EXPECTED_WORLD_MAXIMUM = {
    "center_relation_reflection_residual": {
        "numerator": "600587784276310831225",
        "denominator": "1298074214633706907132624082305024",
        "layout_pair_id": "D02",
    },
    "aabb_midpoint_relation_reflection_residual": {
        "numerator": "887814619467788380221",
        "denominator": "5192296858534827628530496329220096",
        "layout_pair_id": "D02",
    },
    "center_minus_aabb_midpoint_relation_residual_difference": {
        "numerator": "36887548369",
        "denominator": "20769187434139310514121985316880384",
        "layout_pair_id": "D02",
    },
}
EXPECTED_PROJECTED_MAXIMUM = {
    "point_center_relation_reflection_residual": {
        "numerator": "21944469619137212024002863850470516564158723762251009",
        "denominator": "5986310706507378352962293074805895248510699696029696",
        "layout_pair_id": "D02",
        "camera_name": "wrist_cam",
    },
    "envelope_center_relation_reflection_residual": {
        "numerator": "43798080987204409892553547772269009997562908975569",
        "denominator": "23384026197294446691258957323460528314494920687616",
        "layout_pair_id": "D02",
        "camera_name": "wrist_cam",
    },
    "point_minus_envelope_relation_residual_difference": {
        "numerator": "163273266888040406253942121695333010579555373556441",
        "denominator": "23945242826029513411849172299223580994042798784118784",
        "layout_pair_id": "D02",
        "camera_name": "wrist_cam",
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


def load_turn2851_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2851_PACKAGE_COMMIT}^{{commit}}")
        == TURN2851_PACKAGE_COMMIT,
        "Pinned Turn2851 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2851_PACKAGE_COMMIT)
        == TURN2851_PACKAGE_TREE,
        "Pinned Turn2851 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2851_PACKAGE_COMMIT)
        == TURN2851_PACKAGE_PARENT,
        "Pinned Turn2851 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2851_PACKAGE_COMMIT, TURN2851_AUDIT["path"])
    verify_payload(committed_audit, TURN2851_AUDIT, "Pinned Turn2851 audit")
    local_audit_path = REPOSITORY / TURN2851_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2851_AUDIT, "Local Turn2851 audit")
    require(local_audit == committed_audit, "Local Turn2851 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2851_exterior_product")

    committed_report = git_bytes(TURN2851_PACKAGE_COMMIT, TURN2851_REPORT["path"])
    verify_payload(committed_report, TURN2851_REPORT, "Pinned Turn2851 report")
    local_report = (REPOSITORY / TURN2851_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2851_REPORT, "Local Turn2851 report")
    require(local_report == committed_report, "Local Turn2851 report differs")
    parsed = strict_json_loads(committed_report, TURN2851_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2851 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2851 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-residual-triangle-exterior-product-diagnostic-v1",
        "Turn2851 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["exhaustive_object_pairs_per_mapping"] == 3
        and selection["world_source_pair_rows"] == 9
        and selection["projected_source_pair_rows"] == 36
        and selection["world_triangle_rows"] == 9
        and selection["world_signed_cross_product_components"] == 27
        and selection["world_cross_product_norms"] == 9
        and selection["mapping_pair_camera_contexts"] == 12
        and selection["projected_determinant_rows"] == 36
        and selection["availability_selected"] is True
        and selection["outcome_dependent_selection"] is False
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection[
            "P00_shared_left_command_with_extra_reflected_right_member"
        ]
        is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2851 selection boundary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 41
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2851 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2850"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["physical_scene_triangle_area_measured"] is False
        and parsed["image_footprint_or_silhouette_area_measured"] is False
        and parsed["physical_plane_normal_inferred"] is False
        and parsed["orientation_or_chirality_inferred"] is False
        and parsed["collinearity_or_degeneracy_classified"] is False
        and parsed["a7_result"] is False
        and parsed["r0_rhat_r_endpoint"] is False
        and parsed["motion_or_temporal_transition_inferred"] is False
        and parsed["generated_or_executed_correspondence"] is False
        and parsed["normalization_applied"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False,
        "Turn2851 direct boundary changed",
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
        "Turn2851 zero-activity boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2851_PACKAGE_COMMIT,
        "tree": TURN2851_PACKAGE_TREE,
        "parent": TURN2851_PACKAGE_PARENT,
        "audit": TURN2851_AUDIT,
        "report": TURN2851_REPORT,
        "canonical_regeneration_verified": True,
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
    return [
        finite_number(component, f"{label}[{index}]")
        for index, component in enumerate(value)
    ]


def exact_vector(values: list[float]) -> list[Fraction]:
    return [Fraction.from_float(value) for value in values]


def dot(first: list[Fraction], second: list[Fraction]) -> Fraction:
    require(len(first) == len(second), "Dot-product shape changed")
    return sum((left * right for left, right in zip(first, second)), Fraction(0))


def cross(first: list[Fraction], second: list[Fraction]) -> list[Fraction]:
    require(len(first) == len(second) == 3, "Cross-product shape changed")
    return [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]


def float_cross(first: list[float], second: list[float]) -> list[float]:
    require(len(first) == len(second) == 3, "Float cross-product shape changed")
    return [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]


def float_determinant(first: list[float], second: list[float]) -> float:
    require(len(first) == len(second) == 2, "Float determinant shape changed")
    return first[0] * second[1] - first[1] * second[0]


def exact_record(value: Fraction, approximate_key: str) -> dict:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        approximate_key: float(value),
    }


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


def compact_provenance(provenance: dict) -> dict:
    keys = ["phase", "layout_pair_id", "model_id"]
    if "camera_name" in provenance:
        keys.extend(("camera_name", "camera_configuration_sha256"))
    return {key: provenance[key] for key in keys}


def validate_cycle(
    row: dict,
    edge_ab: list[float],
    edge_ac: list[float],
    edge_bc: list[float],
    unit: str,
) -> None:
    length = len(edge_ab)
    require(len(edge_ac) == len(edge_bc) == length, "Source edge shape changed")
    cycle = finite_vector(
        row.get(f"cycle_ab_plus_bc_minus_ac_{unit}"),
        length,
        "Source pair cycle",
    )
    require(cycle == [0.0] * length, "Source pair-cycle values changed")
    require(
        [edge_ab[index] + edge_bc[index] - edge_ac[index] for index in range(length)]
        == cycle,
        "Source pair-cycle identity no longer recomputes",
    )
    require(row.get("pair_cycle_identity_exact") is True, "Source cycle flag changed")
    require(
        row.get("fixed_object_and_operand_order") is True
        and row.get("normalization_applied") is False
        and row.get("scientific_threshold_applied") is False,
        "Source row boundary flag changed",
    )


def world_identity_row(row: dict) -> dict:
    provenance = validate_provenance(row.get("provenance"), projected=False)
    family = row.get("family")
    require(family in WORLD_FAMILIES, "World family changed")
    edge_ab_float = finite_vector(row.get("edge_ab_signed_m"), 3, "World edge AB")
    edge_ac_float = finite_vector(row.get("edge_ac_signed_m"), 3, "World edge AC")
    edge_bc_float = finite_vector(row.get("edge_bc_signed_m"), 3, "World edge BC")
    validate_cycle(row, edge_ab_float, edge_ac_float, edge_bc_float, "m")
    prior_exterior = row.get("exterior_product_edge_ab_cross_edge_ac")
    require(isinstance(prior_exterior, dict), "Prior world exterior product missing")
    prior_cross = finite_vector(prior_exterior.get("signed_m2"), 3, "Prior cross")
    require(
        prior_cross == float_cross(edge_ab_float, edge_ac_float),
        "Prior floating cross product changed",
    )

    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_ab_squared_norm = dot(edge_ab, edge_ab)
    edge_ac_squared_norm = dot(edge_ac, edge_ac)
    edge_dot = dot(edge_ab, edge_ac)
    gram = edge_ab_squared_norm * edge_ac_squared_norm - edge_dot * edge_dot
    exact_cross = cross(edge_ab, edge_ac)
    exterior_squared_norm = dot(exact_cross, exact_cross)
    residual = gram - exterior_squared_norm
    require(gram >= 0, "Exact world Gram determinant is negative")
    require(residual == 0, "Exact world Lagrange identity changed")
    return {
        "provenance": provenance,
        "family": family,
        "edge_ab_signed_m": edge_ab_float,
        "edge_ac_signed_m": edge_ac_float,
        "source_cross_product_signed_m2": prior_cross,
        "exact_stored_binary_float_rational_scalars": {
            "edge_ab_squared_norm_m2": exact_record(
                edge_ab_squared_norm, "approximate_m2"
            ),
            "edge_ac_squared_norm_m2": exact_record(
                edge_ac_squared_norm, "approximate_m2"
            ),
            "edge_ab_dot_edge_ac_m2": exact_record(edge_dot, "approximate_m2"),
            "gram_determinant_m4": exact_record(
                gram, "approximate_m4"
            ),
            "gram_determinant_mm4": exact_record(
                gram * 1_000_000_000_000, "approximate_mm4"
            ),
            "exterior_product_squared_norm_m4": exact_record(
                exterior_squared_norm, "approximate_m4"
            ),
            "lagrange_identity_residual_m4": exact_record(
                residual, "approximate_m4"
            ),
        },
        "lagrange_gram_identity_exact": True,
        "fixed_object_and_operand_order": True,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
    }


def projected_identity_row(row: dict) -> dict:
    provenance = validate_provenance(row.get("provenance"), projected=True)
    family = row.get("family")
    require(family in PROJECTED_FAMILIES, "Projected family changed")
    edge_ab_float = finite_vector(row.get("edge_ab_signed_px"), 2, "Projected edge AB")
    edge_ac_float = finite_vector(row.get("edge_ac_signed_px"), 2, "Projected edge AC")
    edge_bc_float = finite_vector(row.get("edge_bc_signed_px"), 2, "Projected edge BC")
    validate_cycle(row, edge_ab_float, edge_ac_float, edge_bc_float, "px")
    prior_record = row.get("determinant_edge_ab_edge_ac")
    require(isinstance(prior_record, dict), "Prior projected determinant missing")
    prior_determinant = finite_number(
        prior_record.get("signed_px2"), "Prior determinant"
    )
    require(
        prior_determinant == float_determinant(edge_ab_float, edge_ac_float),
        "Prior floating determinant changed",
    )

    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_ab_squared_norm = dot(edge_ab, edge_ab)
    edge_ac_squared_norm = dot(edge_ac, edge_ac)
    edge_dot = dot(edge_ab, edge_ac)
    gram = edge_ab_squared_norm * edge_ac_squared_norm - edge_dot * edge_dot
    exact_determinant = edge_ab[0] * edge_ac[1] - edge_ab[1] * edge_ac[0]
    determinant_squared = exact_determinant * exact_determinant
    residual = gram - determinant_squared
    require(gram >= 0, "Exact projected Gram determinant is negative")
    require(residual == 0, "Exact projected Gram identity changed")
    return {
        "provenance": provenance,
        "family": family,
        "edge_ab_signed_px": edge_ab_float,
        "edge_ac_signed_px": edge_ac_float,
        "source_determinant_signed_px2": prior_determinant,
        "exact_stored_binary_float_rational_scalars": {
            "edge_ab_squared_norm_px2": exact_record(
                edge_ab_squared_norm, "approximate_px2"
            ),
            "edge_ac_squared_norm_px2": exact_record(
                edge_ac_squared_norm, "approximate_px2"
            ),
            "edge_ab_dot_edge_ac_px2": exact_record(
                edge_dot, "approximate_px2"
            ),
            "gram_determinant_px4": exact_record(
                gram, "approximate_px4"
            ),
            "determinant_squared_px4": exact_record(
                determinant_squared, "approximate_px4"
            ),
            "gram_identity_residual_px4": exact_record(
                residual, "approximate_px4"
            ),
        },
        "determinant_gram_identity_exact": True,
        "fixed_object_and_operand_order": True,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
    }


def rational_from_record(record: dict) -> Fraction:
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def maximum_gram(rows: list[dict], family: str, projected: bool) -> dict:
    selected = [row for row in rows if row["family"] == family]
    expected_count = 12 if projected else 3
    require(len(selected) == expected_count, f"Family row count changed: {family}")
    key = "gram_determinant_px4" if projected else "gram_determinant_m4"
    approximate_key = "approximate_px4" if projected else "approximate_m4"
    values = [
        rational_from_record(
            row["exact_stored_binary_float_rational_scalars"][key]
        )
        for row in selected
    ]
    maximum = max(values)
    ties = [
        {
            **compact_provenance(row["provenance"]),
            **row["exact_stored_binary_float_rational_scalars"][key],
        }
        for row, value in zip(selected, values)
        if value == maximum
    ]
    return {
        "rows": len(selected),
        "maximum_exact_stored_coordinate_gram_determinant": {
            "numerator": str(maximum.numerator),
            "denominator": str(maximum.denominator),
            approximate_key: float(maximum),
            "tie_count": len(ties),
            "ties": ties,
        },
    }


def validate_expected_maxima(world: dict, projected: dict) -> None:
    for family, expected in EXPECTED_WORLD_MAXIMUM.items():
        maximum = world[family]["maximum_exact_stored_coordinate_gram_determinant"]
        require(maximum["numerator"] == expected["numerator"], f"{family} world maximum numerator changed")
        require(maximum["denominator"] == expected["denominator"], f"{family} world maximum denominator changed")
        require(maximum["tie_count"] == 1, f"{family} world maximum tie count changed")
        require(maximum["ties"][0]["layout_pair_id"] == expected["layout_pair_id"], f"{family} world maximum provenance changed")
    for family, expected in EXPECTED_PROJECTED_MAXIMUM.items():
        maximum = projected[family]["maximum_exact_stored_coordinate_gram_determinant"]
        require(maximum["numerator"] == expected["numerator"], f"{family} projected maximum numerator changed")
        require(maximum["denominator"] == expected["denominator"], f"{family} projected maximum denominator changed")
        require(maximum["tie_count"] == 1, f"{family} projected maximum tie count changed")
        tie = maximum["ties"][0]
        require(tie["layout_pair_id"] == expected["layout_pair_id"], f"{family} projected maximum mapping changed")
        require(tie["camera_name"] == expected["camera_name"], f"{family} projected maximum camera changed")


def analyze(source: dict) -> tuple[list[dict], list[dict], dict]:
    source_world = source.get("world_triangle_rows")
    source_projected = source.get("projected_determinant_rows")
    require(isinstance(source_world, list), "Source world rows missing")
    require(isinstance(source_projected, list), "Source projected rows missing")
    require(len(source_world) == 9, "Source world row count changed")
    require(len(source_projected) == 36, "Source projected row count changed")

    world_rows = [world_identity_row(row) for row in source_world]
    projected_rows = [projected_identity_row(row) for row in source_projected]
    world_keys = [
        (
            row["provenance"]["phase"],
            row["provenance"]["layout_pair_id"],
            row["provenance"]["model_id"],
            row["family"],
        )
        for row in world_rows
    ]
    projected_keys = [
        (
            row["provenance"]["phase"],
            row["provenance"]["layout_pair_id"],
            row["provenance"]["model_id"],
            row["provenance"]["camera_name"],
            row["family"],
        )
        for row in projected_rows
    ]
    require(len(set(world_keys)) == 9, "Duplicate world identity row")
    require(len(set(projected_keys)) == 36, "Duplicate projected identity row")
    require(
        {row["family"] for row in world_rows} == set(WORLD_FAMILIES),
        "World family set changed",
    )
    require(
        {row["family"] for row in projected_rows} == set(PROJECTED_FAMILIES),
        "Projected family set changed",
    )

    world_summaries = {
        family: maximum_gram(world_rows, family, projected=False)
        for family in WORLD_FAMILIES
    }
    projected_summaries = {
        family: maximum_gram(projected_rows, family, projected=True)
        for family in PROJECTED_FAMILIES
    }
    validate_expected_maxima(world_summaries, projected_summaries)
    summary = {
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_identity_rows": 9,
        "projected_identity_rows": 36,
        "total_exact_gram_identity_rows": 45,
        "world_source_cross_product_components_revalidated": 27,
        "projected_source_determinants_revalidated": 36,
        "source_pair_cycle_components_revalidated_exact": 99,
        "world_exact_rational_scalar_records": 63,
        "projected_exact_rational_scalar_records": 216,
        "total_exact_rational_scalar_records": 279,
        "all_gram_identities_exact": True,
        "world_families": world_summaries,
        "projected_families": projected_summaries,
    }
    return world_rows, projected_rows, summary


def audit() -> dict:
    _, source, source_pin = load_turn2851_package()
    world_rows, projected_rows, summary = analyze(source)
    return {
        "schema_version": "wmf-retained-residual-triangle-gram-identity-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "turn2851_exterior_product_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_source_triangle_rows": 9,
            "projected_source_determinant_rows": 36,
            "world_gram_identity_rows": 9,
            "projected_gram_identity_rows": 36,
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
                "Each finite JSON edge coordinate is interpreted as its exact binary "
                "floating-point rational value. Integer numerator/denominator strings "
                "make the identity checks tolerance-free and independently reproducible."
            ),
            "world_identity": (
                "For frozen eAB and eAC in three stored coordinates, the exact identity "
                "||eAB||^2||eAC||^2-(eAB dot eAC)^2=||eAB cross eAC||^2 "
                "is verified over rational arithmetic."
            ),
            "projected_identity": (
                "For frozen eAB and eAC in two stored image-plane coordinates, the exact "
                "identity ||eAB||^2||eAC||^2-(eAB dot eAC)^2=det(eAB,eAC)^2 "
                "is verified over rational arithmetic."
            ),
            "fixed_order": (
                "A=banana,B=bowl,C=rubiks_cube; eAB=A-B and eAC=A-C. No "
                "operand, object, coordinate or camera reordering is performed."
            ),
            "units_and_approximations": (
                "World Gram determinants are in m^4 with descriptive mm^4 copies; "
                "projected Gram determinants are in px^4. Approximate JSON numbers are "
                "display aids only; exact numerator/denominator strings control equality. "
                "No normalization or tolerance is applied."
            ),
        },
        "summary": summary,
        "world_gram_identity_rows": world_rows,
        "projected_gram_identity_rows": projected_rows,
        "algebraically_derived_from_turn2851": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "physical_scene_triangle_area_measured": False,
        "image_footprint_or_silhouette_area_measured": False,
        "physical_plane_normal_inferred": False,
        "orientation_or_chirality_inferred": False,
        "collinearity_or_degeneracy_classified": False,
        "angle_or_similarity_inferred": False,
        "coordinate_or_frame_invariance_claimed": False,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
        "a7_result": False,
        "r0_rhat_r_endpoint": False,
        "motion_or_temporal_transition_inferred": False,
        "generated_or_executed_correspondence": False,
        "included_in_existing_six_report_annotation_closure": False,
        "finding": (
            "All 9 world and 36 projected retained residual-triangle rows satisfy "
            "their exact rational Gram identities, with 279 serialized exact scalar "
            "records and zero identity residual in every row. This is a deterministic "
            "stored-coordinate consistency result only, not evidence about physical "
            "area, orientation, collinearity, visibility, motion, forecast skill, "
            "effects or any scientific gate."
        ),
        "claim_boundaries": {
            "exactness_domain": (
                "Exact means equality over rational interpretations of the pinned finite "
                "binary floating-point coordinates. It does not authenticate source "
                "precision, physical truth, uncertainty or measurement accuracy."
            ),
            "residual_space_not_physical_geometry": (
                "Gram determinants and exterior-product squares are residual-vector-space "
                "algebra. They are not physical scene areas, plane normals, object "
                "orientations, image footprints, masks, silhouettes or visible areas."
            ),
            "no_degeneracy_or_angle_claim": (
                "No threshold is used and no row is classified for collinearity, "
                "degeneracy, angle, similarity, conditioning, quality or acceptance."
            ),
            "coordinate_description_only": (
                "The reported operands and approximate extrema describe the frozen "
                "stored coordinate representation. No physical, camera, frame or "
                "representation invariance or ranking is claimed."
            ),
            "constructed_reference_only": (
                "The inherited constructed pure-reflection correspondent remains a "
                "deterministic reference, not recovered fixture truth, an observed "
                "image, a counterfactual rollout or an independent measurement."
            ),
            "not_primary_endpoint": (
                "Rows have no request-start, generated-horizon or executed-time "
                "correspondence. They are not r0,rhat,r and answer neither A7 nor any "
                "A1-A9 question."
            ),
            "selection_and_confounding": (
                "The three mapping pairs remain availability-selected. D01/D02 are "
                "command-confounded and P00 has unequal command membership. No layout, "
                "command, model, task-success, population or camera effect is estimated."
            ),
            "science_and_release": (
                "No threshold, fitting, registration, normalization, rank, forecast "
                "metric, current-state inference, qualification, gate advance, dispatch "
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
            "physical_scene_triangle_area_measured": False,
            "image_footprint_or_object_pixels_inferred": False,
            "image_footprint_or_silhouette_area_measured": False,
            "silhouette_or_segmentation_analyzed": False,
            "obb_or_exact_mesh_analyzed": False,
            "physical_plane_normal_inferred": False,
            "orientation_or_chirality_inferred": False,
            "collinearity_or_degeneracy_classified": False,
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
            "cross_time_field_join_performed": False,
            "counterfactual_or_temporal_transition_inferred": False,
            "motion_or_temporal_transition_inferred": False,
            "generated_or_executed_correspondence": False,
            "causal_effect_estimated": False,
            "isolated_layout_or_command_effect_estimated": False,
            "task_success_or_model_rate": False,
            "camera_ranking_estimated": False,
            "representation_ranking_estimated": False,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact retained residual-triangle Gram identities without "
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
            "Verified exact Gram identities for 9 world and 36 projected retained "
            "residual-triangle rows; 279 exact scalar records and no stage advance."
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
