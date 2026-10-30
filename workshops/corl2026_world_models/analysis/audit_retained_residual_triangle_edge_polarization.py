#!/usr/bin/env python3
"""Audit exact edge-cycle polarization identities for retained residual triangles."""

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
TURN2852_PACKAGE_COMMIT = "9d7ec27e11aac1c653e9835ba3b30370aee0280e"
TURN2852_PACKAGE_TREE = "21139dedd3ddda2481b56a67fb1ccb320d6f72fb"
TURN2852_PACKAGE_PARENT = "1a5a2a4b0056881b56d69e6496f84247392658fa"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2852_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_residual_triangle_gram_identity.py"
    ),
    "lines": 893,
    "bytes": 36897,
    "sha256": "625bbe0447ebe58c71f73a0e084fd661912e0a7c4628a124be35eb4aaf5aea8a",
}
TURN2852_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_residual_triangle_gram_identity_diagnostic.json"
    ),
    "lines": 4025,
    "bytes": 157303,
    "sha256": "9072ecdecedaeef4715946ccdb8b08d166969d39f01dc41db74c60dd6b9c8b2c",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_edge_polarization_diagnostic.json"
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


def load_turn2852_package() -> tuple[ModuleType, dict, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2852_PACKAGE_COMMIT}^{{commit}}")
        == TURN2852_PACKAGE_COMMIT,
        "Pinned Turn2852 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2852_PACKAGE_COMMIT)
        == TURN2852_PACKAGE_TREE,
        "Pinned Turn2852 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2852_PACKAGE_COMMIT)
        == TURN2852_PACKAGE_PARENT,
        "Pinned Turn2852 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2852_PACKAGE_COMMIT, TURN2852_AUDIT["path"])
    verify_payload(committed_audit, TURN2852_AUDIT, "Pinned Turn2852 audit")
    local_audit_path = REPOSITORY / TURN2852_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2852_AUDIT, "Local Turn2852 audit")
    require(local_audit == committed_audit, "Local Turn2852 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2852_gram_identity")

    committed_report = git_bytes(TURN2852_PACKAGE_COMMIT, TURN2852_REPORT["path"])
    verify_payload(committed_report, TURN2852_REPORT, "Pinned Turn2852 report")
    local_report = (REPOSITORY / TURN2852_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2852_REPORT, "Local Turn2852 report")
    require(local_report == committed_report, "Local Turn2852 report differs")
    parsed = strict_json_loads(committed_report, TURN2852_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2852 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2852 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-residual-triangle-gram-identity-diagnostic-v1",
        "Turn2852 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_source_triangle_rows"] == 9
        and selection["projected_source_determinant_rows"] == 36
        and selection["world_gram_identity_rows"] == 9
        and selection["projected_gram_identity_rows"] == 36
        and selection["availability_selected"] is True
        and selection["outcome_dependent_selection"] is False
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"]
        is True
        and selection["isolated_layout_or_command_effect_estimated"] is False,
        "Turn2852 selection boundary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 43
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2852 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2851"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["physical_scene_triangle_area_measured"] is False
        and parsed["image_footprint_or_silhouette_area_measured"] is False
        and parsed["physical_plane_normal_inferred"] is False
        and parsed["orientation_or_chirality_inferred"] is False
        and parsed["collinearity_or_degeneracy_classified"] is False
        and parsed["angle_or_similarity_inferred"] is False
        and parsed["coordinate_or_frame_invariance_claimed"] is False
        and parsed["normalization_applied"] is False
        and parsed["scientific_threshold_applied"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False,
        "Turn2852 direct boundary changed",
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
        "Turn2852 zero-activity boundary changed",
    )
    _, exterior, exterior_pin = prior.load_turn2851_package()
    return prior, parsed, exterior, {
        "commit": TURN2852_PACKAGE_COMMIT,
        "tree": TURN2852_PACKAGE_TREE,
        "parent": TURN2852_PACKAGE_PARENT,
        "audit": TURN2852_AUDIT,
        "report": TURN2852_REPORT,
        "canonical_regeneration_verified": True,
        "turn2851_exterior_product_package": exterior_pin,
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


def dot(first: list[Fraction], second: list[Fraction]) -> Fraction:
    require(len(first) == len(second), "Dot-product shape changed")
    return sum((left * right for left, right in zip(first, second)), Fraction(0))


def exact_record(value: Fraction, approximate_key: str) -> dict:
    return {
        "numerator": str(value.numerator),
        "denominator": str(value.denominator),
        approximate_key: float(value),
    }


def rational_from_record(record: object, label: str) -> Fraction:
    require(isinstance(record, dict), f"{label} record missing")
    require(
        set(record) == {"numerator", "denominator", next(
            key for key in record if key.startswith("approximate_")
        )},
        f"{label} record keys changed",
    )
    numerator = record.get("numerator")
    denominator = record.get("denominator")
    require(isinstance(numerator, str), f"{label} numerator changed")
    require(isinstance(denominator, str), f"{label} denominator changed")
    return Fraction(int(numerator), int(denominator))


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


def validate_source_cycle(
    row: dict,
    edge_ab_float: list[float],
    edge_ac_float: list[float],
    edge_bc_float: list[float],
    unit: str,
) -> None:
    length = len(edge_ab_float)
    source_cycle = finite_vector(
        row.get(f"cycle_ab_plus_bc_minus_ac_{unit}"),
        length,
        "Source pair cycle",
    )
    require(source_cycle == [0.0] * length, "Source pair-cycle values changed")
    require(
        [
            edge_ab_float[index] + edge_bc_float[index] - edge_ac_float[index]
            for index in range(length)
        ]
        == source_cycle,
        "Source floating pair-cycle identity changed",
    )
    require(row.get("pair_cycle_identity_exact") is True, "Source cycle flag changed")
    require(
        row.get("fixed_object_and_operand_order") is True
        and row.get("normalization_applied") is False
        and row.get("scientific_threshold_applied") is False,
        "Source row boundary flag changed",
    )


def validate_gram_row(
    gram_row: dict,
    edge_ab: list[Fraction],
    edge_ac: list[Fraction],
    unit: str,
) -> int:
    scalars = gram_row.get("exact_stored_binary_float_rational_scalars")
    require(isinstance(scalars, dict), "Turn2852 exact scalar map missing")
    suffix = f"{unit}2"
    expected = {
        f"edge_ab_squared_norm_{suffix}": dot(edge_ab, edge_ab),
        f"edge_ac_squared_norm_{suffix}": dot(edge_ac, edge_ac),
        f"edge_ab_dot_edge_ac_{suffix}": dot(edge_ab, edge_ac),
    }
    if unit == "m":
        expected["gram_determinant_m4"] = (
            expected["edge_ab_squared_norm_m2"]
            * expected["edge_ac_squared_norm_m2"]
            - expected["edge_ab_dot_edge_ac_m2"] ** 2
        )
    else:
        expected["gram_determinant_px4"] = (
            expected["edge_ab_squared_norm_px2"]
            * expected["edge_ac_squared_norm_px2"]
            - expected["edge_ab_dot_edge_ac_px2"] ** 2
        )
    for key, value in expected.items():
        require(
            rational_from_record(scalars.get(key), f"Turn2852 {key}") == value,
            f"Turn2852 {key} no longer matches exact source operands",
        )
    return len(expected)


def polarization_row(source_row: dict, gram_row: dict, projected: bool) -> tuple[dict, int]:
    unit = "px" if projected else "m"
    length = 2 if projected else 3
    families = PROJECTED_FAMILIES if projected else WORLD_FAMILIES
    provenance = validate_provenance(source_row.get("provenance"), projected)
    family = source_row.get("family")
    require(family in families, "Source family changed")
    require(row_key(source_row, projected) == row_key(gram_row, projected), "Source/Gram row key differs")
    require(gram_row.get("provenance") == provenance, "Source/Gram provenance differs")

    edge_ab_float = finite_vector(source_row.get(f"edge_ab_signed_{unit}"), length, "Edge AB")
    edge_ac_float = finite_vector(source_row.get(f"edge_ac_signed_{unit}"), length, "Edge AC")
    edge_bc_float = finite_vector(source_row.get(f"edge_bc_signed_{unit}"), length, "Edge BC")
    require(gram_row.get(f"edge_ab_signed_{unit}") == edge_ab_float, "Turn2852 edge AB changed")
    require(gram_row.get(f"edge_ac_signed_{unit}") == edge_ac_float, "Turn2852 edge AC changed")
    validate_source_cycle(source_row, edge_ab_float, edge_ac_float, edge_bc_float, unit)

    edge_ab = exact_vector(edge_ab_float)
    edge_ac = exact_vector(edge_ac_float)
    edge_bc = exact_vector(edge_bc_float)
    exact_cycle = [
        edge_ab[index] + edge_bc[index] - edge_ac[index]
        for index in range(length)
    ]
    require(exact_cycle == [Fraction(0)] * length, "Exact rational edge cycle changed")

    ab2 = dot(edge_ab, edge_ab)
    ac2 = dot(edge_ac, edge_ac)
    bc2 = dot(edge_bc, edge_bc)
    ab_ac = dot(edge_ab, edge_ac)
    ab_bc = dot(edge_ab, edge_bc)
    ac_bc = dot(edge_ac, edge_bc)
    residuals = {
        "ab_ac": 2 * ab_ac - (ab2 + ac2 - bc2),
        "ab_bc": 2 * ab_bc - (ac2 - ab2 - bc2),
        "ac_bc": 2 * ac_bc - (ac2 + bc2 - ab2),
    }
    require(all(value == 0 for value in residuals.values()), "Exact polarization identity changed")
    turn2852_records = validate_gram_row(gram_row, edge_ab, edge_ac, unit)
    approximate_key = f"approximate_{unit}2"
    scalars = {
        f"edge_ab_squared_norm_{unit}2": exact_record(ab2, approximate_key),
        f"edge_ac_squared_norm_{unit}2": exact_record(ac2, approximate_key),
        f"edge_bc_squared_norm_{unit}2": exact_record(bc2, approximate_key),
        f"edge_ab_dot_edge_ac_{unit}2": exact_record(ab_ac, approximate_key),
        f"edge_ab_dot_edge_bc_{unit}2": exact_record(ab_bc, approximate_key),
        f"edge_ac_dot_edge_bc_{unit}2": exact_record(ac_bc, approximate_key),
    }
    for name, value in residuals.items():
        scalars[f"polarization_identity_{name}_residual_{unit}2"] = exact_record(value, approximate_key)
    row = {
        "provenance": provenance,
        "family": family,
        f"edge_ab_signed_{unit}": edge_ab_float,
        f"edge_ac_signed_{unit}": edge_ac_float,
        f"edge_bc_signed_{unit}": edge_bc_float,
        f"exact_cycle_ab_plus_bc_minus_ac_{unit}": [
            exact_record(value, f"approximate_{unit}") for value in exact_cycle
        ],
        "exact_stored_binary_float_rational_scalars": scalars,
        "exact_edge_cycle_identity": True,
        "all_three_exact_polarization_identities": True,
        "fixed_object_and_operand_order": True,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
    }
    return row, turn2852_records


def analyze(gram: dict, exterior: dict) -> tuple[list[dict], list[dict], dict]:
    gram_world = gram.get("world_gram_identity_rows")
    gram_projected = gram.get("projected_gram_identity_rows")
    source_world = exterior.get("world_triangle_rows")
    source_projected = exterior.get("projected_determinant_rows")
    require(isinstance(gram_world, list) and len(gram_world) == 9, "Turn2852 world rows changed")
    require(isinstance(gram_projected, list) and len(gram_projected) == 36, "Turn2852 projected rows changed")
    require(isinstance(source_world, list) and len(source_world) == 9, "Source world rows changed")
    require(isinstance(source_projected, list) and len(source_projected) == 36, "Source projected rows changed")

    gram_world_by_key = {row_key(row, False): row for row in gram_world}
    gram_projected_by_key = {row_key(row, True): row for row in gram_projected}
    require(len(gram_world_by_key) == 9, "Duplicate Turn2852 world row")
    require(len(gram_projected_by_key) == 36, "Duplicate Turn2852 projected row")
    require(
        {row_key(row, False) for row in source_world} == set(gram_world_by_key),
        "World source/Turn2852 key set differs",
    )
    require(
        {row_key(row, True) for row in source_projected} == set(gram_projected_by_key),
        "Projected source/Turn2852 key set differs",
    )

    world_rows = []
    projected_rows = []
    turn2852_records = 0
    for source_row in source_world:
        key = row_key(source_row, False)
        row, records = polarization_row(source_row, gram_world_by_key[key], False)
        world_rows.append(row)
        turn2852_records += records
    for source_row in source_projected:
        key = row_key(source_row, True)
        row, records = polarization_row(source_row, gram_projected_by_key[key], True)
        projected_rows.append(row)
        turn2852_records += records

    world_records = 9 * (3 + 6 + 3)
    projected_records = 36 * (2 + 6 + 3)
    require(turn2852_records == 180, "Turn2852 exact record revalidation count changed")
    require(
        {row["family"] for row in world_rows} == set(WORLD_FAMILIES),
        "World family set changed",
    )
    require(
        {row["family"] for row in projected_rows} == set(PROJECTED_FAMILIES),
        "Projected family set changed",
    )
    summary = {
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_identity_rows": 9,
        "projected_identity_rows": 36,
        "total_exact_edge_cycle_and_polarization_rows": 45,
        "world_exact_cycle_components": 27,
        "projected_exact_cycle_components": 72,
        "total_exact_cycle_components": 99,
        "polarization_identities_per_row": 3,
        "total_exact_polarization_identities": 135,
        "turn2852_exact_rational_scalar_records_revalidated": turn2852_records,
        "world_serialized_exact_rational_records": world_records,
        "projected_serialized_exact_rational_records": projected_records,
        "total_serialized_exact_rational_records": world_records + projected_records,
        "all_exact_edge_cycles_zero": True,
        "all_exact_polarization_residuals_zero": True,
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
            require(not item.startswith(("/", "~", "file://", "ssh://")), f"Absolute/raw path forbidden at {location}")
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
    _, gram, exterior, source_pin = load_turn2852_package()
    world_rows, projected_rows, summary = analyze(gram, exterior)
    report = {
        "schema_version": "wmf-retained-residual-triangle-edge-polarization-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "turn2852_gram_identity_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_source_triangle_rows": 9,
            "projected_source_triangle_rows": 36,
            "world_edge_polarization_rows": 9,
            "projected_edge_polarization_rows": 36,
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
                "control equality; approximate numbers are display aids only."
            ),
            "fixed_order_and_cycle": (
                "A=banana,B=bowl,C=rubiks_cube; eAB=A-B,eAC=A-C,eBC=B-C, "
                "so the tested fixed-order cycle is eAB+eBC-eAC=0."
            ),
            "polarization_identities": (
                "For each frozen row, exact rational arithmetic verifies the three "
                "equivalent squared-norm/dot-product identities induced by the edge "
                "cycle. No square root, normalization, tolerance or threshold is used."
            ),
            "polarization_formula_ab_ac": (
                "The ab_ac residual is 2*(eAB dot eAC) - "
                "(||eAB||^2 + ||eAC||^2 - ||eBC||^2)."
            ),
            "polarization_formula_ab_bc": (
                "The ab_bc residual is 2*(eAB dot eBC) - "
                "(||eAC||^2 - ||eAB||^2 - ||eBC||^2)."
            ),
            "polarization_formula_ac_bc": (
                "The ac_bc residual is 2*(eAC dot eBC) - "
                "(||eAC||^2 + ||eBC||^2 - ||eAB||^2)."
            ),
            "units_and_scope": (
                "World scalar records use stored m coordinates and m^2 products; "
                "projected records use stored px coordinates and px^2 products. These "
                "are residual-vector-space coordinate identities, not measured physical "
                "lengths, distances, angles, areas or image-object geometry."
            ),
        },
        "summary": summary,
        "world_edge_polarization_rows": world_rows,
        "projected_edge_polarization_rows": projected_rows,
        "algebraically_derived_from_turn2852": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "physical_distance_or_length_measured": False,
        "physical_scene_triangle_area_measured": False,
        "image_footprint_or_silhouette_area_measured": False,
        "physical_plane_normal_inferred": False,
        "orientation_or_chirality_inferred": False,
        "collinearity_or_degeneracy_classified": False,
        "triangle_inequality_or_metric_classified": False,
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
            "All 9 world and 36 projected retained residual-triangle rows have zero "
            "exact rational eAB+eBC-eAC cycle residual and satisfy all three exact "
            "polarization identities. The package serializes 504 exact rational records "
            "and revalidates 180 Turn2852 exact scalars. This is a deterministic stored-"
            "coordinate consistency check only, not physical or scientific evidence."
        ),
        "claim_boundaries": {
            "exactness_domain": (
                "Exact means equality over rational interpretations of pinned finite "
                "binary floating-point values. It does not authenticate source precision, "
                "physical truth, uncertainty or measurement accuracy."
            ),
            "residual_space_not_physical_metric": (
                "Squared norms and dot products are algebra on retained residual vectors. "
                "They do not measure physical edge lengths, object separations, image "
                "footprints, silhouettes, angles, areas or metric quality."
            ),
            "no_classification_or_ranking": (
                "No square root, threshold, tolerance or normalization is used, and no "
                "row is classified or ranked for distance, triangle inequality, angle, "
                "similarity, collinearity, conditioning, quality or acceptance."
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
        len(report["evidence_boundaries"]) == 46
        and all(value is False for value in report["evidence_boundaries"].values()),
        "Output evidence boundary changed",
    )
    assert_output_hygiene(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify exact retained residual-triangle edge-cycle polarization identities "
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
            "Verified exact edge cycles and three polarization identities for 9 world "
            "and 36 projected retained residual-triangle rows; 504 exact records and "
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
