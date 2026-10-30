#!/usr/bin/env python3
"""Audit retained three-object residual-space exterior products only."""

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
TURN2850_PACKAGE_COMMIT = "6cec04b0ff7bc98cec039b456b647390aa7b4acd"
TURN2850_PACKAGE_TREE = "1755eb4fc9ff1e74f617a19330daa60d36814357"
TURN2850_PACKAGE_PARENT = "74d69ff3ee7b4b26328ef80d86d78d77f42b9db6"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2850_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_pairwise_relation_reflection_residual.py"
    ),
    "lines": 1158,
    "bytes": 43931,
    "sha256": "729ab2f3303de3abc5e990523c96aec4a875c1413f3378bed8fc3a8c19d79538",
}
TURN2850_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_pairwise_relation_reflection_residual_diagnostic.json"
    ),
    "lines": 5986,
    "bytes": 197699,
    "sha256": "4dfcc4da20526d8314f28e41931fd5d3347054b2881b03269bf172dd6ef6c183",
}
OUTPUT_PATH = (
    REPOSITORY
    / "workshops/corl2026_world_models/results/"
    "retained_residual_triangle_exterior_product_diagnostic.json"
)

OBJECT_ORDER = ("banana", "bowl", "rubiks_cube")
PAIR_AB = ("banana", "bowl")
PAIR_AC = ("banana", "rubiks_cube")
PAIR_BC = ("bowl", "rubiks_cube")
PAIR_ORDER = (PAIR_AB, PAIR_AC, PAIR_BC)
COMPARISON_KEYS = (
    ("development", "D01", "D1"),
    ("development", "D02", "D1"),
    ("pilot", "P00", "N3"),
)
CAMERA_NAMES = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
WORLD_AXES = ("x", "y", "z")
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
        "x": {"negative": 1, "zero": 0, "positive": 2},
        "y": {"negative": 2, "zero": 0, "positive": 1},
        "z": {"negative": 2, "zero": 0, "positive": 1},
    },
    "aabb_midpoint_relation_reflection_residual": {
        "x": {"negative": 1, "zero": 0, "positive": 2},
        "y": {"negative": 2, "zero": 0, "positive": 1},
        "z": {"negative": 2, "zero": 0, "positive": 1},
    },
    "center_minus_aabb_midpoint_relation_residual_difference": {
        "x": {"negative": 1, "zero": 0, "positive": 2},
        "y": {"negative": 3, "zero": 0, "positive": 0},
        "z": {"negative": 2, "zero": 0, "positive": 1},
    },
}
EXPECTED_PROJECTED_SIGNS = {
    "point_center_relation_reflection_residual": {
        "negative": 4,
        "zero": 0,
        "positive": 8,
    },
    "envelope_center_relation_reflection_residual": {
        "negative": 3,
        "zero": 0,
        "positive": 9,
    },
    "point_minus_envelope_relation_residual_difference": {
        "negative": 6,
        "zero": 0,
        "positive": 6,
    },
}
EXPECTED_WORLD_EXTREMA = {
    "center_relation_reflection_residual": {
        "maximum_absolute_m2": 6.801963586822168e-07,
        "signed_m2": -6.801963586822168e-07,
        "axis": "z",
        "layout_pair_id": "D02",
        "maximum_norm_m2": 6.802029103057201e-07,
    },
    "aabb_midpoint_relation_reflection_residual": {
        "maximum_absolute_m2": 4.134945214562258e-07,
        "signed_m2": -4.134945214562258e-07,
        "axis": "z",
        "layout_pair_id": "D02",
        "maximum_norm_m2": 4.135055937587511e-07,
    },
    "center_minus_aabb_midpoint_relation_residual_difference": {
        "maximum_absolute_m2": 1.1772249841612847e-12,
        "signed_m2": -1.1772249841612847e-12,
        "axis": "z",
        "layout_pair_id": "D02",
        "maximum_norm_m2": 1.3326930814978523e-12,
    },
}
EXPECTED_PROJECTED_EXTREMA = {
    "point_center_relation_reflection_residual": {
        "maximum_absolute_px2": 1.9146214374472352,
        "signed_px2": 1.9146214374472352,
        "layout_pair_id": "D02",
        "camera_name": "wrist_cam",
    },
    "envelope_center_relation_reflection_residual": {
        "maximum_absolute_px2": 1.3685727761861106,
        "signed_px2": 1.3685727761861106,
        "layout_pair_id": "D02",
        "camera_name": "wrist_cam",
    },
    "point_minus_envelope_relation_residual_difference": {
        "maximum_absolute_px2": 0.08257487377372837,
        "signed_px2": -0.08257487377372837,
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


def load_turn2850_package() -> tuple[ModuleType, dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2850_PACKAGE_COMMIT}^{{commit}}")
        == TURN2850_PACKAGE_COMMIT,
        "Pinned Turn2850 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2850_PACKAGE_COMMIT)
        == TURN2850_PACKAGE_TREE,
        "Pinned Turn2850 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2850_PACKAGE_COMMIT)
        == TURN2850_PACKAGE_PARENT,
        "Pinned Turn2850 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2850_PACKAGE_COMMIT, TURN2850_AUDIT["path"])
    verify_payload(committed_audit, TURN2850_AUDIT, "Pinned Turn2850 audit")
    local_audit_path = REPOSITORY / TURN2850_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2850_AUDIT, "Local Turn2850 audit")
    require(local_audit == committed_audit, "Local Turn2850 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2850_pairwise_relation")

    committed_report = git_bytes(TURN2850_PACKAGE_COMMIT, TURN2850_REPORT["path"])
    verify_payload(committed_report, TURN2850_REPORT, "Pinned Turn2850 report")
    local_report = (REPOSITORY / TURN2850_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2850_REPORT, "Local Turn2850 report")
    require(local_report == committed_report, "Local Turn2850 report differs")
    parsed = strict_json_loads(committed_report, TURN2850_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2850 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2850 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-pairwise-relation-reflection-residual-diagnostic-v1",
        "Turn2850 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["exhaustive_object_pairs_per_mapping"] == 3
        and selection["world_relation_rows"] == 9
        and selection["world_total_vector_components"] == 81
        and selection["mapping_pair_camera_rows"] == 12
        and selection["projected_relation_rows"] == 36
        and selection["projected_total_vector_components"] == 216
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
        "Turn2850 selection boundary changed",
    )
    require(
        len(parsed["evidence_boundaries"]) == 33
        and all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2850 evidence boundary changed",
    )
    require(
        parsed["algebraically_derived_from_turn2849"] is True
        and parsed["independent_observation"] is False
        and parsed["new_scientific_sample"] is False
        and parsed["a7_result"] is False
        and parsed["r0_rhat_r_endpoint"] is False
        and parsed["motion_or_temporal_transition_inferred"] is False
        and parsed["generated_or_executed_correspondence"] is False
        and parsed["normalization_applied"] is False
        and parsed["included_in_existing_six_report_annotation_closure"] is False,
        "Turn2850 direct boundary changed",
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
        "Turn2850 zero-activity boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2850_PACKAGE_COMMIT,
        "tree": TURN2850_PACKAGE_TREE,
        "parent": TURN2850_PACKAGE_PARENT,
        "audit": TURN2850_AUDIT,
        "report": TURN2850_REPORT,
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
    return [finite_number(component, f"{label}[{index}]") for index, component in enumerate(value)]


def sign(value: float) -> str:
    if value < 0.0:
        return "negative"
    if value > 0.0:
        return "positive"
    return "zero"


def cross_product(first: list[float], second: list[float]) -> list[float]:
    require(len(first) == len(second) == 3, "Cross-product shape changed")
    return [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]


def determinant(first: list[float], second: list[float]) -> float:
    require(len(first) == len(second) == 2, "Determinant shape changed")
    return first[0] * second[1] - first[1] * second[0]


def cycle_residual(
    edge_ab: list[float], edge_ac: list[float], edge_bc: list[float]
) -> list[float]:
    require(len(edge_ab) == len(edge_ac) == len(edge_bc), "Cycle shape changed")
    return [
        edge_ab[index] + edge_bc[index] - edge_ac[index]
        for index in range(len(edge_ab))
    ]


def validate_source_provenance(provenance: object, projected: bool) -> dict:
    require(isinstance(provenance, dict), "Source provenance missing")
    required = {
        "phase",
        "layout_pair_id",
        "model_id",
        "first_object_name",
        "second_object_name",
        "pair_direction",
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
    require(provenance["pair_direction"] == "first_minus_second", "Pair direction changed")
    require(
        (provenance["first_object_name"], provenance["second_object_name"])
        in PAIR_ORDER,
        "Object pair changed",
    )
    return provenance


def source_key(provenance: dict, projected: bool) -> tuple[str, ...]:
    keys = ["phase", "layout_pair_id", "model_id"]
    if projected:
        keys.append("camera_name")
    keys.extend(("first_object_name", "second_object_name"))
    return tuple(str(provenance[key]) for key in keys)


def triangle_provenance(rows: list[dict], projected: bool) -> dict:
    require(len(rows) == 3, "Triangle source row count changed")
    provenances = [validate_source_provenance(row.get("provenance"), projected) for row in rows]
    require(
        [
            (provenance["first_object_name"], provenance["second_object_name"])
            for provenance in provenances
        ]
        == list(PAIR_ORDER),
        "Triangle pair order changed",
    )
    ignored = {"first_object_name", "second_object_name", "pair_direction"}
    shared = {key: value for key, value in provenances[0].items() if key not in ignored}
    for provenance in provenances[1:]:
        require(
            {key: value for key, value in provenance.items() if key not in ignored}
            == shared,
            "Triangle provenance differs across source pairs",
        )
    return {
        **shared,
        "object_order": list(OBJECT_ORDER),
        "edge_ab": "banana_minus_bowl",
        "edge_ac": "banana_minus_rubiks_cube",
        "edge_bc": "bowl_minus_rubiks_cube",
        "source_pair_direction": "first_minus_second",
        "source_turn2850_package_commit": TURN2850_PACKAGE_COMMIT,
    }


def validate_source_row(row: dict, family: str, unit: str, length: int) -> list[float]:
    require(
        row.get("published_per_object_residual_then_pair_subtraction") is True,
        "Source operation-order flag changed",
    )
    require(
        row.get("representation_difference_pair_identity_exact") is True,
        "Source representation-identity flag changed",
    )
    record = row.get(family)
    require(isinstance(record, dict), f"Missing source family: {family}")
    return finite_vector(record.get(f"signed_{unit}"), length, f"Source {family}")


def world_exterior_record(values: list[float]) -> dict:
    absolute = [abs(value) for value in values]
    norm = math.sqrt(sum(value * value for value in values))
    return {
        "signed_m2": values,
        "absolute_m2": absolute,
        "signed_mm2": [1_000_000.0 * value for value in values],
        "absolute_mm2": [1_000_000.0 * value for value in absolute],
        "component_signs": dict(zip(WORLD_AXES, map(sign, values))),
        "l1_m2": sum(absolute),
        "l1_mm2": 1_000_000.0 * sum(absolute),
        "norm_m2": norm,
        "norm_mm2": 1_000_000.0 * norm,
        "linf_m2": max(absolute),
        "linf_mm2": 1_000_000.0 * max(absolute),
        "twice_residual_space_triangle_area_m2": norm,
        "twice_residual_space_triangle_area_mm2": 1_000_000.0 * norm,
    }


def build_world_row(rows: list[dict], family: str) -> dict:
    provenance = triangle_provenance(rows, projected=False)
    edge_ab, edge_ac, edge_bc = [
        validate_source_row(row, family, "m", 3) for row in rows
    ]
    closure = cycle_residual(edge_ab, edge_ac, edge_bc)
    require(closure == [0.0, 0.0, 0.0], "World pair-cycle identity changed")
    exterior = cross_product(edge_ab, edge_ac)
    return {
        "provenance": provenance,
        "family": family,
        "edge_ab_signed_m": edge_ab,
        "edge_ac_signed_m": edge_ac,
        "edge_bc_signed_m": edge_bc,
        "cycle_ab_plus_bc_minus_ac_m": closure,
        "pair_cycle_identity_exact": True,
        "exterior_product_edge_ab_cross_edge_ac": world_exterior_record(exterior),
        "fixed_object_and_operand_order": True,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
    }


def build_projected_row(rows: list[dict], family: str) -> dict:
    provenance = triangle_provenance(rows, projected=True)
    edge_ab, edge_ac, edge_bc = [
        validate_source_row(row, family, "px", 2) for row in rows
    ]
    closure = cycle_residual(edge_ab, edge_ac, edge_bc)
    require(closure == [0.0, 0.0], "Projected pair-cycle identity changed")
    signed = determinant(edge_ab, edge_ac)
    return {
        "provenance": provenance,
        "family": family,
        "edge_ab_signed_px": edge_ab,
        "edge_ac_signed_px": edge_ac,
        "edge_bc_signed_px": edge_bc,
        "cycle_ab_plus_bc_minus_ac_px": closure,
        "pair_cycle_identity_exact": True,
        "determinant_edge_ab_edge_ac": {
            "signed_px2": signed,
            "absolute_px2": abs(signed),
            "sign": sign(signed),
            "twice_residual_space_triangle_area_px2": abs(signed),
        },
        "fixed_object_and_operand_order": True,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
    }


def compact_provenance(provenance: dict) -> dict:
    keys = ["phase", "layout_pair_id", "model_id"]
    if "camera_name" in provenance:
        keys.extend(("camera_name", "camera_configuration_sha256"))
    return {key: provenance[key] for key in keys}


def world_family_summary(rows: list[dict], family: str) -> dict:
    selected = [row for row in rows if row["family"] == family]
    require(len(selected) == 3, f"World family row count changed: {family}")
    candidates = []
    census = {
        axis: {"negative": 0, "zero": 0, "positive": 0} for axis in WORLD_AXES
    }
    zero_components = 0
    zero_rows = 0
    for row in selected:
        record = row["exterior_product_edge_ab_cross_edge_ac"]
        values = record["signed_m2"]
        zero_components += sum(value == 0.0 for value in values)
        zero_rows += record["norm_m2"] == 0.0
        for index, axis in enumerate(WORLD_AXES):
            census[axis][sign(values[index])] += 1
            candidates.append((abs(values[index]), row, index, axis))
    maximum = max(candidate[0] for candidate in candidates)
    component_ties = []
    for absolute, row, index, axis in candidates:
        if absolute == maximum:
            values = row["exterior_product_edge_ab_cross_edge_ac"]["signed_m2"]
            component_ties.append(
                {
                    **compact_provenance(row["provenance"]),
                    "axis": axis,
                    "signed_m2": values[index],
                    "signed_mm2": 1_000_000.0 * values[index],
                    "absolute_m2": absolute,
                    "absolute_mm2": 1_000_000.0 * absolute,
                }
            )
    maximum_norm = max(
        row["exterior_product_edge_ab_cross_edge_ac"]["norm_m2"]
        for row in selected
    )
    norm_ties = []
    for row in selected:
        record = row["exterior_product_edge_ab_cross_edge_ac"]
        if record["norm_m2"] == maximum_norm:
            norm_ties.append(
                {
                    **compact_provenance(row["provenance"]),
                    "signed_m2": record["signed_m2"],
                    "signed_mm2": record["signed_mm2"],
                    "norm_m2": maximum_norm,
                    "norm_mm2": 1_000_000.0 * maximum_norm,
                }
            )
    return {
        "rows": len(selected),
        "signed_cross_product_components": len(selected) * 3,
        "exact_zero_components": zero_components,
        "exact_zero_rows": zero_rows,
        "component_sign_census": census,
        "maximum_absolute_component": {
            "maximum_absolute_m2": maximum,
            "maximum_absolute_mm2": 1_000_000.0 * maximum,
            "tie_count": len(component_ties),
            "ties": component_ties,
        },
        "maximum_norm_row": {
            "maximum_norm_m2": maximum_norm,
            "maximum_norm_mm2": 1_000_000.0 * maximum_norm,
            "tie_count": len(norm_ties),
            "ties": norm_ties,
        },
    }


def projected_family_summary(rows: list[dict], family: str) -> dict:
    selected = [row for row in rows if row["family"] == family]
    require(len(selected) == 12, f"Projected family row count changed: {family}")
    census = {"negative": 0, "zero": 0, "positive": 0}
    for row in selected:
        census[row["determinant_edge_ab_edge_ac"]["sign"]] += 1
    maximum = max(
        row["determinant_edge_ab_edge_ac"]["absolute_px2"] for row in selected
    )
    ties = []
    for row in selected:
        record = row["determinant_edge_ab_edge_ac"]
        if record["absolute_px2"] == maximum:
            ties.append(
                {
                    **compact_provenance(row["provenance"]),
                    "signed_px2": record["signed_px2"],
                    "absolute_px2": maximum,
                }
            )
    return {
        "rows": len(selected),
        "signed_determinants": len(selected),
        "exact_zero_determinants": census["zero"],
        "sign_census": census,
        "maximum_absolute_determinant": {
            "maximum_absolute_px2": maximum,
            "tie_count": len(ties),
            "ties": ties,
        },
    }


def projected_group_extrema(
    rows: list[dict], group_keys: tuple[str, ...]
) -> list[dict]:
    groups = {}
    for row in rows:
        group = tuple(row["provenance"][key] for key in group_keys)
        groups.setdefault(group, []).append(row)
    results = []
    for group in sorted(groups):
        for family in PROJECTED_FAMILIES:
            selected = [row for row in groups[group] if row["family"] == family]
            require(selected, "Projected grouped family is empty")
            maximum = max(
                row["determinant_edge_ab_edge_ac"]["absolute_px2"]
                for row in selected
            )
            ties = []
            for row in selected:
                record = row["determinant_edge_ab_edge_ac"]
                if record["absolute_px2"] == maximum:
                    ties.append(
                        {
                            **compact_provenance(row["provenance"]),
                            "signed_px2": record["signed_px2"],
                            "absolute_px2": maximum,
                        }
                    )
            results.append(
                {
                    **dict(zip(group_keys, group)),
                    "family": family,
                    "rows": len(selected),
                    "maximum_absolute_determinant": {
                        "maximum_absolute_px2": maximum,
                        "tie_count": len(ties),
                        "ties": ties,
                    },
                }
            )
    return results


def validate_expected_summaries(world: dict, projected: dict) -> None:
    for family in WORLD_FAMILIES:
        summary = world[family]
        expected = EXPECTED_WORLD_EXTREMA[family]
        require(summary["component_sign_census"] == EXPECTED_WORLD_SIGNS[family], f"{family} world signs")
        require(summary["exact_zero_components"] == 0 and summary["exact_zero_rows"] == 0, f"{family} world zeros")
        component = summary["maximum_absolute_component"]
        norm = summary["maximum_norm_row"]
        require(component["maximum_absolute_m2"] == expected["maximum_absolute_m2"], f"{family} world maximum")
        require(component["tie_count"] == 1, f"{family} world component tie count")
        require(component["ties"][0]["signed_m2"] == expected["signed_m2"], f"{family} world signed maximum")
        require(component["ties"][0]["axis"] == expected["axis"], f"{family} world maximum axis")
        require(component["ties"][0]["layout_pair_id"] == expected["layout_pair_id"], f"{family} world maximum mapping")
        require(norm["maximum_norm_m2"] == expected["maximum_norm_m2"], f"{family} world maximum norm")
        require(norm["tie_count"] == 1 and norm["ties"][0]["layout_pair_id"] == expected["layout_pair_id"], f"{family} world norm provenance")
    for family in PROJECTED_FAMILIES:
        summary = projected[family]
        expected = EXPECTED_PROJECTED_EXTREMA[family]
        require(summary["sign_census"] == EXPECTED_PROJECTED_SIGNS[family], f"{family} projected signs")
        require(summary["exact_zero_determinants"] == 0, f"{family} projected zeros")
        maximum = summary["maximum_absolute_determinant"]
        require(maximum["maximum_absolute_px2"] == expected["maximum_absolute_px2"], f"{family} projected maximum")
        require(maximum["tie_count"] == 1, f"{family} projected tie count")
        tie = maximum["ties"][0]
        require(tie["signed_px2"] == expected["signed_px2"], f"{family} projected signed maximum")
        require(tie["layout_pair_id"] == expected["layout_pair_id"], f"{family} projected mapping")
        require(tie["camera_name"] == expected["camera_name"], f"{family} projected camera")


def analyze(source: dict) -> tuple[list[dict], list[dict], dict]:
    source_world = {}
    for row in source.get("world_relation_rows", []):
        provenance = validate_source_provenance(row.get("provenance"), False)
        key = source_key(provenance, False)
        require(key not in source_world, f"Duplicate source world row: {key}")
        source_world[key] = row
    source_projected = {}
    for row in source.get("projected_relation_rows", []):
        provenance = validate_source_provenance(row.get("provenance"), True)
        key = source_key(provenance, True)
        require(key not in source_projected, f"Duplicate source projected row: {key}")
        source_projected[key] = row
    require(len(source_world) == 9, "Source world row count changed")
    require(len(source_projected) == 36, "Source projected row count changed")

    world_rows = []
    projected_rows = []
    for comparison in COMPARISON_KEYS:
        source_rows = [source_world[(*comparison, *pair)] for pair in PAIR_ORDER]
        for family in WORLD_FAMILIES:
            world_rows.append(build_world_row(source_rows, family))
        for camera_name in CAMERA_NAMES:
            source_rows = [
                source_projected[(*comparison, camera_name, *pair)]
                for pair in PAIR_ORDER
            ]
            for family in PROJECTED_FAMILIES:
                projected_rows.append(build_projected_row(source_rows, family))
    require(len(world_rows) == 9, "World triangle row count changed")
    require(len(projected_rows) == 36, "Projected determinant row count changed")

    world_summaries = {
        family: world_family_summary(world_rows, family) for family in WORLD_FAMILIES
    }
    projected_summaries = {
        family: projected_family_summary(projected_rows, family)
        for family in PROJECTED_FAMILIES
    }
    validate_expected_summaries(world_summaries, projected_summaries)
    summary = {
        "object_order": list(OBJECT_ORDER),
        "edge_order": ["edge_ab", "edge_ac", "edge_bc"],
        "edge_definitions": {
            "edge_ab": "banana_minus_bowl",
            "edge_ac": "banana_minus_rubiks_cube",
            "edge_bc": "bowl_minus_rubiks_cube",
        },
        "source_pair_direction": "first_minus_second",
        "original_reflected_mapping_pairs": 3,
        "mapping_camera_contexts": 12,
        "world_triangle_rows": 9,
        "world_signed_cross_product_components": 27,
        "world_cross_product_norms": 9,
        "projected_determinant_rows": 36,
        "world_pair_cycle_components_verified_exact": 27,
        "projected_pair_cycle_components_verified_exact": 72,
        "total_pair_cycle_components_verified_exact": 99,
        "world_families": world_summaries,
        "projected_families": projected_summaries,
        "projected_extrema_by_mapping": projected_group_extrema(
            projected_rows, ("phase", "layout_pair_id", "model_id")
        ),
        "projected_extrema_by_camera": projected_group_extrema(
            projected_rows, ("camera_name",)
        ),
    }
    return world_rows, projected_rows, summary


def audit() -> dict:
    _, source, source_pin = load_turn2850_package()
    world_rows, projected_rows, summary = analyze(source)
    return {
        "schema_version": (
            "wmf-retained-residual-triangle-exterior-product-diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2850_pairwise_relation_package": source_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "exhaustive_object_pairs_per_mapping": 3,
            "world_source_pair_rows": 9,
            "projected_source_pair_rows": 36,
            "world_triangle_rows": 9,
            "world_signed_cross_product_components": 27,
            "world_cross_product_norms": 9,
            "mapping_pair_camera_contexts": 12,
            "projected_determinant_rows": 36,
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
            "residual_vector_space_only": (
                "Rows describe coordinate algebra among retained reflection-residual "
                "vectors. They do not describe a physical scene triangle or an image "
                "footprint."
            ),
            "fixed_object_and_edge_order": (
                "A=banana,B=bowl,C=rubiks_cube; eAB=A-B,eAC=A-C,eBC=B-C. "
                "The exact stored pair cycle eAB+eBC=eAC is verified componentwise."
            ),
            "fixed_exterior_product_order": (
                "World rows use eAB cross eAC with component operation order "
                "[AB_y*AC_z-AB_z*AC_y,AB_z*AC_x-AB_x*AC_z,"
                "AB_x*AC_y-AB_y*AC_x]. Projected rows use "
                "AB_u*AC_v-AB_v*AC_u. Operand swaps would reverse signs and are not "
                "performed."
            ),
            "units": (
                "World exterior products use m^2 with exact 1,000,000x mm^2 copies. "
                "Their Euclidean norm equals twice the residual-vector-space triangle "
                "area. Projected determinants use px^2 and their absolute value equals "
                "twice the residual-vector-space triangle area. No normalization is "
                "applied."
            ),
        },
        "summary": summary,
        "world_triangle_rows": world_rows,
        "projected_determinant_rows": projected_rows,
        "algebraically_derived_from_turn2850": True,
        "independent_observation": False,
        "new_scientific_sample": False,
        "physical_scene_triangle_area_measured": False,
        "image_footprint_or_silhouette_area_measured": False,
        "physical_plane_normal_inferred": False,
        "orientation_or_chirality_inferred": False,
        "collinearity_or_degeneracy_classified": False,
        "normalization_applied": False,
        "scientific_threshold_applied": False,
        "a7_result": False,
        "r0_rhat_r_endpoint": False,
        "motion_or_temporal_transition_inferred": False,
        "generated_or_executed_correspondence": False,
        "included_in_existing_six_report_annotation_closure": False,
        "finding": (
            "Across three availability-selected original/reflected mapping pairs, the "
            "largest absolute world center-family cross-product component is "
            "0.6801963586822168 mm^2 on D02 z (signed value "
            "-0.6801963586822168 mm^2), and that row's norm is "
            "0.6802029103057201 mm^2. Across 12 mapping-camera contexts, the largest "
            "absolute projected point-family determinant is 1.9146214374472352 px^2 "
            "on D02 wrist_cam (signed value +1.9146214374472352 px^2); the largest "
            "absolute projected point-minus-envelope determinant is "
            "0.08257487377372837 px^2 there (signed value -0.08257487377372837 px^2). "
            "These are coordinate/order-dependent residual-vector-space descriptions, "
            "not physical areas, orientations, effects, forecast endpoints or "
            "scientific gate results."
        ),
        "claim_boundaries": {
            "residual_space_not_physical_geometry": (
                "The world cross product is not a physical scene triangle area, plane "
                "normal, object orientation or chirality. The projected determinant is "
                "not a silhouette, mask, footprint, visible area or image-content "
                "measurement."
            ),
            "coordinate_and_order_dependence": (
                "Cross-product component signs and projected determinant signs depend "
                "on coordinate and frozen object/operand order. They are not semantic "
                "left/right, front/back or clockwise/counterclockwise claims."
            ),
            "exact_zero_boundary": (
                "Exact zero or nonzero refers only to stored floating-point arithmetic. "
                "No tolerance, physical collinearity, degeneracy, shape quality or "
                "acceptance threshold is inferred."
            ),
            "constructed_reference_only": (
                "The inherited constructed pure-reflection correspondent remains a "
                "deterministic reference, not recovered fixture truth, an observed "
                "image, a counterfactual rollout or an independent measurement."
            ),
            "not_primary_endpoint": (
                "Residual-space exterior products are not human-marked cube-minus-bowl "
                "centroid offsets and have no request-start, generated-horizon or "
                "executed-time correspondence. They are not r0,rhat,r and do not answer "
                "A7 or any A1-A9 question."
            ),
            "selection_and_confounding": (
                "The fixed object-pair universe is exhausted without outcome selection, "
                "but the three mapping pairs remain availability-selected. D01/D02 are "
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
            "Describe retained residual-vector-space exterior products without "
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
            "Verified 9 world residual-space exterior-product rows / 27 signed "
            "components and 36 projected determinant rows; 99 exact pair-cycle "
            "components and no stage advance."
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
