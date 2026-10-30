#!/usr/bin/env python3
"""Audit retained original-to-reflected AABB pair-geometry transitions only."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
PAIR_PACKAGE_COMMIT = "5dc9da6b7093beec48597f50f265e3b1c207669e"
REPOSITORY = Path(__file__).resolve().parents[3]
PAIR_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_aabb_pair_geometry.py"
    ),
    "bytes": 29316,
    "sha256": "afd3d390eab2aca7376769d05cc114e326411378b9ea75562e0177d1e37e75b5",
}
PAIR_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_aabb_pair_geometry_diagnostic.json"
    ),
    "bytes": 75778,
    "sha256": "6888c2b0e30e022ec30959fe44549df85c54030a39917ee197fc1ca44df595d7",
}
CAMERA_NAMES = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
OBJECT_PAIRS = (
    ("banana", "bowl"),
    ("banana", "rubiks_cube"),
    ("bowl", "rubiks_cube"),
)
COMPARISON_KEYS = (
    ("development", "D01", "D1"),
    ("development", "D02", "D1"),
    ("pilot", "P00", "N3"),
)
EXPECTED_COMMANDS = {
    ("development", "D01", "D1"): {
        "original": ("left",),
        "reflected": ("right",),
        "classification": "disjoint_single_commands_command_confounded",
    },
    ("development", "D02", "D1"): {
        "original": ("left",),
        "reflected": ("right",),
        "classification": "disjoint_single_commands_command_confounded",
    },
    ("pilot", "P00", "N3"): {
        "original": ("left",),
        "reflected": ("left", "right"),
        "classification": "shared_left_command_with_extra_reflected_right_member",
    },
}
PROJECTED_RELATIONS = (
    "positive_area_overlap",
    "positive_separation",
    "boundary_touch_only",
)
DEPTH_RELATIONS = (
    "first_strictly_nearer_in_camera_z",
    "second_strictly_nearer_in_camera_z",
    "overlapping_camera_z_intervals",
    "camera_z_interval_boundary_touch_only",
)
SCALAR_FIELDS = (
    "positive_overlap_area_px2",
    "euclidean_rectangle_separation_px",
    "positive_camera_z_interval_gap_m",
    "positive_camera_z_interval_overlap_m",
)
PAIR_ROW_KEYS = {
    "objects",
    "projected_envelope_relation",
    "axis_overlap_uv_px",
    "positive_overlap_area_px2",
    "axis_separation_uv_px",
    "euclidean_rectangle_separation_px",
    "camera_z_interval_relation",
    "strictly_nearer_object_in_camera_z",
    "positive_camera_z_interval_gap_m",
    "positive_camera_z_interval_overlap_m",
}
EXPECTED_PROJECTED_TRANSITIONS = {
    "positive_area_overlap->positive_area_overlap": 19,
    "positive_area_overlap->positive_separation": 1,
    "positive_separation->positive_separation": 16,
}
EXPECTED_DEPTH_TRANSITIONS = {
    (
        "first_strictly_nearer_in_camera_z->"
        "first_strictly_nearer_in_camera_z"
    ): 5,
    "overlapping_camera_z_intervals->overlapping_camera_z_intervals": 22,
    (
        "overlapping_camera_z_intervals->"
        "second_strictly_nearer_in_camera_z"
    ): 3,
    (
        "second_strictly_nearer_in_camera_z->"
        "overlapping_camera_z_intervals"
    ): 3,
    (
        "second_strictly_nearer_in_camera_z->"
        "second_strictly_nearer_in_camera_z"
    ): 3,
}
EXPECTED_NEARER_TRANSITIONS = {
    "banana->banana": 3,
    "bowl->bowl": 2,
    "none->none": 22,
    "none->rubiks_cube": 3,
    "rubiks_cube->none": 3,
    "rubiks_cube->rubiks_cube": 3,
}
EXPECTED_DELTA_SIGNS = {
    "positive_overlap_area_px2": {
        "decrease": 8,
        "increase": 12,
        "unchanged": 16,
    },
    "euclidean_rectangle_separation_px": {
        "decrease": 10,
        "increase": 7,
        "unchanged": 19,
    },
    "positive_camera_z_interval_gap_m": {
        "decrease": 9,
        "increase": 5,
        "unchanged": 22,
    },
    "positive_camera_z_interval_overlap_m": {
        "decrease": 12,
        "increase": 16,
        "unchanged": 8,
    },
}
EXPECTED_EXTREMA = {
    "positive_overlap_area_px2": {
        "minimum": -2197.27141122008,
        "maximum": 3818.2612353330132,
    },
    "euclidean_rectangle_separation_px": {
        "minimum": -76.05633019719147,
        "maximum": 83.2817513524775,
    },
    "positive_camera_z_interval_gap_m": {
        "minimum": -0.12991902265876598,
        "maximum": 0.14212397257408016,
    },
    "positive_camera_z_interval_overlap_m": {
        "minimum": -0.14586443344170918,
        "maximum": 0.14633422617754954,
    },
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def strict_json_loads(payload: bytes, source: str) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        value: dict[str, object] = {}
        for key, child in pairs:
            require(key not in value, f"Duplicate JSON key in {source}: {key}")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in {source}: {value}")

    decoded = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    require(isinstance(decoded, dict), f"JSON root is not an object: {source}")
    return decoded


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(REPOSITORY),
            "show",
            f"{commit}:{path}",
        ],
        timeout=30,
    )


def verify_payload(payload: bytes, descriptor: dict, label: str) -> None:
    require(len(payload) == descriptor["bytes"], f"{label} byte count changed")
    require(
        hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
        f"{label} SHA-256 changed",
    )


def import_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "wmf_pinned_retained_aabb_pair_geometry", path
    )
    require(spec is not None and spec.loader is not None, "Cannot load pair audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_pair_package() -> tuple[dict, dict]:
    committed_audit = git_bytes(PAIR_PACKAGE_COMMIT, PAIR_AUDIT["path"])
    verify_payload(committed_audit, PAIR_AUDIT, "Pinned pair audit")
    local_audit_path = REPOSITORY / PAIR_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, PAIR_AUDIT, "Local pair audit dependency")
    require(local_audit == committed_audit, "Local pair audit differs from pinned blob")
    pair_audit = import_module(local_audit_path)

    committed_report = git_bytes(PAIR_PACKAGE_COMMIT, PAIR_REPORT["path"])
    verify_payload(committed_report, PAIR_REPORT, "Pinned pair report")
    local_report = (REPOSITORY / PAIR_REPORT["path"]).read_bytes()
    verify_payload(local_report, PAIR_REPORT, "Local pair report dependency")
    require(
        local_report == committed_report,
        "Local pair report differs from pinned blob",
    )
    pair_report = strict_json_loads(committed_report, PAIR_REPORT["path"])
    regenerated = pair_audit.audit()
    require(
        pair_audit.canonical_bytes(regenerated) == committed_report,
        "Pinned pair report no longer regenerates from its pinned audit",
    )
    require(regenerated == pair_report, "Parsed and regenerated pair reports differ")
    require(
        pair_report["schema_version"]
        == "wmf-retained-aabb-pair-geometry-diagnostic-v1",
        "Pair report schema changed",
    )
    require(pair_report["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = pair_report["selection"]
    require(
        selection["terminal_matched_unique_cells"] == 7
        and selection["unique_arm_level_reset_mappings"] == 6
        and selection["mapping_camera_rows"] == 24
        and selection["mapping_camera_object_pairs"] == 72
        and selection["availability_selected"] is True
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False,
        "Pair report selection boundary changed",
    )
    require(
        all(value is False for value in pair_report["evidence_boundaries"].values()),
        "Pair report evidence boundary changed",
    )
    require(
        pair_report["new_behavioral_cells"] == 0
        and pair_report["new_model_requests"] == 0
        and pair_report["labels_created"] == 0
        and pair_report["forecast_accuracy_estimated"] is False
        and pair_report["resource_budget_qualified"] is False
        and pair_report["confirmation_released"] is False,
        "Pair report zero-science boundary changed",
    )
    return pair_report, {
        "commit": PAIR_PACKAGE_COMMIT,
        "audit": PAIR_AUDIT,
        "report": PAIR_REPORT,
        "canonical_regeneration_verified": True,
    }


def finite_number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape changed")
    output = [finite_number(child, label) for child in value]
    require(all(child >= 0.0 for child in output), f"{label} is negative")
    return output


def validate_pair_row(row: object, label: str) -> dict:
    require(isinstance(row, dict), f"{label} is not an object")
    require(set(row) == PAIR_ROW_KEYS, f"{label} keys changed")
    objects = row["objects"]
    require(
        isinstance(objects, list) and tuple(objects) in OBJECT_PAIRS,
        f"{label} object pair changed",
    )
    overlap = finite_vector(row["axis_overlap_uv_px"], 2, f"{label} overlap")
    separation = finite_vector(
        row["axis_separation_uv_px"], 2, f"{label} separation"
    )
    overlap_area = finite_number(
        row["positive_overlap_area_px2"], f"{label} overlap area"
    )
    rectangle_separation = finite_number(
        row["euclidean_rectangle_separation_px"],
        f"{label} rectangle separation",
    )
    depth_gap = finite_number(
        row["positive_camera_z_interval_gap_m"], f"{label} depth gap"
    )
    depth_overlap = finite_number(
        row["positive_camera_z_interval_overlap_m"], f"{label} depth overlap"
    )
    require(
        min(overlap_area, rectangle_separation, depth_gap, depth_overlap) >= 0.0,
        f"{label} has a negative scalar",
    )
    require(overlap_area == overlap[0] * overlap[1], f"{label} area mismatch")
    require(
        rectangle_separation == math.hypot(*separation),
        f"{label} Euclidean separation mismatch",
    )
    require(
        all(not (overlap[index] > 0.0 and separation[index] > 0.0) for index in range(2)),
        f"{label} overlaps and separates on the same axis",
    )

    projected_relation = row["projected_envelope_relation"]
    require(projected_relation in PROJECTED_RELATIONS, f"{label} projected relation")
    if projected_relation == "positive_area_overlap":
        require(
            overlap_area > 0.0 and rectangle_separation == 0.0,
            f"{label} malformed projected overlap",
        )
    elif projected_relation == "positive_separation":
        require(
            overlap_area == 0.0 and rectangle_separation > 0.0,
            f"{label} malformed projected separation",
        )
    else:
        require(
            overlap_area == 0.0 and rectangle_separation == 0.0,
            f"{label} malformed projected boundary touch",
        )

    depth_relation = row["camera_z_interval_relation"]
    require(depth_relation in DEPTH_RELATIONS, f"{label} depth relation")
    nearer = row["strictly_nearer_object_in_camera_z"]
    if depth_relation == "first_strictly_nearer_in_camera_z":
        require(
            nearer == objects[0] and depth_gap > 0.0 and depth_overlap == 0.0,
            f"{label} malformed first-nearer depth relation",
        )
    elif depth_relation == "second_strictly_nearer_in_camera_z":
        require(
            nearer == objects[1] and depth_gap > 0.0 and depth_overlap == 0.0,
            f"{label} malformed second-nearer depth relation",
        )
    elif depth_relation == "overlapping_camera_z_intervals":
        require(
            nearer is None and depth_gap == 0.0 and depth_overlap > 0.0,
            f"{label} malformed depth overlap",
        )
    else:
        require(
            nearer is None and depth_gap == 0.0 and depth_overlap == 0.0,
            f"{label} malformed depth boundary touch",
        )
    return row


def scalar_transition(original: dict, reflected: dict, field: str) -> dict:
    original_value = finite_number(original[field], f"original {field}")
    reflected_value = finite_number(reflected[field], f"reflected {field}")
    return {
        "original": original_value,
        "reflected": reflected_value,
        "reflected_minus_original": reflected_value - original_value,
    }


def transition_row(original: object, reflected: object, label: str) -> dict:
    original_row = validate_pair_row(original, f"{label} original")
    reflected_row = validate_pair_row(reflected, f"{label} reflected")
    require(
        original_row["objects"] == reflected_row["objects"],
        f"{label} object pair differs across arms",
    )
    original_projected = original_row["projected_envelope_relation"]
    reflected_projected = reflected_row["projected_envelope_relation"]
    original_depth = original_row["camera_z_interval_relation"]
    reflected_depth = reflected_row["camera_z_interval_relation"]
    original_nearer = original_row["strictly_nearer_object_in_camera_z"]
    reflected_nearer = reflected_row["strictly_nearer_object_in_camera_z"]
    return {
        "objects": original_row["objects"],
        "projected_envelope_relation_transition": (
            f"{original_projected}->{reflected_projected}"
        ),
        "projected_envelope_relation_changed": (
            original_projected != reflected_projected
        ),
        "camera_z_interval_relation_transition": (
            f"{original_depth}->{reflected_depth}"
        ),
        "camera_z_interval_relation_changed": original_depth != reflected_depth,
        "strictly_nearer_object_transition": (
            f"{original_nearer if original_nearer is not None else 'none'}->"
            f"{reflected_nearer if reflected_nearer is not None else 'none'}"
        ),
        "strictly_nearer_object_changed": original_nearer != reflected_nearer,
        "reflected_minus_original_positive_overlap_area_px2": scalar_transition(
            original_row, reflected_row, "positive_overlap_area_px2"
        )["reflected_minus_original"],
        "reflected_minus_original_euclidean_rectangle_separation_px": (
            scalar_transition(
                original_row, reflected_row, "euclidean_rectangle_separation_px"
            )["reflected_minus_original"]
        ),
        "reflected_minus_original_positive_camera_z_interval_gap_m": (
            scalar_transition(
                original_row, reflected_row, "positive_camera_z_interval_gap_m"
            )["reflected_minus_original"]
        ),
        "reflected_minus_original_positive_camera_z_interval_overlap_m": (
            scalar_transition(
                original_row,
                reflected_row,
                "positive_camera_z_interval_overlap_m",
            )["reflected_minus_original"]
        ),
    }


def index_camera_rows(mapping: dict, label: str) -> dict[str, dict]:
    rows = mapping.get("camera_rows")
    require(isinstance(rows, list) and len(rows) == 4, f"{label} camera rows changed")
    require(
        tuple(row.get("camera_name") for row in rows) == CAMERA_NAMES,
        f"{label} camera order changed",
    )
    require(
        len({row["camera_name"] for row in rows}) == 4,
        f"{label} duplicate camera row",
    )
    for row in rows:
        require(
            isinstance(row.get("camera_configuration_sha256"), str)
            and len(row["camera_configuration_sha256"]) == 64,
            f"{label} camera hash changed",
        )
        pair_rows = row.get("object_pair_rows")
        require(
            isinstance(pair_rows, list)
            and tuple(tuple(pair["objects"]) for pair in pair_rows) == OBJECT_PAIRS,
            f"{label} object-pair order changed",
        )
    return {row["camera_name"]: row for row in rows}


def index_pair_rows(camera: dict, label: str) -> dict[tuple[str, str], dict]:
    rows = camera["object_pair_rows"]
    require(len(rows) == 3, f"{label} pair-row count changed")
    indexed = {}
    for index, row in enumerate(rows):
        validated = validate_pair_row(row, f"{label} pair {index}")
        key = tuple(validated["objects"])
        require(key not in indexed, f"{label} duplicate object pair")
        indexed[key] = validated
    require(tuple(indexed) == OBJECT_PAIRS, f"{label} pair order changed")
    return indexed


def delta_sign(value: float) -> str:
    if value > 0.0:
        return "increase"
    if value < 0.0:
        return "decrease"
    return "unchanged"


def row_label(comparison: dict, camera_name: str, objects: list[str]) -> dict:
    return {
        "phase": comparison["phase"],
        "layout_pair_id": comparison["layout_pair_id"],
        "model_id": comparison["model_id"],
        "original_representative_cell_id": comparison[
            "original_mapping"
        ]["representative_cell_id"],
        "reflected_representative_cell_id": comparison[
            "reflected_mapping"
        ]["representative_cell_id"],
        "camera_name": camera_name,
        "object_pair": objects,
    }


def extremum_row(row: tuple[float, dict]) -> dict:
    return {"reflected_minus_original": row[0], **row[1]}


def mapping_metadata(mapping: dict) -> dict:
    keys = (
        "representative_cell_id",
        "member_cell_ids",
        "member_commands",
        "representative_snapshot_cell_key",
        "representative_snapshot_source",
    )
    require(all(key in mapping for key in keys), "Mapping metadata changed")
    return {key: mapping[key] for key in keys}


def analyze(pair_report: dict) -> tuple[list[dict], dict]:
    mappings = pair_report.get("mappings")
    require(isinstance(mappings, list) and len(mappings) == 6, "Mapping count changed")
    grouped: dict[tuple[str, str, str], dict[str, dict]] = {}
    for mapping in mappings:
        key = (mapping["phase"], mapping["layout_pair_id"], mapping["model_id"])
        require(key in COMPARISON_KEYS, f"Unexpected comparable mapping: {key}")
        arm = mapping["layout_arm"]
        require(arm in ("original", "reflected"), f"Unexpected layout arm: {arm}")
        require(key not in grouped or arm not in grouped[key], "Duplicate layout arm")
        grouped.setdefault(key, {})[arm] = mapping
    require(tuple(sorted(grouped)) == COMPARISON_KEYS, "Comparison keys changed")

    projected_transitions: Counter[str] = Counter()
    depth_transitions: Counter[str] = Counter()
    nearer_transitions: Counter[str] = Counter()
    scalar_signs: dict[str, Counter[str]] = {
        field: Counter() for field in SCALAR_FIELDS
    }
    scalar_rows: dict[str, list[tuple[float, dict]]] = {
        field: [] for field in SCALAR_FIELDS
    }
    relation_change_counts: Counter[str] = Counter()
    comparisons = []

    for key in COMPARISON_KEYS:
        arms = grouped[key]
        require(set(arms) == {"original", "reflected"}, f"Missing arm for {key}")
        original = arms["original"]
        reflected = arms["reflected"]
        expected_commands = EXPECTED_COMMANDS[key]
        original_commands = tuple(original["member_commands"])
        reflected_commands = tuple(reflected["member_commands"])
        require(
            original_commands == expected_commands["original"]
            and reflected_commands == expected_commands["reflected"],
            f"Command membership changed for {key}",
        )
        require(
            original["representative_snapshot_source"]
            == reflected["representative_snapshot_source"],
            f"Snapshot source differs across arms for {key}",
        )
        shared_commands = sorted(set(original_commands) & set(reflected_commands))
        classification = expected_commands["classification"]
        if "command_confounded" in classification:
            require(not shared_commands, f"Confounded commands unexpectedly overlap: {key}")
        else:
            require(shared_commands == ["left"], f"P00 shared-command boundary changed")

        original_cameras = index_camera_rows(original, f"{key} original")
        reflected_cameras = index_camera_rows(reflected, f"{key} reflected")
        comparison = {
            "phase": key[0],
            "layout_pair_id": key[1],
            "model_id": key[2],
            "original_mapping": mapping_metadata(original),
            "reflected_mapping": mapping_metadata(reflected),
            "command_comparison": {
                "classification": classification,
                "shared_commands": shared_commands,
                "fully_command_matched": original_commands == reflected_commands,
                "isolated_layout_effect_estimate": False,
            },
        }
        camera_rows = []
        for camera_name in CAMERA_NAMES:
            original_camera = original_cameras[camera_name]
            reflected_camera = reflected_cameras[camera_name]
            require(
                original_camera["camera_configuration_sha256"]
                == reflected_camera["camera_configuration_sha256"],
                f"Camera configuration differs across arms for {key}/{camera_name}",
            )
            original_pairs = index_pair_rows(
                original_camera, f"{key}/{camera_name} original"
            )
            reflected_pairs = index_pair_rows(
                reflected_camera, f"{key}/{camera_name} reflected"
            )
            transition_rows = []
            for objects in OBJECT_PAIRS:
                transition = transition_row(
                    original_pairs[objects],
                    reflected_pairs[objects],
                    f"{key}/{camera_name}/{objects}",
                )
                projected_transition = transition[
                    "projected_envelope_relation_transition"
                ]
                depth_transition = transition["camera_z_interval_relation_transition"]
                nearer_transition = transition["strictly_nearer_object_transition"]
                projected_changed = transition[
                    "projected_envelope_relation_changed"
                ]
                depth_changed = transition["camera_z_interval_relation_changed"]
                projected_transitions[projected_transition] += 1
                depth_transitions[depth_transition] += 1
                nearer_transitions[nearer_transition] += 1
                relation_change_counts[
                    "projected_changed" if projected_changed else "projected_unchanged"
                ] += 1
                relation_change_counts[
                    "depth_changed" if depth_changed else "depth_unchanged"
                ] += 1
                relation_change_counts[
                    "any_relation_changed"
                    if projected_changed or depth_changed
                    else "all_relations_unchanged"
                ] += 1
                relation_change_counts[
                    "both_relations_changed"
                    if projected_changed and depth_changed
                    else "not_both_relations_changed"
                ] += 1
                label = row_label(comparison, camera_name, transition["objects"])
                for field in SCALAR_FIELDS:
                    delta = transition[f"reflected_minus_original_{field}"]
                    scalar_signs[field][delta_sign(delta)] += 1
                    scalar_rows[field].append((delta, label))
                transition_rows.append(transition)
            camera_rows.append({
                "camera_name": camera_name,
                "camera_configuration_sha256": original_camera[
                    "camera_configuration_sha256"
                ],
                "object_pair_transition_rows": transition_rows,
            })
        comparisons.append(comparison | {"camera_transition_rows": camera_rows})

    require(
        dict(sorted(projected_transitions.items())) == EXPECTED_PROJECTED_TRANSITIONS,
        "Projected relation transitions changed",
    )
    require(
        dict(sorted(depth_transitions.items())) == EXPECTED_DEPTH_TRANSITIONS,
        "Camera-z interval transitions changed",
    )
    require(
        dict(sorted(nearer_transitions.items())) == EXPECTED_NEARER_TRANSITIONS,
        "Nearer-object transitions changed",
    )
    actual_signs = {
        field: dict(sorted(counter.items()))
        for field, counter in scalar_signs.items()
    }
    require(actual_signs == EXPECTED_DELTA_SIGNS, "Scalar delta signs changed")
    require(sum(projected_transitions.values()) == 36, "Transition row count changed")
    expected_relation_counts = {
        "projected_relation_changed": 1,
        "projected_relation_unchanged": 35,
        "camera_z_interval_relation_changed": 6,
        "camera_z_interval_relation_unchanged": 30,
        "any_relation_changed": 7,
        "all_relations_unchanged": 29,
        "both_relations_changed": 0,
    }
    actual_relation_counts = {
        "projected_relation_changed": relation_change_counts["projected_changed"],
        "projected_relation_unchanged": relation_change_counts["projected_unchanged"],
        "camera_z_interval_relation_changed": relation_change_counts["depth_changed"],
        "camera_z_interval_relation_unchanged": relation_change_counts["depth_unchanged"],
        "any_relation_changed": relation_change_counts["any_relation_changed"],
        "all_relations_unchanged": relation_change_counts[
            "all_relations_unchanged"
        ],
        "both_relations_changed": relation_change_counts["both_relations_changed"],
    }
    require(
        actual_relation_counts == expected_relation_counts,
        "Relation change counts changed",
    )

    extrema = {}
    for field in SCALAR_FIELDS:
        rows = scalar_rows[field]
        require(len(rows) == 36, f"Scalar row count changed for {field}")
        minimum = min(rows, key=lambda row: row[0])
        maximum = max(rows, key=lambda row: row[0])
        require(
            minimum[0] == EXPECTED_EXTREMA[field]["minimum"]
            and maximum[0] == EXPECTED_EXTREMA[field]["maximum"],
            f"Scalar extrema changed for {field}",
        )
        extrema[field] = {
            "minimum": extremum_row(minimum),
            "maximum": extremum_row(maximum),
        }

    return comparisons, {
        "comparable_original_reflected_mapping_pairs": 3,
        "mapping_pair_camera_rows": 12,
        "object_pair_transition_rows": 36,
        "comparison_pairs_with_disjoint_command_sets": 2,
        "comparison_pairs_with_shared_commands": 1,
        "comparison_pairs_with_identical_command_sets": 0,
        "isolated_layout_effect_estimates": 0,
        **actual_relation_counts,
        "projected_relation_transitions": dict(sorted(projected_transitions.items())),
        "camera_z_interval_relation_transitions": dict(
            sorted(depth_transitions.items())
        ),
        "strictly_nearer_object_transitions": dict(
            sorted(nearer_transitions.items())
        ),
        "scalar_delta_sign_counts": actual_signs,
        "scalar_delta_extrema": extrema,
    }


def expect_value_error(function, message: str) -> None:
    try:
        function()
    except ValueError:
        return
    raise ValueError(message)


def synthetic_row(
    projected_relation: str,
    depth_relation: str,
) -> dict:
    if projected_relation == "positive_area_overlap":
        overlap, area, separation, distance = [2.0, 3.0], 6.0, [0.0, 0.0], 0.0
    elif projected_relation == "positive_separation":
        overlap, area, separation, distance = [0.0, 1.0], 0.0, [4.0, 0.0], 4.0
    else:
        overlap, area, separation, distance = [0.0, 2.0], 0.0, [0.0, 0.0], 0.0
    if depth_relation == "first_strictly_nearer_in_camera_z":
        nearer, gap, depth_overlap = "banana", 0.2, 0.0
    elif depth_relation == "second_strictly_nearer_in_camera_z":
        nearer, gap, depth_overlap = "bowl", 0.2, 0.0
    elif depth_relation == "overlapping_camera_z_intervals":
        nearer, gap, depth_overlap = None, 0.0, 0.4
    else:
        nearer, gap, depth_overlap = None, 0.0, 0.0
    return {
        "objects": ["banana", "bowl"],
        "projected_envelope_relation": projected_relation,
        "axis_overlap_uv_px": overlap,
        "positive_overlap_area_px2": area,
        "axis_separation_uv_px": separation,
        "euclidean_rectangle_separation_px": distance,
        "camera_z_interval_relation": depth_relation,
        "strictly_nearer_object_in_camera_z": nearer,
        "positive_camera_z_interval_gap_m": gap,
        "positive_camera_z_interval_overlap_m": depth_overlap,
    }


def self_test_predicates() -> None:
    overlap = synthetic_row(
        "positive_area_overlap", "overlapping_camera_z_intervals"
    )
    separation = synthetic_row(
        "positive_separation", "second_strictly_nearer_in_camera_z"
    )
    touch = synthetic_row(
        "boundary_touch_only", "camera_z_interval_boundary_touch_only"
    )
    transition = transition_row(overlap, separation, "synthetic")
    require(
        transition["projected_envelope_relation_transition"]
        == "positive_area_overlap->positive_separation"
        and transition["camera_z_interval_relation_transition"]
        == (
            "overlapping_camera_z_intervals->"
            "second_strictly_nearer_in_camera_z"
        )
        and transition["reflected_minus_original_positive_overlap_area_px2"]
        == -6.0
        and transition[
            "reflected_minus_original_euclidean_rectangle_separation_px"
        ]
        == 4.0,
        "Synthetic transition predicate failed",
    )
    validate_pair_row(touch, "synthetic touch")
    expect_value_error(
        lambda: strict_json_loads(b'{"x":1,"x":2}', "duplicate fixture"),
        "Duplicate-key fixture was accepted",
    )
    expect_value_error(
        lambda: strict_json_loads(b'{"x":NaN}', "nonfinite fixture"),
        "Non-finite fixture was accepted",
    )
    invalid_area = copy.deepcopy(overlap)
    invalid_area["positive_overlap_area_px2"] = 5.0
    expect_value_error(
        lambda: validate_pair_row(invalid_area, "invalid area fixture"),
        "Invalid area fixture was accepted",
    )
    mismatched_objects = copy.deepcopy(separation)
    mismatched_objects["objects"] = ["bowl", "rubiks_cube"]
    expect_value_error(
        lambda: transition_row(overlap, mismatched_objects, "object mismatch fixture"),
        "Mismatched-object fixture was accepted",
    )
    descriptor = {
        "bytes": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
    }
    verify_payload(b"abc", descriptor, "valid dependency fixture")
    expect_value_error(
        lambda: verify_payload(b"abd", descriptor, "altered dependency fixture"),
        "Altered dependency fixture was accepted",
    )


def audit() -> dict:
    self_test_predicates()
    pair_report, pair_pin = load_pair_package()
    comparisons, summary = analyze(pair_report)
    return {
        "schema_version": "wmf-retained-aabb-reflection-transition-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "turn2845_pair_geometry_package": pair_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "mapping_pair_camera_rows": 12,
            "object_pair_transition_rows": 36,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
            "D01_D02_command_confounded": True,
            "P00_shared_left_command_with_extra_reflected_right_member": True,
            "isolated_layout_effect_estimated": False,
        },
        "predicate_semantics": {
            "comparison": (
                "For each availability-selected phase/layout-pair/model key, compare the "
                "pinned original and reflected reset-mapping rows with identical camera "
                "configuration hashes and object-pair order. This is a paired retained-"
                "geometry description, not a temporal transition or causal intervention."
            ),
            "projected_envelopes": (
                "The inherited rectangles are independent u/v minima and maxima of "
                "projected world.get_bbox-derived axis-aligned-box corners. Relation "
                "transitions and reflected-minus-original overlap-area and Euclidean "
                "rectangle-separation deltas use exact stored floats without epsilon or "
                "scientific threshold. No IoU is computed."
            ),
            "camera_depth_intervals": (
                "The inherited camera-z intervals use the same AABB corners. Relation "
                "transitions and reflected-minus-original positive-gap and interval-"
                "overlap deltas use exact stored floats. Camera z is not Euclidean range "
                "and depth order is not visibility or silhouette order."
            ),
            "command_boundary": (
                "D01 and D02 compare original-left against reflected-right mappings, so "
                "layout arm is command-confounded. P00 contains a shared left command, "
                "but its reflected mapping also contains a right-command member. No "
                "comparison isolates a layout, command, model or task effect."
            ),
        },
        "summary": summary,
        "comparisons": comparisons,
        "finding": (
            "Across 36 availability-selected original-to-reflected retained-geometry "
            "rows, projected AABB-envelope relation is unchanged for 35 and changes "
            "from positive-area overlap to positive separation for one. Camera-z "
            "interval relation is unchanged for 30 and changes between interval overlap "
            "and second-object-strictly-nearer for six. D01 and D02 are command-"
            "confounded; these descriptive transitions do not estimate an effect."
        ),
        "claim_boundaries": {
            "aabb_envelopes_only": (
                "The inherited projected rectangles bound retained axis-aligned boxes. "
                "They are not OBBs, exact meshes, silhouettes, segmentation masks, image "
                "footprints or object pixels."
            ),
            "no_visibility_or_occlusion_inference": (
                "Envelope relation changes do not prove silhouette changes, visibility "
                "or occlusion. Camera-z relation changes do not identify a visible front "
                "object. RGB content, robot/static-scene surfaces and general occlusion "
                "are not analyzed."
            ),
            "no_measurement_or_endpoint_inference": (
                "No RGB artifact, crop/time mapping, labelability, human annotation, "
                "measurement feasibility, r0/rhat/r endpoint or generated/executed "
                "correspondence is authenticated. No A1-A9 result follows."
            ),
            "selection_confounding_and_effects": (
                "The three mapping pairs are availability-selected, not representative. "
                "D01/D02 are command-confounded and P00 has unequal command membership. "
                "The arrow denotes comparison direction, not time, causation or motion. "
                "No layout, command, model, task-success, population or camera-ranking "
                "effect is estimated."
            ),
            "science_and_release": (
                "No scientific acceptance threshold, fixture/camera qualification, "
                "current-state inference, gate promotion, dispatch/release authority or "
                "new behavioral evidence follows."
            ),
        },
        "evidence_boundaries": {
            "raw_camera_bytes_read": False,
            "raw_pvc_bytes_read": False,
            "rgb_or_image_content_analyzed": False,
            "silhouette_or_segmentation_analyzed": False,
            "obb_or_exact_mesh_analyzed": False,
            "image_footprint_or_object_pixels_inferred": False,
            "visibility_inferred": False,
            "general_occlusion_analyzed": False,
            "robot_or_static_scene_occlusion_analyzed": False,
            "camera_crop_or_time_alignment_authenticated": False,
            "labelability_estimated": False,
            "human_image_centroids_used": False,
            "measurement_feasibility_qualified": False,
            "iou_computed": False,
            "temporal_transition_inferred": False,
            "causal_effect_estimated": False,
            "isolated_layout_or_command_effect_estimated": False,
            "camera_ranking_estimated": False,
            "scientific_threshold_applied": False,
            "fixture_pass_fail_classification_created": False,
            "a1_a9_result": False,
            "forecast_or_persistence_metric": False,
            "task_success_or_model_rate": False,
            "current_state_inferred": False,
            "gate_advanced": False,
            "release_authority": False,
        },
        "new_behavioral_cells": 0,
        "new_model_requests": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "resource_budget_qualified": False,
        "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        type=Path,
        help="Verify an existing canonical diagnostic without modifying it",
    )
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(
            args.check.read_bytes() == encoded,
            "Saved retained AABB reflection-transition diagnostic differs",
        )
        print(
            "Verified 3 availability-selected original/reflected mapping pairs, "
            "12 camera comparisons and 36 object-pair transition rows; no causal, "
            "visibility, endpoint or gate evidence."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
