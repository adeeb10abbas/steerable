#!/usr/bin/env python3
"""Audit retained AABB correspondence to a center-reflection reference only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
TURN2846_PACKAGE_COMMIT = "e587e29cb5b5292db98d8b1eb0bcdba076cf17c1"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2846_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_aabb_reflection_transitions.py"
    ),
    "bytes": 39416,
    "sha256": "3ca348838bda8e8ea6e96f6f75aa64934da6a03a0209d304427716bdcbe29136",
}
TURN2846_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_aabb_reflection_transition_diagnostic.json"
    ),
    "bytes": 52327,
    "sha256": "5592a16da4d607ec591f2fb85c7ca7d4cfe5c5ed0f816f108447fbc53437345a",
}
CAMERA_NAMES = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
OBJECT_NAMES = ("banana", "bowl", "rubiks_cube")
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
EXPECTED_MAXIMA = {
    "maximum_world_aabb_bound_residual": {
        "mm": 1.928526908159256,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "object_name": "rubiks_cube",
    },
    "maximum_center_target_l2_residual": {
        "mm": 1.8614546594330106,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "object_name": "rubiks_cube",
    },
    "maximum_aabb_midpoint_l2_residual": {
        "mm": 1.8393798739074432,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "object_name": "rubiks_cube",
    },
    "maximum_world_aabb_extent_residual": {
        "mm": 0.36071985960006714,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "object_name": "rubiks_cube",
    },
    "maximum_projected_envelope_edge_residual": {
        "px": 3.204905157569897,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "camera_name": "wrist_cam",
        "object_name": "rubiks_cube",
    },
    "maximum_projected_envelope_center_l2_residual": {
        "px": 3.384343233964444,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "camera_name": "wrist_cam",
        "object_name": "rubiks_cube",
    },
    "maximum_projected_envelope_extent_residual": {
        "px": 0.7106885741465021,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "camera_name": "wrist_cam",
        "object_name": "rubiks_cube",
    },
    "maximum_camera_z_interval_edge_residual": {
        "mm": 1.414771802770609,
        "phase": "development",
        "layout_pair_id": "D02",
        "model_id": "D1",
        "camera_name": "over_shoulder_left_camera",
        "object_name": "rubiks_cube",
    },
}
EXPECTED_CAMERA_MAXIMA = {
    "head_camera": {
        "maximum_absolute_envelope_edge_residual_px": 0.6261452066672746,
        "maximum_envelope_center_l2_residual_px": 0.5879089717036505,
        "maximum_absolute_envelope_extent_residual_px": 0.10124080442972172,
        "maximum_absolute_camera_z_interval_edge_residual_mm": 0.6262817661075726,
    },
    "over_shoulder_left_camera": {
        "maximum_absolute_envelope_edge_residual_px": 0.8071086828970806,
        "maximum_envelope_center_l2_residual_px": 0.8694251000724288,
        "maximum_absolute_envelope_extent_residual_px": 0.26518403549846425,
        "maximum_absolute_camera_z_interval_edge_residual_mm": 1.414771802770609,
    },
    "over_shoulder_right_camera": {
        "maximum_absolute_envelope_edge_residual_px": 1.0586522924639894,
        "maximum_envelope_center_l2_residual_px": 0.9471558629353677,
        "maximum_absolute_envelope_extent_residual_px": 0.30089526976917114,
        "maximum_absolute_camera_z_interval_edge_residual_mm": 0.6964037463585981,
    },
    "wrist_cam": {
        "maximum_absolute_envelope_edge_residual_px": 3.204905157569897,
        "maximum_envelope_center_l2_residual_px": 3.384343233964444,
        "maximum_absolute_envelope_extent_residual_px": 0.7106885741465021,
        "maximum_absolute_camera_z_interval_edge_residual_mm": 0.24992574217774788,
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


def import_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"Cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_turn2846_package() -> tuple[ModuleType, dict, dict]:
    committed_audit = git_bytes(TURN2846_PACKAGE_COMMIT, TURN2846_AUDIT["path"])
    verify_payload(committed_audit, TURN2846_AUDIT, "Pinned Turn2846 audit")
    local_audit_path = REPOSITORY / TURN2846_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2846_AUDIT, "Local Turn2846 audit")
    require(local_audit == committed_audit, "Local Turn2846 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2846_reflection_transition")

    committed_report = git_bytes(TURN2846_PACKAGE_COMMIT, TURN2846_REPORT["path"])
    verify_payload(committed_report, TURN2846_REPORT, "Pinned Turn2846 report")
    local_report = (REPOSITORY / TURN2846_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2846_REPORT, "Local Turn2846 report")
    require(local_report == committed_report, "Local Turn2846 report differs")
    parsed = strict_json_loads(committed_report, TURN2846_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2846 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2846 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-aabb-reflection-transition-diagnostic-v1",
        "Turn2846 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["mapping_pair_camera_rows"] == 12
        and selection["object_pair_transition_rows"] == 36
        and selection["availability_selected"] is True
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"]
        is True
        and selection["isolated_layout_effect_estimated"] is False,
        "Turn2846 selection boundary changed",
    )
    require(
        all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2846 evidence boundary changed",
    )
    require(
        parsed["new_behavioral_cells"] == 0
        and parsed["new_model_requests"] == 0
        and parsed["labels_created"] == 0
        and parsed["forecast_accuracy_estimated"] is False
        and parsed["resource_budget_qualified"] is False
        and parsed["confirmation_released"] is False,
        "Turn2846 zero-science boundary changed",
    )
    return prior, parsed, {
        "commit": TURN2846_PACKAGE_COMMIT,
        "audit": TURN2846_AUDIT,
        "report": TURN2846_REPORT,
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


def validate_bounds(value: object, label: str) -> dict[str, list[float]]:
    require(isinstance(value, dict) and set(value) == {"lower", "upper"}, label)
    lower = finite_vector(value["lower"], 3, f"{label} lower")
    upper = finite_vector(value["upper"], 3, f"{label} upper")
    require(
        all(lower[index] < upper[index] for index in range(3)),
        f"{label} has non-positive extent",
    )
    return {"lower": lower, "upper": upper}


def mapping_key(mapping: dict) -> tuple[str, str, str, str]:
    return (
        mapping["phase"],
        mapping["layout_pair_id"],
        mapping["model_id"],
        mapping["layout_arm"],
    )


def extract_mapping_geometry(mapping: dict) -> tuple[dict, dict[str, dict]]:
    cameras = mapping.get("cameras")
    require(isinstance(cameras, dict), "Mapping cameras missing")
    require(tuple(cameras) == CAMERA_NAMES, "Retained camera order changed")
    reference_geometry = None
    configurations: dict[str, dict] = {}
    for camera_name in CAMERA_NAMES:
        camera = cameras[camera_name]
        require(isinstance(camera, dict), f"Malformed {camera_name}")
        configuration = camera.get("camera_configuration")
        geometry = camera.get("geometry")
        require(isinstance(configuration, dict), f"{camera_name} configuration")
        require(isinstance(geometry, dict), f"{camera_name} geometry")
        require(
            all(configuration[key] == geometry.get(key) for key in configuration),
            f"{camera_name} nested configuration differs",
        )
        configuration_sha256 = camera.get("camera_configuration_sha256")
        require(
            isinstance(configuration_sha256, str)
            and len(configuration_sha256) == 64
            and all(character in "0123456789abcdef" for character in configuration_sha256),
            f"{camera_name} configuration hash malformed",
        )
        configurations[camera_name] = configuration
        retained = {
            "object_centers_robot_base_m": geometry.get(
                "object_centers_robot_base_m"
            ),
            "object_aabbs_robot_base_m": geometry.get(
                "object_aabbs_robot_base_m"
            ),
        }
        if reference_geometry is None:
            reference_geometry = retained
        else:
            require(retained == reference_geometry, "Geometry copies differ by camera")
    require(isinstance(reference_geometry, dict), "No retained geometry")
    centers = reference_geometry["object_centers_robot_base_m"]
    bounds = reference_geometry["object_aabbs_robot_base_m"]
    require(
        isinstance(centers, dict)
        and isinstance(bounds, dict)
        and tuple(centers) == OBJECT_NAMES
        and tuple(bounds) == OBJECT_NAMES,
        "Retained object order changed",
    )
    normalized = {
        "object_centers_robot_base_m": {
            name: finite_vector(centers[name], 3, f"{name} center")
            for name in OBJECT_NAMES
        },
        "object_aabbs_robot_base_m": {
            name: validate_bounds(bounds[name], f"{name} AABB")
            for name in OBJECT_NAMES
        },
    }
    return normalized, configurations


def index_retained_mappings(
    prior: ModuleType,
) -> tuple[ModuleType, ModuleType, ModuleType, dict[tuple[str, str, str, str], dict]]:
    pair_path = REPOSITORY / prior.PAIR_AUDIT["path"]
    pair = import_module(pair_path, "wmf_pinned_pair_for_reflection_correspondence")
    frame, projection, base, geometry_report, _ = pair.load_prior_package()
    mappings = frame.collapse_unique_mappings(projection, base, geometry_report)
    require(len(mappings) == 6, "Unique retained mapping count changed")
    indexed: dict[tuple[str, str, str, str], dict] = {}
    for mapping in mappings:
        key = mapping_key(mapping)
        require(key not in indexed, f"Duplicate mapping: {key}")
        indexed[key] = mapping
    expected = {
        (*comparison_key, arm)
        for comparison_key in COMPARISON_KEYS
        for arm in ("original", "reflected")
    }
    require(set(indexed) == expected, "Retained mapping identities changed")
    return pair, projection, base, indexed


def ideal_center_reflection(
    original_center: list[float], original_bounds: dict[str, list[float]]
) -> tuple[list[float], list[float], dict[str, list[float]]]:
    target_center = [original_center[0], -original_center[1], original_center[2]]
    translation = [
        target_center[index] - original_center[index] for index in range(3)
    ]
    require(
        translation[0] == 0.0
        and translation[1] == -2.0 * original_center[1]
        and translation[2] == 0.0,
        "Center-reflection translation changed",
    )
    ideal = {
        edge: [
            original_bounds[edge][index] + translation[index]
            for index in range(3)
        ]
        for edge in ("lower", "upper")
    }
    return target_center, translation, validate_bounds(ideal, "ideal reflected AABB")


def signed_vector(first: list[float], second: list[float]) -> list[float]:
    require(len(first) == len(second), "Signed-vector shape changed")
    return [first[index] - second[index] for index in range(len(first))]


def vector_extent(bounds: dict[str, list[float]]) -> list[float]:
    return [bounds["upper"][index] - bounds["lower"][index] for index in range(3)]


def vector_midpoint(bounds: dict[str, list[float]]) -> list[float]:
    return [
        (bounds["lower"][index] + bounds["upper"][index]) / 2.0
        for index in range(3)
    ]


def envelope_center(envelope: list[float]) -> list[float]:
    require(len(envelope) == 4, "Envelope shape changed")
    return [(envelope[0] + envelope[1]) / 2.0, (envelope[2] + envelope[3]) / 2.0]


def envelope_extent(envelope: list[float]) -> list[float]:
    require(len(envelope) == 4, "Envelope shape changed")
    return [envelope[1] - envelope[0], envelope[3] - envelope[2]]


def l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def world_row(
    object_name: str,
    original_center: list[float],
    original_bounds: dict[str, list[float]],
    actual_reflected_center: list[float],
    actual_reflected_bounds: dict[str, list[float]],
) -> tuple[dict, dict[str, list[float]]]:
    target_center, translation, ideal_bounds = ideal_center_reflection(
        original_center, original_bounds
    )
    lower_residual = signed_vector(actual_reflected_bounds["lower"], ideal_bounds["lower"])
    upper_residual = signed_vector(actual_reflected_bounds["upper"], ideal_bounds["upper"])
    center_residual = signed_vector(actual_reflected_center, target_center)
    midpoint_residual = signed_vector(
        vector_midpoint(actual_reflected_bounds), vector_midpoint(ideal_bounds)
    )
    extent_residual = signed_vector(
        vector_extent(actual_reflected_bounds), vector_extent(ideal_bounds)
    )
    bound_residuals = lower_residual + upper_residual
    return {
        "object_name": object_name,
        "original_center_robot_base_m": original_center,
        "constructed_center_reflection_target_robot_base_m": target_center,
        "constructed_translation_robot_base_m": translation,
        "actual_reflected_center_robot_base_m": actual_reflected_center,
        "actual_minus_constructed_center_target_m": center_residual,
        "center_target_l2_residual_mm": 1000.0 * l2(center_residual),
        "original_aabb_robot_base_m": original_bounds,
        "constructed_center_reflected_aabb_robot_base_m": ideal_bounds,
        "actual_reflected_aabb_robot_base_m": actual_reflected_bounds,
        "actual_minus_constructed_lower_bounds_m": lower_residual,
        "actual_minus_constructed_upper_bounds_m": upper_residual,
        "maximum_absolute_bound_residual_mm": 1000.0
        * max(abs(value) for value in bound_residuals),
        "actual_minus_constructed_aabb_midpoint_m": midpoint_residual,
        "aabb_midpoint_l2_residual_mm": 1000.0 * l2(midpoint_residual),
        "actual_minus_constructed_aabb_extent_m": extent_residual,
        "maximum_absolute_extent_residual_mm": 1000.0
        * max(abs(value) for value in extent_residual),
    }, ideal_bounds


def projected_row(
    pair: ModuleType,
    projection: ModuleType,
    object_name: str,
    configuration: dict,
    ideal_bounds: dict[str, list[float]],
    actual_bounds: dict[str, list[float]],
) -> dict:
    ideal = pair.object_geometry(
        projection,
        {"object_aabbs_robot_base_m": {object_name: ideal_bounds}},
        configuration,
        object_name,
    )
    actual = pair.object_geometry(
        projection,
        {"object_aabbs_robot_base_m": {object_name: actual_bounds}},
        configuration,
        object_name,
    )
    ideal_envelope = finite_vector(ideal["envelope_uv_px"], 4, "ideal envelope")
    actual_envelope = finite_vector(actual["envelope_uv_px"], 4, "actual envelope")
    envelope_residual = signed_vector(actual_envelope, ideal_envelope)
    center_residual = signed_vector(
        envelope_center(actual_envelope), envelope_center(ideal_envelope)
    )
    extent_residual = signed_vector(
        envelope_extent(actual_envelope), envelope_extent(ideal_envelope)
    )
    ideal_depth = finite_vector(ideal["camera_z_interval_m"], 2, "ideal depth")
    actual_depth = finite_vector(actual["camera_z_interval_m"], 2, "actual depth")
    depth_residual = signed_vector(actual_depth, ideal_depth)
    return {
        "object_name": object_name,
        "constructed_center_reflected_envelope_uv_px": ideal_envelope,
        "actual_reflected_envelope_uv_px": actual_envelope,
        "actual_minus_constructed_envelope_edges_px": envelope_residual,
        "maximum_absolute_envelope_edge_residual_px": max(
            abs(value) for value in envelope_residual
        ),
        "actual_minus_constructed_envelope_center_uv_px": center_residual,
        "envelope_center_l2_residual_px": l2(center_residual),
        "actual_minus_constructed_envelope_extent_wh_px": extent_residual,
        "maximum_absolute_envelope_extent_residual_px": max(
            abs(value) for value in extent_residual
        ),
        "constructed_center_reflected_camera_z_interval_m": ideal_depth,
        "actual_reflected_camera_z_interval_m": actual_depth,
        "actual_minus_constructed_camera_z_interval_edges_m": depth_residual,
        "maximum_absolute_camera_z_interval_edge_residual_mm": 1000.0
        * max(abs(value) for value in depth_residual),
    }


def metric_record(value: float, label: dict, unit: str) -> dict:
    return {unit: value, **label}


def analyze(
    prior: ModuleType,
) -> tuple[list[dict], dict]:
    pair, projection, base, mappings = index_retained_mappings(prior)
    comparisons = []
    world_bound_metrics: list[tuple[float, dict]] = []
    world_center_metrics: list[tuple[float, dict]] = []
    world_midpoint_metrics: list[tuple[float, dict]] = []
    world_extent_metrics: list[tuple[float, dict]] = []
    envelope_edge_metrics: list[tuple[float, dict]] = []
    envelope_center_metrics: list[tuple[float, dict]] = []
    envelope_extent_metrics: list[tuple[float, dict]] = []
    depth_edge_metrics: list[tuple[float, dict]] = []
    by_camera: dict[str, dict[str, list[float]]] = {
        name: {
            "envelope_edge": [],
            "envelope_center": [],
            "envelope_extent": [],
            "depth_edge_mm": [],
        }
        for name in CAMERA_NAMES
    }
    exact_world_bound_rows = 0
    exact_projected_envelope_rows = 0

    for comparison_key in COMPARISON_KEYS:
        original = mappings[(*comparison_key, "original")]
        reflected = mappings[(*comparison_key, "reflected")]
        expected = EXPECTED_COMMANDS[comparison_key]
        require(
            tuple(original["member_commands"]) == expected["original"]
            and tuple(reflected["member_commands"]) == expected["reflected"],
            f"Command membership changed: {comparison_key}",
        )
        original_geometry, original_configs = extract_mapping_geometry(original)
        reflected_geometry, reflected_configs = extract_mapping_geometry(reflected)
        require(
            original_configs == reflected_configs,
            f"Camera configuration changed across arms: {comparison_key}",
        )

        world_rows = []
        ideal_bounds_by_object = {}
        for object_name in OBJECT_NAMES:
            row, ideal_bounds = world_row(
                object_name,
                original_geometry["object_centers_robot_base_m"][object_name],
                original_geometry["object_aabbs_robot_base_m"][object_name],
                reflected_geometry["object_centers_robot_base_m"][object_name],
                reflected_geometry["object_aabbs_robot_base_m"][object_name],
            )
            ideal_bounds_by_object[object_name] = ideal_bounds
            label = {
                "phase": comparison_key[0],
                "layout_pair_id": comparison_key[1],
                "model_id": comparison_key[2],
                "object_name": object_name,
            }
            world_bound_metrics.append((row["maximum_absolute_bound_residual_mm"], label))
            world_center_metrics.append((row["center_target_l2_residual_mm"], label))
            world_midpoint_metrics.append((row["aabb_midpoint_l2_residual_mm"], label))
            world_extent_metrics.append((row["maximum_absolute_extent_residual_mm"], label))
            if row["maximum_absolute_bound_residual_mm"] == 0.0:
                exact_world_bound_rows += 1
            world_rows.append(row)

        camera_rows = []
        for camera_name in CAMERA_NAMES:
            camera_hash = original["cameras"][camera_name][
                "camera_configuration_sha256"
            ]
            require(
                camera_hash
                == reflected["cameras"][camera_name]["camera_configuration_sha256"],
                f"Camera hash changed across arms: {comparison_key} {camera_name}",
            )
            object_rows = []
            for object_name in OBJECT_NAMES:
                row = projected_row(
                    pair,
                    projection,
                    object_name,
                    original_configs[camera_name],
                    ideal_bounds_by_object[object_name],
                    reflected_geometry["object_aabbs_robot_base_m"][object_name],
                )
                label = {
                    "phase": comparison_key[0],
                    "layout_pair_id": comparison_key[1],
                    "model_id": comparison_key[2],
                    "camera_name": camera_name,
                    "object_name": object_name,
                }
                envelope_edge_metrics.append(
                    (row["maximum_absolute_envelope_edge_residual_px"], label)
                )
                envelope_center_metrics.append(
                    (row["envelope_center_l2_residual_px"], label)
                )
                envelope_extent_metrics.append(
                    (row["maximum_absolute_envelope_extent_residual_px"], label)
                )
                depth_edge_metrics.append(
                    (row["maximum_absolute_camera_z_interval_edge_residual_mm"], label)
                )
                aggregate = by_camera[camera_name]
                aggregate["envelope_edge"].append(
                    row["maximum_absolute_envelope_edge_residual_px"]
                )
                aggregate["envelope_center"].append(
                    row["envelope_center_l2_residual_px"]
                )
                aggregate["envelope_extent"].append(
                    row["maximum_absolute_envelope_extent_residual_px"]
                )
                aggregate["depth_edge_mm"].append(
                    row["maximum_absolute_camera_z_interval_edge_residual_mm"]
                )
                if row["maximum_absolute_envelope_edge_residual_px"] == 0.0:
                    exact_projected_envelope_rows += 1
                object_rows.append(row)
            camera_rows.append(
                {
                    "camera_name": camera_name,
                    "camera_configuration_sha256": camera_hash,
                    "object_rows": object_rows,
                }
            )

        comparisons.append(
            {
                "phase": comparison_key[0],
                "layout_pair_id": comparison_key[1],
                "model_id": comparison_key[2],
                "original_representative_cell_id": original[
                    "representative_cell_id"
                ],
                "reflected_representative_cell_id": reflected[
                    "representative_cell_id"
                ],
                "original_member_cell_ids": original["member_cell_ids"],
                "reflected_member_cell_ids": reflected["member_cell_ids"],
                "original_member_commands": original["member_commands"],
                "reflected_member_commands": reflected["member_commands"],
                "command_membership_classification": expected["classification"],
                "camera_configurations_exactly_identical_across_arms": True,
                "world_object_rows": world_rows,
                "camera_rows": camera_rows,
            }
        )

    require(len(comparisons) == 3, "Comparison count changed")
    require(len(world_bound_metrics) == 9, "World-object row count changed")
    require(len(envelope_edge_metrics) == 36, "Projected-object row count changed")
    camera_summary = {}
    for camera_name in CAMERA_NAMES:
        aggregate = by_camera[camera_name]
        require(
            all(len(values) == 9 for values in aggregate.values()),
            f"Camera aggregation changed: {camera_name}",
        )
        camera_summary[camera_name] = {
            "projected_object_rows": 9,
            "maximum_absolute_envelope_edge_residual_px": max(
                aggregate["envelope_edge"]
            ),
            "maximum_envelope_center_l2_residual_px": max(
                aggregate["envelope_center"]
            ),
            "maximum_absolute_envelope_extent_residual_px": max(
                aggregate["envelope_extent"]
            ),
            "maximum_absolute_camera_z_interval_edge_residual_mm": max(
                aggregate["depth_edge_mm"]
            ),
            "descriptive_only_not_camera_ranking": True,
        }

    summary = {
        "original_reflected_mapping_pairs": 3,
        "camera_configuration_pairs_verified_exactly_identical": 12,
        "world_object_correspondence_rows": 9,
        "projected_camera_object_correspondence_rows": 36,
        "exact_zero_world_aabb_bound_residual_rows": exact_world_bound_rows,
        "exact_zero_projected_envelope_edge_residual_rows": (
            exact_projected_envelope_rows
        ),
        "maximum_world_aabb_bound_residual": metric_record(
            *max(world_bound_metrics, key=lambda row: row[0]),
            "mm",
        ),
        "maximum_center_target_l2_residual": metric_record(
            *max(world_center_metrics, key=lambda row: row[0]),
            "mm",
        ),
        "maximum_aabb_midpoint_l2_residual": metric_record(
            *max(world_midpoint_metrics, key=lambda row: row[0]),
            "mm",
        ),
        "maximum_world_aabb_extent_residual": metric_record(
            *max(world_extent_metrics, key=lambda row: row[0]),
            "mm",
        ),
        "maximum_projected_envelope_edge_residual": metric_record(
            *max(envelope_edge_metrics, key=lambda row: row[0]),
            "px",
        ),
        "maximum_projected_envelope_center_l2_residual": metric_record(
            *max(envelope_center_metrics, key=lambda row: row[0]),
            "px",
        ),
        "maximum_projected_envelope_extent_residual": metric_record(
            *max(envelope_extent_metrics, key=lambda row: row[0]),
            "px",
        ),
        "maximum_camera_z_interval_edge_residual": metric_record(
            *max(depth_edge_metrics, key=lambda row: row[0]),
            "mm",
        ),
        "camera_summary": camera_summary,
    }
    require(
        summary["original_reflected_mapping_pairs"] == 3
        and summary["camera_configuration_pairs_verified_exactly_identical"] == 12
        and summary["world_object_correspondence_rows"] == 9
        and summary["projected_camera_object_correspondence_rows"] == 36
        and summary["exact_zero_world_aabb_bound_residual_rows"] == 0
        and summary["exact_zero_projected_envelope_edge_residual_rows"] == 0,
        "Correspondence summary counts changed",
    )
    for key, expected in EXPECTED_MAXIMA.items():
        require(summary[key] == expected, f"Correspondence extremum changed: {key}")
    for camera_name, expected in EXPECTED_CAMERA_MAXIMA.items():
        actual = summary["camera_summary"][camera_name]
        require(actual["projected_object_rows"] == 9, "Camera row count changed")
        require(
            actual["descriptive_only_not_camera_ranking"] is True,
            "Camera boundary changed",
        )
        for key, value in expected.items():
            require(actual[key] == value, f"Camera maximum changed: {camera_name} {key}")
    return comparisons, summary


def audit() -> dict:
    prior, prior_report, prior_pin = load_turn2846_package()
    comparisons, summary = analyze(prior)
    return {
        "schema_version": "wmf-retained-aabb-reflection-correspondence-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "turn2846_reflection_transition_package": prior_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_object_correspondence_rows": 9,
            "mapping_pair_camera_rows": 12,
            "projected_camera_object_correspondence_rows": 36,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
            "D01_D02_command_confounded": True,
            "P00_shared_left_command_with_extra_reflected_right_member": True,
            "isolated_layout_effect_estimated": False,
        },
        "reference_semantics": {
            "constructed_center_reflection": (
                "For each original retained object center (x,y,z), construct the exact "
                "robot-base target (x,-y,z). Translate the original retained robot-base "
                "axis-aligned box by (0,-2y,0), preserving its original extents. This is "
                "a deterministic geometric reference, not an observed target, frozen "
                "fixture pose, scientific tolerance or pass/fail predicate."
            ),
            "actual_correspondent": (
                "Compare that constructed box with the availability-selected actual "
                "reflected-arm retained box, then project both through the unchanged "
                "retained camera calibration. Residuals use exact stored floats without "
                "epsilon, thresholding, fitting, registration or normalization."
            ),
            "orientation_and_settling_boundary": (
                "Preserving the original AABB extents represents pure center translation "
                "with the original retained box shape. Actual-minus-constructed residuals "
                "can mix center settling, orientation settling, axis-aligned-box shape "
                "changes and numeric capture; they are not attributed to any one cause."
            ),
            "camera_boundary": (
                "The four camera configuration objects and hashes are required to match "
                "exactly across each original/reflected mapping pair. Per-camera residual "
                "summaries are descriptive and do not rank or qualify cameras."
            ),
            "command_boundary": prior_report["predicate_semantics"][
                "command_boundary"
            ],
        },
        "summary": summary,
        "comparisons": comparisons,
        "finding": (
            "Across nine availability-selected original/reflected object correspondences "
            "and their 36 camera projections, camera configurations match exactly across "
            "all 12 arm-camera pairs. Relative to a constructed pure center-reflection "
            "translation of each original retained AABB, the largest actual reflected "
            "bound residual is 1.928526908159256 mm; the largest projected envelope-edge "
            "residual is 3.204905157569897 pixels. Both occur for the D02 cube, and no "
            "tolerance is applied. D01/D02 are command-confounded and P00 has unequal "
            "command membership, so no effect or fixture conclusion is estimated."
        ),
        "claim_boundaries": {
            "constructed_reference_only": (
                "The constructed correspondent is derived from retained original geometry; "
                "it is not a recovered intended fixture array, acceptance target, observed "
                "image, counterfactual rollout or independent measurement."
            ),
            "aabb_envelopes_only": (
                "The retained shapes are world.get_bbox-derived axis-aligned boxes and "
                "their projected min/max envelopes, not OBBs, exact meshes, silhouettes, "
                "segmentation masks, image footprints or object pixels."
            ),
            "no_visibility_or_occlusion_inference": (
                "Positive-depth projected envelopes do not authenticate RGB content, "
                "visibility, general occlusion, or robot/static-scene occlusion."
            ),
            "no_measurement_or_endpoint_inference": (
                "No crop/time mapping, labelability, human annotation, r0/rhat/r endpoint, "
                "generated/executed correspondence or A1-A9 result is authenticated."
            ),
            "selection_confounding_and_effects": (
                "The three mapping pairs are availability-selected. D01/D02 are command-"
                "confounded and P00 has unequal command membership. No layout, command, "
                "model, task-success, population or camera-ranking effect is estimated."
            ),
            "science_and_release": (
                "No scientific threshold, fixture/camera qualification, current-state "
                "inference, gate promotion, dispatch/release authority or new behavioral "
                "evidence follows."
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
            "fixture_target_recovered": False,
            "fixture_pass_fail_classification_created": False,
            "scientific_threshold_applied": False,
            "counterfactual_or_temporal_transition_inferred": False,
            "causal_effect_estimated": False,
            "isolated_layout_or_command_effect_estimated": False,
            "camera_ranking_estimated": False,
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
            "Saved retained AABB reflection-correspondence diagnostic differs",
        )
        print(
            "Verified 3 availability-selected mapping pairs, 9 world-object and "
            "36 projected camera-object correspondence rows; no effect, threshold, "
            "fixture-pass, endpoint or gate evidence."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
