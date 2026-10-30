#!/usr/bin/env python3
"""Decompose frozen retained AABB reflection residuals by signed component."""

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
TURN2847_PACKAGE_COMMIT = "717136db3c518a3e59a01d33a86f95839ee83b23"
TURN2847_PACKAGE_TREE = "8c5512aa56308a9d5abefcc81996e92b57cd6b94"
TURN2847_PACKAGE_PARENT = "4a57181f620845b5b28234968039861e4985f382"
REPOSITORY = Path(__file__).resolve().parents[3]
TURN2847_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_aabb_reflection_correspondence.py"
    ),
    "bytes": 39566,
    "sha256": "c31445dfffa8d697c3d32de9506c0792e3111c1ecab855e76f0d4feca68c6cc7",
}
TURN2847_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_aabb_reflection_correspondence_diagnostic.json"
    ),
    "bytes": 106572,
    "sha256": "9d1944c7b2b4946ce95e60c6e56d3d0dded69be2647d697ae02020aa82ce53fe",
}
COMPARISON_KEYS = (
    ("development", "D01", "D1"),
    ("development", "D02", "D1"),
    ("pilot", "P00", "N3"),
)
OBJECT_NAMES = ("banana", "bowl", "rubiks_cube")
CAMERA_NAMES = (
    "head_camera",
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
WORLD_COMPONENTS = {
    "center": (
        "actual_minus_constructed_center_target_m",
        ("x", "y", "z"),
    ),
    "lower_bound": (
        "actual_minus_constructed_lower_bounds_m",
        ("x", "y", "z"),
    ),
    "upper_bound": (
        "actual_minus_constructed_upper_bounds_m",
        ("x", "y", "z"),
    ),
    "aabb_midpoint": (
        "actual_minus_constructed_aabb_midpoint_m",
        ("x", "y", "z"),
    ),
    "aabb_extent": (
        "actual_minus_constructed_aabb_extent_m",
        ("x", "y", "z"),
    ),
}
PROJECTED_COMPONENTS = {
    "envelope_edge": (
        "actual_minus_constructed_envelope_edges_px",
        ("u_min", "u_max", "v_min", "v_max"),
        "px",
    ),
    "envelope_center": (
        "actual_minus_constructed_envelope_center_uv_px",
        ("u", "v"),
        "px",
    ),
    "envelope_extent": (
        "actual_minus_constructed_envelope_extent_wh_px",
        ("width", "height"),
        "px",
    ),
    "camera_z_interval": (
        "actual_minus_constructed_camera_z_interval_edges_m",
        ("lower", "upper"),
        "m",
    ),
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


def git_line(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(REPOSITORY), *arguments],
        text=True,
        timeout=30,
    ).strip()


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


def load_turn2847_package() -> tuple[dict, dict]:
    require(
        git_line("rev-parse", f"{TURN2847_PACKAGE_COMMIT}^{{commit}}")
        == TURN2847_PACKAGE_COMMIT,
        "Pinned Turn2847 evidence commit does not resolve exactly",
    )
    require(
        git_line("show", "-s", "--format=%T", TURN2847_PACKAGE_COMMIT)
        == TURN2847_PACKAGE_TREE,
        "Pinned Turn2847 evidence tree changed",
    )
    require(
        git_line("show", "-s", "--format=%P", TURN2847_PACKAGE_COMMIT)
        == TURN2847_PACKAGE_PARENT,
        "Pinned Turn2847 evidence parent changed",
    )

    committed_audit = git_bytes(TURN2847_PACKAGE_COMMIT, TURN2847_AUDIT["path"])
    verify_payload(committed_audit, TURN2847_AUDIT, "Pinned Turn2847 audit")
    local_audit_path = REPOSITORY / TURN2847_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, TURN2847_AUDIT, "Local Turn2847 audit")
    require(local_audit == committed_audit, "Local Turn2847 audit differs")
    prior = import_module(local_audit_path, "wmf_pinned_turn2847_correspondence")

    committed_report = git_bytes(TURN2847_PACKAGE_COMMIT, TURN2847_REPORT["path"])
    verify_payload(committed_report, TURN2847_REPORT, "Pinned Turn2847 report")
    local_report = (REPOSITORY / TURN2847_REPORT["path"]).read_bytes()
    verify_payload(local_report, TURN2847_REPORT, "Local Turn2847 report")
    require(local_report == committed_report, "Local Turn2847 report differs")
    parsed = strict_json_loads(committed_report, TURN2847_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == committed_report,
        "Turn2847 report no longer regenerates canonically",
    )
    require(regenerated == parsed, "Parsed and regenerated Turn2847 reports differ")
    require(
        parsed["schema_version"]
        == "wmf-retained-aabb-reflection-correspondence-diagnostic-v1",
        "Turn2847 schema changed",
    )
    require(parsed["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    selection = parsed["selection"]
    require(
        selection["source_reset_mappings"] == 6
        and selection["comparable_original_reflected_mapping_pairs"] == 3
        and selection["world_object_correspondence_rows"] == 9
        and selection["mapping_pair_camera_rows"] == 12
        and selection["projected_camera_object_correspondence_rows"] == 36
        and selection["availability_selected"] is True
        and selection["representative_sample"] is False
        and selection["full_completed_cohort"] is False
        and selection["cross_time_field_join_performed"] is False
        and selection["D01_D02_command_confounded"] is True
        and selection["P00_shared_left_command_with_extra_reflected_right_member"]
        is True
        and selection["isolated_layout_effect_estimated"] is False,
        "Turn2847 selection boundary changed",
    )
    require(
        all(value is False for value in parsed["evidence_boundaries"].values()),
        "Turn2847 evidence boundary changed",
    )
    require(
        parsed["new_behavioral_cells"] == 0
        and parsed["new_model_requests"] == 0
        and parsed["labels_created"] == 0
        and parsed["forecast_accuracy_estimated"] is False
        and parsed["resource_budget_qualified"] is False
        and parsed["confirmation_released"] is False,
        "Turn2847 zero-science boundary changed",
    )
    pin = {
        "commit": TURN2847_PACKAGE_COMMIT,
        "tree": TURN2847_PACKAGE_TREE,
        "parent": TURN2847_PACKAGE_PARENT,
        "audit": TURN2847_AUDIT,
        "report": TURN2847_REPORT,
        "canonical_regeneration_verified": True,
    }
    return parsed, pin


def finite_number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} is not numeric")
    number = float(value)
    require(math.isfinite(number), f"{label} is not finite")
    return number


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape")
    return [finite_number(child, label) for child in value]


def l2(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def provenance(comparison: dict, object_name: str, camera_name: str | None) -> dict:
    value = {
        "source_package_commit": TURN2847_PACKAGE_COMMIT,
        "phase": comparison["phase"],
        "layout_pair_id": comparison["layout_pair_id"],
        "model_id": comparison["model_id"],
        "object_name": object_name,
        "original_representative_cell_id": comparison[
            "original_representative_cell_id"
        ],
        "reflected_representative_cell_id": comparison[
            "reflected_representative_cell_id"
        ],
        "original_member_cell_ids": comparison["original_member_cell_ids"],
        "reflected_member_cell_ids": comparison["reflected_member_cell_ids"],
        "original_member_commands": comparison["original_member_commands"],
        "reflected_member_commands": comparison["reflected_member_commands"],
        "command_membership_classification": comparison[
            "command_membership_classification"
        ],
    }
    if camera_name is not None:
        value["camera_name"] = camera_name
    return value


def component_record(
    family: str,
    component: str,
    signed_value: float,
    unit: str,
) -> dict:
    signed = finite_number(signed_value, f"{family} {component}")
    record = {
        "family": family,
        "component": component,
        f"signed_{unit}": signed,
        f"absolute_{unit}": abs(signed),
    }
    if unit == "m":
        record["signed_mm"] = 1000.0 * signed
        record["absolute_mm"] = 1000.0 * abs(signed)
    return record


def decompose_vectors(
    source: dict,
    definitions: dict,
    label: str,
) -> list[dict]:
    rows = []
    for family, definition in definitions.items():
        source_key = definition[0]
        component_names = definition[1]
        unit = definition[2] if len(definition) == 3 else "m"
        values = finite_vector(
            source.get(source_key), len(component_names), f"{label} {source_key}"
        )
        rows.extend(
            component_record(family, component, value, unit)
            for component, value in zip(component_names, values, strict=True)
        )
    return rows


def validate_world_source(row: dict, components: list[dict]) -> None:
    by_family = {}
    for component in components:
        by_family.setdefault(component["family"], []).append(component["signed_m"])
    require(
        1000.0 * l2(by_family["center"])
        == finite_number(row["center_target_l2_residual_mm"], "center l2"),
        "Source center aggregate changed",
    )
    require(
        1000.0
        * max(
            abs(value)
            for family in ("lower_bound", "upper_bound")
            for value in by_family[family]
        )
        == finite_number(row["maximum_absolute_bound_residual_mm"], "bound max"),
        "Source bound aggregate changed",
    )
    require(
        1000.0 * l2(by_family["aabb_midpoint"])
        == finite_number(row["aabb_midpoint_l2_residual_mm"], "midpoint l2"),
        "Source midpoint aggregate changed",
    )
    require(
        1000.0 * max(abs(value) for value in by_family["aabb_extent"])
        == finite_number(row["maximum_absolute_extent_residual_mm"], "extent max"),
        "Source extent aggregate changed",
    )


def validate_projected_source(row: dict, components: list[dict]) -> None:
    by_family = {}
    for component in components:
        unit = "m" if component["family"] == "camera_z_interval" else "px"
        by_family.setdefault(component["family"], []).append(
            component[f"signed_{unit}"]
        )
    require(
        max(abs(value) for value in by_family["envelope_edge"])
        == finite_number(
            row["maximum_absolute_envelope_edge_residual_px"], "envelope edge max"
        ),
        "Source envelope-edge aggregate changed",
    )
    require(
        l2(by_family["envelope_center"])
        == finite_number(row["envelope_center_l2_residual_px"], "envelope center l2"),
        "Source envelope-center aggregate changed",
    )
    require(
        max(abs(value) for value in by_family["envelope_extent"])
        == finite_number(
            row["maximum_absolute_envelope_extent_residual_px"],
            "envelope extent max",
        ),
        "Source envelope-extent aggregate changed",
    )
    require(
        1000.0 * max(abs(value) for value in by_family["camera_z_interval"])
        == finite_number(
            row["maximum_absolute_camera_z_interval_edge_residual_mm"],
            "camera-z max",
        ),
        "Source camera-z aggregate changed",
    )


def metric_rows(component_rows: list[dict], row_provenance: dict) -> list[dict]:
    metrics = []
    for component in component_rows:
        unit = "m" if "signed_m" in component else "px"
        metrics.append(
            {
                "family": component["family"],
                "component": component["component"],
                "signed_value": component[f"signed_{unit}"],
                "absolute_value": component[f"absolute_{unit}"],
                "unit": unit,
                "provenance": row_provenance,
            }
        )
    return metrics


def extremum_record(metric: dict) -> dict:
    result = {
        "signed_value": metric["signed_value"],
        "absolute_value": metric["absolute_value"],
        "unit": metric["unit"],
        "provenance": metric["provenance"],
    }
    if metric["unit"] == "m":
        result["signed_mm"] = 1000.0 * metric["signed_value"]
        result["absolute_mm"] = 1000.0 * metric["absolute_value"]
    return result


def summarize_extrema(metrics: list[dict]) -> dict:
    require(metrics, "Cannot summarize an empty metric set")
    grouped: dict[str, dict[str, list[dict]]] = {}
    for metric in metrics:
        grouped.setdefault(metric["family"], {}).setdefault(
            metric["component"], []
        ).append(metric)
    result = {}
    for family, components in grouped.items():
        result[family] = {}
        for component, values in components.items():
            units = {value["unit"] for value in values}
            require(len(units) == 1, f"Mixed units for {family} {component}")
            result[family][component] = {
                "signed_minimum": extremum_record(
                    min(values, key=lambda value: value["signed_value"])
                ),
                "signed_maximum": extremum_record(
                    max(values, key=lambda value: value["signed_value"])
                ),
                "maximum_absolute": extremum_record(
                    max(values, key=lambda value: value["absolute_value"])
                ),
            }
    return result


def largest_absolute(metrics: list[dict], unit: str) -> dict:
    eligible = [metric for metric in metrics if metric["unit"] == unit]
    require(eligible, f"No {unit} metrics")
    metric = max(eligible, key=lambda value: value["absolute_value"])
    return {
        "family": metric["family"],
        "component": metric["component"],
        **extremum_record(metric),
    }


def analyze(prior_report: dict) -> tuple[list[dict], list[dict], dict, dict]:
    comparisons = prior_report["comparisons"]
    require(
        tuple(
            (comparison["phase"], comparison["layout_pair_id"], comparison["model_id"])
            for comparison in comparisons
        )
        == COMPARISON_KEYS,
        "Turn2847 comparison identities or order changed",
    )
    world_rows = []
    projected_rows = []
    world_metrics: list[dict] = []
    projected_metrics: list[dict] = []

    for comparison in comparisons:
        require(
            comparison["camera_configurations_exactly_identical_across_arms"] is True,
            "Cross-arm camera identity changed",
        )
        source_world = comparison["world_object_rows"]
        require(
            tuple(row["object_name"] for row in source_world) == OBJECT_NAMES,
            "World object order changed",
        )
        for source_row in source_world:
            object_name = source_row["object_name"]
            row_provenance = provenance(comparison, object_name, None)
            components = decompose_vectors(
                source_row, WORLD_COMPONENTS, f"world {object_name}"
            )
            validate_world_source(source_row, components)
            world_rows.append(
                {
                    "provenance": row_provenance,
                    "signed_absolute_component_rows": components,
                }
            )
            world_metrics.extend(metric_rows(components, row_provenance))

        source_cameras = comparison["camera_rows"]
        require(
            tuple(row["camera_name"] for row in source_cameras) == CAMERA_NAMES,
            "Camera order changed",
        )
        for camera in source_cameras:
            camera_name = camera["camera_name"]
            camera_sha256 = camera["camera_configuration_sha256"]
            require(
                isinstance(camera_sha256, str)
                and len(camera_sha256) == 64
                and all(character in "0123456789abcdef" for character in camera_sha256),
                f"Malformed camera hash: {camera_name}",
            )
            source_objects = camera["object_rows"]
            require(
                tuple(row["object_name"] for row in source_objects) == OBJECT_NAMES,
                f"Projected object order changed: {camera_name}",
            )
            for source_row in source_objects:
                object_name = source_row["object_name"]
                row_provenance = provenance(comparison, object_name, camera_name)
                row_provenance["camera_configuration_sha256"] = camera_sha256
                components = decompose_vectors(
                    source_row,
                    PROJECTED_COMPONENTS,
                    f"projected {camera_name} {object_name}",
                )
                validate_projected_source(source_row, components)
                projected_rows.append(
                    {
                        "provenance": row_provenance,
                        "signed_absolute_component_rows": components,
                    }
                )
                projected_metrics.extend(metric_rows(components, row_provenance))

    require(len(world_rows) == 9, "World row count changed")
    require(len(projected_rows) == 36, "Projected row count changed")
    require(len(world_metrics) == 135, "World component count changed")
    require(len(projected_metrics) == 360, "Projected component count changed")

    extrema = {
        "world": {
            "global": summarize_extrema(world_metrics),
            "by_object": {
                object_name: summarize_extrema(
                    [
                        metric
                        for metric in world_metrics
                        if metric["provenance"]["object_name"] == object_name
                    ]
                )
                for object_name in OBJECT_NAMES
            },
        },
        "projected": {
            "global": summarize_extrema(projected_metrics),
            "by_object": {
                object_name: summarize_extrema(
                    [
                        metric
                        for metric in projected_metrics
                        if metric["provenance"]["object_name"] == object_name
                    ]
                )
                for object_name in OBJECT_NAMES
            },
            "by_camera": {
                camera_name: summarize_extrema(
                    [
                        metric
                        for metric in projected_metrics
                        if metric["provenance"]["camera_name"] == camera_name
                    ]
                )
                for camera_name in CAMERA_NAMES
            },
        },
    }
    summary = {
        "original_reflected_mapping_pairs": 3,
        "world_object_rows": len(world_rows),
        "world_signed_absolute_component_rows": len(world_metrics),
        "projected_camera_object_rows": len(projected_rows),
        "projected_signed_absolute_component_rows": len(projected_metrics),
        "world_component_families": len(WORLD_COMPONENTS),
        "projected_component_families": len(PROJECTED_COMPONENTS),
        "world_component_axes_per_object_row": sum(
            len(definition[1]) for definition in WORLD_COMPONENTS.values()
        ),
        "projected_component_axes_per_camera_object_row": sum(
            len(definition[1]) for definition in PROJECTED_COMPONENTS.values()
        ),
        "largest_absolute_world_component": largest_absolute(world_metrics, "m"),
        "largest_absolute_projected_image_component": largest_absolute(
            projected_metrics, "px"
        ),
        "largest_absolute_camera_z_component": largest_absolute(
            [
                metric
                for metric in projected_metrics
                if metric["family"] == "camera_z_interval"
            ],
            "m",
        ),
    }
    require(
        summary["world_component_axes_per_object_row"] == 15
        and summary["projected_component_axes_per_camera_object_row"] == 10,
        "Per-row component count changed",
    )
    return world_rows, projected_rows, extrema, summary


def audit() -> dict:
    prior_report, prior_pin = load_turn2847_package()
    world_rows, projected_rows, extrema, summary = analyze(prior_report)
    return {
        "schema_version": (
            "wmf-retained-aabb-reflection-residual-decomposition-diagnostic-v1"
        ),
        "results_commit": RESULTS_COMMIT,
        "turn2847_correspondence_package": prior_pin,
        "selection": {
            "source_reset_mappings": 6,
            "comparable_original_reflected_mapping_pairs": 3,
            "world_object_rows": 9,
            "projected_camera_object_rows": 36,
            "world_signed_absolute_component_rows": 135,
            "projected_signed_absolute_component_rows": 360,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
            "D01_D02_command_confounded": True,
            "P00_shared_left_command_with_extra_reflected_right_member": True,
            "isolated_layout_or_command_effect_estimated": False,
        },
        "decomposition_semantics": {
            "source": (
                "Every value is a deterministic expansion of the exact frozen Turn2847 "
                "actual-minus-constructed correspondence residual vectors. No new source "
                "record, raw camera byte, cluster artifact or current-state field is read."
            ),
            "signed_components": (
                "World signs follow robot-base x/y/z. Projected image signs follow stored "
                "u/v coordinates; width and height are upper-minus-lower envelope extents. "
                "Camera-z signs follow stored lower/upper interval edges. Absolute values "
                "are abs(signed value) with no epsilon, clipping or normalization."
            ),
            "extrema": (
                "Minimum, maximum and largest-absolute records are deterministic descriptive "
                "summaries with full source provenance. They do not order scientific merit, "
                "select a camera or establish an acceptance predicate."
            ),
            "constructed_reference": (
                "The frozen upstream reference reflects each retained original center from "
                "(x,y,z) to (x,-y,z) and translates its retained AABB without changing the "
                "original extents. It is constructed geometry, not an intended fixture pose, "
                "an acceptance specification, an observed image or an independent measure."
            ),
            "mixed_residual_sources": (
                "Residual components can mix center settling, orientation settling, AABB "
                "shape changes and numeric capture. This decomposition assigns no component "
                "to any physical cause."
            ),
        },
        "summary": summary,
        "world_rows": world_rows,
        "projected_rows": projected_rows,
        "extrema": extrema,
        "finding": (
            "The nine frozen world rows expand to 135 signed/absolute axis components, and "
            "the 36 frozen camera-object rows expand to 360 signed/absolute projected/depth "
            "components. The report preserves exact row provenance and descriptive extrema "
            "globally, by object and by camera. It introduces no threshold, fitting, "
            "measurement endpoint or scientific comparison."
        ),
        "claim_boundaries": {
            "frozen_correspondence_expansion_only": (
                "This report decomposes a pinned prior diagnostic and adds no independent "
                "observation, scene, mapping, label, model request or robot episode."
            ),
            "aabb_envelopes_only": (
                "Inputs remain world.get_bbox-derived axis-aligned boxes and projected "
                "min/max envelopes, not OBBs, meshes, silhouettes, masks or image pixels."
            ),
            "no_measurement_or_endpoint_inference": (
                "No crop/time mapping, labelability, annotation, r0/rhat/r endpoint, "
                "generated/executed correspondence or A1-A9 result is authenticated."
            ),
            "selection_confounding": (
                "All three mapping pairs remain availability-selected. D01/D02 have "
                "disjoint command membership and P00 has unequal command membership."
            ),
            "descriptive_extrema_only": (
                "Component extrema are not tolerances, quality scores, camera comparisons, "
                "fixture decisions, population estimates or evidence of physical cause."
            ),
            "science_and_dispatch": (
                "No scientific stage, qualification state, dispatch authority or "
                "confirmation status changes from this retained-only decomposition."
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
            "fixture_reference_recovered": False,
            "fixture_pass_fail_classification_created": False,
            "scientific_threshold_applied": False,
            "fitting_or_registration_performed": False,
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
            "Saved retained residual-decomposition diagnostic differs",
        )
        print(
            "Verified 9 world rows / 135 components and 36 projected rows / "
            "360 components; retained-only, threshold-free, and no stage advance."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
