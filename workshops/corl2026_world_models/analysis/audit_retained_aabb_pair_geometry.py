#!/usr/bin/env python3
"""Audit retained pairwise projected-AABB and camera-depth geometry only."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
PRIOR_PACKAGE_COMMIT = "196bc61303c22100a1b9676f29ee01a9be43de9a"
REPOSITORY = Path(__file__).resolve().parents[3]
PRIOR_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_aabb_frame_support.py"
    ),
    "bytes": 27383,
    "sha256": "c0ba1ddaf7c3026d1cfda741a7b68080205751436dc31b315deac236265dd42f",
}
PRIOR_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_aabb_frame_support_diagnostic.json"
    ),
    "bytes": 25499,
    "sha256": "aec60c8a6674e5dc9a57f13adaf568c7c621c0f88c57210a082a6b1ac98e165e",
}
EXPECTED_COUNTS = {
    "unique_reset_mappings": 6,
    "camera_rows": 24,
    "object_pairs": 72,
    "positive_area_projected_envelope_overlaps": 39,
    "positive_projected_envelope_separations": 33,
    "projected_envelope_touches": 0,
    "overlapping_camera_z_intervals": 50,
    "strictly_ordered_camera_z_intervals": 22,
    "touching_camera_z_intervals": 0,
    "strictly_nearer_banana": 6,
    "strictly_nearer_bowl": 4,
    "strictly_nearer_rubiks_cube": 12,
    "projected_overlap_and_depth_overlap": 35,
    "projected_separation_and_depth_overlap": 15,
    "projected_overlap_and_strict_depth_order": 4,
    "projected_separation_and_strict_depth_order": 18,
}
EXPECTED_EXTREMA = {
    "minimum_positive_projected_separation_px": 9.743719763835372,
    "maximum_projected_separation_px": 292.3591634814255,
    "maximum_projected_overlap_area_px2": 10724.657636920785,
    "minimum_positive_camera_z_gap_m": 0.001748286047414227,
    "maximum_camera_z_gap_m": 0.14212397257408016,
    "maximum_camera_z_interval_overlap_m": 0.18103834720123602,
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
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "show", f"{commit}:{path}",
        ],
        timeout=30,
    )


def verify_payload(payload: bytes, descriptor: dict, label: str) -> None:
    require(len(payload) == descriptor["bytes"], f"{label} byte count changed")
    require(
        hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
        f"{label} SHA-256 changed",
    )


def load_prior_package() -> tuple[ModuleType, ModuleType, ModuleType, dict, dict]:
    committed_audit = git_bytes(PRIOR_PACKAGE_COMMIT, PRIOR_AUDIT["path"])
    verify_payload(committed_audit, PRIOR_AUDIT, "Pinned prior audit")
    local_audit_path = REPOSITORY / PRIOR_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, PRIOR_AUDIT, "Local prior audit dependency")
    require(local_audit == committed_audit, "Local prior audit differs from pinned blob")

    spec = importlib.util.spec_from_file_location(
        "wmf_pinned_retained_aabb_frame_support", local_audit_path
    )
    require(spec is not None and spec.loader is not None, "Cannot load prior audit")
    prior = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prior)

    prior_payload = git_bytes(PRIOR_PACKAGE_COMMIT, PRIOR_REPORT["path"])
    verify_payload(prior_payload, PRIOR_REPORT, "Pinned prior report")
    prior_report = strict_json_loads(prior_payload, PRIOR_REPORT["path"])
    regenerated = prior.audit()
    require(
        prior.canonical_bytes(regenerated) == prior_payload,
        "Pinned prior report no longer regenerates from its pinned audit",
    )
    require(regenerated == prior_report, "Parsed and regenerated prior reports differ")
    require(
        prior_report["schema_version"]
        == "wmf-retained-aabb-frame-support-diagnostic-v1",
        "Prior report schema changed",
    )
    require(
        prior_report["selection"]["terminal_matched_unique_cells"] == 7
        and prior_report["selection"]["unique_arm_level_reset_mappings"] == 6
        and prior_report["summary"]["mapping_camera_object_envelopes"] == 72
        and prior_report["summary"]["projected_aabb_corners"] == 576,
        "Prior report scope changed",
    )
    prior_inventory = prior_report["exhaustive_results_inventory"]
    require(
        prior_inventory["strict_json_blobs_parsed"] == 370
        and prior_inventory["strict_json_bytes_parsed"] == 1925346
        and prior_inventory["object_aabbs_robot_base_m_occurrences"] == 32
        and prior_inventory["all_occurrences_confined_to_four_pinned_snapshots"]
        is True,
        "Prior exhaustive source inventory changed",
    )
    require(
        prior_report["evidence_boundaries"]["measurement_feasibility_qualified"]
        is False
        and prior_report["evidence_boundaries"]["gate_advanced"] is False
        and prior_report["evidence_boundaries"]["release_authority"] is False,
        "Prior report evidence boundary changed",
    )
    projection, base, geometry_report, _ = prior.load_prior_package()
    return prior, projection, base, geometry_report, {
        "commit": PRIOR_PACKAGE_COMMIT,
        "audit": PRIOR_AUDIT,
        "report": PRIOR_REPORT,
        "canonical_regeneration_verified": True,
        "verified_source_scope": {
            "strict_json_blobs": 370,
            "strict_json_bytes": 1925346,
            "object_aabb_occurrences": 32,
            "all_occurrences_confined_to_four_pinned_snapshots": True,
        },
    }


def finite_values(values: object, length: int, label: str) -> list[float]:
    require(isinstance(values, list) and len(values) == length, f"{label} shape changed")
    output = []
    for value in values:
        require(type(value) in (int, float), f"{label} has a non-numeric value")
        number = float(value)
        require(math.isfinite(number), f"{label} has a non-finite value")
        output.append(number)
    return output


def object_geometry(
    projection: ModuleType,
    geometry: dict,
    configuration: dict,
    object_name: str,
) -> dict:
    bounds = geometry["object_aabbs_robot_base_m"][object_name]
    lower = finite_values(bounds["lower"], 3, f"{object_name} AABB lower")
    upper = finite_values(bounds["upper"], 3, f"{object_name} AABB upper")
    require(
        all(lower[index] < upper[index] for index in range(3)),
        f"{object_name} AABB extent is not strictly positive",
    )
    origin = finite_values(
        configuration["camera_center_robot_base_m"], 3, "camera center"
    )
    quaternion = finite_values(
        configuration["camera_quaternion_world_wxyz_ros"], 4, "camera quaternion"
    )
    width, height = finite_values(configuration["image_size_wh"], 2, "image size")
    require(width == 1280.0 and height == 720.0, "Retained image size changed")

    projections = []
    depths = []
    for corner_tuple in itertools.product(
        (lower[0], upper[0]),
        (lower[1], upper[1]),
        (lower[2], upper[2]),
    ):
        corner = list(corner_tuple)
        local = projection.inverse_rotate_wxyz(
            quaternion,
            [corner[index] - origin[index] for index in range(3)],
        )
        require(local[2] > 1e-9, "Retained AABB corner has non-positive camera depth")
        uv = projection.project(configuration, corner)
        require(
            all(math.isfinite(value) for value in uv),
            "Retained AABB corner projection is non-finite",
        )
        require(
            0.0 < uv[0] < width and 0.0 < uv[1] < height,
            "Retained AABB corner lies outside the configured frame",
        )
        depths.append(local[2])
        projections.append(uv)
    require(len(projections) == 8, "AABB corner count changed")
    envelope = [
        min(uv[0] for uv in projections),
        max(uv[0] for uv in projections),
        min(uv[1] for uv in projections),
        max(uv[1] for uv in projections),
    ]
    depth_interval = [min(depths), max(depths)]
    require(
        envelope[0] < envelope[1] and envelope[2] < envelope[3],
        "Projected AABB envelope has non-positive extent",
    )
    require(depth_interval[0] < depth_interval[1], "Camera-depth interval collapsed")
    return {
        "envelope_uv_px": envelope,
        "camera_z_interval_m": depth_interval,
    }


def pair_geometry(
    first_name: str,
    second_name: str,
    first: dict,
    second: dict,
) -> dict:
    first_envelope = finite_values(first["envelope_uv_px"], 4, "first envelope")
    second_envelope = finite_values(second["envelope_uv_px"], 4, "second envelope")
    first_depth = finite_values(first["camera_z_interval_m"], 2, "first depth")
    second_depth = finite_values(second["camera_z_interval_m"], 2, "second depth")
    require(
        first_envelope[0] < first_envelope[1]
        and first_envelope[2] < first_envelope[3]
        and second_envelope[0] < second_envelope[1]
        and second_envelope[2] < second_envelope[3],
        "Malformed projected envelope",
    )
    require(
        0.0 < first_depth[0] < first_depth[1]
        and 0.0 < second_depth[0] < second_depth[1],
        "Malformed positive camera-depth interval",
    )

    overlap_u = max(
        0.0,
        min(first_envelope[1], second_envelope[1])
        - max(first_envelope[0], second_envelope[0]),
    )
    overlap_v = max(
        0.0,
        min(first_envelope[3], second_envelope[3])
        - max(first_envelope[2], second_envelope[2]),
    )
    gap_u = max(
        0.0,
        first_envelope[0] - second_envelope[1],
        second_envelope[0] - first_envelope[1],
    )
    gap_v = max(
        0.0,
        first_envelope[2] - second_envelope[3],
        second_envelope[2] - first_envelope[3],
    )
    overlap_area = overlap_u * overlap_v
    separation = math.hypot(gap_u, gap_v)
    if overlap_area > 0.0:
        projected_relation = "positive_area_overlap"
        require(separation == 0.0, "Overlapping rectangles have nonzero separation")
    elif separation > 0.0:
        projected_relation = "positive_separation"
    else:
        projected_relation = "boundary_touch_only"

    depth_overlap = max(
        0.0,
        min(first_depth[1], second_depth[1])
        - max(first_depth[0], second_depth[0]),
    )
    if first_depth[1] < second_depth[0]:
        depth_relation = "first_strictly_nearer_in_camera_z"
        nearer_object = first_name
        depth_gap = second_depth[0] - first_depth[1]
        require(depth_overlap == 0.0, "Strict depth order has nonzero overlap")
    elif second_depth[1] < first_depth[0]:
        depth_relation = "second_strictly_nearer_in_camera_z"
        nearer_object = second_name
        depth_gap = first_depth[0] - second_depth[1]
        require(depth_overlap == 0.0, "Strict depth order has nonzero overlap")
    elif depth_overlap > 0.0:
        depth_relation = "overlapping_camera_z_intervals"
        nearer_object = None
        depth_gap = 0.0
    else:
        depth_relation = "camera_z_interval_boundary_touch_only"
        nearer_object = None
        depth_gap = 0.0

    return {
        "objects": [first_name, second_name],
        "projected_envelope_relation": projected_relation,
        "axis_overlap_uv_px": [overlap_u, overlap_v],
        "positive_overlap_area_px2": overlap_area,
        "axis_separation_uv_px": [gap_u, gap_v],
        "euclidean_rectangle_separation_px": separation,
        "camera_z_interval_relation": depth_relation,
        "strictly_nearer_object_in_camera_z": nearer_object,
        "positive_camera_z_interval_gap_m": depth_gap,
        "positive_camera_z_interval_overlap_m": depth_overlap,
    }


def row_label(mapping: dict, camera_name: str, pair: list[str]) -> dict:
    return {
        "model_id": mapping["model_id"],
        "layout_pair_id": mapping["layout_pair_id"],
        "layout_arm": mapping["layout_arm"],
        "representative_cell_id": mapping["representative_cell_id"],
        "camera_name": camera_name,
        "object_pair": pair,
    }


def extremum_row(row: tuple[float, dict], unit: str) -> dict:
    return {unit: row[0], **row[1]}


def analyze(
    prior: ModuleType,
    projection: ModuleType,
    base: ModuleType,
    geometry_report: dict,
) -> tuple[list[dict], dict]:
    mappings = prior.collapse_unique_mappings(projection, base, geometry_report)
    require(len(mappings) == 6, "Unique reset-mapping count changed")
    pairs = tuple(itertools.combinations(base.OBJECT_NAMES, 2))
    require(
        pairs
        == (
            ("banana", "bowl"),
            ("banana", "rubiks_cube"),
            ("bowl", "rubiks_cube"),
        ),
        "Retained object-pair order changed",
    )

    projected_relations: Counter[str] = Counter()
    depth_relations: Counter[str] = Counter()
    nearer_objects: Counter[str] = Counter()
    joint_relations: Counter[str] = Counter()
    by_camera: dict[str, Counter[str]] = {
        name: Counter() for name in base.CAMERA_NAMES
    }
    by_pair: dict[str, Counter[str]] = {
        "|".join(pair): Counter() for pair in pairs
    }
    separations: list[tuple[float, dict]] = []
    overlap_areas: list[tuple[float, dict]] = []
    depth_gaps: list[tuple[float, dict]] = []
    depth_overlaps: list[tuple[float, dict]] = []
    output = []

    for mapping in mappings:
        cameras = mapping["cameras"]
        require(tuple(cameras) == base.CAMERA_NAMES, "Retained camera order changed")
        camera_rows = []
        for camera_name in base.CAMERA_NAMES:
            camera = cameras[camera_name]
            require(
                camera["visibility_method"]
                == "calibrated_projection_and_obb_occlusion",
                "Retained visibility-method label changed",
            )
            geometry = camera["geometry"]
            configuration = camera["camera_configuration"]
            require(
                all(configuration[key] == geometry[key] for key in configuration),
                "Top-level/nested camera configuration differs",
            )
            objects = {
                name: object_geometry(projection, geometry, configuration, name)
                for name in base.OBJECT_NAMES
            }
            pair_rows = []
            for first_name, second_name in pairs:
                metrics = pair_geometry(
                    first_name,
                    second_name,
                    objects[first_name],
                    objects[second_name],
                )
                label = row_label(mapping, camera_name, metrics["objects"])
                projected_relation = metrics["projected_envelope_relation"]
                depth_relation = metrics["camera_z_interval_relation"]
                projected_relations[projected_relation] += 1
                depth_relations[depth_relation] += 1
                nearer = metrics["strictly_nearer_object_in_camera_z"]
                if nearer is not None:
                    nearer_objects[nearer] += 1
                projected_kind = (
                    "projected_overlap"
                    if projected_relation == "positive_area_overlap"
                    else "projected_separation"
                    if projected_relation == "positive_separation"
                    else "projected_touch"
                )
                depth_kind = (
                    "depth_overlap"
                    if depth_relation == "overlapping_camera_z_intervals"
                    else "strict_depth_order"
                    if "strictly_nearer" in depth_relation
                    else "depth_touch"
                )
                joint_relations[f"{projected_kind}_and_{depth_kind}"] += 1
                by_camera[camera_name][projected_relation] += 1
                by_camera[camera_name][depth_relation] += 1
                pair_key = "|".join(metrics["objects"])
                by_pair[pair_key][projected_relation] += 1
                by_pair[pair_key][depth_relation] += 1

                separation = metrics["euclidean_rectangle_separation_px"]
                overlap_area = metrics["positive_overlap_area_px2"]
                depth_gap = metrics["positive_camera_z_interval_gap_m"]
                depth_overlap = metrics["positive_camera_z_interval_overlap_m"]
                if separation > 0.0:
                    separations.append((separation, label))
                if overlap_area > 0.0:
                    overlap_areas.append((overlap_area, label))
                if depth_gap > 0.0:
                    depth_gaps.append((depth_gap, label))
                if depth_overlap > 0.0:
                    depth_overlaps.append((depth_overlap, label))
                pair_rows.append(metrics)
            camera_rows.append({
                "camera_name": camera_name,
                "camera_configuration_sha256": camera["camera_configuration_sha256"],
                "object_pair_rows": pair_rows,
            })
        output.append({
            key: value for key, value in mapping.items() if key != "cameras"
        } | {"camera_rows": camera_rows})

    actual_counts = {
        "unique_reset_mappings": len(output),
        "camera_rows": len(output) * len(base.CAMERA_NAMES),
        "object_pairs": sum(
            len(camera["object_pair_rows"])
            for mapping in output for camera in mapping["camera_rows"]
        ),
        "positive_area_projected_envelope_overlaps": projected_relations[
            "positive_area_overlap"
        ],
        "positive_projected_envelope_separations": projected_relations[
            "positive_separation"
        ],
        "projected_envelope_touches": projected_relations["boundary_touch_only"],
        "overlapping_camera_z_intervals": depth_relations[
            "overlapping_camera_z_intervals"
        ],
        "strictly_ordered_camera_z_intervals": (
            depth_relations["first_strictly_nearer_in_camera_z"]
            + depth_relations["second_strictly_nearer_in_camera_z"]
        ),
        "touching_camera_z_intervals": depth_relations[
            "camera_z_interval_boundary_touch_only"
        ],
        "strictly_nearer_banana": nearer_objects["banana"],
        "strictly_nearer_bowl": nearer_objects["bowl"],
        "strictly_nearer_rubiks_cube": nearer_objects["rubiks_cube"],
        "projected_overlap_and_depth_overlap": joint_relations[
            "projected_overlap_and_depth_overlap"
        ],
        "projected_separation_and_depth_overlap": joint_relations[
            "projected_separation_and_depth_overlap"
        ],
        "projected_overlap_and_strict_depth_order": joint_relations[
            "projected_overlap_and_strict_depth_order"
        ],
        "projected_separation_and_strict_depth_order": joint_relations[
            "projected_separation_and_strict_depth_order"
        ],
    }
    require(actual_counts == EXPECTED_COUNTS, "Pairwise geometry counts changed")
    require(len(separations) == 33, "Positive projected-separation count changed")
    require(len(overlap_areas) == 39, "Positive projected-overlap count changed")
    require(len(depth_gaps) == 22, "Positive camera-depth-gap count changed")
    require(len(depth_overlaps) == 50, "Positive camera-depth-overlap count changed")

    minimum_separation = min(separations, key=lambda row: row[0])
    maximum_separation = max(separations, key=lambda row: row[0])
    maximum_overlap_area = max(overlap_areas, key=lambda row: row[0])
    minimum_depth_gap = min(depth_gaps, key=lambda row: row[0])
    maximum_depth_gap = max(depth_gaps, key=lambda row: row[0])
    maximum_depth_overlap = max(depth_overlaps, key=lambda row: row[0])
    require(
        minimum_separation[0]
        == EXPECTED_EXTREMA["minimum_positive_projected_separation_px"],
        "Minimum positive projected separation changed",
    )
    require(
        maximum_separation[0]
        == EXPECTED_EXTREMA["maximum_projected_separation_px"],
        "Maximum projected separation changed",
    )
    require(
        maximum_overlap_area[0]
        == EXPECTED_EXTREMA["maximum_projected_overlap_area_px2"],
        "Maximum projected overlap area changed",
    )
    require(
        minimum_depth_gap[0]
        == EXPECTED_EXTREMA["minimum_positive_camera_z_gap_m"],
        "Minimum positive camera-depth gap changed",
    )
    require(
        maximum_depth_gap[0] == EXPECTED_EXTREMA["maximum_camera_z_gap_m"],
        "Maximum camera-depth gap changed",
    )
    require(
        maximum_depth_overlap[0]
        == EXPECTED_EXTREMA["maximum_camera_z_interval_overlap_m"],
        "Maximum camera-depth interval overlap changed",
    )

    camera_summary = {
        camera_name: dict(sorted(counter.items()))
        for camera_name, counter in by_camera.items()
    }
    pair_summary = {
        pair_name: dict(sorted(counter.items()))
        for pair_name, counter in by_pair.items()
    }
    return output, {
        **actual_counts,
        "camera_relation_counts": camera_summary,
        "object_pair_relation_counts": pair_summary,
        "minimum_positive_projected_separation": extremum_row(
            minimum_separation, "pixels"
        ),
        "maximum_projected_separation": extremum_row(
            maximum_separation, "pixels"
        ),
        "maximum_projected_overlap_area": extremum_row(
            maximum_overlap_area, "square_pixels"
        ),
        "minimum_positive_camera_z_gap": extremum_row(
            minimum_depth_gap, "meters"
        ),
        "maximum_camera_z_gap": extremum_row(maximum_depth_gap, "meters"),
        "maximum_camera_z_interval_overlap": extremum_row(
            maximum_depth_overlap, "meters"
        ),
    }


def audit() -> dict:
    prior, projection, base, geometry_report, prior_pin = load_prior_package()
    require(base.RESULTS_COMMIT == RESULTS_COMMIT, "Prior results pin changed")
    mappings, summary = analyze(prior, projection, base, geometry_report)
    return {
        "schema_version": "wmf-retained-aabb-pair-geometry-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "prior_aabb_frame_support_package": prior_pin,
        "selection": {
            "terminal_matched_unique_cells": 7,
            "unique_arm_level_reset_mappings": 6,
            "mapping_camera_rows": 24,
            "mapping_camera_object_pairs": 72,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
        },
        "predicate_semantics": {
            "projected_envelopes": (
                "For each retained world.get_bbox-derived robot-base AABB, project its "
                "eight corners with the pinned camera formula and take independent "
                "u/v minima and maxima. Pairwise positive-area rectangle overlap, "
                "positive rectangle separation or exact boundary touch is then computed "
                "without an epsilon or scientific threshold."
            ),
            "camera_depth_intervals": (
                "Camera z is computed for the same eight AABB corners. One AABB is "
                "strictly nearer in camera z only when its maximum z is below the other "
                "AABB's minimum z; otherwise intervals overlap or touch. Camera z is "
                "not Euclidean range and this order is not a visibility result."
            ),
            "inherited_method_label_boundary": (
                "The retained label calibrated_projection_and_obb_occlusion fronts a "
                "per-axis min/max AABB and target-center-ray-only implementation. This "
                "audit does not reinterpret that label as OBB or general occlusion."
            ),
        },
        "summary": summary,
        "mappings": mappings,
        "finding": (
            "Across six availability-selected unique reset mappings, four cameras and "
            "three retained-object pairs, 39 of 72 projected AABB-envelope pairs have "
            "positive rectangle-overlap area and 33 have positive rectangle separation; "
            "none only touch. Camera-z intervals overlap for 50 pairs and are strictly "
            "ordered for 22; none only touch. These are axis-aligned retained-geometry "
            "descriptors, not observed-object or visibility evidence."
        ),
        "claim_boundaries": {
            "aabb_envelopes_only": (
                "The projected rectangles bound retained axis-aligned boxes derived from "
                "world.get_bbox corner minima/maxima. They are not OBBs, exact meshes, "
                "silhouettes, segmentation masks, image footprints or object pixels."
            ),
            "no_visibility_or_occlusion_inference": (
                "Rectangle overlap does not prove silhouette overlap or occlusion; "
                "rectangle separation does not prove object visibility. Camera-z order "
                "does not identify a visible front object. The robot, static scene, RGB "
                "content and general ray/surface occlusion are not analyzed."
            ),
            "no_measurement_or_endpoint_inference": (
                "No RGB artifact, crop/time mapping, labelability, human annotation, "
                "measurement feasibility, r0/rhat/r endpoint or generated/executed "
                "correspondence is authenticated. No A1-A9 result follows."
            ),
            "selection_and_effects": (
                "The six reset mappings are availability-selected from seven prior-"
                "authenticated cells and are not a representative or full cohort. No "
                "layout, command, model, task-success or population effect is estimated, "
                "and the per-camera descriptors do not define a camera ranking."
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
            "obb_analyzed": False,
            "general_occlusion_analyzed": False,
            "robot_or_static_scene_occlusion_analyzed": False,
            "human_image_centroids_used": False,
            "camera_crop_or_time_alignment_authenticated": False,
            "labelability_estimated": False,
            "measurement_feasibility_qualified": False,
            "camera_ranking_estimated": False,
            "cross_time_field_join_performed": False,
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
        "--check", type=Path,
        help="Verify an existing canonical diagnostic without modifying it",
    )
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(
            args.check.read_bytes() == encoded,
            "Saved retained AABB pair-geometry diagnostic differs",
        )
        print(
            "Verified 6 retained reset mappings, 24 camera rows and 72 pairwise "
            "projected-envelope/depth-interval rows; no RGB, endpoint or gate evidence."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
