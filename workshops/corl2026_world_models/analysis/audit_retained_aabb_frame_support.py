#!/usr/bin/env python3
"""Audit retained reset-stage AABB-corner frame support without gate promotion."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
from types import ModuleType


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
PRIOR_PACKAGE_COMMIT = "9263a6ce0eb37ba3264d09f46b7519abe2d4ae32"
REPOSITORY = Path(__file__).resolve().parents[3]
PRIOR_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_reset_projection.py"
    ),
    "bytes": 34959,
    "sha256": "e97c4c1c5cdb5272ede1e6991dffdd7c8eb27aa9e85f412b61f0ffbff3eb3c57",
}
PRIOR_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_reset_projection_diagnostic.json"
    ),
    "bytes": 64195,
    "sha256": "38e637cfd09a5951684e80361ff44026f353c123f6c83c4d11dc27a7237b3c72",
}
EXPECTED_OCCURRENCES = {
    "results/jobs/d1-development-d01-snapshot-001/publish/diagnostic.json": 8,
    "results/jobs/d1-development-d02-snapshot-001/publish/diagnostic.json": 8,
    "results/jobs/detached-attempt-snapshot-002/publish/diagnostic.json": 4,
    "results/jobs/detached-attempt-snapshot-003/publish/diagnostic.json": 12,
}
EXPECTED_EXTREMA = {
    "minimum_corner_frame_margin_px": 46.21332507478144,
    "minimum_positive_corner_depth_m": 0.3314892624027577,
    "maximum_envelope_width_px": 321.3965762806249,
    "maximum_envelope_height_px": 277.2929814120689,
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


def load_prior_package() -> tuple[ModuleType, ModuleType, dict, dict]:
    committed_audit = git_bytes(PRIOR_PACKAGE_COMMIT, PRIOR_AUDIT["path"])
    verify_payload(committed_audit, PRIOR_AUDIT, "Pinned prior audit")
    local_audit_path = REPOSITORY / PRIOR_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, PRIOR_AUDIT, "Local prior audit dependency")
    require(local_audit == committed_audit, "Local prior audit differs from pinned blob")

    spec = importlib.util.spec_from_file_location(
        "wmf_pinned_retained_reset_projection", local_audit_path
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
        "Pinned prior report no longer regenerates from the pinned audit dependency",
    )
    require(regenerated == prior_report, "Parsed and regenerated prior reports differ")
    base, geometry_report, _ = prior.load_prior_package()
    return prior, base, geometry_report, {
        "commit": PRIOR_PACKAGE_COMMIT,
        "audit": PRIOR_AUDIT,
        "report": PRIOR_REPORT,
        "canonical_regeneration_verified": True,
    }


def count_key_occurrences(value: object, key: str) -> int:
    if isinstance(value, dict):
        return (1 if key in value else 0) + sum(
            count_key_occurrences(child, key) for child in value.values()
        )
    if isinstance(value, list):
        return sum(count_key_occurrences(child, key) for child in value)
    return 0


def exhaustive_aabb_inventory() -> dict:
    listing = subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "ls-tree", "-r", "-z", "-l", RESULTS_COMMIT, "--", "results/jobs",
        ],
        timeout=30,
    )
    blobs: list[tuple[str, int]] = []
    for record in listing.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        fields = metadata.decode("ascii").split()
        path = path_bytes.decode("utf-8")
        if path.endswith(".json"):
            blobs.append((path, int(fields[3])))

    parsed_bytes = 0
    occurrences: dict[str, int] = {}
    for path, declared_size in blobs:
        payload = git_bytes(RESULTS_COMMIT, path)
        require(len(payload) == declared_size, f"Git tree size differs: {path}")
        parsed_bytes += len(payload)
        decoded = strict_json_loads(payload, path)
        count = count_key_occurrences(decoded, "object_aabbs_robot_base_m")
        if count:
            occurrences[path] = count

    require(len(blobs) == 370, "Tracked results/jobs JSON blob count changed")
    require(parsed_bytes == 1925346, "Tracked results/jobs JSON bytes changed")
    require(occurrences == EXPECTED_OCCURRENCES, "Retained AABB occurrences changed")
    return {
        "commit": RESULTS_COMMIT,
        "scope": "results/jobs/**/*.json",
        "strict_json_blobs_parsed": len(blobs),
        "strict_json_bytes_parsed": parsed_bytes,
        "object_aabbs_robot_base_m_occurrences": sum(occurrences.values()),
        "occurrences_by_path": occurrences,
        "all_occurrences_confined_to_four_pinned_snapshots": True,
    }


def verify_aabb_source(base: ModuleType) -> dict:
    observations = []
    source_text: str | None = None
    fragments = (
        "corners, _ = world.get_bbox(name, env_id=0)",
        "rows = self.np.asarray([[float(value) for value in corner] for corner in corners], dtype=self.np.float64)",
        "bounds[name] = (rows.min(axis=0).tolist(), rows.max(axis=0).tolist())",
        '"object_aabbs_robot_base_m": {',
        'name: {"lower": bounds[name][0], "upper": bounds[name][1]} for name in object_names',
        '"visibility_method": "calibrated_projection_and_obb_occlusion"',
    )
    for model_id, commit in base.SOURCE_COMMITS.items():
        payload = git_bytes(commit, base.SOURCE_FILE["path"])
        verify_payload(payload, base.SOURCE_FILE, f"Pinned adapter source for {model_id}")
        text = payload.decode("utf-8")
        require(source_text is None or source_text == text, "N3/D1 adapter sources differ")
        source_text = text
        observations.append({
            "model_id": model_id,
            "source_commit": commit,
            "git_blob": base.SOURCE_FILE["git_blob"],
        })
    assert source_text is not None
    for fragment in fragments:
        require(fragment in source_text, f"AABB source fragment absent: {fragment}")
    return {
        "source_file": base.SOURCE_FILE,
        "identical_at_recorded_source_commits": observations,
        "verified_implementation": (
            "For each of banana, bowl and rubiks_cube, the pinned adapter obtains "
            "world.get_bbox corners, converts them to numeric rows, and stores the "
            "per-axis minima/maxima as a robot-base axis-aligned box before the camera "
            "loop. This audit projects only the eight corners of that retained AABB."
        ),
        "method_label_implementation_mismatch": (
            "The retained visibility_method string says calibrated_projection_and_"
            "obb_occlusion, but the implementation stores axis-aligned min/max bounds "
            "and tests only each target-center ray against the other two retained "
            "movable-object AABBs. The label is not treated as proof of OBB, silhouette, "
            "robot/static-scene, RGB or general occlusion analysis."
        ),
    }


def collapse_unique_mappings(
    prior: ModuleType, base: ModuleType, geometry_report: dict
) -> list[dict]:
    selected = prior.extract_selected_cells(base, geometry_report)
    require(len(selected) == 7, "Selected completed-cell count changed")
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in selected:
        receipt = row["receipt"]
        key = (
            receipt["model_config"],
            receipt["layout_pair_id"],
            receipt["layout_arm"],
        )
        groups.setdefault(key, []).append(row)

    output = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda row: row["receipt"]["cell_id"])
        representative = members[0]["receipt"]
        representative_cameras = representative["fresh_physical_checks"]["visibility"]["cameras"]
        representative_geometry = {
            name: representative_cameras[name]["geometry"]
            for name in base.CAMERA_NAMES
        }
        for member in members[1:]:
            cameras = member["receipt"]["fresh_physical_checks"]["visibility"]["cameras"]
            require(
                {
                    name: cameras[name]["geometry"]
                    for name in base.CAMERA_NAMES
                } == representative_geometry,
                f"Same-arm retained camera geometry differs: {key}",
            )
        output.append({
            "model_id": key[0],
            "layout_pair_id": key[1],
            "layout_arm": key[2],
            "phase": "pilot" if key[0] == "N3" else "development",
            "member_cell_ids": [member["receipt"]["cell_id"] for member in members],
            "member_commands": [member["receipt"]["command"] for member in members],
            "representative_cell_id": representative["cell_id"],
            "representative_snapshot_source": members[0]["snapshot_source"],
            "representative_snapshot_cell_key": members[0]["snapshot_cell_key"],
            "cameras": representative_cameras,
        })
    require(len(output) == 6, "Unique arm-level reset mapping count changed")
    require(
        sum(len(row["member_cell_ids"]) for row in output) == 7,
        "Collapsed member-cell count changed",
    )
    return output


def mapping_label(row: dict) -> dict:
    return {
        "model_id": row["model_id"],
        "layout_pair_id": row["layout_pair_id"],
        "layout_arm": row["layout_arm"],
        "representative_cell_id": row["representative_cell_id"],
    }


def analyze_mappings(prior: ModuleType, base: ModuleType, mappings: list[dict]) -> tuple[list[dict], dict]:
    margins: list[tuple[float, dict, str, str]] = []
    depths: list[tuple[float, dict, str, str]] = []
    widths: list[tuple[float, dict, str, str]] = []
    heights: list[tuple[float, dict, str, str]] = []
    camera_aggregates: dict[str, dict[str, list[float]]] = {
        name: {"margins": [], "depths": [], "widths": [], "heights": []}
        for name in base.CAMERA_NAMES
    }
    output = []

    for mapping in mappings:
        cameras = mapping["cameras"]
        require(tuple(cameras) == base.CAMERA_NAMES, "Retained camera order changed")
        reference_bounds: dict[str, dict] | None = None
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
            bounds = geometry["object_aabbs_robot_base_m"]
            if reference_bounds is None:
                reference_bounds = bounds
            else:
                require(bounds == reference_bounds, "AABB copies differ across cameras")

            width, height = prior.finite_vector(
                configuration["image_size_wh"], 2, "image size"
            )
            require(width == 1280.0 and height == 720.0, "Image size changed")
            origin = prior.finite_vector(
                configuration["camera_center_robot_base_m"], 3, "camera center"
            )
            quaternion = prior.finite_vector(
                configuration["camera_quaternion_world_wxyz_ros"], 4, "camera quaternion"
            )
            object_rows = {}
            for object_name in base.OBJECT_NAMES:
                lower = prior.finite_vector(bounds[object_name]["lower"], 3, "AABB lower")
                upper = prior.finite_vector(bounds[object_name]["upper"], 3, "AABB upper")
                require(
                    all(lower[index] < upper[index] for index in range(3)),
                    "Retained AABB extent is not strictly positive",
                )
                center = prior.finite_vector(
                    geometry["object_centers_robot_base_m"][object_name],
                    3,
                    "object center",
                )
                require(
                    all(lower[index] <= center[index] <= upper[index] for index in range(3)),
                    "Retained object center lies outside its AABB",
                )

                corner_uv = []
                corner_depths = []
                for corner_tuple in itertools.product(
                    (lower[0], upper[0]),
                    (lower[1], upper[1]),
                    (lower[2], upper[2]),
                ):
                    corner = list(corner_tuple)
                    local = prior.inverse_rotate_wxyz(
                        quaternion,
                        [corner[index] - origin[index] for index in range(3)],
                    )
                    require(local[2] > 1e-9, "AABB corner is behind the camera plane")
                    corner_depths.append(local[2])
                    uv = prior.project(configuration, corner)
                    require(
                        0.0 < uv[0] < width and 0.0 < uv[1] < height,
                        "AABB corner is not strictly inside the retained image frame",
                    )
                    corner_uv.append(uv)
                require(len(corner_uv) == 8, "AABB corner count changed")

                u_min = min(uv[0] for uv in corner_uv)
                u_max = max(uv[0] for uv in corner_uv)
                v_min = min(uv[1] for uv in corner_uv)
                v_max = max(uv[1] for uv in corner_uv)
                envelope_width = u_max - u_min
                envelope_height = v_max - v_min
                require(
                    envelope_width > 0.0 and envelope_height > 0.0,
                    "AABB corner envelope has a non-positive image extent",
                )
                margin = min(u_min, width - u_max, v_min, height - v_max)
                minimum_depth = min(corner_depths)
                require(margin > 0.0, "AABB corner envelope lacks strict frame support")

                stored_center_uv = prior.finite_vector(
                    geometry["projected_object_centers_uv"][object_name],
                    2,
                    "stored center projection",
                )
                require(
                    u_min <= stored_center_uv[0] <= u_max
                    and v_min <= stored_center_uv[1] <= v_max,
                    "Stored center projection lies outside AABB corner envelope",
                )
                label = mapping_label(mapping)
                margins.append((margin, label, camera_name, object_name))
                depths.append((minimum_depth, label, camera_name, object_name))
                widths.append((envelope_width, label, camera_name, object_name))
                heights.append((envelope_height, label, camera_name, object_name))
                aggregate = camera_aggregates[camera_name]
                aggregate["margins"].append(margin)
                aggregate["depths"].append(minimum_depth)
                aggregate["widths"].append(envelope_width)
                aggregate["heights"].append(envelope_height)
                object_rows[object_name] = {
                    "minimum_positive_corner_depth_m": minimum_depth,
                    "corner_envelope_uv_px": {
                        "u_min": u_min,
                        "u_max": u_max,
                        "v_min": v_min,
                        "v_max": v_max,
                        "width": envelope_width,
                        "height": envelope_height,
                    },
                    "minimum_corner_envelope_frame_margin_px": margin,
                    "all_8_corners_positive_depth_and_strictly_in_frame": True,
                }
            camera_rows.append({
                "camera_name": camera_name,
                "camera_configuration_sha256": camera["camera_configuration_sha256"],
                "all_24_corners_positive_depth_and_strictly_in_frame": True,
                "minimum_object_envelope_frame_margin_px": min(
                    row["minimum_corner_envelope_frame_margin_px"]
                    for row in object_rows.values()
                ),
                "minimum_positive_corner_depth_m": min(
                    row["minimum_positive_corner_depth_m"]
                    for row in object_rows.values()
                ),
                "maximum_object_envelope_width_px": max(
                    row["corner_envelope_uv_px"]["width"]
                    for row in object_rows.values()
                ),
                "maximum_object_envelope_height_px": max(
                    row["corner_envelope_uv_px"]["height"]
                    for row in object_rows.values()
                ),
            })
        output.append({
            key: value for key, value in mapping.items() if key != "cameras"
        } | {
            "same_aabb_mapping_copied_across_four_cameras": True,
            "camera_rows": camera_rows,
        })

    minimum_margin = min(margins, key=lambda row: row[0])
    minimum_depth = min(depths, key=lambda row: row[0])
    maximum_width = max(widths, key=lambda row: row[0])
    maximum_height = max(heights, key=lambda row: row[0])
    require(
        minimum_margin[0] == EXPECTED_EXTREMA["minimum_corner_frame_margin_px"],
        "Minimum AABB corner-envelope frame margin changed",
    )
    require(
        minimum_depth[0] == EXPECTED_EXTREMA["minimum_positive_corner_depth_m"],
        "Minimum positive AABB corner depth changed",
    )
    require(
        maximum_width[0] == EXPECTED_EXTREMA["maximum_envelope_width_px"],
        "Maximum AABB corner-envelope width changed",
    )
    require(
        maximum_height[0] == EXPECTED_EXTREMA["maximum_envelope_height_px"],
        "Maximum AABB corner-envelope height changed",
    )

    def extremum(row: tuple[float, dict, str, str], unit: str) -> dict:
        return {
            unit: row[0],
            **row[1],
            "camera_name": row[2],
            "object_name": row[3],
        }

    by_camera = {}
    for camera_name in base.CAMERA_NAMES:
        aggregate = camera_aggregates[camera_name]
        require(all(len(values) == 18 for values in aggregate.values()), "Camera aggregate count changed")
        by_camera[camera_name] = {
            "unique_mapping_object_envelopes": 18,
            "projected_aabb_corners": 144,
            "minimum_corner_envelope_frame_margin_px": min(aggregate["margins"]),
            "minimum_positive_corner_depth_m": min(aggregate["depths"]),
            "maximum_corner_envelope_width_px": max(aggregate["widths"]),
            "maximum_corner_envelope_height_px": max(aggregate["heights"]),
            "all_corners_positive_depth_and_strictly_in_frame": True,
        }
    return output, {
        "unique_arm_level_reset_mappings": len(output),
        "camera_rows": len(output) * len(base.CAMERA_NAMES),
        "mapping_camera_object_envelopes": len(margins),
        "projected_aabb_corners": len(margins) * 8,
        "retained_centers_inside_their_aabbs": len(margins),
        "center_projections_inside_corner_envelopes": len(margins),
        "positive_depth_corner_projections": len(margins) * 8,
        "strictly_in_frame_corner_projections": len(margins) * 8,
        "fully_frame_supported_corner_envelopes": len(margins),
        "camera_summary": by_camera,
        "minimum_corner_envelope_frame_margin": extremum(minimum_margin, "pixels"),
        "minimum_positive_corner_depth": extremum(minimum_depth, "meters"),
        "maximum_corner_envelope_width": extremum(maximum_width, "pixels"),
        "maximum_corner_envelope_height": extremum(maximum_height, "pixels"),
    }


def audit() -> dict:
    prior, base, geometry_report, prior_pin = load_prior_package()
    require(base.RESULTS_COMMIT == RESULTS_COMMIT, "Prior audit results pin changed")
    inventory = exhaustive_aabb_inventory()
    source_semantics = verify_aabb_source(base)
    mappings = collapse_unique_mappings(prior, base, geometry_report)
    rows, summary = analyze_mappings(prior, base, mappings)
    return {
        "schema_version": "wmf-retained-aabb-frame-support-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "prior_reset_projection_package": prior_pin,
        "exhaustive_results_inventory": inventory,
        "source_semantics": source_semantics,
        "selection": {
            "terminal_matched_unique_cells": 7,
            "unique_arm_level_reset_mappings": 6,
            "mapping_camera_rows": 24,
            "mapping_camera_object_envelopes": 72,
            "projected_aabb_corners": 576,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
        },
        "summary": summary,
        "mappings": rows,
        "finding": (
            "After collapsing seven terminal-matched retained cells to six unique "
            "arm-level reset mappings, all 576 projections of the eight corners of "
            "72 world.get_bbox-derived axis-aligned boxes have positive camera depth "
            "and lie strictly inside their retained 1280x720 frames. The smallest "
            "corner-envelope boundary margin is 46.21332507478144 pixels. This is "
            "only a retained-geometric AABB-corner frame-support result."
        ),
        "claim_boundaries": {
            "aabb_not_object_silhouette": (
                "The audited shape is the per-axis min/max box derived from retained "
                "world.get_bbox corners. It may be loose and is not an exact mesh, "
                "oriented box, segmentation mask, image footprint or object silhouette."
            ),
            "frame_support_not_visibility": (
                "Positive depth and strict frame inclusion of these AABB corners do not "
                "authenticate RGB bytes, image content, object identity, focus, lighting, "
                "silhouette visibility, crop mapping, camera-time alignment or human "
                "labelability. No measurement-feasibility qualification follows."
            ),
            "occlusion_scope": (
                "The pinned implementation tests only target-center rays against the "
                "other two retained movable-object axis-aligned boxes. It does not test "
                "the projected AABB envelopes for occlusion and does not include the "
                "robot or static scene geometry. No general unoccluded claim follows."
            ),
            "stage_and_selection": (
                "Only reset-stage simulator geometry in six availability-selected unique "
                "mappings from seven prior-authenticated embedded receipts is analyzed. "
                "These are not executed-motion endpoints, generated forecasts, observed "
                "centroids, a representative sample or the full completed cohort."
            ),
            "science_and_release": (
                "No fixture/camera/crop/time qualification, scientific acceptance "
                "threshold, A1-A9 result, forecast/persistence metric, task success, "
                "model comparison, label, current-state inference, gate promotion, "
                "dispatch/release authority or new behavioral evidence follows."
            ),
        },
        "evidence_boundaries": {
            "raw_camera_bytes_read": False,
            "raw_pvc_bytes_read": False,
            "rgb_or_image_content_analyzed": False,
            "silhouette_or_segmentation_analyzed": False,
            "robot_or_static_scene_occlusion_analyzed": False,
            "camera_crop_or_time_alignment_authenticated": False,
            "human_image_centroids_used": False,
            "cross_time_field_join_performed": False,
            "scientific_threshold_applied": False,
            "fixture_pass_fail_classification_created": False,
            "measurement_feasibility_qualified": False,
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
            "Saved retained AABB frame-support diagnostic differs",
        )
        print(
            "Verified 370 strict results JSON blobs, 6 unique retained reset "
            "mappings, 72 AABB corner envelopes and 576 positive-depth in-frame "
            "corner projections; no RGB evidence or gate promotion."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
