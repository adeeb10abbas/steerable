#!/usr/bin/env python3
"""Audit retained reset-stage camera projections without promoting science gates."""

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
PRIOR_PACKAGE_COMMIT = "d5656cb12dcd1a8a5160595dcfeee3fdc0f92c4e"
REPOSITORY = Path(__file__).resolve().parents[3]
PRIOR_AUDIT = {
    "path": (
        "workshops/corl2026_world_models/analysis/"
        "audit_retained_reset_geometry.py"
    ),
    "bytes": 54871,
    "sha256": "c84ebb0b52c0e2c0234d6e3766fcd8a8fdc54fbdcdd7c956eda99b09a62164fa",
}
PRIOR_REPORT = {
    "path": (
        "workshops/corl2026_world_models/results/"
        "retained_reset_geometry_diagnostic.json"
    ),
    "bytes": 44781,
    "sha256": "94777db36b789828f72dfbd8b10347a4451dd95f020f3ca29d052f9c4fa407de",
}
EXPECTED_OCCURRENCES = {
    "results/jobs/d1-development-d01-snapshot-001/publish/diagnostic.json": {
        "projected_object_centers_uv": 16,
        "projected_unoccluded_by_object": 16,
        "camera_configuration": 8,
        "object_centers_robot_base_m": 8,
    },
    "results/jobs/d1-development-d02-snapshot-001/publish/diagnostic.json": {
        "projected_object_centers_uv": 16,
        "projected_unoccluded_by_object": 16,
        "camera_configuration": 8,
        "object_centers_robot_base_m": 8,
    },
    "results/jobs/detached-attempt-snapshot-002/publish/diagnostic.json": {
        "projected_object_centers_uv": 8,
        "projected_unoccluded_by_object": 8,
        "camera_configuration": 4,
        "object_centers_robot_base_m": 4,
    },
    "results/jobs/detached-attempt-snapshot-003/publish/diagnostic.json": {
        "projected_object_centers_uv": 24,
        "projected_unoccluded_by_object": 24,
        "camera_configuration": 12,
        "object_centers_robot_base_m": 12,
    },
}
EXPECTED_TOTALS = {
    "projected_object_centers_uv": 64,
    "projected_unoccluded_by_object": 64,
    "camera_configuration": 32,
    "object_centers_robot_base_m": 32,
}
EXPECTED_CAMERA_CONFIGURATION_SHA256 = {
    "head_camera": "6d43b168602cb25d52ac444cbc0213f0e46f53270d386747dfa34e38f2a05192",
    "over_shoulder_left_camera": "8fbccf8b6dad2608560da9698389795642a2229ae9d6aa930c2fa097c81e403b",
    "over_shoulder_right_camera": "2430537c4923ed0a05f06f08e3af12ba6b95951944bb4329306776d543e8cd39",
    "wrist_cam": "ddc5005188ee810a90c154f9bf69e7b908e3e43c1b9f2b700821a24e68ec1b11",
}
EXPECTED_EXTREMA = {
    "minimum_projected_center_frame_boundary_margin_px": 132.92674114727595,
    "minimum_object_displacement_l2_px": 0.00022944119169895038,
    "maximum_object_displacement_l2_px": 577.2646288861049,
    "minimum_relation_delta_l2_px": 2.8008314195758763,
    "maximum_relation_delta_l2_px": 64.72490043231407,
}
COMPARISONS = (
    (
        "N3", "P00",
        "wmf1__pilot__P00__N3__original__left",
        "wmf1__pilot__P00__N3__reflected__left",
        False,
    ),
    (
        "D1", "D01",
        "wmf1__development__D01__D1__original__left",
        "wmf1__development__D01__D1__reflected__right",
        True,
    ),
    (
        "D1", "D02",
        "wmf1__development__D02__D1__original__left",
        "wmf1__development__D02__D1__reflected__right",
        True,
    ),
)


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


def load_prior_package() -> tuple[ModuleType, dict, dict]:
    committed_audit = git_bytes(PRIOR_PACKAGE_COMMIT, PRIOR_AUDIT["path"])
    verify_payload(committed_audit, PRIOR_AUDIT, "Pinned prior audit")
    local_audit_path = REPOSITORY / PRIOR_AUDIT["path"]
    local_audit = local_audit_path.read_bytes()
    verify_payload(local_audit, PRIOR_AUDIT, "Local prior audit dependency")
    require(local_audit == committed_audit, "Local prior audit differs from pinned blob")

    spec = importlib.util.spec_from_file_location(
        "wmf_pinned_retained_reset_geometry", local_audit_path
    )
    require(spec is not None and spec.loader is not None, "Cannot load prior audit")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    prior_payload = git_bytes(PRIOR_PACKAGE_COMMIT, PRIOR_REPORT["path"])
    verify_payload(prior_payload, PRIOR_REPORT, "Pinned prior report")
    prior_report = strict_json_loads(prior_payload, PRIOR_REPORT["path"])
    regenerated = module.audit()
    require(
        module.canonical_bytes(regenerated) == prior_payload,
        "Pinned prior report no longer regenerates from the pinned audit dependency",
    )
    require(
        regenerated == prior_report,
        "Strictly parsed prior report differs from regenerated prior report",
    )
    return module, prior_report, {
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


def exhaustive_projection_inventory() -> dict:
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

    totals = {key: 0 for key in EXPECTED_TOTALS}
    occurrences_by_path: dict[str, dict[str, int]] = {}
    parsed_bytes = 0
    for path, declared_size in blobs:
        payload = git_bytes(RESULTS_COMMIT, path)
        require(len(payload) == declared_size, f"Git tree size differs: {path}")
        parsed_bytes += len(payload)
        decoded = strict_json_loads(payload, path)
        row = {
            key: count_key_occurrences(decoded, key)
            for key in EXPECTED_TOTALS
        }
        for key, count in row.items():
            totals[key] += count
        if any(row.values()):
            occurrences_by_path[path] = row

    require(len(blobs) == 370, "Tracked results/jobs JSON blob count changed")
    require(parsed_bytes == 1925346, "Tracked results/jobs JSON bytes changed")
    require(totals == EXPECTED_TOTALS, "Projection occurrence totals changed")
    require(
        occurrences_by_path == EXPECTED_OCCURRENCES,
        "Projection occurrences moved outside the four pinned snapshots",
    )
    return {
        "commit": RESULTS_COMMIT,
        "scope": "results/jobs/**/*.json",
        "strict_json_blobs_parsed": len(blobs),
        "strict_json_bytes_parsed": parsed_bytes,
        "occurrence_totals": totals,
        "occurrences_by_path": occurrences_by_path,
        "all_occurrences_confined_to_four_pinned_snapshots": True,
    }


def finite_vector(value: object, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{label} shape changed")
    output = []
    for item in value:
        require(type(item) in (int, float) and math.isfinite(item), f"{label} is non-finite")
        output.append(float(item))
    return output


def inverse_rotate_wxyz(quaternion: list[float], vector: list[float]) -> tuple[float, float, float]:
    w, x, y, z = quaternion
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    require(norm > 0.0, "Camera quaternion has zero norm")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = vector
    return (
        (1 - 2 * (y * y + z * z)) * vx
        + 2 * (x * y + w * z) * vy
        + 2 * (x * z - w * y) * vz,
        2 * (x * y - w * z) * vx
        + (1 - 2 * (x * x + z * z)) * vy
        + 2 * (y * z + w * x) * vz,
        2 * (x * z + w * y) * vx
        + 2 * (y * z - w * x) * vy
        + (1 - 2 * (x * x + y * y)) * vz,
    )


def project(configuration: dict, target: list[float]) -> list[float]:
    center = finite_vector(
        configuration["camera_center_robot_base_m"], 3, "camera center"
    )
    quaternion = finite_vector(
        configuration["camera_quaternion_world_wxyz_ros"], 4, "camera quaternion"
    )
    intrinsic = configuration["intrinsic_matrix_3x3"]
    require(
        isinstance(intrinsic, list)
        and len(intrinsic) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in intrinsic),
        "Camera intrinsic matrix shape changed",
    )
    matrix = [finite_vector(row, 3, "camera intrinsic row") for row in intrinsic]
    local = inverse_rotate_wxyz(
        quaternion, [target[index] - center[index] for index in range(3)]
    )
    require(local[2] > 1e-9, "Object center is behind the camera plane")
    return [
        matrix[0][0] * local[0] / local[2]
        + matrix[0][1] * local[1] / local[2]
        + matrix[0][2],
        matrix[1][0] * local[0] / local[2]
        + matrix[1][1] * local[1] / local[2]
        + matrix[1][2],
    ]


def segment_intersects_aabb_before_target(
    origin: list[float], target: list[float], lower: list[float], upper: list[float]
) -> bool:
    direction = [target[index] - origin[index] for index in range(3)]
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - origin[axis]) / direction[axis]
        second = (upper[axis] - origin[axis]) / direction[axis]
        t_min = max(t_min, min(first, second))
        t_max = min(t_max, max(first, second))
        if t_min > t_max:
            return False
    return t_max > 1e-6 and t_min < 1.0 - 1e-6


def verify_projection_source(base: ModuleType) -> dict:
    observations = []
    source_text: str | None = None
    fragments = (
        "def _quat_inverse_rotate_wxyz(",
        "def _project(",
        "_require(local[2] > 1e-9, \"object center is behind the camera plane\")",
        "def _segment_intersects_aabb_before_target(",
        "in_frame = 0.0 <= uv[0] < frame.shape[1] and 0.0 <= uv[1] < frame.shape[0]",
        "unoccluded[name] = bool(in_frame and not blocked)",
        '"projected_object_centers_uv": projections,',
        '"projected_unoccluded_by_object": unoccluded,',
    )
    for model_id, commit in base.SOURCE_COMMITS.items():
        payload = git_bytes(commit, base.SOURCE_FILE["path"])
        require(
            len(payload) == base.SOURCE_FILE["bytes"]
            and hashlib.sha256(payload).hexdigest() == base.SOURCE_FILE["sha256"],
            f"Pinned projection source differs: {model_id}",
        )
        text = payload.decode("utf-8")
        require(source_text is None or source_text == text, "N3/D1 projection sources differ")
        source_text = text
        observations.append({
            "model_id": model_id,
            "source_commit": commit,
            "git_blob": base.SOURCE_FILE["git_blob"],
        })
    assert source_text is not None
    for fragment in fragments:
        require(fragment in source_text, f"Projection source fragment absent: {fragment}")
    return {
        "source_file": base.SOURCE_FILE,
        "identical_at_recorded_source_commits": observations,
        "verified_semantics": (
            "Pinned source inverse-rotates robot-base object centers into each camera, "
            "applies the retained intrinsic matrix, requires positive camera depth, "
            "checks strict image bounds, and marks a projection unoccluded only when "
            "no other retained object AABB intersects the camera-to-target segment."
        ),
    }


def extract_selected_cells(base: ModuleType, prior_report: dict) -> list[dict]:
    values = {
        name: base.read_blob(*descriptor)[0]
        for name, descriptor in base.PINNED_BLOBS.items()
    }
    expected_ids = [row["cell_id"] for row in prior_report["cells"]]
    cells: list[tuple[str, str, dict]] = []

    n3_cells = values["N3_snapshot_003"]["n3_attempts"][base.N3_ATTEMPT_ID]["cells"]
    for key in base.N3_KEYS:
        cell = n3_cells[key]["cell_receipt.json"]
        require(isinstance(cell, dict), f"N3 embedded receipt absent: {key}")
        cells.append(("N3_snapshot_003", key, cell))
    for pair in ("D01", "D02"):
        wrappers = values[f"{pair}_snapshot"]["simulator"]["cells"]
        for key in base.D1_KEYS[pair]:
            cell = wrappers[key]["cell_receipt"]
            require(isinstance(cell, dict), f"D1 embedded receipt absent: {key}")
            cells.append((f"{pair}_snapshot", key, cell))

    cells.sort(key=lambda item: item[2]["cell_id"])
    require(
        [cell["cell_id"] for _, _, cell in cells] == expected_ids,
        "Selected cell order/identity differs from prior authenticated report",
    )
    return [
        {"snapshot_source": source, "snapshot_cell_key": key, "receipt": cell}
        for source, key, cell in cells
    ]


def analyze_projection_rows(base: ModuleType, selected: list[dict]) -> tuple[list[dict], dict]:
    rows = []
    configurations: dict[str, dict] = {}
    all_residuals: list[float] = []
    margins: list[tuple[float, str, str, str]] = []
    total_blockers = 0
    for selected_row in selected:
        cell = selected_row["receipt"]
        cameras = cell["fresh_physical_checks"]["visibility"]["cameras"]
        require(tuple(cameras) == base.CAMERA_NAMES, f"Camera order differs: {cell['cell_id']}")
        cell_rows = []
        for camera_name in base.CAMERA_NAMES:
            camera = cameras[camera_name]
            configuration = camera["camera_configuration"]
            configuration_sha256 = camera["camera_configuration_sha256"]
            require(
                configuration_sha256 == EXPECTED_CAMERA_CONFIGURATION_SHA256[camera_name],
                f"Camera configuration hash differs: {camera_name}",
            )
            if camera_name in configurations:
                require(
                    configurations[camera_name]["configuration"] == configuration,
                    f"Camera configuration varies across retained cells: {camera_name}",
                )
            else:
                configurations[camera_name] = {
                    "configuration_sha256": configuration_sha256,
                    "configuration": configuration,
                }

            geometry = camera["geometry"]
            require(
                camera["projected_object_centers_uv"]
                == geometry["projected_object_centers_uv"],
                f"Top-level/nested projections differ: {cell['cell_id']}/{camera_name}",
            )
            require(
                camera["projected_unoccluded_by_object"]
                == geometry["projected_unoccluded_by_object"],
                f"Top-level/nested visibility flags differ: {cell['cell_id']}/{camera_name}",
            )
            require(
                all(configuration[key] == geometry[key] for key in configuration),
                f"Top-level/nested camera configuration differs: {cell['cell_id']}/{camera_name}",
            )
            width, height = finite_vector(configuration["image_size_wh"], 2, "image size")
            require(width == 1280.0 and height == 720.0, "Retained image size changed")
            origin = finite_vector(
                configuration["camera_center_robot_base_m"], 3, "camera center"
            )
            projections = {}
            for object_name in base.OBJECT_NAMES:
                target = finite_vector(
                    geometry["object_centers_robot_base_m"][object_name],
                    3,
                    f"{object_name} center",
                )
                stored = finite_vector(
                    geometry["projected_object_centers_uv"][object_name],
                    2,
                    f"{object_name} projection",
                )
                recomputed = project(configuration, target)
                residual = [abs(recomputed[index] - stored[index]) for index in range(2)]
                require(residual == [0.0, 0.0], "Projection does not exactly regenerate")
                all_residuals.extend(residual)
                in_frame = 0.0 <= stored[0] < width and 0.0 <= stored[1] < height
                margin = min(stored[0], width - stored[0], stored[1], height - stored[1])
                bounds = geometry["object_aabbs_robot_base_m"]
                blockers = []
                for other_name in base.OBJECT_NAMES:
                    if other_name == object_name:
                        continue
                    lower = finite_vector(bounds[other_name]["lower"], 3, "AABB lower")
                    upper = finite_vector(bounds[other_name]["upper"], 3, "AABB upper")
                    if segment_intersects_aabb_before_target(origin, target, lower, upper):
                        blockers.append(other_name)
                recomputed_unoccluded = in_frame and not blockers
                stored_unoccluded = geometry["projected_unoccluded_by_object"][object_name]
                require(
                    type(stored_unoccluded) is bool
                    and stored_unoccluded is recomputed_unoccluded
                    and stored_unoccluded is True,
                    "Stored/recomputed projected-unoccluded flag differs",
                )
                total_blockers += len(blockers)
                margins.append((margin, cell["cell_id"], camera_name, object_name))
                projections[object_name] = {
                    "stored_uv_px": stored,
                    "recomputed_uv_px": recomputed,
                    "absolute_residual_px": residual,
                    "strictly_in_frame": in_frame,
                    "frame_boundary_margin_px": margin,
                    "recomputed_blocking_objects": blockers,
                    "stored_and_recomputed_unoccluded": True,
                }
            cell_rows.append({
                "camera_name": camera_name,
                "camera_configuration_sha256": configuration_sha256,
                "projections": projections,
            })
        rows.append({
            "cell_id": cell["cell_id"],
            "model_id": cell["model_config"],
            "phase": "pilot" if cell["model_config"] == "N3" else "development",
            "layout_pair_id": cell["layout_pair_id"],
            "layout_arm": cell["layout_arm"],
            "command": cell["command"],
            "snapshot_source": selected_row["snapshot_source"],
            "snapshot_cell_key": selected_row["snapshot_cell_key"],
            "camera_rows": cell_rows,
        })

    minimum_margin = min(margins)
    require(
        minimum_margin[0]
        == EXPECTED_EXTREMA["minimum_projected_center_frame_boundary_margin_px"],
        "Minimum projected-center frame margin changed",
    )
    require(max(all_residuals) == 0.0, "Maximum reprojection residual changed")
    require(total_blockers == 0, "Retained AABB blocker count changed")
    return rows, {
        "selected_cells": len(rows),
        "selected_cell_camera_rows": len(rows) * len(base.CAMERA_NAMES),
        "selected_cell_camera_object_projections": (
            len(rows) * len(base.CAMERA_NAMES) * len(base.OBJECT_NAMES)
        ),
        "unique_camera_configurations": len(configurations),
        "camera_configurations": configurations,
        "top_level_nested_projection_maps_exactly_equal": True,
        "source_formula_maximum_absolute_reprojection_residual_px": max(all_residuals),
        "recomputed_aabb_blockers": total_blockers,
        "stored_and_recomputed_unoccluded_true_projections": len(rows) * 12,
        "minimum_projected_center_frame_boundary_margin": {
            "pixels": minimum_margin[0],
            "cell_id": minimum_margin[1],
            "camera_name": minimum_margin[2],
            "object_name": minimum_margin[3],
        },
    }


def projection_map(rows: list[dict]) -> dict[str, dict[str, dict[str, list[float]]]]:
    output = {}
    for row in rows:
        output[row["cell_id"]] = {
            camera["camera_name"]: {
                object_name: projection["stored_uv_px"]
                for object_name, projection in camera["projections"].items()
            }
            for camera in row["camera_rows"]
        }
    return output


def compact_projection_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        compact_cameras = []
        for camera in row["camera_rows"]:
            projections = camera["projections"]
            compact_cameras.append({
                "camera_name": camera["camera_name"],
                "camera_configuration_sha256": camera["camera_configuration_sha256"],
                "projected_object_centers_uv_px": {
                    name: value["stored_uv_px"] for name, value in projections.items()
                },
                "minimum_projected_center_frame_boundary_margin_px": min(
                    value["frame_boundary_margin_px"] for value in projections.values()
                ),
                "all_three_source_formula_reprojections_exact": all(
                    value["absolute_residual_px"] == [0.0, 0.0]
                    for value in projections.values()
                ),
                "all_three_strictly_in_frame": all(
                    value["strictly_in_frame"] for value in projections.values()
                ),
                "all_three_recomputed_blocker_lists_empty": all(
                    not value["recomputed_blocking_objects"]
                    for value in projections.values()
                ),
                "all_three_stored_and_recomputed_unoccluded": all(
                    value["stored_and_recomputed_unoccluded"]
                    for value in projections.values()
                ),
            })
        output.append({
            key: value for key, value in row.items() if key != "camera_rows"
        } | {"camera_rows": compact_cameras})
    return output


def vector_delta(original: list[float], reflected: list[float]) -> dict:
    delta = [reflected[index] - original[index] for index in range(2)]
    return {
        "signed_reflected_minus_original_uv_px": delta,
        "l2_px": math.hypot(delta[0], delta[1]),
    }


def compare_reset_mappings(base: ModuleType, rows: list[dict]) -> tuple[list[dict], dict]:
    projections = projection_map(rows)
    comparisons = []
    object_extrema: list[tuple[float, str, str, str]] = []
    relation_extrema: list[tuple[float, str, str]] = []
    for model_id, pair, original_id, reflected_id, command_confounded in COMPARISONS:
        camera_rows = []
        for camera_name in base.CAMERA_NAMES:
            object_rows = {}
            for object_name in base.OBJECT_NAMES:
                metric = vector_delta(
                    projections[original_id][camera_name][object_name],
                    projections[reflected_id][camera_name][object_name],
                )
                object_extrema.append((metric["l2_px"], pair, camera_name, object_name))
                object_rows[object_name] = {
                    "original_uv_px": projections[original_id][camera_name][object_name],
                    "reflected_uv_px": projections[reflected_id][camera_name][object_name],
                    **metric,
                }
            original_relation = [
                projections[original_id][camera_name]["rubiks_cube"][index]
                - projections[original_id][camera_name]["bowl"][index]
                for index in range(2)
            ]
            reflected_relation = [
                projections[reflected_id][camera_name]["rubiks_cube"][index]
                - projections[reflected_id][camera_name]["bowl"][index]
                for index in range(2)
            ]
            relation_metric = vector_delta(original_relation, reflected_relation)
            relation_extrema.append((relation_metric["l2_px"], pair, camera_name))
            camera_rows.append({
                "camera_name": camera_name,
                "object_center_displacements": object_rows,
                "geometric_cube_minus_bowl_relation": {
                    "original_uv_px": original_relation,
                    "reflected_uv_px": reflected_relation,
                    **relation_metric,
                    "not_primary_endpoint_r0": True,
                },
            })
        comparisons.append({
            "model_id": model_id,
            "layout_pair_id": pair,
            "original_cell_id": original_id,
            "reflected_cell_id": reflected_id,
            "command_confounded": command_confounded,
            "camera_rows": camera_rows,
            "threshold_applied": False,
            "pass_fail_classification": None,
        })

    minimum_object = min(object_extrema)
    maximum_object = max(object_extrema)
    minimum_relation = min(relation_extrema)
    maximum_relation = max(relation_extrema)
    require(
        minimum_object[0] == EXPECTED_EXTREMA["minimum_object_displacement_l2_px"]
        and maximum_object[0] == EXPECTED_EXTREMA["maximum_object_displacement_l2_px"],
        "Object displacement extrema changed",
    )
    require(
        minimum_relation[0] == EXPECTED_EXTREMA["minimum_relation_delta_l2_px"]
        and maximum_relation[0] == EXPECTED_EXTREMA["maximum_relation_delta_l2_px"],
        "Geometric relation-delta extrema changed",
    )
    return comparisons, {
        "original_reflected_mapping_comparisons": len(comparisons),
        "object_displacement_vectors": len(object_extrema),
        "geometric_cube_minus_bowl_relation_delta_vectors": len(relation_extrema),
        "minimum_object_displacement": {
            "l2_px": minimum_object[0],
            "layout_pair_id": minimum_object[1],
            "camera_name": minimum_object[2],
            "object_name": minimum_object[3],
        },
        "maximum_object_displacement": {
            "l2_px": maximum_object[0],
            "layout_pair_id": maximum_object[1],
            "camera_name": maximum_object[2],
            "object_name": maximum_object[3],
        },
        "minimum_geometric_relation_delta": {
            "l2_px": minimum_relation[0],
            "layout_pair_id": minimum_relation[1],
            "camera_name": minimum_relation[2],
        },
        "maximum_geometric_relation_delta": {
            "l2_px": maximum_relation[0],
            "layout_pair_id": maximum_relation[1],
            "camera_name": maximum_relation[2],
        },
    }


def audit() -> dict:
    base, prior_report, prior_pin = load_prior_package()
    require(base.RESULTS_COMMIT == RESULTS_COMMIT, "Prior audit results pin changed")
    inventory = exhaustive_projection_inventory()
    source_semantics = verify_projection_source(base)
    selected = extract_selected_cells(base, prior_report)
    rows, projection_summary = analyze_projection_rows(base, selected)
    comparisons, comparison_summary = compare_reset_mappings(base, rows)

    require(len(rows) == 7, "Selected completed-cell count changed")
    require(prior_report["duplicate_collapse"]["embedded_receipt_appearances"] == 8,
            "Prior receipt appearance count changed")
    require(prior_report["duplicate_collapse"]["unique_completed_cells"] == 7,
            "Prior unique completed-cell count changed")
    require(prior_report["duplicate_collapse"]["unique_arm_level_center_mappings"] == 6,
            "Prior unique reset-mapping count changed")
    return {
        "schema_version": "wmf-retained-reset-projection-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "prior_reset_geometry_package": prior_pin,
        "exhaustive_results_inventory": inventory,
        "source_semantics": source_semantics,
        "selection": {
            "completed_receipt_appearances": 8,
            "terminal_matched_unique_cells": 7,
            "unique_reset_mappings": 6,
            "selected_cell_camera_rows": 28,
            "selected_cell_camera_object_projections": 84,
            "unique_reset_mapping_camera_rows": 24,
            "unique_reset_mapping_camera_object_projections": 72,
            "original_reflected_mapping_comparisons": 3,
            "object_displacement_vectors": 36,
            "geometric_cube_minus_bowl_relation_delta_vectors": 12,
            "availability_selected": True,
            "representative_sample": False,
            "full_completed_cohort": False,
            "cross_time_field_join_performed": False,
        },
        "projection_summary": projection_summary,
        "comparison_summary": comparison_summary,
        "cells": compact_projection_rows(rows),
        "mirror_comparisons": comparisons,
        "finding": (
            "Across seven availability-selected terminal-matched retained cells, all "
            "84 stored reset-stage geometric object-center projections regenerate "
            "exactly from four invariant camera configurations and the pinned source "
            "formula. All 84 stored projected-unoccluded booleans equal the recomputed "
            "conjunction of strict in-frame and no retained-object AABB blocker; the "
            "recomputation separately finds all 84 points strictly in-frame and "
            "blocker-free. Three descriptive original/reflected reset-map comparisons yield "
            "36 object displacement vectors and 12 geometric cube-minus-bowl relation "
            "deltas; these are not observed image centroids or paper endpoints."
        ),
        "claim_boundaries": {
            "stage": (
                "Only fresh_physical_checks reset-stage simulator geometry retained in "
                "the seven prior-authenticated embedded receipts is analyzed. The prior "
                "audit's stage-order boundary remains unchanged."
            ),
            "projection_not_image_evidence": (
                "Stored robot-base centers, camera calibration and AABBs are used. No "
                "RGB artifact path is opened, read, decoded or rehashed; projected "
                "geometric centers and center-to-frame margins do not establish full "
                "object visibility or raw artifact availability/integrity."
            ),
            "not_primary_endpoint": (
                "The geometric cube-minus-bowl vectors are not human-marked observed "
                "image centroids and have no authenticated primary-camera crop, request "
                "start, generated horizon or executed-time correspondence. They are not "
                "r0, rhat or r and cannot enter forecast, persistence or skill metrics."
            ),
            "availability_and_confounding": (
                "Exactly three availability-selected original/reflected comparisons are "
                "described. D01 and D02 compare original-LEFT with reflected-RIGHT and "
                "remain command-confounded. No command, layout, model or population "
                "effect is estimated."
            ),
            "science_and_release": (
                "No threshold or pass/fail classification, A1-A9 result, fixture/camera "
                "qualification, task success, model rate/comparison, annotation label, "
                "current GM/PVC inference, gate promotion, dispatch/release authority or "
                "new behavioral evidence follows."
            ),
        },
        "evidence_boundaries": {
            "raw_camera_bytes_read": False,
            "raw_pvc_bytes_read": False,
            "human_image_centroids_used": False,
            "camera_crop_or_time_alignment_authenticated": False,
            "cross_time_field_join_performed": False,
            "threshold_applied": False,
            "pass_fail_classification_created": False,
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
        require(args.check.read_bytes() == encoded,
                "Saved retained reset-projection diagnostic differs")
        print(
            "Verified 370 strict results JSON blobs, 7 terminal-matched retained "
            "cells, 84 exact reset projections and 36 descriptive displacement "
            "vectors; no raw image or gate promotion."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
