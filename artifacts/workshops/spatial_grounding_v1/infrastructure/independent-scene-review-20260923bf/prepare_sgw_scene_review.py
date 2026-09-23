"""Verify an exported native capture and prepare unmodified-pixel review images."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


SCENES = {
    "0-height-left": "20c16465cb215e5458b14e8ec68957eae17b6c609bb0d442af38205e940e9f30",
    "1-height-right": "56c92caa5d7b79de915c237e0dddb15ee5d5e84b4cde13dbc9946a2c2a026b71",
    "2-dist-left": "e5e149208b93e33400e7db47f52c8711f688c7cf32cbeb201b1fdeffae7745fd",
    "3-dist-right": "d7566dd505996d0c658a7f0e0514bf1f55a8d71be1e7f5db1ec0bb1eac0821c9",
}
CAMERAS = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
SAMPLES = (0, 1, 10, 30, 60, 120)
SOURCE = "fccf310c41fa16e8dc703491559c933558f08d96"
ROBOLAB = "0aef241fb088ca21bb4ebd24448940ed56620d17"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_digest(value, field):
    material = {key: item for key, item in value.items() if key != field}
    raw = (json.dumps(material, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def rotation(quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def sheet(path, entries, columns, width=480):
    height = width * 9 // 16
    canvas = Image.new("RGB", (columns * width, ((len(entries) + columns - 1) // columns) * (height + 26)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(entries):
        x, y = (index % columns) * width, (index // columns) * (height + 26)
        draw.text((x + 5, y + 5), label, fill="black")
        canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), (x, y + 26))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.evidence.resolve(), args.output.resolve()
    output.mkdir(exist_ok=False)
    export_path = root / "export_manifest.json"
    export = json.loads(export_path.read_text())
    assert len(export["files"]) == 118
    source_map = {}
    for row in export["files"]:
        path = root / row["export_path"]
        assert path.resolve().is_relative_to(root)
        assert path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"], path
        assert row["source_path"] not in source_map, row["source_path"]
        source_map[row["source_path"]] = row

    def verify_record(record, path_key="path"):
        item = source_map.get(record[path_key])
        if item is None:
            return False
        assert item["sha256"] == record["sha256"], record[path_key]
        if "bytes" in record:
            assert item["bytes"] == record["bytes"], record[path_key]
        return True

    analysis = {
        "export_manifest_sha256": sha256(export_path),
        "verified_exported_files": len(export["files"]),
        "study_source_commit": SOURCE,
        "robolab_commit": ROBOLAB,
        "claim_boundary": "Read-only integrity and geometry checks, not visual verdicts or physical qualification.",
        "scenes": {},
    }
    media_index = []
    for scene, expected in SCENES.items():
        directory = root / scene
        capture_path, manifest_path = directory / "capture.json", directory / "scene/manifest.json"
        assert sha256(capture_path) == expected
        capture = json.loads(capture_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        assert capture["receipt_sha256"] == content_digest(capture, "receipt_sha256")
        assert manifest["manifest_sha256"] == content_digest(manifest, "manifest_sha256")
        assert capture["overlay_manifest_sha256"] == manifest["manifest_sha256"]
        assert capture["study_source_commit"] == SOURCE and capture["robolab_commit"] == ROBOLAB
        assert capture["model_request_count"] == capture["behavioral_episode_count"] == 0
        assert capture["validated_slots"] == []
        assert verify_record(capture["overlay_manifest"])
        assert verify_record(manifest["overlay_usda"])
        assert verify_record(manifest["base_scene"])
        assert verify_record(manifest["base_workspace_receipt"])
        base = json.loads((root / source_map[manifest["base_workspace_receipt"]["path"]]["export_path"]).read_text())
        assert base["receipt_sha256"] == content_digest(base, "receipt_sha256")
        available_dependencies, remote_dependencies = [], []
        for dependency in capture["usd_dependency_inventory"]:
            target = available_dependencies if verify_record(dependency, "real_path") else remote_dependencies
            target.append(dependency)

        warmup = capture["render_only_diagnostic"]
        assert warmup["physics_actions"] == 0 and warmup["simulation_time_unchanged"] is True
        assert warmup["render_frames"] == 120
        assert [s["render_frame"] for s in warmup["snapshots"]] == list(range(121))
        times = sorted({s["sim_time_s"] for s in warmup["snapshots"]})
        assert len(times) == 1
        assert verify_record(warmup["viewport_video"])
        available_arrays = missing_arrays = 0
        for snapshot in warmup["snapshots"]:
            for record in snapshot["views"].values():
                if verify_record(record):
                    available_arrays += 1
                else:
                    missing_arrays += 1
        assert available_arrays == 18 and missing_arrays == 115
        assert set(capture["views"]) == set(CAMERAS)
        for camera in CAMERAS:
            assert verify_record(capture["views"][camera]["lossless_array"])

        previews = output / scene
        previews.mkdir()
        for camera in CAMERAS:
            entries = []
            for frame in SAMPLES:
                source = directory / "render_diagnostic" / f"{camera}-{frame:04d}.npy"
                array = np.load(source, allow_pickle=False)
                assert array.dtype == np.uint8 and array.shape == (720, 1280, 3) and np.ptp(array) > 0
                entries.append((f"{scene} | {camera} | render {frame}", Image.fromarray(array)))
            strip = previews / f"{camera}-warmup-samples.png"
            sheet(strip, entries, 3)
            source = directory / "views" / f"{camera}.npy"
            array = np.load(source, allow_pickle=False)
            assert np.array_equal(array, np.load(directory / "render_diagnostic" / f"{camera}-0120.npy", allow_pickle=False))
            image_path = previews / f"{camera}-final.png"
            Image.fromarray(array).save(image_path)
            media_index.append({
                "scene": scene, "camera": camera, "render_frame": 120,
                "source_array": str(source.relative_to(root)), "source_array_sha256": sha256(source),
                "original_shape": list(array.shape), "png": str(image_path.relative_to(output)),
                "png_sha256": sha256(image_path), "png_pixels_identical_to_source_array": True,
                "warmup_sheet": str(strip.relative_to(output)), "warmup_sheet_sha256": sha256(strip),
                "warmup_render_frame_ids": list(SAMPLES),
            })

        video_path = directory / "render_diagnostic/render_only.mp4"
        video_samples = []
        video_errors = {}
        frame_count = 0
        reader = imageio.get_reader(video_path, "ffmpeg")
        try:
            metadata = reader.get_meta_data()
            for index, array in enumerate(reader):
                assert array.shape == (720, 1280, 3) and array.dtype == np.uint8 and np.ptp(array) > 0
                frame_count += 1
                if index in SAMPLES:
                    video_samples.append((f"{scene} | video frame {index} | display {index / 30:.3f}s", Image.fromarray(array)))
                    original = np.load(directory / "render_diagnostic" / f"over_shoulder_left_camera-{index:04d}.npy", allow_pickle=False)
                    video_errors[str(index)] = float(np.abs(array.astype(float) - original.astype(float)).mean())
        finally:
            reader.close()
        assert frame_count == warmup["viewport_video"]["frame_count"] == 121
        assert metadata["fps"] == 30
        video_sheet = previews / "decoded-video-samples.png"
        sheet(video_sheet, video_samples, 3)

        objects = capture["objects"]
        cube, bowl = objects["rubiks_cube"], objects["bowl"]
        center = lambda row: np.array(row["geometric_center_env_local_xyz_m"], dtype=float)
        lo = lambda row: np.array(row["bbox_env_local_min_xyz_m"], dtype=float)
        hi = lambda row: np.array(row["bbox_env_local_max_xyz_m"], dtype=float)
        table_top = hi(objects["table"])[2]
        reconstruction, quaternion_norms = {}, {}
        for name, row in objects.items():
            quat = np.array(row["root_quaternion_world_wxyz"], dtype=float)
            reconstructed = np.array(row["root_position_env_local_xyz_m"]) + rotation(quat) @ np.array(row["geometric_center_offset_root_local_xyz_m"])
            reconstruction[name] = float(np.max(np.abs(reconstructed - center(row))))
            quaternion_norms[name] = float(np.linalg.norm(quat))
            assert np.isfinite(reconstructed).all() and np.all(lo(row) <= center(row)) and np.all(center(row) <= hi(row))
        supports = manifest["native_import_contract"]["kinematic_bodies"]
        gaps = {name: float(lo(objects[name])[2] - table_top) for name in supports}
        overlaps = []
        for a, b in itertools.combinations(objects, 2):
            overlap = np.minimum(hi(objects[a]), hi(objects[b])) - np.maximum(lo(objects[a]), lo(objects[b]))
            if np.all(overlap > 1e-6):
                overlaps.append({"a": a, "b": b, "axis_overlap_m": overlap.tolist(), "meaning": "AABB overlap only, not mesh penetration proof."})
        contact_norms = {}
        for name, contact in capture["support_contact_measurements"].items():
            matrix = np.array(contact["force_matrix_world_n"], dtype=float)
            assert list(matrix.shape) == contact["shape"] and np.isfinite(matrix).all()
            assert contact["sensor"] in capture["contact_sensor_inventory"]
            norm = float(np.linalg.norm(matrix.reshape(-1, 3), axis=1).sum())
            assert contact["nonzero_force_observed"] == (norm > 0)
            contact_norms[name] = norm
        if manifest["family"] == "HEIGHT":
            relation = float(center(cube)[2] - center(bowl)[2])
            goals = ("height_lower_support", "height_upper_support")
            neutral = "height_neutral_cube_support"
        else:
            plate = objects["plate"]
            relation = float(np.linalg.norm(center(cube) - center(plate)) - np.linalg.norm(center(cube) - center(bowl)))
            goals = ("dist_bowl_landing_support", "dist_plate_landing_support")
            neutral = "dist_neutral_cube_support"
        hypothetical_landings = {}
        for name in goals:
            point = center(objects[name]).copy()
            point[2] = hi(objects[name])[2] + center(cube)[2] - lo(cube)[2]
            if manifest["family"] == "HEIGHT":
                goal_relation = float(point[2] - center(bowl)[2])
            else:
                goal_relation = float(np.linalg.norm(point - center(plate)) - np.linalg.norm(point - center(bowl)))
            hypothetical_landings[name] = {
                "center_xyz_m": point.tolist(), "signed_relation_m": goal_relation,
                "claim_boundary": "Geometric same-orientation AABB extrapolation; not a measured landing, contact, reachability or qualification result.",
            }
        analysis["scenes"][scene] = {
            "capture_sha256": expected, "capture_receipt_content_sha256": capture["receipt_sha256"],
            "manifest_file_sha256": sha256(manifest_path), "manifest_content_sha256": manifest["manifest_sha256"],
            "overlay_sha256": manifest["overlay_usda"]["sha256"], "counterbalance": manifest["counterbalance"],
            "environment_origin_world_xyz_m": capture["environment_origin_world_xyz_m"],
            "simulation_time_s": times[0], "physics_actions": 0,
            "local_recording_files_verified": available_arrays + 4,
            "recorded_warmup_arrays_not_exported": missing_arrays,
            "decoded_video_frames": frame_count, "video_sha256": sha256(video_path),
            "video_fps_display_only": metadata["fps"],
            "video_sample_mean_absolute_rgb_compression_error": video_errors,
            "video_sheet": str(video_sheet.relative_to(output)), "video_sheet_sha256": sha256(video_sheet),
            "locally_verified_usd_dependencies": available_dependencies,
            "recorded_usd_dependencies_not_exported": remote_dependencies,
            "material_asset_records": warmup["material_assets"],
            "measured_objects": objects,
            "center_reconstruction_max_coordinate_error_m": reconstruction,
            "quaternion_norms": quaternion_norms,
            "table_top_z_m": float(table_top), "support_underside_minus_table_top_m": gaps,
            "cube_bottom_minus_neutral_support_top_m": float(lo(cube)[2] - hi(objects[neutral])[2]),
            "neutral_signed_relation_m": relation, "neutral_within_frozen_5mm": abs(relation) <= 0.005,
            "initial_cube_support_contact_force_norm_n": contact_norms,
            "positive_volume_aabb_overlaps": overlaps,
            "hypothetical_same_orientation_landings_not_physical_evidence": hypothetical_landings,
        }
        print(scene, json.dumps({
            "neutral_relation_m": relation, "support_gaps_m": gaps, "contact_norm_n": contact_norms,
            "center_reconstruction_error_max_m": max(reconstruction.values()), "video_frames": frame_count,
        }, sort_keys=True))
    write_json(output / "geometry_and_integrity.json", analysis)
    write_json(output / "media_index.json", media_index)


if __name__ == "__main__":
    main()
