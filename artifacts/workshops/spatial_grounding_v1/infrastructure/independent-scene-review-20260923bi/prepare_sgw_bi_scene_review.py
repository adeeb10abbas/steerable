"""Reproduce the offline bi evidence checks without editing bf or running Isaac."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


sys.dont_write_bytecode = True
SHARED_SHA256 = "d590f61c454769ea92dcc4cec9e95e345352691006f3c204adf5cc8d5920623f"
shared_path = Path(__file__).with_name("prepare_sgw_scene_review.py")
assert hashlib.sha256(shared_path.read_bytes()).hexdigest() == SHARED_SHA256
import prepare_sgw_scene_review as shared


SOURCE = "5f7a0911b7384b98cc2094b4054ad75f9c3b226b"
CAPTURES = {
    "0-height-left": "d8c961be41b65d4d6860f550466a2a0b000d44bd147f631901f6de31d98b9148",
    "1-height-right": "35327020323b406969eef7613d94946bd776001567d0dcc4e223dfc3330ad3b7",
    "2-dist-left": "beb0991e7f11bbd57f4e762627ceffd478aeeab1e91a6f4f65723e2c41d31b32",
    "3-dist-right": "7a8ace8867e0e1807518b9ca073c2004fc80fe9b9c11a1a489d57a49857ee19a",
}
BF_CAPTURES = dict(shared.SCENES)
DETAILS = [
    ("0-height-left", "over_shoulder_right_camera", (400, 255, 770, 470)),
    ("1-height-right", "over_shoulder_right_camera", (400, 255, 770, 470)),
    ("2-dist-left", "over_shoulder_right_camera", (385, 295, 810, 470)),
    ("3-dist-right", "over_shoulder_left_camera", (500, 280, 945, 455)),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.evidence.resolve(), args.output.resolve()
    previous = root.parent / "sgw-bf-review-evidence"

    # Reuse the frozen verifier with this run's explicit identities only.
    shared.SCENES = CAPTURES
    shared.SOURCE = SOURCE
    shared.main()

    result = {
        "claim_boundary": (
            "Measured initial geometry and filtered contacts, not a visual verdict, "
            "sustained stability, physical qualification, or policy recognition."
        ),
        "study_source_commit": SOURCE,
        "shared_verifier_sha256": SHARED_SHA256,
        "engineering_clearance_rule": (
            "SGW-ENG-007's 0.02 m minimum Euclidean XY AABB separation from "
            "added supports; this is not a new scientific success threshold."
        ),
        "scenes": {},
    }
    for scene, expected in CAPTURES.items():
        capture_path = root / scene / "capture.json"
        before_path = previous / scene / "capture.json"
        assert shared.sha256(capture_path) == expected
        assert shared.sha256(before_path) == BF_CAPTURES[scene]
        capture = json.loads(capture_path.read_text())
        before = json.loads(before_path.read_text())
        manifest = json.loads((root / scene / "scene/manifest.json").read_text())
        before_manifest = json.loads((previous / scene / "scene/manifest.json").read_text())
        objects, old_objects = capture["objects"], before["objects"]
        contract = manifest["native_import_contract"]
        assert set(contract["objects_of_interest"]) == set(objects)
        assert set(capture["banana_contact_measurements"]) == set(contract["banana_contact_bodies"])
        low = lambda row: np.array(row["bbox_env_local_min_xyz_m"], dtype=float)
        high = lambda row: np.array(row["bbox_env_local_max_xyz_m"], dtype=float)
        banana, table = objects["banana"], objects["table"]
        bmin, bmax, tmin, tmax = low(banana), high(banana), low(table), high(table)
        banana_contacts = {}
        for body, row in capture["banana_contact_measurements"].items():
            force = np.array(row["force_matrix_world_n"], dtype=float)
            assert force.ndim == 4 and force.shape[0] == 1 and force.shape[-1] == 3
            assert force.size > 0 and np.isfinite(force).all() and list(force.shape) == row["shape"]
            assert row["sensor"] in capture["contact_sensor_inventory"]
            assert row["sensor"] in (f"banana__{body}", f"{body}__banana")
            norm = float(np.linalg.norm(force.reshape(-1, 3), axis=1).sum())
            assert row["nonzero_force_observed"] == (norm > 0)
            banana_contacts[body] = {**row, "sum_contact_force_norm_n": norm}

        support_checks = {}
        for name in contract["kinematic_bodies"]:
            row, old = objects[name], old_objects[name]
            rmin, rmax = low(row), high(row)
            separation_xy = np.maximum(np.maximum(rmin[:2] - bmax[:2], bmin[:2] - rmax[:2]), 0)
            clearance = float(np.linalg.norm(separation_xy))
            support_checks[name] = {
                "bottom_minus_table_top_m": float(rmin[2] - tmax[2]),
                "top_change_from_bf_m": float(rmax[2] - high(old)[2]),
                "footprint_max_coordinate_change_from_bf_m": float(max(
                    np.abs(rmin[:2] - low(old)[:2]).max(),
                    np.abs(rmax[:2] - high(old)[:2]).max(),
                )),
                "banana_xy_axis_separation_m": separation_xy.tolist(),
                "banana_xy_aabb_clearance_m": clearance,
                "banana_meets_20mm_engineering_clearance": clearance >= 0.02,
            }

        unchanged_rows = {}
        for name in ("rubiks_cube", "bowl", "table", *(("plate",) if "plate" in objects else ())):
            unchanged_rows[name] = {
                field: float(np.max(np.abs(np.array(value) - np.array(old_objects[name][field]))))
                for field, value in objects[name].items()
            }
        old_specs = {
            row["name"]: row for row in before_manifest["prospective_design"]["dimensions_and_poses"]
        }
        design_checks = {}
        for row in manifest["prospective_design"]["dimensions_and_poses"]:
            old = old_specs[row["name"]]
            changes = {key: {"bf": old.get(key), "bi": row.get(key)}
                       for key in old.keys() | row.keys() if old.get(key) != row.get(key)}
            design_checks[row["name"]] = changes

        result["scenes"][scene] = {
            "capture_sha256": expected,
            "bf_capture_sha256": BF_CAPTURES[scene],
            "banana_measured_geometry": banana,
            "banana_bottom_minus_table_top_m": float(bmin[2] - tmax[2]),
            "banana_table_xy_edge_margins_m": {
                "min_x": float(bmin[0] - tmin[0]), "max_x": float(tmax[0] - bmax[0]),
                "min_y": float(bmin[1] - tmin[1]), "max_y": float(tmax[1] - bmax[1]),
            },
            "banana_within_measured_table_xy_bounds": bool(
                np.all(bmin[:2] >= tmin[:2]) and np.all(bmax[:2] <= tmax[:2])
            ),
            "banana_contact_measurements": banana_contacts,
            "supports": support_checks,
            "scored_and_table_row_max_coordinate_deltas_from_bf": unchanged_rows,
            "authored_spec_changes_from_bf": design_checks,
            "counterbalance_unchanged_from_bf": manifest["counterbalance"] == before_manifest["counterbalance"],
            "plate_category_caveat": manifest["prospective_design"]["plate_category_caveat"],
        }
        print(scene, json.dumps({
            "max_support_bottom_error_m": max(abs(row["bottom_minus_table_top_m"]) for row in support_checks.values()),
            "minimum_banana_support_clearance_m": min(row["banana_xy_aabb_clearance_m"] for row in support_checks.values()),
            "banana_table_contact_norm_n": banana_contacts["table"]["sum_contact_force_norm_n"],
            "banana_max_nontable_contact_norm_n": max(
                row["sum_contact_force_norm_n"] for name, row in banana_contacts.items() if name != "table"
            ),
            "max_scored_or_table_row_delta": max(
                value for row in unchanged_rows.values() for value in row.values()
            ),
        }, sort_keys=True))
    shared.write_json(output / "repair_checks.json", result)

    image_records, overview = [], []
    for scene, camera, rectangle in DETAILS:
        source = root / scene / "views" / f"{camera}.npy"
        pixels = np.load(source, allow_pickle=False)
        x0, y0, x1, y1 = rectangle
        path = output / scene / f"{camera}-repair-detail.png"
        Image.fromarray(pixels[y0:y1, x0:x1]).save(path)
        assert np.array_equal(np.asarray(Image.open(path)), pixels[y0:y1, x0:x1])
        image_records.append({
            "scene": scene, "camera": camera, "render_frame": 120,
            "source_array": str(source.relative_to(root)),
            "source_array_sha256": shared.sha256(source),
            "source_rectangle_xyxy": list(rectangle),
            "png": str(path.relative_to(output)), "png_sha256": shared.sha256(path),
            "pixels_identical_to_source_region": True,
            "claim_boundary": "Native-pixel diagnostic crop, not an actual policy input.",
        })
        overview.append((f"{scene} | {camera} | native render 120", Image.fromarray(pixels)))
    overview_path = output / "native_repair_overview.png"
    shared.sheet(overview_path, overview, 2, width=640)
    shared.write_json(output / "repair_images.json", {
        "details": image_records,
        "overview": {
            "path": overview_path.name, "sha256": shared.sha256(overview_path),
            "claim_boundary": "Resized display-only overview; never treated as native policy preprocessing.",
        },
    })


if __name__ == "__main__":
    main()
