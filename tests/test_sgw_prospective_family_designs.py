import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from experiments.workshops.spatial_grounding_v1.prospective_family_capture import _manifest as capture_manifest
from experiments.workshops.spatial_grounding_v1.prospective_family_designs import (
    _geometric_rejection,
    author_candidate_overlay,
    build_design_plan,
    require_design_capture,
)
from experiments.workshops.spatial_grounding_v1.prospective_family_scene import build_overlay


WORKSPACE = Path("artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json")


def _base_scene(tmp_path):
    path = tmp_path / "base.usda"
    path.write_text(
        '#usda 1.0\n( defaultPrim = "World" )\ndef Xform "World" {\n'
        ' def Xform "rubiks_cube" {}\n def Xform "bowl" {}\n def Xform "table" {}\n}\n'
    )
    return path


def _baseline_files(tmp_path, family):
    workspace = json.loads(WORKSPACE.read_text())
    base = _base_scene(tmp_path)
    captures, manifests = {}, {}
    for side in ("left", "right"):
        overlay = tmp_path / f"{family.lower()}-{side}.usda"
        manifest = build_overlay(
            family=family, base_scene=base, workspace_receipt=WORKSPACE, output=overlay,
            **{("upper_side" if family == "HEIGHT" else "bowl_side"): side},
        )
        manifest_path = tmp_path / f"{family.lower()}-{side}.manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        objects = {
            name: workspace["objects"][name]
            for name in ("rubiks_cube", "bowl", "banana", "table")
        }
        banana = dict(objects["banana"])
        original_root = banana["root_position_env_local_xyz_m"]
        captured_root = manifest["prospective_design"]["authored_actor_root_overrides_env_local_xyz_m"]["banana"]
        delta = [captured_root[index] - original_root[index] for index in range(3)]
        banana["root_position_env_local_xyz_m"] = captured_root
        banana["geometric_center_env_local_xyz_m"] = [
            value + delta[index] for index, value in enumerate(banana["geometric_center_env_local_xyz_m"])
        ]
        for key in ("bbox_env_local_min_xyz_m", "bbox_env_local_max_xyz_m"):
            banana[key] = [value + delta[index] for index, value in enumerate(banana[key])]
        objects["banana"] = banana
        if family == "DIST":
            plate = next(spec for spec in manifest["prospective_design"]["dimensions_and_poses"] if spec["name"] == "plate")
            objects["plate"] = {
                "root_position_env_local_xyz_m": plate["center_m"],
                "root_quaternion_world_wxyz": [1, 0, 0, 0],
                "geometric_center_offset_root_local_xyz_m": [0, 0, 0],
            }
        receipt = {
            "schema_version": "sgw-01-prospective-family-native-capture-v1",
            "status": "prospective_native_capture_not_candidate_qualified",
            "family": family,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "receipt_sha256": "",
            "overlay_manifest_sha256": manifest["manifest_sha256"],
            "asset_manifest_sha256": manifest["base_workspace_receipt"]["asset_manifest_sha256"],
            "robolab_commit": manifest["base_workspace_receipt"]["robolab_commit"],
            "objects": objects,
            "usd_dependency_inventory": [
                {"real_path": str(overlay.resolve()), "sha256": manifest["overlay_usda"]["sha256"]},
                {"real_path": str(base.resolve()), "sha256": manifest["base_scene"]["sha256"]},
            ],
        }
        receipt["receipt_sha256"] = workspace_digest(receipt)
        capture = tmp_path / f"{family.lower()}-{side}.capture.json"
        capture.write_text(json.dumps(receipt))
        captures[side], manifests[side] = capture, manifest_path
    return captures, manifests


@pytest.mark.parametrize("family", ("HEIGHT", "DIST"))
def test_plan_binds_current_workspace_derived_baselines_and_preserves_fixed_slots(tmp_path, family):
    captures, manifests = _baseline_files(tmp_path, family)
    plan = build_design_plan(
        family=family, seed=91, count=100, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )

    assert plan["design_slot_count"] == 100
    assert len(plan["designs"]) == 100
    assert {row["side"] for row in plan["designs"]} == {"left", "right"}
    assert plan["accepted_design_count"] + plan["geometric_rejection_count"] == 100
    assert all(row["baseline"]["asset_manifest_sha256"] == "3a9b8ceeff333aa3a060dad97707b5f2a50e7c4db1423c84350bf4d8a19b0806" for row in plan["designs"])
    assert all(row["status"].startswith("prospective_design_") for row in plan["designs"])


def test_plan_rejects_mutated_bound_baseline_bytes(tmp_path):
    captures, manifests = _baseline_files(tmp_path, "HEIGHT")
    plan = build_design_plan(
        family="HEIGHT", seed=91, count=4, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    captures["left"].write_text(captures["left"].read_text() + "\n")
    design = next(row for row in plan["designs"] if row["side"] == "left" and row["status"].endswith("capture"))
    with pytest.raises(ValueError, match="bytes differ"):
        author_candidate_overlay(
            plan=plan, design_id=design["design_id"], output=tmp_path / "candidate.usda",
            manifest_output=tmp_path / "candidate.manifest.json",
        )


def test_duplicate_screen_includes_previously_rejected_slots(tmp_path, monkeypatch):
    from experiments.workshops.spatial_grounding_v1 import prospective_family_designs as module

    captures, manifests = _baseline_files(tmp_path, "HEIGHT")
    screen = module._geometric_rejection
    calls = []

    def reject_first(*args, **kwargs):
        calls.append(len(args[5]))
        return "first_slot_geometry_rejected" if len(calls) == 1 else screen(*args, **kwargs)

    monkeypatch.setattr(module, "_geometric_rejection", reject_first)
    monkeypatch.setattr(module, "_translation", lambda *args: [0.0, 0.0])
    plan = build_design_plan(
        family="HEIGHT", seed=91, count=2,
        baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    assert calls == [0, 1]
    assert plan["designs"][1]["geometric_rejection"] == "duplicate_layout_within_3mm_2deg"
    assert plan["geometric_rejection_count"] == 2


@pytest.mark.parametrize("layer", ("overlay_usda", "base_scene"))
def test_candidate_authoring_rejects_mutated_inherited_baseline_usd(tmp_path, layer):
    captures, manifests = _baseline_files(tmp_path, "HEIGHT")
    plan = build_design_plan(
        family="HEIGHT", seed=91, count=4, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    baseline_manifest = json.loads(manifests["left"].read_text())
    baseline_overlay = Path(baseline_manifest[layer]["path"])
    baseline_overlay.write_text(baseline_overlay.read_text() + "\n# mutation")
    design = next(row for row in plan["designs"] if row["side"] == "left" and row["status"].endswith("capture"))
    with pytest.raises(ValueError, match="bytes differ"):
        author_candidate_overlay(
            plan=plan, design_id=design["design_id"], output=tmp_path / "candidate.usda",
            manifest_output=tmp_path / "candidate.manifest.json",
        )


@pytest.mark.parametrize("mutation", ("inherited_layer", "asset_manifest_sha256", "robolab_commit"))
def test_candidate_overlay_requires_its_own_capture_before_materialization(tmp_path, mutation):
    captures, manifests = _baseline_files(tmp_path, "DIST")
    plan = build_design_plan(
        family="DIST", seed=91, count=4, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    design = next(row for row in plan["designs"] if row["status"] == "prospective_design_requires_zero_model_capture")
    overlay = tmp_path / "candidate.usda"
    candidate_manifest_path = tmp_path / "candidate.manifest.json"
    candidate = author_candidate_overlay(
        plan=plan, design_id=design["design_id"], output=overlay, manifest_output=candidate_manifest_path,
    )

    assert candidate["status"] == "prospective_candidate_design_requires_zero_model_capture"
    assert candidate["source_baseline"]["capture"]["sha256"] == design["baseline"]["capture"]["sha256"]
    assert capture_manifest(candidate_manifest_path)["manifest_sha256"] == candidate["manifest_sha256"]
    with pytest.raises(ValueError, match="does not bind"):
        require_design_capture(candidate_manifest_path=candidate_manifest_path, capture_path=captures["left"])

    captured = json.loads(captures["left"].read_text())
    captured["overlay_manifest_sha256"] = candidate["manifest_sha256"]
    captured["usd_dependency_inventory"].append({
        "real_path": candidate["overlay_usda"]["path"], "sha256": candidate["overlay_usda"]["sha256"],
    })
    captured["receipt_sha256"] = workspace_digest(captured)
    candidate_capture = tmp_path / "candidate.capture.json"
    candidate_capture.write_text(json.dumps(captured))
    assert require_design_capture(candidate_manifest_path=candidate_manifest_path, capture_path=candidate_capture)["family"] == "DIST"
    if mutation == "inherited_layer":
        baseline_overlay = Path(candidate["source_baseline"]["used_layer_dependencies"][0]["real_path"])
        baseline_overlay.write_text(baseline_overlay.read_text() + "\n# mutation after capture")
    else:
        captured[mutation] = "0" * len(captured[mutation])
        captured["receipt_sha256"] = workspace_digest(captured)
        candidate_capture.write_text(json.dumps(captured))
    with pytest.raises(ValueError, match="bytes differ|identity differs"):
        require_design_capture(candidate_manifest_path=candidate_manifest_path, capture_path=candidate_capture)


def test_candidate_overlay_preserves_measured_plate_root_over_authored_plate(tmp_path):
    pytest.importorskip("pxr.Usd")
    captures, manifests = _baseline_files(tmp_path, "DIST")
    capture = json.loads(captures["left"].read_text())
    capture["objects"]["plate"]["root_position_env_local_xyz_m"] = [.71, -.22, .15]
    capture["receipt_sha256"] = workspace_digest(capture)
    captures["left"].write_text(json.dumps(capture))
    plan = build_design_plan(
        family="DIST", seed=91, count=2, baseline_capture_paths=captures, baseline_manifest_paths=manifests,
    )
    design = next(row for row in plan["designs"] if row["side"] == "left" and row["status"].endswith("capture"))
    candidate = author_candidate_overlay(
        plan=plan, design_id=design["design_id"], output=tmp_path / "candidate.usda",
        manifest_output=tmp_path / "candidate.manifest.json",
    )
    from pxr import Usd
    stage = Usd.Stage.Open(candidate["overlay_usda"]["path"])
    plate = stage.GetPrimAtPath("/World/plate")
    expected = design["authored_scored_object_roots"]["plate"]["position_m"]
    assert list(plate.GetAttribute("xformOp:translate").Get()) == pytest.approx(expected)


def test_duplicate_gate_uses_fixture_component_tolerance_across_sides():
    roots = {
        "rubiks_cube": {"position_m": [.4, .2, .1], "quaternion_wxyz": [1, 0, 0, 0]},
        "bowl": {"position_m": [.5, .2, .1], "quaternion_wxyz": [1, 0, 0, 0]},
    }
    shifted = {
        name: {**pose, "position_m": [pose["position_m"][0] + .0025, pose["position_m"][1] + .0025, pose["position_m"][2]]}
        for name, pose in roots.items()
    }
    capture = {"objects": {
        "table": {
            "bbox_env_local_min_xyz_m": [0, 0, 0], "bbox_env_local_max_xyz_m": [1, 1, 1],
        },
        "banana": {
            "bbox_env_local_min_xyz_m": [.8, .3, .05], "bbox_env_local_max_xyz_m": [.9, .4, .1],
        },
    }}
    assert _geometric_rejection(capture, "HEIGHT", "right", shifted, [0, 0], [{"roots": roots}]) == (
        "duplicate_layout_within_3mm_2deg"
    )


def test_candidate_geometry_rejects_banana_support_clearance_and_table_escape():
    capture = {"objects": {
        "table": {
            "bbox_env_local_min_xyz_m": [0, 0, 0], "bbox_env_local_max_xyz_m": [1, 1, 1],
        },
        "banana": {
            "bbox_env_local_min_xyz_m": [.745, .301, .05], "bbox_env_local_max_xyz_m": [.855, .479, .09],
        },
    }}
    roots = {
        "rubiks_cube": {"position_m": [.4, .2, .1], "quaternion_wxyz": [1, 0, 0, 0]},
        "bowl": {"position_m": [.5, .2, .1], "quaternion_wxyz": [1, 0, 0, 0]},
    }
    support = {"name": "support", "center_m": [.65, .23, .1], "size_m": [.14, .14, .1]}
    assert _geometric_rejection(capture, "HEIGHT", "left", roots, [.04, .04], [], support_specs=[support]) == (
        "banana_support_clearance_below_20mm:support"
    )
    capture["objects"]["banana"]["bbox_env_local_max_xyz_m"][0] = 1.01
    assert _geometric_rejection(capture, "HEIGHT", "left", roots, [0, 0], [], support_specs=[support]) == (
        "banana_outside_measured_table_xy_bounds"
    )
