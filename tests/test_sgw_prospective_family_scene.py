import json

import pytest

from experiments.workshops.spatial_grounding_v1.prospective_family_scene import build_overlay
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from experiments.workshops.spatial_grounding_v1.prospective_family_capture import _manifest, _usd_dependencies


def _workspace():
    value = {
        "measurement_schema_version": "sgw-01-lat-measured-workspace-v2",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "receipt_sha256": "",
        "asset_manifest_sha256": "a" * 64,
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "task_asset": "rubiks_cube_banana_bowl.usda",
        "objects": {
            "rubiks_cube": {"geometric_center_offset_root_local_xyz_m": [-.01, .02, -.002]},
            "bowl": {"geometric_center_offset_root_local_xyz_m": [0, 0, 0]},
            "table": {},
        },
    }
    value["receipt_sha256"] = workspace_digest(value)
    return value


def _base_scene(tmp_path):
    path = tmp_path / "base.usda"
    path.write_text('#usda 1.0\n(\n defaultPrim = "World"\n)\ndef Xform "World" {\n def Xform "rubiks_cube" {}\n def Xform "bowl" {}\n def Xform "table" {}\n}\n')
    return path


def test_height_overlay_is_prospective_and_parseable_with_usd_core(tmp_path):
    receipt = tmp_path / "workspace.json"
    receipt.write_text(json.dumps(_workspace()))
    output = tmp_path / "height.usda"
    manifest = build_overlay(
        family="HEIGHT", base_scene=_base_scene(tmp_path), workspace_receipt=receipt, output=output,
        upper_side="left",
    )

    assert manifest["status"] == "prospective_scene_design_not_measured_or_qualified"
    assert manifest["counterbalance"] == {"upper_support_side": "left"}
    text = output.read_text()
    assert 'over "World"' in text
    assert "subLayers" in text
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        pytest.skip("usd-core not installed")
    stage = Usd.Stage.Open(str(output))
    assert stage.GetDefaultPrim().GetPath() == "/World"
    assert stage.GetPrimAtPath("/World/height_upper_support").IsValid()
    assert stage.GetPrimAtPath("/World/height_lower_support").IsValid()
    assert stage.GetPrimAtPath("/World/height_upper_support/geometry").HasAPI(UsdPhysics.CollisionAPI)
    assert stage.GetPrimAtPath("/World/height_upper_support").GetAttribute("xformOp:translate").Get()[1] != stage.GetPrimAtPath("/World/height_lower_support").GetAttribute("xformOp:translate").Get()[1]


def test_dist_overlay_has_visual_plate_and_counterbalance(tmp_path):
    receipt = tmp_path / "workspace.json"
    receipt.write_text(json.dumps(_workspace()))
    output = tmp_path / "dist.usda"
    manifest = build_overlay(
        family="DIST", base_scene=_base_scene(tmp_path), workspace_receipt=receipt, output=output,
        bowl_side="right",
    )

    assert manifest["counterbalance"] == {"bowl_side": "right"}
    assert manifest["native_import_contract"]["dynamic_bodies"] == ["plate"]
    assert "def Xform \"plate\"" in output.read_text()
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        pytest.skip("usd-core not installed")
    stage = Usd.Stage.Open(str(output))
    plate = stage.GetPrimAtPath("/World/plate")
    assert plate.HasAPI(UsdPhysics.RigidBodyAPI)
    assert plate.GetChild("geometry").HasAPI(UsdPhysics.CollisionAPI)
    bowl_y = stage.GetPrimAtPath("/World/bowl").GetAttribute("xformOp:translate").Get()[1]
    plate_y = plate.GetAttribute("xformOp:translate").Get()[1]
    assert bowl_y * plate_y < 0


def test_overlay_refuses_non_model_blind_or_wrong_workspace_receipt(tmp_path):
    bad = _workspace()
    bad["model_request_count"] = 1
    receipt = tmp_path / "bad.json"
    receipt.write_text(json.dumps(bad))

    with pytest.raises(ValueError, match="model blind"):
        build_overlay(
            family="HEIGHT", base_scene=_base_scene(tmp_path), workspace_receipt=receipt,
            output=tmp_path / "height.usda", upper_side="left",
        )


def test_overlay_refuses_workspace_with_forged_receipt_digest(tmp_path):
    receipt = tmp_path / "workspace.json"
    value = _workspace()
    value["objects"]["bowl"]["invented"] = True
    receipt.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="content digest"):
        build_overlay(
            family="HEIGHT", base_scene=_base_scene(tmp_path), workspace_receipt=receipt,
            output=tmp_path / "height.usda", upper_side="left",
        )


def test_capture_manifest_rechecks_base_and_workspace_bytes_and_lists_layers(tmp_path):
    receipt = tmp_path / "workspace.json"
    receipt.write_text(json.dumps(_workspace()))
    base = _base_scene(tmp_path)
    output = tmp_path / "height.usda"
    manifest = build_overlay(
        family="HEIGHT", base_scene=base, workspace_receipt=receipt, output=output, upper_side="right",
    )
    manifest_path = tmp_path / "height.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    try:
        dependencies = _usd_dependencies(output)
    except ImportError:
        pytest.skip("usd-core not installed")
    assert {row["real_path"] for row in dependencies} >= {str(base), str(output)}
    assert _manifest(manifest_path)["family"] == "HEIGHT"
    original_base = base.read_text()
    base.write_text(original_base + "\n# tampered")
    with pytest.raises(ValueError, match="base scene hash"):
        _manifest(manifest_path)
    base.write_text(original_base)
    tampered_workspace = json.loads(receipt.read_text())
    tampered_workspace["objects"]["bowl"]["geometric_center_offset_root_local_xyz_m"][0] += 0.001
    receipt.write_text(json.dumps(tampered_workspace))
    with pytest.raises(ValueError, match="workspace receipt hash"):
        _manifest(manifest_path)
