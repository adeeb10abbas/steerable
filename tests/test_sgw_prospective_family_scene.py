import json

import pytest

from experiments.workshops.spatial_grounding_v1.prospective_family_scene import build_overlay


def _workspace():
    return {
        "measurement_schema_version": "sgw-01-lat-measured-workspace-v2",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "receipt_sha256": "r" * 64,
        "asset_manifest_sha256": "a" * 64,
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "task_asset": "rubiks_cube_banana_bowl.usda",
        "objects": {"rubiks_cube": {}, "bowl": {}, "table": {}},
    }


def _base_scene(tmp_path):
    path = tmp_path / "base.usda"
    path.write_text('#usda 1.0\ndef Xform "scene" {}\n')
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
    assert "height_upper_support" in text
    assert "subLayers" in text
    try:
        from pxr import Usd
    except ImportError:
        pytest.skip("usd-core not installed")
    stage = Usd.Stage.Open(str(output))
    assert stage.GetPrimAtPath("/scene/height_upper_support").IsValid()
    assert stage.GetPrimAtPath("/scene/height_lower_support").IsValid()


def test_dist_overlay_has_visual_plate_and_counterbalance(tmp_path):
    receipt = tmp_path / "workspace.json"
    receipt.write_text(json.dumps(_workspace()))
    output = tmp_path / "dist.usda"
    manifest = build_overlay(
        family="DIST", base_scene=_base_scene(tmp_path), workspace_receipt=receipt, output=output,
        bowl_side="right",
    )

    assert manifest["counterbalance"] == {"bowl_side": "right"}
    assert manifest["native_import_contract"]["dynamic_bodies"] == []
    assert "dist_plate" in output.read_text()


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
