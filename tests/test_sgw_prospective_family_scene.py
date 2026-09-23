import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from experiments.workshops.spatial_grounding_v1.prospective_family_scene import build_overlay
from experiments.workshops.spatial_grounding_v1.lat_candidate_generator import workspace_digest
from experiments.workshops.spatial_grounding_v1.lat_workspace_capture import _rotate_wxyz
from experiments.workshops.spatial_grounding_v1 import prospective_family_capture as capture
from experiments.workshops.spatial_grounding_v1.prospective_family_capture import (
    _capture_object_rows,
    _contact_inventory,
    _create_capture_environment,
    _manifest,
    _usd_dependencies,
    _validate_capture_bindings,
)


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
            "rubiks_cube": _object(
                root=[.43, -.09, .081], quat=[math.sqrt(.5), 0, 0, math.sqrt(.5)],
                offset=[-.01, .02, -.002], bbox_min=[.39, -.12, .05], bbox_max=[.45, -.06, .11],
            ),
            "bowl": _object(
                root=[.44, .12, .077], quat=[math.sqrt(.5), 0, 0, math.sqrt(.5)],
                offset=[.008, -.004, .001], bbox_min=[.36, .04, .05], bbox_max=[.53, .21, .105],
            ),
            "table": {},
        },
    }
    value["receipt_sha256"] = workspace_digest(value)
    return value


def _object(*, root, quat, offset, bbox_min, bbox_max):
    center = [root[index] + _rotate_wxyz(quat, offset)[index] for index in range(3)]
    return {
        "root_position_env_local_xyz_m": root,
        "root_quaternion_world_wxyz": quat,
        "geometric_center_offset_root_local_xyz_m": offset,
        "geometric_center_env_local_xyz_m": center,
        "bbox_env_local_min_xyz_m": bbox_min,
        "bbox_env_local_max_xyz_m": bbox_max,
    }


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
    assert stage.GetPrimAtPath("/World/height_neutral_cube_support").IsValid()
    assert stage.GetPrimAtPath("/World/height_upper_support").IsValid()
    assert stage.GetPrimAtPath("/World/height_lower_support").IsValid()
    assert stage.GetPrimAtPath("/World/height_upper_support/geometry").HasAPI(UsdPhysics.CollisionAPI)
    design = manifest["prospective_design"]
    overrides = design["authored_actor_root_overrides_env_local_xyz_m"]
    workspace = _workspace()
    for name in ("rubiks_cube", "bowl"):
        object_row = workspace["objects"][name]
        root = overrides[name]
        center = [root[index] + _rotate_wxyz(object_row["root_quaternion_world_wxyz"], object_row["geometric_center_offset_root_local_xyz_m"])[index] for index in range(3)]
        assert center[2] == pytest.approx(0.16)
    _assert_non_overlapping_supports(design["dimensions_and_poses"])


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
    assert "def Cylinder \"geometry\"" in output.read_text()
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        pytest.skip("usd-core not installed")
    stage = Usd.Stage.Open(str(output))
    plate = stage.GetPrimAtPath("/World/plate")
    assert plate.HasAPI(UsdPhysics.RigidBodyAPI)
    assert plate.GetChild("geometry").HasAPI(UsdPhysics.CollisionAPI)
    assert plate.GetChild("geometry").GetTypeName() == "Cylinder"
    bowl_y = stage.GetPrimAtPath("/World/bowl").GetAttribute("xformOp:translate").Get()[1]
    plate_y = plate.GetAttribute("xformOp:translate").Get()[1]
    assert bowl_y * plate_y < 0
    workspace = _workspace()
    overrides = manifest["prospective_design"]["authored_actor_root_overrides_env_local_xyz_m"]
    bowl_center = _center(workspace["objects"]["bowl"], overrides["bowl"])
    cube_center = _center(workspace["objects"]["rubiks_cube"], overrides["rubiks_cube"])
    plate_center = list(plate.GetAttribute("xformOp:translate").Get())
    assert math.dist(cube_center, bowl_center) == pytest.approx(math.dist(cube_center, plate_center))


@pytest.mark.parametrize("family,side", [
    ("HEIGHT", "left"), ("HEIGHT", "right"), ("DIST", "left"), ("DIST", "right"),
])
def test_real_measured_receipt_authors_supported_poses_and_goal_clearance(tmp_path, family, side):
    from pxr import Usd

    receipt = Path(__file__).parents[1] / (
        "artifacts/workshops/spatial_grounding_v1/infrastructure/a40-20260922r-workspace.json"
    )
    workspace = json.loads(receipt.read_text())
    manifest = build_overlay(
        family=family, base_scene=_base_scene(tmp_path), workspace_receipt=receipt,
        output=tmp_path / "scene.usda", **{("upper_side" if family == "HEIGHT" else "bowl_side"): side},
    )
    stage = Usd.Stage.Open(manifest["overlay_usda"]["path"])
    specs = {row["name"]: row for row in manifest["prospective_design"]["dimensions_and_poses"]}
    roots = manifest["prospective_design"]["authored_actor_root_overrides_env_local_xyz_m"]
    centers = {}
    for name in ("rubiks_cube", "bowl"):
        prim = stage.GetPrimAtPath(f"/World/{name}")
        quat = prim.GetAttribute("xformOp:orient").Get()
        assert [quat.GetReal(), *quat.GetImaginary()] == pytest.approx(
            workspace["objects"][name]["root_quaternion_world_wxyz"], abs=1e-7,
        )
        centers[name] = _center(workspace["objects"][name], roots[name])
        support_name = (
            "height_neutral_cube_support" if name == "rubiks_cube" else "height_reference_support"
        ) if family == "HEIGHT" else (
            "dist_neutral_cube_support" if name == "rubiks_cube" else "dist_bowl_support"
        )
        support = specs[support_name]
        lowest = roots[name][2] + (
            workspace["objects"][name]["bbox_env_local_min_xyz_m"][2]
            - workspace["objects"][name]["root_position_env_local_xyz_m"][2]
        )
        assert lowest == pytest.approx(support["center_m"][2] + support["size_m"][2] / 2)
        assert centers[name][:2] == pytest.approx(support["center_m"][:2])
    _assert_non_overlapping_supports(list(specs.values()))
    if family == "DIST":
        plate = specs["plate"]
        support = specs["dist_plate_support"]
        assert plate["center_m"][2] - plate["thickness_m"] / 2 == pytest.approx(
            support["center_m"][2] + support["size_m"][2] / 2,
        )
        assert math.dist(centers["rubiks_cube"], centers["bowl"]) == pytest.approx(
            math.dist(centers["rubiks_cube"], plate["center_m"]),
        )
        cube = workspace["objects"]["rubiks_cube"]
        center_bottom = cube["geometric_center_env_local_xyz_m"][2] - cube["bbox_env_local_min_xyz_m"][2]
        for name, sign in (("bowl", 1), ("plate", -1)):
            goal = specs[f"dist_{name}_landing_support"]
            target = goal["center_m"][:2] + [goal["center_m"][2] + goal["size_m"][2] / 2 + center_bottom]
            margin = math.dist(target, plate["center_m"]) - math.dist(target, centers["bowl"])
            assert sign * margin >= 0.03


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


@pytest.mark.parametrize("side", ("left", "right"))
def test_height_counterbalance_supports_are_non_overlapping(side, tmp_path):
    receipt = tmp_path / "workspace.json"
    receipt.write_text(json.dumps(_workspace()))
    manifest = build_overlay(
        family="HEIGHT", base_scene=_base_scene(tmp_path), workspace_receipt=receipt,
        output=tmp_path / f"height-{side}.usda", upper_side=side,
    )
    specs = {row["name"]: row for row in manifest["prospective_design"]["dimensions_and_poses"]}
    assert specs["height_upper_support"]["center_m"][1] * specs["height_lower_support"]["center_m"][1] < 0
    _assert_non_overlapping_supports(list(specs.values()))


def _center(row, root):
    return [root[index] + _rotate_wxyz(row["root_quaternion_world_wxyz"], row["geometric_center_offset_root_local_xyz_m"])[index] for index in range(3)]


def _assert_non_overlapping_supports(specs):
    boxes = [row for row in specs if "size_m" in row]
    for index, left in enumerate(boxes):
        for right in boxes[index + 1:]:
            overlap = all(
                abs(left["center_m"][axis] - right["center_m"][axis])
                < (left["size_m"][axis] + right["size_m"][axis]) / 2
                for axis in range(3)
            )
            assert not overlap, f"{left['name']} overlaps {right['name']}"


def test_capture_native_boundary_uses_model_free_constructor_world_and_contacts():
    calls = []

    def create_env(*args, **kwargs):
        calls.append((args, kwargs))
        return "environment", {"unused": True}

    args = SimpleNamespace(
        device="cuda:0", environment_seed=20260922, renderer="realtime", rendering_type="balanced", num_envs=1,
    )
    assert _create_capture_environment(create_env, args) == ("environment", {"unused": True})
    assert calls == [(("SGWProspectiveFamilyCaptureTask",), {
        "device": "cuda:0", "seed": 20260922, "num_envs": 1, "instruction_type": "default",
        "policy": "sgw_01_zero_model_prospective_capture", "renderer": "realtime", "rendering_mode": "balanced",
    })]

    class World:
        def get_pose(self, name, *, env_id):
            assert env_id == 0
            return [1, 2, 3], [1, 0, 0, 0]

        def get_bbox(self, name, *, env_id):
            assert env_id == 0
            return [[0, 1, 2], [2, 3, 4]], [1, 2, 3]

    rows = _capture_object_rows(World(), ["plate"])
    assert rows["plate"]["geometric_center_offset_root_local_xyz_m"] == [0.0, 0.0, 0.0]
    assert rows["plate"]["bbox_env_local_min_xyz_m"] == [0.0, 1.0, 2.0]
    assert _contact_inventory(lambda scene: {"rubiks_cube__plate": object(), "cube__all_objs": object()}, object()) == [
        "rubiks_cube__plate"
    ]


def test_capture_source_and_asset_bindings_are_checked_before_applauncher(tmp_path, monkeypatch):
    root = tmp_path / "robolab"
    source = root / "robolab/core/scenes/utils.py"
    source.parent.mkdir(parents=True)
    source.write_text("pinned scene source")
    assets = tmp_path / "assets.json"
    asset = tmp_path / "base.usda"
    asset.write_text("pinned asset bytes")
    assets.write_text(json.dumps({
        "scene": {"path": str(asset), "sha256": capture._sha256(asset), "bytes": asset.stat().st_size},
        "assets": [],
    }))
    args = SimpleNamespace(robolab_root=root, assets_manifest=assets)
    manifest = {
        "base_workspace_receipt": {
            "asset_manifest_sha256": capture._sha256(assets),
            "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        },
        "native_import_contract": {"robolab_utils_sha256": capture._sha256(source)},
    }
    monkeypatch.setattr(capture.subprocess, "check_output", lambda *args, **kwargs: "0aef241fb088ca21bb4ebd24448940ed56620d17\n")
    _validate_capture_bindings(args, manifest)
    manifest["native_import_contract"]["robolab_utils_sha256"] = "wrong"
    with pytest.raises(ValueError, match="import_scene source"):
        _validate_capture_bindings(args, manifest)
    manifest["native_import_contract"]["robolab_utils_sha256"] = capture._sha256(source)
    asset.write_text("mutated asset bytes")
    with pytest.raises(ValueError, match="asset payload differs"):
        _validate_capture_bindings(args, manifest)


def test_capture_enables_cameras_before_native_application_start(tmp_path, monkeypatch):
    receipt = tmp_path / "renderer.json"
    receipt.write_text(json.dumps({
        "status": "passed_zero_model_renderer_preflight", "model_request_count": 0,
    }))
    args = SimpleNamespace(
        headless=True, num_envs=1, renderer="realtime", rendering_type="balanced",
        enable_cameras=False, renderer_receipt=receipt, overlay_manifest=receipt,
    )
    monkeypatch.setattr(capture, "parse_args", lambda: args)
    monkeypatch.setattr(capture, "_manifest", lambda _: {})
    monkeypatch.setattr(capture, "_validate_capture_bindings", lambda *args: None)
    monkeypatch.setenv("SGW_PROSPECTIVE_OVERLAY_MANIFEST", "")
    monkeypatch.setenv("SGW_PROSPECTIVE_OVERLAY_MANIFEST_SHA256", "")

    class StopAtApplicationBoundary(Exception):
        pass

    def launcher(args):
        assert args.enable_cameras is True
        raise StopAtApplicationBoundary

    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=launcher))
    with pytest.raises(StopAtApplicationBoundary):
        capture.main()
