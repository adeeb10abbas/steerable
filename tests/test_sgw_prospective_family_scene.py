import json
import math
from pathlib import Path
import subprocess
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
            "banana": _object(
                root=[.54, -.08, .07], quat=[1, 0, 0, 0],
                offset=[0, 0, 0], bbox_min=[.49, -.17, .05], bbox_max=[.60, .01, .09],
            ),
            "table": _object(
                root=[.2, 0, .05], quat=[1, 0, 0, 0],
                offset=[.35, 0, -.35], bbox_min=[.2, -.48, -.65], bbox_max=[.9, .52, .05],
            ),
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
    banana_root = overrides["banana"]
    banana = workspace["objects"]["banana"]
    assert _center(banana, banana_root)[:2] == pytest.approx([.80, .39])
    assert banana_root[2] + (banana["bbox_env_local_min_xyz_m"][2] - banana["root_position_env_local_xyz_m"][2]) == pytest.approx(.05)


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
    assert manifest["prospective_design"]["plate_category_caveat"].startswith("The DIST plate is a simplified")
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
    Usd = pytest.importorskip("pxr.Usd")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")

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
    supports = {name for name in specs if name != "plate"}
    assert set(manifest["native_import_contract"]["kinematic_bodies"]) == supports
    for name in supports:
        prim = stage.GetPrimAtPath(f"/World/{name}")
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get() is True
        assert prim.GetChild("geometry").HasAPI(UsdPhysics.CollisionAPI)
        assert specs[name]["center_m"][2] - specs[name]["size_m"][2] / 2 == pytest.approx(
            workspace["objects"]["table"]["bbox_env_local_max_xyz_m"][2]
        )
    if family == "DIST":
        plate = stage.GetPrimAtPath("/World/plate")
        assert UsdPhysics.RigidBodyAPI(plate).GetKinematicEnabledAttr().Get() is False
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
    banana = workspace["objects"]["banana"]
    banana_root = roots["banana"]
    banana_center = _center(banana, banana_root)
    assert banana_center[:2] == pytest.approx([.80, .39])
    assert banana_root[2] + (
        banana["bbox_env_local_min_xyz_m"][2] - banana["root_position_env_local_xyz_m"][2]
    ) == pytest.approx(workspace["objects"]["table"]["bbox_env_local_max_xyz_m"][2])
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


@pytest.mark.parametrize("field", ("table", "banana"))
def test_overlay_refuses_malformed_measured_table_or_banana(tmp_path, field):
    value = _workspace()
    value["objects"][field]["bbox_env_local_max_xyz_m"][2] = float("nan")
    receipt = tmp_path / "bad.json"
    receipt.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="finite|invalid"):
        build_overlay(
            family="HEIGHT", base_scene=_base_scene(tmp_path), workspace_receipt=receipt,
            output=tmp_path / "height.usda", upper_side="left",
        )


def test_overlay_rejects_authored_banana_that_cannot_clear_supports(tmp_path):
    value = _workspace()
    value["objects"]["banana"]["bbox_env_local_min_xyz_m"][0] = .30
    value["objects"]["banana"]["bbox_env_local_max_xyz_m"][0] = .50
    value["receipt_sha256"] = workspace_digest(value)
    receipt = tmp_path / "workspace.json"
    receipt.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="banana clearance"):
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


def test_support_contacts_require_actual_filtered_forces():
    import numpy as np
    from test_sgw_render_warmup import Array

    sensor = SimpleNamespace(data=SimpleNamespace(force_matrix_w=Array(np.array([[[[0., 0., 1.]]]]))))
    sensors = {"rubiks_cube__height_neutral_cube_support": sensor}
    rows = capture._support_contact_measurements(sensors, ["height_neutral_cube_support"])
    assert rows["height_neutral_cube_support"]["force_matrix_world_n"] == [[[[0., 0., 1.]]]]
    assert rows["height_neutral_cube_support"]["nonzero_force_observed"] is True
    sensor.data.force_matrix_w = None
    with pytest.raises(RuntimeError, match="missing filtered"):
        capture._support_contact_measurements(sensors, ["height_neutral_cube_support"])
    with pytest.raises(RuntimeError, match="missing measured"):
        capture._support_contact_measurements(sensors, ["height_upper_support"])
    sensor.data.force_matrix_w = Array(np.array([[[[0., 0., 1.]]]]))
    sensors["banana__table"] = sensor
    sensors["banana__height_neutral_cube_support"] = sensor
    banana = capture._banana_contact_measurements(
        sensors, ["table", "height_neutral_cube_support"],
    )
    assert set(banana) == {"table", "height_neutral_cube_support"}
    with pytest.raises(RuntimeError, match="banana contact"):
        capture._banana_contact_measurements(sensors, [])


def test_capture_enables_cameras_before_native_application_start(tmp_path, monkeypatch):
    receipt = tmp_path / "renderer.json"
    receipt.write_text(json.dumps({
        "status": "passed_zero_model_renderer_preflight", "model_request_count": 0,
    }))
    args = SimpleNamespace(
        headless=True, num_envs=1, renderer="realtime", rendering_type="balanced",
        enable_cameras=False, renderer_receipt=receipt, overlay_manifest=receipt,
        output=tmp_path / "capture.json",
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


def test_capture_records_original_failure_before_application_cleanup_exits_zero(tmp_path, monkeypatch):
    renderer = tmp_path / "renderer.json"
    renderer.write_text(json.dumps({
        "status": "passed_zero_model_renderer_preflight", "model_request_count": 0,
    }))
    args = SimpleNamespace(
        headless=True, num_envs=1, renderer="realtime", rendering_type="balanced",
        renderer_receipt=renderer, overlay_manifest=renderer, output=tmp_path / "capture.json",
    )
    monkeypatch.setattr(capture, "parse_args", lambda: args)
    monkeypatch.setattr(capture, "_manifest", lambda _: {})
    monkeypatch.setattr(capture, "_validate_capture_bindings", lambda *args: None)
    monkeypatch.setenv("SGW_PROSPECTIVE_OVERLAY_MANIFEST", "")
    monkeypatch.setenv("SGW_PROSPECTIVE_OVERLAY_MANIFEST_SHA256", "")

    def close():
        raise SystemExit(0)

    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(
        AppLauncher=lambda _: SimpleNamespace(app=SimpleNamespace(close=close)),
    ))
    monkeypatch.setitem(sys.modules, "robolab", None)
    with pytest.raises(SystemExit) as exited:
        capture.main()
    assert exited.value.code == 0
    failure = json.loads(args.output.with_suffix(".failure.json").read_text())
    assert failure["error_type"] == "ModuleNotFoundError"
    assert "import robolab" in failure["traceback"]
    assert not args.output.exists()
    with pytest.raises(ValueError, match="infrastructure failure"):
        capture.verify_capture_artifacts(args.output)


def test_capture_failure_survives_immediate_native_process_exit(tmp_path):
    path = tmp_path / "capture.json"
    code = """
import os, sys
from pathlib import Path
from experiments.workshops.spatial_grounding_v1.prospective_family_capture import _capture_failure_guard
try:
    with _capture_failure_guard(Path(sys.argv[1])):
        raise RuntimeError("native constructor failed before capture")
finally:
    os._exit(0)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0
    failure = json.loads(path.with_suffix(".failure.json").read_text())
    assert failure["error"] == "native constructor failed before capture"
    assert failure["error"] in result.stderr
    with pytest.raises(FileExistsError, match="reuse"):
        with capture._capture_failure_guard(path):
            pytest.fail("failed capture root must not be reused")
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; from pathlib import Path; "
        "from experiments.workshops.spatial_grounding_v1.prospective_family_capture import verify_capture_artifacts; "
        "verify_capture_artifacts(Path(sys.argv[1]))", str(path),
    ], capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "infrastructure failure" in result.stderr


def test_capture_output_check_requires_actual_receipt_and_complete_video(tmp_path):
    import numpy as np
    from test_sgw_render_warmup import Array

    path = tmp_path / "capture.json"
    with pytest.raises(FileNotFoundError):
        capture.verify_capture_artifacts(path)
    workspace = tmp_path / "workspace.json"
    workspace.write_text(json.dumps(_workspace()))
    overlay = tmp_path / "height.usda"
    manifest = build_overlay(
        family="HEIGHT", base_scene=_base_scene(tmp_path), workspace_receipt=workspace,
        output=overlay, upper_side="left",
    )
    manifest_path = tmp_path / "height.manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    frame = np.arange(192, dtype=np.uint8).reshape(8, 8, 3)
    cameras = ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera")
    obs = {"image_obs": {camera: [Array(frame)] for camera in cameras}}
    env = SimpleNamespace(
        sim=SimpleNamespace(current_time=0.0, render=lambda: None),
        scene={camera: SimpleNamespace(update=lambda *args, **kwargs: None) for camera in cameras},
        observation_manager=SimpleNamespace(compute=lambda: obs),
    )
    _, warmup = capture.render_only_warmup(env, obs, 120, tmp_path / "render_diagnostic")
    views = {}
    for camera in cameras:
        view = tmp_path / f"{camera}.npy"
        np.save(view, frame)
        views[camera] = {"shape": list(frame.shape), "lossless_array": capture._record(view)}
    receipt = {
        "schema_version": "sgw-01-prospective-family-native-capture-v1",
        "status": "prospective_native_capture_not_candidate_qualified",
        "model_request_count": 0, "behavioral_episode_count": 0,
        "objects": {name: {} for name in manifest["native_import_contract"]["objects_of_interest"]},
        "banana_contact_measurements": {
            name: {} for name in manifest["native_import_contract"]["banana_contact_bodies"]
        },
        "views": views, "render_only_diagnostic": warmup,
        "overlay_manifest": capture._record(manifest_path),
        "overlay_manifest_sha256": manifest["manifest_sha256"],
        "usd_dependency_inventory": _usd_dependencies(overlay),
    }
    receipt["receipt_sha256"] = workspace_digest(receipt)
    path.write_text(json.dumps(receipt))
    assert capture.verify_capture_artifacts(path)["verified_recording_files"] == 137
    cleanup_failure = path.with_suffix(".cleanup.failure.json")
    cleanup_failure.write_text('{"error": "native cleanup failed"}')
    with pytest.raises(ValueError, match="infrastructure failure"):
        capture.verify_capture_artifacts(path)
    cleanup_failure.unlink()
    Path(warmup["viewport_video"]["path"]).unlink()
    with pytest.raises(ValueError, match="missing evidence"):
        capture.verify_capture_artifacts(path)
