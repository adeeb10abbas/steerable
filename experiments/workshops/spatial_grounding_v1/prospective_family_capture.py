"""Capture a prospective HEIGHT/DIST overlay with zero physics actions."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Mapping

from .build_asset_manifest import is_git_worktree
from .lat_candidate_generator import workspace_digest
from .lat_workspace_capture import (
    _record, _root_local_offset, _vector, material_asset_paths, render_only_warmup,
)
from .native_geometry_measurements import (
    camera_extrinsics, collision_geometry_local_bounds, native_articulation_path, robot_snapshot,
)


def _manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    digest = value.get("manifest_sha256")
    material = dict(value)
    material.pop("manifest_sha256", None)
    computed = hashlib.sha256((json.dumps(material, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    if digest != computed or value.get("status") not in {
        "prospective_scene_design_not_measured_or_qualified",
        "prospective_candidate_design_requires_zero_model_capture",
    }:
        raise ValueError("prospective overlay manifest is malformed or not capture-eligible")
    overlay = value.get("overlay_usda", {})
    if not isinstance(overlay, Mapping) or not Path(overlay.get("path", "")).is_file():
        raise ValueError("prospective overlay USD is missing")
    if _sha256(Path(overlay["path"])) != overlay.get("sha256"):
        raise ValueError("prospective overlay USD hash differs from manifest")
    base = value.get("base_scene", {})
    workspace = value.get("base_workspace_receipt", {})
    if not isinstance(base, Mapping) or not Path(base.get("path", "")).is_file() or _sha256(Path(base["path"])) != base.get("sha256"):
        raise ValueError("prospective base scene hash differs from manifest")
    if not isinstance(workspace, Mapping) or not Path(workspace.get("path", "")).is_file():
        raise ValueError("prospective base workspace receipt is missing")
    receipt = json.loads(Path(workspace["path"]).read_text(encoding="utf-8"))
    if _sha256(Path(workspace["path"])) != workspace.get("sha256") or receipt.get("receipt_sha256") != workspace.get("receipt_sha256"):
        raise ValueError("prospective base workspace receipt hash differs from manifest")
    if receipt.get("receipt_sha256") != workspace_digest(receipt):
        raise ValueError("prospective base workspace receipt content digest differs")
    if value.get("status") == "prospective_candidate_design_requires_zero_model_capture":
        _validate_inherited_baseline(value)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_inherited_baseline(value: Mapping[str, Any]) -> None:
    """Recheck every baseline layer inherited by a candidate overlay before launch."""

    source = value.get("source_baseline")
    dependencies = value.get("inherited_overlay_dependencies")
    if not isinstance(source, Mapping) or not isinstance(dependencies, list) or not dependencies:
        raise ValueError("candidate overlay lacks hash-bound inherited baseline dependencies")
    overlay = source.get("overlay_manifest")
    if not isinstance(overlay, Mapping) or not Path(overlay.get("path", "")).is_file():
        raise ValueError("candidate overlay baseline manifest is missing")
    if _sha256(Path(overlay["path"])) != overlay.get("sha256"):
        raise ValueError("candidate overlay baseline manifest bytes differ from binding")
    baseline = json.loads(Path(overlay["path"]).read_text(encoding="utf-8"))
    if baseline.get("manifest_sha256") != overlay.get("manifest_sha256"):
        raise ValueError("candidate overlay baseline manifest digest differs from binding")
    for dependency in dependencies:
        if not isinstance(dependency, Mapping) or not Path(dependency.get("real_path", "")).is_file():
            raise ValueError("candidate overlay inherited dependency is missing")
        path = Path(dependency["real_path"])
        if _sha256(path) != dependency.get("sha256") or path.stat().st_size != dependency.get("bytes"):
            raise ValueError("candidate overlay inherited dependency bytes differ from binding")


def _usd_dependencies(overlay: Path) -> list[dict[str, Any]]:
    """Capture composed USD layer dependencies with their current byte hashes."""

    from pxr import Usd

    stage = Usd.Stage.Open(str(overlay))
    if stage is None:
        raise RuntimeError("cannot open prospective overlay USD for dependency inventory")
    rows = []
    for layer in sorted(stage.GetUsedLayers(), key=lambda item: item.realPath):
        if not layer.realPath:
            # USD adds an in-memory session layer; it is not an asset dependency.
            continue
        path = Path(layer.realPath)
        rows.append({
            "identifier": layer.identifier,
            "real_path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        })
    if not rows or any(not row["exists"] for row in rows):
        raise RuntimeError("prospective overlay has unresolved USD layer dependencies")
    return rows


def _create_capture_environment(create_env: Any, args: argparse.Namespace) -> Any:
    """Construct exactly the model-free task environment through RoboLab's native boundary."""

    return create_env(
        "SGWProspectiveFamilyCaptureTask", device=args.device, seed=args.environment_seed, num_envs=1,
        instruction_type="default", policy="sgw_01_zero_model_prospective_capture",
        renderer=args.renderer, rendering_mode=args.rendering_type,
    )


def _capture_object_rows(world: Any, names: list[str], env_id: int = 0) -> dict[str, dict[str, Any]]:
    """Read roots and geometric centers from the native WorldState interface."""

    import numpy as np

    rows = {}
    for name in names:
        root, quat = world.get_pose(name, env_id=env_id)
        corners, center = world.get_bbox(name, env_id=env_id)
        root_values, quat_values, center_values = _vector(root), _vector(quat), _vector(center)
        points = np.asarray([_vector(corner) for corner in corners], dtype=np.float64)
        rows[name] = {
            "root_position_env_local_xyz_m": root_values,
            "root_quaternion_world_wxyz": quat_values,
            "geometric_center_env_local_xyz_m": center_values,
            "geometric_center_offset_root_local_xyz_m": _root_local_offset(root_values, quat_values, center_values),
            "bbox_env_local_min_xyz_m": points.min(axis=0).tolist(),
            "bbox_env_local_max_xyz_m": points.max(axis=0).tolist(),
        }
    return rows


def _contact_inventory(get_contact_sensors: Any, scene: Any) -> list[str]:
    return sorted(name for name in get_contact_sensors(scene) if not name.endswith("__all_objs"))


def _support_contact_measurements(sensors: Mapping[str, Any], supports: list[str]) -> dict[str, Any]:
    rows = {}
    for support in supports:
        rows[support] = _pair_contact_measurement(sensors, "rubiks_cube", support)
    return rows


def _pair_contact_measurement(sensors: Mapping[str, Any], first: str, second: str) -> dict[str, Any]:
    """Record a native filtered pair-contact matrix without inferring contact."""

    import numpy as np

    names = (f"{first}__{second}", f"{second}__{first}")
    sensor_name = next((name for name in names if name in sensors), None)
    if sensor_name is None:
        raise RuntimeError(f"missing measured {first} contact sensor for {second}")
    matrix = sensors[sensor_name].data.force_matrix_w
    if matrix is None:
        raise RuntimeError(f"missing filtered contact forces for {first}/{second}")
    values = np.asarray(matrix.detach().cpu().numpy())
    if values.ndim != 4 or values.shape[0] != 1 or values.shape[-1] != 3 or not values.size or not np.isfinite(values).all():
        raise RuntimeError(f"invalid measured contact force matrix for {first}/{second}")
    return {
        "sensor": sensor_name, "force_matrix_world_n": values.tolist(),
        "shape": list(values.shape),
        "nonzero_force_observed": bool(np.any(np.linalg.norm(values, axis=-1) > 0)),
    }


def _banana_contact_measurements(sensors: Mapping[str, Any], bodies: list[str]) -> dict[str, Any]:
    """Require observed native banana/table-and-support contact instrumentation."""

    if not bodies or len(set(bodies)) != len(bodies):
        raise RuntimeError("banana contact body contract is empty or contains duplicates")
    return {body: _pair_contact_measurement(sensors, "banana", body) for body in bodies}


def _verify_banana_contacts(receipt: Mapping[str, Any], contract: Mapping[str, Any]) -> None:
    import numpy as np

    expected = contract.get("banana_contact_bodies")
    if expected is None:
        return  # Earlier manifests did not request banana contact measurements.
    contacts = receipt.get("banana_contact_measurements")
    if not isinstance(contacts, Mapping) or set(contacts) != set(expected):
        raise ValueError("prospective capture omits required banana contact measurements")
    for body, record in contacts.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid banana contact record for {body}")
        sensor = record.get("sensor")
        if sensor not in (f"banana__{body}", f"{body}__banana") or sensor not in receipt.get("contact_sensor_inventory", ()):
            raise ValueError(f"invalid banana contact sensor binding for {body}")
        values = np.asarray(record.get("force_matrix_world_n"))
        if (
            values.ndim != 4 or values.shape[0] != 1 or values.shape[-1] != 3
            or not values.size or not np.issubdtype(values.dtype, np.number)
            or not np.isfinite(values).all() or list(values.shape) != record.get("shape")
        ):
            raise ValueError(f"invalid banana contact force matrix for {body}")
        if record.get("nonzero_force_observed") is not bool(np.any(np.linalg.norm(values, axis=-1) > 0)):
            raise ValueError(f"invalid banana contact force summary for {body}")


def _validate_capture_bindings(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    """Reject a capture before AppLauncher unless every native source binding matches."""

    workspace = manifest["base_workspace_receipt"]
    if _sha256(args.assets_manifest) != workspace["asset_manifest_sha256"]:
        raise ValueError("assets manifest bytes do not match the measured workspace binding")
    assets = json.loads(args.assets_manifest.read_text())
    for record in (assets["scene"], *assets["assets"]):
        path = Path(record["path"])
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"prospective capture asset payload differs from pinned manifest: {path}")
    scenes_utils = args.robolab_root / "robolab/core/scenes/utils.py"
    if not scenes_utils.is_file() or _sha256(scenes_utils) != manifest["native_import_contract"]["robolab_utils_sha256"]:
        raise ValueError("RoboLab import_scene source does not match the overlay binding")
    commit = subprocess.check_output(
        ["git", "-C", str(args.robolab_root), "rev-parse", "HEAD"], text=True,
    ).strip()
    if commit != workspace["robolab_commit"]:
        raise ValueError("RoboLab checkout commit does not match the measured workspace binding")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--study-root", type=Path, required=True)
    bootstrap.add_argument("--robolab-root", type=Path, required=True)
    bootstrap.add_argument("--assets-manifest", type=Path, required=True)
    bootstrap.add_argument("--renderer-receipt", type=Path, required=True)
    bootstrap.add_argument("--overlay-manifest", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--environment-seed", type=int, default=20260922)
    known, _ = bootstrap.parse_known_args()
    if known.output.exists():
        raise FileExistsError(f"refusing to overwrite prospective capture: {known.output}")
    if not is_git_worktree(known.robolab_root) or not known.assets_manifest.is_file() or not known.renderer_receipt.is_file():
        raise ValueError("capture requires pinned RoboLab, assets, and renderer receipt")
    manifest = _manifest(known.overlay_manifest)
    _validate_capture_bindings(known, manifest)
    if str(known.study_root.resolve()) not in sys.path:
        sys.path.insert(0, str(known.study_root.resolve()))
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser(parents=[bootstrap], allow_abbrev=False)
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


@contextmanager
def _capture_failure_guard(output: Path):
    failure = output.with_suffix(".failure.json")
    if failure.exists():
        raise FileExistsError(f"refusing to reuse a failed capture attempt: {failure}")
    try:
        yield
    except BaseException as error:
        value = {
            "status": "infrastructure_invalid_capture",
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        }
        failure.parent.mkdir(parents=True, exist_ok=True)
        with failure.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(value["traceback"], file=sys.stderr, flush=True)
        raise


def verify_capture_artifacts(path: Path) -> dict[str, Any]:
    """Verify retained output outside Isaac's potentially process-ending cleanup."""
    from .qualification_batch_verifier import _file, _frame, _scoped, _warmup

    if any(path.with_suffix(suffix).exists() for suffix in (".failure.json", ".cleanup.failure.json")):
        raise ValueError("capture attempt has a retained infrastructure failure")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (
        receipt.get("schema_version") != "sgw-01-prospective-family-native-capture-v1"
        or receipt.get("status") != "prospective_native_capture_not_candidate_qualified"
        or receipt.get("receipt_sha256") != workspace_digest(receipt)
        or receipt.get("model_request_count") != 0
        or receipt.get("behavioral_episode_count") != 0
    ):
        raise ValueError("prospective capture receipt identity or zero-model boundary differs")
    _validate_native_measurement_scope(receipt)
    root = path.parent.resolve()
    cameras = {"over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"}
    if set(receipt["views"]) != cameras:
        raise ValueError("prospective capture lacks all three original views")
    shape = None
    for camera, view in receipt["views"].items():
        record = view["lossless_array"]
        frame_path = _scoped(root, record["path"])
        _file(frame_path, record)
        frame = _frame(frame_path)
        if list(frame.shape) != view["shape"]:
            raise ValueError("prospective original view shape differs from receipt")
        if camera == "over_shoulder_left_camera":
            shape = frame.shape
    count = _warmup(root, receipt["render_only_diagnostic"], shape)
    overlay_record = receipt["overlay_manifest"]
    _file(Path(overlay_record["path"]), overlay_record)
    manifest = _manifest(Path(overlay_record["path"]))
    if manifest["manifest_sha256"] != receipt["overlay_manifest_sha256"]:
        raise ValueError("prospective capture overlay differs from its manifest")
    contract = manifest["native_import_contract"]
    objects = receipt.get("objects")
    if not isinstance(objects, Mapping) or not set(contract["objects_of_interest"]).issubset(objects):
        raise ValueError("prospective capture omits required measured objects")
    _verify_banana_contacts(receipt, contract)
    dependencies = receipt["usd_dependency_inventory"]
    required_layers = {manifest[key]["path"] for key in ("overlay_usda", "base_scene")}
    if not required_layers.issubset({row["real_path"] for row in dependencies}):
        raise ValueError("prospective capture dependency inventory omits required source layers")
    for dependency in dependencies:
        _file(Path(dependency["real_path"]), dependency)
    return {"status": "verified_capture_artifacts_not_fixture_qualification",
            "verified_recording_files": count + 3, "receipt": _record(path)}


def _validate_native_measurement_scope(receipt: Mapping[str, Any]) -> None:
    robot = receipt.get("robot_snapshot")
    cameras = receipt.get("camera_extrinsics")
    collision = receipt.get("collision_geometry_local_bounds")
    # Receipts produced before the additive engineering measurement contract
    # remain evidence of their original capture scope; new captures write all
    # three fields below and never interpret their absence as safe geometry.
    if robot is None and cameras is None and collision is None:
        return
    if not isinstance(robot, Mapping) or robot.get("measurement_scope") != "native_robot_state_not_policy_input":
        raise ValueError("prospective capture lacks native robot measurement scope")
    asset = robot.get("asset_usd")
    if not isinstance(asset, Mapping) or not isinstance(asset.get("available"), bool):
        raise ValueError("prospective capture robot asset identity is malformed")
    if asset["available"] and not all(isinstance(asset.get(key), value) for key, value in (("path", str), ("sha256", str), ("bytes", int))):
        raise ValueError("prospective capture robot asset identity is incomplete")
    if not isinstance(cameras, Mapping) or cameras.get("measurement_scope") != "native_camera_extrinsics_not_policy_input":
        raise ValueError("prospective capture lacks native camera extrinsic measurement scope")
    rows = cameras.get("cameras")
    if not isinstance(rows, Mapping) or set(rows) != {"over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"}:
        raise ValueError("prospective capture lacks fixed camera extrinsic inventory")
    for name, row in rows.items():
        if not isinstance(row, Mapping) or not isinstance(row.get("available"), bool):
            raise ValueError(f"prospective capture {name} camera extrinsic row is malformed")
    if not isinstance(collision, Mapping) or not isinstance(collision.get("available"), bool):
        raise ValueError("prospective capture collision geometry measurement is malformed")


def main() -> None:
    args = parse_args()
    args.enable_cameras = True
    if not args.headless or args.num_envs != 1 or args.renderer != "realtime" or args.rendering_type != "balanced":
        raise ValueError("prospective capture requires one headless realtime/balanced RTX environment")
    renderer = json.loads(args.renderer_receipt.read_text(encoding="utf-8"))
    if renderer.get("status") != "passed_zero_model_renderer_preflight" or renderer.get("model_request_count") != 0:
        raise ValueError("prospective capture requires passed zero-model renderer receipt")
    manifest = _manifest(args.overlay_manifest)
    _validate_capture_bindings(args, manifest)
    os.environ["SGW_PROSPECTIVE_OVERLAY_MANIFEST"] = str(args.overlay_manifest.resolve())
    os.environ["SGW_PROSPECTIVE_OVERLAY_MANIFEST_SHA256"] = _sha256(args.overlay_manifest)
    from isaaclab.app import AppLauncher

    app = None
    env = None
    try:
        with _capture_failure_guard(args.output):
            app = AppLauncher(args).app
            import numpy as np
            import robolab
            import robolab.constants
            import omni.usd
            from robolab.constants import set_output_dir
            from robolab.core.environments.runtime import create_env
            from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
            from robolab.core.world.world_state import get_world
            from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
            from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

            if not Path(robolab.__file__).resolve().is_relative_to(args.robolab_root.resolve()):
                raise RuntimeError("effective RoboLab import is outside pinned checkout")
            task_path = args.study_root / "experiments/workshops/spatial_grounding_v1/prospective_family_capture_task.py"
            native_name = (
                "capture_native"
                if manifest["status"] == "prospective_candidate_design_requires_zero_model_capture"
                else "native"
            )
            set_output_dir(str(args.output.parent / native_name))
            robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
            robolab.constants.RECORD_IMAGE_DATA = False
            auto_register_droid_abs_ik_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
            env, _ = _create_capture_environment(create_env, args)
            observation, _ = env.reset()
            observation, warmup = render_only_warmup(env, observation, 120, args.output.parent / "render_diagnostic")
            warmup["material_assets"] = material_asset_paths(omni.usd.get_context().get_stage())
            world, origin = get_world(env), env.scene.env_origins[0].detach().cpu().numpy()
            robot_state = robot_snapshot(env.scene, origin)
            cameras = camera_extrinsics(
                env.scene, ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"),
            )
            collision_geometry = collision_geometry_local_bounds(
                omni.usd.get_context().get_stage(),
                articulation_path=native_articulation_path(env.scene["robot"]),
                body_names=list(env.scene["robot"].data.body_names),
            )
            names = manifest["native_import_contract"]["objects_of_interest"]
            object_rows = _capture_object_rows(world, names)
            contacts = _contact_inventory(get_contact_sensors, env.scene)
            support_contacts = _support_contact_measurements(
                get_contact_sensors(env.scene),
                manifest["native_import_contract"]["kinematic_or_static_bodies"],
            )
            banana_contacts = _banana_contact_measurements(
                get_contact_sensors(env.scene),
                manifest["native_import_contract"]["banana_contact_bodies"],
            )
            views = {}
            root = args.output.parent / "views"
            root.mkdir(parents=True, exist_ok=False)
            for camera in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"):
                frame = np.asarray(observation["image_obs"][camera][0].detach().cpu().numpy(), dtype=np.uint8)
                if frame.ndim != 3 or frame.shape[-1] != 3 or not np.ptp(frame):
                    raise RuntimeError(f"prospective capture has invalid {camera} RGB")
                path = root / f"{camera}.npy"
                np.save(path, frame, allow_pickle=False)
                views[camera] = {"shape": list(frame.shape), "lossless_array": _record(path)}
            receipt = {
                "schema_version": "sgw-01-prospective-family-native-capture-v1",
                "status": "prospective_native_capture_not_candidate_qualified",
                "family": manifest["family"], "model_request_count": 0, "behavioral_episode_count": 0,
                "overlay_manifest": _record(args.overlay_manifest),
                "overlay_manifest_sha256": manifest["manifest_sha256"],
                "asset_manifest_sha256": _record(args.assets_manifest)["sha256"],
                "renderer_receipt": _record(args.renderer_receipt),
                "robolab_commit": subprocess.check_output(["git", "-C", str(args.robolab_root), "rev-parse", "HEAD"], text=True).strip(),
                "study_source_commit": subprocess.check_output(["git", "-C", str(args.study_root), "rev-parse", "HEAD"], text=True).strip(),
                "environment_seed": args.environment_seed,
                "environment_origin_world_xyz_m": _vector(origin),
                "robot_snapshot": robot_state,
                "camera_extrinsics": cameras,
                "collision_geometry_local_bounds": collision_geometry,
                "objects": object_rows, "contact_sensor_inventory": contacts, "views": views,
                "support_contact_measurements": support_contacts,
                "banana_contact_measurements": banana_contacts,
                "usd_dependency_inventory": _usd_dependencies(Path(manifest["overlay_usda"]["path"])),
                "render_only_diagnostic": warmup, "validated_slots": [],
                "versions": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "robolab")},
            }
            receipt["receipt_sha256"] = workspace_digest(receipt)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
    finally:
        try:
            with _capture_failure_guard(args.output.with_suffix(".cleanup.json")):
                if env is not None:
                    env.close()
        finally:
            if app is not None:
                app.close()


if __name__ == "__main__":
    main()
