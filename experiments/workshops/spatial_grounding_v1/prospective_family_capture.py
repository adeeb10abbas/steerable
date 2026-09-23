"""Capture a prospective HEIGHT/DIST overlay with zero physics actions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from .build_asset_manifest import is_git_worktree
from .lat_candidate_generator import workspace_digest
from .lat_workspace_capture import _record, _root_local_offset, _rotate_wxyz, _vector, render_only_warmup


def _manifest(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    digest = value.get("manifest_sha256")
    material = dict(value)
    material.pop("manifest_sha256", None)
    computed = hashlib.sha256((json.dumps(material, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()
    if digest != computed or value.get("status") != "prospective_scene_design_not_measured_or_qualified":
        raise ValueError("prospective overlay manifest is malformed or not prospective-only")
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
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _validate_capture_bindings(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    """Reject a capture before AppLauncher unless every native source binding matches."""

    workspace = manifest["base_workspace_receipt"]
    if _sha256(args.assets_manifest) != workspace["asset_manifest_sha256"]:
        raise ValueError("assets manifest bytes do not match the measured workspace binding")
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


def main() -> None:
    args = parse_args()
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

    app = AppLauncher(args).app
    try:
        import numpy as np
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        if not Path(robolab.__file__).resolve().is_relative_to(args.robolab_root.resolve()):
            raise RuntimeError("effective RoboLab import is outside pinned checkout")
        task_path = args.study_root / "experiments/workshops/spatial_grounding_v1/prospective_family_capture_task.py"
        set_output_dir(str(args.output.parent / "native"))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        auto_register_droid_abs_ik_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        env, _ = _create_capture_environment(create_env, args)
        try:
            observation, _ = env.reset()
            observation, warmup = render_only_warmup(env, observation, 120, args.output.parent / "render_diagnostic")
            world, origin = get_world(env), env.scene.env_origins[0].detach().cpu().numpy()
            names = manifest["native_import_contract"]["objects_of_interest"]
            object_rows = _capture_object_rows(world, names)
            contacts = _contact_inventory(get_contact_sensors, env.scene)
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
        finally:
            env.close()
        receipt = {
            "schema_version": "sgw-01-prospective-family-native-capture-v1",
            "status": "prospective_native_capture_not_candidate_qualified",
            "family": manifest["family"], "model_request_count": 0, "behavioral_episode_count": 0,
            "overlay_manifest": _record(args.overlay_manifest),
            "overlay_manifest_sha256": manifest["manifest_sha256"],
            "asset_manifest_sha256": _record(args.assets_manifest)["sha256"],
            "renderer_receipt": _record(args.renderer_receipt),
            "robolab_commit": subprocess.check_output(["git", "-C", str(args.robolab_root), "rev-parse", "HEAD"], text=True).strip(),
            "environment_seed": args.environment_seed,
            "environment_origin_world_xyz_m": _vector(origin),
            "objects": object_rows, "contact_sensor_inventory": contacts, "views": views,
            "usd_dependency_inventory": _usd_dependencies(Path(manifest["overlay_usda"]["path"])),
            "render_only_diagnostic": warmup, "validated_slots": [],
            "versions": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "robolab")},
        }
        receipt["receipt_sha256"] = workspace_digest(receipt)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    finally:
        app.close()


if __name__ == "__main__":
    main()
