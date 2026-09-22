"""Run one real RoboLab DROID scene reset and RTX frame with zero policy calls."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .build_asset_manifest import is_git_worktree


def _record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--study-root", type=Path, required=True)
    bootstrap.add_argument("--robolab-root", type=Path, required=True)
    bootstrap.add_argument("--assets-manifest", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--environment-seed", type=int, default=20260922)
    known, _ = bootstrap.parse_known_args()
    if known.output.exists():
        raise FileExistsError(f"refusing to overwrite renderer receipt: {known.output}")
    if not is_git_worktree(known.robolab_root) or not known.assets_manifest.is_file():
        raise ValueError("preflight requires a pinned RoboLab checkout and measured asset manifest")
    if str(known.study_root.resolve()) not in sys.path:
        sys.path.insert(0, str(known.study_root.resolve()))

    import cv2
    import numpy as np
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser(parents=[bootstrap])
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args, _ = parser.parse_known_args()
    args.enable_cameras = True
    if not args.headless or args.num_envs != 1:
        parser.error("renderer preflight requires exactly one headless environment")
    if args.renderer != "realtime" or args.rendering_type != "balanced":
        parser.error("renderer preflight requires realtime renderer and balanced rendering mode")
    app = AppLauncher(args).app
    try:
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        effective_import = Path(robolab.__file__).resolve()
        if not effective_import.is_relative_to(known.robolab_root.resolve()):
            raise RuntimeError("effective RoboLab import is outside the pinned checkout")
        task_path = known.study_root / "experiments/workshops/spatial_grounding_v1/renderer_probe_task.py"
        if not task_path.is_file():
            raise FileNotFoundError(task_path)
        native = known.output.parent / "native"
        native.mkdir(parents=True, exist_ok=False)
        set_output_dir(str(native))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        robolab.constants.VERBOSE = False
        auto_register_droid_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        env, _ = create_env(
            "SGWRendererProbeTask",
            device=args.device,
            seed=known.environment_seed,
            num_envs=1,
            instruction_type="default",
            policy="sgw_01_zero_model_renderer_preflight",
            renderer=args.renderer,
            rendering_mode=args.rendering_type,
        )
        try:
            obs, _ = env.reset()
            views: dict[str, dict[str, Any]] = {}
            for camera in ("over_shoulder_left_camera", "wrist_cam", "over_shoulder_right_camera"):
                image = np.asarray(obs["image_obs"][camera][0].detach().cpu().numpy(), dtype=np.uint8)
                if image.ndim != 3 or image.shape[-1] != 3 or not np.ptp(image):
                    raise RuntimeError(f"invalid RTX RGB frame from {camera}")
                views[camera] = {"shape": list(image.shape), "pixel_range": int(np.ptp(image))}
            cube = env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().tolist()
            bowl = env.scene["bowl"].data.root_pos_w[0].detach().cpu().tolist()
        finally:
            env.close()
        commit = subprocess.check_output(
            ["git", "-C", str(known.robolab_root), "rev-parse", "HEAD"], text=True
        ).strip()
        receipt = {
            "schema_version": "sgw-01-robolab-isaac-renderer-preflight-v2",
            "status": "passed_zero_model_renderer_preflight",
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "robolab_commit": commit,
            "robolab_import": _record(effective_import),
            "assets_manifest": _record(known.assets_manifest),
            "task_source": _record(task_path),
            "renderer": {"backend": "realtime RTX Vulkan", "quality": "balanced", "device": args.device},
            "views": views,
            "measured_default_scene_centers_world_m": {"rubiks_cube": cube, "bowl": bowl},
            "versions": {name: importlib.metadata.version(name) for name in ("isaacsim", "isaaclab", "robolab")},
        }
        known.output.write_text(json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n")
    finally:
        app.close()


if __name__ == "__main__":
    main()
