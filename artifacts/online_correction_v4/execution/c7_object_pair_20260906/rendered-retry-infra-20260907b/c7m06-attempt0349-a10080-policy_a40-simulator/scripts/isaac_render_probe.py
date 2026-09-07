#!/usr/bin/env python3
"""Render one real RTX/Vulkan frame for the lane startup preflight."""
from __future__ import annotations

import argparse
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num-envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
if args.num_envs != 1 or not args.headless or args.device != "cuda:0" or args.rendering_mode != "balanced":
    parser.error("render probe requires headless balanced cuda:0")
if args.output.exists():
    parser.error(f"refusing to overwrite rendered-frame evidence: {args.output}")

simulation_app = AppLauncher(args).app
failure: tuple[BaseException, object] | None = None
try:
    import cv2
    import numpy as np
    import omni.replicator.core as rep
    import isaacsim.core.utils.prims as prim_utils
    from isaacsim.core.api.world import World
    from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
    from pxr import UsdGeom

    world = World(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda")
    world.scene.add_default_ground_plane()
    prim_utils.create_prim("/World/LaneProbeLight", "SphereLight", translation=(2.0, 2.0, 4.0))
    prim_utils.create_prim(
        "/World/LaneProbeCube", "Cube", translation=(0.0, 0.0, 1.0), scale=(0.2, 0.2, 0.2)
    )
    SingleGeometryPrim("/World/LaneProbeCube", collision=True)
    world.scene.add(SingleRigidPrim("/World/LaneProbeCube", name="lane_probe_cube", mass=1.0))
    camera = prim_utils.create_prim(
        "/World/LaneProbeCamera",
        "Camera",
        translation=(2.5, 2.5, 2.5),
        orientation=(0.33985113, 0.17591988, 0.42470818, 0.82047324),
    )
    UsdGeom.Camera(camera)
    product = rep.create.render_product("/World/LaneProbeCamera", resolution=(160, 120))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
    rgb.attach(product)
    world.reset()
    for _ in range(60):
        world.step(render=True)
    frame = np.asarray(rgb.get_data())
    if frame.shape[:2] != (120, 160) or not np.count_nonzero(frame):
        raise RuntimeError(f"RTX frame invalid: shape={frame.shape}, nonzero={int(np.count_nonzero(frame))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR)):
        raise RuntimeError("could not write rendered frame")
except BaseException as exc:
    # Isaac shutdown has raised SystemExit in prior incidents. Emit the original
    # probe failure before shutdown, then re-raise it with its traceback after
    # close so shutdown cannot turn a failed render into a successful child.
    traceback.print_exc()
    failure = (exc, exc.__traceback__)
finally:
    try:
        simulation_app.close()
    except BaseException:
        traceback.print_exc()
        if failure is None:
            raise

if failure is not None:
    original, original_traceback = failure
    raise original.with_traceback(original_traceback)
