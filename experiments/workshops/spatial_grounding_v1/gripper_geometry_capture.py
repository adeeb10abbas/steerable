"""Bounded, empty-gripper geometry calibration; never a learned-policy rollout."""

import os
from pathlib import Path
from typing import Any

import numpy as np

from .recorder import atomic_json, encode_viewport_video
from .robolab_measurements import articulation_body_frames


def finger_geometry(stage: Any, body_names: list[str]) -> list[dict[str, Any]]:
    from pxr import Usd, UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath("/World/envs/env_0/robot/Gripper/Robotiq_2F_85")
    if not root:
        raise ValueError("pinned Robotiq gripper prim is missing")
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    transforms = UsdGeom.XformCache()
    rows = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if "finger" not in str(prim.GetPath()).lower() or not prim.IsA(UsdGeom.Gprim):
            continue
        body = prim
        while body and body.GetName() not in body_names:
            body = body.GetParent()
        if not body:
            raise ValueError(f"finger geometry has no measured articulation body: {prim.GetPath()}")
        box = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if box.IsEmpty():
            raise ValueError(f"finger geometry has an empty bound: {prim.GetPath()}")
        relative, resets_stack = transforms.ComputeRelativeTransform(prim, body)
        if resets_stack:
            raise ValueError("finger geometry resets its transform stack")
        center = relative.Transform(box.GetMidpoint())
        rows.append({
            "prim": str(prim.GetPath()), "type": str(prim.GetTypeName()),
            "body": str(body.GetName()), "body_prim": str(body.GetPath()),
            "collision_geometry": prim.HasAPI(UsdPhysics.CollisionAPI),
            "geometry_local_min": list(box.GetMin()), "geometry_local_max": list(box.GetMax()),
            "geometry_to_body_matrix_gf": [[float(v) for v in row] for row in relative],
            "body_local_bounds_center_m": list(center),
        })
    if not rows:
        raise ValueError("no actual finger geometry was measured")
    return rows


def capture_gripper_motion(
    env: Any, observation: dict[str, Any], output: Path, *, geometry: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    robot = env.scene["robot"]
    data = robot.data
    if not np.allclose(data.root_quat_w[0].detach().cpu().numpy(), [1, 0, 0, 0], atol=1e-6):
        raise ValueError("gripper probe requires the measured identity robot-base orientation")
    body_index = list(data.body_names).index("base_link")
    position = (data.body_pos_w[0, body_index] - data.root_pos_w[0]).detach().cpu().numpy().copy()
    quaternion = data.body_quat_w[0, body_index].detach().cpu().numpy().copy()
    dt = float(env.step_dt)
    if not np.isfinite(dt) or dt <= 0 or 60 * dt >= 5:
        raise ValueError("60-action gripper probe must fit the native five-second task")
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "geometry.json", {"finger_geometry": geometry})
    records = []
    frames = []
    issued_commands = 0
    started = float(env.sim.current_time)
    status = "infrastructure_invalid_gripper_probe"

    def observe(index: int) -> None:
        image = np.asarray(observation["image_obs"]["over_shoulder_left_camera"][0].detach().cpu().numpy())
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3 or not np.ptp(image):
            raise ValueError("gripper probe requires nonblank uint8 RGB")
        path = output / f"frame-{index:04d}.npy"
        with path.open("xb") as stream:
            np.save(stream, image, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        record = {
            "action_step": index, "sim_time_s": float(env.sim.current_time) - started,
            "robot_body_frames": articulation_body_frames(data),
            "joint_names": list(robot.joint_names),
            "joint_position_rad": data.joint_pos[0].detach().cpu().numpy().tolist(),
            "object_roots_world_xyz_m": {
                name: env.scene[name].data.root_pos_w[0].detach().cpu().numpy().tolist()
                for name in ("rubiks_cube", "bowl", "banana")
            },
        }
        atomic_json(output / f"state-{index:04d}.json", record)
        records.append(record)
        frames.append(path)
        if not np.isclose(record["sim_time_s"], index * dt, rtol=0, atol=1e-5):
            raise RuntimeError(f"native physical-time mismatch at gripper action {index}")

    try:
        observe(0)
        for index in range(1, 61):
            command = np.concatenate((position, quaternion, [0.785398 if index <= 30 else 0.0])).astype(np.float32)
            atomic_json(output / f"command-{index:04d}.json", {"action_step": index, "action": command.tolist()})
            action = torch.as_tensor(command.reshape(1, 8), dtype=torch.float32, device=env.device)
            issued_commands += 1
            observation, _reward, terminated, truncated, _info = env.step(action)
            observe(index)
            if bool(terminated[0]) or bool(truncated[0]):
                raise RuntimeError(f"native task terminated during gripper calibration at action {index}")
        status = "measured_empty_gripper_motion_not_fixture_qualification"
    finally:
        receipt = {
            "status": status, "model_requests": 0, "behavioral_episodes": 0,
            "control_step_dt_s": dt, "issued_controller_commands": issued_commands,
            "observed_controller_actions": max(0, len(records) - 1),
            "finger_geometry": geometry,
            "command_frame": "robot-root-relative Robotiq base_link flange",
            "static_flange_position_root_xyz_m": position.tolist(),
            "static_flange_quaternion_root_wxyz": quaternion.tolist(),
            "records": records,
            "viewport_video": encode_viewport_video(frames, output / "viewport.mp4", fps=1 / dt) if frames else None,
        }
        atomic_json(output / "receipt.json", receipt)
    return observation, receipt
