#!/usr/bin/env python3
"""Deterministic model-blind absolute-IK reference-controller queue for V3-E002.

The controller uses one waypoint recipe for both directions and both layouts;
only the registered signed robot-Y target displacement changes.  No learned
model or policy client is imported by this module.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--study-root", type=Path, required=True)
ap.add_argument("--robolab-root", type=Path, required=True)
ap.add_argument("--candidate", type=Path, required=True)
ap.add_argument("--candidate-sha256", required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--condition", choices=("control:left", "control:right", "position_mirrored:left", "position_mirrored:right"), required=True)
ap.add_argument("--seed-start", type=int, default=9400)
ap.add_argument("--seed-end", type=int, default=9426)
ap.add_argument("--depth-m", type=float, required=True)
ap.add_argument("--pod", required=True)
ap.add_argument("--gpu-uuid", required=True)
ap.add_argument("--steps-per-waypoint", type=int, default=20)
from robolab.eval.runner import add_common_eval_args
add_common_eval_args(ap); AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args(); args.enable_cameras = True
root = args.study_root.resolve(); sys.path.insert(0, str(root))
os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(args.candidate.resolve())
os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = args.candidate_sha256
app = AppLauncher(args).app

from robolab.core.environments.runtime import create_env
from robolab.core.task.conditionals import object_grabbed, object_dropped, object_left_of, object_right_of
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
from robolab.robots.droid import EEF_OFFSET_ROT

TASKS = {
    "control:left": ("control_left.py", "V3B002Pi05ControlLeftTask"),
    "control:right": ("control_right.py", "V3B002Pi05ControlRightTask"),
    "position_mirrored:left": ("position_mirrored_left.py", "V3B002Pi05PositionMirroredLeftTask"),
    "position_mirrored:right": ("position_mirrored_right.py", "V3B002Pi05PositionMirroredRightTask"),
}

def arr(x):
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

def qmul(a, b):
    w1,x1,y1,z1 = a; w2,x2,y2,z2 = b
    return np.asarray([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def qinv(q): return np.asarray([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)

def main():
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite {args.output}")
    candidate = json.loads(args.candidate.read_text())
    if candidate.get("model_request_count") != 0 or candidate.get("behavioral_episode_count") != 0:
        raise RuntimeError("fixture candidate is not model blind")
    task_root = root / "experiments/v3/pi05_phase_b/task_files"
    auto_register_droid_abs_ik_envs(task=[str(task_root / TASKS[args.condition][0])], cameras=WRIST_LEFT_RIGHT_HEAD)
    _, task_name = TASKS[args.condition]
    relation = args.condition.split(":", 1)[1]
    offset_inv = qinv(np.asarray(EEF_OFFSET_ROT, dtype=np.float64))
    rows=[]
    def pose_command(target, quat, grip):
        return torch.from_numpy(np.concatenate([target, qmul(quat, offset_inv), [grip]]).astype(np.float32)).reshape(1, 8)
    for seed in range(args.seed_start, args.seed_end + 1):
        env, _ = create_env(task_name, device=args.device, seed=seed, num_envs=1, instruction_type="default", policy="v3e002_model_blind_reference_controller", renderer=args.renderer, rendering_mode=args.rendering_type)
        started=time.time(); invalid=None
        try:
            obs,_=env.reset(); obs,_=env.reset()
            frames=env.scene["frames"]; eef_idx=frames.data.target_frame_names.index("eef_frame")
            quat=arr(frames.data.target_quat_w)[0,eef_idx]
            # Settle at the reset pose with the gripper open.
            p0=arr(frames.data.target_pos_w)[0,eef_idx]
            hold=pose_command(p0, quat, 0.0).to(env.device)
            for _ in range(30): env.step(hold)
            cube=arr(env.scene["rubiks_cube"].data.root_pos_w)[0]; bowl=arr(env.scene["bowl"].data.root_pos_w)[0]
            sign=1.0 if relation == "left" else -1.0
            target=bowl.copy(); target[1]+=sign*args.depth_m
            pre_cube=cube.copy(); pre_cube[2]+=0.12
            grasp=cube.copy(); grasp[2]+=0.025
            lift=pre_cube.copy()
            pre_place=target.copy(); pre_place[2]+=0.12
            place=target.copy(); place[2]+=0.04
            waypoints=[(pre_cube,0.0,"pregrasp"),(grasp,0.0,"grasp_approach"),(grasp,0.785398,"close"),(lift,0.785398,"lift"),(pre_place,0.785398,"preplace"),(place,0.785398,"place"),(place,0.0,"release")]
            path=0.0; min_joint=float("inf"); max_vel=0.0; terminal=False; contact_step=None
            prev=arr(frames.data.target_pos_w)[0,eef_idx]
            for target_pose, grip, label in waypoints:
                command=pose_command(target_pose, quat, grip).to(env.device)
                for _ in range(args.steps_per_waypoint):
                    obs,_,term,trunc,_=env.step(command)
                    now=arr(frames.data.target_pos_w)[0,eef_idx]; path += float(np.linalg.norm(now-prev)); prev=now
                    joints=arr(obs["proprio_obs"]["arm_joint_pos"])[0]; min_joint=min(min_joint,float(np.min(np.abs(joints))))
                    if bool(object_grabbed(env, object="rubiks_cube", env_id=0)): contact_step = contact_step if contact_step is not None else len(rows)
                    if bool(term[0]) or bool(trunc[0]): terminal=True; break
                if terminal: break
            final_cube=arr(env.scene["rubiks_cube"].data.root_pos_w)[0]
            success=bool(object_left_of(env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0)) if relation=="left" else bool(object_right_of(env, object="rubiks_cube", reference_object="bowl", frame_of_reference="robot", mirrored=False, require_gripper_detached=True, env_id=0))
            grabbed=bool(object_grabbed(env, object="rubiks_cube", env_id=0)); dropped=bool(object_dropped(env, object="rubiks_cube", env_id=0))
            failure="correct" if success else ("pick_failed" if not grabbed else ("release_failed" if not dropped else ("wrong_side" if (relation=="left" and final_cube[1] < bowl[1]) or (relation=="right" and final_cube[1] > bowl[1]) else "transport_failed")))
            rows.append({"schema_version":"vla-wam-shared-v3e002-reference-controller-episode-v1","model_request_count":0,"behavioral_episode":True,"seed":seed,"layout":args.condition.split(":")[0],"requested_relation":relation,"prompt": "Put the Rubik's cube to the left of the bowl." if relation=="left" else "Put the Rubik's cube to the right of the bowl.","selected_depth_m":args.depth_m,"success":success,"failure_category":failure,"picked_up":grabbed,"detached_release":dropped,"requested_side_depth_m":float(sign*(final_cube[1]-bowl[1])),"final_endpoint_error_m":float(np.linalg.norm(final_cube-target)),"task_space_path_length_m":path,"min_joint_abs_rad":min_joint,"max_joint_velocity":max_vel,"terminal":terminal,"duration_s":time.time()-started,"pod":args.pod,"gpu_uuid":args.gpu_uuid})
        except Exception as exc:
            invalid={"schema_version":"v3e002-infrastructure-invalid-v1","model_request_count":0,"behavioral_episode":False,"seed":seed,"layout":args.condition.split(":")[0],"requested_relation":relation,"error_type":type(exc).__name__,"error":str(exc),"pod":args.pod,"gpu_uuid":args.gpu_uuid}
        finally: env.close()
        if invalid: rows.append(invalid)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in rows))
    print(json.dumps({"condition":args.condition,"rows":len(rows),"valid":sum(r.get("behavioral_episode") is True for r in rows),"invalid":sum(r.get("behavioral_episode") is not True for r in rows)}, indent=2))
    app.close()

if __name__ == "__main__":
    try: main()
    except Exception:
        app.close(); raise
