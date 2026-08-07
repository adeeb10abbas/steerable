"""Static symmetric Abs-IK target sweep for V3-E002.

The gate sends no learned-model request. It only exercises RoboLab's verified
absolute differential-IK action manager at the preregistered target depths and
records finite joint/waypoint outcomes before any controller queue is enabled.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--study-root", type=Path, required=True); ap.add_argument("--robolab-root", type=Path, required=True)
ap.add_argument("--candidate", type=Path, required=True); ap.add_argument("--candidate-sha256", required=True)
ap.add_argument("--output", type=Path, required=True); ap.add_argument("--pod", required=True); ap.add_argument("--gpu-uuid", required=True)
from robolab.eval.runner import add_common_eval_args
add_common_eval_args(ap); AppLauncher.add_app_launcher_args(ap)
args, _ = ap.parse_known_args(); args.enable_cameras = True
root = args.study_root.resolve(); sys.path.insert(0, str(root))
os.environ["VLA_WAM_V3B_FIXTURE_CANDIDATE"] = str(args.candidate.resolve()); os.environ["VLA_WAM_V3B_FIXTURE_SHA256"] = args.candidate_sha256
app = AppLauncher(args).app
from robolab.core.environments.runtime import create_env
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
from robolab.core.world.world_state import get_world

TASKS = {("control","left"): ("control_left.py", "V3B002Pi05ControlLeftTask"), ("control","right"): ("control_right.py", "V3B002Pi05ControlRightTask"), ("position_mirrored","left"): ("position_mirrored_left.py", "V3B002Pi05PositionMirroredLeftTask"), ("position_mirrored","right"): ("position_mirrored_right.py", "V3B002Pi05PositionMirroredRightTask")}
DEPTHS = (0.075, 0.100, 0.150, 0.200)

def arr(x):
    if hasattr(x, "detach"): x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)

def main():
    args.output.parent.mkdir(parents=True, exist_ok=False)
    task_root = root / "experiments/v3/pi05_phase_b/task_files"
    auto_register_droid_abs_ik_envs(task=[str(task_root / v[0]) for v in TASKS.values()], cameras=WRIST_LEFT_RIGHT_HEAD)
    rows=[]
    for (layout, relation), (_, task_name) in TASKS.items():
        env, cfg = create_env(task_name, device=args.device, seed=9400, num_envs=1, instruction_type="default", policy="v3e002_model_blind_ik_gate", renderer=args.renderer, rendering_mode=args.rendering_type)
        try:
            obs,_=env.reset(); obs,_=env.reset()
            eef=arr(obs["proprio_obs"]["eef_pos"])[0]; quat=arr(obs["proprio_obs"]["eef_quat"])[0]
            bowl=arr(env.scene["bowl"].data.root_pos_w)[0]
            for depth in DEPTHS:
                for requested in ("left","right"):
                    sign = 1.0 if requested == "left" else -1.0
                    target = bowl.copy(); target[1] += sign * depth; target[2] += 0.12
                    command = np.concatenate([target, quat, [0.0]]).astype(np.float32)
                    finite=True; limit_margin=float("inf"); errors=[]
                    for _ in range(5):
                        obs,_,term,trunc,_=env.step(torch.from_numpy(command).to(env.device).reshape(1,-1))
                        joints=arr(obs["proprio_obs"]["arm_joint_pos"])[0]
                        finite = finite and bool(np.isfinite(joints).all()); limit_margin=min(limit_margin,float(np.min(np.abs(joints))))
                        errors.append(float(np.linalg.norm(arr(obs["proprio_obs"]["eef_pos"])[0]-target)))
                    rows.append({"layout":layout,"task_relation":relation,"requested_relation":requested,"depth_m":depth,"ik_finite":finite,"waypoint_endpoint_error_m":errors[-1],"min_joint_abs_rad":limit_margin,"symmetric_target_rule":"bowl_xyz + signed robot-Y depth; fixed z offset 0.12m","learned_model_requests":0})
        finally: env.close()
    value={"schema_version":"vla-wam-shared-v3e002-model-blind-ik-gate-v1","amendment_id":"V3-E002","pod":args.pod,"gpu_uuid":args.gpu_uuid,"candidate_sha256":args.candidate_sha256,"candidate_depths_m":list(DEPTHS),"rows":rows,"passed":bool(rows and all(r["ik_finite"] for r in rows)),"claim_boundary":"Finite Abs-IK target-manager sweep only; no behavioral success is inferred."}
    args.output.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps(value,indent=2,sort_keys=True))
    app.close()

if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"schema_version":"v3e002-ik-gate-failure-v1","error_type":type(exc).__name__,"error":str(exc)},indent=2)); raise
