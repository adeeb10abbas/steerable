#!/usr/bin/env python3
"""Capture V2-A010 neutral fixed observation before pi0.5 behavioral inference."""

from __future__ import annotations

import argparse, hashlib, json, subprocess, traceback
from pathlib import Path
import cv2  # noqa: F401
import numpy as np
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument("--study-root",type=Path,required=True)
parser.add_argument("--robolab-root",type=Path,required=True)
parser.add_argument("--registry",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
from robolab.eval.runner import add_common_eval_args  # noqa: E402
add_common_eval_args(parser); AppLauncher.add_app_launcher_args(parser)
args_cli,_=parser.parse_known_args(); args_cli.enable_cameras=True
if args_cli.num_envs != 1: parser.error("V2-A010 fixture requires one environment")
app_launcher=AppLauncher(args_cli); simulation_app=app_launcher.app
import robolab.constants  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.task.conditionals import object_left_of, object_right_of  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from policies.pi0_family.client import Pi0DroidJointposClient  # noqa: E402


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def main()->None:
    head=subprocess.check_output(["git","-C",str(args_cli.robolab_root),"rev-parse","HEAD"],text=True).strip()
    if head!="0aef241fb088ca21bb4ebd24448940ed56620d17": raise ValueError(head)
    registry=json.loads(args_cli.registry.read_text())
    cell=next(row for row in registry["cells"] if row["environment_seed"]==8300 and row["requested_relation"]=="left")
    task_path=args_cli.study_root/"experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py"
    args_cli.output_dir.mkdir(parents=True,exist_ok=True);set_output_dir(str(args_cli.output_dir))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING=False
    auto_register_droid_envs(task=[str(task_path)])
    env,env_cfg=create_env(cell["anchor_task"],device=args_cli.device,seed=8300,num_envs=1,instruction_type="default",policy="pi05_v2a010_fixture",renderer=args_cli.renderer,rendering_mode=args_cli.rendering_type)
    try:
        env.reset();obs,_=env.reset()
        if env_cfg.instruction!=cell["rendered_prompt"]:raise ValueError("prompt mismatch")
        left=bool(object_left_of(env,object="rubiks_cube",reference_object="bowl",frame_of_reference="robot",mirrored=False,require_gripper_detached=True,env_id=0))
        right=bool(object_right_of(env,object="rubiks_cube",reference_object="bowl",frame_of_reference="robot",mirrored=False,require_gripper_detached=True,env_id=0))
        if left or right:raise ValueError("reset is not neutral")
        helper=object.__new__(Pi0DroidJointposClient)
        request=helper._pack_request(helper._extract_observation(obs,env_id=0),cell["rendered_prompt"])
        if request.pop("prompt")!=cell["rendered_prompt"]:raise ValueError("packed prompt mismatch")
        arrays={key:np.asarray(value) for key,value in request.items()}
        fixture=args_cli.output_dir/"seed8300_fixed_observation.npz";np.savez(fixture,**arrays)
        manifest={"schema_version":"vla-wam-v2a010-pi05-current-fixed-observation-v1","registry_path":str(args_cli.registry),"registry_sha256":sha256(args_cli.registry),"robolab_commit":head,"environment_seed":8300,"task":cell["anchor_task"],"prompt":cell["rendered_prompt"],"reset_count":2,"neutral_reset_contract":{"left_predicate_at_reset":left,"right_predicate_at_reset":right},"fixture_path":str(fixture),"fixture_sha256":sha256(fixture),"arrays":{k:{"shape":list(v.shape),"dtype":str(v.dtype)} for k,v in sorted(arrays.items())}}
        path=args_cli.output_dir/"seed8300_fixed_observation.json";path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n");print(json.dumps(manifest,indent=2))
    finally:
        env.close();simulation_app.close()


if __name__=="__main__":
    try:main()
    except Exception as exc:
        print(f"[pi0.5 V2-A010 fixture] technical failure: {exc}");traceback.print_exc();simulation_app.close();raise
