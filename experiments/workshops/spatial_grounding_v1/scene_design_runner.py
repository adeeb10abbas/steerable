"""Measure one revised scene, then run the existing six-trial SGW qualification.

One fresh process per scene. No learned-policy imports or cluster access.
The output contains native capture, unchanged raw trial records and videos.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import traceback

from .fixtures import FixtureCandidate
from .scene_design import write_scene
from .recorder import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--design',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--robolab-root',type=Path,required=True)
    parser.add_argument('--assets-manifest',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--capture-only',action='store_true')
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.headless or args.num_envs != 1 or args.renderer != 'realtime' or args.rendering_type != 'balanced':
        raise ValueError('Use the pinned single-environment realtime/balanced configuration')
    commit=subprocess.check_output(['git','-C',str(args.robolab_root),'rev-parse','HEAD'],text=True).strip()
    if commit != '0aef241fb088ca21bb4ebd24448940ed56620d17':
        raise ValueError('RoboLab source does not match the study')
    row=json.loads(args.design.read_bytes()); workspace=json.loads(args.workspace.read_bytes())
    args.output.mkdir(parents=True)
    scene=write_scene(row,workspace,args.robolab_root,args.output/'scene.usda')
    launch={'design':row,'scene':scene,'design_sha256':hashlib.sha256(args.design.read_bytes()).hexdigest(),
            'robolab_commit':commit,'model_requests':0,
            'versions':{n:importlib.metadata.version(n) for n in ('isaacsim','isaaclab','torch')},
            'asset_manifest_sha256':hashlib.sha256(args.assets_manifest.read_bytes()).hexdigest(),
            'calibration_sha256':hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
            'workspace_sha256':hashlib.sha256(args.workspace.read_bytes()).hexdigest(),
            'source_sha256':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ('scene_design.py','scene_design_task.py','scene_design_runner.py')},
            'visual_style':row.get('visual_style','original-office'),
            'capture_only':args.capture_only}
    atomic_json(args.output/'input.json',launch)
    os.environ['SGW_SCENE_DESIGN_INPUT']=str((args.output/'input.json').resolve())
    os.environ['SGW_SCENE_DESIGN_SHA256']=hashlib.sha256((args.output/'input.json').read_bytes()).hexdigest()
    args.enable_cameras=True
    app=AppLauncher(args).app
    try:
        import numpy as np
        import robolab.constants
        from robolab.constants import set_output_dir
        from robolab.core.environments.runtime import create_env
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        from .lat_workspace_capture import render_only_warmup,_root_local_offset
        from .robolab_lat_qualification import RoboLabLatEnvironment,RoboLabLatScriptedController
        from .robolab_height_dist_qualification import RoboLabFamilyScriptedController
        from .model_blind_qualification import qualify_candidate
        from .native_geometry_measurements import robot_snapshot
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING=False
        robolab.constants.RECORD_IMAGE_DATA=False
        robolab.constants.VERBOSE=False
        set_output_dir(str(args.output/'native'))
        task=Path(__file__).with_name('scene_design_task.py')
        auto_register_droid_abs_ik_envs(task=[str(task)],cameras=WRIST_LEFT_RIGHT_HEAD)
        scene_config='SGWSceneDesignTask'
        if row.get('visual_style') == 'clean-studio-v1':
            from robolab.core.environments.config import parse_env_cfg
            scene_config=parse_env_cfg('SGWSceneDesignTask',device=args.device,seed=row.get('seed',20260923),num_envs=1)
            scene_config.scene.dome_light.spawn.texture_file=''
            scene_config.scene.dome_light.spawn.color=(.65,.67,.70)
            scene_config.scene.dome_light.spawn.intensity=500.0
            scene_config.scene.dome_light.spawn.visible_in_primary_ray=True
        env,_=create_env(scene_config,device=args.device,seed=row.get('seed',20260923),num_envs=1,
                         instruction_type='default',policy='sgw_scripted_scene_design',renderer='realtime',rendering_mode='balanced')
        obs,_=env.reset()
        obs,warmup=render_only_warmup(env,obs,120,args.output/'capture_warmup')
        world=get_world(env); objects={}
        for name in scene['object_names']:
            p,q=world.get_pose(name,env_id=0); corners,c=world.get_bbox(name,env_id=0)
            p=p.detach().cpu().tolist(); q=q.detach().cpu().tolist(); c=np.asarray(c).tolist()
            bounds=np.asarray([v.detach().cpu().numpy() if hasattr(v,'detach') else v for v in corners])
            objects[name]={'root_position_env_local_xyz_m':p,'root_quaternion_world_wxyz':q,
                           'geometric_center_env_local_xyz_m':c,
                           'geometric_center_offset_root_local_xyz_m':_root_local_offset(p,q,c),
                           'bbox_env_local_min_xyz_m':bounds.min(axis=0).tolist(),
                           'bbox_env_local_max_xyz_m':bounds.max(axis=0).tolist()}
        origin=env.scene.env_origins[0].detach().cpu().numpy()
        cameras={}
        from PIL import Image
        for name in ('over_shoulder_left_camera','over_shoulder_right_camera','wrist_cam'):
            Image.fromarray(obs['image_obs'][name][0].detach().cpu().numpy()).save(args.output/f'{name}.png')
            cam=env.scene[name].data
            cameras[name]={'position_world_m':cam.pos_w[0].detach().cpu().tolist(),
                           'quaternion_world_wxyz':cam.quat_w_world[0].detach().cpu().tolist()}
        capture={'objects':objects,'robot':robot_snapshot(env.scene,origin),'cameras':cameras,
                 'warmup':warmup,'model_requests':0,'scene':scene}
        atomic_json(args.output/'capture.json',capture)
        if args.capture_only:
            env.close(); return
        offsets={n:objects[n]['geometric_center_offset_root_local_xyz_m'] for n in ('rubiks_cube','bowl')}
        meta={'scoring_center_offsets_root_local_m':offsets,
              'native_scene':{'asset':scene['path'],'asset_sha256':scene['sha256'],'object_names':scene['object_names']},
              'prospective_design_id':row['design_id'],
              'candidate_capture_sha256':hashlib.sha256((args.output/'capture.json').read_bytes()).hexdigest()}
        if row['family']=='LAT':
            meta['abs_ik_waypoints']={label:[{'position_world_xyz_m':(np.asarray(row['targets'][str(s)])+origin).tolist()}]
                                    for label,s in [('positive',1),('negative',-1)]}
        else:
            cube_half=objects['rubiks_cube']['geometric_center_env_local_xyz_m'][2]-objects['rubiks_cube']['bbox_env_local_min_xyz_m'][2]
            meta['goal_supports']={}
            for key,name in [('higher','height_upper_support'),('lower','height_lower_support')]:
                measured=objects[name]
                target=[*measured['geometric_center_env_local_xyz_m'][:2],measured['bbox_env_local_max_xyz_m'][2]+cube_half]
                meta['goal_supports'][key]={'contact_sensor_id':f'rubiks_cube__{name}','cube_center_env_local_xyz_m':target}
            meta['baseline_banana_pose']={'position_m':objects['banana']['root_position_env_local_xyz_m'],
                                          'quaternion_wxyz':objects['banana']['root_quaternion_world_wxyz']}
            meta['geometry_guard_support_ids']=[s['name'] for s in row['supports']]
        candidate=FixtureCandidate.from_json({'candidate_id':row['design_id'],'family':row['family'],
            'seed':row.get('seed',20260923),'asset_manifest_sha256':launch['asset_manifest_sha256'],
            'task_asset':scene['path'],'object_poses':{n:{'position_m':objects[n]['root_position_env_local_xyz_m'],
                   'quaternion_wxyz':objects[n]['root_quaternion_world_wxyz']} for n in ('rubiks_cube','bowl')},'metadata':meta})
        atomic_json(args.output/'candidate.json',asdict(candidate))
        wrapper=RoboLabLatEnvironment(env,candidate,args.output/'reset_warmup')
        class Bridge:
            def create_environment(self,task,seed): return wrapper
        controller=(RoboLabLatScriptedController if row['family']=='LAT' else RoboLabFamilyScriptedController)(args.calibration)
        receipt=qualify_candidate(candidate,Bridge(),controller,seed=candidate.seed,evidence_root=args.output/'trials')
        atomic_json(args.output/'qualification.json',receipt)
        print(json.dumps({'id':candidate.candidate_id,'status':receipt['status'],'passed':sum(c['passed'] for c in receipt['checks'])}),flush=True)
    except BaseException:
        atomic_json(args.output/'infrastructure_failure.json',{'error':traceback.format_exc(),'model_requests':0})
        raise
    finally:
        app.close()


if __name__=='__main__': main()
