#!/usr/bin/env python3
"""Compile six valid V2-A010 pi0.5 current-stack media-gate cells."""

from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
from typing import Any
import h5py
import numpy as np


TASKS={"left":"RubiksCubeLeftOfBowlMatchedTask","right":"RubiksCubeRightOfBowlMatchedTask"}


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(16*1024*1024),b""):h.update(b)
    return h.hexdigest()


def rec(path:Path)->dict[str,Any]:
    if not path.is_file():raise FileNotFoundError(path)
    return {"path":str(path),"bytes":path.stat().st_size,"sha256":sha256(path)}


def rotation_wxyz(q:np.ndarray)->np.ndarray:
    q=np.asarray(q,dtype=np.float64);q/=np.linalg.norm(q,axis=1,keepdims=True);w,x,y,z=q.T;m=np.empty((len(q),3,3))
    m[:,0,0]=1-2*(y*y+z*z);m[:,0,1]=2*(x*y-z*w);m[:,0,2]=2*(x*z+y*w)
    m[:,1,0]=2*(x*y+z*w);m[:,1,1]=1-2*(x*x+z*z);m[:,1,2]=2*(y*z-x*w)
    m[:,2,0]=2*(x*z-y*w);m[:,2,1]=2*(y*z+x*w);m[:,2,2]=1-2*(x*x+y*y);return m


def main()->None:
    p=argparse.ArgumentParser();p.add_argument("--raw-root",type=Path,required=True);p.add_argument("--action-trace-root",type=Path,required=True);p.add_argument("--registry",type=Path,required=True);p.add_argument("--checkpoint-manifest",type=Path,required=True);p.add_argument("--study-commit",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    registry=json.loads(a.registry.read_text())
    if registry.get("schema_version")!="vla-wam-v2a010-pi05-current-stack-registry-v1" or len(registry["cells"])!=6:raise ValueError("not V2-A010 registry")
    episodes=[];actions={};initial={}
    for cell in registry["cells"]:
        seed=cell["environment_seed"];relation=cell["requested_relation"];task=TASKS[relation];run=a.raw_root/cell["output_folder_name"];task_root=run/task
        results=[json.loads(x) for x in (run/"episode_results.jsonl").read_text().splitlines() if x.strip()]
        if len(results)!=1 or results[0]["instruction"]!=cell["rendered_prompt"]:raise ValueError(f"result mismatch {cell['cell_id']}")
        env=json.loads((task_root/"env_cfg.json").read_text());log=json.loads((task_root/"log_0_env0.json").read_text())
        if env["instruction"]!=cell["rendered_prompt"] or int(env["seed"])!=seed or bool(log["success"])!=bool(results[0]["success"]):raise ValueError(f"provenance mismatch {cell['cell_id']}")
        videos=sorted(task_root.glob("*_viewport.mp4"))
        if len(videos)!=1:raise ValueError(f"video mismatch {cell['cell_id']}")
        stem=cell["action_trace_stem"];trace_path=a.action_trace_root/f"{stem}_action_trace.json";trace=json.loads(trace_path.read_text());action_path=Path(trace["executed_actions"]["path"]);chunks_path=Path(trace["returned_action_chunks"]["path"])
        act=np.load(action_path,allow_pickle=False);chunks=np.load(chunks_path,allow_pickle=False)
        if sha256(action_path)!=trace["executed_actions"]["sha256"] or sha256(chunks_path)!=trace["returned_action_chunks"]["sha256"] or trace["request_sampling_seeds"]!=[seed*1000+i for i in range(len(chunks))]:raise ValueError(f"trace mismatch {cell['cell_id']}")
        h5_path=task_root/"run_0.hdf5"
        with h5py.File(h5_path) as h5:
            d=h5["data/demo_0"];h5act=np.asarray(d["actions"],dtype=np.float32);cube=np.asarray(d["states/rigid_object/rubiks_cube/root_pose"]);bowl=np.asarray(d["states/rigid_object/bowl/root_pose"]);robot=np.asarray(d["states/articulation/robot/root_pose"]);ic=np.asarray(d["initial_state/rigid_object/rubiks_cube/root_pose"][-1]);ib=np.asarray(d["initial_state/rigid_object/bowl/root_pose"][-1])
        if not np.array_equal(act,h5act):raise ValueError(f"action mismatch {cell['cell_id']}")
        delta=np.einsum("tij,ti->tj",rotation_wxyz(robot[:,3:7]),cube[:,:3]-bowl[:,:3]);initial[(seed,relation)]=(ic,ib);actions[(seed,relation)]=act
        episodes.append({"cell_id":cell["cell_id"],"environment_seed":seed,"sampling_seed_base":seed,"prompt_family":"direct_command","requested_relation":relation,"prompt":cell["rendered_prompt"],"success":bool(results[0]["success"]),"executed_action_count":len(act),"endpoint_lateral_display_m":float(-delta[-1,1]),"files":{"episode_results":rec(run/"episode_results.jsonl"),"environment":rec(task_root/"env_cfg.json"),"episode_log":rec(task_root/"log_0_env0.json"),"trajectory":rec(h5_path),"viewport_video":rec(videos[0]),"action_trace":rec(trace_path),"executed_actions":rec(action_path),"returned_action_chunks":rec(chunks_path)}})
    pairs=[]
    for seed in (8300,8301,8302):
        for i,name in enumerate(("cube","bowl")):
            if not np.array_equal(initial[(seed,"left")][i],initial[(seed,"right")][i]):raise ValueError(f"paired {name} reset differs {seed}")
        left=next(x for x in episodes if x["environment_seed"]==seed and x["requested_relation"]=="left");right=next(x for x in episodes if x["environment_seed"]==seed and x["requested_relation"]=="right");overlap=min(10,len(actions[(seed,"left")]),len(actions[(seed,"right")]));d=actions[(seed,"left")][:overlap]-actions[(seed,"right")][:overlap];shift=right["endpoint_lateral_display_m"]-left["endpoint_lateral_display_m"]
        pairs.append({"environment_seed":seed,"left_success":left["success"],"right_success":right["success"],"right_minus_left_endpoint_shift_m":shift,"endpoint_ordering_aligned":shift>0,"first_ten_executed_action_rms":float(np.sqrt(np.mean(d**2))),"executed_actions_distinct":bool(np.any(d!=0))})
    result={"schema_version":"vla-wam-v2a010-pi05-current-result-v1","amendment_id":"V2-A010","status":"complete_6_of_6_valid_current_stack_cells","claim_boundary":registry["claim_boundary"],"openpi_commit":"c23745b5ad24e98f66967ea795a07b2588ed6c79","robolab_commit":"0aef241fb088ca21bb4ebd24448940ed56620d17","openpi_config":"pi05_droid_jointpos_polaris","study_commit":a.study_commit,"registry":rec(a.registry),"checkpoint_manifest":rec(a.checkpoint_manifest),"valid_episode_count":6,"episodes":episodes,"pairs":pairs,"summary":{"left_successes":sum(x["success"] for x in episodes if x["requested_relation"]=="left"),"right_successes":sum(x["success"] for x in episodes if x["requested_relation"]=="right"),"aligned_endpoint_pair_count":sum(x["endpoint_ordering_aligned"] for x in pairs),"distinct_executed_action_pair_count":sum(x["executed_actions_distinct"] for x in pairs)},"infrastructure_invalid_attempts":[]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n");print(json.dumps(result["summary"],indent=2))


if __name__=="__main__":main()
