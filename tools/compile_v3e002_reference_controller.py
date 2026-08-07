#!/usr/bin/env python3
"""Compile the model-blind V3-E002 controller queue into compact evidence."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from statistics import mean, median

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--episodes",type=Path,required=True); ap.add_argument("--gate",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); args=ap.parse_args()
    rows=[]; sources=[]
    for p in sorted(args.episodes.glob("*.jsonl")):
        sources.append({"path":str(p),"sha256":digest(p),"bytes":p.stat().st_size})
        for line in p.read_text().splitlines():
            if line.strip(): rows.append(json.loads(line))
    valid=[r for r in rows if r.get("behavioral_episode") is True]
    invalid=[r for r in rows if r.get("behavioral_episode") is not True]
    by={}
    for r in valid: by[(r["layout"],r["requested_relation"],int(r["seed"]))]=r
    cells={}
    for layout in ("control","position_mirrored"):
        for relation in ("left","right"):
            g=[r for r in valid if r["layout"]==layout and r["requested_relation"]==relation]
            cells[f"{layout}/{relation}"]={
                "episodes":len(g), "successes":sum(bool(r["success"]) for r in g),
                "success_rate":(sum(bool(r["success"]) for r in g)/len(g) if g else None),
                "mean_requested_side_depth_m":(mean(r["requested_side_depth_m"] for r in g) if g else None),
                "mean_endpoint_error_m":(mean(r["final_endpoint_error_m"] for r in g) if g else None),
                "mean_path_length_m":(mean(r["task_space_path_length_m"] for r in g) if g else None),
                "median_min_joint_abs_rad":(median(r["min_joint_abs_rad"] for r in g) if g else None),
                "failure_categories":{k:sum(r["failure_category"]==k for r in g) for k in ("correct","pick_failed","transport_failed","wrong_side","release_failed")},
            }
    pairs=[]
    for layout in ("control","position_mirrored"):
        for seed in range(9400,9427):
            l=by.get((layout,"left",seed)); r=by.get((layout,"right",seed))
            if l and r:
                pairs.append({"layout":layout,"seed":seed,"success_right_minus_left":int(r["success"])-int(l["success"]),"depth_right_minus_left_m":r["requested_side_depth_m"]-l["requested_side_depth_m"],"endpoint_error_right_minus_left_m":r["final_endpoint_error_m"]-l["final_endpoint_error_m"],"path_right_minus_left_m":r["task_space_path_length_m"]-l["task_space_path_length_m"]})
    interactions={}
    for field in ("success_right_minus_left","depth_right_minus_left_m","endpoint_error_right_minus_left_m","path_right_minus_left_m"):
        c=[p[field] for p in pairs if p["layout"]=="control"]; m=[p[field] for p in pairs if p["layout"]=="position_mirrored"]
        interactions[field]={"control_mean":(mean(c) if c else None),"reflected_mean":(mean(m) if m else None),"interaction":(mean(m)-mean(c) if c and m else None),"control_median":(median(c) if c else None),"reflected_median":(median(m) if m else None)}
    gate=json.loads(args.gate.read_text())
    report={"schema_version":"vla-wam-shared-v3e002-results-v1","amendment_id":"V3-E002","status":"complete" if len(valid)==108 and gate.get("passed") is True else "partial","model_request_count":0,"registered_behavioral_episode_count":108,"behavioral_episode_count":len(valid),"infrastructure_invalid_count":len(invalid),"gate":gate,"cells":cells,"matched_pairs":pairs,"interactions":interactions,"source_files":sources,"claim_boundary":"Model-blind deterministic controller diagnostic; no learned-policy success is inferred."}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps({k:report[k] for k in ("status","behavioral_episode_count","infrastructure_invalid_count")},indent=2))
if __name__=="__main__": main()
