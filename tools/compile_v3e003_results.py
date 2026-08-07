#!/usr/bin/env python3
"""Compile compact statistics from completed E003 raw episode JSONL files."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from statistics import mean, median

def wilson(k,n,z=1.959963984540054):
    if not n: return {"estimate": None, "low": None, "high": None}
    p=k/n; d=1+z*z/n
    c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return {"estimate":p,"low":max(0,c-h),"high":min(1,c+h)}
def sha(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def boot(values, fn, reps=20000, seed=20260807):
    if not values:return {"estimate":None,"low":None,"high":None,"n":0}
    import random
    r=random.Random(seed); vals=[]; n=len(values)
    for _ in range(reps): vals.append(fn([values[r.randrange(n)] for _ in range(n)]))
    vals.sort(); return {"estimate":fn(values),"low":vals[int(.025*reps)],"high":vals[int(.975*reps)-1],"n":n,"resamples":reps}
def sign_test(vals):
    pos=sum(x>0 for x in vals); neg=sum(x<0 for x in vals); tie=sum(x==0 for x in vals); m=pos+neg
    p=1.0 if not m else min(1.0,2*sum(math.comb(m,i) for i in range(min(pos,neg)+1))/(2**m))
    return {"positive":pos,"negative":neg,"zero":tie,"p_two_sided":p}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--raw-root",type=Path,required=True); ap.add_argument("--registration",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    reg=json.loads(a.registration.read_text()); base=a.raw_root/"V3-E003_pi05_symmetric"; rows=[]; sources=[]
    for p in sorted(base.glob("**/raw_episode.jsonl")):
        try: v=json.loads(p.read_text().splitlines()[0])
        except Exception: continue
        if not v.get("behavioral_result_valid"): continue
        v["source_raw_episode_sha256"]=sha(p); v["source_raw_episode_path"]=str(p); rows.append(v); sources.append({"path":str(p),"sha256":v["source_raw_episode_sha256"],"bytes":p.stat().st_size})
    rows.sort(key=lambda x:(x["environment_seed"],x["requested_relation"]))
    by={int(v["environment_seed"]):{} for v in rows}
    for v in rows: by[int(v["environment_seed"])][v["requested_relation"]]=v
    pairs=[by[s] for s in sorted(by) if set(by[s])=={"left","right"}]
    left=[p["left"] for p in pairs]; right=[p["right"] for p in pairs]
    gap=[int(r["success"])-int(l["success"]) for l,r in zip(left,right)]
    depth=[float(r["requested_side_depth_m"])-float(l["requested_side_depth_m"]) for l,r in zip(left,right)]
    endpoint=[float(r["signed_final_lateral_offset_m"])-float(l["signed_final_lateral_offset_m"]) for l,r in zip(left,right)]
    def counts(vs):
        out={k:sum(1 for v in vs if v.get("failure_category")==k or (k=="correct" and v.get("success"))) for k in ["pick_failed","transport_failed","wrong_side","release_failed","correct"]}; return out
    result={"schema_version":"vla-wam-shared-v3e003-results-v1","registered_cells":len(reg["queue"]),"valid_behavioral_episodes":len(rows),"source_files":sources,"directions":{"left":{"n":len(left),"successes":sum(v["success"] for v in left),"wilson_95":wilson(sum(v["success"] for v in left),len(left)),"failure_taxonomy":counts(left)},"right":{"n":len(right),"successes":sum(v["success"] for v in right),"wilson_95":wilson(sum(v["success"] for v in right),len(right)),"failure_taxonomy":counts(right)}},"paired":{"n":len(pairs),"success_gap_right_minus_left":sum(gap)/len(gap) if gap else None,"mcnemar":{"left_success_right_failure":sum(l["success"] and not r["success"] for l,r in zip(left,right)),"left_failure_right_success":sum((not l["success"]) and r["success"] for l,r in zip(left,right))},"depth_contrast_right_minus_left_m":{"mean":boot(depth,mean),"median":boot(depth,median),"sign_test":sign_test(depth),"values_m":depth},"endpoint_shift_right_minus_left_m":{"mean":boot(endpoint,mean),"median":boot(endpoint,median),"values_m":endpoint},"success_gap_values":gap},"symmetry_residual":{"max_m":max((v.get("symmetry_residual",{}).get("max_m",float("nan")) for v in rows),default=None),"all_within_1mm":all(v.get("symmetry_residual",{}).get("max_m",1)>-1 and v.get("symmetry_residual",{}).get("max_m",1)<=.001 for v in rows)},"equivalence_margins":reg["equivalence_margins"],"status":"complete" if len(rows)==54 and len(left)==27 and len(right)==27 else "incomplete"}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":result["status"],"valid_behavioral_episodes":len(rows),"left":len(left),"right":len(right)},indent=2))
if __name__=="__main__": main()
