#!/usr/bin/env python3
"""Compile one E003 simulator export without altering frozen predicates."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from typing import Any

import numpy as np

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def file_record(value: Any) -> dict[str, Any]:
    p=Path(str(value)).resolve()
    if not p.is_file() or p.stat().st_size<=0: raise RuntimeError(f"missing artifact {p}")
    return {"path":str(p),"bytes":p.stat().st_size,"sha256":sha(p)}

def cone(step: dict[str,Any], relation: str) -> bool:
    d=np.asarray(step["object_xyz"],float)-np.asarray(step["reference_xyz"],float)
    r=math.hypot(float(d[0]),float(d[1])); margin=float(d[1]) if relation=="left" else -float(d[1])
    return r>1e-8 and margin/r>=math.cos(math.radians(45))

def category(exp: dict[str,Any], steps: list[dict[str,Any]], relation: str) -> str:
    if bool(exp["requested_success"]): return "correct"
    if not any(bool(x["object_grabbed"]) for x in steps): return "pick_failed"
    if not bool(exp.get("final_detached_release")): return "release_failed"
    if not any(cone(x,relation) for x in steps): return "wrong_side"
    return "transport_failed"

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--registration",type=Path,required=True); ap.add_argument("--registration-sha256",required=True); ap.add_argument("--runtime",type=Path,required=True); ap.add_argument("--export",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    if sha(a.registration)!=a.registration_sha256: raise RuntimeError("registration digest mismatch")
    reg=json.loads(a.registration.read_text()); rows={x["cell_id"]:x for x in reg["queue"]}; exp=json.loads(a.export.read_text()); cid=exp["registered_cell_id"]
    if cid not in rows: raise RuntimeError("export cell absent from registration")
    row=rows[cid]; steps=exp["steps"]; rel=row["relation"]
    if len(steps)<2 or any(x.get("action_step")!=i for i,x in enumerate(steps)): raise RuntimeError("non-contiguous state capture")
    lateral=[float(x["object_xyz"][1])-float(x["reference_xyz"][1]) for x in steps]; final=lateral[-1]
    depth=final if rel=="left" else -final
    requested=[cone(x,rel) for x in steps]
    sustained=any(all(requested[i:i+3]) for i in range(max(0,len(requested)-2)))
    pickup=next((i for i,x in enumerate(steps) if x.get("object_grabbed")),None)
    record={
      "schema_version":"vla-wam-shared-v3e003-behavioral-episode-v1","record_type":"behavioral_episode","behavioral_result_valid":True,"study_id":reg["study_id"],"amendment_id":"V3-E003","phase":"E_publication_critical_controls","registered_cell_id":cid,"matched_block_id":row["matched_block_id"],"model_id":"pi05_current_stack_droid","arena":"droid_robolab","environment_seed":row["environment_seed"],"sampling_seed":row["sampling_seed"],"requested_relation":rel,"prompt":row["prompt"],"prompt_sha256":row["prompt_sha256"],"success":bool(exp["requested_success"]),"requested_success":bool(exp["requested_success"]),"failure_category":category(exp,steps,rel),"failure_taxonomy":category(exp,steps,rel),"signed_final_lateral_offset_m":final,"requested_side_depth_m":depth,"cone_entry_step":next((i for i,v in enumerate(requested) if v),None),"cone_entry_sustained":sustained,"episode_length_steps":len(steps)-1,"time_to_first_contact_steps":None,"grasp_step":pickup,"cumulative_lateral_path_m":sum(abs(b-a) for a,b in zip(lateral,lateral[1:])),"peak_lateral_excursion_m":max(abs(x-lateral[0]) for x in lateral),"endpoint_shift_m":None,"action_distinct":None,"symmetry_residual":json.loads((a.registration.parent/"symmetry_gate/candidate.json").read_text())["symmetry_residual"],"final_detached_release":bool(exp.get("final_detached_release")),"right_censored":bool(exp["right_censored"]),"actions_executed":len(steps)-1,"action_cap":450,"initial_state_sha256":exp["initial_state_sha256"],"runtime_identity_sha256":json.loads(a.runtime.read_text())["runtime_identity_sha256"],"release_fingerprint_sha256":exp["release_fingerprint_sha256"],"artifacts":{"viewport_video":file_record(exp["viewport_video_path"]),"executed_action_trace":file_record(exp["executed_action_trace"]["path"] if isinstance(exp["executed_action_trace"],dict) else exp["executed_action_trace"]),"action_trace_metadata":file_record(exp["action_trace_metadata_path"]),"reset_attestation":file_record(exp["reset_attestation_path"])},"steps":steps,"future_interface":"actions_only","future_evidence_status":"not_exposed_by_action_only_interface","missing_future_policy":"action_only_interface_not_applicable_never_zero"
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(record,sort_keys=True)+"\n")
    (a.output.with_name(a.output.name+".manifest.json")).write_text(json.dumps({"schema_version":"vla-wam-shared-v3e003-episode-manifest-v1","row_count":1,"jsonl_sha256":sha(a.output),"bytes":a.output.stat().st_size},indent=2,sort_keys=True)+"\n")
    print(json.dumps({"cell_id":cid,"failure_category":record["failure_category"],"success":record["success"]},indent=2))
if __name__=="__main__": main()
