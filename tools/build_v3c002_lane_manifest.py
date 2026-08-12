#!/usr/bin/env python3
"""Bind one independently isolated C002 lane after the two-lane gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
REPO_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError,file_binding,read_finite_json,repo_file_binding,sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import validate_runtime_manifest
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--registration",type=Path,required=True);p.add_argument("--queue",type=Path,required=True);p.add_argument("--runtime-manifest",type=Path,required=True);p.add_argument("--runtime-manifest-sha256",required=True);p.add_argument("--isolation-gate",type=Path,required=True);p.add_argument("--lane-pod-uid",required=True);p.add_argument("--lane-gpu-uuid",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ContractError(f"refusing to overwrite lane manifest: {a.output}")
    isolation=read_finite_json(a.isolation_gate)
    if not isinstance(isolation,dict) or isolation.get("status")!="passed_two_lane_fixed_observation_isolation" or isolation.get("passed") is not True:raise ContractError("two-lane isolation has not passed")
    runtime=validate_runtime_manifest(a.runtime_manifest,a.runtime_manifest_sha256,registration_path=a.registration,queue_path=a.queue,pod_uid=a.lane_pod_uid,gpu_uuid=a.lane_gpu_uuid)["runtime_identity"]
    registration=read_finite_json(a.registration);exact=registration["exact_e004_pi05_runtime"]
    value={"schema_version":"vla-wam-shared-v3c002-lane-release-manifest-v1","status":"passed_lane_release","passed":True,
        "registration_sha256":sha256_file(a.registration),"queue_sha256":sha256_file(a.queue),"runtime_manifest":file_binding(a.runtime_manifest),"two_lane_isolation_gate":repo_file_binding(a.isolation_gate),"source_commit":runtime["source_commit"],"exact_runtime_contract_sha256":exact["contract_sha256"]}
    for key in ("lane_id","simulator_pod_uid","simulator_gpu_uuid","policy_server_pod_uid","policy_server_gpu_uuid","server_port","raw_root","container_identity","runtime_identity","server_process_identity","server_lock_identity"):value[key]=runtime[key]
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,allow_nan=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"status":value["status"],"sha256":sha256_file(a.output)},sort_keys=True))
if __name__=="__main__":main()
