#!/usr/bin/env python3
"""Compile two independent fixed-observation lane responses into the C002 isolation gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding, repo_file_binding, sha256_file
def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--registration",type=Path,required=True);p.add_argument("--queue",type=Path,required=True);p.add_argument("--excluded-smoke-gate",type=Path,required=True);p.add_argument("--lane-response",type=Path,action="append",required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists() or len(a.lane_response)!=2: raise ContractError("isolation requires one new output and exactly two lane responses")
    rows=[json.loads(path.read_text(encoding="utf-8")) for path in a.lane_response]
    for row in rows:
        if row.get("schema_version")!="vla-wam-shared-v3c002-fixed-observation-lane-v1" or row.get("status")!="completed_excluded_fixed_observation_lane" or row.get("passed") is not True or row.get("model_request_count")!=1 or row.get("behavioral_episode_count")!=0: raise ContractError("an isolation lane response is invalid")
        action=row.get("actions"); path=Path(str(action.get("path",""))) if isinstance(action,dict) else Path("")
        if not path.is_file() or action.get("bytes")!=path.stat().st_size or action.get("sha256")!=sha256_file(path): raise ContractError("isolation action binding changed")
    for key in ("fixture_sha256","prompt","prompt_utf8_hex","prompt_sha256","sampling_seed"):
        if rows[0].get(key)!=rows[1].get(key): raise ContractError(f"isolation lanes differ for {key}")
    for key in ("lane_id","simulator_pod_uid","simulator_gpu_uuid","policy_server_pod_uid","policy_server_gpu_uuid","server_port","server_process_identity","server_lock_identity"):
        if rows[0].get(key)==rows[1].get(key): raise ContractError(f"isolation lanes share {key}")
    if rows[0]["actions"]["sha256"]!=rows[1]["actions"]["sha256"]: raise ContractError("fixed-observation lane outputs differ")
    value={"schema_version":"vla-wam-shared-v3c002-two-lane-isolation-gate-v1","status":"passed_two_lane_fixed_observation_isolation","passed":True,
        "registration":repo_file_binding(a.registration),"queue":repo_file_binding(a.queue),"excluded_smoke_gate":repo_file_binding(a.excluded_smoke_gate),"lane_responses":[file_binding(path) for path in a.lane_response],
        "fixed_observation_equal":True,"fixed_prompt_equal":True,"request_seed_equal":True,"outputs_match":True,"lane_state_isolated":True,"model_request_count":2,"behavioral_episode_count":0,"excluded_from_behavioral_denominators":True,"action_sha256":rows[0]["actions"]["sha256"]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,allow_nan=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"status":value["status"],"sha256":sha256_file(a.output)},sort_keys=True))
if __name__=="__main__":main()
