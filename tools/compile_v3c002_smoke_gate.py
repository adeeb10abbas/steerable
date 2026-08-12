#!/usr/bin/env python3
"""Validate and bind the one excluded four-cell C002 smoke block."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, load_cells, repo_file_binding, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002.runner import _raw_row
def main()->None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--registration",type=Path,required=True);p.add_argument("--queue",type=Path,required=True);p.add_argument("--smoke-authorization",type=Path,required=True);p.add_argument("--completed-block",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ContractError(f"refusing to overwrite smoke gate: {a.output}")
    _,cells=load_cells(registration_path=a.registration,queue_path=a.queue);block={cell.cell_id:cell for cell in cells if cell.seed==12000}
    marker=json.loads(a.completed_block.read_text(encoding="utf-8"));records=marker.get("raw_episodes")
    if marker.get("schema_version")!="vla-wam-shared-v3c002-completed-block-v1" or marker.get("authorization_mode")!="excluded_smoke" or not isinstance(records,list) or len(records)!=4:raise ContractError("excluded smoke marker is incomplete")
    rows=[]
    for record in records:
        cell=block.get(record.get("cell_id"));path=Path(str(record.get("path","")))
        if cell is None or not path.is_file() or record.get("sha256")!=sha256_file(path):raise ContractError("smoke raw episode binding changed")
        rows.append(_raw_row(path,cell=cell,mode="excluded_smoke"))
    if len({row["initial_state_sha256"] for row in rows})!=1 or len({row["request0_pair_identity_sha256"] for row in rows})!=1:raise ContractError("excluded smoke block is not state/request-zero matched")
    request_count=sum(int(row["model_request_count"]) for row in rows)
    value={"schema_version":"vla-wam-shared-v3c002-excluded-smoke-gate-v1","status":"passed_excluded_four_cell_smoke","passed":True,
        "registration":repo_file_binding(a.registration),"queue":repo_file_binding(a.queue),"smoke_authorization":repo_file_binding(a.smoke_authorization),"completed_block":repo_file_binding(a.completed_block),
        "completed_cells":4,"completed_cell_ids":sorted(block),"model_request_count":request_count,"behavioral_episode_count":0,"excluded_from_behavioral_denominators":True,"initial_state_sha256":rows[0]["initial_state_sha256"],"request0_pair_identity_sha256":rows[0]["request0_pair_identity_sha256"],"raw_episode_bindings":records}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,allow_nan=False,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps({"status":value["status"],"sha256":sha256_file(a.output)},sort_keys=True))
if __name__=="__main__":main()
