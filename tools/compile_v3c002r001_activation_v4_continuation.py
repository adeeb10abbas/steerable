#!/usr/bin/env python3
"""Compile outcome-blind continuation after both A003 retries close."""

from __future__ import annotations

import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import file_binding,read_finite_json,require,sha256_file,validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v4_contract import A003_SHA,CONTINUATION_SCHEMA


def main():
    p=argparse.ArgumentParser(); p.add_argument('--a003-release',type=Path,required=True); p.add_argument('--queue',type=Path,required=True); p.add_argument('--assignment',type=Path,required=True); p.add_argument('--registration',type=Path,required=True); p.add_argument('--marker',action='append',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    require(not a.output.exists() and sha256_file(a.a003_release)==A003_SHA,'A004 continuation input/output invalid')
    a003=read_finite_json(a.a003_release); assignments=[json.loads(x) for x in a.assignment.read_text().splitlines()]
    by_lane={f'repair-lane-{i:02d}':[] for i in range(8)}
    for row in assignments: by_lane[row['lane_slot']].append(int(row['episode_seed']))
    completed={}; retry={'repair-lane-00':12060,'repair-lane-01':12101}
    for item in a.marker:
        slot,_,raw=item.partition('='); require(slot in retry and raw,'A004 marker syntax invalid')
        b=file_binding(Path(raw)); m=read_finite_json(Path(raw))
        require(m.get('schema_version')=='vla-wam-shared-v3c002-completed-block-v1' and m.get('status')=='completed_behavioral_block' and m.get('episode_seed')==retry[slot] and str(m.get('attempt_root','')).endswith('/attempt002'),'A004 retry marker invalid')
        require(isinstance(m.get('raw_episodes'),list) and len(m['raw_episodes'])==4,'A004 retry marker partial')
        for r in m['raw_episodes']: validate_file_binding(r,'A004 retry raw')
        completed[slot]=b
    require(set(completed)==set(retry),'A004 requires both retry markers')
    # Frozen pre-replacement completed lists are carried by A003; no raw outcomes are read.
    remaining={}
    for slot in retry:
        forbidden=set(a003['replacements'][slot]['completed_seed_blocks_forbidden'])|{retry[slot]}
        remaining[slot]=[s for s in by_lane[slot] if s not in forbidden]
    value={'schema_version':CONTINUATION_SCHEMA,'status':'passed_outcome_blind_a004_continuation_release','passed':True,'a003_release':file_binding(a.a003_release),'queue':file_binding(a.queue),'assignment_manifest':file_binding(a.assignment),'a004_registration':file_binding(a.registration),'retry_markers':completed,'remaining_seed_blocks_by_lane':remaining,'lane_manifests':a003['lane_manifests'],'completed_blocks_never_rerun':True,'no_cross_lane_failover':True,'outcome_fields_read':False,'science_unchanged':True}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); print(sha256_file(a.output))

if __name__=='__main__': main()
