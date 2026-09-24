"""Recompute prototype qualification from retained native recordings, CPU only."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .scene_completion_contract import restore_scored_order
from .fixtures import ACTION_CAP, FixtureCandidate, FixtureError, validate_reset
from .qualification_batch_verifier import verify_trial_evidence
from .recorder import atomic_json


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    root=Path(root).resolve()
    data=json.loads((root/'candidate.json').read_text())
    # JSON serialization sorts object keys. Restore the producer's iteration
    # order for its ordered reset-comparison receipts; identity hashes sort keys.
    data=restore_scored_order(data)
    candidate=FixtureCandidate.from_json(data)
    result=json.loads((root/'qualification.json').read_text())
    launch=json.loads((root/'input.json').read_text())
    require(result['candidate_id']==candidate.candidate_id==launch['design']['design_id'], 'candidate identity changed')
    require(result['candidate_sha256']==hashlib.sha256(json.dumps(asdict(candidate),sort_keys=True).encode()).hexdigest(), 'candidate hash changed')
    require(result['family']==candidate.family and result['seed']==candidate.seed, 'family/seed changed')
    require(result['action_cap']==ACTION_CAP and result['model_request_count']==result['behavioral_episode_count']==0, 'execution contract changed')
    require(digest(root/'scene.usda')==launch['scene']['sha256'], 'scene changed')
    require(digest(root/'capture.json')==candidate.metadata['candidate_capture_sha256'], 'capture changed')
    if 'calibration_sha256' in launch:
        require(result['controller_identity']['calibration_sha256']==launch['calibration_sha256'], 'controller changed')
    checks=result['checks']
    require([(c['goal_sign'],c['reset_index']) for c in checks]==[(s,r) for s in (1,-1) for r in range(3)], 'six complete trials required')
    reports=[]
    for offset in (0,3):
        group=checks[offset:offset+3]
        verified=[verify_trial_evidence(evidence_root=root,
            trial=root/'trials'/f"goal-{c['goal_sign']:+d}"/f"reset-{c['reset_index']}",
            check=c,candidate=candidate,expect_geometry_guard=True) for c in group]
        for c in group:
            trial=root/'trials'/f"goal-{c['goal_sign']:+d}"/f"reset-{c['reset_index']}"
            guard=json.loads((trial/'preaction-geometry-guard.json').read_text())
            require(guard['design_id']==candidate.candidate_id and guard['candidate_sha256']==result['candidate_sha256'], 'guard identity changed')
            require(guard['candidate_capture_sha256']==candidate.metadata['candidate_capture_sha256'], 'guard capture changed')
            require(guard['raw_reset']['sha256']==digest(trial/'state-0000.json'), 'guard reset changed')
            require(guard['goal_sign']==c['goal_sign'] and guard['reset_index']==c['reset_index'] and guard['controller_actions_executed']==0, 'guard trial changed')
        try:
            rows=validate_reset(candidate,[reset for _,reset in verified])
            reset_error=None
        except FixtureError as error:
            rows=[]; reset_error=str(error)
        for check,(report,_) in zip(group,verified,strict=True):
            passed=report['score']['requested_success'] and reset_error is None
            require(check['passed'] is passed,'aggregate physical/reset result differs')
            if reset_error is None:
                require('reset_error' not in check and check['reset_validation']==[r for r in rows if r['repeat']==check['reset_index']], 'reset comparison differs')
            else:
                require(check.get('reset_error')==reset_error,'reset rejection differs')
            reports.append({**report,'passed':passed,'reset_error':reset_error})
    count=sum(r['passed'] for r in reports)
    require(result['status']==('accepted_model_blind_fixture_candidate' if count==6 else 'rejected_model_blind_fixture_candidate'), 'aggregate status differs')
    return {'candidate_id':candidate.candidate_id,'family':candidate.family,
            'status':'verified_all_six_pass' if count==6 else 'verified_physical_rejection',
            'passed_checks':count,'checks':reports,'qualification_sha256':digest(root/'qualification.json'),
            'scene_sha256':launch['scene']['sha256'],'model_requests':0,'behavioral_episodes':0}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('root',type=Path)
    args=parser.parse_args()
    output=args.root/'verification.json'
    if output.exists(): raise FileExistsError(output)
    report=verify(args.root)
    atomic_json(output,report)
    print(json.dumps({k:report[k] for k in ('candidate_id','status','passed_checks')}),flush=True)


if __name__=='__main__': main()
