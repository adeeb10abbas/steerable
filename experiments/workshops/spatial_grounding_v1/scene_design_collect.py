"""Bounded completion step for this engineering run; never launches experiments."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import traceback

from .recorder import atomic_json
from .scene_design_archive import archive_arrays
from .scene_design_verify import verify


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--names',nargs='+',required=True)
    parser.add_argument('--timeout-seconds',type=int,default=7200)
    args=parser.parse_args()
    deadline=time.monotonic()+args.timeout_seconds
    done={}
    while time.monotonic()<deadline:
        rows=[]
        for name in args.names:
            if name in done: rows.append(done[name]); continue
            root=args.evidence/name
            status={'name':name,'state':'queued' if not root.exists() else 'running'}
            if (root/'infrastructure_failure.json').exists():
                status.update(state='infrastructure_failure',details=str(root/'infrastructure_failure.json'))
                done[name]=status
            elif (root/'qualification.json').exists():
                try:
                    vp=root/'verification.json'
                    if not vp.exists(): atomic_json(vp,verify(root))
                    result=json.loads(vp.read_text())
                    ap=root/'archive.json'
                    if not ap.exists(): archive_arrays(root)
                    status.update(state=result['status'],passed_checks=result['passed_checks'],
                                  family=result['family'],details=str(vp),raw_archive=str(root/'raw-arrays.tar.zst'))
                except Exception:
                    status.update(state='evidence_processing_failure',error=traceback.format_exc())
                done[name]=status
                print(json.dumps(status),flush=True)
            else:
                trials=[]
                for p in sorted(root.glob('trials/*/*/trial.json')):
                    c=json.loads(p.read_text())
                    trials.append({'goal_sign':c.get('goal_sign'),'reset_index':c.get('reset_index'),
                                   'passed':c.get('passed'),'failure_stage':c.get('score',{}).get('failure_stage')})
                status['completed_trials']=trials
            rows.append(status)
        atomic_json(args.evidence/'scene-design-status.json',{'updated_at_unix':time.time(),
            'scope':'Engineering scenes only; no model requests or frozen-pool assignments',
            'scenes':rows,'all_finished':len(done)==len(args.names)})
        if len(done)==len(args.names): return
        time.sleep(20)
    print('Bounded collection deadline reached; unfinished scenes remain explicitly incomplete.',flush=True)


if __name__=='__main__': main()
