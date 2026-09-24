"""Bounded scripted qualification for the missing scene orientations.

Consumes a recorded plan and never imports a learned policy or cluster client.
Completed evidence is reused; valid failures and partial attempts are retained.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

from .recorder import atomic_json
from .scene_design_archive import archive_arrays
from .scene_completion_verify import verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(plan_path, expected_hash, task_root, code_root, gpu, group):
    plan_path=Path(plan_path).resolve(); task_root=Path(task_root).resolve()
    code_root=Path(code_root).resolve()
    if sha(plan_path)!=expected_hash: raise ValueError('Plan hash changed')
    plan=json.loads(plan_path.read_text())
    for path,expected in plan['source_sha256'].items():
        if sha(code_root/path)!=expected: raise ValueError(f'Source changed: {path}')
    root=task_root/'evidence'/plan['id']; root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'.{group}.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/f'{group}.pid').write_text(str(os.getpid())+'\n')
    outcomes=[]
    def status(phase,**extra):
        atomic_json(root/f'{group}-status.json',dict(phase=phase,group=group,
            updated_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=expected_hash,
            outcomes=outcomes,model_requests=0,behavioral_episodes=0,**extra))
    try:
        for name in plan['groups'][group]:
            spec=plan['inputs'][name]; path=plan_path.parent/spec['file']
            if sha(path)!=spec['sha256']: raise ValueError('Registered input changed')
            row=json.loads(path.read_text()); output=root/name
            if not (output/'verification.json').exists():
                if output.exists(): raise ValueError(f'Partial attempt retained: {output}')
                if shutil.disk_usage(task_root).free<40*1024**3:
                    status('stopped_storage_below_40GiB'); return
                active=subprocess.check_output(['nvidia-smi',f'--id={gpu}',
                    '--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if active: raise RuntimeError(f'GPU {gpu} in use by {active}')
                status('running',active_candidate=name)
                with (root/f'{name}.log').open('x') as log:
                    subprocess.run(['bash',str(code_root/'docs/scene_design_rtx/run-completion-scene.sh'),
                        str(path),str(gpu),str(output)],cwd=code_root,
                        env={**os.environ,'SGW_SCENE_ROOT':str(task_root),'SGW_SCENE_CODE_ROOT':str(code_root)},
                        stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
                status('verifying',active_candidate=name)
                atomic_json(output/'verification.json',verify(output))
            launch=json.loads((output/'input.json').read_text())
            result=json.loads((output/'verification.json').read_text())
            if launch['design_sha256']!=spec['sha256'] or result['qualification_sha256']!=sha(output/'qualification.json'):
                raise ValueError('Evidence does not match registered input')
            if not (output/'archive.json').exists():
                status('archiving',active_candidate=name); archive_arrays(output)
            outcomes.append(dict(design_id=name,family=row['family'],
                side=row.get('approach_side',row['side']),status=result['status'],
                passed_checks=result['passed_checks'],verification_sha256=sha(output/'verification.json')))
            status('between_candidates')
        status('completed_scripted_checks')
    except BaseException:
        status('stopped_infrastructure_or_evidence_failure',error=traceback.format_exc())
        raise


def main():
    p=argparse.ArgumentParser(); p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--expected-sha256',required=True); p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True); p.add_argument('--gpu',type=int,choices=(0,1),required=True)
    p.add_argument('--group',required=True); a=p.parse_args()
    run(a.plan,a.expected_sha256,a.task_root,a.code_root,a.gpu,a.group)


if __name__=='__main__': main()
