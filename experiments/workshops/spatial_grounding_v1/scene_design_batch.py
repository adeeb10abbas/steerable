"""Finite workstation qualification of prospectively recorded clean scenes.

Never runs policies or contacts Kubernetes. One family per physical GPU.
Only independently verified all-six passes count toward the 29-layout target.
"""
from __future__ import annotations
import argparse
import copy
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
import traceback

from .recorder import atomic_json
from .scene_design import design
from .scene_design_archive import archive_arrays
from .scene_design_verify import verify

STUDY=Path('artifacts/workshops/spatial_grounding_v1')
SOURCE=Path('experiments/workshops/spatial_grounding_v1')


def encoded(value): return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_pool(family, workspace, lat_template, seed):
    grid=[(x/1000,y/1000) for x in range(-40,41,10) for y in range(-60,61,10) if (x,y)!=(0,0)]
    random.Random(seed).shuffle(grid)
    rows=[]
    for index,(dx,dy) in enumerate(grid[:100]):
        if family=='LAT':
            row=copy.deepcopy(lat_template)
            for group in ('centers','roots','targets'):
                for position in row[group].values(): position[0]+=dx; position[1]+=dy
            row['geometry_revision']='translated-historical-LAT-template'
        elif family=='HEIGHT':
            row=design('HEIGHT','left' if index%2==0 else 'right',dx,dy,workspace)
        else: raise ValueError(family)
        row.update(design_id=f'SGW-CLEAN-20260924-{family}-{index:03d}',seed=seed,
                   translation_xy_m=[dx,dy],visual_style='clean-studio-v1',
                   status='authored_unqualified',model_requests=0)
        rows.append(row)
    return sorted(rows,key=lambda row:hashlib.sha256(encoded(row)).hexdigest())


def enough(family, accepted, seed):
    if family=='LAT': return len(accepted)>=29
    pilot=('left','right')[seed%2]
    return all(sum(r['side']==side for r in accepted)>=14+(side==pilot) for side in ('left','right'))


def assign(family, accepted, seed):
    if not enough(family,accepted,seed): raise ValueError('Insufficient qualified layouts')
    if family=='LAT': selected=accepted[:29]
    else:
        groups={side:[r for r in accepted if r['side']==side] for side in ('left','right')}
        selected=[groups[('left','right')[seed%2]].pop(0)]
        for count in (2,12):
            for side in ('left','right'):
                selected.extend(groups[side][:count]); del groups[side][:count]
    labels=['P01']+[f'D{i:02d}' for i in range(1,5)]+[f'C{i:02d}' for i in range(1,25)]
    return {f'{family}-{label}':row['design_id'] for label,row in zip(labels,selected,strict=True)}


def prepare(output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    workspace=STUDY/'infrastructure/a40-20260922z-workspace.json'
    template=STUDY/'scene_design_rtx_20260923/prototype-06.json'
    seed=20260923
    files=[SOURCE/name for name in ('scene_design.py','scene_design_runner.py','scene_design_task.py',
        'scene_design_verify.py','scene_design_archive.py','scene_design_batch.py',
        'model_blind_qualification.py','grasp_calibration.py','scoring.py','fixtures.py',
        'robolab_lat_qualification.py','robolab_height_dist_qualification.py','qualification_batch_verifier.py')]
    files += [workspace,template,Path('docs/scene_design_rtx/run-scene.sh'),
              STUDY/'controller_calibrations/lat-closed-pad-20260923.json']
    pools={}
    for family in ('LAT','HEIGHT'):
        rows=make_pool(family,json.loads(workspace.read_text()),json.loads(template.read_text()),seed)
        path=output/f'{family}.jsonl'; path.write_bytes(b''.join(encoded(r) for r in rows))
        pools[family]={'file':path.name,'sha256':sha(path),'candidate_ids_in_order':[r['design_id'] for r in rows]}
    plan={'id':'SGW-CLEAN-20260924','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
          'seed':seed,'family_candidate_cap':100,'target_per_family':29,'scripted_trials_per_candidate':6,
          'actions_per_trial':450,'model_requests':0,'pools':pools,'source_sha256':{str(p):sha(p) for p in files},
          'prerequisites':{'LAT':['prototype-06-a'],'HEIGHT':['prototype-04-a','prototype-05-a']},
          'known_outcomes':'Prototype 00 and clean 03 verified 5/6 with a terminal-stability failure. Original HEIGHT 01 verified 6/6. Clean HEIGHT 05 has a native 6/6 aggregate. Historical-geometry clean LAT 06 and clean HEIGHT 04 are still running. Historical SGW-ENG-008 success informed the LAT template.',
          'selection':'Frozen SHA256 row order. LAT takes the first 29 all-six passes. HEIGHT takes a right-side pilot, then two per side for development and twelve per side for confirmation, each in frozen order. Stop the family once its quota is available. No rejected candidate is replaced or retried. Remaining rows are not run.',
          'distinctness':'10mm lattice; (0,0) excluded so the historical reserved pilot is not reused. LAT translates every task object and both targets together. HEIGHT has 50 candidates per support side and distinct cube/bowl positions.',
          'appearance':'clean-studio-v1 throughout; new visual condition, separate from old office-background model results.',
          'bounds':'Wait at most two hours for verified templates; then at most 24 hours per family, at most 100 candidates, 3600 seconds per candidate. Stop with explicit status on insufficient space (<40GiB), occupied GPU, or infrastructure/evidence failure. Preserve partials and every valid failure.',
          'release':'Engineering fixture qualification only. No learned-policy release or active-worker changes.'}
    atomic_json(output/'plan.json',plan)
    print(json.dumps({'plan':str(output/'plan.json'),'sha256':sha(output/'plan.json')}),flush=True)


def run(plan_path, expected_hash, family, task_root, gpu):
    plan_path=Path(plan_path).resolve(); task_root=Path(task_root).resolve()
    if sha(plan_path)!=expected_hash: raise ValueError('Plan hash mismatch')
    plan=json.loads(plan_path.read_text()); repo=task_root/'steerable'
    for path,expected in plan['source_sha256'].items():
        if sha(repo/path)!=expected: raise ValueError(f'Source changed: {path}')
    spec=plan['pools'][family]; pool=plan_path.parent/spec['file']
    if sha(pool)!=spec['sha256']: raise ValueError('Pool hash mismatch')
    rows=[json.loads(line) for line in pool.read_text().splitlines()]
    if len(rows)!=100 or [r['design_id'] for r in rows]!=spec['candidate_ids_in_order']:
        raise ValueError('Pool inventory changed')
    root=task_root/'evidence'/plan['id']/family; root.mkdir(parents=True,exist_ok=True)
    lock=(root/'.worker.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    (root/'worker.pid').write_text(str(os.getpid())+'\n')
    accepted=[]; outcomes=[]
    def status(phase, **extra):
        atomic_json(root/'status.json',{'plan_sha256':expected_hash,'family':family,'phase':phase,
            'updated_at_utc':datetime.now(timezone.utc).isoformat(),'accepted_count':len(accepted),
            'accepted_sides':{s:sum(r['side']==s for r in accepted) for s in ('none','left','right')},
            'outcomes':outcomes,'model_requests':0,**extra})
    try:
        status('waiting_for_verified_templates')
        until=time.monotonic()+7200
        while True:
            required=[task_root/'evidence'/n/'verification.json' for n in plan['prerequisites'][family]]
            if any(p.exists() and json.loads(p.read_text())['status']!='verified_all_six_pass' for p in required):
                status('stopped_template_failed'); return
            if all(p.exists() for p in required): break
            if time.monotonic()>=until: status('stopped_template_wait_timeout'); return
            time.sleep(20)
        deadline=time.monotonic()+24*3600
        for row in rows:
            if enough(family,accepted,plan['seed']): break
            if time.monotonic()>=deadline: status('stopped_time_budget'); return
            name=row['design_id']; output=root/name
            inputs=root/'inputs'; inputs.mkdir(exist_ok=True)
            design_path=inputs/f'{name}.json'
            if design_path.exists() and design_path.read_bytes()!=encoded(row): raise ValueError('Prepared input changed')
            if not design_path.exists(): design_path.write_bytes(encoded(row))
            if not (output/'verification.json').exists():
                if output.exists(): raise ValueError(f'Partial attempt retained; manual investigation required: {output}')
                if shutil.disk_usage(task_root).free<40*1024**3: status('stopped_storage_below_40GiB'); return
                active=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
                if active: status('stopped_gpu_in_use',gpu_processes=active); return
                status('running',active_candidate=name)
                with (root/f'{name}.log').open('x') as log:
                    subprocess.run(['bash',str(repo/'docs/scene_design_rtx/run-scene.sh'),str(design_path),str(gpu),str(output)],
                                   cwd=repo,env={**os.environ,'SGW_SCENE_ROOT':str(task_root)},
                                   stdout=log,stderr=subprocess.STDOUT,timeout=3600,check=True)
                status('verifying',active_candidate=name)
                atomic_json(output/'verification.json',verify(output))
            launch=json.loads((output/'input.json').read_text())
            result=json.loads((output/'verification.json').read_text())
            if launch['design_sha256']!=sha(design_path) or result['qualification_sha256']!=sha(output/'qualification.json'):
                raise ValueError('Completed evidence does not match registered input')
            if not (output/'archive.json').exists():
                status('archiving',active_candidate=name); archive_arrays(output)
            passed=result['status']=='verified_all_six_pass'
            if passed: accepted.append(row)
            outcomes.append({'design_id':name,'side':row['side'],'status':result['status'],'passed_checks':result['passed_checks'],
                             'verification_sha256':sha(output/'verification.json')})
            status('between_candidates')
        if enough(family,accepted,plan['seed']):
            atomic_json(root/'assignments.json',{'plan_sha256':expected_hash,'assignments':assign(family,accepted,plan['seed']),
                'claim':'Qualified clean scenes only; no learned-policy episodes or release.'})
            status('completed_29_qualified_scenes')
        else: status('pool_exhausted_without_required_qualifiers')
    except Exception:
        status('stopped_infrastructure_or_evidence_failure',error=traceback.format_exc()); raise


def main():
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('run'); p.add_argument('--plan',type=Path,required=True); p.add_argument('--expected-sha256',required=True)
    p.add_argument('--family',choices=('LAT','HEIGHT'),required=True); p.add_argument('--task-root',type=Path,required=True)
    p.add_argument('--gpu',type=int,choices=(0,1),required=True)
    args=parser.parse_args()
    if args.mode=='prepare': prepare(args.output)
    else: run(args.plan,args.expected_sha256,args.family,args.task_root,args.gpu)


if __name__=='__main__': main()
