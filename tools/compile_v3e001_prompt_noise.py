#!/usr/bin/env python3
"""Compile hash-bearing V3-E001 request records without inventing missing data."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def sha256(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
def rms(a,b):
    d=np.asarray(a,float)-np.asarray(b,float)
    return float(np.sqrt(np.mean(d*d)))
def percentile(x,q): return float(np.percentile(np.asarray(x,float),q)) if x else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    rows=[]
    for p in sorted(args.input_dir.rglob('*.jsonl')):
        for line in p.read_text().splitlines():
            if line.strip(): rows.append(json.loads(line))
    report={'schema_version':'vla-wam-shared-v3e001-results-v1','status':'blocked_no_requests' if not rows else 'compiled','model_request_count':len(rows),'behavioral_episode_count':0,'records_sha256':sha256(p) if rows else None}
    groups={}
    for r in rows:
        key=(r.get('model_id'),r.get('layout'))
        groups.setdefault(key,[]).append(r)
    metrics={}
    for (model,layout),rs in groups.items():
        by_seed={}
        for r in rs: by_seed.setdefault(int(r['sampling_seed']),{})[r['prompt_relation']]=r
        effects=[]; noise=[]
        for seed,v in by_seed.items():
            if 'left' in v and 'right' in v: effects.append(rms(np.load(v['action_path']),np.load(v['right_action_path'] if 'right_action_path' in v else v['action_path'])))
            if v.get('repeat_action_path') and v.get('action_path'): noise.append(rms(np.load(v['action_path']),np.load(v['repeat_action_path'])))
        metrics[f'{model}/{layout}']={'matched_prompt_effect_rms':effects,'same_prompt_repeat_rms':noise,'prompt_effect_median':percentile(effects,50),'same_prompt_noise_median':percentile(noise,50),'status':'ok' if effects else 'missing_required_pairs'}
    report['metrics']=metrics; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n'); print(json.dumps(report,indent=2,sort_keys=True))
if __name__=='__main__': main()
