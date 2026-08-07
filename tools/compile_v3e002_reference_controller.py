#!/usr/bin/env python3
"""Compile E002 raw episode JSONL while keeping infrastructure failures separate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--episodes-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); rows=[]; infra=[]
    for p in sorted(args.episodes_dir.rglob('*.jsonl')):
        for line in p.read_text().splitlines():
            if not line.strip(): continue
            r=json.loads(line); (infra if r.get('classification')=='infrastructure_invalid' else rows).append(r)
    out={'schema_version':'vla-wam-shared-v3e002-results-v1','behavioral_episode_count':len(rows),'infrastructure_invalid_count':len(infra),'status':'compiled' if rows else 'blocked_no_behavioral_episodes','episodes':rows,'infrastructure_invalid':infra}; args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'status':out['status'],'behavioral_episode_count':len(rows),'infrastructure_invalid_count':len(infra)},indent=2))
if __name__=='__main__': main()
