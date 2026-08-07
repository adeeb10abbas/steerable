#!/usr/bin/env python3
"""Render E002 success and endpoint diagnostics from compiled records."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args(); import matplotlib.pyplot as plt
    d=json.loads(args.results.read_text()); rows=d.get('episodes',[]); labels=[]; vals=[]
    for arm in ('control','position_mirrored'):
      for rel in ('left','right'):
        x=[r for r in rows if r.get('arm')==arm and r.get('relation')==rel]; labels.append(f'{arm}\\n{rel}'); vals.append(sum(bool(r.get('task_success',r.get('success',False))) for r in x)/len(x) if x else None)
    fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(range(4),[v or 0 for v in vals],color=['#d47b32','#2f6f9f']*2); ax.set_xticks(range(4),labels); ax.set_ylim(0,1); ax.set_ylabel('Task success (proportion)'); ax.set_title('V3-E002 model-blind reference-controller symmetry'); ax.grid(axis='y',alpha=.25); fig.tight_layout(); args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=180); plt.close(fig)
if __name__=='__main__': main()
