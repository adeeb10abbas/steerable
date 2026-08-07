#!/usr/bin/env python3
"""Render a compact E001 diagnostic; blocked groups remain visibly empty."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--results',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    import matplotlib.pyplot as plt
    d=json.loads(args.results.read_text()); items=list(d.get('metrics',{}).items()); fig,ax=plt.subplots(figsize=(10,5));
    for i,(label,m) in enumerate(items):
        x=m.get('matched_prompt_effect_rms',[]); y=m.get('same_prompt_repeat_rms',[])
        if x: ax.scatter([i]*len(x),x,color='#2f6f9f',label='prompt effect' if i==0 else None)
        if y: ax.scatter([i]*len(y),y,color='#d47b32',marker='x',label='same-prompt noise' if i==0 else None)
    ax.set_xticks(range(len(items)),[k.replace('/','\\n') for k,_ in items],rotation=0); ax.set_ylabel('Action RMS (native units)'); ax.set_title('V3-E001 fixed-observation prompt effect vs sampling noise'); ax.legend(); ax.grid(axis='y',alpha=.25); fig.tight_layout(); args.output.parent.mkdir(parents=True,exist_ok=True); fig.savefig(args.output,dpi=180); plt.close(fig)
if __name__=='__main__': main()
