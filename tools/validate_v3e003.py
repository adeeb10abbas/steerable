#!/usr/bin/env python3
"""Fail-closed validation for the completed E003 evidence bundle."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path('artifacts/vla_wam_shared_v3/phase_e/bilateral_symmetry_null_control_v3e003')); a=ap.parse_args(); root=a.root
    reg=json.loads((root/'registration.json').read_text()); assert reg['status']=='registered_before_inference'; assert len(reg['queue'])==54
    seeds=sorted({int(x['environment_seed']) for x in reg['queue']}); assert seeds==list(range(9400,9427))
    assert reg['equivalence_margins']['binary_gap_abs'] == 4/27
    assert reg['equivalence_margins']['requested_side_depth_contrast_abs_m'] == .05
    result=json.loads((root/'results.json').read_text()); assert result['status']=='complete'; assert result['valid_behavioral_episodes']==54
    assert result['directions']['left']['n']==27 and result['directions']['right']['n']==27
    assert result['symmetry_residual']['all_within_1mm']
    for src in result['source_files']:
        # Raw episode files intentionally remain on the PVC.  On the
        # work-laptop validator, verify their bound SHA-256 format and PVC
        # namespace; when mounted locally, also re-hash the bytes.
        assert len(src['sha256']) == 64 and all(c in '0123456789abcdef' for c in src['sha256'])
        p=Path(src['path'])
        if p.exists(): assert sha(p)==src['sha256']
        else: assert str(p).startswith('/data/users/ali/vla_wam/raw/')
    print(json.dumps({'status':'valid','behavioral_cells':54,'left':27,'right':27,'source_files':len(result['source_files'])},indent=2))
if __name__=='__main__': main()
