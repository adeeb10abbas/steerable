#!/usr/bin/env python3
"""Fail-closed launcher for E002; no behavior may run without a passed gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gate',type=Path,required=True); ap.add_argument('--queue',type=Path,required=True); ap.add_argument('--output-dir',type=Path,required=True); args=ap.parse_args()
    if not args.gate.is_file(): raise SystemExit('E002 blocked: model-blind gate is absent')
    gate=json.loads(args.gate.read_text())
    if gate.get('passed') is not True: raise SystemExit('E002 blocked: model-blind gate did not pass; no episode launched')
    raise SystemExit('E002 blocked: verified symmetric RoboLab controller implementation is not registered')
if __name__=='__main__': main()
