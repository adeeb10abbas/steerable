#!/usr/bin/env python3
"""Capture raw settled observations from the registered B001 fixture.

This wraps the existing model-blind preflight without changing its fixture,
reset, settle, renderer, or camera code.  It retains the first settled raw
camera/proprio arrays for each requested condition; no policy is contacted.
"""
from __future__ import annotations
import json, os, runpy, sys
from pathlib import Path

def main() -> None:
    # Pass the preflight CLI through unchanged. The preflight parses it before
    # importing RoboLab and starts Isaac Sim exactly as in the established gate.
    forwarded = sys.argv[1:]
    sys.argv = ["model_blind_preflight.py", *forwarded]
    try:
        output_dir = Path(forwarded[forwarded.index("--output-dir") + 1])
    except (ValueError, IndexError) as exc:
        raise SystemExit("--output-dir is required") from exc
    # The preflight writes only after the 60-step settle + 15-step stability
    # window, so captured bytes are genuinely settled and hash-bound.
    os.environ["V3E_CAPTURE_OBSERVATION_DIR"] = str(output_dir)
    ns = runpy.run_module("experiments.v3.pi05_phase_b.model_blind_preflight", run_name="v3e001_capture")
    ns["main"]()
    manifest = output_dir / "settled_observation_capture.json"
    files = sorted(output_dir.glob("settled_observation_*.npz"))
    manifest.write_text(json.dumps({"schema_version":"vla-wam-shared-v3e001-raw-observation-capture-v2","model_request_count":0,"behavioral_episode_count":0,"saved":[str(p) for p in files]}, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"observation_manifest":str(manifest),"saved":[str(p) for p in files]}, indent=2))

if __name__ == "__main__": main()
