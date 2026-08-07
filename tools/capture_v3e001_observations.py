#!/usr/bin/env python3
"""Capture raw settled observations from the registered B001 fixture.

This wraps the existing model-blind preflight without changing its fixture,
reset, settle, renderer, or camera code.  It retains the first settled raw
camera/proprio arrays for each requested condition; no policy is contacted.
"""
from __future__ import annotations
import json, runpy, sys
from pathlib import Path
import numpy as np

def main() -> None:
    # Pass the preflight CLI through unchanged. The preflight parses it before
    # importing RoboLab and starts Isaac Sim exactly as in the established gate.
    forwarded = sys.argv[1:]
    sys.argv = ["model_blind_preflight.py", *forwarded]
    ns = runpy.run_module("experiments.v3.pi05_phase_b.model_blind_preflight", run_name="v3e001_capture")
    output_dir = Path(ns["args_cli"].output_dir)
    original_hold = ns["_hold"]
    saved = {"count": 0, "files": []}
    def hold(obs, device):
        idx = saved["count"]
        if idx == 0:
            arrays = {}
            for name, value in obs["image_obs"].items():
                arrays[f"image_obs/{name}"] = np.asarray(value[0].detach().cpu().numpy(), dtype=np.uint8)
            for name, value in obs["proprio_obs"].items():
                arrays[f"proprio_obs/{name}"] = np.asarray(value[0].detach().cpu().numpy())
            path = output_dir / "settled_observation_raw.npz"
            np.savez_compressed(path, **arrays)
            saved["files"].append({"path": str(path), "keys": sorted(arrays), "bytes": path.stat().st_size})
        saved["count"] += 1
        return original_hold(obs, device)
    ns["_hold"] = hold
    ns["main"]()
    manifest = output_dir / "settled_observation_capture.json"
    manifest.write_text(json.dumps({"schema_version":"vla-wam-shared-v3e001-raw-observation-capture-v1","model_request_count":0,"behavioral_episode_count":0,"saved":saved}, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"observation_manifest":str(manifest),"saved":saved}, indent=2))

if __name__ == "__main__": main()
