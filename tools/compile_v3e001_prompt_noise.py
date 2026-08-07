#!/usr/bin/env python3
"""Compile fixed-observation request shards into compact E001 evidence."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EXPECTED = {"pi05": "pi05_current_stack_droid", "nano": "cosmos3_nano_policy_droid", "dreamzero": "dreamzero_droid_action_cfg"}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def rms(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(a.size, b.size)
    return float(np.sqrt(np.mean((a.reshape(-1)[:n] - b.reshape(-1)[:n]) ** 2)))

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    source_files = []
    for path in sorted(args.input_dir.rglob("requests*.jsonl")):
        source_files.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip(): rows.append(json.loads(line))
    # A crashed shard may leave a partial prefix. Keep the first registered
    # record for a key and retain all infrastructure-invalid records separately.
    unique, invalid = {}, []
    for row in rows:
        key = (row.get("model_id"), row.get("layout"), row.get("relation"), int(row.get("sampling_seed", -1)), bool(row.get("exact_repeat")))
        if row.get("status") == "infrastructure_invalid":
            invalid.append(row)
        elif key not in unique:
            unique[key] = row
    records = list(unique.values())
    metrics = {}
    for raw_model, model_id in EXPECTED.items():
        model_rows = [r for r in records if r.get("model_id") == raw_model]
        if not model_rows:
            # permit callers that name model ids directly
            model_rows = [r for r in records if r.get("model_id") == model_id]
        for layout in ("control", "position_mirrored"):
            base = {(int(r["sampling_seed"]), r["relation"]): r for r in model_rows if r.get("layout") == layout and not r.get("exact_repeat")}
            repeats = {(int(r["sampling_seed"]), r["relation"]): r for r in model_rows if r.get("layout") == layout and r.get("exact_repeat")}
            effects, repeat_rms, dims = [], [], []
            for seed in range(9400, 9427):
                l, rr = base.get((seed, "left")), base.get((seed, "right"))
                if l and rr and l.get("status") == rr.get("status") == "valid":
                    la = np.asarray(l.get("response", {}).get("action", []), float)
                    ra = np.asarray(rr.get("response", {}).get("action", []), float)
                    if la.size and ra.size:
                        effects.append(rms(la, ra)); dims.append(float(np.sqrt(np.mean((la.reshape(-1, la.shape[-1])[:min(la.shape[0],ra.shape[0])] - ra.reshape(-1,ra.shape[-1])[:min(la.shape[0],ra.shape[0])])**2, axis=0).mean())))
                rep = repeats.get((9400, "left"))
                if seed == 9400 and l and rep and l.get("status") == rep.get("status") == "valid":
                    repeat_rms.append(rms(l.get("response", {}).get("action", []), rep.get("response", {}).get("action", [])))
            metrics[f"{model_id}/{layout}"] = {
                "model_request_rows": len(model_rows), "matched_prompt_effect_count": len(effects),
                "matched_prompt_effect_rms": effects, "matched_prompt_effect_median": float(np.median(effects)) if effects else None,
                "exact_repeat_rms": repeat_rms, "exact_repeat_bit_identity": all(v == 0.0 for v in repeat_rms) if repeat_rms else None,
                "status": "complete" if len(effects) == 27 else "incomplete",
            }
    report = {
        "schema_version": "vla-wam-shared-v3e001-results-v2", "amendment_id": "V3-E001",
        "status": "complete" if all(v["status"] == "complete" for v in metrics.values()) else "partial",
        "behavioral_episode_count": 0, "model_request_count": len(records),
        "registered_model_request_count": 336, "valid_record_count": len(records), "infrastructure_invalid_count": len(invalid),
        "deduplication_key": ["model_id", "layout", "relation", "sampling_seed", "exact_repeat"],
        "metrics": metrics, "source_files": source_files,
        "claim_boundary": "Fixed-observation prompt/noise diagnostic; no action was executed and no task success claim is made.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))

if __name__ == "__main__": main()
