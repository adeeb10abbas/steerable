#!/usr/bin/env python3
"""Compile fixed-observation request shards into compact E001 evidence."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from itertools import combinations

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

def quantiles(values):
    if not values:
        return {"mean": None, "median": None, "p05": None, "p95": None, "maximum": None}
    x = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95)), "maximum": float(np.max(x))}

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    source_files = []
    unique, invalid = {}, []
    for path in sorted(args.input_dir.rglob("requests*.jsonl")):
        source_files.append({"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        # Stream one JSON object at a time. Nano's retained decoded future can
        # make a single line hundreds of MB; keep only the action-bearing
        # fields needed by this compact diagnostic and discard video tensors.
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("model_id"), row.get("layout"), row.get("relation"), int(row.get("sampling_seed", -1)), bool(row.get("exact_repeat")))
                if row.get("status") == "infrastructure_invalid":
                    invalid.append({"model_id": row.get("model_id"), "layout": row.get("layout"), "relation": row.get("relation"), "sampling_seed": row.get("sampling_seed"), "error_type": row.get("error_type"), "error": row.get("error")})
                elif key not in unique:
                    response = row.get("response", {})
                    unique[key] = {k: row.get(k) for k in ("model_id", "layout", "relation", "prompt", "sampling_seed", "exact_repeat", "status")}
                    unique[key]["response"] = {k: response.get(k) for k in ("action", "action_shape", "action_sha256", "action_finite") if k in response}
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
            effects, repeat_rms, dims, noise = [], [], [], []
            actions = {}
            for seed in range(9400, 9427):
                l, rr = base.get((seed, "left")), base.get((seed, "right"))
                if l and rr and l.get("status") == rr.get("status") == "valid":
                    la = np.asarray(l.get("response", {}).get("action", []), float)
                    ra = np.asarray(rr.get("response", {}).get("action", []), float)
                    if la.size and ra.size:
                        effects.append(rms(la, ra)); dims.append(float(np.sqrt(np.mean((la.reshape(-1, la.shape[-1])[:min(la.shape[0],ra.shape[0])] - ra.reshape(-1,ra.shape[-1])[:min(la.shape[0],ra.shape[0])])**2, axis=0).mean())))
                for relation in ("left", "right"):
                    row = base.get((seed, relation))
                    if row and row.get("status") == "valid":
                        action = np.asarray(row.get("response", {}).get("action", []), float)
                        if action.size: actions[(relation, seed)] = action
            for relation in ("left", "right"):
                noise.extend(rms(actions[(relation, a)], actions[(relation, b)]) for a, b in combinations(range(9400, 9427), 2) if (relation, a) in actions and (relation, b) in actions)
                rep = repeats.get((9400, "left"))
                if seed == 9400 and l and rep and l.get("status") == rep.get("status") == "valid":
                    repeat_rms.append(rms(l.get("response", {}).get("action", []), rep.get("response", {}).get("action", [])))
            metrics[f"{model_id}/{layout}"] = {
                "model_request_rows": len(model_rows), "matched_prompt_effect_count": len(effects),
                "matched_prompt_effect_rms": effects, "matched_prompt_effect_median": float(np.median(effects)) if effects else None,
                "exact_repeat_rms": repeat_rms, "exact_repeat_bit_identity": all(v == 0.0 for v in repeat_rms) if repeat_rms else None,
                "same_prompt_cross_seed_pairwise_rms": quantiles(noise),
                "same_prompt_cross_seed_pair_count": len(noise),
                "prompt_to_noise_ratio": (float(np.median(effects) / np.median(noise)) if effects and noise and np.median(noise) != 0 else None),
                "prompt_effects_above_noise_p95_fraction": (float(np.mean(np.asarray(effects) > np.quantile(noise, .95))) if effects and noise else None),
                "matched_prompt_effect_mean": (float(np.mean(effects)) if effects else None),
                "matched_prompt_effect_per_dimension_rms_mean": (float(np.mean(dims)) if dims else None),
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
