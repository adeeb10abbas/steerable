#!/usr/bin/env python3
"""Compile the immutable V3-E001 request ledgers.

This compiler is deliberately post-processing only.  It streams the raw
JSONL ledgers, keeps action arrays and compact metadata, and never contacts a
model server or executes an action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED = {
    "pi05": "pi05_current_stack_droid",
    "nano": "cosmos3_nano_policy_droid",
    "dreamzero": "dreamzero_droid_action_cfg",
}
SEEDS = tuple(range(9400, 9427))
MODELS = tuple(EXPECTED.values())
LAYOUTS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
BOOTSTRAP_REPS = 20_000
PERMUTATION_REPS = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def rms(first: Any, second: Any) -> float:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} versus {b.shape}")
    return float(np.sqrt(np.mean(np.square(a - b))))


def per_dimension_rms(first: Any, second: Any) -> list[float]:
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"expected equal rank-2 action arrays, got {a.shape} and {b.shape}")
    return np.sqrt(np.mean(np.square(a - b), axis=0)).tolist()


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p05": None, "p95": None, "maximum": None}
    x = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(x)), "median": float(np.median(x)),
        "p05": float(np.quantile(x, .05)), "p95": float(np.quantile(x, .95)),
        "maximum": float(np.max(x)),
    }


def exact_sign_test(values: list[float]) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    positive, negative = int(np.sum(x > 0)), int(np.sum(x < 0))
    ties = int(np.sum(x == 0))
    n = positive + negative
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, k) for k in range(0, min(positive, negative) + 1)) / (2 ** n)
        p = min(1.0, 2.0 * tail)
    return {"method": "exact_two_sided_sign_test", "positive": positive,
            "negative": negative, "ties": ties, "effective_n": n, "p_value": float(p)}


def bootstrap(values: list[float], seed: int) -> dict[str, Any]:
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {"replicates": BOOTSTRAP_REPS, "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(x), size=(BOOTSTRAP_REPS, len(x)))
    means = np.mean(x[samples], axis=1)
    return {"method": "paired_seed_percentile_bootstrap", "replicates": BOOTSTRAP_REPS,
            "seed": seed, "lower": float(np.quantile(means, .025)),
            "upper": float(np.quantile(means, .975))}


def normalize_error(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\\s+", " ", value).strip()
    if isinstance(value, dict):
        return {str(k): normalize_error(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [normalize_error(item) for item in value]
    return value


def invalid_identity(row: dict[str, Any], source_sha: str) -> tuple[Any, ...]:
    return (
        source_sha, row.get("model_id"), row.get("layout"), row.get("relation"),
        int(row.get("sampling_seed", -1)), bool(row.get("exact_repeat")),
        row.get("error_type"), json.dumps(normalize_error(row.get("error")), sort_keys=True, separators=(",", ":")),
    )


def source_priority(path: Path) -> int:
    """Prefer the final repair shard when historical reruns duplicate a key."""
    text = str(path)
    if "/nano_v3/" in text or "/dreamzero_v3/" in text:
        return 3
    if "/nano_v2/" in text or "/dreamzero_v2/" in text:
        return 2
    return 1


def cosine(first: np.ndarray, second: np.ndarray) -> float | None:
    a, b = first.reshape(-1), second.reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return None if denom == 0 else float(np.dot(a, b) / denom)


def paired_shift_test(deltas: list[np.ndarray]) -> dict[str, Any]:
    if not deltas:
        return {"norm_of_mean_delta": None, "mean_pairwise_cosine_agreement": None,
                "permutation_replicates": PERMUTATION_REPS, "p_value": None}
    # Average over action time, preserving action dimensions for a directional
    # shift vector.  The permutation is paired within seed and flips the
    # complete LEFT/RIGHT contrast, never individual actions.
    seed_vectors = np.asarray([d.mean(axis=0) for d in deltas], dtype=float)
    observed = float(np.linalg.norm(seed_vectors.mean(axis=0)))
    cosines = [cosine(deltas[i], deltas[j]) for i, j in combinations(range(len(deltas)), 2)]
    cosines = [x for x in cosines if x is not None]
    rng = np.random.default_rng(20260807)
    exceed = 0
    chunk = 2_000
    for start in range(0, PERMUTATION_REPS, chunk):
        count = min(chunk, PERMUTATION_REPS - start)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(count, len(seed_vectors)))
        norms = np.linalg.norm(np.mean(signs[:, :, None] * seed_vectors[None, :, :], axis=1), axis=1)
        exceed += int(np.sum(norms >= observed - 1e-15))
    return {
        "norm_of_mean_delta": observed,
        "mean_pairwise_cosine_agreement": float(np.mean(cosines)) if cosines else None,
        "permutation_method": "paired_within_seed_sign_flip",
        "permutation_replicates": PERMUTATION_REPS,
        "permutation_seed": 20260807,
        "p_value": float((exceed + 1) / (PERMUTATION_REPS + 1)),
        "per_seed_delta_mean_vectors": seed_vectors.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_files: list[dict[str, Any]] = []
    valid_rows: dict[tuple[Any, ...], dict[str, Any]] = {}
    valid_duplicates: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    invalid_ids: dict[tuple[Any, ...], dict[str, Any]] = {}

    for path in sorted(args.input_dir.rglob("requests*.jsonl")):
        source_content_sha = sha256(path)
        source_files.append({"path": str(path), "sha256": source_content_sha, "bytes": path.stat().st_size})
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "infrastructure_invalid":
                    item = {
                        "source_file": str(path), "source_sha256": source_content_sha,
                        "line": line_number, "model_id": row.get("model_id"),
                        "layout": row.get("layout"), "relation": row.get("relation"),
                        "sampling_seed": row.get("sampling_seed"), "exact_repeat": bool(row.get("exact_repeat")),
                        "error_type": row.get("error_type"), "error": normalize_error(row.get("error")),
                    }
                    invalid_rows.append(item)
                    invalid_ids.setdefault(invalid_identity(row, source_content_sha), item)
                    continue
                if row.get("status") != "valid":
                    continue
                key = (row.get("model_id"), row.get("layout"), row.get("relation"),
                       int(row.get("sampling_seed", -1)), bool(row.get("exact_repeat")))
                response = row.get("response", {})
                action = response.get("action", response.get("actions"))
                if action is None:
                    raise ValueError(f"valid row has no action: {path}:{line_number}")
                compact = {
                    "model_id": row.get("model_id"), "layout": row.get("layout"),
                    "relation": row.get("relation"), "prompt": row.get("prompt"),
                    "sampling_seed": int(row.get("sampling_seed")), "exact_repeat": bool(row.get("exact_repeat")),
                    "status": "valid", "response": {
                        "action": action,
                        "action_shape": response.get("action_shape", list(np.asarray(action).shape)),
                        "action_sha256": response.get("action_sha256", array_sha(action)),
                        "action_finite": response.get("action_finite", True),
                    },
                    "_source_file": str(path), "_source_priority": source_priority(path),
                }
                fingerprint = (compact["response"]["action_shape"], compact["response"]["action_sha256"],
                               array_sha(action))
                if key in valid_rows:
                    prior = valid_rows[key]
                    prior_fp = (prior["response"]["action_shape"], prior["response"]["action_sha256"],
                                array_sha(prior["response"]["action"]))
                    if fingerprint != prior_fp and compact["_source_priority"] == prior["_source_priority"]:
                        raise ValueError(f"conflicting duplicate valid row at equal source priority for {key}")
                    if compact["_source_priority"] > prior["_source_priority"]:
                        valid_rows[key] = compact
                    valid_duplicates.append({"key": list(key), "source_file": str(path), "line": line_number})
                else:
                    valid_rows[key] = compact

    records = list(valid_rows.values())
    exact_repeat_comparisons: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for model_id in MODELS:
        model_rows = [row for row in records if row["model_id"] in (model_id, next((k for k, v in EXPECTED.items() if v == model_id), ""))]
        for layout in LAYOUTS:
            base = {(r["sampling_seed"], r["relation"]): r for r in model_rows
                    if r["layout"] == layout and not r["exact_repeat"]}
            repeats = {(r["sampling_seed"], r["relation"]): r for r in model_rows
                       if r["layout"] == layout and r["exact_repeat"]}
            if len(base) != 54 or len(repeats) != 2:
                raise ValueError(f"{model_id}/{layout}: expected 54 base + 2 repeat rows, got {len(base)} + {len(repeats)}")
            for relation in RELATIONS:
                base_row, repeat_row = base.get((9400, relation)), repeats.get((9400, relation))
                if base_row is None or repeat_row is None:
                    raise ValueError(f"missing exact-repeat pair {model_id}/{layout}/{relation}")
                a = np.asarray(base_row["response"]["action"], dtype=float)
                b = np.asarray(repeat_row["response"]["action"], dtype=float)
                if base_row["status"] != "valid" or repeat_row["status"] != "valid":
                    raise ValueError(f"invalid exact-repeat pair {model_id}/{layout}/{relation}")
                if a.shape != b.shape:
                    raise ValueError(f"exact-repeat shape mismatch {model_id}/{layout}/{relation}")
                exact_repeat_comparisons.append({
                    "model_id": model_id, "layout": layout, "relation": relation, "sampling_seed": 9400,
                    "action_shape_equal": list(a.shape) == list(b.shape),
                    "action_shape": list(a.shape),
                    "action_sha256_equal": base_row["response"]["action_sha256"] == repeat_row["response"]["action_sha256"],
                    "np_array_equal_bit_identity": bool(np.array_equal(a, b)),
                    "numerical_rms": rms(a, b),
                })

            effects: list[float] = []
            dimension_effects: list[list[float]] = []
            deltas: list[np.ndarray] = []
            actions: dict[tuple[str, int], np.ndarray] = {}
            for seed in SEEDS:
                left = base[(seed, "left")]
                right = base[(seed, "right")]
                la, ra = np.asarray(left["response"]["action"], dtype=float), np.asarray(right["response"]["action"], dtype=float)
                if la.shape != ra.shape:
                    raise ValueError(f"prompt pair shape mismatch {model_id}/{layout}/{seed}")
                effects.append(rms(la, ra))
                dimension_effects.append(per_dimension_rms(la, ra))
                deltas.append(la - ra)
                actions[("left", seed)], actions[("right", seed)] = la, ra
            noise: dict[str, list[float]] = {"left": [], "right": []}
            for relation in RELATIONS:
                noise[relation] = [rms(actions[(relation, a)], actions[(relation, b)]) for a, b in combinations(SEEDS, 2)]
            noise_stats = {relation: quantiles(values) | {"pair_count": len(values)} for relation, values in noise.items()}
            pooled_noise = noise["left"] + noise["right"]
            native = {
                "status": "available", "matched_prompt_effect_rms": effects,
                "matched_prompt_effect_mean": float(np.mean(effects)),
                "matched_prompt_effect_median": float(np.median(effects)),
                "per_dimension_rms_by_seed": dimension_effects,
                "per_dimension_rms_mean": np.mean(np.asarray(dimension_effects), axis=0).tolist(),
                "same_prompt_cross_seed_pairwise_rms": noise_stats,
                "same_prompt_cross_seed_pooled": quantiles(pooled_noise) | {"pair_count": len(pooled_noise)},
                "prompt_to_noise_ratio": (float(np.median(effects) / np.median(pooled_noise)) if np.median(pooled_noise) != 0 else None),
                "prompt_effects_above_noise_p95_fraction": float(np.mean(np.asarray(effects) > np.quantile(pooled_noise, .95))) if pooled_noise else None,
                "paired_systematic_distribution_shift": paired_shift_test(deltas),
            }
            # E001 never forwards an action to a controller.  Preserve an
            # explicit contract record instead of silently relabeling the
            # native chunk as an executed prefix.
            prefix = {
                "status": "unavailable", "reason": "E001 is fixed-observation request-only; no action prefix was consumed by a runner",
                "native_chunk_shape": list(np.asarray(base[(9400, "left")]["response"]["action"]).shape),
            }
            metrics[f"{model_id}/{layout}"] = {
                "model_request_rows": len([r for r in model_rows if r["layout"] == layout]),
                "base_request_rows": len(base), "exact_repeat_request_rows": len(repeats),
                "matched_prompt_effect_count": len(effects),
                "native_full_returned_action_chunk": native,
                "executable_prefix": prefix,
                "semantic_fk": {"status": "unavailable", "reason": "No verified robot-state/action-frame mapping is bound to E001 request ledgers; no simulator/FK was run."},
                "layout_interaction_source": "computed after both layouts are compiled",
                "status": "complete",
            }

    for model_id in MODELS:
        control = metrics[f"{model_id}/control"]["native_full_returned_action_chunk"]["matched_prompt_effect_rms"]
        reflected = metrics[f"{model_id}/position_mirrored"]["native_full_returned_action_chunk"]["matched_prompt_effect_rms"]
        interaction = (np.asarray(reflected) - np.asarray(control)).tolist()
        for layout in LAYOUTS:
            metrics[f"{model_id}/{layout}"]["layout_interaction"] = {
                "all_27_seed_effects": interaction,
                "mean": float(np.mean(interaction)), "median": float(np.median(interaction)),
                "paired_bootstrap_95_ci": bootstrap(interaction, 20260808 + list(MODELS).index(model_id)),
                "exact_two_sided_sign_test": exact_sign_test(interaction),
                "definition": "prompt_effect_reflected(seed) - prompt_effect_control(seed)",
            }

    report = {
        "schema_version": "vla-wam-shared-v3e001-results-v3", "amendment_id": "V3-E001",
        "status": "complete" if len(records) == 336 else "partial",
        "behavioral_episode_count": 0, "registered_model_request_count": 336,
        "model_request_count": len(records), "valid_record_count": len(records),
        "raw_invalid_row_count": len(invalid_rows), "unique_invalid_attempt_count": len(invalid_ids),
        "duplicate_invalid_row_count": len(invalid_rows) - len(invalid_ids),
        "invalid_source_file_count": len({item["source_file"] for item in invalid_rows}),
        "unique_invalid_source_hash_count": len({item["source_sha256"] for item in invalid_rows}),
        "valid_duplicate_row_count": len(valid_duplicates),
        "invalid_attempt_identity_fields": ["source_content_sha256", "model", "layout", "relation", "sampling_seed", "exact_repeat", "error_type", "normalized_error_payload"],
        "deduplication_key": ["model_id", "layout", "relation", "sampling_seed", "exact_repeat"],
        "exact_repeat_comparison_count": len(exact_repeat_comparisons),
        "exact_repeat_comparisons": exact_repeat_comparisons,
        "exact_repeat_summary": {"all_12_complete": len(exact_repeat_comparisons) == 12,
                                 "all_shape_equal": all(x["action_shape_equal"] for x in exact_repeat_comparisons),
                                 "all_action_sha256_equal": all(x["action_sha256_equal"] for x in exact_repeat_comparisons),
                                 "all_bit_identical": all(x["np_array_equal_bit_identity"] for x in exact_repeat_comparisons),
                                 "all_rms_zero": all(x["numerical_rms"] == 0.0 for x in exact_repeat_comparisons)},
        "metrics": metrics, "source_files": source_files, "valid_duplicate_rows": valid_duplicates,
        "claim_boundary": "Fixed-observation prompt/noise diagnostic; no action was executed and no task success claim is made.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "valid": len(records), "raw_invalid": len(invalid_rows),
                      "unique_invalid": len(invalid_ids), "exact_repeats": len(exact_repeat_comparisons)}, indent=2))


if __name__ == "__main__":
    main()
