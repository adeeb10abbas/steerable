#!/usr/bin/env python3
"""Extract completed behavioral experiments from one immutable Git revision.

Reads compact episode rows; does not run models or inspect remote raw media.
Recomputes condition counts and paired contrasts, then reproduces the original
bootstrap draws and exact tests. An inconsistent source stops extraction.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess

import numpy as np

PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
V3 = "artifacts/vla_wam_shared_v3/"
ARMS = ("control", "position_mirrored")
SEEDS = list(range(9400, 9427))


def check(condition, message):
    if not condition:
        raise ValueError(message)


def near(actual, expected, label):
    check(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
          f"{label}: recomputed {actual!r}, source {expected!r}")


class PinnedSource:
    def __init__(self, root):
        self.root = root
        resolved = self.git("rev-parse", PIN + "^{commit}").decode().strip()
        check(resolved == PIN, "Source commit does not match the evidence pin")
        self.manifest = {}

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args])

    def read(self, path):
        data = self.git("show", PIN + ":" + path)
        expected = self.git("rev-parse", PIN + ":" + path).decode().strip()
        computed = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        check(computed == expected, f"Git blob identity failed: {path}")
        self.manifest[path] = {"git_blob": expected, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        return data

    def json(self, path):
        return json.loads(self.read(path))

    def jsonl(self, path):
        return [json.loads(line) for line in self.read(path).splitlines() if line.strip()]


def sign_test(values):
    pos = sum(x > 0 for x in values)
    neg = sum(x < 0 for x in values)
    n = pos + neg
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(pos, neg) + 1)) / 2**n) if n else 1.0
    return {"positive": pos, "negative": neg, "ties": len(values) - n, "p_value": p}


def sign_flip(values):
    """Exact two-sided sign-flip randomization, including zero-valued blocks."""
    n = len(values)
    check(n <= 27, "Exact test is bounded to the registered 27 blocks")
    observed = abs(sum(values))
    if observed <= 1e-14:
        return {"p_value": 1.0, "extreme": 2**n, "permutations": 2**n}

    def signed_sums(part):
        sums = [0.0]
        for value in part:
            sums = [s + value for s in sums] + [s - value for s in sums]
        return sums

    split = n // 2
    left = signed_sums(values[:split])
    right = sorted(signed_sums(values[split:]))
    threshold = observed - 1e-14
    extreme = sum(bisect_right(right, -threshold - x) + len(right) - bisect_left(right, threshold - x) for x in left)
    return {"p_value": extreme / 2**n, "extreme": extreme, "permutations": 2**n}


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return sorted_values[lo] * (hi - position) + sorted_values[hi] * (position - lo) if lo != hi else sorted_values[lo]


def reconcile_continuous(values, reported, *, engine):
    """Verify means, medians, CIs and whichever exact test was prespecified."""
    mean, median = statistics.fmean(values), statistics.median(values)
    near(mean, reported.get("mean_m", reported.get("mean")), "mean")
    near(median, reported.get("median_m", reported.get("median")), "median")
    interval = reported.get("mean_bootstrap_95", reported.get("bootstrap_mean95"))
    if engine == "python_random":
        generator = random.Random(interval["seed"])
        n = len(values)
        draws = sorted(statistics.fmean(values[generator.randrange(n)] for _ in range(n))
                       for _ in range(interval["replicates"]))
        low, high = percentile(draws, 0.025), percentile(draws, 0.975)
        near(low, interval["lower"], "bootstrap lower")
        near(high, interval["upper"], "bootstrap upper")
        source_test = reported["paired_sign_test"]
        test = sign_test(values)
        near(test["p_value"], source_test["p_value"], "sign-test p")
        check(test["positive"] == source_test["positive"] and test["negative"] == source_test["negative"], "Sign counts differ")
    else:
        generator = np.random.default_rng(interval["seed"])
        draws = generator.choice(np.asarray(values), size=(interval["resamples"], len(values)), replace=True).mean(axis=1)
        low, high = (float(v) for v in np.quantile(draws, [0.025, 0.975]))
        near(low, interval["low"], "bootstrap lower")
        near(high, interval["high"], "bootstrap upper")
        if "exact_layout_label_permutation" in reported:
            test = sign_flip(values)
            near(test["p_value"], reported["exact_layout_label_permutation"]["exact_two_sided_p"], "permutation p")
        else:
            test = sign_test(values)
            near(test["p_value"], reported["sign"]["exact_two_sided_p"], "sign-test p")
    return {"n_blocks": len(values), "mean": mean, "median": median,
            "mean_ci95": [low, high], "test": test,
            "bootstrap_source": interval, "reconciliation": "counts, point estimates, interval draws, and test match pinned source"}


def normalize_reflection(row, model):
    nano = model == "nano"
    y = row["measurements"]["signed_final_lateral_offset_m"] if nano else row["signed_final_lateral_offset_m"]
    check(math.isfinite(y), "Non-finite endpoint")
    action = row["artifacts"]["executed_action_trace"] if nano else row["executed_action_trace"]
    return {"cell_id": row["registered_cell_id"], "seed": row["environment_seed"] if nano else row["seed"],
            "arm": row["phase_b_arm"] if nano else row["arm"], "direction": row["requested_relation"],
            "success": row["requested_success"], "y_m": y,
            "failure_category": row["failure_taxonomy"] if nano else row["failure_category"],
            "initial_state_sha256": row["initial_state_sha256"], "pair_identity_sha256": row["initial_state_sha256"],
            "executed_action_sha256": action["sha256"]}


def block_contrasts(rows, arms):
    lookup = {(r["seed"], r["arm"], r["direction"]): r for r in rows}
    check(len(lookup) == len(rows) == 108, "Duplicate or missing behavioral cells")
    expected = {(seed, arm, side) for seed in SEEDS for arm in arms for side in ("left", "right")}
    check(set(lookup) == expected, "Incomplete or extra matched seed block")
    by_arm, conditions, pairs = {}, {}, []
    for arm in arms:
        by_arm[arm] = {"endpoint_response_m": [], "placement_depth_gap_m": [], "success_gap": []}
        for side in ("left", "right"):
            selected = [lookup[(seed, arm, side)] for seed in SEEDS]
            conditions[f"{arm}:{side}"] = {"successes": sum(r["success"] for r in selected), "episodes": len(selected),
                "failure_categories": dict(sorted(Counter(r["failure_category"] for r in selected).items()))}
        for seed in SEEDS:
            left, right = lookup[(seed, arm, "left")], lookup[(seed, arm, "right")]
            check(left["pair_identity_sha256"] == right["pair_identity_sha256"], "Pair identity differs within prompt pair")
            d = left["y_m"] - right["y_m"]
            b = -right["y_m"] - left["y_m"]
            gap = int(right["success"]) - int(left["success"])
            for key, value in (("endpoint_response_m", d), ("placement_depth_gap_m", b), ("success_gap", gap)):
                by_arm[arm][key].append(value)
            action_distinct = (left["executed_action_sha256"] != right["executed_action_sha256"]
                               if "executed_action_sha256" in left and "executed_action_sha256" in right else None)
            pairs.append({"seed": seed, "arm": arm, "endpoint_response_m": d, "placement_depth_gap_m": b,
                          "success_gap": gap, "recorded_action_hashes_differ": action_distinct})
    return by_arm, conditions, pairs


def reflection(source, model):
    prefix = V3 + {"nano": "phase_b/nano_mirror_v3b001/results/nano_v3b001", "dreamzero": "phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003"}[model]
    original_rows = source.jsonl(prefix + "_episodes.jsonl")
    rows = [normalize_reflection(row, model) for row in original_rows]
    summary = source.json(prefix + "_summary.json")
    if model == "dreamzero":
        registration = source.json(V3 + "phase_b/dreamzero_mirror_v3b003/post_result_dreamzero_mirror_v3b003_amendment.json")
        contract = registration["runtime_contract_source"]
        runtime = {key: contract[key] for key in ("checkpoint", "action_cap", "open_loop_horizon", "effective_official_model_noise_seed", "registered_seed_semantics", "instruction_controller", "frozen_predicate_id")}
        check(runtime["effective_official_model_noise_seed"] == 1140, "DreamZero sampling semantics changed")
    else:
        check(all(row["policy_seed"] == row["environment_seed"] for row in original_rows), "Nano sampling seed differs from matched block")
        runtime = {"checkpoint": original_rows[0]["checkpoint"], "action_cap": original_rows[0]["action_cap"],
                   "returned_action_chunk_shape": original_rows[0]["action_chunk_shape"],
                   "policy_seeds": sorted({row["policy_seed"] for row in original_rows}),
                   "frozen_predicate_id": original_rows[0]["predicate_id"]}
    values, conditions, pairs = block_contrasts(rows, ARMS)
    for key, result in conditions.items():
        check(result["episodes"] == summary["condition_outcomes"][key]["episodes"], "Condition denominator differs")
        check(result["successes"] == summary["condition_outcomes"][key]["successes"], "Condition success differs")
    primary = summary["full_sample_primary"]
    effects = {}
    for metric, source_key in (("endpoint_response_m", "D_by_arm"), ("placement_depth_gap_m", "B_by_arm")):
        effects[metric] = {arm: reconcile_continuous(values[arm][metric], primary[source_key][arm], engine="python_random") for arm in ARMS}
        interaction = [a - b for a, b in zip(values[ARMS[1]][metric], values[ARMS[0]][metric])]
        result_key = "J_redirection_interaction" if metric == "endpoint_response_m" else "I_position_reflection_interaction"
        effects[metric]["reflected_minus_control"] = reconcile_continuous(interaction, primary[result_key], engine="python_random")
    binary = [a - b for a, b in zip(values[ARMS[1]]["success_gap"], values[ARMS[0]]["success_gap"])]
    binary_result = {"mean": statistics.fmean(binary), "values": binary}
    if "binary_success_DiD" in primary:
        report = primary["binary_success_DiD"]
        near(binary_result["mean"], report["mean_DiD"], "Binary interaction mean")
        binary_result["test"] = sign_flip(binary)
        near(binary_result["test"]["p_value"], report["exact_permutation_test"]["p_value"], "Binary interaction p")
    else:
        binary_result["test"] = None
        binary_result["note"] = "No prespecified binary test in this source summary; no new inference added."
    return {"cohort": summary["amendment_id"], "model_id": summary["model_id"], "episodes": len(rows),
            "seed_blocks": SEEDS, "conditions": conditions, "effects": effects, "binary_interaction": binary_result,
            "endpoint_ordered_pairs_by_arm": {arm: sum(v > 0 for v in values[arm]["endpoint_response_m"]) for arm in ARMS},
            "distinct_recorded_action_hash_pairs": sum(p["recorded_action_hashes_differ"] for p in pairs),
            "action_hash_interpretation": "Hashes of retained executed-action arrays differ; raw array values were not loaded here.",
            "unique_physical_reset_fingerprints_by_arm": {arm: len({r["initial_state_sha256"] for r in rows if r["arm"] == arm}) for arm in ARMS},
            "sampling_scope": "Nano policy seeds vary with seed block; DreamZero released model-noise seed remains 1140. Seed blocks are not independent physical layouts.",
            "exact_prompts": summary["exact_prompts"], "runtime": runtime, "claim_boundary": summary["claim_boundary"],
            "original_uncertainty_contract": summary["uncertainty_contract"], "pairs": pairs, "episodes_compact": rows}


def dreamzero_symmetry(source):
    base = V3 + "phase_e/symmetric_layout_cohort_v3e004/results/"
    originals = [r for r in source.jsonl(base + "episodes.jsonl") if r["model_id"] == "dreamzero_droid_action_cfg"]
    rows = [{"cell_id": r["cell_id"], "seed": r["environment_seed"], "arm": str(int(r["symmetry_level_s"])),
             "direction": r["relation"], "success": r["success"], "y_m": r["signed_final_lateral_offset"],
             "failure_category": r["failure_category"], "initial_state_sha256": r["initial_state_sha256"],
             "pair_identity_sha256": r["request0_pair_identity_sha256"]} for r in originals]
    original_pairs = {}
    for row in originals:
        original_pairs.setdefault(row["matched_pair_id"], {})[row["relation"]] = row
    for pair in original_pairs.values():
        for field in ("request0_pair_identity_sha256", "request0_observation_payload_sha256", "request0_reset_contract_sha256", "action_distinct"):
            check(pair["left"][field] == pair["right"][field], f"Symmetry paired {field} differs")
    # E004 pairs the replayed first observation and reset contract. Full recorder
    # hashes differ, so an identical whole-state hash is not claimed.
    values, conditions, pairs = block_contrasts(rows, ("0", "1"))
    checkpoint = source.json(base + "results.json")["checkpoints"]["dreamzero_droid_action_cfg"]
    analysis = checkpoint["analysis"]
    effects = {}
    for metric, source_key, interaction_key in (("endpoint_response_m", "endpoint_redirection_LEFT_minus_RIGHT_m", "endpoint_redirection_m"),
                                               ("placement_depth_gap_m", "requested_depth_gap_R_minus_L_m", "depth_gap_m"),
                                               ("success_gap", "binary_gap_R_minus_L", "binary_gap")):
        effects[metric] = {arm: reconcile_continuous(values[arm][metric], analysis["levels"][f"{int(arm):.2f}"][source_key], engine="numpy") for arm in ("0", "1")}
        diffs = [a - b for a, b in zip(values["1"][metric], values["0"][metric])]
        report = analysis["interaction_s1_minus_s0_core"][interaction_key]
        for seed, actual, reported in zip(SEEDS, diffs, report["seed_values"]):
            check(seed == reported["seed"], "Seed interaction order differs")
            near(actual, reported["s1_minus_s0"], "Seed interaction")
        effects[metric]["symmetric_minus_reference"] = reconcile_continuous(diffs, report, engine="numpy")
    for key, result in conditions.items():
        arm, side = key.split(":")
        report = checkpoint["descriptive_progress"][f"{int(arm):.2f}/{side}"]
        check(result["successes"] == report["successes"] and result["episodes"] == report["valid_episodes"], "Symmetry condition differs")
    return {"cohort": "V3-E004", "model_id": "dreamzero_droid_action_cfg", "episodes": len(rows),
            "seed_blocks": SEEDS, "conditions": conditions, "effects": effects,
            "reported_distinct_action_pairs": sum(bool(r["action_distinct"]) for r in originals) // 2,
            "endpoint_ordered_pairs_by_arm": {arm: sum(v > 0 for v in values[arm]["endpoint_response_m"]) for arm in ("0", "1")},
            "equivalence_claim_allowed": checkpoint["claim_gate"]["equivalence_claims"],
            "scope": "Separate completed experiment; reference and symmetric object layouts. Do not pool with reflection or Phase A.",
            "physical_boundary": checkpoint["claim_gate"]["scope"],
            "pairing_check": "54/54 request-zero observation payloads, pair identities and reset contracts agree; full recorded initial-state hashes differ.",
            "pairs": [{k: v for k, v in p.items() if k != "recorded_action_hashes_differ"} for p in pairs]}


def extract(root=ROOT):
    source = PinnedSource(root)
    # Record the implementations defining the two original bootstrap engines.
    source.read("tools/compile_nano_v3b001_results.py")
    source.read("experiments/v3/phase_e/symmetric_layout_cohort_v3e004/analysis.py")
    cohorts = {"nano_reflection": reflection(source, "nano"), "dreamzero_reflection": reflection(source, "dreamzero"),
               "dreamzero_symmetry": dreamzero_symmetry(source)}
    return {"schema": "completed-world-action-model-behavior-paper-v1", "source_commit": PIN,
            "verification": "All reported condition counts, paired means, medians, seeded 95% mean bootstrap intervals and used exact tests reproduced from pinned compact rows.",
            "inference_scope": "Original within-cohort seed-block statistics; no pooled model, cohort, or arena inference. The fixed physical layouts do not establish generalization to unseen scenes.",
            "measurement_scope": "Executed robot behavior only. No claim about the accuracy or causal usefulness of generated future videos.",
            "definitions": {"y": "Final target-minus-reference lateral coordinate in robot frame; positive LEFT, metres.",
                            "D": "y_LEFT - y_RIGHT; positive means endpoints ordered with the command.",
                            "B": "(-y_RIGHT) - y_LEFT; positive means RIGHT has greater requested-side lateral depth.",
                            "success_gap": "success_RIGHT - success_LEFT per matched seed block."},
            "cohorts": cohorts, "source_manifest": source.manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=PACKAGE / "results" / "paper_results.json")
    args = parser.parse_args()
    report = extract(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cohorts": {k: v["episodes"] for k, v in report["cohorts"].items()},
                      "source_blobs_verified": len(report["source_manifest"]), "reconciliation": "passed"}))


if __name__ == "__main__":
    main()
