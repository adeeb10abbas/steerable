#!/usr/bin/env python3
"""Retrospective H01 stopping sensitivity; no model execution or media alignment.

Every default input comes from the immutable evidence commit, including when
the repository is sparse. Positive robot-frame y is LEFT. The primary common
sample is max(intersection(observed LEFT steps, observed RIGHT steps)) strictly
below BOTH episode ends. That sample time remains outcome-dependent.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
EPISODES = "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl"
SOURCE_CODE = {
    "experiments/v3/cosmos_nano_phase_b/compile_cell.py": "_in_cone; _validate_steps; build_behavioral_record",
    "experiments/v3/cosmos_nano_phase_b/robolab_bridge.py": "_sample: cube and bowl positions transformed into robot-base coordinates",
    "experiments/v3/cosmos_nano_phase_b/fixture_tasks.py": "_scene, _LeftTermination and _RightTermination",
}
FRAME = "robot_base_object_minus_reference_xyz_m"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
SIDES = ("left", "right")
# All 32-action request boundaries below the cap, plus the cap; no selected late horizon.
CHECKPOINTS = (*range(0, 450, 32), 450)
METRICS = (
    "offset_separation_m", "cube_y_separation_m", "bowl_y_separation_m",
    "offset_change_separation_m", "cube_y_change_separation_m", "bowl_y_change_separation_m",
    *(f"{side}_{name}" for side in SIDES for name in (
        "offset_m", "cube_displacement_m", "bowl_displacement_m",
        "cube_y_displacement_m", "bowl_y_displacement_m")),
)


def read_pinned_source(root):
    """Read and hash compact evidence and sign-convention source, never execute it."""
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    if git("rev-parse", PIN + "^{commit}").decode().strip() != PIN:
        raise ValueError("Evidence commit does not match pin")
    provenance = {"commit": PIN, "files": {}}
    rows = None
    for path in (EPISODES, *SOURCE_CODE):
        raw = git("show", PIN + ":" + path)
        identity = git("rev-parse", PIN + ":" + path).decode().strip()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if identity != blob:
            raise ValueError(f"Git blob identity mismatch: {path}")
        provenance["files"][path] = {"git_blob": blob, "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        if path == EPISODES:
            rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        else:
            provenance["files"][path]["evidence_functions"] = SOURCE_CODE[path]
    return rows, provenance


def describe(values):
    values = [v for v in values if v is not None]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None, "max": max(values) if values else None}


def position(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in value)):
        return None
    return tuple(float(v) for v in value)


def normalize_episode(raw):
    """Preserve missing observations; reject ambiguous identities and step indices."""
    if raw.get("measurement_frame") != FRAME:
        raise ValueError("Unknown measurement frame; do not infer a sign convention")
    end = raw.get("actions_executed")
    if end is not None and (type(end) is not int or end < 0):
        raise ValueError("actions_executed must be a nonnegative integer or missing")
    events = [e.get("action_step") for e in raw.get("event_timeline", []) if e.get("event") == "episode_end"]
    if events and events != [end]:
        raise ValueError("Conflicting episode termination metadata")
    samples = {}
    for sample in raw.get("steps") or []:
        step = sample.get("action_step")
        if type(step) is not int or step < 0 or (end is not None and step > end):
            raise ValueError("Invalid or post-termination action sample")
        if step in samples:
            raise ValueError("Duplicate action sample")
        samples[step] = {"cube": position(sample.get("object_xyz")), "bowl": position(sample.get("reference_xyz"))}
    valid = raw.get("behavioral_result_valid") is True
    return {"raw": raw, "end": end, "samples": samples, "valid": valid,
            "reason": "invalid_behavioral_episode" if not valid else ("missing_termination_step" if end is None else None)}


def sample_at(episode, step):
    if episode is None:
        return None, "missing_episode"
    if episode["reason"]:
        return None, episode["reason"]
    if step is None:
        return None, "no_common_observed_pretermination_step"
    if step > episode["end"]:
        return None, "stopped_before_checkpoint"
    sample = episode["samples"].get(step)
    if sample is None:
        return None, "step_not_observed"
    if sample["cube"] is None or sample["bowl"] is None:
        return None, "missing_or_nonfinite_position"
    return sample, None


def pair_record(left, right, *, seed, arm, mode, checkpoint=None):
    episodes = {"left": left, "right": right}
    common = None
    pair_reason = None
    if mode == "pretermination_latest_common":
        if all(e is not None and e["reason"] is None for e in episodes.values()):
            shared = set(left["samples"]) & set(right["samples"])
            candidates = [t for t in shared if t < min(left["end"], right["end"])]
            common = max(candidates) if candidates else None
            if common is None:
                pair_reason = "no_common_observed_pretermination_step"
        times = dict.fromkeys(SIDES, common)
    elif mode == "fixed_action_checkpoints":
        common = checkpoint
        times = dict.fromkeys(SIDES, checkpoint)
    elif mode == "terminal_endpoints":
        times = {side: e["end"] if e is not None else None for side, e in episodes.items()}
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")
    record = {"seed": seed, "layout": arm, "mode": mode, "common_action_step": common,
              "pair_reason": pair_reason, **dict.fromkeys(METRICS)}
    selected = {}
    for side, e in episodes.items():
        selected[side], reason = sample_at(e, times[side])
        record[f"{side}_reason"] = reason
        record[f"{side}_action_step"] = times[side]
        record[f"{side}_termination_step"] = e["end"] if e else None
        record[f"{side}_requested_success"] = e["raw"].get("requested_success") if e else None
    record["eligible"] = all(selected.values())
    if not record["eligible"]:
        return record
    for side, sample in selected.items():
        record[f"{side}_offset_m"] = sample["cube"][1] - sample["bowl"][1]
        initial = episodes[side]["samples"].get(0)
        for name in ("cube", "bowl"):
            if initial is not None and initial[name] is not None:
                record[f"{side}_{name}_displacement_m"] = math.dist(sample[name], initial[name])
                record[f"{side}_{name}_y_displacement_m"] = sample[name][1] - initial[name][1]
    record["offset_separation_m"] = record["left_offset_m"] - record["right_offset_m"]
    for name in ("cube", "bowl"):
        record[f"{name}_y_separation_m"] = selected["left"][name][1] - selected["right"][name][1]
        dl, dr = (record[f"{side}_{name}_y_displacement_m"] for side in SIDES)
        if dl is not None and dr is not None:
            record[f"{name}_y_change_separation_m"] = dl - dr
    dc, db = (record[f"{name}_y_change_separation_m"] for name in ("cube", "bowl"))
    if dc is not None and db is not None:
        record["offset_change_separation_m"] = dc - db
    return record


def summarize(records):
    eligible = [r for r in records if r["eligible"]]
    values = [r["offset_separation_m"] for r in eligible]
    ordered = sum(v > 0 for v in values)
    missing = len(records) - len(eligible)
    return {
        "expected_pairs": len(records), "eligible_pairs": len(eligible), "missing_pairs": missing,
        "eligible_seeds": [r["seed"] for r in eligible], "missing_seeds": [r["seed"] for r in records if not r["eligible"]],
        "ordered_pairs": ordered, "reversed_pairs": sum(v < 0 for v in values), "ties": sum(v == 0 for v in values),
        "ordered_fraction_observed": ordered / len(eligible) if eligible else None,
        "ordered_fraction_full_cohort_bounds": [ordered / len(records), (ordered + missing) / len(records)],
        "common_action_step": describe(r["common_action_step"] for r in eligible),
        "left_eligible_episodes": sum(r["left_reason"] is None for r in records),
        "right_eligible_episodes": sum(r["right_reason"] is None for r in records),
        "left_episode_reasons": dict(sorted(Counter(r["left_reason"] for r in records if r["left_reason"]).items())),
        "right_episode_reasons": dict(sorted(Counter(r["right_reason"] for r in records if r["right_reason"]).items())),
        "pair_reasons": dict(sorted(Counter(r["pair_reason"] for r in records if r["pair_reason"]).items())),
        "metrics": {metric: describe(r[metric] for r in eligible) for metric in METRICS},
    }


def analyze_episodes(rows, *, seeds=SEEDS, arms=ARMS, checkpoints=CHECKPOINTS):
    """Compare expected paired cells, keeping absent/truncated observations explicit."""
    seeds, arms, checkpoints = tuple(seeds), tuple(arms), tuple(checkpoints)
    if not seeds or not arms or len(set(seeds)) != len(seeds) or len(set(arms)) != len(arms):
        raise ValueError("Expected seeds and layouts must be nonempty and unique")
    if any(type(t) is not int or t < 0 for t in checkpoints) or len(set(checkpoints)) != len(checkpoints):
        raise ValueError("Checkpoints must be unique nonnegative action counts")
    indexed = {}
    identities = set()
    for raw in rows:
        key = (raw.get("policy_seed"), raw.get("phase_b_arm"), raw.get("requested_relation"))
        if key[0] not in seeds or key[1] not in arms or key[2] not in SIDES:
            raise ValueError(f"Unexpected cohort cell: {key}")
        if raw.get("environment_seed") != key[0]:
            raise ValueError("Policy/environment seed mismatch in H01 paired block")
        identity = raw.get("registered_cell_id")
        if not identity or identity in identities or key in indexed:
            raise ValueError("Missing or duplicate cell identity")
        identities.add(identity)
        indexed[key] = normalize_episode(raw)
    for seed in seeds:
        pair_ids = {e["raw"].get("pair_id") for key, e in indexed.items() if key[0] == seed}
        if len(pair_ids) > 1 or None in pair_ids:
            raise ValueError("Pair identity mismatch")
    valid = [e for e in indexed.values() if e["valid"]]
    expected_count = len(seeds) * len(arms) * 2
    cohort = {
        "expected_episodes": expected_count, "observed_episodes": len(rows),
        "missing_episodes": expected_count - len(rows), "valid_episodes": len(valid),
        "invalid_episodes": len(rows) - len(valid),
        "valid_successes": sum(e["raw"].get("requested_success") is True for e in valid),
        "valid_behavioral_failures": sum(e["raw"].get("requested_success") is False for e in valid),
        "right_censored_episodes": sum(e["raw"].get("right_censored") is True for e in valid),
        "seed_count": len(seeds), "seeds": list(seeds), "layout_count": len(arms),
        "unique_initial_states_by_layout": {
            arm: len({e["raw"]["initial_state_sha256"] for key, e in indexed.items()
                      if key[1] == arm and e["raw"].get("initial_state_sha256")}) for arm in arms},
        "episodes_with_missing_action_samples": sum(e["end"] is not None and set(e["samples"]) != set(range(e["end"] + 1)) for e in indexed.values()),
        "missing_action_sample_count": sum(len(set(range(e["end"] + 1)) - set(e["samples"])) for e in indexed.values() if e["end"] is not None),
        "missing_termination_steps": sum(e["end"] is None for e in indexed.values()),
        "samples_with_missing_positions": sum(s["cube"] is None or s["bowl"] is None for e in indexed.values() for s in e["samples"].values()),
        "observed_action_samples": sum(len(e["samples"]) for e in indexed.values()),
    }
    analyses = {"pretermination_latest_common": {}, "terminal_endpoints": {}, "fixed_action_checkpoints": {str(t): {} for t in checkpoints}}
    detail = []
    for mode in ("pretermination_latest_common", "terminal_endpoints", "fixed_action_checkpoints"):
        for checkpoint in checkpoints if mode == "fixed_action_checkpoints" else (None,):
            for arm in arms:
                records = [pair_record(indexed.get((seed, arm, "left")), indexed.get((seed, arm, "right")),
                                       seed=seed, arm=arm, mode=mode, checkpoint=checkpoint) for seed in seeds]
                target = analyses[mode][str(checkpoint)] if checkpoint is not None else analyses[mode]
                target[arm] = summarize(records)
                if mode == "pretermination_latest_common":
                    detail.extend(records)
    return {"schema_version": "nano-h01-stopping-controls-v1", "cohort": cohort,
            "fixed_action_checkpoints": list(checkpoints), "analyses": analyses, "pair_records": detail}


def write_results(result, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stopping_controls.json").write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    rows = []
    for mode, analyses in result["analyses"].items():
        checkpoints = analyses.items() if mode == "fixed_action_checkpoints" else [(None, analyses)]
        for checkpoint, layouts in checkpoints:
            for layout, summary in layouts.items():
                row = {"comparison": mode, "action_checkpoint": checkpoint, "layout": layout}
                for key, value in summary.items():
                    if key == "metrics":
                        for metric, description in value.items():
                            row.update({f"{metric}_{stat}": v for stat, v in description.items()})
                    elif key == "common_action_step":
                        row.update({f"common_action_step_{stat}": v for stat, v in value.items()})
                    else:
                        row[key] = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                rows.append(row)
    with (output_dir / "stopping_controls.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "results")
    args = parser.parse_args()
    rows, provenance = read_pinned_source(args.repo_root)
    result = analyze_episodes(rows)
    result["provenance"] = provenance
    result["definitions"] = {
        "status": "retrospective descriptive sensitivity analysis of historical H01; not prospective fixed-duration evidence",
        "coordinate_frame": FRAME,
        "offset_m": "s = cube_robot_y - bowl_robot_y; positive is robot LEFT",
        "offset_separation_m": "D = s_LEFT - s_RIGHT; D > 0 denotes the ordered LEFT/RIGHT response",
        "coordinate_evidence": "Pinned compile_cell._in_cone uses +delta_y for LEFT and -delta_y for RIGHT; robolab_bridge._sample rotates both centers into the robot-base frame. No camera/image sign is used.",
        "pretermination_latest_common": "For each seed and layout, max observed step shared by LEFT and RIGHT that is strictly below min(actions_executed_LEFT, actions_executed_RIGHT). Select by recorded step indices before checking coordinate availability.",
        "terminal_endpoints": "Each episode at its own actions_executed, including failures at the cap. These unequal-time endpoints are shown only as the goal-stopping comparison, not proof of directional response.",
        "fixed_action_checkpoints": "Every 32-action boundary below 450, plus 450. Requires the exact observed sample from both episodes, allowing a terminal sample at exactly the checkpoint. Never extrapolate, interpolate, or carry a stopped trace forward.",
        "cube_bowl_decomposition": "D = (cube_y_LEFT - cube_y_RIGHT) - (bowl_y_LEFT - bowl_y_RIGHT). Change versions subtract each episode's step-0 state. Cube/bowl displacement is Euclidean endpoint-to-initial displacement, not path length; reported on the same eligible pair subset.",
        "physical_time": "Action counts are simulator action indices. Equal action count is not established physical-time or generated-video alignment; no FPS conversion is inferred.",
        "time_selection": "Pretermination times remain outcome-dependent because historical success ends episodes. Later fixed steps condition on remaining observable; all expected 27 pairs per layout remain in eligibility denominators.",
        "missingness": "Null remains unavailable. Invalid episodes stay separate from valid behavioral failures. Means/medians use eligible pairs; each metric supplies its own n for missing initial states. Ordered-fraction bounds give every unobserved pair the unfavorable/favorable ordering without filling any coordinates.",
        "ordering_ties": "Exact zero is a tie; strict positive/negative ordering has no calibrated motion threshold. Tiny early-step separations must not be interpreted as substantive motion.",
        "scene_count": "One prescribed base scene and its position-reflected counterpart: two fixed layout states, each repeated under 27 matched policy/environment seed labels 9400–9426. These are not 27 independent scenes. The retained initial-state hash is identical across all 54 episodes within each layout.",
        "confirmation_scope": "Descriptive results only; no confirmation p-values, independent-scene confidence intervals, new rollout, or evidence of generated-forecast accuracy.",
    }
    result["cohort"]["prescribed_base_scenes"] = 1
    result["cohort"]["prescribed_layout_states"] = 2
    write_results(result, args.output_dir)
    print(json.dumps({arm: result["analyses"]["pretermination_latest_common"][arm] for arm in ARMS}, indent=2))


if __name__ == "__main__":
    main()
