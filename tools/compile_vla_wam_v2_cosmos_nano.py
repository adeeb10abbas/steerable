#!/usr/bin/env python3
"""Compile the frozen six-cell V2-A011 Cosmos3 Nano DROID gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import compile_vla_wam_v2_cosmos_edge as shared
import numpy as np

MODEL_ID = "cosmos3_nano_policy_droid"
REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
FRAMEWORK_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
SIMULATOR_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--robolab-output", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--trajectory-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--fixed-observation-gate", type=Path, required=True)
    parser.add_argument("--invalid-attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiled-at-git-head", required=True)
    args = parser.parse_args()
    seeds = sorted(set(args.seeds))
    if seeds != [8300, 8301, 8302]:
        raise RuntimeError("V2-A011 compilation requires exactly seeds 8300, 8301, and 8302")
    registry = json.loads(args.registry.read_text())
    gate = json.loads(args.fixed_observation_gate.read_text())
    if registry["checkpoint"]["revision"] != REVISION or registry["checkpoint"]["hash_gate_passed"] is not True:
        raise RuntimeError("Nano exact-revision checkpoint hash gate has not passed")
    if (
        gate.get("schema_version") != "vla-wam-shared-v2-cosmos3-nano-policy-droid-fixed-observation-v1"
        or gate.get("status") != "passed"
        or gate.get("checkpoint_revision") != REVISION
    ):
        raise RuntimeError("Nano fixed-observation gate has not passed")
    invalid_attempts = json.loads(args.invalid_attempts.read_text())
    attempts = invalid_attempts.get("attempts")
    if not isinstance(attempts, list):
        raise TypeError("Nano invalid-attempt ledger must contain an attempts list")

    shared.MODEL_ID = MODEL_ID
    episodes = []
    actions = {}
    fingerprints = {}
    for seed in seeds:
        for direction in shared.TASKS:
            episode, action, fingerprint = shared.load_episode(
                seed=seed,
                direction=direction,
                output_root=args.robolab_output,
                raw_root=args.raw_root,
                trajectory_dir=args.trajectory_dir,
                output_prefix="v2_cosmos_nano",
                policy_id="cosmos3_nano_v2",
            )
            metadata = json.loads(Path(episode["executed_action_trace_metadata"]["path"]).read_text())
            if (
                metadata.get("model_id") != MODEL_ID
                or metadata.get("checkpoint_revision") != REVISION
                or metadata.get("amendment_id") != "V2-A011"
            ):
                raise RuntimeError(f"Nano trace identity mismatch for {seed}:{direction}")
            if any(item.get("server_sampling_seed") != seed for item in metadata["requests"]):
                raise RuntimeError(f"Nano request seed was not executed exactly for {seed}:{direction}")
            episodes.append(episode)
            actions[(seed, direction)] = action
            fingerprints[(seed, direction)] = fingerprint

    pairs = []
    for seed in seeds:
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        if fingerprints[(seed, "left")] != fingerprints[(seed, "right")]:
            raise RuntimeError(f"Physical initial state mismatch inside pair {seed}")
        overlap = min(len(actions[(seed, "left")]), len(actions[(seed, "right")]))
        delta = actions[(seed, "left")][:overlap].astype(np.float64) - actions[(seed, "right")][:overlap].astype(np.float64)
        first = delta[: min(shared.ACTION_HORIZON, overlap)]
        shift = right["final_lateral_display_m"] - left["final_lateral_display_m"]
        pairs.append({
            "pair_id": f"droid_pair_seed_{seed}",
            "environment_seed": seed,
            "left_success": left["requested_success"],
            "right_success": right["requested_success"],
            "right_minus_left_endpoint_lateral_m": shift,
            "endpoint_response_direction": "aligned" if shift > 0 else "anti_directed" if shift < 0 else "none",
            "first_chunk_action_rms": float(np.sqrt(np.mean(np.square(first)))),
            "overlap_action_rms": float(np.sqrt(np.mean(np.square(delta)))),
            "executed_actions_distinct": bool(not np.array_equal(actions[(seed, "left")][:overlap], actions[(seed, "right")][:overlap])),
            "physical_initial_state_sha256": fingerprints[(seed, "left")],
        })
    by_direction = {}
    for direction in shared.TASKS:
        rows = [row for row in episodes if row["requested_relation"] == direction]
        by_direction[direction] = {
            "episodes": len(rows),
            "successes": sum(row["requested_success"] for row in rows),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(row["ever_entered_requested_region"] for row in rows),
        }
    left_successes = by_direction["left"]["successes"]
    right_successes = by_direction["right"]["successes"]
    competence = (
        "both_directions" if left_successes and right_successes
        else "left_only" if left_successes else "right_only" if right_successes else "zero_direction"
    )
    payload = {
        "schema_version": "vla-wam-shared-v2-cosmos3-nano-policy-droid-result-v1",
        "status": "complete",
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "compiled_at_git_head": args.compiled_at_git_head,
        "model_id": MODEL_ID,
        "checkpoint_revision": REVISION,
        "server_repository_commit": FRAMEWORK_COMMIT,
        "simulator_repository_commit": SIMULATOR_COMMIT,
        "amendment_id": "V2-A011",
        "arena": "droid_robolab",
        "seeds": seeds,
        "registry": shared.file_record(args.registry),
        "fixed_observation_gate": shared.file_record(args.fixed_observation_gate),
        "invalid_attempt_ledger": shared.file_record(args.invalid_attempts),
        "invalid_attempt_count": len(attempts),
        "invalid_attempt_denominator_policy": "excluded",
        "measurement": {
            "oracle_actions": 0,
            "dynamic_prompts": 0,
            "subtask_progress_checking": False,
            "prompt_controller": "episode_static",
            "simulator_state_role": "post_action_scoring_and_visualization_only",
            "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
            "open_loop_horizon": shared.ACTION_HORIZON,
        },
        "summary": {
            "episode_count": len(episodes),
            "pair_count": len(pairs),
            "successes": sum(row["requested_success"] for row in episodes),
            "by_direction": by_direction,
            "failure_stage_counts": dict(sorted(Counter(row["failure_stage"] for row in episodes).items())),
            "aligned_endpoint_pairs": sum(pair["endpoint_response_direction"] == "aligned" for pair in pairs),
            "nonzero_first_chunk_pairs": sum(pair["first_chunk_action_rms"] > 0 for pair in pairs),
            "competence_gate": competence,
        },
        "pairs": pairs,
        "episodes": episodes,
        "claim_boundary": "Separate V2-A011 Nano gate; never pooled with Cosmos3 Edge, Cosmos-Reason2, or RoboTwin.",
    }
    shared.dump_json(args.output, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
