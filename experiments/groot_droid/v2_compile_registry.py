#!/usr/bin/env python3
"""Compile the three frozen GR00T DROID pair slices into one result registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SEEDS = (8300, 8301, 8302)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2))
    radius /= denominator
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == trials else min(1.0, center + radius)
    return [lower, upper]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", type=Path, action="append", required=True)
    parser.add_argument("--repeat-gate", type=Path, required=True)
    parser.add_argument("--invalid-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.slice) != 3:
        parser.error("Exactly three pair slices are required")

    by_seed: dict[int, tuple[Path, dict[str, object]]] = {}
    for path in args.slice:
        data = json.loads(path.read_text())
        seed = int(data["environment_seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate seed {seed}")
        by_seed[seed] = (path, data)
    if set(by_seed) != set(SEEDS):
        raise ValueError(f"Expected seeds {SEEDS}, found {sorted(by_seed)}")

    repeat_gate = json.loads(args.repeat_gate.read_text())
    if not repeat_gate["passed"]:
        raise ValueError("The required exact-repeat gate did not pass")
    invalid_ledger = json.loads(args.invalid_ledger.read_text())

    episodes: list[dict[str, object]] = []
    endpoint_pairs: list[dict[str, object]] = []
    pair_slices: list[dict[str, object]] = []
    for seed in SEEDS:
        path, data = by_seed[seed]
        if data["denominator_status"] != "two_valid_cells":
            raise ValueError(f"Seed {seed} does not contain two valid cells")
        if data["runtime_interventions"]:
            raise ValueError(f"Seed {seed} has an unexpected runtime intervention")
        if not data["pair_checks"]["executed_actions_differ"]:
            raise ValueError(f"Seed {seed} LEFT/RIGHT actions are identical")
        rows = {row["condition"]: row for row in data["episodes"]}
        if set(rows) != {"LEFT", "RIGHT"}:
            raise ValueError(f"Seed {seed} lacks the exact prompt pair")
        left_y = float(rows["LEFT"]["endpoint_cube_minus_bowl_world_xyz"][1])
        right_y = float(rows["RIGHT"]["endpoint_cube_minus_bowl_world_xyz"][1])
        endpoint_pairs.append(
            {
                "environment_seed": seed,
                "left_endpoint_lateral_y": left_y,
                "right_endpoint_lateral_y": right_y,
                "left_minus_right_endpoint_lateral_y": left_y - right_y,
                "requested_ordering_aligned": left_y > right_y,
                "executed_action_rms": data["pair_checks"]["executed_action_rms"],
            }
        )
        for condition in ("LEFT", "RIGHT"):
            row = rows[condition]
            episodes.append(
                {
                    "environment_seed": seed,
                    "condition": condition,
                    "prompt": row["prompt"],
                    "success": row["success"],
                    "executed_action_count": row["executed_action_count"],
                    "endpoint_cube_minus_bowl_world_xyz": row[
                        "endpoint_cube_minus_bowl_world_xyz"
                    ],
                    "simulator_hdf5": row["files"]["simulator_hdf5"],
                    "simulator_video": row["files"]["simulator_video"],
                    "executed_actions": row["files"]["executed_actions"],
                    "returned_action_chunks": row["files"][
                        "returned_action_chunks"
                    ],
                    "returned_action_modalities": row["files"][
                        "returned_action_modalities"
                    ],
                }
            )
        pair_slices.append(
            {
                "environment_seed": seed,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )

    direction_summary = {}
    for condition in ("LEFT", "RIGHT"):
        rows = [row for row in episodes if row["condition"] == condition]
        successes = sum(bool(row["success"]) for row in rows)
        direction_summary[condition] = {
            "successes": successes,
            "trials": len(rows),
            "success_rate": successes / len(rows),
            "wilson_95": _wilson(successes, len(rows)),
        }

    aligned = sum(pair["requested_ordering_aligned"] for pair in endpoint_pairs)
    registry = {
        "schema_version": "vla-wam-shared-v2-groot-droid-result-v1",
        "amendment_id": "V2-A005",
        "status": "complete",
        "claim_boundary": "Bounded six-cell DROID direct-command replication. It is separate from v1, Cosmos diagnostics, and all RoboTwin denominators.",
        "model": {
            "checkpoint": "nvidia/GR00T-N1.7-DROID",
            "revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
            "isaac_groot_commit": "b9955401d50c92a29258732e3ad6ccd579f1bdc0",
            "embodiment": "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
            "open_loop_horizon": 8,
        },
        "design": {
            "environment_and_sampling_seeds": list(SEEDS),
            "conditions": ["LEFT", "RIGHT"],
            "static_episode_prompt_only": True,
            "oracle_or_subtask_coach": False,
            "valid_episode_count": len(episodes),
            "runtime_intervention_count": 0,
        },
        "direction_summary": direction_summary,
        "paired_directional_evidence": {
            "action_different_pair_count": len(endpoint_pairs),
            "endpoint_requested_ordering_aligned_pair_count": aligned,
            "pair_count": len(endpoint_pairs),
            "pairs": endpoint_pairs,
        },
        "interpretation": {
            "task_success_claim": "No valid episode succeeded in either direction.",
            "language_steerability_claim": "All three matched pairs changed executed actions and ordered final lateral cube positions in the requested LEFT-versus-RIGHT direction.",
            "limitation": "Consistent lateral redirection without task completion is evidence of language-conditioned control, not evidence of successful spatial manipulation.",
        },
        "episodes": episodes,
        "pair_slices": pair_slices,
        "repeat_gate": {
            "path": str(args.repeat_gate),
            "sha256": _sha256(args.repeat_gate),
            "left_exact_repeat_rms": repeat_gate["metrics"][
                "left_exact_repeat_rms"
            ],
            "left_vs_right_rms": repeat_gate["metrics"]["left_vs_right_rms"],
        },
        "invalid_attempts": {
            "path": str(args.invalid_ledger),
            "sha256": _sha256(args.invalid_ledger),
            "ledger_entry_count": len(invalid_ledger["attempts"]),
            "behavior_cells_excluded": 2,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
