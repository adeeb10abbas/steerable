#!/usr/bin/env python3
"""Analyze V4 policy-episode throughput limits and completion estimates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_gpu_scheduling as gpu_scheduling  # noqa: E402
import v4_wave_admission as admission  # noqa: E402

DEFAULT_C7_PROGRESS = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260908_c7_wave006_progress_repaired_path.json"
)
DEFAULT_G4_REPORT = (
    ROOT / "artifacts/online_correction_v4/setup/g4_policy_family_hardware_report_20260908.json"
)

C7_EPISODE_MINUTES_PER_LANE = 25.0
C7_TARGET_EPISODES = 768
WAVE007_LANE_COUNT = 23


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_policy_throughput(
    *,
    c7_progress_path: Path = DEFAULT_C7_PROGRESS,
    g4_report_path: Path = DEFAULT_G4_REPORT,
    attempt_summary: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    c7 = _load_json(c7_progress_path)
    g4 = _load_json(g4_report_path)
    healthy_pairs = int(c7.get("lanes_running_policy_sim_pairs") or 15)
    terminal = int(c7.get("wave006_terminal_episode_count") or 0)
    in_progress = int(c7.get("wave006_in_progress_episode_count") or 0)
    remaining_c7 = max(0, C7_TARGET_EPISODES - terminal - in_progress)

    c2_active = 0
    if attempt_summary and "g3rb20260908v" in attempt_summary:
        c2_active = int(attempt_summary["g3rb20260908v"].get("active") or 0)

    pool = admission.GPU_POOL_SIZES
    a10080_pool = pool["NVIDIA-A100-SXM4-80GB"]
    a40_pool = pool["NVIDIA-A40"]

    # Each C7 lane pair consumes one A100-80GB (policy) and one A40 (simulator).
    a10080_for_c7 = healthy_pairs
    a10080_for_c2 = c2_active
    a10080_headroom = max(0, a10080_pool - a10080_for_c7 - a10080_for_c2)
    a40_headroom = max(0, a40_pool - healthy_pairs)

    max_safe_c7_pairs_now = min(
        healthy_pairs + int(a10080_headroom > 0) * 0,  # no expansion while C2 dominates
        a40_pool,
        a10080_pool - max(0, c2_active - 0),
    )
    # After C2 G3 qual completes (~115 seeds), C2 active drops to 0; wave-007 can use up to:
    max_c7_pairs_after_c2 = min(WAVE007_LANE_COUNT, a10080_pool, a40_pool)

    c7_hours_now = (remaining_c7 / max(healthy_pairs, 1)) * (C7_EPISODE_MINUTES_PER_LANE / 60.0)
    c7_hours_best = (remaining_c7 / max(max_c7_pairs_after_c2, 1)) * (
        C7_EPISODE_MINUTES_PER_LANE / 60.0
    )

    achievable = admission.ACHIEVABLE_EPISODE_COUNTS
    qual = admission.build_achievable_episode_estimates(attempt_summary)

    return {
        "schema_version": "v4-policy-episode-throughput-analysis-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "achievable_episode_total": sum(achievable.values()),
        "blocked_episode_total": sum(admission.BLOCKED_EPISODE_COUNTS.values()),
        "concurrency_limit_root_cause": {
            "primary": "rendered_lane_pairs_and_frozen_hardware_stratum",
            "detail": (
                "C7 confirmatory is frozen on a10080-policy_a40-simulator "
                f"({gpu_scheduling.POLICY_EPISODE_HARDWARE_STRATUM['hardware_stratum']}). "
                "Each lane pair binds one A100-80GB policy server and one A40 simulator. "
                f"Currently {healthy_pairs} healthy pairs run (17 dispatched, 2 sim failures). "
                f"C2 G3 qual consumes {c2_active} A100-80GB lanes in parallel, leaving "
                f"{a10080_headroom} headroom on a {a10080_pool}-GPU pool. "
                "Policy server startup is not the bottleneck once lanes are bound; "
                "episode wall-clock is dominated by cosmos-nano query cadence (~25 min/episode/lane)."
            ),
            "not_the_bottleneck": [
                "policy_server_cold_start_after_lane_bind",
                "A40 pool exhaustion while C2 qual runs on A100-80GB",
            ],
        },
        "c7_object_pair": {
            "target_episodes": C7_TARGET_EPISODES,
            "terminal_episodes": terminal,
            "in_progress_episodes": in_progress,
            "remaining_episodes": remaining_c7,
            "healthy_lane_pairs": healthy_pairs,
            "minutes_per_episode_per_lane_assumed": C7_EPISODE_MINUTES_PER_LANE,
            "completion_hours_at_current_pairs": round(c7_hours_now, 1),
            "completion_hours_after_c2_qual_with_wave007": round(c7_hours_best, 1),
            "wave_007_lane_count_planned": WAVE007_LANE_COUNT,
            "max_safe_pairs_after_c2_qual": max_c7_pairs_after_c2,
            "orphaned_policy_lanes": c7.get("orphaned_policy_lanes") or [],
        },
        "qualification_before_policy_episodes": qual,
        "per_stratum_g4_expansion": {
            "c7_confirmatory": {
                "would_legitimately_expand_throughput": False,
                "reason": (
                    "Moving C7 off a10080-policy_a40-simulator requires a new G4 hardware "
                    "receipt and runtime-lock amendment per "
                    f"{g4_report_path.name}. Current bottleneck is lane-pair count and "
                    "A100-80GB sharing with C2 qual, not attestation of A40 sim leg."
                ),
                "owner": "C7 campaign agent (object_pair confirmatory)",
                "governing_receipt": (
                    "artifacts/online_correction_v4/qualification/"
                    "object_pair_g4_nano_a10080_a40_hardware_g4c7a100q20260906b.json"
                ),
            },
            "c2_reference_binding": {
                "would_legitimately_expand_throughput": False,
                "reason": "Policy episodes not yet authorized; still in G3 qual. Agent C owns C2 gates.",
                "owner": "Agent C (reference_binding)",
            },
            "c6_containment": {
                "would_legitimately_expand_throughput": True,
                "reason": (
                    "Model-blind G3 on A100-40GB parallel stratum is legitimate after "
                    "per-product smoke attestation; does not require G4 (simulator-only gate)."
                ),
                "owner": "Agent B (containment)",
            },
        },
        "honest_6400_episode_completion_hours": {
            "note": (
                "Families other than C7 cannot start policy episodes until G3 gates and "
                "runtime locks release. Estimate sums qual wall-clock then serial policy collection."
            ),
            "c7_only_at_current_15_pairs": round(c7_hours_now, 1),
            "c7_at_best_safe_23_pairs_after_c2_qual": round(c7_hours_best, 1),
            "full_6400_sequential_best_case": round(
                c7_hours_best
                + (4096 / 32) * (C7_EPISODE_MINUTES_PER_LANE / 60.0)  # assume 32 C2 lanes post-qual
                + (768 / 16) * (C7_EPISODE_MINUTES_PER_LANE / 60.0)  # C6 after G3
                + (768 / 16) * (C7_EPISODE_MINUTES_PER_LANE / 60.0),  # C8 after G3
                1,
            ),
        },
        "pool_headroom_snapshot": {
            "NVIDIA-A100-SXM4-80GB": {
                "pool_size": a10080_pool,
                "c7_policy_lanes": a10080_for_c7,
                "c2_g3_qual_lanes": a10080_for_c2,
                "headroom": a10080_headroom,
            },
            "NVIDIA-A40": {
                "pool_size": a40_pool,
                "c7_simulator_lanes": healthy_pairs,
                "headroom": a40_headroom,
            },
            "NVIDIA-A100-SXM4-40GB": {
                "pool_size": pool["NVIDIA-A100-SXM4-40GB"],
                "assigned_to_c6_parallel_stratum": 0,
                "status": "idle_available_for_agent_b",
            },
            "NVIDIA-B200": {
                "pool_size": pool["NVIDIA-B200"],
                "status": "idle_alternate_for_c6_attestation",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c7-progress", type=Path, default=DEFAULT_C7_PROGRESS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    payload = analyze_policy_throughput(c7_progress_path=args.c7_progress)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
