#!/usr/bin/env python3
"""C7 productive-drain → released-shard dispatch → C8 A40 handoff sequencing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_capacity_arbitration as arbitration  # noqa: E402
import v4_gpu_pool_sweep as sweep  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

DEFAULT_PLAN_OUT = (
    ROOT
    / "artifacts/online_correction_v4/execution/gpu_widen_20260908/c7_released_shard_sequencing_plan.json"
)
DEFAULT_PROTECT_LIST = gpu_scheduling.DEFAULT_PROTECT_LIST
REACHABILITY_ANALYSIS = (
    ROOT
    / "artifacts/online_correction_v4/execution/c7_object_pair_20260906/c7_episode_reachability_analysis_20260908.json"
)
RELEASED_SHARD_ATTEMPTS = ("attempt0603", "attempt0604", "attempt0605", "attempt0606")
RETRY_R3_ATTEMPTS = (
    "attempt0589",
    "attempt0590",
    "attempt0591",
    "attempt0592",
    "attempt0593",
    "attempt0594",
)
LANE_JOB_RE = re.compile(
    r"^v4-(?P<lane>c7m\d+)-(?P<attempt>attempt\d+)-(?P<hash>[a-f0-9]+)-(?P<role>policy|sim)$",
    re.I,
)


def productive_c7_lane_attempt_keys(protect_list: Mapping[str, Any]) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in protect_list.get("productive_c7_lane_pairs") or []:
        if isinstance(row, dict) and row.get("lane_id") and row.get("attempt_id"):
            keys.add((str(row["lane_id"]), str(row["attempt_id"])))
    return keys


def fetch_productive_c7_jobs(
    *,
    kube_context: str,
    namespace: str,
    protect_list: Mapping[str, Any],
) -> list[dict[str, str]]:
    keys = productive_c7_lane_attempt_keys(protect_list)
    rows: list[dict[str, str]] = []
    for job in sweep.fetch_jobs(kube_context=kube_context, namespace=namespace):
        name = str(job.get("metadata", {}).get("name") or "")
        match = LANE_JOB_RE.match(name)
        if not match:
            continue
        lane_id = str(match.group("lane"))
        attempt_id = str(match.group("attempt"))
        if (lane_id, attempt_id) not in keys:
            continue
        rows.append(
            {
                "job": name,
                "lane_id": lane_id,
                "attempt_id": attempt_id,
                "role": str(match.group("role")),
            }
        )
    return sorted(rows, key=lambda row: (row["lane_id"], row["role"], row["job"]))


def fetch_retry_r3_jobs(*, kube_context: str, namespace: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for job in sweep.fetch_jobs(kube_context=kube_context, namespace=namespace):
        name = str(job.get("metadata", {}).get("name") or "")
        match = LANE_JOB_RE.match(name)
        if not match:
            continue
        if str(match.group("attempt")) not in RETRY_R3_ATTEMPTS:
            continue
        rows.append(
            {
                "job": name,
                "lane_id": str(match.group("lane")),
                "attempt_id": str(match.group("attempt")),
                "role": str(match.group("role")),
            }
        )
    return sorted(rows, key=lambda row: row["job"])


def build_sequencing_plan(
    *,
    kube_context: str,
    namespace: str,
    protect_list: Mapping[str, Any],
    reachability: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reachability = reachability or {}
    productive_jobs = fetch_productive_c7_jobs(
        kube_context=kube_context, namespace=namespace, protect_list=protect_list
    )
    retry_jobs = fetch_retry_r3_jobs(kube_context=kube_context, namespace=namespace)
    raw_pods = sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    pods = [p for p in (sweep.parse_pod(item) for item in raw_pods) if p is not None]
    c7_state = sweep.lane_pair_scheduling_state(pods, lane_prefix="c7m")
    c8_state = sweep.lane_pair_scheduling_state(pods, lane_prefix="c8m")
    usage = sweep.summarize_gpu_usage(pods)
    return {
        "schema_version": "v4-c7-released-shard-sequencing-plan-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "reachability_analysis": str(REACHABILITY_ANALYSIS.relative_to(ROOT)),
        "compiled_accepted": 580,
        "missing_episodes": 188,
        "missing_only_on_released_shards": True,
        "released_shard_attempts": list(RELEASED_SHARD_ATTEMPTS),
        "retry_r3_attempts_superseded_skip": list(RETRY_R3_ATTEMPTS),
        "phases": [
            {
                "phase": "1_productive_drain",
                "status": "in_progress",
                "action": "Hold protect list; let 17 productive C7 pairs finish current shards.",
                "productive_lane_jobs": productive_jobs,
                "do_not_teardown_until": "Agent B compile shows 0 remaining episodes on productive shards",
            },
            {
                "phase": "2_teardown_productive_pairs",
                "status": "blocked_until_phase_1",
                "action": "Coordinate with Agent B; delete whole productive lane pairs to free A40.",
                "job_ids_to_delete": [row["job"] for row in productive_jobs],
                "protect_list_update": "Remove productive_c7_lane_pairs section after in-flight episodes complete",
            },
            {
                "phase": "3_skip_retry_r3",
                "status": "pending",
                "action": "Delete suspended retry r3 0589-0594 (strict subset; no net new coverage).",
                "job_ids_to_delete": [row["job"] for row in retry_jobs],
            },
            {
                "phase": "4_dispatch_released_shards",
                "status": "pending",
                "action": "Dispatch attempt0603-0606 only (220 eps, 188 net new).",
                "bundle_root": "artifacts/online_correction_v4/execution/c7_object_pair_20260906/rendered-retry-runnerfix-20260908",
                "owner": "Agent B render/dispatch; capacity agent admits when A40 headroom exists",
            },
            {
                "phase": "5_route_a40_to_c8",
                "status": "pending",
                "action": "Up to 20 C8 lane pairs per confirmatory handoff plan.",
                "max_c8_lane_pairs": arbitration.G7_C8_CONFIRMATORY_MAX_LANES,
            },
        ],
        "scheduling_state": {"c7": c7_state, "c8": c8_state},
        "pool_utilization": usage,
        "wall_clock_estimates": {
            "c7_released_188_hours_at_17_lanes": round((188 / 17) * (25 / 60), 1),
            "c8_confirmatory_684_remaining_hours_at_20_lanes": round((684 / 20) * (25 / 60), 1),
            "c8_confirmatory_684_remaining_hours_at_2_lanes": round((684 / 2) * (25 / 60), 1),
        },
        "agent_b_coordination": {
            "protect_list_path": str(DEFAULT_PROTECT_LIST.relative_to(ROOT)),
            "phase_2_requires_agent_b_drain_signal": True,
            "never_delete_in_flight_episode_pods": True,
        },
    }


def build_phase2_protect_list_update(protect_list: Mapping[str, Any]) -> dict[str, Any]:
    """Protect list after productive drain: protect released shards, not productive pairs."""
    updated = json.loads(json.dumps(protect_list))
    updated["productive_c7_lane_pairs"] = []
    updated["c7_retry_shards_in_flight_do_not_preempt"] = []
    updated["released_shard_queue_pending_redispatch"]["status"] = "ready_for_dispatch_post_teardown"
    updated["released_shard_protect_during_dispatch"] = [
        {"attempt_id": attempt_id} for attempt_id in RELEASED_SHARD_ATTEMPTS
    ]
    updated["sequencing_phase"] = "2_teardown_complete_protect_released_only"
    updated["compiled_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return updated


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--protect-list", type=Path, default=DEFAULT_PROTECT_LIST)
    parser.add_argument("--plan-out", type=Path, default=DEFAULT_PLAN_OUT)
    parser.add_argument("--write-phase2-protect-list", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    protect_list = gpu_scheduling.load_protect_list(args.protect_list)
    plan = build_sequencing_plan(
        kube_context=args.kube_context,
        namespace=args.namespace,
        protect_list=protect_list,
    )
    text = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    args.plan_out.parent.mkdir(parents=True, exist_ok=True)
    args.plan_out.write_text(text, encoding="utf-8")
    if args.write_phase2_protect_list is not None:
        args.write_phase2_protect_list.write_text(
            json.dumps(build_phase2_protect_list_update(protect_list), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
