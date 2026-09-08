#!/usr/bin/env python3
"""Periodic sim-first gate enforcement: live unsuspend, dry-run sweep only."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_capacity_arbitration as arbitration  # noqa: E402
import v4_gpu_placement_enforce as gpu_placement  # noqa: E402
import v4_gpu_pool_sweep as gpu_sweep  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

DEFAULT_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_periodic_enforcement_receipt.json"
)
DEFAULT_STATE = arbitration.DEFAULT_STATE
DEFAULT_PROTECT_LIST = gpu_scheduling.DEFAULT_PROTECT_LIST
PERIODIC_POLICIES = (
    ("c8_a40_spread", "c8m"),
    ("c6_a10040_spread", "c6m"),
)


def _receipt_relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(path)


def count_sim_first_gate_pairs(
    pods: Sequence[gpu_sweep.PodRef], *, lane_prefix: str
) -> dict[str, int]:
    """Sim Running with policy not yet Running (Suspended/Pending schedule window)."""
    by_lane: dict[str, list[gpu_sweep.PodRef]] = {}
    for pod in pods:
        if pod.lane_id.startswith(lane_prefix):
            by_lane.setdefault(pod.lane_id, []).append(pod)
    sim_ready_policy_waiting = healthy = 0
    for lane_id in by_lane:
        latest = gpu_sweep._latest_pods(by_lane[lane_id])
        policy = latest.get("policy")
        sim = latest.get("sim")
        if (
            sim
            and sim.phase == "Running"
            and sim.gpu_count > 0
            and policy
            and policy.phase != "Running"
        ):
            sim_ready_policy_waiting += 1
        if (
            policy
            and sim
            and policy.phase == "Running"
            and policy.gpu_count > 0
            and sim.phase == "Running"
            and sim.gpu_count > 0
        ):
            healthy += 1
    return {
        "sim_ready_policy_waiting": sim_ready_policy_waiting,
        "healthy_pairs": healthy,
    }


def run_periodic_enforcement(
    *,
    kube_context: str,
    namespace: str,
    protect_list_path: Path,
    c7_remaining_episodes: int = 188,
    c7_tail_lanes: int = 4,
    c8_remaining_episodes: int = 564,
) -> dict[str, Any]:
    placement_receipts: list[dict[str, Any]] = []
    unsuspend_total = 0
    suspend_total = 0
    for policy_id, _lane_prefix in PERIODIC_POLICIES:
        receipt = gpu_placement.enforce_gpu_placement(
            policy_id=policy_id,
            kube_context=kube_context,
            namespace=namespace,
            dry_run=False,
            protect_list_path=protect_list_path,
            rendered_root=None,
            settle_seconds=0,
            gates_only=True,
        )
        policy_unsuspended = sum(
            1 for row in receipt.get("actions") or [] if row.get("action") == "unsuspend_policy_sim_ready"
        )
        policy_suspended = sum(
            1 for row in receipt.get("actions") or [] if row.get("action") == "suspend_until_sim_running"
        )
        unsuspend_total += policy_unsuspended
        suspend_total += policy_suspended
        placement_receipts.append(
            {
                "policy_id": policy_id,
                "unsuspended": policy_unsuspended,
                "suspended": policy_suspended,
                "actions": receipt.get("actions") or [],
                "scheduling_state_after": receipt.get("scheduling_state_after") or {},
            }
        )

    sweep_receipt = gpu_sweep.run_gpu_pool_sweep(
        kube_context=kube_context,
        namespace=namespace,
        dry_run=True,
        c7_remaining_episodes=c7_remaining_episodes,
        protect_list_path=protect_list_path,
        allow_redispatch=False,
    )
    detection = sweep_receipt.get("detection") or {}
    finished_orphan_reclaim = gpu_sweep.reclaim_finished_lane_orphan_policies(
        detection,
        kube_context=kube_context,
        namespace=namespace,
        dry_run=False,
    )
    raw_pods = gpu_sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    pods = [p for p in (gpu_sweep.parse_pod(item) for item in raw_pods) if p is not None]
    usage = gpu_sweep.summarize_gpu_usage(pods)
    c7_state = gpu_sweep.lane_pair_scheduling_state(pods, lane_prefix="c7m")
    c8_state = gpu_sweep.lane_pair_scheduling_state(pods, lane_prefix="c8m")
    c6_state = gpu_sweep.lane_pair_scheduling_state(pods, lane_prefix="c6m")

    a40_used = int((usage.get("running_gpus_by_product") or {}).get("NVIDIA-A40") or 0)
    a40_free = int((usage.get("pool_free_estimate") or {}).get("NVIDIA-A40") or 0)
    c7_a40_sims = int((usage.get("running_gpus_by_family_role") or {}).get("c7_sim") or 0)
    c8_a40 = int((usage.get("running_gpus_by_family_role") or {}).get("c8_policy") or 0) + int(
        (usage.get("running_gpus_by_family_role") or {}).get("c8_sim") or 0
    )
    c8_max_pairs = min(
        arbitration.G7_C8_CONFIRMATORY_MAX_LANES,
        max(0, (arbitration.A40_POOL_SIZE - c7_a40_sims) // arbitration.A40_GPUS_PER_C8_LANE),
    )
    c8_healthy = int(c8_state.get("healthy_pairs") or 0)
    c7_healthy = int(c7_state.get("healthy_pairs") or 0)
    c7_r6_lanes = gpu_sweep.count_c7_r6_reshard_lanes(pods)
    c8_gate = count_sim_first_gate_pairs(pods, lane_prefix="c8m")
    c6_gate = count_sim_first_gate_pairs(pods, lane_prefix="c6m")

    minutes = arbitration.C8_EPISODE_MINUTES_PER_LANE
    c6_minutes = arbitration.C6_EPISODE_MINUTES_PER_LANE

    def hours(episodes: int, lanes: int, *, lane_minutes: float = minutes) -> float:
        if lanes <= 0:
            return float("inf")
        return round((episodes / lanes) * (lane_minutes / 60.0), 1)

    c7_active_lanes = max(c7_r6_lanes, c7_tail_lanes) if c7_r6_lanes else c7_tail_lanes
    if c7_r6_lanes >= 10:
        routing_decision = "split_a40_r6_reshard_plus_c8_ramp"
        reshard_note = (
            f"Agent B r6 resharding live on {c7_r6_lanes} lane identities "
            f"(attempt0611-0624). Remaining A40 routed to C8 up to {c8_max_pairs} pairs."
        )
    else:
        routing_decision = "route_bulk_a40_to_c8_while_c7_tail_runs_4_lanes"
        reshard_note = (
            "Interim: C7 tail on c7m01/02/05/31 (attempt0607-0610) until r6 dispatch completes. "
            f"Route remaining A40 to C8 up to {c8_max_pairs} pairs."
        )

    routing = {
        "decision": routing_decision,
        "reshard_feasible": "yes_agent_b_rendered_r6_dispatched" if c7_r6_lanes else "yes_pending_dispatch",
        "reshard_note": reshard_note,
        "c7_r6_reshard_lanes_observed": c7_r6_lanes,
        "c7_tail_lanes_legacy": c7_tail_lanes,
        "c7_active_lanes_for_estimate": c7_active_lanes,
        "c7_a40_sims_reserved": c7_a40_sims,
        "c8_authorized_lane_pairs": c8_max_pairs,
        "c8_healthy_pairs_observed": c8_healthy,
        "c8_sim_ready_policy_waiting": c8_gate["sim_ready_policy_waiting"],
        "c6_sim_ready_policy_waiting": c6_gate["sim_ready_policy_waiting"],
        "a40_pool": {
            "size": arbitration.A40_POOL_SIZE,
            "used": a40_used,
            "free_estimate": a40_free,
            "c8_gpus_observed": c8_a40,
        },
    }

    finished_lane_orphans = [
        row
        for row in (detection.get("orphan_policies") or [])
        if str(row.get("lane_id") or "") in gpu_scheduling.FINISHED_C8_LANE_ORPHAN_RECLAIM
    ]

    return {
        "schema_version": "v4-gpu-periodic-enforcement-receipt-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kube_context": kube_context,
        "namespace": namespace,
        "automation": {
            "mode": "live_gates_only_plus_dry_run_sweep",
            "interval_seconds_default": 180,
            "policies_enforced": [policy_id for policy_id, _ in PERIODIC_POLICIES],
            "sweep_live": False,
            "allow_redispatch": False,
            "rerun_command": ".venv/bin/python tools/run_v4_gpu_periodic_enforcement.py",
            "loop_command": (
                ".venv/bin/python tools/run_v4_gpu_periodic_enforcement.py "
                "--loop-seconds 180 --record-loop-pid"
            ),
            "finished_lane_orphan_reclaim_live": True,
        },
        "sim_first_gates": {
            "policies_unsuspended_this_tick": unsuspend_total,
            "policies_suspended_this_tick": suspend_total,
            "placement_receipts": placement_receipts,
        },
        "sweep_dry_run": {
            "jobs_would_delete": (sweep_receipt.get("reclaim") or {}).get("jobs_deleted"),
            "reclaimed_gpus_would_be": (sweep_receipt.get("reclaim") or {}).get("reclaimed_gpus_by_product"),
            "finished_lane_orphan_policies": finished_lane_orphans,
        },
        "finished_lane_orphan_reclaim_live": finished_orphan_reclaim,
        "scheduling_state": {"c7": c7_state, "c8": c8_state, "c6": c6_state},
        "pool_utilization": usage,
        "a40_routing": routing,
        "wall_clock_estimates": {
            "c7_tail_188_hours_at_4_lanes": hours(c7_remaining_episodes, 4),
            "c7_tail_188_hours_at_active_lanes": hours(c7_remaining_episodes, c7_active_lanes),
            "c7_tail_188_hours_if_r6_14_lanes": hours(c7_remaining_episodes, 14),
            "c8_remaining_hours_at_current_healthy": hours(c8_remaining_episodes, max(c8_healthy, 1)),
            "c8_remaining_hours_at_max_authorized_pairs": hours(c8_remaining_episodes, max(c8_max_pairs, 1)),
            "c8_remaining_hours_if_c7_r6_14_lanes": hours(
                c8_remaining_episodes,
                max((arbitration.A40_POOL_SIZE - 14) // arbitration.A40_GPUS_PER_C8_LANE, 1),
            ),
            "c6_confirmatory_hours_at_healthy_pairs": hours(
                arbitration.CONFIRMATORY_EPISODE_TARGET,
                max(int(c6_state.get("healthy_pairs") or 1), 1),
                lane_minutes=c6_minutes,
            ),
        },
        "protect_list_path": _receipt_relative_path(protect_list_path),
    }


def update_capacity_state(*, receipt: Mapping[str, Any]) -> None:
    state_path = DEFAULT_STATE
    state: dict[str, Any] = {}
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state.setdefault("gpu_placement", {})
    state["gpu_placement"]["periodic_enforcement"] = {
        **(receipt.get("automation") or {}),
        "loop_active": receipt.get("loop_active", False),
        "loop_pid": receipt.get("loop_pid"),
    }
    state["gpu_placement"]["last_periodic_at_utc"] = receipt.get("observed_at_utc")
    state["gpu_placement"]["last_unsuspended_count"] = (
        (receipt.get("sim_first_gates") or {}).get("policies_unsuspended_this_tick")
    )
    state.setdefault("gpu_pool_sweep", {})
    state["gpu_pool_sweep"]["live_sweep_enabled"] = False
    state["gpu_pool_sweep"]["dry_run_only"] = True
    state["gpu_pool_sweep"]["last_dry_run_at_utc"] = receipt.get("observed_at_utc")
    state["last_enforcement_at_utc"] = receipt.get("observed_at_utc")
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--protect-list", type=Path, default=DEFAULT_PROTECT_LIST)
    parser.add_argument("--c7-remaining", type=int, default=188)
    parser.add_argument("--c7-tail-lanes", type=int, default=4)
    parser.add_argument("--c8-remaining", type=int, default=564)
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--loop-seconds", type=int, default=0, help="If >0, rerun forever at this interval.")
    parser.add_argument("--max-ticks", type=int, default=0, help="Stop loop after N ticks (0 = unlimited).")
    parser.add_argument("--record-loop-pid", action="store_true", help="Mark loop as active in capacity state.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    import os

    tick = 0
    while True:
        receipt = run_periodic_enforcement(
            kube_context=args.kube_context,
            namespace=args.namespace,
            protect_list_path=args.protect_list,
            c7_remaining_episodes=args.c7_remaining,
            c7_tail_lanes=args.c7_tail_lanes,
            c8_remaining_episodes=args.c8_remaining,
        )
        if args.loop_seconds > 0 and args.record_loop_pid:
            receipt["loop_active"] = True
            receipt["loop_pid"] = os.getpid()
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        routing_out = args.receipt_out.parent / "a40_capacity_routing_decision.json"
        routing_out.write_text(
            json.dumps(
                {
                    "schema_version": "v4-a40-capacity-routing-decision-v1",
                    "observed_at_utc": receipt["observed_at_utc"],
                    **receipt["a40_routing"],
                    "wall_clock_estimates": receipt["wall_clock_estimates"],
                    "scheduling_state": receipt["scheduling_state"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        update_capacity_state(receipt=receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        tick += 1
        if args.loop_seconds <= 0:
            break
        if args.max_ticks and tick >= args.max_ticks:
            break
        time.sleep(args.loop_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
