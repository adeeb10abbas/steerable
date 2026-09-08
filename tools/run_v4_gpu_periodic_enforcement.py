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
    ("c6_a10040_spread", "c6m"),
)

C6_SIM_RESTORE_RENDERED_ROOT = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/rendered-c6confirm20260908f-sim-restore"
)
C6_DEFAULT_RENDERED_ROOT = gpu_placement.DEFAULT_C6_RENDERED_ROOT


def _resolve_c6_rendered_root(lane_id: str) -> Path | None:
    for root in (C6_SIM_RESTORE_RENDERED_ROOT, C6_DEFAULT_RENDERED_ROOT):
        if not root.is_dir():
            continue
        if list(root.glob(f"{lane_id}-*")):
            return root
    return None


def recreate_failed_c6_sim_partners(
    *,
    pods: Sequence[gpu_sweep.PodRef],
    kube_context: str,
    namespace: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Recreate Failed/Error C6 sims so sim-first gate can unsuspend policies.

    The gate only fires when sim phase is Running; lanes stuck on Failed /healthz never
    self-heal (same failure class as C7 r6 and C6 wave-D tail). Agent B may also recreate
    lanes ad hoc; this keeps the periodic loop from leaving stragglers.
    """
    try:
        import yaml
    except ImportError:
        return {"actions": [], "lanes_recreated": [], "error": "pyyaml_missing"}

    by_lane: dict[str, list[gpu_sweep.PodRef]] = {}
    for pod in pods:
        if pod.lane_id.startswith("c6m"):
            by_lane.setdefault(pod.lane_id, []).append(pod)

    actions: list[dict[str, Any]] = []
    recreated: list[str] = []
    for lane_id in sorted(by_lane):
        latest = gpu_sweep._latest_pods(by_lane[lane_id])
        sim = latest.get("sim")
        policy = latest.get("policy")
        if sim is None:
            continue
        if sim.phase not in {"Failed", "Error"}:
            continue
        if gpu_sweep._in_startup_grace(sim):
            continue
        rendered_root = _resolve_c6_rendered_root(lane_id)
        if rendered_root is None:
            actions.append(
                {
                    "action": "recreate_sim_skipped",
                    "lane_id": lane_id,
                    "reason": "no_rendered_bundle",
                }
            )
            continue
        matches = sorted(rendered_root.glob(f"{lane_id}-*"))
        sim_path = matches[0] / "simulator-job.yaml"
        if not sim_path.is_file():
            actions.append(
                {
                    "action": "recreate_sim_skipped",
                    "lane_id": lane_id,
                    "reason": "missing_simulator_job_yaml",
                }
            )
            continue
        sim_job = sim.job_name
        if sim_job and not dry_run:
            gpu_placement.delete_job(
                name=sim_job,
                kube_context=kube_context,
                namespace=namespace,
                dry_run=False,
            )
        actions.append(
            {
                "action": "delete_failed_sim",
                "lane_id": lane_id,
                "job": sim_job,
                "sim_phase": sim.phase,
                "deleted": bool(sim_job) and not dry_run,
            }
        )
        doc = yaml.safe_load(sim_path.read_text(encoding="utf-8"))
        create_ok = False
        message = ""
        if not dry_run:
            import subprocess

            completed = subprocess.run(
                ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"],
                input=yaml.safe_dump(doc, sort_keys=False),
                capture_output=True,
                text=True,
            )
            create_ok = completed.returncode == 0 or "AlreadyExists" in (completed.stderr or "")
            message = (completed.stderr or completed.stdout or "").strip()
        else:
            create_ok = True
            message = "dry_run"
        actions.append(
            {
                "action": "create_sim",
                "lane_id": lane_id,
                "job": doc["metadata"]["name"],
                "ok": create_ok,
                "message": message[:200] if isinstance(message, str) else "",
            }
        )
        if policy and policy.job_name and not dry_run:
            unsuspended = gpu_placement.patch_job_suspend(
                name=policy.job_name,
                suspend=False,
                kube_context=kube_context,
                namespace=namespace,
                dry_run=False,
            )
            actions.append(
                {
                    "action": "unsuspend_policy_after_sim_recreate",
                    "lane_id": lane_id,
                    "job": policy.job_name,
                    "unsuspended": unsuspended,
                }
            )
        if create_ok:
            recreated.append(lane_id)

    return {
        "lanes_recreated": recreated,
        "actions": actions,
        "rendered_roots_checked": [
            str(C6_SIM_RESTORE_RENDERED_ROOT.relative_to(ROOT))
            if C6_SIM_RESTORE_RENDERED_ROOT.is_dir()
            else str(C6_DEFAULT_RENDERED_ROOT.relative_to(ROOT)),
        ],
    }


def enforce_sim_first_gate(
    *,
    lane_prefix: str,
    policy_id: str,
    kube_context: str,
    namespace: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Unsuspend policy jobs once their simulator pod is Running (sim-first dispatch)."""
    pods = [
        p
        for p in (
            gpu_sweep.parse_pod(item)
            for item in gpu_sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
        )
        if p is not None and p.lane_id.startswith(lane_prefix)
    ]
    by_lane: dict[str, list[gpu_sweep.PodRef]] = {}
    for pod in pods:
        by_lane.setdefault(pod.lane_id, []).append(pod)
    actions: list[dict[str, Any]] = []
    raw_jobs = gpu_placement.fetch_jobs(kube_context=kube_context, namespace=namespace)
    job_suspend: dict[str, bool] = {}
    for item in raw_jobs:
        name = str(item.get("metadata", {}).get("name") or "")
        if "-policy" not in name or not name.startswith(f"v4-{lane_prefix}"):
            continue
        job_suspend[name] = bool((item.get("spec") or {}).get("suspend"))
    for lane_id in sorted(by_lane):
        latest = gpu_sweep._latest_pods(by_lane[lane_id])
        sim = latest.get("sim")
        policy = latest.get("policy")
        if not sim or sim.phase != "Running" or sim.gpu_count <= 0:
            continue
        if not policy or policy.phase == "Running":
            continue
        job_name = policy.job_name
        if not job_name or not job_suspend.get(job_name):
            continue
        ok = gpu_placement.patch_job_suspend(
            name=job_name,
            suspend=False,
            kube_context=kube_context,
            namespace=namespace,
            dry_run=dry_run,
        )
        actions.append(
            {
                "action": "unsuspend_policy_sim_ready",
                "job": job_name,
                "lane_id": lane_id,
                "unsuspended": ok,
                "policy_id": policy_id,
            }
        )
    return actions


def enforce_c7_sim_first_gate(
    *,
    kube_context: str,
    namespace: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    return enforce_sim_first_gate(
        lane_prefix="c7m",
        policy_id="c7_sim_first_adhoc",
        kube_context=kube_context,
        namespace=namespace,
        dry_run=dry_run,
    )


def enforce_c6_sim_first_gate(
    *,
    kube_context: str,
    namespace: str,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    return enforce_sim_first_gate(
        lane_prefix="c6m",
        policy_id="c6_sim_first_adhoc",
        kube_context=kube_context,
        namespace=namespace,
        dry_run=dry_run,
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
    c6_remaining_episodes: int = 438,
    c6_authorized_lanes: int | None = None,
) -> dict[str, Any]:
    if c6_authorized_lanes is None:
        c6_authorized_lanes = gpu_scheduling.C6_AUTHORIZED_LANE_COUNT

    raw_pods_pre = gpu_sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    pods_pre = [p for p in (gpu_sweep.parse_pod(item) for item in raw_pods_pre) if p is not None]

    finished_family_reclaim = gpu_sweep.reclaim_finished_family_cluster_state(
        kube_context=kube_context,
        namespace=namespace,
        dry_run=False,
        pods=pods_pre,
    )

    raw_pods = gpu_sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    pods = [p for p in (gpu_sweep.parse_pod(item) for item in raw_pods) if p is not None]

    failed_sim_recreate = recreate_failed_c6_sim_partners(
        pods=pods,
        kube_context=kube_context,
        namespace=namespace,
        dry_run=False,
    )
    if failed_sim_recreate.get("lanes_recreated"):
        raw_pods = gpu_sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
        pods = [p for p in (gpu_sweep.parse_pod(item) for item in raw_pods) if p is not None]

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

    c6_gate_actions = enforce_sim_first_gate(
        lane_prefix="c6m",
        policy_id="c6_sim_first_adhoc",
        kube_context=kube_context,
        namespace=namespace,
        dry_run=False,
    )
    c6_unsuspended = sum(1 for row in c6_gate_actions if row.get("unsuspended"))
    unsuspend_total += c6_unsuspended
    placement_receipts.append(
        {
            "policy_id": "c6_sim_first_adhoc",
            "unsuspended": c6_unsuspended,
            "suspended": 0,
            "actions": c6_gate_actions,
        }
    )

    sweep_receipt = gpu_sweep.run_gpu_pool_sweep(
        kube_context=kube_context,
        namespace=namespace,
        dry_run=True,
        c7_remaining_episodes=0,
        protect_list_path=protect_list_path,
        allow_redispatch=False,
    )
    detection = sweep_receipt.get("detection") or {}
    finished_orphan_reclaim = gpu_sweep.reclaim_finished_lane_orphan_policies(
        detection,
        kube_context=kube_context,
        namespace=namespace,
        dry_run=False,
        lane_ids=gpu_scheduling.FINISHED_C8_LANE_IDS,
    )
    raw_pods = gpu_sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    pods = [p for p in (gpu_sweep.parse_pod(item) for item in raw_pods) if p is not None]
    usage = gpu_sweep.summarize_gpu_usage(pods)
    c6_state = gpu_sweep.lane_pair_scheduling_state(pods, lane_prefix="c6m")
    c6_gate = count_sim_first_gate_pairs(pods, lane_prefix="c6m")

    c6_healthy = int(c6_state.get("healthy_pairs") or 0)
    c6_lanes_seen = int(c6_state.get("total_lanes_seen") or 0)
    c6_absorbable_lanes = min(
        c6_authorized_lanes,
        int((usage.get("pool_free_estimate") or {}).get("NVIDIA-B200") or 0)
        + int((usage.get("running_gpus_by_family_role") or {}).get("c6_policy") or 0),
        int((usage.get("pool_free_estimate") or {}).get("NVIDIA-A100-SXM4-40GB") or 0)
        + int((usage.get("running_gpus_by_family_role") or {}).get("c6_sim") or 0),
    )
    c6_idle_b200 = max(
        0,
        int((usage.get("pool_sizes") or {}).get("NVIDIA-B200") or 0)
        - int((usage.get("running_gpus_by_product") or {}).get("NVIDIA-B200") or 0)
        - max(0, c6_absorbable_lanes - c6_healthy),
    )
    c6_idle_a10040 = max(
        0,
        int((usage.get("pool_sizes") or {}).get("NVIDIA-A100-SXM4-40GB") or 0)
        - int((usage.get("running_gpus_by_product") or {}).get("NVIDIA-A100-SXM4-40GB") or 0)
        - max(0, c6_absorbable_lanes - c6_healthy),
    )
    a40_free = int((usage.get("pool_free_estimate") or {}).get("NVIDIA-A40") or 0)
    a10080_free = int((usage.get("pool_free_estimate") or {}).get("NVIDIA-A100-SXM4-80GB") or 0)

    minutes = arbitration.C6_EPISODE_MINUTES_PER_LANE

    def hours(episodes: int, lanes: int, *, lane_minutes: float = minutes) -> float:
        if lanes <= 0:
            return float("inf")
        return round((episodes / lanes) * (lane_minutes / 60.0), 1)

    routing = {
        "decision": "c6_sole_consumer_post_c7_c8_terminal",
        "c7_status": "768/768 terminal — cluster jobs reclaimed",
        "c8_status": "768/768 terminal — cluster jobs reclaimed",
        "c6_remaining_episodes": c6_remaining_episodes,
        "c6_authorized_lanes": c6_authorized_lanes,
        "c6_absorbable_lanes_pool_cap": c6_absorbable_lanes,
        "c6_healthy_pairs_observed": c6_healthy,
        "c6_lanes_seen": c6_lanes_seen,
        "c6_sim_ready_policy_waiting": c6_gate["sim_ready_policy_waiting"],
        "c6_matrix_binding": (
            "authorized_matrix_caps_at_32"
            if c6_absorbable_lanes >= c6_authorized_lanes
            else "pool_b200_or_a10040_binds_below_matrix"
        ),
        "idle_not_absorbable_by_c6": {
            "NVIDIA-A40": a40_free,
            "NVIDIA-A100-SXM4-80GB": a10080_free,
            "note": "C6 stratum is A100-40GB sim + B200 policy; A40/A100-80GB from finished families stay idle unless amended.",
        },
        "amendment_for_extra_c6_lanes": (
            None
            if c6_absorbable_lanes >= c6_authorized_lanes
            else "Post-result capacity amendment to containment_c6_confirmatory_launch_matrix.json only; "
            "qualification gates unchanged."
        ),
    }

    finished_lane_orphans = [
        row
        for row in (detection.get("orphan_policies") or [])
        if str(row.get("lane_id") or "") in gpu_scheduling.FINISHED_C8_LANE_IDS
    ]

    reclaimed_merged: dict[str, int] = {}
    for source in (finished_family_reclaim, finished_orphan_reclaim):
        for product, count in (source.get("reclaimed_gpus_by_product") or {}).items():
            reclaimed_merged[product] = reclaimed_merged.get(product, 0) + int(count)

    return {
        "schema_version": "v4-gpu-periodic-enforcement-receipt-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kube_context": kube_context,
        "namespace": namespace,
        "automation": {
            "mode": "c6_only_live_gates_plus_finished_family_reclaim",
            "interval_seconds_default": 180,
            "policies_enforced": [policy_id for policy_id, _ in PERIODIC_POLICIES],
            "sweep_live": False,
            "allow_redispatch": False,
            "failed_sim_recreate_live": True,
            "finished_family_reclaim_live": True,
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
            "failed_sim_recreate": failed_sim_recreate,
            "periodic_loop_gap_decision": {
                "issue": "sim_first gate unsuspends only when sim Running; Failed /healthz sims deadlock",
                "action": "recreate_failed_c6_sim_partners each tick before unsuspend (live)",
                "agent_b_coordination": "Agent B tail recovery 20260908 used same delete+create pattern",
            },
        },
        "sweep_dry_run": {
            "jobs_would_delete": (sweep_receipt.get("reclaim") or {}).get("jobs_deleted"),
            "reclaimed_gpus_would_be": (sweep_receipt.get("reclaim") or {}).get("reclaimed_gpus_by_product"),
            "finished_lane_orphan_policies": finished_lane_orphans,
        },
        "finished_family_reclaim_live": finished_family_reclaim,
        "finished_lane_orphan_reclaim_live": finished_orphan_reclaim,
        "scheduling_state": {"c6": c6_state},
        "pool_utilization": usage,
        "c6_capacity_routing": routing,
        "gpus_reclaimed_by_pool_this_tick": reclaimed_merged,
        "wall_clock_estimates": {
            "c6_remaining_hours_at_healthy_pairs": hours(
                c6_remaining_episodes, max(c6_healthy, 1), lane_minutes=minutes
            ),
            "c6_remaining_hours_at_absorbable_lanes": hours(
                c6_remaining_episodes, max(c6_absorbable_lanes, 1), lane_minutes=minutes
            ),
            "c6_remaining_hours_at_authorized_matrix": hours(
                c6_remaining_episodes, c6_authorized_lanes, lane_minutes=minutes
            ),
            "c6_achievable_estimate_note": (
                f"Use absorbable_lanes={c6_absorbable_lanes} (min of matrix and B200/A100-40 pool); "
                f"Agent B tail recovery should lift healthy count toward that cap."
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
    parser.add_argument("--c6-remaining", type=int, default=438)
    parser.add_argument("--c6-authorized-lanes", type=int, default=gpu_scheduling.C6_AUTHORIZED_LANE_COUNT)
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
            c6_remaining_episodes=args.c6_remaining,
            c6_authorized_lanes=args.c6_authorized_lanes,
        )
        if args.loop_seconds > 0 and args.record_loop_pid:
            receipt["loop_active"] = True
            receipt["loop_pid"] = os.getpid()
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        routing_out = args.receipt_out.parent / "c6_capacity_routing_decision.json"
        routing_out.write_text(
            json.dumps(
                {
                    "schema_version": "v4-c6-capacity-routing-decision-v1",
                    "observed_at_utc": receipt["observed_at_utc"],
                    **receipt["c6_capacity_routing"],
                    "gpus_reclaimed_by_pool_this_tick": receipt.get("gpus_reclaimed_by_pool_this_tick"),
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
