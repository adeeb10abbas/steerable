#!/usr/bin/env python3
"""Detect and reclaim orphaned/zombie GPU lane allocations across V4 pools."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_capacity_arbitration as arbitration  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

DEFAULT_RECEIPT = (
    ROOT / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_pool_sweep_receipt.json"
)

LANE_POD_RE = re.compile(
    r"^v4-(?P<lane>c7m\d+|c8m\d+|g7c6p\d+|g7c8p\d+|c6m\d+)-"
    r"(?P<attempt>attempt\d+)-(?P<hash>[a-f0-9]+)-(?P<role>policy|sim)-",
    re.I,
)

GPU_PRODUCTS = (
    "NVIDIA-A40",
    "NVIDIA-A100-SXM4-80GB",
    "NVIDIA-A100-SXM4-40GB",
    "NVIDIA-B200",
)


@dataclass(frozen=True)
class PodRef:
    name: str
    lane_id: str
    attempt_id: str
    role: str
    phase: str
    gpu_count: int
    gpu_product: str
    node_name: str
    job_name: str
    age_seconds: float = 0.0
    container_ready: bool = False


class GpuPoolSweepError(RuntimeError):
    """GPU pool sweep precondition failed."""


def fetch_pods(*, kube_context: str, namespace: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["kubectl", "--context", kube_context, "-n", namespace, "get", "pods", "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise GpuPoolSweepError(completed.stderr or completed.stdout or "kubectl get pods failed")
    return json.loads(completed.stdout).get("items", [])


def fetch_jobs(*, kube_context: str, namespace: str) -> list[dict[str, Any]]:
    return arbitration.fetch_namespace_jobs(kube_context=kube_context, namespace=namespace)


def parse_pod(pod: Mapping[str, Any]) -> PodRef | None:
    name = str(pod.get("metadata", {}).get("name") or "")
    match = LANE_POD_RE.match(name)
    if not match:
        return None
    gpu_count = 0
    for container in pod.get("spec", {}).get("containers") or []:
        limits = (container.get("resources") or {}).get("limits") or {}
        gpu_count += int(limits.get("nvidia.com/gpu") or 0)
    node_selector = pod.get("spec", {}).get("nodeSelector") or {}
    owner_refs = pod.get("metadata", {}).get("ownerReferences") or []
    job_name = ""
    for owner in owner_refs:
        if owner.get("kind") == "Job":
            job_name = str(owner.get("name") or "")
            break
    created_raw = str(pod.get("metadata", {}).get("creationTimestamp") or "")
    age_seconds = 0.0
    if created_raw:
        created = datetime.fromisoformat(created_raw.replace("Z", "+00:00"))
        age_seconds = max(0.0, (datetime.now(timezone.utc) - created).total_seconds())
    container_ready = False
    for status in pod.get("status", {}).get("containerStatuses") or []:
        if status.get("ready"):
            container_ready = True
            break
    return PodRef(
        name=name,
        lane_id=str(match.group("lane")),
        attempt_id=str(match.group("attempt")),
        role=str(match.group("role")),
        phase=str((pod.get("status") or {}).get("phase") or ""),
        gpu_count=gpu_count,
        gpu_product=str(node_selector.get("nvidia.com/gpu.product") or ""),
        node_name=str(pod.get("spec", {}).get("nodeName") or ""),
        job_name=job_name,
        age_seconds=age_seconds,
        container_ready=container_ready,
    )


def _in_startup_grace(pod: PodRef | None, *, grace_seconds: int | None = None) -> bool:
    if pod is None:
        return False
    grace = grace_seconds if grace_seconds is not None else gpu_scheduling.ISAAC_STARTUP_GRACE_SECONDS
    if pod.age_seconds >= grace:
        return False
    if pod.phase == "Pending":
        return True
    if pod.phase == "Running" and not pod.container_ready:
        return True
    return False


def _latest_pods(pods: Sequence[PodRef]) -> dict[str, PodRef]:
    latest: dict[str, PodRef] = {}
    for pod in pods:
        current = latest.get(pod.role)
        if current is None or pod.attempt_id > current.attempt_id:
            latest[pod.role] = pod
    return latest


def _is_running_with_gpu(pod: PodRef | None) -> bool:
    return pod is not None and pod.phase == "Running" and pod.gpu_count > 0


def _sim_is_dead(sim: PodRef | None) -> bool:
    if sim is None:
        return True
    if sim.phase in {"Failed", "Error", "Unknown"}:
        return True
    if sim.phase == "Succeeded":
        return True
    return sim.phase != "Running"


def _between_episodes_healthy(policy: PodRef, sim: PodRef | None) -> bool:
    """Running policy may idle between episodes while sim restarts on same attempt."""
    if sim is None or policy is None:
        return False
    if policy.attempt_id != sim.attempt_id:
        return False
    if policy.phase != "Running":
        return False
    if sim.phase in {"Pending", "Running"}:
        return True
    return False


def _partner_is_failed_not_pending(partner: PodRef | None) -> bool:
    if partner is None:
        return False
    if _in_startup_grace(partner):
        return False
    return partner.phase in {"Failed", "Error"}


def _shard_complete_or_between_episodes(policy: PodRef, sim: PodRef | None) -> bool:
    """Running policy with sim Succeeded/Pending on same attempt — not an orphan."""
    if sim is None or policy is None:
        return False
    if policy.attempt_id != sim.attempt_id:
        return False
    if policy.phase != "Running":
        return False
    if sim.phase in {"Pending", "Running", "Succeeded"}:
        return True
    if _in_startup_grace(sim):
        return True
    return False


def detect_lane_mismatches(
    pods: Sequence[PodRef],
    *,
    job_failed_by_name: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    by_lane: dict[str, list[PodRef]] = defaultdict(list)
    for pod in pods:
        by_lane[pod.lane_id].append(pod)

    orphan_policies: list[dict[str, Any]] = []
    zombie_sims: list[dict[str, Any]] = []
    zombie_policies: list[dict[str, Any]] = []
    split_pair_orphans: list[dict[str, Any]] = []
    healthy_pairs: list[str] = []
    clutter_no_gpu: list[dict[str, Any]] = []
    job_failed = job_failed_by_name or {}

    def _job_is_failed(job_name: str, pod: PodRef | None) -> bool:
        if job_failed.get(job_name):
            return True
        if pod is None:
            return False
        return pod.phase in {"Failed", "Error"}

    for lane_id in sorted(by_lane):
        latest = _latest_pods(by_lane[lane_id])
        policy = latest.get("policy")
        sim = latest.get("sim")
        if policy and policy.phase in {"Failed", "Error"} and policy.gpu_count == 0:
            clutter_no_gpu.append(
                {
                    "lane_id": lane_id,
                    "attempt_id": policy.attempt_id,
                    "pod": policy.name,
                    "job": policy.job_name,
                    "role": "policy",
                    "reason_code": "terminated_error_clutter",
                }
            )
        if sim and sim.phase in {"Failed", "Error"} and sim.gpu_count == 0:
            clutter_no_gpu.append(
                {
                    "lane_id": lane_id,
                    "attempt_id": sim.attempt_id,
                    "pod": sim.name,
                    "job": sim.job_name,
                    "role": "sim",
                    "reason_code": "terminated_error_clutter",
                }
            )

        pol_run = _is_running_with_gpu(policy)
        sim_run = _is_running_with_gpu(sim)
        if pol_run and sim_run and policy and sim and policy.attempt_id == sim.attempt_id:
            healthy_pairs.append(lane_id)
            continue
        if pol_run and policy and _shard_complete_or_between_episodes(policy, sim):
            healthy_pairs.append(lane_id)
            continue
        if pol_run and policy and _in_startup_grace(sim):
            continue
        if pol_run and policy and _partner_is_failed_not_pending(sim) and not _between_episodes_healthy(policy, sim):
            reason = "split_pair_orphan_policy_half"
            entry = {
                "lane_id": lane_id,
                "attempt_id": policy.attempt_id,
                "pod": policy.name,
                "job": policy.job_name,
                "gpu_product": policy.gpu_product,
                "gpu_count": policy.gpu_count,
                "sim_phase": sim.phase if sim else "missing",
                "sim_job_failed": _job_is_failed(sim.job_name if sim else "", sim),
                "reason_code": reason,
            }
            orphan_policies.append(entry)
            split_pair_orphans.append(entry)
        elif pol_run and policy and _sim_is_dead(sim) and not _between_episodes_healthy(policy, sim):
            orphan_policies.append(
                {
                    "lane_id": lane_id,
                    "attempt_id": policy.attempt_id,
                    "pod": policy.name,
                    "job": policy.job_name,
                    "gpu_product": policy.gpu_product,
                    "gpu_count": policy.gpu_count,
                    "sim_phase": sim.phase if sim else "missing",
                    "sim_attempt": sim.attempt_id if sim else None,
                    "reason_code": "orphan_policy_dead_sim",
                }
            )
        if sim_run and sim and not pol_run:
            if policy and (_between_episodes_healthy(policy, sim) or _in_startup_grace(policy)):
                pass
            elif policy and _shard_complete_or_between_episodes(policy, sim):
                pass
            elif policy is None or _partner_is_failed_not_pending(policy) or policy.phase in {"Failed", "Error"}:
                reason = (
                    "split_pair_orphan_sim_half"
                    if policy and _partner_is_failed_not_pending(policy)
                    else "zombie_sim_idle_policy_dead"
                )
                entry = {
                    "lane_id": lane_id,
                    "attempt_id": sim.attempt_id,
                    "pod": sim.name,
                    "job": sim.job_name,
                    "gpu_product": sim.gpu_product,
                    "gpu_count": sim.gpu_count,
                    "policy_phase": policy.phase if policy else "missing",
                    "policy_job_failed": _job_is_failed(policy.job_name if policy else "", policy),
                    "reason_code": reason,
                }
                zombie_sims.append(entry)
                if reason.startswith("split_pair"):
                    split_pair_orphans.append(entry)
            elif policy and policy.phase == "Pending":
                zombie_sims.append(
                    {
                        "lane_id": lane_id,
                        "attempt_id": sim.attempt_id,
                        "pod": sim.name,
                        "job": sim.job_name,
                        "gpu_product": sim.gpu_product,
                        "gpu_count": sim.gpu_count,
                        "policy_phase": policy.phase,
                        "reason_code": "zombie_sim_idle_policy_dead",
                    }
                )
        if pol_run and sim_run and policy and sim and policy.attempt_id != sim.attempt_id:
            # Stale attempt on one leg — prefer deleting the non-latest attempt via job delete.
            for pod in (policy, sim):
                other = sim if pod.role == "policy" else policy
                if pod.attempt_id < other.attempt_id:
                    entry = {
                        "lane_id": lane_id,
                        "attempt_id": pod.attempt_id,
                        "pod": pod.name,
                        "job": pod.job_name,
                        "gpu_product": pod.gpu_product,
                        "gpu_count": pod.gpu_count if pod.phase == "Running" else 0,
                        "reason_code": "stale_attempt_mismatch",
                    }
                    if pod.role == "policy":
                        orphan_policies.append(entry)
                    else:
                        zombie_sims.append(entry)

    return {
        "healthy_lane_pairs": healthy_pairs,
        "orphan_policies": orphan_policies,
        "zombie_sims": zombie_sims,
        "zombie_policies": zombie_policies,
        "split_pair_orphans": split_pair_orphans,
        "clutter_no_gpu": clutter_no_gpu,
    }


def summarize_gpu_usage(pods: Sequence[PodRef]) -> dict[str, Any]:
    running_by_product = Counter()
    running_by_family = Counter()
    pending_gpu_pods = 0
    for pod in pods:
        if pod.gpu_count <= 0:
            continue
        family = "other"
        if pod.lane_id.startswith("c7m"):
            family = "c7"
        elif pod.lane_id.startswith("c8m") or pod.lane_id.startswith("g7c8p"):
            family = "c8"
        elif pod.lane_id.startswith("g7c6p") or pod.lane_id.startswith("c6m"):
            family = "c6"
        if pod.phase == "Running":
            running_by_product[pod.gpu_product] += pod.gpu_count
            running_by_family[f"{family}_{pod.role}"] += pod.gpu_count
        elif pod.phase == "Pending":
            pending_gpu_pods += 1
    pool_sizes = {
        "NVIDIA-A40": arbitration.A40_POOL_SIZE,
        "NVIDIA-A100-SXM4-80GB": 95,
        "NVIDIA-A100-SXM4-40GB": 64,
        "NVIDIA-B200": 40,
    }
    return {
        "running_gpus_by_product": dict(running_by_product),
        "running_gpus_by_family_role": dict(running_by_family),
        "pending_gpu_pods": pending_gpu_pods,
        "pool_free_estimate": {
            product: pool_sizes[product] - running_by_product.get(product, 0) for product in pool_sizes
        },
        "pool_sizes": pool_sizes,
    }


def assess_a40_fragmentation(pods: Sequence[PodRef], orphan_policies: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    a40_orphans = [row for row in orphan_policies if row.get("gpu_product") == "NVIDIA-A40"]
    orphan_gpus = sum(int(row.get("gpu_count") or 0) for row in a40_orphans)
    usage = summarize_gpu_usage(pods)
    free = int(usage["pool_free_estimate"].get("NVIDIA-A40") or 0)
    used = int(usage["running_gpus_by_product"].get("NVIDIA-A40") or 0)
    pending = int(usage["pending_gpu_pods"] or 0)
    if orphan_gpus >= pending:
        diagnosis = "orphan_held_gpus_primary"
        note = (
            f"{orphan_gpus} A40 GPUs held by orphaned C7/C8 policies with dead simulators; "
            f"pool shows {free} free but {pending} GPU pods Pending."
        )
    elif free > 0 and pending > 0:
        diagnosis = "mixed_orphans_and_per_node_fragmentation"
        note = (
            f"Some free A40 capacity ({free} GPUs) coexists with {pending} Pending pods; "
            "after orphan reclaim, remaining failures likely per-node bin-packing fragmentation."
        )
    else:
        diagnosis = "pool_saturated"
        note = "A40 pool fully subscribed by healthy running lanes."
    return {
        "diagnosis": diagnosis,
        "a40_orphan_gpu_count": orphan_gpus,
        "a40_orphan_policy_count": len(a40_orphans),
        "a40_pending_gpu_pods": pending,
        "a40_used_gpus": used,
        "a40_free_gpus_estimate": free,
        "note": note,
        "placement_recommendation": (
            "Stay on A40 stratum; no G4 expansion available. After orphan reclaim, if Pending "
            "persists, reduce concurrent C7 sim lanes temporarily or enable pod anti-affinity / "
            "spread constraints so C8 pairs bin-pack across more A40 nodes instead of colliding "
            "on saturated nodes. Do not move C8 to A100/B200 without new G4 receipts."
        ),
    }


def _delete_job(*, job_name: str, kube_context: str, namespace: str, dry_run: bool) -> bool:
    if not job_name:
        return False
    if dry_run:
        return True
    completed = subprocess.run(
        [
            "kubectl", "--context", kube_context, "-n", namespace,
            "delete", "job", job_name, "--ignore-not-found",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def filter_protected_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protect_list: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        lane_id = str(row.get("lane_id") or "")
        attempt_id = str(row.get("attempt_id") or "")
        if lane_id in gpu_scheduling.PRODUCTIVE_C8_LANE_IDS:
            skipped.append({**dict(row), "reason_code": "productive_c8_lane_preserved"})
            continue
        if gpu_scheduling.is_lane_protected(
            lane_id=lane_id,
            attempt_id=attempt_id,
            protect_list=protect_list,
        ):
            skipped.append({**dict(row), "reason_code": "protect_list_preserved"})
        else:
            allowed.append(dict(row))
    return allowed, skipped


def build_job_failed_map(jobs: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    failed: dict[str, bool] = {}
    for job in jobs:
        name = str(job.get("metadata", {}).get("name") or "")
        if not name:
            continue
        status = job.get("status") or {}
        failed[name] = int(status.get("failed") or 0) > 0
    return failed


def delete_lane_attempt_jobs(
    *,
    lane_id: str,
    attempt_id: str,
    kube_context: str,
    namespace: str,
    dry_run: bool,
) -> list[str]:
    deleted: list[str] = []
    prefix = f"v4-{lane_id}-{attempt_id}-"
    for job in fetch_jobs(kube_context=kube_context, namespace=namespace):
        name = str(job.get("metadata", {}).get("name") or "")
        if not name.startswith(prefix):
            continue
        if not (name.endswith("-policy") or name.endswith("-sim")):
            continue
        if _delete_job(job_name=name, kube_context=kube_context, namespace=namespace, dry_run=dry_run):
            deleted.append(name)
    return deleted


def reclaim_detected(
    detection: Mapping[str, Any],
    *,
    kube_context: str,
    namespace: str,
    dry_run: bool,
    protect_list: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    actions: list[dict[str, Any]] = []
    reclaimed_gpus: Counter[str] = Counter()
    protect_list = protect_list or gpu_scheduling.load_protect_list()
    split_lane_keys: set[tuple[str, str]] = set()
    for row in detection.get("split_pair_orphans") or []:
        allowed, skipped = filter_protected_rows([row], protect_list=protect_list)
        for skip in skipped:
            actions.append(
                {
                    "action": "skip_reclaim",
                    "job": skip.get("job"),
                    "lane_id": skip.get("lane_id"),
                    "reason_code": "protect_list_preserved",
                    "deleted": False,
                }
            )
        if allowed:
            lane_id = str(row["lane_id"])
            if lane_id in gpu_scheduling.PRODUCTIVE_C8_LANE_IDS:
                continue
            split_lane_keys.add((lane_id, str(row["attempt_id"])))

    redispatch_lane_ids: list[str] = []
    for lane_id, attempt_id in sorted(split_lane_keys):
        deleted_jobs = delete_lane_attempt_jobs(
            lane_id=lane_id,
            attempt_id=attempt_id,
            kube_context=kube_context,
            namespace=namespace,
            dry_run=dry_run,
        )
        for job_name in deleted_jobs:
            actions.append(
                {
                    "action": "delete_split_pair_whole_lane",
                    "job": job_name,
                    "lane_id": lane_id,
                    "attempt_id": attempt_id,
                    "reason_code": "split_pair_orphan_whole_lane",
                    "deleted": True,
                }
            )
        if deleted_jobs and lane_id not in redispatch_lane_ids:
            redispatch_lane_ids.append(lane_id)

    for row in detection.get("split_pair_orphans") or []:
        if (str(row.get("lane_id")), str(row.get("attempt_id"))) in split_lane_keys:
            product = str(row.get("gpu_product") or "")
            count = int(row.get("gpu_count") or 0)
            if product and count:
                reclaimed_gpus[product] += count

    skip_jobs = {str(row.get("job")) for row in (detection.get("split_pair_orphans") or []) if row.get("job")}

    def reclaim_rows(rows: Sequence[Mapping[str, Any]], *, default_reason: str) -> None:
        allowed, skipped = filter_protected_rows(rows, protect_list=protect_list)
        for row in skipped:
            actions.append(
                {
                    "action": "skip_reclaim",
                    "job": row.get("job"),
                    "lane_id": row.get("lane_id"),
                    "attempt_id": row.get("attempt_id"),
                    "reason_code": "protect_list_preserved",
                    "deleted": False,
                }
            )
        seen_jobs: set[str] = set()
        for row in allowed:
            job = str(row.get("job") or "")
            if not job or job in seen_jobs or job in skip_jobs:
                continue
            if (str(row.get("lane_id")), str(row.get("attempt_id"))) in split_lane_keys:
                continue
            seen_jobs.add(job)
            gpu_product = str(row.get("gpu_product") or "")
            gpu_count = int(row.get("gpu_count") or 0)
            ok = _delete_job(job_name=job, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
            actions.append(
                {
                    "action": "delete_job",
                    "job": job,
                    "lane_id": row.get("lane_id"),
                    "attempt_id": row.get("attempt_id"),
                    "reason_code": row.get("reason_code") or default_reason,
                    "gpu_product": gpu_product,
                    "gpu_count": gpu_count,
                    "deleted": ok,
                }
            )
            if ok and gpu_count > 0 and gpu_product:
                reclaimed_gpus[gpu_product] += gpu_count

    reclaim_rows(detection.get("orphan_policies") or [], default_reason="orphan_policy_dead_sim")
    reclaim_rows(detection.get("zombie_sims") or [], default_reason="zombie_sim_idle_policy_dead")

    clutter_jobs: set[str] = set()
    for row in detection.get("clutter_no_gpu") or []:
        job = str(row.get("job") or "")
        if not job or job in clutter_jobs:
            continue
        clutter_jobs.add(job)
        ok = _delete_job(job_name=job, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
        actions.append(
            {
                "action": "delete_job",
                "job": job,
                "lane_id": row.get("lane_id"),
                "attempt_id": row.get("attempt_id"),
                "reason_code": row.get("reason_code") or "terminated_error_clutter",
                "gpu_product": "",
                "gpu_count": 0,
                "deleted": ok,
            }
        )

    return {
        "actions": actions,
        "reclaimed_gpus_by_product": dict(reclaimed_gpus),
        "reclaimed_gpu_total": sum(reclaimed_gpus.values()),
        "jobs_deleted": sum(1 for action in actions if action.get("deleted")),
        "split_pair_lanes_reclaimed": sorted(redispatch_lane_ids),
        "protect_list_path": str(gpu_scheduling.DEFAULT_PROTECT_LIST.relative_to(ROOT)),
    }


def estimate_wall_clocks(*, c7_remaining: int = 188) -> dict[str, Any]:
    c8_lanes = arbitration.G7_C8_CONFIRMATORY_MAX_LANES
    c6_lanes = 8
    minutes = arbitration.C8_EPISODE_MINUTES_PER_LANE
    target = arbitration.CONFIRMATORY_EPISODE_TARGET

    def hours(episodes: int, lanes: int) -> float:
        return round((episodes / max(lanes, 1)) * (minutes / 60.0), 1)

    return {
        "assumptions": {
            "minutes_per_episode_per_lane": minutes,
            "c7_remaining_episodes": c7_remaining,
            "c8_confirmatory_lanes_post_c7": c8_lanes,
            "c6_confirmatory_lanes": c6_lanes,
        },
        "c7_remaining_hours_at_40_lanes": hours(c7_remaining, arbitration.C7_NORMAL_LANE_CEILING),
        "c8_confirmatory_768_hours_at_20_lanes": hours(target, c8_lanes),
        "c8_confirmatory_768_hours_at_2_lanes": hours(target, 2),
        "c6_confirmatory_768_hours_at_8_lanes": hours(target, c6_lanes),
    }


def lane_pair_scheduling_state(pods: Sequence[PodRef], *, lane_prefix: str) -> dict[str, Any]:
    by_lane: dict[str, list[PodRef]] = defaultdict(list)
    for pod in pods:
        if pod.lane_id.startswith(lane_prefix):
            by_lane[pod.lane_id].append(pod)
    healthy = pending = partial = 0
    for lane_id in sorted(by_lane):
        latest = _latest_pods(by_lane[lane_id])
        policy = latest.get("policy")
        sim = latest.get("sim")
        if _is_running_with_gpu(policy) and _is_running_with_gpu(sim):
            healthy += 1
        elif policy and sim and policy.phase == "Pending" and sim.phase == "Pending":
            pending += 1
        else:
            partial += 1
    return {
        "lane_prefix": lane_prefix,
        "healthy_pairs": healthy,
        "pending_pairs": pending,
        "partial_pairs": partial,
        "total_lanes_seen": len(by_lane),
    }


def run_gpu_pool_sweep(
    *,
    kube_context: str,
    namespace: str,
    dry_run: bool,
    c7_remaining_episodes: int = 188,
    protect_list_path: Path | None = None,
    c8_rendered_root: Path | None = None,
    c6_rendered_root: Path | None = None,
    allow_redispatch: bool = False,
) -> dict[str, Any]:
    protect_list = gpu_scheduling.load_protect_list(protect_list_path)
    raw_jobs = fetch_jobs(kube_context=kube_context, namespace=namespace)
    job_failed = build_job_failed_map(raw_jobs)
    raw_pods = fetch_pods(kube_context=kube_context, namespace=namespace)
    pods = [parsed for parsed in (parse_pod(item) for item in raw_pods) if parsed is not None]
    before_usage = summarize_gpu_usage(pods)
    detection = detect_lane_mismatches(pods, job_failed_by_name=job_failed)
    reclaim = reclaim_detected(
        detection,
        kube_context=kube_context,
        namespace=namespace,
        dry_run=dry_run,
        protect_list=protect_list,
    )
    redispatch_actions: list[dict[str, Any]] = []
    redispatch_blocked_lanes = list(reclaim.get("split_pair_lanes_reclaimed") or [])
    if redispatch_blocked_lanes and not allow_redispatch:
        redispatch_actions.append(
            {
                "action": "redispatch_blocked",
                "reason_code": "same_attempt_redispatch_requires_c8_agent_fresh_attempt_or_lock_clear",
                "lanes": redispatch_blocked_lanes,
                "note": (
                    "Automatic redispatch disabled: re-applying same attempt_id leaves simulator "
                    "PVC attempt locks (.simulator-lane-*-attempt-*.lock) and causes PreflightError."
                ),
            }
        )
    elif not dry_run and allow_redispatch and redispatch_blocked_lanes:
        import v4_gpu_placement_enforce as placement  # noqa: WPS433

        c8_lanes = [
            lane
            for lane in redispatch_blocked_lanes
            if str(lane).startswith("c8m") or str(lane).startswith("g7c8p")
        ]
        c6_lanes = [
            lane
            for lane in redispatch_blocked_lanes
            if str(lane).startswith("c6m") or str(lane).startswith("g7c6p")
        ]
        c8_root = c8_rendered_root or placement.DEFAULT_C8_RENDERED_ROOT
        c6_root = c6_rendered_root or placement.DEFAULT_C6_RENDERED_ROOT
        if c8_lanes and c8_root.is_dir():
            redispatch_actions.extend(
                placement.redispatch_lane_pairs(
                    lane_ids=c8_lanes,
                    rendered_root=c8_root,
                    policy_id="c8_a40_spread",
                    kube_context=kube_context,
                    namespace=namespace,
                    dry_run=False,
                    protect_list=protect_list,
                )
            )
        if c6_lanes and c6_root.is_dir():
            redispatch_actions.extend(
                placement.redispatch_lane_pairs(
                    lane_ids=c6_lanes,
                    rendered_root=c6_root,
                    policy_id="c6_a10040_spread",
                    kube_context=kube_context,
                    namespace=namespace,
                    dry_run=False,
                    protect_list=protect_list,
                )
            )
    if not dry_run:
        raw_pods = fetch_pods(kube_context=kube_context, namespace=namespace)
        pods = [parsed for parsed in (parse_pod(item) for item in raw_pods) if parsed is not None]
    after_usage = summarize_gpu_usage(pods)
    a40_assessment = assess_a40_fragmentation(pods, detection.get("orphan_policies") or [])
    handoff = arbitration.build_confirmatory_handoff_plan(c7_remaining_episodes=c7_remaining_episodes)
    return {
        "schema_version": "v4-gpu-pool-sweep-receipt-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kube_context": kube_context,
        "namespace": namespace,
        "dry_run": dry_run,
        "allow_redispatch": allow_redispatch,
        "startup_grace_seconds": gpu_scheduling.ISAAC_STARTUP_GRACE_SECONDS,
        "failure_root_cause_note": (
            "20260908: C8 sim PreflightError attempt lock already exists after sweep "
            "redispatch reused attempt_id without lock clear — tooling caused, not Isaac warmup."
        ),
        "detection": detection,
        "reclaim": reclaim,
        "redispatch_actions": redispatch_actions,
        "a40_assessment": a40_assessment,
        "pool_utilization_before": before_usage,
        "pool_utilization_after": after_usage,
        "scheduling_state": {
            "c7": lane_pair_scheduling_state(pods, lane_prefix="c7m"),
            "c8": lane_pair_scheduling_state(pods, lane_prefix="c8m"),
            "c6": lane_pair_scheduling_state(pods, lane_prefix="c6m"),
        },
        "wall_clock_estimates": estimate_wall_clocks(c7_remaining=c7_remaining_episodes),
        "confirmatory_handoff_plan": handoff,
        "protect_list": {
            "path": str((protect_list_path or gpu_scheduling.DEFAULT_PROTECT_LIST).relative_to(ROOT)),
            "productive_c7_pairs": len(protect_list.get("productive_c7_lane_pairs") or []),
            "retry_shards": len(protect_list.get("c7_retry_shards_in_flight_do_not_preempt") or []),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--c7-remaining", type=int, default=188)
    parser.add_argument(
        "--allow-redispatch",
        action="store_true",
        help="Dangerous: only when C8 agent supplies fresh attempt_ids or locks are cleared.",
    )
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    receipt = run_gpu_pool_sweep(
        kube_context=args.kube_context,
        namespace=args.namespace,
        dry_run=args.dry_run,
        c7_remaining_episodes=args.c7_remaining,
        allow_redispatch=args.allow_redispatch,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if not args.dry_run:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
