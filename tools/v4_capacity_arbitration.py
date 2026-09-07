#!/usr/bin/env python3
"""GPU capacity arbitration for V4 policy lanes and reclaimable model-blind waves."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / "artifacts/online_correction_v4/execution/gpu_widen_20260908/capacity_arbitration_state.json"

C7_NORMAL_LANE_CEILING = 40
C7_PILOT_PRIORITY_LANE_CEILING = 24

G7_C6_PILOT_LANE_RE = re.compile(r"^g7c6p\d+$", re.I)
G7_C8_PILOT_LANE_RE = re.compile(r"^g7c8p\d+$", re.I)
C7_LANE_RE = re.compile(r"^c7m\d+$", re.I)
LANE_JOB_RE = re.compile(
    r"^v4-(?P<lane>c7m\d+|g7c6p\d+|g7c8p\d+)-(?P<attempt>attempt\d+)-(?P<hash>[a-f0-9]+)-(?P<role>policy|sim)$",
    re.I,
)

G7_C6_PILOT_ATTEMPT_RANGE = tuple(f"attempt{aid:04d}" for aid in range(57, 65))
G7_C8_PILOT_ATTEMPT_IDS: frozenset[str] = frozenset({"attempt0020", "attempt0021"})

RECLAIMABLE_MODEL_BLIND_ATTEMPT_IDS: frozenset[str] = frozenset(
    {
        "g3r20260908g",
        "g3rb20260908v",
        "g2r20260908g",
        "g2gpu20260908g",
        "g3c5p20260908a10080g",
        "g3c6p20260908a10040g",
        "g3c6p20260908a10080g",
    }
)


@dataclass(frozen=True)
class LaneJobRef:
    name: str
    lane_id: str
    attempt_id: str
    role: str
    suspend: bool
    active: int
    succeeded: int
    failed: int
    labels: Mapping[str, str]


class CapacityArbitrationError(RuntimeError):
    """Capacity arbitration precondition failed."""


def load_state(path: Path = DEFAULT_STATE) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": "v4-capacity-arbitration-state-v1",
            "mode": "pilot_priority",
            "c7_lane_ceiling": C7_PILOT_PRIORITY_LANE_CEILING,
            "c7_normal_ceiling": C7_NORMAL_LANE_CEILING,
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(payload: dict[str, Any], path: Path = DEFAULT_STATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_lane_job(job: Mapping[str, Any]) -> LaneJobRef | None:
    name = str(job.get("metadata", {}).get("name") or "")
    match = LANE_JOB_RE.match(name)
    if not match:
        return None
    status = job.get("status") or {}
    return LaneJobRef(
        name=name,
        lane_id=str(match.group("lane")),
        attempt_id=str(match.group("attempt")),
        role=str(match.group("role")),
        suspend=bool(job.get("spec", {}).get("suspend")),
        active=int(status.get("active") or 0),
        succeeded=int(status.get("succeeded") or 0),
        failed=int(status.get("failed") or 0),
        labels={str(k): str(v) for k, v in (job.get("metadata", {}).get("labels") or {}).items()},
    )


def canonical_c6_pilot_attempt(lane_id: str) -> str | None:
    match = re.match(r"^g7c6p(\d+)$", lane_id, re.I)
    if not match:
        return None
    attempt_num = 57 + int(match.group(1))
    return f"attempt{attempt_num:04d}" if attempt_num <= 64 else None


def fetch_namespace_jobs(*, kube_context: str, namespace: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        ["kubectl", "--context", kube_context, "-n", namespace, "get", "jobs", "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise CapacityArbitrationError(completed.stderr or completed.stdout or "kubectl get jobs failed")
    return json.loads(completed.stdout).get("items", [])


def fetch_v4_lane_jobs(*, kube_context: str, namespace: str) -> list[LaneJobRef]:
    return [
        ref
        for ref in (parse_lane_job(job) for job in fetch_namespace_jobs(
            kube_context=kube_context, namespace=namespace
        ))
        if ref is not None
    ]


def _patch_job_suspend(*, job_name: str, suspend: bool, kube_context: str, namespace: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    completed = subprocess.run(
        [
            "kubectl", "--context", kube_context, "-n", namespace, "patch", "job", job_name,
            "--type=merge", "-p", json.dumps({"spec": {"suspend": suspend}}),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _delete_job(*, job_name: str, kube_context: str, namespace: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    completed = subprocess.run(
        ["kubectl", "--context", kube_context, "-n", namespace, "delete", "job", job_name, "--ignore-not-found"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def reclaim_stale_pilot_lane_jobs(
    lane_jobs: Sequence[LaneJobRef], *, kube_context: str, namespace: str, dry_run: bool,
) -> list[dict[str, Any]]:
    deleted: list[dict[str, Any]] = []
    by_lane: dict[str, list[LaneJobRef]] = {}
    for ref in lane_jobs:
        if G7_C6_PILOT_LANE_RE.match(ref.lane_id) or G7_C8_PILOT_LANE_RE.match(ref.lane_id):
            by_lane.setdefault(ref.lane_id, []).append(ref)
    for lane_id, refs in sorted(by_lane.items()):
        canonical = canonical_c6_pilot_attempt(lane_id)
        if G7_C8_PILOT_LANE_RE.match(lane_id):
            nums = [int(re.sub(r"\D", "", ref.attempt_id) or 0) for ref in refs]
            canonical = f"attempt{max(nums):04d}" if nums else None
        if not canonical:
            continue
        for ref in refs:
            if ref.attempt_id == canonical:
                continue
            ok = _delete_job(job_name=ref.name, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
            deleted.append(
                {
                    "job": ref.name,
                    "lane_id": ref.lane_id,
                    "attempt_id": ref.attempt_id,
                    "canonical_attempt_id": canonical,
                    "reason_code": "stale_pilot_lane_attempt",
                    "deleted": ok,
                }
            )
    return deleted


def reclaim_redundant_model_blind_jobs(
    jobs: Sequence[Mapping[str, Any]], *, kube_context: str, namespace: str, dry_run: bool,
) -> list[dict[str, Any]]:
    import v4_wave_admission as admission  # noqa: WPS433

    reclaimed: list[dict[str, Any]] = []
    for job in jobs:
        attempt = str((job.get("metadata", {}).get("labels") or {}).get("v4-attempt-id") or "")
        if not attempt or parse_lane_job(job) is not None:
            continue
        reclaim = (
            attempt in RECLAIMABLE_MODEL_BLIND_ATTEMPT_IDS
            or admission.is_scientifically_blocked_attempt(attempt)
            or admission.is_superseded_attempt(attempt)
            or admission.is_wrong_stratum_parallel_attempt(attempt)
            or admission.is_deprioritized_attempt(attempt)
        )
        if not reclaim:
            continue
        name = str(job["metadata"]["name"])
        status = job.get("status") or {}
        active = int(status.get("active") or 0)
        succeeded = int(status.get("succeeded") or 0)
        if succeeded >= 1 and active == 0:
            continue
        action = "suspend"
        ok = False
        if active == 0 and not job.get("spec", {}).get("suspend"):
            ok = _patch_job_suspend(job_name=name, suspend=True, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
        elif active > 0:
            ok = _delete_job(job_name=name, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
            action = "delete_active"
        reclaimed.append(
            {
                "job": name,
                "attempt_id": attempt,
                "reason_code": "blocked_or_completed_family_reclaim",
                "action": action,
                "active": active,
                "succeeded": succeeded,
                "reclaimed": ok,
            }
        )
    return reclaimed


def active_c7_lane_pairs(lane_jobs: Sequence[LaneJobRef]) -> dict[str, LaneJobRef]:
    c7_policy: dict[str, LaneJobRef] = {}
    for ref in lane_jobs:
        if C7_LANE_RE.match(ref.lane_id) and ref.role == "policy":
            current = c7_policy.get(ref.lane_id)
            if current is None or ref.attempt_id > current.attempt_id:
                c7_policy[ref.lane_id] = ref
    return {
        lane_id: ref
        for lane_id, ref in c7_policy.items()
        if not ref.suspend and (ref.active > 0 or ref.succeeded == 0)
    }


def select_c7_throttle_jobs(lane_jobs: Sequence[LaneJobRef], *, ceiling: int) -> tuple[list[LaneJobRef], dict[str, Any]]:
    active_pairs = active_c7_lane_pairs(lane_jobs)
    sim_failed = {
        ref.lane_id
        for ref in lane_jobs
        if C7_LANE_RE.match(ref.lane_id) and ref.role == "sim" and ref.failed > 0 and ref.active == 0
    }
    ranked = sorted(
        active_pairs.items(),
        key=lambda item: (item[0] in sim_failed, int(item[0][3:]) if item[0][3:].isdigit() else 999),
    )
    keep = {lane for lane, _ in ranked[:ceiling]}
    throttle = {lane for lane, _ in ranked[ceiling:]}
    jobs = [ref for ref in lane_jobs if C7_LANE_RE.match(ref.lane_id) and ref.lane_id in throttle and not ref.suspend]
    report = {
        "active_lane_pairs_before": len(active_pairs),
        "ceiling": ceiling,
        "lanes_kept": sorted(keep),
        "lanes_throttled": sorted(throttle),
        "jobs_to_suspend": len(jobs),
    }
    return jobs, report


def apply_c7_throttle(
    lane_jobs: Sequence[LaneJobRef], *, ceiling: int, kube_context: str, namespace: str, dry_run: bool,
) -> dict[str, Any]:
    jobs, report = select_c7_throttle_jobs(lane_jobs, ceiling=ceiling)
    suspended = []
    for ref in jobs:
        ok = _patch_job_suspend(job_name=ref.name, suspend=True, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
        suspended.append(
            {
                "job": ref.name,
                "lane_id": ref.lane_id,
                "attempt_id": ref.attempt_id,
                "role": ref.role,
                "reason_code": "pilot_priority_c7_throttle",
                "infra_invalid_preemption": True,
                "suspended": ok,
            }
        )
    report["suspended_jobs"] = suspended
    report["suspended_count"] = len(suspended)
    return report


def restore_c7_ceiling(
    lane_jobs: Sequence[LaneJobRef], *, ceiling: int, kube_context: str, namespace: str, dry_run: bool,
) -> dict[str, Any]:
    unsuspended = []
    for ref in lane_jobs:
        if C7_LANE_RE.match(ref.lane_id) and ref.suspend:
            if _patch_job_suspend(job_name=ref.name, suspend=False, kube_context=kube_context, namespace=namespace, dry_run=dry_run):
                unsuspended.append(ref.name)
    return {"ceiling_restored": ceiling, "unsuspended_jobs": unsuspended}


def pilots_need_priority(lane_jobs: Sequence[LaneJobRef]) -> bool:
    c6_healthy = 0
    for lane_id in sorted({ref.lane_id for ref in lane_jobs if G7_C6_PILOT_LANE_RE.match(ref.lane_id)}):
        canonical = canonical_c6_pilot_attempt(lane_id)
        if not canonical:
            continue
        canon = [ref for ref in lane_jobs if ref.lane_id == lane_id and ref.attempt_id == canonical]
        policy = next((ref for ref in canon if ref.role == "policy"), None)
        sim = next((ref for ref in canon if ref.role == "sim"), None)
        if policy and sim and not policy.suspend and not sim.suspend and policy.active > 0 and sim.active > 0:
            c6_healthy += 1
    c8_needs = any(
        G7_C8_PILOT_LANE_RE.match(ref.lane_id)
        and ref.attempt_id in G7_C8_PILOT_ATTEMPT_IDS
        and (ref.failed > 0 or (ref.active == 0 and ref.succeeded == 0))
        for ref in lane_jobs
    )
    return c6_healthy < 8 or c8_needs


def summarize_pool_usage(lane_jobs: Sequence[LaneJobRef]) -> dict[str, Any]:
    running = {k: 0 for k in ("c7_policy", "c7_sim", "c6_pilot_policy", "c6_pilot_sim", "c8_pilot_policy", "c8_pilot_sim")}
    pending = dict(running)
    for ref in lane_jobs:
        if ref.suspend:
            continue
        bucket = pending if ref.active <= 0 and ref.succeeded == 0 and ref.failed == 0 else running if ref.active > 0 else None
        if bucket is None:
            continue
        if C7_LANE_RE.match(ref.lane_id):
            bucket[f"c7_{ref.role}"] += 1
        elif G7_C6_PILOT_LANE_RE.match(ref.lane_id):
            bucket[f"c6_pilot_{ref.role}"] += 1
        elif G7_C8_PILOT_LANE_RE.match(ref.lane_id):
            bucket[f"c8_pilot_{ref.role}"] += 1
    pool_sizes = {"NVIDIA-A100-SXM4-80GB": 95, "NVIDIA-A100-SXM4-40GB": 64, "NVIDIA-A40": 40, "NVIDIA-B200": 40}
    used = {
        "NVIDIA-A100-SXM4-80GB": running["c7_policy"],
        "NVIDIA-A40": running["c7_sim"] + running["c8_pilot_policy"] + running["c8_pilot_sim"],
        "NVIDIA-A100-SXM4-40GB": running["c6_pilot_sim"],
        "NVIDIA-B200": running["c6_pilot_policy"],
    }
    return {
        "running_lane_roles": running,
        "pending_lane_roles": pending,
        "pool_used_estimate": used,
        "pool_free_estimate": {k: pool_sizes[k] - used[k] for k in pool_sizes},
        "pool_sizes": pool_sizes,
    }


def enforce_capacity_arbitration(
    *, kube_context: str, namespace: str, dry_run: bool, mode: str | None = None, c7_ceiling: int | None = None,
) -> dict[str, Any]:
    import v4_wave_admission as admission  # noqa: WPS433

    state = load_state()
    resolved_mode = mode or str(state.get("mode") or "pilot_priority")
    resolved_ceiling = (
        (c7_ceiling or int(state.get("c7_lane_ceiling") or C7_PILOT_PRIORITY_LANE_CEILING))
        if resolved_mode == "pilot_priority"
        else (c7_ceiling or int(state.get("c7_normal_ceiling") or C7_NORMAL_LANE_CEILING))
    )
    all_jobs = fetch_namespace_jobs(kube_context=kube_context, namespace=namespace)
    lane_jobs = [ref for ref in (parse_lane_job(job) for job in all_jobs) if ref is not None]
    wave_cancelled = admission.cancel_superseded_jobs(all_jobs, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
    model_blind_reclaimed = reclaim_redundant_model_blind_jobs(all_jobs, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
    stale_pilot_deleted = reclaim_stale_pilot_lane_jobs(lane_jobs, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
    if not dry_run:
        all_jobs = fetch_namespace_jobs(kube_context=kube_context, namespace=namespace)
        lane_jobs = [ref for ref in (parse_lane_job(job) for job in all_jobs) if ref is not None]
    c7_report: dict[str, Any] = {"skipped": True}
    if resolved_mode == "pilot_priority" and pilots_need_priority(lane_jobs):
        c7_report = apply_c7_throttle(lane_jobs, ceiling=resolved_ceiling, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
    elif resolved_mode == "c7_restore":
        c7_report = restore_c7_ceiling(lane_jobs, ceiling=C7_NORMAL_LANE_CEILING, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
    if not dry_run:
        lane_jobs = [ref for ref in (parse_lane_job(job) for job in fetch_namespace_jobs(kube_context=kube_context, namespace=namespace)) if ref is not None]
    payload = {
        "schema_version": "v4-capacity-arbitration-enforcement-receipt-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kube_context": kube_context,
        "namespace": namespace,
        "dry_run": dry_run,
        "mode": resolved_mode,
        "c7_lane_ceiling": resolved_ceiling,
        "cancelled_superseded_jobs": wave_cancelled,
        "reclaimed_model_blind_jobs": model_blind_reclaimed,
        "deleted_stale_pilot_jobs": stale_pilot_deleted,
        "c7_throttle": c7_report,
        "pool_utilization": summarize_pool_usage(lane_jobs),
        "pilot_priority_still_required": pilots_need_priority(lane_jobs),
    }
    if not dry_run:
        state.update(
            {
                "mode": resolved_mode,
                "c7_lane_ceiling": resolved_ceiling,
                "last_enforcement_at_utc": payload["observed_at_utc"],
                "pilot_priority_still_required": payload["pilot_priority_still_required"],
            }
        )
        save_state(state)
    return payload
