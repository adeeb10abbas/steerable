#!/usr/bin/env python3
"""Serialized wave admission for V4 model-blind Kubernetes queues."""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "artifacts/online_correction_v4/setup/v4_wave_admission_registry.json"

# GPU pool sizes used for wall-clock estimates and concurrency caps.
GPU_POOL_SIZES: dict[str, int] = {
    "NVIDIA-A100-SXM4-80GB": 95,
    "NVIDIA-A100-SXM4-40GB": 64,
    "NVIDIA-A40": 40,
    "NVIDIA-B200": 40,
}

G3_PATH_SEED_MINUTES = 47
G2_SEED_MINUTES = 35
LIVE_CONTROL_JOB_MINUTES = 60

# Attempt ids explicitly superseded — receipts cannot join homogeneous aggregates.
SUPERSEDED_ATTEMPT_IDS: frozenset[str] = frozenset(
    {
        "g3rb20260908k",
        "g3rb20260908r",
        "g3rb20260908r2",
        "g2r20260908a10080w",
        "g3r20260908a10080w",
        "g2r20260908a10080r",
        "g3r20260908a10080r",
        "g3w20260908a10080a",
        "g3c5p20260908a10080",
        "g3c5p20260908a10080b",
        "g3c5p20260908a10080c",
        "g3c5p20260908a10080d",
        "g3c5p20260908a10080e",
        "g3c5p20260908a10080f",
        "g3c6p20260908a10080",
        "g3c6p20260908a10080b",
        "g3c6p20260908a10080c",
        "g3c6p20260908a10080d",
        "g3c6p20260908a10080e",
        "g3c6p20260908a10080f",
        "g2gpu20260908a40c",
        "g2gpu20260908a40b",
        "g2gpu20260908a10040",
        "g2gpu20260908b200",
        "g2r20260908a10080a",
        "g2r20260908a10080b",
        "g2r20260908a10080c",
        "g2r20260908a10080d",
        "g2r20260908a10080e",
        "g2r20260908a10080f",
        "g2r20260908a10080h",
        "g3r20260908a10080a",
        "g3r20260908a10080b",
        "g3r20260908a10080c",
        "g3r20260908a10080d",
        "g3r20260908a10080e",
        "g3r20260908a10080h",
    }
)

# Homogeneous-pin waves at the current branch standard (pin c401fb4 / g-suffix).
CURRENT_HOMOGENEOUS_ATTEMPT_IDS: frozenset[str] = frozenset(
    {
        "g3r20260908g",
        "g2r20260908g",
        "g2gpu20260908g",
        "g3rb20260908v",
        "g3c5p20260908a10080g",
        "g3c6p20260908a10080g",
    }
)

C7_ATTEMPT_RE = re.compile(r"^attempt0\d+$|^c7m\d+$", re.I)
NATURAL_GRASP_ATTEMPT_RE = re.compile(r"g3ngp|nglive|natgrasp|livectrl", re.I)


class WaveAdmissionError(RuntimeError):
    """Admission precondition failed."""


@dataclass(frozen=True)
class AdmissionTier:
    tier_id: str
    priority: int
    attempt_ids: frozenset[str]
    attempt_patterns: tuple[str, ...]
    gpu_product: str | None
    seed_count: int
    seed_minutes: float
    episodes_gated: int
    description: str


ADMISSION_TIERS: tuple[AdmissionTier, ...] = (
    AdmissionTier(
        tier_id="natural_grasp_live_control",
        priority=0,
        attempt_ids=frozenset(),
        attempt_patterns=(r"g3ngp", r"nglive", r"natgrasp", r"livectrl", r"horizng"),
        gpu_product=None,
        seed_count=2,
        seed_minutes=LIVE_CONTROL_JOB_MINUTES,
        episodes_gated=17664,
        description="Per-fixture Isaac/SimplerEnv natural-grasp positive/negative controls",
    ),
    AdmissionTier(
        tier_id="horizontal_g3_path",
        priority=1,
        attempt_ids=frozenset({"g3r20260908g"}),
        attempt_patterns=(),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        seed_count=128,
        seed_minutes=G3_PATH_SEED_MINUTES,
        episodes_gated=9728,
        description="Horizontal geometry_repair_v2 G3 path scale-0.5 homogeneous wave",
    ),
    AdmissionTier(
        tier_id="horizontal_g2",
        priority=2,
        attempt_ids=frozenset({"g2r20260908g", "g2gpu20260908g"}),
        attempt_patterns=(),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        seed_count=128,
        seed_minutes=G2_SEED_MINUTES,
        episodes_gated=9728,
        description="Horizontal G2 homogeneous wave (includes A40 determinism s000 in-wave)",
    ),
    AdmissionTier(
        tier_id="reference_binding_g3",
        priority=3,
        attempt_ids=frozenset({"g3rb20260908v"}),
        attempt_patterns=(),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        seed_count=128,
        seed_minutes=G3_PATH_SEED_MINUTES,
        episodes_gated=4096,
        description="C2 reference_binding G3 path homogeneous wave",
    ),
    AdmissionTier(
        tier_id="c5_c6_g3",
        priority=4,
        attempt_ids=frozenset({"g3c5p20260908a10080g", "g3c6p20260908a10080g"}),
        attempt_patterns=(r"g3c5p20260908a10080", r"g3c6p20260908a10080"),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        seed_count=128,
        seed_minutes=G3_PATH_SEED_MINUTES,
        episodes_gated=2304,
        description="C5 vertical and C6 containment G3 smokes then full waves at homogeneous pin",
    ),
)


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    if not path.is_file():
        return {"schema_version": "v4-wave-admission-registry-v1", "waves": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_registry(payload: dict[str, Any], path: Path = DEFAULT_REGISTRY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_c7_attempt(attempt_id: str) -> bool:
    return bool(C7_ATTEMPT_RE.match(attempt_id))


def is_superseded_attempt(attempt_id: str) -> bool:
    if attempt_id in CURRENT_HOMOGENEOUS_ATTEMPT_IDS:
        return False
    if attempt_id in SUPERSEDED_ATTEMPT_IDS:
        return True
    # Mixed-pin horizontal/C5/C6 waves without the homogeneous g-suffix.
    if re.match(r"^(g2r|g3r)20260908a10080", attempt_id) and not attempt_id.endswith("g"):
        return True
    if re.match(r"^g3c[56]p20260908a10080", attempt_id) and not attempt_id.endswith("g"):
        return True
    if re.match(r"^g3rb20260908r", attempt_id):
        return True
    return False


def tier_for_attempt(attempt_id: str) -> AdmissionTier | None:
    for tier in ADMISSION_TIERS:
        if attempt_id in tier.attempt_ids:
            return tier
        for pattern in tier.attempt_patterns:
            if re.search(pattern, attempt_id, re.I):
                return tier
    return None


def fetch_v4_jobs(*, kube_context: str, namespace: str) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "kubectl",
            "--context",
            kube_context,
            "-n",
            namespace,
            "get",
            "jobs",
            "-l",
            "app.kubernetes.io/part-of=vla-wam-v4",
            "-o",
            "json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise WaveAdmissionError(completed.stderr or completed.stdout or "kubectl get jobs failed")
    return json.loads(completed.stdout).get("items", [])


def summarize_attempts(jobs: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for job in jobs:
        attempt = (job.get("metadata", {}).get("labels") or {}).get("v4-attempt-id")
        if not attempt:
            continue
        attempt = str(attempt)
        bucket = summary.setdefault(
            attempt,
            {"total": 0, "active": 0, "succeeded": 0, "failed": 0, "pending": 0, "suspended": 0},
        )
        bucket["total"] += 1
        if job.get("spec", {}).get("suspend"):
            bucket["suspended"] += 1
        status = job.get("status") or {}
        active = int(status.get("active") or 0)
        succeeded = int(status.get("succeeded") or 0)
        failed = int(status.get("failed") or 0)
        bucket["active"] += active
        bucket["succeeded"] += succeeded
        bucket["failed"] += failed
        if active == 0 and succeeded == 0 and failed == 0:
            bucket["pending"] += 1
    return summary


def attempt_is_complete(stats: Mapping[str, int], *, expected_jobs: int = 128) -> bool:
    return int(stats.get("succeeded") or 0) >= expected_jobs


def attempt_has_work_remaining(stats: Mapping[str, int]) -> bool:
    total = int(stats.get("total") or 0)
    succeeded = int(stats.get("succeeded") or 0)
    failed = int(stats.get("failed") or 0)
    return succeeded + failed < total


def compute_admitted_attempts(
    attempt_summary: Mapping[str, Mapping[str, int]],
) -> tuple[list[str], dict[str, Any]]:
    """Return attempt ids that may run now and diagnostic report."""
    admitted: list[str] = []
    report: dict[str, Any] = {"tiers": [], "c7_always_admitted": []}

    for attempt_id, stats in attempt_summary.items():
        if is_c7_attempt(attempt_id):
            report["c7_always_admitted"].append(attempt_id)

    for tier in ADMISSION_TIERS:
        tier_attempts = [
            aid
            for aid in attempt_summary
            if tier_for_attempt(aid) == tier and not is_superseded_attempt(aid)
        ]
        if not tier_attempts:
            report["tiers"].append({"tier_id": tier.tier_id, "status": "no_jobs"})
            continue
        incomplete = [
            aid
            for aid in tier_attempts
            if attempt_has_work_remaining(attempt_summary[aid])
        ]
        if not incomplete:
            report["tiers"].append({"tier_id": tier.tier_id, "status": "complete"})
            continue
        # Admit all incomplete waves in this tier (e.g. G2 A100-80GB + A40 determinism s000).
        chosen = sorted(incomplete)
        admitted.extend(chosen)
        report["tiers"].append(
            {
                "tier_id": tier.tier_id,
                "status": "admitted",
                "attempt_ids": chosen,
            }
        )
        break  # only one tier active at a time

    report["admitted_attempts"] = admitted
    return admitted, report


def estimate_wave_minutes(
    *,
    seed_count: int,
    pool_size: int,
    seed_minutes: float,
) -> float:
    if pool_size <= 0:
        return float(seed_count) * seed_minutes
    return math.ceil(seed_count / pool_size) * seed_minutes


def build_wall_clock_estimates(
    attempt_summary: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    estimates: list[dict[str, Any]] = []
    cumulative = 0.0
    for tier in ADMISSION_TIERS:
        pool = GPU_POOL_SIZES.get(tier.gpu_product or "NVIDIA-A100-SXM4-80GB", 95)
        remaining_seeds = tier.seed_count
        tier_complete = False
        if attempt_summary is not None:
            tier_attempts = [
                aid
                for aid in attempt_summary
                if tier_for_attempt(aid) == tier and not is_superseded_attempt(aid)
            ]
            if not tier_attempts:
                tier_complete = True
                remaining_seeds = 0
            else:
                remaining_seeds = 0
                for aid in tier_attempts:
                    stats = attempt_summary[aid]
                    expected = min(
                        tier.seed_count,
                        int(stats.get("total") or tier.seed_count),
                    )
                    remaining_seeds += max(
                        0,
                        expected - int(stats.get("succeeded") or 0),
                    )
                tier_complete = remaining_seeds <= 0
        if tier_complete or remaining_seeds <= 0:
            minutes = 0.0
        else:
            minutes = estimate_wave_minutes(
                seed_count=remaining_seeds,
                pool_size=pool if tier.gpu_product else 4,
                seed_minutes=tier.seed_minutes,
            )
        cumulative += minutes
        estimates.append(
            {
                "tier_id": tier.tier_id,
                "attempt_ids": sorted(tier.attempt_ids),
                "gpu_product": tier.gpu_product,
                "remaining_seeds": remaining_seeds,
                "tier_complete": tier_complete,
                "pool_size": pool if tier.gpu_product else 4,
                "seed_minutes_assumed": tier.seed_minutes,
                "estimated_wave_minutes": minutes,
                "cumulative_minutes_from_now": cumulative,
                "episodes_gated": tier.episodes_gated,
            }
        )
    return {
        "schema_version": "v4-wave-admission-estimates-v1",
        "estimated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assumptions": {
            "serialized_tiers": True,
            "g3_path_seed_minutes": G3_PATH_SEED_MINUTES,
            "g2_seed_minutes": G2_SEED_MINUTES,
            "gpu_pool_sizes": GPU_POOL_SIZES,
            "parallel_attestation_products": [
                "NVIDIA-A100-SXM4-40GB",
                "NVIDIA-B200",
            ],
        },
        "qualification_completion_hours": round(cumulative / 60.0, 1),
        "main_episode_start_note": (
            "17,664 policy episodes may dispatch only after all admitted qualification tiers "
            "complete, per-fixture live controls pass at the homogeneous pin, and runtime locks release."
        ),
        "tier_estimates": estimates,
    }


def require_dispatch_admission(*, attempt_id: str, attempt_summary: Mapping[str, Mapping[str, int]] | None = None) -> None:
    if is_c7_attempt(attempt_id) or is_superseded_attempt(attempt_id):
        return
    if attempt_summary is None:
        return
    admitted, report = compute_admitted_attempts(attempt_summary)
    if attempt_id not in admitted:
        raise WaveAdmissionError(
            f"wave admission gate blocked dispatch for {attempt_id}: currently admitted "
            f"{admitted or ['none']}. Serialized tiers: "
            + ", ".join(t.tier_id for t in ADMISSION_TIERS)
            + f". Detail: {json.dumps(report['tiers'], sort_keys=True)}"
        )


def set_job_suspend(
    *,
    job_name: str,
    suspend: bool,
    kube_context: str,
    namespace: str,
    dry_run: bool,
) -> bool:
    patch = json.dumps({"spec": {"suspend": suspend}})
    command = [
        "kubectl",
        "--context",
        kube_context,
        "-n",
        namespace,
        "patch",
        "job",
        job_name,
        "--type=merge",
        "-p",
        patch,
    ]
    if dry_run:
        return True
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.returncode == 0


def delete_job(
    *,
    job_name: str,
    kube_context: str,
    namespace: str,
    dry_run: bool,
) -> bool:
    command = [
        "kubectl",
        "--context",
        kube_context,
        "-n",
        namespace,
        "delete",
        "job",
        job_name,
        "--ignore-not-found",
    ]
    if dry_run:
        return True
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.returncode == 0


def cancel_superseded_jobs(
    jobs: Sequence[Mapping[str, Any]],
    *,
    kube_context: str,
    namespace: str,
    dry_run: bool,
) -> list[dict[str, Any]]:
    cancelled: list[dict[str, Any]] = []
    for job in jobs:
        attempt = str((job.get("metadata", {}).get("labels") or {}).get("v4-attempt-id") or "")
        if not is_superseded_attempt(attempt):
            continue
        name = str(job["metadata"]["name"])
        status = job.get("status") or {}
        ok = delete_job(job_name=name, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
        cancelled.append(
            {
                "job": name,
                "attempt_id": attempt,
                "reason_code": "superseded_mixed_pin_or_stale",
                "active": int(status.get("active") or 0),
                "succeeded": int(status.get("succeeded") or 0),
                "failed": int(status.get("failed") or 0),
                "deleted": ok,
            }
        )
    return cancelled


def enforce_admission_on_cluster(
    jobs: Sequence[Mapping[str, Any]],
    *,
    kube_context: str,
    namespace: str,
    dry_run: bool,
) -> dict[str, Any]:
    summary = summarize_attempts(jobs)
    admitted, admission_report = compute_admitted_attempts(summary)
    admitted_set = set(admitted)
    c7_attempts = {aid for aid in summary if is_c7_attempt(aid)}

    suspended: list[str] = []
    unsuspended: list[str] = []
    for job in jobs:
        attempt = str((job.get("metadata", {}).get("labels") or {}).get("v4-attempt-id") or "")
        if not attempt or is_superseded_attempt(attempt):
            continue
        name = str(job["metadata"]["name"])
        should_run = attempt in admitted_set or attempt in c7_attempts
        currently_suspended = bool(job.get("spec", {}).get("suspend"))
        if should_run and currently_suspended:
            if set_job_suspend(
                job_name=name,
                suspend=False,
                kube_context=kube_context,
                namespace=namespace,
                dry_run=dry_run,
            ):
                unsuspended.append(name)
        elif not should_run and not currently_suspended:
            if set_job_suspend(
                job_name=name,
                suspend=True,
                kube_context=kube_context,
                namespace=namespace,
                dry_run=dry_run,
            ):
                suspended.append(name)

    return {
        "attempt_summary": summary,
        "admission_report": admission_report,
        "admitted_attempts": admitted,
        "jobs_suspended": len(suspended),
        "jobs_unsuspended": len(unsuspended),
        "suspended_sample": suspended[:20],
        "unsuspended_sample": unsuspended[:20],
        "wall_clock_estimates": build_wall_clock_estimates(summary),
    }
