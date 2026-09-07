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
        "g3c6p20260908a10040g",
    }
)

# C6 parallel-stratum migration: A100-80GB attempt competes with C2; Agent B owns a10040g dispatch.
WRONG_STRATUM_PARALLEL_ATTEMPT_IDS: frozenset[str] = frozenset(
    {
        "g3c6p20260908a10080g",
    }
)

# Horizontal G3 confirmatory failed at scale 0.5 — C1/C3/C4 episodes scientifically blocked.
SCIENTIFICALLY_BLOCKED_ATTEMPT_IDS: frozenset[str] = frozenset(
    {
        "g3r20260908g",
    }
)

# C5 vertical positive live control fails on reachability — deprioritized, not cancelled.
DEPRIORITIZED_ATTEMPT_IDS: frozenset[str] = frozenset(
    {
        "g3c5p20260908a10080g",
    }
)

ACHIEVABLE_EPISODE_COUNTS: dict[str, int] = {
    "C7_object_pair": 768,
    "C6_containment": 768,
    "C2_reference_binding": 4096,
    "C8_second_stack": 768,
}

BLOCKED_EPISODE_COUNTS: dict[str, int] = {
    "C1_C3_C4_horizontal": 9728,
    "C5_vertical": 1536,
}

C7_ATTEMPT_RE = re.compile(r"^attempt0\d+$|^c7m\d+$", re.I)
NATURAL_GRASP_ATTEMPT_RE = re.compile(r"g3ngp|g3ngrb|nglive|natgrasp|livectrl", re.I)


class WaveAdmissionError(RuntimeError):
    """Admission precondition failed."""


@dataclass(frozen=True)
class AdmissionTier:
    tier_id: str
    priority: int
    attempt_ids: frozenset[str]
    attempt_patterns: tuple[str, ...]
    gpu_product: str | None
    alternate_gpu_products: tuple[str, ...]
    admission_class: str
    seed_count: int
    seed_minutes: float
    episodes_gated: int
    description: str


ADMISSION_TIERS: tuple[AdmissionTier, ...] = (
    AdmissionTier(
        tier_id="natural_grasp_live_control",
        priority=0,
        attempt_ids=frozenset(),
        attempt_patterns=(r"g3ngp", r"g3ngrb", r"nglive", r"natgrasp", r"livectrl", r"horizng"),
        gpu_product=None,
        alternate_gpu_products=(),
        admission_class="primary",
        seed_count=2,
        seed_minutes=LIVE_CONTROL_JOB_MINUTES,
        episodes_gated=6400,
        description="Per-fixture natural-grasp controls for families with open dispatch gates",
    ),
    AdmissionTier(
        tier_id="reference_binding_g3",
        priority=1,
        attempt_ids=frozenset({"g3rb20260908v"}),
        attempt_patterns=(),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        alternate_gpu_products=(),
        admission_class="primary",
        seed_count=128,
        seed_minutes=G3_PATH_SEED_MINUTES,
        episodes_gated=4096,
        description="C2 reference_binding G3 path homogeneous wave (live controls passed)",
    ),
    AdmissionTier(
        tier_id="c6_containment_g3",
        priority=2,
        attempt_ids=frozenset({"g3c6p20260908a10040g"}),
        attempt_patterns=(r"g3c6p20260908a10040g",),
        gpu_product="NVIDIA-A100-SXM4-40GB",
        alternate_gpu_products=("NVIDIA-B200",),
        admission_class="parallel_stratum",
        seed_count=64,
        seed_minutes=G3_PATH_SEED_MINUTES,
        episodes_gated=768,
        description="C6 containment 64-seed G3 ladder on parallel stratum (Agent B gate owner)",
    ),
    AdmissionTier(
        tier_id="horizontal_g2_evidence",
        priority=3,
        attempt_ids=frozenset({"g2r20260908g", "g2gpu20260908g"}),
        attempt_patterns=(),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        alternate_gpu_products=("NVIDIA-A40",),
        admission_class="backfill",
        seed_count=128,
        seed_minutes=G2_SEED_MINUTES,
        episodes_gated=0,
        description="Horizontal G2 evidence completion and determinism attestation (backfill, non-blocking)",
    ),
    AdmissionTier(
        tier_id="c5_vertical_g3",
        priority=4,
        attempt_ids=frozenset({"g3c5p20260908a10080g"}),
        attempt_patterns=(r"g3c5p20260908a10080g",),
        gpu_product="NVIDIA-A100-SXM4-80GB",
        alternate_gpu_products=(),
        admission_class="deprioritized",
        seed_count=128,
        seed_minutes=G3_PATH_SEED_MINUTES,
        episodes_gated=0,
        description="C5 vertical G3 (positive live control blocked on reachability; evidence preserved)",
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


def is_scientifically_blocked_attempt(attempt_id: str) -> bool:
    return attempt_id in SCIENTIFICALLY_BLOCKED_ATTEMPT_IDS


def is_deprioritized_attempt(attempt_id: str) -> bool:
    return attempt_id in DEPRIORITIZED_ATTEMPT_IDS


def is_wrong_stratum_parallel_attempt(attempt_id: str) -> bool:
    return attempt_id in WRONG_STRATUM_PARALLEL_ATTEMPT_IDS


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


def _tier_incomplete_attempts(
    tier: AdmissionTier,
    attempt_summary: Mapping[str, Mapping[str, int]],
) -> list[str]:
    tier_attempts = [
        aid
        for aid in attempt_summary
        if tier_for_attempt(aid) == tier and not is_superseded_attempt(aid)
    ]
    return sorted(
        aid
        for aid in tier_attempts
        if attempt_has_work_remaining(attempt_summary[aid])
    )


def _achievable_priority_incomplete(
    attempt_summary: Mapping[str, Mapping[str, int]],
) -> bool:
    for tier in ADMISSION_TIERS:
        if tier.admission_class in {"primary", "parallel_stratum", "backfill"}:
            if _tier_incomplete_attempts(tier, attempt_summary):
                return True
    return False


def compute_admitted_attempts(
    attempt_summary: Mapping[str, Mapping[str, int]],
) -> tuple[list[str], dict[str, Any]]:
    """Return attempt ids that may run now and diagnostic report."""
    admitted: list[str] = []
    report: dict[str, Any] = {
        "tiers": [],
        "c7_always_admitted": [],
        "scientifically_blocked": [],
        "wrong_stratum_parallel_suspended": [],
        "deprioritized_deferred": [],
    }

    for attempt_id in attempt_summary:
        if is_c7_attempt(attempt_id):
            report["c7_always_admitted"].append(attempt_id)
        if is_scientifically_blocked_attempt(attempt_id):
            report["scientifically_blocked"].append(attempt_id)
        if is_wrong_stratum_parallel_attempt(attempt_id):
            report["wrong_stratum_parallel_suspended"].append(attempt_id)

    achievable_priority_pending = _achievable_priority_incomplete(attempt_summary)

    for tier in ADMISSION_TIERS:
        incomplete = _tier_incomplete_attempts(tier, attempt_summary)
        if not incomplete:
            tier_attempts = [
                aid
                for aid in attempt_summary
                if tier_for_attempt(aid) == tier and not is_superseded_attempt(aid)
            ]
            if not tier_attempts:
                report["tiers"].append({"tier_id": tier.tier_id, "status": "no_jobs"})
            else:
                report["tiers"].append({"tier_id": tier.tier_id, "status": "complete"})
            continue

        if tier.admission_class == "deprioritized":
            if achievable_priority_pending:
                report["deprioritized_deferred"].extend(incomplete)
                report["tiers"].append(
                    {
                        "tier_id": tier.tier_id,
                        "status": "deprioritized_deferred",
                        "attempt_ids": incomplete,
                    }
                )
                continue
            admitted.extend(incomplete)
            report["tiers"].append(
                {
                    "tier_id": tier.tier_id,
                    "status": "admitted_deprioritized",
                    "attempt_ids": incomplete,
                }
            )
            continue

        if tier.admission_class == "backfill":
            admitted.extend(incomplete)
            report["tiers"].append(
                {
                    "tier_id": tier.tier_id,
                    "status": "admitted_backfill",
                    "attempt_ids": incomplete,
                }
            )
            continue

        admitted.extend(incomplete)
        report["tiers"].append(
            {
                "tier_id": tier.tier_id,
                "status": "admitted",
                "attempt_ids": incomplete,
            }
        )

    report["admitted_attempts"] = sorted(set(admitted))
    return report["admitted_attempts"], report


def estimate_wave_minutes(
    *,
    seed_count: int,
    pool_size: int,
    seed_minutes: float,
) -> float:
    if pool_size <= 0:
        return float(seed_count) * seed_minutes
    return math.ceil(seed_count / pool_size) * seed_minutes


def _remaining_seeds_for_tier(
    tier: AdmissionTier,
    attempt_summary: Mapping[str, Mapping[str, int]],
) -> tuple[int, bool]:
    tier_attempts = [
        aid
        for aid in attempt_summary
        if tier_for_attempt(aid) == tier and not is_superseded_attempt(aid)
    ]
    if not tier_attempts:
        return 0, True
    remaining = 0
    for aid in tier_attempts:
        stats = attempt_summary[aid]
        expected = min(tier.seed_count, int(stats.get("total") or tier.seed_count))
        remaining += max(0, expected - int(stats.get("succeeded") or 0))
    return remaining, remaining <= 0


def build_wall_clock_estimates(
    attempt_summary: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    estimates: list[dict[str, Any]] = []
    cumulative = 0.0
    parallel_stratum_minutes: list[float] = []
    for tier in ADMISSION_TIERS:
        if tier.admission_class == "deprioritized":
            continue
        pool = GPU_POOL_SIZES.get(tier.gpu_product or "NVIDIA-A100-SXM4-80GB", 95)
        remaining_seeds = tier.seed_count
        tier_complete = False
        if attempt_summary is not None:
            remaining_seeds, tier_complete = _remaining_seeds_for_tier(tier, attempt_summary)
        if tier_complete or remaining_seeds <= 0:
            minutes = 0.0
        else:
            minutes = estimate_wave_minutes(
                seed_count=remaining_seeds,
                pool_size=pool if tier.gpu_product else 4,
                seed_minutes=tier.seed_minutes,
            )
        if tier.admission_class == "parallel_stratum":
            parallel_stratum_minutes.append(minutes)
        elif tier.admission_class == "backfill":
            pass  # backfill does not extend serialized critical path
        elif tier.admission_class == "primary":
            cumulative += minutes
        else:
            cumulative += minutes
        estimates.append(
            {
                "tier_id": tier.tier_id,
                "attempt_ids": sorted(tier.attempt_ids),
                "admission_class": tier.admission_class,
                "gpu_product": tier.gpu_product,
                "alternate_gpu_products": list(tier.alternate_gpu_products),
                "remaining_seeds": remaining_seeds,
                "tier_complete": tier_complete,
                "pool_size": pool if tier.gpu_product else 4,
                "seed_minutes_assumed": tier.seed_minutes,
                "estimated_wave_minutes": minutes,
                "cumulative_minutes_from_now": cumulative,
                "episodes_gated": tier.episodes_gated,
            }
        )
    parallel_critical_path = max(parallel_stratum_minutes) if parallel_stratum_minutes else 0.0
    serialized_a10080 = sum(
        e["estimated_wave_minutes"]
        for e in estimates
        if e.get("admission_class") == "primary"
        and e.get("gpu_product") == "NVIDIA-A100-SXM4-80GB"
    )
    parallel_qualification_minutes = max(serialized_a10080, parallel_critical_path)
    return {
        "schema_version": "v4-wave-admission-estimates-v2",
        "estimated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "assumptions": {
            "primary_plus_backfill_on_a10080": True,
            "parallel_strata_enabled": True,
            "g3_path_seed_minutes": G3_PATH_SEED_MINUTES,
            "g2_seed_minutes": G2_SEED_MINUTES,
            "gpu_pool_sizes": GPU_POOL_SIZES,
            "parallel_attestation_products": [
                "NVIDIA-A100-SXM4-40GB",
                "NVIDIA-B200",
            ],
        },
        "qualification_completion_hours": round(parallel_qualification_minutes / 60.0, 1),
        "parallel_qualification_minutes": parallel_qualification_minutes,
        "serialized_a10080_minutes": serialized_a10080,
        "main_episode_start_note": (
            "6,400 achievable episodes (C7/C6/C2/C8) may dispatch after their qualification "
            "tiers complete and per-fixture live controls pass. 11,264 episodes across C1/C3/C4 "
            "horizontal and C5 vertical remain scientifically blocked."
        ),
        "tier_estimates": estimates,
    }


def build_achievable_episode_estimates(
    attempt_summary: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    achievable_total = sum(ACHIEVABLE_EPISODE_COUNTS.values())
    blocked_total = sum(BLOCKED_EPISODE_COUNTS.values())
    qual = build_wall_clock_estimates(attempt_summary)
    c2_remaining = 0
    c6_remaining = 0
    g2_remaining = 0
    if attempt_summary is not None:
        c2_remaining, _ = _remaining_seeds_for_tier(
            next(t for t in ADMISSION_TIERS if t.tier_id == "reference_binding_g3"),
            attempt_summary,
        )
        c6_remaining, _ = _remaining_seeds_for_tier(
            next(t for t in ADMISSION_TIERS if t.tier_id == "c6_containment_g3"),
            attempt_summary,
        )
        g2_remaining, _ = _remaining_seeds_for_tier(
            next(t for t in ADMISSION_TIERS if t.tier_id == "horizontal_g2_evidence"),
            attempt_summary,
        )
    c2_minutes = estimate_wave_minutes(
        seed_count=c2_remaining or 115,
        pool_size=GPU_POOL_SIZES["NVIDIA-A100-SXM4-80GB"],
        seed_minutes=G3_PATH_SEED_MINUTES,
    )
    c6_minutes_a10040 = estimate_wave_minutes(
        seed_count=c6_remaining or 129,
        pool_size=GPU_POOL_SIZES["NVIDIA-A100-SXM4-40GB"],
        seed_minutes=G3_PATH_SEED_MINUTES,
    )
    c6_minutes_b200 = estimate_wave_minutes(
        seed_count=c6_remaining or 129,
        pool_size=GPU_POOL_SIZES["NVIDIA-B200"],
        seed_minutes=G3_PATH_SEED_MINUTES,
    )
    serialized_total = c2_minutes + c6_minutes_a10040
    parallel_total = max(c2_minutes, c6_minutes_a10040)
    parallel_b200 = max(c2_minutes, c6_minutes_b200)
    return {
        "schema_version": "v4-achievable-episode-estimates-v1",
        "estimated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "achievable_episode_counts": ACHIEVABLE_EPISODE_COUNTS,
        "achievable_episode_total": achievable_total,
        "blocked_episode_counts": BLOCKED_EPISODE_COUNTS,
        "blocked_episode_total": blocked_total,
        "qualification_gates": qual,
        "remaining_g3_seeds": {
            "reference_binding_g3rb20260908v": c2_remaining,
            "containment_g3c6p20260908a10040g": c6_remaining,
            "horizontal_g2_evidence_backfill": g2_remaining,
        },
        "qualification_wall_clock_minutes": {
            "serialized_a10080_then_a10040": serialized_total,
            "parallel_a10080_c2_with_a10040_c6": parallel_total,
            "parallel_a10080_c2_with_b200_c6": parallel_b200,
        },
        "qualification_wall_clock_hours": {
            "serialized": round(serialized_total / 60.0, 1),
            "parallel_a10040_c6": round(parallel_total / 60.0, 1),
            "parallel_b200_c6": round(parallel_b200 / 60.0, 1),
        },
        "parallel_strata_materially_faster": parallel_total < serialized_total,
        "parallel_strata_time_saved_minutes": serialized_total - parallel_total,
        "note": (
            "C7 behavioral episodes continue on protected a10080-policy/a40-simulator lanes "
            "throughout. G2 horizontal evidence runs as A100-80GB backfill and does not extend "
            "the C2 critical path when C2 dominates pool occupancy."
        ),
    }


def require_dispatch_admission(*, attempt_id: str, attempt_summary: Mapping[str, Mapping[str, int]] | None = None) -> None:
    if is_c7_attempt(attempt_id) or is_superseded_attempt(attempt_id):
        return
    if is_scientifically_blocked_attempt(attempt_id):
        raise WaveAdmissionError(
            f"wave admission gate blocked dispatch for {attempt_id}: scientifically blocked "
            f"(horizontal G3 confirmatory failed; C1/C3/C4 episodes not authorized)"
        )
    if attempt_summary is None:
        return
    admitted, report = compute_admitted_attempts(attempt_summary)
    if attempt_id not in admitted:
        raise WaveAdmissionError(
            f"wave admission gate blocked dispatch for {attempt_id}: currently admitted "
            f"{admitted or ['none']}. Priority tiers: "
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
        should_run = (
            attempt in admitted_set or attempt in c7_attempts
        ) and not is_scientifically_blocked_attempt(attempt) and not is_wrong_stratum_parallel_attempt(
            attempt
        )
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
        "achievable_episode_estimates": build_achievable_episode_estimates(summary),
    }
