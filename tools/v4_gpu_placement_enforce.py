#!/usr/bin/env python3
"""Apply reusable GPU placement spread policies to live V4 lane Jobs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
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
import v4_gpu_pool_sweep as sweep  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

DEFAULT_RECEIPT = (
    ROOT
    / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_placement_receipt.json"
)
DEFAULT_PROTECT_LIST = gpu_scheduling.DEFAULT_PROTECT_LIST
DEFAULT_C8_RENDERED_ROOT = (
    ROOT
    / "artifacts/online_correction_v4/execution/c8_second_stack_20260908/rendered-confirmatory-r1"
)
DEFAULT_C6_RENDERED_ROOT = (
    ROOT
    / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908/rendered-c6confirm20260908f"
)
PRODUCTIVE_C8_LANES = gpu_scheduling.PRODUCTIVE_C8_LANE_IDS

LANE_JOB_RE = re.compile(
    r"^v4-(?P<lane>c7m\d+|c8m\d+|g7c6p\d+|g7c8p\d+|c6m\d+)-"
    r"(?P<attempt>attempt\d+)-(?P<hash>[a-f0-9]+)-(?P<role>policy|sim)$",
    re.I,
)


class GpuPlacementEnforceError(RuntimeError):
    """GPU placement enforcement failed."""


@dataclass(frozen=True)
class JobRef:
    name: str
    lane_id: str
    attempt_id: str
    role: str
    pod_phase: str
    has_spread: bool
    suspended: bool
    job_failed: bool


def _kubectl_json(args: list[str]) -> Any:
    completed = subprocess.run(
        ["kubectl", *args, "-o", "json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise GpuPlacementEnforceError(completed.stderr or completed.stdout or "kubectl failed")
    return json.loads(completed.stdout)


def fetch_jobs(*, kube_context: str, namespace: str) -> list[dict[str, Any]]:
    payload = _kubectl_json(
        ["--context", kube_context, "-n", namespace, "get", "jobs"]
    )
    return payload.get("items", [])


def fetch_pod_phases_by_job(*, kube_context: str, namespace: str) -> dict[str, str]:
    payload = _kubectl_json(["--context", kube_context, "-n", namespace, "get", "pods"])
    phases: dict[str, str] = {}
    for pod in payload.get("items", []):
        name = str(pod.get("metadata", {}).get("name") or "")
        phase = str((pod.get("status") or {}).get("phase") or "Unknown")
        for owner in pod.get("metadata", {}).get("ownerReferences") or []:
            if owner.get("kind") == "Job":
                phases[str(owner.get("name") or "")] = phase
                break
    return phases


def parse_job(job: Mapping[str, Any], *, pod_phases: Mapping[str, str]) -> JobRef | None:
    name = str(job.get("metadata", {}).get("name") or "")
    match = LANE_JOB_RE.match(name)
    if not match:
        return None
    labels = (job.get("spec", {}) or {}).get("template", {}).get("metadata", {}).get("labels") or {}
    status = job.get("status") or {}
    return JobRef(
        name=name,
        lane_id=str(match.group("lane")),
        attempt_id=str(match.group("attempt")),
        role=str(match.group("role")),
        pod_phase=str(pod_phases.get(name) or "Missing"),
        has_spread=bool(labels.get("v4-gpu-spread-family")),
        suspended=bool((job.get("spec") or {}).get("suspend")),
        job_failed=int(status.get("failed") or 0) > 0,
    )


def lane_matches_policy(lane_id: str, placement_policy: Mapping[str, Any]) -> bool:
    prefixes = placement_policy.get("lane_id_prefixes") or ()
    return any(lane_id.startswith(prefix) for prefix in prefixes)


def job_should_replace(job: JobRef) -> bool:
    """Replace only Pending/Failed pods lacking spread; never disrupt Running GPU work."""
    if job.has_spread:
        return False
    # Missing/Unknown often indicates a between-episodes restart while the partner leg
    # is still Running; treat those as productive partial lanes, not idle clutter.
    return job.pod_phase in {"Pending", "Failed"}


def job_has_running_partner(
    job: JobRef,
    *,
    lanes_by_id: Mapping[str, Sequence[JobRef]],
) -> bool:
    for peer in lanes_by_id.get(job.lane_id) or ():
        if peer.name == job.name or peer.attempt_id != job.attempt_id:
            continue
        if peer.pod_phase == "Running":
            return True
    return False


def build_job_doc_with_placement(
    job: Mapping[str, Any],
    *,
    placement_policy: Mapping[str, Any],
    protected_c7_lanes: Sequence[str],
) -> dict[str, Any]:
    doc = json.loads(json.dumps(job))
    labels = (doc.get("spec", {}) or {}).get("template", {}).get("metadata", {}).get("labels") or {}
    role = str(labels.get("v4-lane-role") or "")
    if role == "sim":
        role = "simulator"
    pod_spec = doc["spec"]["template"]["spec"]
    pod_labels = doc["spec"]["template"]["metadata"].setdefault("labels", {})
    extra = gpu_scheduling.inject_placement_into_pod_spec(
        pod_spec,
        placement_policy=placement_policy,
        role=role,
        protected_c7_lanes=protected_c7_lanes,
    )
    pod_labels.update(extra)
    metadata = doc.setdefault("metadata", {})
    for key in ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields"):
        metadata.pop(key, None)
    doc.pop("status", None)
    spec = doc.setdefault("spec", {})
    spec.pop("selector", None)
    template_meta = spec.setdefault("template", {}).setdefault("metadata", {})
    template_labels = template_meta.setdefault("labels", {})
    for key in (
        "controller-uid",
        "batch.kubernetes.io/controller-uid",
        "job-name",
        "batch.kubernetes.io/job-name",
    ):
        template_labels.pop(key, None)
    return doc


def patch_job_suspend(*, name: str, suspend: bool, kube_context: str, namespace: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    completed = subprocess.run(
        [
            "kubectl", "--context", kube_context, "-n", namespace,
            "patch", "job", name, "--type=merge", "-p", json.dumps({"spec": {"suspend": suspend}}),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def lane_partner_sim_running(job: JobRef, *, lanes_by_id: Mapping[str, Sequence[JobRef]]) -> bool:
    for peer in lanes_by_id.get(job.lane_id) or ():
        if peer.role != "sim" or peer.attempt_id != job.attempt_id:
            continue
        if peer.pod_phase == "Running":
            return True
    return False


def lane_between_episodes(job: JobRef, *, lanes_by_id: Mapping[str, Sequence[JobRef]]) -> bool:
    """Running policy with Pending sim on same attempt (productive restart window)."""
    if job.role != "policy" or job.pod_phase != "Running":
        return False
    if job.lane_id in PRODUCTIVE_C8_LANES:
        for peer in lanes_by_id.get(job.lane_id) or ():
            if peer.role == "sim" and peer.attempt_id == job.attempt_id and peer.pod_phase in {"Pending", "Running"}:
                return True
    return False


def policy_should_suspend_until_sim_running(
    job: JobRef,
    *,
    placement_policy: Mapping[str, Any],
    lanes_by_id: Mapping[str, Sequence[JobRef]],
    pod_refs_by_lane: Mapping[str, Mapping[str, sweep.PodRef]] | None = None,
) -> bool:
    role = "simulator" if job.role == "sim" else job.role
    if not gpu_scheduling.role_is_deferred_until_partner_running(placement_policy, role):
        return False
    if job.suspended:
        return False
    if pod_refs_by_lane and lane_partner_in_startup_grace(
        job, pod_refs_by_lane=pod_refs_by_lane, partner_role="sim"
    ):
        return False
    if lane_between_episodes(job, lanes_by_id=lanes_by_id):
        return False
    if lane_partner_sim_running(job, lanes_by_id=lanes_by_id):
        return False
    return job.pod_phase in {"Running", "Pending", "Missing"}


def delete_job(*, name: str, kube_context: str, namespace: str, dry_run: bool) -> bool:
    if dry_run:
        return True
    completed = subprocess.run(
        [
            "kubectl", "--context", kube_context, "-n", namespace,
            "delete", "job", name, "--ignore-not-found", "--wait=false",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def apply_job(*, doc: Mapping[str, Any], kube_context: str, namespace: str, dry_run: bool) -> tuple[bool, str]:
    if dry_run:
        return True, ""
    payload = json.dumps(doc, sort_keys=True)
    completed = subprocess.run(
        ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"],
        input=payload,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip()
    return False, (completed.stderr or completed.stdout or "kubectl create failed").strip()


def apply_rendered_lane_jobs(
    *,
    rendered_root: Path,
    policy_id: str,
    kube_context: str,
    namespace: str,
    dry_run: bool,
    protect_list: Mapping[str, Any],
    lane_glob: str = "c8m*",
    sim_first: bool = True,
) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise GpuPlacementEnforceError("PyYAML required for --rendered-root; pip install pyyaml") from exc
    protected = gpu_scheduling.protected_c7_lane_ids(protect_list)
    placement_policy = gpu_scheduling.PLACEMENT_POLICIES[policy_id]
    actions: list[dict[str, Any]] = []
    job_files = (
        ("simulator-job.yaml", "policy-job.yaml")
        if sim_first
        else ("policy-job.yaml", "simulator-job.yaml")
    )
    for lane_dir in sorted(rendered_root.glob(lane_glob)):
        for job_file in job_files:
            path = lane_dir / job_file
            if not path.is_file():
                continue
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            role = str(doc["spec"]["template"]["metadata"]["labels"].get("v4-lane-role") or "")
            lane_id = str(doc["spec"]["template"]["metadata"]["labels"].get("v4-lane-id") or "")
            attempt_id = str(doc["spec"]["template"]["metadata"]["labels"].get("v4-attempt-id") or "")
            if gpu_scheduling.is_lane_protected(
                lane_id=lane_id, attempt_id=attempt_id, protect_list=protect_list
            ):
                continue
            if role == "policy" and gpu_scheduling.role_is_deferred_until_partner_running(
                placement_policy, "policy"
            ):
                doc.setdefault("spec", {})["suspend"] = True
            pod_spec = doc["spec"]["template"]["spec"]
            pod_labels = doc["spec"]["template"]["metadata"].setdefault("labels", {})
            extra = gpu_scheduling.inject_placement_into_pod_spec(
                pod_spec,
                placement_policy=placement_policy,
                role=role,
                protected_c7_lanes=protected,
            )
            pod_labels.update(extra)
            rendered = yaml.safe_dump(doc, sort_keys=False)
            cmd = ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"]
            if dry_run:
                cmd = [
                    "kubectl", "--context", kube_context, "-n", namespace,
                    "apply", "--dry-run=server", "-f", "-",
                ]
            completed = subprocess.run(cmd, input=rendered, capture_output=True, text=True)
            ok = completed.returncode == 0 or "AlreadyExists" in (completed.stderr or "")
            actions.append(
                {
                    "action": "apply_rendered_job",
                    "job": doc["metadata"]["name"],
                    "lane_id": lane_id,
                    "role": role,
                    "sim_first": sim_first,
                    "policy_suspended_on_create": bool(doc.get("spec", {}).get("suspend")),
                    "rendered_from": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                    "applied": ok,
                    "message": (completed.stderr or completed.stdout or "").strip()[:240],
                }
            )
    return actions


def apply_rendered_lane_dir(
    *,
    lane_dir: Path,
    policy_id: str,
    kube_context: str,
    namespace: str,
    dry_run: bool,
    protect_list: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return apply_rendered_lane_jobs(
        rendered_root=lane_dir.parent,
        policy_id=policy_id,
        kube_context=kube_context,
        namespace=namespace,
        dry_run=dry_run,
        protect_list=protect_list,
        lane_glob=lane_dir.name,
        sim_first=True,
    )


def redispatch_lane_pairs(
    *,
    lane_ids: Sequence[str],
    rendered_root: Path,
    policy_id: str,
    kube_context: str,
    namespace: str,
    dry_run: bool,
    protect_list: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Delete all jobs for listed lanes then re-apply sim-first whole pair from rendered bundle."""
    actions: list[dict[str, Any]] = []
    wanted = set(lane_ids)
    if not wanted or not rendered_root.is_dir():
        return actions
    raw_jobs = fetch_jobs(kube_context=kube_context, namespace=namespace)
    for job in raw_jobs:
        name = str(job.get("metadata", {}).get("name") or "")
        match = LANE_JOB_RE.match(name)
        if not match or str(match.group("lane")) not in wanted:
            continue
        deleted = delete_job(name=name, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
        actions.append(
            {
                "action": "delete_for_redispatch",
                "job": name,
                "lane_id": match.group("lane"),
                "deleted": deleted,
            }
        )
    for lane_id in sorted(wanted):
        matches = sorted(rendered_root.glob(f"{lane_id}-*"))
        if not matches:
            actions.append({"action": "redispatch_skipped", "lane_id": lane_id, "reason": "no_rendered_dir"})
            continue
        actions.extend(
            apply_rendered_lane_dir(
                lane_dir=matches[0],
                policy_id=policy_id,
                kube_context=kube_context,
                namespace=namespace,
                dry_run=dry_run,
                protect_list=protect_list,
            )
        )
    return actions


def lane_partner_in_startup_grace(
    job: JobRef,
    *,
    pod_refs_by_lane: Mapping[str, Mapping[str, sweep.PodRef]],
    partner_role: str,
) -> bool:
    latest = pod_refs_by_lane.get(job.lane_id) or {}
    partner = latest.get(partner_role)
    if partner is None or partner.attempt_id != job.attempt_id:
        return False
    return sweep._in_startup_grace(partner)


def lane_any_partner_in_startup_grace(
    job: JobRef,
    *,
    pod_refs_by_lane: Mapping[str, Mapping[str, sweep.PodRef]],
) -> bool:
    return lane_partner_in_startup_grace(
        job, pod_refs_by_lane=pod_refs_by_lane, partner_role="sim"
    ) or lane_partner_in_startup_grace(
        job, pod_refs_by_lane=pod_refs_by_lane, partner_role="policy"
    )


def build_pod_refs_by_lane(pods: Sequence[sweep.PodRef]) -> dict[str, dict[str, sweep.PodRef]]:
    by_lane: dict[str, list[sweep.PodRef]] = defaultdict(list)
    for pod in pods:
        by_lane[pod.lane_id].append(pod)
    return {lane_id: sweep._latest_pods(items) for lane_id, items in by_lane.items()}


def job_needs_placement(
    job: JobRef,
    *,
    placement_policy: Mapping[str, Any],
    protect_list: Mapping[str, Any],
    lanes_by_id: Mapping[str, Sequence[JobRef]],
    pod_refs_by_lane: Mapping[str, Mapping[str, sweep.PodRef]] | None = None,
) -> bool:
    if job.lane_id in PRODUCTIVE_C8_LANES:
        return False
    if pod_refs_by_lane and lane_any_partner_in_startup_grace(
        job, pod_refs_by_lane=pod_refs_by_lane
    ):
        return False
    if not lane_matches_policy(job.lane_id, placement_policy):
        return False
    role = "simulator" if job.role == "sim" else job.role
    if role not in placement_policy.get("roles", ()):
        return False
    if gpu_scheduling.is_lane_protected(
        lane_id=job.lane_id,
        attempt_id=job.attempt_id,
        protect_list=protect_list,
    ):
        return False
    if gpu_scheduling.role_is_deferred_until_partner_running(placement_policy, role):
        if not lane_partner_sim_running(job, lanes_by_id=lanes_by_id):
            return False
    if job_has_running_partner(job, lanes_by_id=lanes_by_id):
        return False
    return job_should_replace(job)


def enforce_gpu_placement(
    *,
    policy_id: str,
    kube_context: str,
    namespace: str,
    dry_run: bool,
    protect_list_path: Path,
    rendered_root: Path | None = None,
    settle_seconds: int = 20,
) -> dict[str, Any]:
    placement_policy = gpu_scheduling.PLACEMENT_POLICIES[policy_id]
    protect_list = gpu_scheduling.load_protect_list(protect_list_path)
    protected_c7_lanes = gpu_scheduling.protected_c7_lane_ids(protect_list)

    raw_jobs = fetch_jobs(kube_context=kube_context, namespace=namespace)
    pod_phases = fetch_pod_phases_by_job(kube_context=kube_context, namespace=namespace)
    parsed = [
        row for row in (parse_job(item, pod_phases=pod_phases) for item in raw_jobs) if row is not None
    ]
    before_pods = sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    before_pod_refs = [p for p in (sweep.parse_pod(item) for item in before_pods) if p is not None]
    pod_refs_by_lane = build_pod_refs_by_lane(before_pod_refs)
    before_usage = sweep.summarize_gpu_usage(before_pod_refs)
    before_c8 = sweep.lane_pair_scheduling_state(before_pod_refs, lane_prefix="c8m")

    lanes_by_id: dict[str, list[JobRef]] = defaultdict(list)
    for job in parsed:
        lanes_by_id[job.lane_id].append(job)

    actions: list[dict[str, Any]] = []
    schedule_order = {role: index for index, role in enumerate(placement_policy.get("schedule_order") or ())}

    def role_sort_key(job: JobRef) -> tuple[int, str, str]:
        role = "simulator" if job.role == "sim" else job.role
        return (schedule_order.get(role, 99), job.lane_id, job.name)

    for job in sorted(parsed, key=role_sort_key):
        if policy_should_suspend_until_sim_running(
            job,
            placement_policy=placement_policy,
            lanes_by_id=lanes_by_id,
            pod_refs_by_lane=pod_refs_by_lane,
        ):
            ok = patch_job_suspend(
                name=job.name, suspend=True, kube_context=kube_context, namespace=namespace, dry_run=dry_run
            )
            actions.append(
                {
                    "action": "suspend_until_sim_running",
                    "job": job.name,
                    "lane_id": job.lane_id,
                    "role": job.role,
                    "reason_code": "sim_first_gate",
                    "suspended": ok,
                }
            )
        elif (
            job.role == "policy"
            and job.suspended
            and lane_partner_sim_running(job, lanes_by_id=lanes_by_id)
        ):
            ok = patch_job_suspend(
                name=job.name, suspend=False, kube_context=kube_context, namespace=namespace, dry_run=dry_run
            )
            actions.append(
                {
                    "action": "unsuspend_policy_sim_ready",
                    "job": job.name,
                    "lane_id": job.lane_id,
                    "unsuspended": ok,
                }
            )

    if rendered_root is not None:
        actions.extend(
            apply_rendered_lane_jobs(
                rendered_root=rendered_root,
                policy_id=policy_id,
                kube_context=kube_context,
                namespace=namespace,
                dry_run=dry_run,
                protect_list=protect_list,
            )
        )
    for job in sorted(parsed, key=role_sort_key):
        if not job_needs_placement(
            job,
            placement_policy=placement_policy,
            protect_list=protect_list,
            lanes_by_id=lanes_by_id,
            pod_refs_by_lane=pod_refs_by_lane,
        ):
            continue
        if pod_refs_by_lane and lane_any_partner_in_startup_grace(
            job, pod_refs_by_lane=pod_refs_by_lane
        ):
            actions.append(
                {
                    "action": "skip_startup_grace",
                    "job": job.name,
                    "lane_id": job.lane_id,
                    "reason_code": "isaac_warmup_grace",
                    "grace_seconds": gpu_scheduling.ISAAC_STARTUP_GRACE_SECONDS,
                }
            )
            continue
        if job.pod_phase == "Running":
            actions.append(
                {
                    "action": "skip_running_pod",
                    "job": job.name,
                    "lane_id": job.lane_id,
                    "pod_phase": job.pod_phase,
                    "reason_code": "running_pod_preserved",
                }
            )
            continue
        raw = next(item for item in raw_jobs if item.get("metadata", {}).get("name") == job.name)
        updated = build_job_doc_with_placement(
            raw,
            placement_policy=placement_policy,
            protected_c7_lanes=protected_c7_lanes,
        )
        deleted = delete_job(name=job.name, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
        applied, apply_message = (
            apply_job(doc=updated, kube_context=kube_context, namespace=namespace, dry_run=dry_run)
            if deleted
            else (False, "delete_failed")
        )
        actions.append(
            {
                "action": "replace_job_with_spread",
                "job": job.name,
                "lane_id": job.lane_id,
                "attempt_id": job.attempt_id,
                "role": job.role,
                "pod_phase": job.pod_phase,
                "spread_family": placement_policy["spread_family_label"],
                "deleted": deleted,
                "applied": applied,
                "apply_message": apply_message,
            }
        )

    if not dry_run and settle_seconds > 0:
        time.sleep(settle_seconds)

    after_pods = sweep.fetch_pods(kube_context=kube_context, namespace=namespace)
    after_pod_refs = [p for p in (sweep.parse_pod(item) for item in after_pods) if p is not None]
    after_usage = sweep.summarize_gpu_usage(after_pod_refs)
    after_c8 = sweep.lane_pair_scheduling_state(after_pod_refs, lane_prefix="c8m")
    after_c6 = sweep.lane_pair_scheduling_state(after_pod_refs, lane_prefix="c6m")

    additional_healthy = after_c8["healthy_pairs"] - before_c8["healthy_pairs"]
    binding = gpu_scheduling.analyze_lane_pair_capacity_binding(
        family="c8" if policy_id == "c8_a40_spread" else "c6"
    )

    minutes = arbitration.C8_EPISODE_MINUTES_PER_LANE
    target = arbitration.CONFIRMATORY_EPISODE_TARGET

    def hours(episodes: int, lanes: int, *, minutes_per_lane: float = minutes) -> float:
        return round((episodes / max(lanes, 1)) * (minutes_per_lane / 60.0), 1)

    c6_minutes = arbitration.C6_EPISODE_MINUTES_PER_LANE

    return {
        "schema_version": "v4-gpu-placement-receipt-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kube_context": kube_context,
        "namespace": namespace,
        "dry_run": dry_run,
        "placement_policy": placement_policy,
        "sim_first_ordering": {
            "schedule_order": list(placement_policy.get("schedule_order") or ()),
            "spread_roles": list(placement_policy.get("spread_roles") or ()),
            "defer_roles_until_partner_running": list(
                placement_policy.get("defer_roles_until_partner_running") or ()
            ),
            "productive_c8_lanes_preserved": sorted(PRODUCTIVE_C8_LANES),
            "startup_grace_seconds": gpu_scheduling.ISAAC_STARTUP_GRACE_SECONDS,
        },
        "protect_list_path": str(protect_list_path.relative_to(ROOT)),
        "protected_c7_lane_count": len(protected_c7_lanes),
        "actions": actions,
        "jobs_replaced": sum(
            1
            for row in actions
            if row.get("action") in {"replace_job_with_spread", "apply_rendered_job"}
            and row.get("applied")
        ),
        "scheduling_state_before": {"c8": before_c8},
        "scheduling_state_after": {"c8": after_c8, "c6": after_c6},
        "additional_c8_healthy_pairs": additional_healthy,
        "pool_utilization_before": before_usage,
        "pool_utilization_after": after_usage,
        "lane_pair_capacity_binding": binding,
        "wall_clock_estimates": {
            "c8_confirmatory_768_hours_before": {
                "at_2_lanes": hours(target, before_c8["healthy_pairs"] or 2),
                "at_20_lanes_post_c7": hours(target, arbitration.G7_C8_CONFIRMATORY_MAX_LANES),
            },
            "c8_confirmatory_768_hours_after": {
                "at_current_healthy_lanes": hours(target, max(after_c8["healthy_pairs"], 1)),
                "at_20_lanes_post_c7": hours(target, arbitration.G7_C8_CONFIRMATORY_MAX_LANES),
            },
            "c6_confirmatory_768_hours_at_8_lanes": round(
                (target / 8) * (c6_minutes / 60.0), 1
            ),
        },
        "c8_admission_coordination": {
            "renderer_field": "placement_policy",
            "renderer_value_for_c8": "c8_a40_spread",
            "enforce_command": ".venv/bin/python tools/enforce_v4_capacity_arbitration.py --mode gpu_placement --policy c8_a40_spread",
            "note": "C8 agent must set placement_policy in lane render specs; capacity agent replaces Pending Jobs lacking spread.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument(
        "--policy",
        choices=tuple(gpu_scheduling.PLACEMENT_POLICIES),
        default="c8_a40_spread",
    )
    parser.add_argument("--protect-list", type=Path, default=DEFAULT_PROTECT_LIST)
    parser.add_argument(
        "--rendered-root",
        type=Path,
        default=None,
        help="Optional C8 rendered bundle root; applies spread via kubectl create from YAML.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--settle-seconds", type=int, default=20)
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
    receipt = enforce_gpu_placement(
        policy_id=args.policy,
        kube_context=args.kube_context,
        namespace=args.namespace,
        dry_run=args.dry_run,
        protect_list_path=args.protect_list,
        rendered_root=args.rendered_root,
        settle_seconds=args.settle_seconds,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if not args.dry_run:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
