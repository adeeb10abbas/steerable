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
    return JobRef(
        name=name,
        lane_id=str(match.group("lane")),
        attempt_id=str(match.group("attempt")),
        role=str(match.group("role")),
        pod_phase=str(pod_phases.get(name) or "Missing"),
        has_spread=bool(labels.get("v4-gpu-spread-family")),
    )


def lane_matches_policy(lane_id: str, placement_policy: Mapping[str, Any]) -> bool:
    prefixes = placement_policy.get("lane_id_prefixes") or ()
    return any(lane_id.startswith(prefix) for prefix in prefixes)


def job_should_replace(job: JobRef) -> bool:
    """Replace only Pending/Missing pods lacking spread; never disrupt Running GPU work."""
    if job.has_spread:
        return False
    return job.pod_phase in {"Pending", "Missing", "Failed", "Unknown"}


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
) -> list[dict[str, Any]]:
    try:
        import yaml
    except ImportError as exc:
        raise GpuPlacementEnforceError("PyYAML required for --rendered-root; pip install pyyaml") from exc
    protected = gpu_scheduling.protected_c7_lane_ids(protect_list)
    placement_policy = gpu_scheduling.PLACEMENT_POLICIES[policy_id]
    actions: list[dict[str, Any]] = []
    for lane_dir in sorted(rendered_root.glob("c8m*")):
        for job_file in ("policy-job.yaml", "simulator-job.yaml"):
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
            if role.replace("simulator", "sim") not in (
                "policy",
                "simulator",
            ) and role not in placement_policy.get("roles", ()):
                pass
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
                    "rendered_from": str(path.relative_to(ROOT)),
                    "applied": ok,
                    "message": (completed.stderr or completed.stdout or "").strip()[:240],
                }
            )
    return actions
    by_role = {job.role: job for job in jobs if job.lane_id == lane_id}
    policy = by_role.get("policy")
    sim = by_role.get("sim")
    if policy and sim and policy.pod_phase == "Running" and sim.pod_phase == "Running":
        return "healthy"
    if policy and sim and policy.pod_phase == "Pending" and sim.pod_phase == "Pending":
        return "pending"
    return "partial"


def job_needs_placement(
    job: JobRef,
    *,
    placement_policy: Mapping[str, Any],
    protect_list: Mapping[str, Any],
) -> bool:
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
    before_usage = sweep.summarize_gpu_usage(before_pod_refs)
    before_c8 = sweep.lane_pair_scheduling_state(before_pod_refs, lane_prefix="c8m")

    lanes_by_id: dict[str, list[JobRef]] = defaultdict(list)
    for job in parsed:
        lanes_by_id[job.lane_id].append(job)

    actions: list[dict[str, Any]] = []
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
    for job in sorted(parsed, key=lambda row: (row.lane_id, row.role, row.name)):
        if not job_needs_placement(job, placement_policy=placement_policy, protect_list=protect_list):
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
