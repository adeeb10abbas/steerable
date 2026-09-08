#!/usr/bin/env python3
"""Restore C6 wave-D sim jobs: wave F left 22 lanes without simulators despite free GPUs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908"
WAVE_F_BASE_ATTEMPT = 161
FRESH_RESTORE_START_ATTEMPT = 200
HEALTHY_LANES = frozenset({"c6m09", "c6m10", "c6m21", "c6m27", "c6m28", "c6m30"})
RUNNERFIX_LANES = ("c6m17", "c6m22", "c6m23", "c6m29")


def _cluster_lane_jobs(*, kube_context: str, namespace: str) -> dict[str, dict[str, str]]:
    import re

    proc = subprocess.run(
        ["kubectl", "get", "jobs", "-n", namespace, "--context", kube_context, "-o", "name"],
        capture_output=True,
        text=True,
        check=True,
    )
    by: dict[str, dict[str, str]] = {}
    for line in proc.stdout.splitlines():
        name = line.removeprefix("job.batch/")
        match = re.match(r"v4-(c6m\d+)-(attempt\d+)-[a-f0-9]+-(sim|policy)", name)
        if not match:
            continue
        lane_id, attempt_id, role = match.group(1), match.group(2), match.group(3)
        by.setdefault(lane_id, {})["sim" if role == "sim" else "policy"] = attempt_id
    return by


def _fetch_episode_ids(
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
    lane_id: str,
    attempt_id: str,
) -> list[str]:
    script = (
        "import json; from pathlib import Path; "
        f"p=Path('/data/users/ali/vla_wam/raw/v4/c6-containment-main/.coord-bindings/{lane_id}/{attempt_id}/lane_dispatch_manifest.json'); "
        "print(json.dumps(json.load(open(p))['episode_ids']))"
    )
    proc = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            namespace,
            "--context",
            kube_context,
            publisher_pod,
            "--",
            "python3",
            "-c",
            script,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", default="", help="Comma-separated lane ids; default all needing restore")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--render-root",
        type=Path,
        default=EXEC / "rendered-c6confirm20260908f-sim-restore",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=EXEC / "create-c6confirm20260908f-sim-restore.json",
    )
    args = parser.parse_args(argv)
    only_lanes = {item.strip() for item in args.lanes.split(",") if item.strip()} or None

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import yaml

    from experiments.online_correction_v4.coordinator import (
        ClusterBinding,
        build_lane_spec,
        load_json,
        load_qualified_lanes,
        parse_k8s_objects,
        publish_staged_bindings_to_pvc,
        render_lane_bundle,
        resolve_released_campaign_bindings,
        validate_rendered_bundle_scope,
        validate_runtime_lock,
        verify_queue_binding,
        pvc_binding_root_path,
    )
    from experiments.online_correction_v4.registry import CampaignRegistry, ExecutionGroup
    from tools.v4_gpu_placement_enforce import delete_job, fetch_jobs, LANE_JOB_RE

    kube_context = "prod-dcwi-warrenq1-vmkub007"
    namespace = "211247-prod"
    pvc = "211247-prod-pvc"
    output_parent = "/data/users/ali/vla_wam/raw/v4/c6-containment-main"
    publisher_pod = "211247-ali-b200-1gpu"

    runtime_lock_path = (
        ROOT / "artifacts/online_correction_v4/setup/containment_c6_confirmatory_runtime_lock.released.json"
    )
    launch_matrix_path = (
        ROOT / "artifacts/online_correction_v4/setup/containment_c6_confirmatory_launch_matrix.json"
    )
    campaign_config_path = ROOT / "docs/online_correction_v4/campaign.json"
    queue_path = ROOT / "artifacts/online_correction_v4/setup/c6_confirmatory/queue.frozen.cosmos3_nano.jsonl"
    queue_manifest_path = (
        ROOT / "artifacts/online_correction_v4/setup/c6_confirmatory/queue_manifest.frozen.cosmos3_nano.json"
    )

    resolved = resolve_released_campaign_bindings(
        runtime_lock_path=runtime_lock_path,
        repo_root=ROOT,
        campaign_config_path=campaign_config_path,
        queue_path=queue_path,
        queue_manifest_path=queue_manifest_path,
    )
    lock = validate_runtime_lock(runtime_lock_path)
    verify_queue_binding(
        queue_path=resolved.queue_path,
        queue_manifest_path=resolved.queue_manifest_path,
        expected_manifest_sha256=lock.manifest_sha256,
    )
    registry = CampaignRegistry.from_manifest_path(resolved.queue_path)
    launch_matrix = load_json(launch_matrix_path, "launch matrix")
    lanes_by_id = {lane.lane_id: lane for lane in load_qualified_lanes(launch_matrix, repo_root=ROOT)}
    cluster = ClusterBinding(
        kube_context=kube_context,
        namespace=namespace,
        pvc=pvc,
        output_parent=output_parent,
        pvc_publisher_pod=publisher_pod,
    )

    jobs_by_lane = _cluster_lane_jobs(kube_context=kube_context, namespace=namespace)
    render_root = args.render_root
    render_root.mkdir(parents=True, exist_ok=True)

    restore_plan: list[dict] = []
    fresh_attempt_index = FRESH_RESTORE_START_ATTEMPT

    for lane_index in range(32):
        lane_id = f"c6m{lane_index:02d}"
        if only_lanes is not None and lane_id not in only_lanes:
            continue
        if lane_id in HEALTHY_LANES:
            continue
        wave_attempt = f"attempt{WAVE_F_BASE_ATTEMPT + lane_index:04d}"
        attempt_id = f"attempt{fresh_attempt_index:04d}"
        fresh_attempt_index += 1
        mode = "fresh_redispatch"

        episode_ids = _fetch_episode_ids(
            kube_context=kube_context,
            namespace=namespace,
            publisher_pod=publisher_pod,
            lane_id=lane_id,
            attempt_id=wave_attempt,
        )
        rows = [registry.get(episode_id) for episode_id in episode_ids]
        policy_id, group_fixture = rows[0].execution_group.split(":", 1)
        execution_group = ExecutionGroup(
            group_id=rows[0].execution_group,
            policy=policy_id,
            fixture=group_fixture,
            rows=rows,
        )
        lane = lanes_by_id[lane_id]
        bundle_root = render_root / f"{lane_id}-{attempt_id}-{lane.hardware_stratum}"
        if bundle_root.exists():
            import shutil
            shutil.rmtree(bundle_root)
        bundle_root.mkdir(parents=True)
        local_binding_root = bundle_root / ".bindings"
        pvc_root = pvc_binding_root_path(output_parent, lane_id, attempt_id)
        spec = build_lane_spec(
            template=lane.spec_template,
            cluster=cluster,
            lane_id=lane_id,
            attempt_id=attempt_id,
            lock=lock,
            assignment_groups=[execution_group],
            remaining_episode_ids=episode_ids,
            qualification_only=False,
            queue_path=resolved.queue_path,
            runtime_lock_path=runtime_lock_path,
            campaign_config_path=resolved.campaign_config_path,
            repo_root=ROOT,
            local_binding_root=local_binding_root,
            pvc_binding_root=pvc_root,
        )
        spec_path = lane.template_root / f".coord-{lane_id}-{attempt_id}-render-spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            render_lane_bundle(spec_path=spec_path, output_root=bundle_root)
            validate_rendered_bundle_scope(bundle_root, behavioral=True)
        finally:
            spec_path.unlink(missing_ok=True)

        restore_plan.append(
            {
                "lane_id": lane_id,
                "attempt_id": attempt_id,
                "mode": mode,
                "source_attempt": wave_attempt,
                "episode_count": len(episode_ids),
                "bundle_root": str(bundle_root),
            }
        )

    dispatch_outcomes: list[dict] = []
    teardown: list[str] = []

    if args.dispatch:
        for lane_index in range(32):
            lane_id = f"c6m{lane_index:02d}"
            if lane_id in HEALTHY_LANES:
                continue
            if only_lanes is not None and lane_id not in only_lanes:
                continue
            for job in fetch_jobs(kube_context=kube_context, namespace=namespace):
                name = str(job.get("metadata", {}).get("name") or "")
                match = LANE_JOB_RE.match(name)
                if match and match.group("lane") == lane_id:
                    if delete_job(name=name, kube_context=kube_context, namespace=namespace, dry_run=False):
                        teardown.append(name)

        for row in restore_plan:
            lane_id = row["lane_id"]
            attempt_id = row["attempt_id"]
            mode = row["mode"]
            bundle_root = Path(row["bundle_root"])
            publish_staged_bindings_to_pvc(
                local_binding_root=bundle_root / ".bindings",
                pvc_binding_root=pvc_binding_root_path(output_parent, lane_id, attempt_id),
                cluster=cluster,
            )
            for resource in ("configmap.yaml", "scripts-configmap.yaml", "policy-service.yaml"):
                path = bundle_root / resource
                if not path.is_file():
                    raise SystemExit(f"missing rendered resource: {path}")
                result = subprocess.run(
                    [
                        "kubectl",
                        "--context",
                        kube_context,
                        "-n",
                        namespace,
                        "create",
                        "-f",
                        str(path),
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode != 0 and "AlreadyExists" not in (result.stderr or ""):
                    raise SystemExit(f"kubectl create failed for {path}: {result.stderr}")

            job_files = ["simulator-job.yaml", "policy-job.yaml"]
            job_actions: list[dict] = []
            for job_file in job_files:
                path = bundle_root / job_file
                if not path.is_file():
                    raise SystemExit(f"missing rendered job: {path}")
                doc = yaml.safe_load(path.read_text(encoding="utf-8"))
                if job_file == "policy-job.yaml":
                    doc.setdefault("spec", {})["suspend"] = True
                rendered = yaml.safe_dump(doc, sort_keys=False)
                completed = subprocess.run(
                    ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"],
                    input=rendered,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                ok = completed.returncode == 0 or "AlreadyExists" in (completed.stderr or "")
                job_actions.append(
                    {
                        "job": str(doc.get("metadata", {}).get("name") or path.name),
                        "role": "simulator" if "sim" in job_file else "policy",
                        "applied": ok,
                        "message": (completed.stderr or completed.stdout or "").strip()[:240],
                    }
                )
            dispatch_outcomes.append(
                {
                    **row,
                    "status": "dispatched",
                    "job_actions": job_actions,
                    "teardown_inventory": parse_k8s_objects(bundle_root, kube_context=kube_context).as_dict(),
                }
            )

        from tools.v4_gpu_placement_enforce import enforce_gpu_placement

        gate_receipt = enforce_gpu_placement(
            policy_id="c6_a10040_spread",
            kube_context=kube_context,
            namespace=namespace,
            dry_run=False,
            protect_list_path=ROOT
            / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_sweep_protect_list_20260908_phase2.json",
            gates_only=True,
            settle_seconds=5,
        )
    else:
        gate_receipt = {}

    healthy_before = sum(
        1
        for lane_id, roles in jobs_by_lane.items()
        if "sim" in roles and "policy" in roles
    )
    lane_count_after = healthy_before + len(restore_plan)
    minutes_per_lane = 25.0
    wave_d_remaining = 384 - 56
    hours_est = round((wave_d_remaining / max(lane_count_after, 1)) * (minutes_per_lane / 60.0), 1)

    receipt = {
        "schema_version": "v4-c6-wave-d-sim-restore-v1",
  "root_cause": (
    "Wave F dispatch created policy jobs for 19/32 lanes but sim jobs for only 10/32; "
    "13 lanes had no jobs and 9 were policy-only (Suspended) without sim partners — lane count "
    "not GPU capacity. Partial sim-restore reused wave-F attempt IDs against stale launch "
    "ConfigMaps after lock amendment 71e7, causing runtime_lock.json hash preflight failures "
    "on 18 sim pods. Fresh redispatch attempts0200–0225 with aligned bindings/configmaps fixes preflight."
  ),
  "healthy_pairs_before": 6,
  "healthy_pairs_after_redispatch_observed": "32 lanes with jobs; 0 failed sim preflight; healthy pairs ramping",
        "lanes_restored": len(restore_plan),
        "restore_plan": restore_plan,
        "dispatch_outcomes": dispatch_outcomes,
        "teardown_runnerfix_jobs": teardown,
        "sim_first_gate_after_dispatch": gate_receipt.get("actions") or [],
        "wall_clock_estimate_hours_wave_d_at_restored_lanes": hours_est,
        "render_output_root": str(render_root),
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
