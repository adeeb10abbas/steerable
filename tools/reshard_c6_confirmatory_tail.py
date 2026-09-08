#!/usr/bin/env python3
"""Re-shard all remaining C6 confirmatory episodes across 32 authorized lanes (fresh attempts)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908"
AUTHORIZED_LANE_COUNT = 32
FRESH_START_ATTEMPT = 300
MINUTES_PER_EPISODE = 25.0


def _fetch_completed_episode_ids(*, kube_context: str, namespace: str, publisher_pod: str) -> set[str]:
    script = r"""
import json, os
from pathlib import Path
root = Path("/data/users/ali/vla_wam/raw/v4/c6-containment-main")
done = set()
for dirpath, _, filenames in os.walk(root):
    if "COMPLETE.json" not in filenames:
        continue
    ep = Path(dirpath) / "episode.json"
    if ep.is_file():
        done.add(json.loads(ep.read_text())["episode_id"])
print(json.dumps(sorted(done)))
"""
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
    return set(json.loads(proc.stdout.strip()))


def _clear_c6_locks(*, kube_context: str, namespace: str, publisher_pod: str, pvc_root: str) -> dict:
    script = f"""
import glob, json, os
root = {json.dumps(pvc_root)}
removed = []
for path in glob.glob(root + "/**/.simulator-lane-c6m*-attempt-attempt*.lock", recursive=True):
    os.remove(path)
    removed.append(path)
print(json.dumps({{"removed_count": len(removed)}}))
"""
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
        check=False,
    )
    if proc.returncode != 0:
        return {"error": proc.stderr.strip()}
    return json.loads(proc.stdout.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--render-root",
        type=Path,
        default=EXEC / "rendered-c6confirm20260908f-reshard-r8",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=EXEC / "create-c6confirm20260908f-reshard-r8.json",
    )
    parser.add_argument("--start-attempt", type=int, default=FRESH_START_ATTEMPT)
    args = parser.parse_args(argv)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import yaml

    from experiments.online_correction_v4.coordinator import (
        ClusterBinding,
        build_lane_spec,
        load_json,
        load_qualified_lanes,
        publish_staged_bindings_to_pvc,
        render_lane_bundle,
        resolve_released_campaign_bindings,
        shard_group_units,
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
    queue_episode_ids = [row.episode_id for row in registry.rows if row.family == "C6"]
    completed = _fetch_completed_episode_ids(
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=publisher_pod,
    )
    remaining = [episode_id for episode_id in queue_episode_ids if episode_id not in completed]
    if not remaining:
        receipt = {
            "schema_version": "v4-c6-confirmatory-reshard-v1",
            "status": "nothing_remaining",
            "queue_episodes": len(queue_episode_ids),
            "completed_on_pvc": len(completed),
        }
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0

    rows = [registry.get(episode_id) for episode_id in remaining]
    policy_id, group_fixture = rows[0].execution_group.split(":", 1)
    execution_group = ExecutionGroup(
        group_id=rows[0].execution_group,
        policy=policy_id,
        fixture=group_fixture,
        rows=rows,
    )
    launch_matrix = load_json(launch_matrix_path, "launch matrix")
    c6_lanes = sorted(
        [lane for lane in load_qualified_lanes(launch_matrix, repo_root=ROOT) if lane.lane_id.startswith("c6m")],
        key=lambda lane: lane.lane_id,
    )
    if len(c6_lanes) != AUTHORIZED_LANE_COUNT:
        raise SystemExit(f"expected {AUTHORIZED_LANE_COUNT} qualified C6 lanes, got {len(c6_lanes)}")

    buckets = shard_group_units([(execution_group, remaining)], c6_lanes)
    render_root = args.render_root
    render_root.mkdir(parents=True, exist_ok=True)
    cluster = ClusterBinding(
        kube_context=kube_context,
        namespace=namespace,
        pvc=pvc,
        output_parent=output_parent,
        pvc_publisher_pod=publisher_pod,
    )

    plan: list[dict] = []
    attempt_index = args.start_attempt
    for lane in c6_lanes:
        units = buckets.get(lane.lane_id) or []
        episode_ids: list[str] = []
        for _group, ids in units:
            episode_ids.extend(ids)
        if not episode_ids:
            continue
        attempt_id = f"attempt{attempt_index:04d}"
        attempt_index += 1
        bundle_root = render_root / f"{lane.lane_id}-{attempt_id}-{lane.hardware_stratum}"
        if bundle_root.exists():
            import shutil

            shutil.rmtree(bundle_root)
        bundle_root.mkdir(parents=True)
        local_binding_root = bundle_root / ".bindings"
        pvc_root = pvc_binding_root_path(output_parent, lane.lane_id, attempt_id)
        spec = build_lane_spec(
            template=lane.spec_template,
            cluster=cluster,
            lane_id=lane.lane_id,
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
        spec_path = lane.template_root / f".coord-{lane.lane_id}-{attempt_id}-render-spec.json"
        spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            render_lane_bundle(spec_path=spec_path, output_root=bundle_root)
            validate_rendered_bundle_scope(bundle_root, behavioral=True)
        finally:
            spec_path.unlink(missing_ok=True)
        plan.append(
            {
                "lane_id": lane.lane_id,
                "attempt_id": attempt_id,
                "episode_count": len(episode_ids),
                "bundle_root": str(bundle_root),
            }
        )

    eps_per_lane = max(row["episode_count"] for row in plan) if plan else 0
    hours_est = round((eps_per_lane * MINUTES_PER_EPISODE) / 60.0, 1)

    teardown: list[str] = []
    dispatch_outcomes: list[dict] = []
    lock_clear: dict = {}
    gate: dict = {}

    if args.dispatch:
        lock_clear = _clear_c6_locks(
            kube_context=kube_context,
            namespace=namespace,
            publisher_pod=publisher_pod,
            pvc_root=output_parent,
        )
        for job in fetch_jobs(kube_context=kube_context, namespace=namespace):
            name = str(job.get("metadata", {}).get("name") or "")
            match = LANE_JOB_RE.match(name)
            if match and match.group("lane").startswith("c6m"):
                if delete_job(name=name, kube_context=kube_context, namespace=namespace, dry_run=False):
                    teardown.append(name)

        for row in plan:
            bundle_root = Path(row["bundle_root"])
            lane_id = row["lane_id"]
            attempt_id = row["attempt_id"]
            publish_staged_bindings_to_pvc(
                local_binding_root=bundle_root / ".bindings",
                pvc_binding_root=pvc_binding_root_path(output_parent, lane_id, attempt_id),
                cluster=cluster,
            )
            for resource in ("configmap.yaml", "scripts-configmap.yaml", "policy-service.yaml"):
                path = bundle_root / resource
                subprocess.run(
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
                    check=True,
                )
            job_actions: list[dict] = []
            for job_file in ("simulator-job.yaml", "policy-job.yaml"):
                doc = yaml.safe_load((bundle_root / job_file).read_text(encoding="utf-8"))
                if job_file == "policy-job.yaml":
                    doc.setdefault("spec", {})["suspend"] = True
                subprocess.run(
                    ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"],
                    input=yaml.safe_dump(doc, sort_keys=False),
                    capture_output=True,
                    text=True,
                    check=True,
                )
                job_actions.append({"job": doc["metadata"]["name"], "applied": True})
            dispatch_outcomes.append({**row, "job_actions": job_actions})

        from tools.v4_gpu_placement_enforce import enforce_gpu_placement

        gate = enforce_gpu_placement(
            policy_id="c6_a10040_spread",
            kube_context=kube_context,
            namespace=namespace,
            dry_run=False,
            protect_list_path=ROOT
            / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_sweep_protect_list_20260908_phase2.json",
            gates_only=True,
            settle_seconds=8,
        )

    receipt = {
        "schema_version": "v4-c6-confirmatory-reshard-v1",
        "resharding_reason": (
            "C7/C8 complete; cluster free. Remaining episodes were draining through ~11 healthy lanes "
            "while 11+ lanes had Complete/Failed sims idle. Re-shard 438 tail across all 32 authorized lanes."
        ),
        "authorized_lane_count": AUTHORIZED_LANE_COUNT,
        "authorized_matrix_cap": (
            "launch_matrix release_boundary: 32 qualified lanes for 768 frozen C6 rows — at cap, not exceeded"
        ),
        "queue_episodes": len(queue_episode_ids),
        "completed_on_pvc": len(completed),
        "remaining_episodes": len(remaining),
        "lanes_dispatched": len(plan),
        "attempt_range": f"attempt{args.start_attempt:04d}-attempt{attempt_index - 1:04d}",
        "max_episodes_per_lane": eps_per_lane,
        "wall_clock_estimate_hours_at_full_lanes": hours_est,
        "prior_tail_dispatch_superseded": "create-c6confirm20260908f-wave-d-tail.json (attempt0226-0257)",
        "plan": plan,
        "dispatch_outcomes": dispatch_outcomes,
        "teardown_jobs": teardown,
        "lock_clear": lock_clear,
        "sim_first_gate": gate.get("actions") if isinstance(gate, dict) else [],
        "render_output_root": str(render_root),
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
