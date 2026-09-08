#!/usr/bin/env python3
"""Dispatch a single remaining C7 confirmatory episode on one lane."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c7_object_pair_20260906"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--attempt-id", default="attempt0625")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--render-root",
        type=Path,
        default=EXEC / "rendered-c7-final-tail-20260908",
    )
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=EXEC / "create-c7-final-tail-20260908.json",
    )
    args = parser.parse_args(argv)

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

    kube_context = "prod-dcwi-warrenq1-vmkub007"
    namespace = "211247-prod"
    pvc = "211247-prod-pvc"
    output_parent = "/data/users/ali/vla_wam/raw/v4/c7-object-pair-main"

    runtime_lock_path = (
        ROOT / "artifacts/online_correction_v4/setup/object_pair_c7_confirmatory_runtime_lock.released.json"
    )
    launch_matrix_path = (
        ROOT / "artifacts/online_correction_v4/setup/object_pair_c7_confirmatory_launch_matrix.released.json"
    )
    campaign_config_path = ROOT / "docs/online_correction_v4/campaign.json"
    queue_path = ROOT / "artifacts/online_correction_v4/setup/c7_confirmatory/queue.frozen.jsonl"
    queue_manifest_path = ROOT / "artifacts/online_correction_v4/setup/c7_confirmatory/queue_manifest.frozen.json"

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
    row = registry.get(args.episode_id)
    policy_id, group_fixture = row.execution_group.split(":", 1)
    execution_group = ExecutionGroup(
        group_id=row.execution_group,
        policy=policy_id,
        fixture=group_fixture,
        rows=[row],
    )
    launch_matrix = load_json(launch_matrix_path, "launch matrix")
    lane = {lane.lane_id: lane for lane in load_qualified_lanes(launch_matrix, repo_root=ROOT)}[args.lane_id]
    cluster = ClusterBinding(
        kube_context=kube_context,
        namespace=namespace,
        pvc=pvc,
        output_parent=output_parent,
        pvc_publisher_pod="211247-ali-b200-1gpu",
    )

    render_root = args.render_root
    render_root.mkdir(parents=True, exist_ok=True)
    bundle_root = render_root / f"{args.lane_id}-{args.attempt_id}-{lane.hardware_stratum}"
    if bundle_root.exists():
        import shutil

        shutil.rmtree(bundle_root)
    bundle_root.mkdir(parents=True)
    local_binding_root = bundle_root / ".bindings"
    pvc_root = pvc_binding_root_path(output_parent, args.lane_id, args.attempt_id)
    spec = build_lane_spec(
        template=lane.spec_template,
        cluster=cluster,
        lane_id=args.lane_id,
        attempt_id=args.attempt_id,
        lock=lock,
        assignment_groups=[execution_group],
        remaining_episode_ids=[args.episode_id],
        qualification_only=False,
        queue_path=resolved.queue_path,
        runtime_lock_path=runtime_lock_path,
        campaign_config_path=resolved.campaign_config_path,
        repo_root=ROOT,
        local_binding_root=local_binding_root,
        pvc_binding_root=pvc_root,
    )
    spec_path = lane.template_root / f".coord-{args.lane_id}-{args.attempt_id}-render-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        render_lane_bundle(spec_path=spec_path, output_root=bundle_root)
        validate_rendered_bundle_scope(bundle_root, behavioral=True)
    finally:
        spec_path.unlink(missing_ok=True)

    dispatch_outcome: dict = {"status": "rendered_only"}
    if args.dispatch:
        publish_staged_bindings_to_pvc(
            local_binding_root=local_binding_root,
            pvc_binding_root=pvc_root,
            cluster=cluster,
        )
        for resource in ("configmap.yaml", "scripts-configmap.yaml", "policy-service.yaml"):
            path = bundle_root / resource
            result = subprocess.run(
                ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", str(path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0 and "AlreadyExists" not in (result.stderr or ""):
                raise SystemExit(f"kubectl create failed for {path}: {result.stderr}")

        job_actions: list[dict] = []
        for job_file, suspend in (("simulator-job.yaml", False), ("policy-job.yaml", True)):
            doc = yaml.safe_load((bundle_root / job_file).read_text(encoding="utf-8"))
            if suspend:
                doc.setdefault("spec", {})["suspend"] = True
            completed = subprocess.run(
                ["kubectl", "--context", kube_context, "-n", namespace, "create", "-f", "-"],
                input=yaml.safe_dump(doc, sort_keys=False),
                capture_output=True,
                text=True,
                check=False,
            )
            ok = completed.returncode == 0 or "AlreadyExists" in (completed.stderr or "")
            job_actions.append(
                {
                    "job": str(doc.get("metadata", {}).get("name") or job_file),
                    "applied": ok,
                    "message": (completed.stderr or completed.stdout or "").strip()[:200],
                }
            )
        from tools.run_v4_gpu_periodic_enforcement import enforce_c7_sim_first_gate

        gate = enforce_c7_sim_first_gate(kube_context=kube_context, namespace=namespace, dry_run=False)
        dispatch_outcome = {
            "status": "dispatched",
            "job_actions": job_actions,
            "c7_gate_actions": gate,
            "teardown_inventory": parse_k8s_objects(bundle_root, kube_context=kube_context).as_dict(),
        }

    receipt = {
        "schema_version": "v4-c7-final-tail-dispatch-v1",
        "lane_id": args.lane_id,
        "attempt_id": args.attempt_id,
        "episode_id": args.episode_id,
        "bundle_root": str(bundle_root),
        "dispatch_outcome": dispatch_outcome,
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
