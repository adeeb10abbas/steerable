#!/usr/bin/env python3
"""Re-render and dispatch C6 confirmatory lanes blocked on shared-checkout runner drift."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c6_containment_confirmatory_20260908"
DEFAULT_LANES = ("c6m17", "c6m23", "c6m22", "c6m29")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", default=",".join(DEFAULT_LANES))
    parser.add_argument("--start-attempt-index", type=int, default=195)
    parser.add_argument("--render-root", type=Path, default=EXEC / "rendered-c6confirm20260908f-runnerfix")
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument("--receipt-out", type=Path, default=EXEC / "create-c6confirm20260908f-runnerfix.json")
    args = parser.parse_args(argv)

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

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
    from tools.v4_gpu_placement_enforce import apply_rendered_lane_dir, delete_job, fetch_jobs, LANE_JOB_RE

    kube_context = "prod-dcwi-warrenq1-vmkub007"
    namespace = "211247-prod"
    pvc = "211247-prod-pvc"
    output_parent = "/data/users/ali/vla_wam/raw/v4/c6-containment-main"
    pvc_publisher_pod = "211247-ali-b200-1gpu"
    lane_ids = [item.strip() for item in args.lanes.split(",") if item.strip()]
    render_root = args.render_root
    render_root.mkdir(parents=True, exist_ok=True)

    runtime_lock_path = ROOT / "artifacts/online_correction_v4/setup/containment_c6_confirmatory_runtime_lock.released.json"
    launch_matrix_path = ROOT / "artifacts/online_correction_v4/setup/containment_c6_confirmatory_launch_matrix.json"
    campaign_config_path = ROOT / "docs/online_correction_v4/campaign.json"
    resolved = resolve_released_campaign_bindings(
        runtime_lock_path=runtime_lock_path,
        repo_root=ROOT,
        campaign_config_path=campaign_config_path,
        queue_path=ROOT / "artifacts/online_correction_v4/setup/c6_confirmatory/queue.frozen.cosmos3_nano.jsonl",
        queue_manifest_path=ROOT / "artifacts/online_correction_v4/setup/c6_confirmatory/queue_manifest.frozen.cosmos3_nano.json",
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
        pvc_publisher_pod=pvc_publisher_pod,
    )

    source_manifests = {
        "c6m17": EXEC / "rendered-c6confirm20260908f-retry/c6m17-attempt0193-a10040-policy_b200-simulator/.bindings/lane_dispatch_manifest.json",
        "c6m23": EXEC / "rendered-c6confirm20260908f-retry/c6m23-attempt0194-a10040-policy_b200-simulator/.bindings/lane_dispatch_manifest.json",
    }
    cluster_episode_source = {
        "c6m22": ("c6m22", "attempt0183"),
        "c6m29": ("c6m29", "attempt0190"),
    }

    lane_attempt_map: dict[str, str] = {}
    render_outcomes: list[dict] = []
    attempt_index = args.start_attempt_index
    for lane_id in lane_ids:
        attempt_id = f"attempt{attempt_index:04d}"
        attempt_index += 1
        lane_attempt_map[lane_id] = attempt_id
        if lane_id in source_manifests:
            episode_ids = json.loads(source_manifests[lane_id].read_text(encoding="utf-8"))["episode_ids"]
        else:
            src_lane, src_attempt = cluster_episode_source[lane_id]
            proc = subprocess.run(
                [
                    "kubectl",
                    "exec",
                    "-n",
                    namespace,
                    "--context",
                    kube_context,
                    pvc_publisher_pod,
                    "--",
                    "python3",
                    "-c",
                    (
                        "import json; "
                        f"p='/data/users/ali/vla_wam/raw/v4/c6-containment-main/.coord-bindings/{src_lane}/{src_attempt}/lane_dispatch_manifest.json'; "
                        "print(json.dumps(json.load(open(p))['episode_ids']))"
                    ),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            episode_ids = json.loads(proc.stdout.strip())

        lane = lanes_by_id[lane_id]
        rows = [registry.get(episode_id) for episode_id in episode_ids]
        policy_id, group_fixture = rows[0].execution_group.split(":", 1)
        execution_group = ExecutionGroup(
            group_id=rows[0].execution_group,
            policy=policy_id,
            fixture=group_fixture,
            rows=rows,
        )
        bundle_root = render_root / f"{lane_id}-{attempt_id}-{lane.hardware_stratum}"
        if bundle_root.exists():
            raise SystemExit(f"refusing to overwrite bundle: {bundle_root}")
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
        manifest = json.loads((local_binding_root / "lane_dispatch_manifest.json").read_text(encoding="utf-8"))
        render_outcomes.append(
            {
                "lane_id": lane_id,
                "attempt_id": attempt_id,
                "bundle_root": str(bundle_root),
                "runner_sha256": manifest["runner_sha256"],
                "episode_count": len(episode_ids),
            }
        )

    dispatch_outcomes: list[dict] = []
    if args.dispatch:
        for lane_id in lane_ids:
            for job in fetch_jobs(kube_context=kube_context, namespace=namespace):
                name = str(job.get("metadata", {}).get("name") or "")
                match = LANE_JOB_RE.match(name)
                if match and match.group("lane") == lane_id:
                    delete_job(name=name, kube_context=kube_context, namespace=namespace, dry_run=False)

        for lane_id in lane_ids:
            attempt_id = lane_attempt_map[lane_id]
            matches = sorted(render_root.glob(f"{lane_id}-{attempt_id}-*"))
            if not matches:
                dispatch_outcomes.append({"lane_id": lane_id, "attempt_id": attempt_id, "status": "missing_bundle"})
                continue
            bundle_root = matches[0]
            publish_staged_bindings_to_pvc(
                local_binding_root=bundle_root / ".bindings",
                pvc_binding_root=pvc_binding_root_path(output_parent, lane_id, attempt_id),
                cluster=cluster,
            )
            for resource in ("configmap.yaml", "scripts-configmap.yaml", "policy-service.yaml"):
                path = bundle_root / resource
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
            job_actions = apply_rendered_lane_dir(
                lane_dir=bundle_root,
                policy_id="c6_a10040_spread",
                kube_context=kube_context,
                namespace=namespace,
                dry_run=False,
                protect_list={"protected_lane_ids": []},
            )
            dispatch_outcomes.append(
                {
                    "lane_id": lane_id,
                    "attempt_id": attempt_id,
                    "status": "dispatched",
                    "bundle_root": str(bundle_root),
                    "job_actions": job_actions,
                    "teardown_inventory": parse_k8s_objects(bundle_root, kube_context=kube_context).as_dict(),
                }
            )

    receipt = {
        "schema_version": "v4-c6-shared-checkout-runnerfix-rerender-v1",
        "root_cause": "C6 retry lanes m17/m23 rendered against stale af01 lock copy while shared checkout holds C7-pinned 71e7 runner",
        "released_runner_sha256": lock.runner_sha256,
        "render_output_root": str(render_root),
        "lane_attempt_map": lane_attempt_map,
        "render_outcomes": render_outcomes,
        "dispatch_outcomes": dispatch_outcomes,
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
