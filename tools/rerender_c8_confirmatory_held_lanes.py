#!/usr/bin/env python3
"""Re-render and dispatch C8 confirmatory held lanes with fresh immutable attempt IDs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c8_second_stack_20260908"
PROTECT = frozenset({"c8m05", "c8m16"})
SKIP_COMPLETE = frozenset({"c8m13", "c8m14"})


def _load_completed_episode_ids(ledger_path: Path) -> set[str]:
    completed: set[str] = set()
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            completed.add(json.loads(line)["episode_id"])
    return completed


def _lane_attempt_map() -> dict[str, str]:
    create = json.loads(
        (EXEC / "create-confirmatory-c8main20260908a.json").read_text(encoding="utf-8")
    )
    completed = _load_completed_episode_ids(
        EXEC / "confirmatory-ledger-20260908e/accepted_ledger.jsonl"
    )
    mapping: dict[str, str] = {}
    attempt_index = 21
    for assignment in create["lane_assignments"]:
        lane_id = assignment["lane_id"]
        if lane_id in PROTECT or lane_id in SKIP_COMPLETE:
            continue
        mapping[lane_id] = f"attempt{attempt_index:04d}"
        attempt_index += 1
    return mapping


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--dispatch-only", action="store_true")
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
        sha256_file,
        validate_rendered_bundle_scope,
        validate_runtime_lock,
        verify_queue_binding,
        pvc_binding_root_path,
    )
    from experiments.online_correction_v4.registry import CampaignRegistry
    from tools.v4_gpu_placement_enforce import apply_rendered_lane_dir, delete_job, fetch_jobs, LANE_JOB_RE

    kube_context = "prod-dcwi-warrenq1-vmkub007"
    namespace = "211247-prod"
    pvc = "211247-prod-pvc"
    output_parent = "/data/users/ali/vla_wam/raw/v4/c8-second-stack-main"
    pvc_publisher_pod = "211247-ali-b200-1gpu"
    render_root = EXEC / "rendered-confirmatory-r2"
    protect_list_path = (
        ROOT
        / "artifacts/online_correction_v4/execution/gpu_widen_20260908/gpu_sweep_protect_list_20260908_phase2.json"
    )
    protect_list = json.loads(protect_list_path.read_text(encoding="utf-8"))

    runtime_lock_path = ROOT / "artifacts/online_correction_v4/setup/c8_confirmatory/runtime_lock.released.json"
    launch_matrix_path = ROOT / "artifacts/online_correction_v4/setup/c8_confirmatory/launch_matrix.released.json"
    campaign_config_path = ROOT / "docs/online_correction_v4/campaign.json"
    repo_root = ROOT

    resolved = resolve_released_campaign_bindings(
        runtime_lock_path=runtime_lock_path,
        repo_root=repo_root,
        campaign_config_path=campaign_config_path,
        queue_path=ROOT / "artifacts/online_correction_v4/setup/c8_confirmatory/queue.frozen.jsonl",
        queue_manifest_path=ROOT / "artifacts/online_correction_v4/setup/c8_confirmatory/queue_manifest.json",
    )
    queue_path = resolved.queue_path
    queue_manifest_path = resolved.queue_manifest_path

    lock = validate_runtime_lock(runtime_lock_path)
    verify_queue_binding(
        queue_path=queue_path,
        queue_manifest_path=queue_manifest_path,
        expected_manifest_sha256=lock.manifest_sha256,
    )
    registry = CampaignRegistry.from_manifest_path(queue_path)
    launch_matrix = load_json(launch_matrix_path, "launch matrix")
    lanes_by_id = {lane.lane_id: lane for lane in load_qualified_lanes(launch_matrix, repo_root=repo_root)}

    create = json.loads((EXEC / "create-confirmatory-c8main20260908a.json").read_text(encoding="utf-8"))
    completed = _load_completed_episode_ids(
        EXEC / "confirmatory-ledger-20260908e/accepted_ledger.jsonl"
    )
    lane_attempt_map = _lane_attempt_map()
    cluster = ClusterBinding(
        kube_context=kube_context,
        namespace=namespace,
        pvc=pvc,
        output_parent=output_parent,
        pvc_publisher_pod=pvc_publisher_pod,
    )

    outcomes: list[dict] = []
    if not args.dispatch_only:
        render_root.mkdir(parents=True, exist_ok=True)
        for assignment in create["lane_assignments"]:
            lane_id = assignment["lane_id"]
            if lane_id not in lane_attempt_map:
                continue
            attempt_id = lane_attempt_map[lane_id]
            lane = lanes_by_id[lane_id]
            remaining = [
                episode_id
                for episode_id in assignment["remaining_episode_ids"]
                if episode_id not in completed
            ]
            if not remaining:
                outcomes.append(
                    {
                        "lane_id": lane_id,
                        "attempt_id": attempt_id,
                        "status": "skipped_no_remaining",
                        "remaining_episode_count": 0,
                    }
                )
                continue
            groups = [registry.by_execution_group[group_id] for group_id in assignment["group_ids"]]
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
                assignment_groups=groups,
                remaining_episode_ids=remaining,
                qualification_only=False,
                queue_path=queue_path,
                runtime_lock_path=runtime_lock_path,
                campaign_config_path=campaign_config_path,
                repo_root=repo_root,
                local_binding_root=local_binding_root,
                pvc_binding_root=pvc_root,
            )
            spec_path = lane.template_root / f".coord-{lane_id}-{attempt_id}-render-spec.json"
            try:
                spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                launch_hash = sha256_file(spec_path)
                render_lane_bundle(spec_path=spec_path, output_root=bundle_root)
                validate_rendered_bundle_scope(bundle_root, behavioral=True)
            finally:
                spec_path.unlink(missing_ok=True)
            outcomes.append(
                {
                    "lane_id": lane_id,
                    "attempt_id": attempt_id,
                    "prior_attempt_id": assignment["attempt_id"],
                    "status": "rendered",
                    "remaining_episode_count": len(remaining),
                    "bundle_root": str(bundle_root),
                    "launch_hash": launch_hash,
                    "pvc_binding_root": pvc_root,
                }
            )

    if args.render_only:
        print(json.dumps({"lane_outcomes": outcomes}, indent=2))
        return 0

    held_lane_ids = set(lane_attempt_map)
    for job in fetch_jobs(kube_context=kube_context, namespace=namespace):
        name = str(job.get("metadata", {}).get("name") or "")
        match = LANE_JOB_RE.match(name)
        if match and match.group("lane") in held_lane_ids:
            delete_job(name=name, kube_context=kube_context, namespace=namespace, dry_run=False)

    dispatch_outcomes: list[dict] = []
    for lane_id in sorted(lane_attempt_map):
        attempt_id = lane_attempt_map[lane_id]
        matches = sorted(render_root.glob(f"{lane_id}-{attempt_id}-*"))
        if not matches:
            dispatch_outcomes.append({"lane_id": lane_id, "attempt_id": attempt_id, "status": "missing_bundle"})
            continue
        bundle_root = matches[0]
        local_binding_root = bundle_root / ".bindings"
        pvc_root = pvc_binding_root_path(output_parent, lane_id, attempt_id)
        publish_staged_bindings_to_pvc(
            local_binding_root=local_binding_root,
            pvc_binding_root=pvc_root,
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
            policy_id="c8_a40_spread",
            kube_context=kube_context,
            namespace=namespace,
            dry_run=False,
            protect_list=protect_list,
        )
        inventory = parse_k8s_objects(bundle_root, kube_context=kube_context)
        dispatch_outcomes.append(
            {
                "lane_id": lane_id,
                "attempt_id": attempt_id,
                "status": "dispatched",
                "bundle_root": str(bundle_root),
                "job_actions": job_actions,
                "teardown_inventory": inventory.as_dict(),
            }
        )

    receipt = {
        "schema_version": "v4-c8-fresh-attempt-rerender-dispatch-v1",
        "render_output_root": str(render_root),
        "lane_attempt_map": lane_attempt_map,
        "protected_lanes": sorted(PROTECT),
        "complete_lanes_skipped": sorted(SKIP_COMPLETE),
        "render_outcomes": outcomes,
        "dispatch_outcomes": dispatch_outcomes,
    }
    receipt_path = EXEC / "create-confirmatory-fresh-attempt-20260908r2.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
