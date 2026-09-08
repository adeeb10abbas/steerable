#!/usr/bin/env python3
"""Re-shard C7 released-tail episodes across more lane identities with fresh attempts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXEC = ROOT / "artifacts/online_correction_v4/execution/c7_object_pair_20260906"
R5_ATTEMPTS = ("attempt0607", "attempt0608", "attempt0609", "attempt0610")
DEFAULT_LANES = (
    "c7m03",
    "c7m04",
    "c7m06",
    "c7m07",
    "c7m08",
    "c7m09",
    "c7m10",
    "c7m11",
    "c7m12",
    "c7m13",
    "c7m14",
    "c7m15",
    "c7m16",
    "c7m18",
)


def _load_missing_episode_ids(*, accepted_ledger: Path, queue_path: Path) -> list[str]:
    accepted: set[str] = set()
    for line in accepted_ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            accepted.add(json.loads(line)["episode_id"])
    missing: list[str] = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("family") != "C7" or row.get("cohort") != "confirmatory":
            continue
        episode_id = str(row["episode_id"])
        if episode_id not in accepted:
            missing.append(episode_id)
    return missing


def _fetch_r5_completed_episode_ids(
    *,
    kube_context: str,
    namespace: str,
    publisher_pod: str,
) -> set[str]:
    script = r"""
import json
from pathlib import Path
root = Path("/data/users/ali/vla_wam/raw/v4/c7-object-pair-main")
completed = set()
for att in ("0607", "0608", "0609", "0610"):
    for complete in root.rglob(f"*attempt{att}*/COMPLETE.json"):
        episode_json = complete.parent / "episode.json"
        if episode_json.is_file():
            completed.add(json.loads(episode_json.read_text())["episode_id"])
print(json.dumps(sorted(completed)))
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lanes", default=",".join(DEFAULT_LANES))
    parser.add_argument("--start-attempt-index", type=int, default=611)
    parser.add_argument(
        "--render-root",
        type=Path,
        default=EXEC / "rendered-released-tail-reshard-20260908r6",
    )
    parser.add_argument("--dispatch", action="store_true")
    parser.add_argument(
        "--receipt-out",
        type=Path,
        default=EXEC / "create-released-tail-reshard-20260908r6.json",
    )
    parser.add_argument(
        "--accepted-ledger",
        type=Path,
        default=EXEC / "pvc-ledgers/compiled_ledger_20260908m/accepted_ledger.jsonl",
    )
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
    output_parent = "/data/users/ali/vla_wam/raw/v4/c7-object-pair-main"
    pvc_publisher_pod = "211247-ali-b200-1gpu"
    lane_ids = [item.strip() for item in args.lanes.split(",") if item.strip()]
    render_root = args.render_root

    runtime_lock_path = ROOT / "artifacts/online_correction_v4/setup/object_pair_c7_confirmatory_runtime_lock.released.json"
    launch_matrix_path = ROOT / "artifacts/online_correction_v4/setup/object_pair_c7_confirmatory_launch_matrix.released.json"
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
    missing_all = _load_missing_episode_ids(
        accepted_ledger=args.accepted_ledger,
        queue_path=resolved.queue_path,
    )
    completed_r5 = _fetch_r5_completed_episode_ids(
        kube_context=kube_context,
        namespace=namespace,
        publisher_pod=pvc_publisher_pod,
    )
    remaining = [episode_id for episode_id in missing_all if episode_id not in completed_r5]
    if len(missing_all) != 188:
        raise SystemExit(f"expected 188 missing episodes, got {len(missing_all)}")
    if not remaining:
        raise SystemExit("no remaining episodes to re-shard")

    registry = CampaignRegistry.from_manifest_path(resolved.queue_path)
    rows = [registry.get(episode_id) for episode_id in remaining]
    policy_id, group_fixture = rows[0].execution_group.split(":", 1)
    execution_group = ExecutionGroup(
        group_id=rows[0].execution_group,
        policy=policy_id,
        fixture=group_fixture,
        rows=rows,
    )

    launch_matrix = load_json(launch_matrix_path, "launch matrix")
    lanes_by_id = {lane.lane_id: lane for lane in load_qualified_lanes(launch_matrix, repo_root=ROOT)}
    selected_lanes = [lanes_by_id[lane_id] for lane_id in lane_ids]
    buckets = shard_group_units([(execution_group, remaining)], selected_lanes)

    cluster = ClusterBinding(
        kube_context=kube_context,
        namespace=namespace,
        pvc=pvc,
        output_parent=output_parent,
        pvc_publisher_pod=pvc_publisher_pod,
    )

    render_root.mkdir(parents=True, exist_ok=True)
    lane_attempt_map: dict[str, str] = {}
    render_outcomes: list[dict] = []
    attempt_index = args.start_attempt_index
    for lane_id in lane_ids:
        units = buckets.get(lane_id) or []
        episode_ids: list[str] = []
        for _group, ids in units:
            episode_ids.extend(ids)
        if not episode_ids:
            continue
        attempt_id = f"attempt{attempt_index:04d}"
        attempt_index += 1
        lane_attempt_map[lane_id] = attempt_id
        lane = lanes_by_id[lane_id]
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
                "episode_count": len(episode_ids),
                "runner_sha256": manifest["runner_sha256"],
                "bundle_root": str(bundle_root),
            }
        )

    dispatch_outcomes: list[dict] = []
    teardown: list[str] = []

    def _apply_lane_jobs(bundle_root: Path) -> list[dict]:
        import yaml

        actions: list[dict] = []
        for job_file, suspend in (("simulator-job.yaml", False), ("policy-job.yaml", True)):
            path = bundle_root / job_file
            doc = yaml.safe_load(path.read_text(encoding="utf-8"))
            if suspend:
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
            actions.append(
                {
                    "job": str(doc.get("metadata", {}).get("name") or path.name),
                    "role": "simulator" if "sim" in job_file else "policy",
                    "applied": ok,
                    "message": (completed.stderr or completed.stdout or "").strip(),
                }
            )
            if not ok:
                raise SystemExit(f"kubectl create failed for {path}: {completed.stderr}")
        return actions

    if args.dispatch:
        teardown: list[str] = []
        for job in fetch_jobs(kube_context=kube_context, namespace=namespace):
            name = str(job.get("metadata", {}).get("name") or "")
            match = LANE_JOB_RE.match(name)
            if not match:
                continue
            if str(match.group("attempt")) in R5_ATTEMPTS and str(match.group("lane")).startswith("c7m"):
                if delete_job(name=name, kube_context=kube_context, namespace=namespace, dry_run=False):
                    teardown.append(name)
        for lane_id in sorted(lane_attempt_map):
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
            job_actions = _apply_lane_jobs(bundle_root)
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

    minutes_per_lane = 25.0
    lane_count = len(lane_attempt_map)
    hours_est = round((len(remaining) / max(lane_count, 1)) * (minutes_per_lane / 60.0), 1)
    receipt = {
        "schema_version": "v4-c7-released-tail-reshard-v1",
        "resharding_allowed": True,
        "resharding_reason": (
            "Frozen queue rows carry episode_id/block_id/policy_seed/env_seed only; "
            "lane_id is dispatch-time via coordinator.shard_group_units, not manifest-bound."
        ),
        "supersedes_dispatch": "create-released-shards-20260908r5.json",
        "teardown_r5_attempts": list(R5_ATTEMPTS),
        "completed_r5_episode_ids": sorted(completed_r5),
        "completed_r5_count": len(completed_r5),
        "remaining_episode_count": len(remaining),
        "lane_count": lane_count,
        "lane_attempt_map": lane_attempt_map,
        "a40_capacity_split": {
            "a40_pool_size": 40,
            "c7_lane_pairs": lane_count,
            "c7_a40_gpus": lane_count,
            "reserved_for_c8_lane_pairs_max": (40 - lane_count) // 2,
            "note": "C8 confirmatory uses 2×A40 per pair; leave headroom for C8 scale toward 20 pairs.",
        },
        "wall_clock_estimate_hours": hours_est,
        "render_output_root": str(render_root),
        "render_outcomes": render_outcomes,
        "dispatch_outcomes": dispatch_outcomes,
        "teardown_jobs": teardown if args.dispatch else [],
    }
    args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
