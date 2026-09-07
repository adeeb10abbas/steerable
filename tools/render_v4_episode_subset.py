#!/usr/bin/env python3
"""Render fresh one-episode V4 lanes for explicit infrastructure retries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.coordinator import (  # noqa: E402
    ClusterBinding,
    build_lane_spec,
    pvc_binding_root_path,
    render_lane_bundle,
    validate_rendered_bundle_scope,
)
from experiments.online_correction_v4.droid_contract import (  # noqa: E402
    sha256_file,
    validate_runtime_lock,
)
from experiments.online_correction_v4.registry import (  # noqa: E402
    CampaignRegistry,
    ExecutionGroup,
)


def render_subset(
    *,
    runtime_lock_path: Path,
    queue_path: Path,
    campaign_config_path: Path,
    lane_template_path: Path,
    episode_ids: list[str],
    render_root: Path,
    cluster: ClusterBinding,
    attempt_index: int,
) -> dict:
    if len(set(episode_ids)) != len(episode_ids) or not episode_ids:
        raise ValueError("episode retry IDs must be nonempty and unique")
    lock = validate_runtime_lock(
        runtime_lock_path,
        expected_queue_sha256=sha256_file(queue_path),
        expected_config_sha256=sha256_file(campaign_config_path),
    )
    if not (lock.is_released or lock.is_pilot_released):
        raise ValueError("runtime lock is not released")
    registry = CampaignRegistry.from_manifest_path(queue_path)
    template = json.loads(lane_template_path.read_text(encoding="utf-8"))
    if not isinstance(template, dict):
        raise ValueError("lane template must be a JSON object")
    render_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for offset, episode_id in enumerate(episode_ids):
        if episode_id not in registry.by_episode_id:
            raise ValueError(f"retry episode is outside frozen queue: {episode_id}")
        manifest_row = registry.get(episode_id)
        if manifest_row.family not in lock.released_families:
            raise ValueError(f"retry episode family is not released: {episode_id}")
        policy_id, group_fixture = manifest_row.execution_group.split(":", 1)
        lane_id = f"v4r{offset:02d}"
        attempt_id = f"attempt{attempt_index + offset:04d}"
        bundle_root = render_root / f"{lane_id}-{attempt_id}-explicit-retry"
        bundle_root.mkdir()
        local_binding_root = bundle_root / ".bindings"
        pvc_root = pvc_binding_root_path(
            cluster.output_parent,
            lane_id,
            attempt_id,
        )
        execution_group = ExecutionGroup(
            group_id=manifest_row.execution_group,
            policy=policy_id,
            fixture=group_fixture,
            rows=[manifest_row],
        )
        spec = build_lane_spec(
            template=template,
            cluster=cluster,
            lane_id=lane_id,
            attempt_id=attempt_id,
            lock=lock,
            assignment_groups=[execution_group],
            remaining_episode_ids=[episode_id],
            qualification_only=False,
            queue_path=queue_path,
            runtime_lock_path=runtime_lock_path,
            campaign_config_path=campaign_config_path,
            repo_root=ROOT,
            local_binding_root=local_binding_root,
            pvc_binding_root=pvc_root,
        )
        spec_path = (
            lane_template_path.parent
            / f".retry-{lane_id}-{attempt_id}-render-spec.json"
        )
        if spec_path.exists():
            raise FileExistsError(spec_path)
        try:
            spec_path.write_text(
                json.dumps(spec, allow_nan=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            render_lane_bundle(
                spec_path=spec_path,
                output_root=bundle_root,
            )
            validate_rendered_bundle_scope(bundle_root, behavioral=True)
        finally:
            spec_path.unlink(missing_ok=True)
        rows.append(
            {
                "episode_id": episode_id,
                "lane_id": lane_id,
                "attempt_id": attempt_id,
                "bundle_root": str(bundle_root),
                "local_binding_root": str(local_binding_root),
                "pvc_binding_root": pvc_root,
            }
        )
    receipt = {
        "schema_version": "v4-explicit-episode-retry-render-v1",
        "campaign_id": lock.campaign_id,
        "runtime_lock_sha256": sha256_file(runtime_lock_path),
        "manifest_sha256": sha256_file(queue_path),
        "behavioral_episode_count": len(rows),
        "rows": rows,
    }
    (render_root / "retry-render-receipt.json").write_text(
        json.dumps(receipt, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--lane-template", type=Path, required=True)
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--kube-context", required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--pvc", required=True)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--attempt-index", type=int, required=True)
    args = parser.parse_args()
    receipt = render_subset(
        runtime_lock_path=args.runtime_lock.resolve(),
        queue_path=args.queue.resolve(),
        campaign_config_path=args.campaign_config.resolve(),
        lane_template_path=args.lane_template.resolve(),
        episode_ids=args.episode_id,
        render_root=args.render_root.resolve(),
        cluster=ClusterBinding(
            kube_context=args.kube_context,
            namespace=args.namespace,
            pvc=args.pvc,
            output_parent=args.output_parent,
        ),
        attempt_index=args.attempt_index,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
