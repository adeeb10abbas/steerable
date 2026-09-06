#!/usr/bin/env python3
"""Plan or create one coordinated V4 campaign dispatch wave.

Dry-run renders immutable lane bundles locally without touching a cluster.
``--create`` requires explicit kube context, namespace, PVC, and output parent,
and is the only mode that invokes ``kubectl create``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
DEFAULT_QUEUE_MANIFEST = ROOT / "artifacts/online_correction_v4/queue_manifest.json"
DEFAULT_LAUNCH_MATRIX = ROOT / "artifacts/online_correction_v4/launch_matrix.json"
DEFAULT_RUNTIME_LOCK = ROOT / "docs/online_correction_v4/runtime_lock.template.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or create a dependency-aware V4 campaign dispatch wave with "
            "fresh immutable lane bundles."
        )
    )
    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=DEFAULT_RUNTIME_LOCK,
        help="Absolute path to the released runtime lock JSON.",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help="Absolute path to the frozen queue.jsonl.",
    )
    parser.add_argument(
        "--queue-manifest",
        type=Path,
        default=DEFAULT_QUEUE_MANIFEST,
        help="Absolute path to queue_manifest.json.",
    )
    parser.add_argument(
        "--launch-matrix",
        type=Path,
        default=DEFAULT_LAUNCH_MATRIX,
        help="Absolute path to launch_matrix.json.",
    )
    parser.add_argument(
        "--campaign-config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Campaign config used to bind config_sha256.",
    )
    parser.add_argument(
        "--group-receipts-dir",
        type=Path,
        default=None,
        help="Directory of accepted *.group_receipt.json files for resume.",
    )
    parser.add_argument(
        "--coordination-state",
        type=Path,
        default=None,
        help=(
            "Optional coordination-state JSON with lane/episode infra failures, "
            "reserved attempt IDs, and attempt index."
        ),
    )
    parser.add_argument(
        "--group-lease-root",
        type=Path,
        default=None,
        help="Durable root for exclusive group lease files (required for --create).",
    )
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Optional durable evidence root for storage budget accounting.",
    )
    parser.add_argument(
        "--render-output-root",
        type=Path,
        default=None,
        help="Parent directory for freshly rendered immutable lane bundles.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the dispatch plan and optionally render bundles; never call kubectl.",
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Render bundles and kubectl create them (requires explicit cluster binding).",
    )
    parser.add_argument(
        "--qualification-only",
        action="store_true",
        help=(
            "Render/create infrastructure qualification lanes only. "
            "Reports zero behavioral episodes and keeps /usr/bin/true simulator argv."
        ),
    )
    parser.add_argument(
        "--kube-context",
        default=None,
        help="Authorized kube context (required for --create and bundle rendering).",
    )
    parser.add_argument(
        "--namespace",
        default=None,
        help="Target namespace (required for --create and bundle rendering).",
    )
    parser.add_argument(
        "--pvc",
        default=None,
        help="PersistentVolumeClaim name (required for --create and bundle rendering).",
    )
    parser.add_argument(
        "--output-parent",
        default=None,
        help="PVC-mounted output parent directory (required for --create and rendering).",
    )
    parser.add_argument(
        "--pvc-publisher-pod",
        default=None,
        help=(
            "Existing pod with the PVC mounted; when set, write-once dispatch "
            "bindings are published through kubectl instead of a local mount."
        ),
    )
    parser.add_argument(
        "--attempt-index",
        type=int,
        default=1,
        help="Base attempt index encoded into lane attempt IDs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.create and args.dry_run:
        print("[V4 coordinator] blocked: --create and --dry-run are mutually exclusive", file=sys.stderr)
        return 2
    if not args.create and not args.dry_run:
        args.dry_run = True
    if args.create and not args.qualification_only and args.group_lease_root is None:
        print(
            "[V4 coordinator] blocked: behavioral --create requires --group-lease-root "
            "for durable group leases",
            file=sys.stderr,
        )
        return 2

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from experiments.online_correction_v4.coordinator import (
        ClusterBinding,
        CoordinatorBlockedError,
        CoordinatorError,
        CoordinatorInputs,
        plan_campaign,
    )

    cluster_binding = None
    render_root = args.render_output_root.resolve() if args.render_output_root else None
    needs_binding = args.create or render_root is not None
    if needs_binding:
        if not all([args.kube_context, args.namespace, args.pvc, args.output_parent]):
            print(
                "[V4 coordinator] blocked: --kube-context, --namespace, --pvc, and "
                "--output-parent are required when rendering or creating lanes",
                file=sys.stderr,
            )
            return 2
        cluster_binding = ClusterBinding(
            kube_context=str(args.kube_context),
            namespace=str(args.namespace),
            pvc=str(args.pvc),
            output_parent=str(args.output_parent),
            pvc_publisher_pod=args.pvc_publisher_pod,
        )

    inputs = CoordinatorInputs(
        runtime_lock_path=args.runtime_lock.resolve(),
        queue_path=args.queue.resolve(),
        queue_manifest_path=args.queue_manifest.resolve(),
        launch_matrix_path=args.launch_matrix.resolve(),
        campaign_config_path=args.campaign_config.resolve(),
        group_receipts_dir=args.group_receipts_dir.resolve() if args.group_receipts_dir else None,
        coordination_state_path=args.coordination_state.resolve() if args.coordination_state else None,
        group_lease_root=args.group_lease_root.resolve() if args.group_lease_root else None,
        evidence_root=args.evidence_root.resolve() if args.evidence_root else None,
        render_output_root=render_root,
        cluster_binding=cluster_binding,
        attempt_index=args.attempt_index,
        qualification_only=args.qualification_only,
        repo_root=ROOT,
    )

    try:
        plan = plan_campaign(
            inputs,
            render_bundles=render_root is not None,
            create_on_cluster=args.create,
        )
    except CoordinatorBlockedError as exc:
        print(f"[V4 coordinator] blocked: {exc}", file=sys.stderr)
        return 2
    except CoordinatorError as exc:
        print(f"[V4 coordinator] error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
