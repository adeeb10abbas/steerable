#!/usr/bin/env python3
"""Reclaim idle GPU capacity and arbitrate C7 vs G7 pilot lane concurrency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_v4_gpu_periodic_enforcement as gpu_periodic  # noqa: E402
import v4_capacity_arbitration as arbitration  # noqa: E402
import v4_c7_capacity_sequencing as c7_sequencing  # noqa: E402
import v4_gpu_placement_enforce as gpu_placement  # noqa: E402
import v4_gpu_pool_sweep as gpu_sweep  # noqa: E402
import v4_gpu_scheduling as gpu_scheduling  # noqa: E402

DEFAULT_RECEIPT = ROOT / "artifacts/online_correction_v4/execution/gpu_widen_20260908/capacity_arbitration_receipt.json"
DEFAULT_SWEEP_RECEIPT = gpu_sweep.DEFAULT_RECEIPT
DEFAULT_PLACEMENT_RECEIPT = gpu_placement.DEFAULT_RECEIPT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=(
            "pilot_priority",
            "c7_restore",
            "normal",
            "gpu_sweep",
            "gpu_placement",
            "gpu_periodic",
            "c7_sequencing",
        ),
        default=None,
    )
    parser.add_argument("--c7-ceiling", type=int, default=None)
    parser.add_argument("--c7-remaining", type=int, default=188)
    parser.add_argument("--c7-tail-lanes", type=int, default=4)
    parser.add_argument("--c8-remaining", type=int, default=564)
    parser.add_argument(
        "--policy",
        choices=tuple(gpu_scheduling.PLACEMENT_POLICIES),
        default="c8_a40_spread",
    )
    parser.add_argument("--protect-list", type=Path, default=gpu_scheduling.DEFAULT_PROTECT_LIST)
    parser.add_argument(
        "--rendered-root",
        type=Path,
        default=None,
        help="Optional C8 rendered bundle root for gpu_placement mode.",
    )
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--sweep-receipt-out", type=Path, default=DEFAULT_SWEEP_RECEIPT)
    parser.add_argument("--placement-receipt-out", type=Path, default=DEFAULT_PLACEMENT_RECEIPT)
    parser.add_argument(
        "--periodic-receipt-out",
        type=Path,
        default=gpu_periodic.DEFAULT_RECEIPT,
    )
    parser.add_argument("--loop-seconds", type=int, default=0)
    parser.add_argument(
        "--allow-redispatch",
        action="store_true",
        help="Dangerous: only when C8 agent supplies fresh attempt_ids or locks are cleared.",
    )
    parser.add_argument("--sequencing-plan-out", type=Path, default=c7_sequencing.DEFAULT_PLAN_OUT)
    args = parser.parse_args(argv)
    if args.mode == "gpu_periodic":
        return gpu_periodic.main(
            [
                "--kube-context",
                args.kube_context,
                "--namespace",
                args.namespace,
                "--protect-list",
                str(args.protect_list),
                "--c7-remaining",
                str(args.c7_remaining),
                "--c7-tail-lanes",
                str(args.c7_tail_lanes),
                "--c8-remaining",
                str(args.c8_remaining),
                "--receipt-out",
                str(args.periodic_receipt_out),
                *(["--loop-seconds", str(args.loop_seconds)] if args.loop_seconds else []),
            ]
        )
    if args.mode == "c7_sequencing":
        protect_list = gpu_scheduling.load_protect_list(args.protect_list)
        plan = c7_sequencing.build_sequencing_plan(
            kube_context=args.kube_context,
            namespace=args.namespace,
            protect_list=protect_list,
        )
        if not args.dry_run:
            args.sequencing_plan_out.parent.mkdir(parents=True, exist_ok=True)
            args.sequencing_plan_out.write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if args.mode == "gpu_placement":
        receipt = gpu_placement.enforce_gpu_placement(
            policy_id=args.policy,
            kube_context=args.kube_context,
            namespace=args.namespace,
            dry_run=args.dry_run,
            protect_list_path=args.protect_list,
            rendered_root=args.rendered_root,
            gates_only=False,
        )
        if not args.dry_run:
            args.placement_receipt_out.parent.mkdir(parents=True, exist_ok=True)
            args.placement_receipt_out.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    if args.mode == "gpu_sweep":
        receipt = gpu_sweep.run_gpu_pool_sweep(
            kube_context=args.kube_context,
            namespace=args.namespace,
            dry_run=args.dry_run or not args.allow_redispatch,
            c7_remaining_episodes=args.c7_remaining,
            protect_list_path=args.protect_list,
            allow_redispatch=args.allow_redispatch,
        )
        args.sweep_receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.sweep_receipt_out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    mode = "c7_restore" if args.mode == "normal" else args.mode
    receipt = arbitration.enforce_capacity_arbitration(
        kube_context=args.kube_context,
        namespace=args.namespace,
        dry_run=args.dry_run,
        mode=mode,
        c7_ceiling=args.c7_ceiling,
    )
    if not args.dry_run:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
