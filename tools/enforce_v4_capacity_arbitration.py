#!/usr/bin/env python3
"""Reclaim idle GPU capacity and arbitrate C7 vs G7 pilot lane concurrency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import v4_capacity_arbitration as arbitration  # noqa: E402

DEFAULT_RECEIPT = ROOT / "artifacts/online_correction_v4/execution/gpu_widen_20260908/capacity_arbitration_receipt.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--mode", choices=("pilot_priority", "c7_restore", "normal"), default=None)
    parser.add_argument("--c7-ceiling", type=int, default=None)
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)
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
