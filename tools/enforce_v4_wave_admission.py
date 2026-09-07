#!/usr/bin/env python3
"""Cancel superseded V4 waves and enforce serialized admission on the cluster."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import v4_wave_admission as admission  # noqa: E402

DEFAULT_RECEIPT = (
    ROOT / "artifacts/online_correction_v4/execution/gpu_widen_20260908/admission_enforcement_receipt.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kube-context", default="prod-dcwi-warrenq1-vmkub007")
    parser.add_argument("--namespace", default="211247-prod")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cancel-only", action="store_true")
    parser.add_argument("--enforce-only", action="store_true")
    parser.add_argument("--receipt-out", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args(argv)

    jobs = admission.fetch_v4_jobs(kube_context=args.kube_context, namespace=args.namespace)
    before_summary = admission.summarize_attempts(jobs)

    cancelled: list[dict[str, object]] = []
    if not args.enforce_only:
        cancelled = admission.cancel_superseded_jobs(
            jobs,
            kube_context=args.kube_context,
            namespace=args.namespace,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            jobs = admission.fetch_v4_jobs(kube_context=args.kube_context, namespace=args.namespace)

    enforcement: dict[str, object] = {}
    if not args.cancel_only:
        enforcement = admission.enforce_admission_on_cluster(
            jobs,
            kube_context=args.kube_context,
            namespace=args.namespace,
            dry_run=args.dry_run,
        )

    receipt = {
        "schema_version": "v4-wave-admission-enforcement-receipt-v1",
        "observed_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kube_context": args.kube_context,
        "namespace": args.namespace,
        "dry_run": args.dry_run,
        "before_attempt_summary": before_summary,
        "cancelled_jobs": cancelled,
        "cancelled_job_count": len(cancelled),
        "enforcement": enforcement,
        "superseded_attempt_ids": sorted(admission.SUPERSEDED_ATTEMPT_IDS),
    }
    if not args.dry_run:
        args.receipt_out.parent.mkdir(parents=True, exist_ok=True)
        args.receipt_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
