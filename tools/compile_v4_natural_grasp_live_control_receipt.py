#!/usr/bin/env python3
"""Pull a natural-grasp live-control PVC receipt and register a passing control."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import v4_dispatch_gates as dispatch_gates  # noqa: E402

DEFAULT_REGISTRY = ROOT / "artifacts/online_correction_v4/setup/natural_grasp_live_control_registry.json"
PUBLISHER_POD = "211247-sz5vjy-vla4-b200-4gpu"
NAMESPACE = "211247-prod"


def _pull_json(remote_path: str) -> dict:
    proc = subprocess.run(
        [
            "kubectl",
            "exec",
            "-n",
            NAMESPACE,
            PUBLISHER_POD,
            "--",
            "cat",
            remote_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pvc-receipt-path", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--control-mode", choices=("scripted_grasp", "hold_only"), required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    args = parser.parse_args(argv)

    receipt = _pull_json(args.pvc_receipt_path)
    if not receipt.get("passed"):
        print(json.dumps({"status": "not_passed", "receipt": receipt}, indent=2), file=sys.stderr)
        return 2
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    dispatch_gates.record_passed_natural_grasp_live_control(
        fixture_id=args.fixture_id,
        control_mode=args.control_mode,
        attempt_id=args.attempt_id,
        receipt_path=str(args.out.relative_to(ROOT)),
        receipt_sha256=digest,
        study_commit=args.study_commit.lower(),
        registry_path=args.registry,
    )
    print(json.dumps({"registered": True, "receipt_path": str(args.out), "sha256": digest}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
