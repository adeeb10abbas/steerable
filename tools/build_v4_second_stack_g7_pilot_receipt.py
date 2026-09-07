#!/usr/bin/env python3
"""Compile the C8 engineering-pilot G7 gate from ledger and video review evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_v4_object_pair_g7_receipt import (  # noqa: E402
    artifact,
    build_receipt as build_object_pair_receipt,
    load_json,
    load_jsonl,
)


FIXTURE_ID = "second_stack"
FAMILY_ID = "C8"
POLICY_ID = "groot_bridge_widowx"


def build_receipt(
    *,
    queue_path: Path,
    runtime_lock_path: Path,
    inventory_path: Path,
    review_path: Path,
    accepted_ledger_path: Path,
    ledger_manifest_path: Path,
    ledger_validation_report_path: Path,
) -> dict[str, object]:
    payload = build_object_pair_receipt(
        queue_path=queue_path,
        runtime_lock_path=runtime_lock_path,
        inventory_path=inventory_path,
        review_path=review_path,
        accepted_ledger_path=accepted_ledger_path,
        ledger_manifest_path=ledger_manifest_path,
        ledger_validation_report_path=ledger_validation_report_path,
    )
    lock = load_json(runtime_lock_path)
    queue = load_jsonl(queue_path)
    d_cap_m = float(lock["fixtures"][FIXTURE_ID]["D_cap_m"])
    payload["schema_version"] = "v4-second-stack-g7-engineering-pilot-receipt-v1"
    payload["family_id"] = FAMILY_ID
    payload["fixture_id"] = FIXTURE_ID
    payload["policy_id"] = POLICY_ID
    payload["checks"]["pilot_lock_is_exactly_pilot_released_for_c7"] = (
        lock.get("release_status") == "PILOT_RELEASED"
        and lock.get("released_families") == [FAMILY_ID]
    )
    payload["checks"]["queue_contains_24_disjoint_engineering_rows"] = (
        len(queue) == 24
        and all(not row.get("reuse_episode_ids") for row in queue)
        and all(row.get("family") == FAMILY_ID for row in queue)
    )
    payload["pilot_terminal_metadata_reconciliation"]["d_cap_m"] = d_cap_m
    payload["release_boundary"] = (
        f"A pass completes {FAMILY_ID} G7 engineering pilot only. G8 "
        "miniature-campaign rehearsal and a separate RELEASED lock remain "
        "required before confirmatory episodes."
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--video-inventory", type=Path, required=True)
    parser.add_argument("--video-review", type=Path, required=True)
    parser.add_argument("--accepted-ledger", type=Path, required=True)
    parser.add_argument("--ledger-manifest", type=Path, required=True)
    parser.add_argument("--ledger-validation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_receipt(
        queue_path=args.queue.resolve(),
        runtime_lock_path=args.runtime_lock.resolve(),
        inventory_path=args.video_inventory.resolve(),
        review_path=args.video_review.resolve(),
        accepted_ledger_path=args.accepted_ledger.resolve(),
        ledger_manifest_path=args.ledger_manifest.resolve(),
        ledger_validation_report_path=args.ledger_validation_report.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
    print(json.dumps({"status": payload["status"], "path": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
