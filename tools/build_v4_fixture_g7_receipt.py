#!/usr/bin/env python3
"""Compile fixture-parameterized G7 engineering-pilot gate receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.fixture_qualification import qualification_profile
from tools.build_v4_object_pair_g7_receipt import (  # noqa: E402
    artifact,
    build_receipt as build_object_pair_receipt,
    load_json,
    load_jsonl,
    sha256_file,
)


def build_receipt(
    *,
    fixture_id: str,
    queue_path: Path,
    runtime_lock_path: Path,
    inventory_path: Path,
    review_path: Path,
    accepted_ledger_path: Path,
    ledger_manifest_path: Path,
    ledger_validation_report_path: Path,
) -> dict[str, object]:
    profile = qualification_profile(fixture_id)
    if fixture_id == "object_pair":
        payload = build_object_pair_receipt(
            queue_path=queue_path,
            runtime_lock_path=runtime_lock_path,
            inventory_path=inventory_path,
            review_path=review_path,
            accepted_ledger_path=accepted_ledger_path,
            ledger_manifest_path=ledger_manifest_path,
            ledger_validation_report_path=ledger_validation_report_path,
        )
    else:
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
        ledger = load_jsonl(accepted_ledger_path)
        queue = load_jsonl(queue_path)
        d_cap_m = float(lock["fixtures"][fixture_id]["D_cap_m"])
        payload["schema_version"] = (
            f"v4-{fixture_id.replace('_', '-')}-g7-engineering-pilot-receipt-v1"
        )
        payload["family_id"] = profile.family_id
        payload["fixture_id"] = fixture_id
        payload["checks"]["pilot_lock_is_exactly_pilot_released_for_c7"] = (
            lock.get("release_status") == "PILOT_RELEASED"
            and lock.get("released_families") == [profile.family_id]
        )
        payload["checks"]["queue_contains_24_disjoint_engineering_rows"] = (
            len(queue) == 24
            and all(not row.get("reuse_episode_ids") for row in queue)
            and all(row.get("family") == profile.family_id for row in queue)
        )
        payload["pilot_terminal_metadata_reconciliation"]["d_cap_m"] = d_cap_m
        payload["release_boundary"] = (
            f"A pass completes {profile.family_id} G7 only. G8 miniature-campaign "
            "rehearsal and a separate RELEASED lock remain required before "
            "confirmatory episodes."
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", required=True)
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
        fixture_id=args.fixture_id,
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
