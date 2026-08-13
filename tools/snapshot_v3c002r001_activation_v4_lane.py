#!/usr/bin/env python3
"""Outcome-blind, byte-complete snapshot of one lane before A004 registration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = "vla-wam-shared-v3c002r001-activation-v4-lane-snapshot-v1"


def binding(path: Path) -> dict:
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lane-slot", required=True)
    p.add_argument("--raw-root", type=Path, required=True)
    p.add_argument("--current-incomplete-seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit("snapshot output already exists")
    behavior = args.raw_root / "behavioral"
    ledger = args.raw_root / "infrastructure_invalid.jsonl"
    if not behavior.is_dir() or not ledger.is_file():
        raise SystemExit("lane behavior/ledger is absent")
    rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
    attempts = []
    trace_count = request_count = raw_count = 0
    for row in rows:
        if row.get("infrastructure_status") != "infrastructure_invalid_excluded" or row.get("denominator_eligible") is not False or row.get("entire_partial_block_invalidated") is not True:
            raise SystemExit("infrastructure ledger row changed")
        root = Path(row["attempt_root"]).resolve()
        if not root.is_relative_to(behavior.resolve()) or not root.is_dir():
            raise SystemExit("infrastructure attempt root is absent/outside lane")
        files = [binding(path) for path in sorted(root.rglob("*")) if path.is_file()]
        traces = sorted(root.rglob("*action_trace.json"))
        requests = 0
        for trace in traces:
            value = json.loads(trace.read_text())
            events = value.get("request_events")
            if value.get("schema_version") != "vla-wam-shared-v3c002-pi05-action-trace-v1" or not isinstance(events, list):
                raise SystemExit("action trace schema changed")
            requests += len(events)
        raws = sorted(root.rglob("raw_episode.jsonl"))
        attempts.append({"seed_block_id": row["seed_block_id"], "attempt_root": str(root), "completed_cell_ids_before_failure": row.get("completed_cell_ids_before_failure", []), "successful_request_events_from_action_traces": requests, "action_trace_count": len(traces), "raw_completed_cell_count": len(raws), "files": files})
        trace_count += len(traces); request_count += requests; raw_count += len(raws)
    current = behavior / f"seed{args.current_incomplete_seed}" / "attempt001"
    marker = current.parent / "completed_block.json"; retry = current.parent / "attempt002"
    if not current.is_dir() or marker.exists() or retry.exists():
        raise SystemExit("current A004 retry block is not attempt001/no-marker/no-attempt002")
    markers = sorted(behavior.glob("seed*/completed_block.json"))
    output = {
        "schema_version": SCHEMA, "status": "captured_outcome_blind_before_universal_a004_registration", "passed": True,
        "lane_slot": args.lane_slot, "raw_root": str(args.raw_root.resolve()), "captured_outcomes": False,
        "infrastructure_ledger": binding(ledger), "all_infrastructure_invalid_attempts": attempts,
        "infrastructure_invalid_attempt_count": len(attempts), "successful_request_events_from_action_traces": request_count,
        "action_trace_count": trace_count, "raw_completed_cell_count_in_invalid_attempts": raw_count,
        "completed_block_count": len(markers), "completed_block_markers": [binding(path) for path in markers],
        "current_incomplete_seed": args.current_incomplete_seed, "current_attempt001": str(current),
        "current_attempt001_files": [binding(path) for path in sorted(current.rglob("*")) if path.is_file()],
        "completed_marker_absent": True, "attempt002_absent": True, "whole_block_retry_cell_count": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: output[key] for key in ("lane_slot", "infrastructure_invalid_attempt_count", "successful_request_events_from_action_traces", "action_trace_count", "raw_completed_cell_count_in_invalid_attempts", "completed_block_count")}))


if __name__ == "__main__":
    main()
