#!/usr/bin/env python3
"""Compile the C8 confirmatory G7 gate from queue binding and qualification receipts."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.build_v4_second_stack_g7_pilot_release import (
    FIXTURE_ID,
    POLICY_ID,
    artifact,
    canonical_json_bytes,
    load_json,
    require_passing,
    sha256_file,
    write_exclusive,
)


def load_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def build_receipt(
    *,
    queue_path: Path,
    runtime_lock_path: Path,
    pilot_g3_path_receipt: Path,
    main_g2_path: Path,
    main_g3_path: Path,
    hardware_g4_path: Path,
    g5_path: Path,
    g6_path: Path,
    attempt_id: str,
) -> dict:
    rows = [
        row
        for row in load_jsonl(queue_path)
        if row.get("family") == "C8" and row.get("cohort") == "confirmatory"
    ]
    if len(rows) != 768:
        raise ValueError("confirmatory C8 queue must contain 768 rows")
    lock = load_json(runtime_lock_path)
    if lock.get("release_status") != "PILOT_RELEASED":
        raise ValueError("runtime lock is not C8 pilot released")
    pilot_g3 = load_json(pilot_g3_path_receipt)
    if pilot_g3.get("passed") is not True:
        raise ValueError("pilot G3 path receipt is not passing")
    require_passing(load_json(main_g2_path))
    require_passing(load_json(main_g3_path))
    require_passing(load_json(hardware_g4_path))
    require_passing(load_json(g5_path))
    require_passing(load_json(g6_path))
    policy_seeds = list(dict.fromkeys(int(row["policy_seed"]) for row in rows))
    env_seeds = list(dict.fromkeys(int(row["env_seed"]) for row in rows))
    if len(policy_seeds) != 64:
        raise ValueError("confirmatory C8 must have 64 policy seeds")
    if len(env_seeds) != 64:
        raise ValueError("confirmatory C8 must have 64 environment seeds")
    return {
        "schema_version": "v4-second-stack-g7-confirmatory-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C8",
        "fixture_id": FIXTURE_ID,
        "policy_id": POLICY_ID,
        "gate": "G7",
        "attempt_id": attempt_id,
        "status": "passed",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "compiled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "allocation": {
            "episode_count": 768,
            "block_seed_count": 64,
            "confirmatory_only": True,
        },
        "checks": {
            "confirmatory_queue_has_768_rows": True,
            "pilot_g3_path_gate_passed": True,
            "main_g2_g3_g4_g5_g6_passing": True,
            "pilot_runtime_lock_bound": lock.get("release_status") == "PILOT_RELEASED",
        },
        "qualification_basis": {
            "confirmatory_queue": artifact(queue_path),
            "pilot_runtime_lock": artifact(runtime_lock_path),
            "pilot_g3_path": artifact(pilot_g3_path_receipt),
            "main_g2": artifact(main_g2_path),
            "main_g3": artifact(main_g3_path),
            "hardware_g4": artifact(hardware_g4_path),
            "g5": artifact(g5_path),
            "g6": artifact(g6_path),
        },
        "release_boundary": (
            "G7 confirmatory manifest binding only. G8 miniature rehearsal and "
            "RELEASED runtime lock remain required before 768-episode dispatch."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--pilot-runtime-lock", type=Path, required=True)
    parser.add_argument("--pilot-g3-path", type=Path, required=True)
    parser.add_argument("--main-g2", type=Path, required=True)
    parser.add_argument("--main-g3", type=Path, required=True)
    parser.add_argument("--hardware-g4", type=Path, required=True)
    parser.add_argument("--g5", type=Path, required=True)
    parser.add_argument("--g6", type=Path, required=True)
    parser.add_argument("--attempt-id", default="g7c8q20260908a")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        queue_path=args.queue.resolve(),
        runtime_lock_path=args.pilot_runtime_lock.resolve(),
        pilot_g3_path_receipt=args.pilot_g3_path.resolve(),
        main_g2_path=args.main_g2.resolve(),
        main_g3_path=args.main_g3.resolve(),
        hardware_g4_path=args.hardware_g4.resolve(),
        g5_path=args.g5.resolve(),
        g6_path=args.g6.resolve(),
        attempt_id=args.attempt_id,
    )
    write_exclusive(args.out.resolve(), canonical_json_bytes(receipt))
    print(json.dumps({"receipt": artifact(args.out.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
