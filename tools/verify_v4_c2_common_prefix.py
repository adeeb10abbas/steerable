#!/usr/bin/env python3
"""Verify one C2 sham/move pair has a deterministic fresh-session common prefix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.prefix_replay import (  # noqa: E402
    PrefixReplayTolerance,
    canonical_json_bytes,
    verify_common_prefix,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sham-attempt", type=Path, required=True)
    parser.add_argument("--move-attempt", type=Path, required=True)
    parser.add_argument("--sham-session-receipt", type=Path, required=True)
    parser.add_argument("--move-session-receipt", type=Path, required=True)
    parser.add_argument("--position-tolerance-m", type=float, default=1e-4)
    parser.add_argument("--simulation-time-tolerance-s", type=float, default=1e-9)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = verify_common_prefix(
        left_attempt_dir=args.sham_attempt.resolve(),
        right_attempt_dir=args.move_attempt.resolve(),
        left_session_receipt_path=args.sham_session_receipt.resolve(),
        right_session_receipt_path=args.move_session_receipt.resolve(),
        tolerance=PrefixReplayTolerance(
            position_m=args.position_tolerance_m,
            simulation_time_s=args.simulation_time_tolerance_s,
        ),
    )
    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as handle:
        body = canonical_json_bytes(receipt) + b"\n"
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
