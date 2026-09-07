#!/usr/bin/env python3
"""Run the C8 G8 miniature campaign rehearsal with frozen second_stack defaults."""

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

DEFAULT_CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
DEFAULT_G7_MANIFEST = (
    ROOT / "artifacts/online_correction_v4/setup/second_stack_g7_pilot_queue.jsonl"
)
DEFAULT_PROTOCOL_SHA = "0" * 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_protocol_sha256() -> str:
    protocol_path = ROOT / "artifacts/online_correction_v4/protocol.json"
    if protocol_path.is_file():
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
        digest = payload.get("protocol_sha256") or payload.get("content_sha256")
        if isinstance(digest, str) and len(digest) == 64:
            return digest
    return DEFAULT_PROTOCOL_SHA


def _load_scorer_sha256() -> str:
    scorer_path = ROOT / "experiments/online_correction_v4/droid_scorer.py"
    if scorer_path.is_file():
        return sha256_file(scorer_path)
    return "1" * 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--pilot-queue", type=Path, default=DEFAULT_G7_MANIFEST)
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol-sha256", default=None)
    parser.add_argument("--scorer-sha256", default=None)
    parser.add_argument("--scheduler-receipt", type=Path, action="append", default=[])
    parser.add_argument(
        "--invalid-attempt-receipt",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    command = [
        sys.executable,
        str(ROOT / "tools/run_v4_g8_miniature_rehearsal.py"),
        "--raw-root",
        str(args.raw_root.resolve()),
        "--pilot-queue",
        str(args.pilot_queue.resolve()),
        "--config",
        str(args.config.resolve()),
        "--protocol-sha256",
        args.protocol_sha256 or _load_protocol_sha256(),
        "--scorer-sha256",
        args.scorer_sha256 or _load_scorer_sha256(),
        "--output-dir",
        str(args.output_dir.resolve()),
    ]
    for selection in args.select:
        command.extend(["--select", selection])
    for receipt in args.scheduler_receipt:
        command.extend(["--scheduler-receipt", str(receipt.resolve())])
    for receipt in args.invalid_attempt_receipt:
        command.extend(["--invalid-attempt-receipt", str(receipt.resolve())])

    completed = subprocess.run(command, check=False, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
