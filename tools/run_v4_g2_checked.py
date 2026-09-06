#!/usr/bin/env python3
"""Run a G2 seed child and fail closed on its write-once outcome markers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" not in arguments:
        print(
            "usage: run_v4_g2_checked.py [--expected-fixture FIXTURE] "
            "--expected-environment-seed SEED "
            "-- EXECUTABLE [ARG ...]",
            file=sys.stderr,
        )
        return 2
    separator = arguments.index("--")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expected-fixture", default="horizontal")
    parser.add_argument("--expected-environment-seed", type=int, required=True)
    try:
        expected, extras = parser.parse_known_args(arguments[:separator])
    except SystemExit:
        return 2
    if extras or separator + 1 >= len(arguments):
        print("invalid G2 wrapper arguments", file=sys.stderr)
        return 2
    expected_environment_seed = expected.expected_environment_seed
    expected_fixture = expected.expected_fixture

    output_raw = os.environ.get("EPISODE_OUTPUT_DIR")
    if not output_raw:
        print("EPISODE_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    output_dir = Path(output_raw)
    if not output_dir.is_absolute():
        print("EPISODE_OUTPUT_DIR must be absolute", file=sys.stderr)
        return 2

    completed = subprocess.run(arguments[separator + 1 :], check=False)
    receipt_path = output_dir / "g2_seed_receipt.json"
    failure_path = output_dir / "infrastructure_failure.json"

    if failure_path.is_file():
        try:
            failure = _load_object(failure_path)
            detail = {
                "schema_version": failure.get("schema_version"),
                "status": failure.get("status"),
                "error_type": failure.get("error_type"),
                "error": failure.get("error"),
            }
        except Exception as exc:
            detail = {"invalid_failure_marker": str(exc)}
        print(
            "G2 child wrote an infrastructure-failure marker: "
            + json.dumps(detail, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    if completed.returncode != 0:
        print(
            f"G2 child exited with status {completed.returncode} without a failure marker",
            file=sys.stderr,
        )
        return 1
    if not receipt_path.is_file():
        print(
            "G2 child exited successfully without g2_seed_receipt.json",
            file=sys.stderr,
        )
        return 1

    try:
        receipt = _load_object(receipt_path)
    except Exception as exc:
        print(f"G2 seed receipt is invalid: {exc}", file=sys.stderr)
        return 1
    receipt_schema = (
        f"v4-{expected_fixture.replace('_', '-')}-g2-seed-receipt-v1"
    )
    if receipt.get("schema_version") != receipt_schema:
        print("G2 seed receipt schema differs", file=sys.stderr)
        return 1
    if receipt.get("fixture_id") != expected_fixture:
        print("G2 seed receipt fixture differs", file=sys.stderr)
        return 1
    if receipt.get("passed_reset_and_camera") is not True:
        print("G2 seed receipt did not pass reset and camera checks", file=sys.stderr)
        return 1
    if receipt.get("environment_seed") != expected_environment_seed:
        print("G2 seed receipt environment seed differs", file=sys.stderr)
        return 1
    if (
        receipt.get("model_request_count") != 0
        or receipt.get("behavioral_episode_count") != 0
    ):
        print("G2 seed receipt reports prohibited model or behavioral work", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "checked_g2_seed_receipt": str(receipt_path),
                "environment_seed": receipt.get("environment_seed"),
                "fixture_id": receipt.get("fixture_id"),
                "passed_reset_and_camera": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
