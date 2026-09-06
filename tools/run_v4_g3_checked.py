#!/usr/bin/env python3
"""Run a G3 path-seed child and fail closed on infrastructure outcome markers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.model_blind_g3 import path_receipt_schema


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" not in arguments:
        print(
            "usage: run_v4_g3_checked.py [--expected-fixture FIXTURE] "
            "--expected-environment-seed SEED "
            "--expected-scale SCALE -- EXECUTABLE [ARG ...]",
            file=sys.stderr,
        )
        return 2
    separator = arguments.index("--")
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--expected-fixture", default="horizontal")
    parser.add_argument("--expected-environment-seed", type=int, required=True)
    parser.add_argument("--expected-scale", type=float, required=True)
    try:
        expected, extras = parser.parse_known_args(arguments[:separator])
    except SystemExit:
        return 2
    if extras or separator + 1 >= len(arguments):
        print("invalid G3 wrapper arguments", file=sys.stderr)
        return 2

    expected_environment_seed = expected.expected_environment_seed
    expected_scale = expected.expected_scale
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
    receipt_path = output_dir / "g3_path_seed_receipt.json"
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
            "G3 child wrote an infrastructure-failure marker: "
            + json.dumps(detail, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    if completed.returncode != 0:
        print(
            f"G3 child exited with status {completed.returncode} without a failure marker",
            file=sys.stderr,
        )
        return 1
    if not receipt_path.is_file():
        print(
            "G3 child exited successfully without g3_path_seed_receipt.json",
            file=sys.stderr,
        )
        return 1

    try:
        receipt = _load_object(receipt_path)
    except Exception as exc:
        print(f"G3 path-seed receipt is invalid: {exc}", file=sys.stderr)
        return 1
    if receipt.get("schema_version") != path_receipt_schema(expected_fixture):
        print("G3 path-seed receipt schema differs", file=sys.stderr)
        return 1
    if receipt.get("fixture_id") != expected_fixture:
        print("G3 path-seed receipt fixture differs", file=sys.stderr)
        return 1
    if receipt.get("environment_seed") != expected_environment_seed:
        print("G3 path-seed receipt environment seed differs", file=sys.stderr)
        return 1
    if abs(float(receipt.get("scale", float("nan"))) - expected_scale) > 1e-9:
        print("G3 path-seed receipt scale differs", file=sys.stderr)
        return 1
    if (
        receipt.get("model_request_count") != 0
        or receipt.get("behavioral_episode_count") != 0
    ):
        print(
            "G3 path-seed receipt reports prohibited model or behavioral work",
            file=sys.stderr,
        )
        return 1
    if not isinstance(receipt.get("passed"), bool):
        print("G3 path-seed receipt lacks a boolean passed field", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "checked_g3_path_seed_receipt": str(receipt_path),
                "fixture_id": receipt.get("fixture_id"),
                "environment_seed": receipt.get("environment_seed"),
                "scale": receipt.get("scale"),
                "passed": receipt.get("passed"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
