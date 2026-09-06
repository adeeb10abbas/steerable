#!/usr/bin/env python3
"""Run a G3 scripted-seed child and fail closed on infrastructure outcome markers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

RECEIPT_SCHEMA = "v4-horizontal-g3-scripted-seed-receipt-v1"
SCRIPTED_MODES = ("stationary", "moving")
EXPECTED_CHECK_COUNTS = {"stationary": 12, "moving": 4}


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return value


def _validate_compact_check_receipt(receipt: dict[str, Any]) -> None:
    from experiments.online_correction_v4.model_blind_g3 import (
        validate_scripted_check_receipt,
    )

    validate_scripted_check_receipt(receipt)


def _validate_seed_receipt(
    receipt: dict[str, Any],
    *,
    output_dir: Path,
    expected_environment_seed: int,
    expected_scale: float,
    expected_mode: str,
) -> None:
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("G3 scripted-seed receipt schema differs")
    if receipt.get("environment_seed") != expected_environment_seed:
        raise ValueError("G3 scripted-seed receipt environment seed differs")
    if abs(float(receipt.get("scale", float("nan"))) - expected_scale) > 1e-9:
        raise ValueError("G3 scripted-seed receipt scale differs")
    if receipt.get("mode") != expected_mode:
        raise ValueError("G3 scripted-seed receipt mode differs")
    expected_count = EXPECTED_CHECK_COUNTS.get(expected_mode)
    if expected_count is None:
        raise ValueError("expected mode is invalid")
    if receipt.get("check_count") != expected_count:
        raise ValueError("G3 scripted-seed receipt check_count differs")
    checks = receipt.get("checks")
    if not isinstance(checks, list) or len(checks) != expected_count:
        raise ValueError("G3 scripted-seed receipt checks differ")
    if (
        receipt.get("model_request_count") != 0
        or receipt.get("behavioral_episode_count") != 0
    ):
        raise ValueError(
            "G3 scripted-seed receipt reports prohibited model or behavioral work"
        )
    if not isinstance(receipt.get("passed"), bool):
        raise ValueError("G3 scripted-seed receipt lacks a boolean passed field")
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            raise ValueError(f"G3 scripted-seed check {index} is invalid")
        receipt_identity = check.get("receipt")
        if not isinstance(receipt_identity, dict):
            raise ValueError(f"G3 scripted-seed check {index} lacks receipt identity")
        receipt_path_raw = receipt_identity.get("path")
        if not isinstance(receipt_path_raw, str) or not receipt_path_raw:
            raise ValueError(f"G3 scripted-seed check {index} lacks receipt path")
        receipt_path = Path(receipt_path_raw)
        if not receipt_path.is_absolute():
            receipt_path = output_dir / receipt_path
        if not receipt_path.is_file():
            raise ValueError(f"G3 scripted-seed check {index} receipt file is missing")
        compact = _load_object(receipt_path)
        expected_sha = receipt_identity.get("sha256")
        if (
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and all(char in "0123456789abcdef" for char in expected_sha)
        ):
            from experiments.online_correction_v4.model_blind_g3 import sha256_bytes
            from experiments.online_correction_v4.model_blind_g3 import (
                canonical_json_bytes,
            )

            actual_sha = sha256_bytes(canonical_json_bytes(compact))
            if actual_sha != expected_sha:
                raise ValueError(
                    f"G3 scripted-seed check {index} receipt SHA-256 differs"
                )
        _validate_compact_check_receipt(compact)


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        len(arguments) < 8
        or arguments[0] != "--expected-environment-seed"
        or arguments[2] != "--expected-scale"
        or arguments[4] != "--expected-mode"
        or arguments[6] != "--"
    ):
        print(
            "usage: run_v4_g3_scripted_checked.py --expected-environment-seed SEED "
            "--expected-scale SCALE --expected-mode MODE -- EXECUTABLE [ARG ...]",
            file=sys.stderr,
        )
        return 2
    try:
        expected_environment_seed = int(arguments[1])
    except ValueError:
        print("expected environment seed must be an integer", file=sys.stderr)
        return 2
    try:
        expected_scale = float(arguments[3])
    except ValueError:
        print("expected scale must be numeric", file=sys.stderr)
        return 2
    expected_mode = arguments[5]
    if expected_mode not in SCRIPTED_MODES:
        print("expected mode must be stationary or moving", file=sys.stderr)
        return 2

    output_raw = os.environ.get("EPISODE_OUTPUT_DIR")
    if not output_raw:
        print("EPISODE_OUTPUT_DIR is required", file=sys.stderr)
        return 2
    output_dir = Path(output_raw)
    if not output_dir.is_absolute():
        print("EPISODE_OUTPUT_DIR must be absolute", file=sys.stderr)
        return 2

    completed = subprocess.run(arguments[7:], check=False)
    receipt_path = output_dir / "g3_scripted_seed_receipt.json"
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
            "G3 scripted child wrote an infrastructure-failure marker: "
            + json.dumps(detail, sort_keys=True),
            file=sys.stderr,
        )
        return 1

    if completed.returncode != 0:
        print(
            f"G3 scripted child exited with status {completed.returncode} "
            "without a failure marker",
            file=sys.stderr,
        )
        return 1
    if not receipt_path.is_file():
        print(
            "G3 scripted child exited successfully without "
            "g3_scripted_seed_receipt.json",
            file=sys.stderr,
        )
        return 1

    try:
        receipt = _load_object(receipt_path)
        _validate_seed_receipt(
            receipt,
            output_dir=output_dir,
            expected_environment_seed=expected_environment_seed,
            expected_scale=expected_scale,
            expected_mode=expected_mode,
        )
    except Exception as exc:
        print(f"G3 scripted-seed receipt is invalid: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "checked_g3_scripted_seed_receipt": str(receipt_path),
                "environment_seed": receipt.get("environment_seed"),
                "scale": receipt.get("scale"),
                "mode": receipt.get("mode"),
                "passed": receipt.get("passed"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
