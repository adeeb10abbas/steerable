"""Tests for the parent-side G2 outcome-marker checker."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_v4_g2_checked", ROOT / "tools/run_v4_g2_checked.py"
)
wrapper = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(wrapper)


def _writer(filename: str, payload: dict) -> str:
    return (
        "import json,os,pathlib;"
        "p=pathlib.Path(os.environ['EPISODE_OUTPUT_DIR']);"
        "p.mkdir(parents=True);"
        f"(p/{filename!r}).write_text(json.dumps({payload!r}))"
    )


class G2CheckedRunnerTests(unittest.TestCase):
    def _run(self, output_dir: Path, code: str) -> int:
        with mock.patch.dict(
            os.environ, {"EPISODE_OUTPUT_DIR": str(output_dir)}, clear=False
        ):
            return wrapper.main(
                [
                    "--expected-environment-seed",
                    "2100000000",
                    "--",
                    sys.executable,
                    "-c",
                    code,
                ]
            )

    def test_accepts_only_passing_zero_inference_receipt(self) -> None:
        receipt = {
            "schema_version": "v4-horizontal-g2-seed-receipt-v1",
            "fixture_id": "horizontal",
            "environment_seed": 2100000000,
            "passed_reset_and_camera": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "episode"
            self.assertEqual(
                self._run(output, _writer("g2_seed_receipt.json", receipt)),
                0,
            )

    def test_rejects_false_success_with_infrastructure_marker(self) -> None:
        failure = {
            "schema_version": "v4-horizontal-g2-infrastructure-failure-v1",
            "status": "infrastructure_invalid",
            "error_type": "TypeError",
            "error": "factory mismatch",
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "episode"
            self.assertEqual(
                self._run(output, _writer("infrastructure_failure.json", failure)),
                1,
            )

    def test_rejects_zero_exit_without_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                self._run(Path(tmp) / "episode", "pass"),
                1,
            )

    def test_rejects_receipt_for_a_different_seed(self) -> None:
        receipt = {
            "schema_version": "v4-horizontal-g2-seed-receipt-v1",
            "fixture_id": "horizontal",
            "environment_seed": 2100000001,
            "passed_reset_and_camera": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "episode"
            self.assertEqual(
                self._run(output, _writer("g2_seed_receipt.json", receipt)),
                1,
            )

    def test_accepts_object_pair_receipt_with_explicit_fixture(self) -> None:
        receipt = {
            "schema_version": "v4-object-pair-g2-seed-receipt-v1",
            "fixture_id": "object_pair",
            "environment_seed": 2100040000,
            "passed_reset_and_camera": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "episode"
            with mock.patch.dict(
                os.environ,
                {"EPISODE_OUTPUT_DIR": str(output)},
                clear=False,
            ):
                result = wrapper.main(
                    [
                        "--expected-fixture",
                        "object_pair",
                        "--expected-environment-seed",
                        "2100040000",
                        "--",
                        sys.executable,
                        "-c",
                        _writer("g2_seed_receipt.json", receipt),
                    ]
                )
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
