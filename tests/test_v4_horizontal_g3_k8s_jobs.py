"""Tests for simulator-only V4 horizontal G3 Kubernetes rendering and wrapper."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


renderer = _load(
    "render_v4_horizontal_g3_k8s_jobs",
    "tools/render_v4_horizontal_g3_k8s_jobs.py",
)
validator = _load(
    "validate_v4_horizontal_g3_k8s_jobs",
    "tools/validate_v4_horizontal_g3_k8s_jobs.py",
)
checked = _load("run_v4_g3_checked", "tools/run_v4_g3_checked.py")


def _valid_receipt(*, passed: bool = True, environment_seed: int = 2100000000) -> dict:
    return {
        "schema_version": "v4-horizontal-g3-path-seed-receipt-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "plan_receipt": {
            "path": "/plan.json",
            "sha256": "a" * 64,
            "bytes": 1,
        },
        "scale": 2.0,
        "displacement_m": 0.24,
        "environment_seed": environment_seed,
        "counterbalance": {"block_id": 0},
        "direction_task_coefficients_by_goal": {
            "left": [1.0, 0.0],
            "right": [1.0, 0.0],
            "front": [0.0, 1.0],
            "behind": [0.0, 1.0],
        },
        "check_order": ["goal_declared_order", "scenario_declared_order"],
        "check_count": 24,
        "checks": [],
        "goal_area_cases": [],
        "information_gate_passed": passed,
        "passed": passed,
        "passed_check_count": 0,
        "failed_check_count": 0,
    }


class RunV4G3CheckedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name) / "episode"
        self.output_dir.mkdir()
        self.env_patch = mock.patch.dict(
            os.environ,
            {"EPISODE_OUTPUT_DIR": str(self.output_dir)},
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.tmp.cleanup()

    def _argv(self, executable: str) -> list[str]:
        return [
            "--expected-environment-seed",
            "2100000000",
            "--expected-scale",
            "2.0",
            "--",
            executable,
        ]

    def test_accepts_valid_receipt_when_passed_false(self) -> None:
        (self.output_dir / "g3_path_seed_receipt.json").write_text(
            json.dumps(_valid_receipt(passed=False)),
            encoding="utf-8",
        )
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 0)

    def test_accepts_valid_receipt_when_passed_true(self) -> None:
        (self.output_dir / "g3_path_seed_receipt.json").write_text(
            json.dumps(_valid_receipt(passed=True)),
            encoding="utf-8",
        )
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 0)

    def test_rejects_infrastructure_failure_marker(self) -> None:
        (self.output_dir / "infrastructure_failure.json").write_text(
            json.dumps(
                {
                    "schema_version": "v4-horizontal-g3-infrastructure-failure-v1",
                    "status": "infrastructure_invalid",
                    "error_type": "RuntimeError",
                    "error": "simulator crashed",
                }
            ),
            encoding="utf-8",
        )
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=1)):
            code = checked.main([*self._argv(sys.executable), "-c", "fail"])
        self.assertEqual(code, 1)

    def test_rejects_missing_receipt(self) -> None:
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_invalid_receipt_schema(self) -> None:
        receipt = _valid_receipt(passed=True)
        receipt["schema_version"] = "wrong-schema"
        (self.output_dir / "g3_path_seed_receipt.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_seed_mismatch(self) -> None:
        (self.output_dir / "g3_path_seed_receipt.json").write_text(
            json.dumps(_valid_receipt(environment_seed=2100000001)),
            encoding="utf-8",
        )
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)


@unittest.skipUnless(shutil.which("kubectl"), "kubectl is required")
class HorizontalG3K8sTests(unittest.TestCase):
    def test_default_spec_renders_complete_simulator_only_seed_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = renderer.render(renderer.DEFAULT_SPEC, Path(tmp))
            root = Path(report["bundle_root"])
            validated = validator.validate(root)
        self.assertTrue(validated["ok"])
        self.assertEqual(validated["job_count"], 128)
        self.assertEqual(validated["environment_seed_count"], 128)
        self.assertEqual(validated["scale"], 2.0)
        self.assertEqual(validated["model_request_count"], 0)
        self.assertEqual(validated["behavioral_episode_count"], 0)

    def test_renderer_rejects_partial_seed_limit(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        spec["max_seed_jobs"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3RenderError):
                renderer.render(spec_path, Path(tmp) / "out")

    def test_renderer_rejects_unknown_keys(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        spec["unexpected_field"] = True
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3RenderError):
                renderer.render(spec_path, Path(tmp) / "out")

    def test_renderer_rejects_unregistered_scale(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        spec["scale"] = 9.9
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3RenderError):
                renderer.render(spec_path, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
