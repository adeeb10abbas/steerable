"""Tests for simulator-only V4 horizontal G3 scripted Kubernetes rendering and wrapper."""

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
    "render_v4_horizontal_g3_scripted_k8s_jobs",
    "tools/render_v4_horizontal_g3_scripted_k8s_jobs.py",
)
validator = _load(
    "validate_v4_horizontal_g3_scripted_k8s_jobs",
    "tools/validate_v4_horizontal_g3_scripted_k8s_jobs.py",
)
checked = _load("run_v4_g3_scripted_checked", "tools/run_v4_g3_scripted_checked.py")
runner = _load(
    "run_v4_horizontal_g3_scripted_seed",
    "tools/run_v4_horizontal_g3_scripted_seed.py",
)


def _valid_compact_receipt(
    *,
    check_kind: str = "moving",
    goal: str = "left",
    reference_position: str = "endpoint",
    passed: bool = True,
) -> dict:
    reasons: list[str] = [] if passed else ["goal_not_satisfied"]
    return {
        "schema_version": "v4-horizontal-g3-scripted-check-receipt-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "check_kind": check_kind,
        "environment_seed": 2100000000,
        "goal": goal,
        "reference_position": reference_position,
        "scale": 1.0,
        "displacement_m": 0.12,
        "grasped": passed,
        "transported": passed,
        "released": passed,
        "stably_placed": passed,
        "goal_satisfied": passed,
        "evidence": {"path": "/tmp/t.json", "sha256": "a" * 64, "bytes": 1},
        "reasons": reasons,
        "passed": passed,
    }


def _valid_seed_receipt(
    *,
    mode: str = "moving",
    passed: bool = True,
    environment_seed: int = 2100000000,
) -> dict:
    if mode == "stationary":
        pairs = [
            ("left", "original"),
            ("left", "midpoint"),
            ("left", "endpoint"),
            ("right", "original"),
            ("right", "midpoint"),
            ("right", "endpoint"),
            ("front", "original"),
            ("front", "midpoint"),
            ("front", "endpoint"),
            ("behind", "original"),
            ("behind", "midpoint"),
            ("behind", "endpoint"),
        ]
        check_kind = "stationary"
    else:
        pairs = [
            ("left", "endpoint"),
            ("right", "endpoint"),
            ("front", "endpoint"),
            ("behind", "endpoint"),
        ]
        check_kind = "moving"
    checks = []
    for goal, reference_position in pairs:
        check_passed = passed if goal != "behind" else False if not passed else True
        if mode == "moving" and not passed:
            check_passed = goal != "behind"
        compact = _valid_compact_receipt(
            check_kind=check_kind,
            goal=goal,
            reference_position=reference_position,
            passed=check_passed if passed else check_passed,
        )
        checks.append(
            {
                "goal": goal,
                "reference_position": reference_position,
                "passed": compact["passed"],
                "trajectory": {"path": f"/tmp/t_{goal}.json", "sha256": "b" * 64, "bytes": 1},
                "receipt": {
                    "path": f"receipts/{goal}__{reference_position}.json",
                    "sha256": "c" * 64,
                    "bytes": 1,
                },
            }
        )
    passed_count = sum(1 for check in checks if check["passed"])
    failed_count = len(checks) - passed_count
    return {
        "schema_version": "v4-horizontal-g3-scripted-seed-receipt-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "mode": mode,
        "environment_seed": environment_seed,
        "scale": 1.0,
        "displacement_m": 0.12,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "controller_config": runner.frozen_scripted_controller_config(),
        "plan_sha256": "d" * 64,
        "campaign_sha256": "e" * 64,
        "reset_registry_sha256": "f" * 64,
        "runtime_identity": {"mode": mode},
        "check_order": ["goal_declared_order", "reference_position_declared_order"],
        "check_count": len(checks),
        "passed_check_count": passed_count,
        "failed_check_count": failed_count,
        "passed": failed_count == 0,
        "checks": checks,
        "registered_reset": {},
    }


class RunV4G3ScriptedCheckedTests(unittest.TestCase):
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

    def _argv(self, executable: str, *, mode: str = "moving") -> list[str]:
        return [
            "--expected-environment-seed",
            "2100000000",
            "--expected-scale",
            "1.0",
            "--expected-mode",
            mode,
            "--",
            executable,
        ]

    def _write_receipt_bundle(self, receipt: dict) -> None:
        receipts_dir = self.output_dir / "receipts"
        receipts_dir.mkdir()
        for check in receipt["checks"]:
            compact = _valid_compact_receipt(
                check_kind=receipt["mode"],
                goal=check["goal"],
                reference_position=check["reference_position"],
                passed=check["passed"],
            )
            from experiments.online_correction_v4.model_blind_g3 import (
                canonical_json_bytes,
                sha256_bytes,
            )

            body = canonical_json_bytes(compact)
            path = receipts_dir / f"{check['goal']}__{check['reference_position']}.json"
            path.write_bytes(body)
            check["receipt"]["sha256"] = sha256_bytes(body)
            check["receipt"]["path"] = str(path)
        (self.output_dir / "g3_scripted_seed_receipt.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )

    def test_accepts_valid_receipt_when_passed_false(self) -> None:
        receipt = _valid_seed_receipt(mode="moving", passed=False)
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 0)

    def test_accepts_valid_receipt_when_passed_true(self) -> None:
        receipt = _valid_seed_receipt(mode="moving", passed=True)
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 0)

    def test_accepts_stationary_receipt_with_twelve_checks(self) -> None:
        receipt = _valid_seed_receipt(mode="stationary", passed=True)
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main(
                [*self._argv(sys.executable, mode="stationary"), "-c", "pass"]
            )
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
        receipt = _valid_seed_receipt(mode="moving", passed=True)
        receipt["schema_version"] = "wrong-schema"
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_seed_mismatch(self) -> None:
        receipt = _valid_seed_receipt(mode="moving", passed=True, environment_seed=2100000001)
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_mode_mismatch(self) -> None:
        receipt = _valid_seed_receipt(mode="stationary", passed=True)
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable, mode="moving"), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_wrong_check_count(self) -> None:
        receipt = _valid_seed_receipt(mode="moving", passed=True)
        receipt["checks"] = receipt["checks"][:2]
        receipt["check_count"] = 2
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_model_request_count(self) -> None:
        receipt = _valid_seed_receipt(mode="moving", passed=True)
        receipt["model_request_count"] = 1
        self._write_receipt_bundle(receipt)
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)

    def test_rejects_invalid_compact_receipt(self) -> None:
        receipt = _valid_seed_receipt(mode="moving", passed=True)
        receipts_dir = self.output_dir / "receipts"
        receipts_dir.mkdir()
        for check in receipt["checks"]:
            bad = _valid_compact_receipt(
                check_kind="moving",
                goal=check["goal"],
                reference_position=check["reference_position"],
                passed=True,
            )
            bad["schema_version"] = "wrong-schema"
            path = receipts_dir / f"{check['goal']}__{check['reference_position']}.json"
            path.write_text(json.dumps(bad), encoding="utf-8")
            check["receipt"]["path"] = str(path)
        (self.output_dir / "g3_scripted_seed_receipt.json").write_text(
            json.dumps(receipt),
            encoding="utf-8",
        )
        with mock.patch.object(subprocess, "run", return_value=mock.Mock(returncode=0)):
            code = checked.main([*self._argv(sys.executable), "-c", "pass"])
        self.assertEqual(code, 1)


@unittest.skipUnless(shutil.which("kubectl"), "kubectl is required")
class HorizontalG3ScriptedK8sTests(unittest.TestCase):
    def test_default_spec_renders_complete_simulator_only_scripted_bundle(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        expected_scale = spec["scale"]
        expected_attempt = spec["attempt_id"]
        with tempfile.TemporaryDirectory() as tmp:
            report = renderer.render(renderer.DEFAULT_SPEC, Path(tmp))
            root = Path(report["bundle_root"])
            validated = validator.validate(root)
            manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(validated["ok"])
        self.assertEqual(validated["job_count"], 10)
        self.assertEqual(validated["scripted_job_count"], 10)
        self.assertEqual(validated["scale"], expected_scale)
        self.assertEqual(validated["model_request_count"], 0)
        self.assertEqual(validated["behavioral_episode_count"], 0)
        self.assertEqual(manifest["attempt_id"], expected_attempt)
        modes = {row["mode"] for row in manifest["scripted_jobs"]}
        self.assertEqual(modes, {"stationary", "moving"})
        stationary = [
            row for row in manifest["scripted_jobs"] if row["mode"] == "stationary"
        ]
        moving = [row for row in manifest["scripted_jobs"] if row["mode"] == "moving"]
        self.assertEqual(len(stationary), 9)
        self.assertEqual(len(moving), 1)
        canonical = next(row for row in moving)["environment_seed"]
        self.assertIn(
            (canonical, "stationary"),
            {(row["environment_seed"], row["mode"]) for row in manifest["scripted_jobs"]},
        )

    def test_renderer_rejects_unknown_keys(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        spec["unexpected_field"] = True
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(spec_path, Path(tmp) / "out")

    def test_renderer_rejects_unregistered_scale(self) -> None:
        spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
        spec["scale"] = 9.9
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(spec_path, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
