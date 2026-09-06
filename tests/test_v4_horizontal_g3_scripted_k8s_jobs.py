"""Tests for simulator-only V4 horizontal G3 scripted Kubernetes rendering and wrapper."""

from __future__ import annotations

import hashlib
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

from experiments.online_correction_v4.model_blind_g3 import (
    PATH_SAMPLE_INTERVAL_S,
    canonical_json_bytes,
    compile_path_scale_receipt,
    compile_path_seed_receipt,
    expected_path_check_keys,
    sha256_file,
    validate_path_scale_receipt,
    validate_plan_payload,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json"
)


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


def _evidence(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _passing_path_observations(*, suffix: str = "a") -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for goal, scenario in expected_path_check_keys():
        measured = f"evidence/measured/{goal}_{scenario}_{suffix}.json".encode("utf-8")
        reference = f"evidence/reference/{goal}_{scenario}_{suffix}.json".encode("utf-8")
        observations.append(
            {
                "planned_duration_s": 1.0,
                "sample_interval_s": PATH_SAMPLE_INTERVAL_S,
                "sample_count": 51,
                "measured_pose_evidence": _evidence(
                    f"artifacts/g3/measured/{goal}_{scenario}_{suffix}.json",
                    measured,
                ),
                "reference_pose_evidence": _evidence(
                    f"artifacts/g3/reference/{goal}_{scenario}_{suffix}.json",
                    reference,
                ),
                "path_conformance": True,
                "collision_free": True,
                "support_valid": True,
                "reachable_workspace": True,
                "legal_goal_nonempty": True,
                "reference_robot_contact": False,
                "unmodeled_collision": False,
                "reasons": [],
            }
        )
    return observations


def _goal_area_cases() -> list[dict[str, object]]:
    return [
        {
            "relation": goal,
            "original_area_m2": 0.10,
            "destination_area_m2": 0.075,
            "shrinking_direction": True,
            "removed_area_fraction": 0.25,
            "minimum_shrinking_area_fraction": 0.20,
            "original_goal_empty": False,
            "destination_goal_empty": False,
            "passes_information_gate": True,
        }
        for goal in ("left", "right", "front", "behind")
    ]


def _build_path_seed_receipt(
    *,
    plan: dict[str, object],
    plan_receipt: dict[str, str],
    environment_seed: int,
    scale: float,
    passed: bool = True,
) -> dict[str, object]:
    observations = _passing_path_observations(suffix=str(environment_seed))
    if not passed:
        observations[0]["collision_free"] = False
        observations[0]["reasons"] = ["collision detected"]
    return compile_path_seed_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        environment_seed=environment_seed,
        scale=scale,
        check_observations=observations,
        goal_area_cases=_goal_area_cases(),
    )


def _write_passing_path_scale_receipt(tmp: Path, *, scale: float = 1.0) -> Path:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    validate_plan_payload(plan)
    plan_receipt = {"path": str(PLAN), "sha256": sha256_file(PLAN)}
    receipts = [
        _build_path_seed_receipt(
            plan=plan,
            plan_receipt=plan_receipt,
            environment_seed=int(seed),
            scale=scale,
        )
        for seed in plan["registered_env_seeds"]
    ]
    report = compile_path_scale_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        scale=scale,
        path_seed_receipts=receipts,
    )
    validate_path_scale_receipt(report, plan=plan)
    receipt_path = tmp / "g3_path_scale_receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(report))
    return receipt_path


def _authorized_spec(tmp: Path, receipt_path: Path) -> Path:
    spec = json.loads(renderer.DEFAULT_SPEC.read_text(encoding="utf-8"))
    spec_dir = renderer.DEFAULT_SPEC.parent
    for key in (
        "marker_wrapper_source",
        "runner_source",
        "gate_core_source",
        "campaign_source",
        "plan_source",
        "reset_registry_source",
    ):
        value = spec.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            spec[key] = str((spec_dir / value).resolve())
    spec["authorization_status"] = "authorized_by_passing_path_scale_receipt"
    spec["path_scale_receipt_source"] = str(receipt_path.resolve())
    spec["path_scale_receipt_path"] = str(receipt_path.resolve())
    spec["path_scale_receipt_sha256"] = sha256_file(receipt_path)
    spec_path = tmp / "authorized-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    return spec_path


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
    def test_default_spec_render_is_blocked_pending_path_scale_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(renderer.DEFAULT_SPEC, Path(tmp))

    def test_authorized_spec_renders_complete_simulator_only_scripted_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            receipt_path = _write_passing_path_scale_receipt(tmp_path)
            receipt_sha = sha256_file(receipt_path)
            spec_path = _authorized_spec(tmp_path, receipt_path)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            expected_scale = spec["scale"]
            expected_attempt = spec["attempt_id"]
            report = renderer.render(spec_path, tmp_path / "out")
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
            self.assertEqual(
                manifest["authorization_status"],
                "authorized_by_passing_path_scale_receipt",
            )
            self.assertEqual(manifest["path_scale_receipt_sha256"], receipt_sha)
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

    def test_renderer_rejects_failed_path_scale_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = json.loads(PLAN.read_text(encoding="utf-8"))
            plan_receipt = {"path": str(PLAN), "sha256": sha256_file(PLAN)}
            seeds = list(plan["registered_env_seeds"])
            receipts = [
                _build_path_seed_receipt(
                    plan=plan,
                    plan_receipt=plan_receipt,
                    environment_seed=int(seed),
                    scale=1.0,
                    passed=int(seed) != int(seeds[0]),
                )
                for seed in seeds
            ]
            failed = compile_path_scale_receipt(
                plan=plan,
                plan_receipt=plan_receipt,
                scale=1.0,
                path_seed_receipts=receipts,
            )
            receipt_path = tmp_path / "g3_path_scale_receipt_failed.json"
            receipt_path.write_bytes(canonical_json_bytes(failed))
            spec_path = _authorized_spec(tmp_path, receipt_path)
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(spec_path, tmp_path / "out")

    def test_renderer_rejects_partial_path_scale_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            plan = json.loads(PLAN.read_text(encoding="utf-8"))
            plan_receipt = {"path": str(PLAN), "sha256": sha256_file(PLAN)}
            seeds = list(plan["registered_env_seeds"])[:-1]
            receipts = [
                _build_path_seed_receipt(
                    plan=plan,
                    plan_receipt=plan_receipt,
                    environment_seed=int(seed),
                    scale=1.0,
                )
                for seed in seeds
            ]
            partial = compile_path_scale_receipt(
                plan=plan,
                plan_receipt=plan_receipt,
                scale=1.0,
                path_seed_receipts=receipts,
            )
            receipt_path = tmp_path / "g3_path_scale_receipt_partial.json"
            receipt_path.write_bytes(canonical_json_bytes(partial))
            spec_path = _authorized_spec(tmp_path, receipt_path)
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(spec_path, tmp_path / "out")

    def test_renderer_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            receipt_path = _write_passing_path_scale_receipt(tmp_path)
            spec_path = _authorized_spec(tmp_path, receipt_path)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["unexpected_field"] = True
            bad_spec_path = tmp_path / "bad-spec.json"
            bad_spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(bad_spec_path, tmp_path / "out")

    def test_renderer_rejects_unregistered_scale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            receipt_path = _write_passing_path_scale_receipt(tmp_path)
            spec_path = _authorized_spec(tmp_path, receipt_path)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["scale"] = 9.9
            bad_spec_path = tmp_path / "bad-scale-spec.json"
            bad_spec_path.write_text(json.dumps(spec), encoding="utf-8")
            with self.assertRaises(renderer.G3ScriptedRenderError):
                renderer.render(bad_spec_path, tmp_path / "out")


if __name__ == "__main__":
    unittest.main()
