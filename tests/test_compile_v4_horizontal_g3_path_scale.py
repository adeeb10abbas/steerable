"""Tests for horizontal G3 path-scale evidence compilation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.model_blind_g3 import (
    G3GateError,
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


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "compile_v4_horizontal_g3_path_scale",
        ROOT / "tools/compile_v4_horizontal_g3_path_scale.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


compiler = _load_compiler()


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


class CompileHorizontalG3PathScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        validate_plan_payload(cls.plan)
        cls.plan_receipt = {
            "path": str(PLAN.relative_to(ROOT)),
            "sha256": sha256_file(PLAN),
        }

    def test_missing_seed_reports_failure_without_pass(self) -> None:
        seeds = list(self.plan["registered_env_seeds"])[:-1]
        receipts = [
            _build_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=int(seed),
                scale=1.0,
            )
            for seed in seeds
        ]
        report = compile_path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.0,
            path_seed_receipts=receipts,
        )
        validate_path_scale_receipt(report, plan=self.plan)
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["observed_seed_count"], 127)
        self.assertEqual(len(report["missing_env_seeds"]), 1)
        self.assertEqual(report["expected_path_check_count"], 3072)

    def test_complete_scale_passes_all_3072_checks(self) -> None:
        receipts = [
            _build_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=int(seed),
                scale=1.0,
            )
            for seed in self.plan["registered_env_seeds"]
        ]
        report = compile_path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.0,
            path_seed_receipts=receipts,
        )
        validate_path_scale_receipt(report, plan=self.plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["passed_path_check_count"], 3072)
        self.assertEqual(report["failed_path_check_count"], 0)
        self.assertEqual(report["model_request_count"], 0)

    def test_failed_seed_preserves_scientific_reasons(self) -> None:
        seeds = list(self.plan["registered_env_seeds"])
        receipts = [
            _build_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=int(seed),
                scale=1.0,
                passed=int(seed) != int(seeds[0]),
            )
            for seed in seeds
        ]
        report = compile_path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.0,
            path_seed_receipts=receipts,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(len(report["failed_env_seeds"]), 1)
        path_failures = [
            item
            for item in report["scientific_failure_summary"]
            if item.get("failure_kind") == "path_check"
        ]
        self.assertEqual(path_failures[0]["reasons"], ["collision detected"])

    def test_rejects_nonzero_model_request_count(self) -> None:
        receipt = _build_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=int(self.plan["registered_env_seeds"][0]),
            scale=1.0,
        )
        bad = {**receipt, "model_request_count": 1}
        with self.assertRaises(G3GateError):
            compile_path_scale_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                scale=1.0,
                path_seed_receipts=[bad],
            )

    def test_cli_compiles_from_receipt_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in self.plan["registered_env_seeds"]:
                directory = root / f"seed-{seed}"
                directory.mkdir()
                receipt = _build_path_seed_receipt(
                    plan=self.plan,
                    plan_receipt=self.plan_receipt,
                    environment_seed=int(seed),
                    scale=0.75,
                )
                path = directory / "g3_path_seed_receipt.json"
                path.write_bytes(canonical_json_bytes(receipt))
            output = root / "path-scale.json"
            report = compiler.compile_receipts(
                plan_path=PLAN,
                scale=0.75,
                receipts_root=root,
                output_path=output,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
