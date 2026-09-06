"""Tests for horizontal G3 aggregate evidence compilation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.model_blind_g3 import (
    G3GateError,
    PATH_SCALE_RECEIPT_SCHEMA,
    canonical_json_bytes,
    compile_g3_aggregate_receipt,
    compile_path_scale_receipt,
    compile_path_seed_receipt,
    compile_scripted_check_receipt,
    expected_path_check_keys,
    expected_scripted_check_keys,
    sha256_file,
    validate_g3_aggregate_receipt,
    validate_plan_payload,
)

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json"
)


def _load_compiler():
    spec = importlib.util.spec_from_file_location(
        "compile_v4_horizontal_g3_aggregate",
        ROOT / "tools/compile_v4_horizontal_g3_aggregate.py",
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
    from experiments.online_correction_v4.model_blind_g3 import PATH_SAMPLE_INTERVAL_S

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


def _path_scale_receipt(
    *,
    plan: dict[str, object],
    plan_receipt: dict[str, str],
    scale: float,
    failing_seed: int | None = None,
) -> dict[str, object]:
    seed_receipts: list[dict[str, object]] = []
    for seed in plan["registered_env_seeds"]:
        seed_int = int(seed)
        observations = _passing_path_observations(suffix=str(seed))
        if failing_seed is not None and seed_int == failing_seed:
            observations[0]["collision_free"] = False
            observations[0]["reasons"] = ["collision detected"]
        seed_receipts.append(
            compile_path_seed_receipt(
                plan=plan,
                plan_receipt=plan_receipt,
                environment_seed=seed_int,
                scale=scale,
                check_observations=observations,
                goal_area_cases=_goal_area_cases(),
            )
        )
    return compile_path_scale_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        scale=scale,
        path_seed_receipts=seed_receipts,
    )


def _passing_scripted_receipts(*, scale: float, displacement_m: float) -> list[dict]:
    receipts: list[dict] = []
    for check_kind, environment_seed, goal, reference_position in expected_scripted_check_keys(
        json.loads(PLAN.read_text(encoding="utf-8"))
    ):
        payload = f"{check_kind}-{environment_seed}-{goal}-{reference_position}".encode(
            "utf-8"
        )
        receipts.append(
            compile_scripted_check_receipt(
                check_kind=check_kind,
                environment_seed=environment_seed,
                goal=goal,
                reference_position=reference_position,
                scale=scale,
                displacement_m=displacement_m,
                observation={
                    "grasped": True,
                    "transported": True,
                    "released": True,
                    "stably_placed": True,
                    "goal_satisfied": True,
                    "evidence": _evidence(
                        f"artifacts/g3/scripted/{check_kind}_{goal}_{reference_position}.json",
                        payload,
                    ),
                    "reasons": [],
                },
            )
        )
    return receipts


class CompileHorizontalG3AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        validate_plan_payload(cls.plan)
        cls.plan_receipt = {
            "path": str(PLAN.relative_to(ROOT)),
            "sha256": sha256_file(PLAN),
        }

    def test_lower_scale_without_higher_executed_raises(self) -> None:
        passing = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.0,
        )
        with self.assertRaises(G3GateError):
            compile_g3_aggregate_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                path_scale_receipts=[passing],
            )

    def test_no_passing_scale_emits_pending_without_claiming_pass(self) -> None:
        rejected = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=2.0,
            failing_seed=int(self.plan["registered_env_seeds"][0]),
        )
        report = compile_g3_aggregate_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            path_scale_receipts=[rejected],
        )
        validate_g3_aggregate_receipt(report, plan=self.plan)
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "blocked")
        self.assertIsNone(report["selected_scale"])

    def test_wrong_scripted_key_count_fails(self) -> None:
        rejected_high = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=2.0,
            failing_seed=int(self.plan["registered_env_seeds"][0]),
        )
        passing = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.5,
        )
        scripted = _passing_scripted_receipts(scale=1.5, displacement_m=0.18)
        scripted = scripted[:-1]
        report = compile_g3_aggregate_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            path_scale_receipts=[rejected_high, passing],
            scripted_check_receipts=scripted,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["status"], "blocked_incomplete")
        self.assertEqual(report["selected_scale"], 1.5)
        self.assertEqual(len(report["missing_scripted_check_keys"]), 1)

    def test_nonzero_model_count_rejected(self) -> None:
        scripted = _passing_scripted_receipts(scale=1.0, displacement_m=0.12)[0]
        bad = {**scripted, "model_request_count": 1}
        rejected_high = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=2.0,
            failing_seed=int(self.plan["registered_env_seeds"][0]),
        )
        rejected_mid = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.5,
            failing_seed=int(self.plan["registered_env_seeds"][0]),
        )
        passing = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.0,
        )
        with self.assertRaises(G3GateError):
            compile_g3_aggregate_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                path_scale_receipts=[rejected_high, rejected_mid, passing],
                scripted_check_receipts=[bad],
            )

    def test_complete_aggregate_passes(self) -> None:
        rejected_high = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=2.0,
            failing_seed=int(self.plan["registered_env_seeds"][0]),
        )
        passing = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.5,
        )
        scripted = _passing_scripted_receipts(scale=1.5, displacement_m=0.18)
        report = compile_g3_aggregate_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            path_scale_receipts=[rejected_high, passing],
            scripted_check_receipts=scripted,
        )
        validate_g3_aggregate_receipt(report, plan=self.plan)
        self.assertTrue(report["passed"])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["selected_scale"], 1.5)
        self.assertEqual(report["observed_scripted_check_count"], 112)
        self.assertEqual(report["scripted_failed_check_count"], 0)
        self.assertIn("G4 preparation", report["release_boundary"])

    def test_cli_compiles_aggregate(self) -> None:
        rejected_high = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=2.0,
            failing_seed=int(self.plan["registered_env_seeds"][0]),
        )
        passing = _path_scale_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            scale=1.5,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path_scale_root = root / "path-scale"
            path_scale_root.mkdir()
            for index, receipt in enumerate((rejected_high, passing)):
                path = path_scale_root / f"g3_path_scale_receipt_{index}.json"
                path.write_bytes(canonical_json_bytes(receipt))
            scripted_root = root / "scripted"
            scripted_root.mkdir()
            for index, receipt in enumerate(
                _passing_scripted_receipts(scale=1.5, displacement_m=0.18)
            ):
                directory = scripted_root / f"check-{index}"
                directory.mkdir()
                path = directory / "g3_scripted_check_receipt.json"
                path.write_bytes(canonical_json_bytes(receipt))
            output = root / "aggregate.json"
            report = compiler.compile_receipts(
                plan_path=PLAN,
                path_scale_receipts_root=path_scale_root,
                scripted_receipts_root=scripted_root,
                output_path=output,
            )
            self.assertTrue(report["passed"])
            self.assertTrue(output.is_file())

    def test_discovers_runner_receipt_directory_without_seed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipts = root / "run" / "episodes" / "receipts"
            receipts.mkdir(parents=True)
            check = receipts / "left__original.json"
            check.write_text("{}\n", encoding="utf-8")
            (receipts.parent / "g3_scripted_seed_receipt.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            self.assertEqual(compiler._discover_scripted_receipts(root), [check.resolve()])


if __name__ == "__main__":
    unittest.main()
