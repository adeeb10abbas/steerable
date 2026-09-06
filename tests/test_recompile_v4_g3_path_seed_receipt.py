"""Tests for offline object_pair G3 path-seed receipt recompilation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.model_blind_g3 import (
    HORIZONTAL_GOALS,
    PATH_SAMPLE_INTERVAL_S,
    canonical_json_bytes,
    compile_path_seed_receipt,
    expected_path_check_keys,
    sha256_file,
    validate_path_seed_receipt,
    validate_plan_payload,
)

ROOT = Path(__file__).resolve().parents[1]
OBJECT_PAIR_PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/object_pair_g3_plan.candidate.json"
)


def _load_recompiler():
    spec = importlib.util.spec_from_file_location(
        "recompile_v4_g3_path_seed_receipt",
        ROOT / "tools/recompile_v4_g3_path_seed_receipt.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


recompiler = _load_recompiler()


def _evidence(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _passing_path_observations(*, suffix: str) -> list[dict[str, object]]:
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


def _legacy_goal_area_cases(
    *,
    low_fraction_goals: set[str] | None = None,
    empty_destination_goal: str | None = None,
) -> list[dict[str, object]]:
    low_fraction_goals = low_fraction_goals or set()
    cases: list[dict[str, object]] = []
    for goal in HORIZONTAL_GOALS:
        removed = 0.10 if goal in low_fraction_goals else 0.25
        destination_empty = goal == empty_destination_goal
        passes = (
            not destination_empty
            and goal not in low_fraction_goals
        )
        cases.append(
            {
                "relation": goal,
                "original_area_m2": 0.10,
                "destination_area_m2": 0.0 if destination_empty else 0.09,
                "shrinking_direction": True,
                "removed_area_fraction": removed,
                "minimum_shrinking_area_fraction": 0.20,
                "original_goal_empty": False,
                "destination_goal_empty": destination_empty,
                "passes_information_gate": passes,
            }
        )
    return cases


def _build_legacy_source_receipt(
    *,
    plan: dict[str, object],
    plan_receipt: dict[str, str],
    environment_seed: int,
    scale: float,
    low_fraction_goals: set[str] | None = None,
    empty_destination_goal: str | None = None,
) -> dict[str, object]:
    """Simulate a pre-correction receipt compiled under the old global gate."""
    legacy_goal_area_cases = _legacy_goal_area_cases(
        low_fraction_goals=low_fraction_goals,
        empty_destination_goal=empty_destination_goal,
    )
    compile_goal_area_cases = [
        {
            **case,
            "passes_information_gate": (
                not case["original_goal_empty"] and not case["destination_goal_empty"]
            ),
        }
        for case in legacy_goal_area_cases
    ]
    receipt = compile_path_seed_receipt(
        plan=plan,
        plan_receipt=plan_receipt,
        environment_seed=environment_seed,
        scale=scale,
        check_observations=_passing_path_observations(suffix=str(environment_seed)),
        goal_area_cases=compile_goal_area_cases,
    )
    receipt["goal_area_cases"] = legacy_goal_area_cases
    receipt["information_gate_passed"] = all(
        case["passes_information_gate"] for case in legacy_goal_area_cases
    )
    receipt["passed"] = (
        receipt["failed_check_count"] == 0 and receipt["information_gate_passed"]
    )
    receipt["runtime_identity"] = {
        "study_checkout": {"commit": "abc123"},
        "pod": "object-pair-g3-0",
    }
    receipt["artifacts"] = {
        "checks": {
            "left__original_sham": {
                "path": "artifacts/checks/left__original_sham.json",
                "sha256": "d" * 64,
                "bytes": 128,
            }
        }
    }
    return receipt


class RecompileG3PathSeedReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(OBJECT_PAIR_PLAN.read_text(encoding="utf-8"))
        validate_plan_payload(cls.plan)
        cls.plan_sha256 = sha256_file(OBJECT_PAIR_PLAN)
        cls.plan_receipt = {
            "path": str(OBJECT_PAIR_PLAN),
            "sha256": cls.plan_sha256,
        }
        cls.environment_seed = int(cls.plan["registered_env_seeds"][0])
        cls.scale = float(cls.plan["scale_selection"]["candidate_scales_descending"][-1])

    def test_low_fraction_nonempty_case_flips_false_to_true(self) -> None:
        source = _build_legacy_source_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=self.environment_seed,
            scale=self.scale,
            low_fraction_goals={"left", "front"},
        )
        self.assertFalse(source["information_gate_passed"])
        left_case = next(
            case for case in source["goal_area_cases"] if case["relation"] == "left"
        )
        self.assertFalse(left_case["passes_information_gate"])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.json"
            source_path.write_bytes(canonical_json_bytes(source))
            output_path = tmp_path / "recompiled.json"
            report = recompiler.recompile_receipt(
                source_receipt_path=source_path,
                target_plan_path=OBJECT_PAIR_PLAN,
                target_plan_sha256=self.plan_sha256,
                output_path=output_path,
            )
            recompiled = json.loads(output_path.read_text(encoding="utf-8"))
            validate_path_seed_receipt(recompiled, plan=self.plan)

            self.assertTrue(report["information_gate_passed"])
            self.assertTrue(report["passed"])
            self.assertFalse(report["simulator_rerun"])
            recompiled_left = next(
                case
                for case in recompiled["goal_area_cases"]
                if case["relation"] == "left"
            )
            self.assertTrue(recompiled_left["passes_information_gate"])
            self.assertEqual(
                recompiler._goal_area_evidence(left_case),
                recompiler._goal_area_evidence(recompiled_left),
            )

    def test_empty_destination_remains_false(self) -> None:
        source = _build_legacy_source_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=self.environment_seed,
            scale=self.scale,
            low_fraction_goals={"left", "right", "front", "behind"},
            empty_destination_goal="behind",
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.json"
            source_path.write_bytes(canonical_json_bytes(source))
            output_path = tmp_path / "recompiled.json"
            recompiler.recompile_receipt(
                source_receipt_path=source_path,
                target_plan_path=OBJECT_PAIR_PLAN,
                target_plan_sha256=self.plan_sha256,
                output_path=output_path,
            )
            recompiled = json.loads(output_path.read_text(encoding="utf-8"))
            behind = next(
                case
                for case in recompiled["goal_area_cases"]
                if case["relation"] == "behind"
            )
            self.assertTrue(behind["destination_goal_empty"])
            self.assertFalse(behind["passes_information_gate"])
            self.assertFalse(recompiled["information_gate_passed"])

    def test_path_checks_are_unchanged(self) -> None:
        source = _build_legacy_source_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=self.environment_seed,
            scale=self.scale,
            low_fraction_goals=set(HORIZONTAL_GOALS),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.json"
            source_path.write_bytes(canonical_json_bytes(source))
            output_path = tmp_path / "recompiled.json"
            recompiler.recompile_receipt(
                source_receipt_path=source_path,
                target_plan_path=OBJECT_PAIR_PLAN,
                target_plan_sha256=self.plan_sha256,
                output_path=output_path,
            )
            recompiled = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(source["checks"], recompiled["checks"])
            self.assertEqual(
                source["runtime_identity"],
                recompiled["runtime_identity"],
            )
            self.assertEqual(source["artifacts"], recompiled["artifacts"])

    def test_derivation_binds_source_receipt_hash(self) -> None:
        source = _build_legacy_source_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=self.environment_seed,
            scale=self.scale,
            low_fraction_goals={"left"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.json"
            source_bytes = canonical_json_bytes(source)
            source_path.write_bytes(source_bytes)
            output_path = tmp_path / "recompiled.json"
            recompiler.recompile_receipt(
                source_receipt_path=source_path,
                target_plan_path=OBJECT_PAIR_PLAN,
                target_plan_sha256=self.plan_sha256,
                output_path=output_path,
            )
            recompiled = json.loads(output_path.read_text(encoding="utf-8"))
            derivation = recompiled["derivation"]
            self.assertEqual(derivation["reason"], recompiler.DERIVATION_REASON)
            self.assertFalse(derivation["simulator_rerun"])
            self.assertEqual(
                derivation["source_receipt"]["sha256"],
                hashlib.sha256(source_bytes).hexdigest(),
            )
            self.assertEqual(
                derivation["source_receipt"]["bytes"],
                len(source_bytes),
            )
            self.assertEqual(
                recompiled["plan_receipt"]["sha256"],
                self.plan_sha256,
            )

    def test_write_once_output(self) -> None:
        source = _build_legacy_source_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=self.environment_seed,
            scale=self.scale,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "source.json"
            source_path.write_bytes(canonical_json_bytes(source))
            output_path = tmp_path / "recompiled.json"
            recompiler.recompile_receipt(
                source_receipt_path=source_path,
                target_plan_path=OBJECT_PAIR_PLAN,
                target_plan_sha256=self.plan_sha256,
                output_path=output_path,
            )
            with self.assertRaises(FileExistsError):
                recompiler.recompile_receipt(
                    source_receipt_path=source_path,
                    target_plan_path=OBJECT_PAIR_PLAN,
                    target_plan_sha256=self.plan_sha256,
                    output_path=output_path,
                )


if __name__ == "__main__":
    unittest.main()
