"""Tests for the formula-closed horizontal model-blind G3 plan and receipts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.model_blind_g3 import (
    G3GateError,
    HORIZONTAL_GOALS,
    HORIZONTAL_PATH_CHECKS_PER_SEED,
    PATH_SCENARIOS,
    PATH_SAMPLE_INTERVAL_S,
    compile_path_seed_receipt,
    compile_scripted_check_receipt,
    expected_path_check_keys,
    validate_path_seed_receipt,
    validate_plan_payload,
    validate_scripted_check_receipt,
    build_counterbalance_index,
    select_extreme_reset_seeds,
)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
)
PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json"
)
QUEUE = ROOT / "artifacts/online_correction_v4/queue.jsonl"
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
MOTION = ROOT / "artifacts/online_correction_v4/motion_manifest.json"
G2_AGGREGATE = (
    ROOT
    / "artifacts/online_correction_v4/qualification/20260905_horizontal_g2_aggregate.json"
)
FIXTURE_PLAN_CASES = {
    "vertical": {
        "registry": ROOT
        / "artifacts/online_correction_v4/setup/vertical_reset_registry.candidate.json",
        "g2": ROOT
        / "artifacts/online_correction_v4/qualification/"
        "20260906_vertical_g2_aggregate_g2c5q20260906c.json",
        "plan": ROOT
        / "artifacts/online_correction_v4/setup/vertical_g3_plan.candidate.json",
        "goals": ["above", "below"],
        "path_checks": 768,
        "scripted_checks": 56,
    },
    "containment": {
        "registry": ROOT
        / "artifacts/online_correction_v4/setup/"
        "containment_reset_registry.candidate.json",
        "g2": ROOT
        / "artifacts/online_correction_v4/qualification/"
        "20260906_containment_g2_aggregate_g2c6q20260906c.json",
        "plan": ROOT
        / "artifacts/online_correction_v4/setup/"
        "containment_g3_plan.candidate.json",
        "goals": ["inside"],
        "path_checks": 384,
        "scripted_checks": 28,
    },
}

BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_v4_horizontal_g3_plan",
    ROOT / "tools/build_v4_horizontal_g3_plan.py",
)
builder = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(builder)


def _evidence(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _passing_path_observation(
    *,
    goal: str,
    scenario: str,
    suffix: str = "a",
) -> dict[str, object]:
    measured = f"evidence/measured/{goal}_{scenario}_{suffix}.json".encode("utf-8")
    reference = f"evidence/reference/{goal}_{scenario}_{suffix}.json".encode("utf-8")
    return {
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


def _full_path_observations(*, failing_key: tuple[str, str] | None = None) -> list[dict[str, object]]:
    observations: list[dict[str, object]] = []
    for goal, scenario in expected_path_check_keys():
        obs = _passing_path_observation(goal=goal, scenario=scenario)
        if failing_key == (goal, scenario):
            obs["collision_free"] = False
            obs["reasons"] = ["collision detected"]
        observations.append(obs)
    return observations


def _goal_area_cases(*, failing_goal: str | None = None) -> list[dict[str, object]]:
    return [
        {
            "relation": goal,
            "original_area_m2": 0.10,
            "destination_area_m2": 0.075,
            "shrinking_direction": True,
            "removed_area_fraction": 0.25 if goal != failing_goal else 0.10,
            "minimum_shrinking_area_fraction": 0.20,
            "original_goal_empty": False,
            "destination_goal_empty": False,
            "passes_information_gate": goal != failing_goal,
        }
        for goal in HORIZONTAL_GOALS
    ]


class ModelBlindG3PlanTests(unittest.TestCase):
    def test_frozen_candidate_counts_and_direction_balance(self) -> None:
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        validate_plan_payload(plan)
        self.assertEqual(plan["path_sweep"]["checks_per_scale"], 3072)
        self.assertEqual(
            plan["path_sweep"]["maximum_checks_across_scale_ladder"], 15360
        )
        self.assertEqual(
            plan["scripted_controller"]["checks_per_final_geometry_candidate"],
            112,
        )
        self.assertEqual(plan["plan_status"], "ready_for_live_g3_execution")
        self.assertTrue(plan["g2_prerequisite"]["passed"])
        self.assertTrue(plan["g2_prerequisite"]["axis_review_passed"])
        self.assertEqual(
            plan["source_identity"]["g2_aggregate"]["sha256"],
            plan["g2_prerequisite"]["receipt"]["sha256"],
        )
        for seed, directions in plan[
            "direction_task_coefficients_by_env_seed"
        ].items():
            self.assertEqual(directions["left"], directions["right"], seed)
            self.assertEqual(directions["front"], directions["behind"], seed)

    def test_extreme_selection_is_deterministic_and_state_balanced(self) -> None:
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
        rows = [
            json.loads(line)
            for line in QUEUE.read_text(encoding="utf-8").splitlines()
        ]
        seeds = sorted(int(seed) for seed in registry["resets_by_env_seed"])
        counterbalance = build_counterbalance_index(
            rows, expected_env_seeds=seeds
        )
        selected = select_extreme_reset_seeds(
            resets_by_env_seed=registry["resets_by_env_seed"],
            counterbalance_by_env_seed=counterbalance,
        )
        self.assertEqual(len(selected), 9)
        self.assertEqual(selected[0], 2100000000)
        states = [counterbalance[seed]["state_index"] for seed in selected[1:]]
        self.assertEqual({state: states.count(state) for state in range(4)}, {
            0: 2,
            1: 2,
            2: 2,
            3: 2,
        })

    def test_builder_reproduces_frozen_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rebuilt = Path(tmp) / PLAN.name
            report = builder.build(
                campaign_path=CAMPAIGN,
                queue_path=QUEUE,
                motion_path=MOTION,
                registry_path=REGISTRY,
                g2_aggregate_path=G2_AGGREGATE,
                output_path=rebuilt,
            )
            self.assertEqual(rebuilt.read_bytes(), PLAN.read_bytes())
            self.assertEqual(report["registered_reset_count"], 128)

    def test_fixture_specific_plans_preserve_registered_goal_spaces(self) -> None:
        for fixture_id, case in FIXTURE_PLAN_CASES.items():
            with self.subTest(fixture_id=fixture_id), tempfile.TemporaryDirectory() as tmp:
                rebuilt = Path(tmp) / Path(case["plan"]).name
                report = builder.build(
                    campaign_path=CAMPAIGN,
                    queue_path=QUEUE,
                    motion_path=MOTION,
                    registry_path=Path(case["registry"]),
                    g2_aggregate_path=Path(case["g2"]),
                    output_path=rebuilt,
                    fixture_id=fixture_id,
                )
                plan = json.loads(rebuilt.read_text(encoding="utf-8"))
                validate_plan_payload(plan)
                self.assertEqual(plan["path_sweep"]["goals"], case["goals"])
                self.assertEqual(
                    plan["path_sweep"]["checks_per_scale"],
                    case["path_checks"],
                )
                self.assertEqual(
                    plan["scripted_controller"][
                        "checks_per_final_geometry_candidate"
                    ],
                    case["scripted_checks"],
                )
                self.assertEqual(rebuilt.read_bytes(), Path(case["plan"]).read_bytes())
                self.assertEqual(report["registered_reset_count"], 64)


class ModelBlindG3PathReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        validate_plan_payload(cls.plan)
        cls.plan_receipt = {
            "path": str(PLAN.relative_to(ROOT)),
            "sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        }

    def test_expected_path_check_order_has_exactly_24_cases(self) -> None:
        keys = expected_path_check_keys()
        self.assertEqual(len(keys), 24)
        self.assertEqual(HORIZONTAL_PATH_CHECKS_PER_SEED, 24)
        self.assertEqual(
            keys,
            tuple((goal, scenario) for goal in HORIZONTAL_GOALS for scenario in PATH_SCENARIOS),
        )

    def test_compile_and_validate_passing_path_seed_receipt(self) -> None:
        receipt = compile_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=2100000000,
            scale=1.0,
            check_observations=_full_path_observations(),
            goal_area_cases=_goal_area_cases(),
        )
        validate_path_seed_receipt(receipt, plan=self.plan)
        self.assertEqual(receipt["schema_version"], "v4-horizontal-g3-path-seed-receipt-v1")
        self.assertEqual(receipt["check_count"], 24)
        self.assertEqual(len(receipt["checks"]), 24)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["passed_check_count"], 24)
        self.assertEqual(receipt["failed_check_count"], 0)
        self.assertEqual(receipt["checks"][0]["goal"], "left")
        self.assertEqual(receipt["checks"][0]["scenario"], "original_sham")
        self.assertEqual(receipt["checks"][-1]["goal"], "behind")
        self.assertEqual(receipt["checks"][-1]["scenario"], "reversal")
        self.assertEqual(receipt["displacement_m"], 0.12)
        self.assertEqual(
            receipt["direction_task_coefficients_by_goal"]["left"],
            self.plan["direction_task_coefficients_by_env_seed"]["2100000000"]["left"],
        )

    def test_compile_records_partial_failure(self) -> None:
        receipt = compile_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=2100000004,
            scale=0.75,
            check_observations=_full_path_observations(
                failing_key=("front", "fast_drift")
            ),
            goal_area_cases=_goal_area_cases(),
        )
        validate_path_seed_receipt(receipt, plan=self.plan)
        self.assertFalse(receipt["passed"])
        self.assertEqual(receipt["failed_check_count"], 1)
        failed = next(check for check in receipt["checks"] if not check["passed"])
        self.assertEqual(failed["goal"], "front")
        self.assertEqual(failed["scenario"], "fast_drift")
        self.assertEqual(failed["reasons"], ["collision detected"])

    def test_information_gate_failure_rejects_otherwise_clear_path(self) -> None:
        receipt = compile_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=2100000000,
            scale=1.0,
            check_observations=_full_path_observations(),
            goal_area_cases=_goal_area_cases(failing_goal="left"),
        )
        validate_path_seed_receipt(receipt, plan=self.plan)
        self.assertEqual(receipt["failed_check_count"], 0)
        self.assertFalse(receipt["information_gate_passed"])
        self.assertFalse(receipt["passed"])

    def test_rejects_wrong_observation_count(self) -> None:
        with self.assertRaises(G3GateError):
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=2100000000,
                scale=1.0,
                check_observations=_full_path_observations()[:23],
                goal_area_cases=_goal_area_cases(),
            )

    def test_rejects_unregistered_scale(self) -> None:
        with self.assertRaises(G3GateError):
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=2100000000,
                scale=1.25,
                check_observations=_full_path_observations(),
                goal_area_cases=_goal_area_cases(),
            )

    def test_rejects_non_finite_numeric(self) -> None:
        observations = _full_path_observations()
        observations[0]["planned_duration_s"] = float("nan")
        with self.assertRaises(G3GateError):
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=2100000000,
                scale=1.0,
                check_observations=observations,
                goal_area_cases=_goal_area_cases(),
            )

    def test_rejects_sample_interval_above_cap(self) -> None:
        observations = _full_path_observations()
        observations[0]["sample_interval_s"] = PATH_SAMPLE_INTERVAL_S + 0.001
        with self.assertRaises(G3GateError):
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=2100000000,
                scale=1.0,
                check_observations=observations,
                goal_area_cases=_goal_area_cases(),
            )

    def test_rejects_malformed_evidence_hash(self) -> None:
        observations = _full_path_observations()
        measured = observations[0]["measured_pose_evidence"]
        assert isinstance(measured, dict)
        measured["sha256"] = "not-a-hash"
        with self.assertRaises(G3GateError):
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=2100000000,
                scale=1.0,
                check_observations=observations,
                goal_area_cases=_goal_area_cases(),
            )

    def test_rejects_declared_passed_disagreeing_with_predicates(self) -> None:
        observations = _full_path_observations()
        observations[0]["collision_free"] = False
        observations[0]["reasons"] = ["collision"]
        observations[0]["passed"] = True
        with self.assertRaises(G3GateError):
            compile_path_seed_receipt(
                plan=self.plan,
                plan_receipt=self.plan_receipt,
                environment_seed=2100000000,
                scale=1.0,
                check_observations=observations,
                goal_area_cases=_goal_area_cases(),
            )

    def test_validate_rejects_out_of_order_checks(self) -> None:
        receipt = compile_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=2100000000,
            scale=1.0,
            check_observations=_full_path_observations(),
            goal_area_cases=_goal_area_cases(),
        )
        swapped = list(receipt["checks"])
        swapped[0], swapped[1] = swapped[1], swapped[0]
        bad = {**receipt, "checks": swapped}
        with self.assertRaises(G3GateError):
            validate_path_seed_receipt(bad, plan=self.plan)

    def test_validate_rejects_plan_binding_mismatch(self) -> None:
        receipt = compile_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=2100000000,
            scale=1.0,
            check_observations=_full_path_observations(),
            goal_area_cases=_goal_area_cases(),
        )
        bad = {**receipt, "scale": 2.0, "displacement_m": 0.24}
        with self.assertRaises(G3GateError):
            validate_path_seed_receipt(bad, plan=self.plan)

    def test_validate_rejects_missing_evidence(self) -> None:
        receipt = compile_path_seed_receipt(
            plan=self.plan,
            plan_receipt=self.plan_receipt,
            environment_seed=2100000000,
            scale=1.0,
            check_observations=_full_path_observations(),
            goal_area_cases=_goal_area_cases(),
        )
        checks = list(receipt["checks"])
        first = dict(checks[0])
        del first["measured_pose_evidence"]
        checks[0] = first
        bad = {**receipt, "checks": checks}
        with self.assertRaises(G3GateError):
            validate_path_seed_receipt(bad)


class ModelBlindG3ScriptedReceiptTests(unittest.TestCase):
    def test_compile_and_validate_passing_stationary_receipt(self) -> None:
        payload = b"scripted-stationary-left-original"
        receipt = compile_scripted_check_receipt(
            check_kind="stationary",
            environment_seed=2100000000,
            goal="left",
            reference_position="original",
            scale=1.0,
            displacement_m=0.12,
            observation={
                "grasped": True,
                "transported": True,
                "released": True,
                "stably_placed": True,
                "goal_satisfied": True,
                "evidence": _evidence(
                    "artifacts/g3/scripted/stationary_left_original.json",
                    payload,
                ),
                "reasons": [],
            },
        )
        validate_scripted_check_receipt(receipt)
        self.assertEqual(
            receipt["schema_version"],
            "v4-horizontal-g3-scripted-check-receipt-v1",
        )
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["model_request_count"], 0)
        self.assertEqual(receipt["behavioral_episode_count"], 0)

    def test_compile_and_validate_moving_failure(self) -> None:
        receipt = compile_scripted_check_receipt(
            check_kind="moving",
            environment_seed=2100000098,
            goal="behind",
            reference_position="endpoint",
            scale=0.5,
            displacement_m=0.06,
            observation={
                "grasped": True,
                "transported": True,
                "released": False,
                "stably_placed": False,
                "goal_satisfied": False,
                "evidence": _evidence(
                    "artifacts/g3/scripted/moving_behind_endpoint.json",
                    b"failed-release",
                ),
                "reasons": ["release not stable"],
            },
        )
        validate_scripted_check_receipt(receipt)
        self.assertFalse(receipt["passed"])

    def test_rejects_invalid_check_kind(self) -> None:
        with self.assertRaises(G3GateError):
            compile_scripted_check_receipt(
                check_kind="drifting",
                environment_seed=2100000000,
                goal="left",
                reference_position="original",
                scale=1.0,
                displacement_m=0.12,
                observation={
                    "grasped": True,
                    "transported": True,
                    "released": True,
                    "stably_placed": True,
                    "goal_satisfied": True,
                    "evidence": _evidence("x.json", b"x"),
                    "reasons": [],
                },
            )

    def test_rejects_passed_inconsistent_with_stages(self) -> None:
        with self.assertRaises(G3GateError):
            compile_scripted_check_receipt(
                check_kind="stationary",
                environment_seed=2100000000,
                goal="left",
                reference_position="midpoint",
                scale=1.0,
                displacement_m=0.12,
                observation={
                    "grasped": True,
                    "transported": False,
                    "released": False,
                    "stably_placed": False,
                    "goal_satisfied": False,
                    "passed": True,
                    "evidence": _evidence("x.json", b"x"),
                    "reasons": ["transport failed"],
                },
            )

    def test_validate_rejects_nonzero_model_counts(self) -> None:
        receipt = compile_scripted_check_receipt(
            check_kind="stationary",
            environment_seed=2100000000,
            goal="right",
            reference_position="original",
            scale=1.0,
            displacement_m=0.12,
            observation={
                "grasped": True,
                "transported": True,
                "released": True,
                "stably_placed": True,
                "goal_satisfied": True,
                "evidence": _evidence("x.json", b"x"),
                "reasons": [],
            },
        )
        bad = {**receipt, "model_request_count": 1}
        with self.assertRaises(G3GateError):
            validate_scripted_check_receipt(bad)


if __name__ == "__main__":
    unittest.main()
