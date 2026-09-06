"""Tests for the formula-closed horizontal model-blind G3 plan."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.model_blind_g3 import (
    build_counterbalance_index,
    select_extreme_reset_seeds,
    validate_plan_payload,
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

BUILDER_SPEC = importlib.util.spec_from_file_location(
    "build_v4_horizontal_g3_plan",
    ROOT / "tools/build_v4_horizontal_g3_plan.py",
)
builder = importlib.util.module_from_spec(BUILDER_SPEC)
assert BUILDER_SPEC.loader is not None
BUILDER_SPEC.loader.exec_module(builder)


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

if __name__ == "__main__":
    unittest.main()
