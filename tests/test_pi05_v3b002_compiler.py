#!/usr/bin/env python3
"""Focused synthetic tests for pi0.5 V3-B002 diagnostics/statistics."""

from __future__ import annotations

import unittest

import numpy as np

from experiments.v3.pi05_phase_b.compiler import (
    analyze_pairs,
    exact_layout_swap_permutation,
    exact_sign_test,
)
from experiments.v3.pi05_phase_b.contract import SEEDS
from experiments.v3.pi05_phase_b.diagnostics import (
    DiagnosticError,
    attach_episode_diagnostics,
    derive_episode_diagnostics,
    derive_pair_diagnostics,
)


def _episode(*, seed: int, arm: str, relation: str, success: bool = True) -> dict:
    sign = 1.0 if relation == "left" else -1.0
    lateral = [0.0, 0.01 * sign, 0.04 * sign]
    steps = [
        {
            "action_step": index,
            "object_xyz": [0.1, value, 0.1],
            "reference_xyz": [0.0, 0.0, 0.1],
            "grippers_open": index == 2,
            "object_grabbed": index >= 1,
        }
        for index, value in enumerate(lateral)
    ]
    return {
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B002",
        "model_id": "pi05_current_stack_droid",
        "registered_cell_id": f"v3b002:pi05:seed{seed}:{arm}:{relation}",
        "pair_id": f"v3b002:pi05:seed{seed}",
        "environment_seed": seed,
        "requested_relation": relation,
        "requested_success": success,
        "failure_taxonomy": "correct" if success else "pick_failed",
        "phase_b_arm": arm,
        "initial_state_sha256": "a" * 64,
        "steps": steps,
        "actions_executed": 2,
        "measurements": {
            "signed_final_lateral_offset_m": lateral[-1],
            "final_requested_signed_margin_m": 0.04,
            "first_requested_entry_step": 1,
            "first_sustained_requested_entry_step": None,
            "first_contact_step": None,
            "first_contact_status": "instrumentation_unavailable",
            "first_contact_unavailable_reason": "RoboLab export has no physical contact stream",
        },
    }


class Pi05V3B002DiagnosticsTest(unittest.TestCase):
    def test_episode_diagnostics_use_true_grasp_and_raw_lateral_path(self) -> None:
        diagnostics = derive_episode_diagnostics(
            _episode(seed=9400, arm="control", relation="left")
        )
        self.assertEqual(diagnostics["grasp_step"], 1)
        self.assertEqual(diagnostics["grasp_source"], "retained_object_grabbed_boolean_stream")
        self.assertAlmostEqual(diagnostics["cumulative_lateral_path_m"], 0.04)
        self.assertAlmostEqual(diagnostics["peak_lateral_excursion_m"], 0.04)
        self.assertEqual(diagnostics["first_contact_status"], "instrumentation_unavailable")
        self.assertIsNone(diagnostics["time_to_first_contact_steps"])
        self.assertFalse(diagnostics["cone_entry_sustained"])
        self.assertIsNone(diagnostics["endpoint_shift_m"])
        self.assertIsNone(diagnostics["action_distinct"])
        retained = attach_episode_diagnostics(
            _episode(seed=9400, arm="control", relation="left")
        )
        self.assertEqual(retained["success"], retained["requested_success"])
        self.assertEqual(retained["failure_category"], retained["failure_taxonomy"])
        self.assertAlmostEqual(retained["requested_side_depth_m"], 0.04)

    def test_missing_object_grabbed_is_not_replaced_by_verified_pickup(self) -> None:
        record = _episode(seed=9400, arm="control", relation="left")
        del record["steps"][1]["object_grabbed"]
        with self.assertRaisesRegex(DiagnosticError, "not a grasp substitute"):
            derive_episode_diagnostics(record)

    def test_pair_diagnostics_are_separate_and_use_complete_common_prefix(self) -> None:
        left = _episode(seed=9400, arm="control", relation="left")
        right = _episode(seed=9400, arm="control", relation="right")
        pair = derive_pair_diagnostics(
            seed=9400,
            arm="control",
            left_record=left,
            right_record=right,
            left_actions=np.zeros((2, 8), dtype=np.float32),
            right_actions=np.ones((2, 8), dtype=np.float32),
        )
        self.assertAlmostEqual(pair["endpoint_redirection_D_m"], 0.08)
        self.assertTrue(pair["executed_actions_distinct"])
        self.assertEqual(pair["common_prefix_action_count"], 2)
        self.assertAlmostEqual(pair["common_prefix_action_rms"], 1.0)


class Pi05V3B002StatisticsTest(unittest.TestCase):
    def test_exact_sign_test_counts_ties_explicitly(self) -> None:
        result = exact_sign_test([1.0, 2.0, -1.0, 0.0])
        self.assertEqual(result["positive"], 2)
        self.assertEqual(result["negative"], 1)
        self.assertEqual(result["ties"], 1)
        self.assertEqual(result["effective_n"], 3)
        self.assertEqual(result["p_value"], 1.0)

    def test_exact_layout_swap_uses_all_registered_seed_labelings(self) -> None:
        result = exact_layout_swap_permutation([-2] * 27)
        self.assertEqual(result["total_permutations"], 2**27)
        self.assertEqual(result["extreme_permutations"], 2)
        self.assertEqual(result["p_value"], 2 / 2**27)
        zeros = exact_layout_swap_permutation([0] * 27)
        self.assertEqual(zeros["total_permutations"], 2**27)
        self.assertEqual(zeros["p_value"], 1.0)

    def test_complete_27_seed_analysis_reports_registered_hypotheses(self) -> None:
        pairs = []
        for seed in SEEDS:
            pairs.extend(
                [
                    {
                        "seed": seed,
                        "arm": "control",
                        "endpoint_redirection_D_m": 0.10,
                        "requested_side_depth_contrast_B_m": 0.20,
                        "left_success": False,
                        "right_success": True,
                    },
                    {
                        "seed": seed,
                        "arm": "position_mirrored",
                        "endpoint_redirection_D_m": 0.10,
                        "requested_side_depth_contrast_B_m": -0.05,
                        "left_success": True,
                        "right_success": False,
                    },
                ]
            )
        result = analyze_pairs(
            pairs, bootstrap_replicates=10_000, bootstrap_seed=3_104_159
        )
        self.assertEqual(result["population"]["behavioral_episode_count"], 108)
        self.assertEqual(result["population"]["matched_left_right_pair_count"], 54)
        self.assertAlmostEqual(
            result["H1_endpoint_redirection"]["reflected_minus_control_interaction"]["mean_m"],
            0.0,
        )
        self.assertAlmostEqual(
            result["H2_requested_side_depth"]["reflected_minus_control_interaction"]["mean_m"],
            -0.25,
        )
        h3 = result["H3_binary_success"]
        self.assertEqual(
            h3["cell_success_table_2x2"],
            {
                "control": {
                    "left": {"successes": 0, "episodes": 27, "failures": 27},
                    "right": {"successes": 27, "episodes": 27, "failures": 0},
                },
                "position_mirrored": {
                    "left": {"successes": 27, "episodes": 27, "failures": 0},
                    "right": {"successes": 0, "episodes": 27, "failures": 27},
                },
            },
        )
        self.assertEqual(h3["per_seed_DiD_distribution"]["-2"], 27)
        self.assertEqual(h3["exact_permutation_test"]["p_value"], 2 / 2**27)
        self.assertEqual(len(result["seed_level"]), 27)


if __name__ == "__main__":
    unittest.main()
