import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import v2a015_compilation as compilation  # noqa: E402
from compile_v2a015_cfg_comparison import _transition  # noqa: E402


class V2A015MetricTest(unittest.TestCase):
    def test_trajectory_quality_uses_3d_path_and_frozen_net_threshold(self):
        cube = np.array(
            [
                [0.0, 0.0, 0.0, 1, 0, 0, 0],
                [0.003, 0.004, 0.0, 1, 0, 0, 0],
                [0.006, 0.008, 0.0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        delta = cube[:, :3].copy()
        metrics = compilation.trajectory_quality(cube, delta, "left")
        self.assertAlmostEqual(metrics["cube_path_length_3d_m"], 0.01)
        self.assertAlmostEqual(metrics["cube_net_displacement_3d_m"], 0.01)
        self.assertAlmostEqual(metrics["cube_excess_path_ratio"], 1.0)
        self.assertAlmostEqual(
            metrics["cube_max_requested_lateral_excursion_from_initial_m"], 0.008
        )

        cube[-1, :3] = [0.0, 0.0, 0.0]
        null_metrics = compilation.trajectory_quality(cube, delta, "left")
        self.assertIsNone(null_metrics["cube_excess_path_ratio"])
        self.assertIn("0.01 m", null_metrics["cube_excess_path_ratio_null_reason"])

    def test_action_total_variation_is_successive_joint_l2(self):
        actions = np.zeros((3, 8), dtype=np.float64)
        actions[1, 0] = 3.0
        actions[1, 1] = 4.0
        actions[2, :2] = [3.0, 4.0]
        actions[2, 7] = 1.0
        metrics = compilation.action_quality(actions)
        self.assertEqual(metrics["joint_action_total_variation_l2"], 5.0)
        self.assertEqual(metrics["joint_action_mean_l2_per_transition"], 2.5)
        self.assertEqual(metrics["gripper_switch_count"], 1)

    def test_first_chunk_reports_joint_and_gripper_separately(self):
        left = np.zeros((4, 8), dtype=np.float64)
        right = np.zeros((4, 8), dtype=np.float64)
        right[:2, :7] = 1.0
        right[1, 7] = 1.0
        metrics = compilation.first_chunk_pair_metrics(left, right, horizon=2)
        self.assertEqual(metrics["executed_steps_compared"], 2)
        self.assertAlmostEqual(metrics["joint_only_rms_7d"], 1.0)
        self.assertAlmostEqual(metrics["legacy_all_8d_rms"], (15 / 16) ** 0.5)
        self.assertEqual(metrics["gripper_disagreement_steps"], 1)
        self.assertEqual(metrics["gripper_disagreement_fraction"], 0.5)

    def test_complete_grid_rejects_duplicate_cell(self):
        cells = []
        for seed in compilation.SEEDS:
            for relation, prompt in compilation.PROMPTS.items():
                cells.append(
                    {
                        "environment_seed": seed,
                        "sampling_seed": seed,
                        "requested_relation": relation,
                        "prompt": prompt,
                        "prompt_family": "direct_command",
                        "prompt_controller": "episode_static",
                        "oracle_actions": 0,
                        "dynamic_prompt_switches": 0,
                    }
                )
        compilation.validate_complete_grid(cells)
        cells[-1] = dict(cells[0])
        with self.assertRaisesRegex(RuntimeError, "six-cell"):
            compilation.validate_complete_grid(cells)

    def test_pair_balance_and_endpoint_metrics_keep_signs_distinct(self):
        episodes = []
        actions = {}
        for seed in compilation.SEEDS:
            fingerprint = f"fingerprint-{seed}"
            for relation, endpoint, margin in (
                ("left", -0.1, 0.1),
                ("right", 0.4, 0.4),
            ):
                episodes.append(
                    {
                        "environment_seed": seed,
                        "requested_relation": relation,
                        "requested_success": True,
                        "requested_signed_final_margin_m": margin,
                        "final_lateral_display_m": endpoint,
                        "physical_initial_state_sha256": fingerprint,
                    }
                )
                actions[(seed, relation)] = np.zeros((8, 8), dtype=np.float32)
        pairs = compilation.build_pairs(episodes, actions, horizon=8)
        self.assertEqual(len(pairs), 3)
        self.assertAlmostEqual(
            pairs[0]["seed_balance_gap_right_minus_left_margin_m"], 0.3
        )
        self.assertAlmostEqual(pairs[0]["seed_absolute_direction_imbalance_m"], 0.3)
        self.assertAlmostEqual(pairs[0]["seed_weaker_side_margin_m"], 0.1)
        self.assertAlmostEqual(
            pairs[0]["endpoint_separation_right_minus_left_m"], 0.5
        )
        self.assertEqual(pairs[0]["endpoint_ordering"], "aligned")

    def test_preflight_style_invalid_rows_are_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_text(
                json.dumps(
                    {
                        "invalid_attempts": [
                            {
                                "stage": "server_import",
                                "error": "missing dependency",
                                "effect": "exited before policy request or behavior; excluded from every behavioral denominator",
                            }
                        ]
                    }
                )
            )
            summary = compilation.ledger_summary([path], invalid=True)
            self.assertEqual(summary["row_count"], 1)
            self.assertEqual(summary["behavioral_denominator_policy"], "excluded")

    def test_success_transition_labels_are_exact(self):
        self.assertEqual(_transition(False, True), "improved_failure_to_success")
        self.assertEqual(_transition(True, False), "regressed_success_to_failure")
        self.assertEqual(_transition(True, True), "unchanged_success")
        self.assertEqual(_transition(False, False), "unchanged_failure")


if __name__ == "__main__":
    unittest.main()
