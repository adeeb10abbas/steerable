"""Stopping-time controls must retain missingness and reject endpoint artifacts."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "analyze_stopping_controls.py"
SPEC = importlib.util.spec_from_file_location("stopping_controls", MODULE_PATH)
STOP = importlib.util.module_from_spec(SPEC) if MODULE_PATH.exists() else None
if STOP is not None:
    SPEC.loader.exec_module(STOP)


def episode(side, positions, *, termination=None, seed=1, arm="control"):
    """Positions are literal (step, cube_y, bowl_y) test observations."""
    end = max(p[0] for p in positions) if termination is None else termination
    return {
        "registered_cell_id": f"pair{seed}:{arm}:{side}",
        "pair_id": f"pair{seed}", "phase_b_arm": arm,
        "policy_seed": seed, "environment_seed": seed,
        "requested_relation": side, "behavioral_result_valid": True,
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "actions_executed": end, "requested_success": True,
        "right_censored": False, "initial_state_sha256": arm,
        "event_timeline": [{"event": "episode_end", "action_step": end}],
        "steps": [{"action_step": t, "object_xyz": [0.0, cube, 0.0],
                   "reference_xyz": [0.0, bowl, 0.0]} for t, cube, bowl in positions],
    }


class StoppingControlsTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(STOP, "The stopping-control analyzer must be implemented")

    def analyze(self, rows, checkpoints=(0, 1, 2, 3, 4, 5, 6), seeds=(1,)):
        return STOP.analyze_episodes(rows, seeds=seeds, arms=("control",), checkpoints=checkpoints)

    def test_goal_stopped_identical_path_does_not_imply_directional_response(self):
        # Same path visits LEFT at t=2 and RIGHT at t=4, with different stops.
        path = [(0, 0, 0), (1, 0.5, 0), (2, 1, 0), (3, 0, 0), (4, -1, 0)]
        result = self.analyze([episode("left", path[:3]), episode("right", path)])
        endpoint = result["analyses"]["terminal_endpoints"]["control"]
        common = result["analyses"]["pretermination_latest_common"]["control"]
        fixed = result["analyses"]["fixed_action_checkpoints"]["2"]["control"]
        self.assertEqual(endpoint["metrics"]["offset_separation_m"]["mean"], 2)
        self.assertEqual(common["metrics"]["offset_separation_m"]["mean"], 0)
        self.assertEqual(common["ties"], 1)
        self.assertEqual(fixed["metrics"]["offset_separation_m"]["mean"], 0)
        self.assertEqual(result["pair_records"][0]["common_action_step"], 1)

    def test_latest_shared_observation_excludes_both_terminal_samples(self):
        rows = [episode("left", [(0, 0, 0), (1, 2, 0), (3, 4, 1), (4, 99, 0)]),
                episode("right", [(0, 0, 0), (2, 1, 0), (3, -2, 0), (5, -99, 0), (6, -99, 0)])]
        result = self.analyze(rows)
        pair = result["pair_records"][0]
        self.assertEqual(pair["common_action_step"], 3)
        self.assertEqual(pair["offset_separation_m"], 5)
        self.assertEqual(pair["cube_y_separation_m"], 6)
        self.assertEqual(pair["bowl_y_separation_m"], 1)
        self.assertEqual(pair["left_cube_displacement_m"], 4)
        self.assertEqual(pair["left_bowl_displacement_m"], 1)

    def test_truncated_and_missing_observations_are_not_carried_forward(self):
        rows = [episode("left", [(0, 0, 0), (1, 1, 0), (3, 2, 0)], termination=4),
                episode("right", [(0, 0, 0), (1, -1, 0), (2, -2, 0), (3, -3, 0), (5, -4, 0)])]
        result = self.analyze(rows)
        fixed = result["analyses"]["fixed_action_checkpoints"]
        self.assertEqual(fixed["2"]["control"]["eligible_pairs"], 0)
        self.assertEqual(fixed["4"]["control"]["eligible_pairs"], 0)
        self.assertEqual(fixed["5"]["control"]["eligible_pairs"], 0)
        self.assertEqual(fixed["5"]["control"]["left_episode_reasons"], {"stopped_before_checkpoint": 1})
        self.assertEqual(fixed["4"]["control"]["left_episode_reasons"], {"step_not_observed": 1})
        self.assertIsNone(fixed["4"]["control"]["metrics"]["offset_separation_m"]["mean"])
        self.assertEqual(result["cohort"]["episodes_with_missing_action_samples"], 2)

    def test_absent_episode_preserves_expected_pair_denominator(self):
        result = self.analyze([episode("left", [(0, 0, 0), (1, 1, 0)])], seeds=(1, 2))
        common = result["analyses"]["pretermination_latest_common"]["control"]
        self.assertEqual(common["expected_pairs"], 2)
        self.assertEqual(common["eligible_pairs"], 0)
        self.assertEqual(common["missing_pairs"], 2)
        self.assertEqual(common["right_episode_reasons"], {"missing_episode": 2})
        self.assertEqual(result["cohort"]["missing_episodes"], 3)

    def test_missing_latest_position_does_not_choose_an_earlier_favorable_step(self):
        rows = [episode("left", [(0, 0, 0), (1, 1, 0), (2, 2, 0)]),
                episode("right", [(0, 0, 0), (1, -1, 0), (2, -2, 0)])]
        rows[0]["steps"][1]["object_xyz"] = None
        result = self.analyze(rows)
        self.assertEqual(result["pair_records"][0]["common_action_step"], 1)
        self.assertFalse(result["pair_records"][0]["eligible"])
        self.assertEqual(result["analyses"]["pretermination_latest_common"]["control"]["eligible_pairs"], 0)

    def test_missing_initial_sample_keeps_offset_but_not_displacement(self):
        rows = [episode("left", [(1, 1, 0), (2, 2, 0)]),
                episode("right", [(1, -1, 0), (2, -2, 0)])]
        result = self.analyze(rows)
        pair = result["pair_records"][0]
        self.assertEqual(pair["offset_separation_m"], 2)
        self.assertIsNone(pair["left_cube_displacement_m"])
        summary = result["analyses"]["pretermination_latest_common"]["control"]
        self.assertEqual(summary["metrics"]["left_cube_displacement_m"]["n"], 0)

    def test_no_common_sample_and_missing_termination_remain_unavailable(self):
        rows = [episode("left", [(1, 1, 0)], termination=3),
                episode("right", [(2, -1, 0)], termination=3)]
        result = self.analyze(rows)
        self.assertEqual(result["pair_records"][0]["pair_reason"], "no_common_observed_pretermination_step")
        rows[0].pop("actions_executed")
        rows[0].pop("event_timeline")
        result = self.analyze(rows)
        self.assertEqual(result["pair_records"][0]["left_reason"], "missing_termination_step")

    def test_invalid_episodes_do_not_become_behavioral_failures(self):
        rows = [episode("left", [(0, 0, 0), (1, 1, 0)]),
                episode("right", [(0, 0, 0), (1, -1, 0)])]
        rows[0]["behavioral_result_valid"] = False
        result = self.analyze(rows)
        self.assertEqual(result["cohort"]["invalid_episodes"], 1)
        self.assertEqual(result["cohort"]["valid_behavioral_failures"], 0)
        self.assertEqual(result["pair_records"][0]["left_reason"], "invalid_behavioral_episode")

    def test_duplicates_wrong_frame_and_conflicting_termination_fail_closed(self):
        rows = [episode("left", [(0, 0, 0), (1, 1, 0)]),
                episode("right", [(0, 0, 0), (1, -1, 0)])]
        cases = [rows + [rows[0]]]
        for field, value in [("measurement_frame", "camera_xy"),
                             ("event_timeline", [{"event": "episode_end", "action_step": 8}]),
                             ("steps", rows[0]["steps"] * 2), ("pair_id", "wrong_pair")]:
            changed = copy.deepcopy(rows)
            changed[0][field] = value
            cases.append(changed)
        for changed in cases:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    self.analyze(changed)

    def test_pinned_h01_result_and_fixed_checkpoint_coverage(self):
        rows, provenance = STOP.read_pinned_source(STOP.ROOT)
        result = STOP.analyze_episodes(rows)
        self.assertEqual(len(rows), 108)
        self.assertEqual(provenance["commit"], "ce561e66f82e95055e39d3d7711691982f6b2086")
        self.assertEqual(result["cohort"]["episodes_with_missing_action_samples"], 0)
        self.assertEqual(result["cohort"]["valid_behavioral_failures"], 6)
        self.assertEqual(result["cohort"]["unique_initial_states_by_layout"], {"control": 1, "position_mirrored": 1})
        control = result["analyses"]["pretermination_latest_common"]["control"]
        reflected = result["analyses"]["pretermination_latest_common"]["position_mirrored"]
        self.assertEqual((control["ordered_pairs"], reflected["ordered_pairs"]), (27, 27))
        self.assertAlmostEqual(control["metrics"]["offset_separation_m"]["mean"], 0.3110371443822428, places=12)
        self.assertAlmostEqual(reflected["metrics"]["offset_separation_m"]["mean"], 0.35844855979774837, places=12)
        self.assertEqual(control["common_action_step"], {"n": 27, "min": 89, "max": 290, "median": 133, "mean": 149.7037037037037})
        fixed = result["analyses"]["fixed_action_checkpoints"]
        self.assertEqual([fixed[str(t)]["control"]["eligible_pairs"] for t in STOP.CHECKPOINTS],
                         [27, 27, 27, 26, 18, 8, 5, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual([fixed[str(t)]["position_mirrored"]["eligible_pairs"] for t in STOP.CHECKPOINTS],
                         [27, 27, 27, 27, 17, 5, 4, 2, 2, 1, 1, 0, 0, 0, 0, 0])

    def test_cli_writes_reproducible_json_and_csv_from_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(["python3", str(MODULE_PATH), "--output-dir", directory], check=True, capture_output=True)
            result = json.loads((Path(directory) / "stopping_controls.json").read_text())
            self.assertEqual(result["cohort"]["valid_episodes"], 108)
            self.assertEqual(len((Path(directory) / "stopping_controls.csv").read_text().splitlines()), 37)


if __name__ == "__main__":
    unittest.main()
