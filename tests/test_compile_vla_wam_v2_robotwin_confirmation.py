import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compile_vla_wam_v2_robotwin_confirmation import (  # noqa: E402
    action_trace_record,
    compile_confirmation,
    load_invalid_attempts,
    load_interventions,
    validate_result_ledger_reconciliation,
)


class RoboTwinConfirmationCompilerTest(unittest.TestCase):
    def test_compiles_ten_exact_pairs_without_action_traces(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scenes = [{"pair_id": f"robotwin_pair_{index:02d}", "phase": "completed_pilot" if index < 3 else "new_expansion",
                       "anchor_task": "place_a2b_left" if index % 2 == 0 else "place_a2b_right",
                       "environment_seed": 4300000 + index, "sampling_seed": 8400 + index}
                      for index in range(10)]
            registry = root / "registry.json"
            registry.write_text(json.dumps({"models": ["efficient_wam_rt_robotwin", "fastwam_robotwin", "lingbot_va_robotwin", "lingbot_vla_4b_robotwin"], "scenes": scenes}))
            inputs = root / "inputs"
            for scene in scenes:
                for direction in ("left", "right"):
                    cell = inputs / scene["pair_id"] / direction
                    cell.mkdir(parents=True)
                    trajectory = [{"action_step": 0, "object_xyz": [0, 0, 0.7], "relation_region": False, "grippers_open": True},
                                  {"action_step": 1, "object_xyz": [-0.1 if direction == "left" else 0.1, 0, 0.75], "relation_region": True, "grippers_open": True}]
                    (cell / "trajectory.json").write_text(json.dumps(trajectory))
                    (cell / "simulator.mp4").write_bytes(b"synthetic")
                    result = {"task": scene["anchor_task"], "environment_seed": scene["environment_seed"], "sampling_seed": scene["sampling_seed"],
                              "prompt_family": "direct_command", "requested_relation": direction,
                              "object_name": "blue soap", "target_name": "tea-box", "object_model_id": 1, "target_model_id": 2,
                              "prompt": f"Put the blue soap to the {direction} of the tea-box.", "actions_executed": 1,
                              "requested_success": direction == "left", "wall_seconds": 1.0,
                              "initial": {"object_xyz": [0, 0, 0.7], "target_xyz": [0, -0.1, 0.7], "object_quat": [1, 0, 0, 0], "robot_qpos": [0.1, 0.2], "grippers_open": True, "object_minus_target_x": 0, "object_minus_target_y": 0.1},
                              "final": {"object_minus_target_x": -0.1 if direction == "left" else 0.1, "object_minus_target_y": 0},
                              "trajectory_path": str(cell / "trajectory.json"), "simulator_video": str(cell / "simulator.mp4")}
                    if scene["phase"] == "new_expansion":
                        trace_path = cell / "action_trace.npz"
                        trace_steps = 4 if scene["environment_seed"] == 4300003 else 10
                        executed = np.full((trace_steps, 2), -1 if direction == "left" else 1, dtype=np.float32)
                        np.savez_compressed(trace_path, executed=executed)
                        result["action_trace"] = {"path": str(trace_path), "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(), "count": trace_steps, "shape": [trace_steps, 2]}
                    (cell / "result.json").write_text(json.dumps(result))
            compiled = compile_confirmation(inputs, "fastwam_robotwin", registry)
            self.assertEqual(compiled["summary"]["episode_count"], 20)
            self.assertEqual(compiled["summary"]["pair_count"], 10)
            self.assertEqual(compiled["summary"]["by_direction"]["left"]["successes"], 10)
            self.assertEqual(compiled["summary"]["future_interface_counts"], {"action_only_not_applicable": 20})
            self.assertEqual(compiled["summary"]["first_ten_executed_action_rms_coverage"]["coverage"], "7/10")
            self.assertEqual(compiled["summary"]["paired_endpoint_responses"][0]["first_ten_executed_action_rms"], None)
            self.assertGreater(compiled["summary"]["paired_endpoint_responses"][3]["first_ten_executed_action_rms"], 0)
            self.assertEqual(compiled["summary"]["paired_endpoint_responses"][3]["first_ten_executed_action_rms_steps_used"], 4)
            self.assertIn("object_quat", compiled["episodes"][0]["initial_state_coverage"]["hash_input_initial_fields"])
            self.assertIn("robot_qpos", compiled["episodes"][0]["initial_state_coverage"]["hash_input_initial_fields"])

    def test_rejects_missing_prospective_trace_and_latency_contradiction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "missing result.json action_trace"):
                action_trace_record(root / "result.json", {}, required=True)
            ledger = root / "runtime.json"
            ledger.write_text(json.dumps({"events": [{
                "id": "bad", "model_id": "fastwam_robotwin", "environment_seed": 4300003,
                "requested_relation": "left", "behavioral_result_valid": True, "wall_latency_valid": True,
            }]}))
            with self.assertRaisesRegex(RuntimeError, "contradicts"):
                load_interventions(ledger, "fastwam_robotwin")

    def test_mixed_model_and_two_runtime_ledgers_are_routed_without_loss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            historical = root / "historical.json"
            confirmation = root / "confirmation.json"
            historical.write_text(json.dumps({"events": [
                {"id": "hist-ling", "model_id": "lingbot_va_robotwin", "environment_seed": 4300000,
                 "requested_relation": "right", "behavioral_result_valid": True, "wall_latency_valid": False},
                {"id": "other-fast", "model_id": "fastwam_robotwin", "environment_seed": 4300000,
                 "requested_relation": "left", "behavioral_result_valid": True, "wall_latency_valid": False},
            ]}))
            confirmation.write_text(json.dumps({"events": [
                {"id": "confirm-ling", "model_id": "lingbot_va_robotwin", "environment_seed": 4300003,
                 "requested_relation": "left", "behavioral_result_valid": True, "wall_latency_valid": False},
            ]}))
            routed, sources = load_interventions([historical, confirmation], "lingbot_va_robotwin")
            self.assertEqual({event["id"] for events in routed.values() for event in events}, {"hist-ling", "confirm-ling"})
            self.assertEqual(sources[0]["ignored_other_model_event_count"], 1)
            self.assertEqual(sources[1]["selected_model_event_count"], 1)
            self.assertTrue(all(source["sha256"] for source in sources))

    def test_mixed_model_invalid_ledger_filters_other_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "invalid.json"
            base = {"classification": "partial", "behavioral_result_valid": False,
                    "wall_latency_valid": False, "environment_seed": 4300003,
                    "requested_relation": "left"}
            ledger.write_text(json.dumps({"events": [
                {**base, "id": "ling-partial", "model_id": "lingbot_va_robotwin"},
                {**base, "id": "fast-partial", "model_id": "fastwam_robotwin"},
            ]}))
            events, by_cell, sources = load_invalid_attempts([ledger], "lingbot_va_robotwin")
            self.assertEqual([event["id"] for event in events], ["ling-partial"])
            self.assertEqual(len(by_cell[(4300003, "left")]), 1)
            self.assertEqual(sources[0]["ignored_other_model_event_count"], 1)

    def test_result_reported_thermal_state_requires_a_ledger_event(self):
        with self.assertRaisesRegex(RuntimeError, "not represented"):
            validate_result_ledger_reconciliation(
                Path("result.json"), {"thermally_intervened": True}, [], []
            )


if __name__ == "__main__":
    unittest.main()
