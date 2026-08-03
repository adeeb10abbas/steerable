import json
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compile_vla_wam_v2_droid_confirmation import (  # noqa: E402
    EXPECTED_SEEDS,
    compile_confirmation,
    wilson_95,
)
from compile_vla_wam_v2_droid_pilot import TASKS  # noqa: E402


class Pi0FastConfirmationCompilerTest(unittest.TestCase):
    def test_wilson_extremes_are_bounded(self):
        self.assertEqual(wilson_95(0, 10)["confidence"], 0.95)
        self.assertGreater(wilson_95(0, 10)["upper"], 0)
        self.assertLess(wilson_95(10, 10)["lower"], 1)

    def test_compiles_all_twenty_registered_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = root / "registry.json"
            registry.write_text(json.dumps({
                "model_id": "pi0_fast_droid_vla", "prompt_family": "direct_command",
                "completed_seeds": [8300, 8301, 8302], "new_seeds": list(range(8303, 8310)),
                "prompts": {
                    "left": "Put the Rubik's cube to the left of the bowl.",
                    "right": "Put the Rubik's cube to the right of the bowl.",
                },
            }))
            outputs = root / "outputs"
            for seed in EXPECTED_SEEDS:
                run_root = outputs / f"v2_pi0_fast_direct_seed{seed}"
                results = []
                for direction, task in TASKS.items():
                    prompt = json.loads(registry.read_text())["prompts"][direction]
                    task_dir = run_root / task
                    task_dir.mkdir(parents=True)
                    steps = 4 if seed == 8309 else 10
                    log = {"success": direction == "right", "final_step": steps}
                    result_row = {"env_name": task, "instruction": prompt, "success": direction == "right", "timing": {}}
                    if seed == 8303 and direction == "left":
                        log.update({"thermally_intervened": True, "wall_latency_valid": False,
                                    "runtime_intervention_ids": ["pi0-throttle-8303-left"]})
                        result_row.update({"thermally_intervened": True, "wall_latency_valid": False,
                                           "runtime_intervention_ids": ["pi0-throttle-8303-left"]})
                    (task_dir / "log_0_env0.json").write_text(json.dumps(log))
                    (task_dir / "env_cfg.json").write_text(json.dumps({"seed": seed, "instruction": prompt}))
                    (task_dir / "episode_viewport.mp4").write_bytes(b"synthetic")
                    results.append(result_row)
                    with h5py.File(task_dir / "run_0.hdf5", "w") as handle:
                        demo = handle.create_group("data/demo_0")
                        demo.create_dataset("actions", data=np.full((steps, 2), 1 if direction == "right" else 0, dtype=np.float64))
                        states = demo.create_group("states")
                        rigid = states.create_group("rigid_object")
                        cube = np.zeros((steps, 7)); cube[:, 2] = np.linspace(0, 0.04, steps); cube[:, 0] = -0.12 if direction == "left" else 0.12
                        bowl = np.zeros((steps, 7)); robot = np.zeros((steps, 7)); robot[:, 3] = 1
                        rigid.create_group("rubiks_cube").create_dataset("root_pose", data=cube)
                        rigid.create_group("bowl").create_dataset("root_pose", data=bowl)
                        states.create_group("articulation").create_group("robot").create_dataset("root_pose", data=robot)
                        initial = demo.create_group("initial_state")
                        initial.create_group("rigid_object").create_dataset("state", data=np.asarray([seed, 1]))
                (run_root / "episode_results.jsonl").write_text("\n".join(json.dumps(row) for row in results) + "\n")
            source_log = root / "thermal_seed8303.jsonl"
            source_log.write_text('{"event":"cooldown_started"}\n')
            ledger = root / "runtime_interventions.json"
            ledger.write_text(json.dumps({"events": [{
                "id": "pi0-throttle-8303-left", "model_id": "pi0_fast_droid_vla",
                "environment_seed": 8303, "requested_relation": "left",
                "behavioral_result_valid": True, "wall_latency_valid": False,
                "events": [{"event": "cooldown_started"}, {"event": "cooldown_completed"}],
                "started_at_utc": "2026-08-03T12:00:00Z", "completed_at_utc": "2026-08-03T12:01:00Z",
                "max_temperature_c": 88,
                "source_log": {"path": str(source_log), "sha256": hashlib.sha256(source_log.read_bytes()).hexdigest()},
            }]}))
            compiled = compile_confirmation(
                registry, outputs, root / "trajectories", interventions_paths=[ledger]
            )
            self.assertEqual(compiled["summary"]["episode_count"], 20)
            self.assertEqual(compiled["summary"]["pair_count"], 10)
            self.assertEqual(compiled["summary"]["by_direction"]["left"]["episodes"], 10)
            self.assertEqual(compiled["summary"]["by_direction"]["right"]["successes"], 10)
            self.assertEqual(len(compiled["summary"]["paired_endpoint_responses"]), 10)
            self.assertEqual(compiled["summary"]["paired_endpoint_responses"][-1]["first_ten_action_rms_steps_used"], 4)
            throttled = next(row for row in compiled["episodes"] if row["environment_seed"] == 8303 and row["requested_relation"] == "left")
            self.assertFalse(throttled["operational_wall_latency_valid"])
            self.assertEqual(throttled["runtime_intervention_ids"], ["pi0-throttle-8303-left"])
            self.assertEqual(compiled["summary"]["operational_wall_latency_excluded_episodes"], 1)
            self.assertEqual(compiled["applied_runtime_intervention_ids"], ["pi0-throttle-8303-left"])
            self.assertEqual(compiled["intervention_ledger_sources"][0]["applied_event_ids"], ["pi0-throttle-8303-left"])


if __name__ == "__main__":
    unittest.main()
