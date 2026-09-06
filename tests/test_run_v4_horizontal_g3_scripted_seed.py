"""Focused tests for the horizontal G3 scripted-seed runner helpers."""

from __future__ import annotations

import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PLAN = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_g3_plan.candidate.json"
)
CAMPAIGN = ROOT / "docs/online_correction_v4/campaign.json"
REGISTRY = (
    ROOT
    / "artifacts/online_correction_v4/setup/horizontal_reset_registry.candidate.json"
)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "run_v4_horizontal_g3_scripted_seed",
    ROOT / "tools/run_v4_horizontal_g3_scripted_seed.py",
)
runner = importlib.util.module_from_spec(RUNNER_SPEC)
assert RUNNER_SPEC.loader is not None
RUNNER_SPEC.loader.exec_module(runner)


class ScriptedSeedOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))

    def test_stationary_check_order_has_twelve_cases(self) -> None:
        keys = runner.expected_scripted_seed_checks(plan=self.plan, mode="stationary")
        self.assertEqual(len(keys), runner.STATIONARY_CHECK_COUNT)
        self.assertEqual(
            keys,
            (
                ("left", "original"),
                ("left", "midpoint"),
                ("left", "endpoint"),
                ("right", "original"),
                ("right", "midpoint"),
                ("right", "endpoint"),
                ("front", "original"),
                ("front", "midpoint"),
                ("front", "endpoint"),
                ("behind", "original"),
                ("behind", "midpoint"),
                ("behind", "endpoint"),
            ),
        )

    def test_moving_check_order_has_four_endpoint_cases(self) -> None:
        keys = runner.expected_scripted_seed_checks(plan=self.plan, mode="moving")
        self.assertEqual(len(keys), runner.MOVING_CHECK_COUNT)
        self.assertEqual(
            keys,
            (
                ("left", "endpoint"),
                ("right", "endpoint"),
                ("front", "endpoint"),
                ("behind", "endpoint"),
            ),
        )

    def test_moving_check_order_rejects_changed_goal_inventory(self) -> None:
        changed = json.loads(json.dumps(self.plan))
        changed["scripted_controller"]["moving"]["goals"] = ["left"]
        with self.assertRaises(RuntimeError):
            runner.expected_scripted_seed_checks(plan=changed, mode="moving")

    def test_check_slug_is_stable(self) -> None:
        self.assertEqual(runner.check_slug("left", "midpoint"), "left__midpoint")


class ReferenceFractionTests(unittest.TestCase):
    def test_reference_displacement_fractions(self) -> None:
        self.assertEqual(runner.reference_displacement_m("original", 0.12), 0.0)
        self.assertAlmostEqual(runner.reference_displacement_m("midpoint", 0.12), 0.06)
        self.assertAlmostEqual(runner.reference_displacement_m("endpoint", 0.12), 0.12)

    def test_rejects_unknown_reference_position(self) -> None:
        with self.assertRaises(ValueError):
            runner.reference_displacement_m("unknown", 0.12)


class FrozenControllerConfigTests(unittest.TestCase):
    def test_frozen_config_matches_prospective_contract(self) -> None:
        config = runner.frozen_scripted_controller_config()
        self.assertEqual(config["phase_ticks"]["approach"], 30)
        self.assertEqual(config["phase_ticks"]["settle"], 15)
        self.assertAlmostEqual(config["geometry_offsets"]["approach_height_m"], 0.12)
        self.assertAlmostEqual(config["geometry_offsets"]["descend_offset_m"], 0.025)
        self.assertAlmostEqual(config["geometry_offsets"]["place_descend_offset_m"], 0.04)
        self.assertAlmostEqual(config["gripper_close"], 0.785398)
        self.assertNotIn("eef_yaw_offset_rad", config)

    def test_object_pair_presents_short_sponge_side_to_gripper(self) -> None:
        config = runner.frozen_scripted_controller_config("object_pair")
        self.assertAlmostEqual(config["eef_yaw_offset_rad"], math.pi / 2.0)


class MovingCallbackTests(unittest.TestCase):
    def test_motion_starts_only_once_after_grab(self) -> None:
        offsets: list[tuple[float, tuple[float, float]]] = []
        grabbed = {"value": False}
        motion = json.loads(CAMPAIGN.read_text(encoding="utf-8"))["motion"]
        env = SimpleNamespace(set_reference_kinematic_offset=lambda d, direction: offsets.append((d, direction)))

        def probe() -> bool:
            return grabbed["value"]

        callback, state = runner.build_moving_reference_motion_callback(
            env,
            displacement_m=0.12,
            direction_task=(1.0, 0.0),
            motion_config=motion,
            object_grabbed_probe=probe,
            set_reference_offset=env.set_reference_kinematic_offset,
        )
        for tick in range(1, 6):
            callback(tick, tick * 0.05)
        self.assertFalse(state["motion_started"])
        self.assertEqual(offsets, [(0.0, (1.0, 0.0))] * 5)

        grabbed["value"] = True
        callback(6, 0.30)
        self.assertTrue(state["motion_started"])
        callback(7, 0.35)
        callback(8, 0.40)
        self.assertGreater(offsets[-1][0], 0.0)
        started_at = state["motion_origin_sim_time_s"]
        callback(9, 0.45)
        self.assertEqual(state["motion_origin_sim_time_s"], started_at)


class ScriptedSeedGateValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN.read_text(encoding="utf-8"))
        cls.campaign = json.loads(CAMPAIGN.read_text(encoding="utf-8"))

    def test_validate_stationary_seed_in_selected_resets(self) -> None:
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file

        seed = self.plan["scripted_controller"]["reset_env_seeds"][0]
        runner.validate_scripted_seed_gate_inputs(
            plan=self.plan,
            campaign=self.campaign,
            campaign_path=CAMPAIGN,
            campaign_sha256=sha256_file(CAMPAIGN),
            plan_path=PLAN,
            plan_sha256=sha256_file(PLAN),
            reset_registry_path=REGISTRY,
            reset_registry_sha256=sha256_file(REGISTRY),
            environment_seed=int(seed),
            scale=1.0,
            mode="stationary",
            sha256_file=sha256_file,
        )

    def test_validate_moving_seed_requires_canonical_env_seed(self) -> None:
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file

        canonical = self.plan["scripted_controller"]["moving"]["canonical_env_seed"]
        runner.validate_scripted_seed_gate_inputs(
            plan=self.plan,
            campaign=self.campaign,
            campaign_path=CAMPAIGN,
            campaign_sha256=sha256_file(CAMPAIGN),
            plan_path=PLAN,
            plan_sha256=sha256_file(PLAN),
            reset_registry_path=REGISTRY,
            reset_registry_sha256=sha256_file(REGISTRY),
            environment_seed=int(canonical),
            scale=1.0,
            mode="moving",
            sha256_file=sha256_file,
        )

    def test_stationary_rejects_nonselected_seed(self) -> None:
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file

        with self.assertRaises(RuntimeError):
            runner.validate_scripted_seed_gate_inputs(
                plan=self.plan,
                campaign=self.campaign,
                campaign_path=CAMPAIGN,
                campaign_sha256=sha256_file(CAMPAIGN),
                plan_path=PLAN,
                plan_sha256=sha256_file(PLAN),
                reset_registry_path=REGISTRY,
                reset_registry_sha256=sha256_file(REGISTRY),
                environment_seed=2100000001,
                scale=1.0,
                mode="stationary",
                sha256_file=sha256_file,
            )

    def test_moving_rejects_noncanonical_seed(self) -> None:
        from experiments.online_correction_v4.droid_task_files.binding import sha256_file

        with self.assertRaises(RuntimeError):
            runner.validate_scripted_seed_gate_inputs(
                plan=self.plan,
                campaign=self.campaign,
                campaign_path=CAMPAIGN,
                campaign_sha256=sha256_file(CAMPAIGN),
                plan_path=PLAN,
                plan_sha256=sha256_file(PLAN),
                reset_registry_path=REGISTRY,
                reset_registry_sha256=sha256_file(REGISTRY),
                environment_seed=2100000098,
                scale=1.0,
                mode="moving",
                sha256_file=sha256_file,
            )


class ScriptedSeedSummaryTests(unittest.TestCase):
    def test_compile_summary_counts_passed_and_failed(self) -> None:
        records = [
            {
                "goal": "left",
                "reference_position": "original",
                "passed": True,
                "trajectory": {"path": "t.json", "sha256": "a" * 64, "bytes": 1},
                "receipt": {"path": "r.json", "sha256": "b" * 64, "bytes": 1},
            },
            {
                "goal": "left",
                "reference_position": "midpoint",
                "passed": False,
                "trajectory": {"path": "t2.json", "sha256": "c" * 64, "bytes": 1},
                "receipt": {"path": "r2.json", "sha256": "d" * 64, "bytes": 1},
            },
        ]
        with self.assertRaises(RuntimeError):
            runner.compile_scripted_seed_summary(
                mode="moving",
                environment_seed=2100000000,
                scale=1.0,
                displacement_m=0.12,
                plan_sha256="e" * 64,
                campaign_sha256="f" * 64,
                reset_registry_sha256="0" * 64,
                runtime_identity={"native_control_dt_s": 0.05},
                controller_config=runner.frozen_scripted_controller_config(),
                check_records=records,
                registered_reset={},
            )

        full_records = [
            {
                "goal": goal,
                "reference_position": "endpoint",
                "passed": goal != "behind",
                "trajectory": {"path": f"t_{goal}.json", "sha256": "a" * 64, "bytes": 1},
                "receipt": {"path": f"r_{goal}.json", "sha256": "b" * 64, "bytes": 1},
            }
            for goal in ("left", "right", "front", "behind")
        ]
        summary = runner.compile_scripted_seed_summary(
            mode="moving",
            environment_seed=2100000000,
            scale=1.0,
            displacement_m=0.12,
            plan_sha256="e" * 64,
            campaign_sha256="f" * 64,
            reset_registry_sha256="0" * 64,
            runtime_identity={"native_control_dt_s": 0.05, "mode": "moving"},
            controller_config=runner.frozen_scripted_controller_config(),
            check_records=full_records,
            registered_reset={"reset_attestation": {"path": "x", "sha256": "y" * 64, "bytes": 1}},
        )
        self.assertEqual(summary["schema_version"], runner.SEED_RECEIPT_SCHEMA)
        self.assertEqual(summary["passed_check_count"], 3)
        self.assertEqual(summary["failed_check_count"], 1)
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["model_request_count"], 0)
        self.assertEqual(summary["behavioral_episode_count"], 0)
        self.assertIn("controller_config", summary)


class ExclusiveWriteTests(unittest.TestCase):
    def test_write_json_exclusive_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            runner._write_json_exclusive(path, {"value": 1})
            with self.assertRaises(FileExistsError):
                runner._write_json_exclusive(path, {"value": 2})

    def test_infrastructure_failure_is_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            runner._write_infrastructure_failure(
                output_dir,
                {"schema_version": runner.INFRA_FAILURE_SCHEMA, "status": "infrastructure_invalid"},
            )
            with self.assertRaises(FileExistsError):
                runner._write_infrastructure_failure(
                    output_dir,
                    {"schema_version": runner.INFRA_FAILURE_SCHEMA, "status": "infrastructure_invalid"},
                )


class ScriptedSeedImportSafetyTests(unittest.TestCase):
    def test_runner_module_imports_without_robolab(self) -> None:
        blocked = {
            "isaaclab",
            "isaaclab.app",
            "robolab",
            "robolab.constants",
        }
        original_import = __import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            root = name.split(".", 1)[0]
            if name in blocked or root in {"isaaclab", "robolab", "omni"}:
                raise ImportError(f"blocked import {name}")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            spec = importlib.util.spec_from_file_location(
                "g3_scripted_seed_isolated",
                ROOT / "tools/run_v4_horizontal_g3_scripted_seed.py",
            )
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            self.assertAlmostEqual(
                module.reference_displacement_m("midpoint", 0.10),
                0.05,
            )


if __name__ == "__main__":
    unittest.main()
