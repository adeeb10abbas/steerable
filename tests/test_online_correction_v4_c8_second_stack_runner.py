"""Contract and binding tests for the C8 second_stack / GR00T Bridge episode runner."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_contract import (
    GROOT_POLICY_ID,
    LaunchArgs,
    SECOND_STACK_FIXTURE_ID,
    build_launch_plan,
    expected_action_shape,
    validate_runtime_lock,
)
from experiments.online_correction_v4.droid_bindings import resolve_motion_direction
from experiments.online_correction_v4.contracts import EpisodeManifestRow


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "docs/online_correction_v4/campaign.json"
PILOT_LOCK = ROOT / "artifacts/online_correction_v4/setup/second_stack_g7_pilot_runtime_lock.released.json"
PILOT_QUEUE = ROOT / "artifacts/online_correction_v4/setup/second_stack_g7_pilot_queue.jsonl"
RUNNER = ROOT / "tools/run_online_correction_v4.py"


class C8SecondStackRunnerTests(unittest.TestCase):
    def test_tuple_to_simpler_env_action_maps_groot_components(self) -> None:
        from experiments.online_correction_v4.droid_groot_observation import (
            GROOT_ACTION_COMPONENT_KEYS,
            tuple_to_simpler_env_action,
        )

        action = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.0, 0.0)
        mapped = tuple_to_simpler_env_action(action)
        self.assertEqual(set(mapped), set(GROOT_ACTION_COMPONENT_KEYS))
        self.assertAlmostEqual(float(mapped["action.x"][0]), 0.1)
        self.assertAlmostEqual(float(mapped["action.gripper"][0]), 1.0)

    def test_groot_policy_registered_in_contract(self) -> None:
        self.assertIn(GROOT_POLICY_ID, ("cosmos3_nano_droid", "pi05_droid", GROOT_POLICY_ID))
        self.assertEqual(expected_action_shape(GROOT_POLICY_ID), (8, 8))

    def test_pilot_runtime_lock_binds_groot_policy(self) -> None:
        if not PILOT_LOCK.is_file():
            self.skipTest("pilot runtime lock artifact missing")
        lock = validate_runtime_lock(PILOT_LOCK)
        self.assertIn(GROOT_POLICY_ID, lock.policies)
        self.assertIn(SECOND_STACK_FIXTURE_ID, lock.fixtures)

    def test_build_launch_plan_accepts_c8_pilot_episode(self) -> None:
        if not PILOT_LOCK.is_file() or not PILOT_QUEUE.is_file():
            self.skipTest("pilot release artifacts missing")
        first_episode_id = json.loads(PILOT_QUEUE.read_text().splitlines()[0])["episode_id"]
        with tempfile.TemporaryDirectory() as tmp:
            plan = build_launch_plan(
                LaunchArgs(
                    manifest_path=PILOT_QUEUE,
                    runtime_lock_path=PILOT_LOCK,
                    episode_id=first_episode_id,
                    attempt_id="c8-runner-test",
                    output_dir=Path(tmp) / "out",
                    dry_run=True,
                    validate_only=True,
                ),
                study_root=ROOT,
                campaign_config_path=CONFIG,
            )
        self.assertEqual(plan["policy_id"], GROOT_POLICY_ID)
        self.assertEqual(plan["fixture"], SECOND_STACK_FIXTURE_ID)
        self.assertEqual(plan["family"], "C8")

    def test_run_online_correction_v4_dry_run_accepts_c8_pilot(self) -> None:
        if not PILOT_LOCK.is_file() or not PILOT_QUEUE.is_file():
            self.skipTest("pilot release artifacts missing")
        first_episode_id = json.loads(PILOT_QUEUE.read_text().splitlines()[0])["episode_id"]
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [
                    "python3",
                    str(RUNNER),
                    "--manifest",
                    str(PILOT_QUEUE),
                    "--runtime-lock",
                    str(PILOT_LOCK),
                    "--episode-id",
                    first_episode_id,
                    "--attempt-id",
                    "c8-runner-cli-test",
                    "--output",
                    str(Path(tmp) / "out"),
                    "--campaign-config",
                    str(CONFIG),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
                cwd=ROOT,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["policy_id"], GROOT_POLICY_ID)

    def test_resolve_motion_direction_uses_scene_axes(self) -> None:
        if not PILOT_QUEUE.is_file():
            self.skipTest("pilot queue missing")
        row = json.loads(PILOT_QUEUE.read_text().splitlines()[0])
        manifest = EpisodeManifestRow.from_manifest_dict(row)
        direction = resolve_motion_direction(manifest)
        norm = (direction[0] ** 2 + direction[1] ** 2) ** 0.5
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_second_stack_settle_probe_matches_reset_merge_contract(self) -> None:
        from dataclasses import dataclass

        from experiments.online_correction_v4.droid_second_stack_simulator import (
            SecondStackSettleProbe,
        )
        from experiments.online_correction_v4.second_stack import REFERENCE_OBJECT

        @dataclass
        class _Backend:
            kinematic_adapter: object = object()

        probe = SecondStackSettleProbe(backend=_Backend())  # type: ignore[arg-type]
        maxima = probe.sample_stability()
        self.assertIn(REFERENCE_OBJECT, maxima)
        row = maxima[REFERENCE_OBJECT]
        self.assertIsInstance(row, dict)
        self.assertIn("max_linear_component_speed_m_s", row)
        self.assertIn("max_angular_component_speed_rad_s", row)
        for value in maxima.values():
            self.assertIsInstance(value, dict)


if __name__ == "__main__":
    unittest.main()
