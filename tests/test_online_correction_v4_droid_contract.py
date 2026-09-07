"""Contract tests for the V4 DROID adapter launch layer."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.droid_contract import (
    DroidContractError,
    LaunchArgs,
    PrefixMode,
    build_launch_plan,
    compute_adapter_contract_sha256,
    validate_launch_args,
    validate_prefix_mode_for_family,
    validate_runtime_lock,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs/online_correction_v4/campaign.json"
TEMPLATE_LOCK = ROOT / "docs/online_correction_v4/runtime_lock.template.json"
CLI = ROOT / "tools/run_online_correction_v4.py"


def _released_lock(*, manifest_sha256: str, config_sha256: str) -> dict:
    template = json.loads(TEMPLATE_LOCK.read_text(encoding="utf-8"))
    template["release_status"] = "RELEASED"
    template["released_families"] = ["C1"]
    template["manifest_sha256"] = manifest_sha256
    template["config_sha256"] = config_sha256
    template["prefix_mode"] = PrefixMode.FRESH_SESSION_DETERMINISTIC_REPLAY.value
    template["prefix_mode_receipt_sha256"] = "a" * 64
    template["runner"] = {
        "commit": "c" * 40,
        "entrypoint": "tools/run_online_correction_v4.py",
        "sha256": "b" * 64,
    }
    template["writer_contract"] = {
        "schema_version": "v4-droid-writer-contract-v1",
        "output_parent_uri": "file:///persistent/v4/attempts",
        "viewport_video_required": True,
        "write_once_attempt_directories": True,
        "incremental_fsync_required": True,
        "required_streams": [
            "viewport_video",
            "trajectory",
            "requests",
            "observations",
            "events",
        ],
    }
    for name in ("cosmos3_nano_droid", "pi05_droid"):
        template["policies"][name].update(
            {
                "checkpoint_sha256": "1" * 64,
                "checkpoint_uri": "file:///persistent/v4/checkpoints/" + name,
                "runtime_image_digest": "sha256:" + ("2" * 64),
                "integration_commit": "c" * 40,
                "native_control_dt_s": 0.05,
                "achieved_delay_s": 0.10,
                "achieved_standard_query_period_s": 0.50,
                "achieved_fast_query_period_s": 0.25,
                "prediction_horizon_actions": 32 if name.startswith("cosmos") else 15,
                "policy_reset_and_history_contract_uri": "file:///persistent/v4/contracts/" + name,
            }
        )
    for fixture in template["fixtures"].values():
        fixture.update(
            {
                "geometry_sha256": "3" * 64,
                "scorer_sha256": "4" * 64,
                "reset_registry_sha256": "5" * 64,
                "geometry_uri": "file:///persistent/v4/geometry.json",
                "scorer_uri": "file:///persistent/v4/scorer.json",
                "reset_registry_uri": "file:///persistent/v4/resets.jsonl",
                "frame_transform_uri": "file:///persistent/v4/frame.json",
                "goal_geometry_and_tolerances_uri": "file:///persistent/v4/goals.json",
                "trigger_release_detector_uri": "file:///persistent/v4/detectors.json",
                "intervention_trajectory_registry_uri": "file:///persistent/v4/motion.jsonl",
                "scoring_and_visibility_thresholds_uri": "file:///persistent/v4/thresholds.json",
                "calibration_scale": 0.12,
                "D_cap_m": 0.12,
            }
        )
    for receipt in template["receipts"].values():
        receipt.update(
            {
                "passed": True,
                "family_ids": ["C1"],
                "uri": "file:///persistent/v4/receipts/gate.json",
                "sha256": "6" * 64,
            }
        )
    return template


def _manifest_row(*, episode_id: str = "ep-test", policy: str = "cosmos3_nano_droid") -> dict:
    return {
        "schema_version": 1,
        "manifest_type": "planning_manifest",
        "runtime_bound": True,
        "episode_id": episode_id,
        "campaign": "online_correction_v4",
        "family": "C1",
        "fixture": "horizontal",
        "block_id": 0,
        "block_key": "horizontal:0",
        "env_seed": 2100000000,
        "policy_seed": 42,
        "cohort": "confirmatory",
        "priority": "primary",
        "factors": {
            "policy": policy,
            "goal": "left",
            "wording": "direct",
            "scenario": "move_stop",
            "schedule": "standard",
            "named_reference": "bowl",
        },
        "prefix_group_id": "prefix-test",
        "execution_group": f"{policy}:horizontal",
        "execution_order_key": "000",
        "config_sha256": json.loads(CONFIG_PATH.read_text()) and "",
        "reuse_episode_ids": [],
        "counterbalance": {"event_phase_fraction": 0.0},
        "prompt_recipe": {
            "template": "Place the cube so that the cube is left of the bowl. Use the robot's fixed viewpoint for left, right, front, and behind.",
            "object_role": "cube",
            "reference_role": "bowl",
        },
    }


class DroidContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config_sha = __import__(
            "experiments.online_correction_v4.droid_contract", fromlist=["sha256_file"]
        ).sha256_file(CONFIG_PATH)

    def test_adapter_contract_hash_is_stable(self) -> None:
        digest = compute_adapter_contract_sha256(ROOT)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_template_runtime_lock_fails_closed(self) -> None:
        with self.assertRaises(DroidContractError):
            validate_runtime_lock(TEMPLATE_LOCK)

    def test_c2_blocks_independent_rollout_fallback(self) -> None:
        with self.assertRaises(DroidContractError):
            validate_prefix_mode_for_family(
                PrefixMode.INDEPENDENT_NATURAL_ROLLOUT_FALLBACK,
                "C2",
            )

    def test_verified_prefix_mode_vocabulary_matches_analysis(self) -> None:
        from experiments.online_correction_v4.analysis import (
            VERIFIED_COMMON_PREFIX_MODES,
        )

        self.assertEqual(
            {
                PrefixMode.DETERMINISTIC_FRESH_SESSION_REPLAY.value,
                PrefixMode.QUALIFIED_FULL_STATE_SNAPSHOT.value,
            },
            VERIFIED_COMMON_PREFIX_MODES,
        )

    def test_build_launch_plan_with_released_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            row = _manifest_row()
            row["config_sha256"] = self.config_sha
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_sha = __import__(
                "experiments.online_correction_v4.droid_contract", fromlist=["sha256_file"]
            ).sha256_file(manifest_path)
            lock_path = tmp_path / "runtime_lock.json"
            lock_path.write_text(
                json.dumps(_released_lock(manifest_sha256=manifest_sha, config_sha256=self.config_sha)),
                encoding="utf-8",
            )
            args = LaunchArgs(
                manifest_path=manifest_path,
                runtime_lock_path=lock_path,
                episode_id="ep-test",
                attempt_id="attempt-001",
                output_dir=tmp_path / "output",
                dry_run=True,
                validate_only=True,
            )
            plan = build_launch_plan(args, study_root=ROOT, campaign_config_path=CONFIG_PATH)
            self.assertEqual(plan["episode_id"], "ep-test")
            self.assertEqual(plan["policy_id"], "cosmos3_nano_droid")
            self.assertTrue(plan["dry_run"])

    def test_distinct_planning_and_frozen_queue_hashes_are_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            row = _manifest_row()
            row["config_sha256"] = self.config_sha
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            queue_sha = __import__(
                "experiments.online_correction_v4.droid_contract",
                fromlist=["sha256_file"],
            ).sha256_file(manifest_path)
            lock = _released_lock(
                manifest_sha256="d" * 64,
                config_sha256=self.config_sha,
            )
            lock["frozen_queue_sha256"] = queue_sha
            lock_path = tmp_path / "runtime_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            args = LaunchArgs(
                manifest_path=manifest_path,
                runtime_lock_path=lock_path,
                episode_id="ep-test",
                attempt_id="attempt-001",
                output_dir=tmp_path / "output",
                dry_run=True,
                validate_only=True,
            )

            plan = build_launch_plan(
                args,
                study_root=ROOT,
                campaign_config_path=CONFIG_PATH,
            )

            self.assertEqual(plan["manifest_sha256"], "d" * 64)

    def test_pilot_released_lock_accepts_only_engineering_pilot_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            row = _manifest_row()
            row["cohort"] = "engineering_pilot"
            row["config_sha256"] = self.config_sha
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_sha = __import__(
                "experiments.online_correction_v4.droid_contract",
                fromlist=["sha256_file"],
            ).sha256_file(manifest_path)
            lock = _released_lock(
                manifest_sha256=manifest_sha,
                config_sha256=self.config_sha,
            )
            lock["release_status"] = "PILOT_RELEASED"
            lock_path = tmp_path / "runtime_lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            args = LaunchArgs(
                manifest_path=manifest_path,
                runtime_lock_path=lock_path,
                episode_id="ep-test",
                attempt_id="attempt-001",
                output_dir=tmp_path / "output",
                dry_run=True,
                validate_only=True,
            )
            plan = build_launch_plan(
                args,
                study_root=ROOT,
                campaign_config_path=CONFIG_PATH,
            )
            self.assertEqual(plan["episode_id"], "ep-test")

            row["cohort"] = "confirmatory"
            manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            manifest_sha = __import__(
                "experiments.online_correction_v4.droid_contract",
                fromlist=["sha256_file"],
            ).sha256_file(manifest_path)
            lock["manifest_sha256"] = manifest_sha
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            with self.assertRaisesRegex(
                DroidContractError,
                "only engineering_pilot",
            ):
                build_launch_plan(
                    args,
                    study_root=ROOT,
                    campaign_config_path=CONFIG_PATH,
                )

    def test_missing_episode_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(json.dumps(_manifest_row(episode_id="other")) + "\n", encoding="utf-8")
            manifest_sha = __import__(
                "experiments.online_correction_v4.droid_contract", fromlist=["sha256_file"]
            ).sha256_file(manifest_path)
            lock_path = tmp_path / "runtime_lock.json"
            lock_path.write_text(
                json.dumps(_released_lock(manifest_sha256=manifest_sha, config_sha256=self.config_sha)),
                encoding="utf-8",
            )
            args = LaunchArgs(
                manifest_path=manifest_path,
                runtime_lock_path=lock_path,
                episode_id="missing",
                attempt_id="attempt-001",
                output_dir=tmp_path / "output",
                dry_run=True,
                validate_only=True,
            )
            with self.assertRaises(DroidContractError):
                build_launch_plan(args, study_root=ROOT, campaign_config_path=CONFIG_PATH)

    def test_cli_help(self) -> None:
        result = subprocess.run(
            ["python3", str(CLI), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--dry-run", result.stdout)
        self.assertIn("--validate-only", result.stdout)

    def test_cli_dry_run_rejects_template_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            manifest_path = tmp_path / "manifest.jsonl"
            manifest_path.write_text(json.dumps(_manifest_row()) + "\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "python3",
                    str(CLI),
                    "--manifest",
                    str(manifest_path),
                    "--runtime-lock",
                    str(TEMPLATE_LOCK),
                    "--episode-id",
                    "ep-test",
                    "--attempt-id",
                    "attempt-001",
                    "--output",
                    str(tmp_path / "output"),
                    "--dry-run",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("blocked", result.stderr.lower())

    def test_validate_launch_args_requires_absolute_output(self) -> None:
        with self.assertRaises(DroidContractError):
            validate_launch_args(
                LaunchArgs(
                    manifest_path=CONFIG_PATH,
                    runtime_lock_path=TEMPLATE_LOCK,
                    episode_id="ep",
                    attempt_id="a",
                    output_dir=Path("relative/output"),
                    dry_run=True,
                    validate_only=True,
                )
            )


if __name__ == "__main__":
    unittest.main()
