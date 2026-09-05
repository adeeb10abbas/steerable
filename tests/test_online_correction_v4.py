"""Design/audit checks. These tests do not validate a simulator or policy runner."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("online_correction_v4", ROOT / "tools/online_correction_v4.py")
v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v4)


class CampaignInventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.rows = v4.build_manifest(cls.config, cls.config_sha)
        cls.by_id = {row["episode_id"]: row for row in cls.rows}

    def test_exact_budget_excludes_control_reuse_and_engineering(self):
        self.assertEqual(len(self.rows), 17664)
        self.assertEqual(len({v4._cell_key(row) for row in self.rows}), 17664)
        self.assertEqual(self.config["engineering_pilots"]["expected_policy_episodes"], 240)
        self.assertEqual(sum(row["family"] == "C1" for row in self.rows), 6144)
        self.assertEqual(sum(row["family"] == "C2" for row in self.rows), 4096)
        for row in self.rows:
            expected = 2 if row["family"] == "C3" else 4 if row["family"] == "C4" else 0
            self.assertEqual(len(row["reuse_episode_ids"]), expected)
            for eid in row["reuse_episode_ids"]:
                control = self.by_id[eid]
                self.assertEqual(control["block_id"], row["block_id"])
                self.assertEqual(control["factors"]["policy"], row["factors"]["policy"])
                self.assertEqual(control["factors"]["goal"], row["factors"]["goal"])
                self.assertEqual(control["prefix_group_id"], row["prefix_group_id"])

    def test_treatment_does_not_change_sampling_seed_or_natural_prefix(self):
        subset = [r for r in self.rows if r["fixture"] == "horizontal" and r["block_id"] == 0
                  and r["factors"]["policy"] == "cosmos3_nano_droid"]
        self.assertEqual(len({r["policy_seed"] for r in subset}), 1)
        self.assertEqual(len({r["env_seed"] for r in subset}), 1)
        same_prompt = [r for r in subset if r["factors"]["goal"] == "left"
                       and r["factors"]["wording"] == "direct"]
        original = [r for r in same_prompt if r["factors"]["scenario"] != "destination_static"]
        destination = next(r for r in same_prompt if r["factors"]["scenario"] == "destination_static")
        self.assertEqual(len({r["prefix_group_id"] for r in original}), 1)
        self.assertNotEqual(destination["prefix_group_id"], original[0]["prefix_group_id"])
        self.assertEqual(len({r["execution_order_key"] for r in original}), len(original))

    def test_model_blind_unstable_seed_substitutions_preserve_blocks(self):
        expected = {
            52: (2100000052, 2100000128),
            101: (2100000101, 2100000129),
        }
        for block_id, (retired, replacement) in expected.items():
            rows = [
                row
                for row in self.rows
                if row["fixture"] == "horizontal" and row["block_id"] == block_id
            ]
            self.assertTrue(rows)
            self.assertEqual({row["env_seed"] for row in rows}, {replacement})
            self.assertEqual(
                {
                    (
                        row["env_seed_substitution"]["retired_seed"],
                        row["env_seed_substitution"]["replacement_seed"],
                    )
                    for row in rows
                },
                {(retired, replacement)},
            )
        horizontal_seeds = {
            row["env_seed"] for row in self.rows if row["fixture"] == "horizontal"
        }
        self.assertNotIn(2100000052, horizontal_seeds)
        self.assertNotIn(2100000101, horizontal_seeds)
        self.assertEqual(len(horizontal_seeds), 128)

    def test_counterbalance_is_equal_across_treatments_and_completes_cycle(self):
        reference_rows = [r for r in self.rows if r["family"] == "C2" and r["block_id"] < 16
                          and r["factors"]["policy"] == "cosmos3_nano_droid"]
        by_block = {}
        for row in reference_rows:
            block = row["block_id"]
            if block in by_block:
                self.assertEqual(row["counterbalance"], by_block[block])
            by_block[block] = row["counterbalance"]
        self.assertEqual(len({(cb["phase_index"], cb["state_index"]) for cb in by_block.values()}), 16)
        self.assertEqual(sum(cb["physical_A_color"] == "blue" for cb in by_block.values()), 8)
        self.assertEqual(sum(cb["physical_translation_sign"] == 1 for cb in by_block.values()), 8)

    def test_bad_count_and_unsupported_reuse_are_rejected(self):
        changed = copy.deepcopy(self.config)
        changed["families"][0]["blocks"] += 1
        self.assertTrue(any("allocation" in e for e in v4.config_errors(changed)))
        changed = copy.deepcopy(self.config)
        changed["families"][2]["reuses"][0]["where"]["scenario"] = ["not_a_control"]
        self.assertTrue(any("reuse filter" in e for e in v4.config_errors(changed)))
        changed = copy.deepcopy(self.config)
        changed["fixtures"]["reference_binding"]["seed_slot"] = 0
        self.assertTrue(any("seed_slot" in e for e in v4.config_errors(changed)))
        changed = copy.deepcopy(self.config)
        changed["seed_reservation"]["post_result_environment_seed_substitutions"][1][
            "replacement_seed"
        ] = 2100000128
        self.assertTrue(
            any("replacement_seed is duplicated" in e for e in v4.config_errors(changed))
        )

    def test_manifest_audit_catches_seed_leakage_and_wrong_control(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["policy_seed"] += 1
        recipient = next(r for r in changed if r["family"] == "C4")
        recipient["reuse_episode_ids"][0] = changed[0]["episode_id"]
        errors = v4.manifest_errors(changed, self.config, self.config_sha)
        self.assertTrue(any("policy_seed" in e for e in errors))
        self.assertTrue(any("reuse_episode_ids" in e for e in errors))

    def test_output_is_deterministic_but_bound_to_raw_config_bytes(self):
        self.assertEqual(v4.manifest_bytes(self.rows),
                         v4.manifest_bytes(v4.build_manifest(self.config, self.config_sha)))
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "campaign.json"
            config_path.write_text(json.dumps(self.config, indent=3))
            reloaded, changed_hash = v4.load_json(config_path)
        self.assertEqual(reloaded, self.config)
        self.assertNotEqual(changed_hash, self.config_sha)
        changed = v4.build_manifest(reloaded, changed_hash)
        self.assertEqual(changed[0]["episode_id"], self.rows[0]["episode_id"])
        self.assertNotEqual(v4.digest_bytes(v4.manifest_bytes(changed)),
                            v4.digest_bytes(v4.manifest_bytes(self.rows)))


class ReleaseAndResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.rows = v4.build_manifest(cls.config, cls.config_sha)
        cls.manifest_sha = v4.digest_bytes(v4.manifest_bytes(cls.rows))

    def released_lock(self, released=("C1",)):
        families = {f["id"]: f for f in self.config["families"]}
        needed_policies = {p for fid in released for p in v4._family_values(families[fid], "policy")}
        fixtures = {families[fid]["fixture"] for fid in released}
        sha = "a" * 64
        return {
            "schema_version": 1, "campaign_id": self.config["campaign_id"],
            "config_sha256": self.config_sha, "manifest_sha256": self.manifest_sha,
            "source_commit": self.config["source_commit"], "release_status": "RELEASED",
            "released_families": list(released),
            "blocked_families": {fid: "Fixture implementation unavailable; allocation preserved."
                                 for fid in families if fid not in released},
            "runner": {"commit": "b" * 40, "entrypoint": "example:run", "sha256": sha},
            "policies": {p: {"checkpoint_sha256": sha, "runtime_image_digest": "sha256:" + sha,
                             "native_control_dt_s": 0.05, "checkpoint_uri": "s3://weights/manifest",
                             "integration_commit": "c" * 40, "achieved_delay_s": 0.10,
                             "achieved_standard_query_period_s": 0.50, "achieved_fast_query_period_s": 0.25,
                             "prediction_horizon_actions": 32,
                             "policy_reset_and_history_contract_uri": "s3://runtime/history.json"}
                         for p in needed_policies},
            "fixtures": {f: {"geometry_sha256": sha, "scorer_sha256": sha,
                              "reset_registry_sha256": sha, "geometry_uri": "s3://fixture/geometry.json",
                              "scorer_uri": "s3://fixture/scorer.json", "calibration_scale": 1.0,
                              "D_cap_m": 1.0, "reset_registry_uri": "s3://fixture/resets.json",
                              "frame_transform_uri": "s3://fixture/frame.json",
                              "goal_geometry_and_tolerances_uri": "s3://fixture/goal.json",
                              "trigger_release_detector_uri": "s3://fixture/detector.json",
                              "intervention_trajectory_registry_uri": "s3://fixture/paths.json",
                              "scoring_and_visibility_thresholds_uri": "s3://fixture/thresholds.json"}
                         for f in fixtures},
            "receipts": {r: {"passed": True, "uri": "s3://receipts/" + r,
                             "sha256": sha, "family_ids": list(released)}
                         for r in self.config["required_release_receipts"]},
        }

    def result(self, row, success=False):
        return {
            "episode_id": row["episode_id"], "attempt_id": "attempt-1", "status": "valid",
            "config_sha256": row["config_sha256"], "prefix_group_id": row["prefix_group_id"],
            "success": success, "trigger_eligible": False, "event_delivered": False,
            "event_observed": False, "outcome": {"goal_violation_capped_m": 0.1,
                "goal_set_empty": False, "goal_violation_cap_applied": False,
                "failure_stage": "none" if success else "pickup"},
            "trace_uri": "s3://data/trace", "video_uri": "s3://data/video",
            "trace_sha256": "a" * 64, "video_sha256": "b" * 64,
            "scorer_sha256": "c" * 64, "protocol_sha256": "d" * 64,
            "response_latency_s": None,
        }

    def test_unqualified_lock_never_releases(self):
        lock = self.released_lock()
        self.assertEqual(v4.release_errors(lock, self.config, self.config_sha, self.manifest_sha), [])
        lock["receipts"]["controlled_clock_and_queue"]["passed"] = False
        lock["fixtures"]["horizontal"]["geometry_sha256"] = None
        lock["runner"]["entrypoint"] = "TODO_IMPLEMENT_RUNNER"
        errors = v4.release_errors(lock, self.config, self.config_sha, self.manifest_sha)
        self.assertTrue(any("passed must be true" in e for e in errors))
        self.assertTrue(any("null" in e for e in errors))
        self.assertTrue(any("unresolved" in e for e in errors))

    def test_partial_release_needs_its_reused_controls(self):
        lock = self.released_lock(("C4",))
        errors = v4.release_errors(lock, self.config, self.config_sha, self.manifest_sha)
        self.assertTrue(any("reused control families" in e for e in errors))
        lock = self.released_lock(("C1", "C3", "C4"))
        self.assertEqual(v4.release_errors(lock, self.config, self.config_sha, self.manifest_sha), [])
        lock["manifest_sha256"] = "0" * 64
        self.assertTrue(any("manifest_sha256" in e for e in
                            v4.release_errors(lock, self.config, self.config_sha, self.manifest_sha)))

    def test_unrealizable_cadence_or_queue_cannot_release(self):
        lock = self.released_lock(("C1", "C3", "C4"))
        policy = lock["policies"]["cosmos3_nano_droid"]
        policy["achieved_fast_query_period_s"] = 0.50
        policy["prediction_horizon_actions"] = 2
        errors = v4.release_errors(lock, self.config, self.config_sha, self.manifest_sha)
        self.assertTrue(any("upward tick quantization" in e for e in errors))
        self.assertTrue(any("queue cannot cover" in e for e in errors))
        self.assertTrue(any("strictly shorter" in e for e in errors))

    def test_behavioral_failure_and_null_latency_are_complete_data(self):
        manifest = self.rows[:2]
        infra = {"episode_id": manifest[0]["episode_id"], "attempt_id": "attempt-0",
                 "status": "infra_invalid", "reason": "Server process exited before request."}
        report = v4.check_results(manifest, [infra] + [self.result(row) for row in manifest])
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["valid_failure_records"], 2)
        self.assertEqual(report["infrastructure_attempts"], 1)

    def test_missing_and_duplicate_accepted_cells_are_not_pooled(self):
        manifest = self.rows[:2]
        first = self.result(manifest[0])
        duplicate = copy.deepcopy(first)
        duplicate["attempt_id"] = "attempt-2"
        report = v4.check_results(manifest, [first, duplicate])
        self.assertFalse(report["ok"])
        self.assertEqual(report["missing_episode_ids"], [manifest[1]["episode_id"]])
        self.assertEqual(report["duplicate_accepted_episode_ids"], [manifest[0]["episode_id"]])

    def test_inconsistent_event_and_nonfinite_metric_fail(self):
        row = self.result(self.rows[0])
        row["event_observed"] = True
        row["outcome"]["goal_violation_capped_m"] = float("nan")
        report = v4.check_results(self.rows[:1], [row])
        self.assertTrue(any("observed event was not delivered" in e for e in report["errors"]))
        self.assertTrue(any("finite and nonnegative" in e for e in report["errors"]))

    def test_empty_goal_and_blocked_cells_cannot_be_counted_as_success(self):
        row = self.result(self.rows[0], success=True)
        row["outcome"]["goal_set_empty"] = True
        report = v4.check_results(self.rows[:1], [row])
        self.assertTrue(any("empty goal set" in e for e in report["errors"]))
        self.assertTrue(any("positive geometric goal violation" in e for e in report["errors"]))
        blocked = {"episode_id": self.rows[0]["episode_id"], "attempt_id": "blocked-1",
                   "status": "blocked", "reason": "Fixture not released."}
        report = v4.check_results(self.rows[:1], [blocked])
        self.assertFalse(report["ok"])
        self.assertEqual(report["accepted_unique"], 0)


if __name__ == "__main__":
    unittest.main()
