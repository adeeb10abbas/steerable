"""Tests for prospective V4 design/freeze builder and validator."""

import copy
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DUPLICATE_ARTICLE_RE = re.compile(r"\bthe the\b", re.I)


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load("build_online_correction_v4_freeze", "tools/build_online_correction_v4_freeze.py")
validator = _load("validate_online_correction_v4", "tools/validate_online_correction_v4.py")
v4 = _load("online_correction_v4", "tools/online_correction_v4.py")


class FreezeBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "docs/online_correction_v4/campaign.json"
        cls.artifact_dir = ROOT / "artifacts/online_correction_v4"

    def test_build_produces_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = builder.build_freeze(self.config_path, out)
            self.assertTrue(report["ok"])
            self.assertEqual(report["row_count"], 17664)
            self.assertTrue(report["seed_collision_audit_passed"])
            self.assertNotEqual(report["planning_manifest_sha256"], report["frozen_queue_sha256"])
            for name in builder.ALL_GENERATED_FREEZE_ARTIFACTS:
                self.assertTrue((out / name).is_file(), name)

    def test_prompts_have_no_duplicate_articles(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            rows = validator.read_jsonl(out / "queue.jsonl")
            bad = [r for r in rows if DUPLICATE_ARTICLE_RE.search(r.get("prompt_text", ""))]
            self.assertEqual(bad, [], bad[:1])

    def test_queue_prompts_resolve_symbolically_except_second_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            rows = validator.read_jsonl(out / "queue.jsonl")
            horizontal = next(r for r in rows if r["fixture"] == "horizontal")
            self.assertIn("Place the cube so that the cube is left of the bowl.", horizontal["prompt_text"])
            self.assertTrue(horizontal["launch_critical_names_resolved"])
            bridge = next(r for r in rows if r["fixture"] == "second_stack")
            self.assertIn("UNRESOLVED", bridge["prompt_text"])
            self.assertFalse(bridge["launch_critical_names_resolved"])

    def test_c2_color_follows_counterbalance(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            rows = [r for r in validator.read_jsonl(out / "queue.jsonl") if r["family"] == "C2"]
            for row in rows:
                color_a = row["counterbalance"]["physical_A_color"]
                expected = color_a if row["factors"]["named_reference"] == "A" else (
                    "yellow" if color_a == "blue" else "blue"
                )
                self.assertEqual(row["reference_color"], expected)
                self.assertIn(f"the {expected} bowl", row["prompt_text"])

    def test_gate_report_splits_hard_blocked_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            gate = json.loads((out / "gate_report.json").read_text())
            seed = json.loads((out / "seed_manifest.json").read_text())
            self.assertEqual(set(gate["hard_blocked_families"]), {"C2", "C8"})
            self.assertEqual(set(gate["pending_not_released_families"]), {"C1", "C3", "C4", "C5", "C6", "C7"})
            self.assertEqual(gate["families"]["C8"]["lifecycle_status"], "BLOCKED_RUNTIME")
            self.assertEqual(gate["release_status"], "NOT_RELEASED")
            receipt = gate["required_release_receipts"]["historical_seed_collision_audit"]
            audit = seed["historical_collision_audit"]
            self.assertEqual(receipt["passed"], audit["passed"])
            self.assertEqual(receipt["derived_from"], "seed_manifest.historical_collision_audit")
            self.assertEqual(
                receipt["audit_summary"]["env_collision_count"],
                len(audit.get("env_collisions", [])),
            )

    def test_continuation_authority_includes_freeze_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            continuation = json.loads((out / "continuation_state.json").read_text())
            files = continuation["authoritative_files"]
            self.assertIn("artifacts/online_correction_v4/freeze_manifest.json", files)
            self.assertIn("artifacts/online_correction_v4/continuation_state.json", files)
            receipts = continuation["qualification_receipts"]
            self.assertEqual(
                {Path(row["path"]).name for row in receipts},
                {
                    path.name
                    for path in (
                        ROOT / "artifacts/online_correction_v4/qualification"
                    ).glob("*.json")
                },
            )
            for receipt in receipts:
                receipt_path = ROOT / receipt["path"]
                self.assertIn(receipt["path"], files)
                self.assertEqual(
                    receipt["sha256"], builder.file_sha256(receipt_path)
                )
                self.assertEqual(receipt["behavioral_episode_count"], 0)
            candidates = continuation["model_blind_candidates"]
            self.assertEqual(
                {Path(row["path"]).name for row in candidates},
                {
                    "horizontal_g2_recalibration_amendment.candidate.json",
                    "horizontal_g2_seed_substitution_amendment.candidate.json",
                    "horizontal_g3_plan.candidate.json",
                    "horizontal_reset_registry.candidate.json",
                },
            )
            for candidate in candidates:
                candidate_path = ROOT / candidate["path"]
                self.assertIn(candidate["path"], files)
                self.assertEqual(
                    candidate["sha256"],
                    builder.file_sha256(candidate_path),
                )
                self.assertEqual(candidate["model_request_count"], 0)

    def test_prompt_sha256_may_map_to_multiple_prompt_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            prompt_manifest = json.loads((out / "prompt_manifest.json").read_text())
            semantics = prompt_manifest["prompt_identity_semantics"]
            self.assertTrue(semantics["prompt_sha256_may_map_to_multiple_prompt_ids"])
            self.assertIn("prompt_sha256", semantics["analysis_forbidden_primary_keys"])
            sha_to_ids: dict[str, set[str]] = {}
            for item in prompt_manifest["prompts"]:
                sha_to_ids.setdefault(item["prompt_sha256"], set()).add(item["prompt_id"])
            shared = {sha for sha, ids in sha_to_ids.items() if len(ids) > 1}
            self.assertGreater(len(shared), 0, "C2 counterbalance should produce shared prompt_sha256 values")

    def test_all_generated_artifacts_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = builder.build_freeze(self.config_path, out)
            freeze = json.loads((out / "freeze_manifest.json").read_text())
            expected_in_index = set(builder.ALL_GENERATED_FREEZE_ARTIFACTS) - {
                builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED
            }
            self.assertEqual(set(freeze["artifact_sha256"]), expected_in_index)
            self.assertIn(builder.FREEZE_MANIFEST_SELF_HASH_EXCLUDED, report["artifact_sha256"])

    def test_prompt_sha256_byte_identity_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            prompt_manifest = json.loads((out / "prompt_manifest.json").read_text())
            semantics = prompt_manifest["prompt_identity_semantics"]
            self.assertEqual(
                semantics["prompt_sha256_rule"],
                "sha256(utf8(prompt_text)); identical iff byte-identical resolved text",
            )
            self.assertTrue(semantics["prompt_sha256_must_be_unique_when_text_differs"])
            rows = validator.read_jsonl(out / "queue.jsonl")
            text_to_sha: dict[str, str] = {}
            for row in rows:
                expected = v4.digest_bytes(row["prompt_text"].encode("utf-8"))
                self.assertEqual(row["prompt_sha256"], expected)
                if row["prompt_text"] in text_to_sha:
                    self.assertEqual(text_to_sha[row["prompt_text"]], row["prompt_sha256"])
                else:
                    text_to_sha[row["prompt_text"]] = row["prompt_sha256"]

    def test_seed_receipt_status_derives_from_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            gate = json.loads((out / "gate_report.json").read_text())
            seed = json.loads((out / "seed_manifest.json").read_text())
            audit = seed["historical_collision_audit"]
            receipt = gate["required_release_receipts"]["historical_seed_collision_audit"]
            computed_passed = len(audit["env_collisions"]) == 0 and len(audit["policy_collisions"]) == 0
            self.assertEqual(audit["passed"], computed_passed)
            self.assertEqual(receipt["passed"], computed_passed)
            expected_status = "passed_at_freeze_build" if computed_passed else "failed_at_freeze_build"
            self.assertEqual(receipt["status"], expected_status)

    def test_runtime_and_launch_stubs_match_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            config, _ = v4.load_json(self.config_path)
            runtime = json.loads((out / "runtime_manifest.json").read_text())
            launch = json.loads((out / "launch_matrix.json").read_text())
            self.assertEqual(set(runtime["policies"]), set(config["policies"]))
            self.assertEqual(runtime["campaign_id"], config["campaign_id"])
            self.assertEqual(launch["campaign_id"], config["campaign_id"])
            self.assertIsNone(launch["cluster_context"])

    def test_family_dispositions_match_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            gate = json.loads((out / "gate_report.json").read_text())
            continuation = json.loads((out / "continuation_state.json").read_text())
            self.assertEqual(
                set(gate["hard_blocked_families"]),
                set(continuation["hard_blocked_families"]),
            )
            self.assertEqual(
                set(gate["pending_not_released_families"]),
                set(continuation["pending_not_released_families"]),
            )

    def test_setup_fixture_keys_match_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            config, _ = v4.load_json(self.config_path)
            setup = json.loads((out / "setup_manifest.json").read_text())
            self.assertEqual(set(setup["fixtures"]), set(config["fixtures"]))

    def test_reuse_metrics_distinguish_edges_and_unique_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            manifest = json.loads((out / "queue_manifest.json").read_text())
            self.assertEqual(manifest["total_control_reference_edges_by_family"]["C3"], 3072)
            self.assertEqual(manifest["unique_referenced_control_episode_ids_by_family"]["C3"], 1024)
            self.assertEqual(manifest["total_control_reference_edges_by_family"]["C4"], 8192)
            self.assertEqual(manifest["unique_referenced_control_episode_ids_by_family"]["C4"], 2048)

    def test_all_rows_are_registered_new_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            rows = validator.read_jsonl(out / "queue.jsonl")
            self.assertTrue(all(r["queue_row_kind"] == "new_episode" for r in rows))
            c4_fast = [
                r for r in rows
                if r["family"] == "C4"
                and r["factors"]["schedule"] == "fast_after_grasp"
                and r["factors"]["scenario"] in ("original_sham", "move_stop")
            ]
            self.assertGreater(len(c4_fast), 0)
            self.assertTrue(all(r["episode_id"] for r in c4_fast))


class FreezeValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "docs/online_correction_v4/campaign.json"
        cls.artifact_dir = ROOT / "artifacts/online_correction_v4"
        if not (cls.artifact_dir / "queue.jsonl").is_file():
            builder.build_freeze(cls.config_path, cls.artifact_dir)

    def test_prompt_sha256_collision_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            prompt_manifest = json.loads((out / "prompt_manifest.json").read_text())
            prompts = prompt_manifest["prompts"]
            prompts[0]["prompt_sha256"] = prompts[1]["prompt_sha256"]
            if prompts[0]["prompt_text"] != prompts[1]["prompt_text"]:
                (out / "prompt_manifest.json").write_text(json.dumps(prompt_manifest, indent=2, sort_keys=True) + "\n")
                report = validator.validate_online_correction_v4(out, self.config_path)
                self.assertFalse(report["ok"])
                self.assertTrue(any("collision" in e.lower() or "mismatch" in e.lower() for e in report["errors"]))

    def test_seed_receipt_audit_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            gate = json.loads((out / "gate_report.json").read_text())
            gate["required_release_receipts"]["historical_seed_collision_audit"]["passed"] = False
            (out / "gate_report.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
            report = validator.validate_online_correction_v4(out, self.config_path)
            self.assertFalse(report["ok"])
            self.assertTrue(any("seed receipt" in e.lower() for e in report["errors"]))

    def test_committed_freeze_passes_validator(self):
        report = validator.validate_online_correction_v4(self.artifact_dir, self.config_path)
        self.assertTrue(report["ok"], report.get("errors"))
        self.assertEqual(report["duplicate_article_prompt_count"], 0)
        self.assertEqual(report["c2_color_mismatch_count"], 0)

    def test_duplicate_article_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            rows = validator.read_jsonl(out / "queue.jsonl")
            rows[0]["prompt_text"] = "Place the the cube."
            out.joinpath("queue.jsonl").write_text(
                "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
            )
            report = validator.validate_online_correction_v4(out, self.config_path)
            self.assertFalse(report["ok"])
            self.assertTrue(any("duplicate article" in e for e in report["errors"]))

    def test_historical_protocol_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            ledger = json.loads((out / "historical_protocol_ledger.json").read_text())
            ledger["entries"][0]["sha256"] = "0" * 64
            (out / "historical_protocol_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
            report = validator.validate_online_correction_v4(out, self.config_path)
            self.assertFalse(report["ok"])
            self.assertTrue(any("historical protocol bytes changed" in e for e in report["errors"]))

    def test_control_reuse_integrity(self):
        rows = validator.read_jsonl(self.artifact_dir / "queue.jsonl")
        by_id = {row["episode_id"]: row for row in rows}
        for row in rows:
            for eid in row.get("reuse_episode_ids", []):
                control = by_id[eid]
                self.assertEqual(control["block_id"], row["block_id"])
                self.assertEqual(control["factors"]["policy"], row["factors"]["policy"])
                self.assertEqual(control["factors"]["goal"], row["factors"]["goal"])
                self.assertEqual(control["prefix_group_id"], row["prefix_group_id"])
                self.assertNotEqual(control["episode_id"], row["episode_id"])


class FreezeDeterminismTests(unittest.TestCase):
    def test_rebuild_is_byte_stable_for_queue(self):
        config_path = ROOT / "docs/online_correction_v4/campaign.json"
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            out1 = Path(tmp1)
            out2 = Path(tmp2)
            r1 = builder.build_freeze(config_path, out1)
            r2 = builder.build_freeze(config_path, out2)
            self.assertEqual(r1["frozen_queue_sha256"], r2["frozen_queue_sha256"])
            self.assertEqual(r1["planning_manifest_sha256"], r2["planning_manifest_sha256"])
            self.assertEqual((out1 / "queue.jsonl").read_bytes(), (out2 / "queue.jsonl").read_bytes())

    def test_rebuild_matches_all_generated_artifact_hashes(self):
        config_path = ROOT / "docs/online_correction_v4/campaign.json"
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            r1 = builder.build_freeze(config_path, Path(tmp1))
            r2 = builder.build_freeze(config_path, Path(tmp2))
            for name in builder.ALL_GENERATED_FREEZE_ARTIFACTS:
                self.assertEqual(
                    r1["artifact_sha256"][name],
                    r2["artifact_sha256"][name],
                    name,
                )

    def test_rebuild_matches_all_deterministic_artifact_hashes(self):
        config_path = ROOT / "docs/online_correction_v4/campaign.json"
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            r1 = builder.build_freeze(config_path, Path(tmp1))
            r2 = builder.build_freeze(config_path, Path(tmp2))
            for name in builder.DETERMINISTIC_ARTIFACT_NAMES:
                self.assertEqual(
                    r1["artifact_sha256"][name],
                    r2["artifact_sha256"][name],
                    name,
                )


if __name__ == "__main__":
    unittest.main()
