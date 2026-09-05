"""Tests for prospective V4 design/freeze builder and validator."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
            for name in (
                "protocol.json",
                "prompt_manifest.json",
                "motion_manifest.json",
                "scoring_manifest.json",
                "seed_manifest.json",
                "queue.jsonl",
                "queue_manifest.json",
                "frozen_analysis_manifest.json",
                "gate_report.json",
                "continuation_state.json",
                "historical_protocol_ledger.json",
                "freeze_manifest.json",
            ):
                self.assertTrue((out / name).is_file(), name)

    def test_queue_prompts_resolve_symbolically_except_second_stack(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            rows = validator.read_jsonl(out / "queue.jsonl")
            horizontal = next(r for r in rows if r["fixture"] == "horizontal")
            self.assertIn("the cube", horizontal["prompt_text"])
            self.assertIn("the bowl", horizontal["prompt_text"])
            self.assertTrue(horizontal["launch_critical_names_resolved"])
            bridge = next(r for r in rows if r["fixture"] == "second_stack")
            self.assertIn("UNRESOLVED", bridge["prompt_text"])
            self.assertFalse(bridge["launch_critical_names_resolved"])

    def test_gate_report_blocks_c2_and_c8_with_exact_reasons(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            gate = json.loads((out / "gate_report.json").read_text())
            self.assertEqual(gate["families"]["C8"]["lifecycle_status"], "BLOCKED_RUNTIME")
            self.assertIn("GR00T", gate["families"]["C8"]["block_reason"])
            self.assertIn("common-prefix", gate["families"]["C2"]["block_reason"].lower())
            self.assertEqual(gate["release_status"], "NOT_RELEASED")
            for name, receipt in gate["required_release_receipts"].items():
                if name == "historical_seed_collision_audit":
                    self.assertTrue(receipt["passed"])
                else:
                    self.assertFalse(receipt["passed"])


class FreezeValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_path = ROOT / "docs/online_correction_v4/campaign.json"
        cls.artifact_dir = ROOT / "artifacts/online_correction_v4"
        if not (cls.artifact_dir / "queue.jsonl").is_file():
            builder.build_freeze(cls.config_path, cls.artifact_dir)

    def test_committed_freeze_passes_validator(self):
        report = validator.validate_online_correction_v4(self.artifact_dir, self.config_path)
        self.assertTrue(report["ok"], report.get("errors"))

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

    def test_fake_runtime_receipt_pass_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            gate = json.loads((out / "gate_report.json").read_text())
            gate["required_release_receipts"]["controlled_clock_and_queue"]["passed"] = True
            (out / "gate_report.json").write_text(json.dumps(gate, indent=2) + "\n")
            report = validator.validate_online_correction_v4(out, self.config_path)
            self.assertFalse(report["ok"])
            self.assertTrue(any("fake-pass" in e for e in report["errors"]))

    def test_queue_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            builder.build_freeze(self.config_path, out)
            manifest = json.loads((out / "queue_manifest.json").read_text())
            manifest["queue_sha256"] = "0" * 64
            (out / "queue_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            report = validator.validate_online_correction_v4(out, self.config_path)
            self.assertFalse(report["ok"])
            self.assertTrue(any("queue_sha256" in e for e in report["errors"]))

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


class FreezeDeterminismTests(unittest.TestCase):
    def test_rebuild_is_byte_stable_for_queue(self):
        config_path = ROOT / "docs/online_correction_v4/campaign.json"
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            out1 = Path(tmp1)
            out2 = Path(tmp2)
            r1 = builder.build_freeze(config_path, out1)
            r2 = builder.build_freeze(config_path, out2)
            self.assertEqual(r1["queue_sha256"], r2["queue_sha256"])
            self.assertEqual((out1 / "queue.jsonl").read_bytes(), (out2 / "queue.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
