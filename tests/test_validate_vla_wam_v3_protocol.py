"""Tests for the VLA/WAM v3 fail-closed protocol validator."""

from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "tools" / "validate_vla_wam_v3_protocol.py"
SPEC = importlib.util.spec_from_file_location("validate_v3", VALIDATOR_PATH)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class ValidateV3ProtocolTest(unittest.TestCase):
    def copy_protocol_root(self, root: Path) -> None:
        shutil.copytree(ROOT / "artifacts" / "vla_wam_shared_v3", root / "artifacts" / "vla_wam_shared_v3")
        (root / "docs").mkdir()
        shutil.copy2(ROOT / "docs" / "VLA_WAM_STEERABILITY_V3_PROTOCOL.md", root / "docs")

    def test_checked_in_protocol_passes(self) -> None:
        checks = VALIDATOR.validate(ROOT)
        self.assertGreaterEqual(len(checks), 40)

    def test_tampered_sampling_seed_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "stochastic_rollout_registry.json"
            value = json.loads(path.read_text())
            value["shared_sampling_seed_indices"][15] = 9999
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "sixteen shared"):
                VALIDATOR.validate(root)

    def test_tampered_phase_a_queue_fails_hash_check(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "phase_a_cells.jsonl"
            path.write_text(path.read_text().replace('"status":"authorized_new"', '"status":"tampered"', 1))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "queue manifest hash"):
                VALIDATOR.validate(root)

    def test_tampered_detached_release_contract_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_protocol_root(root)
            path = root / "artifacts" / "vla_wam_shared_v3" / "failure_taxonomy.json"
            value = json.loads(path.read_text())
            value["scorer_consistency_rules"]["prohibited_inference"] = "Use requested_success as detached release."
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "detached release"):
                VALIDATOR.validate(root)


if __name__ == "__main__":
    unittest.main()
