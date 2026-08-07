from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools.validate_v3_checkpoint_provenance import (
    EXPECTED_MODEL_IDS,
    ProvenanceValidationError,
    validate_checkpoint_provenance,
    validate_instance,
)


ROOT = Path(__file__).resolve().parents[1]


class CheckpointProvenanceTest(unittest.TestCase):
    def test_full_release_validates(self) -> None:
        checks = validate_checkpoint_provenance(ROOT)
        self.assertGreaterEqual(len(checks), 18)

    def test_table_has_exact_identity_set(self) -> None:
        path = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance/checkpoint_provenance_table.json"
        table = json.loads(path.read_text())
        self.assertEqual({row["model_id"] for row in table["records"]}, EXPECTED_MODEL_IDS)

    def test_schema_rejects_implicit_missing_runtime_identity(self) -> None:
        schema_path = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance.schema.json"
        schema = json.loads(schema_path.read_text())
        record_path = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance/pi05_current_stack_droid.json"
        record = json.loads(record_path.read_text())
        del record["checkpoint_identity"]["runtime_identity_sha256"]
        with self.assertRaises(ProvenanceValidationError):
            validate_instance(record, schema, schema)

    def test_schema_rejects_false_unknown_hash(self) -> None:
        schema_path = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance.schema.json"
        schema = json.loads(schema_path.read_text())
        record_path = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/checkpoint_provenance/pi0_fast_droid_vla.json"
        record = json.loads(record_path.read_text())
        record["checkpoint_identity"]["runtime_identity_sha256"] = ""
        with self.assertRaises(ProvenanceValidationError):
            validate_instance(record, schema, schema)


if __name__ == "__main__":
    unittest.main()
