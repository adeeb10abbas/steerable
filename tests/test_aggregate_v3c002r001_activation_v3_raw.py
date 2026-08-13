from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError
from tools import aggregate_v3c002r001_activation_v3_raw as aggregate


class ActivationV3RawAggregationTests(unittest.TestCase):
    def test_lane_root_parser_requires_all_eight_unique_slots(self) -> None:
        values = [f"{slot}=/tmp/{slot}" for slot in aggregate.LANE_SLOTS]
        parsed = aggregate._parse_lane_roots(values)
        self.assertEqual(tuple(sorted(parsed)), aggregate.LANE_SLOTS)
        with self.assertRaisesRegex(ContractError, "exactly eight"):
            aggregate._parse_lane_roots(values[:-1])
        with self.assertRaisesRegex(ContractError, "repeats"):
            aggregate._parse_lane_roots(values + [values[0]])

    def test_jsonl_reader_rejects_blank_and_nonobject_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            blank = Path(directory) / "blank.jsonl"; blank.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "blank"):
                aggregate._read_jsonl(blank, "test")
            scalar = Path(directory) / "scalar.jsonl"; scalar.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "not an object"):
                aggregate._read_jsonl(scalar, "test")

    def test_output_writer_refuses_any_existing_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"; raw.mkdir()
            (raw / "episodes.jsonl").write_text("existing\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "refuses to overwrite"):
                aggregate._write_outputs(root, [], [], {"schema_version": "test"})

    def test_output_writer_binds_both_combined_ledgers_before_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hashes = aggregate._write_outputs(root, [{"cell_id": "x"}], [{"cell_id": "y"}], {"schema_version": "test"})
            receipt = json.loads((root / "raw/aggregation_receipt.json").read_text(encoding="utf-8"))
            self.assertEqual(receipt["combined_outputs"]["raw_episodes"]["sha256"], hashes["episodes"])
            self.assertEqual(receipt["combined_outputs"]["infrastructure_attempts"]["sha256"], hashes["infrastructure_attempts"])


if __name__ == "__main__":
    unittest.main()
