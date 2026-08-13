"""Focused fail-closed tests for the additive C002 publication closure."""

import json
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError
import tools.validate_v3c002r001_post_publication_closure as closure


class PostPublicationClosureTests(unittest.TestCase):
    def test_pinned_commits_and_exact_eleven_file_scope(self):
        self.assertEqual(closure.SOURCE_COMMIT, "785ea96419df51b92249ef7cdd1b5dbd59ff0a50")
        self.assertEqual(closure.RESULTS_COMMIT, "700a1f76a2f8ec2ac8e19db669c9afe3668f8a85")
        self.assertEqual(len(closure.PUBLISHED_PATHS), 11)
        self.assertEqual(len(set(closure.PUBLISHED_PATHS)), 11)
        self.assertTrue(all(str(path).startswith(str(closure.FINAL)) for path in closure.PUBLISHED_PATHS))

    def _receipt_fixture(self, root: Path):
        final = Path("final")
        mapping = {
            "compiled_episodes": final / "results/episodes.jsonl", "pairs": final / "results/pairs.jsonl",
            "results": final / "results/results.json", "epoch_diagnostics": final / "results/epoch_diagnostics.json",
            "evidence_manifest": final / "results/evidence_manifest.json", "decision_memo": final / "results/DECISION_MEMO.md",
            "manuscript_insert": final / "results/MANUSCRIPT_INSERT.md", "raw_aggregation_receipt": final / "raw_aggregation_receipt.json",
            "infrastructure_attempts": final / "infrastructure_attempts.jsonl", "invocation_validator_stdout": final / "validator.stdout.json",
        }
        bindings = {}
        receipt_bindings = {}
        for index, (name, relative) in enumerate(mapping.items()):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"value-{index}\n", encoding="utf-8")
            binding = closure._binding(path)
            bindings[relative.as_posix()] = binding
            receipt_bindings[name] = copy.deepcopy(binding)
        receipt = {
            "passed": True, "exit_code": 0, "valid_behavioral_episodes": 1364,
            "complete_seed_blocks": 341, "prompt_form_pairs": 682,
            "infrastructure_attempts_excluded": 14, "bindings": receipt_bindings,
        }
        receipt_path = root / final / "finalization_execution_receipt.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return final, bindings, receipt

    def test_execution_receipt_rejects_mutated_binding_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final, bindings, receipt = self._receipt_fixture(root)
            with patch.object(closure, "REPO_ROOT", root), patch.object(closure, "FINAL", final):
                self.assertEqual(closure.validate_execution_receipt(bindings)["complete_seed_blocks"], 341)
                receipt["bindings"]["results"]["sha256"] = "0" * 64
                (root / final / "finalization_execution_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(ContractError, "execution receipt binding changed"):
                    closure.validate_execution_receipt(bindings)
                receipt["bindings"]["results"] = bindings[(final / "results/results.json").as_posix()]
                receipt["complete_seed_blocks"] = 340
                (root / final / "finalization_execution_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
                with self.assertRaisesRegex(ContractError, "execution receipt changed"):
                    closure.validate_execution_receipt(bindings)


if __name__ == "__main__":
    unittest.main()
