from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from tools import validate_v3_final_merge_reconciliation as reconciliation


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT
    / "artifacts/vla_wam_shared_v3/final_merge_reconciliation/validator_reconciliation.json"
)


class FinalMergeReconciliationTests(unittest.TestCase):
    def test_only_the_registered_cross_experiment_path_is_allowed(self) -> None:
        reconciliation.assert_only_allowed_conflict(
            {"tools/validate_v3e_publication_bundle.py", "experiments/c002_only.py"},
            {"tools/validate_v3e_publication_bundle.py", "experiments/e006_only.py"},
            {"tools/validate_v3e_publication_bundle.py"},
        )

    def test_unregistered_cross_experiment_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(reconciliation.ReconciliationError, "intersection differs"):
            reconciliation.assert_only_allowed_conflict(
                {"tools/validate_v3e_publication_bundle.py", "tools/unexpected.py"},
                {"tools/validate_v3e_publication_bundle.py", "tools/unexpected.py"},
                {"tools/validate_v3e_publication_bundle.py"},
            )

    def test_bad_archival_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "copy.py"
            path.write_bytes(b"frozen bytes\n")
            binding = {
                "path": "copy.py",
                "bytes": path.stat().st_size,
                "sha256": "0" * 64,
            }
            with self.assertRaisesRegex(reconciliation.ReconciliationError, "SHA-256 differs"):
                reconciliation.verify_binding(root, binding, "bad archival copy")

    def test_archival_copies_match_exact_branch_tip_blobs(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for experiment in manifest["experiments"]:
            frozen = experiment["frozen_validator"]
            archived = ROOT / frozen["archival_copy"]["path"]
            expected = reconciliation.git_blob(
                ROOT, experiment["branch_tip_commit"], frozen["canonical_path"]
            )
            self.assertEqual(archived.read_bytes(), expected)
            self.assertEqual(reconciliation.sha256_file(archived), frozen["sha256"])

    def test_static_reconciliation_rejects_no_nonconflicting_binding(self) -> None:
        value = reconciliation.validate_reconciliation(ROOT, run_overlays=False)
        self.assertEqual(value["status"], "valid_nonmutating_v3_final_merge_reconciliation")
        self.assertTrue(value["verified_nonconflicting_bindings"])

    def test_disposable_overlay_worktree_is_removed(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        c002 = next(row for row in manifest["experiments"] if row["experiment_id"] == "V3-C002")
        frozen = c002["frozen_validator"]
        before = reconciliation.worktree_paths(ROOT)
        with reconciliation.temporary_overlaid_worktree(
            ROOT,
            label="test-c002",
            archived_relative=frozen["archival_copy"]["path"],
            expected_sha256=frozen["sha256"],
        ) as worktree:
            self.assertTrue(worktree.is_dir())
            self.assertEqual(
                reconciliation.sha256_file(
                    worktree / "tools/validate_v3e_publication_bundle.py"
                ),
                frozen["sha256"],
            )
        self.assertEqual(reconciliation.worktree_paths(ROOT), before)


if __name__ == "__main__":
    unittest.main()
