import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


MODULE = (
    Path(__file__).resolve().parents[1]
    / "analysis/audit_development_request_recovery_lineage.py"
)
REPORT = (
    Path(__file__).resolve().parents[1]
    / "results/development_request_recovery_lineage.json"
)
spec = importlib.util.spec_from_file_location("request_recovery_lineage", MODULE)
lineage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lineage)


class RequestRecoveryLineageTests(unittest.TestCase):
    def test_saved_report_is_canonical_and_keeps_all_gates_closed(self):
        report = lineage.build_audit()
        self.assertEqual(REPORT.read_bytes(), lineage.canonical_bytes(report))
        self.assertEqual(report["global_study_snapshot"]["planned_core_cells"], 232)
        self.assertEqual(len(report["request_lineage"]["rows"]), 1152)
        self.assertEqual(len(report["cell_lineage"]["rows"]), 32)
        self.assertEqual(len(report["recovery_manifest"]["dependencies"]), 56)
        self.assertEqual(len(report["recovery_scope_edges"]["edges"]), 160)
        self.assertEqual(
            len(report["study_scope_dependencies"]["compiler_dependency_ids"]), 8
        )
        cell_dependency_ids = {
            row["dependency_id"]
            for row in report["recovery_scope_edges"]["edges"]
        }
        compiler_dependency_ids = set(
            report["study_scope_dependencies"]["compiler_dependency_ids"]
        )
        all_dependency_ids = {
            row["dependency_id"]
            for row in report["recovery_manifest"]["dependencies"]
        }
        self.assertEqual(len(cell_dependency_ids), 48)
        self.assertTrue(cell_dependency_ids.isdisjoint(compiler_dependency_ids))
        self.assertEqual(
            cell_dependency_ids | compiler_dependency_ids, all_dependency_ids
        )
        pending = report["pending_raw_verification"]
        self.assertEqual(pending["raw_paths_opened"], 0)
        self.assertEqual(pending["raw_bytes_read"], 0)
        self.assertIsNone(pending["raw_descriptor_count_verified"])
        self.assertIsNone(pending["raw_descriptor_mismatch_count"])
        self.assertIs(pending["retrieval_released"], False)
        self.assertIs(pending["retrieval_queue_descriptor_created"], False)
        self.assertIs(pending["safe_to_release_new_work"], False)
        self.assertTrue(all(value == 0 for value in report["new_activity"].values()))
        self.assertIs(
            report["annotation_and_release_boundary"][
                "safe_to_release_confirmation"
            ],
            False,
        )

    def synthetic_report(self, contents):
        dependencies = []
        for ordinal, (relative, payload) in enumerate(contents.items()):
            dependencies.append(
                {
                    "dependency_id": f"dependency-{ordinal:02d}",
                    "descriptor": {
                        "bytes": len(payload),
                        "path": str(lineage.PVC_ROOT / relative),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    },
                }
            )
        return {
            "input_pins": {},
            "recovery_manifest": {
                "dependencies": dependencies,
                "dependency_index": {},
            },
        }

    def populate(self, root, contents):
        for relative, payload in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    def verify(self, root, contents):
        report = self.synthetic_report(contents)
        with patch.object(lineage, "EXPECTED_DEPENDENCIES", len(contents)), patch.object(
            lineage, "EXPECTED_DEPENDENCY_BYTES", sum(map(len, contents.values()))
        ):
            return lineage.verify_bundle(report, root)

    def test_synthetic_exact_bundle_passes(self):
        contents = {"behavioral/a.json": b"alpha\n", "control/b.npz": b"beta\0"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, contents)
            receipt = self.verify(root, contents)
        self.assertIs(receipt["verification_passed"], True)
        self.assertEqual(receipt["summary"]["verified_descriptors"], 2)
        self.assertEqual(receipt["summary"]["verified_bytes"], 11)
        self.assertIs(
            receipt["content_and_science_boundary"][
                "current_live_PVC_availability_verified"
            ],
            False,
        )
        self.assertIs(receipt["release_boundary"]["queue_descriptor_released"], False)

    def test_any_metadata_failure_prevents_all_hashing(self):
        contents = {"a/one.json": b"one", "b/two.npz": b"two"}
        mutations = ("missing", "extra_file", "extra_directory", "size", "symlink")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.populate(root, contents)
                if mutation == "missing":
                    (root / "a/one.json").unlink()
                elif mutation == "extra_file":
                    (root / "extra").write_bytes(b"x")
                elif mutation == "extra_directory":
                    (root / "empty").mkdir()
                elif mutation == "size":
                    (root / "b/two.npz").write_bytes(b"too-long")
                else:
                    (root / "b/two.npz").unlink()
                    os.symlink(root / "a/one.json", root / "b/two.npz")
                real_open = os.open
                proc_fd_read_opens = []

                def record_open(path, *args, **kwargs):
                    if isinstance(path, str) and path.startswith("/proc/self/fd/"):
                        proc_fd_read_opens.append(path)
                    return real_open(path, *args, **kwargs)

                with patch.object(lineage.os, "open", side_effect=record_open):
                    receipt = self.verify(root, contents)
                self.assertIs(receipt["verification_passed"], False)
                self.assertEqual(receipt["summary"]["verified_descriptors"], 0)
                self.assertEqual(receipt["summary"]["verified_bytes"], 0)
                self.assertEqual(
                    receipt["post_hash_tree_check"]["barrier_bytes_read"], 0
                )
                self.assertEqual(proc_fd_read_opens, [])

    def test_hash_drift_and_hardlinks_fail_closed(self):
        contents = {"a/one.json": b"one", "b/two.npz": b"two"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, contents)
            (root / "b/two.npz").write_bytes(b"TWO")
            receipt = self.verify(root, contents)
            self.assertIs(receipt["verification_passed"], False)
            self.assertEqual(receipt["summary"]["status_counts"]["sha256_mismatch"], 1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, contents)
            (root / "b/two.npz").unlink()
            os.link(root / "a/one.json", root / "b/two.npz")
            receipt = self.verify(root, contents)
            self.assertIs(receipt["verification_passed"], False)
            self.assertEqual(receipt["summary"]["hardlink_paths"], 2)
            self.assertEqual(receipt["summary"]["duplicate_inode_paths"], 2)
            self.assertEqual(receipt["summary"]["verified_descriptors"], 0)
            self.assertEqual(
                receipt["post_hash_tree_check"]["barrier_bytes_read"], 0
            )

    def test_late_tree_mutation_and_fifo_substitution_fail_closed(self):
        contents = {"a/one.json": b"one", "b/two.npz": b"two"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, contents)
            original = lineage.verify_one_file
            mutated = False

            def add_late_extra(*args, **kwargs):
                nonlocal mutated
                result = original(*args, **kwargs)
                if not mutated:
                    (root / "late-extra").write_bytes(b"late")
                    mutated = True
                return result

            with patch.object(lineage, "verify_one_file", side_effect=add_late_extra):
                receipt = self.verify(root, contents)
            self.assertIs(receipt["verification_passed"], False)
            self.assertIs(
                receipt["summary"]["tree_mutated_during_verification"], True
            )
            self.assertEqual(receipt["summary"]["extra_files"], 1)
            self.assertEqual(receipt["summary"]["verified_descriptors"], 0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.populate(root, contents)
            original = lineage.verify_one_file
            mutated = False

            def replace_with_fifo(root_fd, relative, descriptor, scanned):
                nonlocal mutated
                if not mutated:
                    target = root / relative.as_posix()
                    target.unlink()
                    os.mkfifo(target)
                    mutated = True
                return original(root_fd, relative, descriptor, scanned)

            with patch.object(lineage, "verify_one_file", side_effect=replace_with_fifo):
                receipt = self.verify(root, contents)
            self.assertIs(receipt["verification_passed"], False)
            self.assertIs(
                receipt["summary"]["tree_mutated_during_verification"], True
            )
            self.assertGreaterEqual(receipt["summary"]["rejected_entries"], 1)
            self.assertEqual(receipt["summary"]["verified_descriptors"], 0)

    def test_receipt_write_is_dirfd_bound_and_no_clobber(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "receipt.json"
            lineage.atomic_write_new(output, b"complete\n")
            self.assertEqual(output.read_bytes(), b"complete\n")
            with self.assertRaisesRegex(ValueError, "overwrite"):
                lineage.atomic_write_new(output, b"replacement\n")
            target = root / "target"
            target.mkdir()
            symlink = root / "symlink"
            os.symlink(target, symlink)
            with self.assertRaises(OSError):
                lineage.atomic_write_new(symlink / "redirected.json", b"no\n")
            self.assertFalse((target / "redirected.json").exists())

    def test_receipt_temp_collision_is_not_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "receipt.json"
            temporary = root / f".{output.name}.tmp-{os.getpid()}"
            temporary.write_bytes(b"do-not-delete\n")
            with self.assertRaises(FileExistsError):
                lineage.atomic_write_new(output, b"complete\n")
            self.assertEqual(temporary.read_bytes(), b"do-not-delete\n")
            self.assertFalse(output.exists())

    def test_ancestor_symlink_swap_cannot_redirect_held_root(self):
        contents = {"behavioral/a.json": b"alpha\n", "control/b.npz": b"beta\0"}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            checked_parent = base / "checked-parent"
            checked_root = checked_parent / "bundle"
            checked_root.mkdir(parents=True)
            attacker_root = base / "attacker-parent" / "bundle"
            attacker_root.mkdir(parents=True)
            self.populate(attacker_root, contents)
            displaced_parent = base / "checked-parent-displaced"
            real_open = lineage.open_absolute_directory_fd
            swapped = False

            def pin_then_swap(path):
                nonlocal swapped
                descriptor = real_open(path)
                if path == checked_root and not swapped:
                    checked_parent.rename(displaced_parent)
                    os.symlink(attacker_root.parent, checked_parent)
                    swapped = True
                return descriptor

            with patch.object(
                lineage, "open_absolute_directory_fd", side_effect=pin_then_swap
            ):
                receipt = self.verify(checked_root, contents)
            self.assertIs(receipt["verification_passed"], False)
            self.assertEqual(receipt["summary"]["missing_files"], 2)
            self.assertEqual(receipt["summary"]["verified_descriptors"], 0)

    def test_held_fd_barrier_catches_post_scan_nested_and_file_mutation(self):
        contents = {"nested/one.json": b"one", "nested/two.npz": b"two"}
        for mutation in ("nested_extra", "same_size_file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.populate(root, contents)
                real_scan = lineage.scan_bundle
                scan_count = 0

                def mutate_after_final_scan(*args, **kwargs):
                    nonlocal scan_count
                    result = real_scan(*args, **kwargs)
                    scan_count += 1
                    if scan_count == 2:
                        if mutation == "nested_extra":
                            (root / "nested/late-extra").write_bytes(b"late")
                        else:
                            (root / "nested/one.json").write_bytes(b"ONE")
                    return result

                with patch.object(
                    lineage, "scan_bundle", side_effect=mutate_after_final_scan
                ):
                    receipt = self.verify(root, contents)
                self.assertIs(receipt["verification_passed"], False)
                self.assertIs(
                    receipt["summary"]["tree_mutated_during_verification"], True
                )
                self.assertEqual(receipt["summary"]["verified_descriptors"], 0)
                barrier = receipt["post_hash_tree_check"]
                self.assertIs(barrier["barrier_passed"], False)
                if mutation == "nested_extra":
                    self.assertIn("nested", barrier["barrier_changed_directories"])
                else:
                    self.assertIn(
                        "nested/one.json", barrier["barrier_changed_files"]
                    )

    def test_coherent_repin_cannot_promote_fixed_boundaries(self):
        originals = {
            key: lineage.git_blob(*key) for key in lineage.EXPECTED_SOURCES
        }
        original_expected = copy.deepcopy(lineage.EXPECTED_SOURCES)
        mutations = [
            (
                (lineage.SPEC_COMMIT, lineage.SPEC_PATH),
                lambda value: value.__setitem__("core_episode_ceiling", 233),
            ),
            (
                (lineage.SPEC_COMMIT, lineage.SPEC_PATH),
                lambda value: value["models"]["D2"].__setitem__("selected", True),
            ),
            (
                (lineage.PREFLIGHT_COMMIT, lineage.PREFLIGHT_PATH),
                lambda value: value["pointer_dependencies"][0]["descriptor"].__setitem__(
                    "path", "/tmp/escape"
                ),
            ),
            (
                (lineage.JOIN_COMMIT, lineage.JOIN_PATH),
                lambda value: value["mapping"]["rows_detail"].pop(),
            ),
            (
                (lineage.JOIN_COMMIT, lineage.JOIN_PATH),
                lambda value: value["timing_boundary"].__setitem__(
                    "all_requests_timing_complete", True
                ),
            ),
            (
                (lineage.JOIN_COMMIT, lineage.JOIN_PATH),
                lambda value: value["annotation_and_release_boundary"].__setitem__(
                    "safe_to_release_confirmation", True
                ),
            ),
        ]
        for key, mutate in mutations:
            with self.subTest(key=key):
                value = json.loads(originals[key])
                mutate(value)
                payload = lineage.canonical_bytes(value)
                expected = copy.deepcopy(original_expected)
                expected[key] = (len(payload), hashlib.sha256(payload).hexdigest())

                def fake_blob(commit, path):
                    return payload if (commit, path) == key else originals[(commit, path)]

                with patch.object(lineage, "EXPECTED_SOURCES", expected), patch.object(
                    lineage, "git_blob", side_effect=fake_blob
                ):
                    with self.assertRaises((ValueError, KeyError, IndexError, TypeError)):
                        lineage.build_audit()


if __name__ == "__main__":
    unittest.main()
