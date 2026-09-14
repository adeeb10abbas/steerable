"""Adversarial tests for trusted confirmation release evidence production."""

from __future__ import annotations

import ast
from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
MODULE = (
    ROOT
    / "workshops/corl2026_world_models/experiments/forecast_layout/"
    "confirmation_release_evidence_jobs.py"
)
QUEUE_MODULE = (
    ROOT
    / "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evidence = load(MODULE, "confirmation_release_evidence_test_module")


class ConfirmationReleaseEvidenceUnitTests(unittest.TestCase):
    def test_source_has_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(MODULE.read_text(), filename=str(MODULE))
        duplicates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            seen = {}
            for key in node.keys:
                if not isinstance(key, ast.Constant):
                    continue
                identity = (type(key.value), key.value)
                if identity in seen:
                    duplicates.append((seen[identity], key.lineno, key.value))
                else:
                    seen[identity] = key.lineno
        self.assertEqual(duplicates, [])

    def test_contract_forbids_queue_results_write_science_and_kubernetes(self) -> None:
        contract = evidence.load_contract(ROOT)
        self.assertEqual(contract["schemas"]["result_attempt_ledger"], evidence.LEDGER_SCHEMA)
        self.assertFalse(contract["execution_readiness"]["queue_mutation"])
        self.assertFalse(contract["execution_readiness"]["results_branch_workstation_write"])
        self.assertTrue(all(value is False for value in contract["security"].values()))
        self.assertIn("linear descendant", contract["pending_publication"]["ancestry_rule"])
        self.assertTrue(
            contract["worker_attestation"][
                "requires_attestation_release_strictly_before_finalizer"
            ]
        )
        self.assertTrue(
            contract["worker_attestation"]["requires_exact_shared_admission_deadline"]
        )
        self.assertTrue(contract["worker_attestation"]["requires_attempt_scoped_job_ids"])
        self.assertTrue(contract["worker_attestation"]["permits_authenticated_capacity_subset"])
        self.assertTrue(contract["worker_attestation"]["fixed_deployment_inventory_not_sample_size"])
        self.assertTrue(all(contract["lock_authentication"].values()))
        self.assertTrue(all(contract["path_authentication"].values()))
        self.assertTrue(
            contract["attempt_inventory"][
                "native_prefix_requires_model_native_release_admission"
            ]
        )
        self.assertTrue(
            contract["attempt_inventory"][
                "d1_native_prefix_requires_signed_pair_admission_ack"
            ]
        )

    def test_descriptor_builders_cover_all_workers_and_select_lane_role(self) -> None:
        topology = evidence.deployment_topology(ROOT)
        selected = [
            "wmf-forecast-0912-worker-00",
            "wmf-forecast-0912-worker-09",
            "wmf-forecast-0912-worker-d1-00",
            "wmf-forecast-0912-worker-n3-00",
        ]
        fragment = evidence.build_attestation_queue_fragment(
            source_root=ROOT, study_commit="a" * 40, lane="N3",
            attempt_number=1, worker_ids=selected,
        )
        self.assertEqual(len(topology), 32)
        self.assertEqual(len(fragment["jobs"]), 4)
        self.assertEqual(
            {row["role"] for row in fragment["jobs"]},
            {topology[worker_id]["role"] for worker_id in selected},
        )
        self.assertTrue(all(row["publish_log_tail_bytes"] == 0 for row in fragment["jobs"]))
        self.assertTrue(all("attest-worker" in row["argv"] for row in fragment["jobs"]))
        self.assertTrue(all("-n3-a001-" in row["job_id"] for row in fragment["jobs"]))
        self.assertTrue(all(
            row["argv"][row["argv"].index("--expected-finalizer-job-id") + 1]
            == "confirmation-release-finalizer-n3-a001"
            for row in fragment["jobs"]
        ))
        retry = evidence.build_attestation_queue_fragment(
            source_root=ROOT, study_commit="a" * 40, lane="N3",
            attempt_number=2, worker_ids=selected,
        )
        self.assertTrue(
            {row["job_id"] for row in fragment["jobs"]}.isdisjoint(
                row["job_id"] for row in retry["jobs"]
            )
        )
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "mix finalizer attempts",
        ):
            evidence._attestation_scope([
                fragment["jobs"][0]["job_id"], retry["jobs"][1]["job_id"],
            ])
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "selected attestation worker",
        ):
            evidence.build_attestation_queue_fragment(
                source_root=ROOT, study_commit="a" * 40, lane="N3",
                attempt_number=1, worker_ids=["wmf-forecast-0912-worker-unknown"],
            )
        self.assertFalse(evidence.load_contract(ROOT)["security"]["science_or_labels_emitted"])
        for lane, role in (("N3", "n3"), ("D1", "d1")):
            finalizer = evidence.build_finalizer_queue_fragment(
                source_root=ROOT, study_commit="a" * 40, lane=lane,
                attempt_number=1, inputs_path=Path("/pvc/inputs.json"),
                inputs_sha256="b" * 64,
            )
            self.assertEqual(finalizer["jobs"][0]["role"], role)
            self.assertRegex(finalizer["jobs"][0]["job_id"], evidence.FINALIZER_RE)

    def test_zero_launch_requires_terminal_failure_and_preserves_explicit_absence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "state" / "jobs").mkdir(parents=True)
            block = SimpleNamespace(raw_root=root / "raw")
            failed_job = {
                "job_id": "confirmation-c02-n3-a001",
                "result_value": {
                    "status": "failed", "child_pid": None, "returncode": None,
                    "error_type": "OSError",
                },
                "claim_control": {
                    "job_id": "confirmation-c02-n3-a001",
                    "control_commit": "a" * 40,
                    "control_generation": 999,
                    "control_queue_blob_sha256": "b" * 64,
                    "descriptor_sha256": "c" * 64,
                },
            }
            runtime, passed, failures = evidence._build_runtime_evidence(
                model="N3", block=block,
                attempt_id="confirmation-c02-n3-a001",
                job_rows=[failed_job], state_dir=root / "state",
            )
            self.assertEqual(runtime["form"], "zero_launch_technical_failure")
            self.assertTrue(runtime["zero_behavioral_cells_launched"])
            self.assertEqual(passed, [])
            self.assertEqual(failures, [])
            self.assertTrue(all(row["state"] == "absent" for row in runtime["aggregate_slots"]))
            stray = root / "raw" / failed_job["job_id"] / "events.partial.jsonl"
            stray.parent.mkdir(parents=True)
            stray.write_text("{}\n")
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError,
                "behavioral runtime artifacts",
            ):
                evidence._build_runtime_evidence(
                    model="N3", block=block,
                    attempt_id="confirmation-c02-n3-a001",
                    job_rows=[failed_job], state_dir=root / "state",
                )
            succeeded = deepcopy(failed_job)
            succeeded["result_value"] = {"status": "succeeded"}
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "only successful"
            ):
                evidence._build_runtime_evidence(
                    model="N3", block=block,
                    attempt_id="confirmation-c02-n3-a001",
                    job_rows=[succeeded], state_dir=root / "state",
                )

    def test_zero_launch_rejects_aggregate_with_behavioral_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_id = "confirmation-c02-n3-a001"
            aggregate = root / "state" / "jobs" / job_id / "publish" / "n3_behavioral_confirmation_receipt.json"
            aggregate.parent.mkdir(parents=True)
            aggregate.write_text(json.dumps({
                "schema_version": "wmf-n3-behavioral-confirmation-job-v1",
                "status": "technical_failure",
                "counts": {
                    "launched_behavioral_cells": 1,
                    "resumed_valid_behavioral_cells": 0,
                    "newly_launched_behavioral_cells": 1,
                    "completed_valid_behavioral_cells": 0,
                    "technically_invalid_behavioral_cells": 1,
                    "right_censored_behavioral_cells": 0,
                    "actual_behavioral_actions": 0,
                    "actual_behavioral_model_requests": 0,
                },
            }))
            block = SimpleNamespace(raw_root=root / "raw")
            row = {
                "job_id": job_id,
                "result_value": {"status": "failed"},
                "claim_control": {
                    "job_id": job_id, "control_commit": "a" * 40,
                    "control_generation": 999,
                    "control_queue_blob_sha256": "b" * 64,
                    "descriptor_sha256": "c" * 64,
                },
            }
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "contains behavioral work"
            ):
                evidence._build_runtime_evidence(
                    model="N3", block=block, attempt_id=job_id,
                    job_rows=[row], state_dir=root / "state",
                )

    def test_process_and_gpu_parsers_fail_closed(self) -> None:
        with self.assertRaisesRegex(evidence.ConfirmationReleaseEvidenceError, "GPU inventory row"):
            with mock.patch.object(evidence, "_run_checked", return_value="malformed\n"):
                evidence.gpu_inventory()
        with tempfile.TemporaryDirectory() as temporary:
            proc = Path(temporary)
            (proc / "self").mkdir()
            (proc / "self" / "cgroup").write_text("0::/no-pod-identity\n")
            with self.assertRaisesRegex(evidence.ConfirmationReleaseEvidenceError, "pod UID"):
                evidence._pod_uid({}, proc)

    def test_lock_probes_are_nofollow_regular_and_never_create_missing_daemon_locks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            missing = root / "coordinator.lock"
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "pre-existing lock is missing",
            ):
                with evidence._advisory_lock(missing, expect_available=False):
                    pass
            self.assertFalse(missing.exists())
            with self.assertRaises(evidence.ConfirmationReleaseEvidenceError):
                evidence._probe_controller_locks(root, ["worker-00"])
            self.assertFalse(missing.exists())
            self.assertFalse((root / "worker-worker-00.lock").exists())

            target = root / "target.lock"
            target.write_bytes(b"unchanged")
            link = root / "linked.lock"
            link.symlink_to(target)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink|regular file",
            ):
                with evidence._advisory_lock(link, expect_available=True):
                    pass
            self.assertEqual(target.read_bytes(), b"unchanged")

            fifo = root / "fifo.lock"
            os.mkfifo(fifo)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "regular file",
            ):
                with evidence._advisory_lock(fifo, expect_available=True):
                    pass

            release = root / "release.lock"
            with evidence._canonical_release_lock(root) as receipt:
                self.assertEqual(receipt["path"], str(release))
                self.assertTrue(release.is_file())

            raced = root / "raced-release" / "release.lock"
            raced.parent.mkdir()
            real_open = evidence.os.open

            def install_race(path, flags, mode=0o777, *, dir_fd=None):
                if Path(path).name == raced.name and flags & os.O_EXCL:
                    raced.write_bytes(b"raced-in")
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with (
                mock.patch.object(evidence.os, "open", side_effect=install_race),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "open failed safely",
                ),
            ):
                with evidence._canonical_release_lock(raced.parent):
                    pass
            self.assertEqual(raced.read_bytes(), b"raced-in")

            between_lstat_and_open = root / "between-lstat-open.lock"
            between_lstat_and_open.write_bytes(b"original")
            displaced_at_open = root / "between-lstat-open.displaced"

            def replace_during_open(path, flags, mode=0o777, *, dir_fd=None):
                if Path(path).name == between_lstat_and_open.name:
                    between_lstat_and_open.rename(displaced_at_open)
                    between_lstat_and_open.write_bytes(b"replacement")
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with (
                mock.patch.object(evidence.os, "open", side_effect=replace_during_open),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "inode changed",
                ),
            ):
                with evidence._advisory_lock(
                    between_lstat_and_open, expect_available=True,
                ):
                    pass
            self.assertEqual(displaced_at_open.read_bytes(), b"original")
            self.assertEqual(between_lstat_and_open.read_bytes(), b"replacement")

            before_flock = root / "before-flock.lock"
            before_flock.write_bytes(b"")
            displaced_before = root / "before-flock.displaced"
            real_flock = evidence.fcntl.flock

            def replace_before_flock(descriptor, operation):
                if operation == evidence.fcntl.LOCK_EX | evidence.fcntl.LOCK_NB:
                    before_flock.rename(displaced_before)
                    before_flock.write_bytes(b"replacement")
                return real_flock(descriptor, operation)

            with (
                mock.patch.object(evidence.fcntl, "flock", side_effect=replace_before_flock),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "no longer names",
                ),
            ):
                with evidence._advisory_lock(before_flock, expect_available=True):
                    pass

            while_held = root / "while-held.lock"
            while_held.write_bytes(b"")
            displaced_held = root / "while-held.displaced"
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "no longer names",
            ):
                with evidence._advisory_lock(while_held, expect_available=True):
                    while_held.rename(displaced_held)
                    while_held.write_bytes(b"replacement")
            other = root / "not-release.lock"
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "only the canonical",
            ):
                with evidence._validated_lock_handle(
                    other, allow_canonical_release_creation=True, state_dir=root,
                ):
                    pass
            self.assertFalse(other.exists())

    def test_runtime_inventory_rejects_symlinked_attempt_cells_and_publish_parents(self) -> None:
        job_id = "confirmation-c02-n3-a001"
        row = {
            "job_id": job_id,
            "result_value": {
                "status": "failed", "child_pid": None, "returncode": None,
                "error_type": "OSError",
            },
            "claim_control": {
                "job_id": job_id, "control_commit": "a" * 40,
                "control_generation": 999,
                "control_queue_blob_sha256": "b" * 64,
                "descriptor_sha256": "c" * 64,
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            private = root / "private"
            private.mkdir()
            sentinel = private / "sentinel.json"
            sentinel.write_bytes(b"private-sentinel")

            attempt_case = root / "attempt-case"
            (attempt_case / "state" / "jobs").mkdir(parents=True)
            raw = attempt_case / "raw"
            raw.mkdir()
            (raw / job_id).symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink component",
            ):
                evidence._build_runtime_evidence(
                    model="N3", block=SimpleNamespace(raw_root=raw),
                    attempt_id=job_id, job_rows=[row],
                    state_dir=attempt_case / "state",
                )

            cells_case = root / "cells-case"
            (cells_case / "state" / "jobs").mkdir(parents=True)
            cells_attempt = cells_case / "raw" / job_id
            cells_attempt.mkdir(parents=True)
            (cells_attempt / "cells").symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink component",
            ):
                evidence._build_runtime_evidence(
                    model="N3", block=SimpleNamespace(raw_root=cells_case / "raw"),
                    attempt_id=job_id, job_rows=[row],
                    state_dir=cells_case / "state",
                )

            publish_case = root / "publish-case"
            queue_job = publish_case / "state" / "jobs" / job_id
            queue_job.mkdir(parents=True)
            (queue_job / "publish").symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink component",
            ):
                evidence._build_runtime_evidence(
                    model="N3", block=SimpleNamespace(raw_root=publish_case / "raw"),
                    attempt_id=job_id, job_rows=[row],
                    state_dir=publish_case / "state",
                )

            self.assertEqual(sentinel.read_bytes(), b"private-sentinel")

    def test_parent_swap_is_detected_for_read_and_anchored_immutable_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            read_parent = root / "read-parent"
            read_parent.mkdir()
            read_path = read_parent / "receipt.json"
            read_path.write_bytes(b"original")
            displaced_read = root / "read-parent.displaced"
            private_bytes = b"private-sentinel"
            real_open = evidence.os.open

            def swap_read_parent(path, flags, mode=0o777, *, dir_fd=None):
                if Path(path).name == read_path.name:
                    read_parent.rename(displaced_read)
                    read_parent.mkdir()
                    (read_parent / "receipt.json").write_bytes(private_bytes)
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with (
                mock.patch.object(evidence.os, "open", side_effect=swap_read_parent),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError,
                    "component identity changed",
                ),
            ):
                evidence.file_identity(read_path)
            self.assertEqual((displaced_read / "receipt.json").read_bytes(), b"original")
            self.assertEqual((read_parent / "receipt.json").read_bytes(), private_bytes)

            anchor = root / "write-anchor"
            output_parent = anchor / "publish"
            output_parent.mkdir(parents=True)
            displaced_write = root / "publish.displaced"
            output = output_parent / "receipt.json"

            def swap_write_parent(path, flags, mode=0o777, *, dir_fd=None):
                if Path(path).name == output.name and flags & os.O_EXCL:
                    output_parent.rename(displaced_write)
                    output_parent.mkdir()
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with (
                mock.patch.object(evidence.os, "open", side_effect=swap_write_parent),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError,
                    "component identity changed",
                ),
            ):
                evidence.immutable_write(output, {"safe": True}, anchor=anchor)
            self.assertFalse((output_parent / "receipt.json").exists())
            self.assertFalse((displaced_write / "receipt.json").exists())

    def test_regular_file_mutation_during_fd_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "receipt.json"
            original = b"x" * (1024 * 1024 + 7)
            target.write_bytes(original)
            real_read = evidence.os.read
            mutated = False

            def mutate_after_first_read(descriptor, count):
                nonlocal mutated
                chunk = real_read(descriptor, count)
                if chunk and not mutated:
                    mutated = True
                    target.write_bytes(original + b"appended")
                return chunk

            with (
                mock.patch.object(evidence.os, "read", side_effect=mutate_after_first_read),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError,
                    "changed while reading",
                ),
            ):
                evidence.file_identity(target)
            self.assertTrue(mutated)
            self.assertEqual(target.read_bytes(), original + b"appended")

    def test_strict_same_fd_json_rejects_duplicate_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            duplicate = root / "duplicate.json"
            duplicate.write_bytes(b'{"status":"first","status":"second"}\n')
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "duplicate JSON key",
            ):
                evidence.load_json_with_identity(duplicate, "duplicate receipt")

            for constant in (b"NaN", b"Infinity", b"-Infinity", b"1e999"):
                nonfinite = root / f"nonfinite-{constant.decode().replace('-', 'minus')}.json"
                nonfinite.write_bytes(b'{"value":' + constant + b"}\n")
                with self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "non-finite JSON",
                ):
                    evidence.load_json_with_identity(nonfinite, "nonfinite receipt")

    def test_hardlinked_private_evidence_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            private = root / "private"
            evidence_root = root / "raw"
            private.mkdir()
            evidence_root.mkdir()
            sentinel = private / "sentinel.json"
            sentinel.write_bytes(b'{"private":true}\n')
            alias = evidence_root / "receipt.json"
            os.link(sentinel, alias)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "singly linked",
            ):
                evidence.file_identity(alias)
            self.assertEqual(sentinel.read_bytes(), b'{"private":true}\n')
            self.assertEqual(alias.read_bytes(), b'{"private":true}\n')

    def test_out_of_root_path_is_rejected_before_candidate_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            anchor = root / "trusted"
            private = root / "private"
            anchor.mkdir()
            private.mkdir()
            sentinel = private / "sentinel.json"
            sentinel.write_bytes(b"private")
            real_lstat = Path.lstat
            real_open = evidence.os.open

            def guarded_lstat(path, *args, **kwargs):
                if Path(path) == sentinel or private in Path(path).parents:
                    raise AssertionError("private candidate was lstat'ed")
                return real_lstat(path, *args, **kwargs)

            def guarded_open(path, flags, mode=0o777, *, dir_fd=None):
                if dir_fd is None and (
                    Path(path) == sentinel or private in Path(path).parents
                ):
                    raise AssertionError("private candidate was opened")
                return real_open(path, flags, mode, dir_fd=dir_fd)

            with (
                mock.patch.object(Path, "lstat", autospec=True, side_effect=guarded_lstat),
                mock.patch.object(evidence.os, "open", side_effect=guarded_open),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "escapes its trusted root",
                ),
            ):
                evidence._confined_path(
                    sentinel, anchor=anchor, label="private candidate",
                    allow_missing=False,
                )
            self.assertEqual(sentinel.read_bytes(), b"private")

    def test_immutable_existing_output_returns_compared_identity_without_reread(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            output = root / "receipt.json"
            value = {"status": "safe"}
            expected_bytes = evidence.pretty_bytes(value)
            output.write_bytes(expected_bytes)
            replacement = evidence.pretty_bytes({"status": "replacement"})
            real_read = evidence._read_regular_file
            replaced = False

            def read_then_replace(path, label, *, capture_payload):
                nonlocal replaced
                result = real_read(path, label, capture_payload=capture_payload)
                if Path(path) == output and not replaced:
                    replaced = True
                    output.write_bytes(replacement)
                return result

            with mock.patch.object(
                evidence, "_read_regular_file", side_effect=read_then_replace,
            ) as reader:
                identity = evidence.immutable_write(output, value, anchor=root)
            self.assertTrue(replaced)
            self.assertEqual(reader.call_count, 1)
            self.assertEqual(identity, {
                "path": str(output), "bytes": len(expected_bytes),
                "sha256": evidence.sha256_bytes(expected_bytes),
            })
            self.assertEqual(output.read_bytes(), replacement)

    def test_created_release_lock_is_removed_from_displaced_parent_on_pre_yield_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "state"
            root.mkdir()
            displaced = root.parent / "state.displaced"
            real_verify = evidence._verify_directory_chain
            swapped = False

            def swap_before_verification(path, expected, label):
                nonlocal swapped
                if label == "lock parent" and not swapped:
                    swapped = True
                    root.rename(displaced)
                    root.mkdir()
                return real_verify(path, expected, label)

            with (
                mock.patch.object(
                    evidence, "_verify_directory_chain",
                    side_effect=swap_before_verification,
                ),
                self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError,
                    "component identity changed",
                ),
            ):
                with evidence._canonical_release_lock(root):
                    pass
            self.assertTrue(swapped)
            self.assertFalse((root / "release.lock").exists())
            self.assertFalse((displaced / "release.lock").exists())

    def test_native_n3_prefix_binds_persisted_release_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            finalizer = "confirmation-release-finalizer-n3-a001"
            consume_by = "2026-09-14T01:10:00Z"
            job_id = "confirmation-c02-n3-a001"
            admission = evidence.signed_document({
                "schema_version": "wmf-confirmation-runtime-release-admission-v1",
                "job_id": job_id, "worker_id": "n3-worker", "role": "n3",
                "release_finalizer_job_id": finalizer,
                "consume_by_utc": consume_by,
                "pending_publication_commit": "a" * 40,
                "published_queue_fragment_sha256": "b" * 64,
                "queue_job_ids": [job_id],
            })
            admission_path = root / "release_admission.json"
            admission_path.write_bytes(evidence.pretty_bytes(admission))
            admission_identity = evidence.file_identity(admission_path)
            aggregate = {
                "schema_version": "wmf-n3-behavioral-confirmation-job-v1",
                "status": "passed", "release_admission": admission_identity,
            }
            aggregate_path = root / "aggregate.json"
            aggregate_path.write_bytes(evidence.pretty_bytes(aggregate))
            aggregate_identity = evidence.file_identity(aggregate_path)
            descriptor = {
                "role": "n3",
                "argv": [
                    "python", "n3_confirmation_block_job.py", "queue",
                    "--confirmation-release-finalizer-job-id", finalizer,
                    "--confirmation-release-consume-by-utc", consume_by,
                ],
            }

            class NativeWave:
                @staticmethod
                def validate_runtime_admission_receipt(**kwargs):
                    identity, value = evidence.load_json_with_identity(
                        kwargs["receipt_path"], "native admission",
                    )
                    self.assertEqual(identity["sha256"], kwargs["receipt_sha256"])
                    self.assertEqual(value["job_id"], kwargs["expected_job_id"])
                    self.assertFalse(kwargs["verify_local_worker"])
                    return value

            runtime = {
                "form": "native_cell_prefix",
                "aggregate_slots": [{
                    "state": "present", "expected_path": str(aggregate_path),
                    "descriptor": aggregate_identity,
                }],
                "terminal_slots": [],
                "complete_regular_file_inventory": [
                    admission_identity, aggregate_identity,
                ],
            }
            evidence._validate_native_release_admission(
                source_root=ROOT, study_commit="c" * 40, model="N3",
                runtime_evidence=runtime,
                job_rows=[{
                    "job_id": job_id, "descriptor_value": descriptor,
                }],
                wave=NativeWave,
            )
            changed = dict(runtime)
            changed["complete_regular_file_inventory"] = [aggregate_identity]
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError,
                "absent from complete runtime inventory",
            ):
                evidence._validate_native_release_admission(
                    source_root=ROOT, study_commit="c" * 40, model="N3",
                    runtime_evidence=changed,
                    job_rows=[{
                        "job_id": job_id, "descriptor_value": descriptor,
                    }],
                    wave=NativeWave,
                )

    def test_native_d1_prefix_binds_distinct_pair_admissions_and_ack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            coordination = root / "coordination" / "run"
            coordination.mkdir(parents=True)
            finalizer = "confirmation-release-finalizer-d1-a001"
            consume_by = "2026-09-14T01:10:00Z"
            server_id = "confirmation-c02-d1-a001-server"
            simulator_id = "confirmation-c02-d1-a001-simulator"
            common = {
                "release_finalizer_job_id": finalizer,
                "consume_by_utc": consume_by,
                "pending_publication_commit": "a" * 40,
                "published_queue_fragment_sha256": "b" * 64,
                "queue_job_ids": [server_id, simulator_id],
            }
            admissions = []
            admission_identities = []
            for job_id, role, worker in (
                (server_id, "d1", "d1-server"),
                (simulator_id, "sim", "d1-simulator"),
            ):
                value = evidence.signed_document({
                    "schema_version": "wmf-confirmation-runtime-release-admission-v1",
                    "job_id": job_id, "worker_id": worker, "role": role, **common,
                })
                path = root / job_id / "release_admission.json"
                path.parent.mkdir()
                path.write_bytes(evidence.pretty_bytes(value))
                admissions.append(value)
                admission_identities.append(evidence.file_identity(path))
            server_ready_path = coordination / "server_ready.json"
            simulator_claim_path = coordination / "simulator_claim.json"
            server_ready_path.write_bytes(b'{"status":"ready"}\n')
            simulator_claim_path.write_bytes(b'{"status":"claimed"}\n')
            server_ready = evidence.file_identity(server_ready_path)
            simulator_claim = evidence.file_identity(simulator_claim_path)
            server_receipt_path = root / "server_receipt.json"
            simulator_receipt_path = root / "simulator_receipt.json"
            server_receipt_path.write_bytes(evidence.pretty_bytes({
                "schema_version": "wmf-d1-behavioral-confirmation-server-job-v1",
                "status": "passed", "release_admission": admission_identities[0],
            }))
            simulator_receipt_path.write_bytes(evidence.pretty_bytes({
                "schema_version": "wmf-d1-behavioral-confirmation-simulator-job-v1",
                "status": "passed", "release_admission": admission_identities[1],
            }))
            server_receipt = evidence.file_identity(server_receipt_path)
            simulator_receipt = evidence.file_identity(simulator_receipt_path)
            ack = evidence.signed_document({
                "schema_version": "wmf-d1-confirmation-runtime-admission-ack-v1",
                "status": "distinct_server_simulator_admitted_before_science",
                "study_id": evidence.STUDY_ID, "namespace": evidence.NAMESPACE,
                "study_commit": "c" * 40, "run_id": "run", "block_id": "block",
                "layout_pair_id": "C02", "release_finalizer_job_id": finalizer,
                "consume_by_utc": consume_by, "server_job_id": server_id,
                "simulator_job_id": simulator_id, "server_worker_id": "d1-server",
                "simulator_worker_id": "d1-simulator",
                "server_admission": admission_identities[0],
                "simulator_admission": admission_identities[1],
                "server_ready": server_ready, "simulator_claim": simulator_claim,
                "server_publication_verification_payload_sha256": "d" * 64,
                "simulator_publication_verification_payload_sha256": "e" * 64,
                "pending_publication_commit": "a" * 40,
                "server_verified_remote_head": "f" * 40,
                "simulator_verified_remote_head": "f" * 40,
                "published_queue_fragment_sha256": "b" * 64,
                "science_reset_request_action_started": False,
                "queue_mutated": False, "jobs_dispatched": 0,
                "completed_at_utc": "2026-09-14T01:00:00Z",
                "claim_boundary": "native D1 pair admitted before science",
            })
            ack_path = coordination / "release_admission_ack.json"
            ack_path.write_bytes(evidence.pretty_bytes(ack))
            ack_identity = evidence.file_identity(ack_path)

            class NativeWave:
                @staticmethod
                def validate_runtime_admission_receipt(**kwargs):
                    identity, value = evidence.load_json_with_identity(
                        kwargs["receipt_path"], "native D1 admission",
                    )
                    self.assertEqual(identity["sha256"], kwargs["receipt_sha256"])
                    self.assertEqual(value["job_id"], kwargs["expected_job_id"])
                    return value

            def descriptor(role):
                return {
                    "role": role,
                    "argv": [
                        "python", "d1_confirmation_block_jobs.py", "job",
                        "--confirmation-release-finalizer-job-id", finalizer,
                        "--confirmation-release-consume-by-utc", consume_by,
                    ],
                }

            runtime = {
                "form": "native_cell_prefix",
                "roots": [
                    {"path": str(root / server_id)},
                    {"path": str(root / simulator_id)},
                    {"path": str(coordination)},
                ],
                "aggregate_slots": [{
                    "state": "present", "expected_path": str(simulator_receipt_path),
                    "descriptor": simulator_receipt,
                }],
                "terminal_slots": [{
                    "state": "present", "expected_path": str(server_receipt_path),
                    "descriptor": server_receipt,
                }],
                "complete_regular_file_inventory": [
                    *admission_identities, server_ready, simulator_claim,
                    server_receipt, simulator_receipt, ack_identity,
                ],
            }
            evidence._validate_native_release_admission(
                source_root=ROOT, study_commit="c" * 40, model="D1",
                runtime_evidence=runtime,
                job_rows=[
                    {"job_id": server_id, "descriptor_value": descriptor("d1")},
                    {"job_id": simulator_id, "descriptor_value": descriptor("sim")},
                ],
                wave=NativeWave,
            )
            corrupted = json.loads(ack_path.read_text(encoding="utf-8"))
            corrupted["simulator_worker_id"] = "d1-server"
            unsigned = dict(corrupted)
            unsigned.pop("payload_sha256")
            corrupted = evidence.signed_document(unsigned)
            ack_path.write_bytes(evidence.pretty_bytes(corrupted))
            runtime["complete_regular_file_inventory"][-1] = evidence.file_identity(ack_path)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError,
                "ack identity or authority changed",
            ):
                evidence._validate_native_release_admission(
                    source_root=ROOT, study_commit="c" * 40, model="D1",
                    runtime_evidence=runtime,
                    job_rows=[
                        {"job_id": server_id, "descriptor_value": descriptor("d1")},
                        {"job_id": simulator_id, "descriptor_value": descriptor("sim")},
                    ],
                    wave=NativeWave,
                )

    def test_parent_symlinks_reject_queue_inputs_and_immutable_output_before_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            private = root / "private"
            private.mkdir()
            sentinel = private / "owner.json"
            sentinel.write_text("{}\n", encoding="utf-8")
            claim_parent = root / "state" / "jobs" / "job" / "claim"
            claim_parent.parent.mkdir(parents=True)
            claim_parent.symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink|no-follow",
            ):
                evidence.file_identity(claim_parent / "owner.json")
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink|no-follow",
            ):
                evidence.verify_descriptor(
                    {
                        "path": str(claim_parent / "owner.json"),
                        "bytes": sentinel.stat().st_size,
                        "sha256": evidence.sha256_bytes(sentinel.read_bytes()),
                    },
                    "symlinked input",
                )

            output_anchor = root / "output"
            output_anchor.mkdir()
            (output_anchor / "publish").symlink_to(private, target_is_directory=True)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "symlink|no-follow",
            ):
                evidence.immutable_write(
                    output_anchor / "publish" / "escaped" / "receipt.json",
                    {"safe": True}, anchor=output_anchor,
                )
            self.assertFalse((private / "escaped").exists())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "{}\n")

    def test_trusted_git_resolver_accepts_cluster_local_binary_without_path(self) -> None:
        regular = SimpleNamespace(st_uid=0, st_mode=evidence.stat.S_IFREG | 0o755)
        directory = SimpleNamespace(st_uid=0, st_mode=evidence.stat.S_IFDIR | 0o755)

        def fake_lstat(path: Path):
            if str(path) == "/usr/local/bin/git":
                return regular
            raise FileNotFoundError(str(path))

        with (
            mock.patch.object(Path, "lstat", autospec=True, side_effect=fake_lstat),
            mock.patch.object(Path, "stat", autospec=True, return_value=directory),
            mock.patch.object(os, "access", return_value=True),
            mock.patch.dict(os.environ, {"PATH": "/tmp/attacker-bin"}, clear=False),
        ):
            self.assertEqual(evidence._trusted_git_executable(), "/usr/local/bin/git")

    def test_capacity_selection_authorizes_all_fresh_n3_workers(self) -> None:
        finalizer_job = "confirmation-release-finalizer-n3-a001"
        one = [{
            "worker_id": "n3-a", "role": "n3", "gpu_count": 2,
            "idle": True, "active_job_ids": [finalizer_job],
        }]
        self.assertEqual(
            evidence._authorized_capacity_workers(
                lane="N3", workers=one, finalizer_worker="n3-a",
                finalizer_job_id=finalizer_job, d1_simulator_roles=[],
            ),
            ["n3-a"],
        )
        three = one + [
            {"worker_id": "n3-c", "role": "n3", "gpu_count": 2, "idle": True, "active_job_ids": []},
            {"worker_id": "n3-b", "role": "n3", "gpu_count": 2, "idle": True, "active_job_ids": []},
        ]
        self.assertEqual(
            evidence._authorized_capacity_workers(
                lane="N3", workers=three, finalizer_worker="n3-a",
                finalizer_job_id=finalizer_job, d1_simulator_roles=[],
            ),
            ["n3-a", "n3-b", "n3-c"],
        )

    def test_pending_reconciliation_separates_four_attestations_from_fixed_32_topology(self) -> None:
        topology = evidence.deployment_topology(ROOT)
        selected_ids = [
            "wmf-forecast-0912-worker-00",
            "wmf-forecast-0912-worker-09",
            "wmf-forecast-0912-worker-d1-00",
            "wmf-forecast-0912-worker-n3-00",
        ]
        now = evidence.utc_now()
        expiry = evidence.utc_after(now, evidence.MAX_EVIDENCE_AGE_SECONDS)
        workers = [
            {
                "worker_id": worker_id,
                "role": topology[worker_id]["role"],
                "gpu_count": topology[worker_id]["gpu_count"],
                "idle": True,
                "active_job_ids": [],
                "deployment_sha256": "a" * 64,
                "admission_deadline_utc": "2030-01-01T00:00:00Z",
                "attestation": {"path": f"/pvc/{worker_id}.json", "bytes": 1, "sha256": "b" * 64},
                "pod_uid": "1" * 32,
                "hostname": worker_id,
                "observed_at_utc": now,
                "consume_by_utc": expiry,
            }
            for worker_id in selected_ids
        ]
        finalizer_id = "confirmation-release-finalizer-n3-a001"
        finalizer = {
            "job_id": finalizer_id,
            "result": None,
            "claim_value": {
                "worker_id": "wmf-forecast-0912-worker-n3-00",
                "control_commit": "c" * 40,
                "control_generation": 1001,
            },
        }
        deadline = datetime.now(timezone.utc).timestamp() + 200_000
        control = {
            "namespace": evidence.NAMESPACE,
            "control_commit": "c" * 40,
            "control_generation": 1001,
            "admission_deadline_unix": deadline,
            "shutdown": False,
            "active_job_ids": [finalizer_id],
        }
        finalizer_attestation = {
            "worker_id": "wmf-forecast-0912-worker-n3-00",
            "hostname": "n3-pod",
            "pod_uid": "2" * 32,
            "observed_at_utc": now,
            "consume_by_utc": expiry,
            "gpu_inventory": {"idle": True, "compute_processes": []},
        }
        with mock.patch.object(evidence, "validate_pending_reconciliation") as replay:
            value = evidence.build_pending_reconciliation(
                source_root=ROOT,
                state_dir=Path("/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control"),
                study_commit="d" * 40,
                lane="N3",
                finalizer_triplet=finalizer,
                control=control,
                all_jobs={finalizer_id: finalizer},
                ledger_identity={"path": "/pvc/ledger.json", "bytes": 1, "sha256": "e" * 64},
                ledger_validation={"attempt_ids": set(), "job_ids": set()},
                workers=workers,
                finalizer_attestation_identity={"path": "/pvc/finalizer.json", "bytes": 1, "sha256": "f" * 64},
                finalizer_attestation=finalizer_attestation,
                observed_results_ancestor_commit="1" * 40,
                controller_lock_probes=[],
                d1_global_lock_probe={"path": None, "observed": "not_required"},
            )
        replay.assert_called_once()
        gpu = value["task_gpu_state"]
        self.assertEqual(gpu["fixed_deployment_worker_count"], 32)
        self.assertEqual(gpu["fixed_deployment_worker_ids"], sorted(topology))
        self.assertFalse(gpu["deployment_inventory_is_scientific_sample_size"])
        self.assertEqual(gpu["selected_attestation_worker_ids"], selected_ids)
        self.assertEqual(len(gpu["workers"]), 4)
        self.assertEqual(
            gpu["authorized_post_exit_capacity_worker_ids"],
            ["wmf-forecast-0912-worker-n3-00"],
        )

    def test_d1_capacity_requires_distinct_fresh_attested_simulator(self) -> None:
        finalizer_job = "confirmation-release-finalizer-d1-a001"
        workers = [
            {
                "worker_id": "d1-server", "role": "d1", "gpu_count": 2,
                "idle": True, "active_job_ids": [finalizer_job],
            },
            {
                "worker_id": "sim-05", "role": "wmf-forecast-0912-worker-05",
                "gpu_count": 1, "idle": True, "active_job_ids": [],
            },
            {
                "worker_id": "sim-09", "role": "wmf-forecast-0912-worker-09",
                "gpu_count": 1, "idle": True, "active_job_ids": [],
            },
            {
                "worker_id": "n3", "role": "n3", "gpu_count": 2,
                "idle": True, "active_job_ids": [],
            },
        ]
        simulator_roles = [
            "wmf-forecast-0912-worker-05", "wmf-forecast-0912-worker-09",
        ]
        self.assertEqual(
            evidence._authorized_capacity_workers(
                lane="D1", workers=workers, finalizer_worker="d1-server",
                finalizer_job_id=finalizer_job,
                d1_simulator_roles=simulator_roles,
            ),
            ["d1-server", "sim-05"],
        )
        for mutation in (
            {"active_job_ids": ["still-live"]},
            {"idle": False},
            {"gpu_count": 2},
        ):
            bad = deepcopy([workers[0], workers[1], workers[3]])
            bad[1].update(mutation)
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError,
                "simulator-worker attestation",
            ):
                evidence._authorized_capacity_workers(
                    lane="D1", workers=bad, finalizer_worker="d1-server",
                    finalizer_job_id=finalizer_job,
                    d1_simulator_roles=simulator_roles,
                )
        same_worker = [deepcopy(workers[0])]
        same_worker[0].update({
            "role": "wmf-forecast-0912-worker-05", "gpu_count": 1,
            "active_job_ids": [],
        })
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError,
            "D1 server finalizer capacity",
        ):
            evidence._authorized_capacity_workers(
                lane="D1", workers=same_worker, finalizer_worker="d1-server",
                finalizer_job_id=finalizer_job,
                d1_simulator_roles=simulator_roles,
            )

    def test_generation_998_claim_is_rejected(self) -> None:
        queue = load(QUEUE_MODULE, "confirmation_release_generation_queue")
        raw = evidence.build_finalizer_queue_fragment(
            source_root=ROOT, study_commit="a" * 40, lane="N3", attempt_number=1,
            inputs_path=Path("/data/users/ali/release-inputs.json"),
            inputs_sha256="b" * 64,
        )["jobs"][0]
        descriptor = queue.normalize_job(raw)
        claim = {
            "worker_id": "wmf-forecast-0912-worker-n3-00",
            "claimed_at": "2026-09-13T00:00:00Z",
            "claimed_unix": 1789257600.0,
            "worker_pid": 10,
            "control_commit": "c" * 40,
            "control_generation": 998,
            "descriptor_sha256": evidence.sha256_bytes(queue.encode(descriptor)),
            "release_boundary": "claim_committed_under_shared_release_lock",
        }
        with self.assertRaisesRegex(evidence.ConfirmationReleaseEvidenceError, "release boundary"):
            evidence.validate_queue_claim(
                claim, descriptor=descriptor, result=None, queue=queue,
            )

    def test_attestation_wave_must_strictly_precede_finalizer_and_share_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
            subprocess.run(["git", "-C", str(repository), "config", "user.name", "Evidence test"], check=True)
            subprocess.run(["git", "-C", str(repository), "config", "user.email", "evidence@example.invalid"], check=True)
            marker = repository / "marker"
            marker.write_text("attestation\n")
            subprocess.run(["git", "-C", str(repository), "add", "marker"], check=True)
            subprocess.run(["git", "-C", str(repository), "commit", "--quiet", "-m", "attestation"], check=True)
            attestation_commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
            ).strip()
            marker.write_text("finalizer\n")
            subprocess.run(["git", "-C", str(repository), "commit", "--quiet", "-am", "finalizer"], check=True)
            finalizer_commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
            ).strip()
            attestation_ids = (
                "confirmation-release-worker-attest-n3-a001-00",
                "confirmation-release-worker-attest-n3-a001-n3-00",
            )
            expected = {
                "wmf-forecast-0912-worker-00": {"job_id": attestation_ids[0]},
                "wmf-forecast-0912-worker-n3-00": {"job_id": attestation_ids[1]},
            }
            all_jobs = {
                job_id: {
                    "descriptor_value": dict(expected[worker_id]),
                    "claim_value": {
                        "control_generation": 1000,
                        "control_commit": attestation_commit,
                    }
                }
                for job_id, worker_id in zip(
                    attestation_ids, expected, strict=True,
                )
            }
            finalizer = {
                "job_id": "confirmation-release-finalizer-n3-a001",
                "claim_value": {
                    "control_generation": 1001,
                    "control_commit": finalizer_commit,
                }
            }
            with (
                mock.patch.object(evidence, "_expected_attestation_jobs", return_value=expected),
                mock.patch.object(
                    evidence, "load_modules",
                    return_value={"queue": SimpleNamespace(normalize_job=lambda value: value)},
                ),
                mock.patch.object(
                    evidence, "_fetch_authorized_control_history",
                    return_value=__import__("contextlib").nullcontext({"repo": str(repository)}),
                ),
            ):
                selected = evidence._require_attestation_wave_precedes_finalizer(
                    source_root=ROOT, study_commit="a" * 40,
                    all_jobs=all_jobs, finalizer_triplet=finalizer,
                )
                self.assertEqual(selected, sorted(attestation_ids))
                tampered_descriptor = deepcopy(all_jobs)
                tampered_descriptor[attestation_ids[0]]["descriptor_value"]["job_id"] = "tampered"
                with self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "descriptor set changed",
                ):
                    evidence._require_attestation_wave_precedes_finalizer(
                        source_root=ROOT, study_commit="a" * 40,
                        all_jobs=tampered_descriptor, finalizer_triplet=finalizer,
                    )
                equal_generation = deepcopy(all_jobs)
                equal_generation[attestation_ids[0]]["claim_value"]["control_generation"] = 1001
                with self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "does not precede",
                ):
                    evidence._require_attestation_wave_precedes_finalizer(
                        source_root=ROOT, study_commit="a" * 40,
                        all_jobs=equal_generation, finalizer_triplet=finalizer,
                    )
                same_commit = deepcopy(all_jobs)
                same_commit[attestation_ids[0]]["claim_value"]["control_commit"] = finalizer_commit
                with self.assertRaisesRegex(
                    evidence.ConfirmationReleaseEvidenceError, "not before",
                ):
                    evidence._require_attestation_wave_precedes_finalizer(
                        source_root=ROOT, study_commit="a" * 40,
                        all_jobs=same_commit, finalizer_triplet=finalizer,
                    )

            deadline = 2_000_000_000
            deadline_utc = datetime.fromtimestamp(deadline, timezone.utc).isoformat().replace("+00:00", "Z")
            workers = [{"admission_deadline_utc": deadline_utc} for _ in range(32)]
            self.assertEqual(
                evidence._require_worker_shared_admission_deadline(
                    workers=workers,
                    semantic_control={"admission_deadline_unix": deadline},
                ),
                deadline_utc,
            )
            workers[-1]["admission_deadline_utc"] = "2030-01-01T00:00:00Z"
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "shared admission deadline",
            ):
                evidence._require_worker_shared_admission_deadline(
                    workers=workers,
                    semantic_control={"admission_deadline_unix": deadline},
                )


class PublishedFinalizerReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.remote = self.root / "remote.git"
        trusted_fetch_patch = mock.patch.object(
            evidence, "TRUSTED_FETCH_URL", str(self.remote)
        )
        trusted_fetch_patch.start()
        self.addCleanup(trusted_fetch_patch.stop)
        self.state = self.root / "state"
        self.queue = load(QUEUE_MODULE, f"confirmation_release_queue_{id(self)}")
        subprocess.run(["git", "init", "--bare", "--quiet", str(self.remote)], check=True)
        self.git("init", "--quiet")
        self.git("config", "user.name", "Evidence test")
        self.git("config", "user.email", "evidence@example.invalid")
        contract_path = (
            self.source
            / "workshops/corl2026_world_models/experiments/forecast_layout/"
            "confirmation_release_evidence_contract.json"
        )
        contract_path.parent.mkdir(parents=True)
        contract = json.loads(
            (
                ROOT
                / "workshops/corl2026_world_models/experiments/forecast_layout/"
                "confirmation_release_evidence_contract.json"
            ).read_text()
        )
        contract["queue"]["allowed_results_urls"] = [str(self.remote)]
        contract["queue"]["trusted_fetch_url"] = str(self.remote)
        contract["queue"]["state_dir"] = str(self.state.resolve())
        contract["queue"]["control_ref"] = "refs/heads/control"
        contract["queue"]["control_branch"] = "control"
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
        for relative in [
            *contract["source_paths"].values(),
            *contract["deployment_manifests"],
        ]:
            target = self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, target)
        (self.source / "source.txt").write_text("immutable source\n")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "source")
        self.commit = self.git("rev-parse", "HEAD").strip()
        self.inputs = self.state / "release-inputs.json"
        self.inputs.parent.mkdir(parents=True, exist_ok=True)
        self.inputs.write_text("{}\n")
        inputs_sha = evidence.sha256_file(self.inputs)
        self.job_id = "confirmation-release-finalizer-n3-a001"
        self.raw_job = evidence.build_finalizer_queue_fragment(
            source_root=self.source,
            study_commit=self.commit,
            lane="N3",
            attempt_number=1,
            inputs_path=self.inputs,
            inputs_sha256=inputs_sha,
        )["jobs"][0]
        self.queue_manifest = {
            "schema_version": "wmf-cluster-queue-v1",
            "namespace": evidence.NAMESPACE,
            "shutdown": False,
            "jobs": [self.raw_job],
        }
        queue_path = self.source / contract["queue"]["queue_path"]
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path.write_text(json.dumps(self.queue_manifest, indent=2, sort_keys=True) + "\n")
        self.git("add", str(queue_path.relative_to(self.source)))
        self.git("commit", "--quiet", "-m", "release finalizer")
        self.control_commit = self.git("rev-parse", "HEAD").strip()
        self.control_ref = "refs/remotes/origin/control"
        self.results_branch = "codex/forecast-layout-gm-20260912-results"
        self.results_ref = "refs/heads/" + self.results_branch
        self.git("remote", "add", "origin", str(self.remote))
        self.git("push", "--quiet", "origin", "HEAD:refs/heads/control")
        self.git("update-ref", self.control_ref, self.control_commit)
        empty = {
            "schema_version": self.queue.QUEUE_SCHEMA if hasattr(self.queue, "QUEUE_SCHEMA") else "wmf-cluster-queue-v1",
            "namespace": evidence.NAMESPACE,
            "shutdown": False,
            "jobs": [],
        }
        self.queue.stage_queue(
            self.source, self.state, self.control_ref, empty,
            admission_deadline_unix=datetime.now(timezone.utc).timestamp() + 300000,
        )
        self.queue.publish_results(
            self.source, self.state, self.control_ref, self.results_branch
        )
        self.h0 = self.remote_head()
        control = self.queue.stage_queue(
            self.source, self.state, self.control_ref, self.queue_manifest,
            admission_deadline_unix=datetime.now(timezone.utc).timestamp() + 300000,
        )
        control["control_generation"] = 999
        self.queue.atomic_json(self.state / "control.json", control)
        self.assertTrue(self.queue.claim_job(self.state / "jobs" / self.job_id, "wmf-forecast-0912-worker-n3-00"))
        # Real in-flight/pre-result coordinator publication after H0.
        self.queue.publish_results(
            self.source, self.state, self.control_ref, self.results_branch
        )
        self.h_pre = self.remote_head()
        self.assertNotEqual(self.h_pre, self.h0)
        self.created_at = evidence.utc_now()
        self._install_pending_artifacts()
        self._install_outer_result()
        self.queue.publish_results(
            self.source, self.state, self.control_ref, self.results_branch
        )
        self.h_publication = self.remote_head()
        # A real repeated stage/publish poll increments generation with exactly
        # the same semantic control and creates a later status-only commit.
        later = self.queue.stage_queue(
            self.source, self.state, self.control_ref, self.queue_manifest,
            admission_deadline_unix=control["admission_deadline_unix"],
        )
        self.assertGreater(later["control_generation"], 999)
        self.queue.publish_results(
            self.source, self.state, self.control_ref, self.results_branch
        )
        self.h_later = self.remote_head()
        self.assertNotEqual(self.h_later, self.h_publication)

    def git(self, *args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(self.source), *args], text=True,
            stderr=subprocess.PIPE,
        )

    def remote_head(self) -> str:
        return subprocess.check_output(
            ["git", "--git-dir", str(self.remote), "rev-parse", self.results_ref],
            text=True,
        ).strip()

    def _write_json(self, path: Path, value: dict) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(evidence.pretty_bytes(value))
        return evidence.file_identity(path)

    def _install_pending_artifacts(self) -> None:
        job_root = self.state / "jobs" / self.job_id
        publish = job_root / "publish"
        evidence_root = publish / "confirmation_release_evidence"
        descriptor = json.loads((job_root / "descriptor.json").read_text())
        claim = json.loads((job_root / "claim" / "owner.json").read_text())
        queue_blob = subprocess.check_output([
            "git", "-C", str(self.source), "show",
            f"{self.control_commit}:workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json",
        ])
        triplet = {
            "job_id": self.job_id,
            "descriptor": evidence.file_identity(job_root / "descriptor.json"),
            "descriptor_value": descriptor,
            "claim": evidence.file_identity(job_root / "claim" / "owner.json"),
            "claim_value": claim,
            "claim_control": {
                "job_id": self.job_id,
                "control_commit": claim["control_commit"],
                "control_generation": claim["control_generation"],
                "control_queue_blob_sha256": evidence.sha256_bytes(queue_blob),
                "descriptor_sha256": claim["descriptor_sha256"],
            },
        }
        control = json.loads((self.state / "control.json").read_text())
        topology = evidence.deployment_topology(self.source)
        worker_id = "wmf-forecast-0912-worker-n3-00"
        record = topology[worker_id]["records"][0]
        staged_source = self.state / "sources" / self.commit
        staged_manifest = evidence.file_identity(
            staged_source / record["manifest_relative_path"]
        )
        current_pid = os.getpid()
        controller_pid = 111
        controller_argv_sha = "d" * 64
        worker_attestation_value = evidence.signed_document({
            "schema_version": evidence.WORKER_ATTESTATION_SCHEMA,
            "status": "authenticated_worker_gpu_idle_while_attesting",
            "study_id": evidence.STUDY_ID,
            "namespace": evidence.NAMESPACE,
            "study_commit": self.commit,
            "observed_at_utc": self.created_at,
            "consume_by_utc": evidence.utc_after(
                self.created_at, evidence.MAX_EVIDENCE_AGE_SECONDS
            ),
            "worker_id": worker_id,
            "role": "n3",
            "hostname": "n3-test-pod",
            "pod_uid": "1" * 32,
            "pod_uid_source": "proc_self_cgroup",
            "queue_job_id": self.job_id,
            "queue_descriptor": triplet["descriptor"],
            "queue_claim": triplet["claim"],
            "claim_control": triplet["claim_control"],
            "semantic_control": {
                "namespace": evidence.NAMESPACE,
                "control_commit": claim["control_commit"],
                "observed_generation": 999,
                "claim_generation": claim["control_generation"],
                "admission_deadline_unix": control["admission_deadline_unix"],
                "shutdown": False,
                "active_job_ids": [self.job_id],
            },
            "deployment": {
                "manifest": staged_manifest,
                "manifest_relative_path": record["manifest_relative_path"],
                "item_index": record["item_index"],
                "job_name": record["job_name"],
                "container_name": record["container_name"],
                "image": record["image"],
                "gpu_count": record["gpu_count"],
                "controller_argv_sha256": controller_argv_sha,
                "controller_pid": controller_pid,
            },
            "process_inventory": {
                "inventory_complete": True,
                "visible": [
                    {"pid": 1, "ppid": 0, "argv_sha256": "a" * 64},
                    {"pid": controller_pid, "ppid": 1, "argv_sha256": controller_argv_sha},
                    {"pid": current_pid, "ppid": controller_pid, "argv_sha256": "e" * 64},
                ],
                "allowed_ancestry_pids": [1, controller_pid, current_pid],
                "non_ancestry_processes": [],
                "current_pid": current_pid,
                "current_process_is_attestation_only": True,
            },
            "gpu_inventory": {
                "inventory_complete": True,
                "gpus": [
                    {
                        "index": index, "uuid": f"GPU-{index}", "name": "B200",
                        "memory_total_mib": 183359, "memory_used_mib": 0,
                        "utilization_percent": 0,
                    }
                    for index in range(2)
                ],
                "compute_processes": [],
                "idle": True,
            },
            "worker_active_job_ids": [self.job_id],
            "claim_boundary": "This worker attests only its own test pod.",
        })
        worker_attestation = self._write_json(
            publish / "worker_attestation.json", worker_attestation_value
        )
        ledger_value = evidence.signed_document({
            "schema_version": evidence.LEDGER_SCHEMA,
            "status": "authenticated_native_attempt_inventory",
            "study_id": evidence.STUDY_ID,
            "namespace": evidence.NAMESPACE,
            "study_commit": self.commit,
            "attempt_inventory_complete": True,
            "blocks": [],
        })
        ledger = self._write_json(evidence_root / "result_attempt_ledger.json", ledger_value)
        reconciliation_value = evidence.signed_document({
            "schema_version": evidence.RECONCILIATION_SCHEMA,
            "status": "pending_finalizer_publication",
            "study_id": evidence.STUDY_ID,
            "namespace": evidence.NAMESPACE,
            "study_commit": self.commit,
            "results_publication": {
                "observed_results_ancestor_commit": self.h0,
                "ledger": ledger,
            },
            "prior_inventory": {"all_prior_job_ids": [self.job_id]},
            "task_gpu_state": {
                "workers": [{
                    "worker_id": worker_id,
                    "role": "n3",
                    "gpu_count": 2,
                    "idle": True,
                    "active_job_ids": [self.job_id],
                    "deployment_sha256": staged_manifest["sha256"],
                    "admission_deadline_utc": datetime.fromtimestamp(
                        control["admission_deadline_unix"], timezone.utc
                    ).isoformat().replace("+00:00", "Z"),
                    "attestation": worker_attestation,
                    "pod_uid": "1" * 32,
                    "hostname": "n3-test-pod",
                    "observed_at_utc": self.created_at,
                    "consume_by_utc": evidence.utc_after(
                        self.created_at, evidence.MAX_EVIDENCE_AGE_SECONDS
                    ),
                }],
            },
            "created_at_utc": self.created_at,
            "consume_by_utc": evidence.utc_after(
                self.created_at, evidence.MAX_EVIDENCE_AGE_SECONDS
            ),
        })
        reconciliation = self._write_json(
            evidence_root / "cluster_reconciliation.json", reconciliation_value
        )
        plan = self._write_json(evidence_root / "confirmation_release_plan.json", {"status": "pending_h1_verification"})
        queue_fragment = self._write_json(
            evidence_root / "confirmation_release_queue_fragment.json",
            {"schema_version": "wmf-cluster-queue-v1", "namespace": evidence.NAMESPACE, "shutdown": False, "jobs": []},
        )
        wave_receipt = self._write_json(
            evidence_root / "confirmation_release_wave_receipt.json",
            {"status": "pending_h1_verification"},
        )
        pending = evidence.build_publication_pending(
            study_commit=self.commit,
            remote_alias="origin", observed_results_ancestor_commit=self.h0,
            finalizer_triplet=triplet,
            finalizer_attestation={
                "hostname": "n3-test-pod", "pod_uid": "1" * 32,
            },
            reconciliation={
                "created_at_utc": self.created_at,
                "observed_control_generation_lower_bound": claim["control_generation"],
                "observed_control_generation": 999,
                "control_semantics": {
                    "namespace": evidence.NAMESPACE,
                    "control_commit": claim["control_commit"],
                    "admission_deadline_unix": json.loads((self.state / "control.json").read_text())["admission_deadline_unix"],
                    "shutdown": False,
                    "active_job_ids": [self.job_id],
                },
                "consume_by_utc": evidence.utc_after(
                    self.created_at, evidence.MAX_EVIDENCE_AGE_SECONDS
                ),
            },
            artifacts={
                "ledger": ledger,
                "reconciliation": reconciliation,
                "plan": plan,
                "queue_fragment": queue_fragment,
                "wave_receipt": wave_receipt,
            },
        )
        pending_identity = self._write_json(
            evidence_root / "publication_pending.json", pending
        )
        receipt = evidence.signed_document({
            "schema_version": evidence.EVIDENCE_RECEIPT_SCHEMA,
            "status": "pending_coordinator_publication_no_science",
            "study_id": evidence.STUDY_ID,
            "namespace": evidence.NAMESPACE,
            "study_commit": self.commit,
            "created_at_utc": self.created_at,
            "consume_by_utc": evidence.utc_after(
                self.created_at, evidence.MAX_EVIDENCE_AGE_SECONDS
            ),
            "inputs": evidence.file_identity(self.inputs),
            "artifacts": {
                "ledger": ledger,
                "reconciliation": reconciliation,
                "plan": plan,
                "queue_fragment": queue_fragment,
                "wave_receipt": wave_receipt,
                "publication_pending": pending_identity,
            },
            "outer_result_available": False,
            "cluster_status_available": False,
            "publication_commit_available": False,
            "queue_mutated": False,
            "results_branch_written_by_finalizer": False,
            "science_or_labels_emitted": False,
            "claim_boundary": "Synthetic pending evidence; publication remains external.",
        })
        self._write_json(evidence_root / "release_evidence_receipt.json", receipt)

    def _install_outer_result(self) -> None:
        job_root = self.state / "jobs" / self.job_id
        descriptor = json.loads((job_root / "descriptor.json").read_text())
        for stream in ("stdout.log", "stderr.log"):
            (job_root / stream).write_bytes(b"")
        timestamp = evidence.utc_now()
        result = {
            "schema_version": evidence.RESULT_SCHEMA,
            "namespace": evidence.NAMESPACE,
            "job_id": self.job_id,
            "worker_id": "wmf-forecast-0912-worker-n3-00",
            "source_commit": self.commit,
            "descriptor_sha256": evidence.sha256_bytes(self.queue.encode(descriptor)),
            "started_at": json.loads((job_root / "claim" / "owner.json").read_text())["claimed_at"],
            "argv": evidence._expanded_argv(
                descriptor,
                source_root=self.state / "sources" / self.commit,
                job_dir=job_root,
                state_dir=self.state,
            ),
            "job_dir": str(job_root.resolve()),
            "status": "succeeded",
            "returncode": 0,
            "error_type": None,
            "ended_at": timestamp,
            "wall_seconds": 0.1,
            "child_pid": 43210,
            "child_reaped": True,
            "stdout": self.queue.file_identity(job_root / "stdout.log"),
            "stderr": self.queue.file_identity(job_root / "stderr.log"),
        }
        self.queue.atomic_json(job_root / "result.json", result, immutable=True)

    def _rewrite_publication_artifact(self, relative: str, mutate) -> None:
        clone = self.root / ("rewrite-" + str(len(list(self.root.glob("rewrite-*")))))
        subprocess.run(
            [
                "git", "clone", "--quiet", "--branch", self.results_branch,
                str(self.remote), str(clone),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", self.h_publication], check=True,
        )
        target = clone / "results" / "jobs" / self.job_id / "publish" / relative
        mutate(target)
        manifest_path = clone / "results" / "jobs" / self.job_id / "publish_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        rows = {row["path"]: row for row in manifest["files"]}
        if target.exists():
            data = target.read_bytes()
            rows[relative] = {
                "path": relative, "bytes": len(data),
                "sha256": evidence.sha256_bytes(data),
            }
        else:
            rows.pop(relative, None)
        manifest["files"] = [rows[name] for name in sorted(rows)]
        manifest["copied_bytes"] = sum(row["bytes"] for row in manifest["files"])
        manifest_path.write_bytes(evidence.pretty_bytes(manifest))
        subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
        environment = dict(
            os.environ,
            GIT_AUTHOR_NAME="WMF cluster coordinator",
            GIT_AUTHOR_EMAIL="wmf-cluster@users.noreply.github.com",
            GIT_COMMITTER_NAME="WMF cluster coordinator",
            GIT_COMMITTER_EMAIL="wmf-cluster@users.noreply.github.com",
        )
        subprocess.run(
            ["git", "-C", str(clone), "commit", "--quiet", "--amend", "--no-edit"],
            check=True, env=environment,
        )
        replacement = subprocess.check_output(
            ["git", "-C", str(clone), "rev-parse", "HEAD"], text=True,
        ).strip()
        subprocess.run(
            ["git", "-C", str(clone), "push", "--quiet", "origin", "HEAD:refs/heads/rewrite-candidate"],
            check=True,
        )
        subprocess.run(
            ["git", "--git-dir", str(self.remote), "update-ref", self.results_ref, replacement],
            check=True,
        )

    def test_actual_coordinator_replay_accepts_inflight_and_generation_drift(self) -> None:
        before = self.git("status", "--porcelain=v1", "--untracked-files=all")
        verified = evidence.verify_published_finalizer(
            source_root=self.source, finalizer_job_id=self.job_id,
            study_commit=self.commit,
        )
        self.assertEqual(verified["publication_commit"], self.h_publication)
        self.assertEqual(verified["verified_remote_head"], self.h_later)
        self.assertEqual(verified["artifacts"]["publication_pending"]["producer_queue_job"]["job_id"], self.job_id)
        self.assertTrue(verified["document"]["queue_fragment_release_gate_passed"])
        self.assertEqual(self.remote_head(), self.h_later)
        self.assertEqual(self.git("status", "--porcelain=v1", "--untracked-files=all"), before)

    def test_verifier_uses_internal_clock_and_rejects_expired_capacity(self) -> None:
        expired_now = evidence.utc_after(
            self.created_at, evidence.MAX_EVIDENCE_AGE_SECONDS + 1
        )
        with (
            mock.patch.object(evidence, "utc_now", return_value=expired_now),
            self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError,
                "stale or future-dated",
            ),
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )
        self.assertNotIn("as_of_utc", inspect.signature(evidence.verify_published_finalizer).parameters)
        self.assertNotIn("as_of", inspect.signature(evidence.validate_worker_attestation).parameters)
        self.assertNotIn(
            "created_at_utc",
            inspect.signature(evidence.build_pending_reconciliation).parameters,
        )
        self.assertNotIn(
            "created_at_utc",
            inspect.signature(evidence.build_publication_pending).parameters,
        )

    def test_git_config_helper_and_ssh_injection_cannot_redirect_fetch(self) -> None:
        injected = {
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "url.file:///tmp/attacker.insteadOf",
            "GIT_CONFIG_VALUE_0": str(self.remote),
            "GIT_CONFIG_KEY_1": "credential.helper",
            "GIT_CONFIG_VALUE_1": "!/tmp/attacker-helper",
            "GIT_CONFIG_GLOBAL": "/tmp/attacker-global-config",
            "GIT_CONFIG_SYSTEM": "/tmp/attacker-system-config",
            "GIT_EXEC_PATH": "/tmp/attacker-git-exec",
            "GIT_SSH_COMMAND": "/tmp/attacker-ssh",
            "SSH_ASKPASS": "/tmp/attacker-askpass",
            "LD_PRELOAD": "/tmp/attacker-preload.so",
            "LD_LIBRARY_PATH": "/tmp/attacker-libraries",
            "PYTHONPATH": "/tmp/attacker-python",
            "PATH": "/tmp/attacker-bin",
            "HOME": "/tmp/attacker-home",
            "NETRC": "/tmp/attacker-netrc",
            "SSL_CERT_FILE": "/tmp/attacker-ca.pem",
            "CURL_CA_BUNDLE": "/tmp/attacker-ca.pem",
            "HTTPS_PROXY": "https://attacker.invalid:443",
            "ALL_PROXY": "socks5://attacker.invalid:1080",
        }
        safe_environment = evidence._safe_git_environment()
        for name in injected:
            if name in safe_environment:
                self.assertNotEqual(safe_environment.get(name), injected[name])
            else:
                self.assertNotIn(name, safe_environment)
        with mock.patch.dict(os.environ, injected, clear=False):
            verified = evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )
        self.assertEqual(verified["publication_commit"], self.h_publication)

    def test_unrelated_descendant_commit_is_rejected(self) -> None:
        tree = self.state / "results-git"
        (tree / "unrelated.txt").write_text("not coordinator evidence\n")
        subprocess.run(["git", "-C", str(tree), "add", "unrelated.txt"], check=True)
        subprocess.run(
            ["git", "-C", str(tree), "-c", "user.name=Other", "-c", "user.email=other@example.invalid", "commit", "--quiet", "-m", "unrelated"],
            check=True,
        )
        subprocess.run(["git", "-C", str(tree), "push", "--quiet", "origin", "HEAD:" + self.results_ref], check=True)
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "noncoordinator"
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )

    def test_coordinator_authored_nonstatus_descendant_is_rejected(self) -> None:
        tree = self.state / "results-git"
        rogue = tree / "results" / "coordinator-but-not-status.json"
        rogue.parent.mkdir(parents=True, exist_ok=True)
        rogue.write_text("{}\n")
        subprocess.run(
            ["git", "-C", str(tree), "add", str(rogue.relative_to(tree))],
            check=True,
        )
        environment = dict(
            os.environ,
            GIT_AUTHOR_NAME="WMF cluster coordinator",
            GIT_AUTHOR_EMAIL="wmf-cluster@users.noreply.github.com",
            GIT_COMMITTER_NAME="WMF cluster coordinator",
            GIT_COMMITTER_EMAIL="wmf-cluster@users.noreply.github.com",
        )
        subprocess.run(
            [
                "git", "-C", str(tree), "commit", "--quiet", "-m",
                "Record cluster queue receipts",
            ],
            check=True,
            env=environment,
        )
        subprocess.run(
            ["git", "-C", str(tree), "push", "--quiet", "origin", "HEAD:" + self.results_ref],
            check=True,
        )
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "status-only snapshot"
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )

    def test_published_attestation_is_fetched_and_deep_validated(self) -> None:
        def corrupt(path: Path) -> None:
            value = json.loads(path.read_text())
            value.pop("payload_sha256")
            value["gpu_inventory"]["compute_processes"] = [{"pid": 9}]
            path.write_bytes(evidence.pretty_bytes(evidence.signed_document(value)))

        self._rewrite_publication_artifact("worker_attestation.json", corrupt)
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError,
            "attestation|GPU|reconciliation",
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )

    def test_published_evidence_receipt_is_fetched_and_deep_validated(self) -> None:
        relative = "confirmation_release_evidence/release_evidence_receipt.json"

        def corrupt(path: Path) -> None:
            value = json.loads(path.read_text())
            value.pop("payload_sha256")
            value["science_or_labels_emitted"] = True
            path.write_bytes(evidence.pretty_bytes(evidence.signed_document(value)))

        self._rewrite_publication_artifact(relative, corrupt)
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "receipt identity|boundary"
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )

    def test_manifest_cannot_stand_in_for_missing_attestation(self) -> None:
        self._rewrite_publication_artifact(
            "worker_attestation.json", lambda path: path.unlink()
        )
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "manifest omitted|blob is missing"
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )

    def test_local_orphan_control_commit_cannot_authenticate_claim(self) -> None:
        tree = self.git("rev-parse", f"{self.control_commit}^{{tree}}").strip()
        environment = dict(
            os.environ,
            GIT_AUTHOR_NAME="Local attacker", GIT_AUTHOR_EMAIL="local@example.invalid",
            GIT_COMMITTER_NAME="Local attacker", GIT_COMMITTER_EMAIL="local@example.invalid",
        )
        orphan = subprocess.check_output(
            ["git", "-C", str(self.source), "commit-tree", tree, "-m", "local orphan"],
            text=True, env=environment,
        ).strip()
        job_root = self.state / "jobs" / self.job_id
        descriptor = json.loads((job_root / "descriptor.json").read_text())
        claim = json.loads((job_root / "claim" / "owner.json").read_text())
        claim["control_commit"] = orphan
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "control history|Git merge-base"
        ):
            evidence._queue_manifest_at_claim(
                source_root=self.source, claim=claim, descriptor=descriptor,
                queue_path=evidence.load_contract(self.source)["queue"]["queue_path"],
                queue=self.queue,
            )

    def test_unrelated_descriptor_source_commit_cannot_authenticate_claim(self) -> None:
        source_tree = self.git("rev-parse", f"{self.commit}^{{tree}}").strip()
        environment = dict(
            os.environ,
            GIT_AUTHOR_NAME="Local attacker", GIT_AUTHOR_EMAIL="local@example.invalid",
            GIT_COMMITTER_NAME="Local attacker", GIT_COMMITTER_EMAIL="local@example.invalid",
        )
        unrelated_source = subprocess.check_output(
            ["git", "-C", str(self.source), "commit-tree", source_tree, "-m", "unrelated source"],
            text=True, env=environment,
        ).strip()
        queue_path = Path(evidence.load_contract(self.source)["queue"]["queue_path"])
        queue_file = self.source / queue_path
        manifest = json.loads(queue_file.read_text())
        manifest["jobs"][0]["source_commit"] = unrelated_source
        queue_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        self.git("add", str(queue_path))
        self.git("commit", "--quiet", "-m", "coordinated unrelated source")
        claim_commit = self.git("rev-parse", "HEAD").strip()
        self.git("push", "--quiet", "origin", "HEAD:refs/heads/control")
        descriptor = json.loads(
            (self.state / "jobs" / self.job_id / "descriptor.json").read_text()
        )
        descriptor["source_commit"] = unrelated_source
        claim = json.loads(
            (self.state / "jobs" / self.job_id / "claim" / "owner.json").read_text()
        )
        claim["control_commit"] = claim_commit
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError,
            "source commit is outside.*claim-control history",
        ):
            evidence._queue_manifest_at_claim(
                source_root=self.source, claim=claim, descriptor=descriptor,
                queue_path=str(queue_path), queue=self.queue,
            )

    def test_staged_source_untracked_residue_is_rejected(self) -> None:
        staged = self.state / "sources" / self.commit
        residue = staged / "untracked-residue.txt"
        residue.write_text("must fail closed\n")
        try:
            with self.assertRaisesRegex(
                evidence.ConfirmationReleaseEvidenceError, "untracked residue"
            ):
                evidence.validate_source_context(staged, self.commit)
        finally:
            residue.unlink()

    def test_workstation_deployment_manifest_mutation_is_rejected(self) -> None:
        relative = evidence.load_contract(self.source)["deployment_manifests"][0]
        target = self.source / relative
        target.write_text(target.read_text() + "\n")
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError,
            "study commit source differs",
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )

    def test_later_semantic_control_change_is_rejected(self) -> None:
        control = json.loads((self.state / "control.json").read_text())
        control["control_commit"] = "f" * 40
        control["control_generation"] += 1
        self.queue.atomic_json(self.state / "control.json", control)
        self.queue.publish_results(self.source, self.state, self.control_ref, self.results_branch)
        with self.assertRaisesRegex(
            evidence.ConfirmationReleaseEvidenceError, "semantically"
        ):
            evidence.verify_published_finalizer(
                source_root=self.source, finalizer_job_id=self.job_id,
                study_commit=self.commit,
            )


if __name__ == "__main__":
    unittest.main()
