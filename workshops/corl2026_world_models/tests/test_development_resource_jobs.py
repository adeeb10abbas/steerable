from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
REPOSITORY = WORKSHOP.parents[1]
MODULE_PATH = WORKSHOP / "experiments/forecast_layout/development_resource_jobs.py"
SPEC = importlib.util.spec_from_file_location("development_resource_jobs_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
jobs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = jobs
SPEC.loader.exec_module(jobs)

RESULTS_COMMIT = "260338bf3226bedc6103b44209aba73ba579de6e"
STUDY_COMMIT = "a" * 40


class DevelopmentResourceJobTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            jobs._verified_results_commit(REPOSITORY, RESULTS_COMMIT)
        except jobs.ResourceQueueError as error:
            raise unittest.SkipTest(f"exact retained results commit unavailable: {error}")

    def test_source_contains_no_duplicate_literal_dictionary_keys(self) -> None:
        tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
        duplicates: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = [
                key.value
                for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
            duplicates.extend(
                (node.lineno, key) for key in set(keys) if keys.count(key) > 1
            )
        self.assertEqual(duplicates, [])

    def test_exact_results_commit_builds_one_deterministic_worker09_job(self) -> None:
        first = jobs.build_formal_wave(
            study_commit=STUDY_COMMIT,
            repository=REPOSITORY,
            results_commit=RESULTS_COMMIT,
        )
        second = jobs.build_formal_wave(
            study_commit=STUDY_COMMIT,
            repository=REPOSITORY,
            results_commit=RESULTS_COMMIT,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["science_counts"], jobs._zero_science_counts())
        self.assertFalse(first["safe_to_release_confirmation"])
        self.assertEqual(len(first["jobs"]), 1)
        descriptor = first["jobs"][0]
        self.assertEqual(descriptor["job_id"], "development-resource-compiler-formal-001")
        self.assertEqual(descriptor["role"], "wmf-forecast-0912-worker-09")
        self.assertEqual(descriptor["source_commit"], STUDY_COMMIT)
        self.assertEqual(descriptor["argv"][2], "formal-full")
        self.assertLessEqual(len(descriptor["argv"]), 256)
        self.assertEqual(len(first["inputs"]["timing_sidecar_receipts"]), 2)
        self.assertEqual(len(first["inputs"]["aggregate_receipts"]), 8)
        self.assertEqual(len(first["inputs"]["d1_server_receipts"]), 4)
        self.assertEqual(len(first["inputs"]["queue_wrapper_snapshots"]), 14)
        self.assertEqual(
            hashlib.sha256(jobs.queue.canonical_bytes(first)).hexdigest(),
            hashlib.sha256(jobs.queue.canonical_bytes(second)).hexdigest(),
        )

    def test_runtime_parser_round_trips_exact_emitted_descriptor(self) -> None:
        wave = jobs.build_formal_wave(
            study_commit=STUDY_COMMIT,
            repository=REPOSITORY,
            results_commit=RESULTS_COMMIT,
        )
        emitted = wave["jobs"][0]
        args = jobs._parser().parse_args(emitted["argv"][2:])
        rebuilt, implementation, inputs = jobs._runtime_descriptor(args)
        self.assertEqual(rebuilt, emitted)
        self.assertEqual(set(implementation), set(jobs._source_paths(REPOSITORY)))
        self.assertEqual(len(inputs["aggregate_receipts"]), 8)

    def test_all_aggregate_receipts_and_cells_are_immutably_pinned(self) -> None:
        self.assertEqual(set(jobs.RESOURCE_AGGREGATES), {"N3", "D1"})
        for model in jobs.MODELS:
            self.assertEqual(set(jobs.RESOURCE_AGGREGATES[model]), set(jobs.LAYOUTS))
            for layout in jobs.LAYOUTS:
                spec = jobs.RESOURCE_AGGREGATES[model][layout]
                self.assertRegex(str(spec.receipt_sha256), r"^[0-9a-f]{64}$")
                self.assertIsInstance(spec.receipt_bytes, int)
                self.assertGreater(spec.receipt_bytes, 0)
                self.assertEqual(len(spec.cells), 4)
                for cell in spec.cells:
                    self.assertRegex(cell.sha256, r"^[0-9a-f]{64}$")
                    self.assertGreater(cell.bytes, 0)

    def test_runtime_rejects_wrong_role_and_duplicate_input_identity(self) -> None:
        wave = jobs.build_formal_wave(
            study_commit=STUDY_COMMIT,
            repository=REPOSITORY,
            results_commit=RESULTS_COMMIT,
        )
        args = jobs._parser().parse_args(wave["jobs"][0]["argv"][2:])
        args.expected_role = "wmf-forecast-0912-worker-05"
        with self.assertRaisesRegex(jobs.ResourceQueueError, "role changed"):
            jobs._runtime_descriptor(args)

        args = jobs._parser().parse_args(wave["jobs"][0]["argv"][2:])
        args.aggregate_receipt.append(copy.deepcopy(args.aggregate_receipt[0]))
        with self.assertRaisesRegex(jobs.ResourceQueueError, "aggregate identity changed"):
            jobs._runtime_descriptor(args)

    def test_queue_snapshot_semantics_reject_wrong_worker_even_with_matching_hash(self) -> None:
        source_spec = jobs.QUEUE_SNAPSHOTS["n3-development-d01-002"]
        with jobs._materialized_results(REPOSITORY, RESULTS_COMMIT) as (_, paths):
            value = jobs.queue.load_json(paths[source_spec.results_path], "snapshot")
        value["worker_id"] = "wmf-forecast-0912-worker-09"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "snapshot.json"
            path.write_bytes(jobs.queue.canonical_bytes(value))
            digest = jobs.queue.file_identity(path)
            spec = jobs.QueueSnapshotSpec(
                source_spec.job_id,
                source_spec.worker_id,
                source_spec.source_commit,
                source_spec.descriptor_sha256,
                digest["sha256"],
                digest["bytes"],
            )
            with self.assertRaisesRegex(jobs.ResourceQueueError, "worker_id"):
                jobs._semantic_queue_snapshot(path, spec)

    def test_missing_or_noncommit_results_object_fails_closed(self) -> None:
        with self.assertRaisesRegex(jobs.ResourceQueueError, "results commit is invalid"):
            jobs.build_formal_wave(
                study_commit=STUDY_COMMIT,
                repository=REPOSITORY,
                results_commit="bad",
            )
        blob = jobs._run_git(REPOSITORY, "rev-parse", f"{RESULTS_COMMIT}:results/jobs/timing-n3-development-sidecar-001/publish/timing_job_receipt.json").decode().strip()
        with self.assertRaisesRegex(jobs.ResourceQueueError, "not a commit"):
            jobs.build_formal_wave(
                study_commit=STUDY_COMMIT,
                repository=REPOSITORY,
                results_commit=blob,
            )

    def test_failure_publication_is_zero_science_and_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            job_dir = control / "jobs" / jobs.JOB_ID
            publish = job_dir / "publish"
            publish.mkdir(parents=True)
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                jobs._write_failure_receipt(
                    context=None,
                    job_dir=job_dir,
                    error=jobs.ResourceQueueError("expected failure"),
                )
                failure_path = publish / jobs.PUBLISH_FAILURE_NAME
                failure = jobs.queue.load_json(failure_path, "failure")
                jobs.queue.verify_signed_document(failure, "failure")
                self.assertEqual(failure["science_counts"], jobs._zero_science_counts())
                self.assertFalse(failure["safe_to_release_confirmation"])
                self.assertEqual({path.name for path in publish.iterdir()}, {jobs.PUBLISH_FAILURE_NAME})

        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control"
            job_dir = control / "jobs" / jobs.JOB_ID
            publish = job_dir / "publish"
            publish.mkdir(parents=True)
            success = publish / jobs.PUBLISH_RECEIPT_NAME
            success.write_bytes(b"immutable-success")
            with mock.patch.object(jobs, "CONTROL_ROOT", control):
                jobs._write_failure_receipt(
                    context=None,
                    job_dir=job_dir,
                    error=jobs.ResourceQueueError("late failure"),
                )
            self.assertEqual({path.name for path in publish.iterdir()}, {jobs.PUBLISH_RECEIPT_NAME})
            self.assertEqual(success.read_bytes(), b"immutable-success")


if __name__ == "__main__":
    unittest.main()
