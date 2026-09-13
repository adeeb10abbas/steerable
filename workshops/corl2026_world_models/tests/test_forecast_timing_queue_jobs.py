from __future__ import annotations

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments/forecast_layout/forecast_timing_queue_jobs.py"
)
SPEC = importlib.util.spec_from_file_location("forecast_timing_queue_jobs", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
queue_jobs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue_jobs
SPEC.loader.exec_module(queue_jobs)


class ForecastTimingDescriptorTests(unittest.TestCase):
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
            duplicates.extend((node.lineno, key) for key in set(keys) if keys.count(key) > 1)
        self.assertEqual(duplicates, [])

    def test_reviewed_implementation_hashes_are_still_exact(self) -> None:
        root = Path(__file__).resolve().parents[3]
        evidence = queue_jobs._validate_staged_implementation(root, include_n3_runner=True)
        self.assertEqual(
            evidence["timing_validator"]["sha256"], queue_jobs.TOOL_SHA256
        )
        self.assertEqual(
            evidence["timing_contract"]["sha256"], queue_jobs.CONTRACT_SHA256
        )
        self.assertEqual(
            evidence["n3_generation_runner"]["sha256"], queue_jobs.N3_RUNNER_SHA256
        )
        self.assertEqual(
            evidence["n3_runtime_contract"]["sha256"],
            queue_jobs.N3_RUNTIME_CONTRACT_SHA256,
        )

    def test_initial_wave_is_exactly_four_zero_action_descriptor_only_jobs(self) -> None:
        commit = "a" * 40
        wave = queue_jobs.build_initial_wave(commit)
        self.assertEqual(wave["status"], "descriptor_only_not_dispatched")
        self.assertEqual(wave["generation_requests_issued"], 0)
        self.assertEqual(wave["behavioral_actions_executed"], 0)
        self.assertEqual(
            [job["job_id"] for job in wave["jobs"]],
            [job.job_id for job in queue_jobs.INITIAL_JOBS],
        )
        self.assertEqual(
            [job["role"] for job in wave["jobs"]],
            [
                "wmf-forecast-0912-worker-05",
                "wmf-forecast-0912-worker-06",
                "wmf-forecast-0912-worker-09",
                "wmf-forecast-0912-worker-09",
            ],
        )
        for descriptor in wave["jobs"]:
            self.assertIs(descriptor["released"], True)
            self.assertEqual(descriptor["source_commit"], commit)
            command = " ".join(descriptor["argv"])
            self.assertNotIn("n3_first_live.py", command)
            self.assertNotIn("simulator", command)
            self.assertNotIn("behavioral", command)
            self.assertIn(queue_jobs.CONTRACT_SHA256, command)
        self.assertIn("--capture-receipt-sha256", wave["jobs"][2]["argv"])
        self.assertIn(queue_jobs.CAPTURE_RECEIPT_SHA256, wave["jobs"][2]["argv"])
        self.assertIn(queue_jobs.D1_QUALIFICATION_RECEIPT_SHA256, wave["jobs"][3]["argv"])

    def test_worker09_jobs_remain_ordered_and_cannot_claim_concurrently(self) -> None:
        jobs = queue_jobs.build_initial_wave("b" * 40)["jobs"]
        worker09 = [job for job in jobs if job["role"] == "wmf-forecast-0912-worker-09"]
        self.assertEqual(
            [job["job_id"] for job in worker09],
            [
                "timing-n3-live-input-p00-001",
                "timing-d1-normalize-generation-001",
            ],
        )

    def test_invalid_commit_and_unknown_job_fail_closed(self) -> None:
        with self.assertRaisesRegex(queue_jobs.TimingQueueError, "full lowercase"):
            queue_jobs.build_initial_wave("abc")
        outsider = queue_jobs.InitialJob("x", "x", "any", None, 10)
        with self.assertRaisesRegex(queue_jobs.TimingQueueError, "outside"):
            queue_jobs.build_initial_descriptor(outsider, "a" * 40)

    def _preparation_receipt(self) -> dict:
        root = (
            queue_jobs.PREPARATION_CLUSTER_JOB_DIR / "raw" / "n3_live_input"
        )
        return queue_jobs.signed_document(
            {
                "schema_version": queue_jobs.TIMING_JOB_SCHEMA,
                "namespace": queue_jobs.NAMESPACE,
                "study_id": queue_jobs.STUDY_ID,
                "status": "passed",
                "decision": "go",
                "mode": queue_jobs.PREPARATION_JOB.mode,
                "job_id": queue_jobs.PREPARATION_JOB.job_id,
                "job_dir": str(queue_jobs.PREPARATION_CLUSTER_JOB_DIR),
                "study_commit": "c" * 40,
                "queue_role": queue_jobs.PREPARATION_JOB.role,
                "worker_id": queue_jobs.PREPARATION_JOB.role,
                "physical_time_qualified": False,
                "behavioral_policy_skill_evaluated": False,
                "science_counts": queue_jobs._science_counts(),
                "outputs": {
                    "primary": {
                        "path": str(root / "preparation_receipt.json"),
                        "bytes": 1100,
                        "sha256": "1" * 64,
                    },
                    "observation_manifest": {
                        "path": str(root / "observation_manifest.json"),
                        "bytes": 900,
                        "sha256": "2" * 64,
                    },
                    "observation_payload": {
                        "path": str(root / "observation.npz"),
                        "bytes": 500000,
                        "sha256": "3" * 64,
                    },
                },
            }
        )

    def _write_receipt(self, root: Path, receipt: dict | None = None) -> tuple[Path, str]:
        path = root / "timing_job_receipt.json"
        path.write_bytes(queue_jobs.canonical_bytes(receipt or self._preparation_receipt()))
        return path, queue_jobs.sha256_file(path)

    def _n3_generation_receipt(self) -> dict:
        job = queue_jobs.N3_GENERATION_CLUSTER_JOB_DIR
        source = (
            queue_jobs.CONTROL_ROOT
            / "sources"
            / queue_jobs.N3_GENERATION_STUDY_COMMIT
        )

        def descriptor(path: Path, digest: str, size: int = 100) -> dict:
            return {"path": str(path), "bytes": size, "sha256": digest}

        implementation = {
            "timing_validator": descriptor(
                source / queue_jobs.TOOL_RELATIVE, queue_jobs.TOOL_SHA256
            ),
            "timing_contract": descriptor(
                source / queue_jobs.CONTRACT_RELATIVE, queue_jobs.CONTRACT_SHA256
            ),
            "n3_generation_runner": descriptor(
                source / queue_jobs.N3_RUNNER_RELATIVE, queue_jobs.N3_RUNNER_SHA256
            ),
            "n3_runtime_contract": descriptor(
                source / queue_jobs.N3_RUNTIME_CONTRACT_RELATIVE,
                queue_jobs.N3_RUNTIME_CONTRACT_SHA256,
            ),
        }
        probe = descriptor(job / "raw" / "n3_generation_probe.json", "4" * 64, 4729)
        qualification = descriptor(
            job / "raw" / "n3_generation_publish" / "n3_qualification.json",
            "5" * 64,
            2975,
        )
        child = [
            str(queue_jobs.COSMOS_PYTHON),
            str(source / queue_jobs.N3_RUNNER_RELATIVE),
            "--source-root",
            str(queue_jobs.N3_SOURCE),
            "--checkpoint-root",
            str(queue_jobs.N3_CHECKPOINT),
            "--observation-manifest",
            str(
                queue_jobs.PREPARATION_CLUSTER_JOB_DIR
                / "raw"
                / "n3_live_input"
                / "observation_manifest.json"
            ),
            "--output-dir",
            str(job / "raw" / "n3_generation_raw"),
            "--publish-dir",
            str(job / "raw" / "n3_generation_publish"),
            "--effective-seed",
            str(queue_jobs.EFFECTIVE_SEED),
        ]
        return queue_jobs.signed_document(
            {
                "schema_version": queue_jobs.TIMING_JOB_SCHEMA,
                "namespace": queue_jobs.NAMESPACE,
                "study_id": queue_jobs.STUDY_ID,
                "status": "passed",
                "decision": "go",
                "mode": "n3-generate",
                "job_id": queue_jobs.N3_GENERATION_JOB_ID,
                "job_dir": str(job),
                "study_commit": queue_jobs.N3_GENERATION_STUDY_COMMIT,
                "queue_role": queue_jobs.N3_GENERATION_ROLE,
                "worker_id": queue_jobs.N3_GENERATION_WORKER_ID,
                "runtime_identity": {
                    "hostname": queue_jobs.N3_GENERATION_WORKER_ID + "-pod",
                    "pod_uid": "pod-uid",
                    "pid": 42,
                },
                "queue_descriptor": descriptor(job / "descriptor.json", "6" * 64),
                "queue_claim": descriptor(job / "claim" / "owner.json", "7" * 64),
                "implementation": implementation,
                "inputs": {
                    "preparation_job_receipt": descriptor(
                        queue_jobs.PREPARATION_CLUSTER_JOB_RECEIPT,
                        queue_jobs.PREPARATION_CLUSTER_JOB_RECEIPT_SHA256,
                        4125,
                    ),
                    "preparation_receipt": descriptor(
                        queue_jobs.PREPARATION_RAW_ROOT / "preparation_receipt.json",
                        queue_jobs.PREPARATION_RECEIPT_SHA256,
                        1787,
                    ),
                    "observation_manifest": descriptor(
                        queue_jobs.PREPARATION_RAW_ROOT / "observation_manifest.json",
                        queue_jobs.PREPARATION_MANIFEST_SHA256,
                        2128,
                    ),
                    "observation_payload": descriptor(
                        queue_jobs.PREPARATION_RAW_ROOT / "observation.npz",
                        queue_jobs.PREPARATION_PAYLOAD_SHA256,
                        1037592,
                    ),
                },
                "physical_time_qualified": False,
                "behavioral_policy_skill_evaluated": False,
                "safe_to_release_confirmation": False,
                "science_counts": queue_jobs._expected_science_counts(
                    issued=6, referenced=0
                ),
                "child": {"argv": child, "returncode": 0, "reaped": True},
                "outputs": {
                    "normalized_generation_probe": probe,
                    "published_generation_probe": descriptor(
                        job / "publish" / "n3_generation_probe.json", "4" * 64, 4729
                    ),
                    "qualification": qualification,
                    "published_qualification": descriptor(
                        job / "publish" / "n3_qualification.json", "5" * 64, 2975
                    ),
                },
            }
        )

    @staticmethod
    def _resign(receipt: dict) -> dict:
        changed = dict(receipt)
        changed.pop("payload_sha256", None)
        return queue_jobs.signed_document(changed)

    def test_n3_descriptor_requires_and_binds_passed_preparation_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_receipt(Path(temporary))
            descriptor = queue_jobs.build_n3_generation_descriptor(
                "d" * 40,
                preparation_job_receipt_path=path,
                preparation_job_receipt_sha256=digest,
            )
        self.assertEqual(descriptor["job_id"], queue_jobs.N3_GENERATION_JOB_ID)
        self.assertEqual(
            descriptor["job_id"], "timing-n3-live-generation-p00-002"
        )
        self.assertEqual(descriptor["role"], "n3")
        self.assertIs(descriptor["released"], True)
        command = " ".join(descriptor["argv"])
        self.assertIn("CUDA_VISIBLE_DEVICES=0", command)
        self.assertIn("n3-generate", command)
        self.assertIn(
            f"--expected-worker-id {queue_jobs.N3_GENERATION_WORKER_ID}", command
        )
        self.assertIn(str(queue_jobs.PREPARATION_CLUSTER_JOB_RECEIPT), command)
        self.assertIn("1" * 64, command)
        self.assertIn("2" * 64, command)
        self.assertNotIn("--remote", command)

    def test_n3_descriptor_does_not_exist_without_receipt_or_with_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(queue_jobs.TimingQueueError):
                queue_jobs.build_n3_generation_descriptor(
                    "d" * 40,
                    preparation_job_receipt_path=root / "missing.json",
                    preparation_job_receipt_sha256="0" * 64,
                )

            receipt = self._preparation_receipt()
            receipt["science_counts"]["model_requests_issued_by_job"] = 1
            receipt.pop("payload_sha256")
            receipt = queue_jobs.signed_document(receipt)
            path, digest = self._write_receipt(root, receipt)
            with self.assertRaisesRegex(queue_jobs.TimingQueueError, "unexpected science"):
                queue_jobs.build_n3_generation_descriptor(
                    "d" * 40,
                    preparation_job_receipt_path=path,
                    preparation_job_receipt_sha256=digest,
                )

    def test_n3_descriptor_rejects_receipt_hash_or_output_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _ = self._write_receipt(root)
            with self.assertRaisesRegex(queue_jobs.TimingQueueError, "hash mismatch"):
                queue_jobs.validate_preparation_job_receipt(path, "0" * 64)

            receipt = self._preparation_receipt()
            receipt["outputs"]["primary"]["path"] = "/tmp/substituted.json"
            receipt.pop("payload_sha256")
            receipt = queue_jobs.signed_document(receipt)
            path.unlink()
            path, digest = self._write_receipt(root, receipt)
            with self.assertRaisesRegex(queue_jobs.TimingQueueError, "escaped"):
                queue_jobs.validate_preparation_job_receipt(path, digest)

    def test_runtime_n3_descriptor_is_identical_to_receipt_gated_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_receipt(Path(temporary))
            descriptor = queue_jobs.build_n3_generation_descriptor(
                "d" * 40,
                preparation_job_receipt_path=path,
                preparation_job_receipt_sha256=digest,
            )
        receipt = self._preparation_receipt()
        args = argparse.Namespace(
            job_id=queue_jobs.N3_GENERATION_JOB_ID,
            expected_role="n3",
            expected_worker_id=queue_jobs.N3_GENERATION_WORKER_ID,
            contract_sha256=queue_jobs.CONTRACT_SHA256,
            study_commit="d" * 40,
            preparation_job_receipt=queue_jobs.PREPARATION_CLUSTER_JOB_RECEIPT,
            preparation_job_receipt_sha256=digest,
            preparation_receipt=Path(receipt["outputs"]["primary"]["path"]),
            preparation_receipt_sha256="1" * 64,
            observation_manifest=Path(receipt["outputs"]["observation_manifest"]["path"]),
            observation_manifest_sha256="2" * 64,
        )
        self.assertEqual(queue_jobs._runtime_n3_descriptor(args), descriptor)

    def test_authority_wave_requires_attempt002_and_binds_both_models(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_receipt(
                Path(temporary), self._n3_generation_receipt()
            )
            wave = queue_jobs.build_authority_wave(
                "e" * 40,
                n3_generation_job_receipt_path=path,
                n3_generation_job_receipt_sha256=digest,
            )
        self.assertEqual(
            [job["job_id"] for job in wave["jobs"]],
            ["timing-n3-native-authority-001", "timing-d1-native-authority-001"],
        )
        self.assertEqual(
            [job["role"] for job in wave["jobs"]],
            ["wmf-forecast-0912-worker-05", "wmf-forecast-0912-worker-06"],
        )
        self.assertEqual(wave["generation_requests_issued_by_wave"], 0)
        self.assertEqual(wave["behavioral_actions_executed"], 0)
        self.assertIs(wave["safe_to_release_confirmation"], False)
        n3_command = " ".join(wave["jobs"][0]["argv"])
        d1_command = " ".join(wave["jobs"][1]["argv"])
        self.assertIn(digest, n3_command)
        self.assertIn("4" * 64, n3_command)
        self.assertIn(queue_jobs.N3_SOURCE_AUDIT_JOB.receipt_sha256, n3_command)
        self.assertIn(queue_jobs.D1_SOURCE_AUDIT_JOB.receipt_sha256, d1_command)
        self.assertIn(queue_jobs.D1_GENERATION_JOB.receipt_sha256, d1_command)
        for descriptor in wave["jobs"]:
            command = " ".join(descriptor["argv"])
            self.assertNotIn("n3_first_live.py", command)
            self.assertNotIn("simulator", command)
            self.assertIn(queue_jobs.RECORDER_RECEIPT_SHA256, command)

    def test_authority_wave_rejects_missing_failed_or_drifted_attempt002(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(queue_jobs.TimingQueueError):
                queue_jobs.build_authority_wave(
                    "e" * 40,
                    n3_generation_job_receipt_path=root / "missing.json",
                    n3_generation_job_receipt_sha256="0" * 64,
                )
            for field, value, message in (
                ("status", "technical_invalid", "receipt changed: status"),
                ("study_commit", "f" * 40, "receipt changed: study_commit"),
            ):
                receipt = self._n3_generation_receipt()
                receipt[field] = value
                receipt = self._resign(receipt)
                path, digest = self._write_receipt(root, receipt)
                with self.assertRaisesRegex(queue_jobs.TimingQueueError, message):
                    queue_jobs.build_authority_wave(
                        "e" * 40,
                        n3_generation_job_receipt_path=path,
                        n3_generation_job_receipt_sha256=digest,
                    )
                path.unlink()

            receipt = self._n3_generation_receipt()
            receipt["science_counts"]["behavioral_actions"] = 1
            receipt = self._resign(receipt)
            path, digest = self._write_receipt(root, receipt)
            with self.assertRaisesRegex(queue_jobs.TimingQueueError, "science counts"):
                queue_jobs.build_authority_wave(
                    "e" * 40,
                    n3_generation_job_receipt_path=path,
                    n3_generation_job_receipt_sha256=digest,
                )
            path.unlink()

            receipt = self._n3_generation_receipt()
            receipt["inputs"]["observation_manifest"]["sha256"] = "8" * 64
            receipt = self._resign(receipt)
            path, digest = self._write_receipt(root, receipt)
            with self.assertRaisesRegex(queue_jobs.TimingQueueError, "manifest input hash"):
                queue_jobs.build_authority_wave(
                    "e" * 40,
                    n3_generation_job_receipt_path=path,
                    n3_generation_job_receipt_sha256=digest,
                )

    def test_runtime_authority_descriptors_equal_receipt_gated_builder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path, digest = self._write_receipt(
                Path(temporary), self._n3_generation_receipt()
            )
            wave = queue_jobs.build_authority_wave(
                "e" * 40,
                n3_generation_job_receipt_path=path,
                n3_generation_job_receipt_sha256=digest,
            )
        parser = queue_jobs._parser()
        for descriptor in wave["jobs"]:
            runtime_argv = [
                value.replace("{source_root}", "/tmp/source").replace(
                    "{job_dir}", "/tmp/job"
                )
                for value in descriptor["argv"][2:]
            ]
            args = parser.parse_args(runtime_argv)
            _, rebuilt = queue_jobs._runtime_authority_descriptor(args)
            self.assertEqual(rebuilt, descriptor)


class AuthorityRuntimeTests(unittest.TestCase):
    def _args(self, root: Path) -> argparse.Namespace:
        job = queue_jobs.AUTHORITY_BY_MODE["d1-native-authority"]
        return argparse.Namespace(
            command=job.mode,
            source_root=root / "source",
            study_commit="e" * 40,
            job_dir=root / job.job_id,
            job_id=job.job_id,
            expected_role=job.role,
            contract_sha256=queue_jobs.CONTRACT_SHA256,
            source_audit_job_receipt=queue_jobs.D1_SOURCE_AUDIT_JOB.receipt_path,
            source_audit_job_receipt_sha256=queue_jobs.D1_SOURCE_AUDIT_JOB.receipt_sha256,
            source_audit=queue_jobs.D1_SOURCE_AUDIT_JOB.artifact_path,
            source_audit_sha256=queue_jobs.D1_SOURCE_AUDIT_JOB.artifact_sha256,
            generation_job_receipt=queue_jobs.D1_GENERATION_JOB.receipt_path,
            generation_job_receipt_sha256=queue_jobs.D1_GENERATION_JOB.receipt_sha256,
            generation_probe=queue_jobs.D1_GENERATION_JOB.artifact_path,
            generation_probe_sha256=queue_jobs.D1_GENERATION_JOB.artifact_sha256,
            recorder_receipt=queue_jobs.RECORDER_RECEIPT,
            recorder_receipt_sha256=queue_jobs.RECORDER_RECEIPT_SHA256,
            camera_id=queue_jobs.CAMERA_ID,
            expected_target_count=job.expected_target_count,
        )

    def _context(
        self, args: argparse.Namespace, descriptor: dict
    ) -> queue_jobs.QueueContext:
        identity = {"path": "/exact.json", "bytes": 1, "sha256": "9" * 64}
        return queue_jobs.QueueContext(
            source_root=Path(args.source_root),
            job_dir=Path(args.job_dir),
            study_commit=args.study_commit,
            job_id=args.job_id,
            role=args.expected_role,
            descriptor=queue_jobs._normalized_descriptor(descriptor),
            descriptor_identity=identity,
            claim_identity=identity,
            worker_id=args.expected_role,
            hostname=args.expected_role + "-pod",
            pod_uid="pod-uid",
        )

    @staticmethod
    def _timing(target_count: int) -> mock.Mock:
        timing = mock.Mock()
        result = {
            "schema_version": "wmf-forecast-native-target-authority-v1",
            "generated_targets": [
                {"generated_frame_index": index + 1, "target_physical_time_s": index + 0.1}
                for index in range(target_count)
            ],
        }
        timing.qualify_timing.return_value = result
        timing.validate_timing_authority.return_value = result

        def write(path: Path, value: dict) -> None:
            Path(path).write_bytes(queue_jobs.canonical_bytes(value))

        timing.atomic_json.side_effect = write
        return timing

    def test_authority_runtime_emits_zero_action_receipt_and_keeps_hold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            args.job_dir.mkdir()
            _, descriptor = queue_jobs._runtime_authority_descriptor(args)
            context = self._context(args, descriptor)
            timing = self._timing(args.expected_target_count)
            with mock.patch.object(
                queue_jobs, "validate_queue_context", return_value=context
            ), mock.patch.object(
                queue_jobs, "_validate_staged_implementation", return_value={}
            ), mock.patch.object(
                queue_jobs, "_authority_prerequisites", return_value={}
            ), mock.patch.object(queue_jobs, "_load_module", return_value=timing):
                receipt = queue_jobs.run_authority_job(args)
            self.assertEqual(receipt["status"], "passed")
            self.assertIs(receipt["physical_time_qualified"], True)
            self.assertIs(receipt["behavioral_policy_skill_evaluated"], False)
            self.assertIs(receipt["safe_to_release_confirmation"], False)
            self.assertEqual(receipt["science_counts"]["model_requests_issued_by_job"], 0)
            self.assertEqual(receipt["science_counts"]["behavioral_actions"], 0)
            self.assertEqual(receipt["science_counts"]["referenced_generation_requests"], 6)
            self.assertEqual(receipt["evidence_counts"]["qualified_generated_targets"], 2)
            self.assertTrue(
                (args.job_dir / "publish" / "timing_job_receipt.json").is_file()
            )

    def test_authority_target_count_failure_records_zero_science(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            args.job_dir.mkdir()
            _, descriptor = queue_jobs._runtime_authority_descriptor(args)
            context = self._context(args, descriptor)
            timing = self._timing(1)
            with mock.patch.object(
                queue_jobs, "validate_queue_context", return_value=context
            ), mock.patch.object(
                queue_jobs, "_validate_staged_implementation", return_value={}
            ), mock.patch.object(
                queue_jobs, "_authority_prerequisites", return_value={}
            ), mock.patch.object(queue_jobs, "_load_module", return_value=timing):
                with self.assertRaisesRegex(queue_jobs.TimingQueueError, "target count"):
                    queue_jobs.run_authority_job(args)
            failure = queue_jobs.load_json(
                args.job_dir / "publish" / "timing_job_failure.json",
                "authority failure",
            )
            self.assertIs(failure["safe_to_release_confirmation"], False)
            for field in (
                "model_runtime_loads",
                "model_servers_started",
                "model_requests_issued_by_job",
                "physical_resets",
                "robot_episodes",
                "behavioral_actions",
                "behavioral_cells",
            ):
                self.assertEqual(failure["science_counts"][field], 0, field)


class QueueContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.state = self.root / "control"
        repository = self.root / "repository"
        repository.mkdir()
        subprocess.run(["git", "init", "-q", repository], check=True)
        subprocess.run(["git", "-C", repository, "config", "user.name", "Test"], check=True)
        subprocess.run(
            ["git", "-C", repository, "config", "user.email", "test@example.invalid"],
            check=True,
        )
        (repository / "tracked.txt").write_text("immutable\n", encoding="utf-8")
        subprocess.run(["git", "-C", repository, "add", "tracked.txt"], check=True)
        subprocess.run(["git", "-C", repository, "commit", "-qm", "source"], check=True)
        self.commit = subprocess.run(
            ["git", "-C", repository, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.source = self.state / "sources" / self.commit
        self.source.parent.mkdir(parents=True)
        shutil.move(str(repository), self.source)
        self.job_spec = queue_jobs.INITIAL_BY_MODE["n3-source-audit"]
        self.job_dir = self.state / "jobs" / self.job_spec.job_id
        self.job_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _stage(self, *, role: str | None = None, claim_role: str | None = None) -> dict:
        descriptor = queue_jobs.build_initial_descriptor(self.job_spec, self.commit)
        normalized = queue_jobs._normalized_descriptor(descriptor)
        if role is not None:
            normalized["role"] = role
        descriptor_path = self.job_dir / "descriptor.json"
        descriptor_path.write_bytes(queue_jobs.canonical_bytes(normalized))
        claim = self.job_dir / "claim"
        claim.mkdir()
        (claim / "owner.json").write_bytes(
            queue_jobs.canonical_bytes(
                {
                    "worker_id": claim_role or self.job_spec.role,
                    "worker_pid": 42,
                    "control_commit": "f" * 40,
                    "control_generation": 9,
                    "descriptor_sha256": queue_jobs.sha256_file(descriptor_path),
                    "release_boundary": "claim_committed_under_shared_release_lock",
                }
            )
        )
        return descriptor

    def _validate(self, descriptor: dict):
        with mock.patch.object(
            queue_jobs.socket, "gethostname", return_value=self.job_spec.role + "-pod"
        ), mock.patch.dict(os.environ, {"POD_UID": "pod-uid"}, clear=False):
            return queue_jobs.validate_queue_context(
                source_root=self.source,
                job_dir=self.job_dir,
                study_commit=self.commit,
                job_id=self.job_spec.job_id,
                expected_role=self.job_spec.role,
                expected_descriptor=descriptor,
            )

    def _n3_args(self) -> argparse.Namespace:
        prepared = (
            queue_jobs.PREPARATION_CLUSTER_JOB_DIR / "raw" / "n3_live_input"
        )
        return argparse.Namespace(
            command="n3-generate",
            source_root=self.source,
            study_commit=self.commit,
            job_dir=self.state / "jobs" / queue_jobs.N3_GENERATION_JOB_ID,
            job_id=queue_jobs.N3_GENERATION_JOB_ID,
            expected_role=queue_jobs.N3_GENERATION_ROLE,
            expected_worker_id=queue_jobs.N3_GENERATION_WORKER_ID,
            contract_sha256=queue_jobs.CONTRACT_SHA256,
            preparation_job_receipt=queue_jobs.PREPARATION_CLUSTER_JOB_RECEIPT,
            preparation_job_receipt_sha256="0" * 64,
            preparation_receipt=prepared / "preparation_receipt.json",
            preparation_receipt_sha256="1" * 64,
            observation_manifest=prepared / "observation_manifest.json",
            observation_manifest_sha256="2" * 64,
        )

    def _stage_n3(self, *, claim_worker: str) -> tuple[argparse.Namespace, dict]:
        args = self._n3_args()
        args.job_dir.mkdir(parents=True)
        descriptor = queue_jobs._runtime_n3_descriptor(args)
        descriptor_path = args.job_dir / "descriptor.json"
        descriptor_path.write_bytes(
            queue_jobs.canonical_bytes(queue_jobs._normalized_descriptor(descriptor))
        )
        claim = args.job_dir / "claim"
        claim.mkdir()
        (claim / "owner.json").write_bytes(
            queue_jobs.canonical_bytes(
                {
                    "worker_id": claim_worker,
                    "worker_pid": 42,
                    "control_commit": "f" * 40,
                    "control_generation": 10,
                    "descriptor_sha256": queue_jobs.sha256_file(descriptor_path),
                    "release_boundary": "claim_committed_under_shared_release_lock",
                }
            )
        )
        return args, descriptor

    def test_exact_descriptor_claim_role_and_commit_pass(self) -> None:
        descriptor = self._stage()
        context = self._validate(descriptor)
        self.assertEqual(context.study_commit, self.commit)
        self.assertEqual(context.worker_id, self.job_spec.role)
        self.assertEqual(
            context.claim_identity["path"],
            str((self.job_dir / "claim" / "owner.json").resolve()),
        )

    def test_descriptor_role_substitution_fails_before_work(self) -> None:
        descriptor = self._stage(role="wmf-forecast-0912-worker-06")
        with self.assertRaisesRegex(queue_jobs.TimingQueueError, "descriptor differs"):
            self._validate(descriptor)

    def test_claim_role_substitution_fails_before_work(self) -> None:
        descriptor = self._stage(claim_role="wmf-forecast-0912-worker-06")
        with self.assertRaisesRegex(queue_jobs.TimingQueueError, "claim worker identity"):
            self._validate(descriptor)

    def test_dedicated_n3_role_is_distinct_from_exact_claim_worker(self) -> None:
        args, descriptor = self._stage_n3(
            claim_worker=queue_jobs.N3_GENERATION_WORKER_ID
        )
        self.assertEqual(descriptor["role"], "n3")
        with mock.patch.object(
            queue_jobs.socket,
            "gethostname",
            return_value=queue_jobs.N3_GENERATION_WORKER_ID + "-pod",
        ), mock.patch.dict(os.environ, {"POD_UID": "n3-pod-uid"}, clear=False):
            context = queue_jobs.validate_queue_context(
                source_root=self.source,
                job_dir=args.job_dir,
                study_commit=self.commit,
                job_id=args.job_id,
                expected_role=args.expected_role,
                expected_worker_id=args.expected_worker_id,
                expected_descriptor=descriptor,
            )
        self.assertEqual(context.role, "n3")
        self.assertEqual(context.worker_id, queue_jobs.N3_GENERATION_WORKER_ID)

    def test_dedicated_n3_rejects_role_name_as_claim_worker(self) -> None:
        args, descriptor = self._stage_n3(claim_worker="n3")
        with mock.patch.object(
            queue_jobs.socket,
            "gethostname",
            return_value=queue_jobs.N3_GENERATION_WORKER_ID + "-pod",
        ), mock.patch.dict(os.environ, {"POD_UID": "n3-pod-uid"}, clear=False):
            with self.assertRaisesRegex(
                queue_jobs.TimingQueueError, "claim worker identity"
            ):
                queue_jobs.validate_queue_context(
                    source_root=self.source,
                    job_dir=args.job_dir,
                    study_commit=self.commit,
                    job_id=args.job_id,
                    expected_role=args.expected_role,
                    expected_worker_id=args.expected_worker_id,
                    expected_descriptor=descriptor,
                )

    def test_pre_model_n3_identity_failure_records_zero_science(self) -> None:
        args, _ = self._stage_n3(claim_worker="n3")
        with mock.patch.object(
            queue_jobs.socket,
            "gethostname",
            return_value=queue_jobs.N3_GENERATION_WORKER_ID + "-pod",
        ), mock.patch.dict(os.environ, {"POD_UID": "n3-pod-uid"}, clear=False):
            with self.assertRaisesRegex(
                queue_jobs.TimingQueueError, "claim worker identity"
            ):
                queue_jobs.run_n3_generation_job(args)
        failure = queue_jobs.load_json(
            args.job_dir / "publish" / "timing_job_failure.json",
            "N3 timing failure",
        )
        self.assertEqual(failure["status"], "technical_invalid")
        self.assertIs(failure["science_counts"]["n3_generation_child_started"], False)
        for field in (
            "model_runtime_loads",
            "model_servers_started",
            "model_requests_issued_by_job",
            "model_requests_completed_before_failure",
            "physical_resets",
            "robot_episodes",
            "behavioral_actions",
            "behavioral_cells",
        ):
            self.assertEqual(failure["science_counts"][field], 0, field)

    def test_dirty_staged_source_fails_before_work(self) -> None:
        descriptor = self._stage()
        (self.source / "untracked.txt").write_text("drift\n", encoding="utf-8")
        with self.assertRaisesRegex(queue_jobs.TimingQueueError, "dirty"):
            self._validate(descriptor)

    def test_hostname_and_pod_uid_are_part_of_role_identity(self) -> None:
        descriptor = self._stage()
        with mock.patch.object(queue_jobs.socket, "gethostname", return_value="wrong-pod"), \
                mock.patch.dict(os.environ, {"POD_UID": "pod-uid"}, clear=False):
            with self.assertRaisesRegex(queue_jobs.TimingQueueError, "hostname"):
                queue_jobs.validate_queue_context(
                    source_root=self.source,
                    job_dir=self.job_dir,
                    study_commit=self.commit,
                    job_id=self.job_spec.job_id,
                    expected_role=self.job_spec.role,
                    expected_descriptor=descriptor,
                )

    def test_preexisting_output_directory_is_never_reused(self) -> None:
        (self.job_dir / "raw").mkdir()
        with self.assertRaisesRegex(queue_jobs.TimingQueueError, "already exists"):
            queue_jobs._prepare_output_directories(self.job_dir)


if __name__ == "__main__":
    unittest.main()
