"""Local queue-wrapper tests with a real fake child subprocess."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


WORKSHOP = Path(__file__).resolve().parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))
import fixture_job  # noqa: E402


FAKE_GATE = r'''#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

def canonical(value):
    return (json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--source-contract")
parser.add_argument("--source-contract-sha256")
parser.add_argument("--candidate-pool")
parser.add_argument("--candidate-pool-sha256")
parser.add_argument("--layout-pair-id")
parser.add_argument("--candidate-id")
parser.add_argument("--adapter")
parser.add_argument("--adapter-config")
parser.add_argument("--ledger")
parser.add_argument("--attempt-root")
args = parser.parse_args()
counter = Path(os.environ["FAKE_GATE_COUNTER"])
counter.write_text(str((int(counter.read_text()) if counter.exists() else 0) + 1))
mode = os.environ.get("FAKE_GATE_MODE", "accepted")
if mode == "crash":
    print("simulated child crash", file=sys.stderr)
    raise SystemExit(17)
pool = json.loads(Path(args.candidate_pool).read_text())
candidate = next(row for row in pool["candidates"] if row["candidate_id"] == args.candidate_id)
ledger = Path(args.ledger)
records = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
attempt_number = sum(row.get("candidate_id") == args.candidate_id for row in records)
attempt_dir = Path(args.attempt_root) / args.layout_pair_id / args.candidate_id / f"attempt_{attempt_number:03d}"
attempt_dir.mkdir(parents=True)
attempt = {
    "schema_version": "fake-live-gate-attempt-v1",
    "candidate_id": args.candidate_id,
    "decision": mode,
    "cache_paths": {
        name: os.environ[name]
        for name in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "HF_HOME", "PYTHONPYCACHEPREFIX")
    },
}
attempt_payload = canonical(attempt)
attempt_path = attempt_dir / "gate_attempt_receipt.json"
attempt_path.write_bytes(attempt_payload)
previous = records[-1]["record_sha256"] if records else None
record = {
    "schema_version": "wmf-forecast-layout-gate-ledger-record-v1",
    "study_namespace": "wmf_ablation_001_20260912",
    "sequence": len(records),
    "previous_record_sha256": previous,
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "layout_pair_id": args.layout_pair_id,
    "candidate_id": args.candidate_id,
    "candidate_rank": candidate["candidate_rank"],
    "candidate_payload_sha256": candidate["candidate_payload_sha256"],
    "candidate_pool_sha256": args.candidate_pool_sha256,
    "decision": mode,
    "passed": mode == "accepted",
    "model_request_count": 0,
    "behavioral_action_count": 0,
    "attempt_number": attempt_number,
    "attempt_receipt": {
        "path": str(attempt_path.resolve()),
        "sha256": hashlib.sha256(attempt_payload).hexdigest(),
        "bytes": len(attempt_payload),
    },
    "failure_count": 0 if mode == "accepted" else 1,
    "failures": [] if mode == "accepted" else ["simulated physical check"],
}
record["record_sha256"] = hashlib.sha256(canonical(record)).hexdigest()
ledger.parent.mkdir(parents=True, exist_ok=True)
with ledger.open("ab") as stream:
    stream.write(json.dumps(record, allow_nan=False, sort_keys=True, separators=(",", ":")).encode() + b"\n")
print(json.dumps({"candidate_id": args.candidate_id, "decision": mode, "record_sha256": record["record_sha256"]}))
code = {"accepted": 0, "physical_rejection": 2, "technical_invalid": 3}[mode]
if os.environ.get("FAKE_GATE_NORMALIZE_EXIT") == "1":
    code = 0
raise SystemExit(code)
'''


FAKE_NVIDIA_SMI = r'''#!/usr/bin/env python3
import os
import sys
if any(arg.startswith("--query-gpu=") for arg in sys.argv):
    print("0, GPU-test-fixture, NVIDIA B200, 580.95.05")
elif any(arg.startswith("--query-compute-apps=") for arg in sys.argv):
    if os.environ.get("FAKE_GPU_BUSY") == "1":
        print("GPU-test-fixture, 1234, unrelated, 1024")
else:
    raise SystemExit(4)
'''


class QueueHarness:
    def __init__(self, temporary: str, *, dynamic_worker: bool = False) -> None:
        self.root = Path(temporary)
        self.private = self.root / "private"
        self.state = self.private / "queue-state"
        (self.state / "sources").mkdir(parents=True)
        (self.state / "jobs").mkdir()
        staging = self.root / "staging-source"
        forecast = staging / "workshops/corl2026_world_models/experiments/forecast_layout"
        autonomy = staging / "workshops/corl2026_world_models/execution/20260912/autonomy"
        scripts = staging / "workshops/corl2026_world_models/scripts"
        forecast.mkdir(parents=True)
        autonomy.mkdir(parents=True)
        scripts.mkdir(parents=True)
        shutil.copy2(FORECAST / "layout_source_contract.json", forecast / "layout_source_contract.json")
        shutil.copy2(FORECAST / "layout_candidate_pool.json", forecast / "layout_candidate_pool.json")
        shutil.copy2(WORKSHOP / "scripts/cluster_worker_diagnostic.py", scripts / "cluster_worker_diagnostic.py")
        (forecast / "model_blind_fixture_gate.py").write_text(FAKE_GATE)
        self.worker_role = "fixture-worker-00"
        self.hostname = self.worker_role + "-testpod"
        self.pod_uid = "11111111-2222-3333-4444-555555555555"
        deployment = {
            "schema_version": "wmf-deployment-receipt-v1",
            "status": "deployed_and_execution_handed_to_durable_workstation_coordinator",
            "pods": [
                {
                    "pod": self.hostname,
                    "uid": self.pod_uid,
                    "role": "worker",
                    "job": "fixture-worker-job-00",
                    "node": "test-b200-node",
                    "phase": "Running",
                    "ready": True,
                }
            ],
        }
        (autonomy / "deployment_receipt.json").write_text(json.dumps(deployment, indent=2, sort_keys=True) + "\n")
        self._git(staging, "init", "-q")
        self._git(staging, "config", "user.name", "Fixture Test")
        self._git(staging, "config", "user.email", "fixture-test@example.invalid")
        self._git(staging, "add", ".")
        self._git(staging, "commit", "-q", "-m", "test source")
        self.commit = self._git(staging, "rev-parse", "HEAD").stdout.strip()
        self.source = self.state / "sources" / self.commit
        staging.rename(self.source)
        self.job = self.state / "jobs" / "fixture-gate-test"
        self.job.mkdir()
        descriptor = {
            "schema_version": "wmf-cluster-job-v1",
            "namespace": fixture_job.NAMESPACE,
            "job_id": self.job.name,
            "released": True,
            "source_commit": self.commit,
            "role": self.worker_role if dynamic_worker else "any",
            "argv": ["python3", "fixture_job.py"],
            "max_wall_seconds": 1000,
            "publish_log_tail_bytes": 0,
        }
        (self.job / "descriptor.json").write_text(json.dumps(descriptor, indent=2, sort_keys=True) + "\n")
        self.nvidia_smi = self.root / "nvidia-smi"
        self.nvidia_smi.write_text(FAKE_NVIDIA_SMI)
        self.nvidia_smi.chmod(self.nvidia_smi.stat().st_mode | stat.S_IXUSR)
        self.counter = self.root / "child-count"
        self.robolab = self.root / "RoboLab"
        self.robolab.mkdir()

    def create_worker_diagnostic(self) -> tuple[Path, str]:
        diagnostic_job = self.state / "jobs/worker-diagnostic-test"
        publish = diagnostic_job / "publish"
        publish.mkdir(parents=True)
        descriptor = {
            "schema_version": "wmf-cluster-job-v1",
            "namespace": fixture_job.NAMESPACE,
            "job_id": diagnostic_job.name,
            "released": True,
            "source_commit": self.commit,
            "role": self.worker_role,
            "argv": [
                "/usr/bin/python3",
                fixture_job.WORKER_DIAGNOSTIC_SCRIPT,
                "--job-dir",
                "{job_dir}",
                "--source-root",
                "{source_root}",
                "--expect-gpus",
                "1",
                "--expected-worker-id",
                self.worker_role,
            ],
            "max_wall_seconds": 120,
            "publish_log_tail_bytes": 2048,
        }
        (diagnostic_job / "descriptor.json").write_bytes(fixture_job.canonical_bytes(descriptor))
        report = {
            "schema_version": fixture_job.WORKER_DIAGNOSTIC_SCHEMA,
            "namespace": fixture_job.NAMESPACE,
            "observed_at_utc": "2026-09-13T04:30:00+00:00",
            "hostname": self.hostname,
            "pod_uid": self.pod_uid,
            "expected_worker_id": self.worker_role,
            "worker_identity_errors": [],
            "expected_gpu_count": 1,
            "idle_worker_checks_applied": True,
            "idle_worker_errors": [],
            "diagnostic_passed": True,
            "job_dir": str(diagnostic_job.resolve()),
            "job_dir_writable": True,
            "source_root": str(self.source.resolve()),
            "source_commit": self.commit,
            "gpu": {
                "available": True,
                "count": 1,
                "devices": [
                    {
                        "index": 0,
                        "uuid": "GPU-test-fixture",
                        "name": "NVIDIA B200",
                        "driver_version": "580.95.05",
                        "memory.total": 183359,
                        "memory.free": 182632,
                        "utilization.gpu": 0,
                    }
                ],
                "compute_processes": [],
                "errors": [],
            },
            "scientific_qualification": False,
            "cross_pod_lock_qualification": False,
            "scope": fixture_job.WORKER_DIAGNOSTIC_SCOPE,
        }
        path = publish / "diagnostic.json"
        path.write_bytes(fixture_job.canonical_bytes(report))
        return path.resolve(), fixture_job.file_identity(path)["sha256"]

    @staticmethod
    def _git(root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(root), *argv],
            check=True,
            capture_output=True,
            text=True,
        )

    def execute(
        self,
        *,
        candidate: str = "P00__candidate_00",
        mode: str = "accepted",
        normalize_exit: bool = False,
        worker_diagnostic_path: Path | None = None,
        worker_diagnostic_sha256: str | None = None,
    ) -> dict:
        real_verify = fixture_job.verify_clean_git

        def verify(root: Path, expected: str, label: str) -> None:
            if label == "robolab":
                self.assert_robolab_arguments(root, expected)
                return
            real_verify(root, expected, label)

        with (
            patch.object(fixture_job, "verify_clean_git", side_effect=verify),
            patch.dict(
                os.environ,
                {
                    "FAKE_GATE_MODE": mode,
                    "FAKE_GATE_COUNTER": str(self.counter),
                    "FAKE_GATE_NORMALIZE_EXIT": "1" if normalize_exit else "0",
                },
                clear=False,
            ),
        ):
            return fixture_job.execute_job(
                source_root=self.source,
                state_dir=self.state,
                job_dir=self.job,
                layout_pair_id="P00",
                requested_candidate=candidate,
                robolab_root=self.robolab,
                robolab_python=Path(sys.executable),
                nvidia_smi=self.nvidia_smi,
                hostname=self.hostname,
                pod_uid=self.pod_uid,
                worker_diagnostic_path=worker_diagnostic_path,
                worker_diagnostic_sha256=worker_diagnostic_sha256,
            )

    def assert_robolab_arguments(self, root: Path, expected: str) -> None:
        if root != self.robolab.resolve() or expected != fixture_job.ROBOLAB_COMMIT:
            raise AssertionError("production RoboLab identity was not passed to validation")


class FixtureJobTests(unittest.TestCase):
    def test_child_python_path_preserves_venv_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            venv_python = Path(directory) / "venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(Path(sys.executable).resolve())
            self.assertEqual(fixture_job._lexical_absolute(venv_python), venv_python.absolute())
            self.assertNotEqual(fixture_job._lexical_absolute(venv_python), venv_python.resolve())

    def test_success_publishes_only_bounded_identities_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            first = harness.execute()
            receipt_path = harness.job / "publish/fixture_gate_receipt.json"
            first_payload = receipt_path.read_bytes()
            second = harness.execute()
            self.assertEqual(first["decision"], "accepted")
            self.assertEqual(first["exit_code"], 0)
            self.assertTrue(second["idempotent_replay"])
            self.assertEqual(harness.counter.read_text(), "1")
            self.assertEqual(receipt_path.read_bytes(), first_payload)
            self.assertLess(len(first_payload), fixture_job.MAX_PUBLISH_BYTES)
            self.assertEqual(first["gpu_identity"]["name"], "NVIDIA B200")
            self.assertEqual(first["gpu_identity"]["preexisting_compute_process_count"], 0)
            self.assertEqual(first["deployment_identity"]["receipt_kind"], "initial_deployment")
            self.assertNotIn("failures", first)
            raw_attempt = Path(first["gate_evidence"]["raw_attempt_directory"])
            self.assertTrue(raw_attempt.is_relative_to(harness.private / "fixture_gates/P00/attempts"))
            cache_values = json.loads((raw_attempt / "gate_attempt_receipt.json").read_text())["cache_paths"]
            self.assertTrue(all(Path(value).is_relative_to(harness.private / "worker_runtime") for value in cache_values.values()))

    def test_dynamic_worker_diagnostic_authenticates_role_pod_source_and_idle_b200(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory, dynamic_worker=True)
            diagnostic_path, diagnostic_sha256 = harness.create_worker_diagnostic()
            receipt = harness.execute(
                worker_diagnostic_path=diagnostic_path,
                worker_diagnostic_sha256=diagnostic_sha256,
            )
            self.assertEqual(receipt["decision"], "accepted")
            identity = receipt["deployment_identity"]
            self.assertEqual(identity["receipt_kind"], "dynamic_worker_diagnostic")
            self.assertEqual(identity["receipt"]["sha256"], diagnostic_sha256)
            self.assertEqual(identity["diagnostic_job_id"], "worker-diagnostic-test")
            self.assertEqual(identity["pod"], harness.hostname)
            self.assertEqual(identity["pod_uid"], harness.pod_uid)
            self.assertEqual(identity["deployment_job"], harness.worker_role)
            self.assertFalse(identity["scientific_qualification"])
            self.assertEqual(identity["scope"], fixture_job.WORKER_DIAGNOSTIC_SCOPE)
            self.assertEqual(receipt["gpu_identity"]["preexisting_compute_process_count"], 0)
            self.assertEqual(harness.counter.read_text(), "1")

    def test_dynamic_worker_diagnostic_rejects_identity_gpu_and_scope_changes(self):
        cases = {
            "hostname": "worker_diagnostic_hostname_mismatch",
            "pod_uid": "worker_diagnostic_pod_uid_mismatch",
            "report_source": "worker_diagnostic_report_source_mismatch",
            "identity_errors": "worker_diagnostic_identity_errors",
            "gpu_errors": "worker_diagnostic_gpu_errors",
            "gpu_count": "worker_diagnostic_gpu_count_invalid",
            "gpu_index": "worker_diagnostic_gpu_index_invalid",
            "gpu_name": "worker_diagnostic_gpu_not_b200",
            "gpu_busy": "worker_diagnostic_gpu_not_idle",
            "compute_process": "worker_diagnostic_preexisting_compute_processes",
            "gpu_uuid": "worker_diagnostic_gpu_uuid_changed",
            "scientific_scope": "worker_diagnostic_scope_invalid",
        }
        for mutation, expected_reason in cases.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                harness = QueueHarness(directory, dynamic_worker=True)
                diagnostic_path, _ = harness.create_worker_diagnostic()
                report = json.loads(diagnostic_path.read_text())
                if mutation == "hostname":
                    report["hostname"] = "another-worker-testpod"
                elif mutation == "pod_uid":
                    report["pod_uid"] = "99999999-2222-3333-4444-555555555555"
                elif mutation == "report_source":
                    report["source_commit"] = "b" * 40
                elif mutation == "identity_errors":
                    report["worker_identity_errors"] = ["hostname_does_not_match_expected_worker"]
                elif mutation == "gpu_errors":
                    report["gpu"]["errors"] = [{"query": "devices", "reason": "nonzero_exit"}]
                elif mutation == "gpu_count":
                    report["gpu"]["count"] = 2
                elif mutation == "gpu_index":
                    report["gpu"]["devices"][0]["index"] = 1
                elif mutation == "gpu_name":
                    report["gpu"]["devices"][0]["name"] = "NVIDIA H100"
                elif mutation == "gpu_busy":
                    report["gpu"]["devices"][0]["utilization.gpu"] = 17
                elif mutation == "compute_process":
                    report["gpu"]["compute_processes"] = [
                        {
                            "gpu_uuid": "GPU-test-fixture",
                            "pid": 123,
                            "process_name": "unrelated",
                            "used_memory": 1024,
                        }
                    ]
                elif mutation == "gpu_uuid":
                    report["gpu"]["devices"][0]["uuid"] = "GPU-another-fixture"
                elif mutation == "scientific_scope":
                    report["scientific_qualification"] = True
                diagnostic_path.write_bytes(fixture_job.canonical_bytes(report))
                diagnostic_sha256 = fixture_job.file_identity(diagnostic_path)["sha256"]
                receipt = harness.execute(
                    worker_diagnostic_path=diagnostic_path,
                    worker_diagnostic_sha256=diagnostic_sha256,
                )
                self.assertEqual(receipt["decision"], "technical_invalid")
                self.assertEqual(receipt["reason"], expected_reason)
                self.assertFalse(receipt["child_started"])
                self.assertFalse(harness.counter.exists())

    def test_dynamic_worker_diagnostic_rejects_hash_path_and_descriptor_changes(self):
        cases = {
            "hash": "worker_diagnostic_hash_mismatch",
            "path": "worker_diagnostic_outside_queue_jobs",
            "descriptor_job": "worker_diagnostic_descriptor_job_mismatch",
            "descriptor_role": "worker_diagnostic_role_mismatch",
            "descriptor_source": "worker_diagnostic_source_mismatch",
            "descriptor_argv": "worker_diagnostic_descriptor_argv_mismatch",
        }
        for mutation, expected_reason in cases.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                harness = QueueHarness(directory, dynamic_worker=True)
                diagnostic_path, diagnostic_sha256 = harness.create_worker_diagnostic()
                descriptor_path = diagnostic_path.parent.parent / "descriptor.json"
                if mutation == "hash":
                    diagnostic_sha256 = "f" * 64
                elif mutation == "path":
                    outside_path = harness.root / "outside/publish/diagnostic.json"
                    outside_path.parent.mkdir(parents=True)
                    shutil.copy2(diagnostic_path, outside_path)
                    diagnostic_path = outside_path.resolve()
                    diagnostic_sha256 = fixture_job.file_identity(diagnostic_path)["sha256"]
                else:
                    descriptor = json.loads(descriptor_path.read_text())
                    if mutation == "descriptor_job":
                        descriptor["job_id"] = "another-diagnostic-job"
                    elif mutation == "descriptor_role":
                        descriptor["role"] = "fixture-worker-01"
                    elif mutation == "descriptor_source":
                        descriptor["source_commit"] = "b" * 40
                    elif mutation == "descriptor_argv":
                        descriptor["argv"][1] = "{source_root}/wrong.py"
                    descriptor_path.write_bytes(fixture_job.canonical_bytes(descriptor))
                receipt = harness.execute(
                    worker_diagnostic_path=diagnostic_path,
                    worker_diagnostic_sha256=diagnostic_sha256,
                )
                self.assertEqual(receipt["decision"], "technical_invalid")
                self.assertEqual(receipt["reason"], expected_reason)
                self.assertFalse(receipt["child_started"])
                self.assertFalse(harness.counter.exists())

    def test_physical_rejection_preserves_child_exit_two_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            receipt = harness.execute(mode="physical_rejection")
            self.assertEqual(receipt["decision"], "physical_rejection")
            self.assertEqual(receipt["exit_code"], 2)
            self.assertEqual(receipt["child_exit_code"], 2)
            self.assertEqual(receipt["gate_evidence"]["failure_count"], 1)
            self.assertTrue(Path(receipt["gate_evidence"]["gate_attempt_receipt"]["path"]).exists())

    def test_authoritative_rejection_survives_runtime_normalized_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            receipt = harness.execute(mode="physical_rejection", normalize_exit=True)
            self.assertEqual(receipt["decision"], "physical_rejection")
            self.assertEqual(receipt["exit_code"], 2)
            self.assertEqual(receipt["child_exit_code"], 0)
            self.assertEqual(
                receipt["reason"],
                "child_exit_normalized_after_authoritative_gate_record",
            )

    def test_unexpected_child_crash_is_published_as_technical_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            receipt = harness.execute(mode="crash")
            self.assertEqual(receipt["decision"], "technical_invalid")
            self.assertEqual(receipt["exit_code"], 3)
            self.assertEqual(receipt["child_exit_code"], 17)
            self.assertEqual(receipt["reason"], "child_unexpected_exit")
            self.assertEqual(receipt["gate_record_count"], 0)
            self.assertEqual(receipt["fixture_job_ledger"]["record_count"], 1)
            self.assertGreater(receipt["child_logs"]["stderr"]["bytes"], 0)

    def test_child_technical_invalid_exit_three_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            receipt = harness.execute(mode="technical_invalid")
            self.assertEqual(receipt["decision"], "technical_invalid")
            self.assertEqual(receipt["exit_code"], 3)
            self.assertEqual(receipt["child_exit_code"], 3)
            self.assertIsNone(receipt["reason"])

    def test_queue_path_confinement_fails_before_child_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            outside = harness.root / "outside-job"
            outside.mkdir()
            with self.assertRaisesRegex(fixture_job.FixtureJobError, "job_dir_outside_queue_state"):
                fixture_job.execute_job(
                    source_root=harness.source,
                    state_dir=harness.state,
                    job_dir=outside,
                    layout_pair_id="P00",
                    robolab_root=harness.robolab,
                    robolab_python=Path(sys.executable),
                    nvidia_smi=harness.nvidia_smi,
                    hostname=harness.hostname,
                    pod_uid=harness.pod_uid,
                )
            self.assertFalse(harness.counter.exists())
            self.assertFalse((outside / "publish").exists())

    def test_busy_gpu_fails_closed_without_spawning_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = QueueHarness(directory)
            real_verify = fixture_job.verify_clean_git

            def verify(root: Path, expected: str, label: str) -> None:
                if label != "robolab":
                    real_verify(root, expected, label)

            with (
                patch.object(fixture_job, "verify_clean_git", side_effect=verify),
                patch.dict(os.environ, {"FAKE_GPU_BUSY": "1", "FAKE_GATE_COUNTER": str(harness.counter)}),
            ):
                receipt = fixture_job.execute_job(
                    source_root=harness.source,
                    state_dir=harness.state,
                    job_dir=harness.job,
                    layout_pair_id="P00",
                    requested_candidate="P00__candidate_00",
                    robolab_root=harness.robolab,
                    robolab_python=Path(sys.executable),
                    nvidia_smi=harness.nvidia_smi,
                    hostname=harness.hostname,
                    pod_uid=harness.pod_uid,
                )
            self.assertEqual(receipt["decision"], "technical_invalid")
            self.assertEqual(receipt["reason"], "preexisting_compute_processes")
            self.assertFalse(receipt["child_started"])
            self.assertFalse(harness.counter.exists())


if __name__ == "__main__":
    unittest.main()
