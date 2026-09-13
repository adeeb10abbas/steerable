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
    def __init__(self, temporary: str) -> None:
        self.root = Path(temporary)
        self.private = self.root / "private"
        self.state = self.private / "queue-state"
        (self.state / "sources").mkdir(parents=True)
        (self.state / "jobs").mkdir()
        staging = self.root / "staging-source"
        forecast = staging / "workshops/corl2026_world_models/experiments/forecast_layout"
        autonomy = staging / "workshops/corl2026_world_models/execution/20260912/autonomy"
        forecast.mkdir(parents=True)
        autonomy.mkdir(parents=True)
        shutil.copy2(FORECAST / "layout_source_contract.json", forecast / "layout_source_contract.json")
        shutil.copy2(FORECAST / "layout_candidate_pool.json", forecast / "layout_candidate_pool.json")
        (forecast / "model_blind_fixture_gate.py").write_text(FAKE_GATE)
        self.hostname = "fixture-worker-00"
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
            "role": "any",
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
            self.assertNotIn("failures", first)
            raw_attempt = Path(first["gate_evidence"]["raw_attempt_directory"])
            self.assertTrue(raw_attempt.is_relative_to(harness.private / "fixture_gates/P00/attempts"))
            cache_values = json.loads((raw_attempt / "gate_attempt_receipt.json").read_text())["cache_paths"]
            self.assertTrue(all(Path(value).is_relative_to(harness.private / "worker_runtime") for value in cache_values.values()))

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
