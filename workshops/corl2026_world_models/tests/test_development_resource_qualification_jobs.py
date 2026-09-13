from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LAYOUT = ROOT / "experiments/forecast_layout"


def load_module(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


sampler = load_module("wmf_test_resource_gpu_sampler", LAYOUT / "resource_gpu_sampler.py")
jobs = load_module(
    "wmf_test_development_resource_qualification_jobs",
    LAYOUT / "development_resource_qualification_jobs.py",
)


def write_json(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return sampler.sha256_file(path)


def audit_receipt() -> dict:
    return jobs.queue.signed_document({
        "schema_version": jobs.RESOURCE_AUDIT_SCHEMA,
        "job_id": jobs.RESOURCE_AUDIT_JOB_ID,
        "status": "passed_with_declared_missingness",
        "decision": "no_go_resource_gate_incomplete",
        "safe_to_release_confirmation": False,
        "missing_release_requirements": [
            "N3 behavioral model-server and simulator GPU peaks",
            "D1 simulator and simultaneous all-process GPU peaks",
            "two-rater annotation and adjudication time",
            "measured confirmation execution topology, GPU-memory envelope, and safe concurrency freeze",
        ],
    })


def science(mode: str) -> dict:
    result = {
        "model_runtime_loads": 1 if mode != "d1-simulator-profile" else 0,
        "model_servers_started": 1 if mode == "d1-model-profile" else 0,
        "model_requests_issued": 6 if mode != "d1-simulator-profile" else 0,
        "model_requests_completed": 6 if mode != "d1-simulator-profile" else 0,
        "simulator_processes_started": 1 if mode != "d1-model-profile" else 0,
        "physical_resets": 1 if mode != "d1-model-profile" else 0,
        "settling_hold_actions": 96 if mode != "d1-model-profile" else 0,
    }
    result.update(jobs._zero_behavior_counts())
    return result


def probe_receipt(mode: str, job_id: str, commit: str) -> dict:
    return sampler.signed_document({
        "schema_version": jobs.JOB_RECEIPT_SCHEMA,
        "status": "passed",
        "decision": "machine_measurement_passed_confirmation_held",
        "namespace": jobs.NAMESPACE,
        "study_id": jobs.STUDY_ID,
        "mode": mode,
        "job_id": job_id,
        "study_commit": commit,
        "science_counts": science(mode),
        "machine_resource_gate_complete": False,
        "safe_to_release_confirmation": False,
        "confirmation_released": False,
    })


class ResourceDescriptorTests(unittest.TestCase):
    def test_contract_is_serial_and_never_substitutes_annotation_time(self) -> None:
        contract = jobs._contract(jobs.REPOSITORY_ROOT)
        self.assertEqual(contract["topology_freeze_policy"]["global_max_parallel_blocks"], 1)
        self.assertFalse(contract["topology_freeze_policy"]["cross_model_simultaneous_blocks_allowed"])
        self.assertFalse(contract["release_boundary"]["annotation_time_may_be_synthesized"])
        self.assertFalse(contract["release_boundary"]["safe_to_release_confirmation"])
        self.assertEqual(contract["probe_scope"]["new_nonbehavioral_generation_requests_total"], 12)
        self.assertEqual(contract["probe_scope"]["retained_behavioral_cells_rerun"], 0)
        self.assertEqual(contract["runtime"]["peer_failure_poll_interval_seconds"], 0.25)

    def test_all_pinned_dependencies_match(self) -> None:
        contract = jobs._contract(jobs.REPOSITORY_ROOT)
        observed = jobs._validate_dependencies(jobs.REPOSITORY_ROOT, contract)
        self.assertEqual(set(observed), set(contract["pinned_workshop_dependencies"]))

    def test_probe_wave_is_exactly_three_detached_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            digest = write_json(path, audit_receipt())
            wave = jobs.build_probe_wave(
                study_commit="a" * 40,
                resource_audit_receipt=path,
                resource_audit_receipt_sha256=digest,
            )
        self.assertEqual(wave["schema_version"], jobs.WAVE_SCHEMA)
        self.assertEqual(wave["new_nonbehavioral_generation_requests_if_all_jobs_run"], 12)
        self.assertEqual(wave["behavioral_cells"], 0)
        self.assertEqual([row["job_id"] for row in wave["jobs"]], [
            jobs.N3_JOB_ID, jobs.D1_MODEL_JOB_ID, jobs.D1_SIM_JOB_ID,
        ])
        self.assertEqual([row["role"] for row in wave["jobs"]], [
            jobs.N3_ROLE, jobs.D1_MODEL_ROLE, jobs.D1_SIM_ROLE,
        ])
        self.assertTrue(all(row["publish_log_tail_bytes"] == 0 for row in wave["jobs"]))
        self.assertTrue(all(row["released"] is True for row in wave["jobs"]))
        self.assertFalse(wave["safe_to_release_confirmation"])

    def test_probe_wave_rejects_resource_audit_missing_annotation_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audit.json"
            value = audit_receipt()
            value["missing_release_requirements"] = ["gpu"]
            value.pop("payload_sha256")
            value = jobs.queue.signed_document(value)
            digest = write_json(path, value)
            with self.assertRaisesRegex(jobs.ResourceQualificationError, "annotation"):
                jobs.build_probe_wave(
                    study_commit="a" * 40,
                    resource_audit_receipt=path,
                    resource_audit_receipt_sha256=digest,
                )

    def test_finalize_wave_is_receipt_gated_and_issues_no_request(self) -> None:
        commit = "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = {
                "n3": ("n3-profile", jobs.N3_JOB_ID),
                "d1-model": ("d1-model-profile", jobs.D1_MODEL_JOB_ID),
                "d1-simulator": ("d1-simulator-profile", jobs.D1_SIM_JOB_ID),
            }
            paths = {}
            hashes = {}
            for name, (mode, job_id) in specs.items():
                paths[name] = root / f"{name}.json"
                hashes[name] = write_json(paths[name], probe_receipt(mode, job_id, commit))
            wave = jobs.build_finalize_wave(
                study_commit=commit,
                n3_receipt=paths["n3"], n3_receipt_sha256=hashes["n3"],
                d1_model_receipt=paths["d1-model"],
                d1_model_receipt_sha256=hashes["d1-model"],
                d1_simulator_receipt=paths["d1-simulator"],
                d1_simulator_receipt_sha256=hashes["d1-simulator"],
            )
        self.assertEqual(len(wave["jobs"]), 1)
        self.assertEqual(wave["jobs"][0]["job_id"], jobs.FINALIZE_JOB_ID)
        self.assertEqual(wave["new_model_requests_issued_by_finalize"], 0)
        self.assertEqual(wave["robot_episodes"], 0)
        self.assertFalse(wave["safe_to_release_confirmation"])

    def test_d1_prelaunch_failures_release_the_peer_with_signed_markers(self) -> None:
        commit = "c" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode, job_id, expected_files in (
                (
                    "d1-model-profile",
                    jobs.D1_MODEL_JOB_ID,
                    ("model_started.json", "model_done.json"),
                ),
                (
                    "d1-simulator-profile",
                    jobs.D1_SIM_JOB_ID,
                    ("simulator_failed.json",),
                ),
            ):
                with self.subTest(mode=mode):
                    session = root / mode
                    job_dir = root / f"job-{mode}"
                    (job_dir / "raw").mkdir(parents=True)
                    args = SimpleNamespace(
                        command=mode,
                        job_id=job_id,
                        study_commit=commit,
                        job_dir=job_dir,
                    )
                    with mock.patch.object(jobs, "_session_root", return_value=session):
                        jobs._signal_d1_peer_failure(
                            args=args,
                            context=None,
                            error=jobs.ResourceQualificationError("prelaunch failure"),
                        )
                    for name in expected_files:
                        value = json.loads((session / name).read_text(encoding="utf-8"))
                        sampler.verify_signed_document(value, name)
                        self.assertEqual(value["status"], "technical_invalid")
                        self.assertEqual(value["study_commit"], commit)
            with self.assertRaisesRegex(
                jobs.ResourceQualificationError, "simulator peer became technical-invalid"
            ):
                jobs._wait_for_d1_simulator_ready(
                    root / "d1-simulator-profile/simulator_ready.json",
                    root / "d1-simulator-profile/simulator_failed.json",
                    study_commit=commit,
                    timeout_seconds=0.1,
                )

    def test_precreated_logs_never_count_as_successful_child_launch(self) -> None:
        commit = "d" * 40
        cases = (
            (jobs.PROBE_BY_MODE["n3-profile"], "n3.stdout.log", "model_runtime_loads_started"),
            (jobs.PROBE_BY_MODE["n3-profile"], "simulator.stdout.log", "simulator_processes_started"),
            (jobs.PROBE_BY_MODE["d1-model-profile"], "server.stdout.log", "model_servers_started"),
            (jobs.PROBE_BY_MODE["d1-simulator-profile"], "simulator.stdout.log", "simulator_processes_started"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (job, log_name, count_name) in enumerate(cases):
                with self.subTest(mode=job.mode, log=log_name):
                    job_dir = root / f"job-{index}"
                    raw = job_dir / "raw"
                    raw.mkdir(parents=True)
                    (raw / log_name).write_bytes(b"")
                    counts = jobs._failure_science_counts(
                        job, job_dir, study_commit=commit
                    )
                    self.assertEqual(counts[count_name], 0)
                    self.assertTrue(
                        all(
                            row["status"] == "not_launched"
                            for row in counts["launch_evidence"].values()
                        )
                    )

            popen_job_dir = root / "popen-failure"
            popen_raw = popen_job_dir / "raw"
            popen_raw.mkdir(parents=True)
            with mock.patch.object(jobs.subprocess, "Popen", side_effect=OSError("exec failed")):
                with self.assertRaises(OSError):
                    jobs._launch_child(
                        [sys.executable, "-c", "pass"],
                        cwd=root,
                        env=os.environ,
                        stdout_path=popen_raw / "n3.stdout.log",
                        stderr_path=popen_raw / "n3.stderr.log",
                        launch_receipt_path=popen_raw / "n3_model.launch.json",
                        launch_kind="n3_model",
                        study_commit=commit,
                        job_id=jobs.N3_JOB_ID,
                        termination_grace_seconds=0.2,
                    )
            self.assertTrue((popen_raw / "n3.stdout.log").is_file())
            self.assertFalse((popen_raw / "n3_model.launch.json").exists())
            attempt = json.loads(
                (popen_raw / "n3_model.launch.attempt.json").read_text()
            )
            outcome = json.loads(
                (popen_raw / "n3_model.launch.outcome.json").read_text()
            )
            sampler.verify_signed_document(attempt, "failed launch attempt")
            sampler.verify_signed_document(outcome, "failed launch outcome")
            self.assertFalse(outcome["process_started"])
            self.assertEqual(outcome["status"], "popen_failed")
            counts = jobs._failure_science_counts(
                jobs.PROBE_BY_MODE["n3-profile"],
                popen_job_dir,
                study_commit=commit,
            )
            self.assertEqual(counts["model_runtime_loads_started"], 0)
            self.assertEqual(
                counts["launch_evidence"]["model"]["status"],
                "authenticated_popen_failed",
            )

    def test_successful_popen_writes_authenticated_launch_before_counting(self) -> None:
        commit = "e" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            process, stdout, stderr, launch = jobs._launch_child(
                [sys.executable, "-c", "import time; time.sleep(0.2)"],
                cwd=root,
                env=os.environ,
                stdout_path=raw / "n3.stdout.log",
                stderr_path=raw / "n3.stderr.log",
                launch_receipt_path=raw / "n3_model.launch.json",
                launch_kind="n3_model",
                study_commit=commit,
                job_id=jobs.N3_JOB_ID,
                termination_grace_seconds=0.2,
            )
            process.wait(timeout=2)
            jobs._finish_child_handles(stdout, stderr)
            sampler.verify_signed_document(launch, "launch")
            counts = jobs._failure_science_counts(
                jobs.PROBE_BY_MODE["n3-profile"], root, study_commit=commit
            )
            self.assertEqual(counts["model_runtime_loads_started"], 1)
            self.assertEqual(
                counts["launch_evidence"]["model"]["status"],
                "authenticated_launched",
            )

    def test_successful_popen_still_counts_if_final_launch_receipt_write_fails(self) -> None:
        commit = "4" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            raw.mkdir()
            with mock.patch.object(
                jobs,
                "_record_child_launch",
                side_effect=OSError("synthetic final-receipt failure"),
            ):
                with self.assertRaisesRegex(
                    jobs.ResourceQualificationError, "durable launch receipt"
                ):
                    jobs._launch_child(
                        [sys.executable, "-c", "import time; time.sleep(30)"],
                        cwd=root,
                        env=os.environ,
                        stdout_path=raw / "n3.stdout.log",
                        stderr_path=raw / "n3.stderr.log",
                        launch_receipt_path=raw / "n3_model.launch.json",
                        launch_kind="n3_model",
                        study_commit=commit,
                        job_id=jobs.N3_JOB_ID,
                        termination_grace_seconds=0.5,
                    )
            self.assertFalse((raw / "n3_model.launch.json").exists())
            outcome = json.loads(
                (raw / "n3_model.launch.outcome.json").read_text()
            )
            sampler.verify_signed_document(outcome, "successful Popen outcome")
            self.assertTrue(outcome["process_started"])
            counts = jobs._failure_science_counts(
                jobs.PROBE_BY_MODE["n3-profile"], root, study_commit=commit
            )
            self.assertEqual(counts["model_runtime_loads_started"], 1)
            self.assertEqual(
                counts["launch_evidence"]["model"]["status"],
                "authenticated_popen_succeeded_final_receipt_missing",
            )

    def test_post_reset_failure_journal_is_counted_without_ready_receipt(self) -> None:
        commit = "f" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "raw/simulator_hold"
            output.mkdir(parents=True)
            journal = jobs._SimulatorAccountingJournal(
                output / "simulator_accounting.jsonl",
                study_commit=commit,
                model_job_id=jobs.D1_MODEL_JOB_ID,
                simulator_job_id=jobs.D1_SIM_JOB_ID,
            )
            journal.append("simulator_child_entered", pid=os.getpid())
            journal.append("physical_reset_started")
            journal.append("physical_reset_completed")
            summary = jobs._summarize_simulator_accounting(
                output / "simulator_accounting.jsonl",
                study_commit=commit,
                model_job_id=jobs.D1_MODEL_JOB_ID,
                simulator_job_id=jobs.D1_SIM_JOB_ID,
            )
            self.assertEqual(summary["physical_resets"], 1)
            self.assertEqual(summary["physical_resets_lower_bound"], 1)
            self.assertEqual(summary["physical_resets_upper_bound"], 1)
            self.assertEqual(summary["settling_hold_actions"], 0)

            counts = jobs._failure_science_counts(
                jobs.PROBE_BY_MODE["d1-simulator-profile"],
                root,
                study_commit=commit,
            )
            self.assertEqual(counts["simulator_processes_started"], 1)
            self.assertEqual(counts["physical_resets"], 1)
            self.assertEqual(
                counts["launch_evidence"]["simulator"]["status"],
                "authenticated_launched_from_inner_child_entry",
            )

            args = SimpleNamespace(
                output_dir=output,
                study_commit=commit,
                model_job_id=jobs.D1_MODEL_JOB_ID,
                simulator_job_id=jobs.D1_SIM_JOB_ID,
            )

            def fail_after_reset(_args):
                raise jobs.ResourceQualificationError("post-reset failure")

            with mock.patch.object(jobs, "run_internal_simulator_hold", fail_after_reset):
                self.assertEqual(jobs._internal_simulator_main(args), 3)
            failure = json.loads((output / "technical_failure.json").read_text())
            sampler.verify_signed_document(failure, "simulator failure")
            self.assertEqual(failure["physical_reset_count"], 1)
            self.assertEqual(failure["physical_reset_count_bounds"], {"lower": 1, "upper": 1})
            self.assertFalse((output / "simulator_ready.json").exists())

    def test_inflight_reset_or_hold_uses_null_and_tight_bounds(self) -> None:
        commit = "1" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reset = jobs._SimulatorAccountingJournal(
                root / "reset.jsonl",
                study_commit=commit,
                model_job_id=jobs.N3_JOB_ID,
                simulator_job_id=jobs.N3_JOB_ID,
            )
            reset.append("simulator_child_entered", pid=os.getpid())
            reset.append("physical_reset_started")
            summary = jobs._summarize_simulator_accounting(
                root / "reset.jsonl",
                study_commit=commit,
                model_job_id=jobs.N3_JOB_ID,
                simulator_job_id=jobs.N3_JOB_ID,
            )
            self.assertIsNone(summary["physical_resets"])
            self.assertEqual(
                (summary["physical_resets_lower_bound"], summary["physical_resets_upper_bound"]),
                (0, 1),
            )

            hold = jobs._SimulatorAccountingJournal(
                root / "hold.jsonl",
                study_commit=commit,
                model_job_id=jobs.N3_JOB_ID,
                simulator_job_id=jobs.N3_JOB_ID,
            )
            hold.append("simulator_child_entered", pid=os.getpid())
            hold.append("physical_reset_started")
            hold.append("physical_reset_completed")
            hold.append("settling_hold_started", hold_ordinal=0, settling_phase="settle")
            summary = jobs._summarize_simulator_accounting(
                root / "hold.jsonl",
                study_commit=commit,
                model_job_id=jobs.N3_JOB_ID,
                simulator_job_id=jobs.N3_JOB_ID,
            )
            self.assertIsNone(summary["settling_hold_actions"])
            self.assertEqual(
                (
                    summary["settling_hold_actions_lower_bound"],
                    summary["settling_hold_actions_upper_bound"],
                ),
                (0, 1),
            )

    def test_d1_probe_is_promptly_terminated_and_reaped_on_peer_failure(self) -> None:
        commit = "2" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            peer = root / "simulator_failed.json"

            def publish_failure() -> None:
                time.sleep(0.1)
                jobs._immutable_signed(peer, {
                    "schema_version": jobs.SHARED_SCHEMA,
                    "kind": "simulator_failed",
                    "study_id": jobs.STUDY_ID,
                    "study_commit": commit,
                    "session_id": jobs.D1_SESSION_ID,
                    "model_job_id": jobs.D1_MODEL_JOB_ID,
                    "simulator_job_id": jobs.D1_SIM_JOB_ID,
                    "status": "technical_invalid",
                    "error_type": "SyntheticPeerFailure",
                })

            thread = threading.Thread(target=publish_failure)
            thread.start()
            started = time.monotonic()
            with self.assertRaisesRegex(
                jobs.ResourceQualificationError, "simulator_peer_failure; child reaped"
            ):
                jobs._run_peer_monitored_child(
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                    stdout_path=root / "probe.stdout.log",
                    stderr_path=root / "probe.stderr.log",
                    cwd=root,
                    env=os.environ,
                    timeout_seconds=10,
                    poll_interval_seconds=0.02,
                    peer_failure_path=peer,
                    study_commit=commit,
                    job_id=jobs.D1_MODEL_JOB_ID,
                    launch_receipt_path=root / "probe.launch.json",
                    termination_receipt_path=root / "probe.termination.json",
                    termination_grace_seconds=0.5,
                )
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertLess(time.monotonic() - started, 2)
            termination = json.loads((root / "probe.termination.json").read_text())
            sampler.verify_signed_document(termination, "termination")
            self.assertEqual(termination["reason"], "simulator_peer_failure")
            self.assertTrue(termination["termination"]["reaped"])

    def test_d1_monitored_probe_success_and_partial_request_accounting(self) -> None:
        commit = "3" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result, launch = jobs._run_peer_monitored_child(
                [sys.executable, "-c", "pass"],
                stdout_path=root / "probe.stdout.log",
                stderr_path=root / "probe.stderr.log",
                cwd=root,
                env=os.environ,
                timeout_seconds=2,
                poll_interval_seconds=0.02,
                peer_failure_path=root / "absent-peer-failure.json",
                study_commit=commit,
                job_id=jobs.D1_MODEL_JOB_ID,
                launch_receipt_path=root / "probe.launch.json",
                termination_receipt_path=root / "probe.termination.json",
                termination_grace_seconds=0.5,
            )
            self.assertEqual(result.returncode, 0)
            sampler.verify_signed_document(launch, "successful probe launch")
            self.assertFalse((root / "probe.termination.json").exists())

            episodes = root / "episodes"
            complete = episodes / "probe-a/request_0000"
            incomplete = episodes / "probe-b/request_0000"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            write_json(complete / "request_receipt.json", {
                "schema_version": "wmf-d1-request-receipt-v1"
            })
            issued, completed, status = jobs._d1_partial_request_counts(episodes)
            self.assertEqual((issued, completed), (2, 1))
            self.assertEqual(status, "validated_server_artifacts")

    def test_d1_peer_failure_marker_retains_exact_partial_request_counts(self) -> None:
        commit = "5" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job_dir = root / "job"
            complete = job_dir / "raw/d1_future/episodes/probe-a/request_0000"
            incomplete = job_dir / "raw/d1_future/episodes/probe-b/request_0000"
            complete.mkdir(parents=True)
            incomplete.mkdir(parents=True)
            write_json(complete / "request_receipt.json", {
                "schema_version": "wmf-d1-request-receipt-v1"
            })
            args = SimpleNamespace(
                command="d1-model-profile",
                job_id=jobs.D1_MODEL_JOB_ID,
                study_commit=commit,
                job_dir=job_dir,
            )
            session = root / "session"
            with mock.patch.object(jobs, "_session_root", return_value=session):
                jobs._signal_d1_peer_failure(
                    args=args,
                    context=None,
                    error=jobs.ResourceQualificationError("synthetic peer stop"),
                )
            done = json.loads((session / "model_done.json").read_text())
            sampler.verify_signed_document(done, "D1 partial model-done")
            self.assertEqual(done["generation_requests_issued"], 2)
            self.assertEqual(done["generation_requests_completed"], 1)
            self.assertEqual(
                done["request_count_evidence_status"],
                "validated_server_artifacts",
            )


class GpuSamplerTests(unittest.TestCase):
    def make_journal(self, path: Path, *, unknown_owner: bool = False) -> None:
        journal = sampler._SampleJournal(path)
        for sequence, wall in enumerate((1_000_000_000, 1_500_000_000, 2_000_000_000)):
            journal.append({
                "phase": "model_inference",
                "task_process_roots": {"model": 123, "simulator": 456},
                "wall_started_ns": wall,
                "wall_finished_ns": wall + 10_000_000,
                "monotonic_started_ns": wall,
                "monotonic_finished_ns": wall + 10_000_000,
                "devices": [
                    {
                        "index": 0, "uuid": "GPU-aaaaaaaa", "name": "NVIDIA B200",
                        "driver_version": "1", "memory_total_mib": 1000,
                        "memory_used_mib": 100 + sequence, "utilization_gpu_percent": 10,
                    },
                    {
                        "index": 1, "uuid": "GPU-bbbbbbbb", "name": "NVIDIA B200",
                        "driver_version": "1", "memory_total_mib": 1000,
                        "memory_used_mib": 200 + sequence, "utilization_gpu_percent": 20,
                    },
                ],
                "compute_processes": [
                    {
                        "gpu_uuid": "GPU-aaaaaaaa", "pid": 123,
                        "process_name": "model", "used_gpu_memory_mib": 90 + sequence,
                        "task_process_owner": None if unknown_owner else "model",
                    },
                    {
                        "gpu_uuid": "GPU-bbbbbbbb", "pid": 456,
                        "process_name": "sim", "used_gpu_memory_mib": 180 + sequence,
                        "task_process_owner": "simulator",
                    },
                ],
            })

    def test_summary_keeps_device_process_and_owner_peaks_distinct(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            self.make_journal(path)
            summary = sampler.summarize_samples(
                path,
                expected_gpu_count=2,
                required_gpu_name="NVIDIA B200",
                maximum_gap_seconds=1.0,
                intervals={"model_inference": (900_000_000, 2_100_000_000)},
                minimum_samples_by_interval={"model_inference": 2},
            )
        sampler.verify_signed_document(summary, "summary")
        interval = summary["intervals"]["model_inference"]
        self.assertEqual(interval["sample_count"], 3)
        self.assertEqual(interval["devices"][0]["whole_device_peak_bytes"], 102 * sampler.MIB)
        self.assertEqual(
            interval["devices"][0]["task_owner_process_peak_bytes"]["model"],
            92 * sampler.MIB,
        )
        self.assertEqual(interval["co_sampled_whole_device_peak_bytes"], 304 * sampler.MIB)

    def test_unknown_compute_process_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            self.make_journal(path, unknown_owner=True)
            with self.assertRaisesRegex(sampler.GpuSampleError, "unowned"):
                sampler.summarize_samples(
                    path,
                    expected_gpu_count=2,
                    required_gpu_name="NVIDIA B200",
                    maximum_gap_seconds=1.0,
                    intervals={"all": (900_000_000, 2_100_000_000)},
                )

    def test_sample_chain_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "samples.jsonl"
            self.make_journal(path)
            payload = path.read_bytes().replace(b'"memory_used_mib":100', b'"memory_used_mib":999', 1)
            path.write_bytes(payload)
            with self.assertRaisesRegex(sampler.GpuSampleError, "signature"):
                sampler.load_samples(path)

    def test_current_pid_classifies_to_exact_root(self) -> None:
        self.assertEqual(sampler.classify_pid(os.getpid(), {"test": os.getpid()}), "test")

    def test_owner_topology_requires_distinct_n3_devices(self) -> None:
        n3 = [
            {"gpu_uuid": "a", "task_owner_process_peak_bytes": {"model": 1}},
            {"gpu_uuid": "b", "task_owner_process_peak_bytes": {"simulator": 1}},
        ]
        d1_model = [
            {"gpu_uuid": "c", "task_owner_process_peak_bytes": {"model": 1}},
            {"gpu_uuid": "d", "task_owner_process_peak_bytes": {"model": 1}},
        ]
        d1_sim = [{"gpu_uuid": "e", "task_owner_process_peak_bytes": {"simulator": 1}}]
        jobs._validate_owner_topology(n3=n3, d1_model=d1_model, d1_sim=d1_sim)
        n3[1]["gpu_uuid"] = "a"
        with self.assertRaisesRegex(jobs.ResourceQualificationError, "same GPU"):
            jobs._validate_owner_topology(n3=n3, d1_model=d1_model, d1_sim=d1_sim)


if __name__ == "__main__":
    unittest.main()
