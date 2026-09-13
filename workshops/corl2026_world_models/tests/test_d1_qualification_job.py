from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np


WORKSHOP = Path(__file__).resolve().parents[1]
LAYOUT = WORKSHOP / "experiments/forecast_layout"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JOB = load_module("wmf_d1_qualification_job", LAYOUT / "d1_qualification_job.py")


def fixture_arrays() -> dict[str, np.ndarray]:
    image = np.arange(180 * 320 * 3, dtype=np.uint8).reshape(180, 320, 3)
    return {
        "observation/exterior_image_0_left": image,
        "observation/exterior_image_1_left": np.flip(image, axis=1).copy(),
        "observation/wrist_image_left": np.flip(image, axis=0).copy(),
        "observation/joint_position": np.arange(7, dtype=np.float64),
        "observation/cartesian_position": np.arange(6, dtype=np.float64),
        "observation/gripper_position": np.array([0.25], dtype=np.float64),
    }


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_capture_manifest(root: Path, fixture: Path) -> Path:
    def descriptor(path: Path, arrays: dict[str, np.ndarray] | None = None) -> dict:
        result = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": file_sha(path)}
        if arrays is not None:
            result["arrays"] = {
                key: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "order": "C",
                    "value_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
                }
                for key, value in arrays.items()
            }
        return result

    arrays = fixture_arrays()
    artifacts = {"D1": descriptor(fixture, arrays)}
    for name in ("raw_settled_observation", "simulator_state", "preprocessing_intermediates", "N3"):
        path = root / f"{name}.npz"
        np.savez(path, value=np.zeros(1, dtype=np.float32))
        artifacts[name] = descriptor(path)
    cameras = {
        name: {
            "frame_id": 75,
            "capture_time_ns": 5_000_000_000,
            "native_capture_time_source": "_timestamp",
        }
        for name in (
            "head_camera",
            "over_shoulder_left_camera",
            "over_shoulder_right_camera",
            "wrist_cam",
        )
    }
    gate_descriptor = {"path": "/raw/gate.json", "bytes": 123, "sha256": "d" * 64}
    pose_descriptor = {"path": "/raw/pose.json", "bytes": 456, "sha256": "e" * 64}
    capture = {
        "schema_version": JOB.CAPTURE_SCHEMA,
        "study_namespace": JOB.NAMESPACE,
        "status": "passed",
        "capture_id": "P00-test-capture",
        "layout_pair_id": "P00",
        "layout_arm": "original",
        "command_task_used_for_reset": "left",
        "candidate_id": "P00__candidate_00",
        "candidate_payload_sha256": "a" * 64,
        "accepted_gate_record_sha256": "b" * 64,
        "gate_receipt": gate_descriptor,
        "pose_manifest": pose_descriptor,
        "gate_ledger": {"path": "/raw/ledger.jsonl", "bytes": 789, "sha256": "f" * 64},
        "gate_attempt_receipt": {"path": "/raw/attempt.json", "bytes": 321, "sha256": "c" * 64},
        "settled_reset_identity": "fixed-observation:test:P00:1",
        "settled_reset_receipt": {
            "schema_version": JOB.SETTLED_RESET_SCHEMA,
            "passed": True,
            "settled": True,
            "left_success": False,
            "right_success": False,
            "settled_observation_returned": True,
            "model_request_count_during_settle": 0,
            "episode_length_buf_reset_to_zero": True,
            "reset_identity": "fixed-observation:test:P00:1",
            "pose_manifest_sha256": "e" * 64,
            "settle_evidence": {"settle_steps": 60, "stability_window_steps": 15},
            "collision_evidence": {"passed": True},
            "visibility_evidence": {"passed": True},
        },
        "fresh_physical_checks": {
            "collision": {"passed": True},
            "visibility": {"passed": True},
        },
        "native_clock": {
            "physics_step": 600,
            "control_step_since_physical_reset": 75,
            "behavioral_episode_step": 0,
            "camera_counters": cameras,
        },
        "source_capture": {
            "camera_frame_ids": {name: row["frame_id"] for name, row in cameras.items()},
            "camera_capture_time_ns": {name: row["capture_time_ns"] for name, row in cameras.items()},
            "camera_timestamp_source": {
                name: row["native_capture_time_source"] for name, row in cameras.items()
            },
        },
        "preprocessing": {
            "model_request_count": 0,
            "D1": {
                "extraction_method": "_extract_observation",
                "packing_method": "_pack_request",
                "wire_array_keys": list(JOB.RAW_ARRAY_KEYS),
                "configuration": {
                    "cam2_source": "right",
                    "resize": "pad",
                    "image_height": 180,
                    "image_width": 320,
                },
                "overlay_source": {"sha256": "1" * 64},
                "official_client_source": {"sha256": "2" * 64},
            },
        },
        "artifacts": artifacts,
        "model_fixtures": {
            "D1": {
                "fixture": {
                    key: artifacts["D1"][key] for key in ("path", "bytes", "sha256")
                },
                "array_keys": list(JOB.RAW_ARRAY_KEYS),
            }
        },
        "model_request_count": 0,
        "behavioral_action_count": 0,
    }
    path = root / "capture_receipt.json"
    path.write_text(json.dumps(capture, indent=2, sort_keys=True) + "\n")
    return path


def topology_receipt() -> dict:
    gpu_uuids = [
        "GPU-00000000-0000-0000-0000-000000000001",
        "GPU-00000000-0000-0000-0000-000000000002",
    ]
    devices = [
        {
            "index": str(index),
            "uuid": gpu_uuids[index],
            "name": "NVIDIA B200",
            "driver_version": "580.95.05",
            "memory.total": "183359",
        }
        for index in range(2)
    ]
    torch = {
        "schema_version": "wmf-d1-torch-topology-v1",
        "status": "passed",
        "device_count": 2,
        "devices": [
            {
                "logical_index": index,
                "uuid": gpu_uuids[index],
                "name": "NVIDIA B200",
                "total_memory_bytes": 192000000000,
            }
            for index in range(2)
        ],
    }
    return JOB.reconcile_topology(
        nvidia_devices=devices,
        compute_processes=[],
        torch_receipt=torch,
        visible_devices=",".join(gpu_uuids),
    )


class D1QualificationJobTests(unittest.TestCase):
    def test_fixture_is_snapshotted_hash_bound_and_exactly_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.npz"
            np.savez(source, **fixture_arrays())
            digest = file_sha(source)
            identity = JOB.snapshot_regular_file(source, root / "snapshot" / "fixture.npz", digest)
            receipt = JOB.validate_fixture_arrays(root / "snapshot" / "fixture.npz", digest)
            self.assertEqual(identity["source"]["sha256"], digest)
            self.assertEqual(identity["snapshot"]["sha256"], digest)
            self.assertEqual(receipt["array_count"], 6)
            self.assertEqual([row["key"] for row in receipt["arrays"]], list(JOB.RAW_ARRAY_KEYS))
            with self.assertRaisesRegex(JOB.D1JobError, "fixture_sha256_mismatch"):
                JOB.validate_fixture_arrays(root / "snapshot" / "fixture.npz", "0" * 64)

    def test_capture_manifest_binds_accepted_fixture_reset_timing_and_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.npz"
            np.savez(fixture, **fixture_arrays())
            manifest = write_capture_manifest(root, fixture)
            receipt = JOB.validate_capture_manifest(
                manifest,
                file_sha(manifest),
                artifact_root=root,
                fixture_path=fixture,
                fixture_sha256=file_sha(fixture),
            )
            self.assertEqual(receipt["candidate_id"], "P00__candidate_00")
            self.assertEqual(receipt["control_step_since_physical_reset"], 75)
            self.assertEqual(len(receipt["camera_capture_time_ns"]), 4)
            payload = json.loads(manifest.read_text())
            payload["fresh_physical_checks"]["collision"]["passed"] = False
            manifest.write_text(json.dumps(payload))
            with self.assertRaisesRegex(JOB.D1JobError, "fresh_collision_check_failed"):
                JOB.validate_capture_manifest(
                    manifest,
                    file_sha(manifest),
                    artifact_root=root,
                    fixture_path=fixture,
                    fixture_sha256=file_sha(fixture),
                )

    def test_capture_manifest_accepts_hash_bound_frame_identity_when_native_counter_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = root / "fixture.npz"
            np.savez(fixture, **fixture_arrays())
            manifest = write_capture_manifest(root, fixture)
            payload = json.loads(manifest.read_text())
            for name, row in payload["native_clock"]["camera_counters"].items():
                digest = hashlib.sha256(name.encode("ascii")).hexdigest()
                frame_id = f"rgb-sha256:{digest}"
                row.update(
                    frame_id=frame_id,
                    frame_identity_source="exact returned RGB array value identity",
                    native_frame_counter=None,
                    native_frame_counter_status="unavailable_in_pinned_isaaclab_sensorbase",
                    rgb_array_identity={"value_sha256": digest},
                )
                payload["source_capture"]["camera_frame_ids"][name] = frame_id
            manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            receipt = JOB.validate_capture_manifest(
                manifest,
                file_sha(manifest),
                artifact_root=root,
                fixture_path=fixture,
                fixture_sha256=file_sha(fixture),
            )
            self.assertTrue(all(str(value).startswith("rgb-sha256:") for value in receipt["camera_frame_ids"].values()))
            payload = json.loads(manifest.read_text())
            payload["native_clock"]["camera_counters"]["head_camera"]["rgb_array_identity"]["value_sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            with self.assertRaisesRegex(JOB.D1JobError, "capture_rgb_identity_binding_mismatch"):
                JOB.validate_capture_manifest(
                    manifest,
                    file_sha(manifest),
                    artifact_root=root,
                    fixture_path=fixture,
                    fixture_sha256=file_sha(fixture),
                )

    def test_retained_artifact_paths_cannot_escape_attempt_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            allowed = root / "allowed"
            allowed.mkdir()
            outside = root / "outside.npy"
            np.save(outside, np.zeros(1), allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "escaped its attempt root"):
                # This helper lives in d1_probe; import the committed module to
                # exercise the exact path gate used by qualification.
                spec = importlib.util.spec_from_file_location(
                    "wmf_d1_probe_path_test", LAYOUT / "d1_probe.py"
                )
                assert spec is not None and spec.loader is not None
                probe = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = probe
                spec.loader.exec_module(probe)
                probe._resolve_artifact(
                    str(outside), allowed / "manifest.json", allowed_root=allowed
                )

    def test_topology_requires_two_unique_idle_b200s_and_exact_visibility(self) -> None:
        receipt = topology_receipt()
        self.assertEqual(receipt["preexisting_compute_process_count"], 0)
        self.assertEqual(len(receipt["torch_devices"]), 2)
        devices = receipt["nvidia_smi_devices"]
        torch = {
            "schema_version": "wmf-d1-torch-topology-v1",
            "status": "passed",
            "device_count": 2,
            "devices": receipt["torch_devices"],
        }
        with self.assertRaisesRegex(JOB.D1JobError, "nvidia_visible_devices_not_exact_two"):
            JOB.reconcile_topology(
                nvidia_devices=devices,
                compute_processes=[],
                torch_receipt=torch,
                visible_devices="all",
            )
        with self.assertRaisesRegex(JOB.D1JobError, "preexisting_compute_processes"):
            JOB.reconcile_topology(
                nvidia_devices=devices,
                compute_processes=[{"pid": "12"}],
                torch_receipt=torch,
                visible_devices=",".join(row["uuid"] for row in devices),
            )
        torch_without_prefix = copy.deepcopy(torch)
        for row in torch_without_prefix["devices"]:
            row["uuid"] = row["uuid"].removeprefix("GPU-")
        normalized = JOB.reconcile_topology(
            nvidia_devices=devices,
            compute_processes=[],
            torch_receipt=torch_without_prefix,
            visible_devices=",".join(row["uuid"] for row in devices),
        )
        self.assertEqual(
            normalized["normalized_gpu_uuids"]["torch"],
            normalized["normalized_gpu_uuids"]["nvidia_smi"],
        )

    def test_commands_pin_torchrun_two_ranks_official_server_and_six_request_probe(self) -> None:
        runtime = JOB.RuntimePaths(
            python=Path("/pinned/python"),
            source=Path("/pinned/DreamZero"),
            checkpoint=Path("/pinned/checkpoint"),
            tokenizer=Path("/pinned/tokenizer"),
        )
        server = JOB.build_server_command(
            runtime=runtime,
            source_root=Path("/study"),
            future_root=Path("/raw/future"),
            port=18021,
            timeout_seconds=50000,
        )
        probe = JOB.build_probe_command(
            runtime=runtime,
            source_root=Path("/study"),
            fixture=Path("/raw/fixture.npz"),
            fixture_sha256="a" * 64,
            future_root=Path("/raw/future"),
            output_dir=Path("/raw/probe"),
            port=18021,
        )
        self.assertEqual(server[:6], ["/pinned/python", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"])
        self.assertIn("d1_instrumented_server.py", " ".join(server))
        self.assertNotIn("s2", " ".join(server).lower())
        self.assertIn("d1_probe.py", " ".join(probe))
        self.assertEqual(probe[probe.index("--fixture-sha256") + 1], "a" * 64)

    def test_runtime_validation_preserves_venv_python_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / "venv/bin/python"
            python.parent.mkdir(parents=True)
            python.symlink_to(Path(sys.executable).resolve())
            source = root / "source"
            checkpoint = root / "checkpoint"
            tokenizer = root / "tokenizer"
            for directory in (source, checkpoint, tokenizer):
                directory.mkdir()
            runtime = JOB.RuntimePaths(python, source, checkpoint, tokenizer)
            JOB.validate_runtime_paths(runtime, enforce_pinned_locations=False)
            command = JOB.build_server_command(
                runtime=runtime,
                source_root=root / "study",
                future_root=root / "future",
                port=18021,
                timeout_seconds=50000,
            )
            self.assertEqual(command[0], str(python))
            self.assertNotEqual(command[0], str(python.resolve()))

    def test_server_contract_reconciles_runtime_identity_and_both_gpu_uuids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            server_path = study / "workshops/corl2026_world_models/experiments/forecast_layout/d1_instrumented_server.py"
            server_path.parent.mkdir(parents=True)
            server_path.write_text("# server\n")
            runtime = JOB.RuntimePaths(
                python=Path(sys.executable),
                source=root / "dreamzero",
                checkpoint=root / "checkpoint",
                tokenizer=root / "tokenizer",
            )
            for directory in (runtime.source, runtime.checkpoint, runtime.tokenizer):
                directory.mkdir()
            future = root / "future"
            future.mkdir()
            identity = {
                "schema_version": "wmf-d1-runtime-identity-receipt-v1",
                "status": "passed",
                "source": {
                    "commit": JOB.EXPECTED_SOURCE_COMMIT,
                    "git_tree": JOB.EXPECTED_SOURCE_TREE,
                    "aggregate_sha256": "1" * 64,
                },
                "checkpoint": {
                    "revision": JOB.EXPECTED_CHECKPOINT_REVISION,
                    "aggregate_sha256": "2" * 64,
                },
                "tokenizer": {
                    "revision": JOB.EXPECTED_TOKENIZER_REVISION,
                    "aggregate_sha256": "3" * 64,
                },
            }
            identity_path = future / "identity_receipt.json"
            identity_path.write_text(json.dumps(identity))
            topology = topology_receipt()
            contract = {
                "schema_version": "wmf-d1-instrumented-server-v1",
                "status": "passed",
                "configuration_id": "D1",
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "official_repository_commit": JOB.EXPECTED_SOURCE_COMMIT,
                "official_repository_tree": JOB.EXPECTED_SOURCE_TREE,
                "source_root": str(runtime.source),
                "checkpoint_root": str(runtime.checkpoint),
                "tokenizer_root": str(runtime.tokenizer),
                "world_size": 2,
                "port": 18021,
                "returned_action_shape": [24, 8],
                "effective_official_model_noise_seed": 1140,
                "enable_dit_cache": True,
                "dynamic_cache_schedule": False,
                "tensorrt_engine_active": False,
                "topology": [
                    {
                        "rank": index,
                        "hostname": "one-node",
                        "cuda_device_index": index,
                        "cuda_device_name": "NVIDIA B200",
                        "cuda_device_uuid": topology["torch_devices"][index]["uuid"].removeprefix("GPU-"),
                    }
                    for index in range(2)
                ],
                "identity_receipt": str(identity_path),
                "identity_receipt_sha256": file_sha(identity_path),
                "bounded_loader_receipts": [
                    {"rank": index, "receipt": {"passed": True, "forward_path_modified": False}}
                    for index in range(2)
                ],
                "instrumentation_overlay": {"path": str(server_path), "returned_action_modified": False},
            }
            contract_path = future / "server_contract.json"
            contract_path.write_text(json.dumps(contract))
            observed = JOB.validate_server_contract(
                contract_path,
                runtime=runtime,
                source_root=study,
                topology=topology,
                port=18021,
            )
            self.assertFalse(observed["custom_s2_used"])
            contract["custom_s2_used"] = True
            contract_path.write_text(json.dumps(contract))
            with self.assertRaisesRegex(JOB.D1JobError, "server_contract_gate_failed"):
                JOB.validate_server_contract(
                    contract_path,
                    runtime=runtime,
                    source_root=study,
                    topology=topology,
                    port=18021,
                )

    def test_server_process_group_is_terminated_and_reaped_with_log_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stdout_path = root / "server.stdout.log"
            stderr_path = root / "server.stderr.log"
            process, stdout, stderr = JOB.launch_server(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cwd=root,
                env=os.environ,
            )
            receipt = JOB.terminate_server(
                process,
                stdout_handle=stdout,
                stderr_handle=stderr,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                grace_seconds=5,
            )
            self.assertEqual(receipt["status"], "reaped")
            self.assertTrue(receipt["reaped"])
            self.assertEqual(receipt["signal_sent"], "SIGTERM")
            self.assertFalse(receipt["sigkill_required"])
            self.assertEqual(receipt["stdout"]["sha256"], hashlib.sha256(b"").hexdigest())

    def test_outer_sigterm_reaps_nested_server_group_before_queue_kill_grace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_path = LAYOUT / "d1_qualification_job.py"
            nested_code = r'''
import json, os, signal, subprocess, sys, time
grand = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
def stop(_signal, _frame):
    try:
        grand.terminate()
        grand.wait(timeout=1)
    except BaseException:
        try: grand.kill()
        except BaseException: pass
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
open(sys.argv[1], "w").write(json.dumps({"server_pid": os.getpid(), "server_pgid": os.getpgrp(), "rank_pid": grand.pid, "rank_pgid": os.getpgid(grand.pid)}))
while True: time.sleep(1)
'''
            harness = r'''
import importlib.util, json, os, signal, sys, time
from pathlib import Path
spec = importlib.util.spec_from_file_location("signal_harness_job", sys.argv[1])
job = importlib.util.module_from_spec(spec); sys.modules[spec.name] = job; spec.loader.exec_module(job)
root = Path(sys.argv[2]); info = root / "nested.json"
proc, stdout, stderr = job.launch_server(
    [sys.executable, "-c", sys.argv[3], str(info)],
    stdout_path=root / "nested.stdout", stderr_path=root / "nested.stderr",
    cwd=root, env=os.environ,
)
job._ACTIVE_SERVER = proc
signal.signal(signal.SIGTERM, job._signal_handler)
signal.signal(signal.SIGINT, job._signal_handler)
while not info.exists():
    if proc.poll() is not None: raise SystemExit(9)
    time.sleep(.02)
(root / "ready.json").write_text(json.dumps({"wrapper_pid": os.getpid(), "server_pid": proc.pid, "server_pgid": os.getpgid(proc.pid)}))
try:
    while True: time.sleep(1)
except job.D1JobError:
    pass
finally:
    receipt = job.terminate_server(
        proc, stdout_handle=stdout, stderr_handle=stderr,
        stdout_path=root / "nested.stdout", stderr_path=root / "nested.stderr",
        grace_seconds=2, prior_signal_reap=job._SIGNAL_SERVER_REAP,
    )
    (root / "done.json").write_text(json.dumps(receipt))
'''
            wrapper = subprocess.Popen(
                [sys.executable, "-c", harness, str(module_path), str(root), nested_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            try:
                ready_path = root / "ready.json"
                deadline = time.monotonic() + 5
                while not ready_path.exists() and wrapper.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready_path.is_file())
                ready = json.loads(ready_path.read_text())
                nested = json.loads((root / "nested.json").read_text())
                self.assertEqual(ready["server_pgid"], nested["server_pgid"])
                self.assertEqual(nested["rank_pgid"], nested["server_pgid"])
                started = time.monotonic()
                os.kill(wrapper.pid, signal.SIGTERM)
                stdout, stderr = wrapper.communicate(timeout=8)
                self.assertLess(time.monotonic() - started, 8)
                self.assertEqual(wrapper.returncode, 0, (stdout, stderr))
                receipt = json.loads((root / "done.json").read_text())
                self.assertTrue(receipt["reaped"])
                self.assertTrue(receipt["signal_handler_reap"]["reaped"])
                group_deadline = time.monotonic() + 2
                while time.monotonic() < group_deadline:
                    try:
                        os.killpg(nested["server_pgid"], 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail("nested server/rank process group survived outer SIGTERM")
            finally:
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait(timeout=3)
                if (root / "nested.json").is_file():
                    pgid = json.loads((root / "nested.json").read_text())["server_pgid"]
                    try:
                        os.killpg(pgid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_preflight_failure_is_persisted_and_published_as_technical_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            job_dir = root / "state/jobs/d1-preflight-failure"
            source.mkdir()
            job_dir.mkdir(parents=True)
            queue = JOB.QueueContext(source, job_dir, "b" * 40, "d1-preflight-failure")
            missing = JOB.RuntimePaths(
                python=root / "missing-python",
                source=root / "missing-source",
                checkpoint=root / "missing-checkpoint",
                tokenizer=root / "missing-tokenizer",
            )
            with mock.patch.object(JOB, "validate_queue_context", return_value=queue):
                receipt = JOB.execute_job(
                    source_root=source,
                    job_dir=job_dir,
                    fixture=root / "missing-fixture.npz",
                    fixture_sha256="c" * 64,
                    capture_manifest=root / "missing-capture.json",
                    capture_manifest_sha256="d" * 64,
                    port=self._free_port(),
                    runtime=missing,
                    enforce_pinned_locations=False,
                )
            self.assertEqual(receipt["decision"], "technical_invalid")
            self.assertEqual(receipt["exit_code"], 3)
            self.assertEqual(receipt["reason"], "d1_python_missing")
            self.assertTrue((job_dir / "raw/technical_failure.json").is_file())
            published = json.loads(
                (job_dir / "publish/d1_qualification_job_receipt.json").read_text()
            )
            self.assertEqual(published["decision"], "technical_invalid")

    def test_end_to_end_orchestrator_publishes_bounded_pass_and_reaps_server(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            job_dir = root / "state/jobs/d1-test"
            source.mkdir()
            job_dir.mkdir(parents=True)
            fixture = root / "fixture.npz"
            np.savez(fixture, **fixture_arrays())
            fixture_digest = file_sha(fixture)
            capture_manifest = write_capture_manifest(root, fixture)
            runtime = JOB.RuntimePaths(
                python=Path(sys.executable),
                source=root / "dreamzero",
                checkpoint=root / "checkpoint",
                tokenizer=root / "tokenizer",
            )
            for directory in (runtime.source, runtime.checkpoint, runtime.tokenizer):
                directory.mkdir()
            cuda_home = root / "cuda"
            (cuda_home / "bin").mkdir(parents=True)
            (cuda_home / "bin/nvcc").write_text("test-only executable placeholder\n")
            queue = JOB.QueueContext(source, job_dir, "a" * 40, "d1-test")
            real_launch = JOB.launch_server

            def fake_launch(_command, **kwargs):
                return real_launch(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    stdout_path=kwargs["stdout_path"],
                    stderr_path=kwargs["stderr_path"],
                    cwd=kwargs["cwd"],
                    env=kwargs["env"],
                )

            def fake_wait(_process, contract_path, **_kwargs):
                contract_path.parent.mkdir(parents=True)
                identity_path = contract_path.parent / "identity_receipt.json"
                identity = {
                    "source": {"aggregate_sha256": "1" * 64},
                    "checkpoint": {"aggregate_sha256": "2" * 64},
                    "tokenizer": {"aggregate_sha256": "3" * 64},
                }
                identity_path.write_text(json.dumps(identity))
                contract = {
                    "identity_receipt": str(identity_path),
                    "official_repository_commit": JOB.EXPECTED_SOURCE_COMMIT,
                    "official_repository_tree": JOB.EXPECTED_SOURCE_TREE,
                    "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                    "custom_s2_used": False,
                    "patched_s1_used": False,
                    "world_size": 2,
                }
                contract_path.write_text(json.dumps(contract))
                return contract

            def fake_probe_command(**kwargs):
                report_path = kwargs["output_dir"] / "d1_probe_qualification.json"
                report = {
                    "schema_version": "wmf-d1-six-request-qualification-v1",
                    "status": "passed",
                    "passed": True,
                    "generation_request_count": 6,
                    "behavioral_episode_count": 0,
                    "failed_checks": [],
                    "comparisons": {"repeat": {"action_array_equal": True}},
                }
                script = (
                    "from pathlib import Path; import json; "
                    f"p=Path({str(report_path)!r}); p.parent.mkdir(parents=True); "
                    f"p.write_text(json.dumps({report!r}))"
                )
                return [sys.executable, "-c", script]

            with (
                mock.patch.object(JOB, "validate_queue_context", return_value=queue),
                mock.patch.object(JOB, "verify_gpu_topology", return_value=topology_receipt()),
                mock.patch.object(JOB, "launch_server", side_effect=fake_launch),
                mock.patch.object(JOB, "wait_for_server_contract", side_effect=fake_wait),
                mock.patch.object(JOB, "build_probe_command", side_effect=fake_probe_command),
                mock.patch.object(JOB, "D1_CUDA_HOME", cuda_home),
            ):
                receipt = JOB.execute_job(
                    source_root=source,
                    job_dir=job_dir,
                    fixture=fixture,
                    fixture_sha256=fixture_digest,
                    capture_manifest=capture_manifest,
                    capture_manifest_sha256=file_sha(capture_manifest),
                    port=self._free_port(),
                    runtime=runtime,
                    enforce_pinned_locations=False,
                    terminate_grace_seconds=5,
                )
            self.assertEqual(receipt["decision"], "qualified")
            self.assertEqual(receipt["generation_request_count"], 6)
            self.assertEqual(receipt["behavioral_episode_count"], 0)
            self.assertTrue(receipt["server_exit"]["reaped"])
            publish = job_dir / "publish/d1_qualification_job_receipt.json"
            self.assertLess(publish.stat().st_size, JOB.MAX_PUBLISH_BYTES)
            self.assertEqual(json.loads(publish.read_text())["decision"], "qualified")

    @staticmethod
    def _free_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return int(port)


if __name__ == "__main__":
    unittest.main()
