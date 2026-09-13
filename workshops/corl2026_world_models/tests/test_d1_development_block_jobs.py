from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
SOURCE_ROOT = WORKSHOP.parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))

import d1_behavioral_pilot_jobs as pilot  # noqa: E402
import d1_development_block_jobs as development  # noqa: E402
import fixed_observation_job as fixed  # noqa: E402


STUDY_COMMIT = "a" * 40


def _write(path: Path, value: dict) -> dict:
    pilot.immutable_json(path, value)
    return pilot.file_identity(path)


def _artifact(path: Path, payload: bytes = b"artifact") -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return pilot.file_identity(path)


def _fixture_bundle(root: Path, block: development.DevelopmentBlock) -> tuple[dict, Path, str]:
    gate = _artifact(root / "gate.json", b"gate")
    pose = _artifact(root / "pose.json", b"pose")
    gate_ledger = _artifact(root / "gate-ledger.jsonl", b"gate-ledger")
    gate_attempt = _artifact(root / "gate-attempt.json", b"gate-attempt")
    release = {
        "candidate_id": f"{block.layout_pair_id}__candidate_00",
        "candidate_payload_sha256": "b" * 64,
        "accepted_gate_record_sha256": "c" * 64,
        "gate_receipt": gate,
        "pose_manifest": pose,
        "gate_ledger": gate_ledger,
        "gate_attempt_receipt": gate_attempt,
    }
    artifact_keys = {
        "raw_settled_observation": ("image_obs/head_camera",),
        "simulator_state": ("objects/rubiks_cube/position_robot_base_m",),
        "preprocessing_intermediates": ("D1/raw",),
        "N3": fixed.N3_ARRAY_KEYS,
        "D1": fixed.D1_ARRAY_KEYS,
    }
    artifacts = {}
    for name, keys in artifact_keys.items():
        descriptor = _artifact(root / f"{name}.npz", name.encode())
        descriptor["arrays"] = {
            key: {
                "dtype": "|u1",
                "shape": [1],
                "data_sha256": "1" * 64,
            }
            for key in keys
        }
        artifacts[name] = descriptor
    d1_fixture = {
        key: artifacts["D1"][key] for key in ("path", "bytes", "sha256")
    }
    n3_fixture = {
        key: artifacts["N3"][key] for key in ("path", "bytes", "sha256")
    }
    model_fixtures = {
        "N3": {
            "interface": "official packed observation/image plus joint/gripper arrays",
            "array_keys": list(fixed.N3_ARRAY_KEYS),
            "fixture": n3_fixture,
        },
        "D1": {
            "interface": "official conditional DreamZero six-array request fixture",
            "array_keys": list(fixed.D1_ARRAY_KEYS),
            "fixture": d1_fixture,
        },
    }
    reset_identity = (
        f"fixed-observation:{STUDY_COMMIT}:{block.layout_pair_id}:{block.environment_seed}"
    )
    collision = {"passed": True, "forbidden_pairs": {}}
    visibility = {"passed": True, "cameras": {}}
    cameras = {}
    frame_ids = {}
    timestamps = {}
    timestamp_sources = {}
    for index, name in enumerate(fixed.RAW_CAMERAS):
        value_sha = f"{index + 2:x}" * 64
        frame_id = f"rgb-sha256:{value_sha}"
        timestamp = 5_000_000_000 + index
        source = "_timestamp_last_update"
        cameras[name] = {
            "frame_id": frame_id,
            "frame_identity_source": "exact returned RGB array value identity",
            "rgb_array_identity": {"value_sha256": value_sha},
            "native_frame_counter": None,
            "native_frame_counter_status": "unavailable_in_pinned_isaaclab_sensorbase",
            "capture_time_ns": timestamp,
            "native_capture_time_source": source,
        }
        frame_ids[name] = frame_id
        timestamps[name] = timestamp
        timestamp_sources[name] = source
    overlay_source = _artifact(root / "d1_overlay.py", b"overlay")
    official_source = _artifact(root / "official_client.py", b"official")
    robolab_source = _artifact(root / "robolab.py", b"robolab")
    task_source = _artifact(root / "original_left.py", b"task")
    artifact_logical = {
        name: {
            "sha256": descriptor["sha256"],
            "bytes": descriptor["bytes"],
            "arrays": descriptor["arrays"],
        }
        for name, descriptor in artifacts.items()
    }
    raw = {
        "schema_version": development.FIXED_CAPTURE_SCHEMA,
        "study_namespace": pilot.NAMESPACE,
        "status": "passed",
        "capture_id": f"{block.layout_pair_id}-{block.environment_seed}-{'e' * 16}",
        "captured_at_utc": "2026-09-13T00:00:00Z",
        "study_commit": STUDY_COMMIT,
        "robolab_commit": fixed.ROBOLAB_COMMIT,
        "layout_pair_id": block.layout_pair_id,
        "candidate_id": release["candidate_id"],
        "candidate_payload_sha256": release["candidate_payload_sha256"],
        "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
        "environment_seed": block.environment_seed,
        "layout_arm": "original",
        "command_task_used_for_reset": "left",
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "gate_receipt": gate,
        "pose_manifest": pose,
        "gate_ledger": gate_ledger,
        "gate_attempt_receipt": gate_attempt,
        "settled_reset_identity": reset_identity,
        "settled_reset_receipt": {
            "schema_version": "wmf-forecast-layout-settled-reset-receipt-v1",
            "passed": True,
            "settled": True,
            "left_success": False,
            "right_success": False,
            "released": True,
            "reset_identity": reset_identity,
            "pose_manifest_sha256": pose["sha256"],
            "initial_observation_hashes": {"combined_sha256": "f" * 64},
            "settle_evidence": {
                "settle_steps": 60,
                "stability_window_steps": 15,
            },
            "collision_evidence": collision,
            "visibility_evidence": visibility,
            "settled_observation_returned": True,
            "model_request_count_during_settle": 0,
            "episode_length_buf_reset_to_zero": True,
        },
        "fresh_physical_checks": {
            "collision": collision,
            "visibility": visibility,
        },
        "native_clock": {
            "physics_step": 600,
            "physics_step_source": "frame_count",
            "physics_time_s": 5.0,
            "physics_time_source": "current_time",
            "control_step_since_physical_reset": 75,
            "control_step_source": "common_step_counter",
            "behavioral_episode_step": 0,
            "behavioral_episode_step_source": "episode_length_buf",
            "camera_counters": cameras,
            "host_read_window": {
                "started_wall_time_ns": 1,
                "finished_wall_time_ns": 2,
                "started_monotonic_ns": 3,
                "finished_monotonic_ns": 4,
            },
            "runtime_configuration": {
                "physics_dt_s": 1 / 120,
                "decimation": 8,
                "render_interval": 8,
            },
            "timing_claim_boundary": "native runtime counters only",
        },
        "source_capture": {
            "simulator_observation_id": (
                f"{block.layout_pair_id}_original_settled_observation_000000"
            ),
            "camera_frame_ids": frame_ids,
            "camera_capture_time_ns": timestamps,
            "camera_timestamp_source": timestamp_sources,
        },
        "artifacts": artifacts,
        "artifact_logical_sha256": fixed.sha256_bytes(
            fixed.compact_canonical_bytes(artifact_logical)
        ),
        "preprocessing": {
            "D1": {
                "overlay_source": overlay_source,
                "official_client_source": official_source,
                "extraction_method": "_extract_observation",
                "packing_method": "_pack_request",
                "configuration": {
                    "cam2_source": "right",
                    "resize": "pad",
                    "image_height": 180,
                    "image_width": 320,
                },
                "wire_array_keys": list(fixed.D1_ARRAY_KEYS),
            },
            "model_request_count": 0,
        },
        "model_fixtures": model_fixtures,
        "runtime_identity": {
            "pod": "test-pod",
            "pod_uid": "test-uid",
            "python": "3.11-test",
            "python_executable": "/venv/bin/python",
            "robolab_module": robolab_source,
            "task_file": task_source,
            "renderer": "realtime",
            "rendering_type": "balanced",
            "device": "cuda:0",
        },
        "settling_hold_action_count": 75,
    }
    raw_identity = _write(root / "raw_capture.json", raw)
    queue = {
        "schema_version": development.FIXED_CAPTURE_QUEUE_SCHEMA,
        "study_namespace": pilot.NAMESPACE,
        "status": "passed",
        "exit_code": 0,
        "job_id": f"fixed-{block.layout_pair_id.lower()}",
        "study_commit": STUDY_COMMIT,
        "layout_pair_id": block.layout_pair_id,
        "candidate_id": release["candidate_id"],
        "environment_seed": block.environment_seed,
        "gate_receipt_sha256": gate["sha256"],
        "pose_manifest_sha256": pose["sha256"],
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "gpu_identity": {
            "index": "0",
            "uuid": "GPU-test",
            "name": "NVIDIA B200",
            "driver_version": "test",
            "preexisting_compute_process_count": 0,
        },
        "child_started": True,
        "child_exit_code": 0,
        "child_logs": {
            "stdout": _artifact(root / "child.stdout", b"stdout"),
            "stderr": _artifact(root / "child.stderr", b"stderr"),
        },
        "raw_capture_receipt": raw_identity,
        "model_fixtures": model_fixtures,
    }
    path = root / "fixed_observation_job_receipt.json"
    _write(path, queue)
    return release, path, pilot.sha256_file(path)


def _p00_pair(root: Path) -> tuple[Path, str, Path, str, str, str]:
    qualification = _artifact(root / "qualification.json", b"qualification")
    recorder = _artifact(root / "recorder.json", b"recorder")
    ready = _artifact(root / "ready.json", b"ready")
    claim = _artifact(root / "claim.json", b"claim")
    cells = []
    for index in range(4):
        path = root / f"cell-{index}.json"
        cells.append(_write(path, {"condition_index": index}))
    simulator_value = {
        "schema_version": development.P00_SIMULATOR_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "study_id": pilot.STUDY_ID,
        "namespace": pilot.NAMESPACE,
        "run_id": "p00-run",
        "server_job_id": "p00-server",
        "simulator_job_id": "p00-simulator",
        "study_commit": STUDY_COMMIT,
        "block_id": development.P00_BLOCK_ID,
        "phase": "pilot",
        "layout_pair_id": "P00",
        "model_config": "D1",
        "effective_model_noise_seed": 1140,
        "condition_order": list(development.P00_CONDITION_ORDER),
        "cell_ids": list(development.P00_CELL_IDS),
        "all_simulator_children_reaped": True,
        "counts": {
            "planned_behavioral_cells": 4,
            "launched_behavioral_cells": 4,
            "completed_valid_behavioral_cells": 4,
            "technically_invalid_behavioral_cells": 0,
            "right_censored_behavioral_cells": 0,
            "unrun_behavioral_cells": 0,
            "actual_behavioral_actions": 1800,
            "actual_behavioral_model_requests": 228,
            "new_generation_qualification_requests": 0,
            "reused_prerequisite_generation_qualification_requests": 6,
            "recorder_only_episodes_counted_as_behavioral": 0,
        },
        "cell_receipts": cells,
        "prerequisites": {
            "recorder_receipt": recorder,
            "d1_qualification_receipt": qualification,
        },
        "server_ready": ready,
        "simulator_claim": claim,
    }
    simulator_path = root / "d1_behavioral_pilot_receipt.json"
    simulator_identity = _write(simulator_path, simulator_value)
    terminal = _write(
        root / "terminal.json",
        {
            "status": "passed",
            "safe_for_server_shutdown": True,
            "all_simulator_children_reaped": True,
            "run_id": "p00-run",
            "server_job_id": "p00-server",
            "simulator_job_id": "p00-simulator",
            "block_id": development.P00_BLOCK_ID,
            "simulator_receipt": simulator_identity,
        },
    )
    server_value = {
        "schema_version": development.P00_SERVER_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "run_id": "p00-run",
        "server_job_id": "p00-server",
        "paired_simulator_job_id": "p00-simulator",
        "study_commit": STUDY_COMMIT,
        "block_id": development.P00_BLOCK_ID,
        "all_server_children_reaped": True,
        "server_ready": ready,
        "simulator_claim": claim,
        "d1_qualification_receipt": qualification,
        "simulator_terminal": terminal,
        "server_process_exit": {"status": "reaped", "reaped": True},
    }
    server_path = root / "d1_behavioral_server_receipt.json"
    _write(server_path, server_value)
    return (
        simulator_path,
        pilot.sha256_file(simulator_path),
        server_path,
        pilot.sha256_file(server_path),
        recorder["sha256"],
        qualification["sha256"],
    )


class FrozenContractTests(unittest.TestCase):
    def test_all_four_rows_bind_order_fixed_noise_and_physical_seed(self) -> None:
        for layout_id, (environment_seed, order) in development.DEVELOPMENT_SCHEDULE.items():
            with self.subTest(layout_id=layout_id):
                block = development.load_development_block(SOURCE_ROOT, layout_id)
                self.assertEqual(block.environment_seed, environment_seed)
                self.assertEqual(block.condition_order, order)
                self.assertEqual(len(set(block.conditions)), 4)
                self.assertEqual(block.schedule_row["candidate_effective_policy_seed"], 1140)
                self.assertEqual(block.contract["effective_model_noise_seed"], 1140)
                self.assertEqual(block.contract["environment_seed"], environment_seed)
                self.assertEqual(block.contract["per_cell"]["requests"], 57)
                self.assertEqual(block.contract["transport"]["ping_interval"], None)
                self.assertFalse(block.contract["transport"]["stateful_request_replay"])

    def test_only_d01_through_d04_are_supported(self) -> None:
        for layout in ("P00", "C01", "D05", "d01"):
            with self.subTest(layout=layout):
                with self.assertRaisesRegex(
                    pilot.D1BehavioralPilotError, "development_layout_unsupported"
                ):
                    development.load_development_block(SOURCE_ROOT, layout)

    def test_configuration_is_scoped_and_uses_global_d1_lock(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D03")
        original = {
            name: getattr(pilot, name)
            for name in development._PILOT_DEFAULTS
        }
        with development.configured_pilot(block, "wmf-forecast-0912-worker-06"):
            self.assertEqual(pilot.PHASE, "development")
            self.assertEqual(pilot.LAYOUT_PAIR_ID, "D03")
            self.assertEqual(pilot.ENVIRONMENT_SEED, 2026091103)
            self.assertEqual(pilot.EFFECTIVE_MODEL_NOISE_SEED, 1140)
            self.assertEqual(pilot.RUNNER_FILENAME, development.RUNNER_FILENAME)
            self.assertEqual(pilot.BEHAVIORAL_PURPOSE, "d1_behavioral_development")
            self.assertEqual(pilot.GLOBAL_SERVER_LOCK_PATH, development.GLOBAL_D1_SERVER_LOCK)
            self.assertEqual(pilot.SIMULATOR_QUEUE_ROLE, "wmf-forecast-0912-worker-06")
        for name, value in original.items():
            self.assertEqual(getattr(pilot, name), value)


class CommandsAndPairingTests(unittest.TestCase):
    def test_child_reenters_development_runner_with_capture_and_pair_identity(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D02")
        command = development.build_cell_command(
            source_root=SOURCE_ROOT,
            attempt_root=block.raw_root / "simulator_attempts/sim-1",
            study_commit=STUDY_COMMIT,
            gate_receipt=Path("/evidence/gate.json"),
            gate_receipt_sha256="b" * 64,
            pose_manifest=Path("/evidence/pose.json"),
            pose_manifest_sha256="c" * 64,
            capture_receipt=Path("/evidence/capture.json"),
            capture_receipt_sha256="d" * 64,
            execution_prerequisites=Path("/evidence/execution-prerequisites.json"),
            execution_prerequisites_sha256="1" * 64,
            candidate_id="D02__candidate_00",
            simulator_worker_role="wmf-forecast-0912-worker-05",
            run_id="run-1",
            server_job_id="server-1",
            server_ready_sha256="e" * 64,
            simulator_claim_sha256="f" * 64,
            lease_token="lease-token",
            future_root=block.raw_root / "server_attempts/server-1/future",
            condition_index=2,
            block=block,
        )
        self.assertTrue(command[1].endswith("/d1_development_block_jobs.py"))
        self.assertEqual(command[command.index("--layout-pair-id") + 1], "D02")
        self.assertEqual(command[command.index("--layout-arm") + 1], "reflected")
        self.assertEqual(command[command.index("--command") + 1], "left")
        self.assertIn("--capture-receipt-sha256", command)
        self.assertEqual(
            command[command.index("--execution-prerequisites-sha256") + 1],
            "1" * 64,
        )
        self.assertEqual(command[command.index("--remote-host") + 1], pilot.SERVICE_HOST)

    def test_queue_descriptor_pairing_is_fail_closed(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "server-1"
            job.mkdir()
            argv = [
                "/usr/bin/python3",
                "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
                + development.RUNNER_FILENAME,
                "server-job",
                "--layout-pair-id", "D01",
                "--simulator-worker-role", "wmf-forecast-0912-worker-05",
                "--study-commit", STUDY_COMMIT,
                "--job-id", "server-1",
                "--run-id", "run-1",
                "--simulator-job-id", "sim-1",
            ]
            (job / "descriptor.json").write_text(json.dumps({"argv": argv}))
            with mock.patch.object(
                development,
                "_PILOT_VALIDATE_QUEUE_INVOCATION",
                return_value={
                    **pilot.file_identity(job / "descriptor.json"),
                    "role": pilot.SERVER_QUEUE_ROLE,
                    "job_id": "server-1",
                },
            ):
                development.validate_queue_invocation(
                    source_root=SOURCE_ROOT,
                    job_dir=job,
                    study_commit=STUDY_COMMIT,
                    job_id="server-1",
                    expected_role=pilot.SERVER_QUEUE_ROLE,
                    expected_mode="server-job",
                    paired_job_id="sim-1",
                    run_id="run-1",
                    block=block,
                    simulator_worker_role="wmf-forecast-0912-worker-05",
                )
                argv[-1] = "other-simulator"
                (job / "descriptor.json").write_text(json.dumps({"argv": argv}))
                with self.assertRaisesRegex(
                    pilot.D1BehavioralPilotError, "development_queue_pairing_changed"
                ):
                    development.validate_queue_invocation(
                        source_root=SOURCE_ROOT,
                        job_dir=job,
                        study_commit=STUDY_COMMIT,
                        job_id="server-1",
                        expected_role=pilot.SERVER_QUEUE_ROLE,
                        expected_mode="server-job",
                        paired_job_id="sim-1",
                        run_id="run-1",
                        block=block,
                        simulator_worker_role="wmf-forecast-0912-worker-05",
                    )

    def test_queue_descriptor_is_rehashed_after_extended_validation(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            job = Path(temporary) / "server-1"
            job.mkdir()
            argv = [
                "/usr/bin/python3",
                "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
                + development.RUNNER_FILENAME,
                "server-job",
                "--layout-pair-id", "D01",
                "--simulator-worker-role", "wmf-forecast-0912-worker-05",
                "--study-commit", STUDY_COMMIT,
                "--job-id", "server-1",
                "--run-id", "run-1",
                "--simulator-job-id", "sim-1",
            ]
            (job / "descriptor.json").write_text(json.dumps({"argv": argv}))
            stale = {**pilot.file_identity(job / "descriptor.json"), "role": "d1"}
            stale["sha256"] = "0" * 64
            with mock.patch.object(
                development,
                "_PILOT_VALIDATE_QUEUE_INVOCATION",
                return_value=stale,
            ), self.assertRaisesRegex(
                pilot.D1BehavioralPilotError,
                "development_queue_descriptor_changed_during_validation",
            ):
                development.validate_queue_invocation(
                    source_root=SOURCE_ROOT,
                    job_dir=job,
                    study_commit=STUDY_COMMIT,
                    job_id="server-1",
                    expected_role=pilot.SERVER_QUEUE_ROLE,
                    expected_mode="server-job",
                    paired_job_id="sim-1",
                    run_id="run-1",
                    block=block,
                    simulator_worker_role="wmf-forecast-0912-worker-05",
                )

    def test_server_cli_and_launch_are_gated_by_all_layout_prerequisites(self) -> None:
        parser = development.build_parser()
        action = next(item for item in parser._actions if item.dest == "mode")
        server_dests = {item.dest for item in action.choices["server-job"]._actions}
        self.assertTrue(
            {
                "candidate_id", "gate_receipt", "gate_receipt_sha256",
                "pose_manifest", "pose_manifest_sha256", "capture_receipt",
                "capture_receipt_sha256", "recorder_receipt",
                "recorder_receipt_sha256", "d1_qualification_receipt",
                "d1_qualification_receipt_sha256", "pilot_simulator_receipt",
                "pilot_server_receipt",
            }.issubset(server_dests)
        )
        block = development.load_development_block(SOURCE_ROOT, "D01")
        args = development.argparse.Namespace(
            raw_root=block.raw_root,
            source_root=SOURCE_ROOT,
            job_dir=Path("/queue/jobs/server-1"),
            study_commit=STUDY_COMMIT,
            job_id="server-1",
            simulator_job_id="sim-1",
            run_id="run-1",
            simulator_worker_role="wmf-forecast-0912-worker-05",
        )
        with mock.patch.object(
            development, "validate_queue_invocation", return_value={"sha256": "a" * 64}
        ), mock.patch.object(
            development,
            "validate_prerequisites",
            side_effect=pilot.D1BehavioralPilotError("layout_prerequisite_failed"),
        ) as prerequisites, mock.patch.object(pilot, "run_server_job") as launch:
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError, "layout_prerequisite_failed"
            ):
                development.run_server_job(args, block)
        prerequisites.assert_called_once_with(args, block)
        launch.assert_not_called()


class PrerequisiteTests(unittest.TestCase):
    def test_fixed_capture_is_hash_bound_to_gate_pose_candidate_and_d1_arrays(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D04")
        with tempfile.TemporaryDirectory() as temporary:
            release, path, digest = _fixture_bundle(Path(temporary), block)
            verified = development.verify_development_capture(
                path, digest, block=block, release=release
            )
            self.assertEqual(
                verified["capture_receipt"]["sha256"], digest
            )
            self.assertTrue(verified["d1_fixed_observation"]["path"].endswith(".npz"))
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError, "evidence_sha256_mismatch"
            ):
                development.verify_development_capture(
                    path, "0" * 64, block=block, release=release
                )

    def test_fixed_capture_rejects_missing_physical_evidence_and_queue_rebinding(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D03")
        for mutation, error in (
            ("missing_physical", "development_capture_physical_checks_missing"),
            ("queue_fixture_rebound", "development_capture_queue_fixture_binding_changed"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                release, path, _digest = _fixture_bundle(Path(temporary), block)
                queue = json.loads(path.read_text())
                raw_path = Path(queue["raw_capture_receipt"]["path"])
                raw = json.loads(raw_path.read_text())
                if mutation == "missing_physical":
                    raw.pop("fresh_physical_checks")
                    raw_path.write_bytes(pilot.canonical_bytes(raw))
                    queue["raw_capture_receipt"] = pilot.file_identity(raw_path)
                else:
                    queue["model_fixtures"]["D1"]["interface"] = "rebound"
                path.write_bytes(pilot.canonical_bytes(queue))
                with self.assertRaisesRegex(pilot.D1BehavioralPilotError, error):
                    development.verify_development_capture(
                        path,
                        pilot.sha256_file(path),
                        block=block,
                        release=release,
                    )

    def test_complete_p00_pair_is_required_and_cross_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            values = _p00_pair(Path(temporary))

            def validator(path: Path, *, condition_index: int, study_commit: str):
                self.assertEqual(study_commit, STUDY_COMMIT)
                return (
                    {
                        "server_begin_receipt": {
                            "episode_context_id": f"episode-{condition_index}",
                            "client_session_id": f"session-{condition_index}",
                        }
                    },
                    pilot.file_identity(path),
                )

            with mock.patch.object(
                development, "_PILOT_VALIDATE_PASSED_CELL", side_effect=validator
            ), mock.patch.object(
                development,
                "_validate_live_ready_descriptor",
                side_effect=lambda value, _label: (
                    pilot._verify_descriptor(value, "test_ready"),
                    {},
                ),
            ):
                verified = development.verify_passed_p00_pair(
                    simulator_receipt_path=values[0],
                    simulator_receipt_sha256=values[1],
                    server_receipt_path=values[2],
                    server_receipt_sha256=values[3],
                    recorder_receipt_sha256=values[4],
                    d1_qualification_receipt_sha256=values[5],
                )
                self.assertEqual(verified["behavioral_cells"], 4)
                self.assertEqual(verified["behavioral_actions"], 1800)
                self.assertEqual(verified["behavioral_model_requests"], 228)
                with self.assertRaisesRegex(
                    pilot.D1BehavioralPilotError, "p00_recorder_receipt_binding_changed"
                ):
                    development.verify_passed_p00_pair(
                        simulator_receipt_path=values[0],
                        simulator_receipt_sha256=values[1],
                        server_receipt_path=values[2],
                        server_receipt_sha256=values[3],
                        recorder_receipt_sha256="0" * 64,
                        d1_qualification_receipt_sha256=values[5],
                    )


class RecoveryAndReceiptTests(unittest.TestCase):
    def _prerequisites(self, root: Path, block: development.DevelopmentBlock) -> dict:
        root.mkdir(parents=True, exist_ok=True)
        release, capture_path, capture_sha = _fixture_bundle(root, block)
        capture = development.verify_development_capture(
            capture_path, capture_sha, block=block, release=release
        )
        recorder = _artifact(root / "validated-recorder.json", b"recorder")
        recorder_child = _artifact(root / "validated-recorder-child.json", b"recorder-child")
        timing = _artifact(root / "validated-native-timing.json", b"timing")
        qualification = _artifact(root / "validated-d1-qualification.json", b"qualification")
        p00_simulator = _artifact(root / "validated-p00-simulator.json", b"p00-simulator")
        p00_server = _artifact(root / "validated-p00-server.json", b"p00-server")
        return {
            "candidate_id": release["candidate_id"],
            "candidate_payload_sha256": release["candidate_payload_sha256"],
            "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
            "development_gate_receipt": release["gate_receipt"],
            "development_pose_manifest": release["pose_manifest"],
            **capture,
            "recorder_receipt": recorder,
            "recorder_child_receipt": recorder_child,
            "native_timing_support": timing,
            "d1_qualification_receipt": qualification,
            "p00_paired_pilot": {
                "simulator_receipt": p00_simulator,
                "server_receipt": p00_server,
                "behavioral_cells": 4,
                "behavioral_actions": 1800,
                "behavioral_model_requests": 228,
                "generation_qualification_requests_rerun": 0,
            },
            "generation_qualification_requests_reused_not_rerun": 6,
            "mapping_qualification": "validated test prerequisite",
        }

    def _fixture(
        self, root: Path, block: development.DevelopmentBlock,
        prerequisites: dict,
    ) -> dict:
        prerequisites_identity, prerequisites_sha = (
            development.write_execution_prerequisites(
                root / "execution_prerequisites.json",
                block=block,
                prerequisites=prerequisites,
            )
        )
        return {
            "candidate_id": prerequisites["candidate_id"],
            "candidate_payload_sha256": prerequisites["candidate_payload_sha256"],
            "accepted_gate_record_sha256": prerequisites[
                "accepted_gate_record_sha256"
            ],
            "gate_receipt": prerequisites["development_gate_receipt"],
            "pose_manifest": prerequisites["development_pose_manifest"],
            "capture_receipt": prerequisites["capture_receipt"],
            "raw_capture_receipt": prerequisites["raw_capture_receipt"],
            "d1_fixed_observation": prerequisites["d1_fixed_observation"],
            "execution_prerequisites": prerequisites_identity,
            "execution_prerequisites_sha256": prerequisites_sha,
        }

    def test_cell_receipt_metadata_records_no_replay_and_exact_fixture(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prerequisites = self._prerequisites(root, block)
            fixture = self._fixture(root, block, prerequisites)
            path = root / "cell_receipt.json"
            with development.configured_pilot(block, "wmf-forecast-0912-worker-05"):
                with development.installed_receipt_metadata(block, fixture=fixture):
                    pilot.immutable_json(
                        path,
                        {
                            "schema_version": development.CELL_RECEIPT_SCHEMA,
                            "candidate_id": fixture["candidate_id"],
                            "accepted_gate_record_sha256": fixture[
                                "accepted_gate_record_sha256"
                            ],
                        },
                    )
            value = json.loads(path.read_text())
        self.assertEqual(value["phase"], "development")
        self.assertEqual(value["environment_seed"], 2026091101)
        self.assertEqual(value["development_fixture"], fixture)
        self.assertFalse(value["transport_contract"]["stateful_request_replay"])
        self.assertEqual(
            value["claim_boundary"].split(":", 1)[1].strip().split()[0:4],
            ["450", "actual", "actions,", "451"],
        )

    def test_server_ready_is_cross_bound_to_the_same_prerequisite_bundle(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D01")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prerequisites = self._prerequisites(root / "evidence", block)
            expected_sha = development.execution_prerequisites_sha256(
                block, prerequisites
            )
            ready_path = root / "server_ready.json"
            with development.configured_pilot(
                block, "wmf-forecast-0912-worker-05"
            ):
                with development.installed_receipt_metadata(
                    block, prerequisites=prerequisites
                ):
                    pilot.immutable_json(
                        ready_path,
                        {
                            "schema_version": pilot.SERVER_READY_SCHEMA,
                            "status": "ready",
                        },
                    )
            ready = json.loads(ready_path.read_text())
            self.assertEqual(
                ready["development_prerequisites_sha256"], expected_sha
            )
            with mock.patch.object(
                development,
                "_PILOT_VALIDATE_SERVER_READY",
                return_value={"ready": ready},
            ):
                development.validate_server_ready_prerequisites(
                    expected_prerequisites_sha256=expected_sha
                )
                with self.assertRaisesRegex(
                    pilot.D1BehavioralPilotError,
                    "development_server_simulator_prerequisites_changed",
                ):
                    development.validate_server_ready_prerequisites(
                        expected_prerequisites_sha256="0" * 64
                    )

    def test_contiguous_prefix_is_reused_and_fixture_bound(self) -> None:
        block = development.load_development_block(SOURCE_ROOT, "D02")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prerequisites = self._prerequisites(root / "evidence", block)
            fixture = self._fixture(root / "prefix", block, prerequisites)
            receipts = [{"development_fixture": fixture} for _ in range(2)]
            identities = [{"path": f"/cell/{index}", "sha256": "a" * 64} for index in range(2)]
            with mock.patch.object(
                development,
                "_PILOT_DISCOVER_COMPLETED_PREFIX",
                return_value=(
                    receipts,
                    identities,
                    {
                        "completed_prefix_cell_ids": list(block.cell_ids[:2]),
                        "start_cell_index": 2,
                    },
                ),
            ), mock.patch.object(
                development,
                "validate_passed_development_cell",
            ):
                observed = development.discover_completed_prefix(
                    root,
                    study_commit=STUDY_COMMIT,
                    block=block,
                    prerequisites=prerequisites,
                    simulator_worker_role="wmf-forecast-0912-worker-05",
                )
            self.assertEqual(observed[2]["start_cell_index"], 2)
            self.assertEqual(observed[2]["schema_version"], development.RESUME_SCHEMA)

            changed = dict(fixture)
            changed["candidate_id"] = "D02__candidate_01"
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError,
                "development_cell_fixture_binding_changed",
            ):
                development.validate_cells_bind_fixture(
                    [{"development_fixture": changed}],
                    prerequisites=prerequisites,
                    block=block,
                )

            changed_prerequisites = json.loads(json.dumps(prerequisites))
            changed_prerequisites["p00_paired_pilot"]["server_receipt"] = _artifact(
                root / "changed-p00-server.json", b"changed-p00-server"
            )
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError,
                "development_cell_prerequisite_binding_changed",
            ):
                development.validate_cells_bind_fixture(
                    [{"development_fixture": fixture}],
                    prerequisites=changed_prerequisites,
                    block=block,
                )


if __name__ == "__main__":
    unittest.main()
