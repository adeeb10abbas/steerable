from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np


WORKSHOP = Path(__file__).resolve().parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))
import d1_behavioral_pilot_jobs as pilot  # noqa: E402
import recording_adapter as recording  # noqa: E402


STUDY_COMMIT = "a" * 40


def _artifact(path: Path, payload: bytes = b"evidence") -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    identity = pilot.file_identity(path)
    return {"path": identity["path"], "file_sha256": identity["sha256"], "bytes": identity["bytes"]}


def _mapping(root: Path, name: str) -> dict:
    entry = _artifact(root / f"{name}.bin", name.encode())
    entry.update({"key": name, "kind": "json_value"})
    return {"entry_count": 1, "entries": [entry], "content_sha256": "b" * 64}


def _temporal_snapshot(*, empty: bool, frame: int) -> dict:
    return {
        "current_start_frame": frame,
        "fields": {
            field: {"is_none": empty}
            for field in pilot.RESET_FIELDS_TO_NONE
        },
    }


def _rank_metrics() -> list[dict]:
    return [
        {
            "rank": rank,
            "wall_seconds": 1.0,
            "temporal_before": _temporal_snapshot(empty=True, frame=0),
            "temporal_after": _temporal_snapshot(empty=False, frame=3),
            "cache_reinitialization": {
                "_create_kv_caches": [{}],
                "_create_crossattn_caches": [{}],
            },
        }
        for rank in (0, 1)
    ]


def _reset(control: dict) -> dict:
    reset_id = "reset-000001-test"
    ranks = []
    for rank in (0, 1):
        row = {
            "schema_version": pilot.D1_RESET_SCHEMA,
            "reset_id": reset_id,
            "rank": rank,
            "status": "passed",
            "before": _temporal_snapshot(empty=False, frame=9),
            "after": _temporal_snapshot(empty=True, frame=0),
            "fields_cleared": ["current_start_frame", *pilot.RESET_FIELDS_TO_NONE],
            "failures": [],
        }
        if rank == 0:
            row["wrapper_after"] = {
                "frame_buffer_lengths": {"exterior": 0, "wrist": 0},
                "call_count": 0,
                "is_first_call": True,
                "video_across_time_count": 0,
                "current_session_id": None,
            }
        ranks.append(row)
    return {
        "schema_version": pilot.D1_RESET_SCHEMA,
        "reset_id": reset_id,
        "status": "passed",
        "world_size": 2,
        "rank_receipts": ranks,
        "control": control,
    }


class ScientificContractTests(unittest.TestCase):
    def test_transport_disables_protocol_pings_and_stateful_replay(self) -> None:
        connection = object()
        connect = mock.Mock(return_value=connection)
        self.assertIs(
            pilot.connect_websocket_without_keepalive(
                connect, uri="ws://d1.test:18021", auth_headers={"Authorization": "test"}
            ),
            connection,
        )
        connect.assert_called_once_with(
            "ws://d1.test:18021",
            additional_headers={"Authorization": "test"},
            compression=None,
            max_size=None,
            open_timeout=pilot.D1_CONNECT_TIMEOUT_SECONDS,
            ping_interval=None,
            ping_timeout=None,
        )
        self.assertEqual(
            pilot.PILOT_CONTRACT["transport"],
            {
                "compression": None,
                "max_size": None,
                "ping_interval": None,
                "ping_timeout": None,
                "stateful_request_replay": False,
            },
        )

    def test_exact_p00_counts_order_and_fixed_noise(self) -> None:
        self.assertEqual(pilot.REQUEST_COUNT, 57)
        self.assertEqual(pilot.FINAL_EXECUTED_ACTIONS, 2)
        self.assertEqual(pilot.RETURNED_ACTION_HORIZON, 24)
        self.assertEqual(pilot.EXECUTED_PREFIX_HORIZON, 8)
        self.assertEqual(len(pilot.CELL_IDS) * pilot.REQUEST_COUNT, 228)
        self.assertEqual(len(pilot.CELL_IDS) * pilot.ACTION_CAP, 1800)
        self.assertEqual(
            [f"{arm}-{command}" for arm, command, _task in pilot.CONDITIONS],
            ["reflected-right", "reflected-left", "original-left", "original-right"],
        )
        self.assertEqual(pilot.EFFECTIVE_MODEL_NOISE_SEED, 1140)
        self.assertFalse(pilot.PILOT_CONTRACT["custom_s2_used"])
        self.assertFalse(pilot.PILOT_CONTRACT["patched_s1_used"])
        self.assertEqual(
            pilot.PILOT_CONTRACT["official_path"],
            "GrootSimPolicy.lazy_joint_forward_causal",
        )

    def test_schedule_matches_the_frozen_indivisible_d1_row(self) -> None:
        schedule = pilot.validate_schedule(WORKSHOP.parents[1])
        self.assertEqual(schedule["row"]["ordered_cell_ids"], list(pilot.CELL_IDS))
        self.assertTrue(
            schedule["row"]["execution_contract"]["no_global_DreamZero_context_interleaving"]
        )

    def test_official_qualification_payload_is_fail_closed(self) -> None:
        receipt = {
            "schema_version": pilot.D1_QUALIFICATION_SCHEMA,
            "status": "finished",
            "decision": "qualified",
            "exit_code": 0,
            "reason": None,
            "configuration_id": "D1",
            "generation_request_count": 6,
            "behavioral_episode_count": 0,
            "probe": {
                "status": "passed",
                "passed": True,
                "generation_request_count": 6,
                "behavioral_episode_count": 0,
                "failed_checks": [],
            },
            "server_contract": {
                "official_repository_commit": pilot.D1_SOURCE_COMMIT,
                "official_repository_tree": pilot.D1_SOURCE_TREE,
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "world_size": 2,
                "source_aggregate_sha256": pilot.D1_SOURCE_AGGREGATE_SHA256,
                "checkpoint_aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
                "tokenizer_aggregate_sha256": pilot.TOKENIZER_AGGREGATE_SHA256,
            },
        }
        pilot.validate_d1_qualification_payload(receipt)
        receipt["server_contract"]["patched_s1_used"] = True
        with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "d1_qualification_identity_changed"):
            pilot.validate_d1_qualification_payload(receipt)


class HandshakeAndLeaseTests(unittest.TestCase):
    def test_roles_service_and_cli_are_separate(self) -> None:
        parser = pilot.build_parser()
        action = next(item for item in parser._actions if item.dest == "mode")
        self.assertEqual(set(action.choices), {"server-job", "simulator-job", "cell"})
        server_dests = {item.dest for item in action.choices["server-job"]._actions}
        simulator_dests = {item.dest for item in action.choices["simulator-job"]._actions}
        self.assertIn("simulator_job_id", server_dests)
        self.assertIn("server_ready_sha256", simulator_dests)
        self.assertIn("d1_qualification_receipt_sha256", server_dests & simulator_dests)
        self.assertEqual(pilot.SERVER_QUEUE_ROLE, "d1")
        self.assertEqual(pilot.SIMULATOR_QUEUE_ROLE, "wmf-forecast-0912-worker-00")
        self.assertEqual((pilot.SERVICE_HOST, pilot.SERVICE_PORT), ("wmf-forecast-0912-d1", 18021))

    def test_cell_command_uses_worker_robolab_and_exact_service(self) -> None:
        command = pilot.build_cell_command(
            source_root=WORKSHOP.parents[1],
            attempt_root=pilot.RAW_ROOT / "simulator_attempts/sim-1",
            study_commit=STUDY_COMMIT,
            gate_receipt=Path("/tmp/gate.json"), gate_receipt_sha256="b" * 64,
            pose_manifest=Path("/tmp/pose.json"), pose_manifest_sha256="c" * 64,
            run_id="run-1", server_job_id="server-1", server_ready_sha256="d" * 64,
            simulator_claim_sha256="e" * 64, lease_token="lease-token",
            future_root=pilot.RAW_ROOT / "server_attempts/server-1/future",
            condition_index=3,
        )
        self.assertEqual(command[0], str(pilot.ROBOLAB_PYTHON))
        self.assertEqual(command[command.index("--remote-host") + 1], pilot.SERVICE_HOST)
        self.assertEqual(command[command.index("--remote-port") + 1], "18021")
        self.assertEqual(command[command.index("--layout-arm") + 1], "original")
        self.assertEqual(command[command.index("--command") + 1], "right")

    def test_lease_rejects_stale_or_wrong_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "lease.json"
            value = {
                "schema_version": pilot.SERVER_LEASE_SCHEMA,
                "status": "live",
                "run_id": "run-1",
                "owner_job_id": "server-1",
                "server_ready_sha256": "a" * 64,
                "simulator_claim_sha256": None,
                "lease_token": None,
                "generation": 2,
                "heartbeat_unix_ns": time.time_ns(),
                "expires_unix_ns": time.time_ns() + 10**9,
            }
            pilot.replace_json(path, value)
            pilot.validate_live_lease(
                path, schema=pilot.SERVER_LEASE_SCHEMA, owner_job_id="server-1",
                run_id="run-1", server_ready_sha256="a" * 64,
            )
            with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "lease_identity_mismatch"):
                pilot.validate_live_lease(
                    path, schema=pilot.SERVER_LEASE_SCHEMA, owner_job_id="other",
                    run_id="run-1", server_ready_sha256="a" * 64,
                )
            with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "lease_expired"):
                pilot.validate_live_lease(
                    path, schema=pilot.SERVER_LEASE_SCHEMA, owner_job_id="server-1",
                    run_id="run-1", server_ready_sha256="a" * 64,
                    now_ns=value["expires_unix_ns"],
                )

    def test_simulator_claim_is_bound_to_ready_source_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "claim.json"
            claim = {
                "schema_version": pilot.SIMULATOR_CLAIM_SCHEMA,
                "status": "claimed",
                "run_id": "run-1",
                "simulator_job_id": "sim-1",
                "server_job_id": "server-1",
                "server_ready_sha256": "a" * 64,
                "study_commit": STUDY_COMMIT,
                "block_id": pilot.BLOCK_ID,
                "worker_role": pilot.SIMULATOR_QUEUE_ROLE,
                "pilot_contract_sha256": pilot.PILOT_CONTRACT_SHA256,
                "lease_token": "lease-token",
                "start_cell_index": 0,
            }
            pilot.immutable_json(path, claim)
            _, identity = pilot.validate_simulator_claim(
                path, run_id="run-1", simulator_job_id="sim-1",
                server_job_id="server-1", server_ready_sha256="a" * 64,
                study_commit=STUDY_COMMIT,
            )
            self.assertEqual(identity["sha256"], pilot.sha256_file(path))
            claim["server_ready_sha256"] = "b" * 64
            path2 = Path(temporary) / "bad.json"
            pilot.immutable_json(path2, claim)
            with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "simulator_claim_mismatch"):
                pilot.validate_simulator_claim(
                    path2, run_id="run-1", simulator_job_id="sim-1",
                    server_job_id="server-1", server_ready_sha256="a" * 64,
                    study_commit=STUDY_COMMIT,
                )

    def test_supervised_child_reaps_group_when_receipt_write_fails(self) -> None:
        process = mock.Mock()
        stdout = mock.Mock()
        stderr = mock.Mock()

        def launch(*_args, **_kwargs):
            pilot._ACTIVE_CHILDREN.append(process)
            return process, stdout, stderr

        with mock.patch.object(pilot, "_launch_logged", side_effect=launch), mock.patch.object(
            pilot, "terminate_process_group"
        ) as terminate, mock.patch.object(pilot, "_close_logs") as close:
            with self.assertRaisesRegex(OSError, "receipt failed"):
                with pilot.supervised_logged_child(
                    ["child"], cwd=Path("/tmp"), environment={},
                    stdout_path=Path("/tmp/out"), stderr_path=Path("/tmp/err"),
                ):
                    raise OSError("receipt failed")
        terminate.assert_called_once_with(process)
        close.assert_called_once_with(stdout, stderr)
        self.assertNotIn(process, pilot._ACTIVE_CHILDREN)


class ServerEvidenceTests(unittest.TestCase):
    def test_two_rank_reset_scan_requires_every_temporal_field_cleared(self) -> None:
        control = {"episode_id": "episode-1", "expected_session_id": "session-1"}
        reset = _reset(control)
        scan = pilot._validate_temporal_reset(reset, expected_control=control)
        self.assertTrue(scan["passed"])
        self.assertEqual([row["rank"] for row in scan["rank_temporal_state_scan"]], [0, 1])
        reset["rank_receipts"][1]["after"]["fields"]["kv_cache1"]["is_none"] = False
        with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "d1_temporal_field_not_cleared"):
            pilot._validate_temporal_reset(reset, expected_control=control)

    def test_validated_d1_reset_populates_recording_context_attestation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            future = Path(temporary) / "future"
            episode_id = "episode-1"
            control = {"episode_id": episode_id, "expected_session_id": "session-1"}
            reset_path = future / "episodes" / episode_id / "reset_receipt.json"
            reset_path.parent.mkdir(parents=True)
            pilot.immutable_json(reset_path, _reset(control))
            reset, reset_identity, reset_scan = pilot.validate_reset_receipt(
                reset_path, expected_control=control, future_root=future
            )
            attestation = pilot.build_context_reset_attestation(
                reset=reset,
                reset_identity=reset_identity,
                reset_scan=reset_scan,
                episode_id=episode_id,
            )
            self.assertEqual(
                set(attestation),
                {"passed", "reset_scope", "server_context_id", "cache_reset_evidence"},
            )
            self.assertEqual(attestation["reset_scope"], recording.CONTEXT_RESET_SCOPE)
            self.assertEqual(attestation["server_context_id"], episode_id)
            self.assertEqual(
                attestation["cache_reset_evidence"]["server_reset_receipt"],
                reset_identity,
            )
            self.assertEqual(
                [row["rank"] for row in attestation["cache_reset_evidence"]["rank_temporal_state_scan"]],
                [0, 1],
            )
            recorder = recording.ForecastRecordingAdapter(
                Path(temporary) / "recording",
                {
                    "attempt_id": "attempt-reset-test",
                    "cell_id": pilot.CELL_IDS[0],
                    "stage": "pilot",
                    "layout_pair_id": "P00",
                    "layout_arm": "reflected",
                    "command": "right",
                    "prompt": pilot.PROMPTS["right"],
                    "model_config": "D1",
                    "effective_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
                    "source_identity": "source-test",
                    "checkpoint_identity": "checkpoint-test",
                },
            )
            recorder.record_context_reset(attestation)
            self.assertEqual(
                recorder.context_reset["receipt"]["server_context_id"], episode_id
            )

            changed_scan = dict(reset_scan)
            changed_scan["reset_id"] = "different-reset"
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError, "d1_context_reset_scan_invalid"
            ):
                pilot.build_context_reset_attestation(
                    reset=reset,
                    reset_identity=reset_identity,
                    reset_scan=changed_scan,
                    episode_id=episode_id,
                )

    def test_request_receipt_binds_wire_action_latent_decode_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            future = Path(temporary) / "future"
            episode_id = "episode-1"
            request_root = future / "episodes" / episode_id / "request_0000"
            request_root.mkdir(parents=True)
            action = np.arange(24 * 8, dtype=np.float32).reshape(24, 8)
            action_path = request_root / "action.npy"
            np.save(action_path, action, allow_pickle=False)
            action_entry = {
                "path": str(action_path),
                "file_sha256": pilot.sha256_file(action_path),
                "bytes": action_path.stat().st_size,
                "shape": [24, 8],
                "dtype": "float32",
                "data_sha256": "c" * 64,
            }
            latent = _artifact(request_root / "latent.pt", b"latent")
            latent.update({"shape": [1, 4, 4], "dtype": "torch.float32", "data_sha256": "d" * 64})
            decoded_tensor = _artifact(request_root / "decoded.pt", b"decoded")
            decoded_rgb = _artifact(request_root / "decoded.npy", b"rgb")
            receipt = {
                "schema_version": pilot.D1_REQUEST_SCHEMA,
                "configuration_id": "D1",
                "episode_id": episode_id,
                "request_index": 0,
                "probe_id": None,
                "prompt": pilot.PROMPTS["right"],
                "session_id": "session-1",
                "effective_official_model_noise_seed": 1140,
                "noise_semantics": "fixed; this request is not an independent noise draw",
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "official_forward_call_count": 1,
                "raw_inputs": _mapping(request_root, "raw"),
                "converted_inputs": _mapping(request_root, "converted"),
                "normalized_model_inputs": _mapping(request_root, "normalized"),
                "official_returned_action": action_entry,
                "latent_video": latent,
                "offline_decode": {
                    "requested": True,
                    "performed": True,
                    "latent_data_sha256_before": latent["data_sha256"],
                    "latent_data_sha256_after": latent["data_sha256"],
                    "decoded_tensor": decoded_tensor,
                    "decoded_rgb": decoded_rgb,
                },
                "temporal_and_cache_rank_metrics": _rank_metrics(),
                "cost": {"rank_count": 2, "inference_wall_seconds_rank0_wrapper": 1.0},
                "measurement_control": {
                    "probe_id": None,
                    "offline_decode": True,
                    "probe_plan_sha256": pilot.PILOT_CONTRACT_SHA256,
                },
            }
            path = request_root / "request_receipt.json"
            pilot.immutable_json(path, receipt)
            real_import = __import__

            def reject_numpy(name, *args, **kwargs):
                if name == "numpy" or name.startswith("numpy."):
                    raise ModuleNotFoundError("NumPy deliberately unavailable")
                return real_import(name, *args, **kwargs)

            with mock.patch("builtins.__import__", side_effect=reject_numpy):
                _, parent_identity = pilot.validate_request_receipt(
                    path, future_root=future, episode_id=episode_id,
                    session_id="session-1", prompt=pilot.PROMPTS["right"],
                    request_index=0, server_contract_sha256="e" * 64,
                )
            self.assertEqual(parent_identity["sha256"], pilot.sha256_file(path))
            _, identity = pilot.validate_request_receipt(
                path, future_root=future, episode_id=episode_id,
                session_id="session-1", prompt=pilot.PROMPTS["right"],
                request_index=0, server_contract_sha256="e" * 64,
                returned_action=action.copy(),
            )
            self.assertEqual(identity["sha256"], pilot.sha256_file(path))
            with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "d1_wire_action_differs"):
                pilot.validate_request_receipt(
                    path, future_root=future, episode_id=episode_id,
                    session_id="session-1", prompt=pilot.PROMPTS["right"],
                    request_index=0, server_contract_sha256="e" * 64,
                    returned_action=action + 1,
                )

    def test_dependency_free_npy_parser_rejects_unsafe_action_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = np.arange(24 * 8, dtype=np.float32).reshape(24, 8)
            valid_path = root / "valid.npy"
            np.save(valid_path, valid, allow_pickle=False)

            shape, values = pilot._validate_persisted_float32_action_npy(valid_path)
            self.assertEqual(shape, (24, 8))
            self.assertEqual(len(values), 24 * 8)

            cases = {
                "float64": np.arange(24 * 8, dtype=np.float64).reshape(24, 8),
                "big-endian-float32": valid.astype(">f4"),
                "wrong-shape": np.arange(24 * 8, dtype=np.float32).reshape(12, 16),
                "nan": valid.copy(),
                "inf": valid.copy(),
                "fortran": np.asfortranarray(valid),
            }
            cases["nan"][0, 0] = np.nan
            cases["inf"][0, 0] = np.inf
            for name, array in cases.items():
                with self.subTest(name=name):
                    candidate = root / f"{name}.npy"
                    np.save(candidate, array, allow_pickle=False)
                    reason = (
                        "d1_persisted_action_shape_changed"
                        if name == "wrong-shape"
                        else "d1_persisted_action_invalid"
                    )
                    with self.assertRaisesRegex(pilot.D1BehavioralPilotError, reason):
                        pilot._validate_persisted_float32_action_npy(candidate)

            truncated = root / "truncated.npy"
            truncated.write_bytes(valid_path.read_bytes()[:-1])
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError, "d1_persisted_action_invalid"
            ):
                pilot._validate_persisted_float32_action_npy(truncated)

            trailing = root / "trailing.npy"
            trailing.write_bytes(valid_path.read_bytes() + b"unexpected")
            with self.assertRaisesRegex(
                pilot.D1BehavioralPilotError, "d1_persisted_action_invalid"
            ):
                pilot._validate_persisted_float32_action_npy(trailing)

    def test_runner_never_calls_uninstrumented_dreamzero_reset(self) -> None:
        source = (FORECAST / "d1_behavioral_pilot_jobs.py").read_text()
        self.assertIn("from policies.dreamzero.client import DreamZeroClient", source)
        self.assertIn("InferenceClient.reset(self, env_id=env_id)", source)
        self.assertNotIn("DreamZeroClient.reset(self", source)
        self.assertIn('"wmf_d1_measurement"', source)
        self.assertIn('"offline_decode": True', source)
        self.assertIn("d1_stateful_transport_ambiguous_no_retry", source)
        self.assertNotIn("raw = super()._query_server(request)", source)


class ResumeTests(unittest.TestCase):
    def _fake_cell(self, raw: Path, attempt: str, index: int) -> Path:
        path = raw / "simulator_attempts" / attempt / "cells" / f"{index:02d}-cell" / "cell_receipt.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"condition_index": index}))
        return path

    def _validator(self, path: Path, *, condition_index: int, study_commit: str):
        receipt = {
            "cell_id": pilot.CELL_IDS[condition_index],
            "condition_index": condition_index,
            "source_pins": {"study_commit": "b" * 40},
            "server_begin_receipt": {
                "episode_context_id": f"episode-{condition_index}",
                "client_session_id": f"session-{condition_index}",
            },
        }
        return receipt, pilot.file_identity(path)

    def test_contiguous_prefix_resumes_across_source_fix_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            self._fake_cell(raw, "attempt-a", 0)
            self._fake_cell(raw, "attempt-b", 1)
            with mock.patch.object(pilot, "validate_passed_cell_receipt", side_effect=self._validator):
                receipts, identities, provenance = pilot.discover_completed_prefix(
                    raw, study_commit=STUDY_COMMIT
                )
        self.assertEqual([row["cell_id"] for row in receipts], list(pilot.CELL_IDS[:2]))
        self.assertEqual(len(identities), 2)
        self.assertEqual(provenance["start_cell_index"], 2)
        self.assertEqual(provenance["prior_study_commits"], ["b" * 40])

    def test_noncontiguous_and_duplicate_passes_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            self._fake_cell(raw, "attempt-a", 1)
            with mock.patch.object(pilot, "validate_passed_cell_receipt", side_effect=self._validator):
                with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "not_contiguous"):
                    pilot.discover_completed_prefix(raw, study_commit=STUDY_COMMIT)
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            self._fake_cell(raw, "attempt-a", 0)
            self._fake_cell(raw, "attempt-b", 0)
            with mock.patch.object(pilot, "validate_passed_cell_receipt", side_effect=self._validator):
                with self.assertRaisesRegex(pilot.D1BehavioralPilotError, "duplicate_passed_cell"):
                    pilot.discover_completed_prefix(raw, study_commit=STUDY_COMMIT)


if __name__ == "__main__":
    unittest.main()
