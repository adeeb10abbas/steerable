from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


WORKSHOP = Path(__file__).resolve().parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))
import n3_behavioral_pilot_job as pilot  # noqa: E402


STUDY_COMMIT = "a" * 40


def _make_passed_cell(raw_root: Path, attempt_name: str, index: int) -> Path:
    attempt = raw_root / attempt_name
    cell = attempt / "cells" / f"{index:02d}-cell"
    artifacts = cell / "artifacts"
    artifacts.mkdir(parents=True)
    descriptors = {}
    for name in ("adapter_completion", "adapter_journal", "native_timing_support"):
        path = artifacts / f"{name}.json"
        pilot.immutable_json(path, {"name": name, "index": index})
        descriptors[name] = pilot.file_identity(path)
    video = artifacts / "viewport.mp4"
    video.write_bytes(b"representative-video")
    context = f"server-process:cell-{index:02d}:unique"
    arm, command, _task = pilot.CONDITIONS[index]
    receipt = {
        "schema_version": pilot.CELL_RECEIPT_SCHEMA,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "block_id": pilot.BLOCK_ID,
        "cell_id": pilot.CELL_IDS[index],
        "condition_index": index,
        "layout_pair_id": pilot.LAYOUT_PAIR_ID,
        "layout_arm": arm,
        "command": command,
        "prompt": pilot.PROMPTS[command],
        "model_config": pilot.MODEL_CONFIG,
        "effective_seed": pilot.EFFECTIVE_SEED,
        "actions_executed": pilot.ACTION_CAP,
        "observation_count": pilot.OBSERVATION_COUNT,
        "behavioral_model_request_count": pilot.REQUEST_COUNT,
        "behavioral_episode_count": 1,
        "generation_qualification_request_count": 0,
        "final_chunk_executed_actions": pilot.FINAL_EXECUTED_ACTIONS,
        "source_pins": {
            "study_commit": STUDY_COMMIT,
            "robolab_commit": pilot.ROBOLAB_COMMIT,
            "cosmos_commit": pilot.COSMOS_COMMIT,
        },
        "checkpoint_pin": {
            "revision": pilot.CHECKPOINT_REVISION,
            "aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
        },
        **descriptors,
        "viewport_video": pilot.file_identity(video),
        "server_begin_receipt": {
            "passed": True,
            "server_context_id": context,
            "cache_reset_evidence": {
                "passed": True,
                "episode_context_id": context,
                "unresolved_mutable_temporal_fields": [],
            },
        },
        "server_end_receipt": {
            "passed": True,
            "status": "completed",
            "cell_id": pilot.CELL_IDS[index],
            "condition_index": index,
            "server_context_id": context,
            "server_request_count": pilot.REQUEST_COUNT,
            "client_request_count": pilot.REQUEST_COUNT,
            "actions_executed": pilot.ACTION_CAP,
        },
    }
    path = cell / "cell_receipt.json"
    pilot.immutable_json(path, receipt)
    return path


def _begin(index: int, context: str = "context-1") -> tuple[pilot.ServerProtocol, dict]:
    protocol = pilot.ServerProtocol(start_cell_index=index)
    arm, command, _task = pilot.CONDITIONS[index]
    response = protocol.begin(
        {
            "study_id": pilot.STUDY_ID,
            "block_id": pilot.BLOCK_ID,
            "cell_id": pilot.CELL_IDS[index],
            "condition_index": index,
            "layout_arm": arm,
            "command": command,
            "prompt": pilot.PROMPTS[command],
            "effective_seed": pilot.EFFECTIVE_SEED,
            "expected_actions": pilot.ACTION_CAP,
            "expected_requests": pilot.REQUEST_COUNT,
            "client_session_id": f"client-session-{index}",
        },
        server_context_id=context,
        temporal_reset_evidence={
            "passed": True,
            "unresolved_mutable_temporal_fields": [],
        },
    )
    return protocol, response


def _request(protocol: pilot.ServerProtocol, context: str = "context-1") -> dict:
    active = protocol.active
    assert active is not None
    index = active["request_count"]
    return {
        "wmf_request_type": "behavioral",
        "study_id": pilot.STUDY_ID,
        "block_id": pilot.BLOCK_ID,
        "cell_id": active["cell_id"],
        "condition_index": active["condition_index"],
        "layout_arm": active["layout_arm"],
        "command": active["command"],
        "prompt": active["prompt"],
        "sampling_seed": pilot.EFFECTIVE_SEED,
        "effective_seed": pilot.EFFECTIVE_SEED,
        "request_index": index,
        "action_step_start": index * pilot.ACTION_HORIZON,
        "server_context_id": context,
        "client_session_id": active["client_session_id"],
        "observation/image": object(),
        "observation/joint_position": object(),
        "observation/gripper_position": object(),
    }


class ServerProtocolTests(unittest.TestCase):
    def test_exact_four_cell_block_is_60_requests_and_1800_actions(self) -> None:
        protocol = pilot.ServerProtocol()
        requests = 0
        for condition_index, (arm, command, _task) in enumerate(pilot.CONDITIONS):
            response = protocol.begin(
                {
                    "study_id": pilot.STUDY_ID,
                    "block_id": pilot.BLOCK_ID,
                    "cell_id": pilot.CELL_IDS[condition_index],
                    "condition_index": condition_index,
                    "layout_arm": arm,
                    "command": command,
                    "prompt": pilot.PROMPTS[command],
                    "effective_seed": pilot.EFFECTIVE_SEED,
                    "expected_actions": 450,
                    "expected_requests": 15,
                    "client_session_id": f"client-session-{condition_index}",
                },
                server_context_id=f"context-{condition_index}",
                temporal_reset_evidence={
                    "passed": True,
                    "unresolved_mutable_temporal_fields": [],
                },
            )
            self.assertEqual(response["reset_scope"], pilot.CONTEXT_RESET_SCOPE)
            for _ in range(15):
                expected = protocol.validate_behavioral(
                    _request(protocol, context=f"context-{condition_index}")
                )
                self.assertEqual(expected["action_step_start"], (requests % 15) * 32)
                protocol.complete_behavioral()
                requests += 1
            end = protocol.end(
                {
                    "study_id": pilot.STUDY_ID,
                    "block_id": pilot.BLOCK_ID,
                    "cell_id": pilot.CELL_IDS[condition_index],
                    "condition_index": condition_index,
                    "server_context_id": f"context-{condition_index}",
                    "client_session_id": f"client-session-{condition_index}",
                    "status": "completed",
                    "stop_reason": "action_cap",
                    "actions_executed": 450,
                    "request_count": 15,
                    "final_chunk_executed_actions": 2,
                }
            )
            self.assertTrue(end["passed"])
        self.assertEqual(requests, 60)
        self.assertEqual(requests * 32 - 4 * 30, 1800)
        self.assertEqual(protocol.completed_cells, list(pilot.CELL_IDS))

    def test_request_order_seed_and_context_fail_closed(self) -> None:
        protocol, _ = _begin(0)
        row = _request(protocol)
        row["action_step_start"] = 1
        with self.assertRaisesRegex(pilot.N3BehavioralPilotError, "server_behavioral_request_mismatch"):
            protocol.validate_behavioral(row)
        row = _request(protocol)
        row["sampling_seed"] += 1
        with self.assertRaises(pilot.N3BehavioralPilotError):
            protocol.validate_behavioral(row)
        row = _request(protocol)
        row["server_context_id"] = "other-context"
        with self.assertRaises(pilot.N3BehavioralPilotError):
            protocol.validate_behavioral(row)

    def test_invalid_end_does_not_advance_block(self) -> None:
        protocol, _ = _begin(0)
        for _ in range(15):
            protocol.validate_behavioral(_request(protocol))
            protocol.complete_behavioral()
        end = protocol.end(
            {
                "study_id": pilot.STUDY_ID,
                "block_id": pilot.BLOCK_ID,
                "cell_id": pilot.CELL_IDS[0],
                "condition_index": 0,
                "server_context_id": "context-1",
                "client_session_id": "client-session-0",
                "status": "completed",
                "stop_reason": "action_cap",
                "actions_executed": 449,
                "request_count": 15,
                "final_chunk_executed_actions": 1,
            }
        )
        self.assertFalse(end["passed"])
        self.assertEqual(protocol.next_cell_index, 0)

    def test_end_rejects_contradictory_counts_and_fabricated_safety_prefix(self) -> None:
        protocol, _ = _begin(0)
        protocol.validate_behavioral(_request(protocol))
        protocol.complete_behavioral()
        base = {
            "study_id": pilot.STUDY_ID,
            "block_id": pilot.BLOCK_ID,
            "cell_id": pilot.CELL_IDS[0],
            "condition_index": 0,
            "server_context_id": "context-1",
            "client_session_id": "client-session-0",
            "status": "safety_abort",
            "stop_reason": "safety_abort",
            "actions_executed": 1,
            "request_count": 1,
            "final_chunk_executed_actions": None,
        }
        with self.assertRaisesRegex(
            pilot.N3BehavioralPilotError, "server_end_mismatch: request_count"
        ):
            protocol.end({**base, "request_count": 0})
        self.assertIsNotNone(protocol.active)
        with self.assertRaisesRegex(
            pilot.N3BehavioralPilotError, "server_end_mismatch: safety_abort_prefix"
        ):
            protocol.end({**base, "actions_executed": 31})
        self.assertIsNotNone(protocol.active)

    def test_terminal_receipt_survives_lost_reply_and_rejects_tampering(self) -> None:
        protocol, _ = _begin(0)
        protocol.validate_behavioral(_request(protocol))
        protocol.complete_behavioral()
        end = protocol.end(
            {
                "study_id": pilot.STUDY_ID,
                "block_id": pilot.BLOCK_ID,
                "cell_id": pilot.CELL_IDS[0],
                "condition_index": 0,
                "server_context_id": "context-1",
                "client_session_id": "client-session-0",
                "status": "safety_abort",
                "stop_reason": "safety_abort",
                "actions_executed": 17,
                "request_count": 1,
                "final_chunk_executed_actions": None,
            }
        )
        self.assertIsNone(protocol.active)
        with tempfile.TemporaryDirectory() as temporary:
            attempt = Path(temporary)
            descriptor = pilot.persist_server_terminal_receipt(
                attempt_root=attempt,
                end_response=end,
                protocol_context_active=protocol.active is not None,
                model_capture_active=False,
            )
            terminal_path = pilot.server_terminal_path(attempt, pilot.CELL_IDS[0])
            # Recovery uses the canonical server artifact directly; it does not
            # need the possibly lost RPC response.
            observed, identity = pilot.validate_server_terminal_receipt(
                pilot.file_identity(terminal_path),
                attempt_root=attempt,
                cell_id=pilot.CELL_IDS[0],
                condition_index=0,
                server_context_id="context-1",
                client_session_id="client-session-0",
                stop_reason="safety_abort",
                actions_executed=17,
                request_count=1,
            )
            self.assertEqual(identity, descriptor)
            self.assertEqual(observed["terminal_state"], "context_closed")
            for label, server_context_id, client_session_id in (
                ("null-server", None, "client-session-0"),
                ("null-client", "context-1", None),
                ("aliased", "context-1", "context-1"),
                ("unsafe-server", "bad/context", "client-session-0"),
                ("unsafe-client", "context-1", "bad/session"),
            ):
                with self.subTest(identity=label), self.assertRaisesRegex(
                    pilot.N3BehavioralPilotError,
                    "server_context_terminal_identity_invalid",
                ):
                    pilot.validate_server_terminal_receipt(
                        descriptor,
                        attempt_root=attempt,
                        cell_id=pilot.CELL_IDS[0],
                        condition_index=0,
                        server_context_id=server_context_id,
                        client_session_id=client_session_id,
                        stop_reason="safety_abort",
                        actions_executed=17,
                        request_count=1,
                    )
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "immutable_evidence_exists"
            ):
                pilot.persist_server_terminal_receipt(
                    attempt_root=attempt,
                    end_response=end,
                    protocol_context_active=False,
                    model_capture_active=False,
                )

            original = json.loads(terminal_path.read_text())
            for label, mutation, reason in (
                (
                    "extra-key",
                    lambda value: value.update({"injected": True}),
                    "server_context_terminal_keys_changed",
                ),
                (
                    "context-id",
                    lambda value: value.update({"server_context_id": "other"}),
                    "server_context_terminal_mismatch",
                ),
                (
                    "request-count",
                    lambda value: value.update({"request_count": 2}),
                    "server_context_terminal_mismatch",
                ),
                (
                    "malformed-time",
                    lambda value: value.update({"completed_at_utc": "not-utc"}),
                    "invalid_utc_timestamp",
                ),
            ):
                with self.subTest(label=label):
                    changed = dict(original)
                    mutation(changed)
                    terminal_path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(pilot.N3BehavioralPilotError, reason):
                        pilot.validate_server_terminal_receipt(
                            pilot.file_identity(terminal_path),
                            attempt_root=attempt,
                            cell_id=pilot.CELL_IDS[0],
                            condition_index=0,
                            server_context_id="context-1",
                            client_session_id="client-session-0",
                            stop_reason="safety_abort",
                            actions_executed=17,
                            request_count=1,
                        )
                    terminal_path.write_text(json.dumps(original), encoding="utf-8")

            wrong_hash = dict(pilot.file_identity(terminal_path))
            wrong_hash["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "file_descriptor_sha_mismatch"
            ):
                pilot.validate_server_terminal_receipt(
                    wrong_hash,
                    attempt_root=attempt,
                    cell_id=pilot.CELL_IDS[0],
                    condition_index=0,
                    server_context_id="context-1",
                    client_session_id="client-session-0",
                    stop_reason="safety_abort",
                    actions_executed=17,
                    request_count=1,
                )

    def test_context_identity_cannot_be_reused_after_a_begin(self) -> None:
        protocol, _ = _begin(0, context="one-use-context")
        protocol.active = None
        with self.assertRaisesRegex(pilot.N3BehavioralPilotError, "server_context_id_reused"):
            arm, command, _task = pilot.CONDITIONS[0]
            protocol.begin(
                {
                    "study_id": pilot.STUDY_ID,
                    "block_id": pilot.BLOCK_ID,
                    "cell_id": pilot.CELL_IDS[0],
                    "condition_index": 0,
                    "layout_arm": arm,
                    "command": command,
                    "prompt": pilot.PROMPTS[command],
                    "effective_seed": pilot.EFFECTIVE_SEED,
                    "expected_actions": pilot.ACTION_CAP,
                    "expected_requests": pilot.REQUEST_COUNT,
                    "client_session_id": "one-use-client-session",
                },
                server_context_id="one-use-context",
                temporal_reset_evidence={
                    "passed": True,
                    "unresolved_mutable_temporal_fields": [],
                },
            )

    def test_temporal_scan_records_configuration_and_rejects_episode_state(self) -> None:
        class Component:
            history_length = 1

            def __init__(self) -> None:
                self.cache_config = {"mode": "static"}
                self.episode_cache = {"frame": 3}
                self._buffers = {"weight": object()}

        inventory = pilot._temporal_field_inventory(Component(), component="test")
        self.assertEqual(
            inventory["unresolved_mutable_temporal_fields"], ["test.episode_cache"]
        )
        classifications = {
            row["name"]: row["classification"] for row in inventory["temporal_fields"]
        }
        self.assertEqual(
            classifications["cache_config"], "nonempty_scanned_configuration_container"
        )
        self.assertEqual(classifications["_buffers"], "pytorch_static_registry")


class LauncherContractTests(unittest.TestCase):
    def test_raw_n3_qualification_uses_nested_source_and_checkpoint_identity(self) -> None:
        raw = {
            "schema_version": pilot.N3_QUALIFICATION_SCHEMA,
            "status": "passed",
            "qualified": True,
            "model_config": pilot.MODEL_CONFIG,
            "effective_seed": pilot.EFFECTIVE_SEED,
            "generation_request_count": 6,
            "robot_episode_count": 0,
            "source": {
                "commit": pilot.COSMOS_COMMIT,
                "git_tree": pilot.COSMOS_GIT_TREE,
            },
            "checkpoint": {
                "revision": pilot.CHECKPOINT_REVISION,
                "payload_aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
            },
        }
        pilot.validate_n3_qualification_payload(raw)
        flattened = dict(raw)
        flattened.pop("source")
        flattened["source_commit"] = pilot.COSMOS_COMMIT
        with self.assertRaisesRegex(pilot.N3BehavioralPilotError, "n3_source_identity_missing"):
            pilot.validate_n3_qualification_payload(flattened)

    def test_schedule_is_exact_p00_indivisible_block(self) -> None:
        receipt = pilot.validate_schedule(WORKSHOP.parents[1])
        self.assertEqual(receipt["row"]["ordered_cell_ids"], list(pilot.CELL_IDS))

    def test_two_gpu_topology_assigns_distinct_b200s(self) -> None:
        devices = [
            {"index": "0", "uuid": "GPU-a", "name": "NVIDIA B200", "driver_version": "1", "memory.total": "1"},
            {"index": "1", "uuid": "GPU-b", "name": "NVIDIA B200", "driver_version": "1", "memory.total": "1"},
        ]
        with mock.patch.object(pilot, "_query_nvidia", side_effect=[devices, []]):
            topology = pilot.verify_two_idle_b200s()
        self.assertEqual(topology["assignment"]["model_child_cuda_visible_devices"], "0")
        self.assertEqual(topology["assignment"]["simulator_child_cuda_visible_devices"], "1")

    def test_any_preexisting_compute_process_rejects_launch(self) -> None:
        devices = [
            {"index": "0", "uuid": "GPU-a", "name": "NVIDIA B200", "driver_version": "1", "memory.total": "1"},
            {"index": "1", "uuid": "GPU-b", "name": "NVIDIA B200", "driver_version": "1", "memory.total": "1"},
        ]
        with mock.patch.object(
            pilot,
            "_query_nvidia",
            side_effect=[devices, [{"gpu_uuid": "GPU-a", "pid": "3", "process_name": "other", "used_memory": "1"}]],
        ):
            with self.assertRaisesRegex(pilot.N3BehavioralPilotError, "preexisting_compute"):
                pilot.verify_two_idle_b200s()

    def test_commands_and_environments_split_the_two_logical_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = pilot.build_model_environment(source_root=WORKSHOP.parents[1], attempt_root=root, base={})
            sim = pilot.build_simulator_environment(
                source_root=WORKSHOP.parents[1], state_parent=root, base={}
            )
        self.assertEqual(model["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(sim["CUDA_VISIBLE_DEVICES"], "1")
        command = pilot.build_cell_command(
            source_root=WORKSHOP.parents[1],
            attempt_root=Path("/tmp/attempt"),
            study_commit="a" * 40,
            gate_receipt=Path("/tmp/gate.json"),
            gate_receipt_sha256="b" * 64,
            pose_manifest=Path("/tmp/pose.json"),
            pose_manifest_sha256="c" * 64,
            port=18011,
            condition_index=3,
        )
        self.assertEqual(command[0], str(pilot.ROBOLAB_PYTHON))
        self.assertEqual(command[command.index("--layout-arm") + 1], "original")
        self.assertEqual(command[command.index("--command") + 1], "right")
        server_command = pilot.build_server_command(
            source_root=WORKSHOP.parents[1],
            attempt_root=Path("/tmp/attempt"),
            study_commit=STUDY_COMMIT,
            port=18011,
            start_cell_index=2,
        )
        self.assertEqual(server_command[server_command.index("--start-cell-index") + 1], "2")

    def test_cli_exposes_exact_prerequisite_receipt_hashes(self) -> None:
        help_text = pilot.build_parser().format_help()
        self.assertIn("queue", help_text)
        queue = next(action for action in pilot.build_parser()._actions if action.dest == "mode")
        queue_parser = queue.choices["queue"]
        destinations = {action.dest for action in queue_parser._actions}
        self.assertTrue(
            {
                "gate_receipt_sha256",
                "pose_manifest_sha256",
                "capture_receipt_sha256",
                "recorder_receipt_sha256",
                "n3_qualification_receipt_sha256",
            }.issubset(destinations)
        )
        self.assertEqual(pilot.QUEUE_ROLE, "n3")

    def test_supervised_child_is_reaped_when_receipt_write_raises(self) -> None:
        process = mock.Mock()
        process.poll.return_value = None
        stdout = mock.Mock()
        stderr = mock.Mock()

        def launch(*_args, **_kwargs):
            pilot._ACTIVE_CHILDREN.append(process)
            return process, stdout, stderr

        with mock.patch.object(pilot, "_launch_logged", side_effect=launch), mock.patch.object(
            pilot, "terminate_process_group"
        ) as terminate, mock.patch.object(pilot, "_close_process_logs") as close:
            with self.assertRaisesRegex(OSError, "receipt failed"):
                with pilot.supervised_logged_child(
                    ["child"],
                    cwd=Path("/tmp"),
                    environment={},
                    stdout_path=Path("/tmp/out"),
                    stderr_path=Path("/tmp/err"),
                ):
                    raise OSError("receipt failed")
        terminate.assert_called_once_with(process)
        close.assert_called_once_with(stdout, stderr)
        self.assertNotIn(process, pilot._ACTIVE_CHILDREN)


class ResumeContractTests(unittest.TestCase):
    def test_hash_verified_contiguous_prefix_resumes_without_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            first = _make_passed_cell(raw, "attempt-a", 0)
            second = _make_passed_cell(raw, "attempt-b", 1)
            receipts, identities, provenance = pilot.discover_completed_prefix(
                raw, study_commit="b" * 40
            )
        self.assertEqual([row["cell_id"] for row in receipts], list(pilot.CELL_IDS[:2]))
        self.assertEqual([row["path"] for row in identities], [str(first), str(second)])
        self.assertEqual(provenance["start_cell_index"], 2)
        self.assertEqual(provenance["prior_study_commits"], [STUDY_COMMIT])

    def test_noncontiguous_prior_pass_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            _make_passed_cell(raw, "attempt-a", 1)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "resume_passed_cells_not_contiguous"
            ):
                pilot.discover_completed_prefix(raw, study_commit=STUDY_COMMIT)

    def test_duplicate_prior_pass_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            _make_passed_cell(raw, "attempt-a", 0)
            _make_passed_cell(raw, "attempt-b", 0)
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "resume_duplicate_passed_cell"
            ):
                pilot.discover_completed_prefix(raw, study_commit=STUDY_COMMIT)

    def test_changed_artifact_breaks_hash_verified_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary)
            receipt_path = _make_passed_cell(raw, "attempt-a", 0)
            receipt = json.loads(receipt_path.read_text())
            Path(receipt["adapter_completion"]["path"]).write_text("changed")
            with self.assertRaisesRegex(
                pilot.N3BehavioralPilotError, "file_descriptor_sha_mismatch"
            ):
                pilot.discover_completed_prefix(raw, study_commit=STUDY_COMMIT)


if __name__ == "__main__":
    unittest.main()
