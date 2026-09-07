"""Tests for fail-closed C2 fresh-session prefix verification."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from experiments.online_correction_v4.prefix_replay import (
    PrefixReplayError,
    verify_common_prefix,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _attempt(root: Path, *, attempt_id: str, scenario: str) -> Path:
    attempt = root / attempt_id
    _write(
        attempt / "episode.json",
        {
            "attempt_id": attempt_id,
            "prefix_group_id": "prefix-1",
            "env_seed": 101,
            "policy_seed": 202,
            "prompt_id": "prompt-1",
            "prompt_sha256": SHA_A,
            "policy_id": "cosmos3_nano_droid",
            "scenario": scenario,
        },
    )
    _write(
        attempt / "events.json",
        {"rows": [{"kind": "trigger_eligible", "sim_time": 0.1}]},
    )
    _write(
        attempt / "requests.json",
        {
            "rows": [
                {
                    "request_id": "req-00001",
                    "observation_id": "obs-00001",
                    "observation_capture_time": 0.0,
                    "submit_time": 0.0,
                    "executed_action_count": 0,
                },
                {
                    "request_id": "req-00001",
                    "action_sha256": SHA_B,
                    "generated_horizon": 32,
                    "policy_request_audit": {
                        "request_index": 0,
                        "request_sampling_seed": 202,
                        "action_step_start": 0,
                        "observation_sha256": SHA_C,
                        "prompt_sha256": SHA_A,
                        "wire_request_sha256": SHA_D,
                        "future_sha256": SHA_C,
                    },
                },
            ]
        },
    )
    trajectory = []
    for tick, eligible in ((0, False), (1, True)):
        trajectory.append(
            {
                "simulation_time": tick * 0.1,
                "control_step": tick,
                "reference_displacement_m": 0.0,
                "commanded_action": None if tick == 0 else [0.1, 0.2],
                "grasp_eligible": eligible,
                "detach_armed": eligible,
                "object_state": {
                    "object_position_world": [0.1, 0.2, 0.3 + tick * 0.04],
                    "gripper_position_world": [0.1, 0.2, 0.3 + tick * 0.04],
                    "initial_supported_z": 0.3,
                    "contact": True,
                    "detached": False,
                },
                "reference_position_world": [0.4, 0.5, 0.6],
                "controller_state": {
                    "pending_action_count": 31,
                    "executed_action_count": tick,
                    "pending_request_count": 0,
                    "completed_request_count": 1,
                    "next_query_time": 1.0,
                    "fast_schedule_active": False,
                    "policy_phase_active": True,
                    "passive_settling_active": False,
                },
            }
        )
    _write(attempt / "trajectory.json", {"rows": trajectory})
    return attempt


def _session(path: Path, *, attempt_id: str, policy_sha: str, simulator_sha: str) -> Path:
    _write(
        path,
        {
            "schema_version": "v4-c2-fresh-session-attestation-v1",
            "attempt_id": attempt_id,
            "policy_started_fresh": True,
            "simulator_started_fresh": True,
            "policy_reset_before_prefix": True,
            "simulator_reset_before_prefix": True,
            "no_reused_hidden_session_state": True,
            "policy_process_identity_sha256": policy_sha,
            "simulator_process_identity_sha256": simulator_sha,
        },
    )
    return path


class PrefixReplayTests(unittest.TestCase):
    def _case(self, root: Path):
        sham = _attempt(root, attempt_id="a1", scenario="original_sham")
        move = _attempt(root, attempt_id="a2", scenario="move_A")
        sham_session = _session(
            root / "a1-session.json",
            attempt_id="a1",
            policy_sha=SHA_A,
            simulator_sha=SHA_B,
        )
        move_session = _session(
            root / "a2-session.json",
            attempt_id="a2",
            policy_sha=SHA_C,
            simulator_sha=SHA_D,
        )
        return sham, move, sham_session, move_session

    def test_exact_replay_with_distinct_fresh_processes_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sham, move, sham_session, move_session = self._case(Path(tmp))
            receipt = verify_common_prefix(
                left_attempt_dir=sham,
                right_attempt_dir=move,
                left_session_receipt_path=sham_session,
                right_session_receipt_path=move_session,
            )
            self.assertTrue(receipt["passed"])
            self.assertEqual(
                receipt["common_prefix_verification_mode"],
                "deterministic_fresh_session_replay",
            )
            self.assertEqual(receipt["request_count"], 1)

    def test_different_policy_wire_history_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sham, move, sham_session, move_session = self._case(Path(tmp))
            requests = json.loads((move / "requests.json").read_text())
            requests["rows"][1]["policy_request_audit"]["wire_request_sha256"] = SHA_A
            _write(move / "requests.json", requests)
            with self.assertRaisesRegex(PrefixReplayError, "histories differ"):
                verify_common_prefix(
                    left_attempt_dir=sham,
                    right_attempt_dir=move,
                    left_session_receipt_path=sham_session,
                    right_session_receipt_path=move_session,
                )

    def test_reused_policy_process_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sham, move, sham_session, move_session = self._case(Path(tmp))
            receipt = json.loads(move_session.read_text())
            receipt["policy_process_identity_sha256"] = SHA_A
            _write(move_session, receipt)
            with self.assertRaisesRegex(PrefixReplayError, "reused policy"):
                verify_common_prefix(
                    left_attempt_dir=sham,
                    right_attempt_dir=move,
                    left_session_receipt_path=sham_session,
                    right_session_receipt_path=move_session,
                )


if __name__ == "__main__":
    unittest.main()
