from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np


WORKSHOP = Path(__file__).resolve().parents[1]
FORECAST = WORKSHOP / "experiments/forecast_layout"
sys.path.insert(0, str(FORECAST))
import recorder_qualification_job as qualification  # noqa: E402
import recording_adapter as recording  # noqa: E402


def _write_json(path: Path, value: dict) -> dict:
    payload = qualification.canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _accepted_release(root: Path) -> tuple[Path, str, Path, str]:
    pool = json.loads((FORECAST / "layout_candidate_pool.json").read_text())
    candidate = next(row for row in pool["candidates"] if row["candidate_id"] == "P00__candidate_00")
    attempt = {
        "schema_version": qualification.GATE_ATTEMPT_SCHEMA,
        "candidate_id": candidate["candidate_id"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "candidate_pool_sha256": qualification.CANDIDATE_POOL_SHA256,
        "decision": "accepted",
        "passed": True,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "evaluation": {"passed": True},
    }
    attempt_identity = _write_json(root / "attempt/gate_attempt_receipt.json", attempt)
    record = {
        "schema_version": qualification.GATE_RECORD_SCHEMA,
        "study_namespace": qualification.NAMESPACE,
        "sequence": 0,
        "previous_record_sha256": None,
        "recorded_at_utc": "2026-09-13T00:00:00Z",
        "layout_pair_id": qualification.LAYOUT_PAIR_ID,
        "candidate_id": candidate["candidate_id"],
        "candidate_rank": candidate["candidate_rank"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "candidate_pool_sha256": qualification.CANDIDATE_POOL_SHA256,
        "decision": "accepted",
        "passed": True,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "attempt_receipt": attempt_identity,
    }
    record["record_sha256"] = qualification.sha256_bytes(qualification.canonical_bytes(record))
    ledger_path = root / "gate_ledger.jsonl"
    ledger_path.write_bytes(
        json.dumps(record, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        + b"\n"
    )
    ledger_identity = qualification.file_identity(ledger_path)
    pose = {
        "schema_version": qualification.POSE_MANIFEST_SCHEMA,
        "study_namespace": qualification.NAMESPACE,
        "status": "LIVE_GATE_QUALIFIED_POSES_FROZEN_MODEL_EXECUTION_NOT_RELEASED",
        "physical_layout_gate_passed": True,
        "released_for_model_inference": False,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_pool_sha256": qualification.CANDIDATE_POOL_SHA256,
        "gate_ledger_sha256": ledger_identity["sha256"],
        "layout_pairs": {
            qualification.LAYOUT_PAIR_ID: {
                "layout_pair_id": qualification.LAYOUT_PAIR_ID,
                "candidate_id": candidate["candidate_id"],
                "candidate_payload_sha256": candidate["candidate_payload_sha256"],
                "accepted_gate_record_sha256": record["record_sha256"],
                "accepted_gate_attempt_receipt": attempt_identity,
                "layouts": candidate["layouts"],
            }
        },
        "task_contract": {
            "action_cap": 450,
            "termination_terms": ["time_out"],
            "success_is_measurement_only": True,
            "success_termination_present": False,
        },
    }
    pose_path = root / "p00_pose_manifest.json"
    pose_identity = _write_json(pose_path, pose)
    gate = {
        "schema_version": qualification.GATE_RECEIPT_SCHEMA,
        "study_namespace": qualification.NAMESPACE,
        "status": "finished",
        "layout_pair_id": qualification.LAYOUT_PAIR_ID,
        "candidate_id": candidate["candidate_id"],
        "candidate_payload_sha256": candidate["candidate_payload_sha256"],
        "decision": "accepted",
        "exit_code": 0,
        "source_contract_sha256": qualification.SOURCE_CONTRACT_SHA256,
        "candidate_pool_sha256": qualification.CANDIDATE_POOL_SHA256,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "gate_ledger": ledger_identity,
        "gate_evidence": {
            "gate_record_sha256": record["record_sha256"],
            "gate_attempt_receipt": attempt_identity,
        },
    }
    gate_path = root / "fixture_gate_receipt.json"
    gate_identity = _write_json(gate_path, gate)
    return gate_path, gate_identity["sha256"], pose_path, pose_identity["sha256"]


class TimeoutTerm:
    time_out = True


class TimeoutOnly:
    time_out = TimeoutTerm()


class FakeConfig:
    terminations = TimeoutOnly()
    episode_length_s = 30
    decimation = 8
    sim = SimpleNamespace(dt=1 / 120, render_interval=8)


class FakeSensor:
    def __init__(self) -> None:
        self.data = SimpleNamespace(
            frame=np.asarray([75], dtype=np.int64),
            timestamp=np.asarray([5.0], dtype=np.float64),
        )


class FakeEnv:
    num_envs = 1
    max_episode_length = 450
    device = "cpu"

    def __init__(self) -> None:
        self.count = 0
        self.common_step_counter = 75
        self.episode_length_buf = np.asarray([0], dtype=np.int64)
        self.sim = SimpleNamespace(frame_count=600, current_time=5.0)
        self.scene = {name: FakeSensor() for name in qualification.REQUIRED_CAMERAS}
        self._done = False

    @property
    def active_env_ids(self):
        return [] if self._done else [0]

    @property
    def all_terminated(self):
        return self._done

    def observation(self):
        pixel = self.count % 251
        return {
            "image_obs": {
                name: np.full((1, 2, 3, 3), pixel + index, dtype=np.uint8)
                for index, name in enumerate(qualification.REQUIRED_CAMERAS)
            },
            "proprio_obs": {
                "arm_joint_pos": np.full((1, 7), 0.1, dtype=np.float32),
                "gripper_pos": np.full((1, 1), 0.3, dtype=np.float32),
            },
        }

    def reset(self):
        return self.observation(), {"reset": True}

    def step(self, action):
        self.count += 1
        self.common_step_counter += 1
        self.episode_length_buf[0] = self.count
        self.sim.frame_count += 8
        self.sim.current_time += 1 / 15
        for sensor in self.scene.values():
            sensor.data.frame[0] += 1
            sensor.data.timestamp[0] += 1 / 15
        self._done = self.count == 450
        return (
            self.observation(),
            np.asarray([0.0]),
            np.asarray([False]),
            np.asarray([self._done]),
            {"step": self.count},
        )


def _identity() -> dict:
    return {
        "attempt_id": "recorder-test-001",
        "cell_id": "wmf1__recording_qualification__P00__RECORDER_ONLY__original__left",
        "stage": "recording_qualification",
        "layout_pair_id": "P00",
        "layout_arm": "original",
        "command": "left",
        "prompt": qualification.PROMPTS["left"],
        "model_config": recording.RECORDER_ONLY_MODEL_CONFIG,
        "effective_seed": 2026091000,
        "source_identity": "test-source",
        "checkpoint_identity": "none-no-model",
    }


def _reset_attestor(env, observation, info):
    del env, observation, info
    return {
        "passed": True,
        "settled": True,
        "left_success": False,
        "right_success": False,
        "reset_identity": "settled-test-reset",
        "pose_manifest_sha256": "a" * 64,
        "initial_observation_hashes": {"combined_sha256": "b" * 64},
        "settle_evidence": {"passed": True},
        "collision_evidence": {"passed": True},
        "visibility_evidence": {"passed": True},
        "settled_observation_returned": True,
        "model_request_count_during_settle": 0,
        "episode_length_buf_reset_to_zero": True,
    }


class RecorderQualificationTests(unittest.TestCase):
    def test_exact_450_action_recorder_only_path_uses_native_clock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recorder = recording.ForecastRecordingAdapter(root / "attempt", _identity())
            recorder.record_context_reset(
                {
                    "passed": True,
                    "reset_scope": recording.CONTEXT_RESET_SCOPE,
                    "server_context_id": "none-recorder-only",
                    "cache_reset_evidence": {"no_model_attached": True},
                }
            )
            env = FakeEnv()
            clock = qualification.NativeClockSampler(root / "attempt/timing_support.json", FakeConfig())
            proxy = recording.FixedDurationEnvProxy(
                env,
                FakeConfig(),
                recorder,
                state_sampler=lambda live: {
                    "simulator_state_sample_only_not_policy_input": True,
                    "native_control_counter": live.common_step_counter,
                },
                clock_sampler=clock,
                success_sampler=lambda _live: {"left": False, "right": False, "released": True},
                reset_attestor=_reset_attestor,
            )
            observation, _ = proxy.reset()
            duplicate, _ = proxy.reset()
            self.assertIs(duplicate, observation)

            def hold(value, _device):
                return np.concatenate(
                    (value["proprio_obs"]["arm_joint_pos"], value["proprio_obs"]["gripper_pos"]),
                    axis=1,
                ).astype(np.float32)

            receipt = qualification.execute_recorder_only_actions(
                proxy=proxy,
                recorder=recorder,
                observation=observation,
                hold_action_builder=hold,
            )
            self.assertEqual(receipt["actions_executed"], 450)
            self.assertEqual(receipt["observation_count"], 451)
            self.assertEqual(receipt["request_count"], 0)
            self.assertTrue(receipt["recording_qualification_valid"])
            self.assertFalse(receipt["behavioral_result_valid"])
            timing = json.loads((root / "attempt/timing_support.json").read_text())
            self.assertTrue(timing["supported"])
            self.assertFalse(timing["forecast_alignment_inferred"])
            verified = recording.verify_journal(recorder.journal_path)
            self.assertEqual(verified["event_count"], receipt["event_count"])

    def test_missing_camera_timestamp_is_explicitly_unsupported_not_inferred(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            root.mkdir(exist_ok=True)
            env = FakeEnv()
            del env.scene[qualification.REQUIRED_CAMERAS[1]].data.timestamp
            sampler = qualification.NativeClockSampler(root / "timing_support.json", FakeConfig())
            with self.assertRaisesRegex(qualification.RecorderQualificationError, "native runtime"):
                sampler(env, "settled_reset", 0)
            receipt = json.loads((root / "timing_support.json").read_text())
            self.assertFalse(receipt["supported"])
            self.assertFalse(receipt["forecast_alignment_inferred"])
            self.assertFalse(receipt["action_number_used_as_camera_time"])
            self.assertFalse(receipt["video_frame_number_used_as_action_number"])

    def test_p00_release_is_bound_to_gate_ledger_attempt_and_pose_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate, gate_sha, pose, pose_sha = _accepted_release(root)
            release = qualification.verify_fixture_release(
                gate_receipt_path=gate,
                gate_receipt_sha256=gate_sha,
                pose_manifest_path=pose,
                pose_manifest_sha256=pose_sha,
                layout_arm="reflected",
            )
            self.assertEqual(release["candidate_id"], "P00__candidate_00")
            changed = json.loads(pose.read_text())
            changed["task_contract"]["termination_terms"] = ["success", "time_out"]
            pose.write_bytes(qualification.canonical_bytes(changed))
            with self.assertRaisesRegex(qualification.RecorderQualificationError, "argument hash mismatch"):
                qualification.verify_fixture_release(
                    gate_receipt_path=gate,
                    gate_receipt_sha256=gate_sha,
                    pose_manifest_path=pose,
                    pose_manifest_sha256=pose_sha,
                    layout_arm="reflected",
                )

    def test_queue_reexec_uses_pinned_python_and_live_gm_isaac_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            state_parent = root / "private"
            state_parent.mkdir()
            environment = qualification.build_child_environment(
                source_root=source,
                state_parent=state_parent,
                hostname="worker-01",
                base={
                    "DISPLAY": ":99",
                    "LD_PRELOAD": "bad.so",
                    "CUDA_VISIBLE_DEVICES": "7",
                    "NVIDIA_VISIBLE_DEVICES": "GPU-assigned",
                },
            )
            self.assertNotIn("DISPLAY", environment)
            self.assertNotIn("LD_PRELOAD", environment)
            self.assertNotIn("CUDA_VISIBLE_DEVICES", environment)
            self.assertEqual(environment["NVIDIA_VISIBLE_DEVICES"], "GPU-assigned")
            self.assertEqual(environment["OMNI_KIT_ACCEPT_EULA"], "YES")
            self.assertEqual(environment["VK_ICD_FILENAMES"], "/etc/vulkan/icd.d/nvidia_icd.json")
            self.assertEqual(environment["LD_LIBRARY_PATH"], qualification.NATIVE_LIBRARY_PATH)
            for isaac_root in qualification.ISAACLAB_SOURCE_ROOTS:
                self.assertIn(str(isaac_root), environment["PYTHONPATH"].split(":"))
            lexical_venv = root / "venv/bin/python"
            lexical_venv.parent.mkdir(parents=True)
            lexical_venv.symlink_to(Path(sys.executable).resolve())
            argv = qualification.build_child_command(
                source_root=source,
                output_dir=root / "raw/adapter_attempt",
                gate_receipt_path=root / "gate.json",
                gate_receipt_sha256="a" * 64,
                pose_manifest_path=root / "pose.json",
                pose_manifest_sha256="b" * 64,
                study_commit="c" * 40,
                layout_arm="original",
                command="right",
                environment_seed=2026091000,
                attempt_id="recorder-test",
                robolab_python=lexical_venv,
            )
            self.assertEqual(argv[0], str(lexical_venv.absolute()))
            self.assertNotEqual(argv[0], str(Path(sys.executable).resolve()))
            self.assertEqual(argv[2], "record")
            self.assertIn("--layout-arm", argv)
            self.assertIn("--command", argv)


if __name__ == "__main__":
    unittest.main()
