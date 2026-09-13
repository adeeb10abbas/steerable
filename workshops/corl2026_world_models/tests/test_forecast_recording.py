import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np


MODULE = (
    Path(__file__).resolve().parents[1]
    / "experiments/forecast_layout/recording_adapter.py"
)
spec = importlib.util.spec_from_file_location("forecast_recording", MODULE)
recording = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recording)


PROMPT = "Put the Rubik's cube to the left of the bowl."


class FakeTerm:
    def __init__(self, *, time_out):
        self.time_out = time_out


class TimeoutOnly:
    time_out = FakeTerm(time_out=True)


class SuccessAndTimeout:
    time_out = FakeTerm(time_out=True)
    success = FakeTerm(time_out=False)


class FakeTask:
    terminations = SuccessAndTimeout
    episode_length_s = 10


class FakeEnvCfg:
    def __init__(self, termination_type=TimeoutOnly):
        self.terminations = termination_type()
        self.episode_length_s = 30
        self.decimation = 8
        self.sim = SimpleNamespace(dt=1 / 120, render_interval=8)


class FakeEnv:
    num_envs = 1
    max_episode_length = 450

    def __init__(self, *, early_done=None):
        self.step_count = 0
        self.reset_count = 0
        self.closed = False
        self.early_done = early_done
        self._done = False

    @property
    def active_env_ids(self):
        return [] if self._done else [0]

    @property
    def all_terminated(self):
        return self._done

    def observation(self):
        value = self.step_count % 251
        return {
            "image_obs": {
                "over_shoulder_left_camera": np.full((1, 2, 3, 3), value, dtype=np.uint8),
                "over_shoulder_right_camera": np.full((1, 2, 3, 3), value + 1, dtype=np.uint8),
                "wrist_cam": np.full((1, 2, 3, 3), value + 2, dtype=np.uint8),
            },
            "proprio_obs": {
                "arm_joint_pos": np.full((1, 7), self.step_count, dtype=np.float32),
                "gripper_pos": np.full((1, 1), 0.25, dtype=np.float32),
            },
        }

    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        self._done = False
        return self.observation(), {"physical_reset": self.reset_count}

    def step(self, action):
        self.step_count += 1
        self._done = self.step_count == 450 or self.step_count == self.early_done
        term = np.asarray([self.step_count == self.early_done], dtype=bool)
        trunc = np.asarray([self.step_count == 450], dtype=bool)
        return self.observation(), np.asarray([0.0]), term, trunc, {"step": self.step_count}

    def close(self):
        self.closed = True


class FakeOfficialClient:
    def __init__(self, model="N3", *, include_futures=True):
        self.model = model
        self.include_futures = include_futures
        self.open_loop_horizon = 32 if model == "N3" else 8
        self.returned_horizon = 32 if model == "N3" else 24
        self._chunks = {}
        self._counters = {}
        self._env_session_id = {0: "old-session"}
        self.request_number = 0

    def reset(self, *, env_id=None):
        self._chunks.clear()
        self._counters.clear()
        self._env_session_id.clear()

    def _extract_observation(self, raw_obs, *, env_id=0):
        left = raw_obs["image_obs"]["over_shoulder_left_camera"][env_id]
        return {
            "exact_resized_input": left[:, ::-1].copy(),
            "joint_position": raw_obs["proprio_obs"]["arm_joint_pos"][env_id].copy(),
        }

    def _pack_request(self, extracted_obs, instruction):
        return {
            "observation/image": extracted_obs["exact_resized_input"],
            "observation/joint_position": extracted_obs["joint_position"],
            "prompt": instruction,
            "request_number": self.request_number,
        }

    def _query_server(self, request):
        actions = np.stack(
            [
                np.full(8, self.request_number * 100 + offset, dtype=np.float32)
                for offset in range(self.returned_horizon)
            ]
        )
        self.request_number += 1
        if self.model == "N3":
            response = {"action": actions}
            if self.include_futures:
                response["video"] = np.full((3, 2, 2, 3), self.request_number, dtype=np.uint8)
            return response
        response = {"actions": actions}
        if self.include_futures:
            response["future_evidence"] = {
                "latent": np.full((2, 2), self.request_number, dtype=np.float32),
                "decoded": np.full((3, 2, 2, 3), self.request_number, dtype=np.uint8),
            }
        return response

    def _unpack_response(self, response):
        return np.asarray(response.get("action", response.get("actions")), dtype=np.float32)

    def _postprocess_chunk(self, chunk):
        output = np.asarray(chunk, dtype=np.float32).copy()
        output[:, -1] = (output[:, -1] > 0.5).astype(np.float32)
        return output

    def infer(self, obs, instruction, *, env_id=0):
        extracted = self._extract_observation(obs, env_id=env_id)
        if env_id not in self._chunks or self._counters[env_id] >= self.open_loop_horizon:
            request = self._pack_request(extracted, instruction)
            response = self._query_server(request)
            returned = self._unpack_response(response)
            self._chunks[env_id] = self._postprocess_chunk(returned)
            self._counters[env_id] = 0
        action = self._chunks[env_id][self._counters[env_id]]
        self._counters[env_id] += 1
        return {"action": action, "viz": extracted["exact_resized_input"]}


class RecordedFakeClient(recording.RecordingClientMixin, FakeOfficialClient):
    pass


def identity(model="N3", attempt="attempt-01"):
    return {
        "attempt_id": attempt,
        "cell_id": f"wmf1__pilot__P00__{model}__original__left",
        "stage": "pilot",
        "layout_pair_id": "P00",
        "layout_arm": "original",
        "command": "left",
        "prompt": PROMPT,
        "model_config": model,
        "effective_seed": 2026091000 if model == "N3" else 1140,
        "source_identity": "source-commit-test",
        "checkpoint_identity": "checkpoint-test",
    }


def context_receipt():
    return {
        "passed": True,
        "reset_scope": recording.CONTEXT_RESET_SCOPE,
        "server_context_id": "isolated-server-context-test",
        "cache_reset_evidence": {
            "server_acknowledged": True,
            "temporal_buffers_after": 0,
            "denoising_cache_entries_after": 0,
        },
    }


def clock_sampler(env, phase, action_step):
    return {
        "physics_step": action_step * 8,
        "physics_time_s": action_step / 15,
        "control_step": action_step,
        "cameras": {
            camera: {
                "frame_id": f"{camera}:{action_step}",
                "capture_time_ns": 1_000_000_000 + action_step * 66_666_667,
                "timestamp_source": "fake_native_camera_clock",
            }
            for camera in recording.REQUIRED_CAMERAS
        },
        "phase": phase,
    }


def state_sampler(env):
    return {
        "simulator_state_sample_only_not_policy_input": True,
        "cube_robot_xyz": [0.4, 0.0, 0.1],
        "bowl_robot_xyz": [0.5, 0.0, 0.1],
        "physics_step": env.step_count * 8,
    }


def success_sampler(env):
    succeeded = env.step_count >= 7
    return {"left": succeeded, "right": False, "released": succeeded}


def reset_attestor(env, obs, info):
    return {
        "passed": True,
        "settled": True,
        "left_success": False,
        "right_success": False,
        "reset_identity": "reset-test-001",
        "pose_manifest_sha256": "a" * 64,
        "initial_observation_hashes": {camera: "b" * 64 for camera in recording.REQUIRED_CAMERAS},
        "settle_evidence": {"stable": True},
        "collision_evidence": {"clear": True},
        "visibility_evidence": {"visible": True},
        "settled_observation_returned": True,
        "model_request_count_during_settle": 0,
        "episode_length_buf_reset_to_zero": True,
    }


class ForecastRecordingTests(unittest.TestCase):
    def make_runtime(self, root, *, model="N3", env=None, include_futures=True):
        recorder = recording.ForecastRecordingAdapter(Path(root) / "attempt", identity(model))
        client = RecordedFakeClient(model=model, include_futures=include_futures)
        client.attach_forecast_recorder(recorder)
        client.reset_for_recorded_episode(context_receipt)
        env = env or FakeEnv()
        proxy = recording.FixedDurationEnvProxy(
            env,
            FakeEnvCfg(),
            recorder,
            state_sampler=state_sampler,
            clock_sampler=clock_sampler,
            success_sampler=success_sampler,
            reset_attestor=reset_attestor,
        )
        obs, _ = proxy.reset()
        duplicate, _ = proxy.reset()
        self.assertIs(duplicate, obs)
        self.assertEqual(env.reset_count, 1)
        return recorder, client, proxy, obs

    def run_until_done(self, client, proxy, obs):
        while not proxy.all_terminated:
            result = client.infer(obs, PROMPT)
            action = np.expand_dims(result["action"], axis=0)
            obs, *_ = proxy.step(action)

    def test_task_is_timeout_only_before_registration(self):
        task = type("WorkshopTask", (FakeTask,), {})
        recording.configure_fixed_duration_task(
            task, termination_cfg_factory=lambda: TimeoutOnly
        )
        self.assertEqual(task.episode_length_s, 30)
        self.assertEqual(task._wmf_action_cap, 450)
        self.assertNotIn("success", recording._public_termination_terms(task.terminations()))
        with self.assertRaisesRegex(recording.RecordingContractError, "only time_out"):
            recording.configure_fixed_duration_task(
                type("BadTask", (FakeTask,), {}),
                termination_cfg_factory=lambda: SuccessAndTimeout,
            )

    def test_environment_assertion_rejects_success_and_wrong_duration(self):
        env = FakeEnv()
        with self.assertRaisesRegex(recording.RecordingContractError, "non-timeout"):
            recording.assert_fixed_duration_environment(env, FakeEnvCfg(SuccessAndTimeout))
        env.max_episode_length = 449
        with self.assertRaisesRegex(recording.RecordingContractError, "450"):
            recording.assert_fixed_duration_environment(env, FakeEnvCfg())

    def test_n3_runs_450_actual_actions_after_first_success(self):
        with tempfile.TemporaryDirectory() as root:
            recorder, client, proxy, obs = self.make_runtime(root)
            self.run_until_done(client, proxy, obs)
            receipt = json.loads(recorder.completion_path.read_text())
            self.assertTrue(receipt["behavioral_result_valid"])
            self.assertEqual(receipt["stop_reason"], "action_cap")
            self.assertEqual(receipt["actions_executed"], 450)
            self.assertEqual(receipt["observation_count"], 451)
            self.assertEqual(receipt["request_count"], 15)
            self.assertEqual(receipt["first_success"]["action_step"], 7)
            self.assertEqual(receipt["actions_after_first_success"], 443)
            self.assertFalse(receipt["success_configured_as_termination"])
            self.assertEqual(receipt["final_chunk"]["executed_actions"], 2)
            self.assertEqual(receipt["final_chunk"]["unused_executable_prefix_actions"], 30)
            self.assertTrue(receipt["final_two_action_truncation_recorded"])
            self.assertEqual(proxy._env.step_count, 450)
            verified = recording.verify_journal(recorder.journal_path)
            self.assertEqual(verified["event_count"], receipt["event_count"])
            self.assertEqual(verified["tail_sha256"], receipt["journal_tail_sha256"])

    def test_recorder_only_runs_450_actions_without_creating_model_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            attempt_identity = identity(model="N3", attempt="recorder-only-01")
            attempt_identity.update(
                cell_id="wmf1__recording_qualification__P00__original__left",
                stage="recording_qualification",
                model_config=recording.RECORDER_ONLY_MODEL_CONFIG,
                source_identity="scripted_joint_hold_no_policy",
                checkpoint_identity="none_no_model_loaded",
            )
            recorder = recording.ForecastRecordingAdapter(
                Path(root) / "attempt", attempt_identity
            )
            recorder.record_context_reset(
                {
                    "passed": True,
                    "reset_scope": recording.CONTEXT_RESET_SCOPE,
                    "server_context_id": "not-applicable-recorder-only",
                    "cache_reset_evidence": {"no_model_attached": True},
                }
            )
            proxy = recording.FixedDurationEnvProxy(
                FakeEnv(),
                FakeEnvCfg(),
                recorder,
                state_sampler=state_sampler,
                clock_sampler=clock_sampler,
                success_sampler=success_sampler,
                reset_attestor=reset_attestor,
            )
            observation, _ = proxy.reset()
            proxy.reset()
            while not proxy.all_terminated:
                action = np.concatenate(
                    (
                        observation["proprio_obs"]["arm_joint_pos"][0],
                        observation["proprio_obs"]["gripper_pos"][0],
                    )
                ).astype(np.float32)
                recorder.record_scripted_action(
                    action,
                    action_source={
                        "kind": "joint_position_hold",
                        "policy_model": None,
                        "scientific_claim": "recorder_qualification_only",
                    },
                )
                observation, *_ = proxy.step(np.expand_dims(action, axis=0))
            receipt = recorder._final_receipt
            self.assertEqual(receipt["actions_executed"], 450)
            self.assertEqual(receipt["observation_count"], 451)
            self.assertEqual(receipt["request_count"], 0)
            self.assertTrue(receipt["recording_qualification_valid"])
            self.assertFalse(receipt["behavioral_result_valid"])
            self.assertFalse(receipt["model_attached"])
            self.assertFalse(receipt["technical_invalid"])

    def test_request_binds_current_and_preceding_original_observations(self):
        with tempfile.TemporaryDirectory() as root:
            recorder, client, proxy, obs = self.make_runtime(root)
            for _ in range(33):
                result = client.infer(obs, PROMPT)
                obs, *_ = proxy.step(np.expand_dims(result["action"], axis=0))
            first, second = recorder.requests
            self.assertEqual(first["current_observation_id"], "obs_000000")
            self.assertIsNone(first["preceding_observation_id"])
            self.assertEqual(first["constant_velocity_baseline"], "reduces_to_persistence_no_preceding_observation")
            self.assertEqual(second["current_observation_id"], "obs_000032")
            self.assertEqual(second["preceding_observation_id"], "obs_000031")
            payload = recording.load_payload(
                recorder.attempt_dir, first["model_request_artifact"]
            )
            np.testing.assert_array_equal(
                payload["extracted_preprocessing_output"]["exact_resized_input"],
                np.zeros((2, 3, 3), dtype=np.uint8)[:, ::-1],
            )
            np.testing.assert_array_equal(
                payload["wire_request"]["observation/image"],
                payload["extracted_preprocessing_output"]["exact_resized_input"],
            )
            proxy.close()
            self.assertEqual(recorder._final_receipt["stop_reason"], "technical_failure")

    def test_d1_records_returned_tail_and_final_two_action_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            recorder, client, proxy, obs = self.make_runtime(root, model="D1")
            self.run_until_done(client, proxy, obs)
            receipt = recorder._final_receipt
            self.assertTrue(receipt["behavioral_result_valid"])
            self.assertEqual(receipt["request_count"], 57)
            self.assertEqual(receipt["final_chunk"]["returned_actions"], 24)
            self.assertEqual(receipt["final_chunk"]["eligible_executable_prefix_actions"], 8)
            self.assertEqual(receipt["final_chunk"]["executed_actions"], 2)
            self.assertEqual(receipt["final_chunk"]["unused_executable_prefix_actions"], 6)
            self.assertEqual(receipt["final_chunk"]["returned_actions_outside_executable_prefix"], 16)

    def test_safety_abort_retains_real_prefix_as_censored(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = recording.ForecastRecordingAdapter(Path(root) / "attempt", identity())
            client = RecordedFakeClient()
            client.attach_forecast_recorder(recorder)
            client.reset_for_recorded_episode(context_receipt)
            env = FakeEnv()
            proxy = recording.FixedDurationEnvProxy(
                env,
                FakeEnvCfg(),
                recorder,
                state_sampler=state_sampler,
                clock_sampler=clock_sampler,
                success_sampler=success_sampler,
                reset_attestor=reset_attestor,
                safety_checker=lambda env, obs, info: (
                    {"reason": "workspace boundary", "source": "independent safety monitor"}
                    if env.step_count == 3
                    else None
                ),
            )
            obs, _ = proxy.reset()
            proxy.reset()
            self.run_until_done(client, proxy, obs)
            receipt = recorder._final_receipt
            self.assertEqual(receipt["stop_reason"], "safety_abort")
            self.assertTrue(receipt["right_censored"])
            self.assertFalse(receipt["technical_invalid"])
            self.assertFalse(receipt["behavioral_result_valid"])
            self.assertEqual(receipt["actions_executed"], 3)
            self.assertEqual(receipt["observation_count"], 4)

    def test_early_done_is_technical_not_behavioral_failure(self):
        with tempfile.TemporaryDirectory() as root:
            recorder, client, proxy, obs = self.make_runtime(
                root, env=FakeEnv(early_done=9)
            )
            with self.assertRaisesRegex(recording.RecordingContractError, "early"):
                self.run_until_done(client, proxy, obs)
            receipt = recorder._final_receipt
            self.assertEqual(receipt["stop_reason"], "technical_failure")
            self.assertTrue(receipt["technical_invalid"])
            self.assertFalse(receipt["behavioral_result_valid"])
            self.assertEqual(receipt["actions_executed"], 9)

    def test_action_mismatch_is_technical_and_never_reaches_environment(self):
        with tempfile.TemporaryDirectory() as root:
            recorder, client, proxy, obs = self.make_runtime(root)
            result = client.infer(obs, PROMPT)
            changed = np.expand_dims(result["action"].copy(), axis=0)
            changed[0, 0] += 1
            with self.assertRaisesRegex(recording.RecordingContractError, "differs"):
                proxy.step(changed)
            self.assertEqual(proxy._env.step_count, 0)
            self.assertTrue(recorder._final_receipt["technical_invalid"])

    def test_missing_future_fails_first_request_without_robot_action(self):
        with tempfile.TemporaryDirectory() as root:
            recorder, client, proxy, obs = self.make_runtime(
                root, include_futures=False
            )
            with self.assertRaisesRegex(recording.RecordingContractError, "future evidence"):
                client.infer(obs, PROMPT)
            self.assertEqual(proxy._env.step_count, 0)
            self.assertEqual(recorder._final_receipt["actions_executed"], 0)
            self.assertTrue(recorder._final_receipt["technical_invalid"])

    def test_context_and_settled_reset_receipts_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = recording.ForecastRecordingAdapter(Path(root) / "a", identity())
            bad = context_receipt()
            bad["reset_scope"] = "client_chunks_only"
            with self.assertRaisesRegex(recording.RecordingContractError, "scope"):
                recorder.record_context_reset(bad)
        with tempfile.TemporaryDirectory() as root:
            recorder = recording.ForecastRecordingAdapter(Path(root) / "a", identity())
            client = RecordedFakeClient()
            client.attach_forecast_recorder(recorder)
            client.reset_for_recorded_episode(context_receipt)
            env = FakeEnv()
            with self.assertRaisesRegex(recording.RecordingContractError, "success state"):
                proxy = recording.FixedDurationEnvProxy(
                    env,
                    FakeEnvCfg(),
                    recorder,
                    state_sampler=state_sampler,
                    clock_sampler=clock_sampler,
                    success_sampler=lambda env: {"left": True, "right": False, "released": True},
                    reset_attestor=lambda env, obs, info: {
                        **reset_attestor(env, obs, info),
                        "left_success": True,
                    },
                )
                proxy.reset()
            self.assertTrue(recorder._final_receipt["technical_invalid"])

    def test_reset_attestor_can_return_exact_post_settle_observation(self):
        with tempfile.TemporaryDirectory() as root:
            recorder = recording.ForecastRecordingAdapter(Path(root) / "attempt", identity())
            client = RecordedFakeClient()
            client.attach_forecast_recorder(recorder)
            client.reset_for_recorded_episode(context_receipt)
            env = FakeEnv()

            def settle(env, obs, info):
                settled = {
                    "image_obs": {
                        name: np.full_like(value, 99)
                        for name, value in obs["image_obs"].items()
                    },
                    "proprio_obs": {
                        name: value.copy() for name, value in obs["proprio_obs"].items()
                    },
                }
                return settled, {**info, "post_settle": True}, reset_attestor(env, settled, info)

            proxy = recording.FixedDurationEnvProxy(
                env,
                FakeEnvCfg(),
                recorder,
                state_sampler=state_sampler,
                clock_sampler=clock_sampler,
                success_sampler=success_sampler,
                reset_attestor=settle,
            )
            obs, info = proxy.reset()
            self.assertTrue(info["post_settle"])
            self.assertTrue(np.all(obs["image_obs"]["wrist_cam"] == 99))
            payload = recording.load_payload(
                recorder.attempt_dir, recorder.observations[0]["artifact"]
            )
            self.assertTrue(np.all(payload["image_obs"]["wrist_cam"] == 99))
            proxy.reset()
            proxy.close()

    def test_attempt_directory_is_immutable(self):
        with tempfile.TemporaryDirectory() as root:
            attempt = Path(root) / "attempt"
            recording.ForecastRecordingAdapter(attempt, identity())
            with self.assertRaises(FileExistsError):
                recording.ForecastRecordingAdapter(attempt, identity(attempt="attempt-02"))


if __name__ == "__main__":
    unittest.main()
