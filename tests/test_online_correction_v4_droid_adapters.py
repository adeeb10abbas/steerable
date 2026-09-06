"""Adapter protocol tests for the V4 DROID live layer using fakes."""

from __future__ import annotations

import json
import unittest

from experiments.online_correction_v4.adapters import ObservationPacket
from experiments.online_correction_v4.clock import QuerySchedule
from experiments.online_correction_v4.contracts import EpisodeManifestRow, TimingConfig
from experiments.online_correction_v4.droid_bindings import (
    attestation_from_fake_reset,
    build_fake_binding,
    build_episode_runner,
)
from experiments.online_correction_v4.droid_contract import (
    FixtureRuntimeBinding,
    PolicyRuntimeBinding,
    PrefixMode,
    RuntimeLockBinding,
    WriterContract,
    sha256_bytes,
)
from experiments.online_correction_v4.droid_nano_policy import (
    DroidNanoPolicyAdapter,
    NANO_ACTION_SHAPE,
    _normalize_action_chunk,
    _normalize_future,
    fake_nano_transport,
)
from experiments.online_correction_v4.droid_pi05_policy import (
    DroidPi05PolicyAdapter,
    PI05_ACTION_SHAPE,
    fake_pi05_transport,
)
from experiments.online_correction_v4.droid_policy_request import (
    request_audit_projection,
)
from experiments.online_correction_v4.droid_reset import TwoResetAttestationProxy
from experiments.online_correction_v4.droid_simulator import (
    DroidSimulatorAdapter,
    FakeRoboLabEnv,
    FakeSettleProbe,
    import_robolab_stack,
)
from experiments.online_correction_v4.droid_reset import ResetAttestationState
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "docs/online_correction_v4/campaign.json").read_text())


def _policy_binding(*, horizon: int) -> PolicyRuntimeBinding:
    return PolicyRuntimeBinding(
        policy_id="cosmos3_nano_droid" if horizon == 32 else "pi05_droid",
        checkpoint_sha256="1" * 64,
        checkpoint_uri="file:///ckpt",
        runtime_image_digest="sha256:abc",
        integration_commit="c" * 40,
        native_control_dt_s=0.05,
        achieved_delay_s=0.10,
        achieved_standard_query_period_s=0.50,
        achieved_fast_query_period_s=0.25,
        prediction_horizon_actions=horizon,
        policy_reset_and_history_contract_uri="file:///contract",
    )


def _fixture_binding() -> FixtureRuntimeBinding:
    return FixtureRuntimeBinding(
        fixture_id="horizontal",
        geometry_sha256="3" * 64,
        scorer_sha256="4" * 64,
        reset_registry_sha256="5" * 64,
        geometry_uri="file:///geometry",
        scorer_uri="file:///scorer",
        reset_registry_uri="file:///resets",
        calibration_scale=0.12,
        d_cap_m=0.12,
    )


def _manifest(*, policy: str) -> EpisodeManifestRow:
    return EpisodeManifestRow.from_manifest_dict(
        {
            "schema_version": 1,
            "manifest_type": "planning_manifest",
            "runtime_bound": True,
            "episode_id": "ep-droid-test",
            "campaign": "online_correction_v4",
            "family": "C1",
            "fixture": "horizontal",
            "block_id": 0,
            "block_key": "k",
            "env_seed": 2100000000,
            "policy_seed": 42,
            "cohort": "confirmatory",
            "priority": "primary",
            "factors": {
                "policy": policy,
                "goal": "left",
                "wording": "direct",
                "scenario": "original_sham",
                "schedule": "standard",
                "named_reference": "bowl",
            },
            "prefix_group_id": "prefix-test",
            "execution_group": f"{policy}:horizontal",
            "execution_order_key": "000",
            "config_sha256": "abc",
            "reuse_episode_ids": [],
            "counterbalance": {"event_phase_fraction": 0.0},
            "prompt_recipe": {
                "template": "Place the cube so that the cube is left of the bowl.",
            },
        }
    )


def _runtime_lock() -> RuntimeLockBinding:
    return RuntimeLockBinding(
        schema_version=1,
        campaign_id="online_correction_v4",
        config_sha256="abc",
        manifest_sha256="def",
        release_status="RELEASED",
        released_families=("C1",),
        runner_entrypoint="tools/run_online_correction_v4.py",
        runner_sha256="b" * 64,
        prefix_mode=PrefixMode.FRESH_SESSION_DETERMINISTIC_REPLAY,
        prefix_mode_receipt_sha256="a" * 64,
        writer_contract=WriterContract(
            schema_version="v4-droid-writer-contract-v1",
            output_parent_uri="file:///persistent/v4",
            viewport_video_required=True,
            write_once_attempt_directories=True,
            incremental_fsync_required=True,
            required_streams=("viewport_video", "trajectory"),
        ),
        policies={},
        fixtures={},
        raw={},
    )


class DroidResetTests(unittest.TestCase):
    def test_two_reset_one_physical_reset(self) -> None:
        env = FakeRoboLabEnv(native_dt=0.05)
        state = ResetAttestationState(
            episode_id="ep-droid-test",
            env_seed=2100000000,
            fixture_id="horizontal",
            reset_registry_sha256="5" * 64,
        )
        proxy = TwoResetAttestationProxy(env=env, probe=FakeSettleProbe(env=env), state=state)
        first = proxy.reset(seed=2100000000)
        second = proxy.reset(seed=2100000000)
        self.assertIs(first, second)
        self.assertEqual(state.runner_pre_action_reset_calls, 2)
        self.assertEqual(state.physical_reset_calls, 1)
        self.assertTrue(state.duplicate_second_reset_idempotent)


class DroidSimulatorTests(unittest.TestCase):
    def test_native_dt_measured_from_env(self) -> None:
        env = FakeRoboLabEnv(native_dt=0.04)
        adapter = DroidSimulatorAdapter.from_fake(
            episode_id="ep",
            env_seed=1,
            fixture=_fixture_binding(),
            env=env,
        )
        adapter.reset(env_seed=1)
        self.assertAlmostEqual(adapter.native_control_dt_s, 0.04)

    def test_observation_payload_is_real_bytes(self) -> None:
        env = FakeRoboLabEnv(native_dt=0.05)
        adapter = DroidSimulatorAdapter.from_fake(
            episode_id="ep",
            env_seed=1,
            fixture=_fixture_binding(),
            env=env,
        )
        adapter.reset(env_seed=1)
        adapter.finalize_reset_attestation(
            prompt_sha256="p" * 64,
            runtime_identity_sha256="b" * 64,
            initial_state_sha256=sha256_bytes(b"initial"),
        )
        adapter.step_control((0.0,) * 8)
        packet = adapter.capture_observation()
        self.assertGreater(len(packet.payload), 0)
        decoded = json.loads(packet.payload.decode("utf-8"))
        self.assertEqual(decoded["tick"], adapter.env.control_tick)
        self.assertTrue(packet.state_hash)


class DroidPolicyAdapterTests(unittest.TestCase):
    def test_array_backed_nano_response_preserves_exact_shapes(self) -> None:
        class FakeActionArray:
            @staticmethod
            def tolist() -> list[list[float]]:
                return [[0.0] * 8 for _ in range(32)]

        class FakeFutureArray:
            shape = (33, 528, 640, 3)
            dtype = "uint8"

        actions = _normalize_action_chunk(FakeActionArray(), NANO_ACTION_SHAPE)
        future = FakeFutureArray()
        self.assertEqual((len(actions), len(actions[0])), NANO_ACTION_SHAPE)
        self.assertIs(_normalize_future(future), future)

    def test_array_inputs_are_projected_to_compact_json_evidence(self) -> None:
        class FakeArray:
            shape = (2, 2)
            dtype = "float32"

            @staticmethod
            def tobytes() -> bytes:
                return b"array-payload"

        projected = request_audit_projection(
            {"observation/image": FakeArray(), "prompt": "place"}
        )
        self.assertEqual(projected["observation/image"]["shape"], [2, 2])
        self.assertEqual(projected["observation/image"]["size_bytes"], 13)
        json.dumps(projected, allow_nan=False)

    def test_nano_preserves_32x8_envelope(self) -> None:
        prompt = "Place the cube so that the cube is left of the bowl."
        adapter = DroidNanoPolicyAdapter(
            binding=_policy_binding(horizon=32),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=fake_nano_transport(42),
        )
        adapter.reset(policy_seed=42, prompt_text=prompt)
        payload = b'{"tick":1}'
        response = adapter.infer(
            ObservationPacket(
                observation_id="obs-1",
                capture_time_s=0.05,
                payload=payload,
                payload_sha256=sha256_bytes(payload),
            )
        )
        self.assertEqual(len(response.actions), NANO_ACTION_SHAPE[0])
        self.assertEqual(len(response.actions[0]), NANO_ACTION_SHAPE[1])
        self.assertEqual(len(adapter.persisted_future_digests), 1)

    def test_pi05_preserves_15x8_and_request_seed(self) -> None:
        prompt = "Place the cube so that the cube is left of the bowl."
        adapter = DroidPi05PolicyAdapter(
            binding=_policy_binding(horizon=15),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=fake_pi05_transport(42),
        )
        adapter.reset(policy_seed=42, prompt_text=prompt)
        payload = b'{"tick":1}'
        response = adapter.infer(
            ObservationPacket(
                observation_id="obs-1",
                capture_time_s=0.05,
                payload=payload,
                payload_sha256=sha256_bytes(payload),
            )
        )
        self.assertEqual(len(response.actions), PI05_ACTION_SHAPE[0])
        self.assertEqual(adapter.records[0].request_sampling_seed, 42000)

    def test_static_prompt_is_enforced(self) -> None:
        prompt = "Place the cube so that the cube is left of the bowl."
        adapter = DroidNanoPolicyAdapter(
            binding=_policy_binding(horizon=32),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=fake_nano_transport(42),
        )
        with self.assertRaises(Exception):
            adapter.reset(policy_seed=42, prompt_text="different prompt")


class DroidBindingIntegrationTests(unittest.TestCase):
    def test_fake_binding_runs_episode_runner(self) -> None:
        timing = TimingConfig.from_mapping(CONFIG["timing"])
        manifest = _manifest(policy="cosmos3_nano_droid")
        prompt = manifest.prompt_recipe["template"]
        prompt_sha = sha256_bytes(prompt.encode("utf-8"))
        binding = build_fake_binding(
            manifest=manifest,
            lock=_runtime_lock(),
            policy_binding=_policy_binding(horizon=32),
            fixture_binding=_fixture_binding(),
            prompt_text=prompt,
            prompt_sha256=prompt_sha,
            reset_fingerprint_sha256="",
            runtime_identity_sha256="b" * 64,
            timing=timing,
            schedule=QuerySchedule.STANDARD,
        )
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            runner, _finalizer = build_episode_runner(
                binding,
                output_dir=Path(tmp),
                attempt_id="attempt-001",
                displacement_m=0.12,
                motion_direction=(1.0, 0.0),
                scenario="original_sham",
                motion_config=CONFIG["motion"],
            )
            result = runner.run()
            self.assertIn(result.attempt_status, {"valid", "infra_invalid"})

    def test_robolab_import_is_lazy(self) -> None:
        with self.assertRaises(Exception):
            import_robolab_stack()


if __name__ == "__main__":
    unittest.main()
