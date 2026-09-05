"""Import-level and fake-path tests for the V4 DROID live adapter modules."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.online_correction_v4 import geometry as geom
from experiments.online_correction_v4.adapters import ObservationPacket, SimulatorSnapshot, TerminalPhysicalPredicates, ViewportFrame
from experiments.online_correction_v4.attempts import InfraInvalidReason
from experiments.online_correction_v4.clock import QuerySchedule
from experiments.online_correction_v4.contracts import EpisodeManifestRow, EpisodeRuntimeFlags, TimingConfig
from experiments.online_correction_v4.detectors import ObjectKinematicState
from experiments.online_correction_v4.droid_bindings import (
    build_episode_runner,
    build_fake_binding,
    query_schedule_for_manifest,
)
from experiments.online_correction_v4.droid_contract import FixtureRuntimeBinding, PolicyRuntimeBinding, sha256_bytes
from experiments.online_correction_v4.droid_nano_policy import DroidNanoPolicyAdapter, NANO_ACTION_CHUNK_STEPS, fake_nano_transport
from experiments.online_correction_v4.droid_observation import FakeObservationPacker, pack_policy_request
from experiments.online_correction_v4.droid_policy_request import (
    PolicyInfraInvalidError,
    ServerEnvelopeGateError,
    build_v4_request_envelope,
    normalize_pi05_response,
    observation_packed_request,
)
from experiments.online_correction_v4.droid_pi05_policy import DroidPi05PolicyAdapter, PI05_ACTION_SHAPE
from experiments.online_correction_v4.droid_reset import ResetAttestationState, TwoResetAttestationProxy
from experiments.online_correction_v4.droid_reset_verify import (
    verify_measured_native_dt,
    verify_neutral_horizontal_layout,
    verify_physical_reset_against_registry,
)
from experiments.online_correction_v4.droid_scorer import HorizontalDroidTerminalScorer, load_scoring_context
from experiments.online_correction_v4.droid_simulator import FakeRoboLabEnv, DroidSimulatorAdapter, FakeSettleProbe
from experiments.online_correction_v4.droid_task_files.registry import (
    blocked_fixture_ids,
    resolve_active_registration,
    resolve_fixture_registration,
    supported_fixture_ids,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import ObjectRoleBinding, ResetRegistry
from experiments.online_correction_v4.droid_transport import EpisodePolicyTransport, TransportError
from experiments.online_correction_v4.scoring import ScoringContext
from experiments.online_correction_v4.testing import ScriptedTerminalScorer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "docs/online_correction_v4/campaign.json").read_text())


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


def _scoring_context() -> ScoringContext:
    workspace = geom.AxisAlignedBox(-1.0, 1.0, -1.0, 1.0, 0.0, 0.2)
    foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
    return ScoringContext(
        frame=geom.TaskFrame.identity(),
        d_cap_m=0.12,
        planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
    )


class DroidLiveImportTests(unittest.TestCase):
    def test_live_modules_import(self) -> None:
        import experiments.online_correction_v4.droid_observation as observation
        import experiments.online_correction_v4.droid_robolab as robolab
        import experiments.online_correction_v4.droid_scorer as scorer
        import experiments.online_correction_v4.droid_transport as transport

        self.assertTrue(hasattr(observation, "pack_policy_request"))
        self.assertTrue(hasattr(robolab, "build_live_robolab_env"))
        self.assertTrue(hasattr(scorer, "build_terminal_scorer"))
        self.assertTrue(hasattr(transport, "build_live_transport"))

    def test_live_world_tensor_conversion_moves_data_to_cpu(self) -> None:
        import numpy as np

        from experiments.online_correction_v4.droid_robolab import _host_numpy

        calls: list[str] = []

        class _CudaLikeTensor:
            def detach(self):
                calls.append("detach")
                return self

            def cpu(self):
                calls.append("cpu")
                return self

            def numpy(self):
                calls.append("numpy")
                return np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

        converted = _host_numpy(_CudaLikeTensor())
        self.assertEqual(calls, ["detach", "cpu", "numpy"])
        self.assertEqual(converted.tolist(), [1.0, 2.0, 3.0])

    def test_live_action_converter_accepts_batched_hold_tensor(self) -> None:
        from experiments.online_correction_v4.droid_robolab import LiveRoboLabBackend

        calls: list[tuple[str, object, object]] = []

        class _Tensor:
            shape = (1, 8)

            def detach(self):
                calls.append(("detach", None, None))
                return self

            def to(self, *, device, dtype):
                calls.append(("to", device, dtype))
                return self

        fake_torch = SimpleNamespace(Tensor=_Tensor, float32="float32")
        backend = object.__new__(LiveRoboLabBackend)
        backend.env = SimpleNamespace(device="cuda:0")
        action = _Tensor()
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertIs(backend._action_tensor(action), action)
        self.assertEqual(
            calls,
            [("detach", None, None), ("to", "cuda:0", "float32")],
        )

    def test_horizontal_fixture_registry(self) -> None:
        self.assertEqual(supported_fixture_ids(), ("horizontal",))
        reg = resolve_fixture_registration("horizontal", relation="left")
        self.assertTrue(reg.timeout_only)
        self.assertTrue(reg.robolab_success_termination_forbidden)
        with self.assertRaises(Exception):
            resolve_fixture_registration("reference_binding")
        self.assertIn("reference_binding", blocked_fixture_ids())


class DroidObservationTransportTests(unittest.TestCase):
    def test_fake_packer_includes_rgb_and_proprio_keys(self) -> None:
        packed = FakeObservationPacker().pack({"tick": 3}, "prompt")
        self.assertIn("observation/joint_position", packed)
        self.assertIn("observation/exterior_image_1_left", packed)
        self.assertEqual(packed["prompt"], "prompt")

    def test_pack_policy_request_uses_injected_packer(self) -> None:
        packed = pack_policy_request(
            policy_id="cosmos3_nano_droid",
            native_obs={"tick": 1},
            instruction="Place the cube.",
            packer=FakeObservationPacker(),
        )
        self.assertIn("observation/wrist_image_left", packed)

    def test_observation_packed_request_reads_payload_fallback(self) -> None:
        payload = json.dumps(
            {"packed_request": {"prompt": "hello", "observation/joint_position": [0.0]}}
        ).encode("utf-8")
        packet = ObservationPacket(
            observation_id="obs",
            capture_time_s=0.0,
            payload=payload,
            payload_sha256=sha256_bytes(payload),
        )
        packed = observation_packed_request(packet)
        self.assertEqual(packed["prompt"], "hello")

    def test_pi05_normalizes_actions_key(self) -> None:
        normalized = normalize_pi05_response({"actions": [[0.0] * 8], "sampling_seed": 1})
        self.assertIn("action", normalized)
        self.assertEqual(normalized["action"], normalized["actions"])

    def test_pi05_adapter_accepts_actions_response(self) -> None:
        from experiments.online_correction_v4.droid_contract import PolicyRuntimeBinding

        prompt = "Place the cube so that the cube is left of the bowl."
        adapter = DroidPi05PolicyAdapter(
            binding=PolicyRuntimeBinding(
                policy_id="pi05_droid",
                checkpoint_sha256="1" * 64,
                checkpoint_uri="file:///ckpt",
                runtime_image_digest="sha256:abc",
                integration_commit="c" * 40,
                native_control_dt_s=0.05,
                achieved_delay_s=0.10,
                achieved_standard_query_period_s=0.50,
                achieved_fast_query_period_s=0.25,
                prediction_horizon_actions=15,
                policy_reset_and_history_contract_uri="file:///contract",
            ),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=lambda request: {
                "actions": [[0.01 if col == 0 else 0.0 for col in range(8)] for _ in range(PI05_ACTION_SHAPE[0])],
                "v2a010_sampling_seed": request["sampling_seed"],
            },
        )
        adapter.reset(policy_seed=42, prompt_text=prompt)
        payload = json.dumps(
            {
                "packed_request": FakeObservationPacker().pack({"tick": 1}, prompt),
            }
        ).encode("utf-8")
        response = adapter.infer(
            ObservationPacket(
                observation_id="obs-1",
                capture_time_s=0.05,
                payload=payload,
                payload_sha256=sha256_bytes(payload),
            )
        )
        self.assertEqual(len(response.actions), PI05_ACTION_SHAPE[0])


class DroidSimulatorSnapshotTests(unittest.TestCase):
    def test_reference_displacement_propagates_in_snapshot(self) -> None:
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
        adapter.set_reference_offset(0.06, (1.0, 0.0))
        snapshot = adapter.step_control((0.0,) * 8)
        self.assertAlmostEqual(snapshot.reference_displacement_m, 0.06)


class DroidTerminalScorerTests(unittest.TestCase):
    def test_horizontal_terminal_scorer_scores_snapshot(self) -> None:
        scorer = HorizontalDroidTerminalScorer(relation="left", ctx=_scoring_context())
        snapshot = SimulatorSnapshot(
            sim_time=1.0,
            control_tick=20,
            object_state=ObjectKinematicState(
                sim_time=1.0,
                control_tick=20,
                object_z=0.04,
                initial_supported_z=0.04,
                gripper_x=0.0,
                gripper_y=0.0,
                gripper_z=0.2,
                object_x=0.2,
                object_y=0.05,
                object_z_pos=0.04,
                contact=False,
                detached=True,
            ),
            reference_position_world=(0.0, 0.0, 0.0),
        )
        evidence = scorer.score_terminal(
            snapshot=snapshot,
            runtime_flags=EpisodeRuntimeFlags(trigger_eligible=True, event_delivered=True),
            passive_settling_reason="release",
        )
        self.assertTrue(evidence.released)
        self.assertIn("failure_label", evidence.metadata)

    def test_load_scoring_context_from_geometry_file(self) -> None:
        geometry = {
            "task_frame": {
                "u_left": [1.0, 0.0, 0.0],
                "u_front": [0.0, 1.0, 0.0],
                "u_up": [0.0, 0.0, 1.0],
                "origin": [0.0, 0.0, 0.0],
            },
            "workspace": {
                "x_min": -1.0,
                "x_max": 1.0,
                "y_min": -1.0,
                "y_max": 1.0,
                "z_min": 0.0,
                "z_max": 0.2,
            },
            "object_footprint": {"half_left": 0.02, "half_front": 0.02, "half_up": 0.02},
            "reference_footprint": {"half_left": 0.03, "half_front": 0.03, "half_up": 0.03},
            "clearance_m": 0.01,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "geometry.json"
            path.write_text(json.dumps(geometry), encoding="utf-8")
            digest = sha256_bytes(path.read_bytes())
            ctx = load_scoring_context(path, expected_sha256=digest, relation="left", d_cap_m=0.12)
            self.assertIsNotNone(ctx.planar_spec)


class DroidRunnerBindingTests(unittest.TestCase):
    def test_episode_runner_receives_terminal_scorer(self) -> None:
        from experiments.online_correction_v4.clock import QuerySchedule
        from experiments.online_correction_v4.contracts import TimingConfig
        from experiments.online_correction_v4.droid_contract import PolicyRuntimeBinding, PrefixMode, RuntimeLockBinding, WriterContract
        from tests.test_online_correction_v4_droid_adapters import _manifest, _runtime_lock

        timing = TimingConfig.from_mapping(CONFIG["timing"])
        manifest = _manifest(policy="cosmos3_nano_droid")
        prompt = manifest.prompt_recipe["template"]
        scorer = ScriptedTerminalScorer(success=False)
        binding = build_fake_binding(
            manifest=manifest,
            lock=_runtime_lock(),
            policy_binding=PolicyRuntimeBinding(
                policy_id="cosmos3_nano_droid",
                checkpoint_sha256="1" * 64,
                checkpoint_uri="file:///ckpt",
                runtime_image_digest="sha256:abc",
                integration_commit="c" * 40,
                native_control_dt_s=0.05,
                achieved_delay_s=0.10,
                achieved_standard_query_period_s=0.50,
                achieved_fast_query_period_s=0.25,
                prediction_horizon_actions=32,
                policy_reset_and_history_contract_uri="file:///contract",
            ),
            fixture_binding=_fixture_binding(),
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="",
            runtime_identity_sha256="b" * 64,
            timing=timing,
            schedule=QuerySchedule.STANDARD,
            terminal_scorer=scorer,
        )
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
            self.assertIs(runner.terminal_scorer, scorer)


class DroidLiveBlockerTests(unittest.TestCase):
    def test_build_live_robolab_env_fails_without_deps(self) -> None:
        from experiments.online_correction_v4.droid_contract import DroidContractError
        from experiments.online_correction_v4.droid_bindings import build_live_binding
        from experiments.online_correction_v4.clock import QuerySchedule
        from experiments.online_correction_v4.contracts import TimingConfig
        from experiments.online_correction_v4.droid_contract import PolicyRuntimeBinding
        from tests.test_online_correction_v4_droid_adapters import _manifest, _runtime_lock

        timing = TimingConfig.from_mapping(CONFIG["timing"])
        manifest = _manifest(policy="cosmos3_nano_droid")
        prompt = manifest.prompt_recipe["template"]
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch(
                "experiments.online_correction_v4.droid_robolab.RoboLabSession.ensure_started",
                side_effect=__import__(
                    "experiments.online_correction_v4.droid_simulator",
                    fromlist=["DroidDependencyError"],
                ).DroidDependencyError("cv2 unavailable"),
            ):
                with self.assertRaises(DroidContractError):
                    build_live_binding(
                        manifest=manifest,
                        lock=_runtime_lock(),
                        policy_binding=PolicyRuntimeBinding(
                            policy_id="cosmos3_nano_droid",
                            checkpoint_sha256="1" * 64,
                            checkpoint_uri="file:///ckpt",
                            runtime_image_digest="sha256:abc",
                            integration_commit="c" * 40,
                            native_control_dt_s=0.05,
                            achieved_delay_s=0.10,
                            achieved_standard_query_period_s=0.50,
                            achieved_fast_query_period_s=0.25,
                            prediction_horizon_actions=32,
                            policy_reset_and_history_contract_uri="file:///contract",
                        ),
                        fixture_binding=_fixture_binding(),
                        prompt_text=prompt,
                        prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
                        runtime_identity_sha256="b" * 64,
                        timing=timing,
                        schedule=QuerySchedule.STANDARD,
                        policy_host="127.0.0.1",
                        policy_port=18011,
                        output_dir=Path(tmp),
                    )

    def test_live_transport_fails_without_policy_client(self) -> None:
        from experiments.online_correction_v4.droid_transport import TransportError, build_live_transport

        with mock.patch(
            "experiments.online_correction_v4.droid_transport._create_nano_client",
            side_effect=ImportError("policies.cosmos3.client unavailable"),
        ):
            with self.assertRaises(TransportError):
                build_live_transport(policy_id="cosmos3_nano_droid", host="127.0.0.1", port=1)


class DroidReviewFixTests(unittest.TestCase):
    def test_query_schedule_maps_fast_after_grasp(self) -> None:
        from tests.test_online_correction_v4_droid_adapters import _manifest

        base = _manifest(policy="cosmos3_nano_droid")
        for schedule_name in ("fast", "fast_after_grasp"):
            manifest = replace(base, factors={**base.factors, "schedule": schedule_name})
            self.assertIs(query_schedule_for_manifest(manifest), QuerySchedule.FAST_AFTER_GRASP)

    def test_wire_request_excludes_v4_audit_keys(self) -> None:
        packed = FakeObservationPacker().pack({"tick": 1}, "prompt")
        audit = {
            "schema_version": "v4-droid-nano-request-v1",
            "episode_id": "ep",
            "request_index": 0,
            "reset_fingerprint_sha256": "r" * 64,
        }
        wire, _full = build_v4_request_envelope(
            policy_id="cosmos3_nano_droid",
            packed=packed,
            audit=audit,
        )
        self.assertNotIn("schema_version", wire)
        self.assertNotIn("episode_id", wire)
        self.assertNotIn("reset_fingerprint_sha256", wire)
        self.assertIn("prompt", wire)
        self.assertIn("observation/joint_position", wire)

    def test_action_step_start_uses_executed_count_not_request_index(self) -> None:
        from experiments.online_correction_v4.droid_contract import PolicyRuntimeBinding

        prompt = "Place the cube so that the cube is left of the bowl."
        executed = 7
        adapter = DroidNanoPolicyAdapter(
            binding=PolicyRuntimeBinding(
                policy_id="cosmos3_nano_droid",
                checkpoint_sha256="1" * 64,
                checkpoint_uri="file:///ckpt",
                runtime_image_digest="sha256:abc",
                integration_commit="c" * 40,
                native_control_dt_s=0.05,
                achieved_delay_s=0.10,
                achieved_standard_query_period_s=0.50,
                achieved_fast_query_period_s=0.25,
                prediction_horizon_actions=32,
                policy_reset_and_history_contract_uri="file:///contract",
            ),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=fake_nano_transport(42),
            executed_action_count=lambda: executed,
        )
        adapter.reset(policy_seed=42, prompt_text=prompt)
        adapter.request_count = 2
        self.assertEqual(adapter._action_step_start(), executed)
        self.assertNotEqual(adapter._action_step_start(), adapter.request_count * NANO_ACTION_CHUNK_STEPS)

    def test_nano_missing_future_is_infra_invalid_with_artifact(self) -> None:
        from experiments.online_correction_v4.droid_contract import PolicyRuntimeBinding

        prompt = "Place the cube so that the cube is left of the bowl."
        adapter = DroidNanoPolicyAdapter(
            binding=PolicyRuntimeBinding(
                policy_id="cosmos3_nano_droid",
                checkpoint_sha256="1" * 64,
                checkpoint_uri="file:///ckpt",
                runtime_image_digest="sha256:abc",
                integration_commit="c" * 40,
                native_control_dt_s=0.05,
                achieved_delay_s=0.10,
                achieved_standard_query_period_s=0.50,
                achieved_fast_query_period_s=0.25,
                prediction_horizon_actions=32,
                policy_reset_and_history_contract_uri="file:///contract",
            ),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=lambda _request: {"action": [[0.0] * 8] * 32, "sampling_seed": 42},
        )
        adapter.reset(policy_seed=42, prompt_text=prompt)
        payload = json.dumps({"packed_request": FakeObservationPacker().pack({"tick": 1}, prompt)}).encode("utf-8")
        with self.assertRaises(PolicyInfraInvalidError) as ctx:
            adapter.infer(
                ObservationPacket(
                    observation_id="obs-1",
                    capture_time_s=0.05,
                    payload=payload,
                    payload_sha256=sha256_bytes(payload),
                )
            )
        self.assertIs(ctx.exception.reason, InfraInvalidReason.MISSING_MANDATORY_STREAM)
        self.assertIsNotNone(ctx.exception.future_artifact)
        self.assertEqual(ctx.exception.future_artifact.kind, "missing_future")
        self.assertEqual(len(adapter.persisted_future_digests), 1)

    def test_settle_resets_control_tick_via_on_settle_complete(self) -> None:
        env = FakeRoboLabEnv(native_dt=0.05)
        probe = FakeSettleProbe(env=env)
        state = ResetAttestationState(
            episode_id="ep",
            env_seed=1,
            fixture_id="horizontal",
            reset_registry_sha256="5" * 64,
        )
        proxy = TwoResetAttestationProxy(env=env, probe=probe, state=state)
        env.control_tick = 12
        proxy.reset(seed=1)
        proxy.reset(seed=1)
        self.assertEqual(env.control_tick, 0)

    def test_passive_hold_does_not_double_step(self) -> None:
        env = FakeRoboLabEnv(native_dt=0.05)
        env.reset(seed=1)
        before = env.control_tick
        env.hold_robot_target((0.1,) * 8)
        self.assertEqual(env.control_tick, before)
        self.assertEqual(env.last_hold_action, (0.1,) * 8)
        env.step((0.1,) * 8)
        self.assertEqual(env.control_tick, before + 1)

    def test_active_registration_uses_single_task_module(self) -> None:
        reg = resolve_active_registration()
        self.assertTrue(reg.task_module.endswith("horizontal_active.py"))
        self.assertEqual(reg.task_class, "V4HorizontalActiveTask")
        self.assertIn("active_episode_bound", reg.attributes)

    def test_reset_verification_checks_registry_neutrality_and_dt(self) -> None:
        registry = ResetRegistry(
            schema_version="v4-droid-horizontal-reset-registry-v1",
            fixture_id="horizontal",
            status="model_blind_candidate_not_released_for_inference",
            model_request_count=0,
            behavioral_episode_count=0,
            scene_asset="scene",
            scene_metadata_sha256="m" * 64,
            contact_objects=("rubiks_cube", "banana", "bowl"),
            object_roles={
                "target": ObjectRoleBinding("target", "rubiks_cube", "id"),
                "reference": ObjectRoleBinding("reference", "bowl", "id"),
                "distractor": ObjectRoleBinding("distractor", "banana", "id"),
            },
            positions_by_env_seed={
                2100000000: {
                    "rubiks_cube": (0.45, 0.0, 0.83),
                    "bowl": (0.55, 0.12, 0.82),
                    "banana": (0.40, -0.10, 0.81),
                }
            },
            registry_path="/tmp/resets.jsonl",
            registry_sha256="5" * 64,
        )
        physical = {
            "objects": {
                "rubiks_cube": {"position_robot_xyz_m": [0.50, 0.12, 0.83]},
                "bowl": {"position_robot_xyz_m": [0.55, 0.12, 0.82]},
                "banana": {"position_robot_xyz_m": [0.40, -0.10, 0.81]},
            },
            "robot_position_robot_xyz_m": [0.0, 0.0, 0.0],
        }
        verify_physical_reset_against_registry(
            {
                **physical,
                "objects": {
                    "rubiks_cube": {"position_robot_xyz_m": [0.45, 0.0, 0.83]},
                    "bowl": {"position_robot_xyz_m": [0.55, 0.12, 0.82]},
                    "banana": {"position_robot_xyz_m": [0.40, -0.10, 0.81]},
                },
            },
            registry=registry,
            env_seed=2100000000,
        )
        neutral = {
            "objects": {
                "rubiks_cube": {"position_robot_xyz_m": [0.50, 0.12, 0.83]},
                "bowl": {"position_robot_xyz_m": [0.55, 0.12, 0.82]},
            }
        }
        verify_neutral_horizontal_layout(neutral)
        verify_measured_native_dt(measured_s=0.05, locked_s=0.05)
        biased = {
            "objects": {
                "rubiks_cube": {"position_robot_xyz_m": [0.45, 0.25, 0.83]},
                "bowl": {"position_robot_xyz_m": [0.55, 0.12, 0.82]},
            }
        }
        with self.assertRaises(Exception):
            verify_neutral_horizontal_layout(biased)

    def test_terminal_scorer_uses_task_frame_from_geometry_lock(self) -> None:
        workspace = geom.AxisAlignedBox(-1.0, 1.0, -1.0, 1.0, 0.0, 0.2)
        foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        identity_ctx = ScoringContext(
            frame=geom.TaskFrame.identity(),
            d_cap_m=0.12,
            planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
        )
        rotated = geom.TaskFrame(
            u_left=(0.0, 1.0, 0.0),
            u_front=(-1.0, 0.0, 0.0),
            u_up=(0.0, 0.0, 1.0),
            origin=(0.0, 0.0, 0.0),
        )
        rotated_ctx = ScoringContext(
            frame=rotated,
            d_cap_m=0.12,
            planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
        )
        snapshot = SimulatorSnapshot(
            sim_time=1.0,
            control_tick=20,
            object_state=ObjectKinematicState(
                sim_time=1.0,
                control_tick=20,
                object_z=0.04,
                initial_supported_z=0.04,
                gripper_x=0.0,
                gripper_y=0.0,
                gripper_z=0.2,
                object_x=0.2,
                object_y=0.03,
                object_z_pos=0.04,
                contact=False,
                detached=True,
            ),
            reference_position_world=(0.0, 0.0, 0.0),
        )
        identity_score = HorizontalDroidTerminalScorer(relation="left", ctx=identity_ctx).score_terminal(
            snapshot=snapshot,
            runtime_flags=EpisodeRuntimeFlags(trigger_eligible=True, event_delivered=True),
            passive_settling_reason="release",
            grasp_occurred=True,
            carry_verified=True,
            settling_predicates=(
                TerminalPhysicalPredicates(
                    available=True,
                    allowed_support=True,
                    stable_for_dwell=True,
                ),
                TerminalPhysicalPredicates(
                    available=True,
                    allowed_support=True,
                    stable_for_dwell=True,
                ),
            ),
        )
        rotated_score = HorizontalDroidTerminalScorer(relation="left", ctx=rotated_ctx).score_terminal(
            snapshot=snapshot,
            runtime_flags=EpisodeRuntimeFlags(trigger_eligible=True, event_delivered=True),
            passive_settling_reason="release",
            grasp_occurred=True,
            carry_verified=True,
            settling_predicates=(
                TerminalPhysicalPredicates(
                    available=True,
                    allowed_support=True,
                    stable_for_dwell=True,
                ),
                TerminalPhysicalPredicates(
                    available=True,
                    allowed_support=True,
                    stable_for_dwell=True,
                ),
            ),
        )
        self.assertNotEqual(
            identity_score.geometric_relation_correct,
            rotated_score.geometric_relation_correct,
        )

    def test_episode_transport_requires_fresh_session(self) -> None:
        transport = EpisodePolicyTransport(policy_id="cosmos3_nano_droid", host="127.0.0.1", port=1)
        with self.assertRaises(TransportError):
            transport({"prompt": "x", "sampling_seed": 1, "action_step_start": 0})

    def test_server_envelope_rejection_surfaces_live_gate(self) -> None:
        class _RejectClient:
            def _query_server(self, _wire: dict) -> None:
                raise RuntimeError("unknown field schema_version in request")

        transport = EpisodePolicyTransport(policy_id="cosmos3_nano_droid", host="127.0.0.1", port=1)
        transport._client = _RejectClient()
        with self.assertRaises(ServerEnvelopeGateError):
            transport(
                {
                    "prompt": "hello",
                    "sampling_seed": 1,
                    "action_step_start": 0,
                    "observation/joint_position": [0.0],
                }
            )

    def test_fast_after_grasp_manifest_builds_droid_runner(self) -> None:
        from tests.test_online_correction_v4_droid_adapters import _manifest, _runtime_lock

        timing = TimingConfig.from_mapping(CONFIG["timing"])
        base = _manifest(policy="cosmos3_nano_droid")
        manifest = replace(base, factors={**base.factors, "schedule": "fast_after_grasp"})
        prompt = manifest.prompt_recipe["template"]
        binding = build_fake_binding(
            manifest=manifest,
            lock=_runtime_lock(),
            policy_binding=PolicyRuntimeBinding(
                policy_id="cosmos3_nano_droid",
                checkpoint_sha256="1" * 64,
                checkpoint_uri="file:///ckpt",
                runtime_image_digest="sha256:abc",
                integration_commit="c" * 40,
                native_control_dt_s=0.05,
                achieved_delay_s=0.10,
                achieved_standard_query_period_s=0.50,
                achieved_fast_query_period_s=0.25,
                prediction_horizon_actions=32,
                policy_reset_and_history_contract_uri="file:///contract",
            ),
            fixture_binding=_fixture_binding(),
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="",
            runtime_identity_sha256="b" * 64,
            timing=timing,
            schedule=QuerySchedule.FAST_AFTER_GRASP,
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner, _finalizer = build_episode_runner(
                binding,
                output_dir=Path(tmp),
                attempt_id="attempt-fast",
                displacement_m=0.12,
                motion_direction=(1.0, 0.0),
                scenario="original_sham",
                motion_config=CONFIG["motion"],
            )
            self.assertEqual(runner.clock.schedule, QuerySchedule.FAST_AFTER_GRASP)

    def test_nano_future_persisted_as_npz_without_retaining_payload(self) -> None:
        from experiments.online_correction_v4.droid_contract import PolicyRuntimeBinding

        prompt = "Place the cube so that the cube is left of the bowl."
        adapter = DroidNanoPolicyAdapter(
            binding=PolicyRuntimeBinding(
                policy_id="cosmos3_nano_droid",
                checkpoint_sha256="1" * 64,
                checkpoint_uri="file:///ckpt",
                runtime_image_digest="sha256:abc",
                integration_commit="c" * 40,
                native_control_dt_s=0.05,
                achieved_delay_s=0.10,
                achieved_standard_query_period_s=0.50,
                achieved_fast_query_period_s=0.25,
                prediction_horizon_actions=32,
                policy_reset_and_history_contract_uri="file:///contract",
            ),
            episode_id="ep",
            policy_seed=42,
            prompt_text=prompt,
            prompt_sha256=sha256_bytes(prompt.encode("utf-8")),
            reset_fingerprint_sha256="r" * 64,
            runtime_identity_sha256="b" * 64,
            transport=fake_nano_transport(42),
        )
        adapter.reset(policy_seed=42, prompt_text=prompt)
        payload = json.dumps({"packed_request": FakeObservationPacker().pack({"tick": 1}, prompt)}).encode("utf-8")
        response = adapter.infer(
            ObservationPacket(
                observation_id="obs-1",
                capture_time_s=0.05,
                payload=payload,
                payload_sha256=sha256_bytes(payload),
            )
        )
        self.assertIsNotNone(response.future_artifact)
        assert response.future_artifact is not None
        self.assertIn(
            response.future_artifact.kind,
            ("decoded_rgb_future_npz", "decoded_rgb_future_zjson"),
        )
        self.assertGreater(len(response.future_artifact.payload), 0)
        self.assertEqual(len(adapter.persisted_future_digests), 1)
        self.assertEqual(adapter.persisted_future_digests[0], response.future_artifact.payload_sha256)

    def test_cosmos_extract_observation_matches_pinned_v3_signature(self) -> None:
        captured: dict[str, object] = {}

        class _StubCosmos:
            IMAGE_W = 640
            IMAGE_H = 540

            @staticmethod
            def _extract_observation(packer, obs):  # noqa: ANN001
                captured["args"] = (packer, obs)
                return {"image": obs}

            @staticmethod
            def _pack_request(packer, extracted, instruction):  # noqa: ANN001
                return {"prompt": instruction, "observation/image": extracted["image"]}

        stub_module = mock.MagicMock()
        stub_module.Cosmos3Client = _StubCosmos
        with mock.patch.dict(
            "sys.modules",
            {
                "policies": mock.MagicMock(),
                "policies.cosmos3": mock.MagicMock(),
                "policies.cosmos3.client": stub_module,
            },
        ):
            from experiments.online_correction_v4.droid_observation import _NanoClientPacker

            packer = _NanoClientPacker()
            packed = packer.pack({"tick": 3}, "prompt")
        self.assertEqual(len(captured["args"]), 2)  # type: ignore[arg-type]
        self.assertEqual(packed["prompt"], "prompt")

    def test_live_backend_reset_clears_kinematic_cache(self) -> None:
        from experiments.online_correction_v4.droid_robolab import LiveRoboLabBackend, LiveRoboLabConfig

        config = LiveRoboLabConfig(
            episode_id="ep",
            env_seed=1,
            goal="left",
            prompt_text="prompt",
            prompt_sha256="p" * 64,
            policy_id="cosmos3_nano_droid",
            fixture=_fixture_binding(),
            queue_row_path=Path("/tmp/queue.json"),
            queue_row_sha256="q" * 64,
        )
        backend = LiveRoboLabBackend(env=object(), config=config, modules={})
        backend._anchor_reference_motion = lambda: None  # type: ignore[method-assign]
        backend._initial_supported_z = 0.83
        backend._reference_baseline_pose = (1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
        backend.last_hold_action = (0.1,) * 8
        backend.control_tick = 9

        class _Env:
            def reset(self, *, seed: int):
                return {"tick": 0}, {}

        backend.env = _Env()
        backend.reset(seed=1)
        self.assertEqual(backend._initial_supported_z, 0.0)
        self.assertIsNone(backend._reference_baseline_pose)
        self.assertEqual(backend.control_tick, 0)
        self.assertEqual(backend.last_hold_action, ())

    def test_opencv_viewport_writer_encodes_without_retaining_frames(self) -> None:
        import tempfile

        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV unavailable")
        from experiments.online_correction_v4.viewport_video import OpenCVViewportVideoWriter

        image = np.zeros((32, 48, 3), dtype=np.uint8)
        image[:, :, 0] = 200
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        payload = encoded.tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            attempt_path = Path(tmp)
            writer = OpenCVViewportVideoWriter(fps=20.0)
            writer.bind_attempt_path(attempt_path)
            writer.append_frame(
                ViewportFrame(
                    frame_index=0,
                    sim_time_s=0.0,
                    control_tick=0,
                    payload=payload,
                    payload_sha256=sha256_bytes(payload),
                    format_kind="encoded_image",
                    width=48,
                    height=32,
                    channels=3,
                )
            )
            artifact = writer.finalize_video(attempt_path=attempt_path)
            self.assertEqual(artifact.frame_count, 1)
            self.assertTrue((attempt_path / artifact.relative_path).exists())
            self.assertGreater(artifact.size_bytes, 0)

    def test_attest_encoded_and_raw_viewport_formats(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV unavailable")
        from experiments.online_correction_v4.viewport_video import (
            attest_viewport_capture,
            capture_from_ndarray,
            decode_viewport_capture,
        )

        image = np.zeros((12, 10, 3), dtype=np.uint8)
        image[2:8, 2:8, 1] = 180
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        encoded_capture = attest_viewport_capture(encoded.tobytes(), format_kind="encoded_png")
        self.assertEqual(encoded_capture.format_kind, "encoded_png")
        decoded = decode_viewport_capture(encoded_capture)
        self.assertEqual(decoded.shape[:2], (12, 10))

        raw_capture = capture_from_ndarray(image, format_kind="raw_rgb24")
        self.assertEqual(raw_capture.width, 10)
        self.assertEqual(raw_capture.height, 12)
        self.assertEqual(len(raw_capture.payload), 12 * 10 * 3)
        decoded_raw = decode_viewport_capture(raw_capture)
        self.assertEqual(decoded_raw.shape[:2], (12, 10))

        attested_bytes = attest_viewport_capture(
            raw_capture.payload,
            format_kind="raw_rgb24",
            width=10,
            height=12,
            channels=3,
        )
        self.assertEqual(attested_bytes.payload_sha256, raw_capture.payload_sha256)

    def test_attest_fail_closed_on_unattested_bytes(self) -> None:
        from experiments.online_correction_v4.adapters import ViewportVideoRequiredError
        from experiments.online_correction_v4.viewport_video import attest_viewport_capture

        with self.assertRaises(ViewportVideoRequiredError):
            attest_viewport_capture(b"not-an-image")
        with self.assertRaises(ViewportVideoRequiredError):
            attest_viewport_capture(
                b"\x01" * 10,
                format_kind="raw_rgb24",
                width=2,
                height=2,
                channels=3,
            )

    def test_robolab_session_single_episode_guard(self) -> None:
        from experiments.online_correction_v4.droid_robolab import RoboLabBootstrapError, RoboLabSession

        RoboLabSession.end_episode()
        RoboLabSession.begin_episode("ep-a")
        with self.assertRaises(RoboLabBootstrapError):
            RoboLabSession.begin_episode("ep-b")
        RoboLabSession.end_episode()

    def test_live_backend_viewport_recorder_formats(self) -> None:
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV unavailable")
        from experiments.online_correction_v4.droid_robolab import LiveRoboLabBackend, LiveRoboLabConfig
        from experiments.online_correction_v4.viewport_video import ViewportCapture

        config = LiveRoboLabConfig(
            episode_id="ep",
            env_seed=1,
            goal="left",
            prompt_text="prompt",
            prompt_sha256="p" * 64,
            policy_id="cosmos3_nano_droid",
            fixture=_fixture_binding(),
            queue_row_path=Path("/tmp/queue.json"),
            queue_row_sha256="q" * 64,
        )
        backend = LiveRoboLabBackend(env=object(), config=config, modules={})

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, :, 2] = 90
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)

        class _EncodedRecorder:
            latest_frame_bytes = encoded.tobytes()

        backend.env = type("Env", (), {"viewport_recorder": _EncodedRecorder()})()
        encoded_capture = backend.capture_viewport_frame()
        self.assertIsInstance(encoded_capture, ViewportCapture)
        assert isinstance(encoded_capture, ViewportCapture)
        self.assertEqual(encoded_capture.format_kind, "encoded_image")

        class _RawNdarrayRecorder:
            latest_frame_rgb = image

        backend.env = type("Env", (), {"viewport_recorder": _RawNdarrayRecorder()})()
        raw_capture = backend.capture_viewport_frame()
        self.assertIsInstance(raw_capture, ViewportCapture)
        assert isinstance(raw_capture, ViewportCapture)
        self.assertEqual(raw_capture.format_kind, "raw_rgb24")

        class _RawBytesRecorder:
            frame_width = 8
            frame_height = 8
            pixel_format = "raw_rgb24"
            latest_frame_bytes = image.tobytes()

        backend.env = type("Env", (), {"viewport_recorder": _RawBytesRecorder()})()
        bytes_capture = backend.capture_viewport_frame()
        self.assertIsInstance(bytes_capture, ViewportCapture)
        assert isinstance(bytes_capture, ViewportCapture)
        self.assertEqual(bytes_capture.format_kind, "raw_rgb24")

    def test_ffmpeg_viewport_writer_when_available(self) -> None:
        import os
        import shutil
        import tempfile

        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV unavailable")
        ffmpeg = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
        if not ffmpeg:
            self.skipTest("ffmpeg unavailable")
        os.environ["FFMPEG_BIN"] = str(Path(ffmpeg).resolve())
        from experiments.online_correction_v4.viewport_video import FfmpegViewportVideoWriter

        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :, 0] = 120
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        payload = encoded.tobytes()
        with tempfile.TemporaryDirectory() as tmp:
            attempt_path = Path(tmp)
            writer = FfmpegViewportVideoWriter(fps=20.0)
            writer.bind_attempt_path(attempt_path)
            writer.append_frame(
                ViewportFrame(
                    frame_index=0,
                    sim_time_s=0.0,
                    control_tick=0,
                    payload=payload,
                    payload_sha256=sha256_bytes(payload),
                    format_kind="encoded_image",
                    width=16,
                    height=16,
                    channels=3,
                )
            )
            artifact = writer.finalize_video(attempt_path=attempt_path)
            self.assertEqual(artifact.frame_count, 1)
            self.assertIn("ffmpeg/", artifact.codec)
            self.assertTrue((attempt_path / artifact.relative_path).exists())

    def test_close_live_droid_stack_idempotent(self) -> None:
        from experiments.online_correction_v4.droid_robolab import RoboLabSession, close_live_droid_stack

        closed = {"count": 0}

        class _Transport:
            def close(self) -> None:
                closed["count"] += 1

        class _Policy:
            transport = _Transport()

        RoboLabSession.begin_episode("ep-close")
        close_live_droid_stack(policy=_Policy())
        close_live_droid_stack(policy=_Policy())
        self.assertEqual(closed["count"], 1)
        self.assertFalse(RoboLabSession._episode_active)


if __name__ == "__main__":
    unittest.main()
