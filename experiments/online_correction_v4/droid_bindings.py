"""Bind V4 DROID simulator and policy adapters for one registered episode."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable

from experiments.online_correction_v4.adapters import ObservationPacket, PolicyAdapter, PolicyModelInvalidError, SimulatorAdapter, TerminalScoringAdapter
from experiments.online_correction_v4.attempts import InfraInvalidReason
from experiments.online_correction_v4.clock import ActionQueue, ControlledSimulationClock, QuerySchedule
from experiments.online_correction_v4.contracts import EpisodeManifestRow, PolicyTimingAchieved, TimingConfig
from experiments.online_correction_v4.droid_contract import (
    DroidContractError,
    FixtureRuntimeBinding,
    LaunchArgs,
    NANO_POLICY_ID,
    PI05_POLICY_ID,
    PolicyRuntimeBinding,
    RuntimeLockBinding,
    build_launch_plan,
    sha256_bytes,
)
from experiments.online_correction_v4.droid_policy_request import PolicyInfraInvalidError
from experiments.online_correction_v4.droid_scorer import TerminalScorerError, build_terminal_scorer
from experiments.online_correction_v4.droid_simulator import (
    DroidDependencyError,
    DroidSimulatorAdapter,
    FakeRoboLabEnv,
    build_live_robolab_env,
)
from experiments.online_correction_v4.droid_robolab import RoboLabBootstrapError, RoboLabSession, write_queue_row
from experiments.online_correction_v4.droid_transport import EpisodePolicyTransport, TransportError, build_live_transport
from experiments.online_correction_v4.leases import AttemptFinalizer
from experiments.online_correction_v4.recorder import EpisodeEvidenceRecorder, digest_bytes
from experiments.online_correction_v4.runner import EpisodeRunConfig, EpisodeRunner
from experiments.online_correction_v4.motion import MotionDirectionError, ReferenceMotionController
from experiments.online_correction_v4.droid_task_files.registry import supported_fixture_ids


@dataclass(frozen=True)
class DroidEpisodeBinding:
    manifest: EpisodeManifestRow
    lock: RuntimeLockBinding
    policy_binding: PolicyRuntimeBinding
    fixture_binding: FixtureRuntimeBinding
    prompt_text: str
    prompt_sha256: str
    simulator: DroidSimulatorAdapter
    policy: PolicyAdapter
    clock: ControlledSimulationClock
    timing: TimingConfig
    terminal_scorer: TerminalScoringAdapter | None = None


class DroidEpisodeRunner(EpisodeRunner):
    """DROID-specific query-cycle handling for missing-future infra receipts."""

    def _validate_clock_schedule(self) -> None:
        expected = query_schedule_for_manifest(self.manifest_row)
        if self.clock.schedule != expected:
            raise RuntimeError(
                f"clock schedule {self.clock.schedule.value!r} does not match manifest "
                f"schedule {self.manifest_row.factors.get('schedule', 'standard')!r}"
            )
        if self.run_config.schedule != expected:
            raise RuntimeError(
                f"run_config schedule {self.run_config.schedule.value!r} does not match manifest "
                f"schedule {self.manifest_row.factors.get('schedule', 'standard')!r}"
            )

    def _run_query_cycle(self) -> None:
        self.clock.advance_to_next_query()
        if not self.clock.pending_requests:
            return
        request = self.clock.pending_requests[-1]
        captured = self.simulator.capture_observation()
        if not captured.payload:
            self._inference_error = InfraInvalidReason.MISSING_MANDATORY_STREAM
            self.recorder.record_event(
                {"kind": "infra_invalid", "request_id": request.request_id, "reason": "empty_observation_payload"}
            )
            return
        self._maybe_mark_event_observed(captured, request_id=request.request_id)
        payload = captured.payload
        payload_sha256 = captured.state_hash or sha256_bytes(payload)
        executed_action_count = self._executed_action_count()
        self.recorder.record_observation(
            observation_id=request.observation_id,
            capture_time_s=request.observation_capture_time,
            payload=payload,
            camera_ids=captured.camera_ids,
            state_hash=captured.state_hash,
            native_input_present=captured.native_input is not None,
        )
        self.recorder.record_request(
            {
                "request_id": request.request_id,
                "observation_id": request.observation_id,
                "observation_capture_time": request.observation_capture_time,
                "submit_time": request.submit_time,
                "observation_payload_sha256": payload_sha256,
                "observation_state_hash": captured.state_hash,
                "executed_action_count": executed_action_count,
            }
        )
        packet = ObservationPacket(
            observation_id=request.observation_id,
            capture_time_s=request.observation_capture_time,
            payload=payload,
            payload_sha256=payload_sha256,
            camera_ids=captured.camera_ids,
            state_hash=captured.state_hash,
            native_input=captured.native_input,
            executed_action_count=executed_action_count,
            request_metadata={
                "executed_action_count": executed_action_count,
                "queue_replacement_rule": self.clock.action_queue.replacement_rule.value,
            },
        )
        try:
            response = self.policy.infer(packet)
        except PolicyInfraInvalidError as exc:
            if exc.future_artifact is not None:
                self.recorder.record_future_artifact(
                    request_id=request.request_id,
                    kind=exc.future_artifact.kind,
                    payload=exc.future_artifact.payload,
                    payload_sha256=exc.future_artifact.payload_sha256,
                )
            self._inference_error = exc.reason
            self.recorder.record_event(
                {"kind": "infra_invalid", "request_id": request.request_id, "reason": str(exc)}
            )
            self.recorder.flush_incremental(fsync=True)
            return
        except PolicyModelInvalidError as exc:
            self._model_invalid = True
            self.recorder.record_event(
                {"kind": "model_output_invalid", "request_id": request.request_id, "reason": str(exc)}
            )
            self.recorder.flush_incremental(fsync=True)
            return
        except Exception as exc:  # noqa: BLE001
            self._inference_error = InfraInvalidReason.MALFORMED_ACTION_INTERFACE
            self.recorder.record_event(
                {"kind": "infra_invalid", "request_id": request.request_id, "reason": str(exc)}
            )
            self.recorder.flush_incremental(fsync=True)
            return
        if not all(all(math.isfinite(value) for value in action) for action in response.actions):
            self._model_invalid = True
            self.recorder.record_event(
                {
                    "kind": "model_output_invalid",
                    "request_id": request.request_id,
                    "reason": "nonfinite_action_values",
                }
            )
            self.recorder.flush_incremental(fsync=True)
            return
        self._record_policy_response(request.request_id, response)
        self.clock.finish_query_cycle(
            request,
            chunk_id=response.chunk_id,
            actions=response.actions,
            wall_duration_s=response.wall_duration_s,
            advance_reference=lambda t: self.reference.pose_at(t),
        )
        self.recorder.record_request(
            {
                "request_id": request.request_id,
                "response_available_time": request.response_available_time,
                "wall_duration_s": response.wall_duration_s,
                "chunk_id": response.chunk_id,
                "action_sha256": response.action_sha256 or digest_bytes(repr(response.actions).encode("utf-8")),
                "generated_horizon": response.generated_horizon,
                "executed_action_count": executed_action_count,
            }
        )
        self.recorder.flush_incremental(fsync=True)


def resolve_motion_direction(manifest: EpisodeManifestRow) -> tuple[float, float]:
    """Derive the live reference-motion unit vector from manifest counterbalance."""
    return ReferenceMotionController.resolve_live_direction(
        fixture=manifest.fixture,
        goal=manifest.factors["goal"],
        counterbalance=manifest.counterbalance,
        supported_fixtures=supported_fixture_ids(),
    )


def _executed_action_count(clock: ControlledSimulationClock) -> Callable[[], int]:
    return lambda: len(clock.action_queue.executed)


def resolve_prompt_text(manifest: EpisodeManifestRow) -> str:
    recipe = manifest.prompt_recipe
    template = recipe.get("template")
    if not isinstance(template, str) or not template.strip():
        raise DroidContractError("manifest prompt_recipe.template is required")
    return template


def build_fake_binding(
    *,
    manifest: EpisodeManifestRow,
    lock: RuntimeLockBinding,
    policy_binding: PolicyRuntimeBinding,
    fixture_binding: FixtureRuntimeBinding,
    prompt_text: str,
    prompt_sha256: str,
    reset_fingerprint_sha256: str,
    runtime_identity_sha256: str,
    timing: TimingConfig,
    schedule: QuerySchedule,
    terminal_scorer: TerminalScoringAdapter | None = None,
) -> DroidEpisodeBinding:
    native_dt = policy_binding.native_control_dt_s
    achieved = PolicyTimingAchieved(
        native_control_dt_s=native_dt,
        achieved_delay_s=policy_binding.achieved_delay_s,
        achieved_standard_query_period_s=policy_binding.achieved_standard_query_period_s,
        achieved_fast_query_period_s=policy_binding.achieved_fast_query_period_s,
        prediction_horizon_actions=policy_binding.prediction_horizon_actions,
    )
    clock = ControlledSimulationClock(
        timing=timing,
        achieved=achieved,
        schedule=schedule,
        action_queue=ActionQueue(native_control_dt_s=native_dt),
    )
    simulator = DroidSimulatorAdapter.from_fake(
        episode_id=manifest.episode_id,
        env_seed=manifest.env_seed,
        fixture=fixture_binding,
        env=FakeRoboLabEnv(native_dt=native_dt),
    )
    simulator.prompt_sha256 = prompt_sha256
    simulator.runtime_identity_sha256 = runtime_identity_sha256
    simulator.experiment_clock = clock
    executed = _executed_action_count(clock)
    policy_id = manifest.factors["policy"]
    reset_callback = _simulator_reset_fingerprint(simulator)
    if policy_id == NANO_POLICY_ID:
        from experiments.online_correction_v4.droid_nano_policy import (
            DroidNanoPolicyAdapter,
            fake_nano_transport,
        )

        policy: PolicyAdapter = DroidNanoPolicyAdapter(
            binding=policy_binding,
            episode_id=manifest.episode_id,
            policy_seed=manifest.policy_seed,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            reset_fingerprint_sha256=reset_fingerprint_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            transport=fake_nano_transport(manifest.policy_seed),
            ensure_reset_attestation=reset_callback,
            executed_action_count=executed,
        )
    elif policy_id == PI05_POLICY_ID:
        from experiments.online_correction_v4.droid_pi05_policy import (
            DroidPi05PolicyAdapter,
            fake_pi05_transport,
        )

        policy = DroidPi05PolicyAdapter(
            binding=policy_binding,
            episode_id=manifest.episode_id,
            policy_seed=manifest.policy_seed,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            reset_fingerprint_sha256=reset_fingerprint_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            transport=fake_pi05_transport(manifest.policy_seed),
            ensure_reset_attestation=reset_callback,
            executed_action_count=executed,
        )
    else:
        raise DroidContractError(f"unsupported policy for fake binding: {policy_id}")
    return DroidEpisodeBinding(
        manifest=manifest,
        lock=lock,
        policy_binding=policy_binding,
        fixture_binding=fixture_binding,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        simulator=simulator,
        policy=policy,
        clock=clock,
        timing=timing,
        terminal_scorer=terminal_scorer,
    )


def build_live_binding(
    *,
    manifest: EpisodeManifestRow,
    lock: RuntimeLockBinding,
    policy_binding: PolicyRuntimeBinding,
    fixture_binding: FixtureRuntimeBinding,
    prompt_text: str,
    prompt_sha256: str,
    runtime_identity_sha256: str,
    timing: TimingConfig,
    schedule: QuerySchedule,
    policy_host: str,
    policy_port: int,
    output_dir: Path | None = None,
    terminal_scorer: TerminalScoringAdapter | None = None,
) -> DroidEpisodeBinding:
    native_dt = policy_binding.native_control_dt_s
    achieved = PolicyTimingAchieved(
        native_control_dt_s=native_dt,
        achieved_delay_s=policy_binding.achieved_delay_s,
        achieved_standard_query_period_s=policy_binding.achieved_standard_query_period_s,
        achieved_fast_query_period_s=policy_binding.achieved_fast_query_period_s,
        prediction_horizon_actions=policy_binding.prediction_horizon_actions,
    )
    clock = ControlledSimulationClock(
        timing=timing,
        achieved=achieved,
        schedule=schedule,
        action_queue=ActionQueue(native_control_dt_s=native_dt),
    )
    queue_parent = output_dir or Path("/tmp")
    queue_row_path, queue_row_sha256 = write_queue_row(
        output_dir=queue_parent,
        episode_id=manifest.episode_id,
        fixture_id=manifest.fixture,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        env_seed=manifest.env_seed,
        goal=str(manifest.factors.get("goal", "left")),
    )
    try:
        live_env = build_live_robolab_env(
            fixture=fixture_binding,
            env_seed=manifest.env_seed,
            episode_id=manifest.episode_id,
            goal=str(manifest.factors.get("goal", "left")),
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            policy_id=manifest.factors["policy"],
            queue_row_path=queue_row_path,
            queue_row_sha256=queue_row_sha256,
            output_dir=output_dir,
            locked_native_control_dt_s=native_dt,
        )
    except (DroidDependencyError, RoboLabBootstrapError) as exc:
        raise DroidContractError(str(exc)) from exc
    simulator = DroidSimulatorAdapter.from_live_env(
        episode_id=manifest.episode_id,
        env_seed=manifest.env_seed,
        fixture=fixture_binding,
        env=live_env,
        reset_proxy=live_env.reset_proxy,
    )
    simulator.prompt_sha256 = prompt_sha256
    simulator.runtime_identity_sha256 = runtime_identity_sha256
    simulator.experiment_clock = clock
    executed = _executed_action_count(clock)
    reset_callback = _simulator_reset_fingerprint(simulator)
    policy_id = manifest.factors["policy"]
    try:
        transport = build_live_transport(
            policy_id=policy_id,
            host=policy_host,
            port=policy_port,
        )
    except TransportError as exc:
        raise DroidContractError(str(exc)) from exc
    if policy_id == NANO_POLICY_ID:
        from experiments.online_correction_v4.droid_nano_policy import DroidNanoPolicyAdapter

        policy: PolicyAdapter = DroidNanoPolicyAdapter(
            binding=policy_binding,
            episode_id=manifest.episode_id,
            policy_seed=manifest.policy_seed,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            reset_fingerprint_sha256="",
            runtime_identity_sha256=runtime_identity_sha256,
            transport=transport,
            ensure_reset_attestation=reset_callback,
            executed_action_count=executed,
        )
    else:
        from experiments.online_correction_v4.droid_pi05_policy import DroidPi05PolicyAdapter

        policy = DroidPi05PolicyAdapter(
            binding=policy_binding,
            episode_id=manifest.episode_id,
            policy_seed=manifest.policy_seed,
            prompt_text=prompt_text,
            prompt_sha256=prompt_sha256,
            reset_fingerprint_sha256="",
            runtime_identity_sha256=runtime_identity_sha256,
            transport=transport,
            ensure_reset_attestation=reset_callback,
            executed_action_count=executed,
        )
    scorer = terminal_scorer
    if scorer is None:
        try:
            scorer = build_terminal_scorer(
                manifest=manifest,
                fixture_binding=fixture_binding,
                timing=timing,
            )
        except (DroidContractError, TerminalScorerError) as exc:
            raise DroidContractError(str(exc)) from exc
    return DroidEpisodeBinding(
        manifest=manifest,
        lock=lock,
        policy_binding=policy_binding,
        fixture_binding=fixture_binding,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        simulator=simulator,
        policy=policy,
        clock=clock,
        timing=timing,
        terminal_scorer=scorer,
    )


def build_episode_runner(
    binding: DroidEpisodeBinding,
    *,
    output_dir: Path,
    attempt_id: str,
    displacement_m: float,
    motion_direction: tuple[float, float],
    scenario: str,
    motion_config: dict[str, Any],
) -> tuple[EpisodeRunner, AttemptFinalizer]:
    finalizer = AttemptFinalizer(output_dir)
    recorder = EpisodeEvidenceRecorder.open(
        finalizer=finalizer,
        episode_id=binding.manifest.episode_id,
        attempt_id=attempt_id,
        metadata={
            "episode_id": binding.manifest.episode_id,
            "attempt_id": attempt_id,
            "prefix_mode": binding.lock.prefix_mode.value,
            "policy_id": binding.manifest.factors["policy"],
        },
    )
    viewport_writer = _build_viewport_writer(binding)
    if hasattr(viewport_writer, "bind_attempt_path"):
        viewport_writer.bind_attempt_path(recorder.attempt_path)
    runner = DroidEpisodeRunner(
        manifest_row=binding.manifest,
        timing=binding.timing,
        clock=binding.clock,
        simulator=binding.simulator,
        policy=binding.policy,
        recorder=recorder,
        terminal_scorer=binding.terminal_scorer,
        viewport_writer=viewport_writer,
        run_config=EpisodeRunConfig(
            displacement_m=displacement_m,
            motion_direction=motion_direction,
            prompt_text=binding.prompt_text,
            scenario=scenario,
            motion_config=motion_config,
            schedule=binding.clock.schedule,
            viewport_video_required=binding.lock.writer_contract.viewport_video_required,
        ),
    )
    return runner, finalizer


def _build_viewport_writer(binding: DroidEpisodeBinding):
    from experiments.online_correction_v4.droid_robolab import LiveRoboLabEnv
    from experiments.online_correction_v4.testing import FakeViewportVideoWriter

    native_dt = binding.policy_binding.native_control_dt_s
    fps = round(1.0 / native_dt, 6)
    if isinstance(binding.simulator.env, LiveRoboLabEnv):
        from experiments.online_correction_v4.viewport_video import build_live_viewport_writer

        return build_live_viewport_writer(fps=fps)
    return FakeViewportVideoWriter(fps=fps)


def _simulator_reset_fingerprint(simulator: DroidSimulatorAdapter) -> Callable[[], str]:
    def _callback() -> str:
        if simulator.reset_proxy is None or not simulator.reset_proxy.state.attestation_written:
            raise DroidContractError("reset attestation must exist before policy request zero")
        fingerprint = simulator.reset_proxy.state.attestation_body.get("reset_fingerprint_sha256")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise DroidContractError("reset attestation fingerprint is missing")
        return fingerprint

    return _callback


def query_schedule_for_manifest(manifest: EpisodeManifestRow) -> QuerySchedule:
    schedule_name = manifest.factors.get("schedule", "standard")
    if schedule_name == "fast_after_grasp":
        return QuerySchedule.FAST_AFTER_GRASP
    if schedule_name == "fast":
        return QuerySchedule.FAST_AFTER_GRASP
    return QuerySchedule.STANDARD


def validate_only_plan(args: LaunchArgs, *, study_root: Path, campaign_config_path: Path) -> dict[str, Any]:
    return build_launch_plan(args, study_root=study_root, campaign_config_path=campaign_config_path)


def attestation_from_fake_reset(
    *,
    simulator: DroidSimulatorAdapter,
    prompt_sha256: str,
    runtime_identity_sha256: str,
) -> str:
    attestation = simulator.finalize_reset_attestation(
        prompt_sha256=prompt_sha256,
        runtime_identity_sha256=runtime_identity_sha256,
        initial_state_sha256=sha256_bytes(simulator.capture_observation_packet(
            observation_id="reset-obs",
            capture_time_s=0.0,
        ).payload),
    )
    from experiments.online_correction_v4.droid_reset import validate_reset_attestation_payload

    validate_reset_attestation_payload(attestation, episode_id=simulator.episode_id)
    return attestation["reset_fingerprint_sha256"]
