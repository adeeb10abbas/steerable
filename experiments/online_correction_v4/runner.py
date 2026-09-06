"""Audited episode runner tick loop wiring clock, motion, detectors, and evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Optional

from experiments.online_correction_v4.adapters import (
    ObservationPacket,
    PolicyAdapter,
    PolicyModelInvalidError,
    PolicyResponse,
    SimulatorAdapter,
    TerminalPhysicalPredicates,
    TerminalScoringAdapter,
    TerminalScoringEvidence,
    ViewportFrame,
    ViewportVideoRequiredError,
    ViewportVideoWriter,
)
from experiments.online_correction_v4.attempts import (
    AttemptClassifier,
    InfraInvalidReason,
    TerminalEvidenceFlags,
    classify_terminal_outcome,
)
from experiments.online_correction_v4.clock import ActionCommand, ControlledSimulationClock, QuerySchedule
from experiments.online_correction_v4.contracts import EpisodeManifestRow, EpisodeRuntimeFlags, TimingConfig
from experiments.online_correction_v4.detectors import (
    DetachmentDetector,
    DetachmentDetectorConfig,
    GraspDetectorConfig,
    NaturalGraspDetector,
)
from experiments.online_correction_v4.motion import ReferenceMotionController
from experiments.online_correction_v4.observation_audit import (
    evaluate_changed_observation_visibility,
    parse_observation_audit,
)
from experiments.online_correction_v4.recorder import EpisodeEvidenceRecorder, digest_bytes
from experiments.online_correction_v4.viewport_video import (
    ViewportCapture,
    attest_viewport_capture,
    viewport_frame_from_capture,
)


class EpisodeEndReason(str, Enum):
    RELEASE_CONFIRMED = "release_confirmed"
    TIMEOUT = "timeout"
    INFRA_INVALID = "infra_invalid"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    INCOMPLETE = "incomplete"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _actions_are_finite(actions: tuple[tuple[float, ...], ...]) -> bool:
    for action in actions:
        for value in action:
            if not math.isfinite(value):
                return False
    return True


def expected_query_schedule(manifest_row: EpisodeManifestRow) -> QuerySchedule:
    schedule_name = manifest_row.factors.get("schedule", "standard")
    if schedule_name == "fast":
        return QuerySchedule.FAST_AFTER_GRASP
    return QuerySchedule.STANDARD


@dataclass
class EpisodeRunConfig:
    displacement_m: float
    motion_direction: tuple[float, float]
    prompt_text: str
    scenario: str
    motion_config: dict[str, Any]
    schedule: QuerySchedule = QuerySchedule.STANDARD
    viewport_video_required: bool = True
    trajectory_flush_interval: int = 50


@dataclass
class EpisodeRunResult:
    end_reason: EpisodeEndReason
    flags: EpisodeRuntimeFlags
    terminal: TerminalEvidenceFlags
    timing: dict[str, Any]
    attempt_status: str
    failure_label: str
    failure_stage: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeRunner:
    manifest_row: EpisodeManifestRow
    timing: TimingConfig
    clock: ControlledSimulationClock
    simulator: SimulatorAdapter
    policy: PolicyAdapter
    recorder: EpisodeEvidenceRecorder
    run_config: EpisodeRunConfig
    terminal_scorer: TerminalScoringAdapter | None = None
    viewport_writer: ViewportVideoWriter | None = None
    classifier: AttemptClassifier = field(default_factory=AttemptClassifier)
    reference: ReferenceMotionController = field(init=False)
    grasp: NaturalGraspDetector = field(init=False)
    detach: DetachmentDetector = field(init=False)
    flags: EpisodeRuntimeFlags = field(default_factory=EpisodeRuntimeFlags)
    _last_safe_action: tuple[float, ...] = ()
    _latest_snapshot: Any = None
    _inference_error: Optional[InfraInvalidReason] = None
    _model_invalid: bool = False
    _passive_settling_reason: str | None = None
    _t_motion_actual_onset: float | None = None
    _viewport_frame_index: int = 0
    _viewport_failure: str | None = None
    _resources_closed: bool = False
    _settling_predicate_samples: list[TerminalPhysicalPredicates] = field(default_factory=list)
    _missing_visibility_recorded: bool = False

    def __post_init__(self) -> None:
        self._validate_clock_schedule()
        self.reference = ReferenceMotionController.from_scenario(
            self.run_config.scenario,
            displacement_m=self.run_config.displacement_m,
            motion_config=self.run_config.motion_config,
        )
        native_dt = self.clock.achieved.native_control_dt_s
        self.grasp = NaturalGraspDetector(
            config=GraspDetectorConfig(
                min_lift_m=self.timing.natural_grasp_min_lift_m,
                dwell_s=self.timing.natural_grasp_dwell_s,
                relative_drift_max_m=self.timing.kinematic_grasp_relative_drift_max_m,
                trigger_deadline_s=self.timing.trigger_deadline_s,
            ),
            control_dt_s=native_dt,
        )
        self.detach = DetachmentDetector(
            config=DetachmentDetectorConfig(dwell_ticks=self.timing.release_detection_dwell_ticks)
        )
        self.clock.event_phase_fraction = float(
            self.manifest_row.counterbalance.get("event_phase_fraction", 0.0)
        )
        self.clock.register_tick_callback(self._on_control_tick)

    def _validate_clock_schedule(self) -> None:
        expected = expected_query_schedule(self.manifest_row)
        if self.clock.schedule != expected:
            raise RuntimeError(
                f"clock schedule {self.clock.schedule.value!r} does not match manifest "
                f"schedule {self.manifest_row.factors.get('schedule', 'standard')!r}"
            )
        if self.run_config.schedule != expected:
            raise RuntimeError(
                f"run_config schedule {self.run_config.schedule.value!r} does not match manifest"
            )

    def run(self) -> EpisodeRunResult:
        result: EpisodeRunResult | None = None
        try:
            result = self._run_episode()
            return result
        except Exception as exc:
            self.recorder.set_episode_fields(
                runtime_exception={
                    "type": type(exc).__name__,
                    "detail": str(exc),
                }
            )
            self._flush_partial(reason="exception")
            raise
        finally:
            self._close_resources()

    def _run_episode(self) -> EpisodeRunResult:
        from experiments.online_correction_v4.droid_robolab import LiveRoboLabEnv, RoboLabSession

        env = getattr(self.simulator, "env", None)
        if isinstance(env, LiveRoboLabEnv):
            RoboLabSession.begin_episode(self.manifest_row.episode_id)
        self._latest_snapshot = self.simulator.reset(env_seed=self.manifest_row.env_seed)
        self._apply_registered_reference_pose(sim_time=0.0)
        if self._latest_snapshot is not None:
            initial = self.reference.pose_at(0.0)
            self._latest_snapshot.reference_displacement_m = initial.displacement_m
        self._capture_viewport_frame(snapshot=self._latest_snapshot)
        self.policy.reset(policy_seed=self.manifest_row.policy_seed, prompt_text=self.run_config.prompt_text)
        self.recorder.set_episode_fields(
            episode_id=self.manifest_row.episode_id,
            attempt_id=self.recorder.attempt_id,
            prefix_group_id=self.manifest_row.prefix_group_id,
            env_seed=self.manifest_row.env_seed,
            policy_seed=self.manifest_row.policy_seed,
            scenario=self.run_config.scenario,
            query_schedule=self.clock.schedule.value,
        )
        self._record_trajectory(action=None)
        self.recorder.flush_incremental(fsync=True)

        max_time = self.timing.episode_cap_s + self.timing.release_settling_s + 1.0
        while self.clock.episode_end_time is None and self.clock.sim_time <= max_time:
            if self._model_invalid:
                return self._finish_model_invalid()
            if self._inference_error is not None:
                return self._finish_infra(self._inference_error)
            if self._viewport_failure is not None:
                return self._finish_infra(
                    InfraInvalidReason.MISSING_MANDATORY_STREAM,
                    detail=self._viewport_failure,
                )
            if self._passive_settling_reason is None and self.clock.passive_settling_active:
                self._passive_settling_reason = "timeout"
            if self.clock.policy_phase_active and self.clock.due_for_query():
                self._run_query_cycle()
                continue
            self.clock._advance_one_control_tick()

        return self._finalize_episode()

    def _apply_registered_reference_pose(self, *, sim_time: float) -> None:
        state = self.reference.pose_at(sim_time)
        self.simulator.set_reference_offset(state.displacement_m, self.run_config.motion_direction)

    def _executed_action_count(self) -> int:
        return len(self.clock.action_queue.executed)

    def _resolve_viewport_capture(self, raw: Any) -> ViewportCapture | None:
        if raw is None:
            return None
        if isinstance(raw, ViewportCapture):
            return raw
        if isinstance(raw, (bytes, bytearray)):
            if not raw:
                return None
            return attest_viewport_capture(bytes(raw))
        raise ViewportVideoRequiredError(
            f"simulator viewport capture returned unattested type: {type(raw).__name__}"
        )

    def _capture_viewport_frame(self, *, snapshot: Any) -> None:
        if snapshot is None:
            return
        try:
            capture = self._resolve_viewport_capture(self.simulator.capture_viewport_frame())
        except ViewportVideoRequiredError as exc:
            detail = str(exc)
            self._viewport_failure = detail
            self.recorder.record_event(
                {"kind": "infra_invalid", "reason": "invalid_viewport_frame", "detail": detail}
            )
            return
        if capture is None:
            if self.run_config.viewport_video_required:
                detail = (
                    f"missing viewport frame at sim_time={snapshot.sim_time} "
                    f"control_tick={snapshot.control_tick}"
                )
                self._viewport_failure = detail
                self.recorder.record_event(
                    {"kind": "infra_invalid", "reason": "missing_viewport_frame", "detail": detail}
                )
            return
        fps = self.viewport_writer.fps if self.viewport_writer is not None else round(
            1.0 / self.clock.achieved.native_control_dt_s, 6
        )
        frame = viewport_frame_from_capture(
            frame_index=self._viewport_frame_index,
            sim_time_s=snapshot.sim_time,
            control_tick=snapshot.control_tick,
            capture=capture,
        )
        self._viewport_frame_index += 1
        if self.viewport_writer is not None:
            self.recorder.record_viewport_frame_index(
                frame_index=frame.frame_index,
                sim_time_s=frame.sim_time_s,
                control_tick=frame.control_tick,
                fps=fps,
                payload_sha256=frame.payload_sha256,
                format_kind=frame.format_kind,
                width=frame.width,
                height=frame.height,
                channels=frame.channels,
            )
        else:
            self.recorder.record_viewport_frame(
                frame_index=frame.frame_index,
                sim_time_s=frame.sim_time_s,
                control_tick=frame.control_tick,
                fps=fps,
                payload=frame.payload,
                payload_sha256=frame.payload_sha256,
                format_kind=frame.format_kind,
                width=frame.width,
                height=frame.height,
                channels=frame.channels,
            )
        if self.viewport_writer is not None:
            try:
                self.viewport_writer.append_frame(frame)
            except ViewportVideoRequiredError as exc:
                detail = str(exc)
                self._viewport_failure = detail
                self.recorder.record_event(
                    {"kind": "infra_invalid", "reason": "invalid_viewport_frame", "detail": detail}
                )

    def _on_control_tick(
        self,
        sim_time: float,
        control_tick: int,
        command: ActionCommand | None,
    ) -> None:
        self._update_reference_motion(sim_time)
        action_values: tuple[float, ...] | None = None
        if command is not None:
            action_values = command.values
            self._last_safe_action = action_values
        elif self.clock.passive_settling_active and self._last_safe_action:
            action_values = self._last_safe_action
            self.simulator.hold_robot_target(action_values)

        snapshot = self.simulator.step_control(action_values)
        self._latest_snapshot = snapshot
        self._capture_viewport_frame(snapshot=snapshot)
        self._post_physics(snapshot)
        if self.clock.passive_settling_active:
            self._record_settling_predicates(snapshot)
        self._record_trajectory(action=action_values)
        if self.recorder.should_flush_trajectory(interval=self.run_config.trajectory_flush_interval):
            self.recorder.flush_incremental(fsync=True)

    def _record_settling_predicates(self, snapshot: Any) -> None:
        sample = getattr(self.simulator, "sample_terminal_predicates", None)
        if callable(sample):
            self._settling_predicate_samples.append(sample())
            return
        if snapshot.terminal_predicates is not None:
            self._settling_predicate_samples.append(snapshot.terminal_predicates)

    def _evaluate_event_visibility(self, captured: Any) -> tuple[bool, str | None, dict[str, Any]]:
        if not self.flags.event_delivered:
            return False, "event_not_delivered", {}
        threshold = self.timing.changed_observation_reference_displacement_m
        if self._latest_snapshot is not None:
            if self._latest_snapshot.reference_displacement_m + 1e-12 < threshold:
                return False, "insufficient_reference_displacement", {}
        audit = parse_observation_audit(getattr(captured, "payload", b""))
        if audit is None:
            return False, "invalid_observation_audit", {}
        policy_camera_ids = getattr(captured, "camera_ids", ()) or audit.camera_ids
        result = evaluate_changed_observation_visibility(
            audit,
            displacement_threshold_m=threshold,
            policy_camera_ids=policy_camera_ids,
        )
        metadata = {
            "reference_displacement_m": audit.reference_displacement_m,
            "policy_camera_ids": list(policy_camera_ids),
            "moved_object_mask_pixels_by_camera": dict(audit.moved_object_mask_pixels_by_camera),
        }
        if result.qualifying_camera_id is not None:
            metadata["qualifying_camera_id"] = result.qualifying_camera_id
            metadata["qualifying_mask_pixels"] = result.qualifying_mask_pixels
        return result.qualified, result.reason, metadata

    def _maybe_mark_event_observed(self, captured: Any, *, request_id: str) -> None:
        if self.flags.event_observed:
            return
        if not self.flags.event_delivered:
            return
        qualified, reason, metadata = self._evaluate_event_visibility(captured)
        if qualified:
            self.flags.event_observed = True
            self.recorder.record_event(
                {
                    "event_index": 4,
                    "kind": "event_observed",
                    "request_id": request_id,
                    "sim_time": self.clock.sim_time,
                    **metadata,
                }
            )
            return
        if reason in {"missing_visibility_evidence", "no_nonempty_mask_in_policy_cameras"} and (
            not self._missing_visibility_recorded
        ):
            self._missing_visibility_recorded = True
            self.recorder.record_event(
                {
                    "event_index": 4,
                    "kind": "event_not_observed",
                    "reason": reason,
                    "request_id": request_id,
                    "sim_time": self.clock.sim_time,
                    **metadata,
                }
            )

    def _update_reference_motion(self, sim_time: float) -> None:
        if self.reference.frozen_at is not None:
            return
        state = self.reference.pose_at(sim_time)
        prior = self._latest_snapshot.reference_displacement_m if self._latest_snapshot is not None else 0.0
        self.simulator.set_reference_offset(state.displacement_m, self.run_config.motion_direction)
        self._maybe_mark_event_delivered(sim_time=sim_time, displacement_m=state.displacement_m, prior_m=prior)

    def _maybe_mark_event_delivered(self, *, sim_time: float, displacement_m: float, prior_m: float) -> None:
        if self.flags.event_delivered:
            return
        if self.run_config.scenario in {"original_sham", "destination_static"}:
            return
        if self.reference.event_onset_s is None or sim_time + 1e-12 < self.reference.event_onset_s:
            return
        if displacement_m <= 1e-9 and prior_m <= 1e-9:
            return
        self.flags.event_delivered = True
        self._t_motion_actual_onset = sim_time
        self.recorder.record_event(
            {
                "event_index": 3,
                "kind": "event_delivered",
                "sim_time": sim_time,
                "reference_displacement_m": displacement_m,
            }
        )

    def _post_physics(self, snapshot: Any) -> None:
        grasp_event = self.grasp.update(snapshot.object_state)
        if grasp_event is not None and not self.flags.trigger_eligible:
            self.flags.trigger_eligible = True
            self.detach.arm_after_verified_carry()
            self.clock.register_natural_grasp(grasp_event.t_eligible)
            onset = self.clock.plan_event_onset_after_grasp()
            self.reference.schedule_event(onset)
            self.recorder.record_event(
                {"event_index": 0, "kind": "trigger_eligible", "sim_time": grasp_event.t_eligible}
            )
            self.recorder.record_event({"event_index": 1, "kind": "event_planned", "sim_time": onset})

        if self.detach.armed:
            detach_event = self.detach.update(snapshot.object_state)
            if detach_event is not None and not self.flags.passive_settling_started:
                self.reference.freeze_at(detach_event.t_detected, reason="release_detected")
                self.flags.motion_truncated_by_release = True
                self.clock.start_passive_settling(detach_event.t_detected)
                self.flags.passive_settling_started = True
                self._passive_settling_reason = "release"
                anchor = getattr(self.simulator, "anchor_passive_settling_baseline", None)
                if callable(anchor):
                    anchor(snapshot)
                self.recorder.record_event(
                    {
                        "event_index": 2,
                        "kind": "release_detected",
                        "sim_time": detach_event.t_detected,
                        "t_onset": detach_event.t_onset,
                    }
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
        payload_sha256 = captured.state_hash or _sha256_bytes(payload)
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
        if not _actions_are_finite(response.actions):
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

    def _record_policy_response(self, request_id: str, response: PolicyResponse) -> None:
        if response.future_artifact is None:
            return
        artifact = response.future_artifact
        self.recorder.record_future_artifact(
            request_id=request_id,
            kind=artifact.kind,
            payload=artifact.payload,
            payload_sha256=artifact.payload_sha256,
        )

    def _record_trajectory(self, *, action: tuple[float, ...] | None) -> None:
        if self._latest_snapshot is None:
            return
        snapshot = self._latest_snapshot
        self.recorder.record_trajectory_row(
            {
                "simulation_time": float(snapshot.sim_time),
                "control_step": int(snapshot.control_tick),
                "reference_displacement_m": float(
                    snapshot.reference_displacement_m
                ),
                "commanded_action": list(action) if action is not None else None,
                "grasp_eligible": bool(self.grasp.eligible),
                "detach_armed": bool(self.detach.armed),
            }
        )

    def _finalize_viewport_video(self) -> None:
        if self.viewport_writer is None:
            if self.run_config.viewport_video_required:
                raise ViewportVideoRequiredError("viewport writer is required but not configured")
            return
        artifact = self.viewport_writer.finalize_video(attempt_path=self.recorder.attempt_path)
        self.recorder.record_viewport_video(artifact)

    def _terminal_evidence(self) -> TerminalScoringEvidence | None:
        if self.terminal_scorer is None or self._latest_snapshot is None:
            return None
        return self.terminal_scorer.score_terminal(
            snapshot=self._latest_snapshot,
            runtime_flags=self.flags,
            passive_settling_reason=self._passive_settling_reason,
            grasp_occurred=self.grasp.grasp_occurred,
            carry_verified=self.grasp.carry_verified,
            settling_predicates=tuple(self._settling_predicate_samples),
        )

    def _terminal_flags(self) -> TerminalEvidenceFlags:
        scorer = self._terminal_evidence()
        released = self.detach.detected
        timeout_passive = self._passive_settling_reason == "timeout"
        timeout = timeout_passive and not released
        if scorer is not None:
            return TerminalEvidenceFlags(
                success=scorer.success,
                grasp_occurred=scorer.grasp_occurred,
                carry_verified=scorer.carry_verified,
                grasp_lost=scorer.grasp_lost,
                released=scorer.released,
                trigger_eligible=self.flags.trigger_eligible,
                event_delivered=self.flags.event_delivered,
                transport_incomplete=scorer.transport_incomplete,
                geometric_relation_correct=scorer.geometric_relation_correct,
                allowed_support=scorer.allowed_support,
                allowed_containment=scorer.allowed_containment,
                stable_for_dwell=scorer.stable_for_dwell,
                boundary_violation=scorer.boundary_violation,
                collision_terminal_failure=scorer.collision_terminal_failure,
                model_output_invalid=scorer.model_output_invalid,
                unresolved_behavioral_failure=scorer.unresolved_behavioral_failure,
                timeout_without_completion=timeout and self.flags.trigger_eligible,
                timeout_after_no_grasp=timeout and not self.flags.trigger_eligible,
            )
        return TerminalEvidenceFlags(
            success=False,
            grasp_occurred=self.grasp.grasp_occurred,
            carry_verified=self.grasp.carry_verified,
            released=released,
            trigger_eligible=self.flags.trigger_eligible,
            event_delivered=self.flags.event_delivered,
            transport_incomplete=self.flags.trigger_eligible and not released and timeout,
            timeout_without_completion=timeout and self.flags.trigger_eligible,
            timeout_after_no_grasp=timeout and not self.flags.trigger_eligible,
        )

    def _resolve_end_reason(self) -> EpisodeEndReason:
        if self.detach.detected:
            return EpisodeEndReason.RELEASE_CONFIRMED
        if self._passive_settling_reason == "timeout":
            return EpisodeEndReason.TIMEOUT
        if self.clock.episode_end_time is not None:
            return EpisodeEndReason.TIMEOUT
        return EpisodeEndReason.INCOMPLETE

    def _build_timing_record(self) -> dict[str, Any]:
        return {
            "t_episode_end": self.clock.episode_end_time,
            "t_trigger_eligible": self.grasp.event.t_eligible if self.grasp.event else None,
            "t_event_planned": self.clock.event_onset_time,
            "t_motion_actual_onset": self._t_motion_actual_onset,
            "t_release_detected": self.detach.event.t_detected if self.detach.event else None,
            "t_settling_start": self.detach.event.t_detected if self.detach.event else None,
            "passive_settling_reason": self._passive_settling_reason,
            "query_times": [row.get("observation_capture_time") for row in self.recorder.requests],
            "viewport_frame_count": self._viewport_frame_index,
        }

    def _flush_partial(self, *, reason: str) -> None:
        if self.recorder._finalized:
            return
        try:
            self.recorder.flush_partial(reason=reason)
        except Exception:
            return

    def _close_resources(self) -> None:
        if self._resources_closed:
            return
        self._resources_closed = True
        for resource in (self.viewport_writer, self.simulator, self.policy):
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        from experiments.online_correction_v4.droid_robolab import close_live_droid_stack

        close_live_droid_stack(policy=self.policy)

    def _finalize_episode(self) -> EpisodeRunResult:
        try:
            self._finalize_viewport_video()
        except ViewportVideoRequiredError as exc:
            return self._finish_infra(InfraInvalidReason.MISSING_MANDATORY_STREAM, detail=str(exc))
        terminal = self._terminal_flags()
        status, label, stage, meta = classify_terminal_outcome(terminal)
        end_reason = self._resolve_end_reason()
        timing = self._build_timing_record()
        receipt = {
            "status": status.value,
            "failure_label": label.value,
            "failure_stage": stage.value,
            "end_reason": end_reason.value,
            "passive_settling_reason": self._passive_settling_reason,
            **meta,
        }
        scorer = self._terminal_evidence()
        if scorer is not None and scorer.metadata:
            receipt["terminal_scoring"] = dict(scorer.metadata)
        self.recorder.record_timing(timing)
        self.recorder.set_episode_fields(
            trigger_eligible=self.flags.trigger_eligible,
            event_delivered=self.flags.event_delivered,
            event_observed=self.flags.event_observed,
            motion_truncated_by_release=self.flags.motion_truncated_by_release,
            **receipt,
        )
        self.recorder.finalize(terminal_receipt=receipt)
        return EpisodeRunResult(
            end_reason=end_reason,
            flags=self.flags,
            terminal=terminal,
            timing=timing,
            attempt_status=status.value,
            failure_label=label.value,
            failure_stage=stage.value,
            metadata=meta,
        )

    def _finish_model_invalid(self) -> EpisodeRunResult:
        try:
            self._finalize_viewport_video()
        except ViewportVideoRequiredError as exc:
            return self._finish_infra(InfraInvalidReason.MISSING_MANDATORY_STREAM, detail=str(exc))
        terminal = TerminalEvidenceFlags(
            model_output_invalid=True,
            trigger_eligible=self.flags.trigger_eligible,
            event_delivered=self.flags.event_delivered,
        )
        status, label, stage, meta = classify_terminal_outcome(terminal)
        timing = self._build_timing_record()
        receipt = {
            "status": status.value,
            "failure_label": label.value,
            "failure_stage": stage.value,
            "end_reason": EpisodeEndReason.MODEL_OUTPUT_INVALID.value,
            "model_output_invalid": True,
            **meta,
        }
        self.recorder.record_timing(timing)
        self.recorder.set_episode_fields(**receipt)
        self.recorder.finalize(terminal_receipt=receipt)
        return EpisodeRunResult(
            end_reason=EpisodeEndReason.MODEL_OUTPUT_INVALID,
            flags=self.flags,
            terminal=terminal,
            timing=timing,
            attempt_status=status.value,
            failure_label=label.value,
            failure_stage=stage.value,
            metadata=meta,
        )

    def _finish_infra(self, reason: InfraInvalidReason, *, detail: str = "") -> EpisodeRunResult:
        self._flush_partial(reason="infra_invalid")
        receipt = {
            "status": "infra_invalid",
            "infra_invalid_reason": reason.value,
            "end_reason": EpisodeEndReason.INFRA_INVALID.value,
        }
        if detail:
            receipt["detail"] = detail
        self.recorder.set_episode_fields(**receipt)
        self.recorder.finalize(terminal_receipt=receipt)
        return EpisodeRunResult(
            end_reason=EpisodeEndReason.INFRA_INVALID,
            flags=self.flags,
            terminal=self._terminal_flags(),
            timing={"t_episode_end": self.clock.sim_time},
            attempt_status="infra_invalid",
            failure_label="unresolved_behavioral_failure",
            failure_stage="other",
            metadata=receipt,
        )
