"""In-memory test doubles implementing V4 adapter protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Optional

from experiments.online_correction_v4.adapters import (
    CapturedObservation,
    EncodedVideoArtifact,
    FutureArtifact,
    ObservationPacket,
    PolicyModelInvalidError,
    PolicyResponse,
    SimulatorSnapshot,
    TerminalPhysicalPredicates,
    TerminalScoringAdapter,
    TerminalScoringEvidence,
    ViewportFrame,
    ViewportVideoRequiredError,
    ViewportVideoWriter,
)
from experiments.online_correction_v4.contracts import EpisodeRuntimeFlags
from experiments.online_correction_v4.detectors import ObjectKinematicState
from experiments.online_correction_v4.observation_audit import build_observation_audit_payload
from experiments.online_correction_v4.recorder import digest_bytes
from experiments.online_correction_v4.terminal_stability import (
    evaluate_horizontal_terminal_sample,
    geodesic_orientation_delta_rad,
    position_drift_m,
)


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _test_viewport_capture(*, tick: int, z: float) -> Any:
    """Deterministic non-blank viewport for scripted simulators."""
    from experiments.online_correction_v4.viewport_video import attest_viewport_capture

    try:
        import cv2
        import numpy as np
    except ImportError:
        width, height = 4, 4
        payload = bytearray([0] * (width * height * 3))
        for index in range(width * height):
            base = index * 3
            payload[base] = min(255, 40 + tick + index)
            payload[base + 1] = min(255, int(z * 100) % 256)
            payload[base + 2] = 1
        return attest_viewport_capture(
            bytes(payload),
            format_kind="raw_rgb24",
            width=width,
            height=height,
            channels=3,
        )

    from experiments.online_correction_v4.viewport_video import capture_from_ndarray

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :, 0] = min(255, 40 + tick)
    image[0, 0, 2] = min(255, int(z * 100) % 256 or 1)
    try:
        ok, encoded = cv2.imencode(".png", image)
    except Exception:
        ok = False
    if ok:
        return attest_viewport_capture(encoded.tobytes(), format_kind="encoded_png")
    return capture_from_ndarray(image, format_kind="raw_rgb24")


@dataclass
class FakeViewportVideoWriter:
    """Deterministic fake encoder producing a content-addressed pseudo-video artifact."""

    fps: float = 20.0
    codec: str = "fake-v4-test-encoder-v1"
    _frames: list[ViewportFrame] = field(default_factory=list)
    _closed: bool = False

    def append_frame(self, frame: ViewportFrame) -> None:
        if self._closed:
            raise RuntimeError("viewport writer already finalized")
        self._frames.append(frame)

    def finalize_video(self, *, attempt_path: Path) -> EncodedVideoArtifact:
        if self._closed:
            raise ViewportVideoRequiredError("viewport writer already finalized")
        if not self._frames:
            raise ViewportVideoRequiredError("no viewport frames captured")
        header = json.dumps(
            {
                "codec": self.codec,
                "fps": self.fps,
                "frame_count": len(self._frames),
                "frame_indices": [frame.frame_index for frame in self._frames],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        body = b"".join(frame.payload for frame in self._frames)
        payload = header + b"\n---frames---\n" + body
        digest = digest_bytes(payload)
        rel = f"blobs/{digest[:16]}_viewport.mp4"
        target = attempt_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        self._closed = True
        return EncodedVideoArtifact(
            relative_path=rel,
            sha256=digest,
            size_bytes=len(payload),
            fps=self.fps,
            frame_count=len(self._frames),
            codec=self.codec,
        )

    def bind_attempt_path(self, attempt_path: Path) -> None:
        return None

    def close(self) -> None:
        self._closed = True


@dataclass
class ScriptedSimulator:
    native_dt: float = 0.05
    control_tick: int = 0
    reference_displacement_m: float = 0.0
    initial_supported_z: float = 0.0
    object_z: float = 0.0
    object_xy: tuple[float, float] = (0.0, 0.0)
    gripper_xy: tuple[float, float] = (0.0, 0.0)
    gripper_z: float = 0.10
    contact: bool = False
    detached: bool = False
    camera_ids: tuple[str, ...] = ("wrist", "external")
    carry_script: Callable[[float, int], tuple[bool, bool, float]] | None = None
    release_at_tick: Optional[int] = None
    crash_at_tick: Optional[int] = None
    viewport_unavailable: bool = False
    closed: bool = False
    support_surface_tol_m: float = 0.015
    stability_speed_max_m_s: float = 0.02
    object_linear_speed_m_s: float = 0.0
    object_angular_speed_rad_s: float = 0.0
    predicates_available: bool = True
    moved_object_mask_pixels: int = 0
    moved_object_mask_pixels_by_camera: dict[str, int] = field(default_factory=dict)
    support_contacts: tuple[str, ...] | None = ("table",)
    object_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    _settling_baseline_position: tuple[float, float, float] | None = None
    _settling_baseline_orientation_wxyz: tuple[float, float, float, float] | None = None
    boundary_violation: bool = False

    def reset(self, *, env_seed: int) -> SimulatorSnapshot:
        self.control_tick = 0
        self.reference_displacement_m = 0.0
        self.object_z = self.initial_supported_z
        self.contact = False
        self.detached = False
        self.closed = False
        return self._snapshot()

    def step_control(self, action: tuple[float, ...] | None) -> SimulatorSnapshot:
        if self.crash_at_tick is not None and self.control_tick >= self.crash_at_tick:
            raise RuntimeError("simulator crash")
        self.control_tick += 1
        sim_time = self.control_tick * self.native_dt
        if self.carry_script is not None:
            self.contact, self.detached, self.object_z = self.carry_script(sim_time, self.control_tick)
        if self.release_at_tick is not None and self.control_tick >= self.release_at_tick:
            self.detached = True
            self.contact = False
        return self._snapshot()

    def capture_observation(self) -> CapturedObservation:
        native = build_observation_audit_payload(
            reference_displacement_m=self.reference_displacement_m,
            camera_ids=self.camera_ids,
            moved_object_mask_pixels_by_camera=self.moved_object_mask_pixels_by_camera or None,
            extra={
                "tick": self.control_tick,
                "object_z": self.object_z,
                "contact": self.contact,
            },
        )
        payload = json.dumps(native, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return CapturedObservation(
            payload=payload,
            camera_ids=self.camera_ids,
            state_hash=_digest_bytes(payload),
            native_input=native,
        )

    def capture_viewport_frame(self) -> Any:
        if self.viewport_unavailable:
            return None
        return _test_viewport_capture(tick=self.control_tick, z=self.object_z)

    def set_reference_offset(self, displacement_m: float, direction: tuple[float, float]) -> None:
        self.reference_displacement_m = displacement_m

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        return None

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        if snapshot is not None:
            state = snapshot.object_state
            self._settling_baseline_position = (state.object_x, state.object_y, state.object_z_pos)
        else:
            self._settling_baseline_position = (self.object_xy[0], self.object_xy[1], self.object_z)
        self._settling_baseline_orientation_wxyz = self.object_orientation_wxyz

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        if not self.predicates_available:
            return TerminalPhysicalPredicates(
                available=False,
                missing_fields=("terminal_predicates",),
            )
        drift_m = 0.0
        orientation_drift = 0.0
        if self._settling_baseline_position is not None:
            drift_m = position_drift_m(
                self._settling_baseline_position,
                (self.object_xy[0], self.object_xy[1], self.object_z),
            )
        if self._settling_baseline_orientation_wxyz is not None:
            orientation_drift = geodesic_orientation_delta_rad(
                self._settling_baseline_orientation_wxyz,
                self.object_orientation_wxyz,
            )
        sample = evaluate_horizontal_terminal_sample(
            detached=self.detached,
            linear_speed_m_s=self.object_linear_speed_m_s,
            angular_speed_rad_s=self.object_angular_speed_rad_s,
            support_contacts=self.support_contacts,
            position_drift_m=drift_m,
            orientation_drift_rad=orientation_drift,
        )
        return TerminalPhysicalPredicates(
            available=sample.available,
            allowed_support=sample.allowed_support,
            stable_for_dwell=sample.stable_for_dwell,
            boundary_violation=self.boundary_violation,
            linear_speed_m_s=sample.linear_speed_m_s,
            angular_speed_rad_s=sample.angular_speed_rad_s,
            position_drift_m=sample.position_drift_m,
            orientation_drift_rad=sample.orientation_drift_rad,
            support_contacts=sample.support_contacts,
            support_evidence_available=sample.support_evidence_available,
            moved_object_mask_pixels=self.moved_object_mask_pixels,
            missing_fields=sample.missing_fields,
        )

    def close(self) -> None:
        self.closed = True

    def _snapshot(self) -> SimulatorSnapshot:
        sim_time = self.control_tick * self.native_dt
        return SimulatorSnapshot(
            sim_time=sim_time,
            control_tick=self.control_tick,
            object_state=ObjectKinematicState(
                sim_time=sim_time,
                control_tick=self.control_tick,
                object_z=self.object_z,
                initial_supported_z=self.initial_supported_z,
                gripper_x=self.gripper_xy[0],
                gripper_y=self.gripper_xy[1],
                gripper_z=self.gripper_z,
                object_x=self.object_xy[0],
                object_y=self.object_xy[1],
                object_z_pos=self.object_z,
                contact=self.contact,
                detached=self.detached,
            ),
            reference_displacement_m=self.reference_displacement_m,
            terminal_predicates=self.sample_terminal_predicates(),
        )


@dataclass
class ScriptedPolicy:
    actions: tuple[tuple[float, ...], ...] = ((0.1,),)
    fail_on_request: Optional[int] = None
    invalid_on_request: Optional[int] = None
    future_on_request: Optional[bytes] = None
    _count: int = 0
    last_executed_action_count: int | None = None
    last_native_input: Any = None
    closed: bool = False

    def reset(self, *, policy_seed: int, prompt_text: str) -> None:
        self._count = 0
        self.last_executed_action_count = None
        self.last_native_input = None
        self.closed = False

    def infer(self, observation: ObservationPacket) -> PolicyResponse:
        self._count += 1
        self.last_executed_action_count = observation.executed_action_count
        self.last_native_input = observation.native_input
        if self.fail_on_request is not None and self._count >= self.fail_on_request:
            raise RuntimeError("policy interface failure")
        if self.invalid_on_request is not None and self._count >= self.invalid_on_request:
            raise PolicyModelInvalidError("nonfinite action chunk")
        action_bytes = repr(self.actions).encode("utf-8")
        future = None
        if self.future_on_request is not None:
            future = FutureArtifact(
                kind="nano_decoded_future",
                payload=self.future_on_request,
                payload_sha256=_digest_bytes(self.future_on_request),
            )
        return PolicyResponse(
            chunk_id=f"chunk-{self._count:03d}",
            actions=self.actions,
            wall_duration_s=0.01,
            action_sha256=_digest_bytes(action_bytes),
            generated_horizon=len(self.actions),
            future_artifact=future,
        )

    def close(self) -> None:
        self.closed = True


@dataclass
class ScriptedTerminalScorer:
    """Deterministic terminal scorer stub for runtime tests."""

    geometric_relation_correct: bool = False
    success: bool = False
    stable_for_dwell: bool = True
    allowed_support: bool = True
    predicates_available: bool = True

    def score_terminal(
        self,
        *,
        snapshot: SimulatorSnapshot,
        runtime_flags: EpisodeRuntimeFlags,
        passive_settling_reason: str | None,
        grasp_occurred: bool = False,
        carry_verified: bool = False,
        settling_predicates: tuple[TerminalPhysicalPredicates, ...] = (),
    ) -> TerminalScoringEvidence:
        released = snapshot.object_state.detached
        if settling_predicates:
            from experiments.online_correction_v4.droid_scorer import aggregate_settling_predicates

            physical = aggregate_settling_predicates(settling_predicates, dwell_ticks=2)
        elif snapshot.terminal_predicates is not None:
            physical = snapshot.terminal_predicates
        else:
            physical = TerminalPhysicalPredicates(available=self.predicates_available)
        support_ok = physical.allowed_support if physical.available else self.allowed_support
        stable_ok = physical.stable_for_dwell if physical.available else self.stable_for_dwell
        return TerminalScoringEvidence(
            success=self.success and released and carry_verified and support_ok and stable_ok,
            grasp_occurred=grasp_occurred,
            carry_verified=carry_verified,
            released=released,
            geometric_relation_correct=self.geometric_relation_correct,
            allowed_support=support_ok,
            stable_for_dwell=stable_ok and released,
            predicates_available=physical.available,
            transport_incomplete=carry_verified and passive_settling_reason == "timeout" and not released,
            unresolved_behavioral_failure=passive_settling_reason == "release" and not physical.available,
        )


def actions_are_finite(actions: tuple[tuple[float, ...], ...]) -> bool:
    for action in actions:
        for value in action:
            if not math.isfinite(value):
                return False
    return True
