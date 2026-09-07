"""Abstract simulator and policy adapter protocols for the V4 episode runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from experiments.online_correction_v4.contracts import EpisodeRuntimeFlags
from experiments.online_correction_v4.detectors import ObjectKinematicState


class PolicyModelInvalidError(Exception):
    """Policy emitted a non-applicable output with intact inputs and no transport defect."""


class ViewportVideoRequiredError(Exception):
    """Mandatory viewport video could not be encoded from captured frames."""


@dataclass(frozen=True)
class CapturedObservation:
    """Simulator-native policy observation bytes and provenance at capture time."""

    payload: bytes
    camera_ids: tuple[str, ...] = ()
    state_hash: str = ""
    native_input: Any = None


@dataclass(frozen=True)
class ObservationPacket:
    observation_id: str
    capture_time_s: float
    payload: bytes
    payload_sha256: str
    camera_ids: tuple[str, ...] = ()
    state_hash: str = ""
    native_input: Any = None
    executed_action_count: int = 0
    request_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FutureArtifact:
    kind: str
    payload: bytes
    payload_sha256: str = ""


@dataclass(frozen=True)
class PolicyResponse:
    chunk_id: str
    actions: tuple[tuple[float, ...], ...]
    wall_duration_s: float
    action_sha256: str = ""
    generated_horizon: int = 0
    future_artifact: FutureArtifact | None = None
    request_audit: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewportFrame:
    frame_index: int
    sim_time_s: float
    control_tick: int
    payload: bytes
    payload_sha256: str = ""
    format_kind: str = "encoded_image"
    width: int = 0
    height: int = 0
    channels: int = 3


@dataclass(frozen=True)
class EncodedVideoArtifact:
    relative_path: str
    sha256: str
    size_bytes: int
    fps: float
    frame_count: int
    codec: str


@dataclass(frozen=True)
class TerminalPhysicalPredicates:
    """Evaluator-only terminal support/stability evidence sampled over passive settling."""

    available: bool = False
    allowed_support: bool = False
    allowed_containment: bool = False
    stable_for_dwell: bool = False
    boundary_violation: bool = False
    collision_terminal_failure: bool = False
    linear_speed_m_s: float = 0.0
    angular_speed_rad_s: float = 0.0
    position_drift_m: float = 0.0
    orientation_drift_rad: float = 0.0
    support_contacts: tuple[str, ...] = ()
    support_evidence_available: bool = False
    moved_object_mask_pixels: int = 0
    missing_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class TerminalScoringEvidence:
    """Frozen terminal scorer output consumed by the episode runner."""

    success: bool = False
    grasp_occurred: bool = False
    carry_verified: bool = False
    grasp_lost: bool = False
    released: bool = False
    geometric_relation_correct: bool = False
    allowed_support: bool = False
    allowed_containment: bool = False
    stable_for_dwell: bool = False
    boundary_violation: bool = False
    collision_terminal_failure: bool = False
    transport_incomplete: bool = False
    model_output_invalid: bool = False
    unresolved_behavioral_failure: bool = False
    predicates_available: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulatorSnapshot:
    sim_time: float
    control_tick: int
    object_state: ObjectKinematicState
    reference_position_world: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reference_displacement_m: float = 0.0
    robot_safe_hold: tuple[float, ...] = ()
    boundary_violation: bool = False
    support_contacts: tuple[str, ...] = ()
    terminal_predicates: TerminalPhysicalPredicates | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ViewportVideoWriter(Protocol):
    """Encodes captured viewport frames into a durable video artifact."""

    @property
    def fps(self) -> float:
        """Nominal capture frames per simulated second."""

    def append_frame(self, frame: ViewportFrame) -> None:
        """Record one viewport frame with fixed index metadata."""

    def finalize_video(self, *, attempt_path: Any) -> EncodedVideoArtifact:
        """Write the encoded video artifact or raise ViewportVideoRequiredError."""


@runtime_checkable
class SimulatorAdapter(Protocol):
    """Minimal simulator surface required by EpisodeRunner."""

    def reset(self, *, env_seed: int) -> SimulatorSnapshot:
        """Restore registered reset and return the initial snapshot."""

    def step_control(self, action: tuple[float, ...] | None) -> SimulatorSnapshot:
        """Advance one native control tick under the supplied action."""

    def capture_observation(self) -> CapturedObservation:
        """Capture the actual policy observation bytes, camera IDs, and state hash."""

    def capture_viewport_frame(self) -> bytes | None:
        """Capture one viewport RGB frame for the current tick, if available."""

    def set_reference_offset(self, displacement_m: float, direction: tuple[float, float]) -> None:
        """Kinematically move the reference object to the prescribed offset."""

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        """Hold last safe robot target during passive settling."""

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        """Sample evaluator-only support/stability/containment predicates for the current tick."""

    def close(self) -> None:
        """Release simulator resources without deleting recorded evidence."""


@runtime_checkable
class PolicyAdapter(Protocol):
    """Minimal policy surface required by EpisodeRunner."""

    def reset(self, *, policy_seed: int, prompt_text: str) -> None:
        """Reset policy session, RNG, and observation history."""

    def infer(self, observation: ObservationPacket) -> PolicyResponse:
        """Run inference on the captured observation packet."""


@runtime_checkable
class TerminalScoringAdapter(Protocol):
    """Frozen terminal scorer surface; never receives policy inputs."""

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
        """Return settled first-placement evidence for terminal adjudication."""
