"""RoboLab DROID simulator adapter with lazy imports and test fakes."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from experiments.online_correction_v4.adapters import CapturedObservation, ObservationPacket, SimulatorSnapshot, TerminalPhysicalPredicates
from experiments.online_correction_v4.detectors import ObjectKinematicState
from experiments.online_correction_v4.observation_audit import build_observation_audit_payload
from experiments.online_correction_v4.terminal_stability import (
    evaluate_horizontal_terminal_sample,
    geodesic_orientation_delta_rad,
    position_drift_m,
)
from experiments.online_correction_v4.droid_contract import (
    FixtureRuntimeBinding,
    sha256_bytes,
)
from experiments.online_correction_v4.droid_reset import (
    ResetAttestationState,
    SettleProbe,
    TwoResetAttestationProxy,
)


class DroidDependencyError(ImportError):
    """Raised when live RoboLab/Cosmos/OpenPI imports are unavailable."""


@runtime_checkable
class RoboLabEnvSurface(Protocol):
    control_dt_s: float

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def step(self, action: tuple[float, ...] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        ...

    def capture_observation_bytes(self) -> bytes:
        ...

    def capture_viewport_frame(self) -> bytes | None:
        ...

    def object_kinematic_state(self) -> ObjectKinematicState:
        ...

    def set_reference_kinematic_offset(
        self, displacement_m: float, direction: tuple[float, float]
    ) -> None:
        ...

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        ...


@dataclass
class FakeRoboLabEnv:
    """Deterministic in-memory RoboLab stand-in for contract tests."""

    native_dt: float = 0.05
    env_seed: int = 0
    control_tick: int = 0
    reference_displacement_m: float = 0.0
    reference_direction: tuple[float, float] = (1.0, 0.0)
    last_hold_action: tuple[float, ...] = ()
    object_state: ObjectKinematicState = field(
        default_factory=lambda: ObjectKinematicState(
            sim_time=0.0,
            control_tick=0,
            object_z=0.0,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.10,
            object_x=0.0,
            object_y=0.0,
            object_z_pos=0.0,
            contact=False,
            detached=False,
        )
    )
    observation_builder: Callable[[], bytes] | None = None
    reset_calls: int = 0
    support_surface_tol_m: float = 0.015
    stability_speed_max_m_s: float = 0.02
    object_linear_speed_m_s: float = 0.0
    object_angular_speed_rad_s: float = 0.0
    boundary_violation: bool = False
    collision_terminal_failure: bool = False
    predicates_available: bool = True
    moved_object_mask_pixels: int = 0
    moved_object_mask_pixels_by_camera: dict[str, int] = field(default_factory=dict)
    policy_camera_ids: tuple[str, ...] = ("head", "wrist_left", "wrist_right")
    support_contacts: tuple[str, ...] | None = ("table",)
    _settling_baseline_position: tuple[float, float, float] | None = None
    _settling_baseline_orientation_wxyz: tuple[float, float, float, float] | None = None
    _object_orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @property
    def control_dt_s(self) -> float:
        return self.native_dt

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self.env_seed = seed
        self.reset_calls += 1
        self.control_tick = 0
        self.reference_displacement_m = 0.0
        return {"seed": seed, "tick": self.control_tick}, {"reset_index": self.reset_calls}

    def step(self, action: tuple[float, ...] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.control_tick += 1
        sim_time = self.control_tick * self.native_dt
        state = self.object_state
        self.object_state = ObjectKinematicState(
            sim_time=sim_time,
            control_tick=self.control_tick,
            object_z=state.object_z,
            initial_supported_z=state.initial_supported_z,
            gripper_x=state.gripper_x,
            gripper_y=state.gripper_y,
            gripper_z=state.gripper_z,
            object_x=state.object_x,
            object_y=state.object_y,
            object_z_pos=state.object_z_pos,
            contact=state.contact,
            detached=state.detached,
        )
        return {"tick": self.control_tick, "action": action}, {}

    def capture_observation_bytes(self) -> bytes:
        if self.observation_builder is not None:
            return self.observation_builder()
        payload = build_observation_audit_payload(
            reference_displacement_m=self.reference_displacement_m,
            camera_ids=self.policy_camera_ids,
            moved_object_mask_pixels_by_camera=self.moved_object_mask_pixels_by_camera or None,
            extra={
                "seed": self.env_seed,
                "tick": self.control_tick,
                "object_state": self.object_state.__dict__,
            },
        )
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def capture_viewport_frame(self) -> Any:
        from experiments.online_correction_v4.testing import _test_viewport_capture

        return _test_viewport_capture(tick=self.control_tick, z=self.object_state.object_z)

    def object_kinematic_state(self) -> ObjectKinematicState:
        return self.object_state

    def set_reference_kinematic_offset(
        self, displacement_m: float, direction: tuple[float, float]
    ) -> None:
        self.reference_displacement_m = displacement_m
        self.reference_direction = direction

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        self.last_hold_action = action

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        state = snapshot.object_state if snapshot is not None else self.object_state
        self._settling_baseline_position = (state.object_x, state.object_y, state.object_z_pos)
        self._settling_baseline_orientation_wxyz = self._object_orientation_wxyz

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        if not self.predicates_available:
            return TerminalPhysicalPredicates(
                available=False,
                missing_fields=("terminal_predicates",),
            )
        state = self.object_state
        drift_m = 0.0
        orientation_drift = 0.0
        if self._settling_baseline_position is not None:
            drift_m = position_drift_m(
                self._settling_baseline_position,
                (state.object_x, state.object_y, state.object_z_pos),
            )
        if self._settling_baseline_orientation_wxyz is not None:
            orientation_drift = geodesic_orientation_delta_rad(
                self._settling_baseline_orientation_wxyz,
                self._object_orientation_wxyz,
            )
        sample = evaluate_horizontal_terminal_sample(
            detached=state.detached,
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
            collision_terminal_failure=self.collision_terminal_failure,
            linear_speed_m_s=sample.linear_speed_m_s,
            angular_speed_rad_s=sample.angular_speed_rad_s,
            position_drift_m=sample.position_drift_m,
            orientation_drift_rad=sample.orientation_drift_rad,
            support_contacts=sample.support_contacts,
            support_evidence_available=sample.support_evidence_available,
            moved_object_mask_pixels=self.moved_object_mask_pixels,
            missing_fields=sample.missing_fields,
        )


@dataclass
class FakeSettleProbe:
    env: FakeRoboLabEnv
    episode_length_buf: list[int] = field(default_factory=lambda: [0])

    def hold_action(self) -> tuple[float, ...]:
        return (0.0,) * 8

    def sample_stability(self) -> dict[str, Any]:
        return {
            "rubiks_cube": {
                "max_linear_component_speed_m_s": 0.0,
                "max_angular_component_speed_rad_s": 0.0,
            }
        }

    def physical_reset_payload(self) -> dict[str, Any]:
        return {
            "schema_version": "v4-droid-physical-reset-v1",
            "environment_seed": self.env.env_seed,
            "control_tick": self.env.control_tick,
        }

    def zero_episode_length_buf(self) -> tuple[list[float], list[int]]:
        before = [float(value) for value in self.episode_length_buf]
        self.episode_length_buf = [0]
        return before, list(self.episode_length_buf)

    def on_settle_complete(self, post_settle_obs: Any) -> None:
        if isinstance(post_settle_obs, dict):
            tick = post_settle_obs.get("tick")
            if isinstance(tick, int):
                self.env.control_tick = 0
        else:
            self.env.control_tick = 0


@dataclass
class DroidSimulatorAdapter:
    """Common RoboLab binding implementing the V4 SimulatorAdapter protocol."""

    episode_id: str
    env_seed: int
    fixture: FixtureRuntimeBinding
    env: RoboLabEnvSurface
    reset_proxy: TwoResetAttestationProxy | None = None
    attestation_path: str | None = None
    prompt_sha256: str = ""
    runtime_identity_sha256: str = ""
    viewport_frames: list[bytes] = field(default_factory=list)
    observation_records: list[dict[str, Any]] = field(default_factory=list)
    _latest_snapshot: SimulatorSnapshot | None = None
    _native_dt_measured: float | None = None
    experiment_clock: Any | None = None

    @classmethod
    def from_fake(
        cls,
        *,
        episode_id: str,
        env_seed: int,
        fixture: FixtureRuntimeBinding,
        env: FakeRoboLabEnv | None = None,
    ) -> DroidSimulatorAdapter:
        backend = env or FakeRoboLabEnv()
        state = ResetAttestationState(
            episode_id=episode_id,
            env_seed=env_seed,
            fixture_id=fixture.fixture_id,
            reset_registry_sha256=fixture.reset_registry_sha256,
        )
        probe = FakeSettleProbe(env=backend)
        proxy = TwoResetAttestationProxy(env=backend, probe=probe, state=state)
        return cls(
            episode_id=episode_id,
            env_seed=env_seed,
            fixture=fixture,
            env=backend,
            reset_proxy=proxy,
        )

    @classmethod
    def from_live_env(
        cls,
        *,
        episode_id: str,
        env_seed: int,
        fixture: FixtureRuntimeBinding,
        env: RoboLabEnvSurface,
        reset_proxy: TwoResetAttestationProxy,
    ) -> DroidSimulatorAdapter:
        return cls(
            episode_id=episode_id,
            env_seed=env_seed,
            fixture=fixture,
            env=env,
            reset_proxy=reset_proxy,
        )

    @property
    def native_control_dt_s(self) -> float:
        measured = self._native_dt_measured
        if measured is None:
            measured = float(getattr(self.env, "control_dt_s"))
            if measured <= 0:
                raise RuntimeError("native control dt must be measured from the environment")
            self._native_dt_measured = measured
        return measured

    def reset(self, *, env_seed: int) -> SimulatorSnapshot:
        if env_seed != self.env_seed:
            raise RuntimeError("manifest env_seed differs from bound simulator seed")
        if self.reset_proxy is not None:
            self.reset_proxy.reset(seed=env_seed)
            self.reset_proxy.reset(seed=env_seed)
            if (
                not self.reset_proxy.state.attestation_written
                and self.prompt_sha256
                and self.runtime_identity_sha256
            ):
                initial = self.env.capture_observation_bytes()
                self.finalize_reset_attestation(
                    prompt_sha256=self.prompt_sha256,
                    runtime_identity_sha256=self.runtime_identity_sha256,
                    initial_state_sha256=sha256_bytes(initial),
                )
        else:
            self.env.reset(seed=env_seed)
        if self.experiment_clock is not None:
            self.experiment_clock.reset()
        return self._snapshot_after_step()

    def capture_observation(self) -> CapturedObservation:
        if hasattr(self.env, "capture_policy_observation"):
            captured = self.env.capture_policy_observation()
            audit = json.loads(captured.payload.decode("utf-8"))
            native = captured.native_input if isinstance(captured.native_input, Mapping) else {}
            packed = native.get("packed_request")
            if isinstance(packed, Mapping):
                audit["packed_request"] = dict(packed)
            payload = json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
            return CapturedObservation(
                payload=payload,
                camera_ids=captured.camera_ids,
                state_hash=sha256_bytes(payload),
                native_input=captured.native_input,
            )
        payload = self.env.capture_observation_bytes()
        if not payload:
            raise RuntimeError("observation payload capture returned empty bytes")
        state_hash = self._state_hash()
        return CapturedObservation(
            payload=payload,
            camera_ids=("head", "wrist_left", "wrist_right"),
            state_hash=state_hash,
        )

    def step_control(self, action: tuple[float, ...] | None) -> SimulatorSnapshot:
        if self.reset_proxy is not None:
            self.reset_proxy.step(action)
        else:
            self.env.step(action)
        return self._snapshot_after_step()

    def set_reference_offset(self, displacement_m: float, direction: tuple[float, float]) -> None:
        self.env.set_reference_kinematic_offset(displacement_m, direction)

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        self.env.hold_robot_target(action)

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        anchor = getattr(self.env, "anchor_passive_settling_baseline", None)
        if callable(anchor):
            anchor(snapshot)

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        sample = getattr(self.env, "sample_terminal_predicates", None)
        if callable(sample):
            return sample()
        return TerminalPhysicalPredicates(
            available=False,
            missing_fields=("sample_terminal_predicates",),
        )

    def capture_viewport_frame(self) -> bytes | None:
        return self.env.capture_viewport_frame()

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def capture_observation_packet(
        self,
        *,
        observation_id: str,
        capture_time_s: float,
        camera_ids: tuple[str, ...] = ("head", "wrist_left", "wrist_right"),
    ) -> ObservationPacket:
        captured = self.capture_observation()
        if camera_ids != captured.camera_ids:
            camera_ids = captured.camera_ids
        record = {
            "observation_id": observation_id,
            "capture_time_s": capture_time_s,
            "payload_sha256": sha256_bytes(captured.payload),
            "camera_ids": list(camera_ids),
            "native_control_dt_s": self.native_control_dt_s,
        }
        self.observation_records.append(record)
        viewport = self.env.capture_viewport_frame()
        if viewport:
            self.viewport_frames.append(viewport)
        return ObservationPacket(
            observation_id=observation_id,
            capture_time_s=capture_time_s,
            payload=captured.payload,
            payload_sha256=sha256_bytes(captured.payload),
            camera_ids=camera_ids,
            state_hash=captured.state_hash,
        )

    def finalize_reset_attestation(
        self,
        *,
        prompt_sha256: str,
        runtime_identity_sha256: str,
        initial_state_sha256: str,
    ) -> dict[str, Any]:
        if self.reset_proxy is None:
            raise RuntimeError("reset attestation requires a live reset proxy")
        self.prompt_sha256 = prompt_sha256
        self.runtime_identity_sha256 = runtime_identity_sha256
        return self.reset_proxy.finalize_attestation(
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            initial_state_sha256=initial_state_sha256,
        )

    def _snapshot_after_step(self) -> SimulatorSnapshot:
        state = self.env.object_kinematic_state()
        ref_pos = (0.0, 0.0, 0.0)
        if hasattr(self.env, "reference_position_world"):
            ref_pos = self.env.reference_position_world()
        elif isinstance(self.env, FakeRoboLabEnv):
            ref_pos = (0.0, 0.0, 0.0)
        snapshot = SimulatorSnapshot(
            sim_time=state.sim_time,
            control_tick=state.control_tick,
            object_state=state,
            reference_position_world=ref_pos,
            reference_displacement_m=float(getattr(self.env, "reference_displacement_m", 0.0)),
            robot_safe_hold=getattr(self.env, "last_hold_action", ()),
            boundary_violation=bool(getattr(self.env, "boundary_violation", False)),
            terminal_predicates=self.sample_terminal_predicates(),
            extra={"native_control_dt_s": self.native_control_dt_s},
        )
        self._latest_snapshot = snapshot
        return snapshot

    def _state_hash(self) -> str:
        if self._latest_snapshot is None:
            return ""
        state = self._latest_snapshot.object_state
        payload = json.dumps(state.__dict__, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def import_robolab_stack() -> dict[str, Any]:
    """Lazy import boundary for live RoboLab execution."""
    missing: list[str] = []
    modules: dict[str, Any] = {}
    for name, import_path in (
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("AppLauncher", "isaaclab.app.AppLauncher"),
        ("robolab_runtime", "robolab.core.environments.runtime"),
    ):
        try:
            if "." in import_path:
                parent, child = import_path.rsplit(".", 1)
                imported = __import__(parent, fromlist=[child])
                modules[name] = getattr(imported, child)
            else:
                modules[name] = __import__(import_path)
        except ImportError:
            missing.append(import_path)
    if missing:
        raise DroidDependencyError(
            "live RoboLab stack unavailable; missing imports: " + ", ".join(sorted(missing))
        )
    return modules


def build_live_robolab_env(**kwargs: Any) -> RoboLabEnvSurface:
    """Factory hook for cluster execution; fails closed without RoboLab."""
    from experiments.online_correction_v4.droid_robolab import build_live_robolab_env as _build

    return _build(**kwargs)
