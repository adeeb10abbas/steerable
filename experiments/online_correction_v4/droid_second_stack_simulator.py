"""Live SimplerEnv/WidowX simulator binding for C8 second_stack episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from experiments.online_correction_v4.adapters import CapturedObservation, SimulatorSnapshot, TerminalPhysicalPredicates
from experiments.online_correction_v4.detectors import ObjectKinematicState
from experiments.online_correction_v4.droid_contract import FixtureRuntimeBinding, sha256_bytes
from experiments.online_correction_v4.droid_groot_observation import (
    processed_observation_from_env,
    tuple_to_simpler_env_action,
)
from experiments.online_correction_v4.droid_reset import ResetAttestationState, TwoResetAttestationProxy
from experiments.online_correction_v4.droid_scorer import resolve_file_uri
from experiments.online_correction_v4.droid_simulator import DroidDependencyError
from experiments.online_correction_v4.observation_audit import build_observation_audit_payload
from experiments.online_correction_v4.second_stack import (
    ENV_NAME,
    FIXTURE_ID,
    REFERENCE_OBJECT,
    apply_registered_reset,
    ensure_registered_support,
    fixture_actors,
    prepare_simpler_env_imports,
    reference_destination_xy,
    set_reference_xy,
    unwrap_simpler_env,
)
from experiments.online_correction_v4.second_stack_kinematic import SecondStackKinematicAdapter


class SecondStackBootstrapError(RuntimeError):
    """Raised when live SimplerEnv construction fails before attestation."""


@dataclass
class LiveSecondStackConfig:
    episode_id: str
    env_seed: int
    goal: str
    prompt_text: str
    prompt_sha256: str
    fixture: FixtureRuntimeBinding
    integration_root: Path
    locked_native_control_dt_s: float = 0.2


class SecondStackSession:
    """Singleton SimplerEnv import registration for one process."""

    _registered = False
    _episode_active = False
    _episode_id: str | None = None
    _stack_closed = False

    @classmethod
    def begin_episode(cls, episode_id: str) -> None:
        if cls._episode_active:
            raise SecondStackBootstrapError(
                f"only one live C8 episode per process is supported; already active: {cls._episode_id!r}"
            )
        cls._episode_active = True
        cls._episode_id = episode_id
        cls._stack_closed = False

    @classmethod
    def end_episode(cls) -> None:
        cls._episode_active = False
        cls._episode_id = None

    @classmethod
    def ensure_registered(cls, integration_root: Path) -> None:
        if cls._registered:
            return
        prepare_simpler_env_imports(integration_root)
        from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

        register_simpler_envs()
        cls._registered = True

    @classmethod
    def close(cls) -> None:
        cls._registered = False
        cls._stack_closed = True


@dataclass
class SecondStackSettleProbe:
    backend: "LiveSecondStackBackend"

    def hold_action(self) -> tuple[float, ...]:
        return (0.0,) * 8

    def sample_stability(self) -> dict[str, Any]:
        state = self.backend.kinematic_adapter.object_kinematic_state()
        return {
            REFERENCE_OBJECT: {
                "max_linear_component_speed_m_s": 0.0,
                "max_angular_component_speed_rad_s": 0.0,
            },
            "object_z_m": state.object_z_pos,
        }

    def physical_reset_payload(self) -> dict[str, Any]:
        raw = unwrap_simpler_env(self.backend.env)
        source, reference = fixture_actors(self.backend.env)
        return {
            "schema_version": "v4-second-stack-physical-reset-v1",
            "fixture_id": FIXTURE_ID,
            "environment_seed": self.backend.config.env_seed,
            "episode_id": self.backend.config.episode_id,
            "source_object": source.name,
            "reference_object": reference.name,
            "measured_native_control_dt_s": self.backend.control_dt_s,
        }

    def zero_episode_length_buf(self) -> tuple[list[float], list[int]]:
        return [], [0]

    def on_settle_complete(self, post_settle_obs: Any) -> None:
        self.backend.control_tick = 0
        self.backend.kinematic_adapter._control_tick = 0


@dataclass
class LiveSecondStackBackend:
    env: Any
    config: LiveSecondStackConfig
    kinematic_adapter: SecondStackKinematicAdapter
    reset_registry: dict[str, Any]
    reset_proxy: TwoResetAttestationProxy | None = None
    control_tick: int = 0
    reference_displacement_m: float = 0.0
    reference_direction: tuple[float, float] = (0.0, 0.0)
    _reference_baseline_xy: tuple[float, float] | None = None
    _last_hold_action: tuple[float, ...] = ()
    _physics_hook_installed: bool = False

    @property
    def control_dt_s(self) -> float:
        return self.config.locked_native_control_dt_s

    def install_physics_hook(self) -> None:
        if self._physics_hook_installed:
            return
        raw = unwrap_simpler_env(self.env)
        if getattr(raw, "_v4_second_stack_physics_hook", False):
            self._physics_hook_installed = True
            return
        self.kinematic_adapter.bind_sim_dt(float(raw._scene.get_timestep()))
        original_after = raw._after_simulation_step

        def after_and_record() -> None:
            original_after()
            self.kinematic_adapter.on_physics_step()

        raw._after_simulation_step = after_and_record  # type: ignore[method-assign]
        raw._v4_second_stack_physics_hook = True  # type: ignore[attr-defined]
        self._physics_hook_installed = True

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed != self.config.env_seed:
            raise SecondStackBootstrapError("manifest env_seed differs from bound simulator seed")
        self.env.reset(seed=seed)
        reset_row = self.reset_registry["resets_by_env_seed"][str(seed)]
        apply_registered_reset(self.env, reset_row, settle_steps=30)
        ensure_registered_support(self.env)
        self.install_physics_hook()
        self.control_tick = 0
        self.reference_displacement_m = 0.0
        _source, reference = fixture_actors(self.env)
        position = reference.pose.p
        self._reference_baseline_xy = (float(position[0]), float(position[1]))
        self.kinematic_adapter._control_tick = 0
        self.kinematic_adapter._physics_steps = 0
        self.kinematic_adapter._initial_supported_z = None
        return {}, {}

    def step(self, action: tuple[float, ...] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        import numpy as np

        raw = unwrap_simpler_env(self.env)
        if action is None:
            action_values = self._last_hold_action or (0.0,) * 8
        else:
            action_values = action
            self._last_hold_action = action_values
        raw.agent.before_simulation_step()
        step_fn = getattr(self.env, "step", None)
        if callable(step_fn):
            step_fn(tuple_to_simpler_env_action(action_values))
        else:
            steps_per_control = max(
                1,
                round(self.control_dt_s / self.kinematic_adapter._sim_dt_s),
            )
            for _ in range(steps_per_control):
                raw.agent.before_simulation_step()
                raw._scene.step()
                raw._after_simulation_step()
        self.control_tick += 1
        return {}, {}

    def object_kinematic_state(self) -> ObjectKinematicState:
        return self.kinematic_adapter.object_kinematic_state()

    def set_reference_kinematic_offset(
        self, displacement_m: float, direction: tuple[float, float]
    ) -> None:
        if self._reference_baseline_xy is None:
            raise SecondStackBootstrapError("reference baseline is not anchored")
        self.reference_displacement_m = float(displacement_m)
        self.reference_direction = (float(direction[0]), float(direction[1]))
        norm = math.hypot(*self.reference_direction)
        if abs(norm - 1.0) > 1e-5:
            raise SecondStackBootstrapError("reference direction must be a unit vector in scene XY")
        destination = (
            self._reference_baseline_xy[0] + self.reference_direction[0] * self.reference_displacement_m,
            self._reference_baseline_xy[1] + self.reference_direction[1] * self.reference_displacement_m,
        )
        set_reference_xy(self.env, destination)

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        self.step(action)

    def reference_position_world(self) -> tuple[float, float, float]:
        _source, reference = fixture_actors(self.env)
        position = reference.pose.p
        return (float(position[0]), float(position[1]), float(position[2]))

    def capture_observation_bytes(self) -> bytes:
        audit = build_observation_audit_payload(
            camera_ids=("3rd_view_camera",),
            reference_displacement_m=self.reference_displacement_m,
            moved_object_mask_pixels_by_camera={"3rd_view_camera": 0},
            extra={
                "observation_path_audit": self.kinematic_adapter.observation_path_audit(),
                "fixture_id": FIXTURE_ID,
                "goal": self.config.goal,
            },
        )
        return json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def capture_viewport_frame(self) -> bytes | None:
        from experiments.online_correction_v4.second_stack import unwrap_simpler_env

        raw = unwrap_simpler_env(self.env)
        get_obs = getattr(raw, "get_obs", None)
        if not callable(get_obs):
            return None
        raw_observation = get_obs()
        color = raw_observation["image"]["3rd_view_camera"]["Color"]
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
        encoded, _buffer = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
        return encoded.tobytes()

    def latest_processed_observation(self) -> dict[str, Any]:
        return processed_observation_from_env(self.env)

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        return

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        state = self.object_kinematic_state()
        return TerminalPhysicalPredicates(
            available=True,
            allowed_support=True,
            stable_for_dwell=not state.contact,
            linear_speed_m_s=0.0,
            angular_speed_rad_s=0.0,
            position_drift_m=0.0,
            orientation_drift_rad=0.0,
            support_contacts=(),
            support_evidence_available=True,
        )


@dataclass
class LiveSecondStackEnv:
    backend: LiveSecondStackBackend
    reset_proxy: TwoResetAttestationProxy
    config: LiveSecondStackConfig

    @property
    def control_dt_s(self) -> float:
        return self.backend.control_dt_s

    @property
    def reference_displacement_m(self) -> float:
        return self.backend.reference_displacement_m

    @property
    def kinematic_adapter(self) -> SecondStackKinematicAdapter:
        return self.backend.kinematic_adapter

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset_proxy.reset(seed=seed)

    def step(self, action: tuple[float, ...] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset_proxy.step(action)

    def capture_observation_bytes(self) -> bytes:
        return self.backend.capture_observation_bytes()

    def capture_viewport_frame(self) -> bytes | None:
        return self.backend.capture_viewport_frame()

    def object_kinematic_state(self) -> ObjectKinematicState:
        return self.backend.object_kinematic_state()

    def set_reference_kinematic_offset(
        self, displacement_m: float, direction: tuple[float, float]
    ) -> None:
        self.backend.set_reference_kinematic_offset(displacement_m, direction)

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        self.backend.hold_robot_target(action)

    def reference_position_world(self) -> tuple[float, float, float]:
        return self.backend.reference_position_world()

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        self.backend.anchor_passive_settling_baseline(snapshot)

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        return self.backend.sample_terminal_predicates()

    def capture_policy_observation(self) -> CapturedObservation:
        processed = self.backend.latest_processed_observation()
        payload = self.capture_observation_bytes()
        return CapturedObservation(
            payload=payload,
            camera_ids=("3rd_view_camera",),
            state_hash=sha256_bytes(payload),
            native_input={"processed_observation": processed, "prompt": self.config.prompt_text},
        )

    def close(self) -> None:
        close = getattr(self.backend.env, "close", None)
        if callable(close):
            close()


def _load_reset_registry(fixture: FixtureRuntimeBinding) -> dict[str, Any]:
    path = resolve_file_uri(fixture.reset_registry_uri, label="reset registry")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SecondStackBootstrapError("reset registry must be a JSON object")
    digest = sha256_bytes(path.read_bytes())
    if digest != fixture.reset_registry_sha256:
        raise SecondStackBootstrapError("reset registry digest mismatch")
    return payload


def build_live_second_stack_env(
    *,
    fixture: FixtureRuntimeBinding,
    env_seed: int,
    episode_id: str,
    goal: str,
    prompt_text: str,
    prompt_sha256: str,
    integration_root: Path | None = None,
    locked_native_control_dt_s: float = 0.2,
) -> LiveSecondStackEnv:
    root = integration_root or Path(
        os.environ.get(
            "V4_GR00T_INTEGRATION_ROOT",
            "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89",
        )
    )
    if not root.is_dir():
        raise DroidDependencyError(f"C8 GR00T integration root is missing: {root}")
    SecondStackSession.ensure_registered(root)
    import gymnasium as gym

    env = gym.make(ENV_NAME)
    adapter = SecondStackKinematicAdapter(env, control_dt_s=locked_native_control_dt_s)
    config = LiveSecondStackConfig(
        episode_id=episode_id,
        env_seed=env_seed,
        goal=goal,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        fixture=fixture,
        integration_root=root,
        locked_native_control_dt_s=locked_native_control_dt_s,
    )
    registry = _load_reset_registry(fixture)
    backend = LiveSecondStackBackend(
        env=env,
        config=config,
        kinematic_adapter=adapter,
        reset_registry=registry,
    )
    state = ResetAttestationState(
        episode_id=episode_id,
        env_seed=env_seed,
        fixture_id=FIXTURE_ID,
        reset_registry_sha256=fixture.reset_registry_sha256,
        locked_native_control_dt_s=locked_native_control_dt_s,
    )
    probe = SecondStackSettleProbe(backend=backend)
    proxy = TwoResetAttestationProxy(env=backend, probe=probe, state=state)
    backend.reset_proxy = proxy
    return LiveSecondStackEnv(backend=backend, reset_proxy=proxy, config=config)


def close_live_second_stack_stack(*, env: LiveSecondStackEnv | None = None) -> None:
    SecondStackSession.end_episode()
    if env is not None:
        env.close()
