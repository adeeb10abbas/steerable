"""Live RoboLab bootstrap for V4 DROID episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments.online_correction_v4.adapters import CapturedObservation, SimulatorSnapshot, TerminalPhysicalPredicates
from experiments.online_correction_v4.detectors import ObjectKinematicState
from experiments.online_correction_v4.droid_contract import FixtureRuntimeBinding, sha256_bytes
from experiments.online_correction_v4.droid_observation import pack_policy_request
from experiments.online_correction_v4.observation_audit import build_observation_audit_payload
from experiments.online_correction_v4.terminal_stability import (
    evaluate_horizontal_terminal_sample,
    geodesic_orientation_delta_rad,
    position_drift_m,
    target_object_contact_names,
)
from experiments.online_correction_v4.droid_reset import ResetAttestationState, TwoResetAttestationProxy
from experiments.online_correction_v4.droid_scorer import resolve_file_uri
from experiments.online_correction_v4.droid_simulator import DroidDependencyError, import_robolab_stack
from experiments.online_correction_v4.droid_task_files.constants import (
    ENV_ACTIVE_GOAL,
    ENV_QUEUE_ROW,
    ENV_QUEUE_ROW_SHA256,
    ENV_RESET_REGISTRY,
    ENV_RESET_REGISTRY_SHA256,
    MOVABLE_OBJECTS,
    REFERENCE_OBJECT,
    TARGET_OBJECT,
)
from experiments.online_correction_v4.droid_task_files.registry import (
    blocked_fixture_ids,
    resolve_active_registration,
)
from experiments.online_correction_v4.droid_task_files.reset_registry import (
    MODEL_BLIND_CANDIDATE_STATUS,
    RELEASED_FOR_POLICY_STATUS,
)
from experiments.online_correction_v4.droid_reset_verify import (
    load_bound_reset_registry,
    verify_measured_native_dt,
    verify_neutral_horizontal_layout,
    verify_physical_reset_against_registry,
)


SETTLE_OBJECTS = MOVABLE_OBJECTS
ACTION_DIM = 8


class RoboLabBootstrapError(RuntimeError):
    """Raised when live RoboLab construction fails before attestation."""


@dataclass(frozen=True)
class ResetFixtureBinding:
    """Minimal fixture binding allowed for zero-inference reset qualification."""

    fixture_id: str
    reset_registry_sha256: str
    reset_registry_uri: str


@dataclass
class LiveRoboLabConfig:
    episode_id: str
    env_seed: int
    goal: str
    prompt_text: str
    prompt_sha256: str
    policy_id: str
    fixture: FixtureRuntimeBinding | ResetFixtureBinding
    queue_row_path: Path
    queue_row_sha256: str
    output_dir: Path | None = None
    headless: bool = True
    locked_native_control_dt_s: float = 0.05
    g3_contact_probe: bool = False
    action_mode: str = "joint_position"


class RoboLabSession:
    """Singleton AppLauncher session; Isaac must start before RoboLab imports."""

    _started = False
    _simulation_app: Any = None
    _modules: dict[str, Any] = field(default_factory=dict) if False else {}
    _episode_active = False
    _episode_id: str | None = None
    _stack_closed = False
    _policy_transport_closed = False

    @classmethod
    def begin_episode(cls, episode_id: str) -> None:
        if cls._episode_active:
            raise RoboLabBootstrapError(
                f"only one live episode per process is supported; already active: {cls._episode_id!r}"
            )
        cls._episode_active = True
        cls._episode_id = episode_id
        cls._stack_closed = False
        cls._policy_transport_closed = False

    @classmethod
    def end_episode(cls) -> None:
        cls._episode_active = False
        cls._episode_id = None

    @classmethod
    def ensure_started(cls, *, headless: bool = True) -> Any:
        if cls._started and cls._simulation_app is not None:
            return cls._simulation_app
        import cv2  # noqa: F401 — RoboLab requires OpenCV before Isaac Lab startup
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
        import argparse

        from isaaclab.app import AppLauncher

        parser = argparse.ArgumentParser(add_help=False)
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args([])
        args.headless = headless
        args.enable_cameras = True
        launcher = AppLauncher(args)
        cls._simulation_app = launcher.app
        cls._started = True
        return cls._simulation_app

    @classmethod
    def close(cls) -> None:
        if cls._simulation_app is not None:
            cls._simulation_app.close()
        cls._simulation_app = None
        cls._started = False
        cls._stack_closed = True


def close_live_droid_stack(*, policy: Any = None) -> None:
    """Tear down live RoboLab resources once for any runner caller."""
    if RoboLabSession._started and not RoboLabSession._stack_closed:
        try:
            from experiments.online_correction_v4.droid_task_files.horizontal_shared import clear_episode_caches

            clear_episode_caches()
        except ImportError:
            pass
        RoboLabSession.end_episode()
        RoboLabSession.close()
    elif RoboLabSession._episode_active:
        RoboLabSession.end_episode()
    if policy is not None and not RoboLabSession._policy_transport_closed:
        transport = getattr(policy, "transport", None)
        if transport is not None and hasattr(transport, "close"):
            transport.close()
            RoboLabSession._policy_transport_closed = True


def build_live_robolab_env(
    *,
    fixture: FixtureRuntimeBinding | ResetFixtureBinding,
    env_seed: int,
    episode_id: str,
    goal: str,
    prompt_text: str,
    prompt_sha256: str,
    policy_id: str,
    queue_row_path: Path,
    queue_row_sha256: str,
    output_dir: Path | None = None,
    locked_native_control_dt_s: float = 0.05,
    g3_contact_probe: bool = False,
    action_mode: str = "joint_position",
) -> LiveRoboLabEnv:
    if fixture.fixture_id in blocked_fixture_ids():
        raise RoboLabBootstrapError(
            f"fixture {fixture.fixture_id!r} is blocked: {blocked_fixture_ids()[fixture.fixture_id]}"
        )
    if fixture.fixture_id != "horizontal":
        raise RoboLabBootstrapError(
            f"fixture {fixture.fixture_id!r} is not physically implemented in this checkout"
        )
    if action_mode not in {"joint_position", "absolute_ik"}:
        raise RoboLabBootstrapError(
            f"unsupported live RoboLab action mode: {action_mode!r}"
        )
    config = LiveRoboLabConfig(
        episode_id=episode_id,
        env_seed=env_seed,
        goal=goal,
        prompt_text=prompt_text,
        prompt_sha256=prompt_sha256,
        policy_id=policy_id,
        fixture=fixture,
        queue_row_path=queue_row_path,
        queue_row_sha256=queue_row_sha256,
        output_dir=output_dir,
        locked_native_control_dt_s=locked_native_control_dt_s,
        g3_contact_probe=g3_contact_probe,
        action_mode=action_mode,
    )
    _bind_runtime_env(config)
    RoboLabSession.ensure_started(headless=config.headless)
    modules = _import_robolab_modules(config)
    return _create_live_env(config, modules)


def _bind_runtime_env(config: LiveRoboLabConfig) -> None:
    os.environ[ENV_QUEUE_ROW] = str(config.queue_row_path.resolve())
    os.environ[ENV_QUEUE_ROW_SHA256] = config.queue_row_sha256
    os.environ[ENV_ACTIVE_GOAL] = config.goal
    reset_registry = resolve_file_uri(config.fixture.reset_registry_uri, label="reset_registry")
    os.environ[ENV_RESET_REGISTRY] = str(reset_registry)
    os.environ[ENV_RESET_REGISTRY_SHA256] = config.fixture.reset_registry_sha256


def _import_robolab_modules(config: LiveRoboLabConfig) -> dict[str, Any]:
    try:
        import robolab.constants
        from robolab.constants import set_output_dir
        import robolab.core.environments.runtime as robolab_runtime
        from robolab.core.task.conditionals import object_dropped, object_grabbed
        from robolab.core.world.world_state import get_world
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD
        import robolab.robots.droid as droid_robot
    except ImportError as exc:
        raise DroidDependencyError(
            "RoboLab modules are unavailable after AppLauncher startup"
        ) from exc

    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    robolab.constants.VERBOSE = False
    robolab.constants.DEBUG = False
    if os.environ.get("ONLINE_CORRECTION_V4_OUTPUT_DIR"):
        set_output_dir(os.environ["ONLINE_CORRECTION_V4_OUTPUT_DIR"])

    active = resolve_active_registration()
    if config.g3_contact_probe:
        droid_robot.contact_gripper = {
            **droid_robot.contact_gripper,
            "robot_all": "{ENV_REGEX_NS}/robot/.*",
        }
    if config.action_mode == "absolute_ik":
        from robolab.registrations.droid.auto_env_registrations_abs_ik import (
            auto_register_droid_abs_ik_envs as register_env,
        )
    else:
        from robolab.registrations.droid.auto_env_registrations_jointpos import (
            auto_register_droid_envs as register_env,
        )
    register_env(
        task=[active.task_module],
        cameras=WRIST_LEFT_RIGHT_HEAD,
    )
    return {
        "robolab_runtime": robolab_runtime,
        "object_dropped": object_dropped,
        "object_grabbed": object_grabbed,
        "get_world": get_world,
        "active_registration": active,
    }


def _create_live_env(config: LiveRoboLabConfig, modules: dict[str, Any]) -> LiveRoboLabEnv:
    active = modules["active_registration"]
    robolab_runtime = modules["robolab_runtime"]
    if config.output_dir is not None:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ONLINE_CORRECTION_V4_OUTPUT_DIR"] = str(config.output_dir.resolve())

    env, _env_cfg = robolab_runtime.create_env(
        active.task_class,
        device=os.environ.get("V4_DROID_DEVICE", "cuda:0"),
        seed=config.env_seed,
        num_envs=1,
        instruction_type="default",
        policy="online_correction_v4_droid",
        renderer=os.environ.get("V4_DROID_RENDERER", "realtime"),
        rendering_mode=os.environ.get("V4_DROID_RENDERING_MODE", "balanced"),
    )
    backend = LiveRoboLabBackend(
        env=env,
        config=config,
        modules=modules,
    )
    reset_proxy = _new_reset_proxy(backend=backend, config=config, modules=modules)
    live_env = LiveRoboLabEnv(
        backend=backend,
        reset_proxy=reset_proxy,
        config=config,
    )
    backend.reset_proxy = reset_proxy
    return live_env


def _new_reset_proxy(
    *,
    backend: LiveRoboLabBackend,
    config: LiveRoboLabConfig,
    modules: dict[str, Any],
) -> TwoResetAttestationProxy:
    state = ResetAttestationState(
        episode_id=config.episode_id,
        env_seed=config.env_seed,
        fixture_id=config.fixture.fixture_id,
        reset_registry_sha256=config.fixture.reset_registry_sha256,
        locked_native_control_dt_s=config.locked_native_control_dt_s,
        reset_registry_path=str(resolve_file_uri(config.fixture.reset_registry_uri, label="reset_registry")),
    )
    probe = RoboLabSettleProbe(backend=backend, modules=modules, config=config)
    registry = load_bound_reset_registry(
        registry_path=state.reset_registry_path or "",
        registry_sha256=config.fixture.reset_registry_sha256,
        required_status=(
            MODEL_BLIND_CANDIDATE_STATUS
            if isinstance(config.fixture, ResetFixtureBinding)
            else RELEASED_FOR_POLICY_STATUS
        ),
    )

    def _attestation_validator(_attestation: dict[str, Any], physical: Mapping[str, Any]) -> None:
        verify_physical_reset_against_registry(
            physical,
            registry=registry,
            env_seed=config.env_seed,
        )
        verify_neutral_horizontal_layout(physical)
        verify_measured_native_dt(
            measured_s=backend.control_dt_s,
            locked_s=config.locked_native_control_dt_s,
        )

    reset_proxy = TwoResetAttestationProxy(
        env=backend,
        probe=probe,
        state=state,
        attestation_validator=_attestation_validator,
    )
    return reset_proxy


@dataclass
class RoboLabSettleProbe:
    backend: LiveRoboLabBackend
    modules: dict[str, Any]
    config: LiveRoboLabConfig

    def hold_action(self) -> Any:
        return self.backend.hold_action_tensor()

    def sample_stability(self) -> dict[str, Any]:
        return self.backend.sample_stability_maxima()

    def physical_reset_payload(self) -> dict[str, Any]:
        return self.backend.physical_reset_payload()

    def zero_episode_length_buf(self) -> tuple[list[float], list[int]]:
        return self.backend.zero_episode_length_buf()

    def on_settle_complete(self, post_settle_obs: Any) -> None:
        self.backend.on_settle_complete(post_settle_obs)


@dataclass
class LiveRoboLabBackend:
    env: Any
    config: LiveRoboLabConfig
    modules: dict[str, Any]
    reset_proxy: TwoResetAttestationProxy | None = None
    control_tick: int = 0
    reference_displacement_m: float = 0.0
    reference_direction: tuple[float, float] = (1.0, 0.0)
    last_hold_action: tuple[float, ...] = ()
    _latest_raw_obs: Any = None
    _initial_supported_z: float = 0.0
    _reference_baseline_pose: tuple[float, ...] | None = None
    _native_dt_measured: float | None = None
    _g3_physics_tick: int = 0
    _settling_baseline_position: tuple[float, float, float] | None = None
    _settling_baseline_orientation_wxyz: tuple[float, float, float, float] | None = None
    _moved_object_mask_pixels_by_camera: dict[str, int] | None = None
    support_surface_tol_m: float = 0.015
    stability_speed_max_m_s: float = 0.02

    @property
    def control_dt_s(self) -> float:
        if self._native_dt_measured is None:
            step_dt = getattr(self.env, "step_dt", None)
            if step_dt is None:
                raise RoboLabBootstrapError("RoboLab env did not expose step_dt")
            measured = float(step_dt)
            if measured <= 0:
                raise RoboLabBootstrapError("native control dt must be positive")
            self._native_dt_measured = measured
        return self._native_dt_measured

    @property
    def physics_dt_s(self) -> float:
        value = getattr(self.env, "physics_dt", None)
        if value is None:
            simulation = getattr(self.env, "sim", None)
            getter = getattr(simulation, "get_physics_dt", None)
            value = getter() if callable(getter) else None
        if value is None:
            raise RoboLabBootstrapError("RoboLab env did not expose physics_dt")
        measured = float(value)
        if measured <= 0:
            raise RoboLabBootstrapError("physics dt must be positive")
        return measured

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed != self.config.env_seed:
            raise RoboLabBootstrapError("env_seed differs from bound episode seed")
        result = self.env.reset(seed=seed)
        obs, info = result
        self._latest_raw_obs = obs
        self.control_tick = 0
        self.reference_displacement_m = 0.0
        self._initial_supported_z = 0.0
        self._reference_baseline_pose = None
        self._g3_physics_tick = 0
        self.last_hold_action = ()
        self._anchor_reference_motion()
        return obs, info

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        tensor_action = self._action_tensor(action)
        obs, reward, terminated, truncated, info = self.env.step(tensor_action)
        self._latest_raw_obs = obs
        self.control_tick += 1
        return obs, {"reward": reward, "terminated": terminated, "truncated": truncated, **info}

    def hold_action_tensor(self) -> Any:
        import torch

        if self._latest_raw_obs is None:
            raise RoboLabBootstrapError("hold action requested before first observation")
        if self.config.action_mode == "absolute_ik":
            robot = self.env.scene["robot"]
            body_names = list(robot.data.body_names)
            if "base_link" not in body_names:
                raise RoboLabBootstrapError(
                    "absolute-IK hold requires robot base_link"
                )
            body_index = body_names.index("base_link")
            position = robot.data.body_pos_w[0, body_index].detach().to(
                self.env.device
            )
            quaternion = robot.data.body_quat_w[0, body_index].detach().to(
                self.env.device
            )
            gripper = self._latest_raw_obs["proprio_obs"]["gripper_pos"]
            gripper_value = (
                gripper.detach().to(self.env.device).reshape(-1)[0] > 0.5
            ).to(dtype=torch.float32)
            action = torch.cat((position, quaternion, gripper_value[None]))[None]
            if tuple(action.shape) != (1, ACTION_DIM):
                raise RoboLabBootstrapError(
                    f"unexpected absolute-IK hold shape: {tuple(action.shape)}"
                )
            return action
        obs = self._latest_raw_obs
        arm = obs["proprio_obs"]["arm_joint_pos"].detach().to(self.env.device)
        gripper = obs["proprio_obs"]["gripper_pos"].detach().to(self.env.device)
        if gripper.ndim == 1:
            gripper = gripper[:, None]
        action = torch.cat((arm, gripper), dim=1)
        if tuple(action.shape) != (1, ACTION_DIM):
            raise RoboLabBootstrapError(f"unexpected hold-action shape: {tuple(action.shape)}")
        return action

    def begin_g3_physics_sampling(self) -> None:
        """Latch a stationary robot command for simulator-substep G3 probes."""
        action_manager = getattr(self.env, "action_manager", None)
        if action_manager is None or not hasattr(action_manager, "process_action"):
            raise RoboLabBootstrapError(
                "RoboLab env lacks action manager for G3 physics sampling"
            )
        action_manager.process_action(self.hold_action_tensor())
        self._g3_physics_tick = 0

    def step_g3_physics_substeps(self, count: int) -> None:
        """Advance raw physics without coarsening G3's 20 ms path sampling."""
        if type(count) is not int or count <= 0:
            raise RoboLabBootstrapError("G3 physics substep count must be positive")
        action_manager = getattr(self.env, "action_manager", None)
        scene = getattr(self.env, "scene", None)
        simulation = getattr(self.env, "sim", None)
        if (
            action_manager is None
            or not hasattr(action_manager, "apply_action")
            or scene is None
            or not hasattr(scene, "write_data_to_sim")
            or not hasattr(scene, "update")
            or simulation is None
            or not hasattr(simulation, "step")
        ):
            raise RoboLabBootstrapError(
                "RoboLab env lacks raw physics hooks required by G3"
            )
        for _ in range(count):
            action_manager.apply_action()
            scene.write_data_to_sim()
            simulation.step(render=False)
            scene.update(dt=self.physics_dt_s)
            self._g3_physics_tick += 1

    def g3_contact_force_evidence(self) -> dict[str, float]:
        """Return maximum filtered force magnitudes for every live pair sensor."""
        try:
            from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
        except ImportError as exc:
            raise RoboLabBootstrapError(
                "RoboLab contact sensors are unavailable for G3"
            ) from exc
        import numpy as np

        records: dict[str, float] = {}
        sensors = get_contact_sensors(self.env.scene)
        if not sensors:
            raise RoboLabBootstrapError("G3 found no live contact sensors")
        for name, sensor in sorted(sensors.items()):
            if name.endswith("__all_objs"):
                continue
            data = getattr(sensor, "data", None)
            force = getattr(data, "force_matrix_w", None) if data is not None else None
            if force is None:
                force = getattr(data, "net_forces_w", None) if data is not None else None
            if force is None:
                raise RoboLabBootstrapError(
                    f"G3 contact sensor {name!r} exposes no force tensor"
                )
            array = np.asarray(_host_numpy(force), dtype=float)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise RoboLabBootstrapError(
                    f"G3 contact sensor {name!r} force tensor is invalid"
                )
            vector_axis = array.shape[-1] if array.ndim else 0
            if vector_axis != 3:
                raise RoboLabBootstrapError(
                    f"G3 contact sensor {name!r} force tensor lacks xyz vectors"
                )
            magnitudes = np.linalg.norm(array.reshape(-1, 3), axis=1)
            records[name] = float(np.max(magnitudes))
        return records

    def g3_scene_state(self) -> dict[str, Any]:
        """Capture privileged model-blind poses, velocities, bounds, and contacts."""
        get_world = self.modules["get_world"]
        world = get_world(self.env)
        objects: dict[str, Any] = {}
        for name in SETTLE_OBJECTS:
            pos, quat = world.get_pose(name, env_id=0)
            velocity = world.get_velocity(name, env_id=0)
            objects[name] = {
                "position_world_xyz_m": _host_numpy(pos).tolist(),
                "quaternion_world_wxyz": _host_numpy(quat).tolist(),
                "velocity_world_xyz_rad_s": _host_numpy(velocity).tolist(),
                "world_aabb_m": self.g3_world_aabb(name),
            }
        return {
            "physics_tick": self._g3_physics_tick,
            "physics_time_s": self._g3_physics_tick * self.physics_dt_s,
            "physics_dt_s": self.physics_dt_s,
            "objects": objects,
            "robot_world_aabb_m": self.g3_world_aabb("robot"),
            "table_world_aabb_m": self.g3_world_aabb("table"),
            "contact_force_n_by_pair": self.g3_contact_force_evidence(),
        }

    def capture_g3_restore_state(self) -> dict[str, Any]:
        """Capture the full dynamic scene state reused across one seed's paths."""
        get_world = self.modules["get_world"]
        world = get_world(self.env)
        objects: dict[str, Any] = {}
        for name in SETTLE_OBJECTS:
            pos, quat = world.get_pose(name, env_id=0)
            velocity = world.get_velocity(name, env_id=0)
            objects[name] = {
                "pose": tuple(
                    float(value)
                    for value in [
                        *_host_numpy(pos),
                        *_host_numpy(quat),
                    ]
                ),
                "velocity": tuple(
                    float(value) for value in _host_numpy(velocity)
                ),
            }
        robot = self.env.scene["robot"]
        return {
            "objects": objects,
            "robot_joint_position": robot.data.joint_pos.detach().clone(),
            "robot_joint_velocity": robot.data.joint_vel.detach().clone(),
        }

    def restore_g3_state(self, state: Mapping[str, Any]) -> None:
        """Restore one captured seed state before a model-blind path replay."""
        import torch

        objects = state.get("objects")
        if not isinstance(objects, Mapping) or set(objects) != set(SETTLE_OBJECTS):
            raise RoboLabBootstrapError("G3 restore state has invalid objects")
        for name in SETTLE_OBJECTS:
            row = objects[name]
            if not isinstance(row, Mapping):
                raise RoboLabBootstrapError(
                    f"G3 restore state is invalid for {name}"
                )
            pose = row.get("pose")
            velocity = row.get("velocity")
            if (
                not isinstance(pose, tuple)
                or len(pose) != 7
                or not isinstance(velocity, tuple)
                or len(velocity) != 6
            ):
                raise RoboLabBootstrapError(
                    f"G3 restore state shape differs for {name}"
                )
            asset = self.env.scene[name]
            asset.write_root_pose_to_sim(
                torch.tensor(
                    [pose], dtype=torch.float32, device=self.env.device
                )
            )
            asset.write_root_velocity_to_sim(
                torch.tensor(
                    [velocity], dtype=torch.float32, device=self.env.device
                )
            )
        joint_position = state.get("robot_joint_position")
        joint_velocity = state.get("robot_joint_velocity")
        if joint_position is None or joint_velocity is None:
            raise RoboLabBootstrapError("G3 restore state lacks robot joints")
        robot = self.env.scene["robot"]
        robot.write_joint_state_to_sim(
            joint_position.detach().to(self.env.device),
            joint_velocity.detach().to(self.env.device),
        )
        self.env.scene.write_data_to_sim()
        self.env.sim.step(render=False)
        self.env.scene.update(dt=self.physics_dt_s)
        counter = getattr(self.env, "episode_length_buf", None)
        if counter is not None and hasattr(counter, "zero_"):
            counter.zero_()
        self.control_tick = 0
        self._g3_physics_tick = 0
        self.reference_displacement_m = 0.0
        self._anchor_reference_motion()

    def g3_world_aabb(self, scene_name: str) -> dict[str, Any]:
        """Resolve a live scene asset to its USD world-aligned bounding box."""
        try:
            asset = self.env.scene[scene_name]
        except (KeyError, TypeError) as exc:
            raise RoboLabBootstrapError(
                f"G3 scene asset is unavailable: {scene_name}"
            ) from exc
        view = getattr(asset, "root_physx_view", None)
        prim_paths = getattr(view, "prim_paths", None)
        if not prim_paths:
            raise RoboLabBootstrapError(
                f"G3 scene asset {scene_name!r} exposes no concrete prim path"
            )
        prim_path = str(prim_paths[0])
        try:
            import omni.usd
            from pxr import Usd, UsdGeom
        except ImportError as exc:
            raise RoboLabBootstrapError(
                "USD bounds APIs are unavailable for G3"
            ) from exc
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            raise RoboLabBootstrapError(
                f"G3 could not resolve prim path for {scene_name!r}: {prim_path}"
            )
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = [float(value) for value in aligned.GetMin()]
        maximum = [float(value) for value in aligned.GetMax()]
        if (
            len(minimum) != 3
            or len(maximum) != 3
            or any(not value < upper for value, upper in zip(minimum, maximum))
        ):
            raise RoboLabBootstrapError(
                f"G3 world bounds are invalid for {scene_name!r}"
            )
        return {
            "min_xyz": minimum,
            "max_xyz": maximum,
            "prim_path": prim_path,
        }

    def sample_stability_maxima(self) -> dict[str, Any]:
        import numpy as np

        get_world = self.modules["get_world"]
        world = get_world(self.env)
        maxima: dict[str, Any] = {}
        for name in SETTLE_OBJECTS:
            velocity = _host_numpy(world.get_velocity(name, env_id=0))
            maxima[name] = {
                "max_linear_component_speed_m_s": float(np.max(np.abs(velocity[:3]))),
                "max_angular_component_speed_rad_s": float(np.max(np.abs(velocity[3:]))),
            }
        return maxima

    def physical_reset_payload(self) -> dict[str, Any]:
        get_world = self.modules["get_world"]
        world = get_world(self.env)
        robot_pos, robot_quat = world.get_pose("robot", env_id=0)
        robot_pos = _host_numpy(robot_pos)
        robot_quat = _host_numpy(robot_quat)
        objects: dict[str, Any] = {}
        for name in SETTLE_OBJECTS:
            pos, quat = world.get_pose(name, env_id=0)
            pos = _host_numpy(pos)
            quat_obj = _host_numpy(quat)
            objects[name] = {
                "position_world_xyz_m": pos.tolist(),
                "position_robot_xyz_m": _quat_inverse_rotate(robot_quat, pos - robot_pos).tolist(),
                "quaternion_world_wxyz": quat_obj.tolist(),
            }
        return {
            "schema_version": "v4-droid-physical-reset-v1",
            "episode_id": self.config.episode_id,
            "fixture_id": self.config.fixture.fixture_id,
            "environment_seed": self.config.env_seed,
            "objects": objects,
            "robot_position_world_xyz_m": robot_pos.tolist(),
            "robot_position_robot_xyz_m": [0.0, 0.0, 0.0],
            "robot_quaternion_world_wxyz": robot_quat.tolist(),
            "measured_native_control_dt_s": self.control_dt_s,
        }

    def camera_geometry_payload(
        self, camera_names: tuple[str, ...]
    ) -> dict[str, dict[str, Any]]:
        """Return live ROS-camera geometry for model-blind axis projection."""
        rows: dict[str, dict[str, Any]] = {}
        raw_observation = self.latest_raw_observation()
        image_obs = raw_observation.get("image_obs", {})
        for name in camera_names:
            try:
                sensor = self.env.scene[name]
            except (KeyError, TypeError):
                sensors = getattr(self.env.scene, "sensors", {})
                if name not in sensors:
                    raise RoboLabBootstrapError(
                        f"live camera sensor is unavailable: {name}"
                    )
                sensor = sensors[name]
            data = sensor.data

            def _host(value: Any) -> list[float]:
                if hasattr(value, "detach"):
                    value = value.detach().cpu().tolist()
                return [float(item) for item in value]

            center = _host(data.pos_w[0])
            quaternion = _host(data.quat_w_ros[0])
            intrinsic_raw = data.intrinsic_matrices[0]
            if hasattr(intrinsic_raw, "detach"):
                intrinsic_raw = intrinsic_raw.detach().cpu().tolist()
            intrinsic = [
                [float(item) for item in row] for row in intrinsic_raw
            ]
            frame = image_obs.get(name)
            if hasattr(frame, "detach"):
                frame = frame.detach().cpu().numpy()
            import numpy as np

            frame_array = np.asarray(frame)
            if frame_array.ndim == 4 and frame_array.shape[0] == 1:
                frame_array = frame_array[0]
            if frame_array.ndim != 3 or frame_array.shape[-1] != 3:
                raise RoboLabBootstrapError(
                    f"camera {name} has no HxWx3 frame for geometry binding"
                )
            rows[name] = {
                "camera_center_world_m": center,
                "camera_quaternion_world_wxyz_ros": quaternion,
                "intrinsic_matrix_3x3": intrinsic,
                "image_size_wh": [
                    int(frame_array.shape[1]),
                    int(frame_array.shape[0]),
                ],
            }
        return rows

    def zero_episode_length_buf(self) -> tuple[list[float], list[int]]:
        counter = getattr(self.env, "episode_length_buf", None)
        if counter is None or not hasattr(counter, "zero_"):
            raise RoboLabBootstrapError("RoboLab env lacks resettable episode_length_buf")
        before = [float(value) for value in counter.detach().cpu().tolist()]
        counter.zero_()
        after = [int(value) for value in counter.detach().cpu().tolist()]
        return before, after

    def capture_observation_bytes(self) -> bytes:
        audit = build_observation_audit_payload(
            reference_displacement_m=self.reference_displacement_m,
            camera_ids=("head", "wrist_left", "wrist_right"),
            moved_object_mask_pixels_by_camera=self._moved_object_mask_pixels_by_camera,
            extra={
                "episode_id": self.config.episode_id,
                "control_tick": self.control_tick,
                "object_state": self.object_kinematic_state().__dict__,
            },
        )
        return json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def capture_viewport_frame(self) -> Any:
        from experiments.online_correction_v4.viewport_video import (
            ViewportCapture,
            attest_viewport_capture,
            capture_from_ndarray,
        )

        recorder = getattr(self.env, "viewport_recorder", None)
        if recorder is None:
            return None

        def _read_attr(name: str) -> Any:
            value = getattr(recorder, name, None)
            if callable(value):
                return value()
            return value

        for attr, kind in (
            ("latest_frame_rgb", "raw_rgb24"),
            ("latest_rgb", "raw_rgb24"),
            ("latest_frame", "raw_bgr24"),
        ):
            frame = _read_attr(attr)
            if frame is not None:
                try:
                    import numpy as np
                except ImportError:
                    break
                if isinstance(frame, np.ndarray):
                    return capture_from_ndarray(frame, format_kind=kind)

        width = _read_attr("frame_width") or _read_attr("width")
        height = _read_attr("frame_height") or _read_attr("height")
        channels = _read_attr("channels") or 3
        pixel_format = _read_attr("pixel_format") or _read_attr("frame_format")

        for attr in ("latest_frame_bytes", "latest_encoded_frame"):
            frame = _read_attr(attr)
            if isinstance(frame, (bytes, bytearray)) and frame:
                payload = bytes(frame)
                if isinstance(pixel_format, str) and pixel_format.startswith("raw"):
                    return attest_viewport_capture(
                        payload,
                        format_kind=pixel_format,
                        width=int(width) if width is not None else None,
                        height=int(height) if height is not None else None,
                        channels=int(channels),
                    )
                if width is not None and height is not None and pixel_format in {
                    "raw_rgb24",
                    "raw_bgr24",
                }:
                    return attest_viewport_capture(
                        payload,
                        format_kind=pixel_format,
                        width=int(width),
                        height=int(height),
                        channels=int(channels),
                    )
                return attest_viewport_capture(payload, format_kind="encoded_image")
        return None

    def object_kinematic_state(self) -> ObjectKinematicState:
        get_world = self.modules["get_world"]
        object_dropped = self.modules["object_dropped"]
        object_grabbed = self.modules["object_grabbed"]
        world = get_world(self.env)
        obj_pos, _ = world.get_pose(TARGET_OBJECT, env_id=0)
        ref_pos, _ = world.get_pose(REFERENCE_OBJECT, env_id=0)
        robot_pos, _ = world.get_pose("robot", env_id=0)
        obj_pos = _host_numpy(obj_pos)
        ref_pos = _host_numpy(ref_pos)
        robot_pos = _host_numpy(robot_pos)
        sim_time = self.control_tick * self.control_dt_s
        if self._initial_supported_z == 0.0:
            self._initial_supported_z = float(obj_pos[2])
        contact = bool(object_grabbed(self.env, object=TARGET_OBJECT, env_id=0))
        detached = bool(object_dropped(self.env, object=TARGET_OBJECT, env_id=0))
        return ObjectKinematicState(
            sim_time=sim_time,
            control_tick=self.control_tick,
            object_z=float(obj_pos[2]),
            initial_supported_z=self._initial_supported_z,
            gripper_x=float(robot_pos[0]),
            gripper_y=float(robot_pos[1]),
            gripper_z=float(robot_pos[2]),
            object_x=float(obj_pos[0]),
            object_y=float(obj_pos[1]),
            object_z_pos=float(obj_pos[2]),
            contact=contact,
            detached=detached,
        )

    def set_reference_kinematic_offset(
        self, displacement_m: float, direction: tuple[float, float]
    ) -> None:
        import numpy as np
        import torch

        self.reference_displacement_m = float(displacement_m)
        self.reference_direction = direction
        if self._reference_baseline_pose is None:
            raise RoboLabBootstrapError("reference baseline pose is not anchored")
        task_left, task_front = (float(direction[0]), float(direction[1]))
        norm = float(np.linalg.norm([task_left, task_front]))
        if abs(norm - 1.0) > 1e-6:
            raise RoboLabBootstrapError(
                "reference direction must be a unit vector in task coordinates"
            )
        get_world = self.modules["get_world"]
        _robot_pos, robot_quaternion = get_world(self.env).get_pose(
            "robot", env_id=0
        )
        # Frozen DROID task frame: robot +Y is left and robot -X is front.
        offset_robot = np.asarray(
            [-task_front, task_left, 0.0], dtype=float
        ) * float(displacement_m)
        offset_world = _quat_rotate(robot_quaternion, offset_robot)
        dx = float(offset_world[0])
        dy = float(offset_world[1])
        dz = float(offset_world[2])
        x0, y0, z0, qw, qx, qy, qz = self._reference_baseline_pose
        pose = (x0 + dx, y0 + dy, z0 + dz, qw, qx, qy, qz)
        asset = self.env.scene[REFERENCE_OBJECT]
        pose_tensor = torch.tensor([pose], dtype=torch.float32, device=self.env.device)
        velocity_tensor = torch.zeros((1, 6), dtype=torch.float32, device=self.env.device)
        asset.write_root_pose_to_sim(pose_tensor)
        asset.write_root_velocity_to_sim(velocity_tensor)

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        self.last_hold_action = action

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        get_world = self.modules["get_world"]
        pos, quat = get_world(self.env).get_pose(TARGET_OBJECT, env_id=0)
        pos = _host_numpy(pos)
        quat = _host_numpy(quat)
        self._settling_baseline_position = (float(pos[0]), float(pos[1]), float(pos[2]))
        self._settling_baseline_orientation_wxyz = (
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3]),
        )

    def probe_support_contacts(self) -> tuple[str, ...] | None:
        try:
            from robolab.core.sensors.contact_sensor_utils import get_contact_sensors
        except ImportError:
            return None
        threshold_n = 0.05
        active: list[str] = []
        for name, sensor in get_contact_sensors(self.env.scene).items():
            if name.endswith("__all_objs"):
                continue
            data = getattr(sensor, "data", None)
            forces = getattr(data, "net_forces_w", None) if data is not None else None
            if forces is None:
                continue
            try:
                import torch

                force_norm = float(torch.linalg.norm(forces[0]).detach().cpu())
            except Exception:
                continue
            if force_norm >= threshold_n:
                active.append(name)
        return target_object_contact_names(active)

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        import numpy as np

        get_world = self.modules["get_world"]
        object_dropped = self.modules["object_dropped"]
        world = get_world(self.env)
        obj_pos, obj_quat = world.get_pose(TARGET_OBJECT, env_id=0)
        obj_pos = _host_numpy(obj_pos)
        obj_quat = _host_numpy(obj_quat)
        velocity = _host_numpy(world.get_velocity(TARGET_OBJECT, env_id=0))
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        detached = bool(object_dropped(self.env, object=TARGET_OBJECT, env_id=0))
        drift_m = 0.0
        orientation_drift = 0.0
        if self._settling_baseline_position is not None:
            drift_m = position_drift_m(self._settling_baseline_position, obj_pos)
        if self._settling_baseline_orientation_wxyz is not None:
            orientation_drift = geodesic_orientation_delta_rad(
                self._settling_baseline_orientation_wxyz,
                obj_quat,
            )
        sample = evaluate_horizontal_terminal_sample(
            detached=detached,
            linear_speed_m_s=linear_speed,
            angular_speed_rad_s=angular_speed,
            support_contacts=self.probe_support_contacts(),
            position_drift_m=drift_m,
            orientation_drift_rad=orientation_drift,
        )
        return sample

    def on_settle_complete(self, post_settle_obs: Any) -> None:
        self._latest_raw_obs = post_settle_obs
        self.control_tick = 0

    def latest_raw_observation(self) -> Any:
        if self._latest_raw_obs is None:
            raise RoboLabBootstrapError("raw observation is unavailable before reset")
        return self._latest_raw_obs

    def reference_position_world(self) -> tuple[float, float, float]:
        get_world = self.modules["get_world"]
        pos, _ = get_world(self.env).get_pose(REFERENCE_OBJECT, env_id=0)
        pos = _host_numpy(pos)
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def _anchor_reference_motion(self) -> None:
        get_world = self.modules["get_world"]
        pos, quat = get_world(self.env).get_pose(REFERENCE_OBJECT, env_id=0)
        pos = _host_numpy(pos)
        quat = _host_numpy(quat)
        self._reference_baseline_pose = tuple(float(v) for v in [*pos, *quat])

    def _action_tensor(self, action: Any) -> Any:
        import torch

        if action is None:
            return self.hold_action_tensor()
        if isinstance(action, torch.Tensor):
            tensor = action.detach().to(
                device=self.env.device,
                dtype=torch.float32,
            )
        else:
            tensor = torch.as_tensor(
                [list(action)],
                dtype=torch.float32,
                device=self.env.device,
            )
        if tuple(tensor.shape) != (1, ACTION_DIM):
            raise RoboLabBootstrapError(f"unexpected action shape: {tuple(tensor.shape)}")
        return tensor


@dataclass
class LiveRoboLabEnv:
    backend: LiveRoboLabBackend
    reset_proxy: TwoResetAttestationProxy
    config: LiveRoboLabConfig

    @property
    def control_dt_s(self) -> float:
        return self.backend.control_dt_s

    @property
    def reference_displacement_m(self) -> float:
        return self.backend.reference_displacement_m

    def reset_for_model_blind_g3(
        self,
        *,
        check_id: str,
        prompt_sha256: str,
        runtime_identity_sha256: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Perform a fresh attested physical reset for one zero-policy G3 check."""
        if not check_id:
            raise RoboLabBootstrapError("G3 check_id must be non-empty")
        self.config.episode_id = check_id
        proxy = _new_reset_proxy(
            backend=self.backend,
            config=self.config,
            modules=self.backend.modules,
        )
        self.reset_proxy = proxy
        self.backend.reset_proxy = proxy
        proxy.reset(seed=self.config.env_seed)
        proxy.reset(seed=self.config.env_seed)
        initial_state_sha256 = sha256_bytes(
            self.backend.capture_observation_bytes()
        )
        attestation = proxy.finalize_attestation(
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
            initial_state_sha256=initial_state_sha256,
        )
        return attestation, self.backend.physical_reset_payload()

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset_proxy.reset(seed=seed)

    def step(self, action: tuple[float, ...] | None) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.reset_proxy.step(action)

    def capture_observation_bytes(self) -> bytes:
        return self.backend.capture_observation_bytes()

    def capture_viewport_frame(self) -> Any:
        return self.backend.capture_viewport_frame()

    def object_kinematic_state(self) -> ObjectKinematicState:
        return self.backend.object_kinematic_state()

    def set_reference_kinematic_offset(
        self, displacement_m: float, direction: tuple[float, float]
    ) -> None:
        self.backend.set_reference_kinematic_offset(displacement_m, direction)

    def hold_robot_target(self, action: tuple[float, ...]) -> None:
        self.backend.hold_robot_target(action)

    def anchor_passive_settling_baseline(self, snapshot: SimulatorSnapshot | None = None) -> None:
        self.backend.anchor_passive_settling_baseline(snapshot)

    def sample_terminal_predicates(self) -> TerminalPhysicalPredicates:
        return self.backend.sample_terminal_predicates()

    def capture_policy_observation(self) -> CapturedObservation:
        native = self.backend.latest_raw_observation()
        packed = pack_policy_request(
            policy_id=self.config.policy_id,
            native_obs=native,
            instruction=self.config.prompt_text,
        )
        payload = self.capture_observation_bytes()
        return CapturedObservation(
            payload=payload,
            camera_ids=("head", "wrist_left", "wrist_right"),
            state_hash=sha256_bytes(payload),
            native_input={"raw_observation": native, "packed_request": packed},
        )

    def close(self) -> None:
        close = getattr(self.backend.env, "close", None)
        if callable(close):
            close()


def write_queue_row(
    *,
    output_dir: Path,
    episode_id: str,
    fixture_id: str,
    prompt_text: str,
    prompt_sha256: str,
    env_seed: int,
    goal: str,
) -> tuple[Path, str]:
    row = {
        "campaign": "online_correction_v4",
        "episode_id": episode_id,
        "fixture": fixture_id,
        "prompt_text": prompt_text,
        "prompt_sha256": prompt_sha256,
        "env_seed": env_seed,
        "factors": {"goal": goal},
    }
    path = output_dir / f"{episode_id}.queue_row.json"
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path, sha256_bytes(path.read_bytes())


def _host_numpy(value: Any, *, dtype: Any = float) -> Any:
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value) if dtype is None else np.asarray(value, dtype=dtype)


def _quat_inverse_rotate(quaternion: Any, vector: Any) -> Any:
    import numpy as np

    q = _host_numpy(quaternion)
    v = _host_numpy(vector)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise RoboLabBootstrapError("robot quaternion is invalid during reset verification")
    q = q / norm
    w, xyz = q[0], q[1:]
    inverse_xyz = -xyz
    return (
        2.0 * np.dot(inverse_xyz, v) * inverse_xyz
        + (w * w - np.dot(inverse_xyz, inverse_xyz)) * v
        + 2.0 * w * np.cross(inverse_xyz, v)
    )


def _quat_rotate(quaternion: Any, vector: Any) -> Any:
    import numpy as np

    q = _host_numpy(quaternion)
    v = _host_numpy(vector)
    norm = np.linalg.norm(q)
    if norm <= 0:
        raise RoboLabBootstrapError(
            "robot quaternion is invalid during task-frame motion"
        )
    q = q / norm
    w, xyz = q[0], q[1:]
    return (
        2.0 * np.dot(xyz, v) * xyz
        + (w * w - np.dot(xyz, xyz)) * v
        + 2.0 * w * np.cross(xyz, v)
    )
