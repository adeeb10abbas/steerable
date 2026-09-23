"""Native joint-position execution with the qualified geometry measurement path.

AppLauncher must already be running in the simulator process. Model workers
must use a separately qualified simulator transport, not import Isaac on B200.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
import uuid

import numpy as np

from .adapters import AdapterError
from .fixtures import (
    FixtureCandidate, NEUTRAL_TOLERANCE_M, RESET_ANGLE_TOLERANCE_DEGREES,
    RESET_POSITION_TOLERANCE_M, pose_error,
)
from .policy_observations import native_policy_observation
from .robolab_lat_qualification import RoboLabLatEnvironment
from .runtime import D1_ROBOLAB_CLIENT_COMMIT, _required_env, _verify_git_checkout
from .scoring import relation_m


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class JointPositionBinding:
    source_root: Path
    robolab_root: Path
    assets_manifest: Path
    assets_manifest_sha256: str
    cells: Mapping[str, Mapping[str, Any]]

    @classmethod
    def load(cls) -> "JointPositionBinding":
        path = Path(_required_env("SGW01_ENV_BINDING")).resolve()
        if not path.is_file() or _sha256(path) != _required_env("SGW01_ENV_BINDING_SHA256"):
            raise AdapterError("joint-position environment binding is absent or hash-mismatched")
        value = json.loads(path.read_text())
        source_root = Path(value["source_root"]).resolve()
        robolab_root = Path(value["robolab_root"]).resolve()
        _verify_git_checkout(source_root, value["source_commit"], "SGW simulator")
        if value["robolab_commit"] != D1_ROBOLAB_CLIENT_COMMIT:
            raise AdapterError("joint-position RoboLab revision differs from measured runtime")
        _verify_git_checkout(robolab_root, D1_ROBOLAB_CLIENT_COMMIT, "RoboLab simulator")
        manifest = Path(value["assets_manifest"]).resolve()
        expected = value["assets_manifest_sha256"]
        if not manifest.is_file() or _sha256(manifest) != expected:
            raise AdapterError("joint-position asset manifest is absent or hash-mismatched")
        assets = json.loads(manifest.read_text())
        for record in (assets["scene"], *assets["assets"]):
            asset = Path(record["path"])
            if (not asset.is_file() or asset.stat().st_size != record["bytes"]
                    or _sha256(asset) != record["sha256"]):
                raise AdapterError(f"joint-position asset payload changed: {asset}")
        if not isinstance(value["cells"], Mapping) or not value["cells"]:
            raise AdapterError("joint-position binding requires explicit cells")
        return cls(source_root, robolab_root, manifest, expected, value["cells"])

    def cell(self, cell: Any) -> tuple[Mapping[str, Any], FixtureCandidate]:
        row = getattr(cell, "row", cell)
        record = self.cells.get(row["cell_id"])
        if not isinstance(record, Mapping) or row.get("status") != "RELEASED":
            raise AdapterError("cell has no released joint-position environment binding")
        for key in ("family", "layout_id", "fixture_sha256", "prompt_sha256"):
            if not record.get(key) or record[key] != row.get(key):
                raise AdapterError(f"joint-position cell binding differs for {key}")
        if hashlib.sha256(row["prompt"].encode()).hexdigest() != record["prompt_sha256"]:
            raise AdapterError("joint-position binding prompt differs from released static prompt")
        path = Path(record["candidate_path"])
        if not path.is_file() or _sha256(path) != record["candidate_file_sha256"]:
            raise AdapterError("joint-position candidate file hash mismatch")
        candidate = FixtureCandidate.from_json(json.loads(path.read_text()))
        if candidate.family != row["family"] or candidate.asset_manifest_sha256 != self.assets_manifest_sha256:
            raise AdapterError("joint-position candidate differs from bound family/assets")
        if type(record.get("scene_seed")) is not int:
            raise AdapterError("joint-position cell requires an explicit integer scene seed")
        scene = candidate.metadata.get("native_scene")
        if candidate.family != "LAT":
            if not isinstance(scene, Mapping) or not isinstance(scene.get("object_names"), list):
                raise AdapterError("HEIGHT/DIST joint-position cell lacks measured native scene")
            files = record.get("native_scene_files")
            if not isinstance(files, list) or not any(item.get("path") == scene.get("asset") for item in files):
                raise AdapterError("joint-position overlay lacks bound native scene files")
            for item in files:
                asset = Path(item["path"])
                if not asset.is_file() or _sha256(asset) != item["sha256"]:
                    raise AdapterError(f"joint-position native scene file changed: {asset}")
        return record, candidate


@dataclass(frozen=True)
class PolicyReset:
    snapshot: Mapping[str, Any]
    receipt: Mapping[str, Any]


class JointPositionEnvironment:
    def __init__(self, env: Any, *, candidate: FixtureCandidate, cell_id: str, evidence_root: Path) -> None:
        from isaaclab.envs import mdp

        body = env.cfg.actions.body
        if (not isinstance(body, mdp.JointPositionActionCfg) or body.use_default_offset is not False
                or body.asset_name != "robot" or env.action_manager.total_action_dim != 8):
            raise AdapterError("SGW production execution requires native absolute 8D joint-position control")
        if not np.isclose(float(env.step_dt), 1 / 15, rtol=0, atol=1e-8):
            raise AdapterError("joint-position environment must expose the frozen 15 Hz step_dt")
        names = candidate.metadata.get("native_scene", {}).get(
            "object_names", ["rubiks_cube", "banana", "bowl", "table"],
        )
        # The physical predicate is support, not landing on a particular goal
        # shelf. Observe every imported non-gripper object pair separately.
        supports = tuple(f"rubiks_cube__{name}" for name in names if name != "rubiks_cube")
        self._measurements = RoboLabLatEnvironment(
            env, candidate, evidence_root, support_sensor_names=supports,
        )
        self._candidate, self._cell_id = candidate, cell_id
        self._last_snapshot = None
        self._closed = False

    def reset(self) -> PolicyReset:
        if self._closed:
            raise AdapterError("joint-position environment is closed")
        reset = self._measurements.reset()
        self._last_snapshot = reset.snapshot
        roots = reset.snapshot.reset_root_poses
        if roots is None or set(roots) != set(self._candidate.object_poses):
            raise AdapterError("joint-position reset lacks measured actor-root inventory")
        errors = {}
        for name, expected in self._candidate.object_poses.items():
            position, angle = pose_error(roots[name], expected)
            if position > RESET_POSITION_TOLERANCE_M or angle > RESET_ANGLE_TOLERANCE_DEGREES:
                raise AdapterError(f"joint-position {name} reset exceeds 3 mm / 2 degree tolerance")
            errors[name] = {"position_error_m": position, "angle_error_degrees": angle}
        state = self.snapshot()
        if abs(relation_m(self._candidate.family, state["cube_xyz_m"], state["bowl_xyz_m"],
                          state["plate_xyz_m"])) > NEUTRAL_TOLERANCE_M:
            raise AdapterError("joint-position measured reset is not neutral within 5 mm")
        return PolicyReset(state, {
            **reset.receipt, "reset_id": f"{self._cell_id}:{uuid.uuid4()}",
            "native_action_mode": "joint_position", "actor_root_reset_errors": errors,
            "candidate_fingerprint": self._candidate.fingerprint(),
        })

    def step(self, action: Any) -> dict[str, Any]:
        if self._closed or self._last_snapshot is None:
            raise AdapterError("joint-position action requires an open reset environment")
        if self._measurements._steps >= 450 or self._last_snapshot.termination_reason:
            raise AdapterError("joint-position action exceeds the episode boundary")
        array = np.asarray(action)
        if array.shape != (8,) or not np.isfinite(array).all():
            raise AdapterError("joint-position action must be a finite 8-vector")
        self._last_snapshot = self._measurements.step(array.reshape(1, 8))
        terminal = self._last_snapshot.termination_reason
        return {"safety_terminated": terminal is not None, "termination_reason": terminal}

    def snapshot(self) -> dict[str, Any]:
        if self._last_snapshot is None:
            raise AdapterError("scoring state requested before physical reset")
        return {
            **self._last_snapshot.scoring_state(self._measurements._steps),
            "sim_time": self._last_snapshot.simulated_time_s,
            "raw_snapshot": asdict(self._last_snapshot),
        }

    def render_viewport(self) -> np.ndarray:
        return self._measurements.render_viewport()

    def policy_observation(self) -> Mapping[str, Any]:
        if self._closed or self._measurements._observation is None:
            raise AdapterError("policy observation requires an open reset environment")
        return native_policy_observation(self._measurements._observation)

    def close(self) -> None:
        if not self._closed:
            self._measurements.close()
            self._closed = True


def create_environment(*, cell: Any, evidence_root: Path) -> JointPositionEnvironment:
    """Simulator-process factory; call only after AppLauncher and release gates."""
    binding = JointPositionBinding.load()
    record, candidate = binding.cell(cell)
    if not Path(__file__).resolve().is_relative_to(binding.source_root):
        raise AdapterError("joint-position factory import is outside the bound study checkout")
    import robolab
    import robolab.constants
    from robolab.constants import set_output_dir
    from robolab.core.environments.runtime import create_env
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
    from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

    if not Path(robolab.__file__).resolve().is_relative_to(binding.robolab_root):
        raise AdapterError("effective RoboLab import is outside the bound native checkout")
    row = getattr(cell, "row", cell)
    payload = {"candidate": candidate.task_payload(), "prompt": row["prompt"]}
    raw = json.dumps(payload, sort_keys=True, allow_nan=False)
    os.environ["SGW01_JOINTPOS_CELL_JSON"] = raw
    os.environ["SGW01_JOINTPOS_CELL_SHA256"] = hashlib.sha256(raw.encode()).hexdigest()
    native = evidence_root / "native"
    native.mkdir(parents=True, exist_ok=False)
    set_output_dir(str(native))
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
    robolab.constants.RECORD_IMAGE_DATA = False
    auto_register_droid_envs(
        task=[str(Path(__file__).with_name("sgw_jointpos_task.py"))], cameras=WRIST_LEFT_RIGHT_HEAD,
    )
    env, _ = create_env(
        "SGWJointPositionTask", device=_required_env("SGW01_SIMULATOR_DEVICE"),
        seed=record["scene_seed"], num_envs=1, instruction_type="default",
        policy="sgw01_static_policy", renderer="realtime", rendering_mode="balanced",
    )
    try:
        return JointPositionEnvironment(env, candidate=candidate, cell_id=row["cell_id"], evidence_root=evidence_root)
    except Exception:
        env.close()
        raise
