"""Pinned RoboLab bridge factories for the measured HEIGHT/DIST overlays.

The bridge deliberately reuses the model-blind Abs-IK environment/controller
contract.  It is not a policy adapter: a learned policy never sees the
scripted actions, native scoring state, or support geometry.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

from .robolab_lat_qualification import RoboLabLatEnvironment, RoboLabLatScriptedController
from .simulator_bridge import Environment, SimulatorBridgeError
from .task_definitions import RoboLabTaskDefinition


class RoboLabFamilyBridge:
    def __init__(self, *, study_root: Path, device: str, renderer: str, rendering_type: str) -> None:
        self._study_root = Path(study_root).resolve()
        self._device = device
        if renderer != "realtime" or rendering_type != "balanced":
            raise SimulatorBridgeError("HEIGHT/DIST qualification requires realtime/balanced RTX")

    def create_environment(self, task: RoboLabTaskDefinition, seed: int) -> Environment:
        if task.candidate.family not in {"HEIGHT", "DIST"}:
            raise SimulatorBridgeError("family bridge only accepts HEIGHT or DIST candidates")
        if not task.candidate.metadata.get("goal_supports"):
            raise SimulatorBridgeError("family candidate lacks measured released goal supports")
        from robolab.core.environments.runtime import create_env
        from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

        payload = json.dumps(task.bridge_config(), sort_keys=True, separators=(",", ":"))
        os.environ["SGW_FAMILY_CANDIDATE_JSON"] = payload
        os.environ["SGW_FAMILY_CANDIDATE_SHA256"] = hashlib.sha256(payload.encode()).hexdigest()
        task_path = self._study_root / "experiments/workshops/spatial_grounding_v1/family_qualification_task.py"
        if not task_path.is_file():
            raise SimulatorBridgeError("HEIGHT/DIST task overlay is missing")
        auto_register_droid_abs_ik_envs(task=[str(task_path)], cameras=WRIST_LEFT_RIGHT_HEAD)
        env, _ = create_env(
            "SGWFamilyQualificationTask", device=self._device, seed=seed, num_envs=1,
            instruction_type="default", policy="sgw_01_model_blind_family_controller",
            renderer="realtime", rendering_mode="balanced",
        )
        # Parent LAT evidence support added an optional evidence_root constructor
        # argument.  Preserve compatibility with both pinned bridge revisions.
        if len(inspect.signature(RoboLabLatEnvironment).parameters) == 3:
            return RoboLabLatEnvironment(env, task.candidate, Path(os.environ["SGW_QUALIFICATION_EVIDENCE_ROOT"]))
        return RoboLabLatEnvironment(env, task.candidate)


def create_bridge(
    *, robolab_root: Path, assets_manifest: Path, device: str, renderer: str, rendering_type: str, **_: Any
) -> RoboLabFamilyBridge:
    if not Path(assets_manifest).is_file():
        raise SimulatorBridgeError("verified actual asset manifest is required")
    return RoboLabFamilyBridge(
        study_root=Path(__file__).resolve().parents[3],
        device=device,
        renderer=renderer,
        rendering_type=rendering_type,
    )


def create_controller(**_: Any) -> RoboLabLatScriptedController:
    """Return the existing Abs-IK scripted controller, never a learned adapter."""

    return RoboLabLatScriptedController()
