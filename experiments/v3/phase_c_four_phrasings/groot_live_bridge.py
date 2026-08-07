#!/usr/bin/env python3
"""Execute one released eight-cell GR00T V3-C001 seed block.

The script consumes the hash-bound execution and live task-registration gates,
routes every prompt by exact bytes, retains initial plus every post-action
state, records every executed action and returned action chunk, writes one
viewport MP4 per cell, and emits one schema-validated behavioral JSONL row.
It refuses partial seed blocks and pre-existing raw destinations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
import traceback
from typing import Any


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--bridge-preflight", type=Path, required=True)
BOOTSTRAP.add_argument("--task-registration", type=Path, required=True)
BOOTSTRAP.add_argument("--release-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--registration-manifest", type=Path, required=True)
BOOTSTRAP.add_argument("--execution-plan", type=Path, required=True)
BOOTSTRAP.add_argument("--runner-output-root", type=Path, required=True)
BOOTSTRAP.add_argument("--launch-evidence", type=Path, required=True)
BOOTSTRAP.add_argument("--remote-host", required=True)
BOOTSTRAP.add_argument("--remote-port", type=int, default=5555)
BOOTSTRAP.add_argument("--open-loop-horizon", type=int, default=8)
BOOTSTRAP.add_argument("--instruction-controller", choices=["static"], default="static")
bootstrap, _ = BOOTSTRAP.parse_known_args()

study_root = bootstrap.study_root.resolve()
if str(study_root) not in sys.path:
    sys.path.insert(0, str(study_root))

from experiments.v3.groot_droid.adapter import validate_runtime_identity  # noqa: E402
from experiments.v3.phase_c_four_phrasings.contract import (  # noqa: E402
    EXPERIMENT_ID,
    canonical_json_bytes,
    sha256_file,
    validate_release_manifest,
)
from experiments.v3.phase_c_four_phrasings.groot_behavioral_contract import (  # noqa: E402
    MODEL_ID,
    prompt_condition,
    validate_live_output_contract,
    validate_live_task_registration,
    validate_task_sources,
)

if bootstrap.open_loop_horizon != 8:
    BOOTSTRAP.error("the frozen GR00T open-loop horizon is 8")
bridge, task_registration = validate_live_task_registration(
    bridge_preflight_path=bootstrap.bridge_preflight,
    task_registration_path=bootstrap.task_registration,
)
if sha256_file(bootstrap.execution_plan) != bridge.get("execution_plan_sha256"):
    BOOTSTRAP.error("execution plan no longer matches the seed-block preflight")
released = validate_release_manifest(
    bootstrap.release_manifest,
    model_id=MODEL_ID,
    registration_manifest_sha256=sha256_file(bootstrap.registration_manifest),
)
if released.release_manifest_sha256 != bridge.get("release_manifest_sha256"):
    BOOTSTRAP.error("release manifest no longer matches the seed-block preflight")
release_payload = json.loads(bootstrap.release_manifest.read_text())
runtime_identity_path = Path(release_payload["runtime_identity"]["path"])
runtime_identity = validate_runtime_identity(
    study_root, runtime_identity_path, check_live_repositories=True
)
if validate_task_sources(study_root) != bridge.get("task_source_sha256"):
    BOOTSTRAP.error("GR00T Phase-C task sources changed after preflight")
for cell in bridge["cells"]:
    validate_live_output_contract(cell)
if bootstrap.runner_output_root.exists() or bootstrap.launch_evidence.exists():
    BOOTSTRAP.error("refusing to overwrite retained GR00T Phase-C launch output")

import cv2  # noqa: E402,F401 -- RoboLab requires this before Isaac Lab
import numpy as np  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(parents=[BOOTSTRAP])
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.video_mode != "viewport":
    parser.error("V3-C001 GR00T requires viewport video")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("V3-C001 GR00T requires one environment and one run per cell")
if args_cli.enable_subtask:
    parser.error("progress-conditioned subtask coaching is prohibited")
if args_cli.renderer != "realtime" or args_cli.rendering_type != "balanced":
    parser.error("V3-C001 GR00T requires the released realtime balanced renderer")
args_cli.output_folder_name = str(bootstrap.runner_output_root)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.core.task.conditionals import (  # noqa: E402
    object_dropped,
    object_left_of,
    object_right_of,
)
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402

sys.path.insert(0, str(study_root / "experiments/groot_droid"))
from v2_robolab_client import V2GR00TDroidJointposClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

cells = list(bridge["cells"])
cell_by_task = {cell["task_name"]: cell for cell in cells}
if len(cell_by_task) != 8:
    raise ValueError("the GR00T Phase-C task map is not one-to-one")
auto_register_droid_envs(
    task=[cell["task_file"] for cell in cells], cameras=WRIST_LEFT_RIGHT_HEAD
)
args_cli.task = [cell["task_name"] for cell in cells]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"required retained artifact is missing or empty: {path}")
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _quat_inverse_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    if q.shape != (4,) or v.shape != (3,):
        raise ValueError("robot pose must provide a 4D quaternion and 3D vector")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("robot quaternion is invalid")
    q = q / norm
    w, xyz = q[0], q[1:]
    inverse_xyz = -xyz
    return (
        2.0 * np.dot(inverse_xyz, v) * inverse_xyz
        + (w * w - np.dot(inverse_xyz, inverse_xyz)) * v
        + 2.0 * w * np.cross(inverse_xyz, v)
    )


def _in_requested_region(sample: dict[str, Any], relation: str) -> bool:
    obj = np.asarray(sample["object_xyz"], dtype=np.float64)
    ref = np.asarray(sample["reference_xyz"], dtype=np.float64)
    forward, lateral = (obj - ref)[:2]
    distance = math.hypot(float(forward), float(lateral))
    margin = float(lateral) if relation == "left" else -float(lateral)
    return distance > 1e-8 and margin / distance >= math.cos(math.radians(45.0))


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


class StateCaptureProxy:
    """Retain the second reset and every post-action physical state."""

    def __init__(self, env: Any, cell: dict[str, Any]) -> None:
        self._env = env
        self.cell = cell
        self.outputs = validate_live_output_contract(cell)
        self.samples: list[dict[str, Any]] = []
        self.started = time.monotonic()
        self.actions_started = False
        self.reset_count = 0
        self.capture_path = Path(cell["raw_cell_directory"]) / "state_capture.json"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> dict[str, Any]:
        cube_world = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl_world = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        cube_robot = _quat_inverse_rotate_wxyz(robot_quat, cube_world - robot_pos)
        bowl_robot = _quat_inverse_rotate_wxyz(robot_quat, bowl_world - robot_pos)
        return {
            "action_step": action_step,
            "object_xyz": cube_robot.tolist(),
            "reference_xyz": bowl_robot.tolist(),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
        }

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if self.actions_started:
            raise RuntimeError("unexpected environment reset after GR00T actions began")
        result = self._env.reset(*args, **kwargs)
        self.reset_count += 1
        self.started = time.monotonic()
        self.samples = [self._sample(0)]
        return result

    def step(self, action: Any) -> Any:
        if not self.samples:
            raise RuntimeError("GR00T action attempted before the matched reset")
        if not self.actions_started:
            raw_dir = Path(self.cell["raw_cell_directory"])
            raw_dir.mkdir(parents=True, exist_ok=False)
            with self.outputs["state_trace"].open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(self.samples[0], sort_keys=True, separators=(",", ":")) + "\n")
            self.actions_started = True
        result = self._env.step(action)
        sample = self._sample(len(self.samples))
        self.samples.append(sample)
        with self.outputs["state_trace"].open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
        return result

    def write_capture(self) -> None:
        if not self.actions_started or not self.samples:
            return
        relation = self.cell["relation"]
        predicate = object_left_of if relation == "left" else object_right_of
        success = bool(predicate(
            self._env,
            object="rubiks_cube",
            reference_object="bowl",
            frame_of_reference="robot",
            mirrored=False,
            require_gripper_detached=True,
            env_id=0,
        ))
        actions_executed = len(self.samples) - 1
        initial = np.asarray(self.samples[0]["object_xyz"], dtype=np.float64)
        interaction = _first_sustained([
            float(np.linalg.norm(np.asarray(row["object_xyz"]) - initial)) >= 0.01
            for row in self.samples
        ])
        pickup = _first_sustained([
            float(row["object_xyz"][2]) - float(initial[2]) >= 0.03 for row in self.samples
        ])
        entered = any(_in_requested_region(row, relation) for row in self.samples)
        failure_stage = (
            "success" if success else
            "no_object_interaction" if interaction is None else
            "object_moved_no_verified_pickup" if pickup is None else
            "picked_never_entered_requested_region" if not entered else
            "entered_requested_region_not_released"
        )
        capture = {
            "schema_version": "vla-wam-shared-v3c-groot-state-capture-v1",
            "experiment_id": EXPERIMENT_ID,
            "registered_cell_id": self.cell["registered_cell_id"],
            "attempt_id": self.cell["registered_cell_id"].replace(":", "-") + "-attempt01",
            "environment_seed": self.cell["environment_seed"],
            "policy_seed": self.cell["sampling_seed"],
            "prompt_family": self.cell["prompt_family"],
            "requested_relation": relation,
            "prompt": self.cell["prompt"],
            "prompt_sha256": self.cell["prompt_sha256"],
            "reset_count_before_actions": self.reset_count,
            "actions_executed": actions_executed,
            "action_cap": self.cell.get("action_cap", 450),
            "right_censored": not success and actions_executed == self.cell.get("action_cap", 450),
            "requested_success": success,
            "final_detached_release": bool(self.samples[-1]["grippers_open"]),
            "frozen_failure_stage": failure_stage,
            "first_contact_step": None,
            "first_contact_unavailable_reason": (
                "RoboLab frozen GR00T adapter exposes grasp and detached-release "
                "conditionals but no verified physical contact stream"
            ),
            "wall_time_s": time.monotonic() - self.started,
            "operational_wall_time_valid": True,
            "samples": self.samples,
            "behavioral_result_valid_candidate": success or actions_executed == self.cell.get("action_cap", 450),
        }
        if self.capture_path.exists():
            raise FileExistsError(f"refusing to overwrite GR00T Phase-C capture: {self.capture_path}")
        self.capture_path.write_bytes(canonical_json_bytes(capture))

    def close(self) -> Any:
        self.write_capture()
        return self._env.close()


class PhaseCGrootClient(V2GR00TDroidJointposClient):
    """Exact-prompt client retaining action artifacts at registered paths."""

    def __init__(self, *, cell: dict[str, Any], **kwargs: Any) -> None:
        self.cell = cell
        self.outputs = validate_live_output_contract(cell, fresh=False)
        self.trace_written = False
        super().__init__(action_trace_dir=Path(cell["raw_cell_directory"]), **kwargs)

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if instruction != self.cell["prompt"]:
            raise ValueError("live GR00T instruction does not match exact registered bytes")
        if prompt_condition(instruction) != (self.cell["prompt_family"], self.cell["relation"]):
            raise ValueError("live GR00T prompt routed to the wrong registered condition")
        return super().infer(obs, instruction, env_id=env_id)

    def _write_trace(self) -> None:
        if self.trace_written or (not self.executed_actions and not self.returned_action_chunks):
            return
        if self.prompt != self.cell["prompt"]:
            raise ValueError("GR00T action trace prompt changed within the episode")
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_action_chunks).astype(np.float32, copy=False)
        modalities = {
            key: np.stack([row[key] for row in self.returned_action_modalities])
            for key in sorted(self.returned_action_modalities[0])
        }
        if self.outputs["executed_actions"].exists():
            raise FileExistsError("refusing to overwrite GR00T Phase-C actions")
        np.save(self.outputs["executed_actions"], actions, allow_pickle=False)
        chunk_path = self.outputs["executed_actions"].with_name("returned_action_chunks.npy")
        modality_path = self.outputs["executed_actions"].with_name("returned_action_modalities.npz")
        np.save(chunk_path, chunks, allow_pickle=False)
        np.savez(modality_path, **modalities)
        metadata = {
            "schema_version": "vla-wam-shared-v3c-groot-action-trace-v1",
            "registered_cell_id": self.cell["registered_cell_id"],
            "prompt": self.prompt,
            "prompt_sha256": self.cell["prompt_sha256"],
            "prompt_family": self.cell["prompt_family"],
            "relation": self.cell["relation"],
            "sampling_seed_base": self.sampling_seed_base,
            "request_sampling_seeds": self.request_sampling_seeds,
            "executed_actions": _file_record(self.outputs["executed_actions"]),
            "returned_action_chunks": _file_record(chunk_path),
            "returned_action_chunk_shape": list(chunks.shape),
            "returned_action_modalities": _file_record(modality_path),
            "returned_action_modality_shapes": {
                key: list(value.shape) for key, value in modalities.items()
            },
            "model_request_count": len(self.request_sampling_seeds),
        }
        metadata_path = self.outputs["executed_actions"].with_name("executed_actions.json")
        metadata_path.write_bytes(canonical_json_bytes(metadata))
        self.trace_written = True


_create_env = runtime.create_env
active_cell: dict[str, Any] | None = None
captures: dict[str, StateCaptureProxy] = {}
clients: dict[str, PhaseCGrootClient] = {}


def _seeded_captured_create_env(*args: Any, **kwargs: Any) -> Any:
    global active_cell
    task_name = str(args[0] if args else kwargs.get("task"))
    cell = cell_by_task.get(task_name)
    if cell is None:
        raise ValueError(f"unregistered task reached GR00T Phase-C bridge: {task_name}")
    kwargs["seed"] = bridge["seed"]
    env, env_cfg = _create_env(*args, **kwargs)
    if env_cfg.instruction != cell["prompt"]:
        raise ValueError("registered environment exposed changed prompt bytes")
    active_cell = cell
    proxy = StateCaptureProxy(env, cell)
    captures[cell["registered_cell_id"]] = proxy
    return proxy, env_cfg


runtime.create_env = _seeded_captured_create_env


def make_client(_: argparse.Namespace) -> PhaseCGrootClient:
    if active_cell is None:
        raise RuntimeError("GR00T client constructed before its registered environment")
    client = PhaseCGrootClient(
        cell=active_cell,
        remote_host=bootstrap.remote_host,
        remote_port=bootstrap.remote_port,
        open_loop_horizon=bootstrap.open_loop_horizon,
        sampling_seed_base=bridge["seed"],
    )
    clients[active_cell["registered_cell_id"]] = client
    return client


def _event_timeline(capture: dict[str, Any]) -> list[dict[str, Any]]:
    relation = capture["requested_relation"]
    samples = capture["samples"]
    initial_z = float(samples[0]["object_xyz"][2])
    pickup = _first_sustained([
        float(row["object_xyz"][2]) - initial_z >= 0.03 for row in samples
    ])
    requested = next((i for i, row in enumerate(samples) if _in_requested_region(row, relation)), None)
    opposite = "right" if relation == "left" else "left"
    opposite_step = next((i for i, row in enumerate(samples) if _in_requested_region(row, opposite)), None)
    events = [{"event": "episode_start", "action_step": 0}]
    for name, step in (
        ("verified_pickup", pickup),
        ("requested_region_entry", requested),
        ("opposite_region_entry", opposite_step),
    ):
        if step is not None:
            events.append({"event": name, "action_step": int(step)})
    events.append({"event": "episode_end", "action_step": capture["actions_executed"]})
    rank = {"episode_start": 0, "verified_pickup": 1, "requested_region_entry": 2, "opposite_region_entry": 3, "episode_end": 4}
    return sorted(events, key=lambda row: (row["action_step"], rank[row["event"]]))


def _move_video(cell: dict[str, Any], task_name: str) -> Path:
    outputs = validate_live_output_contract(cell, fresh=False)
    candidates = sorted((bootstrap.runner_output_root / task_name).glob("*_viewport.mp4"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one closed viewport video for {cell['registered_cell_id']}, found {candidates}"
        )
    target = outputs["simulator_viewport_video"]
    if target.exists():
        raise FileExistsError(f"refusing to overwrite retained viewport video: {target}")
    shutil.move(str(candidates[0]), str(target))
    capture = cv2.VideoCapture(str(target))
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
    finally:
        capture.release()
    if frames <= 0:
        raise ValueError(f"viewport video has no decodable frames: {target}")
    return target


def _build_episode(cell: dict[str, Any], video: Path) -> dict[str, Any]:
    raw_dir = Path(cell["raw_cell_directory"])
    capture = json.loads((raw_dir / "state_capture.json").read_text())
    outputs = validate_live_output_contract(cell, fresh=False)
    if not capture.get("behavioral_result_valid_candidate"):
        raise ValueError(f"partial cell cannot enter denominator: {cell['registered_cell_id']}")
    actions = np.load(outputs["executed_actions"], allow_pickle=False)
    if actions.ndim != 2 or actions.shape[0] != capture["actions_executed"] or actions.shape[1] != 8:
        raise ValueError("executed-action trace does not match captured simulator steps")
    record = {
        "schema_version": "vla-wam-shared-v3-raw-episode-v1",
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": "vla_wam_language_steerability_v3",
        "arena": "droid_robolab",
        "registered_cell_id": cell["registered_cell_id"],
        "attempt_id": capture["attempt_id"],
        "model_id": MODEL_ID,
        "pair_id": cell["seed_block_id"] + ":" + cell["prompt_family"],
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        "reset_id": f"v3c001:droid_robolab:neutral_reset:environment_seed_{cell['environment_seed']}",
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "requested_relation": cell["relation"],
        "requested_success": capture["requested_success"],
        "failure_stage": capture["frozen_failure_stage"],
        "frozen_failure_stage": capture["frozen_failure_stage"],
        "failure_taxonomy": "transport_failed",
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "measurement_frame_description": (
            "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
            "forward is object-minus-reference x and lateral is object-minus-reference y, "
            "with positive lateral denoting robot LEFT."
        ),
        "checkpoint": {
            "id": release_payload["runtime_identity"]["checkpoint"],
            "revision": release_payload["runtime_identity"]["checkpoint_revision"],
        },
        "runtime_identity": {
            "id": runtime_identity["runtime_id"],
            "sha256": sha256_file(runtime_identity_path),
        },
        "artifacts": {
            "viewport_video": _file_record(video),
            "executed_action_trace": _file_record(outputs["executed_actions"]),
            "raw_result_jsonl": {
                "path": str(outputs["behavioral_jsonl"]),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "steps": capture["samples"],
        "actions_executed": capture["actions_executed"],
        "action_cap": capture["action_cap"],
        "right_censored": capture["right_censored"],
        "first_contact_step": None,
        "first_contact_unavailable_reason": capture["first_contact_unavailable_reason"],
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": capture["operational_wall_time_valid"],
        "event_timeline": _event_timeline(capture),
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import (  # type: ignore
        derive_failure_taxonomy,
        derive_initial_state_sha256,
        derive_measurements,
        validate_behavioral_record,
    )
    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    measurements = derive_measurements(record)
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    return validate_behavioral_record(record)


def _finalize() -> dict[str, Any]:
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import write_jsonl  # type: ignore
    rows = []
    for cell in cells:
        cell_id = cell["registered_cell_id"]
        if cell_id not in captures or cell_id not in clients:
            raise ValueError(f"whole-seed execution is missing {cell_id}")
        clients[cell_id]._write_trace()
        video = _move_video(cell, cell["task_name"])
        record = _build_episode(cell, video)
        output = validate_live_output_contract(cell, fresh=False)["behavioral_jsonl"]
        jsonl_manifest = write_jsonl(output, [record])
        rows.append({
            "within_seed_execution_order": cell["within_seed_execution_order"],
            "registered_cell_id": cell_id,
            "prompt_family": cell["prompt_family"],
            "relation": cell["relation"],
            "requested_success": record["requested_success"],
            "failure_taxonomy": record["failure_taxonomy"],
            "actions_executed": record["actions_executed"],
            "initial_state_sha256": record["initial_state_sha256"],
            "measurements": record["measurements"],
            "artifacts": {
                "viewport_video": record["artifacts"]["viewport_video"],
                "executed_actions": record["artifacts"]["executed_action_trace"],
                "state_trace": _file_record(Path(cell["required_outputs"]["state_trace"])),
                "behavioral_jsonl": {
                    "path": str(output),
                    "sha256": jsonl_manifest["jsonl_sha256"],
                    "bytes": jsonl_manifest["jsonl_bytes"],
                },
            },
        })
    initial_hashes = {row["initial_state_sha256"] for row in rows}
    if len(initial_hashes) != 1:
        raise ValueError("GR00T Phase-C whole-seed cells did not share an identical reset")
    report = {
        "schema_version": "vla-wam-shared-v3c-groot-whole-seed-smoke-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": MODEL_ID,
        "seed": bridge["seed"],
        "passed": True,
        "behavioral_episode_count": 8,
        "infrastructure_episode_count": 0,
        "model_request_count": sum(len(client.request_sampling_seeds) for client in clients.values()),
        "release_manifest_sha256": sha256_file(bootstrap.release_manifest),
        "bridge_preflight_sha256": sha256_file(bootstrap.bridge_preflight),
        "task_registration_sha256": sha256_file(bootstrap.task_registration),
        "execution_plan_sha256": sha256_file(bootstrap.execution_plan),
        "runtime_identity_sha256": sha256_file(runtime_identity_path),
        "matched_initial_state_sha256": next(iter(initial_hashes)),
        "cells": rows,
    }
    bootstrap.launch_evidence.parent.mkdir(parents=True, exist_ok=True)
    bootstrap.launch_evidence.write_bytes(canonical_json_bytes(report))
    return report


def main() -> None:
    try:
        run_evaluation(args_cli, policy="groot_v3c001", client_factory=make_client)
        report = _finalize()
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        for client in clients.values():
            client._write_trace()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[GR00T V3-C001 live bridge] infrastructure failure: {error}")
        traceback.print_exc()
        raise
