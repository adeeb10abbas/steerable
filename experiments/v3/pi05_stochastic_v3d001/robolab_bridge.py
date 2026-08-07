#!/usr/bin/env python3
"""Execute one released V3-D001 π0.5 cell in RoboLab."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any


BOOT = argparse.ArgumentParser(add_help=False)
BOOT.add_argument("--study-root", type=Path, required=True)
BOOT.add_argument("--release-manifest", type=Path, required=True)
BOOT.add_argument("--runtime-identity", type=Path, required=True)
BOOT.add_argument("--phase-a-release-gate", type=Path, required=True)
BOOT.add_argument("--cell-id", required=True)
BOOT.add_argument("--attempt-id", required=True)
BOOT.add_argument("--state-capture-dir", type=Path, required=True)
BOOT.add_argument("--action-trace-dir", type=Path, required=True)
BOOT.add_argument("--simulator-export", type=Path, required=True)
BOOT.add_argument("--output-dir", type=Path, required=True)
BOOT.add_argument("--remote-host", required=True)
BOOT.add_argument("--remote-port", type=int, default=8001)
BOOT.add_argument("--lane-pod-uid", required=True)
BOOT.add_argument("--lane-gpu-uuid", required=True)
BOOT.add_argument("--open-loop-horizon", type=int, default=15)
BOOT.add_argument("--instruction-controller", choices=["static"], default="static")
boot, _ = BOOT.parse_known_args()

study_root = boot.study_root.resolve()
sys.path.insert(0, str(study_root))

from experiments.v3.pi05_stochastic_v3d001.contract import (  # noqa: E402
    ACTION_CAP, ACTION_CHUNK_STEPS, ACTION_DIM, ACTION_SPACE, ARENA, MODEL_ID,
    PHASE, QUEUE_SHA256, RELEASE_MANIFEST_SHA256, STUDY_ID, SUCCESS_PREDICATE_ID, TASKS,
    load_release, sha256_file, validate_runtime,
)

release = load_release(study_root, boot.release_manifest)
if sha256_file(boot.release_manifest.resolve()) != RELEASE_MANIFEST_SHA256:
    BOOT.error("release manifest digest changed")
cell = release.cell(boot.cell_id)
runtime = validate_runtime(study_root, boot.runtime_identity, boot.phase_a_release_gate)
if boot.open_loop_horizon != ACTION_CHUNK_STEPS:
    BOOT.error("V3-D001 π0.5 open-loop horizon is exactly 15")
visible_gpu_uuids = {
    row.strip() for row in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
}
if boot.lane_gpu_uuid not in visible_gpu_uuids:
    BOOT.error("declared V3-D001 lane GPU UUID is not visible")
for path in (boot.simulator_export, boot.simulator_export.with_name("bridge_failure.json")):
    if path.exists():
        BOOT.error(f"refusing to overwrite retained V3-D001 evidence: {path}")
for path in (boot.state_capture_dir, boot.action_trace_dir, boot.output_dir):
    if path.exists():
        BOOT.error(f"refusing to reuse V3-D001 output directory: {path}")

import cv2  # noqa: E402,F401
import numpy as np  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(parents=[BOOT])
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402
add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
if args_cli.video_mode != "viewport" or args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("V3-D001 requires one environment, one run, and viewport video")
if args_cli.enable_subtask or args_cli.instruction_type != "default":
    parser.error("V3-D001 permits only the frozen static direct command")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as robolab_runtime  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.task.conditionals import (  # noqa: E402
    object_dropped, object_grabbed, object_left_of, object_right_of,
)
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs  # noqa: E402
from experiments.v3.pi05_stochastic_v3d001.client import V3D001Pi05Client  # noqa: E402
from tools.vla_wam_v3_episode_schema import MEASUREMENT_FRAME_ID, derive_initial_state_sha256  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False
set_output_dir(str(boot.output_dir.resolve()))
task_root = study_root / "experiments/groot_droid/robolab_v2_tasks"
task_file = task_root / f"rubiks_cube_{cell.relation}_of_bowl_matched.py"
auto_register_droid_envs(task=[str(task_file)])
args_cli.task = [TASKS[cell.relation]]


def _inverse_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64)
    v = np.asarray(vector, dtype=np.float64)
    q = q / np.linalg.norm(q)
    w, xyz = q[0], q[1:]
    inverse = -xyz
    return 2 * np.dot(inverse, v) * inverse + (w*w - np.dot(inverse, inverse))*v + 2*w*np.cross(inverse, v)


def _cone(sample: dict[str, Any], relation: str) -> bool:
    delta = np.asarray(sample["object_xyz"], dtype=float) - np.asarray(sample["reference_xyz"], dtype=float)
    radius = math.hypot(float(delta[0]), float(delta[1]))
    margin = float(delta[1]) if relation == "left" else -float(delta[1])
    return radius > 1e-8 and margin / radius >= math.cos(math.radians(45.0))


class StateCapture:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.samples: list[dict[str, Any]] = []
        self.reset_attestations: list[dict[str, Any]] = []
        self.reset_count = 0
        self.started = time.monotonic()
        self.written = False
        stem = cell.cell_id.replace(":", "__")
        self.stem = stem
        self.partial: Path | None = None
        self.capture = boot.state_capture_dir / f"{stem}.capture.json"

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _sample(self, action_step: int) -> dict[str, Any]:
        cube = self._env.scene["rubiks_cube"].data.root_pos_w[0].detach().cpu().numpy()
        bowl = self._env.scene["bowl"].data.root_pos_w[0].detach().cpu().numpy()
        robot = self._env.scene["robot"].data
        robot_pos = robot.root_pos_w[0].detach().cpu().numpy()
        robot_quat = robot.root_quat_w[0].detach().cpu().numpy()
        return {
            "action_step": action_step,
            "object_xyz": _inverse_rotate(robot_quat, cube - robot_pos).tolist(),
            "reference_xyz": _inverse_rotate(robot_quat, bowl - robot_pos).tolist(),
            "grippers_open": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
            "object_grabbed": bool(object_grabbed(self._env, object="rubiks_cube", env_id=0)),
        }

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        if len(self.samples) > 1:
            raise RuntimeError("V3-D001 forbids a reset after any executed action")
        if self.capture.exists():
            raise FileExistsError("refusing to overwrite completed V3-D001 state evidence")
        if self.samples:
            self._write_reset_attestation()
        result = self._env.reset(*args, **kwargs)
        self.samples = [self._sample(0)]
        self.started = time.monotonic()
        boot.state_capture_dir.mkdir(parents=True, exist_ok=True)
        self.partial = boot.state_capture_dir / f"{self.stem}.reset{self.reset_count:02d}.states.partial.jsonl"
        if self.partial.exists():
            raise FileExistsError("refusing to overwrite V3-D001 reset state evidence")
        self.partial.write_text(json.dumps(self.samples[0], sort_keys=True) + "\n", encoding="utf-8")
        initial = derive_initial_state_sha256({"measurement_frame": MEASUREMENT_FRAME_ID, "steps": self.samples})
        if initial != cell.row["source_phase_a_initial_state_sha256"]:
            raise RuntimeError("V3-D001 reset differs from its exact Phase-A condition")
        self.reset_count += 1
        self._write_reset_attestation()
        return result

    def _write_reset_attestation(self) -> None:
        index = self.reset_count - 1
        if index < 0 or not self.samples or self.partial is None:
            raise RuntimeError("cannot attest an absent V3-D001 reset")
        if len(self.samples) != 1:
            raise RuntimeError("only pre-action V3-D001 resets may be attested here")
        path = boot.state_capture_dir / f"{self.stem}.reset{index:02d}.attestation.json"
        if any(row["path"] == str(path.resolve()) for row in self.reset_attestations):
            return
        if path.exists():
            raise FileExistsError("refusing to overwrite V3-D001 reset attestation")
        initial = derive_initial_state_sha256({"measurement_frame": MEASUREMENT_FRAME_ID, "steps": self.samples})
        if initial != cell.row["source_phase_a_initial_state_sha256"]:
            raise RuntimeError("V3-D001 pre-action reset attestation changed")
        value = {
            "schema_version": "vla-wam-shared-v3d001-pi05-reset-attestation-v1",
            "registered_cell_id": cell.cell_id, "attempt_id": boot.attempt_id,
            "pre_action_reset_index": index, "actions_executed_before_next_reset": 0,
            "initial_state_sha256": initial, "sample": self.samples[0],
            "partial_state_stream": str(self.partial.resolve()),
        }
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.reset_attestations.append({"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size})

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        sample = self._sample(len(self.samples))
        self.samples.append(sample)
        if self.partial is None:
            raise RuntimeError("V3-D001 step occurred before a retained reset")
        with self.partial.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        return result

    def write(self) -> Path | None:
        if self.written or not self.samples:
            return None
        predicate = object_left_of if cell.relation == "left" else object_right_of
        success = bool(predicate(
            self._env, object="rubiks_cube", reference_object="bowl",
            frame_of_reference="robot", mirrored=False,
            require_gripper_detached=True, env_id=0,
        ))
        actions = len(self.samples) - 1
        complete = success or actions == ACTION_CAP
        requested = [_cone(sample, cell.relation) for sample in self.samples]
        if len(self.samples) == 1:
            self._write_reset_attestation()
        value = {
            "schema_version": "vla-wam-shared-v3d001-pi05-state-capture-v1",
            "registered_cell_id": cell.cell_id,
            "matched_stochastic_block_id": cell.block_id,
            "environment_seed": cell.environment_seed,
            "shared_policy_sampling_seed_index": cell.sampling_index,
            "policy_sampling_seed_base": cell.sampling_seed_base,
            "requested_relation": cell.relation,
            "prompt": cell.row["prompt"],
            "requested_success": success,
            "actions_executed": actions,
            "action_cap": ACTION_CAP,
            "right_censored": (not success and actions == ACTION_CAP),
            "final_detached_release": bool(object_dropped(self._env, object="rubiks_cube", env_id=0)),
            "first_contact_step": None,
            "first_contact_unavailable_reason": "The pinned RoboLab runtime exposes grasp but no verified contact stream; grasp is not substituted for contact.",
            "grasp_step": next((index for index, sample in enumerate(self.samples) if sample["object_grabbed"]), None),
            "cone_entry_step": next((index for index, inside in enumerate(requested) if inside), None),
            "wall_time_s": time.monotonic() - self.started,
            "operational_wall_time_valid": True,
            "samples": self.samples,
            "behavioral_result_valid_candidate": complete,
            "partial_attempt_reason": None if complete else "episode ended before success or 450 actions",
            "pre_action_reset_count": self.reset_count,
            "pre_action_reset_attestations": self.reset_attestations,
        }
        self.capture.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.written = True
        return self.capture

    def close(self) -> Any:
        self.write()
        return self._env.close()


create_env = robolab_runtime.create_env
captures: list[StateCapture] = []
clients: list[V3D001Pi05Client] = []


def captured_create_env(*args: Any, **kwargs: Any) -> Any:
    kwargs["seed"] = cell.environment_seed
    env, cfg = create_env(*args, **kwargs)
    proxy = StateCapture(env)
    captures.append(proxy)
    return proxy, cfg


robolab_runtime.create_env = captured_create_env


def make_client(_: argparse.Namespace) -> V3D001Pi05Client:
    value = V3D001Pi05Client(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        cell=cell,
        trace_dir=boot.action_trace_dir,
        release_fingerprint=release.fingerprint(cell),
    )
    clients.append(value)
    return value


def _one_video() -> Path:
    videos = [path for path in boot.output_dir.rglob("*.mp4") if path.is_file() and path.stat().st_size > 0]
    if len(videos) != 1:
        raise RuntimeError(f"V3-D001 requires exactly one viewport video, found {videos}")
    return videos[0].resolve()


def write_export() -> None:
    if len(captures) != 1 or len(clients) != 1 or clients[0].trace_path is None:
        raise RuntimeError("V3-D001 requires exactly one completed capture/client/trace")
    capture = json.loads(captures[0].capture.read_text(encoding="utf-8"))
    trace = json.loads(clients[0].trace_path.read_text(encoding="utf-8"))
    if capture["behavioral_result_valid_candidate"] is not True:
        raise RuntimeError("partial V3-D001 attempt cannot enter the behavioral denominator")
    value = {
        "schema_version": "vla-wam-shared-v3d001-pi05-simulator-export-v1",
        "study_id": STUDY_ID,
        "registration_id": "V3-D001",
        "phase": PHASE,
        "arena": ARENA,
        "model_id": MODEL_ID,
        "registered_cell_id": cell.cell_id,
        "attempt_id": boot.attempt_id,
        "matched_stochastic_block_id": cell.block_id,
        "nested_condition_id": cell.row["nested_condition_id"],
        "environment_seed": cell.environment_seed,
        "shared_policy_sampling_seed_index": cell.sampling_index,
        "policy_sampling_seed_base": cell.sampling_seed_base,
        "per_request_sampling_seed_rule": cell.row["per_request_sampling_seed_rule"],
        "requested_relation": cell.relation,
        "prompt": cell.row["prompt"],
        "prompt_mode": "static_episode_prompt",
        "success_predicate_id": SUCCESS_PREDICATE_ID,
        "release_manifest_sha256": RELEASE_MANIFEST_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "cell_sha256": cell.row["cell_sha256"],
        "release_fingerprint_sha256": release.fingerprint(cell),
        "source_phase_a_runtime_identity_sha256": cell.row["source_phase_a_runtime_identity_sha256"],
        "source_phase_a_initial_state_sha256": cell.row["source_phase_a_initial_state_sha256"],
        "runtime_identity_sha256": sha256_file(boot.runtime_identity.resolve()),
        "lane_pod_uid": boot.lane_pod_uid,
        "lane_gpu_uuid": boot.lane_gpu_uuid,
        "action_space": ACTION_SPACE,
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "steps": capture["samples"],
        "actions_executed": capture["actions_executed"],
        "requested_success": capture["requested_success"],
        "right_censored": capture["right_censored"],
        "final_detached_release": capture["final_detached_release"],
        "first_contact_step": None,
        "first_contact_unavailable_reason": capture["first_contact_unavailable_reason"],
        "request_sampling_seeds": trace["request_sampling_seeds"],
        "executed_action_trace": trace["executed_actions"],
        "returned_action_chunks": trace["returned_action_chunks"],
        "action_trace_metadata_path": str(clients[0].trace_path),
        "state_capture_path": str(captures[0].capture.resolve()),
        "pre_action_reset_attestations": capture["pre_action_reset_attestations"],
        "viewport_video_path": str(_one_video()),
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": True,
        "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
    }
    boot.simulator_export.parent.mkdir(parents=True, exist_ok=True)
    boot.simulator_export.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    failure: BaseException | None = None
    try:
        try:
            run_evaluation(args_cli, policy="pi05_v2a010_current", client_factory=make_client)
        except BaseException as exc:
            failure = exc
        finally:
            for client in clients:
                client.write_trace()
            for capture in captures:
                capture.write()
        if failure is not None:
            raise failure
        write_export()
    except BaseException as exc:
        target = boot.simulator_export.with_name("bridge_failure.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_text(json.dumps({
                "schema_version": "vla-wam-shared-v3d001-infrastructure-failure-v1",
                "registered_cell_id": cell.cell_id,
                "denominator_eligible": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
