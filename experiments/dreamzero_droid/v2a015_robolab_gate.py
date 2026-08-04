#!/usr/bin/env python3
"""Run one frozen V2-A015 DreamZero s=2 LEFT/RIGHT DROID seed pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

import cv2  # noqa: F401 -- RoboLab requires this before Isaac Lab
from isaaclab.app import AppLauncher


AMENDMENT_ID = "V2-A015"
ARM_ID = "dreamzero_action_cfg_s2"
LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
ACTION_CFG_STYLE_SCALE = 2.0
VIDEO_CFG_SCALE = 5.0
BASELINE_RESULT_ARTIFACT = (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "dreamzero_droid_direct_gate.json"
)
BASELINE_RESULT_SHA256 = (
    "4c76cdc3ca9eaf227d21d160199408f22e1b3dd7a71176a5a5dbe22223714461"
)
FIXED_GATE_SCHEMA = (
    "vla-wam-shared-v2-dreamzero-v2a015-fixed-observation-probe-v1"
)
SERVER_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-server-contract-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--amendment", type=Path, required=True)
parser.add_argument("--fixed-observation-gate", type=Path, required=True)
parser.add_argument("--server-contract", type=Path, required=True)
parser.add_argument("--future-root", type=Path, required=True)
parser.add_argument("--pair-manifest", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed", type=int, required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument(
    "--simulator-lane",
    choices=["raytrace-rtxpro6000-ali"],
    required=True,
)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, required=True)
parser.add_argument("--action-cfg-style-scale", type=float, required=True)
parser.add_argument("--open-loop-horizon", type=int, default=8)
parser.add_argument("--instruction-controller", choices=["static"], default="static")
parser.add_argument(
    "--condition",
    choices=["left", "right", "both"],
    default="both",
    help="Run one frozen condition or the complete pair after an infra partial.",
)

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.remote_port == 5000:
    parser.error("V2-A015 prohibits requests to the pre-existing port 5000")
if args_cli.action_cfg_style_scale != ACTION_CFG_STYLE_SCALE:
    parser.error("The authorized DreamZero V2-A015 behavioral arm is exactly s=2")
if args_cli.open_loop_horizon != 8:
    parser.error("The frozen DreamZero V2-A015 gate requires --open-loop-horizon 8")
if args_cli.video_mode != "viewport":
    parser.error("The frozen DreamZero V2-A015 gate requires --video-mode viewport")
if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("The authorized environment seeds are exactly 8300, 8301, 8302")
if args_cli.sampling_seed != args_cli.environment_seed:
    parser.error("Environment and sampling seed labels must match within a DROID pair")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("Each frozen task cell requires one environment and one run")
if args_cli.device != "cuda:0":
    parser.error("The ali-owned RTX simulator lane must be exposed as --device cuda:0")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")

amendment_path = args_cli.amendment.resolve()
fixed_gate_path = args_cli.fixed_observation_gate.resolve()
server_contract_path = args_cli.server_contract.resolve()
future_root = args_cli.future_root.resolve()
pair_manifest_path = args_cli.pair_manifest.resolve()
for required_path in (amendment_path, fixed_gate_path, server_contract_path):
    if not required_path.is_file():
        parser.error(f"Required V2-A015 provenance file is absent: {required_path}")
if not future_root.is_dir():
    parser.error(f"V2-A015 server future root is absent: {future_root}")
if pair_manifest_path.exists():
    parser.error(f"Refusing to overwrite V2-A015 pair manifest: {pair_manifest_path}")

amendment = json.loads(amendment_path.read_text())
if amendment.get("amendment_id") != AMENDMENT_ID:
    parser.error("The supplied amendment is not V2-A015")
arms = {arm.get("arm_id"): arm for arm in amendment.get("arms", [])}
arm = arms.get(ARM_ID, {})
expected_arm = {
    "model_id": "dreamzero_droid_action_cfg",
    "checkpoint": "GEAR-Dreams/DreamZero-DROID",
    "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942",
    "source_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
    "action_guidance": ACTION_CFG_STYLE_SCALE,
    "baseline_action_guidance_equivalent": 1.0,
    "video_guidance": VIDEO_CFG_SCALE,
    "runtime_num_inference_steps": 16,
    "dit_cache": True,
    "evaluated_dit_steps": 8,
    "action_chunk_shape": [24, 8],
    "executed_open_loop_horizon": 8,
    "behavioral_episode_count": 6,
}
for key, expected in expected_arm.items():
    if arm.get(key) != expected:
        parser.error(
            f"V2-A015 DreamZero arm mismatch for {key}: "
            f"expected={expected!r}, observed={arm.get(key)!r}"
        )
grid = amendment.get("behavioral_grid", {})
if grid.get("prompts") != {"left": LEFT, "right": RIGHT}:
    parser.error("V2-A015 amendment prompt bytes changed")
if grid.get("environment_seeds") != [8300, 8301, 8302]:
    parser.error("V2-A015 amendment environment seeds changed")
if grid.get("sampling_seed_labels") != [8300, 8301, 8302]:
    parser.error("V2-A015 amendment sampling seeds changed")
if grid.get("prompt_controller") != "episode_static":
    parser.error("V2-A015 amendment no longer requires episode-static prompts")
if any(
    (
        grid.get("oracle_actions") != 0,
        grid.get("subtask_coach") is not False,
        grid.get("prompt_switching") is not False,
        grid.get("progress_conditioned_language") is not False,
        grid.get("simulator_video_required") is not True,
        grid.get("executed_action_trace_required") is not True,
        grid.get("all_exposed_futures_retained") is not True,
    )
):
    parser.error("V2-A015 behavioral safety/retention contract changed")
baseline = amendment.get("known_result_disclosure", {}).get(
    "dreamzero_baseline", {}
)
if (
    baseline.get("artifact") != BASELINE_RESULT_ARTIFACT
    or baseline.get("sha256") != BASELINE_RESULT_SHA256
    or baseline.get("action_guidance_equivalent") != 1.0
    or baseline.get("video_guidance") != VIDEO_CFG_SCALE
):
    parser.error("V2-A015 amendment does not bind the preserved DreamZero baseline")

fixed_gate = json.loads(fixed_gate_path.read_text())
expected_fixed_gate = {
    "schema_version": FIXED_GATE_SCHEMA,
    "amendment_id": AMENDMENT_ID,
    "status": "passed",
    "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
    "video_cfg_scale": VIDEO_CFG_SCALE,
    "sampling_seed_label": 8300,
    "internal_gate_passed": True,
    "comparison_gate_passed": True,
    "release_gate_passed": True,
}
for key, expected in expected_fixed_gate.items():
    if fixed_gate.get(key) != expected:
        parser.error(
            f"Fixed-observation release gate mismatch for {key}: "
            f"expected={expected!r}, observed={fixed_gate.get(key)!r}"
        )
expected_records = {
    "left_a": (LEFT, "left", "primary"),
    "left_b": (LEFT, "left", "exact_repeat"),
    "right": (RIGHT, "right", "primary"),
}
records = fixed_gate.get("records", {})
if set(records) != set(expected_records):
    parser.error("V2-A015 fixed gate does not contain the exact three requests")
for label, (prompt, relation, role) in expected_records.items():
    record = records[label]
    if (
        record.get("prompt") != prompt
        or record.get("requested_relation") != relation
        or record.get("condition_role") != role
    ):
        parser.error(f"V2-A015 fixed-gate record changed for {label}")
metrics = fixed_gate.get("metrics", {})
for key in (
    "all_actions_finite_shape_24x8",
    "all_latents_finite",
    "left_exact_repeat_action_array_equal",
    "left_exact_repeat_latent_tensor_equal",
):
    if metrics.get(key) is not True:
        parser.error(f"V2-A015 fixed-observation gate did not pass {key}")
if not float(metrics.get("left_vs_right_action_rms", 0.0)) > 0.0:
    parser.error("V2-A015 fixed-observation gate has no LEFT/RIGHT action response")
comparison = fixed_gate.get("comparison", {})
if (
    comparison.get("status") != "passed"
    or comparison.get("reference_action_cfg_style_scale") != 1.0
):
    parser.error("V2-A015 s=2 fixed gate is not bound to the passed s=1 comparison")

contract = json.loads(server_contract_path.read_text())
expected_contract = {
    "schema_version": SERVER_SCHEMA,
    "amendment_id": AMENDMENT_ID,
    "official_repository_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
    "port": args_cli.remote_port,
    "world_size": 2,
    "official_noise_seed": 1140,
    "enable_dit_cache": True,
    "runtime_num_inference_steps": 16,
    "evaluated_dit_steps_with_cache": 8,
    "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
    "video_cfg_scale": VIDEO_CFG_SCALE,
    "future_root": str(future_root),
}
for key, expected in expected_contract.items():
    if contract.get(key) != expected:
        parser.error(
            f"V2-A015 server contract mismatch for {key}: "
            f"expected={expected!r}, observed={contract.get(key)!r}"
        )
fixed_contract = fixed_gate.get("server_contract", {})
if (
    Path(fixed_contract.get("path", "")).resolve() != server_contract_path
    or fixed_contract.get("sha256") != _sha256(server_contract_path)
    or fixed_gate.get("future_root") != str(future_root)
):
    parser.error("V2-A015 fixed gate and live server contract are not the same run")

left_task = (
    args_cli.study_root
    / "experiments/dreamzero_droid/robolab_v2_tasks/"
    "rubiks_cube_left_of_bowl_matched.py"
)
right_task = (
    args_cli.study_root
    / "experiments/dreamzero_droid/robolab_v2_tasks/"
    "rubiks_cube_right_of_bowl_matched.py"
)
for task_path, exact_prompt in ((left_task, LEFT), (right_task, RIGHT)):
    if not task_path.is_file():
        parser.error(f"Missing frozen task file: {task_path}")
    if exact_prompt not in task_path.read_text():
        parser.error(f"Frozen task no longer contains its exact prompt: {task_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
import robolab.core.environments.runtime as runtime  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_jointpos import (  # noqa: E402
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT  # noqa: E402

sys.path.insert(0, str(args_cli.study_root / "experiments/dreamzero_droid"))
from v2a015_robolab_client import V2A015DreamZeroDroidClient  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

task_paths = {
    "left": [str(left_task)],
    "right": [str(right_task)],
    "both": [str(left_task), str(right_task)],
}
task_names = {
    "left": ["RubiksCubeLeftOfBowlMatchedTask"],
    "right": ["RubiksCubeRightOfBowlMatchedTask"],
    "both": [
        "RubiksCubeLeftOfBowlMatchedTask",
        "RubiksCubeRightOfBowlMatchedTask",
    ],
}
auto_register_droid_envs(
    task=task_paths[args_cli.condition], cameras=WRIST_LEFT_RIGHT
)
args_cli.task = task_names[args_cli.condition]

_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = args_cli.environment_seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V2A015DreamZeroDroidClient:
    return V2A015DreamZeroDroidClient(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        environment_seed=args_cli.environment_seed,
        sampling_seed_label=args_cli.sampling_seed,
        action_trace_dir=args_cli.action_trace_dir,
        amendment_path=amendment_path,
        fixed_observation_gate_path=fixed_gate_path,
        server_contract_path=server_contract_path,
        future_root=future_root,
    )


def _file_record(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expected completed V2-A015 evidence file: {path}")
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


def _one_file(task_dir: Path, pattern: str) -> Path:
    candidates = sorted(task_dir.glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one {pattern!r} in {task_dir}, found "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0]


def _write_pair_manifest() -> None:
    output_folder_name = str(args_cli.output_folder_name)
    required_tokens = ("v2a015", "dreamzero", "s2", str(args_cli.environment_seed))
    if any(token not in output_folder_name.lower() for token in required_tokens):
        raise ValueError(
            "--output-folder-name must visibly identify V2-A015, DreamZero, s2, "
            f"and seed {args_cli.environment_seed}: {output_folder_name!r}"
        )
    simulator_root = (Path.cwd() / "output" / output_folder_name).resolve()
    relations = {
        "left": (LEFT, "RubiksCubeLeftOfBowlMatchedTask"),
        "right": (RIGHT, "RubiksCubeRightOfBowlMatchedTask"),
    }
    selected = (
        tuple(relations)
        if args_cli.condition == "both"
        else (args_cli.condition,)
    )
    cells: list[dict[str, object]] = []
    for relation in selected:
        prompt, task_name = relations[relation]
        trace_path = (
            args_cli.action_trace_dir
            / f"seed{args_cli.sampling_seed}_{relation}_executed_actions.json"
        ).resolve()
        trace = json.loads(trace_path.read_text())
        required_trace = {
            "schema_version": (
                "vla-wam-shared-v2-dreamzero-v2a015-action-trace-v1"
            ),
            "amendment_id": AMENDMENT_ID,
            "arm_id": ARM_ID,
            "environment_seed": args_cli.environment_seed,
            "sampling_seed_label": args_cli.sampling_seed,
            "prompt": prompt,
            "requested_relation": relation,
            "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
            "video_cfg_scale": VIDEO_CFG_SCALE,
        }
        for key, expected in required_trace.items():
            if trace.get(key) != expected:
                raise ValueError(
                    f"Completed {relation} trace mismatch for {key}: "
                    f"expected={expected!r}, observed={trace.get(key)!r}"
                )
        future_manifest = trace.get("future_manifest", {})
        future_manifest_path = Path(future_manifest.get("path", ""))
        if (
            not future_manifest_path.is_file()
            or future_manifest.get("sha256") != _sha256(future_manifest_path)
        ):
            raise ValueError(f"Completed {relation} trace has invalid future binding")

        task_dir = simulator_root / task_name
        cells.append(
            {
                "cell_id": (
                    f"dreamzero_action_cfg_s2_seed{args_cli.environment_seed}_{relation}"
                ),
                "environment_seed": args_cli.environment_seed,
                "sampling_seed": args_cli.sampling_seed,
                "effective_official_model_noise_seed": 1140,
                "requested_relation": relation,
                "prompt": prompt,
                "prompt_family": "direct_command",
                "prompt_controller": "episode_static",
                "oracle_actions": 0,
                "dynamic_prompt_switches": 0,
                "open_loop_execution_horizon": 8,
                "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
                "video_cfg_scale": VIDEO_CFG_SCALE,
                "simulator_gpu_lane": args_cli.simulator_lane,
                "simulator_task_dir": str(task_dir),
                "simulator_artifacts": {
                    "environment_config": _file_record(task_dir / "env_cfg.json"),
                    "rollout_hdf5": _file_record(task_dir / "run_0.hdf5"),
                    "episode_log": _file_record(task_dir / "log_0_env0.json"),
                    "viewport_video": _file_record(
                        _one_file(task_dir, "*_viewport.mp4")
                    ),
                },
                "action_trace_metadata": _file_record(trace_path),
                "executed_actions": trace["executed_actions"],
                "returned_raw_chunks": trace["returned_raw_chunks"],
                "returned_executable_chunks": trace[
                    "returned_executable_chunks"
                ],
                "future_manifest": future_manifest,
            }
        )

    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pair_manifest = {
        "schema_version": (
            "vla-wam-shared-v2-dreamzero-v2a015-pair-collection-v1"
        ),
        "status": "complete_behavioral_pair_candidate",
        "amendment_id": AMENDMENT_ID,
        "arm_id": ARM_ID,
        "model_id": "dreamzero_droid_action_cfg",
        "checkpoint": "GEAR-Dreams/DreamZero-DROID",
        "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942",
        "official_repository_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
        "pair_id": f"seed{args_cli.environment_seed}",
        "condition": args_cli.condition,
        "environment_seed": args_cli.environment_seed,
        "sampling_seed": args_cli.sampling_seed,
        "effective_official_model_noise_seed": 1140,
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "baseline_action_cfg_equivalent": 1.0,
        "video_cfg_scale": VIDEO_CFG_SCALE,
        "output_folder_name": output_folder_name,
        "simulator_output_root": str(simulator_root),
        "simulator_gpu_lane": args_cli.simulator_lane,
        "amendment": _file_record(amendment_path),
        "fixed_observation_release_gate": _file_record(fixed_gate_path),
        "server_contract": _file_record(server_contract_path),
        "future_root": str(future_root),
        "cells": cells,
        "claim_boundary": (
            "This manifest records completed raw candidates for one frozen V2-A015 "
            "seed pair. Final behavioral validity and success are assigned only by "
            "the hash-bearing compiler; infrastructure-invalid attempts stay separate."
        ),
    }
    pair_manifest_path.write_text(
        json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    # Keep the released RoboLab integration key. The derived client binds every
    # trace to the exact V2-A015 s=2 intervention and passed release gate.
    run_evaluation(args_cli, policy="dreamzero_v2", client_factory=make_client)
    _write_pair_manifest()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[DreamZero V2-A015 s=2] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
