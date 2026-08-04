#!/usr/bin/env python3
"""Run one frozen V2-A015 Cosmos3 Nano g=1 LEFT/RIGHT DROID pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

import cv2  # noqa: F401 -- RoboLab requires this before Isaac Lab
from isaaclab.app import AppLauncher

AMENDMENT_ID = "V2-A015"
ARM_ID = "cosmos3_nano_no_cfg_g1"
MODEL_ID = "cosmos3_nano_policy_droid"
CHECKPOINT_ID = "nvidia/Cosmos3-Nano-Policy-DROID"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
OFFICIAL_SOURCE_COMMIT = "411d25b2e35bc441126f48c44a4b93e1c0564274"
SIMULATOR_SOURCE_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
GUIDANCE = 1.0
BASELINE_GUIDANCE = 3.0
NUM_STEPS = 4
SHIFT = 5.0
BASELINE_RESULT_ARTIFACT = (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "cosmos3_nano_policy_droid_direct_gate.json"
)
BASELINE_RESULT_SHA256 = (
    "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expected completed V2-A015 evidence file: {path}")
    return {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}


parser = argparse.ArgumentParser()
parser.add_argument("--study-root", type=Path, required=True)
parser.add_argument("--amendment", type=Path, required=True)
parser.add_argument("--environment-seed", type=int, required=True)
parser.add_argument("--sampling-seed-base", type=int, required=True)
parser.add_argument("--action-trace-dir", type=Path, required=True)
parser.add_argument("--future-trace-dir", type=Path, required=True)
parser.add_argument("--fixed-observation-gate", type=Path, required=True)
parser.add_argument("--simulator-output-root", type=Path, required=True)
parser.add_argument("--pair-manifest", type=Path, required=True)
parser.add_argument("--remote-host", required=True)
parser.add_argument("--remote-port", type=int, default=18021)
parser.add_argument("--open-loop-horizon", type=int, default=32)
parser.add_argument("--instruction-controller", choices=["static"], default="static")

from robolab.eval.runner import add_common_eval_args, run_evaluation

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

if args_cli.open_loop_horizon != 32:
    parser.error("The frozen V2-A015 Nano DROID gate requires --open-loop-horizon 32")
if args_cli.video_mode != "viewport":
    parser.error("The frozen V2-A015 Nano DROID gate requires --video-mode viewport")
if args_cli.environment_seed not in {8300, 8301, 8302}:
    parser.error("The authorized environment seeds are exactly 8300, 8301, 8302")
if args_cli.sampling_seed_base != args_cli.environment_seed:
    parser.error("Environment and sampling seed integers must match within a DROID pair")
if args_cli.num_envs != 1 or args_cli.num_runs != 1:
    parser.error("Each frozen task cell requires one environment and one run")
if args_cli.enable_subtask:
    parser.error("Pass --disable-subtask; progress-conditioned coaching is forbidden")

amendment_path = args_cli.amendment.resolve()
fixed_gate_path = args_cli.fixed_observation_gate.resolve()
simulator_output_root = args_cli.simulator_output_root.resolve()
pair_manifest_path = args_cli.pair_manifest.resolve()
baseline_result_path = (args_cli.study_root / BASELINE_RESULT_ARTIFACT).resolve()
for required_path in (amendment_path, fixed_gate_path, baseline_result_path):
    if not required_path.is_file():
        parser.error(f"Required V2-A015 provenance file is absent: {required_path}")
if _sha256(baseline_result_path) != BASELINE_RESULT_SHA256:
    parser.error("The local preserved Cosmos g=3 baseline result hash changed")
if pair_manifest_path.exists():
    parser.error(f"Refusing to overwrite V2-A015 pair manifest: {pair_manifest_path}")
expected_output_folder = f"v2a015_cosmos3_nano_g1_seed{args_cli.environment_seed}"
if args_cli.output_folder_name != expected_output_folder:
    parser.error(f"Use frozen --output-folder-name {expected_output_folder!r}")
if simulator_output_root.name != expected_output_folder:
    parser.error(
        "--simulator-output-root basename must equal the frozen output folder: "
        f"expected={expected_output_folder!r}, observed={simulator_output_root.name!r}"
    )

amendment = json.loads(amendment_path.read_text())
if amendment.get("amendment_id") != AMENDMENT_ID:
    parser.error("The supplied amendment is not V2-A015")
arms = {arm.get("arm_id"): arm for arm in amendment.get("arms", [])}
arm = arms.get(ARM_ID, {})
expected_arm = {
    "model_id": MODEL_ID,
    "checkpoint": CHECKPOINT_ID,
    "checkpoint_revision": CHECKPOINT_REVISION,
    "source_commit": OFFICIAL_SOURCE_COMMIT,
    "guidance": GUIDANCE,
    "baseline_guidance": BASELINE_GUIDANCE,
    "num_steps": NUM_STEPS,
    "shift": SHIFT,
    "action_chunk_shape": [32, 8],
    "future_contract": "decoded 33-frame RGB future for every policy request",
    "behavioral_episode_count": 6,
}
for key, value in expected_arm.items():
    if arm.get(key) != value:
        parser.error(
            f"V2-A015 Cosmos arm mismatch for {key}: "
            f"expected={value!r}, observed={arm.get(key)!r}"
        )
grid = amendment.get("behavioral_grid", {})
if grid.get("prompts") != {"left": LEFT, "right": RIGHT}:
    parser.error("V2-A015 amendment prompt bytes changed")
if grid.get("environment_seeds") != [8300, 8301, 8302]:
    parser.error("V2-A015 amendment environment seeds changed")
if grid.get("sampling_seed_labels") != [8300, 8301, 8302]:
    parser.error("V2-A015 amendment sampling seeds changed")
if grid.get("requested_relations") != ["left", "right"]:
    parser.error("V2-A015 amendment requested relations changed")
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
    "cosmos3_nano_baseline", {}
)
if (
    baseline.get("artifact") != BASELINE_RESULT_ARTIFACT
    or baseline.get("sha256") != BASELINE_RESULT_SHA256
    or baseline.get("guidance") != BASELINE_GUIDANCE
):
    parser.error("V2-A015 amendment does not bind the preserved Cosmos g=3 baseline")

fixed_gate = json.loads(fixed_gate_path.read_text())
expected_fixed_gate = {
    "schema_version": (
        "vla-wam-shared-v2-cosmos3-nano-policy-droid-v2a015-g1-"
        "fixed-observation-v1"
    ),
    "status": "passed",
    "amendment_id": AMENDMENT_ID,
    "arm_id": ARM_ID,
    "guidance": GUIDANCE,
    "baseline_guidance": BASELINE_GUIDANCE,
}
for key, value in expected_fixed_gate.items():
    if fixed_gate.get(key) != value:
        parser.error(
            f"Fixed-observation release gate mismatch for {key}: "
            f"expected={value!r}, observed={fixed_gate.get(key)!r}"
        )
fixed_metrics = fixed_gate.get("metrics", {})
for key in (
    "left_repeat_action_bit_identical",
    "left_repeat_future_bit_identical",
    "left_right_action_distinct",
    "left_right_future_distinct",
):
    if fixed_metrics.get(key) is not True:
        parser.error(f"Fixed-observation release gate did not pass {key}")
if fixed_gate.get("baseline_result") != {
    "artifact": (
        "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "cosmos3_nano_policy_droid_direct_gate.json"
    ),
    "sha256": "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93",
    "reported_result": "LEFT 3/3; RIGHT 3/3; 3/3 aligned endpoint pairs",
}:
    parser.error("Fixed-observation release gate does not bind the frozen g=3 baseline")
if [record.get("condition") for record in fixed_gate.get("records", [])] != [
    "left",
    "left_exact_repeat",
    "right",
]:
    parser.error("Fixed-observation release gate does not contain the exact three requests")
if [record.get("prompt") for record in fixed_gate["records"]] != [LEFT, LEFT, RIGHT]:
    parser.error("Fixed-observation release gate prompt bytes changed")
if fixed_gate.get("amendment_sha256") != _sha256(amendment_path):
    parser.error("Fixed-observation release gate is not bound to this amendment file")

left_task = (
    args_cli.study_root
    / "experiments/groot_droid/robolab_v2_tasks/"
    "rubiks_cube_left_of_bowl_matched.py"
)
right_task = (
    args_cli.study_root
    / "experiments/groot_droid/robolab_v2_tasks/"
    "rubiks_cube_right_of_bowl_matched.py"
)
for task_path, exact_prompt in ((left_task, LEFT), (right_task, RIGHT)):
    if not task_path.is_file():
        parser.error(f"Missing frozen task file: {task_path}")
    if exact_prompt not in task_path.read_text():
        parser.error(f"Frozen task no longer contains its exact prompt: {task_path}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants
from robolab.core.environments import runtime
from robolab.registrations.droid.auto_env_registrations_jointpos import (
    auto_register_droid_envs,
)
from robolab.registrations.droid.camera_presets import (
    WRIST_LEFT_RIGHT_HEAD,
)

sys.path.insert(0, str(args_cli.study_root / "experiments/cosmos"))
from v2a015_nano_robolab_client import V2A015NanoCosmos3Client

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
robolab.constants.RECORD_IMAGE_DATA = False
robolab.constants.VERBOSE = False
robolab.constants.DEBUG = False

auto_register_droid_envs(task=[str(left_task), str(right_task)], cameras=WRIST_LEFT_RIGHT_HEAD)
args_cli.task = ["RubiksCubeLeftOfBowlMatchedTask", "RubiksCubeRightOfBowlMatchedTask"]

_create_env = runtime.create_env


def _seeded_create_env(*args, **kwargs):
    kwargs["seed"] = args_cli.environment_seed
    return _create_env(*args, **kwargs)


runtime.create_env = _seeded_create_env


def make_client(_: argparse.Namespace) -> V2A015NanoCosmos3Client:
    return V2A015NanoCosmos3Client(
        remote_host=args_cli.remote_host,
        remote_port=args_cli.remote_port,
        environment_seed=args_cli.environment_seed,
        sampling_seed_base=args_cli.sampling_seed_base,
        action_trace_dir=args_cli.action_trace_dir,
        future_trace_dir=args_cli.future_trace_dir,
        amendment_path=amendment_path,
        fixed_observation_gate_path=fixed_gate_path,
    )


def _validated_trace_file(
    entry: dict,
    *,
    expected_path: Path,
    path_key: str,
    sha256_key: str,
    label: str,
) -> dict[str, object]:
    expected_path = expected_path.resolve()
    observed_path = Path(entry.get(path_key, "")).resolve()
    if observed_path != expected_path:
        raise ValueError(
            f"{label} path mismatch: expected={expected_path}, observed={observed_path}"
        )
    record = _file_record(expected_path)
    if entry.get(sha256_key) != record["sha256"]:
        raise ValueError(f"{label} hash does not match the completed file")
    return record


def _cell_manifest(
    *,
    relation: str,
    prompt: str,
    task_name: str,
    viewport_filename: str,
) -> dict[str, object]:
    seed = args_cli.environment_seed
    stem = f"seed{seed}_{relation}"
    trace_path = (args_cli.action_trace_dir / f"{stem}_executed_actions.json").resolve()
    trace = json.loads(trace_path.read_text())
    expected_trace = {
        "schema_version": (
            "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-"
            "action-future-trace-v1"
        ),
        "amendment_id": AMENDMENT_ID,
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "environment_seed": seed,
        "sampling_seed_base": seed,
        "prompt": prompt,
        "requested_relation": relation,
        "guidance": GUIDANCE,
        "baseline_guidance": BASELINE_GUIDANCE,
        "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
        "baseline_result_sha256": BASELINE_RESULT_SHA256,
    }
    for key, value in expected_trace.items():
        if trace.get(key) != value:
            raise ValueError(
                f"Completed {relation} trace mismatch for {key}: "
                f"expected={value!r}, observed={trace.get(key)!r}"
            )
    if trace.get("amendment") != {
        "path": str(amendment_path),
        "sha256": _sha256(amendment_path),
    }:
        raise ValueError(f"Completed {relation} trace has the wrong amendment binding")
    if trace.get("fixed_observation_release_gate") != {
        "path": str(fixed_gate_path),
        "sha256": _sha256(fixed_gate_path),
    }:
        raise ValueError(f"Completed {relation} trace has the wrong release-gate binding")

    executed_entry = trace.get("executed_actions", {})
    executed_path = (args_cli.action_trace_dir / f"{stem}_executed_actions.npy").resolve()
    executed_record = _validated_trace_file(
        executed_entry,
        expected_path=executed_path,
        path_key="path",
        sha256_key="sha256",
        label=f"{relation} executed actions",
    )
    executed_shape = executed_entry.get("shape")
    if (
        not isinstance(executed_shape, list)
        or len(executed_shape) != 2
        or executed_shape[0] != executed_entry.get("count")
        or executed_shape[1] != 8
        or executed_entry.get("dtype") != "float32"
    ):
        raise ValueError(f"Completed {relation} executed-action contract changed")

    requests = trace.get("requests", [])
    if not requests:
        raise ValueError(f"Completed {relation} cell retained no model requests")
    retained_requests: list[dict[str, object]] = []
    for index, request in enumerate(requests):
        expected_request = {
            "request_index": index,
            "requested_sampling_seed": seed,
            "server_sampling_seed": seed,
            "environment_seed": seed,
            "prompt": prompt,
            "requested_relation": relation,
            "model_id": MODEL_ID,
            "checkpoint": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_source_commit": OFFICIAL_SOURCE_COMMIT,
            "amendment_id": AMENDMENT_ID,
            "arm_id": ARM_ID,
            "guidance": GUIDANCE,
            "baseline_guidance": BASELINE_GUIDANCE,
            "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
            "baseline_result_sha256": BASELINE_RESULT_SHA256,
            "amendment_sha256": _sha256(amendment_path),
            "fixed_observation_gate_sha256": _sha256(fixed_gate_path),
            "action_shape": [32, 8],
        }
        for key, value in expected_request.items():
            if request.get(key) != value:
                raise ValueError(
                    f"{relation} request {index} mismatch for {key}: "
                    f"expected={value!r}, observed={request.get(key)!r}"
                )
        future_shape = request.get("future_shape")
        if (
            not isinstance(future_shape, list)
            or len(future_shape) != 4
            or future_shape[0] != 33
            or future_shape[-1] != 3
        ):
            raise ValueError(f"{relation} request {index} is not a 33-frame RGB future")
        request_stem = f"{stem}_request{index:03d}"
        action_path = (
            args_cli.action_trace_dir / f"{request_stem}_returned_action.npy"
        ).resolve()
        future_path = (
            args_cli.future_trace_dir / f"{request_stem}_future.npy"
        ).resolve()
        returned_action = _validated_trace_file(
            request,
            expected_path=action_path,
            path_key="action_path",
            sha256_key="action_sha256",
            label=f"{relation} request {index} returned action",
        )
        decoded_future = _validated_trace_file(
            request,
            expected_path=future_path,
            path_key="future_path",
            sha256_key="future_sha256",
            label=f"{relation} request {index} decoded future",
        )
        retained_requests.append(
            {
                "request_index": index,
                "sampling_seed": seed,
                "returned_action": {
                    **returned_action,
                    "shape": [32, 8],
                    "dtype": "float32",
                },
                "decoded_future": {
                    **decoded_future,
                    "shape": future_shape,
                    "dtype": "uint8",
                    "frame_count": 33,
                },
            }
        )

    task_dir = (simulator_output_root / task_name).resolve()
    simulator_artifacts = {
        "environment_config": _file_record(task_dir / "env_cfg.json"),
        "rollout_hdf5": _file_record(task_dir / "run_0.hdf5"),
        "episode_log": _file_record(task_dir / "log_0_env0.json"),
        "viewport_video": _file_record(task_dir / viewport_filename),
    }
    return {
        "cell_id": f"cosmos3_nano_no_cfg_g1_seed{seed}_{relation}",
        "amendment_id": AMENDMENT_ID,
        "amendment_sha256": _sha256(amendment_path),
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_repository_commit": OFFICIAL_SOURCE_COMMIT,
        "fixed_observation_gate_sha256": _sha256(fixed_gate_path),
        "environment_seed": seed,
        "sampling_seed": seed,
        "requested_relation": relation,
        "prompt": prompt,
        "prompt_family": "direct_command",
        "prompt_controller": "episode_static",
        "oracle_actions": 0,
        "dynamic_prompt_switches": 0,
        "open_loop_execution_horizon": 32,
        "guidance": GUIDANCE,
        "baseline_guidance": BASELINE_GUIDANCE,
        "num_steps": NUM_STEPS,
        "shift": SHIFT,
        "action_chunk_shape": [32, 8],
        "future_contract": "decoded 33-frame RGB future for every policy request",
        "simulator_task": task_name,
        "simulator_task_dir": str(task_dir),
        "simulator_artifacts": simulator_artifacts,
        "action_future_trace_metadata": _file_record(trace_path),
        "executed_actions": {
            **executed_record,
            "count": executed_entry["count"],
            "shape": executed_shape,
            "dtype": executed_entry["dtype"],
        },
        "model_requests": retained_requests,
        "decoded_future_count": len(retained_requests),
    }


def _write_pair_manifest() -> None:
    cells = [
        _cell_manifest(
            relation="left",
            prompt=LEFT,
            task_name="RubiksCubeLeftOfBowlMatchedTask",
            viewport_filename=(
                "Put_the_Rubiks_cube_to_the_left_of_the_bowl_0_viewport.mp4"
            ),
        ),
        _cell_manifest(
            relation="right",
            prompt=RIGHT,
            task_name="RubiksCubeRightOfBowlMatchedTask",
            viewport_filename=(
                "Put_the_Rubiks_cube_to_the_right_of_the_bowl_0_viewport.mp4"
            ),
        ),
    ]
    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pair_manifest = {
        "schema_version": (
            "vla-wam-shared-v2-cosmos3-nano-v2a015-g1-pair-collection-v1"
        ),
        "status": "complete_behavioral_pair_candidate",
        "amendment_id": AMENDMENT_ID,
        "arm_id": ARM_ID,
        "model_id": MODEL_ID,
        "checkpoint": CHECKPOINT_ID,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "official_repository_commit": OFFICIAL_SOURCE_COMMIT,
        "simulator_repository_commit": SIMULATOR_SOURCE_COMMIT,
        "pair_id": f"seed{args_cli.environment_seed}",
        "environment_seed": args_cli.environment_seed,
        "sampling_seed": args_cli.sampling_seed_base,
        "guidance": GUIDANCE,
        "baseline_guidance": BASELINE_GUIDANCE,
        "num_steps": NUM_STEPS,
        "shift": SHIFT,
        "action_chunk_shape": [32, 8],
        "future_contract": "decoded 33-frame RGB future for every policy request",
        "output_folder_name": args_cli.output_folder_name,
        "simulator_output_root": str(simulator_output_root),
        "amendment": _file_record(amendment_path),
        "fixed_observation_release_gate": _file_record(fixed_gate_path),
        "baseline_result": _file_record(baseline_result_path),
        "adapter_files": {
            "runner": _file_record(Path(__file__)),
            "client": _file_record(
                args_cli.study_root
                / "experiments/cosmos/v2a015_nano_robolab_client.py"
            ),
        },
        "cells": cells,
        "pair_checks": {
            "cell_count": 2,
            "relations": [cell["requested_relation"] for cell in cells],
            "prompts": [cell["prompt"] for cell in cells],
            "all_prompts_episode_static": True,
            "all_exposed_futures_retained": True,
        },
        "claim_boundary": (
            "This manifest records completed raw candidates for one frozen V2-A015 "
            "Cosmos3 Nano g=1 seed pair. Final behavioral validity and success are "
            "assigned only by the hash-bearing compiler; infrastructure-invalid "
            "attempts remain outside the denominator."
        ),
    }
    pair_manifest_path.write_text(
        json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(pair_manifest, indent=2, sort_keys=True))


def main() -> None:
    # Keep the official V2-A011 runner integration identifier unchanged.  The
    # derived client binds every raw request and final trace to V2-A015/g=1.
    run_evaluation(args_cli, policy="cosmos3_nano_v2", client_factory=make_client)
    _write_pair_manifest()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Cosmos3 Nano V2-A015 g=1] technical failure: {exc}")
        traceback.print_exc()
        simulation_app.close()
        raise
