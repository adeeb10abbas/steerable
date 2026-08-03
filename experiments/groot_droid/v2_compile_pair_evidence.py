#!/usr/bin/env python3
"""Compile one frozen GR00T DROID LEFT/RIGHT pair into compact evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


TASKS = {
    "left": {
        "task": "RubiksCubeLeftOfBowlMatchedTask",
        "prompt": "Put the Rubik's cube to the left of the bowl.",
    },
    "right": {
        "task": "RubiksCubeRightOfBowlMatchedTask",
        "prompt": "Put the Rubik's cube to the right of the bowl.",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=[8300, 8301, 8302], required=True)
    parser.add_argument("--simulator-output", type=Path, required=True)
    parser.add_argument("--action-trace-dir", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--study-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    fixture = json.loads(args.fixture_manifest.read_text())
    neutral = fixture["neutral_reset_contract"]
    if neutral["left_predicate_at_reset"] or neutral["right_predicate_at_reset"]:
        raise ValueError("Fixture does not satisfy the neutral reset contract")

    result_path = args.simulator_output / "episode_results.jsonl"
    result_rows = [
        json.loads(line) for line in result_path.read_text().splitlines() if line.strip()
    ]
    if len(result_rows) != 2:
        raise ValueError(f"Expected two completed cells, found {len(result_rows)}")
    rows_by_task = {row["task_name"]: row for row in result_rows}

    episodes: list[dict[str, object]] = []
    initial_states: dict[str, dict[str, np.ndarray]] = {}
    executed_actions: dict[str, np.ndarray] = {}
    for condition, contract in TASKS.items():
        task = contract["task"]
        row = rows_by_task.get(task)
        if row is None or row["instruction"] != contract["prompt"]:
            raise ValueError(f"Missing or mismatched {condition} result row")
        task_dir = args.simulator_output / task
        hdf5_path = task_dir / "run_0.hdf5"
        video_paths = sorted(task_dir.glob("*_viewport.mp4"))
        if len(video_paths) != 1:
            raise ValueError(f"Expected one viewport video for {condition}")

        trace_prefix = args.action_trace_dir / f"seed{args.seed}_{condition}"
        trace_manifest_path = trace_prefix.with_name(trace_prefix.name + "_executed_actions.json")
        trace_manifest = json.loads(trace_manifest_path.read_text())
        action_path = trace_prefix.with_name(trace_prefix.name + "_executed_actions.npy")
        chunks_path = trace_prefix.with_name(
            trace_prefix.name + "_returned_action_chunks.npy"
        )
        modalities_path = trace_prefix.with_name(
            trace_prefix.name + "_returned_action_modalities.npz"
        )

        saved_actions = np.load(action_path, allow_pickle=False)
        with h5py.File(hdf5_path) as h5:
            demo = h5["data/demo_0"]
            hdf5_actions = demo["actions"][:]
            cube = demo["states/rigid_object/rubiks_cube/root_pose"][:]
            bowl = demo["states/rigid_object/bowl/root_pose"][:]
            initial_cube = demo[
                "initial_state/rigid_object/rubiks_cube/root_pose"
            ][-1]
            initial_bowl = demo["initial_state/rigid_object/bowl/root_pose"][-1]
        if not np.array_equal(saved_actions, hdf5_actions):
            raise ValueError(f"Saved and simulator actions differ for {condition}")
        if len(saved_actions) != row["episode_step"] or len(cube) != row["episode_step"]:
            raise ValueError(f"Step-count mismatch for {condition}")
        if trace_manifest["prompt"] != contract["prompt"]:
            raise ValueError(f"Trace prompt mismatch for {condition}")
        if trace_manifest["count"] != row["episode_step"]:
            raise ValueError(f"Trace count mismatch for {condition}")

        initial_states[condition] = {"cube": initial_cube, "bowl": initial_bowl}
        executed_actions[condition] = saved_actions
        endpoint_delta = cube[-1, :3] - bowl[-1, :3]
        episodes.append(
            {
                "condition": condition.upper(),
                "environment_seed": args.seed,
                "sampling_seed_base": args.seed,
                "prompt": contract["prompt"],
                "success": bool(row["success"]),
                "executed_action_count": int(row["episode_step"]),
                "initial_cube_world_xyz": initial_cube[:3].tolist(),
                "initial_bowl_world_xyz": initial_bowl[:3].tolist(),
                "endpoint_cube_world_xyz": cube[-1, :3].tolist(),
                "endpoint_bowl_world_xyz": bowl[-1, :3].tolist(),
                "endpoint_cube_minus_bowl_world_xyz": endpoint_delta.tolist(),
                "requested_direction_endpoint_scalar": float(
                    endpoint_delta[1] if condition == "left" else -endpoint_delta[1]
                ),
                "files": {
                    "simulator_hdf5": _file_record(hdf5_path),
                    "simulator_video": _file_record(video_paths[0]),
                    "simulator_env_config": _file_record(task_dir / "env_cfg.json"),
                    "simulator_episode_log": _file_record(task_dir / "log_0_env0.json"),
                    "executed_actions": _file_record(action_path),
                    "action_trace_manifest": _file_record(trace_manifest_path),
                    "returned_action_chunks": _file_record(chunks_path),
                    "returned_action_modalities": _file_record(modalities_path),
                },
                "future_interface": {
                    "returned_action_chunks": trace_manifest["returned_action_chunks"],
                    "returned_action_modalities": trace_manifest[
                        "returned_action_modalities"
                    ],
                },
            }
        )

    for name in ("cube", "bowl"):
        if not np.array_equal(initial_states["left"][name], initial_states["right"][name]):
            raise ValueError(f"LEFT/RIGHT initial {name} states differ")
    action_delta = executed_actions["left"] - executed_actions["right"]
    manifest = {
        "schema_version": "vla-wam-shared-v2-groot-droid-pair-slice-v1",
        "amendment_id": "V2-A005",
        "model": "nvidia/GR00T-N1.7-DROID",
        "model_revision": "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5",
        "study_commit": args.study_commit,
        "environment_seed": args.seed,
        "fixture": {
            "manifest": str(args.fixture_manifest),
            "npz_path": fixture["npz_path"],
            "npz_sha256": fixture["npz_sha256"],
            "neutral_reset_contract": neutral,
        },
        "instruction_controller": "static",
        "oracle_or_subtask_coach": False,
        "episodes": episodes,
        "pair_checks": {
            "valid_cell_count": 2,
            "identical_initial_cube_state": True,
            "identical_initial_bowl_state": True,
            "executed_action_rms": float(np.sqrt(np.mean(action_delta**2))),
            "executed_action_max_abs": float(np.max(np.abs(action_delta))),
            "executed_actions_differ": bool(np.any(action_delta != 0)),
        },
        "raw_episode_results": _file_record(result_path),
        "denominator_status": "two_valid_cells",
        "runtime_interventions": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
