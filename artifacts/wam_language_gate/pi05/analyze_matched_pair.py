#!/usr/bin/env python3
"""Summarize exact-start pi0.5 left/right RoboLab rollouts.

Run this inside an environment with h5py and numpy.  The script intentionally
keeps RoboLab's binary termination separate from a two-item, paper-style task
progression rubric:

1. the requested object was picked up (persistent credit), and
2. the object reached the requested spatial relation.

This prevents a rollout that merely pushes an object across the relation
boundary from being counted as a complete pick-and-place execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}


def _rms(a: np.ndarray, b: np.ndarray, steps: int | None = None) -> float:
    length = min(len(a), len(b))
    if steps is not None:
        length = min(length, steps)
    return float(np.sqrt(np.mean(np.square(a[:length] - b[:length]))))


def _initial_arrays(group: h5py.Group) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}

    def collect(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset):
            arrays[name] = item[...]

    group.visititems(collect)
    return arrays


def _initial_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _max_initial_difference(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> float:
    if a.keys() != b.keys():
        raise ValueError("Initial-state dataset keys differ")
    return max(float(np.max(np.abs(a[key] - b[key]))) for key in a)


def _common_initial_comparison(
    a: dict[str, np.ndarray], b: dict[str, np.ndarray]
) -> dict[str, Any]:
    common = sorted(a.keys() & b.keys())
    return {
        "common_dataset_count": len(common),
        "common_max_abs_difference": max(
            float(np.max(np.abs(a[key] - b[key]))) for key in common
        ),
        "only_in_primary": sorted(a.keys() - b.keys()),
        "only_in_comparison": sorted(b.keys() - a.keys()),
    }


def load_rollout(output_root: Path, direction: str, run: int) -> dict[str, Any]:
    task = TASKS[direction]
    task_dir = output_root / task
    h5_path = task_dir / f"run_{run}.hdf5"
    log_path = task_dir / f"log_{run}_env0.json"

    with h5py.File(h5_path, "r") as handle:
        demo = handle["data/demo_0"]
        actions = demo["actions"][...]
        cube = demo["states/rigid_object/rubiks_cube/root_pose"][...]
        bowl = demo["states/rigid_object/bowl/root_pose"][...]
        initial = _initial_arrays(demo["initial_state"])

    log = json.loads(log_path.read_text())
    events = log["events"]
    picked = any(
        event["name"] == "OBJECT_GRABBED_SUCCESS"
        and "rubiks_cube" in event.get("info", "")
        for event in events
    )
    relation_event = f"OBJECT_{direction.upper()}_OF_SUCCESS"
    reached_relation = any(event["name"] == relation_event for event in events)
    relation_step = next(
        (event["step"] for event in events if event["name"] == relation_event), None
    )
    pickup_step = next(
        (
            event["step"]
            for event in events
            if event["name"] == "OBJECT_GRABBED_SUCCESS"
            and "rubiks_cube" in event.get("info", "")
        ),
        None,
    )
    paper_progression = (int(picked) + int(reached_relation)) / 2.0
    strict_progression = (int(picked) + int(picked and reached_relation)) / 2.0

    return {
        "actions": actions,
        "cube": cube,
        "bowl": bowl,
        "initial": initial,
        "summary": {
            "direction": direction,
            "run": run,
            "steps": int(log["final_step"]),
            "binary_success": bool(log["success"]),
            "picked_correct_object": picked,
            "pickup_step": pickup_step,
            "reached_requested_relation": reached_relation,
            "relation_step": relation_step,
            "paper_progression": paper_progression,
            "strict_pick_then_place_progression": strict_progression,
            "initial_cube_minus_bowl_y_m": float(cube[0, 1] - bowl[0, 1]),
            "final_cube_minus_bowl_y_m": float(cube[-1, 1] - bowl[-1, 1]),
            "initial_state_sha256": _initial_fingerprint(initial),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--paraphrase-root", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    rollouts = {
        direction: {
            run: load_rollout(args.output_root, direction, run) for run in (0, 1)
        }
        for direction in TASKS
    }
    paraphrases = (
        {
            direction: load_rollout(args.paraphrase_root, direction, 0)
            for direction in TASKS
        }
        if args.paraphrase_root
        else None
    )
    comparison = (
        {
            direction: load_rollout(args.comparison_root, direction, 0)
            for direction in TASKS
        }
        if args.comparison_root
        else None
    )

    opposite_first_chunk = {
        f"run_{run}": _rms(
            rollouts["left"][run]["actions"],
            rollouts["right"][run]["actions"],
            steps=15,
        )
        for run in (0, 1)
    }
    same_prompt_first_chunk = {
        direction: _rms(
            rollouts[direction][0]["actions"],
            rollouts[direction][1]["actions"],
            steps=15,
        )
        for direction in TASKS
    }
    effect = float(np.mean(list(opposite_first_chunk.values())))
    noise = float(np.mean(list(same_prompt_first_chunk.values())))

    result = {
        "schema_version": 1,
        "model": "pi0.5 DROID joint-position",
        "checkpoint": "gs://openpi-assets-simeval/pi05_droid_jointpos",
        "action_chunk_steps": 15,
        "rollouts": [
            rollouts[direction][run]["summary"]
            for direction in TASKS
            for run in (0, 1)
        ],
        "controls": {
            "initial_state_max_abs_difference": {
                "left_vs_right_run_0": _max_initial_difference(
                    rollouts["left"][0]["initial"],
                    rollouts["right"][0]["initial"],
                ),
                "left_vs_right_run_1": _max_initial_difference(
                    rollouts["left"][1]["initial"],
                    rollouts["right"][1]["initial"],
                ),
                "left_repeat": _max_initial_difference(
                    rollouts["left"][0]["initial"],
                    rollouts["left"][1]["initial"],
                ),
                "right_repeat": _max_initial_difference(
                    rollouts["right"][0]["initial"],
                    rollouts["right"][1]["initial"],
                ),
            },
            "opposite_prompt_action_rms_first_chunk": opposite_first_chunk,
            "same_prompt_repeat_action_rms_first_chunk": same_prompt_first_chunk,
            "mean_opposite_prompt_effect": effect,
            "mean_same_prompt_noise_floor": noise,
            "effect_to_noise_ratio": effect / noise if noise else None,
            "same_prompt_repeat_action_rms_shared_horizon": {
                direction: _rms(
                    rollouts[direction][0]["actions"],
                    rollouts[direction][1]["actions"],
                )
                for direction in TASKS
            },
        },
        "aggregate": {
            "binary_success_rate": {
                direction: float(
                    np.mean(
                        [
                            rollouts[direction][run]["summary"]["binary_success"]
                            for run in (0, 1)
                        ]
                    )
                )
                for direction in TASKS
            },
            "mean_paper_progression": {
                direction: float(
                    np.mean(
                        [
                            rollouts[direction][run]["summary"]["paper_progression"]
                            for run in (0, 1)
                        ]
                    )
                )
                for direction in TASKS
            },
            "mean_strict_pick_then_place_progression": {
                direction: float(
                    np.mean(
                        [
                            rollouts[direction][run]["summary"]
                            ["strict_pick_then_place_progression"]
                            for run in (0, 1)
                        ]
                    )
                )
                for direction in TASKS
            },
        },
        "interpretation_guardrail": (
            "Treat the language effect as resolved only when opposite-prompt action "
            "separation exceeds same-prompt repeat variation and behavior changes in "
            "the requested directions. Binary success alone is insufficient."
        ),
    }

    if paraphrases:
        result["paraphrase_rollouts"] = [
            paraphrases[direction]["summary"] for direction in TASKS
        ]
        result["controls"]["canonical_vs_paraphrase_action_rms_first_chunk"] = {
            direction: _rms(
                rollouts[direction][0]["actions"],
                paraphrases[direction]["actions"],
                steps=15,
            )
            for direction in TASKS
        }
        result["controls"]["paraphrase_opposite_prompt_action_rms_first_chunk"] = (
            _rms(
                paraphrases["left"]["actions"],
                paraphrases["right"]["actions"],
                steps=15,
            )
        )

    if comparison:
        result["cross_model_initial_state"] = {
            direction: _common_initial_comparison(
                rollouts[direction][0]["initial"],
                comparison[direction]["initial"],
            )
            for direction in TASKS
        }

    rendered = json.dumps(result, indent=2) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
