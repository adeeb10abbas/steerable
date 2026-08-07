#!/usr/bin/env python3
"""Exact-repeat and LEFT/RIGHT prompt-sensitivity gate for FastWAM v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROMPT_LEFT = "Put the small woodenblock to the left of the red playingcards box."
PROMPT_RIGHT = "Put the small woodenblock to the right of the red playingcards box."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict:
    path = path.resolve()
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sampling-seed", type=int, default=8503)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = args.repository.resolve()
    sys.path.insert(0, str(repository / "experiments" / "robotwin_language_gate"))
    sys.path.insert(0, str(repository))
    import closed_loop_language_gate as gate

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.observation.resolve()) as payload:
        observation = {
            "observation": {
                "head_camera": {"rgb": payload["head"].copy()},
                "left_camera": {"rgb": payload["left"].copy()},
                "right_camera": {"rgb": payload["right"].copy()},
            },
            "joint_action": {"vector": payload["state"].copy()},
        }

    model_args = SimpleNamespace(
        checkpoint=args.checkpoint.resolve(),
        dataset_stats=args.dataset_stats.resolve(),
        action_horizon=32,
        replan_steps=24,
        num_inference_steps=10,
        text_cfg_scale=2.0,
    )
    torch.cuda.reset_peak_memory_stats()
    loaded_at = time.perf_counter()
    policy = gate.build_policy(model_args)
    load_seconds = time.perf_counter() - loaded_at

    actions = {}
    latencies = {}
    for name, prompt in (("left", PROMPT_LEFT), ("left_repeat", PROMPT_LEFT), ("right", PROMPT_RIGHT)):
        gate.seed_everything(args.sampling_seed)
        policy.seed = args.sampling_seed
        policy.text_cfg_scale = 2.0
        policy.negative_prompt = ""
        policy.reset()
        started = time.perf_counter()
        value = np.asarray(policy._infer_action_chunk(observation, prompt), dtype=np.float32)
        torch.cuda.synchronize()
        latencies[name] = time.perf_counter() - started
        actions[name] = value
        print(f"{name}: shape={value.shape} seconds={latencies[name]:.3f}", flush=True)

    repeat_identical = bool(np.array_equal(actions["left"], actions["left_repeat"]))
    delta = actions["right"].astype(np.float64) - actions["left"].astype(np.float64)
    left_right_rms = float(np.sqrt(np.mean(np.square(delta))))
    finite = bool(all(np.isfinite(value).all() for value in actions.values()))
    checks = {
        "exact_left_prompt": True,
        "exact_right_prompt": True,
        "repeat_bit_identical": repeat_identical,
        "left_right_distinct": math.isfinite(left_right_rms) and left_right_rms > 0.0,
        "finite_actions": finite,
        "one_registered_sampling_seed": True,
        "frozen_text_cfg_scale": policy.text_cfg_scale == 2.0,
        "frozen_action_horizon": actions["left"].shape[0] == 32,
    }
    if not all(checks.values()):
        raise SystemExit(f"FastWAM fixed-observation release gate failed: {checks}")

    actions_path = args.output_dir / "actions.npz"
    np.savez_compressed(actions_path, **actions)
    native_manifest = {
        "schema_version": "vla-wam-shared-v3-robotwin-fastwam-fixed-observation-native-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": "fastwam_robotwin",
        "status": "passed",
        "behavioral_episodes": 0,
        "model_action_requests": 3,
        "sampling_seed": args.sampling_seed,
        "prompts": {"left": PROMPT_LEFT, "left_repeat": PROMPT_LEFT, "right": PROMPT_RIGHT},
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_stats": str(args.dataset_stats.resolve()),
        "observation": file_record(args.observation.resolve()),
        "action_shape": list(actions["left"].shape),
        "left_right_action_rms": left_right_rms,
        "repeat_bit_identical": repeat_identical,
        "load_seconds": load_seconds,
        "condition_seconds": latencies,
        "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
    }
    native_manifest_path = args.output_dir / "manifest.json"
    native_manifest_path.write_text(json.dumps(native_manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    release = {
        "schema_version": "vla-wam-shared-v3-robotwin-fixed-observation-release-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": "fastwam_robotwin",
        "status": "passed_exact_repeat_and_left_right_prompt_sensitivity",
        "behavioral_episodes": 0,
        "model_action_requests": 3,
        "requested_release_checks": checks,
        "prompts": {"left": PROMPT_LEFT, "right": PROMPT_RIGHT},
        "action_array_shape": list(actions["left"].shape),
        "left_right_action_rms": left_right_rms,
        "actions": file_record(actions_path),
        "native_probe_manifest": file_record(native_manifest_path),
    }
    release_path = args.output_dir / "fixed_observation_release.json"
    release_path.write_text(json.dumps(release, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(release, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
