#!/usr/bin/env python3
"""Fail-closed validator/compiler for one completed GR00T v3 matched pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np


PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}


def fail(message: str) -> None:
    raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        fail(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        fail(f"expected JSONL objects: {path}")
    return rows


def record(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size <= 0:
        fail(f"missing or empty retained artifact: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_trace(raw_root: Path, seed: int, relation: str, actions_executed: int) -> dict:
    trace_root = raw_root / "actions"
    stem = f"seed{seed}_{relation}"
    metadata_path = trace_root / f"{stem}_executed_actions.json"
    metadata = load_json(metadata_path)
    actions_path = trace_root / f"{stem}_executed_actions.npy"
    chunks_path = trace_root / f"{stem}_returned_action_chunks.npy"
    modalities_path = trace_root / f"{stem}_returned_action_modalities.npz"
    actions = np.load(actions_path, allow_pickle=False)
    chunks = np.load(chunks_path, allow_pickle=False)
    modalities = np.load(modalities_path, allow_pickle=False)
    if actions.shape != (actions_executed, 8):
        fail(f"executed action shape mismatch for {seed}/{relation}: {actions.shape}")
    request_count = math.ceil(actions_executed / 8)
    if chunks.shape != (request_count, 40, 8):
        fail(f"returned chunk shape mismatch for {seed}/{relation}: {chunks.shape}")
    required_modalities = {
        "action.eef_9d", "action.gripper_position", "action.joint_position"
    }
    if not required_modalities.issubset(set(modalities.files)):
        fail(f"missing returned modalities for {seed}/{relation}")
    expected_request_seeds = [seed * 1000 + index for index in range(request_count)]
    checks = {
        "prompt": PROMPTS[relation],
        "sampling_seed_base": seed,
        "request_sampling_seeds": expected_request_seeds,
        "count": actions_executed,
        "shape": list(actions.shape),
        "sha256": sha256(actions_path),
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            fail(f"action metadata mismatch for {seed}/{relation}.{key}")
    if Path(metadata.get("path", "")) != actions_path:
        fail(f"executed action path mismatch for {seed}/{relation}")
    chunk_entry = metadata.get("returned_action_chunks", {})
    if (
        chunk_entry.get("sha256") != sha256(chunks_path)
        or chunk_entry.get("shape") != list(chunks.shape)
        or int(chunk_entry.get("count", -1)) != request_count
    ):
        fail(f"returned chunk metadata mismatch for {seed}/{relation}")
    modality_entry = metadata.get("returned_action_modalities", {})
    if modality_entry.get("sha256") != sha256(modalities_path):
        fail(f"returned modality metadata mismatch for {seed}/{relation}")
    return {
        "metadata": record(metadata_path),
        "executed_actions": record(actions_path),
        "returned_action_chunks": record(chunks_path),
        "returned_action_modalities": record(modalities_path),
        "request_count": request_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--robolab-output", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.robolab_output.resolve()
    compiled_root = raw_root / "compiled"
    compiled_root.mkdir(parents=True, exist_ok=False)

    thermal_rows = load_jsonl(raw_root / "thermal/events.jsonl")
    if thermal_rows[-1].get("event") != "monitor_completed":
        fail("thermal lifecycle is not complete")
    if thermal_rows[-1].get("worker_exit_code") != 0:
        fail("simulator worker did not exit cleanly")
    if any(row.get("event") in {"emergency_hold", "temperature_query_failed"} for row in thermal_rows):
        fail("retained pair contains an emergency thermal event")

    result_rows = load_jsonl(output_root / "episode_results.jsonl")
    results = {row.get("env_name"): row for row in result_rows}
    if set(results) != set(TASKS.values()):
        fail(f"expected exactly both matched tasks, got {sorted(results)}")

    captures: dict[str, dict] = {}
    pair_artifacts: dict[str, dict] = {}
    for relation, task in TASKS.items():
        state_root = raw_root / "simulator/state_capture"
        capture_path = state_root / f"seed{args.seed}_{relation}.json"
        stream_path = state_root / f"seed{args.seed}_{relation}_states.partial.jsonl"
        warmup_path = state_root / f"seed{args.seed}_{relation}_warmup_reset0.jsonl"
        capture = load_json(capture_path)
        captures[relation] = capture
        actions_executed = capture.get("actions_executed")
        if type(actions_executed) is not int or actions_executed <= 0:
            fail(f"invalid action count for {args.seed}/{relation}")
        if capture.get("behavioral_result_valid_candidate") is not True:
            fail(f"partial behavioral candidate for {args.seed}/{relation}")
        if capture.get("prompt") != PROMPTS[relation]:
            fail(f"prompt mismatch in state capture for {args.seed}/{relation}")
        if capture.get("environment_seed") != args.seed or capture.get("policy_seed") != args.seed:
            fail(f"seed mismatch in state capture for {args.seed}/{relation}")
        samples = capture.get("samples")
        if not isinstance(samples, list) or len(samples) != actions_executed + 1:
            fail(f"state N+1 mismatch for {args.seed}/{relation}")
        if [row.get("action_step") for row in samples] != list(range(actions_executed + 1)):
            fail(f"non-contiguous state sequence for {args.seed}/{relation}")
        stream_rows = load_jsonl(stream_path)
        if len(stream_rows) != actions_executed + 1:
            fail(f"partial stream length mismatch for {args.seed}/{relation}")
        for index, (stream_row, sample) in enumerate(zip(stream_rows, samples)):
            for key in ("action_step", "object_xyz", "reference_xyz", "grippers_open"):
                if stream_row.get(key) != sample.get(key):
                    fail(f"state stream/capture mismatch at {args.seed}/{relation}/{index}.{key}")
        if len(load_jsonl(warmup_path)) != 1:
            fail(f"warm-up reset evidence mismatch for {args.seed}/{relation}")

        task_root = output_root / task
        env_cfg = load_json(task_root / "env_cfg.json")
        log = load_json(task_root / "log_0_env0.json")
        videos = sorted(task_root.glob("*_viewport.mp4"))
        if len(videos) != 1:
            fail(f"expected one viewport video for {args.seed}/{relation}")
        result = results[task]
        if env_cfg.get("seed") != args.seed or env_cfg.get("instruction") != PROMPTS[relation]:
            fail(f"environment provenance mismatch for {args.seed}/{relation}")
        if result.get("instruction") != PROMPTS[relation]:
            fail(f"episode result prompt mismatch for {args.seed}/{relation}")
        if bool(result.get("success")) != bool(log.get("success")):
            fail(f"RoboLab result/log success mismatch for {args.seed}/{relation}")
        if bool(result.get("success")) != bool(capture.get("requested_success")):
            fail(f"RoboLab/state-capture success mismatch for {args.seed}/{relation}")
        if int(log.get("final_step", -1)) != actions_executed:
            fail(f"RoboLab/state-capture action-count mismatch for {args.seed}/{relation}")

        trace = validate_trace(raw_root, args.seed, relation, actions_executed)
        output_jsonl = compiled_root / f"seed{args.seed}_{relation}.jsonl"
        adapter_command = [
            sys.executable,
            str(args.study_root / "experiments/v3/groot_droid/adapter.py"),
            "compile-behavioral",
            "--study-root", str(args.study_root),
            "--seed", str(args.seed),
            "--relation", relation,
            "--runtime-identity", str(args.runtime_identity),
            "--capture", str(capture_path),
            "--output-jsonl", str(output_jsonl),
            "--video", str(videos[0]),
            "--action-trace", str(raw_root / "actions" / f"seed{args.seed}_{relation}_executed_actions.npy"),
        ]
        subprocess.run(adapter_command, check=True)
        compiled_rows = load_jsonl(output_jsonl)
        if len(compiled_rows) != 1 or compiled_rows[0].get("behavioral_result_valid") is not True:
            fail(f"compiled behavioral record invalid for {args.seed}/{relation}")
        pair_artifacts[relation] = {
            "capture": record(capture_path),
            "state_stream": record(stream_path),
            "warmup_reset": record(warmup_path),
            "episode_results": record(output_root / "episode_results.jsonl"),
            "episode_log": record(task_root / "log_0_env0.json"),
            "environment": record(task_root / "env_cfg.json"),
            "viewport_video": record(videos[0]),
            "compiled_jsonl": record(output_jsonl),
            "trace": trace,
        }

    for key in ("object_xyz", "reference_xyz"):
        left = np.asarray(captures["left"]["samples"][0][key], dtype=np.float64)
        right = np.asarray(captures["right"]["samples"][0][key], dtype=np.float64)
        if not np.array_equal(left, right):
            fail(f"matched initial reset mismatch for seed {args.seed}: {key}")

    manifest = {
        "schema_version": "vla-wam-shared-v3-groot-pair-raw-manifest-v1",
        "model_id": "groot_n17_droid_vla",
        "environment_seed": args.seed,
        "sampling_seed": args.seed,
        "matched_pair_valid": True,
        "relations": pair_artifacts,
        "thermal_events": record(raw_root / "thermal/events.jsonl"),
    }
    manifest_path = compiled_root / f"seed{args.seed}_pair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "valid", "manifest": record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
