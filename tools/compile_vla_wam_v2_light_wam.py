#!/usr/bin/env python3
"""Compile the frozen V2-A006 Light-WAM six-cell RoboTwin slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


SCENES = {
    "robotwin_pair_00": ("place_a2b_left", 4_300_000, 8_400),
    "robotwin_pair_01": ("place_a2b_right", 4_300_001, 8_401),
    "robotwin_pair_02": ("place_a2b_left", 4_300_002, 8_402),
}

THERMAL_LOGS = {
    "robotwin_pair_00": "thermal_robotwin_pair_00_attempt02.jsonl",
    "robotwin_pair_01": "thermal_robotwin_pair_01_attempt02.jsonl",
    "robotwin_pair_02": "thermal_robotwin_pair_02.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing evidence file: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rms_delta(left: np.ndarray, right: np.ndarray, limit: int | None = None) -> float:
    count = min(len(left), len(right))
    if limit is not None:
        count = min(count, limit)
    if count == 0:
        raise RuntimeError("Cannot compare empty action traces")
    delta = left[:count].astype(np.float64) - right[:count].astype(np.float64)
    return float(math.sqrt(float(np.mean(np.square(delta)))))


def thermal_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    if not rows or rows[0]["event"] != "worker_started" or rows[-1]["event"] != "monitor_completed":
        raise RuntimeError(f"Incomplete thermal lifecycle: {path}")
    temperatures = [row["temperature_c"] for row in rows if "temperature_c" in row]
    events = [row["event"] for row in rows]
    return {
        **file_record(path),
        "record_count": len(rows),
        "temperature_sample_count": len(temperatures),
        "maximum_temperature_c": max(temperatures),
        "pause_count": events.count("cooldown_started"),
        "resume_count": events.count("cooldown_completed"),
        "emergency_count": events.count("emergency_hold"),
        "worker_exit_code": rows[-1].get("worker_exit_code"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-head", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.raw_root.resolve()
    episodes: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    evidence_files: list[dict[str, Any]] = []
    thermal: list[dict[str, Any]] = []
    excluded_setup_thermal: list[dict[str, Any]] = []
    invalid_attempts: list[dict[str, Any]] = []

    for pair_id, (task, environment_seed, sampling_seed) in SCENES.items():
        pair_root = root / "episodes" / pair_id
        manifest_path = pair_root / "manifest.json"
        results_path = pair_root / "results.jsonl"
        invalid_path = pair_root / "invalid_attempts.jsonl"
        manifest = json.loads(manifest_path.read_text())
        results = load_jsonl(results_path)
        invalid = load_jsonl(invalid_path)
        if manifest.get("expected_episode_count") != 2 or len(results) != 2 or invalid:
            raise RuntimeError(f"Pair is not exactly two valid cells: {pair_id}")
        if {row["requested_relation"] for row in results} != {"left", "right"}:
            raise RuntimeError(f"Pair lacks exact LEFT/RIGHT cells: {pair_id}")
        by_relation = {row["requested_relation"]: row for row in results}
        left, right = by_relation["left"], by_relation["right"]
        for row in results:
            if (
                row["model_id"] != "light_wam_robotwin"
                or row["pair_id"] != pair_id
                or row["task"] != task
                or row["environment_seed"] != environment_seed
                or row["sampling_seed"] != sampling_seed
                or row["prompt_family"] != "direct_command"
                or row["prompt_controller"] != "episode_static"
                or row["oracle_actions"] != 0
                or row["dynamic_prompt_switches"] != 0
                or row["future_interface"] != "action_only_infer_action"
                or row["imagined_future_video"] is not None
                or row["imagined_future_artifact"] is not None
            ):
                raise RuntimeError(f"Protocol mismatch: {pair_id}/{row.get('requested_relation')}")
            trace_path = Path(row["action_trace"]["path"])
            trajectory_path = Path(row["trajectory_path"])
            video_path = Path(row["simulator_video"])
            for path in (trace_path, trajectory_path, video_path, trace_path.parent / "result.json"):
                evidence_files.append(file_record(path))
            if sha256_file(trace_path) != row["action_trace"]["sha256"]:
                raise RuntimeError(f"Action trace hash mismatch: {trace_path}")
            episodes.append(row)

        if left["initial_observation_sha256"] != right["initial_observation_sha256"]:
            raise RuntimeError(f"Paired initial observation mismatch: {pair_id}")
        left_actions = np.load(left["action_trace"]["path"])["executed"]
        right_actions = np.load(right["action_trace"]["path"])["executed"]
        pair_record = {
            "pair_id": pair_id,
            "initial_observation_sha256": left["initial_observation_sha256"],
            "left_requested_success": bool(left["requested_success"]),
            "right_requested_success": bool(right["requested_success"]),
            "left_final_object_minus_target_x": left["final"]["object_minus_target_x"],
            "right_final_object_minus_target_x": right["final"]["object_minus_target_x"],
            "right_minus_left_endpoint_x": (
                right["final"]["object_minus_target_x"] - left["final"]["object_minus_target_x"]
            ),
            "endpoint_ordering_aligned": bool(
                right["final"]["object_minus_target_x"] > left["final"]["object_minus_target_x"]
            ),
            "executed_actions_distinct": bool(not np.array_equal(left_actions, right_actions)),
            "first_10_action_rms": rms_delta(left_actions, right_actions, 10),
            "overlap_action_rms": rms_delta(left_actions, right_actions),
        }
        pairs.append(pair_record)
        evidence_files.extend(file_record(path) for path in (manifest_path, results_path, invalid_path))
        thermal.append(thermal_summary(root / THERMAL_LOGS[pair_id]))
        invalid_attempts.extend(invalid)

    setup_root = root / "invalid_setup" / "ffmpeg_missing"
    for pair_id in ("robotwin_pair_00", "robotwin_pair_01"):
        setup_invalid_path = setup_root / pair_id / "invalid_attempts.jsonl"
        setup_invalid = load_jsonl(setup_invalid_path)
        if len(setup_invalid) != 2 or any(
            row.get("included_in_model_denominator") is not False for row in setup_invalid
        ):
            raise RuntimeError(f"Expected two excluded ffmpeg setup attempts: {pair_id}")
        invalid_attempts.extend(setup_invalid)
        evidence_files.append(file_record(setup_invalid_path))
        excluded_setup_thermal.append(thermal_summary(root / f"thermal_{pair_id}.jsonl"))

    success = {
        relation: sum(
            bool(row["requested_success"])
            for row in episodes
            if row["requested_relation"] == relation
        )
        for relation in ("left", "right")
    }
    competence = (
        "both_directions" if success["left"] and success["right"]
        else "left_only" if success["left"]
        else "right_only" if success["right"]
        else "zero_direction"
    )
    payload = {
        "schema_version": "vla-wam-shared-v2-light-wam-robotwin-slice-v1",
        "status": "complete",
        "model_id": "light_wam_robotwin",
        "amendment_id": "V2-A006",
        "compiled_at_git_head": args.git_head,
        "raw_root": str(root),
        "valid_episode_count": len(episodes),
        "invalid_attempt_count": len(invalid_attempts),
        "runtime_intervention_count": sum(row["pause_count"] for row in thermal),
        "requested_success_count": sum(success.values()),
        "success_by_relation": {
            relation: {"successes": count, "trials": 3} for relation, count in success.items()
        },
        "competence_gate": competence,
        "wording_grid_eligible": competence == "both_directions",
        "future_interface": "action_only_infer_action",
        "imagined_future_evidence": None,
        "pairs": pairs,
        "episodes": episodes,
        "thermal_lifecycles": thermal,
        "excluded_setup_thermal_lifecycles": excluded_setup_thermal,
        "invalid_attempts": invalid_attempts,
        "evidence_files": sorted(evidence_files, key=lambda row: row["path"]),
    }
    if len(episodes) != 6:
        raise RuntimeError(f"Expected six valid episodes, found {len(episodes)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "valid_episode_count": len(episodes),
        "requested_success_count": sum(success.values()),
        "success_by_relation": success,
        "competence_gate": competence,
    }, indent=2))


if __name__ == "__main__":
    main()
