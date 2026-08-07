#!/usr/bin/env python3
"""Exact nine-request fixed-observation gate for Nano V3-B005."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from experiments.v3.cosmos_nano_lateral_sweep.live_support import (
    PINNED_SERVER_PORT,
    PROBE_SEQUENCE,
    observation_component_hashes,
    validate_fixed_observation_report,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    AMENDMENT_ID,
    FIXED_OBSERVATION_SCHEMA,
    MODEL_ID,
    PROBE_LEVELS,
    PROMPTS,
    ReleaseBundle,
    RuntimeContractError,
    load_release_bundle,
    sha256_bytes,
    sha256_file,
)


def _fail(message: str) -> None:
    raise RuntimeContractError(message)


def _array_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def _rms(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(
        first.astype(np.float64) - second.astype(np.float64)
    ))))


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(
        first.astype(np.float64) - second.astype(np.float64)
    )))


def collect_responses(
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    observations: Mapping[int, Mapping[str, Any]],
    infer: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[tuple[int, str], dict[str, Any]]:
    """Issue level0/3/6 × LEFT/repeat/RIGHT in the frozen order."""

    if set(observations) != set(PROBE_LEVELS):
        _fail("V3-B005 fixed gate requires observations for exactly levels 0, 3, and 6")
    responses: dict[tuple[int, str], dict[str, Any]] = {}
    for request_index, (level, condition) in enumerate(PROBE_SEQUENCE):
        relation = "left" if condition.startswith("left") else "right"
        cell = release.cell(f"v3b005:nano:seed9500:level{level}:{relation}")
        source = observations[level]
        request = {
            "observation/image": np.array(source["observation/image"], copy=True),
            "observation/joint_position": np.array(
                source["observation/joint_position"], copy=True
            ),
            "observation/gripper_position": np.array(
                source["observation/gripper_position"], copy=True
            ),
            "v3b005_server_mode": "probe_only",
            "amendment_id": AMENDMENT_ID,
            "probe_request_index": request_index,
            "probe_level_index": level,
            "probe_condition": condition,
            "registered_cell_id": cell.cell_id,
            "sampling_seed": 9500,
            "prompt": PROMPTS[relation],
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        }
        request["observation_hashes"] = observation_component_hashes(request)
        started = time.perf_counter()
        response = dict(infer(request))
        response["wall_time_s"] = time.perf_counter() - started
        response["observation_hashes"] = request["observation_hashes"]
        responses[(level, condition)] = response
    return responses


def evaluate_responses(
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    responses: Mapping[tuple[int, str], Mapping[str, Any]],
) -> dict[str, Any]:
    if set(responses) != set(PROBE_SEQUENCE):
        _fail("V3-B005 gate requires exactly the registered nine responses")
    actions: dict[tuple[int, str], np.ndarray] = {}
    futures: dict[tuple[int, str], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for request_index, (level, condition) in enumerate(PROBE_SEQUENCE):
        relation = "left" if condition.startswith("left") else "right"
        cell = release.cell(f"v3b005:nano:seed9500:level{level}:{relation}")
        response = responses[(level, condition)]
        expected_metadata = {
            "v3b005_server_mode": "probe_only",
            "amendment_id": AMENDMENT_ID,
            "registered_cell_id": cell.cell_id,
            "sampling_seed": 9500,
            "request_index": request_index,
            "probe_request_index": request_index,
            "probe_level_index": level,
            "probe_condition": condition,
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        }
        for key, wanted in expected_metadata.items():
            if response.get(key) != wanted:
                _fail(f"V3-B005 {level}/{condition} response mismatch for {key}")
        action = np.asarray(response.get("action", response.get("actions")), dtype=np.float32)
        future = np.asarray(response.get("video"), dtype=np.uint8)
        if action.shape != (ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(action).all():
            _fail(f"V3-B005 {level}/{condition} action is not finite [32,8]")
        if future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
            _fail(f"V3-B005 {level}/{condition} future is not 33-frame RGB")
        observation_hashes = response.get("observation_hashes")
        if not isinstance(observation_hashes, Mapping):
            _fail(f"V3-B005 {level}/{condition} lacks observation hashes")
        actions[(level, condition)] = action
        futures[(level, condition)] = future
        records.append({
            "request_index": request_index,
            "level_index": level,
            "reference_object_initial_lateral_position_y_m": cell.row[
                "reference_object_initial_lateral_position_y_m"
            ],
            "condition": condition,
            "relation": relation,
            "registered_cell_id": cell.cell_id,
            "prompt": PROMPTS[relation],
            "sampling_seed": 9500,
            "observation_hashes": dict(observation_hashes),
            "action_shape": list(action.shape),
            "action_array_sha256": _array_sha256(action),
            "future_shape": list(future.shape),
            "future_array_sha256": _array_sha256(future),
            "wall_time_s": response.get("wall_time_s"),
        })

    metrics: dict[str, dict[str, Any]] = {}
    passed = True
    for level in PROBE_LEVELS:
        left = actions[(level, "left")]
        repeat = actions[(level, "left_exact_repeat")]
        right = actions[(level, "right")]
        left_future = futures[(level, "left")]
        repeat_future = futures[(level, "left_exact_repeat")]
        right_future = futures[(level, "right")]
        hashes = [
            row["observation_hashes"] for row in records if row["level_index"] == level
        ]
        row = {
            "byte_identical_observations_within_level": all(value == hashes[0] for value in hashes),
            "left_exact_repeat_action_array_equal": bool(np.array_equal(left, repeat)),
            "left_exact_repeat_future_array_equal": bool(np.array_equal(left_future, repeat_future)),
            "left_exact_repeat_action_rms": _rms(left, repeat),
            "left_exact_repeat_future_pixel_mae": _mae(left_future, repeat_future),
            "left_right_action_rms": _rms(left, right),
            "left_right_future_pixel_mae": _mae(left_future, right_future),
        }
        level_passed = (
            row["byte_identical_observations_within_level"]
            and row["left_exact_repeat_action_array_equal"]
            and row["left_exact_repeat_future_array_equal"]
            and row["left_exact_repeat_action_rms"] == 0.0
            and row["left_exact_repeat_future_pixel_mae"] == 0.0
            and row["left_right_action_rms"] > 0.0
            and row["left_right_future_pixel_mae"] > 0.0
        )
        row["passed"] = level_passed
        metrics[f"level{level}"] = row
        passed = passed and level_passed
    return {
        "schema_version": FIXED_OBSERVATION_SCHEMA,
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "status": "passed" if passed else "failed",
        "release_gate_passed": passed,
        "probe_only": True,
        "behavioral_episode_count": 0,
        "model_request_count": 9,
        "probe_levels": list(PROBE_LEVELS),
        "probe_sequence": [list(item) for item in PROBE_SEQUENCE],
        "prospective_artifact_sha256": release.hashes,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "records": records,
        "metrics": metrics,
        "claim_boundary": "Repeatability and prompt sensitivity only; not robot success.",
    }


def _load_observation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        aliases = {
            "observation/image": ("observation/image", "image"),
            "observation/joint_position": (
                "observation/joint_position",
                "joint_position",
                "joint",
            ),
            "observation/gripper_position": (
                "observation/gripper_position",
                "gripper_position",
                "gripper",
            ),
        }
        result = {}
        for target, choices in aliases.items():
            key = next((choice for choice in choices if choice in data), None)
            if key is None:
                _fail(f"{path} lacks {target}")
            result[target] = np.asarray(data[key])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument(
        "--observation",
        action="append",
        required=True,
        help="LEVEL=/absolute/path/to/observation.npz; provide exactly 0, 3, and 6",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=PINNED_SERVER_PORT)
    args = parser.parse_args()
    if args.port != PINNED_SERVER_PORT:
        parser.error("port differs from the fresh V3-B005 Nano server contract")
    release = load_release_bundle(
        args.manifest,
        expected_manifest_sha256=args.manifest_sha256,
    )
    runtime = verify_live_runtime_identity(
        args.runtime_manifest,
        study_root=args.study_root,
        release=release,
    )
    observations: dict[int, dict[str, np.ndarray]] = {}
    observation_sources: dict[str, dict[str, Any]] = {}
    for value in args.observation:
        level_text, separator, path_text = value.partition("=")
        if not separator:
            parser.error("--observation must be LEVEL=PATH")
        level = int(level_text)
        path = Path(path_text).resolve()
        if level in observations:
            parser.error(f"duplicate observation level {level}")
        observations[level] = _load_observation(path)
        observation_sources[str(level)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if set(observations) != set(PROBE_LEVELS):
        parser.error("--observation must specify exactly levels 0, 3, and 6")

    from openpi_client import websocket_client_policy

    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    responses = collect_responses(
        release=release,
        runtime=runtime,
        observations=observations,
        infer=client.infer,
    )
    report = evaluate_responses(release=release, runtime=runtime, responses=responses)
    for level, condition in PROBE_SEQUENCE:
        response = responses[(level, condition)]
        np.save(
            args.output_dir / f"level{level}_{condition}_action.npy",
            np.asarray(response.get("action", response.get("actions"))),
            allow_pickle=False,
        )
        np.save(
            args.output_dir / f"level{level}_{condition}_future.npy",
            np.asarray(response["video"]),
            allow_pickle=False,
        )
    report["observation_sources"] = observation_sources
    output = args.output_dir / "fixed_observation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    validate_fixed_observation_report(output, release=release, runtime=runtime)
    print(json.dumps({
        "path": str(output),
        "sha256": sha256_file(output),
        "status": report["status"],
    }, indent=2, sort_keys=True))
    if not report["release_gate_passed"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
