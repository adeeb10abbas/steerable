#!/usr/bin/env python3
"""Exact arm-wise fixed-observation gates for V3-B008 and V3-B009."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    CONFIG,
    MODEL_ID,
    STUDY_ID,
    ContractError,
    ReleaseBundle,
    canonical_json_bytes,
    load_release,
    load_runtime,
    sha256_file,
)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _observation_hashes(value: Mapping[str, np.ndarray]) -> dict[str, str]:
    return {key: _array_sha256(np.asarray(array)) for key, array in value.items()}


def _load_observation(path: Path) -> dict[str, np.ndarray]:
    aliases = {
        "observation/image": ("observation/image", "image"),
        "observation/joint_position": ("observation/joint_position", "joint_position", "joint"),
        "observation/gripper_position": ("observation/gripper_position", "gripper_position", "gripper"),
    }
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for target, choices in aliases.items():
            source = next((choice for choice in choices if choice in data), None)
            if source is None:
                raise ContractError(f"{path} lacks {target}")
            result[target] = np.asarray(data[source])
    return result


def _rms(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(first.astype(np.float64) - second.astype(np.float64)))))


def _mae(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.mean(np.abs(first.astype(np.float64) - second.astype(np.float64))))


def collect(
    *, release: ReleaseBundle, runtime: Mapping[str, Any], observations: Mapping[str, Mapping[str, np.ndarray]], infer: Any
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    responses: dict[tuple[str, str], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    request_index = 0
    for arm in release.config["arms"]:
        source = observations[arm]
        hashes = _observation_hashes(source)
        for condition in ("left", "left_exact_repeat", "right"):
            relation = "right" if condition == "right" else "left"
            cell = release.probe_cell(arm, relation)
            request = {key: np.array(value, copy=True) for key, value in source.items()}
            request.update({
                "nano_tier_b_server_mode": "probe_only",
                "amendment_id": release.amendment_id,
                "probe_request_index": request_index,
                "probe_arm": arm,
                "probe_condition": condition,
                "registered_cell_id": cell.cell_id,
                "sampling_seed": cell.seed,
                "prompt": cell.row["prompt"],
                "release_fingerprint_sha256": release.release_fingerprint(cell),
                "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            })
            started = time.perf_counter()
            response = dict(infer(request))
            wall = time.perf_counter() - started
            expected_metadata = {
                "nano_tier_b_live_stack": "isolated_v3b008_v3b009_v1",
                "nano_tier_b_server_mode": "probe_only",
                "amendment_id": release.amendment_id,
                "registered_cell_id": cell.cell_id,
                "sampling_seed": cell.seed,
                "request_index": request_index,
                "probe_request_index": request_index,
                "probe_arm": arm,
                "probe_condition": condition,
                "release_fingerprint_sha256": release.release_fingerprint(cell),
                "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            }
            for key, wanted in expected_metadata.items():
                if response.get(key) != wanted:
                    raise ContractError(f"fixed-observation response mismatch for {key}")
            action = np.asarray(response.get("action", response.get("actions")))
            future = np.asarray(response.get("video"))
            if action.shape != (ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(action).all():
                raise ContractError(f"{arm}/{condition} action is not finite [32,8]")
            if future.size == 0 or not np.issubdtype(future.dtype, np.number) or not np.isfinite(future).all():
                raise ContractError(f"{arm}/{condition} lacks a finite decoded future")
            responses[(arm, condition)] = {"action": action, "future": future}
            records.append({
                "request_index": request_index,
                "arm": arm,
                "condition": condition,
                "relation": relation,
                "registered_cell_id": cell.cell_id,
                "prompt": cell.row["prompt"],
                "sampling_seed": cell.seed,
                "observation_hashes": hashes,
                "action_shape": list(action.shape),
                "action_array_sha256": _array_sha256(action),
                "future_shape": list(future.shape),
                "future_array_sha256": _array_sha256(future),
                "wall_time_s": wall,
            })
            request_index += 1
    return responses, records


def evaluate(*, release: ReleaseBundle, runtime: Mapping[str, Any], responses: Mapping[tuple[str, str], Mapping[str, np.ndarray]], records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    passed = True
    for arm in release.config["arms"]:
        left = responses[(arm, "left")]
        repeat = responses[(arm, "left_exact_repeat")]
        right = responses[(arm, "right")]
        arm_records = [row for row in records if row["arm"] == arm]
        row = {
            "byte_identical_observations_within_arm": all(
                record["observation_hashes"] == arm_records[0]["observation_hashes"] for record in arm_records
            ),
            "left_exact_repeat_action_array_equal": bool(np.array_equal(left["action"], repeat["action"])),
            "left_exact_repeat_future_array_equal": bool(np.array_equal(left["future"], repeat["future"])),
            "left_exact_repeat_action_rms": _rms(left["action"], repeat["action"]),
            "left_exact_repeat_future_pixel_mae": _mae(left["future"], repeat["future"]),
            "left_right_action_rms": _rms(left["action"], right["action"]),
            "left_right_future_pixel_mae": _mae(left["future"], right["future"]),
        }
        row["passed"] = bool(
            row["byte_identical_observations_within_arm"]
            and row["left_exact_repeat_action_array_equal"]
            and row["left_exact_repeat_future_array_equal"]
            and row["left_exact_repeat_action_rms"] == 0.0
            and row["left_exact_repeat_future_pixel_mae"] == 0.0
            and row["left_right_action_rms"] > 0.0
            and row["left_right_future_pixel_mae"] > 0.0
        )
        passed = passed and row["passed"]
        metrics[arm] = row
    return {
        "schema_version": "vla-wam-shared-v3b008-v3b009-nano-fixed-observation-v1",
        "study_id": STUDY_ID,
        "amendment_id": release.amendment_id,
        "model_id": MODEL_ID,
        "status": "passed" if passed else "failed",
        "release_gate_passed": passed,
        "probe_only": True,
        "behavioral_episode_count": 0,
        "model_request_count": len(records),
        "release_manifest_sha256": release.manifest_sha256,
        "cells_sha256": release.cells_sha256,
        "model_blind_gate_sha256": release.config["gate_sha256"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "server_port": release.config["port"],
        "probe_order": [
            [arm, condition]
            for arm in release.config["arms"]
            for condition in ("left", "left_exact_repeat", "right")
        ],
        "records": records,
        "metrics": metrics,
        "claim_boundary": "Deterministic repeatability and prompt sensitivity only; no robot behavior or success evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--amendment-id", choices=tuple(CONFIG), required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--observation", action="append", required=True, help="ARM=/absolute/path.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    release = load_release(args.study_root, args.amendment_id, args.release_manifest)
    runtime = load_runtime(args.runtime_manifest, study_root=args.study_root, release=release)
    port = release.config["port"] if args.port is None else args.port
    if port != release.config["port"]:
        parser.error("port differs from the amendment-isolated server contract")
    observations: dict[str, dict[str, np.ndarray]] = {}
    observation_sources: dict[str, dict[str, Any]] = {}
    for value in args.observation:
        arm, separator, path_text = value.partition("=")
        if not separator or arm in observations:
            parser.error("--observation must provide each ARM=PATH exactly once")
        path = Path(path_text).resolve()
        observations[arm] = _load_observation(path)
        observation_sources[arm] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    if set(observations) != set(release.config["arms"]):
        parser.error(f"observations must cover exactly {release.config['arms']}")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, port)
    responses, records = collect(release=release, runtime=runtime, observations=observations, infer=client.infer)
    report = evaluate(release=release, runtime=runtime, responses=responses, records=records)
    for (arm, condition), response in responses.items():
        np.save(args.output_dir / f"{arm}_{condition}_action.npy", response["action"], allow_pickle=False)
        np.save(args.output_dir / f"{arm}_{condition}_future.npy", response["future"], allow_pickle=False)
    report["observation_sources"] = observation_sources
    output = args.output_dir / "fixed_observation_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(output), "sha256": sha256_file(output), "status": report["status"]}, indent=2, sort_keys=True))
    if not report["release_gate_passed"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()

