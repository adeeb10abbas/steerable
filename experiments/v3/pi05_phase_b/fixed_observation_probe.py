#!/usr/bin/env python3
"""Three-request π0.5 exact-repeat and LEFT/RIGHT sensitivity gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy

from experiments.v3.pi05_phase_b.contract import (
    ACTION_CHUNK_STEPS, ACTION_DIM, AMENDMENT_ID, MODEL_ID, PROMPTS, STUDY_ID,
    sha256_file,
)
from experiments.v3.pi05_phase_b.runtime import FIXED_OBSERVATION_SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=8001)
    parser.add_argument("--sampling-seed", type=int, default=9_400_000)
    args = parser.parse_args()
    if not args.fixture.is_file() or sha256_file(args.fixture) != args.fixture_sha256:
        raise ValueError("fixed-observation fixture hash changed")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite fixed-observation evidence: {args.output_dir}")
    with np.load(args.fixture, allow_pickle=False) as loaded:
        observation = {key: loaded[key] for key in loaded.files}
    client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict[str, object]] = {}
    for label, relation in (("left_a", "left"), ("left_b", "left"), ("right", "right")):
        response = client.infer({**observation, "prompt": PROMPTS[relation], "sampling_seed": args.sampling_seed})
        if response.get("v2a010_sampling_seed") != args.sampling_seed:
            raise RuntimeError("π0.5 server did not attest the exact fixed-observation seed")
        action = np.asarray(response["actions"], dtype=np.float32)
        if action.shape != (ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(action).all():
            raise RuntimeError("fixed-observation response is not finite [15,8]")
        path = args.output_dir/f"{label}_actions.npy"
        np.save(path, action, allow_pickle=False)
        arrays[label] = action
        records[label] = {"prompt": PROMPTS[relation], "sampling_seed": args.sampling_seed,
                          "path": str(path.resolve()), "sha256": sha256_file(path),
                          "bytes": path.stat().st_size, "shape": list(action.shape), "dtype": str(action.dtype)}
    repeat = bool(np.array_equal(arrays["left_a"], arrays["left_b"]))
    rms = float(np.sqrt(np.mean((arrays["left_a"].astype(float)-arrays["right"].astype(float))**2)))
    value = {
        "schema_version": FIXED_OBSERVATION_SCHEMA, "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID, "model_id": MODEL_ID,
        "fixture_path": str(args.fixture.resolve()), "fixture_sha256": args.fixture_sha256,
        "left_prompt": PROMPTS["left"], "right_prompt": PROMPTS["right"],
        "sampling_seed": args.sampling_seed, "model_request_count": 3,
        "action_shape": [ACTION_CHUNK_STEPS, ACTION_DIM], "records": records,
        "left_action_sha256": records["left_a"]["sha256"],
        "right_action_sha256": records["right"]["sha256"],
        "left_exact_repeat_bit_identical": repeat,
        "fixed_observation_exact_repeat_passed": repeat,
        "left_right_action_rms": rms,
        "fixed_observation_left_right_prompt_sensitivity_passed": rms > 0,
        "passed": repeat and rms > 0,
    }
    output = args.output_dir/"fixed_observation_gate.json"
    output.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not value["passed"]:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
