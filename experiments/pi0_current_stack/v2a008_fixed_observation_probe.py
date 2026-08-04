#!/usr/bin/env python3
"""Run V2-A008 exact-repeat and LEFT/RIGHT prompt-sensitivity release gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-manifest", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--sampling-seed", type=int, default=8300000)
    args = parser.parse_args()

    fixture_manifest = json.loads(args.fixture_manifest.read_text())
    if fixture_manifest["fixture_sha256"] != sha256(args.fixture):
        raise ValueError("Fixed-observation fixture hash mismatch")
    registry = json.loads(args.registry.read_text())
    prompts = {
        row["requested_relation"]: row["rendered_prompt"]
        for row in registry["cells"]
        if row["environment_seed"] == 8300
        and row["prompt_family"] == "short_command"
    }
    if set(prompts) != {"left", "right"}:
        raise ValueError("Registry lacks the seed-8300 short-command pair")
    with np.load(args.fixture, allow_pickle=False) as archive:
        observation = {key: archive[key] for key in archive.files}
    client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    for label, relation in (("left_a", "left"), ("left_b", "left"), ("right", "right")):
        response = client.infer(
            {
                **observation,
                "prompt": prompts[relation],
                "sampling_seed": args.sampling_seed,
            }
        )
        if response.get("v2a008_sampling_seed") != args.sampling_seed:
            raise ValueError("Server did not attest the requested sampling seed")
        action = np.asarray(response["actions"], dtype=np.float32)
        if action.shape != (10, 8):
            raise ValueError(f"Unexpected pi0-FAST action shape: {action.shape}")
        path = args.output_dir / f"{label}_actions.npy"
        np.save(path, action, allow_pickle=False)
        arrays[label] = action
        records[label] = {
            "prompt": prompts[relation],
            "sampling_seed": args.sampling_seed,
            "action_path": str(path),
            "action_sha256": sha256(path),
            "shape": list(action.shape),
            "dtype": str(action.dtype),
        }
    repeat_equal = bool(np.array_equal(arrays["left_a"], arrays["left_b"]))
    prompt_rms = float(np.sqrt(np.mean((arrays["left_a"] - arrays["right"]) ** 2)))
    passed = repeat_equal and prompt_rms > 0.0
    manifest = {
        "schema_version": "vla-wam-v2a008-pi0-current-release-probe-v1",
        "fixture_manifest": str(args.fixture_manifest),
        "fixture_sha256": sha256(args.fixture),
        "registry_path": str(args.registry),
        "registry_sha256": sha256(args.registry),
        "records": records,
        "metrics": {
            "left_exact_repeat_bit_identical": repeat_equal,
            "left_vs_right_action_rms": prompt_rms,
        },
        "passed": passed,
    }
    output = args.output_dir / "release_probe.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
