#!/usr/bin/env python3
"""Run the fail-closed π0-FAST revision-bridge sensitivity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from openpi_client import websocket_client_policy


PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
EXPECTED_FIXTURE_SHA256 = (
    "ce8be012347718a162bf0d92ba2fb71a01c570a3462d72ef2c16a86082131778"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=8010)
    parser.add_argument("--sampling-seed", type=int, default=8310000)
    args = parser.parse_args()

    if sha256(args.fixture) != EXPECTED_FIXTURE_SHA256:
        raise ValueError("fixed-observation fixture bytes changed")
    with np.load(args.fixture, allow_pickle=False) as archive:
        observation = {key: archive[key] for key in archive.files}
    client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host, args.remote_port
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    arrays: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    for label, relation in (("left_a", "left"), ("left_b", "left"), ("right", "right")):
        prompt = PROMPTS[relation]
        response = client.infer(
            {**observation, "prompt": prompt, "sampling_seed": args.sampling_seed}
        )
        if response.get("pi0_fast_revision_bridge") != "v3a001":
            raise ValueError("server did not attest the revision-bridge identity")
        if response.get("sampling_seed") != args.sampling_seed:
            raise ValueError("server did not attest the exact sampling seed")
        expected_prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        if response.get("prompt_sha256") != expected_prompt_hash:
            raise ValueError("prompt bytes changed before inference")
        action = np.asarray(response["actions"], dtype=np.float32)
        if action.shape != (10, 8) or not np.isfinite(action).all():
            raise ValueError(f"invalid π0-FAST action tensor: {action.shape}")
        path = args.output_dir / f"{label}_actions.npy"
        np.save(path, action, allow_pickle=False)
        arrays[label] = action
        records[label] = {
            "prompt": prompt,
            "prompt_sha256": expected_prompt_hash,
            "tokenized_prompt_sha256": response["tokenized_prompt_sha256"],
            "sampling_seed": args.sampling_seed,
            "action_path": str(path),
            "action_sha256": sha256(path),
            "shape": list(action.shape),
            "dtype": str(action.dtype),
        }

    repeat_equal = bool(np.array_equal(arrays["left_a"], arrays["left_b"]))
    action_equal = bool(np.array_equal(arrays["left_a"], arrays["right"]))
    prompt_tokens_differ = (
        records["left_a"]["tokenized_prompt_sha256"]
        != records["right"]["tokenized_prompt_sha256"]
    )
    prompt_rms = float(np.sqrt(np.mean((arrays["left_a"] - arrays["right"]) ** 2)))
    passed = repeat_equal and prompt_tokens_differ and not action_equal and prompt_rms > 0.0
    manifest = {
        "schema_version": "vla-wam-shared-v3-pi0-fast-revision-bridge-gate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": "pi0_fast_revision_bridge_v3a001",
        "status": "passed" if passed else "failed",
        "fixture_path": str(args.fixture),
        "fixture_sha256": sha256(args.fixture),
        "records": records,
        "metrics": {
            "left_exact_repeat_bit_identical": repeat_equal,
            "left_right_token_bytes_differ": prompt_tokens_differ,
            "left_right_actions_bit_identical": action_equal,
            "left_right_action_rms": prompt_rms,
        },
        "behavioral_release": passed,
        "claim_boundary": (
            "Three no-environment model requests; not a behavioral episode or "
            "evidence of task competence."
        ),
    }
    output = args.output_dir / "release_gate.json"
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
