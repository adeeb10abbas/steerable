#!/usr/bin/env python3
"""Serve the disclosed π0-FAST revision-bridge release probe.

This is not the missing historical runtime.  It binds the frozen public
checkpoint to the last reachable OpenPI source tree whose checked-in config
names that exact ``openpi-assets-simeval`` payload.  Behavioral execution is
allowed only by the separate v3 amendment and only after its release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
from typing import Any

import jax
import numpy as np

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


EXPECTED_OPENPI_COMMIT = "4cc827620360246dda0fa9d09a9e68269b186ecb"
EXPECTED_OPENPI_TREE = "5c4129878e04359b55903e468922aedbf91843da"
EXPECTED_CONFIG = "pi0_fast_droid_jointpos_polaris"
EXPECTED_CONFIG_SOURCE_SHA256 = (
    "8f40086881304d7cb33f1ab36cad2f36b4e012e508fd246bbc6252912ae7c0ad"
)
EXPECTED_CHECKPOINT_MANIFEST_SHA256 = (
    "47b38eb2f17be802c126ef0a7e93b16693823ee2df62b8007f51bb0514baf5c5"
)
EXPECTED_PROMPTS = {
    "Put the Rubik's cube to the left of the bowl.",
    "Put the Rubik's cube to the right of the bowl.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(root: Path, expression: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", expression], text=True
    ).strip()


def verify_source(root: Path) -> None:
    if git_value(root, "HEAD") != EXPECTED_OPENPI_COMMIT:
        raise ValueError("OpenPI revision-bridge commit changed")
    if git_value(root, "HEAD^{tree}") != EXPECTED_OPENPI_TREE:
        raise ValueError("OpenPI revision-bridge tree changed")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise ValueError("OpenPI revision-bridge worktree is not clean")
    config_source = root / "src/openpi/training/misc/polaris_config.py"
    if sha256(config_source) != EXPECTED_CONFIG_SOURCE_SHA256:
        raise ValueError("checkpoint-matched OpenPI config source changed")
    source = config_source.read_text()
    required = (
        f'name="{EXPECTED_CONFIG}"',
        "gs://openpi-assets-simeval/pi0_fast_droid_jointpos/params",
        "action_dim=8",
        "action_horizon=10",
        "max_token_len=180",
    )
    if not all(fragment in source for fragment in required):
        raise ValueError("OpenPI config no longer binds the frozen checkpoint contract")


def verify_checkpoint(checkpoint: Path, manifest_path: Path) -> None:
    if sha256(manifest_path) != EXPECTED_CHECKPOINT_MANIFEST_SHA256:
        raise ValueError("π0-FAST checkpoint manifest changed")
    manifest = json.loads(manifest_path.read_text())
    checkpoint_record = manifest.get("checkpoint", {})
    files = checkpoint_record.get("files", [])
    if (
        checkpoint_record.get("file_count") != 19
        or checkpoint_record.get("payload_bytes") != 10_844_314_410
        or len(files) != 19
    ):
        raise ValueError("checkpoint is not the frozen 19-file π0-FAST payload")
    observed_paths = set()
    for record in files:
        path = checkpoint / record["path"]
        observed_paths.add(record["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"checkpoint file mismatch: {path}")
    if "assets/droid/norm_stats.json" not in observed_paths:
        raise ValueError("checkpoint-local DROID norm statistics are absent")


class RevisionBridgePolicy:
    """Seed every request and attest the prompt at the tokenizer boundary."""

    def __init__(self, policy: Any) -> None:
        if not hasattr(policy, "_rng") or not hasattr(policy, "_input_transform"):
            raise TypeError("revision bridge requires the pinned JAX OpenPI policy")
        self._policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._policy.metadata,
            "pi0_fast_revision_bridge": "v3a001",
            "openpi_commit": EXPECTED_OPENPI_COMMIT,
            "openpi_tree": EXPECTED_OPENPI_TREE,
            "openpi_config": EXPECTED_CONFIG,
            "sampling_contract": "required_request_field:sampling_seed",
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = dict(obs)
        seed = request.pop("sampling_seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"invalid revision-bridge sampling_seed: {seed!r}")
        prompt = request.get("prompt")
        if prompt not in EXPECTED_PROMPTS:
            raise ValueError(f"prompt is outside the frozen direct pair: {prompt!r}")

        # Inspect the actual pre-model transform.  This proves that the exact
        # request text survives task/default handling and reaches distinct
        # token bytes; it is diagnostic only and does not modify inference.
        transformed = self._policy._input_transform(  # noqa: SLF001
            jax.tree.map(lambda value: value, request)
        )
        tokens = np.asarray(transformed["tokenized_prompt"])
        token_mask = np.asarray(transformed["tokenized_prompt_mask"])
        token_payload = tokens.tobytes() + token_mask.tobytes()

        self._policy._rng = jax.random.key(seed)  # noqa: SLF001
        result = self._policy.infer(request)
        result.update(
            {
                "pi0_fast_revision_bridge": "v3a001",
                "sampling_seed": seed,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "tokenized_prompt_sha256": hashlib.sha256(token_payload).hexdigest(),
            }
        )
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--config", default=EXPECTED_CONFIG)
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()

    verify_source(args.openpi_root)
    if args.config != EXPECTED_CONFIG:
        parser.error(f"revision bridge requires --config {EXPECTED_CONFIG}")
    verify_checkpoint(args.checkpoint, args.checkpoint_manifest)
    policy = policy_config.create_trained_policy(
        training_config.get_config(args.config), str(args.checkpoint)
    )
    wrapped = RevisionBridgePolicy(policy)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=wrapped.metadata,
    )
    logging.info("Serving π0-FAST revision bridge on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
