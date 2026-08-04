#!/usr/bin/env python3
"""Serve current-stack pi0-FAST with an explicit per-request sampling seed.

This adapter is intentionally separate from the historical pi0-FAST evidence.
It is for the post-result V2-A008 current-stack replication only.
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

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as training_config


EXPECTED_OPENPI_COMMIT = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
EXPECTED_CONFIG = "pi0_fast_droid_jointpos_polaris"


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(checkpoint: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    checkpoint_record = manifest.get("checkpoint", {})
    files = checkpoint_record.get("files", [])
    if (
        checkpoint_record.get("file_count") != 19
        or checkpoint_record.get("payload_bytes") != 10_844_314_410
        or len(files) != 19
    ):
        raise ValueError("Checkpoint manifest is not the frozen 19-file pi0-FAST record")
    for record in files:
        path = checkpoint / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Checkpoint file does not match the frozen manifest: {path}")


class PerRequestSeedPolicy:
    """Set the JAX policy RNG from a required request field before inference."""

    def __init__(self, policy: Any) -> None:
        if not hasattr(policy, "_rng"):
            raise TypeError("V2-A008 requires a JAX OpenPI policy with an explicit RNG")
        self._policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._policy.metadata,
            "v2a008_sampling_contract": "required_request_field:sampling_seed",
            "v2a008_openpi_commit": EXPECTED_OPENPI_COMMIT,
            "v2a008_config": EXPECTED_CONFIG,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = dict(obs)
        if "sampling_seed" not in request:
            raise ValueError("V2-A008 requests must include integer sampling_seed")
        seed = request.pop("sampling_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid V2-A008 sampling_seed: {seed!r}")
        self._policy._rng = jax.random.key(seed)  # noqa: SLF001 - frozen adapter contract
        result = self._policy.infer(request)
        result["v2a008_sampling_seed"] = seed
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--config", default=EXPECTED_CONFIG)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    actual_commit = git_head(args.openpi_root)
    if actual_commit != EXPECTED_OPENPI_COMMIT:
        parser.error(
            f"Current-stack OpenPI commit changed: {actual_commit}; "
            f"expected {EXPECTED_OPENPI_COMMIT}"
        )
    if args.config != EXPECTED_CONFIG:
        parser.error(f"V2-A008 requires --config {EXPECTED_CONFIG}")
    if not args.checkpoint.is_dir():
        parser.error(f"Checkpoint directory is absent: {args.checkpoint}")
    verify_checkpoint(args.checkpoint, args.checkpoint_manifest)

    policy = policy_config.create_trained_policy(
        training_config.get_config(args.config), str(args.checkpoint)
    )
    seeded = PerRequestSeedPolicy(policy)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=seeded,
        host="0.0.0.0",
        port=args.port,
        metadata=seeded.metadata,
    )
    logging.info("Serving V2-A008 current-stack pi0-FAST on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
