#!/usr/bin/env python3
"""Serve V2-A010 pi0.5 with a required per-request JAX sampling seed."""

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


OPENPI_COMMIT = "c23745b5ad24e98f66967ea795a07b2588ed6c79"
CONFIG = "pi05_droid_jointpos_polaris"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checkpoint(root: Path, manifest_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema_version")
        != "vla-wam-v2a010-pi05-current-checkpoint-manifest-v1"
        or manifest.get("status") != "complete_sha256_hashed_before_model_load"
        or manifest.get("file_count") != 26
        or manifest.get("payload_bytes") != 12_434_530_510
        or len(manifest.get("files", [])) != 26
    ):
        raise ValueError("Not the frozen V2-A010 pi0.5 checkpoint manifest")
    for record in manifest["files"]:
        path = root / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Checkpoint payload mismatch: {path}")


class SeededPi05Policy:
    def __init__(self, policy: Any) -> None:
        if not hasattr(policy, "_rng"):
            raise TypeError("V2-A010 requires a JAX OpenPI policy")
        self.policy = policy

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self.policy.metadata,
            "v2a010_sampling_contract": "required_request_field:sampling_seed",
            "v2a010_openpi_commit": OPENPI_COMMIT,
            "v2a010_config": CONFIG,
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = dict(obs)
        seed = request.pop("sampling_seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"Invalid or missing V2-A010 sampling_seed: {seed!r}")
        self.policy._rng = jax.random.key(seed)  # noqa: SLF001 - frozen adapter seam
        result = self.policy.infer(request)
        result["v2a010_sampling_seed"] = seed
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    head = subprocess.check_output(
        ["git", "-C", str(args.openpi_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != OPENPI_COMMIT or args.config != CONFIG:
        parser.error(f"V2-A010 requires OpenPI {OPENPI_COMMIT} and config {CONFIG}")
    verify_checkpoint(args.checkpoint, args.checkpoint_manifest)
    policy = policy_config.create_trained_policy(
        training_config.get_config(CONFIG), str(args.checkpoint)
    )
    seeded = SeededPi05Policy(policy)
    websocket_policy_server.WebsocketPolicyServer(
        policy=seeded, host="0.0.0.0", port=args.port, metadata=seeded.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
