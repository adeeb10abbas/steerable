#!/usr/bin/env python3
"""Serve the disclosed old-name public-config π0-FAST bridge."""

from __future__ import annotations

import argparse
import dataclasses
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


EXPECTED_OPENPI_COMMIT = "235044ed8a1502c0a18338eedc5d7adfe705af05"
EXPECTED_OPENPI_TREE = "03a4387bedbc0fa1467c367c60fc24e28b61ec6c"
EXPECTED_CONFIG = "pi0_fast_droid_jointpos"
EXPECTED_CONFIG_SOURCE_SHA256 = (
    "96ddf85ff5903e68acca310d7af9d9d093373f7dc060fe94dbb379c6828481ad"
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


def verify_import_origin(root: Path) -> None:
    expected = root.resolve()
    imported = {
        "openpi.policies.policy_config": Path(policy_config.__file__).resolve(),
        "openpi.training.config": Path(training_config.__file__).resolve(),
    }
    for module, path in imported.items():
        if not path.is_relative_to(expected):
            raise ValueError(
                f"{module} imported from {path}, outside pinned OpenPI tree {expected}"
            )


def verify_source(root: Path) -> None:
    if git_value(root, "HEAD") != EXPECTED_OPENPI_COMMIT:
        raise ValueError("old-name bridge OpenPI commit changed")
    if git_value(root, "HEAD^{tree}") != EXPECTED_OPENPI_TREE:
        raise ValueError("old-name bridge OpenPI tree changed")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1"], text=True
    )
    if status:
        raise ValueError("old-name OpenPI worktree is not clean")
    config_source = root / "src/openpi/training/misc/polaris_config.py"
    if sha256(config_source) != EXPECTED_CONFIG_SOURCE_SHA256:
        raise ValueError("old-name OpenPI config source changed")
    source = config_source.read_text()
    required = (
        f'name="{EXPECTED_CONFIG}"',
        "Pi0FASTConfig(action_dim=8, action_horizon=10)",
        "data=SimpleDataConfig(",
        "DroidInputs(model_type=ModelType.PI0_FAST)",
        "AbsoluteActions(_transforms.make_bool_mask(7, -1))",
        "droid_policy.DroidOutputs()",
    )
    if not all(fragment in source for fragment in required):
        raise ValueError("old-name OpenPI config contract changed")


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


class OldNameConfigBridgePolicy:
    """Seed each request and attest the actual tokenizer-boundary prompt."""

    def __init__(self, policy: Any, checkpoint_assets_dir: Path) -> None:
        if not hasattr(policy, "_rng") or not hasattr(policy, "_input_transform"):
            raise TypeError("old-name bridge requires the pinned JAX OpenPI policy")
        self._policy = policy
        self._checkpoint_assets_dir = checkpoint_assets_dir.resolve()

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **self._policy.metadata,
            "pi0_fast_old_name_config_bridge": "v3a002",
            "openpi_commit": EXPECTED_OPENPI_COMMIT,
            "openpi_tree": EXPECTED_OPENPI_TREE,
            "openpi_config": EXPECTED_CONFIG,
            "max_token_len": 250,
            "checkpoint_assets_override": str(self._checkpoint_assets_dir),
            "checkpoint_assets_rule": "checkpoint_local_assets_only",
            "sampling_contract": "required_request_field:sampling_seed",
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        request = dict(obs)
        seed = request.pop("sampling_seed", None)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"invalid old-name bridge sampling_seed: {seed!r}")
        prompt = request.get("prompt")
        if prompt not in EXPECTED_PROMPTS:
            raise ValueError(f"prompt is outside the frozen direct pair: {prompt!r}")
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
                "pi0_fast_old_name_config_bridge": "v3a002",
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
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    verify_import_origin(args.openpi_root)
    verify_source(args.openpi_root)
    if args.config != EXPECTED_CONFIG:
        parser.error(f"old-name bridge requires --config {EXPECTED_CONFIG}")
    verify_checkpoint(args.checkpoint, args.checkpoint_manifest)
    config = training_config.get_config(args.config)
    if (
        config.model.action_dim != 8
        or config.model.action_horizon != 10
        or config.model.max_token_len != 250
        or type(config.data).__name__ != "SimpleDataConfig"
        or config.data.assets.asset_id != "droid"
    ):
        raise ValueError("old-name π0-FAST config semantics changed")
    checkpoint_assets = (args.checkpoint / "assets").resolve()
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=dataclasses.replace(
                config.data.assets,
                assets_dir=str(checkpoint_assets),
            ),
        ),
    )
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    wrapped = OldNameConfigBridgePolicy(policy, checkpoint_assets)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=wrapped,
        host="0.0.0.0",
        port=args.port,
        metadata=wrapped.metadata,
    )
    logging.info("Serving old-name-config π0-FAST bridge on port %d", args.port)
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
