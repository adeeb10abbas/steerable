#!/usr/bin/env python3
"""Create a hash-bound live-runtime identity before any Cosmos v3 request."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from experiments.v3.cosmos_droid.contract import (
    EMPTY_DIFF_SHA256,
    MODEL_CONTRACTS,
    STUDY_ID,
    ContractError,
    compute_adapter_contract_hash,
    load_authorized_pair,
    sha256_file,
    verify_repository_pins,
    verify_runtime_identity,
)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _expected_checkpoint_files(study_root: Path, model_id: str) -> dict[str, str]:
    spec = MODEL_CONTRACTS[model_id]
    registry = json.loads((study_root / spec["v2_registry"]).read_text())
    checkpoint = registry["checkpoint"]
    if model_id == "cosmos3_nano_policy_droid":
        return {name: row["sha256"] for name, row in checkpoint["files"].items()}
    result = {
        "checkpoint.json": checkpoint["checkpoint_json_sha256"],
        "config.json": checkpoint["config_json_sha256"],
        "transformer/config.json": checkpoint["transformer_config_sha256"],
    }
    result.update(checkpoint["weight_file_sha256"])
    return result


def _verify_checkpoint(study_root: Path, model_id: str, checkpoint_dir: Path) -> str:
    expected = _expected_checkpoint_files(study_root, model_id)
    inventory = []
    for relative, expected_hash in sorted(expected.items()):
        path = checkpoint_dir / relative
        if not path.is_file():
            raise ContractError(f"checkpoint is missing registered file: {relative}")
        observed = sha256_file(path)
        if observed != expected_hash:
            raise ContractError(f"checkpoint hash mismatch: {relative}")
        inventory.append({"path": relative, "sha256": observed, "bytes": path.stat().st_size})
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_runtime_identity(
    *, study_root: Path, model_id: str, seed: int, checkpoint_dir: Path,
    external_repo: Path, robolab_repo: Path, environment_lock: Path, simulator_version: str,
    renderer_backend: str,
) -> dict[str, Any]:
    root = study_root.resolve()
    load_authorized_pair(root, model_id, seed)
    pins = verify_repository_pins(root, model_id)
    spec = MODEL_CONTRACTS[model_id]
    head = _git(external_repo, "rev-parse", "HEAD")
    if head != spec["server_repository_commit"]:
        raise ContractError("external Cosmos repository commit does not match the v2 stack")
    dirt = _git(external_repo, "status", "--porcelain", "--untracked-files=all")
    if dirt:
        raise ContractError("external Cosmos repository must be clean")
    robolab_head = _git(robolab_repo, "rev-parse", "HEAD")
    if robolab_head != spec["robolab_repository_commit"]:
        raise ContractError("RoboLab repository commit does not match the v2 stack")
    robolab_dirt = _git(robolab_repo, "status", "--porcelain", "--untracked-files=all")
    if robolab_dirt:
        raise ContractError("RoboLab repository must be clean")
    if not environment_lock.is_file():
        raise ContractError("environment lock file does not exist")
    payload: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3-cosmos-runtime-identity-v1",
        "study_id": STUDY_ID,
        "model_id": model_id,
        "checkpoint_identifier": spec["checkpoint_id"],
        "checkpoint_revision": spec["checkpoint_revision"],
        "checkpoint_sha256": _verify_checkpoint(root, model_id, checkpoint_dir.resolve()),
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": head,
        "external_repository_diff_hash": EMPTY_DIFF_SHA256,
        "simulator_repository_commit": robolab_head,
        "simulator_repository_diff_hash": EMPTY_DIFF_SHA256,
        "environment_lock_hash": sha256_file(environment_lock.resolve()),
        "adapter_contract_hash": compute_adapter_contract_hash(root),
        "simulator_version": simulator_version,
        "renderer_backend": renderer_backend,
        "repository_pins": pins,
    }
    payload["runtime_identity_sha256"] = hashlib.sha256(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--model-id", choices=sorted(MODEL_CONTRACTS), required=True)
    parser.add_argument("--seed", type=int, default=8303)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--external-repo", type=Path, required=True)
    parser.add_argument("--robolab-repo", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--simulator-version", required=True)
    parser.add_argument("--renderer-backend", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_runtime_identity(**{key: value for key, value in vars(args).items() if key != "output"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    verify_runtime_identity(args.study_root, args.model_id, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
