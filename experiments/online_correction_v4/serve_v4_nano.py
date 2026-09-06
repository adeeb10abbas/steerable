#!/usr/bin/env python3
"""Serve the pinned Cosmos3 Nano policy for registered V4 qualification seeds."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from contextvars import ContextVar
from typing import Any


SEED_REGISTRY_SCHEMA = "v4-nano-policy-seed-registry-v1"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"
ALLOWED_SCOPES = {
    "g4_policy_session_only",
    "g7_engineering_pilot",
    "g8_miniature_campaign",
    "released_c7",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pop_custom_args(argv: list[str]) -> tuple[list[str], Path, str]:
    cleaned = [argv[0]]
    registry: Path | None = None
    expected_sha256: str | None = None
    index = 1
    while index < len(argv):
        item = argv[index]
        if item in {"--allowed-seeds-registry", "--allowed-seeds-sha256"}:
            if index + 1 >= len(argv):
                raise RuntimeError(f"{item} requires a value")
            value = argv[index + 1]
            if item == "--allowed-seeds-registry":
                registry = Path(value).resolve()
            else:
                expected_sha256 = value
            index += 2
            continue
        cleaned.append(item)
        index += 1
    if registry is None or expected_sha256 is None:
        raise RuntimeError(
            "--allowed-seeds-registry and --allowed-seeds-sha256 are required"
        )
    return cleaned, registry, expected_sha256


def _load_allowed_seeds(path: Path, expected_sha256: str) -> tuple[frozenset[int], str]:
    if _sha256(path) != expected_sha256:
        raise RuntimeError("V4 Nano seed registry digest mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SEED_REGISTRY_SCHEMA:
        raise RuntimeError("V4 Nano seed registry schema mismatch")
    scope = payload.get("scope")
    if scope not in ALLOWED_SCOPES:
        raise RuntimeError(f"unsupported V4 Nano seed registry scope: {scope!r}")
    if payload.get("checkpoint_revision") != CHECKPOINT_REVISION:
        raise RuntimeError("V4 Nano seed registry checkpoint revision mismatch")
    seeds = payload.get("allowed_sampling_seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(type(seed) is not int or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise RuntimeError("V4 Nano seed registry must contain unique nonnegative integers")
    return frozenset(seeds), str(scope)


sys.argv, _seed_registry_path, _seed_registry_sha256 = _pop_custom_args(sys.argv)
_allowed_seeds, _seed_scope = _load_allowed_seeds(
    _seed_registry_path,
    _seed_registry_sha256,
)

# The pinned checkpoint references its exact Wan2.2 VAE through ``uvx hf``.
# Preserve the already-qualified resolver and cache independently of pod HOME.
_python_bin = Path(sys.executable).resolve().parent
os.environ["PATH"] = f"{_python_bin}:{os.environ.get('PATH', '')}"
os.environ.setdefault(
    "HF_HOME",
    "/data/users/ali/vla_wam/cache/huggingface-cosmos",
)

from cosmos_framework.scripts import action_policy_server_robolab as server  # noqa: E402


_official_build_setup_args = server.RobolabPolicyService._build_setup_args
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer
_request_seed: ContextVar[int | None] = ContextVar("v4_nano_request_seed", default=None)
_lock = threading.Lock()


def _build_setup_args_without_guardrails(
    self: server.RobolabPolicyService,
    args: server.RobolabServerArgs,
) -> Any:
    setup_args = _official_build_setup_args(self, args)
    return setup_args.model_copy(update={"guardrails": False})


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_v4(
    self: server.RobolabPolicyService,
    obs: dict[str, Any],
) -> dict[str, Any]:
    seed = obs.get("sampling_seed")
    if type(seed) is not int or seed not in _allowed_seeds:
        raise ValueError("V4 Nano sampling_seed is not present in the bound seed registry")
    prompt = obs.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise TypeError("V4 Nano requests require a nonempty static prompt")
    action_step_start = obs.get("action_step_start", 0)
    if type(action_step_start) is not int or action_step_start < 0:
        raise TypeError("V4 Nano action_step_start must be a nonnegative integer")
    with _lock:
        token = _request_seed.set(seed)
        try:
            output = _official_infer(self, obs)
        finally:
            _request_seed.reset(token)
    return {
        **output,
        "sampling_seed": seed,
        "v4_seed_registry_sha256": _seed_registry_sha256,
        "v4_seed_scope": _seed_scope,
    }


server.RobolabPolicyService._build_setup_args = _build_setup_args_without_guardrails
server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer_v4


if __name__ == "__main__":
    server.main()
