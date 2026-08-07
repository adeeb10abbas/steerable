"""Exact Cosmos3 Nano server adapter for V3-E001 fixed observations."""
from __future__ import annotations
from contextvars import ContextVar
from typing import Any
from cosmos_framework.scripts import action_policy_server_robolab as server
from experiments.cosmos import serve_robolab_without_guardrails  # noqa: F401

AUTHORIZED_SEEDS = frozenset(range(9400, 9427))
_request_seed: ContextVar[int | None] = ContextVar("v3e001_request_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer

def _next_seed(self: server.RobolabPolicyService) -> int:
    value = _request_seed.get()
    return int(value) if value is not None else int(_official_next_seed(self))

def _infer(self: server.RobolabPolicyService, obs: dict[str, Any]) -> dict[str, Any]:
    value = obs.get("sampling_seed")
    if isinstance(value, bool) or not isinstance(value, int) or value not in AUTHORIZED_SEEDS:
        raise ValueError("V3-E001 Nano sampling_seed must be an integer in 9400..9426")
    token = _request_seed.set(value)
    try:
        output = _official_infer(self, obs)
    finally:
        _request_seed.reset(token)
    return {**output, "v3e001_sampling_seed": value, "v3e001_probe_only": True}

server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer

if __name__ == "__main__":
    server.main()
