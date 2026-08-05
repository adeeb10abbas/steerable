"""Serve the exact V2-A011 Nano stack for the preregistered v3 seed range.

The official inference method and seed injection mechanism are unchanged.  The
only prospective extension is accepting the frozen v3 Phase-A seeds 8303..8329.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from cosmos_framework.scripts import action_policy_server_robolab as server

from experiments.cosmos import serve_robolab_without_guardrails  # noqa: F401


AUTHORIZED_SEEDS = frozenset(range(8303, 8330))
_request_seed: ContextVar[int | None] = ContextVar("v3_request_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_with_requested_seed(
    self: server.RobolabPolicyService, obs: dict[str, Any]
) -> dict[str, Any]:
    value = obs.get("sampling_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("v3 Nano requests require an integer sampling_seed")
    if value not in AUTHORIZED_SEEDS:
        raise ValueError("v3 Nano sampling_seed must be in 8303..8329")
    token = _request_seed.set(value)
    try:
        output = _official_infer(self, obs)
    finally:
        _request_seed.reset(token)
    return {**output, "sampling_seed": value}


server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer_with_requested_seed


if __name__ == "__main__":
    server.main()
