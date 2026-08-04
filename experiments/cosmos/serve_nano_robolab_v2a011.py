"""Serve pinned Cosmos3 Nano DROID with explicit per-request sampling seeds.

The official 411d25b RoboLab server samples from a server-side RNG and does not
consume the client's ``sampling_seed`` field. V2-A011 requires the paired seed
to be injected and recorded, so this wrapper routes that integer into the
official generator without changing checkpoint loading, transforms, sampling,
action decoding, or generated-future decoding. It also reuses the existing
study wrapper that disables only the separately gated guardrail download.
"""

from __future__ import annotations

import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cosmos_framework.scripts import action_policy_server_robolab as server

from experiments.cosmos import serve_robolab_without_guardrails  # noqa: F401

_request_seed: ContextVar[int | None] = ContextVar("v2a011_request_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_with_requested_seed(
    self: server.RobolabPolicyService,
    obs: dict[str, Any],
) -> dict[str, Any]:
    value = obs.get("sampling_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("V2-A011 requests require an integer 'sampling_seed'")
    if value not in {8300, 8301, 8302}:
        raise ValueError("V2-A011 sampling_seed must be one of 8300, 8301, 8302")
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
