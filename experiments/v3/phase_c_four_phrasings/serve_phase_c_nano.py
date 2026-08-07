"""Narrow seed/prompt overlay for the released Nano V3-C001 server."""

from __future__ import annotations

from contextvars import ContextVar
import threading
from typing import Any

from cosmos_framework.scripts import action_policy_server_robolab as server

from experiments.cosmos import serve_robolab_without_guardrails  # noqa: F401
from experiments.v3.phase_c_four_phrasings.contract import PROMPTS, SEEDS


AUTHORIZED_PROMPTS = frozenset(
    prompt for relations in PROMPTS.values() for prompt in relations.values()
)
AUTHORIZED_SEEDS = frozenset(SEEDS)
_request_seed: ContextVar[int | None] = ContextVar("v3c001_nano_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer
_request_lock = threading.Lock()


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_registered_prompt(
    self: server.RobolabPolicyService,
    obs: dict[str, Any],
) -> dict[str, Any]:
    seed = obs.get("sampling_seed")
    prompt = obs.get("prompt")
    if type(seed) is not int or seed not in AUTHORIZED_SEEDS:
        raise ValueError("V3-C001 Nano sampling_seed must be one of 8500..8519")
    if not isinstance(prompt, str) or prompt not in AUTHORIZED_PROMPTS:
        raise ValueError("V3-C001 Nano prompt must match one exact registered string")
    with _request_lock:
        token = _request_seed.set(seed)
        try:
            output = _official_infer(self, obs)
        finally:
            _request_seed.reset(token)
    return {
        **output,
        "experiment_id": "V3-C001",
        "sampling_seed": seed,
        "prompt": prompt,
    }


server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer_registered_prompt


if __name__ == "__main__":
    server.main()
