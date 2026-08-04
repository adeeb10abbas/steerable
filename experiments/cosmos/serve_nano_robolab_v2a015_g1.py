#!/usr/bin/env python3
"""Serve the frozen Cosmos3 Nano V2-A015 no-CFG arm.

This is the V2-A015 counterpart to the preserved V2-A011 seed adapter.  It
changes no model, transform, sampler, action-decoding, or future-decoding code.
The only sampling intervention is the command-line guidance scale: V2-A015
requires ``guidance=1``, which is the conditional prediction without a CFG
blend.  The adapter rejects any other guidance/step/shift configuration and
echoes the arm identity in every response.
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

AMENDMENT_ID = "V2-A015"
ARM_ID = "cosmos3_nano_no_cfg_g1"
GUIDANCE = 1.0
BASELINE_GUIDANCE = 3.0
BASELINE_RESULT_ARTIFACT = (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "cosmos3_nano_policy_droid_direct_gate.json"
)
BASELINE_RESULT_SHA256 = "4a6cc1d61593c7ba5272e1707f6bbe51261f7d23438070992bd75fd9e95fdb93"

_request_seed: ContextVar[int | None] = ContextVar("v2a015_request_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer


def _numeric_cli_value(flag: str) -> float:
    """Read exactly one numeric ``--flag value`` or ``--flag=value`` option."""

    values: list[str] = []
    index = 0
    while index < len(sys.argv):
        token = sys.argv[index]
        if token == flag:
            if index + 1 >= len(sys.argv):
                raise SystemExit(f"{flag} requires a value")
            values.append(sys.argv[index + 1])
            index += 2
            continue
        if token.startswith(f"{flag}="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) != 1:
        raise SystemExit(f"V2-A015 requires exactly one {flag} argument")
    try:
        return float(values[0])
    except ValueError as exc:
        raise SystemExit(f"{flag} must be numeric, got {values[0]!r}") from exc


def _require_frozen_sampling_configuration() -> None:
    required = {
        "--guidance": GUIDANCE,
        "--num-steps": 4.0,
        "--shift": 5.0,
    }
    for flag, expected in required.items():
        observed = _numeric_cli_value(flag)
        if observed != expected:
            raise SystemExit(
                f"V2-A015 {ARM_ID} requires {flag}={expected:g}; observed {observed:g}"
            )


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_with_requested_seed(
    self: server.RobolabPolicyService,
    obs: dict[str, Any],
) -> dict[str, Any]:
    value = obs.get("sampling_seed")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("V2-A015 requests require an integer 'sampling_seed'")
    if value not in {8300, 8301, 8302}:
        raise ValueError("V2-A015 sampling_seed must be one of 8300, 8301, 8302")
    token = _request_seed.set(value)
    try:
        output = _official_infer(self, obs)
    finally:
        _request_seed.reset(token)
    return {
        **output,
        "sampling_seed": value,
        "amendment_id": AMENDMENT_ID,
        "arm_id": ARM_ID,
        "guidance": GUIDANCE,
        "baseline_guidance": BASELINE_GUIDANCE,
        "baseline_result_artifact": BASELINE_RESULT_ARTIFACT,
        "baseline_result_sha256": BASELINE_RESULT_SHA256,
    }


server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer_with_requested_seed


if __name__ == "__main__":
    _require_frozen_sampling_configuration()
    server.main()
