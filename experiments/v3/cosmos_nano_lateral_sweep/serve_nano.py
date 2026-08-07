#!/usr/bin/env python3
"""Fresh Nano server envelope for the V3-B005 probe and behavior phases."""

from __future__ import annotations

import os
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from experiments.v3.cosmos_nano_lateral_sweep.live_support import (
    PINNED_CHECKPOINT_PATH,
    PROBE_SEQUENCE,
    authorize_behavior_request,
    authorize_probe_request,
    server_response_metadata,
    validate_pinned_server_cli,
    verify_behavioral_release_gate,
    verify_live_runtime_identity,
)
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    load_release_bundle,
    sha256_file,
)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required before starting the V3-B005 Nano server")
    return value


STUDY_ROOT = Path(_required("VLA_WAM_STUDY_ROOT")).resolve()
RELEASE_MANIFEST = Path(_required("VLA_WAM_V3B005_MANIFEST")).resolve()
RUNTIME_MANIFEST = Path(_required("VLA_WAM_V3B005_RUNTIME_MANIFEST")).resolve()
MODE = _required("VLA_WAM_V3B005_MODE")
if MODE not in {"probe_only", "behavior_only"}:
    raise RuntimeError("VLA_WAM_V3B005_MODE must be probe_only or behavior_only")

RELEASE = load_release_bundle(
    RELEASE_MANIFEST,
    expected_manifest_sha256=_required("VLA_WAM_V3B005_MANIFEST_SHA256"),
)
RUNTIME = verify_live_runtime_identity(
    RUNTIME_MANIFEST,
    study_root=STUDY_ROOT,
    release=RELEASE,
)
SERVER_CLI = validate_pinned_server_cli(sys.argv[1:])
if not PINNED_CHECKPOINT_PATH.is_dir():
    raise RuntimeError(f"pinned Nano checkpoint directory is missing: {PINNED_CHECKPOINT_PATH}")

BEHAVIORAL_RELEASE_GATE_SHA256: str | None = None
if MODE == "behavior_only":
    gate_path = Path(_required("VLA_WAM_V3B005_BEHAVIORAL_RELEASE_GATE")).resolve()
    verify_behavioral_release_gate(gate_path, release=RELEASE, runtime=RUNTIME)
    BEHAVIORAL_RELEASE_GATE_SHA256 = sha256_file(gate_path)
elif os.environ.get("VLA_WAM_V3B005_BEHAVIORAL_RELEASE_GATE"):
    raise RuntimeError("probe-only mode must not be presented as behaviorally released")

# Import the official implementation only after every startup binding passes.
from cosmos_framework.scripts import action_policy_server_robolab as server  # noqa: E402
from experiments.cosmos import serve_robolab_without_guardrails  # noqa: E402,F401


_request_seed: ContextVar[int | None] = ContextVar("v3b005_request_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer
_lock = threading.Lock()
_probe_request_count = 0
_probe_observation_hashes: dict[int, dict[str, Any]] = {}
_behavior_request_counts: dict[str, int] = {}


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_v3b005(
    self: server.RobolabPolicyService,
    obs: dict[str, Any],
) -> dict[str, Any]:
    global _probe_request_count
    with _lock:
        if MODE == "probe_only":
            request_index = _probe_request_count
            reset_fingerprint = None
            cell, _ = authorize_probe_request(
                obs,
                release=RELEASE,
                runtime=RUNTIME,
                expected_request_index=request_index,
                observation_hashes_by_level=_probe_observation_hashes,
            )
        else:
            cell_id = obs.get("registered_cell_id")
            if not isinstance(cell_id, str):
                raise TypeError("V3-B005 behavior request requires registered_cell_id")
            request_index = _behavior_request_counts.get(cell_id, 0)
            assert BEHAVIORAL_RELEASE_GATE_SHA256 is not None
            cell, _, reset_fingerprint = authorize_behavior_request(
                obs,
                release=RELEASE,
                runtime=RUNTIME,
                behavioral_release_gate_sha256=BEHAVIORAL_RELEASE_GATE_SHA256,
                expected_request_index=request_index,
            )
        token = _request_seed.set(cell.seed)
        try:
            output = _official_infer(self, obs)
        finally:
            _request_seed.reset(token)
        if MODE == "probe_only":
            level, condition = PROBE_SEQUENCE[request_index]
            _probe_request_count += 1
        else:
            _behavior_request_counts[cell.cell_id] = request_index + 1
    metadata = server_response_metadata(
        mode=MODE,
        cell=cell,
        release=RELEASE,
        runtime=RUNTIME,
        request_index=request_index,
        behavioral_release_gate_sha256=BEHAVIORAL_RELEASE_GATE_SHA256,
        reset_fingerprint_sha256=reset_fingerprint,
    )
    if MODE == "probe_only":
        metadata.update(
            probe_request_index=request_index,
            probe_level_index=level,
            probe_condition=condition,
        )
    return {**output, **metadata}


server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer_v3b005


if __name__ == "__main__":
    server.main()
