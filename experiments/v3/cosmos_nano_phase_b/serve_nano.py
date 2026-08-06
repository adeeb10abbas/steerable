"""Policy-server overlay for only the released V3-B001 Nano cells.

Required paths and the externally pinned release digest are supplied through
environment variables so the official Cosmos CLI remains unchanged.  Import
or startup fails before checkpoint loading when any live contract differs.
"""

from __future__ import annotations

import os
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from experiments.v3.cosmos_nano_phase_b.live_support import (
    PINNED_CHECKPOINT_PATH,
    authorize_server_request,
    server_response_metadata,
    verify_live_runtime_identity,
    validate_pinned_server_cli,
)
from experiments.v3.cosmos_nano_phase_b.runtime_adapter import load_release_bundle


def _required_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required before starting the V3-B001 Nano server")
    return Path(value).resolve()


STUDY_ROOT = _required_path("VLA_WAM_STUDY_ROOT")
RELEASE_MANIFEST = _required_path("VLA_WAM_V3B_RELEASE_MANIFEST")
RUNTIME_MANIFEST = _required_path("VLA_WAM_V3B_RUNTIME_MANIFEST")
RELEASE_MANIFEST_SHA256 = os.environ.get("VLA_WAM_V3B_RELEASE_MANIFEST_SHA256", "")

RELEASE = load_release_bundle(
    RELEASE_MANIFEST,
    expected_manifest_sha256=RELEASE_MANIFEST_SHA256,
)
RUNTIME = verify_live_runtime_identity(
    RUNTIME_MANIFEST,
    study_root=STUDY_ROOT,
    release=RELEASE,
)
SERVER_CLI = validate_pinned_server_cli(sys.argv[1:])
if not PINNED_CHECKPOINT_PATH.is_dir():
    raise RuntimeError(f"pinned Nano checkpoint directory is missing: {PINNED_CHECKPOINT_PATH}")

# These imports preserve the exact official Nano server implementation and the
# already-frozen no-guardrail compatibility patch.  This file changes only the
# prospective seed/request authorization envelope.
from cosmos_framework.scripts import action_policy_server_robolab as server  # noqa: E402
from experiments.cosmos import serve_robolab_without_guardrails  # noqa: E402,F401


_request_seed: ContextVar[int | None] = ContextVar("v3b001_request_seed", default=None)
_official_next_seed = server.RobolabPolicyService._next_seed
_official_infer = server.RobolabPolicyService.infer
_request_counts: dict[str, int] = {}
_request_lock = threading.Lock()


def _next_seed(self: server.RobolabPolicyService) -> int:
    seed = _request_seed.get()
    return int(seed) if seed is not None else int(_official_next_seed(self))


def _infer_released_cell(
    self: server.RobolabPolicyService,
    obs: dict[str, Any],
) -> dict[str, Any]:
    cell_id = obs.get("registered_cell_id")
    if not isinstance(cell_id, str):
        raise TypeError("V3-B001 Nano request requires registered_cell_id")
    # One Nano worker owns this queue.  Serializing the verification+inference
    # block also makes request indices deterministic if multiple simulator
    # clients accidentally connect concurrently.
    with _request_lock:
        request_index = _request_counts.get(cell_id, 0)
        cell, _, reset_fingerprint = authorize_server_request(
            obs,
            study_root=STUDY_ROOT,
            release=RELEASE,
            runtime=RUNTIME,
            expected_request_index=request_index,
        )
        token = _request_seed.set(cell.seed)
        try:
            output = _official_infer(self, obs)
        finally:
            _request_seed.reset(token)
        _request_counts[cell_id] = request_index + 1
    return {
        **output,
        **server_response_metadata(
            cell=cell,
            release=RELEASE,
            runtime=RUNTIME,
            reset_fingerprint_sha256=reset_fingerprint,
            request_index=request_index,
        ),
    }


server.RobolabPolicyService._next_seed = _next_seed
server.RobolabPolicyService.infer = _infer_released_cell


if __name__ == "__main__":
    server.main()
