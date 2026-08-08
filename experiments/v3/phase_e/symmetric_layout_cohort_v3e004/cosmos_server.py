"""Serialized E004 authorization overlay for the official Cosmos servers."""

from __future__ import annotations

from contextvars import ContextVar
import os
from pathlib import Path
import sys
import threading
from typing import Any

from .cosmos_runtime import (
    MODEL_SPECS,
    CosmosRuntimeError,
    add_response_envelope,
    ensure_exact_server_cli,
    load_registration_bundle,
    load_runtime_identity,
    validate_request_envelope,
)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise CosmosRuntimeError(f"{name} is required before E004 server startup")
    return value


def run_server(model_id: str) -> None:
    """Run one model-specific process; this function launches no simulator."""

    if model_id not in MODEL_SPECS:
        raise CosmosRuntimeError(f"unsupported E004 Cosmos model: {model_id}")
    spec = MODEL_SPECS[model_id]
    if _required("VLA_WAM_V3E004_MODEL_ENVIRONMENT") != spec["environment_id"]:
        raise CosmosRuntimeError(
            "E004 Cosmos server is not in its registered model-specific environment"
        )
    study_root = Path(_required("VLA_WAM_STUDY_ROOT")).resolve()
    registration_commit = _required("VLA_WAM_V3E004_REGISTRATION_COMMIT")
    runtime_path = Path(_required("VLA_WAM_V3E004_RUNTIME_MANIFEST")).resolve()
    session_root = Path(_required("VLA_WAM_V3E004_SESSION_ROOT")).resolve()
    bundle = load_registration_bundle(
        study_root, registration_commit=registration_commit
    )
    runtime = load_runtime_identity(runtime_path, model_id=model_id)
    ensure_exact_server_cli(model_id, sys.argv[1:])
    if not Path(spec["checkpoint_path"]).is_dir():
        raise CosmosRuntimeError(
            f"registered {model_id} checkpoint is missing: {spec['checkpoint_path']}"
        )
    if not session_root.is_dir():
        raise CosmosRuntimeError("E004 session root must exist before server startup")

    # Import only after every local identity gate passes.  Nano and Edge use
    # distinct entry points, external checkouts, and environments; one process
    # therefore never imports both incompatible model stacks.
    from cosmos_framework.scripts import action_policy_server_robolab as official
    from experiments.cosmos import serve_robolab_without_guardrails  # noqa: F401

    request_seed: ContextVar[int | None] = ContextVar(
        f"v3e004_{model_id}_request_seed", default=None
    )
    original_next_seed = official.RobolabPolicyService._next_seed
    original_infer = official.RobolabPolicyService.infer
    lock = threading.Lock()
    request_counts: dict[tuple[str, str], int] = {}
    session_cells: dict[str, str] = {}

    def next_seed(self: Any) -> int:
        seed = request_seed.get()
        return int(seed) if seed is not None else int(original_next_seed(self))

    def infer(self: Any, obs: dict[str, Any]) -> dict[str, Any]:
        # The complete authorization + inference + output-hashing critical
        # section is serialized because the official sampler stores seed state
        # in process scope.
        with lock:
            session_id = obs.get("session_id")
            cell_id = obs.get("registered_cell_id")
            if not isinstance(session_id, str) or not isinstance(cell_id, str):
                raise CosmosRuntimeError("E004 Cosmos request lacks cell/session identity")
            key = (session_id, cell_id)
            request_index = request_counts.get(key, 0)
            cell, session, _, _ = validate_request_envelope(
                obs,
                bundle=bundle,
                runtime=runtime,
                model_id=model_id,
                expected_request_index=request_index,
                session_root=session_root,
            )
            owner = session_cells.setdefault(session_id, cell_id)
            if owner != cell_id:
                raise CosmosRuntimeError("one E004 session cannot own multiple cells")
            token = request_seed.set(cell.seed)
            try:
                native_output = original_infer(self, obs)
            finally:
                request_seed.reset(token)
            response = add_response_envelope(
                native_output,
                cell=cell,
                runtime=runtime,
                session=session,
                request=obs,
            )
            request_counts[key] = request_index + 1
            return response

    official.RobolabPolicyService._next_seed = next_seed
    official.RobolabPolicyService.infer = infer
    official.main()
