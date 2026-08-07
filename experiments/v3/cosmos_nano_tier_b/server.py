#!/usr/bin/env python3
"""Fresh, isolated Nano server envelope for V3-B008 or V3-B009."""

from __future__ import annotations

import hashlib
import os
import sys
import threading
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np

from experiments.v3.cosmos_nano_tier_b.runtime_contract import (
    CONFIG,
    PINNED_CHECKPOINT_PATH,
    ContractError,
    ReleaseBundle,
    load_release,
    load_runtime,
    sha256_file,
    validate_behavioral_release_gate,
    validate_server_cli,
)


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ContractError(f"{name} is required before starting an isolated Nano Tier-B server")
    return value


def _observation_hashes(obs: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("observation/image", "observation/joint_position", "observation/gripper_position"):
        if key not in obs:
            raise ContractError(f"fixed-observation probe lacks {key}")
        array = np.ascontiguousarray(np.asarray(obs[key]))
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
        result[key] = digest.hexdigest()
    return result


def _expect(request: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    for key, wanted in expected.items():
        if request.get(key) != wanted:
            raise ContractError(f"{label} mismatch for {key}")


def _probe_sequence(release: ReleaseBundle) -> tuple[tuple[str, str], ...]:
    return tuple(
        (arm, condition)
        for arm in release.config["arms"]
        for condition in ("left", "left_exact_repeat", "right")
    )


def run_server(amendment_id: str) -> None:
    if amendment_id not in CONFIG:
        raise ContractError(f"unsupported Nano Tier-B server amendment {amendment_id}")
    study_root = Path(_required("VLA_WAM_STUDY_ROOT")).resolve()
    manifest_path = Path(_required("VLA_WAM_NANO_TIERB_MANIFEST")).resolve()
    runtime_path = Path(_required("VLA_WAM_NANO_TIERB_RUNTIME_MANIFEST")).resolve()
    mode = _required("VLA_WAM_NANO_TIERB_MODE")
    if mode not in {"probe_only", "behavior_only"}:
        raise ContractError("VLA_WAM_NANO_TIERB_MODE must be probe_only or behavior_only")
    release = load_release(study_root, amendment_id, manifest_path)
    runtime = load_runtime(runtime_path, study_root=study_root, release=release)
    if not PINNED_CHECKPOINT_PATH.is_dir():
        raise ContractError(f"pinned Nano checkpoint is missing: {PINNED_CHECKPOINT_PATH}")
    validate_server_cli(amendment_id, sys.argv[1:])

    behavior_gate_sha256: str | None = None
    if mode == "behavior_only":
        gate_path = Path(_required("VLA_WAM_NANO_TIERB_BEHAVIORAL_RELEASE_GATE")).resolve()
        validate_behavioral_release_gate(gate_path, release=release, runtime=runtime)
        behavior_gate_sha256 = sha256_file(gate_path)
    elif os.environ.get("VLA_WAM_NANO_TIERB_BEHAVIORAL_RELEASE_GATE"):
        raise ContractError("probe-only server must not receive a behavioral release gate")

    # Import the official server only after all startup bindings pass.
    from cosmos_framework.scripts import action_policy_server_robolab as official
    from experiments.cosmos import serve_robolab_without_guardrails  # noqa: F401

    request_seed: ContextVar[int | None] = ContextVar("nano_tier_b_request_seed", default=None)
    original_next_seed = official.RobolabPolicyService._next_seed
    original_infer = official.RobolabPolicyService.infer
    lock = threading.Lock()
    probe_count = 0
    probe_hashes: dict[str, dict[str, str]] = {}
    behavior_counts: dict[str, int] = {}
    sequence = _probe_sequence(release)

    def next_seed(self: Any) -> int:
        value = request_seed.get()
        return int(value) if value is not None else int(original_next_seed(self))

    def infer(self: Any, obs: dict[str, Any]) -> dict[str, Any]:
        nonlocal probe_count
        with lock:
            if mode == "probe_only":
                if probe_count >= len(sequence):
                    raise ContractError("fixed-observation probe exceeded its exact request count")
                arm, condition = sequence[probe_count]
                relation = "right" if condition == "right" else "left"
                cell = release.probe_cell(arm, relation)
                _expect(obs, {
                    "nano_tier_b_server_mode": "probe_only",
                    "amendment_id": amendment_id,
                    "probe_request_index": probe_count,
                    "probe_arm": arm,
                    "probe_condition": condition,
                    "registered_cell_id": cell.cell_id,
                    "sampling_seed": cell.seed,
                    "prompt": cell.row["prompt"],
                    "release_fingerprint_sha256": release.release_fingerprint(cell),
                    "runtime_identity_sha256": runtime["runtime_identity_sha256"],
                }, "fixed-observation request")
                hashes = _observation_hashes(obs)
                if arm in probe_hashes and hashes != probe_hashes[arm]:
                    raise ContractError(f"{amendment_id} fixed observations differ within arm {arm}")
                probe_hashes.setdefault(arm, hashes)
                request_index = probe_count
                reset_fingerprint = None
            else:
                cell_id = obs.get("registered_cell_id")
                if not isinstance(cell_id, str):
                    raise ContractError("behavior request lacks registered_cell_id")
                cell = release.cell(cell_id)
                request_index = behavior_counts.get(cell_id, 0)
                reset_fingerprint = obs.get("reset_fingerprint_sha256")
                if not isinstance(reset_fingerprint, str) or len(reset_fingerprint) != 64:
                    raise ContractError("behavior request lacks a reset fingerprint")
                assert behavior_gate_sha256 is not None
                _expect(obs, {
                    "nano_tier_b_server_mode": "behavior_only",
                    "amendment_id": amendment_id,
                    "registered_cell_id": cell.cell_id,
                    "sampling_seed": cell.seed,
                    "prompt": cell.row["prompt"],
                    "request_index": request_index,
                    "action_step_start": request_index * 32,
                    "release_fingerprint_sha256": release.release_fingerprint(cell),
                    "runtime_identity_sha256": runtime["runtime_identity_sha256"],
                    "behavioral_release_gate_sha256": behavior_gate_sha256,
                }, "behavior request")
                if request_index * 32 >= 450:
                    raise ContractError("behavior request begins at or beyond the action cap")
            token = request_seed.set(cell.seed)
            try:
                output = original_infer(self, obs)
            finally:
                request_seed.reset(token)
            if mode == "probe_only":
                probe_count += 1
            else:
                behavior_counts[cell.cell_id] = request_index + 1
        metadata = {
            "nano_tier_b_live_stack": "isolated_v3b008_v3b009_v1",
            "nano_tier_b_server_mode": mode,
            "amendment_id": amendment_id,
            "registered_cell_id": cell.cell_id,
            "sampling_seed": cell.seed,
            "request_index": request_index,
            "release_fingerprint_sha256": release.release_fingerprint(cell),
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            "behavioral_release_gate_sha256": behavior_gate_sha256,
            "reset_fingerprint_sha256": reset_fingerprint,
        }
        if mode == "probe_only":
            metadata.update(probe_request_index=request_index, probe_arm=arm, probe_condition=condition)
        return {**output, **metadata}

    official.RobolabPolicyService._next_seed = next_seed
    official.RobolabPolicyService.infer = infer
    official.main()

