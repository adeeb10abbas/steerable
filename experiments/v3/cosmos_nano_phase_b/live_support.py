#!/usr/bin/env python3
"""Pure live-stack checks for the V3-B001 Nano behavioral queue.

Nothing in this module imports Isaac, RoboLab, or Cosmos.  Both the simulator
bridge and policy-server overlay use these checks so a request can cross the
process boundary only after the same released cell, runtime, and reset
fingerprints have been verified on both sides.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from experiments.v3.cosmos_droid.contract import (
    ContractError as PhaseARuntimeError,
    verify_runtime_identity as verify_phase_a_runtime_identity,
)
from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    ACTION_SPACE,
    AMENDMENT_ID,
    CHECKPOINT_REVISION,
    COSMOS_REPOSITORY_COMMIT,
    EMPTY_SHA256,
    MODEL_ID,
    MODEL_REPOSITORY,
    ROBOLAB_REPOSITORY_COMMIT,
    RUNTIME_SCHEMA,
    STUDY_ID,
    AuthorizedCell,
    ReleaseBundle,
    RuntimeContractError,
    canonical_json_bytes,
    compute_adapter_contract_sha256,
    sha256_bytes,
    sha256_file,
    validate_reset_attestation,
    verify_runtime_identity,
)


LIVE_STACK_FILES = (
    "experiments/v3/cosmos_nano_phase_b/live_support.py",
    "experiments/v3/cosmos_nano_phase_b/live_client.py",
    "experiments/v3/cosmos_nano_phase_b/serve_nano.py",
    "experiments/v3/cosmos_nano_phase_b/robolab_bridge.py",
    "experiments/v3/cosmos_nano_phase_b/queue_launcher.py",
)
PINNED_CHECKPOINT_PATH = Path(
    "/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid"
)


def _fail(message: str) -> None:
    raise RuntimeContractError(message)


def compute_live_stack_sha256(study_root: Path) -> str:
    root = Path(study_root).resolve()
    inventory: list[dict[str, Any]] = []
    for relative in LIVE_STACK_FILES:
        path = root / relative
        if not path.is_file():
            _fail(f"missing Phase-B live-stack source: {relative}")
        inventory.append(
            {"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size}
        )
    return sha256_bytes(canonical_json_bytes(inventory))


def validate_pinned_server_cli(argv: list[str]) -> dict[str, Any]:
    """Parse and reject any deviation from the released Nano serving CLI."""

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--hf-revision", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--domain-name", required=True)
    parser.add_argument("--decode-video", action="store_true")
    parser.add_argument("--action-chunk-size", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--action-space", required=True)
    parser.add_argument("--history-length", type=int, required=True)
    parser.add_argument("--use-state", action="store_true")
    parser.add_argument("--conditioning-fps", type=int, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--guidance", type=float, required=True)
    parser.add_argument("--num-steps", type=float, required=True)
    parser.add_argument("--shift", type=float, required=True)
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        raise RuntimeContractError("Nano server CLI differs from the released contract") from exc
    observed = vars(args)
    observed["checkpoint_path"] = str(args.checkpoint_path.resolve())
    expected = {
        "checkpoint_path": str(PINNED_CHECKPOINT_PATH),
        "hf_revision": CHECKPOINT_REVISION,
        "host": "0.0.0.0",
        "port": 18011,
        "domain_name": "droid_lerobot",
        "decode_video": True,
        "action_chunk_size": ACTION_CHUNK_STEPS,
        "action_dim": ACTION_DIM,
        "action_space": "joint_pos",
        "history_length": 1,
        "use_state": True,
        "conditioning_fps": 15,
        "resolution": 480,
        "guidance": 3.0,
        "num_steps": 4.0,
        "shift": 5.0,
    }
    if observed != expected:
        changed = [key for key, value in expected.items() if observed.get(key) != value]
        _fail(f"Nano server CLI mismatch for {', '.join(changed)}")
    return observed


def bind_live_stack_runtime(
    *,
    study_root: Path,
    release: ReleaseBundle,
    base_runtime_manifest: Path,
) -> dict[str, Any]:
    """Return a live-stack-bound runtime derived from verified Phase-A identity.

    The existing Nano runtime is a Phase-A manifest, not a Phase-B manifest.
    Verify it with its native contract first, then explicitly map the immutable
    model/repository/environment fields into the prospective Phase-B schema.
    The caller must write the returned object to a new path; Phase-A is never
    mutated or relabeled.
    """

    try:
        base = verify_phase_a_runtime_identity(
            Path(study_root).resolve(),
            MODEL_ID,
            Path(base_runtime_manifest).resolve(),
        )
    except PhaseARuntimeError as exc:
        raise RuntimeContractError(f"Phase-A runtime identity is invalid: {exc}") from exc
    expected_base = {
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
    }
    for key, value in expected_base.items():
        if base.get(key) != value:
            _fail(f"Phase-A runtime cannot be mapped: mismatch for {key}")
    environment_hash = base.get("environment_lock_hash")
    checkpoint_hash = base.get("checkpoint_sha256")
    if not isinstance(environment_hash, str) or len(environment_hash) != 64:
        _fail("Phase-A runtime lacks a valid environment_lock_hash")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        _fail("Phase-A runtime lacks a valid checkpoint_sha256")

    payload: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        # The name changes across schemas; the digest does not.
        "environment_lock_sha256": environment_hash,
        "phase_b_adapter_contract_sha256": compute_adapter_contract_sha256(study_root),
        "release_manifest_sha256": release.manifest_sha256,
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "phase_a_runtime_identity_sha256": base["runtime_identity_sha256"],
        "phase_a_runtime_manifest_sha256": sha256_file(base_runtime_manifest),
        "phase_a_adapter_contract_hash": base["adapter_contract_hash"],
        "phase_a_repository_pins": base["repository_pins"],
        "simulator_version": base["simulator_version"],
        "renderer_backend": base["renderer_backend"],
    }
    payload["phase_b_live_stack_sha256"] = compute_live_stack_sha256(study_root)
    payload["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def verify_live_runtime_identity(
    runtime_manifest: Path,
    *,
    study_root: Path,
    release: ReleaseBundle,
) -> dict[str, Any]:
    runtime = verify_runtime_identity(
        runtime_manifest,
        study_root=study_root,
        release=release,
    )
    if runtime.get("phase_b_live_stack_sha256") != compute_live_stack_sha256(study_root):
        _fail("runtime does not bind the checked-in Phase-B live queue stack")
    return runtime


def authorize_server_request(
    request: Mapping[str, Any],
    *,
    study_root: Path,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    expected_request_index: int,
) -> tuple[AuthorizedCell, dict[str, Any], str]:
    """Validate one request without invoking the policy implementation."""

    if not isinstance(request, Mapping):
        _fail("Nano server request must be an object")
    cell_id = request.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("Nano server request lacks registered_cell_id")
    cell = release.cell(cell_id)
    release_fingerprint = release.release_fingerprint(cell)
    expected = {
        "sampling_seed": cell.seed,
        "prompt": cell.row["prompt"],
        "request_index": expected_request_index,
        "action_step_start": expected_request_index * ACTION_CHUNK_STEPS,
        "release_fingerprint_sha256": release_fingerprint,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "amendment_id": AMENDMENT_ID,
    }
    for key, wanted in expected.items():
        if request.get(key) != wanted:
            _fail(f"Nano server request mismatch for {key}")
    if expected_request_index * ACTION_CHUNK_STEPS >= ACTION_CAP:
        _fail("Nano server request begins at or beyond the 450-action cap")
    reset_path_value = request.get("reset_attestation_path")
    if not isinstance(reset_path_value, str) or not reset_path_value:
        _fail("Nano server request lacks reset_attestation_path")
    reset_path = Path(reset_path_value).resolve()
    reset, reset_fingerprint = validate_reset_attestation(
        reset_path,
        cell=cell,
        release=release,
        runtime=runtime,
    )
    if request.get("reset_fingerprint_sha256") != reset_fingerprint:
        _fail("Nano server request reset fingerprint does not match its attestation")
    if reset.get("model_request_count_before_attestation") != 0:
        _fail("live reset attestation was not completed before the first model request")
    return cell, reset, reset_fingerprint


def server_response_metadata(
    *,
    cell: AuthorizedCell,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    reset_fingerprint_sha256: str,
    request_index: int,
) -> dict[str, Any]:
    return {
        "v3b001_nano_live_stack": "position_mirror_v1",
        "amendment_id": AMENDMENT_ID,
        "registered_cell_id": cell.cell_id,
        "sampling_seed": cell.seed,
        "request_index": request_index,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "reset_fingerprint_sha256": reset_fingerprint_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
