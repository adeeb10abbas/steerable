#!/usr/bin/env python3
"""Pure runtime, probe, and behavioral-release checks for Nano V3-B005."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np

from experiments.v3.cosmos_droid.contract import (
    ContractError as PhaseARuntimeError,
    verify_runtime_identity as verify_phase_a_runtime_identity,
)
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    ACTION_SPACE,
    AMENDMENT_ID,
    CHECKPOINT_REVISION,
    COSMOS_REPOSITORY_COMMIT,
    EMPTY_SHA256,
    EXPECTED_SHA256,
    FIXED_OBSERVATION_SCHEMA,
    LEVELS,
    MODEL_ID,
    MODEL_REPOSITORY,
    PROBE_LEVELS,
    PROMPTS,
    RELEASE_GATE_SCHEMA,
    ROBOLAB_REPOSITORY_COMMIT,
    RUNTIME_SCHEMA,
    STUDY_ID,
    AuthorizedCell,
    ReleaseBundle,
    RuntimeContractError,
    canonical_json_bytes,
    compute_contract_sha256,
    load_json,
    sha256_bytes,
    sha256_file,
    validate_reset_attestation,
    verify_runtime_identity,
)


PINNED_CHECKPOINT_PATH = Path(
    "/data/users/ali/vla_wam/checkpoints/cosmos3_nano_policy_droid"
)
PINNED_SERVER_PORT = 18011
PROBE_SEQUENCE = tuple(
    (level, condition)
    for level in PROBE_LEVELS
    for condition in ("left", "left_exact_repeat", "right")
)
OBSERVATION_KEYS = (
    "observation/image",
    "observation/joint_position",
    "observation/gripper_position",
)


def _fail(message: str) -> None:
    raise RuntimeContractError(message)


def validate_pinned_server_cli(argv: list[str]) -> dict[str, Any]:
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
        raise RuntimeContractError("Nano V3-B005 server CLI differs from the contract") from exc
    observed = vars(args)
    observed["checkpoint_path"] = str(args.checkpoint_path.resolve())
    expected = {
        "checkpoint_path": str(PINNED_CHECKPOINT_PATH),
        "hf_revision": CHECKPOINT_REVISION,
        "host": "0.0.0.0",
        "port": PINNED_SERVER_PORT,
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
        _fail(f"Nano V3-B005 server CLI mismatch for {', '.join(changed)}")
    return observed


def bind_live_stack_runtime(
    *,
    study_root: Path,
    release: ReleaseBundle,
    base_runtime_manifest: Path,
) -> dict[str, Any]:
    """Map a verified Phase-A Nano identity into a fresh V3-B005 identity."""

    try:
        base = verify_phase_a_runtime_identity(
            Path(study_root).resolve(),
            MODEL_ID,
            Path(base_runtime_manifest).resolve(),
        )
    except PhaseARuntimeError as exc:
        raise RuntimeContractError(f"Phase-A Nano runtime identity is invalid: {exc}") from exc
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
    for key, expected in expected_base.items():
        if base.get(key) != expected:
            _fail(f"Phase-A Nano runtime cannot be mapped: mismatch for {key}")
    payload: dict[str, Any] = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "checkpoint_identifier": MODEL_REPOSITORY,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "checkpoint_hash_gate_passed": True,
        "external_repository_commit": COSMOS_REPOSITORY_COMMIT,
        "external_repository_diff_hash": EMPTY_SHA256,
        "simulator_repository_commit": ROBOLAB_REPOSITORY_COMMIT,
        "simulator_repository_diff_hash": EMPTY_SHA256,
        "environment_lock_sha256": base["environment_lock_hash"],
        "prospective_artifact_sha256": release.hashes,
        "v3b005_contract_sha256": compute_contract_sha256(study_root),
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static",
        "fixed_observation_gate_levels": list(PROBE_LEVELS),
        "phase_a_runtime_identity_sha256": base["runtime_identity_sha256"],
        "phase_a_runtime_manifest_sha256": sha256_file(base_runtime_manifest),
        "phase_a_adapter_contract_hash": base["adapter_contract_hash"],
        "phase_a_repository_pins": base["repository_pins"],
        "simulator_version": base["simulator_version"],
        "renderer_backend": base["renderer_backend"],
    }
    payload["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def verify_live_runtime_identity(
    runtime_manifest: Path,
    *,
    study_root: Path,
    release: ReleaseBundle,
) -> dict[str, Any]:
    return verify_runtime_identity(
        runtime_manifest,
        study_root=study_root,
        release=release,
    )


def _array_fingerprint(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": sha256_bytes(contiguous.tobytes()),
    }


def observation_component_hashes(request: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in OBSERVATION_KEYS if key not in request]
    if missing:
        _fail(f"Nano V3-B005 request lacks observations: {', '.join(missing)}")
    return {key: _array_fingerprint(request[key]) for key in OBSERVATION_KEYS}


def authorize_probe_request(
    request: Mapping[str, Any],
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    expected_request_index: int,
    observation_hashes_by_level: MutableMapping[int, dict[str, Any]],
) -> tuple[AuthorizedCell, dict[str, Any]]:
    """Authorize exactly one item of the nine-request, probe-only sequence."""

    if expected_request_index not in range(len(PROBE_SEQUENCE)):
        _fail("probe-only server accepts exactly nine requests")
    level, condition = PROBE_SEQUENCE[expected_request_index]
    relation = "left" if condition.startswith("left") else "right"
    cell = release.cell(f"v3b005:nano:seed9500:level{level}:{relation}")
    expected = {
        "v3b005_server_mode": "probe_only",
        "amendment_id": AMENDMENT_ID,
        "probe_request_index": expected_request_index,
        "probe_level_index": level,
        "probe_condition": condition,
        "registered_cell_id": cell.cell_id,
        "sampling_seed": 9500,
        "prompt": PROMPTS[relation],
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    for key, wanted in expected.items():
        if request.get(key) != wanted:
            _fail(f"Nano V3-B005 probe request mismatch for {key}")
    observed_hashes = observation_component_hashes(request)
    declared_hashes = request.get("observation_hashes")
    if declared_hashes != observed_hashes:
        _fail("Nano V3-B005 probe observation hashes do not match request bytes")
    previous = observation_hashes_by_level.setdefault(level, observed_hashes)
    if previous != observed_hashes:
        _fail(f"Nano V3-B005 level {level} probe observations are not byte-identical")
    return cell, observed_hashes


def validate_fixed_observation_report(
    report_path: Path,
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    report = load_json(report_path, "V3-B005 fixed-observation report")
    if (
        report.get("schema_version") != FIXED_OBSERVATION_SCHEMA
        or report.get("study_id") != STUDY_ID
        or report.get("amendment_id") != AMENDMENT_ID
        or report.get("model_id") != MODEL_ID
        or report.get("status") != "passed"
        or report.get("release_gate_passed") is not True
        or report.get("probe_only") is not True
        or report.get("behavioral_episode_count") != 0
        or report.get("model_request_count") != 9
        or report.get("probe_levels") != list(PROBE_LEVELS)
        or report.get("probe_sequence") != [list(item) for item in PROBE_SEQUENCE]
        or report.get("runtime_identity_sha256") != runtime["runtime_identity_sha256"]
        or report.get("prospective_artifact_sha256") != release.hashes
    ):
        _fail("V3-B005 fixed-observation report did not pass the exact nine-request gate")
    records = report.get("records")
    metrics = report.get("metrics")
    if not isinstance(records, list) or len(records) != 9 or not isinstance(metrics, Mapping):
        _fail("V3-B005 fixed-observation report is incomplete")
    for level in PROBE_LEVELS:
        rows = [row for row in records if row.get("level_index") == level]
        if (
            len(rows) != 3
            or [row.get("condition") for row in rows]
            != ["left", "left_exact_repeat", "right"]
            or len({json.dumps(row.get("observation_hashes"), sort_keys=True) for row in rows}) != 1
        ):
            _fail(f"V3-B005 level {level} did not retain one exact three-request probe")
        level_metrics = metrics.get(f"level{level}")
        if (
            not isinstance(level_metrics, Mapping)
            or level_metrics.get("left_exact_repeat_action_array_equal") is not True
            or level_metrics.get("left_exact_repeat_future_array_equal") is not True
            or float(level_metrics.get("left_right_action_rms", 0.0)) <= 0.0
            or float(level_metrics.get("left_right_future_pixel_mae", 0.0)) <= 0.0
        ):
            _fail(f"V3-B005 level {level} repeat/sensitivity gate failed")
    return report


def verify_behavioral_release_gate(
    gate_path: Path,
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    gate_path = Path(gate_path).resolve()
    gate = load_json(gate_path, "V3-B005 behavioral release gate")
    expected = {
        "schema_version": RELEASE_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "prospective_artifact_sha256": release.hashes,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "physical_gate_passed": True,
        "fixed_observation_release_passed": True,
        "model_request_count_before_release": 9,
        "behavioral_episode_count_before_release": 0,
        "behavioral_release": True,
        "authorized_behavioral_cell_count": 210,
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            _fail(f"V3-B005 behavioral release gate mismatch for {key}")
    fixed_record = gate.get("fixed_observation_report")
    if not isinstance(fixed_record, Mapping):
        _fail("V3-B005 release gate lacks fixed-observation report binding")
    fixed_path = Path(str(fixed_record.get("path", ""))).resolve()
    if (
        not fixed_path.is_file()
        or fixed_record.get("sha256") != sha256_file(fixed_path)
        or fixed_record.get("bytes") != fixed_path.stat().st_size
    ):
        _fail("V3-B005 fixed-observation report binding changed")
    validate_fixed_observation_report(fixed_path, release=release, runtime=runtime)
    return gate


def authorize_behavior_request(
    request: Mapping[str, Any],
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    behavioral_release_gate_sha256: str,
    expected_request_index: int,
) -> tuple[AuthorizedCell, dict[str, Any] | None, str | None]:
    """Authorize a static-prompt behavioral request after the independent gate."""

    cell_id = request.get("registered_cell_id")
    if not isinstance(cell_id, str):
        _fail("Nano V3-B005 behavior request lacks registered_cell_id")
    cell = release.cell(cell_id)
    expected = {
        "v3b005_server_mode": "behavior_only",
        "amendment_id": AMENDMENT_ID,
        "sampling_seed": cell.seed,
        "prompt": cell.row["prompt"],
        "request_index": expected_request_index,
        "action_step_start": expected_request_index * ACTION_CHUNK_STEPS,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "behavioral_release_gate_sha256": behavioral_release_gate_sha256,
    }
    for key, wanted in expected.items():
        if request.get(key) != wanted:
            _fail(f"Nano V3-B005 behavior request mismatch for {key}")
    if expected_request_index * ACTION_CHUNK_STEPS >= ACTION_CAP:
        _fail("Nano V3-B005 behavior request begins at or beyond the action cap")
    reset_value = request.get("reset_attestation_path")
    if not isinstance(reset_value, str) or not reset_value:
        _fail("Nano V3-B005 behavior request lacks reset_attestation_path")
    reset, fingerprint = validate_reset_attestation(
        Path(reset_value).resolve(),
        cell=cell,
        release=release,
        runtime=runtime,
        behavioral_release_gate_sha256=behavioral_release_gate_sha256,
    )
    if request.get("reset_fingerprint_sha256") != fingerprint:
        _fail("Nano V3-B005 reset fingerprint does not match its attestation")
    return cell, reset, fingerprint


def server_response_metadata(
    *,
    mode: str,
    cell: AuthorizedCell,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    request_index: int,
    behavioral_release_gate_sha256: str | None = None,
    reset_fingerprint_sha256: str | None = None,
) -> dict[str, Any]:
    result = {
        "v3b005_nano_live_stack": "lateral_dose_response_v1",
        "v3b005_server_mode": mode,
        "amendment_id": AMENDMENT_ID,
        "registered_cell_id": cell.cell_id,
        "sampling_seed": cell.seed,
        "request_index": request_index,
        "release_fingerprint_sha256": release.release_fingerprint(cell),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
    }
    if behavioral_release_gate_sha256 is not None:
        result["behavioral_release_gate_sha256"] = behavioral_release_gate_sha256
    if reset_fingerprint_sha256 is not None:
        result["reset_fingerprint_sha256"] = reset_fingerprint_sha256
    return result
