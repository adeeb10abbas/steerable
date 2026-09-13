#!/usr/bin/env python3
"""Run one frozen D01--D04 DreamZero D1 development block as paired jobs.

The official conditional DreamZero service is global-state and requires two
GPUs, while RoboLab must run in a distinct one-GPU worker.  This module binds
the proven P00 paired launcher to one immutable development layout and makes
both queue jobs re-enter this module.  It requires the passed D1 generation
qualification, the complete paired P00 pilot, the recorder qualification, and
the selected layout's hash-bound gate, pose, and fixed-observation capture.

Every cell receives 451 original observations, executes 450 actions, and makes
57 official joint action/future requests (24 returned actions, eight eligible
actions per request, and two actions from the final request).  Stateful reset
or inference requests are never replayed after an ambiguous transport failure.
Only a deeply validated contiguous prefix may be reused across attempts.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

import d1_behavioral_pilot_jobs as pilot


DEVELOPMENT_LAYOUT_IDS = ("D01", "D02", "D03", "D04")
DEVELOPMENT_SCHEDULE: dict[str, tuple[int, tuple[str, ...]]] = {
    "D01": (
        2026091101,
        ("original-left", "reflected-right", "original-right", "reflected-left"),
    ),
    "D02": (
        2026091102,
        ("reflected-right", "original-left", "reflected-left", "original-right"),
    ),
    "D03": (
        2026091103,
        ("reflected-right", "original-right", "original-left", "reflected-left"),
    ),
    "D04": (
        2026091104,
        ("original-left", "reflected-left", "reflected-right", "original-right"),
    ),
}
ALLOWED_SIMULATOR_ROLES = (
    "wmf-forecast-0912-worker-00",
    "wmf-forecast-0912-worker-05",
    "wmf-forecast-0912-worker-06",
    "wmf-forecast-0912-worker-09",
)
P00_BLOCK_ID = "wmf_ablation_001_20260912__pilot__P00__D1"
P00_CONDITION_ORDER = (
    "reflected-right",
    "reflected-left",
    "original-left",
    "original-right",
)
P00_CELL_IDS = tuple(
    f"wmf1__pilot__P00__D1__{label.replace('-', '__')}"
    for label in P00_CONDITION_ORDER
)
P00_SIMULATOR_RECEIPT_SCHEMA = "wmf-d1-behavioral-pilot-simulator-job-v1"
P00_SERVER_RECEIPT_SCHEMA = "wmf-d1-behavioral-pilot-server-job-v1"

RUNNER_FILENAME = "d1_development_block_jobs.py"
RAW_PARENT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/development/D1"
)
GLOBAL_D1_SERVER_LOCK = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/.locks/"
    "d1-global-server.lock"
)
SIMULATOR_RECEIPT_FILENAME = "d1_behavioral_development_receipt.json"
SERVER_RECEIPT_SCHEMA = "wmf-d1-behavioral-development-server-job-v1"
SIMULATOR_RECEIPT_SCHEMA = "wmf-d1-behavioral-development-simulator-job-v1"
CELL_RECEIPT_SCHEMA = "wmf-d1-behavioral-development-cell-v1"
RESUME_SCHEMA = "wmf-d1-behavioral-development-resume-v1"
DEVELOPMENT_CONTRACT_SCHEMA = "wmf-d1-behavioral-development-contract-v1"
EXECUTION_PREREQUISITES_SCHEMA = "wmf-d1-development-execution-prerequisites-v1"
PREREQUISITE_PREFLIGHT_SCHEMA = "wmf-d1-development-prerequisite-preflight-v1"
FIXED_CAPTURE_QUEUE_SCHEMA = "wmf-forecast-layout-fixed-observation-job-v1"
FIXED_CAPTURE_SCHEMA = "wmf-forecast-layout-fixed-observation-capture-v1"
CONTROL_ROOT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/control"
)
DEVELOPMENT_CAPTURE_WRAPPER_RELEASES: dict[str, dict[str, str]] = {
    "D01": {
        "job_id": "fixed-observation-d01-001",
        "sha256": "8531d11584a36f2326074b287e42394a1f9cf3b04fd1016855f472152b547077",
    },
    "D02": {
        "job_id": "fixed-observation-d02-001",
        "sha256": "a700280562d3640349fe8ec5a2594decf4a7b2b9f30450c9e9af55420bb907ac",
    },
    "D03": {
        "job_id": "fixed-observation-d03-001",
        "sha256": "728416e0c31bfacb5cb36dc46c82370d2e157e508c4c6349ea934ac4bee99a29",
    },
    "D04": {
        "job_id": "fixed-observation-d04-001",
        "sha256": "f943e9fa2a968bc34fea9ed7cc0cbc4508f86f1a28071a82972863dc602807bc",
    },
}
PREFLIGHT_QUEUE_ROLES = ("any", pilot.SERVER_QUEUE_ROLE, *ALLOWED_SIMULATOR_ROLES)
NO_REPLAY_TRANSPORT_CONTRACT = {
    "schema_version": "wmf-d1-no-replay-websocket-v1",
    "compression": None,
    "max_size": None,
    "ping_interval": None,
    "ping_timeout": None,
    "connection_establishment_retries_only": pilot.D1_CONNECT_RETRIES,
    "stateful_request_replay": False,
    "lost_response_policy": "fail_cell_without_replaying_reset_or_inference",
    "outer_bound": "paired_queue_cell_timeout_and_leases",
}

SAFE_LAYOUT_RE = re.compile(r"D0[1-4]\Z")
_CONFIGURATION_ACTIVE = False


_PILOT_DEFAULTS = {
    name: getattr(pilot, name)
    for name in (
        "PHASE",
        "LAYOUT_PAIR_ID",
        "BLOCK_ID",
        "CONDITIONS",
        "CELL_IDS",
        "ENVIRONMENT_SEED",
        "RAW_ROOT",
        "PILOT_CONTRACT",
        "PILOT_CONTRACT_SHA256",
        "RUNNER_FILENAME",
        "BEHAVIORAL_PURPOSE",
        "BEHAVIORAL_FINALIZE_PURPOSE",
        "EPISODE_ID_PREFIX",
        "POLICY_LABEL",
        "SIMULATOR_RECEIPT_FILENAME",
        "GLOBAL_SERVER_LOCK_PATH",
        "SERVER_RECEIPT_SCHEMA",
        "SIMULATOR_RECEIPT_SCHEMA",
        "CELL_RECEIPT_SCHEMA",
        "SIMULATOR_QUEUE_ROLE",
    )
}
_PILOT_VALIDATE_QUEUE_INVOCATION = pilot.validate_queue_invocation
_PILOT_VALIDATE_SERVER_READY = pilot.validate_server_ready
_PILOT_VALIDATE_PASSED_CELL = pilot.validate_passed_cell_receipt
_PILOT_DISCOVER_COMPLETED_PREFIX = pilot.discover_completed_prefix
_PILOT_IMMUTABLE_JSON = pilot.immutable_json


@dataclass(frozen=True)
class DevelopmentBlock:
    """Exact immutable identities for one D1 development layout block."""

    layout_pair_id: str
    environment_seed: int
    condition_order: tuple[str, ...]
    conditions: tuple[tuple[str, str, str], ...]
    cell_ids: tuple[str, ...]
    block_id: str
    raw_root: Path
    schedule_path: Path
    schedule_sha256: str
    schedule_row: Mapping[str, Any]
    contract: Mapping[str, Any]
    contract_sha256: str


def _condition_tuple(label: str) -> tuple[str, str, str]:
    pilot.require(label.count("-") == 1, "development_condition_label_invalid", label)
    arm, command = label.split("-", 1)
    pilot.require(arm in {"original", "reflected"}, "development_condition_arm_invalid")
    pilot.require(command in {"left", "right"}, "development_condition_command_invalid")
    return arm, command, f"WMFForecast{arm.title()}{command.title()}Task"


def _expected_cell_ids(layout_pair_id: str, order: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"wmf1__development__{layout_pair_id}__D1__{label.replace('-', '__')}"
        for label in order
    )


def _development_contract(
    *, layout_pair_id: str, environment_seed: int,
    condition_order: Sequence[str], cell_ids: Sequence[str], block_id: str,
) -> dict[str, Any]:
    contract = dict(_PILOT_DEFAULTS["PILOT_CONTRACT"])
    contract.update(
        {
            "schema_version": DEVELOPMENT_CONTRACT_SCHEMA,
            "phase": "development",
            "layout_pair_id": layout_pair_id,
            "block_id": block_id,
            "condition_order": list(condition_order),
            "cell_ids": list(cell_ids),
            "environment_seed": environment_seed,
            "effective_model_noise_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
            "noise_semantics": (
                "fixed at 1140; layouts, cells, and requests are not independent noise draws"
            ),
            "transport": {
                "compression": None,
                "max_size": None,
                "ping_interval": None,
                "ping_timeout": None,
                "stateful_request_replay": False,
            },
        }
    )
    return contract


def load_development_block(source_root: Path, layout_pair_id: str) -> DevelopmentBlock:
    pilot.require(
        isinstance(layout_pair_id, str)
        and SAFE_LAYOUT_RE.fullmatch(layout_pair_id) is not None
        and layout_pair_id in DEVELOPMENT_SCHEDULE,
        "development_layout_unsupported",
        str(layout_pair_id),
    )
    environment_seed, order = DEVELOPMENT_SCHEDULE[layout_pair_id]
    block_id = f"wmf_ablation_001_20260912__development__{layout_pair_id}__D1"
    cell_ids = _expected_cell_ids(layout_pair_id, order)
    schedule_path = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
    )
    schedule = pilot.load_json(schedule_path, "parallel_schedule_unreadable")
    rows = [row for row in schedule.get("jobs", []) if row.get("job_id") == block_id]
    pilot.require(len(rows) == 1, "development_schedule_row_missing_or_duplicate")
    row = rows[0]
    expected = {
        "phase": "development",
        "layout_pair_id": layout_pair_id,
        "model_config": "D1",
        "candidate_effective_policy_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
        "condition_order": list(order),
        "ordered_cell_ids": list(cell_ids),
        "indivisible": True,
        "execution_contract": {
            "full_model_and_simulator_reset_before_each_condition": True,
            "no_global_DreamZero_context_interleaving": True,
            "same_isolated_worker_type_within_job": True,
            "sequential_conditions": True,
        },
        "depends_on_jobs": [P00_BLOCK_ID],
        "required_gates": [
            "all_four_pilot_recordings_qualified_or_documented_exact_runtime_waiver",
            "qualified_development_fixtures",
            "bounded_resource_authorization",
        ],
        "released": False,
        "status": "NOT_RELEASED",
    }
    for key, wanted in expected.items():
        pilot.require(row.get(key) == wanted, "development_schedule_mismatch", key)
    contract = _development_contract(
        layout_pair_id=layout_pair_id,
        environment_seed=environment_seed,
        condition_order=order,
        cell_ids=cell_ids,
        block_id=block_id,
    )
    return DevelopmentBlock(
        layout_pair_id=layout_pair_id,
        environment_seed=environment_seed,
        condition_order=tuple(order),
        conditions=tuple(_condition_tuple(label) for label in order),
        cell_ids=cell_ids,
        block_id=block_id,
        raw_root=RAW_PARENT / layout_pair_id,
        schedule_path=schedule_path,
        schedule_sha256=pilot.sha256_file(schedule_path),
        schedule_row=dict(row),
        contract=contract,
        contract_sha256=pilot.sha256_bytes(pilot.canonical_bytes(contract)),
    )


@contextmanager
def _patched_pilot(values: Mapping[str, Any]) -> Iterator[None]:
    originals = {name: getattr(pilot, name) for name in values}
    for name, value in values.items():
        setattr(pilot, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(pilot, name, value)


def _configuration_values(
    block: DevelopmentBlock, simulator_worker_role: str
) -> dict[str, Any]:
    pilot.require(
        simulator_worker_role in ALLOWED_SIMULATOR_ROLES,
        "development_simulator_role_unsupported",
    )
    return {
        "PHASE": "development",
        "LAYOUT_PAIR_ID": block.layout_pair_id,
        "BLOCK_ID": block.block_id,
        "CONDITIONS": block.conditions,
        "CELL_IDS": block.cell_ids,
        "ENVIRONMENT_SEED": block.environment_seed,
        "RAW_ROOT": block.raw_root,
        "PILOT_CONTRACT": dict(block.contract),
        "PILOT_CONTRACT_SHA256": block.contract_sha256,
        "RUNNER_FILENAME": RUNNER_FILENAME,
        "BEHAVIORAL_PURPOSE": "d1_behavioral_development",
        "BEHAVIORAL_FINALIZE_PURPOSE": "d1_behavioral_development_finalize",
        "EPISODE_ID_PREFIX": f"d1{block.layout_pair_id.lower()}",
        "POLICY_LABEL": "wmf_d1_behavioral_development",
        "SIMULATOR_RECEIPT_FILENAME": SIMULATOR_RECEIPT_FILENAME,
        "GLOBAL_SERVER_LOCK_PATH": GLOBAL_D1_SERVER_LOCK,
        "SERVER_RECEIPT_SCHEMA": SERVER_RECEIPT_SCHEMA,
        "SIMULATOR_RECEIPT_SCHEMA": SIMULATOR_RECEIPT_SCHEMA,
        "CELL_RECEIPT_SCHEMA": CELL_RECEIPT_SCHEMA,
        "SIMULATOR_QUEUE_ROLE": simulator_worker_role,
    }


@contextmanager
def configured_pilot(
    block: DevelopmentBlock, simulator_worker_role: str
) -> Iterator[None]:
    global _CONFIGURATION_ACTIVE
    pilot.require(not _CONFIGURATION_ACTIVE, "development_configuration_overlap")
    _CONFIGURATION_ACTIVE = True
    try:
        with _patched_pilot(_configuration_values(block, simulator_worker_role)):
            yield
    finally:
        _CONFIGURATION_ACTIVE = False


@contextmanager
def configured_p00_pilot() -> Iterator[None]:
    """Temporarily restore P00 constants for deep pilot validation."""

    with _patched_pilot(_PILOT_DEFAULTS):
        yield


def _verify_descriptor(
    value: Any, label: str, *, allowed_root: Path | None = None
) -> dict[str, Any]:
    return pilot._verify_descriptor(value, label, allowed_root=allowed_root)


def _verify_direct_capture_artifact(
    value: Any, *, artifact_root: Path, label: str
) -> dict[str, Any]:
    identity = _verify_descriptor(value, label, allowed_root=artifact_root)
    pilot.require(
        Path(identity["path"]).parent == Path(artifact_root).resolve(),
        "development_capture_artifact_not_direct_child",
        label,
    )
    return identity


def _validate_live_ready_descriptor(
    value: Any, label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Revalidate the full official live server identity from persisted evidence."""

    ready_identity = _verify_descriptor(value, label)
    ready = pilot.load_json(Path(ready_identity["path"]), f"{label}_unreadable")
    contract_identity = _verify_descriptor(
        ready.get("server_contract"), f"{label}_server_contract"
    )
    pilot.require(
        ready.get("server_contract_sha256") == contract_identity["sha256"],
        "persisted_server_contract_binding_changed",
    )
    contract = pilot.load_json(
        Path(contract_identity["path"]), f"{label}_server_contract_unreadable"
    )
    overlay = contract.get("instrumentation_overlay")
    pilot.require(isinstance(overlay, Mapping), "persisted_server_overlay_missing")
    overlay_path = Path(str(overlay.get("path", ""))).resolve()
    expected_suffix = Path(
        "workshops/corl2026_world_models/experiments/forecast_layout/"
        "d1_instrumented_server.py"
    )
    pilot.require(
        len(overlay_path.parents) >= 5
        and Path(*overlay_path.parts[-len(expected_suffix.parts):]) == expected_suffix,
        "persisted_server_overlay_path_changed",
    )
    source_root = overlay_path.parents[4]
    future_root = Path(str(ready.get("future_root", ""))).resolve()
    pilot.validate_live_server_contract_payload(
        contract, source_root=source_root, future_root=future_root
    )
    return ready_identity, ready


def verify_development_fixture_release(
    *, gate_receipt_path: Path, gate_receipt_sha256: str,
    pose_manifest_path: Path, pose_manifest_sha256: str,
    candidate_id: str, layout_arm: str, block: DevelopmentBlock,
) -> dict[str, Any]:
    forecast = Path(__file__).resolve().parent
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import fixed_observation_job as fixed

    try:
        release = fixed.verify_gate_and_pose_manifest(
            gate_receipt_path=Path(gate_receipt_path),
            gate_receipt_sha256=gate_receipt_sha256,
            pose_manifest_path=Path(pose_manifest_path),
            pose_manifest_sha256=pose_manifest_sha256,
            layout_pair_id=block.layout_pair_id,
            candidate_id=candidate_id,
        )
    except BaseException as error:
        raise pilot.D1BehavioralPilotError(
            "development_fixture_release_invalid", str(error)
        ) from error
    pilot.require(layout_arm in {"original", "reflected"}, "development_layout_arm_invalid")
    layouts = release.get("pose_row", {}).get("layouts")
    pilot.require(
        isinstance(layouts, Mapping)
        and set(layouts) == {"original", "reflected"}
        and layout_arm in layouts,
        "development_pose_arms_changed",
    )
    return release


def expected_development_capture_wrapper(
    block: DevelopmentBlock,
) -> dict[str, str]:
    """Return the one released queue wrapper authorized for this layout."""

    release = DEVELOPMENT_CAPTURE_WRAPPER_RELEASES.get(block.layout_pair_id)
    pilot.require(release is not None, "development_capture_wrapper_release_missing")
    job_id = release["job_id"]
    return {
        "job_id": job_id,
        "path": str(
            CONTROL_ROOT
            / "jobs"
            / job_id
            / "publish"
            / "fixed_observation_job_receipt.json"
        ),
        "sha256": release["sha256"],
    }


def validate_released_capture_argument(
    path: Path, expected_sha256: str, *, block: DevelopmentBlock,
) -> dict[str, str]:
    """Fail closed unless argv names the released queue wrapper, not its child."""

    expected = expected_development_capture_wrapper(block)
    supplied = Path(path)
    pilot.require(supplied.is_absolute(), "development_capture_wrapper_path_not_absolute")
    lexical = Path(os.path.abspath(os.fspath(supplied)))
    pilot.require(
        lexical == Path(expected["path"]),
        "development_capture_wrapper_path_changed",
        str(lexical),
    )
    pilot.require(
        expected_sha256 == expected["sha256"],
        "development_capture_wrapper_sha256_changed",
    )
    return expected


def verify_development_capture(
    path: Path, expected_sha256: str, *, block: DevelopmentBlock,
    release: Mapping[str, Any],
) -> dict[str, Any]:
    forecast = Path(__file__).resolve().parent
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import fixed_observation_job as fixed

    identity = pilot.verify_exact_file(path, expected_sha256, "development_capture_receipt")
    receipt = pilot.load_json(Path(identity["path"]), "development_capture_receipt_unreadable")
    expected = {
        "schema_version": FIXED_CAPTURE_QUEUE_SCHEMA,
        "study_namespace": pilot.NAMESPACE,
        "status": "passed",
        "exit_code": 0,
        "layout_pair_id": block.layout_pair_id,
        "candidate_id": release["candidate_id"],
        "environment_seed": block.environment_seed,
        "gate_receipt_sha256": release["gate_receipt"]["sha256"],
        "pose_manifest_sha256": release["pose_manifest"]["sha256"],
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "child_started": True,
        "child_exit_code": 0,
    }
    for key, wanted in expected.items():
        pilot.require(receipt.get(key) == wanted, "development_capture_mismatch", key)
    study_commit = receipt.get("study_commit")
    pilot.require(
        isinstance(study_commit, str)
        and pilot.COMMIT_RE.fullmatch(study_commit) is not None,
        "development_capture_study_commit_invalid",
    )
    job_id = receipt.get("job_id")
    pilot.require(
        isinstance(job_id, str) and pilot.SAFE_ID_RE.fullmatch(job_id) is not None,
        "development_capture_job_id_invalid",
    )
    gpu = receipt.get("gpu_identity")
    pilot.require(isinstance(gpu, Mapping), "development_capture_gpu_identity_missing")
    pilot.require(
        gpu.get("index") == "0"
        and gpu.get("name") == "NVIDIA B200"
        and isinstance(gpu.get("uuid"), str)
        and gpu["uuid"].startswith("GPU-")
        and isinstance(gpu.get("driver_version"), str)
        and bool(gpu["driver_version"])
        and gpu.get("preexisting_compute_process_count") == 0,
        "development_capture_gpu_identity_invalid",
    )
    logs = receipt.get("child_logs")
    pilot.require(
        isinstance(logs, Mapping) and set(logs) == {"stdout", "stderr"},
        "development_capture_child_logs_invalid",
    )
    for name in ("stdout", "stderr"):
        _verify_descriptor(logs[name], f"development_capture_child_{name}")

    raw_identity = _verify_descriptor(
        receipt.get("raw_capture_receipt"), "development_raw_capture_receipt"
    )
    raw = pilot.load_json(Path(raw_identity["path"]), "development_raw_capture_unreadable")
    raw_expected = {
        "schema_version": FIXED_CAPTURE_SCHEMA,
        "study_namespace": pilot.NAMESPACE,
        "status": "passed",
        "study_commit": study_commit,
        "robolab_commit": fixed.ROBOLAB_COMMIT,
        "layout_pair_id": block.layout_pair_id,
        "candidate_id": release["candidate_id"],
        "candidate_payload_sha256": release["candidate_payload_sha256"],
        "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
        "environment_seed": block.environment_seed,
        "layout_arm": "original",
        "command_task_used_for_reset": "left",
        "model_request_count": 0,
        "behavioral_action_count": 0,
    }
    for key, wanted in raw_expected.items():
        pilot.require(raw.get(key) == wanted, "development_raw_capture_mismatch", key)
    capture_id = raw.get("capture_id")
    pilot.require(
        isinstance(capture_id, str)
        and pilot.SAFE_ID_RE.fullmatch(capture_id) is not None
        and capture_id.startswith(f"{block.layout_pair_id}-{block.environment_seed}-"),
        "development_capture_id_invalid",
    )
    pilot.require(
        isinstance(raw.get("captured_at_utc"), str) and bool(raw["captured_at_utc"]),
        "development_capture_timestamp_missing",
    )
    for key in ("gate_receipt", "pose_manifest", "gate_ledger", "gate_attempt_receipt"):
        observed = _verify_descriptor(raw.get(key), f"development_capture_{key}")
        pilot.require(
            observed == release[key],
            "development_capture_release_binding_changed",
            key,
        )

    reset = raw.get("settled_reset_receipt")
    pilot.require(isinstance(reset, Mapping), "development_capture_reset_missing")
    reset_checks = {
        "schema": reset.get("schema_version")
        == "wmf-forecast-layout-settled-reset-receipt-v1",
        "passed": reset.get("passed") is True,
        "settled": reset.get("settled") is True,
        "left_not_success": reset.get("left_success") is False,
        "right_not_success": reset.get("right_success") is False,
        "released_boolean": type(reset.get("released")) is bool,
        "returned": reset.get("settled_observation_returned") is True,
        "no_model": reset.get("model_request_count_during_settle") == 0,
        "counter_zero": reset.get("episode_length_buf_reset_to_zero") is True,
    }
    pilot.require(
        all(reset_checks.values()),
        "development_capture_reset_invalid",
        ",".join(sorted(key for key, value in reset_checks.items() if not value)),
    )
    expected_reset_identity = (
        f"fixed-observation:{study_commit}:{block.layout_pair_id}:{block.environment_seed}"
    )
    pilot.require(
        reset.get("reset_identity") == expected_reset_identity
        and raw.get("settled_reset_identity") == expected_reset_identity,
        "development_capture_reset_identity_changed",
    )
    pilot.require(
        reset.get("pose_manifest_sha256") == release["pose_manifest"]["sha256"],
        "development_capture_reset_pose_changed",
    )
    observation_hashes = reset.get("initial_observation_hashes")
    pilot.require(
        isinstance(observation_hashes, Mapping)
        and pilot.SHA256_RE.fullmatch(
            str(observation_hashes.get("combined_sha256", ""))
        )
        is not None,
        "development_capture_reset_observation_identity_invalid",
    )
    settle = reset.get("settle_evidence")
    pilot.require(isinstance(settle, Mapping), "development_capture_settle_missing")
    settle_steps = settle.get("settle_steps")
    stability_steps = settle.get("stability_window_steps")
    pilot.require(
        type(settle_steps) is int
        and settle_steps > 0
        and type(stability_steps) is int
        and stability_steps >= 2,
        "development_capture_settle_counts_invalid",
    )
    pilot.require(
        raw.get("settling_hold_action_count") == settle_steps + stability_steps,
        "development_capture_settle_count_binding_changed",
    )
    fresh = raw.get("fresh_physical_checks")
    pilot.require(isinstance(fresh, Mapping), "development_capture_physical_checks_missing")
    for name in ("collision", "visibility"):
        row = reset.get(f"{name}_evidence")
        pilot.require(
            isinstance(row, Mapping)
            and row.get("passed") is True
            and isinstance(fresh.get(name), Mapping)
            and fresh[name].get("passed") is True
            and fresh[name] == row,
            "development_capture_physical_check_invalid",
            name,
        )

    native = raw.get("native_clock")
    source_capture = raw.get("source_capture")
    pilot.require(
        isinstance(native, Mapping) and isinstance(source_capture, Mapping),
        "development_capture_native_timing_missing",
    )
    pilot.require(
        type(native.get("physics_step")) is int
        and native["physics_step"] >= 0
        and isinstance(native.get("physics_time_s"), (int, float))
        and not isinstance(native.get("physics_time_s"), bool)
        and native["physics_time_s"] >= 0
        and type(native.get("control_step_since_physical_reset")) is int
        and native["control_step_since_physical_reset"] > 0
        and native.get("behavioral_episode_step") == 0,
        "development_capture_native_counters_invalid",
    )
    for key in (
        "physics_step_source", "physics_time_source", "control_step_source",
        "behavioral_episode_step_source", "timing_claim_boundary",
    ):
        pilot.require(
            isinstance(native.get(key), str) and bool(native[key]),
            "development_capture_native_counter_source_missing",
            key,
        )
    read_window = native.get("host_read_window")
    pilot.require(isinstance(read_window, Mapping), "development_capture_read_window_missing")
    window_values = [
        read_window.get("started_wall_time_ns"),
        read_window.get("finished_wall_time_ns"),
        read_window.get("started_monotonic_ns"),
        read_window.get("finished_monotonic_ns"),
    ]
    pilot.require(
        all(type(value) is int and value > 0 for value in window_values)
        and window_values[0] <= window_values[1]
        and window_values[2] <= window_values[3],
        "development_capture_read_window_invalid",
    )
    runtime_configuration = native.get("runtime_configuration")
    pilot.require(
        isinstance(runtime_configuration, Mapping)
        and isinstance(runtime_configuration.get("physics_dt_s"), (int, float))
        and not isinstance(runtime_configuration.get("physics_dt_s"), bool)
        and runtime_configuration["physics_dt_s"] > 0
        and type(runtime_configuration.get("decimation")) is int
        and runtime_configuration["decimation"] > 0
        and type(runtime_configuration.get("render_interval")) is int
        and runtime_configuration["render_interval"] > 0,
        "development_capture_runtime_configuration_invalid",
    )
    cameras = native.get("camera_counters")
    frame_ids = source_capture.get("camera_frame_ids")
    timestamps = source_capture.get("camera_capture_time_ns")
    timestamp_sources = source_capture.get("camera_timestamp_source")
    expected_cameras = set(fixed.RAW_CAMERAS)
    for value, reason in (
        (cameras, "development_capture_camera_inventory_invalid"),
        (frame_ids, "development_capture_frame_inventory_invalid"),
        (timestamps, "development_capture_timestamp_inventory_invalid"),
        (timestamp_sources, "development_capture_timestamp_source_inventory_invalid"),
    ):
        pilot.require(
            isinstance(value, Mapping) and set(value) == expected_cameras,
            reason,
        )
    pilot.require(
        source_capture.get("simulator_observation_id")
        == f"{block.layout_pair_id}_original_settled_observation_000000",
        "development_capture_observation_identity_changed",
    )
    for name in fixed.RAW_CAMERAS:
        row = cameras[name]
        pilot.require(isinstance(row, Mapping), "development_capture_camera_counter_invalid", name)
        frame_id = row.get("frame_id")
        native_frame = type(frame_id) is int and frame_id >= 0
        rgb_frame = (
            isinstance(frame_id, str)
            and re.fullmatch(r"rgb-sha256:[0-9a-f]{64}", frame_id) is not None
        )
        pilot.require(native_frame or rgb_frame, "development_capture_frame_id_invalid", name)
        if rgb_frame:
            rgb_identity = row.get("rgb_array_identity")
            pilot.require(
                row.get("frame_identity_source") == "exact returned RGB array value identity"
                and row.get("native_frame_counter") is None
                and row.get("native_frame_counter_status")
                == "unavailable_in_pinned_isaaclab_sensorbase"
                and isinstance(rgb_identity, Mapping)
                and rgb_identity.get("value_sha256")
                == frame_id.removeprefix("rgb-sha256:"),
                "development_capture_rgb_frame_identity_invalid",
                name,
            )
        timestamp = row.get("capture_time_ns")
        source = row.get("native_capture_time_source")
        pilot.require(
            type(timestamp) is int
            and timestamp >= 0
            and isinstance(source, str)
            and bool(source)
            and frame_ids[name] == frame_id
            and timestamps[name] == timestamp
            and timestamp_sources[name] == source,
            "development_capture_camera_timing_binding_changed",
            name,
        )

    preprocessing = raw.get("preprocessing")
    pilot.require(
        isinstance(preprocessing, Mapping)
        and preprocessing.get("model_request_count") == 0,
        "development_capture_preprocessing_invalid",
    )
    d1_preprocessing = preprocessing.get("D1")
    pilot.require(isinstance(d1_preprocessing, Mapping), "development_capture_d1_preprocessing_missing")
    pilot.require(
        d1_preprocessing.get("extraction_method") == "_extract_observation"
        and d1_preprocessing.get("packing_method") == "_pack_request"
        and d1_preprocessing.get("wire_array_keys") == list(fixed.D1_ARRAY_KEYS)
        and d1_preprocessing.get("configuration")
        == {
            "cam2_source": "right", "resize": "pad",
            "image_height": 180, "image_width": 320,
        },
        "development_capture_d1_preprocessing_changed",
    )
    for key in ("overlay_source", "official_client_source"):
        _verify_descriptor(
            d1_preprocessing.get(key), f"development_capture_d1_{key}"
        )

    artifact_root = Path(raw_identity["path"]).parent
    artifacts = raw.get("artifacts")
    required_artifacts = {
        "raw_settled_observation", "simulator_state", "preprocessing_intermediates",
        "N3", "D1",
    }
    pilot.require(
        isinstance(artifacts, Mapping) and set(artifacts) == required_artifacts,
        "development_capture_artifact_inventory_changed",
    )
    artifact_identities: dict[str, dict[str, Any]] = {}
    artifact_logical: dict[str, dict[str, Any]] = {}
    for name in sorted(required_artifacts):
        descriptor = artifacts[name]
        pilot.require(isinstance(descriptor, Mapping), "development_capture_artifact_invalid", name)
        arrays = descriptor.get("arrays")
        pilot.require(isinstance(arrays, Mapping), "development_capture_artifact_arrays_missing", name)
        artifact_identities[name] = _verify_direct_capture_artifact(
            descriptor, artifact_root=artifact_root,
            label=f"development_capture_artifact_{name}",
        )
        artifact_logical[name] = {
            "sha256": descriptor.get("sha256"),
            "bytes": descriptor.get("bytes"),
            "arrays": arrays,
        }
    pilot.require(
        set(artifacts["D1"]["arrays"]) == set(fixed.D1_ARRAY_KEYS),
        "development_capture_d1_array_descriptors_changed",
    )
    pilot.require(
        raw.get("artifact_logical_sha256")
        == fixed.sha256_bytes(fixed.compact_canonical_bytes(artifact_logical)),
        "development_capture_artifact_logical_hash_changed",
    )

    fixtures = raw.get("model_fixtures")
    pilot.require(
        isinstance(fixtures, Mapping) and set(fixtures) == {"N3", "D1"},
        "development_capture_model_fixtures_missing",
    )
    pilot.require(
        receipt.get("model_fixtures") == fixtures,
        "development_capture_queue_fixture_binding_changed",
    )
    d1 = fixtures.get("D1")
    pilot.require(isinstance(d1, Mapping), "development_capture_d1_fixture_missing")
    pilot.require(
        d1.get("interface") == "official conditional DreamZero six-array request fixture"
        and d1.get("array_keys") == list(fixed.D1_ARRAY_KEYS),
        "development_capture_d1_array_keys_changed",
    )
    fixture_identity = _verify_descriptor(d1.get("fixture"), "development_capture_d1_fixture")
    pilot.require(
        fixture_identity == artifact_identities["D1"]
        and {
            key: artifacts["D1"].get(key) for key in ("path", "bytes", "sha256")
        }
        == dict(d1["fixture"]),
        "development_capture_d1_artifact_binding_changed",
    )
    n3 = fixtures.get("N3")
    pilot.require(
        isinstance(n3, Mapping)
        and n3.get("interface")
        == "official packed observation/image plus joint/gripper arrays"
        and n3.get("array_keys") == list(fixed.N3_ARRAY_KEYS),
        "development_capture_n3_fixture_invalid",
    )
    n3_identity = _verify_descriptor(n3.get("fixture"), "development_capture_n3_fixture")
    pilot.require(
        n3_identity == artifact_identities["N3"]
        and {
            key: artifacts["N3"].get(key) for key in ("path", "bytes", "sha256")
        }
        == dict(n3["fixture"]),
        "development_capture_n3_artifact_binding_changed",
    )

    runtime = raw.get("runtime_identity")
    pilot.require(isinstance(runtime, Mapping), "development_capture_runtime_identity_missing")
    pilot.require(
        isinstance(runtime.get("pod"), str)
        and bool(runtime["pod"])
        and (runtime.get("pod_uid") is None or isinstance(runtime.get("pod_uid"), str))
        and isinstance(runtime.get("python"), str)
        and bool(runtime["python"])
        and isinstance(runtime.get("python_executable"), str)
        and Path(runtime["python_executable"]).is_absolute()
        and runtime.get("renderer") == "realtime"
        and runtime.get("rendering_type") == "balanced"
        and runtime.get("device") == "cuda:0",
        "development_capture_runtime_identity_invalid",
    )
    for key in ("robolab_module", "task_file"):
        _verify_descriptor(runtime.get(key), f"development_capture_runtime_{key}")
    pilot.require(
        Path(runtime["task_file"]["path"]).name == "original_left.py",
        "development_capture_runtime_task_changed",
    )
    pilot.require(
        pilot.file_identity(Path(raw_identity["path"])) == raw_identity,
        "development_raw_capture_changed_during_validation",
    )
    pilot.require(
        pilot.file_identity(Path(identity["path"])) == identity,
        "development_capture_changed_during_validation",
    )
    return {
        "capture_receipt": identity,
        "raw_capture_receipt": raw_identity,
        "d1_fixed_observation": fixture_identity,
    }


def verify_recorder_qualification(path: Path, expected_sha256: str) -> dict[str, Any]:
    identity = pilot.verify_exact_file(path, expected_sha256, "recorder_receipt")
    receipt = pilot.load_json(Path(identity["path"]), "recorder_receipt_unreadable")
    expected = {
        "schema_version": pilot.RECORDER_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "layout_pair_id": "P00",
        "environment_seed": 2026091000,
        "actions_executed": pilot.ACTION_CAP,
        "observation_count": pilot.OBSERVATION_COUNT,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "recording_qualification_count": 1,
    }
    for key, wanted in expected.items():
        pilot.require(receipt.get(key) == wanted, "recorder_qualification_mismatch", key)
    child_identity = _verify_descriptor(receipt.get("raw_child_receipt"), "recorder_child_receipt")
    child = pilot.load_json(Path(child_identity["path"]), "recorder_child_receipt_unreadable")
    pilot.require(child.get("native_timing_supported") is True, "recorder_native_timing_unqualified")
    timing_identity = _verify_descriptor(child.get("timing_support"), "recorder_native_timing")
    timing = pilot.load_json(Path(timing_identity["path"]), "recorder_native_timing_unreadable")
    pilot.require(
        timing.get("supported") is True and timing.get("status") == "supported",
        "recorder_native_timing_unqualified",
    )
    return {
        "recorder_receipt": identity,
        "recorder_child_receipt": child_identity,
        "native_timing_support": timing_identity,
    }


def verify_passed_p00_pair(
    *, simulator_receipt_path: Path, simulator_receipt_sha256: str,
    server_receipt_path: Path, server_receipt_sha256: str,
    recorder_receipt_sha256: str, d1_qualification_receipt_sha256: str,
) -> dict[str, Any]:
    """Deeply authenticate the complete paired P00 behavioral prerequisite."""

    simulator_identity = pilot.verify_exact_file(
        simulator_receipt_path, simulator_receipt_sha256, "p00_simulator_receipt"
    )
    server_identity = pilot.verify_exact_file(
        server_receipt_path, server_receipt_sha256, "p00_server_receipt"
    )
    simulator = pilot.load_json(
        Path(simulator_identity["path"]), "p00_simulator_receipt_unreadable"
    )
    server = pilot.load_json(Path(server_identity["path"]), "p00_server_receipt_unreadable")
    expected_simulator = {
        "schema_version": P00_SIMULATOR_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "study_id": pilot.STUDY_ID,
        "namespace": pilot.NAMESPACE,
        "block_id": P00_BLOCK_ID,
        "phase": "pilot",
        "layout_pair_id": "P00",
        "model_config": "D1",
        "effective_model_noise_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
        "condition_order": list(P00_CONDITION_ORDER),
        "cell_ids": list(P00_CELL_IDS),
        "all_simulator_children_reaped": True,
    }
    for key, wanted in expected_simulator.items():
        pilot.require(simulator.get(key) == wanted, "p00_simulator_receipt_mismatch", key)
    counts = simulator.get("counts")
    expected_counts = {
        "planned_behavioral_cells": 4,
        "launched_behavioral_cells": 4,
        "completed_valid_behavioral_cells": 4,
        "technically_invalid_behavioral_cells": 0,
        "right_censored_behavioral_cells": 0,
        "unrun_behavioral_cells": 0,
        "actual_behavioral_actions": 4 * pilot.ACTION_CAP,
        "actual_behavioral_model_requests": 4 * pilot.REQUEST_COUNT,
        "new_generation_qualification_requests": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "recorder_only_episodes_counted_as_behavioral": 0,
    }
    pilot.require(isinstance(counts, Mapping), "p00_simulator_counts_missing")
    for key, wanted in expected_counts.items():
        pilot.require(counts.get(key) == wanted, "p00_simulator_counts_mismatch", key)
    study_commit = simulator.get("study_commit")
    pilot.require(
        isinstance(study_commit, str) and pilot.COMMIT_RE.fullmatch(study_commit) is not None,
        "p00_simulator_study_commit_invalid",
    )
    descriptors = simulator.get("cell_receipts")
    pilot.require(
        isinstance(descriptors, list) and len(descriptors) == 4,
        "p00_simulator_cell_inventory_invalid",
    )
    episode_ids: list[str] = []
    session_ids: list[str] = []
    cells: list[dict[str, Any]] = []
    with configured_p00_pilot():
        for index, descriptor in enumerate(descriptors):
            observed = _verify_descriptor(descriptor, f"p00_cell_{index}")
            cell, identity = _PILOT_VALIDATE_PASSED_CELL(
                Path(observed["path"]), condition_index=index, study_commit=study_commit
            )
            pilot.require(identity == observed, "p00_cell_descriptor_changed")
            episode_ids.append(cell["server_begin_receipt"]["episode_context_id"])
            session_ids.append(cell["server_begin_receipt"]["client_session_id"])
            cells.append(observed)
    pilot.require(len(episode_ids) == len(set(episode_ids)), "p00_episode_context_reused")
    pilot.require(len(session_ids) == len(set(session_ids)), "p00_client_session_reused")
    pilot.require(not set(episode_ids).intersection(session_ids), "p00_context_identity_alias")

    prerequisites = simulator.get("prerequisites")
    pilot.require(isinstance(prerequisites, Mapping), "p00_prerequisites_missing")
    recorder = _verify_descriptor(prerequisites.get("recorder_receipt"), "p00_recorder_receipt")
    qualification = _verify_descriptor(
        prerequisites.get("d1_qualification_receipt"), "p00_d1_qualification_receipt"
    )
    pilot.require(
        recorder["sha256"] == recorder_receipt_sha256,
        "p00_recorder_receipt_binding_changed",
    )
    pilot.require(
        qualification["sha256"] == d1_qualification_receipt_sha256,
        "p00_d1_qualification_binding_changed",
    )
    with configured_p00_pilot():
        ready, _ready_value = _validate_live_ready_descriptor(
            simulator.get("server_ready"), "p00_server_ready"
        )
    claim = _verify_descriptor(simulator.get("simulator_claim"), "p00_simulator_claim")

    expected_server = {
        "schema_version": P00_SERVER_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "run_id": simulator.get("run_id"),
        "server_job_id": simulator.get("server_job_id"),
        "paired_simulator_job_id": simulator.get("simulator_job_id"),
        "study_commit": study_commit,
        "block_id": P00_BLOCK_ID,
        "all_server_children_reaped": True,
    }
    for key, wanted in expected_server.items():
        pilot.require(server.get(key) == wanted, "p00_server_receipt_mismatch", key)
    pilot.require(
        _verify_descriptor(server.get("server_ready"), "p00_paired_server_ready") == ready,
        "p00_pair_ready_binding_changed",
    )
    pilot.require(
        _verify_descriptor(server.get("simulator_claim"), "p00_paired_simulator_claim") == claim,
        "p00_pair_claim_binding_changed",
    )
    server_qualification = _verify_descriptor(
        server.get("d1_qualification_receipt"), "p00_server_d1_qualification"
    )
    pilot.require(server_qualification == qualification, "p00_pair_qualification_changed")
    terminal_identity = _verify_descriptor(
        server.get("simulator_terminal"), "p00_simulator_terminal"
    )
    terminal = pilot.load_json(
        Path(terminal_identity["path"]), "p00_simulator_terminal_unreadable"
    )
    pilot.require(
        terminal.get("status") == "passed"
        and terminal.get("safe_for_server_shutdown") is True
        and terminal.get("all_simulator_children_reaped") is True
        and terminal.get("run_id") == simulator.get("run_id")
        and terminal.get("server_job_id") == simulator.get("server_job_id")
        and terminal.get("simulator_job_id") == simulator.get("simulator_job_id")
        and terminal.get("block_id") == P00_BLOCK_ID,
        "p00_pair_terminal_invalid",
    )
    terminal_receipt = _verify_descriptor(
        terminal.get("simulator_receipt"), "p00_terminal_simulator_receipt"
    )
    pilot.require(
        terminal_receipt["sha256"] == simulator_identity["sha256"],
        "p00_terminal_simulator_receipt_changed",
    )
    process_exit = server.get("server_process_exit")
    pilot.require(
        isinstance(process_exit, Mapping)
        and process_exit.get("status") == "reaped"
        and process_exit.get("reaped") is True,
        "p00_server_process_not_reaped",
    )
    return {
        "simulator_receipt": simulator_identity,
        "server_receipt": server_identity,
        "cell_receipts": cells,
        "server_ready": ready,
        "simulator_claim": claim,
        "simulator_terminal": terminal_identity,
        "behavioral_cells": 4,
        "behavioral_actions": 4 * pilot.ACTION_CAP,
        "behavioral_model_requests": 4 * pilot.REQUEST_COUNT,
        "generation_qualification_requests_rerun": 0,
    }


def validate_prerequisites(
    args: argparse.Namespace, block: DevelopmentBlock
) -> dict[str, Any]:
    validate_released_capture_argument(
        Path(args.capture_receipt), args.capture_receipt_sha256, block=block
    )
    original = verify_development_fixture_release(
        gate_receipt_path=Path(args.gate_receipt),
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=Path(args.pose_manifest),
        pose_manifest_sha256=args.pose_manifest_sha256,
        candidate_id=args.candidate_id,
        layout_arm="original",
        block=block,
    )
    reflected = verify_development_fixture_release(
        gate_receipt_path=Path(args.gate_receipt),
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=Path(args.pose_manifest),
        pose_manifest_sha256=args.pose_manifest_sha256,
        candidate_id=args.candidate_id,
        layout_arm="reflected",
        block=block,
    )
    pilot.require(
        original["candidate_id"] == reflected["candidate_id"],
        "development_pose_arms_bind_different_candidates",
    )
    capture = verify_development_capture(
        Path(args.capture_receipt), args.capture_receipt_sha256,
        block=block, release=original,
    )
    recorder = verify_recorder_qualification(
        Path(args.recorder_receipt), args.recorder_receipt_sha256
    )
    d1_qualification = pilot.validate_d1_qualification(
        Path(args.d1_qualification_receipt), args.d1_qualification_receipt_sha256
    )
    p00 = verify_passed_p00_pair(
        simulator_receipt_path=Path(args.pilot_simulator_receipt),
        simulator_receipt_sha256=args.pilot_simulator_receipt_sha256,
        server_receipt_path=Path(args.pilot_server_receipt),
        server_receipt_sha256=args.pilot_server_receipt_sha256,
        recorder_receipt_sha256=recorder["recorder_receipt"]["sha256"],
        d1_qualification_receipt_sha256=d1_qualification["sha256"],
    )
    return {
        "development_gate_receipt": original["gate_receipt"],
        "development_pose_manifest": original["pose_manifest"],
        "development_gate_ledger": original["gate_ledger"],
        "development_gate_attempt_receipt": original["gate_attempt_receipt"],
        "candidate_id": original["candidate_id"],
        "candidate_payload_sha256": original["candidate_payload_sha256"],
        "accepted_gate_record_sha256": original["accepted_gate_record_sha256"],
        **capture,
        **recorder,
        "d1_qualification_receipt": d1_qualification,
        "p00_paired_pilot": p00,
        "generation_qualification_requests_reused_not_rerun": 6,
        "mapping_qualification": (
            "The passed P00 D1 pilot authenticates all four physical forecast/action/camera "
            "mappings. This development layout additionally requires its own fixed observation."
        ),
    }


def execution_prerequisites_payload(
    block: DevelopmentBlock, prerequisites: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the exact immutable prerequisite bundle bound into every cell."""

    return {
        "schema_version": EXECUTION_PREREQUISITES_SCHEMA,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "phase": "development",
        "layout_pair_id": block.layout_pair_id,
        "block_id": block.block_id,
        "development_contract_sha256": block.contract_sha256,
        "prerequisites": dict(prerequisites),
    }


def execution_prerequisites_sha256(
    block: DevelopmentBlock, prerequisites: Mapping[str, Any]
) -> str:
    return pilot.sha256_bytes(
        pilot.canonical_bytes(execution_prerequisites_payload(block, prerequisites))
    )


def validate_server_ready_prerequisites(
    *args: Any, expected_prerequisites_sha256: str, **kwargs: Any
) -> dict[str, Any]:
    ready_bundle = _PILOT_VALIDATE_SERVER_READY(*args, **kwargs)
    ready = ready_bundle.get("ready")
    pilot.require(
        isinstance(ready, Mapping)
        and ready.get("development_prerequisites_sha256")
        == expected_prerequisites_sha256,
        "development_server_simulator_prerequisites_changed",
    )
    return ready_bundle


def verify_execution_prerequisites(
    path: Path, expected_sha256: str, *, block: DevelopmentBlock,
    expected_prerequisites: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = pilot.verify_exact_file(
        path, expected_sha256, "development_execution_prerequisites"
    )
    bundle = pilot.load_json(
        Path(identity["path"]), "development_execution_prerequisites_unreadable"
    )
    expected = {
        "schema_version": EXECUTION_PREREQUISITES_SCHEMA,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "phase": "development",
        "layout_pair_id": block.layout_pair_id,
        "block_id": block.block_id,
        "development_contract_sha256": block.contract_sha256,
    }
    for key, wanted in expected.items():
        pilot.require(
            bundle.get(key) == wanted,
            "development_execution_prerequisites_mismatch",
            key,
        )
    observed = bundle.get("prerequisites")
    pilot.require(
        isinstance(observed, Mapping),
        "development_execution_prerequisites_payload_missing",
    )
    if expected_prerequisites is not None:
        pilot.require(
            observed == expected_prerequisites,
            "development_execution_prerequisites_changed",
        )
    pilot.require(
        pilot.sha256_bytes(pilot.canonical_bytes(bundle)) == expected_sha256,
        "development_execution_prerequisites_canonical_hash_changed",
    )
    return identity, bundle


def write_execution_prerequisites(
    path: Path, *, block: DevelopmentBlock, prerequisites: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    payload = execution_prerequisites_payload(block, prerequisites)
    digest = pilot.sha256_bytes(pilot.canonical_bytes(payload))
    path = Path(path)
    if not path.exists():
        _PILOT_IMMUTABLE_JSON(path, payload)
    identity, _bundle = verify_execution_prerequisites(
        path, digest, block=block, expected_prerequisites=prerequisites
    )
    return identity, digest


def _descriptor_option(argv: Sequence[Any], option: str) -> str:
    matches = [index for index, item in enumerate(argv) if item == option]
    pilot.require(
        len(matches) == 1 and matches[0] + 1 < len(argv),
        "development_queue_option_invalid",
        option,
    )
    value = argv[matches[0] + 1]
    pilot.require(isinstance(value, str), "development_queue_option_invalid", option)
    return value


def validate_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str,
    expected_role: str, expected_mode: str, paired_job_id: str,
    run_id: str, block: DevelopmentBlock, simulator_worker_role: str,
) -> dict[str, Any]:
    identity = _PILOT_VALIDATE_QUEUE_INVOCATION(
        source_root=source_root,
        job_dir=job_dir,
        study_commit=study_commit,
        job_id=job_id,
        expected_role=expected_role,
    )
    descriptor = pilot.load_json(
        Path(job_dir).resolve() / "descriptor.json", "queue_descriptor_unreadable"
    )
    argv = descriptor.get("argv")
    pilot.require(isinstance(argv, list) and len(argv) >= 4, "development_queue_argv_invalid")
    pilot.require(argv[0] == "/usr/bin/python3", "development_queue_python_changed")
    pilot.require(
        argv[1]
        == "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        + RUNNER_FILENAME,
        "development_queue_runner_changed",
    )
    pilot.require(argv[2] == expected_mode, "development_queue_mode_changed")
    expected_options = {
        "--layout-pair-id": block.layout_pair_id,
        "--simulator-worker-role": simulator_worker_role,
        "--study-commit": study_commit,
        "--job-id": job_id,
        "--run-id": run_id,
        (
            "--simulator-job-id"
            if expected_mode == "server-job"
            else "--server-job-id"
        ): paired_job_id,
    }
    for option, wanted in expected_options.items():
        pilot.require(
            _descriptor_option(argv, option) == wanted,
            "development_queue_pairing_changed",
            option,
        )
    observed = pilot.file_identity(Path(job_dir).resolve() / "descriptor.json")
    pilot.require(
        all(identity.get(key) == observed[key] for key in ("path", "bytes", "sha256")),
        "development_queue_descriptor_changed_during_validation",
    )
    return {**observed, "role": expected_role, "job_id": job_id}


def validate_preflight_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str,
    queue_role: str, block: DevelopmentBlock, simulator_worker_role: str,
) -> dict[str, Any]:
    """Authenticate a detached prerequisite-only queue descriptor."""

    identity = _PILOT_VALIDATE_QUEUE_INVOCATION(
        source_root=source_root,
        job_dir=job_dir,
        study_commit=study_commit,
        job_id=job_id,
        expected_role=queue_role,
    )
    descriptor_path = Path(job_dir).resolve() / "descriptor.json"
    descriptor = pilot.load_json(descriptor_path, "queue_descriptor_unreadable")
    argv = descriptor.get("argv")
    pilot.require(isinstance(argv, list) and len(argv) >= 4, "development_queue_argv_invalid")
    pilot.require(argv[0] == "/usr/bin/python3", "development_queue_python_changed")
    pilot.require(
        argv[1]
        == "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        + RUNNER_FILENAME,
        "development_queue_runner_changed",
    )
    pilot.require(
        argv[2] == "prerequisite-preflight",
        "development_queue_mode_changed",
    )
    expected_options = {
        "--layout-pair-id": block.layout_pair_id,
        "--simulator-worker-role": simulator_worker_role,
        "--queue-role": queue_role,
        "--study-commit": study_commit,
        "--job-id": job_id,
    }
    capture = expected_development_capture_wrapper(block)
    expected_options.update(
        {
            "--capture-receipt": capture["path"],
            "--capture-receipt-sha256": capture["sha256"],
        }
    )
    for option, wanted in expected_options.items():
        pilot.require(
            _descriptor_option(argv, option) == wanted,
            "development_preflight_queue_option_changed",
            option,
        )
    observed = pilot.file_identity(descriptor_path)
    pilot.require(
        all(identity.get(key) == observed[key] for key in ("path", "bytes", "sha256")),
        "development_queue_descriptor_changed_during_validation",
    )
    return {**observed, "role": queue_role, "job_id": job_id}


def build_cell_command(
    *, source_root: Path, attempt_root: Path, study_commit: str,
    gate_receipt: Path, gate_receipt_sha256: str,
    pose_manifest: Path, pose_manifest_sha256: str,
    capture_receipt: Path, capture_receipt_sha256: str,
    execution_prerequisites: Path, execution_prerequisites_sha256: str,
    candidate_id: str, simulator_worker_role: str,
    run_id: str, server_job_id: str, server_ready_sha256: str,
    simulator_claim_sha256: str, lease_token: str,
    future_root: Path, condition_index: int, block: DevelopmentBlock,
) -> list[str]:
    pilot.require(0 <= condition_index < len(block.conditions), "cell_condition_index_invalid")
    layout_arm, command, _task = block.conditions[condition_index]
    script = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout"
        / RUNNER_FILENAME
    )
    return [
        os.path.abspath(os.fspath(pilot.ROBOLAB_PYTHON)),
        str(script),
        "cell",
        "--layout-pair-id", block.layout_pair_id,
        "--simulator-worker-role", simulator_worker_role,
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--candidate-id", candidate_id,
        "--gate-receipt", str(Path(gate_receipt).resolve()),
        "--gate-receipt-sha256", gate_receipt_sha256,
        "--pose-manifest", str(Path(pose_manifest).resolve()),
        "--pose-manifest-sha256", pose_manifest_sha256,
        "--capture-receipt", str(Path(capture_receipt).resolve()),
        "--capture-receipt-sha256", capture_receipt_sha256,
        "--execution-prerequisites", str(Path(execution_prerequisites).resolve()),
        "--execution-prerequisites-sha256", execution_prerequisites_sha256,
        "--run-id", run_id,
        "--server-job-id", server_job_id,
        "--server-ready-sha256", server_ready_sha256,
        "--simulator-claim-sha256", simulator_claim_sha256,
        "--lease-token", lease_token,
        "--future-root", str(Path(future_root).resolve()),
        "--layout-arm", layout_arm,
        "--command", command,
        "--condition-index", str(condition_index),
        "--remote-host", pilot.SERVICE_HOST,
        "--remote-port", str(pilot.SERVICE_PORT),
    ]


def _fixture_from_args(args: argparse.Namespace, block: DevelopmentBlock) -> dict[str, Any]:
    prerequisites_identity, prerequisites_bundle = verify_execution_prerequisites(
        Path(args.execution_prerequisites),
        args.execution_prerequisites_sha256,
        block=block,
    )
    pilot.require(
        Path(prerequisites_identity["path"])
        == Path(args.attempt_root).resolve() / "execution_prerequisites.json",
        "development_execution_prerequisites_path_changed",
    )
    prerequisites = prerequisites_bundle["prerequisites"]
    release = verify_development_fixture_release(
        gate_receipt_path=Path(args.gate_receipt),
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=Path(args.pose_manifest),
        pose_manifest_sha256=args.pose_manifest_sha256,
        candidate_id=args.candidate_id,
        layout_arm=args.layout_arm,
        block=block,
    )
    capture = verify_development_capture(
        Path(args.capture_receipt), args.capture_receipt_sha256,
        block=block, release=release,
    )
    expected = {
        "candidate_id": release["candidate_id"],
        "candidate_payload_sha256": release["candidate_payload_sha256"],
        "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
        "development_gate_receipt": release["gate_receipt"],
        "development_pose_manifest": release["pose_manifest"],
        **capture,
    }
    for key, wanted in expected.items():
        pilot.require(
            prerequisites.get(key) == wanted,
            "development_cell_prerequisite_bundle_changed",
            key,
        )
    for key in (
        "recorder_receipt", "recorder_child_receipt", "native_timing_support",
        "d1_qualification_receipt",
    ):
        _verify_descriptor(
            prerequisites.get(key), f"development_cell_prerequisite_{key}"
        )
    p00 = prerequisites.get("p00_paired_pilot")
    pilot.require(
        isinstance(p00, Mapping)
        and p00.get("behavioral_cells") == 4
        and p00.get("behavioral_actions") == 4 * pilot.ACTION_CAP
        and p00.get("behavioral_model_requests") == 4 * pilot.REQUEST_COUNT
        and p00.get("generation_qualification_requests_rerun") == 0,
        "development_cell_p00_prerequisite_invalid",
    )
    for key in ("simulator_receipt", "server_receipt"):
        _verify_descriptor(p00.get(key), f"development_cell_p00_{key}")
    return {
        "candidate_id": release["candidate_id"],
        "candidate_payload_sha256": release["candidate_payload_sha256"],
        "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
        "gate_receipt": release["gate_receipt"],
        "pose_manifest": release["pose_manifest"],
        **capture,
        "execution_prerequisites": prerequisites_identity,
        "execution_prerequisites_sha256": args.execution_prerequisites_sha256,
    }


def validate_cells_bind_fixture(
    cells: Sequence[Mapping[str, Any]], *, prerequisites: Mapping[str, Any],
    block: DevelopmentBlock,
) -> None:
    expected_prerequisites_sha256 = execution_prerequisites_sha256(block, prerequisites)
    expected = {
        "candidate_id": prerequisites["candidate_id"],
        "candidate_payload_sha256": prerequisites["candidate_payload_sha256"],
        "accepted_gate_record_sha256": prerequisites["accepted_gate_record_sha256"],
        "gate_receipt": prerequisites["development_gate_receipt"],
        "pose_manifest": prerequisites["development_pose_manifest"],
        "capture_receipt": prerequisites["capture_receipt"],
        "raw_capture_receipt": prerequisites["raw_capture_receipt"],
        "d1_fixed_observation": prerequisites["d1_fixed_observation"],
    }
    for cell in cells:
        fixture = cell.get("development_fixture")
        pilot.require(isinstance(fixture, Mapping), "development_cell_fixture_missing")
        for key, wanted in expected.items():
            pilot.require(
                fixture.get(key) == wanted,
                "development_cell_fixture_binding_changed",
                key,
            )
        prerequisites_identity = _verify_descriptor(
            fixture.get("execution_prerequisites"),
            "development_cell_execution_prerequisites",
        )
        pilot.require(
            fixture.get("execution_prerequisites_sha256")
            == expected_prerequisites_sha256
            and prerequisites_identity["sha256"] == expected_prerequisites_sha256,
            "development_cell_prerequisite_binding_changed",
        )
        _identity, bundle = verify_execution_prerequisites(
            Path(prerequisites_identity["path"]),
            expected_prerequisites_sha256,
            block=block,
            expected_prerequisites=prerequisites,
        )
        pilot.require(
            bundle == execution_prerequisites_payload(block, prerequisites),
            "development_cell_prerequisite_binding_changed",
        )


def _cell_simulator_role(path: Path) -> str:
    receipt = pilot.load_json(Path(path), "development_cell_receipt_unreadable")
    claim_identity = _verify_descriptor(
        receipt.get("simulator_claim"), "development_cell_simulator_claim"
    )
    claim = pilot.load_json(
        Path(claim_identity["path"]), "development_cell_simulator_claim_unreadable"
    )
    role = claim.get("worker_role")
    pilot.require(
        isinstance(role, str) and role in ALLOWED_SIMULATOR_ROLES,
        "development_cell_simulator_role_invalid",
    )
    return role


@contextmanager
def _configured_for_validation(
    block: DevelopmentBlock, simulator_worker_role: str
) -> Iterator[None]:
    if (
        pilot.BLOCK_ID == block.block_id
        and pilot.CELL_IDS == block.cell_ids
        and pilot.PILOT_CONTRACT_SHA256 == block.contract_sha256
        and pilot.SIMULATOR_QUEUE_ROLE == simulator_worker_role
    ):
        yield
        return
    with _patched_pilot(_configuration_values(block, simulator_worker_role)):
        yield


def validate_passed_development_cell(
    path: Path, *, condition_index: int, study_commit: str,
    block: DevelopmentBlock,
) -> tuple[dict[str, Any], dict[str, Any]]:
    simulator_worker_role = _cell_simulator_role(path)
    with _configured_for_validation(block, simulator_worker_role):
        receipt, identity = _PILOT_VALIDATE_PASSED_CELL(
            Path(path), condition_index=condition_index, study_commit=study_commit
        )
        _ready_identity, ready = _validate_live_ready_descriptor(
            receipt.get("server_ready"), "development_cell_server_ready"
        )
    pilot.require(receipt.get("phase") == "development", "development_cell_phase_changed")
    pilot.require(
        receipt.get("development_contract_sha256") == block.contract_sha256,
        "development_cell_contract_changed",
    )
    pilot.require(
        receipt.get("transport_contract") == NO_REPLAY_TRANSPORT_CONTRACT,
        "development_cell_transport_contract_changed",
    )
    completion = pilot.load_json(
        Path(receipt["adapter_completion"]["path"]),
        "development_adapter_completion_unreadable",
    )
    identity_row = completion.get("identity")
    pilot.require(isinstance(identity_row, Mapping), "development_adapter_identity_missing")
    expected_identity = {
        "cell_id": block.cell_ids[condition_index],
        "stage": "development",
        "layout_pair_id": block.layout_pair_id,
        "model_config": "D1",
        "effective_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
    }
    for key, wanted in expected_identity.items():
        pilot.require(
            identity_row.get(key) == wanted,
            "development_adapter_identity_changed",
            key,
        )
    fixture = receipt.get("development_fixture")
    pilot.require(isinstance(fixture, Mapping), "development_cell_fixture_missing")
    pilot.require(
        fixture.get("candidate_id") == receipt.get("candidate_id")
        and fixture.get("accepted_gate_record_sha256")
        == receipt.get("accepted_gate_record_sha256"),
        "development_cell_fixture_receipt_disagrees",
    )
    for key in (
        "gate_receipt", "pose_manifest", "capture_receipt",
        "raw_capture_receipt", "d1_fixed_observation",
    ):
        _verify_descriptor(fixture.get(key), f"development_cell_{key}")
    execution_prerequisites = _verify_descriptor(
        fixture.get("execution_prerequisites"),
        "development_cell_execution_prerequisites",
    )
    pilot.require(
        fixture.get("execution_prerequisites_sha256")
        == execution_prerequisites["sha256"],
        "development_cell_prerequisite_descriptor_changed",
    )
    pilot.require(
        ready.get("development_prerequisites_sha256")
        == execution_prerequisites["sha256"],
        "development_cell_server_prerequisite_binding_changed",
    )
    verify_execution_prerequisites(
        Path(execution_prerequisites["path"]),
        execution_prerequisites["sha256"],
        block=block,
    )
    source_identity = identity_row.get("source_identity")
    pose_sha = fixture["pose_manifest"]["sha256"]
    pilot.require(
        isinstance(source_identity, str) and source_identity.endswith(f";pose:{pose_sha}"),
        "development_adapter_pose_binding_changed",
    )
    return receipt, identity


def discover_completed_prefix(
    raw_root: Path, *, study_commit: str, block: DevelopmentBlock,
    prerequisites: Mapping[str, Any], simulator_worker_role: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    def validator(
        path: Path, *, condition_index: int, study_commit: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return validate_passed_development_cell(
            path,
            condition_index=condition_index,
            study_commit=study_commit,
            block=block,
        )

    receipt_paths = sorted(
        Path(raw_root).resolve().glob("simulator_attempts/*/cells/*/cell_receipt.json")
    )
    role = (
        _cell_simulator_role(receipt_paths[0])
        if receipt_paths
        else simulator_worker_role
    )
    pilot.require(
        isinstance(role, str) and role in ALLOWED_SIMULATOR_ROLES,
        "development_resume_simulator_role_missing",
    )
    with _configured_for_validation(block, role):
        with _patched_pilot({"validate_passed_cell_receipt": validator}):
            receipts, identities, provenance = _PILOT_DISCOVER_COMPLETED_PREFIX(
                raw_root, study_commit=study_commit
            )
    validate_cells_bind_fixture(receipts, prerequisites=prerequisites, block=block)
    updated = dict(provenance)
    updated.update(
        {
            "schema_version": RESUME_SCHEMA,
            "layout_pair_id": block.layout_pair_id,
            "development_contract_sha256": block.contract_sha256,
        }
    )
    return receipts, identities, updated


@contextmanager
def installed_receipt_metadata(
    block: DevelopmentBlock, *, fixture: Mapping[str, Any] | None = None,
    p00: Mapping[str, Any] | None = None,
    prerequisites: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    def development_immutable_json(
        path: Path, value: Mapping[str, Any], *, publish: bool = False
    ) -> None:
        updated = dict(value)
        schema = updated.get("schema_version")
        if schema in {
            SERVER_RECEIPT_SCHEMA,
            SIMULATOR_RECEIPT_SCHEMA,
            CELL_RECEIPT_SCHEMA,
            pilot.SERVER_READY_SCHEMA,
        }:
            updated.update(
                {
                    "phase": "development",
                    "layout_pair_id": block.layout_pair_id,
                    "environment_seed": block.environment_seed,
                    "effective_model_noise_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
                    "development_contract_sha256": block.contract_sha256,
                }
            )
        if schema in {SERVER_RECEIPT_SCHEMA, SIMULATOR_RECEIPT_SCHEMA} and p00 is not None:
            updated["p00_paired_pilot"] = dict(p00)
        if schema == SERVER_RECEIPT_SCHEMA and prerequisites is not None:
            updated["development_prerequisites"] = dict(prerequisites)
        if schema == pilot.SERVER_READY_SCHEMA and prerequisites is not None:
            updated["development_prerequisites_sha256"] = (
                execution_prerequisites_sha256(block, prerequisites)
            )
        if schema == CELL_RECEIPT_SCHEMA:
            pilot.require(fixture is not None, "development_cell_fixture_metadata_missing")
            updated["development_fixture"] = dict(fixture)
            updated["transport_contract"] = dict(NO_REPLAY_TRANSPORT_CONTRACT)
            updated["claim_boundary"] = (
                "One valid learned-policy D1 behavioral development cell: 450 actual actions, "
                "451 original observations, and 57 official conditional action/future requests."
            )
        if schema == SIMULATOR_RECEIPT_SCHEMA:
            updated["simulator_receipt_filename"] = SIMULATOR_RECEIPT_FILENAME
            updated["claim_boundary"] = (
                "This receipt counts only valid 450-action D1 development cells. The passed "
                "P00 behavioral pilot, six-request generation qualification, recorder-only "
                "qualification, and fixed observation are prerequisites and are not recounted."
            )
        _PILOT_IMMUTABLE_JSON(path, updated, publish=publish)

    with _patched_pilot({"immutable_json": development_immutable_json}):
        yield


def _schedule_identity(block: DevelopmentBlock) -> dict[str, Any]:
    return {
        "path": str(block.schedule_path),
        "sha256": block.schedule_sha256,
        "row": dict(block.schedule_row),
    }


def _resolve_raw_root(args: argparse.Namespace, block: DevelopmentBlock) -> None:
    supplied = block.raw_root if args.raw_root is None else Path(args.raw_root).resolve()
    pilot.require(supplied == block.raw_root.resolve(), "d1_development_raw_root_changed")
    args.raw_root = supplied


def _preflight_science_counts() -> dict[str, int]:
    return {
        "model_server_starts": 0,
        "simulator_process_starts": 0,
        "physical_resets": 0,
        "model_requests": 0,
        "behavioral_actions": 0,
        "behavioral_cells": 0,
    }


def _preflight_prerequisite_summary(
    prerequisites: Mapping[str, Any],
) -> dict[str, Any]:
    p00 = prerequisites["p00_paired_pilot"]
    return {
        "gate_receipt": prerequisites["development_gate_receipt"],
        "pose_manifest": prerequisites["development_pose_manifest"],
        "capture_wrapper_receipt": prerequisites["capture_receipt"],
        "raw_capture_receipt": prerequisites["raw_capture_receipt"],
        "d1_fixed_observation": prerequisites["d1_fixed_observation"],
        "recorder_receipt": prerequisites["recorder_receipt"],
        "d1_qualification_receipt": prerequisites["d1_qualification_receipt"],
        "p00_simulator_receipt": p00["simulator_receipt"],
        "p00_server_receipt": p00["server_receipt"],
    }


def run_prerequisite_preflight(
    args: argparse.Namespace, block: DevelopmentBlock
) -> int:
    """Deeply validate one release without starting model or simulator work."""

    destination = (
        Path(args.job_dir).resolve()
        / "publish"
        / "d1_development_prerequisite_preflight_receipt.json"
    )
    queue_identity: dict[str, Any] | None = None
    expected_capture = expected_development_capture_wrapper(block)
    try:
        queue_identity = validate_preflight_queue_invocation(
            source_root=Path(args.source_root),
            job_dir=Path(args.job_dir),
            study_commit=args.study_commit,
            job_id=args.job_id,
            queue_role=args.queue_role,
            block=block,
            simulator_worker_role=args.simulator_worker_role,
        )
        prerequisites = validate_prerequisites(args, block)
    except BaseException as error:
        failure = {
            "schema_version": PREREQUISITE_PREFLIGHT_SCHEMA,
            "status": "technical_invalid",
            "decision": "no_go",
            "safe_to_release_behavioral_pair": False,
            "study_id": pilot.STUDY_ID,
            "namespace": pilot.NAMESPACE,
            "source_commit": args.study_commit,
            "job_id": args.job_id,
            "queue_role": args.queue_role,
            "simulator_worker_role": args.simulator_worker_role,
            "layout_pair_id": block.layout_pair_id,
            "block_id": block.block_id,
            "environment_seed": block.environment_seed,
            "development_contract_sha256": block.contract_sha256,
            "expected_capture_wrapper": expected_capture,
            "supplied_capture_wrapper": {
                "path": str(Path(args.capture_receipt)),
                "sha256": args.capture_receipt_sha256,
            },
            "queue_descriptor": queue_identity,
            "failure": {
                "error_type": type(error).__name__,
                "reason": getattr(error, "reason", "unexpected_preflight_failure"),
                "detail": str(error),
            },
            "science_counts": _preflight_science_counts(),
            "claim_boundary": (
                "Failed prerequisite validation only; no model server, simulator, "
                "physical reset, model request, behavioral action, or cell was launched."
            ),
            "completed_at_utc": pilot.utc_now(),
        }
        try:
            _PILOT_IMMUTABLE_JSON(destination, failure, publish=True)
        except BaseException:
            # The queue controller still retains the nonzero exit and full logs.
            # Never obscure the original fail-closed prerequisite reason.
            pass
        raise

    prerequisite_sha256 = execution_prerequisites_sha256(block, prerequisites)
    receipt = {
        "schema_version": PREREQUISITE_PREFLIGHT_SCHEMA,
        "status": "passed",
        "decision": "go",
        "safe_to_release_behavioral_pair": True,
        "study_id": pilot.STUDY_ID,
        "namespace": pilot.NAMESPACE,
        "source_commit": args.study_commit,
        "job_id": args.job_id,
        "queue_role": args.queue_role,
        "simulator_worker_role": args.simulator_worker_role,
        "layout_pair_id": block.layout_pair_id,
        "block_id": block.block_id,
        "environment_seed": block.environment_seed,
        "development_contract_sha256": block.contract_sha256,
        "expected_capture_wrapper": expected_capture,
        "queue_descriptor": queue_identity,
        "validated_execution_prerequisites_sha256": prerequisite_sha256,
        "validated_prerequisites": _preflight_prerequisite_summary(prerequisites),
        "science_counts": _preflight_science_counts(),
        "claim_boundary": (
            "Detached prerequisite validation only; no model server, simulator, "
            "physical reset, model request, behavioral action, or cell was launched."
        ),
        "completed_at_utc": pilot.utc_now(),
    }
    _PILOT_IMMUTABLE_JSON(destination, receipt, publish=True)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


def run_server_job(args: argparse.Namespace, block: DevelopmentBlock) -> int:
    _resolve_raw_root(args, block)
    queue_identity = validate_queue_invocation(
        source_root=Path(args.source_root), job_dir=Path(args.job_dir),
        study_commit=args.study_commit, job_id=args.job_id,
        expected_role=pilot.SERVER_QUEUE_ROLE, expected_mode="server-job",
        paired_job_id=args.simulator_job_id, run_id=args.run_id,
        block=block, simulator_worker_role=args.simulator_worker_role,
    )
    prerequisites = validate_prerequisites(args, block)
    p00 = prerequisites["p00_paired_pilot"]

    def queue_verifier(**_kwargs: Any) -> dict[str, Any]:
        return dict(queue_identity)

    with _patched_pilot({"validate_queue_invocation": queue_verifier}):
        with installed_receipt_metadata(
            block, p00=p00, prerequisites=prerequisites
        ):
            return pilot.run_server_job(args)


def run_simulator_job(args: argparse.Namespace, block: DevelopmentBlock) -> int:
    _resolve_raw_root(args, block)
    queue_identity = validate_queue_invocation(
        source_root=Path(args.source_root), job_dir=Path(args.job_dir),
        study_commit=args.study_commit, job_id=args.job_id,
        expected_role=args.simulator_worker_role, expected_mode="simulator-job",
        paired_job_id=args.server_job_id, run_id=args.run_id,
        block=block, simulator_worker_role=args.simulator_worker_role,
    )
    prerequisites = validate_prerequisites(args, block)
    prerequisites_sha256 = execution_prerequisites_sha256(block, prerequisites)

    def queue_verifier(**_kwargs: Any) -> dict[str, Any]:
        return dict(queue_identity)

    def prerequisite_verifier(**_kwargs: Any) -> dict[str, Any]:
        return dict(prerequisites)

    def ready_verifier(*ready_args: Any, **ready_kwargs: Any) -> dict[str, Any]:
        return validate_server_ready_prerequisites(
            *ready_args,
            expected_prerequisites_sha256=prerequisites_sha256,
            **ready_kwargs,
        )

    def cell_builder(**kwargs: Any) -> list[str]:
        prerequisites_identity, observed_prerequisites_sha256 = write_execution_prerequisites(
            Path(kwargs["attempt_root"]) / "execution_prerequisites.json",
            block=block,
            prerequisites=prerequisites,
        )
        return build_cell_command(
            **kwargs,
            capture_receipt=Path(args.capture_receipt),
            capture_receipt_sha256=args.capture_receipt_sha256,
            execution_prerequisites=Path(prerequisites_identity["path"]),
            execution_prerequisites_sha256=observed_prerequisites_sha256,
            candidate_id=args.candidate_id,
            simulator_worker_role=args.simulator_worker_role,
            block=block,
        )

    def cell_validator(
        path: Path, *, condition_index: int, study_commit: str
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt, identity = validate_passed_development_cell(
            path, condition_index=condition_index, study_commit=study_commit, block=block
        )
        validate_cells_bind_fixture([receipt], prerequisites=prerequisites, block=block)
        return receipt, identity

    def prefix_discoverer(
        raw_root: Path, *, study_commit: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        return discover_completed_prefix(
            raw_root, study_commit=study_commit,
            block=block, prerequisites=prerequisites,
            simulator_worker_role=args.simulator_worker_role,
        )

    hooks = {
        "validate_queue_invocation": queue_verifier,
        "validate_schedule": lambda _source_root: _schedule_identity(block),
        "validate_prerequisites": prerequisite_verifier,
        "validate_server_ready": ready_verifier,
        "build_cell_command": cell_builder,
        "validate_passed_cell_receipt": cell_validator,
        "discover_completed_prefix": prefix_discoverer,
    }
    with _patched_pilot(hooks):
        with installed_receipt_metadata(
            block, p00=prerequisites["p00_paired_pilot"]
        ):
            return pilot.run_simulator_job(args)


def run_cell(args: argparse.Namespace, block: DevelopmentBlock) -> int:
    fixture = _fixture_from_args(args, block)
    forecast = Path(args.source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    original_verifier = recorder_job.verify_fixture_release

    def fixture_verifier(**kwargs: Any) -> dict[str, Any]:
        return verify_development_fixture_release(
            **kwargs,
            candidate_id=args.candidate_id,
            block=block,
        )

    def ready_verifier(*ready_args: Any, **ready_kwargs: Any) -> dict[str, Any]:
        return validate_server_ready_prerequisites(
            *ready_args,
            expected_prerequisites_sha256=fixture[
                "execution_prerequisites_sha256"
            ],
            **ready_kwargs,
        )

    recorder_job.verify_fixture_release = fixture_verifier
    try:
        with _patched_pilot({"validate_server_ready": ready_verifier}):
            with installed_receipt_metadata(block, fixture=fixture):
                return pilot.run_cell(args)
    finally:
        recorder_job.verify_fixture_release = original_verifier


def _add_pairing_prerequisites(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pilot-simulator-receipt", type=Path, required=True)
    parser.add_argument("--pilot-simulator-receipt-sha256", required=True)
    parser.add_argument("--pilot-server-receipt", type=Path, required=True)
    parser.add_argument("--pilot-server-receipt-sha256", required=True)
    parser.add_argument("--recorder-receipt-sha256", required=True)


def _add_layout_prerequisites(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--gate-receipt", type=Path, required=True)
    parser.add_argument("--gate-receipt-sha256", required=True)
    parser.add_argument("--pose-manifest", type=Path, required=True)
    parser.add_argument("--pose-manifest-sha256", required=True)
    parser.add_argument("--capture-receipt", type=Path, required=True)
    parser.add_argument("--capture-receipt-sha256", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    server = subparsers.add_parser("server-job", help="run one block-bound official D1 service")
    server.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    server.add_argument("--simulator-worker-role", choices=ALLOWED_SIMULATOR_ROLES, required=True)
    server.add_argument("--source-root", type=Path, required=True)
    server.add_argument("--study-commit", required=True)
    server.add_argument("--job-dir", type=Path, required=True)
    server.add_argument("--job-id", required=True)
    server.add_argument("--simulator-job-id", required=True)
    server.add_argument("--run-id", required=True)
    server.add_argument("--raw-root", type=Path)
    _add_layout_prerequisites(server)
    server.add_argument("--recorder-receipt", type=Path, required=True)
    server.add_argument("--d1-qualification-receipt", type=Path, required=True)
    server.add_argument("--d1-qualification-receipt-sha256", required=True)
    _add_pairing_prerequisites(server)
    server.add_argument("--port", type=int, default=pilot.SERVICE_PORT)
    server.add_argument("--server-ready-timeout", type=float, default=7200.0)
    server.add_argument("--simulator-claim-timeout", type=float, default=7200.0)
    server.add_argument("--server-group-timeout-seconds", type=int, default=100000)
    server.add_argument("--terminate-grace-seconds", type=float, default=10.0)

    simulator = subparsers.add_parser("simulator-job", help="run four ordered cells against one D1 service")
    simulator.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    simulator.add_argument("--simulator-worker-role", choices=ALLOWED_SIMULATOR_ROLES, required=True)
    simulator.add_argument("--source-root", type=Path, required=True)
    simulator.add_argument("--study-commit", required=True)
    simulator.add_argument("--job-dir", type=Path, required=True)
    simulator.add_argument("--job-id", required=True)
    simulator.add_argument("--server-job-id", required=True)
    simulator.add_argument("--run-id", required=True)
    simulator.add_argument("--server-ready-sha256")
    simulator.add_argument("--server-ready-timeout", type=float, default=7200.0)
    simulator.add_argument("--raw-root", type=Path)
    _add_layout_prerequisites(simulator)
    simulator.add_argument("--recorder-receipt", type=Path, required=True)
    simulator.add_argument("--d1-qualification-receipt", type=Path, required=True)
    simulator.add_argument("--d1-qualification-receipt-sha256", required=True)
    _add_pairing_prerequisites(simulator)
    simulator.add_argument("--remote-host", default=pilot.SERVICE_HOST)
    simulator.add_argument("--remote-port", type=int, default=pilot.SERVICE_PORT)
    simulator.add_argument("--cell-timeout", type=float, default=43200.0)

    cell = subparsers.add_parser("cell", help="run one fresh ordered development cell")
    cell.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    cell.add_argument("--simulator-worker-role", choices=ALLOWED_SIMULATOR_ROLES, required=True)
    cell.add_argument("--source-root", type=Path, required=True)
    cell.add_argument("--study-commit", required=True)
    cell.add_argument("--attempt-root", type=Path, required=True)
    _add_layout_prerequisites(cell)
    cell.add_argument("--execution-prerequisites", type=Path, required=True)
    cell.add_argument("--execution-prerequisites-sha256", required=True)
    cell.add_argument("--run-id", required=True)
    cell.add_argument("--server-job-id", required=True)
    cell.add_argument("--server-ready-sha256", required=True)
    cell.add_argument("--simulator-claim-sha256", required=True)
    cell.add_argument("--lease-token", required=True)
    cell.add_argument("--future-root", type=Path, required=True)
    cell.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    cell.add_argument("--command", choices=("left", "right"), required=True)
    cell.add_argument("--condition-index", type=int, required=True)
    cell.add_argument("--remote-host", default=pilot.SERVICE_HOST)
    cell.add_argument("--remote-port", type=int, default=pilot.SERVICE_PORT)
    cell.add_argument("--evidence-timeout", type=float, default=120.0)
    cell.add_argument("--reset-timeout", type=float, default=1200.0)
    cell.add_argument("--inference-timeout", type=float, default=1200.0)

    preflight = subparsers.add_parser(
        "prerequisite-preflight",
        help="deeply validate one released D1 development prerequisite bundle only",
    )
    preflight.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    preflight.add_argument(
        "--simulator-worker-role", choices=ALLOWED_SIMULATOR_ROLES, required=True
    )
    preflight.add_argument("--queue-role", choices=PREFLIGHT_QUEUE_ROLES, required=True)
    preflight.add_argument("--source-root", type=Path, required=True)
    preflight.add_argument("--study-commit", required=True)
    preflight.add_argument("--job-dir", type=Path, required=True)
    preflight.add_argument("--job-id", required=True)
    _add_layout_prerequisites(preflight)
    preflight.add_argument("--recorder-receipt", type=Path, required=True)
    preflight.add_argument("--d1-qualification-receipt", type=Path, required=True)
    preflight.add_argument("--d1-qualification-receipt-sha256", required=True)
    _add_pairing_prerequisites(preflight)
    return parser


def _validate_sha_options(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if name.endswith("_sha256") and value is not None:
            pilot.require(
                isinstance(value, str) and pilot.SHA256_RE.fullmatch(value) is not None,
                "invalid_sha256",
                name,
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    block = load_development_block(Path(args.source_root), args.layout_pair_id)
    if args.mode != "prerequisite-preflight":
        pilot.require(pilot.SAFE_ID_RE.fullmatch(args.run_id) is not None, "invalid_run_id")
    pilot.require(pilot.COMMIT_RE.fullmatch(args.study_commit) is not None, "invalid_study_commit")
    _validate_sha_options(args)
    with configured_pilot(block, args.simulator_worker_role):
        if args.mode == "prerequisite-preflight":
            pilot.require(
                pilot.SAFE_ID_RE.fullmatch(args.job_id) is not None,
                "invalid_job_id",
            )
            return run_prerequisite_preflight(args, block)
        if args.mode == "server-job":
            pilot.require(
                args.port == pilot.SERVICE_PORT
                and args.server_ready_timeout > 0
                and args.simulator_claim_timeout > 0
                and args.server_group_timeout_seconds > 0
                and args.terminate_grace_seconds > 0,
                "invalid_timeout_or_service_endpoint",
            )
            return run_server_job(args, block)
        if args.mode == "simulator-job":
            pilot.require(
                args.remote_host == pilot.SERVICE_HOST
                and args.remote_port == pilot.SERVICE_PORT,
                "d1_service_endpoint_changed",
            )
            pilot.require(
                args.server_ready_timeout > 0 and args.cell_timeout > 0,
                "invalid_timeout",
            )
            return run_simulator_job(args, block)
        if args.mode == "cell":
            pilot.require(
                args.remote_host == pilot.SERVICE_HOST
                and args.remote_port == pilot.SERVICE_PORT,
                "d1_service_endpoint_changed",
            )
            pilot.require(
                args.evidence_timeout > 0
                and args.reset_timeout > 0
                and args.inference_timeout > 0,
                "invalid_timeout",
            )
            return run_cell(args, block)
    raise pilot.D1BehavioralPilotError("unknown_mode")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        print(
            json.dumps(
                {
                    "status": "technical_failure",
                    "error_type": type(error).__name__,
                    "reason": getattr(error, "reason", None),
                    "detail": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
