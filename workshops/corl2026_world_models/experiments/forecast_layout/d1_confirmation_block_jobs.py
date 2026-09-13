#!/usr/bin/env python3
"""Run one released C01--C24 official-conditional D1 confirmation block.

DreamZero's temporal state is global, so each intact four-condition block uses
one exclusive paired two-GPU server job and one separate simulator worker.  All
three executable modes revalidate the post-development scientific freeze and
the immutable 24-layout fixture freeze.  A valid contiguous prefix is reused;
reset and inference requests are never replayed after an ambiguous response.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

import confirmation_fixture_freeze as fixture_freeze
import confirmation_runtime_common as common
import d1_behavioral_pilot_jobs as pilot
import d1_development_block_jobs as development


RUNNER_FILENAME = "d1_confirmation_block_jobs.py"
RAW_PARENT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/confirmation/D1"
)
GLOBAL_D1_SERVER_LOCK = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/.locks/"
    "d1-global-server.lock"
)
SIMULATOR_RECEIPT_FILENAME = "d1_behavioral_confirmation_receipt.json"
SERVER_RECEIPT_SCHEMA = "wmf-d1-behavioral-confirmation-server-job-v1"
SIMULATOR_RECEIPT_SCHEMA = "wmf-d1-behavioral-confirmation-simulator-job-v1"
CELL_RECEIPT_SCHEMA = "wmf-d1-behavioral-confirmation-cell-v1"
RESUME_SCHEMA = "wmf-d1-behavioral-confirmation-resume-v1"
CONFIRMATION_CONTRACT_SCHEMA = "wmf-d1-behavioral-confirmation-contract-v1"
EXECUTION_PREREQUISITES_SCHEMA = "wmf-d1-confirmation-execution-prerequisites-v1"
NO_REPLAY_TRANSPORT_CONTRACT = dict(development.NO_REPLAY_TRANSPORT_CONTRACT)
ALLOWED_SIMULATOR_ROLES = development.ALLOWED_SIMULATOR_ROLES
_CONFIGURATION_ACTIVE = False


def _confirmation_contract(
    *, layout_pair_id: str, environment_seed: int,
    condition_order: Sequence[str], cell_ids: Sequence[str], block_id: str,
) -> dict[str, Any]:
    contract = dict(development._PILOT_DEFAULTS["PILOT_CONTRACT"])
    contract.update(
        {
            "schema_version": CONFIRMATION_CONTRACT_SCHEMA,
            "phase": "confirmation",
            "layout_pair_id": layout_pair_id,
            "block_id": block_id,
            "condition_order": list(condition_order),
            "cell_ids": list(cell_ids),
            "environment_seed": environment_seed,
            "effective_model_noise_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
            "noise_semantics": (
                "fixed at 1140; layouts, cells, and requests are not independent noise draws"
            ),
            "action_path": "official conditional DreamZero-DROID path",
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


def load_confirmation_block(source_root: Path, layout_pair_id: str) -> development.DevelopmentBlock:
    schedule = common.load_confirmation_schedule_block(source_root, layout_pair_id, "D1")
    contract = _confirmation_contract(
        layout_pair_id=schedule.layout_pair_id,
        environment_seed=schedule.environment_seed,
        condition_order=schedule.condition_order,
        cell_ids=schedule.cell_ids,
        block_id=schedule.block_id,
    )
    return development.DevelopmentBlock(
        layout_pair_id=schedule.layout_pair_id,
        environment_seed=schedule.environment_seed,
        condition_order=schedule.condition_order,
        conditions=schedule.conditions,
        cell_ids=schedule.cell_ids,
        block_id=schedule.block_id,
        raw_root=RAW_PARENT / schedule.layout_pair_id,
        schedule_path=schedule.schedule_path,
        schedule_sha256=schedule.schedule_sha256,
        schedule_row=schedule.schedule_row,
        contract=contract,
        contract_sha256=pilot.sha256_bytes(pilot.canonical_bytes(contract)),
    )


def _configuration_values(
    block: development.DevelopmentBlock, simulator_worker_role: str
) -> dict[str, Any]:
    pilot.require(simulator_worker_role in ALLOWED_SIMULATOR_ROLES, "confirmation_simulator_role_unsupported")
    return {
        "PHASE": "confirmation",
        "LAYOUT_PAIR_ID": block.layout_pair_id,
        "BLOCK_ID": block.block_id,
        "CONDITIONS": block.conditions,
        "CELL_IDS": block.cell_ids,
        "ENVIRONMENT_SEED": block.environment_seed,
        "RAW_ROOT": block.raw_root,
        "PILOT_CONTRACT": dict(block.contract),
        "PILOT_CONTRACT_SHA256": block.contract_sha256,
        "RUNNER_FILENAME": RUNNER_FILENAME,
        "BEHAVIORAL_PURPOSE": "d1_behavioral_confirmation",
        "BEHAVIORAL_FINALIZE_PURPOSE": "d1_behavioral_confirmation_finalize",
        "EPISODE_ID_PREFIX": f"d1{block.layout_pair_id.lower()}",
        "POLICY_LABEL": "wmf_d1_behavioral_confirmation",
        "SIMULATOR_RECEIPT_FILENAME": SIMULATOR_RECEIPT_FILENAME,
        "GLOBAL_SERVER_LOCK_PATH": GLOBAL_D1_SERVER_LOCK,
        "SERVER_RECEIPT_SCHEMA": SERVER_RECEIPT_SCHEMA,
        "SIMULATOR_RECEIPT_SCHEMA": SIMULATOR_RECEIPT_SCHEMA,
        "CELL_RECEIPT_SCHEMA": CELL_RECEIPT_SCHEMA,
        "SIMULATOR_QUEUE_ROLE": simulator_worker_role,
    }


@contextmanager
def configured_pilot(
    block: development.DevelopmentBlock, simulator_worker_role: str
) -> Iterator[None]:
    global _CONFIGURATION_ACTIVE
    pilot.require(not _CONFIGURATION_ACTIVE, "confirmation_configuration_overlap")
    _CONFIGURATION_ACTIVE = True
    try:
        with development._patched_pilot(_configuration_values(block, simulator_worker_role)):
            yield
    finally:
        _CONFIGURATION_ACTIVE = False


@contextmanager
def _configured_for_validation(
    block: development.DevelopmentBlock, simulator_worker_role: str
) -> Iterator[None]:
    if (
        pilot.BLOCK_ID == block.block_id
        and pilot.CELL_IDS == block.cell_ids
        and pilot.PILOT_CONTRACT_SHA256 == block.contract_sha256
        and pilot.SIMULATOR_QUEUE_ROLE == simulator_worker_role
    ):
        yield
    else:
        with development._patched_pilot(_configuration_values(block, simulator_worker_role)):
            yield


def _compact_release(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "confirmation_freeze": dict(value["confirmation_freeze"]),
        "cohort_branch": value["cohort_branch"],
        "qualified_model_ids": list(value["qualified_model_ids"]),
        "alignment_contract": dict(value["alignment_contract"]),
    }


def validate_prerequisites(
    args: argparse.Namespace, block: development.DevelopmentBlock,
) -> dict[str, Any]:
    release = common.verify_confirmation_freeze(
        Path(args.confirmation_freeze), args.confirmation_freeze_sha256,
        source_root=Path(args.source_root), model_config="D1",
    )
    fixtures = fixture_freeze.validate_fixture_freeze(
        Path(args.fixture_freeze), args.fixture_freeze_sha256,
        source_root=Path(args.source_root),
        expected_layout_pair_id=block.layout_pair_id,
        expected_study_commit=args.study_commit,
    )
    selected = fixtures["selected_layout"]
    pilot.require(isinstance(selected, Mapping), "confirmation_selected_fixture_missing")
    bindings = {
        "candidate_id": args.candidate_id,
        "gate_receipt": common.verify_exact_file(
            Path(args.gate_receipt), args.gate_receipt_sha256, "confirmation_gate_receipt"
        ),
        "pose_manifest": common.verify_exact_file(
            Path(args.pose_manifest), args.pose_manifest_sha256, "confirmation_pose_manifest"
        ),
        "capture_receipt": common.verify_exact_file(
            Path(args.capture_receipt), args.capture_receipt_sha256, "confirmation_capture_receipt"
        ),
    }
    for key, wanted in bindings.items():
        pilot.require(selected.get(key) == wanted, "confirmation_fixture_cli_binding_changed", key)
    for key, wanted in {
        "gate_receipt": args.gate_receipt_sha256,
        "pose_manifest": args.pose_manifest_sha256,
        "capture_receipt": args.capture_receipt_sha256,
    }.items():
        pilot.require(selected[key]["sha256"] == wanted, "confirmation_fixture_cli_hash_changed", key)

    recorder = development.verify_recorder_qualification(
        Path(args.recorder_receipt), args.recorder_receipt_sha256
    )
    qualification = pilot.validate_d1_qualification(
        Path(args.d1_qualification_receipt), args.d1_qualification_receipt_sha256
    )
    p00 = development.verify_passed_p00_pair(
        simulator_receipt_path=Path(args.pilot_simulator_receipt),
        simulator_receipt_sha256=args.pilot_simulator_receipt_sha256,
        server_receipt_path=Path(args.pilot_server_receipt),
        server_receipt_sha256=args.pilot_server_receipt_sha256,
        recorder_receipt_sha256=recorder["recorder_receipt"]["sha256"],
        d1_qualification_receipt_sha256=qualification["sha256"],
    )
    return {
        "confirmation_release": _compact_release(release),
        "confirmation_fixture_freeze": fixtures["fixture_freeze"],
        "confirmation_fixture_layout_count": fixtures["layout_count"],
        "confirmation_fixture_selection_rule": fixtures["candidate_selection_rule"],
        "selected_fixture": dict(selected),
        "candidate_id": selected["candidate_id"],
        "candidate_payload_sha256": selected["candidate_payload_sha256"],
        "accepted_gate_record_sha256": selected["accepted_gate_record_sha256"],
        "confirmation_gate_receipt": selected["gate_receipt"],
        "confirmation_pose_manifest": selected["pose_manifest"],
        "capture_receipt": selected["capture_receipt"],
        "raw_capture_receipt": selected["raw_capture_receipt"],
        "d1_fixed_observation": selected["d1_fixed_observation"],
        **recorder,
        "d1_qualification_receipt": qualification,
        "p00_paired_pilot": p00,
        "generation_qualification_requests_reused_not_rerun": 6,
    }


def execution_prerequisites_payload(
    block: development.DevelopmentBlock, prerequisites: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_PREREQUISITES_SCHEMA,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "phase": "confirmation",
        "layout_pair_id": block.layout_pair_id,
        "block_id": block.block_id,
        "confirmation_contract_sha256": block.contract_sha256,
        "prerequisites": dict(prerequisites),
    }


def execution_prerequisites_sha256(
    block: development.DevelopmentBlock, prerequisites: Mapping[str, Any]
) -> str:
    return pilot.sha256_bytes(pilot.canonical_bytes(execution_prerequisites_payload(block, prerequisites)))


def verify_execution_prerequisites(
    path: Path, expected_sha256: str, *, block: development.DevelopmentBlock,
    expected_prerequisites: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = pilot.verify_exact_file(path, expected_sha256, "confirmation_execution_prerequisites")
    bundle = pilot.load_json(Path(identity["path"]), "confirmation_execution_prerequisites_unreadable")
    expected = {
        "schema_version": EXECUTION_PREREQUISITES_SCHEMA,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "phase": "confirmation",
        "layout_pair_id": block.layout_pair_id,
        "block_id": block.block_id,
        "confirmation_contract_sha256": block.contract_sha256,
    }
    for key, wanted in expected.items():
        pilot.require(bundle.get(key) == wanted, "confirmation_execution_prerequisites_mismatch", key)
    observed = bundle.get("prerequisites")
    pilot.require(isinstance(observed, Mapping), "confirmation_execution_prerequisites_payload_missing")
    if expected_prerequisites is not None:
        pilot.require(observed == expected_prerequisites, "confirmation_execution_prerequisites_changed")
    pilot.require(pilot.sha256_bytes(pilot.canonical_bytes(bundle)) == expected_sha256, "confirmation_execution_prerequisites_canonical_hash_changed")
    return identity, bundle


def write_execution_prerequisites(
    path: Path, *, block: development.DevelopmentBlock,
    prerequisites: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    payload = execution_prerequisites_payload(block, prerequisites)
    digest = pilot.sha256_bytes(pilot.canonical_bytes(payload))
    if not Path(path).exists():
        development._PILOT_IMMUTABLE_JSON(Path(path), payload)
    identity, _bundle = verify_execution_prerequisites(
        path, digest, block=block, expected_prerequisites=prerequisites
    )
    return identity, digest


def validate_server_ready_prerequisites(
    *args: Any, expected_prerequisites_sha256: str, **kwargs: Any
) -> dict[str, Any]:
    ready_bundle = development._PILOT_VALIDATE_SERVER_READY(*args, **kwargs)
    ready = ready_bundle.get("ready")
    pilot.require(
        isinstance(ready, Mapping)
        and ready.get("confirmation_prerequisites_sha256") == expected_prerequisites_sha256,
        "confirmation_server_simulator_prerequisites_changed",
    )
    return ready_bundle


def validate_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str,
    expected_role: str, expected_mode: str, paired_job_id: str, run_id: str,
    block: development.DevelopmentBlock, simulator_worker_role: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    identity = development._PILOT_VALIDATE_QUEUE_INVOCATION(
        source_root=source_root, job_dir=job_dir, study_commit=study_commit,
        job_id=job_id, expected_role=expected_role,
    )
    descriptor_path = Path(job_dir).resolve() / "descriptor.json"
    descriptor = pilot.load_json(descriptor_path, "queue_descriptor_unreadable")
    argv = descriptor.get("argv")
    pilot.require(isinstance(argv, list) and len(argv) >= 4, "confirmation_queue_argv_invalid")
    pilot.require(argv[0] == "/usr/bin/python3", "confirmation_queue_python_changed")
    pilot.require(
        argv[1] == "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/" + RUNNER_FILENAME,
        "confirmation_queue_runner_changed",
    )
    pilot.require(argv[2] == expected_mode, "confirmation_queue_mode_changed")
    expected = {
        "--layout-pair-id": block.layout_pair_id,
        "--simulator-worker-role": simulator_worker_role,
        "--study-commit": study_commit,
        "--job-id": job_id,
        "--run-id": run_id,
        "--candidate-id": args.candidate_id,
        "--gate-receipt-sha256": args.gate_receipt_sha256,
        "--pose-manifest-sha256": args.pose_manifest_sha256,
        "--capture-receipt-sha256": args.capture_receipt_sha256,
        "--confirmation-freeze-sha256": args.confirmation_freeze_sha256,
        "--fixture-freeze-sha256": args.fixture_freeze_sha256,
        "--recorder-receipt-sha256": args.recorder_receipt_sha256,
        "--d1-qualification-receipt-sha256": args.d1_qualification_receipt_sha256,
        "--pilot-simulator-receipt-sha256": args.pilot_simulator_receipt_sha256,
        "--pilot-server-receipt-sha256": args.pilot_server_receipt_sha256,
        ("--simulator-job-id" if expected_mode == "server-job" else "--server-job-id"): paired_job_id,
    }
    for option, wanted in expected.items():
        pilot.require(common.descriptor_option(argv, option) == wanted, "confirmation_queue_pairing_changed", option)
    observed = pilot.file_identity(descriptor_path)
    pilot.require(all(identity.get(key) == observed[key] for key in ("path", "bytes", "sha256")), "confirmation_queue_descriptor_changed_during_validation")
    return {**observed, "role": expected_role, "job_id": job_id}


def build_cell_command(
    *, source_root: Path, attempt_root: Path, study_commit: str,
    gate_receipt: Path, gate_receipt_sha256: str,
    pose_manifest: Path, pose_manifest_sha256: str,
    capture_receipt: Path, capture_receipt_sha256: str,
    execution_prerequisites: Path, execution_prerequisites_sha256: str,
    candidate_id: str, simulator_worker_role: str, run_id: str,
    server_job_id: str, server_ready_sha256: str, simulator_claim_sha256: str,
    lease_token: str, future_root: Path, condition_index: int,
    block: development.DevelopmentBlock,
) -> list[str]:
    pilot.require(0 <= condition_index < 4, "cell_condition_index_invalid")
    arm, command, _task = block.conditions[condition_index]
    script = Path(source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout" / RUNNER_FILENAME
    return [
        os.path.abspath(os.fspath(pilot.ROBOLAB_PYTHON)), str(script), "cell",
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
        "--layout-arm", arm,
        "--command", command,
        "--condition-index", str(condition_index),
        "--remote-host", pilot.SERVICE_HOST,
        "--remote-port", str(pilot.SERVICE_PORT),
    ]


def _fixture_from_args(
    args: argparse.Namespace, block: development.DevelopmentBlock,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    prerequisites_identity, bundle = verify_execution_prerequisites(
        Path(args.execution_prerequisites), args.execution_prerequisites_sha256,
        block=block,
    )
    pilot.require(
        Path(prerequisites_identity["path"]) == Path(args.attempt_root).resolve() / "execution_prerequisites.json",
        "confirmation_execution_prerequisites_path_changed",
    )
    prerequisites = bundle["prerequisites"]
    # Re-run the actual post-development and 24-layout validators inside every
    # simulator cell; an internally consistent fabricated bundle is not enough.
    release = common.verify_confirmation_freeze(
        Path(prerequisites["confirmation_release"]["confirmation_freeze"]["path"]),
        prerequisites["confirmation_release"]["confirmation_freeze"]["sha256"],
        source_root=Path(args.source_root), model_config="D1",
    )
    fixtures = fixture_freeze.validate_fixture_freeze(
        Path(prerequisites["confirmation_fixture_freeze"]["path"]),
        prerequisites["confirmation_fixture_freeze"]["sha256"],
        source_root=Path(args.source_root), expected_layout_pair_id=block.layout_pair_id,
        expected_study_commit=args.study_commit,
    )
    selected = fixtures["selected_layout"]
    pilot.require(
        prerequisites["confirmation_release"] == _compact_release(release)
        and prerequisites["selected_fixture"] == selected,
        "confirmation_cell_global_prerequisites_changed",
    )
    expected_cli = {
        "candidate_id": args.candidate_id,
        "gate_receipt": common.verify_exact_file(
            Path(args.gate_receipt), args.gate_receipt_sha256, "confirmation_gate_receipt"
        ),
        "pose_manifest": common.verify_exact_file(
            Path(args.pose_manifest), args.pose_manifest_sha256, "confirmation_pose_manifest"
        ),
        "capture_receipt": common.verify_exact_file(
            Path(args.capture_receipt), args.capture_receipt_sha256, "confirmation_capture_receipt"
        ),
    }
    for key, wanted in expected_cli.items():
        pilot.require(selected.get(key) == wanted, "confirmation_cell_fixture_cli_changed", key)
    fixture = {
        "candidate_id": selected["candidate_id"],
        "candidate_payload_sha256": selected["candidate_payload_sha256"],
        "accepted_gate_record_sha256": selected["accepted_gate_record_sha256"],
        "gate_receipt": selected["gate_receipt"],
        "pose_manifest": selected["pose_manifest"],
        "capture_receipt": selected["capture_receipt"],
        "raw_capture_receipt": selected["raw_capture_receipt"],
        "d1_fixed_observation": selected["d1_fixed_observation"],
        "execution_prerequisites": prerequisites_identity,
        "execution_prerequisites_sha256": args.execution_prerequisites_sha256,
    }
    return fixture, prerequisites


def validate_cells_bind_prerequisites(
    cells: Sequence[Mapping[str, Any]], *, prerequisites: Mapping[str, Any],
    block: development.DevelopmentBlock,
) -> None:
    expected_sha = execution_prerequisites_sha256(block, prerequisites)
    selected = prerequisites["selected_fixture"]
    expected = {
        "candidate_id": selected["candidate_id"],
        "candidate_payload_sha256": selected["candidate_payload_sha256"],
        "accepted_gate_record_sha256": selected["accepted_gate_record_sha256"],
        "gate_receipt": selected["gate_receipt"],
        "pose_manifest": selected["pose_manifest"],
        "capture_receipt": selected["capture_receipt"],
        "raw_capture_receipt": selected["raw_capture_receipt"],
        "d1_fixed_observation": selected["d1_fixed_observation"],
    }
    for cell in cells:
        fixture = cell.get("confirmation_fixture")
        pilot.require(isinstance(fixture, Mapping), "confirmation_cell_fixture_missing")
        for key, wanted in expected.items():
            pilot.require(fixture.get(key) == wanted, "confirmation_cell_fixture_binding_changed", key)
        identity = development._verify_descriptor(fixture.get("execution_prerequisites"), "confirmation_cell_execution_prerequisites")
        pilot.require(
            fixture.get("execution_prerequisites_sha256") == expected_sha
            and identity["sha256"] == expected_sha,
            "confirmation_cell_prerequisite_binding_changed",
        )
        verify_execution_prerequisites(
            Path(identity["path"]), expected_sha, block=block,
            expected_prerequisites=prerequisites,
        )


def validate_passed_confirmation_cell(
    path: Path, *, condition_index: int, study_commit: str,
    block: development.DevelopmentBlock,
) -> tuple[dict[str, Any], dict[str, Any]]:
    role = development._cell_simulator_role(path)
    with _configured_for_validation(block, role):
        receipt, identity = development._PILOT_VALIDATE_PASSED_CELL(
            Path(path), condition_index=condition_index, study_commit=study_commit
        )
        pilot.require(
            isinstance(receipt.get("server_context_terminal"), Mapping),
            "confirmation_cell_terminal_context_missing",
        )
        _ready_identity, ready = development._validate_live_ready_descriptor(
            receipt.get("server_ready"), "confirmation_cell_server_ready"
        )
    pilot.require(receipt.get("phase") == "confirmation", "confirmation_cell_phase_changed")
    pilot.require(receipt.get("confirmation_contract_sha256") == block.contract_sha256, "confirmation_cell_contract_changed")
    pilot.require(receipt.get("transport_contract") == NO_REPLAY_TRANSPORT_CONTRACT, "confirmation_cell_transport_contract_changed")
    completion = pilot.load_json(Path(receipt["adapter_completion"]["path"]), "confirmation_adapter_completion_unreadable")
    identity_row = completion.get("identity")
    pilot.require(isinstance(identity_row, Mapping), "confirmation_adapter_identity_missing")
    expected_identity = {
        "cell_id": block.cell_ids[condition_index],
        "stage": "confirmation",
        "layout_pair_id": block.layout_pair_id,
        "model_config": "D1",
        "effective_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
    }
    for key, wanted in expected_identity.items():
        pilot.require(identity_row.get(key) == wanted, "confirmation_adapter_identity_changed", key)
    fixture = receipt.get("confirmation_fixture")
    pilot.require(isinstance(fixture, Mapping), "confirmation_cell_fixture_missing")
    execution = development._verify_descriptor(fixture.get("execution_prerequisites"), "confirmation_cell_execution_prerequisites")
    pilot.require(
        ready.get("confirmation_prerequisites_sha256") == execution["sha256"],
        "confirmation_cell_server_prerequisite_binding_changed",
    )
    verify_execution_prerequisites(Path(execution["path"]), execution["sha256"], block=block)
    source_identity = identity_row.get("source_identity")
    pilot.require(
        isinstance(source_identity, str)
        and source_identity.endswith(f";pose:{fixture['pose_manifest']['sha256']}"),
        "confirmation_adapter_pose_binding_changed",
    )
    return receipt, identity


def validate_failed_confirmation_cell(
    path: Path, *, condition_index: int, block: development.DevelopmentBlock,
) -> dict[str, Any]:
    path = Path(path).resolve()
    attempt_root = path.parents[2]
    cell_id = block.cell_ids[condition_index]
    expected_path = (
        path.parents[2]
        / "cells"
        / f"{condition_index:02d}-{pilot.safe_component(cell_id)}"
        / "technical_failure.json"
    ).resolve()
    pilot.require(path == expected_path, "confirmation_failure_path_changed")
    failure = pilot.load_json(path, "confirmation_failure_receipt_unreadable")
    expected = {
        "schema_version": CELL_RECEIPT_SCHEMA,
        "study_id": pilot.STUDY_ID,
        "phase": "confirmation",
        "block_id": block.block_id,
        "layout_pair_id": block.layout_pair_id,
        "model_config": "D1",
        "cell_id": cell_id,
        "condition_index": condition_index,
    }
    for key, wanted in expected.items():
        pilot.require(failure.get(key) == wanted, "confirmation_failure_identity_changed", key)
    run_id = failure.get("run_id")
    simulator_job_id = failure.get("simulator_job_id")
    server_job_id = failure.get("server_job_id")
    study_commit = failure.get("study_commit")
    for key, value in (
        ("run_id", run_id),
        ("simulator_job_id", simulator_job_id),
        ("server_job_id", server_job_id),
    ):
        pilot.require(
            isinstance(value, str) and pilot.SAFE_ID_RE.fullmatch(value) is not None,
            "confirmation_failure_runtime_id_invalid",
            key,
        )
    pilot.require(
        simulator_job_id == attempt_root.name
        and isinstance(study_commit, str)
        and pilot.COMMIT_RE.fullmatch(study_commit) is not None,
        "confirmation_failure_runtime_identity_changed",
    )
    status = failure.get("status")
    stop_reason = failure.get("recorded_stop_reason")
    actions = failure.get("actions_executed")
    requests = failure.get("request_count")
    pilot.require(status in {"safety_abort", "technical_failure"}, "confirmation_failure_status_invalid")
    pilot.require(
        stop_reason in {"action_cap", "safety_abort", "technical_failure"},
        "confirmation_failure_stop_reason_invalid",
    )
    pilot.require(type(actions) is int and 0 <= actions <= pilot.ACTION_CAP, "confirmation_failure_actions_invalid")
    pilot.require(type(requests) is int and 0 <= requests <= pilot.REQUEST_COUNT, "confirmation_failure_requests_invalid")
    if status == "safety_abort":
        pilot.require(
            stop_reason == "safety_abort"
            and 1 <= actions < pilot.ACTION_CAP
            and requests == math.ceil(actions / pilot.EXECUTED_PREFIX_HORIZON),
            "confirmation_safety_abort_prefix_invalid",
        )
    terminal_descriptor = failure.get("server_context_terminal")
    if terminal_descriptor is None:
        pilot.require(status == "technical_failure", "confirmation_safety_abort_terminal_missing")
        pilot.require(
            failure.get("server_episode_manifest") is None
            and failure.get("server_terminal_reset_scan") is None,
            "confirmation_terminal_sidecars_without_terminal",
        )
        return failure

    ready_identity, ready = development._validate_live_ready_descriptor(
        failure.get("server_ready"), "confirmation_failure_server_ready"
    )
    ready_expected = {
        "schema_version": pilot.SERVER_READY_SCHEMA,
        "status": "ready",
        "study_id": pilot.STUDY_ID,
        "phase": "confirmation",
        "block_id": block.block_id,
        "layout_pair_id": block.layout_pair_id,
        "model_config": "D1",
        "run_id": run_id,
        "server_job_id": server_job_id,
        "paired_simulator_job_id": simulator_job_id,
        "study_commit": study_commit,
        "service_host": pilot.SERVICE_HOST,
        "service_port": pilot.SERVICE_PORT,
        "pilot_contract_sha256": block.contract_sha256,
        "confirmation_contract_sha256": block.contract_sha256,
        "expected_cell_ids": list(block.cell_ids),
        "returned_action_shape": [pilot.RETURNED_ACTION_HORIZON, pilot.ACTION_DIM],
        "executed_prefix_horizon": pilot.EXECUTED_PREFIX_HORIZON,
        "effective_model_noise_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
        "global_state_noninterleaving": True,
    }
    for key, wanted in ready_expected.items():
        pilot.require(
            ready.get(key) == wanted,
            "confirmation_failure_server_ready_changed",
            key,
        )
    paths = pilot.coordination_paths(block.raw_root, str(run_id))
    pilot.require(
        Path(ready_identity["path"]).resolve() == paths["server_ready"].resolve(),
        "confirmation_failure_server_ready_path_changed",
    )
    fixture = failure.get("confirmation_fixture")
    pilot.require(isinstance(fixture, Mapping), "confirmation_failure_fixture_missing")
    execution = development._verify_descriptor(
        fixture.get("execution_prerequisites"),
        "confirmation_failure_execution_prerequisites",
    )
    pilot.require(
        fixture.get("execution_prerequisites_sha256") == execution["sha256"]
        and ready.get("confirmation_prerequisites_sha256") == execution["sha256"],
        "confirmation_failure_ready_prerequisites_changed",
    )
    verify_execution_prerequisites(
        Path(execution["path"]), execution["sha256"], block=block
    )
    claim_identity = pilot._verify_descriptor(
        failure.get("simulator_claim"), "confirmation_failure_simulator_claim"
    )
    pilot.require(
        Path(claim_identity["path"]).resolve() == paths["simulator_claim"].resolve(),
        "confirmation_failure_simulator_claim_path_changed",
    )
    claim = pilot.load_json(Path(claim_identity["path"]), "confirmation_failure_claim_unreadable")
    validated_claim, observed_claim_identity = pilot.validate_simulator_claim(
        Path(claim_identity["path"]),
        run_id=str(ready.get("run_id")),
        simulator_job_id=str(ready.get("paired_simulator_job_id")),
        server_job_id=str(ready.get("server_job_id")),
        server_ready_sha256=ready_identity["sha256"],
        study_commit=str(ready.get("study_commit")),
    )
    pilot.require(
        validated_claim == claim and observed_claim_identity == claim_identity,
        "confirmation_failure_claim_descriptor_changed",
    )
    resume = pilot.load_json(
        attempt_root / "resume.json", "confirmation_failure_resume_unreadable"
    )
    pilot.require(
        resume.get("schema_version") == RESUME_SCHEMA
        and resume.get("block_id") == block.block_id
        and resume.get("layout_pair_id") == block.layout_pair_id
        and attempt_root.name == claim.get("simulator_job_id")
        and claim.get("start_cell_index") == resume.get("start_cell_index")
        and type(claim.get("start_cell_index")) is int
        and claim["start_cell_index"] <= condition_index,
        "confirmation_failure_claim_attempt_changed",
    )
    begin = failure.get("server_begin_receipt")
    pilot.require(isinstance(begin, Mapping) and begin.get("passed") is True, "confirmation_failure_begin_missing")
    episode_id = begin.get("episode_context_id")
    session_id = begin.get("client_session_id")
    pilot.require(
        episode_id == failure.get("episode_id")
        == failure.get("episode_context_id")
        == failure.get("server_context_id")
        and session_id == failure.get("client_session_id")
        and isinstance(episode_id, str)
        and isinstance(session_id, str)
        and episode_id != session_id,
        "confirmation_failure_context_ids_changed",
    )
    layout_arm, command, _task = block.conditions[condition_index]
    begin_control = {
        "episode_id": episode_id,
        "expected_session_id": session_id,
        "purpose": pilot.BEHAVIORAL_PURPOSE,
        "model_config": "D1",
        "study_id": pilot.STUDY_ID,
        "phase": "confirmation",
        "block_id": block.block_id,
        "layout_pair_id": block.layout_pair_id,
        "cell_id": cell_id,
        "condition_index": condition_index,
        "layout_arm": layout_arm,
        "command": command,
        "server_ready_sha256": ready_identity["sha256"],
        "simulator_claim_sha256": claim_identity["sha256"],
        "simulator_lease_token": claim.get("lease_token"),
        "pilot_contract_sha256": pilot.PILOT_CONTRACT_SHA256,
    }
    finalize_control = {
        "finalize_only": True,
        "purpose": pilot.BEHAVIORAL_FINALIZE_PURPOSE,
        "previous_episode_id": episode_id,
        "previous_session_id": session_id,
        "model_config": "D1",
        "study_id": pilot.STUDY_ID,
        "phase": "confirmation",
        "block_id": block.block_id,
        "layout_pair_id": block.layout_pair_id,
        "cell_id": cell_id,
        "condition_index": condition_index,
        "stop_reason": stop_reason,
        "actions_executed": actions,
        "request_count": requests,
        "server_ready_sha256": ready_identity["sha256"],
        "simulator_claim_sha256": claim_identity["sha256"],
        "simulator_lease_token": claim.get("lease_token"),
        "pilot_contract_sha256": pilot.PILOT_CONTRACT_SHA256,
    }
    server_contract = pilot._verify_descriptor(
        ready.get("server_contract"), "confirmation_failure_server_contract"
    )
    future_root_raw = Path(str(ready.get("future_root", "")))
    pilot.require(
        future_root_raw.is_absolute()
        and not future_root_raw.is_symlink(),
        "confirmation_failure_future_root_invalid",
    )
    future_root = future_root_raw.resolve()
    expected_future_root = (
        Path(block.raw_root).resolve()
        / "server_attempts"
        / str(server_job_id)
        / "future"
    ).resolve()
    pilot.require(
        future_root == expected_future_root,
        "confirmation_failure_future_root_changed",
    )
    (
        _terminal,
        observed_terminal_identity,
        _manifest,
        manifest_identity,
        _request_identities,
        terminal_reset_scan,
    ) = pilot.validate_server_terminal_receipt(
        terminal_descriptor,
        future_root=future_root,
        episode_id=episode_id,
        session_id=session_id,
        prompt=pilot.PROMPTS[command],
        expected_begin_control=begin_control,
        expected_finalize_control=finalize_control,
        server_contract_sha256=server_contract["sha256"],
    )
    pilot.require(observed_terminal_identity == terminal_descriptor, "confirmation_terminal_descriptor_changed")
    pilot.require(failure.get("server_episode_manifest") == manifest_identity, "confirmation_terminal_manifest_changed")
    pilot.require(failure.get("server_terminal_reset_scan") == terminal_reset_scan, "confirmation_terminal_reset_scan_changed")
    pilot.require(
        status != "safety_abort" or stop_reason == "safety_abort",
        "confirmation_safety_abort_stop_reason_changed",
    )
    pilot.require(
        not (status == "technical_failure" and stop_reason == "safety_abort"),
        "confirmation_safety_abort_was_incorrectly_demoted",
    )
    return failure


def discover_completed_prefix(
    raw_root: Path, *, study_commit: str, block: development.DevelopmentBlock,
    prerequisites: Mapping[str, Any], simulator_worker_role: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    def validator(path: Path, *, condition_index: int, study_commit: str):
        return validate_passed_confirmation_cell(
            path, condition_index=condition_index, study_commit=study_commit,
            block=block,
        )

    receipt_paths = sorted(Path(raw_root).resolve().glob("simulator_attempts/*/cells/*/cell_receipt.json"))
    role = development._cell_simulator_role(receipt_paths[0]) if receipt_paths else simulator_worker_role
    pilot.require(isinstance(role, str) and role in ALLOWED_SIMULATOR_ROLES, "confirmation_resume_simulator_role_missing")
    with _configured_for_validation(block, role):
        for failure_path in sorted(
            Path(raw_root).resolve().glob(
                "simulator_attempts/*/cells/*/technical_failure.json"
            )
        ):
            raw_failure = pilot.load_json(
                failure_path, "confirmation_resume_failure_unreadable"
            )
            has_terminal_safety_claim = (
                raw_failure.get("recorded_stop_reason") == "safety_abort"
                and raw_failure.get("server_context_terminal") is not None
            )
            if raw_failure.get("status") != "safety_abort" and not has_terminal_safety_claim:
                continue
            index = raw_failure.get("condition_index")
            pilot.require(type(index) is int and 0 <= index < 4, "confirmation_resume_failure_index_invalid")
            censored = validate_failed_confirmation_cell(
                failure_path, condition_index=index, block=block
            )
            validate_cells_bind_prerequisites(
                [censored], prerequisites=prerequisites, block=block
            )
            raise pilot.D1BehavioralPilotError(
                "confirmation_safety_censored_block_is_terminal"
            )
        with development._patched_pilot({"validate_passed_cell_receipt": validator}):
            receipts, identities, provenance = development._PILOT_DISCOVER_COMPLETED_PREFIX(
                raw_root, study_commit=study_commit
            )
    validate_cells_bind_prerequisites(receipts, prerequisites=prerequisites, block=block)
    return receipts, identities, {
        **dict(provenance),
        "schema_version": RESUME_SCHEMA,
        "layout_pair_id": block.layout_pair_id,
        "confirmation_contract_sha256": block.contract_sha256,
        "confirmation_freeze": prerequisites["confirmation_release"]["confirmation_freeze"],
        "fixture_freeze": prerequisites["confirmation_fixture_freeze"],
    }


@contextmanager
def installed_receipt_metadata(
    block: development.DevelopmentBlock, *,
    fixture: Mapping[str, Any] | None = None,
    p00: Mapping[str, Any] | None = None,
    prerequisites: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    def immutable(path: Path, value: Mapping[str, Any], *, publish: bool = False) -> None:
        updated = dict(value)
        schema = updated.get("schema_version")
        if schema in {SERVER_RECEIPT_SCHEMA, SIMULATOR_RECEIPT_SCHEMA, CELL_RECEIPT_SCHEMA, pilot.SERVER_READY_SCHEMA}:
            updated.update(
                {
                    "phase": "confirmation",
                    "layout_pair_id": block.layout_pair_id,
                    "environment_seed": block.environment_seed,
                    "effective_model_noise_seed": pilot.EFFECTIVE_MODEL_NOISE_SEED,
                    "confirmation_contract_sha256": block.contract_sha256,
                }
            )
        if schema in {SERVER_RECEIPT_SCHEMA, SIMULATOR_RECEIPT_SCHEMA} and p00 is not None:
            updated["p00_paired_pilot"] = dict(p00)
        if schema == SERVER_RECEIPT_SCHEMA and prerequisites is not None:
            updated["confirmation_prerequisites"] = dict(prerequisites)
        if schema == pilot.SERVER_READY_SCHEMA and prerequisites is not None:
            updated["confirmation_prerequisites_sha256"] = execution_prerequisites_sha256(block, prerequisites)
        if schema == CELL_RECEIPT_SCHEMA:
            pilot.require(fixture is not None, "confirmation_cell_fixture_metadata_missing")
            updated["confirmation_fixture"] = dict(fixture)
            updated["transport_contract"] = dict(NO_REPLAY_TRANSPORT_CONTRACT)
            updated["claim_boundary"] = (
                "One valid learned-policy D1 behavioral confirmation cell: 450 actual actions, "
                "451 original observations, and 57 official conditional action/future requests."
                if Path(path).name == "cell_receipt.json"
                else "A failed D1 confirmation cell; safety censoring requires a "
                "hash-bound server terminal context receipt."
            )
        if schema == SIMULATOR_RECEIPT_SCHEMA:
            descriptors = updated.get("cell_receipts")
            pilot.require(
                isinstance(descriptors, list),
                "confirmation_cell_receipt_inventory_missing",
            )
            cells = [
                pilot.load_json(
                    Path(development._verify_descriptor(descriptor, "confirmation_context_cell")["path"]),
                    "confirmation_context_cell_unreadable",
                )
                for descriptor in descriptors
            ]
            episode_ids = [
                cell["server_begin_receipt"]["episode_context_id"] for cell in cells
            ]
            session_ids = [
                cell["server_begin_receipt"]["client_session_id"] for cell in cells
            ]
            pilot.require(
                len(episode_ids) == len(set(episode_ids))
                and len(session_ids) == len(set(session_ids))
                and not set(episode_ids).intersection(session_ids),
                "confirmation_context_id_reused_or_aliased",
            )
            updated["simulator_receipt_filename"] = SIMULATOR_RECEIPT_FILENAME
            updated["claim_boundary"] = (
                "This receipt counts only valid 450-action D1 confirmation cells. Pilot, "
                "development, qualification, annotation, and fixture-freeze evidence are "
                "prerequisites and are never recounted."
            )
        development._PILOT_IMMUTABLE_JSON(path, updated, publish=publish)

    with development._patched_pilot({"immutable_json": immutable}):
        yield


def _resolve_raw_root(args: argparse.Namespace, block: development.DevelopmentBlock) -> None:
    argument = block.raw_root if args.raw_root is None else Path(args.raw_root)
    pilot.require(not argument.is_symlink(), "d1_confirmation_raw_root_is_symlink")
    supplied = argument.resolve()
    pilot.require(supplied == block.raw_root.resolve(), "d1_confirmation_raw_root_changed")
    args.raw_root = supplied


def run_server_job(args: argparse.Namespace, block: development.DevelopmentBlock) -> int:
    _resolve_raw_root(args, block)
    queue_identity = validate_queue_invocation(
        source_root=Path(args.source_root), job_dir=Path(args.job_dir),
        study_commit=args.study_commit, job_id=args.job_id,
        expected_role=pilot.SERVER_QUEUE_ROLE, expected_mode="server-job",
        paired_job_id=args.simulator_job_id, run_id=args.run_id,
        block=block, simulator_worker_role=args.simulator_worker_role, args=args,
    )
    prerequisites = validate_prerequisites(args, block)

    def queue_verifier(**_kwargs: Any) -> dict[str, Any]:
        return dict(queue_identity)

    with development._patched_pilot({"validate_queue_invocation": queue_verifier}):
        with installed_receipt_metadata(
            block, p00=prerequisites["p00_paired_pilot"], prerequisites=prerequisites
        ):
            return pilot.run_server_job(args)


def run_simulator_job(args: argparse.Namespace, block: development.DevelopmentBlock) -> int:
    _resolve_raw_root(args, block)
    queue_identity = validate_queue_invocation(
        source_root=Path(args.source_root), job_dir=Path(args.job_dir),
        study_commit=args.study_commit, job_id=args.job_id,
        expected_role=args.simulator_worker_role, expected_mode="simulator-job",
        paired_job_id=args.server_job_id, run_id=args.run_id,
        block=block, simulator_worker_role=args.simulator_worker_role, args=args,
    )
    prerequisites = validate_prerequisites(args, block)
    prerequisites_sha = execution_prerequisites_sha256(block, prerequisites)

    def queue_verifier(**_kwargs: Any) -> dict[str, Any]:
        return dict(queue_identity)

    def prerequisite_verifier(**_kwargs: Any) -> dict[str, Any]:
        return dict(prerequisites)

    def ready_verifier(*ready_args: Any, **ready_kwargs: Any) -> dict[str, Any]:
        return validate_server_ready_prerequisites(
            *ready_args, expected_prerequisites_sha256=prerequisites_sha,
            **ready_kwargs,
        )

    def cell_builder(**kwargs: Any) -> list[str]:
        identity, digest = write_execution_prerequisites(
            Path(kwargs["attempt_root"]) / "execution_prerequisites.json",
            block=block, prerequisites=prerequisites,
        )
        return build_cell_command(
            **kwargs,
            capture_receipt=Path(args.capture_receipt),
            capture_receipt_sha256=args.capture_receipt_sha256,
            execution_prerequisites=Path(identity["path"]),
            execution_prerequisites_sha256=digest,
            candidate_id=args.candidate_id,
            simulator_worker_role=args.simulator_worker_role,
            block=block,
        )

    def cell_validator(path: Path, *, condition_index: int, study_commit: str):
        receipt, identity = validate_passed_confirmation_cell(
            path, condition_index=condition_index, study_commit=study_commit,
            block=block,
        )
        validate_cells_bind_prerequisites([receipt], prerequisites=prerequisites, block=block)
        return receipt, identity

    def prefix_discoverer(raw_root: Path, *, study_commit: str):
        return discover_completed_prefix(
            raw_root, study_commit=study_commit, block=block,
            prerequisites=prerequisites,
            simulator_worker_role=args.simulator_worker_role,
        )

    def failure_validator(path: Path, *, condition_index: int):
        failure = validate_failed_confirmation_cell(
            path, condition_index=condition_index, block=block
        )
        validate_cells_bind_prerequisites(
            [failure], prerequisites=prerequisites, block=block
        )
        return failure

    hooks = {
        "validate_queue_invocation": queue_verifier,
        "validate_schedule": lambda _source_root: {
            "path": str(block.schedule_path), "sha256": block.schedule_sha256,
            "row": dict(block.schedule_row),
        },
        "validate_prerequisites": prerequisite_verifier,
        "validate_server_ready": ready_verifier,
        "build_cell_command": cell_builder,
        "validate_passed_cell_receipt": cell_validator,
        "validate_failure_cell_receipt": failure_validator,
        "discover_completed_prefix": prefix_discoverer,
    }
    with development._patched_pilot(hooks):
        with installed_receipt_metadata(block, p00=prerequisites["p00_paired_pilot"]):
            return pilot.run_simulator_job(args)


def run_cell(args: argparse.Namespace, block: development.DevelopmentBlock) -> int:
    fixture, _prerequisites = _fixture_from_args(args, block)
    forecast = Path(args.source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    original_verifier = recorder_job.verify_fixture_release

    def fixture_verifier(**_kwargs: Any) -> dict[str, Any]:
        return {
            "gate_receipt": fixture["gate_receipt"],
            "pose_manifest": fixture["pose_manifest"],
            "gate_ledger": _prerequisites["selected_fixture"]["gate_ledger"],
            "gate_attempt_receipt": _prerequisites["selected_fixture"]["gate_attempt_receipt"],
            "candidate_id": fixture["candidate_id"],
            "candidate_payload_sha256": fixture["candidate_payload_sha256"],
            "accepted_gate_record_sha256": fixture["accepted_gate_record_sha256"],
        }

    def ready_verifier(*ready_args: Any, **ready_kwargs: Any) -> dict[str, Any]:
        return validate_server_ready_prerequisites(
            *ready_args,
            expected_prerequisites_sha256=fixture["execution_prerequisites_sha256"],
            **ready_kwargs,
        )

    recorder_job.verify_fixture_release = fixture_verifier
    try:
        with development._patched_pilot({"validate_server_ready": ready_verifier}):
            with installed_receipt_metadata(block, fixture=fixture):
                return pilot.run_cell(args)
    finally:
        recorder_job.verify_fixture_release = original_verifier


def _add_global_prerequisites(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--confirmation-freeze-sha256", required=True)
    parser.add_argument("--fixture-freeze", type=Path, required=True)
    parser.add_argument("--fixture-freeze-sha256", required=True)
    parser.add_argument("--recorder-receipt", type=Path, required=True)
    parser.add_argument("--recorder-receipt-sha256", required=True)
    parser.add_argument("--d1-qualification-receipt", type=Path, required=True)
    parser.add_argument("--d1-qualification-receipt-sha256", required=True)
    parser.add_argument("--pilot-simulator-receipt", type=Path, required=True)
    parser.add_argument("--pilot-simulator-receipt-sha256", required=True)
    parser.add_argument("--pilot-server-receipt", type=Path, required=True)
    parser.add_argument("--pilot-server-receipt-sha256", required=True)


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
    server = subparsers.add_parser("server-job")
    server.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS, required=True)
    server.add_argument("--simulator-worker-role", choices=ALLOWED_SIMULATOR_ROLES, required=True)
    server.add_argument("--source-root", type=Path, required=True)
    server.add_argument("--study-commit", required=True)
    server.add_argument("--job-dir", type=Path, required=True)
    server.add_argument("--job-id", required=True)
    server.add_argument("--simulator-job-id", required=True)
    server.add_argument("--run-id", required=True)
    server.add_argument("--raw-root", type=Path)
    _add_layout_prerequisites(server)
    _add_global_prerequisites(server)
    server.add_argument("--port", type=int, default=pilot.SERVICE_PORT)
    server.add_argument("--server-ready-timeout", type=float, default=7200.0)
    server.add_argument("--simulator-claim-timeout", type=float, default=7200.0)
    server.add_argument("--server-group-timeout-seconds", type=int, default=100000)
    server.add_argument("--terminate-grace-seconds", type=float, default=10.0)

    simulator = subparsers.add_parser("simulator-job")
    simulator.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS, required=True)
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
    _add_global_prerequisites(simulator)
    simulator.add_argument("--remote-host", default=pilot.SERVICE_HOST)
    simulator.add_argument("--remote-port", type=int, default=pilot.SERVICE_PORT)
    simulator.add_argument("--cell-timeout", type=float, default=43200.0)

    cell = subparsers.add_parser("cell")
    cell.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS, required=True)
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
    return parser


def _validate_sha_options(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if name.endswith("_sha256") and value is not None:
            pilot.require(isinstance(value, str) and pilot.SHA256_RE.fullmatch(value) is not None, "invalid_sha256", name)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    block = load_confirmation_block(Path(args.source_root), args.layout_pair_id)
    pilot.require(pilot.SAFE_ID_RE.fullmatch(args.run_id) is not None, "invalid_run_id")
    pilot.require(pilot.COMMIT_RE.fullmatch(args.study_commit) is not None, "invalid_study_commit")
    _validate_sha_options(args)
    with configured_pilot(block, args.simulator_worker_role):
        if args.mode == "server-job":
            pilot.require(
                args.port == pilot.SERVICE_PORT and args.server_ready_timeout > 0
                and args.simulator_claim_timeout > 0
                and args.server_group_timeout_seconds > 0
                and args.terminate_grace_seconds > 0,
                "invalid_timeout_or_service_endpoint",
            )
            return run_server_job(args, block)
        if args.mode == "simulator-job":
            pilot.require(args.remote_host == pilot.SERVICE_HOST and args.remote_port == pilot.SERVICE_PORT, "d1_service_endpoint_changed")
            pilot.require(args.server_ready_timeout > 0 and args.cell_timeout > 0, "invalid_timeout")
            return run_simulator_job(args, block)
        if args.mode == "cell":
            pilot.require(args.remote_host == pilot.SERVICE_HOST and args.remote_port == pilot.SERVICE_PORT, "d1_service_endpoint_changed")
            pilot.require(args.evidence_timeout > 0 and args.reset_timeout > 0 and args.inference_timeout > 0, "invalid_timeout")
            pilot.require(0 <= args.condition_index < 4, "invalid_condition_index")
            arm, command, _task = block.conditions[args.condition_index]
            pilot.require((args.layout_arm, args.command) == (arm, command), "confirmation_cell_condition_changed")
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
