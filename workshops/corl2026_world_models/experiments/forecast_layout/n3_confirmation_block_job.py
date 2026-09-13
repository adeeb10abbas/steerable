#!/usr/bin/env python3
"""Run one released C01--C24 N3 confirmation block.

Each invocation executes one intact, frozen four-condition layout block with a
single isolated official Nano context.  The prepared schedule is not a release:
queue, server and cell modes all require the exact post-development scientific
freeze and the exact 24-layout model-blind fixture freeze.  Completed cells are
reused only as one deeply validated contiguous prefix; inference requests are
never replayed after an ambiguous transport failure.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

import confirmation_fixture_freeze as fixture_freeze
import confirmation_runtime_common as common
import n3_behavioral_pilot_job as pilot
import n3_development_block_job as development


RUNNER_FILENAME = "n3_confirmation_block_job.py"
RAW_PARENT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/confirmation/N3"
)
AGGREGATE_FILENAME = "n3_behavioral_confirmation_receipt.json"
QUEUE_RECEIPT_SCHEMA = "wmf-n3-behavioral-confirmation-job-v1"
CELL_RECEIPT_SCHEMA = "wmf-n3-behavioral-confirmation-cell-v1"
RESUME_SCHEMA = "wmf-n3-behavioral-confirmation-resume-v1"
NO_REPLAY_TRANSPORT_CONTRACT = dict(development.NO_REPLAY_TRANSPORT_CONTRACT)
_CONFIGURATION_ACTIVE = False


def load_confirmation_block(source_root: Path, layout_pair_id: str) -> development.DevelopmentBlock:
    schedule = common.load_confirmation_schedule_block(source_root, layout_pair_id, "N3")
    return development.DevelopmentBlock(
        layout_pair_id=schedule.layout_pair_id,
        effective_seed=schedule.effective_model_seed,
        condition_order=schedule.condition_order,
        conditions=schedule.conditions,
        cell_ids=schedule.cell_ids,
        block_id=schedule.block_id,
        raw_root=RAW_PARENT / schedule.layout_pair_id,
        schedule_path=schedule.schedule_path,
        schedule_sha256=schedule.schedule_sha256,
        schedule_row=schedule.schedule_row,
    )


@contextmanager
def configured_pilot(block: development.DevelopmentBlock) -> Iterator[None]:
    global _CONFIGURATION_ACTIVE
    pilot.require(not _CONFIGURATION_ACTIVE, "confirmation_configuration_overlap")
    _CONFIGURATION_ACTIVE = True
    replacements = {
        "PHASE": "confirmation",
        "LAYOUT_PAIR_ID": block.layout_pair_id,
        "BLOCK_ID": block.block_id,
        "EFFECTIVE_SEED": block.effective_seed,
        "CONDITIONS": block.conditions,
        "CELL_IDS": block.cell_ids,
        "RAW_ROOT": block.raw_root,
        "QUEUE_RECEIPT_SCHEMA": QUEUE_RECEIPT_SCHEMA,
        "CELL_RECEIPT_SCHEMA": CELL_RECEIPT_SCHEMA,
    }
    originals = {name: getattr(pilot, name) for name in replacements}
    for name, value in replacements.items():
        setattr(pilot, name, value)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(pilot, name, value)
        _CONFIGURATION_ACTIVE = False


def _compact_release(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "confirmation_freeze": dict(value["confirmation_freeze"]),
        "cohort_branch": value["cohort_branch"],
        "qualified_model_ids": list(value["qualified_model_ids"]),
        "alignment_contract": dict(value["alignment_contract"]),
    }


def validate_prerequisites(
    args: argparse.Namespace,
    block: development.DevelopmentBlock,
    *,
    include_pilot: bool,
) -> dict[str, Any]:
    release = common.verify_confirmation_freeze(
        Path(args.confirmation_freeze),
        args.confirmation_freeze_sha256,
        source_root=Path(args.source_root),
        model_config="N3",
    )
    fixtures = fixture_freeze.validate_fixture_freeze(
        Path(args.fixture_freeze),
        args.fixture_freeze_sha256,
        source_root=Path(args.source_root),
        expected_layout_pair_id=block.layout_pair_id,
        expected_study_commit=args.study_commit,
    )
    selected = fixtures["selected_layout"]
    pilot.require(isinstance(selected, Mapping), "confirmation_selected_fixture_missing")
    expected_fixture = {
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
    expected_hashes = {
        "gate_receipt": args.gate_receipt_sha256,
        "pose_manifest": args.pose_manifest_sha256,
        "capture_receipt": args.capture_receipt_sha256,
    }
    for key, wanted in expected_fixture.items():
        pilot.require(selected.get(key) == wanted, "confirmation_fixture_cli_binding_changed", key)
    for key, wanted in expected_hashes.items():
        pilot.require(selected[key]["sha256"] == wanted, "confirmation_fixture_cli_hash_changed", key)
    result: dict[str, Any] = {
        "confirmation_release": _compact_release(release),
        "confirmation_fixture_freeze": fixtures["fixture_freeze"],
        "confirmation_fixture_layout_count": fixtures["layout_count"],
        "confirmation_fixture_selection_rule": fixtures["candidate_selection_rule"],
        "selected_fixture": dict(selected),
        "candidate_id": selected["candidate_id"],
        "candidate_payload_sha256": selected["candidate_payload_sha256"],
        "accepted_gate_record_sha256": selected["accepted_gate_record_sha256"],
        "seed_audit": development.verify_seed_audit(Path(args.source_root), block),
    }
    if include_pilot:
        result["p00_pilot"] = development.verify_passed_p00_pilot(
            Path(args.pilot_receipt), args.pilot_receipt_sha256
        )
        result["generation_qualification_requests_reused_not_rerun"] = 6
    return result


def validate_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str,
    block: development.DevelopmentBlock, args: argparse.Namespace,
) -> dict[str, Any]:
    identity = pilot.validate_queue_invocation(
        source_root=source_root,
        job_dir=job_dir,
        study_commit=study_commit,
        job_id=job_id,
    )
    descriptor = pilot.load_json(Path(job_dir).resolve() / "descriptor.json", "queue_descriptor_unreadable")
    argv = descriptor.get("argv")
    pilot.require(isinstance(argv, list) and len(argv) >= 4, "confirmation_queue_argv_invalid")
    pilot.require(argv[0] == "/usr/bin/python3", "confirmation_queue_python_changed")
    pilot.require(
        argv[1]
        == "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        + RUNNER_FILENAME,
        "confirmation_queue_runner_changed",
    )
    pilot.require(argv[2] == "queue", "confirmation_queue_mode_changed")
    expected = {
        "--layout-pair-id": block.layout_pair_id,
        "--study-commit": study_commit,
        "--job-id": job_id,
        "--candidate-id": args.candidate_id,
        "--gate-receipt-sha256": args.gate_receipt_sha256,
        "--pose-manifest-sha256": args.pose_manifest_sha256,
        "--capture-receipt-sha256": args.capture_receipt_sha256,
        "--confirmation-freeze-sha256": args.confirmation_freeze_sha256,
        "--fixture-freeze-sha256": args.fixture_freeze_sha256,
        "--pilot-receipt-sha256": args.pilot_receipt_sha256,
    }
    for option, wanted in expected.items():
        pilot.require(
            common.descriptor_option(argv, option) == wanted,
            "confirmation_queue_option_changed",
            option,
        )
    observed = pilot.file_identity(Path(job_dir).resolve() / "descriptor.json")
    pilot.require(
        all(identity.get(key) == observed[key] for key in ("path", "bytes", "sha256")),
        "confirmation_queue_descriptor_changed_during_validation",
    )
    return {**observed, "role": pilot.QUEUE_ROLE, "job_id": job_id}


def build_server_command(
    *, source_root: Path, attempt_root: Path, port: int, study_commit: str,
    start_cell_index: int, block: development.DevelopmentBlock,
    candidate_id: str, gate_receipt: Path, gate_receipt_sha256: str,
    pose_manifest: Path, pose_manifest_sha256: str,
    capture_receipt: Path, capture_receipt_sha256: str,
    confirmation_freeze: Path, confirmation_freeze_sha256: str,
    fixture_freeze_path: Path, fixture_freeze_sha256: str,
) -> list[str]:
    pilot.require(0 <= start_cell_index < 4, "invalid_start_cell_index")
    script = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout"
        / RUNNER_FILENAME
    )
    return [
        os.path.abspath(os.fspath(pilot.COSMOS_PYTHON)),
        str(script), "server",
        "--layout-pair-id", block.layout_pair_id,
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--port", str(port),
        "--start-cell-index", str(start_cell_index),
        "--candidate-id", candidate_id,
        "--gate-receipt", str(Path(gate_receipt).resolve()),
        "--gate-receipt-sha256", gate_receipt_sha256,
        "--pose-manifest", str(Path(pose_manifest).resolve()),
        "--pose-manifest-sha256", pose_manifest_sha256,
        "--capture-receipt", str(Path(capture_receipt).resolve()),
        "--capture-receipt-sha256", capture_receipt_sha256,
        "--confirmation-freeze", str(Path(confirmation_freeze).resolve()),
        "--confirmation-freeze-sha256", confirmation_freeze_sha256,
        "--fixture-freeze", str(Path(fixture_freeze_path).resolve()),
        "--fixture-freeze-sha256", fixture_freeze_sha256,
    ]


def build_cell_command(
    *, source_root: Path, attempt_root: Path, study_commit: str,
    gate_receipt: Path, gate_receipt_sha256: str, pose_manifest: Path,
    pose_manifest_sha256: str, capture_receipt: Path, capture_receipt_sha256: str,
    candidate_id: str, confirmation_freeze: Path, confirmation_freeze_sha256: str,
    fixture_freeze_path: Path, fixture_freeze_sha256: str,
    port: int, condition_index: int, block: development.DevelopmentBlock,
) -> list[str]:
    pilot.require(0 <= condition_index < 4, "cell_condition_index_invalid")
    arm, command, _task = block.conditions[condition_index]
    script = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout"
        / RUNNER_FILENAME
    )
    return [
        os.path.abspath(os.fspath(pilot.ROBOLAB_PYTHON)),
        str(script), "cell",
        "--layout-pair-id", block.layout_pair_id,
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
        "--confirmation-freeze", str(Path(confirmation_freeze).resolve()),
        "--confirmation-freeze-sha256", confirmation_freeze_sha256,
        "--fixture-freeze", str(Path(fixture_freeze_path).resolve()),
        "--fixture-freeze-sha256", fixture_freeze_sha256,
        "--layout-arm", arm,
        "--command", command,
        "--condition-index", str(condition_index),
        "--remote-host", "127.0.0.1",
        "--remote-port", str(port),
    ]


def _expected_prerequisite_binding(prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "confirmation_release": dict(prerequisites["confirmation_release"]),
        "confirmation_fixture_freeze": dict(prerequisites["confirmation_fixture_freeze"]),
        "selected_fixture": dict(prerequisites["selected_fixture"]),
    }


def validate_cells_bind_prerequisites(
    receipts: Sequence[Mapping[str, Any]], *, prerequisites: Mapping[str, Any]
) -> None:
    expected = _expected_prerequisite_binding(prerequisites)
    for receipt in receipts:
        pilot.require(
            receipt.get("confirmation_prerequisites") == expected,
            "confirmation_cell_prerequisite_binding_changed",
            str(receipt.get("cell_id")),
        )


def validate_passed_confirmation_cell(
    path: Path, *, condition_index: int, study_commit: str,
    block: development.DevelopmentBlock,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, identity = development._validate_cell_receipt(
        Path(path).resolve(),
        phase="confirmation",
        layout_pair_id=block.layout_pair_id,
        block_id=block.block_id,
        effective_seed=block.effective_seed,
        conditions=block.conditions,
        cell_ids=block.cell_ids,
        schema_version=CELL_RECEIPT_SCHEMA,
        condition_index=condition_index,
    )
    pilot.require(receipt.get("phase") == "confirmation", "confirmation_cell_phase_changed")
    pilot.require(
        receipt.get("source_pins", {}).get("study_commit") == study_commit,
        "confirmation_cell_study_commit_changed",
    )
    pilot.require(
        receipt.get("transport_contract") == NO_REPLAY_TRANSPORT_CONTRACT,
        "confirmation_cell_transport_contract_changed",
    )
    prerequisites = receipt.get("confirmation_prerequisites")
    pilot.require(isinstance(prerequisites, Mapping), "confirmation_cell_prerequisites_missing")
    return receipt, identity


def discover_completed_prefix(
    raw_root: Path, *, block: development.DevelopmentBlock,
    prerequisites: Mapping[str, Any], study_commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_root = Path(raw_root).resolve()
    found: dict[int, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    for path in sorted(raw_root.glob("*/cells/*/cell_receipt.json")):
        attempt_root = path.parents[2].resolve()
        pilot.require(not path.parents[2].is_symlink(), "resume_attempt_is_symlink", str(attempt_root))
        value = pilot.load_json(path, "resume_cell_receipt_unreadable")
        index = value.get("condition_index")
        pilot.require(type(index) is int and 0 <= index < 4, "resume_cell_index_invalid", str(path))
        receipt, identity = validate_passed_confirmation_cell(
            path, condition_index=index, study_commit=study_commit, block=block
        )
        validate_cells_bind_prerequisites([receipt], prerequisites=prerequisites)
        pilot.require(index not in found, "resume_duplicate_passed_cell", block.cell_ids[index])
        pilot.require(not (path.parent / "technical_failure.json").exists(), "resume_contradictory_cell_receipts", block.cell_ids[index])
        found[index] = (receipt, identity, attempt_root)

    valid_paths = {
        Path(identity["path"]).resolve(): index
        for index, (_receipt, identity, _attempt) in found.items()
    }
    for aggregate_path in sorted(raw_root.glob(f"*/publish/{AGGREGATE_FILENAME}")):
        aggregate = pilot.load_json(aggregate_path, "resume_aggregate_receipt_unreadable")
        pilot.require(aggregate.get("schema_version") == QUEUE_RECEIPT_SCHEMA, "resume_aggregate_schema_changed")
        pilot.require(aggregate.get("phase") == "confirmation", "resume_aggregate_phase_changed")
        pilot.require(aggregate.get("block_id") == block.block_id, "resume_aggregate_block_mismatch")
        descriptors = aggregate.get("cell_receipts")
        pilot.require(isinstance(descriptors, list), "resume_aggregate_descriptors_invalid")
        paths: list[Path] = []
        for offset, descriptor in enumerate(descriptors):
            observed = development._verify_descriptor(descriptor, f"resume_aggregate_cell_{offset}")
            paths.append(Path(observed["path"]).resolve())
        pilot.require(all(path in valid_paths for path in paths), "resume_aggregate_cell_inventory_contradiction")
        pilot.require([valid_paths[path] for path in paths] == list(range(len(paths))), "resume_aggregate_cell_order_contradiction")
        if aggregate.get("status") == "passed":
            pilot.require(len(paths) == 4, "resume_passed_aggregate_incomplete")

    indices = sorted(found)
    pilot.require(indices == list(range(len(indices))), "resume_passed_cells_not_contiguous")
    receipts = [found[index][0] for index in indices]
    identities = [found[index][1] for index in indices]
    contexts = [receipt["server_begin_receipt"]["server_context_id"] for receipt in receipts]
    pilot.require(len(contexts) == len(set(contexts)), "resume_server_context_id_reused")
    return receipts, identities, {
        "schema_version": RESUME_SCHEMA,
        "status": "passed",
        "block_id": block.block_id,
        "layout_pair_id": block.layout_pair_id,
        "completed_prefix_cell_ids": list(block.cell_ids[: len(indices)]),
        "start_cell_index": len(indices),
        "prior_cell_receipts": identities,
        "prior_attempt_roots": sorted({str(found[index][2]) for index in indices}),
        "prior_study_commits": sorted({receipt["source_pins"]["study_commit"] for receipt in receipts}),
        "confirmation_freeze": prerequisites["confirmation_release"]["confirmation_freeze"],
        "fixture_freeze": prerequisites["confirmation_fixture_freeze"],
        "scan_completed_at_utc": pilot.utc_now(),
    }


def _receipt_counts(
    *, block: development.DevelopmentBlock, start_cell_index: int, launched: int,
    completed: Sequence[Mapping[str, Any]], attempt_root: Path,
) -> dict[str, Any]:
    completed_ids = {str(row.get("cell_id")) for row in completed}
    invalid = censored = 0
    actions = sum(int(row.get("actions_executed", 0)) for row in completed)
    requests = sum(int(row.get("behavioral_model_request_count", 0)) for row in completed)
    for index in range(start_cell_index, start_cell_index + launched):
        cell_id = block.cell_ids[index]
        if cell_id in completed_ids:
            continue
        failure_path = (
            Path(attempt_root) / "cells"
            / f"{index:02d}-{pilot.safe_cell_component(cell_id)}"
            / "technical_failure.json"
        )
        if failure_path.is_file():
            failure = pilot.load_json(failure_path)
            actions += int(failure.get("actions_executed", 0))
            requests += int(failure.get("request_count", 0))
            if failure.get("status") == "safety_abort":
                censored += 1
            else:
                invalid += 1
        else:
            invalid += 1
    return {
        "planned_behavioral_cells": 4,
        "launched_behavioral_cells": start_cell_index + launched,
        "resumed_valid_behavioral_cells": start_cell_index,
        "newly_launched_behavioral_cells": launched,
        "completed_valid_behavioral_cells": len(completed),
        "technically_invalid_behavioral_cells": invalid,
        "right_censored_behavioral_cells": censored,
        "unrun_behavioral_cells": 4 - start_cell_index - launched,
        "actual_behavioral_actions": actions,
        "actual_behavioral_model_requests": requests,
        "new_generation_qualification_requests": 0,
        "reused_prerequisite_generation_qualification_requests": 6,
        "recorder_only_episodes_counted_as_behavioral": 0,
    }


def resolve_attempt_root(raw_root: Path, job_id: str) -> Path:
    pilot.require(pilot.SAFE_ID_RE.fullmatch(job_id) is not None, "invalid_job_id")
    root = Path(raw_root).resolve()
    attempt = (root / job_id).resolve()
    pilot.require(attempt.parent == root, "n3_confirmation_attempt_root_escape")
    return attempt


def _verify_ready_binding(path: Path, prerequisites: Mapping[str, Any]) -> dict[str, Any]:
    ready = pilot.load_json(path, "confirmation_server_ready_unreadable")
    pilot.require(
        ready.get("schema_version") == pilot.SERVER_READY_SCHEMA
        and ready.get("status") == "ready"
        and ready.get("block_id") == pilot.BLOCK_ID,
        "confirmation_server_ready_invalid",
    )
    pilot.require(
        ready.get("confirmation_release") == prerequisites["confirmation_release"]
        and ready.get("confirmation_fixture_freeze") == prerequisites["confirmation_fixture_freeze"],
        "confirmation_server_ready_freeze_binding_changed",
    )
    return ready


def run_queue(args: argparse.Namespace, block: development.DevelopmentBlock) -> int:
    source_root = Path(args.source_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    raw_root_argument = block.raw_root if args.raw_root is None else Path(args.raw_root)
    pilot.require(not raw_root_argument.is_symlink(), "n3_confirmation_raw_root_is_symlink")
    raw_root = raw_root_argument.resolve()
    pilot.require(raw_root == block.raw_root.resolve(), "n3_confirmation_raw_root_changed")
    queue_identity = validate_queue_invocation(
        source_root=source_root, job_dir=job_dir, study_commit=args.study_commit,
        job_id=args.job_id, block=block, args=args,
    )
    prerequisites = validate_prerequisites(args, block, include_pilot=True)
    publish_path = job_dir / "publish" / AGGREGATE_FILENAME
    attempt_root = resolve_attempt_root(raw_root, args.job_id)
    completed: list[dict[str, Any]] = []
    resumed_identities: list[dict[str, Any]] = []
    resume: dict[str, Any] = {}
    start_cell_index = 0
    new_identities: list[dict[str, Any]] = []
    launched = 0
    server_process: subprocess.Popen[bytes] | None = None
    server_stdout = server_stderr = None
    topology = server_ready = server_exit = None
    failure: BaseException | None = None

    raw_root.mkdir(parents=True, exist_ok=True)
    lock_path = raw_root / ".locks" / f"{block.block_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise pilot.N3BehavioralPilotError("n3_confirmation_block_lock_is_held") from error
        try:
            # Reconciliation is protected by the same block lock as launch.  A
            # late queue claimant therefore cannot scan an empty prefix, wait
            # behind another attempt, and then rerun cells that just finished.
            completed, resumed_identities, resume = discover_completed_prefix(
                raw_root, block=block, prerequisites=prerequisites,
                study_commit=args.study_commit,
            )
            start_cell_index = len(completed)
            pilot.require(start_cell_index < 4, "valid_n3_confirmation_block_already_exists")
            pilot.require(not attempt_root.exists(), "n3_confirmation_attempt_directory_already_exists")
            attempt_root.mkdir(parents=True)
            (attempt_root / "publish").mkdir()
            pilot.immutable_json(attempt_root / "resume.json", resume)
            with pilot.installed_signal_handlers():
                try:
                    topology = pilot.verify_two_idle_b200s()
                    pilot.immutable_json(attempt_root / "topology.json", topology)
                    server_command = build_server_command(
                        source_root=source_root, attempt_root=attempt_root,
                        port=args.port, study_commit=args.study_commit,
                        start_cell_index=start_cell_index, block=block,
                        candidate_id=args.candidate_id,
                        gate_receipt=Path(args.gate_receipt),
                        gate_receipt_sha256=args.gate_receipt_sha256,
                        pose_manifest=Path(args.pose_manifest),
                        pose_manifest_sha256=args.pose_manifest_sha256,
                        capture_receipt=Path(args.capture_receipt),
                        capture_receipt_sha256=args.capture_receipt_sha256,
                        confirmation_freeze=Path(args.confirmation_freeze),
                        confirmation_freeze_sha256=args.confirmation_freeze_sha256,
                        fixture_freeze_path=Path(args.fixture_freeze),
                        fixture_freeze_sha256=args.fixture_freeze_sha256,
                    )
                    server_environment = pilot.build_model_environment(
                        source_root=source_root, attempt_root=attempt_root
                    )
                    server_process, server_stdout, server_stderr = pilot._launch_logged(
                        server_command, cwd=pilot.COSMOS_ROOT,
                        environment=server_environment,
                        stdout_path=attempt_root / "server" / "stdout.log",
                        stderr_path=attempt_root / "server" / "stderr.log",
                    )
                    pilot.immutable_json(
                        attempt_root / "server" / "process.json",
                        {
                            "schema_version": "wmf-n3-behavioral-confirmation-server-process-v1",
                            "block_id": block.block_id,
                            "layout_pair_id": block.layout_pair_id,
                            "command": server_command,
                            "process": pilot._proc_identity(server_process),
                            "started_at_utc": pilot.utc_now(),
                        },
                    )
                    server_ready = pilot._wait_for_server(
                        server_process, attempt_root / "server" / "ready.json",
                        port=args.port, timeout=args.server_ready_timeout,
                    )
                    _verify_ready_binding(attempt_root / "server" / "ready.json", prerequisites)
                    for condition_index in range(start_cell_index, 4):
                        pilot.require(server_process.poll() is None, "n3_server_exited_between_cells")
                        cell_command = build_cell_command(
                            source_root=source_root, attempt_root=attempt_root,
                            study_commit=args.study_commit,
                            gate_receipt=Path(args.gate_receipt),
                            gate_receipt_sha256=args.gate_receipt_sha256,
                            pose_manifest=Path(args.pose_manifest),
                            pose_manifest_sha256=args.pose_manifest_sha256,
                            capture_receipt=Path(args.capture_receipt),
                            capture_receipt_sha256=args.capture_receipt_sha256,
                            candidate_id=args.candidate_id,
                            confirmation_freeze=Path(args.confirmation_freeze),
                            confirmation_freeze_sha256=args.confirmation_freeze_sha256,
                            fixture_freeze_path=Path(args.fixture_freeze),
                            fixture_freeze_sha256=args.fixture_freeze_sha256,
                            port=args.port, condition_index=condition_index, block=block,
                        )
                        cell_id = block.cell_ids[condition_index]
                        log_root = attempt_root / "cell_logs" / f"{condition_index:02d}-{pilot.safe_cell_component(cell_id)}"
                        simulator_environment = pilot.build_simulator_environment(
                            source_root=source_root, state_parent=attempt_root
                        )
                        with pilot.supervised_logged_child(
                            cell_command, cwd=pilot.ROBOLAB_ROOT,
                            environment=simulator_environment,
                            stdout_path=log_root / "stdout.log",
                            stderr_path=log_root / "stderr.log",
                        ) as (child, _stdout, _stderr):
                            launched += 1
                            pilot.immutable_json(
                                log_root / "process.json",
                                {
                                    "schema_version": "wmf-n3-behavioral-confirmation-cell-process-v1",
                                    "block_id": block.block_id,
                                    "cell_id": cell_id,
                                    "condition_index": condition_index,
                                    "command": cell_command,
                                    "process": pilot._proc_identity(child),
                                    "started_at_utc": pilot.utc_now(),
                                },
                            )
                            try:
                                code = child.wait(timeout=args.cell_timeout)
                            except subprocess.TimeoutExpired:
                                pilot.terminate_process_group(child)
                                raise pilot.N3BehavioralPilotError("n3_behavioral_cell_timeout", cell_id)
                            pilot.require(code == 0, "n3_behavioral_cell_failed", f"{cell_id}:{code}")
                        receipt_path = (
                            attempt_root / "cells"
                            / f"{condition_index:02d}-{pilot.safe_cell_component(cell_id)}"
                            / "cell_receipt.json"
                        )
                        receipt, identity = validate_passed_confirmation_cell(
                            receipt_path, condition_index=condition_index,
                            study_commit=args.study_commit, block=block,
                        )
                        validate_cells_bind_prerequisites([receipt], prerequisites=prerequisites)
                        completed.append(receipt)
                        new_identities.append(identity)
                    pilot.require(len(completed) == 4, "n3_confirmation_block_incomplete")
                    context_ids = [
                        receipt["server_begin_receipt"]["server_context_id"]
                        for receipt in completed
                    ]
                    pilot.require(
                        len(context_ids) == len(set(context_ids)),
                        "confirmation_server_context_id_reused",
                    )
                except BaseException as error:
                    failure = error
                finally:
                    for residual in reversed(list(pilot._ACTIVE_CHILDREN)):
                        if residual is server_process:
                            continue
                        try:
                            pilot.terminate_process_group(residual)
                        finally:
                            if residual in pilot._ACTIVE_CHILDREN:
                                pilot._ACTIVE_CHILDREN.remove(residual)
                    if server_process is not None:
                        pilot.terminate_process_group(server_process)
                        if server_process in pilot._ACTIVE_CHILDREN:
                            pilot._ACTIVE_CHILDREN.remove(server_process)
                        server_exit = {
                            "schema_version": pilot.SERVER_EXIT_SCHEMA,
                            "returncode": server_process.returncode,
                            "child_reaped": server_process.poll() is not None,
                            "terminated_by_queue_supervisor": True,
                            "ended_at_utc": pilot.utc_now(),
                        }
                        if not (attempt_root / "server" / "supervisor_exit.json").exists():
                            pilot.immutable_json(attempt_root / "server" / "supervisor_exit.json", server_exit)
                    if server_stdout is not None and server_stderr is not None:
                        pilot._close_process_logs(server_stdout, server_stderr)
                    remaining = pilot._wait_for_gpu_cleanup()
                    pilot.immutable_json(
                        attempt_root / "cleanup.json",
                        {
                            "schema_version": "wmf-n3-behavioral-confirmation-cleanup-v1",
                            "all_children_reaped": not pilot._ACTIVE_CHILDREN,
                            "remaining_compute_processes": remaining,
                            "completed_at_utc": pilot.utc_now(),
                        },
                    )
                    if remaining and failure is None:
                        failure = pilot.N3BehavioralPilotError("gpu_process_remained_after_cleanup")

            counts = _receipt_counts(
                block=block, start_cell_index=start_cell_index,
                launched=launched, completed=completed, attempt_root=attempt_root,
            )
            status = "passed" if failure is None else "technical_failure"
            receipt = {
                "schema_version": QUEUE_RECEIPT_SCHEMA,
                "status": status,
                "exit_code": 0 if failure is None else 1,
                "study_id": pilot.STUDY_ID,
                "namespace": pilot.NAMESPACE,
                "block_id": block.block_id,
                "phase": "confirmation",
                "layout_pair_id": block.layout_pair_id,
                "model_config": "N3",
                "effective_seed": block.effective_seed,
                "condition_order": list(block.condition_order),
                "cell_ids": list(block.cell_ids),
                "counts": counts,
                "source_commit": args.study_commit,
                "queue_descriptor": queue_identity,
                "schedule": {"path": str(block.schedule_path), "sha256": block.schedule_sha256},
                "prerequisites": prerequisites,
                "topology": pilot.file_identity(attempt_root / "topology.json") if topology is not None else None,
                "server_ready": pilot.file_identity(attempt_root / "server" / "ready.json") if server_ready is not None else None,
                "server_exit": server_exit,
                "resume": pilot.file_identity(attempt_root / "resume.json"),
                "resumed_cell_receipts": resumed_identities,
                "new_cell_receipts": new_identities,
                "cell_receipts": resumed_identities + new_identities,
                "raw_attempt_root": str(attempt_root),
                "raw_attempt_recoverable_on_gm_pvc": True,
                "failure": None if failure is None else {
                    "error_type": type(failure).__name__,
                    "reason": getattr(failure, "reason", None),
                    "detail": str(failure),
                },
                "completed_at_utc": pilot.utc_now(),
                "claim_boundary": (
                    "This receipt counts only valid 450-action N3 confirmation cells. Pilot, "
                    "development, qualification, annotation, and fixture-freeze evidence are "
                    "prerequisites and are never recounted."
                ),
            }
            pilot.immutable_json(attempt_root / "publish" / AGGREGATE_FILENAME, receipt, publish=True)
            publish_path.parent.mkdir(parents=True, exist_ok=True)
            pilot.immutable_json(publish_path, receipt, publish=True)
            print(json.dumps({"status": status, "counts": counts, "raw_attempt_root": str(attempt_root)}, sort_keys=True), flush=True)
            return 0 if failure is None else 1
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def run_server(args: argparse.Namespace, block: development.DevelopmentBlock) -> int:
    prerequisites = validate_prerequisites(args, block, include_pilot=False)
    original = pilot.immutable_json

    def immutable(path: Path, value: Mapping[str, Any], *, publish: bool = False) -> None:
        updated = dict(value)
        if updated.get("schema_version") == pilot.SERVER_READY_SCHEMA:
            updated["phase"] = "confirmation"
            updated["confirmation_release"] = prerequisites["confirmation_release"]
            updated["confirmation_fixture_freeze"] = prerequisites["confirmation_fixture_freeze"]
        original(path, updated, publish=publish)

    pilot.immutable_json = immutable
    try:
        return pilot.run_server(args)
    finally:
        pilot.immutable_json = original


def run_cell(args: argparse.Namespace, block: development.DevelopmentBlock) -> int:
    prerequisites = validate_prerequisites(args, block, include_pilot=False)
    _verify_ready_binding(Path(args.attempt_root) / "server" / "ready.json", prerequisites)
    forecast = Path(args.source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    original_verifier = recorder_job.verify_fixture_release
    original_immutable = pilot.immutable_json

    def fixture_verifier(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        selected = prerequisites["selected_fixture"]
        return {
            "gate_receipt": selected["gate_receipt"],
            "pose_manifest": selected["pose_manifest"],
            "gate_ledger": selected["gate_ledger"],
            "gate_attempt_receipt": selected["gate_attempt_receipt"],
            "accepted_gate_record_sha256": selected["accepted_gate_record_sha256"],
            "candidate_id": selected["candidate_id"],
            "candidate_payload_sha256": selected["candidate_payload_sha256"],
        }

    def immutable(path: Path, value: Mapping[str, Any], *, publish: bool = False) -> None:
        updated = dict(value)
        if Path(path).name == "cell_receipt.json" and updated.get("schema_version") == CELL_RECEIPT_SCHEMA:
            updated["phase"] = "confirmation"
            updated["confirmation_prerequisites"] = _expected_prerequisite_binding(prerequisites)
            updated["transport_contract"] = dict(NO_REPLAY_TRANSPORT_CONTRACT)
            updated["claim_boundary"] = (
                "One valid learned-policy N3 behavioral confirmation cell: 450 actual actions, "
                "451 original observations, and 15 jointly generated action/future requests."
            )
        original_immutable(path, updated, publish=publish)

    recorder_job.verify_fixture_release = fixture_verifier
    pilot.immutable_json = immutable
    try:
        with development.installed_no_replay_cosmos_client():
            return pilot.run_cell(args)
    finally:
        recorder_job.verify_fixture_release = original_verifier
        pilot.immutable_json = original_immutable


def _add_freezes(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--confirmation-freeze", type=Path, required=True)
    parser.add_argument("--confirmation-freeze-sha256", required=True)
    parser.add_argument("--fixture-freeze", type=Path, required=True)
    parser.add_argument("--fixture-freeze-sha256", required=True)


def _add_fixture(parser: argparse.ArgumentParser) -> None:
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
    queue = subparsers.add_parser("queue")
    queue.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS, required=True)
    queue.add_argument("--source-root", type=Path, required=True)
    queue.add_argument("--study-commit", required=True)
    queue.add_argument("--job-dir", type=Path, required=True)
    queue.add_argument("--job-id", required=True)
    queue.add_argument("--raw-root", type=Path)
    _add_fixture(queue)
    _add_freezes(queue)
    queue.add_argument("--pilot-receipt", type=Path, required=True)
    queue.add_argument("--pilot-receipt-sha256", required=True)
    queue.add_argument("--port", type=int, default=pilot.DEFAULT_PORT)
    queue.add_argument("--server-ready-timeout", type=float, default=7200.0)
    queue.add_argument("--cell-timeout", type=float, default=10800.0)

    server = subparsers.add_parser("server")
    server.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS, required=True)
    server.add_argument("--source-root", type=Path, required=True)
    server.add_argument("--study-commit", required=True)
    server.add_argument("--attempt-root", type=Path, required=True)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=pilot.DEFAULT_PORT)
    server.add_argument("--start-cell-index", type=int, default=0)
    _add_fixture(server)
    _add_freezes(server)

    cell = subparsers.add_parser("cell")
    cell.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS, required=True)
    cell.add_argument("--source-root", type=Path, required=True)
    cell.add_argument("--study-commit", required=True)
    cell.add_argument("--attempt-root", type=Path, required=True)
    _add_fixture(cell)
    _add_freezes(cell)
    cell.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    cell.add_argument("--command", choices=("left", "right"), required=True)
    cell.add_argument("--condition-index", type=int, required=True)
    cell.add_argument("--remote-host", default="127.0.0.1")
    cell.add_argument("--remote-port", type=int, default=pilot.DEFAULT_PORT)
    return parser


def _validate_sha_options(args: argparse.Namespace) -> None:
    for name, value in vars(args).items():
        if name.endswith("_sha256") and value is not None:
            pilot.require(isinstance(value, str) and pilot.SHA256_RE.fullmatch(value) is not None, "invalid_sha256", name)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    block = load_confirmation_block(Path(args.source_root), args.layout_pair_id)
    pilot.require(pilot.COMMIT_RE.fullmatch(args.study_commit) is not None, "invalid_study_commit")
    _validate_sha_options(args)
    port = getattr(args, "port", getattr(args, "remote_port", 0))
    pilot.require(1 <= port <= 65535, "invalid_port")
    with configured_pilot(block):
        if args.mode == "queue":
            pilot.require(args.server_ready_timeout > 0 and args.cell_timeout > 0, "invalid_timeout")
            return run_queue(args, block)
        if args.mode == "server":
            pilot.require(0 <= args.start_cell_index < 4, "invalid_start_cell_index")
            return run_server(args, block)
        if args.mode == "cell":
            pilot.require(0 <= args.condition_index < 4, "invalid_condition_index")
            arm, command, _task = block.conditions[args.condition_index]
            pilot.require((args.layout_arm, args.command) == (arm, command), "confirmation_cell_condition_changed")
            return run_cell(args, block)
    raise pilot.N3BehavioralPilotError("unknown_mode")


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
