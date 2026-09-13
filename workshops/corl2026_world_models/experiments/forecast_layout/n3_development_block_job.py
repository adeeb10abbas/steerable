#!/usr/bin/env python3
"""Run one frozen D01--D04 N3 development layout block.

This launcher is deliberately limited to the four development blocks.  It
loads the exact order and effective seed from ``parallel_schedule.json``,
requires a hash-bound accepted gate and frozen pose manifest for the selected
layout, and requires a fully passed four-cell P00 N3 pilot before starting a
model server.  Confirmation layouts are not supported by this program.

The model, simulator, recorder, temporal-reset protocol, and child-process
supervision are reused from :mod:`n3_behavioral_pilot_job`.  Each executable
mode installs one immutable development block configuration for the lifetime
of its process.  Queue attempts recover only a deeply validated contiguous
prefix; a valid cell is never launched again under a later attempt ID.
"""

from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterator, Mapping, Sequence


# Queue source checkouts are immutable.  Importing the shared implementation
# under the system Python must not dirty them with bytecode.
sys.dont_write_bytecode = True

import n3_behavioral_pilot_job as pilot


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
P00_CONDITION_ORDER = (
    "reflected-right",
    "reflected-left",
    "original-left",
    "original-right",
)
P00_BLOCK_ID = "wmf_ablation_001_20260912__pilot__P00__N3"
P00_EFFECTIVE_SEED = 2026091000
P00_CELL_IDS = tuple(
    f"wmf1__pilot__P00__N3__{condition.replace('-', '__')}"
    for condition in P00_CONDITION_ORDER
)
LEGACY_P00_SOURCE_COMMIT = "b8914413f6293c165dd3a57c737f4b977e7e84be"
LEGACY_P00_AGGREGATE_SHA256 = (
    "5a7a861acc82615b4eb7aa92308f40f7a0c965e65b6edaf12f776cb64b18b0e0"
)
LEGACY_P00_CELL_RECEIPT_SHA256S = (
    "0a53e9b6df118d79a6361f2f1d1853ec42667120fc1d7caa50d4c3887351b8b9",
    "7a1a9ff1b3f31f4f56634a75c27ada25db5c7113f986bf64202e85febbc8794d",
    "805e6723040a4fb5a7e7574a8fe489211a61207379833db5b14924401282220e",
    "83444671630fbe97f0da50fa938c0c76be1cf65fa7143b33b9abc7fb5038cea1",
)
LEGACY_P00_SERVER_READY_SHA256 = (
    "07ff8b7370d811f98e1474cd8fbc8575d0062f2c76b19f010196f1dbe940b5ba"
)

RAW_PARENT = Path(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912/behavioral/development/N3"
)
RUNNER_FILENAME = "n3_development_block_job.py"
AGGREGATE_FILENAME = "n3_behavioral_development_receipt.json"
QUEUE_RECEIPT_SCHEMA = "wmf-n3-behavioral-development-job-v1"
CELL_RECEIPT_SCHEMA = "wmf-n3-behavioral-development-cell-v1"
RESUME_SCHEMA = "wmf-n3-behavioral-development-resume-v1"
PILOT_QUEUE_RECEIPT_SCHEMA = "wmf-n3-behavioral-pilot-job-v1"
PILOT_CELL_RECEIPT_SCHEMA = "wmf-n3-behavioral-pilot-cell-v1"
SEED_AUDIT_SCHEMA = "wmf-bounded-seed-audit-v1"
SAFE_LAYOUT_RE = re.compile(r"D0[1-4]\Z")
_CONFIGURATION_ACTIVE = False
NO_REPLAY_TRANSPORT_CONTRACT = {
    "schema_version": "wmf-n3-no-replay-websocket-v1",
    "compression": None,
    "max_size": None,
    "ping_interval": None,
    "ping_timeout": None,
    "behavioral_request_retry_count": 0,
    "lost_response_policy": "fail_cell_without_replaying_request",
    "outer_bound": "queue_cell_timeout",
}


@dataclass(frozen=True)
class DevelopmentBlock:
    """The exact frozen identity of one N3 development block."""

    layout_pair_id: str
    effective_seed: int
    condition_order: tuple[str, ...]
    conditions: tuple[tuple[str, str, str], ...]
    cell_ids: tuple[str, ...]
    block_id: str
    raw_root: Path
    schedule_path: Path
    schedule_sha256: str
    schedule_row: Mapping[str, Any]


def _condition_tuple(label: str) -> tuple[str, str, str]:
    pilot.require(label.count("-") == 1, "development_condition_label_invalid", label)
    arm, command = label.split("-", 1)
    pilot.require(arm in {"original", "reflected"}, "development_condition_arm_invalid")
    pilot.require(command in {"left", "right"}, "development_condition_command_invalid")
    task_name = f"WMFForecast{arm.title()}{command.title()}Task"
    return arm, command, task_name


def _expected_cell_ids(layout_pair_id: str, order: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        f"wmf1__development__{layout_pair_id}__N3__{label.replace('-', '__')}"
        for label in order
    )


def load_development_block(source_root: Path, layout_pair_id: str) -> DevelopmentBlock:
    """Load one exact development row and reject every non-development layout."""

    pilot.require(
        isinstance(layout_pair_id, str)
        and SAFE_LAYOUT_RE.fullmatch(layout_pair_id) is not None
        and layout_pair_id in DEVELOPMENT_SCHEDULE,
        "development_layout_unsupported",
        str(layout_pair_id),
    )
    effective_seed, order = DEVELOPMENT_SCHEDULE[layout_pair_id]
    block_id = f"wmf_ablation_001_20260912__development__{layout_pair_id}__N3"
    schedule_path = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
    )
    schedule = pilot.load_json(schedule_path, "parallel_schedule_unreadable")
    rows = [row for row in schedule.get("jobs", []) if row.get("job_id") == block_id]
    pilot.require(len(rows) == 1, "development_schedule_row_missing_or_duplicate")
    row = rows[0]
    expected_cells = _expected_cell_ids(layout_pair_id, order)
    expected_contract = {
        "full_model_and_simulator_reset_before_each_condition": True,
        "no_global_DreamZero_context_interleaving": True,
        "same_isolated_worker_type_within_job": True,
        "sequential_conditions": True,
    }
    expected_gates = [
        "all_four_pilot_recordings_qualified_or_documented_exact_runtime_waiver",
        "qualified_development_fixtures",
        "bounded_resource_authorization",
    ]
    expected = {
        "phase": "development",
        "layout_pair_id": layout_pair_id,
        "model_config": "N3",
        "candidate_effective_policy_seed": effective_seed,
        "condition_order": list(order),
        "ordered_cell_ids": list(expected_cells),
        "indivisible": True,
        "execution_contract": expected_contract,
        "depends_on_jobs": [P00_BLOCK_ID],
        "required_gates": expected_gates,
        "released": False,
        "status": "NOT_RELEASED",
    }
    for key, wanted in expected.items():
        pilot.require(row.get(key) == wanted, "development_schedule_mismatch", key)
    return DevelopmentBlock(
        layout_pair_id=layout_pair_id,
        effective_seed=effective_seed,
        condition_order=tuple(order),
        conditions=tuple(_condition_tuple(label) for label in order),
        cell_ids=expected_cells,
        block_id=block_id,
        raw_root=RAW_PARENT / layout_pair_id,
        schedule_path=schedule_path,
        schedule_sha256=pilot.sha256_file(schedule_path),
        schedule_row=dict(row),
    )


@contextmanager
def configured_pilot(block: DevelopmentBlock) -> Iterator[None]:
    """Install a block only inside this isolated executable process."""

    global _CONFIGURATION_ACTIVE
    pilot.require(not _CONFIGURATION_ACTIVE, "development_configuration_overlap")
    _CONFIGURATION_ACTIVE = True
    replacements = {
        "PHASE": "development",
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


def build_no_replay_transport_types(
    *, original_client_class: type, websocket_policy_class: type,
    connect: Any, unpackb: Any,
) -> tuple[type, type]:
    """Build the exact bounded transport used after the live replay finding."""

    class DevelopmentRecordedTransport(websocket_policy_class):
        def _wait_for_server(self):
            # Official inference can block the server event loop longer than
            # the library ping timeout.  The queue bounds the entire cell.
            connection = connect(
                self._uri,
                compression=None,
                max_size=None,
                ping_interval=None,
                ping_timeout=None,
            )
            try:
                return connection, unpackb(connection.recv())
            except BaseException:
                connection.close()
                raise

    class DevelopmentNoReplayClient(original_client_class):
        def _connect(self):
            return DevelopmentRecordedTransport(self._remote_host, self._remote_port)

        def _query_server(self, request: dict) -> dict:
            # A lost response is technical-invalid evidence.  Retrying here
            # could execute a second real generation under one request index.
            return self.client.infer(request)

    return DevelopmentRecordedTransport, DevelopmentNoReplayClient


@contextmanager
def installed_no_replay_cosmos_client() -> Iterator[None]:
    """Patch the pinned client exactly when RoboLab imports it after Isaac starts."""

    module_name = "policies.cosmos3.client"
    original_import = builtins.__import__
    patched_module: Any = None
    original_client_class: type | None = None

    def patch_loaded_module() -> None:
        nonlocal patched_module, original_client_class
        if patched_module is not None or module_name not in sys.modules:
            return
        module = sys.modules[module_name]
        websockets_client = original_import(
            "websockets.sync.client", fromlist=("connect",)
        )
        openpi_client = original_import(
            "openpi_client", fromlist=("msgpack_numpy", "websocket_client_policy")
        )
        original_client_class = module.Cosmos3Client
        _transport, replacement = build_no_replay_transport_types(
            original_client_class=original_client_class,
            websocket_policy_class=openpi_client.websocket_client_policy.WebsocketClientPolicy,
            connect=websockets_client.connect,
            unpackb=openpi_client.msgpack_numpy.unpackb,
        )
        module.Cosmos3Client = replacement
        patched_module = module
        # The target has been patched; restore the interpreter-wide import
        # function immediately rather than holding a hook during the episode.
        builtins.__import__ = original_import

    def guarded_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        try:
            result = original_import(name, globals, locals, fromlist, level)
            if name == module_name:
                patch_loaded_module()
            return result
        except BaseException:
            builtins.__import__ = original_import
            raise

    try:
        if module_name in sys.modules:
            patch_loaded_module()
        else:
            builtins.__import__ = guarded_import
        yield
    finally:
        builtins.__import__ = original_import
        if patched_module is not None and original_client_class is not None:
            patched_module.Cosmos3Client = original_client_class


def _verify_descriptor(value: Any, label: str) -> dict[str, Any]:
    return pilot._verify_descriptor(value, label)


def _legacy_payload_canonical_bytes(value: Any) -> bytes:
    """Match the recorder's payload-descriptor canonicalization exactly."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _thaw_legacy_zero_array_payload(value: Any) -> Any:
    """Thaw only the recorder's JSON-only payload vocabulary.

    The released P00 context-reset receipts contain no arrays.  Keeping this
    decoder local lets the queue's deliberately minimal system Python verify
    those exact descriptors without importing the NumPy-backed recorder.
    """

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    pilot.require(
        isinstance(value, Mapping),
        "legacy_p00_context_reset_payload_node_invalid",
    )
    kind = value.get("__type__")
    if kind == "mapping":
        pilot.require(
            set(value) == {"__type__", "items"}
            and isinstance(value.get("items"), Mapping)
            and all(isinstance(key, str) for key in value["items"]),
            "legacy_p00_context_reset_mapping_invalid",
        )
        return {
            key: _thaw_legacy_zero_array_payload(item)
            for key, item in value["items"].items()
        }
    if kind in {"list", "tuple"}:
        pilot.require(
            set(value) == {"__type__", "items"}
            and isinstance(value.get("items"), list),
            "legacy_p00_context_reset_sequence_invalid",
        )
        items = [_thaw_legacy_zero_array_payload(item) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "path":
        pilot.require(
            set(value) == {"__type__", "value"}
            and isinstance(value.get("value"), str),
            "legacy_p00_context_reset_path_invalid",
        )
        return Path(value["value"])
    # In particular, never accept an ndarray node on this zero-array path.
    raise pilot.N3BehavioralPilotError(
        "legacy_p00_context_reset_payload_type_invalid"
    )


def _load_legacy_zero_array_payload(descriptor: Mapping[str, Any]) -> Any:
    """Authenticate and thaw one released JSON-only recorder payload."""

    pilot.require(
        set(descriptor)
        == {"role", "structure", "array_count", "payload_sha256"},
        "legacy_p00_context_reset_descriptor_shape_invalid",
    )
    pilot.require(
        descriptor.get("role") == "context_reset"
        and type(descriptor.get("array_count")) is int
        and descriptor["array_count"] == 0
        and "artifact" not in descriptor,
        "legacy_p00_context_reset_descriptor_requires_arrays",
    )
    expected_sha256 = descriptor.get("payload_sha256")
    pilot.require(
        isinstance(expected_sha256, str)
        and pilot.SHA256_RE.fullmatch(expected_sha256) is not None,
        "legacy_p00_context_reset_payload_sha256_invalid",
    )
    unsigned = {
        key: value for key, value in descriptor.items() if key != "payload_sha256"
    }
    observed_sha256 = hashlib.sha256(
        _legacy_payload_canonical_bytes(unsigned)
    ).hexdigest()
    pilot.require(
        observed_sha256 == expected_sha256,
        "legacy_p00_context_reset_payload_hash_mismatch",
    )
    return _thaw_legacy_zero_array_payload(descriptor["structure"])


def _load_gate_ledger(path: Path) -> list[dict[str, Any]]:
    """Read and authenticate the generic append-only physical-gate ledger."""

    forecast = Path(__file__).resolve().parent
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    try:
        return recorder_job._read_gate_ledger(path)
    except BaseException as error:
        raise pilot.N3BehavioralPilotError("development_gate_ledger_invalid") from error


def verify_development_fixture_release(
    *,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    layout_arm: str,
    block: DevelopmentBlock | None = None,
) -> dict[str, Any]:
    """Verify the exact accepted-gate -> frozen-pose chain for one D layout."""

    if block is None:
        active = getattr(verify_development_fixture_release, "_active_block", None)
        pilot.require(isinstance(active, DevelopmentBlock), "development_block_not_configured")
        block = active
    pilot.require(layout_arm in {"original", "reflected"}, "development_layout_arm_invalid")
    gate_identity = pilot.verify_exact_file(gate_receipt_path, gate_receipt_sha256, "gate_receipt")
    pose_identity = pilot.verify_exact_file(pose_manifest_path, pose_manifest_sha256, "pose_manifest")
    gate = pilot.load_json(Path(gate_identity["path"]), "development_gate_receipt_unreadable")
    pose = pilot.load_json(Path(pose_identity["path"]), "development_pose_manifest_unreadable")

    forecast = Path(__file__).resolve().parent
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job
    from fixture_layouts import validate_frozen_pose_manifest

    pilot.require(gate.get("schema_version") == recorder_job.GATE_RECEIPT_SCHEMA, "development_gate_schema_changed")
    pilot.require(gate.get("study_namespace") == pilot.NAMESPACE, "development_gate_namespace_changed")
    pilot.require(gate.get("status") == "finished", "development_gate_unfinished")
    pilot.require(gate.get("layout_pair_id") == block.layout_pair_id, "development_gate_layout_mismatch")
    pilot.require(
        gate.get("decision") == "accepted" and gate.get("exit_code") == 0,
        "development_fixture_not_accepted",
    )
    pilot.require(gate.get("model_request_count") == 0, "development_gate_contains_model_requests")
    pilot.require(gate.get("behavioral_action_count") == 0, "development_gate_contains_behavioral_actions")
    pilot.require(
        gate.get("source_contract_sha256") == recorder_job.SOURCE_CONTRACT_SHA256,
        "development_gate_source_contract_changed",
    )
    pilot.require(
        gate.get("candidate_pool_sha256") == recorder_job.CANDIDATE_POOL_SHA256,
        "development_gate_candidate_pool_changed",
    )
    candidate_id = gate.get("candidate_id")
    candidate_sha = gate.get("candidate_payload_sha256")
    pilot.require(
        isinstance(candidate_id, str) and candidate_id.startswith(f"{block.layout_pair_id}__candidate_"),
        "development_gate_candidate_mismatch",
    )
    pilot.require(
        isinstance(candidate_sha, str) and pilot.SHA256_RE.fullmatch(candidate_sha) is not None,
        "development_gate_candidate_hash_invalid",
    )

    try:
        validate_frozen_pose_manifest(pose)
    except BaseException as error:
        raise pilot.N3BehavioralPilotError("development_pose_manifest_invalid") from error
    pilot.require(
        pose.get("candidate_pool_sha256") == recorder_job.CANDIDATE_POOL_SHA256,
        "development_pose_candidate_pool_changed",
    )
    pilot.require(pose.get("layout_pair_ids") == [block.layout_pair_id], "development_pose_layout_inventory_changed")
    pilot.require(pose.get("qualified_layout_count") == 1, "development_pose_layout_count_changed")
    task = pose.get("task_contract")
    pilot.require(isinstance(task, Mapping), "development_pose_task_contract_missing")
    pilot.require(task.get("action_cap") == pilot.ACTION_CAP, "development_pose_action_cap_changed")
    pilot.require(task.get("termination_terms") == ["time_out"], "development_pose_not_timeout_only")
    pilot.require(task.get("success_is_measurement_only") is True, "development_pose_success_contract_changed")
    pilot.require(task.get("success_termination_present") is False, "development_pose_success_termination_present")
    rows = pose.get("layout_pairs")
    pilot.require(isinstance(rows, Mapping), "development_pose_rows_invalid")
    row = rows.get(block.layout_pair_id)
    pilot.require(isinstance(row, Mapping), "development_pose_row_missing")
    pilot.require(row.get("layout_pair_id") == block.layout_pair_id, "development_pose_row_id_changed")
    pilot.require(row.get("candidate_id") == candidate_id, "development_pose_candidate_mismatch")
    pilot.require(row.get("candidate_payload_sha256") == candidate_sha, "development_pose_candidate_hash_changed")
    layouts = row.get("layouts")
    pilot.require(
        isinstance(layouts, Mapping) and set(layouts) == {"original", "reflected"},
        "development_pose_arms_changed",
    )
    pilot.require(layout_arm in layouts, "development_pose_arm_missing")

    gate_evidence = gate.get("gate_evidence")
    pilot.require(isinstance(gate_evidence, Mapping), "development_gate_evidence_missing")
    record_sha = gate_evidence.get("gate_record_sha256")
    pilot.require(
        isinstance(record_sha, str) and pilot.SHA256_RE.fullmatch(record_sha) is not None,
        "development_gate_record_hash_invalid",
    )
    pilot.require(row.get("accepted_gate_record_sha256") == record_sha, "development_pose_gate_record_changed")
    ledger_identity = _verify_descriptor(gate.get("gate_ledger"), "development_gate_ledger")
    pilot.require(pose.get("gate_ledger_sha256") == ledger_identity["sha256"], "development_pose_gate_ledger_changed")
    records = _load_gate_ledger(Path(ledger_identity["path"]))
    matches = [
        record
        for record in records
        if record.get("record_sha256") == record_sha
        and record.get("layout_pair_id") == block.layout_pair_id
    ]
    pilot.require(len(matches) == 1, "development_gate_record_missing_or_duplicate")
    record = matches[0]
    pilot.require(record.get("decision") == "accepted" and record.get("passed") is True, "development_gate_record_not_accepted")
    pilot.require(record.get("candidate_id") == candidate_id, "development_gate_record_candidate_changed")
    pilot.require(record.get("candidate_payload_sha256") == candidate_sha, "development_gate_record_candidate_hash_changed")
    pilot.require(
        record.get("candidate_pool_sha256") == recorder_job.CANDIDATE_POOL_SHA256,
        "development_gate_record_pool_changed",
    )
    pilot.require(
        record.get("model_request_count") == 0 and record.get("behavioral_action_count") == 0,
        "development_gate_record_contains_behavioral_work",
    )
    attempt_identity = _verify_descriptor(record.get("attempt_receipt"), "development_gate_attempt")
    gate_attempt_identity = _verify_descriptor(
        gate_evidence.get("gate_attempt_receipt"), "development_receipt_gate_attempt"
    )
    pose_attempt_identity = _verify_descriptor(
        row.get("accepted_gate_attempt_receipt"), "development_pose_gate_attempt"
    )
    pilot.require(
        attempt_identity == gate_attempt_identity == pose_attempt_identity,
        "development_gate_attempt_chain_changed",
    )
    attempt = pilot.load_json(Path(attempt_identity["path"]), "development_gate_attempt_unreadable")
    pilot.require(attempt.get("schema_version") == recorder_job.GATE_ATTEMPT_SCHEMA, "development_gate_attempt_schema_changed")
    pilot.require(attempt.get("candidate_id") == candidate_id, "development_gate_attempt_candidate_changed")
    pilot.require(attempt.get("candidate_payload_sha256") == candidate_sha, "development_gate_attempt_candidate_hash_changed")
    pilot.require(
        attempt.get("candidate_pool_sha256") == recorder_job.CANDIDATE_POOL_SHA256,
        "development_gate_attempt_pool_changed",
    )
    pilot.require(attempt.get("decision") == "accepted" and attempt.get("passed") is True, "development_gate_attempt_not_accepted")
    pilot.require(
        attempt.get("model_request_count") == 0 and attempt.get("behavioral_action_count") == 0,
        "development_gate_attempt_contains_behavioral_work",
    )
    evaluation = attempt.get("evaluation")
    pilot.require(isinstance(evaluation, Mapping) and evaluation.get("passed") is True, "development_gate_attempt_evaluation_failed")
    return {
        "gate_receipt": gate_identity,
        "pose_manifest": pose_identity,
        "gate_ledger": ledger_identity,
        "gate_attempt_receipt": attempt_identity,
        "accepted_gate_record_sha256": record_sha,
        "candidate_id": candidate_id,
        "candidate_payload_sha256": candidate_sha,
        "pose_row": dict(row),
    }


def _validate_cell_receipt(
    path: Path,
    *,
    phase: str,
    layout_pair_id: str,
    block_id: str,
    effective_seed: int,
    conditions: Sequence[tuple[str, str, str]],
    cell_ids: Sequence[str],
    schema_version: str,
    condition_index: int,
    legacy_p00_source_commit: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    pilot.require(0 <= condition_index < len(cell_ids), "resume_cell_index_invalid")
    receipt = pilot.load_json(path, "resume_cell_receipt_unreadable")
    arm, command, _task = conditions[condition_index]
    expected = {
        "schema_version": schema_version,
        "status": "passed",
        "study_id": pilot.STUDY_ID,
        "block_id": block_id,
        "cell_id": cell_ids[condition_index],
        "condition_index": condition_index,
        "layout_pair_id": layout_pair_id,
        "layout_arm": arm,
        "command": command,
        "prompt": pilot.PROMPTS[command],
        "model_config": pilot.MODEL_CONFIG,
        "effective_seed": effective_seed,
        "actions_executed": pilot.ACTION_CAP,
        "observation_count": pilot.OBSERVATION_COUNT,
        "behavioral_model_request_count": pilot.REQUEST_COUNT,
        "behavioral_episode_count": 1,
        "generation_qualification_request_count": 0,
        "final_chunk_executed_actions": pilot.FINAL_EXECUTED_ACTIONS,
    }
    for key, wanted in expected.items():
        pilot.require(receipt.get(key) == wanted, "resume_cell_receipt_mismatch", key)
    candidate_id = receipt.get("candidate_id")
    pilot.require(
        isinstance(candidate_id, str)
        and candidate_id.startswith(f"{layout_pair_id}__candidate_"),
        "resume_cell_candidate_id_invalid",
    )
    accepted_gate_record_sha256 = receipt.get("accepted_gate_record_sha256")
    pilot.require(
        isinstance(accepted_gate_record_sha256, str)
        and pilot.SHA256_RE.fullmatch(accepted_gate_record_sha256) is not None,
        "resume_cell_gate_record_sha256_invalid",
    )
    if phase == "development":
        pilot.require(
            receipt.get("transport_contract") == NO_REPLAY_TRANSPORT_CONTRACT,
            "resume_cell_transport_contract_mismatch",
        )
    if legacy_p00_source_commit is None:
        source_pins = receipt.get("source_pins")
        pilot.require(isinstance(source_pins, Mapping), "resume_cell_source_pin_mismatch")
        pilot.require(
            isinstance(source_pins.get("study_commit"), str)
            and pilot.COMMIT_RE.fullmatch(source_pins["study_commit"]) is not None,
            "resume_cell_study_commit_invalid",
        )
        pilot.require(
            source_pins.get("robolab_commit") == pilot.ROBOLAB_COMMIT
            and source_pins.get("cosmos_commit") == pilot.COSMOS_COMMIT,
            "resume_cell_source_pin_mismatch",
        )
        pilot.require(
            receipt.get("checkpoint_pin")
            == {
                "revision": pilot.CHECKPOINT_REVISION,
                "aggregate_sha256": pilot.CHECKPOINT_AGGREGATE_SHA256,
            },
            "resume_cell_checkpoint_pin_mismatch",
        )
        validated_study_commit = source_pins["study_commit"]
    else:
        pilot.require(phase == "pilot", "legacy_p00_profile_used_outside_pilot")
        pilot.require(
            legacy_p00_source_commit == LEGACY_P00_SOURCE_COMMIT,
            "legacy_p00_source_commit_changed",
        )
        pilot.require(
            all(
                key not in receipt
                for key in ("source_pins", "checkpoint_pin", "server_begin_receipt")
            ),
            "legacy_p00_receipt_shape_changed",
        )
        # The aggregate commit is authenticated by the exact released
        # aggregate hash.  Validate it against the completion's existing
        # identity below; do not manufacture the absent source_pins field.
        validated_study_commit = legacy_p00_source_commit
    for key in ("adapter_journal", "native_timing_support", "viewport_video"):
        _verify_descriptor(receipt.get(key), f"resume_{phase}_{key}")
    completion_identity = _verify_descriptor(
        receipt.get("adapter_completion"), f"resume_{phase}_adapter_completion"
    )
    completion = pilot.load_json(
        Path(completion_identity["path"]), "resume_adapter_completion_unreadable"
    )
    expected_completion = {
        "schema_version": "wmf-forecast-recording-attempt-v1",
        "study_id": pilot.STUDY_ID,
        "stop_reason": "action_cap",
        "behavioral_result_valid": True,
        "recording_qualification_valid": False,
        "model_attached": True,
        "right_censored": False,
        "technical_invalid": False,
        "actions_executed": pilot.ACTION_CAP,
        "action_cap": pilot.ACTION_CAP,
        "observation_count": pilot.OBSERVATION_COUNT,
        "request_count": pilot.REQUEST_COUNT,
        "success_configured_as_termination": False,
        "runner_reset_calls": 2,
        "physical_reset_calls": 1,
        "final_two_action_truncation_recorded": True,
        "validation_errors": [],
    }
    for key, wanted in expected_completion.items():
        pilot.require(
            completion.get(key) == wanted,
            "resume_adapter_completion_mismatch",
            key,
        )
    completion_identity_fields = completion.get("identity")
    pilot.require(
        isinstance(completion_identity_fields, Mapping),
        "resume_adapter_identity_missing",
    )
    adapter_source_identity = completion_identity_fields.get("source_identity")
    pilot.require(
        isinstance(adapter_source_identity, str)
        and adapter_source_identity.count(";pose:") == 1,
        "resume_adapter_source_identity_invalid",
    )
    pose_manifest_sha256 = adapter_source_identity.rsplit(";pose:", 1)[-1]
    pilot.require(
        pilot.SHA256_RE.fullmatch(pose_manifest_sha256) is not None,
        "resume_adapter_pose_sha256_invalid",
    )
    expected_adapter_identity = {
        "study_id": pilot.STUDY_ID,
        "cell_id": cell_ids[condition_index],
        "stage": phase,
        "layout_pair_id": layout_pair_id,
        "layout_arm": arm,
        "command": command,
        "prompt": pilot.PROMPTS[command],
        "model_config": pilot.MODEL_CONFIG,
        "effective_seed": effective_seed,
        "source_identity": (
            f"study:{validated_study_commit};robolab:{pilot.ROBOLAB_COMMIT};"
            f"cosmos:{pilot.COSMOS_COMMIT};pose:{pose_manifest_sha256}"
        ),
        "checkpoint_identity": (
            f"revision:{pilot.CHECKPOINT_REVISION};"
            f"aggregate:{pilot.CHECKPOINT_AGGREGATE_SHA256}"
        ),
    }
    for key, wanted in expected_adapter_identity.items():
        pilot.require(
            completion_identity_fields.get(key) == wanted,
            "resume_adapter_identity_mismatch",
            key,
        )
    if legacy_p00_source_commit is None:
        begin = receipt.get("server_begin_receipt")
    else:
        context_descriptor = completion.get("context_reset_artifact")
        pilot.require(
            isinstance(context_descriptor, Mapping)
            and context_descriptor.get("role") == "context_reset",
            "legacy_p00_context_reset_artifact_missing",
        )
        try:
            begin = _load_legacy_zero_array_payload(context_descriptor)
        except BaseException as error:
            raise pilot.N3BehavioralPilotError(
                "legacy_p00_context_reset_artifact_invalid"
            ) from error
    pilot.require(isinstance(begin, Mapping) and begin.get("passed") is True, "resume_begin_receipt_invalid")
    pilot.require(
        begin.get("reset_scope") == pilot.CONTEXT_RESET_SCOPE
        and begin.get("cell_id") == cell_ids[condition_index]
        and begin.get("condition_index") == condition_index
        and begin.get("effective_seed") == effective_seed,
        "resume_begin_receipt_identity_mismatch",
    )
    context_id = begin.get("server_context_id")
    pilot.require(isinstance(context_id, str) and context_id, "resume_context_id_missing")
    reset = begin.get("cache_reset_evidence")
    pilot.require(
        isinstance(reset, Mapping)
        and reset.get("passed") is True
        and reset.get("episode_context_id") == context_id
        and reset.get("unresolved_mutable_temporal_fields") == [],
        "resume_temporal_reset_evidence_invalid",
    )
    end = receipt.get("server_end_receipt")
    pilot.require(
        isinstance(end, Mapping)
        and end.get("passed") is True
        and end.get("status") == "completed"
        and end.get("cell_id") == cell_ids[condition_index]
        and end.get("condition_index") == condition_index
        and end.get("server_context_id") == context_id
        and end.get("server_request_count") == pilot.REQUEST_COUNT
        and end.get("client_request_count") == pilot.REQUEST_COUNT
        and end.get("actions_executed") == pilot.ACTION_CAP,
        "resume_server_end_receipt_invalid",
    )
    return receipt, pilot.file_identity(path)


def validate_cells_bind_fixture(
    receipts: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    accepted_gate_record_sha256: str,
    pose_manifest_sha256: str,
) -> None:
    """Bind reusable cell evidence to the one fixture release selected now."""

    pilot.require(
        isinstance(candidate_id, str) and candidate_id,
        "fixture_binding_candidate_id_invalid",
    )
    pilot.require(
        isinstance(accepted_gate_record_sha256, str)
        and pilot.SHA256_RE.fullmatch(accepted_gate_record_sha256) is not None,
        "fixture_binding_gate_record_sha256_invalid",
    )
    pilot.require(
        isinstance(pose_manifest_sha256, str)
        and pilot.SHA256_RE.fullmatch(pose_manifest_sha256) is not None,
        "fixture_binding_pose_sha256_invalid",
    )
    for receipt in receipts:
        pilot.require(
            receipt.get("candidate_id") == candidate_id,
            "cell_fixture_candidate_mismatch",
            str(receipt.get("cell_id")),
        )
        pilot.require(
            receipt.get("accepted_gate_record_sha256")
            == accepted_gate_record_sha256,
            "cell_fixture_gate_record_mismatch",
            str(receipt.get("cell_id")),
        )
        completion_identity = _verify_descriptor(
            receipt.get("adapter_completion"), "cell_fixture_adapter_completion"
        )
        completion = pilot.load_json(
            Path(completion_identity["path"]), "cell_fixture_adapter_completion_unreadable"
        )
        adapter_identity = completion.get("identity")
        pilot.require(
            isinstance(adapter_identity, Mapping), "cell_fixture_adapter_identity_missing"
        )
        source_identity = adapter_identity.get("source_identity")
        pilot.require(
            isinstance(source_identity, str)
            and source_identity.endswith(f";pose:{pose_manifest_sha256}"),
            "cell_fixture_pose_manifest_mismatch",
            str(receipt.get("cell_id")),
        )


def _validate_n3_qualification(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": pilot.N3_QUALIFICATION_SCHEMA,
        "status": "passed",
        "qualified": True,
        "model_config": "N3",
        "effective_seed": P00_EFFECTIVE_SEED,
        "generation_request_count": 6,
        "robot_episode_count": 0,
    }
    for key, wanted in expected.items():
        pilot.require(value.get(key) == wanted, "pilot_n3_qualification_mismatch", key)
    source = value.get("source")
    checkpoint = value.get("checkpoint")
    pilot.require(
        isinstance(source, Mapping)
        and source.get("commit") == pilot.COSMOS_COMMIT
        and source.get("git_tree") == pilot.COSMOS_GIT_TREE,
        "pilot_n3_source_identity_changed",
    )
    pilot.require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("revision") == pilot.CHECKPOINT_REVISION
        and checkpoint.get("payload_aggregate_sha256") == pilot.CHECKPOINT_AGGREGATE_SHA256,
        "pilot_n3_checkpoint_identity_changed",
    )


def verify_passed_p00_pilot(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate the completed four-cell pilot and its mapping prerequisites."""

    identity = pilot.verify_exact_file(path, expected_sha256, "p00_pilot_receipt")
    receipt = pilot.load_json(Path(identity["path"]), "p00_pilot_receipt_unreadable")
    expected = {
        "schema_version": PILOT_QUEUE_RECEIPT_SCHEMA,
        "status": "passed",
        "exit_code": 0,
        "study_id": pilot.STUDY_ID,
        "namespace": pilot.NAMESPACE,
        "block_id": P00_BLOCK_ID,
        "phase": "pilot",
        "layout_pair_id": "P00",
        "model_config": "N3",
        "effective_seed": P00_EFFECTIVE_SEED,
        "condition_order": list(P00_CONDITION_ORDER),
        "cell_ids": list(P00_CELL_IDS),
    }
    for key, wanted in expected.items():
        pilot.require(receipt.get(key) == wanted, "p00_pilot_receipt_mismatch", key)
    pilot.require(
        isinstance(receipt.get("source_commit"), str)
        and pilot.COMMIT_RE.fullmatch(receipt["source_commit"]) is not None,
        "p00_pilot_source_commit_invalid",
    )
    legacy_profile = receipt["source_commit"] == LEGACY_P00_SOURCE_COMMIT
    if legacy_profile:
        pilot.require(
            identity["sha256"] == LEGACY_P00_AGGREGATE_SHA256,
            "p00_legacy_aggregate_sha256_changed",
        )
    counts = receipt.get("counts")
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
    pilot.require(isinstance(counts, Mapping), "p00_pilot_counts_missing")
    for key, wanted in expected_counts.items():
        pilot.require(counts.get(key) == wanted, "p00_pilot_counts_mismatch", key)

    conditions = tuple(_condition_tuple(label) for label in P00_CONDITION_ORDER)
    descriptors = receipt.get("cell_receipts")
    pilot.require(isinstance(descriptors, list) and len(descriptors) == 4, "p00_pilot_cell_inventory_invalid")
    if legacy_profile:
        pilot.require(
            tuple(row.get("sha256") for row in descriptors)
            == LEGACY_P00_CELL_RECEIPT_SHA256S,
            "p00_legacy_cell_receipt_inventory_changed",
        )
    contexts: list[str] = []
    cell_values: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for index, descriptor in enumerate(descriptors):
        observed = _verify_descriptor(descriptor, f"p00_pilot_cell_{index}")
        cell, cell_identity = _validate_cell_receipt(
            Path(observed["path"]),
            phase="pilot",
            layout_pair_id="P00",
            block_id=P00_BLOCK_ID,
            effective_seed=P00_EFFECTIVE_SEED,
            conditions=conditions,
            cell_ids=P00_CELL_IDS,
            schema_version=PILOT_CELL_RECEIPT_SCHEMA,
            condition_index=index,
            legacy_p00_source_commit=(
                LEGACY_P00_SOURCE_COMMIT if legacy_profile else None
            ),
        )
        pilot.require(cell_identity == observed, "p00_pilot_cell_descriptor_changed")
        if not legacy_profile:
            pilot.require(
                cell["source_pins"]["study_commit"] == receipt["source_commit"],
                "p00_pilot_cell_source_commit_changed",
            )
        contexts.append(cell["server_end_receipt"]["server_context_id"])
        cell_values.append(cell)
        cells.append(observed)
    pilot.require(len(contexts) == len(set(contexts)), "p00_pilot_context_id_reused")

    prerequisites = receipt.get("prerequisites")
    pilot.require(isinstance(prerequisites, Mapping), "p00_pilot_prerequisites_missing")
    required_descriptors = (
        "gate_receipt",
        "pose_manifest",
        "capture_receipt",
        "recorder_receipt",
        "recorder_child_receipt",
        "native_timing_support",
        "n3_qualification_receipt",
    )
    prerequisite_identities = {
        key: _verify_descriptor(prerequisites.get(key), f"p00_pilot_{key}")
        for key in required_descriptors
    }
    pilot.require(
        isinstance(prerequisites.get("candidate_id"), str)
        and prerequisites["candidate_id"].startswith("P00__candidate_"),
        "p00_pilot_candidate_id_invalid",
    )
    pilot.require(
        isinstance(prerequisites.get("candidate_payload_sha256"), str)
        and pilot.SHA256_RE.fullmatch(prerequisites["candidate_payload_sha256"])
        is not None,
        "p00_pilot_candidate_payload_sha256_invalid",
    )
    pilot.require(
        isinstance(prerequisites.get("accepted_gate_record_sha256"), str)
        and pilot.SHA256_RE.fullmatch(
            prerequisites["accepted_gate_record_sha256"]
        )
        is not None,
        "p00_pilot_gate_record_sha256_invalid",
    )
    validate_cells_bind_fixture(
        cell_values,
        candidate_id=prerequisites["candidate_id"],
        accepted_gate_record_sha256=prerequisites[
            "accepted_gate_record_sha256"
        ],
        pose_manifest_sha256=prerequisite_identities["pose_manifest"]["sha256"],
    )
    qualification = pilot.load_json(
        Path(prerequisite_identities["n3_qualification_receipt"]["path"]),
        "p00_n3_qualification_unreadable",
    )
    _validate_n3_qualification(qualification)
    capture = pilot.load_json(
        Path(prerequisite_identities["capture_receipt"]["path"]),
        "p00_capture_receipt_unreadable",
    )
    pilot.require(
        capture.get("schema_version") == pilot.CAPTURE_RECEIPT_SCHEMA
        and capture.get("status") == "passed"
        and capture.get("layout_pair_id") == "P00"
        and capture.get("environment_seed") == P00_EFFECTIVE_SEED
        and capture.get("model_request_count") == 0
        and capture.get("behavioral_action_count") == 0,
        "p00_capture_receipt_invalid",
    )
    recorder = pilot.load_json(
        Path(prerequisite_identities["recorder_receipt"]["path"]),
        "p00_recorder_receipt_unreadable",
    )
    pilot.require(
        recorder.get("schema_version") == pilot.RECORDER_RECEIPT_SCHEMA
        and recorder.get("status") == "passed"
        and recorder.get("exit_code") == 0
        and recorder.get("layout_pair_id") == "P00"
        and recorder.get("environment_seed") == P00_EFFECTIVE_SEED
        and recorder.get("actions_executed") == pilot.ACTION_CAP
        and recorder.get("observation_count") == pilot.OBSERVATION_COUNT
        and recorder.get("model_request_count") == 0
        and recorder.get("behavioral_episode_count") == 0,
        "p00_recorder_receipt_invalid",
    )
    pilot.require(
        prerequisites.get("generation_qualification_requests_reused_not_rerun") == 6,
        "p00_pilot_qualification_reuse_count_changed",
    )
    topology_identity = _verify_descriptor(receipt.get("topology"), "p00_pilot_topology")
    topology = pilot.load_json(
        Path(topology_identity["path"]), "p00_pilot_topology_unreadable"
    )
    devices = topology.get("devices")
    pilot.require(
        topology.get("schema_version") == pilot.TOPOLOGY_SCHEMA
        and topology.get("status") == "passed"
        and topology.get("preexisting_compute_process_count") == 0
        and isinstance(devices, list)
        and len(devices) == 2
        and all(isinstance(row, Mapping) and row.get("name") == "NVIDIA B200" for row in devices),
        "p00_pilot_topology_invalid",
    )
    ready_identity = _verify_descriptor(receipt.get("server_ready"), "p00_pilot_server_ready")
    if legacy_profile:
        pilot.require(
            ready_identity["sha256"] == LEGACY_P00_SERVER_READY_SHA256,
            "p00_legacy_server_ready_changed",
        )
    ready = pilot.load_json(
        Path(ready_identity["path"]), "p00_pilot_server_ready_unreadable"
    )
    ready_request_count = (
        ready.get("expected_behavioral_request_count")
        if legacy_profile
        else ready.get("planned_block_behavioral_request_count")
    )
    pilot.require(
        ready.get("schema_version") == pilot.SERVER_READY_SCHEMA
        and ready.get("status") == "ready"
        and ready.get("block_id") == P00_BLOCK_ID
        and ready.get("model_config") == "N3"
        and ready.get("expected_cell_order") == list(P00_CELL_IDS)
        and ready_request_count == 60
        and ready.get("generation_qualification_requests_rerun") == 0,
        "p00_pilot_server_ready_invalid",
    )
    server_exit = receipt.get("server_exit")
    pilot.require(
        isinstance(server_exit, Mapping)
        and server_exit.get("schema_version") == pilot.SERVER_EXIT_SCHEMA
        and server_exit.get("child_reaped") is True
        and server_exit.get("terminated_by_queue_supervisor") is True,
        "p00_pilot_server_exit_invalid",
    )
    return {
        "pilot_receipt": identity,
        "pilot_cell_receipts": cells,
        "pilot_prerequisites": prerequisite_identities,
        "pilot_behavioral_cells": 4,
        "pilot_behavioral_actions": 4 * pilot.ACTION_CAP,
        "pilot_behavioral_model_requests": 4 * pilot.REQUEST_COUNT,
        "new_generation_qualification_requests": 0,
        "validation_profile": {
            "name": (
                "exact_b891_p00_attempt004_legacy_success"
                if legacy_profile
                else "strict_current_p00_receipt"
            ),
            "source_commit": receipt["source_commit"],
            "aggregate_sha256": identity["sha256"],
            "missing_fields_synthesized": False,
        },
    }


def verify_seed_audit(source_root: Path, block: DevelopmentBlock) -> dict[str, Any]:
    path = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/execution/20260912/nano_seed_audit.json"
    )
    audit = pilot.load_json(path, "nano_seed_audit_unreadable")
    pilot.require(audit.get("schema_version") == SEED_AUDIT_SCHEMA, "nano_seed_audit_schema_changed")
    pilot.require(audit.get("bounded_audit_clear") is True, "nano_seed_bounded_audit_not_clear")
    pilot.require(audit.get("seed_field_collisions") == [], "nano_seed_collision_detected")
    key = str(block.effective_seed)
    results = audit.get("candidate_results")
    pilot.require(
        isinstance(results, Mapping) and results.get(key) == "not_found_in_scanned_seed_fields",
        "nano_development_seed_not_cleared",
    )
    pilot.require(block.effective_seed in audit.get("candidate_seeds", []), "nano_development_seed_not_audited")
    return {
        **pilot.file_identity(path),
        "effective_seed": block.effective_seed,
        "bounded_collision_audit_clear": True,
        "runtime_acceptance_boundary": (
            "The committed audit checks historical seed-field collisions only. Runtime acceptance "
            "is recorded by the first official behavioral request or preserved as technical invalid."
        ),
    }


def validate_prerequisites(args: argparse.Namespace, block: DevelopmentBlock) -> dict[str, Any]:
    original = verify_development_fixture_release(
        gate_receipt_path=Path(args.gate_receipt),
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=Path(args.pose_manifest),
        pose_manifest_sha256=args.pose_manifest_sha256,
        layout_arm="original",
        block=block,
    )
    reflected = verify_development_fixture_release(
        gate_receipt_path=Path(args.gate_receipt),
        gate_receipt_sha256=args.gate_receipt_sha256,
        pose_manifest_path=Path(args.pose_manifest),
        pose_manifest_sha256=args.pose_manifest_sha256,
        layout_arm="reflected",
        block=block,
    )
    pilot.require(original["candidate_id"] == reflected["candidate_id"], "development_pose_arms_bind_different_candidates")
    passed_pilot = verify_passed_p00_pilot(Path(args.pilot_receipt), args.pilot_receipt_sha256)
    return {
        "development_gate_receipt": original["gate_receipt"],
        "development_pose_manifest": original["pose_manifest"],
        "development_gate_ledger": original["gate_ledger"],
        "development_gate_attempt_receipt": original["gate_attempt_receipt"],
        "candidate_id": original["candidate_id"],
        "candidate_payload_sha256": original["candidate_payload_sha256"],
        "accepted_gate_record_sha256": original["accepted_gate_record_sha256"],
        "p00_pilot": passed_pilot,
        "seed_audit": verify_seed_audit(Path(args.source_root), block),
        "per_development_layout_fixed_capture_required": False,
        "mapping_qualification_source": (
            "The exact passed P00 four-condition pilot transitively authenticates the P00 fixed "
            "capture and 450-action native-clock recorder qualification. Every development cell "
            "records its own original observations and native identities."
        ),
    }


def _descriptor_option(argv: Sequence[Any], option: str) -> str:
    matches = [index for index, item in enumerate(argv) if item == option]
    pilot.require(len(matches) == 1 and matches[0] + 1 < len(argv), "development_queue_option_invalid", option)
    value = argv[matches[0] + 1]
    pilot.require(isinstance(value, str), "development_queue_option_invalid", option)
    return value


def validate_queue_invocation(
    *, source_root: Path, job_dir: Path, study_commit: str, job_id: str, block: DevelopmentBlock
) -> dict[str, Any]:
    identity = pilot.validate_queue_invocation(
        source_root=source_root,
        job_dir=job_dir,
        study_commit=study_commit,
        job_id=job_id,
    )
    descriptor = pilot.load_json(Path(job_dir).resolve() / "descriptor.json", "queue_descriptor_unreadable")
    argv = descriptor.get("argv")
    pilot.require(isinstance(argv, list) and len(argv) >= 4, "development_queue_argv_invalid")
    pilot.require(argv[0] == "/usr/bin/python3", "development_queue_python_changed")
    pilot.require(
        argv[1]
        == "{source_root}/workshops/corl2026_world_models/experiments/forecast_layout/"
        + RUNNER_FILENAME,
        "development_queue_runner_changed",
    )
    pilot.require(argv[2] == "queue", "development_queue_mode_changed")
    pilot.require(_descriptor_option(argv, "--layout-pair-id") == block.layout_pair_id, "development_queue_layout_changed")
    pilot.require(_descriptor_option(argv, "--study-commit") == study_commit, "development_queue_commit_changed")
    pilot.require(_descriptor_option(argv, "--job-id") == job_id, "development_queue_job_id_changed")
    return identity


def build_server_command(
    *, source_root: Path, attempt_root: Path, port: int, study_commit: str,
    start_cell_index: int, block: DevelopmentBlock
) -> list[str]:
    pilot.require(0 <= start_cell_index < len(block.cell_ids), "invalid_start_cell_index")
    script = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout"
        / RUNNER_FILENAME
    )
    return [
        os.path.abspath(os.fspath(pilot.COSMOS_PYTHON)),
        str(script),
        "server",
        "--layout-pair-id", block.layout_pair_id,
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--port", str(port),
        "--start-cell-index", str(start_cell_index),
    ]


def build_cell_command(
    *, source_root: Path, attempt_root: Path, study_commit: str,
    gate_receipt: Path, gate_receipt_sha256: str, pose_manifest: Path,
    pose_manifest_sha256: str, port: int, condition_index: int,
    block: DevelopmentBlock,
) -> list[str]:
    pilot.require(0 <= condition_index < len(block.conditions), "cell_condition_index_invalid")
    arm, command, _task = block.conditions[condition_index]
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
        "--source-root", str(Path(source_root).resolve()),
        "--study-commit", study_commit,
        "--attempt-root", str(Path(attempt_root).resolve()),
        "--gate-receipt", str(Path(gate_receipt).resolve()),
        "--gate-receipt-sha256", gate_receipt_sha256,
        "--pose-manifest", str(Path(pose_manifest).resolve()),
        "--pose-manifest-sha256", pose_manifest_sha256,
        "--layout-arm", arm,
        "--command", command,
        "--condition-index", str(condition_index),
        "--remote-host", "127.0.0.1",
        "--remote-port", str(port),
    ]


def validate_passed_development_cell(
    path: Path, *, condition_index: int, block: DevelopmentBlock
) -> tuple[dict[str, Any], dict[str, Any]]:
    return _validate_cell_receipt(
        Path(path).resolve(),
        phase="development",
        layout_pair_id=block.layout_pair_id,
        block_id=block.block_id,
        effective_seed=block.effective_seed,
        conditions=block.conditions,
        cell_ids=block.cell_ids,
        schema_version=CELL_RECEIPT_SCHEMA,
        condition_index=condition_index,
    )


def discover_completed_prefix(
    raw_root: Path, *, block: DevelopmentBlock
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return the sole hash-verified contiguous prefix across all attempts."""

    raw_root = Path(raw_root).resolve()
    found: dict[int, tuple[dict[str, Any], dict[str, Any], Path]] = {}
    for path in sorted(raw_root.glob("*/cells/*/cell_receipt.json")):
        attempt_root = path.parents[2].resolve()
        pilot.require(not attempt_root.is_symlink(), "resume_attempt_is_symlink", str(attempt_root))
        value = pilot.load_json(path, "resume_cell_receipt_unreadable")
        index = value.get("condition_index")
        pilot.require(type(index) is int, "resume_cell_index_invalid", str(path))
        receipt, identity = validate_passed_development_cell(path, condition_index=index, block=block)
        pilot.require(index not in found, "resume_duplicate_passed_cell", block.cell_ids[index])
        pilot.require(
            not (path.parent / "technical_failure.json").exists(),
            "resume_contradictory_cell_receipts",
            block.cell_ids[index],
        )
        found[index] = (receipt, identity, attempt_root)

    valid_paths = {
        Path(identity["path"]).resolve(): index
        for index, (_receipt, identity, _attempt_root) in found.items()
    }
    for aggregate_path in sorted(raw_root.glob(f"*/publish/{AGGREGATE_FILENAME}")):
        aggregate = pilot.load_json(aggregate_path, "resume_aggregate_receipt_unreadable")
        pilot.require(aggregate.get("schema_version") == QUEUE_RECEIPT_SCHEMA, "resume_aggregate_schema_changed")
        pilot.require(aggregate.get("block_id") == block.block_id, "resume_aggregate_block_mismatch")
        pilot.require(aggregate.get("layout_pair_id") == block.layout_pair_id, "resume_aggregate_layout_mismatch")
        pilot.require(aggregate.get("model_config") == "N3", "resume_aggregate_model_mismatch")
        descriptors = aggregate.get("cell_receipts")
        pilot.require(isinstance(descriptors, list), "resume_aggregate_descriptors_invalid")
        descriptor_paths: list[Path] = []
        for offset, descriptor in enumerate(descriptors):
            observed = _verify_descriptor(descriptor, f"resume_aggregate_cell_{offset}")
            descriptor_paths.append(Path(observed["path"]).resolve())
        pilot.require(
            all(path in valid_paths for path in descriptor_paths),
            "resume_aggregate_cell_inventory_contradiction",
        )
        pilot.require(
            [valid_paths[path] for path in descriptor_paths] == list(range(len(descriptor_paths))),
            "resume_aggregate_cell_order_contradiction",
        )
        if aggregate.get("status") == "passed":
            pilot.require(len(descriptor_paths) == len(block.cell_ids), "resume_passed_aggregate_incomplete")

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
        "prior_study_commits": sorted(
            {receipt["source_pins"]["study_commit"] for receipt in receipts}
        ),
        "scan_completed_at_utc": pilot.utc_now(),
    }


def _receipt_counts(
    *, block: DevelopmentBlock, start_cell_index: int, launched: int,
    completed: Sequence[Mapping[str, Any]], attempt_root: Path,
) -> dict[str, Any]:
    completed_ids = {str(row.get("cell_id")) for row in completed}
    invalid = 0
    censored = 0
    actions = sum(int(row.get("actions_executed", 0)) for row in completed)
    requests = sum(int(row.get("behavioral_model_request_count", 0)) for row in completed)
    for index in range(start_cell_index, start_cell_index + launched):
        cell_id = block.cell_ids[index]
        if cell_id in completed_ids:
            continue
        failure_path = (
            Path(attempt_root)
            / "cells"
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
    """Resolve one immutable attempt without permitting a path escape."""

    pilot.require(pilot.SAFE_ID_RE.fullmatch(job_id) is not None, "invalid_job_id")
    root = Path(raw_root).resolve()
    attempt = (root / job_id).resolve()
    pilot.require(attempt.parent == root, "n3_development_attempt_root_escape")
    return attempt


def run_queue(args: argparse.Namespace, block: DevelopmentBlock) -> int:
    source_root = Path(args.source_root).resolve()
    job_dir = Path(args.job_dir).resolve()
    raw_root = block.raw_root if args.raw_root is None else Path(args.raw_root).resolve()
    pilot.require(raw_root == block.raw_root.resolve(), "n3_development_raw_root_changed")
    publish_path = job_dir / "publish" / AGGREGATE_FILENAME
    attempt_root = resolve_attempt_root(raw_root, args.job_id)
    completed: list[dict[str, Any]] = []
    resumed_identities: list[dict[str, Any]] = []
    new_identities: list[dict[str, Any]] = []
    resume: dict[str, Any] | None = None
    start_cell_index = 0
    launched = 0
    server_process: subprocess.Popen[bytes] | None = None
    server_stdout = None
    server_stderr = None
    queue_identity: dict[str, Any] | None = None
    prerequisites: dict[str, Any] | None = None
    topology: dict[str, Any] | None = None
    server_ready: dict[str, Any] | None = None
    server_exit: dict[str, Any] | None = None
    failure: BaseException | None = None

    raw_root.mkdir(parents=True, exist_ok=True)
    pilot.require(not raw_root.is_symlink(), "n3_development_raw_root_is_symlink")
    lock_path = raw_root / ".locks" / f"{block.block_id}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise pilot.N3BehavioralPilotError("n3_development_block_lock_is_held") from error
        try:
            completed, resumed_identities, resume = discover_completed_prefix(raw_root, block=block)
            start_cell_index = len(completed)
            pilot.require(
                start_cell_index < len(block.cell_ids),
                "valid_n3_development_block_already_exists",
                resumed_identities[-1]["path"] if resumed_identities else None,
            )
            pilot.require(not attempt_root.exists(), "n3_development_attempt_directory_already_exists")
            attempt_root.mkdir(parents=True)
            (attempt_root / "publish").mkdir()
            pilot.immutable_json(attempt_root / "resume.json", resume)
            with pilot.installed_signal_handlers():
                try:
                    queue_identity = validate_queue_invocation(
                        source_root=source_root,
                        job_dir=job_dir,
                        study_commit=args.study_commit,
                        job_id=args.job_id,
                        block=block,
                    )
                    prerequisites = validate_prerequisites(args, block)
                    validate_cells_bind_fixture(
                        completed,
                        candidate_id=prerequisites["candidate_id"],
                        accepted_gate_record_sha256=prerequisites[
                            "accepted_gate_record_sha256"
                        ],
                        pose_manifest_sha256=prerequisites[
                            "development_pose_manifest"
                        ]["sha256"],
                    )
                    topology = pilot.verify_two_idle_b200s()
                    pilot.immutable_json(attempt_root / "topology.json", topology)
                    server_command = build_server_command(
                        source_root=source_root,
                        attempt_root=attempt_root,
                        port=args.port,
                        study_commit=args.study_commit,
                        start_cell_index=start_cell_index,
                        block=block,
                    )
                    server_environment = pilot.build_model_environment(
                        source_root=source_root, attempt_root=attempt_root
                    )
                    server_process, server_stdout, server_stderr = pilot._launch_logged(
                        server_command,
                        cwd=pilot.COSMOS_ROOT,
                        environment=server_environment,
                        stdout_path=attempt_root / "server" / "stdout.log",
                        stderr_path=attempt_root / "server" / "stderr.log",
                    )
                    pilot.immutable_json(
                        attempt_root / "server" / "process.json",
                        {
                            "schema_version": "wmf-n3-behavioral-development-server-process-v1",
                            "block_id": block.block_id,
                            "layout_pair_id": block.layout_pair_id,
                            "command": server_command,
                            "environment_contract": {
                                key: server_environment.get(key)
                                for key in (
                                    "CUDA_VISIBLE_DEVICES", "DS_IGNORE_CUDA_DETECTION", "HF_HOME",
                                    "PATH", "LD_LIBRARY_PATH", "PYTHONPATH",
                                    "TORCHINDUCTOR_CACHE_DIR", "TMPDIR",
                                )
                            },
                            "process": pilot._proc_identity(server_process),
                            "started_at_utc": pilot.utc_now(),
                        },
                    )
                    server_ready = pilot._wait_for_server(
                        server_process,
                        attempt_root / "server" / "ready.json",
                        port=args.port,
                        timeout=args.server_ready_timeout,
                    )

                    for condition_index in range(start_cell_index, len(block.conditions)):
                        pilot.require(server_process.poll() is None, "n3_server_exited_between_cells")
                        cell_command = build_cell_command(
                            source_root=source_root,
                            attempt_root=attempt_root,
                            study_commit=args.study_commit,
                            gate_receipt=Path(args.gate_receipt),
                            gate_receipt_sha256=args.gate_receipt_sha256,
                            pose_manifest=Path(args.pose_manifest),
                            pose_manifest_sha256=args.pose_manifest_sha256,
                            port=args.port,
                            condition_index=condition_index,
                            block=block,
                        )
                        cell_id = block.cell_ids[condition_index]
                        log_root = attempt_root / "cell_logs" / (
                            f"{condition_index:02d}-{pilot.safe_cell_component(cell_id)}"
                        )
                        simulator_environment = pilot.build_simulator_environment(
                            source_root=source_root, state_parent=attempt_root
                        )
                        with pilot.supervised_logged_child(
                            cell_command,
                            cwd=pilot.ROBOLAB_ROOT,
                            environment=simulator_environment,
                            stdout_path=log_root / "stdout.log",
                            stderr_path=log_root / "stderr.log",
                        ) as (child, _stdout, _stderr):
                            launched += 1
                            pilot.immutable_json(
                                log_root / "process.json",
                                {
                                    "schema_version": "wmf-n3-behavioral-development-cell-process-v1",
                                    "block_id": block.block_id,
                                    "cell_id": cell_id,
                                    "condition_index": condition_index,
                                    "command": cell_command,
                                    "cuda_visible_devices": simulator_environment.get("CUDA_VISIBLE_DEVICES"),
                                    "process": pilot._proc_identity(child),
                                    "started_at_utc": pilot.utc_now(),
                                },
                            )
                            try:
                                code = child.wait(timeout=args.cell_timeout)
                            except subprocess.TimeoutExpired:
                                pilot.terminate_process_group(child)
                                raise pilot.N3BehavioralPilotError(
                                    "n3_behavioral_cell_timeout", cell_id
                                )
                            pilot.require(code == 0, "n3_behavioral_cell_failed", f"{cell_id}:{code}")
                        receipt_path = (
                            attempt_root
                            / "cells"
                            / f"{condition_index:02d}-{pilot.safe_cell_component(cell_id)}"
                            / "cell_receipt.json"
                        )
                        receipt, identity = validate_passed_development_cell(
                            receipt_path, condition_index=condition_index, block=block
                        )
                        validate_cells_bind_fixture(
                            [receipt],
                            candidate_id=prerequisites["candidate_id"],
                            accepted_gate_record_sha256=prerequisites[
                                "accepted_gate_record_sha256"
                            ],
                            pose_manifest_sha256=prerequisites[
                                "development_pose_manifest"
                            ]["sha256"],
                        )
                        completed.append(receipt)
                        new_identities.append(identity)
                    pilot.require(len(completed) == 4, "n3_development_block_incomplete")
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
                            pilot.immutable_json(
                                attempt_root / "server" / "supervisor_exit.json", server_exit
                            )
                    if server_stdout is not None and server_stderr is not None:
                        pilot._close_process_logs(server_stdout, server_stderr)
                    remaining_processes = pilot._wait_for_gpu_cleanup()
                    pilot.immutable_json(
                        attempt_root / "cleanup.json",
                        {
                            "schema_version": "wmf-n3-behavioral-development-cleanup-v1",
                            "all_children_reaped": not pilot._ACTIVE_CHILDREN,
                            "remaining_compute_processes": remaining_processes,
                            "completed_at_utc": pilot.utc_now(),
                        },
                    )
                    if remaining_processes and failure is None:
                        failure = pilot.N3BehavioralPilotError(
                            "gpu_process_remained_after_cleanup"
                        )

            counts = _receipt_counts(
                block=block,
                start_cell_index=start_cell_index,
                launched=launched,
                completed=completed,
                attempt_root=attempt_root,
            )
            status = "passed" if failure is None else "technical_failure"
            receipt = {
                "schema_version": QUEUE_RECEIPT_SCHEMA,
                "status": status,
                "exit_code": 0 if failure is None else 1,
                "study_id": pilot.STUDY_ID,
                "namespace": pilot.NAMESPACE,
                "block_id": block.block_id,
                "phase": "development",
                "layout_pair_id": block.layout_pair_id,
                "model_config": "N3",
                "effective_seed": block.effective_seed,
                "condition_order": list(block.condition_order),
                "cell_ids": list(block.cell_ids),
                "counts": counts,
                "source_commit": args.study_commit,
                "queue_descriptor": queue_identity,
                "schedule": {
                    "path": str(block.schedule_path),
                    "sha256": block.schedule_sha256,
                },
                "prerequisites": prerequisites,
                "topology": (
                    pilot.file_identity(attempt_root / "topology.json")
                    if topology is not None else None
                ),
                "server_ready": (
                    pilot.file_identity(attempt_root / "server" / "ready.json")
                    if server_ready is not None else None
                ),
                "server_exit": server_exit,
                "resume": pilot.file_identity(attempt_root / "resume.json"),
                "resumed_cell_receipts": resumed_identities,
                "new_cell_receipts": new_identities,
                "cell_receipts": resumed_identities + new_identities,
                "raw_attempt_root": str(attempt_root),
                "raw_attempt_recoverable_on_gm_pvc": True,
                "failure": (
                    None
                    if failure is None
                    else {
                        "error_type": type(failure).__name__,
                        "reason": getattr(failure, "reason", None),
                        "detail": str(failure),
                    }
                ),
                "completed_at_utc": pilot.utc_now(),
                "claim_boundary": (
                    "This receipt counts only valid 450-action N3 development cells. The passed "
                    "P00 pilot and its fixed-capture, recorder-only, and six-request generation "
                    "qualification prerequisites are reused and never counted again."
                ),
            }
            pilot.immutable_json(
                attempt_root / "publish" / AGGREGATE_FILENAME, receipt, publish=True
            )
            publish_path.parent.mkdir(parents=True, exist_ok=True)
            pilot.immutable_json(publish_path, receipt, publish=True)
            print(
                json.dumps(
                    {"status": status, "counts": counts, "raw_attempt_root": str(attempt_root)},
                    sort_keys=True,
                ),
                flush=True,
            )
            return 0 if failure is None else 1
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def run_server(args: argparse.Namespace, block: DevelopmentBlock) -> int:
    del block
    return pilot.run_server(args)


def run_cell(args: argparse.Namespace, block: DevelopmentBlock) -> int:
    """Use the shared cell implementation with a D-layout gate verifier."""

    forecast = Path(args.source_root).resolve() / "workshops/corl2026_world_models/experiments/forecast_layout"
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import recorder_qualification_job as recorder_job

    old_verifier = recorder_job.verify_fixture_release
    old_immutable = pilot.immutable_json
    verify_development_fixture_release._active_block = block  # type: ignore[attr-defined]

    def development_immutable_json(
        path: Path, value: Mapping[str, Any], *, publish: bool = False
    ) -> None:
        updated = dict(value)
        if (
            Path(path).name == "cell_receipt.json"
            and updated.get("schema_version") == CELL_RECEIPT_SCHEMA
        ):
            updated["claim_boundary"] = (
                "One valid learned-policy behavioral development cell: 450 actual actions, "
                "451 original observations, and 15 jointly generated action/future requests."
            )
            updated["transport_contract"] = dict(NO_REPLAY_TRANSPORT_CONTRACT)
        old_immutable(path, updated, publish=publish)

    recorder_job.verify_fixture_release = verify_development_fixture_release
    pilot.immutable_json = development_immutable_json
    try:
        with installed_no_replay_cosmos_client():
            return pilot.run_cell(args)
    finally:
        recorder_job.verify_fixture_release = old_verifier
        pilot.immutable_json = old_immutable
        delattr(verify_development_fixture_release, "_active_block")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    queue = subparsers.add_parser("queue", help="supervise one exact four-cell development block")
    queue.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    queue.add_argument("--source-root", type=Path, required=True)
    queue.add_argument("--study-commit", required=True)
    queue.add_argument("--job-dir", type=Path, required=True)
    queue.add_argument("--job-id", required=True)
    queue.add_argument("--raw-root", type=Path)
    queue.add_argument("--gate-receipt", type=Path, required=True)
    queue.add_argument("--gate-receipt-sha256", required=True)
    queue.add_argument("--pose-manifest", type=Path, required=True)
    queue.add_argument("--pose-manifest-sha256", required=True)
    queue.add_argument("--pilot-receipt", type=Path, required=True)
    queue.add_argument("--pilot-receipt-sha256", required=True)
    queue.add_argument("--port", type=int, default=pilot.DEFAULT_PORT)
    queue.add_argument("--server-ready-timeout", type=float, default=7200.0)
    queue.add_argument("--cell-timeout", type=float, default=10800.0)

    server = subparsers.add_parser("server", help="run the block-bound official N3 server")
    server.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    server.add_argument("--source-root", type=Path, required=True)
    server.add_argument("--study-commit", required=True)
    server.add_argument("--attempt-root", type=Path, required=True)
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=pilot.DEFAULT_PORT)
    server.add_argument("--start-cell-index", type=int, default=0)

    cell = subparsers.add_parser("cell", help="run one fresh ordered development cell")
    cell.add_argument("--layout-pair-id", choices=DEVELOPMENT_LAYOUT_IDS, required=True)
    cell.add_argument("--source-root", type=Path, required=True)
    cell.add_argument("--study-commit", required=True)
    cell.add_argument("--attempt-root", type=Path, required=True)
    cell.add_argument("--gate-receipt", type=Path, required=True)
    cell.add_argument("--gate-receipt-sha256", required=True)
    cell.add_argument("--pose-manifest", type=Path, required=True)
    cell.add_argument("--pose-manifest-sha256", required=True)
    cell.add_argument("--layout-arm", choices=("original", "reflected"), required=True)
    cell.add_argument("--command", choices=("left", "right"), required=True)
    cell.add_argument("--condition-index", type=int, required=True)
    cell.add_argument("--remote-host", default="127.0.0.1")
    cell.add_argument("--remote-port", type=int, default=pilot.DEFAULT_PORT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    block = load_development_block(Path(args.source_root), args.layout_pair_id)
    port = getattr(args, "port", getattr(args, "remote_port", 0))
    pilot.require(1 <= port <= 65535, "invalid_port")
    with configured_pilot(block):
        if args.mode == "queue":
            pilot.require(
                args.server_ready_timeout > 0 and args.cell_timeout > 0,
                "invalid_timeout",
            )
            return run_queue(args, block)
        if args.mode == "server":
            pilot.require(0 <= args.start_cell_index < 4, "invalid_start_cell_index")
            return run_server(args, block)
        if args.mode == "cell":
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
