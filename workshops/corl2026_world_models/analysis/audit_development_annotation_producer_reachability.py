#!/usr/bin/env python3
"""Prove the pinned authorized annotation activation chain is unreachable.

This audit reads six commit-addressed Git blobs and parses Python with ``ast``.
It never imports or invokes a producer, opens a raw/PVC path, contacts a remote,
or creates a queue descriptor.  Its positive claim is deliberately negative and
narrow: the retained camera attempt is diagnostic-only and its authorized queue
entrypoint cannot terminally publish its two prospective crop contracts, while
the retained physical-alignment producer requires a later passed camera attempt
before it can publish its four outputs.  Diagnostic raw outputs may still exist.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[3]

SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
ACTIVATION_COMMIT = "9122bbf8c6b8cbdcaba5d9eda345b6c518dde8e3"
CAMERA_COMMIT = "93993dae1902ba561b594b9ae0849bd45b37ecc2"
ALIGNMENT_COMMIT = "a38b73ca539fe3cc7618fb1294b67b0e24861b58"

SPEC_PATH = "workshops/corl2026_world_models/docs/ABLATION_SPEC.md"
ACTIVATION_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_activation_conformance.json"
)
CAMERA_JOBS_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "camera_crop_witness_jobs.py"
)
CAMERA_CONTRACT_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "camera_crop_witness_contract.json"
)
ALIGNMENT_JOBS_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_physical_alignment_jobs.py"
)
ALIGNMENT_CONTRACT_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_physical_alignment_contract.json"
)

EXPECTED_SOURCES: dict[tuple[str, str], tuple[int, str]] = {
    (SPEC_COMMIT, SPEC_PATH): (
        27094,
        "fa8cbbac7bf2eff2e01f73150e5cf073f7e4f177d3c0bfd4fc1f41fb3ea9309f",
    ),
    (ACTIVATION_COMMIT, ACTIVATION_PATH): (
        34824,
        "ac0ed739b2e8026999ba318411596697a102efb5bd551ef8b96bc5822a9b4ebd",
    ),
    (CAMERA_COMMIT, CAMERA_JOBS_PATH): (
        51025,
        "6613de67a170d1719d5b4e1a5be1e2b9305e1a4b6087e8f1d2006f3e8b0f3145",
    ),
    (CAMERA_COMMIT, CAMERA_CONTRACT_PATH): (
        9862,
        "31313288aac8cd917707d52eea8c4fca9ee77de1d3c0b42ade466aa78449706a",
    ),
    (ALIGNMENT_COMMIT, ALIGNMENT_JOBS_PATH): (
        90462,
        "b805df1e251ca848a82a7ff0c39043d40154ce1030dd7db2e68b30b1ed09c44a",
    ),
    (ALIGNMENT_COMMIT, ALIGNMENT_CONTRACT_PATH): (
        2900,
        "1f2cd1f30959d50b2e7096192240cbeeb7b6fd22c71d6d599a18fcf5feb18f61",
    ),
}

MODELS = ("N3", "D1")
EXPECTED_CROP_NAMES = {
    "N3": "n3_camera_crop_contract.json",
    "D1": "d1_camera_crop_contract.json",
}
EXPECTED_ALIGNMENT_NAMES = {
    "N3": {
        "mapping": "n3_physical_alignment_receipt.json",
        "alignment": "n3_alignment_contract.json",
    },
    "D1": {
        "mapping": "d1_physical_alignment_receipt.json",
        "alignment": "d1_alignment_contract.json",
    },
}
EXPECTED_ACTIVATION_SLOTS = {
    f"{model}.camera_crop_contract": {
        "model_id": model,
        "field": "camera_crop_contract",
        "output_name": EXPECTED_CROP_NAMES[model],
        "producer_commit": CAMERA_COMMIT,
        "producer_path": CAMERA_JOBS_PATH,
        "producer_constant": "MODEL_CONTRACT_NAMES",
    }
    for model in MODELS
}
EXPECTED_ACTIVATION_SLOTS.update({
    f"{model}.{field}": {
        "model_id": model,
        "field": field,
        "output_name": EXPECTED_ALIGNMENT_NAMES[model][role],
        "producer_commit": ALIGNMENT_COMMIT,
        "producer_path": ALIGNMENT_JOBS_PATH,
        "producer_constant": "OUTPUT_NAMES",
    }
    for model in MODELS
    for field, role in (
        ("physical_alignment_receipt", "mapping"),
        ("alignment_contract", "alignment"),
    )
})
EXPECTED_FUNCTIONS: dict[tuple[str, str], tuple[int, int, int, str]] = {
    (CAMERA_JOBS_PATH, "build_wave"): (
        300, 333, 1452,
        "93092f9a95ec69bf63bb619aea92cef504c36b02b1b2df2471cf3d8f6138af70",
    ),
    (CAMERA_JOBS_PATH, "_publish_contracts"): (
        1068, 1098, 1285,
        "e9a64dd3690821adbd75e6e80e36f798f38c8df40132c95ddf455f94ddfb1ca8",
    ),
    (CAMERA_JOBS_PATH, "run_job"): (
        1186, 1226, 1897,
        "2081a6e5a246a0fc9f3fa5a10d8ea7a50c47e054584229e63b7f9e8d36e28d3c",
    ),
    (ALIGNMENT_JOBS_PATH, "_camera_attempt"): (
        677, 686, 435,
        "1d19422bf045d545d14b8739a3e958ddba5c35de7d833b9ebf64b773cf7a0913",
    ),
    (ALIGNMENT_JOBS_PATH, "_validate_camera_job"): (
        689, 767, 3270,
        "b5f7911ca3cab219a30515770e425768a2358bd1e147cf133553f413530ac674",
    ),
    (ALIGNMENT_JOBS_PATH, "validate_local_prerequisites"): (
        805, 912, 4432,
        "1d5a2aae095e1f244d8455d407a9110c02181ea262c6f001cbaa7393fa4a3fc3",
    ),
    (ALIGNMENT_JOBS_PATH, "build_wave_from_local_files"): (
        1141, 1160, 1098,
        "b9d49d21013acbdaccdfa8a3a8c44830204958e3f90547c47f40b4268ef00af4",
    ),
}
EXPECTED_GLOBAL_SNAPSHOT = {
    "active_launched_cells": 0,
    "confirmation_inference_cells": 0,
    "launched_core_cells": 44,
    "nonbehavioral_requests": 18,
    "planned_core_cells": 232,
    "technical_invalid_behavioral_requests": 2,
    "terminal_requests": 1460,
    "terminal_technical_invalid_cells": 4,
    "terminal_valid_cells": 40,
    "valid_behavioral_actions": 18000,
    "valid_behavioral_observations": 18040,
    "valid_behavioral_requests": 1440,
}
EXPECTED_GATE_STATE = {
    "annotation_media_ready": False,
    "camera_alignment_ready": False,
    "confirmation_release_ready": False,
    "core_episode_ceiling": 232,
    "current_GM_PVC_worker_state_reconciled": False,
    "development_annotation_gate_ready": False,
    "human_annotation_ready": False,
    "live_quiescence_inferred": False,
    "resource_freeze_ready": False,
    "retained_snapshot_active_launched_cells": 0,
    "safe_to_release_new_work": False,
}
ZERO_ACTIVITY = {
    "annotation_media": 0,
    "behavioral_actions": 0,
    "behavioral_cells": 0,
    "confirmation_inference_cells": 0,
    "forecast_accuracy_estimates": 0,
    "human_responses": 0,
    "labels": 0,
    "model_requests": 0,
    "queue_descriptors": 0,
    "request_rows": 0,
    "request_selections": 0,
    "retrieval_jobs": 0,
    "runnable_manifests": 0,
}


class AuditError(ValueError):
    """A pinned source or claimed invariant changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AuditError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _nonfinite(value: str) -> Any:
    raise AuditError(f"Non-finite JSON constant: {value}")


def load_json(data: bytes, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_nonfinite
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"Invalid JSON for {label}: {error}") from error


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def load_pinned_blobs() -> dict[tuple[str, str], bytes]:
    blobs: dict[tuple[str, str], bytes] = {}
    for commit, path in EXPECTED_SOURCES:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=REPOSITORY,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise AuditError(
                f"Cannot read pinned source {commit}:{path}: "
                f"{result.stderr.decode('utf-8', 'replace').strip()}"
            )
        blobs[(commit, path)] = result.stdout
    verify_pinned_blobs(blobs)
    return blobs


def verify_pinned_blobs(blobs: Mapping[tuple[str, str], bytes]) -> None:
    require(set(blobs) == set(EXPECTED_SOURCES), "Pinned source inventory changed")
    for key, expected in EXPECTED_SOURCES.items():
        data = blobs[key]
        actual = (len(data), hashlib.sha256(data).hexdigest())
        require(actual == expected, f"Pinned source changed: {key[0]}:{key[1]}")


def _module_assignments(source: str) -> dict[str, Any]:
    tree = ast.parse(source)
    result: dict[str, Any] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            result[target.id] = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
    return result


def _function(source: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(matches) == 1, f"Function inventory changed: {name}")
    return matches[0]


def _function_descriptor(source: str, path: str, name: str) -> dict[str, Any]:
    node = _function(source, name)
    lines = source.splitlines(keepends=True)
    data = "".join(lines[node.lineno - 1:node.end_lineno]).encode("utf-8")
    result = {
        "name": name,
        "path": path,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    expected = EXPECTED_FUNCTIONS.get((path, name))
    require(expected is not None, f"Unexpected function descriptor: {path}:{name}")
    require(
        (
            result["start_line"], result["end_line"],
            result["bytes"], result["sha256"],
        ) == expected,
        f"Function changed: {path}:{name}",
    )
    return result


def _call_names(node: ast.AST) -> list[str]:
    result: list[str] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            result.append(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            result.append(child.func.attr)
    return result


def _literal_dict_assignment(function: ast.FunctionDef, name: str) -> dict[str, Any]:
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        require(isinstance(node.value, ast.Dict),
                f"{name} is no longer a dictionary")
        value: dict[str, Any] = {}
        for key_node, value_node in zip(node.value.keys, node.value.values):
            try:
                key = ast.literal_eval(key_node)
            except (ValueError, TypeError) as error:
                raise AuditError(f"{name} gained a nonliteral key") from error
            try:
                value[key] = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                # Function calls and constant names are retained in the pinned
                # source but are outside this audit's deliberately small claim.
                continue
        return value
    raise AuditError(f"Dictionary assignment changed: {name}")


def analyze_camera_source(source: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    assignments = _module_assignments(source)
    require(assignments.get("JOB_ID") == "camera-crop-replay-witness-002",
            "Camera retained attempt changed")
    require(assignments.get("MODEL_CONTRACT_NAMES") == EXPECTED_CROP_NAMES,
            "Camera prospective output names changed")
    require(assignments.get("SUCCESS_RECEIPT") == "camera_crop_witness_job_receipt.json",
            "Camera success receipt name changed")
    require(assignments.get("FAILURE_RECEIPT") == "camera_crop_witness_job_failure.json",
            "Camera failure receipt name changed")

    require(contract.get("schema_version")
            == "wmf-camera-crop-witness-runtime-contract-v1",
            "Camera runtime contract schema changed")
    require(contract.get("job", {}).get("job_id") == assignments["JOB_ID"],
            "Camera source/contract job identity changed")
    child = contract.get("child_runtime")
    require(isinstance(child, Mapping), "Camera child runtime contract is missing")
    require(child.get("diagnostic_only") is True,
            "Camera diagnostic-only boundary changed")
    require(child.get("publish_crop_contracts") is False,
            "Camera publication boundary changed")
    require(contract.get("safe_to_release_confirmation") is False
            and contract.get("confirmation_released") is False,
            "Camera release authority changed")
    science = contract.get("science_counts")
    require(isinstance(science, Mapping) and science and set(science.values()) == {0},
            "Camera science activity changed")

    tree = ast.parse(source)
    publisher_calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_publish_contracts"
    ]
    require(not publisher_calls, "Camera publisher gained a call site")
    publisher_references = [
        node for node in ast.walk(tree)
        if (isinstance(node, ast.Name) and node.id == "_publish_contracts")
        or (isinstance(node, ast.Attribute) and node.attr == "_publish_contracts")
    ]
    require(not publisher_references, "Camera publisher gained a name reference")

    run = _function(source, "run_job")
    run_calls = _call_names(run)
    require("_launch_children" in run_calls, "Camera diagnostic child launch changed")
    require("_publish_contracts" not in run_calls,
            "Camera run_job now calls the publisher")
    direct_raises = []
    for statement in run.body:
        if not isinstance(statement, ast.Try):
            continue
        for nested in statement.body:
            if not isinstance(nested, ast.Raise) or not isinstance(nested.exc, ast.Call):
                continue
            if isinstance(nested.exc.func, ast.Name):
                direct_raises.append(nested.exc.func.id)
    require(direct_raises == ["CameraCropChildError"],
            "Camera diagnostic terminal raise changed")
    run_text = ast.unparse(run)
    require("runtime['child_runtime']['diagnostic_only'] is True" in run_text
            and "runtime['child_runtime']['publish_crop_contracts'] is False" in run_text,
            "Camera run_job diagnostic guard changed")
    require("raw contracts were retained but intentionally not published" in run_text,
            "Camera nonpublication terminal detail changed")

    wave = _function(source, "build_wave")
    wave_text = ast.unparse(wave)
    require("'technical_invalid_diagnostic_receipt': FAILURE_RECEIPT" in wave_text,
            "Camera wave expected-output boundary changed")
    require("'diagnostic_only': True" in wave_text,
            "Camera wave diagnostic flag changed")

    return {
        "retained_job_id": assignments["JOB_ID"],
        "diagnostic_only": True,
        "publish_crop_contracts": False,
        "prospective_output_names": assignments["MODEL_CONTRACT_NAMES"],
        "publisher_definition_count": sum(
            isinstance(node, ast.FunctionDef) and node.name == "_publish_contracts"
            for node in tree.body
        ),
        "publisher_call_site_count": 0,
        "publisher_name_reference_count": 0,
        "publisher_reachable_from_authorized_run_job_entrypoint": False,
        "terminal_behavior": (
            "technical_invalid_after_successful_diagnostic_children_with_raw_outputs_"
            "retained_but_not_terminally_published"
        ),
        "success_receipt_reachable_from_authorized_run_job_entrypoint": False,
        "functions": [
            _function_descriptor(source, CAMERA_JOBS_PATH, name)
            for name in ("build_wave", "_publish_contracts", "run_job")
        ],
    }


def _comparison_facts(function: ast.FunctionDef) -> tuple[bool, bool]:
    minimum = False
    forbidden = False
    for node in ast.walk(function):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        left = ast.unparse(node.left)
        right = ast.unparse(node.comparators[0])
        if left == "ordinal" and isinstance(node.ops[0], ast.GtE) and right == "3":
            minimum = True
        if (left == "job_id" and isinstance(node.ops[0], ast.NotIn)
                and right == "FORBIDDEN_CAMERA_JOB_IDS"):
            forbidden = True
    return minimum, forbidden


def analyze_alignment_source(
    source: str, contract: Mapping[str, Any]
) -> dict[str, Any]:
    assignments = _module_assignments(source)
    require(assignments.get("OUTPUT_NAMES") == EXPECTED_ALIGNMENT_NAMES,
            "Alignment prospective output names changed")
    require(assignments.get("FORBIDDEN_CAMERA_JOB_IDS")
            == {"camera-crop-replay-witness-001", "camera-crop-replay-witness-002"},
            "Alignment forbidden camera attempts changed")

    prerequisites = contract.get("prerequisites")
    require(isinstance(prerequisites, Mapping),
            "Alignment prerequisite contract is missing")
    require(prerequisites.get("camera_terminal_minimum_attempt") == 3,
            "Alignment camera minimum attempt changed")
    require(set(prerequisites.get("forbidden_camera_job_ids", ()))
            == assignments["FORBIDDEN_CAMERA_JOB_IDS"],
            "Alignment source/contract forbidden attempts changed")
    require(prerequisites.get("camera_terminal_receipt_schema")
            == "wmf-camera-crop-witness-queue-job-v1",
            "Alignment camera receipt schema changed")
    require(prerequisites.get("camera_contracts") == 2
            and prerequisites.get("camera_terminal_output_declarations") == 2,
            "Alignment camera output cardinality changed")
    require(contract.get("annotation_authority") is False
            and contract.get("confirmation_authority") is False
            and contract.get("safe_to_release_confirmation") is False,
            "Alignment authority changed")
    science = contract.get("science_counts")
    require(isinstance(science, Mapping) and science and set(science.values()) == {0},
            "Alignment science activity changed")

    camera_attempt = _function(source, "_camera_attempt")
    expected_attempt_body = (
        "require(isinstance(job_id, str), 'camera terminal job ID is missing')",
        "match = CAMERA_JOB_RE.fullmatch(job_id)",
        "require(match is not None, 'camera terminal job ID is invalid')",
        "ordinal = int(match.group(1))",
        (
            "require(ordinal >= 3 and job_id not in FORBIDDEN_CAMERA_JOB_IDS, "
            "'diagnostic/nonterminal camera attempt cannot feed alignment')"
        ),
        "return ordinal",
    )
    require(tuple(ast.unparse(statement) for statement in camera_attempt.body)
            == expected_attempt_body,
            "Alignment camera attempt control flow changed")
    minimum, forbidden = _comparison_facts(camera_attempt)
    require(minimum, "Alignment camera attempt minimum guard changed")
    require(forbidden, "Alignment camera forbidden-attempt guard changed")

    validate_camera = _function(source, "_validate_camera_job")
    require("_camera_attempt" in _call_names(validate_camera),
            "Alignment camera validator no longer applies attempt gate")
    exact = _literal_dict_assignment(validate_camera, "exact")
    require(exact.get("status") == "passed"
            and exact.get("decision") == "qualified_from_original_camera_pixels"
            and exact.get("safe_for_physical_alignment_input") is True
            and exact.get("raw_outputs_recoverable_on_gm_pvc") is True,
            "Alignment terminal camera qualification changed")
    local = _function(source, "validate_local_prerequisites")
    require("_validate_camera_job" in _call_names(local),
            "Alignment local prerequisite chain changed")
    wave = _function(source, "build_wave_from_local_files")
    require("validate_local_prerequisites" in _call_names(wave),
            "Alignment wave prerequisite chain changed")

    published_files = contract.get("release", {}).get("published_files")
    expected_published = [
        EXPECTED_ALIGNMENT_NAMES[model][role]
        for model in MODELS for role in ("mapping", "alignment")
    ] + ["development_physical_alignment_job_receipt.json"]
    require(published_files == expected_published,
            "Alignment contract published-file inventory changed")

    return {
        "camera_terminal_minimum_attempt": 3,
        "forbidden_camera_job_ids": sorted(assignments["FORBIDDEN_CAMERA_JOB_IDS"]),
        "required_terminal_status": exact["status"],
        "required_terminal_decision": exact["decision"],
        "requires_original_camera_pixels": True,
        "requires_safe_for_physical_alignment_input": True,
        "prospective_output_names": assignments["OUTPUT_NAMES"],
        "prerequisite_chain": [
            "build_wave_from_local_files",
            "validate_local_prerequisites",
            "_validate_camera_job",
            "_camera_attempt",
        ],
        "functions": [
            _function_descriptor(source, ALIGNMENT_JOBS_PATH, name)
            for name in (
                "_camera_attempt", "_validate_camera_job",
                "validate_local_prerequisites", "build_wave_from_local_files",
            )
        ],
    }


def activation_slot_bindings(value: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    matrix = value.get("runtime_slot_activation_matrix", {})
    rows = matrix.get("slots")
    require(isinstance(rows, list) and len(rows) == 14,
            "Activation slot inventory changed")
    require(len({row.get("slot_id") for row in rows if isinstance(row, Mapping)}) == 14,
            "Activation slot identities changed")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, Mapping), "Activation slot row changed")
        prospective = row.get("prospective_output")
        if prospective is None:
            continue
        slot_id = row.get("slot_id")
        require(isinstance(slot_id, str) and slot_id not in result,
                "Activation prospective slot identity changed")
        require(isinstance(prospective, Mapping),
                "Activation prospective producer changed")
        result[slot_id] = {
            "model_id": row.get("model_id"),
            "field": row.get("field"),
            "output_name": prospective.get("output_name"),
            "producer_commit": prospective.get("producer_commit"),
            "producer_path": prospective.get("producer_path"),
            "producer_constant": prospective.get("producer_constant"),
        }
        require(row.get("current_source_state")
                == "no_authenticated_descriptor_in_pinned_sources"
                and row.get("identity_bound") is False
                and row.get("prospective_output_is_not_existing_evidence") is True
                and row.get("runtime_byte_verified") is False
                and row.get("semantic_replay_complete") is False,
                f"Activation prospective slot evidence promoted: {slot_id}")
    require(result == EXPECTED_ACTIVATION_SLOTS,
            "Activation prospective slot bindings changed")
    return result


def validate_activation_report(value: Mapping[str, Any]) -> None:
    matrix = value.get("runtime_slot_activation_matrix", {})
    fields = value.get("request_field_provenance", {})
    selection = value.get("selection_contract", {})
    decision = value.get("activation_decision", {})
    require(matrix.get("required_slots") == 14
            and matrix.get("unauthenticated_slots") == 6
            and matrix.get("runtime_byte_verified_slots") == 0
            and matrix.get("semantic_replay_complete_slots") == 0,
            "Activation runtime closure counts changed")
    require(fields.get("field_count") == 30
            and fields.get("authenticated_value_field_count") == 4
            and fields.get("missing_value_field_count") == 26
            and fields.get("complete_rows") == 0
            and fields.get("total_rows") == 1152,
            "Activation request field coverage changed")
    require(selection.get("sampling_key_feasible_rows") == 0
            and selection.get("eligibility_feasible_rows") == 0
            and selection.get("observed_selected_requests") == 0,
            "Activation selection state changed")
    require(decision.get("status")
            == "not_activatable_from_authenticated_retained_evidence"
            and decision.get("safe_to_release") is False,
            "Activation decision changed")
    require(value.get("global_study_snapshot") == EXPECTED_GLOBAL_SNAPSHOT,
            "Global study snapshot changed")
    activation_slot_bindings(value)
    gate = value.get("gate_state", {})
    require(gate == EXPECTED_GATE_STATE, "Activation gate state changed")
    require(value.get("new_activity") == ZERO_ACTIVITY,
            "Activation science activity changed")


def _source_inventory(blobs: Mapping[tuple[str, str], bytes]) -> dict[str, Any]:
    rows = []
    for commit, path in sorted(blobs):
        data = blobs[(commit, path)]
        rows.append({
            "commit": commit,
            "path": path,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    framed = b"".join(
        canonical_bytes(row) for row in rows
    )
    return {
        "count": len(rows),
        "source_bytes": sum(row["bytes"] for row in rows),
        "sources": rows,
        "canonical_framed_inventory_bytes": len(framed),
        "canonical_framed_inventory_sha256": hashlib.sha256(framed).hexdigest(),
    }


def _chains(
    camera: Mapping[str, Any],
    alignment: Mapping[str, Any],
    activation_slots: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for model in MODELS:
        result.append({
            "model_id": model,
            "authorized_queue_entrypoint_terminal_activation_reachable": False,
            "nodes": [
                {
                    "slot_id": f"{model}.camera_crop_contract",
                    "output_name": activation_slots[
                        f"{model}.camera_crop_contract"
                    ]["output_name"],
                    "producer_job_id": camera["retained_job_id"],
                    "state": (
                        "diagnostic_raw_outputs_may_exist_but_authorized_entrypoint_"
                        "terminal_publication_is_unreachable"
                    ),
                    "runtime_byte_verified": False,
                    "semantic_replay_complete": False,
                },
                {
                    "slot_id": f"{model}.physical_alignment_receipt",
                    "output_name": activation_slots[
                        f"{model}.physical_alignment_receipt"
                    ]["output_name"],
                    "producer_job_id": "development-physical-alignment-001",
                    "state": (
                        "blocked_by_absent_authenticated_passed_camera_attempt_ge_003_"
                        "in_pinned_activation_evidence"
                    ),
                    "runtime_byte_verified": False,
                    "semantic_replay_complete": False,
                },
                {
                    "slot_id": f"{model}.alignment_contract",
                    "output_name": activation_slots[
                        f"{model}.alignment_contract"
                    ]["output_name"],
                    "producer_job_id": "development-physical-alignment-001",
                    "state": (
                        "blocked_by_absent_authenticated_passed_camera_attempt_ge_003_"
                        "in_pinned_activation_evidence"
                    ),
                    "runtime_byte_verified": False,
                    "semantic_replay_complete": False,
                },
            ],
        })
    return result


def build_report(blobs: Mapping[tuple[str, str], bytes]) -> dict[str, Any]:
    verify_pinned_blobs(blobs)
    activation = load_json(
        blobs[(ACTIVATION_COMMIT, ACTIVATION_PATH)], "activation report"
    )
    camera_contract = load_json(
        blobs[(CAMERA_COMMIT, CAMERA_CONTRACT_PATH)], "camera contract"
    )
    alignment_contract = load_json(
        blobs[(ALIGNMENT_COMMIT, ALIGNMENT_CONTRACT_PATH)], "alignment contract"
    )
    require(isinstance(activation, Mapping), "Activation report must be an object")
    require(isinstance(camera_contract, Mapping), "Camera contract must be an object")
    require(isinstance(alignment_contract, Mapping),
            "Alignment contract must be an object")
    validate_activation_report(activation)
    activation_slots = activation_slot_bindings(activation)
    camera = analyze_camera_source(
        blobs[(CAMERA_COMMIT, CAMERA_JOBS_PATH)].decode("utf-8"), camera_contract
    )
    alignment = analyze_alignment_source(
        blobs[(ALIGNMENT_COMMIT, ALIGNMENT_JOBS_PATH)].decode("utf-8"),
        alignment_contract,
    )
    chains = _chains(camera, alignment, activation_slots)
    report = {
        "schema_version": "wmf-development-annotation-producer-reachability-v1",
        "study_id": "WMF-ABLATION-001",
        "status": "static_authorized_terminal_activation_chain_unreachable",
        "authenticated_source_blobs": _source_inventory(blobs),
        "camera_producer": camera,
        "physical_alignment_producer": alignment,
        "activation_slot_bindings": activation_slots,
        "authorized_terminal_activation_chain": {
            "model_count": 2,
            "required_slot_count": 6,
            "authorized_queue_entrypoint_terminal_activation_reachable": False,
            "chains": chains,
            "end_to_end_activation_reachable_slot_count": 0,
            "diagnostic_only_unreachable_slot_count": 2,
            "downstream_blocked_slot_count": 4,
            "runtime_byte_verified_slot_count": 0,
            "semantic_replay_complete_slot_count": 0,
        },
        "retained_activation_snapshot": {
            "required_runtime_slots": 14,
            "unauthenticated_runtime_slots": 6,
            "runtime_byte_verified_slots": 0,
            "semantic_replay_complete_slots": 0,
            "request_field_values_authenticated": 4,
            "request_field_values_missing": 26,
            "complete_request_rows": 0,
            "total_request_rows": 1152,
            "sampling_key_feasible_rows": 0,
            "eligibility_feasible_rows": 0,
            "selected_request_rows": 0,
        },
        "global_study_snapshot": EXPECTED_GLOBAL_SNAPSHOT,
        "gate_state": EXPECTED_GATE_STATE,
        "new_activity": ZERO_ACTIVITY,
        "audit_activity": {
            "authenticated_git_blobs_read": len(blobs),
            "authenticated_git_bytes_read": sum(map(len, blobs.values())),
            "python_sources_parsed_with_ast": 2,
            "producer_modules_imported": 0,
            "producer_functions_invoked": 0,
            "external_system_reads": 0,
            "raw_paths_opened": 0,
            "runtime_bytes_read": 0,
        },
        "decision": {
            "status": (
                "pinned_authorized_terminal_chain_cannot_authenticate_and_activate_"
                "six_missing_slots"
            ),
            "authorized_queue_entrypoint_terminal_activation_reachable": False,
            "camera_attempt_003_or_later_authenticated_by_pinned_activation_evidence": False,
            "safe_to_dispatch_replacement": False,
            "safe_to_materialize_annotation_package": False,
            "safe_to_release": False,
            "blocking_reasons": [
                "retained camera attempt 002 is diagnostic-only",
                "camera publisher has zero call sites",
                "alignment rejects camera attempts below 003 and attempts 001/002",
                "no authenticated passed camera attempt 003 or later is pinned",
                "current GM PVC queue worker and live-quiescence state are unreconciled",
            ],
        },
        "claim_boundaries": {
            "positive_claim": (
                "Commit-pinned activation evidence and source prove that the authorized "
                "queue entrypoints cannot terminally authenticate and activate the six "
                "missing annotation slots through the pinned producer chain."
            ),
            "static_reachability_is_not_runtime_observation": True,
            "prospective_output_names_are_not_existing_artifacts": True,
            "diagnostic_raw_outputs_may_exist_without_terminal_publication": True,
            "publisher_helper_callable_outside_authorized_entrypoint_is_not_assessed": True,
            "retained_active_zero_is_not_live_quiescence": True,
            "not_established": [
                "current GM PVC queue worker or release-lock state",
                "current existence of any raw camera or alignment artifact",
                "camera or physical alignment qualification",
                "request inventory, eligibility, or selection",
                "annotation media, labels, or rater responses",
                "forecast accuracy or scientific treatment effect",
                "authority to dispatch camera attempt 003 or any replacement work",
            ],
        },
    }
    validate_output_contract(report)
    return report


def validate_output_contract(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version")
            == "wmf-development-annotation-producer-reachability-v1"
            and value.get("study_id") == "WMF-ABLATION-001",
            "Output identity changed")
    require(value.get("status")
            == "static_authorized_terminal_activation_chain_unreachable",
            "Reachability status changed")
    source_inventory = value.get("authenticated_source_blobs", {})
    expected_source_rows = [
        {
            "commit": commit,
            "path": path,
            "bytes": EXPECTED_SOURCES[(commit, path)][0],
            "sha256": EXPECTED_SOURCES[(commit, path)][1],
        }
        for commit, path in sorted(EXPECTED_SOURCES)
    ]
    expected_framed = b"".join(canonical_bytes(row) for row in expected_source_rows)
    require(source_inventory == {
        "count": 6,
        "source_bytes": 216167,
        "sources": expected_source_rows,
        "canonical_framed_inventory_bytes": len(expected_framed),
        "canonical_framed_inventory_sha256": hashlib.sha256(expected_framed).hexdigest(),
    }, "Output source inventory changed")

    camera = value.get("camera_producer", {})
    camera_functions = camera.get("functions")
    require(isinstance(camera_functions, list), "Camera function inventory changed")
    expected_camera_functions = [
        {
            "name": name,
            "path": CAMERA_JOBS_PATH,
            "start_line": EXPECTED_FUNCTIONS[(CAMERA_JOBS_PATH, name)][0],
            "end_line": EXPECTED_FUNCTIONS[(CAMERA_JOBS_PATH, name)][1],
            "bytes": EXPECTED_FUNCTIONS[(CAMERA_JOBS_PATH, name)][2],
            "sha256": EXPECTED_FUNCTIONS[(CAMERA_JOBS_PATH, name)][3],
        }
        for name in ("build_wave", "_publish_contracts", "run_job")
    ]
    require(camera_functions == expected_camera_functions,
            "Camera function identities changed")
    camera_without_functions = dict(camera)
    camera_without_functions.pop("functions", None)
    require(camera_without_functions == {
        "retained_job_id": "camera-crop-replay-witness-002",
        "diagnostic_only": True,
        "publish_crop_contracts": False,
        "prospective_output_names": EXPECTED_CROP_NAMES,
        "publisher_definition_count": 1,
        "publisher_call_site_count": 0,
        "publisher_name_reference_count": 0,
        "publisher_reachable_from_authorized_run_job_entrypoint": False,
        "terminal_behavior": (
            "technical_invalid_after_successful_diagnostic_children_with_raw_outputs_"
            "retained_but_not_terminally_published"
        ),
        "success_receipt_reachable_from_authorized_run_job_entrypoint": False,
    }, "Camera producer summary changed")

    alignment = value.get("physical_alignment_producer", {})
    alignment_functions = alignment.get("functions")
    require(isinstance(alignment_functions, list),
            "Alignment function inventory changed")
    expected_alignment_functions = [
        {
            "name": name,
            "path": ALIGNMENT_JOBS_PATH,
            "start_line": EXPECTED_FUNCTIONS[(ALIGNMENT_JOBS_PATH, name)][0],
            "end_line": EXPECTED_FUNCTIONS[(ALIGNMENT_JOBS_PATH, name)][1],
            "bytes": EXPECTED_FUNCTIONS[(ALIGNMENT_JOBS_PATH, name)][2],
            "sha256": EXPECTED_FUNCTIONS[(ALIGNMENT_JOBS_PATH, name)][3],
        }
        for name in (
            "_camera_attempt", "_validate_camera_job",
            "validate_local_prerequisites", "build_wave_from_local_files",
        )
    ]
    require(alignment_functions == expected_alignment_functions,
            "Alignment function identities changed")
    alignment_without_functions = dict(alignment)
    alignment_without_functions.pop("functions", None)
    require(alignment_without_functions == {
        "camera_terminal_minimum_attempt": 3,
        "forbidden_camera_job_ids": [
            "camera-crop-replay-witness-001",
            "camera-crop-replay-witness-002",
        ],
        "required_terminal_status": "passed",
        "required_terminal_decision": "qualified_from_original_camera_pixels",
        "requires_original_camera_pixels": True,
        "requires_safe_for_physical_alignment_input": True,
        "prospective_output_names": EXPECTED_ALIGNMENT_NAMES,
        "prerequisite_chain": [
            "build_wave_from_local_files", "validate_local_prerequisites",
            "_validate_camera_job", "_camera_attempt",
        ],
    }, "Alignment producer summary changed")

    require(value.get("activation_slot_bindings") == EXPECTED_ACTIVATION_SLOTS,
            "Output activation slot bindings changed")
    chain = value.get("authorized_terminal_activation_chain", {})
    expected_chains = _chains(
        {
            "retained_job_id": "camera-crop-replay-witness-002",
            "prospective_output_names": EXPECTED_CROP_NAMES,
        },
        {"prospective_output_names": EXPECTED_ALIGNMENT_NAMES},
        EXPECTED_ACTIVATION_SLOTS,
    )
    require(chain == {
        "model_count": 2,
        "required_slot_count": 6,
        "authorized_queue_entrypoint_terminal_activation_reachable": False,
        "chains": expected_chains,
        "end_to_end_activation_reachable_slot_count": 0,
        "diagnostic_only_unreachable_slot_count": 2,
        "downstream_blocked_slot_count": 4,
        "runtime_byte_verified_slot_count": 0,
        "semantic_replay_complete_slot_count": 0,
    }, "Authorized terminal activation chain changed")
    require(value.get("retained_activation_snapshot") == {
        "required_runtime_slots": 14,
        "unauthenticated_runtime_slots": 6,
        "runtime_byte_verified_slots": 0,
        "semantic_replay_complete_slots": 0,
        "request_field_values_authenticated": 4,
        "request_field_values_missing": 26,
        "complete_request_rows": 0,
        "total_request_rows": 1152,
        "sampling_key_feasible_rows": 0,
        "eligibility_feasible_rows": 0,
        "selected_request_rows": 0,
    }, "Retained activation snapshot changed")
    require(value.get("global_study_snapshot") == EXPECTED_GLOBAL_SNAPSHOT,
            "Output global study snapshot changed")
    require(value.get("new_activity") == ZERO_ACTIVITY,
            "Output science activity changed")
    require(value.get("gate_state") == EXPECTED_GATE_STATE,
            "Output gate state changed")
    require(value.get("audit_activity") == {
        "authenticated_git_blobs_read": 6,
        "authenticated_git_bytes_read": 216167,
        "python_sources_parsed_with_ast": 2,
        "producer_modules_imported": 0,
        "producer_functions_invoked": 0,
        "external_system_reads": 0,
        "raw_paths_opened": 0,
        "runtime_bytes_read": 0,
    }, "Audit activity changed")
    require(value.get("decision") == {
        "status": (
            "pinned_authorized_terminal_chain_cannot_authenticate_and_activate_"
            "six_missing_slots"
        ),
        "authorized_queue_entrypoint_terminal_activation_reachable": False,
        "camera_attempt_003_or_later_authenticated_by_pinned_activation_evidence": False,
        "safe_to_dispatch_replacement": False,
        "safe_to_materialize_annotation_package": False,
        "safe_to_release": False,
        "blocking_reasons": [
            "retained camera attempt 002 is diagnostic-only",
            "camera publisher has zero call sites",
            "alignment rejects camera attempts below 003 and attempts 001/002",
            "no authenticated passed camera attempt 003 or later is pinned",
            "current GM PVC queue worker and live-quiescence state are unreconciled",
        ],
    }, "Decision claim changed")
    require(value.get("claim_boundaries") == {
        "positive_claim": (
            "Commit-pinned activation evidence and source prove that the authorized "
            "queue entrypoints cannot terminally authenticate and activate the six "
            "missing annotation slots through the pinned producer chain."
        ),
        "static_reachability_is_not_runtime_observation": True,
        "prospective_output_names_are_not_existing_artifacts": True,
        "diagnostic_raw_outputs_may_exist_without_terminal_publication": True,
        "publisher_helper_callable_outside_authorized_entrypoint_is_not_assessed": True,
        "retained_active_zero_is_not_live_quiescence": True,
        "not_established": [
            "current GM PVC queue worker or release-lock state",
            "current existence of any raw camera or alignment artifact",
            "camera or physical alignment qualification",
            "request inventory, eligibility, or selection",
            "annotation media, labels, or rater responses",
            "forecast accuracy or scientific treatment effect",
            "authority to dispatch camera attempt 003 or any replacement work",
        ],
    }, "Claim boundaries changed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(load_pinned_blobs())
    data = canonical_bytes(report)
    if args.check:
        require(args.output is not None, "--check requires --output")
        require(args.output.read_bytes() == data, "Saved report is not canonical")
    elif args.output is not None:
        args.output.write_bytes(data)
    else:
        sys.stdout.buffer.write(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
