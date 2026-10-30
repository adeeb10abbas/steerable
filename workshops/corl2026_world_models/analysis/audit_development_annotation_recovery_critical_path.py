#!/usr/bin/env python3
"""Audit the retained annotation-recovery critical path without releasing work.

The audit is deliberately static.  It authenticates commit-pinned Git evidence,
scans the pinned forecast-layout queue-source subtree, and evaluates whether
either of the two mandatory runtime-input cuts can independently make the
annotation-media input runnable.  It never opens the raw/PVC namespace, imports
or invokes a producer, contacts GM, or creates a queue descriptor.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[3]
STUDY_ID = "WMF-ABLATION-001"
SCHEMA = "wmf-development-annotation-recovery-critical-path-v1"

SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
PRODUCER_REPORT_COMMIT = "fd0f89d9e27e340dc24334a3c253b3ecfc60cad0"
ACTIVATION_REPORT_COMMIT = "9122bbf8c6b8cbdcaba5d9eda345b6c518dde8e3"
MEDIA_REPORT_COMMIT = "393982967660f490717576fb9bf82033dbc0753e"
CAMERA_SOURCE_COMMIT = "93993dae1902ba561b594b9ae0849bd45b37ecc2"
ALIGNMENT_SOURCE_COMMIT = "a38b73ca539fe3cc7618fb1294b67b0e24861b58"
QUEUE_SOURCE_COMMIT = "b6c78c2d50dcc71a37ca17bdf3b2bd1333d78242"
QUEUE_MANIFEST_COMMIT = "b850cc576776af53cc4000d1f70a18c9b8ec9826"

SPEC_PATH = "workshops/corl2026_world_models/docs/ABLATION_SPEC.md"
PRODUCER_REPORT_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_producer_reachability.json"
)
ACTIVATION_REPORT_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_activation_conformance.json"
)
MEDIA_REPORT_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_media_input_closure.json"
)
SELECTION_REPORT_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_selection_readiness.json"
)
CAMERA_SOURCE_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "camera_crop_witness_jobs.py"
)
ALIGNMENT_SOURCE_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_physical_alignment_jobs.py"
)
QUEUE_SOURCE_PATH = (
    "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.py"
)
QUEUE_MANIFEST_PATH = (
    "workshops/corl2026_world_models/execution/20260912/autonomy/cluster_queue.json"
)
FORECAST_SOURCE_ROOT = (
    "workshops/corl2026_world_models/experiments/forecast_layout"
)
FORECAST_SOURCE_TREE = "da46e07db17eb59ca6e0b820ed09e4bfc08594a5"
FORECAST_PYTHON_INDEX = {
    "files": 47,
    "bytes": 2_855_968,
    "canonical_bytes": 7_527,
    "sha256": "43202f33467f687d3fd52b523b837b5ebf6e87f2be6a3d4e875ad6235fc0576b",
}

PINS = {
    (SPEC_COMMIT, SPEC_PATH): (
        27_094,
        "fa8cbbac7bf2eff2e01f73150e5cf073f7e4f177d3c0bfd4fc1f41fb3ea9309f",
    ),
    (PRODUCER_REPORT_COMMIT, PRODUCER_REPORT_PATH): (
        15_613,
        "1b8c847d5c13757825645077243349e6dfcc1e21ad8fd3298f8e12d10fa35c44",
    ),
    (ACTIVATION_REPORT_COMMIT, ACTIVATION_REPORT_PATH): (
        34_824,
        "ac0ed739b2e8026999ba318411596697a102efb5bd551ef8b96bc5822a9b4ebd",
    ),
    (MEDIA_REPORT_COMMIT, MEDIA_REPORT_PATH): (
        19_058,
        "762e95ce616ecc42486f382792ad4216af54715173ccffe3af353bd21ed8bf48",
    ),
    (MEDIA_REPORT_COMMIT, SELECTION_REPORT_PATH): (
        13_980,
        "1ecd3be3f4a26483581454a625db17da302dc66bbbf558286dd24c30d83a70f0",
    ),
    (CAMERA_SOURCE_COMMIT, CAMERA_SOURCE_PATH): (
        51_025,
        "6613de67a170d1719d5b4e1a5be1e2b9305e1a4b6087e8f1d2006f3e8b0f3145",
    ),
    (ALIGNMENT_SOURCE_COMMIT, ALIGNMENT_SOURCE_PATH): (
        90_462,
        "b805df1e251ca848a82a7ff0c39043d40154ce1030dd7db2e68b30b1ed09c44a",
    ),
    (QUEUE_SOURCE_COMMIT, QUEUE_SOURCE_PATH): (
        30_658,
        "c9fe3d2da0b889eae2bb6cff4c8c149583e4293b366840134e14dcb07a18ba22",
    ),
    (QUEUE_MANIFEST_COMMIT, QUEUE_MANIFEST_PATH): (
        126,
        "9c1bd52373df4308c3954c5b4598d03d66c9d5940182674904b1279d01a7346f",
    ),
}
PIN_INDEX = {
    "count": 9,
    "bytes": 282_840,
    "canonical_bytes": 1_760,
    "sha256": "f1a1a593f34b78fe224488b8bf6763c5edfdd9d4783ca7d64a20f29c165d6668",
}

CAMERA_JOB_RE = re.compile(r"camera-crop-replay-witness-(\d{3})\Z")
SCIENTIFIC_QUEUE_FIELDS = {
    "core_episode_ceiling",
    "current_GM_PVC_worker_state_reconciled",
    "live_quiescence_inferred",
    "safe_to_release_new_work",
    "terminal_valid_cells",
}
EXPECTED_QUEUE_DESCRIPTOR_FIELDS = {
    "argv",
    "job_id",
    "max_wall_seconds",
    "publish_log_tail_bytes",
    "released",
    "role",
    "source_commit",
}
EXPECTED_FUNCTION_SHA256 = {
    (CAMERA_SOURCE_PATH, "run_job"): "2081a6e5a246a0fc9f3fa5a10d8ea7a50c47e054584229e63b7f9e8d36e28d3c",
    (CAMERA_SOURCE_PATH, "_publish_contracts"): "e9a64dd3690821adbd75e6e80e36f798f38c8df40132c95ddf455f94ddfb1ca8",
    (ALIGNMENT_SOURCE_PATH, "_camera_attempt"): "1d19422bf045d545d14b8739a3e958ddba5c35de7d833b9ebf64b773cf7a0913",
    (ALIGNMENT_SOURCE_PATH, "_validate_camera_job"): "b5f7911ca3cab219a30515770e425768a2358bd1e147cf133553f413530ac674",
    (QUEUE_SOURCE_PATH, "normalize_job"): "85676453356c0b32368682accf5353e1720e669a9af24626a73251ec1ae5d099",
    (QUEUE_SOURCE_PATH, "stage_queue"): "1529c14c2f1fd9df68fd6c4822c72904ce99660bc0dd75ddd2abe447286bb659",
    (QUEUE_SOURCE_PATH, "claim_job"): "b0195925eaaf82455a736b2098339f36662cd3bea281962081e09c535fe4764c",
}
EXPECTED_NEW_ACTIVITY = {
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {label}") from error
    require(type(value) is dict, f"Expected JSON object: {label}")
    return value


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def git_output(*args: str) -> bytes:
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(REPOSITORY), *args],
        timeout=30,
    )


def git_blob(commit: str, path: str) -> bytes:
    return git_output("show", f"{commit}:{path}")


def pinned_blob(commit: str, path: str) -> tuple[bytes, dict[str, Any]]:
    payload = git_blob(commit, path)
    identity = {
        "commit": commit,
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    require(
        (identity["bytes"], identity["sha256"]) == PINS[(commit, path)],
        f"Pinned blob changed: {commit}:{path}",
    )
    return payload, identity


def pin_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = b"".join(
        (
            f"{row['commit']}\0{row['path']}\0{row['bytes']}\0"
            f"{row['sha256']}\n"
        ).encode("ascii")
        for row in sorted(rows, key=lambda item: (item["commit"], item["path"]))
    )
    value = {
        "count": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "canonical_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    require(value == PIN_INDEX, f"Authenticated pin index changed: {value}")
    return value


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(matches) == 1, f"Expected one function: {name}")
    return matches[0]


def _function_sha256(payload: bytes, tree: ast.Module, path: str, name: str) -> str:
    node = _function(tree, name)
    text = payload.decode("utf-8")
    lines = text.splitlines(keepends=True)
    body = "".join(lines[node.lineno - 1:node.end_lineno]).encode("utf-8")
    digest = hashlib.sha256(body).hexdigest()
    require(
        digest == EXPECTED_FUNCTION_SHA256[(path, name)],
        f"Function changed: {path}:{name}",
    )
    return digest


def _literal_assignment(tree: ast.Module, name: str) -> Any:
    matches: list[Any] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                matches.append(ast.literal_eval(node.value))
    require(len(matches) == 1, f"Expected one literal assignment: {name}")
    return matches[0]


def analyze_camera_source(payload: bytes) -> dict[str, Any]:
    tree = ast.parse(payload.decode("utf-8"), filename=CAMERA_SOURCE_PATH)
    job_id = _literal_assignment(tree, "JOB_ID")
    prior_job_id = _literal_assignment(tree, "PRIOR_JOB_ID")
    output_names = _literal_assignment(tree, "MODEL_CONTRACT_NAMES")
    require(job_id == "camera-crop-replay-witness-002", "Camera producer job changed")
    require(prior_job_id == "camera-crop-replay-witness-001", "Camera prior job changed")
    require(
        output_names
        == {"N3": "n3_camera_crop_contract.json", "D1": "d1_camera_crop_contract.json"},
        "Camera output names changed",
    )
    run_job = _function(tree, "run_job")
    publisher = _function(tree, "_publish_contracts")
    run_job_sha256 = _function_sha256(payload, tree, CAMERA_SOURCE_PATH, "run_job")
    publisher_sha256 = _function_sha256(
        payload, tree, CAMERA_SOURCE_PATH, "_publish_contracts"
    )
    references = [
        node for node in ast.walk(run_job)
        if isinstance(node, ast.Name) and node.id == "_publish_contracts"
    ]
    require(not references, "Diagnostic run_job gained a publisher reference")
    raises = [node for node in ast.walk(run_job) if isinstance(node, ast.Raise)]
    require(raises, "Diagnostic run_job lost terminal failure")
    constants = {
        node.value for node in ast.walk(run_job)
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool))
    }
    require(False in constants, "Diagnostic run_job lost false publication boundary")
    return {
        "job_id": job_id,
        "attempt_ordinal": 2,
        "prior_job_id": prior_job_id,
        "diagnostic_only": True,
        "publish_crop_contracts": False,
        "publisher_definition_count": 1 if publisher else 0,
        "authorized_run_job_sha256": run_job_sha256,
        "publisher_sha256": publisher_sha256,
        "publisher_reference_count_from_authorized_run_job": len(references),
        "authorized_terminal_success_path": False,
        "prospective_output_names": output_names,
    }


def analyze_alignment_source(payload: bytes) -> dict[str, Any]:
    tree = ast.parse(payload.decode("utf-8"), filename=ALIGNMENT_SOURCE_PATH)
    forbidden = set(_literal_assignment(tree, "FORBIDDEN_CAMERA_JOB_IDS"))
    output_names = _literal_assignment(tree, "OUTPUT_NAMES")
    camera_output_key = _literal_assignment(tree, "CAMERA_OUTPUT_KEY")
    require(
        forbidden
        == {"camera-crop-replay-witness-001", "camera-crop-replay-witness-002"},
        "Forbidden camera attempts changed",
    )
    require(
        output_names
        == {
            "N3": {
                "mapping": "n3_physical_alignment_receipt.json",
                "alignment": "n3_alignment_contract.json",
            },
            "D1": {
                "mapping": "d1_physical_alignment_receipt.json",
                "alignment": "d1_alignment_contract.json",
            },
        },
        "Physical-alignment outputs changed",
    )
    attempt = _function(tree, "_camera_attempt")
    attempt_sha256 = _function_sha256(
        payload, tree, ALIGNMENT_SOURCE_PATH, "_camera_attempt"
    )
    comparisons = [node for node in ast.walk(attempt) if isinstance(node, ast.Compare)]
    minimum_guard = any(
        any(isinstance(operator, ast.GtE) for operator in node.ops)
        and any(isinstance(item, ast.Constant) and item.value == 3 for item in [node.left, *node.comparators])
        for node in comparisons
    )
    require(minimum_guard, "Camera attempt minimum guard changed")
    validation = _function(tree, "_validate_camera_job")
    validation_sha256 = _function_sha256(
        payload, tree, ALIGNMENT_SOURCE_PATH, "_validate_camera_job"
    )
    strings = {
        node.value for node in ast.walk(validation)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for required in (
        "passed",
        "qualified_from_original_camera_pixels",
        "safe_for_physical_alignment_input",
    ):
        require(required in strings, f"Terminal camera requirement changed: {required}")
    require(camera_output_key == "published_camera_crop_contracts",
            "Terminal camera output key changed")
    return {
        "minimum_attempt_ordinal": 3,
        "forbidden_job_ids": sorted(forbidden),
        "required_status": "passed",
        "required_decision": "qualified_from_original_camera_pixels",
        "requires_original_camera_pixels": True,
        "requires_safe_for_physical_alignment_input": True,
        "output_key": camera_output_key,
        "required_published_crop_contracts": 2,
        "required_downstream_outputs": output_names,
        "camera_attempt_validator_sha256": attempt_sha256,
        "terminal_receipt_validator_sha256": validation_sha256,
    }


def analyze_queue_source(payload: bytes) -> dict[str, Any]:
    tree = ast.parse(payload.decode("utf-8"), filename=QUEUE_SOURCE_PATH)
    normalize = _function(tree, "normalize_job")
    normalize_sha256 = _function_sha256(
        payload, tree, QUEUE_SOURCE_PATH, "normalize_job"
    )
    stage_sha256 = _function_sha256(payload, tree, QUEUE_SOURCE_PATH, "stage_queue")
    claim_sha256 = _function_sha256(payload, tree, QUEUE_SOURCE_PATH, "claim_job")
    allowed_sets = []
    for node in ast.walk(normalize):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "allowed"
            for target in node.targets
        ):
            allowed_sets.append(set(ast.literal_eval(node.value)))
    require(allowed_sets == [EXPECTED_QUEUE_DESCRIPTOR_FIELDS], "Queue descriptor fields changed")
    source_strings = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    present_scientific = sorted(SCIENTIFIC_QUEUE_FIELDS & source_strings)
    require(not present_scientific, "Generic queue gained audited scientific-gate fields")
    return {
        "descriptor_fields": sorted(EXPECTED_QUEUE_DESCRIPTOR_FIELDS),
        "descriptor_field_count": len(EXPECTED_QUEUE_DESCRIPTOR_FIELDS),
        "syntactic_release_flags": ["released"],
        "scientific_gate_fields_audited": sorted(SCIENTIFIC_QUEUE_FIELDS),
        "scientific_gate_fields_enforced": present_scientific,
        "claim_time_control_predicates": [
            "queue_not_shutdown",
            "job_id_in_current_active_job_ids",
            "before_optional_admission_deadline",
        ],
        "immutable_single_claim_directory": True,
        "generic_queue_admission_is_not_scientific_authority": True,
        "function_sha256": {
            "normalize_job": normalize_sha256,
            "stage_queue": stage_sha256,
            "claim_job": claim_sha256,
        },
    }


def scan_forecast_python_tree() -> tuple[dict[str, Any], dict[str, bytes]]:
    tree = git_output("rev-parse", f"{PRODUCER_REPORT_COMMIT}:{FORECAST_SOURCE_ROOT}").decode().strip()
    require(tree == FORECAST_SOURCE_TREE, "Forecast-layout source tree changed")
    paths = git_output(
        "ls-tree", "-r", "--name-only", PRODUCER_REPORT_COMMIT, FORECAST_SOURCE_ROOT
    ).decode().splitlines()
    python_paths = sorted(path for path in paths if path.endswith(".py"))
    blobs = {path: git_blob(PRODUCER_REPORT_COMMIT, path) for path in python_paths}
    rows = b"".join(
        f"{path}|{len(blobs[path])}|{hashlib.sha256(blobs[path]).hexdigest()}\n".encode("ascii")
        for path in python_paths
    )
    value = {
        "tree": tree,
        "files": len(blobs),
        "bytes": sum(len(payload) for payload in blobs.values()),
        "canonical_bytes": len(rows),
        "sha256": hashlib.sha256(rows).hexdigest(),
    }
    require(value == {"tree": FORECAST_SOURCE_TREE, **FORECAST_PYTHON_INDEX},
            f"Forecast Python source index changed: {value}")
    return value, blobs


def camera_producer_inventory(blobs: Mapping[str, bytes]) -> dict[str, Any]:
    literal_references: list[dict[str, Any]] = []
    producer_assignments: list[dict[str, Any]] = []
    nonliteral_job_id_assignments: list[dict[str, Any]] = []
    for path, payload in sorted(blobs.items()):
        tree = ast.parse(payload.decode("utf-8"), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                match = CAMERA_JOB_RE.fullmatch(node.value)
                if match:
                    literal_references.append({
                        "path": path,
                        "line": node.lineno,
                        "job_id": node.value,
                        "attempt_ordinal": int(match.group(1)),
                    })
        for node in tree.body:
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(isinstance(target, ast.Name) and target.id == "JOB_ID" for target in targets):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                nonliteral_job_id_assignments.append({"path": path, "line": node.lineno})
                continue
            if isinstance(value, str) and (match := CAMERA_JOB_RE.fullmatch(value)):
                producer_assignments.append({
                    "path": path,
                    "line": node.lineno,
                    "job_id": value,
                    "attempt_ordinal": int(match.group(1)),
                })
    require(
        producer_assignments
        == [{
            "path": CAMERA_SOURCE_PATH,
            "line": 78,
            "job_id": "camera-crop-replay-witness-002",
            "attempt_ordinal": 2,
        }],
        "Pinned camera producer inventory changed",
    )
    require(not nonliteral_job_id_assignments,
            "Pinned subtree gained a nonliteral top-level JOB_ID assignment")
    require(
        sorted({row["job_id"] for row in literal_references})
        == ["camera-crop-replay-witness-001", "camera-crop-replay-witness-002"],
        "Pinned camera literal inventory changed",
    )
    literal_attempt_ge_003_assignments = [
        row for row in producer_assignments if row["attempt_ordinal"] >= 3
    ]
    return {
        "scope": f"{PRODUCER_REPORT_COMMIT}:{FORECAST_SOURCE_ROOT}",
        "literal_reference_count": len(literal_references),
        "unique_literal_job_ids": sorted({row["job_id"] for row in literal_references}),
        "producer_assignments": producer_assignments,
        "producer_assignment_count": len(producer_assignments),
        "nonliteral_top_level_job_id_assignment_count": len(
            nonliteral_job_id_assignments
        ),
        "module_level_literal_attempt_ge_003_producer_assignment_count": len(
            literal_attempt_ge_003_assignments
        ),
        "scan_definition": (
            "Module-level literal JOB_ID assignments matching "
            "camera-crop-replay-witness-NNN in the pinned Python subtree. The scan does not "
            "exclude dynamically constructed or non-module-level future producers."
        ),
    }


def _validate_reports(
    producer: Mapping[str, Any],
    activation: Mapping[str, Any],
    media: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    require(producer.get("schema_version") == "wmf-development-annotation-producer-reachability-v1",
            "Producer report schema changed")
    require(activation.get("schema_version") == "wmf-development-annotation-activation-conformance-v1",
            "Activation report schema changed")
    require(media.get("schema_version") == "wmf-development-annotation-media-input-closure-v1",
            "Media report schema changed")
    require(selection.get("schema_version") == "wmf-development-annotation-selection-readiness-v1",
            "Selection report schema changed")

    snapshot = producer["global_study_snapshot"]
    require(snapshot == activation["global_study_snapshot"] == selection["global_study_snapshot"],
            "Global retained snapshots differ")
    require(snapshot == {
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
    }, "Global retained snapshot changed")
    require(producer["new_activity"] == EXPECTED_NEW_ACTIVITY,
            "Producer report activity changed")

    gate = producer["gate_state"]
    require(gate.get("core_episode_ceiling") == 232, "Core ceiling changed")
    for key in (
        "annotation_media_ready",
        "camera_alignment_ready",
        "confirmation_release_ready",
        "current_GM_PVC_worker_state_reconciled",
        "development_annotation_gate_ready",
        "human_annotation_ready",
        "live_quiescence_inferred",
        "resource_freeze_ready",
        "safe_to_release_new_work",
    ):
        require(gate.get(key) is False, f"Closed gate changed: {key}")

    chain = producer["authorized_terminal_activation_chain"]
    require(
        chain.get("required_slot_count") == 6
        and chain.get("diagnostic_only_unreachable_slot_count") == 2
        and chain.get("downstream_blocked_slot_count") == 4
        and chain.get("end_to_end_activation_reachable_slot_count") == 0,
        "Producer-chain partition changed",
    )
    matrix = activation["runtime_slot_activation_matrix"]
    require(
        matrix.get("required_slots") == 14
        and matrix.get("recovery_manifest_pointer_slots") == 5
        and matrix.get("retained_git_payload_slots_outside_recovery_manifest") == 3
        and matrix.get("identity_bound_slots") == 8
        and matrix.get("unauthenticated_slots") == 6
        and matrix.get("runtime_byte_verified_slots") == 0
        and matrix.get("semantic_replay_complete_slots") == 0,
        "Runtime-slot partition changed",
    )
    source_counts = matrix.get("source_state_counts")
    require(source_counts == {
        "declared_raw_pointer_in_recovery_manifest_current_availability_unknown": 5,
        "no_authenticated_descriptor_in_pinned_sources": 6,
        "retained_git_blob_outside_recovery_manifest_runtime_materialization_unverified": 3,
    }, "Runtime source-state partition changed")
    exact_slot_ids = {
        "D1.alignment_contract",
        "D1.camera_crop_contract",
        "D1.physical_alignment_receipt",
        "D1.request_provenance",
        "D1.timing_job_receipt",
        "D1.timing_sidecar",
        "N3.alignment_contract",
        "N3.camera_crop_contract",
        "N3.physical_alignment_receipt",
        "N3.request_provenance",
        "N3.timing_job_receipt",
        "N3.timing_sidecar",
        "compiler.compiler_receipt",
        "compiler.job_receipt",
    }
    observed_slot_ids = [row.get("slot_id") for row in matrix.get("slots", [])]
    require(
        len(observed_slot_ids) == len(set(observed_slot_ids)) == 14
        and set(observed_slot_ids) == exact_slot_ids,
        "Runtime slot IDs changed",
    )
    producer_slot_ids = {
        node.get("slot_id")
        for model_chain in chain.get("chains", [])
        for node in model_chain.get("nodes", [])
    }
    require(
        producer_slot_ids
        == {
            "D1.alignment_contract",
            "D1.camera_crop_contract",
            "D1.physical_alignment_receipt",
            "N3.alignment_contract",
            "N3.camera_crop_contract",
            "N3.physical_alignment_receipt",
        },
        "Producer-chain slot IDs changed",
    )
    require(
        media["runtime_input_contract"].get("required_top_level_slots") == 14
        and media["runtime_input_contract"].get("runnable_input_manifest_created") is False
        and media["runtime_input_contract"].get("queue_descriptor_created") is False,
        "Media runtime-input boundary changed",
    )
    readiness = selection["selection_readiness"]
    population = selection["retained_population"]
    require(
        population.get("rows") == 1152
        and population.get("annotation_request_schema_field_count") == 30
        and population.get("exact_annotation_schema_fields_available_count") == 4
        and population.get("exact_annotation_schema_fields_missing_count") == 26,
        "Retained request population changed",
    )
    require(
        readiness.get("sampling_key_feasible_rows") == 0
        and readiness.get("eligibility_decision_feasible_rows") == 0
        and readiness.get("selected_request_count") == 0
        and readiness.get("selection_manifest_created") is False,
        "Selection readiness changed",
    )
    return {
        "global_snapshot": dict(snapshot),
        "gate_state": dict(gate),
        "producer_chain": dict(chain),
        "runtime_matrix": dict(matrix),
        "population": dict(population),
        "selection_readiness": dict(readiness),
        "runtime_slot_ids": sorted(exact_slot_ids),
        "producer_slot_ids": sorted(producer_slot_ids),
    }


def counterfactuals(*, producer_slots: int, identity_bound_slots: int, total: int) -> list[dict[str, Any]]:
    require((producer_slots, identity_bound_slots, total) == (6, 8, 14),
            "Critical-path partition changed")
    rows = [
        {
            "scenario": "oracle_producer_chain_only_fully_verified_and_replayed",
            "oracle_assumption": "all six producer-chain slots are available, byte-verified, and semantically replayed",
            "operational_feasibility_or_order_asserted": False,
            "hypothetically_closed_slots": producer_slots,
            "still_open_slots": total - producer_slots,
            "runnable_manifest": False,
            "materializable_request_rows": 0,
            "reason": "five recovery pointers and three retained-Git runtime payloads remain unverified and unreplayed",
        },
        {
            "scenario": "oracle_identity_bound_eight_only_fully_verified_and_replayed",
            "oracle_assumption": "all eight identity-bound runtime slots are available, byte-verified, and semantically replayed",
            "operational_feasibility_or_order_asserted": False,
            "hypothetically_closed_slots": identity_bound_slots,
            "still_open_slots": total - identity_bound_slots,
            "runnable_manifest": False,
            "materializable_request_rows": 0,
            "reason": "two camera and four downstream alignment/mapping slots remain unauthenticated",
        },
        {
            "scenario": "all_fourteen_identity_bound_without_runtime_verification_or_replay",
            "oracle_assumption": "all fourteen slots have identity declarations only",
            "operational_feasibility_or_order_asserted": False,
            "hypothetically_identity_bound_slots": total,
            "runtime_byte_verified_slots": 0,
            "semantic_replay_complete_slots": 0,
            "runnable_manifest": False,
            "materializable_request_rows": 0,
            "reason": "identity declarations are not runtime bytes or semantic replay",
        },
    ]
    require(all(row["runnable_manifest"] is False for row in rows),
            "A single-cut counterfactual became runnable")
    return rows


def audit() -> dict[str, Any]:
    identities: list[dict[str, Any]] = []

    def load(commit: str, path: str) -> bytes:
        payload, identity = pinned_blob(commit, path)
        identities.append(identity)
        return payload

    spec = load(SPEC_COMMIT, SPEC_PATH).decode("utf-8")
    producer = load_json(load(PRODUCER_REPORT_COMMIT, PRODUCER_REPORT_PATH), PRODUCER_REPORT_PATH)
    activation = load_json(load(ACTIVATION_REPORT_COMMIT, ACTIVATION_REPORT_PATH), ACTIVATION_REPORT_PATH)
    media = load_json(load(MEDIA_REPORT_COMMIT, MEDIA_REPORT_PATH), MEDIA_REPORT_PATH)
    selection = load_json(load(MEDIA_REPORT_COMMIT, SELECTION_REPORT_PATH), SELECTION_REPORT_PATH)
    camera_source = load(CAMERA_SOURCE_COMMIT, CAMERA_SOURCE_PATH)
    alignment_source = load(ALIGNMENT_SOURCE_COMMIT, ALIGNMENT_SOURCE_PATH)
    queue_source = load(QUEUE_SOURCE_COMMIT, QUEUE_SOURCE_PATH)
    queue_manifest = load_json(load(QUEUE_MANIFEST_COMMIT, QUEUE_MANIFEST_PATH), QUEUE_MANIFEST_PATH)

    require("## 3. Exact core run matrix" in spec, "Scientific matrix section changed")
    require("Across all 232 planned core episodes" in spec, "Core ceiling text changed")
    require("These are launch blockers with defined outputs, not permissions to invent values" in spec,
            "Unresolved-prerequisite boundary changed")
    require("Camera mapping" in spec and "Annotation" in spec and "Resources" in spec,
            "Scientific prerequisites changed")

    sources = pin_index(identities)
    report_state = _validate_reports(producer, activation, media, selection)
    camera = analyze_camera_source(camera_source)
    alignment = analyze_alignment_source(alignment_source)
    queue = analyze_queue_source(queue_source)
    require(queue_manifest == {
        "jobs": [],
        "namespace": "wmf_ablation_001_20260912",
        "schema_version": "wmf-cluster-queue-v1",
        "shutdown": False,
    }, "Retained queue manifest changed")
    tree_index, blobs = scan_forecast_python_tree()
    inventory = camera_producer_inventory(blobs)

    require(camera["authorized_terminal_success_path"] is False,
            "Diagnostic camera producer became terminal")
    require(inventory["module_level_literal_attempt_ge_003_producer_assignment_count"] == 0,
            "Pinned subtree gained an attempt >=003 producer")
    require(producer["decision"].get("camera_attempt_003_or_later_authenticated_by_pinned_activation_evidence") is False,
            "Pinned activation evidence gained attempt >=003")

    runtime = report_state["runtime_matrix"]
    cuts = counterfactuals(
        producer_slots=report_state["producer_chain"]["required_slot_count"],
        identity_bound_slots=runtime["identity_bound_slots"],
        total=runtime["required_slots"],
    )

    return {
        "schema_version": SCHEMA,
        "study_id": STUDY_ID,
        "status": "two_independent_mandatory_runtime_cuts_open_no_literal_attempt_ge_003_producer_or_retained_queue_descriptor",
        "authenticated_source_blobs": sources,
        "pinned_forecast_layout_python_tree": tree_index,
        "retained_camera_replacement_inventory": {
            **inventory,
            "retained_queue_manifest_jobs": 0,
            "retained_queue_manifest_shutdown": False,
            "retained_queue_manifest_is_historical_not_live_GM_state": True,
            "diagnostic_producer": camera,
            "terminal_downstream_interface": alignment,
            "retained_queue_manifest_terminal_replacement_descriptor_count": 0,
            "pinned_activation_evidence_authenticates_attempt_ge_003": False,
        },
        "generic_queue_admission": queue,
        "runtime_critical_path": {
            "required_slots": 14,
            "producer_chain_slots": 6,
            "diagnostic_camera_slots": 2,
            "downstream_alignment_mapping_slots": 4,
            "recovery_manifest_pointer_slots": 5,
            "retained_git_runtime_payload_slots": 3,
            "identity_bound_runtime_slots": 8,
            "current_runtime_byte_verified_slots": 0,
            "current_semantic_replay_complete_slots": 0,
            "partition_equation": "6 + 5 + 3 = 14",
            "producer_partition_equation": "2 + 4 = 6",
            "required_slot_ids": report_state["runtime_slot_ids"],
            "producer_chain_slot_ids": report_state["producer_slot_ids"],
            "counterfactuals": cuts,
            "camera_recovery_necessary_in_pinned_chain": True,
            "camera_recovery_independently_sufficient": False,
            "runtime_payload_recovery_independently_sufficient": False,
        },
        "current_request_closure": {
            "rows": 1152,
            "schema_fields": 30,
            "authenticated_field_values": 4,
            "missing_field_values": 26,
            "complete_rows": 0,
            "sampling_key_feasible_rows": 0,
            "eligibility_decision_feasible_rows": 0,
            "selected_rows": 0,
            "selection_manifest_created": False,
        },
        "global_study_snapshot": report_state["global_snapshot"],
        "gate_state": report_state["gate_state"],
        "new_activity": dict(EXPECTED_NEW_ACTIVITY),
        "decision": {
            "runnable_annotation_media_manifest": False,
            "safe_to_materialize_camera_replacement_descriptor": False,
            "safe_to_dispatch_camera_replacement": False,
            "safe_to_release_new_work": False,
            "safe_to_materialize_annotation_package": False,
            "blocking_reasons": [
                "zero module-level literal camera producer JOB_ID assignments at ordinal at least 003 occur in the pinned 47-file forecast-layout Python subtree",
                "the retained queue manifest contains zero descriptors and pinned activation evidence authenticates no attempt at ordinal at least 003",
                "six camera/alignment producer-chain slots remain unauthenticated",
                "five recovery-pointer and three retained-Git runtime slots remain unverified and unreplayed",
                "current GM PVC queue worker GPU child reap release-lock and live-quiescence state are unreconciled",
            ],
        },
        "claim_boundaries": {
            "positive_claim": (
                "Within the pinned forecast-layout Python subtree there are zero module-level literal "
                "camera producer JOB_ID assignments at ordinal at least 003; the retained queue manifest "
                "contains zero descriptors and pinned activation evidence authenticates no attempt at "
                "ordinal at least 003. Independently, neither "
                "the six-slot producer cut nor the eight-slot runtime-payload cut can alone make the "
                "fourteen-slot annotation-media input runnable."
            ),
            "dynamic_or_non_module_level_future_producers_excluded_by_scan": False,
            "counterfactuals_are_not_observed_state": True,
            "generic_queue_release_is_not_scientific_authority": True,
            "pinned_subtree_absence_is_not_repo_global_or_external_absence": True,
            "retained_empty_queue_is_not_live_quiescence": True,
            "camera_recovery_is_necessary_but_not_independently_sufficient": True,
            "not_established": [
                "current GM PVC queue worker GPU child reap release-lock or live-quiescence state",
                "current existence or absence of raw camera attempt 002 or any external attempt 003",
                "safety or authority to create dispatch or reuse any job identifier",
                "runtime byte verification or semantic replay of any required slot",
                "request-row materialization eligibility selection media human evidence labels or accuracy",
                "measured resource qualification or confirmation release",
            ],
        },
        "audit_activity": {
            "authenticated_git_blobs_read": 9,
            "authenticated_git_bytes_read": 282840,
            "forecast_layout_python_files_parsed": 47,
            "forecast_layout_python_bytes_parsed": 2855968,
            "external_system_reads": 0,
            "raw_paths_opened": 0,
            "runtime_bytes_read": 0,
            "producer_modules_imported": 0,
            "producer_functions_invoked": 0,
            "queue_descriptors_created": 0,
            "remote_contacts": 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical report")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved recovery critical-path report differs")
        print(
            "Verified two independent open runtime cuts, zero pinned module-level literal "
            "camera producers at ordinal >=003, and zero runnable annotation rows."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
