#!/usr/bin/env python3
"""Audit retained Git evidence for annotation-media input closure without release.

This audit is deliberately not an annotation-media input manifest.  It opens no
raw path, emits no queue descriptor, and treats every PVC availability claim as
unknown.  Its only data plane is authenticated ``git show`` of immutable blobs.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping


REPOSITORY = Path(__file__).resolve().parents[3]

LINEAGE_COMMIT = "296fefc0ad14b3320d7c45a8a11fde6f0908deb6"
READINESS_COMMIT = "84332bf638595ff7ed955fd263586ecc0e06ab93"
STATUS_COMMIT = "26fd6a4c426df3b20c0f5276eee52d34cf9194f0"
MEDIA_COMMIT = "cfcfd241fbd8b82e416f814a019c8c20d9c1c359"
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"

LINEAGE_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_request_recovery_lineage.json"
)
READINESS_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_readiness_inventory.json"
)
STATUS_PATH = "workshops/corl2026_world_models/execution/20260912/status.json"
MEDIA_CONTRACT_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_annotation_media_bridge_contract.json"
)
MEDIA_JOBS_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_annotation_media_jobs.py"
)
COMPILER_JOB_RECEIPT_PATH = (
    "results/jobs/development-evidence-compiler-formal-004/publish/"
    "development_evidence_compiler_job_receipt.json"
)
D1_TIMING_JOB_RECEIPT_PATH = (
    "results/jobs/timing-d1-development-sidecar-002/publish/"
    "timing_job_receipt.json"
)
N3_TIMING_JOB_RECEIPT_PATH = (
    "results/jobs/timing-n3-development-sidecar-001/publish/"
    "timing_job_receipt.json"
)

EXPECTED_BLOBS: dict[tuple[str, str], tuple[int, str]] = {
    (LINEAGE_COMMIT, LINEAGE_PATH): (
        573283,
        "5714efd75bde846c41d1bb3f0ec88df427d3b18bf8c0df62633b6b38b8a95878",
    ),
    (READINESS_COMMIT, READINESS_PATH): (
        10092,
        "41303ab00b78f90f0f7e3fc721550a8ca6547f74f8d2c38f9b912aae06e8c672",
    ),
    (STATUS_COMMIT, STATUS_PATH): (
        106292,
        "b01b9171211aa197caf41698194153fd8bdc2a1f5c4c270c6e95f2e929473881",
    ),
    (MEDIA_COMMIT, MEDIA_CONTRACT_PATH): (
        3498,
        "893996c8157dad803c1bd57e57b3922f4c340a66846aa457e0d6bf27c1491d39",
    ),
    (MEDIA_COMMIT, MEDIA_JOBS_PATH): (
        76709,
        "13d1bbbbd19e3ded30d3bc9d27f1e8a909d478b691afa320bc7e6c68a5425190",
    ),
    (RESULTS_COMMIT, COMPILER_JOB_RECEIPT_PATH): (
        12374,
        "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434",
    ),
    (RESULTS_COMMIT, D1_TIMING_JOB_RECEIPT_PATH): (
        79824,
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
    ),
    (RESULTS_COMMIT, N3_TIMING_JOB_RECEIPT_PATH): (
        29935,
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
    ),
}

RAW_ROOT = "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912"
MODELS = ("D1", "N3")
MODEL_FIELDS = (
    "alignment_contract",
    "camera_crop_contract",
    "physical_alignment_receipt",
    "request_provenance",
    "timing_job_receipt",
    "timing_sidecar",
)
EXPECTED_CONTRACT_COUNTS = {
    "alignment_contracts": 2,
    "compiler_job_receipts": 1,
    "compiler_receipts": 1,
    "model_request_provenance_files": 2,
    "passed_timing_job_receipts": 2,
    "passed_timing_sidecars": 2,
    "signed_camera_crop_contracts": 2,
    "signed_physical_alignment_receipts": 2,
}
EXPECTED_LINEAGE_INDEX = {
    "count": 1152,
    "canonical_bytes": 160740,
    "canonicalization": "ordered UTF-8 rows with LF terminators",
    "sha256": "165b87ad0d260478fb2ba13348887edc723d962e3ed29b53539cf2025402d935",
}
EXPECTED_DEPENDENCY_INDEX = {
    "count": 56,
    "canonical_bytes": 17801,
    "canonicalization": "ordered UTF-8 rows with LF terminators",
    "sha256": "0c13131e11b40ab710f8622c761f68b19f35a776d316d41cc128d3eb39294ad1",
}
EXPECTED_EDGE_INDEX = {
    "count": 160,
    "canonical_bytes": 23008,
    "canonicalization": "ordered UTF-8 rows with LF terminators",
    "sha256": "18c4c537ca3514184579779fd4f81ae6d44b3f6b4c06d9d4770d55598ecc2686",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_json(payload: bytes, source: str) -> dict:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON: {source}") from error
    require(type(value) is dict, f"Expected JSON object: {source}")
    return value


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def compact_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_sha256(value: object, label: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


def require_zero_counts(value: object, label: str) -> None:
    require(type(value) is dict and value, f"Missing zero-count object: {label}")
    require(
        all(type(item) is int and item == 0 for item in value.values()),
        f"Nonzero science activity: {label}",
    )


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(REPOSITORY),
            "show",
            f"{commit}:{path}",
        ],
        timeout=30,
    )


def load_pinned_blobs() -> dict[tuple[str, str], bytes]:
    blobs: dict[tuple[str, str], bytes] = {}
    for key, expected in EXPECTED_BLOBS.items():
        payload = git_blob(*key)
        actual = (len(payload), sha256_bytes(payload))
        require(actual == expected, f"Pinned blob changed: {key[0]}:{key[1]}")
        blobs[key] = payload
    return blobs


def source_descriptor(commit: str, path: str, payload: bytes) -> dict:
    return {
        "commit": commit,
        "path": path,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def line_index(records: list[str], label: str, *, ordered: bool = False) -> dict:
    require(len(records) == len(set(records)), f"Duplicate {label} records")
    rows = records if ordered else sorted(records)
    payload = "".join(f"{record}\n" for record in rows).encode("utf-8")
    return {
        "count": len(records),
        "canonical_bytes": len(payload),
        "canonicalization": (
            "ordered UTF-8 rows with LF terminators"
            if ordered
            else "lexicographically sorted UTF-8 rows with LF terminators"
        ),
        "sha256": sha256_bytes(payload),
    }


def verify_signed(value: Mapping[str, Any], label: str) -> None:
    supplied = require_sha256(value.get("payload_sha256"), f"{label} payload")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(compact_bytes(unsigned)) == supplied, f"Signed payload changed: {label}")


def top_level_assignments(tree: ast.Module) -> dict[str, object]:
    result: dict[str, object] = {}
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


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(matches) == 1, f"Expected one function: {name}")
    return matches[0]


def literal_string_sets(node: ast.AST) -> list[set[str]]:
    result: list[set[str]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Set):
            continue
        if all(isinstance(item, ast.Constant) and type(item.value) is str for item in child.elts):
            result.append({item.value for item in child.elts})
    return result


def validate_media_source(payload: bytes) -> dict:
    try:
        source = payload.decode("utf-8")
        tree = ast.parse(source, filename=MEDIA_JOBS_PATH)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("Media jobs source is not valid pinned Python") from error

    assignments = top_level_assignments(tree)
    expected_assignments = {
        "INPUT_SCHEMA": "wmf-development-annotation-media-input-v1",
        "MODE": "formal_full_development",
        "MODELS": ("N3", "D1"),
        "EXPECTED_EPISODES": 32,
        "EXPECTED_REQUESTS": 1152,
        "EXPECTED_ELIGIBLE": 448,
        "EXPECTED_SELECTED": 128,
        "EXPECTED_D1_UNMAPPED": 672,
    }
    for name, expected in expected_assignments.items():
        require(assignments.get(name) == expected, f"Media source constant changed: {name}")

    validator = function_node(tree, "validate_input_shape")
    string_sets = literal_string_sets(validator)
    top_fields = {"schema_version", "study_id", "mode", "raw_root", "compiler", "models"}
    compiler_fields = {"job_receipt", "compiler_receipt"}
    model_fields = set(MODEL_FIELDS)
    require(top_fields in string_sets, "Media input top-level fields changed")
    require(compiler_fields in string_sets, "Media compiler fields changed")
    require(model_fields in string_sets, "Media model fields changed")

    validator_strings = {
        child.value for child in ast.walk(validator)
        if isinstance(child, ast.Constant) and type(child.value) is str
    }
    require(
        "annotation media input paths are ambiguous" in validator_strings,
        "Media input path-uniqueness guard changed",
    )

    descriptor_builder = function_node(tree, "_job_descriptor")
    released_values: list[ast.AST] = []
    for child in ast.walk(descriptor_builder):
        if not isinstance(child, ast.Dict):
            continue
        for key, value in zip(child.keys, child.values, strict=True):
            if isinstance(key, ast.Constant) and key.value == "released":
                released_values.append(value)
    require(
        len(released_values) == 1
        and isinstance(released_values[0], ast.Constant)
        and released_values[0].value is True,
        "Media job release behavior changed",
    )
    function_node(tree, "_validate_runtime_inputs")
    function_node(tree, "build_formal_wave")

    return {
        "runtime_input_schema": assignments["INPUT_SCHEMA"],
        "mode": assignments["MODE"],
        "models": list(assignments["MODELS"]),
        "top_level_fields": sorted(top_fields),
        "compiler_fields": sorted(compiler_fields),
        "model_fields": sorted(model_fields),
        "descriptor_fields": ["bytes", "path", "sha256"],
        "emitted_job_descriptor_released": True,
    }


def validate_job_receipt(
    value: dict,
    *,
    kind: str,
    job_id: str,
    model_id: str | None,
) -> str:
    verify_signed(value, f"{kind} {job_id}")
    require(value.get("job_id") == job_id, f"Job identity changed: {job_id}")
    require(value.get("status") == "passed" and value.get("decision") == "go", f"Job did not pass: {job_id}")
    require(value.get("safe_to_release_confirmation") is False, f"Job released confirmation: {job_id}")
    require_zero_counts(value.get("science_counts"), f"job {job_id}")
    job_dir = value.get("job_dir")
    require(type(job_dir) is str, f"Job directory missing: {job_id}")
    job_path = PurePosixPath(job_dir)
    require(job_path.is_absolute(), f"Job directory is not absolute: {job_id}")
    require(
        str(job_path) == f"{RAW_ROOT}/control/jobs/{job_id}",
        f"Job directory changed: {job_id}",
    )

    if kind == "compiler":
        require(
            value.get("schema_version") == "wmf-development-evidence-compiler-queue-job-v1"
            and value.get("formal_cohort_complete") is True
            and value.get("confirmation_released") is False,
            "Compiler job gate changed",
        )
        require(
            value.get("counts") == {
                "compiled_cells": 32,
                "compiled_source_behavioral_actions": 14400,
                "compiled_source_behavioral_requests": 1152,
            },
            "Compiler job counts changed",
        )
        require_zero_counts(value.get("compiler_science_activity"), "compiler activity")
        filename = "development_evidence_compiler_job_receipt.json"
        require(value.get("published_files") == [filename], "Compiler published files changed")
    else:
        require(
            value.get("schema_version") == "wmf-development-timing-sidecar-queue-job-v1"
            and value.get("model_id") == model_id
            and value.get("physical_time_qualified") is True,
            f"Timing job gate changed: {model_id}",
        )
        require(
            value.get("referenced_behavioral_cells") == 16
            and value.get("referenced_behavioral_model_requests")
            == ({"D1": 912, "N3": 240}[str(model_id)]),
            f"Timing job coverage changed: {model_id}",
        )
        filename = "timing_job_receipt.json"
    return str(job_path / "publish" / filename)


def validate_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict and set(value) == {"bytes", "path", "sha256"}, f"Descriptor keys changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Descriptor bytes changed: {label}")
    require_sha256(value["sha256"], label)
    path = value["path"]
    require(type(path) is str and path.startswith(RAW_ROOT + "/"), f"Descriptor path changed: {label}")
    require(".." not in PurePosixPath(path).parts, f"Descriptor path escapes root: {label}")
    return dict(value)


def build_report(blobs: Mapping[tuple[str, str], bytes]) -> dict:
    require(set(blobs) == set(EXPECTED_BLOBS), "Source blob set changed")
    sources = [source_descriptor(commit, path, blobs[(commit, path)]) for commit, path in blobs]
    require(len({(row["commit"], row["path"]) for row in sources}) == 8, "Duplicate source blobs")

    lineage = load_json(blobs[(LINEAGE_COMMIT, LINEAGE_PATH)], "lineage")
    readiness = load_json(blobs[(READINESS_COMMIT, READINESS_PATH)], "readiness")
    status = load_json(blobs[(STATUS_COMMIT, STATUS_PATH)], "status")
    contract = load_json(blobs[(MEDIA_COMMIT, MEDIA_CONTRACT_PATH)], "media contract")
    compiler_job = load_json(
        blobs[(RESULTS_COMMIT, COMPILER_JOB_RECEIPT_PATH)], "compiler job receipt"
    )
    d1_timing_job = load_json(
        blobs[(RESULTS_COMMIT, D1_TIMING_JOB_RECEIPT_PATH)], "D1 timing job receipt"
    )
    n3_timing_job = load_json(
        blobs[(RESULTS_COMMIT, N3_TIMING_JOB_RECEIPT_PATH)], "N3 timing job receipt"
    )
    source_contract = validate_media_source(blobs[(MEDIA_COMMIT, MEDIA_JOBS_PATH)])

    require(lineage.get("schema_version") == "wmf-development-request-recovery-lineage-v1", "Lineage schema changed")
    require(lineage.get("study_id") == "WMF-ABLATION-001", "Lineage study changed")
    require(lineage["request_lineage"].get("index") == EXPECTED_LINEAGE_INDEX, "Lineage index changed")
    require(lineage["request_lineage"].get("requests") == 1152, "Lineage requests changed")
    require(lineage["request_lineage"].get("cells") == 32, "Lineage cells changed")
    require(lineage["request_lineage"].get("requests_by_model") == {"D1": 912, "N3": 240}, "Lineage model counts changed")
    require(lineage["request_lineage"].get("requests_per_cell") == {"D1": 57, "N3": 15}, "Lineage per-cell counts changed")
    require(lineage["request_lineage"].get("cross_model_identity_overlap") == 0, "Lineage cross-model overlap changed")

    recovery = lineage["recovery_manifest"]
    require(recovery.get("expected_descriptor_count") == 56, "Recovery descriptor count changed")
    require(recovery.get("declared_bytes") == 32664538, "Recovery byte total changed")
    require(recovery.get("dependency_index") == EXPECTED_DEPENDENCY_INDEX, "Recovery dependency index changed")
    require(lineage["recovery_scope_edges"].get("index") == EXPECTED_EDGE_INDEX, "Recovery edge index changed")
    require(lineage["recovery_scope_edges"].get("edges_per_cell") == 5, "Recovery edge fanout changed")
    require(
        lineage["study_scope_dependencies"].get("cell_scope_unique_dependencies") == 48
        and lineage["study_scope_dependencies"].get("compiler_dependencies") == 8
        and lineage["study_scope_dependencies"].get("all_dependencies_partitioned") is True,
        "Recovery scope partition changed",
    )
    pending = lineage["pending_raw_verification"]
    require(
        pending.get("safe_to_release_new_work") is False
        and pending.get("retrieval_released") is False
        and pending.get("retrieval_queue_descriptor_created") is False
        and pending.get("raw_paths_opened") == 0
        and pending.get("raw_bytes_read") == 0
        and pending.get("current_raw_availability_verified") is False,
        "Lineage raw/release boundary changed",
    )
    annotation_boundary = lineage["annotation_and_release_boundary"]
    require(
        annotation_boundary.get("production_eligibility_determined") is False
        and annotation_boundary.get("production_request_selection_manifest_created") is False
        and annotation_boundary.get("selected_request_count") == 0
        and annotation_boundary.get("annotation_media_created") == 0
        and annotation_boundary.get("human_responses") == 0
        and annotation_boundary.get("labels") == 0
        and annotation_boundary.get("camera_crop_contract_authenticated") is False
        and annotation_boundary.get("physical_alignment_authenticated") is False,
        "Lineage annotation boundary changed",
    )

    require(readiness.get("schema_version") == "wmf-development-annotation-readiness-inventory-v1", "Readiness schema changed")
    require(readiness.get("pins", {}).get("results_commit") == RESULTS_COMMIT, "Readiness result pin changed")
    require(readiness.get("design") == {
        "confirmation_cells": 192,
        "core_episode_ceiling": 232,
        "development_cells": 32,
        "development_selected_request_ceiling": 128,
        "independent_raters_required": 2,
        "optional_D2_selected": False,
        "request_sample_cap_per_episode": 4,
    }, "Readiness design changed")
    require(
        readiness["camera_and_media_gate"].get("crop_contract_published") is False
        and readiness["camera_and_media_gate"].get("media_renderable_for_human_annotation") is False
        and readiness["human_annotation_gate"].get("production_request_selection_manifest_authenticated_by_this_audit") is False
        and readiness["human_annotation_gate"].get("human_labels_consumed") == 0
        and readiness["resource_and_release_gate"].get("safe_to_release_confirmation") is False,
        "Readiness closed gates changed",
    )

    incident = status["results_publication_incident"]
    require(
        incident.get("safe_to_release_new_work") is False
        and incident.get("current_job_states_known") is False
        and incident.get("fresh_snapshot_after_empty_queue") is False,
        "Current release incident changed",
    )
    require(
        status.get("current_owned_job_state")
        == "unknown_until_fresh_results_publication_after_authorized_gm_reconciliation",
        "Current owned-job boundary changed",
    )
    status_lineage = status["development_request_recovery_lineage_20261030"]
    require(
        status_lineage.get("source_commit") == LINEAGE_COMMIT
        and status_lineage["pending_raw_verification"].get("safe_to_release_new_work") is False,
        "Status lineage pin/release boundary changed",
    )

    require(contract.get("schema_version") == "wmf-development-annotation-media-bridge-contract-v1", "Media contract schema changed")
    require(contract.get("study_id") == "WMF-ABLATION-001", "Media contract study changed")
    require(contract["job"].get("raw_root") == RAW_ROOT, "Media raw root changed")
    require(contract["job"].get("mode") == source_contract["mode"], "Media mode differs from source")
    input_policy = contract["input_policy"]
    for key, expected in EXPECTED_CONTRACT_COUNTS.items():
        require(input_policy.get(key) == expected, f"Media input count changed: {key}")
    require(input_policy.get("join_key") == ["model_id", "cell_id", "request_index"], "Media join key changed")
    require(sum(EXPECTED_CONTRACT_COUNTS.values()) == 14, "Media slot arithmetic changed")
    exact_outputs = contract["exact_output_counts"]
    require(exact_outputs == {
        "d1_forecast_timing_unavailable_requests": 672,
        "early_horizon_required_for_both_models": True,
        "eligible_requests_per_episode": 14,
        "episodes": 32,
        "request_inclusion_probability_exact": "4/14",
        "requests": 1152,
        "requests_by_model": {"D1": 912, "N3": 240},
        "selected_requests": 128,
        "selected_requests_by_model": {"D1": 64, "N3": 64},
        "source_timing_capable_requests": 480,
        "timing_camera_action_eligible_requests": 448,
    }, "Media output expectations changed")
    require(contract.get("safe_for_rater_distribution") is False and contract.get("safe_to_release_confirmation") is False, "Media contract release boundary changed")
    require_zero_counts(contract.get("science_counts"), "media contract")

    compiler_runtime_path = validate_job_receipt(
        compiler_job,
        kind="compiler",
        job_id="development-evidence-compiler-formal-004",
        model_id=None,
    )
    d1_timing_runtime_path = validate_job_receipt(
        d1_timing_job,
        kind="timing",
        job_id="timing-d1-development-sidecar-002",
        model_id="D1",
    )
    n3_timing_runtime_path = validate_job_receipt(
        n3_timing_job,
        kind="timing",
        job_id="timing-n3-development-sidecar-001",
        model_id="N3",
    )

    dependencies = recovery["dependencies"]
    require(type(dependencies) is list and len(dependencies) == 56, "Recovery dependency rows changed")
    dependency_by_id: dict[str, dict] = {}
    dependency_paths: set[str] = set()
    for ordinal, row in enumerate(dependencies):
        require(type(row) is dict, f"Recovery dependency is not an object: {ordinal}")
        dependency_id = row.get("dependency_id")
        require(type(dependency_id) is str and dependency_id not in dependency_by_id, f"Recovery dependency ID changed: {ordinal}")
        descriptor = validate_descriptor(row.get("descriptor"), dependency_id)
        require(descriptor["path"] not in dependency_paths, f"Recovery path duplicated: {dependency_id}")
        dependency_paths.add(descriptor["path"])
        require(row.get("availability") == "declared_pointer_only_current_availability_unknown", f"Recovery availability promoted: {dependency_id}")
        dependency_by_id[dependency_id] = row
    require(set(dependency_by_id) == {f"dependency-{ordinal:02d}" for ordinal in range(56)}, "Recovery dependency ID set changed")

    direct_specs = {
        "compiler.compiler_receipt": ("dependency-33", "compiler_output", None, "compiler_receipt"),
        "D1.request_provenance": ("dependency-35", "compiler_output", None, "primary_compiled_outputs/d1_development_request_provenance.json"),
        "N3.request_provenance": ("dependency-38", "compiler_output", None, "primary_compiled_outputs/n3_development_request_provenance.json"),
        "D1.timing_sidecar": ("dependency-52", "timing_output", "D1", "raw_development_timing_sidecar"),
        "N3.timing_sidecar": ("dependency-54", "timing_output", "N3", "raw_development_timing_sidecar"),
    }
    direct_rows: dict[str, tuple[str, dict]] = {}
    for slot_id, (dependency_id, category, model_id, name) in direct_specs.items():
        row = dependency_by_id[dependency_id]
        require(
            row.get("category") == category
            and row.get("model_id") == model_id
            and row.get("name") == name,
            f"Direct media slot dependency changed: {slot_id}",
        )
        direct_rows[slot_id] = (dependency_id, validate_descriptor(row["descriptor"], slot_id))

    retained_specs = {
        "compiler.job_receipt": (
            RESULTS_COMMIT,
            COMPILER_JOB_RECEIPT_PATH,
            compiler_runtime_path,
        ),
        "D1.timing_job_receipt": (
            RESULTS_COMMIT,
            D1_TIMING_JOB_RECEIPT_PATH,
            d1_timing_runtime_path,
        ),
        "N3.timing_job_receipt": (
            RESULTS_COMMIT,
            N3_TIMING_JOB_RECEIPT_PATH,
            n3_timing_runtime_path,
        ),
    }
    blocked_slots = {
        "D1.alignment_contract",
        "D1.camera_crop_contract",
        "D1.physical_alignment_receipt",
        "N3.alignment_contract",
        "N3.camera_crop_contract",
        "N3.physical_alignment_receipt",
    }
    all_slots = {"compiler.job_receipt", "compiler.compiler_receipt"} | {
        f"{model}.{field}" for model in MODELS for field in MODEL_FIELDS
    }
    require(
        set(direct_specs) | set(retained_specs) | blocked_slots == all_slots
        and not (set(direct_specs) & set(retained_specs))
        and not ((set(direct_specs) | set(retained_specs)) & blocked_slots),
        "Media input slots do not partition exactly",
    )

    slots: list[dict] = []
    for slot_id in sorted(all_slots):
        model_id = slot_id.split(".", 1)[0] if slot_id.split(".", 1)[0] in MODELS else None
        field = slot_id.split(".", 1)[1]
        base = {
            "slot_id": slot_id,
            "model_id": model_id,
            "field": field,
            "runtime_byte_verified": False,
            "semantic_replay_complete": False,
        }
        if slot_id in direct_rows:
            dependency_id, descriptor = direct_rows[slot_id]
            base.update({
                "source_state": "declared_raw_pointer_in_recovery_manifest_current_availability_unknown",
                "lineage_dependency_id": dependency_id,
                "declared_runtime_descriptor": descriptor,
                "retained_git_blob": None,
                "prospective_runtime_descriptor": None,
            })
        elif slot_id in retained_specs:
            commit, path, runtime_path = retained_specs[slot_id]
            payload = blobs[(commit, path)]
            retained = source_descriptor(commit, path, payload)
            prospective = {
                "path": runtime_path,
                "bytes": retained["bytes"],
                "sha256": retained["sha256"],
            }
            validate_descriptor(prospective, f"prospective {slot_id}")
            base.update({
                "source_state": "retained_git_blob_outside_recovery_manifest_runtime_materialization_unverified",
                "lineage_dependency_id": None,
                "declared_runtime_descriptor": None,
                "retained_git_blob": retained,
                "prospective_runtime_descriptor": prospective,
            })
        else:
            base.update({
                "source_state": "no_authenticated_descriptor_in_pinned_sources",
                "lineage_dependency_id": None,
                "declared_runtime_descriptor": None,
                "retained_git_blob": None,
                "prospective_runtime_descriptor": None,
            })
        slots.append(base)

    direct_bytes = sum(row[1]["bytes"] for row in direct_rows.values())
    retained_bytes = sum(len(blobs[(commit, path)]) for commit, path, _ in retained_specs.values())
    require(direct_bytes == 20803049, "Direct media slot byte total changed")
    require(retained_bytes == 122133, "Retained job receipt byte total changed")

    request_rows = lineage["request_lineage"]["rows"]
    require(type(request_rows) is list and len(request_rows) == 1152, "Request lineage rows changed")
    join_records: list[str] = []
    join_keys: set[tuple[str, str, int]] = set()
    per_cell = Counter()
    cells_by_model: dict[str, set[str]] = {model: set() for model in MODELS}
    for ordinal, row in enumerate(request_rows):
        require(type(row) is dict, f"Request lineage row changed: {ordinal}")
        model = row.get("model_id")
        cell_id = row.get("cell_id")
        request_index = row.get("request_index")
        require(model in MODELS and type(cell_id) is str and type(request_index) is int, f"Media join key invalid: {ordinal}")
        key = (model, cell_id, request_index)
        require(key not in join_keys, f"Media join key duplicated: {ordinal}")
        join_keys.add(key)
        per_cell[cell_id] += 1
        cells_by_model[model].add(cell_id)
        join_records.append(f"{model}|{cell_id}|{request_index}")
    require(len(join_keys) == 1152, "Media join key coverage changed")
    require({model: len(cells) for model, cells in cells_by_model.items()} == {"D1": 16, "N3": 16}, "Media join cell coverage changed")
    require(
        sorted(count for cell, count in per_cell.items() if "__D1__" in cell) == [57] * 16
        and sorted(count for cell, count in per_cell.items() if "__N3__" in cell) == [15] * 16,
        "Media join per-cell coverage changed",
    )
    join_index = line_index(join_records, "media join keys", ordered=True)

    direct_dependency_ids = {value[0] for value in direct_rows.values()}
    non_direct_ids = sorted(set(dependency_by_id) - direct_dependency_ids)
    non_direct_index = line_index(non_direct_ids, "non-direct recovery dependencies")

    source_index = line_index(
        [
            f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}"
            for row in sources
        ],
        "source blobs",
    )
    source_index["source_bytes"] = sum(row["bytes"] for row in sources)

    return {
        "schema_version": "wmf-development-annotation-media-input-closure-v1",
        "study_id": "WMF-ABLATION-001",
        "input_pins": {
            "lineage_commit": LINEAGE_COMMIT,
            "readiness_commit": READINESS_COMMIT,
            "status_commit": STATUS_COMMIT,
            "media_contract_commit": MEDIA_COMMIT,
            "results_commit": RESULTS_COMMIT,
        },
        "authenticated_source_blobs": {
            **source_index,
            "sources": sorted(sources, key=lambda row: (row["commit"], row["path"])),
        },
        "runtime_input_contract": {
            **source_contract,
            "raw_root": RAW_ROOT,
            "join_key": input_policy["join_key"],
            "required_top_level_slots": 14,
            "closure_schema_is_not_runtime_input_schema": True,
            "runnable_input_manifest_created": False,
            "queue_descriptor_created": False,
        },
        "media_join_domain": {
            "cells": 32,
            "cells_by_model": {"D1": 16, "N3": 16},
            "requests": 1152,
            "requests_by_model": {"D1": 912, "N3": 240},
            "requests_per_cell": {"D1": 57, "N3": 15},
            "duplicate_join_keys": 0,
            "join_key_index": join_index,
            "lineage_index": lineage["request_lineage"]["index"],
        },
        "top_level_input_closure": {
            "required_slots": 14,
            "exact_identity_bound_slots": 8,
            "recovery_manifest_pointer_slots": 5,
            "retained_git_payload_slots_outside_recovery_manifest": 3,
            "unauthenticated_slots": 6,
            "current_runtime_byte_verified_slots": 0,
            "semantic_replay_complete_slots": 0,
            "recovery_pointer_declared_bytes": direct_bytes,
            "retained_git_payload_bytes": retained_bytes,
            "direct_recovery_dependency_ids": sorted(direct_dependency_ids),
            "unfilled_required_slot_ids": sorted(blocked_slots),
            "slots": slots,
        },
        "recovery_manifest_boundary": {
            "all_declared_dependencies": 56,
            "all_declared_bytes": 32664538,
            "direct_top_level_media_slots": 5,
            "non_direct_top_level_dependencies": len(non_direct_ids),
            "non_direct_dependency_id_index": non_direct_index,
            "compiler_output_inventory_is_not_compiler_job_receipt": True,
            "raw_request_inventories_are_not_timing_job_receipts": True,
            "transitive_dependency_closure_authenticated": False,
            "current_raw_availability_verified": False,
            "raw_paths_opened": 0,
            "raw_bytes_read": 0,
        },
        "contract_expectations_not_observed_results": {
            "episodes": exact_outputs["episodes"],
            "requests": exact_outputs["requests"],
            "source_timing_capable_requests": exact_outputs["source_timing_capable_requests"],
            "d1_forecast_timing_unavailable_requests": exact_outputs["d1_forecast_timing_unavailable_requests"],
            "timing_camera_action_eligible_requests_expected": exact_outputs["timing_camera_action_eligible_requests"],
            "eligible_requests_per_episode_expected": exact_outputs["eligible_requests_per_episode"],
            "selected_requests_expected": exact_outputs["selected_requests"],
            "selected_requests_by_model_expected": exact_outputs["selected_requests_by_model"],
            "request_inclusion_probability_exact_expected": exact_outputs["request_inclusion_probability_exact"],
            "production_eligibility_determined": False,
            "observed_selected_requests": 0,
            "production_selection_manifest_created": False,
        },
        "gate_state": {
            "safe_to_release_new_work": False,
            "current_GM_PVC_worker_state_reconciled": False,
            "camera_alignment_ready": False,
            "annotation_media_ready": False,
            "human_annotation_ready": False,
            "resource_freeze_ready": False,
            "confirmation_release_ready": False,
            "core_episode_ceiling": 232,
        },
        "new_activity": {
            "raw_paths_opened": 0,
            "raw_bytes_read": 0,
            "queue_descriptors": 0,
            "retrieval_jobs": 0,
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "request_selections": 0,
            "annotation_media": 0,
            "human_responses": 0,
            "labels": 0,
            "forecast_accuracy_estimates": 0,
            "confirmation_inference_cells": 0,
        },
        "claim_boundaries": {
            "artifact": "This is a retained-Git top-level input-closure ledger, not a runnable annotation-media input manifest or queue descriptor.",
            "identity": "Five exact raw-pointer identities and three exact retained Git payloads bind eight of fourteen top-level slots; no current runtime placement or availability is inferred.",
            "missing": "Six slots lack authenticated descriptors in the pinned sources. That is an evidence-closure statement, not proof that files are absent from every external system.",
            "transitive": "The 56-file recovery set does not establish the nested payload closure required by request provenance, timing validation, camera replay, or rendering.",
            "annotation": "Contract eligibility and selection counts are expectations only. No eligibility, selection, media, rater handoff, response, label, consensus, threshold or usability result was created.",
            "operations": "Retained Git evidence does not establish current GM, PVC, queue, worker, GPU, child, reap, lock or live-quiescence state.",
            "release": "safe_to_release_new_work remains false; no descriptor, retrieval job, semantic replay, scientific cell or confirmation work was released.",
            "science": "No policy skill, forecast accuracy, model comparison, resource qualification or confirmation result is inferred; the 232-cell ceiling is unchanged.",
        },
    }


def audit() -> dict:
    return build_report(load_pinned_blobs())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical closure report")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved media-input closure report differs")
        print(
            "Verified 14 top-level media-input slots: five recovery pointers, "
            "three retained Git payloads, six unauthenticated; no release."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
