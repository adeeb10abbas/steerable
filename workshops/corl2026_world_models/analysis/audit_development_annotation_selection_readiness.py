#!/usr/bin/env python3
"""Audit development annotation-selection readiness from pinned Git blobs only.

This audit deliberately does not open raw paths, rank requests, decide eligibility,
or create a selection manifest.  It cross-binds the frozen annotation workflow to
the retained development request-lineage schema and reports the exact metadata gap.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[3]

SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
WORKFLOW_COMMIT = "fad1441a5e30b6a3c0895f26413fa9d6978835f3"
RECOVERY_COMMIT = "296fefc0ad14b3320d7c45a8a11fde6f0908deb6"

SPEC_PATH = "workshops/corl2026_world_models/experiments/forecast_layout/ablation_spec.json"
WORKFLOW_PATH = "workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py"
RECOVERY_PATH = "workshops/corl2026_world_models/results/development_request_recovery_lineage.json"

EXPECTED_SOURCES = {
    (SPEC_COMMIT, SPEC_PATH): (
        8590,
        "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
    ),
    (WORKFLOW_COMMIT, WORKFLOW_PATH): (
        158628,
        "483fbb64059ee0e76932e6ba6904bbf850710c17093b54fd45b38474109e12e8",
    ),
    (RECOVERY_COMMIT, RECOVERY_PATH): (
        573283,
        "5714efd75bde846c41d1bb3f0ec88df427d3b18bf8c0df62633b6b38b8a95878",
    ),
}

EXPECTED_REQUEST_KEYS = {
    "action_manifest_sha256",
    "alignment_contract_id",
    "alignment_contract_sha256",
    "alignment_receipt_id",
    "alignment_receipt_sha256",
    "camera_crop_id",
    "camera_crop_sha256",
    "camera_id",
    "camera_identity_match",
    "cell_id",
    "condition_id",
    "early_horizon_supported",
    "episode_id",
    "executed_prefix_actions",
    "generated_frame_index",
    "history_mode",
    "layout_pair_id",
    "model_id",
    "request_index",
    "request_start_action_index",
    "source_request_id",
    "source_video_id",
    "source_video_sha256",
    "target_executed_action_offset",
    "target_physical_time_s",
    "target_within_executed_prefix",
    "technical_invalid_reason",
    "technical_valid",
    "timestamp_error_s",
    "timestamp_tolerance_s",
}
EXPECTED_RETAINED_ROW_KEYS = {
    "cell_id",
    "layout_pair_id",
    "model_id",
    "model_request_ordinal",
    "request_index",
    "scheduled_condition",
    "scheduled_ordinal",
    "source_request_receipt_sha256",
}
EXPECTED_ELIGIBILITY_FIELDS = {
    "camera_identity_match",
    "target_within_executed_prefix",
    "technical_valid",
    "timestamp_error_s",
    "timestamp_tolerance_s",
}
EXPECTED_SAMPLING_FIELDS = {"episode_id", "source_request_id"}
EXPECTED_SEED = 2026091302
EXPECTED_CAP = 4
EXPECTED_ALGORITHM = (
    "Within each episode, sort eligible source_request_id values by "
    "SHA256(UTF8(seed) || NUL || UTF8(episode_id) || NUL || "
    "UTF8(source_request_id)), breaking digest ties by source_request_id; "
    "select the first min(4, eligible_count)."
)
EXPECTED_ELIGIBILITY_FUNCTION_SHA256 = (
    "219ed71740c4b87613030d436153a6ebd35278596ad94e1166808629cc700c6b"
)
EXPECTED_SELECTION_FUNCTION_SHA256 = (
    "3c855e540147cd1611f0ea7751b69bda063a50993bcf8fa7904720aa166f001e"
)

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
EXPECTED_TIMING_BOUNDARY = {
    "D1_action_prefix_truncated_requests": 16,
    "D1_native_clock_matched_requests": 224,
    "D1_physical_time_coverage_complete": False,
    "D1_retained_request_identities": 912,
    "D1_source_proven_full_decode_requests": 240,
    "D1_source_unmapped_incremental_decode_requests": 672,
    "N3_retained_request_identities": 240,
    "all_request_identities_mapped_to_cells": True,
    "all_requests_timing_complete": False,
}
EXPECTED_ANNOTATION_BOUNDARY = {
    "annotation_media_created": 0,
    "camera_crop_contract_authenticated": False,
    "confirmation_inference_cells": 0,
    "forecast_accuracy_estimates": 0,
    "human_handoff_ready": False,
    "human_responses": 0,
    "labels": 0,
    "measured_resource_freeze_authenticated": False,
    "physical_alignment_authenticated": False,
    "production_eligibility_determined": False,
    "production_request_selection_manifest_created": False,
    "safe_to_release_confirmation": False,
    "selected_request_count": 0,
}
EXPECTED_GATE_STATE = {
    "annotation_media_ready": False,
    "camera_alignment_ready": False,
    "compiler_ready": True,
    "confirmation_release_ready": False,
    "development_annotation_gate_ready": False,
    "human_annotation_ready": False,
    "resource_freeze_ready": False,
    "timing_partially_ready": True,
}
ZERO_ACTIVITY = {
    "annotation_media": 0,
    "behavioral_actions": 0,
    "behavioral_cells": 0,
    "forecast_accuracy_estimates": 0,
    "human_responses": 0,
    "labels": 0,
    "model_requests": 0,
    "request_selections": 0,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def load_json(payload: bytes, source: str) -> dict[str, Any]:
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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def require_exact_json(value: object, expected: object, label: str) -> None:
    require(canonical_bytes(value) == canonical_bytes(expected), f"{label} changed")


def require_sha256(value: object, label: str) -> str:
    require(type(value) is str and len(value) == 64, f"Invalid SHA-256: {label}")
    require(
        all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


def line_index(lines: list[str], *, ordered: bool = False) -> dict[str, object]:
    material = list(lines) if ordered else sorted(lines)
    require(len(material) == len(set(material)), "Index rows are not unique")
    payload = "".join(f"{line}\n" for line in material).encode("utf-8")
    return {
        "canonical_bytes": len(payload),
        "canonicalization": (
            "ordered UTF-8 rows with LF terminators"
            if ordered
            else "lexicographically sorted UTF-8 rows with LF terminators"
        ),
        "count": len(material),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


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


def source_blob(commit: str, path: str) -> tuple[bytes, dict[str, object]]:
    payload = git_blob(commit, path)
    expected_bytes, expected_sha256 = EXPECTED_SOURCES[(commit, path)]
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    require(len(payload) == expected_bytes, f"Source byte count changed: {path}")
    require(actual_sha256 == expected_sha256, f"Source SHA-256 changed: {path}")
    return payload, {
        "bytes": len(payload),
        "commit": commit,
        "path": path,
        "sha256": actual_sha256,
    }


def literal_assignment(tree: ast.Module, name: str) -> object:
    matches: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                matches.append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                matches.append(node.value)
    require(len(matches) == 1, f"Expected one top-level assignment: {name}")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise ValueError(f"Assignment is not a literal: {name}") from error


def function_contract(
    source: str, tree: ast.Module, name: str, expected_sha256: str
) -> tuple[ast.FunctionDef, dict[str, object], str]:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    require(len(matches) == 1, f"Expected one top-level function: {name}")
    node = matches[0]
    segment = ast.get_source_segment(source, node)
    require(segment is not None, f"Could not recover function source: {name}")
    payload = segment.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_sha256, f"Workflow function changed: {name}")
    return node, {
        "bytes": len(payload),
        "end_line": node.end_lineno,
        "sha256": digest,
        "start_line": node.lineno,
    }, segment


def row_subscript_fields(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.Subscript):
            continue
        if not isinstance(candidate.value, ast.Name) or candidate.value.id != "row":
            continue
        if isinstance(candidate.slice, ast.Constant) and isinstance(
            candidate.slice.value, str
        ):
            result.add(candidate.slice.value)
    return result


def parse_workflow_contract(payload: bytes) -> dict[str, object]:
    try:
        source = payload.decode("utf-8")
        tree = ast.parse(source, filename=WORKFLOW_PATH)
    except (UnicodeDecodeError, SyntaxError) as error:
        raise ValueError("Pinned annotation workflow is not valid UTF-8 Python") from error

    request_keys = literal_assignment(tree, "REQUEST_KEYS")
    seed = literal_assignment(tree, "REQUEST_SAMPLING_SEED")
    cap = literal_assignment(tree, "REQUEST_SAMPLE_CAP")
    algorithm = literal_assignment(tree, "REQUEST_SAMPLING_ALGORITHM")
    require(request_keys == EXPECTED_REQUEST_KEYS, "Annotation request schema changed")
    require(seed == EXPECTED_SEED, "Request sampling seed changed")
    require(cap == EXPECTED_CAP, "Request sample cap changed")
    require(algorithm == EXPECTED_ALGORITHM, "Request sampling algorithm changed")

    eligibility_node, eligibility_descriptor, _ = function_contract(
        source,
        tree,
        "_eligibility_reasons",
        EXPECTED_ELIGIBILITY_FUNCTION_SHA256,
    )
    selection_node, selection_descriptor, selection_source = function_contract(
        source,
        tree,
        "select_requests",
        EXPECTED_SELECTION_FUNCTION_SHA256,
    )
    eligibility_fields = row_subscript_fields(eligibility_node)
    require(
        eligibility_fields == EXPECTED_ELIGIBILITY_FIELDS,
        "Eligibility metadata fields changed",
    )
    selection_snippets = (
        'require(inventory.get("inventory_complete") is True',
        'require(inventory.get("annotation_state") == "not_started"',
        'eligible = [row for row in candidates if not _eligibility_reasons(row)]',
        'payload = f"{seed}\\0{episode_id}\\0{row[\'source_request_id\']}".encode("utf-8")',
        "for row in sorted(eligible, key=rank)[: min(cap_per_episode, len(eligible))]",
        '"selection_uses_object_visibility_or_forecast_quality": False',
    )
    for snippet in selection_snippets:
        require(snippet in selection_source, f"Selection invariant changed: {snippet}")
    require(
        "source_request_id" in row_subscript_fields(selection_node),
        "Selection request identity field changed",
    )

    return {
        "annotation_request_schema_field_count": len(request_keys),
        "annotation_request_schema_fields": sorted(request_keys),
        "eligibility_function": eligibility_descriptor,
        "eligibility_metadata_fields": sorted(eligibility_fields),
        "request_sample_cap_per_episode": cap,
        "request_sampling_algorithm": algorithm,
        "request_sampling_seed": seed,
        "sampling_identity_components": ["seed", "episode_id", "source_request_id"],
        "selection_function": selection_descriptor,
        "selection_requires_complete_inventory": True,
        "selection_requires_not_started_annotation_state": True,
        "selection_uses_object_visibility_or_forecast_quality": False,
    }


def validate_spec(spec: dict[str, Any]) -> dict[str, object]:
    require(spec.get("spec_id") == "WMF-ABLATION-001", "Study changed")
    require(spec.get("core_episode_ceiling") == 232, "Core ceiling changed")
    stages = spec.get("stages")
    require(type(stages) is dict, "Stage design is missing")
    expected_stage_counts = {"pilot": 8, "development": 32, "confirmation": 192}
    for stage, expected in expected_stage_counts.items():
        require(stages[stage]["episodes"] == expected, f"{stage} count changed")
    require(sum(expected_stage_counts.values()) == 232, "Core stage sum changed")
    require(spec["models"]["D2"]["selected"] is False, "D2 selection changed")
    require(
        spec.get("counts_do_not_authorize_execution") is True,
        "Count authorization boundary changed",
    )
    analysis = spec.get("analysis")
    require(analysis["request_sampling_seed"] == EXPECTED_SEED, "Spec seed changed")
    require(
        analysis["request_sample_cap_per_episode"] == EXPECTED_CAP,
        "Spec cap changed",
    )
    require(
        analysis["request_manifest_timing"]
        == "algorithm and seed frozen before confirmation; selection manifest generated from actual metadata after execution and before labels",
        "Selection timing boundary changed",
    )
    require(
        analysis["sampling"]
        == "uniform from time/action/camera-qualified metadata before labels; unclear images never replaced",
        "Sampling boundary changed",
    )
    require(spec["annotation"]["raters"] == 2, "Rater count changed")
    require_exact_json(
        spec["post_execution_pre_labeling_requirements"],
        ["actual_request_inventory", "metadata_only_request_selection_manifest"],
        "Post-execution requirements",
    )
    return {
        "confirmation_cells": 192,
        "core_cell_ceiling": 232,
        "counts_authorize_execution": False,
        "development_cells": 32,
        "pilot_cells": 8,
        "raters": 2,
        "selection_manifest_timing": analysis["request_manifest_timing"],
        "selection_metadata_policy": analysis["sampling"],
    }


def validate_recovery(report: dict[str, Any]) -> dict[str, object]:
    require(
        report.get("schema_version") == "wmf-development-request-recovery-lineage-v1",
        "Recovery schema changed",
    )
    require(report.get("study_id") == "WMF-ABLATION-001", "Recovery study changed")
    require_exact_json(
        report.get("global_study_snapshot"),
        EXPECTED_GLOBAL_SNAPSHOT,
        "Global study snapshot",
    )
    snapshot = report["global_study_snapshot"]
    require(snapshot["launched_core_cells"] == 40 + 4, "44-cell accounting changed")
    require(snapshot["terminal_valid_cells"] == 40, "Valid-cell count changed")
    require(
        snapshot["terminal_requests"] == 18 + 1440 + 2,
        "Request accounting changed",
    )
    require(snapshot["active_launched_cells"] == 0, "Active-cell count changed")
    require(
        snapshot["confirmation_inference_cells"] == 0,
        "Confirmation count changed",
    )
    require_exact_json(
        report.get("timing_boundary"), EXPECTED_TIMING_BOUNDARY, "Timing boundary"
    )
    require_exact_json(
        report.get("annotation_and_release_boundary"),
        EXPECTED_ANNOTATION_BOUNDARY,
        "Annotation boundary",
    )
    require_exact_json(report.get("gate_state"), EXPECTED_GATE_STATE, "Gate state")
    require_exact_json(report.get("new_activity"), ZERO_ACTIVITY, "New activity")

    pending = report.get("pending_raw_verification")
    require(type(pending) is dict, "Pending raw boundary is missing")
    require(
        pending["raw_paths_opened"] == 0
        and pending["raw_bytes_read"] == 0
        and pending["raw_descriptor_count_verified"] is None
        and pending["raw_descriptor_mismatch_count"] is None
        and pending["current_raw_availability_verified"] is False
        and pending["retrieval_released"] is False
        and pending["retrieval_queue_descriptor_created"] is False
        and pending["safe_to_release_new_work"] is False,
        "Raw or release boundary changed",
    )

    lineage = report.get("request_lineage")
    require(type(lineage) is dict, "Request lineage is missing")
    require(lineage["requests"] == 1152 and lineage["cells"] == 32, "Population changed")
    require_exact_json(
        lineage["requests_by_model"], {"D1": 912, "N3": 240}, "Model counts"
    )
    require_exact_json(
        lineage["requests_per_cell"], {"D1": 57, "N3": 15}, "Cell counts"
    )
    require(lineage["cross_model_identity_overlap"] == 0, "Identity overlap changed")
    rows = lineage.get("rows")
    require(type(rows) is list and len(rows) == 1152, "Lineage rows changed")
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_counts: Counter[str] = Counter()
    mapping_keys: set[tuple[str, int]] = set()
    receipt_hashes: set[str] = set()
    for ordinal, row in enumerate(rows):
        require(type(row) is dict, f"Lineage row {ordinal} is not an object")
        require(set(row) == EXPECTED_RETAINED_ROW_KEYS, f"Lineage row {ordinal} schema changed")
        require(row["model_id"] in {"D1", "N3"}, f"Lineage row {ordinal} model changed")
        require(type(row["cell_id"]) is str and row["cell_id"], f"Lineage row {ordinal} cell is invalid")
        require(type(row["request_index"]) is int, f"Lineage row {ordinal} request index is invalid")
        receipt_hash = require_sha256(
            row["source_request_receipt_sha256"], f"lineage row {ordinal} receipt"
        )
        key = (row["cell_id"], row["request_index"])
        require(key not in mapping_keys, f"Duplicate lineage mapping key: {key}")
        require(receipt_hash not in receipt_hashes, "Duplicate declared request receipt hash")
        mapping_keys.add(key)
        receipt_hashes.add(receipt_hash)
        by_cell[row["cell_id"]].append(row)
        model_counts[row["model_id"]] += 1
    require(model_counts == Counter({"D1": 912, "N3": 240}), "Model row counts changed")
    require(len(by_cell) == 32, "Cell count changed")
    model_cell_counts: Counter[str] = Counter()
    for cell_id, cell_rows in by_cell.items():
        models = {row["model_id"] for row in cell_rows}
        require(len(models) == 1, f"Cell mixes models: {cell_id}")
        model = next(iter(models))
        expected = 57 if model == "D1" else 15
        require(len(cell_rows) == expected, f"Cell request count changed: {cell_id}")
        require(
            sorted(row["request_index"] for row in cell_rows) == list(range(expected)),
            f"Cell request indexes changed: {cell_id}",
        )
        model_cell_counts[model] += 1
    require(model_cell_counts == Counter({"D1": 16, "N3": 16}), "Model cell counts changed")
    return {
        "cells": 32,
        "cells_by_model": {"D1": 16, "N3": 16},
        "declared_request_mapping_authenticated": True,
        "mapping_index": lineage["index"],
        "mapping_index_reconstruction": lineage["mapping_index_reconstruction"],
        "request_identity_index": lineage["request_identity_index"],
        "retained_row_fields": sorted(EXPECTED_RETAINED_ROW_KEYS),
        "rows": 1152,
        "rows_by_model": {"D1": 912, "N3": 240},
        "rows_per_cell": {"D1": 57, "N3": 15},
    }


def validate_output_contract(report: dict[str, Any]) -> None:
    require(report.get("schema_version") == "wmf-development-annotation-selection-readiness-v1", "Output schema changed")
    require(report.get("study_id") == "WMF-ABLATION-001", "Output study changed")
    require_exact_json(report["global_study_snapshot"], EXPECTED_GLOBAL_SNAPSHOT, "Output study snapshot")
    require_exact_json(report["timing_boundary"], EXPECTED_TIMING_BOUNDARY, "Output timing boundary")
    require_exact_json(report["annotation_and_release_boundary"], EXPECTED_ANNOTATION_BOUNDARY, "Output annotation boundary")
    require_exact_json(report["gate_state"], EXPECTED_GATE_STATE, "Output gates")
    require_exact_json(report["new_activity"], ZERO_ACTIVITY, "Output activity")
    readiness = report["selection_readiness"]
    require(readiness["sampling_key_feasible_rows"] == 0, "Sampling feasibility promoted")
    require(readiness["eligibility_decision_feasible_rows"] == 0, "Eligibility feasibility promoted")
    require(readiness["production_eligibility_determined"] is False, "Eligibility gate promoted")
    require(readiness["selection_manifest_created"] is False, "Selection manifest promoted")
    require(readiness["selected_request_count"] == 0, "Selection count promoted")
    audit_activity = report["audit_activity"]
    require(all(value == 0 for value in audit_activity.values()), "Audit created prohibited activity")
    serialized = canonical_bytes(report)
    require(b"/data/" not in serialized, "Output leaked or materialized a raw path")


def build_audit() -> dict[str, Any]:
    spec_payload, spec_source = source_blob(SPEC_COMMIT, SPEC_PATH)
    workflow_payload, workflow_source = source_blob(WORKFLOW_COMMIT, WORKFLOW_PATH)
    recovery_payload, recovery_source = source_blob(RECOVERY_COMMIT, RECOVERY_PATH)
    sources = [spec_source, workflow_source, recovery_source]

    spec = load_json(spec_payload, f"{SPEC_COMMIT}:{SPEC_PATH}")
    recovery = load_json(recovery_payload, f"{RECOVERY_COMMIT}:{RECOVERY_PATH}")
    design_contract = validate_spec(spec)
    workflow_contract = parse_workflow_contract(workflow_payload)
    retained_population = validate_recovery(recovery)
    require(
        workflow_contract["request_sampling_seed"]
        == spec["analysis"]["request_sampling_seed"],
        "Workflow and specification seeds differ",
    )
    require(
        workflow_contract["request_sample_cap_per_episode"]
        == spec["analysis"]["request_sample_cap_per_episode"],
        "Workflow and specification caps differ",
    )

    retained_fields = set(retained_population["retained_row_fields"])
    exact_available = EXPECTED_REQUEST_KEYS & retained_fields
    exact_missing = EXPECTED_REQUEST_KEYS - retained_fields
    require(
        exact_available == {"cell_id", "layout_pair_id", "model_id", "request_index"},
        "Exact annotation-field overlap changed",
    )
    require(len(exact_missing) == 26, "Missing annotation-field count changed")
    require(not (EXPECTED_SAMPLING_FIELDS & retained_fields), "Sampling identity unexpectedly became available")
    require(not (EXPECTED_ELIGIBILITY_FIELDS & retained_fields), "Eligibility metadata unexpectedly became available")

    source_index = line_index(
        [
            f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}"
            for row in sources
        ]
    )
    selected_ceiling = design_contract["development_cells"] * EXPECTED_CAP
    raters = design_contract["raters"]
    require(selected_ceiling == 128 and raters == 2, "Development workload basis changed")

    report = {
        "schema_version": "wmf-development-annotation-selection-readiness-v1",
        "study_id": "WMF-ABLATION-001",
        "input_pins": {
            "recovery_lineage_commit": RECOVERY_COMMIT,
            "spec_commit": SPEC_COMMIT,
            "workflow_commit": WORKFLOW_COMMIT,
        },
        "authenticated_source_blobs": {
            **source_index,
            "source_bytes": sum(row["bytes"] for row in sources),
            "sources": sorted(sources, key=lambda row: (row["commit"], row["path"])),
        },
        "design_contract": design_contract,
        "workflow_contract": workflow_contract,
        "global_study_snapshot": EXPECTED_GLOBAL_SNAPSHOT,
        "retained_population": {
            **retained_population,
            "annotation_request_schema_field_count": len(EXPECTED_REQUEST_KEYS),
            "exact_annotation_schema_fields_available": sorted(exact_available),
            "exact_annotation_schema_fields_available_count": len(exact_available),
            "exact_annotation_schema_fields_missing": sorted(exact_missing),
            "exact_annotation_schema_fields_missing_count": len(exact_missing),
            "exact_annotation_schema_coverage_complete": False,
        },
        "selection_readiness": {
            "actual_request_inventory_authenticated": False,
            "action_manifests_authenticated": False,
            "alignment_contracts_authenticated": False,
            "eligibility_decision_feasible_rows": 0,
            "eligibility_decision_required_fields": sorted(EXPECTED_ELIGIBILITY_FIELDS),
            "eligibility_decision_status": "undetermined_not_ineligible",
            "eligibility_rows_emitted": 0,
            "episode_roster_authenticated": False,
            "production_eligibility_determined": False,
            "sampling_key_feasible_rows": 0,
            "sampling_key_required_fields": sorted(EXPECTED_SAMPLING_FIELDS),
            "sampling_ranks_emitted": 0,
            "selected_request_count": 0,
            "selection_manifest_created": False,
            "total_retained_rows": 1152,
            "boundary": (
                "All 1,152 retained rows have an authenticated declared cell/index/receipt-hash "
                "mapping, but zero rows have the authenticated episode_id plus source_request_id "
                "needed for the frozen rank or all five metadata fields needed for eligibility. "
                "The rows are undetermined, not ineligible."
            ),
        },
        "forbidden_identity_substitutions": [
            {
                "required_field": "source_request_id",
                "retained_fields_that_are_not_equivalent": [
                    "model_request_ordinal",
                    "request_index",
                    "source_request_receipt_sha256",
                ],
            },
            {
                "required_field": "episode_id",
                "retained_fields_that_are_not_equivalent": ["cell_id"],
            },
            {
                "required_field": "condition_id",
                "retained_fields_that_are_not_equivalent": ["scheduled_condition"],
            },
        ],
        "annotation_workload_ceilings": {
            "boundary": (
                "Arithmetic ceilings only; these are not selections, media, responses, labels, "
                "or completed workload, and A9 applies only where an earlier horizon is supported."
            ),
            "development_episode_count": 32,
            "maximum_selected_requests": selected_ceiling,
            "primary_current_predicted_executed": {
                "images": selected_ceiling * 3,
                "two_rater_judgments": selected_ceiling * 3 * raters,
            },
            "request_sample_cap_per_episode": EXPECTED_CAP,
            "with_A4_preceding": {
                "images": selected_ceiling * 4,
                "two_rater_judgments": selected_ceiling * 4 * raters,
            },
            "with_A4_and_A9_if_supported": {
                "images": selected_ceiling * 6,
                "two_rater_judgments": selected_ceiling * 6 * raters,
            },
        },
        "timing_boundary": EXPECTED_TIMING_BOUNDARY,
        "annotation_and_release_boundary": EXPECTED_ANNOTATION_BOUNDARY,
        "gate_state": EXPECTED_GATE_STATE,
        "new_activity": ZERO_ACTIVITY,
        "audit_activity": {
            "eligibility_decisions": 0,
            "external_system_reads": 0,
            "labels_created": 0,
            "raw_bytes_read": 0,
            "raw_paths_opened": 0,
            "sampling_ranks_created": 0,
            "selection_manifests_created": 0,
        },
        "negative_invariants": [
            "reject any sampling seed, cap, algorithm, workflow function, or request schema drift",
            "reject any source_request_id or episode_id fallback to receipt hash, ordinal, index, cell, or condition",
            "reject eligibility or selection before a complete authenticated actual request inventory",
            "reject any use of object visibility, forecast quality, outcomes, or labels in eligibility or sampling",
            "reject altered 232/44/40+4/1460/18+1440+2/18000/18040/0-of-192 accounting",
            "reject request population drift, duplicate cell/index mappings, duplicate receipt hashes, or cross-model overlap",
            "reject row-level D1 timing-class inference and any N3 or all-request timing-completeness promotion",
            "reject any raw, camera, selection, media, human, resource-freeze, confirmation, or release gate promotion",
            "reject any new model request, action, cell, selection, media, response, label, or estimate",
        ],
        "claim_boundaries": {
            "lineage": (
                "The audit authenticates the declared 1,152-row/32-cell recovery-lineage population "
                "and its exact schema; it does not authenticate request content."
            ),
            "identity": (
                "Receipt hashes, request indexes, model ordinals, cell IDs, and scheduled conditions "
                "are not substitutes for source_request_id, episode_id, or condition_id."
            ),
            "timing": (
                "D1 remains aggregate-only: 224 native-clock matched plus 16 prefix-truncated "
                "among 240 source-proven requests, with 672 source-unmapped incremental decodes. "
                "N3 remains 240 retained identities without a broader timing-completeness claim."
            ),
            "annotation": (
                "No eligibility boolean, exclusion reason, sampling rank, selected flag, media path, "
                "response, label, consensus, threshold, usability decision, or estimate was emitted."
            ),
            "science": (
                "This is a preregistration/readiness receipt only. It does not establish policy skill, "
                "forecast accuracy, model comparison, or a confirmation result."
            ),
            "release": (
                "The audit cannot authorize raw retrieval, annotation packaging, confirmation, or any "
                "queue/release action; every scientific gate and the 232-cell ceiling are unchanged."
            ),
        },
    }
    validate_output_contract(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an exact canonical report")
    parser.add_argument("--output", type=Path, help="Write the canonical report")
    args = parser.parse_args()
    require(not (args.check and args.output), "Use only one of --check and --output")

    payload = canonical_bytes(build_audit())
    if args.check is not None:
        require(args.check.read_bytes() == payload, "Saved report is not canonical or current")
    elif args.output is not None:
        args.output.write_bytes(payload)
    else:
        sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    main()
