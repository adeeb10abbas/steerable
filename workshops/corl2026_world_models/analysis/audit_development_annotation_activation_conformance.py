#!/usr/bin/env python3
"""Authenticate the frozen development-annotation activation path without activating it.

The audit reads only commit-addressed Git blobs.  It never imports or invokes the
media builder, assembler, compiler, selection workflow, producer programs, or any
cluster/runtime interface.  Its positive claim is deliberately narrow: the frozen
builder's 30-field dependency graph is fully accounted for.  Field values, runtime
availability, eligibility, selection, media, labels, and release remain unproved.
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
SELECTION_COMMIT = "393982967660f490717576fb9bf82033dbc0753e"
LINEAGE_COMMIT = "296fefc0ad14b3320d7c45a8a11fde6f0908deb6"
ALIGNMENT_COMMIT = "a38b73ca539fe3cc7618fb1294b67b0e24861b58"
CROP_COMMIT = "93993dae1902ba561b594b9ae0849bd45b37ecc2"

SPEC_PATH = "workshops/corl2026_world_models/docs/ABLATION_SPEC.md"
SELECTION_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_selection_readiness.json"
)
MEDIA_CLOSURE_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_media_input_closure.json"
)
LINEAGE_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_request_recovery_lineage.json"
)
WORKFLOW_PATH = (
    "workshops/corl2026_world_models/analysis/forecast_annotation_workflow.py"
)
QUEUE_SUPPORT_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "forecast_timing_queue_jobs.py"
)
TIMING_VALIDATOR_PATH = (
    "workshops/corl2026_world_models/analysis/qualify_forecast_timing.py"
)
FREEZE_VALIDATOR_PATH = (
    "workshops/corl2026_world_models/analysis/freeze_development_release.py"
)
CAMERA_REPLAY_PATH = (
    "workshops/corl2026_world_models/analysis/camera_crop_replay_witness.py"
)
MEDIA_CONTRACT_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_annotation_media_bridge_contract.json"
)
COMPILER_PATH = (
    "workshops/corl2026_world_models/analysis/compile_development_evidence.py"
)
ASSEMBLER_PATH = (
    "workshops/corl2026_world_models/analysis/prepare_development_annotation_media.py"
)
MEDIA_JOBS_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_annotation_media_jobs.py"
)
ALIGNMENT_JOBS_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_physical_alignment_jobs.py"
)
CROP_JOBS_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "camera_crop_witness_jobs.py"
)

EXPECTED_SOURCES: dict[tuple[str, str], tuple[int, str]] = {
    (SPEC_COMMIT, SPEC_PATH): (
        27094,
        "fa8cbbac7bf2eff2e01f73150e5cf073f7e4f177d3c0bfd4fc1f41fb3ea9309f",
    ),
    (SELECTION_COMMIT, SELECTION_PATH): (
        13980,
        "1ecd3be3f4a26483581454a625db17da302dc66bbbf558286dd24c30d83a70f0",
    ),
    (SELECTION_COMMIT, MEDIA_CLOSURE_PATH): (
        19058,
        "762e95ce616ecc42486f382792ad4216af54715173ccffe3af353bd21ed8bf48",
    ),
    (SELECTION_COMMIT, MEDIA_JOBS_PATH): (
        76709,
        "13d1bbbbd19e3ded30d3bc9d27f1e8a909d478b691afa320bc7e6c68a5425190",
    ),
    (SELECTION_COMMIT, QUEUE_SUPPORT_PATH): (
        86978,
        "d448a02ad627bc65c143a3932ae2a4378be642e36a49645f5dc8c3727f00c353",
    ),
    (SELECTION_COMMIT, ASSEMBLER_PATH): (
        152381,
        "5fc8f03d31fceb3609786c97cb8dcdc9ed2299586be5eede6e15539a1ac58d32",
    ),
    (SELECTION_COMMIT, COMPILER_PATH): (
        159374,
        "e8dc390f47925b8ec4e06555a417bf80f662a46d718a2ee991c930e80f013deb",
    ),
    (SELECTION_COMMIT, TIMING_VALIDATOR_PATH): (
        143587,
        "4d8994d75fd0fe8fd9178bad1f88efad010b067a90ef97352bd9ca4ab65f49ff",
    ),
    (SELECTION_COMMIT, WORKFLOW_PATH): (
        158628,
        "483fbb64059ee0e76932e6ba6904bbf850710c17093b54fd45b38474109e12e8",
    ),
    (SELECTION_COMMIT, FREEZE_VALIDATOR_PATH): (
        91793,
        "1dbba3d3afb736e178782d39e0eaea9b451b2ca196bb6d6a94e80461bcd39e44",
    ),
    (SELECTION_COMMIT, CAMERA_REPLAY_PATH): (
        64040,
        "30d13920383ea36d26a8a3208685ab057fb070a39e520fcdc6caa4e8449a9f4f",
    ),
    (SELECTION_COMMIT, MEDIA_CONTRACT_PATH): (
        3498,
        "893996c8157dad803c1bd57e57b3922f4c340a66846aa457e0d6bf27c1491d39",
    ),
    (LINEAGE_COMMIT, LINEAGE_PATH): (
        573283,
        "5714efd75bde846c41d1bb3f0ec88df427d3b18bf8c0df62633b6b38b8a95878",
    ),
    (ALIGNMENT_COMMIT, ALIGNMENT_JOBS_PATH): (
        90462,
        "b805df1e251ca848a82a7ff0c39043d40154ce1030dd7db2e68b30b1ed09c44a",
    ),
    (CROP_COMMIT, CROP_JOBS_PATH): (
        51025,
        "6613de67a170d1719d5b4e1a5be1e2b9305e1a4b6087e8f1d2006f3e8b0f3145",
    ),
}

EXPECTED_FUNCTIONS: dict[tuple[str, str], tuple[int, int, int, str]] = {
    (WORKFLOW_PATH, "_eligibility_reasons"): (
        568, 582, 604,
        "219ed71740c4b87613030d436153a6ebd35278596ad94e1166808629cc700c6b",
    ),
    (WORKFLOW_PATH, "select_requests"): (
        865, 1096, 12411,
        "3c855e540147cd1611f0ea7751b69bda063a50993bcf8fa7904720aa166f001e",
    ),
    (MEDIA_JOBS_PATH, "validate_input_shape"): (
        274, 324, 2505,
        "f72824e3edd3ceec2e0f29470ad8d109c37599da39b990e199b4f31b6c7b7774",
    ),
    (MEDIA_JOBS_PATH, "_source_paths"): (
        149, 160, 522,
        "09408b7cc1ce8a7a8fe8ea3c1fd19aef63bd2689f34cd778063087af855a6a59",
    ),
    (MEDIA_JOBS_PATH, "_descriptor_shape"): (
        258, 271, 853,
        "5f5c54b61c380a9c056c6b75f00177b40d798c71d47cdfb58b97ad278d395837",
    ),
    (MEDIA_JOBS_PATH, "_job_descriptor"): (
        336, 366, 1149,
        "18cffa020338e6630e6321c9947f0c6018c4e0b81f11bca0810af470f36f94ce",
    ),
    (MEDIA_JOBS_PATH, "build_formal_wave"): (
        369, 397, 1203,
        "e8b28981134503f07e1121d0baa53ea505fdd3ea9c80e0f22662493737dc6e7c",
    ),
    (ASSEMBLER_PATH, "_build_request_inventory"): (
        1522, 1672, 8776,
        "e1a56b3cccadb7347b9e6ca5e30dcb5d5fa609cadecd171b72cd48fe14d3023c",
    ),
    (ASSEMBLER_PATH, "_assert_inventory_and_selection"): (
        1675, 1818, 7889,
        "65d918a86c82f3dc11d5bf83f92c0d2f452224a2526a2f9fbe1a2e259cba3255",
    ),
    (ASSEMBLER_PATH, "prepare"): (
        2720, 3121, 19818,
        "42144746ac1cc98a251e20f1c29a0dd7932bea16ef872550c646ecae278e6eb0",
    ),
}

RETAINED_FIELDS = (
    "cell_id", "layout_pair_id", "model_id", "request_index",
)
REQUEST_PROVENANCE_FIELDS = (
    "action_manifest_sha256", "condition_id", "episode_id",
    "executed_prefix_actions", "history_mode", "request_start_action_index",
    "source_request_id", "source_video_id", "source_video_sha256",
)
ALIGNMENT_CONTRACT_FIELDS = (
    "alignment_contract_id", "alignment_contract_sha256", "alignment_receipt_id",
    "camera_crop_id", "camera_crop_sha256", "camera_id",
    "early_horizon_supported", "generated_frame_index",
    "target_executed_action_offset", "target_physical_time_s",
    "timestamp_tolerance_s",
)
PHYSICAL_MAPPING_FIELDS = ("alignment_receipt_sha256",)
CROSS_SOURCE_DERIVATION_FIELDS = (
    "camera_identity_match", "target_within_executed_prefix", "technical_valid",
    "technical_invalid_reason", "timestamp_error_s",
)
FIELD_GROUPS = {
    "retained_mapping": RETAINED_FIELDS,
    "request_provenance": REQUEST_PROVENANCE_FIELDS,
    "alignment_contract": ALIGNMENT_CONTRACT_FIELDS,
    "physical_mapping_identity": PHYSICAL_MAPPING_FIELDS,
    "cross_source_derivation": CROSS_SOURCE_DERIVATION_FIELDS,
}
EXPECTED_REQUEST_KEYS = tuple(sorted(
    field for group in FIELD_GROUPS.values() for field in group
))

EXPECTED_ROW_EXPRESSIONS = {
    "source_request_id": "source['source_request_id']",
    "cell_id": "source['cell_id']",
    "source_video_id": "source['source_video_id']",
    "source_video_sha256": "source['source_video']['sha256']",
    "model_id": "model",
    "layout_pair_id": "source['layout_pair_id']",
    "condition_id": "source['condition_id']",
    "episode_id": "source['recording_id']",
    "request_index": "source['request_index']",
    "request_start_action_index": "source['action_step_start']",
    "action_manifest_sha256": "action_manifest['sha256']",
    "camera_id": "alignment['camera_id']",
    "camera_crop_id": "alignment['camera_crop_id']",
    "camera_crop_sha256": "alignment['camera_crop_sha256']",
    "alignment_contract_id": "alignment['contract_id']",
    "alignment_contract_sha256": "alignment['contract_sha256']",
    "technical_valid": "technical_valid",
    "technical_invalid_reason": (
        "None if technical_valid else 'forecast_timing_unavailable'"
    ),
    "camera_identity_match": "binding.get('camera_id') == alignment['camera_id']",
    "target_within_executed_prefix": "within_prefix",
    "executed_prefix_actions": "executed_prefix",
    "target_executed_action_offset": "alignment['target_executed_action_offset']",
    "generated_frame_index": "alignment['generated_frame_index']",
    "target_physical_time_s": "alignment['primary_horizon_s']",
    "timestamp_error_s": "timestamp_error",
    "timestamp_tolerance_s": "alignment['timestamp_tolerance_s']",
    "history_mode": "source['history_mode']",
    "early_horizon_supported": "alignment['early_horizon'] is not None",
    "alignment_receipt_id": "alignment['mapping_receipt_id']",
    "alignment_receipt_sha256": "mapping_sha_by_model[model]",
}

EXPECTED_OUTPUT_NAMES = {
    "N3": {
        "mapping": "n3_physical_alignment_receipt.json",
        "alignment": "n3_alignment_contract.json",
    },
    "D1": {
        "mapping": "d1_physical_alignment_receipt.json",
        "alignment": "d1_alignment_contract.json",
    },
}
EXPECTED_CROP_NAMES = {
    "N3": "n3_camera_crop_contract.json",
    "D1": "d1_camera_crop_contract.json",
}
EXPECTED_IMPLEMENTATION_PATHS = {
    "queue_wrapper": MEDIA_JOBS_PATH,
    "queue_support": QUEUE_SUPPORT_PATH,
    "bridge": ASSEMBLER_PATH,
    "compiler": COMPILER_PATH,
    "timing_validator": TIMING_VALIDATOR_PATH,
    "annotation_workflow": WORKFLOW_PATH,
    "freeze_validator": FREEZE_VALIDATOR_PATH,
    "camera_replay": CAMERA_REPLAY_PATH,
    "queue_contract": MEDIA_CONTRACT_PATH,
}
EXPECTED_SOURCE_PATH_EXPRESSIONS = {
    "queue_wrapper": "root / THIS_RELATIVE",
    "queue_support": "root / QUEUE_RELATIVE",
    "bridge": "root / BRIDGE_RELATIVE",
    "compiler": "root / COMPILER_RELATIVE",
    "timing_validator": "root / TIMING_RELATIVE",
    "annotation_workflow": "root / ANNOTATION_RELATIVE",
    "freeze_validator": "root / FREEZE_RELATIVE",
    "camera_replay": "root / CAMERA_REPLAY_RELATIVE",
    "queue_contract": "root / CONTRACT_RELATIVE",
}
EXPECTED_INVENTORY_FIELDS = (
    "schema_version", "study_id", "stage", "cohort_branch",
    "inventory_complete", "inventory_finalized_at", "annotation_state",
    "episode_roster", "alignment_contracts", "requests",
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
EXPECTED_ALGORITHM = (
    "Within each episode, sort eligible source_request_id values by "
    "SHA256(UTF8(seed) || NUL || UTF8(episode_id) || NUL || "
    "UTF8(source_request_id)), breaking digest ties by source_request_id; "
    "select the first min(4, eligible_count)."
)
ZERO_NEW_ACTIVITY = {
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
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")


def require_exact(value: object, expected: object, label: str) -> None:
    require(canonical_bytes(value) == canonical_bytes(expected), f"{label} changed")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "show", f"{commit}:{path}",
        ],
        timeout=30,
    )


def verify_pinned_blobs(blobs: Mapping[tuple[str, str], bytes]) -> None:
    require(set(blobs) == set(EXPECTED_SOURCES), "Pinned source inventory changed")
    for key, expected in EXPECTED_SOURCES.items():
        payload = blobs[key]
        actual = (len(payload), sha256_bytes(payload))
        require(actual == expected, f"Pinned source changed: {key[0]}:{key[1]}")


def load_pinned_blobs() -> dict[tuple[str, str], bytes]:
    blobs = {key: git_blob(*key) for key in EXPECTED_SOURCES}
    verify_pinned_blobs(blobs)
    return blobs


def line_index(lines: list[str]) -> dict[str, object]:
    ordered = sorted(lines)
    require(len(ordered) == len(set(ordered)), "Index rows are not unique")
    payload = "".join(f"{line}\n" for line in ordered).encode("utf-8")
    return {
        "canonicalization": "lexicographically sorted UTF-8 rows with LF terminators",
        "count": len(ordered),
        "canonical_bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def source_descriptor(commit: str, path: str, payload: bytes) -> dict[str, object]:
    return {
        "commit": commit,
        "path": path,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def assignment_literal(source: str, name: str) -> object:
    tree = ast.parse(source)
    matches: list[ast.AST] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            matches.append(node.value)
    require(len(matches) == 1, f"Expected one assignment: {name}")
    try:
        return ast.literal_eval(matches[0])
    except (ValueError, TypeError) as error:
        raise ValueError(f"Assignment is not a literal: {name}") from error


def function_descriptor(source: str, path: str, name: str) -> dict[str, object]:
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    require(len(nodes) == 1, f"Expected one function: {path}:{name}")
    node = nodes[0]
    segment = ast.get_source_segment(source, node)
    require(segment is not None, f"Function source unavailable: {path}:{name}")
    payload = segment.encode("utf-8")
    expected = EXPECTED_FUNCTIONS[(path, name)]
    actual = (node.lineno, node.end_lineno, len(payload), sha256_bytes(payload))
    require(actual == expected, f"Function changed: {path}:{name}")
    return {
        "name": name,
        "path": path,
        "start_line": node.lineno,
        "end_line": node.end_lineno,
        "bytes": len(payload),
        "sha256": sha256_bytes(payload),
    }


def request_row_expressions(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_request_inventory"
    ]
    require(len(functions) == 1, "Request inventory builder changed")
    candidates: list[ast.Dict] = []
    for node in ast.walk(functions[0]):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "row" for target in node.targets):
            continue
        candidates.append(node.value)
    require(len(candidates) == 1, "Request row dictionary changed")
    result: dict[str, str] = {}
    row = candidates[0]
    for key_node, value_node in zip(row.keys, row.values):
        require(
            isinstance(key_node, ast.Constant) and type(key_node.value) is str,
            "Request row key changed",
        )
        key = key_node.value
        require(key not in result, f"Duplicate request row field: {key}")
        result[key] = ast.unparse(value_node)
    require_exact(result, EXPECTED_ROW_EXPRESSIONS, "Request row expressions")
    return result


def implementation_source_inventory(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_source_paths"
    ]
    require(len(functions) == 1, "Implementation source inventory function changed")
    returns = [node for node in ast.walk(functions[0]) if isinstance(node, ast.Return)]
    require(len(returns) == 1 and isinstance(returns[0].value, ast.Dict),
            "Implementation source inventory return changed")
    result: dict[str, str] = {}
    dictionary = returns[0].value
    for key_node, value_node in zip(dictionary.keys, dictionary.values):
        require(isinstance(key_node, ast.Constant) and type(key_node.value) is str,
                "Implementation source inventory key changed")
        result[key_node.value] = ast.unparse(value_node)
    require_exact(result, EXPECTED_SOURCE_PATH_EXPRESSIONS,
                  "Implementation source path expressions")
    require(set(result) == set(EXPECTED_IMPLEMENTATION_PATHS),
            "Implementation source path names changed")
    return result


def inventory_dictionary_contract(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "_build_request_inventory"),
        None,
    )
    require(function is not None, "Request inventory builder changed")
    candidates: list[ast.Dict] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Dict):
            continue
        if any(isinstance(target, ast.Name) and target.id == "inventory"
               for target in node.targets):
            candidates.append(node.value)
    require(len(candidates) == 1, "Request inventory dictionary changed")
    result: dict[str, str] = {}
    for key_node, value_node in zip(candidates[0].keys, candidates[0].values):
        require(isinstance(key_node, ast.Constant) and type(key_node.value) is str,
                "Request inventory field changed")
        result[key_node.value] = ast.unparse(value_node)
    require(tuple(result) == EXPECTED_INVENTORY_FIELDS,
            "Request inventory field order changed")
    require(result["stage"] == "'development'", "Request inventory stage changed")
    require(result["cohort_branch"] == "'full_two_model'",
            "Request inventory cohort changed")
    require(result["inventory_complete"] == "True",
            "Request inventory completion gate changed")
    require(result["annotation_state"] == "'not_started'",
            "Request inventory annotation gate changed")
    require(result["requests"] == "rows", "Request inventory rows changed")
    return result


def prepare_call_graph(source: str) -> dict[str, int]:
    tree = ast.parse(source)
    function = next(
        (node for node in tree.body
         if isinstance(node, ast.FunctionDef) and node.name == "prepare"),
        None,
    )
    require(function is not None, "Prepare function changed")
    counts = {
        "_build_request_inventory": 0,
        "annotation.select_requests": 0,
        "_assert_inventory_and_selection": 0,
        "_render_selected": 0,
    }
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in counts:
            counts[name] += 1
    require_exact(counts, {
        "_build_request_inventory": 1,
        "annotation.select_requests": 2,
        "_assert_inventory_and_selection": 2,
        "_render_selected": 1,
    }, "Prepare call graph")
    return counts


def validate_lineage_report(report: Mapping[str, Any]) -> None:
    require(report.get("schema_version") ==
            "wmf-development-request-recovery-lineage-v1",
            "Lineage report schema changed")
    lineage = report.get("request_lineage")
    require(type(lineage) is dict, "Request lineage missing")
    require(lineage.get("requests") == 1152 and lineage.get("cells") == 32,
            "Request lineage population changed")
    require_exact(lineage.get("requests_by_model"), {"D1": 912, "N3": 240},
                  "Lineage model population")
    require_exact(lineage.get("requests_per_cell"), {"D1": 57, "N3": 15},
                  "Lineage per-cell population")
    require_exact(lineage.get("index"), {
        "canonical_bytes": 160740,
        "canonicalization": "ordered UTF-8 rows with LF terminators",
        "count": 1152,
        "sha256": "165b87ad0d260478fb2ba13348887edc723d962e3ed29b53539cf2025402d935",
    }, "Lineage index")
    rows = lineage.get("rows")
    require(type(rows) is list and len(rows) == 1152, "Lineage rows changed")
    expected_row_fields = {
        "cell_id", "layout_pair_id", "model_id", "model_request_ordinal",
        "request_index", "scheduled_condition", "scheduled_ordinal",
        "source_request_receipt_sha256",
    }
    seen: set[tuple[str, str, int]] = set()
    counts = {"D1": 0, "N3": 0}
    for ordinal, row in enumerate(rows):
        require(type(row) is dict and set(row) == expected_row_fields,
                f"Lineage row fields changed: {ordinal}")
        model = row.get("model_id")
        cell = row.get("cell_id")
        request_index = row.get("request_index")
        require(model in counts and type(cell) is str and type(request_index) is int,
                f"Lineage join key invalid: {ordinal}")
        key = (model, cell, request_index)
        require(key not in seen, f"Lineage join key duplicated: {ordinal}")
        seen.add(key)
        counts[model] += 1
    require(counts == {"D1": 912, "N3": 240}, "Lineage row counts changed")
    require_exact(report.get("global_study_snapshot"), EXPECTED_GLOBAL_SNAPSHOT,
                  "Lineage global snapshot")
    require_exact(report.get("timing_boundary"), EXPECTED_TIMING_BOUNDARY,
                  "Lineage timing boundary")
    require(all(value == 0 for value in report.get("new_activity", {}).values()),
            "Lineage activity changed")
    boundary = report.get("annotation_and_release_boundary", {})
    require(boundary and all(value == 0 or value is False for value in boundary.values()),
            "Lineage annotation boundary promoted")


def validate_spec(payload: bytes) -> None:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Scientific specification is not UTF-8") from error
    required = (
        "The maximum core workload is **232 new episodes**.",
        "generate the actual request-selection manifest after execution provides "
        "the request inventory and before any image labels are inspected",
        "Eligibility uses camera/time/action-prefix correspondence, not object "
        "visibility or forecast quality.",
        "This spec is complete as a scientific run matrix. It is deliberately "
        "not an executable release",
    )
    require(all(fragment in text for fragment in required),
            "Scientific stage-gate specification changed")


def validate_media_contract(report: Mapping[str, Any]) -> None:
    require(report.get("schema_version") ==
            "wmf-development-annotation-media-bridge-contract-v1",
            "Media bridge contract schema changed")
    policy = report.get("input_policy")
    require(type(policy) is dict, "Media bridge input policy missing")
    require_exact(policy.get("join_key"), ["model_id", "cell_id", "request_index"],
                  "Media bridge join key")
    expected = report.get("exact_output_counts")
    require(type(expected) is dict, "Media bridge expected counts missing")
    require(expected.get("episodes") == 32 and expected.get("requests") == 1152,
            "Media bridge population changed")
    require(expected.get("timing_camera_action_eligible_requests") == 448,
            "Media bridge eligibility expectation changed")
    require(expected.get("selected_requests") == 128,
            "Media bridge selection expectation changed")
    require(report.get("safe_for_rater_distribution") is False,
            "Media bridge distribution gate promoted")
    require(report.get("safe_to_release_confirmation") is False,
            "Media bridge confirmation gate promoted")


def validate_selection_report(report: Mapping[str, Any]) -> None:
    require(
        report.get("schema_version") == "wmf-development-annotation-selection-readiness-v1",
        "Selection report schema changed",
    )
    population = report.get("retained_population")
    require(type(population) is dict, "Selection population missing")
    require_exact(
        {key: population.get(key) for key in (
            "rows", "cells", "rows_by_model", "cells_by_model", "rows_per_cell",
        )},
        {
            "rows": 1152, "cells": 32,
            "rows_by_model": {"D1": 912, "N3": 240},
            "cells_by_model": {"D1": 16, "N3": 16},
            "rows_per_cell": {"D1": 57, "N3": 15},
        },
        "Retained population",
    )
    require(population.get("annotation_request_schema_field_count") == 30,
            "Request schema count changed")
    require_exact(population.get("exact_annotation_schema_fields_available"),
                  list(RETAINED_FIELDS), "Retained field coverage")
    require(population.get("exact_annotation_schema_fields_available_count") == 4,
            "Retained field count changed")
    missing = population.get("exact_annotation_schema_fields_missing")
    require(type(missing) is list and tuple(sorted(missing)) == tuple(
        field for field in EXPECTED_REQUEST_KEYS if field not in RETAINED_FIELDS
    ), "Missing field coverage changed")
    require(population.get("exact_annotation_schema_fields_missing_count") == 26,
            "Missing field count changed")
    require(population.get("exact_annotation_schema_coverage_complete") is False,
            "Field coverage promoted")
    workflow = report.get("workflow_contract")
    require(type(workflow) is dict, "Workflow contract missing")
    require(tuple(workflow.get("annotation_request_schema_fields", [])) == EXPECTED_REQUEST_KEYS,
            "Workflow request fields changed")
    require(workflow.get("request_sampling_seed") == 2026091302,
            "Sampling seed changed")
    require(workflow.get("request_sample_cap_per_episode") == 4,
            "Sampling cap changed")
    require(workflow.get("request_sampling_algorithm") == EXPECTED_ALGORITHM,
            "Sampling algorithm changed")
    require(workflow.get("selection_requires_complete_inventory") is True,
            "Complete-inventory gate changed")
    require(workflow.get("selection_requires_not_started_annotation_state") is True,
            "Annotation-state gate changed")
    require(workflow.get("selection_uses_object_visibility_or_forecast_quality") is False,
            "Outcome-blind selection gate changed")
    selection = report.get("selection_readiness")
    require(type(selection) is dict, "Selection readiness missing")
    require(selection.get("total_retained_rows") == 1152,
            "Selection population changed")
    require(selection.get("sampling_key_feasible_rows") == 0,
            "Sampling feasibility promoted")
    require(selection.get("eligibility_decision_feasible_rows") == 0,
            "Eligibility feasibility promoted")
    require(selection.get("eligibility_decision_status") == "undetermined_not_ineligible",
            "Eligibility status changed")
    require(selection.get("selected_request_count") == 0,
            "Selection count promoted")
    require(selection.get("selection_manifest_created") is False,
            "Selection manifest promoted")
    require(selection.get("actual_request_inventory_authenticated") is False,
            "Request inventory promoted")
    require(selection.get("production_eligibility_determined") is False,
            "Production eligibility promoted")
    require_exact(report.get("global_study_snapshot"), EXPECTED_GLOBAL_SNAPSHOT,
                  "Global study snapshot")
    require_exact(report.get("timing_boundary"), EXPECTED_TIMING_BOUNDARY,
                  "Timing boundary")
    require(all(value == 0 for value in report.get("new_activity", {}).values()),
            "Selection report science activity changed")
    require(all(value == 0 for value in report.get("audit_activity", {}).values()),
            "Selection report audit activity changed")


def validate_media_report(report: Mapping[str, Any]) -> None:
    require(
        report.get("schema_version") == "wmf-development-annotation-media-input-closure-v1",
        "Media closure schema changed",
    )
    runtime = report.get("runtime_input_contract")
    require(type(runtime) is dict, "Runtime input contract missing")
    require_exact(runtime.get("top_level_fields"),
                  ["compiler", "mode", "models", "raw_root", "schema_version", "study_id"],
                  "Runtime top-level fields")
    require_exact(runtime.get("compiler_fields"), ["compiler_receipt", "job_receipt"],
                  "Runtime compiler fields")
    require_exact(runtime.get("model_fields"), [
        "alignment_contract", "camera_crop_contract", "physical_alignment_receipt",
        "request_provenance", "timing_job_receipt", "timing_sidecar",
    ], "Runtime model fields")
    require_exact(runtime.get("descriptor_fields"), ["bytes", "path", "sha256"],
                  "Runtime descriptor fields")
    require_exact(runtime.get("models"), ["N3", "D1"], "Runtime models")
    require_exact(runtime.get("join_key"), ["model_id", "cell_id", "request_index"],
                  "Runtime join key")
    require(runtime.get("required_top_level_slots") == 14,
            "Runtime slot count changed")
    require(runtime.get("runnable_input_manifest_created") is False,
            "Runtime input manifest promoted")
    require(runtime.get("queue_descriptor_created") is False,
            "Queue descriptor promoted")
    join = report.get("media_join_domain")
    require(type(join) is dict, "Media join domain missing")
    require(join.get("requests") == 1152 and join.get("cells") == 32,
            "Media join population changed")
    require_exact(join.get("requests_by_model"), {"D1": 912, "N3": 240},
                  "Media model population")
    require(join.get("duplicate_join_keys") == 0, "Media join duplication changed")
    require_exact(join.get("join_key_index"), {
        "canonical_bytes": 57280,
        "canonicalization": "ordered UTF-8 rows with LF terminators",
        "count": 1152,
        "sha256": "a736e247c263db5f91639954407e2d47deadce400a8d74af84e002e5f49daac4",
    }, "Media join index")
    closure = report.get("top_level_input_closure")
    require(type(closure) is dict, "Runtime closure missing")
    require_exact(
        {key: closure.get(key) for key in (
            "required_slots", "exact_identity_bound_slots",
            "recovery_manifest_pointer_slots",
            "retained_git_payload_slots_outside_recovery_manifest",
            "unauthenticated_slots",
            "current_runtime_byte_verified_slots", "semantic_replay_complete_slots",
        )},
        {
            "required_slots": 14, "exact_identity_bound_slots": 8,
            "recovery_manifest_pointer_slots": 5,
            "retained_git_payload_slots_outside_recovery_manifest": 3,
            "unauthenticated_slots": 6, "current_runtime_byte_verified_slots": 0,
            "semantic_replay_complete_slots": 0,
        },
        "Runtime closure counts",
    )
    slots = closure.get("slots")
    require(type(slots) is list and len(slots) == 14, "Runtime slots changed")
    ids = [row.get("slot_id") for row in slots if type(row) is dict]
    require(len(ids) == len(set(ids)) == 14, "Runtime slot identities changed")
    require(all(row.get("runtime_byte_verified") is False for row in slots),
            "Runtime byte verification promoted")
    require(all(row.get("semantic_replay_complete") is False for row in slots),
            "Semantic replay promoted")
    missing = {
        f"{model}.{field}"
        for model in ("D1", "N3")
        for field in ("alignment_contract", "camera_crop_contract", "physical_alignment_receipt")
    }
    require(set(closure.get("unfilled_required_slot_ids", [])) == missing,
            "Unauthenticated slot set changed")
    for row in slots:
        if row["slot_id"] in missing:
            require(row.get("source_state") == "no_authenticated_descriptor_in_pinned_sources",
                    "Missing slot acquired unsupported evidence")
        else:
            require(row.get("source_state") != "no_authenticated_descriptor_in_pinned_sources",
                    "Identity-bound slot lost evidence")
    expectations = report.get("contract_expectations_not_observed_results")
    require(type(expectations) is dict, "Contract expectations missing")
    require(expectations.get("timing_camera_action_eligible_requests_expected") == 448,
            "Eligibility expectation changed")
    require(expectations.get("selected_requests_expected") == 128,
            "Selection expectation changed")
    require(expectations.get("observed_selected_requests") == 0,
            "Observed selections promoted")
    require(expectations.get("production_eligibility_determined") is False,
            "Production eligibility promoted")
    require(expectations.get("production_selection_manifest_created") is False,
            "Production selection promoted")
    require(report.get("gate_state", {}).get("core_episode_ceiling") == 232,
            "Core ceiling changed")
    require(report.get("gate_state", {}).get("safe_to_release_new_work") is False,
            "Release gate promoted")
    require(all(value == 0 for value in report.get("new_activity", {}).values()),
            "Media closure activity changed")


def producer_outputs(alignment_source: str, crop_source: str) -> dict[str, dict[str, str]]:
    output_names = assignment_literal(alignment_source, "OUTPUT_NAMES")
    crop_names = assignment_literal(crop_source, "MODEL_CONTRACT_NAMES")
    require_exact(output_names, EXPECTED_OUTPUT_NAMES, "Alignment producer outputs")
    require_exact(crop_names, EXPECTED_CROP_NAMES, "Crop producer outputs")
    return {
        f"{model}.camera_crop_contract": {
            "output_name": EXPECTED_CROP_NAMES[model],
            "producer_commit": CROP_COMMIT,
            "producer_path": CROP_JOBS_PATH,
            "producer_constant": "MODEL_CONTRACT_NAMES",
        }
        for model in ("D1", "N3")
    } | {
        f"{model}.alignment_contract": {
            "output_name": EXPECTED_OUTPUT_NAMES[model]["alignment"],
            "producer_commit": ALIGNMENT_COMMIT,
            "producer_path": ALIGNMENT_JOBS_PATH,
            "producer_constant": "OUTPUT_NAMES",
        }
        for model in ("D1", "N3")
    } | {
        f"{model}.physical_alignment_receipt": {
            "output_name": EXPECTED_OUTPUT_NAMES[model]["mapping"],
            "producer_commit": ALIGNMENT_COMMIT,
            "producer_path": ALIGNMENT_JOBS_PATH,
            "producer_constant": "OUTPUT_NAMES",
        }
        for model in ("D1", "N3")
    }


def validate_field_groups() -> None:
    flattened = [field for group in FIELD_GROUPS.values() for field in group]
    require(len(flattened) == 30, "Field partition count changed")
    require(len(flattened) == len(set(flattened)), "Field partition is not disjoint")
    require(tuple(sorted(flattened)) == EXPECTED_REQUEST_KEYS,
            "Field partition union changed")
    require(set(RETAINED_FIELDS).isdisjoint(set(flattened) - set(RETAINED_FIELDS)),
            "Retained field partition overlaps")


def build_report(blobs: Mapping[tuple[str, str], bytes]) -> dict[str, Any]:
    verify_pinned_blobs(blobs)
    validate_field_groups()
    selection = load_json(blobs[(SELECTION_COMMIT, SELECTION_PATH)], SELECTION_PATH)
    media = load_json(blobs[(SELECTION_COMMIT, MEDIA_CLOSURE_PATH)], MEDIA_CLOSURE_PATH)
    lineage = load_json(blobs[(LINEAGE_COMMIT, LINEAGE_PATH)], LINEAGE_PATH)
    media_contract = load_json(
        blobs[(SELECTION_COMMIT, MEDIA_CONTRACT_PATH)], MEDIA_CONTRACT_PATH
    )
    validate_selection_report(selection)
    validate_media_report(media)
    validate_lineage_report(lineage)
    validate_media_contract(media_contract)
    validate_spec(blobs[(SPEC_COMMIT, SPEC_PATH)])

    sources = [
        source_descriptor(commit, path, blobs[(commit, path)])
        for commit, path in EXPECTED_SOURCES
    ]
    source_index = line_index([
        f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}"
        for row in sources
    ])
    source_index["source_bytes"] = sum(int(row["bytes"]) for row in sources)

    source_text = {
        path: blobs[(commit, path)].decode("utf-8")
        for commit, path in EXPECTED_SOURCES
        if path.endswith(".py")
    }
    functions = [
        function_descriptor(source_text[path], path, name)
        for path, name in EXPECTED_FUNCTIONS
    ]
    row_expressions = request_row_expressions(source_text[ASSEMBLER_PATH])
    inventory_fields = inventory_dictionary_contract(source_text[ASSEMBLER_PATH])
    call_graph = prepare_call_graph(source_text[ASSEMBLER_PATH])
    implementation_expressions = implementation_source_inventory(
        source_text[MEDIA_JOBS_PATH]
    )
    implementation_inventory = {
        name: source_descriptor(
            SELECTION_COMMIT, path, blobs[(SELECTION_COMMIT, path)]
        )
        for name, path in EXPECTED_IMPLEMENTATION_PATHS.items()
    }
    prospective = producer_outputs(
        source_text[ALIGNMENT_JOBS_PATH], source_text[CROP_JOBS_PATH]
    )

    closure = media["top_level_input_closure"]
    slot_rows = []
    for retained in closure["slots"]:
        slot_id = retained["slot_id"]
        missing = slot_id in prospective
        slot_rows.append({
            "slot_id": slot_id,
            "model_id": retained["model_id"],
            "field": retained["field"],
            "identity_bound": not missing,
            "current_source_state": retained["source_state"],
            "runtime_byte_verified": False,
            "semantic_replay_complete": False,
            "prospective_output": prospective.get(slot_id),
            "prospective_output_is_not_existing_evidence": missing,
        })
    source_state_counts = {
        state: sum(row["current_source_state"] == state for row in slot_rows)
        for state in (
            "declared_raw_pointer_in_recovery_manifest_current_availability_unknown",
            "retained_git_blob_outside_recovery_manifest_runtime_materialization_unverified",
            "no_authenticated_descriptor_in_pinned_sources",
        )
    }
    require_exact(source_state_counts, {
        "declared_raw_pointer_in_recovery_manifest_current_availability_unknown": 5,
        "retained_git_blob_outside_recovery_manifest_runtime_materialization_unverified": 3,
        "no_authenticated_descriptor_in_pinned_sources": 6,
    }, "Runtime slot source-state counts")

    entrypoint_map = {row["name"]: row for row in functions}
    report: dict[str, Any] = {
        "schema_version": "wmf-development-annotation-activation-conformance-v1",
        "study_id": "WMF-ABLATION-001",
        "authenticated_source_blobs": {
            **source_index,
            "sources": sorted(sources, key=lambda row: (row["commit"], row["path"])),
        },
        "existing_pipeline_entrypoints": {
            "functions": functions,
            "implementation_source_path_count": 9,
            "implementation_source_paths_function": entrypoint_map["_source_paths"],
            "implementation_source_path_expressions": implementation_expressions,
            "implementation_source_inventory": implementation_inventory,
            "validate_input_shape": entrypoint_map["validate_input_shape"],
            "descriptor_shape": entrypoint_map["_descriptor_shape"],
            "builder": entrypoint_map["build_formal_wave"],
            "assembler": entrypoint_map["prepare"],
            "selection": entrypoint_map["select_requests"],
            "prepare_static_call_graph": call_graph,
            "functions_imported_by_audit": False,
            "functions_invoked_by_audit": False,
            "builder_release_flag_is_future_descriptor_behavior_not_current_release": True,
        },
        "runtime_manifest_contract": {
            "schema_version": media["runtime_input_contract"]["runtime_input_schema"],
            "top_level_fields": media["runtime_input_contract"]["top_level_fields"],
            "compiler_fields": media["runtime_input_contract"]["compiler_fields"],
            "model_fields": media["runtime_input_contract"]["model_fields"],
            "descriptor_fields": media["runtime_input_contract"]["descriptor_fields"],
            "models": media["runtime_input_contract"]["models"],
            "required_slots": 14,
            "join_key": media["runtime_input_contract"]["join_key"],
            "runnable_manifest_created": False,
            "queue_descriptor_created": False,
            "builder_invoked": False,
        },
        "runtime_slot_activation_matrix": {
            "required_slots": 14,
            "identity_bound_slots": 8,
            "recovery_manifest_pointer_slots": 5,
            "retained_git_payload_slots_outside_recovery_manifest": 3,
            "unauthenticated_slots": 6,
            "runtime_byte_verified_slots": 0,
            "semantic_replay_complete_slots": 0,
            "source_state_counts": source_state_counts,
            "slots": sorted(slot_rows, key=lambda row: row["slot_id"]),
        },
        "request_inventory_contract": {
            "rows": 1152,
            "cells": 32,
            "rows_by_model": {"D1": 912, "N3": 240},
            "rows_per_cell": {"D1": 57, "N3": 15},
            "join_key": ["model_id", "cell_id", "request_index"],
            "join_key_index": media["media_join_domain"]["join_key_index"],
            "request_schema_field_count": 30,
            "builder_function": entrypoint_map["_build_request_inventory"],
            "inventory_dictionary_fields": inventory_fields,
            "complete_inventory_required": True,
            "required_annotation_state": "not_started",
            "request_rows_materialized_by_audit": 0,
        },
        "request_field_provenance": {
            "field_count": 30,
            "groups": {
                name: {"count": len(fields), "fields": list(fields)}
                for name, fields in FIELD_GROUPS.items()
            },
            "partition_is_disjoint": True,
            "partition_union_is_exact": True,
            "builder_row_expressions": row_expressions,
            "authenticated_value_fields": list(RETAINED_FIELDS),
            "authenticated_value_field_count": 4,
            "missing_value_fields": [
                field for field in EXPECTED_REQUEST_KEYS if field not in RETAINED_FIELDS
            ],
            "missing_value_field_count": 26,
            "complete_rows": 0,
            "total_rows": 1152,
            "identity_substitution_permitted": False,
        },
        "selection_contract": {
            "seed": 2026091302,
            "cap_per_episode": 4,
            "algorithm": EXPECTED_ALGORITHM,
            "eligibility_fields": [
                "camera_identity_match", "target_within_executed_prefix",
                "technical_valid", "timestamp_error_s", "timestamp_tolerance_s",
            ],
            "sampling_identity_fields": ["episode_id", "source_request_id"],
            "eligibility_function": entrypoint_map["_eligibility_reasons"],
            "selection_function": entrypoint_map["select_requests"],
            "requires_complete_inventory": True,
            "requires_annotation_state_not_started": True,
            "uses_labels_visibility_or_forecast_quality": False,
            "sampling_key_feasible_rows": 0,
            "eligibility_feasible_rows": 0,
            "eligibility_status": "undetermined_not_ineligible",
            "expected_eligible_requests_contract_only": 448,
            "expected_selected_requests_contract_only": 128,
            "observed_selected_requests": 0,
            "selection_manifest_created": False,
        },
        "activation_decision": {
            "status": "not_activatable_from_authenticated_retained_evidence",
            "pipeline_dependency_graph_fully_accounted_for": True,
            "authenticated_field_value_coverage": "4/30",
            "complete_rows": "0/1152",
            "runtime_byte_verified_slots": "0/14",
            "semantic_replay_complete_slots": "0/14",
            "blocking_reasons": [
                "six required runtime descriptors are unauthenticated",
                "no required runtime slot has current byte verification",
                "no required runtime slot has semantic replay",
                "twenty-six request field values are not authenticated",
                "zero rows have feasible sampling identities",
                "zero rows have feasible eligibility metadata",
            ],
            "safe_to_release": False,
        },
        "gate_state": {
            "core_episode_ceiling": 232,
            "retained_snapshot_active_launched_cells": 0,
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "camera_alignment_ready": False,
            "development_annotation_gate_ready": False,
            "annotation_media_ready": False,
            "human_annotation_ready": False,
            "resource_freeze_ready": False,
            "confirmation_release_ready": False,
            "safe_to_release_new_work": False,
        },
        "global_study_snapshot": EXPECTED_GLOBAL_SNAPSHOT,
        "timing_boundary": EXPECTED_TIMING_BOUNDARY,
        "new_activity": ZERO_NEW_ACTIVITY,
        "audit_activity": {
            "authenticated_git_blobs_read": len(blobs),
            "authenticated_git_bytes_read": sum(len(payload) for payload in blobs.values()),
            "external_system_reads": 0,
            "raw_paths_opened": 0,
            "raw_bytes_read": 0,
            "runtime_functions_imported": 0,
            "runtime_functions_invoked": 0,
        },
        "claim_boundaries": {
            "positive_claim": (
                "The frozen builder's 30-field expression dependency graph is fully "
                "accounted for by authenticated retained Git sources."
            ),
            "not_established": [
                "request field values", "current raw or PVC availability",
                "production eligibility", "request selection", "timing completion",
                "camera or alignment qualification", "annotation readiness",
                "release authority", "forecast accuracy",
                "current GM PVC queue worker or live-quiescence state",
            ],
            "prospective_output_names_are_not_evidence_of_existing_outputs": True,
            "contract_expectations_are_not_observed_results": True,
            "software_conformance_is_not_scientific_evidence": True,
        },
        "negative_invariants": {
            "raw_or_pvc_reads": 0,
            "runtime_manifest_emitted": False,
            "queue_descriptor_emitted": False,
            "request_rows_emitted": 0,
            "sampling_ranks_emitted": 0,
            "eligibility_decisions_emitted": 0,
            "selected_flags_emitted": 0,
            "media_emitted": 0,
            "labels_emitted": 0,
            "human_responses_emitted": 0,
            "scientific_estimates_emitted": 0,
            "core_ceiling_changed": False,
        },
    }
    validate_output_contract(report)
    return report


def validate_output_contract(report: Mapping[str, Any]) -> None:
    require(report.get("schema_version") ==
            "wmf-development-annotation-activation-conformance-v1",
            "Output schema changed")
    decision = report.get("activation_decision", {})
    require(decision.get("status") ==
            "not_activatable_from_authenticated_retained_evidence",
            "Activation decision promoted")
    require(decision.get("safe_to_release") is False, "Activation release promoted")
    require(decision.get("pipeline_dependency_graph_fully_accounted_for") is True,
            "Dependency graph claim changed")
    matrix = report.get("runtime_slot_activation_matrix", {})
    require(matrix.get("required_slots") == 14, "Output slot count changed")
    require(matrix.get("identity_bound_slots") == 8, "Output identity count changed")
    require(matrix.get("recovery_manifest_pointer_slots") == 5,
            "Output recovery-pointer count changed")
    require(matrix.get("retained_git_payload_slots_outside_recovery_manifest") == 3,
            "Output retained-payload count changed")
    require(matrix.get("unauthenticated_slots") == 6,
            "Output missing slot count changed")
    require(
        matrix.get("recovery_manifest_pointer_slots")
        + matrix.get("retained_git_payload_slots_outside_recovery_manifest")
        + matrix.get("unauthenticated_slots") == 14,
        "Output slot partition changed",
    )
    require_exact(matrix.get("source_state_counts"), {
        "declared_raw_pointer_in_recovery_manifest_current_availability_unknown": 5,
        "retained_git_blob_outside_recovery_manifest_runtime_materialization_unverified": 3,
        "no_authenticated_descriptor_in_pinned_sources": 6,
    }, "Output source-state counts")
    require(matrix.get("runtime_byte_verified_slots") == 0,
            "Output runtime verification promoted")
    require(matrix.get("semantic_replay_complete_slots") == 0,
            "Output semantic replay promoted")
    fields = report.get("request_field_provenance", {})
    require(fields.get("field_count") == 30, "Output field count changed")
    require(fields.get("authenticated_value_field_count") == 4,
            "Output field coverage promoted")
    require(fields.get("missing_value_field_count") == 26,
            "Output missing field count changed")
    require(fields.get("complete_rows") == 0, "Output complete rows promoted")
    selection = report.get("selection_contract", {})
    require(selection.get("sampling_key_feasible_rows") == 0,
            "Output sampling feasibility promoted")
    require(selection.get("eligibility_feasible_rows") == 0,
            "Output eligibility feasibility promoted")
    require(selection.get("observed_selected_requests") == 0,
            "Output selection promoted")
    require(selection.get("selection_manifest_created") is False,
            "Output selection manifest promoted")
    require(report.get("gate_state", {}).get("core_episode_ceiling") == 232,
            "Output core ceiling changed")
    require(report.get("gate_state", {}).get(
        "retained_snapshot_active_launched_cells") == 0,
        "Output retained activity snapshot changed")
    require(report.get("gate_state", {}).get(
        "current_GM_PVC_worker_state_reconciled") is False,
        "Output current worker state promoted")
    require(report.get("gate_state", {}).get("live_quiescence_inferred") is False,
            "Output live quiescence promoted")
    require(report.get("gate_state", {}).get("safe_to_release_new_work") is False,
            "Output release gate promoted")
    require_exact(report.get("new_activity"), ZERO_NEW_ACTIVITY,
                  "Output science activity")
    negative = report.get("negative_invariants", {})
    require(negative and all(
        value == 0 if type(value) is int else value is False
        for value in negative.values()
    ), "Output prohibited activity changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    require(not (args.check and args.output), "Use only one of --check or --output")
    payload = canonical_bytes(build_report(load_pinned_blobs()))
    if args.check:
        require(args.check.read_bytes() == payload,
                "Saved report is not canonical or current")
        return
    if args.output:
        args.output.write_bytes(payload)
        return
    sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
