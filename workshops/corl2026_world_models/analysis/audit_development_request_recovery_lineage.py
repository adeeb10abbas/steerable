#!/usr/bin/env python3
"""Build the pending development request-recovery lineage and verifier contract.

The default audit reads only byte-pinned Git objects.  It joins the previously
authenticated declared request-to-cell mapping to the preflight's declared
recovery dependencies.  It does not inspect the GM PVC, release retrieval,
select requests, create annotation media, or advance any scientific gate.

The explicit ``--verify-root`` mode is for a future, authorized, isolated,
quiescent retrieval bundle with no concurrent writers.  The bundle must contain
exactly the 56 declared files under their PVC-root-relative paths.  Verification
rejects missing or extra files, links, special files, path escape, byte-count
drift, SHA-256 drift, and observed file or tree mutation.  Its descriptor-held
barrier is a bounded observation, not a filesystem snapshot or proof of future
immutability.  Merely building the default report never enters that mode or
opens a declared raw path.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
from typing import Iterable


REPOSITORY = Path(__file__).resolve().parents[3]
PVC_ROOT = PurePosixPath(
    "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912"
)

SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
PREFLIGHT_COMMIT = "45c3a5d591c539779e8ff7db53d4af1c64273271"
JOIN_COMMIT = "dec8f8a63e3a45795cc47897396c20922961b4d6"
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"

SPEC_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "ablation_spec.json"
)
PREFLIGHT_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_raw_retrieval_preflight.json"
)
JOIN_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_request_cell_join.json"
)

EXPECTED_SOURCES = {
    (SPEC_COMMIT, SPEC_PATH): (
        8590,
        "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
    ),
    (PREFLIGHT_COMMIT, PREFLIGHT_PATH): (
        143329,
        "3d5e1c030bf8d7b607cd586f8a80b5ac1382529d3ce81210b2495cf4a734b35f",
    ),
    (JOIN_COMMIT, JOIN_PATH): (
        442880,
        "5099b8e3630bd9b292f549a8703ea7cfd1389af3d0c4227665c78d42955e79e9",
    ),
}

MODELS = ("D1", "N3")
LAYOUTS = ("D01", "D02", "D03", "D04")
REQUESTS_PER_CELL = {"D1": 57, "N3": 15}
REQUESTS_BY_MODEL = {"D1": 912, "N3": 240}

GLOBAL_STUDY_SNAPSHOT = {
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

GATE_STATE = {
    "annotation_media_ready": False,
    "camera_alignment_ready": False,
    "compiler_ready": True,
    "confirmation_release_ready": False,
    "development_annotation_gate_ready": False,
    "human_annotation_ready": False,
    "resource_freeze_ready": False,
    "timing_partially_ready": True,
}

EXPECTED_PROTOCOL = {
    "confirmation_cells": 192,
    "core_episode_ceiling": 232,
    "counts_authorize_execution": False,
    "development_cells": 32,
    "independent_raters_required": 2,
    "optional_D2_selected": False,
    "request_manifest_timing": (
        "algorithm and seed frozen before confirmation; selection manifest generated "
        "from actual metadata after execution and before labels"
    ),
    "request_sample_cap_per_episode": 4,
    "request_sampling_seed": 2026091302,
}

EXPECTED_ANNOTATION_AND_RELEASE_BOUNDARY = {
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

EXPECTED_PREFLIGHT_INDEXES = {
    "cell_lineage": (
        32,
        37360,
        "0fc384c9b32a415ae8ee8fbd7231327d04a3f0b806020382f2e85acf82117666",
    ),
    "cell_receipts": (
        32,
        10384,
        "93268de436fea9788913cea35d8aac74a089fa8565d446975c130fe2b0441220",
    ),
    "combined": (
        56,
        17017,
        "518ef94a22d156bba50640e7b42e94bc044e3d737dac12bd967cf8eb0162adb7",
    ),
    "compiler_outputs": (
        8,
        2474,
        "4b88b8c7fb4523b462da78bd3d9b2f434477bcef2b643fb7dc0ca99702617490",
    ),
    "fixture_inputs": (
        12,
        3112,
        "2156b35857d4bea77906ce17bf20f9b94f5ff78cce22db222304901cec8cfe18",
    ),
    "timing_outputs": (
        4,
        1047,
        "158544edb2f77651277affc00375d1bed9c897757e27ea5860d52a835f2c78c6",
    ),
}

EXPECTED_JOIN_INDEXES = {
    "D1": (
        912,
        123872,
        "4a620dc2a859a703c6d3ae327ddfbdb21f8a3fbb26277bf54579e994b322282b",
    ),
    "N3": (
        240,
        32480,
        "3e2a85618beabd88e33de69edfa1f1398bc277a2c5c9791153b0ecf3c4b0e3ab",
    ),
    "combined": (
        1152,
        156352,
        "6d84f5801a2b0caab68b201fec9f0066fde9ba4a5466bc154d6b48e63c783ade",
    ),
}

EXPECTED_IDENTITY_INDEX = (
    1152,
    78336,
    "3a6206f1e5fc790f1b8c92d9d023c1ce71f969ec95c19ea4482073a96cb86a71",
)

EXPECTED_DEPENDENCY_COUNTS = {
    "cell_receipt": (32, 1711371),
    "compiler_output": (8, 14007403),
    "fixed_observation_capture_receipt": (4, 320938),
    "fixed_observation_fixture": (8, 6231896),
    "timing_output": (4, 10392930),
}
EXPECTED_DEPENDENCIES = 56
EXPECTED_DEPENDENCY_BYTES = 32664538
EXPECTED_RECOVERY_EDGES = 160
EXPECTED_FILE_TYPES = {
    ".json": (48, 26432642),
    ".npz": (8, 6231896),
}

EXPECTED_D1_TIMING = {
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

CELL_KEYS = {
    "accepted_model_fixture",
    "aggregate_accepted_gate_record_sha256",
    "aggregate_binding_scope",
    "aggregate_receipt",
    "availability",
    "cell_id",
    "layout_pair_id",
    "layout_recovery_context",
    "model_id",
    "raw_cell_receipt",
    "scheduled_behavioral_request_count",
    "scheduled_condition",
    "scheduled_ordinal",
}

PER_CELL_KEYS = {
    "cell_id",
    "first_model_request_ordinal",
    "last_model_request_ordinal",
    "layout_pair_id",
    "model_id",
    "requests",
    "scheduled_condition",
    "scheduled_ordinal",
}

REQUEST_KEYS = {
    "cell_id",
    "layout_pair_id",
    "model_id",
    "request_index",
    "scheduled_condition",
    "scheduled_ordinal",
    "source_request_receipt_sha256",
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
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def require_exact_json(value: object, expected: object, label: str) -> None:
    require(canonical_bytes(value) == canonical_bytes(expected), f"{label} changed")


def require_sha256(value: object, label: str) -> str:
    require(type(value) is str and len(value) == 64, f"Invalid SHA-256: {label}")
    require(
        all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


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


def source_json(commit: str, path: str) -> tuple[dict, dict]:
    payload = git_blob(commit, path)
    expected_bytes, expected_sha256 = EXPECTED_SOURCES[(commit, path)]
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    require(len(payload) == expected_bytes, f"Source byte count changed: {path}")
    require(actual_sha256 == expected_sha256, f"Source SHA-256 changed: {path}")
    return load_json(payload, f"{commit}:{path}"), {
        "bytes": len(payload),
        "commit": commit,
        "path": path,
        "sha256": actual_sha256,
    }


def require_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Descriptor is not an object: {label}")
    require(
        set(value) == {"bytes", "path", "sha256"},
        f"Descriptor keys changed: {label}",
    )
    require(
        type(value["bytes"]) is int and value["bytes"] > 0,
        f"Descriptor byte count changed: {label}",
    )
    require(type(value["path"]) is str, f"Descriptor path changed: {label}")
    path = PurePosixPath(value["path"])
    require(path.is_absolute(), f"Descriptor path is not absolute: {label}")
    require(str(path) == value["path"], f"Descriptor path is not canonical: {label}")
    require(".." not in path.parts, f"Descriptor path traversal: {label}")
    try:
        relative = path.relative_to(PVC_ROOT)
    except ValueError as error:
        raise ValueError(f"Descriptor leaves PVC root: {label}") from error
    require(relative.parts and "." not in relative.parts, f"Bad relative path: {label}")
    require_sha256(value["sha256"], label)
    return dict(value)


def require_git_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Git descriptor is not an object: {label}")
    require(
        set(value) == {"bytes", "git_path", "sha256"},
        f"Git descriptor keys changed: {label}",
    )
    require(
        type(value["bytes"]) is int and value["bytes"] > 0,
        f"Git descriptor byte count changed: {label}",
    )
    require(
        type(value["git_path"]) is str
        and value["git_path"]
        and not value["git_path"].startswith("/"),
        f"Git descriptor path changed: {label}",
    )
    require(
        ".." not in PurePosixPath(value["git_path"]).parts,
        f"Git descriptor path traversal: {label}",
    )
    require_sha256(value["sha256"], label)
    return dict(value)


def line_index(rows: Iterable[str], label: str, *, ordered: bool = False) -> dict:
    materialized = list(rows)
    require(len(materialized) == len(set(materialized)), f"Duplicate row: {label}")
    if not ordered:
        materialized.sort()
    payload = "".join(f"{row}\n" for row in materialized).encode("utf-8")
    return {
        "canonical_bytes": len(payload),
        "canonicalization": (
            "ordered UTF-8 rows with LF terminators"
            if ordered
            else "lexicographically sorted UTF-8 rows with LF terminators"
        ),
        "count": len(materialized),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def descriptor_index(records: list[dict], label: str) -> dict:
    rows = []
    for record in records:
        descriptor = record["descriptor"]
        rows.append(
            "|".join(
                [
                    record["category"],
                    record.get("model_id", "-"),
                    record.get("layout_pair_id", "-"),
                    record.get("cell_id", "-"),
                    record.get("name", "-"),
                    descriptor["path"],
                    str(descriptor["bytes"]),
                    descriptor["sha256"],
                ]
            )
        )
    return line_index(rows, label)


def mapping_index(rows: list[dict], label: str) -> dict:
    fields = (
        "model_id",
        "layout_pair_id",
        "scheduled_ordinal",
        "scheduled_condition",
        "cell_id",
        "request_index",
        "source_request_receipt_sha256",
    )
    canonical_rows = [
        "|".join(str(row[field]) for field in fields) for row in rows
    ]
    index = line_index(canonical_rows, f"mapping {label}", ordered=True)
    expected_count, expected_bytes, expected_sha256 = EXPECTED_JOIN_INDEXES[label]
    require(
        (index["count"], index["canonical_bytes"], index["sha256"])
        == (expected_count, expected_bytes, expected_sha256),
        f"Mapping index changed: {label}",
    )
    index["canonicalization"] = (
        "ordered ASCII pipe-delimited mapping rows with LF terminators"
    )
    return index


def normalize_dependency(row: object, label: str) -> dict:
    require(type(row) is dict, f"Dependency is not an object: {label}")
    category = row.get("category")
    require(category in EXPECTED_DEPENDENCY_COUNTS, f"Bad category: {label}")
    require(
        row.get("availability")
        == "declared_pointer_only_current_availability_unknown",
        f"Availability promoted: {label}",
    )
    descriptor = require_descriptor(row.get("descriptor"), label)
    if category == "cell_receipt":
        expected_keys = {
            "availability",
            "category",
            "cell_id",
            "descriptor",
            "layout_pair_id",
            "model_id",
            "scheduled_condition",
            "scheduled_ordinal",
        }
    elif category == "timing_output":
        expected_keys = {
            "availability",
            "category",
            "descriptor",
            "model_id",
            "name",
        }
    elif category == "compiler_output":
        expected_keys = {"availability", "category", "descriptor", "name"}
    else:
        expected_keys = {
            "availability",
            "category",
            "descriptor",
            "layout_pair_id",
            "model_id",
            "name",
        }
    require(set(row) == expected_keys, f"Dependency keys changed: {label}")
    normalized = {key: row[key] for key in sorted(row) if key != "descriptor"}
    normalized["descriptor"] = descriptor
    return normalized


def dependency_lookup(
    dependencies: list[dict], category: str, **fields: object
) -> dict:
    matches = [
        row
        for row in dependencies
        if row["category"] == category
        and all(row.get(key) == value for key, value in fields.items())
    ]
    require(
        len(matches) == 1,
        f"Expected one dependency for {category} {fields}; found {len(matches)}",
    )
    return matches[0]


def descriptor_key(value: dict) -> tuple[str, int, str]:
    return value["path"], value["bytes"], value["sha256"]


def git_descriptor_key(value: dict) -> tuple[str, int, str]:
    return value["git_path"], value["bytes"], value["sha256"]


def build_audit() -> dict:
    spec, spec_source = source_json(SPEC_COMMIT, SPEC_PATH)
    preflight, preflight_source = source_json(PREFLIGHT_COMMIT, PREFLIGHT_PATH)
    join, join_source = source_json(JOIN_COMMIT, JOIN_PATH)
    sources = [spec_source, preflight_source, join_source]

    require(spec.get("spec_id") == "WMF-ABLATION-001", "Study changed")
    require(spec.get("core_episode_ceiling") == 232, "Core ceiling changed")
    require(
        spec["stages"]["development"]["episodes"] == 32,
        "Development design changed",
    )
    require(
        spec["stages"]["confirmation"]["episodes"] == 192,
        "Confirmation design changed",
    )
    require(spec["models"]["D2"]["selected"] is False, "D2 selection changed")
    require(
        spec.get("counts_do_not_authorize_execution") is True,
        "Execution boundary changed",
    )

    require(
        preflight.get("schema_version")
        == "wmf-development-annotation-raw-retrieval-preflight-v1",
        "Preflight schema changed",
    )
    require(
        join.get("schema_version") == "wmf-development-request-cell-join-v1",
        "Join schema changed",
    )
    require(
        preflight.get("study_id") == join.get("study_id") == "WMF-ABLATION-001",
        "Study IDs differ",
    )
    require(
        preflight["pins"]["spec_commit"] == SPEC_COMMIT,
        "Preflight spec pin changed",
    )
    require(
        join["pins"]["preflight_commit"] == PREFLIGHT_COMMIT,
        "Join preflight pin changed",
    )
    require(join["pins"]["results_commit"] == RESULTS_COMMIT, "Results pin changed")
    for label, report in (("preflight", preflight), ("join", join)):
        require_exact_json(
            report["global_study_snapshot"],
            GLOBAL_STUDY_SNAPSHOT,
            f"{label} study snapshot",
        )
        require_exact_json(report["gate_state"], GATE_STATE, f"{label} gates")
        require_exact_json(
            report["new_activity"], ZERO_ACTIVITY, f"{label} activity"
        )
    require_exact_json(preflight["protocol"], EXPECTED_PROTOCOL, "Preflight protocol")
    require_exact_json(
        join["annotation_and_release_boundary"],
        EXPECTED_ANNOTATION_AND_RELEASE_BOUNDARY,
        "Annotation and release boundary",
    )
    require_exact_json(join["timing_boundary"], EXPECTED_D1_TIMING, "Timing boundary")
    require(
        join["raw_retrieval_boundary"]["current_raw_availability_verified"]
        is False
        and join["raw_retrieval_boundary"]["raw_PVC_paths_revalidated"]
        is False
        and join["raw_retrieval_boundary"]["raw_PVC_bytes_opened"] is False
        and join["raw_retrieval_boundary"]["raw_bytes_rehashed"] is False,
        "Raw boundary promoted",
    )
    require(
        join["current_state_boundary"]["current_GM_PVC_worker_state_reconciled"]
        is False
        and join["current_state_boundary"]["live_quiescence_inferred"] is False
        and join["current_state_boundary"]["cached_active_scientific_ledger_cells"]
        == 0,
        "Current operations boundary promoted",
    )
    require(
        join["mapping"]["declared_request_identity_to_cell_mapping_materialized"]
        is True
        and join["mapping"][
            "declared_request_identity_to_cell_mapping_authenticated_by_order_replay"
        ]
        is True
        and join["mapping"]["raw_request_receipt_content_authenticated"] is False
        and join["mapping"]["rows"] == 1152
        and join["mapping"]["cells"] == 32
        and join["mapping"]["missing_identity_rows"] == 0
        and join["mapping"]["extra_identity_rows"] == 0
        and join["mapping"]["duplicate_mapping_keys"] == 0
        and join["mapping"]["cross_model_identity_overlap"] == 0,
        "Published mapping boundary changed",
    )

    raw_dependencies = [
        normalize_dependency(row, f"pointer {ordinal}")
        for ordinal, row in enumerate(preflight["pointer_dependencies"])
    ] + [
        normalize_dependency(row, f"fixture {ordinal}")
        for ordinal, row in enumerate(preflight["fixture_dependencies"])
    ]
    require(len(raw_dependencies) == EXPECTED_DEPENDENCIES, "Dependency count changed")
    paths = [row["descriptor"]["path"] for row in raw_dependencies]
    hashes = [row["descriptor"]["sha256"] for row in raw_dependencies]
    triples = [descriptor_key(row["descriptor"]) for row in raw_dependencies]
    require(len(paths) == len(set(paths)), "Dependency paths are duplicated")
    require(len(hashes) == len(set(hashes)), "Dependency hashes are duplicated")
    require(len(triples) == len(set(triples)), "Dependency triples are duplicated")
    for category, (expected_count, expected_bytes) in EXPECTED_DEPENDENCY_COUNTS.items():
        category_rows = [row for row in raw_dependencies if row["category"] == category]
        require(len(category_rows) == expected_count, f"{category} count changed")
        require(
            sum(row["descriptor"]["bytes"] for row in category_rows)
            == expected_bytes,
            f"{category} bytes changed",
        )
    require(
        sum(row["descriptor"]["bytes"] for row in raw_dependencies)
        == EXPECTED_DEPENDENCY_BYTES,
        "Combined dependency bytes changed",
    )
    for suffix, (expected_count, expected_bytes) in EXPECTED_FILE_TYPES.items():
        type_rows = [
            row
            for row in raw_dependencies
            if PurePosixPath(row["descriptor"]["path"]).suffix == suffix
        ]
        require(len(type_rows) == expected_count, f"{suffix} file count changed")
        require(
            sum(row["descriptor"]["bytes"] for row in type_rows) == expected_bytes,
            f"{suffix} byte count changed",
        )
    preflight_indexes = preflight["recovery_scope"]["indexes"]
    for name, (count, byte_count, sha256) in EXPECTED_PREFLIGHT_INDEXES.items():
        actual = preflight_indexes[name]
        require(
            (actual["count"], actual["canonical_bytes"], actual["sha256"])
            == (count, byte_count, sha256),
            f"Preflight index changed: {name}",
        )
    require(
        descriptor_index(raw_dependencies, "dependencies")["sha256"]
        == EXPECTED_PREFLIGHT_INDEXES["combined"][2],
        "Combined descriptor index cannot be reconstructed",
    )

    dependencies = sorted(
        raw_dependencies,
        key=lambda row: (
            row["category"],
            row.get("model_id", "-"),
            row.get("layout_pair_id", "-"),
            row.get("cell_id", "-"),
            row.get("name", "-"),
            row["descriptor"]["path"],
        ),
    )
    for ordinal, row in enumerate(dependencies):
        row["dependency_id"] = f"dependency-{ordinal:02d}"

    preflight_cells: dict[str, dict] = {}
    for ordinal, cell in enumerate(preflight["cells"]):
        require(type(cell) is dict and set(cell) == CELL_KEYS, f"Cell keys: {ordinal}")
        cell_id = cell["cell_id"]
        require(type(cell_id) is str and cell_id not in preflight_cells, "Duplicate cell")
        require(cell["model_id"] in MODELS, f"Bad cell model: {cell_id}")
        require(cell["layout_pair_id"] in LAYOUTS, f"Bad cell layout: {cell_id}")
        require(
            type(cell["scheduled_ordinal"]) is int
            and 0 <= cell["scheduled_ordinal"] < 4,
            f"Bad cell ordinal: {cell_id}",
        )
        require(
            cell["scheduled_behavioral_request_count"]
            == REQUESTS_PER_CELL[cell["model_id"]],
            f"Bad request count: {cell_id}",
        )
        require(
            cell["availability"]
            == "declared_pointer_only_current_availability_unknown",
            f"Cell availability promoted: {cell_id}",
        )
        require_descriptor(cell["raw_cell_receipt"], f"cell receipt {cell_id}")
        require_descriptor(cell["accepted_model_fixture"], f"fixture {cell_id}")
        require_git_descriptor(cell["aggregate_receipt"], f"aggregate {cell_id}")
        require_sha256(
            cell["aggregate_accepted_gate_record_sha256"], f"gate {cell_id}"
        )
        context = cell["layout_recovery_context"]
        require(type(context) is dict, f"Bad layout context: {cell_id}")
        require_descriptor(context["raw_capture_receipt"], f"capture {cell_id}")
        require_descriptor(
            context["model_fixtures"][cell["model_id"]], f"context fixture {cell_id}"
        )
        preflight_cells[cell_id] = cell
    require(len(preflight_cells) == 32, "Preflight cell count changed")

    per_cell_rows = join["mapping"]["per_cell"]
    require(type(per_cell_rows) is list and len(per_cell_rows) == 32, "Join cells changed")
    join_cells: dict[str, dict] = {}
    for ordinal, row in enumerate(per_cell_rows):
        require(type(row) is dict and set(row) == PER_CELL_KEYS, f"Join cell {ordinal}")
        cell_id = row["cell_id"]
        require(cell_id not in join_cells and cell_id in preflight_cells, "Join cell mismatch")
        cell = preflight_cells[cell_id]
        require(
            row["model_id"] == cell["model_id"]
            and row["layout_pair_id"] == cell["layout_pair_id"]
            and row["scheduled_condition"] == cell["scheduled_condition"]
            and row["scheduled_ordinal"] == cell["scheduled_ordinal"]
            and row["requests"] == cell["scheduled_behavioral_request_count"],
            f"Cell metadata differs: {cell_id}",
        )
        join_cells[cell_id] = row
    require(set(join_cells) == set(preflight_cells), "Cell key sets differ")

    request_rows = join["mapping"]["rows_detail"]
    require(type(request_rows) is list and len(request_rows) == 1152, "Requests changed")
    lineage_rows: list[dict] = []
    model_ordinals = {model: 0 for model in MODELS}
    by_cell: dict[str, list[dict]] = {cell_id: [] for cell_id in preflight_cells}
    identity_pairs: set[tuple[str, str]] = set()
    for ordinal, row in enumerate(request_rows):
        require(type(row) is dict and set(row) == REQUEST_KEYS, f"Request {ordinal}")
        cell_id = row["cell_id"]
        require(cell_id in join_cells, f"Unknown request cell: {ordinal}")
        cell = join_cells[cell_id]
        require(
            row["model_id"] == cell["model_id"]
            and row["layout_pair_id"] == cell["layout_pair_id"]
            and row["scheduled_condition"] == cell["scheduled_condition"]
            and row["scheduled_ordinal"] == cell["scheduled_ordinal"],
            f"Request metadata differs: {ordinal}",
        )
        require(
            type(row["request_index"]) is int
            and 0 <= row["request_index"] < REQUESTS_PER_CELL[row["model_id"]],
            f"Request index changed: {ordinal}",
        )
        receipt_sha256 = require_sha256(
            row["source_request_receipt_sha256"], f"request {ordinal}"
        )
        identity = (row["model_id"], receipt_sha256)
        require(identity not in identity_pairs, f"Duplicate request identity: {ordinal}")
        identity_pairs.add(identity)
        output = dict(row)
        output["model_request_ordinal"] = model_ordinals[row["model_id"]]
        model_ordinals[row["model_id"]] += 1
        lineage_rows.append(output)
        by_cell[cell_id].append(output)
    require(model_ordinals == REQUESTS_BY_MODEL, "Model request totals changed")
    require(len(identity_pairs) == 1152, "Request identity count changed")
    require(
        not ({sha for model, sha in identity_pairs if model == "D1"}
             & {sha for model, sha in identity_pairs if model == "N3"}),
        "Cross-model request identity overlap",
    )
    for cell_id, rows in by_cell.items():
        cell = join_cells[cell_id]
        require(
            [row["request_index"] for row in rows]
            == list(range(cell["requests"])),
            f"Request indexes are not contiguous: {cell_id}",
        )
        require(
            rows[0]["model_request_ordinal"] == cell["first_model_request_ordinal"]
            and rows[-1]["model_request_ordinal"]
            == cell["last_model_request_ordinal"],
            f"Model ordinal range differs: {cell_id}",
        )

    d1_rows = [row for row in request_rows if row["model_id"] == "D1"]
    n3_rows = [row for row in request_rows if row["model_id"] == "N3"]
    reconstructed_mapping_indexes = {
        "D1": mapping_index(d1_rows, "D1"),
        "N3": mapping_index(n3_rows, "N3"),
        "combined": mapping_index(request_rows, "combined"),
    }
    require_exact_json(
        reconstructed_mapping_indexes,
        join["mapping"]["indexes"],
        "Published mapping indexes",
    )
    identity_rows = [f"{model}\t{sha256}" for model, sha256 in identity_pairs]
    identity_index = line_index(identity_rows, "request identities")
    require(
        (
            identity_index["count"],
            identity_index["canonical_bytes"],
            identity_index["sha256"],
        )
        == EXPECTED_IDENTITY_INDEX,
        "Request identity index changed",
    )

    aggregate_sources = join["order_replay_authority"][
        "aggregate_receipt_sources"
    ]
    require(
        type(aggregate_sources) is list and len(aggregate_sources) == 8,
        "Aggregate sources changed",
    )
    aggregate_by_path = {}
    for source in aggregate_sources:
        require(
            type(source) is dict
            and set(source) == {"bytes", "commit", "path", "sha256"},
            "Aggregate source keys changed",
        )
        require(source["commit"] == RESULTS_COMMIT, "Aggregate commit changed")
        require_sha256(source["sha256"], f"aggregate {source.get('path')}")
        require(source["path"] not in aggregate_by_path, "Aggregate path duplicated")
        aggregate_by_path[source["path"]] = source
    aggregate_gate_by_layout: dict[str, set[str]] = {layout: set() for layout in LAYOUTS}
    for cell in preflight_cells.values():
        aggregate = cell["aggregate_receipt"]
        require(aggregate["git_path"] in aggregate_by_path, "Aggregate source missing")
        source = aggregate_by_path[aggregate["git_path"]]
        require(
            git_descriptor_key(aggregate)
            == (source["path"], source["bytes"], source["sha256"]),
            f"Aggregate source differs: {cell['cell_id']}",
        )
        aggregate_gate_by_layout[cell["layout_pair_id"]].add(
            cell["aggregate_accepted_gate_record_sha256"]
        )
    require(
        all(len(values) == 1 for values in aggregate_gate_by_layout.values()),
        "Accepted gate record differs across a layout",
    )

    dependency_by_path = {
        row["descriptor"]["path"]: row for row in dependencies
    }
    cell_lineage = []
    recovery_edges = []
    edge_roles = (
        "cell_receipt",
        "fixed_observation_capture_receipt",
        "fixed_observation_model_fixture",
        "timing_raw_development_sidecar",
        "timing_raw_request_inventory",
    )
    for cell in sorted(
        preflight_cells.values(),
        key=lambda row: (
            row["model_id"],
            row["layout_pair_id"],
            row["scheduled_ordinal"],
        ),
    ):
        model = cell["model_id"]
        layout = cell["layout_pair_id"]
        cell_id = cell["cell_id"]
        context = cell["layout_recovery_context"]
        edge_specs = [
            {
                "role": "cell_receipt",
                "dependency": dependency_lookup(
                    dependencies, "cell_receipt", cell_id=cell_id
                ),
                "binding_scope": (
                    "exact_declared_cell_receipt_and_declared_cell_request_identities"
                ),
            },
            {
                "role": "fixed_observation_capture_receipt",
                "dependency": dependency_lookup(
                    dependencies,
                    "fixed_observation_capture_receipt",
                    layout_pair_id=layout,
                    name="raw_capture_receipt",
                ),
                "binding_scope": "layout_recovery_context_only_no_request_meaning",
            },
            {
                "role": "fixed_observation_model_fixture",
                "dependency": dependency_lookup(
                    dependencies,
                    "fixed_observation_fixture",
                    layout_pair_id=layout,
                    model_id=model,
                    name=f"{model}_fixture",
                ),
                "binding_scope": (
                    "aggregate_declared_episode_input"
                    if model == "D1"
                    else "layout_model_inventory_context_only_not_aggregate_declared_input"
                ),
            },
            {
                "role": "timing_raw_development_sidecar",
                "dependency": dependency_lookup(
                    dependencies,
                    "timing_output",
                    model_id=model,
                    name="raw_development_timing_sidecar",
                ),
                "binding_scope": (
                    "declared_model_cohort_timing_output_content_not_authenticated"
                ),
            },
            {
                "role": "timing_raw_request_inventory",
                "dependency": dependency_lookup(
                    dependencies,
                    "timing_output",
                    model_id=model,
                    name="raw_request_inventory",
                ),
                "binding_scope": (
                    "declared_model_cohort_request_inventory_content_not_authenticated"
                ),
            },
        ]
        selected_dependencies = [row["dependency"] for row in edge_specs]
        require(
            descriptor_key(selected_dependencies[0]["descriptor"])
            == descriptor_key(cell["raw_cell_receipt"]),
            f"Cell dependency differs: {cell_id}",
        )
        require(
            descriptor_key(selected_dependencies[1]["descriptor"])
            == descriptor_key(context["raw_capture_receipt"]),
            f"Capture dependency differs: {cell_id}",
        )
        require(
            descriptor_key(selected_dependencies[2]["descriptor"])
            == descriptor_key(cell["accepted_model_fixture"])
            == descriptor_key(context["model_fixtures"][model]),
            f"Fixture dependency differs: {cell_id}",
        )
        cell_lineage.append(
            {
                "accepted_gate_record_sha256": cell[
                    "aggregate_accepted_gate_record_sha256"
                ],
                "aggregate_receipt": cell["aggregate_receipt"],
                "cell_id": cell_id,
                "first_model_request_ordinal": join_cells[cell_id][
                    "first_model_request_ordinal"
                ],
                "last_model_request_ordinal": join_cells[cell_id][
                    "last_model_request_ordinal"
                ],
                "layout_pair_id": layout,
                "model_id": model,
                "requests": join_cells[cell_id]["requests"],
                "scheduled_condition": cell["scheduled_condition"],
                "scheduled_ordinal": cell["scheduled_ordinal"],
            }
        )
        require(
            tuple(row["role"] for row in edge_specs) == edge_roles,
            f"Explicit edge roles changed: {cell_id}",
        )
        for edge_spec in edge_specs:
            role = edge_spec["role"]
            dependency = edge_spec["dependency"]
            binding_scope = edge_spec["binding_scope"]
            require(
                dependency_by_path[dependency["descriptor"]["path"]]
                is dependency,
                "Dependency lookup identity changed",
            )
            recovery_edges.append(
                {
                    "binding_scope": binding_scope,
                    "cell_id": cell_id,
                    "dependency_id": dependency["dependency_id"],
                    "role": role,
                }
            )
    require(len(cell_lineage) == 32, "Cell lineage count changed")
    require(len(recovery_edges) == EXPECTED_RECOVERY_EDGES, "Recovery edge count changed")
    for cell_id in preflight_cells:
        roles = [row["role"] for row in recovery_edges if row["cell_id"] == cell_id]
        require(tuple(roles) == edge_roles, f"Recovery roles changed: {cell_id}")
    cell_scope_dependency_ids = {
        row["dependency_id"] for row in recovery_edges
    }
    compiler_dependency_ids = {
        row["dependency_id"]
        for row in dependencies
        if row["category"] == "compiler_output"
    }
    all_dependency_ids = {row["dependency_id"] for row in dependencies}
    require(
        len(cell_scope_dependency_ids) == 48,
        "Cell-scoped unique dependency count changed",
    )
    require(
        len(compiler_dependency_ids) == 8,
        "Study-scoped compiler dependency count changed",
    )
    require(
        not (cell_scope_dependency_ids & compiler_dependency_ids)
        and cell_scope_dependency_ids | compiler_dependency_ids
        == all_dependency_ids,
        "Cell/study dependency scopes do not partition the 56-file manifest",
    )

    source_index = line_index(
        [
            f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}"
            for row in sources
        ],
        "sources",
    )
    source_index["source_bytes"] = sum(row["bytes"] for row in sources)
    request_lineage_index = line_index(
        [
            "|".join(
                str(row[field])
                for field in (
                    "model_id",
                    "model_request_ordinal",
                    "layout_pair_id",
                    "scheduled_ordinal",
                    "scheduled_condition",
                    "cell_id",
                    "request_index",
                    "source_request_receipt_sha256",
                )
            )
            for row in lineage_rows
        ],
        "request lineage",
        ordered=True,
    )
    cell_lineage_index = line_index(
        [
            "|".join(
                str(row[field])
                for field in (
                    "model_id",
                    "layout_pair_id",
                    "scheduled_ordinal",
                    "scheduled_condition",
                    "cell_id",
                    "requests",
                    "first_model_request_ordinal",
                    "last_model_request_ordinal",
                    "accepted_gate_record_sha256",
                )
            )
            for row in cell_lineage
        ],
        "cell lineage",
        ordered=True,
    )
    dependency_manifest_index = line_index(
        [
            "|".join(
                [
                    row["dependency_id"],
                    row["category"],
                    row.get("model_id", "-"),
                    row.get("layout_pair_id", "-"),
                    row.get("cell_id", "-"),
                    row.get("name", "-"),
                    row["descriptor"]["path"],
                    str(row["descriptor"]["bytes"]),
                    row["descriptor"]["sha256"],
                ]
            )
            for row in dependencies
        ],
        "dependency manifest",
        ordered=True,
    )
    recovery_edge_index = line_index(
        [
            f"{row['cell_id']}|{row['role']}|{row['dependency_id']}|{row['binding_scope']}"
            for row in recovery_edges
        ],
        "recovery edges",
        ordered=True,
    )
    aggregate_index = line_index(
        [
            f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}"
            for row in aggregate_sources
        ],
        "aggregate sources",
    )

    return {
        "schema_version": "wmf-development-request-recovery-lineage-v1",
        "study_id": "WMF-ABLATION-001",
        "input_pins": {
            "join_commit": JOIN_COMMIT,
            "preflight_commit": PREFLIGHT_COMMIT,
            "results_commit": RESULTS_COMMIT,
            "spec_commit": SPEC_COMMIT,
        },
        "authenticated_source_blobs": {
            **source_index,
            "sources": sorted(sources, key=lambda row: (row["commit"], row["path"])),
        },
        "global_study_snapshot": GLOBAL_STUDY_SNAPSHOT,
        "request_lineage": {
            "cells": 32,
            "cross_model_identity_overlap": 0,
            "index": request_lineage_index,
            "mapping_index_reconstruction": reconstructed_mapping_indexes,
            "request_identity_index": identity_index,
            "requests": 1152,
            "requests_by_model": REQUESTS_BY_MODEL,
            "requests_per_cell": REQUESTS_PER_CELL,
            "rows": lineage_rows,
        },
        "cell_lineage": {
            "cells": 32,
            "cells_by_model": {"D1": 16, "N3": 16},
            "index": cell_lineage_index,
            "rows": cell_lineage,
        },
        "recovery_manifest": {
            "current_availability": "unknown_not_observed",
            "declared_bytes": EXPECTED_DEPENDENCY_BYTES,
            "dependencies": dependencies,
            "dependency_index": dependency_manifest_index,
            "expected_descriptor_count": EXPECTED_DEPENDENCIES,
            "expected_counts_and_bytes_by_category": {
                key: {"bytes": value[1], "count": value[0]}
                for key, value in EXPECTED_DEPENDENCY_COUNTS.items()
            },
            "expected_counts_and_bytes_by_file_type": {
                key: {"bytes": value[1], "count": value[0]}
                for key, value in EXPECTED_FILE_TYPES.items()
            },
            "preflight_combined_descriptor_index": preflight_indexes["combined"],
        },
        "recovery_scope_edges": {
            "binding_boundary": (
                "Edges identify only joins justified by the frozen reports. They do not "
                "authenticate dependency contents or promote timing, fixture, capture, or "
                "request meaning before authorized byte verification and semantic replay."
            ),
            "edges": recovery_edges,
            "edges_per_cell": 5,
            "index": recovery_edge_index,
        },
        "study_scope_dependencies": {
            "all_dependencies_partitioned": True,
            "cell_scope_unique_dependencies": len(cell_scope_dependency_ids),
            "compiler_dependency_ids": sorted(compiler_dependency_ids),
            "compiler_dependencies": 8,
            "boundary": (
                "The frozen pair justifies exact byte identity for these eight compiler "
                "outputs at study scope only. It does not justify model, cell, request, "
                "timing, eligibility, or selection semantics from their filenames."
            ),
        },
        "aggregate_sources": {
            "count": 8,
            "index": aggregate_index,
            "rows": sorted(aggregate_sources, key=lambda row: row["path"]),
        },
        "pending_raw_verification": {
            "expected_manifest_created": True,
            "retrieval_released": False,
            "retrieval_queue_descriptor_created": False,
            "raw_paths_opened": 0,
            "raw_bytes_read": 0,
            "raw_descriptor_count_verified": None,
            "raw_descriptor_mismatch_count": None,
            "current_raw_availability_verified": False,
            "safe_to_release_new_work": False,
            "reason": (
                "An authorized GM operator or existing approved management path must "
                "first reconcile ali-owned coordinator/workers/task PVC state and publish "
                "a fresh results snapshot that reopens the release gate."
            ),
        },
        "future_bundle_verifier_contract": {
            "activation": (
                "Explicit --verify, --verify-root, --allow-byte-read, and "
                "--verification-output after an authorized isolated, quiescent retrieval "
                "bundle with no concurrent writers exists; default audit mode cannot "
                "open raw files."
            ),
            "bundle_root_represents": str(PVC_ROOT),
            "exact_expected_regular_files": EXPECTED_DEPENDENCIES,
            "rejects": [
                "missing file",
                "extra file",
                "extra directory",
                "symbolic link",
                "special file",
                "hard link",
                "duplicate inode alias",
                "path escape",
                "non-regular expected file",
                "declared byte-count drift",
                "declared SHA-256 drift",
                "file mutation while read",
                "tree mutation during verification",
            ],
            "observation_boundary": (
                "Verification detects inventory and metadata changes observed between "
                "the initial scan, hashing, final scan, and held-descriptor barrier. "
                "It does not establish external-writer quiescence, filesystem-snapshot "
                "atomicity, or continued immutability after the barrier."
            ),
            "receipt_does_not_promote": [
                "request-content authentication",
                "timing completeness",
                "camera or physical alignment",
                "production eligibility or request selection",
                "annotation media or human evidence",
                "resource freeze or confirmation release",
            ],
        },
        "timing_boundary": EXPECTED_D1_TIMING,
        "annotation_and_release_boundary": join[
            "annotation_and_release_boundary"
        ],
        "current_state_boundary": join["current_state_boundary"],
        "gate_state": GATE_STATE,
        "new_activity": ZERO_ACTIVITY,
        "claim_boundaries": {
            "lineage": (
                "The exact declared request-to-cell mapping is cross-bound to the exact "
                "declared recovery manifest and five cell-scoped recovery roles per cell; "
                "eight compiler dependencies remain study-scoped only."
            ),
            "raw": (
                "Default audit mode opened zero raw paths and read zero raw bytes; current "
                "availability and mismatch counts are unknown, not inferred missing or valid."
            ),
            "content": (
                "Even a future 56-file byte/SHA pass authenticates those files only. "
                "Request inventories and sidecar contents still require strict semantic replay."
            ),
            "timing": (
                "D1 remains 240 source-proven requests: 224 native-clock matched plus "
                "16 prefix-truncated; 672 incremental decodes remain source-unmapped. "
                "N3 contributes 240 retained identities without a new completeness claim."
            ),
            "annotation": (
                "No eligibility, selection, media, rater handoff, response, label, consensus, "
                "threshold, usability decision or estimate was created."
            ),
            "operations": (
                "Retained Git evidence does not establish current GM/PVC/queue/worker/GPU/"
                "child/reap/release-lock state or live quiescence."
            ),
            "science": (
                "No policy skill, forecast accuracy, model comparison or confirmation result "
                "is inferred. The 232-cell ceiling and every scientific stage gate are unchanged."
            ),
            "release": "No queue descriptor, retrieval job or scientific cell was released.",
        },
    }


def relative_bundle_path(descriptor: dict) -> PurePosixPath:
    path = PurePosixPath(descriptor["path"])
    relative = path.relative_to(PVC_ROOT)
    require(
        relative.parts
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"Unsafe bundle path: {path}",
    )
    return relative


def require_symlink_free_absolute_path(path: Path, *, leaf_may_not_exist: bool) -> None:
    require(path.is_absolute(), f"Path must be absolute: {path}")
    require(str(path) == os.path.abspath(str(path)), f"Path is not normalized: {path}")
    current = Path(path.anchor)
    parts = path.parts[1:]
    for ordinal, component in enumerate(parts):
        current /= component
        is_leaf = ordinal == len(parts) - 1
        try:
            info = current.lstat()
        except FileNotFoundError:
            require(
                leaf_may_not_exist and is_leaf,
                f"Path component does not exist: {current}",
            )
            continue
        require(not stat.S_ISLNK(info.st_mode), f"Path component is a symlink: {current}")


def open_absolute_directory_fd(path: Path) -> int:
    require(path.is_absolute(), f"Directory path must be absolute: {path}")
    require(str(path) == os.path.abspath(str(path)), f"Path is not normalized: {path}")
    current_fd = os.open(
        path.anchor,
        os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
    )
    try:
        for component in path.parts[1:]:
            next_fd = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def stat_metadata(info: os.stat_result) -> dict:
    return {
        "ctime_ns": info.st_ctime_ns,
        "dev": info.st_dev,
        "ino": info.st_ino,
        "mode": info.st_mode,
        "mtime_ns": info.st_mtime_ns,
        "nlink": info.st_nlink,
        "size": info.st_size,
    }


def atomic_write_new(path: Path, payload: bytes) -> None:
    require(path.is_absolute(), f"Output path must be absolute: {path}")
    require(str(path) == os.path.abspath(str(path)), f"Output is not normalized: {path}")
    require(path.name not in ("", ".", ".."), "Invalid output leaf")
    parent_fd = open_absolute_directory_fd(path.parent)
    temporary = f".{path.name}.tmp-{os.getpid()}"
    file_fd: int | None = None
    temporary_created = False
    try:
        try:
            os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ValueError(f"Refusing to overwrite verification receipt: {path}")
        file_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
            0o600,
            dir_fd=parent_fd,
        )
        temporary_created = True
        offset = 0
        while offset < len(payload):
            offset += os.write(file_fd, payload[offset:])
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = None
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if temporary_created:
            try:
                os.unlink(temporary, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)


def scan_bundle(root_fd: int, *, retain_fds: bool = False) -> tuple:
    files: dict[str, dict] = {}
    directories: dict[str, dict] = {}
    rejected: list[dict] = []
    held_directories: dict[str, int] = {}
    held_files: dict[str, int] = {}
    if retain_fds:
        held_directories["."] = os.dup(root_fd)
    stack: list[tuple[int, PurePosixPath]] = [(os.dup(root_fd), PurePosixPath())]
    try:
        while stack:
            directory_fd, prefix = stack.pop()
            try:
                with os.scandir(directory_fd) as entries:
                    for entry in entries:
                        relative = prefix / entry.name
                        relative_text = relative.as_posix()
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except OSError as error:
                            rejected.append(
                                {
                                    "path": relative_text,
                                    "reason": f"lstat_failed:{errno.errorcode.get(error.errno, error.errno)}",
                                }
                            )
                            continue
                        if stat.S_ISLNK(info.st_mode):
                            rejected.append(
                                {"path": relative_text, "reason": "symbolic_link"}
                            )
                        elif stat.S_ISDIR(info.st_mode):
                            try:
                                child_fd = os.open(
                                    entry.name,
                                    os.O_RDONLY
                                    | os.O_DIRECTORY
                                    | os.O_CLOEXEC
                                    | os.O_NOFOLLOW,
                                    dir_fd=directory_fd,
                                )
                            except OSError as error:
                                rejected.append(
                                    {
                                        "path": relative_text,
                                        "reason": f"directory_open_failed:{errno.errorcode.get(error.errno, error.errno)}",
                                    }
                                )
                            else:
                                pinned = stat_metadata(os.fstat(child_fd))
                                directories[relative_text] = pinned
                                if pinned != stat_metadata(info):
                                    rejected.append(
                                        {
                                            "path": relative_text,
                                            "reason": "directory_changed_while_opening",
                                        }
                                    )
                                if retain_fds:
                                    held_directories[relative_text] = os.dup(child_fd)
                                stack.append((child_fd, relative))
                        elif stat.S_ISREG(info.st_mode):
                            require(
                                relative_text not in files,
                                "Bundle path enumerated twice",
                            )
                            if retain_fds:
                                try:
                                    path_fd = os.open(
                                        entry.name,
                                        os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW,
                                        dir_fd=directory_fd,
                                    )
                                except OSError as error:
                                    rejected.append(
                                        {
                                            "path": relative_text,
                                            "reason": f"file_pin_failed:{errno.errorcode.get(error.errno, error.errno)}",
                                        }
                                    )
                                    files[relative_text] = stat_metadata(info)
                                else:
                                    pinned = stat_metadata(os.fstat(path_fd))
                                    files[relative_text] = pinned
                                    held_files[relative_text] = path_fd
                                    if pinned != stat_metadata(info):
                                        rejected.append(
                                            {
                                                "path": relative_text,
                                                "reason": "file_changed_while_pinning",
                                            }
                                        )
                            else:
                                files[relative_text] = stat_metadata(info)
                        else:
                            rejected.append(
                                {"path": relative_text, "reason": "special_file"}
                            )
            finally:
                os.close(directory_fd)
    except BaseException:
        for descriptor in (*held_directories.values(), *held_files.values()):
            os.close(descriptor)
        raise
    result = (
        files,
        directories,
        sorted(rejected, key=lambda row: (row["path"], row["reason"])),
    )
    if retain_fds:
        return (*result, held_directories, held_files)
    return result


def held_fd_metadata_barrier(
    held_directories: dict[str, int],
    held_files: dict[str, int],
    root_baseline: dict,
    directory_baselines: dict[str, dict],
    file_baselines: dict[str, dict],
    expected: dict[str, dict],
    *,
    allow_content_read: bool,
) -> tuple[dict, list[str], list[str], list[dict], int]:
    changed_directories: set[str] = set()
    changed_files: set[str] = set()
    failures: list[dict] = []
    barrier_bytes_read = 0
    root_after = root_baseline
    expected_children: dict[str, dict[str, dict]] = {
        relative: {} for relative in held_directories
    }
    for relative, metadata in {**directory_baselines, **file_baselines}.items():
        path = PurePosixPath(relative)
        parent = path.parent.as_posix()
        expected_children.setdefault(parent, {})[path.name] = metadata
    try:
        for relative, descriptor in sorted(held_directories.items()):
            try:
                with os.scandir(descriptor) as entries:
                    observed_children = {
                        entry.name: stat_metadata(entry.stat(follow_symlinks=False))
                        for entry in entries
                    }
                if observed_children != expected_children.get(relative, {}):
                    changed_directories.add(relative)
                current = stat_metadata(os.fstat(descriptor))
            except OSError as error:
                failures.append(
                    {
                        "path": relative,
                        "reason": f"directory_barrier_fstat_failed:{errno.errorcode.get(error.errno, error.errno)}",
                    }
                )
                continue
            baseline = root_baseline if relative == "." else directory_baselines[relative]
            if relative == ".":
                root_after = current
            if current != baseline:
                changed_directories.add(relative)
        for relative, descriptor in sorted(held_files.items()):
            read_fd: int | None = None
            try:
                current = stat_metadata(os.fstat(descriptor))
                if relative not in expected:
                    if current != file_baselines[relative]:
                        changed_files.add(relative)
                    continue
                if current != file_baselines[relative]:
                    changed_files.add(relative)
                    continue
                if not stat.S_ISREG(current["mode"]):
                    changed_files.add(relative)
                    failures.append(
                        {"path": relative, "reason": "file_barrier_not_regular"}
                    )
                    continue
                if not allow_content_read:
                    continue
                read_fd = os.open(
                    f"/proc/self/fd/{descriptor}",
                    os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
                )
                read_before = stat_metadata(os.fstat(read_fd))
                digest = hashlib.sha256()
                file_bytes_read = 0
                while True:
                    block = os.read(read_fd, 1024 * 1024)
                    if not block:
                        break
                    digest.update(block)
                    file_bytes_read += len(block)
                read_after = stat_metadata(os.fstat(read_fd))
                pinned_after = stat_metadata(os.fstat(descriptor))
                barrier_bytes_read += file_bytes_read
            except OSError as error:
                failures.append(
                    {
                        "path": relative,
                        "reason": f"file_barrier_fstat_failed:{errno.errorcode.get(error.errno, error.errno)}",
                    }
                )
                continue
            finally:
                if read_fd is not None:
                    os.close(read_fd)
            descriptor_expected = expected[relative]["descriptor"]
            if (
                read_before != file_baselines[relative]
                or read_after != file_baselines[relative]
                or pinned_after != file_baselines[relative]
                or file_bytes_read != descriptor_expected["bytes"]
                or digest.hexdigest() != descriptor_expected["sha256"]
            ):
                changed_files.add(relative)
    finally:
        for descriptor in (*held_directories.values(), *held_files.values()):
            os.close(descriptor)
    return (
        root_after,
        sorted(changed_directories),
        sorted(changed_files),
        failures,
        barrier_bytes_read,
    )


def verify_one_file(
    root_fd: int,
    relative: PurePosixPath,
    descriptor: dict,
    scanned: dict,
) -> dict:
    current_fd = os.dup(root_fd)
    path_fd: int | None = None
    read_fd: int | None = None
    try:
        try:
            for component in relative.parts[:-1]:
                next_fd = os.open(
                    component,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | os.O_CLOEXEC
                    | os.O_NOFOLLOW,
                    dir_fd=current_fd,
                )
                os.close(current_fd)
                current_fd = next_fd
            require(hasattr(os, "O_PATH"), "Linux O_PATH support is required")
            path_fd = os.open(
                relative.parts[-1],
                os.O_PATH | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=current_fd,
            )
        except OSError as error:
            return {
                "actual_bytes": None,
                "actual_sha256": None,
                "bytes_read": 0,
                "status": f"open_failed:{errno.errorcode.get(error.errno, error.errno)}",
            }
        before = os.fstat(path_fd)
        if not stat.S_ISREG(before.st_mode):
            return {
                "actual_bytes": before.st_size,
                "actual_sha256": None,
                "bytes_read": 0,
                "status": "not_regular",
            }
        scan_identity = (
            scanned["dev"],
            scanned["ino"],
            scanned["mode"],
            scanned["nlink"],
            scanned["size"],
            scanned["mtime_ns"],
            scanned["ctime_ns"],
        )
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if scan_identity != before_identity:
            return {
                "actual_bytes": before.st_size,
                "actual_sha256": None,
                "bytes_read": 0,
                "status": "changed_after_metadata_pass",
            }
        if before.st_size != descriptor["bytes"]:
            return {
                "actual_bytes": before.st_size,
                "actual_sha256": None,
                "bytes_read": 0,
                "status": "byte_count_mismatch",
            }
        try:
            read_fd = os.open(
                f"/proc/self/fd/{path_fd}",
                os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK,
            )
            read_before = os.fstat(read_fd)
        except OSError as error:
            return {
                "actual_bytes": before.st_size,
                "actual_sha256": None,
                "bytes_read": 0,
                "status": f"read_open_failed:{errno.errorcode.get(error.errno, error.errno)}",
            }
        read_identity = (
            read_before.st_dev,
            read_before.st_ino,
            read_before.st_mode,
            read_before.st_nlink,
            read_before.st_size,
            read_before.st_mtime_ns,
            read_before.st_ctime_ns,
        )
        if read_identity != before_identity:
            return {
                "actual_bytes": read_before.st_size,
                "actual_sha256": None,
                "bytes_read": 0,
                "status": "read_descriptor_identity_mismatch",
            }
        digest = hashlib.sha256()
        bytes_read = 0
        try:
            while True:
                block = os.read(read_fd, 1024 * 1024)
                if not block:
                    break
                digest.update(block)
                bytes_read += len(block)
            read_after = os.fstat(read_fd)
            path_after = os.fstat(path_fd)
        except OSError as error:
            return {
                "actual_bytes": None,
                "actual_sha256": None,
                "bytes_read": bytes_read,
                "status": f"read_failed:{errno.errorcode.get(error.errno, error.errno)}",
            }
        read_after_identity = (
            read_after.st_dev,
            read_after.st_ino,
            read_after.st_mode,
            read_after.st_nlink,
            read_after.st_size,
            read_after.st_mtime_ns,
            read_after.st_ctime_ns,
        )
        path_after_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_mode,
            path_after.st_nlink,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        actual_sha256 = digest.hexdigest()
        if (
            before_identity != read_after_identity
            or before_identity != path_after_identity
            or bytes_read != before.st_size
        ):
            status_value = "mutated_while_read"
        elif actual_sha256 != descriptor["sha256"]:
            status_value = "sha256_mismatch"
        else:
            status_value = "verified"
        return {
            "actual_bytes": read_after.st_size,
            "actual_sha256": actual_sha256,
            "bytes_read": bytes_read,
            "status": status_value,
        }
    finally:
        if read_fd is not None:
            os.close(read_fd)
        if path_fd is not None:
            os.close(path_fd)
        os.close(current_fd)


def verify_bundle(audit_report: dict, root: Path) -> dict:
    root_fd = open_absolute_directory_fd(root)
    try:
        root_before = stat_metadata(os.fstat(root_fd))
        dependencies = audit_report["recovery_manifest"]["dependencies"]
        expected = {
            relative_bundle_path(row["descriptor"]).as_posix(): row
            for row in dependencies
        }
        require(len(expected) == EXPECTED_DEPENDENCIES, "Verifier manifest changed")
        observed_files, observed_directories, rejected_entries = scan_bundle(root_fd)
        expected_files = set(expected)
        expected_directories = {
            PurePosixPath(*relative.parts[:depth]).as_posix()
            for relative_text in expected_files
            for relative in [PurePosixPath(relative_text)]
            for depth in range(1, len(relative.parts))
        }
        missing_files = sorted(expected_files - set(observed_files))
        extra_files = sorted(set(observed_files) - expected_files)
        missing_directories = sorted(expected_directories - set(observed_directories))
        extra_directories = sorted(set(observed_directories) - expected_directories)
        hardlink_paths = sorted(
            path for path, metadata in observed_files.items() if metadata["nlink"] != 1
        )
        inodes: dict[tuple[int, int], list[str]] = {}
        for path, metadata in observed_files.items():
            inodes.setdefault((metadata["dev"], metadata["ino"]), []).append(path)
        duplicate_inode_paths = sorted(
            path
            for paths in inodes.values()
            if len(paths) > 1
            for path in paths
        )
        size_mismatch_paths = sorted(
            relative_text
            for relative_text, dependency in expected.items()
            if relative_text in observed_files
            and observed_files[relative_text]["size"]
            != dependency["descriptor"]["bytes"]
        )
        metadata_passed = not any(
            (
                missing_files,
                extra_files,
                missing_directories,
                extra_directories,
                rejected_entries,
                hardlink_paths,
                duplicate_inode_paths,
                size_mismatch_paths,
            )
        )
        rows = []
        for relative_text in sorted(expected):
            dependency = expected[relative_text]
            if relative_text in missing_files:
                verification = {
                    "actual_bytes": None,
                    "actual_sha256": None,
                    "bytes_read": 0,
                    "status": "missing",
                }
            elif relative_text in size_mismatch_paths:
                verification = {
                    "actual_bytes": observed_files[relative_text]["size"],
                    "actual_sha256": None,
                    "bytes_read": 0,
                    "status": "byte_count_mismatch",
                }
            elif not metadata_passed:
                verification = {
                    "actual_bytes": observed_files[relative_text]["size"],
                    "actual_sha256": None,
                    "bytes_read": 0,
                    "status": "not_hashed_due_to_metadata_failure",
                }
            else:
                verification = verify_one_file(
                    root_fd,
                    PurePosixPath(relative_text),
                    dependency["descriptor"],
                    observed_files[relative_text],
                )
            rows.append(
                {
                    "dependency_id": dependency["dependency_id"],
                    "expected": dependency["descriptor"],
                    "observed_metadata": observed_files.get(relative_text),
                    "relative_path": relative_text,
                    **verification,
                }
            )
        final_scan_root = stat_metadata(os.fstat(root_fd))
        (
            final_files,
            final_directories,
            final_rejected_entries,
            held_directories,
            held_files,
        ) = scan_bundle(root_fd, retain_fds=True)
        final_inode_identities = [
            (metadata["dev"], metadata["ino"])
            for metadata in final_files.values()
        ]
        final_metadata_passed = (
            set(final_files) == expected_files
            and set(final_directories) == expected_directories
            and not final_rejected_entries
            and all(metadata["nlink"] == 1 for metadata in final_files.values())
            and len(set(final_inode_identities)) == len(final_inode_identities)
            and all(
                final_files[relative]["size"] == expected[relative]["descriptor"]["bytes"]
                for relative in expected_files
            )
        )
        (
            root_after,
            barrier_changed_directories,
            barrier_changed_files,
            barrier_failures,
            barrier_bytes_read,
        ) = held_fd_metadata_barrier(
            held_directories,
            held_files,
            final_scan_root,
            final_directories,
            final_files,
            expected,
            allow_content_read=metadata_passed and final_metadata_passed,
        )
    finally:
        os.close(root_fd)
    final_missing_files = sorted(expected_files - set(final_files))
    final_extra_files = sorted(set(final_files) - expected_files)
    final_missing_directories = sorted(expected_directories - set(final_directories))
    final_extra_directories = sorted(set(final_directories) - expected_directories)
    changed_file_metadata = sorted(
        path
        for path in set(observed_files) & set(final_files)
        if observed_files[path] != final_files[path]
    )
    changed_directory_metadata = sorted(
        path
        for path in set(observed_directories) & set(final_directories)
        if observed_directories[path] != final_directories[path]
    )
    tree_mutated = any(
        (
            root_before != root_after,
            observed_files != final_files,
            observed_directories != final_directories,
            rejected_entries != final_rejected_entries,
            barrier_changed_directories,
            barrier_changed_files,
            barrier_failures,
        )
    )
    missing_files = sorted(set(missing_files) | set(final_missing_files))
    extra_files = sorted(set(extra_files) | set(final_extra_files))
    missing_directories = sorted(
        set(missing_directories) | set(final_missing_directories)
    )
    extra_directories = sorted(
        set(extra_directories) | set(final_extra_directories)
    )
    rejected_entries = sorted(
        {
            (row["path"], row["reason"])
            for row in rejected_entries + final_rejected_entries
        }
    )
    rejected_entries = [
        {"path": path, "reason": reason} for path, reason in rejected_entries
    ]
    final_hardlink_paths = {
        path for path, metadata in final_files.items() if metadata["nlink"] != 1
    }
    hardlink_paths = sorted(set(hardlink_paths) | final_hardlink_paths)
    final_inodes: dict[tuple[int, int], list[str]] = {}
    for path, metadata in final_files.items():
        final_inodes.setdefault((metadata["dev"], metadata["ino"]), []).append(path)
    final_duplicate_inode_paths = {
        path
        for paths in final_inodes.values()
        if len(paths) > 1
        for path in paths
    }
    duplicate_inode_paths = sorted(
        set(duplicate_inode_paths) | final_duplicate_inode_paths
    )
    if tree_mutated:
        for row in rows:
            if row["status"] == "verified":
                row["status"] = "tree_mutated_after_hash"
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    verified = status_counts.get("verified", 0)
    passed = (
        verified == EXPECTED_DEPENDENCIES
        and len(status_counts) == 1
        and not missing_files
        and not extra_files
        and not missing_directories
        and not extra_directories
        and not rejected_entries
        and not hardlink_paths
        and not duplicate_inode_paths
        and not tree_mutated
    )
    return {
        "schema_version": "wmf-development-raw-retrieval-byte-verification-v1",
        "study_id": "WMF-ABLATION-001",
        "status": "verified" if passed else "failed",
        "mode": "explicit_isolated_retrieval_bundle_byte_verification",
        "lineage_pins": audit_report["input_pins"],
        "expected_manifest_index": audit_report["recovery_manifest"][
            "dependency_index"
        ],
        "bundle_contract": {
            "exact_file_set_required": True,
            "isolated_quiescent_bundle_without_concurrent_writers_required": True,
            "provided_root": str(root),
            "root_represents": str(PVC_ROOT),
            "observation_boundary": (
                "The receipt detects changes observed across two inventories, hashing, "
                "and a held-descriptor metadata barrier. It does not prove external-writer "
                "quiescence, atomic filesystem snapshotting, or post-barrier immutability."
            ),
        },
        "verification_passed": passed,
        "summary": {
            "expected_descriptors": EXPECTED_DEPENDENCIES,
            "expected_bytes": EXPECTED_DEPENDENCY_BYTES,
            "extra_files": len(extra_files),
            "extra_directories": len(extra_directories),
            "hardlink_paths": len(hardlink_paths),
            "duplicate_inode_paths": len(duplicate_inode_paths),
            "missing_files": len(missing_files),
            "missing_directories": len(missing_directories),
            "rejected_entries": len(rejected_entries),
            "size_mismatch_files": len(size_mismatch_paths),
            "tree_mutated_during_verification": tree_mutated,
            "status_counts": dict(sorted(status_counts.items())),
            "verified_descriptors": verified,
            "verified_bytes": sum(
                row["bytes_read"] for row in rows if row["status"] == "verified"
            ),
        },
        "missing_paths": missing_files,
        "extra_paths": extra_files,
        "missing_directories": missing_directories,
        "extra_directories": extra_directories,
        "hardlink_paths": hardlink_paths,
        "duplicate_inode_paths": duplicate_inode_paths,
        "size_mismatch_paths": size_mismatch_paths,
        "post_hash_tree_check": {
            "barrier_changed_directories": barrier_changed_directories,
            "barrier_changed_files": barrier_changed_files,
            "barrier_failures": barrier_failures,
            "barrier_bytes_read": barrier_bytes_read,
            "barrier_passed": not any(
                (
                    barrier_changed_directories,
                    barrier_changed_files,
                    barrier_failures,
                )
            ),
            "changed_directory_metadata": changed_directory_metadata,
            "changed_file_metadata": changed_file_metadata,
            "final_extra_directories": final_extra_directories,
            "final_extra_files": final_extra_files,
            "final_missing_directories": final_missing_directories,
            "final_missing_files": final_missing_files,
            "root_after": root_after,
            "root_before": root_before,
            "held_fd_barrier_performed": True,
            "tree_mutated": tree_mutated,
        },
        "rejected_entries": rejected_entries,
        "descriptor_results": rows,
        "content_and_science_boundary": {
            "retrieved_copy_bytes_verified": passed,
            "current_live_PVC_availability_verified": False,
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "request_content_authenticated": False,
            "timing_complete": False,
            "camera_alignment_authenticated": False,
            "production_eligibility_determined": False,
            "request_selection_created": False,
            "annotation_media_created": 0,
            "human_responses": 0,
            "labels": 0,
            "forecast_accuracy_estimates": 0,
            "resource_freeze_authenticated": False,
            "confirmation_inference_cells": 0,
            "safe_to_release_confirmation": False,
        },
        "release_boundary": {
            "job_id": None,
            "queue_descriptor_released": False,
            "results_branch_written": False,
            "safe_to_release_new_work": False,
            "safe_to_release_confirmation": False,
        },
        "new_activity": ZERO_ACTIVITY,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify a canonical lineage report")
    parser.add_argument("--output", type=Path, help="Write the canonical lineage report")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enter explicit byte-verification mode",
    )
    parser.add_argument(
        "--verify-root",
        type=Path,
        help="Explicitly verify an authorized isolated 56-file retrieval bundle",
    )
    parser.add_argument(
        "--allow-byte-read",
        action="store_true",
        help="Required acknowledgement for explicit byte-verification mode",
    )
    parser.add_argument(
        "--verification-output",
        type=Path,
        help="Write a canonical byte-verification receipt (required with --verify-root)",
    )
    args = parser.parse_args()
    require(not (args.check and args.output), "Use only one of --check and --output")
    verification_requested = any(
        (
            args.verify,
            args.verify_root is not None,
            args.allow_byte_read,
            args.verification_output is not None,
        )
    )
    if verification_requested:
        require(not args.check and not args.output, "Verification is a separate mode")
        require(
            args.verify
            and args.verify_root is not None
            and args.allow_byte_read
            and args.verification_output is not None,
            "Verification requires --verify, --verify-root, --allow-byte-read, and --verification-output",
        )
    else:
        require(not args.verify and not args.allow_byte_read, "Incomplete verification mode")

    report = build_audit()
    if verification_requested:
        require_symlink_free_absolute_path(
            args.verification_output, leaf_may_not_exist=True
        )
        try:
            args.verification_output.relative_to(args.verify_root)
        except ValueError:
            pass
        else:
            raise ValueError("Verification receipt must be outside the retrieval root")
        receipt = verify_bundle(report, args.verify_root)
        atomic_write_new(args.verification_output, canonical_bytes(receipt))
        if receipt["verification_passed"]:
            print(
                "Verified all 56 declared recovery descriptors "
                f"({EXPECTED_DEPENDENCY_BYTES} bytes)"
            )
        else:
            print(
                "Retrieval bundle verification failed closed; receipt written to "
                f"{args.verification_output}"
            )
            raise SystemExit(1)
        return

    encoded = canonical_bytes(report)
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved recovery lineage differs")
        print("Verified 1,152 requests, 32 cells, 56 dependencies, and 160 edges")
    elif args.output:
        args.output.write_bytes(encoded)
        print(f"Wrote {args.output} ({len(encoded)} bytes)")
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
