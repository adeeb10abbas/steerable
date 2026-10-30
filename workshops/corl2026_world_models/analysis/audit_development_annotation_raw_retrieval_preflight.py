#!/usr/bin/env python3
"""Build a fail-closed development raw-retrieval preflight without opening raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[3]
PVC_ROOT = "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912"

SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
FIXTURE_COMMIT = "4dbb162020fff36d3791735d8bcb665610b30170"
READINESS_COMMIT = "84332bf638595ff7ed955fd263586ecc0e06ab93"
IDENTITY_COMMIT = "33a7483b4cc8255fd97fadb1684b3a4082ee3c9f"
PARITY_COMMIT = "9c1a17acda5f6565cf116c26b941e47bb4c681d6"
POINTER_COMMIT = "a0cfa6cabdf76d230026c4055729630cdf8e93f9"

SPEC_PATH = "workshops/corl2026_world_models/experiments/forecast_layout/ablation_spec.json"
FIXTURE_PATH = "workshops/corl2026_world_models/results/fixed_observation_capture_inventory.json"
READINESS_PATH = "workshops/corl2026_world_models/results/development_annotation_readiness_inventory.json"
IDENTITY_PATH = "workshops/corl2026_world_models/results/development_request_identity_spine.json"
PARITY_PATH = "workshops/corl2026_world_models/results/development_design_execution_parity.json"
POINTER_PATH = "workshops/corl2026_world_models/results/development_annotation_input_pointer_inventory.json"

EXPECTED_SOURCES = {
    (SPEC_COMMIT, SPEC_PATH): (
        8590,
        "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
    ),
    (FIXTURE_COMMIT, FIXTURE_PATH): (
        53104,
        "76c6e12920608c00212048fd807bfe42e7058d10ebee7229657e5d62b5ba3a54",
    ),
    (READINESS_COMMIT, READINESS_PATH): (
        10092,
        "41303ab00b78f90f0f7e3fc721550a8ca6547f74f8d2c38f9b912aae06e8c672",
    ),
    (IDENTITY_COMMIT, IDENTITY_PATH): (
        181453,
        "9b534d2e821b2d98ec128cd87f5c32b4674177b6a308286e161801b6a2d58d8e",
    ),
    (PARITY_COMMIT, PARITY_PATH): (
        92895,
        "d19c574d539ef695f1d07cf0a913c5efd0102643c09e400e2c23c546d8e460e1",
    ),
    (POINTER_COMMIT, POINTER_PATH): (
        56533,
        "0128232a690b6aeb7a77729d9902884fa2e4d95ffc1f2a6c0c4b60ffb177475e",
    ),
}

EXPECTED_SOURCE_INDEX = {
    "count": 6,
    "source_bytes": 402667,
    "canonical_bytes": 1173,
    "canonicalization": "lexicographically sorted UTF-8 rows with LF terminators",
    "sha256": "87c7adb9f6f400e29446ee0407a8f22462f38420868693be8eaacbc1ea020464",
}

LAYOUTS = ("D01", "D02", "D03", "D04")
MODELS = ("D1", "N3")
REQUESTS_PER_CELL = {"D1": 57, "N3": 15}
REQUESTS_BY_MODEL = {"D1": 912, "N3": 240}
EXPECTED_IDENTITY_INDEXES = {
    "D1": {
        "canonical_bytes": 62016,
        "count": 912,
        "sha256": "0ca707976b59e0957b4bb636982743bb4e5458f5bd08f8177eda09dcec0f0110",
    },
    "N3": {
        "canonical_bytes": 16320,
        "count": 240,
        "sha256": "d25a670fbc024960aeff5d667b4ccf9c6b90b2f3c24691348bc1db3bd0367051",
    },
    "combined": {
        "canonical_bytes": 78336,
        "count": 1152,
        "sha256": "3a6206f1e5fc790f1b8c92d9d023c1ce71f969ec95c19ea4482073a96cb86a71",
    },
}

EXPECTED_RECOVERY_INDEXES = {
    "cell_receipts": (32, 10384, "93268de436fea9788913cea35d8aac74a089fa8565d446975c130fe2b0441220"),
    "timing_outputs": (4, 1047, "158544edb2f77651277affc00375d1bed9c897757e27ea5860d52a835f2c78c6"),
    "compiler_outputs": (8, 2474, "4b88b8c7fb4523b462da78bd3d9b2f434477bcef2b643fb7dc0ca99702617490"),
    "fixture_inputs": (12, 3112, "2156b35857d4bea77906ce17bf20f9b94f5ff78cce22db222304901cec8cfe18"),
    "combined": (56, 17017, "518ef94a22d156bba50640e7b42e94bc044e3d737dac12bd967cf8eb0162adb7"),
    "cell_lineage": (32, 37360, "0fc384c9b32a415ae8ee8fbd7231327d04a3f0b806020382f2e85acf82117666"),
}
EXPECTED_COMBINED_DECLARED_BYTES = 32664538
EXPECTED_READINESS_DETAIL_BYTES = 3181
EXPECTED_READINESS_DETAIL_SHA256 = (
    "16c72ffde39f6d18e3231861ac62d0c55c6aa1ff9937679454300df0470c78c6"
)

EXPECTED_HISTORICAL_TIMING_SCOPE = {
    "boundary": (
        "Pinned receipt metadata only; no raw timing output was opened and no current raw "
        "availability or broader timing coverage is inferred."
    ),
    "by_model": {
        "D1": {
            "historical_receipt_requests": 912,
            "physical_time_coverage_complete": False,
            "physical_time_qualified": True,
            "physical_time_qualified_scope": "source-proven full conditioning-origin decodes only",
            "request_timing_coverage": {
                "action_prefix_truncated_request_count": 16,
                "action_prefix_truncated_target_binding_count": 32,
                "native_clock_matched_request_count": 224,
                "native_clock_matched_target_binding_count": 448,
                "source_proven_full_decode_request_count": 240,
                "source_unmapped_incremental_decode_request_count": 672,
                "total_request_count": 912,
                "unmapped_potential_authority_target_count": 1344,
            },
        },
        "N3": {
            "historical_receipt_requests": 240,
            "physical_time_coverage_complete_field_present": False,
            "physical_time_qualified": True,
            "request_timing_coverage_breakdown_present": False,
        },
    },
}

EXPECTED_ANNOTATION_PACKAGING = {
    "annotation_media_created": False,
    "development_selected_request_ceiling": 128,
    "human_handoff_ready": False,
    "human_labels": 0,
    "human_responses": 0,
    "production_eligibility_determined": False,
    "production_request_selection_manifest_created": False,
    "request_sample_cap_per_episode": 4,
    "request_sampling_seed": 2026091302,
    "selected_request_count": 0,
}

EXPECTED_REQUEST_IDENTITY_POPULATION = {
    "boundary": (
        "The committed request identities and historical cell pointers are authenticated "
        "as separate populations. No request identity is joined to a cell; that mapping "
        "requires authorized raw inventory retrieval and replay."
    ),
    "independent_raw_inventory_replay_performed": False,
    "indexes": EXPECTED_IDENTITY_INDEXES,
    "joined_to_cell_pointers": False,
    "request_to_cell_mapping_authenticated": False,
    "request_to_cell_mapping_materialized": False,
    "rows": 1152,
    "rows_by_model": REQUESTS_BY_MODEL,
    "timing_receipt_identity_sets_match_committed_spine": True,
}

PARITY_CELL_KEYS = {
    "accepted_fixture",
    "action_cap",
    "aggregate_receipt",
    "candidate_effective_policy_seed",
    "cell_id",
    "command",
    "layout_arm",
    "layout_pair_id",
    "model_id",
    "phase",
    "raw_cell_receipt_declaration",
    "replicate_id",
    "scheduled_condition",
    "scheduled_ordinal",
    "selected_in_frozen_core",
    "stop_on_success",
}

POINTER_KEYS_BY_CATEGORY = {
    "cell_receipt": {
        "category",
        "cell_id",
        "descriptor",
        "layout_pair_id",
        "model_id",
        "scheduled_condition",
        "scheduled_ordinal",
    },
    "timing_output": {"category", "descriptor", "model_id", "name"},
    "compiler_output": {"category", "descriptor", "name"},
}

GLOBAL_SNAPSHOT = {
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
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
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def require_exact_json(value: object, expected: object, label: str) -> None:
    require(canonical_bytes(value) == canonical_bytes(expected), f"{label} changed")


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
    expected_bytes, expected_sha256 = EXPECTED_SOURCES[(commit, path)]
    payload = git_blob(commit, path)
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    require(len(payload) == expected_bytes, f"Source byte count changed: {path}")
    require(actual_sha256 == expected_sha256, f"Source SHA-256 changed: {path}")
    return load_json(payload, f"{commit}:{path}"), {
        "bytes": len(payload),
        "commit": commit,
        "path": path,
        "sha256": actual_sha256,
    }


def require_sha256(value: object, label: str) -> str:
    require(type(value) is str and len(value) == 64, f"Invalid SHA-256: {label}")
    require(all(char in "0123456789abcdef" for char in value), f"Invalid SHA-256: {label}")
    return value


def require_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Descriptor is not an object: {label}")
    require(set(value) == {"bytes", "path", "sha256"}, f"Descriptor keys changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Invalid bytes: {label}")
    require(type(value["path"]) is str, f"Invalid path: {label}")
    path = PurePosixPath(value["path"])
    require(path.is_absolute(), f"Descriptor path is not absolute: {label}")
    require(".." not in path.parts, f"Descriptor path traversal: {label}")
    require(value["path"].startswith(PVC_ROOT + "/"), f"Descriptor leaves PVC root: {label}")
    require_sha256(value["sha256"], label)
    return dict(value)


def require_compact_descriptor(value: object, label: str, layout_id: str) -> dict:
    require(type(value) is dict, f"Compact descriptor is not an object: {label}")
    require(
        set(value) == {"bytes", "git_path", "layout_pair_id", "sha256"},
        f"Compact descriptor keys changed: {label}",
    )
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Invalid bytes: {label}")
    require(value["layout_pair_id"] == layout_id, f"Compact layout changed: {label}")
    require(type(value["git_path"]) is str and not value["git_path"].startswith("/"), f"Bad Git path: {label}")
    require(".." not in PurePosixPath(value["git_path"]).parts, f"Git path traversal: {label}")
    require_sha256(value["sha256"], label)
    return dict(value)


def require_git_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Git descriptor is not an object: {label}")
    require(set(value) == {"bytes", "git_path", "sha256"}, f"Git descriptor keys changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Invalid Git bytes: {label}")
    require(type(value["git_path"]) is str and not value["git_path"].startswith("/"), f"Bad Git path: {label}")
    require(".." not in PurePosixPath(value["git_path"]).parts, f"Git path traversal: {label}")
    require_sha256(value["sha256"], label)
    return dict(value)


def require_zero_activity(value: object, label: str) -> None:
    require(type(value) is dict and value, f"Missing activity object: {label}")
    for key, count in value.items():
        require(type(count) is int and count == 0, f"Nonzero activity: {label}.{key}")


def line_index(rows: list[str], label: str) -> dict:
    require(len(rows) == len(set(rows)), f"Duplicate index row: {label}")
    payload = "".join(f"{row}\n" for row in sorted(rows)).encode("utf-8")
    return {
        "canonical_bytes": len(payload),
        "canonicalization": "lexicographically sorted UTF-8 rows with LF terminators",
        "count": len(rows),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def descriptor_index(records: list[dict], label: str) -> dict:
    rows = [
        "|".join(
            [
                record["category"],
                record.get("model_id", "-"),
                record.get("layout_pair_id", "-"),
                record.get("cell_id", "-"),
                record.get("name", "-"),
                record["descriptor"]["path"],
                str(record["descriptor"]["bytes"]),
                record["descriptor"]["sha256"],
            ]
        )
        for record in records
    ]
    return line_index(rows, label)


def audit() -> dict:
    spec, spec_source = source_json(SPEC_COMMIT, SPEC_PATH)
    fixture, fixture_source = source_json(FIXTURE_COMMIT, FIXTURE_PATH)
    readiness, readiness_source = source_json(READINESS_COMMIT, READINESS_PATH)
    identity, identity_source = source_json(IDENTITY_COMMIT, IDENTITY_PATH)
    parity, parity_source = source_json(PARITY_COMMIT, PARITY_PATH)
    pointer, pointer_source = source_json(POINTER_COMMIT, POINTER_PATH)
    sources = [
        spec_source,
        fixture_source,
        readiness_source,
        identity_source,
        parity_source,
        pointer_source,
    ]
    source_rows = [
        f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}" for row in sources
    ]
    source_index = line_index(source_rows, "source")
    source_index["source_bytes"] = sum(row["bytes"] for row in sources)
    require(source_index == EXPECTED_SOURCE_INDEX, "Authenticated source index changed")

    require(spec.get("spec_id") == "WMF-ABLATION-001", "Study specification changed")
    require(type(spec.get("core_episode_ceiling")) is int and spec["core_episode_ceiling"] == 232, "Core ceiling changed")
    require(type(spec["stages"]["development"]["episodes"]) is int and spec["stages"]["development"]["episodes"] == 32, "Development design changed")
    require(type(spec["stages"]["confirmation"]["episodes"]) is int and spec["stages"]["confirmation"]["episodes"] == 192, "Confirmation design changed")
    require(spec["models"]["D2"]["selected"] is False, "D2 selection changed")
    require(type(spec["analysis"]["request_sample_cap_per_episode"]) is int and spec["analysis"]["request_sample_cap_per_episode"] == 4, "Sample cap changed")
    require(type(spec["analysis"]["request_sampling_seed"]) is int and spec["analysis"]["request_sampling_seed"] == 2026091302, "Sampling seed changed")
    require(
        spec["analysis"]["request_manifest_timing"]
        == "algorithm and seed frozen before confirmation; selection manifest generated from actual metadata after execution and before labels",
        "Selection timing changed",
    )
    require(type(spec["annotation"]["raters"]) is int and spec["annotation"]["raters"] == 2, "Independent-rater requirement changed")
    require(spec["counts_do_not_authorize_execution"] is True, "Execution boundary changed")

    reports = [readiness, identity, parity, pointer]
    expected_schemas = [
        "wmf-development-annotation-readiness-inventory-v1",
        "wmf-development-request-identity-spine-v1",
        "wmf-development-design-execution-parity-v1",
        "wmf-development-annotation-input-pointer-inventory-v1",
    ]
    for report, schema in zip(reports, expected_schemas, strict=True):
        require(report.get("schema_version") == schema, f"Schema changed: {schema}")
        require(report.get("study_id") == "WMF-ABLATION-001", f"Study changed: {schema}")
        require_exact_json(report["global_study_snapshot"], GLOBAL_SNAPSHOT, f"Snapshot: {schema}")
        gate_key = "readiness" if report is readiness else "gate_state"
        require_exact_json(report[gate_key], GATE_STATE, f"Gate state: {schema}")
        require_zero_activity(report["new_activity"], schema)

    require_exact_json(readiness["design"], {
        "confirmation_cells": 192,
        "core_episode_ceiling": 232,
        "development_cells": 32,
        "development_selected_request_ceiling": 128,
        "independent_raters_required": 2,
        "optional_D2_selected": False,
        "request_sample_cap_per_episode": 4,
    }, "Readiness design")
    readiness_details = {
        key: readiness[key]
        for key in [
            "camera_and_media_gate",
            "fixture_boundary",
            "human_annotation_gate",
            "resource_and_release_gate",
            "timing_readiness",
        ]
    }
    readiness_detail_payload = canonical_bytes(readiness_details)
    require(
        len(readiness_detail_payload) == EXPECTED_READINESS_DETAIL_BYTES
        and hashlib.sha256(readiness_detail_payload).hexdigest()
        == EXPECTED_READINESS_DETAIL_SHA256,
        "Readiness detail boundary changed",
    )
    require(readiness["camera_and_media_gate"]["media_renderable_for_human_annotation"] is False, "Media gate promoted")
    require(readiness["human_annotation_gate"]["production_request_selection_manifest_authenticated_by_this_audit"] is False, "Selection gate promoted")
    require(readiness["human_annotation_gate"]["human_handoff_ready"] is False, "Human handoff promoted")
    require(
        type(readiness["resource_and_release_gate"]["confirmation_inference_cells"])
        is int
        and readiness["resource_and_release_gate"]["confirmation_inference_cells"] == 0,
        "Confirmation count changed",
    )
    require(readiness["resource_and_release_gate"]["safe_to_release_confirmation"] is False, "Release gate promoted")
    require(readiness["fixture_boundary"]["raw_pvc_revalidated"] is False, "Fixture raw availability promoted")

    require(fixture.get("schema_version") == "wmf-fixed-observation-capture-inventory-audit-v1", "Fixture schema changed")
    require(fixture.get("results_commit") == "12839900e5c46299e11fff4813e8602949055b86", "Fixture results pin changed")
    require(fixture.get("raw_pvc_revalidated") is False, "Fixture raw PVC status changed")
    require(type(fixture.get("new_model_requests")) is int and fixture["new_model_requests"] == 0, "Fixture model requests changed")
    require(type(fixture.get("new_behavioral_actions")) is int and fixture["new_behavioral_actions"] == 0, "Fixture actions changed")
    require(type(fixture.get("new_behavioral_cells")) is int and fixture["new_behavioral_cells"] == 0, "Fixture cells changed")
    require(type(fixture.get("labels_created")) is int and fixture["labels_created"] == 0, "Fixture labels changed")
    require(fixture.get("confirmation_released") is False, "Fixture release changed")
    require(fixture.get("forecast_accuracy_estimated") is False, "Fixture estimate changed")

    fixture_by_layout = {}
    fixture_dependencies = []
    for row in fixture["layouts"]:
        if row.get("stage") != "development":
            continue
        layout_id = row.get("layout_pair_id")
        require(layout_id in LAYOUTS, f"Unexpected development fixture: {layout_id}")
        require(layout_id not in fixture_by_layout, f"Duplicate development fixture: {layout_id}")
        require(row["candidate_id"] == f"{layout_id}__candidate_00", f"Candidate changed: {layout_id}")
        require(row["environment_seed"] == 2026091100 + int(layout_id[1:]), f"Environment seed changed: {layout_id}")
        require_sha256(row["gate_receipt_sha256"], f"gate {layout_id}")
        require_sha256(row["pose_manifest_sha256"], f"pose {layout_id}")
        compact = require_compact_descriptor(row["compact_receipt"], f"compact {layout_id}", layout_id)
        d1_fixture = require_descriptor(row["D1_fixture"], f"D1 fixture {layout_id}")
        n3_fixture = require_descriptor(row["N3_fixture"], f"N3 fixture {layout_id}")
        capture = require_descriptor(row["raw_capture_receipt"], f"capture receipt {layout_id}")
        fixture_by_layout[layout_id] = {
            "candidate_id": row["candidate_id"],
            "compact_receipt": compact,
            "environment_seed": row["environment_seed"],
            "gate_receipt_sha256": row["gate_receipt_sha256"],
            "model_fixtures": {"D1": d1_fixture, "N3": n3_fixture},
            "pose_manifest_sha256": row["pose_manifest_sha256"],
            "raw_capture_receipt": capture,
        }
        for category, model_id, name, descriptor in [
            ("fixed_observation_fixture", "D1", "D1_fixture", d1_fixture),
            ("fixed_observation_fixture", "N3", "N3_fixture", n3_fixture),
            ("fixed_observation_capture_receipt", "-", "raw_capture_receipt", capture),
        ]:
            fixture_dependencies.append(
                {
                    "availability": "declared_pointer_only_current_availability_unknown",
                    "category": category,
                    "descriptor": descriptor,
                    "layout_pair_id": layout_id,
                    "model_id": model_id,
                    "name": name,
                }
            )
    require(set(fixture_by_layout) == set(LAYOUTS), "Development fixture layout set changed")
    require(len(fixture_dependencies) == 12, "Fixture dependency count changed")

    require(
        type(parity["parity"]["planned_development_cells"]) is int
        and parity["parity"]["planned_development_cells"] == 32,
        "Planned cell count changed",
    )
    require(
        type(parity["parity"]["executed_terminal_valid_development_cells"])
        is int
        and parity["parity"]["executed_terminal_valid_development_cells"] == 32,
        "Valid cell count changed",
    )
    require(parity["parity"]["planned_equals_executed_exactly"] is True, "Parity no longer exact")
    require(
        type(parity["parity"]["frozen_core_episode_ceiling"]) is int
        and parity["parity"]["frozen_core_episode_ceiling"] == 232,
        "Parity ceiling changed",
    )
    require(parity["parity"]["optional_D2_selected"] is False, "Parity D2 selection changed")
    require_exact_json(pointer["development_population"], {
        "behavioral_actions": 14400,
        "behavioral_requests": 1152,
        "cells_by_model": {"D1": 16, "N3": 16},
        "requests_by_model": REQUESTS_BY_MODEL,
        "requests_per_cell": REQUESTS_PER_CELL,
        "valid_cells": 32,
    }, "Pointer development population")
    require_exact_json(pointer["historical_timing_scope"], EXPECTED_HISTORICAL_TIMING_SCOPE, "Historical timing boundary")
    require(
        pointer["current_state_boundary"]["current_GM_PVC_worker_state_reconciled"] is False
        and pointer["current_state_boundary"]["live_quiescence_inferred"] is False,
        "Current-state boundary changed",
    )

    require(type(parity["cells"]) is list and len(parity["cells"]) == 32, "Parity cell row count changed")
    require(type(pointer["cells"]) is list and len(pointer["cells"]) == 32, "Pointer cell row count changed")
    for row in parity["cells"]:
        require(type(row) is dict and set(row) == PARITY_CELL_KEYS, "Parity cell keys changed")
    parity_cells = {row["cell_id"]: row for row in parity["cells"]}
    pointer_cells = {row["cell_id"]: row for row in pointer["cells"]}
    require(len(parity_cells) == len(pointer_cells) == 32, "Cell population changed")
    require(set(parity_cells) == set(pointer_cells), "Parity/pointer cell sets differ")
    require(len(parity["blocks"]) == 8, "Development block count changed")
    blocks = {(row["model_id"], row["layout_pair_id"]): row for row in parity["blocks"]}
    require(set(blocks) == {(model, layout) for model in MODELS for layout in LAYOUTS}, "Block set changed")
    for layout_id in LAYOUTS:
        require(
            blocks[("D1", layout_id)]["aggregate_accepted_gate_record_sha256"]
            == blocks[("N3", layout_id)]["aggregate_accepted_gate_record_sha256"],
            f"Cross-model accepted gate record differs: {layout_id}",
        )

    cell_recovery = []
    for cell_id in sorted(parity_cells):
        parity_row = parity_cells[cell_id]
        pointer_row = pointer_cells[cell_id]
        require(
            set(pointer_row)
            == {
                "cell_id",
                "layout_pair_id",
                "model_id",
                "raw_cell_receipt_pointer",
                "scheduled_behavioral_request_count",
                "scheduled_condition",
                "scheduled_ordinal",
            },
            f"Pointer-cell keys changed: {cell_id}",
        )
        model_id = parity_row["model_id"]
        layout_id = parity_row["layout_pair_id"]
        require(model_id in MODELS and layout_id in LAYOUTS, f"Unexpected cell: {cell_id}")
        require(
            pointer_row["model_id"] == model_id
            and pointer_row["layout_pair_id"] == layout_id
            and pointer_row["scheduled_condition"] == parity_row["scheduled_condition"]
            and pointer_row["scheduled_ordinal"] == parity_row["scheduled_ordinal"],
            f"Cell schedule join changed: {cell_id}",
        )
        raw_cell = require_descriptor(pointer_row["raw_cell_receipt_pointer"], f"raw cell {cell_id}")
        require(raw_cell == parity_row["raw_cell_receipt_declaration"], f"Cell pointer differs: {cell_id}")
        require(type(pointer_row["scheduled_behavioral_request_count"]) is int and pointer_row["scheduled_behavioral_request_count"] == REQUESTS_PER_CELL[model_id], f"Request count changed: {cell_id}")
        accepted = parity_row["accepted_fixture"]
        frozen = fixture_by_layout[layout_id]
        expected_binding_scope = (
            "exact_compact_receipt_and_D1_fixture_descriptor"
            if model_id == "D1"
            else (
                "accepted_candidate_pose_and_gate_only; fixed-observation N3 descriptor "
                "is inventory context, not an aggregate-declared episode input"
            )
        )
        require(
            accepted["aggregate_binding_scope"] == expected_binding_scope,
            f"Aggregate fixture-binding scope changed: {cell_id}",
        )
        require(
            accepted["aggregate_accepted_gate_record_sha256"]
            == blocks[(model_id, layout_id)]["aggregate_accepted_gate_record_sha256"],
            f"Accepted gate-record binding changed: {cell_id}",
        )
        require(accepted["candidate_id"] == frozen["candidate_id"], f"Candidate join changed: {cell_id}")
        require(accepted["environment_seed"] == frozen["environment_seed"], f"Seed join changed: {cell_id}")
        require(accepted["gate_receipt_sha256"] == frozen["gate_receipt_sha256"], f"Gate join changed: {cell_id}")
        require(accepted["pose_manifest_sha256"] == frozen["pose_manifest_sha256"], f"Pose join changed: {cell_id}")
        require(accepted["compact_receipt"] == frozen["compact_receipt"], f"Compact receipt join changed: {cell_id}")
        require(accepted["model_fixture_declaration"] == frozen["model_fixtures"][model_id], f"Model fixture join changed: {cell_id}")
        aggregate_receipt = require_git_descriptor(
            parity_row["aggregate_receipt"], f"aggregate receipt {cell_id}"
        )
        require(
            aggregate_receipt == blocks[(model_id, layout_id)]["aggregate_receipt"],
            f"Cell/block aggregate receipt differs: {cell_id}",
        )
        cell_recovery.append(
            {
                "aggregate_accepted_gate_record_sha256": accepted[
                    "aggregate_accepted_gate_record_sha256"
                ],
                "aggregate_binding_scope": expected_binding_scope,
                "aggregate_receipt": aggregate_receipt,
                "accepted_model_fixture": frozen["model_fixtures"][model_id],
                "availability": "declared_pointer_only_current_availability_unknown",
                "cell_id": cell_id,
                "layout_recovery_context": frozen,
                "layout_pair_id": layout_id,
                "model_id": model_id,
                "raw_cell_receipt": raw_cell,
                "scheduled_behavioral_request_count": pointer_row["scheduled_behavioral_request_count"],
                "scheduled_condition": pointer_row["scheduled_condition"],
                "scheduled_ordinal": pointer_row["scheduled_ordinal"],
            }
        )
    require(sum(row["scheduled_behavioral_request_count"] for row in cell_recovery) == 1152, "Scheduled request total changed")

    pointer_records = []
    category_counts = {}
    for row in pointer["pointer_inventory"]["pointers"]:
        require(type(row) is dict and "category" in row and "descriptor" in row, "Malformed pointer record")
        category = row["category"]
        require(category in {"cell_receipt", "timing_output", "compiler_output"}, f"Unexpected pointer category: {category}")
        require(set(row) == POINTER_KEYS_BY_CATEGORY[category], f"Pointer keys changed: {category}")
        descriptor = require_descriptor(row["descriptor"], f"pointer {category}")
        clean = dict(row)
        clean["descriptor"] = descriptor
        clean["availability"] = "declared_pointer_only_current_availability_unknown"
        pointer_records.append(clean)
        category_counts[category] = category_counts.get(category, 0) + 1
    require(category_counts == {"cell_receipt": 32, "timing_output": 4, "compiler_output": 8}, "Pointer categories changed")
    require(
        type(pointer["pointer_inventory"]["pointer_count"]) is int
        and len(pointer_records) == pointer["pointer_inventory"]["pointer_count"] == 44,
        "Pointer count changed",
    )
    cell_pointer_records = {
        row["cell_id"]: row for row in pointer_records if row["category"] == "cell_receipt"
    }
    require(len(cell_pointer_records) == 32, "Cell pointer-record population changed")
    require(set(cell_pointer_records) == set(pointer_cells), "Cell pointer-record IDs differ")
    for cell_id, cell_row in pointer_cells.items():
        require(
            cell_pointer_records[cell_id]
            == {
                "availability": "declared_pointer_only_current_availability_unknown",
                "category": "cell_receipt",
                "cell_id": cell_id,
                "descriptor": cell_row["raw_cell_receipt_pointer"],
                "layout_pair_id": cell_row["layout_pair_id"],
                "model_id": cell_row["model_id"],
                "scheduled_condition": cell_row["scheduled_condition"],
                "scheduled_ordinal": cell_row["scheduled_ordinal"],
            },
            f"Cell pointer record differs from cell inventory: {cell_id}",
        )
    require(pointer["pointer_inventory"]["current_raw_availability_verified"] is False, "Pointer availability promoted")
    require(pointer["pointer_inventory"]["raw_PVC_bytes_opened"] is False, "Pointer raw bytes opened")
    require(pointer["pointer_inventory"]["raw_PVC_paths_revalidated"] is False, "Pointer paths revalidated")
    require(pointer["pointer_inventory"]["raw_bytes_rehashed"] is False, "Pointer bytes rehashed")

    all_dependencies = pointer_records + fixture_dependencies
    paths = [row["descriptor"]["path"] for row in all_dependencies]
    shas = [row["descriptor"]["sha256"] for row in all_dependencies]
    require(len(all_dependencies) == 56, "Combined dependency count changed")
    require(len(set(paths)) == 56, "Combined dependency paths are not unique")
    require(len(set(shas)) == 56, "Combined dependency hashes are not unique")
    indexes = {
        "cell_receipts": descriptor_index([row for row in pointer_records if row["category"] == "cell_receipt"], "cell receipt"),
        "timing_outputs": descriptor_index([row for row in pointer_records if row["category"] == "timing_output"], "timing output"),
        "compiler_outputs": descriptor_index([row for row in pointer_records if row["category"] == "compiler_output"], "compiler output"),
        "fixture_inputs": descriptor_index(fixture_dependencies, "fixture input"),
        "combined": descriptor_index(all_dependencies, "combined dependency"),
    }
    cell_lineage_rows = []
    for row in cell_recovery:
        fixture_row = row["layout_recovery_context"]
        raw_cell = row["raw_cell_receipt"]
        model_fixture = row["accepted_model_fixture"]
        compact = fixture_row["compact_receipt"]
        aggregate = row["aggregate_receipt"]
        cell_lineage_rows.append(
            "|".join(
                str(value)
                for value in [
                    row["cell_id"],
                    row["model_id"],
                    row["layout_pair_id"],
                    row["scheduled_ordinal"],
                    row["scheduled_condition"],
                    row["scheduled_behavioral_request_count"],
                    raw_cell["path"],
                    raw_cell["bytes"],
                    raw_cell["sha256"],
                    model_fixture["path"],
                    model_fixture["bytes"],
                    model_fixture["sha256"],
                    fixture_row["candidate_id"],
                    fixture_row["environment_seed"],
                    fixture_row["gate_receipt_sha256"],
                    fixture_row["pose_manifest_sha256"],
                    compact["git_path"],
                    compact["bytes"],
                    compact["sha256"],
                    row["aggregate_binding_scope"],
                    row["aggregate_accepted_gate_record_sha256"],
                    aggregate["git_path"],
                    aggregate["bytes"],
                    aggregate["sha256"],
                ]
            )
        )
    indexes["cell_lineage"] = line_index(cell_lineage_rows, "cell lineage")
    for name, (count, canonical_size, digest) in EXPECTED_RECOVERY_INDEXES.items():
        require(
            indexes[name]["count"] == count
            and indexes[name]["canonical_bytes"] == canonical_size
            and indexes[name]["sha256"] == digest,
            f"Recovery index changed: {name}",
        )
    combined_declared_bytes = sum(row["descriptor"]["bytes"] for row in all_dependencies)
    require(
        combined_declared_bytes == EXPECTED_COMBINED_DECLARED_BYTES,
        "Combined declared byte count changed",
    )

    identity_rows = identity["request_identity_spine"]["identities"]
    require(type(identity_rows) is list and len(identity_rows) == 1152, "Identity row count changed")
    identity_sets = {model: set() for model in MODELS}
    for row in identity_rows:
        require(set(row) == {"model_id", "source_request_receipt_sha256"}, "Identity row acquired join fields")
        model_id = row["model_id"]
        require(model_id in MODELS, "Unexpected identity model")
        identity_sets[model_id].add(require_sha256(row["source_request_receipt_sha256"], "identity"))
    require({model: len(rows) for model, rows in identity_sets.items()} == REQUESTS_BY_MODEL, "Identity model counts changed")
    require(not identity_sets["D1"] & identity_sets["N3"], "Cross-model identity overlap")
    actual_identity_indexes = {
        "D1": {key: value for key, value in identity["request_identity_spine"]["per_model_indexes"]["D1"].items() if key != "canonicalization"},
        "N3": {key: value for key, value in identity["request_identity_spine"]["per_model_indexes"]["N3"].items() if key != "canonicalization"},
        "combined": {key: value for key, value in identity["request_identity_spine"]["combined_index"].items() if key != "canonicalization"},
    }
    require_exact_json(actual_identity_indexes, EXPECTED_IDENTITY_INDEXES, "Identity indexes")
    require_exact_json(
        pointer["request_identity_population"],
        EXPECTED_REQUEST_IDENTITY_POPULATION,
        "Request identity population boundary",
    )
    require_exact_json(
        pointer["annotation_packaging"],
        EXPECTED_ANNOTATION_PACKAGING,
        "Annotation packaging boundary",
    )

    return {
        "schema_version": "wmf-development-annotation-raw-retrieval-preflight-v1",
        "study_id": "WMF-ABLATION-001",
        "pins": {
            "design_execution_parity_commit": PARITY_COMMIT,
            "fixed_observation_inventory_commit": FIXTURE_COMMIT,
            "input_pointer_inventory_commit": POINTER_COMMIT,
            "readiness_commit": READINESS_COMMIT,
            "request_identity_commit": IDENTITY_COMMIT,
            "spec_commit": SPEC_COMMIT,
        },
        "authenticated_source_blobs": {
            **source_index,
            "sources": sorted(sources, key=lambda row: (row["commit"], row["path"])),
        },
        "protocol": {
            "core_episode_ceiling": 232,
            "development_cells": 32,
            "confirmation_cells": 192,
            "optional_D2_selected": False,
            "request_sample_cap_per_episode": 4,
            "request_sampling_seed": 2026091302,
            "request_manifest_timing": spec["analysis"]["request_manifest_timing"],
            "independent_raters_required": 2,
            "counts_authorize_execution": False,
        },
        "recovery_scope": {
            "development_layouts": list(LAYOUTS),
            "development_cells": 32,
            "cells_by_model": {"D1": 16, "N3": 16},
            "scheduled_requests": 1152,
            "scheduled_requests_by_model": REQUESTS_BY_MODEL,
            "declared_pointer_dependencies": 44,
            "declared_fixture_dependencies": 12,
            "combined_unique_dependencies": 56,
            "combined_declared_bytes": combined_declared_bytes,
            "indexes": indexes,
            "availability": "declared_pointer_only_current_availability_unknown",
        },
        "cells": cell_recovery,
        "fixture_dependencies": sorted(
            fixture_dependencies,
            key=lambda row: (row["layout_pair_id"], row["model_id"], row["name"]),
        ),
        "pointer_dependencies": sorted(
            pointer_records,
            key=lambda row: (
                row["category"],
                row.get("model_id", ""),
                row.get("layout_pair_id", ""),
                row.get("cell_id", ""),
                row.get("name", ""),
            ),
        ),
        "unjoined_request_identity_population": {
            "rows": 1152,
            "rows_by_model": REQUESTS_BY_MODEL,
            "indexes": actual_identity_indexes,
            "identity_rows_copied_into_this_report": 0,
            "joined_to_cells": False,
            "request_to_cell_mapping_materialized": False,
            "request_to_cell_mapping_authenticated": False,
            "boundary": (
                "Request identities remain a separate authenticated population. Raw request "
                "inventories must be retrieved and rehashed before any request-to-cell join."
            ),
        },
        "historical_timing_scope": pointer["historical_timing_scope"],
        "availability_and_gate_boundary": {
            "raw_PVC_paths_revalidated": False,
            "raw_PVC_bytes_opened": False,
            "raw_bytes_rehashed": False,
            "current_raw_availability_verified": False,
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "camera_crop_contract_authenticated": False,
            "physical_alignment_authenticated": False,
            "annotation_media_renderable": False,
            "production_eligibility_determined": False,
            "production_request_selection_manifest_created": False,
            "human_handoff_ready": False,
            "labels": 0,
            "human_responses": 0,
            "forecast_accuracy_estimates": 0,
            "confirmation_inference_cells": 0,
            "safe_to_release_confirmation": False,
            "remaining_required_work": [
                "authorized current GM/PVC/worker/GPU/release-lock reconciliation",
                "read-only retrieval and byte/SHA verification of declared dependencies",
                "fail-closed request-inventory to cell join",
                "signed camera crop and physical-time alignment",
                "real blinded annotation media packaging",
                "approved two-rater annotation workflow and adjudication evidence",
                "measured resource freeze and confirmation release authorization",
            ],
        },
        "gate_state": GATE_STATE,
        "global_study_snapshot": GLOBAL_SNAPSHOT,
        "new_activity": {
            "annotation_media": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "forecast_accuracy_estimates": 0,
            "human_responses": 0,
            "labels": 0,
            "model_requests": 0,
            "request_selections": 0,
        },
        "claim_boundaries": {
            "authenticated": (
                "Exact compact Git report bytes, the 32-cell frozen lineage, 44 raw "
                "pointer declarations, 12 fixed-observation input declarations, and the "
                "separate 1,152-identity population indexes."
            ),
            "raw_PVC": (
                "No declared raw path was opened, revalidated, or rehashed. Current "
                "availability remains unknown."
            ),
            "mapping": (
                "No request identity is joined to a cell and no eligibility or selection "
                "decision is materialized."
            ),
            "timing": (
                "D1 coverage remains aggregate historical evidence with unmapped records; "
                "N3 completeness is not inferred from physical-time qualification."
            ),
            "annotation": (
                "No media, rubric approval, rater handoff, label, response, consensus, "
                "adjudication, movement threshold, or usability decision is claimed."
            ),
            "science": (
                "This is recovery-lineage evidence only: no policy result, forecast "
                "estimate, model comparison, or confirmation inference is claimed."
            ),
            "operations": (
                "Historical pointers do not establish current GM/PVC/queue/worker/GPU/"
                "child/reap/release-lock state or live quiescence."
            ),
            "release": (
                "This audit released no new queue descriptor, valid episode, or confirmation cell."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical report")
    parser.add_argument("--output", type=Path, help="Write the canonical report")
    args = parser.parse_args()
    require(not (args.check and args.output), "Use only one of --check and --output")
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved raw-retrieval preflight differs")
        print("Verified 32 development cells, 56 declared raw dependencies, and 1,152 unjoined request identities")
    elif args.output:
        args.output.write_bytes(encoded)
        print(f"Wrote {args.output} ({len(encoded)} bytes)")
    else:
        sys.stdout.buffer.write(encoded)


if __name__ == "__main__":
    main()
