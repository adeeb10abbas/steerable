#!/usr/bin/env python3
"""Audit development annotation-input pointers without claiming raw availability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
READINESS_COMMIT = "84332bf638595ff7ed955fd263586ecc0e06ab93"
IDENTITY_COMMIT = "33a7483b4cc8255fd97fadb1684b3a4082ee3c9f"
PARITY_COMMIT = "9c1a17acda5f6565cf116c26b941e47bb4c681d6"
PVC_ROOT = "/data/users/ali/vla_wam/raw/wmf_ablation_001_20260912"

SPEC_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/ablation_spec.json"
)
READINESS_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_readiness_inventory.json"
)
IDENTITY_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_request_identity_spine.json"
)
PARITY_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_design_execution_parity.json"
)
TIMING_PATHS = {
    "D1": (
        "results/jobs/timing-d1-development-sidecar-002/publish/"
        "timing_job_receipt.json"
    ),
    "N3": (
        "results/jobs/timing-n3-development-sidecar-001/publish/"
        "timing_job_receipt.json"
    ),
}
COMPILER_PATH = (
    "results/jobs/development-evidence-compiler-formal-004/publish/"
    "development_evidence_compiler_job_receipt.json"
)

EXPECTED_STATIC = {
    (SPEC_COMMIT, SPEC_PATH): (
        8590,
        "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
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
    (RESULTS_COMMIT, TIMING_PATHS["D1"]): (
        79824,
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
    ),
    (RESULTS_COMMIT, TIMING_PATHS["N3"]): (
        29935,
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
    ),
    (RESULTS_COMMIT, COMPILER_PATH): (
        12374,
        "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434",
    ),
}

AGGREGATES = {
    ("D1", "D01"): (
        "results/jobs/d1-development-d01-simulator-003/publish/"
        "d1_behavioral_development_receipt.json",
        16510,
        "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb",
    ),
    ("D1", "D02"): (
        "results/jobs/d1-development-d02-simulator-003/publish/"
        "d1_behavioral_development_receipt.json",
        16510,
        "d1dc66c57d36cb3e620731c69a6e41fdf52c3cc2a797c2dae7ee36d63d10c6c9",
    ),
    ("D1", "D03"): (
        "results/jobs/d1-development-d03-simulator-002/publish/"
        "d1_behavioral_development_receipt.json",
        16510,
        "eb7f4f894b1e56c1b8971ee1155d77e7aeb1b62c9c6cfc41de0483e3baf2dd4f",
    ),
    ("D1", "D04"): (
        "results/jobs/d1-development-d04-simulator-003/publish/"
        "d1_behavioral_development_receipt.json",
        16510,
        "78a9b8b7095e48ce5412a902a7120f3c9ad71e3edb6eeeddb228720fe1b458fa",
    ),
    ("N3", "D01"): (
        "results/jobs/n3-development-d01-002/publish/"
        "n3_behavioral_development_receipt.json",
        12551,
        "a9ca815c6e99b270357220fa5ccb97562c50d667dd3c23745d460e5ec8c053de",
    ),
    ("N3", "D02"): (
        "results/jobs/n3-development-d02-002/publish/"
        "n3_behavioral_development_receipt.json",
        12551,
        "efd2ea9f2f015c671a9d9d2d9dc22c1c1f9f5f3ffd87e4810acd5e3767b2f73e",
    ),
    ("N3", "D03"): (
        "results/jobs/n3-development-d03-001/publish/"
        "n3_behavioral_development_receipt.json",
        12551,
        "d3e85f2087d7e24d57974a9344c333408e9769552715d4669e16f1e059082d0e",
    ),
    ("N3", "D04"): (
        "results/jobs/n3-development-d04-001/publish/"
        "n3_behavioral_development_receipt.json",
        12551,
        "e5f3206c6da2b9caa2ca9cd4c3ed82142157656ac9a1d1744486d14fa6e57318",
    ),
}

REQUESTS_PER_CELL = {"D1": 57, "N3": 15}
REQUESTS_BY_MODEL = {"D1": 912, "N3": 240}
LAYOUTS = ("D01", "D02", "D03", "D04")
EXPECTED_IDENTITY_INDEXES = {
    "D1": (912, 62016, "0ca707976b59e0957b4bb636982743bb4e5458f5bd08f8177eda09dcec0f0110"),
    "N3": (240, 16320, "d25a670fbc024960aeff5d667b4ccf9c6b90b2f3c24691348bc1db3bd0367051"),
    "combined": (
        1152,
        78336,
        "3a6206f1e5fc790f1b8c92d9d023c1ce71f969ec95c19ea4482073a96cb86a71",
    ),
}
EXPECTED_POINTER_INDEXES = {
    "cell_receipts": (
        32,
        10128,
        "7e871644a116001cae321f4cf15b0eda210668238bf6418ca49e108b92b1ff60",
    ),
    "timing_outputs": (
        4,
        1003,
        "bf5977dc3142f8bc5cbe729e5c0021c02ee73a3c5faaf0d7b3f27e6477fb0d8f",
    ),
    "compiler_outputs": (
        8,
        2370,
        "885e13cb4152952cb3347efb323b7010c28c5145c09e4ad18e1b67a656da7876",
    ),
    "combined": (
        44,
        13501,
        "99b869e4189ffa3d2a2041c433c14d798527c581af227108a64a97c49817e6ea",
    ),
}
EXPECTED_POINTER_BYTES = {
    "cell_receipts": 1711371,
    "timing_outputs": 10392930,
    "compiler_outputs": 14007403,
    "combined": 26111704,
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
D1_TIMING_COVERAGE = {
    "action_prefix_truncated_request_count": 16,
    "action_prefix_truncated_target_binding_count": 32,
    "native_clock_matched_request_count": 224,
    "native_clock_matched_target_binding_count": 448,
    "source_proven_full_decode_request_count": 240,
    "source_unmapped_incremental_decode_request_count": 672,
    "total_request_count": 912,
    "unmapped_potential_authority_target_count": 1344,
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
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def source_blob(commit: str, path: str, expected: tuple[int, str]) -> tuple[bytes, dict]:
    payload = git_blob(commit, path)
    descriptor = {
        "commit": commit,
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    require(
        (descriptor["bytes"], descriptor["sha256"]) == expected,
        f"Pinned blob changed: {commit}:{path}",
    )
    return payload, descriptor


def require_sha256(value: object, label: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


def require_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Missing descriptor: {label}")
    require(set(value) == {"bytes", "path", "sha256"}, f"Descriptor changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Bad bytes: {label}")
    require(
        type(value["path"]) is str and value["path"].startswith("/data/users/ali/"),
        f"Unexpected PVC path: {label}",
    )
    require_sha256(value["sha256"], label)
    return dict(value)


def require_zero_counts(value: object, label: str) -> None:
    require(type(value) is dict and value, f"Missing zero counts: {label}")
    require(
        all(type(item) is int and item == 0 for item in value.values()),
        f"Nonzero activity: {label}",
    )


def index(rows: list[str], expected: tuple[int, int, str], label: str) -> dict:
    payload = "".join(rows).encode("utf-8")
    actual = (len(rows), len(payload), hashlib.sha256(payload).hexdigest())
    require(actual == expected, f"{label} index changed: {actual}")
    return {
        "count": actual[0],
        "canonical_bytes": actual[1],
        "sha256": actual[2],
    }


def audit() -> dict:
    sources = []

    def load_json_source(commit: str, path: str, expected: tuple[int, str]) -> dict:
        payload, descriptor = source_blob(commit, path, expected)
        sources.append(descriptor)
        return load_json(payload, f"{commit}:{path}")

    spec = load_json_source(SPEC_COMMIT, SPEC_PATH, EXPECTED_STATIC[(SPEC_COMMIT, SPEC_PATH)])
    readiness = load_json_source(
        READINESS_COMMIT,
        READINESS_PATH,
        EXPECTED_STATIC[(READINESS_COMMIT, READINESS_PATH)],
    )
    identity = load_json_source(
        IDENTITY_COMMIT,
        IDENTITY_PATH,
        EXPECTED_STATIC[(IDENTITY_COMMIT, IDENTITY_PATH)],
    )
    parity = load_json_source(
        PARITY_COMMIT,
        PARITY_PATH,
        EXPECTED_STATIC[(PARITY_COMMIT, PARITY_PATH)],
    )
    timing = {
        model_id: load_json_source(
            RESULTS_COMMIT,
            path,
            EXPECTED_STATIC[(RESULTS_COMMIT, path)],
        )
        for model_id, path in sorted(TIMING_PATHS.items())
    }
    compiler = load_json_source(
        RESULTS_COMMIT,
        COMPILER_PATH,
        EXPECTED_STATIC[(RESULTS_COMMIT, COMPILER_PATH)],
    )

    aggregates = {}
    for key, (path, expected_bytes, expected_sha256) in sorted(AGGREGATES.items()):
        aggregates[key] = load_json_source(
            RESULTS_COMMIT,
            path,
            (expected_bytes, expected_sha256),
        )

    require(spec.get("spec_id") == "WMF-ABLATION-001", "Study changed")
    require(spec.get("core_episode_ceiling") == 232, "Core ceiling changed")
    require(spec["stages"]["development"]["episodes"] == 32, "Development changed")
    require(spec["stages"]["confirmation"]["episodes"] == 192, "Confirmation changed")
    require(spec["analysis"]["request_sample_cap_per_episode"] == 4, "Cap changed")
    require(spec["analysis"]["request_sampling_seed"] == 2026091302, "Seed changed")
    require(spec["annotation"]["raters"] == 2, "Rater count changed")
    require(spec["models"]["D2"]["selected"] is False, "D2 became selected")

    require(readiness.get("study_id") == "WMF-ABLATION-001", "Readiness study changed")
    require(readiness["global_study_snapshot"] == GLOBAL_SNAPSHOT, "Snapshot changed")
    require(readiness["readiness"] == GATE_STATE, "Readiness gates changed")
    require_zero_counts(readiness["new_activity"], "readiness")
    require(
        readiness["development_evidence"]["valid_cells"] == 32
        and readiness["development_evidence"]["behavioral_requests"] == 1152
        and readiness["development_evidence"]["behavioral_actions"] == 14400,
        "Development population changed",
    )

    require(identity.get("study_id") == "WMF-ABLATION-001", "Identity study changed")
    require(identity["global_study_snapshot"] == GLOBAL_SNAPSHOT, "Identity snapshot changed")
    require(identity["gate_state"] == GATE_STATE, "Identity gates changed")
    require_zero_counts(identity["new_activity"], "identity")
    require(
        identity["prospective_annotation_packaging"]
        ["production_eligibility_determined"]
        is False
        and identity["prospective_annotation_packaging"]
        ["production_request_selection_manifest_created"]
        is False
        and identity["prospective_annotation_packaging"]["selected_request_count"] == 0,
        "Identity report promoted selection",
    )

    require(parity.get("study_id") == "WMF-ABLATION-001", "Parity study changed")
    require(parity["global_study_snapshot"] == GLOBAL_SNAPSHOT, "Parity snapshot changed")
    require(parity["gate_state"] == GATE_STATE, "Parity gates changed")
    require_zero_counts(parity["new_activity"], "parity")
    require(
        parity["parity"]["planned_equals_executed_exactly"] is True
        and parity["parity"]["planned_development_cells"] == 32
        and parity["parity"]["executed_terminal_valid_development_cells"] == 32,
        "Development parity changed",
    )
    require(
        parity["unbound_request_population"]["request_to_cell_mapping_authenticated"]
        is False,
        "Prior boundary unexpectedly changed",
    )

    identity_rows = identity["request_identity_spine"]["identities"]
    require(type(identity_rows) is list and len(identity_rows) == 1152, "Identity rows changed")
    identities = {"D1": set(), "N3": set()}
    for row in identity_rows:
        require(type(row) is dict and set(row) == {
            "model_id",
            "source_request_receipt_sha256",
        }, "Identity row changed")
        model_id = row["model_id"]
        require(model_id in identities, "Unexpected identity model")
        identities[model_id].add(
            require_sha256(row["source_request_receipt_sha256"], "identity row")
        )
    require(
        {model_id: len(values) for model_id, values in identities.items()}
        == REQUESTS_BY_MODEL,
        "Identity population changed",
    )
    require(not identities["D1"] & identities["N3"], "Cross-model identity overlap")
    identity_index_rows = {
        model_id: [
            f"{model_id}\t{request_sha256}\n"
            for request_sha256 in sorted(identities[model_id])
        ]
        for model_id in ("D1", "N3")
    }

    parity_cells = {row["cell_id"]: row for row in parity["cells"]}
    require(len(parity_cells) == 32, "Parity cell inventory changed")
    cell_rows = []
    cell_pointer_rows = []
    timing_pointer_rows = []
    compiler_pointer_rows = []
    all_pointer_descriptors = []
    declared_pointer_records = []
    timing_scope = {}

    for model_id in ("D1", "N3"):
        receipt = timing[model_id]
        require(
            receipt.get("schema_version") == "wmf-development-timing-sidecar-queue-job-v1"
            and receipt.get("study_id") == "WMF-ABLATION-001"
            and receipt.get("model_id") == model_id
            and receipt.get("status") == "passed"
            and receipt.get("decision") == "go",
            f"{model_id} timing status changed",
        )
        require(
            receipt.get("behavioral_policy_skill_evaluated") is False,
            f"{model_id} timing policy boundary changed",
        )
        require(
            receipt.get("referenced_behavioral_cells") == 16
            and receipt.get("referenced_behavioral_model_requests")
            == REQUESTS_BY_MODEL[model_id]
            and receipt.get("development_timing_sidecar_valid") is True
            and receipt.get("safe_to_release_confirmation") is False,
            f"{model_id} timing scope changed",
        )
        require_zero_counts(receipt["science_counts"], f"{model_id} timing")
        if model_id == "D1":
            require(
                receipt.get("physical_time_qualified") is True
                and receipt.get("physical_time_coverage_complete") is False
                and receipt.get("physical_time_qualified_scope")
                == "source-proven full conditioning-origin decodes only"
                and receipt.get("request_timing_coverage") == D1_TIMING_COVERAGE,
                "D1 timing coverage changed",
            )
            timing_scope[model_id] = {
                "historical_receipt_requests": REQUESTS_BY_MODEL[model_id],
                "physical_time_qualified": True,
                "physical_time_coverage_complete": False,
                "physical_time_qualified_scope": (
                    "source-proven full conditioning-origin decodes only"
                ),
                "request_timing_coverage": D1_TIMING_COVERAGE,
            }
        else:
            require(
                receipt.get("physical_time_qualified") is True
                and "physical_time_coverage_complete" not in receipt
                and "request_timing_coverage" not in receipt,
                "N3 timing scope changed",
            )
            timing_scope[model_id] = {
                "historical_receipt_requests": REQUESTS_BY_MODEL[model_id],
                "physical_time_qualified": True,
                "physical_time_coverage_complete_field_present": False,
                "request_timing_coverage_breakdown_present": False,
            }
        outputs = receipt["outputs"]
        require(
            set(outputs) == {"raw_request_inventory", "raw_development_timing_sidecar"},
            f"{model_id} timing outputs changed",
        )
        for name, raw in sorted(outputs.items()):
            descriptor = require_descriptor(raw, f"{model_id} {name}")
            timing_pointer_rows.append(
                "\t".join(
                    [
                        "timing",
                        model_id,
                        name,
                        descriptor["path"],
                        str(descriptor["bytes"]),
                        descriptor["sha256"],
                    ]
                )
                + "\n"
            )
            all_pointer_descriptors.append(descriptor)
            declared_pointer_records.append(
                {
                    "category": "timing_output",
                    "model_id": model_id,
                    "name": name,
                    "descriptor": descriptor,
                }
            )

        request_hashes = receipt["source_request_receipt_sha256s"]
        require(
            type(request_hashes) is list
            and len(request_hashes) == REQUESTS_BY_MODEL[model_id]
            and len(set(request_hashes)) == len(request_hashes),
            f"{model_id} request hash population changed",
        )
        for value in request_hashes:
            require_sha256(value, f"{model_id} request hash")
        require(set(request_hashes) == identities[model_id], f"{model_id} identity set changed")

        for layout in LAYOUTS:
            aggregate = aggregates[(model_id, layout)]
            require(
                aggregate.get("status") == "passed"
                and aggregate.get("study_id") == "WMF-ABLATION-001"
                and aggregate.get("model_config") == model_id
                and aggregate.get("layout_pair_id") == layout
                and aggregate.get("phase") == "development",
                f"{model_id} {layout} aggregate changed",
            )
            counts = aggregate["counts"]
            require(
                counts["completed_valid_behavioral_cells"] == 4
                and counts["actual_behavioral_actions"] == 1800
                and counts["actual_behavioral_model_requests"]
                == 4 * REQUESTS_PER_CELL[model_id]
                and counts["technically_invalid_behavioral_cells"] == 0
                and counts["right_censored_behavioral_cells"] == 0
                and counts["unrun_behavioral_cells"] == 0,
                f"{model_id} {layout} aggregate counts changed",
            )
            cell_ids = aggregate["cell_ids"]
            cell_receipts = aggregate["cell_receipts"]
            require(
                type(cell_ids) is list
                and type(cell_receipts) is list
                and len(cell_ids) == len(cell_receipts) == 4,
                f"{model_id} {layout} cell inventory changed",
            )
            require(
                aggregate["resumed_cell_receipts"] + aggregate["new_cell_receipts"]
                == cell_receipts,
                f"{model_id} {layout} resume/new inventory changed",
            )
            timing_input = receipt["inputs"]["development_aggregates"][layout]
            aggregate_path, expected_bytes, expected_sha256 = AGGREGATES[(model_id, layout)]
            aggregate_job = Path(aggregate_path).parts[2]
            expected_aggregate_pointer = {
                "path": (
                    f"{PVC_ROOT}/control/jobs/{aggregate_job}/publish/"
                    f"{Path(aggregate_path).name}"
                ),
                "sha256": expected_sha256,
                "bytes": expected_bytes,
            }
            require(
                require_descriptor(
                    timing_input["aggregate_receipt"],
                    f"{model_id} {layout} timing aggregate",
                )
                == expected_aggregate_pointer,
                f"{model_id} {layout} timing aggregate pointer changed",
            )
            require(
                timing_input["cell_receipts"] == cell_receipts,
                f"{model_id} {layout} timing cell pointers changed",
            )

            for ordinal, (cell_id, raw) in enumerate(zip(cell_ids, cell_receipts)):
                descriptor = require_descriptor(
                    raw, f"{model_id} {layout} cell {ordinal}"
                )
                require(cell_id in parity_cells, f"Unknown cell: {cell_id}")
                parity_cell = parity_cells[cell_id]
                expected_condition = (
                    f"{parity_cell['layout_arm']}-{parity_cell['command']}"
                )
                require(
                    parity_cell["model_id"] == model_id
                    and parity_cell["layout_pair_id"] == layout
                    and parity_cell["scheduled_ordinal"] == ordinal
                    and parity_cell["phase"] == "development"
                    and parity_cell["scheduled_condition"] == expected_condition
                    and cell_id.endswith(
                        f"__{parity_cell['layout_arm']}__{parity_cell['command']}"
                    ),
                    f"Parity cell binding changed: {cell_id}",
                )
                require(
                    parity_cell["raw_cell_receipt_declaration"] == descriptor,
                    f"Parity pointer changed: {cell_id}",
                )
                cell_pointer_rows.append(
                    "\t".join(
                        [
                            "cell",
                            model_id,
                            layout,
                            str(ordinal),
                            cell_id,
                            descriptor["path"],
                            str(descriptor["bytes"]),
                            descriptor["sha256"],
                        ]
                    )
                    + "\n"
                )
                all_pointer_descriptors.append(descriptor)
                declared_pointer_records.append(
                    {
                        "category": "cell_receipt",
                        "model_id": model_id,
                        "layout_pair_id": layout,
                        "cell_id": cell_id,
                        "scheduled_ordinal": ordinal,
                        "scheduled_condition": parity_cell["scheduled_condition"],
                        "descriptor": descriptor,
                    }
                )
                cell_rows.append(
                    {
                        "model_id": model_id,
                        "layout_pair_id": layout,
                        "scheduled_ordinal": ordinal,
                        "scheduled_condition": parity_cell["scheduled_condition"],
                        "cell_id": cell_id,
                        "raw_cell_receipt_pointer": descriptor,
                        "scheduled_behavioral_request_count": REQUESTS_PER_CELL[model_id],
                    }
                )

    require(
        compiler.get("schema_version")
        == "wmf-development-evidence-compiler-queue-job-v1"
        and compiler.get("study_id") == "WMF-ABLATION-001"
        and compiler.get("status") == "passed"
        and compiler.get("decision") == "go"
        and compiler.get("formal_cohort_complete") is True
        and compiler.get("confirmation_released") is False,
        "Compiler status changed",
    )
    require(
        compiler.get("safe_to_release_confirmation") is False,
        "Compiler release boundary changed",
    )
    require(
        compiler.get("safe_for_timing_binding") is True
        and compiler.get("behavioral_policy_skill_evaluated") is False,
        "Compiler evidence boundary changed",
    )
    require(
        compiler["counts"]
        == {
            "compiled_cells": 32,
            "compiled_source_behavioral_actions": 14400,
            "compiled_source_behavioral_requests": 1152,
        },
        "Compiler counts changed",
    )
    require_zero_counts(compiler["compiler_science_activity"], "compiler")
    require_zero_counts(compiler["science_counts"], "compiler science counts")
    compiler_outputs = compiler["outputs"]
    require(
        compiler_outputs["semantically_validated_nonreceipt_file_count"] == 70,
        "Compiler validation count changed",
    )
    compiler_descriptors = {
        name: value
        for name, value in compiler_outputs.items()
        if name not in {"primary_compiled_outputs", "semantically_validated_nonreceipt_file_count"}
    }
    primary = compiler_outputs["primary_compiled_outputs"]
    require(
        set(primary)
        == {
            "d1_development_freeze_cells.json",
            "d1_development_request_provenance.json",
            "d1_development_timing_request_inventory.json",
            "n3_development_freeze_cells.json",
            "n3_development_request_provenance.json",
            "n3_development_timing_request_inventory.json",
        },
        "Primary compiler output inventory changed",
    )
    for name, raw in sorted(compiler_descriptors.items()):
        descriptor = require_descriptor(raw, f"compiler {name}")
        compiler_pointer_rows.append(
            "\t".join(
                [
                    "compiler",
                    name,
                    descriptor["path"],
                    str(descriptor["bytes"]),
                    descriptor["sha256"],
                ]
            )
            + "\n"
        )
        all_pointer_descriptors.append(descriptor)
        declared_pointer_records.append(
            {
                "category": "compiler_output",
                "name": name,
                "descriptor": descriptor,
            }
        )
    for name, raw in sorted(primary.items()):
        descriptor = require_descriptor(raw, f"compiler primary {name}")
        compiler_pointer_rows.append(
            "\t".join(
                [
                    "compiler",
                    f"primary_compiled_outputs/{name}",
                    descriptor["path"],
                    str(descriptor["bytes"]),
                    descriptor["sha256"],
                ]
            )
            + "\n"
        )
        all_pointer_descriptors.append(descriptor)
        declared_pointer_records.append(
            {
                "category": "compiler_output",
                "name": f"primary_compiled_outputs/{name}",
                "descriptor": descriptor,
            }
        )

    require(len(cell_rows) == 32, "Cell pointer population changed")
    require(
        len({row["cell_id"] for row in cell_rows}) == 32,
        "Duplicate cell declaration",
    )
    require(
        len({row["path"] for row in all_pointer_descriptors}) == 44
        and len({row["sha256"] for row in all_pointer_descriptors}) == 44,
        "Pointer paths or hashes overlap",
    )
    require(len(declared_pointer_records) == 44, "Emitted pointer population changed")

    identity_indexes = {
        model_id: index(
            identity_index_rows[model_id],
            EXPECTED_IDENTITY_INDEXES[model_id],
            f"{model_id} unjoined identity population",
        )
        for model_id in ("D1", "N3")
    }
    identity_indexes["combined"] = index(
        identity_index_rows["D1"] + identity_index_rows["N3"],
        EXPECTED_IDENTITY_INDEXES["combined"],
        "combined unjoined identity population",
    )
    pointer_indexes = {
        "cell_receipts": index(
            cell_pointer_rows,
            EXPECTED_POINTER_INDEXES["cell_receipts"],
            "cell pointer",
        ),
        "timing_outputs": index(
            timing_pointer_rows,
            EXPECTED_POINTER_INDEXES["timing_outputs"],
            "timing output pointer",
        ),
        "compiler_outputs": index(
            compiler_pointer_rows,
            EXPECTED_POINTER_INDEXES["compiler_outputs"],
            "compiler output pointer",
        ),
    }
    pointer_indexes["combined"] = index(
        sorted(cell_pointer_rows + timing_pointer_rows + compiler_pointer_rows),
        EXPECTED_POINTER_INDEXES["combined"],
        "combined pointer",
    )
    actual_pointer_bytes = {
        "cell_receipts": sum(row["raw_cell_receipt_pointer"]["bytes"] for row in cell_rows),
        "timing_outputs": sum(
            require_descriptor(raw, f"timing output {model_id} {name}")["bytes"]
            for model_id, receipt in timing.items()
            for name, raw in receipt["outputs"].items()
        ),
        "compiler_outputs": sum(
            require_descriptor(raw, f"compiler output {name}")["bytes"]
            for name, raw in {**compiler_descriptors, **primary}.items()
        ),
    }
    actual_pointer_bytes["combined"] = sum(actual_pointer_bytes.values())
    require(actual_pointer_bytes == EXPECTED_POINTER_BYTES, "Pointer byte totals changed")

    source_rows = [
        "\t".join(
            [
                row["commit"],
                row["path"],
                str(row["bytes"]),
                row["sha256"],
            ]
        )
        + "\n"
        for row in sorted(sources, key=lambda item: (item["commit"], item["path"]))
    ]
    source_payload = "".join(source_rows).encode("utf-8")

    return {
        "schema_version": "wmf-development-annotation-input-pointer-inventory-v1",
        "study_id": "WMF-ABLATION-001",
        "pins": {
            "spec_commit": SPEC_COMMIT,
            "readiness_commit": READINESS_COMMIT,
            "request_identity_commit": IDENTITY_COMMIT,
            "design_execution_parity_commit": PARITY_COMMIT,
            "results_commit": RESULTS_COMMIT,
        },
        "authenticated_source_blobs": {
            "count": len(sources),
            "bytes": sum(row["bytes"] for row in sources),
            "index_canonical_bytes": len(source_payload),
            "index_sha256": hashlib.sha256(source_payload).hexdigest(),
            "sources": sorted(sources, key=lambda item: (item["commit"], item["path"])),
        },
        "development_population": {
            "valid_cells": 32,
            "behavioral_requests": 1152,
            "behavioral_actions": 14400,
            "cells_by_model": {"D1": 16, "N3": 16},
            "requests_by_model": REQUESTS_BY_MODEL,
            "requests_per_cell": REQUESTS_PER_CELL,
        },
        "historical_timing_scope": {
            "by_model": timing_scope,
            "boundary": (
                "Pinned receipt metadata only; no raw timing output was opened and no "
                "current raw availability or broader timing coverage is inferred."
            ),
        },
        "request_identity_population": {
            "rows": 1152,
            "rows_by_model": REQUESTS_BY_MODEL,
            "indexes": identity_indexes,
            "timing_receipt_identity_sets_match_committed_spine": True,
            "joined_to_cell_pointers": False,
            "request_to_cell_mapping_materialized": False,
            "request_to_cell_mapping_authenticated": False,
            "independent_raw_inventory_replay_performed": False,
            "boundary": (
                "The committed request identities and historical cell pointers are "
                "authenticated as separate populations. No request identity is joined "
                "to a cell; that mapping requires authorized raw inventory retrieval and "
                "replay."
            ),
        },
        "pointer_inventory": {
            "pointer_count": 44,
            "declared_bytes": actual_pointer_bytes,
            "unique_paths": 44,
            "unique_sha256s": 44,
            "indexes": pointer_indexes,
            "cell_to_receipt_pointer_authenticated": True,
            "current_raw_availability_verified": False,
            "raw_PVC_bytes_opened": False,
            "raw_PVC_paths_revalidated": False,
            "raw_bytes_rehashed": False,
            "categories": {
                "cell_receipts": 32,
                "timing_outputs": 4,
                "compiler_outputs": 8,
            },
            "pointers": sorted(
                declared_pointer_records,
                key=lambda row: row["descriptor"]["path"],
            ),
        },
        "cells": sorted(cell_rows, key=lambda row: row["cell_id"]),
        "annotation_packaging": {
            "request_sampling_seed": 2026091302,
            "request_sample_cap_per_episode": 4,
            "development_selected_request_ceiling": 128,
            "production_eligibility_determined": False,
            "production_request_selection_manifest_created": False,
            "selected_request_count": 0,
            "annotation_media_created": False,
            "human_handoff_ready": False,
            "human_responses": 0,
            "human_labels": 0,
        },
        "global_study_snapshot": GLOBAL_SNAPSHOT,
        "gate_state": GATE_STATE,
        "current_state_boundary": {
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "boundary": (
                "Historical published pointers do not establish current GM/PVC/queue/"
                "worker/GPU/child/reap/release-lock state or raw artifact availability."
            ),
        },
        "new_activity": {
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "request_selections": 0,
            "annotation_media": 0,
            "labels": 0,
            "human_responses": 0,
            "forecast_accuracy_estimates": 0,
        },
        "claim_boundaries": {
            "authenticated": (
                "Exact published aggregate cell-receipt pointers, timing output pointers, "
                "compiler output pointers, and unjoined request identity populations only."
            ),
            "raw_PVC": (
                "No raw cell receipt, request inventory, timing sidecar, compiler bundle, "
                "fixture payload, media file or PVC byte was opened or rehashed."
            ),
            "mapping": (
                "Request identities and cell pointers remain separate. No request-to-cell "
                "join is materialized or authenticated."
            ),
            "annotation": (
                "No eligibility decision, sample, media, rubric approval, rater response, "
                "label, consensus, adjudication, usability decision or threshold exists."
            ),
            "science": (
                "No forecast error, skill, baseline, missingness bound, model comparison, "
                "resource qualification or confirmation result is inferred."
            ),
            "release": "No queue descriptor or confirmation cell was released.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical report")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved pointer inventory differs")
        print(
            "Verified 44 development annotation-input pointers and 1,152 unjoined "
            "request identities; no raw replay, selection, labels or release."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
