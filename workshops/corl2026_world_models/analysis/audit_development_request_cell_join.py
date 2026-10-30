#!/usr/bin/env python3
"""Replay the retained builder order into an explicit request-to-cell join.

This audit uses only immutable compact Git blobs.  It does not open the raw
request receipts, raw request inventories, cell receipts, timing sidecars, or
media on the GM PVC.  The join is therefore an authenticated mapping of the
published *declared receipt identities*, not authentication of their contents.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
PREFLIGHT_COMMIT = "45c3a5d591c539779e8ff7db53d4af1c64273271"
IDENTITY_COMMIT = "33a7483b4cc8255fd97fadb1684b3a4082ee3c9f"
PREFLIGHT_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_raw_retrieval_preflight.json"
)
IDENTITY_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_request_identity_spine.json"
)
BUILDER_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_timing_sidecar_jobs.py"
)
QUEUE_HELPER_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "forecast_timing_queue_jobs.py"
)
TIMING_PATHS = {
    "D1": "results/jobs/timing-d1-development-sidecar-002/publish/timing_job_receipt.json",
    "N3": "results/jobs/timing-n3-development-sidecar-001/publish/timing_job_receipt.json",
}
BUILDER_COMMITS = {
    "D1": "fcaa098cc45a3f8e32fa0a72203fd64795fa456a",
    "N3": "1f25ecf74b2342aa36e6d168fc1f34a5df273d85",
}
AGGREGATE_PATHS = {
    "D1": {
        "D01": "results/jobs/d1-development-d01-simulator-003/publish/d1_behavioral_development_receipt.json",
        "D02": "results/jobs/d1-development-d02-simulator-003/publish/d1_behavioral_development_receipt.json",
        "D03": "results/jobs/d1-development-d03-simulator-002/publish/d1_behavioral_development_receipt.json",
        "D04": "results/jobs/d1-development-d04-simulator-003/publish/d1_behavioral_development_receipt.json",
    },
    "N3": {
        "D01": "results/jobs/n3-development-d01-002/publish/n3_behavioral_development_receipt.json",
        "D02": "results/jobs/n3-development-d02-002/publish/n3_behavioral_development_receipt.json",
        "D03": "results/jobs/n3-development-d03-001/publish/n3_behavioral_development_receipt.json",
        "D04": "results/jobs/n3-development-d04-001/publish/n3_behavioral_development_receipt.json",
    },
}
DIAGNOSTIC_PATH = (
    "results/jobs/timing-d1-development-sidecar-decode-diagnostic-001/"
    "publish/diagnostic.json"
)
LAYOUTS = ("D01", "D02", "D03", "D04")
MODELS = ("D1", "N3")
REQUESTS_PER_CELL = {"D1": 57, "N3": 15}
REQUESTS_BY_MODEL = {"D1": 912, "N3": 240}
BUILDER_FUNCTION_AST = (
    15360,
    "2e01f4969186ad2630b9da91873f214d72547e1f6855d1badb15fbc6261e1240",
)
RUN_SIDECAR_JOB_ASTS = {
    "D1": (
        15982,
        "84f14c199828047ba044f8e3cfbc3b951a44d4eb438c33c3c0180fc4423fbb9a",
    ),
    "N3": (
        12925,
        "1dc35ded6a7d58f678f5a48cf4fa143822a5f4588f1d19f6968dc683647ed3a5",
    ),
}
QUEUE_HELPER_FUNCTION_ASTS = {
    "compact_bytes": (
        1108,
        "a00a887ed9cf1c1f8e411111286d90375887cff7042caf98f1d1234ac4a4fb83",
    ),
    "signed_document": (
        1189,
        "a46b427c671b084254ebe7aaffd9ce05f32056c63e7cd619e594805bfe7e5866",
    ),
}

EXPECTED_D1_TIMING_COVERAGE = {
    "action_prefix_truncated_request_count": 16,
    "action_prefix_truncated_target_binding_count": 32,
    "native_clock_matched_request_count": 224,
    "native_clock_matched_target_binding_count": 448,
    "source_proven_full_decode_request_count": 240,
    "source_unmapped_incremental_decode_request_count": 672,
    "total_request_count": 912,
    "unmapped_potential_authority_target_count": 1344,
}

EXPECTED = {
    (PREFLIGHT_COMMIT, PREFLIGHT_PATH): (
        143329,
        "3d5e1c030bf8d7b607cd586f8a80b5ac1382529d3ce81210b2495cf4a734b35f",
    ),
    (IDENTITY_COMMIT, IDENTITY_PATH): (
        181453,
        "9b534d2e821b2d98ec128cd87f5c32b4674177b6a308286e161801b6a2d58d8e",
    ),
    (RESULTS_COMMIT, TIMING_PATHS["D1"]): (
        79824,
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
    ),
    (RESULTS_COMMIT, TIMING_PATHS["N3"]): (
        29935,
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
    ),
    (RESULTS_COMMIT, DIAGNOSTIC_PATH): (
        54598,
        "45b4817f4961255bf3595011e86f31b8f1056317cdc2dd731ada9459664a8df6",
    ),
    (BUILDER_COMMITS["D1"], BUILDER_PATH): (
        61548,
        "f6a5a58970dd3e53ce95ed5e736f89657f79c17cb0446dd9c5dbc5821bd2733d",
    ),
    (BUILDER_COMMITS["N3"], BUILDER_PATH): (
        47163,
        "636828f6dcfce3d1f9d163dc8fc0d65b5853056f0d15bf83af56254f4e1f45a1",
    ),
    (BUILDER_COMMITS["D1"], QUEUE_HELPER_PATH): (
        86978,
        "d448a02ad627bc65c143a3932ae2a4378be642e36a49645f5dc8c3727f00c353",
    ),
    (BUILDER_COMMITS["N3"], QUEUE_HELPER_PATH): (
        86978,
        "d448a02ad627bc65c143a3932ae2a4378be642e36a49645f5dc8c3727f00c353",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["D1"]["D01"]): (
        16510,
        "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["D1"]["D02"]): (
        16510,
        "d1dc66c57d36cb3e620731c69a6e41fdf52c3cc2a797c2dae7ee36d63d10c6c9",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["D1"]["D03"]): (
        16510,
        "eb7f4f894b1e56c1b8971ee1155d77e7aeb1b62c9c6cfc41de0483e3baf2dd4f",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["D1"]["D04"]): (
        16510,
        "78a9b8b7095e48ce5412a902a7120f3c9ad71e3edb6eeeddb228720fe1b458fa",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["N3"]["D01"]): (
        12551,
        "a9ca815c6e99b270357220fa5ccb97562c50d667dd3c23745d460e5ec8c053de",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["N3"]["D02"]): (
        12551,
        "efd2ea9f2f015c671a9d9d2d9dc22c1c1f9f5f3ffd87e4810acd5e3767b2f73e",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["N3"]["D03"]): (
        12551,
        "d3e85f2087d7e24d57974a9344c333408e9769552715d4669e16f1e059082d0e",
    ),
    (RESULTS_COMMIT, AGGREGATE_PATHS["N3"]["D04"]): (
        12551,
        "e5f3206c6da2b9caa2ca9cd4c3ed82142157656ac9a1d1744486d14fa6e57318",
    ),
}

EXPECTED_MAPPING_INDEXES = {
    "D1": (912, 123872, "4a620dc2a859a703c6d3ae327ddfbdb21f8a3fbb26277bf54579e994b322282b"),
    "N3": (240, 32480, "3e2a85618beabd88e33de69edfa1f1398bc277a2c5c9791153b0ecf3c4b0e3ab"),
    "combined": (1152, 156352, "6d84f5801a2b0caab68b201fec9f0066fde9ba4a5466bc154d6b48e63c783ade"),
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


def compact_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
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


def source_blob(commit: str, path: str) -> tuple[bytes, dict]:
    payload = git_blob(commit, path)
    descriptor = {
        "bytes": len(payload),
        "commit": commit,
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    require(
        (descriptor["bytes"], descriptor["sha256"]) == EXPECTED[(commit, path)],
        f"Pinned blob changed: {commit}:{path}",
    )
    return payload, descriptor


def source_json(commit: str, path: str) -> tuple[dict, dict]:
    payload, descriptor = source_blob(commit, path)
    return load_json(payload, f"{commit}:{path}"), descriptor


def require_sha256(value: object, label: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


def require_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Descriptor is not an object: {label}")
    require(set(value) == {"bytes", "path", "sha256"}, f"Descriptor keys changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] > 0, f"Descriptor bytes changed: {label}")
    require(type(value["path"]) is str and value["path"].startswith("/data/"), f"Descriptor path changed: {label}")
    require_sha256(value["sha256"], label)
    return dict(value)


def verify_signed(value: dict, label: str) -> None:
    observed = require_sha256(value.get("payload_sha256"), f"{label} payload")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(hashlib.sha256(compact_bytes(unsigned)).hexdigest() == observed, f"Signature changed: {label}")


def builder_source_identity(payload: bytes, label: str) -> dict:
    tree = ast.parse(payload)
    functions = {}
    function_nodes = {}
    function_expectations = {
        "build_request_inventory": BUILDER_FUNCTION_AST,
        "run_sidecar_job": RUN_SIDECAR_JOB_ASTS[label],
    }
    for name, expected in function_expectations.items():
        matches = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        require(
            len(matches) == 1,
            f"Builder function population changed: {label} {name}",
        )
        function_nodes[name] = matches[0]
        encoded = (
            ast.dump(matches[0], annotate_fields=True, include_attributes=False)
            + "\n"
        ).encode("utf-8")
        actual = (len(encoded), hashlib.sha256(encoded).hexdigest())
        require(
            actual == expected,
            f"Builder function semantics changed: {label} {name}",
        )
        functions[name] = {
            "ast_dump_bytes": actual[0],
            "ast_dump_sha256": actual[1],
        }

    receipt_hash_bindings = []
    for node in ast.walk(function_nodes["run_sidecar_job"]):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "source_request_receipt_sha256s"
            ):
                receipt_hash_bindings.append(value)
    require(
        len(receipt_hash_bindings) == 1
        and isinstance(receipt_hash_bindings[0], ast.Name)
        and receipt_hash_bindings[0].id == "request_hashes",
        f"Published request-hash binding changed: {label}",
    )

    expected_globals = {
        "LAYOUTS": LAYOUTS,
        "REQUEST_COUNT": REQUESTS_BY_MODEL,
        "REQUESTS_PER_CELL": REQUESTS_PER_CELL,
    }
    source_globals = {}
    for name, expected in expected_globals.items():
        values = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                values.append(ast.literal_eval(node.value))
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == name
            ):
                values.append(ast.literal_eval(node.value))
        require(
            len(values) == 1 and values[0] == expected,
            f"Builder global changed: {label} {name}",
        )
        source_globals[name] = values[0]

    return {
        "functions": functions,
        "published_request_hash_binding": (
            "source_request_receipt_sha256s=request_hashes"
        ),
        "source_globals": source_globals,
    }


def queue_helper_identity(payload: bytes, label: str) -> dict:
    tree = ast.parse(payload)
    identities = {}
    for name, expected in QUEUE_HELPER_FUNCTION_ASTS.items():
        matches = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == name
        ]
        require(len(matches) == 1, f"Queue helper function population changed: {label} {name}")
        encoded = (
            ast.dump(matches[0], annotate_fields=True, include_attributes=False) + "\n"
        ).encode("utf-8")
        actual = (len(encoded), hashlib.sha256(encoded).hexdigest())
        require(actual == expected, f"Queue helper semantics changed: {label} {name}")
        identities[name] = {
            "ast_dump_bytes": actual[0],
            "ast_dump_sha256": actual[1],
        }
    return identities


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
    payload = "".join(
        "|".join(str(row[field]) for field in fields) + "\n" for row in rows
    ).encode("ascii")
    actual = (len(rows), len(payload), hashlib.sha256(payload).hexdigest())
    require(actual == EXPECTED_MAPPING_INDEXES[label], f"Mapping index changed: {label}")
    return {
        "canonical_bytes": actual[1],
        "canonicalization": "ordered ASCII pipe-delimited mapping rows with LF terminators",
        "count": actual[0],
        "sha256": actual[2],
    }


def audit() -> dict:
    sources = []

    def load(commit: str, path: str) -> dict:
        value, descriptor = source_json(commit, path)
        sources.append(descriptor)
        return value

    preflight = load(PREFLIGHT_COMMIT, PREFLIGHT_PATH)
    identity = load(IDENTITY_COMMIT, IDENTITY_PATH)
    timing = {model: load(RESULTS_COMMIT, TIMING_PATHS[model]) for model in MODELS}
    diagnostic = load(RESULTS_COMMIT, DIAGNOSTIC_PATH)
    builders = {}
    builder_identities = {}
    queue_helpers = {}
    queue_helper_identities = {}
    for model in MODELS:
        payload, descriptor = source_blob(BUILDER_COMMITS[model], BUILDER_PATH)
        sources.append(descriptor)
        builders[model] = descriptor
        builder_identities[model] = builder_source_identity(payload, model)
        helper_payload, helper_descriptor = source_blob(
            BUILDER_COMMITS[model], QUEUE_HELPER_PATH
        )
        sources.append(helper_descriptor)
        queue_helpers[model] = helper_descriptor
        queue_helper_identities[model] = queue_helper_identity(
            helper_payload, model
        )

    aggregates = {model: {} for model in MODELS}
    for model in MODELS:
        for layout in LAYOUTS:
            aggregates[model][layout] = load(
                RESULTS_COMMIT, AGGREGATE_PATHS[model][layout]
            )

    require(preflight.get("schema_version") == "wmf-development-annotation-raw-retrieval-preflight-v1", "Preflight schema changed")
    require(preflight.get("study_id") == "WMF-ABLATION-001", "Preflight study changed")
    require(preflight["global_study_snapshot"] == GLOBAL_SNAPSHOT, "Global snapshot changed")
    require(preflight["recovery_scope"]["development_cells"] == 32, "Development cells changed")
    require(preflight["recovery_scope"]["scheduled_requests"] == 1152, "Scheduled requests changed")
    require(preflight["recovery_scope"]["combined_unique_dependencies"] == 56, "Dependency count changed")
    require(preflight["recovery_scope"]["combined_declared_bytes"] == 32664538, "Dependency bytes changed")
    previous_boundary = preflight["unjoined_request_identity_population"]
    require(
        previous_boundary["rows"] == 1152
        and previous_boundary["joined_to_cells"] is False
        and previous_boundary["request_to_cell_mapping_materialized"] is False
        and previous_boundary["request_to_cell_mapping_authenticated"] is False,
        "Preflight mapping boundary changed",
    )
    require(preflight["availability_and_gate_boundary"]["safe_to_release_confirmation"] is False, "Preflight release gate promoted")

    require(identity.get("schema_version") == "wmf-development-request-identity-spine-v1", "Identity schema changed")
    identity_rows = identity["request_identity_spine"]["identities"]
    require(type(identity_rows) is list and len(identity_rows) == 1152, "Identity population changed")
    identity_set = set()
    for ordinal, row in enumerate(identity_rows):
        require(type(row) is dict and set(row) == {"model_id", "source_request_receipt_sha256"}, f"Identity row changed: {ordinal}")
        require(row["model_id"] in MODELS, f"Identity model changed: {ordinal}")
        identity_set.add((row["model_id"], require_sha256(row["source_request_receipt_sha256"], f"identity {ordinal}")))
    require(len(identity_set) == 1152, "Identity spine is not unique")

    preflight_cells = {}
    for row in preflight["cells"]:
        descriptor = require_descriptor(row["raw_cell_receipt"], f"preflight cell {row.get('cell_id')}")
        key = (descriptor["path"], descriptor["bytes"], descriptor["sha256"])
        require(key not in preflight_cells, "Preflight cell descriptor duplicated")
        preflight_cells[key] = row
    require(len(preflight_cells) == 32, "Preflight cell population changed")

    mapping_rows = []
    per_cell = []
    aggregate_sources = []
    d1_timing_coverage = None
    d1_physical_time_coverage_complete = None
    for model in MODELS:
        receipt = timing[model]
        require(
            receipt.get("schema_version") == "wmf-development-timing-sidecar-queue-job-v1"
            and receipt.get("status") == "passed"
            and receipt.get("decision") == "go"
            and receipt.get("study_id") == "WMF-ABLATION-001"
            and receipt.get("model_id") == model
            and receipt.get("study_commit") == BUILDER_COMMITS[model]
            and receipt.get("referenced_behavioral_cells") == 16
            and receipt.get("referenced_behavioral_model_requests") == REQUESTS_BY_MODEL[model]
            and receipt.get("safe_to_release_confirmation") is False,
            f"Timing receipt changed: {model}",
        )
        verify_signed(receipt, f"timing {model}")
        require(all(type(value) is int and value == 0 for value in receipt["science_counts"].values()), f"Timing science counts changed: {model}")
        if model == "D1":
            require(
                receipt.get("request_timing_coverage")
                == EXPECTED_D1_TIMING_COVERAGE,
                "D1 timing coverage changed",
            )
            require(
                receipt.get("physical_time_coverage_complete") is False,
                "D1 physical-time coverage boundary changed",
            )
            d1_timing_coverage = dict(receipt["request_timing_coverage"])
            d1_physical_time_coverage_complete = receipt[
                "physical_time_coverage_complete"
            ]
        implementation = receipt["implementation"]["sidecar_builder"]
        require(
            implementation["bytes"] == builders[model]["bytes"]
            and implementation["sha256"] == builders[model]["sha256"]
            and f"/sources/{BUILDER_COMMITS[model]}/{BUILDER_PATH}" in implementation["path"],
            f"Builder receipt binding changed: {model}",
        )
        hashes = receipt.get("source_request_receipt_sha256s")
        require(type(hashes) is list and len(hashes) == REQUESTS_BY_MODEL[model], f"Request hash count changed: {model}")
        hashes = [require_sha256(value, f"{model} request hash {ordinal}") for ordinal, value in enumerate(hashes)]
        require(len(set(hashes)) == len(hashes), f"Request hashes duplicated: {model}")
        cursor = 0
        for layout in LAYOUTS:
            aggregate = aggregates[model][layout]
            evidence = receipt["inputs"]["development_aggregates"][layout]
            aggregate_descriptor = require_descriptor(evidence["aggregate_receipt"], f"aggregate {model} {layout}")
            source_descriptor = next(
                source
                for source in sources
                if source["commit"] == RESULTS_COMMIT
                and source["path"] == AGGREGATE_PATHS[model][layout]
            )
            require(
                aggregate_descriptor["bytes"] == source_descriptor["bytes"]
                and aggregate_descriptor["sha256"] == source_descriptor["sha256"],
                f"Aggregate Git/raw identity differs: {model} {layout}",
            )
            aggregate_sources.append(source_descriptor)
            require(
                aggregate.get("status") == "passed"
                and aggregate.get("study_id") == "WMF-ABLATION-001"
                and aggregate.get("model_config") == model
                and aggregate.get("layout_pair_id") == layout
                and aggregate.get("phase") == "development",
                f"Aggregate identity changed: {model} {layout}",
            )
            cell_ids = aggregate.get("cell_ids")
            cell_descriptors = aggregate.get("cell_receipts")
            condition_order = aggregate.get("condition_order")
            require(type(cell_ids) is list and len(cell_ids) == 4 and len(set(cell_ids)) == 4, f"Aggregate cell IDs changed: {model} {layout}")
            require(type(cell_descriptors) is list and len(cell_descriptors) == 4, f"Aggregate cell descriptors changed: {model} {layout}")
            require(
                type(condition_order) is list
                and len(condition_order) == 4
                and len(set(condition_order)) == 4,
                f"Aggregate condition order changed: {model} {layout}",
            )
            require(cell_descriptors == evidence["cell_receipts"], f"Timing/aggregate cell order differs: {model} {layout}")
            for cell_index, (cell_id, raw_descriptor) in enumerate(zip(cell_ids, cell_descriptors)):
                descriptor = require_descriptor(raw_descriptor, f"cell {model} {layout} {cell_index}")
                key = (descriptor["path"], descriptor["bytes"], descriptor["sha256"])
                require(key in preflight_cells, f"Cell is absent from preflight: {model} {layout} {cell_index}")
                preflight_cell = preflight_cells[key]
                require(
                    preflight_cell["cell_id"] == cell_id
                    and preflight_cell["model_id"] == model
                    and preflight_cell["layout_pair_id"] == layout
                    and preflight_cell["scheduled_ordinal"] == cell_index,
                    f"Cell lineage differs: {model} {layout} {cell_index}",
                )
                require(
                    condition_order[cell_index]
                    == preflight_cell["scheduled_condition"],
                    f"Scheduled condition differs: {model} {layout} {cell_index}",
                )
                start = cursor
                for request_index in range(REQUESTS_PER_CELL[model]):
                    mapping_rows.append(
                        {
                            "cell_id": cell_id,
                            "layout_pair_id": layout,
                            "model_id": model,
                            "request_index": request_index,
                            "scheduled_condition": preflight_cell["scheduled_condition"],
                            "scheduled_ordinal": cell_index,
                            "source_request_receipt_sha256": hashes[cursor],
                        }
                    )
                    cursor += 1
                per_cell.append(
                    {
                        "cell_id": cell_id,
                        "first_model_request_ordinal": start,
                        "last_model_request_ordinal": cursor - 1,
                        "layout_pair_id": layout,
                        "model_id": model,
                        "requests": REQUESTS_PER_CELL[model],
                        "scheduled_condition": preflight_cell["scheduled_condition"],
                        "scheduled_ordinal": cell_index,
                    }
                )
        require(cursor == len(hashes), f"Request hash replay incomplete: {model}")

    require(
        d1_timing_coverage is not None
        and d1_physical_time_coverage_complete is not None,
        "D1 timing boundary was not authenticated",
    )
    require(len(mapping_rows) == 1152, "Mapping row count changed")
    mapping_set = {
        (row["model_id"], row["source_request_receipt_sha256"])
        for row in mapping_rows
    }
    missing_identity_rows = len(identity_set - mapping_set)
    extra_identity_rows = len(mapping_set - identity_set)
    require(
        len(mapping_set) == 1152
        and missing_identity_rows == 0
        and extra_identity_rows == 0,
        "Mapping does not exactly cover identity spine",
    )
    mapping_keys = {(row["model_id"], row["cell_id"], row["request_index"]) for row in mapping_rows}
    duplicate_mapping_keys = len(mapping_rows) - len(mapping_keys)
    require(duplicate_mapping_keys == 0, "Mapping keys are duplicated")
    model_hash_sets = {
        model: {
            row["source_request_receipt_sha256"]
            for row in mapping_rows
            if row["model_id"] == model
        }
        for model in MODELS
    }
    cross_model_identity_overlap = model_hash_sets["D1"] & model_hash_sets["N3"]
    require(not cross_model_identity_overlap, "D1/N3 request identities overlap")
    require(len(per_cell) == 32, "Per-cell mapping population changed")

    require(
        diagnostic.get("schema_version") == "wmf-d1-development-decode-diagnostic-v1"
        and diagnostic.get("status") == "read_only_summary_complete"
        and diagnostic.get("request_count") == 912
        and diagnostic.get("unique_request_receipt_hashes") == 912
        and diagnostic.get("mismatch_count") == 672
        and diagnostic.get("mismatch_examples_truncated") is True
        and diagnostic.get("scientific_qualification") is False,
        "D1 decode diagnostic changed",
    )
    require(all(type(value) is int and value == 0 for value in diagnostic["science_counts"].values()), "D1 diagnostic science counts changed")
    diagnostic_rows = diagnostic.get("first_mismatches_against_prior_nine_frame_assumption")
    require(type(diagnostic_rows) is list and len(diagnostic_rows) == 64, "D1 diagnostic row count changed")
    d1_rows = [row for row in mapping_rows if row["model_id"] == "D1"]
    diagnostic_ordinals = set()
    diagnostic_hashes = set()
    for check_index, check in enumerate(diagnostic_rows):
        ordinal = check.get("ordinal")
        require(type(ordinal) is int and 0 <= ordinal < len(d1_rows), f"D1 diagnostic ordinal changed: {check_index}")
        row = d1_rows[ordinal]
        require(
            check.get("cell_id") == row["cell_id"]
            and check.get("entry_request_index") == row["request_index"]
            and check.get("receipt_request_index") == row["request_index"]
            and check.get("receipt_sha256") == row["source_request_receipt_sha256"],
            f"D1 diagnostic mapping differs: {check_index}",
        )
        diagnostic_ordinals.add(ordinal)
        diagnostic_hashes.add(check.get("receipt_sha256"))
    require(
        len(diagnostic_ordinals) == 64 and len(diagnostic_hashes) == 64,
        "D1 diagnostic corroboration rows are duplicated",
    )

    indexes = {
        "D1": mapping_index(d1_rows, "D1"),
        "N3": mapping_index([row for row in mapping_rows if row["model_id"] == "N3"], "N3"),
        "combined": mapping_index(mapping_rows, "combined"),
    }
    source_rows = [
        f"{row['commit']}|{row['path']}|{row['bytes']}|{row['sha256']}"
        for row in sorted(sources, key=lambda row: (row["commit"], row["path"]))
    ]
    source_payload = "".join(f"{row}\n" for row in source_rows).encode("ascii")
    require(len({(row["commit"], row["path"]) for row in sources}) == len(sources), "Source blobs duplicated")

    return {
        "schema_version": "wmf-development-request-cell-join-v1",
        "study_id": "WMF-ABLATION-001",
        "pins": {
            "identity_commit": IDENTITY_COMMIT,
            "preflight_commit": PREFLIGHT_COMMIT,
            "results_commit": RESULTS_COMMIT,
            "timing_builder_commits": BUILDER_COMMITS,
        },
        "authenticated_source_blobs": {
            "bytes": sum(row["bytes"] for row in sources),
            "count": len(sources),
            "index_canonical_bytes": len(source_payload),
            "index_sha256": hashlib.sha256(source_payload).hexdigest(),
            "sources": sorted(sources, key=lambda row: (row["commit"], row["path"])),
        },
        "order_replay_authority": {
            "builder_function_identity": BUILDER_FUNCTION_AST[1],
            "builder_function_semantics": (
                "The byte-pinned executed function iterates D01-D04, then each "
                "aggregate cell_receipts list in order, then enumerates each cell's "
                "model_response_received events by request_index before emitting the "
                "source_request_receipt_sha256s list without reordering."
            ),
            "builder_sources": builders,
            "builder_ast_identities": builder_identities,
            "queue_helper_sources": queue_helpers,
            "queue_helper_ast_identities": queue_helper_identities,
            "list_preservation_boundary": (
                "The pinned helper copies the receipt object and signs compact JSON; "
                "object keys are sorted but array order is not changed."
            ),
            "aggregate_receipt_sources": sorted(
                aggregate_sources, key=lambda row: row["path"]
            ),
            "d1_explicit_diagnostic_rows_checked": 64,
            "d1_explicit_diagnostic_rows_matched": 64,
            "d1_explicit_diagnostic_rows_sample_truncated": True,
            "d1_explicit_diagnostic_scope": (
                "Corroborative 64-row truncated sample; it is not complete mapping coverage."
            ),
        },
        "mapping": {
            "declared_request_identity_to_cell_mapping_materialized": True,
            "declared_request_identity_to_cell_mapping_authenticated_by_order_replay": True,
            "raw_request_receipt_content_authenticated": False,
            "rows": len(mapping_rows),
            "rows_by_model": REQUESTS_BY_MODEL,
            "cells": len(per_cell),
            "cells_by_model": {"D1": 16, "N3": 16},
            "requests_per_cell": REQUESTS_PER_CELL,
            "missing_identity_rows": missing_identity_rows,
            "extra_identity_rows": extra_identity_rows,
            "duplicate_mapping_keys": duplicate_mapping_keys,
            "cross_model_identity_overlap": len(cross_model_identity_overlap),
            "indexes": indexes,
            "per_cell": per_cell,
            "rows_detail": mapping_rows,
        },
        "raw_retrieval_boundary": {
            "declared_dependencies": 56,
            "declared_dependency_bytes": 32664538,
            "raw_PVC_paths_revalidated": False,
            "raw_PVC_bytes_opened": False,
            "raw_bytes_rehashed": False,
            "current_raw_availability_verified": False,
            "remaining_use": (
                "Raw retrieval is still required to authenticate request contents, "
                "timing inventories, cell receipts, fixtures and media inputs; it is "
                "no longer required merely to materialize the declared hash-to-cell index."
            ),
        },
        "annotation_and_release_boundary": {
            "camera_crop_contract_authenticated": False,
            "physical_alignment_authenticated": False,
            "production_eligibility_determined": False,
            "production_request_selection_manifest_created": False,
            "selected_request_count": 0,
            "annotation_media_created": 0,
            "human_handoff_ready": False,
            "human_responses": 0,
            "labels": 0,
            "forecast_accuracy_estimates": 0,
            "measured_resource_freeze_authenticated": False,
            "confirmation_inference_cells": 0,
            "safe_to_release_confirmation": False,
        },
        "timing_boundary": {
            "N3_retained_request_identities": 240,
            "D1_retained_request_identities": 912,
            "D1_source_proven_full_decode_requests": d1_timing_coverage[
                "source_proven_full_decode_request_count"
            ],
            "D1_native_clock_matched_requests": d1_timing_coverage[
                "native_clock_matched_request_count"
            ],
            "D1_action_prefix_truncated_requests": d1_timing_coverage[
                "action_prefix_truncated_request_count"
            ],
            "D1_source_unmapped_incremental_decode_requests": d1_timing_coverage[
                "source_unmapped_incremental_decode_request_count"
            ],
            "D1_physical_time_coverage_complete": (
                d1_physical_time_coverage_complete
            ),
            "all_request_identities_mapped_to_cells": True,
            "all_requests_timing_complete": d1_physical_time_coverage_complete,
        },
        "gate_state": preflight["gate_state"],
        "global_study_snapshot": GLOBAL_SNAPSHOT,
        "current_state_boundary": {
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "cached_active_scientific_ledger_cells": 0,
        },
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
            "join": (
                "The mapping is a deterministic replay of exact ordered hash lists from "
                "published timing receipts through the byte-pinned builders and ordered "
                "published aggregate receipts."
            ),
            "raw": (
                "No raw request receipt, raw request inventory, cell receipt, timing "
                "sidecar, fixture or media byte was opened or rehashed."
            ),
            "timing": (
                "Cell assignment does not promote D1's 672 source-unmapped requests or "
                "establish complete physical-time correspondence."
            ),
            "annotation": (
                "No eligibility, selection, media, rater handoff, label, consensus, "
                "threshold, usability decision or estimate was created."
            ),
            "science": (
                "No policy skill, forecast accuracy, model comparison or confirmation "
                "result is inferred."
            ),
            "operations": (
                "Retained Git evidence does not establish current GM/PVC/worker/GPU/"
                "claim/reap/release-lock state or live quiescence."
            ),
            "release": "No queue descriptor or confirmation cell was released.",
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
        require(args.check.read_bytes() == encoded, "Saved request-cell join differs")
        print("Verified 1,152 declared request identities across 32 development cells")
    elif args.output:
        args.output.write_bytes(encoded)
        print(f"Wrote {args.output} ({len(encoded)} bytes)")
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
