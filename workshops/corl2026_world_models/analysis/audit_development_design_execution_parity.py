#!/usr/bin/env python3
"""Audit exact development design/execution parity without selecting requests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
SCHEDULE_COMMIT = "defa113bfef8de9c514daac63f236c1f2a878813"
FIXTURE_COMMIT = "4dbb162020fff36d3791735d8bcb665610b30170"
READINESS_COMMIT = "84332bf638595ff7ed955fd263586ecc0e06ab93"
IDENTITY_COMMIT = "33a7483b4cc8255fd97fadb1684b3a4082ee3c9f"

PLANNED_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/planned_cells.csv"
)
SCHEDULE_PATH = (
    "workshops/corl2026_world_models/execution/20260912/parallel_schedule.json"
)
FIXTURE_PATH = (
    "workshops/corl2026_world_models/results/fixed_observation_capture_inventory.json"
)
READINESS_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_readiness_inventory.json"
)
IDENTITY_PATH = (
    "workshops/corl2026_world_models/results/development_request_identity_spine.json"
)
COMPILER_PATH = (
    "results/jobs/development-evidence-compiler-formal-004/publish/"
    "development_evidence_compiler_job_receipt.json"
)

EXPECTED_STATIC = {
    (SPEC_COMMIT, PLANNED_PATH): (
        32948,
        "7d06120a56419877d1acdfdc498c6dc054bf6860ce6bda5b2a25f55ae3b4166e",
    ),
    (SCHEDULE_COMMIT, SCHEDULE_PATH): (
        118108,
        "9a4a1cce1e980a624ec4161cf4438bfa01a6681dcd8fcb6e2b01150b62409a58",
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
    (RESULTS_COMMIT, COMPILER_PATH): (
        12374,
        "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434",
    ),
}

EXPECTED_AGGREGATES = {
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

EXPECTED_ORDERS = {
    "D01": ["original-left", "reflected-right", "original-right", "reflected-left"],
    "D02": ["reflected-right", "original-left", "reflected-left", "original-right"],
    "D03": ["reflected-right", "original-right", "original-left", "reflected-left"],
    "D04": ["original-left", "reflected-left", "reflected-right", "original-right"],
}

PLANNED_HEADER = [
    "cell_id",
    "phase",
    "layout_pair_id",
    "model_config",
    "layout_arm",
    "command",
    "replicate_id",
    "candidate_effective_policy_seed",
    "action_cap",
    "stop_on_success",
    "selected",
    "status",
    "pose_manifest",
    "forecast_mapping",
]

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


def source_payload(commit: str, path: str, expected: tuple[int, str]) -> tuple[bytes, dict]:
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


def source_json(commit: str, path: str, expected: tuple[int, str]) -> tuple[dict, dict]:
    payload, descriptor = source_payload(commit, path, expected)
    return load_json(payload, f"{commit}:{path}"), descriptor


def require_sha256(value: object, label: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


def declared_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Descriptor is not an object: {label}")
    require(set(value) == {"bytes", "path", "sha256"}, f"Descriptor keys changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] >= 0, f"Invalid bytes: {label}")
    require(type(value["path"]) is str and value["path"].startswith("/"), f"Invalid path: {label}")
    require_sha256(value["sha256"], label)
    return value


def fixed_descriptor(value: object, label: str) -> dict:
    descriptor = declared_descriptor(value, label)
    return {
        "bytes": descriptor["bytes"],
        "path": descriptor["path"],
        "sha256": descriptor["sha256"],
    }


def published_job_path(raw_path: str) -> str:
    marker = "/control/jobs/"
    require(marker in raw_path, f"Cannot map control job path: {raw_path}")
    return "results/jobs/" + raw_path.split(marker, 1)[1]


def line_index(records: list[str], label: str) -> dict:
    require(len(records) == len(set(records)), f"Duplicate {label} records")
    payload = "".join(f"{record}\n" for record in sorted(records)).encode("utf-8")
    return {
        "count": len(records),
        "canonical_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def require_zero_counts(value: object, label: str) -> None:
    require(type(value) is dict and value, f"Missing zero-count object: {label}")
    require(
        all(type(item) is int and item == 0 for item in value.values()),
        f"Nonzero new science: {label}",
    )


def parse_planned(payload: bytes) -> list[dict]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("Invalid planned-cells UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    require(reader.fieldnames == PLANNED_HEADER, "Planned-cell CSV header changed")
    rows = list(reader)
    require(len(rows) == 232, "Planned core cell count changed")
    require(all(set(row) == set(PLANNED_HEADER) for row in rows), "Planned row changed")
    require(len({row["cell_id"] for row in rows}) == 232, "Duplicate planned cell ID")
    return rows


def planned_row(row: dict) -> dict:
    require(row["replicate_id"] == "0", f"Replicate changed: {row['cell_id']}")
    require(row["action_cap"] == "450", f"Action cap changed: {row['cell_id']}")
    require(row["stop_on_success"] == "false", f"Stop policy changed: {row['cell_id']}")
    require(row["selected"] == "true", f"Core cell deselected: {row['cell_id']}")
    require(row["status"] == "NOT_RELEASED", f"Frozen plan status changed: {row['cell_id']}")
    require(
        row["pose_manifest"] == row["forecast_mapping"] == "UNQUALIFIED",
        f"Frozen plan placeholder changed: {row['cell_id']}",
    )
    return {
        "cell_id": row["cell_id"],
        "phase": row["phase"],
        "layout_pair_id": row["layout_pair_id"],
        "model_id": row["model_config"],
        "layout_arm": row["layout_arm"],
        "command": row["command"],
        "replicate_id": 0,
        "candidate_effective_policy_seed": int(row["candidate_effective_policy_seed"]),
        "action_cap": 450,
        "stop_on_success": False,
        "selected_in_frozen_core": True,
    }


def audit() -> dict:
    sources = []

    planned_payload, descriptor = source_payload(
        SPEC_COMMIT, PLANNED_PATH, EXPECTED_STATIC[(SPEC_COMMIT, PLANNED_PATH)]
    )
    sources.append(descriptor)
    all_planned = parse_planned(planned_payload)

    def load(commit: str, path: str) -> dict:
        value, item = source_json(commit, path, EXPECTED_STATIC[(commit, path)])
        sources.append(item)
        return value

    schedule = load(SCHEDULE_COMMIT, SCHEDULE_PATH)
    fixture_inventory = load(FIXTURE_COMMIT, FIXTURE_PATH)
    readiness = load(READINESS_COMMIT, READINESS_PATH)
    identity = load(IDENTITY_COMMIT, IDENTITY_PATH)
    compiler = load(RESULTS_COMMIT, COMPILER_PATH)

    development_rows = [planned_row(row) for row in all_planned if row["phase"] == "development"]
    require(len(development_rows) == 32, "Development planned-cell count changed")
    require(
        {row["layout_pair_id"] for row in development_rows} == set(EXPECTED_ORDERS),
        "Development layouts changed",
    )
    require(
        {row["model_id"] for row in development_rows} == {"D1", "N3"},
        "Development models changed",
    )
    planned_by_id = {row["cell_id"]: row for row in development_rows}
    require(len(planned_by_id) == 32, "Development planned IDs are not unique")

    require(schedule.get("D2_selected") is False, "Optional D2 became selected")
    require(schedule.get("cell_count") == 232, "Schedule core ceiling changed")
    require(schedule.get("job_count") == 58, "Schedule job count changed")
    jobs = schedule.get("jobs")
    require(type(jobs) is list and len(jobs) == 58, "Schedule jobs changed")
    development_jobs = [job for job in jobs if job.get("phase") == "development"]
    require(len(development_jobs) == 8, "Development schedule block count changed")
    schedule_by_block = {}
    for job in development_jobs:
        key = (job.get("model_config"), job.get("layout_pair_id"))
        require(key in EXPECTED_AGGREGATES, f"Unexpected scheduled block: {key}")
        require(key not in schedule_by_block, f"Duplicate scheduled block: {key}")
        require(job.get("indivisible") is True, f"Block became divisible: {key}")
        require(job.get("released") is False, f"Frozen schedule release changed: {key}")
        require(job.get("status") == "NOT_RELEASED", f"Frozen schedule status changed: {key}")
        require(
            job.get("condition_order") == EXPECTED_ORDERS[key[1]],
            f"Frozen condition order changed: {key}",
        )
        require(
            job.get("ordered_cell_ids")
            == [
                f"wmf1__development__{key[1]}__{key[0]}__{condition.replace('-', '__')}"
                for condition in EXPECTED_ORDERS[key[1]]
            ],
            f"Frozen ordered cells changed: {key}",
        )
        expected_seed = 1140 if key[0] == "D1" else 2026091100 + int(key[1][1:])
        require(
            job.get("candidate_effective_policy_seed") == expected_seed,
            f"Frozen model seed changed: {key}",
        )
        schedule_by_block[key] = job
    require(set(schedule_by_block) == set(EXPECTED_AGGREGATES), "Scheduled block set changed")

    require(
        readiness.get("schema_version")
        == "wmf-development-annotation-readiness-inventory-v1",
        "Readiness schema changed",
    )
    require(readiness.get("global_study_snapshot") == GLOBAL_SNAPSHOT, "Global snapshot changed")
    require(readiness.get("readiness") == GATE_STATE, "Readiness gate state changed")
    require(
        readiness["development_evidence"].get("valid_cells") == 32
        and readiness["development_evidence"].get("behavioral_requests") == 1152
        and readiness["development_evidence"].get("behavioral_actions") == 14400,
        "Readiness population changed",
    )
    require_zero_counts(readiness["new_activity"], "readiness")

    require(
        fixture_inventory.get("schema_version")
        == "wmf-fixed-observation-capture-inventory-audit-v1",
        "Fixture inventory schema changed",
    )
    require(
        fixture_inventory.get("results_commit") == RESULTS_COMMIT,
        "Fixture result pin changed",
    )
    require(fixture_inventory.get("raw_pvc_revalidated") is False, "Fixture raw PVC claim changed")
    require(
        fixture_inventory.get("new_behavioral_cells") == 0
        and fixture_inventory.get("new_behavioral_actions") == 0
        and fixture_inventory.get("new_model_requests") == 0
        and fixture_inventory.get("labels_created") == 0,
        "Fixture inventory created science",
    )
    development_fixtures = {
        row["layout_pair_id"]: row
        for row in fixture_inventory["layouts"]
        if row.get("stage") == "development"
    }
    require(set(development_fixtures) == set(EXPECTED_ORDERS), "Development fixtures changed")
    require(len(development_fixtures) == 4, "Duplicate development fixture")

    require(
        identity.get("schema_version") == "wmf-development-request-identity-spine-v1",
        "Identity-spine schema changed",
    )
    require(identity.get("global_study_snapshot") == GLOBAL_SNAPSHOT, "Identity snapshot changed")
    require(identity.get("gate_state") == GATE_STATE, "Identity gate state changed")
    require_zero_counts(identity["new_activity"], "identity spine")
    identity_population = identity["development_population"]
    require(
        identity_population
        == {
            "behavioral_actions": 14400,
            "behavioral_requests": 1152,
            "cross_model_identity_overlap": 0,
            "requests_by_model": {"D1": 912, "N3": 240},
            "source_request_receipt_identities_unique": 1152,
            "valid_cells": 32,
        },
        "Request population changed",
    )
    identity_index = identity["request_identity_spine"]["combined_index"]
    require(
        identity_index.get("count") == 1152
        and identity_index.get("canonical_bytes") == 78336
        and identity_index.get("sha256")
        == "3a6206f1e5fc790f1b8c92d9d023c1ce71f969ec95c19ea4482073a96cb86a71",
        "Request identity index changed",
    )

    require(
        compiler.get("schema_version") == "wmf-development-evidence-compiler-queue-job-v1"
        and compiler.get("status") == "passed"
        and compiler.get("decision") == "go"
        and compiler.get("formal_cohort_complete") is True,
        "Formal compiler did not pass",
    )
    require(
        compiler.get("counts")
        == {
            "compiled_cells": 32,
            "compiled_source_behavioral_actions": 14400,
            "compiled_source_behavioral_requests": 1152,
        },
        "Compiler population changed",
    )
    require(
        compiler.get("safe_to_release_confirmation") is False,
        "Compiler released confirmation",
    )
    require_zero_counts(compiler["compiler_science_activity"], "compiler activity")
    require_zero_counts(compiler["science_counts"], "compiler science")

    aggregate_declarations = {}
    for row in compiler["inputs"]["aggregate_receipts"]:
        require(
            type(row) is dict and set(row) == {"layout_pair_id", "model_id", "receipt"},
            "Compiler aggregate declaration changed",
        )
        key = (row["model_id"], row["layout_pair_id"])
        require(key in EXPECTED_AGGREGATES, f"Unexpected compiler aggregate: {key}")
        require(key not in aggregate_declarations, f"Duplicate compiler aggregate: {key}")
        descriptor = declared_descriptor(row["receipt"], f"compiler aggregate {key}")
        expected_path, expected_bytes, expected_sha = EXPECTED_AGGREGATES[key]
        require(
            published_job_path(descriptor["path"]) == expected_path
            and descriptor["bytes"] == expected_bytes
            and descriptor["sha256"] == expected_sha,
            f"Compiler aggregate descriptor changed: {key}",
        )
        aggregate_declarations[key] = descriptor
    require(set(aggregate_declarations) == set(EXPECTED_AGGREGATES), "Compiler block set changed")

    aggregate_by_block = {}
    block_rows = []
    cell_rows = []
    executed_cell_ids = []
    aggregate_index_records = []
    cell_index_records = []
    for key in sorted(EXPECTED_AGGREGATES):
        model_id, layout_id = key
        aggregate_path, expected_bytes, expected_sha = EXPECTED_AGGREGATES[key]
        aggregate, item = source_json(
            RESULTS_COMMIT,
            aggregate_path,
            (expected_bytes, expected_sha),
        )
        sources.append(item)
        aggregate_by_block[key] = aggregate
        schedule_job = schedule_by_block[key]
        fixture = development_fixtures[layout_id]

        require(
            aggregate.get("status") == "passed"
            and aggregate.get("model_config") == model_id
            and aggregate.get("layout_pair_id") == layout_id
            and aggregate.get("phase") == "development",
            f"Aggregate did not pass: {key}",
        )
        require(
            aggregate.get("condition_order") == schedule_job["condition_order"]
            and aggregate.get("cell_ids") == schedule_job["ordered_cell_ids"],
            f"Aggregate order differs from frozen schedule: {key}",
        )
        counts = aggregate["counts"]
        expected_requests = 228 if model_id == "D1" else 60
        require(
            counts.get("completed_valid_behavioral_cells") == 4
            and counts.get("launched_behavioral_cells") == 4
            and counts.get("actual_behavioral_actions") == 1800
            and counts.get("actual_behavioral_model_requests") == expected_requests
            and counts.get("right_censored_behavioral_cells") == 0
            and counts.get("technically_invalid_behavioral_cells") == 0
            and counts.get("unrun_behavioral_cells") == 0,
            f"Aggregate counts changed: {key}",
        )

        expected_environment_seed = 2026091100 + int(layout_id[1:])
        if model_id == "D1":
            require(
                aggregate.get("effective_model_noise_seed") == 1140
                and aggregate.get("environment_seed") == expected_environment_seed,
                f"D1 seed semantics changed: {key}",
            )
        else:
            require(
                aggregate.get("effective_seed") == expected_environment_seed,
                f"N3 seed changed: {key}",
            )

        require(
            fixture.get("layout_pair_id") == layout_id
            and fixture.get("candidate_id") == f"{layout_id}__candidate_00"
            and fixture.get("environment_seed") == expected_environment_seed,
            f"Accepted fixture identity changed: {layout_id}",
        )
        require_sha256(fixture.get("pose_manifest_sha256"), f"fixture pose {layout_id}")
        require_sha256(fixture.get("gate_receipt_sha256"), f"fixture gate {layout_id}")
        prerequisites = aggregate["prerequisites"]
        require_sha256(
            prerequisites.get("accepted_gate_record_sha256"),
            f"accepted gate record {key}",
        )
        require(
            prerequisites.get("candidate_id") == fixture["candidate_id"],
            f"Aggregate fixture candidate changed: {key}",
        )
        gate_descriptor = declared_descriptor(
            prerequisites["development_gate_receipt"], f"aggregate gate receipt {key}"
        )
        require(
            gate_descriptor["sha256"] == fixture["gate_receipt_sha256"],
            f"Aggregate gate receipt differs from accepted fixture: {key}",
        )
        accepted_gate_record_sha256 = prerequisites["accepted_gate_record_sha256"]
        pose_descriptor = declared_descriptor(
            prerequisites["development_pose_manifest"], f"aggregate pose {key}"
        )
        require(
            pose_descriptor["sha256"] == fixture["pose_manifest_sha256"],
            f"Aggregate pose differs from accepted fixture: {key}",
        )
        if model_id == "D1":
            compact = fixture["compact_receipt"]
            capture = declared_descriptor(
                prerequisites["capture_receipt"], f"D1 capture {layout_id}"
            )
            require(
                capture["bytes"] == compact["bytes"]
                and capture["sha256"] == compact["sha256"]
                and published_job_path(capture["path"]) == compact["git_path"],
                f"D1 compact capture differs from accepted fixture: {layout_id}",
            )
            require(
                fixed_descriptor(prerequisites["d1_fixed_observation"], f"D1 fixture {layout_id}")
                == fixed_descriptor(fixture["D1_fixture"], f"fixed D1 fixture {layout_id}"),
                f"D1 fixture descriptor changed: {layout_id}",
            )
            aggregate_capture_binding = "exact_compact_receipt_and_D1_fixture_descriptor"
        else:
            require(
                prerequisites.get("per_development_layout_fixed_capture_required") is False,
                f"N3 fixed-capture boundary changed: {layout_id}",
            )
            aggregate_capture_binding = (
                "accepted_candidate_pose_and_gate_only; fixed-observation N3 descriptor "
                "is inventory context, not an aggregate-declared episode input"
            )

        cell_descriptors = aggregate["cell_receipts"]
        require(
            type(cell_descriptors) is list and len(cell_descriptors) == 4,
            f"Cell descriptors changed: {key}",
        )
        for ordinal, (cell_id, raw_descriptor) in enumerate(
            zip(aggregate["cell_ids"], cell_descriptors, strict=True)
        ):
            descriptor = declared_descriptor(raw_descriptor, f"cell {cell_id}")
            require(cell_id in planned_by_id, f"Executed cell is outside frozen plan: {cell_id}")
            planned = planned_by_id[cell_id]
            condition = f"{planned['layout_arm']}-{planned['command']}"
            require(
                condition == EXPECTED_ORDERS[layout_id][ordinal],
                f"Condition ordinal changed: {cell_id}",
            )
            expected_slug = cell_id.replace("__", "-")
            require(
                Path(descriptor["path"]).parent.name == f"{ordinal:02d}-{expected_slug}",
                f"Cell receipt path ordinal/slug changed: {cell_id}",
            )
            expected_seed = 1140 if model_id == "D1" else expected_environment_seed
            require(
                planned["candidate_effective_policy_seed"] == expected_seed,
                f"Planned cell seed changed: {cell_id}",
            )
            executed_cell_ids.append(cell_id)
            cell_index_records.append(
                f"{cell_id}|{ordinal}|{descriptor['bytes']}|{descriptor['sha256']}"
            )
            cell_rows.append(
                {
                    **planned,
                    "scheduled_ordinal": ordinal,
                    "scheduled_condition": condition,
                    "aggregate_receipt": {
                        "git_path": aggregate_path,
                        "bytes": expected_bytes,
                        "sha256": expected_sha,
                    },
                    "raw_cell_receipt_declaration": fixed_descriptor(
                        descriptor, f"cell output {cell_id}"
                    ),
                    "accepted_fixture": {
                        "candidate_id": fixture["candidate_id"],
                        "environment_seed": fixture["environment_seed"],
                        "pose_manifest_sha256": fixture["pose_manifest_sha256"],
                        "gate_receipt_sha256": fixture["gate_receipt_sha256"],
                        "aggregate_accepted_gate_record_sha256": (
                            accepted_gate_record_sha256
                        ),
                        "compact_receipt": fixture["compact_receipt"],
                        "model_fixture_declaration": fixed_descriptor(
                            fixture[f"{model_id}_fixture"], f"{model_id} fixture {layout_id}"
                        ),
                        "aggregate_binding_scope": aggregate_capture_binding,
                    },
                }
            )

        aggregate_index_records.append(
            f"{model_id}|{layout_id}|{aggregate_path}|{expected_bytes}|{expected_sha}"
        )
        block_rows.append(
            {
                "model_id": model_id,
                "layout_pair_id": layout_id,
                "condition_order": list(EXPECTED_ORDERS[layout_id]),
                "cell_ids": list(aggregate["cell_ids"]),
                "completed_valid_cells": 4,
                "behavioral_actions": 1800,
                "behavioral_requests": expected_requests,
                "right_censored_cells": 0,
                "technical_invalid_cells": 0,
                "unrun_cells": 0,
                "candidate_effective_policy_seed": (
                    1140 if model_id == "D1" else expected_environment_seed
                ),
                "environment_seed": expected_environment_seed,
                "accepted_candidate_id": fixture["candidate_id"],
                "accepted_pose_manifest_sha256": fixture["pose_manifest_sha256"],
                "accepted_gate_receipt_sha256": fixture["gate_receipt_sha256"],
                "aggregate_accepted_gate_record_sha256": (
                    accepted_gate_record_sha256
                ),
                "aggregate_receipt": {
                    "git_path": aggregate_path,
                    "bytes": expected_bytes,
                    "sha256": expected_sha,
                },
            }
        )

    require(
        len(executed_cell_ids) == len(set(executed_cell_ids)) == 32,
        "Executed cell IDs are not unique",
    )
    require(
        set(executed_cell_ids) == set(planned_by_id),
        "Executed set differs from frozen development plan",
    )
    for layout_id in EXPECTED_ORDERS:
        require(
            aggregate_by_block[("N3", layout_id)]["condition_order"]
            == aggregate_by_block[("D1", layout_id)]["condition_order"],
            f"Cross-model condition order differs: {layout_id}",
        )
        require(
            aggregate_by_block[("N3", layout_id)]["prerequisites"][
                "accepted_gate_record_sha256"
            ]
            == aggregate_by_block[("D1", layout_id)]["prerequisites"][
                "accepted_gate_record_sha256"
            ],
            f"Cross-model accepted gate record differs: {layout_id}",
        )

    require(
        len({(item["commit"], item["path"]) for item in sources}) == len(sources),
        "Duplicate source blobs",
    )
    source_records = [
        f"{item['commit']}|{item['path']}|{item['bytes']}|{item['sha256']}"
        for item in sources
    ]

    return {
        "schema_version": "wmf-development-design-execution-parity-v1",
        "study_id": "WMF-ABLATION-001",
        "pins": {
            "spec_commit": SPEC_COMMIT,
            "schedule_commit": SCHEDULE_COMMIT,
            "fixture_inventory_commit": FIXTURE_COMMIT,
            "readiness_commit": READINESS_COMMIT,
            "request_identity_commit": IDENTITY_COMMIT,
            "results_commit": RESULTS_COMMIT,
        },
        "authenticated_source_blobs": {
            "count": len(sources),
            "bytes": sum(item["bytes"] for item in sources),
            "index": line_index(source_records, "source"),
            "sources": sorted(sources, key=lambda item: (item["commit"], item["path"])),
        },
        "parity": {
            "frozen_core_episode_ceiling": 232,
            "optional_D2_selected": False,
            "planned_development_cells": 32,
            "executed_terminal_valid_development_cells": 32,
            "planned_equals_executed_exactly": True,
            "development_blocks": 8,
            "layouts": 4,
            "models": 2,
            "conditions_per_block": 4,
            "aggregate_condition_order_matches_frozen_schedule": True,
            "cross_model_condition_order_matches_per_layout": True,
            "accepted_candidate_pose_gate_identity_matches_both_models": True,
            "behavioral_actions": 14400,
            "behavioral_requests": 1152,
            "per_model": {
                "D1": {"cells": 16, "actions": 7200, "requests": 912},
                "N3": {"cells": 16, "actions": 7200, "requests": 240},
            },
        },
        "indexes": {
            "aggregate_receipts": line_index(aggregate_index_records, "aggregate"),
            "cell_receipt_declarations": line_index(cell_index_records, "cell"),
        },
        "blocks": sorted(block_rows, key=lambda row: (row["layout_pair_id"], row["model_id"])),
        "cells": sorted(cell_rows, key=lambda row: row["cell_id"]),
        "unbound_request_population": {
            "D1": 912,
            "N3": 240,
            "combined_unique": 1152,
            "cross_model_overlap": 0,
            "combined_identity_index": identity_index,
            "request_to_cell_mapping_authenticated": False,
            "production_eligibility_determined": False,
            "production_request_selection_manifest_created": False,
            "selected_request_count": 0,
            "boundary": (
                "The request identity population remains deliberately unbound to cells. "
                "Raw request inventories and receipt content were not opened."
            ),
        },
        "global_study_snapshot": GLOBAL_SNAPSHOT,
        "gate_state": GATE_STATE,
        "current_state_boundary": {
            "current_GM_PVC_worker_state_reconciled": False,
            "live_quiescence_inferred": False,
            "boundary": (
                "Pinned compact evidence authenticates historical development parity only; "
                "current GM/PVC/queue/worker/GPU/child/reap/release-lock state is unknown."
            ),
        },
        "new_activity": {
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "labels": 0,
            "human_responses": 0,
            "forecast_accuracy_estimates": 0,
            "request_selections": 0,
            "annotation_media": 0,
        },
        "claim_boundaries": {
            "design_execution": (
                "Exact planned-cell set, frozen schedule order, passed aggregate declarations "
                "and accepted candidate/pose/gate identities only."
            ),
            "raw_PVC": (
                "Raw cell receipts, fixtures, request inventories, media and PVC bytes were "
                "not opened or rehashed."
            ),
            "fixture": (
                "D1 aggregate receipts explicitly bind compact fixed observations; N3 "
                "aggregates bind the same accepted candidate, pose and gate while recording "
                "their own episode observations. Inventory fixture declarations are not "
                "promoted to episode-input evidence."
            ),
            "annotation": (
                "This is a pre-annotation provenance skeleton, not eligibility, request "
                "selection, rendered media, labels, consensus or a movement threshold."
            ),
            "science": (
                "No forecast skill, model comparison, confirmation result or resource "
                "qualification is inferred."
            ),
            "release": (
                "No gate changed and no confirmation cell or other work was released."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical report")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved parity report differs")
        print(
            "Verified exact 32-cell development design/execution parity; "
            "no request selection, labels, media or release."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
