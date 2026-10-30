#!/usr/bin/env python3
"""Audit retained development evidence for annotation readiness without labeling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
STATUS_COMMIT = "10dc0155c086d4fa79928da003a2720ac0604a47"
SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
ANNOTATION_GATE_COMMIT = "bd860b1a65a9f0d65c2b4e38f641045a390ee52e"
STATUS_PATH = "workshops/corl2026_world_models/execution/20260912/status.json"
SPEC_PATH = "workshops/corl2026_world_models/experiments/forecast_layout/ablation_spec.json"
ANNOTATION_GATE_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_annotation_gate_contract.json"
)
COMPILER_PATH = (
    "results/jobs/development-evidence-compiler-formal-004/publish/"
    "development_evidence_compiler_job_receipt.json"
)
N3_TIMING_PATH = (
    "results/jobs/timing-n3-development-sidecar-001/publish/timing_job_receipt.json"
)
D1_TIMING_PATH = (
    "results/jobs/timing-d1-development-sidecar-002/publish/timing_job_receipt.json"
)
CAMERA_FAILURE_PATH = (
    "results/jobs/camera-crop-replay-witness-001/publish/"
    "camera_crop_witness_job_failure.json"
)
CAMERA_JOB_PATH = (
    "results/wmf_ablation_001_20260912/jobs/camera-crop-replay-witness-001.json"
)
FIXED_INVENTORY_PATH = (
    "workshops/corl2026_world_models/results/fixed_observation_capture_inventory.json"
)
EXPECTED = {
    (STATUS_COMMIT, STATUS_PATH): (
        61094,
        "87b441a25887bf6cf701d57e92ca197b191d42a5f74264cf0234971485f434f2",
    ),
    (SPEC_COMMIT, SPEC_PATH): (
        8590,
        "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
    ),
    (ANNOTATION_GATE_COMMIT, ANNOTATION_GATE_PATH): (
        2516,
        "57817caaf6f41c33708749809f21ace4d6f92017322fc1f365c8d3e4d20fe39f",
    ),
    (RESULTS_COMMIT, COMPILER_PATH): (
        12374,
        "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434",
    ),
    (RESULTS_COMMIT, N3_TIMING_PATH): (
        29935,
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
    ),
    (RESULTS_COMMIT, D1_TIMING_PATH): (
        79824,
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
    ),
    (RESULTS_COMMIT, CAMERA_FAILURE_PATH): (
        2304,
        "b8ba6bb884f7c2daaf6695bafd7fed97675d8bc9bc11a125e4353eb52df42550",
    ),
    (STATUS_COMMIT, FIXED_INVENTORY_PATH): (
        53104,
        "76c6e12920608c00212048fd807bfe42e7058d10ebee7229657e5d62b5ba3a54",
    ),
}
EXPECTED_AGGREGATE_INDEX = (
    8,
    624,
    "7e54075f96e4d4953f5c460df8e97963f833c4d1ed26cf908fdaf8396f57ffe8",
)
EXPECTED_IMPLEMENTATION_INDEX = (
    9,
    780,
    "45d98e3c1d7a4426feda5bfad0d54327429de194946ec48bb3f5a2868ea183c1",
)
EXPECTED_CELL_INDEX = (
    32,
    3680,
    "01e99bb3cc6d67566e22dd20ab4f1cab5d8fab0c33eec510bf96be2aef9ac135",
)
EXPECTED_PREDECESSOR_INDEX = (
    3,
    668,
    "7884a7a470869e61c5dbb97dc3e763886ed83e93fd12f13b4fab961b61bc7583",
)


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


def source_blob(commit: str, path: str) -> tuple[dict, dict]:
    payload = git_blob(commit, path)
    size = len(payload)
    digest = hashlib.sha256(payload).hexdigest()
    expected = EXPECTED.get((commit, path))
    if expected is not None:
        require((size, digest) == expected, f"Pinned blob changed: {commit}:{path}")
    return load_json(payload, f"{commit}:{path}"), {
        "commit": commit,
        "path": path,
        "bytes": size,
        "sha256": digest,
    }


def declared_descriptor(value: object, label: str) -> dict:
    require(type(value) is dict, f"Descriptor is not an object: {label}")
    require(set(value) == {"bytes", "path", "sha256"}, f"Descriptor keys changed: {label}")
    require(type(value["bytes"]) is int and value["bytes"] >= 0, f"Invalid bytes: {label}")
    require(type(value["path"]) is str and value["path"].startswith("/"), f"Invalid path: {label}")
    require(
        type(value["sha256"]) is str
        and len(value["sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["sha256"]),
        f"Invalid SHA-256: {label}",
    )
    return value


def line_index(records: list[str], expected: tuple[int, int, str], label: str) -> dict:
    require(len(records) == len(set(records)), f"Duplicate {label} records")
    payload = "".join(f"{record}\n" for record in sorted(records)).encode("ascii")
    actual = (len(records), len(payload), hashlib.sha256(payload).hexdigest())
    require(actual == expected, f"{label} index changed: {actual}")
    return {"count": actual[0], "canonical_bytes": actual[1], "sha256": actual[2]}


def require_zero_counts(value: object, label: str) -> None:
    require(type(value) is dict and value, f"Missing zero-count object: {label}")
    require(all(type(item) is int and item == 0 for item in value.values()), f"Nonzero science: {label}")


def published_job_path(raw_path: str) -> str:
    marker = "/control/jobs/"
    require(marker in raw_path, f"Cannot map control job path: {raw_path}")
    return "results/jobs/" + raw_path.split(marker, 1)[1]


def audit() -> dict:
    sources = []

    def load(commit: str, path: str) -> dict:
        value, descriptor = source_blob(commit, path)
        sources.append(descriptor)
        return value

    status = load(STATUS_COMMIT, STATUS_PATH)
    spec = load(SPEC_COMMIT, SPEC_PATH)
    annotation_gate = load(ANNOTATION_GATE_COMMIT, ANNOTATION_GATE_PATH)
    compiler = load(RESULTS_COMMIT, COMPILER_PATH)
    n3_timing = load(RESULTS_COMMIT, N3_TIMING_PATH)
    d1_timing = load(RESULTS_COMMIT, D1_TIMING_PATH)
    camera_failure = load(RESULTS_COMMIT, CAMERA_FAILURE_PATH)
    camera_job = load(RESULTS_COMMIT, CAMERA_JOB_PATH)
    fixed_inventory = load(STATUS_COMMIT, FIXED_INVENTORY_PATH)

    require(spec.get("spec_id") == "WMF-ABLATION-001", "Study specification changed")
    require(spec.get("core_episode_ceiling") == 232, "Core ceiling changed")
    require(spec["stages"]["development"]["episodes"] == 32, "Development design changed")
    require(spec["stages"]["confirmation"]["episodes"] == 192, "Confirmation design changed")
    require(spec["analysis"]["request_sample_cap_per_episode"] == 4, "Sample cap changed")
    require(spec["annotation"]["raters"] == 2, "Rater count changed")
    require(spec["models"]["D2"]["selected"] is False, "Optional D2 became selected")

    require(status.get("core_cells_completed_valid") == 40, "Valid core count changed")
    require(status.get("core_cells_completed_technical_invalid") == 4, "Invalid count changed")
    require(status.get("core_cells_active_not_yet_counted") == 0, "Active core count changed")
    require(status.get("core_episodes_launched") == 44, "Launched core count changed")
    require(
        status.get("generation_requests_launched") == 1460
        and status.get("generation_requests_completed") == 1460
        and status.get("request_arithmetic")
        == {
            "active_requests_completed_excluded": 0,
            "active_requests_launched_excluded": 0,
            "invariant": "18_nonbehavioral + 1440_valid_behavioral + 2_technical_invalid_behavioral = 1460_terminal_requests",
            "nonbehavioral_generation_qualification": 18,
            "terminal_behavioral_technical_invalid": 2,
            "terminal_behavioral_valid": 1440,
            "terminal_total": 1460,
        },
        "Global request accounting changed",
    )
    development = status["behavioral_execution"]["development"]
    confirmation = status["behavioral_execution"]["confirmation"]
    require(development.get("completed_valid_cells") == 32, "Development count changed")
    require(
        confirmation.get("completed_valid_cells") == 0
        and confirmation.get("inference_cells_launched") == 0
        and confirmation.get("released") is False,
        "Confirmation release state changed",
    )
    annotation_status = status["development_annotation_gate"]
    require(
        annotation_status.get("human_labels_consumed") == 0
        and annotation_status.get("human_pixel_blindness_receipts_consumed") == 0
        and annotation_status.get("human_responses_consumed") == 0
        and annotation_status.get("labels_created_by_job") == 0,
        "Human annotation counters changed",
    )
    require(
        "real media" in annotation_status.get("remaining", "")
        and "rubric/examples" in annotation_status.get("remaining", "")
        and "consensus" in annotation_status.get("remaining", "")
        and "movement threshold" in annotation_status.get("remaining", ""),
        "Annotation hold reasons changed",
    )
    resource_status = status["development_resource_qualification"]
    require(
        resource_status.get("qualification_wave_released") is False
        and resource_status.get("safe_to_release_confirmation") is False
        and resource_status.get("last_published_status")
        == "claimed_current_state_unknown",
        "Resource qualification state changed",
    )
    require(
        status.get("current_owned_job_state")
        == "unknown_until_fresh_results_publication_after_authorized_gm_reconciliation",
        "Current owned-job boundary changed",
    )
    publication_status = status["results_publication_incident"]
    require(
        publication_status.get("current_job_states_known") is False
        and publication_status.get("safe_to_release_new_work") is False
        and publication_status.get("fresh_snapshot_after_empty_queue") is False
        and publication_status.get("status")
        == "external_gm_inspection_required_git_only_recovery_exhausted",
        "Results-publication incident boundary changed",
    )

    require(
        compiler.get("schema_version") == "wmf-development-evidence-compiler-queue-job-v1"
        and compiler.get("status") == "passed"
        and compiler.get("decision") == "go"
        and compiler.get("formal_cohort_complete") is True,
        "Formal compiler receipt did not pass",
    )
    require(compiler.get("safe_to_release_confirmation") is False, "Compiler released confirmation")
    require_zero_counts(compiler["compiler_science_activity"], "compiler activity")
    require_zero_counts(compiler["science_counts"], "compiler science counts")
    require(
        compiler.get("counts")
        == {
            "compiled_cells": 32,
            "compiled_source_behavioral_actions": 14400,
            "compiled_source_behavioral_requests": 1152,
        },
        "Compiler cohort counts changed",
    )

    aggregate_records = []
    cell_records = []
    aggregate_sources = []
    cells_by_model = Counter()
    requests_by_model = Counter()
    actions_by_model = Counter()
    all_cell_ids = []
    aggregates = compiler["inputs"]["aggregate_receipts"]
    require(type(aggregates) is list and len(aggregates) == 8, "Aggregate binding count changed")
    for row in aggregates:
        require(type(row) is dict and set(row) == {"layout_pair_id", "model_id", "receipt"}, "Aggregate row changed")
        model = row["model_id"]
        layout = row["layout_pair_id"]
        require(model in {"N3", "D1"} and layout in {"D01", "D02", "D03", "D04"}, "Aggregate identity changed")
        descriptor = declared_descriptor(row["receipt"], f"aggregate {model}/{layout}")
        aggregate_records.append(f"{model}|{layout}|{descriptor['bytes']}|{descriptor['sha256']}")
        path = published_job_path(descriptor["path"])
        receipt, actual = source_blob(RESULTS_COMMIT, path)
        sources.append(actual)
        require(
            (actual["bytes"], actual["sha256"]) == (descriptor["bytes"], descriptor["sha256"]),
            f"Published aggregate differs: {model}/{layout}",
        )
        require(
            receipt.get("status") == "passed"
            and receipt.get("model_config") == model
            and receipt.get("layout_pair_id") == layout,
            f"Aggregate did not pass: {model}/{layout}",
        )
        counts = receipt["counts"]
        expected_requests = 60 if model == "N3" else 228
        require(
            counts.get("completed_valid_behavioral_cells") == 4
            and counts.get("actual_behavioral_actions") == 1800
            and counts.get("actual_behavioral_model_requests") == expected_requests
            and counts.get("right_censored_behavioral_cells") == 0
            and counts.get("technically_invalid_behavioral_cells") == 0
            and counts.get("unrun_behavioral_cells") == 0,
            f"Aggregate counts changed: {model}/{layout}",
        )
        cell_ids = receipt["cell_ids"]
        cell_descriptors = receipt["cell_receipts"]
        require(len(cell_ids) == len(cell_descriptors) == 4, f"Cell binding count changed: {model}/{layout}")
        for cell_id, cell_descriptor in zip(cell_ids, cell_descriptors, strict=True):
            item = declared_descriptor(cell_descriptor, f"cell {cell_id}")
            require(f"__{layout}__{model}__" in cell_id, f"Cell identity changed: {cell_id}")
            cell_records.append(f"{cell_id}|{item['bytes']}|{item['sha256']}")
            all_cell_ids.append(cell_id)
            cells_by_model[model] += 1
        requests_by_model[model] += expected_requests
        actions_by_model[model] += 1800
        aggregate_sources.append(
            {
                "model_id": model,
                "layout_pair_id": layout,
                "git_path": path,
                "bytes": actual["bytes"],
                "sha256": actual["sha256"],
            }
        )
    require(len(all_cell_ids) == len(set(all_cell_ids)) == 32, "Development cell IDs are not unique")
    require(cells_by_model == {"D1": 16, "N3": 16}, "Per-model cell count changed")
    require(requests_by_model == {"D1": 912, "N3": 240}, "Per-model request count changed")
    require(actions_by_model == {"D1": 7200, "N3": 7200}, "Per-model action count changed")
    valid_behavioral_actions = (
        actions_by_model["D1"]
        + actions_by_model["N3"]
        + status["behavioral_execution"]["pilot"]["D1"]["valid_executed_actions"]
        + status["behavioral_execution"]["pilot"]["N3"]["published_behavioral_actions"]
    )
    valid_behavioral_observations = (
        development["D1"]["valid_behavioral_observations"]
        + development["N3"]["valid_behavioral_observations"]
        + status["behavioral_execution"]["pilot"]["D1"]["valid_observations"]
        + status["behavioral_execution"]["pilot"]["N3"]["published_behavioral_observations"]
    )
    require(valid_behavioral_actions == 18000, "Global valid action count changed")
    require(valid_behavioral_observations == 18040, "Global valid observation count changed")

    implementation_records = []
    for key, descriptor in compiler["implementation"].items():
        item = declared_descriptor(descriptor, f"implementation {key}")
        implementation_records.append(f"{key}|{item['bytes']}|{item['sha256']}")

    predecessor_records = []
    predecessor_sources = []
    for ordinal in range(1, 4):
        job_id = f"development-evidence-compiler-formal-{ordinal:03d}"
        path = f"results/jobs/{job_id}/publish/development_evidence_compiler_job_failure.json"
        failure, actual = source_blob(RESULTS_COMMIT, path)
        sources.append(actual)
        require(
            failure.get("job_id") == job_id
            and failure.get("status") == "technical_invalid"
            and failure.get("decision") == "no_go",
            f"Predecessor failure changed: {job_id}",
        )
        require_zero_counts(failure["science_counts"], f"predecessor {job_id}")
        detail = failure["failure"]["detail"]
        predecessor_records.append(
            f"{job_id}|{failure['study_commit']}|{actual['sha256']}|{detail}"
        )
        predecessor_sources.append(
            {
                "job_id": job_id,
                "bytes": actual["bytes"],
                "sha256": actual["sha256"],
                "failure_detail": detail,
            }
        )

    aggregate_index = line_index(aggregate_records, EXPECTED_AGGREGATE_INDEX, "aggregate")
    implementation_index = line_index(
        implementation_records, EXPECTED_IMPLEMENTATION_INDEX, "implementation"
    )
    cell_index = line_index(cell_records, EXPECTED_CELL_INDEX, "cell")
    predecessor_index = line_index(
        predecessor_records, EXPECTED_PREDECESSOR_INDEX, "predecessor"
    )

    compiled_outputs = compiler["outputs"]["primary_compiled_outputs"]
    require(type(compiled_outputs) is dict and len(compiled_outputs) == 6, "Compiled output set changed")
    require(
        not any("selection" in key or "annotation" in key for key in compiled_outputs),
        "Compiler unexpectedly contains a production annotation selection",
    )
    compiled_output_descriptors = {
        key: declared_descriptor(value, f"compiled output {key}")
        for key, value in sorted(compiled_outputs.items())
    }
    compiled_output_bytes = sum(item["bytes"] for item in compiled_output_descriptors.values())
    require(compiled_output_bytes == 13917649, "Compiled output byte total changed")
    require(
        compiler["outputs"].get("semantically_validated_nonreceipt_file_count") == 70,
        "Compiler validated-file count changed",
    )

    for timing, model, cells, requests in (
        (n3_timing, "N3", 16, 240),
        (d1_timing, "D1", 16, 912),
    ):
        require(
            timing.get("schema_version") == "wmf-development-timing-sidecar-queue-job-v1"
            and timing.get("status") == "passed"
            and timing.get("decision") == "go"
            and timing.get("model_id") == model
            and timing.get("physical_time_qualified") is True
            and timing.get("referenced_behavioral_cells") == cells
            and timing.get("referenced_behavioral_model_requests") == requests
            and timing.get("safe_to_release_confirmation") is False,
            f"Timing receipt changed: {model}",
        )
        require_zero_counts(timing["science_counts"], f"timing {model}")
    d1_coverage = d1_timing["request_timing_coverage"]
    require(
        d1_coverage
        == {
            "action_prefix_truncated_request_count": 16,
            "action_prefix_truncated_target_binding_count": 32,
            "native_clock_matched_request_count": 224,
            "native_clock_matched_target_binding_count": 448,
            "source_proven_full_decode_request_count": 240,
            "source_unmapped_incremental_decode_request_count": 672,
            "total_request_count": 912,
            "unmapped_potential_authority_target_count": 1344,
        },
        "D1 timing coverage changed",
    )
    require(d1_timing.get("physical_time_coverage_complete") is False, "D1 coverage became complete")

    require(
        camera_failure.get("status") == "technical_invalid"
        and camera_failure.get("decision") == "no_go"
        and camera_failure.get("safe_to_release_confirmation") is False
        and camera_failure.get("whole_frame_identity") is False
        and camera_failure.get("simulator_state_render_used") is False,
        "Camera witness failure changed",
    )
    require_zero_counts(camera_failure["science_counts"], "camera witness")
    require(
        camera_failure["failure"].get("detail")
        == "camera replay child exits changed: {'N3': 1, 'D1': 1}",
        "Camera witness failure detail changed",
    )
    require(
        camera_job.get("status") == "failed"
        and camera_job.get("returncode") == 1
        and camera_job.get("child_reaped") is True,
        "Camera witness child state changed",
    )

    require(
        annotation_gate.get("schema_version") == "wmf-development-annotation-gate-contract-v1"
        and annotation_gate["confirmation_policy"].get("safe_to_release_confirmation") is False
        and annotation_gate["confirmation_policy"].get("confirmation_jobs_released") == 0,
        "Annotation gate release policy changed",
    )
    require(
        annotation_gate["confirmation_policy"].get(
            "development_summary_consensus_and_human_usability_decision_still_required"
        )
        is True,
        "Consensus/usability requirement changed",
    )
    require_zero_counts(annotation_gate["science_counts"], "annotation gate source")
    human_policy = annotation_gate["human_input_policy"]
    require(
        human_policy.get("approved_rubric_and_examples_required_through_freeze") is True
        and human_policy.get("job_may_create_or_fill_labels") is False
        and human_policy.get("job_may_invent_reviewer_rater_or_adjudicator") is False
        and human_policy.get("locked_response_required_for_collection") is True
        and human_policy.get("pixel_blindness_receipt_required") is True,
        "Human input policy changed",
    )

    require(
        fixed_inventory["capture_job_accounting"] == {
            "behavioral_actions": 0,
            "jobs": 29,
            "model_requests": 0,
        }
        and fixed_inventory["declared_raw_artifacts"]["count"] == 145
        and fixed_inventory.get("raw_pvc_revalidated") is False
        and fixed_inventory.get("confirmation_released") is False,
        "Fixed-observation inventory boundary changed",
    )

    source_index_payload = "".join(
        f"{item['commit']}|{item['path']}|{item['bytes']}|{item['sha256']}\n"
        for item in sorted(sources, key=lambda item: (item["commit"], item["path"]))
    ).encode("ascii")
    require(len({(item["commit"], item["path"]) for item in sources}) == len(sources), "Duplicate source blobs")

    return {
        "schema_version": "wmf-development-annotation-readiness-inventory-v1",
        "study_id": "WMF-ABLATION-001",
        "pins": {
            "status_commit": STATUS_COMMIT,
            "spec_commit": SPEC_COMMIT,
            "annotation_gate_commit": ANNOTATION_GATE_COMMIT,
            "results_commit": RESULTS_COMMIT,
        },
        "authenticated_source_blobs": {
            "count": len(sources),
            "bytes": sum(item["bytes"] for item in sources),
            "index_canonical_bytes": len(source_index_payload),
            "index_sha256": hashlib.sha256(source_index_payload).hexdigest(),
        },
        "global_study_snapshot": {
            "planned_core_cells": 232,
            "launched_core_cells": 44,
            "terminal_valid_cells": 40,
            "terminal_technical_invalid_cells": 4,
            "active_launched_cells": 0,
            "terminal_requests": 1460,
            "nonbehavioral_requests": 18,
            "valid_behavioral_requests": 1440,
            "technical_invalid_behavioral_requests": 2,
            "valid_behavioral_actions": valid_behavioral_actions,
            "valid_behavioral_observations": valid_behavioral_observations,
            "confirmation_inference_cells": 0,
        },
        "design": {
            "core_episode_ceiling": 232,
            "development_cells": 32,
            "confirmation_cells": 192,
            "request_sample_cap_per_episode": 4,
            "development_selected_request_ceiling": 128,
            "independent_raters_required": 2,
            "optional_D2_selected": False,
        },
        "development_evidence": {
            "formal_compiler_status": "passed",
            "valid_cells": 32,
            "behavioral_requests": 1152,
            "behavioral_actions": 14400,
            "cells_by_model": dict(sorted(cells_by_model.items())),
            "requests_by_model": dict(sorted(requests_by_model.items())),
            "actions_by_model": dict(sorted(actions_by_model.items())),
            "aggregate_receipt_index": aggregate_index,
            "cell_descriptor_index": cell_index,
            "aggregate_sources": sorted(
                aggregate_sources, key=lambda item: (item["model_id"], item["layout_pair_id"])
            ),
            "implementation_declaration_index": implementation_index,
            "predecessor_fail_closed_index": predecessor_index,
            "predecessor_failures": predecessor_sources,
            "declared_primary_compiled_outputs": {
                "count": len(compiled_output_descriptors),
                "declared_bytes": compiled_output_bytes,
                "semantically_validated_nonreceipt_file_count": 70,
                "raw_bytes_rehashed_by_this_audit": False,
            },
        },
        "timing_readiness": {
            "N3": {
                "referenced_cells": 16,
                "referenced_requests": 240,
                "physical_time_qualified": True,
                "coverage_scope": "all retained N3 development requests",
            },
            "D1": {
                "referenced_cells": 16,
                "referenced_requests": 912,
                "physical_time_qualified": True,
                "physical_time_coverage_complete": False,
                "coverage_scope": "source-proven full conditioning-origin decodes only",
                "request_timing_coverage": d1_coverage,
            },
            "all_retained_requests_timing_covered": False,
            "boundary": "N3 timing is qualified for all 240 retained requests; D1 timing is qualified only for its source-proven full-decode scope. Missingness is retained rather than promoted to gate failure.",
        },
        "camera_and_media_gate": {
            "camera_witness_status": "technical_invalid",
            "camera_witness_decision": "no_go",
            "camera_child_exit_codes": {"N3": 1, "D1": 1},
            "camera_queue_wrapper_child_reaped": True,
            "crop_contract_published": False,
            "whole_frame_identity": False,
            "real_blinded_development_media_authenticated_by_this_audit": False,
            "media_renderable_for_human_annotation": False,
        },
        "fixture_boundary": {
            "inventory_reported_passed_fixed_observation_receipts": 29,
            "inventory_reported_declared_raw_artifacts": 145,
            "inventory_report_blob_rehashed": True,
            "underlying_29_receipt_blobs_rehashed_by_this_audit": False,
            "raw_pvc_revalidated": False,
            "use": "Physical-observation and preprocessing-fixture identity only; not annotation media.",
        },
        "human_annotation_gate": {
            "request_sampling_algorithm_declared_in_spec": spec["analysis"]["sampling"],
            "request_sampling_seed_declared_in_spec": spec["analysis"]["request_sampling_seed"],
            "production_request_selection_manifest_authenticated_by_this_audit": False,
            "approved_rubric_and_examples_authenticated_by_this_audit": False,
            "landmark_and_ambiguity_rules_authenticated_by_this_audit": False,
            "pixel_blindness_receipts_consumed": 0,
            "human_responses_consumed": 0,
            "human_labels_consumed": 0,
            "consensus_record_authenticated_by_this_audit": False,
            "adjudication_record_authenticated_by_this_audit": False,
            "human_usability_decision_authenticated_by_this_audit": False,
            "empirical_movement_threshold_authenticated_by_this_audit": False,
            "human_handoff_ready": False,
        },
        "resource_and_release_gate": {
            "status_reports_current_GM_PVC_worker_state_reconciled": False,
            "measured_resource_freeze_authenticated_by_this_audit": False,
            "confirmation_released": False,
            "confirmation_inference_cells": 0,
            "safe_to_release_confirmation": False,
        },
        "readiness": {
            "compiler_ready": True,
            "timing_partially_ready": True,
            "camera_alignment_ready": False,
            "annotation_media_ready": False,
            "human_annotation_ready": False,
            "resource_freeze_ready": False,
            "development_annotation_gate_ready": False,
            "confirmation_release_ready": False,
        },
        "new_activity": {
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "labels": 0,
            "forecast_accuracy_estimates": 0,
        },
        "claim_boundaries": {
            "published": "Published compact Git blobs are rehashed and strictly parsed at immutable commits.",
            "raw_PVC": "Compiler bundles, timing inventories, cell receipts and media referenced only by descriptors were not read or rehashed.",
            "annotation": "This is a readiness inventory, not an annotation package, label set, consensus result or movement threshold.",
            "science": "No forecast skill, model comparison, confirmation result or resource qualification is inferred.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical inventory")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved annotation-readiness inventory differs")
        print(
            "Verified 32 valid development cells, 1,152 requests, timing coverage, "
            "and fail-closed camera/human gates; readiness evidence only."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
