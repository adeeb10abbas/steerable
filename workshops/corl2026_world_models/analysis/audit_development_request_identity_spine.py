#!/usr/bin/env python3
"""Commit the retained development request identity spine without selecting requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


REPOSITORY = Path(__file__).resolve().parents[3]
RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
SPEC_COMMIT = "e67e6c4f990030fa8dc9b40a7e5dc713c99457e5"
ANNOTATION_GATE_COMMIT = "bd860b1a65a9f0d65c2b4e38f641045a390ee52e"
READINESS_COMMIT = "84332bf638595ff7ed955fd263586ecc0e06ab93"
SPEC_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/ablation_spec.json"
)
ANNOTATION_GATE_PATH = (
    "workshops/corl2026_world_models/experiments/forecast_layout/"
    "development_annotation_gate_contract.json"
)
READINESS_PATH = (
    "workshops/corl2026_world_models/results/"
    "development_annotation_readiness_inventory.json"
)
TIMING_PATHS = {
    "D1": "results/jobs/timing-d1-development-sidecar-002/publish/timing_job_receipt.json",
    "N3": "results/jobs/timing-n3-development-sidecar-001/publish/timing_job_receipt.json",
}
EXPECTED = {
    (SPEC_COMMIT, SPEC_PATH): (
        8590,
        "93f9ec18ca30a549867e6a642188a7264514df34985ea3fa494b4cd5f9dd1a15",
    ),
    (ANNOTATION_GATE_COMMIT, ANNOTATION_GATE_PATH): (
        2516,
        "57817caaf6f41c33708749809f21ace4d6f92017322fc1f365c8d3e4d20fe39f",
    ),
    (READINESS_COMMIT, READINESS_PATH): (
        10092,
        "41303ab00b78f90f0f7e3fc721550a8ca6547f74f8d2c38f9b912aae06e8c672",
    ),
    (RESULTS_COMMIT, TIMING_PATHS["D1"]): (
        79824,
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
    ),
    (RESULTS_COMMIT, TIMING_PATHS["N3"]): (
        29935,
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
    ),
}
EXPECTED_IDENTITY_INDEXES = {
    "D1": (
        912,
        62016,
        "0ca707976b59e0957b4bb636982743bb4e5458f5bd08f8177eda09dcec0f0110",
    ),
    "N3": (
        240,
        16320,
        "d25a670fbc024960aeff5d667b4ccf9c6b90b2f3c24691348bc1db3bd0367051",
    ),
    "combined": (
        1152,
        78336,
        "3a6206f1e5fc790f1b8c92d9d023c1ce71f969ec95c19ea4482073a96cb86a71",
    ),
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


def source_blob(commit: str, path: str) -> tuple[dict, dict]:
    payload = git_blob(commit, path)
    descriptor = {
        "commit": commit,
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    require(
        (descriptor["bytes"], descriptor["sha256"]) == EXPECTED[(commit, path)],
        f"Pinned blob changed: {commit}:{path}",
    )
    return load_json(payload, f"{commit}:{path}"), descriptor


def require_zero_counts(value: object, label: str) -> None:
    require(type(value) is dict and value, f"Missing zero-count object: {label}")
    require(
        all(type(item) is int and item == 0 for item in value.values()),
        f"Nonzero science: {label}",
    )


def require_sha256(value: object, label: str) -> str:
    require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"Invalid SHA-256: {label}",
    )
    return value


def identity_index(model_id: str, identities: list[str], label: str) -> dict:
    require(len(identities) == len(set(identities)), f"Duplicate identities: {label}")
    payload = "".join(
        f"{model_id}\t{identity}\n" for identity in sorted(identities)
    ).encode("ascii")
    expected = EXPECTED_IDENTITY_INDEXES[label]
    actual = (len(identities), len(payload), hashlib.sha256(payload).hexdigest())
    require(actual == expected, f"Identity index changed: {label}: {actual}")
    return {
        "count": actual[0],
        "canonical_bytes": actual[1],
        "sha256": actual[2],
        "canonicalization": (
            "lexicographically sorted UTF-8 rows "
            "model_id<TAB>source_request_receipt_sha256<LF>"
        ),
    }


def audit() -> dict:
    sources = []

    def load(commit: str, path: str) -> dict:
        value, descriptor = source_blob(commit, path)
        sources.append(descriptor)
        return value

    spec = load(SPEC_COMMIT, SPEC_PATH)
    annotation_gate = load(ANNOTATION_GATE_COMMIT, ANNOTATION_GATE_PATH)
    readiness = load(READINESS_COMMIT, READINESS_PATH)
    timing = {
        model_id: load(RESULTS_COMMIT, path)
        for model_id, path in sorted(TIMING_PATHS.items())
    }

    require(spec.get("spec_id") == "WMF-ABLATION-001", "Study specification changed")
    require(spec.get("core_episode_ceiling") == 232, "Core ceiling changed")
    require(spec["stages"]["development"]["episodes"] == 32, "Development design changed")
    require(spec["stages"]["confirmation"]["episodes"] == 192, "Confirmation design changed")
    require(spec["analysis"]["request_sample_cap_per_episode"] == 4, "Sample cap changed")
    require(spec["analysis"]["request_sampling_seed"] == 2026091302, "Sampling seed changed")
    require(spec["annotation"]["raters"] == 2, "Rater count changed")
    require(spec["models"]["D2"]["selected"] is False, "Optional D2 became selected")

    require(
        annotation_gate.get("schema_version")
        == "wmf-development-annotation-gate-contract-v1",
        "Annotation gate contract changed",
    )
    require(annotation_gate.get("study_id") == "WMF-ABLATION-001", "Gate study changed")
    require_zero_counts(annotation_gate["science_counts"], "annotation gate")
    require(
        annotation_gate["confirmation_policy"]
        == {
            "confirmation_jobs_released": 0,
            "development_summary_consensus_and_human_usability_decision_still_required": True,
            "safe_to_release_confirmation": False,
        },
        "Confirmation policy changed",
    )
    human_policy = annotation_gate["human_input_policy"]
    require(
        human_policy.get("job_may_create_or_fill_labels") is False
        and human_policy.get("job_may_invent_reviewer_rater_or_adjudicator") is False
        and human_policy.get("pixel_blindness_receipt_required") is True,
        "Human-input boundary changed",
    )

    require(
        readiness.get("schema_version")
        == "wmf-development-annotation-readiness-inventory-v1",
        "Readiness schema changed",
    )
    require(readiness.get("study_id") == "WMF-ABLATION-001", "Readiness study changed")
    require(
        readiness["development_evidence"].get("valid_cells") == 32
        and readiness["development_evidence"].get("behavioral_requests") == 1152
        and readiness["development_evidence"].get("behavioral_actions") == 14400
        and readiness["development_evidence"].get("requests_by_model")
        == {"D1": 912, "N3": 240},
        "Readiness population changed",
    )
    require(
        readiness["global_study_snapshot"]
        == {
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
        },
        "Global study snapshot changed",
    )
    require(
        readiness["readiness"]
        == {
            "annotation_media_ready": False,
            "camera_alignment_ready": False,
            "compiler_ready": True,
            "confirmation_release_ready": False,
            "development_annotation_gate_ready": False,
            "human_annotation_ready": False,
            "resource_freeze_ready": False,
            "timing_partially_ready": True,
        },
        "Readiness gate state changed",
    )
    n3_readiness = readiness["timing_readiness"]["N3"]
    require(
        n3_readiness
        == {
            "coverage_scope": "all retained N3 development requests",
            "physical_time_qualified": True,
            "referenced_cells": 16,
            "referenced_requests": 240,
        },
        "N3 retained timing scope changed",
    )
    require(
        readiness["timing_readiness"].get("all_retained_requests_timing_covered")
        is False
        and readiness["timing_readiness"].get("boundary")
        == (
            "N3 timing is qualified for all 240 retained requests; D1 timing is "
            "qualified only for its source-proven full-decode scope. Missingness is "
            "retained rather than promoted to gate failure."
        ),
        "Retained timing boundary changed",
    )
    require(
        readiness["resource_and_release_gate"]
        == {
            "confirmation_inference_cells": 0,
            "confirmation_released": False,
            "measured_resource_freeze_authenticated_by_this_audit": False,
            "safe_to_release_confirmation": False,
            "status_reports_current_GM_PVC_worker_state_reconciled": False,
        },
        "Current-state reconciliation boundary changed",
    )
    require_zero_counts(readiness["new_activity"], "readiness inventory")

    expected_requests = {"D1": 912, "N3": 240}
    identities_by_model = {}
    timing_summary = {}
    for model_id in ("D1", "N3"):
        receipt = timing[model_id]
        require(
            receipt.get("schema_version")
            == "wmf-development-timing-sidecar-queue-job-v1"
            and receipt.get("status") == "passed"
            and receipt.get("decision") == "go"
            and receipt.get("model_id") == model_id
            and receipt.get("physical_time_qualified") is True
            and receipt.get("referenced_behavioral_cells") == 16
            and receipt.get("referenced_behavioral_model_requests")
            == expected_requests[model_id]
            and receipt.get("safe_to_release_confirmation") is False,
            f"Timing receipt changed: {model_id}",
        )
        require_zero_counts(receipt["science_counts"], f"timing {model_id}")
        declared = receipt.get("source_request_receipt_sha256s")
        require(type(declared) is list, f"Missing request identities: {model_id}")
        identities = [
            require_sha256(value, f"{model_id} source request {ordinal}")
            for ordinal, value in enumerate(declared)
        ]
        require(
            len(identities) == expected_requests[model_id],
            f"Request identity count changed: {model_id}",
        )
        identities_by_model[model_id] = identities
        timing_summary[model_id] = {
            "referenced_cells": 16,
            "referenced_requests": expected_requests[model_id],
            "physical_time_qualified": True,
        }

    d1_coverage = timing["D1"].get("request_timing_coverage")
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
    require(timing["D1"].get("physical_time_coverage_complete") is False, "D1 became complete")
    timing_summary["D1"]["request_timing_coverage"] = d1_coverage
    timing_summary["D1"]["physical_time_coverage_complete"] = False

    d1_set = set(identities_by_model["D1"])
    n3_set = set(identities_by_model["N3"])
    require(len(d1_set) == 912, "D1 identities are not unique")
    require(len(n3_set) == 240, "N3 identities are not unique")
    require(not d1_set.intersection(n3_set), "Cross-model request identity overlap")

    per_model_indexes = {
        model_id: identity_index(model_id, identities_by_model[model_id], model_id)
        for model_id in ("D1", "N3")
    }
    combined_identities = [
        {"model_id": model_id, "source_request_receipt_sha256": identity}
        for model_id in ("D1", "N3")
        for identity in sorted(identities_by_model[model_id])
    ]
    combined_payload = "".join(
        f"{item['model_id']}\t{item['source_request_receipt_sha256']}\n"
        for item in combined_identities
    ).encode("ascii")
    combined_actual = (
        len(combined_identities),
        len(combined_payload),
        hashlib.sha256(combined_payload).hexdigest(),
    )
    require(
        combined_actual == EXPECTED_IDENTITY_INDEXES["combined"],
        f"Combined identity index changed: {combined_actual}",
    )
    combined_index = {
        "count": combined_actual[0],
        "canonical_bytes": combined_actual[1],
        "sha256": combined_actual[2],
        "canonicalization": (
            "lexicographically sorted UTF-8 rows "
            "model_id<TAB>source_request_receipt_sha256<LF>"
        ),
    }

    require(
        len({(item["commit"], item["path"]) for item in sources}) == len(sources),
        "Duplicate source blobs",
    )
    source_index_payload = "".join(
        f"{item['commit']}|{item['path']}|{item['bytes']}|{item['sha256']}\n"
        for item in sorted(sources, key=lambda item: (item["commit"], item["path"]))
    ).encode("ascii")

    return {
        "schema_version": "wmf-development-request-identity-spine-v1",
        "study_id": "WMF-ABLATION-001",
        "pins": {
            "spec_commit": SPEC_COMMIT,
            "annotation_gate_commit": ANNOTATION_GATE_COMMIT,
            "readiness_commit": READINESS_COMMIT,
            "results_commit": RESULTS_COMMIT,
        },
        "authenticated_source_blobs": {
            "count": len(sources),
            "bytes": sum(item["bytes"] for item in sources),
            "index_canonical_bytes": len(source_index_payload),
            "index_sha256": hashlib.sha256(source_index_payload).hexdigest(),
            "sources": sorted(sources, key=lambda item: (item["commit"], item["path"])),
        },
        "development_population": {
            "valid_cells": 32,
            "behavioral_requests": 1152,
            "behavioral_actions": 14400,
            "requests_by_model": expected_requests,
            "source_request_receipt_identities_unique": 1152,
            "cross_model_identity_overlap": 0,
        },
        "request_identity_spine": {
            "identity_semantics": (
                "Each identity is a SHA-256 value declared by a compact timing receipt "
                "for one source request receipt; the referenced raw receipt was not read."
            ),
            "per_model_indexes": per_model_indexes,
            "combined_index": combined_index,
            "identities": combined_identities,
        },
        "timing_classes": {
            "N3": {
                **timing_summary["N3"],
                "coverage_scope": n3_readiness["coverage_scope"],
                "qualified_request_count": 240,
                "retained_request_scope_complete": True,
                "unmapped_request_count": 0,
            },
            "D1": {
                **timing_summary["D1"],
                "native_clock_matched_request_count": 224,
                "action_prefix_truncated_request_count": 16,
                "source_unmapped_incremental_decode_request_count": 672,
            },
            "all_request_identities_committed": True,
            "all_request_contents_authenticated": False,
            "all_requests_timing_covered": False,
        },
        "prospective_annotation_packaging": {
            "request_sampling_seed_declared_in_spec": 2026091302,
            "request_sample_cap_per_episode": 4,
            "development_episode_count": 32,
            "selected_request_ceiling": 128,
            "production_eligibility_determined": False,
            "production_request_selection_manifest_created": False,
            "selected_request_count": 0,
            "annotation_media_created": False,
            "human_handoff_ready": False,
            "use": (
                "Stable metadata join spine for a future camera/alignment-qualified "
                "selection and media package; never a request selection itself."
            ),
        },
        "global_study_snapshot": readiness["global_study_snapshot"],
        "current_state_boundary": {
            "snapshot_semantics": (
                "Cached terminal/core scientific accounting only; active_launched_cells=0 "
                "is not evidence of current process or queue quiescence."
            ),
            "current_GM_PVC_worker_state_reconciled": False,
            "current_GM_PVC_worker_state": "not_reconciled_by_pinned_evidence",
            "live_quiescence_inferred": False,
        },
        "gate_state": readiness["readiness"],
        "new_activity": {
            "model_requests": 0,
            "behavioral_actions": 0,
            "behavioral_cells": 0,
            "labels": 0,
            "human_responses": 0,
            "forecast_accuracy_estimates": 0,
        },
        "claim_boundaries": {
            "published": (
                "Five compact Git blobs were rehashed and strictly parsed at immutable commits."
            ),
            "raw_PVC": (
                "The 1,152 referenced source request receipt blobs, timing inventories, "
                "media and other PVC artifacts were not read or rehashed."
            ),
            "identity": (
                "This commits declared source-request receipt hashes, not request content, "
                "cell assignment, camera identity, alignment or production eligibility."
            ),
            "annotation": (
                "No production selection, rendering, label, consensus, usability decision "
                "or empirical movement threshold was created."
            ),
            "science": (
                "No forecast skill, model comparison, confirmation result or resource "
                "qualification is inferred."
            ),
            "operations": (
                "Cached active_launched_cells=0 is scientific ledger accounting only; "
                "current GM/PVC/queue/worker and release-lock state remains unreconciled."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical spine")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved request identity spine differs")
        print(
            "Verified 1,152 unique retained development request identities; "
            "no selection, media, labels or release."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
