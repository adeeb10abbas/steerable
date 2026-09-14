#!/usr/bin/env python3
"""Reproduce a receipt-level coverage audit without reading raw PVC data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
PINNED_RECEIPTS = {
    "compiler": (
        "development-evidence-compiler-formal-004",
        "development_evidence_compiler_job_receipt.json",
        "00329a3c991ee6eb0c68996ecaca1aed59ea3017329272cd82b4b5bfb5cee434",
    ),
    "N3": (
        "timing-n3-development-sidecar-001", "timing_job_receipt.json",
        "7c4aa556d37f170308ef5529489936e3fd7fbce819909d8e1d5937c16965f9d9",
    ),
    "D1": (
        "timing-d1-development-sidecar-002", "timing_job_receipt.json",
        "75490aab357d8290afa10f0e1a85437794a37c81f6a9a0524c94c746198da68d",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def receipt(job: str, filename: str, expected_hash: str) -> tuple[dict, dict]:
    relative = f"results/jobs/{job}/publish/{filename}"
    payload = subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(REPOSITORY),
         "show", f"{RESULTS_COMMIT}:{relative}"], timeout=30,
    )
    digest = hashlib.sha256(payload).hexdigest()
    require(digest == expected_hash, f"Published receipt hash mismatch: {relative}")
    value = json.loads(payload.decode("utf-8"))
    require(value["status"] == "passed", f"Receipt did not pass: {relative}")
    return value, {"git_path": relative, "bytes": len(payload), "sha256": digest}


def audit() -> dict:
    sources, values = {}, {}
    for name, args in PINNED_RECEIPTS.items():
        values[name], sources[name] = receipt(*args)
    compiler = values["compiler"]
    require(compiler["formal_cohort_complete"] is True, "Formal cohort incomplete")
    require(compiler["safe_to_release_confirmation"] is False, "Receipt gate changed")
    models = {}
    aggregates = compiler["inputs"]["aggregate_receipts"]
    require(len(aggregates) == 8, "Expected eight complete development blocks")
    for model in ("N3", "D1"):
        rows = [row for row in aggregates if row["model_id"] == model]
        require(sorted(row["layout_pair_id"] for row in rows) == ["D01", "D02", "D03", "D04"],
                f"Development layout inventory changed for {model}")
        total = {"valid_cells": 0, "actions": 0, "requests": 0}
        cells = set()
        for row in rows:
            raw = Path(row["receipt"]["path"])
            value, identity = receipt(raw.parent.parent.name, raw.name, row["receipt"]["sha256"])
            require(identity["bytes"] == row["receipt"]["bytes"], "Aggregate byte count changed")
            require(value["phase"] == "development" and value["model_config"] == model
                    and value["layout_pair_id"] == row["layout_pair_id"], "Aggregate identity mismatch")
            counts = value["counts"]
            require(counts["completed_valid_behavioral_cells"] == 4, "Incomplete block")
            require(counts["actual_behavioral_actions"] == 1800, "Fixed-duration count changed")
            require(len(value["cell_ids"]) == len(set(value["cell_ids"])) == 4
                    and not cells.intersection(value["cell_ids"]),
                    "Missing or duplicate cells")
            cells.update(value["cell_ids"])
            total["valid_cells"] += counts["completed_valid_behavioral_cells"]
            total["actions"] += counts["actual_behavioral_actions"]
            total["requests"] += counts["actual_behavioral_model_requests"]
            sources[f"{model}_{row['layout_pair_id']}"] = identity
        timing = values[model]
        require(total["valid_cells"] == timing["referenced_behavioral_cells"] == 16,
                "Timing/cell count disagreement")
        require(total["requests"] == timing["referenced_behavioral_model_requests"],
                "Timing/request count disagreement")
        require(timing["behavioral_policy_skill_evaluated"] is False
                and timing["safe_to_release_confirmation"] is False, "Timing claim boundary changed")
        models[model] = {"base_layout_pairs": 4, **total,
                         "timing_receipt_referenced_requests": total["requests"]}
    require(sum(row["valid_cells"] for row in models.values()) == compiler["counts"]["compiled_cells"] == 32,
            "Compiler cell count disagreement")
    require(sum(row["requests"] for row in models.values()) == compiler["counts"]["compiled_source_behavioral_requests"] == 1152,
            "Compiler request count disagreement")
    require(sum(row["actions"] for row in models.values()) == compiler["counts"]["compiled_source_behavioral_actions"] == 14400,
            "Compiler action count disagreement")
    coverage = values["D1"]["request_timing_coverage"]
    require(coverage["total_request_count"] == coverage["source_proven_full_decode_request_count"]
            + coverage["source_unmapped_incremental_decode_request_count"] == 912, "D1 source partition mismatch")
    require(coverage["source_proven_full_decode_request_count"] == coverage["native_clock_matched_request_count"]
            + coverage["action_prefix_truncated_request_count"] == 240, "D1 clock partition mismatch")
    models["D1"]["request_timing_coverage"] = coverage
    models["D1"]["native_clock_matched_fraction"] = {
        "numerator": coverage["native_clock_matched_request_count"],
        "denominator": coverage["total_request_count"],
    }
    models["N3"]["target_binding_coverage"] = "Not quantified by this compact receipt; no inference from request count."
    return {
        "schema_version": "wmf-published-development-coverage-audit-v1",
        "results_commit": RESULTS_COMMIT,
        "scope": "Eight immutable valid development blocks only; pilot and invalid attempts remain in the execution ledger.",
        "verification_level": "Published compact Git receipt bytes and cross-receipt arithmetic; no fresh raw-PVC revalidation.",
        "sources": sources,
        "models": models,
        "new_model_requests": 0,
        "new_behavioral_cells": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "forecast_error_or_skill_bounds": None,
        "bounds_unavailable_reason": "Signed camera/alignment and independent human labels/consensus are not available; timing availability is not prediction correctness.",
        "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Compare an existing audit; never modify it")
    args = parser.parse_args()
    result = audit()
    if args.check:
        require(json.loads(args.check.read_text(encoding="utf-8")) == result, "Saved audit differs")
        print("Verified 11 published receipts; 32 valid development cells, 1152 requests, 14400 actions.")
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
