#!/usr/bin/env python3
"""Authenticate four published D1 cell receipts; report operational values only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
SNAPSHOT_BOUNDARY = (
    "Read-only progress snapshot; not a terminal receipt or scientific classification."
)
PINNED_BLOBS = {
    "D01_snapshot": (
        "results/jobs/d1-development-d01-snapshot-001/publish/diagnostic.json",
        191154, "4f3563f92b28e3dcf713ccdf45df43aab745fb2a91562751f78ae5b1133a6883",
    ),
    "D02_snapshot": (
        "results/jobs/d1-development-d02-snapshot-001/publish/diagnostic.json",
        191132, "108f1d7dd1f85c8c3af6b8cbff42a39ba06521aa72bb2e4f19b8bbfcaf6c727e",
    ),
    "D04_snapshot": (
        "results/jobs/d1-development-d04-snapshot-001/publish/diagnostic.json",
        17615, "cc801b6f262e21fae22bfd3f2093d6898218052026c024c2a7b180eb15b21e1b",
    ),
    "D01_terminal": (
        "results/jobs/d1-development-d01-simulator-003/publish/d1_behavioral_development_receipt.json",
        16510, "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb",
    ),
    "D02_terminal": (
        "results/jobs/d1-development-d02-simulator-003/publish/d1_behavioral_development_receipt.json",
        16510, "d1dc66c57d36cb3e620731c69a6e41fdf52c3cc2a797c2dae7ee36d63d10c6c9",
    ),
}
SELECTED_CELLS = {
    "D01": (
        "00-wmf1-development-D01-D1-original-left",
        "01-wmf1-development-D01-D1-reflected-right",
    ),
    "D02": (
        "00-wmf1-development-D02-D1-reflected-right",
        "01-wmf1-development-D02-D1-original-left",
    ),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def read_blob(path: str, size: int, digest: str) -> tuple[dict, dict]:
    payload = subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(REPOSITORY),
         "show", f"{RESULTS_COMMIT}:{path}"], timeout=30,
    )
    require(len(payload) == size and hashlib.sha256(payload).hexdigest() == digest,
            f"Pinned published blob differs: {path}")
    return json.loads(payload.decode("utf-8")), {
        "git_path": path, "bytes": size, "sha256": digest,
    }


def audit() -> dict:
    values, sources = {}, {}
    for name, pin in PINNED_BLOBS.items():
        values[name], sources[name] = read_blob(*pin)
    for pair in ("D01", "D02", "D04"):
        snapshot = values[f"{pair}_snapshot"]
        require(snapshot["schema_version"] == "wmf-d1-development-readonly-snapshot-v1"
                and snapshot["layout_pair_id"] == pair
                and snapshot["science_claim_boundary"] == SNAPSHOT_BOUNDARY,
                f"Snapshot identity/boundary mismatch: {pair}")

    rows, excluded_cells = [], []
    for pair, keys in SELECTED_CELLS.items():
        snapshot = values[f"{pair}_snapshot"]
        terminal = values[f"{pair}_terminal"]
        attempt_id = f"d1-development-{pair.lower()}-simulator-003"
        require(terminal["schema_version"] == "wmf-d1-behavioral-development-simulator-job-v1"
                and terminal["status"] == "passed"
                and terminal["phase"] == "development"
                and terminal["model_config"] == "D1"
                and terminal["layout_pair_id"] == pair
                and terminal["run_id"] == snapshot["run_id"]
                and terminal["simulator_job_id"] == attempt_id
                and terminal["queue_descriptor"]["job_id"] == attempt_id,
                f"Terminal aggregate identity mismatch: {pair}")
        cells = snapshot["simulator"]["cells"]
        require({key for key, value in cells.items() if value["cell_receipt"] is not None}
                == set(keys), f"Embedded completed-cell selection changed: {pair}")
        for key in keys:
            cell = cells[key]["cell_receipt"]
            encoded = canonical_bytes(cell)
            digest = hashlib.sha256(encoded).hexdigest()
            matches = [ref for ref in terminal["cell_receipts"]
                       if ref["sha256"] == digest and ref["bytes"] == len(encoded)]
            require(len(matches) == 1, f"Embedded cell bytes lack unique terminal match: {key}")
            raw_receipt = matches[0]
            require(raw_receipt["path"].endswith(f"/{attempt_id}/cells/{key}/cell_receipt.json"),
                    f"Raw cell attempt/path mismatch: {key}")
            require(cell["schema_version"] == "wmf-d1-behavioral-development-cell-v1"
                    and cell["status"] == "passed"
                    and cell["phase"] == "development"
                    and cell["model_config"] == "D1"
                    and cell["layout_pair_id"] == pair
                    and cell["cell_id"] in terminal["cell_ids"]
                    and cell["cell_id"] == f"wmf1__development__{pair}__D1__{cell['layout_arm']}__{cell['command']}"
                    and key == f"{cell['condition_index']:02d}-wmf1-development-{pair}-D1-{cell['layout_arm']}-{cell['command']}",
                    f"Embedded cell identity mismatch: {key}")
            require(type(cell["actions_executed"]) is int and cell["actions_executed"] == 450
                    and type(cell["observation_count"]) is int and cell["observation_count"] == 451,
                    f"Fixed-duration fields changed: {key}")
            seconds = cell["runner_timing"]["wall_total_s"]
            require(type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0,
                    f"Invalid wall-time value: {key}")
            rows.append({
                "cell_id": cell["cell_id"], "phase": "development", "model_id": "D1",
                "layout_pair_id": pair, "layout_arm": cell["layout_arm"],
                "command": cell["command"], "condition_index": cell["condition_index"],
                "attempt_id": attempt_id, "paired_run_id": terminal["run_id"],
                "actions_executed": cell["actions_executed"],
                "observation_count": cell["observation_count"],
                "runner_reported_wall_total_s": seconds,
                "wall_time_source_field": "cell_receipt.runner_timing.wall_total_s",
                "wall_time_units": "seconds; unchanged from the published receipt",
                "snapshot_source": f"{pair}_snapshot", "snapshot_cell_key": key,
                "snapshot_at_utc": snapshot["at_utc"],
                "terminal_aggregate_source": f"{pair}_terminal",
                "embedded_cell_bytes_match_terminal_descriptor": True,
                "raw_cell_receipt": raw_receipt,
                "raw_adapter_completion": cell["adapter_completion"],
                "raw_adapter_journal": cell["adapter_journal"],
            })
        for key in sorted(set(cells) - set(keys)):
            require(cells[key]["cell_receipt"] is None, "Unaccounted completed snapshot cell")
            excluded_cells.append({
                "snapshot_source": f"{pair}_snapshot", "snapshot_cell_key": key,
                "reason": "No completed cell receipt in this progress snapshot; no terminal status inferred.",
            })
    require(len(rows) == len({row["cell_id"] for row in rows}) == 4, "Subset identity count changed")
    d04 = values["D04_snapshot"]
    require(d04["simulator"]["cells"]
            and all(value["cell_receipt"] is None for value in d04["simulator"]["cells"].values()),
            "Excluded D04 snapshot now contains a completed receipt")
    return {
        "schema_version": "wmf-retained-d1-operational-subset-audit-v1",
        "results_commit": RESULTS_COMMIT,
        "selection": {
            "included_cells": 4, "previously_audited_d1_development_cells": 16,
            "prior_cohort_audit": {
                "commit": "df576e458e546dcd69c5d676077d16df9b0ae47d",
                "git_path": "workshops/corl2026_world_models/results/interim_development_coverage.json",
                "sha256": "465482c121fedc6c986a2deb013b3ccb9f3b3001f49ee6816df7ae730a1d31e7",
                "denominator_boundary": "The denominator belongs to the previously reproduced full-cohort coverage audit; it is not re-derived by this four-cell subset audit.",
            },
            "rule": "Opportunistic subset selected only by availability of complete cell receipts embedded in published compact development snapshots, with exact byte/hash matches to final aggregates.",
            "representative_sample": False,
        },
        "verification_level": "Five pinned compact Git blobs and four reconstructed embedded cell receipt byte strings; no live process or raw-PVC revalidation.",
        "embedded_receipt_reconstruction": "UTF-8 JSON with sorted keys, indent=2, allow_nan=False and one terminal newline; both byte count and SHA256 must equal a unique final aggregate descriptor.",
        "sources": sources, "cells": rows,
        "excluded_partial_cells": excluded_cells,
        "excluded_published_snapshots": [{
            "source": "D04_snapshot", "snapshot_at_utc": d04["at_utc"],
            "snapshot_cell_keys": sorted(d04["simulator"]["cells"]),
            "reason": "Only a partial cell appears; no completed cell receipt is embedded. This does not classify the later terminal D04 block.",
        }],
        "claim_boundaries": {
            "snapshot": SNAPSHOT_BOUNDARY,
            "timing": "Individual runner-reported wall_total_s values only; these are not asserted to measure whole-job/model-loading time. No average, quantile, full-cohort resource budget, resource qualification, or between-model timing comparison is computed.",
            "stopping": "Snapshot journal event counts and runner success representations are not analyzed. No first-success action index or first-success/action-450 cube-bowl coordinates are available here; this is not A5.",
            "science": "No forecast error, skill, movement decomposition, cohort success rate, annotation labels, or release authority follows from this operational subset.",
        },
        "new_behavioral_cells": 0, "new_model_requests": 0, "labels_created": 0,
        "forecast_accuracy_estimated": False, "resource_budget_qualified": False,
        "a5_stopping_contrast_estimated": False, "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, help="Verify an existing canonical audit without modifying it")
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved operational subset audit differs")
        print("Verified 5 pinned published blobs and 4 byte-matched retained D1 cell receipts; operational subset only, no new science.")
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
