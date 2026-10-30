#!/usr/bin/env python3
"""Authenticate three retained N3 pilot cells; report operational values only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
ATTEMPT_ID = "n3-p00-pilot-004"
SNAPSHOT_AT_UTC = "2026-09-13T04:38:28.348384+00:00"
PINNED_BLOBS = {
    "progress_snapshot": (
        "results/jobs/detached-attempt-snapshot-003/publish/diagnostic.json",
        173811,
        "57f2a346a1bc181fe0901cd7193fd5bbf952a68aa1210b35b70cfedef6a1394d",
    ),
    "terminal_aggregate": (
        "results/jobs/n3-p00-pilot-004/publish/n3_behavioral_pilot_receipt.json",
        6473,
        "5a7a861acc82615b4eb7aa92308f40f7a0c965e65b6edaf12f776cb64b18b0e0",
    ),
}
COMPLETE_CELL_KEYS = (
    "00-wmf1-pilot-P00-N3-reflected-right",
    "01-wmf1-pilot-P00-N3-reflected-left",
    "02-wmf1-pilot-P00-N3-original-left",
)
PARTIAL_CELL_KEY = "03-wmf1-pilot-P00-N3-original-right"
ABSENT_SCIENCE_KEYS = (
    "first_success_action_index",
    "first_success_positions",
    "first_success_position",
    "first_success_cube_bowl_positions",
    "action_450_positions",
    "action450_positions",
    "forecast_error",
    "persistence_error",
    "primary_skill",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(REPOSITORY),
            "show",
            f"{RESULTS_COMMIT}:{path}",
        ],
        timeout=30,
    )


def read_blob(path: str, size: int, digest: str) -> tuple[dict, dict]:
    payload = git_bytes(path)
    require(
        len(payload) == size and hashlib.sha256(payload).hexdigest() == digest,
        f"Pinned published blob differs: {path}",
    )
    return json.loads(payload.decode("utf-8")), {
        "git_path": path,
        "bytes": size,
        "sha256": digest,
    }


def iter_key_paths(value: object, prefix: tuple[str, ...] = ()):
    if isinstance(value, dict):
        for key, child in value.items():
            path = prefix + (key,)
            yield path, child
            yield from iter_key_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_key_paths(child, prefix + (str(index),))


def retained_capability_scan() -> dict:
    listing = subprocess.check_output(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(REPOSITORY),
            "ls-tree",
            "-r",
            "-l",
            RESULTS_COMMIT,
            "--",
            "results/jobs",
        ],
        text=True,
        timeout=30,
    )
    rows: list[tuple[str, int]] = []
    pattern = re.compile(r"^\d+\s+\w+\s+[0-9a-f]+\s+(\d+)\t(.+)$")
    for line in listing.splitlines():
        match = pattern.match(line)
        require(match is not None, f"Unexpected ls-tree record: {line}")
        rows.append((match.group(2), int(match.group(1))))
    require(len(rows) == 370, "Tracked results/jobs path count changed")
    require(all(path.endswith(".json") for path, _ in rows), "Non-JSON job artifact appeared")
    require(sum(size for _, size in rows) == 1925346, "Tracked job JSON byte count changed")
    manifest_paths = [path for path, _ in rows if path.endswith("/publish_manifest.json")]
    payload_paths = [path for path, _ in rows if not path.endswith("/publish_manifest.json")]
    require(len(manifest_paths) == 184 and len(payload_paths) == 186,
            "Manifest/payload partition changed")
    require(not any("/raw/" in path for path, _ in rows), "Tracked raw path appeared")

    key_counts: dict[str, int] = {}
    object_center_paths: list[str] = []
    label_activity_values: list[int] = []
    embedded_cells: dict[str, str] = {}
    for path, _ in rows:
        payload = json.loads(git_bytes(path).decode("utf-8"))
        for key_path, child in iter_key_paths(payload):
            key = key_path[-1]
            key_counts[key] = key_counts.get(key, 0) + 1
            if key == "object_centers_robot_base_m":
                require(isinstance(child, dict), "Object-center field is not an object")
                object_center_paths.append(f"{path}:{'/'.join(key_path)}")
            if key in {"labels_created", "labels_created_by_job"}:
                require(type(child) is int, "Label activity field is not an integer")
                label_activity_values.append(child)
            if key in {"cell_receipt", "cell_receipt.json"} and isinstance(child, dict):
                schema = child.get("schema_version")
                cell_id = child.get("cell_id")
                if schema in {
                    "wmf-d1-behavioral-development-cell-v1",
                    "wmf-n3-behavioral-pilot-cell-v1",
                } and isinstance(cell_id, str):
                    prior = embedded_cells.setdefault(cell_id, schema)
                    require(prior == schema, f"Embedded cell schema changed: {cell_id}")
    require(len(object_center_paths) == 32, "Object-center occurrence count changed")
    require(all("/fresh_physical_checks/" in item for item in object_center_paths),
            "Object-center field escaped fresh physical checks")
    require(all(key_counts.get(key, 0) == 0 for key in ABSENT_SCIENCE_KEYS),
            "A previously absent science endpoint key appeared")
    require(len(label_activity_values) == 6 and set(label_activity_values) == {0},
            "Label activity fields changed")
    schema_counts: dict[str, int] = {}
    for schema in embedded_cells.values():
        schema_counts[schema] = schema_counts.get(schema, 0) + 1
    require(schema_counts == {
        "wmf-d1-behavioral-development-cell-v1": 4,
        "wmf-n3-behavioral-pilot-cell-v1": 3,
    }, "Unique complete embedded-cell inventory changed")
    return {
        "tracked_job_json_paths": len(rows),
        "tracked_job_json_bytes": sum(size for _, size in rows),
        "publish_manifests": len(manifest_paths),
        "publish_payloads": len(payload_paths),
        "tracked_raw_paths": 0,
        "tracked_media_or_binary_paths": 0,
        "object_center_occurrences": len(object_center_paths),
        "object_center_scope": "All occur beneath fresh_physical_checks; fixture/reset geometry only.",
        "absent_exact_endpoint_keys": list(ABSENT_SCIENCE_KEYS),
        "label_activity_fields": len(label_activity_values),
        "nonzero_label_activity_fields": sum(value != 0 for value in label_activity_values),
        "unique_complete_embedded_cells": len(embedded_cells),
        "unique_complete_embedded_cells_by_schema": schema_counts,
        "boundary": "Pinned results/jobs Git JSON only; no raw PVC, external, dynamic, future, untracked, LFS, or submodule content.",
    }


def audit() -> dict:
    values, sources = {}, {}
    for name, pin in PINNED_BLOBS.items():
        values[name], sources[name] = read_blob(*pin)
    snapshot = values["progress_snapshot"]
    terminal = values["terminal_aggregate"]
    require(snapshot.get("at_utc") == SNAPSHOT_AT_UTC, "Snapshot time changed")
    require(set(snapshot.get("n3_attempts", {})) == {ATTEMPT_ID},
            "Snapshot N3 attempt inventory changed")
    require(terminal.get("schema_version") == "wmf-n3-behavioral-pilot-job-v1"
            and terminal.get("status") == "passed"
            and terminal.get("phase") == "pilot"
            and terminal.get("model_config") == "N3"
            and terminal.get("layout_pair_id") == "P00"
            and terminal.get("queue_descriptor", {}).get("job_id") == ATTEMPT_ID,
            "Terminal aggregate identity changed")
    require(terminal.get("condition_order") == [
        "reflected-right", "reflected-left", "original-left", "original-right"
    ], "Terminal condition order changed")

    cells = snapshot["n3_attempts"][ATTEMPT_ID]["cells"]
    require(set(cells) == set(COMPLETE_CELL_KEYS) | {PARTIAL_CELL_KEY},
            "Snapshot cell inventory changed")
    terminal_refs = terminal["cell_receipts"]
    require(len(terminal_refs) == 4, "Terminal descriptor count changed")
    rows = []
    for key in COMPLETE_CELL_KEYS:
        wrapper = cells[key]
        cell = wrapper.get("cell_receipt.json")
        require(isinstance(cell, dict), f"Completed cell disappeared: {key}")
        encoded = canonical_bytes(cell)
        digest = hashlib.sha256(encoded).hexdigest()
        matches = [
            ref for ref in terminal_refs
            if ref.get("sha256") == digest and ref.get("bytes") == len(encoded)
        ]
        require(len(matches) == 1, f"Embedded cell lacks unique terminal match: {key}")
        raw_receipt = matches[0]
        require(raw_receipt["path"].endswith(f"/{ATTEMPT_ID}/cells/{key}/cell_receipt.json"),
                f"Raw cell path changed: {key}")
        require(cell.get("schema_version") == "wmf-n3-behavioral-pilot-cell-v1"
                and cell.get("status") == "passed"
                and cell.get("model_config") == "N3"
                and cell.get("layout_pair_id") == "P00"
                and cell.get("block_id") == terminal.get("block_id")
                and cell.get("cell_id") in terminal.get("cell_ids", [])
                and cell.get("cell_id") == f"wmf1__pilot__P00__N3__{cell['layout_arm']}__{cell['command']}"
                and key == f"{cell['condition_index']:02d}-wmf1-pilot-P00-N3-{cell['layout_arm']}-{cell['command']}",
                f"Embedded cell identity changed: {key}")
        require(cell.get("actions_executed") == 450
                and cell.get("observation_count") == 451
                and cell.get("behavioral_model_request_count") == 15
                and wrapper.get("actions_observed") == 450,
                f"Fixed-duration fields changed: {key}")
        events = wrapper.get("event_counts", {})
        require(events.get("environment_step_started") == 450
                and events.get("environment_step_completed") == 450
                and events.get("model_request_completed") == 15,
                f"Snapshot event counts changed: {key}")
        timing = cell.get("runner_timing", {})
        for name in ("env_step_s", "policy_inference_s", "video_write_s", "wall_total_s"):
            value = timing.get(name)
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                    f"Invalid timing field {name}: {key}")
        rows.append({
            "cell_id": cell["cell_id"],
            "phase": "pilot",
            "model_id": "N3",
            "layout_pair_id": "P00",
            "layout_arm": cell["layout_arm"],
            "command": cell["command"],
            "condition_index": cell["condition_index"],
            "attempt_id": ATTEMPT_ID,
            "actions_executed": cell["actions_executed"],
            "observation_count": cell["observation_count"],
            "behavioral_model_requests": cell["behavioral_model_request_count"],
            "runner_reported_timing_s": {
                name: timing[name]
                for name in ("env_step_s", "policy_inference_s", "video_write_s", "wall_total_s")
            },
            "timing_source_field": "cell_receipt.runner_timing",
            "timing_units": "seconds; unchanged individual values from the published receipt",
            "snapshot_source": "progress_snapshot",
            "snapshot_cell_key": key,
            "snapshot_at_utc": snapshot["at_utc"],
            "terminal_aggregate_source": "terminal_aggregate",
            "embedded_cell_bytes_match_terminal_descriptor": True,
            "raw_cell_receipt": raw_receipt,
            "raw_adapter_completion": cell["adapter_completion"],
            "raw_adapter_journal": cell["adapter_journal"],
        })
    require(len(rows) == len({row["cell_id"] for row in rows}) == 3,
            "Included subset identity count changed")

    partial = cells[PARTIAL_CELL_KEY]
    require(partial.get("cell_receipt.json") is None
            and partial.get("technical_failure.json") is None
            and partial.get("actions_observed") == 236,
            "Partial snapshot boundary changed")
    partial_events = partial.get("event_counts", {})
    require(partial_events.get("environment_step_started") == 236
            and partial_events.get("environment_step_completed") == 235
            and partial_events.get("model_request_completed") == 8,
            "Partial event counts changed")
    excluded_terminal = [
        ref for ref in terminal_refs if ref["path"].endswith(f"/{PARTIAL_CELL_KEY}/cell_receipt.json")
    ]
    require(len(excluded_terminal) == 1, "Later terminal descriptor for partial cell changed")

    return {
        "schema_version": "wmf-retained-n3-pilot-operational-subset-audit-v1",
        "results_commit": RESULTS_COMMIT,
        "selection": {
            "included_cells": 3,
            "terminal_pilot_cells": 4,
            "rule": "Opportunistic subset selected only by complete cell receipts embedded in one pinned detached progress snapshot, with exact byte/hash matches to the final aggregate.",
            "representative_sample": False,
        },
        "verification_level": "Two pinned compact Git blobs and three reconstructed embedded cell receipt byte strings; no live process or raw-PVC revalidation.",
        "embedded_receipt_reconstruction": "UTF-8 JSON with sorted keys, indent=2, allow_nan=False and one terminal newline; byte count and SHA256 must equal a unique final aggregate descriptor.",
        "sources": sources,
        "cells": rows,
        "excluded_partial_cells": [{
            "snapshot_source": "progress_snapshot",
            "snapshot_cell_key": PARTIAL_CELL_KEY,
            "snapshot_actions_observed": partial["actions_observed"],
            "snapshot_environment_steps_started": partial_events["environment_step_started"],
            "snapshot_environment_steps_completed": partial_events["environment_step_completed"],
            "snapshot_model_requests_completed": partial_events["model_request_completed"],
            "later_terminal_descriptor": excluded_terminal[0],
            "reason": "No completed cell receipt existed in the pinned progress snapshot. The later aggregate classifies the four-cell block, but this cell is excluded from the three-receipt embedded subset.",
        }],
        "retained_capability_scan": retained_capability_scan(),
        "claim_boundaries": {
            "snapshot": "Detached read-only progress snapshot; not itself a terminal scientific classification.",
            "timing": "Individual runner-reported timing components only. No mean, quantile, full-cohort resource budget, resource qualification, or between-model timing comparison is computed.",
            "stopping": "Event-count first_success values are not analyzed. No first-success action index or first-success/action-450 cube-bowl coordinates are present; this is not A5.",
            "science": "No A1-A9 estimate, forecast error, persistence error, skill, movement decomposition, cohort success rate, annotation label, or release authority follows from this operational subset.",
            "scope": "The broader capability scan is exact only for tracked JSON under results/jobs at the pinned results commit.",
        },
        "new_behavioral_cells": 0,
        "new_model_requests": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "resource_budget_qualified": False,
        "a5_stopping_contrast_estimated": False,
        "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", type=Path,
        help="Verify an existing canonical audit without modifying it",
    )
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded, "Saved operational subset audit differs")
        print(
            "Verified 2 pinned blobs, 3 byte-matched retained N3 pilot cells, "
            "and the pinned results/jobs capability boundary; operational subset only."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
