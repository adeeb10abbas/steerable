#!/usr/bin/env python3
"""Audit two distinct retained success-summary channels without promoting gates."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


RESULTS_COMMIT = "12839900e5c46299e11fff4813e8602949055b86"
REPOSITORY = Path(__file__).resolve().parents[3]
RUNNER_RESULT_REPR = "[{'env_id': 0, 'success': False, 'step': 450}]"
SNAPSHOT_BOUNDARY = (
    "Read-only progress snapshot; not a terminal receipt or scientific classification."
)
PINNED_BLOBS = {
    "N3_snapshot": (
        "results/jobs/detached-attempt-snapshot-003/publish/diagnostic.json",
        173811,
        "57f2a346a1bc181fe0901cd7193fd5bbf952a68aa1210b35b70cfedef6a1394d",
    ),
    "N3_terminal": (
        "results/jobs/n3-p00-pilot-004/publish/n3_behavioral_pilot_receipt.json",
        6473,
        "5a7a861acc82615b4eb7aa92308f40f7a0c965e65b6edaf12f776cb64b18b0e0",
    ),
    "D01_snapshot": (
        "results/jobs/d1-development-d01-snapshot-001/publish/diagnostic.json",
        191154,
        "4f3563f92b28e3dcf713ccdf45df43aab745fb2a91562751f78ae5b1133a6883",
    ),
    "D01_terminal": (
        "results/jobs/d1-development-d01-simulator-003/publish/d1_behavioral_development_receipt.json",
        16510,
        "9b1fd7d7479ad275d4baa943b9d8d6a4ed1d223e2d9ca20b3bff0342711ec5eb",
    ),
    "D02_snapshot": (
        "results/jobs/d1-development-d02-snapshot-001/publish/diagnostic.json",
        191132,
        "108f1d7dd1f85c8c3af6b8cbff42a39ba06521aa72bb2e4f19b8bbfcaf6c727e",
    ),
    "D02_terminal": (
        "results/jobs/d1-development-d02-simulator-003/publish/d1_behavioral_development_receipt.json",
        16510,
        "d1dc66c57d36cb3e620731c69a6e41fdf52c3cc2a797c2dae7ee36d63d10c6c9",
    ),
}
SOURCE_COMMITS = {
    "N3": "b8914413f6293c165dd3a57c737f4b977e7e84be",
    "D1": "25ff299ca0d2b9964eb48286990ee2301cb207b8",
}
SOURCE_FILES = {
    "recording_adapter": {
        "path": "workshops/corl2026_world_models/experiments/forecast_layout/recording_adapter.py",
        "git_blob": "5470b0a5b2915414b5e6a972014e601d29c61124",
        "bytes": 66356,
        "sha256": "7bf3ebc451e29d647923e5047a7503e235256b225d174654cf658ec980c08d33",
    },
    "fixture_tasks": {
        "path": "workshops/corl2026_world_models/experiments/forecast_layout/fixture_tasks.py",
        "git_blob": "9d05d1f0fbc649b2529d5bcc6ab920e585cf3a47",
        "bytes": 16143,
        "sha256": "e3891120e73bd0cd313097b0c6c6d0385cdfa4f240709a3320bbbf2716f8e5ba",
    },
}
N3_ATTEMPT_ID = "n3-p00-pilot-004"
N3_CELL_KEYS = (
    "00-wmf1-pilot-P00-N3-reflected-right",
    "01-wmf1-pilot-P00-N3-reflected-left",
    "02-wmf1-pilot-P00-N3-original-left",
)
D1_CELL_KEYS = {
    "D01": (
        "00-wmf1-development-D01-D1-original-left",
        "01-wmf1-development-D01-D1-reflected-right",
    ),
    "D02": (
        "00-wmf1-development-D02-D1-reflected-right",
        "01-wmf1-development-D02-D1-original-left",
    ),
}
EXPECTED_FIRST_SUCCESS = {
    "wmf1__pilot__P00__N3__reflected__right": 1,
    "wmf1__pilot__P00__N3__reflected__left": 1,
    "wmf1__pilot__P00__N3__original__left": 1,
    "wmf1__development__D01__D1__original__left": 0,
    "wmf1__development__D01__D1__reflected__right": 1,
    "wmf1__development__D02__D1__reflected__right": 1,
    "wmf1__development__D02__D1__original__left": 1,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def strict_json_loads(payload: bytes, source: str) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        value: dict[str, object] = {}
        for key, child in pairs:
            require(key not in value, f"Duplicate JSON key in {source}: {key}")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value in {source}: {value}")

    decoded = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    require(isinstance(decoded, dict), f"Pinned JSON root is not an object: {source}")
    return decoded


def git_bytes(commit: str, path: str) -> bytes:
    return subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "show", f"{commit}:{path}",
        ],
        timeout=30,
    )


def git_object_id(commit: str, path: str) -> str:
    return subprocess.check_output(
        [
            "git", "--no-replace-objects", "-C", str(REPOSITORY),
            "rev-parse", f"{commit}:{path}",
        ],
        text=True,
        timeout=30,
    ).strip()


def read_blob(path: str, size: int, digest: str) -> tuple[dict, dict]:
    payload = git_bytes(RESULTS_COMMIT, path)
    require(
        len(payload) == size and hashlib.sha256(payload).hexdigest() == digest,
        f"Pinned published blob differs: {path}",
    )
    return strict_json_loads(payload, path), {
        "git_path": path,
        "bytes": size,
        "sha256": digest,
    }


def verify_source_semantics() -> dict:
    verified_files: dict[str, dict] = {}
    source_text: dict[str, str] = {}
    for source_name, descriptor in SOURCE_FILES.items():
        observations = []
        for model_id, commit in SOURCE_COMMITS.items():
            path = descriptor["path"]
            payload = git_bytes(commit, path)
            object_id = git_object_id(commit, path)
            require(object_id == descriptor["git_blob"],
                    f"Pinned source Git blob differs: {model_id}/{source_name}")
            require(
                len(payload) == descriptor["bytes"]
                and hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
                f"Pinned source bytes differ: {model_id}/{source_name}",
            )
            observations.append({
                "model_id": model_id,
                "source_commit": commit,
                "git_blob": object_id,
            })
            source_text.setdefault(source_name, payload.decode("utf-8"))
        verified_files[source_name] = {
            "git_path": descriptor["path"],
            "bytes": descriptor["bytes"],
            "sha256": descriptor["sha256"],
            "identical_at_source_commits": observations,
        }

    adapter = source_text["recording_adapter"]
    fixture = source_text["fixture_tasks"]
    for fragment in (
        'target = self.identity["command"]',
        "if success[target] and self.first_success is None:",
        '"action_step": action_step,',
        'self._append_event("first_success", self.first_success)',
        '"success_configured_as_termination": False,',
    ):
        require(fragment in adapter, f"Pinned first-success source fragment absent: {fragment}")
    for fragment in (
        "class ForecastLayoutTimeoutOnlyTermination:",
        "time_out = DoneTerm(func=mdp.time_out, time_out=True)",
        "def success_measurements(env: Any) -> dict[str, bool]:",
        '"frame_of_reference": "robot",',
        '"mirrored": False,',
        '"require_gripper_detached": True,',
        '"left": bool(object_left_of(env, **common)),',
        '"right": bool(object_right_of(env, **common)),',
        "wmf_success_sampler = staticmethod(success_measurements)",
    ):
        require(fragment in fixture, f"Pinned relation source fragment absent: {fragment}")

    return {
        "source_commits": SOURCE_COMMITS,
        "verified_files": verified_files,
        "adapter_event_semantics": (
            "The pinned adapter emits first_success once, when the requested relation "
            "predicate is first sampled true and no earlier first_success was recorded."
        ),
        "requested_relation_semantics": (
            "The pinned fixture samples object_left_of/object_right_of relative to the "
            "robot with mirrored=False and require_gripper_detached=True."
        ),
    }


def match_terminal_receipt(cell: dict, terminal: dict, expected_suffix: str) -> dict:
    encoded = canonical_bytes(cell)
    digest = hashlib.sha256(encoded).hexdigest()
    matches = [
        descriptor for descriptor in terminal["cell_receipts"]
        if descriptor.get("bytes") == len(encoded)
        and descriptor.get("sha256") == digest
    ]
    require(len(matches) == 1, f"Embedded receipt lacks unique terminal match: {expected_suffix}")
    require(matches[0]["path"].endswith(expected_suffix),
            f"Terminal receipt path differs: {expected_suffix}")
    return matches[0]


def validate_row(
    *,
    cell: dict,
    event_counts: dict,
    event_source_path: str,
    last_event_kind: str,
    snapshot_source: str,
    snapshot_cell_key: str,
    snapshot_at_utc: str,
    terminal_source: str,
    terminal: dict,
    attempt_id: str,
    phase: str,
    model_id: str,
    layout_pair_id: str,
    expected_requests: int,
) -> dict:
    require(type(event_counts) is dict and event_counts,
            f"Snapshot journal counts absent: {snapshot_cell_key}")
    require(all(type(value) is int and value >= 0 for value in event_counts.values()),
            f"Snapshot journal count invalid: {snapshot_cell_key}")
    require(event_counts.get("attempt_finalized") == 1
            and last_event_kind == "attempt_finalized",
            f"Snapshot journal is not finalized: {snapshot_cell_key}")
    require(cell["schema_version"] in {
        "wmf-n3-behavioral-pilot-cell-v1",
        "wmf-d1-behavioral-development-cell-v1",
    } and cell["status"] == "passed"
            and cell["model_config"] == model_id
            and cell["layout_pair_id"] == layout_pair_id,
            f"Embedded cell identity differs: {snapshot_cell_key}")
    if model_id == "N3":
        require("phase" not in cell
                and cell["schema_version"] == "wmf-n3-behavioral-pilot-cell-v1",
                f"N3 embedded cell schema differs: {snapshot_cell_key}")
    else:
        require(cell["phase"] == phase
                and cell["schema_version"] == "wmf-d1-behavioral-development-cell-v1"
                and cell["source_pins"]["study_commit"] == SOURCE_COMMITS["D1"],
                f"D1 embedded cell schema/source differs: {snapshot_cell_key}")
    require(cell["actions_executed"] == 450
            and cell["observation_count"] == 451
            and cell["behavioral_model_request_count"] == expected_requests,
            f"Fixed-duration cell fields differ: {snapshot_cell_key}")
    require(event_counts.get("environment_step_started") == 450
            and event_counts.get("environment_step_completed") == 450
            and event_counts.get("observation_captured") == 451
            and event_counts.get("policy_action_returned") == 450
            and event_counts.get("model_request_packed") == expected_requests
            and event_counts.get("model_request_sent") == expected_requests
            and event_counts.get("model_response_received") == expected_requests
            and event_counts.get("model_request_completed") == expected_requests,
            f"Snapshot event counts differ: {snapshot_cell_key}")
    journal_total = sum(event_counts.values())
    require(cell["adapter_journal"]["event_count"] == journal_total,
            f"Snapshot counts do not cover the terminal-matched journal: {snapshot_cell_key}")
    require(cell["runner_environment_result_repr"] == RUNNER_RESULT_REPR,
            f"Opaque runner terminal summary differs: {snapshot_cell_key}")
    parsed_runner_summary = ast.literal_eval(cell["runner_environment_result_repr"])
    require(type(parsed_runner_summary) is list and len(parsed_runner_summary) == 1
            and type(parsed_runner_summary[0]) is dict
            and set(parsed_runner_summary[0]) == {"env_id", "success", "step"}
            and type(parsed_runner_summary[0]["env_id"]) is int
            and parsed_runner_summary[0]["env_id"] == 0
            and type(parsed_runner_summary[0]["success"]) is bool
            and parsed_runner_summary[0]["success"] is False
            and type(parsed_runner_summary[0]["step"]) is int
            and parsed_runner_summary[0]["step"] == 450,
            f"Opaque runner terminal summary structure differs: {snapshot_cell_key}")
    first_success_count = event_counts.get("first_success", 0)
    require(first_success_count in (0, 1)
            and first_success_count == EXPECTED_FIRST_SUCCESS[cell["cell_id"]],
            f"First-success event count differs: {snapshot_cell_key}")
    raw_receipt = match_terminal_receipt(
        cell,
        terminal,
        f"/{attempt_id}/cells/{snapshot_cell_key}/cell_receipt.json",
    )
    return {
        "cell_id": cell["cell_id"],
        "phase": phase,
        "model_id": model_id,
        "layout_pair_id": layout_pair_id,
        "layout_arm": cell["layout_arm"],
        "command": cell["command"],
        "condition_index": cell["condition_index"],
        "attempt_id": attempt_id,
        "actions_executed": cell["actions_executed"],
        "observation_count": cell["observation_count"],
        "behavioral_model_requests": cell["behavioral_model_request_count"],
        "snapshot_source": snapshot_source,
        "snapshot_cell_key": snapshot_cell_key,
        "snapshot_at_utc": snapshot_at_utc,
        "terminal_aggregate_source": terminal_source,
        "embedded_cell_bytes_match_terminal_descriptor": True,
        "raw_cell_receipt": raw_receipt,
        "adapter_event_channel": {
            "source_field": event_source_path,
            "first_success_count": first_success_count,
            "retained_event_count_total": journal_total,
            "last_event_kind": last_event_kind,
        },
        "runner_terminal_summary_channel": {
            "source_field": "cell_receipt.runner_environment_result_repr",
            "raw_repr": cell["runner_environment_result_repr"],
            "interpretation_boundary": (
                "Preserved as an opaque runner-generated representation; it is not "
                "treated as the adapter event history or an action-450 goal predicate."
            ),
        },
        "raw_adapter_journal": cell["adapter_journal"],
    }


def audit() -> dict:
    values: dict[str, dict] = {}
    sources: dict[str, dict] = {}
    for name, pin in PINNED_BLOBS.items():
        values[name], sources[name] = read_blob(*pin)

    n3_snapshot = values["N3_snapshot"]
    n3_terminal = values["N3_terminal"]
    require(n3_terminal["source_commit"] == SOURCE_COMMITS["N3"]
            and n3_terminal["schema_version"] == "wmf-n3-behavioral-pilot-job-v1"
            and n3_terminal["status"] == "passed"
            and n3_terminal["phase"] == "pilot"
            and n3_terminal["model_config"] == "N3"
            and n3_terminal["layout_pair_id"] == "P00"
            and n3_terminal["queue_descriptor"]["job_id"] == N3_ATTEMPT_ID,
            "N3 terminal aggregate identity differs")
    require(n3_snapshot["n3_attempts"][N3_ATTEMPT_ID]["server_ready"]["source_commit"]
            == SOURCE_COMMITS["N3"], "N3 snapshot source pin differs")
    require(set(n3_snapshot["n3_attempts"]) == {N3_ATTEMPT_ID},
            "N3 snapshot attempt inventory differs")
    n3_cells = n3_snapshot["n3_attempts"][N3_ATTEMPT_ID]["cells"]
    require({key for key, wrapper in n3_cells.items()
             if isinstance(wrapper.get("cell_receipt.json"), dict)} == set(N3_CELL_KEYS),
            "N3 completed embedded-receipt selection differs")
    rows = []
    for key in N3_CELL_KEYS:
        wrapper = n3_cells[key]
        require(wrapper["cell_receipt.json"] is not None
                and wrapper["technical_failure.json"] is None
                and wrapper["actions_observed"] == 450,
                f"N3 complete snapshot cell differs: {key}")
        rows.append(validate_row(
            cell=wrapper["cell_receipt.json"],
            event_counts=wrapper["event_counts"],
            event_source_path=(
                f"n3_attempts.{N3_ATTEMPT_ID}.cells.{key}."
                "event_counts.first_success"
            ),
            last_event_kind=wrapper["last_event_kind"],
            snapshot_source="N3_snapshot",
            snapshot_cell_key=key,
            snapshot_at_utc=n3_snapshot["at_utc"],
            terminal_source="N3_terminal",
            terminal=n3_terminal,
            attempt_id=N3_ATTEMPT_ID,
            phase="pilot",
            model_id="N3",
            layout_pair_id="P00",
            expected_requests=15,
        ))

    partial_rows = []
    n3_partial_key = "03-wmf1-pilot-P00-N3-original-right"
    n3_partial = n3_cells[n3_partial_key]
    require(n3_partial["cell_receipt.json"] is None
            and n3_partial["technical_failure.json"] is None
            and n3_partial["last_event_kind"] == "environment_step_started"
            and n3_partial["actions_observed"] == 236
            and n3_partial["event_counts"]["environment_step_started"] == 236
            and n3_partial["event_counts"]["environment_step_completed"] == 235
            and n3_partial["event_counts"].get("attempt_finalized", 0) == 0,
            "N3 partial-prefix exclusion boundary differs")
    partial_rows.append({
        "snapshot_source": "N3_snapshot",
        "snapshot_cell_key": n3_partial_key,
        "completed_actions": 235,
        "last_event_kind": n3_partial["last_event_kind"],
        "reason": (
            "No embedded completed receipt or attempt_finalized event exists at the "
            "pinned snapshot boundary; a later terminal descriptor is not joined to "
            "this partial prefix."
        ),
    })

    d1_partial_keys = {
        "D01": "02-wmf1-development-D01-D1-original-right",
        "D02": "02-wmf1-development-D02-D1-reflected-left",
    }
    d1_partial_completed = {"D01": 128, "D02": 208}
    for pair in ("D01", "D02"):
        snapshot = values[f"{pair}_snapshot"]
        terminal = values[f"{pair}_terminal"]
        attempt_id = f"d1-development-{pair.lower()}-simulator-003"
        require(snapshot["schema_version"] == "wmf-d1-development-readonly-snapshot-v1"
                and snapshot["layout_pair_id"] == pair
                and snapshot["science_claim_boundary"] == SNAPSHOT_BOUNDARY,
                f"D1 snapshot identity differs: {pair}")
        require(terminal["study_commit"] == SOURCE_COMMITS["D1"]
                and terminal["schema_version"] == "wmf-d1-behavioral-development-simulator-job-v1"
                and terminal["status"] == "passed"
                and terminal["phase"] == "development"
                and terminal["model_config"] == "D1"
                and terminal["layout_pair_id"] == pair
                and terminal["run_id"] == snapshot["run_id"]
                and terminal["queue_descriptor"]["job_id"] == attempt_id
                and terminal["simulator_job_id"] == attempt_id,
                f"D1 terminal aggregate identity differs: {pair}")
        cells = snapshot["simulator"]["cells"]
        require({key for key, wrapper in cells.items()
                 if isinstance(wrapper.get("cell_receipt"), dict)}
                == set(D1_CELL_KEYS[pair]),
                f"D1 completed embedded-receipt selection differs: {pair}")
        for key in D1_CELL_KEYS[pair]:
            wrapper = cells[key]
            require(wrapper["cell_receipt"] is not None
                    and wrapper["technical_failure"] is None
                    and wrapper["journal"]["exists"] is True
                    and wrapper["journal"]["max_action_step"] == 450
                    and wrapper["journal"]["max_request_index"] == 56,
                    f"D1 complete snapshot cell differs: {key}")
            rows.append(validate_row(
                cell=wrapper["cell_receipt"],
                event_counts=wrapper["journal"]["event_counts"],
                event_source_path=(
                    f"simulator.cells.{key}.journal.event_counts.first_success"
                ),
                last_event_kind=wrapper["journal"]["last_kind"],
                snapshot_source=f"{pair}_snapshot",
                snapshot_cell_key=key,
                snapshot_at_utc=snapshot["at_utc"],
                terminal_source=f"{pair}_terminal",
                terminal=terminal,
                attempt_id=attempt_id,
                phase="development",
                model_id="D1",
                layout_pair_id=pair,
                expected_requests=57,
            ))
        partial_key = d1_partial_keys[pair]
        partial = cells[partial_key]
        completed = d1_partial_completed[pair]
        require(partial["cell_receipt"] is None
                and partial["technical_failure"] is None
                and partial["journal"]["exists"] is True
                and partial["journal"]["last_kind"] == "model_request_sent"
                and partial["journal"]["max_action_step"] == completed
                and partial["journal"]["event_counts"]["environment_step_started"] == completed
                and partial["journal"]["event_counts"]["environment_step_completed"] == completed
                and partial["journal"]["event_counts"].get("attempt_finalized", 0) == 0,
                f"D1 partial-prefix exclusion boundary differs: {pair}")
        partial_rows.append({
            "snapshot_source": f"{pair}_snapshot",
            "snapshot_cell_key": partial_key,
            "completed_actions": completed,
            "last_event_kind": partial["journal"]["last_kind"],
            "reason": (
                "No embedded completed receipt or attempt_finalized event exists at "
                "the pinned snapshot boundary."
            ),
        })

    require(len(rows) == len({row["cell_id"] for row in rows}) == 7,
            "Retained diagnostic row identity count differs")
    first_success_rows = sum(
        row["adapter_event_channel"]["first_success_count"] == 1 for row in rows
    )
    zero_first_success_rows = sum(
        row["adapter_event_channel"]["first_success_count"] == 0 for row in rows
    )
    false_runner_repr_rows = sum(
        row["runner_terminal_summary_channel"]["raw_repr"] == RUNNER_RESULT_REPR
        for row in rows
    )
    require((first_success_rows, zero_first_success_rows, false_runner_repr_rows)
            == (6, 1, 7), "Retained success-channel inventory differs")

    return {
        "schema_version": "wmf-retained-success-channel-diagnostic-v1",
        "results_commit": RESULTS_COMMIT,
        "selection": {
            "included_cells": 7,
            "included_d1_development_cells": 4,
            "included_n3_pilot_cells": 3,
            "rule": (
                "Availability-selected cells with complete receipts embedded in pinned "
                "compact snapshots and exact byte/hash matches to terminal aggregates."
            ),
            "representative_sample": False,
        },
        "verification_level": (
            "Six pinned compact Git blobs, seven reconstructed embedded cell receipt "
            "byte strings, and two exact source blobs at both recorded source commits; "
            "no live process or raw-PVC revalidation."
        ),
        "embedded_receipt_reconstruction": (
            "UTF-8 JSON with sorted keys, indent=2, allow_nan=False and one terminal "
            "newline; byte count and SHA256 equal a unique terminal descriptor."
        ),
        "sources": sources,
        "source_semantics": verify_source_semantics(),
        "cells": rows,
        "excluded_partial_prefixes": partial_rows,
        "inventory_counts": {
            "availability_selected_rows": 7,
            "rows_with_adapter_first_success_count_one": first_success_rows,
            "rows_with_adapter_first_success_count_zero": zero_first_success_rows,
            "rows_with_exact_false_runner_terminal_repr": false_runner_repr_rows,
        },
        "finding": (
            "For these seven availability-selected retained rows, the adapter "
            "first-success event channel and the opaque runner terminal-summary channel "
            "are not interchangeable: six rows retain one first_success event while all "
            "seven retain the same runner representation containing success=False at "
            "step 450. The runner representation must not substitute for adapter events."
        ),
        "claim_boundaries": {
            "channel_scope": (
                "The adapter event means the requested relation predicate was sampled "
                "true at least once under the pinned measurement implementation. The "
                "runner string remains opaque and is not established as semantically "
                "equivalent to the adapter event channel."
            ),
            "no_persistence_or_endpoint_claim": (
                "No first-success action index, endpoint coordinates, or action-450 goal "
                "predicate is retained here. No physical nonpersistence or action-450 "
                "failure is inferred from the two channels."
            ),
            "no_rate_or_comparison": (
                "These inventory counts are not a success rate, representative sample, "
                "full-cohort estimate, or between-model comparison."
            ),
            "science_and_release": (
                "No A1-A9 estimate, A5 stopping contrast, forecast error, skill, labels, "
                "current GM/PVC state, gate promotion, request closure, confirmation "
                "release, dispatch authority, or new behavioral evidence follows."
            ),
        },
        "new_behavioral_cells": 0,
        "new_model_requests": 0,
        "labels_created": 0,
        "forecast_accuracy_estimated": False,
        "a5_stopping_contrast_estimated": False,
        "confirmation_released": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", type=Path,
        help="Verify an existing canonical diagnostic without modifying it",
    )
    args = parser.parse_args()
    encoded = canonical_bytes(audit())
    if args.check:
        require(args.check.read_bytes() == encoded,
                "Saved retained success-channel diagnostic differs")
        print(
            "Verified 6 pinned results blobs, 7 terminal-matched retained cells, "
            "and shared pinned source semantics; diagnostic only, no gate promotion."
        )
    else:
        print(encoded.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
