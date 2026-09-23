"""Prospective MAIN N3/LAT P fixture assignment without creating a release."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .contract import canonical_bytes, sha256_file


SCHEMA = "sgw-01-main-n3-lat-p-fixture-assignment-v1"
FIXTURE_ID = "SGW-ENG-008-LAT-057"
SOURCE_COMMIT = "4f04e31ee0219dae052bba2ef95b9e2010dbceeb"


def build_assignment(
    *, planned_queue: Path, translated_capture: Path, translated_capture_sha256: str,
    qualification_verification: Path, qualification_verification_sha256: str,
) -> dict[str, Any]:
    """Bind the qualified redesigned fixture to the pre-existing MAIN P cells.

    This records an engineering selection after scripted qualification.  It is
    deliberately neither a runtime release nor an authorization to execute a
    learned outcome.
    """

    rows = _main_p_rows(planned_queue)
    capture = _bound_file(translated_capture, translated_capture_sha256, "translated capture")
    verification = _bound_file(
        qualification_verification, qualification_verification_sha256, "qualification verification",
    )
    _verify_fixture_evidence(verification["json"])
    value = {
        "schema_version": SCHEMA,
        "assignment_id": "SGW-MAIN-N3-LAT-P-20260923",
        "authority": (
            "User-confirmed MAIN P->D->C priority. This assigns only the existing "
            "six-cell N3/LAT P queue to a separately disclosed redesigned fixture."
        ),
        "source_commit": SOURCE_COMMIT,
        "candidate_id": FIXTURE_ID,
        "selection_timing": "after_six_scripted_passes_before_any_learned_outcome",
        "fixture_evidence": {
            "translated_capture": capture["record"],
            "qualification_verification": verification["record"],
            "required_physical_criteria": {
                "verified_all_six_scripted_passes": True,
                "same_reset_tolerance_and_physical_criteria": True,
                "model_requests": 0,
                "behavioral_episodes": 0,
            },
        },
        "main_p_frozen_queue": {
            "source_queue_sha256": sha256_file(planned_queue),
            "cells": rows,
            "preserved_cell_ids_prompts_seeds_and_order": True,
            "existing_episode_allocation": len(rows),
        },
        "duplicate_reservation": {
            "reserved_for_main_p_only": True,
            "later_d_c_duplicate_within_tolerance_reuse_forbidden": True,
            "later_d_c_selection_counterbalance_and_full_scope": "unreleased",
        },
        "exclusions": {
            "rejected_lat_candidate_057": True,
            "old_blocked_pool": True,
            "historical_novelty_gate": "not_required_for_this_distinct_prospective_assignment",
            "old_generator_and_outcomes": "immutable",
        },
        "runtime_release": {
            "release_permitted": False,
            "pending": [
                "passed_native_fixed_input_receipt",
                "passed_runtime_receipt",
                "passed_time_map_receipt",
            ],
        },
        "model_requests": 0,
        "behavioral_episodes": 0,
    }
    value["assignment_sha256"] = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return value


def write_assignment(*, output: Path, **kwargs: Any) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite prospective P fixture assignment")
    value = build_assignment(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_bytes(value))
    return value


def materialize_release_fixture(
    *, assignment_path: Path, qualified_fixture_input: Path, output: Path,
) -> dict[str, Any]:
    """Bind the P fixture hash into an already-qualified release fixture receipt.

    A caller must separately provide the runtime/time-map-qualified receipt.
    This function only adds the selected physical-layout binding and never
    changes `release_permitted` or creates a runtime authorization.
    """

    if output.exists():
        raise FileExistsError("refusing to overwrite release fixture materialization")
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    if (
        assignment.get("schema_version") != SCHEMA
        or assignment.get("assignment_sha256") != hashlib.sha256(
            canonical_bytes({key: value for key, value in assignment.items() if key != "assignment_sha256"})
        ).hexdigest()
        or assignment.get("runtime_release", {}).get("release_permitted") is not False
    ):
        raise ValueError("P fixture assignment is malformed or has been promoted")
    source = json.loads(qualified_fixture_input.read_text(encoding="utf-8"))
    if source.get("status") != "qualified" or not isinstance(source.get("layouts"), dict) or not isinstance(source.get("time_maps"), dict):
        raise ValueError("release fixture input must already be qualified with time maps")
    if "N3" not in source["time_maps"]:
        raise ValueError("release fixture input lacks the independently qualified N3 time map")
    value = dict(source)
    value["layouts"] = dict(source["layouts"])
    if "LAT-P01" in value["layouts"]:
        raise ValueError("refusing to replace an existing LAT-P01 fixture binding")
    value["layouts"]["LAT-P01"] = {
        "fixture_sha256": assignment["assignment_sha256"],
        "source_assignment": {"path": str(assignment_path.resolve()), "sha256": sha256_file(assignment_path)},
        "candidate_id": FIXTURE_ID,
        "release_permitted": False,
        "claim_boundary": "physical fixture binding only; runtime release remains independently pending",
    }
    value["main_p_fixture_materialization"] = {
        "assignment_sha256": assignment["assignment_sha256"],
        "model_requests": 0, "behavioral_episodes": 0, "release_permitted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(canonical_bytes(value))
    return value


def _main_p_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [
            row for row in csv.DictReader(stream)
            if (row["model"], row["family"], row["stage"]) == ("N3", "LAT", "P")
        ]
    if len(rows) != 6:
        raise ValueError("frozen MAIN N3/LAT P queue must contain exactly six rows")
    expected = [("I", "1"), ("D", "1"), ("C", "1"), ("C", "-1"), ("D", "-1"), ("I", "-1")]
    if (
        [(row["form"], row["physical_goal_sign"]) for row in rows] != expected
        or any(row["status"] != "PLANNED_NOT_RELEASED" for row in rows)
        or any(row["layout_id"] != "LAT-P01" for row in rows)
    ):
        raise ValueError("frozen MAIN P identifiers/order/status differ")
    return [
        {key: row[key] for key in (
            "cell_id", "block_id", "layout_id", "within_block_order", "form", "physical_goal_sign",
            "prompt_id", "prompt", "prompt_sha256", "environment_seed", "effective_policy_seed", "action_cap",
        )}
        for row in rows
    ]


def _bound_file(path: Path, digest: str, label: str) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label} bytes differ from its disclosed binding")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not JSON") from error
    return {"json": value, "record": {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}}


def _verify_fixture_evidence(value: Mapping[str, Any]) -> None:
    if (
        value.get("candidate_id") != FIXTURE_ID
        or value.get("status") != "verified_all_six_pass"
        or value.get("passed_checks") != 6
        or value.get("model_requests") != 0
        or value.get("behavioral_episodes") != 0
    ):
        raise ValueError("LAT-057 verification does not meet all six physical fixture criteria")
