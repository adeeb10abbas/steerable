#!/usr/bin/env python3
"""Pinned H01 requested-region entry analysis from retained robot-frame XYZ.

This is a retrospective analysis of 108 already-recorded H01 episodes.  It
does not execute a model, replay an episode, read an old absolute artifact
path, or use media.  Source bytes are read from authenticated Git blobs at the
fixed evidence commit.  The requested 45-degree cone and both first-entry
definitions are rederived from every retained object/reference XYZ sample.

The primary endpoint is the first observed requested-cone entry action index.
The four episodes with no such entry all reach the administrative action cap;
they remain right-censored at 450.  A value of 450 in the restricted waiting
summary is their RMST-style contribution, not an imputed entry event.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
from typing import Any, Iterable, Mapping


PIN = "ce561e66f82e95055e39d3d7711691982f6b2086"
PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
DEFAULT_JSON = PACKAGE / "paper" / "data" / "h01_requested_region_entry.json"
DEFAULT_CSV = PACKAGE / "paper" / "data" / "h01_requested_region_entry.csv"

EPISODES = (
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/"
    "nano_v3b001_episodes.jsonl"
)
COMPILE_CELL = "experiments/v3/cosmos_nano_phase_b/compile_cell.py"
EPISODE_SCHEMA = "tools/vla_wam_v3_episode_schema.py"
EXPECTED_SOURCES: dict[str, dict[str, Any]] = {
    EPISODES: {
        "git_blob": "5c2f4a6fa4795ec054dbce2d4d68554245ceb4cb",
        "sha256": "ce67aecb2705394e8161fa4b65df7130e62648e2f6d8b6a93408c8da685a7207",
        "bytes": 5_822_218,
        "role": "108 retained H01 behavioral episodes and step XYZ traces",
    },
    COMPILE_CELL: {
        "git_blob": "f47ec3cae608a6d983b9b37641f7af6193421964",
        "sha256": "468bb9231316c4670a2dc3388ab7eb6f2ef5f42afc8fb04b16faca7b823f2a75",
        "bytes": 18_316,
        "role": "SUSTAINED_STEPS, _first_sustained, _in_cone, and event construction",
    },
    EPISODE_SCHEMA: {
        "git_blob": "a2f7c0b79e5ef82a49723842248969182aab4d60",
        "sha256": "f71c1ab5e499a118e074b304e6b8aef84f8beb01ac3e670692cb2597efe0197c",
        "bytes": 37_213,
        "role": "independent retained-record derivation and validation contract",
    },
}

FRAME = "robot_base_object_minus_reference_xyz_m"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
SIDES = ("left", "right")
CONDITION_ORDER = tuple((arm, side) for arm in ARMS for side in SIDES)
ACTION_CAP = 450
SUSTAINED_STEPS = 3
EXPECTED_EPISODES = len(SEEDS) * len(CONDITION_ORDER)
EXPECTED_ACTION_SAMPLES = 22_080
COS_45 = math.cos(math.radians(45.0))


def _git(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def parse_json_line(raw: bytes, *, line_number: int = 1) -> dict[str, Any]:
    """Parse one strict JSON object, rejecting duplicate keys and NaN/Infinity."""
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid strict JSON on source line {line_number}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"source line {line_number} must be a JSON object")
    return value


def read_pinned_sources(root: Path = ROOT) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Authenticate exact source objects, then parse the episode JSONL bytes."""
    root = Path(root)
    commit = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if commit != PIN:
        raise ValueError("resolved evidence commit does not equal the fixed pin")

    raw_by_path: dict[str, bytes] = {}
    files: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", PIN + ":" + path).decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob mismatch for {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        framed_git_id = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        if framed_git_id != object_id:
            raise ValueError(f"recomputed Git blob identity mismatch for {path}")
        observed = {
            "git_blob": object_id,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "role": expected["role"],
        }
        if observed != expected:
            raise ValueError(f"pinned source bytes disagree with manifest for {path}")
        raw_by_path[path] = raw
        files[path] = observed

    lines = raw_by_path[EPISODES].splitlines()
    if any(not line.strip() for line in lines):
        raise ValueError("episode JSONL contains an empty line")
    rows = [parse_json_line(line, line_number=index) for index, line in enumerate(lines, 1)]
    return rows, {"commit": PIN, "files": files}


def _is_int(value: Any) -> bool:
    return type(value) is int


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if not _is_int(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element array")
    result: list[float] = []
    for index, component in enumerate(value):
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise ValueError(f"{name}[{index}] must be a finite number")
        number = float(component)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(number)
    return result[0], result[1], result[2]


def _expect_equal(observed: Any, expected: Any, name: str) -> None:
    if observed != expected or type(observed) is not type(expected):
        raise ValueError(f"{name} disagrees with the XYZ-derived value")


def _in_requested_cone(
    object_xyz: tuple[float, float, float],
    reference_xyz: tuple[float, float, float],
    relation: str,
) -> bool:
    dx = object_xyz[0] - reference_xyz[0]
    dy = object_xyz[1] - reference_xyz[1]
    horizontal = math.hypot(dx, dy)
    requested = dy if relation == "left" else -dy
    return horizontal > 1e-8 and requested / horizontal >= COS_45


def _first_true(mask: list[bool]) -> int | None:
    return next((index for index, value in enumerate(mask) if value), None)


def _first_sustained(mask: list[bool], length: int = SUSTAINED_STEPS) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == length:
            return index - length + 1
    return None


def _entry_kind(mask: list[bool]) -> str:
    if not any(mask):
        return "none"
    return "sustained" if _first_sustained(mask) is not None else "transient"


def _validate_timeline(raw: Mapping[str, Any], first_entry: int | None, end: int) -> None:
    timeline = raw.get("event_timeline")
    if not isinstance(timeline, list) or len(timeline) < 2:
        raise ValueError("event_timeline must retain episode_start and episode_end")
    parsed: list[tuple[str, int]] = []
    for index, value in enumerate(timeline):
        event = _require_mapping(value, f"event_timeline[{index}]")
        name = event.get("event")
        if not isinstance(name, str) or not name:
            raise ValueError(f"event_timeline[{index}].event must be a string")
        step = _require_int(event.get("action_step"), f"event_timeline[{index}].action_step", minimum=0)
        if step > end or (parsed and step < parsed[-1][1]):
            raise ValueError("event_timeline must be ordered inside the observed action range")
        parsed.append((name, step))
    if parsed[0] != ("episode_start", 0) or parsed[-1] != ("episode_end", end):
        raise ValueError("event_timeline boundary events disagree with retained steps")
    entries = [step for name, step in parsed if name == "requested_region_entry"]
    expected = [] if first_entry is None else [first_entry]
    if entries != expected:
        raise ValueError("requested_region_entry event disagrees with retained XYZ")


def normalize_episode(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one retained episode and derive entry endpoints from raw XYZ."""
    if raw.get("schema_version") != "vla-wam-shared-v3-raw-episode-v1":
        raise ValueError("unexpected H01 episode schema")
    if raw.get("record_type") != "behavioral_episode" or raw.get("behavioral_result_valid") is not True:
        raise ValueError("only valid retained behavioral episodes are admissible")
    if raw.get("study_id") != "vla_wam_language_steerability_v3":
        raise ValueError("unexpected study identity")
    if raw.get("amendment_id") != "V3-B001" or raw.get("arena") != "droid_robolab":
        raise ValueError("unexpected H01 amendment or arena")
    if raw.get("measurement_frame") != FRAME:
        raise ValueError("unknown measurement frame; do not infer cone signs")

    seed = _require_int(raw.get("policy_seed"), "policy_seed", minimum=0)
    environment_seed = _require_int(raw.get("environment_seed"), "environment_seed", minimum=0)
    arm = raw.get("phase_b_arm")
    side = raw.get("requested_relation")
    if seed not in SEEDS or environment_seed != seed or arm not in ARMS or side not in SIDES:
        raise ValueError("episode is outside the fixed H01 matched cohort")
    expected_pair = f"v3b001:nano:seed{seed}"
    expected_cell = f"{expected_pair}:{arm}:{side}"
    if raw.get("pair_id") != expected_pair or raw.get("registered_cell_id") != expected_cell:
        raise ValueError("registered H01 pair/cell identity mismatch")

    cap = _require_int(raw.get("action_cap"), "action_cap", minimum=0)
    end = _require_int(raw.get("actions_executed"), "actions_executed", minimum=0)
    if cap != ACTION_CAP or end > cap:
        raise ValueError("H01 action cap or episode end is outside the frozen contract")
    right_censored = _require_bool(raw.get("right_censored"), "right_censored")
    success = _require_bool(raw.get("requested_success"), "requested_success")
    if right_censored and (success or end != cap):
        raise ValueError("right-censored H01 cells must be unsuccessful and end at the cap")

    raw_steps = raw.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != end + 1:
        raise ValueError("steps must contain the initial state and every post-action state")
    requested_mask: list[bool] = []
    for index, raw_step in enumerate(raw_steps):
        step = _require_mapping(raw_step, f"steps[{index}]")
        if _require_int(step.get("action_step"), f"steps[{index}].action_step", minimum=0) != index:
            raise ValueError("action_step values must be contiguous from zero")
        object_xyz = _require_xyz(step.get("object_xyz"), f"steps[{index}].object_xyz")
        reference_xyz = _require_xyz(step.get("reference_xyz"), f"steps[{index}].reference_xyz")
        _require_bool(step.get("grippers_open"), f"steps[{index}].grippers_open")
        if "requested_region" in step or "lateral_offset_m" in step or "forward_offset_m" in step:
            raise ValueError("requested cone membership must be rederived from raw XYZ")
        requested_mask.append(_in_requested_cone(object_xyz, reference_xyz, side))

    first_entry = _first_true(requested_mask)
    first_sustained = _first_sustained(requested_mask)
    kind = _entry_kind(requested_mask)
    measurements = _require_mapping(raw.get("measurements"), "measurements")
    checks = {
        "actions_executed": end,
        "episode_length_steps": end,
        "executed_action_count": end,
        "scored_state_sample_count": end + 1,
        "right_censored": right_censored,
        "sustained_steps": SUSTAINED_STEPS,
        "first_requested_entry_step": first_entry,
        "first_cone_or_native_region_entry_step": first_entry,
        "first_sustained_requested_entry_step": first_sustained,
        "first_requested_region_step": first_sustained,
        "requested_entry_kind": kind,
        "entry_kind": kind,
    }
    for name, expected in checks.items():
        _expect_equal(measurements.get(name), expected, f"measurements.{name}")
    _validate_timeline(raw, first_entry, end)

    # The primary restricted waiting value is identified only because every
    # no-entry episode is administratively censored at the common action cap.
    if first_entry is None and not (right_censored and end == cap):
        raise ValueError("no-entry episode lacks common-cap administrative censoring")
    restricted_wait = first_entry if first_entry is not None else cap

    if first_sustained is not None:
        sustained_status = "observed_sustained_entry"
    elif kind == "transient" and right_censored:
        sustained_status = "cap_ended_after_transient_entry_without_three_sample_run"
    elif kind == "transient":
        sustained_status = "goal_terminated_after_transient_entry_without_three_sample_run"
    else:
        sustained_status = "administratively_right_censored_at_cap_without_entry"

    initial_hash = raw.get("initial_state_sha256")
    if not isinstance(initial_hash, str) or len(initial_hash) != 64:
        raise ValueError("initial_state_sha256 is missing or malformed")
    return {
        "seed": seed,
        "arm": arm,
        "side": side,
        "registered_cell_id": expected_cell,
        "pair_id": expected_pair,
        "initial_state_sha256": initial_hash,
        "actions_executed": end,
        "action_sample_count": len(raw_steps),
        "requested_success": success,
        "failure_taxonomy": raw.get("failure_taxonomy"),
        "right_censored": right_censored,
        "entry_observed": first_entry is not None,
        "first_entry_step": first_entry,
        "restricted_waiting_actions": restricted_wait,
        "entry_kind": kind,
        "sustained_entry_observed": first_sustained is not None,
        "first_sustained_entry_step": first_sustained,
        "sustained_status": sustained_status,
    }


def _describe(values: Iterable[int | float]) -> dict[str, Any]:
    materialized = list(values)
    return {
        "n": len(materialized),
        "mean": statistics.mean(materialized) if materialized else None,
        "median": statistics.median(materialized) if materialized else None,
        "min": min(materialized) if materialized else None,
        "max": max(materialized) if materialized else None,
    }


def _sign_summary(values: Iterable[int | float]) -> dict[str, int]:
    materialized = list(values)
    return {
        "positive": sum(value > 0 for value in materialized),
        "negative": sum(value < 0 for value in materialized),
        "ties": sum(value == 0 for value in materialized),
    }


def _event_ecdf(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kaplan-Meier event CDF; censoring occurs only at the common cap here."""
    events = Counter(
        episode["first_entry_step"] for episode in episodes if episode["entry_observed"]
    )
    censors = Counter(
        ACTION_CAP for episode in episodes if not episode["entry_observed"]
    )
    times = sorted(set(events) | set(censors) | {ACTION_CAP})
    survival = 1.0
    removed = 0
    curve: list[dict[str, Any]] = []
    for step in times:
        at_risk = len(episodes) - removed
        event_count = events[step]
        censored_count = censors[step]
        if event_count > at_risk or event_count + censored_count > at_risk:
            raise ValueError("invalid event/censor risk set")
        if event_count:
            survival *= 1.0 - event_count / at_risk
        curve.append({
            "action_step": step,
            "at_risk_before_step": at_risk,
            "events_at_step": event_count,
            "cap_censored_at_step": censored_count,
            "cumulative_event_count": sum(count for time, count in events.items() if time <= step),
            "event_cdf": 1.0 - survival,
        })
        removed += event_count + censored_count
    return curve


def _condition_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if len(episodes) != len(SEEDS):
        raise ValueError("each H01 condition must contain all 27 matched seed labels")
    observed = [episode["first_entry_step"] for episode in episodes if episode["entry_observed"]]
    sustained = [
        episode["first_sustained_entry_step"]
        for episode in episodes
        if episode["sustained_entry_observed"]
    ]
    return {
        "episodes": len(episodes),
        "entry_events": len(observed),
        "cap_right_censored_without_entry": sum(not episode["entry_observed"] for episode in episodes),
        "entry_kind_counts": {
            kind: sum(episode["entry_kind"] == kind for episode in episodes)
            for kind in ("sustained", "transient", "none")
        },
        "requested_successes": sum(episode["requested_success"] for episode in episodes),
        "behavioral_failures": sum(not episode["requested_success"] for episode in episodes),
        "entry_events_among_behavioral_failures": sum(
            episode["entry_observed"] and not episode["requested_success"] for episode in episodes
        ),
        "restricted_waiting_actions": _describe(
            episode["restricted_waiting_actions"] for episode in episodes
        ),
        "observed_entry_action": _describe(observed),
        "event_time_ecdf": _event_ecdf(episodes),
        "sustained_entry_audit": {
            "observed_sustained_entries": len(sustained),
            "observed_sustained_entry_action": _describe(sustained),
            "status_counts": dict(sorted(Counter(episode["sustained_status"] for episode in episodes).items())),
            "scope": (
                "Trace-status audit only. Goal-terminated transient entries do not retain a common "
                "three-sample post-entry window and are not assigned an action-450 event time."
            ),
        },
    }


def _block_record(seed: int, indexed: Mapping[tuple[int, str, str], dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, dict[str, Any]] = {}
    for arm, side in CONDITION_ORDER:
        episode = indexed[(seed, arm, side)]
        cells[f"{arm}_{side}"] = {
            "registered_cell_id": episode["registered_cell_id"],
            "actions_executed": episode["actions_executed"],
            "entry_observed": episode["entry_observed"],
            "first_entry_step": episode["first_entry_step"],
            "restricted_waiting_actions": episode["restricted_waiting_actions"],
            "entry_kind": episode["entry_kind"],
            "sustained_entry_observed": episode["sustained_entry_observed"],
            "first_sustained_entry_step": episode["first_sustained_entry_step"],
            "sustained_status": episode["sustained_status"],
            "requested_success": episode["requested_success"],
        }
    control_gap = (
        cells["control_right"]["restricted_waiting_actions"]
        - cells["control_left"]["restricted_waiting_actions"]
    )
    mirrored_gap = (
        cells["position_mirrored_right"]["restricted_waiting_actions"]
        - cells["position_mirrored_left"]["restricted_waiting_actions"]
    )
    return {
        "seed": seed,
        "pair_id": f"v3b001:nano:seed{seed}",
        "cells": cells,
        "contrasts": {
            "control_right_minus_left_actions": control_gap,
            "position_mirrored_right_minus_left_actions": mirrored_gap,
            "difference_in_differences_actions": mirrored_gap - control_gap,
            "position_mirrored_minus_control_left_actions": (
                cells["position_mirrored_left"]["restricted_waiting_actions"]
                - cells["control_left"]["restricted_waiting_actions"]
            ),
            "position_mirrored_minus_control_right_actions": (
                cells["position_mirrored_right"]["restricted_waiting_actions"]
                - cells["control_right"]["restricted_waiting_actions"]
            ),
        },
    }


def _paired_attainment(
    indexed: Mapping[tuple[int, str, str], dict[str, Any]], side: str
) -> dict[str, int]:
    counts = Counter()
    for seed in SEEDS:
        control = indexed[(seed, "control", side)]["entry_observed"]
        mirrored = indexed[(seed, "position_mirrored", side)]["entry_observed"]
        if control and mirrored:
            counts["both_observed"] += 1
        elif control:
            counts["control_observed_mirrored_censored"] += 1
        elif mirrored:
            counts["mirrored_observed_control_censored"] += 1
        else:
            counts["both_censored"] += 1
    return {
        name: counts[name]
        for name in (
            "both_observed",
            "control_observed_mirrored_censored",
            "mirrored_observed_control_censored",
            "both_censored",
        )
    }


def analyze_rows(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact H01 cohort and calculate the predeclared contrasts."""
    if len(rows) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} H01 episodes, received {len(rows)}")
    normalized = [normalize_episode(row) for row in rows]
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    identities: set[str] = set()
    for episode in normalized:
        key = (episode["seed"], episode["arm"], episode["side"])
        if key in indexed or episode["registered_cell_id"] in identities:
            raise ValueError("duplicate H01 cohort cell")
        indexed[key] = episode
        identities.add(episode["registered_cell_id"])
    expected_keys = {(seed, arm, side) for seed in SEEDS for arm, side in CONDITION_ORDER}
    if set(indexed) != expected_keys:
        raise ValueError("H01 cohort does not contain the exact 27-by-2-by-2 factorial")

    for seed in SEEDS:
        pair_ids = {indexed[(seed, arm, side)]["pair_id"] for arm, side in CONDITION_ORDER}
        if pair_ids != {f"v3b001:nano:seed{seed}"}:
            raise ValueError("four H01 cells in a seed block do not share one pair identity")
        for arm in ARMS:
            hashes = {indexed[(seed, arm, side)]["initial_state_sha256"] for side in SIDES}
            if len(hashes) != 1:
                raise ValueError("LEFT/RIGHT cells do not share the retained layout state")
    unique_hashes = {
        arm: len({episode["initial_state_sha256"] for episode in normalized if episode["arm"] == arm})
        for arm in ARMS
    }
    if unique_hashes != {"control": 1, "position_mirrored": 1}:
        raise ValueError("H01 no longer represents one base state and one reflected state")
    action_samples = sum(episode["action_sample_count"] for episode in normalized)
    if action_samples != EXPECTED_ACTION_SAMPLES:
        raise ValueError("retained H01 action-sample count disagrees with the pinned cohort")

    conditions: dict[str, Any] = {}
    for arm, side in CONDITION_ORDER:
        condition = [indexed[(seed, arm, side)] for seed in SEEDS]
        conditions[f"{arm}_{side}"] = _condition_summary(condition)

    blocks = [_block_record(seed, indexed) for seed in SEEDS]
    interactions = [block["contrasts"]["difference_in_differences_actions"] for block in blocks]
    control_gaps = [block["contrasts"]["control_right_minus_left_actions"] for block in blocks]
    mirrored_gaps = [
        block["contrasts"]["position_mirrored_right_minus_left_actions"] for block in blocks
    ]
    left_reflection = [
        block["contrasts"]["position_mirrored_minus_control_left_actions"] for block in blocks
    ]
    right_reflection = [
        block["contrasts"]["position_mirrored_minus_control_right_actions"] for block in blocks
    ]

    entry_events = sum(episode["entry_observed"] for episode in normalized)
    sustained_events = sum(episode["sustained_entry_observed"] for episode in normalized)
    transient_entries = sum(episode["entry_kind"] == "transient" for episode in normalized)
    no_entries = sum(episode["entry_kind"] == "none" for episode in normalized)
    successes = sum(episode["requested_success"] for episode in normalized)
    cap_censored = sum(episode["right_censored"] for episode in normalized)
    if (entry_events, sustained_events, transient_entries, no_entries, successes, cap_censored) != (
        104, 91, 13, 4, 102, 6
    ):
        raise ValueError("pinned H01 endpoint counts disagree with the authorized retained cohort")

    return {
        "schema_version": "wmf-retained-h01-requested-region-entry-v1",
        "provenance": provenance,
        "definitions": {
            "requested_region": (
                "Robot-base horizontal object-minus-reference vector inside the requested 45-degree "
                "cone: +delta_y for LEFT, -delta_y for RIGHT, nonzero horizontal distance."
            ),
            "primary_event": "First retained state-sample action index inside the requested cone.",
            "administrative_censoring": (
                "No observed primary event contributes action 450 only when the episode is explicitly "
                "right-censored at the common 450-action cap. It remains a censored observation, not an event."
            ),
            "restricted_waiting_actions": (
                "min(T, 450) for the primary first-entry endpoint. For a cap-censored cell, 450 is an "
                "RMST-style restricted contribution and not an imputed entry time."
            ),
            "matched_estimand": (
                "For each seed s, [U(position_mirrored,RIGHT)-U(position_mirrored,LEFT)] "
                "- [U(control,RIGHT)-U(control,LEFT)], where U is restricted waiting actions. "
                "Positive means position reflection increases the RIGHT-vs-LEFT entry delay."
            ),
            "event_time_ecdf": (
                "Kaplan-Meier event CDF over the 27 fixed seed labels in a condition. The only primary-"
                "endpoint censoring is administrative at action 450, after all observed entry events."
            ),
            "sustained_entry": (
                "Start index of the first run of three consecutive requested-cone samples. Reported as "
                "a trace-status audit, not a common-horizon time estimand."
            ),
        },
        "cohort": {
            "episodes": len(normalized),
            "matched_seed_labels": len(SEEDS),
            "seed_range": [SEEDS[0], SEEDS[-1]],
            "episodes_per_condition": len(SEEDS),
            "condition_order": [f"{arm}_{side}" for arm, side in CONDITION_ORDER],
            "retained_action_samples": action_samples,
            "behavioral_successes": successes,
            "behavioral_failures": len(normalized) - successes,
            "right_censored_episodes": cap_censored,
            "first_entry_events": entry_events,
            "cap_right_censored_without_entry": no_entries,
            "sustained_entry_events": sustained_events,
            "transient_entries": transient_entries,
            "unique_initial_states_by_layout": unique_hashes,
            "prescribed_base_scenes": 1,
            "prescribed_layout_states": 2,
        },
        "conditions": conditions,
        "matched_analysis": {
            "difference_in_differences_actions": {
                "description": (
                    "Position-mirrored RIGHT-minus-LEFT restricted waiting minus control "
                    "RIGHT-minus-LEFT restricted waiting."
                ),
                "summary": _describe(interactions),
                "sign_counts": _sign_summary(interactions),
            },
            "within_layout_right_minus_left_actions": {
                "control": {"summary": _describe(control_gaps), "sign_counts": _sign_summary(control_gaps)},
                "position_mirrored": {
                    "summary": _describe(mirrored_gaps),
                    "sign_counts": _sign_summary(mirrored_gaps),
                },
            },
            "position_mirrored_minus_control_actions": {
                "left": {"summary": _describe(left_reflection), "sign_counts": _sign_summary(left_reflection)},
                "right": {"summary": _describe(right_reflection), "sign_counts": _sign_summary(right_reflection)},
            },
            "paired_entry_attainment": {
                side: _paired_attainment(indexed, side) for side in SIDES
            },
        },
        "matched_blocks": blocks,
        "claim_boundary": {
            "status": "retrospective retained-data description",
            "new_model_inference": False,
            "new_robot_episodes": False,
            "new_human_labels": False,
            "forecast_accuracy_evidence": False,
            "independent_scene_replications": 1,
            "matched_seed_labels_are_independent_scenes": False,
            "action_index_is_physical_time": False,
            "first_entry_is_behavioral_success": False,
            "sustained_entry_has_common_post_entry_observation_window": False,
            "confirmation_test_or_stage_gate": False,
            "boundaries": [
                "The 27 seed labels are matched policy/environment labels on one base scene and its reflection, not 27 independent scenes.",
                "Action indices are simulator control indices; no conversion to wall-clock or video time is supported.",
                "Requested-cone entry can be transient and is not pickup, transport, detached release, success, or language understanding.",
                "Two behavioral failures entered the requested cone; 104 first entries must not be reported as 104 successes (the retained cohort has 102 successes).",
                "Goal termination occurs at or just after all 13 transient entries, so no post-entry dwell or common-450 sustained-entry claim is supported.",
                "The analysis does not complete A5/A6/A7, score forecast accuracy, supply annotations, or pass a confirmation gate.",
                "No coordinate is interpolated, extrapolated, or carried forward after an episode ends.",
            ],
        },
    }


def build_report(root: Path = ROOT) -> dict[str, Any]:
    rows, provenance = read_pinned_sources(root)
    return analyze_rows(rows, provenance)


CSV_CELL_FIELDS = (
    "actions_executed",
    "entry_observed",
    "first_entry_step",
    "restricted_waiting_actions",
    "entry_kind",
    "sustained_entry_observed",
    "first_sustained_entry_step",
    "sustained_status",
    "requested_success",
)
CSV_CONTRAST_FIELDS = (
    "control_right_minus_left_actions",
    "position_mirrored_right_minus_left_actions",
    "difference_in_differences_actions",
    "position_mirrored_minus_control_left_actions",
    "position_mirrored_minus_control_right_actions",
)


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if type(value) is bool:
        return "true" if value else "false"
    return value


def render_csv(report: Mapping[str, Any]) -> str:
    fieldnames = ["seed", "pair_id"]
    for arm, side in CONDITION_ORDER:
        prefix = f"{arm}_{side}"
        fieldnames.extend(f"{prefix}_{field}" for field in CSV_CELL_FIELDS)
    fieldnames.extend(CSV_CONTRAST_FIELDS)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for block in report["matched_blocks"]:
        row: dict[str, Any] = {"seed": block["seed"], "pair_id": block["pair_id"]}
        for arm, side in CONDITION_ORDER:
            prefix = f"{arm}_{side}"
            cell = block["cells"][prefix]
            for field in CSV_CELL_FIELDS:
                row[f"{prefix}_{field}"] = _csv_scalar(cell[field])
        for field in CSV_CONTRAST_FIELDS:
            row[field] = block["contrasts"][field]
        writer.writerow(row)
    return buffer.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    report = build_report(args.repo_root)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(render_json(report), encoding="utf-8")
    args.csv_output.write_text(render_csv(report), encoding="utf-8")
    print(json.dumps({
        "json_output": str(args.json_output),
        "csv_output": str(args.csv_output),
        "episodes": report["cohort"]["episodes"],
        "entry_events": report["cohort"]["first_entry_events"],
        "cap_censored": report["cohort"]["cap_right_censored_without_entry"],
        "matched_blocks": len(report["matched_blocks"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
