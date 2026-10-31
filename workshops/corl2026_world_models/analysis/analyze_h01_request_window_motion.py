#!/usr/bin/env python3
"""Authenticate and describe retained H01 request-window execution motion.

This retrospective analysis reads one immutable Git object.  It binds each
recorded Nano forecast request to the *observed* simulator-state interval from
its action start through start + 32.  A window that ends after episode
termination is censored: it is never interpolated, extrapolated, or carried
forward.  Recorded forecast-descriptor metadata are authenticated as provenance
only; payload arrays are not reopened and no generated-versus-executed
comparison is made.
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
DEFAULT_JSON = PACKAGE / "paper/data/h01_request_window_motion.json"
DEFAULT_CSV = PACKAGE / "paper/data/h01_request_window_motion.csv"
EPISODES = (
    "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/"
    "nano_v3b001_episodes.jsonl"
)
EXPECTED_SOURCES = {
    EPISODES: {
        "git_blob": "5c2f4a6fa4795ec054dbce2d4d68554245ceb4cb",
        "sha256": "ce67aecb2705394e8161fa4b65df7130e62648e2f6d8b6a93408c8da685a7207",
        "bytes": 5_822_218,
        "role": "108 retained H01 cells, 738 request descriptors, and robot-frame XYZ traces",
    },
    "experiments/v3/cosmos_nano_phase_b/compile_cell.py": {
        "git_blob": "f47ec3cae608a6d983b9b37641f7af6193421964",
        "sha256": "468bb9231316c4670a2dc3388ab7eb6f2ef5f42afc8fb04b16faca7b823f2a75",
        "bytes": 18_316,
        "role": "retained-cell and request-record construction semantics",
    },
    "experiments/v3/cosmos_nano_phase_b/robolab_bridge.py": {
        "git_blob": "d6261c83e9ac2ab31453a5aad1f4c1360aec74b7",
        "sha256": "d93ceb145430764341640c9925830fc10dce1329626a5076b52196f8b2d3a62b",
        "bytes": 34_272,
        "role": "robot-base object and reference XYZ sampling semantics",
    },
    "experiments/v3/cosmos_nano_phase_b/fixture_tasks.py": {
        "git_blob": "67879fb86b222bd6dcb8e3b243a4ee693aa2f60d",
        "sha256": "c2a9c10ffad7c51910448f70e7b4957f6ede75f27eb761e54224a2144d456148",
        "bytes": 4_928,
        "role": "frozen scene and requested-side task semantics",
    },
}
SEEDS = tuple(range(9400, 9427))
LAYOUTS = ("control", "position_mirrored")
COMMANDS = ("left", "right")
WINDOW_ACTIONS = 32
FRAME = "robot_base_object_minus_reference_xyz_m"
FUTURE_SHAPE = [33, 528, 640, 3]
ACTION_SHAPE = [32, 8]
FUTURE_BYTES = 33_454_208
ACTION_BYTES = 1_152
METRICS = (
    "cube_displacement_m",
    "bowl_displacement_m",
    "relative_displacement_m",
    "signed_lateral_relative_change_m",
)

NEW_ACTIVITY = {
    "model_requests": 0,
    "behavioral_actions": 0,
    "behavioral_cells": 0,
    "request_rows": 0,
    "request_selections": 0,
    "annotation_media": 0,
    "human_responses": 0,
    "labels": 0,
    "forecast_accuracy_estimates": 0,
    "retrieval_jobs": 0,
    "queue_descriptors": 0,
    "runnable_manifests": 0,
    "confirmation_inference_cells": 0,
}
GATE_STATE = {
    "current_GM_PVC_worker_state_reconciled": False,
    "live_quiescence_inferred": False,
    "camera_alignment_ready": False,
    "development_annotation_gate_ready": False,
    "annotation_media_ready": False,
    "human_annotation_ready": False,
    "resource_freeze_ready": False,
    "confirmation_release_ready": False,
    "safe_to_release_new_work": False,
}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _reject_nonfinite(value: Any, *, label: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number in {label}")
    if isinstance(value, list):
        for item in value:
            _reject_nonfinite(item, label=label)
    elif isinstance(value, dict):
        for item in value.values():
            _reject_nonfinite(item, label=label)


def parse_json_object(raw: bytes, *, label: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid {label}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _reject_nonfinite(value, label=label)
    return value


def read_pinned_sources(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commit = _git(root, "rev-parse", "--verify", PIN + "^{commit}").decode().strip()
    if commit != PIN:
        raise ValueError("pinned H01 commit identity changed")
    source_bytes: dict[str, bytes] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for path, expected in EXPECTED_SOURCES.items():
        object_id = _git(root, "rev-parse", "--verify", PIN + ":" + path).decode().strip()
        if object_id != expected["git_blob"]:
            raise ValueError(f"pinned Git blob changed: {path}")
        raw = _git(root, "cat-file", "blob", object_id)
        framed = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        observed = {
            "git_blob": framed,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "role": expected["role"],
        }
        if observed != expected or object_id != framed:
            raise ValueError(f"pinned source bytes changed: {path}")
        source_bytes[path] = raw
        receipts[path] = observed
    lines = source_bytes[EPISODES].splitlines()
    if len(lines) != 108 or any(not line.strip() for line in lines):
        raise ValueError("H01 JSONL must contain exactly 108 nonempty rows")
    rows = [parse_json_object(line, label=f"H01 row {index}") for index, line in enumerate(lines, 1)]
    return rows, {"commit": PIN, "files": receipts}


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 hex digest") from error
    return value.lower()


def _xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-element XYZ array")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name} must contain finite numbers")
        item = float(item)
        if not math.isfinite(item):
            raise ValueError(f"{name} must contain finite numbers")
        result.append(item)
    return tuple(result)  # type: ignore[return-value]


def _descriptor(
    value: Any, *, name: str, shape: Any, expected_shape: list[int], expected_bytes: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "bytes"}:
        raise ValueError(f"{name} descriptor schema changed")
    path = value["path"]
    if not isinstance(path, str) or not path.startswith("/") or not path.strip():
        raise ValueError(f"{name} source path must be a nonempty absolute recovery pointer")
    if shape != expected_shape or any(type(item) is not int for item in shape):
        raise ValueError(f"{name} shape changed")
    count = _integer(value["bytes"], f"{name}.bytes", minimum=1)
    if count != expected_bytes:
        raise ValueError(f"{name} byte count changed")
    digest = _sha256(value["sha256"], f"{name}.sha256")
    return {
        "bytes": count,
        "sha256": digest,
        # Preserve uniqueness and recovery-lineage identity without exporting
        # the private absolute source prefix.
        "source_path_sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
        "shape": list(shape),
    }


def motion_metrics(
    start_cube: Iterable[float],
    start_bowl: Iterable[float],
    end_cube: Iterable[float],
    end_bowl: Iterable[float],
) -> dict[str, float]:
    sc, sb, ec, eb = tuple(start_cube), tuple(start_bowl), tuple(end_cube), tuple(end_bowl)
    if any(len(value) != 3 for value in (sc, sb, ec, eb)):
        raise ValueError("motion metric inputs must be three-dimensional")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item))
           for value in (sc, sb, ec, eb) for item in value):
        raise ValueError("motion metric inputs must be finite numbers")
    cube_delta = tuple(float(ec[i]) - float(sc[i]) for i in range(3))
    bowl_delta = tuple(float(eb[i]) - float(sb[i]) for i in range(3))
    relative_delta = tuple(cube_delta[i] - bowl_delta[i] for i in range(3))
    return {
        "cube_displacement_m": math.sqrt(sum(value * value for value in cube_delta)),
        "bowl_displacement_m": math.sqrt(sum(value * value for value in bowl_delta)),
        "relative_displacement_m": math.sqrt(sum(value * value for value in relative_delta)),
        "signed_lateral_relative_change_m": relative_delta[1],
    }


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    values = list(values)
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("summary requires finite nonempty values")
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _condition(layout: str, command: str) -> str:
    return f"{layout}_{command}"


def _sign_counts(values: Iterable[float]) -> dict[str, int]:
    values = list(values)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("sign counts require finite values")
    return {
        "positive": sum(value > 0.0 for value in values),
        "negative": sum(value < 0.0 for value in values),
        "exact_zero": sum(value == 0.0 for value in values),
    }


def _fixed_index_support(seed_values: Iterable[tuple[int, float]]) -> dict[str, Any]:
    """Summarize one prespecified request index without filling attrition.

    Every H01 layout has 27 registered seed labels.  A later index is observed
    only when both historical command cells retain the exact +32-action
    endpoint.  Missing labels remain unknown in the strict-positive fraction
    bounds; they are never carried forward or assigned a sign.
    """
    seed_values = list(seed_values)
    seeds = [seed for seed, _ in seed_values]
    values = [value for _, value in seed_values]
    observed = len(values)
    if (
        observed > len(SEEDS)
        or len(set(seeds)) != observed
        or not set(seeds).issubset(SEEDS)
    ):
        raise ValueError("fixed-index support has invalid H01 seed identities")
    signs = _sign_counts(values)
    missing = len(SEEDS) - observed
    return {
        "registered_seed_labels": len(SEEDS),
        "observable_seed_labels": observed,
        "unavailable_seed_labels": missing,
        "observable_seed_ids": sorted(seeds),
        "unavailable_seed_ids": sorted(set(SEEDS) - set(seeds)),
        "observed_signed_contrast_m": (
            {**_summary(values), "sign_counts": signs} if values else None
        ),
        "strict_positive_full_cohort_fraction_bounds": {
            "lower": signs["positive"] / len(SEEDS),
            "upper": (signs["positive"] + missing) / len(SEEDS),
        },
    }


def matched_complete_window_analysis(
    requests: list[dict[str, Any]],
    indexed: Mapping[tuple[int, str, str], dict[str, Any]],
) -> dict[str, Any]:
    """Pair only exact +32 windows observed for both commands.

    This is selection accounting for the retained request-window census.  It is
    deliberately outcome-conditioned: a window is retained only through the
    shorter historical LEFT/RIGHT trace.  The equal-seed mean telescopes to the
    last shared request-boundary response divided by a variable window count;
    it is not an independent endpoint or a physical-time rate.
    """
    all_windows: dict[tuple[int, str, str], dict[int, dict[str, Any]]] = {}
    complete: dict[tuple[int, str, str], dict[int, dict[str, Any]]] = {}
    for seed in SEEDS:
        for layout in LAYOUTS:
            for command in COMMANDS:
                observed = {
                    request["request_index"]: request
                    for request in requests
                    if request["seed"] == seed
                    and request["layout"] == layout
                    and request["command"] == command
                }
                if sorted(observed) != list(range(len(observed))):
                    raise ValueError("H01 request windows must be a contiguous prefix")
                rows = {
                    index: request for index, request in observed.items()
                    if request["complete_32_action_window"]
                }
                if sorted(rows) != list(range(len(rows))):
                    raise ValueError("complete H01 request windows must be a contiguous prefix")
                all_windows[(seed, layout, command)] = observed
                complete[(seed, layout, command)] = rows

    fixed_request_indices = sorted(set().union(*(
        set(all_windows[(seed, layout, command)])
        for seed in SEEDS
        for layout in LAYOUTS
        for command in COMMANDS
    )))
    if fixed_request_indices != list(range(len(fixed_request_indices))):
        raise ValueError("fixed request-index inventory must be a contiguous prefix")
    fixed_index_rows = []
    for index in fixed_request_indices:
        by_fixed_layout: dict[str, Any] = {}
        for layout in LAYOUTS:
            seed_values = []
            for seed in SEEDS:
                left = complete[(seed, layout, "left")]
                right = complete[(seed, layout, "right")]
                if index in left and index in right:
                    seed_values.append((
                        seed,
                        left[index]["signed_lateral_relative_change_m"]
                        - right[index]["signed_lateral_relative_change_m"],
                    ))
            by_fixed_layout[layout] = _fixed_index_support(seed_values)

        same_support_interactions = []
        for seed in SEEDS:
            if all(
                index in complete[(seed, layout, command)]
                for layout in LAYOUTS
                for command in COMMANDS
            ):
                contrasts = {
                    layout: (
                        complete[(seed, layout, "left")][index][
                            "signed_lateral_relative_change_m"
                        ]
                        - complete[(seed, layout, "right")][index][
                            "signed_lateral_relative_change_m"
                        ]
                    )
                    for layout in LAYOUTS
                }
                same_support_interactions.append((
                    seed,
                    contrasts["position_mirrored"] - contrasts["control"],
                ))
        fixed_index_rows.append({
            "request_index": index,
            "action_step_start": index * WINDOW_ACTIONS,
            "action_step_target": (index + 1) * WINDOW_ACTIONS,
            "by_layout": by_fixed_layout,
            "four_condition_same_index_interaction_m": _fixed_index_support(
                same_support_interactions
            ),
        })

    by_layout: dict[str, Any] = {}
    for layout in LAYOUTS:
        records: list[dict[str, Any]] = []
        window_contrasts: list[float] = []
        outcome_counts: Counter[str] = Counter()
        outcome_windows: Counter[str] = Counter()
        for seed in SEEDS:
            left = complete[(seed, layout, "left")]
            right = complete[(seed, layout, "right")]
            common = sorted(set(left) & set(right))
            if not common or common != list(range(len(common))):
                raise ValueError("matched H01 request windows must be a nonempty contiguous prefix")
            contrasts = [
                left[index]["signed_lateral_relative_change_m"]
                - right[index]["signed_lateral_relative_change_m"]
                for index in common
            ]
            if any(value is None or not math.isfinite(value) for value in contrasts):
                raise ValueError("matched H01 request-window contrast is unavailable")
            left_raw = indexed[(seed, layout, "left")]
            right_raw = indexed[(seed, layout, "right")]
            boundary = WINDOW_ACTIONS * len(common)

            def relative_y(raw: Mapping[str, Any], action_step: int) -> float:
                sample = raw["steps"][action_step]
                return _xyz(sample["object_xyz"], "matched cube")[1] - _xyz(
                    sample["reference_xyz"], "matched bowl"
                )[1]

            direct = (
                relative_y(left_raw, boundary) - relative_y(left_raw, 0)
                - (relative_y(right_raw, boundary) - relative_y(right_raw, 0))
            )
            cumulative = sum(contrasts)
            residual = cumulative - direct
            if abs(residual) > 1e-15:
                raise ValueError("matched request-window telescoping identity changed")
            outcome = (
                ("success" if left_raw["requested_success"] else "cap_failure")
                + "_left__"
                + ("success" if right_raw["requested_success"] else "cap_failure")
                + "_right"
            )
            outcome_counts[outcome] += 1
            outcome_windows[outcome] += len(common)
            record = {
                "seed": seed,
                "layout": layout,
                "shared_complete_window_pairs": len(common),
                "shared_request_indices": common,
                "last_shared_action_boundary": boundary,
                "left_complete_windows": len(left),
                "right_complete_windows": len(right),
                "left_requested_success": left_raw["requested_success"],
                "right_requested_success": right_raw["requested_success"],
                "pair_outcome": outcome,
                "window_contrast_signs": _sign_counts(contrasts),
                "mean_signed_lateral_contrast_m_per_32_action_window": statistics.mean(contrasts),
                "cumulative_signed_lateral_contrast_m": cumulative,
                "direct_boundary_contrast_m": direct,
                "telescoping_residual_m": residual,
                "first_window_contrast_m": contrasts[0],
            }
            records.append(record)
            window_contrasts.extend(contrasts)
        seed_means = [row["mean_signed_lateral_contrast_m_per_32_action_window"] for row in records]
        cumulative = [row["cumulative_signed_lateral_contrast_m"] for row in records]
        first = [row["first_window_contrast_m"] for row in records]
        left_complete = sum(len(complete[(seed, layout, "left")]) for seed in SEEDS)
        right_complete = sum(len(complete[(seed, layout, "right")]) for seed in SEEDS)
        paired = len(window_contrasts)
        exclusion_timing = {
            "counterpart_request_terminated_before_32_action_endpoint": {
                "left": 0, "right": 0,
            },
            "owner_request_started_at_or_after_counterpart_episode_end": {
                "left": 0, "right": 0,
            },
        }
        for seed in SEEDS:
            for owner, counterpart in (("left", "right"), ("right", "left")):
                owner_complete = complete[(seed, layout, owner)]
                counterpart_complete = complete[(seed, layout, counterpart)]
                counterpart_all = all_windows[(seed, layout, counterpart)]
                for index in set(owner_complete) - set(counterpart_complete):
                    if index in counterpart_all:
                        if counterpart_all[index]["complete_32_action_window"]:
                            raise ValueError("counterpart exclusion classification changed")
                        exclusion_timing[
                            "counterpart_request_terminated_before_32_action_endpoint"
                        ][owner] += 1
                    else:
                        owner_start = owner_complete[index]["action_step_start"]
                        counterpart_end = indexed[(seed, layout, counterpart)]["actions_executed"]
                        if owner_start < counterpart_end:
                            raise ValueError("absent counterpart request predates counterpart termination")
                        exclusion_timing[
                            "owner_request_started_at_or_after_counterpart_episode_end"
                        ][owner] += 1
        for timing in exclusion_timing.values():
            timing["total"] = timing["left"] + timing["right"]
        excluded = {
            "left": left_complete - paired,
            "right": right_complete - paired,
            "total": left_complete + right_complete - 2 * paired,
        }
        if sum(timing["total"] for timing in exclusion_timing.values()) != excluded["total"]:
            raise ValueError("counterpart exclusion timing does not close")
        by_layout[layout] = {
            "matched_seed_labels": len(records),
            "shared_complete_window_pairs": paired,
            "shared_window_pairs_by_pair_outcome": dict(sorted(outcome_windows.items())),
            "pair_outcome_counts": dict(sorted(outcome_counts.items())),
            "complete_window_records": {"left": left_complete, "right": right_complete},
            "complete_window_records_used_in_pairs": {"left": paired, "right": paired},
            "complete_window_records_excluded_without_exact_counterpart_endpoint": excluded,
            "excluded_record_timing": exclusion_timing,
            "shared_window_pair_count_per_seed": _summary(
                row["shared_complete_window_pairs"] for row in records
            ),
            "request_window_signed_lateral_contrast_m": {
                **_summary(window_contrasts),
                "sign_counts": _sign_counts(window_contrasts),
            },
            "equal_seed_mean_signed_lateral_contrast_m_per_32_action_window": {
                **_summary(seed_means),
                "sign_counts": _sign_counts(seed_means),
            },
            "last_shared_boundary_cumulative_signed_lateral_contrast_m": {
                **_summary(cumulative),
                "sign_counts": _sign_counts(cumulative),
            },
            "fixed_first_0_to_32_action_window_contrast_m": {
                **_summary(first),
                "sign_counts": _sign_counts(first),
            },
            "per_seed": records,
        }

    same_support_records = []
    for seed in SEEDS:
        shared = sorted(set.intersection(*(
            set(complete[(seed, layout, command)])
            for layout in LAYOUTS
            for command in COMMANDS
        )))
        if not shared or shared != list(range(len(shared))):
            raise ValueError("four-condition shared H01 support must be a nonempty contiguous prefix")
        layout_means = {}
        for layout in LAYOUTS:
            layout_means[layout] = statistics.mean(
                complete[(seed, layout, "left")][index]["signed_lateral_relative_change_m"]
                - complete[(seed, layout, "right")][index]["signed_lateral_relative_change_m"]
                for index in shared
            )
        same_support_records.append({
            "seed": seed,
            "shared_complete_window_pairs_per_layout": len(shared),
            "shared_request_indices": shared,
            "control_mean_contrast_m_per_32_action_window": layout_means["control"],
            "position_mirrored_mean_contrast_m_per_32_action_window": layout_means["position_mirrored"],
            "position_mirrored_minus_control_mean_contrast_m_per_32_action_window": (
                layout_means["position_mirrored"] - layout_means["control"]
            ),
        })
    same_support_differences = [
        row["position_mirrored_minus_control_mean_contrast_m_per_32_action_window"]
        for row in same_support_records
    ]
    return {
        "status": "retrospective matched-window selection and execution-motion audit",
        "by_layout": by_layout,
        "fixed_request_index_profile": {
            "rows": fixed_index_rows,
            "layout_observable_seed_labels": {
                layout: [
                    row["by_layout"][layout]["observable_seed_labels"]
                    for row in fixed_index_rows
                ]
                for layout in LAYOUTS
            },
            "four_condition_observable_seed_labels": [
                row["four_condition_same_index_interaction_m"][
                    "observable_seed_labels"
                ]
                for row in fixed_index_rows
            ],
        },
        "four_condition_same_request_index_support": {
            "matched_seed_labels": len(same_support_records),
            "shared_complete_window_pairs_per_layout": sum(
                row["shared_complete_window_pairs_per_layout"] for row in same_support_records
            ),
            "shared_window_pair_count_per_seed": _summary(
                row["shared_complete_window_pairs_per_layout"] for row in same_support_records
            ),
            "position_mirrored_minus_control_mean_contrast_m_per_32_action_window": {
                **_summary(same_support_differences),
                "sign_counts": _sign_counts(same_support_differences),
            },
            "per_seed": same_support_records,
        },
        "definitions": {
            "pairing": "Within a seed and layout, retain request indices whose exact +32-action endpoint exists in both LEFT and RIGHT historical cells. An otherwise-complete record is excluded when its counterpart episode ends before that endpoint; never fill a stopped counterpart.",
            "window_contrast": "D_window = change(cube_y-bowl_y)_LEFT - change(cube_y-bowl_y)_RIGHT in robot-base coordinates. Positive means the two commands increased LEFT/RIGHT ordering over that window; exact zero is a tie without a calibrated motion threshold.",
            "seed_weighting": "Average the matched windows within each seed; summaries then weight the 27 seed labels equally. Window counts remain variable and outcome-conditioned.",
            "telescoping": "Contiguous window contrasts sum exactly (within 1e-15 m) to the paired relative-offset response at the last shared 32-action boundary. The seed mean is that response divided by a variable window count, not independent evidence or a physical-time rate.",
            "same_support_layout_audit": "The layout comparison restricts each seed to request indices complete in all four layout-command cells. This removes unequal request-index support between layouts but not historical goal/cap outcome selection.",
            "first_window": "Availability and selection of the 0-to-32 action window are outcome-independent in this retained cohort because all cells survive past action 32. Its exact-zero measurement duplicates the stopping-control action-32 information rather than adding an endpoint.",
            "fixed_request_index_profile": "Each row fixes one recorded request index before comparing commands. Support requires the exact +32-action endpoint in both command cells; four-condition interactions require all four cells. Strict-positive bounds count every unavailable registered seed first as non-positive and then as positive. Later rows remain historical survival-selected action-index evidence, not physical time, independent windows, a calibrated movement test or forecast scoring.",
        },
    }


def analyze_rows(rows: list[dict[str, Any]], provenance: dict[str, Any]) -> dict[str, Any]:
    expected_cells = {(seed, layout, command) for seed in SEEDS for layout in LAYOUTS for command in COMMANDS}
    indexed: dict[tuple[int, str, str], dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    seen_request_ids: set[str] = set()
    seen_future_paths: set[str] = set()
    seen_future_hashes: set[str] = set()
    seen_action_paths: set[str] = set()
    seen_action_hashes: set[str] = set()

    for row in rows:
        if row.get("schema_version") != "vla-wam-shared-v3-raw-episode-v1":
            raise ValueError("unexpected H01 row schema")
        if row.get("study_id") != "vla_wam_language_steerability_v3" or row.get("amendment_id") != "V3-B001":
            raise ValueError("unexpected H01 study identity")
        if row.get("record_type") != "behavioral_episode" or row.get("behavioral_result_valid") is not True:
            raise ValueError("H01 request-window analysis requires valid behavioral episodes")
        if row.get("measurement_frame") != FRAME:
            raise ValueError("unknown H01 measurement frame")
        seed = _integer(row.get("policy_seed"), "policy_seed")
        if row.get("environment_seed") != seed:
            raise ValueError("policy/environment seed mismatch")
        layout, command = row.get("phase_b_arm"), row.get("requested_relation")
        key = (seed, layout, command)
        if key not in expected_cells or key in indexed:
            raise ValueError("missing, duplicate, or unexpected H01 cohort cell")
        expected_pair = f"v3b001:nano:seed{seed}"
        expected_cell = f"{expected_pair}:{layout}:{command}"
        if row.get("pair_id") != expected_pair or row.get("registered_cell_id") != expected_cell:
            raise ValueError("H01 pair/cell identity mismatch")
        if row.get("future_interface") != "decoded_rgb_uint8_33_frames_per_policy_request":
            raise ValueError("H01 future interface changed")
        if row.get("action_chunk_shape") != ACTION_SHAPE:
            raise ValueError("H01 action chunk shape changed")
        end = _integer(row.get("actions_executed"), "actions_executed")
        if row.get("action_cap") != 450:
            raise ValueError("H01 action cap changed")
        requested_success = row.get("requested_success")
        right_censored = row.get("right_censored")
        if type(requested_success) is not bool or type(right_censored) is not bool:
            raise ValueError("H01 outcome/censoring flags must be booleans")
        if requested_success == right_censored:
            raise ValueError("H01 success/right-censoring relationship changed")
        if right_censored and end != 450:
            raise ValueError("H01 right-censored episode did not reach its action cap")
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != end + 1:
            raise ValueError("H01 steps must be contiguous through episode termination")
        steps: list[dict[str, Any]] = []
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, dict) or raw_step.get("action_step") != index:
                raise ValueError("H01 action steps must be unique and contiguous")
            steps.append({
                "cube": _xyz(raw_step.get("object_xyz"), f"step {index} cube"),
                "bowl": _xyz(raw_step.get("reference_xyz"), f"step {index} bowl"),
            })
        raw_requests = row.get("future_requests")
        if not isinstance(raw_requests, list) or not raw_requests:
            raise ValueError("H01 cell must contain retained future requests")
        if len(raw_requests) != math.ceil(end / WINDOW_ACTIONS):
            raise ValueError("H01 per-episode request count/action-count lineage changed")
        reset = _sha256(row.get("reset_fingerprint_sha256"), "episode reset fingerprint")
        release = _sha256(row.get("release_fingerprint_sha256"), "episode release fingerprint")
        for expected_index, raw_request in enumerate(raw_requests):
            if not isinstance(raw_request, dict):
                raise ValueError("H01 request must be a JSON object")
            index = _integer(raw_request.get("request_index"), "request_index")
            start = _integer(raw_request.get("action_step_start"), "action_step_start")
            if index != expected_index or start != WINDOW_ACTIONS * index or start > end:
                raise ValueError("H01 request index/action-start lineage changed")
            if raw_request.get("future_evidence_status") != "exposed_and_retained":
                raise ValueError("H01 future evidence status changed")
            if raw_request.get("reset_fingerprint_sha256") != reset or raw_request.get("release_fingerprint_sha256") != release:
                raise ValueError("H01 request fingerprint disagrees with its cell")
            future_path = raw_request.get("decoded_future", {}).get("path") if isinstance(raw_request.get("decoded_future"), dict) else None
            action_path = raw_request.get("returned_action", {}).get("path") if isinstance(raw_request.get("returned_action"), dict) else None
            if raw_request.get("sampling_seed") != seed:
                raise ValueError("H01 request sampling seed disagrees with its cell")
            future = _descriptor(
                raw_request.get("decoded_future"), name="decoded_future",
                shape=raw_request.get("decoded_future_shape"), expected_shape=FUTURE_SHAPE,
                expected_bytes=FUTURE_BYTES,
            )
            action = _descriptor(
                raw_request.get("returned_action"), name="returned_action",
                shape=raw_request.get("returned_action_shape"), expected_shape=ACTION_SHAPE,
                expected_bytes=ACTION_BYTES,
            )
            request_id = f"{expected_cell}:request{index:03d}"
            if request_id in seen_request_ids or future_path in seen_future_paths or future["sha256"] in seen_future_hashes or action_path in seen_action_paths or action["sha256"] in seen_action_hashes:
                raise ValueError("duplicate H01 request or descriptor identity")
            seen_request_ids.add(request_id)
            seen_future_paths.add(str(future_path))
            seen_future_hashes.add(str(future["sha256"]))
            seen_action_paths.add(str(action_path))
            seen_action_hashes.add(str(action["sha256"]))
            target = start + WINDOW_ACTIONS
            complete = target <= end
            observed_prefix_actions = min(WINDOW_ACTIONS, end - start)
            censor_outcome = None
            if not complete:
                censor_outcome = (
                    "goal_stopped_success"
                    if requested_success
                    else "administrative_cap_censored_failure"
                )
            public = {
                "request_id": request_id,
                "cell_id": expected_cell,
                "pair_id": expected_pair,
                "seed": seed,
                "layout": layout,
                "command": command,
                "request_index": index,
                "action_step_start": start,
                "action_step_target": target if complete else None,
                "episode_actions_executed": end,
                "episode_requested_success": requested_success,
                "episode_right_censored": right_censored,
                "observed_prefix_actions": observed_prefix_actions,
                "complete_32_action_window": complete,
                "censor_reason": None if complete else "episode_terminated_before_action_start_plus_32",
                "censor_outcome": censor_outcome,
                "decoded_future": future,
                "returned_action": action,
                **{metric: None for metric in METRICS},
            }
            if complete:
                public.update(motion_metrics(
                    steps[start]["cube"], steps[start]["bowl"],
                    steps[target]["cube"], steps[target]["bowl"],
                ))
            requests.append(public)
        indexed[key] = row

    if set(indexed) != expected_cells or len(requests) != 738:
        raise ValueError("H01 cohort/request inventory changed")
    unique_initial_states = {
        layout: {
            row.get("initial_state_sha256")
            for (seed, observed_layout, command), row in indexed.items()
            if observed_layout == layout
        }
        for layout in LAYOUTS
    }
    if any(len(values) != 1 or None in values for values in unique_initial_states.values()):
        raise ValueError("H01 cells no longer share one retained initial state per layout")

    condition_summaries: dict[str, Any] = {}
    for layout in LAYOUTS:
        for command in COMMANDS:
            name = _condition(layout, command)
            selected = [request for request in requests if request["layout"] == layout and request["command"] == command]
            complete = [request for request in selected if request["complete_32_action_window"]]
            episode_means = []
            for seed in SEEDS:
                cell = [request for request in complete if request["seed"] == seed]
                if not cell:
                    raise ValueError("each H01 cell must retain at least one complete request window")
                episode_means.append({metric: statistics.mean(request[metric] for request in cell) for metric in METRICS})
            condition_summaries[name] = {
                "episodes": 27,
                "requests": len(selected),
                "complete_32_action_windows": len(complete),
                "censored_windows": len(selected) - len(complete),
                "censored_windows_by_episode_outcome": {
                    outcome: sum(request["censor_outcome"] == outcome for request in selected)
                    for outcome in (
                        "goal_stopped_success",
                        "administrative_cap_censored_failure",
                    )
                },
                "request_level_complete_window_metrics": {
                    metric: _summary(request[metric] for request in complete) for metric in METRICS
                },
                "episode_balanced_complete_window_metrics": {
                    metric: _summary(episode[metric] for episode in episode_means) for metric in METRICS
                },
            }

    starts = sorted({request["action_step_start"] for request in requests})
    by_start = []
    for start in starts:
        selected = [request for request in requests if request["action_step_start"] == start]
        complete = sum(request["complete_32_action_window"] for request in selected)
        by_start.append({
            "action_step_start": start,
            "requests": len(selected),
            "complete_32_action_windows": complete,
            "censored_windows": len(selected) - complete,
        })

    successes = sum(row.get("requested_success") is True for row in indexed.values())
    cap_failures = sum(row.get("right_censored") is True for row in indexed.values())
    if (successes, cap_failures) != (102, 6):
        raise ValueError("H01 episode outcome inventory changed")
    censored = [request for request in requests if not request["complete_32_action_window"]]
    outcome_counts = {
        outcome: sum(request["censor_outcome"] == outcome for request in censored)
        for outcome in (
            "goal_stopped_success",
            "administrative_cap_censored_failure",
        )
    }
    prefix_actions = [request["observed_prefix_actions"] for request in censored]
    if outcome_counts != {
        "goal_stopped_success": 100,
        "administrative_cap_censored_failure": 6,
    } or (sum(prefix_actions), min(prefix_actions), max(prefix_actions)) != (1748, 1, 31):
        raise ValueError("H01 incomplete-window censoring inventory changed")

    matched = matched_complete_window_analysis(requests, indexed)
    return {
        "schema_version": "wmf-retained-h01-request-window-motion-v3",
        "provenance": provenance,
        "cohort": {
            "historical_id": "H01",
            "study": "V3-B001",
            "episodes": 108,
            "matched_seed_labels": 27,
            "prescribed_layout_states": 2,
            "unique_initial_states_by_layout": {
                layout: len(unique_initial_states[layout]) for layout in LAYOUTS
            },
            "commands": 2,
            "requested_successes": successes,
            "administrative_cap_right_censored_failures": cap_failures,
            "retained_requests": 738,
            "request_window_actions": WINDOW_ACTIONS,
            "decoded_future_descriptors": 738,
            "returned_action_descriptors": 738,
            "unique_decoded_future_path_identities": len(seen_future_paths),
            "unique_decoded_future_hashes": len(seen_future_hashes),
            "unique_returned_action_path_identities": len(seen_action_paths),
            "unique_returned_action_hashes": len(seen_action_hashes),
            "decoded_future_manifest_bytes_not_retrieved": sum(request["decoded_future"]["bytes"] for request in requests),
            "returned_action_manifest_bytes_not_retrieved": sum(request["returned_action"]["bytes"] for request in requests),
        },
        "coverage": {
            "requests": len(requests),
            "complete_32_action_windows": sum(request["complete_32_action_window"] for request in requests),
            "censored_windows": sum(not request["complete_32_action_window"] for request in requests),
            "censored_windows_by_episode_outcome": outcome_counts,
            "censored_observed_prefix_actions": {
                **_summary(prefix_actions),
                "sum": sum(prefix_actions),
            },
            "by_action_step_start": by_start,
            "by_condition": condition_summaries,
        },
        "matched_complete_window_analysis": matched,
        "request_windows": requests,
        "definitions": {
            "window": "Recorded simulator state at request action_step_start versus the exact observed state at start+32 actions.",
            "censoring": "A window ending after episode termination is unavailable; its 1--31 observed prefix actions are counted but no endpoint is interpolated, extrapolated or carried forward.",
            "cube_displacement_m": "Euclidean robot-base XYZ displacement of the cube over the observed window.",
            "bowl_displacement_m": "Euclidean robot-base XYZ displacement of the bowl over the observed window.",
            "relative_displacement_m": "Euclidean displacement of the cube-minus-bowl XYZ relation over the observed window.",
            "signed_lateral_relative_change_m": "Change in robot-base (cube_y-bowl_y); positive is toward robot LEFT.",
            "request_level": "Descriptive complete-window requests; variable request counts do not represent independent episodes.",
            "episode_balanced": "Mean complete-window metric within each cell, then descriptive summary over 27 matched seed labels per condition.",
            "matched_complete_window_analysis": "Selection-transparent LEFT/RIGHT pairing of exact complete +32 windows. It remains a post-hoc, outcome-conditioned execution-motion view and is algebraically linked to the last shared request boundary.",
            "descriptor_provenance": "Public output authenticates recorded descriptor byte counts, recorded content hashes and shapes from the pinned JSONL plus hashes of private recovery pointers. Payload files were not reopened or byte-reverified, and absolute prefixes are never exported.",
        },
        "claim_boundary": {
            "retrospective_execution_motion_evidence": True,
            "forecast_descriptor_provenance_evidence": True,
            "forecast_arrays_retrieved": False,
            "forecast_payload_bytes_reopened_or_reverified": False,
            "forecast_accuracy_or_skill_estimated": False,
            "generated_versus_executed_comparison": False,
            "A3_A9_completed": False,
            "physical_time_established": False,
            "A6_movement_threshold_established": False,
            "independent_scene_or_layout_generalization_established": False,
            "language_understanding_established": False,
            "new_episodes_or_requests_created": False,
            "new_labels_or_retrieval_created": False,
            "models_or_cohorts_pooled": False,
            "maximum_core_behavioral_cells": 232,
            "changes_232_cell_core_bound": False,
        },
        "new_activity": dict(NEW_ACTIVITY),
        "gate_state": dict(GATE_STATE),
    }


CSV_FIELDS = (
    "request_id", "cell_id", "pair_id", "seed", "layout", "command",
    "request_index", "action_step_start", "action_step_target",
    "episode_actions_executed", "episode_requested_success", "episode_right_censored",
    "observed_prefix_actions", "complete_32_action_window", "censor_reason", "censor_outcome",
    "decoded_future_bytes", "decoded_future_sha256", "decoded_future_source_path_sha256",
    "returned_action_bytes", "returned_action_sha256", "returned_action_source_path_sha256",
    *METRICS,
)


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv(report: Mapping[str, Any]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for request in report["request_windows"]:
        row = {key: request.get(key) for key in CSV_FIELDS}
        for prefix in ("decoded_future", "returned_action"):
            descriptor = request[prefix]
            row[f"{prefix}_bytes"] = descriptor["bytes"]
            row[f"{prefix}_sha256"] = descriptor["sha256"]
            row[f"{prefix}_source_path_sha256"] = descriptor["source_path_sha256"]
        writer.writerow(row)
    return handle.getvalue()


def write_outputs(report: Mapping[str, Any], json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(render_json(report), encoding="utf-8")
    csv_path.write_text(render_csv(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    rows, provenance = read_pinned_sources(args.repo_root)
    report = analyze_rows(rows, provenance)
    write_outputs(report, args.json_output, args.csv_output)
    print(json.dumps({
        "json": str(args.json_output),
        "csv": str(args.csv_output),
        "episodes": report["cohort"]["episodes"],
        "requests": report["coverage"]["requests"],
        "complete_32_action_windows": report["coverage"]["complete_32_action_windows"],
        "censored_windows": report["coverage"]["censored_windows"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
