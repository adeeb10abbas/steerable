#!/usr/bin/env python3
"""Deterministic per-episode and matched-pair diagnostics for V3-B002.

The shared v3 schema owns the frozen success predicate and failure taxonomy.
This module adds only the prospectively registered B002 measurements that are
not part of that common schema.  In particular, ``grasp_step`` comes from the
retained ``object_grabbed`` stream and is never inferred from object lift.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


EPISODE_DIAGNOSTICS_SCHEMA = "vla-wam-shared-v3b-pi05-reflection-episode-diagnostics-v1"
PAIR_DIAGNOSTICS_SCHEMA = "vla-wam-shared-v3b-pi05-reflection-pair-v1"
EPISODE_DIAGNOSTICS_FIELD = "pi05_v3b002_diagnostics"


class DiagnosticError(ValueError):
    """Raised when raw evidence cannot support a registered diagnostic."""


def _fail(message: str) -> None:
    raise DiagnosticError(message)


def _finite_number(value: Any, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail(f"{label} must be a finite number")
    return float(value)


def _integer_or_none(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        _fail(f"{label} must be a non-negative integer or null")
    return value


def _require_steps(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("steps must be a non-empty array")
    output: list[Mapping[str, Any]] = []
    for index, raw in enumerate(steps):
        if not isinstance(raw, Mapping) or raw.get("action_step") != index:
            _fail("steps must be mappings with contiguous action_step values")
        output.append(raw)
    return output


def _lateral_offsets(steps: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for index, step in enumerate(steps):
        obj = step.get("object_xyz")
        ref = step.get("reference_xyz")
        if not isinstance(obj, list) or len(obj) != 3 or not isinstance(ref, list) or len(ref) != 3:
            _fail(f"steps[{index}] requires three-element object_xyz/reference_xyz")
        values.append(
            _finite_number(obj[1], f"steps[{index}].object_xyz[1]")
            - _finite_number(ref[1], f"steps[{index}].reference_xyz[1]")
        )
    return values


def derive_episode_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Derive all registered B002 episode fields from retained raw streams."""

    if record.get("record_type") != "behavioral_episode" or record.get("behavioral_result_valid") is not True:
        _fail("episode diagnostics require one valid behavioral episode")
    relation = record.get("requested_relation")
    if relation not in {"left", "right"}:
        _fail("requested_relation must be left or right")
    success = record.get("requested_success")
    if type(success) is not bool:
        _fail("requested_success must be boolean")
    failure = record.get("failure_taxonomy")
    if failure not in {"correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"}:
        _fail("failure_taxonomy is outside the frozen five-class taxonomy")
    if (failure == "correct") != success:
        _fail("correct failure category must agree with requested_success")

    measurements = record.get("measurements")
    if not isinstance(measurements, Mapping):
        _fail("validated schema-derived measurements are required")
    steps = _require_steps(record)
    lateral = _lateral_offsets(steps)
    final_offset = _finite_number(
        measurements.get("signed_final_lateral_offset_m"),
        "measurements.signed_final_lateral_offset_m",
    )
    if not math.isclose(final_offset, lateral[-1], rel_tol=0.0, abs_tol=1e-12):
        _fail("signed final lateral offset disagrees with retained XYZ states")
    requested_depth = _finite_number(
        measurements.get("final_requested_signed_margin_m"),
        "measurements.final_requested_signed_margin_m",
    )
    expected_depth = final_offset if relation == "left" else -final_offset
    if not math.isclose(requested_depth, expected_depth, rel_tol=0.0, abs_tol=1e-12):
        _fail("requested-side depth disagrees with the frozen sign convention")

    grabbed: list[bool] = []
    for index, step in enumerate(steps):
        value = step.get("object_grabbed")
        if type(value) is not bool:
            _fail(
                f"steps[{index}].object_grabbed must be a retained boolean; "
                "verified pickup is not a grasp substitute"
            )
        grabbed.append(value)
    grasp_step = next((index for index, value in enumerate(grabbed) if value), None)

    contact_status = measurements.get("first_contact_status")
    if contact_status not in {"observed", "not_observed", "instrumentation_unavailable"}:
        _fail("first_contact_status is invalid")
    first_contact = _integer_or_none(
        measurements.get("first_contact_step"), "measurements.first_contact_step"
    )
    contact_reason = measurements.get("first_contact_unavailable_reason")
    if contact_status == "observed" and first_contact is None:
        _fail("observed contact requires a first-contact step")
    if contact_status != "observed" and first_contact is not None:
        _fail("non-observed contact cannot carry a first-contact step")
    if contact_status == "instrumentation_unavailable":
        if not isinstance(contact_reason, str) or not contact_reason.strip():
            _fail("unavailable contact requires a non-empty reason")
    elif contact_reason is not None:
        _fail("contact-unavailable reason is only valid when instrumentation is unavailable")

    cone_entry = _integer_or_none(
        measurements.get("first_requested_entry_step"),
        "measurements.first_requested_entry_step",
    )
    sustained_entry = _integer_or_none(
        measurements.get("first_sustained_requested_entry_step"),
        "measurements.first_sustained_requested_entry_step",
    )
    actions = record.get("actions_executed")
    if type(actions) is not int or actions < 0:
        _fail("actions_executed must be a non-negative integer")
    if len(steps) != actions + 1:
        _fail("steps must contain the pre-action sample plus every post-action sample")

    cumulative_lateral_path = sum(
        abs(current - previous) for previous, current in zip(lateral, lateral[1:])
    )
    peak_lateral_excursion = max(abs(value - lateral[0]) for value in lateral)
    return {
        "schema_version": EPISODE_DIAGNOSTICS_SCHEMA,
        "success": success,
        "failure_category": failure,
        "signed_final_lateral_offset_m": final_offset,
        "requested_side_depth_m": requested_depth,
        "cone_entry_step": cone_entry,
        "cone_entry_sustained": sustained_entry is not None,
        "episode_length_steps": actions,
        "time_to_first_contact_steps": first_contact,
        "first_contact_status": contact_status,
        "first_contact_unavailable_reason": contact_reason,
        "grasp_step": grasp_step,
        "grasp_source": "retained_object_grabbed_boolean_stream",
        "cumulative_lateral_path_m": cumulative_lateral_path,
        "peak_lateral_excursion_m": peak_lateral_excursion,
        # These are pair-derived by design and therefore never fabricated in
        # a single-cell episode row.
        "endpoint_shift_m": None,
        "action_distinct": None,
        "pair_diagnostics_location": "separate_matched_pair_jsonl",
    }


def attach_episode_diagnostics(record: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with canonical diagnostics, rejecting a conflicting copy."""

    derived = derive_episode_diagnostics(record)
    declared = record.get(EPISODE_DIAGNOSTICS_FIELD)
    if declared is not None and declared != derived:
        _fail("declared V3-B002 episode diagnostics disagree with retained raw streams")
    output = dict(record)
    # Keep the registered fields directly queryable in raw JSONL while also
    # grouping their derivation metadata under one versioned object.
    top_level = {
        key: value
        for key, value in derived.items()
        if key
        not in {
            "schema_version",
            "endpoint_shift_m",
            "action_distinct",
            "pair_diagnostics_location",
        }
    }
    for key, value in top_level.items():
        if key in output and output[key] != value:
            _fail(f"declared top-level {key} disagrees with retained raw streams")
        output[key] = value
    output[EPISODE_DIAGNOSTICS_FIELD] = derived
    return output


def derive_pair_diagnostics(
    *,
    seed: int,
    arm: str,
    left_record: Mapping[str, Any],
    right_record: Mapping[str, Any],
    left_actions: np.ndarray,
    right_actions: np.ndarray,
) -> dict[str, Any]:
    """Derive the separately retained matched LEFT/RIGHT pair diagnostics."""

    if type(seed) is not int or arm not in {"control", "position_mirrored"}:
        _fail("pair seed/layout is invalid")
    left = attach_episode_diagnostics(left_record)
    right = attach_episode_diagnostics(right_record)
    if left.get("environment_seed") != seed or right.get("environment_seed") != seed:
        _fail("pair records do not match the requested seed")
    if left.get("requested_relation") != "left" or right.get("requested_relation") != "right":
        _fail("pair records must be ordered LEFT then RIGHT")
    if left.get("phase_b_arm") != arm or right.get("phase_b_arm") != arm:
        _fail("pair records do not match the requested layout")
    if left.get("initial_state_sha256") != right.get("initial_state_sha256"):
        _fail("matched LEFT/RIGHT pair does not share an identical reset")

    arrays: list[np.ndarray] = []
    for name, raw in (("left_actions", left_actions), ("right_actions", right_actions)):
        value = np.asarray(raw)
        if value.ndim != 2 or value.shape[1] != 8 or value.shape[0] < 1:
            _fail(f"{name} must have finite shape [N,8] with N >= 1")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            _fail(f"{name} must contain only finite numeric values")
        arrays.append(value)
    left_array, right_array = arrays
    if left_array.shape[0] != left.get("actions_executed"):
        _fail("LEFT action count disagrees with its raw episode")
    if right_array.shape[0] != right.get("actions_executed"):
        _fail("RIGHT action count disagrees with its raw episode")
    common = min(left_array.shape[0], right_array.shape[0])
    left_prefix = left_array[:common]
    right_prefix = right_array[:common]
    delta = left_prefix.astype(np.float64) - right_prefix.astype(np.float64)

    left_diag = left[EPISODE_DIAGNOSTICS_FIELD]
    right_diag = right[EPISODE_DIAGNOSTICS_FIELD]
    endpoint_d = (
        float(left_diag["signed_final_lateral_offset_m"])
        - float(right_diag["signed_final_lateral_offset_m"])
    )
    requested_depth_b = (
        float(right_diag["requested_side_depth_m"])
        - float(left_diag["requested_side_depth_m"])
    )
    return {
        "schema_version": PAIR_DIAGNOSTICS_SCHEMA,
        "study_id": left.get("study_id"),
        "amendment_id": "V3-B002",
        "model_id": left.get("model_id"),
        "seed": seed,
        "matched_block_id": left.get("pair_id"),
        "arm": arm,
        "left_registered_cell_id": left.get("registered_cell_id"),
        "right_registered_cell_id": right.get("registered_cell_id"),
        "initial_state_sha256": left.get("initial_state_sha256"),
        "endpoint_redirection_D_m": endpoint_d,
        "endpoint_shift_m": endpoint_d,
        "requested_side_depth_contrast_B_m": requested_depth_b,
        "left_success": bool(left_diag["success"]),
        "right_success": bool(right_diag["success"]),
        "right_minus_left_success": int(bool(right_diag["success"])) - int(bool(left_diag["success"])),
        "executed_actions_distinct": not np.array_equal(left_prefix, right_prefix),
        "action_distinct": not np.array_equal(left_prefix, right_prefix),
        "action_distinct_definition": "bitwise inequality on the complete common executed prefix",
        "left_executed_action_count": int(left_array.shape[0]),
        "right_executed_action_count": int(right_array.shape[0]),
        "common_prefix_action_count": int(common),
        "common_prefix_action_rms": float(math.sqrt(float(np.mean(delta * delta)))),
    }
