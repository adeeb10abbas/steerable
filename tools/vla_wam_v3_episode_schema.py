#!/usr/bin/env python3
"""Validation and deterministic serialization for v3 raw episode JSONL.

The module is deliberately independent of the frozen v2 compilers.  A v3
behavioral record contains raw, action-indexed measurements; this module
derives the shared continuous fields and rejects a claimed taxonomy unless it
matches the frozen precedence.  Technical/partial attempts use a separate
infrastructure record type and can never receive a behavioral taxonomy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


BEHAVIORAL_SCHEMA_VERSION = "vla-wam-shared-v3-raw-episode-v1"
INFRASTRUCTURE_SCHEMA_VERSION = "vla-wam-shared-v3-infrastructure-attempt-v1"
ARENAS = frozenset({"droid_robolab", "robotwin_place_a2b"})
TAXONOMY = frozenset(
    {"correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"}
)
FROZEN_PRECEDENCE = (
    "correct", "pick_failed", "wrong_side", "release_failed", "transport_failed",
)
INFRASTRUCTURE_CLASSIFICATIONS = frozenset({"technical_invalid", "partial"})
_WORKSPACE = Path(__file__).resolve().parents[1]
_V3_ARTIFACTS = _WORKSPACE / "artifacts" / "vla_wam_shared_v3"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MEASUREMENT_FRAME_ID = "robot_base_object_minus_reference_xyz_m"
MEASUREMENT_FRAME_DESCRIPTION = (
    "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
    "forward is object-minus-reference x and lateral is object-minus-reference y, "
    "with positive lateral denoting robot LEFT."
)
_DROID_OBJECT_MOTION_THRESHOLD_M = 0.01


class EpisodeSchemaError(ValueError):
    """Raised when one JSONL record violates the v3 raw-data contract."""


def _fail(message: str) -> None:
    raise EpisodeSchemaError(message)


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if type(value) is not bool:
        _fail(f"{name} must be a boolean")
    return value


def _require_int(value: Any, name: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        _fail(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        _fail(f"{name} must be >= {minimum}")
    return value


def _require_number(value: Any, name: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        _fail(f"{name} must be a finite number")
    return float(value)


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be a non-empty string")
    return value


@lru_cache(maxsize=1)
def _frozen_contract() -> dict[str, Any]:
    """Load the committed v3 protocol and taxonomy rather than duplicating them."""

    try:
        protocol = json.loads((_V3_ARTIFACTS / "protocol.json").read_text())
        taxonomy = json.loads((_V3_ARTIFACTS / "failure_taxonomy.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise EpisodeSchemaError(f"could not load frozen v3 contract: {error}") from error
    if (
        protocol.get("schema_version") != "vla-wam-shared-v3-protocol-v1"
        or taxonomy.get("schema_version") != "vla-wam-shared-v3-failure-taxonomy-v1"
        or protocol.get("study_id") != taxonomy.get("study_id")
    ):
        _fail("committed v3 protocol/taxonomy contract is inconsistent")
    precedence = taxonomy.get("primary_precedence")
    if tuple(precedence) != FROZEN_PRECEDENCE:
        _fail("committed v3 taxonomy classes are inconsistent")
    scoring = taxonomy.get("state_and_scoring_contract", {})
    required_outputs = protocol.get("common_execution_contract", {}).get("required_raw_outputs")
    sustained_steps = scoring.get("sustained_samples")
    pickup_lift_m = scoring.get("pickup_threshold_m")
    if type(sustained_steps) is not int or sustained_steps <= 0:
        _fail("committed v3 taxonomy sustained_samples is invalid")
    if type(pickup_lift_m) not in {int, float} or not math.isfinite(float(pickup_lift_m)):
        _fail("committed v3 taxonomy pickup_threshold_m is invalid")
    if not isinstance(required_outputs, list) or set(required_outputs) != {
        "viewport_video", "executed_action_trace", "raw_result_jsonl"
    }:
        _fail("committed v3 protocol required raw outputs are inconsistent")
    contact_statuses = taxonomy.get("contact_conditional_rules", {}).get("allowed_statuses")
    if set(contact_statuses or []) != {
        "observed", "not_observed", "instrumentation_unavailable"
    }:
        _fail("committed v3 taxonomy contact statuses are inconsistent")
    return {
        "study_id": protocol["study_id"],
        "precedence": tuple(precedence),
        "sustained_steps": sustained_steps,
        "pickup_lift_m": float(pickup_lift_m),
        "required_raw_outputs": tuple(required_outputs),
        "contact_statuses": frozenset(contact_statuses),
    }


def _sustained_steps() -> int:
    return _frozen_contract()["sustained_steps"]


def _pickup_lift_m() -> float:
    return _frozen_contract()["pickup_lift_m"]


def _require_xyz(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        _fail(f"{name} must be a three-element array")
    return tuple(_require_number(component, f"{name}[{index}]") for index, component in enumerate(value))  # type: ignore[return-value]


def _first_true(mask: list[bool]) -> int | None:
    return next((index for index, value in enumerate(mask) if value), None)


def _first_sustained(mask: list[bool], steps: int | None = None) -> int | None:
    steps = _sustained_steps() if steps is None else steps
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == steps:
            return index - steps + 1
    return None


def _final_sustained(mask: list[bool], steps: int | None = None) -> bool:
    steps = _sustained_steps() if steps is None else steps
    return len(mask) >= steps and all(mask[-steps:])


def _entry_kind(mask: list[bool]) -> str:
    if not any(mask):
        return "none"
    return "sustained" if _first_sustained(mask) is not None else "transient"


def _validate_artifact_record(value: Any, name: str) -> None:
    record = _require_mapping(value, name)
    _require_string(record.get("path"), f"{name}.path")
    digest = _require_string(record.get("sha256"), f"{name}.sha256")
    if not _SHA256_RE.fullmatch(digest):
        _fail(f"{name}.sha256 must be a lowercase SHA-256 hex digest")
    # A zero-byte file is not viewport/action evidence.  Infrastructure rows
    # omit artifacts that did not exist; any artifact they do claim must be a
    # real, non-empty retained file.
    _require_int(record.get("bytes"), f"{name}.bytes", minimum=1)


def _validate_provenance(
    record: dict[str, Any], *, require_behavioral_artifacts: bool = True
) -> None:
    """Require the scientific identity needed to reproduce one attempted cell."""

    contract = _frozen_contract()
    if record.get("study_id") != contract["study_id"]:
        _fail("study_id does not match the frozen v3 protocol")
    for key in (
        "registered_cell_id", "attempt_id", "model_id", "pair_id", "prompt",
        "prompt_family", "predicate_id", "reset_id",
    ):
        _require_string(record.get(key), key)
    if record.get("measurement_frame") != MEASUREMENT_FRAME_ID:
        _fail("measurement_frame must use the frozen robot-base frame ID")
    if record.get("measurement_frame_description") != MEASUREMENT_FRAME_DESCRIPTION:
        _fail("measurement_frame_description must use the frozen robot-base frame description")
    _require_int(record.get("environment_seed"), "environment_seed", minimum=0)
    _require_int(record.get("policy_seed"), "policy_seed", minimum=0)
    checkpoint = _require_mapping(record.get("checkpoint"), "checkpoint")
    _require_string(checkpoint.get("id"), "checkpoint.id")
    _require_string(checkpoint.get("revision"), "checkpoint.revision")
    runtime = _require_mapping(record.get("runtime_identity"), "runtime_identity")
    _require_string(runtime.get("id"), "runtime_identity.id")
    runtime_digest = _require_string(runtime.get("sha256"), "runtime_identity.sha256")
    if not _SHA256_RE.fullmatch(runtime_digest):
        _fail("runtime_identity.sha256 must be a lowercase SHA-256 hex digest")
    artifacts = _require_mapping(record.get("artifacts"), "artifacts")
    required_outputs = _frozen_contract()["required_raw_outputs"]
    raw_jsonl = _require_mapping(
        artifacts.get("raw_result_jsonl"), "artifacts.raw_result_jsonl"
    )
    _require_string(raw_jsonl.get("path"), "artifacts.raw_result_jsonl.path")
    if raw_jsonl.get("integrity_scope") != "batch_manifest_after_close":
        _fail("artifacts.raw_result_jsonl requires batch_manifest_after_close integrity_scope")
    if "sha256" in raw_jsonl or "bytes" in raw_jsonl:
        _fail("artifacts.raw_result_jsonl must not make an inline self-hash or byte claim")

    # A setup failure can occur before a viewport or action trace exists.  It
    # still needs the common scientific identity and post-close JSONL
    # integrity record, but inventing zero-byte behavioral artifacts would be
    # false provenance.  Behavioral rows retain the complete-output gate;
    # infrastructure rows validate only artifacts that were actually emitted.
    for key in required_outputs:
        if key == "raw_result_jsonl":
            continue
        if require_behavioral_artifacts or key in artifacts:
            _validate_artifact_record(artifacts.get(key), f"artifacts.{key}")


def _validate_behavioral_event_timeline(record: dict[str, Any], measurements: dict[str, Any]) -> None:
    events = record.get("event_timeline")
    if not isinstance(events, list) or len(events) < 2:
        _fail("event_timeline must contain episode_start and episode_end")
    parsed: list[tuple[str, int]] = []
    for index, raw_event in enumerate(events):
        event = _require_mapping(raw_event, f"event_timeline[{index}]")
        name = _require_string(event.get("event"), f"event_timeline[{index}].event")
        step = _require_int(event.get("action_step"), f"event_timeline[{index}].action_step", minimum=0)
        if step > measurements["actions_executed"]:
            _fail("event_timeline event is outside the episode action range")
        if parsed and step < parsed[-1][1]:
            _fail("event_timeline must be in nondecreasing action-step order")
        parsed.append((name, step))
    if parsed[0] != ("episode_start", 0):
        _fail("event_timeline must start with episode_start at action step zero")
    if parsed[-1] != ("episode_end", measurements["actions_executed"]):
        _fail("event_timeline must end with episode_end at the final action step")
    if len({name for name, _ in parsed}) != len(parsed):
        _fail("event_timeline event names must be unique")
    expected_events = {
        "first_contact": measurements["first_contact_step"],
        "verified_pickup": measurements["first_verified_pickup_step"],
        "requested_region_entry": measurements["first_requested_entry_step"],
        "opposite_region_entry": measurements["first_opposite_entry_step"],
    }
    observed = dict(parsed)
    for name, step in expected_events.items():
        if name in observed and observed[name] != step:
            _fail(f"event_timeline {name} does not match derived measurement")
        if step is not None and name not in observed:
            _fail(f"event_timeline must retain derived {name}")


def _path_length(points: list[tuple[float, float, float]]) -> float:
    return sum(
        math.dist(previous, current) for previous, current in zip(points, points[1:])
    )


def derive_initial_state_sha256(record: dict[str, Any]) -> str:
    """Hash the shared physical reset fields without prompt/direction labels.

    The hash deliberately covers only measurements available in every arena:
    the robot-base object/reference poses and the initial detached/open state.
    Model-specific compilers may retain a richer native simulator fingerprint,
    but this common digest is sufficient to fail closed on a LEFT/RIGHT scene
    mismatch instead of trusting a caller-supplied hash.
    """

    record = _require_mapping(record, "record")
    steps = record.get("steps")
    if not isinstance(steps, list) or not steps:
        _fail("steps must be a non-empty array before deriving initial-state identity")
    first = _require_mapping(steps[0], "steps[0]")
    payload = {
        "measurement_frame": record.get("measurement_frame"),
        "object_xyz": list(_require_xyz(first.get("object_xyz"), "steps[0].object_xyz")),
        "reference_xyz": list(
            _require_xyz(first.get("reference_xyz"), "steps[0].reference_xyz")
        ),
        "grippers_open": _require_bool(
            first.get("grippers_open"), "steps[0].grippers_open"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def derive_frozen_failure_stage(
    record: dict[str, Any], measurements: dict[str, Any] | None = None
) -> str:
    """Reproduce the arena-specific v2 stage alongside the v3 taxonomy."""

    record = _require_mapping(record, "record")
    measurements = measurements or derive_measurements(record)
    if record["requested_success"]:
        return "success"
    steps = record["steps"]
    if record["arena"] == "droid_robolab":
        initial = _require_xyz(steps[0].get("object_xyz"), "steps[0].object_xyz")
        moved = [
            math.dist(initial, _require_xyz(step.get("object_xyz"), f"steps[{index}].object_xyz"))
            >= _DROID_OBJECT_MOTION_THRESHOLD_M
            for index, step in enumerate(steps)
        ]
        if _first_sustained(moved) is None:
            return "no_object_interaction"
        if not measurements["verified_pickup"]:
            return "object_moved_no_verified_pickup"
        if measurements["first_requested_entry_step"] is None:
            return "picked_never_entered_requested_region"
        return "entered_requested_region_not_released"

    requested_entry = measurements["first_requested_entry_step"] is not None
    initial_z = _require_xyz(steps[0].get("object_xyz"), "steps[0].object_xyz")[2]
    pickup_with_closed_gripper = _first_sustained(
        [
            _require_xyz(step.get("object_xyz"), f"steps[{index}].object_xyz")[2]
            - initial_z
            >= _pickup_lift_m()
            and not _require_bool(step.get("grippers_open"), f"steps[{index}].grippers_open")
            for index, step in enumerate(steps)
        ]
    ) is not None
    if requested_entry:
        return "entered_requested_region_without_verified_completion"
    if pickup_with_closed_gripper:
        return "picked_never_entered_requested_region"
    if any(not bool(step["grippers_open"]) for step in steps):
        return "closed_gripper_no_verified_pickup"
    return "no_verified_interaction"


def _direction_sign(relation: str) -> float:
    # This is the scorer coordinate, not the website/display convention that
    # negates robot-frame y.  Robot LEFT is positive robot-frame y.
    return 1.0 if relation == "left" else -1.0


def _droid_region(lateral_offset_m: float, forward_offset_m: float, relation: str) -> bool:
    """Frozen DROID relation: requested half-plane inside a 45-degree cone."""

    requested_margin = _direction_sign(relation) * lateral_offset_m
    horizontal_distance = math.hypot(lateral_offset_m, forward_offset_m)
    return horizontal_distance > 1e-8 and requested_margin / horizontal_distance >= math.cos(math.radians(45.0))


def _validate_steps(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[bool], list[bool], list[float], list[tuple[float, float, float]], list[bool] | None]:
    arena = record["arena"]
    relation = record["requested_relation"]
    steps_value = record.get("steps")
    if not isinstance(steps_value, list) or not steps_value:
        _fail("steps must be a non-empty array")

    requested: list[bool] = []
    opposite: list[bool] = []
    lateral: list[float] = []
    object_xyz: list[tuple[float, float, float]] = []
    contacts: list[bool] = []
    has_contact: bool | None = None
    normalized_steps: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps_value):
        step = _require_mapping(raw_step, f"steps[{index}]")
        if _require_int(step.get("action_step"), f"steps[{index}].action_step", minimum=0) != index:
            _fail("action_step values must begin at zero and increase by one")
        point = _require_xyz(step.get("object_xyz"), f"steps[{index}].object_xyz")
        if "target_xyz" in step:
            _fail("steps use reference_xyz, not target_xyz")
        reference = _require_xyz(step.get("reference_xyz"), f"steps[{index}].reference_xyz")
        if "lateral_offset_m" in step or "forward_offset_m" in step:
            _fail("lateral_offset_m and forward_offset_m are derived from robot-base XYZ samples")
        forward_value = point[0] - reference[0]
        lateral_value = point[1] - reference[1]
        _require_bool(step.get("grippers_open"), f"steps[{index}].grippers_open")

        if arena == "droid_robolab":
            requested_value = _droid_region(lateral_value, forward_value, relation)
            opposite_value = _droid_region(lateral_value, forward_value, "right" if relation == "left" else "left")
            if "requested_region" in step or "opposite_region" in step:
                _fail("DROID region fields are derived from the frozen 45-degree cone")
            region_kind = "droid_45_degree_cone"
        else:
            requested_value = _require_bool(step.get("requested_region"), f"steps[{index}].requested_region")
            opposite_value = _require_bool(step.get("opposite_region"), f"steps[{index}].opposite_region")
            if requested_value and opposite_value:
                _fail("RoboTwin step cannot be in requested and opposite native regions")
            region_kind = "robotwin_native_relation_region"

        contact_present = "contact_detected" in step
        if has_contact is None:
            has_contact = contact_present
        elif has_contact != contact_present:
            _fail("contact_detected must be present for every step or for no steps")
        if contact_present:
            contacts.append(_require_bool(step["contact_detected"], f"steps[{index}].contact_detected"))

        normalized_steps.append({
            "action_step": index,
            "object_xyz": list(point),
            "lateral_offset_m": lateral_value,
            "forward_offset_m": forward_value,
            "requested_region": requested_value,
            "opposite_region": opposite_value,
            "region_kind": region_kind,
        })
        requested.append(requested_value)
        opposite.append(opposite_value)
        lateral.append(lateral_value)
        object_xyz.append(point)
    return normalized_steps, requested, opposite, lateral, object_xyz, contacts if has_contact else None


def derive_measurements(record: dict[str, Any]) -> dict[str, Any]:
    """Derive v3 continuous fields from exactly one raw behavioral record."""

    record = _require_mapping(record, "record")
    _validate_behavioral_header(record)
    _validate_provenance(record)
    steps, requested, opposite, lateral, object_xyz, contacts = _validate_steps(record)
    actions_executed = _require_int(record.get("actions_executed"), "actions_executed", minimum=0)
    if actions_executed != steps[-1]["action_step"]:
        _fail("actions_executed must equal the final action_step")
    action_cap = _require_int(record.get("action_cap"), "action_cap", minimum=0)
    right_censored = _require_bool(record.get("right_censored"), "right_censored")
    if right_censored and (record["requested_success"] or actions_executed != action_cap):
        _fail("right-censored episodes must be unsuccessful and end exactly at action_cap")
    if not right_censored and actions_executed > action_cap:
        _fail("actions_executed cannot exceed action_cap")

    initial_z = object_xyz[0][2]
    lift = [point[2] - initial_z for point in object_xyz]
    pickup_mask = [value >= _pickup_lift_m() for value in lift]
    first_pickup = _first_sustained(pickup_mask)
    first_requested = _first_true(requested)
    first_opposite = _first_true(opposite)
    first_sustained_requested = _first_sustained(requested)
    first_sustained_opposite = _first_sustained(opposite)

    declared_contact = record.get("first_contact_step")
    unavailable_reason = record.get("first_contact_unavailable_reason")
    if contacts is None:
        if declared_contact is not None:
            _fail("first_contact_step must be null when no contact stream is retained")
        if not isinstance(unavailable_reason, str) or not unavailable_reason.strip():
            _fail("null first_contact_step requires a non-empty first_contact_unavailable_reason")
        first_contact = None
        first_contact_status = "instrumentation_unavailable"
    else:
        first_contact = _first_true(contacts)
        if declared_contact != first_contact:
            _fail("first_contact_step must exactly match the retained contact stream")
        if unavailable_reason is not None:
            _fail("first_contact_unavailable_reason is only allowed when contact is unavailable")
        first_contact_status = "observed" if first_contact is not None else "not_observed"

    requested_margin = _direction_sign(record["requested_relation"]) * lateral[-1]
    requested_margins = [_direction_sign(record["requested_relation"]) * value for value in lateral]
    wall_time = _require_number(record.get("wall_time_s"), "wall_time_s")
    if wall_time < 0:
        _fail("wall_time_s must be non-negative")
    operational_wall_time_valid = _require_bool(
        record.get("operational_wall_time_valid"), "operational_wall_time_valid"
    )
    final_detached_release = _require_bool(
        record.get("final_detached_release"), "final_detached_release"
    )
    if record["requested_success"] and not final_detached_release:
        _fail("requested_success requires the frozen scorer final_detached_release predicate")
    if (
        not record["requested_success"]
        and _final_sustained(requested)
        and final_detached_release
    ):
        _fail(
            "scorer inconsistency: sustained requested region plus detached release "
            "cannot be a behavioral failure"
        )
    measurements = {
        "region_kind": steps[0]["region_kind"],
        "sustained_steps": _sustained_steps(),
        "pickup_lift_threshold_m": _pickup_lift_m(),
        # State samples include the pre-action sample at index zero, so there
        # is one more sample than executed simulator actions.
        "episode_length_steps": actions_executed,
        "scored_state_sample_count": len(steps),
        "actions_executed": actions_executed,
        "executed_action_count": actions_executed,
        "right_censored": right_censored,
        "signed_final_lateral_offset_m": lateral[-1],
        "requested_final_lateral_margin_m": requested_margin,
        "final_requested_signed_margin_m": requested_margin,
        "final_opposite_signed_margin_m": -requested_margin,
        "maximum_requested_signed_margin_m": max(requested_margins),
        "object_path_length_m": _path_length(object_xyz),
        "max_object_lift_m": max(lift),
        "maximum_pickup_height_m": max(lift),
        "verified_pickup": first_pickup is not None,
        "first_verified_pickup_step": first_pickup,
        "first_requested_entry_step": first_requested,
        "first_cone_or_native_region_entry_step": first_requested,
        "first_opposite_entry_step": first_opposite,
        "first_sustained_requested_entry_step": first_sustained_requested,
        "first_sustained_opposite_entry_step": first_sustained_opposite,
        "first_requested_region_step": first_sustained_requested,
        "requested_entry_kind": _entry_kind(requested),
        "entry_kind": _entry_kind(requested),
        "opposite_entry_kind": _entry_kind(opposite),
        "final_requested_region_sustained": _final_sustained(requested),
        "final_opposite_region_sustained": _final_sustained(opposite),
        "final_detached_release": final_detached_release,
        "wall_time_s": wall_time,
        "operational_wall_time_valid": operational_wall_time_valid,
        "first_contact_step": first_contact,
        "first_contact_status": first_contact_status,
        "first_contact_unavailable_reason": unavailable_reason if contacts is None else None,
    }
    _validate_measurements(measurements, actions_executed)
    _validate_behavioral_event_timeline(record, measurements)
    return measurements


def derive_failure_taxonomy(record: dict[str, Any], measurements: dict[str, Any] | None = None) -> str:
    """Apply the frozen taxonomy precedence to a valid behavioral record."""

    measurements = measurements or derive_measurements(record)
    if record["requested_success"]:
        return "correct"
    if not measurements["verified_pickup"]:
        return "pick_failed"
    if measurements["final_opposite_region_sustained"]:
        return "wrong_side"
    if measurements["final_requested_region_sustained"] and not measurements["final_detached_release"]:
        return "release_failed"
    return "transport_failed"


def _validate_behavioral_header(record: dict[str, Any]) -> None:
    if record.get("schema_version") != BEHAVIORAL_SCHEMA_VERSION:
        _fail("unexpected behavioral schema_version")
    if record.get("record_type") != "behavioral_episode":
        _fail("behavioral record_type must be behavioral_episode")
    if _require_bool(record.get("behavioral_result_valid"), "behavioral_result_valid") is not True:
        _fail("behavioral records require behavioral_result_valid=true")
    if record.get("arena") not in ARENAS:
        _fail(f"arena must be one of {sorted(ARENAS)}")
    if record.get("requested_relation") not in {"left", "right"}:
        _fail("requested_relation must be left or right")
    _require_bool(record.get("requested_success"), "requested_success")
    if not isinstance(record.get("failure_stage"), str) or not record["failure_stage"].strip():
        _fail("legacy failure_stage must be a non-empty string")
    if record.get("frozen_failure_stage") != record["failure_stage"]:
        _fail("frozen_failure_stage must preserve the exact legacy failure_stage")
    taxonomy = record.get("failure_taxonomy")
    if taxonomy not in _frozen_contract()["precedence"]:
        _fail(f"failure_taxonomy must be one of {sorted(TAXONOMY)}")


def _validate_measurements(measurements: dict[str, Any], actions_executed: int) -> None:
    for key in (
        "signed_final_lateral_offset_m", "requested_final_lateral_margin_m",
        "final_requested_signed_margin_m", "final_opposite_signed_margin_m",
        "maximum_requested_signed_margin_m", "object_path_length_m", "max_object_lift_m",
        "maximum_pickup_height_m", "wall_time_s",
    ):
        _require_number(measurements[key], f"measurements.{key}")
    if measurements["object_path_length_m"] < 0 or measurements["max_object_lift_m"] < 0:
        _fail("path length and maximum lift must be non-negative")
    for key in (
        "first_verified_pickup_step", "first_requested_entry_step", "first_opposite_entry_step",
        "first_sustained_requested_entry_step", "first_sustained_opposite_entry_step", "first_contact_step",
    ):
        value = measurements[key]
        if value is not None and (type(value) is not int or not 0 <= value <= actions_executed):
            _fail(f"measurements.{key} is outside the episode action range")
    if measurements["verified_pickup"] != (measurements["first_verified_pickup_step"] is not None):
        _fail("verified_pickup must agree with first_verified_pickup_step")
    if measurements["first_sustained_requested_entry_step"] is not None and measurements["first_requested_entry_step"] is None:
        _fail("sustained requested entry requires a requested entry")
    if measurements["first_sustained_opposite_entry_step"] is not None and measurements["first_opposite_entry_step"] is None:
        _fail("sustained opposite entry requires an opposite entry")
    if measurements["first_requested_region_step"] != measurements["first_sustained_requested_entry_step"]:
        _fail("first_requested_region_step must use the frozen sustained-entry definition")
    if measurements["first_cone_or_native_region_entry_step"] != measurements["first_requested_entry_step"]:
        _fail("first_cone_or_native_region_entry_step must match first requested entry")
    if measurements["entry_kind"] != measurements["requested_entry_kind"]:
        _fail("entry_kind must match requested_entry_kind")
    if measurements["executed_action_count"] != actions_executed:
        _fail("executed_action_count must match actions_executed")
    if measurements["episode_length_steps"] != actions_executed:
        _fail("episode_length_steps must count executed actions, not state samples")
    if measurements["scored_state_sample_count"] != actions_executed + 1:
        _fail("scored_state_sample_count must include the initial state sample")
    contact_status = measurements["first_contact_status"]
    if contact_status not in _frozen_contract()["contact_statuses"]:
        _fail("first_contact_status is invalid")
    if contact_status == "observed" and measurements["first_contact_step"] is None:
        _fail("observed first_contact_status requires first_contact_step")
    if contact_status != "observed" and measurements["first_contact_step"] is not None:
        _fail("non-observed contact status requires null first_contact_step")
    reason = measurements["first_contact_unavailable_reason"]
    if contact_status == "instrumentation_unavailable":
        _require_string(reason, "measurements.first_contact_unavailable_reason")
    elif reason is not None:
        _fail("first_contact_unavailable_reason is only valid for instrumentation_unavailable")


def validate_behavioral_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a canonical behavioral record with derived measurements."""

    measurements = derive_measurements(record)
    expected = derive_failure_taxonomy(record, measurements)
    if record["failure_taxonomy"] != expected:
        _fail(f"failure_taxonomy {record['failure_taxonomy']!r} disagrees with frozen precedence {expected!r}")
    expected_stage = derive_frozen_failure_stage(record, measurements)
    if record["frozen_failure_stage"] != expected_stage:
        _fail(
            f"frozen_failure_stage {record['frozen_failure_stage']!r} disagrees "
            f"with the v2 arena classifier {expected_stage!r}"
        )
    initial_state_sha256 = _require_string(
        record.get("initial_state_sha256"), "initial_state_sha256"
    )
    expected_initial_state_sha256 = derive_initial_state_sha256(record)
    if initial_state_sha256 != expected_initial_state_sha256:
        _fail("initial_state_sha256 does not match the retained initial physical state")
    normalized = dict(record)
    normalized["measurements"] = measurements
    return normalized


def validate_infrastructure_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one separately-accounted technical or partial attempt."""

    record = _require_mapping(record, "record")
    if record.get("schema_version") != INFRASTRUCTURE_SCHEMA_VERSION:
        _fail("unexpected infrastructure schema_version")
    if record.get("record_type") != "infrastructure_attempt":
        _fail("infrastructure record_type must be infrastructure_attempt")
    if _require_bool(record.get("behavioral_result_valid"), "behavioral_result_valid") is not False:
        _fail("infrastructure records require behavioral_result_valid=false")
    if record.get("classification") not in INFRASTRUCTURE_CLASSIFICATIONS:
        _fail(f"classification must be one of {sorted(INFRASTRUCTURE_CLASSIFICATIONS)}")
    if record.get("arena") not in ARENAS:
        _fail(f"arena must be one of {sorted(ARENAS)}")
    _validate_provenance(record, require_behavioral_artifacts=False)
    _require_string(record.get("stage"), "stage")
    _require_string(record.get("error"), "error")
    log_hash = _require_string(record.get("log_hash"), "log_hash")
    if not _SHA256_RE.fullmatch(log_hash):
        _fail("log_hash must be a lowercase SHA-256 hex digest")
    _require_bool(record.get("runtime_intervention"), "runtime_intervention")
    repair_attempt_id = record.get("repair_attempt_id")
    if repair_attempt_id is not None:
        _require_string(repair_attempt_id, "repair_attempt_id")
    timeline = record.get("event_timeline")
    if not isinstance(timeline, list) or not timeline:
        _fail("infrastructure event_timeline must be a non-empty array")
    previous = -1
    for index, raw_event in enumerate(timeline):
        event = _require_mapping(raw_event, f"event_timeline[{index}]")
        sequence = _require_int(event.get("sequence"), f"event_timeline[{index}].sequence", minimum=0)
        _require_string(event.get("stage"), f"event_timeline[{index}].stage")
        if sequence != previous + 1:
            _fail("infrastructure event_timeline sequences must be contiguous and ordered")
        previous = sequence
    if "failure_taxonomy" in record or "requested_success" in record:
        _fail("infrastructure records must not carry behavioral taxonomy or success")
    return dict(record)


def validate_raw_episode_record(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one parsed JSONL object, routing by its explicit record type."""

    record = _require_mapping(record, "record")
    kind = record.get("record_type")
    if kind == "behavioral_episode":
        return validate_behavioral_record(record)
    if kind == "infrastructure_attempt":
        return validate_infrastructure_record(record)
    _fail("record_type must be behavioral_episode or infrastructure_attempt")


def parse_jsonl_record(line: str) -> dict[str, Any]:
    """Parse and validate exactly one non-empty JSONL line."""

    if not isinstance(line, str) or not line.strip() or "\n" in line.rstrip("\r\n"):
        _fail("expected exactly one non-empty JSONL line")
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise EpisodeSchemaError(f"invalid JSONL record: {error.msg}") from error
    return validate_raw_episode_record(value)


def encode_jsonl_record(record: dict[str, Any]) -> str:
    """Validate and deterministically encode one record for JSONL output."""

    normalized = validate_raw_episode_record(record)
    return json.dumps(normalized, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Write JSONL, then emit its companion integrity manifest after close.

    A line cannot truthfully contain the hash or byte count of the still-open
    JSONL file that contains it.  The post-close batch manifest is therefore
    the only integrity record for ``raw_result_jsonl`` itself.
    """

    path = Path(path)
    manifest_path = path.with_name(path.name + ".manifest.json")
    if path.exists() or manifest_path.exists():
        _fail(f"refusing to overwrite retained JSONL evidence: {path}")
    normalized_records = [validate_raw_episode_record(record) for record in records]
    if not normalized_records:
        _fail("a JSONL batch must contain at least one record")
    study_ids: set[str] = set()
    schemas: set[str] = set()
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for normalized in normalized_records:
            handle.write(json.dumps(normalized, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
            study_ids.add(normalized["study_id"])
            schemas.add(normalized["schema_version"])
    if len(study_ids) != 1:
        _fail("a JSONL batch must contain exactly one study_id")
    manifest = {
        "schema_version": "vla-wam-shared-v3-jsonl-batch-manifest-v1",
        "study_id": next(iter(study_ids)),
        "jsonl_path": str(path),
        "jsonl_sha256": _sha256_file(path),
        "jsonl_bytes": path.stat().st_size,
        "row_count": len(normalized_records),
        "record_schema_versions": sorted(schemas),
    }
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n")
    return manifest
