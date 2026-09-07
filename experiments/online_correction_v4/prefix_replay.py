"""Fail-closed verification of C2 deterministic fresh-session prefix replay."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


class PrefixReplayError(ValueError):
    """Raised when two continuations do not establish a verified common prefix."""


@dataclass(frozen=True)
class PrefixReplayTolerance:
    position_m: float = 1e-4
    simulation_time_s: float = 1e-9

    def __post_init__(self) -> None:
        for label, value in (
            ("position_m", self.position_m),
            ("simulation_time_s", self.simulation_time_s),
        ):
            if not math.isfinite(value) or value < 0:
                raise PrefixReplayError(f"{label} must be finite and non-negative")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PrefixReplayError(f"{path}: expected a JSON object")
    return value


def _load_rows(attempt_dir: Path, name: str) -> list[dict[str, Any]]:
    payload = _load_object(attempt_dir / name)
    rows = payload.get("rows")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise PrefixReplayError(f"{attempt_dir / name}: rows are malformed")
    return rows


def _require_sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PrefixReplayError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _trigger_time(events: Sequence[Mapping[str, Any]]) -> float:
    triggers = [row for row in events if row.get("kind") == "trigger_eligible"]
    if len(triggers) != 1:
        raise PrefixReplayError("prefix attempt must contain exactly one natural trigger")
    value = triggers[0].get("sim_time")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PrefixReplayError("natural trigger time is invalid")
    trigger_time = float(value)
    for row in events:
        if row.get("kind") not in {"event_delivered", "event_observed"}:
            continue
        event_time = row.get("sim_time")
        if isinstance(event_time, (int, float)) and float(event_time) <= trigger_time + 1e-12:
            raise PrefixReplayError("intervention occurred before the common-prefix boundary")
    return trigger_time


def _fresh_session_projection(
    receipt: Mapping[str, Any],
    *,
    attempt_id: str,
) -> dict[str, Any]:
    if receipt.get("schema_version") != "v4-c2-fresh-session-attestation-v1":
        raise PrefixReplayError("fresh-session attestation schema differs")
    if receipt.get("attempt_id") != attempt_id:
        raise PrefixReplayError("fresh-session attestation attempt differs")
    for key in (
        "policy_started_fresh",
        "simulator_started_fresh",
        "policy_reset_before_prefix",
        "simulator_reset_before_prefix",
        "no_reused_hidden_session_state",
    ):
        if receipt.get(key) is not True:
            raise PrefixReplayError(f"fresh-session attestation lacks {key}")
    return {
        "policy_process_identity_sha256": _require_sha(
            receipt.get("policy_process_identity_sha256"),
            "policy process identity",
        ),
        "simulator_process_identity_sha256": _require_sha(
            receipt.get("simulator_process_identity_sha256"),
            "simulator process identity",
        ),
    }


def _request_projection(
    requests: Sequence[Mapping[str, Any]],
    *,
    trigger_time: float,
) -> list[dict[str, Any]]:
    submissions: dict[str, Mapping[str, Any]] = {}
    responses: dict[str, Mapping[str, Any]] = {}
    for row in requests:
        request_id = row.get("request_id")
        if not isinstance(request_id, str):
            raise PrefixReplayError("request row lacks request_id")
        if "observation_capture_time" in row:
            submissions[request_id] = row
        if "action_sha256" in row:
            responses[request_id] = row
    result: list[dict[str, Any]] = []
    for request_id, submission in submissions.items():
        capture_time = submission.get("observation_capture_time")
        if not isinstance(capture_time, (int, float)):
            raise PrefixReplayError("request observation time is invalid")
        if float(capture_time) > trigger_time + 1e-12:
            continue
        response = responses.get(request_id)
        if response is None:
            raise PrefixReplayError("common prefix contains an incomplete policy request")
        audit = response.get("policy_request_audit")
        if not isinstance(audit, Mapping):
            raise PrefixReplayError("request lacks complete policy wire audit")
        projected_audit = {
            key: audit.get(key)
            for key in (
                "request_index",
                "request_sampling_seed",
                "action_step_start",
                "observation_sha256",
                "prompt_sha256",
                "wire_request_sha256",
                "future_sha256",
            )
            if key in audit
        }
        _require_sha(projected_audit.get("wire_request_sha256"), "wire request")
        _require_sha(response.get("action_sha256"), "action chunk")
        result.append(
            {
                "request_id": request_id,
                "observation_id": submission.get("observation_id"),
                "observation_capture_time": float(capture_time),
                "submit_time": submission.get("submit_time"),
                "executed_action_count": submission.get("executed_action_count"),
                "action_sha256": response.get("action_sha256"),
                "generated_horizon": response.get("generated_horizon"),
                "policy_request_audit": projected_audit,
            }
        )
    if not result:
        raise PrefixReplayError("common prefix contains no completed policy request")
    return result


def _trajectory_prefix(
    rows: Sequence[Mapping[str, Any]],
    *,
    trigger_time: float,
) -> list[dict[str, Any]]:
    result = [
        dict(row)
        for row in rows
        if isinstance(row.get("simulation_time"), (int, float))
        and float(row["simulation_time"]) <= trigger_time + 1e-12
    ]
    if not result or result[-1].get("grasp_eligible") is not True:
        raise PrefixReplayError("trajectory does not reach the verified trigger boundary")
    for row in result:
        if not isinstance(row.get("object_state"), Mapping):
            raise PrefixReplayError("trajectory lacks prefix object state")
        if not isinstance(row.get("controller_state"), Mapping):
            raise PrefixReplayError("trajectory lacks prefix controller state")
        reference = row.get("reference_position_world")
        if not isinstance(reference, list) or len(reference) != 3:
            raise PrefixReplayError("trajectory lacks reference position")
    return result


def _compare_numeric(
    left: Any,
    right: Any,
    *,
    tolerance: float,
    label: str,
) -> None:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
        or not math.isfinite(float(left))
        or not math.isfinite(float(right))
        or abs(float(left) - float(right)) > tolerance + 1e-15
    ):
        raise PrefixReplayError(f"{label} differs beyond tolerance")


def _compare_vectors(
    left: Any,
    right: Any,
    *,
    tolerance: float,
    label: str,
) -> None:
    if (
        not isinstance(left, list)
        or not isinstance(right, list)
        or len(left) != len(right)
    ):
        raise PrefixReplayError(f"{label} vector shape differs")
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        _compare_numeric(
            left_value,
            right_value,
            tolerance=tolerance,
            label=f"{label}[{index}]",
        )


def _compare_trajectories(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    tolerance: PrefixReplayTolerance,
) -> None:
    if len(left) != len(right):
        raise PrefixReplayError("prefix trajectory lengths differ")
    for index, (left_row, right_row) in enumerate(zip(left, right)):
        for key in (
            "control_step",
            "commanded_action",
            "grasp_eligible",
            "detach_armed",
            "controller_state",
        ):
            if left_row.get(key) != right_row.get(key):
                raise PrefixReplayError(f"trajectory row {index} {key} differs")
        _compare_numeric(
            left_row.get("simulation_time"),
            right_row.get("simulation_time"),
            tolerance=tolerance.simulation_time_s,
            label=f"trajectory row {index} simulation_time",
        )
        _compare_numeric(
            left_row.get("reference_displacement_m"),
            right_row.get("reference_displacement_m"),
            tolerance=tolerance.position_m,
            label=f"trajectory row {index} reference displacement",
        )
        _compare_vectors(
            left_row.get("reference_position_world"),
            right_row.get("reference_position_world"),
            tolerance=tolerance.position_m,
            label=f"trajectory row {index} reference position",
        )
        left_state = left_row["object_state"]
        right_state = right_row["object_state"]
        assert isinstance(left_state, Mapping) and isinstance(right_state, Mapping)
        for key in ("object_position_world", "gripper_position_world"):
            _compare_vectors(
                left_state.get(key),
                right_state.get(key),
                tolerance=tolerance.position_m,
                label=f"trajectory row {index} {key}",
            )
        for key in ("initial_supported_z",):
            _compare_numeric(
                left_state.get(key),
                right_state.get(key),
                tolerance=tolerance.position_m,
                label=f"trajectory row {index} {key}",
            )
        for key in ("contact", "detached"):
            if left_state.get(key) != right_state.get(key):
                raise PrefixReplayError(f"trajectory row {index} {key} differs")


def verify_common_prefix(
    *,
    left_attempt_dir: Path,
    right_attempt_dir: Path,
    left_session_receipt_path: Path,
    right_session_receipt_path: Path,
    tolerance: PrefixReplayTolerance = PrefixReplayTolerance(),
) -> dict[str, Any]:
    """Verify one sham/movement pair within the same C2 policy and prompt."""
    left_episode = _load_object(left_attempt_dir / "episode.json")
    right_episode = _load_object(right_attempt_dir / "episode.json")
    left_attempt_id = str(left_episode.get("attempt_id"))
    right_attempt_id = str(right_episode.get("attempt_id"))
    if left_attempt_id == right_attempt_id:
        raise PrefixReplayError("fresh replay attempts must be distinct")
    for key in (
        "prefix_group_id",
        "env_seed",
        "policy_seed",
        "prompt_id",
        "prompt_sha256",
        "policy_id",
    ):
        if left_episode.get(key) != right_episode.get(key):
            raise PrefixReplayError(f"common-prefix identity field {key} differs")
    scenarios = {left_episode.get("scenario"), right_episode.get("scenario")}
    if scenarios != {"original_sham", "move_A"}:
        raise PrefixReplayError("C2 prefix pair must contain sham and move-A continuations")

    left_events = _load_rows(left_attempt_dir, "events.json")
    right_events = _load_rows(right_attempt_dir, "events.json")
    left_trigger = _trigger_time(left_events)
    right_trigger = _trigger_time(right_events)
    _compare_numeric(
        left_trigger,
        right_trigger,
        tolerance=tolerance.simulation_time_s,
        label="trigger time",
    )
    left_requests = _request_projection(
        _load_rows(left_attempt_dir, "requests.json"),
        trigger_time=left_trigger,
    )
    right_requests = _request_projection(
        _load_rows(right_attempt_dir, "requests.json"),
        trigger_time=right_trigger,
    )
    if left_requests != right_requests:
        raise PrefixReplayError("policy request/action histories differ before trigger")
    left_trajectory = _trajectory_prefix(
        _load_rows(left_attempt_dir, "trajectory.json"),
        trigger_time=left_trigger,
    )
    right_trajectory = _trajectory_prefix(
        _load_rows(right_attempt_dir, "trajectory.json"),
        trigger_time=right_trigger,
    )
    _compare_trajectories(left_trajectory, right_trajectory, tolerance=tolerance)

    left_session = _fresh_session_projection(
        _load_object(left_session_receipt_path),
        attempt_id=left_attempt_id,
    )
    right_session = _fresh_session_projection(
        _load_object(right_session_receipt_path),
        attempt_id=right_attempt_id,
    )
    for key in (
        "policy_process_identity_sha256",
        "simulator_process_identity_sha256",
    ):
        if left_session[key] == right_session[key]:
            raise PrefixReplayError(f"fresh replay reused {key}")

    identity = {
        "prefix_group_id": left_episode.get("prefix_group_id"),
        "env_seed": left_episode.get("env_seed"),
        "policy_seed": left_episode.get("policy_seed"),
        "prompt_id": left_episode.get("prompt_id"),
        "prompt_sha256": left_episode.get("prompt_sha256"),
        "policy_id": left_episode.get("policy_id"),
        "trigger_time_s": left_trigger,
        "request_history": left_requests,
        "trajectory": left_trajectory,
    }
    return {
        "schema_version": "v4-c2-common-prefix-verification-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C2",
        "status": "passed",
        "passed": True,
        "common_prefix_verification_mode": "deterministic_fresh_session_replay",
        "common_prefix_identity_hash_sha256": hashlib.sha256(
            canonical_json_bytes(identity)
        ).hexdigest(),
        "prefix_group_id": left_episode.get("prefix_group_id"),
        "trigger_time_s": left_trigger,
        "request_count": len(left_requests),
        "control_tick_count": len(left_trajectory),
        "tolerances": {
            "position_m": tolerance.position_m,
            "simulation_time_s": tolerance.simulation_time_s,
        },
        "fresh_session_attestations": {
            "left": left_session,
            "right": right_session,
        },
        "evidence": {
            "left_attempt_dir": str(left_attempt_dir),
            "right_attempt_dir": str(right_attempt_dir),
            "left_session_receipt": {
                "path": str(left_session_receipt_path),
                "sha256": sha256_file(left_session_receipt_path),
            },
            "right_session_receipt": {
                "path": str(right_session_receipt_path),
                "sha256": sha256_file(right_session_receipt_path),
            },
        },
    }
