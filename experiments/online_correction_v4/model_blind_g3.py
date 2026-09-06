"""Prospective contracts for horizontal model-blind motion/feasibility gate G3."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from experiments.online_correction_v4.motion import ReferenceMotionController

PLAN_SCHEMA = "v4-horizontal-g3-plan-v1"
PATH_RECEIPT_SCHEMA = "v4-horizontal-g3-path-seed-receipt-v1"
SCRIPTED_RECEIPT_SCHEMA = "v4-horizontal-g3-scripted-check-receipt-v1"
AGGREGATE_SCHEMA = "v4-horizontal-g3-aggregate-receipt-v1"
SCRIPTED_CHECK_KINDS = ("stationary", "moving")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HORIZONTAL_GOALS = ("left", "right", "front", "behind")
PATH_SCENARIOS = (
    "original_sham",
    "destination_static",
    "move_stop",
    "slow_drift",
    "fast_drift",
    "reversal",
)
HORIZONTAL_PATH_CHECKS_PER_SEED = len(HORIZONTAL_GOALS) * len(PATH_SCENARIOS)
REFERENCE_POSITIONS = ("original", "midpoint", "endpoint")
PATH_SAMPLE_INTERVAL_S = 0.02
HORIZONTAL_STATIONARY_CHECK_COUNT = 9 * 4 * 3
HORIZONTAL_MOVING_CHECK_COUNT = 4
HORIZONTAL_SCRIPTED_CHECK_COUNT = (
    HORIZONTAL_STATIONARY_CHECK_COUNT + HORIZONTAL_MOVING_CHECK_COUNT
)


class G3GateError(ValueError):
    """Raised when a G3 plan or receipt is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise G3GateError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def expected_path_check_keys() -> tuple[tuple[str, str], ...]:
    return tuple(
        (goal, scenario)
        for goal in HORIZONTAL_GOALS
        for scenario in PATH_SCENARIOS
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise G3GateError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G3GateError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise G3GateError(f"{label} must be finite")
    return number


def _require_positive_finite(value: Any, label: str) -> float:
    number = _require_finite_number(value, label)
    if number <= 0:
        raise G3GateError(f"{label} must be positive")
    return number


def _require_non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise G3GateError(f"{label} must be a non-negative integer")
    if value < 0:
        raise G3GateError(f"{label} must be non-negative")
    return value


def _require_positive_int(value: Any, label: str) -> int:
    count = _require_non_negative_int(value, label)
    if count <= 0:
        raise G3GateError(f"{label} must be positive")
    return count


def _require_bool(value: Any, label: str) -> bool:
    if value is not True and value is not False:
        raise G3GateError(f"{label} must be a boolean")
    return value


def _require_evidence_identity(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G3GateError(f"{label} evidence is missing")
    path = value.get("path")
    digest = _require_sha256(value.get("sha256"), f"{label}.sha256")
    size = _require_positive_int(value.get("bytes"), f"{label}.bytes")
    if not isinstance(path, str) or not path.strip():
        raise G3GateError(f"{label}.path must be a non-empty string")
    return {"path": path, "sha256": digest, "bytes": size}


def _require_direction_coefficients(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise G3GateError(f"{label} must be a 2-vector")
    return [float(_require_finite_number(item, f"{label}[{index}]")) for index, item in enumerate(value)]


def _path_check_passes(check: Mapping[str, Any]) -> bool:
    return (
        check.get("collision_free") is True
        and check.get("support_valid") is True
        and check.get("reachable_workspace") is True
        and check.get("legal_goal_nonempty") is True
        and check.get("reference_robot_contact") is False
        and check.get("unmodeled_collision") is False
    )


def _scripted_check_passes(check: Mapping[str, Any]) -> bool:
    return (
        check.get("grasped") is True
        and check.get("transported") is True
        and check.get("released") is True
        and check.get("stably_placed") is True
        and check.get("goal_satisfied") is True
    )


def _plan_scale_candidates(plan: Mapping[str, Any]) -> tuple[float, ...]:
    selection = plan.get("scale_selection")
    if not isinstance(selection, Mapping):
        raise G3GateError("G3 plan lacks scale selection")
    raw = selection.get("candidate_scales_descending")
    if not isinstance(raw, list) or not raw:
        raise G3GateError("G3 plan lacks descending scale candidates")
    return tuple(_require_positive_finite(item, "scale candidate") for item in raw)


def _plan_nominal_displacement_m(plan: Mapping[str, Any]) -> float:
    selection = plan.get("scale_selection")
    if not isinstance(selection, Mapping):
        raise G3GateError("G3 plan lacks scale selection")
    return _require_positive_finite(
        selection.get("nominal_displacement_m"),
        "nominal displacement",
    )


def _plan_counterbalance_for_seed(
    plan: Mapping[str, Any], environment_seed: int
) -> dict[str, Any]:
    raw = plan.get("counterbalance_by_env_seed")
    if not isinstance(raw, Mapping):
        raise G3GateError("G3 plan lacks counterbalance index")
    counterbalance = raw.get(str(environment_seed))
    if not isinstance(counterbalance, Mapping):
        raise G3GateError(f"G3 plan lacks counterbalance for seed {environment_seed}")
    return dict(counterbalance)


def _plan_direction_coefficients_for_seed(
    plan: Mapping[str, Any], environment_seed: int
) -> dict[str, list[float]]:
    raw = plan.get("direction_task_coefficients_by_env_seed")
    if not isinstance(raw, Mapping):
        raise G3GateError("G3 plan lacks direction task coefficients")
    directions = raw.get(str(environment_seed))
    if not isinstance(directions, Mapping):
        raise G3GateError(
            f"G3 plan lacks direction task coefficients for seed {environment_seed}"
        )
    result: dict[str, list[float]] = {}
    for goal in HORIZONTAL_GOALS:
        result[goal] = _require_direction_coefficients(
            directions.get(goal),
            f"direction_task_coefficients[{environment_seed}][{goal}]",
        )
    return result


def _require_plan_receipt_identity(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise G3GateError(f"{label} is missing")
    path = value.get("path")
    digest = _require_sha256(value.get("sha256"), f"{label}.sha256")
    if not isinstance(path, str) or not path.strip():
        raise G3GateError(f"{label}.path must be a non-empty string")
    return {"path": path, "sha256": digest}


def _validate_path_check_order(checks: Sequence[Mapping[str, Any]]) -> None:
    expected = expected_path_check_keys()
    if len(checks) != HORIZONTAL_PATH_CHECKS_PER_SEED:
        raise G3GateError(
            f"path receipt must contain exactly {HORIZONTAL_PATH_CHECKS_PER_SEED} checks"
        )
    seen: set[tuple[str, str]] = set()
    for index, (expected_goal, expected_scenario) in enumerate(expected):
        check = checks[index]
        goal = check.get("goal")
        scenario = check.get("scenario")
        if goal != expected_goal or scenario != expected_scenario:
            raise G3GateError(
                "path receipt checks are missing or out of declared order"
            )
        key = (goal, scenario)
        if key in seen:
            raise G3GateError("path receipt contains duplicate checks")
        seen.add(key)


def _compile_path_check(
    *,
    goal: str,
    scenario: str,
    displacement_m: float,
    direction_task_coefficients: list[float],
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    planned_duration_s = _require_positive_finite(
        observation.get("planned_duration_s"),
        f"{goal}/{scenario} planned_duration_s",
    )
    sample_interval_s = _require_positive_finite(
        observation.get("sample_interval_s"),
        f"{goal}/{scenario} sample_interval_s",
    )
    if sample_interval_s > PATH_SAMPLE_INTERVAL_S + 1e-12:
        raise G3GateError(
            f"{goal}/{scenario} sample_interval_s exceeds PATH_SAMPLE_INTERVAL_S"
        )
    sample_count = _require_positive_int(
        observation.get("sample_count"),
        f"{goal}/{scenario} sample_count",
    )
    predicates = {
        "collision_free": _require_bool(
            observation.get("collision_free"), f"{goal}/{scenario} collision_free"
        ),
        "support_valid": _require_bool(
            observation.get("support_valid"), f"{goal}/{scenario} support_valid"
        ),
        "reachable_workspace": _require_bool(
            observation.get("reachable_workspace"),
            f"{goal}/{scenario} reachable_workspace",
        ),
        "legal_goal_nonempty": _require_bool(
            observation.get("legal_goal_nonempty"),
            f"{goal}/{scenario} legal_goal_nonempty",
        ),
        "reference_robot_contact": _require_bool(
            observation.get("reference_robot_contact"),
            f"{goal}/{scenario} reference_robot_contact",
        ),
        "unmodeled_collision": _require_bool(
            observation.get("unmodeled_collision"),
            f"{goal}/{scenario} unmodeled_collision",
        ),
    }
    reasons = observation.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise G3GateError(f"{goal}/{scenario} reasons must be a string list")
    compiled = {
        "goal": goal,
        "scenario": scenario,
        "direction_task_coefficients": list(direction_task_coefficients),
        "displacement_m": displacement_m,
        "planned_duration_s": planned_duration_s,
        "sample_interval_s": sample_interval_s,
        "sample_count": sample_count,
        "measured_pose_evidence": _require_evidence_identity(
            observation.get("measured_pose_evidence"),
            f"{goal}/{scenario} measured_pose_evidence",
        ),
        "reference_pose_evidence": _require_evidence_identity(
            observation.get("reference_pose_evidence"),
            f"{goal}/{scenario} reference_pose_evidence",
        ),
        **predicates,
        "reasons": list(reasons),
    }
    passed = _path_check_passes(compiled)
    if observation.get("passed") is not None:
        declared = _require_bool(observation.get("passed"), f"{goal}/{scenario} passed")
        if declared != passed:
            raise G3GateError(
                f"{goal}/{scenario} passed disagrees with predicate conjunction"
            )
    compiled["passed"] = passed
    if not passed and not reasons:
        raise G3GateError(f"{goal}/{scenario} failure lacks reasons")
    return compiled


def compile_path_seed_receipt(
    *,
    plan: Mapping[str, Any],
    plan_receipt: Mapping[str, Any],
    environment_seed: int,
    scale: float,
    check_observations: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    validate_plan_payload(plan)
    if type(environment_seed) is not int:
        raise G3GateError("environment_seed must be an integer")
    scale_value = _require_positive_finite(scale, "scale")
    candidates = _plan_scale_candidates(plan)
    if scale_value not in candidates:
        raise G3GateError("scale is not a registered G3 candidate")
    nominal = _plan_nominal_displacement_m(plan)
    displacement_m = nominal * scale_value
    counterbalance = _plan_counterbalance_for_seed(plan, environment_seed)
    directions = _plan_direction_coefficients_for_seed(plan, environment_seed)
    pinned_plan = _require_plan_receipt_identity(plan_receipt, "plan_receipt")
    observations = list(check_observations)
    if len(observations) != HORIZONTAL_PATH_CHECKS_PER_SEED:
        raise G3GateError(
            f"path seed receipt requires exactly {HORIZONTAL_PATH_CHECKS_PER_SEED} observations"
        )
    checks: list[dict[str, Any]] = []
    for (goal, scenario), observation in zip(expected_path_check_keys(), observations):
        if not isinstance(observation, Mapping):
            raise G3GateError(f"{goal}/{scenario} observation must be a mapping")
        checks.append(
            _compile_path_check(
                goal=goal,
                scenario=scenario,
                displacement_m=displacement_m,
                direction_task_coefficients=directions[goal],
                observation=observation,
            )
        )
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": PATH_RECEIPT_SCHEMA,
        "campaign_id": plan.get("campaign_id"),
        "fixture_id": plan.get("fixture_id"),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "plan_receipt": pinned_plan,
        "scale": scale_value,
        "displacement_m": displacement_m,
        "environment_seed": environment_seed,
        "counterbalance": counterbalance,
        "direction_task_coefficients_by_goal": directions,
        "check_order": ["goal_declared_order", "scenario_declared_order"],
        "check_count": HORIZONTAL_PATH_CHECKS_PER_SEED,
        "checks": checks,
        "passed": passed,
        "passed_check_count": sum(1 for check in checks if check["passed"]),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
    }


def validate_path_seed_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
) -> None:
    _require(
        receipt.get("schema_version") == PATH_RECEIPT_SCHEMA,
        "path seed receipt schema differs",
    )
    _require(receipt.get("campaign_id") == "online_correction_v4", "campaign differs")
    _require(receipt.get("fixture_id") == "horizontal", "fixture differs")
    _require(receipt.get("model_request_count") == 0, "path receipt records model requests")
    _require(
        receipt.get("behavioral_episode_count") == 0,
        "path receipt records behavioral episodes",
    )
    pinned_plan = _require_plan_receipt_identity(receipt.get("plan_receipt"), "plan_receipt")
    scale = _require_positive_finite(receipt.get("scale"), "scale")
    displacement_m = _require_positive_finite(receipt.get("displacement_m"), "displacement_m")
    environment_seed = receipt.get("environment_seed")
    if type(environment_seed) is not int:
        raise G3GateError("environment_seed must be an integer")
    counterbalance = receipt.get("counterbalance")
    if not isinstance(counterbalance, Mapping):
        raise G3GateError("path receipt lacks counterbalance")
    directions = receipt.get("direction_task_coefficients_by_goal")
    if not isinstance(directions, Mapping):
        raise G3GateError("path receipt lacks direction task coefficients")
    for goal in HORIZONTAL_GOALS:
        _require_direction_coefficients(
            directions.get(goal),
            f"direction_task_coefficients_by_goal[{goal}]",
        )
    if receipt.get("check_count") != HORIZONTAL_PATH_CHECKS_PER_SEED:
        raise G3GateError("path receipt check_count differs")
    if receipt.get("check_order") != [
        "goal_declared_order",
        "scenario_declared_order",
    ]:
        raise G3GateError("path receipt check_order differs")
    checks_raw = receipt.get("checks")
    if not isinstance(checks_raw, list):
        raise G3GateError("path receipt lacks checks")
    checks = [check for check in checks_raw if isinstance(check, Mapping)]
    _validate_path_check_order(checks)
    for check in checks:
        goal = check.get("goal")
        scenario = check.get("scenario")
        label = f"{goal}/{scenario}"
        _require_positive_finite(check.get("displacement_m"), f"{label} displacement_m")
        if abs(float(check["displacement_m"]) - displacement_m) > 1e-9:
            raise G3GateError(f"{label} displacement_m differs from receipt")
        expected_direction = _require_direction_coefficients(
            check.get("direction_task_coefficients"),
            f"{label} direction_task_coefficients",
        )
        if list(expected_direction) != list(directions[str(goal)]):
            raise G3GateError(f"{label} direction_task_coefficients differ from receipt")
        _require_positive_finite(check.get("planned_duration_s"), f"{label} planned_duration_s")
        sample_interval_s = _require_positive_finite(
            check.get("sample_interval_s"), f"{label} sample_interval_s"
        )
        if sample_interval_s > PATH_SAMPLE_INTERVAL_S + 1e-12:
            raise G3GateError(f"{label} sample_interval_s exceeds PATH_SAMPLE_INTERVAL_S")
        _require_positive_int(check.get("sample_count"), f"{label} sample_count")
        _require_evidence_identity(
            check.get("measured_pose_evidence"), f"{label} measured_pose_evidence"
        )
        _require_evidence_identity(
            check.get("reference_pose_evidence"), f"{label} reference_pose_evidence"
        )
        for field in (
            "collision_free",
            "support_valid",
            "reachable_workspace",
            "legal_goal_nonempty",
            "reference_robot_contact",
            "unmodeled_collision",
        ):
            _require_bool(check.get(field), f"{label} {field}")
        reasons = check.get("reasons")
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise G3GateError(f"{label} reasons must be a string list")
        passed = _require_bool(check.get("passed"), f"{label} passed")
        expected_passed = _path_check_passes(check)
        if passed != expected_passed:
            raise G3GateError(f"{label} passed disagrees with predicate conjunction")
        if not passed and not reasons:
            raise G3GateError(f"{label} failure lacks reasons")
    passed_count = sum(1 for check in checks if check.get("passed") is True)
    failed_count = sum(1 for check in checks if check.get("passed") is False)
    if receipt.get("passed_check_count") != passed_count:
        raise G3GateError("path receipt passed_check_count differs")
    if receipt.get("failed_check_count") != failed_count:
        raise G3GateError("path receipt failed_check_count differs")
    receipt_passed = _require_bool(receipt.get("passed"), "receipt passed")
    if receipt_passed != (failed_count == 0):
        raise G3GateError("path receipt passed disagrees with check outcomes")
    if plan is not None:
        validate_plan_payload(plan)
        _require(plan.get("campaign_id") == receipt.get("campaign_id"), "campaign binding differs")
        _require(plan.get("fixture_id") == receipt.get("fixture_id"), "fixture binding differs")
        if scale not in _plan_scale_candidates(plan):
            raise G3GateError("scale is not bound to the G3 plan")
        nominal = _plan_nominal_displacement_m(plan)
        if abs(displacement_m - nominal * scale) > 1e-9:
            raise G3GateError("displacement_m is not bound to plan scale selection")
        expected_counterbalance = _plan_counterbalance_for_seed(plan, environment_seed)
        if dict(counterbalance) != expected_counterbalance:
            raise G3GateError("counterbalance is not bound to the G3 plan")
        expected_directions = _plan_direction_coefficients_for_seed(plan, environment_seed)
        for goal in HORIZONTAL_GOALS:
            if list(directions[goal]) != expected_directions[goal]:
                raise G3GateError(
                    "direction_task_coefficients_by_goal is not bound to the G3 plan"
                )


def compile_scripted_check_receipt(
    *,
    check_kind: str,
    environment_seed: int,
    goal: str,
    reference_position: str,
    scale: float,
    displacement_m: float,
    observation: Mapping[str, Any],
) -> dict[str, Any]:
    if check_kind not in SCRIPTED_CHECK_KINDS:
        raise G3GateError("scripted check kind differs")
    if type(environment_seed) is not int:
        raise G3GateError("environment_seed must be an integer")
    if goal not in HORIZONTAL_GOALS:
        raise G3GateError("scripted check goal differs")
    if reference_position not in REFERENCE_POSITIONS:
        raise G3GateError("scripted check reference_position differs")
    scale_value = _require_positive_finite(scale, "scale")
    displacement_value = _require_positive_finite(displacement_m, "displacement_m")
    stages = {
        "grasped": _require_bool(observation.get("grasped"), "grasped"),
        "transported": _require_bool(observation.get("transported"), "transported"),
        "released": _require_bool(observation.get("released"), "released"),
        "stably_placed": _require_bool(observation.get("stably_placed"), "stably_placed"),
        "goal_satisfied": _require_bool(
            observation.get("goal_satisfied"), "goal_satisfied"
        ),
    }
    reasons = observation.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise G3GateError("reasons must be a string list")
    compiled = {
        "schema_version": SCRIPTED_RECEIPT_SCHEMA,
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "check_kind": check_kind,
        "environment_seed": environment_seed,
        "goal": goal,
        "reference_position": reference_position,
        "scale": scale_value,
        "displacement_m": displacement_value,
        **stages,
        "evidence": _require_evidence_identity(observation.get("evidence"), "evidence"),
        "reasons": list(reasons),
    }
    passed = _scripted_check_passes(compiled)
    if observation.get("passed") is not None:
        declared = _require_bool(observation.get("passed"), "passed")
        if declared != passed:
            raise G3GateError("passed disagrees with stage predicate conjunction")
    compiled["passed"] = passed
    if not passed and not reasons:
        raise G3GateError("scripted check failure lacks reasons")
    return compiled


def validate_scripted_check_receipt(receipt: Mapping[str, Any]) -> None:
    _require(
        receipt.get("schema_version") == SCRIPTED_RECEIPT_SCHEMA,
        "scripted receipt schema differs",
    )
    _require(receipt.get("campaign_id") == "online_correction_v4", "campaign differs")
    _require(receipt.get("fixture_id") == "horizontal", "fixture differs")
    _require(
        receipt.get("model_request_count") == 0,
        "scripted receipt records model requests",
    )
    _require(
        receipt.get("behavioral_episode_count") == 0,
        "scripted receipt records behavioral episodes",
    )
    check_kind = receipt.get("check_kind")
    if check_kind not in SCRIPTED_CHECK_KINDS:
        raise G3GateError("scripted check kind differs")
    environment_seed = receipt.get("environment_seed")
    if type(environment_seed) is not int:
        raise G3GateError("environment_seed must be an integer")
    goal = receipt.get("goal")
    if goal not in HORIZONTAL_GOALS:
        raise G3GateError("scripted check goal differs")
    reference_position = receipt.get("reference_position")
    if reference_position not in REFERENCE_POSITIONS:
        raise G3GateError("scripted check reference_position differs")
    _require_positive_finite(receipt.get("scale"), "scale")
    _require_positive_finite(receipt.get("displacement_m"), "displacement_m")
    for field in (
        "grasped",
        "transported",
        "released",
        "stably_placed",
        "goal_satisfied",
    ):
        _require_bool(receipt.get(field), field)
    _require_evidence_identity(receipt.get("evidence"), "evidence")
    reasons = receipt.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
        raise G3GateError("reasons must be a string list")
    passed = _require_bool(receipt.get("passed"), "passed")
    expected_passed = _scripted_check_passes(receipt)
    if passed != expected_passed:
        raise G3GateError("passed disagrees with stage predicate conjunction")
    if not passed and not reasons:
        raise G3GateError("scripted check failure lacks reasons")


def select_extreme_reset_seeds(
    *,
    resets_by_env_seed: Mapping[str, Any],
    counterbalance_by_env_seed: Mapping[int, Mapping[str, Any]],
) -> tuple[int, ...]:
    """Select eight directional extrema, balanced two per counterbalance state."""
    seeds = sorted(int(seed) for seed in resets_by_env_seed)
    _require(len(seeds) >= 9, "horizontal G3 requires at least nine reset seeds")
    canonical = seeds[0]
    selected: list[int] = []
    for sector in range(8):
        angle = 2.0 * math.pi * sector / 8.0
        axis = (math.cos(angle), math.sin(angle))
        required_state = sector % 4
        candidates: list[tuple[float, int]] = []
        for seed in seeds:
            if seed == canonical or seed in selected:
                continue
            counterbalance = counterbalance_by_env_seed.get(seed)
            reset = resets_by_env_seed.get(str(seed))
            if not isinstance(counterbalance, Mapping) or not isinstance(reset, Mapping):
                continue
            if counterbalance.get("state_index") != required_state:
                continue
            jitter = reset.get("jitter_robot_base_xy_m")
            if not isinstance(jitter, list) or len(jitter) != 2:
                raise G3GateError(f"reset {seed} lacks registered xy jitter")
            score = float(jitter[0]) * axis[0] + float(jitter[1]) * axis[1]
            candidates.append((score, seed))
        _require(
            bool(candidates),
            f"no unused state-{required_state} reset for extreme sector {sector}",
        )
        selected.append(max(candidates, key=lambda item: (item[0], -item[1]))[1])
    _require(len(set(selected)) == 8, "extreme reset selection is not unique")
    states = [
        int(counterbalance_by_env_seed[seed]["state_index"]) for seed in selected
    ]
    _require(
        all(states.count(state) == 2 for state in range(4)),
        "extreme reset selection must contain two seeds per state",
    )
    return (canonical, *selected)


def build_counterbalance_index(
    queue_rows: Iterable[Mapping[str, Any]],
    *,
    expected_env_seeds: Iterable[int],
) -> dict[int, dict[str, Any]]:
    expected = set(int(seed) for seed in expected_env_seeds)
    result: dict[int, dict[str, Any]] = {}
    for row in queue_rows:
        if row.get("family") != "C1" or row.get("fixture") != "horizontal":
            continue
        seed = row.get("env_seed")
        counterbalance = row.get("counterbalance")
        if type(seed) is not int or seed not in expected:
            continue
        if not isinstance(counterbalance, Mapping):
            raise G3GateError(f"C1 row for seed {seed} lacks counterbalance")
        frozen = {
            "block_id": row.get("block_id"),
            "state_index": counterbalance.get("state_index"),
            "physical_translation_sign": counterbalance.get(
                "physical_translation_sign"
            ),
            "event_phase_fraction": counterbalance.get("event_phase_fraction"),
        }
        prior = result.setdefault(seed, frozen)
        if prior != frozen:
            raise G3GateError(f"C1 counterbalance differs within seed {seed}")
    missing = sorted(expected - set(result))
    _require(not missing, f"C1 counterbalance index misses seeds: {missing[:5]}")
    for seed, row in result.items():
        _require(type(row["block_id"]) is int, f"seed {seed} block_id is invalid")
        _require(
            row["state_index"] in (0, 1, 2, 3),
            f"seed {seed} state_index is invalid",
        )
        _require(
            row["physical_translation_sign"] in (-1, 1),
            f"seed {seed} physical translation sign is invalid",
        )
    return result


def build_plan_payload(
    *,
    source_identity: Mapping[str, Any],
    g2_prerequisite: Mapping[str, Any],
    reset_registry: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
    scale_candidates: Iterable[float],
    nominal_displacement_m: float,
    minimum_shrinking_area_fraction: float,
) -> dict[str, Any]:
    _validate_g2_prerequisite(g2_prerequisite)
    resets = reset_registry.get("resets_by_env_seed")
    _require(isinstance(resets, Mapping) and resets, "reset registry has no resets")
    seeds = tuple(sorted(int(seed) for seed in resets))
    counterbalance = build_counterbalance_index(
        queue_rows, expected_env_seeds=seeds
    )
    extremes = select_extreme_reset_seeds(
        resets_by_env_seed=resets,
        counterbalance_by_env_seed=counterbalance,
    )
    scales = tuple(float(value) for value in scale_candidates)
    _require(
        scales == tuple(sorted(scales, reverse=True))
        and all(value > 0 for value in scales),
        "scale candidates must be positive and descending",
    )
    _require(
        math.isfinite(nominal_displacement_m) and nominal_displacement_m > 0,
        "nominal displacement must be positive",
    )
    directions: dict[str, Any] = {}
    for seed in seeds:
        sign = int(counterbalance[seed]["physical_translation_sign"])
        directions[str(seed)] = {
            goal: list(
                ReferenceMotionController.displacement_vector(
                    goal=goal,
                    fixture="horizontal",
                    physical_sign=sign,
                )
            )
            for goal in HORIZONTAL_GOALS
        }
    path_checks_per_scale = len(seeds) * len(HORIZONTAL_GOALS) * len(PATH_SCENARIOS)
    _require(path_checks_per_scale == 3072, "horizontal path count differs")
    _require(
        HORIZONTAL_SCRIPTED_CHECK_COUNT == 112,
        "horizontal scripted count differs",
    )
    return {
        "schema_version": PLAN_SCHEMA,
        "campaign_id": "online_correction_v4",
        "fixture_id": "horizontal",
        "status": "model_blind_candidate_not_released_for_inference",
        "plan_status": "ready_for_live_g3_execution",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_identity": dict(source_identity),
        "g2_prerequisite": dict(g2_prerequisite),
        "scale_selection": {
            "candidate_scales_descending": list(scales),
            "nominal_displacement_m": nominal_displacement_m,
            "candidate_displacements_m": [
                nominal_displacement_m * scale for scale in scales
            ],
            "selection_rule": (
                "first descending candidate with complete geometric path, "
                "goal-area, and scripted-controller checks passing"
            ),
            "minimum_shrinking_area_fraction": minimum_shrinking_area_fraction,
            "policy_outcome_used": False,
        },
        "registered_reset_count": len(seeds),
        "registered_env_seed_count": len(seeds),
        "registered_env_seeds": list(seeds),
        "counterbalance_by_env_seed": {
            str(seed): counterbalance[seed] for seed in seeds
        },
        "direction_task_coefficients_by_env_seed": directions,
        "path_sweep": {
            "goals": list(HORIZONTAL_GOALS),
            "scenarios": list(PATH_SCENARIOS),
            "sample_interval_s": PATH_SAMPLE_INTERVAL_S,
            "cartesian_order": [
                "scale_descending",
                "environment_seed_ascending",
                "goal_declared_order",
                "scenario_declared_order",
            ],
            "checks_per_scale": path_checks_per_scale,
            "maximum_checks_across_scale_ladder": path_checks_per_scale
            * len(scales),
            "requires_live_collision_and_contact_evidence": True,
        },
        "scripted_controller": {
            "reset_selection_rule": (
                "minimum seed canonical plus one directional xy-jitter extremum "
                "per 45-degree sector, constrained to two seeds per state_index"
            ),
            "reset_env_seeds": list(extremes),
            "stationary": {
                "goals": list(HORIZONTAL_GOALS),
                "reference_positions": list(REFERENCE_POSITIONS),
                "checks_per_scale": HORIZONTAL_STATIONARY_CHECK_COUNT,
            },
            "moving": {
                "canonical_env_seed": extremes[0],
                "goals": list(HORIZONTAL_GOALS),
                "scenario": "move_stop",
                "checks_per_scale": HORIZONTAL_MOVING_CHECK_COUNT,
            },
            "checks_per_final_geometry_candidate": HORIZONTAL_SCRIPTED_CHECK_COUNT,
            "candidate_attempt_rule": (
                "run scripted checks only after that scale passes exhaustive path "
                "and information checks; preserve every rejected candidate receipt"
            ),
        },
        "release_boundary": (
            "This formula-closed plan executes no model. G3 remains blocked until "
            "live collision, geometry, and scripted-controller receipts select "
            "one scale."
        ),
    }


def _validate_g2_prerequisite(prerequisite: Mapping[str, Any]) -> None:
    _require(
        prerequisite.get("schema_version")
        == "v4-horizontal-g2-aggregate-receipt-v1",
        "G3 plan requires the horizontal G2 aggregate schema",
    )
    _require(
        prerequisite.get("status") == "passed"
        and prerequisite.get("passed") is True,
        "G3 plan requires a passing G2 aggregate",
    )
    _require(
        prerequisite.get("axis_review_passed") is True,
        "G3 plan requires the passing G2 axis review",
    )
    _require(
        prerequisite.get("expected_seed_count") == 128
        and prerequisite.get("observed_seed_count") == 128,
        "G3 plan requires complete G2 seed coverage",
    )
    _require(
        prerequisite.get("model_request_count") == 0
        and prerequisite.get("behavioral_episode_count") == 0,
        "G2 prerequisite is not model-blind",
    )
    receipt = prerequisite.get("receipt")
    _require(
        isinstance(receipt, Mapping)
        and isinstance(receipt.get("path"), str)
        and isinstance(receipt.get("sha256"), str)
        and len(receipt["sha256"]) == 64,
        "G3 plan lacks a hash-pinned G2 receipt",
    )


def validate_plan_payload(plan: Mapping[str, Any]) -> None:
    _require(plan.get("schema_version") == PLAN_SCHEMA, "G3 plan schema differs")
    _require(plan.get("fixture_id") == "horizontal", "G3 plan fixture differs")
    _require(plan.get("model_request_count") == 0, "G3 plan records model requests")
    _require(
        plan.get("behavioral_episode_count") == 0,
        "G3 plan records behavioral episodes",
    )
    _require(
        plan.get("plan_status") == "ready_for_live_g3_execution",
        "G3 plan is not authorized for live model-blind execution",
    )
    prerequisite = plan.get("g2_prerequisite")
    _require(isinstance(prerequisite, Mapping), "G3 plan lacks G2 prerequisite")
    _validate_g2_prerequisite(prerequisite)
    _require(plan.get("registered_reset_count") == 128, "G3 reset count differs")
    path = plan.get("path_sweep")
    scripted = plan.get("scripted_controller")
    _require(isinstance(path, Mapping), "G3 plan lacks path sweep")
    _require(isinstance(scripted, Mapping), "G3 plan lacks scripted checks")
    _require(path.get("checks_per_scale") == 3072, "G3 path count differs")
    _require(
        scripted.get("checks_per_final_geometry_candidate") == 112,
        "G3 scripted count differs",
    )
    reset_seeds = scripted.get("reset_env_seeds")
    _require(
        isinstance(reset_seeds, list)
        and len(reset_seeds) == 9
        and len(set(reset_seeds)) == 9,
        "G3 scripted reset selection differs",
    )
