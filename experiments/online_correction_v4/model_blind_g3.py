"""Prospective contracts for model-blind motion/feasibility gate G3."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from experiments.online_correction_v4.model_blind_g2 import aggregate_receipt_schema
from experiments.online_correction_v4.motion import ReferenceMotionController


def plan_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-plan-v1"


def path_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-path-seed-receipt-v1"


def path_scale_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-path-scale-receipt-v1"


def scripted_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-scripted-check-receipt-v1"


def scripted_seed_receipt_schema(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-scripted-seed-receipt-v1"


def aggregate_receipt_schema_g3(fixture_id: str) -> str:
    return f"v4-{fixture_id.replace('_', '-')}-g3-aggregate-receipt-v1"


PLAN_SCHEMA = plan_schema("horizontal")
PATH_RECEIPT_SCHEMA = path_receipt_schema("horizontal")
PATH_SCALE_RECEIPT_SCHEMA = path_scale_receipt_schema("horizontal")
SCRIPTED_RECEIPT_SCHEMA = scripted_receipt_schema("horizontal")
AGGREGATE_SCHEMA = aggregate_receipt_schema_g3("horizontal")

DEFAULT_GOAL_AREA_GATE_FIXTURES: tuple[str, ...] = ("horizontal", "reference_binding")


DEFAULT_MODEL_BLIND_G3_GEOMETRY: dict[str, Any] = {
    "extent_convention": "live_usd_world_aabb_projected_into_registered_task_frame",
    "supported_workspace_convention": (
        "live_table_top_aabb_eroded_by_target_projected_half_extents_and_edge_margin"
    ),
    "relation_clearance_m": 0.01,
    "support_edge_margin_m": 0.005,
    "active_contact_force_threshold_n": 0.05,
    "reference_pose_error_max_m": 0.002,
    "stationary_object_drift_max_m": 0.005,
    "path_sample_max_interval_s": 0.02,
    "robot_reference_contact_probe": (
        "full_robot_articulation_regex_against_reference_pair_sensor"
    ),
    "policy_outcome_used": False,
}


@dataclass(frozen=True)
class G3FixtureConfig:
    fixture_id: str
    counterbalance_family: str
    expected_seed_count: int
    goals: tuple[str, ...]


G3_FIXTURE_CONFIGS: dict[str, G3FixtureConfig] = {
    "horizontal": G3FixtureConfig(
        "horizontal", "C1", 128, ("left", "right", "front", "behind")
    ),
    "vertical": G3FixtureConfig("vertical", "C5", 64, ("above", "below")),
    "containment": G3FixtureConfig("containment", "C6", 64, ("inside",)),
    "object_pair": G3FixtureConfig(
        "object_pair", "C7", 64, ("left", "right", "front", "behind")
    ),
}

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


def g3_fixture_config(fixture_id: str) -> G3FixtureConfig:
    try:
        return G3_FIXTURE_CONFIGS[fixture_id]
    except KeyError as exc:
        raise G3GateError(f"unsupported G3 fixture: {fixture_id!r}") from exc


def g3_expected_seed_count(
    fixture_id: str,
    qualification_scope: str | None = None,
) -> int:
    config = g3_fixture_config(fixture_id)
    scope = qualification_scope or "confirmatory"
    if scope == "confirmatory":
        return config.expected_seed_count
    if fixture_id == "object_pair" and scope == "engineering_pilot":
        return 24
    raise G3GateError(
        f"unsupported G3 qualification scope {scope!r} for {fixture_id!r}"
    )


def goals_for_fixture(fixture_id: str) -> tuple[str, ...]:
    return g3_fixture_config(fixture_id).goals


def path_checks_per_scale_for_seed_count(
    seed_count: int,
    fixture_id: str = "horizontal",
) -> int:
    return seed_count * len(goals_for_fixture(fixture_id)) * len(PATH_SCENARIOS)


def scripted_check_count(fixture_id: str) -> int:
    goal_count = len(goals_for_fixture(fixture_id))
    return 9 * goal_count * len(REFERENCE_POSITIONS) + goal_count


def resolve_geometry_contract(
    campaign: Mapping[str, Any],
    fixture_id: str,
) -> dict[str, Any]:
    fixtures = campaign.get("fixtures")
    if isinstance(fixtures, Mapping):
        fixture_cfg = fixtures.get(fixture_id)
        if isinstance(fixture_cfg, Mapping):
            geometry = fixture_cfg.get("model_blind_g3_geometry")
            if isinstance(geometry, Mapping):
                return dict(geometry)
    if fixture_id in {"object_pair", "vertical", "containment"}:
        return dict(DEFAULT_MODEL_BLIND_G3_GEOMETRY)
    raise G3GateError(
        f"campaign lacks model_blind_g3_geometry for fixture {fixture_id!r}"
    )


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_path_check_keys(
    fixture_id: str = "horizontal",
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (goal, scenario)
        for goal in goals_for_fixture(fixture_id)
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
        check.get("path_conformance") is True
        and check.get("collision_free") is True
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
    fixture_id = str(plan.get("fixture_id"))
    result: dict[str, list[float]] = {}
    for goal in goals_for_fixture(fixture_id):
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


def _plan_receipt_sha256(value: Any, label: str) -> str:
    return _require_plan_receipt_identity(value, label)["sha256"]


def _validate_path_check_order(
    checks: Sequence[Mapping[str, Any]],
    *,
    fixture_id: str = "horizontal",
) -> None:
    expected = expected_path_check_keys(fixture_id)
    if len(checks) != len(expected):
        raise G3GateError(
            f"path receipt must contain exactly {len(expected)} checks"
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
        "path_conformance": _require_bool(
            observation.get("path_conformance"),
            f"{goal}/{scenario} path_conformance",
        ),
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


def resolve_goal_area_gate_fixtures(
    campaign: Mapping[str, Any],
) -> tuple[str, ...]:
    motion = campaign.get("motion")
    if not isinstance(motion, Mapping):
        return DEFAULT_GOAL_AREA_GATE_FIXTURES
    fixtures = motion.get("goal_area_gate_fixtures")
    if not isinstance(fixtures, list) or not fixtures:
        return DEFAULT_GOAL_AREA_GATE_FIXTURES
    return tuple(str(fixture_id) for fixture_id in fixtures)


def shrinking_area_fraction_gate_applicable(
    fixture_id: str,
    *,
    goal_area_gate_fixtures: Sequence[str] | None = None,
) -> bool:
    fixtures = (
        tuple(goal_area_gate_fixtures)
        if goal_area_gate_fixtures is not None
        else DEFAULT_GOAL_AREA_GATE_FIXTURES
    )
    return fixture_id in fixtures


def expected_information_gate_pass(
    *,
    original_empty: bool,
    destination_empty: bool,
    shrinking: bool,
    removed_fraction: float,
    minimum_fraction: float,
    apply_shrinking_fraction_gate: bool,
) -> bool:
    if original_empty or destination_empty:
        return False
    if not apply_shrinking_fraction_gate:
        return True
    return not shrinking or removed_fraction + 1e-12 >= minimum_fraction


def _plan_shrinking_area_fraction_gate_applicable(plan: Mapping[str, Any]) -> bool:
    information_gate = plan.get("information_gate")
    if isinstance(information_gate, Mapping):
        return _require_bool(
            information_gate.get("shrinking_area_fraction_gate_applicable"),
            "shrinking_area_fraction_gate_applicable",
        )
    fixture_id = plan.get("fixture_id")
    _require(isinstance(fixture_id, str), "G3 plan lacks fixture_id")
    return shrinking_area_fraction_gate_applicable(fixture_id)


def _compile_goal_area_cases(
    cases: Iterable[Mapping[str, Any]],
    *,
    minimum_fraction: float,
    apply_shrinking_fraction_gate: bool = True,
    goals: Sequence[str] = HORIZONTAL_GOALS,
) -> list[dict[str, Any]]:
    raw_cases = list(cases)
    _require(
        len(raw_cases) == len(goals),
        "path receipt requires one goal-area case per fixture goal",
    )
    compiled: list[dict[str, Any]] = []
    for goal, case in zip(goals, raw_cases):
        _require(isinstance(case, Mapping), f"goal-area case {goal} is invalid")
        _require(case.get("relation") == goal, "goal-area cases are out of order")
        original_area = _require_finite_number(
            case.get("original_area_m2"), f"{goal} original_area_m2"
        )
        destination_area = _require_finite_number(
            case.get("destination_area_m2"), f"{goal} destination_area_m2"
        )
        removed_fraction = _require_finite_number(
            case.get("removed_area_fraction"), f"{goal} removed_area_fraction"
        )
        _require(
            original_area >= 0 and destination_area >= 0,
            f"{goal} goal areas must be non-negative",
        )
        _require(
            -1e-12 <= removed_fraction <= 1.0 + 1e-12,
            f"{goal} removed-area fraction is outside [0,1]",
        )
        shrinking = _require_bool(
            case.get("shrinking_direction"), f"{goal} shrinking_direction"
        )
        original_empty = _require_bool(
            case.get("original_goal_empty"), f"{goal} original_goal_empty"
        )
        destination_empty = _require_bool(
            case.get("destination_goal_empty"), f"{goal} destination_goal_empty"
        )
        declared_minimum = _require_positive_finite(
            case.get("minimum_shrinking_area_fraction"),
            f"{goal} minimum_shrinking_area_fraction",
        )
        _require(
            abs(declared_minimum - minimum_fraction) <= 1e-12,
            f"{goal} information-gate threshold differs from plan",
        )
        expected_passes = expected_information_gate_pass(
            original_empty=original_empty,
            destination_empty=destination_empty,
            shrinking=shrinking,
            removed_fraction=removed_fraction,
            minimum_fraction=minimum_fraction,
            apply_shrinking_fraction_gate=apply_shrinking_fraction_gate,
        )
        declared_passes = _require_bool(
            case.get("passes_information_gate"),
            f"{goal} passes_information_gate",
        )
        _require(
            declared_passes == expected_passes,
            f"{goal} information-gate disposition differs from evidence",
        )
        compiled.append(
            {
                "relation": goal,
                "original_area_m2": original_area,
                "destination_area_m2": destination_area,
                "shrinking_direction": shrinking,
                "removed_area_fraction": removed_fraction,
                "minimum_shrinking_area_fraction": declared_minimum,
                "original_goal_empty": original_empty,
                "destination_goal_empty": destination_empty,
                "passes_information_gate": declared_passes,
            }
        )
    return compiled


def compile_path_seed_receipt(
    *,
    plan: Mapping[str, Any],
    plan_receipt: Mapping[str, Any],
    environment_seed: int,
    scale: float,
    check_observations: Iterable[Mapping[str, Any]],
    goal_area_cases: Iterable[Mapping[str, Any]],
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
    selection = plan.get("scale_selection")
    _require(isinstance(selection, Mapping), "G3 plan lacks scale selection")
    minimum_fraction = _require_positive_finite(
        selection.get("minimum_shrinking_area_fraction"),
        "minimum shrinking-area fraction",
    )
    apply_shrinking_fraction_gate = _plan_shrinking_area_fraction_gate_applicable(plan)
    fixture_id = str(plan.get("fixture_id"))
    goals = goals_for_fixture(fixture_id)
    compiled_goal_areas = _compile_goal_area_cases(
        goal_area_cases,
        minimum_fraction=minimum_fraction,
        apply_shrinking_fraction_gate=apply_shrinking_fraction_gate,
        goals=goals,
    )
    observations = list(check_observations)
    expected_checks = expected_path_check_keys(fixture_id)
    if len(observations) != len(expected_checks):
        raise G3GateError(
            f"path seed receipt requires exactly {len(expected_checks)} observations"
        )
    checks: list[dict[str, Any]] = []
    for (goal, scenario), observation in zip(expected_checks, observations):
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
    information_gate_passed = all(
        case["passes_information_gate"] for case in compiled_goal_areas
    )
    passed = all(check["passed"] for check in checks) and information_gate_passed
    return {
        "schema_version": path_receipt_schema(fixture_id),
        "campaign_id": plan.get("campaign_id"),
        "fixture_id": fixture_id,
        "qualification_scope": plan.get(
            "qualification_scope",
            "confirmatory",
        ),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "plan_receipt": pinned_plan,
        "scale": scale_value,
        "displacement_m": displacement_m,
        "environment_seed": environment_seed,
        "counterbalance": counterbalance,
        "direction_task_coefficients_by_goal": directions,
        "check_order": ["goal_declared_order", "scenario_declared_order"],
        "check_count": len(expected_checks),
        "checks": checks,
        "goal_area_cases": compiled_goal_areas,
        "information_gate_passed": information_gate_passed,
        "passed": passed,
        "passed_check_count": sum(1 for check in checks if check["passed"]),
        "failed_check_count": sum(1 for check in checks if not check["passed"]),
    }


def validate_path_seed_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
) -> None:
    fixture_id = receipt.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise G3GateError("path receipt lacks fixture_id")
    _require(
        receipt.get("schema_version") == path_receipt_schema(fixture_id),
        "path seed receipt schema differs",
    )
    _require(receipt.get("campaign_id") == "online_correction_v4", "campaign differs")
    if plan is not None:
        _require(receipt.get("fixture_id") == plan.get("fixture_id"), "fixture binding differs")
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
    goals = goals_for_fixture(fixture_id)
    for goal in goals:
        _require_direction_coefficients(
            directions.get(goal),
            f"direction_task_coefficients_by_goal[{goal}]",
        )
    expected_check_count = len(expected_path_check_keys(fixture_id))
    if receipt.get("check_count") != expected_check_count:
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
    _validate_path_check_order(checks, fixture_id=fixture_id)
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
            "path_conformance",
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
    goal_area_cases = receipt.get("goal_area_cases")
    if not isinstance(goal_area_cases, list):
        raise G3GateError("path receipt lacks goal-area cases")
    minimum_fraction = 0.20
    apply_shrinking_fraction_gate = True
    if plan is not None:
        selection = plan.get("scale_selection")
        _require(isinstance(selection, Mapping), "G3 plan lacks scale selection")
        minimum_fraction = _require_positive_finite(
            selection.get("minimum_shrinking_area_fraction"),
            "minimum shrinking-area fraction",
        )
        apply_shrinking_fraction_gate = _plan_shrinking_area_fraction_gate_applicable(
            plan
        )
    compiled_goal_areas = _compile_goal_area_cases(
        goal_area_cases,
        minimum_fraction=minimum_fraction,
        apply_shrinking_fraction_gate=apply_shrinking_fraction_gate,
        goals=goals,
    )
    information_gate_passed = all(
        case["passes_information_gate"] for case in compiled_goal_areas
    )
    _require(
        receipt.get("information_gate_passed") is information_gate_passed,
        "path receipt information-gate disposition differs",
    )
    receipt_passed = _require_bool(receipt.get("passed"), "receipt passed")
    if receipt_passed != (failed_count == 0 and information_gate_passed):
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
        for goal in goals:
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
    fixture_id: str = "horizontal",
) -> dict[str, Any]:
    if check_kind not in SCRIPTED_CHECK_KINDS:
        raise G3GateError("scripted check kind differs")
    if type(environment_seed) is not int:
        raise G3GateError("environment_seed must be an integer")
    if goal not in goals_for_fixture(fixture_id):
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
        "schema_version": scripted_receipt_schema(fixture_id),
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
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
    fixture_id = receipt.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise G3GateError("scripted receipt lacks fixture_id")
    _require(
        receipt.get("schema_version") == scripted_receipt_schema(fixture_id),
        "scripted receipt schema differs",
    )
    _require(receipt.get("campaign_id") == "online_correction_v4", "campaign differs")
    _require(receipt.get("fixture_id") == fixture_id, "fixture differs")
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
    if goal not in goals_for_fixture(fixture_id):
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
    counterbalance_family: str = "C1",
    counterbalance_fixture: str = "horizontal",
) -> dict[int, dict[str, Any]]:
    expected = set(int(seed) for seed in expected_env_seeds)
    result: dict[int, dict[str, Any]] = {}
    for row in queue_rows:
        if (
            row.get("family") != counterbalance_family
            or row.get("fixture") != counterbalance_fixture
        ):
            continue
        seed = row.get("env_seed")
        counterbalance = row.get("counterbalance")
        if type(seed) is not int or seed not in expected:
            continue
        if not isinstance(counterbalance, Mapping):
            raise G3GateError(
                f"{counterbalance_family} row for seed {seed} lacks counterbalance"
            )
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
            raise G3GateError(
                f"{counterbalance_family} counterbalance differs within seed {seed}"
            )
    missing = sorted(expected - set(result))
    _require(
        not missing,
        f"{counterbalance_family} counterbalance index misses seeds: {missing[:5]}",
    )
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


def build_pilot_counterbalance_index(
    queue_rows: Iterable[Mapping[str, Any]],
    *,
    resets_by_env_seed: Mapping[str, Any],
    counterbalance_family: str,
    counterbalance_fixture: str,
) -> dict[int, dict[str, Any]]:
    by_block: dict[int, dict[str, Any]] = {}
    for row in queue_rows:
        if (
            row.get("family") != counterbalance_family
            or row.get("fixture") != counterbalance_fixture
        ):
            continue
        block = row.get("block_id")
        counterbalance = row.get("counterbalance")
        if type(block) is not int or not isinstance(counterbalance, Mapping):
            continue
        frozen = {
            "block_id": block,
            "state_index": counterbalance.get("state_index"),
            "physical_translation_sign": counterbalance.get(
                "physical_translation_sign"
            ),
            "event_phase_fraction": counterbalance.get("event_phase_fraction"),
        }
        prior = by_block.setdefault(block, frozen)
        if prior != frozen:
            raise G3GateError(
                f"{counterbalance_family} counterbalance differs within block {block}"
            )
    result: dict[int, dict[str, Any]] = {}
    for key, reset in resets_by_env_seed.items():
        if not isinstance(reset, Mapping):
            raise G3GateError(f"pilot reset {key!r} is not an object")
        env_seed = int(key)
        block = reset.get("block_index")
        if type(block) is not int or block not in by_block:
            raise G3GateError(
                f"pilot reset {env_seed} lacks a registered counterbalance block"
            )
        result[env_seed] = dict(by_block[block])
    return result


def build_plan_payload(
    *,
    source_identity: Mapping[str, Any],
    g2_prerequisite: Mapping[str, Any],
    geometry_contract: Mapping[str, Any],
    reset_registry: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
    scale_candidates: Iterable[float],
    nominal_displacement_m: float,
    minimum_shrinking_area_fraction: float,
    goal_area_gate_fixtures: Sequence[str] | None = None,
    fixture_id: str = "horizontal",
    qualification_scope: str = "confirmatory",
) -> dict[str, Any]:
    if not isinstance(qualification_scope, str):
        raise G3GateError("G3 plan qualification_scope is invalid")
    config = g3_fixture_config(fixture_id)
    expected_seed_count = g3_expected_seed_count(fixture_id, qualification_scope)
    _validate_g2_prerequisite(
        g2_prerequisite,
        fixture_id=fixture_id,
        expected_seed_count=expected_seed_count,
    )
    _validate_geometry_contract(geometry_contract)
    resets = reset_registry.get("resets_by_env_seed")
    _require(isinstance(resets, Mapping) and resets, "reset registry has no resets")
    seeds = tuple(sorted(int(seed) for seed in resets))
    if qualification_scope == "engineering_pilot":
        counterbalance = build_pilot_counterbalance_index(
            queue_rows,
            resets_by_env_seed=resets,
            counterbalance_family=config.counterbalance_family,
            counterbalance_fixture=config.fixture_id,
        )
    else:
        counterbalance = build_counterbalance_index(
            queue_rows,
            expected_env_seeds=seeds,
            counterbalance_family=config.counterbalance_family,
            counterbalance_fixture=config.fixture_id,
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
    goals = goals_for_fixture(fixture_id)
    directions: dict[str, Any] = {}
    for seed in seeds:
        sign = int(counterbalance[seed]["physical_translation_sign"])
        directions[str(seed)] = {
            goal: list(
                ReferenceMotionController.displacement_vector(
                    goal=goal,
                    fixture=fixture_id,
                    physical_sign=sign,
                )
            )
            for goal in goals
        }
    path_checks_per_scale = path_checks_per_scale_for_seed_count(
        len(seeds),
        fixture_id,
    )
    expected_path_checks = path_checks_per_scale_for_seed_count(
        expected_seed_count,
        fixture_id,
    )
    _require(
        path_checks_per_scale == expected_path_checks,
        f"{fixture_id} path count differs",
    )
    scripted_checks = scripted_check_count(fixture_id)
    gate_fixtures = (
        tuple(goal_area_gate_fixtures)
        if goal_area_gate_fixtures is not None
        else DEFAULT_GOAL_AREA_GATE_FIXTURES
    )
    shrinking_gate_applicable = shrinking_area_fraction_gate_applicable(
        fixture_id,
        goal_area_gate_fixtures=gate_fixtures,
    )
    payload: dict[str, Any] = {
        "schema_version": plan_schema(fixture_id),
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "status": "model_blind_candidate_not_released_for_inference",
        "plan_status": "ready_for_live_g3_execution",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_identity": dict(source_identity),
        "g2_prerequisite": dict(g2_prerequisite),
        "geometry_contract": dict(geometry_contract),
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
            "goals": list(goals),
            "scenarios": list(PATH_SCENARIOS),
            "sample_interval_s": PATH_SAMPLE_INTERVAL_S,
            "reset_replay_rule": (
                "one attested physical reset per environment seed followed by "
                "full object and robot joint state restoration before each path"
            ),
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
                "goals": list(goals),
                "reference_positions": list(REFERENCE_POSITIONS),
                "checks_per_scale": 9 * len(goals) * len(REFERENCE_POSITIONS),
            },
            "moving": {
                "canonical_env_seed": extremes[0],
                "goals": list(goals),
                "scenario": "move_stop",
                "checks_per_scale": len(goals),
            },
            "checks_per_final_geometry_candidate": scripted_checks,
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
    if fixture_id != "horizontal" or qualification_scope != "confirmatory":
        payload["qualification_scope"] = qualification_scope
    if fixture_id == "object_pair":
        payload["information_gate"] = {
            "shrinking_area_fraction_gate_applicable": shrinking_gate_applicable,
            "goal_area_gate_fixtures": list(gate_fixtures),
            "nonempty_original_and_destination_required": True,
            "policy_outcome_used": False,
        }
    return payload


def _validate_g2_prerequisite(
    prerequisite: Mapping[str, Any],
    *,
    fixture_id: str = "horizontal",
    expected_seed_count: int | None = None,
) -> None:
    config = g3_fixture_config(fixture_id)
    expected = (
        config.expected_seed_count
        if expected_seed_count is None
        else expected_seed_count
    )
    _require(
        prerequisite.get("schema_version")
        == aggregate_receipt_schema(fixture_id),
        f"G3 plan requires the {fixture_id} G2 aggregate schema",
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
        prerequisite.get("expected_seed_count") == expected
        and prerequisite.get("observed_seed_count") == expected,
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


def _validate_geometry_contract(contract: Mapping[str, Any]) -> None:
    _require(
        contract.get("extent_convention")
        == "live_usd_world_aabb_projected_into_registered_task_frame",
        "G3 extent convention differs",
    )
    _require(
        contract.get("supported_workspace_convention")
        == (
            "live_table_top_aabb_eroded_by_target_projected_half_extents_and_"
            "edge_margin"
        ),
        "G3 supported-workspace convention differs",
    )
    for key in (
        "relation_clearance_m",
        "support_edge_margin_m",
        "active_contact_force_threshold_n",
        "reference_pose_error_max_m",
        "stationary_object_drift_max_m",
        "path_sample_max_interval_s",
    ):
        value = contract.get(key)
        _require(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and float(value) > 0,
            f"G3 geometry contract {key} must be positive and finite",
        )
    _require(
        abs(float(contract["path_sample_max_interval_s"]) - PATH_SAMPLE_INTERVAL_S)
        <= 1e-12,
        "G3 path-sampling interval differs",
    )
    _require(
        contract.get("robot_reference_contact_probe")
        == "full_robot_articulation_regex_against_reference_pair_sensor",
        "G3 robot/reference contact probe differs",
    )
    _require(
        contract.get("policy_outcome_used") is False,
        "G3 geometry contract may not use policy outcomes",
    )


def validate_plan_payload(plan: Mapping[str, Any]) -> None:
    fixture_id = plan.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise G3GateError("G3 plan lacks fixture_id")
    qualification_scope = plan.get("qualification_scope", "confirmatory")
    if not isinstance(qualification_scope, str):
        raise G3GateError("G3 plan qualification_scope is invalid")
    expected_seed_count = g3_expected_seed_count(
        fixture_id,
        qualification_scope,
    )
    _require(plan.get("schema_version") == plan_schema(fixture_id), "G3 plan schema differs")
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
    _validate_g2_prerequisite(
        prerequisite,
        fixture_id=fixture_id,
        expected_seed_count=expected_seed_count,
    )
    geometry_contract = plan.get("geometry_contract")
    _require(
        isinstance(geometry_contract, Mapping),
        "G3 plan lacks geometry contract",
    )
    _validate_geometry_contract(geometry_contract)
    _require(
        plan.get("registered_reset_count") == expected_seed_count,
        "G3 reset count differs",
    )
    path = plan.get("path_sweep")
    scripted = plan.get("scripted_controller")
    _require(isinstance(path, Mapping), "G3 plan lacks path sweep")
    _require(isinstance(scripted, Mapping), "G3 plan lacks scripted checks")
    expected_path_checks = path_checks_per_scale_for_seed_count(
        expected_seed_count,
        fixture_id,
    )
    _require(path.get("checks_per_scale") == expected_path_checks, "G3 path count differs")
    goals = goals_for_fixture(fixture_id)
    _require(path.get("goals") == list(goals), "G3 path goals differ")
    _require(
        path.get("reset_replay_rule")
        == (
            "one attested physical reset per environment seed followed by full "
            "object and robot joint state restoration before each path"
        ),
        "G3 path reset/replay rule differs",
    )
    _require(
        scripted.get("checks_per_final_geometry_candidate")
        == scripted_check_count(fixture_id),
        "G3 scripted count differs",
    )
    information_gate = plan.get("information_gate")
    expected_applicable = shrinking_area_fraction_gate_applicable(fixture_id)
    if fixture_id == "object_pair":
        _require(
            isinstance(information_gate, Mapping),
            "object_pair G3 plan lacks information_gate",
        )
        _require(
            information_gate.get("shrinking_area_fraction_gate_applicable") is False,
            "object_pair must not apply the shrinking-area fraction gate",
        )
        gate_fixtures = information_gate.get("goal_area_gate_fixtures")
        _require(
            isinstance(gate_fixtures, list)
            and tuple(gate_fixtures) == DEFAULT_GOAL_AREA_GATE_FIXTURES,
            "object_pair goal_area_gate_fixtures differ from campaign",
        )
        _require(
            information_gate.get("nonempty_original_and_destination_required") is True,
            "object_pair requires nonempty original and destination goal sets",
        )
    elif isinstance(information_gate, Mapping):
        _require(
            information_gate.get("shrinking_area_fraction_gate_applicable")
            == expected_applicable,
            "information_gate applicability disagrees with fixture",
        )
        gate_fixtures = information_gate.get("goal_area_gate_fixtures")
        _require(
            isinstance(gate_fixtures, list)
            and tuple(gate_fixtures) == DEFAULT_GOAL_AREA_GATE_FIXTURES,
            "goal_area_gate_fixtures differ from campaign",
        )
    reset_seeds = scripted.get("reset_env_seeds")
    _require(
        isinstance(reset_seeds, list)
        and len(reset_seeds) == 9
        and len(set(reset_seeds)) == 9,
        "G3 scripted reset selection differs",
    )


def expected_scripted_check_keys(
    plan: Mapping[str, Any],
) -> tuple[tuple[str, int, str, str], ...]:
    validate_plan_payload(plan)
    scripted = plan.get("scripted_controller")
    _require(isinstance(scripted, Mapping), "G3 plan lacks scripted checks")
    reset_seeds = scripted.get("reset_env_seeds")
    _require(
        isinstance(reset_seeds, list) and len(reset_seeds) == 9,
        "G3 scripted reset selection differs",
    )
    stationary = scripted.get("stationary")
    moving = scripted.get("moving")
    _require(isinstance(stationary, Mapping), "G3 plan lacks stationary scripted checks")
    _require(isinstance(moving, Mapping), "G3 plan lacks moving scripted checks")
    goals = stationary.get("goals")
    positions = stationary.get("reference_positions")
    _require(
        list(goals) == list(goals_for_fixture(str(plan.get("fixture_id")))),
        "G3 stationary scripted goals differ",
    )
    _require(
        list(positions) == list(REFERENCE_POSITIONS),
        "G3 stationary scripted reference positions differ",
    )
    moving_goals = moving.get("goals")
    _require(
        list(moving_goals)
        == list(goals_for_fixture(str(plan.get("fixture_id")))),
        "G3 moving scripted goals differ",
    )
    canonical_seed = moving.get("canonical_env_seed")
    if type(canonical_seed) is not int:
        raise G3GateError("G3 moving canonical seed differs")
    _require(
        moving.get("scenario") == "move_stop",
        "G3 moving scripted scenario differs",
    )
    keys: list[tuple[str, int, str, str]] = []
    fixture_goals = goals_for_fixture(str(plan.get("fixture_id")))
    for seed in reset_seeds:
        if type(seed) is not int:
            raise G3GateError("G3 scripted reset seed differs")
        for goal in fixture_goals:
            for position in REFERENCE_POSITIONS:
                keys.append(("stationary", seed, goal, position))
    for goal in fixture_goals:
        keys.append(("moving", canonical_seed, goal, "endpoint"))
    _require(
        len(keys) == scripted_check_count(str(plan.get("fixture_id"))),
        "fixture scripted count differs",
    )
    return tuple(keys)


def _scripted_check_key(receipt: Mapping[str, Any]) -> tuple[str, int, str, str]:
    check_kind = receipt.get("check_kind")
    environment_seed = receipt.get("environment_seed")
    goal = receipt.get("goal")
    reference_position = receipt.get("reference_position")
    fixture_id = receipt.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise G3GateError("scripted receipt lacks fixture_id")
    if check_kind not in SCRIPTED_CHECK_KINDS:
        raise G3GateError("scripted check kind differs")
    if type(environment_seed) is not int:
        raise G3GateError("environment_seed must be an integer")
    if goal not in goals_for_fixture(fixture_id):
        raise G3GateError("scripted check goal differs")
    if reference_position not in REFERENCE_POSITIONS:
        raise G3GateError("scripted check reference_position differs")
    return (check_kind, environment_seed, goal, reference_position)


def _runtime_stratum_from_receipt(receipt: Mapping[str, Any]) -> dict[str, Any] | None:
    runtime = receipt.get("runtime_identity")
    if not isinstance(runtime, Mapping):
        return None
    study = runtime.get("study_checkout")
    robolab = runtime.get("robolab_checkout")
    gpu = runtime.get("gpu")
    if not all(isinstance(item, Mapping) for item in (study, robolab, gpu)):
        return None
    return {
        "study_commit": study.get("commit"),
        "robolab_commit": robolab.get("commit"),
        "gpu_name": gpu.get("name"),
        "driver_version": gpu.get("driver_version"),
        "gate_entrypoint_sha256": runtime.get("gate_entrypoint_sha256"),
        "gate_core_sha256": runtime.get("gate_core_sha256"),
        "droid_robolab_sha256": runtime.get("droid_robolab_sha256"),
        "droid_g3_sha256": runtime.get("droid_g3_sha256"),
        "plan_sha256": runtime.get("plan_sha256"),
        "campaign_sha256": runtime.get("campaign_sha256"),
        "reset_registry_sha256": runtime.get("reset_registry_sha256"),
        "native_control_dt_s": runtime.get("native_control_dt_s"),
    }


def _merge_runtime_stratum(
    current: dict[str, Any] | None,
    receipt: Mapping[str, Any],
) -> dict[str, Any] | None:
    stratum = _runtime_stratum_from_receipt(receipt)
    if stratum is None:
        return current
    if current is None:
        return stratum
    if current != stratum:
        raise G3GateError("runtime stratum differs across receipts")
    return current


def _collect_path_seed_failures(
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    seed = receipt.get("environment_seed")
    if receipt.get("information_gate_passed") is False:
        failures.append(
            {
                "environment_seed": seed,
                "failure_kind": "information_gate",
                "reasons": [
                    f"{case.get('relation')} information gate failed"
                    for case in receipt.get("goal_area_cases", [])
                    if isinstance(case, Mapping)
                    and case.get("passes_information_gate") is False
                ],
            }
        )
    checks = receipt.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, Mapping) or check.get("passed") is True:
                continue
            failures.append(
                {
                    "environment_seed": seed,
                    "failure_kind": "path_check",
                    "goal": check.get("goal"),
                    "scenario": check.get("scenario"),
                    "reasons": list(check.get("reasons") or []),
                }
            )
    return failures


def compile_path_scale_receipt(
    *,
    plan: Mapping[str, Any],
    plan_receipt: Mapping[str, Any],
    scale: float,
    path_seed_receipts: Iterable[Mapping[str, Any]],
    path_seed_receipt_files: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_plan_payload(plan)
    scale_value = _require_positive_finite(scale, "scale")
    candidates = _plan_scale_candidates(plan)
    if scale_value not in candidates:
        raise G3GateError("scale is not a registered G3 candidate")
    nominal = _plan_nominal_displacement_m(plan)
    displacement_m = nominal * scale_value
    pinned_plan = _require_plan_receipt_identity(plan_receipt, "plan_receipt")
    fixture_id = str(plan.get("fixture_id"))
    expected_seed_count = g3_expected_seed_count(
        fixture_id,
        str(plan.get("qualification_scope", "confirmatory")),
    )
    expected_seeds = tuple(int(seed) for seed in plan["registered_env_seeds"])
    _require(len(expected_seeds) == expected_seed_count, "G3 reset count differs")

    observed: dict[int, Mapping[str, Any]] = {}
    runtime_stratum: dict[str, Any] | None = None
    passed_path_check_count = 0
    failed_path_check_count = 0
    scientific_failures: list[dict[str, Any]] = []

    for receipt in path_seed_receipts:
        validate_path_seed_receipt(receipt, plan=plan)
        if _require_positive_finite(receipt.get("scale"), "scale") != scale_value:
            raise G3GateError("path seed receipt scale differs")
        if _plan_receipt_sha256(receipt.get("plan_receipt"), "plan_receipt") != pinned_plan["sha256"]:
            raise G3GateError("path seed receipt plan binding differs")
        seed = receipt.get("environment_seed")
        if type(seed) is not int or seed in observed:
            raise G3GateError("path scale receipts contain a missing or duplicate seed")
        observed[seed] = receipt
        runtime_stratum = _merge_runtime_stratum(runtime_stratum, receipt)
        passed_path_check_count += int(receipt.get("passed_check_count") or 0)
        failed_path_check_count += int(receipt.get("failed_check_count") or 0)
        scientific_failures.extend(_collect_path_seed_failures(receipt))

    missing = sorted(set(expected_seeds) - set(observed))
    unexpected = sorted(set(observed) - set(expected_seeds))
    failed_seeds = sorted(
        seed for seed, receipt in observed.items() if receipt.get("passed") is not True
    )
    information_gate_failed_seeds = sorted(
        seed
        for seed, receipt in observed.items()
        if receipt.get("information_gate_passed") is not True
    )
    checks_per_seed = len(expected_path_check_keys(fixture_id))
    expected_path_check_count = len(expected_seeds) * checks_per_seed
    _require(
        expected_path_check_count
        == path_checks_per_scale_for_seed_count(
            expected_seed_count,
            fixture_id,
        ),
        f"{fixture_id} path count differs",
    )

    receipt_files: dict[str, Any] = {}
    if path_seed_receipt_files is not None:
        for seed in expected_seeds:
            record = path_seed_receipt_files.get(seed)
            if record is None:
                continue
            receipt_files[str(seed)] = dict(record)

    passed = (
        not missing
        and not unexpected
        and not failed_seeds
        and passed_path_check_count == expected_path_check_count
        and failed_path_check_count == 0
    )
    return {
        "schema_version": path_scale_receipt_schema(fixture_id),
        "campaign_id": plan.get("campaign_id"),
        "fixture_id": fixture_id,
        "qualification_scope": plan.get(
            "qualification_scope",
            "confirmatory",
        ),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "plan_receipt": pinned_plan,
        "scale": scale_value,
        "displacement_m": displacement_m,
        "expected_seed_count": len(expected_seeds),
        "observed_seed_count": len(observed),
        "missing_env_seeds": missing,
        "unexpected_env_seeds": unexpected,
        "failed_env_seeds": failed_seeds,
        "information_gate_failed_seeds": information_gate_failed_seeds,
        "expected_path_check_count": expected_path_check_count,
        "passed_path_check_count": passed_path_check_count,
        "failed_path_check_count": failed_path_check_count,
        "path_seed_receipt_files_by_env_seed": receipt_files,
        "path_seed_receipt_sha256_by_env_seed": {
            str(seed): sha256_bytes(canonical_json_bytes(dict(receipt)))
            for seed, receipt in sorted(observed.items())
        },
        "runtime_stratum": runtime_stratum,
        "scientific_failure_summary": scientific_failures,
        "release_boundary": (
            "A passing path-scale receipt completes one G3 geometry candidate only. "
            "Scripted-controller checks and aggregate selection remain required before "
            "G3 completion or G4 preparation."
        ),
    }


def validate_path_scale_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
) -> None:
    fixture_id = receipt.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise G3GateError("path scale receipt lacks fixture_id")
    qualification_scope = (
        plan.get("qualification_scope", "confirmatory")
        if plan is not None
        else receipt.get("qualification_scope", "confirmatory")
    )
    if not isinstance(qualification_scope, str):
        raise G3GateError("path scale qualification_scope is invalid")
    expected_seed_count = g3_expected_seed_count(
        fixture_id,
        qualification_scope,
    )
    _require(
        receipt.get("schema_version") == path_scale_receipt_schema(fixture_id),
        "path scale receipt schema differs",
    )
    _require(receipt.get("campaign_id") == "online_correction_v4", "campaign differs")
    if plan is not None:
        _require(receipt.get("fixture_id") == plan.get("fixture_id"), "fixture binding differs")
    _require(
        receipt.get("model_request_count") == 0,
        "path scale receipt records model requests",
    )
    _require(
        receipt.get("behavioral_episode_count") == 0,
        "path scale receipt records behavioral episodes",
    )
    scale = _require_positive_finite(receipt.get("scale"), "scale")
    displacement_m = _require_positive_finite(receipt.get("displacement_m"), "displacement_m")
    _require_plan_receipt_identity(receipt.get("plan_receipt"), "plan_receipt")
    expected_path_check_count = _require_positive_int(
        receipt.get("expected_path_check_count"),
        "expected_path_check_count",
    )
    _require(
        expected_path_check_count
        == path_checks_per_scale_for_seed_count(
            expected_seed_count,
            fixture_id,
        ),
        "path scale check count differs",
    )
    passed_count = _require_non_negative_int(
        receipt.get("passed_path_check_count"), "passed_path_check_count"
    )
    failed_count = _require_non_negative_int(
        receipt.get("failed_path_check_count"), "failed_path_check_count"
    )
    observed_seed_count = _require_non_negative_int(
        receipt.get("observed_seed_count"), "observed_seed_count"
    )
    _require(
        passed_count + failed_count
        == observed_seed_count * len(expected_path_check_keys(fixture_id)),
        "path scale check totals differ",
    )
    missing = receipt.get("missing_env_seeds")
    unexpected = receipt.get("unexpected_env_seeds")
    failed_seeds = receipt.get("failed_env_seeds")
    if not isinstance(missing, list) or not isinstance(unexpected, list):
        raise G3GateError("path scale receipt lacks seed coverage lists")
    if not isinstance(failed_seeds, list):
        raise G3GateError("path scale receipt lacks failed_env_seeds")
    receipt_passed = _require_bool(receipt.get("passed"), "passed")
    if receipt_passed != (
        not missing
        and not unexpected
        and not failed_seeds
        and failed_count == 0
        and passed_count == expected_path_check_count
    ):
        raise G3GateError("path scale receipt passed disagrees with evidence")
    if plan is not None:
        validate_plan_payload(plan)
        if scale not in _plan_scale_candidates(plan):
            raise G3GateError("scale is not bound to the G3 plan")
        nominal = _plan_nominal_displacement_m(plan)
        if abs(displacement_m - nominal * scale) > 1e-9:
            raise G3GateError("displacement_m is not bound to plan scale selection")


def _validate_scale_ladder_prefix(
    *,
    plan: Mapping[str, Any],
    path_scale_receipts_by_scale: Mapping[float, Mapping[str, Any]],
) -> None:
    scales = _plan_scale_candidates(plan)
    if not path_scale_receipts_by_scale:
        return
    highest_index = max(scales.index(scale) for scale in path_scale_receipts_by_scale)
    for index in range(highest_index + 1):
        scale = scales[index]
        if scale not in path_scale_receipts_by_scale:
            raise G3GateError(
                f"scale ladder gap: {scale} is missing while lower scales are present"
            )


def _select_path_scale(
    *,
    plan: Mapping[str, Any],
    path_scale_receipts_by_scale: Mapping[float, Mapping[str, Any]],
) -> tuple[float | None, Mapping[str, Any] | None, list[float], str]:
    scales = _plan_scale_candidates(plan)
    _validate_scale_ladder_prefix(
        plan=plan,
        path_scale_receipts_by_scale=path_scale_receipts_by_scale,
    )
    rejected_scales: list[float] = []
    for scale in scales:
        receipt = path_scale_receipts_by_scale.get(scale)
        if receipt is None:
            if not rejected_scales:
                return None, None, rejected_scales, "pending"
            return None, None, rejected_scales, "blocked"
        validate_path_scale_receipt(receipt, plan=plan)
        if receipt.get("passed") is True:
            expected_rejected = list(scales[: scales.index(scale)])
            if [float(item) for item in rejected_scales] != [
                float(item) for item in expected_rejected
            ]:
                raise G3GateError(
                    "completed rejected scales are not a strict prefix before the passing scale"
                )
            return scale, receipt, rejected_scales, "selected"
        rejected_scales.append(scale)
    if rejected_scales:
        return None, None, rejected_scales, "blocked"
    return None, None, rejected_scales, "pending"


def compile_g3_aggregate_receipt(
    *,
    plan: Mapping[str, Any],
    plan_receipt: Mapping[str, Any],
    path_scale_receipts: Iterable[Mapping[str, Any]],
    path_scale_receipt_files: Mapping[float, Mapping[str, Any]] | None = None,
    scripted_check_receipts: Iterable[Mapping[str, Any]] | None = None,
    scripted_check_receipt_files: Mapping[
        tuple[str, int, str, str], Mapping[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    validate_plan_payload(plan)
    pinned_plan = _require_plan_receipt_identity(plan_receipt, "plan_receipt")
    scales = _plan_scale_candidates(plan)
    path_scale_by_scale: dict[float, Mapping[str, Any]] = {}
    for receipt in path_scale_receipts:
        validate_path_scale_receipt(receipt, plan=plan)
        if _plan_receipt_sha256(receipt.get("plan_receipt"), "plan_receipt") != pinned_plan["sha256"]:
            raise G3GateError("path scale receipt plan binding differs")
        scale = _require_positive_finite(receipt.get("scale"), "scale")
        if scale in path_scale_by_scale:
            raise G3GateError("duplicate path scale receipt")
        path_scale_by_scale[scale] = receipt

    selected_scale, selected_path_scale, rejected_scales, ladder_status = (
        _select_path_scale(plan=plan, path_scale_receipts_by_scale=path_scale_by_scale)
    )

    expected_scripted_keys = expected_scripted_check_keys(plan)
    expected_scripted_count = scripted_check_count(str(plan.get("fixture_id")))
    observed_scripted: dict[tuple[str, int, str, str], Mapping[str, Any]] = {}
    scripted_failures: list[dict[str, Any]] = []
    scripted_passed_count = 0
    scripted_failed_count = 0
    missing_scripted_keys: list[tuple[str, int, str, str]] = []
    unexpected_scripted_keys: list[tuple[str, int, str, str]] = []
    scripted_complete = False
    scripted_passed = False
    displacement_m: float | None = None

    if selected_scale is not None and selected_path_scale is not None:
        displacement_m = _require_positive_finite(
            selected_path_scale.get("displacement_m"),
            "displacement_m",
        )
        if scripted_check_receipts is None:
            missing_scripted_keys = list(expected_scripted_keys)
        else:
            for receipt in scripted_check_receipts:
                validate_scripted_check_receipt(receipt)
                if _require_positive_finite(receipt.get("scale"), "scale") != selected_scale:
                    raise G3GateError("scripted receipt scale differs from selected scale")
                receipt_displacement = _require_positive_finite(
                    receipt.get("displacement_m"), "displacement_m"
                )
                if abs(receipt_displacement - displacement_m) > 1e-9:
                    raise G3GateError(
                        "scripted receipt displacement differs from selected scale"
                    )
                key = _scripted_check_key(receipt)
                if key in observed_scripted:
                    raise G3GateError("duplicate scripted check receipt")
                observed_scripted[key] = receipt
                if receipt.get("passed") is True:
                    scripted_passed_count += 1
                else:
                    scripted_failed_count += 1
                    scripted_failures.append(
                        {
                            "check_kind": key[0],
                            "environment_seed": key[1],
                            "goal": key[2],
                            "reference_position": key[3],
                            "reasons": list(receipt.get("reasons") or []),
                        }
                    )

            for index, expected_key in enumerate(expected_scripted_keys):
                receipt = observed_scripted.get(expected_key)
                if receipt is None:
                    missing_scripted_keys.append(expected_key)
                    continue
                actual_key = _scripted_check_key(receipt)
                if actual_key != expected_key:
                    raise G3GateError("scripted receipts are out of declared order")
                if index > 0:
                    prior_key = expected_scripted_keys[index - 1]
                    if prior_key not in observed_scripted:
                        raise G3GateError("scripted receipts are missing required prefix")
            unexpected_scripted_keys = sorted(
                set(observed_scripted) - set(expected_scripted_keys)
            )
            scripted_complete = (
                not missing_scripted_keys
                and not unexpected_scripted_keys
                and len(observed_scripted) == expected_scripted_count
            )
            scripted_passed = (
                scripted_complete
                and scripted_failed_count == 0
                and scripted_passed_count == expected_scripted_count
            )

    if selected_scale is None:
        status = ladder_status
        passed = False
    elif not scripted_complete:
        status = "blocked_incomplete"
        passed = False
    elif not scripted_passed:
        status = "failed"
        passed = False
    else:
        status = "passed"
        passed = True

    path_scale_files: dict[str, Any] = {}
    if path_scale_receipt_files is not None:
        for scale, record in path_scale_receipt_files.items():
            path_scale_files[str(scale)] = dict(record)

    scripted_files: list[dict[str, Any]] = []
    if scripted_check_receipt_files is not None:
        for key in expected_scripted_keys:
            record = scripted_check_receipt_files.get(key)
            if record is not None:
                scripted_files.append({"key": list(key), **dict(record)})

    return {
        "schema_version": aggregate_receipt_schema_g3(str(plan.get("fixture_id"))),
        "campaign_id": plan.get("campaign_id"),
        "fixture_id": plan.get("fixture_id"),
        "status": status,
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "plan_receipt": pinned_plan,
        "candidate_scales_descending": list(scales),
        "rejected_scales": rejected_scales,
        "selected_scale": selected_scale,
        "selected_displacement_m": displacement_m,
        "selected_path_scale_receipt_sha256": (
            sha256_bytes(canonical_json_bytes(dict(selected_path_scale)))
            if selected_path_scale is not None
            else None
        ),
        "expected_scripted_check_count": expected_scripted_count,
        "observed_scripted_check_count": len(observed_scripted),
        "missing_scripted_check_keys": [
            list(key) for key in missing_scripted_keys
        ],
        "unexpected_scripted_check_keys": [
            list(key) for key in unexpected_scripted_keys
        ],
        "scripted_passed_check_count": scripted_passed_count,
        "scripted_failed_check_count": scripted_failed_count,
        "path_scale_receipt_files_by_scale": path_scale_files,
        "scripted_check_receipt_files": scripted_files,
        "scientific_failure_summary": {
            "path_scale": (
                selected_path_scale.get("scientific_failure_summary")
                if selected_path_scale is not None
                else []
            ),
            "scripted": scripted_failures,
        },
        "release_boundary": (
            "A passing aggregate completes this fixture's G3 only and authorizes G4 "
            "preparation. Policy inference and reset-registry release remain blocked."
        ),
    }


def validate_g3_aggregate_receipt(
    receipt: Mapping[str, Any],
    *,
    plan: Mapping[str, Any] | None = None,
) -> None:
    fixture_id = receipt.get("fixture_id")
    if not isinstance(fixture_id, str):
        raise G3GateError("G3 aggregate receipt lacks fixture_id")
    _require(
        receipt.get("schema_version") == aggregate_receipt_schema_g3(fixture_id),
        "G3 aggregate receipt schema differs",
    )
    _require(receipt.get("campaign_id") == "online_correction_v4", "campaign differs")
    _require(receipt.get("fixture_id") == fixture_id, "fixture differs")
    _require(
        receipt.get("model_request_count") == 0,
        "G3 aggregate records model requests",
    )
    _require(
        receipt.get("behavioral_episode_count") == 0,
        "G3 aggregate records behavioral episodes",
    )
    _require_plan_receipt_identity(receipt.get("plan_receipt"), "plan_receipt")
    passed = _require_bool(receipt.get("passed"), "passed")
    status = receipt.get("status")
    if passed:
        _require(status == "passed", "passed aggregate must report passed status")
    else:
        _require(
            status in {"pending", "blocked", "blocked_incomplete", "failed"},
            "aggregate status differs",
        )
    if plan is not None:
        validate_plan_payload(plan)
        _require(plan.get("fixture_id") == fixture_id, "fixture binding differs")
        selected = receipt.get("selected_scale")
        if selected is not None:
            if selected not in _plan_scale_candidates(plan):
                raise G3GateError("selected scale is not bound to the G3 plan")
