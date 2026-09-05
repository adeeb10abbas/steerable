"""Prospective contracts for horizontal model-blind motion/feasibility gate G3."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from experiments.online_correction_v4.motion import ReferenceMotionController

PLAN_SCHEMA = "v4-horizontal-g3-plan-v1"
PATH_RECEIPT_SCHEMA = "v4-horizontal-g3-path-seed-receipt-v1"
AGGREGATE_SCHEMA = "v4-horizontal-g3-aggregate-receipt-v1"
HORIZONTAL_GOALS = ("left", "right", "front", "behind")
PATH_SCENARIOS = (
    "original_sham",
    "destination_static",
    "move_stop",
    "slow_drift",
    "fast_drift",
    "reversal",
)
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
    reset_registry: Mapping[str, Any],
    queue_rows: Iterable[Mapping[str, Any]],
    scale_candidates: Iterable[float],
    nominal_displacement_m: float,
    minimum_shrinking_area_fraction: float,
) -> dict[str, Any]:
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
        "plan_status": "pending_g2_and_live_execution",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "source_identity": dict(source_identity),
        "g2_prerequisite": {
            "required_schema": "v4-horizontal-g2-aggregate-receipt-v1",
            "required_passed": True,
            "status": "pending_live_receipt",
        },
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
            "G2 passes and live collision, geometry, and scripted-controller "
            "receipts select one scale."
        ),
    }


def validate_plan_payload(plan: Mapping[str, Any]) -> None:
    _require(plan.get("schema_version") == PLAN_SCHEMA, "G3 plan schema differs")
    _require(plan.get("fixture_id") == "horizontal", "G3 plan fixture differs")
    _require(plan.get("model_request_count") == 0, "G3 plan records model requests")
    _require(
        plan.get("behavioral_episode_count") == 0,
        "G3 plan records behavioral episodes",
    )
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
