import pytest

from experiments.workshops.spatial_grounding_v1.scoring import (
    FrozenScoringConfig,
    GoalSpec,
    OutcomeStatus,
    canonical_status,
    relation_m,
    score_episode,
)


def _episode(final, *, states=None, **extra):
    if states is None:
        initial = _state(cube=(final["cube"][0], final["cube"][1], final["cube"][2] - 0.04))
        initial["gripper_holding"] = False
        initial["cube_height_lift_m"] = 0.0
        states = [initial] + [dict(final, sim_time_s=i * 0.01, supported=True,
                                   linear_speed_m_s=0.0, angular_speed_rad_s=0.0)
                              for i in range(1, 451)]
    return {
        "states": states,
        "terminal_observed": True,
        "success_events": [{"step": 120, "success": True}],
        "status": "valid_model",
        **extra,
    }


def _state(cube=(0.0, 0.05, 0.1), bowl=(0.0, 0.0, 0.1), plate=(0.0, -0.2, 0.1)):
    return {
        "cube": cube,
        "bowl": bowl,
        "plate": plate,
        "gripper_holding": True,
        "cube_height_lift_m": max(0.0, cube[2] - 0.1),
        "final_detached_release": True,
        "stable_for_seconds": True,
    }


def test_all_18_prompt_semantics_have_correct_signed_relation():
    for family in ("LAT", "HEIGHT", "DIST"):
        for sign in (1, -1):
            positive = _state(
                cube=(0.0, 0.05 if sign == 1 else -0.05, 0.1),
                bowl=(0.0, 0.0, 0.1),
            )
            if family == "HEIGHT":
                positive = _state(
                    cube=(0.0, 0.0, 0.15 if sign == 1 else 0.05),
                    bowl=(0.0, 0.0, 0.1),
                )
            if family == "DIST":
                positive = _state(
                    cube=(0.0, 0.0, 0.1),
                    bowl=(0.0, 0.05 if sign == 1 else 0.2, 0.1),
                    plate=(0.0, 0.2 if sign == 1 else 0.05, 0.1),
                )
            value = relation_m(family, positive["cube"], positive["bowl"], positive["plate"])
            assert sign * value >= 0
            if family == "LAT":
                negative = _state(cube=(0.0, 0.05 if sign == -1 else -0.05, 0.1))
            elif family == "HEIGHT":
                negative = _state(
                    cube=(0.0, 0.0, 0.05 if sign == 1 else 0.15),
                    bowl=(0.0, 0.0, 0.1),
                )
            else:
                negative = _state(
                    cube=(0.0, 0.0, 0.1),
                    bowl=(0.0, 0.2 if sign == 1 else 0.05, 0.1),
                    plate=(0.0, 0.05 if sign == 1 else 0.2, 0.1),
                )
            assert relation_m(family, negative["cube"], negative["bowl"], negative["plate"]) * sign < 0


def test_pickup_requires_three_consecutive_steps_and_success_is_not_early_stop():
    states = [_state(cube=(0.0, 0.0, 0.1),) for _ in range(451)]
    for i in range(2, 451):
        states[i]["cube"] = (0.0, 0.0, 0.14)
        states[i]["cube_height_lift_m"] = 0.04
        states[i]["sim_time_s"] = i * 0.01
        states[i]["supported"] = True
        states[i]["linear_speed_m_s"] = 0.0
        states[i]["angular_speed_rad_s"] = 0.0
    for state in states[:2]:
        state["gripper_holding"] = False
        state["cube_height_lift_m"] = 0.0
        state["sim_time_s"] = states.index(state) * 0.01
    result = score_episode(_episode(states[-1], states=states), GoalSpec("LAT", 1))
    assert result.requested_success is False
    assert result.pickup_step == 2
    assert result.terminal_step == 450
    assert result.first_success_step == 120


def test_anchor_drift_fails_success_and_is_retained_as_model_outcome():
    states = [_state(cube=(0.0, 0.05, 0.1)) for _ in range(451)]
    for i, state in enumerate(states):
        state["cube"] = (0.0, 0.05, 0.14) if i >= 1 else (0.0, 0.05, 0.1)
        state["cube_height_lift_m"] = 0.04 if i >= 1 else 0.0
        state["gripper_holding"] = i >= 1
        state["sim_time_s"] = i * 0.01
        state["supported"] = True
        state["linear_speed_m_s"] = 0.0
        state["angular_speed_rad_s"] = 0.0
    states[-1]["bowl"] = (0.0, 0.006, 0.1)
    result = score_episode(_episode(states[-1], states=states), GoalSpec("LAT", 1))
    assert result.status is OutcomeStatus.VALID_MODEL
    assert result.requested_success is False
    assert result.failure_stage == "anchor_disturbance"


def test_safety_censor_has_s0_but_no_terminal_margin():
    result = score_episode(
        _episode(_state(), states=[_state()] * 120, safety_terminated=True, termination_reason="safety",
                 terminal_observed=False),
        GoalSpec("LAT", 1),
    )
    assert result.status is OutcomeStatus.SAFETY_CENSORED
    assert result.requested_success is False
    assert result.terminal_margin_m is None


def test_infrastructure_has_no_model_outcome():
    result = score_episode({"infrastructure_reason": "renderer unavailable"}, GoalSpec("LAT", 1))
    assert result.status is OutcomeStatus.INFRA_INVALID
    assert result.requested_success is None
    assert canonical_status(result) == "technical_invalid"


def test_short_trace_is_technical_missingness_even_if_flagged_terminal():
    state = _state(cube=(0.0, 0.05, 0.14))
    state["cube_height_lift_m"] = 0.0
    result = score_episode(
        {"states": [state], "terminal_observed": True},
        GoalSpec("LAT", 1),
    )
    assert result.status is OutcomeStatus.INFRA_INVALID
    assert result.requested_success is None


def test_nonfinite_geometry_and_missing_anchor_fail_closed():
    state = _state()
    state["cube"] = (float("nan"), 0.0, 0.1)
    result = score_episode({"states": [state] * 451, "terminal_observed": True}, GoalSpec("LAT", 1))
    assert result.status is OutcomeStatus.INFRA_INVALID
    state = _state()
    del state["bowl"]
    result = score_episode({"states": [state] * 451, "terminal_observed": True}, GoalSpec("LAT", 1))
    assert result.status is OutcomeStatus.INFRA_INVALID
