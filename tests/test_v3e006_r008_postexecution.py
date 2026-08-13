from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from tools import validate_v3e006_r008_postexecution as corrected


def evidence(quaternion: list[float]) -> dict[str, object]:
    normalized = np.asarray(quaternion, dtype=np.float64)
    normalized = (normalized / np.linalg.norm(normalized)).tolist()
    return {
        "command_base_position_world_m": [0.25, -0.01, 0.31],
        "command_base_quaternion_world_wxyz": normalized,
    }


def test_runtime_action_uses_exact_frozen_sign_canonicalization() -> None:
    row = evidence([-0.8, 0.1, -0.5, 0.2])
    mathematical = corrected.mathematical_action_float32(row)
    runtime = corrected.runtime_action_float32(row)
    assert mathematical[:3] == runtime[:3]
    assert mathematical[7:] == runtime[7:]
    assert np.array_equal(
        np.asarray(mathematical[3:7], dtype=np.float32),
        -np.asarray(runtime[3:7], dtype=np.float32),
    )
    assert not corrected.same_float32(mathematical, runtime)

    positive = evidence([0.8, -0.1, 0.5, -0.2])
    assert corrected.mathematical_action_float32(positive) == corrected.runtime_action_float32(positive)


def test_sign_boundary_and_non_antipode_mutations_fail() -> None:
    row = evidence([0.0, -1.0, 0.0, 0.0])
    runtime = corrected.runtime_action_float32(row)
    assert runtime[3:7] == [0.0, 1.0, 0.0, 0.0]

    mutated_position = deepcopy(runtime)
    mutated_position[0] = float(np.nextafter(np.float32(runtime[0]), np.float32(np.inf)))
    assert not corrected.same_float32(mutated_position, runtime)

    mutated_rotation = deepcopy(runtime)
    mutated_rotation[4] = float(np.nextafter(np.float32(runtime[4]), np.float32(np.inf)))
    assert not corrected.same_float32(mutated_rotation, runtime)

    mutated_grip = deepcopy(runtime)
    mutated_grip[7] = 0.0
    assert not corrected.same_float32(mutated_grip, runtime)


def test_normalization_requires_exact_profile_and_preserves_input(monkeypatch) -> None:
    state = {
        "construction": {
            "construction_action_trace": [
                {"command_action_8d": [0.0] * 8} for _ in range(1290)
            ],
            "object_space_servo_trace": [],
        }
    }
    row_evidence = evidence([-0.8, 0.1, -0.5, 0.2])
    runtime = corrected.runtime_action_float32(row_evidence)
    for _ in range(360):
        state["construction"]["object_space_servo_trace"].append(
            {
                "pre_action_object_space_servo": deepcopy(row_evidence),
                "command_action_8d": deepcopy(runtime),
            }
        )
    state["construction"]["construction_action_trace"][330:690] = deepcopy(
        state["construction"]["object_space_servo_trace"]
    )
    original = deepcopy(state)
    monkeypatch.setitem(
        corrected.EXPECTED_ANTIPODE_COUNTS, (1, "canonical_carry"), 360
    )
    normalized = corrected.normalize_servo_actions_for_frozen_validator(
        state, label="canonical_carry", candidate_rank=1
    )
    mathematical = corrected.mathematical_action_float32(row_evidence)
    assert normalized["construction"]["object_space_servo_trace"][0]["command_action_8d"] == mathematical
    assert normalized["construction"]["construction_action_trace"][330]["command_action_8d"] == mathematical
    assert state == original

    bad = deepcopy(state)
    bad["construction"]["object_space_servo_trace"][60]["command_action_8d"][4] = 0.25
    bad["construction"]["construction_action_trace"][390]["command_action_8d"][4] = 0.25
    with pytest.raises(corrected.frozen.ValidationError):
        corrected.normalize_servo_actions_for_frozen_validator(
            bad, label="canonical_carry", candidate_rank=1
        )


def test_candidate_state_delegates_only_after_copy_normalization(monkeypatch) -> None:
    expected_stage = {"stage": "canonical_grasp"}
    state = {"original": True}
    normalized = {"normalized": True}
    observed = {}

    def normalize(value, *, label, candidate_rank):
        assert value is state
        assert label == "canonical_grasp"
        assert candidate_rank == 2
        return normalized

    def frozen_stub(value, expected, rank, schedule):
        observed.update(value=value, expected=expected, rank=rank, schedule=schedule)

    monkeypatch.setattr(corrected, "normalize_servo_actions_for_frozen_validator", normalize)
    monkeypatch.setattr(corrected, "_FROZEN_VALIDATE_CANDIDATE_STATE", frozen_stub)
    corrected.validate_candidate_state(state, expected_stage, 2, {"frozen": True})
    assert observed == {
        "value": normalized,
        "expected": expected_stage,
        "rank": 2,
        "schedule": {"frozen": True},
    }
