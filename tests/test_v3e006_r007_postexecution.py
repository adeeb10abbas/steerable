from __future__ import annotations

from copy import deepcopy

import pytest

from tools import validate_v3e006_r007_postexecution as corrected
def test_exact_runtime_rank_annotation_is_required_and_normalized(monkeypatch) -> None:
    expected_stage = {"stage": "canonical_carry", "target": {"x": 1}}
    state = {
        "construction": {
            "candidate_rank": 3,
            "registered_stage_schedule": {
                **deepcopy(expected_stage),
                "candidate_rank": 3,
            },
        }
    }
    observed = {}

    def frozen_stub(normalized, label, expected, *, candidate_rank):
        observed["schedule"] = normalized["construction"]["registered_stage_schedule"]
        observed["label"] = label
        observed["rank"] = candidate_rank
        assert expected == expected_stage

    monkeypatch.setattr(corrected, "_FROZEN_VALIDATE_OPEN_CONTACT_STATE", frozen_stub)
    corrected.validate_open_contact_state(
        state, "canonical_carry", expected_stage, candidate_rank=3
    )
    assert observed == {
        "schedule": expected_stage,
        "label": "canonical_carry",
        "rank": 3,
    }
    assert state["construction"]["registered_stage_schedule"]["candidate_rank"] == 3

    for retained in (
        expected_stage,
        {**expected_stage, "candidate_rank": 2},
        {**expected_stage, "candidate_rank": 3, "unregistered": True},
    ):
        bad = deepcopy(state)
        bad["construction"]["registered_stage_schedule"] = retained
        with pytest.raises(corrected.frozen.ValidationError):
            corrected.validate_open_contact_state(
                bad, "canonical_carry", expected_stage, candidate_rank=3
            )
