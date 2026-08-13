from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_v3e006_r009_closure import (
    OFFICIAL_BBOX_DOC,
    STATE_GATE,
    attachment_finding,
    attachment_rows,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r009/results/results.json"
)


def _geometry_attempts() -> list[dict]:
    geometry = {
        "live_tensor_collision_geometry": {
            "left": {
                "live_tensor_pose": {"position_env_local_m": [0.0, -0.04, 0.0]},
                "reconstructed_bounds_env_local": {
                    "collision_center_env_local_m": [0.0, -0.172, 0.0]
                },
            },
            "right": {
                "live_tensor_pose": {"position_env_local_m": [0.0, 0.04, 0.0]},
                "reconstructed_bounds_env_local": {
                    "collision_center_env_local_m": [0.0, 0.172, 0.0]
                },
            },
        },
        "pinch_alignment_command": {
            "inner_finger_collision_center_separation_m": 0.344
        },
    }
    state = {
        "passed": False,
        "construction": {
            "acquisition_lift_transport_trace": [
                {"pre_action_pinch_geometry": copy.deepcopy(geometry)}
            ]
            * 1020
        },
        "physics_gate": {
            "observed": {
                "max_unintended_contact_force_n_by_pair": {
                    "rubiks_cube__table": 5.0
                },
                "object_grabbed_all_steps": True,
                "minimum_intended_cube_gripper_contact_force_n": 4.0,
            }
        },
    }
    return [
        {
            "candidate_rank": rank,
            "stages": {
                "canonical_grasp": {"candidate_state": copy.deepcopy(state)},
                "canonical_carry": {"candidate_state": copy.deepcopy(state)},
            },
        }
        for rank in range(1, 5)
    ]


def test_attachment_geometry_recomputation_and_mutations() -> None:
    attempts = _geometry_attempts()
    rows = attachment_rows(attempts)
    source = {"path": "state_repair_gate.py", "bytes": 1, "sha256": "0" * 64}
    finding = attachment_finding(rows, source)
    assert finding["intended_collision_pinch_semantics_attachment_valid"] is False
    assert finding["quantitative_signature"]["grabbed_all_final_steps_stage_count"] == 8
    assert finding["official_openusd_reference"]["url"] == OFFICIAL_BBOX_DOC

    bad = copy.deepcopy(attempts)
    bad[0]["stages"]["canonical_grasp"]["candidate_state"]["construction"][
        "acquisition_lift_transport_trace"
    ][-1]["pre_action_pinch_geometry"]["pinch_alignment_command"][
        "inner_finger_collision_center_separation_m"
    ] = 0.3
    with pytest.raises(ValueError, match="pad separation"):
        attachment_rows(bad)

    for key, value in (
        ("final_reconstructed_pad_center_separation_m", 0.2),
        ("max_cube_table_contact_force_n", 0.1),
        ("state_passed", True),
    ):
        mutated = copy.deepcopy(rows)
        mutated[0][key] = value
        with pytest.raises(ValueError):
            attachment_finding(mutated, source)


def test_bound_buggy_source_contains_exact_double_transform() -> None:
    text = STATE_GATE.read_text(encoding="utf-8")
    assert "bbox_cache.ComputeLocalBound(prim).ComputeAlignedRange()" in text
    assert "world_body.Transform(prim_world.Transform(corner))" in text
    assert "ComputeRelativeBound(prim, body)" not in text


def test_compiled_closure_is_explicitly_nonreleasable() -> None:
    value = json.loads(RESULT.read_text(encoding="utf-8"))
    assert value["mechanically_valid_frozen_execution"] is True
    assert value["intended_collision_pinch_semantics_attachment_valid"] is False
    assert value["intended_collision_pinch_algorithm_scientifically_exhausted"] is False
    assert value["behavioral_activation_released"] is False
    assert value["model_request_count"] == value["behavioral_episode_count"] == 0
    signature = value["geometry_attachment_finding"]["quantitative_signature"]
    assert signature["reconstructed_pad_center_separation_m_min"] > 0.3
    assert signature["finger_body_origin_separation_m_max"] < 0.1
    assert signature["all_stages_table_loaded"] is True
