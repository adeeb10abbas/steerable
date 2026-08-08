from __future__ import annotations

import math

import pytest

from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.analysis import (
    compile_checkpoint,
    exact_sign_flip_permutation,
    exact_two_sided_binomial,
    wilson,
)


def rows():
    output = []
    levels = [(0.0, 3.0), (0.25, 2.25), (0.5, 1.5), (0.75, 0.75), (1.0, 0.0)]
    for seed in range(9400, 9427):
        for level, A in levels:
            for relation in ("left", "right"):
                endpoint = (0.10 + 0.01 * (seed - 9400)) if relation == "left" else (-0.12 - 0.01 * (seed - 9400))
                success = relation == "right" and level < 1.0
                output.append(
                    {
                        "cell_id": f"{seed}:{level}:{relation}",
                        "model_id": "synthetic",
                        "environment_seed": seed,
                        "symmetry_level_s": level,
                        "relation": relation,
                        "success": success,
                        "failure_category": "correct" if success else "wrong_side",
                        "requested_side_depth": max(0.0, (-endpoint if relation == "right" else endpoint)),
                        "signed_final_lateral_offset": endpoint,
                        "asymmetry_metric_A": A,
                    }
                )
    return output


def test_exact_binomial_and_wilson():
    assert exact_two_sided_binomial(0, 10) == pytest.approx(2 / 1024)
    interval = wilson(5, 10)
    assert interval["proportion"] == 0.5
    assert interval["wilson95_low"] < 0.5 < interval["wilson95_high"]


def test_exact_sign_flip_detects_consistent_direction():
    result = exact_sign_flip_permutation([1.0] * 10)
    assert result["permutations"] == 1024
    assert result["exact_two_sided_p"] == pytest.approx(2 / 1024)


def test_checkpoint_compiler_keeps_pairing_and_inventory_matched_slope():
    result = compile_checkpoint(
        rows(),
        model_id="synthetic",
        margins={"binary_gap": 0.2, "depth_gap_m": 0.05},
        power_status={"binary_gap": "test", "depth_gap_m": "test"},
        resamples=10_000,
    )
    assert result["levels"]["0.00"]["pairs"] == 27
    assert result["levels"]["1.00"]["left_success"]["successes"] == 0
    assert result["levels"]["0.00"]["right_success"]["successes"] == 27
    assert result["interaction_s1_minus_s0_core"]["binary_gap"]["mean"] == -1.0
    dose = result["dose_response_on_realised_A"]
    assert dose["s0_excluded_due_registered_inventory_transition"] is True
    assert dose["primary_levels"] == [0.25, 0.5, 0.75, 1.0]
    assert math.isfinite(dose["binary_gap"]["mean_slope"])
