from __future__ import annotations

from pathlib import Path

import pytest

from experiments.v3.cosmos_nano_tier_b.analyze_results import _factor_contrasts
from experiments.v3.cosmos_nano_tier_b.runtime_contract import load_release


ROOT = Path(__file__).resolve().parents[1]


def _release(amendment_id: str):
    directory = amendment_id.lower().replace("-", "")
    return load_release(
        ROOT,
        amendment_id,
        ROOT
        / "artifacts/vla_wam_shared_v3/prospective_tier_b/releases"
        / directory
        / "release_manifest.json",
    )


def test_role_swap_interaction_is_second_level_minus_first_level() -> None:
    release = _release("V3-B009")
    first, second = release.config["arms"]
    pairs = {}
    for seed in release.config["seed_range"]:
        pairs[(seed, first)] = {
            "endpoint_redirection_D_m": 0.1,
            "requested_side_depth_contrast_B_m": 0.2,
            "right_minus_left_success": 1,
        }
        pairs[(seed, second)] = {
            "endpoint_redirection_D_m": 0.4,
            "requested_side_depth_contrast_B_m": -0.1,
            "right_minus_left_success": 0,
        }
    result = _factor_contrasts(
        pairs,
        release=release,
        replicates=10_000,
        bootstrap_seed=3_104_159,
    )
    contrast = result["pairwise_factor_interactions"][f"{second}_minus_{first}"]
    assert contrast["endpoint_redirection_interaction_m"]["mean_m"] == 0.30000000000000004
    assert contrast["requested_side_depth_interaction_m"]["mean_m"] == -0.30000000000000004
    binary = contrast["binary_success_interaction"]
    assert binary["mean"] == -1.0
    assert binary["exact_permutation_test"]["p_value"] < 1e-7
    assert binary["exact_permutation_test"]["method"].endswith(
        "factor_label_sign_flip_permutation"
    )


def test_start_side_emits_all_three_pairwise_interactions() -> None:
    release = _release("V3-B008")
    pairs = {
        (seed, arm): {
            "endpoint_redirection_D_m": float(index),
            "requested_side_depth_contrast_B_m": float(index) / 10,
            "right_minus_left_success": 0,
        }
        for seed in release.config["seed_range"]
        for index, arm in enumerate(release.config["arms"])
    }
    result = _factor_contrasts(
        pairs,
        release=release,
        replicates=10_000,
        bootstrap_seed=3_104_159,
    )
    assert len(result["pairwise_factor_interactions"]) == 3
    trend = result["ordered_start_side_trend"]
    assert trend["factor_levels_m"] == {
        "target_start_left": 0.1,
        "target_start_center": 0.0,
        "target_start_right": -0.1,
    }
    assert trend["endpoint_redirection_D_slope_per_m"]["mean_m"] == pytest.approx(-10.0)
    extreme = result["pairwise_factor_interactions"][
        "target_start_right_minus_target_start_left"
    ]
    assert extreme["endpoint_redirection_interaction_m"]["mean_m"] == 2.0
