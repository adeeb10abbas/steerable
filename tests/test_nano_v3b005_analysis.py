from __future__ import annotations

import math
from pathlib import Path

from experiments.v3.cosmos_nano_lateral_sweep.analyze_results import analyze
from experiments.v3.cosmos_nano_lateral_sweep.compile_pair import SCHEMA as PAIR_SCHEMA
from experiments.v3.cosmos_nano_lateral_sweep.runtime_adapter import (
    AMENDMENT_ID,
    LEVELS,
    MODEL_ID,
    SEEDS,
)
from tools.render_nano_v3b005_dose_response import render


def _fixture() -> tuple[list[dict], list[dict]]:
    center = sum(LEVELS) / len(LEVELS)
    pairs: list[dict] = []
    episodes: list[dict] = []
    for seed in SEEDS:
        for level, y_m in enumerate(LEVELS):
            left_success = level >= 2
            right_success = level <= 4
            pairs.append({
                "schema_version": PAIR_SCHEMA,
                "model_id": MODEL_ID,
                "seed": seed,
                "level_index": level,
                "reference_object_initial_lateral_position_y_m": y_m,
                "requested_side_depth_contrast_B_m": 2.0 * (y_m - center),
                "endpoint_redirection_D_m": 0.1,
                "left_success": left_success,
                "right_success": right_success,
            })
            for relation, success in (("left", left_success), ("right", right_success)):
                episodes.append({
                    "behavioral_result_valid": True,
                    "model_id": MODEL_ID,
                    "amendment_id": AMENDMENT_ID,
                    "environment_seed": seed,
                    "level_index": level,
                    "requested_relation": relation,
                    "requested_success": success,
                    "failure_taxonomy": "correct" if success else "transport_failed",
                    "nano_v3b005_diagnostics": {"requested_side_depth_m": 0.05},
                })
    return pairs, episodes


def test_registered_dose_response_analysis_uses_all_matched_cells() -> None:
    pairs, episodes = _fixture()
    report = analyze(pairs, episodes, bootstrap_replicates=10_000, bootstrap_seed=7)

    assert report["population"] == {
        "matched_seed_count": 15,
        "level_count": 7,
        "matched_pair_count": 105,
        "behavioral_episode_count": 210,
        "valid_behavioral_failures_included": True,
        "infrastructure_attempts_included": False,
        "missing_value_imputation": "none",
    }
    primary = report["primary_depth_dose_response"]
    assert math.isclose(primary["slope_m_per_m"]["mean"], 2.0)
    assert primary["slope_m_per_m"]["sign_test"]["positive"] == 15
    crossing = primary["population_linear_zero_crossing"]
    assert crossing["in_registered_support"] is True
    assert math.isclose(crossing["reference_object_lateral_y_m"], sum(LEVELS) / 7)
    assert len(report["by_level"]) == 7
    assert report["by_level"][0]["binary_success"]["left"]["successes"] == 0
    assert report["by_level"][0]["binary_success"]["right"]["successes"] == 15
    assert report["failure_taxonomy_counts"]["0"]["left"]["transport_failed"] == 15


def test_registered_dose_response_figures_render_complete_report(tmp_path: Path) -> None:
    pairs, episodes = _fixture()
    report = analyze(pairs, episodes, bootstrap_replicates=10_000, bootstrap_seed=7)

    outputs = render(report, tmp_path)

    assert {path.name for path in outputs} == {
        "figure3_nano_lateral_dose_response.png",
        "figure3_nano_lateral_dose_response.svg",
        "nano_v3b005_failure_taxonomy_by_level.png",
        "nano_v3b005_failure_taxonomy_by_level.svg",
    }
    assert all(path.stat().st_size > 1_000 for path in outputs)
