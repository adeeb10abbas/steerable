from __future__ import annotations

from pathlib import Path

import numpy as np

from tools.compile_v3c_phase_c_results import (
    PROMPT_FAMILIES,
    SEEDS,
    exact_mcnemar,
    summarize,
    wilson_interval,
)


def test_wilson_interval_is_bounded() -> None:
    assert wilson_interval(0, 20)[0] == 0.0
    assert wilson_interval(20, 20)[1] == 1.0
    low, high = wilson_interval(10, 20)
    assert low < 0.5 < high


def test_exact_mcnemar_uses_discordant_pairs() -> None:
    result = exact_mcnemar(
        [False, False, False, True],
        [True, True, True, True],
    )
    assert result["left_only"] == 0
    assert result["right_only"] == 3
    assert result["both_success"] == 1
    assert result["exact_two_sided_p"] == 0.25


def test_summary_keeps_phrasing_direction_and_pairing(tmp_path: Path) -> None:
    left_actions = tmp_path / "left.npy"
    right_actions = tmp_path / "right.npy"
    np.save(left_actions, np.zeros((10, 8), dtype=np.float32))
    np.save(right_actions, np.ones((10, 8), dtype=np.float32))
    rows = []
    for family in PROMPT_FAMILIES:
        for seed in SEEDS:
            for relation in ("left", "right"):
                success = relation == "right" or seed % 2 == 0
                rows.append({
                    "seed": seed,
                    "prompt_family": family,
                    "relation": relation,
                    "requested_success": success,
                    "failure_taxonomy": "correct" if success else "transport_failed",
                    "model_request_count": None,
                    "measurements": {
                        "signed_final_lateral_offset_m": 0.1 if relation == "left" else -0.1,
                    },
                    "artifacts": {
                        "executed_actions": {
                            "path": str(left_actions if relation == "left" else right_actions),
                        },
                    },
                })
    result = summarize(rows, model_id="groot_n17_droid_vla")
    assert result["counts"]["valid_behavioral_episodes"] == 160
    assert result["success_by_condition"]["direct_command:left"]["successes"] == 10
    assert result["success_by_condition"]["direct_command:right"]["successes"] == 20
    paired = result["paired_diagnostics_by_prompt_family"]["direct_command"]
    assert paired["endpoint_ordering_aligned"] == 20
    assert paired["endpoint_ordering_anti_aligned"] == 0
    assert "positive toward robot LEFT" in paired["endpoint_shift_definition"]
    assert paired["first_10_executed_actions_distinct"] == 20
    assert paired["success_discordance"]["right_only"] == 10
