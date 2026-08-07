from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.v3.dreamzero_phase_b.analyze_results import analyze
from experiments.v3.dreamzero_phase_b.contract import (
    AMENDMENT_ID,
    ARMS,
    MODEL_ID,
    PROMPTS,
    RELATIONS,
    SEEDS,
    STUDY_ID,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dreamzero_registered_analysis_reuses_the_exact_mirror_design(tmp_path: Path) -> None:
    rows = []
    for seed in SEEDS:
        for arm in ARMS:
            initial = f"{seed}:{arm}:identical-reset"
            for relation in RELATIONS:
                success = relation == "right" or arm == "position_mirrored"
                offset = (0.1 if relation == "left" else -0.2) + (
                    0.05 if arm == "position_mirrored" else 0.0
                )
                action = np.full((2, 8), float(seed) + (relation == "right"), dtype=np.float32)
                path = tmp_path / f"{seed}_{arm}_{relation}.npy"
                np.save(path, action, allow_pickle=False)
                rows.append({
                    "record_type": "behavioral_episode",
                    "behavioral_result_valid": True,
                    "study_id": STUDY_ID,
                    "amendment_id": AMENDMENT_ID,
                    "model_id": MODEL_ID,
                    "arena": "droid_robolab",
                    "registered_cell_id": f"v3b003:dreamzero:seed{seed}:{arm}:{relation}",
                    "environment_seed": seed,
                    "requested_relation": relation,
                    "prompt": PROMPTS[relation],
                    "arm": arm,
                    "pair_id": f"v3b003:dreamzero:seed{seed}",
                    "initial_state_sha256": initial,
                    "requested_success": success,
                    "failure_taxonomy": "correct" if success else "transport_failed",
                    "signed_final_lateral_offset_m": offset,
                    "requested_side_depth_m": offset if relation == "left" else -offset,
                    "actions_executed": len(action),
                    "artifacts": {
                        "executed_action_trace": {
                            "path": str(path),
                            "sha256": _sha(path),
                            "bytes": path.stat().st_size,
                        }
                    },
                })

    report = analyze(
        repo_root=Path(__file__).resolve().parents[1],
        episode_rows=rows,
        bootstrap_replicates=10_000,
        bootstrap_seed=13,
    )

    assert report["population"]["behavioral_episode_count"] == 108
    assert report["population"]["matched_left_right_pair_count"] == 54
    table = report["registered_analysis"]["H3_binary_success"]["cell_success_table_2x2"]
    assert table["control"]["left"]["successes"] == 0
    assert table["control"]["right"]["successes"] == 27
    assert table["position_mirrored"]["left"]["successes"] == 27
    assert table["position_mirrored"]["right"]["successes"] == 27
    assert report["requested_margin_secondary"]["realized_seed_n"] == 0
    assert report["condition_outcomes"]["control:left"] == {
        "episodes": 27,
        "successes": 0,
        "failure_taxonomy_counts": {"transport_failed": 27},
    }
    primary = report["full_sample_primary"]
    assert set(primary) == {
        "population",
        "formulas",
        "D_by_arm",
        "B_by_arm",
        "J_redirection_interaction",
        "I_position_reflection_interaction",
        "binary_success_DiD",
    }
    assert primary["binary_success_DiD"]["cell_success_table_2x2"] == table
    assert report["failure_taxonomy_counts"] == {
        "correct": 81,
        "pick_failed": 0,
        "transport_failed": 27,
        "wrong_side": 0,
        "release_failed": 0,
    }
