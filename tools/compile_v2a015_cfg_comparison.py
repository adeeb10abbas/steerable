#!/usr/bin/env python3
"""Compile paired baseline/intervention evidence for both V2-A015 arms.

The comparison is descriptive at six cells per arm.  It reports exact paired
success transitions and effect sizes; it does not manufacture p-values or
promote the post-result pilot to a powered general claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

import v2a015_compilation as shared
from compile_v2a015_cosmos_nano import (
    BASELINE_SHA256 as COSMOS_BASELINE_SHA256,
    MODEL_ID as COSMOS_MODEL_ID,
    SCHEMA as COSMOS_INTERVENTION_SCHEMA,
)
from compile_v2a015_dreamzero import (
    BASELINE_SHA256 as DREAMZERO_BASELINE_SHA256,
    MODEL_ID as DREAMZERO_MODEL_ID,
    SCHEMA as DREAMZERO_INTERVENTION_SCHEMA,
)


SCHEMA = "vla-wam-shared-v2-cfg-ablation-v2a015-comparison-v1"


def _raw_record(episode: dict[str, Any], key: str) -> dict[str, Any]:
    if "simulator_artifacts" in episode:
        mapping = {
            "hdf5": "rollout_hdf5",
            "log": "episode_log",
            "video": "viewport_video",
        }
        return episode["simulator_artifacts"][mapping[key]]
    mapping = {"hdf5": "raw_hdf5", "log": "raw_log", "video": "simulator_video"}
    if key == "video" and mapping[key] not in episode:
        mapping[key] = "executed_video"
    return episode[mapping[key]]


def _normalize_episode(episode: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    import h5py

    seed = int(episode["environment_seed"])
    relation = episode["requested_relation"]
    if seed not in shared.SEEDS or episode.get("prompt") != shared.PROMPTS[relation]:
        raise RuntimeError(f"Compiled episode prompt/grid mismatch: {seed}/{relation}")
    hdf5_record = _raw_record(episode, "hdf5")
    hdf5_path = shared.validate_file_record(hdf5_record, Path.cwd(), "comparison HDF5")
    action_path = shared.validate_file_record(
        episode["executed_action_trace"], Path.cwd(), "comparison executed action trace"
    )
    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data/demo_0"]
        hdf5_actions = np.asarray(demo["actions"], dtype=np.float32)
        cube = np.asarray(demo["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64)
        bowl = np.asarray(demo["states/rigid_object/bowl/root_pose"], dtype=np.float64)
        robot = np.asarray(demo["states/articulation/robot/root_pose"], dtype=np.float64)
        fingerprint = shared.initial_fingerprint(demo["initial_state"])
    actions = np.load(action_path, allow_pickle=False)
    if actions.shape != hdf5_actions.shape or not np.array_equal(actions, hdf5_actions):
        raise RuntimeError(f"Compiled action/HDF5 mismatch during comparison: {seed}/{relation}")
    delta = shared.robot_frame_delta(cube, bowl, robot)
    final_display = float(-delta[-1, 1])
    if not np.isclose(final_display, float(episode["final_lateral_display_m"]), atol=1e-9):
        raise RuntimeError(f"Compiled endpoint differs from raw HDF5: {seed}/{relation}")
    requested_margin = -final_display if relation == "left" else final_display
    if "requested_signed_final_margin_m" in episode and not np.isclose(
        requested_margin, float(episode["requested_signed_final_margin_m"]), atol=1e-9
    ):
        raise RuntimeError(f"Compiled requested margin differs from raw HDF5: {seed}/{relation}")
    if episode.get("physical_initial_state_sha256") != fingerprint:
        raise RuntimeError(f"Compiled initial-state fingerprint differs from HDF5: {seed}/{relation}")
    success = bool(episode["requested_success"])
    steps = int(episode["actions_executed"])
    if success and not 0 < steps <= shared.ACTION_CAP:
        raise RuntimeError(f"Invalid observed completion step: {seed}/{relation}/{steps}")
    if not success and steps != shared.ACTION_CAP:
        raise RuntimeError(f"Failure is not censored at action cap: {seed}/{relation}/{steps}")
    normalized = {
        **episode,
        "requested_signed_final_margin_m": requested_margin,
        "trajectory_quality": shared.trajectory_quality(cube, delta, relation),
        "action_quality": shared.action_quality(actions),
        "completion_actions_observed": steps if success else None,
        "completion_action_status": (
            "observed_success_event"
            if success
            else f"right_censored_at_{shared.ACTION_CAP}_action_cap"
        ),
    }
    return normalized, actions


def _normalize_result(
    result: dict[str, Any], *, expected_episode_model: str | None = None
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], np.ndarray]]:
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 6:
        raise RuntimeError("Every compared configuration must contain exactly six episodes")
    shared.validate_complete_grid(episodes)
    normalized, actions = [], {}
    for episode in episodes:
        if expected_episode_model is not None and episode.get("model_id") != expected_episode_model:
            raise RuntimeError(
                f"Unexpected baseline episode model: {episode.get('model_id')!r}"
            )
        row, action = _normalize_episode(episode)
        key = (int(row["environment_seed"]), row["requested_relation"])
        normalized.append(row)
        actions[key] = action
    return normalized, actions


def _transition(before: bool, after: bool) -> str:
    if not before and after:
        return "improved_failure_to_success"
    if before and not after:
        return "regressed_success_to_failure"
    return "unchanged_success" if before else "unchanged_failure"


def _numeric_effect(values_by_cell: list[tuple[str, float]]) -> dict[str, Any]:
    values = [value for _, value in values_by_cell]
    if not values:
        return {
            "values_by_cell": [],
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "values_by_cell": [
            {"cell_id": cell_id, "value": value}
            for cell_id, value in values_by_cell
        ],
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "minimum": min(values),
        "maximum": max(values),
    }


def _difference(after: float | int | None, before: float | int | None) -> float | int | None:
    if after is None or before is None:
        return None
    return after - before


def _compare_configuration(
    *,
    model_label: str,
    baseline_label: str,
    intervention_label: str,
    first_chunk_horizon: int,
    baseline_result: dict[str, Any],
    intervention_result: dict[str, Any],
    baseline_episode_model: str,
) -> dict[str, Any]:
    before_rows, before_actions = _normalize_result(
        baseline_result, expected_episode_model=baseline_episode_model
    )
    after_rows, after_actions = _normalize_result(intervention_result)
    before = {(row["environment_seed"], row["requested_relation"]): row for row in before_rows}
    after = {(row["environment_seed"], row["requested_relation"]): row for row in after_rows}
    cells = []
    for seed in shared.SEEDS:
        for relation in shared.PROMPTS:
            key = (seed, relation)
            old, new = before[key], after[key]
            if old["physical_initial_state_sha256"] != new[
                "physical_initial_state_sha256"
            ]:
                raise RuntimeError(
                    f"Baseline/intervention physical reset differs: {model_label}/{seed}/{relation}"
                )
            old_quality = old["trajectory_quality"]
            new_quality = new["trajectory_quality"]
            old_action = old["action_quality"]
            new_action = new["action_quality"]
            completion_delta = _difference(
                new["completion_actions_observed"], old["completion_actions_observed"]
            )
            cells.append(
                {
                    "cell_id": f"seed{seed}_{relation}",
                    "environment_seed": seed,
                    "requested_relation": relation,
                    "prompt": shared.PROMPTS[relation],
                    "physical_initial_state_sha256": old[
                        "physical_initial_state_sha256"
                    ],
                    "success": {
                        baseline_label: bool(old["requested_success"]),
                        intervention_label: bool(new["requested_success"]),
                        "transition": _transition(
                            bool(old["requested_success"]), bool(new["requested_success"])
                        ),
                        "numeric_delta": int(bool(new["requested_success"]))
                        - int(bool(old["requested_success"])),
                    },
                    "requested_signed_final_margin_m": {
                        baseline_label: old["requested_signed_final_margin_m"],
                        intervention_label: new["requested_signed_final_margin_m"],
                        "intervention_minus_baseline": new[
                            "requested_signed_final_margin_m"
                        ]
                        - old["requested_signed_final_margin_m"],
                    },
                    "trajectory_quality": {
                        "cube_path_length_3d_m": {
                            baseline_label: old_quality["cube_path_length_3d_m"],
                            intervention_label: new_quality["cube_path_length_3d_m"],
                            "intervention_minus_baseline": new_quality[
                                "cube_path_length_3d_m"
                            ]
                            - old_quality["cube_path_length_3d_m"],
                        },
                        "cube_net_displacement_3d_m": {
                            baseline_label: old_quality["cube_net_displacement_3d_m"],
                            intervention_label: new_quality["cube_net_displacement_3d_m"],
                            "intervention_minus_baseline": new_quality[
                                "cube_net_displacement_3d_m"
                            ]
                            - old_quality["cube_net_displacement_3d_m"],
                        },
                        "cube_excess_path_ratio": {
                            baseline_label: old_quality["cube_excess_path_ratio"],
                            intervention_label: new_quality["cube_excess_path_ratio"],
                            "intervention_minus_baseline": _difference(
                                new_quality["cube_excess_path_ratio"],
                                old_quality["cube_excess_path_ratio"],
                            ),
                        },
                        "cube_max_excursion_from_initial_3d_m": {
                            baseline_label: old_quality[
                                "cube_max_excursion_from_initial_3d_m"
                            ],
                            intervention_label: new_quality[
                                "cube_max_excursion_from_initial_3d_m"
                            ],
                            "intervention_minus_baseline": new_quality[
                                "cube_max_excursion_from_initial_3d_m"
                            ]
                            - old_quality["cube_max_excursion_from_initial_3d_m"],
                        },
                        "cube_max_requested_lateral_excursion_from_initial_m": {
                            baseline_label: old_quality[
                                "cube_max_requested_lateral_excursion_from_initial_m"
                            ],
                            intervention_label: new_quality[
                                "cube_max_requested_lateral_excursion_from_initial_m"
                            ],
                            "intervention_minus_baseline": new_quality[
                                "cube_max_requested_lateral_excursion_from_initial_m"
                            ]
                            - old_quality[
                                "cube_max_requested_lateral_excursion_from_initial_m"
                            ],
                        },
                        "cube_max_opposite_lateral_excursion_from_initial_m": {
                            baseline_label: old_quality[
                                "cube_max_opposite_lateral_excursion_from_initial_m"
                            ],
                            intervention_label: new_quality[
                                "cube_max_opposite_lateral_excursion_from_initial_m"
                            ],
                            "intervention_minus_baseline": new_quality[
                                "cube_max_opposite_lateral_excursion_from_initial_m"
                            ]
                            - old_quality[
                                "cube_max_opposite_lateral_excursion_from_initial_m"
                            ],
                        },
                    },
                    "executed_action_quality": {
                        "joint_action_total_variation_l2": {
                            baseline_label: old_action["joint_action_total_variation_l2"],
                            intervention_label: new_action[
                                "joint_action_total_variation_l2"
                            ],
                            "intervention_minus_baseline": new_action[
                                "joint_action_total_variation_l2"
                            ]
                            - old_action["joint_action_total_variation_l2"],
                        },
                        "joint_action_mean_l2_per_transition": {
                            baseline_label: old_action[
                                "joint_action_mean_l2_per_transition"
                            ],
                            intervention_label: new_action[
                                "joint_action_mean_l2_per_transition"
                            ],
                            "intervention_minus_baseline": new_action[
                                "joint_action_mean_l2_per_transition"
                            ]
                            - old_action["joint_action_mean_l2_per_transition"],
                        },
                        "gripper_switch_count": {
                            baseline_label: old_action["gripper_switch_count"],
                            intervention_label: new_action["gripper_switch_count"],
                            "intervention_minus_baseline": new_action[
                                "gripper_switch_count"
                            ]
                            - old_action["gripper_switch_count"],
                        },
                    },
                    "completion_actions": {
                        baseline_label: old["completion_actions_observed"],
                        intervention_label: new["completion_actions_observed"],
                        "intervention_minus_baseline_when_both_observed": completion_delta,
                        "comparison_status": (
                            "both_success_events_observed"
                            if completion_delta is not None
                            else "not_compared_because_at_least_one_cell_is_right_censored"
                        ),
                    },
                }
            )

    before_pairs = shared.build_pairs(before_rows, before_actions, horizon=first_chunk_horizon)
    after_pairs = shared.build_pairs(after_rows, after_actions, horizon=first_chunk_horizon)
    pair_comparisons = []
    for seed in shared.SEEDS:
        old = next(row for row in before_pairs if row["environment_seed"] == seed)
        new = next(row for row in after_pairs if row["environment_seed"] == seed)
        pair_comparisons.append(
            {
                "pair_id": f"droid_pair_seed_{seed}",
                "environment_seed": seed,
                "left_prompt": shared.PROMPTS["left"],
                "right_prompt": shared.PROMPTS["right"],
                "endpoint_separation_right_minus_left_m": {
                    baseline_label: old["endpoint_separation_right_minus_left_m"],
                    intervention_label: new["endpoint_separation_right_minus_left_m"],
                    "intervention_minus_baseline": new[
                        "endpoint_separation_right_minus_left_m"
                    ]
                    - old["endpoint_separation_right_minus_left_m"],
                },
                "seed_balance_gap_right_minus_left_margin_m": {
                    baseline_label: old[
                        "seed_balance_gap_right_minus_left_margin_m"
                    ],
                    intervention_label: new[
                        "seed_balance_gap_right_minus_left_margin_m"
                    ],
                    "intervention_minus_baseline": new[
                        "seed_balance_gap_right_minus_left_margin_m"
                    ]
                    - old["seed_balance_gap_right_minus_left_margin_m"],
                },
                "seed_absolute_direction_imbalance_m": {
                    baseline_label: old["seed_absolute_direction_imbalance_m"],
                    intervention_label: new["seed_absolute_direction_imbalance_m"],
                    "intervention_minus_baseline": new[
                        "seed_absolute_direction_imbalance_m"
                    ]
                    - old["seed_absolute_direction_imbalance_m"],
                },
                "seed_weaker_side_margin_m": {
                    baseline_label: old["seed_weaker_side_margin_m"],
                    intervention_label: new["seed_weaker_side_margin_m"],
                    "intervention_minus_baseline": new["seed_weaker_side_margin_m"]
                    - old["seed_weaker_side_margin_m"],
                },
                "first_chunk_prompt_response": {
                    baseline_label: old["first_chunk_prompt_response"],
                    intervention_label: new["first_chunk_prompt_response"],
                },
                "endpoint_ordering": {
                    baseline_label: old["endpoint_ordering"],
                    intervention_label: new["endpoint_ordering"],
                },
            }
        )

    transitions = Counter(row["success"]["transition"] for row in cells)
    transitions_by_relation = {
        relation: dict(
            sorted(
                Counter(
                    row["success"]["transition"]
                    for row in cells
                    if row["requested_relation"] == relation
                ).items()
            )
        )
        for relation in shared.PROMPTS
    }
    effects = {}
    effect_paths = {
        "requested_signed_final_margin_m": lambda row: row[
            "requested_signed_final_margin_m"
        ]["intervention_minus_baseline"],
        "cube_path_length_3d_m": lambda row: row["trajectory_quality"][
            "cube_path_length_3d_m"
        ]["intervention_minus_baseline"],
        "cube_net_displacement_3d_m": lambda row: row["trajectory_quality"][
            "cube_net_displacement_3d_m"
        ]["intervention_minus_baseline"],
        "cube_excess_path_ratio": lambda row: row["trajectory_quality"][
            "cube_excess_path_ratio"
        ]["intervention_minus_baseline"],
        "cube_max_excursion_from_initial_3d_m": lambda row: row[
            "trajectory_quality"
        ]["cube_max_excursion_from_initial_3d_m"]["intervention_minus_baseline"],
        "cube_max_requested_lateral_excursion_from_initial_m": lambda row: row[
            "trajectory_quality"
        ]["cube_max_requested_lateral_excursion_from_initial_m"][
            "intervention_minus_baseline"
        ],
        "cube_max_opposite_lateral_excursion_from_initial_m": lambda row: row[
            "trajectory_quality"
        ]["cube_max_opposite_lateral_excursion_from_initial_m"][
            "intervention_minus_baseline"
        ],
        "joint_action_total_variation_l2": lambda row: row[
            "executed_action_quality"
        ]["joint_action_total_variation_l2"]["intervention_minus_baseline"],
        "joint_action_mean_l2_per_transition": lambda row: row[
            "executed_action_quality"
        ]["joint_action_mean_l2_per_transition"]["intervention_minus_baseline"],
        "gripper_switch_count": lambda row: row["executed_action_quality"][
            "gripper_switch_count"
        ]["intervention_minus_baseline"],
    }
    for name, getter in effect_paths.items():
        observed = [
            (row["cell_id"], getter(row))
            for row in cells
            if getter(row) is not None
        ]
        effects[name] = _numeric_effect(
            observed
        )
        effects[name]["comparable_cell_count"] = len(observed)
        effects[name]["noncomparable_cell_count"] = len(cells) - len(observed)

    before_summary = shared.configuration_summary(before_rows, before_pairs)
    after_summary = shared.configuration_summary(after_rows, after_pairs)
    completion_pairs = [
        {
            "cell_id": row["cell_id"],
            "intervention_minus_baseline": row["completion_actions"][
                "intervention_minus_baseline_when_both_observed"
            ],
        }
        for row in cells
        if row["completion_actions"]["intervention_minus_baseline_when_both_observed"]
        is not None
    ]
    return {
        "model": model_label,
        "baseline_label": baseline_label,
        "intervention_label": intervention_label,
        "exact_prompts": shared.PROMPTS,
        "success": {
            "baseline_total": before_summary["requested_success_count"],
            "intervention_total": after_summary["requested_success_count"],
            "net_success_change": after_summary["requested_success_count"]
            - before_summary["requested_success_count"],
            "exact_paired_transitions": dict(sorted(transitions.items())),
            "exact_paired_transitions_by_relation": transitions_by_relation,
            "completion_actions": {
                "baseline_observed_success_events": [
                    {
                        "cell_id": f"seed{row['environment_seed']}_{row['requested_relation']}",
                        "prompt": row["prompt"],
                        "actions": row["completion_actions_observed"],
                    }
                    for row in before_rows
                    if row["completion_actions_observed"] is not None
                ],
                "intervention_observed_success_events": [
                    {
                        "cell_id": f"seed{row['environment_seed']}_{row['requested_relation']}",
                        "prompt": row["prompt"],
                        "actions": row["completion_actions_observed"],
                    }
                    for row in after_rows
                    if row["completion_actions_observed"] is not None
                ],
                "both_observed_paired_deltas": completion_pairs,
                "failure_policy": (
                    f"Failures are right-censored at {shared.ACTION_CAP} actions and never entered as completion times."
                ),
            },
        },
        "baseline_configuration_summary": before_summary,
        "intervention_configuration_summary": after_summary,
        "paired_effect_sizes_intervention_minus_baseline": effects,
        "paired_seed_diagnostics": pair_comparisons,
        "cells": cells,
    }


def compile_comparison(args: argparse.Namespace) -> dict[str, Any]:
    cosmos_baseline_record = shared.validate_exact_file(
        args.cosmos_baseline,
        expected_sha256=COSMOS_BASELINE_SHA256,
        label="Cosmos3 Nano g=3 baseline",
    )
    dreamzero_baseline_record = shared.validate_exact_file(
        args.dreamzero_baseline,
        expected_sha256=DREAMZERO_BASELINE_SHA256,
        label="DreamZero s=1-equivalent baseline",
    )
    cosmos_baseline = shared.load_json(args.cosmos_baseline)
    cosmos_intervention = shared.load_json(args.cosmos_intervention)
    dreamzero_baseline = shared.load_json(args.dreamzero_baseline)
    dreamzero_intervention = shared.load_json(args.dreamzero_intervention)
    if (
        cosmos_baseline.get("model_id") != COSMOS_MODEL_ID
        or cosmos_baseline.get("amendment_id") != "V2-A011"
    ):
        raise RuntimeError("Cosmos baseline identity changed")
    if (
        cosmos_intervention.get("schema_version") != COSMOS_INTERVENTION_SCHEMA
        or cosmos_intervention.get("status") != "complete"
        or cosmos_intervention.get("amendment_id") != shared.AMENDMENT_ID
        or cosmos_intervention.get("model_id") != COSMOS_MODEL_ID
    ):
        raise RuntimeError("Cosmos intervention result is not a complete V2-A015 arm")
    if (
        dreamzero_baseline.get("model_id") != "dreamzero_droid"
        or dreamzero_baseline.get("amendment_id") != "V2-A007"
    ):
        raise RuntimeError("DreamZero baseline identity changed")
    if (
        dreamzero_intervention.get("schema_version")
        != DREAMZERO_INTERVENTION_SCHEMA
        or dreamzero_intervention.get("status") != "complete"
        or dreamzero_intervention.get("amendment_id") != shared.AMENDMENT_ID
        or dreamzero_intervention.get("model_id") != DREAMZERO_MODEL_ID
    ):
        raise RuntimeError("DreamZero intervention result is not a complete V2-A015 arm")

    cosmos = _compare_configuration(
        model_label="Cosmos3 Nano Policy DROID",
        baseline_label="g=3 baseline",
        intervention_label="g=1 intervention",
        first_chunk_horizon=32,
        baseline_result=cosmos_baseline,
        intervention_result=cosmos_intervention,
        baseline_episode_model=COSMOS_MODEL_ID,
    )
    dreamzero = _compare_configuration(
        model_label="DreamZero DROID",
        baseline_label="s=1 conditional-action equivalent",
        intervention_label="s=2 CFG-style negative-branch action guidance",
        first_chunk_horizon=8,
        baseline_result=dreamzero_baseline,
        intervention_result=dreamzero_intervention,
        baseline_episode_model="dreamzero_droid",
    )
    return {
        "schema_version": SCHEMA,
        "status": "complete",
        "compiled_at_git_head": args.compiled_at_git_head,
        "amendment_id": shared.AMENDMENT_ID,
        "arena": "droid_robolab",
        "exact_prompts": shared.PROMPTS,
        "metric_definitions": shared.metric_definitions(),
        "comparisons": {
            "cosmos3_nano": cosmos,
            "dreamzero": dreamzero,
        },
        "provenance": {
            "cosmos3_nano_baseline": cosmos_baseline_record,
            "cosmos3_nano_intervention": shared.file_record(
                args.cosmos_intervention
            ),
            "dreamzero_baseline": dreamzero_baseline_record,
            "dreamzero_intervention": shared.file_record(
                args.dreamzero_intervention
            ),
        },
        "inference_boundary": (
            "Each comparison is an exact paired, descriptive n=6 post-result pilot. "
            "Improved/regressed/unchanged cell transitions and paired effect sizes are reported without a powered or general performance-gain claim. "
            "Cosmos3 Nano and DreamZero denominators remain separate, and neither is pooled with RoboTwin."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-baseline", type=Path, required=True)
    parser.add_argument("--cosmos-intervention", type=Path, required=True)
    parser.add_argument("--dreamzero-baseline", type=Path, required=True)
    parser.add_argument("--dreamzero-intervention", type=Path, required=True)
    parser.add_argument("--result-output", type=Path, required=True)
    parser.add_argument("--compiled-at-git-head", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in (
        "cosmos_baseline",
        "cosmos_intervention",
        "dreamzero_baseline",
        "dreamzero_intervention",
        "result_output",
    ):
        setattr(args, key, getattr(args, key).resolve())
    result = compile_comparison(args)
    shared.dump_json(args.result_output, result, overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": "complete",
                "cosmos_success": result["comparisons"]["cosmos3_nano"]["success"],
                "dreamzero_success": result["comparisons"]["dreamzero"]["success"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
