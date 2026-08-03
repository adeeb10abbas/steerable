#!/usr/bin/env python3
"""Compile the complete prospective VLA-versus-WAM study evidence.

This is the second-stage compiler. ``compile_vla_wam_evidence.py`` owns the
closed-loop episode extraction; this script verifies and joins that output with
the frozen Cosmos semantic-future scorer, the fixed-observation command probe,
operational evidence, and the explicitly retrospective WAM evidence tier.

The compiler fails closed unless every registered episode and every frozen
secondary probe is present. Generated plots are descriptive: the study has one
checkpoint per model class, and future chunks within an episode are correlated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


EXPECTED_EPISODES = 160
EXPECTED_CONFIRMATORY_EPISODES = 80
EXPECTED_DIRECT_STRESS_EPISODES = 80
EXPECTED_PROBE_CONDITIONS = 16
EXPECTED_DIRECT_TASK_PROBE_CONDITIONS = 11
EXPECTED_THERMAL_LOGS = (
    "cosmos_canonical",
    "cosmos_vague",
    "cosmos_declarative",
    "cosmos_contrastive",
    "pi05_canonical",
    "pi05_vague",
    "pi05_declarative",
    "pi05_contrastive",
)
WORDINGS = (
    "canonical",
    "short_paraphrase",
    "declarative_goal",
    "contrastive_goal",
)
WORDING_LABELS = {
    "canonical": "canonical",
    "short_paraphrase": "short",
    "declarative_goal": "declarative",
    "contrastive_goal": "contrastive",
}
TASK_DIRS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
MODEL_LABELS = {
    "pi05_droid_vla": "π0.5 DROID (VLA)",
    "cosmos3_edge_droid_wam": "Cosmos3 Edge DROID (WAM)",
}
MODEL_COLORS = {
    "pi05_droid_vla": "#2864dc",
    "cosmos3_edge_droid_wam": "#e36a2e",
}
QUADRANT_COLORS = {
    "imagines_requested_executes_requested": "#2a9d6f",
    "imagines_requested_executes_not_requested": "#e9c46a",
    "does_not_imagine_requested_executes_requested": "#4f86c6",
    "neither_imagines_nor_executes_requested": "#777777",
    "uncertain_future": "#d7d7d7",
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _trajectory_evidence_summary(
    root: Path, workspace: Path, closed: dict[str, Any]
) -> dict[str, Any]:
    evidence_root = root / "trajectory_evidence"
    summary_path = evidence_root / "summary.json"
    index_path = evidence_root / "trajectory_index.json"
    summary = _load(summary_path)
    rows = _load(index_path)
    if summary.get("status") != "complete":
        raise RuntimeError(f"Trajectory evidence is not complete: {summary.get('status')}")
    if (
        summary.get("expected_episode_count") != EXPECTED_EPISODES
        or summary.get("rendered_episode_count") != EXPECTED_EPISODES
        or summary.get("missing_episode_count") != 0
        or len(rows) != EXPECTED_EPISODES
    ):
        raise RuntimeError("Trajectory evidence does not cover all registered episodes")
    plan_path = root / "trajectory_visualization_plan.json"
    if summary.get("selection_plan_sha256") != _sha256(plan_path):
        raise RuntimeError("Trajectory evidence selection-plan hash mismatch")

    closed_index = {
        (row["condition_id"], row["direction"], int(row["episode_seed"])): row
        for row in closed["episodes"]
    }
    trajectory_index = {
        (row["condition_id"], row["direction"], int(row["episode_seed"])): row
        for row in rows
    }
    if len(closed_index) != EXPECTED_EPISODES or set(closed_index) != set(trajectory_index):
        raise RuntimeError("Trajectory index keys disagree with closed-loop evidence")
    for key, trajectory in trajectory_index.items():
        episode = closed_index[key]
        if bool(trajectory["binary_success"]) != bool(episode["binary_success"]):
            raise RuntimeError(f"Trajectory success label disagrees with closed-loop evidence: {key}")
        if bool(trajectory["final_requested_relation"]) != bool(
            episode["final_requested_relation"]
        ):
            raise RuntimeError(f"Trajectory endpoint relation disagrees with closed-loop evidence: {key}")
        if not np.isclose(
            float(trajectory["requested_signed_final_offset_m"]),
            float(episode["requested_signed_final_offset_m"]),
            atol=1e-8,
            rtol=0.0,
        ):
            raise RuntimeError(f"Trajectory endpoint offset disagrees with closed-loop evidence: {key}")
        figure = workspace / trajectory["figure_path"]
        if not figure.exists():
            raise FileNotFoundError(figure)

    landscape = evidence_root / "social/first_seed_stress_landscape_1600x900.png"
    square = evidence_root / "social/first_seed_stress_square_1200x1200.png"
    scorecard_landscape = evidence_root / "social/steerability_scorecard_1600x900.png"
    scorecard_square = evidence_root / "social/steerability_scorecard_1200x1200.png"
    failure_square = evidence_root / "social/failure_progress_anatomy_1200x1200.png"
    atlas = evidence_root / "blog/all_executed_paths_and_endpoints.png"
    failure_anatomy = evidence_root / "blog/failure_progress_anatomy.png"
    gallery = evidence_root / "gallery/index.html"
    for path in (
        landscape,
        square,
        scorecard_landscape,
        scorecard_square,
        failure_square,
        atlas,
        failure_anatomy,
        gallery,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if cv2.imread(str(landscape)).shape[:2] != (900, 1600):
        raise RuntimeError("Landscape social trajectory export is not 1600x900")
    if cv2.imread(str(square)).shape[:2] != (1200, 1200):
        raise RuntimeError("Square social trajectory export is not 1200x1200")
    if cv2.imread(str(scorecard_landscape)).shape[:2] != (900, 1600):
        raise RuntimeError("Landscape social scorecard export is not 1600x900")
    if cv2.imread(str(scorecard_square)).shape[:2] != (1200, 1200):
        raise RuntimeError("Square social scorecard export is not 1200x1200")
    if cv2.imread(str(failure_square)).shape[:2] != (1200, 1200):
        raise RuntimeError("Square failure-anatomy export is not 1200x1200")

    return {
        "status": summary["status"],
        "rendered_episode_count": summary["rendered_episode_count"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "group_summaries": summary["group_summaries"],
        "coordinate_convention": summary["coordinate_convention"],
        "scored_goal": summary["scored_goal"],
        "illustrative_route_disclaimer": summary["illustrative_route_disclaimer"],
        "all_episode_policy": summary["all_episode_policy"],
        "selection_plan_sha256": summary["selection_plan_sha256"],
        "summary_sha256": _sha256(summary_path),
        "index_sha256": _sha256(index_path),
        "index_csv_sha256": _sha256(evidence_root / "trajectory_index.csv"),
        "artifacts": {
            "atlas": _relative(atlas, workspace),
            "failure_anatomy": _relative(failure_anatomy, workspace),
            "gallery": _relative(gallery, workspace),
            "landscape_social": _relative(landscape, workspace),
            "square_social": _relative(square, workspace),
            "scorecard_landscape_social": _relative(scorecard_landscape, workspace),
            "scorecard_square_social": _relative(scorecard_square, workspace),
            "failure_anatomy_square_social": _relative(failure_square, workspace),
        },
    }


def _semantic_visualization_summary(
    root: Path, workspace: Path, expected_chunks: int
) -> dict[str, Any]:
    evidence_root = root / "semantic_future_visualization"
    summary_path = evidence_root / "summary.json"
    selection_path = evidence_root / "selection.json"
    summary = _load(summary_path)
    selection = _load(selection_path)
    plan_path = root / "semantic_future_visualization_plan.json"
    plan_sha = _sha256(plan_path)
    if summary.get("status") != "complete":
        raise RuntimeError("Semantic-future example visualization is not complete")
    if summary.get("selection_plan_sha256") != plan_sha:
        raise RuntimeError("Semantic-future visualization plan hash mismatch")
    if selection.get("selection_plan_sha256") != plan_sha:
        raise RuntimeError("Semantic-future selection record plan hash mismatch")
    if int(summary.get("eligible_chunk_count", -1)) != expected_chunks:
        raise RuntimeError("Semantic-future example population disagrees with scored chunks")
    if int(selection.get("eligible_chunk_count", -1)) != expected_chunks:
        raise RuntimeError("Semantic-future selection population disagrees with scored chunks")
    if summary.get("selection_sha256") != _sha256(selection_path):
        raise RuntimeError("Semantic-future selection hash mismatch")
    expected_categories = {
        "imagines_requested_executes_requested",
        "imagines_requested_executes_not_requested",
        "does_not_imagine_requested_executes_requested",
        "neither_imagines_nor_executes_requested",
        "uncertain_future",
    }
    if set(selection.get("categories", {})) != expected_categories:
        raise RuntimeError("Semantic-future example category set is incomplete")
    observed = sum(value is not None for value in selection["categories"].values())
    if int(summary.get("observed_category_count", -1)) != observed:
        raise RuntimeError("Semantic-future observed-category count mismatch")
    blog = evidence_root / "blog/selected_semantic_future_examples.png"
    landscape = evidence_root / "social/wam_semantic_quadrants_1600x900.png"
    square = evidence_root / "social/wam_semantic_quadrants_1200x1200.png"
    for path in (blog, landscape, square):
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    landscape_image = cv2.imread(str(landscape))
    square_image = cv2.imread(str(square))
    if landscape_image is None or landscape_image.shape[:2] != (900, 1600):
        raise RuntimeError("Semantic landscape social export is not 1600x900")
    if square_image is None or square_image.shape[:2] != (1200, 1200):
        raise RuntimeError("Semantic square social export is not 1200x1200")
    return {
        "status": summary["status"],
        "eligible_chunk_count": summary["eligible_chunk_count"],
        "observed_category_count": observed,
        "selection_plan_sha256": plan_sha,
        "selection_sha256": _sha256(selection_path),
        "summary_sha256": _sha256(summary_path),
        "categories": selection["categories"],
        "artifacts": {
            "blog": _relative(blog, workspace),
            "landscape_social": _relative(landscape, workspace),
            "square_social": _relative(square, workspace),
        },
    }


def _repo_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    status = run("status", "--short").splitlines()
    return {
        "path": str(path.resolve()),
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_path_count": len(status),
    }


def _thermal_log_summary(root: Path) -> dict[str, Any]:
    """Validate the frozen safety guard lifecycle for every definitive batch."""

    summaries = []
    for batch in EXPECTED_THERMAL_LOGS:
        path = root / "thermal_logs" / f"{batch}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise RuntimeError(f"Empty thermal guard log: {path}")
        events = [row["event"] for row in rows]
        if events[0] != "monitor_started" or events[-1] != "monitor_completed":
            raise RuntimeError(f"Incomplete thermal guard lifecycle in {path}: {events}")
        emergency = [event for event in events if event.startswith("emergency_stop")]
        if emergency:
            raise RuntimeError(f"Emergency thermal stop in definitive batch {path}: {emergency}")
        cooldown_started = sum(event == "cooldown_started" for event in events)
        completed_rows = [row for row in rows if row["event"] == "cooldown_completed"]
        if cooldown_started != len(completed_rows):
            raise RuntimeError(
                f"Unpaired thermal cooldown in {path}: starts={cooldown_started} "
                f"completions={len(completed_rows)}"
            )
        temperatures = []
        for row in rows:
            for field in ("temperature_c", "peak_temperature_c"):
                if field in row:
                    temperatures.append(int(row[field]))
        summaries.append(
            {
                "batch": batch,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "events": len(rows),
                "cooldowns": len(completed_rows),
                "cooldown_seconds": float(
                    sum(float(row["cooldown_seconds"]) for row in completed_rows)
                ),
                "max_logged_temperature_c": max(temperatures) if temperatures else None,
            }
        )
    return {
        "status": "all_definitive_guard_lifecycles_complete_without_emergency_stop",
        "batches": len(summaries),
        "total_cooldowns": sum(row["cooldowns"] for row in summaries),
        "total_cooldown_seconds": float(sum(row["cooldown_seconds"] for row in summaries)),
        "max_logged_temperature_c": max(
            (
                row["max_logged_temperature_c"]
                for row in summaries
                if row["max_logged_temperature_c"] is not None
            ),
            default=None,
        ),
        "summaries": summaries,
    }


def _paired_exact(
    rows: list[dict[str, Any]],
    left_filter: dict[str, Any],
    right_filter: dict[str, Any],
    pair_keys: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    def subset(criteria: dict[str, Any]) -> dict[tuple, dict[str, Any]]:
        return {
            tuple(row[key] for key in pair_keys): row
            for row in rows
            if all(row.get(key) == value for key, value in criteria.items())
        }

    left = subset(left_filter)
    right = subset(right_filter)
    keys = sorted(set(left) & set(right))
    pairs = [(bool(left[key]["binary_success"]), bool(right[key]["binary_success"])) for key in keys]
    left_only = sum(a and not b for a, b in pairs)
    right_only = sum(b and not a for a, b in pairs)
    discordant = left_only + right_only
    p_value = (
        float(binomtest(min(left_only, right_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "label": label,
        "pair_keys": list(pair_keys),
        "pairs": len(pairs),
        "left_filter": left_filter,
        "right_filter": right_filter,
        "both_success": sum(a and b for a, b in pairs),
        "both_failure": sum(not a and not b for a, b in pairs),
        "left_success_right_failure": left_only,
        "left_failure_right_success": right_only,
        "discordant_pairs": discordant,
        "two_sided_exact_mcnemar_p": p_value,
        "interpretation_guardrail": "Exploratory paired diagnostic; no multiplicity correction.",
    }


def _paired_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for model_id, horizon in (
        ("pi05_droid_vla", 15),
        ("cosmos3_edge_droid_wam", 32),
    ):
        results.append(
            _paired_exact(
                rows,
                {
                    "model_id": model_id,
                    "wording": "canonical",
                    "controller": "static",
                    "open_loop_horizon": horizon,
                },
                {
                    "model_id": model_id,
                    "wording": "short_paraphrase",
                    "controller": "static",
                    "open_loop_horizon": horizon,
                },
                ("direction", "episode_seed"),
                f"{model_id}: canonical versus short paraphrase",
            )
        )
        results.append(
            _paired_exact(
                rows,
                {
                    "model_id": model_id,
                    "wording": "declarative_goal",
                    "controller": "static",
                    "open_loop_horizon": horizon,
                },
                {
                    "model_id": model_id,
                    "wording": "contrastive_goal",
                    "controller": "static",
                    "open_loop_horizon": horizon,
                },
                ("direction", "episode_seed"),
                f"{model_id}: declarative versus contrastive direct task language",
            )
        )
        for wording in WORDINGS:
            results.append(
                _paired_exact(
                    rows,
                    {
                        "model_id": model_id,
                        "wording": wording,
                        "controller": "static",
                        "open_loop_horizon": horizon,
                        "direction": "left",
                    },
                    {
                        "model_id": model_id,
                        "wording": wording,
                        "controller": "static",
                        "open_loop_horizon": horizon,
                        "direction": "right",
                    },
                    ("episode_seed",),
                    f"{model_id}: {wording} left versus right directional asymmetry",
                )
            )
    for wording in WORDINGS:
        for direction in ("left", "right"):
            results.append(
                _paired_exact(
                    rows,
                    {
                        "model_id": "pi05_droid_vla",
                        "wording": wording,
                        "controller": "static",
                        "open_loop_horizon": 15,
                        "direction": direction,
                    },
                    {
                        "model_id": "cosmos3_edge_droid_wam",
                        "wording": wording,
                        "controller": "static",
                        "open_loop_horizon": 32,
                        "direction": direction,
                    },
                    ("episode_seed",),
                    f"{wording} {direction}: pi05 VLA versus Cosmos WAM at native horizons",
                )
            )
    return results


def _load_probe(
    path: Path,
    expected_model: str,
    plan_sha: str,
    expected_conditions: int = EXPECTED_PROBE_CONDITIONS,
) -> dict[str, Any]:
    value = _load(path)
    if value["model"] != expected_model:
        raise RuntimeError(f"Probe model mismatch in {path}: {value['model']}")
    if value["plan_sha256"] != plan_sha:
        raise RuntimeError(f"Probe plan hash mismatch in {path}")
    if len(value["records"]) != expected_conditions:
        raise RuntimeError(f"Expected {expected_conditions} probe conditions in {path}")
    seeds = {row["server_sampling_seed"] for row in value["records"]}
    if seeds != {value["sampling_seed"]}:
        raise RuntimeError(f"Server did not honor the frozen probe seed in {path}: {seeds}")
    return value


def _pairwise_probe_action(probe_dir: Path) -> list[dict[str, Any]]:
    pairs = [
        ("task", "task_left", "task_right"),
        ("atomic motion", "atomic_left", "atomic_right"),
        ("grounded point", "point_left_target", "point_right_target"),
        ("combination", "combination_left", "combination_right"),
    ]
    rows = []
    for style, left_name, right_name in pairs:
        left = np.load(probe_dir / f"{left_name}_action.npy").astype(np.float64)
        right = np.load(probe_dir / f"{right_name}_action.npy").astype(np.float64)
        if left.shape != right.shape:
            raise RuntimeError(f"Probe pair shape mismatch: {left_name}, {right_name}")
        rows.append(
            {
                "style": style,
                "left_condition": left_name,
                "right_condition": right_name,
                "action_rms": float(np.sqrt(np.mean(np.square(left - right)))),
            }
        )
    return rows


def _probe_summary(
    root: Path, plan_sha: str, future_semantics: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model, directory in (
        ("pi05", root / "command_probe/pi05"),
        ("cosmos", root / "command_probe/cosmos_gpu1"),
    ):
        manifest = _load_probe(directory / "manifest.json", model, plan_sha)
        request_times = [row.get("request_wall_seconds") for row in manifest["records"]]
        if any(value is None or float(value) <= 0 for value in request_times):
            raise RuntimeError(f"Missing fixed-probe endpoint timing in {directory}")
        timing = np.asarray(request_times, dtype=np.float64)
        result[model] = {
            "manifest": manifest,
            "left_right_action_pairs": _pairwise_probe_action(directory),
            "endpoint_timing": {
                "requests": int(len(timing)),
                "median_request_wall_seconds": float(np.median(timing)),
                "mean_request_wall_seconds": float(np.mean(timing)),
                "min_request_wall_seconds": float(np.min(timing)),
                "max_request_wall_seconds": float(np.max(timing)),
                "interface_note": manifest["timing_note"],
            },
        }
    if future_semantics["conditions"] != EXPECTED_PROBE_CONDITIONS:
        raise RuntimeError("Incomplete Cosmos command-probe semantic scoring")
    result["cosmos"]["semantic_futures"] = future_semantics
    return result


def _direct_task_probe_pairs(probe_dir: Path) -> list[dict[str, Any]]:
    pairs = (
        ("canonical", "task_left", "task_right"),
        ("short", "short_left", "short_right"),
        ("declarative", "declarative_left", "declarative_right"),
        ("contrastive target first", "contrastive_first_left", "contrastive_first_right"),
        ("contrastive target last", "contrastive_last_left", "contrastive_last_right"),
    )
    rows = []
    for family, left_name, right_name in pairs:
        left = np.load(probe_dir / f"{left_name}_action.npy").astype(np.float64)
        right = np.load(probe_dir / f"{right_name}_action.npy").astype(np.float64)
        if left.shape != right.shape:
            raise RuntimeError(f"Direct-probe pair shape mismatch: {left_name}, {right_name}")
        rows.append(
            {
                "prompt_family": family,
                "left_condition": left_name,
                "right_condition": right_name,
                "action_rms": float(np.sqrt(np.mean(np.square(left - right)))),
            }
        )
    for direction in ("left", "right"):
        target_first = np.load(probe_dir / f"contrastive_first_{direction}_action.npy").astype(
            np.float64
        )
        target_last = np.load(probe_dir / f"contrastive_last_{direction}_action.npy").astype(
            np.float64
        )
        rows.append(
            {
                "prompt_family": f"contrastive order {direction}",
                "left_condition": f"contrastive_first_{direction}",
                "right_condition": f"contrastive_last_{direction}",
                "action_rms": float(
                    np.sqrt(np.mean(np.square(target_first - target_last)))
                ),
            }
        )
    return rows


def _direct_task_probe_summary(
    root: Path, plan_sha: str, future_semantics: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model, directory in (
        ("pi05", root / "command_probe/direct_task_pi05"),
        ("cosmos", root / "command_probe/direct_task_cosmos"),
    ):
        manifest = _load_probe(
            directory / "manifest.json",
            model,
            plan_sha,
            EXPECTED_DIRECT_TASK_PROBE_CONDITIONS,
        )
        if manifest["exact_repeat_action_rms"] != 0.0:
            raise RuntimeError(f"Direct probe is not exactly repeatable in {directory}")
        result[model] = {
            "manifest": manifest,
            "paired_action_rms": _direct_task_probe_pairs(directory),
        }
    if future_semantics["conditions"] != EXPECTED_DIRECT_TASK_PROBE_CONDITIONS:
        raise RuntimeError("Incomplete Cosmos direct-task probe semantic scoring")
    result["cosmos"]["semantic_futures"] = future_semantics
    return result


def _chunk_imagined_at_threshold(
    cache: dict[str, Any], requested: str, threshold_m: float
) -> bool | None:
    relations = []
    for frame in cache["frames"]:
        semantics = frame["semantics"]
        by_camera = semantics.get("relations_by_camera", {})
        disagreements = semantics.get("cross_camera_disagreement_m", {})
        camera_relations = [by_camera.get(name) for name in ("left_camera", "right_camera")]
        reliable = (
            None not in camera_relations
            and camera_relations[0] == camera_relations[1]
            and disagreements.get("cube", math.inf) <= threshold_m
            and disagreements.get("bowl", math.inf) <= threshold_m
        )
        if reliable:
            relations.append(camera_relations[0])
    if len(relations) < 2:
        return None
    fraction = sum(relation == requested for relation in relations) / len(relations)
    if fraction >= 0.75:
        return True
    if fraction <= 0.25:
        return False
    return None


def _semantic_threshold_sensitivity(
    summaries: list[tuple[str, Path, dict[str, Any]]]
) -> list[dict[str, Any]]:
    output = []
    for threshold_m in (0.10, 0.15, 0.20):
        grouped: dict[tuple[str, str], list[tuple[bool | None, bool]]] = defaultdict(list)
        for wording, directory, summary in summaries:
            for row in summary["rows"]:
                task_dir = Path(row["task_dir"])
                task_key = f"{task_dir.parent.name}__{task_dir.name}"
                cache_path = (
                    directory
                    / "localization_cache"
                    / task_key
                    / f"episode_{row['episode_index']:03d}_chunk_{row['replan_index']:03d}.json"
                )
                cache = _load(cache_path)
                imagined = _chunk_imagined_at_threshold(
                    cache, row["requested_relation"], threshold_m
                )
                if math.isclose(threshold_m, 0.20) and imagined != row["imagined_requested"]:
                    raise RuntimeError(f"Threshold replay disagrees with frozen label: {cache_path}")
                grouped[(wording, row["requested_relation"])].append(
                    (imagined, bool(row["executed_requested"]))
                )
        for (wording, direction), pairs in sorted(grouped.items()):
            certain = [pair for pair in pairs if pair[0] is not None]
            agreement_count = sum(pair[0] == pair[1] for pair in certain)
            imagined_requested_count = sum(bool(pair[0]) for pair in certain)
            executed_requested_count = sum(bool(pair[1]) for pair in certain)
            executed_requested_all = sum(bool(pair[1]) for pair in pairs)
            output.append(
                {
                    "cross_camera_threshold_m": threshold_m,
                    "wording": wording,
                    "direction": direction,
                    "chunks": len(pairs),
                    "certain_chunks": len(certain),
                    "coverage_fraction": len(certain) / len(pairs),
                    "imagined_requested_count_among_certain": imagined_requested_count,
                    "imagined_requested_rate_among_certain": (
                        imagined_requested_count / len(certain)
                        if certain
                        else None
                    ),
                    "executed_requested_count_among_certain": executed_requested_count,
                    "executed_requested_count_all": executed_requested_all,
                    "executed_positive_future_coverage_fraction": (
                        executed_requested_count / executed_requested_all
                        if executed_requested_all
                        else None
                    ),
                    "imagination_execution_agreement_count_among_certain": agreement_count,
                    "imagination_execution_agreement_among_certain": (
                        agreement_count / len(certain)
                        if certain
                        else None
                    ),
                }
            )
    return output


def _semantic_aggregate(summaries: list[tuple[str, Path, dict[str, Any]]]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    episode_groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    all_rows = []
    input_summaries = []
    for wording, directory, summary in summaries:
        input_summaries.append(
            {
                "wording": wording,
                "num_chunks": summary["num_chunks"],
                "coverage_fraction": summary["coverage_fraction"],
                "calibration_sha256": summary["calibration_sha256"],
            }
        )
        for row in summary["rows"]:
            enriched = {**row, "wording": wording}
            groups[(wording, row["requested_relation"])].append(enriched)
            episode_groups[
                (
                    wording,
                    row["requested_relation"],
                    row["task_dir"],
                    int(row["episode_index"]),
                )
            ].append(enriched)
            all_rows.append(enriched)
    grouped = []
    for (wording, direction), rows in sorted(groups.items()):
        certain = [row for row in rows if row["imagined_requested"] is not None]
        counts = Counter(row["quadrant"] for row in rows)
        grouped.append(
            {
                "wording": wording,
                "direction": direction,
                "chunks": len(rows),
                "episodes": len({(row["task_dir"], row["episode_index"]) for row in rows}),
                "certain_chunks": len(certain),
                "coverage_fraction": len(certain) / len(rows),
                "imagined_requested_rate_among_certain": (
                    sum(bool(row["imagined_requested"]) for row in certain) / len(certain)
                    if certain
                    else None
                ),
                "executed_requested_rate_among_certain": (
                    sum(bool(row["executed_requested"]) for row in certain) / len(certain)
                    if certain
                    else None
                ),
                "imagination_execution_agreement_among_certain": (
                    sum(row["imagined_requested"] == row["executed_requested"] for row in certain)
                    / len(certain)
                    if certain
                    else None
                ),
                "quadrant_counts": dict(counts),
            }
        )
    episode_summaries = []
    for (wording, direction, task_dir, episode_index), rows in sorted(
        episode_groups.items()
    ):
        ordered = sorted(rows, key=lambda row: int(row["replan_index"]))
        certain = [row for row in ordered if row["imagined_requested"] is not None]
        last = ordered[-1]
        episode_summaries.append(
            {
                "wording": wording,
                "direction": direction,
                "task_dir": task_dir,
                "episode_index": episode_index,
                "chunks": len(ordered),
                "certain_chunks": len(certain),
                "coverage_fraction": len(certain) / len(ordered),
                "any_imagined_requested": (
                    any(bool(row["imagined_requested"]) for row in certain)
                    if certain
                    else None
                ),
                "any_executed_requested": any(
                    bool(row["executed_requested"]) for row in ordered
                ),
                "terminal_imagined_requested": last["imagined_requested"],
                "terminal_executed_requested": bool(last["executed_requested"]),
                "terminal_quadrant": last["quadrant"],
            }
        )
    certain_rows = [row for row in all_rows if row["imagined_requested"] is not None]
    quadrant_counts = Counter(row["quadrant"] for row in all_rows)
    true_positive = quadrant_counts["imagines_requested_executes_requested"]
    false_positive = quadrant_counts[
        "imagines_requested_executes_not_requested"
    ]
    false_negative = quadrant_counts[
        "does_not_imagine_requested_executes_requested"
    ]
    true_negative = quadrant_counts["neither_imagines_nor_executes_requested"]
    executed_positive_all = sum(bool(row["executed_requested"]) for row in all_rows)
    executed_positive_certain = true_positive + false_negative
    overall = {
        "chunks": len(all_rows),
        "certain_chunks": len(certain_rows),
        "uncertain_chunks": len(all_rows) - len(certain_rows),
        "coverage_fraction": len(certain_rows) / len(all_rows),
        "quadrant_counts": dict(quadrant_counts),
        "imagination_execution_agreement_among_certain": (
            (true_positive + true_negative) / len(certain_rows)
            if certain_rows
            else None
        ),
        "imagined_positive_precision_for_execution_among_certain": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        "imagined_positive_recall_of_executed_positive_among_certain": (
            true_positive / executed_positive_certain
            if executed_positive_certain
            else None
        ),
        "executed_positive_chunks_all": executed_positive_all,
        "executed_positive_chunks_with_certain_future": executed_positive_certain,
        "executed_positive_chunks_with_uncertain_future": (
            executed_positive_all - executed_positive_certain
        ),
        "executed_positive_future_coverage_fraction": (
            executed_positive_certain / executed_positive_all
            if executed_positive_all
            else None
        ),
        "episodes": len(episode_summaries),
        "episodes_without_any_certain_chunk": sum(
            row["certain_chunks"] == 0 for row in episode_summaries
        ),
        "episodes_with_any_imagined_requested": sum(
            row["any_imagined_requested"] is True for row in episode_summaries
        ),
        "episodes_with_any_executed_requested": sum(
            row["any_executed_requested"] for row in episode_summaries
        ),
    }
    threshold_sensitivity = _semantic_threshold_sensitivity(summaries)
    threshold_sensitivity_overall = []
    for threshold_m in (0.10, 0.15, 0.20):
        threshold_rows = [
            row
            for row in threshold_sensitivity
            if math.isclose(row["cross_camera_threshold_m"], threshold_m)
        ]
        chunks = sum(row["chunks"] for row in threshold_rows)
        certain_chunks = sum(row["certain_chunks"] for row in threshold_rows)
        agreement_count = sum(
            row["imagination_execution_agreement_count_among_certain"]
            for row in threshold_rows
        )
        executed_requested_all = sum(
            row["executed_requested_count_all"] for row in threshold_rows
        )
        executed_requested_certain = sum(
            row["executed_requested_count_among_certain"] for row in threshold_rows
        )
        threshold_sensitivity_overall.append(
            {
                "cross_camera_threshold_m": threshold_m,
                "chunks": chunks,
                "certain_chunks": certain_chunks,
                "coverage_fraction": certain_chunks / chunks,
                "imagination_execution_agreement_among_certain": (
                    agreement_count / certain_chunks if certain_chunks else None
                ),
                "executed_requested_count_all": executed_requested_all,
                "executed_requested_count_among_certain": executed_requested_certain,
                "executed_positive_future_coverage_fraction": (
                    executed_requested_certain / executed_requested_all
                    if executed_requested_all
                    else None
                ),
            }
        )
    return {
        "input_summaries": input_summaries,
        "total_chunks": len(all_rows),
        "total_episodes": len({(row["task_dir"], row["episode_index"]) for row in all_rows}),
        "guardrail": "Chunk rates are descriptive and receive no binomial interval because replans within an episode are correlated.",
        "threshold_sensitivity": threshold_sensitivity,
        "threshold_sensitivity_overall": threshold_sensitivity_overall,
        "overall": overall,
        "groups": grouped,
        "episode_summaries": episode_summaries,
        "rows": all_rows,
    }


def _image_difference(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if left.shape != right.shape:
        raise RuntimeError(f"Conditioning-image shape mismatch: {left.shape} versus {right.shape}")
    difference = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return {
        "mae_0_255": float(np.mean(difference)),
        "nonidentical_channel_fraction": float(np.mean(difference > 0)),
        "p99_abs_difference": float(np.percentile(difference, 99)),
        "max_abs_difference": int(np.max(difference)),
    }


def _cosmos_observation_variation(run_manifest: dict[str, Any]) -> dict[str, Any]:
    images: dict[tuple[str, str, int], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    prediction_chunks_validated = 0
    conditions = [
        condition
        for condition in run_manifest["conditions"]
        if condition["model_id"] == "cosmos3_edge_droid_wam"
        and int(condition["open_loop_horizon"]) == 32
    ]
    if len(conditions) != 4:
        raise RuntimeError(f"Expected four direct-language Cosmos conditions, got {len(conditions)}")
    for condition in conditions:
        root = Path(condition["output_root"])
        episode_seeds = [int(seed) for seed in condition["episode_seeds"]]
        for direction, task_name in TASK_DIRS.items():
            task_dir = root / task_name
            manifest_path = task_dir / "predicted_chunks/manifest.jsonl"
            manifest_rows = [
                json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
            ]
            seen_chunks = set()
            for row in manifest_rows:
                episode_index = int(row["episode_index"])
                replan_index = int(row["replan_index"])
                chunk_key = (episode_index, replan_index)
                if chunk_key in seen_chunks:
                    raise RuntimeError(f"Duplicate Cosmos chunk {chunk_key} in {manifest_path}")
                seen_chunks.add(chunk_key)
                if episode_index >= len(episode_seeds):
                    raise RuntimeError(f"Unexpected episode index in {manifest_path}: {row}")
                expected_seed = episode_seeds[episode_index] * 1000 + replan_index
                if row["requested_sampling_seed"] != expected_seed:
                    raise RuntimeError(f"Requested seed mismatch in {manifest_path}: {row}")
                if row["server_sampling_seed"] != expected_seed:
                    raise RuntimeError(f"Server seed mismatch in {manifest_path}: {row}")
                if int(row["executed_step_start"]) != replan_index * 32:
                    raise RuntimeError(f"Executed-step alignment mismatch in {manifest_path}: {row}")
                if row["conditioning_shape"] != [540, 640, 3]:
                    raise RuntimeError(f"Conditioning shape mismatch in {manifest_path}: {row}")
                if row["action_shape"] != [32, 8] or row["future_shape"] != [33, 528, 640, 3]:
                    raise RuntimeError(f"Cosmos output shape mismatch in {manifest_path}: {row}")
                prediction_chunks_validated += 1
            first_rows = {
                int(row["episode_index"]): row
                for row in manifest_rows
                if int(row["replan_index"]) == 0
            }
            if set(first_rows) != set(range(len(episode_seeds))):
                raise RuntimeError(f"Missing Cosmos first-chunk records in {manifest_path}")
            for episode_index in range(len(episode_seeds)):
                row = first_rows[episode_index]
                image_path = (
                    task_dir
                    / "predicted_chunks"
                    / f"episode_{episode_index:03d}"
                    / "chunk_000"
                    / "conditioning.png"
                )
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    raise FileNotFoundError(image_path)
                # RoboLab records the SHA-256 of the contiguous RGB pixel
                # array sent to the policy, not the PNG container bytes.
                # Re-encoding the same lossless pixels changes the file hash,
                # so verify the declared interface at its actual byte level.
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pixel_sha256 = hashlib.sha256(
                    np.ascontiguousarray(rgb).tobytes()
                ).hexdigest()
                if pixel_sha256 != row["conditioning_sha256"]:
                    raise RuntimeError(f"Conditioning pixel hash mismatch: {image_path}")
                images[(condition["id"], direction, episode_index)] = bgr

    for condition in conditions:
        episode_count = len(condition["episode_seeds"])
        for direction in TASK_DIRS:
            for left_index in range(episode_count):
                for right_index in range(left_index + 1, episode_count):
                    records.append(
                        {
                            "comparison": "within_condition_direction",
                            "condition_id": condition["id"],
                            "direction": direction,
                            "left_episode_index": left_index,
                            "right_episode_index": right_index,
                            **_image_difference(
                                images[(condition["id"], direction, left_index)],
                                images[(condition["id"], direction, right_index)],
                            ),
                        }
                    )
        for episode_index in range(episode_count):
            records.append(
                {
                    "comparison": "matched_left_right",
                    "condition_id": condition["id"],
                    "direction": "left_vs_right",
                    "left_episode_index": episode_index,
                    "right_episode_index": episode_index,
                    **_image_difference(
                        images[(condition["id"], "left", episode_index)],
                        images[(condition["id"], "right", episode_index)],
                    ),
                }
            )
    by_wording = {condition["wording"]: condition["id"] for condition in conditions}
    wording_pairs = (
        ("canonical", "short_paraphrase", "matched_canonical_short"),
        ("declarative_goal", "contrastive_goal", "matched_declarative_contrastive"),
    )
    for left_wording, right_wording, comparison in wording_pairs:
        for direction in TASK_DIRS:
            for episode_index in range(10):
                records.append(
                    {
                        "comparison": comparison,
                        "condition_id": f"cosmos_{left_wording}_vs_{right_wording}",
                        "direction": direction,
                        "left_episode_index": episode_index,
                        "right_episode_index": episode_index,
                        **_image_difference(
                            images[(by_wording[left_wording], direction, episode_index)],
                            images[(by_wording[right_wording], direction, episode_index)],
                        ),
                    }
                )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(row["comparison"], row["condition_id"], row["direction"])].append(row)
    summaries = []
    for (comparison, condition_id, direction), rows in sorted(grouped.items()):
        maes = [row["mae_0_255"] for row in rows]
        summaries.append(
            {
                "comparison": comparison,
                "condition_id": condition_id,
                "direction": direction,
                "pairs": len(rows),
                "mean_mae_0_255": float(np.mean(maes)),
                "median_mae_0_255": float(np.median(maes)),
                "p90_mae_0_255": float(np.percentile(maes, 90)),
                "max_mae_0_255": float(np.max(maes)),
            }
        )
    return {
        "first_conditioning_images": len(images),
        "prediction_chunks_validated": prediction_chunks_validated,
        "all_server_seeds_match_frozen_schedule": True,
        "interpretation": "Exact simulator state does not yield byte-identical realtime-rendered observations. These differences contaminate closed-loop first-action prompt/noise contrasts but not the exact fixed-observation command probe.",
        "summaries": summaries,
        "pairs": records,
    }


def _initial_physical_variation(closed: dict[str, Any]) -> dict[str, Any]:
    episodes = closed["episodes"]

    def difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
        cube = float(
            np.linalg.norm(
                np.asarray(left["first_recorded_cube_centroid_m"], dtype=np.float64)
                - np.asarray(right["first_recorded_cube_centroid_m"], dtype=np.float64)
            )
        )
        bowl = float(
            np.linalg.norm(
                np.asarray(left["first_recorded_bowl_centroid_m"], dtype=np.float64)
                - np.asarray(right["first_recorded_bowl_centroid_m"], dtype=np.float64)
            )
        )
        return {
            "cube_centroid_distance_m": cube,
            "bowl_centroid_distance_m": bowl,
            "max_object_centroid_distance_m": max(cube, bowl),
        }

    pairs: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        grouped[(row["condition_id"], row["direction"])].append(row)
    for (condition_id, direction), rows in sorted(grouped.items()):
        ordered = sorted(rows, key=lambda row: int(row["episode_seed"]))
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                left, right = ordered[left_index], ordered[right_index]
                pairs.append(
                    {
                        "comparison": "within_condition_direction",
                        "label": f"{condition_id}:{direction}",
                        "left_seed": left["episode_seed"],
                        "right_seed": right["episode_seed"],
                        **difference(left, right),
                    }
                )

    index = {
        (
            row["condition_id"],
            row["direction"],
            int(row["episode_seed"]),
        ): row
        for row in episodes
    }
    conditions = {row["condition_id"] for row in episodes}
    for condition_id in sorted(conditions):
        for seed in sorted(
            {row["episode_seed"] for row in episodes if row["condition_id"] == condition_id}
        ):
            left = index.get((condition_id, "left", int(seed)))
            right = index.get((condition_id, "right", int(seed)))
            if left is not None and right is not None:
                pairs.append(
                    {
                        "comparison": "matched_left_right",
                        "label": condition_id,
                        "left_seed": seed,
                        "right_seed": seed,
                        **difference(left, right),
                    }
                )

    def add_matched(
        comparison: str,
        label: str,
        left_filter: dict[str, Any],
        right_filter: dict[str, Any],
        pair_keys: tuple[str, ...],
    ) -> None:
        def select(criteria: dict[str, Any]) -> dict[tuple[Any, ...], dict[str, Any]]:
            return {
                tuple(row[key] for key in pair_keys): row
                for row in episodes
                if all(row.get(key) == value for key, value in criteria.items())
            }

        left_rows, right_rows = select(left_filter), select(right_filter)
        for key in sorted(set(left_rows) & set(right_rows)):
            pairs.append(
                {
                    "comparison": comparison,
                    "label": label,
                    "left_seed": left_rows[key]["episode_seed"],
                    "right_seed": right_rows[key]["episode_seed"],
                    **difference(left_rows[key], right_rows[key]),
                }
            )

    for model_id, native_horizon in (
        ("pi05_droid_vla", 15),
        ("cosmos3_edge_droid_wam", 32),
    ):
        add_matched(
            "matched_canonical_short",
            model_id,
            {"model_id": model_id, "wording": "canonical", "open_loop_horizon": native_horizon},
            {"model_id": model_id, "wording": "short_paraphrase", "open_loop_horizon": native_horizon},
            ("direction", "episode_seed"),
        )
        add_matched(
            "matched_declarative_contrastive",
            model_id,
            {
                "model_id": model_id,
                "wording": "declarative_goal",
                "controller": "static",
                "open_loop_horizon": native_horizon,
            },
            {
                "model_id": model_id,
                "wording": "contrastive_goal",
                "controller": "static",
                "open_loop_horizon": native_horizon,
            },
            ("direction", "episode_seed"),
        )
    for wording in WORDINGS:
        for direction in TASK_DIRS:
            add_matched(
                "matched_pi05_cosmos",
                f"{wording}:{direction}",
                {
                    "model_id": "pi05_droid_vla",
                    "wording": wording,
                    "controller": "static",
                    "open_loop_horizon": 15,
                    "direction": direction,
                },
                {
                    "model_id": "cosmos3_edge_droid_wam",
                    "wording": wording,
                    "controller": "static",
                    "open_loop_horizon": 32,
                    "direction": direction,
                },
                ("episode_seed",),
            )

    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        summary_groups[(row["comparison"], row["label"])].append(row)
    summaries = []
    for (comparison, label), rows in sorted(summary_groups.items()):
        distances = [row["max_object_centroid_distance_m"] for row in rows]
        summaries.append(
            {
                "comparison": comparison,
                "label": label,
                "pairs": len(rows),
                "mean_max_object_centroid_distance_m": float(np.mean(distances)),
                "p90_max_object_centroid_distance_m": float(np.percentile(distances, 90)),
                "max_object_centroid_distance_m": float(np.max(distances)),
            }
        )
    return {
        "episodes": len(episodes),
        "exact_physical_reset_state_fingerprints": len(
            closed["physical_initial_state_fingerprint_counts"]
        ),
        "full_recorded_initial_state_fingerprints": len(
            closed["full_recorded_initial_state_fingerprint_counts"]
        ),
        "interpretation": "Robot/object reset arrays are exact. Full recorded reset groups have one schema-specific hash per checkpoint because Cosmos records two additional camera poses. First-recorded object centroids may still differ after settling; the distances below quantify that variation separately from pixel-render variation.",
        "summaries": summaries,
        "pairs": pairs,
    }


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_static_success(closed: dict[str, Any], output: Path) -> None:
    groups = [
        row
        for row in closed["group_summaries"]
        if row["controller"] == "static"
        and row["open_loop_horizon"] in {15, 32}
    ]
    slots = [(wording, direction) for wording in WORDINGS for direction in ("left", "right")]
    labels = [f"{WORDING_LABELS[wording]}\n{direction}" for wording, direction in slots]
    fig, ax = plt.subplots(figsize=(12.5, 4.4))
    x = np.arange(len(slots))
    width = 0.34
    for model_index, model_id in enumerate(MODEL_LABELS):
        selected = {
            (row["wording"], row["direction"]): row
            for row in groups
            if row["model_id"] == model_id
        }
        values = [selected[slot]["success_rate"] for slot in slots]
        intervals = [selected[slot]["success_beta11_interval_95"] for slot in slots]
        positions = x + (model_index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
        )
        lower = np.asarray([interval[0] for interval in intervals])
        upper = np.asarray([interval[1] for interval in intervals])
        # A central Bayesian credible interval need not include the observed
        # boundary proportion (for example, 0/10). Draw its endpoints directly
        # instead of misusing a non-negative error-bar distance around the raw
        # bar height.
        ax.vlines(positions, lower, upper, color="#20252b", linewidth=1.2, zorder=4)
        cap_half_width = width * 0.12
        ax.hlines(
            np.concatenate((lower, upper)),
            np.concatenate((positions - cap_half_width, positions - cap_half_width)),
            np.concatenate((positions + cap_half_width, positions + cap_half_width)),
            color="#20252b",
            linewidth=1.2,
            zorder=4,
        )
        for bar, value, interval in zip(bars, values, intervals):
            label_height = max(value, interval[1]) + 0.025
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_height,
                f"{value:.0%}",
                ha="center",
            )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Requested-goal success rate")
    ax.set_title(
        "Matched neutral-start closed-loop success "
        "(10 seeds per bar; whiskers are Beta(1,1) 95% credible intervals)"
    )
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "direct_language_success_with_intervals.png", bbox_inches="tight")
    plt.close(fig)


def _plot_offsets(closed: dict[str, Any], output: Path) -> None:
    episodes = [
        row
        for row in closed["episodes"]
        if row["controller"] == "static" and row["open_loop_horizon"] in {15, 32}
    ]
    slots = []
    data = []
    colors = []
    for model_id in MODEL_LABELS:
        for wording in WORDINGS:
            for direction in ("left", "right"):
                values = [
                    row["requested_signed_final_offset_m"]
                    for row in episodes
                    if row["model_id"] == model_id
                    and row["wording"] == wording
                    and row["direction"] == direction
                ]
                slots.append(
                    f"{MODEL_LABELS[model_id].split(' ')[0]}\n{WORDING_LABELS[wording]}\n{direction}"
                )
                data.append(values)
                colors.append(MODEL_COLORS[model_id])
    fig, ax = plt.subplots(figsize=(15.5, 4.8))
    parts = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.45)
    parts["cmedians"].set_color("#222222")
    rng = np.random.default_rng(0)
    for index, (values, color) in enumerate(zip(data, colors), start=1):
        ax.scatter(
            index + rng.uniform(-0.07, 0.07, len(values)),
            values,
            s=15,
            color=color,
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks(np.arange(1, len(slots) + 1), slots)
    ax.set_ylabel("Final offset toward requested side (m)")
    ax.set_title("Directionality of final cube position; positive is prompt-consistent")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "direct_language_requested_side_offsets.png", bbox_inches="tight")
    plt.close(fig)


def _plot_direct_prompt_robustness(closed: dict[str, Any], output: Path) -> None:
    groups = {
        (row["wording"], row["model_id"], row["direction"]): row
        for row in closed["group_summaries"]
        if row["controller"] == "static" and row["open_loop_horizon"] in {15, 32}
    }
    columns = [
        ("pi05_droid_vla", "left"),
        ("pi05_droid_vla", "right"),
        ("cosmos3_edge_droid_wam", "left"),
        ("cosmos3_edge_droid_wam", "right"),
    ]
    values = np.asarray(
        [
            [groups[(wording, model_id, direction)]["success_rate"] for model_id, direction in columns]
            for wording in WORDINGS
        ]
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    image = ax.imshow(values, vmin=0, vmax=1, cmap="YlGn", aspect="auto")
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{values[row_index, column_index]:.0%}\n(n=10)",
                ha="center",
                va="center",
                color="white" if values[row_index, column_index] >= 0.62 else "#222222",
            )
    ax.set_yticks(np.arange(len(WORDINGS)), [WORDING_LABELS[wording] for wording in WORDINGS])
    ax.set_xticks(
        np.arange(len(columns)),
        [
            f"{MODEL_LABELS[model_id].split(' ')[0]}\n{direction}"
            for model_id, direction in columns
        ],
    )
    ax.set_xlabel("Checkpoint and requested direction")
    ax.set_ylabel("Episode-static task wording")
    ax.set_title("Direct task-language robustness without a subtask coach")
    fig.colorbar(image, ax=ax, label="Success rate")
    fig.tight_layout()
    fig.savefig(output / "direct_prompt_robustness.png", bbox_inches="tight")
    plt.close(fig)


def _plot_semantic_quadrants(semantic: dict[str, Any], output: Path) -> None:
    order = [
        "imagines_requested_executes_requested",
        "imagines_requested_executes_not_requested",
        "does_not_imagine_requested_executes_requested",
        "neither_imagines_nor_executes_requested",
        "uncertain_future",
    ]
    labels = {
        "imagines_requested_executes_requested": "imagines + executes",
        "imagines_requested_executes_not_requested": "imagines, not executes",
        "does_not_imagine_requested_executes_requested": "executes, not imagines",
        "neither_imagines_nor_executes_requested": "neither",
        "uncertain_future": "future uncertain",
    }
    groups = semantic["groups"]
    names = [f"{WORDING_LABELS[row['wording']]}\n{row['direction']}" for row in groups]
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    bottoms = np.zeros(len(groups))
    for quadrant in order:
        values = np.asarray(
            [row["quadrant_counts"].get(quadrant, 0) / row["chunks"] for row in groups]
        )
        ax.bar(
            np.arange(len(groups)),
            values,
            bottom=bottoms,
            color=QUADRANT_COLORS[quadrant],
            label=labels[quadrant],
        )
        bottoms += values
    for index, row in enumerate(groups):
        ax.text(index, 1.025, f"n={row['chunks']}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(groups)), names)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Fraction of replan chunks")
    ax.set_title("Cosmos imagined requested relation versus executed relation")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    fig.savefig(output / "cosmos_imagination_execution_quadrants.png", bbox_inches="tight")
    plt.close(fig)


def _plot_observation_variation(audit: dict[str, Any], output: Path) -> None:
    rows = audit["summaries"]
    labels = []
    means = []
    p90 = []
    colors = []
    for row in rows:
        comparison = row["comparison"]
        condition = row["condition_id"].replace("cosmos_", "").replace("_static32", "")
        direction = row["direction"].replace("left_vs_right", "L/R")
        if comparison == "within_condition_direction":
            label = f"{condition}\nwithin {direction}"
            color = "#8da0cb"
        elif comparison == "matched_left_right":
            label = f"{condition}\nmatched L/R"
            color = "#fc8d62"
        else:
            prompt_pair = comparison.replace("matched_", "").replace("_", "/")
            label = f"{prompt_pair}\n{direction}"
            color = "#66c2a5"
        labels.append(label)
        means.append(row["mean_mae_0_255"])
        p90.append(row["p90_mae_0_255"])
        colors.append(color)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.8, 4.4))
    bars = ax.bar(x, means, color=colors)
    ax.scatter(x, p90, marker="_", s=180, color="#222222", label="pairwise p90")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.08, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("First-conditioning-image MAE (0–255)")
    ax.set_title("Exact physical resets are not byte-identical realtime renders")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "cosmos_conditioning_image_variation.png", bbox_inches="tight")
    plt.close(fig)


def _plot_semantic_threshold_sensitivity(semantic: dict[str, Any], output: Path) -> None:
    rows = semantic["threshold_sensitivity"]
    overall = semantic["threshold_sensitivity_overall"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True, sharey=True)
    groups = sorted({(row["wording"], row["direction"]) for row in rows})
    colors = {"left": "#6a51a3", "right": "#d95f0e"}
    styles = {
        "canonical": "-",
        "short_paraphrase": "--",
        "declarative_goal": "-.",
        "contrastive_goal": ":",
    }
    for wording, direction in groups:
        selected = sorted(
            [row for row in rows if row["wording"] == wording and row["direction"] == direction],
            key=lambda row: row["cross_camera_threshold_m"],
        )
        label = f"{WORDING_LABELS[wording]} {direction}"
        x = [row["cross_camera_threshold_m"] for row in selected]
        series = (
            [row["coverage_fraction"] for row in selected],
            [
                row["executed_positive_future_coverage_fraction"]
                if row["executed_positive_future_coverage_fraction"] is not None
                else np.nan
                for row in selected
            ],
            [
                row["imagination_execution_agreement_among_certain"]
                if row["imagination_execution_agreement_among_certain"] is not None
                else np.nan
                for row in selected
            ],
        )
        for axis, values in zip(axes, series):
            axis.plot(
                x,
                values,
                marker="o",
                markersize=3.5,
                linewidth=1.1,
                alpha=0.55,
                color=colors[direction],
                linestyle=styles[wording],
                label=label,
            )
    overall_x = [row["cross_camera_threshold_m"] for row in overall]
    overall_series = (
        [row["coverage_fraction"] for row in overall],
        [row["executed_positive_future_coverage_fraction"] for row in overall],
        [row["imagination_execution_agreement_among_certain"] for row in overall],
    )
    for axis, values in zip(axes, overall_series):
        axis.plot(
            overall_x,
            values,
            marker="o",
            markersize=5,
            linewidth=2.5,
            color="#17222f",
            label="all conditions",
            zorder=10,
        )
        axis.annotate(
            f"{values[-1]:.0%}",
            xy=(overall_x[-1], values[-1]),
            xytext=(-6, -16),
            textcoords="offset points",
            ha="right",
            color="#17222f",
            fontsize=9,
            fontweight="bold",
        )
    titles = (
        "All-chunk coverage",
        "Coverage when execution\nreached the requested relation",
        "Agreement among\nscored chunks",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel("Cross-camera disagreement threshold (m)")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.2)
        axis.axvline(0.20, color="#8a8f98", linewidth=0.9, linestyle=":")
    axes[0].set_ylabel("Fraction")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=5, loc="lower center")
    fig.suptitle(
        "Tighter cross-camera agreement sharply reduces usable semantic coverage"
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    fig.savefig(output / "semantic_threshold_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def _plot_probe(probes: dict[str, Any], output: Path) -> None:
    condition_order = [row["condition"] for row in probes["pi05"]["manifest"]["records"]]
    short = {
        "task_left": "task L",
        "task_left_repeat": "repeat",
        "task_left_paraphrase": "paraphrase",
        "task_right": "task R",
        "subtask_grasp": "subtask",
        "atomic_left": "atomic L",
        "atomic_right": "atomic R",
        "gripper_trace_to_cube": "trace",
        "point_cube": "point cube",
        "point_left_target": "point L",
        "point_right_target": "point R",
        "combination_left": "combo L",
        "combination_right": "combo R",
        "unrelated_control": "unrelated",
        "noun_swap_control": "noun swap",
        "contradictory_control": "contradictory",
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.3), sharex=True)
    x = np.arange(len(condition_order))
    for axis, model in zip(axes, ("pi05", "cosmos")):
        records = {row["condition"]: row for row in probes[model]["manifest"]["records"]}
        values = [records[name]["action_rms_vs_task_left"] for name in condition_order]
        axis.bar(x, values, color=MODEL_COLORS["pi05_droid_vla" if model == "pi05" else "cosmos3_edge_droid_wam"])
        axis.set_ylabel("Action RMS")
        axis.set_title("π0.5 VLA" if model == "pi05" else "Cosmos WAM")
        axis.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x, [short[name] for name in condition_order], rotation=38, ha="right")
    fig.suptitle("Frozen-observation command sensitivity relative to canonical left task")
    fig.tight_layout()
    fig.savefig(output / "command_probe_action_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def _plot_direct_task_probe(probe: dict[str, Any], output: Path) -> None:
    families = [
        "canonical",
        "short",
        "declarative",
        "contrastive target first",
        "contrastive target last",
    ]
    order_labels = ["contrastive order left", "contrastive order right"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    width = 0.35
    for model_index, model in enumerate(("pi05", "cosmos")):
        rows = {row["prompt_family"]: row for row in probe[model]["paired_action_rms"]}
        model_id = "pi05_droid_vla" if model == "pi05" else "cosmos3_edge_droid_wam"
        positions = np.arange(len(families)) + (model_index - 0.5) * width
        axes[0].bar(
            positions,
            [rows[family]["action_rms"] for family in families],
            width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
        )
        positions = np.arange(len(order_labels)) + (model_index - 0.5) * width
        axes[1].bar(
            positions,
            [rows[label]["action_rms"] for label in order_labels],
            width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
        )
    axes[0].set_xticks(np.arange(len(families)), families, rotation=24, ha="right")
    axes[0].set_title("Left-versus-right action separation")
    axes[1].set_xticks(np.arange(len(order_labels)), ["left target", "right target"])
    axes[1].set_title("Target-first versus target-last sensitivity")
    for axis in axes:
        axis.set_ylabel("Action RMS on identical input bytes")
        axis.grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    fig.suptitle("Exact-observation direct task-language probe")
    fig.tight_layout()
    fig.savefig(output / "direct_task_exact_probe.png", bbox_inches="tight")
    plt.close(fig)


def _read_probe_frames(path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted:
            ok, bgr = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    if wanted:
        raise RuntimeError(f"Missing frames {sorted(wanted)} in {path}")
    return [frames[index] for index in indices]


def _plot_selected_probe_futures(root: Path, probes: dict[str, Any], output: Path) -> None:
    selected = [
        "task_left",
        "task_left_repeat",
        "task_left_paraphrase",
        "task_right",
        "unrelated_control",
        "noun_swap_control",
    ]
    labels = {
        "task_left": "canonical left",
        "task_left_repeat": "exact repeat",
        "task_left_paraphrase": "left paraphrase",
        "task_right": "opposite right",
        "unrelated_control": "unrelated drawer",
        "noun_swap_control": "banana noun swap",
    }
    semantic = {
        row["condition"]: row
        for row in probes["cosmos"]["semantic_futures"]["rows"]
    }
    indices = (0, 16, 32)
    fig, axes = plt.subplots(len(selected), len(indices), figsize=(9.4, 13.0))
    for row_index, condition in enumerate(selected):
        frames = _read_probe_frames(
            root / "command_probe/cosmos_gpu1" / f"{condition}_future.mp4", indices
        )
        relation = semantic[condition]["predicted_relation"] or "uncertain"
        for column_index, (frame_index, frame) in enumerate(zip(indices, frames)):
            axis = axes[row_index, column_index]
            axis.imshow(frame)
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title("conditioning" if frame_index == 0 else f"future frame {frame_index}")
            if column_index == 0:
                axis.set_ylabel(f"{labels[condition]}\nsemantic: {relation}")
    fig.suptitle("Same observation and seed: selected Cosmos prompt-conditioned futures")
    fig.tight_layout()
    fig.savefig(output / "command_probe_selected_futures.png", bbox_inches="tight")
    plt.close(fig)


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _write_raw_evidence_manifest(
    path: Path, roots: list[Path], *, scope: str
) -> dict[str, Any]:
    allowed_suffixes = {
        ".csv",
        ".hdf5",
        ".jpg",
        ".json",
        ".jsonl",
        ".md",
        ".mp4",
        ".npy",
        ".npz",
        ".png",
        ".py",
        ".sh",
        ".txt",
    }
    files: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(root)
        candidates = [root] if root.is_file() else root.rglob("*")
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
                files[str(candidate.resolve())] = candidate
    rows = []
    for absolute, candidate in sorted(files.items()):
        rows.append(
            {
                "absolute_path": absolute,
                "bytes": candidate.stat().st_size,
                "sha256": _sha256(candidate),
            }
        )
    _write_summary_csv(path, rows)
    return {
        "path": str(path.resolve()),
        "files": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "sha256": _sha256(path),
        "scope": scope,
    }


def _evidence_markdown(compiled: dict[str, Any], root: Path) -> str:
    closed = compiled["prospective"]["closed_loop"]
    semantic = compiled["prospective"]["cosmos_semantic_futures"]
    probes = compiled["prospective"]["command_probe"]
    direct_probe = compiled["prospective"]["direct_task_command_probe"]
    trajectories = compiled["prospective"]["trajectory_evidence"]
    semantic_visuals = compiled["prospective"]["semantic_future_visualization"]
    lines = [
        "# VLA-WAM study evidence index",
        "",
        "Status: complete direct-language grid with an original confirmatory tier, a prospectively frozen post-interim stress tier, and a separately labeled retrospective WAM tier.",
        "",
        "## Integrity checks",
        "",
        f"- Closed-loop episodes: **{closed['episode_count']}/{EXPECTED_EPISODES}**.",
        f"- Original confirmatory episodes: **{compiled['integrity']['original_confirmatory_episode_count']}/{EXPECTED_CONFIRMATORY_EPISODES}**.",
        f"- Post-interim direct-language stress episodes: **{compiled['integrity']['post_interim_direct_stress_episode_count']}/{EXPECTED_DIRECT_STRESS_EPISODES}**.",
        "- Oracle or dynamic-prompt episodes in the analysis: **0**.",
        f"- Shared physical robot/object reset fingerprints: **{len(closed['physical_initial_state_fingerprint_counts'])}**; full recorder-schema fingerprints: **{len(closed['full_recorded_initial_state_fingerprint_counts'])}**.",
        f"- First-recorded cube/bowl physical observations audited: **{compiled['integrity']['initial_physical_observations_audited']}**.",
        f"- Fixed-observation probe conditions: **{len(probes['pi05']['manifest']['records'])} per model**.",
        f"- Exact direct-task prompt conditions: **{len(direct_probe['pi05']['manifest']['records'])}/{EXPECTED_DIRECT_TASK_PROBE_CONDITIONS} per model**.",
        f"- Cosmos semantically scored confirmation chunks: **{semantic['total_chunks']}** across **{semantic['total_episodes']} episodes**.",
        f"- Prompt-blind semantic scorer coverage: **{semantic['overall']['certain_chunks']}/{semantic['overall']['chunks']} chunks**; every abstention remains explicit.",
        f"- Frozen-order semantic categories rendered: **{semantic_visuals['observed_category_count']}/5 observed**; absent categories remain explicit empty panels.",
        f"- Cosmos recorded prediction chunks with verified request/server seeds, step alignment, and shapes: **{compiled['integrity']['cosmos_prediction_chunks_validated']}**.",
        f"- Completed thermal-guard lifecycles without emergency stop: **{compiled['integrity']['thermal_guard_logs_verified']}/{len(EXPECTED_THERMAL_LOGS)}**.",
        f"- Executed trajectory panels indexed: **{trajectories['rendered_episode_count']}/{EXPECTED_EPISODES}**, including **{trajectories['success_count']} successes** and **{trajectories['failure_count']} failures**.",
        "- Calibration, command-probe, and run-manifest hashes were verified by the compilers.",
        "",
        "## Prospective evidence",
        "",
        "| Evidence | Artifact | Purpose |",
        "| --- | --- | --- |",
        "| Frozen design | `../preregistration.json` | Questions, fixed grid, primary/secondary metrics, stopping rule |",
        "| Direct-language scope amendment | `../direct_language_scope_amendment_003.json` | Retires the oracle grid and freezes declarative/contrastive task-language stress conditions before those runs |",
        "| Metric amendment | `../metric_amendment_001.json` | Exact paper-style progression after primary-source verification |",
        "| Observation amendment | `../observation_variation_amendment_001.json` | Downgrades closed-loop action contrast after measured renderer variation |",
        "| Initial-state schema amendment | `../initial_state_schema_amendment_006.json` | Separates exact physical reset identity from checkpoint-specific camera recorder schemas |",
        "| Thermal-control amendment | `../thermal_control_amendment_001.json` | Freezes pause/resume and emergency-stop behavior after the first matched-role thermal stop |",
        "| Thermal timing amendment | `../thermal_timing_amendment_002.json` | Treats guarded client request timing as an upper bound and forbids fabricated phase attribution |",
        "| Semantic target parser amendment | `../semantic_target_parser_amendment_004.json` | Uses matched task identity rather than interpreting contrastive prompt negation inside the visual scorer |",
        "| Execution geometry amendment | `../execution_geometry_amendment_005.json` | Aligns derived task/execution relations with RoboLab rigid-object root poses while preserving visual-centroid calibration |",
        "| Trajectory visualization plan | `../trajectory_visualization_plan.json` | Freezes complete-gallery, deterministic social-panel, and retrospective-exemplar rules |",
        "| Semantic-future visualization plan | `../semantic_future_visualization_plan.json` | Freezes first-in-order example selection before confirmation semantic labels |",
        "| Grounded probe plan | `../command_probe_plan.json` | Hash-pinned observation, six command styles, controls, seed |",
        "| Direct-task probe plan | `../direct_task_command_probe_plan.json` | Exact-input syntax, contrastive scope, and target-token-order diagnostic |",
        "| Closed-loop episode table | `episodes.csv` | One row per registered direct-language rollout, with analysis tier |",
        "| Closed-loop summary | `closed_loop_summary.json` | Success, progression, offsets, timing, contrasts |",
        "| Complete trajectory index | `../trajectory_evidence/trajectory_index.csv` and `.json` | Every success and failure with endpoint class, event steps, raw paths, and rendered panel |",
        "| Trajectory evidence gallery | `../trajectory_evidence/gallery/index.html` | Filterable visual audit of every registered episode |",
        "| Cosmos future semantics | `compiled_evidence.json` | Prompt-blind imagined/executed quadrants and coverage |",
        "| Semantic-future examples | `../semantic_future_visualization/selection.json` | Deterministic source rows, videos, caches, and hashes for each observed quadrant |",
        "| Renderer variation audit | `cosmos_observation_variation.csv` | First-conditioning-image differences within and across static conditions |",
        "| Physical settling audit | `initial_physical_variation.csv` | Reset-state identity versus first-recorded cube/bowl centroid differences |",
        "| Human semantic audit | `../semantic_confirmation_audit_plan.json`, `../semantic_confirmation_audit_amendment_002.json`, and `../semantic_confirmation_audit.md` | Outcome-independent sheet samples and completed visual review |",
        "| Publication article | `../../../docs/VLA_VS_WAM_STEERABILITY_STUDY.md` | Long-form interpretation with complete claim boundaries and visual evidence |",
        "| Social launch kit | `../../../docs/VLA_WAM_STEERABILITY_SOCIAL_COPY.md` | Post-ready X/LinkedIn copy, carousel order, alt text, and claim guardrails |",
        "| Command probes | `compiled_evidence.json` | Exact repeat, command sensitivity, semantic futures |",
        "| GPU assignment audit | `../cosmos_gpu_assignment_audit.json` | Quantifies why cross-card Cosmos output was excluded |",
        "| Cosmos resource snapshot | `../operational_snapshot_cosmos_confirmation.json` | Temperatures, memory, utilization, and physical GPU roles during a valid WAM request |",
        "| pi0.5 resource snapshot | `../operational_snapshot_pi05_confirmation.json` | Temperatures, memory, utilization, and physical GPU roles during a valid VLA request |",
        "| Thermal event logs | `../thermal_logs/*.jsonl` | Complete pause/resume lifecycle, cooldown duration, sampled peak, and emergency-stop audit for all eight definitive batches |",
        "| Raw file hash ledger | `raw_evidence_manifest.csv` | Byte size and SHA-256 for every prospective raw/derived evidence file |",
        "| Supporting hash ledger | `supporting_evidence_manifest.csv` | Calibration, exclusions, and separately labeled retrospective raw/derived evidence |",
        "| Setup exclusion | `../setup_exclusions/2026-08-02_cosmos_canonical_driver_check.md` | Failed startup with zero requests, excluded transparently |",
        "| Thermal exclusion | `../setup_exclusions/2026-08-02_cosmos_gpu0_thermal_restart.md` | Interrupted and cross-GPU batches preserved outside estimates |",
        "| Confirmation thermal exclusion | `../setup_exclusions/2026-08-02_cosmos_confirmation_thermal_stop.md` | Whole matched-role batch excluded after the simulator reached the 90 C stop threshold |",
        "| Pre-guard wording exclusion | `../setup_exclusions/2026-08-02_cosmos_vague_pre_thermal_guard.md` | Completed short-paraphrase batch rerun so both wordings share one logged thermal cadence |",
        "| Oracle scope exclusion | `../setup_exclusions/2026-08-02_oracle_scope_change.md` | Preserves the interrupted coached batch while excluding it from every direct-language estimate |",
        "",
        "## Figures",
        "",
        "- `direct_language_success_with_intervals.png`: binary success for all four static task wordings with Beta(1,1) 95% intervals.",
        "- `direct_language_requested_side_offsets.png`: endpoint directionality, including failures.",
        "- `direct_prompt_robustness.png`: model-by-wording-by-direction success matrix without a coach.",
        "- `../trajectory_evidence/blog/all_executed_paths_and_endpoints.png`: every executed cube path and endpoint, faceted by checkpoint and wording.",
        "- `../trajectory_evidence/blog/failure_progress_anatomy.png`: mutually exclusive action-stage anatomy for every success and failure.",
        "- `../trajectory_evidence/social/first_seed_stress_landscape_1600x900.png`: deterministic same-seed stress-language comparison for social sharing.",
        "- `../trajectory_evidence/social/first_seed_stress_square_1200x1200.png`: square social crop of the same deterministic comparison.",
        "- `../trajectory_evidence/social/steerability_scorecard_1600x900.png` and `...1200x1200.png`: complete 16-cell checkpoint/wording/direction scorecard in share-ready formats.",
        "- `../trajectory_evidence/social/failure_progress_anatomy_1200x1200.png`: square share card retaining every success and failure stage.",
        "- `cosmos_conditioning_image_variation.png`: measured realtime-renderer variation despite exact physical resets.",
        "- `cosmos_imagination_execution_quadrants.png`: WAM-only semantic future/action agreement.",
        "- `semantic_threshold_sensitivity.png`: scorer coverage/agreement at 0.10, 0.15, and frozen 0.20 m reliability thresholds.",
        "- `../semantic_future_visualization/blog/selected_semantic_future_examples.png`: frozen first-in-order generated-video strip for every observed semantic category.",
        "- `../semantic_future_visualization/social/wam_semantic_quadrants_1600x900.png` and `...1200x1200.png`: share-ready actual-future examples for the four certain imagination/execution outcomes.",
        "- `command_probe_action_sensitivity.png`: same-observation six-style prompt response.",
        "- `direct_task_exact_probe.png`: same-input left/right and contrastive word-order action separation.",
        "- `command_probe_selected_futures.png`: selected Cosmos future strips with frozen prompt-blind relation labels.",
        "",
        "## Retrospective evidence tier",
        "",
        "Efficient-WAM, FastWAM, LingBot-VA, and the earlier π0.5/Cosmos pilots remain in `../../wam_language_gate/`. They inform model selection and failure analysis, but they are not pooled into the prospective confidence intervals.",
        "",
        "## Statistical guardrails",
        "",
        "- A Beta(1,1) interval accompanies each success proportion.",
        "- Declarative and contrastive conditions are explicitly post-interim stress tests, not retroactively presented as part of the original preregistration.",
        "- Exact paired McNemar tests are exploratory and uncorrected for multiple comparisons.",
        "- Replan chunks are correlated within episodes; semantic quadrant rates are descriptive and receive no pseudo-replicated binomial interval.",
        "- One checkpoint represents each model class, so no result establishes a general VLA-versus-WAM class effect.",
        "- Fixed-observation distances establish sensitivity only. Directionally appropriate closed-loop outcomes establish control.",
        "",
        "## Provenance",
        "",
    ]
    for name, state in compiled["provenance"]["repositories"].items():
        lines.append(f"- {name}: `{state['commit']}` ({state['branch']}).")
    lines.extend(["", "Core file hashes are stored under `provenance.files` in `compiled_evidence.json`.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=Path("artifacts/vla_wam_shared_v1"))
    parser.add_argument("--closed-loop", type=Path)
    parser.add_argument("--retrospective", type=Path, default=Path("artifacts/wam_language_gate/summary.json"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    root = (workspace / args.study_root).resolve() if not args.study_root.is_absolute() else args.study_root.resolve()
    output = args.output_dir or (root / "final_evidence")
    output = output.resolve()
    blog_path = workspace / "docs/VLA_VS_WAM_STEERABILITY_STUDY.md"
    blog_text = blog_path.read_text()
    if "RESULT_TBD" in blog_text:
        raise RuntimeError(
            "Publication draft still contains RESULT_TBD markers; resolve every claim "
            "from the closed evidence before final compilation"
        )
    normalized_blog = " ".join(blog_text.split()).lower()
    for required_boundary in (
        "personal research analysis; views are my own.",
        "one checkpoint cannot represent a whole model class.",
        "study has no subtask coach",
    ):
        if required_boundary not in normalized_blog:
            raise RuntimeError(
                f"Publication draft is missing required claim boundary: {required_boundary!r}"
            )
    social_copy_path = workspace / "docs/VLA_WAM_STEERABILITY_SOCIAL_COPY.md"
    social_copy_text = social_copy_path.read_text()
    if "TBD" in social_copy_text:
        raise RuntimeError(
            "Social launch kit still contains TBD markers; resolve every public claim "
            "from the closed evidence before final compilation"
        )
    run_manifest = _load(root / "run_manifest.json")
    if int(run_manifest["expected_episode_count"]) != EXPECTED_EPISODES:
        raise RuntimeError("Run-manifest episode count disagrees with the study compiler")
    if any(condition["controller"] != "static" for condition in run_manifest["conditions"]):
        raise RuntimeError("Run manifest contains an oracle or dynamic controller")
    closed_path = args.closed_loop or (output / "closed_loop_summary.json")
    closed = _load(closed_path)
    if not closed.get("complete") or closed["episode_count"] != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Closed-loop grid incomplete: complete={closed.get('complete')} episodes={closed.get('episode_count')}"
        )
    invalid_controller_rows = [
        row
        for row in closed["episodes"]
        if row["controller"] != "static" or row["open_loop_horizon"] not in {15, 32}
    ]
    if invalid_controller_rows:
        raise RuntimeError("Closed-loop evidence contains a coached or non-native-horizon episode")
    confirmatory_rows = [
        row for row in closed["episodes"] if row["analysis_tier"] == "original_confirmatory"
    ]
    stress_rows = [
        row
        for row in closed["episodes"]
        if row["analysis_tier"] == "post_interim_direct_stress"
    ]
    if (
        len(confirmatory_rows) != EXPECTED_CONFIRMATORY_EPISODES
        or len(stress_rows) != EXPECTED_DIRECT_STRESS_EPISODES
    ):
        raise RuntimeError("Confirmatory/stress episode accounting does not match the amended grid")
    if len(closed["physical_initial_state_fingerprint_counts"]) != 1:
        raise RuntimeError(
            "Closed-loop inputs do not share one exact physical reset-state fingerprint"
        )
    if len(closed["full_recorded_initial_state_fingerprint_counts"]) != 2:
        raise RuntimeError(
            "Expected two checkpoint-specific full recorder-state fingerprints"
        )
    initial_schema = _load(root / "initial_state_schema_amendment_006.json")
    observed_hashes = initial_schema["observed_hashes"]
    expected_physical_hashes = {
        observed_hashes["shared_physical_initial_state_sha256"]
    }
    if set(closed["physical_initial_state_fingerprint_counts"]) != expected_physical_hashes:
        raise RuntimeError("Physical reset-state hash disagrees with the schema amendment")
    expected_full_hashes = {
        observed_hashes["cosmos_full_recorded_initial_state_sha256"],
        observed_hashes["pi05_full_recorded_initial_state_sha256"],
    }
    if set(closed["full_recorded_initial_state_fingerprint_counts"]) != expected_full_hashes:
        raise RuntimeError("Full recorder-state hashes disagree with the schema amendment")
    geometry_sources = {row.get("relation_geometry_source") for row in closed["episodes"]}
    if geometry_sources != {"rigid_object_root_pose_in_robot_frame"}:
        raise RuntimeError(f"Unexpected closed-loop relation geometry: {geometry_sources}")

    plan_path = root / "command_probe_plan.json"
    plan_sha = _sha256(plan_path)
    cosmos_probe_semantic_path = (
        root / "command_probe/cosmos_gpu1_semantics/semantic_future_summary.json"
    )
    cosmos_probe_semantic = _load(cosmos_probe_semantic_path)
    probes = _probe_summary(root, plan_sha, cosmos_probe_semantic)
    direct_plan_path = root / "direct_task_command_probe_plan.json"
    direct_plan_sha = _sha256(direct_plan_path)
    direct_cosmos_probe_semantic_path = (
        root / "command_probe/direct_task_cosmos_semantics/semantic_future_summary.json"
    )
    direct_probes = _direct_task_probe_summary(
        root,
        direct_plan_sha,
        _load(direct_cosmos_probe_semantic_path),
    )
    calibration_sha = _sha256(root / "semantic_future_calibration.json")
    semantic_inputs = []
    for wording, path in (
        ("canonical", root / "semantic_confirmation/cosmos_canonical/semantic_quadrants_summary.json"),
        ("short_paraphrase", root / "semantic_confirmation/cosmos_vague/semantic_quadrants_summary.json"),
        ("declarative_goal", root / "semantic_confirmation/cosmos_declarative/semantic_quadrants_summary.json"),
        ("contrastive_goal", root / "semantic_confirmation/cosmos_contrastive/semantic_quadrants_summary.json"),
    ):
        summary = _load(path)
        if summary["calibration_sha256"] != calibration_sha:
            raise RuntimeError(f"Semantic calibration hash mismatch in {path}")
        if summary.get("execution_geometry_source") != "rigid_object_root_pose_in_robot_frame":
            raise RuntimeError(f"Semantic execution geometry mismatch in {path}")
        if {
            row.get("execution_geometry_source") for row in summary["rows"]
        } != {"rigid_object_root_pose_in_robot_frame"}:
            raise RuntimeError(f"Semantic row geometry mismatch in {path}")
        semantic_inputs.append((wording, path.parent, summary))
    semantic = _semantic_aggregate(semantic_inputs)
    semantic_visualization = _semantic_visualization_summary(
        root, workspace, semantic["total_chunks"]
    )

    observation_variation = _cosmos_observation_variation(run_manifest)
    physical_variation = _initial_physical_variation(closed)
    thermal_control = _thermal_log_summary(root)
    trajectory_evidence = _trajectory_evidence_summary(root, workspace, closed)
    raw_roots = [Path(condition["output_root"]) for condition in run_manifest["conditions"]]
    raw_roots.extend(
        [
            root / "command_probe/pi05",
            root / "command_probe/cosmos_gpu1",
            root / "command_probe/cosmos_gpu1_semantics",
            root / "command_probe/direct_task_pi05",
            root / "command_probe/direct_task_cosmos",
            root / "command_probe/direct_task_cosmos_semantics",
            root / "semantic_confirmation/cosmos_canonical",
            root / "semantic_confirmation/cosmos_vague",
            root / "semantic_confirmation/cosmos_declarative",
            root / "semantic_confirmation/cosmos_contrastive",
            root / "thermal_logs",
            root / "trajectory_evidence/summary.json",
            root / "trajectory_evidence/trajectory_index.csv",
            root / "trajectory_evidence/trajectory_index.json",
            root / "trajectory_evidence/blog",
            root / "trajectory_evidence/social",
            root / "trajectory_evidence/gallery/index.html",
            root / "semantic_future_visualization",
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(
        output / "cosmos_observation_variation.csv",
        observation_variation["pairs"],
    )
    _write_summary_csv(
        output / "initial_physical_variation.csv",
        physical_variation["pairs"],
    )
    raw_manifest = _write_raw_evidence_manifest(
        output / "raw_evidence_manifest.csv",
        raw_roots,
        scope="All supported data, image, video, documentation, and thermal-event files under the eight direct-language run roots plus command-probe, semantic-scoring, and thermal-log roots.",
    )
    supporting_roots = [
        Path("/home/ali/projects/RoboLab/output/v1_calibration_cosmos_left_5100"),
        Path("/home/ali/projects/RoboLab/output/v1_calibration_cosmos_right_5100"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_canonical_original_hot_gpu0"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_vague_interrupted_hot_gpu0"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_vague_pre_thermal_guard"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_canonical_interrupted_thermal_gpu1roles"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_h5_static"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_h5_oracle_scope_change_excluded"),
        root / "calibration_semantic_dry_run",
        root / "command_probe/cosmos",
        root / "command_probe/cosmos_semantics",
        workspace / "artifacts/wam_language_gate",
        Path("/home/ali/projects/Efficient-WAM/experiments/robotwin_language_gate"),
        Path("/home/ali/projects/Efficient-WAM/outputs/robotwin_language_gate"),
        Path("/home/ali/projects/FastWAM/experiments/robotwin_language_gate"),
        Path("/home/ali/projects/FastWAM/outputs/robotwin_language_gate"),
        Path("/home/ali/projects/lerobot-lingbot/experiments/lingbot_language_gate"),
        Path("/home/ali/projects/lerobot-lingbot/outputs/lingbot_language_gate"),
    ]
    supporting_manifest = _write_raw_evidence_manifest(
        output / "supporting_evidence_manifest.csv",
        supporting_roots,
        scope="Calibration, excluded thermal/GPU-role/oracle runs, the excluded Cosmos GPU0 command probe, and separately labeled retrospective Efficient-WAM, FastWAM, LingBot-VA, Cosmos, and pi0.5 evidence.",
    )

    core_files = [
        root / "preregistration.json",
        root / "run_manifest.json",
        root / "hierarchy_amendment_001.json",
        root / "direct_language_scope_amendment_003.json",
        root / "metric_amendment_001.json",
        root / "observation_variation_amendment_001.json",
        root / "initial_state_schema_amendment_006.json",
        root / "thermal_control_amendment_001.json",
        root / "thermal_timing_amendment_002.json",
        root / "command_probe_plan.json",
        root / "direct_task_command_probe_plan.json",
        root / "command_probe_amendment_001.json",
        root / "semantic_future_calibration.json",
        root / "semantic_future_visualization_plan.json",
        root / "semantic_target_parser_amendment_004.json",
        root / "execution_geometry_amendment_005.json",
        root / "trajectory_visualization_plan.json",
        root / "semantic_confirmation_audit_plan.json",
        root / "semantic_confirmation_audit_amendment_002.json",
        root / "semantic_confirmation_audit.md",
        root / "checkpoint_provenance.json",
        root / "operational_snapshot_cosmos.json",
        root / "operational_snapshot_cosmos_confirmation.json",
        root / "operational_snapshot_pi05_confirmation.json",
        root / "cosmos_gpu_assignment_audit.json",
        root / "setup_exclusions/2026-08-02_cosmos_canonical_driver_check.md",
        root / "setup_exclusions/2026-08-02_cosmos_gpu0_thermal_restart.md",
        root / "setup_exclusions/2026-08-02_cosmos_confirmation_thermal_stop.md",
        root / "setup_exclusions/2026-08-02_cosmos_vague_pre_thermal_guard.md",
        root / "setup_exclusions/2026-08-02_oracle_scope_change.md",
        root / "setup_exclusions/cosmos_h5_oracle_scope_change_thermal.jsonl",
        workspace / "artifacts/wam_language_gate/summary.json",
        workspace / "docs/VLA_WAM_SHARED_BENCHMARK_V1.md",
        workspace / "docs/SEMANTIC_FUTURE_SCORER_V1.md",
        workspace / "docs/PAPER_PROTOCOL_ALIGNMENT.md",
        workspace / "docs/VLA_WAM_LOCAL_RUNBOOK.md",
        workspace / "docs/VLA_VS_WAM_STEERABILITY_STUDY.md",
        workspace / "docs/VLA_WAM_STEERABILITY_SOCIAL_COPY.md",
        workspace / "tools/compile_vla_wam_evidence.py",
        workspace / "tools/compile_vla_wam_study.py",
        workspace / "tools/compare_command_probe_hardware.py",
        workspace / "tools/run_fixed_observation_command_probe.py",
        workspace / "tools/score_cosmos_semantic_futures.py",
        workspace / "tools/render_semantic_future_examples.py",
        workspace / "tools/run_vla_wam_semantic_confirmation.sh",
        workspace / "tools/thermal_guard.py",
        workspace / "tools/render_trajectory_evidence.py",
        workspace / "tools/vla_wam_study_requirements.txt",
    ]
    compiled = {
        "schema_version": 1,
        "status": "complete_direct_language_grid_with_confirmatory_and_post_interim_stress_tiers",
        "integrity": {
            "expected_episode_count": EXPECTED_EPISODES,
            "original_confirmatory_episode_count": len(confirmatory_rows),
            "post_interim_direct_stress_episode_count": len(stress_rows),
            "oracle_episode_count": 0,
            "dynamic_prompt_episode_count": 0,
            "one_exact_physical_initial_state_fingerprint": True,
            "checkpoint_specific_full_recorder_fingerprint_count": len(
                closed["full_recorded_initial_state_fingerprint_counts"]
            ),
            "physical_initial_state_sha256": next(
                iter(closed["physical_initial_state_fingerprint_counts"])
            ),
            "command_probe_plan_sha256": plan_sha,
            "direct_task_command_probe_plan_sha256": direct_plan_sha,
            "semantic_calibration_sha256": calibration_sha,
            "raw_evidence_manifest": raw_manifest,
            "supporting_evidence_manifest": supporting_manifest,
            "cosmos_first_conditioning_images_audited": observation_variation[
                "first_conditioning_images"
            ],
            "cosmos_prediction_chunks_validated": observation_variation[
                "prediction_chunks_validated"
            ],
            "initial_physical_observations_audited": physical_variation["episodes"],
            "thermal_guard_logs_verified": thermal_control["batches"],
            "trajectory_episodes_rendered": trajectory_evidence[
                "rendered_episode_count"
            ],
            "trajectory_selection_plan_sha256": trajectory_evidence[
                "selection_plan_sha256"
            ],
        },
        "prospective": {
            "closed_loop": closed,
            "paired_diagnostics": _paired_diagnostics(closed["episodes"]),
            "cosmos_semantic_futures": semantic,
            "semantic_future_visualization": semantic_visualization,
            "cosmos_observation_variation": observation_variation,
            "initial_physical_variation": physical_variation,
            "command_probe": probes,
            "direct_task_command_probe": direct_probes,
            "trajectory_evidence": trajectory_evidence,
        },
        "retrospective": _load((workspace / args.retrospective).resolve() if not args.retrospective.is_absolute() else args.retrospective),
        "operational": {
            "cosmos_confirmation_snapshot": _load(
                root / "operational_snapshot_cosmos_confirmation.json"
            ),
            "pi05_confirmation_snapshot": _load(
                root / "operational_snapshot_pi05_confirmation.json"
            ),
            "cosmos_excluded_initial_snapshot": _load(
                root / "operational_snapshot_cosmos.json"
            ),
            "cosmos_gpu_assignment_audit": _load(
                root / "cosmos_gpu_assignment_audit.json"
            ),
            "checkpoint_provenance": _load(root / "checkpoint_provenance.json"),
            "thermal_control": thermal_control,
        },
        "provenance": {
            "files": {
                _relative(path, workspace): {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for path in core_files
            },
            "repositories": {
                "steerable": _repo_state(workspace),
                "RoboLab": _repo_state(Path("/home/ali/projects/RoboLab")),
                "cosmos-framework": _repo_state(Path("/home/ali/cosmos-framework")),
                "openpi-robolab": _repo_state(Path("/home/ali/openpi-robolab")),
            },
        },
        "claim_boundary": "These results compare one public VLA checkpoint with one public WAM checkpoint on one neutral-start spatial task pair. They do not establish a model-class ranking.",
    }

    _configure_plotting()
    _plot_static_success(closed, output)
    _plot_offsets(closed, output)
    _plot_direct_prompt_robustness(closed, output)
    _plot_observation_variation(observation_variation, output)
    _plot_semantic_quadrants(semantic, output)
    _plot_semantic_threshold_sensitivity(semantic, output)
    _plot_probe(probes, output)
    _plot_direct_task_probe(direct_probes, output)
    _plot_selected_probe_futures(root, probes, output)
    _dump(output / "compiled_evidence.json", compiled)
    _write_summary_csv(output / "semantic_future_groups.csv", semantic["groups"])
    _write_summary_csv(
        output / "semantic_future_episodes.csv", semantic["episode_summaries"]
    )
    _write_summary_csv(
        output / "semantic_threshold_sensitivity.csv",
        semantic["threshold_sensitivity"],
    )
    _write_summary_csv(
        output / "semantic_threshold_sensitivity_overall.csv",
        semantic["threshold_sensitivity_overall"],
    )
    _write_summary_csv(output / "paired_diagnostics.csv", compiled["prospective"]["paired_diagnostics"])
    (output / "EVIDENCE_INDEX.md").write_text(_evidence_markdown(compiled, root))
    print(
        f"complete: {closed['episode_count']} closed-loop episodes, "
        f"{semantic['total_chunks']} scored WAM chunks, "
        f"{EXPECTED_PROBE_CONDITIONS * 2} rich command probes and "
        f"{EXPECTED_DIRECT_TASK_PROBE_CONDITIONS * 2} direct task probes -> {output}"
    )


if __name__ == "__main__":
    main()
