#!/usr/bin/env python3
"""Compile the complete prospective VLA-versus-WAM study evidence.

This is the second-stage compiler. ``compile_vla_wam_evidence.py`` owns the
closed-loop episode extraction; this script verifies and joins that output with
the frozen Cosmos semantic-future scorer, the fixed-observation command probe,
operational evidence, and the explicitly retrospective WAM evidence tier.

The compiler fails closed unless every preregistered episode and every frozen
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


EXPECTED_EPISODES = 120
EXPECTED_STATIC_EPISODES = 80
EXPECTED_HIERARCHY_EPISODES = 40
EXPECTED_PROBE_CONDITIONS = 16
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
                    "wording": "canonical",
                    "controller": "static",
                    "open_loop_horizon": 5,
                },
                {
                    "model_id": model_id,
                    "wording": "canonical",
                    "controller": "predicate_oracle",
                    "open_loop_horizon": 5,
                },
                ("direction", "episode_seed"),
                f"{model_id}: horizon-5 static task versus predicate oracle",
            )
        )
    for direction in ("left", "right"):
        results.append(
            _paired_exact(
                rows,
                {
                    "model_id": "pi05_droid_vla",
                    "wording": "canonical",
                    "controller": "static",
                    "open_loop_horizon": 15,
                    "direction": direction,
                },
                {
                    "model_id": "cosmos3_edge_droid_wam",
                    "wording": "canonical",
                    "controller": "static",
                    "open_loop_horizon": 32,
                    "direction": direction,
                },
                ("episode_seed",),
                f"canonical {direction}: pi05 VLA versus Cosmos WAM at native horizons",
            )
        )
    return results


def _load_probe(path: Path, expected_model: str, plan_sha: str) -> dict[str, Any]:
    value = _load(path)
    if value["model"] != expected_model:
        raise RuntimeError(f"Probe model mismatch in {path}: {value['model']}")
    if value["plan_sha256"] != plan_sha:
        raise RuntimeError(f"Probe plan hash mismatch in {path}")
    if len(value["records"]) != EXPECTED_PROBE_CONDITIONS:
        raise RuntimeError(f"Expected 16 probe conditions in {path}")
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
        result[model] = {
            "manifest": manifest,
            "left_right_action_pairs": _pairwise_probe_action(directory),
        }
    if future_semantics["conditions"] != EXPECTED_PROBE_CONDITIONS:
        raise RuntimeError("Incomplete Cosmos command-probe semantic scoring")
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
            output.append(
                {
                    "cross_camera_threshold_m": threshold_m,
                    "wording": wording,
                    "direction": direction,
                    "chunks": len(pairs),
                    "certain_chunks": len(certain),
                    "coverage_fraction": len(certain) / len(pairs),
                    "imagined_requested_rate_among_certain": (
                        sum(bool(pair[0]) for pair in certain) / len(certain)
                        if certain
                        else None
                    ),
                    "imagination_execution_agreement_among_certain": (
                        sum(pair[0] == pair[1] for pair in certain) / len(certain)
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
    return {
        "input_summaries": input_summaries,
        "total_chunks": len(all_rows),
        "total_episodes": len({(row["task_dir"], row["episode_index"]) for row in all_rows}),
        "guardrail": "Chunk rates are descriptive and receive no binomial interval because replans within an episode are correlated.",
        "threshold_sensitivity": _semantic_threshold_sensitivity(summaries),
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
    conditions = [
        condition
        for condition in run_manifest["conditions"]
        if condition["model_id"] == "cosmos3_edge_droid_wam"
        and int(condition["open_loop_horizon"]) == 32
    ]
    if len(conditions) != 2:
        raise RuntimeError(f"Expected two static Cosmos conditions, got {len(conditions)}")
    for condition in conditions:
        root = Path(condition["output_root"])
        for direction, task_name in TASK_DIRS.items():
            task_dir = root / task_name
            manifest_path = task_dir / "predicted_chunks/manifest.jsonl"
            manifest_rows = [
                json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()
            ]
            first_rows = {
                int(row["episode_index"]): row
                for row in manifest_rows
                if int(row["replan_index"]) == 0
            }
            if set(first_rows) != set(range(10)):
                raise RuntimeError(f"Missing Cosmos first-chunk records in {manifest_path}")
            for episode_index in range(10):
                row = first_rows[episode_index]
                expected_seed = (6100 + episode_index) * 1000
                if row["requested_sampling_seed"] != expected_seed:
                    raise RuntimeError(f"Requested seed mismatch in {manifest_path}: {row}")
                if row["server_sampling_seed"] != expected_seed:
                    raise RuntimeError(f"Server seed mismatch in {manifest_path}: {row}")
                if row["conditioning_shape"] != [540, 640, 3]:
                    raise RuntimeError(f"Conditioning shape mismatch in {manifest_path}: {row}")
                if row["action_shape"] != [32, 8] or row["future_shape"] != [33, 528, 640, 3]:
                    raise RuntimeError(f"Cosmos output shape mismatch in {manifest_path}: {row}")
                image_path = (
                    task_dir
                    / "predicted_chunks"
                    / f"episode_{episode_index:03d}"
                    / "chunk_000"
                    / "conditioning.png"
                )
                if _sha256(image_path) != row["conditioning_sha256"]:
                    raise RuntimeError(f"Conditioning hash mismatch: {image_path}")
                bgr = cv2.imread(str(image_path))
                if bgr is None:
                    raise FileNotFoundError(image_path)
                images[(condition["id"], direction, episode_index)] = bgr

    for condition in conditions:
        for direction in TASK_DIRS:
            for left_index in range(10):
                for right_index in range(left_index + 1, 10):
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
        for episode_index in range(10):
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
    for direction in TASK_DIRS:
        for episode_index in range(10):
            records.append(
                {
                    "comparison": "matched_canonical_short",
                    "condition_id": "cosmos_canonical_vs_short",
                    "direction": direction,
                    "left_episode_index": episode_index,
                    "right_episode_index": episode_index,
                    **_image_difference(
                        images[(by_wording["canonical"], direction, episode_index)],
                        images[(by_wording["short_paraphrase"], direction, episode_index)],
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
        "all_server_seeds_match_frozen_schedule": True,
        "interpretation": "Exact simulator state does not yield byte-identical realtime-rendered observations. These differences contaminate closed-loop first-action prompt/noise contrasts but not the exact fixed-observation command probe.",
        "summaries": summaries,
        "pairs": records,
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
    slots = [
        ("canonical", "left"),
        ("canonical", "right"),
        ("short_paraphrase", "left"),
        ("short_paraphrase", "right"),
    ]
    labels = ["canonical\nleft", "canonical\nright", "short\nleft", "short\nright"]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
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
        errors = np.asarray(
            [[value - interval[0] for value, interval in zip(values, intervals)],
             [interval[1] - value for value, interval in zip(values, intervals)]]
        )
        positions = x + (model_index - 0.5) * width
        bars = ax.bar(
            positions,
            values,
            width,
            color=MODEL_COLORS[model_id],
            label=MODEL_LABELS[model_id],
            yerr=errors,
            capsize=3,
        )
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.0%}", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Requested-goal success rate")
    ax.set_title("Matched neutral-start closed-loop success (10 seeds per bar)")
    ax.legend(frameon=False, ncol=2, loc="upper center")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "static_success_with_intervals.png", bbox_inches="tight")
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
        for wording in ("canonical", "short_paraphrase"):
            for direction in ("left", "right"):
                values = [
                    row["requested_signed_final_offset_m"]
                    for row in episodes
                    if row["model_id"] == model_id
                    and row["wording"] == wording
                    and row["direction"] == direction
                ]
                slots.append(
                    f"{MODEL_LABELS[model_id].split(' ')[0]}\n{wording.replace('_paraphrase', '')}\n{direction}"
                )
                data.append(values)
                colors.append(MODEL_COLORS[model_id])
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
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
    fig.savefig(output / "static_requested_side_offsets.png", bbox_inches="tight")
    plt.close(fig)


def _plot_hierarchy(closed: dict[str, Any], output: Path) -> None:
    groups = [
        row
        for row in closed["group_summaries"]
        if row["open_loop_horizon"] == 5
        and row["wording"] == "canonical"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.1), sharey=True)
    for axis, model_id in zip(axes, MODEL_LABELS):
        selected = {
            (row["controller"], row["direction"]): row
            for row in groups
            if row["model_id"] == model_id
        }
        x = np.arange(2)
        width = 0.34
        for controller_index, controller in enumerate(("static", "predicate_oracle")):
            values = [selected[(controller, direction)]["success_rate"] for direction in ("left", "right")]
            intervals = [
                selected[(controller, direction)]["success_beta11_interval_95"]
                for direction in ("left", "right")
            ]
            errors = np.asarray(
                [
                    [value - interval[0] for value, interval in zip(values, intervals)],
                    [interval[1] - value for value, interval in zip(values, intervals)],
                ]
            )
            bars = axis.bar(
                x + (controller_index - 0.5) * width,
                values,
                width,
                label="Static task" if controller == "static" else "Predicate oracle",
                color="#7b7b7b" if controller == "static" else MODEL_COLORS[model_id],
                yerr=errors,
                capsize=3,
            )
            for bar, value in zip(bars, values):
                axis.text(bar.get_x() + bar.get_width() / 2, value + 0.035, f"{value:.0%}", ha="center")
        axis.set_xticks(x, ["left", "right"])
        axis.set_ylim(0, 1.14)
        axis.set_title(MODEL_LABELS[model_id])
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Success rate (5 matched seeds)")
    axes[1].legend(frameon=False, loc="upper center")
    fig.suptitle("Does perfect state-aware command selection help at horizon 5?")
    fig.tight_layout()
    fig.savefig(output / "hierarchy_static_vs_oracle.png", bbox_inches="tight")
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
    names = [f"{row['wording'].replace('_paraphrase', '')}\n{row['direction']}" for row in groups]
    fig, ax = plt.subplots(figsize=(8.7, 4.4))
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


def _plot_semantic_threshold_sensitivity(semantic: dict[str, Any], output: Path) -> None:
    rows = semantic["threshold_sensitivity"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True, sharey=True)
    groups = sorted({(row["wording"], row["direction"]) for row in rows})
    colors = {"left": "#6a51a3", "right": "#d95f0e"}
    styles = {"canonical": "-", "short_paraphrase": "--"}
    for wording, direction in groups:
        selected = sorted(
            [row for row in rows if row["wording"] == wording and row["direction"] == direction],
            key=lambda row: row["cross_camera_threshold_m"],
        )
        label = f"{wording.replace('_paraphrase', '')} {direction}"
        x = [row["cross_camera_threshold_m"] for row in selected]
        axes[0].plot(
            x,
            [row["coverage_fraction"] for row in selected],
            marker="o",
            color=colors[direction],
            linestyle=styles[wording],
            label=label,
        )
        axes[1].plot(
            x,
            [
                row["imagination_execution_agreement_among_certain"]
                if row["imagination_execution_agreement_among_certain"] is not None
                else np.nan
                for row in selected
            ],
            marker="o",
            color=colors[direction],
            linestyle=styles[wording],
            label=label,
        )
    for axis, title in zip(axes, ("Coverage", "Agreement among scored chunks")):
        axis.set_title(title)
        axis.set_xlabel("Cross-camera disagreement threshold (m)")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Frozen semantic-future scorer sensitivity (labels unchanged at 0.20 m)")
    fig.tight_layout()
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
    lines = [
        "# VLA-WAM study evidence index",
        "",
        "Status: complete prospective grid plus a separately labeled retrospective WAM tier.",
        "",
        "## Integrity checks",
        "",
        f"- Closed-loop episodes: **{closed['episode_count']}/{EXPECTED_EPISODES}**.",
        f"- Static episodes: **{compiled['integrity']['static_episode_count']}/{EXPECTED_STATIC_EPISODES}**.",
        f"- Hierarchy episodes: **{compiled['integrity']['hierarchy_episode_count']}/{EXPECTED_HIERARCHY_EPISODES}**.",
        f"- Shared initial-state fingerprints: **{len(closed['initial_state_fingerprint_counts'])}**.",
        f"- Fixed-observation probe conditions: **{len(probes['pi05']['manifest']['records'])} per model**.",
        f"- Cosmos semantically scored confirmation chunks: **{semantic['total_chunks']}** across **{semantic['total_episodes']} episodes**.",
        "- Calibration, command-probe, and run-manifest hashes were verified by the compilers.",
        "",
        "## Prospective evidence",
        "",
        "| Evidence | Artifact | Purpose |",
        "| --- | --- | --- |",
        "| Frozen design | `../preregistration.json` | Questions, fixed grid, primary/secondary metrics, stopping rule |",
        "| Hierarchy amendment | `../hierarchy_amendment_001.json` | Correct 40-episode matched static/oracle arithmetic |",
        "| Metric amendment | `../metric_amendment_001.json` | Exact paper-style progression after primary-source verification |",
        "| Observation amendment | `../observation_variation_amendment_001.json` | Downgrades closed-loop action contrast after measured renderer variation |",
        "| Grounded probe plan | `../command_probe_plan.json` | Hash-pinned observation, six command styles, controls, seed |",
        "| Closed-loop episode table | `episodes.csv` | One row per preregistered rollout |",
        "| Closed-loop summary | `closed_loop_summary.json` | Success, progression, offsets, timing, contrasts |",
        "| Cosmos future semantics | `compiled_evidence.json` | Prompt-blind imagined/executed quadrants and coverage |",
        "| Renderer variation audit | `cosmos_observation_variation.csv` | First-conditioning-image differences within and across static conditions |",
        "| Human semantic audit | `../semantic_confirmation_audit_plan.json` and `../semantic_confirmation_audit.md` | Outcome-independent sheet sample and completed visual review |",
        "| Command probes | `compiled_evidence.json` | Exact repeat, command sensitivity, semantic futures |",
        "| GPU assignment audit | `../cosmos_gpu_assignment_audit.json` | Quantifies why cross-card Cosmos output was excluded |",
        "| Confirmation resource snapshot | `../operational_snapshot_cosmos_confirmation.json` | Temperatures, memory, utilization, and physical GPU roles during a valid request |",
        "| Raw file hash ledger | `raw_evidence_manifest.csv` | Byte size and SHA-256 for every prospective raw/derived evidence file |",
        "| Supporting hash ledger | `supporting_evidence_manifest.csv` | Calibration, exclusions, and separately labeled retrospective raw/derived evidence |",
        "| Setup exclusion | `../setup_exclusions/2026-08-02_cosmos_canonical_driver_check.md` | Failed startup with zero requests, excluded transparently |",
        "| Thermal exclusion | `../setup_exclusions/2026-08-02_cosmos_gpu0_thermal_restart.md` | Interrupted and cross-GPU batches preserved outside estimates |",
        "",
        "## Figures",
        "",
        "- `static_success_with_intervals.png`: primary binary success with Beta(1,1) 95% intervals.",
        "- `static_requested_side_offsets.png`: endpoint directionality, including failures.",
        "- `hierarchy_static_vs_oracle.png`: matched five-step static versus predicate-oracle control.",
        "- `cosmos_imagination_execution_quadrants.png`: WAM-only semantic future/action agreement.",
        "- `semantic_threshold_sensitivity.png`: scorer coverage/agreement at 0.10, 0.15, and frozen 0.20 m reliability thresholds.",
        "- `command_probe_action_sensitivity.png`: same-observation six-style prompt response.",
        "- `command_probe_selected_futures.png`: selected Cosmos future strips with frozen prompt-blind relation labels.",
        "",
        "## Retrospective evidence tier",
        "",
        "Efficient-WAM, FastWAM, LingBot-VA, and the earlier π0.5/Cosmos pilots remain in `../../wam_language_gate/`. They inform model selection and failure analysis, but they are not pooled into the prospective confidence intervals.",
        "",
        "## Statistical guardrails",
        "",
        "- A Beta(1,1) interval accompanies each success proportion.",
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
    closed_path = args.closed_loop or (output / "closed_loop_summary.json")
    closed = _load(closed_path)
    if not closed.get("complete") or closed["episode_count"] != EXPECTED_EPISODES:
        raise RuntimeError(
            f"Closed-loop grid incomplete: complete={closed.get('complete')} episodes={closed.get('episode_count')}"
        )
    static_rows = [row for row in closed["episodes"] if row["open_loop_horizon"] in {15, 32}]
    hierarchy_rows = [row for row in closed["episodes"] if row["open_loop_horizon"] == 5]
    if len(static_rows) != EXPECTED_STATIC_EPISODES or len(hierarchy_rows) != EXPECTED_HIERARCHY_EPISODES:
        raise RuntimeError("Static/hierarchy episode accounting does not match the frozen grid")
    if len(closed["initial_state_fingerprint_counts"]) != 1:
        raise RuntimeError("Closed-loop inputs do not share one exact initial-state fingerprint")

    plan_path = root / "command_probe_plan.json"
    plan_sha = _sha256(plan_path)
    cosmos_probe_semantic_path = (
        root / "command_probe/cosmos_gpu1_semantics/semantic_future_summary.json"
    )
    cosmos_probe_semantic = _load(cosmos_probe_semantic_path)
    probes = _probe_summary(root, plan_sha, cosmos_probe_semantic)
    calibration_sha = _sha256(root / "semantic_future_calibration.json")
    semantic_inputs = []
    for wording, path in (
        ("canonical", root / "semantic_confirmation/cosmos_canonical/semantic_quadrants_summary.json"),
        ("short_paraphrase", root / "semantic_confirmation/cosmos_vague/semantic_quadrants_summary.json"),
    ):
        summary = _load(path)
        if summary["calibration_sha256"] != calibration_sha:
            raise RuntimeError(f"Semantic calibration hash mismatch in {path}")
        semantic_inputs.append((wording, path.parent, summary))
    semantic = _semantic_aggregate(semantic_inputs)

    run_manifest = _load(root / "run_manifest.json")
    observation_variation = _cosmos_observation_variation(run_manifest)
    raw_roots = [Path(condition["output_root"]) for condition in run_manifest["conditions"]]
    raw_roots.extend(
        [
            root / "command_probe/pi05",
            root / "command_probe/cosmos_gpu1",
            root / "command_probe/cosmos_gpu1_semantics",
            root / "semantic_confirmation/cosmos_canonical",
            root / "semantic_confirmation/cosmos_vague",
        ]
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(
        output / "cosmos_observation_variation.csv",
        observation_variation["pairs"],
    )
    raw_manifest = _write_raw_evidence_manifest(
        output / "raw_evidence_manifest.csv",
        raw_roots,
        scope="All supported data, image, video, and documentation files under the eight definitive run roots plus prospective command-probe and semantic-scoring roots.",
    )
    supporting_roots = [
        Path("/home/ali/projects/RoboLab/output/v1_calibration_cosmos_left_5100"),
        Path("/home/ali/projects/RoboLab/output/v1_calibration_cosmos_right_5100"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_canonical_original_hot_gpu0"),
        Path("/home/ali/projects/RoboLab/output/v1_cosmos_vague_interrupted_hot_gpu0"),
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
        scope="Calibration, excluded thermal/GPU-role runs, the excluded Cosmos GPU0 command probe, and separately labeled retrospective Efficient-WAM, Fast-WAM, LingBot-VA, Cosmos, and pi0.5 evidence.",
    )

    core_files = [
        root / "preregistration.json",
        root / "run_manifest.json",
        root / "hierarchy_amendment_001.json",
        root / "metric_amendment_001.json",
        root / "observation_variation_amendment_001.json",
        root / "command_probe_plan.json",
        root / "command_probe_amendment_001.json",
        root / "semantic_future_calibration.json",
        root / "semantic_confirmation_audit_plan.json",
        root / "semantic_confirmation_audit.md",
        root / "checkpoint_provenance.json",
        root / "operational_snapshot_cosmos.json",
        root / "operational_snapshot_cosmos_confirmation.json",
        root / "cosmos_gpu_assignment_audit.json",
        root / "setup_exclusions/2026-08-02_cosmos_canonical_driver_check.md",
        root / "setup_exclusions/2026-08-02_cosmos_gpu0_thermal_restart.md",
        workspace / "artifacts/wam_language_gate/summary.json",
        workspace / "docs/VLA_WAM_SHARED_BENCHMARK_V1.md",
        workspace / "docs/SEMANTIC_FUTURE_SCORER_V1.md",
        workspace / "docs/PAPER_PROTOCOL_ALIGNMENT.md",
        workspace / "docs/VLA_WAM_LOCAL_RUNBOOK.md",
        workspace / "docs/VLA_VS_WAM_STEERABILITY_STUDY.md",
        workspace / "tools/compile_vla_wam_evidence.py",
        workspace / "tools/compile_vla_wam_study.py",
        workspace / "tools/compare_command_probe_hardware.py",
        workspace / "tools/run_fixed_observation_command_probe.py",
        workspace / "tools/score_cosmos_semantic_futures.py",
        workspace / "tools/vla_wam_study_requirements.txt",
    ]
    compiled = {
        "schema_version": 1,
        "status": "complete_prospective_grid_with_separate_retrospective_tier",
        "integrity": {
            "expected_episode_count": EXPECTED_EPISODES,
            "static_episode_count": len(static_rows),
            "hierarchy_episode_count": len(hierarchy_rows),
            "one_exact_initial_state_fingerprint": True,
            "command_probe_plan_sha256": plan_sha,
            "semantic_calibration_sha256": calibration_sha,
            "raw_evidence_manifest": raw_manifest,
            "supporting_evidence_manifest": supporting_manifest,
            "cosmos_first_conditioning_images_audited": observation_variation[
                "first_conditioning_images"
            ],
        },
        "prospective": {
            "closed_loop": closed,
            "paired_diagnostics": _paired_diagnostics(closed["episodes"]),
            "cosmos_semantic_futures": semantic,
            "cosmos_observation_variation": observation_variation,
            "command_probe": probes,
        },
        "retrospective": _load((workspace / args.retrospective).resolve() if not args.retrospective.is_absolute() else args.retrospective),
        "operational": {
            "cosmos_confirmation_snapshot": _load(
                root / "operational_snapshot_cosmos_confirmation.json"
            ),
            "cosmos_excluded_initial_snapshot": _load(
                root / "operational_snapshot_cosmos.json"
            ),
            "cosmos_gpu_assignment_audit": _load(
                root / "cosmos_gpu_assignment_audit.json"
            ),
            "checkpoint_provenance": _load(root / "checkpoint_provenance.json"),
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
    _plot_hierarchy(closed, output)
    _plot_semantic_quadrants(semantic, output)
    _plot_semantic_threshold_sensitivity(semantic, output)
    _plot_probe(probes, output)
    _plot_selected_probe_futures(root, probes, output)
    _dump(output / "compiled_evidence.json", compiled)
    _write_summary_csv(output / "semantic_future_groups.csv", semantic["groups"])
    _write_summary_csv(
        output / "semantic_threshold_sensitivity.csv",
        semantic["threshold_sensitivity"],
    )
    _write_summary_csv(output / "paired_diagnostics.csv", compiled["prospective"]["paired_diagnostics"])
    (output / "EVIDENCE_INDEX.md").write_text(_evidence_markdown(compiled, root))
    print(
        f"complete: {closed['episode_count']} closed-loop episodes, "
        f"{semantic['total_chunks']} scored WAM chunks, "
        f"{EXPECTED_PROBE_CONDITIONS * 2} command probes -> {output}"
    )


if __name__ == "__main__":
    main()
