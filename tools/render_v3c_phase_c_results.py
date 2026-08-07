#!/usr/bin/env python3
"""Render compact publication figures for the complete V3-C001 cohort."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tools.compile_v3c_phase_c_results import (
    FAILURE_CLASSES,
    MODELS,
    PROMPT_FAMILIES,
    RELATIONS,
    canonical_json_bytes,
)


MODEL_LABELS = {
    "groot_n17_droid_vla": "GR00T N1.7",
    "cosmos3_edge_policy_droid": "Cosmos3 Edge",
    "cosmos3_nano_policy_droid": "Cosmos3 Nano",
}
FAMILY_LABELS = {
    "direct_command": "Direct instruction",
    "short_command": "Shortened instruction",
    "goal_as_outcome": "Goal statement",
    "desired_plus_negated_opposite": "Contrastive instruction",
}
FAMILY_SHORT_LABELS = {
    "direct_command": "Direct",
    "short_command": "Shortened",
    "goal_as_outcome": "Goal",
    "desired_plus_negated_opposite": "Contrastive",
}
COLORS = {
    "left": "#D85A30",
    "right": "#2367B1",
    "shift": "#65756A",
    "median": "#147D64",
    "correct": "#238B68",
    "pick_failed": "#9A948A",
    "transport_failed": "#E39B38",
    "wrong_side": "#C84545",
    "release_failed": "#7255A6",
}
FAILURE_LABELS = {
    "correct": "Correct",
    "pick_failed": "Pick failed",
    "transport_failed": "Transport failed",
    "wrong_side": "Wrong side",
    "release_failed": "Release failed",
}
BACKGROUND = "#F5F1E8"
INK = "#17232B"
GRID = "#D7D0C3"


class RenderError(ValueError):
    """Raised when compact model evidence is incomplete."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inputs(summaries: list[Path], episodes: list[Path]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary_by_model = {}
    for path in summaries:
        value = json.loads(path.read_text())
        model_id = value.get("model_id")
        if value.get("status") != "complete_20_seed_160_behavioral_episode_result" or model_id not in MODELS:
            raise RenderError(f"not a complete Phase-C model summary: {path}")
        if model_id in summary_by_model:
            raise RenderError(f"duplicate model summary: {model_id}")
        summary_by_model[model_id] = value
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    for path in episodes:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 160 or len({row.get("model_id") for row in rows}) != 1:
            raise RenderError(f"episode file must contain one complete 160-row model: {path}")
        model_id = rows[0]["model_id"]
        if model_id in rows_by_model:
            raise RenderError(f"duplicate model episode file: {model_id}")
        rows_by_model[model_id] = rows
    if set(summary_by_model) != set(MODELS) or set(rows_by_model) != set(MODELS):
        raise RenderError("all three preregistered Phase-C models are required")
    return summary_by_model, rows_by_model


def _style_axis(axis: plt.Axes) -> None:
    axis.set_facecolor(BACKGROUND)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=INK, labelsize=9, length=0)
    axis.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7, zorder=0)


def render_scope_figure(summary_by_model: dict[str, Any], output: Path) -> None:
    model_order = tuple(MODELS)
    all_shifts = [
        shift
        for model_id in model_order
        for family in PROMPT_FAMILIES
        for shift in summary_by_model[model_id]["paired_diagnostics_by_prompt_family"][family]["right_minus_left_endpoint_shift_m"]
    ]
    shift_limit = max(0.12, max(abs(value) for value in all_shifts) * 1.1)
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(14.2, 12.0),
        gridspec_kw={"width_ratios": [1.0, 1.15], "hspace": 0.56, "wspace": 0.23},
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(top=0.77, bottom=0.10, left=0.16, right=0.97)
    figure.suptitle(
        "Phrasing modulates directional competence;\nendpoint response is a separate diagnostic",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=20,
        fontfamily="DejaVu Serif",
        fontweight="semibold",
        color=INK,
    )
    figure.text(
        0.06,
        0.915,
        "Same 20 seeds in every cell. Only the static instruction changes; LEFT and RIGHT always begin from an identical reset.",
        ha="left",
        fontsize=11.5,
        color="#4C585D",
    )
    prompt_text = (
        'DIRECT  “Put the Rubik\'s cube to the {left|right} of the bowl.”   ·   '
        'SHORTENED  “Put the cube {left|right} of the bowl.”\n'
        'GOAL  “The Rubik\'s cube should end up to the {left|right} of the bowl.”   ·   '
        'CONTRASTIVE  “Put the Rubik\'s cube to the {left|right} of the bowl, not to the {right|left} of the bowl.”'
    )
    figure.text(
        0.06,
        0.875,
        prompt_text,
        ha="left",
        va="top",
        fontsize=9.3,
        linespacing=1.5,
        color=INK,
        bbox={"boxstyle": "round,pad=0.65", "facecolor": "#FBF9F4", "edgecolor": GRID},
    )
    y = np.arange(len(PROMPT_FAMILIES))
    offsets = {"left": -0.12, "right": 0.12}
    for row_index, model_id in enumerate(model_order):
        summary = summary_by_model[model_id]
        success_axis, shift_axis = axes[row_index]
        _style_axis(success_axis)
        _style_axis(shift_axis)
        success_axis.set_title(
            MODEL_LABELS[model_id], loc="left", fontsize=14, fontweight="bold", color=INK, pad=10
        )
        for relation in RELATIONS:
            values = []
            lower = []
            upper = []
            for family in PROMPT_FAMILIES:
                result = summary["success_by_condition"][f"{family}:{relation}"]
                values.append(result["success_rate"])
                lower.append(result["success_rate"] - result["wilson_95"][0])
                upper.append(result["wilson_95"][1] - result["success_rate"])
            success_axis.errorbar(
                values,
                y + offsets[relation],
                xerr=np.asarray([lower, upper]),
                fmt="o",
                color=COLORS[relation],
                markersize=6,
                capsize=3,
                linewidth=1.5,
                label=f"Prompt requests {relation.upper()}",
                zorder=3,
            )
            for family_index, family in enumerate(PROMPT_FAMILIES):
                result = summary["success_by_condition"][f"{family}:{relation}"]
                x = min(0.94, result["success_rate"] + 0.045)
                ha = "left"
                if result["success_rate"] > 0.9:
                    x = result["success_rate"] - 0.045
                    ha = "right"
                success_axis.text(
                    x,
                    family_index + offsets[relation],
                    f'{result["successes"]}/20',
                    ha=ha,
                    va="center",
                    fontsize=8.2,
                    color=COLORS[relation],
                    fontweight="bold",
                )
        success_axis.set_xlim(-0.02, 1.02)
        success_axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
        success_axis.set_yticks(y, [FAMILY_LABELS[family] for family in PROMPT_FAMILIES])
        success_axis.invert_yaxis()
        success_axis.set_xlabel("Requested-task success (%) · Wilson 95% interval", color=INK, fontsize=9.5)
        for family_index, family in enumerate(PROMPT_FAMILIES):
            shifts = summary["paired_diagnostics_by_prompt_family"][family]["right_minus_left_endpoint_shift_m"]
            jitter = np.linspace(-0.13, 0.13, len(shifts))
            shift_axis.scatter(
                shifts,
                family_index + jitter,
                s=18,
                color=COLORS["shift"],
                alpha=0.62,
                edgecolors="none",
                zorder=2,
            )
            paired = summary["paired_diagnostics_by_prompt_family"][family]
            shift_axis.scatter(
                [paired["median_right_minus_left_endpoint_shift_m"]],
                [family_index],
                s=55,
                marker="D",
                color=COLORS["median"],
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
            shift_axis.text(
                0.98,
                family_index,
                f'{paired["endpoint_ordering_aligned"]}/20 aligned',
                ha="right",
                va="center",
                fontsize=8.2,
                color=INK,
                transform=shift_axis.get_yaxis_transform(),
                bbox={"facecolor": BACKGROUND, "edgecolor": "none", "pad": 1.0, "alpha": 0.9},
                zorder=5,
            )
        shift_axis.axvline(0, color=INK, linewidth=1.1, zorder=1)
        shift_axis.set_xlim(-shift_limit, shift_limit)
        shift_axis.set_yticks(y, [])
        shift_axis.invert_yaxis()
        shift_axis.set_xlabel(
            "Matched endpoint shift: RIGHT prompt − LEFT prompt (m; negative = requested ordering)",
            color=INK,
            fontsize=9.5,
        )
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], marker="o", color=COLORS["left"], label="Prompt requests LEFT", linestyle="none"),
            Line2D([0], [0], marker="o", color=COLORS["right"], label="Prompt requests RIGHT", linestyle="none"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.18),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    axes[0, 1].legend(
        handles=[
            Line2D([0], [0], marker="o", color=COLORS["shift"], label="One matched seed", linestyle="none"),
            Line2D([0], [0], marker="D", color=COLORS["median"], label="Median", linestyle="none"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.18),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    figure.text(
        0.06,
        0.035,
        "Success requires pickup, transport into the requested 45° cone, and detached release. Negative RIGHT-minus-LEFT endpoint shift follows the requested LEFT→RIGHT ordering;\n"
        "it measures redirection, not completion. Phase C is exploratory; DROID and RoboTwin are never pooled.",
        fontsize=9.3,
        color="#4C585D",
        ha="left",
    )
    figure.savefig(output, dpi=220, facecolor=BACKGROUND)
    plt.close(figure)


def render_failure_figure(rows_by_model: dict[str, list[dict[str, Any]]], output: Path) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14.2, 9.8), facecolor=BACKGROUND, sharex=True)
    figure.subplots_adjust(top=0.84, bottom=0.15, left=0.17, right=0.97, hspace=0.48)
    figure.suptitle(
        "Failure composition changes with checkpoint, wording,\nand requested direction",
        x=0.06,
        y=0.985,
        ha="left",
        fontsize=20,
        fontfamily="DejaVu Serif",
        fontweight="semibold",
        color=INK,
    )
    figure.text(
        0.06,
        0.905,
        "Each bar is exactly 20 matched-seed episodes. Behavioral failures remain in the denominator; infrastructure attempts do not.",
        ha="left",
        fontsize=11.2,
        color="#4C585D",
    )
    bar_labels = [
        f"{FAMILY_SHORT_LABELS[family]}\n{relation.upper()}"
        for family in PROMPT_FAMILIES
        for relation in RELATIONS
    ]
    for axis, model_id in zip(axes, MODELS, strict=True):
        axis.set_facecolor(BACKGROUND)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=INK, labelsize=8.7, length=0)
        axis.set_title(MODEL_LABELS[model_id], loc="left", fontsize=14, fontweight="bold", color=INK)
        bottoms = np.zeros(8)
        rows = rows_by_model[model_id]
        for failure in FAILURE_CLASSES:
            counts = []
            for family in PROMPT_FAMILIES:
                for relation in RELATIONS:
                    counts.append(sum(
                        row["prompt_family"] == family
                        and row["relation"] == relation
                        and row["failure_taxonomy"] == failure
                        for row in rows
                    ))
            fractions = np.asarray(counts) / 20.0
            bars = axis.bar(
                np.arange(8),
                fractions,
                bottom=bottoms,
                width=0.72,
                color=COLORS[failure],
                edgecolor=BACKGROUND,
                linewidth=0.8,
                label=FAILURE_LABELS[failure],
            )
            for bar, count, base, fraction in zip(bars, counts, bottoms, fractions, strict=True):
                if count >= 2:
                    axis.text(
                        bar.get_x() + bar.get_width() / 2,
                        base + fraction / 2,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if failure != "transport_failed" else INK,
                        fontweight="bold",
                    )
            bottoms += fractions
        axis.set_ylim(0, 1)
        axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
        axis.set_ylabel("Episodes (%)", color=INK)
        axis.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.7, zorder=0)
    axes[-1].set_xticks(np.arange(8), bar_labels)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.89), ncol=5, frameon=False, fontsize=9.5)
    figure.text(
        0.06,
        0.045,
        "Taxonomy precedence: correct; pick failed; wrong side; release failed; otherwise transport failed. Counts inside segments are raw episodes out of 20.",
        fontsize=9.3,
        color="#4C585D",
        ha="left",
    )
    figure.savefig(output, dpi=220, facecolor=BACKGROUND)
    plt.close(figure)


def render(*, summaries: list[Path], episodes: list[Path], output_dir: Path) -> dict[str, Any]:
    summary_by_model, rows_by_model = _load_inputs(summaries, episodes)
    output_dir.mkdir(parents=True, exist_ok=False)
    outputs = []
    for suffix in ("png", "svg"):
        scope = output_dir / f"figure7_phase_c_phrasing_direction.{suffix}"
        failure = output_dir / f"figure7_phase_c_failure_taxonomy.{suffix}"
        render_scope_figure(summary_by_model, scope)
        render_failure_figure(rows_by_model, failure)
        outputs.extend([scope, failure])
    manifest = {
        "schema_version": "vla-wam-shared-v3c-figure-manifest-v1",
        "experiment_id": "V3-C001",
        "input_summaries": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in summaries
        ],
        "input_episode_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": 160}
            for path in episodes
        ],
        "outputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
        ],
        "interpretation_boundary": "Success and endpoint redirection are shown as separate diagnostics. Phase C is exploratory; DROID and RoboTwin are never pooled.",
    }
    manifest_path = output_dir / "phase_c_figure_manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--episodes", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = render(summaries=args.summary, episodes=args.episodes, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
