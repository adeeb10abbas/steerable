#!/usr/bin/env python3
"""Render every publication figure for the VLA-versus-WAM steerability study.

The renderer reads only ``final_evidence/compiled_evidence.json``. It performs no
inference, opens no checkpoint, and derives no new statistic: every number drawn
here already exists in the compiled evidence produced by
``compile_vla_wam_study.py``. Separating rendering from compilation means a
figure can be restyled without re-running the study, and a restyle can never
silently change a reported value.

Usage
-----
    python tools/render_study_figures.py \
        --evidence artifacts/vla_wam_shared_v1/final_evidence/compiled_evidence.json \
        --output   artifacts/vla_wam_shared_v1/final_evidence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

import figure_style as fs


ANALYSED = {"controller": "static", "horizons": {15, 32}}


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------


def _analysed(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["controller"] == ANALYSED["controller"]
        and row["open_loop_horizon"] in ANALYSED["horizons"]
    ]


def _group_index(closed: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["model_id"], row["wording"], row["direction"]): row
        for row in _analysed(closed["group_summaries"])
    }


def _episode_index(closed: dict[str, Any]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in _analysed(closed["episodes"]):
        index.setdefault((row["model_id"], row["wording"], row["direction"]), []).append(row)
    return index


# --------------------------------------------------------------------------
# Figure 1 — closed-loop success, paired by requested direction
# --------------------------------------------------------------------------


def figure_success(closed: dict[str, Any], output: Path) -> None:
    groups = _group_index(closed)
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.0), sharey=True, sharex=True)
    fig.subplots_adjust(wspace=0.10)

    for ax, model_id in zip(axes, fs.MODEL_IDS):
        fs.clean(ax, grid_axis="x", keep=())
        base = fs.MODEL_COLORS[model_id]
        light = fs.MODEL_LIGHT[model_id]

        for index, wording in enumerate(fs.WORDINGS):
            row_centre = len(fs.WORDINGS) - 1 - index
            offsets = {"left": row_centre + 0.20, "right": row_centre - 0.20}
            rates = {}
            for direction in fs.DIRECTIONS:
                row = groups[(model_id, wording, direction)]
                y = offsets[direction]
                low, high = row["success_beta11_interval_95"]
                rate = row["success_rate"]
                rates[direction] = rate
                colour = light if direction == "left" else base

                ax.plot(
                    [low, high],
                    [y, y],
                    color=colour,
                    linewidth=3.4,
                    alpha=0.32,
                    solid_capstyle="round",
                    zorder=2,
                )
                ax.plot(
                    [rate],
                    [y],
                    marker=fs.DIRECTION_MARKERS[direction],
                    markersize=10.5 if direction == "left" else 9.0,
                    markerfacecolor=colour,
                    markeredgecolor=fs.PAPER,
                    markeredgewidth=1.6,
                    linestyle="none",
                    zorder=4,
                )
                ax.text(
                    -0.035,
                    y,
                    direction.upper(),
                    ha="right",
                    va="center",
                    fontsize=8.6,
                    fontweight="bold",
                    color=colour if direction == "right" else fs.MUTED,
                )
                ax.text(
                    1.13,
                    y,
                    f"{row['successes']}/{row['episodes']}",
                    ha="right",
                    va="center",
                    fontsize=9.8,
                    fontweight="bold",
                    color=fs.INK,
                )

            gap = rates["right"] - rates["left"]
            span = sorted((rates["left"], rates["right"]))
            ax.fill_betweenx(
                [row_centre - 0.20, row_centre + 0.20],
                span[0],
                span[1],
                color=base,
                alpha=0.10,
                zorder=1,
            )
            if abs(gap) >= 0.05:
                ax.text(
                    (span[0] + span[1]) / 2,
                    row_centre,
                    f"{gap:+.0%}",
                    ha="center",
                    va="center",
                    fontsize=9.0,
                    fontweight="bold",
                    color=base,
                    bbox=dict(boxstyle="round,pad=0.22", facecolor=fs.PAPER, edgecolor="none"),
                )

        for boundary in range(1, len(fs.WORDINGS)):
            ax.axhline(boundary - 0.5, color=fs.GRID, linewidth=0.9, zorder=0)

        ax.set_title(
            f"{fs.MODEL_LABELS[model_id]}  ·  {fs.MODEL_CLASS[model_id]}",
            color=base,
            pad=14,
        )
        ax.axvline(1.045, color=fs.GRID, linewidth=1.0, zorder=0)
        ax.set_xlim(-0.17, 1.15)
        ax.set_ylim(-0.72, len(fs.WORDINGS) - 0.28)
        ax.set_xlabel("Requested-goal success rate")

    axes[0].set_yticks(
        np.arange(len(fs.WORDINGS)),
        [fs.WORDING_LABELS[w] for w in reversed(fs.WORDINGS)],
    )
    axes[0].tick_params(axis="y", labelsize=11)
    for label in axes[0].get_yticklabels():
        label.set_fontweight("bold")
        label.set_color(fs.INK)
    fs.percent_axis(axes[0], "x", upper=1.0, step=0.25)
    axes[1].tick_params(labelleft=False)

    fig.legend(
        handles=[
            Line2D([], [], marker="o", markersize=9, linestyle="none",
                   color=fs.FAINT, alpha=0.55, label="LEFT request"),
            Line2D([], [], marker="D", markersize=8, linestyle="none",
                   color=fs.FAINT, label="RIGHT request"),
            Line2D([], [], color=fs.FAINT, linewidth=3.4, alpha=0.35,
                   label="95% Beta(1,1) credible interval"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.015),
        ncol=3,
    )
    fs.title_block(
        fig,
        "Neither checkpoint is steerable across direction and wording",
        "Closed-loop success on a matched neutral-start DROID task · 10 seeds per point · "
        "shaded band and label give the RIGHT-minus-LEFT gap",
        y=1.08,
    )
    fs.footnote(
        fig,
        "Ten trials per point bound brittleness, not deployment reliability; the intervals are wide by construction. "
        "A central Beta(1,1) credible interval need not contain a boundary proportion such as 0/10.",
        y=-0.055,
    )
    fs.save(fig, output / "direct_language_success_with_intervals.png")


# --------------------------------------------------------------------------
# Figure 2 — signed endpoint offsets
# --------------------------------------------------------------------------


def figure_offsets(closed: dict[str, Any], output: Path) -> None:
    episodes = _episode_index(closed)
    conditions = fs.iter_conditions()
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.08)
    rng = np.random.default_rng(11)

    limit = 0.52
    for ax, model_id in zip(axes, fs.MODEL_IDS):
        fs.clean(ax, grid_axis="x", keep=())
        base = fs.MODEL_COLORS[model_id]
        ax.axvspan(-limit, 0.0, color=fs.NEGATIVE_WASH, zorder=0)
        ax.axvline(0.0, color=fs.RULE, linewidth=1.2, zorder=1)

        for index, (wording, direction) in enumerate(conditions):
            y = len(conditions) - 1 - index
            values = np.asarray(
                [row["requested_signed_final_offset_m"] for row in episodes[(model_id, wording, direction)]]
            )
            colour = fs.model_color(model_id, direction)
            jitter = rng.uniform(-0.15, 0.15, values.size)
            ax.scatter(
                values,
                y + jitter,
                s=46,
                color=colour,
                edgecolor=fs.PAPER,
                linewidth=1.0,
                alpha=0.95,
                zorder=3,
            )
            mean = float(values.mean())
            ax.plot(
                [mean, mean],
                [y - 0.30, y + 0.30],
                color=fs.INK,
                linewidth=2.4,
                solid_capstyle="butt",
                zorder=4,
            )
            wrong_side = mean < 0
            ax.text(
                0.70,
                y,
                f"{mean*100:+.1f} cm",
                ha="right",
                va="center",
                fontsize=9.2,
                fontweight="bold",
                color=fs.NEGATIVE if wrong_side else fs.INK,
                bbox=dict(
                    boxstyle="round,pad=0.28",
                    facecolor=fs.NEGATIVE_WASH if wrong_side else fs.PAPER,
                    edgecolor=fs.NEGATIVE if wrong_side else "none",
                    linewidth=0.9,
                ),
                zorder=5,
            )

        for boundary in range(1, len(fs.WORDINGS)):
            ax.axhline(2 * boundary - 0.5, color=fs.GRID, linewidth=0.9, zorder=0)

        ax.axvline(0.575, color=fs.GRID, linewidth=1.0, zorder=0)
        ax.set_title(f"{fs.MODEL_LABELS[model_id]}  ·  {fs.MODEL_CLASS[model_id]}", color=base, pad=14)
        ax.set_xlim(-limit, 0.72)
        ax.set_xticks([-0.4, -0.2, 0.0, 0.2, 0.4], ["−0.4", "−0.2", "0", "+0.2", "+0.4"])
        ax.set_ylim(-0.7, len(conditions) - 0.3)
        ax.set_xlabel("Signed final cube offset toward the requested side (m)")

    axes[0].set_yticks(
        np.arange(len(conditions)),
        [f"{fs.WORDING_LABELS[w]} · {d.upper()}" for w, d in reversed(conditions)],
    )
    for label in axes[0].get_yticklabels():
        label.set_color(fs.INK)
        label.set_fontsize(9.6)
    axes[1].tick_params(labelleft=False)

    axes[0].text(
        -limit + 0.02,
        len(conditions) - 0.42,
        "OPPOSITE SIDE",
        fontsize=8.6,
        fontweight="bold",
        color=fs.NEGATIVE,
        ha="left",
        va="center",
    )

    fig.legend(
        handles=[
            Line2D([], [], marker="o", markersize=8, linestyle="none", color=fs.FAINT, label="one episode"),
            Line2D([], [], color=fs.INK, linewidth=2.4, label="condition mean"),
            Patch(facecolor=fs.NEGATIVE_WASH, label="cube ended on the side that was not requested"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=3,
    )
    fs.title_block(
        fig,
        "Failures are geometric, not a termination-threshold artefact",
        "Final cube position relative to the bowl, signed so positive always means the requested side · "
        "every registered episode, successes and failures alike",
        y=1.06,
    )
    fs.footnote(
        fig,
        "Endpoint geometry uses RoboLab rigid-object root poses in the robot frame (execution-geometry amendment 005), "
        "not rendered bounding-box centroids.",
        y=-0.05,
    )
    fs.save(fig, output / "direct_language_requested_side_offsets.png")


# --------------------------------------------------------------------------
# Figure 3 — wording robustness matrix
# --------------------------------------------------------------------------


def figure_robustness(closed: dict[str, Any], output: Path) -> None:
    groups = _group_index(closed)
    columns = [(model_id, direction) for model_id in fs.MODEL_IDS for direction in fs.DIRECTIONS]
    values = np.asarray(
        [[groups[(m, w, d)]["success_rate"] for m, d in columns] for w in fs.WORDINGS]
    )
    counts = np.asarray(
        [[groups[(m, w, d)]["successes"] for m, d in columns] for w in fs.WORDINGS]
    )

    ramp = LinearSegmentedColormap.from_list(
        "steerability",
        ["#F7FAFC", "#DCEFE7", "#9BD8C1", "#3FB794", fs.POSITIVE, "#046B4E"],
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.9))
    ax.imshow(values, vmin=0.0, vmax=1.0, cmap=ramp, aspect="auto")
    fs.strip_spines(ax)
    ax.tick_params(length=0)

    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            value = values[r, c]
            ax.text(
                c,
                r - 0.10,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=17,
                fontweight="bold",
                color=fs.PAPER if value >= 0.55 else fs.INK,
            )
            ax.text(
                c,
                r + 0.22,
                f"{counts[r, c]}/10",
                ha="center",
                va="center",
                fontsize=9.2,
                color="#E8F3EE" if value >= 0.55 else fs.MUTED,
            )

    for boundary in np.arange(0.5, values.shape[0] - 0.5):
        ax.axhline(boundary, color=fs.PAPER, linewidth=3.0)
    ax.axvline(1.5, color=fs.PAPER, linewidth=5.0)
    for boundary in (0.5, 2.5):
        ax.axvline(boundary, color=fs.PAPER, linewidth=3.0)

    ax.set_yticks(np.arange(len(fs.WORDINGS)), [fs.WORDING_LABELS[w] for w in fs.WORDINGS])
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
        label.set_fontsize(11)
        label.set_color(fs.INK)
    ax.set_xticks(np.arange(len(columns)), [d.upper() for _, d in columns])
    for label, (model_id, _) in zip(ax.get_xticklabels(), columns):
        label.set_fontweight("bold")
        label.set_color(fs.MODEL_COLORS[model_id])
        label.set_fontsize(10)

    for offset, model_id in zip((0.5, 2.5), fs.MODEL_IDS):
        ax.text(
            offset,
            -0.70,
            f"{fs.MODEL_LABELS[model_id]}  ·  {fs.MODEL_CLASS[model_id]}",
            ha="center",
            va="center",
            fontsize=11.5,
            fontweight="bold",
            color=fs.MODEL_COLORS[model_id],
        )
    ax.set_ylim(len(fs.WORDINGS) - 0.5, -0.92)

    ax.text(
        -0.70,
        2.5,
        "post-interim\nstress tier",
        rotation=90,
        ha="center",
        va="center",
        fontsize=8.8,
        color=fs.MUTED,
        linespacing=1.4,
    )
    ax.plot([-0.60, -0.60], [1.54, 3.46], color=fs.RULE, linewidth=1.8, clip_on=False)

    fs.title_block(
        fig,
        "The same task, reworded four ways",
        "Requested-goal success rate for each checkpoint and requested direction · "
        "no subtask coach, no privileged state, one episode-static prompt",
        y=0.99,
    )
    fs.footnote(
        fig,
        "Rows 3–4 are the prospectively frozen post-interim stress tier disclosed in "
        "direct_language_scope_amendment_003.json; rows 1–2 are the original confirmatory grid.",
        y=-0.035,
    )
    fs.save(fig, output / "direct_prompt_robustness.png")


# --------------------------------------------------------------------------
# Figure 4 — imagination versus execution
# --------------------------------------------------------------------------


def figure_quadrants(semantic: dict[str, Any], output: Path) -> None:
    groups = semantic["groups"]
    overall = semantic["overall"]
    labels = [f"{fs.WORDING_LABELS[row['wording']]}\n{row['direction'].upper()}" for row in groups]
    x = np.arange(len(groups))

    fig, axes = plt.subplots(
        2, 1, figsize=(13.6, 8.8), gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.58}
    )

    # (a) composition of every scored replan chunk
    top = axes[0]
    fs.clean(top, grid_axis=None, keep=())
    bottoms = np.zeros(len(groups))
    for quadrant in fs.QUADRANT_ORDER:
        fractions = np.asarray(
            [row["quadrant_counts"].get(quadrant, 0) / row["chunks"] for row in groups]
        )
        top.bar(
            x,
            fractions,
            bottom=bottoms,
            width=0.72,
            color=fs.QUADRANT_COLORS[quadrant],
            label=fs.QUADRANT_LABELS[quadrant],
            zorder=2,
        )
        bottoms += fractions
    for index, row in enumerate(groups):
        certain = row["coverage_fraction"]
        top.plot([index - 0.36, index + 0.36], [certain, certain], color=fs.INK, linewidth=1.8, zorder=5)
        fs.value_label(top, index, 1.035, f"n = {row['chunks']}", color=fs.MUTED, size=9.0, weight="regular")
    top.set_xticks(x, labels)
    fs.percent_axis(top, "y")
    top.set_ylim(0, 1.10)
    top.set_ylabel("Share of replan chunks")
    top.set_title("Most horizons are unscorable or certainly negative", pad=12)
    handles, labels_ = top.get_legend_handles_labels()
    handles.append(Line2D([], [], color=fs.INK, linewidth=1.8, label="evaluator coverage"))
    labels_.append("evaluator coverage")
    top.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.17), ncol=6)

    # (b) the informative quadrants, in counts
    bottom = axes[1]
    fs.clean(bottom, grid_axis="y")
    informative = (
        "imagines_requested_executes_requested",
        "imagines_requested_executes_not_requested",
        "does_not_imagine_requested_executes_requested",
    )
    width = 0.24
    for offset, quadrant in zip((-1, 0, 1), informative):
        counts = [row["quadrant_counts"].get(quadrant, 0) for row in groups]
        bars = bottom.bar(
            x + offset * width,
            counts,
            width * 0.88,
            color=fs.QUADRANT_COLORS[quadrant],
            label=fs.QUADRANT_LABELS[quadrant],
            zorder=3,
        )
        for bar, count in zip(bars, counts):
            if count:
                fs.value_label(
                    bottom,
                    bar.get_x() + bar.get_width() / 2,
                    count + 0.12,
                    str(count),
                    color=fs.QUADRANT_COLORS[quadrant],
                    size=9.4,
                )
    bottom.set_xticks(x, labels)
    bottom.set_ylim(0, 7.8)
    bottom.set_yticks(np.arange(0, 7, 2))
    bottom.set_ylabel("Replan chunks")
    bottom.set_title(
        "Contrastive LEFT produced zero aligned positives across 146 replans", pad=32
    )
    bottom.legend(loc="upper left", bbox_to_anchor=(0.0, 1.10), ncol=3)
    bottom.text(
        len(groups) - 0.5,
        7.2,
        "counts, not shares — the informative quadrants only",
        ha="right",
        va="center",
        fontsize=8.8,
        color=fs.MUTED,
        style="italic",
    )

    fs.panel_tag(top, "a", dx=-0.045, dy=1.10)
    fs.panel_tag(bottom, "b", dx=-0.045, dy=1.24)

    fs.title_block(
        fig,
        "Does the world model imagine what it executes?",
        f"Prompt-blind semantic labels for all {semantic['total_chunks']} Cosmos replan chunks across "
        f"{semantic['total_episodes']} episodes · frozen 0.20 m two-camera agreement threshold",
        y=1.04,
    )
    fs.footnote(
        fig,
        "Chunks within an episode are correlated, so these are descriptive rates and carry no binomial interval. "
        f"The evaluator abstained on {overall['executed_positive_chunks_with_uncertain_future']} of the "
        f"{overall['executed_positive_chunks_all']} horizons in which execution actually reached the requested relation.",
        y=-0.045,
    )
    fs.save(fig, output / "cosmos_imagination_execution_quadrants.png")


# --------------------------------------------------------------------------
# Figure 5 — semantic scorer threshold sensitivity
# --------------------------------------------------------------------------


def figure_threshold(semantic: dict[str, Any], output: Path) -> None:
    rows = semantic["threshold_sensitivity"]
    overall = semantic["threshold_sensitivity_overall"]
    conditions = fs.iter_conditions()

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.12)

    series_keys = (
        "coverage_fraction",
        "executed_positive_future_coverage_fraction",
        "imagination_execution_agreement_among_certain",
    )
    titles = (
        "Coverage over all chunks",
        "Coverage on executed-positive horizons",
        "Agreement among scored chunks",
    )
    notes = (
        "the evaluator labels fewer horizons…",
        "…positive events are discarded first…",
        "…yet agreement looks untouched",
    )

    for ax, key, title, note in zip(axes, series_keys, titles, notes):
        fs.clean(ax, grid_axis="y")
        for wording, direction in conditions:
            selected = sorted(
                [r for r in rows if r["wording"] == wording and r["direction"] == direction],
                key=lambda r: r["cross_camera_threshold_m"],
            )
            xs = [r["cross_camera_threshold_m"] for r in selected]
            ys = [np.nan if r[key] is None else r[key] for r in selected]
            ax.plot(
                xs,
                ys,
                color=fs.FAINT,
                linewidth=1.1,
                alpha=0.55,
                marker="o",
                markersize=2.6,
                zorder=2,
            )
        xs = [r["cross_camera_threshold_m"] for r in overall]
        ys = [r[key] for r in overall]
        ax.plot(xs, ys, color=fs.INK, linewidth=2.8, zorder=5, solid_capstyle="round")
        ax.scatter(xs, ys, s=52, color=fs.INK, edgecolor=fs.PAPER, linewidth=1.6, zorder=6)
        for xv, yv in zip(xs, ys):
            fs.value_label(
                ax,
                xv,
                yv + 0.045,
                f"{yv:.0%}",
                color=fs.INK,
                size=9.6,
                ha="center",
            )
        ax.axvline(0.20, color=fs.RULE, linewidth=1.1, linestyle=(0, (3, 3)), zorder=1)
        ax.set_title(title, pad=12)
        ax.set_xlabel("Cross-camera disagreement threshold (m)")
        ax.set_xticks([0.10, 0.15, 0.20])
        ax.set_xlim(0.089, 0.211)
        ax.set_ylim(-0.06, 1.18)
        ax.text(
            0.5,
            -0.25,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9.4,
            color=fs.MUTED,
            style="italic",
        )

    fs.percent_axis(axes[0], "y")
    axes[0].set_ylabel("Fraction")
    axes[0].text(
        0.199,
        1.11,
        "frozen threshold →",
        ha="right",
        va="center",
        fontsize=8.6,
        color=fs.MUTED,
    )

    fig.legend(
        handles=[
            Line2D([], [], color=fs.INK, linewidth=2.8, label="all conditions pooled"),
            Line2D([], [], color=fs.FAINT, linewidth=1.1, alpha=0.7, label="one wording × direction condition"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.005),
        ncol=2,
    )
    fs.title_block(
        fig,
        "A stricter evaluator buys agreement by discarding the evidence",
        "Replaying the frozen semantic labels at three cross-camera agreement thresholds",
        y=1.12,
    )
    fs.footnote(
        fig,
        "Read the middle panel first: at 0.10 m the scorer labels only 1% of the horizons in which execution reached "
        "the request, yet the right panel still reports 98% agreement.",
        y=-0.30,
    )
    fs.save(fig, output / "semantic_threshold_sensitivity.png")


# --------------------------------------------------------------------------
# Figure 6 — six-style fixed-observation probe
# --------------------------------------------------------------------------

PROBE_GROUPS = (
    (
        "Task language",
        (
            ("task_left", "task · LEFT (reference)"),
            ("task_left_repeat", "exact repeat"),
            ("task_left_paraphrase", "paraphrase"),
            ("task_right", "task · RIGHT"),
        ),
    ),
    ("Subtask", (("subtask_grasp", "reach for the cube"),)),
    ("Atomic motion", (("atomic_left", "move left"), ("atomic_right", "move right"))),
    (
        "Grounded point / trace",
        (
            ("gripper_trace_to_cube", "trace to cube"),
            ("point_cube", "point · cube"),
            ("point_left_target", "point · LEFT target"),
            ("point_right_target", "point · RIGHT target"),
        ),
    ),
    ("Combination", (("combination_left", "combo · LEFT"), ("combination_right", "combo · RIGHT"))),
    (
        "Negative controls",
        (
            ("unrelated_control", "unrelated task"),
            ("noun_swap_control", "noun swap"),
            ("contradictory_control", "contradictory"),
        ),
    ),
)


def figure_command_probe(probes: dict[str, Any], output: Path) -> None:
    records = {
        model: {row["condition"]: row for row in probes[model]["manifest"]["records"]}
        for model in ("pi05", "cosmos")
    }
    model_of = {"pi05": "pi05_droid_vla", "cosmos": "cosmos3_edge_droid_wam"}

    # Lay the rows out top-to-bottom, inserting a header slot before each group.
    rows: list[dict[str, Any]] = []
    cursor = 0.0
    for group_index, (group_name, entries) in enumerate(PROBE_GROUPS):
        if group_index:
            cursor += 0.55
        rows.append({"kind": "header", "label": group_name, "y": cursor})
        cursor += 0.95
        for condition, label in entries:
            rows.append({"kind": "item", "label": label, "condition": condition, "y": cursor})
            cursor += 1.0
    items = [row for row in rows if row["kind"] == "item"]
    headers = [row for row in rows if row["kind"] == "header"]

    values_by_model = {
        model: [records[model][row["condition"]]["action_rms_vs_task_left"] for row in items]
        for model in ("pi05", "cosmos")
    }
    ceiling = max(max(values) for values in values_by_model.values())

    fig, ax = plt.subplots(figsize=(12.2, 9.0))
    fig.subplots_adjust(left=0.235, right=0.985, top=0.965, bottom=0.075)
    fs.clean(ax, grid_axis="x", keep=())

    height = 0.36
    for model_index, model in enumerate(("pi05", "cosmos")):
        model_id = model_of[model]
        values = values_by_model[model]
        offset = (0.5 - model_index) * height
        bars = ax.barh(
            np.asarray([row["y"] for row in items]) + offset,
            values,
            height * 0.9,
            color=fs.MODEL_COLORS[model_id],
            label=f"{fs.MODEL_LABELS[model_id]} ({fs.MODEL_CLASS[model_id]})",
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                value + ceiling * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8.6,
                color=fs.MODEL_COLORS[model_id] if value > 0 else fs.MUTED,
                fontweight="bold" if value > 0 else "regular",
            )

    for header in headers:
        ax.text(
            -0.30,
            header["y"],
            header["label"].upper(),
            transform=ax.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=9.4,
            fontweight="bold",
            color=fs.INK,
        )
        ax.plot(
            [-0.30, 1.0],
            [header["y"] - 0.52, header["y"] - 0.52],
            transform=ax.get_yaxis_transform(),
            color=fs.GRID,
            linewidth=1.0,
            clip_on=False,
            zorder=0,
        )

    ax.set_yticks([row["y"] for row in items], [row["label"] for row in items])
    for label in ax.get_yticklabels():
        label.set_fontsize(9.6)
        label.set_color(fs.INK_SOFT)
    ax.set_xlabel("Action RMS relative to the canonical LEFT task, on identical observation bytes")
    ax.set_xlim(0, ceiling * 1.13)
    ax.set_ylim(cursor - 0.5, -1.0)
    ax.legend(loc="lower right", ncol=1, bbox_to_anchor=(1.0, 0.01))

    fs.title_block(
        fig,
        "The interface reacts to far more than task captions",
        "One hash-pinned observation, one sampling seed, sixteen commands · all six command styles from Chen et al. "
        "plus four negative controls",
        y=1.005,
    )
    fs.footnote(
        fig,
        "Distance is a sensitivity diagnostic, never a success metric. A large tensor movement shows the command "
        "reached the model; it does not show the command was obeyed. Both endpoints returned RMS 0.0 on the exact repeat.",
        y=-0.02,
    )
    fs.save(fig, output / "command_probe_action_sensitivity.png")


# --------------------------------------------------------------------------
# Figure 7 — exact-input direct task probe
# --------------------------------------------------------------------------


def figure_direct_task_probe(probe: dict[str, Any], output: Path) -> None:
    families = ["canonical", "short", "declarative", "contrastive target first", "contrastive target last"]
    family_labels = ["Canonical", "Short", "Declarative", "Contrastive\ntarget first", "Contrastive\ntarget last"]
    order_families = ["contrastive order left", "contrastive order right"]
    order_labels = ["LEFT target", "RIGHT target"]

    rows = {
        model: {row["prompt_family"]: row["action_rms"] for row in probe[model]["paired_action_rms"]}
        for model in ("pi05", "cosmos")
    }
    model_of = {"pi05": "pi05_droid_vla", "cosmos": "cosmos3_edge_droid_wam"}

    fig, axes = plt.subplots(
        1, 2, figsize=(13.6, 5.6), gridspec_kw={"width_ratios": [2.5, 1.0], "wspace": 0.18}
    )
    width = 0.36

    for model_index, model in enumerate(("pi05", "cosmos")):
        model_id = model_of[model]
        colour = fs.MODEL_COLORS[model_id]
        left_values = [rows[model][family] for family in families]
        axes[0].bar(
            np.arange(len(families)) + (model_index - 0.5) * width,
            left_values,
            width * 0.9,
            color=colour,
            label=f"{fs.MODEL_LABELS[model_id]} ({fs.MODEL_CLASS[model_id]})",
            zorder=3,
        )
        order_values = [rows[model][family] for family in order_families]
        axes[1].bar(
            np.arange(len(order_families)) + (model_index - 0.5) * width,
            order_values,
            width * 0.9,
            color=colour,
            zorder=3,
        )

    for ax in axes:
        fs.clean(ax, grid_axis="y")
        ax.set_ylim(0, 0.0245)

    # π0.5 word-order reference line: its order effect exceeds every relation effect.
    pi_order_max = max(rows["pi05"][family] for family in order_families)
    for ax in axes:
        ax.axhline(pi_order_max, color=fs.VLA, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    axes[0].text(
        len(families) - 0.55,
        pi_order_max + 0.0003,
        "π0.5 largest word-order effect",
        fontsize=8.8,
        color=fs.VLA,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    axes[0].set_xticks(np.arange(len(families)), family_labels)
    axes[0].set_title("Changing the requested relation", pad=12)
    axes[0].set_ylabel("Action RMS on identical input bytes")

    axes[1].set_xticks(np.arange(len(order_families)), order_labels)
    axes[1].set_title("Moving the target before vs. after the negated clause", pad=12)
    axes[1].tick_params(labelleft=False)

    for ax, keys in ((axes[0], families), (axes[1], order_families)):
        for model_index, model in enumerate(("pi05", "cosmos")):
            for position, key in enumerate(keys):
                value = rows[model][key]
                ax.text(
                    position + (model_index - 0.5) * width,
                    value + 0.00055,
                    f"{value:.4f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    color=fs.MODEL_COLORS[model_of[model]],
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor=fs.PAPER, edgecolor="none"),
                    zorder=4,
                )

    fig.legend(
        handles=[
            Patch(facecolor=fs.MODEL_COLORS[model_of[m]],
                  label=f"{fs.MODEL_LABELS[model_of[m]]} ({fs.MODEL_CLASS[model_of[m]]})")
            for m in ("pi05", "cosmos")
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
    )
    fs.panel_tag(axes[0], "a", dx=-0.055, dy=1.10)
    fs.panel_tag(axes[1], "b", dx=-0.055, dy=1.10)

    fs.title_block(
        fig,
        "Word order can matter more than the word",
        "Byte-identical observation and sampling seed · paired action distance for four task wordings "
        "and for two semantically equivalent contrastive orderings",
        y=1.10,
    )
    fs.footnote(
        fig,
        "For π0.5 the word-order contrast exceeds the LEFT-versus-RIGHT contrast in every prompt family, so a nonzero "
        "response to the requested relation cannot be read as evidence that the relation was the operative variable.",
        y=-0.055,
    )
    fs.save(fig, output / "direct_task_exact_probe.png")


# --------------------------------------------------------------------------
# Figure 8 — renderer variation across exact physical resets
# --------------------------------------------------------------------------


def figure_observation_variation(audit: dict[str, Any], output: Path) -> None:
    """Dot plot: every contrast sits far above zero, so none of them is clean."""
    rows = audit["summaries"]
    wording_label = {
        "canonical": "canonical",
        "short_paraphrase": "short",
        "declarative_goal": "declarative",
        "contrastive_goal": "contrastive",
        "short": "short",
        "declarative": "declarative",
        "contrastive": "contrastive",
    }

    def classify(row: dict[str, Any]) -> tuple[str, str, str]:
        comparison = row["comparison"]
        condition = row["condition_id"].replace("cosmos_", "").replace("_static32", "")
        direction = row["direction"].replace("left_vs_right", "LEFT vs RIGHT")
        if comparison == "within_condition_direction":
            name = wording_label.get(condition, condition)
            return (
                "Same prompt, same direction\n(pure nuisance variation)",
                f"{name} · {direction.upper()}",
                fs.FAINT,
            )
        if comparison == "matched_left_right":
            name = wording_label.get(condition.replace("_vs_", " vs "), condition)
            return (
                "Matched LEFT vs RIGHT\n(the closed-loop language contrast)",
                f"{name}",
                fs.WAM,
            )
        pair = comparison.replace("matched_", "")
        known = ("canonical", "declarative", "contrastive", "short")
        left = next((name for name in known if pair.startswith(name)), pair)
        right = pair[len(left) :].strip("_") or "?"
        label = f"{wording_label.get(left, left)} vs {wording_label.get(right, right)} · {direction.upper()}"
        return ("Across wordings\n(different prompt, same reset)", label, fs.NEUTRAL)

    buckets: dict[str, list[tuple[str, float, float, str]]] = {}
    for row in rows:
        bucket, label, colour = classify(row)
        buckets.setdefault(bucket, []).append(
            (label, row["mean_mae_0_255"], row["p90_mae_0_255"], colour)
        )

    order = [name for name in buckets if "Same prompt" in name]
    order += [name for name in buckets if "Matched" in name]
    order += [name for name in buckets if "Across" in name]

    entries: list[tuple[str | None, str, float, float, str]] = []
    for bucket in order:
        for index, (label, mean, p90, colour) in enumerate(buckets[bucket]):
            entries.append((bucket if index == 0 else None, label, mean, p90, colour))

    total = len(entries) + len(order)
    fig, ax = plt.subplots(figsize=(12.0, 0.34 * total + 2.2))
    fig.subplots_adjust(left=0.30, right=0.97, top=0.94, bottom=0.11)
    fs.clean(ax, grid_axis="x", keep=())

    y = 0.0
    ticks: list[float] = []
    labels: list[str] = []
    for bucket, label, mean, p90, colour in entries:
        if bucket is not None:
            y += 1.15
            ax.text(
                -0.38,
                y,
                bucket,
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                fontsize=9.4,
                fontweight="bold",
                color=fs.INK,
                linespacing=1.45,
            )
            ax.plot(
                [-0.38, 1.0],
                [y - 0.62, y - 0.62],
                transform=ax.get_yaxis_transform(),
                color=fs.GRID,
                linewidth=1.0,
                clip_on=False,
                zorder=0,
            )
            y += 0.95
        ax.plot([mean, p90], [y, y], color=colour, linewidth=2.6, alpha=0.35, solid_capstyle="round", zorder=2)
        ax.plot([mean], [y], marker="o", markersize=8.5, color=colour,
                markeredgecolor=fs.PAPER, markeredgewidth=1.4, linestyle="none", zorder=4)
        ax.plot([p90], [y], marker="|", markersize=9, color=colour, markeredgewidth=2.0,
                linestyle="none", zorder=4)
        ax.text(p90 + 0.14, y, f"{mean:.2f}", va="center", ha="left",
                fontsize=8.8, fontweight="bold", color=fs.INK)
        ticks.append(y)
        labels.append(label)
        y += 1.0

    ax.set_yticks(ticks, labels)
    for label_artist in ax.get_yticklabels():
        label_artist.set_fontsize(9.4)
        label_artist.set_color(fs.INK_SOFT)
    ax.set_ylim(y - 0.1, -0.9)
    ax.set_xlim(0, 5.9)
    ax.axvline(0.0, color=fs.RULE, linewidth=1.4, zorder=1)
    ax.set_xlabel("First-conditioning-image mean absolute pixel difference (0–255)")

    ax.annotate(
        "a byte-identical render would sit here",
        xy=(0.0, y - 0.75),
        xytext=(24, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.8,
        color=fs.MUTED,
        arrowprops=dict(arrowstyle="->", color=fs.RULE, linewidth=1.0),
    )

    fig.legend(
        handles=[
            Line2D([], [], marker="o", markersize=8, linestyle="none", color=fs.FAINT, label="mean over pairs"),
            Line2D([], [], marker="|", markersize=9, markeredgewidth=2.0, linestyle="none",
                   color=fs.FAINT, label="pairwise p90"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.97, 1.005),
        ncol=2,
    )
    fs.title_block(
        fig,
        "An exact physical reset is not a byte-identical render",
        "Difference between first conditioning images drawn from resets whose robot and rigid-object state hash "
        "identically",
        y=0.995,
    )
    fs.footnote(
        fig,
        "Same-prompt resets already differ by roughly 3 grey levels, so the closed-loop opposite-prompt action "
        "contrast carries prompt, settling and renderer variation together. It is reported as a diagnostic, never as "
        "a causal language effect; the byte-identical fixed-observation probe is the causal intervention.",
        y=-0.045,
    )
    fs.save(fig, output / "cosmos_conditioning_image_variation.png")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


FIGURE_NAMES = (
    "direct_language_success_with_intervals.png",
    "direct_language_requested_side_offsets.png",
    "direct_prompt_robustness.png",
    "cosmos_imagination_execution_quadrants.png",
    "semantic_threshold_sensitivity.png",
    "command_probe_action_sensitivity.png",
    "direct_task_exact_probe.png",
    "cosmos_conditioning_image_variation.png",
)


def render_all(compiled: dict[str, Any], output: Path, *, scale: float = 1.0) -> tuple[str, ...]:
    """Render every data figure from an already-compiled evidence dictionary.

    ``compile_vla_wam_study.py`` calls this so that the pipeline and a manual
    restyle always emit byte-comparable figures from one code path.
    """
    fs.use_style(scale)
    prospective = compiled["prospective"]
    output.mkdir(parents=True, exist_ok=True)

    figure_success(prospective["closed_loop"], output)
    figure_offsets(prospective["closed_loop"], output)
    figure_robustness(prospective["closed_loop"], output)
    figure_quadrants(prospective["cosmos_semantic_futures"], output)
    figure_threshold(prospective["cosmos_semantic_futures"], output)
    figure_command_probe(prospective["command_probe"], output)
    figure_direct_task_probe(prospective["direct_task_command_probe"], output)
    figure_observation_variation(prospective["cosmos_observation_variation"], output)
    return FIGURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    names = render_all(json.loads(args.evidence.read_text()), args.output, scale=args.scale)
    print(f"Rendered {len(names)} figures into {args.output}")


if __name__ == "__main__":
    main()
