#!/usr/bin/env python3
"""Render reader-first v2 explainers from the frozen protocol and v1 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


BACKGROUND = "#F7F4EE"
CARD = "#FFFFFF"
TEXT = "#172A3A"
MUTED = "#61717F"
GRID = "#D7DEE3"
LEFT = "#D97706"
RIGHT = "#2563A8"
SUCCESS = "#138A62"
FAILURE = "#FFFFFF"
PROMPT_ORDER = [
    "canonical",
    "short_paraphrase",
    "declarative_goal",
    "contrastive_goal",
]
PROMPT_META = {
    "canonical": {
        "key": "D",
        "name": "Direct command",
        "left": "Put the Rubik's cube to the LEFT of the bowl.",
        "change": "Reference wording",
        "question": "Can it ground a direct task instruction?",
    },
    "short_paraphrase": {
        "key": "S",
        "name": "Short command",
        "left": "Put the cube LEFT of the bowl.",
        "change": "Removed ‘Rubik’s’ and ‘to the’",
        "question": "Does grounding survive lexical compression?",
    },
    "declarative_goal": {
        "key": "G",
        "name": "Goal as outcome",
        "left": "The Rubik's cube should end up to the LEFT of the bowl.",
        "change": "Imperative → desired end state",
        "question": "Does a declarative goal ground like a command?",
    },
    "contrastive_goal": {
        "key": "C",
        "name": "Desired + negated opposite",
        "left": "Put the Rubik's cube to the LEFT of the bowl, not to the RIGHT of the bowl.",
        "change": "Added an explicitly rejected RIGHT",
        "question": "Can it resolve negation and semantic scope?",
    },
}
MODEL_META = {
    "pi05_droid_vla": ("π0.5 DROID", "VLA"),
    "cosmos3_edge_droid_wam": ("Cosmos3 Edge DROID", "WAM"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": TEXT,
            "axes.titleweight": "bold",
            "axes.facecolor": BACKGROUND,
            "figure.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100, facecolor=BACKGROUND)
    plt.close(fig)


def draw_header(fig: plt.Figure, title: str, subtitle: str, square: bool) -> None:
    fig.text(0.055, 0.946, title, fontsize=26 if not square else 24, weight="bold", va="top")
    rendered_subtitle = "\n".join(
        textwrap.wrap(subtitle, 128 if square else 145)
    )
    fig.text(
        0.055,
        0.895 if not square else 0.907,
        rendered_subtitle,
        fontsize=12 if not square else 11,
        color=MUTED,
        va="top",
    )


def prompt_semantics(output: Path, square: bool) -> None:
    if square:
        fig = plt.figure(figsize=(12, 12))
        positions = [
            (0.055, 0.52, 0.425, 0.30),
            (0.52, 0.52, 0.425, 0.30),
            (0.055, 0.185, 0.425, 0.30),
            (0.52, 0.185, 0.425, 0.30),
        ]
        title_size, prompt_size, body_size = 14, 13, 10.5
    else:
        fig = plt.figure(figsize=(16, 9))
        positions = [
            (0.055 + index * 0.235, 0.24, 0.215, 0.55) for index in range(4)
        ]
        title_size, prompt_size, body_size = 14, 12.5, 10.5
    draw_header(
        fig,
        "What changed in the sentence?",
        "The scene, reset, model, seed, requested physical relation, and success checker stay fixed. Only the episode-static sentence changes.",
        square,
    )
    for position, wording in zip(positions, PROMPT_ORDER, strict=True):
        x, y, width, height = position
        meta = PROMPT_META[wording]
        card = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            transform=fig.transFigure,
            facecolor=CARD,
            edgecolor=GRID,
            linewidth=1.2,
        )
        fig.add_artist(card)
        badge = FancyBboxPatch(
            (x + 0.018, y + height - 0.072),
            0.042,
            0.045,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor=TEXT,
            edgecolor=TEXT,
        )
        fig.add_artist(badge)
        fig.text(
            x + 0.039,
            y + height - 0.049,
            meta["key"],
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="white",
        )
        fig.text(
            x + 0.073,
            y + height - 0.048,
            "Contrastive negation"
            if wording == "contrastive_goal" and not square
            else meta["name"],
            fontsize=title_size,
            weight="bold",
            va="center",
        )
        prompt_width = 39 if square else 27
        fig.text(
            x + 0.022,
            y + height - 0.125,
            "\n".join(textwrap.wrap(meta["left"], prompt_width)),
            fontsize=prompt_size,
            weight="bold",
            va="top",
            linespacing=1.35,
        )
        relation_y = y + (0.06 if square else 0.19)
        fig.text(
            x + 0.022,
            relation_y + 0.072,
            "DESIRED",
            fontsize=8.5,
            weight="bold",
            color=MUTED,
        )
        fig.text(
            x + 0.022,
            relation_y + 0.035,
            "LEFT",
            fontsize=12,
            weight="bold",
            color=LEFT,
        )
        if wording == "contrastive_goal":
            fig.text(
                x + width * 0.45,
                relation_y + 0.072,
                "EXPLICITLY REJECTED",
                fontsize=8.5,
                weight="bold",
                color=MUTED,
            )
            fig.text(
                x + width * 0.45,
                relation_y + 0.035,
                "RIGHT",
                fontsize=12,
                weight="bold",
                color=RIGHT,
            )
        fig.text(
            x + 0.022,
            y + 0.075,
            meta["change"],
            fontsize=body_size,
            color=MUTED,
            va="bottom",
        )
        fig.text(
            x + 0.022,
            y + 0.026,
            "\n".join(textwrap.wrap(meta["question"], 48 if square else 33)),
            fontsize=body_size,
            weight="bold",
            va="bottom",
        )
    footer_y = 0.075 if square else 0.105
    footer_text = (
        "One side is requested; the opposite side appears only inside a negated phrase. "
        "A direction-word bag sees both. A grounded policy must recover scope."
    )
    fig.text(
        0.055,
        footer_y,
        "Contrastive ≠ contradictory",
        fontsize=12,
        weight="bold",
    )
    fig.text(
        0.235 if not square else 0.28,
        footer_y,
        "\n".join(textwrap.wrap(footer_text, 92)) if square else footer_text,
        fontsize=11,
        color=MUTED,
    )
    save(fig, output)


def wilson(successes: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def scorecard(rows: list[dict[str, str]], output: Path, square: bool) -> None:
    summaries = {
        (row["model_id"], row["wording"], row["direction"]): row for row in rows
    }
    fig = plt.figure(figsize=(12, 12) if square else (16, 9))
    draw_header(
        fig,
        "Did the robot obey the requested side?",
        "Existing DROID reference evidence · requested-relation task success · raw counts and 95% Wilson intervals · 10 episodes per cell",
        square,
    )
    if square:
        axes = [
            fig.add_axes((0.29, 0.565, 0.64, 0.25)),
            fig.add_axes((0.29, 0.205, 0.64, 0.25)),
        ]
    else:
        axes = [
            fig.add_axes((0.24, 0.18, 0.32, 0.61)),
            fig.add_axes((0.64, 0.18, 0.32, 0.61)),
        ]
    for axis, model_id in zip(axes, MODEL_META, strict=True):
        name, model_class = MODEL_META[model_id]
        for index, wording in enumerate(PROMPT_ORDER):
            y_base = 3 - index
            for direction, delta, color in (("left", 0.12, LEFT), ("right", -0.12, RIGHT)):
                row = summaries[(model_id, wording, direction)]
                success = int(row["successes"])
                total = int(row["episodes"])
                rate = success / total
                low, high = wilson(success, total)
                axis.errorbar(
                    rate,
                    y_base + delta,
                    xerr=[[max(0.0, rate - low)], [max(0.0, high - rate)]],
                    fmt="o",
                    color=color,
                    ecolor=color,
                    markersize=8,
                    elinewidth=2,
                    capsize=3,
                    zorder=3,
                )
                label_x = min(rate + 0.055, 1.035)
                ha = "left" if rate < 0.93 else "right"
                if ha == "right":
                    label_x = rate - 0.045
                axis.text(
                    label_x,
                    y_base + delta,
                    f"{success}/{total}",
                    va="center",
                    ha=ha,
                    fontsize=9.5,
                    weight="bold",
                    color=TEXT,
                )
        axis.set_xlim(-0.03, 1.08)
        axis.set_ylim(-0.55, 3.55)
        axis.set_yticks(range(4))
        axis.set_yticklabels(
            [
                PROMPT_META[wording]["name"]
                for wording in reversed(PROMPT_ORDER)
            ],
            fontsize=10.5,
        )
        axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axis.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        axis.grid(axis="x", color=GRID, linewidth=0.8)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(axis="y", length=0, pad=9)
        axis.set_title(f"{name}  ·  {model_class}", fontsize=15, loc="left", pad=12)
        axis.set_xlabel("Episodes satisfying the requested physical relation", fontsize=10)
    legend = [
        Line2D([0], [0], marker="o", color=LEFT, label="Prompt asks LEFT", linewidth=2),
        Line2D([0], [0], marker="o", color=RIGHT, label="Prompt asks RIGHT", linewidth=2),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.055 if not square else 0.035),
        frameon=False,
        ncol=2,
        fontsize=10.5,
    )
    takeaway = (
        "Visible checkpoint-specific bias: π0.5 succeeds far more often on RIGHT; "
        "Cosmos is strong on RIGHT but loses LEFT grounding under short and contrastive wording."
    )
    fig.text(
        0.05 if square else 0.43,
        0.12 if square else 0.072,
        "\n".join(textwrap.wrap(takeaway, 74 if not square else 85)),
        fontsize=10.5,
        color=MUTED,
        va="center",
    )
    save(fig, output)


def endpoint_pairs(rows: list[dict[str, str]], output: Path, square: bool) -> None:
    grouped: dict[tuple[str, str, int], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["model_id"], row["wording"], int(row["episode_seed"]))][
            row["direction"]
        ] = row
    fig = plt.figure(figsize=(12, 12) if square else (16, 9))
    draw_header(
        fig,
        "Where did the object actually finish?",
        "Every dot is one episode. Lines connect the same model, wording, and seed after changing only LEFT ↔ RIGHT. Filled dots passed the full task checker; open dots did not.",
        square,
    )
    axis = fig.add_axes((0.28 if square else 0.22, 0.17, 0.66 if square else 0.73, 0.66))
    y_labels: list[str] = []
    y_positions: list[float] = []
    model_order = list(MODEL_META)
    row_index = 0
    for model_id in model_order:
        for wording in PROMPT_ORDER:
            y = 7 - row_index
            y_positions.append(y)
            y_labels.append(f"{PROMPT_META[wording]['key']}  {PROMPT_META[wording]['name']}")
            keys = sorted(
                key
                for key in grouped
                if key[0] == model_id and key[1] == wording
            )
            left_values: list[float] = []
            right_values: list[float] = []
            for key in keys:
                pair = grouped[key]
                if set(pair) != {"left", "right"}:
                    continue
                left_row, right_row = pair["left"], pair["right"]
                left_x = -float(left_row["final_cube_minus_bowl_y_m"])
                right_x = -float(right_row["final_cube_minus_bowl_y_m"])
                left_values.append(left_x)
                right_values.append(right_x)
                axis.plot(
                    [left_x, right_x],
                    [y + 0.10, y - 0.10],
                    color="#BFC8CE",
                    linewidth=0.8,
                    alpha=0.65,
                    zorder=1,
                )
                for value, row, dy, color in (
                    (left_x, left_row, 0.10, LEFT),
                    (right_x, right_row, -0.10, RIGHT),
                ):
                    succeeded = row["binary_success"] == "True"
                    axis.scatter(
                        [value],
                        [y + dy],
                        s=38,
                        facecolor=color if succeeded else FAILURE,
                        edgecolor=color,
                        linewidth=1.4,
                        zorder=3,
                    )
            for values, dy, color in ((left_values, 0.10, LEFT), (right_values, -0.10, RIGHT)):
                if values:
                    median = sorted(values)[len(values) // 2 - (0 if len(values) % 2 else 1)]
                    if len(values) % 2 == 0:
                        ordered = sorted(values)
                        median = (ordered[len(values) // 2 - 1] + ordered[len(values) // 2]) / 2
                    axis.plot(
                        [median, median],
                        [y + dy - 0.13, y + dy + 0.13],
                        color=TEXT,
                        linewidth=2.2,
                        zorder=4,
                    )
            row_index += 1
        if model_id != model_order[-1]:
            axis.axhline(3.5, color=GRID, linewidth=1.2)
    axis.axvspan(-0.5, 0, color=LEFT, alpha=0.055, zorder=0)
    axis.axvspan(0, 0.5, color=RIGHT, alpha=0.045, zorder=0)
    axis.axvline(0, color=TEXT, linewidth=1.0, alpha=0.8)
    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.55, 7.55)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(y_labels, fontsize=10)
    axis.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    axis.set_xticklabels(["0.4 m", "0.2 m", "BOWL", "0.2 m", "0.4 m"])
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="y", length=0, pad=8)
    axis.set_xlabel("Robot LEFT   ←   final bowl-relative lateral endpoint   →   Robot RIGHT", fontsize=11)
    axis.text(
        -0.495,
        7.72,
        "π0.5 DROID · VLA",
        fontsize=12,
        weight="bold",
        va="bottom",
        clip_on=False,
    )
    axis.text(
        -0.495,
        3.72,
        "Cosmos3 Edge DROID · WAM",
        fontsize=12,
        weight="bold",
        va="bottom",
        clip_on=False,
    )
    legend = [
        Line2D([0], [0], marker="o", color=LEFT, markerfacecolor=LEFT, label="Asked LEFT · success", linewidth=0),
        Line2D([0], [0], marker="o", color=RIGHT, markerfacecolor=RIGHT, label="Asked RIGHT · success", linewidth=0),
        Line2D([0], [0], marker="o", color=MUTED, markerfacecolor=FAILURE, label="Open marker · failure", linewidth=0),
        Line2D([0], [0], color=TEXT, label="Black tick · median", linewidth=2),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.04),
        frameon=False,
        ncol=2 if square else 4,
        fontsize=9.5,
    )
    fig.text(
        0.28 if square else 0.22,
        0.105 if square else 0.105,
        "Endpoint direction is intuitive; task success still uses the full release-inside-45°-cone predicate, not lateral position alone.",
        fontsize=9.5,
        color=MUTED,
    )
    save(fig, output)


def efficient_pilot_paths(compiled: dict[str, Any], output: Path, square: bool) -> None:
    episodes = compiled["episodes"]
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for episode in episodes:
        by_pair[episode["pair_id"]][episode["requested_relation"]] = episode
    if len(by_pair) != 3 or any(set(pair) != {"left", "right"} for pair in by_pair.values()):
        raise RuntimeError("Efficient-WAM path figure requires three exact left/right pairs")

    fig = plt.figure(figsize=(12, 12) if square else (16, 9))
    draw_header(
        fig,
        "Same scene. Opposite command. Where did the object go?",
        "Efficient-WAM-RT standardized v2 pilot · solid lines are executed target-relative paths · dotted arrows show the shortest expected move into each requested success region · simulator state is used only for scoring and visualization",
        square,
    )
    if square:
        positions = [
            (0.06, 0.57, 0.42, 0.26),
            (0.54, 0.57, 0.42, 0.26),
            (0.30, 0.17, 0.42, 0.26),
        ]
    else:
        positions = [(0.055 + index * 0.315, 0.22, 0.285, 0.58) for index in range(3)]

    x_grid = np.linspace(-0.3, 0.3, 401)
    y_grid = np.linspace(-0.18, 0.30, 321)
    xx, yy = np.meshgrid(x_grid, y_grid)
    radius = np.sqrt(xx * xx + yy * yy)
    base_region = (radius > 0.08) & (radius < 0.2) & (np.abs(yy) < 0.05)

    for index, (pair_id, pair) in enumerate(sorted(by_pair.items())):
        axis = fig.add_axes(positions[index])
        axis.contourf(
            xx,
            yy,
            (base_region & (xx < 0)).astype(float),
            levels=[0.5, 1.5],
            colors=[LEFT],
            alpha=0.09,
        )
        axis.contourf(
            xx,
            yy,
            (base_region & (xx > 0)).astype(float),
            levels=[0.5, 1.5],
            colors=[RIGHT],
            alpha=0.08,
        )
        axis.axhline(0, color=GRID, linewidth=0.8, zorder=0)
        axis.axvline(0, color=TEXT, linewidth=1.0, alpha=0.7, zorder=0)

        left_episode = pair["left"]
        object_label = left_episode["movable_description"]
        reference_label = left_episode["reference_description"]
        title = f"Scene {index + 1} · {object_label} → {reference_label}"
        axis.set_title("\n".join(textwrap.wrap(title, 42)), loc="left", fontsize=11.5, pad=9)

        start = None
        for direction, color, desired_x in (
            ("left", LEFT, -0.14),
            ("right", RIGHT, 0.14),
        ):
            episode = pair[direction]
            trajectory = json.loads(Path(episode["raw_trajectory"]["path"]).read_text())
            xs = np.asarray([step["object_minus_target_x"] for step in trajectory])
            ys = np.asarray([step["object_minus_target_y"] for step in trajectory])
            if start is None:
                start = (float(xs[0]), float(ys[0]))
            axis.annotate(
                "",
                xy=(desired_x, 0),
                xytext=(float(xs[0]), float(ys[0])),
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "linestyle": ":",
                    "linewidth": 1.25,
                    "alpha": 0.65,
                },
                zorder=1,
            )
            axis.plot(xs, ys, color=color, linewidth=2.0, alpha=0.88, zorder=2)
            endpoint_marker = "o" if episode["requested_success"] else "X"
            axis.scatter(
                [xs[-1]],
                [ys[-1]],
                s=72,
                marker=endpoint_marker,
                facecolor=color,
                edgecolor=CARD if endpoint_marker == "o" else color,
                linewidth=1.2,
                zorder=4,
            )
            status = "SUCCESS" if episode["requested_success"] else "FAIL"
            axis.text(
                0.02 if direction == "left" else 0.98,
                -0.18,
                f"ASKED {direction.upper()}  ·  {status}\nend x {float(xs[-1]):+.3f} m",
                transform=axis.transAxes,
                ha="left" if direction == "left" else "right",
                va="top",
                fontsize=8.7,
                color=TEXT,
                weight="bold",
            )
        if start is not None:
            axis.scatter(
                [start[0]],
                [start[1]],
                s=45,
                marker="D",
                facecolor=TEXT,
                edgecolor=CARD,
                linewidth=0.8,
                zorder=5,
            )
            axis.text(start[0], start[1] + 0.018, "same start", ha="center", fontsize=8, color=MUTED)
        axis.text(-0.14, -0.043, "LEFT success region", ha="center", fontsize=7.8, color=TEXT)
        axis.text(0.14, -0.043, "RIGHT success region", ha="center", fontsize=7.8, color=TEXT)
        axis.set_xlim(-0.30, 0.30)
        axis.set_ylim(-0.18, 0.30)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        axis.set_yticks([-0.1, 0, 0.1, 0.2])
        axis.set_xticklabels(["−.2", "−.1", "TARGET", "+.1", "+.2"], fontsize=8)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(color=GRID, linewidth=0.55, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color(GRID)
        axis.set_xlabel("target-relative x (m)", fontsize=8.5, labelpad=7)
        axis.set_ylabel("target-relative y (m)", fontsize=8.5)

    summary = compiled["summary"]
    takeaway = (
        f"Pilot gate: LEFT {summary['by_direction']['left']['successes']}/3 · "
        f"RIGHT {summary['by_direction']['right']['successes']}/3. "
        "All six runs picked up the object, so the failures are post-pick placement failures—not inactivity. "
        "Two of three paired endpoint responses move in the wrong direction when LEFT is changed to RIGHT."
    )
    fig.text(
        0.055,
        0.065 if not square else 0.025,
        "\n".join(textwrap.wrap(takeaway, 138 if not square else 96)),
        fontsize=10.5 if not square else 9.7,
        color=MUTED,
        weight="bold",
        va="bottom",
    )
    save(fig, output)


ROBOTWIN_MODEL_META = {
    "efficient_wam_rt_robotwin": ("Efficient-WAM-RT", "decoded future video"),
    "fastwam_robotwin": ("FastWAM", "action-only at inference"),
    "lingbot_va_robotwin": ("LingBot-VA", "predicted latent retained"),
}


def robotwin_progression(
    pilots: list[dict[str, Any]], output: Path, square: bool
) -> None:
    by_model = {pilot["model_id"]: pilot for pilot in pilots}
    fig = plt.figure(figsize=(12, 12) if square else (16, 9))
    draw_header(
        fig,
        "How far did each command get?",
        "Standardized RoboTwin direct-command pilot · three exact scene pairs per model · counts show episodes reaching each observable stage · success requires released placement inside the requested relation region",
        square,
    )
    if square:
        positions = [
            (0.22, 0.65, 0.70, 0.17),
            (0.22, 0.40, 0.70, 0.17),
            (0.22, 0.15, 0.70, 0.17),
        ]
    else:
        positions = [(0.055 + index * 0.315, 0.23, 0.285, 0.55) for index in range(3)]
    stages = ["Started", "Verified\npickup", "Entered requested\nregion", "Released there\n(success)"]

    for axis, model_id in zip(positions, ROBOTWIN_MODEL_META, strict=True):
        plot = fig.add_axes(axis)
        pilot = by_model[model_id]
        label, future_note = ROBOTWIN_MODEL_META[model_id]
        for direction, color, offset in (("left", LEFT, 0.04), ("right", RIGHT, -0.04)):
            episodes = [
                episode
                for episode in pilot["episodes"]
                if episode["requested_relation"] == direction
            ]
            counts = [
                len(episodes),
                sum(episode["verified_pickup_proxy"] for episode in episodes),
                sum(episode["ever_entered_requested_region"] for episode in episodes),
                sum(episode["requested_success"] for episode in episodes),
            ]
            xs = np.arange(4, dtype=float) + offset
            plot.plot(xs, counts, color=color, linewidth=2.2, marker="o", markersize=7, zorder=3)
            for x_value, count in zip(xs, counts, strict=True):
                plot.text(
                    x_value,
                    count + 0.13,
                    str(count),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    weight="bold",
                    color=TEXT,
                )
            plot.text(
                xs[-1] + 0.08,
                counts[-1] + (0.03 if direction == "left" else -0.03),
                direction.upper(),
                ha="left",
                va="center",
                fontsize=8.5,
                weight="bold",
                color=TEXT,
            )
        plot.set_xlim(-0.18, 3.52)
        plot.set_ylim(-0.15, 3.5)
        plot.set_xticks(range(4))
        plot.set_xticklabels(stages, fontsize=8.5)
        plot.set_yticks([0, 1, 2, 3])
        plot.set_ylabel("episodes (of 3)", fontsize=8.5)
        plot.grid(axis="y", color=GRID, linewidth=0.7)
        plot.spines[["top", "right"]].set_visible(False)
        plot.spines[["left", "bottom"]].set_color(GRID)
        plot.set_title(f"{label}  ·  {future_note}", loc="left", fontsize=11.5, pad=8)

    legend = [
        Line2D([0], [0], color=LEFT, marker="o", label="Prompt asks LEFT", linewidth=2),
        Line2D([0], [0], color=RIGHT, marker="o", label="Prompt asks RIGHT", linewidth=2),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.075 if not square else 0.045),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    takeaway = (
        "The asymmetry is not inactivity: every Efficient-WAM run picked up the object, "
        "and all three LingBot LEFT prompts completed. Across all three checkpoints, "
        "RIGHT produced 0/9 released requested placements."
    )
    fig.text(
        0.43 if not square else 0.05,
        0.087 if not square else 0.015,
        "\n".join(textwrap.wrap(takeaway, 92 if not square else 110)),
        fontsize=9.7,
        color=MUTED,
        weight="bold",
        va="center" if not square else "bottom",
    )
    save(fig, output)


def robotwin_paired_endpoints(
    pilots: list[dict[str, Any]], output: Path, square: bool
) -> None:
    by_model = {pilot["model_id"]: pilot for pilot in pilots}
    fig = plt.figure(figsize=(12, 12) if square else (16, 9))
    draw_header(
        fig,
        "Did changing LEFT to RIGHT redirect the endpoint?",
        "Each line is one exact scene and seed pair. The two markers are final target-relative x after changing only the direction word. Circles passed the full relation-and-release checker; crosses failed it.",
        square,
    )
    axis = fig.add_axes(
        (0.31, 0.15, 0.63, 0.69)
        if square
        else (0.24, 0.19, 0.71, 0.65)
    )
    axis.axvspan(-0.20, -0.08, color=LEFT, alpha=0.085, zorder=0)
    axis.axvspan(0.08, 0.20, color=RIGHT, alpha=0.075, zorder=0)
    axis.axvline(0, color=TEXT, linewidth=1.0, alpha=0.75)
    axis.text(-0.14, 8.55, "LEFT lateral\nsuccess band", ha="center", fontsize=8, color=MUTED)
    axis.text(0.14, 8.55, "RIGHT lateral\nsuccess band", ha="center", fontsize=8, color=MUTED)

    y_labels: list[str] = []
    y_positions: list[int] = []
    row_index = 0
    short_names = {
        "efficient_wam_rt_robotwin": "Efficient-WAM",
        "fastwam_robotwin": "FastWAM",
        "lingbot_va_robotwin": "LingBot-VA",
    }
    for model_index, model_id in enumerate(ROBOTWIN_MODEL_META):
        episodes_by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for episode in by_model[model_id]["episodes"]:
            episodes_by_pair[episode["pair_id"]][episode["requested_relation"]] = episode
        for pair_number, pair_id in enumerate(sorted(episodes_by_pair), start=1):
            y = 8 - row_index
            pair = episodes_by_pair[pair_id]
            left_episode, right_episode = pair["left"], pair["right"]
            left_x = float(left_episode["final_dx_m"])
            right_x = float(right_episode["final_dx_m"])
            axis.plot([left_x, right_x], [y + 0.10, y - 0.10], color="#BFC8CE", linewidth=1.2, zorder=1)
            for x_value, episode, dy, color in (
                (left_x, left_episode, 0.10, LEFT),
                (right_x, right_episode, -0.10, RIGHT),
            ):
                marker = "o" if episode["requested_success"] else "X"
                axis.scatter(
                    [x_value],
                    [y + dy],
                    s=58,
                    marker=marker,
                    facecolor=color,
                    edgecolor=CARD if marker == "o" else color,
                    linewidth=1.1,
                    zorder=3,
                )
                label_x = x_value + (0.012 if x_value < 0.22 else -0.012)
                axis.text(
                    label_x,
                    y + dy,
                    f"{x_value:+.3f}",
                    ha="left" if x_value < 0.22 else "right",
                    va="center",
                    fontsize=7.7,
                    color=TEXT,
                )
            y_positions.append(y)
            y_labels.append(f"{short_names[model_id]}  ·  scene {pair_number}")
            row_index += 1
        if model_index < 2:
            axis.axhline(8 - row_index + 0.5, color=GRID, linewidth=1.1)

    axis.set_xlim(-0.31, 0.31)
    axis.set_ylim(-0.55, 8.85)
    axis.set_yticks(y_positions)
    axis.set_yticklabels(y_labels, fontsize=9)
    axis.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    axis.set_xticklabels(["−.3", "−.2", "−.1", "TARGET", "+.1", "+.2", "+.3"])
    axis.set_xlabel("final target-relative x (m)  ·  negative is robot LEFT", fontsize=10)
    axis.grid(axis="x", color=GRID, linewidth=0.65)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(axis="y", length=0, pad=8)
    legend = [
        Line2D([0], [0], marker="o", color=LEFT, markerfacecolor=LEFT, label="Asked LEFT · success", linewidth=0),
        Line2D([0], [0], marker="X", color=LEFT, label="Asked LEFT · failure", linewidth=0),
        Line2D([0], [0], marker="o", color=RIGHT, markerfacecolor=RIGHT, label="Asked RIGHT · success", linewidth=0),
        Line2D([0], [0], marker="X", color=RIGHT, label="Asked RIGHT · failure", linewidth=0),
    ]
    fig.legend(
        handles=legend,
        loc="lower left",
        bbox_to_anchor=(0.05, 0.045),
        ncol=2 if square else 4,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.31 if square else 0.24,
        0.105 if square else 0.125,
        "The shaded bands show only the lateral slice. A marker is a success only when distance, y-offset, and release also pass.",
        fontsize=8.8,
        color=MUTED,
    )
    save(fig, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/figures"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    configure()

    summaries_path = workspace / "artifacts/vla_wam_shared_v1/final_evidence/group_summaries.csv"
    episodes_path = workspace / "artifacts/vla_wam_shared_v1/final_evidence/episodes.csv"
    protocol_path = workspace / "artifacts/vla_wam_shared_v2/protocol.json"
    efficient_result_path = (
        workspace
        / "artifacts/vla_wam_shared_v2/pilot/results/efficient_wam_rt_direct_gate.json"
    )
    pilot_result_paths = [
        efficient_result_path,
        workspace / "artifacts/vla_wam_shared_v2/pilot/results/fastwam_direct_gate.json",
        workspace / "artifacts/vla_wam_shared_v2/pilot/results/lingbot_va_direct_gate.json",
    ]
    with summaries_path.open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with episodes_path.open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    with efficient_result_path.open() as handle:
        efficient_result = json.load(handle)
    pilot_results = [json.loads(path.read_text()) for path in pilot_result_paths]
    if len(summaries) != 16 or len(episodes) != 160:
        raise RuntimeError("Reader figures require all 16 group cells and all 160 episodes")

    outputs = {
        "prompt_semantics_landscape": output_dir / "prompt_semantics_1600x900.png",
        "prompt_semantics_square": output_dir / "prompt_semantics_1200x1200.png",
        "obedience_scorecard_landscape": output_dir / "droid_v1_obedience_scorecard_1600x900.png",
        "obedience_scorecard_square": output_dir / "droid_v1_obedience_scorecard_1200x1200.png",
        "paired_endpoints_landscape": output_dir / "droid_v1_paired_endpoints_1600x900.png",
        "paired_endpoints_square": output_dir / "droid_v1_paired_endpoints_1200x1200.png",
        "efficient_pilot_paths_landscape": output_dir / "efficient_wam_rt_pilot_paths_1600x900.png",
        "efficient_pilot_paths_square": output_dir / "efficient_wam_rt_pilot_paths_1200x1200.png",
        "robotwin_wam_progression_landscape": output_dir / "robotwin_wam_progression_1600x900.png",
        "robotwin_wam_progression_square": output_dir / "robotwin_wam_progression_1200x1200.png",
        "robotwin_wam_paired_endpoints_landscape": output_dir / "robotwin_wam_paired_endpoints_1600x900.png",
        "robotwin_wam_paired_endpoints_square": output_dir / "robotwin_wam_paired_endpoints_1200x1200.png",
    }
    render_jobs = [
        ("prompt semantics landscape", prompt_semantics, outputs["prompt_semantics_landscape"], False),
        ("prompt semantics square", prompt_semantics, outputs["prompt_semantics_square"], True),
        ("obedience scorecard landscape", lambda path, square: scorecard(summaries, path, square), outputs["obedience_scorecard_landscape"], False),
        ("obedience scorecard square", lambda path, square: scorecard(summaries, path, square), outputs["obedience_scorecard_square"], True),
        ("paired endpoints landscape", lambda path, square: endpoint_pairs(episodes, path, square), outputs["paired_endpoints_landscape"], False),
        ("paired endpoints square", lambda path, square: endpoint_pairs(episodes, path, square), outputs["paired_endpoints_square"], True),
        ("Efficient-WAM pilot paths landscape", lambda path, square: efficient_pilot_paths(efficient_result, path, square), outputs["efficient_pilot_paths_landscape"], False),
        ("Efficient-WAM pilot paths square", lambda path, square: efficient_pilot_paths(efficient_result, path, square), outputs["efficient_pilot_paths_square"], True),
        ("RoboTwin WAM progression landscape", lambda path, square: robotwin_progression(pilot_results, path, square), outputs["robotwin_wam_progression_landscape"], False),
        ("RoboTwin WAM progression square", lambda path, square: robotwin_progression(pilot_results, path, square), outputs["robotwin_wam_progression_square"], True),
        ("RoboTwin WAM paired endpoints landscape", lambda path, square: robotwin_paired_endpoints(pilot_results, path, square), outputs["robotwin_wam_paired_endpoints_landscape"], False),
        ("RoboTwin WAM paired endpoints square", lambda path, square: robotwin_paired_endpoints(pilot_results, path, square), outputs["robotwin_wam_paired_endpoints_square"], True),
    ]
    for label, renderer, path, square in render_jobs:
        print(f"Rendering {label}...", flush=True)
        renderer(path, square)
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "protocol_sha256": sha256(protocol_path),
        "source_group_summaries_sha256": sha256(summaries_path),
        "source_episodes_sha256": sha256(episodes_path),
        "source_episode_count": len(episodes),
        "source_group_count": len(summaries),
        "efficient_pilot_result_sha256": sha256(efficient_result_path),
        "robotwin_pilot_result_sha256": {
            path.stem: sha256(path) for path in pilot_result_paths
        },
        "figures": {
            key: {
                "path": str(path.relative_to(workspace)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
        },
        "claim_boundary": "The first six figures explain the frozen prompt design and the two existing DROID reference checkpoints. The remaining six figures contain only the 18 standardized direct-command RoboTwin WAM pilot episodes. Six episodes per model are a base-competence and expansion gate, not stable population rates or a WAM-class estimate.",
    }
    manifest_path = output_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
