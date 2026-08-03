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

import matplotlib.pyplot as plt
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
    rendered_subtitle = (
        "\n".join(textwrap.wrap(subtitle, 128)) if square else subtitle
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
    with summaries_path.open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with episodes_path.open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    if len(summaries) != 16 or len(episodes) != 160:
        raise RuntimeError("Reader figures require all 16 group cells and all 160 episodes")

    outputs = {
        "prompt_semantics_landscape": output_dir / "prompt_semantics_1600x900.png",
        "prompt_semantics_square": output_dir / "prompt_semantics_1200x1200.png",
        "obedience_scorecard_landscape": output_dir / "droid_v1_obedience_scorecard_1600x900.png",
        "obedience_scorecard_square": output_dir / "droid_v1_obedience_scorecard_1200x1200.png",
        "paired_endpoints_landscape": output_dir / "droid_v1_paired_endpoints_1600x900.png",
        "paired_endpoints_square": output_dir / "droid_v1_paired_endpoints_1200x1200.png",
    }
    prompt_semantics(outputs["prompt_semantics_landscape"], square=False)
    prompt_semantics(outputs["prompt_semantics_square"], square=True)
    scorecard(summaries, outputs["obedience_scorecard_landscape"], square=False)
    scorecard(summaries, outputs["obedience_scorecard_square"], square=True)
    endpoint_pairs(episodes, outputs["paired_endpoints_landscape"], square=False)
    endpoint_pairs(episodes, outputs["paired_endpoints_square"], square=True)
    manifest = {
        "schema_version": "1.0.0",
        "status": "complete",
        "protocol_sha256": sha256(protocol_path),
        "source_group_summaries_sha256": sha256(summaries_path),
        "source_episodes_sha256": sha256(episodes_path),
        "source_episode_count": len(episodes),
        "source_group_count": len(summaries),
        "figures": {
            key: {
                "path": str(path.relative_to(workspace)),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for key, path in outputs.items()
        },
        "claim_boundary": "These six figures explain the frozen prompt design and the two existing DROID reference checkpoints. They contain no standardized v2 expansion-model outcome.",
    }
    manifest_path = output_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
