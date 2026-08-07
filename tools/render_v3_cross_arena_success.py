#!/usr/bin/env python3
"""Render the V3 cross-arena directional-success figure.

The two arenas are deliberately faceted and never pooled. Every mark is a
direction-specific binomial proportion with a Wilson 95% interval; matched
pairing is part of the design but does not change the marginal interval.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "artifacts/vla_wam_shared_v3/results"
OUTPUT_DIR = ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures"

BACKGROUND = "#F6F1E8"
INK = "#17232B"
MUTED = "#5E6A70"
GRID = "#D7D0C5"
LEFT = "#B85F35"
RIGHT = "#286EA6"

DROID = (
    ("pi05_current_stack_droid_phase_a_summary.json", "π0.5", "VLA"),
    ("groot_n17_droid_phase_a_summary.json", "GR00T N1.7", "VLA"),
    ("cosmos3_edge_policy_droid_phase_a_summary.json", "Cosmos3 Edge", "WAM"),
    ("cosmos3_nano_policy_droid_phase_a_summary.json", "Cosmos3 Nano", "WAM"),
    ("dreamzero_droid_action_cfg_phase_a_summary.json", "DreamZero", "WAM"),
)

ROBOTWIN = (
    ("efficient_wam_rt_robotwin_phase_a_summary.json", "Efficient-WAM-RT", "WAM"),
    ("fastwam_robotwin_phase_a_summary.json", "FastWAM", "WAM"),
    ("lingbot_va_robotwin_phase_a_summary.json", "LingBot-VA", "WAM"),
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def direction_row(block: dict[str, Any]) -> tuple[int, int]:
    successes = int(block["successes"])
    trials = int(block.get("valid_denominator", block.get("trials", block.get("valid_episodes"))))
    if not (0 <= successes <= trials and trials > 0):
        raise ValueError(f"invalid binomial row: {successes}/{trials}")
    return successes, trials


def load_droid() -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for filename, label, family in DROID:
        path = RESULTS / filename
        data = read_json(path)
        left = direction_row(data["directional"]["left"])
        right = direction_row(data["directional"]["right"])
        rows.append({"label": label, "family": family, "left": left, "right": right, "pairs": len(data["pairs"])})
        paths.append(path)
    return rows, paths


def load_robotwin() -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    for filename, label, family in ROBOTWIN:
        path = RESULTS / filename
        data = read_json(path)["v3_primary_results"]
        left = direction_row(data["by_direction"]["left"])
        right = direction_row(data["by_direction"]["right"])
        rows.append({"label": label, "family": family, "left": left, "right": right, "pairs": int(data["matched_pairs"])})
        paths.append(path)
    return rows, paths


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    proportion = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (proportion + z2 / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / trials + z2 / (4.0 * trials * trials)) / denominator
    return proportion, max(0.0, center - half_width), min(1.0, center + half_width)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": BACKGROUND,
            "savefig.facecolor": BACKGROUND,
            "font.family": "Arial",
            "font.size": 12,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "axes.titlecolor": INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "vla-wam-v3-cross-arena-success",
        }
    )


def draw_panel(ax: plt.Axes, rows: list[dict[str, Any]], arena: str, prompt_note: str) -> None:
    y = np.arange(len(rows), dtype=float)[::-1]
    offset = 0.15
    for index, row in enumerate(rows):
        base_y = y[index]
        for direction, delta, color in (("left", offset, LEFT), ("right", -offset, RIGHT)):
            successes, trials = row[direction]
            point, lower, upper = wilson(successes, trials)
            ax.errorbar(
                point * 100.0,
                base_y + delta,
                xerr=np.array([[(point - lower) * 100.0], [(upper - point) * 100.0]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=7.4,
                markeredgecolor=BACKGROUND,
                markeredgewidth=1.0,
                elinewidth=2.0,
                capsize=4.0,
                capthick=1.6,
                zorder=3,
            )
            label_x = min(102.0, upper * 100.0 + 2.0)
            ax.text(label_x, base_y + delta, f"{successes}/{trials}", va="center", ha="left", fontsize=10.4, color=INK)

    labels = [f"{row['label']}  ·  {row['family']}" for row in rows]
    ax.set_yticks(y, labels, fontsize=11.6)
    ax.set_xlim(-2, 113)
    ax.set_ylim(-0.75, len(rows) - 0.25)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_title(arena, loc="left", pad=14, fontfamily="Georgia", fontsize=18, fontweight="bold")
    ax.text(0.0, 1.005, prompt_note, transform=ax.transAxes, ha="left", va="bottom", fontsize=9.8, color=MUTED)
    pair_counts = sorted({row["pairs"] for row in rows})
    pair_note = str(pair_counts[0]) if len(pair_counts) == 1 else ", ".join(str(value) for value in pair_counts)
    ax.text(1.0, -0.19, f"Matched pairs per checkpoint: {pair_note}", transform=ax.transAxes, ha="right", fontsize=9.8, color=MUTED)


def render(droid: list[dict[str, Any]], robotwin: list[dict[str, Any]]) -> list[Path]:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.8), gridspec_kw={"width_ratios": [1.12, 0.88]})
    fig.subplots_adjust(left=0.16, right=0.965, top=0.73, bottom=0.17, wspace=0.27)
    fig.suptitle(
        "Directional task success varies by checkpoint and arena",
        x=0.055,
        y=0.965,
        ha="left",
        fontfamily="Georgia",
        fontsize=28,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.907,
        "Points are direction-specific success rates; bars are 95% Wilson intervals. Counts are printed beside each interval. Arenas use different tasks and success rules and are never pooled.",
        ha="left",
        color=MUTED,
        fontsize=12.2,
    )
    fig.text(0.055, 0.852, "●", color=LEFT, fontsize=14, va="center")
    fig.text(0.071, 0.852, "Prompt asks LEFT", color=INK, fontsize=11.2, va="center")
    fig.text(0.188, 0.852, "●", color=RIGHT, fontsize=14, va="center")
    fig.text(0.204, 0.852, "Prompt asks RIGHT", color=INK, fontsize=11.2, va="center")

    draw_panel(
        axes[0],
        droid,
        "DROID / RoboLab",
        "Exact prompts: “Put the Rubik’s cube to the left/right of the bowl.”",
    )
    draw_panel(
        axes[1],
        robotwin,
        "RoboTwin",
        "Exact scene-specific object nouns; only “left” ↔ “right” changes within a pair.",
    )
    for ax in axes:
        ax.set_xlabel("Episodes satisfying the frozen requested-relation predicate", labelpad=11, fontsize=11.3)

    fig.text(
        0.055,
        0.045,
        "Intervals quantify binomial uncertainty for each marginal direction cohort. Matched-pair contrasts and endpoint diagnostics are reported separately; overlapping intervals are not a paired hypothesis test.",
        ha="left",
        color=INK,
        fontsize=10.8,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    stem = OUTPUT_DIR / "figure5_cross_arena_directional_success"
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.14, metadata={"Date": None})
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.14, metadata={"Software": "steerable V3 cross-arena renderer"})
    plt.close(fig)
    outputs.extend((svg, png))
    return outputs


def main() -> None:
    droid, droid_paths = load_droid()
    robotwin, robotwin_paths = load_robotwin()
    outputs = render(droid, robotwin)
    manifest = {
        "schema_version": "vla-wam-shared-v3-cross-arena-figure-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_retrospective_visualization_no_new_inference",
        "inputs": [file_record(path) for path in droid_paths + robotwin_paths],
        "renderer": file_record(Path(__file__).resolve()),
        "outputs": [file_record(path) for path in outputs],
        "claim_boundary": "DROID/RoboLab and RoboTwin are faceted and never pooled. Wilson intervals are marginal direction-specific summaries, not matched-pair tests.",
    }
    manifest_path = OUTPUT_DIR / "figure5_cross_arena_directional_success.manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(file_record(manifest_path), indent=2))


if __name__ == "__main__":
    main()
