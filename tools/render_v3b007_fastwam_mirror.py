#!/usr/bin/env python3
"""Render the compact RoboTwin V3-B007 negative-control figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


BACKGROUND = "#F6F1E8"
INK = "#17232B"
MUTED = "#5E6A70"
GRID = "#D7D0C5"
LEFT = "#B85F35"
RIGHT = "#286EA6"
CONTROL = "#768078"
REFLECTED = "#2D7A63"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def configure() -> None:
    mpl.rcParams.update({
        "figure.facecolor": BACKGROUND,
        "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND,
        "font.family": "Arial",
        "font.size": 11.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlecolor": INK,
        "svg.fonttype": "none",
        "svg.hashsalt": "v3b007-fastwam-robotwin-mirror",
    })


def interaction_label(row: dict) -> str:
    interval = row["mean_bootstrap_95"]
    p_value = row["paired_sign_test"]["p_value"]
    return (
        f"reflected − control: {row['mean_m'] * 100:+.1f} cm "
        f"[95% CI {interval['lower'] * 100:+.1f}, {interval['upper'] * 100:+.1f}] · sign p={p_value:.3g}"
    )


def main() -> None:
    args = parse_args()
    summary_path = args.summary.resolve()
    pairs_path = args.pairs.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    summary = json.loads(summary_path.read_text())
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    if summary.get("status") != "complete_27_matched_seeds_108_valid_episodes" or len(pairs) != 54:
        raise ValueError("V3-B007 result is incomplete")

    configure()
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.4))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.76, wspace=0.38)
    fig.suptitle(
        "A mirrored RoboTwin scene tests whether FastWAM’s directional response follows geometry",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=20.5,
        fontfamily="Georgia",
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.902,
        "One fixed pair03 scene · 27 matched policy seeds · identical LEFT/RIGHT reset within layout · RoboTwin is reported separately from DROID",
        ha="left",
        color=MUTED,
        fontsize=11.7,
    )

    ax = axes[0]
    positions = {("control", "left"): 3.2, ("control", "right"): 2.35, ("position_mirrored", "left"): 1.0, ("position_mirrored", "right"): 0.15}
    for arm in ("control", "position_mirrored"):
        for relation, color in (("left", LEFT), ("right", RIGHT)):
            row = summary["condition_outcomes"][f"{arm}:{relation}"]
            successes, n = row["successes"], row["episodes"]
            p = successes / n
            lower, upper = wilson(successes, n)
            y = positions[(arm, relation)]
            ax.errorbar(p, y, xerr=[[p - lower], [upper - p]], fmt="o", color=color, ecolor=color, markersize=8, capsize=4, linewidth=2)
            ax.text(min(1.03, upper + 0.035), y, f"{successes}/{n}", va="center", fontsize=10.8, color=INK)
    ax.set_yticks([2.78, 0.58], ["Control", "Position reflected"], fontweight="bold")
    ax.set_xlim(-0.03, 1.12)
    ax.set_ylim(-0.45, 3.75)
    ax.set_xticks(np.linspace(0, 1, 5), ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("Requested-relation success")
    ax.set_title("A  Binary outcome", loc="left", fontfamily="Georgia", fontsize=16, fontweight="bold", pad=12)
    ax.text(0.02, -0.24, "● LEFT prompt", transform=ax.transAxes, color=LEFT, fontsize=10.5)
    ax.text(0.40, -0.24, "● RIGHT prompt", transform=ax.transAxes, color=RIGHT, fontsize=10.5)

    by_seed = {seed: {} for seed in range(9900, 9927)}
    for row in pairs:
        by_seed[row["seed"]][row["arm"]] = row
    for panel, key, title, x_label, interaction_key in (
        (axes[1], "endpoint_redirection_left_minus_right_m", "B  Endpoint redirection", "LEFT endpoint − RIGHT endpoint (m)", "endpoint_redirection_interaction"),
        (axes[2], "right_minus_left_requested_side_depth_m", "C  Side-depth asymmetry", "RIGHT depth − LEFT depth (m)", "requested_side_depth_interaction"),
    ):
        rng = np.random.default_rng(7007)
        for seed in range(9900, 9927):
            control = by_seed[seed]["control"][key]
            reflected = by_seed[seed]["position_mirrored"][key]
            jitter = float(rng.uniform(-0.045, 0.045))
            panel.plot([control, reflected], [1 + jitter, 0 + jitter], color="#AEB5B0", alpha=0.42, linewidth=0.8, zorder=1)
            panel.scatter(control, 1 + jitter, color=CONTROL, s=25, alpha=0.78, zorder=2)
            panel.scatter(reflected, 0 + jitter, color=REFLECTED, s=25, alpha=0.78, zorder=2)
        panel.axvline(0, color=INK, linewidth=1.05, alpha=0.65)
        panel.set_yticks([1, 0], ["Control", "Position reflected"], fontweight="bold")
        panel.set_ylim(-0.42, 1.42)
        panel.grid(axis="x", color=GRID, linewidth=0.8)
        panel.set_xlabel(x_label)
        panel.set_title(title, loc="left", fontfamily="Georgia", fontsize=16, fontweight="bold", pad=12)
        panel.text(0.0, -0.24, interaction_label(summary["full_sample_primary"][interaction_key]), transform=panel.transAxes, ha="left", fontsize=9.5, color=MUTED)

    for ax in axes:
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.text(
        0.055,
        0.045,
        "Exact prompts: “Put the small woodenblock to the left of the red playingcards box.” / “…to the right…”  ·  Dots are matched seeds; lines connect layouts for the same policy seed.",
        ha="left",
        color=MUTED,
        fontsize=10.1,
    )

    stem = output / "fastwam_v3b007_robotwin_mirror"
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.12, metadata={"Date": None})
    fig.savefig(png, dpi=220, bbox_inches="tight", pad_inches=0.12, metadata={"Software": "steerable V3-B007 renderer"})
    plt.close(fig)
    manifest = {
        "schema_version": "vla-wam-shared-v3b007-figure-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "arena": "robotwin",
        "source_summary": record(summary_path),
        "source_pairs": record(pairs_path),
        "figures": [record(svg), record(png)],
        "caption": "FastWAM RoboTwin position-reflection negative control. Binary requested-relation success is shown with Wilson 95% intervals; continuous panels retain every matched seed and report the reflected-minus-control interaction. RoboTwin is not pooled with DROID.",
    }
    manifest_path = output / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": record(manifest_path), "figures": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
