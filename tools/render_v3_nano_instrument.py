#!/usr/bin/env python3
"""Render Figure 1: why requested-side depth is the sensitive Nano instrument."""

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
SUMMARY = ROOT / "artifacts/vla_wam_shared_v3/results/cosmos3_nano_policy_droid_phase_a_summary.json"
COVERAGE = ROOT / "artifacts/vla_wam_shared_v3/measurement_coverage_audit.json"
OUTPUT_DIR = ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures"

BACKGROUND = "#F6F1E8"
INK = "#17232B"
MUTED = "#5E6A70"
GRID = "#D7D0C5"
LEFT = "#B85F35"
RIGHT = "#286EA6"
TEAL = "#2D7A63"
SOFT_NEGATIVE = "#F3E6DF"
SOFT_POSITIVE = "#E5EFEA"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_path(path)}


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
            "svg.hashsalt": "vla-wam-v3-nano-instrument",
        }
    )


def metric_counts(summary: dict[str, Any]) -> list[dict[str, Any]]:
    left = summary["directional"]["left"]
    right = summary["directional"]["right"]
    rows = [
        {
            "label": "Full task correct",
            "definition": "pickup + transport + sustained requested cone + detached release",
            "left": (int(left["successes"]), int(left["trials"])),
            "right": (int(right["successes"]), int(right["trials"])),
        },
        {
            "label": "Verified pickup",
            "definition": "three-sample lift of at least 3 cm",
            "left": (int(left["verified_pickup_counts"]["observed"]), int(left["trials"])),
            "right": (int(right["verified_pickup_counts"]["observed"]), int(right["trials"])),
        },
        {
            "label": "Detached release",
            "definition": "gripper detached at the final state",
            "left": (int(left["detached_release_counts"]["true"]), int(left["trials"])),
            "right": (int(right["detached_release_counts"]["true"]), int(right["trials"])),
        },
    ]
    return rows


def paired_margin_gaps(summary: dict[str, Any]) -> np.ndarray:
    gaps = []
    for pair in summary["pairs"]:
        # +Y is robot LEFT. Requested margin is +Y for LEFT and -Y for RIGHT.
        left_margin = float(pair["left_raw_robot_y_m"])
        right_margin = -float(pair["right_raw_robot_y_m"])
        gaps.append(right_margin - left_margin)
    values = np.asarray(gaps, dtype=float)
    if values.shape != (27,) or not np.isfinite(values).all():
        raise ValueError("expected 27 finite Nano requested-margin gaps")
    return values


def bootstrap_mean_interval(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(110519)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def draw_binary_panel(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    y = np.arange(len(rows), dtype=float)[::-1]
    for index, row in enumerate(rows):
        base_y = y[index]
        for direction, delta, color in (("left", 0.14, LEFT), ("right", -0.14, RIGHT)):
            successes, trials = row[direction]
            point, lower, upper = wilson(successes, trials)
            ax.errorbar(
                point * 100,
                base_y + delta,
                xerr=np.array(
                    [
                        [max(0.0, (point - lower) * 100)],
                        [max(0.0, (upper - point) * 100)],
                    ]
                ),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=8.3,
                markeredgecolor=BACKGROUND,
                markeredgewidth=1.1,
                elinewidth=2.2,
                capsize=4.2,
                capthick=1.7,
                zorder=3,
            )
            ax.text(min(103.5, upper * 100 + 1.5), base_y + delta, f"{successes}/{trials}", va="center", fontsize=10.8)
        ax.text(0.0, base_y - 0.39, row["definition"], fontsize=9.4, color=MUTED, va="center")

    ax.set_yticks(y, [row["label"] for row in rows], fontsize=12.2, fontweight="bold")
    ax.set_xlim(-1.5, 112)
    ax.set_ylim(-0.72, len(rows) - 0.48)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_xlabel("Episodes meeting the named binary gate", labelpad=10)
    ax.set_title("Three binary gates are near ceiling", loc="left", pad=15, fontfamily="Georgia", fontsize=18, fontweight="bold")
    ax.text(0.0, 1.015, "Marginal proportions with 95% Wilson intervals", transform=ax.transAxes, color=MUTED, fontsize=10.2)


def draw_margin_panel(ax: plt.Axes, values_m: np.ndarray, audit: dict[str, Any]) -> None:
    values_cm = values_m * 100.0
    lower_m, upper_m = bootstrap_mean_interval(values_m)
    mean_cm = float(values_cm.mean())
    rng = np.random.default_rng(3941)
    y = rng.uniform(-0.16, 0.16, len(values_cm))

    minimum = min(-12.0, math.floor(values_cm.min() / 5.0) * 5.0 - 2.5)
    maximum = max(38.0, math.ceil(values_cm.max() / 5.0) * 5.0 + 2.5)
    ax.axvspan(minimum, 0, color=SOFT_NEGATIVE, alpha=0.9, zorder=0)
    ax.axvspan(0, maximum, color=SOFT_POSITIVE, alpha=0.9, zorder=0)
    ax.axvline(0, color=INK, linewidth=1.2, zorder=1)
    ax.scatter(values_cm, y, s=43, color="#7E8986", edgecolor=BACKGROUND, linewidth=0.9, alpha=0.9, zorder=3)
    ax.errorbar(
        mean_cm,
        -0.46,
        xerr=np.array([[mean_cm - lower_m * 100.0], [upper_m * 100.0 - mean_cm]]),
        fmt="D",
        color=TEAL,
        ecolor=TEAL,
        markeredgecolor=BACKGROUND,
        markeredgewidth=1.0,
        markersize=9,
        elinewidth=3.0,
        capsize=5.0,
        capthick=2.0,
        zorder=4,
    )
    ax.text(mean_cm, -0.67, f"mean +{mean_cm:.1f} cm\n95% bootstrap CI [{lower_m*100:.1f}, {upper_m*100:.1f}]", ha="center", va="top", fontsize=10.8, color=INK)
    ax.text(minimum + 1.0, 0.35, "LEFT finishes deeper", ha="left", va="center", fontsize=10.1, color=MUTED)
    ax.text(maximum - 1.0, 0.35, "RIGHT finishes deeper", ha="right", va="center", fontsize=10.1, color=MUTED)

    ax.set_xlim(minimum, maximum)
    ax.set_ylim(-1.08, 0.57)
    ax.set_yticks([])
    ax.set_xlabel("Per-seed RIGHT minus LEFT requested-side depth (cm)", labelpad=10)
    ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("Continuous depth exposes the asymmetry", loc="left", pad=15, fontfamily="Georgia", fontsize=18, fontweight="bold")
    ax.text(
        0.0,
        1.015,
        "Every dot is one matched seed; failures are retained",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10.2,
    )
    positive, ties, negative = audit["positive_zero_negative_pair_counts"]
    ax.text(
        0.0,
        -0.29,
        f"Mean margin: LEFT {audit['left_mean_requested_side_margin_m']*100:.1f} cm · RIGHT {audit['right_mean_requested_side_margin_m']*100:.1f} cm\n"
        f"Paired signs: {positive} positive, {negative} negative, {ties} ties · exact two-sided p = {audit['exact_two_sided_sign_test_p_excluding_ties']:.4g}",
        transform=ax.transAxes,
        fontsize=10.3,
        color=INK,
        linespacing=1.45,
    )


def render(summary: dict[str, Any], coverage: dict[str, Any]) -> list[Path]:
    configure_style()
    audit = coverage["nano_phase_a_margin_sensitivity_reproduction"]
    values = paired_margin_gaps(summary)
    if not math.isclose(float(values.mean()), float(audit["right_minus_left_mean_margin_gap_m"]), abs_tol=1e-12):
        raise ValueError("paired margin gaps do not reproduce the committed audit")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.5), gridspec_kw={"width_ratios": [1.02, 0.98]})
    fig.subplots_adjust(left=0.17, right=0.965, top=0.69, bottom=0.25, wspace=0.23)
    fig.suptitle(
        "Near-ceiling outcomes can hide a directional placement asymmetry",
        x=0.055,
        y=0.965,
        ha="left",
        fontfamily="Georgia",
        fontsize=27,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.906,
        "Cosmos3 Nano Policy DROID · 27 matched seeds · identical reset within each LEFT/RIGHT pair · all behavioral failures retained",
        ha="left",
        color=MUTED,
        fontsize=12.0,
    )
    fig.text(
        0.055,
        0.856,
        "LEFT prompt: “Put the Rubik’s cube to the left of the bowl.”     RIGHT prompt: “Put the Rubik’s cube to the right of the bowl.”",
        ha="left",
        color=INK,
        fontsize=11.5,
    )
    fig.text(0.055, 0.808, "●", color=LEFT, fontsize=14, va="center")
    fig.text(0.071, 0.808, "Prompt asks LEFT", color=INK, fontsize=11.0, va="center")
    fig.text(0.188, 0.808, "●", color=RIGHT, fontsize=14, va="center")
    fig.text(0.204, 0.808, "Prompt asks RIGHT", color=INK, fontsize=11.0, va="center")

    draw_binary_panel(axes[0], metric_counts(summary))
    draw_margin_panel(axes[1], values, audit)
    fig.text(
        0.055,
        0.035,
        "Instrument choice changes the inference: binary gates are saturated, while signed requested-side depth preserves within-pair variation for every episode. Depth measures placement quality, not task success.",
        ha="left",
        color=INK,
        fontsize=10.8,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "figure1_nano_instrument_sensitivity"
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.14, metadata={"Date": None})
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.14, metadata={"Software": "steerable V3 Nano instrument renderer"})
    plt.close(fig)
    return [svg, png]


def main() -> None:
    summary = read_json(SUMMARY)
    coverage = read_json(COVERAGE)
    outputs = render(summary, coverage)
    manifest = {
        "schema_version": "vla-wam-shared-v3-nano-instrument-figure-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_retrospective_visualization_no_new_inference",
        "inputs": [file_record(SUMMARY), file_record(COVERAGE)],
        "renderer": file_record(Path(__file__).resolve()),
        "outputs": [file_record(path) for path in outputs],
        "bootstrap": {"replicates": 20_000, "seed": 110_519, "unit": "matched seed"},
        "claim_boundary": "The binary gates and requested-side depth are distinct diagnostics. Near-ceiling binary counts do not establish equivalence, and requested-side depth does not replace the frozen task predicate.",
    }
    manifest_path = OUTPUT_DIR / "figure1_nano_instrument_sensitivity.manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(file_record(manifest_path), indent=2))


if __name__ == "__main__":
    main()
