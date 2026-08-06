#!/usr/bin/env python3
"""Render the V3 A1 failure split and B1 gap-versus-competence figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism/figures"
FAILURE_REPORT = ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis/failure_mode_split_report.json"
GAP_REPORT = ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism/gap_vs_competence_report.json"

BACKGROUND = "#F6F1E8"
INK = "#17232B"
MUTED = "#5E6A70"
GRID = "#D7D0C5"
LEFT = "#B85F35"
RIGHT = "#286EA6"
COLORS = {
    "correct": "#2D7A63",
    "pick_failed": "#D2A24D",
    "transport_failed": "#B96B56",
    "wrong_side": "#77598D",
    "release_failed": "#7A858D",
}
LABELS = {
    "correct": "Correct",
    "pick_failed": "Pick failed",
    "transport_failed": "Transport failed",
    "wrong_side": "Wrong side",
    "release_failed": "Release failed",
}
ORDER = tuple(COLORS)


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
            "ytick.color": MUTED,
            "axes.titlecolor": INK,
            "svg.fonttype": "none",
            "svg.hashsalt": "vla-wam-v3-mechanism-figures",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.14, metadata={"Date": None})
    fig.savefig(
        png,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.14,
        metadata={"Software": "steerable V3 mechanism figure renderer"},
    )
    plt.close(fig)
    return [svg, png]


def render_failure_taxonomy(report: dict[str, Any], output_dir: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 1, figsize=(13.2, 10.2), sharex=True)
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.18, right=0.985, hspace=0.84)
    fig.suptitle(
        "Direction changes more than the failure rate for DreamZero",
        x=0.055,
        y=0.972,
        ha="left",
        fontfamily="Georgia",
        fontsize=27,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.928,
        "Each bar is one complete 27-episode direction cohort. The exact test compares the composition of failures only; correct episodes are retained here to show competence.",
        ha="left",
        fontsize=12.4,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.886,
        "LEFT prompt: “Put the Rubik’s cube to the left of the bowl.”     RIGHT prompt: “Put the Rubik’s cube to the right of the bowl.”",
        ha="left",
        fontsize=11.8,
        color=INK,
    )

    for ax, result in zip(axes, report["results"]):
        ax.set_xlim(0, 27)
        ax.set_ylim(-0.70, 1.68)
        ax.set_yticks([1.0, 0.0], ["LEFT", "RIGHT"], fontsize=12.5, fontweight="bold")
        ax.set_xticks([0, 5, 10, 15, 20, 25, 27])
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)

        for y, direction in ((1.0, "left"), (0.0, "right")):
            counts = result["directions"][direction]["raw_counts"]
            start = 0
            for category in ORDER:
                count = int(counts[category])
                if count:
                    ax.barh(y, count, left=start, height=0.58, color=COLORS[category], edgecolor=BACKGROUND, linewidth=1.1)
                    if count >= 2:
                        ax.text(start + count / 2, y, str(count), ha="center", va="center", color="white", fontsize=11.3, fontweight="bold")
                    elif count == 1:
                        ax.text(start + 0.5, y, "1", ha="center", va="center", color="white", fontsize=9.3, fontweight="bold")
                start += count
            correct = int(counts["correct"])
            ax.text(27.18, y, f"{correct}/27 correct", ha="left", va="center", fontsize=11.2, color=INK, clip_on=False)

        p_value = result["failure_only_exact_test"]["p_value"]
        left_failures = result["directions"]["left"]["failure_count"]
        right_failures = result["directions"]["right"]["failure_count"]
        detected = result["failure_only_exact_test"]["failure_shape_difference_detected_at_alpha"]
        p_label = f"p = {p_value:.5f}" if p_value < 0.01 else f"p = {p_value:.3f}"
        conclusion = "failure composition differs by direction" if detected else "no difference detected; sparse smaller row"
        display_name = {
            "pi0.5 current stack DROID": "π0.5 DROID",
            "DreamZero DROID action guidance s=2": "DreamZero DROID · action guidance s=2",
            "Cosmos3 Edge Policy DROID": "Cosmos3 Edge Policy DROID",
        }.get(result["display_name"], result["display_name"])
        ax.set_title(display_name, loc="left", pad=10, fontsize=17, fontfamily="Georgia", fontweight="bold")
        ax.text(
            0,
            -0.55,
            f"Failures only: LEFT n={left_failures}, RIGHT n={right_failures} · Fisher–Freeman–Halton {p_label} · {conclusion}",
            fontsize=10.8,
            color=INK if detected else MUTED,
            ha="left",
            va="center",
        )

    axes[-1].set_xlabel("Episodes in the 27-seed direction cohort", fontsize=12)
    legend = [Patch(facecolor=COLORS[key], label=LABELS[key]) for key in ORDER]
    fig.legend(
        handles=legend,
        ncol=5,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.052, 0.035),
        fontsize=11.2,
        handlelength=1.2,
        columnspacing=1.7,
    )
    fig.text(
        0.985,
        0.045,
        "Exact, probability-ordered two-sided tests; cohorts are not pooled.",
        ha="right",
        fontsize=10.3,
        color=MUTED,
    )
    return save_figure(fig, output_dir / "figure6_failure_taxonomy_by_direction")


def render_gap_vs_competence(report: dict[str, Any], output_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(12.7, 8.0))
    fig.subplots_adjust(left=0.115, right=0.965, top=0.81, bottom=0.17)
    fig.suptitle(
        "Competence limits the possible gap; it does not explain its sign",
        x=0.055,
        y=0.965,
        ha="left",
        fontfamily="Georgia",
        fontsize=27,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.910,
        "Five separate 54-episode DROID cohorts · 27 matched seeds per checkpoint · directional gap = RIGHT success rate − LEFT success rate",
        ha="left",
        color=MUTED,
        fontsize=12.4,
    )

    x_curve = np.linspace(0.0, 1.0, 501)
    envelope = 2.0 * np.minimum(x_curve, 1.0 - x_curve)
    ax.fill_between(x_curve * 100.0, -envelope * 100.0, envelope * 100.0, color="#E5E8E3", alpha=0.82, zorder=0)
    ax.plot(x_curve * 100.0, envelope * 100.0, color="#89938E", linewidth=1.4, zorder=1)
    ax.plot(x_curve * 100.0, -envelope * 100.0, color="#89938E", linewidth=1.4, zorder=1)
    ax.axhline(0, color=INK, linewidth=1.1, zorder=1)
    ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)

    offsets = {
        "π0.5": (2.2, 5.0, "left"),
        "DreamZero": (-2.0, 5.0, "right"),
        "Cosmos3 Edge": (-2.0, 5.0, "right"),
        "Cosmos3 Nano": (-2.0, -8.0, "right"),
        "GR00T N1.7": (2.0, -8.0, "left"),
    }
    colors = {
        "π0.5": "#72568A",
        "GR00T N1.7": "#717A80",
        "Cosmos3 Edge": "#2D7A63",
        "Cosmos3 Nano": "#2D6EA3",
        "DreamZero": "#B65D36",
    }
    for row in report["results"]:
        x = 100.0 * row["overall_success_rate"]
        y = 100.0 * row["directional_gap_right_minus_left"]
        name = row["display_name"]
        ax.scatter(x, y, s=125, color=colors[name], edgecolor=BACKGROUND, linewidth=1.8, zorder=3)
        dx, dy, align = offsets[name]
        counts = f"L {row['left_successes']}/27 · R {row['right_successes']}/27"
        ax.annotate(
            f"{name}\n{counts}",
            (x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            ha=align,
            va="bottom" if dy >= 0 else "top",
            fontsize=11.1,
            color=INK,
            linespacing=1.28,
        )

    signed = report["descriptive_associations"]["competence_vs_signed_gap"]["spearman"]
    absolute = report["descriptive_associations"]["competence_vs_absolute_gap"]["spearman"]
    ax.text(
        3,
        86,
        "Gray region: maximum binary gap allowed by the overall success rate",
        fontsize=10.8,
        color=MUTED,
        ha="left",
    )
    ax.text(
        97,
        -88,
        f"Signed gap: Spearman ρ = {signed['coefficient']:.1f}, exact p = {signed['two_sided_exact_permutation_p']:.2f}\n"
        f"Absolute gap: ρ = {absolute['coefficient']:.1f}, exact p = {absolute['two_sided_exact_permutation_p']:.2f}",
        ha="right",
        va="bottom",
        fontsize=10.8,
        color=MUTED,
        linespacing=1.35,
    )

    ax.set_xlim(0, 100)
    ax.set_ylim(-100, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["0%", "20%", "40%", "60%", "80%", "100%"])
    ax.set_yticks([-100, -50, 0, 50, 100], ["−100", "−50", "0", "+50", "+100"])
    ax.set_xlabel("Overall task success across LEFT and RIGHT episodes", fontsize=12.5, labelpad=11)
    ax.set_ylabel("Directional success gap (percentage points)", fontsize=12.5, labelpad=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    fig.text(
        0.055,
        0.045,
        "The relationship is not monotonic. Floor and ceiling compress the observable gap; intermediate competence permits a gap but does not determine its direction or cause.",
        ha="left",
        color=INK,
        fontsize=11.7,
    )
    return save_figure(fig, output_dir / "figure4_gap_vs_competence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("output directory must be inside the repository") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    configure_style()
    failure = json.loads(FAILURE_REPORT.read_text())
    gap = json.loads(GAP_REPORT.read_text())
    if len(failure.get("results", [])) != 3:
        raise ValueError("failure report must contain three checkpoint cohorts")
    if len(gap.get("results", [])) != 5:
        raise ValueError("gap report must contain five checkpoint cohorts")

    outputs = render_failure_taxonomy(failure, output_dir)
    outputs += render_gap_vs_competence(gap, output_dir)
    manifest = {
        "schema_version": "vla-wam-shared-v3-mechanism-figure-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_retrospective_analysis_no_new_inference",
        "inputs": [file_record(FAILURE_REPORT), file_record(GAP_REPORT)],
        "renderer": file_record(Path(__file__).resolve()),
        "outputs": [file_record(path) for path in outputs],
        "figure_claim_boundaries": {
            "figure4_gap_vs_competence": "Descriptive association across five fixed checkpoints; no population-level model claim.",
            "figure6_failure_taxonomy_by_direction": "Exact tests compare failure-only marginal compositions within each checkpoint; nonsignificant results do not establish equivalence.",
        },
    }
    manifest_path = output_dir / "mechanism_figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(file_record(manifest_path), indent=2))


if __name__ == "__main__":
    main()
