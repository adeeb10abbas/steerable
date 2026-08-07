#!/usr/bin/env python3
"""Render the registered Nano V3-B005 dose-response result.

The renderer consumes only the hash-closed machine-readable analysis report.
It does not reopen simulator state or recompute any behavioral statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results"
    / "nano_v3b005_dose_response_report.json"
)
DEFAULT_OUTPUT_DIR = DEFAULT_REPORT.parent / "figures"
REPORT_SCHEMA = "vla-wam-shared-v3b005-nano-dose-response-report-v1"

BACKGROUND = "#F7F3EC"
PANEL = "#FFFCF7"
INK = "#18303A"
MUTED = "#637179"
GRID = "#D9D3C8"
LEFT = "#C66A3B"
RIGHT = "#2E74A8"
DEPTH = "#2D7C67"
MEDIAN = "#18303A"
FAILURE_COLORS = {
    "correct": "#2D7C67",
    "pick_failed": "#D3A345",
    "transport_failed": "#BF7059",
    "wrong_side": "#7B5A8E",
    "release_failed": "#7C878C",
}
FAILURE_LABELS = {
    "correct": "Correct",
    "pick_failed": "Pick failed",
    "transport_failed": "Transport failed",
    "wrong_side": "Wrong side",
    "release_failed": "Release failed",
}


class RenderError(RuntimeError):
    """Raised when a report cannot support the registered figures."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
    }


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "savefig.facecolor": BACKGROUND,
            "font.family": "Arial",
            "font.size": 11.5,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "svg.hashsalt": "vla-wam-v3b005-dose-response",
        }
    )


def _validate(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != REPORT_SCHEMA:
        raise RenderError("unexpected V3-B005 analysis schema")
    population = report.get("population")
    if not isinstance(population, Mapping) or (
        population.get("matched_seed_count") != 15
        or population.get("level_count") != 7
        or population.get("matched_pair_count") != 105
        or population.get("behavioral_episode_count") != 210
    ):
        raise RenderError("V3-B005 report does not contain the complete registered cohort")
    levels = report.get("by_level")
    seeds = report.get("seed_level")
    if not isinstance(levels, list) or len(levels) != 7:
        raise RenderError("V3-B005 report requires seven ordered level rows")
    if not isinstance(seeds, list) or len(seeds) != 15:
        raise RenderError("V3-B005 report requires fifteen matched-seed rows")
    expected = list(range(7))
    if [row.get("level_index") for row in levels] != expected:
        raise RenderError("V3-B005 levels are not in frozen order")


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.16, metadata={"Date": None})
    # Matplotlib emits spaces before newlines in SVG path data. Normalize the
    # generated text so the committed publication artifact passes diff checks.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(
        png,
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.16,
        metadata={"Software": "steerable V3-B005 registered renderer"},
    )
    plt.close(fig)
    return [svg, png]


def render_dose_response(report: Mapping[str, Any], output_dir: Path) -> list[Path]:
    levels = report["by_level"]
    x_m = np.asarray(
        [row["reference_object_initial_lateral_position_y_m"] for row in levels],
        dtype=np.float64,
    )
    center_m = float(x_m.mean())
    x_cm = 100.0 * (x_m - center_m)

    seed_depth_cm = 100.0 * np.asarray(
        [row["depth_contrast_by_level_m"] for row in report["seed_level"]],
        dtype=np.float64,
    )
    depth_mean_cm = 100.0 * np.asarray(
        [row["depth_contrast_B_m"]["mean"] for row in levels], dtype=np.float64
    )
    depth_median_cm = 100.0 * np.asarray(
        [row["depth_contrast_B_m"]["median"] for row in levels], dtype=np.float64
    )
    depth_ci_cm = 100.0 * np.asarray(
        [row["depth_contrast_B_m"]["ci95"] for row in levels], dtype=np.float64
    )

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 10.0), sharex=True)
    fig.subplots_adjust(left=0.105, right=0.975, bottom=0.105, top=0.79, hspace=0.36)
    fig.suptitle(
        "Directional margin changes continuously with reference-object position",
        x=0.055,
        y=0.973,
        ha="left",
        fontfamily="Georgia",
        fontsize=25,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.927,
        "Cosmos3 Nano Policy DROID · 15 matched seeds at each of seven bowl positions · 210 valid episodes",
        ha="left",
        color=MUTED,
        fontsize=12.1,
    )
    fig.text(
        0.055,
        0.890,
        "LEFT: “Put the Rubik’s cube to the left of the bowl.”     RIGHT: “Put the Rubik’s cube to the right of the bowl.”",
        ha="left",
        color=INK,
        fontsize=11.5,
    )

    ax = axes[0]
    for values in seed_depth_cm:
        ax.plot(x_cm, values, color="#8FA099", alpha=0.20, linewidth=1.05, zorder=1)
    ax.axhline(0.0, color=INK, linewidth=1.0, zorder=2)
    ax.fill_between(x_cm, depth_ci_cm[:, 0], depth_ci_cm[:, 1], color=DEPTH, alpha=0.16, zorder=2)
    ax.plot(x_cm, depth_mean_cm, color=DEPTH, linewidth=2.5, marker="o", markersize=7, zorder=3)
    ax.scatter(x_cm, depth_median_cm, marker="D", s=42, color=MEDIAN, zorder=4, label="Level median")
    ax.set_ylabel("RIGHT − LEFT requested-side depth (cm)")
    ax.set_title("A. Placement margin", loc="left", fontsize=17, fontfamily="Georgia", fontweight="bold", pad=11)
    ax.text(
        0.995,
        0.96,
        "Positive: more margin for RIGHT\nNegative: more margin for LEFT",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.4,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower left", fontsize=10.3)

    ax = axes[1]
    offsets = {"left": -0.42, "right": 0.42}
    colors = {"left": LEFT, "right": RIGHT}
    labels = {"left": "LEFT prompt", "right": "RIGHT prompt"}
    for relation in ("left", "right"):
        rates = 100.0 * np.asarray(
            [row["binary_success"][relation]["rate"] for row in levels], dtype=np.float64
        )
        intervals = 100.0 * np.asarray(
            [row["binary_success"][relation]["wilson_95"] for row in levels], dtype=np.float64
        )
        x_plot = x_cm + offsets[relation]
        errors = np.vstack([rates - intervals[:, 0], intervals[:, 1] - rates])
        ax.errorbar(
            x_plot,
            rates,
            yerr=errors,
            color=colors[relation],
            linewidth=2.0,
            marker="o",
            markersize=7,
            capsize=4,
            label=labels[relation],
            zorder=3,
        )
        for x_value, rate, level in zip(x_plot, rates, levels):
            successes = level["binary_success"][relation]["successes"]
            ax.text(x_value, min(102.0, rate + 4.2), f"{successes}/15", ha="center", va="bottom", fontsize=9.2, color=colors[relation])
    ax.set_ylim(-3.0, 108.0)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Episodes satisfying the frozen task predicate")
    ax.set_title("B. Binary completion", loc="left", fontsize=17, fontfamily="Georgia", fontweight="bold", pad=11)
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=10.8)

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.8, alpha=0.75)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[-1].set_xticks(x_cm, [f"{value:+.0f}" for value in x_cm])
    axes[-1].set_xlabel(
        "Bowl displacement from registered center (cm; + is robot-left)\n"
        f"registered center y = {center_m:.6f} m",
        labelpad=10,
    )
    fig.text(
        0.975,
        0.024,
        "Thin lines are matched seeds; green band is the 95% seed-bootstrap interval; success bars use Wilson 95% intervals. All valid failures remain in denominators.",
        ha="right",
        fontsize=9.7,
        color=MUTED,
    )
    return _save(fig, output_dir / "figure3_nano_lateral_dose_response")


def render_failure_taxonomy(report: Mapping[str, Any], output_dir: Path) -> list[Path]:
    taxonomy = report["failure_taxonomy_counts"]
    levels = report["by_level"]
    x_m = np.asarray(
        [row["reference_object_initial_lateral_position_y_m"] for row in levels],
        dtype=np.float64,
    )
    x_cm = 100.0 * (x_m - float(x_m.mean()))
    categories = tuple(FAILURE_COLORS)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.6), sharey=True)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.19, top=0.78, wspace=0.16)
    fig.suptitle(
        "Failure modes across the registered geometry sweep",
        x=0.055,
        y=0.965,
        ha="left",
        fontfamily="Georgia",
        fontsize=25,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.910,
        "Each horizontal bar is 15 episodes at one bowl position. Categories are mutually exclusive; every behavioral failure remains in the denominator.",
        ha="left",
        color=MUTED,
        fontsize=11.8,
    )
    fig.text(
        0.055,
        0.874,
        "Exact prompts differ only in ‘left’ versus ‘right’; the robot, cameras, controller, scorer, and non-movable geometry are fixed.",
        ha="left",
        color=INK,
        fontsize=11.2,
    )

    y = np.arange(7)
    for ax, relation, color in zip(axes, ("left", "right"), (LEFT, RIGHT)):
        starts = np.zeros(7)
        for category in categories:
            counts = np.asarray(
                [taxonomy[str(level)][relation][category] for level in range(7)],
                dtype=np.int64,
            )
            ax.barh(
                y,
                counts,
                left=starts,
                height=0.66,
                color=FAILURE_COLORS[category],
                edgecolor=BACKGROUND,
                linewidth=1.0,
            )
            for row_index, (start, count) in enumerate(zip(starts, counts)):
                if count >= 2:
                    ax.text(start + count / 2, row_index, str(int(count)), ha="center", va="center", color="white", fontsize=10, fontweight="bold")
            starts += counts
        ax.set_xlim(0, 15)
        ax.set_xticks([0, 3, 6, 9, 12, 15])
        ax.set_xlabel("Episodes")
        ax.set_title(f"{relation.upper()} prompt", loc="left", color=color, fontsize=17, fontfamily="Georgia", fontweight="bold", pad=10)
        ax.grid(axis="x", color=GRID, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].set_yticks(y, [f"{value:+.0f} cm" for value in x_cm])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Bowl displacement from registered center")

    handles = [Patch(facecolor=FAILURE_COLORS[key], label=FAILURE_LABELS[key]) for key in categories]
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.055, 0.045),
        ncol=5,
        frameon=False,
        fontsize=10.6,
        columnspacing=1.5,
    )
    return _save(fig, output_dir / "nano_v3b005_failure_taxonomy_by_level")


def render(report: Mapping[str, Any], output_dir: Path) -> list[Path]:
    _validate(report)
    configure_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    outputs.extend(render_dose_response(report, output_dir))
    outputs.extend(render_failure_taxonomy(report, output_dir))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = render(report, args.output_dir.resolve())
    plot_data = {
        "schema_version": "vla-wam-shared-v3b005-nano-dose-response-plot-data-v1",
        "source_report": file_record(report_path),
        "by_level": report["by_level"],
        "primary_depth_dose_response": report["primary_depth_dose_response"],
        "binary_success_secondary": report["binary_success_secondary"],
    }
    plot_data_path = args.output_dir.resolve() / "nano_v3b005_plot_data.json"
    plot_data_path.write_bytes(canonical_json_bytes(plot_data))
    outputs.append(plot_data_path)
    manifest = {
        "schema_version": "vla-wam-shared-v3b005-nano-dose-response-figure-manifest-v1",
        "source_report": file_record(report_path),
        "outputs": [file_record(path) for path in outputs],
        "claim_boundary": "The plots describe Nano V3-B005 only; no DROID/RoboTwin or checkpoint pooling.",
    }
    manifest_path = args.output_dir.resolve() / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    print(json.dumps({"manifest": file_record(manifest_path), "outputs": manifest["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
