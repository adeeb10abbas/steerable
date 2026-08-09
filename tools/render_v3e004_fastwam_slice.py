#!/usr/bin/env python3
"""Render the complete, arena-separated V3-E004 FastWAM slice.

This renderer is intentionally descriptive.  It never upgrades the complete
RoboTwin stretch slice into a claim about the still-running multi-checkpoint
E004 cohort, and it never pools RoboTwin with DROID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/slices/fastwam_robotwin"
LEFT = "#C56A2D"
RIGHT = "#2F6FAE"
INK = "#1F2933"
MUTED = "#64748B"
GREEN = "#16856B"
RED = "#C2413B"
FAILURE_COLORS = {
    "correct": "#16856B",
    "pick_failed": "#C9ADA7",
    "transport_failed": "#E9B949",
    "wrong_side": "#D95D39",
    "release_failed": "#7A5195",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "#F7F4EE",
            "axes.facecolor": "#FFFEFB",
            "savefig.facecolor": "#F7F4EE",
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.7,
        }
    )


def _interval(row: dict[str, Any], scale: float = 1.0) -> tuple[float, float, float]:
    point = float(row["mean"]) * scale
    ci = row["bootstrap_mean95"]
    return point, float(ci["low"]) * scale, float(ci["high"]) * scale


def render(results_path: Path, output_dir: Path) -> dict[str, Any]:
    report = json.loads(results_path.read_text(encoding="utf-8"))
    checkpoint = report["checkpoints"]["fastwam_robotwin"]
    if checkpoint["arena"] != "robotwin" or checkpoint["valid_episodes"] != 108:
        raise ValueError("FastWAM slice is not the complete 108-cell RoboTwin cohort")
    if checkpoint["core_s0_s1_complete"] is not True:
        raise ValueError("FastWAM paired s=0/s=1 core is incomplete")
    analysis = checkpoint["analysis"]
    levels = analysis["levels"]

    configure()
    fig = plt.figure(figsize=(14.2, 9.4), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.87, bottom=0.14, hspace=0.31, wspace=0.25)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05))
    ax_success = fig.add_subplot(grid[0, 0])
    ax_depth = fig.add_subplot(grid[0, 1])
    ax_endpoint = fig.add_subplot(grid[1, 0])
    ax_failure = fig.add_subplot(grid[1, 1])

    # A. Paired binary task outcome.
    x = np.arange(2)
    width = 0.34
    for offset, relation, color in ((-width / 2, "left", LEFT), (width / 2, "right", RIGHT)):
        rows = [levels[level][f"{relation}_success"] for level in ("0.00", "1.00")]
        values = np.asarray([row["proportion"] * 100 for row in rows])
        low = values - np.asarray([row["wilson95_low"] * 100 for row in rows])
        high = np.asarray([row["wilson95_high"] * 100 for row in rows]) - values
        ax_success.bar(x + offset, values, width, color=color, label=f"Prompt asks {relation.upper()}")
        ax_success.errorbar(x + offset, values, yerr=np.vstack([low, high]), fmt="none", ecolor=INK, capsize=3, linewidth=1)
        for xpos, value, row in zip(x + offset, values, rows):
            ax_success.text(xpos, max(value, 0) + 1.1, f"{row['successes']}/{row['trials']}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax_success.set_xticks(x, ("Asymmetric fixture\ns = 0", "Symmetric-object fixture\ns = 1"))
    ax_success.set_ylim(0, 28)
    ax_success.set_ylabel("Task success (%)")
    ax_success.set_title("A  Binary success remained near floor", loc="left", fontweight="bold")
    ax_success.legend(frameon=False, loc="upper left")
    ax_success.grid(axis="y")

    # B. Requested-depth directional contrast.
    depth_rows = [levels[level]["requested_depth_gap_R_minus_L_m"] for level in ("0.00", "1.00")]
    depth_values = [_interval(row, 100.0) for row in depth_rows]
    for index, (point, low, high) in enumerate(depth_values):
        ax_depth.errorbar(index, point, yerr=[[point - low], [high - point]], fmt="o", color=GREEN, capsize=4, markersize=8, linewidth=2)
        ax_depth.text(index, high + 2.2, f"{point:+.1f} cm", ha="center", fontweight="bold")
    interaction = analysis["interaction_s1_minus_s0_core"]["depth_gap_m"]
    inter = _interval(interaction, 100.0)
    ax_depth.text(
        0.02,
        0.04,
        f"s=1 − s=0 interaction: {inter[0]:+.1f} cm\n95% CI [{inter[1]:+.1f}, {inter[2]:+.1f}] cm\npaired exact p = {interaction['exact_layout_label_permutation']['exact_two_sided_p']:.2g}",
        transform=ax_depth.transAxes,
        va="bottom",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#ECFDF5", "edgecolor": "#A7F3D0"},
    )
    ax_depth.axhline(0, color=INK, linewidth=1)
    ax_depth.set_ylim(-15, 42)
    ax_depth.set_xticks(x, ("Asymmetric\ns = 0", "Symmetric-object\ns = 1"))
    ax_depth.set_ylabel("RIGHT − LEFT requested depth (cm)")
    ax_depth.set_title("B  The continuous depth contrast reversed", loc="left", fontweight="bold")
    ax_depth.grid(axis="y")

    # C. Registered positive control.
    endpoint_rows = [levels[level]["endpoint_redirection_LEFT_minus_RIGHT_m"] for level in ("0.00", "1.00")]
    endpoint_values = [_interval(row, 100.0) for row in endpoint_rows]
    for index, (point, low, high) in enumerate(endpoint_values):
        ax_endpoint.errorbar(index, point, yerr=[[point - low], [high - point]], fmt="o", color=RED, capsize=4, markersize=8, linewidth=2)
        ax_endpoint.text(index, high + 0.8, f"{point:+.1f} cm", ha="center", fontweight="bold")
    ax_endpoint.axhline(0, color=INK, linewidth=1)
    ax_endpoint.set_ylim(-5.5, 8.4)
    ax_endpoint.set_xticks(x, ("Asymmetric\ns = 0", "Symmetric-object\ns = 1"))
    ax_endpoint.set_ylabel("LEFT − RIGHT endpoint redirection (cm)")
    ax_endpoint.set_title("C  The endpoint-redirection positive control did not pass", loc="left", fontweight="bold")
    ax_endpoint.text(
        0.02,
        0.04,
        "Both paired 95% intervals cross zero.\nThe registered equalisation interpretation therefore fails closed.",
        transform=ax_endpoint.transAxes,
        va="bottom",
        fontsize=9.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FFF1F2", "edgecolor": "#FECDD3"},
    )
    ax_endpoint.grid(axis="y")

    # D. Failure decomposition; all failures remain in the denominator.
    categories = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")
    conditions = (("0.00", "left"), ("0.00", "right"), ("1.00", "left"), ("1.00", "right"))
    labels = ("s=0 · LEFT", "s=0 · RIGHT", "s=1 · LEFT", "s=1 · RIGHT")
    bottom = np.zeros(4)
    for category in categories:
        counts = np.asarray([levels[level]["failure_taxonomy"][relation][category] for level, relation in conditions])
        values = counts / 27.0
        ax_failure.barh(np.arange(4), values, left=bottom, color=FAILURE_COLORS[category], label=category.replace("_", " "))
        bottom += values
    ax_failure.set_yticks(np.arange(4), labels)
    ax_failure.invert_yaxis()
    ax_failure.set_xlim(0, 1)
    ax_failure.set_xticks((0, 0.5, 1), ("0%", "50%", "100%"))
    ax_failure.set_xlabel("Outcome composition (27 episodes per row)")
    ax_failure.set_title("D  Most episodes failed during picking", loc="left", fontweight="bold")
    ax_failure.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    ax_failure.grid(axis="x")

    fig.suptitle(
        "FastWAM on RoboTwin: geometry changed the continuous contrast, but not enough evidence supported equalisation",
        x=0.02,
        y=0.975,
        ha="left",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.925,
        "Complete arena-separated stretch slice · 27 matched seeds × 2 layouts × 2 exact static prompts · descriptive until the full V3-E004 cohort closes",
        color=MUTED,
        fontsize=10.3,
    )
    fig.text(
        0.02,
        0.018,
        'Exact prompts: “Put the small woodenblock to the left of the red playingcards box.” / “Put the small woodenblock to the right of the red playingcards box.”  RoboTwin is never pooled with DROID.',
        color=MUTED,
        fontsize=8.8,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for suffix in ("png", "svg"):
        path = output_dir / f"v3e004_fastwam_robotwin_slice.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        if suffix == "svg":
            # Matplotlib emits trailing spaces throughout SVG path data. Keep
            # the committed vector artifact compatible with git diff --check.
            svg_text = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
                encoding="utf-8",
            )
        records.append(
            {
                "path": str(path.relative_to(ROOT)),
                "format": suffix,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    plt.close(fig)
    manifest = {
        "schema_version": "vla-wam-shared-v3e004-fastwam-slice-figure-manifest-v1",
        "amendment_id": "V3-E004",
        "model_id": "fastwam_robotwin",
        "arena": "robotwin",
        "status": "complete_arena_slice_descriptive_only",
        "results_sha256": sha256(results_path),
        "figures": records,
        "claim_boundary": {
            "no_cross_model_claim": True,
            "robotwin_never_pooled_with_droid": True,
            "positive_control_failed_closed": True,
            "equivalence_not_claimed": True,
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=BASE / "results/results.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "figures")
    args = parser.parse_args()
    manifest = render(args.results, args.output_dir)
    print(json.dumps({"status": manifest["status"], "figures": len(manifest["figures"])}, indent=2))


if __name__ == "__main__":
    main()
