#!/usr/bin/env python3
"""Render the H4-first V3-E005 result without cross-arena pooling."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/cross_arena_geometry_v3e005"
INK = "#17324D"
BLUE = "#2F6690"
GREEN = "#2A9D8F"
ORANGE = "#E76F51"
PAPER = "#F8F5EF"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#495057",
            "axes.linewidth": 0.8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": "#495057",
            "ytick.color": "#495057",
            "figure.facecolor": PAPER,
            "axes.facecolor": "#FFFEFA",
            "savefig.facecolor": PAPER,
            "grid.color": "#DEE2E6",
            "svg.hashsalt": "vla-wam-v3e005",
        }
    )


def save(fig: plt.Figure, output: Path, stem: str, caption: str, claim_status: str) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight", pad_inches=0.12)
        if suffix == "svg":
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
        records.append(
            {
                "path": display_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "format": suffix,
                "caption": caption,
                "claim_status": claim_status,
            }
        )
    plt.close(fig)
    return records


def render_progress(report: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    valid = int(report["valid_behavioral_episodes"])
    total = int(report["registered_behavioral_cells"])
    ax.barh([0], [total], color="#DFE5EA", height=0.42)
    ax.barh([0], [valid], color=BLUE, height=0.42)
    ax.set_xlim(0, total)
    ax.set_yticks([])
    ax.set_xlabel("Hash-bound RoboTwin behavioral episodes")
    ax.set_title("V3-E005 remains incomplete; all scientific claims are withheld", loc="left")
    ax.text(valid, 0, f"  {valid}/{total}", va="center", fontweight="bold")
    ax.spines[["left", "right", "top"]].set_visible(False)
    return save(fig, output, "v3e005_progress", "Progress only. H4 and H1–H3 are not evaluated on partial evidence.", "no_publication_claim")


def render_h4(report: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    gate = report["h4_gate"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    levels = ["0.00", "1.00"]
    labels = ["Asymmetric control\n(s = 0)", "Symmetric object layout\n(s = 1)"]
    means = [gate["levels"][level]["mean_m"] for level in levels]
    lows = [gate["levels"][level]["scene_clustered_bootstrap_mean95"]["low"] for level in levels]
    highs = [gate["levels"][level]["scene_clustered_bootstrap_mean95"]["high"] for level in levels]
    colors = [BLUE if gate["levels"][level]["pass"] else ORANGE for level in levels]
    for index, level in enumerate(levels):
        effects = gate["levels"][level]["seed_level_effects"]
        jitter = np.linspace(-0.11, 0.11, len(effects))
        ax.scatter(index + jitter, [item["effect_m"] for item in effects], s=23, alpha=0.45, color=colors[index], zorder=2)
    ax.errorbar(
        np.arange(2),
        means,
        yerr=[[value - low for value, low in zip(means, lows)], [high - value for value, high in zip(means, highs)]],
        fmt="D",
        markersize=8,
        color=INK,
        ecolor=INK,
        capsize=5,
        linewidth=1.8,
        zorder=3,
    )
    ax.axhline(0.0, color="#6C757D", linewidth=1)
    ax.axhline(float(gate["threshold_m"]), color=ORANGE, linestyle="--", linewidth=1.4, label="registered +5 cm threshold")
    ax.set_xticks(range(2), labels)
    ax.set_ylabel("Endpoint redirection: LEFT − RIGHT (m)")
    ax.set_title("H4 first: does the prompt reliably redirect the endpoint?", loc="left")
    subtitle = "H4 PASSED at both levels" if gate["hard_gate_passed"] else "H4 FAILED — H1–H3 are withheld"
    ax.text(0.01, 0.97, subtitle, transform=ax.transAxes, va="top", fontweight="bold", color=GREEN if gate["hard_gate_passed"] else ORANGE)
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False, loc="lower right")
    caption = "Each point is one seed; diamonds are means and bars are 95% scene-clustered bootstrap intervals. H4 must pass independently at s=0 and s=1."
    return save(fig, output, "v3e005_h4_endpoint_gate", caption, "hard_gate_pass" if gate["hard_gate_passed"] else "hard_gate_fail_h1_h3_withheld")


def render_h1(report: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    h1 = report["hypotheses"]["H1"]
    interaction = h1["interaction_s1_minus_s0"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), constrained_layout=True)
    specs = (
        ("binary", "Success-gap interaction\n(s1 − s0)", "Probability difference"),
        ("requested_depth_m", "Requested-depth interaction\n(s1 − s0)", "Meters"),
    )
    for ax, (key, title, ylabel) in zip(axes, specs):
        item = interaction[key]
        ci = item["scene_clustered_bootstrap_mean95"]
        point = item["mean"]
        ax.errorbar([0], [point], yerr=[[point - ci["low"]], [ci["high"] - point]], fmt="D", color=BLUE, capsize=6, markersize=9)
        ax.axhline(0.0, color="#6C757D", linewidth=1)
        ax.set_xlim(-0.7, 0.7)
        ax.set_xticks([])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", linewidth=0.7)
        ax.text(0, point, f"  {point:+.3f}", va="center", fontweight="bold")
    fig.suptitle("H1 is shown only because H4 passed at both layouts", x=0.02, ha="left", fontsize=14, fontweight="bold")
    caption = "Seed-matched s1-minus-s0 interactions with 95% scene-clustered bootstrap intervals; RoboTwin only."
    return save(fig, output, "v3e005_h1_interactions", caption, "reported_after_h4_pass")


def render_failures(report: Mapping[str, Any], output: Path) -> list[dict[str, Any]]:
    h3 = report["hypotheses"]["H3"]
    categories = ["correct", "pick_failed", "transport_failed", "wrong_side", "release_failed"]
    colors = {"correct": GREEN, "pick_failed": "#C9ADA7", "transport_failed": "#E9C46A", "wrong_side": ORANGE, "release_failed": "#7A5195"}
    groups = [("0.00", "left"), ("0.00", "right"), ("1.00", "left"), ("1.00", "right")]
    fig, ax = plt.subplots(figsize=(9.8, 5.0), constrained_layout=True)
    bottom = np.zeros(len(groups))
    for category in categories:
        values = [h3["levels"][level]["directions"][direction]["row_normalized"][category] for level, direction in groups]
        ax.bar(range(len(groups)), values, bottom=bottom, color=colors[category], label=category.replace("_", " "))
        bottom += np.asarray(values)
    ax.set_xticks(range(len(groups)), ["s0 LEFT", "s0 RIGHT", "s1 LEFT", "s1 RIGHT"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of valid episodes")
    ax.set_title("Failure composition is descriptive and remains within RoboTwin", loc="left")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    caption = "All valid behavioral failures remain in the denominator. Missing failure-only shares are NR, never zero."
    return save(fig, output, "v3e005_h3_failure_taxonomy", caption, "reported_after_h4_pass")


def render(results_path: Path, output_dir: Path) -> dict[str, Any]:
    report = json.loads(
        Path(results_path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if report.get("amendment_id") != "V3-E005" or report.get("arena") != "robotwin":
        raise ValueError("renderer accepts only V3-E005 RoboTwin results")
    configure()
    records: list[dict[str, Any]] = []
    if not report["coverage"]["complete"]:
        records.extend(render_progress(report, output_dir))
    else:
        records.extend(render_h4(report, output_dir))
        if report["h4_gate"]["hard_gate_passed"]:
            records.extend(render_h1(report, output_dir))
            records.extend(render_failures(report, output_dir))
    manifest = {
        "schema_version": "vla-wam-shared-v3e005-figure-manifest-v1",
        "amendment_id": "V3-E005",
        "arena": "robotwin",
        "results": {"path": display_path(results_path), "bytes": results_path.stat().st_size, "sha256": sha256(results_path)},
        "status": "complete_figures" if report["coverage"]["complete"] else "partial_progress_figure_only",
        "h4_outcome": report["h4_gate"]["outcome"],
        "h1_h3_rendered": bool(report["coverage"]["complete"] and report["h4_gate"]["hard_gate_passed"]),
        "figures": records,
        "scientific_boundaries": {
            "droid_imported_or_pooled": False,
            "nested_seed_replicates_are_independent_scenes": False,
            "h1_h3_withheld_when_h4_fails": True,
            "symmetric_object_layout_not_symmetric_robot": True,
        },
    }
    path = output_dir / "figure_manifest.json"
    path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=BASE / "results/results.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results/figures")
    args = parser.parse_args()
    manifest = render(args.results, args.output_dir)
    print(json.dumps({"status": manifest["status"], "h4_outcome": manifest["h4_outcome"], "figure_files": len(manifest["figures"])}, indent=2))


if __name__ == "__main__":
    main()
