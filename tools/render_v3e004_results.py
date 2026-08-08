#!/usr/bin/env python3
"""Render publication-ready V3-E004 figures from the compact result JSON."""
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
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
COLORS = {
    "pi05_current_stack_droid": "#235789",
    "cosmos3_nano_policy_droid": "#2A9D8F",
    "dreamzero_droid_action_cfg": "#7A5195",
    "cosmos3_edge_policy_droid": "#E76F51",
    "fastwam_robotwin": "#6C757D",
}
LABELS = {
    "pi05_current_stack_droid": "π0.5",
    "cosmos3_nano_policy_droid": "Cosmos3 Nano",
    "dreamzero_droid_action_cfg": "DreamZero",
    "cosmos3_edge_policy_droid": "Cosmos3 Edge",
    "fastwam_robotwin": "FastWAM (RoboTwin)",
}
FAILURE_COLORS = {
    "correct": "#2A9D8F",
    "pick_failed": "#C9ADA7",
    "transport_failed": "#E9C46A",
    "wrong_side": "#E76F51",
    "release_failed": "#7A5195",
}
MODEL_ORDER = (
    "pi05_current_stack_droid",
    "cosmos3_nano_policy_droid",
    "dreamzero_droid_action_cfg",
    "cosmos3_edge_policy_droid",
    "fastwam_robotwin",
)


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
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.edgecolor": "#343A40",
            "axes.linewidth": 0.8,
            "xtick.color": "#495057",
            "ytick.color": "#495057",
            "text.color": "#212529",
            "figure.facecolor": "#F8F5EF",
            "axes.facecolor": "#FFFEFA",
            "savefig.facecolor": "#F8F5EF",
            "grid.color": "#DEE2E6",
            "grid.linewidth": 0.7,
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, caption: str, claim_status: str) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for suffix in ("png", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None, bbox_inches="tight")
        records.append(
            {
                "path": str(path),
                "format": suffix,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "caption": caption,
                "claim_status": claim_status,
            }
        )
    plt.close(fig)
    return records


def _analysis(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        model: report["checkpoints"][model]["analysis"]
        for model in MODEL_ORDER
        if model in report["checkpoints"] and report["checkpoints"][model].get("analysis") is not None
    }


def render_progress(report: Mapping[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    items = [report["checkpoints"][model] for model in MODEL_ORDER if model in report["checkpoints"]]
    registered = []
    for item in items:
        model = item["model_id"]
        registered.append(
            sum(
                row["registered"]
                for row in report["coverage"]["by_model_level_direction"]
                if row["model_id"] == model
            )
        )
    valid = [item["valid_episodes"] for item in items]
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    positions = np.arange(len(items))
    ax.bar(positions, registered, color="#DEE2E6", label="registered")
    ax.bar(positions, valid, color=[COLORS[item["model_id"]] for item in items], label="valid")
    ax.set_xticks(positions, [LABELS[item["model_id"]] for item in items], rotation=18, ha="right")
    ax.set_ylabel("Behavioral episodes")
    ax.set_title("V3-E004 queue progress — estimates are not publication claims", loc="left", fontweight="bold")
    ax.grid(axis="y")
    ax.legend(frameon=False)
    for index, (done, total) in enumerate(zip(valid, registered)):
        ax.text(index, max(done, 1), f"{done}/{total}", ha="center", va="bottom", fontsize=9)
    return save_figure(
        fig,
        output_dir,
        "v3e004_progress_coverage",
        "Valid hash-bound behavioral episodes versus the registered E004 queue. Partial values are operational diagnostics only.",
        "partial_no_publication_claims",
    )


def render_interactions(report: Mapping[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    analyses = _analysis(report)
    models = list(analyses)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), constrained_layout=True)
    specs = (
        ("binary_gap", "Success gap (RIGHT − LEFT)", 1.0),
        ("depth_gap_m", "Requested-depth gap (RIGHT − LEFT), cm", 100.0),
    )
    positions = np.arange(len(models))
    for ax, (estimand, ylabel, scale) in zip(axes, specs):
        for layout_index, (level, marker) in enumerate((("0.00", "o"), ("1.00", "s"))):
            x = positions + (layout_index - 0.5) * 0.18
            values, lower, upper = [], [], []
            for model in models:
                key = "binary_gap_R_minus_L" if estimand == "binary_gap" else "requested_depth_gap_R_minus_L_m"
                row = analyses[model]["levels"][level][key]
                values.append(row["mean"] * scale)
                lower.append((row["mean"] - row["bootstrap_mean95"]["low"]) * scale)
                upper.append((row["bootstrap_mean95"]["high"] - row["mean"]) * scale)
            ax.errorbar(
                x,
                values,
                yerr=np.asarray([lower, upper]),
                fmt=marker,
                markersize=7,
                capsize=3,
                color="#495057" if level == "0.00" else "#2A9D8F",
                label="asymmetric reference, s=0" if level == "0.00" else "symmetric object layout, s=1",
            )
        ax.axhline(0, color="#212529", linewidth=0.9)
        if models and models[-1] == "fastwam_robotwin":
            ax.axvspan(len(models) - 1.45, len(models) - 0.55, color="#E9ECEF", alpha=0.65, zorder=-2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y")
    axes[0].set_title("Does removing object-layout asymmetry shrink the directional gap?", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xticks(positions, [LABELS[model] for model in models], rotation=15, ha="right")
    return save_figure(
        fig,
        output_dir,
        "v3e004_interactions",
        "Directional success and requested-depth gaps under the registered asymmetric (s=0) and symmetric-object (s=1) layouts. Intervals are paired seed bootstraps; DROID and RoboTwin are not pooled.",
        report["publication_claim_status"],
    )


def render_dose_response(report: Mapping[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    analyses = _analysis(report)
    models = [model for model in ("pi05_current_stack_droid", "cosmos3_nano_policy_droid") if model in analyses]
    if not models:
        return []
    fig, axes = plt.subplots(2, len(models), figsize=(11, 7.0), squeeze=False, constrained_layout=True)
    for column, model in enumerate(models):
        levels = analyses[model]["levels"]
        rows = [levels[key] for key in sorted(levels, key=float) if float(key) > 0]
        A = np.asarray([row["realised_A"]["mean"] for row in rows])
        for row_index, (key, ylabel, scale) in enumerate(
            (("binary_gap_R_minus_L", "Success gap\n(RIGHT − LEFT)", 1.0), ("requested_depth_gap_R_minus_L_m", "Depth gap, cm\n(RIGHT − LEFT)", 100.0))
        ):
            values = np.asarray([row[key]["mean"] * scale for row in rows])
            low = np.asarray([row[key]["bootstrap_mean95"]["low"] * scale for row in rows])
            high = np.asarray([row[key]["bootstrap_mean95"]["high"] * scale for row in rows])
            order = np.argsort(A)
            ax = axes[row_index, column]
            ax.fill_between(A[order], low[order], high[order], color=COLORS[model], alpha=0.16)
            ax.plot(A[order], values[order], marker="o", color=COLORS[model], linewidth=2)
            ax.axhline(0, color="#212529", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.grid()
        axes[0, column].set_title(LABELS[model], fontweight="bold")
        axes[1, column].set_xlabel("Realised asymmetry A (0 = symmetric)")
    fig.suptitle("Inventory-matched dose response: s=0.25, 0.50, 0.75, 1.00", x=0.02, ha="left", fontweight="bold")
    return save_figure(
        fig,
        output_dir,
        "v3e004_dose_response",
        "Directional gap versus realised object-layout asymmetry for the preregistered inventory-matched positive-s levels. The s=0 anchor is excluded from the primary slope.",
        report["publication_claim_status"],
    )


def render_endpoint_control(report: Mapping[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    analyses = _analysis(report)
    if not analyses:
        return []
    models = list(analyses)
    fig, axes = plt.subplots(len(models), 1, figsize=(10.5, 2.05 * len(models) + 1.0), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes[:, 0], models):
        levels = analyses[model]["levels"]
        x, y, low, high = [], [], [], []
        for label in sorted(levels, key=float):
            row = levels[label]["endpoint_redirection_LEFT_minus_RIGHT_m"]
            x.append(float(label))
            y.append(row["mean"] * 100)
            low.append((row["mean"] - row["bootstrap_mean95"]["low"]) * 100)
            high.append((row["bootstrap_mean95"]["high"] - row["mean"]) * 100)
        ax.errorbar(x, y, yerr=np.asarray([low, high]), color=COLORS[model], marker="o", capsize=3, linewidth=2)
        ax.axhline(0, color="#212529", linewidth=0.8)
        ax.set_ylabel("LEFT−RIGHT\nendpoint, cm")
        ax.set_title(LABELS[model], loc="left", fontsize=10.5, fontweight="bold")
        ax.grid()
    axes[-1, 0].set_xlabel("Registered symmetry level, s")
    fig.suptitle("Positive control: does the prompt still redirect endpoints?", x=0.02, ha="left", fontweight="bold")
    return save_figure(
        fig,
        output_dir,
        "v3e004_endpoint_positive_control",
        "Paired endpoint redirection at every registered symmetry level. A level whose 95% interval does not remain positive fails closed against an equalisation interpretation.",
        report["publication_claim_status"],
    )


def render_failures(report: Mapping[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    checkpoints = [
        report["checkpoints"][model]
        for model in MODEL_ORDER
        if model in report["checkpoints"] and report["checkpoints"][model]["failure_signature"]["levels"]
    ]
    if not checkpoints:
        return []
    fig, axes = plt.subplots(len(checkpoints), 1, figsize=(11, 2.25 * len(checkpoints) + 1.2), squeeze=False, constrained_layout=True)
    categories = ("correct", "pick_failed", "transport_failed", "wrong_side", "release_failed")
    for ax, item in zip(axes[:, 0], checkpoints):
        level_items = item["failure_signature"]["levels"]
        labels = list(level_items)
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for category in categories:
            values = np.asarray(
                [level_items[label]["failure_taxonomy"][category] / level_items[label]["episodes"] for label in labels]
            )
            ax.bar(x, values, bottom=bottom, color=FAILURE_COLORS[category], label=category.replace("_", " "))
            bottom += values
        ax.set_ylim(0, 1)
        ax.set_yticks((0, 0.5, 1.0), ("0%", "50%", "100%"))
        ax.set_xticks(x, [f"s={label}" for label in labels])
        ax.set_title(f"{LABELS[item['model_id']]} — outcome composition", loc="left", fontsize=10.5, fontweight="bold")
        ax.grid(axis="y")
    axes[0, 0].legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.45))
    fig.suptitle("Where does competence fail as object-layout asymmetry is removed?", x=0.02, ha="left", fontweight="bold")
    return save_figure(
        fig,
        output_dir,
        "v3e004_failure_taxonomy",
        "Outcome composition by checkpoint and symmetry level. Behavioral failures remain in denominators; a wrong-side share is unavailable, not zero, when a cell has no failures.",
        report["publication_claim_status"],
    )


def render(results_path: Path, output_dir: Path) -> dict[str, Any]:
    report = json.loads(
        Path(results_path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )
    if report.get("amendment_id") != "V3-E004":
        raise ValueError("wrong result amendment")
    configure()
    records: list[dict[str, Any]] = []
    if report["coverage"]["complete"]:
        records.extend(render_interactions(report, output_dir))
        records.extend(render_dose_response(report, output_dir))
        records.extend(render_endpoint_control(report, output_dir))
        records.extend(render_failures(report, output_dir))
    else:
        records.extend(render_progress(report, output_dir))
    manifest = {
        "schema_version": "vla-wam-shared-v3e004-figure-manifest-v1",
        "amendment_id": "V3-E004",
        "results": {"path": str(results_path), "bytes": results_path.stat().st_size, "sha256": sha256(results_path)},
        "status": "complete_figures" if report["coverage"]["complete"] else "partial_progress_figure_only",
        "figures": records,
        "scientific_boundaries": {
            "droid_and_robotwin_not_pooled": True,
            "partial_results_have_no_publication_claim": not report["coverage"]["complete"],
            "object_layout_not_robot_symmetry": True,
        },
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=BASE / "results/results.json")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results/figures")
    args = parser.parse_args()
    manifest = render(args.results, args.output_dir)
    print(json.dumps({"status": manifest["status"], "figure_files": len(manifest["figures"])}, indent=2))


if __name__ == "__main__":
    main()
