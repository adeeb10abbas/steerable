#!/usr/bin/env python3
"""Render Figure 2 from the three completed DROID position-reflection ablations.

The figure keeps the two registered outcomes separate.  Requested-side depth is
continuous and defined for every behavioral episode; binary success remains the
frozen pickup/transport/cone/release predicate.  Every point is one prespecified
matched seed's reflected-minus-control difference-in-differences.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NANO = ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_summary.json"
PI05 = ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_report.json"
DREAM = ROOT / "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003_summary.json"
OUTPUT_DIR = ROOT / "artifacts/vla_wam_shared_v3/analysis/paper_figures"

BACKGROUND = "#F6F1E8"
INK = "#17232B"
MUTED = "#5E6A70"
GRID = "#D7D0C5"
DEPTH = "#2D7A63"
SUCCESS = "#6B55A5"
SOFT_NEGATIVE = "#E5EFEA"
SOFT_POSITIVE = "#F3E6DF"


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required closed evidence is missing: {path}")
    return json.loads(path.read_text())


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256_path(path)}


def _success_did_from_cells(cells: dict[str, Any]) -> int:
    def success(key: str) -> int:
        row = cells[key]
        return int(bool(row.get("requested_success", row.get("success"))))

    return (
        success("position_mirrored:right")
        - success("position_mirrored:left")
        - success("control:right")
        + success("control:left")
    )


def _seed_values(data: dict[str, Any], source: str) -> tuple[np.ndarray, np.ndarray]:
    if source == "nano":
        rows = data["seed_level"]
        depth = [row["full_sample"]["I_position_reflection_interaction_m"] for row in rows]
        success = [_success_did_from_cells(row["cells"]) for row in rows]
    else:
        analysis = data.get("registered_analysis", data.get("analysis"))
        if not isinstance(analysis, dict):
            raise ValueError(f"{source} report has no registered analysis")
        rows = analysis["seed_level"]
        depth = [row["I_requested_side_depth_interaction_m"] for row in rows]
        success = [row["binary_success_DiD"] for row in rows]
    depth_values = np.asarray(depth, dtype=float)
    success_values = np.asarray(success, dtype=int)
    if depth_values.shape != (27,) or success_values.shape != (27,):
        raise ValueError(f"{source} must contain exactly 27 matched seeds")
    if not np.isfinite(depth_values).all() or not np.isin(success_values, [-2, -1, 0, 1, 2]).all():
        raise ValueError(f"{source} contains invalid per-seed interaction values")
    return depth_values, success_values


def _aggregate(data: dict[str, Any], source: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if source == "nano":
        primary = data["full_sample_primary"]
        depth = primary["I_position_reflection_interaction"]
        cells = data["condition_outcomes"]
        success = {
            "cell_success_table_2x2": {
                arm: {
                    direction: {
                        "episodes": int(cells[f"{arm}:{direction}"]["episodes"]),
                        "successes": int(cells[f"{arm}:{direction}"]["successes"]),
                    }
                    for direction in ("left", "right")
                }
                for arm in ("control", "position_mirrored")
            }
        }
    else:
        primary = data.get("full_sample_primary")
        registered = data.get("registered_analysis", data.get("analysis"))
        if primary is None:
            depth = registered["H2_requested_side_depth"]["reflected_minus_control_interaction"]
            success = registered["H3_binary_success"]
        else:
            depth = primary["I_position_reflection_interaction"]
            success = primary["binary_success_DiD"]
    return depth, success


def _bootstrap_mean(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    return tuple(float(value) for value in np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))


def _configure_style() -> None:
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
            "svg.fonttype": "none",
            "svg.hashsalt": "vla-wam-v3-three-checkpoint-mirror",
        }
    )


def _cell_line(success: dict[str, Any]) -> str:
    cells = success["cell_success_table_2x2"]
    control = cells["control"]
    reflected = cells["position_mirrored"]
    return (
        f"control L {control['left']['successes']}/27 · R {control['right']['successes']}/27     "
        f"reflected L {reflected['left']['successes']}/27 · R {reflected['right']['successes']}/27"
    )


def render(models: list[dict[str, Any]]) -> list[Path]:
    _configure_style()
    fig, axes = plt.subplots(3, 2, figsize=(15.2, 10.6), sharex="col")
    fig.subplots_adjust(left=0.09, right=0.965, top=0.76, bottom=0.12, hspace=0.72, wspace=0.18)
    fig.suptitle(
        "Reflecting object positions reverses the directional depth advantage",
        x=0.055,
        y=0.972,
        ha="left",
        fontfamily="Georgia",
        fontsize=27,
        fontweight="bold",
    )
    fig.text(
        0.055,
        0.918,
        "Three DROID checkpoints · same 27 prespecified seeds · 108 valid episodes per checkpoint · robot base, cameras, prompts, and non-movable geometry held fixed",
        ha="left",
        fontsize=11.7,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.875,
        "LEFT: “Put the Rubik’s cube to the left of the bowl.”     RIGHT: “Put the Rubik’s cube to the right of the bowl.”",
        ha="left",
        fontsize=11.5,
        color=INK,
    )
    fig.text(
        0.055,
        0.828,
        "Negative interaction = reflection shifts the RIGHT-minus-LEFT advantage toward LEFT. Dots are matched seeds; diamonds are means with 95% bootstrap intervals.",
        ha="left",
        fontsize=10.7,
        color=MUTED,
    )

    depth_limits = (-92.0, 24.0)
    jitter_rng = np.random.default_rng(31107)
    for index, model in enumerate(models):
        depth_cm = model["depth"] * 100.0
        success_did = model["success"]
        depth_summary = model["depth_summary"]
        success_summary = model["success_summary"]

        depth_ax, success_ax = axes[index]
        depth_ax.axvspan(depth_limits[0], 0, color=SOFT_NEGATIVE, alpha=0.9, zorder=0)
        depth_ax.axvspan(0, depth_limits[1], color=SOFT_POSITIVE, alpha=0.9, zorder=0)
        depth_ax.axvline(0, color=INK, linewidth=1.1, zorder=1)
        depth_ax.scatter(
            depth_cm,
            jitter_rng.uniform(-0.13, 0.13, 27),
            s=38,
            color="#7E8986",
            edgecolor=BACKGROUND,
            linewidth=0.8,
            zorder=3,
        )
        ci = depth_summary["mean_bootstrap_95"]
        mean = float(depth_summary["mean_m"]) * 100.0
        depth_ax.errorbar(
            mean,
            -0.31,
            xerr=np.asarray([[mean - float(ci["lower"]) * 100.0], [float(ci["upper"]) * 100.0 - mean]]),
            fmt="D",
            color=DEPTH,
            ecolor=DEPTH,
            markersize=7.8,
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.9,
            elinewidth=2.8,
            capsize=4.5,
            zorder=4,
        )
        signs = depth_summary["paired_sign_test"]
        depth_ax.text(
            depth_limits[0] + 1.5,
            -0.54,
            f"mean {mean:+.1f} cm  [{float(ci['lower'])*100:+.1f}, {float(ci['upper'])*100:+.1f}]\n"
            f"signs {signs['negative']}− / {signs['positive']}+ · exact p={float(signs['p_value']):.3g}",
            fontsize=9.5,
            va="top",
            color=INK,
        )
        depth_ax.set_xlim(*depth_limits)
        depth_ax.set_ylim(-0.95, 0.38)
        depth_ax.set_yticks([])
        depth_ax.grid(axis="x", color=GRID, linewidth=0.75)
        depth_ax.spines[["top", "right", "left"]].set_visible(False)
        depth_ax.set_title(model["label"], loc="left", fontsize=15.8, fontfamily="Georgia", fontweight="bold", pad=8)

        success_ax.axvspan(-2.35, 0, color=SOFT_NEGATIVE, alpha=0.9, zorder=0)
        success_ax.axvspan(0, 2.35, color=SOFT_POSITIVE, alpha=0.9, zorder=0)
        success_ax.axvline(0, color=INK, linewidth=1.1, zorder=1)
        counts = {value: int(np.sum(success_did == value)) for value in range(-2, 3)}
        for value in range(-2, 3):
            count = counts[value]
            if count:
                ys = np.linspace(-0.13, 0.13, count) if count > 1 else np.asarray([0.0])
                success_ax.scatter(
                    np.full(count, value, dtype=float),
                    ys,
                    s=38,
                    color="#7E8986",
                    edgecolor=BACKGROUND,
                    linewidth=0.8,
                    zorder=3,
                )
                success_ax.text(value, 0.26, str(count), ha="center", va="center", fontsize=9.2, color=MUTED)
        lower, upper = _bootstrap_mean(success_did.astype(float), 8100 + index)
        mean_did = float(success_did.mean())
        success_ax.errorbar(
            mean_did,
            -0.31,
            xerr=np.asarray([[mean_did - lower], [upper - mean_did]]),
            fmt="D",
            color=SUCCESS,
            ecolor=SUCCESS,
            markersize=7.8,
            markeredgecolor=BACKGROUND,
            markeredgewidth=0.9,
            elinewidth=2.8,
            capsize=4.5,
            zorder=4,
        )
        test = success_summary.get("exact_permutation_test", {})
        p_value = test.get("p_value")
        p_text = f" · exact p={float(p_value):.3g}" if p_value is not None else ""
        success_ax.text(-2.32, -0.54, f"mean DiD {mean_did:+.2f}  [{lower:+.2f}, {upper:+.2f}]{p_text}", fontsize=9.5, va="top")
        success_ax.text(-2.32, -0.76, _cell_line(success_summary), fontsize=9.2, va="top", color=MUTED)
        success_ax.set_xlim(-2.4, 2.4)
        success_ax.set_ylim(-1.03, 0.38)
        success_ax.set_yticks([])
        success_ax.set_xticks([-2, -1, 0, 1, 2])
        success_ax.grid(axis="x", color=GRID, linewidth=0.75)
        success_ax.spines[["top", "right", "left"]].set_visible(False)

    axes[0, 0].text(0.0, 1.30, "REQUESTED-SIDE DEPTH INTERACTION", transform=axes[0, 0].transAxes, fontsize=10.6, fontweight="bold")
    axes[0, 1].text(0.0, 1.30, "BINARY SUCCESS INTERACTION", transform=axes[0, 1].transAxes, fontsize=10.6, fontweight="bold")
    axes[-1, 0].set_xlabel("Reflected minus control depth contrast (cm)", labelpad=9)
    axes[-1, 1].set_xlabel("Per-seed success difference-in-differences", labelpad=9)
    fig.text(
        0.055,
        0.035,
        "Interpretation: the position-only intervention changes which requested direction is geometrically advantaged. It does not isolate training distribution, embodiment handedness, or a full-scene symmetry transform.",
        fontsize=10.5,
        color=INK,
        ha="left",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "figure2_three_checkpoint_position_reflection"
    svg = stem.with_suffix(".svg")
    png = stem.with_suffix(".png")
    fig.savefig(svg, bbox_inches="tight", pad_inches=0.14, metadata={"Date": None})
    fig.savefig(png, dpi=240, bbox_inches="tight", pad_inches=0.14, metadata={"Software": "steerable V3 mirror renderer"})
    plt.close(fig)
    return [svg, png]


def main() -> None:
    sources = [("Cosmos3 Nano Policy DROID", NANO, "nano"), ("π0.5 DROID", PI05, "pi05"), ("DreamZero DROID", DREAM, "dream")]
    models = []
    for label, path, source in sources:
        data = read_json(path)
        depth, success = _seed_values(data, source)
        depth_summary, success_summary = _aggregate(data, source)
        models.append(
            {
                "label": label,
                "depth": depth,
                "success": success,
                "depth_summary": depth_summary,
                "success_summary": success_summary,
            }
        )
    outputs = render(models)
    manifest = {
        "schema_version": "vla-wam-shared-v3-three-checkpoint-mirror-figure-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_retrospective_visualization_no_new_inference",
        "inputs": [file_record(path) for _, path, _ in sources],
        "renderer": file_record(Path(__file__).resolve()),
        "outputs": [file_record(path) for path in outputs],
        "claim_boundary": "Position-only reflection holds the base, cameras, prompts, and non-movable geometry fixed. The depth outcome is full-sample; binary success uses the unchanged frozen task predicate. DROID only.",
    }
    manifest_path = OUTPUT_DIR / "figure2_three_checkpoint_position_reflection.manifest.json"
    manifest_path.write_text(json.dumps(manifest, allow_nan=False, sort_keys=True, indent=2) + "\n")
    print(json.dumps(file_record(manifest_path), indent=2))


if __name__ == "__main__":
    main()
