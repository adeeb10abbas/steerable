#!/usr/bin/env python3
"""Shared publication figure style for the VLA-versus-WAM steerability study.

One design system for every figure in the paper, the blog article, and the
social exports. Importing this module is the only supported way to configure
matplotlib inside this repository, so that a figure regenerated a year from now
still matches the ones already published.

Encoding contract
-----------------
Hue encodes the *checkpoint* (indigo = pi0.5 VLA, amber = Cosmos WAM).
Tint encodes the *requested direction* (light = LEFT, saturated = RIGHT).
Grey encodes *no information* (evaluator abstention, uncertain, out of scope).
Nothing in this repository may use hue to encode direction and checkpoint at
the same time.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

INK = "#0B1220"          # primary text and data ink
INK_SOFT = "#3D4756"     # secondary text
MUTED = "#6B7789"        # captions, units, annotations
FAINT = "#9AA5B4"        # de-emphasised annotation
GRID = "#E4E9F0"         # gridlines
RULE = "#CBD3DE"         # separators and baselines
PAPER = "#FFFFFF"        # canvas
PANEL = "#F6F8FB"        # panel wash for grouping

# Checkpoint hues.
VLA = "#2F53CC"          # pi0.5 DROID (VLA)
VLA_LIGHT = "#93A6EA"
VLA_WASH = "#E4E9FB"

WAM = "#E2681A"          # Cosmos3 Edge DROID (WAM)
WAM_LIGHT = "#F3AE7A"
WAM_WASH = "#FCEBDD"

# Outcome / status hues, used only where the quantity is not a checkpoint.
POSITIVE = "#0E9C74"     # requested outcome reached
POSITIVE_WASH = "#DCF2EA"
CAUTION = "#E9A21B"      # partial or one-sided evidence
NEGATIVE = "#C8402F"     # opposite of the requested outcome
NEGATIVE_WASH = "#FBE7E4"
NEUTRAL = "#7C8797"      # certain negative
ABSTAIN = "#DDE3EB"      # evaluator could not decide

MODEL_IDS = ("pi05_droid_vla", "cosmos3_edge_droid_wam")

MODEL_LABELS = {
    "pi05_droid_vla": "π0.5 DROID",
    "cosmos3_edge_droid_wam": "Cosmos3 Edge DROID",
}
MODEL_CLASS = {
    "pi05_droid_vla": "VLA",
    "cosmos3_edge_droid_wam": "WAM",
}
MODEL_COLORS = {"pi05_droid_vla": VLA, "cosmos3_edge_droid_wam": WAM}
MODEL_LIGHT = {"pi05_droid_vla": VLA_LIGHT, "cosmos3_edge_droid_wam": WAM_LIGHT}
MODEL_WASH = {"pi05_droid_vla": VLA_WASH, "cosmos3_edge_droid_wam": WAM_WASH}

DIRECTIONS = ("left", "right")
DIRECTION_MARKERS = {"left": "o", "right": "D"}

WORDINGS = ("canonical", "short_paraphrase", "declarative_goal", "contrastive_goal")
WORDING_LABELS = {
    "canonical": "Canonical",
    "short_paraphrase": "Short",
    "declarative_goal": "Declarative",
    "contrastive_goal": "Contrastive",
}

QUADRANT_ORDER = (
    "imagines_requested_executes_requested",
    "imagines_requested_executes_not_requested",
    "does_not_imagine_requested_executes_requested",
    "neither_imagines_nor_executes_requested",
    "uncertain_future",
)
QUADRANT_LABELS = {
    "imagines_requested_executes_requested": "imagined + executed",
    "imagines_requested_executes_not_requested": "imagined only",
    "does_not_imagine_requested_executes_requested": "executed only",
    "neither_imagines_nor_executes_requested": "neither (certain)",
    "uncertain_future": "evaluator abstained",
}
QUADRANT_COLORS = {
    "imagines_requested_executes_requested": POSITIVE,
    "imagines_requested_executes_not_requested": CAUTION,
    "does_not_imagine_requested_executes_requested": VLA,
    "neither_imagines_nor_executes_requested": NEUTRAL,
    "uncertain_future": ABSTAIN,
}


def model_color(model_id: str, direction: str | None = None) -> str:
    """Checkpoint hue, tinted by requested direction when one is given."""
    if direction is None:
        return MODEL_COLORS[model_id]
    return MODEL_LIGHT[model_id] if direction == "left" else MODEL_COLORS[model_id]


# --------------------------------------------------------------------------
# Typography and rcParams
# --------------------------------------------------------------------------

_SANS_STACK = ["Lato", "Nimbus Sans", "TeX Gyre Heros", "DejaVu Sans"]
_MONO_STACK = ["Noto Sans Mono", "DejaVu Sans Mono"]


def use_style(scale: float = 1.0) -> None:
    """Install the study style. Call once before creating any figure."""
    mpl.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "figure.edgecolor": PAPER,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.facecolor": PAPER,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.28,
            "font.family": "sans-serif",
            "font.sans-serif": _SANS_STACK,
            "font.monospace": _MONO_STACK,
            "font.size": 10.0 * scale,
            "font.weight": "regular",
            "text.color": INK,
            "axes.facecolor": PAPER,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK_SOFT,
            "axes.labelsize": 10.0 * scale,
            "axes.labelweight": "semibold",
            "axes.labelpad": 7.0,
            "axes.titlesize": 11.5 * scale,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.titlepad": 11.0,
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.9,
            "grid.alpha": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "xtick.labelsize": 9.4 * scale,
            "ytick.labelsize": 9.4 * scale,
            "xtick.major.size": 0.0,
            "ytick.major.size": 0.0,
            "xtick.major.pad": 6.0,
            "ytick.major.pad": 5.0,
            "legend.frameon": False,
            "legend.fontsize": 9.4 * scale,
            "legend.labelcolor": INK_SOFT,
            "legend.handlelength": 1.1,
            "legend.handleheight": 1.1,
            "legend.handletextpad": 0.6,
            "legend.columnspacing": 1.8,
            "legend.borderpad": 0.0,
            "lines.linewidth": 1.9,
            "lines.solid_capstyle": "round",
            "patch.linewidth": 0.0,
            "hatch.linewidth": 0.8,
        }
    )


# --------------------------------------------------------------------------
# Layout helpers
# --------------------------------------------------------------------------


def title_block(
    fig: Figure,
    title: str,
    subtitle: str | None = None,
    *,
    x: float = 0.0,
    y: float = 1.0,
    scale: float = 1.0,
) -> None:
    """Left-aligned bold title with a muted subtitle, anchored to the figure."""
    fig.text(
        x,
        y,
        title,
        ha="left",
        va="bottom",
        fontsize=16.0 * scale,
        fontweight="bold",
        color=INK,
    )
    if subtitle:
        fig.text(
            x,
            y - 0.052 * scale,
            subtitle,
            ha="left",
            va="bottom",
            fontsize=10.4 * scale,
            fontweight="regular",
            color=MUTED,
        )


def footnote(fig: Figure, text: str, *, x: float = 0.0, y: float = -0.02) -> None:
    """Method note or claim boundary, set small and muted under the figure."""
    fig.text(
        x,
        y,
        text,
        ha="left",
        va="top",
        fontsize=8.6,
        color=FAINT,
        linespacing=1.5,
    )


def grid(ax: Axes, axis: str = "y") -> None:
    ax.set_axisbelow(True)
    ax.grid(axis=axis, color=GRID, linewidth=0.9, zorder=0)


def strip_spines(ax: Axes, keep: Sequence[str] = ()) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)


def clean(ax: Axes, *, grid_axis: str | None = "y", keep: Sequence[str] = ("bottom",)) -> None:
    """The default axis treatment: hairline baseline, soft grid, no ticks."""
    strip_spines(ax, keep)
    for side in keep:
        ax.spines[side].set_color(RULE)
        ax.spines[side].set_linewidth(0.9)
    if grid_axis:
        grid(ax, grid_axis)
    ax.tick_params(length=0)


def panel_tag(ax: Axes, letter: str, *, dx: float = -0.055, dy: float = 1.06) -> None:
    ax.text(
        dx,
        dy,
        letter,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        ha="left",
        va="bottom",
    )


def model_legend(
    fig: Figure,
    *,
    y: float = -0.06,
    x: float = 0.0,
    directions: bool = True,
) -> None:
    """Shared checkpoint (hue) and direction (tint) key."""
    from matplotlib.lines import Line2D

    handles: list[Line2D] = []
    for model_id in MODEL_IDS:
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                markersize=8,
                linestyle="none",
                color=MODEL_COLORS[model_id],
                label=f"{MODEL_LABELS[model_id]} ({MODEL_CLASS[model_id]})",
            )
        )
    if directions:
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                markersize=8,
                linestyle="none",
                color=FAINT,
                alpha=0.45,
                label="light fill = LEFT request",
            )
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                markersize=8,
                linestyle="none",
                color=FAINT,
                label="solid fill = RIGHT request",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(x, y),
        ncol=len(handles),
        frameon=False,
    )


def value_label(
    ax: Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str = INK,
    size: float = 9.2,
    weight: str = "bold",
    ha: str = "center",
    va: str = "bottom",
    **kwargs,
) -> None:
    ax.text(x, y, text, color=color, fontsize=size, fontweight=weight, ha=ha, va=va, **kwargs)


def annotate_soft(ax: Axes, text: str, xy, xytext, *, color: str = MUTED, size: float = 8.8):
    return ax.annotate(
        text,
        xy=xy,
        xytext=xytext,
        textcoords="offset points",
        fontsize=size,
        color=color,
        arrowprops=dict(arrowstyle="-", color=RULE, linewidth=0.9, shrinkA=0, shrinkB=2),
    )


def save(fig: Figure, path, *, close: bool = True) -> None:
    fig.savefig(path)
    if close:
        plt.close(fig)


def percent_axis(ax: Axes, axis: str = "y", *, upper: float = 1.0, step: float = 0.25) -> None:
    import numpy as np

    ticks = np.arange(0.0, upper + 1e-9, step)
    labels = [f"{value:.0%}" for value in ticks]
    if axis == "y":
        ax.set_yticks(ticks, labels)
    else:
        ax.set_xticks(ticks, labels)


def iter_conditions(wordings: Iterable[str] = WORDINGS) -> list[tuple[str, str]]:
    return [(wording, direction) for wording in wordings for direction in DIRECTIONS]
