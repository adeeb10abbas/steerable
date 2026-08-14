#!/usr/bin/env python3
"""Regenerate manuscript-facing figures from registered summary values.

The figures are sized for their final IEEE one- or two-column placement. The
source values and their claim boundaries are listed in ``EVIDENCE_MAP.md``.
"""
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[1] / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)
COLORS = {
    "left": "#2f6f83",
    "right": "#bb5e2b",
    "neutral": "#4c566a",
    "green": "#287f6b",
}


def finish(fig, name, tight=True):
    save_kwargs = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(OUT / f"{name}.pdf", **save_kwargs)
    fig.savefig(OUT / f"{name}.png", dpi=400, **save_kwargs)
    plt.close(fig)


def wilson(successes, trials, z=1.959963984540054):
    """Return a binomial proportion and Wilson 95% interval."""
    p = successes / trials
    z2 = z * z
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(p * (1.0 - p) / trials + z2 / (4.0 * trials * trials))
        / denominator
    )
    return p, max(0.0, center - half_width), min(1.0, center + half_width)


# Figure 1: the complete eight-row Phase-A screen underlying the 324-pair claim.
# Values are pinned in the eight Phase-A summaries indexed by EVIDENCE_MAP.md.
screening = {
    "DROID / RoboLab": [
        ("pi0.5", 5, 24, 27),
        ("GR00T N1.7", 3, 0, 27),
        ("Cosmos3 Edge", 18, 25, 27),
        ("Cosmos3 Nano", 26, 25, 27),
        ("DreamZero", 3, 17, 27),
    ],
    "RoboTwin": [
        ("Efficient-WAM-RT", 26, 28, 63),
        ("FastWAM", 24, 20, 63),
        ("LingBot-VA", 19, 19, 63),
    ],
}
fig, axs = plt.subplots(
    1, 2, figsize=(7.0, 2.05), gridspec_kw={"width_ratios": [1.08, 0.92]}
)
for ax, (arena, rows) in zip(axs, screening.items()):
    y = np.arange(len(rows))[::-1]
    for direction, delta, marker, color in [
        ("LEFT", 0.13, "o", COLORS["left"]),
        ("RIGHT", -0.13, "s", COLORS["right"]),
    ]:
        counts = np.array([row[1 if direction == "LEFT" else 2] for row in rows])
        totals = np.array([row[3] for row in rows])
        intervals = [wilson(int(k), int(n)) for k, n in zip(counts, totals)]
        points = np.array([item[0] for item in intervals])
        lower = np.array([item[1] for item in intervals])
        upper = np.array([item[2] for item in intervals])
        ax.errorbar(
            points,
            y + delta,
            xerr=np.vstack([points - lower, upper - points]),
            fmt=marker,
            ms=4.2,
            mfc="white" if direction == "LEFT" else color,
            mec=color,
            mew=0.9,
            color=color,
            ecolor=color,
            capsize=2.0,
            lw=1.0,
            label=direction,
            zorder=3,
        )
        for yy, count, total in zip(y + delta, counts, totals):
            # A fixed count column prevents near-ceiling intervals from colliding.
            x_text = 1.035
            ax.text(
                x_text,
                yy,
                f"{count}/{total}",
                va="center",
                ha="left",
                fontsize=7.1,
                color=color,
            )
    ax.set_yticks(y, [row[0] for row in rows])
    ax.set_xlim(-0.03, 1.17)
    ax.set_xticks([0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_xlabel("Frozen task success")
    denominator = rows[0][3]
    ax.set_title(
        f"{arena} ({denominator} pairs each)", fontweight="bold", pad=2
    )
    ax.grid(axis="x", color="#d7dce0", linewidth=0.6, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
legend_handles, legend_labels = axs[0].get_legend_handles_labels()
fig.legend(
    legend_handles,
    legend_labels,
    frameon=False,
    ncol=2,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    borderaxespad=0,
    handletextpad=0.35,
    columnspacing=0.9,
)
fig.tight_layout(w_pad=1.4, rect=(0, 0, 1, 0.87))
finish(fig, "checkpoint_screen")


# Figure 2: C002 reference inversion in one compact forest plot. Endpoint
# intervals are 95%; depth-equivalence intervals are the registered 90% CIs.
fig, ax = plt.subplots(figsize=(3.35, 1.90))
c002_rows = [
    ("Canonical", 23.2812, 21.8861, 24.5555, 3.25, "o", COLORS["neutral"], True),
    (
        "Reference-inverted",
        0.1907,
        -0.8843,
        1.2720,
        2.55,
        "o",
        COLORS["neutral"],
        False,
    ),
    (
        "Physical LEFT\nsuccess 210 $\\to$ 151",
        -9.7831,
        -10.9227,
        -8.5857,
        0.82,
        "o",
        COLORS["left"],
        False,
    ),
    (
        "Physical RIGHT\nsuccess 297 $\\to$ 240",
        -13.3074,
        -14.0739,
        -12.5448,
        -0.12,
        "s",
        COLORS["right"],
        True,
    ),
]
for label, mean, lo, hi, yi, marker, color, filled in c002_rows:
    ax.errorbar(
        mean,
        yi,
        xerr=np.array([[mean - lo], [hi - mean]]),
        fmt=marker,
        ms=4.8,
        mfc=color if filled else "white",
        mec=color,
        mew=0.9,
        color=color,
        capsize=2.3,
        lw=1.15,
        zorder=3,
    )
    ax.annotate(
        f"{mean:+.2f}",
        (mean, yi),
        xytext=(-5 if mean > 20 else 5, 0),
        textcoords="offset points",
        va="center",
        ha="right" if mean > 20 else "left",
        fontsize=7.1,
        color=color,
    )
ax.text(
    -17.6,
    3.88,
    "Endpoint separation (95% CI)",
    fontsize=7.0,
    fontweight="bold",
    ha="left",
    va="center",
)
ax.text(
    -17.6,
    1.55,
    "Inverted - canonical depth (90% CI)",
    fontsize=7.0,
    fontweight="bold",
    ha="left",
    va="center",
)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlim(-18.0, 29.0)
ax.set_ylim(-0.70, 4.12)
ax.set_xticks([-10, 0, 10, 20])
ax.set_yticks([row[4] for row in c002_rows], [row[0] for row in c002_rows])
ax.set_xlabel("Lateral contrast (cm)")
ax.grid(axis="x", color="#d7dce0", linewidth=0.55, zorder=0)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0, pad=3, labelsize=7.2)
for tick_label in ax.get_yticklabels():
    tick_label.set_linespacing(0.9)
fig.subplots_adjust(left=0.39, right=0.98, bottom=0.23, top=0.98)
finish(fig, "reference_inversion", tight=False)


# Figure 2: exact position-reflection estimates and intervals.
fig, axs = plt.subplots(
    1, 2, figsize=(7.0, 2.25), gridspec_kw={"width_ratios": [1.18, 1]}
)
models = ["pi0.5", "Nano", "DreamZero"]
means = np.array([-34.6, -24.8, -14.1])
lo = np.array([-41.4, -32.4, -19.9])
hi = np.array([-28.5, -17.3, -8.2])
y = np.arange(3)
axs[0].barh(
    y,
    means,
    color="#c7cdd2",
    edgecolor=COLORS["neutral"],
    hatch="///",
    alpha=0.95,
)
axs[0].errorbar(
    means,
    y,
    xerr=np.vstack([means - lo, hi - means]),
    fmt="none",
    ecolor="black",
    capsize=2.5,
    lw=1,
)
axs[0].axvline(0, color="black", lw=0.8)
axs[0].set_yticks(y, models)
axs[0].invert_yaxis()
axs[0].set_xlabel("Depth interaction (cm)\nreflected minus control")
for yi, v, txt in zip(y, means, ["27/27", "24/27", "23/27"]):
    axs[0].text(v + 1.2, yi, txt, va="center", ha="left", fontsize=8)
axs[0].set_title("Placement advantage reverses")

x = np.array([0, 1])
w = 0.33
left = np.array([4 / 27, 25 / 27])
right = np.array([25 / 27, 9 / 27])
axs[1].bar(
    x - w / 2,
    left,
    w,
    label="LEFT",
    color="white",
    edgecolor=COLORS["left"],
    hatch="///",
    lw=1.2,
)
axs[1].bar(
    x + w / 2,
    right,
    w,
    label="RIGHT",
    color=COLORS["right"],
    edgecolor="black",
    lw=0.8,
)
axs[1].set_xticks(x, ["Control", "Reflected"])
axs[1].set_ylim(0, 1.16)
axs[1].set_ylabel("pi0.5 task success")
bar_items = [
    (x[0] - w / 2, left[0], "4/27", "L"),
    (x[0] + w / 2, right[0], "25/27", "R"),
    (x[1] - w / 2, left[1], "25/27", "L"),
    (x[1] + w / 2, right[1], "9/27", "R"),
]
for xx, val, label, direct in bar_items:
    axs[1].text(xx, val + 0.035, label, ha="center", fontsize=8)
    axs[1].text(xx, 0.035, direct, ha="center", va="bottom", fontsize=8)
axs[1].set_title("Completion advantage inverts")
fig.tight_layout(w_pad=1.2)
finish(fig, "reflection_core")


# Figure 3: registered seven-position reference-object sweep.
x = np.array([-9, -6, -3, 0, 3, 6, 9], dtype=float)
depth = np.array([1.3067, 9.2222, 10.2998, 9.7954, 9.2060, 17.1777, 27.8600])
depth_lo = np.array([-8.0808, 3.9959, 4.9951, 4.3885, -3.5232, 9.8856, 21.6043])
depth_hi = np.array([9.5876, 13.9880, 16.0118, 14.9294, 18.8071, 24.3819, 34.4550])
left = np.array([15, 14, 15, 14, 13, 12, 10]) / 15
right = np.array([13, 15, 14, 15, 14, 13, 15]) / 15
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.25))
axs[0].errorbar(
    x,
    depth,
    yerr=np.vstack([depth - depth_lo, depth_hi - depth]),
    fmt="o-",
    color=COLORS["green"],
    mfc="white",
    capsize=2.5,
    lw=1.3,
)
axs[0].axhline(0, color="black", lw=0.8)
axs[0].set_xlabel("Bowl displacement (cm)")
axs[0].set_ylabel("R-L requested depth (cm)")
axs[0].set_title("Placement margin changes gradually")
axs[1].plot(
    x, left, "o-", label="LEFT", color=COLORS["left"], mfc="white", lw=1.3
)
axs[1].plot(
    x, right, "s--", label="RIGHT", color=COLORS["right"], mfc="white", lw=1.3
)
axs[1].set_ylim(0.6, 1.04)
axs[1].set_xlabel("Bowl displacement (cm)")
axs[1].set_ylabel("Task success")
axs[1].set_title("Binary success is near ceiling")
axs[1].legend(frameon=False, ncol=2, loc="lower left")
fig.tight_layout(w_pad=1.2)
finish(fig, "lateral_sweep")


# Figure 4: symmetric minus control interactions with exact E004 intervals.
models = ["Edge", "Nano", "DreamZero", "pi0.5"]
b_mean = np.array([-0.148, 0.000, -0.556, -0.519])
b_lo = np.array([-0.407, -0.111, -0.815, -0.815])
b_hi = np.array([0.074, 0.111, -0.296, -0.222])
d_mean = np.array([-0.192, -0.139, -0.124, -0.124])
d_lo = np.array([-0.260, -0.203, -0.173, -0.197])
d_hi = np.array([-0.115, -0.077, -0.075, -0.049])
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.25))
y = np.arange(len(models))
for ax, mean, lo, hi, title, xlab in [
    (
        axs[0],
        b_mean,
        b_lo,
        b_hi,
        "Binary gap interaction",
        "Symmetric minus control (R-L success)",
    ),
    (
        axs[1],
        d_mean,
        d_lo,
        d_hi,
        "Requested-depth interaction",
        "Symmetric minus control (m)",
    ),
]:
    err = np.vstack([mean - lo, hi - mean])
    ax.errorbar(
        mean,
        y,
        xerr=err,
        fmt="o",
        mfc="white",
        mec=COLORS["neutral"],
        color=COLORS["neutral"],
        ecolor=COLORS["neutral"],
        capsize=3,
        lw=1.2,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.set_xlabel(xlab)
    ax.set_title(title)
fig.tight_layout(w_pad=1.3)
finish(fig, "symmetry_core")


# Figure 5: matched E004 pi0.5 failures under control and symmetry.
cats = ["Pick", "Transport", "Wrong side", "Release"]
control = np.array([1, 267, 41, 2])
sym = np.array([0, 158, 7, 0])
fig, ax = plt.subplots(figsize=(3.35, 2.18))
y = np.arange(len(cats))
offset = 0.13
ax.scatter(
    control,
    y - offset,
    marker="o",
    s=30,
    facecolors="white",
    edgecolors=COLORS["neutral"],
    label="Control",
    zorder=3,
)
ax.scatter(
    sym,
    y + offset,
    marker="s",
    s=27,
    color=COLORS["right"],
    edgecolors="black",
    linewidths=0.5,
    label="Symmetric scene",
    zorder=3,
)
for values, ys in [(control, y - offset), (sym, y + offset)]:
    for value, yy in zip(values, ys):
        ax.annotate(
            str(int(value)),
            (value, yy),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )
ax.set_xscale("symlog", linthresh=2)
ax.set_xlim(-0.4, 420)
ax.set_yticks(y, cats)
ax.invert_yaxis()
ax.set_xlabel("Failure count (symlog scale)")
ax.grid(axis="x", color="#d7dce0", linewidth=0.6, zorder=0)
ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.18))
fig.tight_layout()
finish(fig, "failure_composition")
