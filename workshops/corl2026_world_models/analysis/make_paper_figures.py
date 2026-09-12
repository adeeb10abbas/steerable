"""Paper figures from verified executed-behavior results; no invented rollouts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


def main():
    package = Path(__file__).resolve().parents[1]
    results = json.loads((package / "results/paper_results.json").read_text())
    output = package / "paper/figures"
    output.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False})
    original, reflected = "#245D80", "#BD6432"

    # Qualitative plan view. Coordinates illustrate reflection only and are
    # deliberately not presented as measured object positions or trajectories.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.52))
    for ax, mirror, title in zip(axes, [False, True], ["Original layout", "Reflected layout"]):
        x = lambda value: 1 - value if mirror else value
        ax.set(xlim=(0, 1), ylim=(0, .90), aspect="equal")
        ax.axis("off")
        ax.add_patch(Rectangle((.05, .06), .90, .73, facecolor="#F3F5F5", edgecolor="#D3DBDF", lw=.7))
        ax.plot([.5, .5], [.14, .77], "--", color="#98A7B0", lw=.7)
        ax.add_patch(Circle((x(.62), .59), .075, facecolor="white", edgecolor="#424C54", lw=1.4))
        ax.text(x(.62), .72, "Bowl", ha="center", fontsize=7.4)
        ax.add_patch(Rectangle((x(.28)-.047, .36), .094, .094, facecolor="#478F88", edgecolor="#225E5A", lw=1))
        ax.text(x(.28), .29, "Cube", ha="center", fontsize=7.4)
        ax.add_patch(Rectangle((x(.82)-.035, .35), .07, .07, facecolor="#B8C0C5", edgecolor="none"))
        ax.add_patch(Rectangle((.41, .045), .18, .11, facecolor="#4E5860", edgecolor="none"))
        ax.add_patch(FancyArrowPatch((.5, .12), (.5, .28), arrowstyle="-|>", mutation_scale=8, color="#4E5860", lw=1))
        ax.text(.5, -.04, "Robot stays fixed", ha="center", va="top", fontsize=7)
        ax.text(.13, .82, "LEFT", ha="center", fontsize=7, color="#59666E")
        ax.text(.87, .82, "RIGHT", ha="center", fontsize=7, color="#59666E")
        ax.set_title(title, fontsize=8.7, fontweight="bold", pad=3)
    fig.subplots_adjust(left=.05, right=.95, bottom=.12, top=.81, wspace=.18)
    for suffix in ["pdf", "png"]:
        fig.savefig(output / f"reflection_design.{suffix}", dpi=220,
                    metadata={"Title": "Schematic object-layout reflection"} if suffix == "pdf" else None)
    plt.close(fig)

    fig, (left, right) = plt.subplots(1, 2, figsize=(5.5, 2.02), gridspec_kw={"width_ratios": [1, 1.25]})
    names, keys = ["Nano", "DreamZero"], ["nano_reflection", "dreamzero_reflection"]
    for i, key in enumerate(keys):
        cohort = results["cohorts"][key]
        for j, (arm, color) in enumerate([("control", original), ("position_mirrored", reflected)]):
            n = sum(cohort["conditions"][f"{arm}:{direction}"]["episodes"] for direction in ["left", "right"])
            successes = sum(cohort["conditions"][f"{arm}:{direction}"]["successes"] for direction in ["left", "right"])
            xpos = i + (j - .5) * .35
            left.bar(xpos, successes/n*100, width=.30, color=color, zorder=3)
            left.text(xpos, successes/n*100+3, f"{successes}/{n}", ha="center", fontsize=7)
        effects = cohort["effects"]["placement_depth_gap_m"]
        means = [100*effects[arm]["mean"] for arm in ["control", "position_mirrored"]]
        right.plot(means, [1-i, 1-i], color="#B8C1C7", lw=1.3, zorder=1)
        for arm, color, marker in [("control", original, "o"), ("position_mirrored", reflected, "s")]:
            effect = effects[arm]
            value = 100*effect["mean"]
            lo, hi = [100*v for v in effect["mean_ci95"]]
            uncertainty = [[value-lo], [hi-value]] if key == "nano_reflection" else None
            right.errorbar(value, 1-i, xerr=uncertainty, fmt=marker, color=color,
                           markersize=4.5, capsize=2, elinewidth=.8, zorder=3)
            right.text(value, 1-i+.17, f"{value:+.1f}", ha="center", color=color, fontsize=7.5)
    left.set(xticks=[0, 1], xticklabels=names, ylim=(0, 117), yticks=[0, 50, 100])
    left.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    left.set_title("a  Task completion", loc="left", fontsize=8.5, fontweight="bold", pad=16)
    left.grid(axis="y", color="#E2E6E8", linewidth=.5, zorder=0)
    left.spines["left"].set_visible(False)
    left.spines["bottom"].set_color("#BAC3C9")
    left.tick_params(length=0, labelsize=7.3)
    right.set(xlim=(-22, 25), ylim=(-.4, 1.5), yticks=[1, 0], yticklabels=names, xticks=[-20, 0, 20])
    right.set_title("b  Placement depth", loc="left", fontsize=8.5, fontweight="bold", pad=16)
    right.axvline(0, color="#77858D", ls="--", lw=.7, zorder=0)
    right.set_xlabel("RIGHT minus LEFT depth (cm)", fontsize=7.3, labelpad=3)
    right.spines["left"].set_visible(False)
    right.spines["bottom"].set_color("#BAC3C9")
    right.tick_params(length=0, labelsize=7.3)
    handles = [Line2D([0], [0], color=original, marker="o", lw=0, label="Original layout"),
               Line2D([0], [0], color=reflected, marker="s", lw=0, label="Reflected layout")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.01), ncol=2,
               frameon=False, fontsize=7.5, handletextpad=.25, columnspacing=1.8)
    fig.subplots_adjust(left=.075, right=.985, bottom=.20, top=.69, wspace=.43)
    for suffix in ["pdf", "png"]:
        fig.savefig(output / f"reflection_results.{suffix}", dpi=220,
                    metadata={"Title": "Executed task performance under layout reflection"} if suffix == "pdf" else None)
    plt.close(fig)
    print("Wrote reflection_design and reflection_results as vector PDF and PNG.")


if __name__ == "__main__":
    main()
