"""Render descriptive historical-label statistics, never corrected fidelity.

Run after evidence_audit.py. Numeric labels and bars are derived from its JSON.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    package = Path(__file__).resolve().parents[1]
    source = package / "results" / "audit_summary.json"
    audit = json.loads(source.read_text())
    stats = audit["legacy_replay"]
    arithmetic = stats["historical_label_arithmetic"]
    pairs = [
        (arithmetic["observed_agreement_numerator"], arithmetic["denominator"]),
        (stats["certain"], stats["n"]),
        (stats["covered_executed_positive"], stats["executed_positive"]),
    ]
    percentages = [100 * numerator / denominator for numerator, denominator in pairs]
    labels = ["Agreement among\nscored labels", "Coverage across\nall chunks", "Coverage on\npositive endpoints"]
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
    })
    fig, ax = plt.subplots(figsize=(5.5, 1.95))
    positions = [2, 1, 0]
    ax.barh(positions, [100] * 3, height=.42, color="#EDF0F2", zorder=1)
    ax.barh(positions, percentages, height=.42,
            color=["#354F66", "#477F8C", "#477F8C"], zorder=2)
    for y, percentage, pair in zip(positions, percentages, pairs):
        ax.text(105, y, f"{percentage:.1f}%", va="center", ha="left",
                fontweight="bold", color="#162B3B")
        ax.text(105, y - .30, f"{pair[0]}/{pair[1]}", va="center", ha="left",
                fontsize=7.5, color="#505C66")
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, pad=8)
    ax.set_xlim(0, 124)
    ax.set_ylim(-.57, 2.42)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.tick_params(axis="x", length=0, colors="#64727C", labelsize=7.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linewidth=.5, color="#D9DFE3")
    fig.subplots_adjust(left=.235, right=.985, top=.97, bottom=.14)
    output = package / "paper" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "coverage_audit.pdf", metadata={"Title": "Historical label agreement and coverage", "CreationDate": None})
    fig.savefig(output / "coverage_audit.png", dpi=220)
    plt.close(fig)
    print(json.dumps({"input": str(source), "pairs": pairs, "percentages": percentages,
                      "outputs": [str(output / "coverage_audit.pdf"), str(output / "coverage_audit.png")]}))


if __name__ == "__main__":
    main()
