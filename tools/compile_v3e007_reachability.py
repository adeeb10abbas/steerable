#!/usr/bin/env python3
"""Compile V3-E007 IK volumes against the already-completed policy outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/zero_model_reachability_v3e007"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sign(value: float, tolerance: float = 1e-12) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def aggregate_rows(
    rows: Iterable[dict[str, Any]],
    *,
    model_key: str,
    layout_key: str,
    relation_key: str,
    depth_getter,
    success_getter,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[tuple[float, bool]]] = defaultdict(list)
    for row in rows:
        depth = depth_getter(row)
        success = success_getter(row)
        if depth is None or success is None:
            raise RuntimeError("behavioral source is missing a registered endpoint outcome")
        groups[(str(row[model_key]), str(row[layout_key]), str(row[relation_key]))].append(
            (float(depth), bool(success))
        )
    output = []
    for (model, layout, relation), values in groups.items():
        output.append(
            {
                "model_id": model,
                "layout": layout,
                "relation": relation,
                "episodes": len(values),
                "mean_requested_side_depth_m": float(np.mean([value[0] for value in values])),
                "successes": sum(value[1] for value in values),
                "success_rate": float(np.mean([value[1] for value in values])),
            }
        )
    return output


def paired_advantages(rows: Iterable[dict[str, Any]], family: str, layout_map) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_key[(row["model_id"], row["layout"])][row["relation"]] = row
    output = []
    for (model, layout), sides in sorted(by_key.items()):
        if set(sides) != {"left", "right"}:
            raise RuntimeError(f"incomplete LEFT/RIGHT behavioral group: {model}/{layout}")
        left, right = sides["left"], sides["right"]
        output.append(
            {
                "family": family,
                "layout_id": layout_map(layout),
                "model_id": model,
                "episodes_per_relation": [left["episodes"], right["episodes"]],
                "mean_requested_side_depth_m": {
                    "left": left["mean_requested_side_depth_m"],
                    "right": right["mean_requested_side_depth_m"],
                },
                "right_minus_left_requested_depth_m": right["mean_requested_side_depth_m"] - left["mean_requested_side_depth_m"],
                "binary_success": {
                    "left": [left["successes"], left["episodes"]],
                    "right": [right["successes"], right["episodes"]],
                },
                "right_minus_left_success_rate": right["success_rate"] - left["success_rate"],
            }
        )
    return output


def exact_spearman_permutation(x: list[float], y: list[float]) -> dict[str, Any]:
    if len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "rho": None,
            "permutations": 0,
            "two_sided_p": None,
            "undefined_reason": "constant_input",
        }
    observed = float(spearmanr(x, y).statistic)
    statistics = [
        float(spearmanr(x, permutation).statistic)
        for permutation in itertools.permutations(y)
    ]
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in statistics)
    return {
        "rho": observed,
        "permutations": len(statistics),
        "two_sided_p": extreme / len(statistics),
    }


def make_figure(results: dict[str, Any], png: Path, pdf: Path) -> None:
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.45), constrained_layout=True)
    volume_by_layout = {
        row["layout_id"]: row["right_minus_left_feasible_volume_m3"] * 1e6
        for row in results["workspace_layouts"]
    }
    models = sorted({row["model_id"] for row in results["policy_layout_rows"]})
    model_labels = {
        "cosmos3_edge_policy_droid": "Edge",
        "cosmos3_nano_policy_droid": "Nano",
        "dreamzero_droid_action_cfg": "DreamZero",
        "pi05_current_stack_droid": "π0.5",
    }
    colors = dict(zip(models, plt.cm.tab10(np.linspace(0.05, 0.85, len(models))), strict=True))

    def panel(ax, family: str, layout_ids: list[str], labels: list[str], title: str) -> None:
        x = np.arange(len(layout_ids), dtype=float)
        volumes = [volume_by_layout[layout_id] for layout_id in layout_ids]
        ax.bar(x, volumes, color="#158f8b", alpha=0.82, width=0.58, label="IK volume R-L")
        ax.axhline(0.0, color="0.25", linewidth=0.7)
        ax.set_xticks(x, labels)
        ax.tick_params(axis="x", labelsize=7)
        ax.set_ylabel("Feasible volume R-L (cm³)", color="#106c69")
        ax.tick_params(axis="y", colors="#106c69")
        ax.set_title(title, loc="left", fontweight="bold")
        if all(abs(value) < 1e-12 for value in volumes):
            ax.text(
                0.5,
                0.92,
                "160/160 feasible per side",
                transform=ax.transAxes,
                ha="center",
                va="top",
                color="#106c69",
                fontsize=7,
                fontweight="bold",
            )
        twin = ax.twinx()
        for model in models:
            rows = [
                row for row in results["policy_layout_rows"]
                if row["family"] == family and row["model_id"] == model and row["layout_id"] in layout_ids
            ]
            if not rows:
                continue
            lookup = {row["layout_id"]: row for row in rows}
            xs, ys = [], []
            for index, layout_id in enumerate(layout_ids):
                if layout_id in lookup:
                    xs.append(index)
                    ys.append(lookup[layout_id]["right_minus_left_requested_depth_m"] * 100.0)
            twin.plot(xs, ys, marker="o", markersize=3.2, linewidth=1.0, color=colors[model], label=model)
        twin.axhline(0.0, color="0.25", linewidth=0.7)
        twin.set_ylabel("Policy depth R-L (cm)", color="#6b3fa0")
        twin.tick_params(axis="y", colors="#6b3fa0")

    panel(
        axes[0],
        "reflection",
        ["reflection_control", "reflection_mirrored"],
        ["Control", "Reflected"],
        "a  Reflection",
    )
    panel(
        axes[1],
        "nano_lateral_sweep",
        [f"nano_sweep_level_{index}" for index in range(7)],
        ["-90", "-60", "-30", "0", "+30", "+60", "+90"],
        "b  Reference sweep",
    )
    axes[1].set_xlabel("Bowl displacement (mm)")
    panel(
        axes[2],
        "symmetric_scene",
        ["symmetry_s_0_00", "symmetry_s_0_25", "symmetry_s_0_50", "symmetry_s_0_75", "symmetry_s_1_00"],
        ["0", ".25", ".50", ".75", "1"],
        "c  Symmetry package",
    )
    axes[2].set_xlabel("Symmetry level s")
    # One compact model legend above the figure; volume is identified by axis color.
    model_handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[model],
            marker="o",
            linewidth=1,
            markersize=3,
            label=model_labels.get(model, model),
        )
        for model in models
    ]
    fig.legend(
        handles=model_handles,
        loc="outside upper center",
        ncol=min(4, len(model_handles)),
        frameon=False,
        title="Observed policy depth contrast",
    )
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", type=Path, default=DEFAULT_BASE / "registration.json")
    parser.add_argument("--raw-summary", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_BASE / "results")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    registration = json.loads(args.registration.read_text(encoding="utf-8"))
    raw = json.loads(args.raw_summary.read_text(encoding="utf-8"))
    if raw["registration_sha256"] != sha256(args.registration):
        raise RuntimeError("raw computation does not bind the frozen registration")
    raw_points = Path(raw["raw_points"]["path"])
    if raw_points.stat().st_size != raw["raw_points"]["bytes"] or sha256(raw_points) != raw["raw_points"]["sha256"]:
        raise RuntimeError("raw point stream differs")

    reflection_specs = [
        (
            ROOT / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/results/pi05_v3b002_episodes.jsonl",
            lambda row: row["requested_side_depth_m"],
            lambda row: row["success"],
            "phase_b_arm",
        ),
        (
            ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_mirror_v3b001/results/nano_v3b001_episodes.jsonl",
            lambda row: row["measurements"]["final_requested_signed_margin_m"],
            lambda row: row["requested_success"],
            "phase_b_arm",
        ),
        (
            ROOT / "artifacts/vla_wam_shared_v3/phase_b/dreamzero_mirror_v3b003/results/dreamzero_v3b003_episodes.jsonl",
            lambda row: row["requested_side_depth_m"],
            lambda row: row["requested_success"],
            "arm",
        ),
    ]
    policy_rows: list[dict[str, Any]] = []
    for path, depth_getter, success_getter, layout_key in reflection_specs:
        grouped = aggregate_rows(
            read_jsonl(path),
            model_key="model_id",
            layout_key=layout_key,
            relation_key="requested_relation",
            depth_getter=depth_getter,
            success_getter=success_getter,
        )
        policy_rows.extend(
            paired_advantages(
                grouped,
                "reflection",
                lambda value: "reflection_control" if value == "control" else "reflection_mirrored",
            )
        )

    sweep = json.loads(
        (ROOT / "artifacts/vla_wam_shared_v3/phase_b/nano_lateral_sweep_v3b005/results/nano_v3b005_dose_response_report.json").read_text(encoding="utf-8")
    )
    for row in sweep["by_level"]:
        policy_rows.append(
            {
                "family": "nano_lateral_sweep",
                "layout_id": f"nano_sweep_level_{row['level_index']}",
                "model_id": sweep["model_id"],
                "episodes_per_relation": [
                    row["binary_success"]["left"]["episodes"],
                    row["binary_success"]["right"]["episodes"],
                ],
                "mean_requested_side_depth_m": None,
                "right_minus_left_requested_depth_m": row["depth_contrast_B_m"]["mean"],
                "binary_success": {
                    "left": [row["binary_success"]["left"]["successes"], row["binary_success"]["left"]["episodes"]],
                    "right": [row["binary_success"]["right"]["successes"], row["binary_success"]["right"]["episodes"]],
                },
                "right_minus_left_success_rate": row["binary_success"]["right"]["rate"] - row["binary_success"]["left"]["rate"],
            }
        )

    symmetry_source = read_jsonl(
        ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004/results/episodes.jsonl"
    )
    symmetry_source = [row for row in symmetry_source if row["arena"] == "droid_robolab"]
    grouped = aggregate_rows(
        symmetry_source,
        model_key="model_id",
        layout_key="symmetry_level_s",
        relation_key="requested_relation",
        depth_getter=lambda row: row["requested_side_depth"],
        success_getter=lambda row: row["success"],
    )
    policy_rows.extend(
        paired_advantages(
            grouped,
            "symmetric_scene",
            lambda value: f"symmetry_s_{float(value):.2f}".replace(".", "_"),
        )
    )

    volume_lookup = {row["layout_id"]: row for row in raw["layouts"]}
    for row in policy_rows:
        volume = volume_lookup[row["layout_id"]]["right_minus_left_feasible_volume_m3"]
        row["right_minus_left_feasible_volume_m3"] = volume
        row["depth_sign_alignment"] = sign(volume) == sign(row["right_minus_left_requested_depth_m"]) if sign(volume) and sign(row["right_minus_left_requested_depth_m"]) else None
        row["success_sign_alignment"] = sign(volume) == sign(row["right_minus_left_success_rate"]) if sign(volume) and sign(row["right_minus_left_success_rate"]) else None

    reflection_rows = [row for row in policy_rows if row["family"] == "reflection"]
    reflection_non_ties = [row for row in reflection_rows if row["depth_sign_alignment"] is not None]
    sweep_rows = sorted(
        [row for row in policy_rows if row["family"] == "nano_lateral_sweep"],
        key=lambda row: int(row["layout_id"].rsplit("_", 1)[1]),
    )
    sweep_test = exact_spearman_permutation(
        [row["right_minus_left_feasible_volume_m3"] for row in sweep_rows],
        [row["right_minus_left_requested_depth_m"] for row in sweep_rows],
    )
    symmetry_rows = [row for row in policy_rows if row["family"] == "symmetric_scene"]
    s0_volume = abs(volume_lookup["symmetry_s_0_00"]["right_minus_left_feasible_volume_m3"])
    s1_volume = abs(volume_lookup["symmetry_s_1_00"]["right_minus_left_feasible_volume_m3"])
    endpoint_models = sorted({row["model_id"] for row in symmetry_rows if row["layout_id"] == "symmetry_s_0_00"})
    symmetry_attenuation = []
    for model in endpoint_models:
        rows = {row["layout_id"]: row for row in symmetry_rows if row["model_id"] == model}
        if "symmetry_s_0_00" not in rows or "symmetry_s_1_00" not in rows:
            continue
        symmetry_attenuation.append(
            {
                "model_id": model,
                "absolute_depth_disparity_s0_m": abs(rows["symmetry_s_0_00"]["right_minus_left_requested_depth_m"]),
                "absolute_depth_disparity_s1_m": abs(rows["symmetry_s_1_00"]["right_minus_left_requested_depth_m"]),
                "depth_disparity_decreased": abs(rows["symmetry_s_1_00"]["right_minus_left_requested_depth_m"]) < abs(rows["symmetry_s_0_00"]["right_minus_left_requested_depth_m"]),
            }
        )

    reflection_all_align = bool(reflection_non_ties) and len(reflection_non_ties) == len(reflection_rows) and all(row["depth_sign_alignment"] for row in reflection_non_ties)
    mechanism_supported = bool(
        reflection_all_align
        and sweep_test["rho"] is not None
        and sweep_test["rho"] > 0.0
    )
    results = {
        "schema_version": "vla-wam-shared-v3e007-zero-model-reachability-results-v1",
        "amendment_id": "V3-E007",
        "status": "complete",
        "analysis_character": registration["analysis_character"],
        "registration_sha256": sha256(args.registration),
        "raw_summary": {"path": str(args.raw_summary), "bytes": args.raw_summary.stat().st_size, "sha256": sha256(args.raw_summary)},
        "learned_model_request_count": 0,
        "behavioral_episode_count": 0,
        "workspace_layouts": raw["layouts"],
        "policy_layout_rows": policy_rows,
        "tests": {
            "reflection_depth_sign_concordance": {
                "aligned_non_tied_rows": sum(bool(row["depth_sign_alignment"]) for row in reflection_non_ties),
                "non_tied_rows": len(reflection_non_ties),
                "all_rows_non_tied": len(reflection_non_ties) == len(reflection_rows),
                "all_align": reflection_all_align,
            },
            "nano_sweep_exact_spearman": sweep_test,
            "symmetric_scene": {
                "absolute_volume_asymmetry_s0_m3": s0_volume,
                "absolute_volume_asymmetry_s1_m3": s1_volume,
                "volume_asymmetry_decreased": s1_volume < s0_volume,
                "policy_depth_attenuation": symmetry_attenuation,
            },
        },
        "mechanism_supported_under_frozen_rule": mechanism_supported,
        "interpretation": (
            "Kinematic reachability is supported as a contributor to the layout effect; residual policy gaps are not attributed to reachability."
            if mechanism_supported
            else "The frozen IK analysis does not support kinematic reachability as the common mechanism; scope the claim to scene configuration."
        ),
        "claim_boundary": "Post-result deterministic joint-limit pose IK. It tests a mechanism without policy inference, but does not include collision, contact, dynamics, or prove that IK volume fully mediates behavioral outcomes.",
    }
    results_path = args.output_root / "results.json"
    results_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = args.output_root / "layout_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family", "layout_id", "model_id", "right_minus_left_feasible_volume_m3",
                "right_minus_left_requested_depth_m", "right_minus_left_success_rate",
                "depth_sign_alignment", "success_sign_alignment",
            ],
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in policy_rows)

    png = args.output_root / "v3e007_reachability_mechanism.png"
    pdf = args.output_root / "v3e007_reachability_mechanism.pdf"
    make_figure(results, png, pdf)
    memo = args.output_root / "DECISION_MEMO.md"
    sweep_sentence = (
        f"The seven-level Nano sweep had Spearman rho={sweep_test['rho']:.3f} "
        f"(exact two-sided p={sweep_test['two_sided_p']:.4g}).\n\n"
        if sweep_test["rho"] is not None
        else "The seven-level volume contrast was constant, so rank correlation is undefined.\n\n"
    )
    memo.write_text(
        "# V3-E007 zero-model reachability result\n\n"
        + f"The frozen mechanism rule is **{'supported' if mechanism_supported else 'not supported'}**. "
        + f"Reflection depth-sign concordance was {results['tests']['reflection_depth_sign_concordance']['aligned_non_tied_rows']}/"
        + f"{results['tests']['reflection_depth_sign_concordance']['non_tied_rows']}. "
        + sweep_sentence
        + "This is a disclosed post-result CPU-only calculation with zero model requests and zero behavioral episodes. "
        + "It measures strict joint-limit pose-IK volume, not collision-free task feasibility, contact, or dynamics.\n",
        encoding="utf-8",
    )
    manifest_paths = [results_path, csv_path, png, pdf, memo]
    manifest = {
        "schema_version": "vla-wam-shared-v3e007-evidence-manifest-v1",
        "registration_sha256": sha256(args.registration),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in manifest_paths
        ],
        "raw": results["raw_summary"],
    }
    (args.output_root / "evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"results": str(results_path), "sha256": sha256(results_path), "mechanism_supported": mechanism_supported}, indent=2))


if __name__ == "__main__":
    main()
