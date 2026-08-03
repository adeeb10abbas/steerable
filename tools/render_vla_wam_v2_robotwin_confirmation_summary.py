#!/usr/bin/env python3
"""Render the completed three-WAM RoboTwin direction-confirmation summary."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


BACKGROUND = "#F7F4EE"
CARD = "#FFFEFB"
TEXT = "#172A3A"
MUTED = "#61717F"
GRID = "#D7DEE3"
LEFT = "#D97706"
RIGHT = "#2563A8"
ALIGNED = "#138A62"
ANTI = "#C8D0D6"
TRACE = "#7C3AED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def file_record(path: Path, workspace: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(workspace)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def summarize_slice(path: Path, expected_model_id: str) -> dict[str, Any]:
    payload = load_json(path)
    if payload["model_id"] != expected_model_id:
        raise RuntimeError(f"Unexpected model_id in {path}: {payload['model_id']}")
    episodes = payload["episodes"]
    pairs = payload["summary"]["paired_endpoint_responses"]
    if len(episodes) != 14 or len(pairs) != 7:
        raise RuntimeError(f"{path} must contain 14 episodes and 7 paired responses")

    episode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        episode_groups[episode["pair_id"]].append(episode)
    expected_pairs = {f"robotwin_pair_{index:02d}" for index in range(3, 10)}
    if set(episode_groups) != expected_pairs:
        raise RuntimeError(f"{path} does not contain exactly pairs03-09")

    distinct = 0
    for response in pairs:
        pair_episodes = episode_groups[response["pair_id"]]
        if {episode["requested_relation"] for episode in pair_episodes} != {"left", "right"}:
            raise RuntimeError(f"Incomplete relation pair in {path}: {response['pair_id']}")
        trace_hashes = {episode["action_trace"]["sha256"] for episode in pair_episodes}
        rms = response["first_ten_executed_action_rms"]
        if len(trace_hashes) == 2 and rms is not None and float(rms) > 0:
            distinct += 1

    summary = payload["summary"]
    return {
        "model_id": payload["model_id"],
        "model_label": payload["model_label"],
        "source_paths": [path],
        "episode_count": len(episodes),
        "pair_count": len(pairs),
        "left_successes": summary["by_direction"]["left"]["successes"],
        "right_successes": summary["by_direction"]["right"]["successes"],
        "aligned_pairs": sum(
            response["endpoint_response_direction"] == "aligned" for response in pairs
        ),
        "distinct_trace_pairs": distinct,
        "future_interface_counts": summary["future_interface_counts"],
    }


def summarize_efficient(pair03_path: Path, pairs04_09_path: Path) -> dict[str, Any]:
    pair03 = load_json(pair03_path)
    later = load_json(pairs04_09_path)
    if pair03["model_id"] != "efficient_wam_rt_robotwin":
        raise RuntimeError(f"Unexpected model_id in {pair03_path}")
    if later["model_id"] != "efficient_wam_rt_robotwin":
        raise RuntimeError(f"Unexpected model_id in {pairs04_09_path}")
    if pair03["pair"]["pair_id"] != "robotwin_pair_03":
        raise RuntimeError(f"{pair03_path} is not the pair03 integration")
    if len(pair03["cells"]) != 2 or len(later["episodes"]) != 12:
        raise RuntimeError("Efficient-WAM inputs must contain 2 + 12 valid episodes")

    by_relation: dict[str, int] = {"left": 0, "right": 0}
    for cell in pair03["cells"]:
        by_relation[cell["requested_relation"]] += int(cell["requested_success"])
    for relation in by_relation:
        by_relation[relation] += later["summary"]["by_direction"][relation]["successes"]

    pair03_trace_hashes = {
        cell["files"]["action_trace"]["sha256"] for cell in pair03["cells"]
    }
    pair03_rms = pair03["paired_metrics"]["first_ten_executed_action_rms"]
    pair03_distinct = len(pair03_trace_hashes) == 2 and float(pair03_rms) > 0
    later_pairs = later["summary"]["paired_endpoint_responses"]
    later_episode_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in later["episodes"]:
        later_episode_groups[episode["pair_id"]].append(episode)
    later_distinct = sum(
        len({episode["action_trace"]["sha256"] for episode in later_episode_groups[row["pair_id"]]})
        == 2
        and row["first_ten_executed_action_rms"] is not None
        and float(row["first_ten_executed_action_rms"]) > 0
        for row in later_pairs
    )

    return {
        "model_id": "efficient_wam_rt_robotwin",
        "model_label": "Efficient-WAM-RT",
        "source_paths": [pair03_path, pairs04_09_path],
        "episode_count": 14,
        "pair_count": 7,
        "left_successes": by_relation["left"],
        "right_successes": by_relation["right"],
        "aligned_pairs": later["summary"]["aligned_endpoint_pairs"],
        "distinct_trace_pairs": later_distinct + int(pair03_distinct),
        "future_interface_counts": {"decoded_future_video": 14},
    }


def interface_label(counts: dict[str, int]) -> str:
    if set(counts) == {"decoded_future_video"}:
        return "decoded future video + actions"
    if set(counts) == {"action_only_not_applicable"}:
        return "action-only at test time"
    if set(counts) == {"latent_only_future_not_decodable"}:
        return "latent future + actions (not decoded)"
    return "future interface reported separately"


def draw_count_cells(
    fig: plt.Figure,
    *,
    x: float,
    y: float,
    count: int,
    total: int,
    color: str,
) -> None:
    cell_width = 0.018
    gap = 0.005
    for index in range(total):
        fig.patches.append(
            FancyBboxPatch(
                (x + index * (cell_width + gap), y),
                cell_width,
                0.027,
                boxstyle="round,pad=0.001,rounding_size=0.004",
                transform=fig.transFigure,
                facecolor=color if index < count else "white",
                edgecolor=color if index < count else GRID,
                linewidth=1.0,
            )
        )


def render(rows: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TEXT})
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BACKGROUND)
    fig.text(
        0.055,
        0.945,
        "Did static language steer the three WAMs?",
        fontsize=27,
        weight="bold",
        va="top",
    )
    fig.text(
        0.055,
        0.895,
        "Prospective RoboTwin pairs03–09 · same scene and seed within each pair · only the LEFT/RIGHT direct command changes · 42 valid episodes",
        fontsize=12,
        color=MUTED,
        va="top",
    )

    card_y = [0.635, 0.385, 0.135]
    for row, y in zip(rows, card_y, strict=True):
        fig.patches.append(
            FancyBboxPatch(
                (0.055, y),
                0.89,
                0.205,
                boxstyle="round,pad=0.008,rounding_size=0.014",
                transform=fig.transFigure,
                facecolor=CARD,
                edgecolor=GRID,
                linewidth=1.1,
            )
        )
        fig.text(0.078, y + 0.160, row["model_label"], fontsize=17, weight="bold")
        fig.text(
            0.078,
            y + 0.126,
            interface_label(row["future_interface_counts"]),
            fontsize=9.7,
            color=MUTED,
        )

        fig.text(0.295, y + 0.160, "REQUESTED SUCCESS", fontsize=9.2, weight="bold", color=MUTED)
        fig.text(0.295, y + 0.110, f"LEFT  {row['left_successes']}/7", fontsize=11, weight="bold", color=LEFT)
        draw_count_cells(fig, x=0.382, y=y + 0.098, count=row["left_successes"], total=7, color=LEFT)
        fig.text(0.295, y + 0.058, f"RIGHT {row['right_successes']}/7", fontsize=11, weight="bold", color=RIGHT)
        draw_count_cells(fig, x=0.382, y=y + 0.046, count=row["right_successes"], total=7, color=RIGHT)

        fig.text(0.590, y + 0.160, "MATCHED-PAIR RESPONSE", fontsize=9.2, weight="bold", color=MUTED)
        fig.text(0.590, y + 0.110, "endpoint ordering aligned", fontsize=10.3)
        fig.patches.append(Rectangle((0.747, y + 0.099), 0.158, 0.025, transform=fig.transFigure, facecolor=ANTI, edgecolor="none"))
        fig.patches.append(Rectangle((0.747, y + 0.099), 0.158 * row["aligned_pairs"] / 7, 0.025, transform=fig.transFigure, facecolor=ALIGNED, edgecolor="none"))
        fig.text(0.916, y + 0.111, f"{row['aligned_pairs']}/7", fontsize=11, weight="bold", ha="right", va="center")
        fig.text(0.590, y + 0.058, "first 10 executed actions differ", fontsize=10.3)
        fig.patches.append(Rectangle((0.747, y + 0.047), 0.158, 0.025, transform=fig.transFigure, facecolor=ANTI, edgecolor="none"))
        fig.patches.append(Rectangle((0.747, y + 0.047), 0.158 * row["distinct_trace_pairs"] / 7, 0.025, transform=fig.transFigure, facecolor=TRACE, edgecolor="none"))
        fig.text(0.916, y + 0.059, f"{row['distinct_trace_pairs']}/7", fontsize=11, weight="bold", ha="right", va="center")

    fig.text(
        0.055,
        0.060,
        "Read the two diagnostics separately:",
        fontsize=10.5,
        weight="bold",
    )
    fig.text(
        0.275,
        0.060,
        "different executed traces show language sensitivity; aligned endpoints show whether that change followed the requested LEFT→RIGHT ordering.",
        fontsize=10.5,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.025,
        "Every valid failure remains in the denominator. Infrastructure-invalid and partial attempts are excluded. No DROID results are pooled here.",
        fontsize=9.5,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=BACKGROUND)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/figures/robotwin_wam_confirmation_pairs03_09_1600x900.png"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/figures/robotwin_wam_confirmation_pairs03_09_manifest.json"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    manifest_path = args.manifest if args.manifest.is_absolute() else workspace / args.manifest
    evidence = workspace / "artifacts/vla_wam_shared_v2/pilot/directional_confirmation"

    rows = [
        summarize_efficient(
            evidence / "efficient_wam_rt_pair03_integration.json",
            evidence / "efficient_wam_rt_pairs04_09_slice.json",
        ),
        summarize_slice(evidence / "fastwam_pairs03_09_slice.json", "fastwam_robotwin"),
        summarize_slice(evidence / "lingbot_va_pairs03_09_slice.json", "lingbot_va_robotwin"),
    ]
    for row in rows:
        if row["episode_count"] != 14 or row["pair_count"] != 7:
            raise RuntimeError(f"Incomplete confirmation summary for {row['model_id']}")
    render(rows, output)

    source_paths = [path for row in rows for path in row.pop("source_paths")]
    manifest = {
        "schema_version": "vla-wam-shared-v2-robotwin-confirmation-summary-figure-v1",
        "status": "complete",
        "scope": "RoboTwin prospective direct-command pairs03-09 only",
        "valid_episode_count": sum(row["episode_count"] for row in rows),
        "model_pair_count": sum(row["pair_count"] for row in rows),
        "sources": [file_record(path, workspace) for path in source_paths],
        "summary": rows,
        "figures": {"landscape": file_record(output, workspace)},
        "claim_boundary": "Requested success and endpoint ordering are physical behavioral evidence. Distinct first-ten executed action traces establish command sensitivity only, not correct steering. Valid failures remain in denominators; infrastructure-invalid and partial attempts are excluded. DROID evidence is not pooled.",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
