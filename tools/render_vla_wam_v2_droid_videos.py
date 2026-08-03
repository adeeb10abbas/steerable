#!/usr/bin/env python3
"""Render the deterministic paired pi0-FAST DROID success/failure evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from render_vla_wam_v2_robotwin_videos import (
    CARD,
    GRID,
    LEFT,
    MUTED,
    PAPER,
    RIGHT,
    TEXT,
    ffprobe_duration,
    file_record,
    render_poster,
)


PAIR_ID = "droid_pair_seed_8300"
MODEL_LABEL = "π0-FAST DROID"
INTERFACE_NOTE = "VLA · actions only"


def status_text(episode: dict[str, Any]) -> str:
    if episode["requested_success"]:
        return "SUCCESS · released inside requested region"
    if episode["ever_entered_requested_region"]:
        return "FAIL · entered region but did not release there"
    if episode["verified_pickup_proxy"]:
        return "FAIL · picked up, never entered requested region"
    return "FAIL · no object interaction"


def load_trajectory(episode: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trajectory = json.loads(Path(episode["raw_trajectory"]["path"]).read_text())
    return (
        np.asarray([step["object_minus_target_x"] for step in trajectory]),
        np.asarray([step["object_minus_target_y"] for step in trajectory]),
    )


def draw_path_axis(axis, episode: dict[str, Any], direction: str) -> None:
    xs, ys = load_trajectory(episode)
    color = LEFT if direction == "left" else RIGHT
    desired_x = -0.14 if direction == "left" else 0.14
    axis.axvspan(-0.20, -0.08, color=LEFT, alpha=0.10, zorder=0)
    axis.axvspan(0.08, 0.20, color=RIGHT, alpha=0.08, zorder=0)
    axis.axvline(0, color=TEXT, linewidth=0.8, alpha=0.7)
    axis.axhline(0, color=GRID, linewidth=0.7)
    axis.annotate(
        "",
        xy=(desired_x, 0),
        xytext=(float(xs[0]), float(ys[0])),
        arrowprops={
            "arrowstyle": "->",
            "color": color,
            "linestyle": ":",
            "linewidth": 1.5,
            "alpha": 0.75,
        },
        zorder=1,
    )
    axis.plot(xs, ys, color=color, linewidth=2.0, zorder=2)
    axis.scatter(
        [xs[0]], [ys[0]], marker="D", s=28, facecolor=TEXT,
        edgecolor=CARD, linewidth=0.6, zorder=4,
    )
    marker = "o" if episode["requested_success"] else "X"
    axis.scatter(
        [xs[-1]], [ys[-1]], marker=marker, s=52, facecolor=color,
        edgecolor=CARD if marker == "o" else color, linewidth=1.0, zorder=5,
    )
    axis.set_xlim(-0.45, 0.45)
    axis.set_ylim(-0.30, 0.30)
    axis.set_xticks([-0.4, -0.2, 0, 0.2, 0.4])
    axis.set_xticklabels(["−.4", "−.2", "TARGET", "+.2", "+.4"], fontsize=7.5)
    axis.set_yticks([])
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.set_title(
        f"Expected: {direction.upper()} band (dashed)  ·  actual endpoint x={float(xs[-1]):+.3f} m",
        loc="left", fontsize=8.2, color=TEXT, pad=3,
    )


def render_background(
    left_episode: dict[str, Any], right_episode: dict[str, Any], output: Path
) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=PAPER)
    fig.text(0.035, 0.95, f"{MODEL_LABEL}: same scene, one word changed",
             fontsize=25, weight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.965, 0.95, INTERFACE_NOTE, fontsize=10.5, color=MUTED,
             ha="right", va="top")
    fig.text(
        0.035, 0.897,
        "Matched DROID intervention · seed 8300 · no oracle or subtask coach · exact direct commands.",
        fontsize=11, color=MUTED, ha="left", va="top",
    )
    for episode, direction, color, x0 in [
        (left_episode, "left", LEFT, 0.035),
        (right_episode, "right", RIGHT, 0.5225),
    ]:
        fig.patches.append(Rectangle((x0, 0.805), 0.4425, 0.072,
                           transform=fig.transFigure, facecolor=color,
                           alpha=0.09, edgecolor="none"))
        fig.text(x0 + 0.012, 0.855, "ASKED " + direction.upper(), fontsize=10,
                 weight="bold", color=TEXT, ha="left", va="center")
        prompt_lines = "\n".join(textwrap.wrap(episode["prompt"], 49))
        fig.text(x0 + 0.105, 0.84, f'“{prompt_lines}”', fontsize=11.2,
                 weight="bold", color=TEXT, ha="left", va="center")
        fig.text(x0 + 0.012, 0.79, status_text(episode), fontsize=10.5,
                 weight="bold", color=TEXT, ha="left", va="top")
        fig.text(x0 + 0.43, 0.79, f"{episode['actions_executed']} actions",
                 fontsize=9.5, color=MUTED, ha="right", va="top")
        fig.patches.append(Rectangle((x0 - 0.002, 0.286), 0.448, 0.48,
                           transform=fig.transFigure, facecolor=CARD,
                           edgecolor=GRID, linewidth=1.2, zorder=-1))
        path_axis = fig.add_axes((x0 + 0.012, 0.055, 0.418, 0.16))
        draw_path_axis(path_axis, episode, direction)
    fig.text(
        0.5, 0.235,
        "Robot viewport above · target-relative Rubik’s-cube path below · right-positive display convention",
        fontsize=9.2, color=MUTED, ha="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_square_background(
    left_episode: dict[str, Any], right_episode: dict[str, Any], output: Path
) -> None:
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=PAPER)
    fig.text(0.04, 0.967, f"{MODEL_LABEL}: did one word redirect the robot?",
             fontsize=22, weight="bold", color=TEXT, ha="left", va="top")
    fig.text(0.96, 0.927, INTERFACE_NOTE, fontsize=9.5, color=MUTED,
             ha="right", va="top")
    fig.text(0.04, 0.927, "Same DROID scene and seed · no oracle or subtask coach.",
             fontsize=10.2, color=MUTED, ha="left", va="top")
    for episode, direction, color, x0 in [
        (left_episode, "left", LEFT, 0.04),
        (right_episode, "right", RIGHT, 0.515),
    ]:
        fig.patches.append(Rectangle((x0, 0.82), 0.445, 0.077,
                           transform=fig.transFigure, facecolor=color,
                           alpha=0.09, edgecolor="none"))
        fig.text(x0 + 0.012, 0.88, "ASKED " + direction.upper(), fontsize=9.5,
                 weight="bold", color=TEXT, ha="left", va="center")
        prompt_lines = "\n".join(textwrap.wrap(episode["prompt"], 39))
        fig.text(x0 + 0.012, 0.846, f'“{prompt_lines}”', fontsize=10,
                 weight="bold", color=TEXT, ha="left", va="center")
        fig.text(x0 + 0.012, 0.805, status_text(episode), fontsize=9.2,
                 weight="bold", color=TEXT, ha="left", va="top")
        fig.text(x0 + 0.43, 0.805, f"{episode['actions_executed']} actions",
                 fontsize=8.5, color=MUTED, ha="right", va="top")
        fig.patches.append(Rectangle((x0 - 0.005, 0.49), 0.458, 0.30,
                           transform=fig.transFigure, facecolor=CARD,
                           edgecolor=GRID, linewidth=1.2, zorder=-1))
        path_axis = fig.add_axes((x0 + 0.018, 0.12, 0.405, 0.26))
        draw_path_axis(path_axis, episode, direction)
        path_axis.tick_params(axis="x", labelsize=8.5)
    fig.text(0.5, 0.425, "Complete robot viewport above · state-derived path below",
             fontsize=9.5, color=MUTED, ha="center", weight="bold")
    fig.text(0.5, 0.052,
             "Circle = relation-and-release success · cross = failure · dashed route is illustrative only",
             fontsize=9.2, color=MUTED, ha="center")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_video(
    background: Path, left: Path, right: Path, output: Path, *, square: bool
) -> float:
    duration = max(ffprobe_duration(left), ffprobe_duration(right))
    if square:
        size, left_xy, right_xy = "540:300", "48:252", "618:252"
    else:
        size, left_xy, right_xy = "700:389", "60:211", "840:211"
    filter_graph = (
        f"[1:v]fps=10,scale={size}:flags=lanczos,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[lv];"
        f"[2:v]fps=10,scale={size}:flags=lanczos,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[rv];"
        f"[0:v][lv]overlay={left_xy}:shortest=0[tmp];"
        f"[tmp][rv]overlay={right_xy}:shortest=1[outv]"
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", "10",
        "-i", str(background), "-i", str(left), "-i", str(right),
        "-filter_complex", filter_graph, "-map", "[outv]", "-t", f"{duration:.3f}",
        "-r", "10", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    return duration


def write_vtt(
    path: Path, duration: float, left_episode: dict[str, Any], right_episode: dict[str, Any]
) -> None:
    millis = int(round(duration * 1000))
    minutes, remainder = divmod(millis, 60_000)
    seconds, ms = divmod(remainder, 1000)
    end = f"00:{minutes:02d}:{seconds:02d}.{ms:03d}"
    path.write_text(
        "WEBVTT\n\n"
        f"00:00:00.000 --> {end}\n"
        f"{MODEL_LABEL}. Same DROID scene and seed. Left panel prompt: {left_episode['prompt']} "
        f"Result: {status_text(left_episode)}. Right panel prompt: {right_episode['prompt']} "
        f"Result: {status_text(right_episode)}.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/vla_wam_shared_v2/media/droid_pi0_fast_pairs"),
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = workspace / "artifacts/vla_wam_shared_v2/pilot/results/pi0_fast_direct_gate.json"
    result = json.loads(result_path.read_text())
    pair = {episode["requested_relation"]: episode for episode in result["episodes"]
            if episode["pair_id"] == PAIR_ID}
    if set(pair) != {"left", "right"}:
        raise RuntimeError(f"Incomplete pair {PAIR_ID}")
    left_episode, right_episode = pair["left"], pair["right"]
    if left_episode["requested_success"] or not right_episode["requested_success"]:
        raise RuntimeError("Expected the frozen seed-8300 LEFT-failure/RIGHT-success pair")

    slug = "pi0_fast_seed8300_left_failure_right_success"
    background = output_dir / f"{slug}_background.png"
    square_background = output_dir / f"{slug}_square_background.png"
    video = output_dir / f"{slug}.mp4"
    square_video = output_dir / f"{slug}_1200x1200.mp4"
    poster = output_dir / f"{slug}_poster.jpg"
    square_poster = output_dir / f"{slug}_1200x1200_poster.jpg"
    captions = output_dir / f"{slug}.vtt"
    render_background(left_episode, right_episode, background)
    render_square_background(left_episode, right_episode, square_background)
    left_video = Path(left_episode["executed_video"]["path"])
    right_video = Path(right_episode["executed_video"]["path"])
    duration = render_video(background, left_video, right_video, video, square=False)
    square_duration = render_video(
        square_background, left_video, right_video, square_video, square=True
    )
    if abs(square_duration - duration) > 1e-6:
        raise RuntimeError("Landscape/square duration mismatch")
    render_poster(video, poster, duration)
    render_poster(square_video, square_poster, duration)
    write_vtt(captions, duration, left_episode, right_episode)
    alt_text = (
        "Side-by-side pi0-FAST DROID rollouts from seed 8300. The LEFT prompt fails "
        f"without object interaction at x={left_episode['final_lateral_display_m']:+.3f} m; "
        f"the mirrored RIGHT prompt succeeds at x={right_episode['final_lateral_display_m']:+.3f} m. "
        "Target-relative path strips show the requested lateral bands and actual Rubik's-cube paths."
    )
    index = {
        "schema_version": "vla-wam-shared-v2-droid-paired-media-v1",
        "status": "complete",
        "selection_scope": "The first compiled exact-prompt pi0-FAST pair: seed 8300 LEFT failure and matched RIGHT success.",
        "selection_rule": "First compiled same-seed pair containing a LEFT failure and RIGHT success; selected before rendering.",
        "shorter_rollout_policy": "Hold the final frame of the shorter rollout until the longer matched rollout ends; never trim the longer failure.",
        "metric_boundary": "The dashed path is illustrative. Success is determined from recorded 3D state by the full relation-and-release checker, not from pixels or the drawn route.",
        "item": {
            "id": slug,
            "model_id": result["model_id"],
            "model_label": MODEL_LABEL,
            "interface_note": INTERFACE_NOTE,
            "arena": "droid_rubiks_cube_relative_to_bowl",
            "pair_id": PAIR_ID,
            "environment_seed": left_episode["environment_seed"],
            "sampling_seed": left_episode["sampling_seed"],
            "duration_seconds": duration,
            "left": {
                "prompt": left_episode["prompt"],
                "success": left_episode["requested_success"],
                "final_lateral_display_m": left_episode["final_lateral_display_m"],
                "failure_stage": left_episode["failure_stage"],
                "source_video": file_record(left_video),
                "source_trajectory": left_episode["raw_trajectory"],
            },
            "right": {
                "prompt": right_episode["prompt"],
                "success": right_episode["requested_success"],
                "final_lateral_display_m": right_episode["final_lateral_display_m"],
                "failure_stage": right_episode["failure_stage"],
                "source_video": file_record(right_video),
                "source_trajectory": right_episode["raw_trajectory"],
            },
            "video": file_record(video, workspace),
            "poster": file_record(poster, workspace),
            "square_video": file_record(square_video, workspace),
            "square_poster": file_record(square_poster, workspace),
            "captions": file_record(captions, workspace),
            "alt_text": alt_text,
        },
    }
    (output_dir / "media_index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    background.unlink()
    square_background.unlink()
    print(json.dumps({"video": str(video), "duration_seconds": duration}, indent=2))


if __name__ == "__main__":
    main()
