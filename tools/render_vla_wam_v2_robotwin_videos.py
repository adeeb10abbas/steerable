#!/usr/bin/env python3
"""Render deterministic paired WAM success/failure videos for the v2 article."""

from __future__ import annotations

import argparse
import hashlib
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


TEXT = "#182B3A"
MUTED = "#5D7182"
PAPER = "#F7F4EE"
CARD = "#FFFEFB"
GRID = "#D5DEE3"
LEFT = "#D97706"
RIGHT = "#2869AD"

SELECTIONS = [
    ("efficient_wam_rt", "pair00", "Efficient-WAM-RT", "decoded future video available"),
    (
        "fastwam",
        "pair02",
        "FastWAM",
        "action-only · source pixel layout reconstructed",
    ),
    ("lingbot_va", "pair00", "LingBot-VA", "predicted future retained as latent"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, workspace: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    display_path: str
    if workspace is not None:
        try:
            display_path = str(resolved.relative_to(workspace))
        except ValueError:
            display_path = str(resolved)
    else:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def ffprobe_duration(path: Path) -> float:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(completed.stdout.strip())


def reconstruct_fastwam_head_video(source: Path, output: Path) -> dict[str, Any]:
    """Recover head RGB from a known 640x480 buffer encoded as 320x240 packets.

    FastWAM's RoboTwin fork writes a 640x480 composite buffer, while the pilot
    runner declared the ffmpeg raw stream as 320x240. Each intended frame was
    therefore encoded as four sequential packets. Reassemble the decoded RGB
    packets, then retain the 320x240 head-camera crop used by the other WAMs.
    """
    width, height, fps = 320, 240, 10
    frame_bytes = width * height * 3
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    if decoder.stdout is None or encoder.stdin is None:
        raise RuntimeError("Failed to open FastWAM reconstruction pipes")
    reconstructed_frames = 0
    incomplete_packet_count = 0
    while True:
        packets: list[bytes] = []
        for _ in range(4):
            packet = decoder.stdout.read(frame_bytes)
            if len(packet) != frame_bytes:
                if packet:
                    incomplete_packet_count += 1
                break
            packets.append(packet)
        if not packets:
            break
        if len(packets) != 4:
            incomplete_packet_count += len(packets)
            break
        pixels = np.concatenate(
            [np.frombuffer(packet, dtype=np.uint8).reshape(-1, 3) for packet in packets],
            axis=0,
        )
        composite = pixels.reshape(480, 640, 3)
        head_rgb = composite[:240, :320]
        encoder.stdin.write(head_rgb.tobytes())
        reconstructed_frames += 1
    encoder.stdin.close()
    encoder_return = encoder.wait()
    decoder_return = decoder.wait()
    if encoder_return != 0 or decoder_return != 0 or reconstructed_frames == 0:
        raise RuntimeError(
            "FastWAM reconstruction failed: "
            f"decoder={decoder_return}, encoder={encoder_return}, frames={reconstructed_frames}"
        )
    return {
        "method": "Reassembled each four decoded 320x240 packets into the original 640x480 RGB buffer, then retained the 320x240 head-camera crop.",
        "reconstructed_frames": reconstructed_frames,
        "incomplete_packet_count": incomplete_packet_count,
        "derived_video": file_record(output),
        "behavioral_effect": "none; pixel-layout repair only",
    }


def load_trajectory(episode: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    trajectory = json.loads(Path(episode["raw_trajectory"]["path"]).read_text())
    return (
        np.asarray([step["object_minus_target_x"] for step in trajectory]),
        np.asarray([step["object_minus_target_y"] for step in trajectory]),
    )


def status_text(episode: dict[str, Any]) -> str:
    if episode["requested_success"]:
        return "SUCCESS · released inside requested region"
    if episode["ever_entered_requested_region"]:
        return "FAIL · entered region but did not release there"
    if episode["verified_pickup_proxy"]:
        return "FAIL · picked up, never entered requested region"
    return "FAIL · no verified pickup"


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
        [xs[0]],
        [ys[0]],
        marker="D",
        s=28,
        facecolor=TEXT,
        edgecolor=CARD,
        linewidth=0.6,
        zorder=4,
    )
    marker = "o" if episode["requested_success"] else "X"
    axis.scatter(
        [xs[-1]],
        [ys[-1]],
        marker=marker,
        s=52,
        facecolor=color,
        edgecolor=CARD if marker == "o" else color,
        linewidth=1.0,
        zorder=5,
    )
    axis.set_xlim(-0.30, 0.30)
    axis.set_ylim(-0.24, 0.24)
    axis.set_xticks([-0.2, 0, 0.2])
    axis.set_xticklabels(["−.2", "TARGET", "+.2"], fontsize=7.5)
    axis.set_yticks([])
    axis.grid(axis="x", color=GRID, linewidth=0.5)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.set_title(
        f"Expected: {direction.upper()} band (dashed)  ·  actual endpoint x={float(xs[-1]):+.3f} m",
        loc="left",
        fontsize=8.2,
        color=TEXT,
        pad=3,
    )


def render_background(
    model_label: str,
    interface_note: str,
    left_episode: dict[str, Any],
    right_episode: dict[str, Any],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=PAPER)
    fig.text(
        0.035,
        0.945,
        f"{model_label}: same scene, one word changed",
        fontsize=25,
        weight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.965,
        0.945,
        interface_note,
        fontsize=10.5,
        color=MUTED,
        ha="right",
        va="top",
    )
    fig.text(
        0.035,
        0.895,
        "Matched RoboTwin intervention: anchor task, seed, objects, initial state, checkpoint, and sampler are held fixed.",
        fontsize=11,
        color=MUTED,
        ha="left",
        va="top",
    )

    panels = [
        (left_episode, "left", LEFT, 0.035),
        (right_episode, "right", RIGHT, 0.5225),
    ]
    for episode, direction, color, x0 in panels:
        fig.patches.append(
            Rectangle(
                (x0, 0.818),
                0.4425,
                0.058,
                transform=fig.transFigure,
                facecolor=color,
                alpha=0.09,
                edgecolor="none",
            )
        )
        fig.text(
            x0 + 0.012,
            0.852,
            "ASKED " + direction.upper(),
            fontsize=10,
            weight="bold",
            color=TEXT,
            ha="left",
            va="center",
        )
        prompt_lines = "\n".join(textwrap.wrap(episode["prompt"], 58))
        fig.text(
            x0 + 0.105,
            0.852,
            f'“{prompt_lines}”',
            fontsize=11.5,
            weight="bold",
            color=TEXT,
            ha="left",
            va="center",
        )
        status = status_text(episode)
        fig.text(
            x0 + 0.012,
            0.805,
            status,
            fontsize=10.5,
            weight="bold",
            color=TEXT,
            ha="left",
            va="top",
        )
        fig.text(
            x0 + 0.43,
            0.805,
            f"{episode['actions_executed']} actions",
            fontsize=9.5,
            color=MUTED,
            ha="right",
            va="top",
        )
        fig.patches.append(
            Rectangle(
                (x0 - 0.002, 0.195),
                0.448,
                0.592,
                transform=fig.transFigure,
                facecolor=CARD,
                edgecolor=GRID,
                linewidth=1.2,
                zorder=-1,
            )
        )
        path_axis = fig.add_axes((x0 + 0.012, 0.035, 0.418, 0.12))
        draw_path_axis(path_axis, episode, direction)

    fig.text(
        0.5,
        0.19,
        "Robot viewport above · target-relative object path below · a circle means the full distance, side, y-offset, and release check passed; a cross means it failed",
        fontsize=9.2,
        color=MUTED,
        ha="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_square_background(
    model_label: str,
    interface_note: str,
    left_episode: dict[str, Any],
    right_episode: dict[str, Any],
    output: Path,
) -> None:
    fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=PAPER)
    fig.text(
        0.04,
        0.965,
        f"{model_label}: did one word redirect the robot?",
        fontsize=22,
        weight="bold",
        color=TEXT,
        ha="left",
        va="top",
    )
    fig.text(
        0.96,
        0.925,
        interface_note,
        fontsize=9.5,
        color=MUTED,
        ha="right",
        va="top",
    )
    fig.text(
        0.04,
        0.925,
        "Same scene, seed, objects, initial state, checkpoint, and sampler.",
        fontsize=10.2,
        color=MUTED,
        ha="left",
        va="top",
    )

    panels = [
        (left_episode, "left", LEFT, 0.04),
        (right_episode, "right", RIGHT, 0.515),
    ]
    for episode, direction, color, x0 in panels:
        fig.patches.append(
            Rectangle(
                (x0, 0.815),
                0.445,
                0.075,
                transform=fig.transFigure,
                facecolor=color,
                alpha=0.09,
                edgecolor="none",
            )
        )
        fig.text(
            x0 + 0.012,
            0.872,
            "ASKED " + direction.upper(),
            fontsize=9.5,
            weight="bold",
            color=TEXT,
            ha="left",
            va="center",
        )
        prompt_lines = "\n".join(textwrap.wrap(episode["prompt"], 43))
        fig.text(
            x0 + 0.012,
            0.842,
            f'“{prompt_lines}”',
            fontsize=10,
            weight="bold",
            color=TEXT,
            ha="left",
            va="center",
        )
        fig.text(
            x0 + 0.012,
            0.8,
            status_text(episode),
            fontsize=9.2,
            weight="bold",
            color=TEXT,
            ha="left",
            va="top",
        )
        fig.text(
            x0 + 0.43,
            0.8,
            f"{episode['actions_executed']} actions",
            fontsize=8.5,
            color=MUTED,
            ha="right",
            va="top",
        )
        fig.patches.append(
            Rectangle(
                (x0 - 0.005, 0.445),
                0.458,
                0.35,
                transform=fig.transFigure,
                facecolor=CARD,
                edgecolor=GRID,
                linewidth=1.2,
                zorder=-1,
            )
        )
        path_axis = fig.add_axes((x0 + 0.018, 0.11, 0.405, 0.245))
        draw_path_axis(path_axis, episode, direction)
        path_axis.tick_params(axis="x", labelsize=8.5)

    fig.text(
        0.5,
        0.405,
        "Complete robot viewport above · target-relative object path below",
        fontsize=9.5,
        color=MUTED,
        ha="center",
        weight="bold",
    )
    fig.text(
        0.5,
        0.045,
        "Circle = full relation-and-release success · cross = failure · dashed route is illustrative and never enters the metric",
        fontsize=9.2,
        color=MUTED,
        ha="center",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def render_video(background: Path, left: Path, right: Path, output: Path) -> float:
    duration = max(ffprobe_duration(left), ffprobe_duration(right))
    filter_graph = (
        f"[1:v]fps=10,scale=700:525:flags=lanczos,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[lv];"
        f"[2:v]fps=10,scale=700:525:flags=lanczos,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[rv];"
        "[0:v][lv]overlay=60:195:shortest=0[tmp];"
        "[tmp][rv]overlay=840:195:shortest=1[outv]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "10",
            "-i",
            str(background),
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-t",
            f"{duration:.3f}",
            "-r",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return duration


def render_square_video(background: Path, left: Path, right: Path, output: Path) -> float:
    duration = max(ffprobe_duration(left), ffprobe_duration(right))
    filter_graph = (
        f"[1:v]fps=10,scale=540:405:flags=lanczos,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[lv];"
        f"[2:v]fps=10,scale=540:405:flags=lanczos,"
        f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f}[rv];"
        "[0:v][lv]overlay=48:250:shortest=0[tmp];"
        "[tmp][rv]overlay=618:250:shortest=1[outv]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "10",
            "-i",
            str(background),
            "-i",
            str(left),
            "-i",
            str(right),
            "-filter_complex",
            filter_graph,
            "-map",
            "[outv]",
            "-t",
            f"{duration:.3f}",
            "-r",
            "10",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return duration


def render_poster(video: Path, output: Path, duration: float) -> None:
    timestamp = min(max(duration * 0.72, 0.1), max(duration - 0.1, 0.1))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-ss",
            f"{timestamp:.3f}",
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def write_vtt(
    path: Path,
    duration: float,
    model_label: str,
    left_episode: dict[str, Any],
    right_episode: dict[str, Any],
) -> None:
    def timestamp(seconds: float) -> str:
        millis = int(round(seconds * 1000))
        hours, remainder = divmod(millis, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        secs, ms = divmod(remainder, 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"

    text = (
        "WEBVTT\n\n"
        f"00:00:00.000 --> {timestamp(duration)}\n"
        f"{model_label}. Same scene. Left panel prompt: {left_episode['prompt']} "
        f"Result: {status_text(left_episode)}. Right panel prompt: {right_episode['prompt']} "
        f"Result: {status_text(right_episode)}.\n"
    )
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/media/robotwin_wam_pairs"),
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir if args.output_dir.is_absolute() else workspace / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    media: list[dict[str, Any]] = []
    for result_stem, pair_id, model_label, interface_note in SELECTIONS:
        result_path = (
            workspace
            / "artifacts/vla_wam_shared_v2/pilot/results"
            / f"{result_stem}_direct_gate.json"
        )
        result = json.loads(result_path.read_text())
        pair = {
            episode["requested_relation"]: episode
            for episode in result["episodes"]
            if episode["pair_id"] == pair_id
        }
        if set(pair) != {"left", "right"}:
            raise RuntimeError(f"Incomplete pair {result_stem}/{pair_id}")
        left_episode, right_episode = pair["left"], pair["right"]
        if not left_episode["requested_success"] or right_episode["requested_success"]:
            raise RuntimeError(
                f"Selection {result_stem}/{pair_id} is not a LEFT-success/RIGHT-failure pair"
            )

        slug = f"{result_stem}_{pair_id}_left_success_right_failure"
        background = output_dir / f"{slug}_background.png"
        square_background = output_dir / f"{slug}_square_background.png"
        video = output_dir / f"{slug}.mp4"
        square_video = output_dir / f"{slug}_1200x1200.mp4"
        poster = output_dir / f"{slug}_poster.jpg"
        square_poster = output_dir / f"{slug}_1200x1200_poster.jpg"
        captions = output_dir / f"{slug}.vtt"
        render_background(
            model_label,
            interface_note,
            left_episode,
            right_episode,
            background,
        )
        render_square_background(
            model_label,
            interface_note,
            left_episode,
            right_episode,
            square_background,
        )
        left_video = Path(left_episode["executed_video"]["path"])
        right_video = Path(right_episode["executed_video"]["path"])
        publication_left_video = left_video
        publication_right_video = right_video
        source_video_correction = None
        if result_stem == "fastwam":
            publication_left_video = output_dir / f"{slug}_left_reconstructed_head.mp4"
            publication_right_video = output_dir / f"{slug}_right_reconstructed_head.mp4"
            source_video_correction = {
                "classification": "source_capture_pixel_layout_reconstruction",
                "cause": "The pilot declared a 320x240 raw ffmpeg stream while this RoboTwin fork wrote a 640x480 three-camera buffer, producing four encoded packets per intended frame.",
                "left": reconstruct_fastwam_head_video(
                    left_video, publication_left_video
                ),
                "right": reconstruct_fastwam_head_video(
                    right_video, publication_right_video
                ),
            }
        duration = render_video(
            background, publication_left_video, publication_right_video, video
        )
        square_duration = render_square_video(
            square_background,
            publication_left_video,
            publication_right_video,
            square_video,
        )
        if abs(square_duration - duration) > 1e-6:
            raise RuntimeError(f"Landscape/square duration mismatch for {slug}")
        render_poster(video, poster, duration)
        render_poster(square_video, square_poster, duration)
        write_vtt(captions, duration, model_label, left_episode, right_episode)
        alt_text = (
            f"Side-by-side RoboTwin rollouts from {model_label} in the same {pair_id} scene. "
            f"The LEFT prompt succeeds at x={left_episode['final_dx_m']:+.3f} m; "
            f"the mirrored RIGHT prompt fails at x={right_episode['final_dx_m']:+.3f} m. "
            "Target-relative trajectory strips show the requested lateral bands and actual object paths."
        )
        media.append(
            {
                "id": slug,
                "model_id": result["model_id"],
                "model_label": model_label,
                "interface_note": interface_note,
                "arena": "robotwin_place_a2b",
                "selection_rule": "First pair in compiled pair order containing a LEFT success and its matched RIGHT failure.",
                "pair_id": pair_id,
                "anchor_task": left_episode["task"],
                "environment_seed": left_episode["environment_seed"],
                "sampling_seed": left_episode["sampling_seed"],
                "duration_seconds": duration,
                "left": {
                    "prompt": left_episode["prompt"],
                    "success": left_episode["requested_success"],
                    "final_dx_m": left_episode["final_dx_m"],
                    "failure_stage": left_episode["failure_stage"],
                    "source_video": file_record(left_video),
                    "source_trajectory": left_episode["raw_trajectory"],
                },
                "right": {
                    "prompt": right_episode["prompt"],
                    "success": right_episode["requested_success"],
                    "final_dx_m": right_episode["final_dx_m"],
                    "failure_stage": right_episode["failure_stage"],
                    "source_video": file_record(right_video),
                    "source_trajectory": right_episode["raw_trajectory"],
                },
                "publication_video_inputs": {
                    "left": file_record(publication_left_video, workspace),
                    "right": file_record(publication_right_video, workspace),
                },
                "source_video_correction": source_video_correction,
                "video": file_record(video, workspace),
                "poster": file_record(poster, workspace),
                "square_video": file_record(square_video, workspace),
                "square_poster": file_record(square_poster, workspace),
                "captions": file_record(captions, workspace),
                "alt_text": alt_text,
            }
        )
        background.unlink()
        square_background.unlink()
        print(f"Rendered {video}")

    index = {
        "schema_version": "vla-wam-shared-v2-robotwin-paired-media-v1",
        "status": "complete",
        "selection_scope": "Three deterministic matched LEFT-success/RIGHT-failure examples from the 18-episode standardized direct-command WAM pilot.",
        "shorter_rollout_policy": "Hold the final frame of the shorter rollout until the longer matched rollout ends; never trim the longer failure.",
        "metric_boundary": "The dashed path is illustrative. Success is determined from recorded 3D state by the full relation-and-release checker, not from pixels or the drawn route.",
        "items": media,
    }
    index_path = output_dir / "media_index.json"
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index": str(index_path), "items": len(media)}, indent=2))


if __name__ == "__main__":
    main()
