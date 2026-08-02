#!/usr/bin/env python3
"""Render deterministically selected Cosmos semantic-future examples.

The selection rule is frozen in ``semantic_future_visualization_plan.json``.
This script never chooses a visually attractive example: for each semantic
quadrant it renders the first eligible chunk in the preregistered ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = (
    ("canonical", "cosmos_canonical", 6100),
    ("short_paraphrase", "cosmos_vague", 6100),
    ("declarative_goal", "cosmos_declarative", 7200),
    ("contrastive_goal", "cosmos_contrastive", 7200),
)
CAMERAS = ("left_camera", "right_camera")
BACKGROUND = "#07111f"
PANEL = "#101e31"
TEXT = "#f4f7fb"
MUTED = "#a9b7c9"
QUADRANT_STYLE = {
    "imagines_requested_executes_requested": (
        "Imagines requested · executes requested",
        "#37c99a",
        "aligned success",
    ),
    "imagines_requested_executes_not_requested": (
        "Imagines requested · executes something else",
        "#f4b860",
        "imagination–action mismatch",
    ),
    "does_not_imagine_requested_executes_requested": (
        "Does not imagine requested · executes requested",
        "#62a8ff",
        "policy succeeds despite its forecast",
    ),
    "neither_imagines_nor_executes_requested": (
        "Neither imagines nor executes requested",
        "#ef6f6c",
        "aligned failure",
    ),
    "uncertain_future": (
        "Future scorer abstains",
        "#8796aa",
        "insufficient or inconsistent visual evidence",
    ),
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _read_frames(path: Path, wanted_indices: tuple[int, ...]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    wanted = set(wanted_indices)
    frames: dict[int, np.ndarray] = {}
    frame_index = 0
    try:
        while wanted:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in wanted:
                frames[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                wanted.remove(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if wanted:
        raise RuntimeError(f"Missing frames {sorted(wanted)} in {path}")
    return frames


def _annotated_camera_row(
    frame: np.ndarray, frame_result: dict[str, Any]
) -> np.ndarray:
    bottom = frame[(frame.shape[0] * 2) // 3 :].copy()
    height, width = bottom.shape[:2]
    panel_width = width // 2
    cv2.line(bottom, (panel_width, 0), (panel_width, height), (220, 226, 234), 1)
    for camera_index, camera in enumerate(CAMERAS):
        localization = frame_result["localization"][camera]
        relation = frame_result["semantics"]["relations_by_camera"].get(camera)
        cv2.putText(
            bottom,
            f"camera {camera_index + 1}: {relation or 'uncertain'}",
            (camera_index * panel_width + 7, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (245, 248, 252),
            1,
            cv2.LINE_AA,
        )
        for object_name, color, letter in (
            ("cube", (0, 238, 238), "C"),
            ("bowl", (238, 78, 88), "B"),
        ):
            point = localization.get(f"{object_name}_center")
            if point is None:
                continue
            x = int(round(camera_index * panel_width + point[0] / 1000.0 * panel_width))
            y = int(round(point[1] / 1000.0 * height))
            x = min(max(x, camera_index * panel_width), (camera_index + 1) * panel_width - 1)
            y = min(max(y, 0), height - 1)
            cv2.circle(bottom, (x, y), 6, color, 2, cv2.LINE_AA)
            cv2.putText(
                bottom,
                letter,
                (min(x + 7, width - 12), max(y - 7, 11)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )
    return bottom


def _collect_rows(study_root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    wording_order = plan["selection_rule"]["wording_order"]
    if wording_order != [condition[0] for condition in CONDITIONS]:
        raise RuntimeError("Visualization plan wording order disagrees with renderer")
    direction_order = plan["selection_rule"]["direction_order"]
    rows: list[dict[str, Any]] = []
    for wording, output_name, seed_base in CONDITIONS:
        output_dir = study_root / "semantic_confirmation" / output_name
        summary_path = output_dir / "semantic_quadrants_summary.json"
        summary = _load(summary_path)
        for source_row in summary["rows"]:
            row = dict(source_row)
            row["wording"] = wording
            row["semantic_output_dir"] = str(output_dir.resolve())
            row["episode_seed"] = seed_base + int(row["episode_index"])
            expected_sampling_seed = (
                row["episode_seed"] * 1000 + int(row["replan_index"])
            )
            if int(row["sampling_seed"]) != expected_sampling_seed:
                raise RuntimeError(
                    "Semantic row sampling seed disagrees with the registered schedule: "
                    f"{row}"
                )
            rows.append(row)
    wording_rank = {value: index for index, value in enumerate(wording_order)}
    direction_rank = {value: index for index, value in enumerate(direction_order)}
    rows.sort(
        key=lambda row: (
            wording_rank[row["wording"]],
            direction_rank[row["requested_relation"]],
            int(row["episode_index"]),
            int(row["replan_index"]),
        )
    )
    return rows


def _cache_path(row: dict[str, Any]) -> Path:
    task_dir = Path(row["task_dir"])
    task_key = f"{task_dir.parent.name}__{task_dir.name}"
    return (
        Path(row["semantic_output_dir"])
        / "localization_cache"
        / task_key
        / (
            f"episode_{int(row['episode_index']):03d}_"
            f"chunk_{int(row['replan_index']):03d}.json"
        )
    )


def _materialize_selection(
    selected_row: dict[str, Any] | None, frame_indices: tuple[int, ...]
) -> dict[str, Any] | None:
    if selected_row is None:
        return None
    future_path = Path(selected_row["chunk_dir"]) / "future.mp4"
    cache_path = _cache_path(selected_row)
    cache = _load(cache_path)
    frame_results = {int(item["frame_index"]): item for item in cache["frames"]}
    if set(frame_results) != set(frame_indices):
        raise RuntimeError(f"Localization cache frame mismatch in {cache_path}")
    frames = _read_frames(future_path, frame_indices)
    annotated = {
        frame_index: _annotated_camera_row(frames[frame_index], frame_results[frame_index])
        for frame_index in frame_indices
    }
    published = {
        key: selected_row[key]
        for key in (
            "wording",
            "requested_relation",
            "episode_index",
            "episode_seed",
            "replan_index",
            "sampling_seed",
            "imagined_requested",
            "reliable_future_frames",
            "requested_future_frames",
            "executed_relation",
            "executed_requested",
            "quadrant",
            "task_dir",
            "chunk_dir",
        )
    }
    published.update(
        {
            "future_video": str(future_path.resolve()),
            "future_video_sha256": _sha256(future_path),
            "localization_cache": str(cache_path.resolve()),
            "localization_cache_sha256": _sha256(cache_path),
        }
    )
    return {"record": published, "frames": annotated}


def _format_wording(value: str) -> str:
    return value.replace("_", " ").replace("paraphrase", "paraphrase")


def _render_blog(
    selections: dict[str, dict[str, Any] | None],
    categories: list[str],
    frame_indices: tuple[int, ...],
    output: Path,
) -> None:
    figure = plt.figure(figsize=(18, 12.5), facecolor=BACKGROUND)
    grid = figure.add_gridspec(
        len(categories),
        len(frame_indices) + 1,
        width_ratios=[1.35, 1, 1, 1, 1],
        hspace=0.34,
        wspace=0.06,
        left=0.025,
        right=0.985,
        bottom=0.065,
        top=0.885,
    )
    figure.text(
        0.025,
        0.956,
        "What the WAM imagined—and what its actions actually did",
        color=TEXT,
        fontsize=26,
        fontweight="bold",
    )
    figure.text(
        0.025,
        0.922,
        "One deterministic, frozen-order example per semantic category · frames 8, 16, 24, 32 · cyan = cube · red = bowl",
        color=MUTED,
        fontsize=12,
    )
    for row_index, category in enumerate(categories):
        label, color, interpretation = QUADRANT_STYLE[category]
        selection = selections[category]
        info_axis = figure.add_subplot(grid[row_index, 0])
        info_axis.set_facecolor(PANEL)
        info_axis.set_xticks([])
        info_axis.set_yticks([])
        for spine in info_axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.5)
        info_axis.text(
            0.055,
            0.88,
            "\n".join(textwrap.wrap(label, 29)),
            transform=info_axis.transAxes,
            color=color,
            fontsize=11.5,
            fontweight="bold",
            va="top",
        )
        info_axis.text(
            0.055,
            0.59,
            interpretation,
            transform=info_axis.transAxes,
            color=MUTED,
            fontsize=9.2,
            va="top",
        )
        if selection is None:
            metadata = "No eligible confirmation chunk observed."
        else:
            record = selection["record"]
            metadata = (
                f"request: {record['requested_relation'].upper()}\n"
                f"imagined: {'YES' if record['imagined_requested'] is True else 'NO' if record['imagined_requested'] is False else 'UNCERTAIN'}\n"
                f"executed: {record['executed_relation'].upper()}\n"
                f"{_format_wording(record['wording'])} · seed {record['episode_seed']} · chunk {record['replan_index']}"
            )
        info_axis.text(
            0.055,
            0.39,
            metadata,
            transform=info_axis.transAxes,
            color=TEXT,
            fontsize=8.8,
            va="top",
            linespacing=1.2,
        )
        for column_index, frame_index in enumerate(frame_indices, start=1):
            axis = figure.add_subplot(grid[row_index, column_index])
            axis.set_facecolor(PANEL)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if selection is None:
                axis.text(
                    0.5,
                    0.5,
                    "not observed",
                    color=MUTED,
                    fontsize=11,
                    ha="center",
                    va="center",
                )
            else:
                axis.imshow(selection["frames"][frame_index])
            if row_index == 0:
                axis.set_title(f"future frame {frame_index}", color=TEXT, fontsize=11, pad=6)
    figure.text(
        0.025,
        0.022,
        "Retrospective examples selected by a plan frozen before confirmation semantic labels. They explain aggregate labels; they are not independent trials or rate estimates.",
        color=MUTED,
        fontsize=10,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


def _render_social(
    selections: dict[str, dict[str, Any] | None],
    frame_index: int,
    output: Path,
    width_px: int,
    height_px: int,
) -> None:
    categories = list(QUADRANT_STYLE)[:4]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(width_px / 100, height_px / 100),
        dpi=100,
        facecolor=BACKGROUND,
    )
    figure.subplots_adjust(left=0.035, right=0.985, bottom=0.09, top=0.82, hspace=0.36, wspace=0.08)
    figure.text(
        0.035,
        0.94,
        "A WAM can imagine and act in four different ways",
        color=TEXT,
        fontsize=24 if width_px > height_px else 20,
        fontweight="bold",
    )
    figure.text(
        0.035,
        0.885,
        "Prompt-blind future semantics × executed state at the matched action horizon",
        color=MUTED,
        fontsize=12 if width_px > height_px else 10.5,
    )
    for axis, category in zip(axes.flat, categories, strict=True):
        label, color, _ = QUADRANT_STYLE[category]
        selection = selections[category]
        axis.set_facecolor(PANEL)
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2.0)
        if selection is None:
            axis.text(0.5, 0.5, "No observed example", ha="center", va="center", color=MUTED)
            subtitle = "not observed in this checkpoint/grid"
        else:
            record = selection["record"]
            axis.imshow(selection["frames"][frame_index])
            subtitle = (
                f"ask {record['requested_relation'].upper()} · "
                f"future {'yes' if record['imagined_requested'] else 'no'} · "
                f"execution {record['executed_relation'].upper()}"
            )
        axis.set_title(label, loc="left", color=color, fontsize=11.5, fontweight="bold", pad=8)
        axis.text(
            0.0,
            -0.13,
            subtitle,
            transform=axis.transAxes,
            color=TEXT,
            fontsize=9.5,
            va="top",
        )
    figure.text(
        0.035,
        0.025,
        "Frozen first-in-order examples · cyan cube · red bowl · one public Cosmos3 Edge DROID checkpoint · Ali Adeeb Abbas",
        color=MUTED,
        fontsize=9.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    image = cv2.imread(str(output))
    if image is None or image.shape[:2] != (height_px, width_px):
        raise RuntimeError(f"Social export has wrong dimensions: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--study-root", type=Path, default=Path("artifacts/vla_wam_shared_v1")
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v1/semantic_future_visualization_plan.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v1/semantic_future_visualization"),
    )
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    study_root = (workspace / args.study_root).resolve() if not args.study_root.is_absolute() else args.study_root.resolve()
    plan_path = (workspace / args.plan).resolve() if not args.plan.is_absolute() else args.plan.resolve()
    output = (workspace / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    plan = _load(plan_path)
    if plan.get("status") != "fixed_before_any_confirmation_semantic_label":
        raise RuntimeError("Semantic visualization plan is not frozen")
    categories = plan["category_order"]
    if categories != list(QUADRANT_STYLE):
        raise RuntimeError("Semantic visualization category order disagrees with renderer")
    frame_indices = tuple(int(value) for value in plan["rendering"]["future_frame_indices"])
    rows = _collect_rows(study_root, plan)
    selections: dict[str, dict[str, Any] | None] = {}
    published_selection: dict[str, Any] = {}
    for category in categories:
        selected_row = next((row for row in rows if row["quadrant"] == category), None)
        materialized = _materialize_selection(selected_row, frame_indices)
        selections[category] = materialized
        published_selection[category] = materialized["record"] if materialized else None

    blog_path = output / "blog/selected_semantic_future_examples.png"
    landscape_path = output / "social/wam_semantic_quadrants_1600x900.png"
    square_path = output / "social/wam_semantic_quadrants_1200x1200.png"
    _render_blog(selections, categories, frame_indices, blog_path)
    _render_social(selections, frame_indices[-1], landscape_path, 1600, 900)
    _render_social(selections, frame_indices[-1], square_path, 1200, 1200)
    selection_path = output / "selection.json"
    _dump(
        selection_path,
        {
            "schema_version": 1,
            "selection_plan": str(plan_path),
            "selection_plan_sha256": _sha256(plan_path),
            "eligible_chunk_count": len(rows),
            "categories": published_selection,
        },
    )
    _dump(
        output / "summary.json",
        {
            "schema_version": 1,
            "status": "complete",
            "selection_plan_sha256": _sha256(plan_path),
            "eligible_chunk_count": len(rows),
            "observed_category_count": sum(value is not None for value in published_selection.values()),
            "selection_sha256": _sha256(selection_path),
            "artifacts": {
                "blog": _relative_or_absolute(blog_path, workspace),
                "landscape_social": _relative_or_absolute(landscape_path, workspace),
                "square_social": _relative_or_absolute(square_path, workspace),
            },
        },
    )
    print(
        f"rendered {len(rows)} eligible chunks across "
        f"{sum(value is not None for value in published_selection.values())}/5 observed categories -> {output}"
    )


if __name__ == "__main__":
    main()
