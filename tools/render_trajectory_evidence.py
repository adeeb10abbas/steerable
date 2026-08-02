#!/usr/bin/env python3
"""Render auditable path evidence for the oracle-free VLA/WAM study.

The renderer deliberately separates the scored goal from an illustrative path:
the matched RoboLab task scores a released cube inside the requested 45-degree
robot-frame cone.  There is no single privileged trajectory to that region.
Every valid episode is rendered and indexed so that the blog can show successes
and failures without hiding inconvenient trials.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon


TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
MODEL_LABELS = {
    "pi05_droid_vla": "π0.5 DROID · VLA",
    "cosmos3_edge_droid_wam": "Cosmos3 Edge DROID · WAM",
}
WORDING_LABELS = {
    "canonical": "Canonical task",
    "short_paraphrase": "Short paraphrase",
    "declarative_goal": "Declarative goal",
    "contrastive_goal": "Contrastive goal",
}

# Editorial palette chosen for legibility in light-mode documents and social
# crops. Meaning is always duplicated with text and marker shape.
INK = "#142B3A"
MUTED = "#667785"
GRID = "#D8E0E5"
PAPER = "#FBFCFD"
TARGET = "#1F9D78"
TARGET_PALE = "#DDF3EA"
FORBIDDEN = "#D95D5D"
FORBIDDEN_PALE = "#F8E3E3"
SUCCESS = "#17845F"
FAILURE = "#C84B4B"
LEFT_COLOR = "#1976A3"
RIGHT_COLOR = "#D5792A"
EE_COLOR = "#93A1AB"
PATH_LIGHT = "#A9C7D4"
PATH_DARK = "#153F57"
BOWL_FACE = "#E7B96D"
OUTCOME_STAGE_LABELS = {
    "success": "success",
    "ended_in_goal_without_terminal_success": "ended in goal · terminal failure",
    "entered_goal_then_lost_it": "entered goal · lost it",
    "picked_never_entered_goal": "picked · never entered goal",
    "interaction_without_verified_pickup": "interaction · no verified pickup",
    "no_cube_interaction": "no cube interaction",
}
OUTCOME_STAGE_COLORS = {
    "no_cube_interaction": "#C9D1D7",
    "interaction_without_verified_pickup": "#8EA3B2",
    "picked_never_entered_goal": "#E5A443",
    "entered_goal_then_lost_it": "#D96D5F",
    "ended_in_goal_without_terminal_success": "#9E5A8A",
    "success": SUCCESS,
}
OUTCOME_STAGE_ORDER = tuple(OUTCOME_STAGE_COLORS)

LATERAL_LIMIT_M = 0.55
FORWARD_MIN_M = -0.40
FORWARD_MAX_M = 0.35
DIRECT_GOAL_M = 0.30
RELATION_COSINE = math.cos(math.radians(45.0))


@dataclass
class Episode:
    condition_id: str
    model_id: str
    model_class: str
    wording: str
    analysis_tier: str
    direction: str
    run: int
    seed: int
    instruction: str
    success: bool
    steps: int
    dt: float
    cube_robot: np.ndarray
    ee_robot: np.ndarray
    requested_mask: np.ndarray
    opposite_mask: np.ndarray
    pickup_step: int | None
    release_step: int | None
    first_relation_step: int | None
    interaction_step: int | None
    hdf5_path: Path
    log_path: Path
    result_path: Path

    @property
    def cube_plot(self) -> np.ndarray:
        # Plot page-left as robot-left. In the robot frame +y is left, so the
        # page-horizontal coordinate is -y.
        return np.column_stack((-self.cube_robot[:, 1], self.cube_robot[:, 0]))

    @property
    def ee_plot(self) -> np.ndarray:
        return np.column_stack((-self.ee_robot[:, 1], self.ee_robot[:, 0]))

    @property
    def endpoint_class(self) -> str:
        if bool(self.requested_mask[-1]):
            return "requested_region"
        if bool(self.opposite_mask[-1]):
            return "opposite_region"
        return "neutral_region"

    @property
    def requested_signed_offset_m(self) -> float:
        delta_y = float(self.cube_robot[-1, 1])
        return delta_y if self.direction == "left" else -delta_y

    @property
    def cube_path_length_xy_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.cube_robot[:, :2], axis=0), axis=1).sum())

    @property
    def outcome_stage(self) -> str:
        if self.success:
            return "success"
        if bool(self.requested_mask[-1]):
            return "ended_in_goal_without_terminal_success"
        if self.first_relation_step is not None:
            return "entered_goal_then_lost_it"
        if self.pickup_step is not None:
            return "picked_never_entered_goal"
        if self.interaction_step is not None:
            return "interaction_without_verified_pickup"
        return "no_cube_interaction"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _quaternion_wxyz_matrix(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise ValueError("Robot root pose contains a zero quaternion")
    w, x, y, z = (quaternion / norm).T
    matrix = np.empty((len(quaternion), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def _relative_robot_frame(
    points_world: np.ndarray, bowl_pose: np.ndarray, robot_pose: np.ndarray
) -> np.ndarray:
    if not (len(points_world) == len(bowl_pose) == len(robot_pose)):
        raise ValueError("Pose trajectories have different lengths")
    delta_world = points_world[:, :3] - bowl_pose[:, :3]
    robot_local_to_world = _quaternion_wxyz_matrix(robot_pose[:, 3:7])
    # Row-vector multiplication by R is equivalent to R^T @ delta and matches
    # RoboLab's world-to-robot directional predicate.
    return np.einsum("tij,ti->tj", robot_local_to_world, delta_world)


def _relation_mask(delta_robot: np.ndarray, direction: str) -> np.ndarray:
    horizontal_norm = np.linalg.norm(delta_robot[:, :2], axis=1)
    sign = 1.0 if direction == "left" else -1.0
    cosine = np.divide(
        sign * delta_robot[:, 1],
        horizontal_norm,
        out=np.zeros_like(horizontal_norm),
        where=horizontal_norm > 1e-8,
    )
    return cosine >= RELATION_COSINE


def _first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def _event_mentions_cube(event: dict[str, Any]) -> bool:
    return "rubiks_cube" in str(event.get("info", "")).lower()


def _event_steps(events: list[dict[str, Any]]) -> tuple[int | None, int | None, int | None]:
    pickups = [
        int(event["step"])
        for event in events
        if event.get("name") == "OBJECT_GRABBED_SUCCESS" and _event_mentions_cube(event)
    ]
    interactions = [
        int(event["step"])
        for event in events
        if event.get("name")
        in {"OBJECT_GRABBED_SUCCESS", "GRIPPER_HIT_OBJECT", "OBJECT_BUMPED"}
        and _event_mentions_cube(event)
    ]
    pickup = pickups[0] if pickups else None
    releases: list[int] = []
    if pickup is not None:
        for event in events:
            step = int(event.get("step", -1))
            if step <= pickup or not _event_mentions_cube(event):
                continue
            info = str(event.get("info", "")).lower()
            if event.get("name") == "OBJECT_DROPPED_SUCCESS" or "detached" in info:
                releases.append(step)
    return (
        pickup,
        # The task may re-grasp or bump the cube after its first release. The
        # last recorded detachment best identifies where the final placement
        # attempt ended; the first verified pickup remains the progression cue.
        releases[-1] if releases else None,
        interactions[0] if interactions else None,
    )


def _result_index(root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    path = root / "episode_results.jsonl"
    if not path.exists():
        return {}
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["env_name"]), int(row["run"]))
        if key in result:
            raise RuntimeError(f"Duplicate episode result {key} in {path}")
        result[key] = row
    return result


def _load_episode(
    condition: dict[str, Any], direction: str, run: int, seed: int
) -> Episode:
    root = Path(condition["output_root"])
    task = TASKS[direction]
    task_dir = root / task
    hdf5_path = task_dir / f"run_{run}.hdf5"
    log_path = task_dir / f"log_{run}_env0.json"
    result_path = root / "episode_results.jsonl"
    for path in (hdf5_path, log_path, result_path):
        if not path.exists():
            raise FileNotFoundError(path)

    result = _result_index(root).get((task, run))
    if result is None:
        raise RuntimeError(f"Missing completed result for {condition['id']} {direction} run {run}")
    log = _load_json(log_path)
    if bool(result["success"]) != bool(log["success"]):
        raise RuntimeError(f"Success disagreement for {condition['id']} {direction} run {run}")
    expected_instruction = condition["expected_instruction"][direction]
    if result.get("instruction") != expected_instruction:
        raise RuntimeError(
            f"Instruction mismatch for {condition['id']} {direction} run {run}: "
            f"{result.get('instruction')!r}"
        )

    with h5py.File(hdf5_path, "r") as handle:
        demo = handle["data/demo_0"]
        cube_pose = np.asarray(
            demo["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64
        )
        bowl_pose = np.asarray(
            demo["states/rigid_object/bowl/root_pose"], dtype=np.float64
        )
        robot_pose = np.asarray(
            demo["states/articulation/robot/root_pose"], dtype=np.float64
        )
        ee_position = np.asarray(demo["ee_pose/position"], dtype=np.float64)
    steps = int(log["final_step"])
    if not (len(cube_pose) == len(bowl_pose) == len(robot_pose) == len(ee_position) == steps):
        raise RuntimeError(
            f"Trajectory length mismatch for {condition['id']} {direction} run {run}: "
            f"HDF={len(cube_pose)}, log={steps}"
        )
    cube_robot = _relative_robot_frame(cube_pose, bowl_pose, robot_pose)
    ee_robot = _relative_robot_frame(ee_position, bowl_pose, robot_pose)
    pickup, release, interaction = _event_steps(log.get("events", []))
    requested = _relation_mask(cube_robot, direction)
    opposite = _relation_mask(cube_robot, "right" if direction == "left" else "left")
    return Episode(
        condition_id=str(condition["id"]),
        model_id=str(condition["model_id"]),
        model_class=str(condition["model_class"]),
        wording=str(condition["wording"]),
        analysis_tier=str(condition["analysis_tier"]),
        direction=direction,
        run=run,
        seed=seed,
        instruction=expected_instruction,
        success=bool(log["success"]),
        steps=steps,
        dt=float(log["dt"]),
        cube_robot=cube_robot,
        ee_robot=ee_robot,
        requested_mask=requested,
        opposite_mask=opposite,
        pickup_step=pickup,
        release_step=release,
        first_relation_step=_first_true(requested),
        interaction_step=interaction,
        hdf5_path=hdf5_path.resolve(),
        log_path=log_path.resolve(),
        result_path=result_path.resolve(),
    )


def _configure_plotting() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.facecolor": PAPER,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
        }
    )


def _goal_polygons() -> tuple[np.ndarray, np.ndarray]:
    left = np.asarray(
        [[0.0, 0.0], [-LATERAL_LIMIT_M, FORWARD_MIN_M], [-LATERAL_LIMIT_M, FORWARD_MAX_M]]
    )
    right = np.asarray(
        [[0.0, 0.0], [LATERAL_LIMIT_M, FORWARD_MIN_M], [LATERAL_LIMIT_M, FORWARD_MAX_M]]
    )
    return left, right


def _draw_goal_field(ax: plt.Axes, direction: str, *, labels: bool = True) -> None:
    left, right = _goal_polygons()
    target_polygon, opposite_polygon = (left, right) if direction == "left" else (right, left)
    ax.add_patch(Polygon(target_polygon, closed=True, facecolor=TARGET_PALE, edgecolor="none", zorder=0))
    ax.add_patch(
        Polygon(opposite_polygon, closed=True, facecolor=FORBIDDEN_PALE, edgecolor="none", zorder=0)
    )
    for polygon, color in ((target_polygon, TARGET), (opposite_polygon, FORBIDDEN)):
        ax.plot(
            [polygon[1, 0], 0.0, polygon[2, 0]],
            [polygon[1, 1], 0.0, polygon[2, 1]],
            color=color,
            linewidth=0.8,
            alpha=0.65,
            zorder=1,
        )
    ax.scatter([0], [0], s=115, marker="o", color=BOWL_FACE, edgecolor=INK, linewidth=0.8, zorder=8)
    if labels:
        target_x = -0.39 if direction == "left" else 0.39
        opposite_x = -target_x
        ax.text(
            target_x,
            FORWARD_MAX_M - 0.025,
            "requested\ngoal region",
            ha="center",
            va="top",
            color=TARGET,
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            opposite_x,
            FORWARD_MIN_M + 0.025,
            "opposite\nregion",
            ha="center",
            va="bottom",
            color=FORBIDDEN,
            fontsize=8,
        )
        ax.text(0.0, 0.028, "bowl", ha="center", va="bottom", fontsize=7.5, color=INK)
    ax.axhline(0, color=GRID, linewidth=0.7, zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.7, zorder=0)
    ax.set_xlim(-LATERAL_LIMIT_M, LATERAL_LIMIT_M)
    ax.set_ylim(FORWARD_MIN_M, FORWARD_MAX_M)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)


def _line_collection(points: np.ndarray, *, linewidth: float = 2.6, alpha: float = 1.0) -> LineCollection:
    if len(points) < 2:
        points = np.vstack([points, points])
    segments = np.stack((points[:-1], points[1:]), axis=1)
    cmap = LinearSegmentedColormap.from_list("actual_path", [PATH_LIGHT, PATH_DARK])
    collection = LineCollection(segments, cmap=cmap, linewidth=linewidth, alpha=alpha, zorder=5)
    collection.set_array(np.linspace(0.0, 1.0, len(segments)))
    return collection


def _bounded_step(step: int | None, length: int) -> int | None:
    if step is None:
        return None
    return min(max(int(step), 0), length - 1)


def _render_episode_axis(
    ax: plt.Axes,
    episode: Episode,
    *,
    compact: bool = False,
    show_ee: bool = True,
    show_prompt: bool = True,
) -> None:
    _draw_goal_field(ax, episode.direction, labels=not compact)
    cube = episode.cube_plot
    ee = episode.ee_plot
    if show_ee:
        ax.plot(
            ee[:, 0],
            ee[:, 1],
            color=EE_COLOR,
            linewidth=0.7 if compact else 0.9,
            alpha=0.45,
            zorder=2,
            linestyle=(0, (2, 2)),
        )

    anchor = np.asarray([-DIRECT_GOAL_M if episode.direction == "left" else DIRECT_GOAL_M, 0.0])
    ax.annotate(
        "",
        xy=anchor,
        xytext=cube[0],
        arrowprops={"arrowstyle": "-|>", "color": TARGET, "lw": 1.25, "linestyle": (0, (4, 3))},
        zorder=3,
    )
    if not compact:
        route_midpoint = 0.20 * cube[0] + 0.80 * anchor
        ax.text(
            route_midpoint[0],
            route_midpoint[1] + 0.030,
            "illustrative direct route",
            color=TARGET,
            fontsize=7.5,
            ha="center",
            va="bottom",
            rotation=0,
            zorder=4,
        )

    ax.add_collection(_line_collection(cube, linewidth=2.2 if compact else 3.0))
    status_color = SUCCESS if episode.success else FAILURE
    ax.scatter(
        [cube[0, 0]],
        [cube[0, 1]],
        s=55 if compact else 72,
        marker="o",
        facecolor="white",
        edgecolor=INK,
        linewidth=1.1,
        zorder=9,
    )
    ax.scatter(
        [cube[-1, 0]],
        [cube[-1, 1]],
        s=120 if compact else 165,
        marker="*",
        color=status_color,
        edgecolor="white",
        linewidth=0.8,
        zorder=11,
    )

    marker_specs = [
        (episode.pickup_step, "D", "#6F54A3", "pickup", (7, -15)),
        (episode.release_step, "s", "#DA8C24", "release", (7, 13)),
        (episode.first_relation_step, "X", TARGET, "first enters goal", (7, -18)),
    ]
    occupied: list[np.ndarray] = []
    for step, marker, color, label, label_offset in marker_specs:
        bounded = _bounded_step(step, len(cube))
        if bounded is None:
            continue
        point = cube[bounded]
        if any(np.linalg.norm(point - other) < 0.012 for other in occupied):
            continue
        occupied.append(point)
        if np.linalg.norm(point - cube[-1]) < 0.025:
            if not compact:
                ax.annotate(
                    f"{label} · {bounded}",
                    xy=point,
                    xytext=label_offset,
                    textcoords="offset points",
                    fontsize=7.2,
                    color=INK,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.7},
                    zorder=12,
                )
            continue
        ax.scatter(
            [point[0]],
            [point[1]],
            s=42 if compact else 64,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.65,
            zorder=10,
        )
        if not compact:
            ax.annotate(
                f"{label} · {bounded}",
                xy=point,
                xytext=label_offset,
                textcoords="offset points",
                fontsize=7.2,
                color=INK,
                zorder=12,
            )

    outcome = "SUCCESS" if episode.success else "FAILURE"
    endpoint = episode.endpoint_class.replace("_", " ")
    if compact:
        ax.set_title(
            f"{MODEL_LABELS[episode.model_id]} · {WORDING_LABELS[episode.wording]}",
            loc="left",
            color=INK,
            fontsize=9.5,
            fontweight="bold",
            pad=7,
        )
        ax.text(
            0.99,
            1.018,
            outcome,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color=status_color,
            fontsize=9.5,
            fontweight="bold",
        )
    else:
        ax.text(
            0.0,
            1.205,
            f"{outcome} · {MODEL_LABELS[episode.model_id]}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=status_color,
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )
        ax.text(
            0.0,
            1.145,
            f"{WORDING_LABELS[episode.wording]} · {episode.direction.upper()} · seed {episode.seed}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
            clip_on=False,
        )
    if show_prompt and not compact:
        prompt = "\n".join(textwrap.wrap(f'“{episode.instruction}”', width=66))
        ax.text(
            0.0,
            1.085,
            prompt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.3,
            color=MUTED,
            clip_on=False,
        )
    if compact:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
    else:
        ax.set_xlabel("← robot left · lateral displacement from bowl (m) · robot right →")
        ax.set_ylabel("Robot-forward displacement from bowl (m)")
        picked = f"pickup {episode.pickup_step}" if episode.pickup_step is not None else "no verified pickup"
        released = f"release {episode.release_step}" if episode.release_step is not None else "no verified release"
        ax.text(
            0.0,
            -0.17,
            f"seed {episode.seed}  ·  {episode.steps} steps  ·  {picked}  ·  {released}  ·  final: {endpoint}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            color=MUTED,
        )
        path_legend = [
            Line2D([0], [0], color=PATH_DARK, linewidth=2.5, label="cube path"),
            Line2D([0], [0], color=EE_COLOR, linewidth=1, linestyle=(0, (2, 2)), label="gripper path"),
        ]
        ax.legend(
            handles=path_legend,
            loc="upper center",
            bbox_to_anchor=(0.52, 0.995),
            ncol=2,
            frameon=False,
            fontsize=7.5,
            handlelength=2.2,
            columnspacing=1.2,
        )
    for spine in ax.spines.values():
        spine.set_color(status_color)
        spine.set_linewidth(1.4 if compact else 1.8)


def _render_episode(episode: Episode, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.35, 5.75))
    _render_episode_axis(ax, episode)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.22)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _sort_key(episode: Episode) -> tuple[Any, ...]:
    model_order = {"pi05_droid_vla": 0, "cosmos3_edge_droid_wam": 1}
    wording_order = {
        "canonical": 0,
        "short_paraphrase": 1,
        "declarative_goal": 2,
        "contrastive_goal": 3,
    }
    return (
        model_order.get(episode.model_id, 99),
        wording_order.get(episode.wording, 99),
        0 if episode.direction == "left" else 1,
        episode.seed,
    )


def _episode_slug(episode: Episode) -> str:
    outcome = "success" if episode.success else "failure"
    return f"{episode.model_id}__{episode.wording}__{episode.direction}__seed-{episode.seed}__{outcome}.png"


def _trajectory_index_row(episode: Episode, figure: Path, repo_root: Path) -> dict[str, Any]:
    final = episode.cube_robot[-1]
    downsample_indices = np.unique(np.linspace(0, len(episode.cube_plot) - 1, 48, dtype=int))
    downsampled = np.round(episode.cube_plot[downsample_indices], 5).tolist()
    return {
        "condition_id": episode.condition_id,
        "model_id": episode.model_id,
        "model_class": episode.model_class,
        "wording": episode.wording,
        "analysis_tier": episode.analysis_tier,
        "direction": episode.direction,
        "run": episode.run,
        "episode_seed": episode.seed,
        "instruction": episode.instruction,
        "binary_success": episode.success,
        "outcome_stage": episode.outcome_stage,
        "outcome_stage_label": OUTCOME_STAGE_LABELS[episode.outcome_stage],
        "endpoint_class": episode.endpoint_class,
        "final_requested_relation": bool(episode.requested_mask[-1]),
        "final_opposite_relation": bool(episode.opposite_mask[-1]),
        "verified_pickup": episode.pickup_step is not None,
        "pickup_step": episode.pickup_step,
        "release_step": episode.release_step,
        "first_interaction_step": episode.interaction_step,
        "first_requested_relation_step": episode.first_relation_step,
        "steps": episode.steps,
        "duration_s": episode.steps * episode.dt,
        "final_cube_minus_bowl_robot_x_m": float(final[0]),
        "final_cube_minus_bowl_robot_y_m": float(final[1]),
        "final_cube_minus_bowl_robot_z_m": float(final[2]),
        "requested_signed_final_offset_m": episode.requested_signed_offset_m,
        "cube_xy_path_length_m": episode.cube_path_length_xy_m,
        "downsampled_plot_path_lateral_forward_m": downsampled,
        "plot_coordinate_convention": "page_x=-robot_y so robot-left appears on page-left; page_y=robot_x",
        "hdf5_path": str(episode.hdf5_path),
        "log_path": str(episode.log_path),
        "episode_result_path": str(episode.result_path),
        "figure_path": str(figure.resolve().relative_to(repo_root.resolve())),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {"downsampled_plot_path_lateral_forward_m", "plot_coordinate_convention"}
    fieldnames = [key for key in rows[0] if key not in excluded]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _render_endpoint_atlas(episodes: list[Episode], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.4, 6.6), sharex=True, sharey=True)
    models = ["pi05_droid_vla", "cosmos3_edge_droid_wam"]
    wordings = ["canonical", "short_paraphrase", "declarative_goal", "contrastive_goal"]
    for row_index, model_id in enumerate(models):
        for column_index, wording in enumerate(wordings):
            ax = axes[row_index, column_index]
            # Both goal cones are legitimate because the panel contains left- and
            # right-request trials. They are softly separated by request color.
            left, right = _goal_polygons()
            ax.add_patch(Polygon(left, closed=True, facecolor="#E4F1F6", edgecolor="none"))
            ax.add_patch(Polygon(right, closed=True, facecolor="#F8EBDD", edgecolor="none"))
            ax.scatter([0], [0], s=55, color=BOWL_FACE, edgecolor=INK, linewidth=0.6, zorder=6)
            group = [e for e in episodes if e.model_id == model_id and e.wording == wording]
            for direction, color, marker in (("left", LEFT_COLOR, "<"), ("right", RIGHT_COLOR, ">")):
                selected = [e for e in group if e.direction == direction]
                for episode in selected:
                    path = episode.cube_plot
                    ax.plot(path[:, 0], path[:, 1], color=color, alpha=0.18, linewidth=0.65, zorder=2)
                    ax.scatter(
                        [path[-1, 0]],
                        [path[-1, 1]],
                        s=42,
                        marker=marker if episode.success else "x",
                        color=color if episode.success else FAILURE,
                        linewidth=1.4,
                        zorder=7,
                    )
            successes = sum(e.success for e in group)
            ax.set_title(
                f"{WORDING_LABELS[wording]}\n{successes}/{len(group)} successes",
                loc="left",
                fontsize=9.5,
                fontweight="bold",
            )
            _format_compact_axes(ax)
            if column_index == 0:
                ax.set_ylabel(f"{MODEL_LABELS[model_id]}\nrobot-forward (m)")
            if row_index == 1:
                ax.set_xlabel("← left · lateral (m) · right →")
    legend = [
        Line2D([0], [0], color=LEFT_COLOR, marker="<", linestyle="none", label="left request · success"),
        Line2D([0], [0], color=RIGHT_COLOR, marker=">", linestyle="none", label="right request · success"),
        Line2D([0], [0], color=FAILURE, marker="x", linestyle="none", label="failure endpoint"),
        Line2D([0], [0], color=MUTED, linewidth=1, alpha=0.4, label="executed cube path"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Every executed cube path and endpoint", x=0.055, y=0.99, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.055,
        0.95,
        "Matched scenes and seeds · triangles are successes · red × marks are failures",
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.12, wspace=0.18, hspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _render_failure_anatomy(episodes: list[Episode], output: Path) -> None:
    models = ["pi05_droid_vla", "cosmos3_edge_droid_wam"]
    slots = [
        (wording, direction)
        for wording in ("canonical", "short_paraphrase", "declarative_goal", "contrastive_goal")
        for direction in ("left", "right")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 7.2), sharex=True, sharey=True)
    for ax, model_id in zip(axes, models):
        left = np.zeros(len(slots), dtype=float)
        for stage in OUTCOME_STAGE_ORDER:
            values = []
            for wording, direction in slots:
                group = [
                    episode
                    for episode in episodes
                    if episode.model_id == model_id
                    and episode.wording == wording
                    and episode.direction == direction
                ]
                values.append(sum(episode.outcome_stage == stage for episode in group))
            bars = ax.barh(
                np.arange(len(slots)),
                values,
                left=left,
                color=OUTCOME_STAGE_COLORS[stage],
                edgecolor="white",
                linewidth=0.55,
                label=OUTCOME_STAGE_LABELS[stage],
            )
            for bar, value in zip(bars, values):
                if value:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_y() + bar.get_height() / 2,
                        str(value),
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color="white" if stage not in {"no_cube_interaction", "interaction_without_verified_pickup"} else INK,
                    )
            left += np.asarray(values)
        ax.set_title(MODEL_LABELS[model_id], loc="left", fontsize=12, fontweight="bold")
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 11, 2))
        ax.set_xlabel("Episodes (n = 10 per row)")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
    axes[0].invert_yaxis()
    axes[0].set_yticks(
        np.arange(len(slots)),
        [f"{WORDING_LABELS[wording]} · {direction.upper()}" for wording, direction in slots],
    )
    fig.suptitle("Where each rollout stopped making progress", x=0.08, y=0.99, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.08,
        0.935,
        "Mutually exclusive terminal anatomy from raw events and robot-frame cube paths · every episode included",
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=8.5,
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.84, bottom=0.16, wspace=0.10)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _format_compact_axes(ax: plt.Axes) -> None:
    ax.axhline(0, color=GRID, linewidth=0.55)
    ax.axvline(0, color=GRID, linewidth=0.55)
    ax.set_xlim(-LATERAL_LIMIT_M, LATERAL_LIMIT_M)
    ax.set_ylim(FORWARD_MIN_M, FORWARD_MAX_M)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.7)


def _find_episode(
    episodes: Iterable[Episode], model_id: str, wording: str, direction: str, seed: int
) -> Episode | None:
    return next(
        (
            episode
            for episode in episodes
            if episode.model_id == model_id
            and episode.wording == wording
            and episode.direction == direction
            and episode.seed == seed
        ),
        None,
    )


def _render_first_seed_stress(episodes: list[Episode], output: Path, *, square: bool) -> bool:
    models = ["pi05_droid_vla", "cosmos3_edge_droid_wam"]
    wordings = ["declarative_goal", "contrastive_goal"]
    selected = [
        _find_episode(episodes, model, wording, "left", 7200)
        for model in models
        for wording in wordings
    ]
    if any(episode is None for episode in selected):
        return False
    figsize, dpi = ((12, 12), 100) if square else ((16, 9), 100)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, episode in zip(axes.flat, selected):
        assert episode is not None
        _render_episode_axis(ax, episode, compact=True, show_ee=False, show_prompt=False)
        ax.text(
            0.02,
            0.02,
            f"seed {episode.seed} · final {episode.endpoint_class.replace('_', ' ')}",
            transform=ax.transAxes,
            fontsize=8.5,
            color=MUTED,
            ha="left",
            va="bottom",
        )
    fig.suptitle(
        "Same scene. Same seed. One wording change.",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=22 if not square else 19,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.94,
        "Direct declarative vs contrastive LEFT requests · dashed green is an illustrative route, not the scored path",
        ha="left",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.055,
        0.015,
        "Primary display rule: lowest registered stress seed (7200), fixed across both checkpoints and both wordings.",
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.07, wspace=0.08, hspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return True


def _first_discordant_pair(episodes: list[Episode], model_id: str) -> tuple[Episode, Episode] | None:
    for direction in ("left", "right"):
        seeds = sorted(
            {
                episode.seed
                for episode in episodes
                if episode.model_id == model_id
                and episode.direction == direction
                and episode.wording in {"declarative_goal", "contrastive_goal"}
            }
        )
        for seed in seeds:
            declarative = _find_episode(episodes, model_id, "declarative_goal", direction, seed)
            contrastive = _find_episode(episodes, model_id, "contrastive_goal", direction, seed)
            if declarative and contrastive and declarative.success != contrastive.success:
                return declarative, contrastive
    return None


def _render_retrospective_discordance(episodes: list[Episode], output: Path) -> bool:
    pairs = [_first_discordant_pair(episodes, model) for model in MODEL_LABELS]
    if any(pair is None for pair in pairs):
        return False
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for row, pair in enumerate(pairs):
        assert pair is not None
        for column, episode in enumerate(pair):
            _render_episode_axis(axes[row, column], episode, compact=True, show_ee=False, show_prompt=False)
            axes[row, column].text(
                0.02,
                0.02,
                f"seed {episode.seed} · {episode.direction.upper()} request",
                transform=axes[row, column].transAxes,
                fontsize=8.5,
                color=MUTED,
                ha="left",
            )
    fig.suptitle(
        "When wording flips the outcome",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.055,
        0.94,
        "First same-seed declarative/contrastive discordance per checkpoint · retrospective diagnostic exemplar",
        ha="left",
        color=MUTED,
        fontsize=11,
    )
    fig.text(
        0.055,
        0.015,
        "Exemplar selection is explicitly outcome-aware; aggregate rates and the complete gallery carry the inferential weight.",
        ha="left",
        color=MUTED,
        fontsize=9,
    )
    fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.07, wspace=0.08, hspace=0.20)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100)
    plt.close(fig)
    return True


def _render_social_scorecard(episodes: list[Episode], output: Path, *, square: bool) -> bool:
    wordings = ["canonical", "short_paraphrase", "declarative_goal", "contrastive_goal"]
    columns = [
        ("pi05_droid_vla", "left"),
        ("pi05_droid_vla", "right"),
        ("cosmos3_edge_droid_wam", "left"),
        ("cosmos3_edge_droid_wam", "right"),
    ]
    counts = np.zeros((len(wordings), len(columns)), dtype=int)
    totals = np.zeros_like(counts)
    for row, wording in enumerate(wordings):
        for column, (model_id, direction) in enumerate(columns):
            group = [
                episode
                for episode in episodes
                if episode.model_id == model_id
                and episode.wording == wording
                and episode.direction == direction
            ]
            totals[row, column] = len(group)
            counts[row, column] = sum(episode.success for episode in group)
    if not np.all(totals == 10):
        return False

    figsize, dpi = ((12, 12), 100) if square else ((16, 9), 100)
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    if square:
        ax = fig.add_axes([0.15, 0.27, 0.80, 0.46])
        title_y, subtitle_y, byline_y, note_y = 0.91, 0.84, 0.125, 0.075
        title_size = 25
    else:
        ax = fig.add_axes([0.19, 0.22, 0.72, 0.54])
        title_y, subtitle_y, byline_y, note_y = 0.91, 0.835, 0.105, 0.055
        title_size = 29
    cmap = LinearSegmentedColormap.from_list(
        "scorecard", ["#EDF2F4", "#9ECDBD", "#15785A"]
    )
    rates = counts / totals
    ax.imshow(rates, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for row in range(rates.shape[0]):
        for column in range(rates.shape[1]):
            value = rates[row, column]
            ax.text(
                column,
                row - 0.07,
                f"{counts[row, column]}/10",
                ha="center",
                va="center",
                fontsize=16 if not square else 14,
                fontweight="bold",
                color="white" if value >= 0.62 else INK,
            )
            ax.text(
                column,
                row + 0.24,
                f"{value:.0%}",
                ha="center",
                va="center",
                fontsize=9.5,
                color="white" if value >= 0.62 else MUTED,
            )
    ax.set_xticks(
        np.arange(4),
        ["π0.5 VLA\nLEFT", "π0.5 VLA\nRIGHT", "Cosmos WAM\nLEFT", "Cosmos WAM\nRIGHT"],
        fontweight="bold",
    )
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=12, length=0)
    ax.set_yticks(np.arange(4), [WORDING_LABELS[wording] for wording in wordings])
    ax.tick_params(axis="y", pad=10, length=0)
    ax.axvline(1.5, color="white", linewidth=8)
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(
        0.075,
        title_y,
        "Steerability is not one number.",
        ha="left",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.075,
        subtitle_y,
        "Same neutral start · episode-static prompts · matched seeds · no coach or oracle",
        ha="left",
        va="top",
        fontsize=12 if not square else 11,
        color=MUTED,
    )
    fig.text(
        0.075,
        byline_y,
        "Ali Adeeb Abbas · Senior Scientist, General Motors · personal analysis",
        ha="left",
        va="bottom",
        fontsize=10,
        color=INK,
    )
    fig.text(
        0.075,
        note_y,
        "Binary RoboLab task success. One public checkpoint per model class; this is a checkpoint comparison, not a class ranking.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)
    return True


def _render_gallery(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    cards = []
    for row in rows:
        relative_image = Path("..") / "episodes" / Path(row["figure_path"]).name
        outcome = "success" if row["binary_success"] else "failure"
        stage = row["outcome_stage"]
        cards.append(
            f'''<article class="episode" data-model="{html.escape(row['model_id'])}" data-wording="{html.escape(row['wording'])}" data-direction="{row['direction']}" data-outcome="{outcome}" data-stage="{stage}">
  <a href="{html.escape(str(relative_image))}"><img loading="lazy" src="{html.escape(str(relative_image))}" alt="{html.escape(outcome)} trajectory for {html.escape(row['instruction'])}, seed {row['episode_seed']}"></a>
  <div class="meta"><strong>{outcome.upper()}</strong> · {html.escape(row['outcome_stage_label'])} · {html.escape(MODEL_LABELS[row['model_id']])} · {html.escape(WORDING_LABELS[row['wording']])} · {row['direction'].upper()} · seed {row['episode_seed']}</div>
</article>'''
        )
    document = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Steerability trajectory evidence</title>
<style>
:root{{--ink:#142b3a;--muted:#667785;--paper:#fbfcfd;--line:#d8e0e5;--success:#17845f;--failure:#c84b4b}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
header{{max-width:1500px;margin:auto;padding:42px 28px 24px}} h1{{font-size:clamp(30px,5vw,54px);line-height:1;margin:0 0 14px;letter-spacing:-.035em}} p{{max-width:920px;color:var(--muted);line-height:1.55}}
.stats{{display:flex;gap:20px;flex-wrap:wrap;margin:22px 0 0}} .stat{{border-left:3px solid var(--line);padding-left:11px}} .stat b{{font-size:24px;display:block}} .stat span{{color:var(--muted);font-size:13px}}
.controls{{position:sticky;top:0;z-index:20;background:rgba(251,252,253,.96);border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:12px 28px;display:flex;gap:10px;flex-wrap:wrap}}
select{{font:inherit;color:var(--ink);background:white;border:1px solid var(--line);border-radius:999px;padding:8px 34px 8px 12px}}
main{{max-width:1500px;margin:auto;padding:24px 28px 80px;display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:18px}}
.episode{{background:white;border:1px solid var(--line);border-radius:14px;overflow:hidden}} .episode img{{display:block;width:100%;height:auto}} .meta{{border-top:1px solid var(--line);padding:11px 13px;color:var(--muted);font-size:12px;line-height:1.45}} .episode[data-outcome="success"] strong{{color:var(--success)}} .episode[data-outcome="failure"] strong{{color:var(--failure)}}
.hidden{{display:none}} footer{{max-width:1500px;margin:auto;padding:0 28px 40px;color:var(--muted);font-size:12px}}
@media(max-width:520px){{header{{padding:28px 16px 18px}}.controls{{padding:10px 16px}}main{{padding:18px 16px 60px;grid-template-columns:1fr}}}}
</style>
</head>
<body>
<header>
  <h1>Steerability, path by path.</h1>
  <p>Every completed oracle-free episode is included. The translucent cone is the scored requested side. The dashed green arrow is only an illustrative direct route: RoboLab does not require one canonical path. The dark line is the executed cube path; the star is its final endpoint.</p>
  <div class="stats"><div class="stat"><b>{summary['rendered_episode_count']}</b><span>rendered episodes</span></div><div class="stat"><b>{summary['success_count']}</b><span>successes</span></div><div class="stat"><b>{summary['failure_count']}</b><span>failures</span></div></div>
</header>
<div class="controls" aria-label="Gallery filters">
  <select id="model" aria-label="Model"><option value="all">All models</option><option value="pi05_droid_vla">π0.5 VLA</option><option value="cosmos3_edge_droid_wam">Cosmos WAM</option></select>
  <select id="wording" aria-label="Wording"><option value="all">All wordings</option><option value="canonical">Canonical</option><option value="short_paraphrase">Short</option><option value="declarative_goal">Declarative</option><option value="contrastive_goal">Contrastive</option></select>
  <select id="direction" aria-label="Direction"><option value="all">Both directions</option><option value="left">Left</option><option value="right">Right</option></select>
  <select id="outcome" aria-label="Outcome"><option value="all">Successes + failures</option><option value="success">Success only</option><option value="failure">Failure only</option></select>
  <select id="stage" aria-label="Outcome stage"><option value="all">All progress stages</option><option value="success">Success</option><option value="ended_in_goal_without_terminal_success">Ended in goal · terminal failure</option><option value="entered_goal_then_lost_it">Entered goal · lost it</option><option value="picked_never_entered_goal">Picked · never entered goal</option><option value="interaction_without_verified_pickup">Interaction · no verified pickup</option><option value="no_cube_interaction">No cube interaction</option></select>
</div>
<main>{''.join(cards)}</main>
<footer>Binary success is the official RoboLab terminal predicate. Endpoint region is shown separately in the machine-readable index.</footer>
<script>
const controls=[...document.querySelectorAll('select')]; const cards=[...document.querySelectorAll('.episode')];
function apply(){{const f=Object.fromEntries(controls.map(x=>[x.id,x.value]));cards.forEach(c=>{{const show=Object.entries(f).every(([k,v])=>v==='all'||c.dataset[k]===v);c.classList.toggle('hidden',!show)}})}}
controls.forEach(x=>x.addEventListener('change',apply));
</script>
</body>
</html>
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document)


def _group_summary(episodes: list[Episode]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[(episode.model_id, episode.wording, episode.direction)].append(episode)
    output = []
    for (model_id, wording, direction), values in sorted(groups.items()):
        endpoint_counts = Counter(value.endpoint_class for value in values)
        stage_counts = Counter(value.outcome_stage for value in values)
        output.append(
            {
                "model_id": model_id,
                "wording": wording,
                "direction": direction,
                "episodes": len(values),
                "successes": sum(value.success for value in values),
                "failures": sum(not value.success for value in values),
                "endpoint_class_counts": dict(sorted(endpoint_counts.items())),
                "outcome_stage_counts": {
                    stage: stage_counts.get(stage, 0) for stage in OUTCOME_STAGE_ORDER
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--selection-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    manifest = _load_json(args.manifest)
    selection_plan = _load_json(args.selection_plan)
    expected = int(manifest["expected_episode_count"])
    episodes: list[Episode] = []
    missing: list[dict[str, Any]] = []
    for condition in manifest["conditions"]:
        for direction in ("left", "right"):
            for run, seed in enumerate(condition["episode_seeds"]):
                try:
                    episodes.append(_load_episode(condition, direction, run, int(seed)))
                except (FileNotFoundError, KeyError, OSError, RuntimeError) as error:
                    missing.append(
                        {
                            "condition_id": condition["id"],
                            "direction": direction,
                            "run": run,
                            "episode_seed": int(seed),
                            "reason": str(error),
                        }
                    )
    if missing and not args.allow_incomplete:
        preview = "\n".join(json.dumps(value, sort_keys=True) for value in missing[:10])
        raise RuntimeError(f"Missing {len(missing)} of {expected} registered episodes:\n{preview}")
    episodes.sort(key=_sort_key)
    if not episodes:
        raise RuntimeError("No complete episodes were found")

    output = args.output.resolve()
    episode_dir = output / "episodes"
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        figure = episode_dir / _episode_slug(episode)
        _render_episode(episode, figure)
        rows.append(_trajectory_index_row(episode, figure, repo_root))

    _write_csv(output / "trajectory_index.csv", rows)
    _dump_json(output / "trajectory_index.json", rows)
    _render_endpoint_atlas(episodes, output / "blog" / "all_executed_paths_and_endpoints.png")
    _render_failure_anatomy(episodes, output / "blog" / "failure_progress_anatomy.png")

    social_outputs: dict[str, str | None] = {}
    for name, square in (("first_seed_stress_landscape_1600x900.png", False), ("first_seed_stress_square_1200x1200.png", True)):
        path = output / "social" / name
        rendered = _render_first_seed_stress(episodes, path, square=square)
        social_outputs[name] = str(path.relative_to(repo_root)) if rendered else None
    retrospective = output / "social" / "retrospective_wording_discordance_1600x900.png"
    rendered = _render_retrospective_discordance(episodes, retrospective)
    social_outputs[retrospective.name] = str(retrospective.relative_to(repo_root)) if rendered else None
    for name, square in (
        ("steerability_scorecard_1600x900.png", False),
        ("steerability_scorecard_1200x1200.png", True),
    ):
        path = output / "social" / name
        rendered = _render_social_scorecard(episodes, path, square=square)
        social_outputs[name] = str(path.relative_to(repo_root)) if rendered else None

    summary = {
        "schema_version": "1.0.0",
        "status": "complete" if not missing else "incomplete_preview",
        "expected_episode_count": expected,
        "rendered_episode_count": len(episodes),
        "missing_episode_count": len(missing),
        "success_count": sum(episode.success for episode in episodes),
        "failure_count": sum(not episode.success for episode in episodes),
        "model_counts": dict(sorted(Counter(episode.model_id for episode in episodes).items())),
        "outcome_counts": {
            "success": sum(episode.success for episode in episodes),
            "failure": sum(not episode.success for episode in episodes),
        },
        "outcome_stage_counts": {
            stage: sum(episode.outcome_stage == stage for episode in episodes)
            for stage in OUTCOME_STAGE_ORDER
        },
        "group_summaries": _group_summary(episodes),
        "missing": missing,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "selection_plan": str(args.selection_plan.resolve()),
        "selection_plan_sha256": _sha256(args.selection_plan),
        "coordinate_convention": "page x = -robot-frame y, so robot-left appears left; page y = robot-frame x (forward)",
        "scored_goal": "released cube inside the requested RoboLab 45-degree horizontal cone",
        "illustrative_route_disclaimer": "The dashed direct route is explanatory only and is not a scored or claimed ground-truth trajectory.",
        "all_episode_policy": selection_plan["all_episode_policy"],
        "social_outputs": social_outputs,
        "blog_outputs": {
            "all_executed_paths_and_endpoints": str(
                (output / "blog" / "all_executed_paths_and_endpoints.png").relative_to(repo_root)
            ),
            "failure_progress_anatomy": str(
                (output / "blog" / "failure_progress_anatomy.png").relative_to(repo_root)
            ),
        },
    }
    _dump_json(output / "summary.json", summary)
    _render_gallery(output / "gallery" / "index.html", rows, summary)
    print(json.dumps({key: summary[key] for key in ("status", "rendered_episode_count", "success_count", "failure_count", "missing_episode_count")}, indent=2))


if __name__ == "__main__":
    _configure_plotting()
    main()
