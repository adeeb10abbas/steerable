#!/usr/bin/env python3
"""Score Cosmos imagined spatial relations against executed RoboLab state.

The visual localizer is deliberately prompt-blind: it never receives the
policy instruction or requested direction.  It locates the cube and bowl in
the two fixed third-person panels.  A calibration-only homography maps each
panel to the robot/table XY plane.  Confirmation scoring is frozen by the
calibration JSON and uses no confirmation labels to tune thresholds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DEFAULT_MODEL = Path(
    "/home/ali/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-2B-Instruct/snapshots/"
    "89644892e4d85e24eaac8bacfd4f463576704203"
)
FRAME_INDICES = (8, 16, 24, 32)
CAMERAS = ("left_camera", "right_camera")
RELATION_PROBE_STYLES = {
    "task",
    "exact_repeat_control",
    "paraphrase_control",
    "opposite_relation_control",
    "point_spatial",
    "point_spatial_counterfactual",
    "combination",
    "combination_counterfactual",
}
CAMERA_GEOMETRY = {
    "left_camera": {
        "position": [0.05, 0.57, 0.66],
        "quaternion_wxyz": [-0.393, -0.195, 0.399, 0.805],
    },
    "right_camera": {
        "position": [0.05, -0.57, 0.66],
        "quaternion_wxyz": [0.805, 0.399, -0.195, -0.393],
    },
}
for _geometry in CAMERA_GEOMETRY.values():
    _geometry.update(
        {
            "source_width_px": 1280,
            "source_height_px": 720,
            "focal_length": 2.1,
            "horizontal_aperture": 5.376,
            "vertical_aperture": 3.024,
        }
    )
LOCALIZER_PROMPT = (
    "This is one fixed third-person camera looking at a robot and wood table. "
    "Locate the physical multicolored Rubik's cube and the red bowl. Return "
    "their center points as normalized coordinates from 0 to 1000. Ignore the "
    "robot, banana, wood, and background. JSON only: "
    '{"cube_center":[x,y] or null,"bowl_center":[x,y] or null}.'
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object in localizer output: {text!r}")
    value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object, got {type(value).__name__}")
    return value


def _normalize_point(value: Any) -> list[float] | None:
    if value is None or not isinstance(value, list) or len(value) != 2:
        return None
    try:
        point = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(point).all():
        return None
    # Qwen occasionally uses [0, 1] despite the explicit [0, 1000] request.
    if np.max(np.abs(point)) <= 1.5:
        point *= 1000.0
    if np.any(point < 0.0) or np.any(point > 1000.0):
        return None
    return point.tolist()


@dataclass
class QwenLocalizer:
    model_path: Path

    def __post_init__(self) -> None:
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map="cuda",
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)

    def locate(self, camera_image: np.ndarray) -> dict[str, Any]:
        image = Image.fromarray(cv2.resize(camera_image, (640, 352), interpolation=cv2.INTER_CUBIC))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": LOCALIZER_PROMPT},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        elapsed = time.perf_counter() - started
        raw = self.processor.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )[0]
        try:
            parsed = _parse_json_object(raw)
        except (ValueError, TypeError, json.JSONDecodeError):
            return {
                "cube_center": None,
                "bowl_center": None,
                "raw": raw,
                "latency_s": elapsed,
                "parse_ok": False,
            }
        return {
            "cube_center": _normalize_point(parsed.get("cube_center")),
            "bowl_center": _normalize_point(parsed.get("bowl_center")),
            "raw": raw,
            "latency_s": elapsed,
            "parse_ok": True,
        }


def _bottom_camera_panels(frame: np.ndarray) -> dict[str, np.ndarray]:
    height, width = frame.shape[:2]
    bottom = frame[(height * 2) // 3 :, :]
    half = width // 2
    return {"left_camera": bottom[:, :half], "right_camera": bottom[:, half:]}


def _read_rgb_frames(path: Path, indices: tuple[int, ...]) -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    index = 0
    try:
        while wanted:
            ok, bgr = capture.read()
            if not ok:
                break
            if index in wanted:
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                wanted.remove(index)
            index += 1
    finally:
        capture.release()
    return frames


def _episode_hdf(task_dir: Path, episode_index: int) -> Path:
    path = task_dir / f"run_{episode_index}.hdf5"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _episode_state(task_dir: Path, episode_index: int) -> dict[str, np.ndarray]:
    with h5py.File(_episode_hdf(task_dir, episode_index), "r") as handle:
        # RoboLab writes one HDF5 file per run and numbers the sole trajectory
        # locally as demo_0. The run index belongs in the filename, not the
        # group name. Calibration covered only run 0, so enforce the on-disk
        # invariant explicitly before scoring the multi-run confirmation grid.
        demo_names = list(handle["data"])
        if demo_names != ["demo_0"]:
            raise RuntimeError(
                f"Expected one per-run trajectory named demo_0 in {handle.filename}; "
                f"found {demo_names}"
            )
        demo = handle["data/demo_0"]
        return {
            "cube": np.asarray(demo["bbox/centroid/rubiks_cube"], dtype=np.float64),
            "bowl": np.asarray(demo["bbox/centroid/bowl"], dtype=np.float64),
        }


def _iter_chunks(task_dir: Path):
    prediction_root = task_dir / "predicted_chunks"
    for episode_dir in sorted(prediction_root.glob("episode_*")):
        episode_index = int(episode_dir.name.rsplit("_", 1)[1])
        for chunk_dir in sorted(episode_dir.glob("chunk_*")):
            metadata = json.loads((chunk_dir / "metadata.json").read_text())
            yield episode_index, chunk_dir, metadata


def _quaternion_matrix(quaternion_wxyz: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _camera_intrinsics(geometry: dict[str, Any]) -> tuple[float, float, float, float]:
    width = float(geometry["source_width_px"])
    height = float(geometry["source_height_px"])
    fx = float(geometry["focal_length"]) / float(geometry["horizontal_aperture"]) * width
    fy = float(geometry["focal_length"]) / float(geometry["vertical_aperture"]) * height
    return fx, fy, width / 2.0, height / 2.0


def _project_world_to_normalized(geometry: dict[str, Any], world_xyz: list[float]) -> np.ndarray:
    position = np.asarray(geometry["position"], dtype=np.float64)
    rotation = _quaternion_matrix(geometry["quaternion_wxyz"])
    camera_xyz = rotation.T @ (np.asarray(world_xyz, dtype=np.float64) - position)
    fx, fy, cx, cy = _camera_intrinsics(geometry)
    depth = -camera_xyz[2]
    if depth <= 0:
        raise ValueError("Calibration point is behind the camera")
    pixel = np.asarray(
        [fx * camera_xyz[0] / depth + cx, cy - fy * camera_xyz[1] / depth],
        dtype=np.float64,
    )
    return pixel / np.asarray([geometry["source_width_px"], geometry["source_height_px"]]) * 1000.0


def _unproject_to_plane(
    camera: dict[str, Any], point: list[float] | None, plane_z_m: float
) -> np.ndarray | None:
    if point is None:
        return None
    geometry = camera["geometry"]
    corrected = np.asarray(point, dtype=np.float64) + np.asarray(
        camera["normalized_pixel_bias"], dtype=np.float64
    )
    width = float(geometry["source_width_px"])
    height = float(geometry["source_height_px"])
    pixel = corrected / 1000.0 * np.asarray([width, height])
    fx, fy, cx, cy = _camera_intrinsics(geometry)
    ray_camera = np.asarray([(pixel[0] - cx) / fx, -(pixel[1] - cy) / fy, -1.0])
    ray_world = _quaternion_matrix(geometry["quaternion_wxyz"]) @ ray_camera
    origin = np.asarray(geometry["position"], dtype=np.float64)
    if abs(float(ray_world[2])) <= 1e-8:
        return None
    distance = (float(plane_z_m) - origin[2]) / ray_world[2]
    if distance <= 0:
        return None
    return (origin + distance * ray_world)[:2]


def _fit_camera_calibration(
    records: list[dict[str, Any]], camera: str, plane_z_m: float
) -> dict[str, Any]:
    geometry = CAMERA_GEOMETRY[camera]
    samples: list[dict[str, Any]] = []
    biases: list[np.ndarray] = []
    for record in records:
        localization = record["localization"][camera]
        for object_name in ("cube", "bowl"):
            point = localization[f"{object_name}_center"]
            if point is None:
                continue
            expected = _project_world_to_normalized(geometry, record["world"][object_name])
            bias = expected - np.asarray(point, dtype=np.float64)
            biases.append(bias)
            samples.append(
                {
                    "task_dir": record["task_dir"],
                    "episode_index": record["episode_index"],
                    "replan_index": record["replan_index"],
                    "object": object_name,
                    "localized_normalized": point,
                    "expected_normalized": expected.tolist(),
                    "bias_normalized": bias.tolist(),
                }
            )
    if len(biases) < 8:
        raise RuntimeError(f"Only {len(biases)} calibration points for {camera}")
    median_bias = np.median(np.asarray(biases), axis=0)
    camera_calibration = {
        "geometry": geometry,
        "normalized_pixel_bias": median_bias.tolist(),
    }
    residuals = []
    for sample in samples:
        estimated = _unproject_to_plane(
            camera_calibration, sample["localized_normalized"], plane_z_m
        )
        source_record = next(
            record
            for record in records
            if record["task_dir"] == sample["task_dir"]
            and record["episode_index"] == sample["episode_index"]
            and record["replan_index"] == sample["replan_index"]
        )
        target = np.asarray(source_record["world"][sample["object"]][:2])
        residual = float(np.linalg.norm(estimated - target))
        sample["estimated_world_xy"] = estimated.tolist()
        sample["target_world_xy"] = target.tolist()
        sample["world_residual_m"] = residual
        residuals.append(residual)
    return {
        **camera_calibration,
        "num_points": len(biases),
        "residual_median_m": float(np.median(residuals)),
        "residual_p90_m": float(np.percentile(residuals, 90)),
        "residual_max_m": float(np.max(residuals)),
        "points": samples,
    }


def _direction(cube: np.ndarray, bowl: np.ndarray) -> str:
    delta = cube[:2] - bowl[:2]
    norm = float(np.linalg.norm(delta))
    if norm <= 1e-6:
        return "neutral"
    cosine = float(delta[1] / norm)
    threshold = math.cos(math.radians(45.0))
    if cosine >= threshold:
        return "left"
    if cosine <= -threshold:
        return "right"
    return "neutral"


def _official_execution_relation(cube: np.ndarray, bowl: np.ndarray) -> str:
    if abs(float(cube[2] - bowl[2])) > 0.1:
        return "neutral"
    return _direction(cube, bowl)


def _request_direction(prompt: str) -> str:
    found = re.findall(r"\b(left|right)\b", prompt.lower())
    if len(set(found)) != 1:
        raise ValueError(f"Expected exactly one requested direction in {prompt!r}")
    return found[0]


def _frame_semantics(
    localization: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    estimates: dict[str, dict[str, list[float] | None]] = {}
    relations: dict[str, str | None] = {}
    for camera in CAMERAS:
        estimates[camera] = {}
        for object_name in ("cube", "bowl"):
            estimate = _unproject_to_plane(
                calibration["cameras"][camera],
                localization[camera][f"{object_name}_center"],
                calibration["object_centroid_plane_z_m"],
            )
            estimates[camera][object_name] = estimate.tolist() if estimate is not None else None
        cube = estimates[camera]["cube"]
        bowl = estimates[camera]["bowl"]
        relations[camera] = (
            _direction(np.asarray(cube), np.asarray(bowl)) if cube is not None and bowl is not None else None
        )

    reasons: list[str] = []
    if any(relations[camera] is None for camera in CAMERAS):
        reasons.append("missing_localization")
    if len({relations[camera] for camera in CAMERAS}) != 1:
        reasons.append("camera_relation_disagreement")
    disagreements = {}
    for object_name in ("cube", "bowl"):
        left = estimates["left_camera"][object_name]
        right = estimates["right_camera"][object_name]
        if left is None or right is None:
            disagreements[object_name] = None
            continue
        distance = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
        disagreements[object_name] = distance
        if distance > calibration["cross_camera_disagreement_threshold_m"]:
            reasons.append(f"{object_name}_cross_camera_disagreement")
    relation = relations["left_camera"] if not reasons else None
    return {
        "reliable": not reasons,
        "relation": relation,
        "reasons": reasons,
        "relations_by_camera": relations,
        "world_xy_by_camera": estimates,
        "cross_camera_disagreement_m": disagreements,
    }


def calibrate(args: argparse.Namespace) -> None:
    localizer = QwenLocalizer(args.model_path)
    records: list[dict[str, Any]] = []
    for task_dir in args.task_dir:
        state_cache: dict[int, dict[str, np.ndarray]] = {}
        for episode_index, chunk_dir, metadata in _iter_chunks(task_dir):
            state = state_cache.setdefault(episode_index, _episode_state(task_dir, episode_index))
            step = min(int(metadata["executed_step_start"]), len(state["cube"]) - 1)
            rgb = cv2.cvtColor(cv2.imread(str(chunk_dir / "conditioning.png")), cv2.COLOR_BGR2RGB)
            panels = _bottom_camera_panels(rgb)
            localization = {camera: localizer.locate(panels[camera]) for camera in CAMERAS}
            record = {
                "task_dir": str(task_dir.resolve()),
                "episode_index": episode_index,
                "replan_index": int(metadata["replan_index"]),
                "sampling_seed": metadata["server_sampling_seed"],
                "state_step": step,
                "world": {
                    "cube": state["cube"][step].tolist(),
                    "bowl": state["bowl"][step].tolist(),
                },
                "localization": localization,
            }
            records.append(record)
            print(
                f"calibration {task_dir.name} episode={episode_index} "
                f"chunk={metadata['replan_index']}"
            )

    plane_z_values = [
        record["world"][object_name][2]
        for record in records
        for object_name in ("cube", "bowl")
        if abs(record["world"]["cube"][2] - record["world"]["bowl"][2]) <= 0.03
    ]
    plane_z_m = float(np.median(plane_z_values))
    cameras = {
        camera: _fit_camera_calibration(records, camera, plane_z_m) for camera in CAMERAS
    }
    preliminary = {
        "cameras": cameras,
        "object_centroid_plane_z_m": plane_z_m,
        "cross_camera_disagreement_threshold_m": float("inf"),
    }
    disagreements: list[float] = []
    relation_checks: list[dict[str, Any]] = []
    for record in records:
        semantics = _frame_semantics(record["localization"], preliminary)
        for value in semantics["cross_camera_disagreement_m"].values():
            if value is not None:
                disagreements.append(value)
        expected = _direction(np.asarray(record["world"]["cube"]), np.asarray(record["world"]["bowl"]))
        observed = semantics["relations_by_camera"]
        relation_checks.append(
            {
                "task_dir": record["task_dir"],
                "episode_index": record["episode_index"],
                "replan_index": record["replan_index"],
                "expected": expected,
                "left_camera": observed["left_camera"],
                "right_camera": observed["right_camera"],
            }
        )
    # Frozen from the excluded 51xx future dry run: its p90 cross-camera
    # disagreement was 0.187 m.  Round upward to 0.20 m, while still requiring
    # the two cameras to agree on the categorical relation. Confirmation data
    # must not alter this threshold.
    threshold = 0.20
    camera_agreement = sum(
        check["left_camera"] == check["expected"] and check["right_camera"] == check["expected"]
        for check in relation_checks
    ) / len(relation_checks)
    calibration = {
        "schema_version": 1,
        "status": "frozen_calibration_only",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_path": str(args.model_path.resolve()),
        "model_snapshot": args.model_path.name,
        "localizer_prompt": LOCALIZER_PROMPT,
        "localizer_prompt_sha256": hashlib.sha256(LOCALIZER_PROMPT.encode()).hexdigest(),
        "frame_indices": list(FRAME_INDICES),
        "future_rule": (
            "at least 2 of 4 non-conditioning frames must be reliable; true if at least 75% of "
            "reliable frames satisfy the request; false if at most 25% do; uncertain otherwise"
        ),
        "relation_rule": "robot-frame 45-degree cone; execution additionally requires |delta_z| <= 0.1 m",
        "camera_mapping_rule": (
            "exact RoboLab pinhole intrinsics/extrinsics intersected with the calibrated table-object "
            "centroid plane; calibration estimates only a per-camera localizer pixel bias"
        ),
        "object_centroid_plane_z_m": plane_z_m,
        "cross_camera_disagreement_threshold_m": threshold,
        "cross_camera_threshold_source": (
            "excluded 51xx future dry run p90=0.187 m, rounded up to 0.20 m; both-camera "
            "categorical agreement remains mandatory"
        ),
        "cross_camera_disagreement_calibration_p90_m": float(np.percentile(disagreements, 90)),
        "conditioning_relation_agreement_fraction": camera_agreement,
        "conditioning_relation_checks": relation_checks,
        "cameras": cameras,
        "calibration_records": records,
    }
    _json_dump(args.output, calibration)
    print(f"wrote frozen calibration: {args.output}")
    print(f"conditioning relation agreement: {camera_agreement:.3f}")
    if camera_agreement < 0.8:
        raise RuntimeError(
            "Conditioning-frame relation agreement is below 0.8; do not use this calibration for confirmation"
        )


def _annotated_row(
    frames: dict[int, np.ndarray], frame_results: list[dict[str, Any]], chunk_label: str
) -> np.ndarray:
    cells = []
    by_index = {item["frame_index"]: item for item in frame_results}
    for frame_index in FRAME_INDICES:
        frame = frames[frame_index][(frames[frame_index].shape[0] * 2) // 3 :]
        cell = cv2.cvtColor(cv2.resize(frame, (480, 132)), cv2.COLOR_RGB2BGR)
        item = by_index[frame_index]
        label = f"{chunk_label} f{frame_index} {item['semantics']['relation'] or 'uncertain'}"
        color = (40, 210, 40) if item["semantics"]["reliable"] else (30, 90, 240)
        cv2.putText(cell, label, (7, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        # Draw the prompt-blind Qwen localizations used by the scorer. The
        # resized contact-sheet cell contains two equal-width camera panels.
        # Cube is cyan; bowl is red. This makes spot-auditing the semantic
        # judgment possible without trusting a text-only cache.
        panel_width = cell.shape[1] // 2
        for camera_index, camera in enumerate(CAMERAS):
            localization = item["localization"][camera]
            for object_name, point_color in (("cube", (255, 255, 0)), ("bowl", (0, 0, 255))):
                point = localization[f"{object_name}_center"]
                if point is None:
                    continue
                x = int(round(camera_index * panel_width + point[0] / 1000.0 * panel_width))
                y = int(round(point[1] / 1000.0 * cell.shape[0]))
                cv2.circle(cell, (x, y), 5, point_color, 2, cv2.LINE_AA)
        cells.append(cell)
    return np.concatenate(cells, axis=1)


def score(args: argparse.Namespace) -> None:
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("status") != "frozen_calibration_only":
        raise ValueError("Calibration is not frozen_calibration_only")
    if tuple(calibration["frame_indices"]) != FRAME_INDICES:
        raise ValueError("Frame-index mismatch between scorer and calibration")
    if args.model_path.resolve() != Path(calibration["model_path"]).resolve():
        raise ValueError(
            f"Localizer model mismatch: calibration={calibration['model_path']} "
            f"requested={args.model_path.resolve()}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    localizer: QwenLocalizer | None = None
    all_rows: list[dict[str, Any]] = []
    for task_dir in args.task_dir:
        task_key = f"{task_dir.parent.name}__{task_dir.name}"
        state_cache: dict[int, dict[str, np.ndarray]] = {}
        contact_rows: dict[int, list[np.ndarray]] = {}
        for episode_index, chunk_dir, metadata in _iter_chunks(task_dir):
            cache_path = (
                args.output_dir
                / "localization_cache"
                / task_key
                / f"episode_{episode_index:03d}_chunk_{int(metadata['replan_index']):03d}.json"
            )
            frames = _read_rgb_frames(chunk_dir / "future.mp4", FRAME_INDICES)
            if len(frames) != len(FRAME_INDICES):
                raise RuntimeError(f"Missing scored frames in {chunk_dir / 'future.mp4'}")
            if cache_path.exists() and not args.force:
                frame_results = json.loads(cache_path.read_text())["frames"]
                # Localization is calibration-independent, while its projected
                # semantics are not. Always refresh semantics from the frozen
                # calibration so a pre-freeze dry-run cache cannot leak stale
                # thresholds into confirmation scoring.
                for item in frame_results:
                    item["semantics"] = _frame_semantics(item["localization"], calibration)
                _json_dump(
                    cache_path,
                    {
                        "calibration_sha256": _sha256(args.calibration),
                        "frame_indices": list(FRAME_INDICES),
                        "frames": frame_results,
                    },
                )
            else:
                if localizer is None:
                    localizer = QwenLocalizer(args.model_path)
                frame_results = []
                for frame_index in FRAME_INDICES:
                    panels = _bottom_camera_panels(frames[frame_index])
                    localization = {camera: localizer.locate(panels[camera]) for camera in CAMERAS}
                    semantics = _frame_semantics(localization, calibration)
                    frame_results.append(
                        {
                            "frame_index": frame_index,
                            "localization": localization,
                            "semantics": semantics,
                        }
                    )
                _json_dump(
                    cache_path,
                    {
                        "calibration_sha256": _sha256(args.calibration),
                        "frame_indices": list(FRAME_INDICES),
                        "frames": frame_results,
                    },
                )

            requested = _request_direction(metadata["prompt"])
            reliable = [item for item in frame_results if item["semantics"]["reliable"]]
            requested_count = sum(item["semantics"]["relation"] == requested for item in reliable)
            requested_fraction = requested_count / len(reliable) if reliable else None
            if len(reliable) >= 2 and requested_fraction >= 0.75:
                imagined_requested: bool | None = True
            elif len(reliable) >= 2 and requested_fraction <= 0.25:
                imagined_requested = False
            else:
                imagined_requested = None

            state = state_cache.setdefault(episode_index, _episode_state(task_dir, episode_index))
            end_step = min(
                int(metadata["executed_step_start"]) + int(metadata["open_loop_horizon"]) - 1,
                len(state["cube"]) - 1,
            )
            executed_relation = _official_execution_relation(
                state["cube"][end_step], state["bowl"][end_step]
            )
            executed_requested = executed_relation == requested
            if imagined_requested is None:
                quadrant = "uncertain_future"
            elif imagined_requested and executed_requested:
                quadrant = "imagines_requested_executes_requested"
            elif imagined_requested and not executed_requested:
                quadrant = "imagines_requested_executes_not_requested"
            elif not imagined_requested and executed_requested:
                quadrant = "does_not_imagine_requested_executes_requested"
            else:
                quadrant = "neither_imagines_nor_executes_requested"
            row = {
                "task_dir": str(task_dir.resolve()),
                "episode_index": episode_index,
                "replan_index": int(metadata["replan_index"]),
                "sampling_seed": metadata["server_sampling_seed"],
                "requested_relation": requested,
                "imagined_requested": imagined_requested,
                "reliable_future_frames": len(reliable),
                "requested_future_frames": requested_count,
                "execution_end_step": end_step,
                "executed_relation": executed_relation,
                "executed_requested": executed_requested,
                "quadrant": quadrant,
                "chunk_dir": str(chunk_dir.resolve()),
            }
            all_rows.append(row)
            contact_rows.setdefault(episode_index, []).append(
                _annotated_row(frames, frame_results, f"c{metadata['replan_index']:02d}")
            )
            print(
                f"score {task_dir.name} episode={episode_index} chunk={metadata['replan_index']} "
                f"quadrant={quadrant}"
            )

        audit_dir = args.output_dir / "audit" / task_key
        audit_dir.mkdir(parents=True, exist_ok=True)
        for episode_index, rows in contact_rows.items():
            cv2.imwrite(str(audit_dir / f"episode_{episode_index:03d}_contact_sheet.jpg"), np.concatenate(rows))

    csv_path = args.output_dir / "semantic_quadrants.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    quadrant_counts: dict[str, int] = {}
    for row in all_rows:
        quadrant_counts[row["quadrant"]] = quadrant_counts.get(row["quadrant"], 0) + 1
    summary = {
        "schema_version": 1,
        "calibration_path": str(args.calibration.resolve()),
        "calibration_sha256": _sha256(args.calibration),
        "num_chunks": len(all_rows),
        "num_certain_chunks": sum(row["imagined_requested"] is not None for row in all_rows),
        "coverage_fraction": sum(row["imagined_requested"] is not None for row in all_rows) / len(all_rows),
        "quadrant_counts": quadrant_counts,
        "rows": all_rows,
    }
    _json_dump(args.output_dir / "semantic_quadrants_summary.json", summary)


def score_probe(args: argparse.Namespace) -> None:
    """Apply the frozen prompt-blind future scorer to a fixed-observation probe."""
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("status") != "frozen_calibration_only":
        raise ValueError("Calibration is not frozen_calibration_only")
    if tuple(calibration["frame_indices"]) != FRAME_INDICES:
        raise ValueError("Frame-index mismatch between scorer and calibration")
    if args.model_path.resolve() != Path(calibration["model_path"]).resolve():
        raise ValueError(
            f"Localizer model mismatch: calibration={calibration['model_path']} "
            f"requested={args.model_path.resolve()}"
        )
    probe_manifest = json.loads((args.probe_dir / "manifest.json").read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    localizer: QwenLocalizer | None = None
    rows = []
    contact_rows = []
    for record in probe_manifest["records"]:
        condition = record["condition"]
        video_path = args.probe_dir / f"{condition}_future.mp4"
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        frames = _read_rgb_frames(video_path, FRAME_INDICES)
        if len(frames) != len(FRAME_INDICES):
            raise RuntimeError(f"Missing scored frames in {video_path}")
        cache_path = args.output_dir / "localization_cache" / f"{condition}.json"
        if cache_path.exists() and not args.force:
            frame_results = json.loads(cache_path.read_text())["frames"]
            for item in frame_results:
                item["semantics"] = _frame_semantics(item["localization"], calibration)
        else:
            if localizer is None:
                localizer = QwenLocalizer(args.model_path)
            frame_results = []
            for frame_index in FRAME_INDICES:
                panels = _bottom_camera_panels(frames[frame_index])
                localization = {camera: localizer.locate(panels[camera]) for camera in CAMERAS}
                frame_results.append(
                    {
                        "frame_index": frame_index,
                        "localization": localization,
                        "semantics": _frame_semantics(localization, calibration),
                    }
                )
        _json_dump(
            cache_path,
            {
                "calibration_sha256": _sha256(args.calibration),
                "frame_indices": list(FRAME_INDICES),
                "frames": frame_results,
            },
        )
        reliable = [item for item in frame_results if item["semantics"]["reliable"]]
        relation_counts = {
            relation: sum(item["semantics"]["relation"] == relation for item in reliable)
            for relation in ("left", "right", "neutral")
        }
        predicted_relation = None
        if len(reliable) >= 2:
            winner, count = max(relation_counts.items(), key=lambda item: item[1])
            if count / len(reliable) >= 0.75:
                predicted_relation = winner
        # The frozen localizer measures the cube/bowl relation. Applying that
        # predicate to an atomic gripper-motion command would silently score a
        # different behavior than the command asks for. Only styles whose goal
        # is actually a left/right cube placement receive a semantic label.
        semantic_applicable = record["style"] in RELATION_PROBE_STYLES
        if semantic_applicable:
            requested_relation = _request_direction(record["prompt"])
        else:
            requested_relation = None
        requested_fraction = (
            relation_counts[requested_relation] / len(reliable)
            if requested_relation is not None and reliable
            else None
        )
        if requested_relation is None or len(reliable) < 2:
            imagined_requested = None
        elif requested_fraction >= 0.75:
            imagined_requested = True
        elif requested_fraction <= 0.25:
            imagined_requested = False
        else:
            imagined_requested = None
        rows.append(
            {
                "condition": condition,
                "style": record["style"],
                "prompt": record["prompt"],
                "semantic_target": "cube_relative_to_bowl" if semantic_applicable else None,
                "semantic_applicable": semantic_applicable,
                "requested_relation": requested_relation,
                "predicted_relation": predicted_relation,
                "imagined_requested": imagined_requested,
                "reliable_future_frames": len(reliable),
                "relation_counts": relation_counts,
            }
        )
        contact_rows.append(_annotated_row(frames, frame_results, condition))
        print(
            f"probe {condition} predicted={predicted_relation} "
            f"imagined_requested={imagined_requested}"
        )

    cv2.imwrite(
        str(args.output_dir / "semantic_future_contact_sheet.jpg"),
        np.concatenate(contact_rows, axis=0),
    )
    _json_dump(
        args.output_dir / "semantic_future_summary.json",
        {
            "schema_version": 1,
            "status": "fixed_observation_secondary_diagnostic",
            "probe_manifest": str((args.probe_dir / "manifest.json").resolve()),
            "probe_manifest_sha256": _sha256(args.probe_dir / "manifest.json"),
            "calibration": str(args.calibration.resolve()),
            "calibration_sha256": _sha256(args.calibration),
            "conditions": len(rows),
            "conditions_with_predicted_relation": sum(
                row["predicted_relation"] is not None for row in rows
            ),
            "rows": rows,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--task-dir", type=Path, nargs="+", required=True)
    calibration.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.set_defaults(func=calibrate)
    scoring = subparsers.add_parser("score")
    scoring.add_argument("--task-dir", type=Path, nargs="+", required=True)
    scoring.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    scoring.add_argument("--calibration", type=Path, required=True)
    scoring.add_argument("--output-dir", type=Path, required=True)
    scoring.add_argument("--force", action="store_true")
    scoring.set_defaults(func=score)
    probe = subparsers.add_parser("score-probe")
    probe.add_argument("--probe-dir", type=Path, required=True)
    probe.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    probe.add_argument("--calibration", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--force", action="store_true")
    probe.set_defaults(func=score_probe)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
