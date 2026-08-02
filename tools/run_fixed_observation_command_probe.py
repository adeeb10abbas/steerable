#!/usr/bin/env python3
"""Run the frozen paper-inspired command probe against a local policy server.

The probe uses one byte-pinned neutral RoboLab observation for both models.
It is an interface diagnostic: action/video distances establish sensitivity,
not correctness. Closed-loop requested-goal success remains the primary metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from openpi_client import image_tools, websocket_client_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare arrays with shapes {a.shape} and {b.shape}")
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64) - b.astype(np.float64)))))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"Cannot compare arrays with shapes {a.shape} and {b.shape}")
    return float(np.mean(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def _save_mp4(video: np.ndarray, path: Path, fps: float) -> None:
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(f"Expected [T,H,W,3] future, got {video.shape}")
    height, width = video.shape[1:3]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {path}")
    for frame in video:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _split_conditioning(image: np.ndarray) -> dict[str, np.ndarray]:
    if image.shape != (540, 640, 3):
        raise ValueError(f"Expected frozen 540x640 RGB conditioning image, got {image.shape}")
    return {
        "wrist": image[:360],
        "left": image[360:, :320],
        "right": image[360:, 320:],
    }


def _request(
    model: str,
    image: np.ndarray,
    panels: dict[str, np.ndarray],
    source: dict[str, Any],
    prompt: str,
    sampling_seed: int,
) -> dict[str, Any]:
    common = {
        "observation/joint_position": np.asarray(source["joint_position"], dtype=np.float32),
        "observation/gripper_position": np.asarray(source["gripper_position"], dtype=np.float32),
        "prompt": prompt,
        "sampling_seed": sampling_seed,
    }
    if model == "cosmos":
        return {**common, "observation/image": image}
    return {
        **common,
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            panels["left"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            panels["wrist"], 224, 224
        ),
    }


def _annotated_observation(
    image: np.ndarray, grounding: dict[str, Any]
) -> np.ndarray:
    annotated = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    source_width, source_height = grounding["source_resolution_px"]
    colors = {
        "gripper": (255, 255, 0),
        "rubiks_cube": (0, 255, 0),
        "red_bowl": (0, 0, 255),
        "left_of_bowl_target": (255, 0, 255),
        "right_of_bowl_target": (0, 165, 255),
    }
    for name, point in grounding["points_px"].items():
        x = int(round(point[0] / source_width * 320))
        y = int(round(360 + point[1] / source_height * 180))
        cv2.circle(annotated, (x, y), 6, colors[name], 2, cv2.LINE_AA)
        cv2.putText(
            annotated,
            name,
            (x + 7, max(y - 6, 372)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.33,
            colors[name],
            1,
            cv2.LINE_AA,
        )
    return annotated


def _future_contact_sheet(
    records: list[dict[str, Any]], outputs: dict[str, dict[str, np.ndarray]]
) -> np.ndarray:
    rows = []
    for record in records:
        condition = record["condition"]
        video = outputs[condition]["video"]
        indices = (0, min(8, len(video) - 1), min(16, len(video) - 1), len(video) - 1)
        cells = []
        for frame_index in indices:
            cell = cv2.cvtColor(
                cv2.resize(video[frame_index], (320, 264), interpolation=cv2.INTER_AREA),
                cv2.COLOR_RGB2BGR,
            )
            cv2.putText(
                cell,
                f"{condition} f{frame_index}",
                (7, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (40, 220, 40),
                1,
                cv2.LINE_AA,
            )
            cells.append(cell)
        rows.append(np.concatenate(cells, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model", choices=["cosmos", "pi05"], required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--future-fps", type=float, default=15.0)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text())
    if plan.get("status") != "frozen_before_fixed_observation_probe":
        raise ValueError("Command probe plan is not frozen")
    source = plan["source"]
    image_path = Path(source["conditioning_png"])
    observed_sha = _sha256(image_path)
    if observed_sha != source["conditioning_png_sha256"]:
        raise ValueError(
            f"Conditioning image hash mismatch: expected {source['conditioning_png_sha256']}, "
            f"got {observed_sha}"
        )
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    raw_rgb_sha = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    if raw_rgb_sha != source["conditioning_raw_rgb_sha256"]:
        raise ValueError(
            f"Decoded RGB hash mismatch: expected {source['conditioning_raw_rgb_sha256']}, "
            f"got {raw_rgb_sha}"
        )
    panels = _split_conditioning(image)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(args.output_dir / "grounded_observation.png"),
        _annotated_observation(image, plan["grounding"]),
    )
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    records: list[dict[str, Any]] = []
    for condition in plan["conditions"]:
        condition_id = condition["id"]
        request = _request(
            args.model,
            image,
            panels,
            source,
            condition["prompt"],
            int(plan["sampling_seed"]),
        )
        request_started = time.perf_counter()
        response = client.infer(request)
        request_wall_seconds = time.perf_counter() - request_started
        action_key = "action" if "action" in response else "actions"
        action = np.asarray(response[action_key], dtype=np.float32)
        if action.ndim != 2 or action.shape[1] != 8:
            raise ValueError(f"Expected [T,8] action for {condition_id}, got {action.shape}")
        output: dict[str, np.ndarray] = {"action": action}
        np.save(args.output_dir / f"{condition_id}_action.npy", action)
        video = None
        if "video" in response:
            video = np.asarray(response["video"], dtype=np.uint8)
            output["video"] = video
            _save_mp4(video, args.output_dir / f"{condition_id}_future.mp4", args.future_fps)
        outputs[condition_id] = output
        record = {
            "condition": condition_id,
            "style": condition["style"],
            "prompt": condition["prompt"],
            "requested_sampling_seed": int(plan["sampling_seed"]),
            "server_sampling_seed": response.get("sampling_seed"),
            "request_wall_seconds": request_wall_seconds,
            "action_shape": list(action.shape),
            "future_shape": list(video.shape) if video is not None else None,
        }
        for optional_key in ("prompt_family", "requested_relation", "target_token_order"):
            if optional_key in condition:
                record[optional_key] = condition[optional_key]
        records.append(record)
        print(f"completed {args.model} {condition_id}")

    canonical = outputs["task_left"]
    repeat = outputs["task_left_repeat"]
    action_noise = _rms(canonical["action"], repeat["action"])
    future_noise = (
        _mae(canonical["video"], repeat["video"])
        if "video" in canonical and "video" in repeat
        else None
    )
    for record in records:
        condition_output = outputs[record["condition"]]
        action_effect = _rms(canonical["action"], condition_output["action"])
        record["action_rms_vs_task_left"] = action_effect
        record["action_effect_to_repeat_noise"] = (
            action_effect / action_noise if action_noise > 0.0 else None
        )
        record["action_exceeds_repeat_noise"] = action_effect > action_noise
        if "video" in canonical and "video" in condition_output:
            future_effect = _mae(canonical["video"], condition_output["video"])
            record["future_pixel_mae_vs_task_left"] = future_effect
            record["future_effect_to_repeat_noise"] = (
                future_effect / future_noise if future_noise and future_noise > 0.0 else None
            )
            record["future_exceeds_repeat_noise"] = future_effect > (future_noise or 0.0)

    csv_path = args.output_dir / "metrics.csv"
    fieldnames = list(records[0])
    for record in records[1:]:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    if all("video" in output for output in outputs.values()):
        cv2.imwrite(
            str(args.output_dir / "future_contact_sheet.jpg"),
            _future_contact_sheet(records, outputs),
        )

    manifest = {
        "schema_version": 1,
        "status": "fixed_observation_secondary_diagnostic",
        "model": args.model,
        "server": f"ws://{args.host}:{args.port}",
        "plan_path": str(args.plan.resolve()),
        "plan_sha256": _sha256(args.plan),
        "conditioning_png_sha256": observed_sha,
        "conditioning_raw_rgb_sha256": raw_rgb_sha,
        "sampling_seed": int(plan["sampling_seed"]),
        "timing_note": (
            "Client WebSocket round-trip around infer(), including transport, server inference, "
            "serialization, and any returned-video decoding. Cosmos returns actions plus decoded "
            "future video; pi0.5 returns actions only, so the timings represent deployed interfaces "
            "rather than an architecture-only comparison."
        ),
        "exact_repeat_action_rms": action_noise,
        "exact_repeat_future_pixel_mae": future_noise,
        "records": records,
        "claim_boundary": plan["reporting"]["claim_boundary"],
    }
    _json_dump(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
