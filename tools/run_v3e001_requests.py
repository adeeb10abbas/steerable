#!/usr/bin/env python3
"""Issue V3-E001 fixed-observation requests without executing actions.

This runner is deliberately model-client only: it reads the four settled
observations, hashes the raw/preprocessed request fields, and writes every
response or infrastructure-invalid attempt to JSONL.  It never instantiates
Isaac or forwards an action to a controller.
"""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
SEEDS = range(9400, 9427)


def sha(value: Any) -> str:
    if isinstance(value, np.ndarray):
        payload = np.ascontiguousarray(value).tobytes()
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def resize_pad(image: np.ndarray, h: int, w: int) -> np.ndarray:
    scale = min(w / image.shape[1], h / image.shape[0])
    nw, nh = max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = resized
    return out


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def first(z: dict[str, np.ndarray], *keys: str) -> np.ndarray:
    for k in keys:
        if k in z:
            return z[k]
    raise KeyError(keys)


def model_request(model: str, z: dict[str, np.ndarray], prompt: str, seed: int, client: Any) -> tuple[dict, dict]:
    # Captures are [H,W,3] and state arrays are [7]/[1].  These transforms are
    # the exact preprocessing paths in RoboLab's released clients.
    shoulder = first(z, "image_obs/over_shoulder_left_camera", "over_shoulder_left_camera")
    wrist = first(z, "image_obs/wrist_cam", "wrist_cam")
    head = first(z, "image_obs/head_camera", "head_camera")
    joints = first(z, "proprio_obs/arm_joint_pos", "arm_joint_pos").astype(np.float32)
    grip = first(z, "proprio_obs/gripper_pos", "gripper_pos").astype(np.float32)
    if shoulder.ndim == 4: shoulder = shoulder[0]
    if wrist.ndim == 4: wrist = wrist[0]
    if head.ndim == 4: head = head[0]
    if joints.ndim > 1: joints = joints[0]
    if grip.ndim > 1: grip = grip[0]
    if model == "pi05":
        req = {
            "observation/exterior_image_1_left": resize_pad(shoulder, 224, 224),
            "observation/wrist_image_left": resize_pad(wrist, 224, 224),
            "observation/joint_position": joints,
            "observation/gripper_position": grip,
            "prompt": prompt,
            "sampling_seed": int(seed),
        }
    elif model == "nano":
        # Official Cosmos RoboLab server accepts the composed [540,640,3]
        # image plus state.  The fixture has all three exterior views; retain
        # the exact raw image hash before this registered resize.
        req = {
            "observation/image": cv2.resize(shoulder, (640, 540), interpolation=cv2.INTER_AREA).astype(np.uint8),
            "observation/joint_position": joints,
            "observation/gripper_position": grip,
            "prompt": prompt,
            "sampling_seed": int(seed),
        }
    elif model == "dreamzero":
        req = {
            "observation/exterior_image_0_left": resize_pad(shoulder, 180, 320),
            "observation/exterior_image_1_left": resize_pad(head, 180, 320),
            "observation/wrist_image_left": resize_pad(wrist, 180, 320),
            "observation/joint_position": joints.astype(np.float64),
            "observation/cartesian_position": np.zeros(6, dtype=np.float64),
            "observation/gripper_position": grip.astype(np.float64),
            "prompt": prompt,
            "session_id": f"v3e001-{seed}-{sha(prompt)[:12]}",
            "endpoint": "infer",
        }
    else:
        raise ValueError(model)
    req_hashes = {k: sha(v) for k, v in req.items() if k != "prompt"}
    req_hashes["prompt"] = sha(req["prompt"])
    started = time.perf_counter()
    response = dict(client.infer(req))
    response["wall_time_s"] = time.perf_counter() - started
    action = response.get("actions", response.get("action"))
    if action is not None:
        arr = np.asarray(action)
        response["action_shape"] = list(arr.shape)
        response["action_sha256"] = sha(arr)
        response["action_finite"] = bool(np.isfinite(arr).all())
        response["action"] = arr.tolist()
    if "video" in response:
        arr = np.asarray(response["video"])
        response["future_shape"] = list(arr.shape)
        response["future_sha256"] = sha(arr)
        response["video"] = arr.tolist()
    return {"request_hashes": req_hashes, "request_shapes": {k: list(np.asarray(v).shape) for k,v in req.items() if k != "prompt"}}, jsonable(response)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("pi05", "nano", "dreamzero"), required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--control-left", type=Path, required=True)
    ap.add_argument("--control-right", type=Path, required=True)
    ap.add_argument("--mirror-left", type=Path, required=True)
    ap.add_argument("--mirror-right", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--only-exact-repeats", action="store_true")
    args = ap.parse_args()
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    client = WebsocketClientPolicy(args.host, args.port)
    paths = {
        ("control", "left"): args.control_left,
        ("control", "right"): args.control_right,
        ("position_mirrored", "left"): args.mirror_left,
        ("position_mirrored", "right"): args.mirror_right,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out:
        def issue(layout: str, relation: str, seed: int, repeat: bool = False) -> None:
            prompt = LEFT if relation == "left" else RIGHT
            rec = {"schema_version":"vla-wam-shared-v3e001-request-v1","model_id":args.model,"layout":layout,"relation":relation,"prompt":prompt,"sampling_seed":seed,"exact_repeat":repeat,"status":"valid"}
            try:
                z = load_npz(paths[(layout, relation)])
                rh, response = model_request(args.model, z, prompt, seed, client)
                rec.update(rh, response=response)
            except Exception as exc:
                rec.update(status="infrastructure_invalid", error_type=type(exc).__name__, error=str(exc))
            out.write(json.dumps(jsonable(rec), sort_keys=True, separators=(",", ":")) + "\n"); out.flush()
        # Full matched queue (108 requests) plus the two exact repeats per
        # layout required by the registration (4 requests).
        if not args.only_exact_repeats:
            for seed in SEEDS:
                for layout, relation in (("control","left"),("control","right"),("position_mirrored","left"),("position_mirrored","right")):
                    issue(layout, relation, seed)
            for layout in ("control", "position_mirrored"):
                issue(layout, "left", 9400, repeat=True)
        for layout in ("control", "position_mirrored"):
            issue(layout, "right", 9400, repeat=True)


if __name__ == "__main__":
    main()
