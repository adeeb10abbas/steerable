"""Six registered, recorded Nano requests from one retained native observation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from typing import Any
import urllib.request

import numpy as np

from .adapters import NANO_CONFIG
from .family_campaign_executor import _fsync_json
from .paper_engineering import bound_file, record
from .policy_observations import CAMERAS, nano_observation
from .producer import NanoEvidenceProducer, make_nano_http_server

SCHEMA = "sgw-01-n3-fixed-input-registration-v1"
ORDER = ["LAT-D-POS", "LAT-D-POS", "LAT-D-NEG"] * 2


def save_array(path: Path, array: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, array, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def load_registration(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if (value.get("schema_version") != SCHEMA or value.get("model_config") != NANO_CONFIG
            or value.get("request_prompt_ids") != ORDER or value.get("maximum_model_requests") != 6
            or value.get("behavioral_episodes") != 0 or value.get("executed_actions") != 0
            or value.get("sampling_seed") != 1140
            or value.get("native_decode_video") is not True
            or value.get("release_permitted") is not False):
        raise ValueError("not the bounded six-request nonbehavioral N3 registration")
    for key in ("capture", "native_proprio_source", "native_server_source", "prompts", "engineering_verification"):
        bound_file(value[key])
    verified = json.loads(bound_file(value["engineering_verification"]).read_bytes())
    if verified["candidate_id"] != "SGW-ENG-008-LAT-057" or verified["status"] != "verified_all_six_pass":
        raise ValueError("fixed observation lacks its registered engineering qualification")
    return value


def native_observation(capture: dict[str, Any]) -> dict[str, Any]:
    images = {
        camera: np.load(bound_file(capture["views"][camera]["lossless_array"]), allow_pickle=False)[None]
        for camera in CAMERAS
    }
    robot = capture["robot_snapshot"]
    names = robot["joint_names"]
    if len(names) != len(set(names)) or names[:7] != [f"panda_joint{i}" for i in range(1, 8)]:
        raise ValueError("native arm joint order differs")
    positions = np.asarray(robot["joint_position_rad"], dtype=np.float32)
    if positions.shape != (len(names),) or "finger_joint" not in names:
        raise ValueError("native joint inventory differs")
    # Exact pinned RoboLab droid.py proprioception: seven arm joints and
    # finger_joint / (pi/4), before the official policy's gripper inversion.
    proprio = {
        "arm_joint_pos": positions[None, :7],
        "gripper_pos": positions[None, [names.index("finger_joint")]] / np.float32(np.pi / 4),
    }
    return nano_observation({"image_obs": images, "proprio_obs": proprio})


def post(url: str, payload: dict[str, Any], *, timeout: float = 900) -> dict[str, Any]:
    data = json.dumps(payload, default=lambda item: np.asarray(item).tolist(), separators=(",", ":")).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def compare_actions(actions: list[np.ndarray]) -> dict[str, Any]:
    if len(actions) != 6 or any(a.shape != (32, 8) or not np.isfinite(a).all() for a in actions):
        raise ValueError("six finite native 32x8 outputs are required")
    return {
        "repeat_equal": [bool(np.array_equal(actions[i], actions[i + 1])) for i in (0, 3)],
        "opposite_prompt_distinct": [not bool(np.array_equal(actions[i], actions[i + 2])) for i in (0, 3)],
        "opposite_prompt_action_rms": [
            float(np.sqrt(np.mean((actions[i].astype(float) - actions[i + 2]) ** 2))) for i in (0, 3)
        ],
        "between_groups_equal": [bool(np.array_equal(actions[i], actions[i + 3])) for i in range(3)],
    }


def run(registration_path: Path, output: Path) -> dict[str, Any]:
    registration = load_registration(registration_path)
    output.mkdir(parents=True, exist_ok=False)
    _fsync_json(output / "intent.json", {
        "registration": record(registration_path), "maximum_model_requests": 6,
        "behavioral_episodes": 0, "executed_actions": 0, "release_permitted": False,
    })
    requests_started = 0
    server = None
    thread = None
    try:
        import importlib.metadata
        import torch
        from .nano_backend import build_pinned_nano_backend

        capture = json.loads(bound_file(registration["capture"]).read_bytes())
        observation = native_observation(capture)
        inputs = output / "inputs"
        inputs.mkdir()
        input_records = {}
        for key, value in observation.items():
            path = inputs / (key.replace("/", "-") + ".npy")
            save_array(path, value)
            input_records[key] = {**record(path), "shape": list(value.shape), "dtype": str(value.dtype)}
        fingerprint = hashlib.sha256(json.dumps(input_records, sort_keys=True).encode()).hexdigest()
        _fsync_json(output / "input-manifest.json", {
            "capture": registration["capture"], "native_proprio_source": registration["native_proprio_source"],
            "inputs": input_records, "fingerprint": fingerprint,
            "scope": "Retained native camera/proprio snapshot only; no object poses or scoring metadata sent to policy.",
            "physical_reset_performed_in_this_job": False,
        })
        prompts = {row["prompt_id"]: row for row in json.loads(bound_file(registration["prompts"]).read_bytes())["prompts"]}
        for prompt_id in set(ORDER):
            row = prompts[prompt_id]
            if hashlib.sha256(row["text"].encode()).hexdigest() != row["sha256"]:
                raise ValueError("frozen prompt hash differs")
        backend = build_pinned_nano_backend()
        _fsync_json(output / "loaded-runtime.json", {
            "config": asdict(backend.service.cfg), "source_root": backend.source_root,
            "checkpoint_path": backend.checkpoint_path,
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")},
            "cuda_device": torch.cuda.get_device_name(0),
        })
        trace = output / "trace.jsonl"
        producer = NanoEvidenceProducer(
            backend, trace_path=trace, future_dir=output / "futures",
            attestation_path=output / "server-attestation.json",
        )
        server = make_nano_http_server(producer, host="127.0.0.1", port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        endpoint = f"http://127.0.0.1:{server.server_port}"
        _fsync_json(output / "endpoint.json", {"url": endpoint, "pid": os.getpid(), "loopback_only": True})
        original_decode = backend.service.model.decode
        latent_receipts = []
        current_request = {}

        def retain_then_decode(latent, *args, **kwargs):
            index = current_request["index"]
            if len(latent_receipts) != index:
                raise ValueError("expected exactly one native decode per fixed-input request")
            path = output / f"request-{index:02d}" / "vision-latent.npy"
            if latent.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError("unsupported latent dtype for lossless retention")
            save_array(path, latent.detach().float().cpu().numpy())
            latent_receipts.append({
                **record(path), "native_dtype": str(latent.dtype), "shape": list(latent.shape),
                "retention": "Lossless float32 representation of native float32/float16/bfloat16 values.",
            })
            return original_decode(latent, *args, **kwargs)

        backend.service.model.decode = retain_then_decode
        actions = []
        results = []
        try:
            for index, prompt_id in enumerate(ORDER):
                request_root = output / f"request-{index:02d}"
                request_root.mkdir()
                current_request["index"] = index
                reset = post(endpoint + "/reset", {"camera_name": "over_shoulder_left_camera"}, timeout=30)
                if reset.get("status") != "reset":
                    raise ValueError("native HTTP wrapper reset was not acknowledged")
                _fsync_json(request_root / "reset.json", {
                    **reset, "physical_reset": "retained_fixed_observation_not_new_simulator_reset",
                    "native_cache_semantics": "Official history_length=1 builds a fresh one-frame sample per request.",
                })
                packet = {
                    "request_id": f"{registration['registration_id']}-request-{index:02d}",
                    "request_index": 0, "registered_cell_id": registration["registration_id"],
                    "reset_id": reset["reset_id"], "reset_fingerprint": fingerprint,
                    "camera_id": "over_shoulder_left_camera", "camera_name": "over_shoulder_left_camera",
                    "sampling_seed": registration["sampling_seed"], "prompt": prompts[prompt_id]["text"],
                    "observation": observation,
                }
                _fsync_json(request_root / "intent.json", {
                    **{k: v for k, v in packet.items() if k != "observation"},
                    "input_manifest": record(output / "input-manifest.json"),
                    "prompt_id": prompt_id, "native_decode_video": True,
                    "group": "native_required_decode" if index < 3 else "native_required_decode_plus_offline_redecode",
                    "model_requests_started_before_this": requests_started,
                })
                requests_started += 1
                started = time.monotonic()
                response = post(endpoint + "/predict", packet)
                action = np.asarray(response["action"], dtype=np.float32)
                if action.shape != (32, 8) or not np.isfinite(action).all():
                    raise ValueError("invalid native action output")
                save_array(request_root / "actions.npy", action)
                rows = [json.loads(line) for line in trace.read_text().splitlines()]
                row = rows[-1]
                if len(rows) != requests_started or row["request_id"] != packet["request_id"]:
                    raise ValueError("native request trace attribution differs")
                if row["actions_sha256"] != hashlib.sha256(action.tobytes()).hexdigest():
                    raise ValueError("returned action differs from server trace")
                future = np.load(row["future_path"], allow_pickle=False)
                if future.dtype != np.uint8 or future.ndim != 4 or future.shape[0] != 33 or future.shape[-1] != 3:
                    raise ValueError("native decoded future is not 33 RGB uint8 frames")
                result = {
                    "index": index, "request_id": row["request_id"], "prompt_id": prompt_id,
                    "seconds": time.monotonic() - started, "actions": record(request_root / "actions.npy"),
                    "future": {**record(Path(row["future_path"])), "shape": list(future.shape)},
                    "latent": latent_receipts[index], "trace": row, "executed_actions": 0,
                }
                _fsync_json(request_root / "result.json", result)
                actions.append(action)
                results.append(result)
        finally:
            backend.service.model.decode = original_decode

        redecodes = []
        for index in (3, 4, 5):
            latent_record = latent_receipts[index]
            latent = torch.from_numpy(np.load(bound_file(latent_record), allow_pickle=False)).to(
                device="cuda", dtype=getattr(torch, latent_record["native_dtype"].removeprefix("torch.")),
            )
            with torch.inference_mode():
                decoded = original_decode(latent)
                video = ((decoded[0].clamp(-1, 1) + 1) * 127.5).to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy()
            path = output / f"request-{index:02d}" / "offline-redecoded-future.npy"
            save_array(path, video)
            original = np.load(bound_file(results[index]["future"]), allow_pickle=False)
            redecodes.append({
                "index": index, "output": record(path), "native_decode_equal": bool(np.array_equal(video, original)),
                "new_model_requests": 0,
            })
        result = {
            "status": "six_fixed_input_requests_completed_not_behavioral_release",
            "registration": record(registration_path), "model_requests": 6,
            "behavioral_episodes": 0, "executed_actions": 0, "release_permitted": False,
            "comparisons": compare_actions(actions), "requests": results, "offline_redecodes": redecodes,
            "future_semantics": "Generated local predictions only; no executed ground truth or physical success score.",
            "decode_condition": "Native decode retained for all requests; extra decoding performed offline on retained latents, not by changing inference.",
        }
        _fsync_json(output / "result.json", result)
        return result
    except BaseException:
        _fsync_json(output / "failure.json", {
            "status": "fixed_input_runtime_failure_preserve_partial_no_automatic_retry",
            "model_requests_started": requests_started, "behavioral_episodes": 0,
            "executed_actions": 0, "traceback": traceback.format_exc(), "release_permitted": False,
        })
        raise
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.registration, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
