#!/usr/bin/env python3
"""Run the frozen Cosmos3-Edge base V2-A013 three-request probe."""

from __future__ import annotations

import argparse
import av
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch


LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
CONDITIONS = (("left", LEFT), ("left_exact_repeat", LEFT), ("right", RIGHT))
MODEL = "nvidia/Cosmos3-Edge"
SEED = "8300"
PNG_SHA256 = "2a431b0fa288890b3509b314c0351c91123d5f64b237678fed972848e29cd55b"
RGB_SHA256 = "6261ce5ab21383342c2012c14f7ff97d3dcd74e5f4202f2b3444355cc7ba3332"
EXTRA_PARAMS = {
    "action_chunk_size": 16,
    "action_mode": "policy",
    "domain_name": "droid_lerobot",
    "raw_action_dim": 10,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def request_fields(prompt: str) -> list[tuple[str, str]]:
    return [
        ("model", MODEL),
        ("prompt", prompt),
        ("size", "640x480"),
        ("num_frames", "17"),
        ("fps", "5"),
        ("num_inference_steps", "30"),
        ("guidance_scale", "1.0"),
        ("flow_shift", "5.0"),
        ("extra_params", json.dumps(EXTRA_PARAMS, separators=(",", ":"), sort_keys=True)),
        ("seed", SEED),
    ]


def inspect_video(path: Path) -> dict[str, Any]:
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1:
            raise ValueError(f"{path}: expected one video stream")
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        return {
            "codec_name": stream.codec_context.name,
            "width": stream.width,
            "height": stream.height,
            "frame_count": len(frames),
            "average_rate": str(stream.average_rate),
        }


def action_array(payload: dict[str, Any], condition: str) -> tuple[np.ndarray, dict[str, Any]]:
    action = payload.get("action")
    if not isinstance(action, dict):
        raise ValueError(f"{condition}: missing top-level action object")
    shape = action.get("shape")
    dtype_name = action.get("dtype")
    if shape != [16, 10]:
        raise ValueError(f"{condition}: action shape is {shape}, expected [16, 10]")
    if action.get("raw_action_dim") != 10:
        raise ValueError(f"{condition}: raw_action_dim is not 10")
    if action.get("domain_id") != 8:
        raise ValueError(f"{condition}: domain_id is not 8")
    if not isinstance(dtype_name, str):
        raise ValueError(f"{condition}: missing action dtype")
    if dtype_name == "torch.bfloat16":
        tensor = torch.tensor(action.get("data"), dtype=torch.bfloat16)
        if tuple(tensor.shape) != (16, 10):
            raise ValueError(f"{condition}: action data shape is {tuple(tensor.shape)}")
        if not torch.isfinite(tensor.float()).all():
            raise ValueError(f"{condition}: action data contains non-finite values")
        array = tensor.view(torch.uint16).cpu().numpy().copy()
        storage_dtype = "uint16_bfloat16_bits"
    else:
        try:
            dtype = np.dtype(dtype_name)
        except TypeError as exc:
            raise ValueError(f"{condition}: unsupported action dtype {dtype_name!r}") from exc
        array = np.asarray(action.get("data"), dtype=dtype)
        if array.shape != (16, 10):
            raise ValueError(f"{condition}: action data shape is {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{condition}: action data contains non-finite values")
        storage_dtype = str(array.dtype)
    metadata = {
        "shape": shape,
        "dtype": dtype_name,
        "raw_action_dim": action["raw_action_dim"],
        "domain_id": action["domain_id"],
        "canonical_storage_dtype": storage_dtype,
    }
    return np.ascontiguousarray(array), metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18013")
    parser.add_argument("--conditioning-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--resume-existing-left", action="store_true")
    args = parser.parse_args()

    image_bytes = args.conditioning_image.read_bytes()
    if sha256_bytes(image_bytes) != PNG_SHA256:
        raise ValueError("conditioning PNG does not match the frozen V2-A013 bytes")
    args.output_dir.mkdir(parents=True, exist_ok=args.resume_existing_left)
    if args.resume_existing_left and not (args.output_dir / "left" / "final_response.json").is_file():
        raise ValueError("--resume-existing-left requires the completed LEFT response")

    session = requests.Session()
    health = session.get(f"{args.base_url}/health", timeout=30)
    health.raise_for_status()
    models = session.get(f"{args.base_url}/v1/models", timeout=30)
    models.raise_for_status()
    models_payload = models.json()
    model_rows = models_payload.get("data", [])
    if len(model_rows) != 1 or model_rows[0].get("id") != MODEL:
        raise ValueError(f"server model identity mismatch: {models_payload}")
    write_json(args.output_dir / "server_models.json", models_payload)

    records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    video_hashes: dict[str, str] = {}
    requests_issued_this_invocation = 0
    for condition, prompt in CONDITIONS:
        condition_dir = args.output_dir / condition
        resumed_left = args.resume_existing_left and condition == "left"
        condition_dir.mkdir(exist_ok=resumed_left)
        fields = request_fields(prompt)
        provenance = {
            "condition": condition,
            "endpoint": "POST /v1/videos",
            "fields": fields,
            "input_filename": "conditioning.png",
            "input_content_type": "image/png",
            "conditioning_png_sha256": PNG_SHA256,
            "conditioning_rgb_sha256": RGB_SHA256,
            "requested_sampling_seed": 8300,
        }
        provenance_bytes = json.dumps(
            {"fields": fields, "input_content_type": "image/png", "input_sha256": PNG_SHA256},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        provenance["multipart_fields_sha256"] = sha256_bytes(provenance_bytes)
        if resumed_left:
            existing_provenance = json.loads(
                (condition_dir / "request_provenance.json").read_text()
            )
            if existing_provenance.get("multipart_fields_sha256") != provenance["multipart_fields_sha256"]:
                raise ValueError("existing LEFT request provenance does not match the frozen fields")
        else:
            write_json(condition_dir / "request_provenance.json", provenance)

        started = time.time()
        if resumed_left:
            create_payload = json.loads((condition_dir / "create_response.json").read_text())
            final_payload = json.loads((condition_dir / "final_response.json").read_text())
            poll_trace = json.loads((condition_dir / "poll_trace.json").read_text())
            job_id = create_payload.get("id")
            if final_payload.get("id") != job_id or final_payload.get("status") != "completed":
                raise ValueError("existing LEFT job identity/status is invalid")
        else:
            requests_issued_this_invocation += 1
            response = session.post(
                f"{args.base_url}/v1/videos",
                headers={"Accept": "application/json"},
                data=fields,
                files={"input_reference": ("conditioning.png", image_bytes, "image/png")},
                timeout=120,
            )
            (condition_dir / "create_response_body.bin").write_bytes(response.content)
            if response.status_code >= 400:
                raise RuntimeError(f"{condition}: POST returned HTTP {response.status_code}")
            create_payload = response.json()
            write_json(condition_dir / "create_response.json", create_payload)
            job_id = create_payload.get("id")
            if not isinstance(job_id, str) or not job_id:
                raise ValueError(f"{condition}: POST response lacks a job id")

            poll_trace = []
            deadline = time.monotonic() + args.poll_timeout_seconds
            final_payload = None
            while time.monotonic() < deadline:
                poll_response = session.get(f"{args.base_url}/v1/videos/{job_id}", timeout=60)
                poll_response.raise_for_status()
                payload = poll_response.json()
                poll_trace.append(
                    {
                        "observed_unix_seconds": time.time(),
                        "status": payload.get("status"),
                        "error": payload.get("error"),
                    }
                )
                status = payload.get("status")
                if status == "completed":
                    final_payload = payload
                    break
                if status in {"failed", "cancelled"}:
                    write_json(condition_dir / "poll_trace.json", poll_trace)
                    write_json(condition_dir / "final_response.json", payload)
                    raise RuntimeError(f"{condition}: job ended with status {status}")
                time.sleep(args.poll_seconds)
            if final_payload is None:
                write_json(condition_dir / "poll_trace.json", poll_trace)
                raise TimeoutError(f"{condition}: job did not complete before the frozen timeout")
            write_json(condition_dir / "poll_trace.json", poll_trace)
            write_json(condition_dir / "final_response.json", final_payload)

        array, action_metadata = action_array(final_payload, condition)
        action_path = condition_dir / "action_storage.npy"
        np.save(action_path, array, allow_pickle=False)
        action_bytes_sha256 = sha256_bytes(array.tobytes(order="C"))

        video_path = condition_dir / "model_prediction.mp4"
        if not video_path.is_file():
            content_response = session.get(
                f"{args.base_url}/v1/videos/{job_id}/content", timeout=300
            )
            content_response.raise_for_status()
            video_path.write_bytes(content_response.content)
        video_sha256 = sha256_file(video_path)
        stream = inspect_video(video_path)
        if stream["frame_count"] != 17:
            raise ValueError(f"{condition}: expected 17 video frames, got {stream['frame_count']}")
        if (stream.get("width"), stream.get("height")) != (640, 480):
            raise ValueError(f"{condition}: unexpected video dimensions: {stream}")

        arrays[condition] = array
        video_hashes[condition] = video_sha256
        records.append(
            {
                "condition": condition,
                "prompt": prompt,
                "job_id": job_id,
                "requested_sampling_seed": 8300,
                "wall_seconds_this_invocation": time.time() - started,
                "recovered_existing_completed_job": resumed_left,
                "action": action_metadata,
                "action_values_sha256": action_bytes_sha256,
                "action_npy_sha256": sha256_file(action_path),
                "future_mp4_sha256": video_sha256,
                "future_mp4_bytes": video_path.stat().st_size,
                "future_stream": stream,
                "multipart_fields_sha256": provenance["multipart_fields_sha256"],
            }
        )

    left_repeat_action_equal = np.array_equal(arrays["left"], arrays["left_exact_repeat"])
    left_right_action_equal = np.array_equal(arrays["left"], arrays["right"])
    left_repeat_video_equal = video_hashes["left"] == video_hashes["left_exact_repeat"]
    left_right_video_equal = video_hashes["left"] == video_hashes["right"]
    repeat_fields_equal = (
        records[0]["multipart_fields_sha256"] == records[1]["multipart_fields_sha256"]
    )
    checks = {
        "authorized_request_count_exact": len(records) == 3,
        "left_repeat_multipart_fields_identical": repeat_fields_equal,
        "left_repeat_action_bit_identical": left_repeat_action_equal,
        "left_right_action_different": not left_right_action_equal,
        "left_repeat_video_bit_identical": left_repeat_video_equal,
        "left_right_video_different": not left_right_video_equal,
        "all_wall_times_finite": all(
            math.isfinite(row["wall_seconds_this_invocation"]) for row in records
        ),
    }
    passed = all(checks.values())
    manifest = {
        "schema_version": "vla-wam-shared-v2-cosmos3-edge-base-v2a013-fixed-observation-v1",
        "status": "passed" if passed else "failed",
        "model": MODEL,
        "sampling_seed": 8300,
        "checkpoint_revision": "ff48d22144de52de296a7b4d3a78914831007212",
        "conditioning_png_sha256": PNG_SHA256,
        "conditioning_rgb_sha256": RGB_SHA256,
        "authorized_request_count": 3,
        "requests_issued_this_invocation": requests_issued_this_invocation,
        "left_recovered_from_existing_completed_job": args.resume_existing_left,
        "behavioral_episode_count": 0,
        "records": records,
        "checks": checks,
        "claim_boundary": (
            "Fixed-observation action-plus-exposed-future interface feasibility only; "
            "no simulator action was sent and no behavioral denominator is released."
        ),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(20)


if __name__ == "__main__":
    main()
