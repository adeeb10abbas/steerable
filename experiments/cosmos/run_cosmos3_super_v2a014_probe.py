#!/usr/bin/env python3
"""Run the frozen V2-A014 Cosmos3-Super image-only interface probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import av
import torch


MODEL = "nvidia/Cosmos3-Super"
IMAGE_SHA256 = "2a431b0fa288890b3509b314c0351c91123d5f64b237678fed972848e29cd55b"
LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
REQUESTS = (
    ("V2-A014-P00", "LEFT", LEFT),
    ("V2-A014-P01", "LEFT_exact_repeat", LEFT),
    ("V2-A014-P02", "RIGHT", RIGHT),
)
EXTRA_PARAMS = {
    "action_mode": "policy",
    "domain_name": "droid_lerobot",
    "raw_action_dim": 10,
    "action_chunk_size": 16,
}
BOUNDARY = "----vla-wam-v2-a014-frozen-boundary"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def multipart_body(prompt: str, image: bytes) -> bytes:
    chunks: list[bytes] = []

    def add_text(name: str, value: str) -> None:
        chunks.extend(
            [
                f"--{BOUNDARY}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"),
                b"\r\n",
            ]
        )

    add_text("model", MODEL)
    add_text("prompt", prompt)
    chunks.extend(
        [
            f"--{BOUNDARY}\r\n".encode(),
            b'Content-Disposition: form-data; name="input_reference"; filename="conditioning.png"\r\n',
            b"Content-Type: image/png\r\n\r\n",
            image,
            b"\r\n",
        ]
    )
    add_text("size", "640x480")
    add_text("num_frames", "17")
    add_text("fps", "5")
    add_text("num_inference_steps", "30")
    add_text("guidance_scale", "1.0")
    add_text("flow_shift", "5.0")
    add_text("extra_params", canonical_json(EXTRA_PARAMS).decode("utf-8"))
    add_text("seed", "8300")
    chunks.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(chunks)


def request_bytes(url: str, *, data: bytes | None = None, content_type: str | None = None) -> bytes:
    headers = {"Accept": "application/json"}
    if content_type is not None:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        body = error.read()
        raise RuntimeError(f"HTTP {error.code} for {url}: {body[:2000]!r}") from error


def request_json(url: str, *, data: bytes | None = None, content_type: str | None = None) -> dict[str, Any]:
    body = request_bytes(url, data=data, content_type=content_type)
    try:
        result = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Expected JSON from {url}, received {body[:2000]!r}") from error
    if not isinstance(result, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(result)!r}")
    return result


def decode_action(action: dict[str, Any], request_id: str) -> tuple[np.ndarray, bool, str]:
    reported_shape = action.get("shape")
    reported_dtype = action.get("dtype")
    if reported_shape != [16, 10]:
        raise RuntimeError(
            f"{request_id}: reported action shape {reported_shape!r}, expected [16, 10]"
        )
    if not isinstance(reported_dtype, str):
        raise RuntimeError(f"{request_id}: action payload has no reported dtype")
    if reported_dtype == "torch.bfloat16":
        tensor = torch.tensor(action.get("data"), dtype=torch.bfloat16)
        finite = bool(torch.isfinite(tensor.float()).all())
        array = tensor.view(torch.uint16).cpu().numpy().copy()
        storage_dtype = "uint16_bfloat16_bits"
    else:
        try:
            dtype = np.dtype(reported_dtype)
        except TypeError as error:
            raise RuntimeError(
                f"{request_id}: unsupported action dtype {reported_dtype!r}"
            ) from error
        array = np.asarray(action.get("data"), dtype=dtype)
        finite = bool(np.all(np.isfinite(array)))
        storage_dtype = str(array.dtype)
    if array.shape != (16, 10):
        raise RuntimeError(f"{request_id}: action data shape is {array.shape}, expected (16, 10)")
    return np.ascontiguousarray(array), finite, storage_dtype


def video_probe(path: Path) -> dict[str, Any]:
    decoded = hashlib.sha256()
    decoded_bytes = 0
    frame_count = 0
    with av.open(str(path)) as container:
        video_streams = container.streams.video
        if len(video_streams) != 1:
            raise RuntimeError(
                f"Expected one video stream in {path}, found {len(video_streams)}"
            )
        video_stream = video_streams[0]
        stream = {
            "codec_name": video_stream.codec_context.name,
            "width": video_stream.width,
            "height": video_stream.height,
            "pix_fmt": video_stream.codec_context.pix_fmt,
            "avg_frame_rate": str(video_stream.average_rate),
            "nb_frames": video_stream.frames,
        }
        for frame in container.decode(video=0):
            rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            decoded.update(rgb.tobytes())
            decoded_bytes += rgb.nbytes
            frame_count += 1
    return {
        "container_bytes": path.stat().st_size,
        "container_sha256": sha256_file(path),
        "stream": stream,
        "decoded_frame_count": frame_count,
        "decoded_rgb_bytes": decoded_bytes,
        "decoded_rgb_sha256": decoded.hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18014")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args()

    image = args.image.read_bytes()
    if sha256_bytes(image) != IMAGE_SHA256:
        raise ValueError("Conditioning image does not match the frozen transport hash")
    models = request_json(f"{args.base_url}/v1/models")
    model_ids = [row.get("id") for row in models.get("data", []) if isinstance(row, dict)]
    if MODEL not in model_ids:
        raise RuntimeError(f"Frozen served model name {MODEL!r} not present: {model_ids!r}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for request_id, condition, prompt in REQUESTS:
        request_dir = args.output_dir / request_id
        request_dir.mkdir()
        body = multipart_body(prompt, image)
        body_path = request_dir / "request.multipart"
        body_path.write_bytes(body)
        request_contract = {
            "request_id": request_id,
            "condition": condition,
            "endpoint": "POST /v1/videos",
            "model": MODEL,
            "prompt": prompt,
            "input_reference_sha256": IMAGE_SHA256,
            "size": "640x480",
            "num_frames": 17,
            "fps": 5,
            "num_inference_steps": 30,
            "guidance_scale": 1.0,
            "flow_shift": 5.0,
            "seed": 8300,
            "extra_params": EXTRA_PARAMS,
            "multipart_content_type": f"multipart/form-data; boundary={BOUNDARY}",
            "multipart_body_bytes": len(body),
            "multipart_body_sha256": sha256_bytes(body),
        }
        (request_dir / "request.json").write_text(
            json.dumps(request_contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        started = time.monotonic()
        create = request_json(
            f"{args.base_url}/v1/videos",
            data=body,
            content_type=f"multipart/form-data; boundary={BOUNDARY}",
        )
        (request_dir / "create_response.json").write_text(
            json.dumps(create, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        server_id = create.get("id")
        if not isinstance(server_id, str) or not server_id:
            raise RuntimeError(f"{request_id}: create response has no id: {create!r}")
        poll_count = 0
        while True:
            final = request_json(f"{args.base_url}/v1/videos/{server_id}")
            (request_dir / f"poll_{poll_count:04d}.json").write_text(
                json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            status = str(final.get("status", "")).lower()
            if status in {"completed", "failed", "cancelled"}:
                break
            if time.monotonic() - started > args.poll_timeout_seconds:
                raise TimeoutError(f"{request_id}: polling timed out with status {status!r}")
            poll_count += 1
            time.sleep(args.poll_seconds)
        final_path = request_dir / "final_response.json"
        final_path.write_text(json.dumps(final, indent=2, sort_keys=True, allow_nan=False) + "\n")
        if status != "completed":
            raise RuntimeError(f"{request_id}: final status {status!r}: {final!r}")

        action = final.get("action")
        if not isinstance(action, dict):
            raise RuntimeError(f"{request_id}: completed response has no action object")
        action_storage, finite, action_storage_dtype = decode_action(action, request_id)
        action_shape = list(action_storage.shape)
        action_path = request_dir / "action.npy"
        np.save(action_path, action_storage, allow_pickle=False)
        action_metadata_path = request_dir / "action.json"
        action_metadata_path.write_text(
            json.dumps(action, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        video_path = request_dir / "model_generated.mp4"
        video_path.write_bytes(request_bytes(f"{args.base_url}/v1/videos/{server_id}/content"))
        video = video_probe(video_path)
        frame_count = video["decoded_frame_count"]
        records.append(
            {
                "request_id": request_id,
                "condition": condition,
                "prompt": prompt,
                "server_id": server_id,
                "status": status,
                "wall_seconds": time.monotonic() - started,
                "poll_count": poll_count + 1,
                "request_body_bytes": len(body),
                "request_body_sha256": sha256_bytes(body),
                "create_response_sha256": sha256_file(request_dir / "create_response.json"),
                "final_response_sha256": sha256_file(final_path),
                "action_shape": action_shape,
                "action_finite": finite,
                "action_dtype_reported": action.get("dtype"),
                "action_storage_dtype": action_storage_dtype,
                "action_raw_dim_reported": action.get("raw_action_dim"),
                "action_mode_reported": action.get("action_mode"),
                "action_domain_id_reported": action.get("domain_id"),
                "action_npy_sha256": sha256_file(action_path),
                "action_data_sha256": sha256_bytes(action_storage.tobytes(order="C")),
                "action_payload_sha256": sha256_bytes(canonical_json(action)),
                "action_json_sha256": sha256_file(action_metadata_path),
                "video": video,
                "video_frame_count": frame_count,
            }
        )

    left, repeat, right = records
    checks = {
        "request_count_exactly_three": len(records) == 3,
        "left_repeat_request_body_identical": left["request_body_sha256"] == repeat["request_body_sha256"],
        "left_repeat_action_identical": left["action_data_sha256"] == repeat["action_data_sha256"],
        "left_repeat_action_payload_identical": left["action_payload_sha256"]
        == repeat["action_payload_sha256"],
        "left_repeat_decoded_video_identical": left["video"]["decoded_rgb_sha256"]
        == repeat["video"]["decoded_rgb_sha256"],
        "all_actions_finite_16x10": all(
            row["action_finite"] and row["action_shape"] == [16, 10] for row in records
        ),
        "all_videos_decode_to_17_frames": all(row["video_frame_count"] == 17 for row in records),
        "left_right_actions_differ": left["action_data_sha256"] != right["action_data_sha256"],
        "left_right_decoded_videos_differ": left["video"]["decoded_rgb_sha256"]
        != right["video"]["decoded_rgb_sha256"],
    }
    manifest = {
        "schema_version": "vla-wam-shared-v2-cosmos3-super-image-only-v2a014-raw-result-v1",
        "amendment_id": "V2-A014",
        "status": "passed" if all(checks.values()) else "failed",
        "model": MODEL,
        "model_revision": "e0262be9d8f7586bc24c069a2aed2b665bdff266",
        "checkpoint_path": "/data/users/ali/vla_wam/checkpoints/cosmos3_super_base",
        "conditioning_png_sha256": IMAGE_SHA256,
        "robot_state_present": False,
        "simulator_or_controller_invoked": False,
        "records": records,
        "checks": checks,
        "claim_boundary": "Image-only action-and-video interface evidence; not DROID policy behavior, execution, or success.",
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    if not all(checks.values()):
        raise SystemExit(20)


if __name__ == "__main__":
    main()
