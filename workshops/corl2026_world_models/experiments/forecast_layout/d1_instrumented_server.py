#!/usr/bin/env python3
"""Fail-closed official DreamZero D1 server with measurement-only retention.

This overlay deliberately calls the released ``lazy_joint_forward_causal`` path
from DreamZero commit ab790c1.  It does not implement the historical custom s2
action-guidance patch.  Its only runtime changes are bounded model construction,
evidence retention, optional *post-inference* VAE decoding, and an explicit
two-rank episode reset handshake.

The module is importable without the external DreamZero environment so that its
identity, reset, and artifact contracts can be unit tested on the workstation.
The heavy imports are performed only by :func:`main`.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
import pickle
import re
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.distributed as dist


LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "wmf-d1-instrumented-server-v1"
REQUEST_SCHEMA_VERSION = "wmf-d1-request-receipt-v1"
EPISODE_SCHEMA_VERSION = "wmf-d1-episode-manifest-v1"
RESET_SCHEMA_VERSION = "wmf-d1-two-rank-reset-receipt-v1"
TERMINAL_SCHEMA_VERSION = "wmf-d1-terminal-context-receipt-v1"

OFFICIAL_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
OFFICIAL_TREE = "6b7ba27f1af81e963a6507f1204c05c65a94098c"
OFFICIAL_SERVER_SHA256 = "7ef17f66064bac8defafc1a84551089b124546729a98be8c0515b33d2e159d48"
OFFICIAL_ACTION_HEAD_SHA256 = "7193cd73423472aa252bee73bd80e0d673c89d773ec852e90f50154729b50845"
BOUNDED_LOADER_SHA256 = "6edcc1b0b237cfacdca73b3ccb3d1251445192f0fe35639a2d450f4262234905"

OFFICIAL_NOISE_SEED = 1140
EXPECTED_WORLD_SIZE = 2
EXPECTED_VISIBLE_GPU_COUNT = 2
EXPECTED_GPU_NAME_SUBSTRING = "B200"
EXPECTED_ACTION_SHAPE = (24, 8)
EXECUTED_ACTION_PREFIX = 8
FULL_ACTION_CAP = 450
FULL_REQUEST_COUNT = 57
VIDEO_GUIDANCE_SCALE = 5.0
CONFIGURED_INFERENCE_STEPS = 16
EXPECTED_DIT_STEP_MASK = [
    True,
    True,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    False,
    True,
    False,
    False,
    True,
    True,
    True,
]

LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
FROZEN_PROMPTS = {LEFT, RIGHT}

RAW_ARRAY_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/cartesian_position",
    "observation/gripper_position",
)
IMAGE_KEYS = RAW_ARRAY_KEYS[:3]
STATE_KEYS = RAW_ARRAY_KEYS[3:]
MEASUREMENT_KEY = "wmf_d1_measurement"
RESET_KEY = "wmf_d1_reset"
RESET_SIGNAL = 3
SHUTDOWN_SIGNAL = 1
INFER_SIGNAL = 0

RESET_FIELDS_TO_NONE = (
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "clip_feas",
    "ys",
    "language",
)
CACHE_FIELDS = RESET_FIELDS_TO_NONE[:4]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
FINALIZE_CONTROL_KEYS = {
    "finalize_only",
    "purpose",
    "previous_episode_id",
    "previous_session_id",
    "model_config",
    "study_id",
    "phase",
    "block_id",
    "layout_pair_id",
    "cell_id",
    "condition_index",
    "stop_reason",
    "actions_executed",
    "request_count",
    "server_ready_sha256",
    "simulator_claim_sha256",
    "simulator_lease_token",
    "pilot_contract_sha256",
}
BEHAVIORAL_PURPOSE_BY_PHASE = {
    "pilot": "d1_behavioral_pilot",
    "development": "d1_behavioral_development",
    "confirmation": "d1_behavioral_confirmation",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: Any) -> dt.datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    parsed = dt.datetime.fromisoformat(value[:-1] + "+00:00")
    if parsed.tzinfo is None or parsed.utcoffset() != dt.timedelta(0):
        raise ValueError(f"Invalid UTC timestamp: {value!r}")
    return parsed


def sha256_file(path: Path, *, block_bytes: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Evidence file is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def immutable_write_json(path: Path, value: Any) -> None:
    """Atomically create a JSON receipt without an overwrite path."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(
                (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
                    "utf-8"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_numpy(path: Path, value: np.ndarray) -> None:
    with Path(path).open("xb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(Path(path).parent)


def save_tensor(path: Path, value: torch.Tensor) -> None:
    with Path(path).open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(Path(path).parent)


def array_data_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return sha256_bytes(array.tobytes(order="C"))


def tensor_data_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return sha256_bytes(b"")
    byte_view = tensor.view(torch.uint8).numpy()
    return sha256_bytes(byte_view.tobytes(order="C"))


def _content_identity(entry: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("key", "kind", "shape", "dtype", "data_sha256", "json_sha256")
    return {field: entry[field] for field in fields if field in entry}


def _safe_copy(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, np.ndarray):
        return np.ascontiguousarray(value).copy()
    return copy.deepcopy(value)


def clone_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _safe_copy(value) for key, value in values.items()}


def save_exact_mapping(
    values: Mapping[str, Any],
    directory: Path,
    *,
    prefix: str,
) -> dict[str, Any]:
    """Retain exact arrays/tensors and hash a path-independent content identity."""

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(values)):
        value = values[key]
        stem = f"{prefix}_{index:03d}"
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            path = directory / f"{stem}.pt"
            save_tensor(path, tensor)
            entry = {
                "key": key,
                "kind": "torch_tensor",
                "path": str(path),
                "file_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "data_sha256": tensor_data_sha256(tensor),
            }
        elif isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            path = directory / f"{stem}.npy"
            save_numpy(path, array)
            entry = {
                "key": key,
                "kind": "numpy_array",
                "path": str(path),
                "file_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "data_sha256": array_data_sha256(array),
            }
        elif value is None or isinstance(value, (str, int, float, bool, list, dict)):
            try:
                encoded = canonical_json_bytes(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"Input {key!r} is not exactly JSON serializable") from exc
            path = directory / f"{stem}.json"
            atomic_write_json(path, value)
            entry = {
                "key": key,
                "kind": "json_value",
                "path": str(path),
                "file_sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "json_sha256": sha256_bytes(encoded),
            }
        else:
            raise TypeError(
                f"Refusing to summarize unsupported exact input {key!r}: {type(value)!r}"
            )
        records.append(entry)

    identities = [_content_identity(record) for record in records]
    return {
        "entries": records,
        "entry_count": len(records),
        "content_sha256": sha256_bytes(canonical_json_bytes(identities)),
        "content_hash_definition": (
            "SHA-256 of canonical JSON over sorted keys, value kinds, shapes, dtypes, "
            "and exact tensor/array data hashes; artifact paths are excluded"
        ),
    }


def validate_raw_request(obs: Mapping[str, Any]) -> None:
    missing = set(RAW_ARRAY_KEYS) - set(obs)
    if missing:
        raise ValueError(f"D1 request is missing exact arrays: {sorted(missing)}")
    for key in RAW_ARRAY_KEYS:
        if not isinstance(obs[key], np.ndarray):
            raise TypeError(f"D1 wire input {key} must be a numpy array")
        if not np.isfinite(obs[key]).all():
            raise ValueError(f"D1 wire input {key} contains non-finite values")
    for key in IMAGE_KEYS:
        value = obs[key]
        if value.shape != (180, 320, 3) or value.dtype != np.uint8:
            raise ValueError(f"D1 image contract changed for {key}: {value.shape}/{value.dtype}")
    for key in STATE_KEYS:
        value = obs[key]
        expected_shape = {
            "observation/joint_position": (7,),
            "observation/cartesian_position": (6,),
            "observation/gripper_position": (1,),
        }[key]
        if value.shape != expected_shape or value.dtype != np.float64:
            raise ValueError(f"D1 state contract changed for {key}: {value.shape}/{value.dtype}")
    if obs.get("prompt") not in FROZEN_PROMPTS:
        raise ValueError(f"D1 prompt is outside the frozen direct commands: {obs.get('prompt')!r}")
    if not isinstance(obs.get("session_id"), str) or not obs["session_id"]:
        raise ValueError("D1 request requires a nonempty string session_id")


def _value_signature(value: Any, *, depth: int = 0) -> dict[str, Any]:
    if value is None:
        return {"is_none": True}
    signature: dict[str, Any] = {
        "is_none": False,
        "python_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "object_id": id(value),
    }
    if torch.is_tensor(value):
        signature.update(
            {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "numel": int(value.numel()),
                "data_ptr": int(value.data_ptr()),
            }
        )
    elif isinstance(value, (list, tuple)):
        signature["length"] = len(value)
        if depth < 1:
            signature["elements"] = [
                _value_signature(item, depth=depth + 1) for item in value
            ]
    return signature


def action_head(policy: Any) -> Any:
    try:
        return policy.trained_model.action_head
    except AttributeError as exc:
        raise RuntimeError("DreamZero policy has no trained_model.action_head") from exc


def temporal_snapshot(policy: Any, *, rank: int) -> dict[str, Any]:
    head = action_head(policy)
    fields = {
        field: _value_signature(getattr(head, field))
        for field in RESET_FIELDS_TO_NONE
        if hasattr(head, field)
    }
    return {
        "rank": rank,
        "captured_at_utc": utc_now(),
        "current_start_frame": getattr(head, "current_start_frame", None),
        "fields": fields,
        "fixed_seed": getattr(head, "seed", None),
        "video_guidance_scale": getattr(head, "cfg_scale", None),
        "configured_inference_steps": getattr(head, "num_inference_steps", None),
        "num_frame_per_block": getattr(head, "num_frame_per_block", None),
        "action_horizon": getattr(head, "action_horizon", None),
        "ip_rank": getattr(head, "ip_rank", None),
        "ip_size": getattr(head, "ip_size", None),
    }


def pinned_frame_block_contract(
    checkpoint_root: Path,
    identity_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Read the frame-block size from the already identity-verified checkpoint.

    The official action-head source copies ``config.num_frame_per_block`` onto
    the runtime head.  The pinned DROID checkpoint carries that value both on
    the action-head config and its nested diffusion-model config.  Binding the
    gate to those exact config bytes avoids imposing a configuration from a
    different DreamZero backbone.
    """

    if identity_receipt.get("status") != "passed":
        raise ValueError("D1 frame-block authority requires a passed identity receipt")

    source = identity_receipt.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("D1 identity receipt is missing source evidence")
    required_files = source.get("required_files")
    if not isinstance(required_files, list):
        raise ValueError("D1 identity receipt is missing required source files")
    action_head_records = [
        record
        for record in required_files
        if isinstance(record, Mapping)
        and record.get("path")
        == "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py"
    ]
    if len(action_head_records) != 1:
        raise ValueError("D1 identity receipt does not identify the official action-head source")
    action_head_record = action_head_records[0]
    if action_head_record.get("sha256") != OFFICIAL_ACTION_HEAD_SHA256:
        raise ValueError("D1 official action-head source identity changed")

    checkpoint = identity_receipt.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("D1 identity receipt is missing checkpoint evidence")
    root = Path(checkpoint_root).resolve()
    if root != Path(str(checkpoint.get("path", ""))).resolve():
        raise ValueError("D1 frame-block checkpoint path changed")
    files = checkpoint.get("files")
    if not isinstance(files, list):
        raise ValueError("D1 identity receipt is missing checkpoint file evidence")
    config_records = [
        record
        for record in files
        if isinstance(record, Mapping) and record.get("path") == "config.json"
    ]
    if len(config_records) != 1:
        raise ValueError("D1 identity receipt does not identify checkpoint config.json")
    config_record = config_records[0]
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("D1 checkpoint config.json is missing")
    if config_path.stat().st_size != config_record.get("bytes"):
        raise ValueError("D1 checkpoint config.json byte count changed")
    config_sha256 = sha256_file(config_path)
    if config_sha256 != config_record.get("sha256"):
        raise ValueError("D1 checkpoint config.json identity changed")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        action_head_config = config["action_head_cfg"]["config"]
        action_head_value = action_head_config["num_frame_per_block"]
        diffusion_value = action_head_config["diffusion_model_cfg"]["num_frame_per_block"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("D1 checkpoint frame-block config is unreadable") from error
    values = (action_head_value, diffusion_value)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("D1 checkpoint frame-block config is not a positive integer")
    if action_head_value != diffusion_value:
        raise ValueError("D1 checkpoint action-head and diffusion frame-block configs disagree")

    return {
        "schema_version": "wmf-d1-frame-block-contract-v1",
        "status": "passed",
        "num_frame_per_block": action_head_value,
        "checkpoint_config": {
            "path": str(config_path),
            "bytes": config_path.stat().st_size,
            "sha256": config_sha256,
            "fields": {
                "action_head_cfg.config.num_frame_per_block": action_head_value,
                "action_head_cfg.config.diffusion_model_cfg.num_frame_per_block": diffusion_value,
            },
        },
        "official_action_head_source": dict(action_head_record),
    }


def validate_official_head(
    policy: Any,
    *,
    rank: int,
    frame_block_contract: Mapping[str, Any],
) -> dict[str, Any]:
    head = action_head(policy)
    failures: list[str] = []
    if frame_block_contract.get("status") != "passed":
        raise ValueError("D1 frame-block contract did not pass")
    expected_num_frame_per_block = frame_block_contract.get("num_frame_per_block")
    if type(expected_num_frame_per_block) is not int or expected_num_frame_per_block <= 0:
        raise ValueError("D1 frame-block contract has an invalid expected value")
    expected = {
        "seed": OFFICIAL_NOISE_SEED,
        "cfg_scale": VIDEO_GUIDANCE_SCALE,
        "num_inference_steps": CONFIGURED_INFERENCE_STEPS,
        "action_horizon": EXPECTED_ACTION_SHAPE[0],
        "ip_rank": rank,
        "ip_size": EXPECTED_WORLD_SIZE,
        "num_frame_per_block": expected_num_frame_per_block,
        "dynamic_cache_schedule": False,
    }
    observed = {name: getattr(head, name, None) for name in expected}
    for name, expected_value in expected.items():
        if observed[name] != expected_value:
            failures.append(f"{name}={observed[name]!r}, expected {expected_value!r}")
    if hasattr(head, "action_cfg_scale"):
        failures.append("official D1 action head unexpectedly exposes custom action_cfg_scale")
    mask = list(getattr(head, "dit_step_mask", []))
    if mask != EXPECTED_DIT_STEP_MASK:
        failures.append(f"dit_step_mask changed: {mask}")
    if getattr(head, "trt_engine", None) is not None:
        failures.append("LOAD_TRT_ENGINE unexpectedly changed the official D1 path")
    receipt = {
        "rank": rank,
        "status": "failed" if failures else "passed",
        "official_conditional_method": "GrootSimPolicy.lazy_joint_forward_causal",
        "custom_s2_present": hasattr(head, "action_cfg_scale"),
        "observed": observed,
        "dit_step_mask": mask,
        "evaluated_dit_step_count": int(sum(bool(value) for value in mask)),
        "tensorrt_engine_active": getattr(head, "trt_engine", None) is not None,
        "frame_block_contract": dict(frame_block_contract),
        "failures": failures,
    }
    if failures:
        raise RuntimeError("Official D1 head contract failed: " + "; ".join(failures))
    return receipt


def reset_temporal_state(policy: Any, *, rank: int, reset_id: str) -> dict[str, Any]:
    """Clear every persistent official action-head episode field and prove it."""

    head = action_head(policy)
    required = {"current_start_frame", *RESET_FIELDS_TO_NONE}
    missing = sorted(name for name in required if not hasattr(head, name))
    if missing:
        raise RuntimeError(f"Official D1 reset fields disappeared: {missing}")
    if getattr(head, "seed", None) != OFFICIAL_NOISE_SEED:
        raise RuntimeError("D1 effective model noise is not the fixed official seed 1140")
    before = temporal_snapshot(policy, rank=rank)
    head.current_start_frame = 0
    for field in RESET_FIELDS_TO_NONE:
        setattr(head, field, None)
    after = temporal_snapshot(policy, rank=rank)
    failures: list[str] = []
    if after["current_start_frame"] != 0:
        failures.append("current_start_frame did not reset to zero")
    for field in RESET_FIELDS_TO_NONE:
        if not after["fields"][field]["is_none"]:
            failures.append(f"{field} did not reset to None")
    receipt = {
        "schema_version": RESET_SCHEMA_VERSION,
        "reset_id": reset_id,
        "rank": rank,
        "status": "failed" if failures else "passed",
        "before": before,
        "after": after,
        "fields_cleared": ["current_start_frame", *RESET_FIELDS_TO_NONE],
        "failures": failures,
        "completed_at_utc": utc_now(),
    }
    if failures:
        raise RuntimeError("D1 temporal reset failed: " + "; ".join(failures))
    return receipt


@contextlib.contextmanager
def capture_cache_reinitialization(head: Any):
    """Measure the official cache constructors without changing their results."""

    method_names = ("_create_kv_caches", "_create_crossattn_caches")
    missing = [name for name in method_names if not hasattr(head, name)]
    if missing:
        raise RuntimeError(f"Official D1 cache constructors disappeared: {missing}")
    originals: dict[str, Any] = {}
    events: dict[str, list[dict[str, Any]]] = {name: [] for name in method_names}
    for name in method_names:
        original = getattr(head, name)
        originals[name] = original

        def measured(*args: Any, _name: str = name, _original: Any = original, **kwargs: Any):
            started = time.perf_counter_ns()
            result = _original(*args, **kwargs)
            events[_name].append(
                {
                    "elapsed_seconds": (time.perf_counter_ns() - started) / 1e9,
                    "result": _value_signature(result),
                }
            )
            return result

        setattr(head, name, measured)
    try:
        yield events
    finally:
        for name, original in originals.items():
            setattr(head, name, original)


def validate_first_request_cache_evidence(rank_metrics: Iterable[Mapping[str, Any]]) -> None:
    failures: list[str] = []
    records = list(rank_metrics)
    if sorted(record.get("rank") for record in records) != [0, 1]:
        failures.append("request metrics do not contain exactly ranks 0 and 1")
    for record in records:
        rank = record.get("rank")
        pre = record.get("temporal_before", {})
        post = record.get("temporal_after", {})
        if pre.get("current_start_frame") != 0:
            failures.append(f"rank {rank} did not enter first request at frame zero")
        for field in CACHE_FIELDS:
            if not pre.get("fields", {}).get(field, {}).get("is_none", False):
                failures.append(f"rank {rank} entered first request with populated {field}")
            if post.get("fields", {}).get(field, {}).get("is_none", True):
                failures.append(f"rank {rank} did not populate {field}")
        events = record.get("cache_reinitialization", {})
        for method in ("_create_kv_caches", "_create_crossattn_caches"):
            if len(events.get(method, [])) != 1:
                failures.append(f"rank {rank} called {method} {len(events.get(method, []))} times")
        if not isinstance(post.get("current_start_frame"), int) or post["current_start_frame"] <= 0:
            failures.append(f"rank {rank} did not advance current_start_frame")
    if failures:
        raise RuntimeError("D1 cache reinitialization evidence failed: " + "; ".join(failures))


def _cuda_measurement_start() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return {
        "cuda_available": True,
        "allocated_bytes_before": int(torch.cuda.memory_allocated()),
        "reserved_bytes_before": int(torch.cuda.memory_reserved()),
    }


def _cuda_measurement_finish(start: Mapping[str, Any]) -> dict[str, Any]:
    if not start.get("cuda_available"):
        return dict(start)
    torch.cuda.synchronize()
    return {
        **start,
        "allocated_bytes_after": int(torch.cuda.memory_allocated()),
        "reserved_bytes_after": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def measure_rank_forward(policy: Any, forward: Any, *args: Any, rank: int, **kwargs: Any):
    head = action_head(policy)
    before = temporal_snapshot(policy, rank=rank)
    cuda = _cuda_measurement_start()
    started = time.perf_counter_ns()
    with capture_cache_reinitialization(head) as cache_events:
        result = forward(*args, **kwargs)
    elapsed = (time.perf_counter_ns() - started) / 1e9
    cuda = _cuda_measurement_finish(cuda)
    after = temporal_snapshot(policy, rank=rank)
    return result, {
        "rank": rank,
        "wall_seconds": elapsed,
        "temporal_before": before,
        "temporal_after": after,
        "cache_reinitialization": cache_events,
        "cuda_memory": cuda,
    }


def _run_git(source_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(source_root), *arguments],
        text=True,
        stderr=subprocess.STDOUT,
    ).strip()


def verify_source_identity(source_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    expected = contract["source"]
    source_root = source_root.resolve()
    if source_root != Path(expected["path"]).resolve():
        raise ValueError(f"D1 source path changed: {source_root} != {expected['path']}")
    if _run_git(source_root, "rev-parse", "HEAD") != expected["commit"]:
        raise ValueError("D1 source commit mismatch")
    if _run_git(source_root, "rev-parse", "HEAD^{tree}") != expected["git_tree"]:
        raise ValueError("D1 source tree mismatch")
    tracked_status = _run_git(source_root, "status", "--porcelain", "--untracked-files=no")
    if tracked_status:
        raise ValueError(f"D1 exact source has tracked changes: {tracked_status}")
    tracked = _run_git(source_root, "ls-files", "-z").split("\0")
    tracked = sorted(path for path in tracked if path)
    lines: list[bytes] = []
    total_bytes = 0
    for relative in tracked:
        path = source_root / relative
        digest = sha256_file(path)
        total_bytes += path.stat().st_size
        lines.append(f"{digest}  {relative}\n".encode())
    aggregate = sha256_bytes(b"".join(lines))
    if len(tracked) != expected["tracked_file_count"]:
        raise ValueError("D1 source tracked-file count mismatch")
    if total_bytes != expected["tracked_bytes"]:
        raise ValueError("D1 source tracked-byte count mismatch")
    if aggregate != expected["aggregate_sha256"]:
        raise ValueError(f"D1 source aggregate mismatch: {aggregate}")
    required_records = []
    for relative, expected_hash in sorted(expected["required_files"].items()):
        observed = sha256_file(source_root / relative)
        if observed != expected_hash:
            raise ValueError(f"D1 source file mismatch: {relative}")
        required_records.append({"path": relative, "sha256": observed})
    return {
        "status": "passed",
        "path": str(source_root),
        "commit": expected["commit"],
        "git_tree": expected["git_tree"],
        "tracked_file_count": len(tracked),
        "tracked_bytes": total_bytes,
        "aggregate_sha256": aggregate,
        "required_files": required_records,
    }


def _payload_files(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }


def verify_payload_identity(
    root: Path,
    expected: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, Any]:
    root = root.resolve()
    if root != Path(expected["path"]).resolve():
        raise ValueError(f"D1 {role} path changed: {root} != {expected['path']}")
    expected_paths = {entry["path"] for entry in expected["files"]}
    observed_paths = _payload_files(root)
    if observed_paths != expected_paths:
        raise ValueError(
            f"D1 {role} payload inventory mismatch: "
            f"missing={sorted(expected_paths - observed_paths)}, "
            f"extra={sorted(observed_paths - expected_paths)}"
        )
    lines: list[bytes] = []
    total_bytes = 0
    records = []
    for entry in sorted(expected["files"], key=lambda value: value["path"]):
        path = root / entry["path"]
        size = path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"D1 {role} byte mismatch: {entry['path']}")
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise ValueError(f"D1 {role} SHA-256 mismatch: {entry['path']}")
        total_bytes += size
        lines.append(f"{digest}  {entry['path']}\n".encode())
        records.append({"path": entry["path"], "bytes": size, "sha256": digest})
    aggregate = sha256_bytes(b"".join(lines))
    if len(records) != expected["payload_file_count"]:
        raise ValueError(f"D1 {role} payload-file count mismatch")
    if total_bytes != expected["payload_bytes"]:
        raise ValueError(f"D1 {role} payload-byte count mismatch")
    if aggregate != expected["aggregate_sha256"]:
        raise ValueError(f"D1 {role} aggregate mismatch: {aggregate}")
    return {
        "status": "passed",
        "role": role,
        "path": str(root),
        "repository": expected["repository"],
        "revision": expected["revision"],
        "payload_file_count": len(records),
        "payload_bytes": total_bytes,
        "aggregate_sha256": aggregate,
        "files": records,
    }


def verify_exact_identities(
    *,
    source_root: Path,
    checkpoint_root: Path,
    tokenizer_root: Path,
    identity_contract_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    contract = json.loads(identity_contract_path.read_text())
    if contract.get("schema_version") != "wmf-d1-identity-contract-v1":
        raise ValueError("Unknown D1 identity contract")
    source = verify_source_identity(source_root, contract)
    checkpoint = verify_payload_identity(checkpoint_root, contract["checkpoint"], role="checkpoint")
    tokenizer = verify_payload_identity(tokenizer_root, contract["tokenizer"], role="tokenizer")
    return {
        "schema_version": "wmf-d1-runtime-identity-receipt-v1",
        "status": "passed",
        "verified_at_utc": utc_now(),
        "verification_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "identity_contract": str(identity_contract_path.resolve()),
        "identity_contract_sha256": sha256_file(identity_contract_path),
        "source": source,
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
    }


def _validate_episode_id(value: Any) -> str:
    if not isinstance(value, str) or not SAFE_ID.fullmatch(value):
        raise ValueError(f"Unsafe or missing D1 episode id: {value!r}")
    return value


def _validate_behavioral_finalize_counts(
    *,
    stop_reason: str,
    actions_executed: Any,
    request_count: Any,
    server_request_count: int,
) -> None:
    """Validate the exact completed or censored sequential request prefix."""

    if (
        type(actions_executed) is not int
        or not 0 <= actions_executed <= FULL_ACTION_CAP
        or type(request_count) is not int
        or not 0 <= request_count <= FULL_REQUEST_COUNT
        or request_count != server_request_count
        or actions_executed
        > min(FULL_ACTION_CAP, request_count * EXECUTED_ACTION_PREFIX)
    ):
        raise ValueError("D1 finalize control counts do not match the active episode")
    if stop_reason == "action_cap" and (
        actions_executed != FULL_ACTION_CAP
        or request_count != FULL_REQUEST_COUNT
    ):
        raise ValueError("D1 completed finalize counts changed")
    if stop_reason == "safety_abort" and not (
        0 < actions_executed < FULL_ACTION_CAP
        and request_count
        == (actions_executed + EXECUTED_ACTION_PREFIX - 1)
        // EXECUTED_ACTION_PREFIX
    ):
        raise ValueError("D1 safety-abort finalize is not a closed nonempty prefix")


def _decode_latent_measurement_only(
    *,
    head: Any,
    latent: torch.Tensor,
    request_dir: Path,
) -> dict[str, Any]:
    before_hash = tensor_data_sha256(latent)
    cuda = _cuda_measurement_start()
    started = time.perf_counter_ns()
    with torch.inference_mode():
        decoded = head.vae.decode(
            latent.detach(),
            tiled=head.tiled,
            tile_size=(head.tile_size_height, head.tile_size_width),
            tile_stride=(head.tile_stride_height, head.tile_stride_width),
        )
    elapsed = (time.perf_counter_ns() - started) / 1e9
    cuda = _cuda_measurement_finish(cuda)
    after_hash = tensor_data_sha256(latent)
    if before_hash != after_hash:
        raise RuntimeError("Measurement-only D1 VAE decode mutated the retained latent")
    decoded_cpu = decoded.detach().cpu().contiguous()
    tensor_path = request_dir / "offline_decoded_tensor.pt"
    save_tensor(tensor_path, decoded_cpu)
    if decoded_cpu.ndim != 5 or decoded_cpu.shape[0] != 1:
        raise ValueError(f"Unexpected D1 decoded tensor shape: {tuple(decoded_cpu.shape)}")
    rgb = decoded_cpu.permute(0, 2, 3, 4, 1)[0]
    rgb = ((rgb.float() + 1.0) * 127.5).clip(0, 255).to(torch.uint8).numpy()
    rgb_path = request_dir / "offline_decoded_rgb.npy"
    save_numpy(rgb_path, rgb)
    return {
        "requested": True,
        "performed": True,
        "mode": "offline VAE rendering after official joint action/video inference",
        "latent_data_sha256_before": before_hash,
        "latent_data_sha256_after": after_hash,
        "wall_seconds": elapsed,
        "cuda_memory": cuda,
        "decoded_tensor": {
            "path": str(tensor_path),
            "file_sha256": sha256_file(tensor_path),
            "data_sha256": tensor_data_sha256(decoded_cpu),
            "shape": list(decoded_cpu.shape),
            "dtype": str(decoded_cpu.dtype),
            "bytes": tensor_path.stat().st_size,
        },
        "decoded_rgb": {
            "path": str(rgb_path),
            "file_sha256": sha256_file(rgb_path),
            "data_sha256": array_data_sha256(rgb),
            "shape": list(rgb.shape),
            "dtype": str(rgb.dtype),
            "bytes": rgb_path.stat().st_size,
        },
    }


def _no_decode_receipt() -> dict[str, Any]:
    return {
        "requested": False,
        "performed": False,
        "mode": (
            "rendering skipped only; official jointly denoised video latent was still "
            "generated and retained"
        ),
    }


def _all_gather_object(local: Any, *, group: Any) -> list[Any]:
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, local, group=group)
    return gathered


def make_instrumented_policy_class(official_policy_class: type):
    """Create the rank-zero wrapper without importing the official repo at import time."""

    class D1InstrumentedPolicy(official_policy_class):
        def __init__(
            self,
            *args: Any,
            future_root: Path,
            signal_group: Any,
            server_contract_sha256: str,
            **kwargs: Any,
        ) -> None:
            # output_dir=None prevents the old reset-time concatenated decode. D1
            # performs only explicitly requested per-latent offline decodes.
            super().__init__(
                *args,
                signal_group=signal_group,
                output_dir=None,
                **kwargs,
            )
            self._future_root = Path(future_root).resolve()
            self._episodes_root = self._future_root / "episodes"
            self._episodes_root.mkdir(parents=True, exist_ok=False)
            self._signal_group = signal_group
            self._server_contract_sha256 = server_contract_sha256
            self._episode_id: str | None = None
            self._episode_dir: Path | None = None
            self._expected_session_id: str | None = None
            self._reset_receipt: dict[str, Any] | None = None
            self._measurement_records: list[dict[str, Any]] = []
            self._reset_generation = 0

        def _finalize_episode(self) -> dict[str, Any] | None:
            if self._episode_id is None:
                return None
            if self._episode_dir is None or self._reset_receipt is None:
                raise RuntimeError("D1 episode bookkeeping is incomplete")
            episode_id = self._episode_id
            episode_dir = self._episode_dir
            expected_session_id = self._expected_session_id
            manifest = {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "configuration_id": "D1",
                "episode_id": self._episode_id,
                "status": "complete",
                "official_repository_commit": OFFICIAL_COMMIT,
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
                "noise_semantics": "fixed; not an independent draw",
                "request_count": len(self._measurement_records),
                "two_rank_reset": self._reset_receipt,
                "requests": self._measurement_records,
                "server_contract_sha256": self._server_contract_sha256,
                "finalized_at_utc": utc_now(),
            }
            manifest_path = episode_dir / "episode_manifest.json"
            atomic_write_json(manifest_path, manifest)
            finalized = {
                "episode_id": episode_id,
                "episode_dir": episode_dir,
                "expected_session_id": expected_session_id,
                "episode_manifest": file_identity(manifest_path),
                "server_request_count": len(self._measurement_records),
                "begin_reset": copy.deepcopy(self._reset_receipt),
                "manifest_finalized_at_utc": manifest["finalized_at_utc"],
            }
            self._episode_id = None
            self._episode_dir = None
            self._expected_session_id = None
            self._reset_receipt = None
            self._measurement_records = []
            return finalized

        def reset(self, reset_info: dict) -> None:
            reset_info = copy.deepcopy(reset_info)
            control = reset_info.pop(RESET_KEY, None)
            if not isinstance(control, dict):
                raise ValueError(f"D1 reset requires a {RESET_KEY} mapping")
            finalize_only = bool(control.get("finalize_only", False))
            finalized: dict[str, Any] | None = None
            active_begin_control = (
                self._reset_receipt.get("control")
                if isinstance(self._reset_receipt, Mapping)
                else None
            )
            behavioral_finalize = (
                finalize_only
                and isinstance(active_begin_control, Mapping)
                and active_begin_control.get("model_config") == "D1"
            )
            if not finalize_only and control.get("model_config") == "D1":
                expected_begin_purpose = BEHAVIORAL_PURPOSE_BY_PHASE.get(
                    control.get("phase")
                )
                if (
                    expected_begin_purpose is None
                    or control.get("purpose") != expected_begin_purpose
                ):
                    raise ValueError("D1 behavioral begin purpose changed")
            if behavioral_finalize:
                if set(control) != FINALIZE_CONTROL_KEYS:
                    raise ValueError("D1 finalize control keys changed")
                if control.get("finalize_only") is not True:
                    raise ValueError("D1 behavioral finalize_only must be literal true")
                previous_episode_id = _validate_episode_id(control.get("previous_episode_id"))
                previous_session_id = control.get("previous_session_id")
                if (
                    self._episode_id is None
                    or previous_episode_id != self._episode_id
                    or not isinstance(previous_session_id, str)
                    or previous_session_id != self._expected_session_id
                    or previous_session_id == previous_episode_id
                ):
                    raise ValueError("D1 finalize control does not match the active episode/session")
                if control.get("model_config") != "D1":
                    raise ValueError("D1 finalize control has the wrong model")
                if self._reset_receipt is None:
                    raise RuntimeError("D1 active episode is missing its begin reset receipt")
                begin_control = self._reset_receipt.get("control")
                if not isinstance(begin_control, Mapping):
                    raise RuntimeError("D1 begin reset control is unavailable")
                expected_begin_purpose = BEHAVIORAL_PURPOSE_BY_PHASE.get(
                    begin_control.get("phase")
                )
                if (
                    expected_begin_purpose is None
                    or begin_control.get("model_config") != "D1"
                    or begin_control.get("purpose") != expected_begin_purpose
                    or control.get("purpose") != f"{expected_begin_purpose}_finalize"
                ):
                    raise ValueError("D1 behavioral begin/finalize purpose changed")
                for key in (
                    "model_config",
                    "study_id",
                    "phase",
                    "block_id",
                    "layout_pair_id",
                    "cell_id",
                    "condition_index",
                    "server_ready_sha256",
                    "simulator_claim_sha256",
                    "simulator_lease_token",
                    "pilot_contract_sha256",
                ):
                    if control.get(key) != begin_control.get(key):
                        raise ValueError(f"D1 finalize control changed active episode field {key}")
                if (
                    begin_control.get("episode_id") != previous_episode_id
                    or begin_control.get("expected_session_id") != previous_session_id
                ):
                    raise ValueError("D1 finalize IDs differ from the begin reset control")
                stop_reason = control.get("stop_reason")
                if stop_reason not in {"action_cap", "safety_abort", "technical_failure"}:
                    raise ValueError("D1 finalize control has an invalid stop reason")
                actions_executed = control.get("actions_executed")
                request_count = control.get("request_count")
                _validate_behavioral_finalize_counts(
                    stop_reason=stop_reason,
                    actions_executed=actions_executed,
                    request_count=request_count,
                    server_request_count=len(self._measurement_records),
                )
                finalized = self._finalize_episode()
                if finalized is None:
                    raise RuntimeError("D1 finalize did not close an active episode")
            elif finalize_only:
                # Qualification probes predate the behavioral terminal schema.
                # Preserve their finalize-only control without promoting their
                # manifests to behavioral terminal-context evidence.
                finalized = self._finalize_episode()
            elif self._episode_id is not None:
                if (
                    isinstance(active_begin_control, Mapping)
                    and active_begin_control.get("model_config") == "D1"
                ):
                    raise RuntimeError(
                        "D1 active behavioral episode requires an explicit "
                        "finalize-only reset"
                    )
                # The frozen six-probe qualification client starts the next
                # legacy probe with a reset and emits one finalize-only reset
                # after the loop.  Keep that transition exactly: a new legacy
                # begin closes the preceding legacy episode before resetting.
                self._finalize_episode()
            self._reset_generation += 1
            reset_id = f"reset-{self._reset_generation:06d}-{uuid.uuid4().hex}"

            signal = torch.full((1,), RESET_SIGNAL, dtype=torch.int32, device="cpu")
            dist.broadcast(signal, src=0, group=self._signal_group)
            payload = [{"reset_id": reset_id, "control": control}]
            dist.broadcast_object_list(payload, src=0, group=self._signal_group)

            wrapper_before = {
                "frame_buffer_lengths": {
                    key: len(value) for key, value in self._frame_buffers.items()
                },
                "call_count": self._call_count,
                "is_first_call": self._is_first_call,
                "video_across_time_count": len(self.video_across_time),
                "current_session_id": self._current_session_id,
            }
            super().reset(reset_info)
            self._current_session_id = None
            local = reset_temporal_state(self._policy, rank=0, reset_id=reset_id)
            local["wrapper_before"] = wrapper_before
            local["wrapper_after"] = {
                "frame_buffer_lengths": {
                    key: len(value) for key, value in self._frame_buffers.items()
                },
                "call_count": self._call_count,
                "is_first_call": self._is_first_call,
                "video_across_time_count": len(self.video_across_time),
                "current_session_id": self._current_session_id,
            }
            gathered = _all_gather_object(local, group=self._signal_group)
            if sorted(item.get("rank") for item in gathered) != [0, 1]:
                raise RuntimeError("D1 reset did not receive receipts from exactly two ranks")
            if any(item.get("status") != "passed" for item in gathered):
                raise RuntimeError("D1 reset failed on at least one distributed rank")

            reset_receipt = {
                "schema_version": RESET_SCHEMA_VERSION,
                "reset_id": reset_id,
                "status": "passed",
                "world_size": EXPECTED_WORLD_SIZE,
                "rank_receipts": sorted(gathered, key=lambda item: item["rank"]),
                "control": control,
                "completed_at_utc": utc_now(),
            }
            if finalize_only:
                if not behavioral_finalize:
                    return
                assert finalized is not None
                begin_reset = finalized.get("begin_reset")
                if not isinstance(begin_reset, Mapping):
                    raise RuntimeError("D1 finalized episode lost its begin reset")
                begin_rank_ids = {
                    row.get("reset_id")
                    for row in begin_reset.get("rank_receipts", [])
                    if isinstance(row, Mapping)
                }
                final_rank_ids = {
                    row.get("reset_id")
                    for row in reset_receipt.get("rank_receipts", [])
                    if isinstance(row, Mapping)
                }
                if (
                    reset_receipt["reset_id"] == begin_reset.get("reset_id")
                    or begin_rank_ids.intersection(final_rank_ids)
                ):
                    raise RuntimeError("D1 terminal reset is not distinct from begin")
                self._reset_receipt = None
                bookkeeping_cleared = (
                    self._episode_id is None
                    and self._episode_dir is None
                    and self._expected_session_id is None
                    and self._reset_receipt is None
                    and self._measurement_records == []
                )
                if not bookkeeping_cleared or self._current_session_id is not None:
                    raise RuntimeError("D1 terminal reset left active episode state")
                terminal = {
                    "schema_version": TERMINAL_SCHEMA_VERSION,
                    "status": "passed",
                    "terminal_state": "context_closed",
                    "model_config": "D1",
                    "study_id": control.get("study_id"),
                    "phase": control.get("phase"),
                    "block_id": control.get("block_id"),
                    "layout_pair_id": control.get("layout_pair_id"),
                    "cell_id": control.get("cell_id"),
                    "condition_index": control.get("condition_index"),
                    "episode_id": finalized["episode_id"],
                    "episode_context_id": finalized["episode_id"],
                    "server_context_id": finalized["episode_id"],
                    "client_session_id": finalized["expected_session_id"],
                    "stop_reason": control.get("stop_reason"),
                    "actions_executed": control.get("actions_executed"),
                    "request_count": control.get("request_count"),
                    "server_request_count": finalized["server_request_count"],
                    "episode_manifest": finalized["episode_manifest"],
                    "final_two_rank_reset": reset_receipt,
                    "episode_bookkeeping_cleared": True,
                    "context_active_after_finalize": False,
                    "completed_at_utc": utc_now(),
                }
                begin_completed = parse_utc(begin_reset.get("completed_at_utc"))
                manifest_completed = parse_utc(finalized["manifest_finalized_at_utc"])
                final_reset_completed = parse_utc(reset_receipt["completed_at_utc"])
                terminal_completed = parse_utc(terminal["completed_at_utc"])
                begin_rank_completed = [
                    parse_utc(row.get("completed_at_utc"))
                    for row in begin_reset.get("rank_receipts", [])
                    if isinstance(row, Mapping)
                ]
                final_rank_completed = [
                    parse_utc(row.get("completed_at_utc"))
                    for row in reset_receipt.get("rank_receipts", [])
                    if isinstance(row, Mapping)
                ]
                if not (
                    len(begin_rank_completed) == len(final_rank_completed) == EXPECTED_WORLD_SIZE
                    and all(value <= begin_completed for value in begin_rank_completed)
                    and begin_completed < manifest_completed < final_reset_completed
                    <= terminal_completed
                    and all(
                        manifest_completed < value <= final_reset_completed
                        for value in final_rank_completed
                    )
                ):
                    raise RuntimeError("D1 terminal evidence timestamps are out of order")
                immutable_write_json(
                    finalized["episode_dir"] / "terminal_context_receipt.json",
                    terminal,
                )
                return
            episode_id = _validate_episode_id(control.get("episode_id"))
            expected_session_id = control.get("expected_session_id")
            if not isinstance(expected_session_id, str) or not expected_session_id:
                raise ValueError("D1 reset requires expected_session_id")
            episode_dir = self._episodes_root / episode_id
            episode_dir.mkdir(parents=False, exist_ok=False)
            self._episode_id = episode_id
            self._episode_dir = episode_dir
            self._expected_session_id = expected_session_id
            self._reset_receipt = reset_receipt
            atomic_write_json(episode_dir / "reset_receipt.json", reset_receipt)

        def infer(self, obs: dict) -> np.ndarray:
            if self._episode_id is None or self._episode_dir is None:
                raise RuntimeError("D1 infer is fail-closed until an explicit two-rank reset")
            obs = dict(obs)
            control = obs.pop(MEASUREMENT_KEY, {})
            if not isinstance(control, dict):
                raise TypeError(f"{MEASUREMENT_KEY} must be a mapping")
            unknown_control = set(control) - {"probe_id", "offline_decode", "probe_plan_sha256"}
            if unknown_control:
                raise ValueError(f"Unknown D1 measurement controls: {sorted(unknown_control)}")
            validate_raw_request(obs)
            if obs["session_id"] != self._expected_session_id:
                raise ValueError(
                    f"D1 session changed without reset: {obs['session_id']!r} != "
                    f"{self._expected_session_id!r}"
                )
            probe_id = control.get("probe_id")
            if probe_id is not None and probe_id != self._episode_id:
                raise ValueError("D1 probe id must equal its reset-isolated episode id")
            if probe_id is not None and self._measurement_records:
                raise ValueError("Each D1 six-request probe episode must contain one request")
            offline_decode = control.get("offline_decode", False)
            if not isinstance(offline_decode, bool):
                raise TypeError("D1 offline_decode must be a boolean")

            request_index = len(self._measurement_records)
            request_dir = self._episode_dir / f"request_{request_index:04d}"
            request_dir.mkdir(parents=False, exist_ok=False)
            raw_mapping = {key: obs[key] for key in RAW_ARRAY_KEYS}
            raw_artifacts = save_exact_mapping(raw_mapping, request_dir / "raw_inputs", prefix="raw")

            captured: dict[str, Any] = {}
            original_policy_forward = self._policy.lazy_joint_forward_causal
            official_forward_call_count = 0

            def measured_policy_forward(batch: Any, *args: Any, **kwargs: Any):
                nonlocal official_forward_call_count
                official_forward_call_count += 1
                captured["converted_inputs"] = clone_mapping(batch.obs)
                trained_model = self._policy.trained_model
                original_model_forward = trained_model.lazy_joint_video_action_causal

                def measured_model_forward(normalized_input: Mapping[str, Any], *inner_args: Any, **inner_kwargs: Any):
                    captured["normalized_inputs"] = clone_mapping(normalized_input)
                    return original_model_forward(normalized_input, *inner_args, **inner_kwargs)

                trained_model.lazy_joint_video_action_causal = measured_model_forward
                try:
                    result, metrics = measure_rank_forward(
                        self._policy,
                        original_policy_forward,
                        batch,
                        *args,
                        rank=0,
                        **kwargs,
                    )
                finally:
                    trained_model.lazy_joint_video_action_causal = original_model_forward
                captured["rank_metrics"] = metrics
                captured["video_pred"] = result[1]
                return result

            self._policy.lazy_joint_forward_causal = measured_policy_forward
            wall_started = time.perf_counter_ns()
            try:
                returned_action = super().infer(obs)
            finally:
                self._policy.lazy_joint_forward_causal = original_policy_forward
            inference_wall_seconds = (time.perf_counter_ns() - wall_started) / 1e9
            if official_forward_call_count != 1:
                raise RuntimeError(
                    f"D1 official conditional forward called {official_forward_call_count} times"
                )
            if "video_pred" not in captured or captured["video_pred"] is None:
                raise RuntimeError("Official D1 forward did not return a video latent")
            action = np.asarray(returned_action)
            if action.shape != EXPECTED_ACTION_SHAPE or action.dtype != np.float32:
                raise ValueError(f"Official D1 action changed: {action.shape}/{action.dtype}")
            if not np.isfinite(action).all():
                raise ValueError("Official D1 returned non-finite actions")

            rank_metrics = _all_gather_object(captured["rank_metrics"], group=self._signal_group)
            if request_index == 0:
                validate_first_request_cache_evidence(rank_metrics)

            action_path = request_dir / "official_returned_action.npy"
            save_numpy(action_path, action)
            latent = captured["video_pred"].detach().cpu().contiguous()
            latent_path = request_dir / "official_video_pred.pt"
            save_tensor(latent_path, latent)
            converted = save_exact_mapping(
                captured["converted_inputs"],
                request_dir / "converted_inputs",
                prefix="converted",
            )
            normalized = save_exact_mapping(
                captured["normalized_inputs"],
                request_dir / "normalized_model_inputs",
                prefix="normalized",
            )
            if offline_decode:
                decode = _decode_latent_measurement_only(
                    head=action_head(self._policy),
                    latent=captured["video_pred"],
                    request_dir=request_dir,
                )
            else:
                decode = _no_decode_receipt()

            rank_gpu_seconds = sum(float(item["wall_seconds"]) for item in rank_metrics)
            output_bytes = action_path.stat().st_size + latent_path.stat().st_size
            if decode.get("performed"):
                output_bytes += decode["decoded_tensor"]["bytes"] + decode["decoded_rgb"]["bytes"]
            record = {
                "schema_version": REQUEST_SCHEMA_VERSION,
                "configuration_id": "D1",
                "episode_id": self._episode_id,
                "request_index": request_index,
                "probe_id": probe_id,
                "prompt": obs["prompt"],
                "prompt_utf8_sha256": sha256_bytes(obs["prompt"].encode("utf-8")),
                "session_id": obs["session_id"],
                "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
                "noise_semantics": "fixed; this request is not an independent noise draw",
                "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
                "custom_s2_used": False,
                "patched_s1_used": False,
                "official_forward_call_count": official_forward_call_count,
                "raw_inputs": raw_artifacts,
                "converted_inputs": converted,
                "normalized_model_inputs": normalized,
                "official_returned_action": {
                    "path": str(action_path),
                    "file_sha256": sha256_file(action_path),
                    "data_sha256": array_data_sha256(action),
                    "shape": list(action.shape),
                    "dtype": str(action.dtype),
                    "bytes": action_path.stat().st_size,
                    "definition": (
                        "Exact 24x8 array returned by the official conditional server before "
                        "client gripper binarization; the first eight actions are executable"
                    ),
                },
                "latent_video": {
                    "path": str(latent_path),
                    "file_sha256": sha256_file(latent_path),
                    "data_sha256": tensor_data_sha256(latent),
                    "shape": list(latent.shape),
                    "dtype": str(latent.dtype),
                    "bytes": latent_path.stat().st_size,
                    "definition": "Exact video_pred returned beside the official action batch",
                },
                "offline_decode": decode,
                "temporal_and_cache_rank_metrics": sorted(
                    rank_metrics, key=lambda item: item["rank"]
                ),
                "cost": {
                    "rank_count": EXPECTED_WORLD_SIZE,
                    "inference_wall_seconds_rank0_wrapper": inference_wall_seconds,
                    "summed_rank_forward_gpu_seconds_proxy": rank_gpu_seconds,
                    "offline_decode_wall_seconds": float(decode.get("wall_seconds", 0.0)),
                    "retained_output_bytes": output_bytes,
                    "definition": (
                        "GPU-seconds proxy is the sum of measured distributed-rank forward wall "
                        "times; it is not a cloud billing claim"
                    ),
                },
                "measurement_control": control,
                "completed_at_utc": utc_now(),
            }
            atomic_write_json(request_dir / "request_receipt.json", record)
            self._measurement_records.append(record)
            # Return the exact object produced by the official wrapper. No copy,
            # truncation, gripper threshold, or action guidance is applied here.
            return returned_action

    D1InstrumentedPolicy.__name__ = "D1InstrumentedARDroidPolicy"
    return D1InstrumentedPolicy


def _receive_batch_from_rank_zero() -> Any:
    from tianshou.data import Batch

    size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
    dist.broadcast(size_tensor, src=0)
    size = int(size_tensor.item())
    if size <= 0:
        raise RuntimeError(f"Invalid D1 distributed input byte count: {size}")
    data_tensor = torch.zeros(size, dtype=torch.uint8, device="cuda")
    dist.broadcast(data_tensor, src=0)
    obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
    return Batch(obs=obs)


async def run_instrumented_worker(policy: Any, *, signal_group: Any) -> None:
    rank = dist.get_rank()
    if rank != 1:
        raise RuntimeError(f"D1 worker loop may run only on rank 1, got {rank}")
    signal = torch.zeros(1, dtype=torch.int32, device="cpu")
    while True:
        dist.broadcast(signal, src=0, group=signal_group)
        command = int(signal.item())
        if command == SHUTDOWN_SIGNAL:
            return
        if command == RESET_SIGNAL:
            payload: list[Any] = [None]
            dist.broadcast_object_list(payload, src=0, group=signal_group)
            reset_id = payload[0]["reset_id"]
            receipt = reset_temporal_state(policy, rank=rank, reset_id=reset_id)
            _all_gather_object(receipt, group=signal_group)
            continue
        if command != INFER_SIGNAL:
            raise RuntimeError(f"Unknown D1 distributed signal: {command}")
        batch = _receive_batch_from_rank_zero()
        dist.barrier()
        result, metrics = measure_rank_forward(
            policy,
            policy.lazy_joint_forward_causal,
            batch,
            rank=rank,
        )
        dist.barrier()
        del result
        _all_gather_object(metrics, group=signal_group)


def _infer_study_root() -> Path:
    return Path(__file__).resolve().parents[4]


@dataclasses.dataclass(frozen=True)
class RuntimeArgs:
    port: int
    future_root: Path
    source_root: Path
    checkpoint_root: Path
    tokenizer_root: Path
    identity_contract: Path
    study_root: Path
    timeout_seconds: int
    enable_dit_cache: bool


def parse_args(argv: list[str] | None = None) -> RuntimeArgs:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/users/ali/vla_wam/external/DreamZero-v3e004-clean-ab790c1"),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/data/users/ali/vla_wam/checkpoints/DreamZero-DROID-96ad344"),
    )
    parser.add_argument(
        "--tokenizer-root",
        type=Path,
        default=Path("/data/users/ali/vla_wam/checkpoints/umt5-xxl-tokenizer-66cb9e7"),
    )
    parser.add_argument(
        "--identity-contract", type=Path, default=here / "d1_identity_contract.json"
    )
    parser.add_argument("--study-root", type=Path, default=_infer_study_root())
    parser.add_argument("--timeout-seconds", type=int, default=50000)
    parser.add_argument(
        "--enable-dit-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.port == 5000:
        parser.error("D1 refuses the historical shared port 5000")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    return RuntimeArgs(
        port=args.port,
        future_root=args.future_root.resolve(),
        source_root=args.source_root.resolve(),
        checkpoint_root=args.checkpoint_root.resolve(),
        tokenizer_root=args.tokenizer_root.resolve(),
        identity_contract=args.identity_contract.resolve(),
        study_root=args.study_root.resolve(),
        timeout_seconds=args.timeout_seconds,
        enable_dit_cache=args.enable_dit_cache,
    )


def _topology_receipt(*, rank: int, signal_group: Any) -> list[dict[str, Any]]:
    if dist.get_world_size() != EXPECTED_WORLD_SIZE:
        raise RuntimeError("Official D1 requires exactly two distributed ranks")
    if torch.cuda.device_count() != EXPECTED_VISIBLE_GPU_COUNT:
        raise RuntimeError("Official D1 requires exactly two visible GPUs")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "-1"))
    if local_rank != rank or local_world_size != EXPECTED_WORLD_SIZE:
        raise RuntimeError(
            "Official D1 must use one two-GPU torchrun node with LOCAL_RANK equal to rank"
        )
    current_device = torch.cuda.current_device()
    name = torch.cuda.get_device_name(current_device)
    if EXPECTED_GPU_NAME_SUBSTRING not in name:
        raise RuntimeError(f"Official D1 rank {rank} is not on a B200: {name}")
    properties = torch.cuda.get_device_properties(current_device)
    device_uuid = getattr(properties, "uuid", None)
    if isinstance(device_uuid, bytes):
        device_uuid = device_uuid.decode("ascii", "replace")
    elif device_uuid is not None:
        device_uuid = str(device_uuid)
    local = {
        "rank": rank,
        "local_rank": local_rank,
        "hostname": socket.gethostname(),
        "cuda_device_index": current_device,
        "cuda_device_name": name,
        "cuda_device_uuid": device_uuid,
        "cuda_total_memory_bytes": int(properties.total_memory),
        "cuda_device_properties": str(properties),
    }
    gathered = _all_gather_object(local, group=signal_group)
    if sorted(item["rank"] for item in gathered) != [0, 1]:
        raise RuntimeError("D1 topology did not gather exactly ranks 0 and 1")
    if len({item["hostname"] for item in gathered}) != 1:
        raise RuntimeError("Pinned D1 init_mesh is qualified only for one two-GPU node")
    if len({item["cuda_device_index"] for item in gathered}) != 2:
        raise RuntimeError("D1 ranks did not bind distinct visible GPU indices")
    uuids = [item["cuda_device_uuid"] for item in gathered]
    if all(uuids) and len(set(uuids)) != 2:
        raise RuntimeError("D1 ranks did not bind distinct B200 UUIDs")
    return sorted(gathered, key=lambda item: item["rank"])


def _broadcast_rank_zero_result(value: Any, *, group: Any) -> Any:
    payload = [value if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0, group=group)
    return payload[0]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    forbidden_overrides = (
        "DREAMZERO_ACTION_CFG_SCALE",
        "NUM_DIT_STEPS",
        "DYNAMIC_CACHE_SCHEDULE",
        "LOAD_TRT_ENGINE",
    )
    present_overrides = [name for name in forbidden_overrides if os.environ.get(name) is not None]
    if present_overrides:
        raise RuntimeError(f"D1 refuses runtime forward overrides: {present_overrides}")
    if not args.identity_contract.is_file():
        raise FileNotFoundError(args.identity_contract)
    if not args.source_root.is_dir():
        raise FileNotFoundError(args.source_root)
    official_server_path = args.source_root / "socket_test_optimized_AR.py"
    if sha256_file(official_server_path) != OFFICIAL_SERVER_SHA256:
        raise RuntimeError("D1 refuses non-official socket_test_optimized_AR.py bytes")
    sys.path.insert(0, str(args.source_root))
    loader_root = args.study_root / "experiments/dreamzero_droid"
    loader_path = loader_root / "v2_bounded_loader.py"
    if sha256_file(loader_path) != BOUNDED_LOADER_SHA256:
        raise RuntimeError("D1 bounded construction overlay bytes changed")
    sys.path.insert(1, str(loader_root))

    # DeepSpeed 0.19 must finish importing before Transformers 4.51 reaches
    # modeling_utils. This is the pinned v2 load-order compatibility shim; it
    # changes no model bytes, inputs, RNG, forward method, or outputs.
    import deepspeed  # noqa: F401

    # These imports are intentionally after the exact path/hash gates.
    import socket_test_optimized_AR as official
    from eval_utils.policy_server import PolicyServerConfig
    from groot.vla.data.schema import EmbodimentTag
    from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
    from v2_bounded_loader import install_bounded_loader

    if Path(official.__file__).resolve() != official_server_path.resolve():
        raise RuntimeError("Python imported DreamZero official server from the wrong checkout")

    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    device_mesh = official.init_mesh()
    rank = dist.get_rank()
    timeout = dt.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout)
    topology = _topology_receipt(rank=rank, signal_group=signal_group)

    if rank == 0:
        if args.future_root.exists() and any(args.future_root.iterdir()):
            raise FileExistsError(f"D1 future root must start empty: {args.future_root}")
        args.future_root.mkdir(parents=True, exist_ok=True)
        try:
            identity: dict[str, Any] = verify_exact_identities(
                source_root=args.source_root,
                checkpoint_root=args.checkpoint_root,
                tokenizer_root=args.tokenizer_root,
                identity_contract_path=args.identity_contract,
            )
        except BaseException as exc:
            identity = {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }
    else:
        identity = {}
    identity = _broadcast_rank_zero_result(identity, group=signal_group)
    if rank == 0:
        atomic_write_json(args.future_root / "identity_receipt.json", identity)
    if identity.get("status") != "passed":
        raise RuntimeError(f"D1 exact identity verification failed: {identity}")
    dist.barrier()

    bounded_path = args.future_root / f"bounded_loader_rank{rank}.json"
    install_bounded_loader(contract_path=bounded_path)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=str(args.checkpoint_root),
        device="cuda",
        device_mesh=device_mesh,
        tokenizer_path_override=str(args.tokenizer_root),
        model_config_overrides=[],
    )
    frame_block_contract = pinned_frame_block_contract(args.checkpoint_root, identity)
    local_head = validate_official_head(
        policy,
        rank=rank,
        frame_block_contract=frame_block_contract,
    )
    head_receipts = _all_gather_object(local_head, group=signal_group)
    dist.barrier()

    if rank == 0:
        loader_receipts = []
        for loader_rank in range(EXPECTED_WORLD_SIZE):
            path = args.future_root / f"bounded_loader_rank{loader_rank}.json"
            receipt = json.loads(path.read_text())
            if not receipt.get("passed") or receipt.get("forward_path_modified"):
                raise RuntimeError(f"D1 bounded loader failed on rank {loader_rank}")
            loader_receipts.append(
                {"rank": loader_rank, "path": str(path), "sha256": sha256_file(path), "receipt": receipt}
            )
        contract = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed",
            "configuration_id": "D1",
            "created_at_utc": utc_now(),
            "official_repository_commit": OFFICIAL_COMMIT,
            "official_repository_tree": OFFICIAL_TREE,
            "official_action_path": "GrootSimPolicy.lazy_joint_forward_causal",
            "custom_s2_used": False,
            "patched_s1_used": False,
            "source_root": str(args.source_root),
            "checkpoint_root": str(args.checkpoint_root),
            "tokenizer_root": str(args.tokenizer_root),
            "instrumentation_overlay": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__).resolve()),
                "role": "measurement_retention_and_full_episode_reset_only",
                "returned_action_modified": False,
            },
            "identity_contract": {
                "path": str(args.identity_contract),
                "sha256": sha256_file(args.identity_contract),
            },
            "bounded_loader_overlay": {
                "path": str(loader_path.resolve()),
                "sha256": sha256_file(loader_path),
                "role": "construction_only",
                "forward_path_modified": False,
            },
            "identity_receipt": str(args.future_root / "identity_receipt.json"),
            "identity_receipt_sha256": sha256_file(args.future_root / "identity_receipt.json"),
            "world_size": dist.get_world_size(),
            "topology": topology,
            "head_contracts": sorted(head_receipts, key=lambda item: item["rank"]),
            "bounded_loader_receipts": loader_receipts,
            "load_order_compatibility": {
                "deepspeed_imported_before_official_server": True,
                "model_forward_modified": False,
            },
            "port": args.port,
            "future_root": str(args.future_root),
            "returned_action_shape": list(EXPECTED_ACTION_SHAPE),
            "executed_action_prefix": EXECUTED_ACTION_PREFIX,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "noise_semantics": "fixed; no request is an independent noise draw",
            "video_guidance_scale": VIDEO_GUIDANCE_SCALE,
            "configured_inference_steps": CONFIGURED_INFERENCE_STEPS,
            "evaluated_dit_step_mask": EXPECTED_DIT_STEP_MASK,
            "evaluated_dit_step_count": sum(EXPECTED_DIT_STEP_MASK),
            "dynamic_cache_schedule": False,
            "tensorrt_engine_active": False,
            "enable_dit_cache": args.enable_dit_cache,
            "offline_decode_semantics": (
                "optional per retained latent after official joint inference; never an action-only mode"
            ),
        }
        contract_path = args.future_root / "server_contract.json"
        atomic_write_json(contract_path, contract)
        contract_hash = sha256_file(contract_path)
        Policy = make_instrumented_policy_class(official.ARDroidRoboarenaPolicy)
        wrapper = Policy(
            groot_policy=policy,
            signal_group=signal_group,
            future_root=args.future_root,
            server_contract_sha256=contract_hash,
        )
        config = PolicyServerConfig(
            image_resolution=(180, 320),
            needs_wrist_camera=True,
            n_external_cameras=2,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space="joint_position",
        )
        LOGGER.info("Serving exact official conditional D1 on %s:%d", socket.gethostname(), args.port)
        official.RoboarenaServer(
            policy=wrapper,
            server_config=config,
            host="0.0.0.0",
            port=args.port,
        ).serve_forever()
    else:
        asyncio.run(run_instrumented_worker(policy, signal_group=signal_group))


if __name__ == "__main__":
    main()
