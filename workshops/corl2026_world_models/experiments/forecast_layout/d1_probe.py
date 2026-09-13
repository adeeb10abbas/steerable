#!/usr/bin/env python3
"""Run and evaluate the six-request official-conditional DreamZero D1 probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


PROBE_SCHEMA = "wmf-d1-six-request-probe-plan-v1"
REPORT_SCHEMA = "wmf-d1-six-request-qualification-v1"
SERVER_SCHEMA = "wmf-d1-instrumented-server-v1"
EPISODE_SCHEMA = "wmf-d1-episode-manifest-v1"
REQUEST_SCHEMA = "wmf-d1-request-receipt-v1"
EXPECTED_ACTION_SHAPE = (24, 8)
OFFICIAL_NOISE_SEED = 1140
MEASUREMENT_KEY = "wmf_d1_measurement"
RESET_KEY = "wmf_d1_reset"
RAW_ARRAY_KEYS = (
    "observation/exterior_image_0_left",
    "observation/exterior_image_1_left",
    "observation/wrist_image_left",
    "observation/joint_position",
    "observation/cartesian_position",
    "observation/gripper_position",
)


def utc_now() -> str:
    import datetime as dt

    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def save_numpy(path: Path, value: np.ndarray) -> None:
    with Path(path).open("xb") as handle:
        np.save(handle, np.ascontiguousarray(value), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    fsync_directory(Path(path).parent)


def array_data_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def tensor_data_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    if tensor.numel() == 0:
        return sha256_bytes(b"")
    return sha256_bytes(tensor.view(torch.uint8).numpy().tobytes(order="C"))


class MsgPackNumpy:
    """The released RoboLab/DreamZero numpy msgpack wire codec."""

    def __init__(self) -> None:
        import msgpack

        self._msgpack = msgpack

    def pack(self, value: Any) -> bytes:
        return self._msgpack.packb(value, default=self._encode_numpy)

    def unpack(self, value: bytes) -> Any:
        return self._msgpack.unpackb(
            value,
            object_hook=self._decode_numpy,
            strict_map_key=False,
        )

    @staticmethod
    def _encode_numpy(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            if value.dtype.kind in ("V", "O", "c"):
                raise ValueError(f"Unsupported numpy wire dtype: {value.dtype}")
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }
        if isinstance(value, np.generic):
            return {
                b"__npgeneric__": True,
                b"data": value.item(),
                b"dtype": value.dtype.str,
            }
        raise TypeError(f"Cannot msgpack {type(value)!r}")

    @staticmethod
    def _decode_numpy(value: Any) -> Any:
        if b"__ndarray__" in value:
            return np.ndarray(
                buffer=value[b"data"],
                dtype=np.dtype(value[b"dtype"]),
                shape=value[b"shape"],
            )
        if b"__npgeneric__" in value:
            return np.dtype(value[b"dtype"]).type(value[b"data"])
        return value


def load_probe_plan(path: Path) -> tuple[dict[str, Any], str]:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != PROBE_SCHEMA:
        raise ValueError("Unknown D1 probe plan schema")
    probes = plan.get("probes", [])
    expected_ids = [
        "left_no_decode",
        "left_repeat_no_decode",
        "right_no_decode",
        "left_decode",
        "left_repeat_decode",
        "right_decode",
    ]
    if [probe.get("id") for probe in probes] != expected_ids:
        raise ValueError("D1 six-request order changed")
    if [probe.get("index") for probe in probes] != list(range(6)):
        raise ValueError("D1 probe indices changed")
    if [probe.get("offline_decode") for probe in probes] != [False] * 3 + [True] * 3:
        raise ValueError("D1 decode/no-decode sequence changed")
    if plan.get("effective_noise_seed") != OFFICIAL_NOISE_SEED:
        raise ValueError("D1 probe plan is not fixed at noise seed 1140")
    if not isinstance(plan.get("fixed_session_id"), str) or not plan["fixed_session_id"]:
        raise ValueError("D1 probe plan requires one fixed session id")
    sensitivity = plan.get("input_sensitivity_check", {})
    if sensitivity.get("qualification_gate") is not False:
        raise ValueError("D1 prompt sensitivity must remain a measurement, not a gate")
    if sensitivity.get("retain_action_difference_measurement") is not True:
        raise ValueError("D1 action sensitivity measurement was removed")
    if sensitivity.get("retain_latent_difference_measurement") is not True:
        raise ValueError("D1 latent sensitivity measurement was removed")
    return plan, sha256_file(path)


def load_fixture(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise ValueError(f"D1 fixture SHA-256 mismatch: {observed} != {expected_sha256}")
    with np.load(path, allow_pickle=False) as archive:
        fixture = {key: archive[key] for key in archive.files}
    if set(fixture) != set(RAW_ARRAY_KEYS):
        raise ValueError(
            "D1 fixture array keys changed: "
            f"missing={sorted(set(RAW_ARRAY_KEYS) - set(fixture))}, "
            f"extra={sorted(set(fixture) - set(RAW_ARRAY_KEYS))}"
        )
    expected = {
        "observation/exterior_image_0_left": ((180, 320, 3), np.dtype("uint8")),
        "observation/exterior_image_1_left": ((180, 320, 3), np.dtype("uint8")),
        "observation/wrist_image_left": ((180, 320, 3), np.dtype("uint8")),
        "observation/joint_position": ((7,), np.dtype("float64")),
        "observation/cartesian_position": ((6,), np.dtype("float64")),
        "observation/gripper_position": ((1,), np.dtype("float64")),
    }
    for key, (shape, dtype) in expected.items():
        value = fixture[key]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"D1 fixture contract changed for {key}: {value.shape}/{value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"D1 fixture contains non-finite values: {key}")
    return fixture


def validate_server_contract(path: Path, *, remote_port: int) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    checks = {
        "schema_version": contract.get("schema_version") == SERVER_SCHEMA,
        "status": contract.get("status") == "passed",
        "configuration": contract.get("configuration_id") == "D1",
        "official_conditional": contract.get("official_action_path")
        == "GrootSimPolicy.lazy_joint_forward_causal",
        "no_s2": contract.get("custom_s2_used") is False,
        "no_patched_s1": contract.get("patched_s1_used") is False,
        "world_size": contract.get("world_size") == 2,
        "action_shape": contract.get("returned_action_shape") == [24, 8],
        "noise_seed": contract.get("effective_official_model_noise_seed") == 1140,
        "port": contract.get("port") == remote_port,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D1 server contract checks failed: {failed}")
    return contract


def connect(uri: str, packer: MsgPackNumpy):
    import websockets.sync.client

    connection = websockets.sync.client.connect(
        uri,
        compression=None,
        max_size=None,
        open_timeout=300,
        ping_interval=60,
        ping_timeout=600,
    )
    raw_metadata = connection.recv(timeout=300)
    metadata = packer.unpack(raw_metadata) if not isinstance(raw_metadata, str) else raw_metadata
    return connection, metadata


def recv_action(connection: Any, packer: MsgPackNumpy) -> np.ndarray:
    raw = connection.recv(timeout=1200)
    if isinstance(raw, str):
        raise RuntimeError(f"D1 server error:\n{raw}")
    response = packer.unpack(raw)
    if isinstance(response, dict):
        response = response.get("actions", response)
    action = np.asarray(response)
    if action.shape != EXPECTED_ACTION_SHAPE or action.dtype != np.float32:
        raise ValueError(f"D1 returned action changed: {action.shape}/{action.dtype}")
    if not np.isfinite(action).all():
        raise ValueError("D1 returned non-finite actions")
    return action


def send_reset(connection: Any, packer: MsgPackNumpy, control: Mapping[str, Any]) -> None:
    connection.send(
        packer.pack(
            {
                "endpoint": "reset",
                "session_ids": [control.get("expected_session_id")]
                if control.get("expected_session_id")
                else None,
                RESET_KEY: dict(control),
            }
        )
    )
    reply = connection.recv(timeout=1200)
    if not isinstance(reply, str) or reply != "reset successful":
        if not isinstance(reply, str):
            reply = packer.unpack(reply)
        raise RuntimeError(f"D1 reset failed: {reply!r}")


def run_requests(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    fixture: Mapping[str, np.ndarray],
    future_root: Path,
    output_dir: Path,
    remote_host: str,
    remote_port: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    uri = f"ws://{remote_host}:{remote_port}"
    packer = MsgPackNumpy()
    connection, server_metadata = connect(uri, packer)
    returned: dict[str, dict[str, Any]] = {}
    session_id = plan["fixed_session_id"]
    started = time.perf_counter_ns()
    try:
        for probe in plan["probes"]:
            probe_id = probe["id"]
            send_reset(
                connection,
                packer,
                {
                    "episode_id": probe_id,
                    "expected_session_id": session_id,
                    "purpose": "d1_six_request_qualification",
                    "probe_plan_sha256": plan_sha256,
                },
            )
            request = {key: value for key, value in fixture.items()}
            request.update(
                {
                    "endpoint": "infer",
                    "prompt": probe["prompt"],
                    "session_id": session_id,
                    MEASUREMENT_KEY: {
                        "probe_id": probe_id,
                        "offline_decode": probe["offline_decode"],
                        "probe_plan_sha256": plan_sha256,
                    },
                }
            )
            request_started = time.perf_counter_ns()
            connection.send(packer.pack(request))
            action = recv_action(connection, packer)
            elapsed = (time.perf_counter_ns() - request_started) / 1e9
            action_path = output_dir / f"{probe_id}_returned_action.npy"
            save_numpy(action_path, action)
            returned[probe_id] = {
                "path": str(action_path),
                "file_sha256": sha256_file(action_path),
                "data_sha256": array_data_sha256(action),
                "shape": list(action.shape),
                "dtype": str(action.dtype),
                "client_round_trip_seconds": elapsed,
            }
        send_reset(
            connection,
            packer,
            {
                "finalize_only": True,
                "purpose": "d1_six_request_qualification_finalize",
                "probe_plan_sha256": plan_sha256,
            },
        )
    finally:
        connection.close()
    return {
        "uri": uri,
        "server_metadata": server_metadata,
        "returned_actions": returned,
        "total_client_wall_seconds": (time.perf_counter_ns() - started) / 1e9,
        "future_root": str(future_root.resolve()),
    }


def _resolve_artifact(raw_path: str, manifest_path: Path, *, allowed_root: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        candidate = path
    else:
        candidate = manifest_path.parent / path
    lexical = Path(os.path.abspath(candidate))
    root = Path(allowed_root).resolve()
    if not lexical.is_relative_to(root):
        raise ValueError(f"D1 retained artifact escaped its attempt root: {lexical}")
    cursor = lexical
    while cursor != root:
        if cursor.is_symlink():
            raise ValueError(f"D1 retained artifact path contains a symlink: {lexical}")
        cursor = cursor.parent
    resolved = lexical.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"D1 retained artifact escaped its attempt root: {resolved}")
    if not resolved.is_file():
        raise ValueError(f"D1 retained artifact is not a regular file: {resolved}")
    return resolved


def _verify_mapping(
    mapping: Mapping[str, Any], manifest_path: Path, *, allowed_root: Path
) -> None:
    identities = []
    entries = mapping.get("entries", [])
    if not isinstance(entries, list) or mapping.get("entry_count") != len(entries):
        raise ValueError("D1 exact input entry count changed")
    keys = [entry.get("key") for entry in entries if isinstance(entry, Mapping)]
    if len(keys) != len(entries) or len(set(keys)) != len(keys):
        raise ValueError("D1 exact input entries are invalid or duplicated")
    for entry in entries:
        path = _resolve_artifact(entry["path"], manifest_path, allowed_root=allowed_root)
        if sha256_file(path) != entry["file_sha256"]:
            raise ValueError(f"D1 retained input file mismatch: {path}")
        if entry["kind"] == "numpy_array":
            value = np.load(path, allow_pickle=False)
            if array_data_sha256(value) != entry["data_sha256"]:
                raise ValueError(f"D1 retained numpy data mismatch: {path}")
        elif entry["kind"] == "torch_tensor":
            value = torch.load(path, map_location="cpu", weights_only=True)
            if tensor_data_sha256(value) != entry["data_sha256"]:
                raise ValueError(f"D1 retained tensor data mismatch: {path}")
        elif entry["kind"] == "json_value":
            value = json.loads(path.read_text())
            if sha256_bytes(canonical_json_bytes(value)) != entry["json_sha256"]:
                raise ValueError(f"D1 retained JSON data mismatch: {path}")
        else:
            raise ValueError(f"Unknown D1 exact input artifact kind: {entry['kind']}")
        identities.append(
            {
                key: entry[key]
                for key in ("key", "kind", "shape", "dtype", "data_sha256", "json_sha256")
                if key in entry
            }
        )
    observed = sha256_bytes(canonical_json_bytes(identities))
    if observed != mapping.get("content_sha256"):
        raise ValueError("D1 exact input content aggregate mismatch")


def _load_action(
    entry: Mapping[str, Any], manifest_path: Path, *, allowed_root: Path
) -> np.ndarray:
    path = _resolve_artifact(entry["path"], manifest_path, allowed_root=allowed_root)
    if sha256_file(path) != entry["file_sha256"]:
        raise ValueError(f"D1 action file mismatch: {path}")
    action = np.load(path, allow_pickle=False)
    if action_data := entry.get("data_sha256"):
        if array_data_sha256(action) != action_data:
            raise ValueError(f"D1 action data mismatch: {path}")
    if action.shape != EXPECTED_ACTION_SHAPE or action.dtype != np.float32:
        raise ValueError(f"D1 server action changed: {action.shape}/{action.dtype}")
    return action


def _load_latent(
    entry: Mapping[str, Any], manifest_path: Path, *, allowed_root: Path
) -> torch.Tensor:
    path = _resolve_artifact(entry["path"], manifest_path, allowed_root=allowed_root)
    if sha256_file(path) != entry["file_sha256"]:
        raise ValueError(f"D1 latent file mismatch: {path}")
    latent = torch.load(path, map_location="cpu", weights_only=True)
    if tensor_data_sha256(latent) != entry["data_sha256"]:
        raise ValueError(f"D1 latent data mismatch: {path}")
    return latent


def _verify_reset_and_cache(
    manifest: Mapping[str, Any],
    *,
    probe_id: str,
    session_id: str,
    plan_sha256: str,
) -> None:
    reset = manifest.get("two_rank_reset", {})
    if reset.get("status") != "passed" or reset.get("world_size") != 2:
        raise ValueError("D1 episode lacks a passed two-rank reset")
    control = reset.get("control")
    if not isinstance(control, Mapping):
        raise ValueError("D1 episode reset lacks its exact control binding")
    expected_control = {
        "episode_id": probe_id,
        "expected_session_id": session_id,
        "purpose": "d1_six_request_qualification",
        "probe_plan_sha256": plan_sha256,
    }
    if dict(control) != expected_control:
        raise ValueError(f"D1 episode reset control changed for {probe_id}")
    rank_receipts = reset.get("rank_receipts", [])
    if sorted(receipt.get("rank") for receipt in rank_receipts) != [0, 1]:
        raise ValueError("D1 reset receipts do not cover exactly ranks 0 and 1")
    for receipt in rank_receipts:
        after = receipt.get("after", {})
        if after.get("current_start_frame") != 0:
            raise ValueError("D1 reset did not leave current_start_frame at zero")
        for field in (
            "kv_cache1",
            "kv_cache_neg",
            "crossattn_cache",
            "crossattn_cache_neg",
            "clip_feas",
            "ys",
            "language",
        ):
            if not after.get("fields", {}).get(field, {}).get("is_none", False):
                raise ValueError(f"D1 reset did not clear {field}")

    request = manifest["requests"][0]
    metrics = request.get("temporal_and_cache_rank_metrics", [])
    if sorted(record.get("rank") for record in metrics) != [0, 1]:
        raise ValueError("D1 request metrics do not cover exactly ranks 0 and 1")
    for record in metrics:
        pre = record.get("temporal_before", {})
        post = record.get("temporal_after", {})
        if pre.get("current_start_frame") != 0:
            raise ValueError("D1 first request did not start at current_start_frame zero")
        if not isinstance(post.get("current_start_frame"), int) or post["current_start_frame"] <= 0:
            raise ValueError("D1 first request did not advance current_start_frame")
        for field in ("kv_cache1", "kv_cache_neg", "crossattn_cache", "crossattn_cache_neg"):
            if not pre.get("fields", {}).get(field, {}).get("is_none", False):
                raise ValueError(f"D1 first request began with populated {field}")
            if post.get("fields", {}).get(field, {}).get("is_none", True):
                raise ValueError(f"D1 first request did not reinitialize {field}")
        events = record.get("cache_reinitialization", {})
        if len(events.get("_create_kv_caches", [])) != 1:
            raise ValueError("D1 first request did not recreate KV caches exactly once")
        if len(events.get("_create_crossattn_caches", [])) != 1:
            raise ValueError("D1 first request did not recreate cross-attention caches exactly once")


def _rms(left: np.ndarray, right: np.ndarray) -> float:
    delta = left.astype(np.float64) - right.astype(np.float64)
    return float(np.sqrt(np.mean(delta * delta)))


def _tensor_rms(left: torch.Tensor, right: torch.Tensor) -> float:
    delta = left.detach().cpu().float() - right.detach().cpu().float()
    return float(torch.sqrt(torch.mean(delta * delta)).item())


def evaluate(
    *,
    plan: Mapping[str, Any],
    plan_sha256: str,
    future_root: Path,
    run_receipt: Mapping[str, Any],
    fixture_path: Path,
    fixture_sha256: str,
    server_contract_path: Path,
    probe_output_dir: Path,
) -> dict[str, Any]:
    future_root = Path(future_root).resolve()
    probe_output_dir = Path(probe_output_dir).resolve()
    server_contract_path = Path(server_contract_path).resolve()
    if not server_contract_path.is_relative_to(future_root):
        raise ValueError("D1 server contract escaped the current future root")
    if server_contract_path.is_symlink() or not server_contract_path.is_file():
        raise ValueError("D1 server contract is not a regular current-attempt file")
    if not probe_output_dir.is_dir() or probe_output_dir.is_symlink():
        raise ValueError("D1 probe output root is not a regular current-attempt directory")
    server_contract_payload = json.loads(server_contract_path.read_text())
    server_port = server_contract_payload.get("port")
    if type(server_port) is not int:
        raise ValueError("D1 server contract lacks an integer port")
    server_contract = validate_server_contract(server_contract_path, remote_port=server_port)
    server_contract_sha256 = sha256_file(server_contract_path)
    expected_probe_ids = [probe["id"] for probe in plan["probes"]]
    returned_actions = run_receipt.get("returned_actions")
    if not isinstance(returned_actions, Mapping) or set(returned_actions) != set(expected_probe_ids):
        raise ValueError("D1 client receipt does not contain exactly the six planned actions")
    if Path(str(run_receipt.get("future_root", ""))).resolve() != future_root:
        raise ValueError("D1 client receipt is bound to another future root")
    actions: dict[str, np.ndarray] = {}
    latents: dict[str, torch.Tensor] = {}
    records: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}

    for probe in plan["probes"]:
        probe_id = probe["id"]
        manifest_path = future_root / "episodes" / probe_id / "episode_manifest.json"
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError(f"D1 episode manifest is not a regular file: {probe_id}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != EPISODE_SCHEMA:
            raise ValueError(f"Unknown D1 episode schema for {probe_id}")
        if manifest.get("status") != "complete" or manifest.get("request_count") != 1:
            raise ValueError(f"D1 probe episode is not one complete request: {probe_id}")
        if manifest.get("official_action_path") != "GrootSimPolicy.lazy_joint_forward_causal":
            raise ValueError("D1 episode did not use the official conditional path")
        if manifest.get("custom_s2_used") is not False or manifest.get("patched_s1_used") is not False:
            raise ValueError("D1 episode used a forbidden guidance path")
        if manifest.get("server_contract_sha256") != server_contract_sha256:
            raise ValueError(f"D1 episode is not bound to this server contract: {probe_id}")
        _verify_reset_and_cache(
            manifest,
            probe_id=probe_id,
            session_id=plan["fixed_session_id"],
            plan_sha256=plan_sha256,
        )
        request = manifest["requests"][0]
        if request.get("schema_version") != REQUEST_SCHEMA:
            raise ValueError(f"Unknown D1 request schema for {probe_id}")
        if request.get("probe_id") != probe_id or request.get("prompt") != probe["prompt"]:
            raise ValueError(f"D1 probe identity/prompt mismatch: {probe_id}")
        if request.get("session_id") != plan["fixed_session_id"]:
            raise ValueError(f"D1 request session changed for {probe_id}")
        expected_measurement = {
            "probe_id": probe_id,
            "offline_decode": probe["offline_decode"],
            "probe_plan_sha256": plan_sha256,
        }
        if request.get("measurement_control") != expected_measurement:
            raise ValueError(f"D1 request measurement/plan binding changed for {probe_id}")
        if request.get("effective_official_model_noise_seed") != OFFICIAL_NOISE_SEED:
            raise ValueError("D1 request did not use fixed official seed 1140")
        if request.get("official_forward_call_count") != 1:
            raise ValueError("D1 request did not invoke official conditional forward exactly once")
        for mapping_name in ("raw_inputs", "converted_inputs", "normalized_model_inputs"):
            _verify_mapping(request[mapping_name], manifest_path, allowed_root=future_root)
        action = _load_action(
            request["official_returned_action"], manifest_path, allowed_root=future_root
        )
        latent = _load_latent(request["latent_video"], manifest_path, allowed_root=future_root)
        client_entry = returned_actions[probe_id]
        client_action = _load_action(
            client_entry, probe_output_dir / "client_run_receipt.json", allowed_root=probe_output_dir
        )
        if not np.array_equal(action, client_action):
            raise ValueError(f"D1 instrumentation changed returned action for {probe_id}")
        decode = request.get("offline_decode", {})
        if decode.get("requested") is not probe["offline_decode"]:
            raise ValueError(f"D1 decode request mismatch: {probe_id}")
        if decode.get("performed") is not probe["offline_decode"]:
            raise ValueError(f"D1 decode execution mismatch: {probe_id}")
        if probe["offline_decode"]:
            decoded_tensor = _load_latent(
                decode["decoded_tensor"], manifest_path, allowed_root=future_root
            )
            del decoded_tensor
            decoded_rgb_path = _resolve_artifact(
                decode["decoded_rgb"]["path"], manifest_path, allowed_root=future_root
            )
            if sha256_file(decoded_rgb_path) != decode["decoded_rgb"]["file_sha256"]:
                raise ValueError(f"D1 decoded RGB file mismatch: {probe_id}")
            decoded_rgb = np.load(decoded_rgb_path, allow_pickle=False)
            if array_data_sha256(decoded_rgb) != decode["decoded_rgb"]["data_sha256"]:
                raise ValueError(f"D1 decoded RGB data mismatch: {probe_id}")
            if decode.get("latent_data_sha256_before") != request["latent_video"]["data_sha256"]:
                raise ValueError(f"D1 decode input latent mismatch: {probe_id}")
            if decode.get("latent_data_sha256_after") != request["latent_video"]["data_sha256"]:
                raise ValueError(f"D1 decode mutated latent: {probe_id}")
        actions[probe_id] = action
        latents[probe_id] = latent
        records[probe_id] = {
            "episode_manifest": str(manifest_path),
            "episode_manifest_sha256": sha256_file(manifest_path),
            "raw_input_content_sha256": request["raw_inputs"]["content_sha256"],
            "converted_input_content_sha256": request["converted_inputs"]["content_sha256"],
            "normalized_input_content_sha256": request["normalized_model_inputs"]["content_sha256"],
            "action_data_sha256": request["official_returned_action"]["data_sha256"],
            "latent_data_sha256": request["latent_video"]["data_sha256"],
            "offline_decode_performed": decode["performed"],
            "cost": request["cost"],
        }

    raw_hashes = {records[probe["id"]]["raw_input_content_sha256"] for probe in plan["probes"]}
    checks["all_six_raw_array_inputs_bit_exact"] = len(raw_hashes) == 1
    checks["all_six_reset_isolated"] = True
    checks["all_six_fixed_noise_1140"] = True
    checks["all_six_official_conditional_no_s2"] = True

    comparisons: dict[str, Any] = {}
    for left, right in plan["required_exact_equalities"]["fixed_left_repeats"]:
        key = f"repeat__{left}__{right}"
        action_equal = bool(np.array_equal(actions[left], actions[right]))
        latent_equal = bool(torch.equal(latents[left], latents[right]))
        comparisons[key] = {
            "action_array_equal": action_equal,
            "latent_tensor_equal": latent_equal,
            "action_rms": _rms(actions[left], actions[right]),
            "latent_rms": _tensor_rms(latents[left], latents[right]),
        }
        checks[f"{key}__action_equal"] = action_equal
        checks[f"{key}__latent_equal"] = latent_equal

    for left, right in plan["required_exact_equalities"]["decode_toggle_action_and_latent"]:
        key = f"decode_toggle__{left}__{right}"
        action_equal = bool(np.array_equal(actions[left], actions[right]))
        latent_equal = bool(torch.equal(latents[left], latents[right]))
        comparisons[key] = {
            "action_array_equal": action_equal,
            "latent_tensor_equal": latent_equal,
            "action_rms": _rms(actions[left], actions[right]),
            "latent_rms": _tensor_rms(latents[left], latents[right]),
        }
        checks[f"{key}__action_equal"] = action_equal
        checks[f"{key}__latent_equal"] = latent_equal

    for left, right in plan["required_exact_equalities"]["same_prompt_preprocessed_inputs"]:
        key = f"preprocessed_input__{left}__{right}"
        equal = (
            records[left]["converted_input_content_sha256"]
            == records[right]["converted_input_content_sha256"]
            and records[left]["normalized_input_content_sha256"]
            == records[right]["normalized_input_content_sha256"]
        )
        checks[key] = equal

    sensitivity_measurements: dict[str, Any] = {}
    for suffix in ("no_decode", "decode"):
        left = "left_no_decode" if suffix == "no_decode" else "left_decode"
        right = "right_no_decode" if suffix == "no_decode" else "right_decode"
        key = f"prompt_sensitivity__{suffix}"
        action_rms = _rms(actions[left], actions[right])
        latent_rms = _tensor_rms(latents[left], latents[right])
        sensitivity_measurements[key] = {
            "action_array_equal": bool(np.array_equal(actions[left], actions[right])),
            "latent_tensor_equal": bool(torch.equal(latents[left], latents[right])),
            "action_rms": action_rms,
            "latent_rms": latent_rms,
            "action_difference_observed": action_rms > 0.0,
            "latent_difference_observed": latent_rms > 0.0,
            "qualification_gate": False,
        }

    decode_count = sum(record["offline_decode_performed"] for record in records.values())
    checks["exactly_three_offline_decodes"] = decode_count == 3
    checks["exactly_six_generation_requests"] = len(records) == 6
    failed = sorted(name for name, passed in checks.items() if not passed)
    finite_cost = all(
        math.isfinite(float(record["cost"]["inference_wall_seconds_rank0_wrapper"]))
        and float(record["cost"]["inference_wall_seconds_rank0_wrapper"]) > 0
        for record in records.values()
    )
    checks["finite_positive_request_timing"] = finite_cost
    if not finite_cost:
        failed.append("finite_positive_request_timing")

    return {
        "schema_version": REPORT_SCHEMA,
        "status": "passed" if not failed else "failed",
        "passed": not failed,
        "configuration_id": "D1",
        "completed_at_utc": utc_now(),
        "probe_plan_sha256": plan_sha256,
        "fixture": {"path": str(fixture_path.resolve()), "sha256": fixture_sha256},
        "server_contract": {
            "path": str(server_contract_path.resolve()),
            "sha256": sha256_file(server_contract_path),
            "official_repository_commit": server_contract["official_repository_commit"],
            "checkpoint_root": server_contract["checkpoint_root"],
            "tokenizer_root": server_contract["tokenizer_root"],
            "world_size": server_contract["world_size"],
        },
        "generation_request_count": len(records),
        "behavioral_episode_count": 0,
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "noise_semantics": "fixed; six requests are not independent noise draws",
        "decode_semantics": (
            "three optional offline decodes of retained latents; all six requests execute "
            "the unchanged joint video/action forward"
        ),
        "records": records,
        "comparisons": comparisons,
        "sensitivity_measurements": sensitivity_measurements,
        "checks": checks,
        "failed_checks": failed,
        "client_run": run_receipt,
        "claim_boundary": (
            "This report qualifies source/runtime determinism, reset isolation, action/latent "
            "retention, and measurement-only decoding. Prompt sensitivity is a measured outcome, "
            "not a runtime gate. It is not a "
            "robot episode or physical forecast/action/camera alignment result."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--plan", type=Path, default=here / "d1_probe_plan.json")
    args = parser.parse_args(argv)
    if args.remote_port == 5000:
        parser.error("D1 refuses the historical shared port 5000")
    if len(args.fixture_sha256) != 64:
        parser.error("--fixture-sha256 must be a full SHA-256")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    failure_path = args.output_dir.parent / f"{args.output_dir.name}_failure.json"
    try:
        plan, plan_sha256 = load_probe_plan(args.plan.resolve())
        fixture = load_fixture(args.fixture.resolve(), args.fixture_sha256)
        future_root = args.future_root.resolve()
        server_contract_path = future_root / "server_contract.json"
        validate_server_contract(server_contract_path, remote_port=args.remote_port)
        if args.output_dir.exists():
            raise FileExistsError(f"D1 probe output directory must be new: {args.output_dir}")
        for probe in plan["probes"]:
            manifest = future_root / "episodes" / probe["id"] / "episode_manifest.json"
            if manifest.exists():
                raise FileExistsError(f"D1 probe episode already exists: {manifest}")
        run_receipt = run_requests(
            plan=plan,
            plan_sha256=plan_sha256,
            fixture=fixture,
            future_root=future_root,
            output_dir=args.output_dir.resolve(),
            remote_host=args.remote_host,
            remote_port=args.remote_port,
        )
        report = evaluate(
            plan=plan,
            plan_sha256=plan_sha256,
            future_root=future_root,
            run_receipt=run_receipt,
            fixture_path=args.fixture,
            fixture_sha256=args.fixture_sha256,
            server_contract_path=server_contract_path,
            probe_output_dir=args.output_dir,
        )
        report_path = args.output_dir / "d1_probe_qualification.json"
        atomic_write_json(report_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(20)
    except BaseException as exc:
        failure = {
            "schema_version": "wmf-d1-six-request-probe-failure-v1",
            "status": "technical_failure",
            "recorded_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "future_root": str(args.future_root.resolve()),
            "safe_to_retry_in_new_attempt_roots": True,
        }
        if not isinstance(exc, SystemExit) or exc.code != 20:
            atomic_write_json(failure_path, failure)
        raise


if __name__ == "__main__":
    main()
