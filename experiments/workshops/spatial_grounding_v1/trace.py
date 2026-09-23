"""Concrete native request/future trace-sidecar reader for SGW-01."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .adapters import AdapterError


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_trace_sidecar(
    *, request: Mapping[str, Any], response: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Read one server-written JSONL binding; never synthesize provenance."""

    path_value = os.environ.get("SGW01_TRACE_SIDECAR", "").strip()
    if not path_value:
        raise AdapterError("SGW01_TRACE_SIDECAR is required for native attribution")
    path = Path(path_value)
    if not path.is_file():
        raise AdapterError(f"native trace sidecar is missing: {path}")
    matches: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict) and record.get("request_id") == request["request_id"]:
            matches.append(record)
    if len(matches) != 1:
        raise AdapterError("native trace sidecar does not contain exactly one request binding")
    record = matches[0]
    required = (
        "request_id",
        "registered_cell_id",
        "request_index",
        "reset_id",
        "camera_id",
        "camera_name",
        "reset_fingerprint",
    )
    if any(key not in record for key in required):
        raise AdapterError("native trace sidecar binding is incomplete")
    if any(record[key] != request[key] for key in required):
        raise AdapterError("native trace sidecar binding differs from request")
    domain = record.get("actions_hash_domain", "returned_actions")
    if domain not in {"returned_actions", "raw_server_actions"}:
        raise AdapterError("native trace action hash domain is unknown")
    action_key = "raw_actions" if domain == "raw_server_actions" else "actions"
    actions = np.asarray(response.get(action_key), dtype=np.float32)
    if actions.shape != tuple(record.get("actions_shape", ())):
        raise AdapterError("native trace action shape differs from returned response")
    if record.get("actions_sha256") != hashlib.sha256(actions.tobytes()).hexdigest():
        raise AdapterError("native trace action hash differs from returned response")
    future_path = record.get("future_path")
    if future_path:
        artifact = Path(str(future_path))
        if not artifact.is_file() or record.get("future_sha256") != _sha256(artifact):
            raise AdapterError("native future artifact is missing or hash-mismatched")
        value = np.load(artifact, allow_pickle=False)
        if record.get("future_status") == "latent_only_retained":
            record["future_latents"] = value
        elif record.get("future_status") == "decoded_unmapped":
            if record.get("future_encoding") != "decoded_rgb_uint8":
                raise AdapterError("native future artifact is not declared decoded RGB")
            record["future"] = value
        elif record.get("future_status") == "exposed_and_retained":
            if str(record.get("future_encoding", "")).startswith("native_latent"):
                raise AdapterError("native latent evidence cannot be treated as decoded future")
            record["future"] = value
        else:
            raise AdapterError("native retained future has no recognized evidence classification")
    latent_path = record.get("future_latent_path")
    if latent_path:
        latent_artifact = Path(str(latent_path))
        if (
            not latent_artifact.is_file()
            or record.get("future_latent_sha256") != _sha256(latent_artifact)
        ):
            raise AdapterError("native future latent artifact is missing or hash-mismatched")
        record["future_latent"] = np.load(latent_artifact, allow_pickle=False)
    latent_chunks = record.get("future_latent_chunks")
    if latent_chunks is not None:
        if not isinstance(latent_chunks, list) or not latent_chunks:
            raise AdapterError("native future latent chunk manifest is invalid")
        record["future_latent_chunks"] = []
        for chunk in latent_chunks:
            if not isinstance(chunk, Mapping):
                raise AdapterError("native future latent chunk record is invalid")
            chunk_path = Path(str(chunk.get("path", "")))
            if (
                not chunk_path.is_file()
                or chunk.get("sha256") != _sha256(chunk_path)
            ):
                raise AdapterError("native future latent chunk is missing or hash-mismatched")
            record["future_latent_chunks"].append(np.load(chunk_path, allow_pickle=False))
    if (
        not future_path
        and not latent_path
        and record.get("future_status") not in {"not_exposed", "decode_error"}
    ):
        raise AdapterError("native trace must explicitly classify missing future evidence")
    return record
