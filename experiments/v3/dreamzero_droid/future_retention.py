#!/usr/bin/env python3
"""Partition DreamZero's concurrent server retention by exact session ID.

The released V2-A015 server writes one global manifest whenever any client
resets.  Concurrent clients can therefore be co-batched in one manifest and a
single client session can span several manifests.  This module creates a
lossless session view: requests are selected only by the UUID sent with every
inference request, their returned actions are checked against the client's
received chunks, and the decode is selected only from the reset that names the
same UUID.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FUTURE_SCHEMA = "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1"
EXPECTED_IDENTITY = {
    "schema_version": FUTURE_SCHEMA,
    "amendment_id": "V2-A015",
    "action_cfg_style_scale": 2.0,
    "video_cfg_scale": 5.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"retained DreamZero file is absent or empty: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _load_manifests(future_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    for path in Path(future_root).resolve().glob("episode_*/future_manifest.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("schema_version") != FUTURE_SCHEMA:
            continue
        loaded.append((path.resolve(), manifest))
    return sorted(
        loaded,
        key=lambda item: (int(item[1].get("episode_index", -1)), str(item[0])),
    )


def _session_ids(manifests: Iterable[tuple[Path, dict[str, Any]]]) -> set[str]:
    return {
        str(request["session_id"])
        for _, manifest in manifests
        for request in manifest.get("requests", [])
        if isinstance(request, dict) and request.get("session_id")
    }


def partition_session(
    future_root: Path,
    *,
    session_id: str,
    prompt: str,
    returned_raw_chunks: np.ndarray,
) -> dict[str, Any]:
    """Return one exact session manifest or fail closed.

    The result retains the released schema so existing downstream validators
    can validate it, while extra provenance fields identify all immutable
    global manifests from which the session view was derived.
    """

    chunks = np.asarray(returned_raw_chunks, dtype=np.float32)
    if chunks.ndim != 3 or chunks.shape[1:] != (24, 8) or chunks.shape[0] <= 0:
        raise ValueError("DreamZero retained raw chunks must have shape [N,24,8]")
    manifests = _load_manifests(future_root)
    closing = [
        (path, manifest)
        for path, manifest in manifests
        if session_id in (manifest.get("reset_info", {}).get("session_ids") or [])
    ]
    if len(closing) != 1:
        raise ValueError(
            f"expected one finalized reset for DreamZero session {session_id}, "
            f"found {len(closing)}"
        )
    selected: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path, manifest in manifests:
        records = [
            request
            for request in manifest.get("requests", [])
            if isinstance(request, dict) and request.get("session_id") == session_id
        ]
        if records:
            if any(manifest.get(key) != value for key, value in EXPECTED_IDENTITY.items()):
                raise ValueError(f"DreamZero source manifest identity changed: {path}")
            selected.extend(records)
            sources.append(path)
    closing_path, closing_manifest = closing[0]
    if any(
        closing_manifest.get(key) != value for key, value in EXPECTED_IDENTITY.items()
    ):
        raise ValueError(f"DreamZero closing manifest identity changed: {closing_path}")
    if closing_path not in sources:
        sources.append(closing_path)
    if len(selected) != chunks.shape[0]:
        raise ValueError(
            f"DreamZero session {session_id} has {len(selected)} retained requests, "
            f"client received {chunks.shape[0]}"
        )
    for index, (request, raw_chunk) in enumerate(zip(selected, chunks, strict=True)):
        if request.get("prompt") != prompt or request.get("action_cfg_style_scale") != 2.0:
            raise ValueError(f"DreamZero session request {index} changed prompt/s=2 identity")
        action_entry = request.get("returned_action", {})
        action_path = Path(str(action_entry.get("path", "")))
        if (
            not action_path.is_file()
            or action_entry.get("sha256") != sha256_file(action_path)
            or not np.array_equal(np.load(action_path, allow_pickle=False), raw_chunk)
        ):
            raise ValueError(
                f"DreamZero session request {index} differs from the client response"
            )
        latent_entry = request.get("latent_video", {})
        latent_path = Path(str(latent_entry.get("path", "")))
        if (
            not latent_path.is_file()
            or latent_entry.get("sha256") != sha256_file(latent_path)
            or (
                "bytes" in latent_entry
                and latent_entry.get("bytes") != latent_path.stat().st_size
            )
        ):
            raise ValueError(f"DreamZero latent future {index} failed its hash check")
    decoded = closing_manifest.get("official_reset_decode")
    if not isinstance(decoded, list) or not decoded:
        raise ValueError(f"DreamZero session {session_id} has no reset decode")
    for index, entry in enumerate(decoded):
        decoded_path = Path(str(entry.get("path", "")))
        if (
            not decoded_path.is_file()
            or entry.get("sha256") != sha256_file(decoded_path)
            or ("bytes" in entry and entry.get("bytes") != decoded_path.stat().st_size)
        ):
            raise ValueError(f"DreamZero reset decode {index} failed its hash check")
    session_manifest = dict(closing_manifest)
    session_manifest.update({
        "request_count": len(selected),
        "requests": selected,
        "official_reset_decode": decoded,
        "reset_info": {"session_ids": [session_id]},
        "session_id": session_id,
        "concurrent_session_partition": {
            "method": "exact_server_session_id_then_client_action_tensor_equality",
            "source_future_manifests": [file_record(path) for path in sources],
        },
    })
    return session_manifest


def identify_and_partition_session(
    future_root: Path,
    *,
    prompt: str,
    returned_raw_chunks: np.ndarray,
) -> dict[str, Any]:
    """Recover a failed client's UUID using exact action-tensor equality."""

    matches: list[dict[str, Any]] = []
    manifests = _load_manifests(future_root)
    for session_id in sorted(_session_ids(manifests)):
        try:
            candidate = partition_session(
                future_root,
                session_id=session_id,
                prompt=prompt,
                returned_raw_chunks=returned_raw_chunks,
            )
        except ValueError:
            continue
        matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(
            "exact DreamZero session recovery must have one action-identical match, "
            f"found {len(matches)}"
        )
    return matches[0]


def write_session_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite DreamZero session evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path
