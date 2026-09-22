"""SGW-owned Nano server boundary and evidence producer.

The pinned Cosmos service does model inference; this wrapper owns the SGW
request boundary.  It never passes SGW metadata to the model backend and never
turns wrapper-generated identifiers into native model identifiers.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .adapters import AdapterError, NANO_CONFIG


CHECKPOINT_REGISTRY = Path(__file__).resolve().parents[3] / (
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "cosmos3_nano_policy_droid_v2a011_registry.json"
)


class NanoBackend(Protocol):
    resolved_config: Mapping[str, Any]
    source_root: str
    checkpoint_path: str

    def predict(
        self, observation: Mapping[str, Any], prompt: str, sampling_seed: int
    ) -> Mapping[str, Any]:
        """Run exactly one pinned backend request."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _proc_start_identity(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split(") ", 1)[-1].split()[19]
    except (OSError, IndexError) as exc:
        raise AdapterError("SGW Nano attestation requires Linux process start identity") from exc


def _git_revision(source_root: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", source_root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status = subprocess.run(
            ["git", "-C", source_root, "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AdapterError(f"cannot derive pinned Nano source revision: {source_root}") from exc
    revision = result.stdout.strip()
    if not revision:
        raise AdapterError("pinned Nano source revision is empty")
    if status.stdout.strip():
        raise AdapterError("pinned Nano source has tracked modifications")
    return revision


def _checkpoint_revision(checkpoint_path: str) -> str:
    return _checkpoint_identity(checkpoint_path)[0]


def _checkpoint_identity(checkpoint_path: str) -> tuple[str, str]:
    root = Path(checkpoint_path)
    registry_path = CHECKPOINT_REGISTRY
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read immutable Nano checkpoint registry: {registry_path}") from exc
    checkpoint = registry.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise AdapterError("immutable Nano checkpoint registry lacks checkpoint provenance")
    expected_files = checkpoint.get("files")
    if not isinstance(expected_files, Mapping) or not expected_files:
        raise AdapterError("immutable Nano checkpoint registry lacks complete file hashes")
    revision = checkpoint.get("revision")
    asset = checkpoint.get("id")
    if revision != NANO_CONFIG["revision"] or asset != NANO_CONFIG["asset"]:
        raise AdapterError("immutable Nano checkpoint registry differs from pinned asset identity")
    for relative, metadata in expected_files.items():
        if not isinstance(relative, str) or not isinstance(metadata, Mapping):
            raise AdapterError("invalid Nano checkpoint file manifest entry")
        expected_sha = metadata.get("sha256")
        expected_bytes = metadata.get("bytes")
        path = root / relative
        if not isinstance(expected_sha, str) or not isinstance(expected_bytes, int) or not path.is_file():
            raise AdapterError(f"Nano checkpoint file is missing from immutable manifest: {relative}")
        try:
            actual_bytes = path.stat().st_size
            hasher = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        except OSError as exc:
            raise AdapterError(f"cannot hash Nano checkpoint file: {path}") from exc
        if actual_bytes != expected_bytes or digest != expected_sha:
            raise AdapterError(f"Nano checkpoint file hash mismatch: {relative}")
    return revision, asset


def derive_nano_attestation(
    backend: NanoBackend,
    *,
    expected_config: Mapping[str, Any] = NANO_CONFIG,
) -> dict[str, Any]:
    """Derive identity from loaded backend/source/checkpoint facts."""
    actual_config = dict(backend.resolved_config)
    if actual_config != dict(expected_config):
        raise AdapterError("loaded Nano backend config differs from SGW-01 N3 config")
    source_commit = _git_revision(backend.source_root)
    if source_commit != expected_config["source_commit"]:
        raise AdapterError("loaded Nano source revision differs from SGW-01 N3 source commit")
    checkpoint_revision = _checkpoint_revision(backend.checkpoint_path)
    if checkpoint_revision != expected_config["revision"]:
        raise AdapterError("loaded Nano checkpoint revision differs from SGW-01 N3 revision")
    pid = os.getpid()
    return {
        "server_pid": pid,
        "server_start_time": _proc_start_identity(pid),
        "model": "N3",
        "config": actual_config,
        "source_commit": source_commit,
        "checkpoint_revision": checkpoint_revision,
        "identity_route": "clean_pinned_source_and_verified_checkpoint_file_manifest",
    }


class NanoEvidenceProducer:
    """Own SGW request/reset identity and write append-only native evidence."""

    def __init__(
        self,
        backend: NanoBackend,
        *,
        trace_path: Path,
        future_dir: Path,
        attestation_path: Path,
        expected_config: Mapping[str, Any] = NANO_CONFIG,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.backend = backend
        self.trace_path = trace_path
        self.future_dir = future_dir
        self.clock = clock
        self._lock = threading.Lock()
        self._request_index = 0
        self._reset_id = ""
        self._wrapper_reset_id = ""
        self._camera_name = ""
        self._bound_cell_id = ""
        self._bound_fingerprint = ""
        self._bound_reset_id = ""
        self._prompt: str | None = None
        self.attestation = derive_nano_attestation(backend, expected_config=expected_config)
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        attestation_path.write_text(
            json.dumps(self.attestation, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def reset(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        camera = packet.get("camera_name")
        if not isinstance(camera, str) or not camera:
            raise AdapterError("SGW Nano reset requires camera_name")
        with self._lock:
            self._reset_id = f"sgw-reset-{uuid.uuid4().hex}"
            self._wrapper_reset_id = self._reset_id
            self._camera_name = camera
            self._bound_cell_id = ""
            self._bound_fingerprint = ""
            self._bound_reset_id = ""
            self._prompt = None
            self._request_index = 0
        return {
            "status": "reset",
            "reset_id": self._reset_id,
            "wrapper_reset_id": self._wrapper_reset_id,
            "camera_name": self._camera_name,
            "provenance": "sgw_wrapper_generated",
        }

    def predict(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = packet.get("prompt")
        observation = packet.get("observation")
        sampling_seed = packet.get("sampling_seed")
        if not isinstance(prompt, str) or not isinstance(observation, Mapping):
            raise AdapterError("SGW Nano packet lacks prompt or native observation")
        if type(sampling_seed) is not int:
            raise AdapterError("SGW Nano sampling_seed must be an integer")
        with self._lock:
            if not self._reset_id or not self._camera_name:
                raise AdapterError("SGW Nano request arrived before reset")
            if self._prompt is None:
                self._prompt = prompt
            elif prompt != self._prompt:
                raise AdapterError("SGW Nano prompt changed within an episode")
            request_index = packet.get("request_index")
            if type(request_index) is not int or request_index != self._request_index:
                raise AdapterError("SGW Nano request_index is stale or non-contiguous")
            packet_reset_id = packet.get("reset_id")
            packet_camera_name = packet.get("camera_name")
            packet_cell_id = packet.get("registered_cell_id")
            packet_fingerprint = packet.get("reset_fingerprint")
            packet_request_id = packet.get("request_id")
            if not isinstance(packet_reset_id, str) or not packet_reset_id:
                raise AdapterError("SGW Nano packet lacks reset_id")
            if packet_camera_name != self._camera_name:
                raise AdapterError("SGW Nano packet camera differs from reset camera")
            if not isinstance(packet_cell_id, str) or not packet_cell_id:
                raise AdapterError("SGW Nano packet lacks registered_cell_id")
            if not isinstance(packet_fingerprint, str) or not packet_fingerprint:
                raise AdapterError("SGW Nano packet lacks reset_fingerprint")
            if not isinstance(packet_request_id, str) or not packet_request_id:
                raise AdapterError("SGW Nano packet lacks request_id")
            if not self._bound_cell_id:
                self._bound_cell_id = packet_cell_id
                self._bound_fingerprint = packet_fingerprint
                self._bound_reset_id = packet_reset_id
            elif (
                packet_cell_id != self._bound_cell_id
                or packet_fingerprint != self._bound_fingerprint
                or packet_reset_id != self._bound_reset_id
            ):
                raise AdapterError("SGW Nano request changed cell or reset binding")
            reset_id = packet_reset_id
            camera_name = packet_camera_name
            backend_packet = self.backend.predict(observation, prompt, sampling_seed)
            actions = np.asarray(backend_packet.get("action"), dtype=np.float32)
            if actions.shape != (32, 8) or not np.isfinite(actions).all():
                raise AdapterError("pinned Nano backend did not return finite (32,8) actions")
            record: dict[str, Any] = {
            "request_id": packet_request_id,
            "wrapper_request_id": f"sgw-n3-request-{uuid.uuid4().hex}",
            "request_index": request_index,
            "reset_id": reset_id,
            "wrapper_reset_id": self._wrapper_reset_id,
            "camera_name": camera_name,
            "camera_id": packet.get("camera_id", camera_name),
            "registered_cell_id": packet_cell_id,
            "reset_fingerprint": packet_fingerprint,
            "sampling_seed": sampling_seed,
            "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
            "created_at": self.clock(),
            "provenance": "sgw_wrapper_generated",
            "backend_request": {
                "prompt": prompt,
                "sampling_seed": sampling_seed,
                "observation_keys": sorted(map(str, observation.keys())),
            },
            "actions_shape": list(actions.shape),
            "actions_sha256": _sha256_bytes(actions.tobytes()),
            }
            for key in ("request_id", "registered_cell_id", "reset_fingerprint"):
                if not isinstance(record[key], str) or not record[key]:
                    raise AdapterError(f"SGW Nano packet lacks {key}")
            future = backend_packet.get("future")
            if future is not None:
                future_array = np.asarray(future)
                if future_array.ndim < 1:
                    raise AdapterError("Nano backend future must be an array")
                self.future_dir.mkdir(parents=True, exist_ok=True)
                future_path = self.future_dir / f"{record['wrapper_request_id']}.npy"
                np.save(future_path, future_array, allow_pickle=False)
                with future_path.open("rb") as future_handle:
                    os.fsync(future_handle.fileno())
                record.update(
                {
                    "future_status": "exposed_and_retained",
                    "future_path": str(future_path),
                    "future_sha256": _sha256_bytes(future_path.read_bytes()),
                    "future_shape": list(future_array.shape),
                }
                )
            else:
                record["future_status"] = "not_exposed"
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._request_index += 1
            return {
                "action": actions,
                "request_id": record["request_id"],
                "future_status": record["future_status"],
                "provenance": "sgw_wrapper_generated",
            }


def serve_nano_wrapper(
    backend: NanoBackend,
    *,
    host: str,
    port: int,
    trace_path: Path,
    future_dir: Path,
    attestation_path: Path,
) -> None:
    producer = NanoEvidenceProducer(
        backend,
        trace_path=trace_path,
        future_dir=future_dir,
        attestation_path=attestation_path,
    )
    make_nano_http_server(producer, host=host, port=port).serve_forever()


def make_nano_http_server(
    producer: NanoEvidenceProducer, *, host: str, port: int
) -> ThreadingHTTPServer:
    """Build the production HTTP boundary for in-process integration tests."""
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: Mapping[str, Any]) -> None:
            body = json.dumps(value, default=lambda item: np.asarray(item).tolist()).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            try:
                packet = json.loads(self.rfile.read(length).decode("utf-8"))
                if self.path == "/reset":
                    self._json(200, producer.reset(packet))
                elif self.path == "/predict":
                    self._json(200, producer.predict(packet))
                else:
                    self._json(404, {"error": "not found"})
            except (AdapterError, ValueError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return None

    return ThreadingHTTPServer((host, port), Handler)
