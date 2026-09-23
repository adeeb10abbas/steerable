"""SGW-owned HTTP evidence producer for the official D1 server binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .adapters import AdapterError, DREAMZERO_CONFIG
from .dreamzero_backend import validate_dreamzero_result


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class D1ProducerBackend(Protocol):
    source_root: str
    checkpoint_path: str
    resolved_config: Mapping[str, Any]

    def predict(self, observation: Mapping[str, Any], prompt: str, sampling_seed: int, **kwargs: Any) -> Mapping[str, Any]: ...
    def reset(self, session_id: str | None) -> Mapping[str, Any] | None: ...


class DreamZeroEvidenceProducer:
    def __init__(
        self,
        backend: D1ProducerBackend,
        *,
        trace_path: Path,
        future_dir: Path,
        attestation_path: Path,
        clock: Callable[[], float] = time.time,
    ) -> None:
        config = dict(backend.resolved_config)
        missing = [key for key in DREAMZERO_CONFIG if config.get(key) != DREAMZERO_CONFIG[key]]
        if missing:
            raise AdapterError("D1 backend config is not the pinned official configuration")
        self.backend = backend
        self.trace_path = trace_path
        self.future_dir = future_dir
        self.clock = clock
        self._lock = threading.Lock()
        self._request_index = 0
        self._reset_id = ""
        self._camera_name = ""
        self._cell_id = ""
        self._fingerprint = ""
        self._prompt: str | None = None
        self._session_id: str | None = None
        attestation = {
            "model": "D1",
            "config": config,
            "source_root": str(Path(backend.source_root).resolve()),
            "checkpoint_path": str(Path(backend.checkpoint_path).resolve()),
            "source_commit": DREAMZERO_CONFIG["source_commit"],
            "checkpoint_revision": DREAMZERO_CONFIG["revision"],
            "identity_route": "verified_dreamzero_server_factory",
        }
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        attestation_path.write_text(json.dumps(attestation, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    def reset(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        camera = packet.get("camera_name")
        if not isinstance(camera, str) or not camera:
            raise AdapterError("D1 reset requires camera_name")
        with self._lock:
            native_reset = self.backend.reset(self._session_id)
            self._reset_id = f"sgw-d1-reset-{uuid.uuid4().hex}"
            self._camera_name = camera
            self._cell_id = ""
            self._fingerprint = ""
            self._prompt = None
            self._session_id = None
            self._request_index = 0
        return {
            "status": "reset",
            "reset_id": self._reset_id,
            "camera_name": camera,
            "native_reset": native_reset,
            "provenance": "sgw_wrapper_generated",
        }

    def predict(self, packet: Mapping[str, Any]) -> Mapping[str, Any]:
        prompt = packet.get("prompt")
        observation = packet.get("observation")
        seed = packet.get("sampling_seed")
        with self._lock:
            if not self._reset_id:
                raise AdapterError("D1 request arrived before reset")
            if not isinstance(prompt, str) or not isinstance(observation, Mapping) or type(seed) is not int:
                raise AdapterError("D1 request has invalid prompt, observation, or sampling_seed")
            if self._prompt is None:
                self._prompt = prompt
            elif prompt != self._prompt:
                raise AdapterError("D1 prompt changed within an episode")
            if packet.get("request_index") != self._request_index:
                raise AdapterError("D1 request_index is stale or non-contiguous")
            for field, current in (
                ("reset_id", self._reset_id),
                ("camera_name", self._camera_name),
            ):
                if packet.get(field) != current:
                    raise AdapterError(f"D1 packet {field} does not match reset binding")
            for field in ("registered_cell_id", "reset_fingerprint", "request_id"):
                if not isinstance(packet.get(field), str) or not packet[field]:
                    raise AdapterError(f"D1 packet lacks {field}")
            if not self._cell_id:
                self._cell_id = packet["registered_cell_id"]
                self._fingerprint = packet["reset_fingerprint"]
            elif (packet["registered_cell_id"], packet["reset_fingerprint"]) != (self._cell_id, self._fingerprint):
                raise AdapterError("D1 packet changed cell or reset fingerprint")
            started = time.time_ns()
            result = self.backend.predict(
                observation,
                prompt,
                seed,
                action_guidance=DREAMZERO_CONFIG["action_guidance"],
                video_guidance=DREAMZERO_CONFIG["video_guidance"],
                steps=DREAMZERO_CONFIG["configured_steps"],
                session_id=self._session_id,
            )
            actions, future = validate_dreamzero_result(result)
            if isinstance(result.get("session_id"), str):
                self._session_id = result["session_id"]
            finished = time.time_ns()
            wrapper_request_id = f"sgw-d1-request-{uuid.uuid4().hex}"
            record: dict[str, Any] = {
                "request_id": packet["request_id"],
                "wrapper_request_id": wrapper_request_id,
                "request_index": self._request_index,
                "reset_id": self._reset_id,
                "camera_name": self._camera_name,
                "registered_cell_id": self._cell_id,
                "reset_fingerprint": self._fingerprint,
                "sampling_seed": seed,
                "prompt_sha256": _sha(prompt.encode()),
                "request_started_ns": started,
                "request_finished_ns": finished,
                "actions_shape": list(actions.shape),
                "actions_sha256": _sha(actions.tobytes()),
                "provenance": "sgw_wrapper_generated",
            }
            if future is None:
                record["future_status"] = "not_exposed"
            else:
                future_array = np.asarray(future)
                if future_array.ndim < 1:
                    raise AdapterError("D1 native future must be an array")
                self.future_dir.mkdir(parents=True, exist_ok=True)
                path = self.future_dir / f"{wrapper_request_id}.npy"
                np.save(path, future_array, allow_pickle=False)
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                record.update({
                    "future_status": "exposed_and_retained",
                    "future_path": str(path),
                    "future_shape": list(future_array.shape),
                    "future_sha256": _sha(path.read_bytes()),
                })
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._request_index += 1
            return {
                "actions": actions,
                "request_id": packet["request_id"],
                "future_status": record["future_status"],
                "provenance": "sgw_wrapper_generated",
            }


def make_dreamzero_http_server(producer: DreamZeroEvidenceProducer, *, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, value: Mapping[str, Any]) -> None:
            body = json.dumps(value, default=lambda item: np.asarray(item).tolist()).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802
            try:
                packet = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode())
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
