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
        self._physical_reset_id = ""
        self._sampling_seed: int | None = None
        self._prompt: str | None = None
        self._session_id: str | None = None
        self.native_metadata = dict(getattr(backend, "native_metadata", {
            key: value for key, value in config.items() if key not in DREAMZERO_CONFIG
        }))
        attestation = {
            "model": "D1",
            "config": {key: config[key] for key in DREAMZERO_CONFIG},
            "native_metadata": self.native_metadata,
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
            self._physical_reset_id = ""
            self._sampling_seed = None
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
            if type(packet.get("request_index")) is not int or packet["request_index"] != self._request_index:
                raise AdapterError("D1 request_index is stale or non-contiguous")
            for field, current in (
                ("wrapper_reset_id", self._reset_id),
                ("camera_name", self._camera_name),
            ):
                supplied = packet.get(field)
                if supplied != current:
                    raise AdapterError(f"D1 packet {field} does not match reset binding")
            for field in ("registered_cell_id", "reset_fingerprint", "request_id", "reset_id"):
                if not isinstance(packet.get(field), str) or not packet[field]:
                    raise AdapterError(f"D1 packet lacks {field}")
            if packet.get("camera_id") != self._camera_name:
                raise AdapterError("D1 packet camera_id differs from reset camera")
            session_id = observation.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise AdapterError("D1 packet lacks the official client session_id")
            if not self._cell_id:
                self._cell_id = packet["registered_cell_id"]
                self._fingerprint = packet["reset_fingerprint"]
                self._physical_reset_id = packet["reset_id"]
                self._sampling_seed = seed
                self._session_id = session_id
            elif (
                packet["registered_cell_id"], packet["reset_fingerprint"],
                packet["reset_id"], seed, session_id,
            ) != (
                self._cell_id, self._fingerprint, self._physical_reset_id,
                self._sampling_seed, self._session_id,
            ):
                raise AdapterError("D1 packet changed cell, reset, seed or native session")
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
            if result.get("session_id", self._session_id) != self._session_id:
                raise AdapterError("D1 native response changed the official client session")
            finished = time.time_ns()
            wrapper_request_id = f"sgw-d1-request-{uuid.uuid4().hex}"
            record: dict[str, Any] = {
                "request_id": packet["request_id"],
                "wrapper_request_id": wrapper_request_id,
                "request_index": self._request_index,
                "reset_id": self._physical_reset_id,
                "wrapper_reset_id": self._reset_id,
                "session_id": self._session_id,
                "camera_id": self._camera_name,
                "camera_name": self._camera_name,
                "registered_cell_id": self._cell_id,
                "reset_fingerprint": self._fingerprint,
                "sampling_seed": seed,
                "effective_noise_seed": DREAMZERO_CONFIG["effective_noise_seed"],
                "prompt_sha256": _sha(prompt.encode()),
                "request_started_ns": started,
                "request_finished_ns": finished,
                "actions_shape": list(actions.shape),
                "actions_sha256": _sha(actions.tobytes()),
                "actions_hash_domain": "raw_server_actions",
                "provenance": "sgw_wrapper_generated",
            }
            if future is None:
                record["future_status"] = "not_exposed"
            else:
                if hasattr(future, "detach"):
                    future_cpu = future.detach().cpu()
                    if str(future_cpu.dtype) == "torch.bfloat16":
                        future_cpu = future_cpu.float()
                    future_array = future_cpu.numpy()
                    future_encoding = "native_latent_tensor_cpu"
                else:
                    future_array = np.asarray(future)
                    future_encoding = "native_latent_array"
                if (
                    future_array.ndim < 1
                    or not np.issubdtype(future_array.dtype, np.number)
                    or not np.isfinite(future_array).all()
                ):
                    raise AdapterError("D1 native future must be a finite numeric array")
                self.future_dir.mkdir(parents=True, exist_ok=True)
                path = self.future_dir / f"{wrapper_request_id}.npy"
                np.save(path, future_array, allow_pickle=False)
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
                record.update({
                    "future_status": "latent_only_retained",
                    "future_path": str(path),
                    "future_shape": list(future_array.shape),
                    "future_sha256": _sha(path.read_bytes()),
                    "future_encoding": future_encoding,
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


def make_dreamzero_http_server(
    producer: DreamZeroEvidenceProducer,
    *,
    host: str,
    port: int,
    healthcheck: Callable[[], None] | None = None,
) -> ThreadingHTTPServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise AdapterError("owned D1 HTTP server must bind loopback")
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
                if healthcheck is not None:
                    healthcheck()
                packet = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))).decode())
                if not isinstance(packet, Mapping):
                    raise AdapterError("D1 HTTP packet must be an object")
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
