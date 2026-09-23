"""Concrete lazy bindings for the pinned SGW-01 policy runtimes.

This module performs no model import or server connection at import time.
``create_runtime`` is called only by the worker after the coordinator has
bound endpoints, model assets, and the simulator environment factory.
"""

from __future__ import annotations

import ast
import base64
import hashlib
from dataclasses import dataclass
import importlib
import inspect
import json
import os
import signal
import socket
import subprocess
import textwrap
import time
import urllib.error
import urllib.request
import zlib
import struct
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .adapters import AdapterError, DREAMZERO_CONFIG, NANO_CONFIG

D1_ROBOLAB_CLIENT_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
D1_CLIENT_SOURCE_SHA256 = "96de16927536f2b48427a6a2dcc3111d03204e1832e50e159cadf67b3fe956ac"
D1_BASE_CLIENT_SOURCE_SHA256 = "6d357550f55763d6c23dc7d9efb85af9e7821f5b0d0edf76213e6b2d9d7b3f29"


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AdapterError(f"SGW-01 runtime requires {name}")
    return value


def _load_callable(spec: str, label: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise AdapterError(f"{label} must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    try:
        function = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise AdapterError(f"cannot load {label} {spec}: {exc}") from exc
    if not callable(function):
        raise AdapterError(f"{label} is not callable: {spec}")
    return function


def _verify_dreamzero_identity() -> dict[str, str]:
    source_root = Path(_required_env("SGW01_D1_SERVER_SOURCE_ROOT")).resolve()
    client_source_root = Path(_required_env("SGW01_D1_CLIENT_SOURCE_ROOT")).resolve()
    checkpoint_root = Path(_required_env("SGW01_D1_CHECKPOINT_PATH")).resolve()
    source = _verify_git_checkout(
        source_root, DREAMZERO_CONFIG["source_commit"], "DreamZero server"
    )
    client_source = _verify_git_checkout(
        client_source_root, D1_ROBOLAB_CLIENT_COMMIT, "RoboLab client"
    )
    if source != DREAMZERO_CONFIG["source_commit"]:
        raise AdapterError("DreamZero source checkout is not the pinned commit")
    manifest_path = Path(__file__).resolve().parents[3] / (
        "artifacts/vla_wam_shared_v2/pilot/expansion/"
        "dreamzero_official_source_checkpoint_manifest.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError("cannot read pinned DreamZero identity manifest") from exc
    checkpoint = manifest.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise AdapterError("DreamZero identity manifest lacks checkpoint payload")
    expected_files = checkpoint.get("files")
    if not isinstance(expected_files, list):
        raise AdapterError("DreamZero identity manifest lacks file hashes")
    for item in expected_files:
        if not isinstance(item, Mapping):
            raise AdapterError("DreamZero checkpoint manifest entry is invalid")
        relative = item.get("path")
        expected_sha = item.get("sha256")
        expected_bytes = item.get("bytes")
        path = checkpoint_root / str(relative)
        if not isinstance(relative, str) or not isinstance(expected_sha, str) or not path.is_file():
            raise AdapterError(f"DreamZero checkpoint file is missing: {relative}")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            raise AdapterError(f"DreamZero checkpoint byte count mismatch: {relative}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected_sha:
            raise AdapterError(f"DreamZero checkpoint hash mismatch: {relative}")
    return {
        "source_commit": source,
        "client_source_commit": client_source,
        "checkpoint_revision": str(checkpoint.get("revision")),
        "checkpoint_content_sha256": str(checkpoint.get("aggregate_sha256")),
        "source_root": str(source_root),
        "checkpoint_path": str(checkpoint_root),
    }


def _verify_git_checkout(root: Path, expected: str, label: str) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AdapterError(f"{label} source checkout is missing, dirty, or unreadable") from exc
    if revision != expected:
        raise AdapterError(f"{label} source checkout is not pinned")
    return revision


def _proc_start_time(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split(") ", 1)[-1].split()[19]
    except (OSError, IndexError) as exc:
        raise AdapterError("runtime process start identity cannot be read") from exc


def _load_receipt() -> Mapping[str, Any]:
    path = Path(_required_env("SGW01_RUNTIME_RECEIPT"))
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AdapterError(f"cannot read SGW-01 runtime receipt: {path}") from exc
    if not isinstance(receipt, Mapping):
        raise AdapterError("SGW-01 runtime receipt must be an object")
    for key in (
        "server_pid",
        "server_start_time",
        "server_cmdline",
        "source_commit",
        "checkpoint_revision",
        "model",
        "config",
    ):
        if key not in receipt:
            raise AdapterError(f"runtime receipt lacks {key}")
    try:
        os.kill(int(receipt["server_pid"]), 0)
    except (OSError, ValueError) as exc:
        raise AdapterError("runtime receipt server process is not alive") from exc
    proc_cmdline = Path(f"/proc/{int(receipt['server_pid'])}/cmdline")
    if not proc_cmdline.is_file():
        raise AdapterError("runtime process identity cannot be verified on this host")
    if _proc_start_time(int(receipt["server_pid"])) != str(receipt["server_start_time"]):
        raise AdapterError("runtime receipt server start identity mismatch")
    observed_cmdline = proc_cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    expected_cmdline = " ".join(map(str, receipt["server_cmdline"]))
    if observed_cmdline != expected_cmdline:
        raise AdapterError("runtime receipt server command identity mismatch")
    return receipt


def _parse_server_argv() -> list[str]:
    try:
        argv = json.loads(_required_env("SGW01_SERVER_ARGV"))
    except json.JSONDecodeError as exc:
        raise AdapterError("SGW01_SERVER_ARGV must be a JSON argv array") from exc
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise AdapterError("SGW01_SERVER_ARGV must be a non-empty string array")
    return argv


def _assert_port_free(host: str, port: int) -> None:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise AdapterError("owned SGW-01 servers must bind a loopback host")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.2)
        try:
            if probe.connect_ex((host, port)) == 0:
                raise AdapterError(f"SGW-01 policy endpoint is already occupied: {host}:{port}")
        except OSError as exc:
            raise AdapterError("SGW-01 policy port ownership cannot be verified") from exc


def _listening_socket_inodes(port: int) -> set[str]:
    inodes: set[str] = set()
    for proc_net in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        if not proc_net.is_file():
            raise AdapterError("Linux socket ownership cannot be verified")
        for line in proc_net.read_text(encoding="utf-8").splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 10 and fields[1].rsplit(":", 1)[-1] == f"{port:04X}" and fields[3] == "0A":
                inodes.add(fields[9])
    return inodes


def _owned_listener(process: subprocess.Popen[bytes], port: int) -> bool:
    if process.pid is None:
        return False
    inodes = _listening_socket_inodes(port)
    if not inodes:
        return False
    process_group = os.getpgid(process.pid)
    proc_root = Path("/proc")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if os.getpgid(int(entry.name)) != process_group:
                continue
            for fd in (entry / "fd").iterdir():
                target = os.readlink(fd) if fd.is_symlink() else ""
                if target.startswith("socket:[") and target[8:-1] in inodes:
                    return True
        except (OSError, PermissionError, ValueError):
            continue
    return False


def _wait_for_endpoint(process: subprocess.Popen[bytes], host: str, port: int) -> None:
    deadline = time.monotonic() + float(os.environ.get("SGW01_READINESS_TIMEOUT", "15"))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AdapterError("owned policy server exited before readiness")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            try:
                if probe.connect_ex((host, port)) == 0 and _owned_listener(process, port):
                    return
            except OSError:
                pass
        time.sleep(0.1)
    raise AdapterError(f"owned policy server did not become ready at {host}:{port}")


def _read_server_attestation(path: Path, process: subprocess.Popen[bytes]) -> Mapping[str, Any]:
    deadline = time.monotonic() + float(os.environ.get("SGW01_READINESS_TIMEOUT", "15"))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AdapterError("owned policy server exited before writing attestation")
        if path.is_file():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                value = None
            if isinstance(value, Mapping):
                required = ("server_pid", "server_start_time", "model", "config", "source_commit", "checkpoint_revision")
                if all(key in value for key in required):
                    if int(value["server_pid"]) != process.pid:
                        raise AdapterError("server attestation PID differs from owned process")
                    if str(value["server_start_time"]) != _proc_start_time(process.pid):
                        raise AdapterError("server attestation start identity differs from owned process")
                    return value
        time.sleep(0.1)
    raise AdapterError(f"owned policy server did not write attestation: {path}")


def _is_noop_method(method: Callable[..., Any]) -> bool:
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    except (OSError, TypeError, IndentationError):
        return False
    function = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ),
        None,
    )
    if function is None:
        return False
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(
        getattr(body[0], "value", None), ast.Constant
    ) and isinstance(body[0].value.value, str):
        body.pop(0)
    return bool(body) and all(isinstance(node, ast.Pass) for node in body)


def _launch_owned_server(argv: list[str], log_dir: Path) -> subprocess.Popen[bytes]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "server.stdout.log").open("ab")
    stderr = (log_dir / "server.stderr.log").open("ab")
    try:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
    finally:
        stdout.close()
        stderr.close()


def _write_launch_receipt(
    process: subprocess.Popen[bytes],
    *,
    attestation: Mapping[str, Any],
    path: Path,
    argv: list[str],
) -> None:
    if process.pid is None:
        raise AdapterError("owned policy server did not expose a PID")
    start_time = _proc_start_time(process.pid)
    receipt = {
        "server_pid": process.pid,
        "server_start_time": start_time,
        "server_cmdline": argv,
        "model": attestation["model"],
        "config": dict(attestation["config"]),
        "source_commit": attestation["source_commit"],
        "checkpoint_revision": attestation["checkpoint_revision"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _NanoTransport:
    def __init__(self, host: str, port: int, trace_reader: Callable[..., Any]) -> None:
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise AdapterError("pinned OpenPI websocket client is unavailable") from exc
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.trace_reader = trace_reader

    def reset(self) -> None:
        for name in ("reset", "clear_temporal_cache", "reset_episode"):
            reset = getattr(self.client, name, None)
            if callable(reset):
                if _is_noop_method(reset):
                    raise AdapterError(
                        "pinned OpenPI client exposes only a no-op temporal reset method"
                    )
                reset()
                return
        raise AdapterError("pinned OpenPI client exposes no verified temporal reset method")

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise AdapterError("N3 observation must be a mapping for the native client")
        payload = dict(observation)
        payload["prompt"] = request["prompt"]
        payload["sampling_seed"] = request["sampling_seed"]
        response = self.client.infer(payload)
        if not isinstance(response, Mapping):
            raise AdapterError("native Nano server returned a non-mapping response")
        if "actions" not in response and "action" in response:
            response = {**response, "actions": response["action"]}
        trace = self.trace_reader(request=request, response=response)
        if not isinstance(trace, Mapping):
            raise AdapterError("Nano trace reader did not return actual request binding")
        return {**response, "native_trace": dict(trace)}


def _png_rgb(image: Any) -> bytes:
    """Encode one native uint8 RGB frame without a heavyweight image import."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
        raise AdapterError("Nano observation image must be an HWC uint8 RGB array")
    height, width, _ = array.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in array)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=3))
        + chunk(b"IEND", b"")
    )


def _nano_image(observation: Mapping[str, Any]) -> np.ndarray:
    image = observation.get("observation/image")
    if image is None:
        required = (
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
        )
        if not all(key in observation for key in required):
            raise AdapterError(
                "Cosmos HTTP request requires observation/image or all RoboLab camera views"
            )
        wrist = np.asarray(observation[required[0]])
        left = np.asarray(observation[required[1]])
        right = np.asarray(observation[required[2]])
        if any(value.ndim != 3 or value.shape[-1] != 3 for value in (wrist, left, right)):
            raise AdapterError("Nano camera observations must be HWC RGB arrays")
        half_h = wrist.shape[0] // 2
        half_w = wrist.shape[1] // 2
        # The pinned server's compose path uses the wrist view on top and two
        # resized exterior views below.  Refuse silent interpolation changes:
        # callers should provide the canonical composite when exact parity is
        # required.
        if left.shape[:2] != (half_h, half_w) or right.shape[:2] != (half_h, half_w):
            raise AdapterError("Nano multi-view input must already use canonical half resolution")
        image = np.concatenate((wrist, np.concatenate((left, right), axis=1)), axis=0)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise AdapterError("Nano observation image must be HWC RGB")
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and image.min() >= 0 and image.max() <= 1:
            image = np.rint(image * 255).astype(np.uint8)
        else:
            raise AdapterError("Nano observation image must be uint8 or [0,1] float")
    return np.ascontiguousarray(image)


class _NanoHttpTransport:
    """Client for the SGW-owned HTTP wrapper around the pinned native service.

    The audited server has no session state: ``history_length=1`` and every
    request constructs a fresh one-frame sample.  Reset therefore clears only
    transport request bookkeeping, but is accepted only with the attested
    history setting; it is not a guessed OpenPI ``reset()`` call.
    """

    def __init__(
        self,
        host: str,
        port: int,
        trace_reader: Callable[..., Any],
        receipt: Mapping[str, Any],
    ) -> None:
        config = receipt.get("config")
        if not isinstance(config, Mapping) or config.get("history_length") != 1:
            raise AdapterError("Nano HTTP reset requires attested history_length=1")
        self.url = f"http://{host}:{port}/predict"
        self.trace_reader = trace_reader
        self._requests = 0

    def reset(self) -> None:
        camera_name = os.environ.get("SGW01_CAMERA_NAME", "").strip()
        if not camera_name:
            raise AdapterError("SGW01_CAMERA_NAME is required for the owned Nano reset")
        payload = json.dumps({"camera_name": camera_name}).encode("utf-8")
        request = urllib.request.Request(
            self.url.rsplit("/", 1)[0] + "/reset",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AdapterError(
                f"native Cosmos HTTP request failed: HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise AdapterError(f"owned Nano reset request failed: {exc}") from exc
        if not isinstance(result, Mapping) or result.get("status") != "reset":
            raise AdapterError("owned Nano wrapper did not acknowledge reset")
        self._requests = 0

    def close(self) -> None:
        return None

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise AdapterError("N3 observation must be a mapping for the native HTTP server")
        for key in (
            "request_id",
            "request_index",
            "registered_cell_id",
            "camera_id",
            "camera_name",
            "reset_id",
            "reset_fingerprint",
        ):
            if key == "request_index":
                if type(request.get(key)) is not int or request[key] < 0:
                    raise AdapterError("N3 request_index must be a non-negative integer")
            elif not isinstance(request.get(key), str) or not request[key]:
                raise AdapterError(f"N3 request lacks {key}")
        if type(request.get("sampling_seed")) is not int:
            raise AdapterError("N3 sampling_seed must be an integer")
        def json_value(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, Mapping):
                return {str(key): json_value(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [json_value(item) for item in value]
            return value

        payload = {
            "client_request_id": request["request_id"],
            "request_id": request["request_id"],
            "request_index": request["request_index"],
            "registered_cell_id": request["registered_cell_id"],
            "camera_id": request["camera_id"],
            "camera_name": request["camera_name"],
            "reset_id": request["reset_id"],
            "reset_fingerprint": request["reset_fingerprint"],
            "sampling_seed": request["sampling_seed"],
            "observation": json_value(observation),
            "prompt": request["prompt"],
        }
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        http_request = urllib.request.Request(
            self.url,
            data=encoded,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_request, timeout=180) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AdapterError(
                f"native Cosmos HTTP request failed: HTTP {exc.code}: {detail}"
            ) from exc
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise AdapterError(f"native Cosmos HTTP request failed: {exc}") from exc
        if not isinstance(raw, Mapping) or raw.get("error") or "action" not in raw:
            raise AdapterError("native Cosmos HTTP response lacks an action payload")
        actions = np.asarray(raw["action"], dtype=np.float32)
        if actions.shape != (32, 8) or not np.isfinite(actions).all():
            raise AdapterError(f"native Cosmos response action shape is {actions.shape}, expected (32, 8)")
        self._requests += 1
        response: dict[str, Any] = {
            "actions": actions,
            "raw_native_response": dict(raw),
        }
        trace = self.trace_reader(request=request, response=response)
        if not isinstance(trace, Mapping):
            raise AdapterError("Nano trace reader did not return actual request binding")
        response["native_trace"] = dict(trace)
        return response


class _OfficialDreamZeroClient:
    """Official conditional DreamZero client with raw 24x8 capture."""

    def __init__(self, host: str, port: int, trace_reader: Callable[..., Any]) -> None:
        self.identity = _verify_dreamzero_identity()
        try:
            from policies.dreamzero.client import DreamZeroClient
        except ImportError as exc:
            raise AdapterError("pinned DreamZero client is unavailable") from exc
        client_module = importlib.import_module("policies.dreamzero.client")
        client_origin = Path(str(getattr(client_module, "__file__", ""))).resolve()
        try:
            client_origin.relative_to(Path(_required_env("SGW01_D1_CLIENT_SOURCE_ROOT")).resolve())
        except ValueError as exc:
            raise AdapterError("DreamZero client import is outside the pinned source checkout") from exc
        if self.identity["checkpoint_revision"] != DREAMZERO_CONFIG["revision"]:
            raise AdapterError("DreamZero checkpoint revision is not pinned")

        http_url = os.environ.get("SGW01_D1_HTTP_URL", "").strip().rstrip("/")

        class Client(DreamZeroClient):
            def __init__(self, **kwargs: Any) -> None:
                self.returned_chunks: list[np.ndarray] = []
                self.processed_chunks: list[np.ndarray] = []
                self.returned_future: Any | None = None
                self.raw_response: Any | None = None
                super().__init__(**kwargs)

            def _connect_with_retries(self) -> None:
                if http_url:
                    return
                return super()._connect_with_retries()

            def _owned_http(self, endpoint: str, packet: Mapping[str, Any]) -> Any:
                payload = json.dumps(
                    dict(packet),
                    default=lambda value: np.asarray(value).tolist(),
                ).encode("utf-8")
                request = urllib.request.Request(
                    f"{http_url}/{endpoint}",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(request, timeout=180) as response:
                        result = json.loads(response.read().decode("utf-8"))
                except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                    raise AdapterError(f"owned D1 HTTP transport failed: {exc}") from exc
                if not isinstance(result, Mapping) or result.get("error"):
                    raise AdapterError("owned D1 HTTP transport returned an error")
                return result

            def _query_server_owned(self, request: dict[str, Any]) -> Any:
                context = getattr(self, "_sgw_context", {})
                observation = {
                    key: value
                    for key, value in request.items()
                    if key.startswith("observation/")
                }
                packet = {
                    "request_id": context["request_id"],
                    "request_index": context["request_index"],
                    "reset_id": context["reset_id"],
                    "wrapper_reset_id": self._sgw_reset,
                    "camera_name": context["camera_name"],
                    "registered_cell_id": context["registered_cell_id"],
                    "reset_fingerprint": context["reset_fingerprint"],
                    "prompt": request["prompt"],
                    "sampling_seed": context["sampling_seed"],
                    "observation": observation,
                }
                return self._owned_http("predict", packet)

            def reset(self, *, env_id: int | None = None) -> None:
                if not http_url:
                    return super().reset(env_id=env_id)
                response = self._owned_http(
                    "reset",
                    {"camera_name": os.environ.get("SGW01_D1_CAMERA_NAME", "over_shoulder_left_camera")},
                )
                self._sgw_reset = response["reset_id"]
                self._env_session_id.clear()
                super(DreamZeroClient, self).reset(env_id=env_id)
                self.returned_chunks.clear()
                self.processed_chunks.clear()

            def _query_server(self, request: dict[str, Any]) -> Any:
                self.raw_response = (
                    self._query_server_owned(request)
                    if http_url
                    else super()._query_server(request)
                )
                if isinstance(self.raw_response, Mapping):
                    self.returned_future = self.raw_response.get(
                        "future", self.raw_response.get("video")
                    )
                return self.raw_response

            def _unpack_response(self, response: Any) -> np.ndarray:
                raw = np.asarray(super()._unpack_response(response), dtype=np.float32)
                if raw.shape != (24, 8) or not np.isfinite(raw).all():
                    raise AdapterError("official D1 response is not finite 24x8")
                self.returned_chunks.append(raw.copy())
                return raw

            def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
                processed = np.asarray(super()._postprocess_chunk(chunk), dtype=np.float32)
                if processed.shape != (24, 8) or not np.isfinite(processed).all():
                    raise AdapterError("official D1 processed chunk is not finite 24x8")
                self.processed_chunks.append(processed.copy())
                return processed

        self.client = Client(
            remote_host=host,
            remote_port=port,
            open_loop_horizon=8,
            image_height=180,
            image_width=320,
            binarize_gripper=True,
            resize="pad",
            cam2_source="right",
        )
        if getattr(self.client, "open_loop_horizon", None) != DREAMZERO_CONFIG["executed_action_horizon"]:
            raise AdapterError("official D1 client cadence is not pinned to 8 actions")
        self.trace_reader = trace_reader
        self._request_count = 0

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise AdapterError("D1 observation must be a mapping for the native client")
        self.client.returned_chunks.clear()
        self.client.processed_chunks.clear()
        self.client.returned_future = None
        self.client._sgw_context = dict(request)
        started_ns = time.time_ns()
        # RoboLab BaseClient.infer returns one action per call. With the
        # pinned open_loop_horizon=8, eight calls consume the native cache:
        # only the first call performs a server query and captures its fresh
        # 24x8 response; the remaining seven calls take cached actions.
        executed_actions: list[np.ndarray] = []
        for _ in range(DREAMZERO_CONFIG["executed_action_horizon"]):
            result = self.client.infer(observation, str(request["prompt"]))
            if not isinstance(result, Mapping) or "action" not in result:
                raise AdapterError("official D1 infer did not return an action mapping")
            action = np.asarray(result["action"], dtype=np.float32)
            if action.shape != (8,) or not np.isfinite(action).all():
                raise AdapterError("official D1 infer returned an invalid action")
            executed_actions.append(action.copy())
        finished_ns = time.time_ns()
        if len(self.client.returned_chunks) != 1 or len(self.client.processed_chunks) != 1:
            raise AdapterError("official D1 client exposed an unexpected number of returned chunks")
        raw_actions = self.client.returned_chunks[-1]
        actions = self.client.processed_chunks[-1]
        executed = np.stack(executed_actions, axis=0)
        if not np.array_equal(executed, actions[:8]):
            raise AdapterError("official D1 infer actions diverge from processed chunk")
        response = {
            "actions": actions,
            "raw_actions": raw_actions,
            "executed_actions": executed,
            "returned_horizon": 24,
            "executed_horizon": 8,
            "effective_noise_seed": DREAMZERO_CONFIG["effective_noise_seed"],
            "action_guidance": DREAMZERO_CONFIG["action_guidance"],
            "request_started_ns": started_ns,
            "request_finished_ns": finished_ns,
            "request_duration_ns": finished_ns - started_ns,
        }
        if self.client.returned_future is not None:
            response["future"] = self.client.returned_future
            response["future_status"] = "exposed_and_retained"
        else:
            response["future_status"] = "not_exposed"
        self._request_count = getattr(self, "_request_count", 0) + 1
        response["request_count"] = self._request_count
        trace = self.trace_reader(request=request, response=response)
        if not isinstance(trace, Mapping):
            raise AdapterError("DreamZero trace reader did not return actual request binding")
        return {**response, "native_trace": dict(trace)}

    def reset(self) -> None:
        reset = getattr(self.client, "reset", None)
        if not callable(reset):
            raise AdapterError("native DreamZero client lacks a verified reset method")
        if _is_noop_method(reset):
            raise AdapterError("official DreamZero reset method is a no-op")
        reset()
        self.client.returned_chunks.clear()
        self.client.processed_chunks.clear()
        self.client.returned_future = None
        self._request_count = 0


@dataclass
class NativeRuntime:
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    environment_factory: Callable[..., Any]
    client: Any
    receipt: Mapping[str, Any]
    server_process: subprocess.Popen[bytes]
    log_dir: Path

    def reset(self) -> None:
        reset = getattr(self.client, "reset", None)
        if not callable(reset):
            raise AdapterError("native runtime lacks a verified temporal reset method")
        reset()

    def clear_temporal_cache(self) -> None:
        self.reset()

    def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            close()
        if self.server_process.poll() is None:
            os.killpg(self.server_process.pid, signal.SIGTERM)
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.server_process.pid, signal.SIGKILL)
                self.server_process.wait(timeout=10)


def create_runtime(*, model: str, config: Mapping[str, Any]) -> NativeRuntime:
    """Construct the real native client and simulator binding for one model."""

    expected = NANO_CONFIG if model == "N3" else DREAMZERO_CONFIG if model == "D1" else None
    if expected is None or dict(config) != dict(expected):
        raise AdapterError("runtime config does not match the exact SGW-01 model identity")
    host = _required_env(f"SGW01_{model}_HOST")
    try:
        port = int(_required_env(f"SGW01_{model}_PORT"))
    except ValueError as exc:
        raise AdapterError(f"SGW01_{model}_PORT must be an integer") from exc
    environment_factory = _load_callable(
        _required_env("SGW01_ENV_FACTORY"),
        "SGW01_ENV_FACTORY",
    )
    receipt_path = Path(_required_env("SGW01_RUNTIME_RECEIPT"))
    argv = _parse_server_argv()
    trace_spec = os.environ.get(
        "SGW01_TRACE_READER",
        "experiments.workshops.spatial_grounding_v1.trace:read_trace_sidecar",
    )
    if trace_spec != "experiments.workshops.spatial_grounding_v1.trace:read_trace_sidecar":
        raise AdapterError("SGW-01 trace reader must be the concrete JSONL sidecar reader")
    from .trace import read_trace_sidecar

    _assert_port_free(host, port)
    if receipt_path.exists():
        raise AdapterError(f"refusing to reuse an existing runtime receipt: {receipt_path}")
    log_dir = Path(os.environ.get("SGW01_SERVER_LOG_DIR", str(receipt_path.parent / "server-logs")))
    attestation_path = Path(_required_env("SGW01_SERVER_ATTESTATION"))
    server_process: subprocess.Popen[bytes] | None = None
    try:
        server_process = _launch_owned_server(argv, log_dir)
        attestation = _read_server_attestation(attestation_path, server_process)
        if (
            attestation.get("model") != model
            or attestation.get("config") != dict(expected)
            or attestation.get("checkpoint_revision") != expected["revision"]
            or attestation.get("source_commit") != expected["source_commit"]
        ):
            raise AdapterError("server attestation model/config/source identity mismatch")
        _write_launch_receipt(
            server_process, attestation=attestation, path=receipt_path, argv=argv
        )
        _wait_for_endpoint(server_process, host, port)
        receipt = _load_receipt()
        if (
            receipt.get("model") != model
            or receipt.get("config") != dict(expected)
            or receipt.get("checkpoint_revision") != expected["revision"]
            or receipt.get("source_commit") != expected["source_commit"]
        ):
            raise AdapterError("runtime receipt model/config/source identity mismatch")
        client = (
            _NanoHttpTransport(host, port, read_trace_sidecar, receipt)
            if model == "N3"
            else _OfficialDreamZeroClient(host, port, read_trace_sidecar)
        )
        return NativeRuntime(
            transport=client,
            environment_factory=environment_factory,
            client=client,
            receipt=receipt,
            server_process=server_process,
            log_dir=log_dir,
        )
    except Exception:
        if server_process is not None and server_process.poll() is None:
            os.killpg(server_process.pid, signal.SIGTERM)
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(server_process.pid, signal.SIGKILL)
                server_process.wait(timeout=5)
        raise
