"""Concrete lazy bindings for the pinned SGW-01 policy runtimes.

This module performs no model import or server connection at import time.
``create_runtime`` is called only by the worker after the coordinator has
bound endpoints, model assets, and the simulator environment factory.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .adapters import AdapterError, DREAMZERO_CONFIG, NANO_CONFIG


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
    observed_cmdline = proc_cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace").strip()
    expected_cmdline = " ".join(map(str, receipt["server_cmdline"]))
    if observed_cmdline != expected_cmdline:
        raise AdapterError("runtime receipt server command identity mismatch")
    return receipt


def _launch_owned_server() -> subprocess.Popen[bytes]:
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
    return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class _NanoTransport:
    def __init__(self, host: str, port: int, trace_reader: Callable[..., Any]) -> None:
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise AdapterError("pinned OpenPI websocket client is unavailable") from exc
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.trace_reader = trace_reader

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
        return {**response, **trace}


class _OfficialDreamZeroClient:
    """Official conditional DreamZero client with raw 24x8 capture."""

    def __init__(self, host: str, port: int, trace_reader: Callable[..., Any]) -> None:
        try:
            from policies.dreamzero.client import DreamZeroClient
        except ImportError as exc:
            raise AdapterError("pinned DreamZero client is unavailable") from exc

        class Client(DreamZeroClient):
            def __init__(self, **kwargs: Any) -> None:
                self.returned_chunks: list[np.ndarray] = []
                super().__init__(**kwargs)

            def _unpack_response(self, response: Any) -> np.ndarray:
                raw = np.asarray(super()._unpack_response(response), dtype=np.float32)
                if raw.shape != (24, 8) or not np.isfinite(raw).all():
                    raise AdapterError("official D1 response is not finite 24x8")
                self.returned_chunks.append(raw.copy())
                return raw

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
        self.trace_reader = trace_reader

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        observation = request.get("observation")
        if not isinstance(observation, Mapping):
            raise AdapterError("D1 observation must be a mapping for the native client")
        result = self.client.infer(observation, str(request["prompt"]))
        if not self.client.returned_chunks:
            raise AdapterError("official D1 client exposed no returned action chunk")
        actions = self.client.returned_chunks[-1]
        response = {
            "actions": actions,
        }
        trace = self.trace_reader(request=request, response=response)
        if not isinstance(trace, Mapping):
            raise AdapterError("DreamZero trace reader did not return actual request binding")
        return {**response, **trace}

    def reset(self) -> None:
        reset = getattr(self.client, "reset", None)
        if not callable(reset):
            raise AdapterError("native DreamZero client lacks a verified reset method")
        reset()
        self.client.returned_chunks.clear()


@dataclass
class NativeRuntime:
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    environment_factory: Callable[..., Any]
    client: Any
    receipt: Mapping[str, Any]
    server_process: subprocess.Popen[bytes]

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
            self.server_process.terminate()
            try:
                self.server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.server_process.kill()
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
    server_process = _launch_owned_server()
    receipt_path = Path(_required_env("SGW01_RUNTIME_RECEIPT"))
    deadline = time.monotonic() + 60
    while not receipt_path.is_file() and time.monotonic() < deadline:
        if server_process.poll() is not None:
            raise AdapterError("owned policy server exited before writing its runtime receipt")
        time.sleep(0.25)
    receipt = _load_receipt()
    if (
        receipt.get("model") != model
        or receipt.get("config") != dict(expected)
        or receipt.get("checkpoint_revision") != expected["revision"]
    ):
        raise AdapterError("runtime receipt checkpoint revision mismatch")
    if receipt.get("source_commit") != expected["source_commit"]:
        raise AdapterError("runtime receipt source commit mismatch")
    trace_reader = _load_callable(
        _required_env("SGW01_TRACE_READER"),
        "SGW01_TRACE_READER",
    )
    client = (
        _NanoTransport(host, port, trace_reader)
        if model == "N3"
        else _OfficialDreamZeroClient(host, port, trace_reader)
    )
    transport = client
    return NativeRuntime(
        transport=transport,
        environment_factory=environment_factory,
        client=client,
        receipt=receipt,
        server_process=server_process,
    )
