"""Concrete lazy bindings for the pinned SGW-01 policy runtimes.

This module performs no model import or server connection at import time.
``create_runtime`` is called only by the worker after the coordinator has
bound endpoints, model assets, and the simulator environment factory.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
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


class _NanoTransport:
    def __init__(self, host: str, port: int) -> None:
        try:
            from openpi_client import websocket_client_policy
        except ImportError as exc:
            raise AdapterError("pinned OpenPI websocket client is unavailable") from exc
        self.client = websocket_client_policy.WebsocketClientPolicy(host, port)

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
        return {
            **response,
            "request_id": request["request_id"],
            "registered_cell_id": request["registered_cell_id"],
            "request_index": request["request_index"],
            "reset_id": request["reset_id"],
            "camera_id": request["camera_id"],
            "camera_name": request["camera_name"],
            "reset_fingerprint": request["reset_fingerprint"],
        }


class _OfficialDreamZeroClient:
    """Official conditional DreamZero client with raw 24x8 capture."""

    def __init__(self, host: str, port: int) -> None:
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
            "future": result.get("future") if isinstance(result, Mapping) else None,
            "effective_noise_seed": 1140,
            "action_guidance": 1,
            "request_id": request["request_id"],
            "registered_cell_id": request["registered_cell_id"],
            "request_index": request["request_index"],
            "reset_id": request["reset_id"],
            "camera_id": request["camera_id"],
            "camera_name": request["camera_name"],
            "reset_fingerprint": request["reset_fingerprint"],
        }
        if isinstance(result, Mapping) and result.get("future") is not None:
            response["future"] = result["future"]
        return response

    def reset(self) -> None:
        if hasattr(self.client, "reset"):
            self.client.reset()
        self.client.returned_chunks.clear()


@dataclass
class NativeRuntime:
    transport: Callable[[Mapping[str, Any]], Mapping[str, Any]]
    environment_factory: Callable[..., Any]
    client: Any

    def reset(self) -> None:
        if hasattr(self.client, "reset"):
            self.client.reset()

    def clear_temporal_cache(self) -> None:
        self.reset()


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
    client = _NanoTransport(host, port) if model == "N3" else _OfficialDreamZeroClient(host, port)
    transport = client
    return NativeRuntime(
        transport=transport,
        environment_factory=environment_factory,
        client=client,
    )
