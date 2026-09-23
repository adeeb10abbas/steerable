"""Fail-closed D1 server binding and testable backend contract.

The pinned DreamZero server source is not vendored in this repository.  This
module therefore verifies identity before loading a caller-supplied native
factory and refuses to guess the server's websocket/model constructor API.
Tests can use ``DreamZeroBackend`` implementations without importing models.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import hashlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .adapters import AdapterError, DREAMZERO_CONFIG
from .runtime import _verify_dreamzero_identity

D1_SERVER_EXPORT_SHA256 = "422f6181762756f71fbe1c1513e0076482e4aab07dfab1a11402176c11df870a"
D1_SERVER_SURFACE = {
    "eval_utils/policy_server.py": "5c541300759ac211aa00639223e707c80a12bf548520a70981162b8a0c534117",
    "eval_utils/policy_client.py": "4f9f062bdc5bec081a459b75a29825c0438f15d22da601e20305e843cd05b5f1",
    "groot/vla/model/n1_5/sim_policy.py": "c7b692b84a03a70adc7e0d21fb7632a9866285645e8d43c916100e6f5fb7497a",
}


class DreamZeroBackend(Protocol):
    source_root: str
    checkpoint_path: str
    resolved_config: Mapping[str, Any]

    def predict(
        self,
        observation: Mapping[str, Any],
        prompt: str,
        sampling_seed: int,
        *,
        action_guidance: int,
        video_guidance: int,
        steps: int,
        session_id: str | None,
    ) -> Mapping[str, Any]:
        """Return native actions and optional exposed future evidence."""

    def reset(self, session_id: str | None) -> Mapping[str, Any] | None:
        """Evict the native session and clear temporal server state."""


@dataclass(frozen=True)
class DreamZeroServerConfig:
    checkpoint_path: str
    source_root: str
    source_commit: str
    checkpoint_revision: str
    action_guidance: int = 1
    video_guidance: int = 5
    steps: int = 16
    returned_horizon: int = 24
    executed_horizon: int = 8
    effective_noise_seed: int = 1140


class _FactoryBackend:
    def __init__(self, native: Any, config: DreamZeroServerConfig) -> None:
        native_config = getattr(native, "resolved_config", None)
        if not isinstance(native_config, Mapping) or dict(native_config) != dict(DREAMZERO_CONFIG):
            raise AdapterError(
                "reviewed D1 native binding must expose its constructed resolved_config"
            )
        self.native = native
        self.source_root = config.source_root
        self.checkpoint_path = config.checkpoint_path
        self.resolved_config = dict(native_config)

    def predict(self, observation: Mapping[str, Any], prompt: str, sampling_seed: int, **kwargs: Any) -> Mapping[str, Any]:
        result = self.native.predict(
            observation,
            prompt,
            sampling_seed,
            **kwargs,
        )
        if not isinstance(result, Mapping):
            raise AdapterError("pinned D1 native factory returned a non-mapping result")
        return result

    def reset(self, session_id: str | None) -> Mapping[str, Any] | None:
        result = self.native.reset(session_id)
        if result is not None and not isinstance(result, Mapping):
            raise AdapterError("pinned D1 native reset returned a non-mapping result")
        return result


def _load_factory(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise AdapterError("SGW01_D1_SERVER_FACTORY must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise AdapterError("cannot load the reviewed D1 server factory") from exc
    if not callable(factory):
        raise AdapterError("SGW01_D1_SERVER_FACTORY is not callable")
    return factory


def _verify_exported_server_surface(source_root: Path) -> None:
    """Verify the source files that define the reviewed websocket boundary."""
    for relative, expected in D1_SERVER_SURFACE.items():
        path = source_root / relative
        if not path.is_file():
            raise AdapterError(f"pinned D1 server source file is missing: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise AdapterError(f"pinned D1 server source file hash mismatch: {relative}")


def build_pinned_dreamzero_backend() -> DreamZeroBackend:
    """Verify identity, then invoke only an explicitly reviewed native factory."""
    identity = _verify_dreamzero_identity()
    _verify_exported_server_surface(Path(identity["source_root"]))
    factory_spec = os.environ.get("SGW01_D1_SERVER_FACTORY", "").strip()
    if not factory_spec:
        raise AdapterError(
            "D1 native server factory is unavailable; export the exact reviewed "
            "DreamZero websocket/server binding before model construction"
        )
    config = DreamZeroServerConfig(
        checkpoint_path=identity["checkpoint_path"],
        source_root=identity["source_root"],
        source_commit=identity["source_commit"],
        checkpoint_revision=identity["checkpoint_revision"],
    )
    try:
        native = _load_factory(factory_spec)(config)
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError("reviewed D1 native server factory failed") from exc
    if not callable(getattr(native, "predict", None)) or not callable(getattr(native, "reset", None)):
        raise AdapterError("reviewed D1 native binding must provide predict and reset")
    return _FactoryBackend(native, config)


def validate_dreamzero_result(result: Mapping[str, Any]) -> tuple[np.ndarray, Any | None]:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.shape != (24, 8) or not np.isfinite(actions).all():
        raise AdapterError("D1 native server must expose finite 24x8 actions")
    return actions, result.get("future", result.get("video"))
