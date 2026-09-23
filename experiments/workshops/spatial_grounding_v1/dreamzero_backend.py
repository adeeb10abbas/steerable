"""Fail-closed D1 server binding and testable backend contract.

The pinned DreamZero server source is not vendored in this repository.  This
module therefore verifies identity before loading a caller-supplied native
factory and refuses to guess the server's websocket/model constructor API.
Tests can use ``DreamZeroBackend`` implementations without importing models.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from .adapters import AdapterError, DREAMZERO_CONFIG
from .runtime import _verify_dreamzero_identity


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


def build_pinned_dreamzero_backend() -> DreamZeroBackend:
    """Verify identity, then invoke only an explicitly reviewed native factory."""
    identity = _verify_dreamzero_identity()
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
