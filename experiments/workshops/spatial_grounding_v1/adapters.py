"""Fail-closed policy/runtime adapters for the SGW-01 model branches.

The adapters deliberately stop at an injected transport boundary.  Model
servers and simulators are owned by the cluster worker; this module enforces
the release identity, reset/request ordering, action horizons, and evidence
ownership without importing either model's heavyweight runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import importlib
import os
from typing import Any, Callable, Mapping, Protocol

import numpy as np


ACTION_DIM = 8
ACTION_CAP = 450
NANO_RETURNED_HORIZON = 32
NANO_EXECUTED_HORIZON = 32
DREAMZERO_RETURNED_HORIZON = 24
DREAMZERO_EXECUTED_HORIZON = 8

NANO_CONFIG = {
    "model": "N3",
    "asset": "nvidia/Cosmos3-Nano-Policy-DROID",
    "revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
    "guidance": 3,
    "denoising_steps": 4,
    "shift": 5,
    "history_length": 1,
    "conditioning_fps": 15,
    "resolution": 480,
    "returned_action_horizon": NANO_RETURNED_HORIZON,
    "executed_action_horizon": NANO_EXECUTED_HORIZON,
}

DREAMZERO_CONFIG = {
    "model": "D1",
    "asset": "GEAR-Dreams/DreamZero-DROID",
    "revision": "96ad344138c66e82536422432ad742f015784942",
    "source_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
    "action_path": "official_conditional_no_custom_s2",
    "action_guidance": 1,
    "video_guidance": 5,
    "configured_steps": 16,
    "returned_action_horizon": DREAMZERO_RETURNED_HORIZON,
    "executed_action_horizon": DREAMZERO_EXECUTED_HORIZON,
    "effective_noise_seed": 1140,
}


class AdapterError(ValueError):
    """Raised when an adapter cannot prove a request is scientifically valid."""


class Transport(Protocol):
    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class ResetState:
    """Identity of the physical reset to which requests are bound."""

    reset_id: str
    camera_id: str
    fingerprint: str

    @property
    def camera_name(self) -> str:
        return self.camera_id

    @property
    def camera_fingerprint(self) -> str:
        return self.fingerprint

    def as_dict(self) -> dict[str, Any]:
        return {
            "full_reset": True,
            "reset_id": self.reset_id,
            "camera_name": self.camera_name,
            "camera_fingerprint": self.camera_fingerprint,
            "reset_sha256": self.fingerprint,
        }


@dataclass(frozen=True)
class Prediction:
    """One model response and the executable prefix used by the worker."""

    request_id: str
    request_index: int
    action_step_start: int
    target_action_step: int
    camera_name: str
    reset_id: str
    decoded: bool
    executed_action_count: int
    returned_actions: np.ndarray
    executable_actions: np.ndarray
    returned_horizon: int
    executed_horizon: int
    future: Any
    future_status: str
    raw_request: Mapping[str, Any]
    raw_response: Mapping[str, Any]


def _fingerprint(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    else:
        payload = repr(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _finite_actions(value: Any, horizon: int, label: str) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.shape != (horizon, ACTION_DIM):
        raise AdapterError(f"{label} must have shape [{horizon}, {ACTION_DIM}]")
    if not np.isfinite(actions).all():
        raise AdapterError(f"{label} contains non-finite values")
    return actions.copy()


def _identity(response: Mapping[str, Any], key: str, expected: Any) -> None:
    if response.get(key) != expected:
        raise AdapterError(f"model response identity mismatch for {key}")


class _BaseAdapter:
    config: Mapping[str, Any]
    returned_horizon: int
    executed_horizon: int

    def __init__(
        self,
        *,
        cell_id: str,
        prompt: str,
        transport: Transport,
        runtime: Any | None = None,
        action_cap: int = ACTION_CAP,
        sampling_seed: int | None = None,
    ) -> None:
        if not cell_id or not prompt or not callable(transport):
            raise AdapterError("cell_id, static prompt, and callable transport are required")
        if action_cap != ACTION_CAP:
            raise AdapterError("SGW-01 action cap is exactly 450")
        self.cell_id = cell_id
        self.prompt = prompt
        self.transport = transport
        self.runtime = runtime
        self.sampling_seed = sampling_seed
        self.request_index = 0
        self.executed_steps = 0
        self.reset_state: ResetState | None = None
        self.request_records: list[dict[str, Any]] = []
        self.predictions: list[Prediction] = []

    def reset(
        self,
        *,
        reset_fn: Callable[[], Mapping[str, Any]],
        reset_id: str,
        camera_id: str,
    ) -> ResetState:
        """Perform a complete physical/runtime reset and clear temporal state."""

        if not reset_id or not camera_id:
            raise AdapterError("reset_id and camera_id are required")
        if self.runtime is not None and hasattr(self.runtime, "reset"):
            self.runtime.reset()
        evidence = reset_fn()
        if not isinstance(evidence, Mapping):
            raise AdapterError("reset callback must return identity evidence")
        observed_reset = str(evidence.get("reset_id", reset_id))
        observed_camera = str(evidence.get("camera_name", evidence.get("camera_id", camera_id)))
        if observed_reset != reset_id or observed_camera != camera_id:
            raise AdapterError("reset evidence does not match requested reset/camera")
        fingerprint = str(evidence.get("fingerprint") or _fingerprint(evidence))
        if len(fingerprint) != 64:
            raise AdapterError("reset fingerprint must be a SHA-256 digest")
        self.request_index = 0
        self.executed_steps = 0
        self.request_records.clear()
        self.predictions.clear()
        self.reset_state = ResetState(reset_id, camera_id, fingerprint)
        if self.runtime is not None and hasattr(self.runtime, "clear_temporal_cache"):
            self.runtime.clear_temporal_cache()
        return self.reset_state

    def _require_ready(self, prompt: str, action_step_start: int) -> ResetState:
        if self.reset_state is None:
            raise AdapterError("policy request issued before a full episode reset")
        if prompt != self.prompt:
            raise AdapterError("episode prompt is not byte-identical to the released prompt")
        expected_start = self.executed_steps
        if type(action_step_start) is not int or action_step_start != expected_start:
            raise AdapterError("action_step_start is not the contiguous executed prefix")
        if action_step_start >= ACTION_CAP:
            raise AdapterError("policy request begins at or beyond the 450-action cap")
        return self.reset_state

    def _request(self, observation: Any, prompt: str, action_step_start: int) -> Prediction:
        state = self._require_ready(prompt, action_step_start)
        request_id = f"{self.cell_id}:request:{self.request_index}"
        request: dict[str, Any] = {
            "request_id": request_id,
            "registered_cell_id": self.cell_id,
            "request_index": self.request_index,
            "sampling_seed": self.sampling_seed,
            "prompt": prompt,
            "observation": observation,
            "action_step_start": action_step_start,
            "reset_id": state.reset_id,
            "camera_id": state.camera_id,
            "camera_name": state.camera_id,
            "reset_fingerprint": state.fingerprint,
            "model": self.config["model"],
            "asset": self.config["asset"],
            "revision": self.config["revision"],
        }
        response = self.transport(request)
        if not isinstance(response, Mapping):
            raise AdapterError("policy transport must return a mapping")
        for key, expected in (
            ("request_id", request_id),
            ("registered_cell_id", self.cell_id),
            ("request_index", self.request_index),
            ("reset_id", state.reset_id),
            ("camera_id", state.camera_id),
            ("camera_name", state.camera_id),
            ("reset_fingerprint", state.fingerprint),
        ):
            _identity(response, key, expected)
        self._validate_response(response)
        returned = _finite_actions(response.get("actions"), self.returned_horizon, "returned actions")
        remaining = ACTION_CAP - action_step_start
        execute_count = min(self.executed_horizon, remaining)
        executable = returned[:execute_count].copy()
        future = response.get("future")
        future_status = "exposed_and_retained" if future is not None else "not_exposed"
        executed_action_count = action_step_start + execute_count
        if action_step_start >= executed_action_count:
            raise AdapterError("prediction target_action_step is not before executed prefix")
        prediction = Prediction(
            request_id=request_id,
            request_index=self.request_index,
            action_step_start=action_step_start,
            target_action_step=action_step_start,
            camera_name=state.camera_id,
            reset_id=state.reset_id,
            decoded=future is not None,
            executed_action_count=executed_action_count,
            returned_actions=returned,
            executable_actions=executable,
            returned_horizon=self.returned_horizon,
            executed_horizon=execute_count,
            future=future,
            future_status=future_status,
            raw_request=dict(request),
            raw_response=dict(response),
        )
        self.request_records.append({"request": dict(request), "response": dict(response)})
        self.predictions.append(prediction)
        self.request_index += 1
        return prediction

    def commit_executed(self, count: int) -> None:
        if type(count) is not int or count < 0:
            raise AdapterError("executed action count must be a non-negative integer")
        if count > self.executed_horizon or self.executed_steps + count > ACTION_CAP:
            raise AdapterError("executed action count exceeds the released horizon/cap")
        self.executed_steps += count
        if self.predictions:
            last = self.predictions[-1]
            self.predictions[-1] = replace(
                last,
                executed_horizon=self.executed_steps - last.action_step_start,
                executed_action_count=self.executed_steps,
            )

    def _validate_response(self, response: Mapping[str, Any]) -> None:
        """Hook for model-specific response identity checks."""


class ProductionAdapter:
    """Worker-facing production wrapper around one concrete policy adapter.

    A transport factory is required at runtime.  The default loader resolves a
    user-provided factory lazily from ``SGW01_RUNTIME_FACTORY`` and never
    substitutes a synthetic implementation.
    """

    def __init__(
        self,
        policy_type: type[_BaseAdapter],
        *,
        transport: Transport,
        transport_factory: Callable[..., Transport],
        environment_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.policy_type = policy_type
        self.transport = transport
        self.transport_factory = transport_factory
        self.environment_factory = environment_factory
        self.policy: _BaseAdapter | None = None
        self.environment: Any | None = None

    @staticmethod
    def _cell_value(cell: Any, key: str) -> Any:
        row = getattr(cell, "row", cell)
        if isinstance(row, Mapping):
            if key in row:
                return row[key]
        value = getattr(cell, key, None)
        if value is None:
            raise AdapterError(f"released cell does not expose {key}")
        return value

    def reset(self, cell: Any, recorder: Any) -> dict[str, Any]:
        """Reset the environment and policy temporal state for one attempt."""

        environment = getattr(cell, "environment", None)
        if environment is None and self.environment_factory is not None:
            environment = self.environment_factory(cell=cell)
        if environment is None or not hasattr(environment, "reset"):
            raise AdapterError("production cell must provide Environment.reset()")
        reset_result = environment.reset()
        evidence = getattr(reset_result, "receipt", reset_result)
        if not isinstance(evidence, Mapping):
            raise AdapterError("Environment.reset() must return ResetResult.receipt")
        required_receipt = ("reset_id", "camera_id", "camera_name", "fingerprint")
        if any(not str(evidence.get(key, "")).strip() for key in required_receipt):
            raise AdapterError("reset receipt lacks required physical/camera identity fields")
        if evidence.get("temporal_cache_reset") is not True:
            raise AdapterError("reset receipt lacks temporal_cache_reset=true")
        reset_id = str(evidence["reset_id"])
        camera_name = str(evidence["camera_name"])
        self.policy = self.policy_type(
            cell_id=str(self._cell_value(cell, "cell_id")),
            prompt=str(self._cell_value(cell, "prompt")),
            transport=self.transport,
            sampling_seed=int(self._cell_value(cell, "sampling_seed")),
        )
        reset = self.policy.reset(
            reset_fn=lambda: evidence,
            reset_id=reset_id,
            camera_id=camera_name,
        )
        payload = reset.as_dict()
        if hasattr(recorder, "record_reset"):
            recorder.record_reset(payload)
        else:
            raise AdapterError("recorder must expose record_reset()")
        self.environment = environment
        return payload

    def run_episode(self, cell: Any, recorder: Any, reset: Mapping[str, Any]) -> dict[str, Any]:
        """Run exactly one static-prompt episode through the released cap."""

        if reset.get("full_reset") is not True:
            raise AdapterError("run_episode requires a verified full reset")
        if self.policy is None or self.policy.reset_state is None:
            raise AdapterError("run_episode requires reset() on the same adapter")
        if reset.get("reset_sha256") != self.policy.reset_state.fingerprint:
            raise AdapterError("run_episode reset identity does not match the policy")
        environment = self.environment
        if environment is None or not all(
            hasattr(environment, name) for name in ("step", "snapshot", "render_viewport")
        ):
            raise AdapterError("production environment lacks required execution methods")
        policy_observation = getattr(environment, "policy_observation", None)
        if not callable(policy_observation):
            raise AdapterError("environment must expose policy_observation() separate from snapshot()")
        prompt = str(self._cell_value(cell, "prompt"))
        safety_reason: str | None = None
        episode_mapping: list[dict[str, Any]] = []
        while self.policy.executed_steps < ACTION_CAP:
            observation = policy_observation()
            request_id = f"{self.policy.cell_id}:request:{self.policy.request_index}"
            metadata = {
                "request_id": request_id,
                "request_index": self.policy.request_index,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
                "prompt_sha256": str(self._cell_value(cell, "prompt_sha256")),
                "reset_sha256": reset["reset_sha256"],
                "camera_fingerprint": reset["camera_fingerprint"],
                "camera_name": reset["camera_name"],
                "target_action_step": self.policy.executed_steps,
            }
            if not hasattr(recorder, "request"):
                raise AdapterError("recorder must expose request() for raw request persistence")
            recorder.request(metadata)
            prediction = self.policy.predict(
                observation, prompt, action_step_start=self.policy.executed_steps
            )
            mapping_start = len(episode_mapping)
            for action in prediction.executable_actions:
                step_result = environment.step(action)
                step_index = self.policy.executed_steps
                scoring_snapshot = environment.snapshot()
                viewport_frame = environment.render_viewport()
                self.policy.commit_executed(1)
                episode_mapping.append(
                    {
                        "action_step": step_index,
                        "action": np.asarray(action, dtype=np.float32).copy(),
                        "state": scoring_snapshot,
                        "viewport_frame": viewport_frame,
                        "step_result": step_result,
                    }
                )
                if isinstance(step_result, Mapping) and step_result.get("safety_terminated") is True:
                    safety_reason = str(step_result.get("termination_reason") or "safety_terminal")
                    break
            if not hasattr(recorder, "prediction"):
                raise AdapterError("recorder must expose prediction()")
            prediction.raw_request["episode_mapping"] = episode_mapping[mapping_start:]
            recorder.prediction(prediction)
            if safety_reason is not None:
                break
            if self.policy.executed_steps >= ACTION_CAP:
                break
        # Semantic outcome classification belongs to the worker/scorer.  The
        # adapter returns execution evidence and never invents success/failure.
        return {
            "status": "censored" if safety_reason is not None else "valid_model_failure",
            "termination_reason": safety_reason or "action_cap",
            "safety_terminated": safety_reason is not None,
            "executed_action_count": self.policy.executed_steps,
            "prediction_count": len(self.policy.predictions),
            "episode_mapping": episode_mapping,
        }

    def close(self) -> None:
        if self.environment is not None and hasattr(self.environment, "close"):
            self.environment.close()
        self.environment = None
        self.policy = None


class NanoPolicyAdapter(_BaseAdapter):
    """Cosmos3 Nano N3 adapter with the exact SGW-01 configuration."""

    config = NANO_CONFIG
    returned_horizon = NANO_RETURNED_HORIZON
    executed_horizon = NANO_EXECUTED_HORIZON

    def predict(self, observation: Any, prompt: str, *, action_step_start: int) -> Prediction:
        return self._request(observation, prompt, action_step_start)


class DreamZeroPolicyAdapter(_BaseAdapter):
    """Official DreamZero D1 conditional-action adapter.

    The historical V2-A015 negative-branch `s=2` overlay is deliberately
    rejected; it is not the SGW-01 D1 action path.
    """

    config = DREAMZERO_CONFIG
    returned_horizon = DREAMZERO_RETURNED_HORIZON
    executed_horizon = DREAMZERO_EXECUTED_HORIZON

    def __init__(self, *, action_guidance: float = 1, **kwargs: Any) -> None:
        if action_guidance != 1:
            raise AdapterError("SGW-01 D1 requires the official conditional action path")
        super().__init__(**kwargs)

    def predict(self, observation: Any, prompt: str, *, action_step_start: int) -> Prediction:
        return self._request(observation, prompt, action_step_start)

    def _validate_response(self, response: Mapping[str, Any]) -> None:
        if response.get("effective_noise_seed") != 1140:
            raise AdapterError("D1 response does not expose the pinned effective noise seed")
        if response.get("action_guidance", 1) != 1:
            raise AdapterError("D1 response uses custom action guidance")


def make_adapter(model: str, **kwargs: Any) -> NanoPolicyAdapter | DreamZeroPolicyAdapter:
    """Construct only one of the two released SGW-01 adapters."""

    if model == "N3":
        return NanoPolicyAdapter(**kwargs)
    if model == "D1":
        return DreamZeroPolicyAdapter(**kwargs)
    raise AdapterError(f"unsupported SGW-01 model: {model}")


def _load_transport_factory() -> Callable[..., Transport]:
    spec = os.environ.get("SGW01_RUNTIME_FACTORY", "")
    if not spec:
        from .runtime import create_runtime
        return create_runtime
    if ":" not in spec:
        raise AdapterError("SGW01_RUNTIME_FACTORY must use module:function syntax")
    module_name, function_name = spec.split(":", 1)
    try:
        factory = getattr(importlib.import_module(module_name), function_name)
    except (ImportError, AttributeError) as exc:
        raise AdapterError(f"cannot load pinned SGW-01 runtime factory {spec}: {exc}") from exc
    if not callable(factory):
        raise AdapterError("SGW-01 runtime factory is not callable")
    return factory


def load_production_adapter(model: str) -> ProductionAdapter:
    """Load a real pinned runtime adapter; never returns a fake/test transport."""

    if model not in {"N3", "D1"}:
        raise AdapterError(f"unsupported SGW-01 model: {model}")
    factory = _load_transport_factory()
    config = dict(NANO_CONFIG if model == "N3" else DREAMZERO_CONFIG)
    try:
        runtime = factory(model=model, config=config)
    except TypeError as exc:
        raise AdapterError("pinned runtime factory must accept model and config") from exc
    environment_factory = getattr(runtime, "environment_factory", None)
    transport = getattr(runtime, "transport", runtime)
    if not callable(transport):
        raise AdapterError("pinned runtime factory did not return a callable transport")
    policy_type = NanoPolicyAdapter if model == "N3" else DreamZeroPolicyAdapter
    return ProductionAdapter(
        policy_type,
        transport=transport,
        transport_factory=factory,
        environment_factory=environment_factory,
    )
