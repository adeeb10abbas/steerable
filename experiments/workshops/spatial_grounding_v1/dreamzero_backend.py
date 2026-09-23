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
import json
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
D1_14B_ENTRYPOINT_SHA256 = "7ef17f66064bac8defafc1a84551089b124546729a98be8c0515b33d2e159d48"


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
        if not isinstance(native_config, Mapping):
            raise AdapterError(
                "reviewed D1 native binding must expose constructed native configuration"
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


def _verify_14b_entrypoint(source_root: Path) -> None:
    path = source_root / "socket_test_optimized_AR.py"
    if not path.is_file():
        raise AdapterError("official DreamZero 14B entrypoint is missing")
    if hashlib.sha256(path.read_bytes()).hexdigest() != D1_14B_ENTRYPOINT_SHA256:
        raise AdapterError("official DreamZero 14B entrypoint hash mismatch")


def _read_native_checkpoint_config(checkpoint_path: str) -> dict[str, Any]:
    path = Path(checkpoint_path) / "config.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        head = payload["action_head_cfg"]["config"]
        horizon = int(payload["action_horizon"])
        action_dim = int(payload["action_dim"])
        steps = int(head["num_inference_timesteps"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise AdapterError("D1 checkpoint config.json lacks audited action-head fields") from exc
    return {
        "model": "D1",
        "asset": DREAMZERO_CONFIG["asset"],
        "revision": DREAMZERO_CONFIG["revision"],
        "source_commit": DREAMZERO_CONFIG["source_commit"],
        "action_path": DREAMZERO_CONFIG["action_path"],
        "checkpoint_num_inference_timesteps": steps,
        "returned_action_horizon": horizon,
        "checkpoint_native_action_dim": action_dim,
        "native_checkpoint_config": str(path),
    }


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


class OfficialDreamZero14BBackend:
    """Adapter around the exported ``ARDroidRoboarenaPolicy`` route."""

    def __init__(
        self,
        policy: Any,
        *,
        source_root: str,
        checkpoint_path: str,
        resolved_config: Mapping[str, Any],
        native_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        native_config = dict(resolved_config)
        for key, expected in DREAMZERO_CONFIG.items():
            if native_config.get(key) != expected:
                raise AdapterError(f"constructed 14B policy config differs at {key}")
        self.policy = policy
        self.source_root = source_root
        self.checkpoint_path = checkpoint_path
        self.resolved_config = dict(DREAMZERO_CONFIG)
        self.native_metadata = dict(native_metadata or {})

    def predict(self, observation: Mapping[str, Any], prompt: str, sampling_seed: int, **kwargs: Any) -> Mapping[str, Any]:
        del sampling_seed, kwargs
        native_observation = dict(observation)
        native_observation["prompt"] = prompt
        actions = np.asarray(self.policy.infer(native_observation), dtype=np.float32)
        output: dict[str, Any] = {"actions": actions}
        futures = getattr(self.policy, "video_across_time", [])
        if futures:
            output["future"] = futures[-1]
        session_id = observation.get("session_id")
        if isinstance(session_id, str):
            output["session_id"] = session_id
        return output

    def reset(self, session_id: str | None) -> Mapping[str, Any]:
        self.policy.reset({"session_id": session_id})
        return {"evicted_session_id": session_id, "route": "ARDroidRoboarenaPolicy.reset"}


def build_official_14b_dreamzero_backend() -> OfficialDreamZero14BBackend:
    """Construct the exact AR 14B route after identity and config checks."""
    identity = _verify_dreamzero_identity()
    source_root = Path(identity["source_root"])
    _verify_exported_server_surface(source_root)
    _verify_14b_entrypoint(source_root)
    model_path = os.environ.get("SGW01_D1_MODEL_PATH", "").strip()
    if not model_path:
        raise AdapterError("SGW01_D1_MODEL_PATH is required for explicit 14B construction")
    if Path(model_path).resolve() != Path(identity["checkpoint_path"]).resolve():
        raise AdapterError("D1 model path differs from the attested checkpoint path")
    native_config = _read_native_checkpoint_config(identity["checkpoint_path"])
    if native_config["returned_action_horizon"] != DREAMZERO_CONFIG["returned_action_horizon"]:
        raise AdapterError("attested D1 checkpoint action horizon is not 24")
    try:
        module = importlib.import_module("socket_test_optimized_AR")
        module_origin = Path(str(getattr(module, "__file__", ""))).resolve()
        module_origin.relative_to(source_root)
        device_mesh = module.init_mesh()
        signal_group = module.dist.new_group(
            backend="gloo",
            timeout=module.datetime.timedelta(seconds=50000),
        )
        policy = module.GrootSimPolicy(
            embodiment_tag=module.EmbodimentTag("oxe_droid"),
            model_path=model_path,
            device="cuda" if module.torch.cuda.is_available() else "cpu",
            device_mesh=device_mesh,
        )
        wrapper = module.ARDroidRoboarenaPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            output_dir=os.environ.get("SGW01_D1_VIDEO_OUTPUT_DIR") or None,
        )
        head = _find_action_head(policy)
        observed = _observe_action_head(head)
        if observed["num_inference_steps"] != DREAMZERO_CONFIG["configured_steps"]:
            raise AdapterError("constructed D1 action head sampler steps are not 16")
        if observed["seed"] != DREAMZERO_CONFIG["effective_noise_seed"]:
            raise AdapterError("constructed D1 action head seed is not 1140")
        if observed["cfg_scale"] != 5.0:
            raise AdapterError("constructed D1 action head cfg_scale is not 5.0")
    except AdapterError:
        raise
    except Exception as exc:
        raise AdapterError("official exported DreamZero 14B construction failed") from exc
    return OfficialDreamZero14BBackend(
        wrapper,
        source_root=str(source_root),
        checkpoint_path=identity["checkpoint_path"],
        resolved_config=DREAMZERO_CONFIG,
        native_metadata={
            "native_checkpoint_num_inference_timesteps": native_config[
                "checkpoint_num_inference_timesteps"
            ],
            "native_checkpoint_action_dim": native_config["checkpoint_native_action_dim"],
            "native_sampler_steps": observed["num_inference_steps"],
            "native_sampler_seed": observed["seed"],
            "native_sampler_cfg_scale": observed["cfg_scale"],
            "native_output_action_dim": 8,
        },
    )


def _find_action_head(policy: Any) -> Any:
    roots = [policy, getattr(policy, "_policy", None), getattr(policy, "trained_model", None)]
    for root in roots:
        if root is None:
            continue
        for name in ("trained_model", "model", "_model"):
            model = getattr(root, name, None)
            head = getattr(model, "action_head", None)
            if head is not None:
                return head
        head = getattr(root, "action_head", None)
        if head is not None:
            return head
    raise AdapterError("constructed D1 policy does not expose trained_model.action_head")


def _observe_action_head(head: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for key in ("num_inference_steps", "seed", "cfg_scale"):
        value = getattr(head, key, None)
        if value is None:
            raise AdapterError(f"constructed D1 action head lacks {key}")
        values[key] = int(value) if key != "cfg_scale" else float(value)
    return values




def validate_dreamzero_result(result: Mapping[str, Any]) -> tuple[np.ndarray, Any | None]:
    actions = np.asarray(result.get("actions"), dtype=np.float32)
    if actions.shape != (24, 8) or not np.isfinite(actions).all():
        raise AdapterError("D1 native server must expose finite 24x8 actions")
    return actions, result.get("future", result.get("video"))
