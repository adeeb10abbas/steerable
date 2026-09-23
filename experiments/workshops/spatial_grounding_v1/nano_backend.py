"""Concrete lazy adapter for the audited Cosmos RoboLab service."""

from __future__ import annotations

from dataclasses import replace
import importlib
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from .adapters import AdapterError, NANO_CONFIG
from .producer import _checkpoint_identity, _git_revision


NATIVE_MODULE = "cosmos_framework.scripts.action_policy_server_robolab"
NATIVE_CLASS = "RobolabPolicyService"
NATIVE_ARGS = "RobolabServerArgs"


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise AdapterError(f"{name} is required for the pinned Nano backend")
    return value


def _native_module() -> Any:
    try:
        return importlib.import_module(NATIVE_MODULE)
    except ImportError as exc:
        raise AdapterError(f"pinned Cosmos Nano server module is unavailable: {NATIVE_MODULE}") from exc


class CosmosNanoBackend:
    """Translate SGW native observations to the pinned service's ``infer`` API."""

    def __init__(
        self,
        service: Any,
        *,
        resolved_config: Mapping[str, Any],
        source_root: str,
        checkpoint_path: str,
    ) -> None:
        self.service = service
        self.resolved_config = dict(resolved_config)
        self.source_root = source_root
        self.checkpoint_path = checkpoint_path
        self._lock = threading.Lock()

    def predict(
        self, observation: Mapping[str, Any], prompt: str, sampling_seed: int
    ) -> Mapping[str, Any]:
        native_observation = dict(observation)
        native_observation["prompt"] = prompt
        with self._lock:
            cfg = getattr(self.service, "cfg", None)
            if cfg is None or not hasattr(cfg, "seed") or not hasattr(cfg, "deterministic_seed"):
                raise AdapterError("constructed Cosmos service lacks resolved sampling config")
            self.service.cfg = replace(cfg, seed=int(sampling_seed), deterministic_seed=True)
            result = self.service.infer(native_observation)
        if not isinstance(result, Mapping):
            raise AdapterError("pinned Cosmos service returned a non-mapping result")
        action = np.asarray(result.get("action"), dtype=np.float32)
        if action.shape != (32, 8) or not np.isfinite(action).all():
            raise AdapterError(f"pinned Cosmos service returned action shape {action.shape}, expected (32, 8)")
        output: dict[str, Any] = {"action": action}
        if "video" in result:
            output["future"] = result["video"]
        return output


def build_pinned_nano_backend() -> CosmosNanoBackend:
    """Construct the exact audited ``RobolabPolicyService`` lazily."""
    checkpoint_path = str(Path(_required("SGW01_NANO_CHECKPOINT_PATH")).resolve())
    source_root = str(Path(_required("SGW01_NANO_SOURCE_ROOT")).resolve())
    source_commit = _git_revision(source_root)
    if source_commit != NANO_CONFIG["source_commit"]:
        raise AdapterError("Nano source commit is not pinned; refusing model construction")
    checkpoint_revision, checkpoint_asset = _checkpoint_identity(checkpoint_path)
    if checkpoint_revision != NANO_CONFIG["revision"] or checkpoint_asset != NANO_CONFIG["asset"]:
        raise AdapterError("Nano checkpoint metadata is not the pinned asset/revision")
    native = _native_module()
    try:
        args_type = getattr(native, NATIVE_ARGS)
        service_type = getattr(native, NATIVE_CLASS)
    except AttributeError as exc:
        raise AdapterError("pinned Cosmos module lacks the audited Nano constructor") from exc
    module = native.__file__ if isinstance(getattr(native, "__file__", None), str) else ""
    try:
        Path(module).resolve().relative_to(Path(source_root))
    except (ValueError, OSError):
        raise AdapterError("imported Cosmos module is outside the attested source checkout")
    output_dir = os.environ.get("SGW01_NANO_OUTPUT_DIR", "").strip() or None
    # These are the audited RobolabServerArgs fields, not a generic config
    # dictionary.  Constructor validation is deliberately delegated to the
    # pinned Pydantic model before the service loads the checkpoint.
    try:
        args = args_type(
            checkpoint_path=checkpoint_path,
            hf_revision=NANO_CONFIG["revision"],
            host="127.0.0.1",
            port=0,
            domain_name="droid_lerobot",
            decode_video=True,
            output_dir=output_dir,
            seed=1140,
            deterministic_seed=True,
            guidance=3.0,
            num_steps=4,
            shift=5.0,
            resolution="480",
            conditioning_fps=15.0,
            action_chunk_size=32,
            action_dim=8,
            history_length=1,
        )
        service = service_type(args)
    except Exception as exc:
        raise AdapterError("pinned Cosmos Nano service construction failed") from exc

    cfg = getattr(service, "cfg", None)
    fields = (
        "guidance", "num_steps", "shift", "conditioning_fps",
        "action_chunk_size", "action_dim", "resolution", "action_space",
        "use_state", "history_length", "decode_video", "domain_name",
        "image_height", "image_width",
    )
    if cfg is None or any(not hasattr(cfg, field) for field in fields):
        raise AdapterError("constructed Cosmos service lacks audited resolved config fields")
    if (
        cfg.action_space != "joint_pos"
        or cfg.use_state is not True
        or cfg.action_dim != 8
        or cfg.decode_video is not True
        or cfg.domain_name != "droid_lerobot"
        or (cfg.image_height, cfg.image_width) != (540, 640)
    ):
        raise AdapterError("constructed Cosmos service resolved action interface differs from SGW N3")
    actual_config = {
        "model": "N3",
        "asset": checkpoint_asset,
        "revision": checkpoint_revision,
        "source_commit": source_commit,
        "guidance": cfg.guidance,
        "denoising_steps": cfg.num_steps,
        "shift": cfg.shift,
        "history_length": cfg.history_length,
        "conditioning_fps": cfg.conditioning_fps,
        "resolution": int(cfg.resolution),
        "returned_action_horizon": int(cfg.action_chunk_size),
        "executed_action_horizon": 32,
    }
    if actual_config != dict(NANO_CONFIG):
        raise AdapterError("constructed Cosmos service resolved config differs from SGW N3")
    return CosmosNanoBackend(
        service,
        resolved_config=actual_config,
        source_root=source_root,
        checkpoint_path=checkpoint_path,
    )
