"""RoboLab Cosmos3 client instrumentation for the frozen v2 direct gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from policies.cosmos3.client import Cosmos3Client


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V2Cosmos3Client(Cosmos3Client):
    """Enforce static prompts and retain actions plus every decoded future."""

    def __init__(
        self,
        *,
        sampling_seed_base: int,
        action_trace_dir: Path,
        future_trace_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sampling_seed_base = int(sampling_seed_base)
        self.action_trace_dir = Path(action_trace_dir)
        self.future_trace_dir = Path(future_trace_dir)
        self.request_index = 0
        self.request_records: list[dict[str, Any]] = []
        self.executed_actions: list[np.ndarray] = []
        self.prompt: str | None = None
        self._trace_written = False

    def _relation(self) -> str:
        prompt = (self.prompt or "unknown").lower()
        if " left of " in prompt:
            return "left"
        if " right of " in prompt:
            return "right"
        return "unknown"

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = super()._pack_request(extracted_obs, instruction)
        request["sampling_seed"] = self.sampling_seed_base
        return request

    def _unpack_response(self, response: dict) -> np.ndarray:
        action = np.asarray(response["action"], dtype=np.float32)
        video = np.asarray(response.get("video"), dtype=np.uint8)
        if action.shape != (32, 8):
            raise ValueError(f"Expected Cosmos action chunk (32,8), got {action.shape}")
        if video.ndim != 4 or video.shape[0] != 33 or video.shape[-1] != 3:
            raise ValueError(f"Expected Cosmos 33-frame RGB future, got {video.shape}")
        relation = self._relation()
        stem = f"seed{self.sampling_seed_base}_{relation}_request{self.request_index:03d}"
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        self.future_trace_dir.mkdir(parents=True, exist_ok=True)
        action_path = self.action_trace_dir / f"{stem}_returned_action.npy"
        future_path = self.future_trace_dir / f"{stem}_future.npy"
        np.save(action_path, action, allow_pickle=False)
        np.save(future_path, video, allow_pickle=False)
        self.request_records.append(
            {
                "request_index": self.request_index,
                "requested_sampling_seed": self.sampling_seed_base,
                "action_path": str(action_path),
                "action_sha256": _sha256(action_path),
                "action_shape": list(action.shape),
                "future_path": str(future_path),
                "future_sha256": _sha256(future_path),
                "future_shape": list(video.shape),
            }
        )
        self.request_index += 1
        return action

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("The v2 Cosmos prompt must remain episode-static")
        result = super().infer(obs, instruction, env_id=env_id)
        self.executed_actions.append(np.asarray(result["action"], dtype=np.float32).copy())
        return result

    def _write_trace(self) -> None:
        if self._trace_written or (not self.executed_actions and not self.request_records):
            return
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        relation = self._relation()
        stem = f"seed{self.sampling_seed_base}_{relation}"
        actions_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        metadata_path = self.action_trace_dir / f"{stem}_executed_actions.json"
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        np.save(actions_path, actions, allow_pickle=False)
        metadata = {
            "schema_version": "vla-wam-shared-v2-cosmos3-action-future-trace-v1",
            "prompt": self.prompt,
            "sampling_seed_base": self.sampling_seed_base,
            "executed_actions": {
                "path": str(actions_path),
                "sha256": _sha256(actions_path),
                "count": int(actions.shape[0]),
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
            },
            "requests": self.request_records,
            "future_interface": "decoded RGB uint8 future retained losslessly per request",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self._trace_written = True

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._write_trace()
        super().reset(env_id=env_id)

