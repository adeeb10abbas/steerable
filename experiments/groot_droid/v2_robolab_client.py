"""RoboLab GR00T client additions for the frozen v2 direct gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from policies.gr00t.client import (
    GR00TDroidJointposClient,
    _MsgSerializer,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V2GR00TDroidJointposClient(GR00TDroidJointposClient):
    """Add the frozen seed schedule and retain every executed action."""

    def __init__(
        self,
        *,
        sampling_seed_base: int,
        action_trace_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.sampling_seed_base = int(sampling_seed_base)
        self.action_trace_dir = Path(action_trace_dir)
        self.request_index = 0
        self.request_sampling_seeds: list[int] = []
        self.returned_action_chunks: list[np.ndarray] = []
        self.returned_action_modalities: list[dict[str, np.ndarray]] = []
        self.executed_actions: list[np.ndarray] = []
        self.prompt: str | None = None

    def _query_server(self, request: dict) -> tuple:
        sampling_seed = self.sampling_seed_base * 1000 + self.request_index
        rpc = {
            "endpoint": "get_action",
            "data": {
                "observation": request,
                "options": {"sampling_seed": sampling_seed},
            },
        }
        self.client.socket.send(_MsgSerializer.to_bytes(rpc))
        message = self.client.socket.recv()
        response = _MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")
        self.request_sampling_seeds.append(sampling_seed)
        self.request_index += 1
        return tuple(response)

    def _unpack_response(self, response: tuple) -> np.ndarray:
        action_dict = response[0]
        raw_modalities = {
            str(key): np.asarray(value, dtype=np.float32).copy()
            for key, value in action_dict.items()
        }
        required = {
            "action.eef_9d",
            "action.gripper_position",
            "action.joint_position",
        }
        if not required.issubset(raw_modalities):
            raise ValueError(
                "GR00T response is missing required raw action modalities: "
                f"{sorted(required - raw_modalities.keys())}"
            )
        chunk = super()._unpack_response(response)
        if chunk.shape != (40, 8):
            raise ValueError(
                "The frozen GR00T DROID contract requires returned action chunks "
                f"with shape (40, 8), got {chunk.shape}"
            )
        self.returned_action_chunks.append(
            np.asarray(chunk, dtype=np.float32).copy()
        )
        self.returned_action_modalities.append(raw_modalities)
        return chunk

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("The v2 GR00T prompt must remain episode-static")
        result = super().infer(obs, instruction, env_id=env_id)
        self.executed_actions.append(np.asarray(result["action"], dtype=np.float32).copy())
        return result

    def _relation(self) -> str:
        prompt = (self.prompt or "unknown").lower()
        if " left of " in prompt:
            return "left"
        if " right of " in prompt:
            return "right"
        return "unknown"

    def _write_trace(self) -> None:
        if not self.executed_actions and not self.returned_action_chunks:
            return
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{self.sampling_seed_base}_{self._relation()}"
        array_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        chunks_path = self.action_trace_dir / f"{stem}_returned_action_chunks.npy"
        modalities_path = (
            self.action_trace_dir / f"{stem}_returned_action_modalities.npz"
        )
        metadata_path = self.action_trace_dir / f"{stem}_executed_actions.json"
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_action_chunks).astype(np.float32, copy=False)
        np.save(array_path, actions, allow_pickle=False)
        np.save(chunks_path, chunks, allow_pickle=False)
        modality_keys = sorted(self.returned_action_modalities[0])
        if any(sorted(item) != modality_keys for item in self.returned_action_modalities):
            raise ValueError("Returned GR00T action modality keys changed within an episode")
        stacked_modalities = {
            key: np.stack([item[key] for item in self.returned_action_modalities])
            for key in modality_keys
        }
        np.savez(modalities_path, **stacked_modalities)
        metadata = {
            "schema_version": "vla-wam-shared-v2-groot-action-trace-v1",
            "prompt": self.prompt,
            "sampling_seed_base": self.sampling_seed_base,
            "request_sampling_seeds": self.request_sampling_seeds,
            "path": str(array_path),
            "sha256": _sha256(array_path),
            "count": int(actions.shape[0]),
            "shape": list(actions.shape),
            "dtype": str(actions.dtype),
            "returned_action_chunks": {
                "path": str(chunks_path),
                "sha256": _sha256(chunks_path),
                "count": int(chunks.shape[0]),
                "shape": list(chunks.shape),
                "dtype": str(chunks.dtype),
                "note": "Raw model futures before client gripper binarization.",
            },
            "returned_action_modalities": {
                "path": str(modalities_path),
                "sha256": _sha256(modalities_path),
                "keys": {
                    key: {
                        "shape": list(value.shape),
                        "dtype": str(value.dtype),
                    }
                    for key, value in stacked_modalities.items()
                },
                "note": "All raw future modalities returned by the server.",
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._write_trace()
        super().reset(env_id=env_id)
