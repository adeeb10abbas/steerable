"""Seeded, hash-bearing RoboLab client for current-stack pi0-FAST."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from policies.pi0_family.client import Pi0DroidJointposClient


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V2A008Pi0FastClient(Pi0DroidJointposClient):
    """Require static language, seed every request, and retain action traces."""

    def __init__(
        self,
        *,
        sampling_seed_base: int,
        prompt_family: str,
        requested_relation: str,
        expected_prompt: str,
        action_trace_dir: Path,
        **kwargs: Any,
    ) -> None:
        super().__init__(policy_variant="pi0_fast", **kwargs)
        self.sampling_seed_base = int(sampling_seed_base)
        self.prompt_family = prompt_family
        self.requested_relation = requested_relation
        self.expected_prompt = expected_prompt
        self.action_trace_dir = Path(action_trace_dir)
        self.request_index = 0
        self.request_sampling_seeds: list[int] = []
        self.returned_action_chunks: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.prompt: str | None = None
        self._trace_written = False

    def _query_server(self, request: dict) -> dict:
        sampling_seed = self.sampling_seed_base * 1000 + self.request_index
        seeded_request = {**request, "sampling_seed": sampling_seed}
        response = super()._query_server(seeded_request)
        if response.get("v2a008_sampling_seed") != sampling_seed:
            raise ValueError("Policy server did not attest the requested sampling seed")
        self.request_sampling_seeds.append(sampling_seed)
        self.request_index += 1
        return response

    def _unpack_response(self, response: dict) -> np.ndarray:
        chunk = np.asarray(super()._unpack_response(response), dtype=np.float32)
        if chunk.shape != (10, 8):
            raise ValueError(f"Expected pi0-FAST action chunks shaped (10, 8), got {chunk.shape}")
        self.returned_action_chunks.append(chunk.copy())
        return chunk

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if instruction != self.expected_prompt:
            raise ValueError(
                f"V2-A008 prompt mismatch: {instruction!r} != {self.expected_prompt!r}"
            )
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("V2-A008 requires one episode-static prompt")
        result = super().infer(obs, instruction, env_id=env_id)
        self.executed_actions.append(np.asarray(result["action"], dtype=np.float32).copy())
        return result

    def _write_trace(self) -> None:
        if self._trace_written or not self.executed_actions:
            return
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"seed{self.sampling_seed_base}_{self.prompt_family}_"
            f"{self.requested_relation}"
        )
        action_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        chunks_path = self.action_trace_dir / f"{stem}_returned_action_chunks.npy"
        metadata_path = self.action_trace_dir / f"{stem}_action_trace.json"
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_action_chunks).astype(np.float32, copy=False)
        np.save(action_path, actions, allow_pickle=False)
        np.save(chunks_path, chunks, allow_pickle=False)
        metadata = {
            "schema_version": "vla-wam-v2a008-pi0-current-action-trace-v1",
            "prompt": self.prompt,
            "prompt_family": self.prompt_family,
            "requested_relation": self.requested_relation,
            "sampling_seed_base": self.sampling_seed_base,
            "request_sampling_seeds": self.request_sampling_seeds,
            "executed_actions": {
                "path": str(action_path),
                "sha256": sha256(action_path),
                "count": int(actions.shape[0]),
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
            },
            "returned_action_chunks": {
                "path": str(chunks_path),
                "sha256": sha256(chunks_path),
                "count": int(chunks.shape[0]),
                "shape": list(chunks.shape),
                "dtype": str(chunks.dtype),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self._trace_written = True

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._write_trace()
        super().reset(env_id=env_id)
