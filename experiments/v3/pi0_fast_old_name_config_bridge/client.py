#!/usr/bin/env python3
"""Fail-closed RoboLab client for the V3-A002 π0-FAST old-name-config bridge.

The adapter preserves the public RoboLab π0-FAST observation/action path,
executes ten actions from every returned [10, 8] chunk, requires one exact
episode-static direct prompt, seeds every model request, and retains both the
returned chunks and the exact actions sent to env.step.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from policies.pi0_family.client import Pi0DroidJointposClient


MODEL_ID = "pi0_fast_old_name_config_v3a002"
STUDY_ID = "vla_wam_language_steerability_v3"
PROMPTS = {
    "Put the Rubik's cube to the left of the bowl.": "left",
    "Put the Rubik's cube to the right of the bowl.": "right",
}
OPEN_LOOP_HORIZON = 10
ACTION_SHAPE = (10, 8)
SERVER_BRIDGE_ID = "v3a002"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


class V3Pi0FastOldNameConfigClient(Pi0DroidJointposClient):
    """Retain and attest every model request, returned chunk, and executed action."""

    def __init__(
        self,
        *,
        sampling_seed_base: int,
        action_trace_dir: Path,
        **kwargs: Any,
    ) -> None:
        if sampling_seed_base not in range(8310, 8330):
            raise ValueError("V3-A002 bridge seeds are exactly 8310-8329")
        super().__init__(
            policy_variant="pi0_fast",
            open_loop_horizon=OPEN_LOOP_HORIZON,
            **kwargs,
        )
        self.sampling_seed_base = int(sampling_seed_base)
        self.action_trace_dir = Path(action_trace_dir)
        self.request_index = 0
        self.request_attestations: list[dict[str, Any]] = []
        self.returned_action_chunks: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.prompt: str | None = None
        self.trace_written = False

    def _query_server(self, request: dict[str, Any]) -> dict[str, Any]:
        request_seed = self.sampling_seed_base * 1000 + self.request_index
        response = super()._query_server({**request, "sampling_seed": request_seed})
        prompt = request.get("prompt")
        if prompt not in PROMPTS:
            raise ValueError("request prompt is outside the frozen direct pair")
        prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
        expected = {
            "pi0_fast_old_name_config_bridge": SERVER_BRIDGE_ID,
            "sampling_seed": request_seed,
            "prompt_sha256": prompt_sha,
        }
        for key, wanted in expected.items():
            if response.get(key) != wanted:
                raise ValueError(
                    f"π0-FAST old-name-config bridge server attestation mismatch for {key}"
                )
        token_sha = response.get("tokenized_prompt_sha256")
        if not _is_sha256(token_sha):
            raise ValueError("server omitted a valid tokenized-prompt SHA-256")
        self.request_attestations.append(
            {
                "request_index": self.request_index,
                "sampling_seed": request_seed,
                "prompt_sha256": prompt_sha,
                "tokenized_prompt_sha256": token_sha,
            }
        )
        self.request_index += 1
        return response

    def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
        chunk = np.asarray(super()._unpack_response(response), dtype=np.float32)
        if chunk.shape != ACTION_SHAPE or not np.isfinite(chunk).all():
            raise ValueError(
                f"π0-FAST response must be finite with shape {ACTION_SHAPE}, "
                f"got {chunk.shape}"
            )
        if len(self.request_attestations) != len(self.returned_action_chunks) + 1:
            raise ValueError("request/chunk attestation order is inconsistent")
        self.request_attestations[-1]["action_chunk_payload_sha256"] = _sha256_bytes(
            chunk.tobytes(order="C")
        )
        self.returned_action_chunks.append(chunk.copy())
        return chunk

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if instruction not in PROMPTS:
            raise ValueError(f"prompt is outside the frozen direct pair: {instruction!r}")
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("V3-A002 prompt changed during the episode")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(
                f"executed π0-FAST action must be finite with shape [8], got {action.shape}"
            )
        self.executed_actions.append(action.copy())
        return result

    def write_trace(self) -> Path | None:
        """Write retained evidence once and refuse any overwrite."""

        if self.trace_written or not self.executed_actions:
            return None
        if self.prompt not in PROMPTS:
            raise ValueError("cannot identify the frozen requested relation")
        if len(self.returned_action_chunks) != len(self.request_attestations):
            raise ValueError("cannot write an incomplete request/chunk attestation")
        relation = PROMPTS[self.prompt]
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{self.sampling_seed_base}_direct_command_{relation}"
        actions_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        chunks_path = self.action_trace_dir / f"{stem}_returned_action_chunks.npy"
        metadata_path = self.action_trace_dir / f"{stem}_action_trace.json"
        existing = [
            path
            for path in (actions_path, chunks_path, metadata_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite V3-A002 behavioral evidence: {existing}"
            )

        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_action_chunks).astype(np.float32, copy=False)
        np.save(actions_path, actions, allow_pickle=False)
        np.save(chunks_path, chunks, allow_pickle=False)
        metadata = {
            "schema_version": (
                "vla-wam-shared-v3-pi0-fast-old-name-config-action-trace-v1"
            ),
            "study_id": STUDY_ID,
            "model_id": MODEL_ID,
            "bridge_id": SERVER_BRIDGE_ID,
            "environment_seed": self.sampling_seed_base,
            "sampling_seed_base": self.sampling_seed_base,
            "prompt": self.prompt,
            "prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            "requested_relation": relation,
            "prompt_controller": "episode_static",
            "open_loop_execution_horizon": OPEN_LOOP_HORIZON,
            "request_sampling_seeds": [
                row["sampling_seed"] for row in self.request_attestations
            ],
            "request_attestations": self.request_attestations,
            "executed_actions": {
                "path": str(actions_path.resolve()),
                "sha256": _sha256(actions_path),
                "bytes": actions_path.stat().st_size,
                "count": int(actions.shape[0]),
                "shape": list(actions.shape),
                "dtype": str(actions.dtype),
                "definition": "Exact float32 action passed to RoboLab env.step.",
            },
            "returned_action_chunks": {
                "path": str(chunks_path.resolve()),
                "sha256": _sha256(chunks_path),
                "bytes": chunks_path.stat().st_size,
                "count": int(chunks.shape[0]),
                "shape": list(chunks.shape),
                "dtype": str(chunks.dtype),
            },
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        self.trace_written = True
        return metadata_path

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self.write_trace()
        super().reset(env_id=env_id)
