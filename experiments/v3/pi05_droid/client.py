#!/usr/bin/env python3
"""Current-stack pi0.5 client with v3 seed attestation and raw action retention.

This is a seed-range extension of the frozen V2-A010 adapter.  It preserves
the OpenPI/RoboLab action path, 15-action open-loop horizon, prompt bytes, and
per-request JAX seed derivation.  The only behavioral additions it accepts are
the prospectively registered v3 seeds 8303--8329.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from policies.pi0_family.client import Pi0DroidJointposClient


MODEL_ID = "pi05_current_stack_droid"
PROMPTS = {
    "Put the Rubik's cube to the left of the bowl.": "left",
    "Put the Rubik's cube to the right of the bowl.": "right",
}
OPEN_LOOP_HORIZON = 15
ACTION_SHAPE = (15, 8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V3Pi05DroidClient(Pi0DroidJointposClient):
    """Retain every server chunk and exact action passed to ``env.step``."""

    def __init__(
        self,
        *,
        sampling_seed_base: int,
        action_trace_dir: Path,
        **kwargs: Any,
    ) -> None:
        if sampling_seed_base not in range(8303, 8330):
            raise ValueError("pi0.5 v3 Phase-A seeds are exactly 8303-8329")
        super().__init__(
            policy_variant="pi05",
            open_loop_horizon=OPEN_LOOP_HORIZON,
            **kwargs,
        )
        self.sampling_seed_base = int(sampling_seed_base)
        self.action_trace_dir = Path(action_trace_dir)
        self.request_index = 0
        self.request_sampling_seeds: list[int] = []
        self.returned_action_chunks: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.prompt: str | None = None
        self.trace_written = False

    def _query_server(self, request: dict[str, Any]) -> dict[str, Any]:
        request_seed = self.sampling_seed_base * 1000 + self.request_index
        response = super()._query_server({**request, "sampling_seed": request_seed})
        if response.get("v2a010_sampling_seed") != request_seed:
            raise ValueError("pi0.5 server did not attest the exact request seed")
        self.request_sampling_seeds.append(request_seed)
        self.request_index += 1
        return response

    def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
        chunk = np.asarray(super()._unpack_response(response), dtype=np.float32)
        if chunk.shape != ACTION_SHAPE or not np.isfinite(chunk).all():
            raise ValueError(
                f"pi0.5 response must be finite with shape {ACTION_SHAPE}, got {chunk.shape}"
            )
        self.returned_action_chunks.append(chunk.copy())
        return chunk

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if instruction not in PROMPTS:
            raise ValueError(f"prompt is outside the frozen direct gate: {instruction!r}")
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("pi0.5 v3 prompt changed during the episode")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError(f"executed pi0.5 action must be finite [8], got {action.shape}")
        self.executed_actions.append(action.copy())
        return result

    def write_trace(self) -> Path | None:
        """Write once, refusing to overwrite any retained behavioral evidence."""

        if self.trace_written or not self.executed_actions:
            return None
        if self.prompt not in PROMPTS:
            raise ValueError("cannot identify the frozen pi0.5 requested relation")
        relation = PROMPTS[self.prompt]
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{self.sampling_seed_base}_direct_command_{relation}"
        actions_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        chunks_path = self.action_trace_dir / f"{stem}_returned_action_chunks.npy"
        metadata_path = self.action_trace_dir / f"{stem}_action_trace.json"
        existing = [path for path in (actions_path, chunks_path, metadata_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite v3 pi0.5 evidence: {existing}")
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_action_chunks).astype(np.float32, copy=False)
        np.save(actions_path, actions, allow_pickle=False)
        np.save(chunks_path, chunks, allow_pickle=False)
        metadata = {
            "schema_version": "vla-wam-shared-v3-pi05-action-trace-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "model_id": MODEL_ID,
            "environment_seed": self.sampling_seed_base,
            "sampling_seed_base": self.sampling_seed_base,
            "prompt": self.prompt,
            "requested_relation": relation,
            "prompt_controller": "episode_static",
            "open_loop_execution_horizon": OPEN_LOOP_HORIZON,
            "request_sampling_seeds": self.request_sampling_seeds,
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
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self.trace_written = True
        return metadata_path

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self.write_trace()
        super().reset(env_id=env_id)
