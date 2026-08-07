#!/usr/bin/env python3
"""Exact-seeded π0.5 client for one V3-D001 behavioral cell."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from policies.pi0_family.client import Pi0DroidJointposClient

from .contract import ACTION_CHUNK_STEPS, ACTION_DIM, AuthorizedCell, sha256_file


def request_sampling_seed(cell: AuthorizedCell, zero_based_request_index: int) -> int:
    """Return the prospectively released request seed, with no implicit fallback."""

    if type(zero_based_request_index) is not int or zero_based_request_index < 0:
        raise ValueError("zero_based_request_index must be a non-negative integer")
    return cell.sampling_seed_base + zero_based_request_index


class V3D001Pi05Client(Pi0DroidJointposClient):
    def __init__(self, *, cell: AuthorizedCell, trace_dir: Path, release_fingerprint: str, **kwargs: Any) -> None:
        super().__init__(policy_variant="pi05", open_loop_horizon=ACTION_CHUNK_STEPS, **kwargs)
        self.cell = cell
        self.trace_dir = Path(trace_dir)
        self.release_fingerprint = release_fingerprint
        self.request_index = 0
        self.request_sampling_seeds: list[int] = []
        self.returned_chunks: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.trace_path: Path | None = None
        self._written = False

    def _query_server(self, request: dict[str, Any]) -> dict[str, Any]:
        request_seed = request_sampling_seed(self.cell, self.request_index)
        response = super()._query_server({
            **request,
            "sampling_seed": request_seed,
            "v3d001_cell_sha256": self.cell.row["cell_sha256"],
            "v3d001_release_fingerprint_sha256": self.release_fingerprint,
        })
        if response.get("v2a010_sampling_seed") != request_seed:
            raise RuntimeError("π0.5 server did not echo the exact V3-D001 request seed")
        self.request_sampling_seeds.append(request_seed)
        self.request_index += 1
        return response

    def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
        value = np.asarray(super()._unpack_response(response), dtype=np.float32)
        if value.shape != (ACTION_CHUNK_STEPS, ACTION_DIM) or not np.isfinite(value).all():
            raise RuntimeError("V3-D001 π0.5 response must be finite [15,8]")
        self.returned_chunks.append(value.copy())
        return value

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if instruction != self.cell.row["prompt"]:
            raise RuntimeError("V3-D001 static episode prompt changed")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("V3-D001 executed action must be finite [8]")
        self.executed_actions.append(action.copy())
        return result

    def write_trace(self) -> Path | None:
        if self._written or not self.executed_actions:
            return None
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        stem = self.cell.cell_id.replace(":", "__")
        actions_path = self.trace_dir / f"{stem}.executed_actions.npy"
        chunks_path = self.trace_dir / f"{stem}.returned_action_chunks.npy"
        metadata_path = self.trace_dir / f"{stem}.action_trace.json"
        if any(path.exists() for path in (actions_path, chunks_path, metadata_path)):
            raise FileExistsError("refusing to overwrite V3-D001 action evidence")
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_chunks).astype(np.float32, copy=False)
        np.save(actions_path, actions, allow_pickle=False)
        np.save(chunks_path, chunks, allow_pickle=False)
        value = {
            "schema_version": "vla-wam-shared-v3d001-pi05-action-trace-v1",
            "registered_cell_id": self.cell.cell_id,
            "matched_stochastic_block_id": self.cell.block_id,
            "environment_seed": self.cell.environment_seed,
            "shared_policy_sampling_seed_index": self.cell.sampling_index,
            "policy_sampling_seed_base": self.cell.sampling_seed_base,
            "per_request_sampling_seed_rule": "policy_sampling_seed_base + zero_based_request_index",
            "request_sampling_seeds": self.request_sampling_seeds,
            "prompt": self.cell.row["prompt"],
            "instruction_controller": "static_episode_prompt",
            "release_fingerprint_sha256": self.release_fingerprint,
            "executed_actions": {"path": str(actions_path.resolve()), "bytes": actions_path.stat().st_size, "sha256": sha256_file(actions_path), "shape": list(actions.shape), "dtype": str(actions.dtype)},
            "returned_action_chunks": {"path": str(chunks_path.resolve()), "bytes": chunks_path.stat().st_size, "sha256": sha256_file(chunks_path), "shape": list(chunks.shape), "dtype": str(chunks.dtype)},
            "future_interface": "actions_only",
        }
        metadata_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self.trace_path = metadata_path.resolve()
        self._written = True
        return self.trace_path

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self.write_trace()
        super().reset(env_id=env_id)
