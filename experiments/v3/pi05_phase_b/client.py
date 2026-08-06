#!/usr/bin/env python3
"""Hash-bound π0.5 client for V3-B002 (15×8 action-only contract)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
from policies.pi0_family.client import Pi0DroidJointposClient

from experiments.v3.pi05_phase_b.contract import (
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    AMENDMENT_ID,
    MODEL_ID,
    PROMPTS,
    SEEDS,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


ACTION_SHAPE = (ACTION_CHUNK_STEPS, ACTION_DIM)


class V3B002Pi05Client(Pi0DroidJointposClient):
    """Retain exact server chunks/actions and attest reset before request zero."""

    def __init__(
        self,
        *,
        sampling_seed_base: int,
        action_trace_dir: Path,
        prompt: str,
        release_fingerprint_sha256: str,
        runtime_identity_sha256: str,
        reset_attestation_path: Path,
        ensure_reset_attestation: Callable[[], Path] | None = None,
        **kwargs: Any,
    ) -> None:
        if sampling_seed_base not in SEEDS:
            raise ValueError("V3-B002 seeds are exactly 9400..9426")
        if prompt not in PROMPTS.values():
            raise ValueError("prompt is outside the two registered direct commands")
        super().__init__(
            policy_variant="pi05",
            open_loop_horizon=ACTION_CHUNK_STEPS,
            **kwargs,
        )
        self.sampling_seed_base = sampling_seed_base
        self.action_trace_dir = Path(action_trace_dir)
        self.prompt = prompt
        self.release_fingerprint_sha256 = release_fingerprint_sha256
        self.runtime_identity_sha256 = runtime_identity_sha256
        self.reset_attestation_path = Path(reset_attestation_path)
        self._ensure_reset_attestation_callback = ensure_reset_attestation
        self.request_index = 0
        self.request_sampling_seeds: list[int] = []
        self.returned_action_chunks: list[np.ndarray] = []
        self.executed_actions: list[np.ndarray] = []
        self.reset_fingerprint_sha256: str | None = None
        self._written = False
        self.trace_path: Path | None = None

    def ensure_reset_attestation(self) -> str:
        if not self.reset_attestation_path.is_file() and self._ensure_reset_attestation_callback:
            produced = self._ensure_reset_attestation_callback()
            if Path(produced).resolve() != self.reset_attestation_path.resolve():
                raise RuntimeError("reset callback returned an unexpected attestation path")
        if not self.reset_attestation_path.is_file():
            raise RuntimeError("live reset attestation must exist before policy request zero")
        value = json.loads(self.reset_attestation_path.read_text(encoding="utf-8"))
        expected = {
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": AMENDMENT_ID,
            "model_id": MODEL_ID,
            "environment_seed": self.sampling_seed_base,
            "release_fingerprint_sha256": self.release_fingerprint_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "passed": True,
            "runner_pre_action_reset_calls": 2,
            "physical_reset_calls": 1,
            "settle_gate_runs": 1,
            "duplicate_second_reset_idempotent": True,
            "settle_steps": 60,
            "stable_window_steps": 15,
        }
        for key, wanted in expected.items():
            if value.get(key) != wanted:
                raise RuntimeError(f"live reset attestation mismatch for {key}")
        claimed = value.get("reset_fingerprint_sha256")
        body = {key: child for key, child in value.items() if key != "reset_fingerprint_sha256"}
        if claimed != sha256_bytes(canonical_json_bytes(body)):
            raise RuntimeError("live reset fingerprint does not bind its fields")
        if self.reset_fingerprint_sha256 not in (None, claimed):
            raise RuntimeError("live reset attestation changed during the episode")
        self.reset_fingerprint_sha256 = claimed
        return claimed

    def _query_server(self, request: dict[str, Any]) -> dict[str, Any]:
        reset_hash = self.ensure_reset_attestation()
        request_seed = self.sampling_seed_base * 1000 + self.request_index
        response = super()._query_server(
            {
                **request,
                "sampling_seed": request_seed,
                "v3b002_release_fingerprint_sha256": self.release_fingerprint_sha256,
                "v3b002_reset_fingerprint_sha256": reset_hash,
            }
        )
        if response.get("v2a010_sampling_seed") != request_seed:
            raise RuntimeError("π0.5 server did not attest the exact request seed")
        self.request_sampling_seeds.append(request_seed)
        self.request_index += 1
        return response

    def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
        chunk = np.asarray(super()._unpack_response(response), dtype=np.float32)
        if chunk.shape != ACTION_SHAPE or not np.isfinite(chunk).all():
            raise RuntimeError(f"π0.5 response must be finite {ACTION_SHAPE}, got {chunk.shape}")
        self.returned_action_chunks.append(chunk.copy())
        return chunk

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if instruction != self.prompt:
            raise RuntimeError("static episode prompt changed or does not match released cell")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeError("executed π0.5 action must be finite shape [8]")
        self.executed_actions.append(action.copy())
        return result

    def trace_metadata(self) -> Mapping[str, Any]:
        if not self.executed_actions or not self.returned_action_chunks:
            raise RuntimeError("cannot write an empty behavioral action trace")
        self.ensure_reset_attestation()
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"v3b002_seed{self.sampling_seed_base}"
        actions_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        chunks_path = self.action_trace_dir / f"{stem}_returned_action_chunks.npy"
        metadata_path = self.action_trace_dir / f"{stem}_action_trace.json"
        if any(path.exists() for path in (actions_path, chunks_path, metadata_path)):
            raise FileExistsError("refusing to overwrite retained V3-B002 action evidence")
        actions = np.stack(self.executed_actions).astype(np.float32, copy=False)
        chunks = np.stack(self.returned_action_chunks).astype(np.float32, copy=False)
        np.save(actions_path, actions, allow_pickle=False)
        np.save(chunks_path, chunks, allow_pickle=False)
        value = {
            "schema_version": "vla-wam-shared-v3b-pi05-action-trace-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": AMENDMENT_ID,
            "model_id": MODEL_ID,
            "environment_seed": self.sampling_seed_base,
            "prompt": self.prompt,
            "instruction_controller": "static_episode_prompt",
            "release_fingerprint_sha256": self.release_fingerprint_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "reset_fingerprint_sha256": self.reset_fingerprint_sha256,
            "request_sampling_seeds": self.request_sampling_seeds,
            "future_interface": "actions_only",
            "missing_future_policy": "action_only_interface_not_applicable_never_zero",
            "executed_actions": {
                "path": str(actions_path.resolve()), "sha256": sha256_file(actions_path),
                "bytes": actions_path.stat().st_size, "shape": list(actions.shape),
                "dtype": str(actions.dtype),
            },
            "returned_action_chunks": {
                "path": str(chunks_path.resolve()), "sha256": sha256_file(chunks_path),
                "bytes": chunks_path.stat().st_size, "shape": list(chunks.shape),
                "dtype": str(chunks.dtype),
            },
        }
        metadata_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
        self.trace_path = metadata_path.resolve()
        return {**value, "metadata_path": str(metadata_path.resolve()), "metadata_sha256": sha256_file(metadata_path)}

    def write_trace(self) -> Path | None:
        if self._written or not self.executed_actions:
            return None
        value = self.trace_metadata()
        self._written = True
        return Path(str(value["metadata_path"]))

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self.write_trace()
        super().reset(env_id=env_id)
