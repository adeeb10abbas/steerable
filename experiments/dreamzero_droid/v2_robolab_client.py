#!/usr/bin/env python3
"""Frozen DreamZero DROID client with measurement-only v2 trace retention."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from policies.dreamzero.client import DreamZeroClient


LEFT = "Put the Rubik's cube to the left of the bowl."
RIGHT = "Put the Rubik's cube to the right of the bowl."
PROMPTS = {LEFT: "left", RIGHT: "right"}
OFFICIAL_RETURNED_SHAPE = (24, 8)
FROZEN_EXECUTION_HORIZON = 8
OFFICIAL_NOISE_SEED = 1140


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V2DreamZeroDroidClient(DreamZeroClient):
    """Preserve the official sim-eval action path and retain exact actions."""

    def __init__(
        self,
        *,
        remote_host: str,
        remote_port: int,
        sampling_seed_label: int,
        action_trace_dir: Path,
    ) -> None:
        if sampling_seed_label not in {8300, 8301, 8302}:
            raise ValueError("DreamZero v2 seed labels are exactly 8300, 8301, 8302")
        self.sampling_seed_label = int(sampling_seed_label)
        self.action_trace_dir = Path(action_trace_dir)
        self.prompt: str | None = None
        self.executed_actions: list[np.ndarray] = []
        self.returned_raw_chunks: list[np.ndarray] = []
        self.returned_executable_chunks: list[np.ndarray] = []
        self.request_count = 0
        super().__init__(
            remote_host=remote_host,
            remote_port=remote_port,
            open_loop_horizon=FROZEN_EXECUTION_HORIZON,
            image_height=180,
            image_width=320,
            binarize_gripper=True,
            resize="pad",
            cam2_source="right",
        )

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        if instruction not in PROMPTS:
            raise ValueError(f"Prompt is outside the frozen direct gate: {instruction!r}")
        request = super()._pack_request(extracted_obs, instruction)
        # Match the released DreamZero sim-eval client exactly. The server also
        # casts these fields to float64, but freezing the wire dtype removes an
        # otherwise unnecessary adapter difference.
        request["observation/joint_position"] = np.asarray(
            extracted_obs["joint_position"], dtype=np.float64
        )
        request["observation/cartesian_position"] = np.zeros(6, dtype=np.float64)
        request["observation/gripper_position"] = np.asarray(
            extracted_obs["gripper_position"], dtype=np.float64
        )
        for key in (
            "observation/exterior_image_0_left",
            "observation/exterior_image_1_left",
            "observation/wrist_image_left",
        ):
            value = np.asarray(request[key])
            if value.shape != (180, 320, 3) or value.dtype != np.uint8:
                raise ValueError(f"DreamZero image contract changed for {key}: {value.shape}/{value.dtype}")
        return request

    def _unpack_response(self, response: Any) -> np.ndarray:
        raw = super()._unpack_response(response)
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != OFFICIAL_RETURNED_SHAPE:
            raise ValueError(
                f"DreamZero must return {OFFICIAL_RETURNED_SHAPE}, got {raw.shape}"
            )
        if not np.isfinite(raw).all():
            raise ValueError("DreamZero returned a non-finite action")
        self.returned_raw_chunks.append(raw.copy())
        self.request_count += 1
        return raw

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        executable = super()._postprocess_chunk(chunk)
        if executable.shape != OFFICIAL_RETURNED_SHAPE:
            raise ValueError(f"Executable DreamZero chunk changed shape: {executable.shape}")
        if not np.isin(executable[:, -1], [0.0, 1.0]).all():
            raise ValueError("DreamZero gripper binarization did not produce 0/1")
        self.returned_executable_chunks.append(executable.copy())
        return executable

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("DreamZero v2 prompt must remain episode-static")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (8,):
            raise ValueError(f"Executed action must have shape (8,), got {action.shape}")
        self.executed_actions.append(action.copy())
        return result

    def _write_trace(self) -> None:
        if not self.executed_actions:
            return
        if self.prompt not in PROMPTS:
            raise ValueError("Cannot identify the frozen DreamZero relation")
        relation = PROMPTS[self.prompt]
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{self.sampling_seed_label}_{relation}"
        executed = np.stack(self.executed_actions).astype(np.float32, copy=False)
        raw_chunks = np.stack(self.returned_raw_chunks).astype(np.float32, copy=False)
        executable_chunks = np.stack(self.returned_executable_chunks).astype(
            np.float32, copy=False
        )
        executed_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        raw_path = self.action_trace_dir / f"{stem}_returned_raw_chunks.npy"
        executable_path = (
            self.action_trace_dir / f"{stem}_returned_executable_chunks.npy"
        )
        metadata_path = self.action_trace_dir / f"{stem}_executed_actions.json"
        np.save(executed_path, executed, allow_pickle=False)
        np.save(raw_path, raw_chunks, allow_pickle=False)
        np.save(executable_path, executable_chunks, allow_pickle=False)
        metadata = {
            "schema_version": "vla-wam-shared-v2-dreamzero-action-trace-v1",
            "prompt": self.prompt,
            "requested_relation": relation,
            "sampling_seed_label": self.sampling_seed_label,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "seed_note": (
                "The released checkpoint hard-codes noise seed 1140. The v2 seed is an "
                "exact-pair label; overriding the checkpoint seed would change released behavior."
            ),
            "open_loop_execution_horizon": FROZEN_EXECUTION_HORIZON,
            "returned_action_horizon": OFFICIAL_RETURNED_SHAPE[0],
            "request_count": self.request_count,
            "executed_actions": {
                "path": str(executed_path),
                "sha256": _sha256(executed_path),
                "count": int(executed.shape[0]),
                "shape": list(executed.shape),
                "dtype": str(executed.dtype),
                "definition": "Exact post-binarization array passed to RoboLab env.step.",
            },
            "returned_raw_chunks": {
                "path": str(raw_path),
                "sha256": _sha256(raw_path),
                "count": int(raw_chunks.shape[0]),
                "shape": list(raw_chunks.shape),
                "dtype": str(raw_chunks.dtype),
                "definition": (
                    "Official 24x8 absolute joint-position/gripper response before client gripper binarization."
                ),
            },
            "returned_executable_chunks": {
                "path": str(executable_path),
                "sha256": _sha256(executable_path),
                "count": int(executable_chunks.shape[0]),
                "shape": list(executable_chunks.shape),
                "dtype": str(executable_chunks.dtype),
                "definition": "Returned chunks after official >0.5 gripper binarization.",
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._write_trace()
        super().reset(env_id=env_id)
