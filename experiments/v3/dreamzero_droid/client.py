#!/usr/bin/env python3
"""DreamZero V2-A015 s=2 client extended to v3 Phase-A seed labels.

The server's official stochastic seed remains the released constant 1140.
The registered v3 seed is therefore an exact environment/pair label, as in
the frozen DreamZero evidence, and is never represented as a new effective
model-noise seed.  All returned actions, latent futures, and official reset
decodes are retained and hash-checked before an episode can compile.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from policies.dreamzero.client import DreamZeroClient


STUDY_ID = "vla_wam_language_steerability_v3"
MODEL_ID = "dreamzero_droid_action_cfg"
IDENTITY_BINDING = "V2-A015:dreamzero_action_cfg_s2"
PROMPTS = {
    "Put the Rubik's cube to the left of the bowl.": "left",
    "Put the Rubik's cube to the right of the bowl.": "right",
}
RETURNED_SHAPE = (24, 8)
OPEN_LOOP_HORIZON = 8
OFFICIAL_NOISE_SEED = 1140
ACTION_CFG_STYLE_SCALE = 2.0
VIDEO_CFG_SCALE = 5.0
SOURCE_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
CHECKPOINT_ID = "GEAR-Dreams/DreamZero-DROID"
CHECKPOINT_REVISION = "96ad344138c66e82536422432ad742f015784942"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class V3DreamZeroS2Client(DreamZeroClient):
    """Unchanged DreamZero action path plus measurement-only v3 retention."""

    def __init__(
        self,
        *,
        remote_host: str,
        remote_port: int,
        environment_seed: int,
        sampling_seed_label: int,
        action_trace_dir: Path,
        server_contract_path: Path,
        release_gate_path: Path,
        future_root: Path,
    ) -> None:
        if environment_seed not in range(8303, 8330):
            raise ValueError("DreamZero v3 Phase-A seeds are exactly 8303-8329")
        if sampling_seed_label != environment_seed:
            raise ValueError("DreamZero environment and registered sampling labels must match")
        if remote_port == 5000:
            raise ValueError("the protected pre-existing DreamZero port 5000 is prohibited")
        self.environment_seed = int(environment_seed)
        self.sampling_seed_label = int(sampling_seed_label)
        self.action_trace_dir = Path(action_trace_dir)
        self.server_contract_path = Path(server_contract_path).resolve()
        self.release_gate_path = Path(release_gate_path).resolve()
        self.future_root = Path(future_root).resolve()
        self.prompt: str | None = None
        self.executed_actions: list[np.ndarray] = []
        self.returned_raw_chunks: list[np.ndarray] = []
        self.returned_executable_chunks: list[np.ndarray] = []
        self.request_count = 0
        self.trace_written = False
        super().__init__(
            remote_host=remote_host,
            remote_port=remote_port,
            open_loop_horizon=OPEN_LOOP_HORIZON,
            image_height=180,
            image_width=320,
            binarize_gripper=True,
            resize="pad",
            cam2_source="right",
        )

    def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        if instruction not in PROMPTS:
            raise ValueError(f"prompt is outside the frozen direct gate: {instruction!r}")
        request = super()._pack_request(extracted_obs, instruction)
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
                raise ValueError(f"DreamZero image contract changed for {key}")
        return request

    def _unpack_response(self, response: Any) -> np.ndarray:
        raw = np.asarray(super()._unpack_response(response), dtype=np.float32)
        if raw.shape != RETURNED_SHAPE or not np.isfinite(raw).all():
            raise ValueError(f"DreamZero response must be finite {RETURNED_SHAPE}")
        self.returned_raw_chunks.append(raw.copy())
        self.request_count += 1
        return raw

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        executable = np.asarray(super()._postprocess_chunk(chunk), dtype=np.float32)
        if executable.shape != RETURNED_SHAPE:
            raise ValueError("DreamZero executable chunk shape changed")
        if not np.isin(executable[:, -1], [0.0, 1.0]).all():
            raise ValueError("DreamZero official gripper binarization is not binary")
        self.returned_executable_chunks.append(executable.copy())
        return executable

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if instruction not in PROMPTS:
            raise ValueError(f"prompt is outside the frozen direct gate: {instruction!r}")
        if self.prompt is None:
            self.prompt = instruction
        elif instruction != self.prompt:
            raise ValueError("DreamZero v3 prompt changed during the episode")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("DreamZero executed action must be finite [8]")
        self.executed_actions.append(action.copy())
        return result

    def _metadata_path(self) -> Path:
        if self.prompt not in PROMPTS:
            raise ValueError("cannot identify the DreamZero requested relation")
        relation = PROMPTS[self.prompt]
        return self.action_trace_dir / (
            f"seed{self.sampling_seed_label}_{relation}_executed_actions.json"
        )

    def write_trace(self) -> Path | None:
        if self.trace_written or not self.executed_actions:
            return None
        relation = PROMPTS[self.prompt]  # type: ignore[index]
        self.action_trace_dir.mkdir(parents=True, exist_ok=True)
        stem = f"seed{self.sampling_seed_label}_{relation}"
        executed_path = self.action_trace_dir / f"{stem}_executed_actions.npy"
        raw_path = self.action_trace_dir / f"{stem}_returned_raw_chunks.npy"
        executable_path = self.action_trace_dir / f"{stem}_returned_executable_chunks.npy"
        metadata_path = self._metadata_path()
        existing = [
            path for path in (executed_path, raw_path, executable_path, metadata_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(f"refusing to overwrite DreamZero v3 evidence: {existing}")
        executed = np.stack(self.executed_actions).astype(np.float32, copy=False)
        raw = np.stack(self.returned_raw_chunks).astype(np.float32, copy=False)
        executable = np.stack(self.returned_executable_chunks).astype(np.float32, copy=False)
        np.save(executed_path, executed, allow_pickle=False)
        np.save(raw_path, raw, allow_pickle=False)
        np.save(executable_path, executable, allow_pickle=False)
        metadata = {
            "schema_version": "vla-wam-shared-v3-dreamzero-s2-action-trace-v1",
            "study_id": STUDY_ID,
            "model_id": MODEL_ID,
            "identity_binding": IDENTITY_BINDING,
            "environment_seed": self.environment_seed,
            "sampling_seed_label": self.sampling_seed_label,
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "sampling_seed_semantics": (
                "registered matched-pair label; released checkpoint noise remains fixed at 1140"
            ),
            "prompt": self.prompt,
            "requested_relation": relation,
            "prompt_controller": "episode_static",
            "checkpoint": CHECKPOINT_ID,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "official_repository_commit": SOURCE_COMMIT,
            "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
            "video_cfg_scale": VIDEO_CFG_SCALE,
            "negative_branch_caveat": (
                "Derived CFG-style guidance uses DreamZero's fixed visual-quality "
                "negative prompt; it is not an official DreamZero action-CFG feature."
            ),
            "open_loop_execution_horizon": OPEN_LOOP_HORIZON,
            "request_count": self.request_count,
            "server_contract": {
                "path": str(self.server_contract_path),
                "sha256": _sha256(self.server_contract_path),
            },
            "release_gate": {
                "path": str(self.release_gate_path),
                "sha256": _sha256(self.release_gate_path),
            },
            "executed_actions": {
                "path": str(executed_path.resolve()),
                "sha256": _sha256(executed_path),
                "bytes": executed_path.stat().st_size,
                "count": int(executed.shape[0]),
                "shape": list(executed.shape),
                "dtype": str(executed.dtype),
            },
            "returned_raw_chunks": {
                "path": str(raw_path.resolve()),
                "sha256": _sha256(raw_path),
                "bytes": raw_path.stat().st_size,
                "count": int(raw.shape[0]),
                "shape": list(raw.shape),
                "dtype": str(raw.dtype),
            },
            "returned_executable_chunks": {
                "path": str(executable_path.resolve()),
                "sha256": _sha256(executable_path),
                "bytes": executable_path.stat().st_size,
                "count": int(executable.shape[0]),
                "shape": list(executable.shape),
                "dtype": str(executable.dtype),
            },
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self.trace_written = True
        return metadata_path

    def _bind_completed_future_manifest(self, manifests_before: set[Path]) -> None:
        deadline = time.monotonic() + 30.0
        new_manifests: set[Path] = set()
        while time.monotonic() < deadline:
            manifests_after = set(self.future_root.glob("episode_*/future_manifest.json"))
            new_manifests = manifests_after - manifests_before
            if new_manifests:
                break
            time.sleep(0.2)
        if len(new_manifests) != 1:
            raise RuntimeError(
                "expected exactly one finalized DreamZero future manifest, found "
                f"{sorted(str(path) for path in new_manifests)}"
            )
        manifest_path = next(iter(new_manifests)).resolve()
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("schema_version")
            != "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1"
            or manifest.get("amendment_id") != "V2-A015"
            or manifest.get("action_cfg_style_scale") != ACTION_CFG_STYLE_SCALE
            or manifest.get("video_cfg_scale") != VIDEO_CFG_SCALE
        ):
            raise ValueError("server future manifest is not the exact V2-A015 s=2 identity")
        requests = manifest.get("requests", [])
        if len(requests) != self.request_count:
            raise ValueError("DreamZero client/server future request count mismatch")
        if any(record.get("prompt") != self.prompt for record in requests):
            raise ValueError("DreamZero future manifest contains a non-static prompt")
        for index, (record, raw_chunk) in enumerate(
            zip(requests, self.returned_raw_chunks, strict=True)
        ):
            if record.get("action_cfg_style_scale") != ACTION_CFG_STYLE_SCALE:
                raise ValueError(f"future request {index} is not DreamZero s=2")
            action_entry = record.get("returned_action", {})
            action_path = Path(str(action_entry.get("path", "")))
            if not action_path.is_file() or action_entry.get("sha256") != _sha256(action_path):
                raise ValueError(f"future request {index} returned-action hash is invalid")
            if not np.array_equal(np.load(action_path, allow_pickle=False), raw_chunk):
                raise ValueError(f"future request {index} action differs from client response")
            latent_entry = record.get("latent_video", {})
            latent_path = Path(str(latent_entry.get("path", "")))
            if not latent_path.is_file() or latent_entry.get("sha256") != _sha256(latent_path):
                raise ValueError(f"future request {index} latent-video hash is invalid")
        decoded = manifest.get("official_reset_decode", [])
        if not decoded:
            raise ValueError("DreamZero episode has no official full reset decode")
        for entry in decoded:
            decoded_path = Path(str(entry.get("path", "")))
            if not decoded_path.is_file() or entry.get("sha256") != _sha256(decoded_path):
                raise ValueError("DreamZero official decoded-future hash is invalid")
        metadata_path = self._metadata_path()
        metadata = json.loads(metadata_path.read_text())
        metadata["future_manifest"] = {
            "path": str(manifest_path),
            "sha256": _sha256(manifest_path),
            "request_count": len(requests),
            "official_decode_count": len(decoded),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def reset(self, *, env_id: int | None = None) -> None:
        retain_episode = env_id is None and bool(self.executed_actions)
        manifests_before = (
            set(self.future_root.glob("episode_*/future_manifest.json"))
            if retain_episode else set()
        )
        if retain_episode:
            self.write_trace()
        super().reset(env_id=env_id)
        if retain_episode:
            self._bind_completed_future_manifest(manifests_before)
