#!/usr/bin/env python3
"""Launch the exact DreamZero release with measurement-only future retention."""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import hashlib
import json
import logging
import os
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import tyro

import socket_test_optimized_AR as official
from eval_utils.policy_server import PolicyServerConfig
from groot.vla.data.schema import EmbodimentTag
from groot.vla.model.n1_5.sim_policy import GrootSimPolicy


LOGGER = logging.getLogger(__name__)
OFFICIAL_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
OFFICIAL_NOISE_SEED = 1140


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclasses.dataclass
class Args:
    port: int
    model_path: str
    tokenizer_path: str
    future_root: str
    timeout_seconds: int = 50000
    enable_dit_cache: bool = True


class InstrumentedARDroidPolicy(official.ARDroidRoboarenaPolicy):
    """Copy internal latent futures after inference; return official actions unchanged."""

    def __init__(self, *args: Any, future_root: Path, **kwargs: Any) -> None:
        super().__init__(*args, output_dir=str(future_root / "decoded"), **kwargs)
        self._future_root = future_root
        self._future_root.mkdir(parents=True, exist_ok=True)
        self._episode_index = 0
        self._measurement_records: list[dict[str, Any]] = []

    def infer(self, obs: dict) -> np.ndarray:
        captured: dict[str, torch.Tensor] = {}
        original_forward = self._policy.lazy_joint_forward_causal

        def measured_forward(*args: Any, **kwargs: Any):
            result, video_pred = original_forward(*args, **kwargs)
            captured["video_pred"] = video_pred
            return result, video_pred

        self._policy.lazy_joint_forward_causal = measured_forward
        try:
            action = super().infer(obs)
        finally:
            self._policy.lazy_joint_forward_causal = original_forward

        if "video_pred" not in captured:
            raise RuntimeError("Official DreamZero call did not expose video_pred")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (24, 8):
            raise ValueError(f"Official DreamZero action shape changed: {action.shape}")
        request_index = len(self._measurement_records)
        episode_dir = self._future_root / f"episode_{self._episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        action_path = episode_dir / f"request_{request_index:04d}_official_action.npy"
        latent_path = episode_dir / f"request_{request_index:04d}_latent_video.pt"
        np.save(action_path, action, allow_pickle=False)
        # Copying the already-computed tensor is measurement-only. The exact
        # action array returned to the websocket server is not modified.
        latent = captured["video_pred"].detach().cpu()
        torch.save(latent, latent_path)
        record = {
            "request_index": request_index,
            "session_id": obs.get("session_id"),
            "prompt": obs.get("prompt"),
            "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
            "official_action": {
                "path": str(action_path),
                "sha256": _sha256(action_path),
                "shape": list(action.shape),
                "dtype": str(action.dtype),
            },
            "latent_video": {
                "path": str(latent_path),
                "sha256": _sha256(latent_path),
                "shape": list(latent.shape),
                "dtype": str(latent.dtype),
                "definition": "Exact video_pred tensor returned beside the action batch.",
            },
            "input_hashes": {
                key: _array_sha(obs[key])
                for key in (
                    "observation/exterior_image_0_left",
                    "observation/exterior_image_1_left",
                    "observation/wrist_image_left",
                    "observation/joint_position",
                    "observation/gripper_position",
                )
            },
        }
        self._measurement_records.append(record)
        return action

    def reset(self, reset_info: dict) -> None:
        decoded_dir = Path(self._output_dir)
        decoded_before = {path.resolve() for path in decoded_dir.glob("*.mp4")}
        super().reset(reset_info)
        decoded_after = {path.resolve() for path in decoded_dir.glob("*.mp4")}
        decoded_new = sorted(decoded_after - decoded_before)
        episode_dir = self._future_root / f"episode_{self._episode_index:03d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": "vla-wam-shared-v2-dreamzero-future-retention-v1",
            "episode_index": self._episode_index,
            "official_repository_commit": OFFICIAL_COMMIT,
            "instrumentation_role": "measurement_only",
            "request_count": len(self._measurement_records),
            "requests": self._measurement_records,
            "official_reset_decode": [
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in decoded_new
            ],
            "reset_info": reset_info,
            "claim_boundary": (
                "Latent tensors and official reset-path decoded MP4s are retained without "
                "changing policy inputs, action values, horizon, or control flow."
            ),
        }
        (episode_dir / "future_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        self._measurement_records = []
        self._episode_index += 1


def main(args: Args) -> None:
    if args.port == 5000:
        raise ValueError("V2-A007 prohibits the pre-existing DreamZero port 5000")
    model_path = Path(args.model_path).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    future_root = Path(args.future_root).resolve()
    if not model_path.is_dir() or not tokenizer_path.is_dir():
        raise FileNotFoundError("Exact model and tokenizer directories must exist")
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ["ATTENTION_BACKEND"] = "TE"
    torch._dynamo.config.recompile_limit = 800

    device_mesh = official.init_mesh()
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise ValueError("V2-A007 requires exactly two DreamZero server ranks")
    if "B200" not in torch.cuda.get_device_name(torch.cuda.current_device()):
        raise ValueError("V2-A007 policy ranks must run on ali-owned B200 GPUs")
    timeout = datetime.timedelta(seconds=args.timeout_seconds)
    signal_group = dist.new_group(backend="gloo", timeout=timeout)
    policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("oxe_droid"),
        model_path=str(model_path),
        device="cuda",
        device_mesh=device_mesh,
        tokenizer_path_override=str(tokenizer_path),
    )

    if rank == 0:
        future_root.mkdir(parents=True, exist_ok=True)
        contract = {
            "schema_version": "vla-wam-shared-v2-dreamzero-server-contract-v1",
            "official_repository_commit": OFFICIAL_COMMIT,
            "model_path": str(model_path),
            "tokenizer_path": str(tokenizer_path),
            "port": args.port,
            "world_size": dist.get_world_size(),
            "visible_device_names": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
            "official_noise_seed": OFFICIAL_NOISE_SEED,
            "enable_dit_cache": args.enable_dit_cache,
            "future_root": str(future_root),
        }
        (future_root / "server_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )
        wrapper = InstrumentedARDroidPolicy(
            groot_policy=policy,
            signal_group=signal_group,
            future_root=future_root,
        )
        config = PolicyServerConfig(
            image_resolution=(180, 320),
            needs_wrist_camera=True,
            n_external_cameras=2,
            needs_stereo_camera=False,
            needs_session_id=True,
            action_space="joint_position",
        )
        LOGGER.info("Creating isolated DreamZero v2 server on %s:%d", socket.gethostname(), args.port)
        official.RoboarenaServer(
            policy=wrapper,
            server_config=config,
            host="0.0.0.0",
            port=args.port,
        ).serve_forever()
    else:
        server = official.WebsocketPolicyServer(
            policy=policy,
            host="0.0.0.0",
            port=args.port,
            metadata={
                "embodiment": "oxe_droid",
                "model_name": "dreamzero",
                "model_path": str(model_path),
            },
            output_dir=None,
            signal_group=signal_group,
        )
        asyncio.run(server._worker_loop())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
