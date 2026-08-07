#!/usr/bin/env python3
"""V3-B003 wrapper around the frozen DreamZero s=2 client."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from policies.dreamzero.client import DreamZeroClient

from experiments.v3.dreamzero_droid.client import PROMPTS, V3DreamZeroS2Client
from experiments.v3.dreamzero_droid.future_retention import (
    file_record,
    partition_session,
    write_session_manifest,
)


class V3B003DreamZeroClient(V3DreamZeroS2Client):
    """Reuse the exact action path while changing only registered seed labels."""

    def __init__(
        self,
        *,
        environment_seed: int,
        cell_id: str,
        reset_attestation: Path,
        **kwargs: Any,
    ) -> None:
        if environment_seed not in range(9400, 9427):
            raise ValueError("DreamZero V3-B003 seeds are exactly 9400-9426")
        super().__init__(environment_seed=8303, sampling_seed_label=8303, **kwargs)
        self.environment_seed = environment_seed
        self.sampling_seed_label = environment_seed
        self.cell_id = cell_id
        self.reset_attestation = Path(reset_attestation).resolve()

    def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        if self.request_count == 0:
            attestation = json.loads(self.reset_attestation.read_text())
            if (
                attestation.get("passed") is not True
                or attestation.get("registered_cell_id") != self.cell_id
                or attestation.get("prompt") != instruction
                or attestation.get("model_request_count_at_write") != 0
            ):
                raise RuntimeError("DreamZero V3-B003 reset attestation is absent or invalid")
        return super()._pack_request(extracted_obs, instruction)

    def _bind_session_future_manifest(self, session_id: str) -> None:
        deadline = time.monotonic() + 30.0
        session_manifest: dict[str, Any] | None = None
        raw_chunks = np.stack(self.returned_raw_chunks).astype(np.float32, copy=False)
        while time.monotonic() < deadline:
            try:
                session_manifest = partition_session(
                    self.future_root,
                    session_id=session_id,
                    prompt=str(self.prompt),
                    returned_raw_chunks=raw_chunks,
                )
                break
            except ValueError:
                time.sleep(0.2)
        if session_manifest is None:
            raise RuntimeError(
                f"DreamZero session {session_id} did not finalize with exact request ownership"
            )
        if self.prompt not in PROMPTS:
            raise RuntimeError("DreamZero V3-B003 future has an unregistered prompt")
        relation = PROMPTS[self.prompt]
        manifest_path = self.action_trace_dir / (
            f"seed{self.sampling_seed_label}_{relation}_future_manifest.json"
        )
        write_session_manifest(manifest_path, session_manifest)
        metadata_path = self._metadata_path()
        metadata = json.loads(metadata_path.read_text())
        metadata["future_manifest"] = {
            **file_record(manifest_path),
            "request_count": len(session_manifest["requests"]),
            "official_decode_count": len(session_manifest["official_reset_decode"]),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def reset(self, *, env_id: int | None = None) -> None:
        retain_episode = env_id is None and bool(self.executed_actions)
        session_ids = list(self._env_session_id.values()) if retain_episode else []
        if retain_episode and len(session_ids) != 1:
            raise RuntimeError(
                "DreamZero V3-B003 retained episode requires one active server session"
            )
        if retain_episode:
            self.write_trace()
        # Deliberately bypass the Phase-A manifest-diff retention hook. The
        # released RoboLab reset call is unchanged; only V3-B003 bookkeeping is
        # partitioned by the UUID that the official client already transmits.
        DreamZeroClient.reset(self, env_id=env_id)
        if retain_episode:
            self._bind_session_future_manifest(session_ids[0])
