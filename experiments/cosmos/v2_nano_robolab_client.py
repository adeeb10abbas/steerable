"""Nano-specific identity and seed checks over the frozen Cosmos v2 client."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from v2_robolab_client import V2Cosmos3Client

MODEL_ID = "cosmos3_nano_policy_droid"
CHECKPOINT_REVISION = "6706d7680581c255ff61e0f3bb49d90eac55c79e"


class V2NanoCosmos3Client(V2Cosmos3Client):
    def _unpack_response(self, response: dict) -> np.ndarray:
        action = super()._unpack_response(response)
        server_seed = response.get("sampling_seed")
        if server_seed != self.sampling_seed_base:
            raise ValueError(
                "Nano server did not echo the requested sampling seed: "
                f"expected={self.sampling_seed_base}, observed={server_seed}"
            )
        self.request_records[-1].update(
            model_id=MODEL_ID,
            checkpoint_revision=CHECKPOINT_REVISION,
            server_sampling_seed=int(server_seed),
        )
        return action

    def _write_trace(self) -> None:
        already_written = self._trace_written
        super()._write_trace()
        if already_written or not self._trace_written:
            return
        relation = self._relation()
        metadata_path = Path(self.action_trace_dir) / (
            f"seed{self.sampling_seed_base}_{relation}_executed_actions.json"
        )
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            model_id=MODEL_ID,
            checkpoint_revision=CHECKPOINT_REVISION,
            amendment_id="V2-A011",
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
