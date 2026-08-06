#!/usr/bin/env python3
"""V3-B003 wrapper around the frozen DreamZero s=2 client."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from experiments.v3.dreamzero_droid.client import V3DreamZeroS2Client


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
