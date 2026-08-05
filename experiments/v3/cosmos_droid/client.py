"""Cosmos client overlay for registered v3 DROID cells.

This subclasses the exact v2 evidence-retaining client.  It changes no model
request or response tensor contract; it only broadens the prospectively
registered seed range and adds v3 cell/runtime provenance to the trace sidecar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = STUDY_ROOT / "experiments" / "cosmos"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from v2_robolab_client import V2Cosmos3Client  # noqa: E402

from experiments.v3.cosmos_droid.contract import (  # noqa: E402
    AuthorizedPair,
    ContractError,
    MODEL_CONTRACTS,
)


class V3CosmosDroidClient(V2Cosmos3Client):
    """Retain exact actions/futures and bind them to one authorized v3 cell."""

    def __init__(
        self,
        *,
        pair: AuthorizedPair,
        runtime_identity: dict[str, Any],
        relation: str | None = None,
        **kwargs: Any,
    ) -> None:
        if relation not in {None, "left", "right"}:
            raise ContractError("relation must be left or right when provided")
        if kwargs.get("sampling_seed_base") != pair.seed:
            raise ContractError("client sampling seed must equal the authorized pair seed")
        super().__init__(**kwargs)
        self.v3_pair = pair
        self.v3_relation = relation
        self.v3_cell = pair.cell(relation) if relation is not None else None
        self.runtime_identity = runtime_identity

    def _bind_prompt(self, instruction: str) -> None:
        matches = [
            relation for relation in ("left", "right")
            if instruction == self.v3_pair.cell(relation)["prompt"]
        ]
        if len(matches) != 1:
            raise ContractError("runtime prompt bytes do not match either registered cell")
        relation = matches[0]
        if self.v3_relation is not None and self.v3_relation != relation:
            raise ContractError("client received the opposite registered condition")
        self.v3_relation = relation
        self.v3_cell = self.v3_pair.cell(relation)

    def _unpack_response(self, response: dict) -> np.ndarray:
        if self.v3_cell is None or self.v3_relation is None:
            raise ContractError("response arrived before the static prompt was bound")
        action = super()._unpack_response(response)
        server_seed = response.get("sampling_seed")
        if MODEL_CONTRACTS[self.v3_pair.model_id]["sampling_seed_echo_required"]:
            if server_seed != self.v3_pair.seed:
                raise ContractError(
                    f"server did not echo sampling_seed={self.v3_pair.seed}: {server_seed!r}"
                )
        self.request_records[-1].update(
            study_id=self.v3_cell["study_id"],
            registered_cell_id=self.v3_cell["cell_id"],
            pair_id=self.v3_pair.pair_id,
            model_id=self.v3_pair.model_id,
            requested_relation=self.v3_relation,
            environment_seed=self.v3_pair.seed,
            server_sampling_seed=server_seed,
            runtime_identity_sha256=self.runtime_identity["runtime_identity_sha256"],
            future_evidence_status="exposed_and_retained",
        )
        return action

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        self._bind_prompt(instruction)
        if self.v3_cell is None or instruction != self.v3_cell["prompt"]:
            raise ContractError("runtime prompt bytes do not match the registered cell")
        return super().infer(obs, instruction, env_id=env_id)

    def _write_trace(self) -> None:
        already_written = self._trace_written
        super()._write_trace()
        if already_written or not self._trace_written:
            return
        if self.v3_cell is None or self.v3_relation is None:
            raise ContractError("cannot write an unbound v3 trace")
        metadata_path = self.action_trace_dir / (
            f"seed{self.v3_pair.seed}_{self.v3_relation}_executed_actions.json"
        )
        metadata = json.loads(metadata_path.read_text())
        metadata.update(
            schema_version="vla-wam-shared-v3-cosmos3-action-future-trace-v1",
            study_id=self.v3_cell["study_id"],
            registered_cell_id=self.v3_cell["cell_id"],
            pair_id=self.v3_pair.pair_id,
            model_id=self.v3_pair.model_id,
            checkpoint_revision=MODEL_CONTRACTS[self.v3_pair.model_id]["checkpoint_revision"],
            requested_relation=self.v3_relation,
            environment_seed=self.v3_pair.seed,
            sampling_seed=self.v3_pair.seed,
            queue_sha256=self.v3_pair.queue_sha256,
            runtime_identity_sha256=self.runtime_identity["runtime_identity_sha256"],
            future_evidence_policy=(
                "Every exposed decoded future is retained with a content hash. "
                "Missing output is infrastructure-invalid and is never encoded as zero."
            ),
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
