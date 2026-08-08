"""Evidence-retaining Cosmos client for one released V3-E004 session.

The model request itself remains the exact Phase-A/Phase-C RoboLab request.
This overlay adds only the fail-closed E004 authorization envelope and records
the cell, session, native-input, native-output, and response binding hashes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[4]
V2_DIR = STUDY_ROOT / "experiments" / "cosmos"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from v2_robolab_client import V2Cosmos3Client  # noqa: E402

from .cosmos_runtime import (  # noqa: E402
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    AuthorizedCell,
    CosmosRuntimeError,
    RegistrationBundle,
    add_request_envelope,
    validate_response_envelope,
    validate_runtime_payload,
    validate_session_manifest,
)


class E004CosmosClient(V2Cosmos3Client):
    """Bind every native Cosmos request and retained future to one E004 cell."""

    def __init__(
        self,
        *,
        bundle: RegistrationBundle,
        cell: AuthorizedCell,
        runtime: Mapping[str, Any],
        session_manifest_path: Path,
        **kwargs: Any,
    ) -> None:
        if cell.cell_id not in bundle.by_cell_id:
            raise CosmosRuntimeError("Cosmos client cell is not in the E004 bundle")
        if kwargs.get("sampling_seed_base") != cell.seed:
            raise CosmosRuntimeError("Cosmos client seed differs from its E004 cell")
        super().__init__(**kwargs)
        self.bundle = bundle
        self.cell = cell
        self.runtime = validate_runtime_payload(runtime, model_id=cell.model_id)
        self.session_manifest_path = Path(session_manifest_path).resolve()
        self.session, self.session_manifest_sha256 = validate_session_manifest(
            self.session_manifest_path,
            bundle=bundle,
            cell=cell,
            runtime=self.runtime,
        )
        self._pending_request: dict[str, Any] | None = None
        self.trace_path: Path | None = None

    def _pack_request(self, extracted_obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        if instruction != self.cell.row["prompt"]:
            raise CosmosRuntimeError("Cosmos client prompt differs from its E004 cell")
        request_index = self.request_index
        action_step_start = len(self.executed_actions)
        if action_step_start != request_index * ACTION_CHUNK_STEPS:
            raise CosmosRuntimeError("Cosmos request is not on a contiguous chunk boundary")
        if action_step_start >= ACTION_CAP:
            raise CosmosRuntimeError("Cosmos request starts at or beyond the action cap")
        native = super()._pack_request(extracted_obs, instruction)
        request = add_request_envelope(
            native,
            bundle=self.bundle,
            cell=self.cell,
            runtime=self.runtime,
            session=self.session,
            session_manifest_path=self.session_manifest_path,
            session_manifest_sha256=self.session_manifest_sha256,
            request_index=request_index,
            action_step_start=action_step_start,
        )
        self._pending_request = {
            key: request[key]
            for key in (
                "request_index",
                "action_step_start",
                "model_input_sha256",
                "request_binding_sha256",
            )
        }
        return request

    def _unpack_response(self, response: dict[str, Any]) -> np.ndarray:
        if self._pending_request is None:
            raise CosmosRuntimeError("Cosmos response arrived without a pending E004 request")
        pending = dict(self._pending_request)
        validate_response_envelope(
            response,
            cell=self.cell,
            runtime=self.runtime,
            session=self.session,
            pending_request=pending,
        )
        action = super()._unpack_response(response)
        if self.request_index != pending["request_index"] + 1:
            raise CosmosRuntimeError("official Cosmos client request index did not advance once")
        record = self.request_records[-1]
        record.update(pending)
        record.update(
            schema_version="vla-wam-shared-v3e004-cosmos-request-evidence-v1",
            study_id=self.cell.row["study_id"],
            amendment_id=self.cell.row["amendment_id"],
            model_id=self.cell.model_id,
            registered_cell_id=self.cell.cell_id,
            cell_sha256=self.cell.cell_sha256,
            session_id=self.session["session_id"],
            session_sha256=self.session["session_sha256"],
            session_manifest_path=str(self.session_manifest_path),
            session_manifest_sha256=self.session_manifest_sha256,
            registration_commit=self.bundle.registration_commit,
            registration_sha256=self.bundle.registration_sha256,
            queue_sha256=self.bundle.queue_sha256,
            candidate_sha256=self.bundle.candidate_sha256,
            runtime_identity_sha256=self.runtime["runtime_identity_sha256"],
            model_output_sha256=response["model_output_sha256"],
            response_binding_sha256=response["response_binding_sha256"],
            server_sampling_seed=response["sampling_seed"],
            returned_action_path=record["action_path"],
            decoded_future_path=record["future_path"],
            future_evidence_status="exposed_and_retained",
        )
        self._pending_request = None
        return action

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict[str, Any]:
        if instruction != self.cell.row["prompt"]:
            raise CosmosRuntimeError("E004 requires the registered episode-static prompt")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result.get("action"), dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise CosmosRuntimeError("executed Cosmos action must be finite joint-position [8]")
        return result

    def _write_trace(self) -> None:
        already_written = self._trace_written
        super()._write_trace()
        if already_written or not self._trace_written:
            return
        relation = self.cell.relation
        metadata_path = self.action_trace_dir / (
            f"seed{self.cell.seed}_{relation}_executed_actions.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            schema_version="vla-wam-shared-v3e004-cosmos-action-future-trace-v1",
            study_id=self.cell.row["study_id"],
            amendment_id=self.cell.row["amendment_id"],
            model_id=self.cell.model_id,
            registered_cell_id=self.cell.cell_id,
            cell_sha256=self.cell.cell_sha256,
            matched_pair_id=self.cell.row["matched_pair_id"],
            symmetry_level_s=self.cell.symmetry_level,
            requested_relation=relation,
            environment_seed=self.cell.row["environment_seed"],
            sampling_seed=self.cell.seed,
            session_id=self.session["session_id"],
            session_sha256=self.session["session_sha256"],
            session_manifest_sha256=self.session_manifest_sha256,
            registration_commit=self.bundle.registration_commit,
            registration_sha256=self.bundle.registration_sha256,
            queue_sha256=self.bundle.queue_sha256,
            candidate_sha256=self.bundle.candidate_sha256,
            runtime_identity_sha256=self.runtime["runtime_identity_sha256"],
            action_space="joint_position_8d",
            open_loop_horizon=ACTION_CHUNK_STEPS,
            action_cap=ACTION_CAP,
            prompt_controller="episode_static",
            missing_future_policy="infrastructure_invalid_never_zero",
        )
        metadata_path.write_text(
            json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.trace_path = metadata_path

    def write_trace(self) -> Path | None:
        self._write_trace()
        return self.trace_path
