"""Exact Cosmos3 Nano client overlay for one released V3-B001 cell."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

STUDY_ROOT = Path(__file__).resolve().parents[3]
V2_DIR = STUDY_ROOT / "experiments" / "cosmos"
if str(V2_DIR) not in sys.path:
    sys.path.insert(0, str(V2_DIR))

from v2_robolab_client import V2Cosmos3Client  # noqa: E402

from experiments.v3.cosmos_nano_phase_b.runtime_adapter import (  # noqa: E402
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    AMENDMENT_ID,
    AuthorizedCell,
    ReleaseBundle,
    RuntimeContractError,
    validate_reset_attestation,
)


class V3BNanoLiveClient(V2Cosmos3Client):
    """Retain every action/future while binding each request to one live reset."""

    def __init__(
        self,
        *,
        cell: AuthorizedCell,
        release: ReleaseBundle,
        runtime: Mapping[str, Any],
        reset_attestation_path: Path,
        ensure_reset_attestation: Callable[[], Path],
        **kwargs: Any,
    ) -> None:
        if kwargs.get("sampling_seed_base") != cell.seed:
            raise RuntimeContractError("live client seed differs from its released cell")
        super().__init__(**kwargs)
        self.cell = cell
        self.release = release
        self.runtime = dict(runtime)
        self.reset_attestation_path = Path(reset_attestation_path).resolve()
        self.ensure_reset_attestation = ensure_reset_attestation
        self.reset_attestation: dict[str, Any] | None = None
        self.reset_fingerprint_sha256: str | None = None
        self._pending_request: dict[str, Any] | None = None
        self.trace_path: Path | None = None

    def _ensure_reset_bound(self) -> None:
        if self.reset_attestation is not None:
            return
        produced = Path(self.ensure_reset_attestation()).resolve()
        if produced != self.reset_attestation_path:
            raise RuntimeContractError("reset-attestation callback returned an unexpected path")
        reset, fingerprint = validate_reset_attestation(
            produced,
            cell=self.cell,
            release=self.release,
            runtime=self.runtime,
        )
        self.reset_attestation = reset
        self.reset_fingerprint_sha256 = fingerprint

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        if instruction != self.cell.row["prompt"]:
            raise RuntimeContractError("live Nano prompt differs from the released cell bytes")
        self._ensure_reset_bound()
        if self.reset_fingerprint_sha256 is None:
            raise RuntimeContractError("live reset fingerprint is unavailable")
        request_index = self.request_index
        action_step_start = len(self.executed_actions)
        if action_step_start != request_index * ACTION_CHUNK_STEPS:
            raise RuntimeContractError("Nano request is not on a contiguous 32-action boundary")
        if action_step_start >= ACTION_CAP:
            raise RuntimeContractError("Nano request begins at or beyond the 450-action cap")
        request = super()._pack_request(extracted_obs, instruction)
        request.update(
            amendment_id=AMENDMENT_ID,
            registered_cell_id=self.cell.cell_id,
            request_index=request_index,
            action_step_start=action_step_start,
            release_fingerprint_sha256=self.release.release_fingerprint(self.cell),
            reset_fingerprint_sha256=self.reset_fingerprint_sha256,
            reset_attestation_path=str(self.reset_attestation_path),
            runtime_identity_sha256=self.runtime["runtime_identity_sha256"],
        )
        if request.get("prompt") != self.cell.row["prompt"]:
            raise RuntimeContractError("official client packed unexpected prompt bytes")
        self._pending_request = {
            "request_index": request_index,
            "action_step_start": action_step_start,
            "sampling_seed": self.cell.seed,
            "prompt": self.cell.row["prompt"],
            "release_fingerprint_sha256": request["release_fingerprint_sha256"],
            "reset_fingerprint_sha256": self.reset_fingerprint_sha256,
        }
        return request

    def _unpack_response(self, response: dict) -> np.ndarray:
        if self._pending_request is None or self.reset_fingerprint_sha256 is None:
            raise RuntimeContractError("Nano response arrived without a request fingerprint")
        request_index = self._pending_request["request_index"]
        expected_metadata = {
            "v3b001_nano_live_stack": "position_mirror_v1",
            "amendment_id": AMENDMENT_ID,
            "registered_cell_id": self.cell.cell_id,
            "sampling_seed": self.cell.seed,
            "request_index": request_index,
            "release_fingerprint_sha256": self.release.release_fingerprint(self.cell),
            "reset_fingerprint_sha256": self.reset_fingerprint_sha256,
            "runtime_identity_sha256": self.runtime["runtime_identity_sha256"],
        }
        for key, wanted in expected_metadata.items():
            if response.get(key) != wanted:
                raise RuntimeContractError(f"Nano server response mismatch for {key}")
        action = super()._unpack_response(response)
        if self.request_index != request_index + 1:
            raise RuntimeContractError("official client request index did not advance exactly once")
        record = self.request_records[-1]
        record.update(self._pending_request)
        record.update(
            returned_action_path=record["action_path"],
            decoded_future_path=record["future_path"],
            server_sampling_seed=self.cell.seed,
            registered_cell_id=self.cell.cell_id,
            runtime_identity_sha256=self.runtime["runtime_identity_sha256"],
            future_evidence_status="exposed_and_retained",
        )
        self._pending_request = None
        return action

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        if instruction != self.cell.row["prompt"]:
            raise RuntimeContractError("live Nano instruction is not the released static prompt")
        result = super().infer(obs, instruction, env_id=env_id)
        action = np.asarray(result.get("action"), dtype=np.float32)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise RuntimeContractError("executed Nano action must be finite joint-position [8]")
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
            schema_version="vla-wam-shared-v3b-nano-action-future-trace-v1",
            study_id=self.cell.row["study_id"],
            amendment_id=AMENDMENT_ID,
            registered_cell_id=self.cell.cell_id,
            matched_block_id=self.cell.row["matched_block_id"],
            model_id=self.cell.row["model_id"],
            arm=self.cell.arm,
            requested_relation=relation,
            environment_seed=self.cell.seed,
            sampling_seed=self.cell.seed,
            release_manifest_sha256=self.release.manifest_sha256,
            release_fingerprint_sha256=self.release.release_fingerprint(self.cell),
            reset_fingerprint_sha256=self.reset_fingerprint_sha256,
            runtime_identity_sha256=self.runtime["runtime_identity_sha256"],
            action_space="joint_position_8d",
            open_loop_horizon=ACTION_CHUNK_STEPS,
            action_cap=ACTION_CAP,
            prompt_controller="episode_static",
            missing_future_policy="infrastructure_invalid_never_zero",
        )
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        self.trace_path = metadata_path

    def write_trace(self) -> Path | None:
        self._write_trace()
        return self.trace_path
