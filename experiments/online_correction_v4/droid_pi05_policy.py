"""π0.5 policy adapter preserving the Phase-B request envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Callable, Mapping

from experiments.online_correction_v4.adapters import ObservationPacket, PolicyResponse
from experiments.online_correction_v4.droid_contract import (
    PI05_ACTION_CHUNK_STEPS,
    PI05_ACTION_SHAPE,
    PI05_POLICY_ID,
    PolicyRuntimeBinding,
    canonical_json_bytes,
    sha256_bytes,
)
from experiments.online_correction_v4.droid_nano_policy import _hash_action_chunk, _normalize_action_chunk
from experiments.online_correction_v4.droid_policy_request import (
    build_v4_request_envelope,
    normalize_pi05_response,
    observation_packed_request,
    request_audit_projection,
)
from experiments.online_correction_v4.droid_transport import EpisodePolicyTransport


class Pi05PolicyContractError(RuntimeError):
    """Raised when π0.5 request/response contracts diverge."""


Transport = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass
class Pi05RequestRecord:
    request_index: int
    observation_id: str
    observation_sha256: str
    action_step_start: int
    request_sampling_seed: int
    prompt_sha256: str
    reset_fingerprint_sha256: str
    runtime_identity_sha256: str
    action_sha256: str = ""
    wall_duration_s: float = 0.0
    wire_request_sha256: str = ""


@dataclass
class DroidPi05PolicyAdapter:
    """V4 PolicyAdapter for π0.5 with exact 15×8 chunks and per-request seed semantics."""

    binding: PolicyRuntimeBinding
    episode_id: str
    policy_seed: int
    prompt_text: str
    prompt_sha256: str
    reset_fingerprint_sha256: str
    runtime_identity_sha256: str
    transport: Transport | EpisodePolicyTransport | None = None
    ensure_reset_attestation: Callable[[], str] | None = None
    executed_action_count: Callable[[], int] | None = None
    request_count: int = 0
    records: list[Pi05RequestRecord] = field(default_factory=list)

    @property
    def policy_id(self) -> str:
        return PI05_POLICY_ID

    def reset(self, *, policy_seed: int, prompt_text: str) -> None:
        if policy_seed != self.policy_seed:
            raise Pi05PolicyContractError("policy_seed changed across session reset")
        if prompt_text != self.prompt_text:
            raise Pi05PolicyContractError("static episode prompt changed across session reset")
        self.request_count = 0
        self.records.clear()
        if isinstance(self.transport, EpisodePolicyTransport):
            self.transport.begin_episode()

    def infer(self, observation: ObservationPacket) -> PolicyResponse:
        if observation.payload_sha256 != sha256_bytes(observation.payload):
            raise Pi05PolicyContractError("observation payload hash mismatch")
        reset_hash = self._ensure_reset()
        request_seed = self.policy_seed * 1000 + self.request_count
        action_step_start = self._action_step_start()
        packed = observation_packed_request(observation)
        audit = {
            "schema_version": "v4-droid-pi05-request-v1",
            "policy_id": PI05_POLICY_ID,
            "episode_id": self.episode_id,
            "observation_id": observation.observation_id,
            "observation_capture_time_s": observation.capture_time_s,
            "observation_sha256": observation.payload_sha256,
            "instruction": self.prompt_text,
            "prompt_sha256": self.prompt_sha256,
            "sampling_seed": request_seed,
            "request_index": self.request_count,
            "action_step_start": action_step_start,
            "reset_fingerprint_sha256": reset_hash,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "checkpoint_sha256": self.binding.checkpoint_sha256,
            "future_interface": "actions_only",
            "missing_future_policy": "action_only_interface_not_applicable_never_zero",
        }
        wire, full_request = build_v4_request_envelope(
            policy_id=PI05_POLICY_ID,
            packed=packed,
            audit=audit,
        )
        started = time.monotonic()
        response = self._invoke_transport(full_request)
        wall_duration_s = time.monotonic() - started
        normalized = normalize_pi05_response(dict(response))
        raw_action = normalized.get("action", normalized.get("actions"))
        actions = _normalize_action_chunk(raw_action, PI05_ACTION_SHAPE)
        echoed = normalized.get("v2a010_sampling_seed", normalized.get("sampling_seed"))
        if echoed != request_seed:
            raise Pi05PolicyContractError("π0.5 server did not attest the exact request seed")
        action_sha = _hash_action_chunk(actions)
        record = Pi05RequestRecord(
            request_index=self.request_count,
            observation_id=observation.observation_id,
            observation_sha256=observation.payload_sha256,
            action_step_start=action_step_start,
            request_sampling_seed=request_seed,
            prompt_sha256=self.prompt_sha256,
            reset_fingerprint_sha256=reset_hash,
            runtime_identity_sha256=self.runtime_identity_sha256,
            action_sha256=action_sha,
            wall_duration_s=wall_duration_s,
            wire_request_sha256=sha256_bytes(
                canonical_json_bytes(request_audit_projection(wire))
            ),
        )
        self.records.append(record)
        self.request_count += 1
        chunk_id = hashlib.sha256(
            json.dumps(
                {
                    "episode_id": self.episode_id,
                    "request_index": record.request_index,
                    "request_sampling_seed": request_seed,
                    "action_sha256": action_sha,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return PolicyResponse(
            chunk_id=chunk_id,
            actions=actions,
            wall_duration_s=wall_duration_s,
            action_sha256=action_sha,
            generated_horizon=PI05_ACTION_CHUNK_STEPS,
            request_audit={
                "request_index": record.request_index,
                "request_sampling_seed": record.request_sampling_seed,
                "action_step_start": record.action_step_start,
                "observation_sha256": record.observation_sha256,
                "prompt_sha256": record.prompt_sha256,
                "reset_fingerprint_sha256": record.reset_fingerprint_sha256,
                "runtime_identity_sha256": record.runtime_identity_sha256,
                "wire_request_sha256": record.wire_request_sha256,
            },
        )

    def _action_step_start(self) -> int:
        if self.executed_action_count is not None:
            return int(self.executed_action_count())
        return self.request_count * PI05_ACTION_CHUNK_STEPS

    def _ensure_reset(self) -> str:
        if self.ensure_reset_attestation is not None:
            produced = self.ensure_reset_attestation()
            if self.reset_fingerprint_sha256 and produced != self.reset_fingerprint_sha256:
                raise Pi05PolicyContractError("reset attestation fingerprint mismatch")
            if not self.reset_fingerprint_sha256:
                self.reset_fingerprint_sha256 = produced
        if not self.reset_fingerprint_sha256:
            raise Pi05PolicyContractError("reset attestation is required before request zero")
        return self.reset_fingerprint_sha256

    def _invoke_transport(self, request: dict[str, Any]) -> Mapping[str, Any]:
        if self.transport is None:
            raise Pi05PolicyContractError(
                "π0.5 transport is not bound; live inference is blocked in this checkout"
            )
        response = self.transport(request)
        if not isinstance(response, Mapping):
            raise Pi05PolicyContractError("π0.5 transport response must be an object")
        return response

    def close(self) -> None:
        from experiments.online_correction_v4.droid_transport import EpisodePolicyTransport

        if isinstance(self.transport, EpisodePolicyTransport):
            self.transport.close()


def fake_pi05_transport(base_seed: int) -> Transport:
    """Deterministic transport for contract tests."""

    def _transport(request: Mapping[str, Any]) -> dict[str, Any]:
        index = int(request["request_index"])
        request_seed = int(request["sampling_seed"])
        action = [[0.02 * (index + 1) if col == 0 else 0.0 for col in range(8)] for _ in range(PI05_ACTION_CHUNK_STEPS)]
        return {
            "action": action,
            "v2a010_sampling_seed": request_seed,
            "sampling_seed": request_seed,
            "base_seed": base_seed,
        }

    return _transport
