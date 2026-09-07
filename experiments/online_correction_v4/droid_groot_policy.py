"""GR00T Bridge WidowX policy adapter for C8 second_stack episodes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import time
from typing import Any, Callable, Mapping

from experiments.online_correction_v4.adapters import ObservationPacket, PolicyResponse
from experiments.online_correction_v4.droid_contract import (
    GROOT_ACTION_SHAPE,
    GROOT_POLICY_ID,
    PolicyRuntimeBinding,
    canonical_json_bytes,
    sha256_bytes,
)
from experiments.online_correction_v4.droid_groot_observation import normalize_groot_action_chunk
from experiments.online_correction_v4.droid_policy_request import (
    PolicyInfraInvalidError,
    build_v4_request_envelope,
    request_audit_projection,
)
from experiments.online_correction_v4.droid_transport import EpisodePolicyTransport


class GrootPolicyContractError(RuntimeError):
    """Raised when GR00T Bridge request/response contracts diverge."""


Transport = Callable[[dict[str, Any]], Mapping[str, Any]]


@dataclass
class GrootRequestRecord:
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


def _hash_action_chunk(actions: tuple[tuple[float, ...], ...]) -> str:
    return sha256_bytes(canonical_json_bytes(actions))


def _observation_packed_request(observation: ObservationPacket) -> dict[str, Any]:
    native = observation.native_input if isinstance(observation.native_input, Mapping) else {}
    processed = native.get("processed_observation")
    prompt = native.get("prompt")
    if not isinstance(processed, Mapping):
        raise GrootPolicyContractError("native processed_observation is required for GR00T Bridge")
    if not isinstance(prompt, str) or not prompt.strip():
        raise GrootPolicyContractError("static episode prompt is required for GR00T Bridge")
    return {
        "processed_observation": dict(processed),
        "prompt": prompt,
    }


@dataclass
class DroidGrootPolicyAdapter:
    """V4 PolicyAdapter for GR00T Bridge with exact 8×8 chunks."""

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
    records: list[GrootRequestRecord] = field(default_factory=list)

    @property
    def policy_id(self) -> str:
        return GROOT_POLICY_ID

    def reset(self, *, policy_seed: int, prompt_text: str) -> None:
        if policy_seed != self.policy_seed:
            raise GrootPolicyContractError("policy_seed changed across session reset")
        if prompt_text != self.prompt_text:
            raise GrootPolicyContractError("static episode prompt changed across session reset")
        self.request_count = 0
        self.records.clear()
        if isinstance(self.transport, EpisodePolicyTransport):
            self.transport.begin_episode()

    def infer(self, observation: ObservationPacket) -> PolicyResponse:
        if observation.payload_sha256 != sha256_bytes(observation.payload):
            raise GrootPolicyContractError("observation payload hash mismatch")
        reset_hash = self._ensure_reset()
        request_seed = self.policy_seed * 1000 + self.request_count
        action_step_start = self._action_step_start()
        packed = _observation_packed_request(observation)
        audit = {
            "schema_version": "v4-droid-groot-request-v1",
            "policy_id": GROOT_POLICY_ID,
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
            "future_interface": "none",
            "missing_future_policy": "groot_bridge_no_future_surface",
        }
        wire, full_request = build_v4_request_envelope(
            policy_id=GROOT_POLICY_ID,
            packed=packed,
            audit=audit,
        )
        started = time.monotonic()
        response = self._invoke_transport(full_request)
        wall_duration_s = time.monotonic() - started
        actions = normalize_groot_action_chunk(response.get("action"), GROOT_ACTION_SHAPE)
        if response.get("sampling_seed") not in (None, request_seed):
            raise GrootPolicyContractError("GR00T server returned an unexpected sampling seed")
        action_sha = _hash_action_chunk(actions)
        record = GrootRequestRecord(
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
            wire_request_sha256=sha256_bytes(canonical_json_bytes(wire)),
        )
        self.records.append(record)
        self.request_count += 1
        return PolicyResponse(
            chunk_id=f"groot-{record.request_index}",
            actions=actions,
            wall_duration_s=wall_duration_s,
            action_sha256=action_sha,
            generated_horizon=len(actions),
            request_audit=request_audit_projection(full_request),
        )

    def _ensure_reset(self) -> str:
        if self.ensure_reset_attestation is None:
            if self.reset_fingerprint_sha256:
                return self.reset_fingerprint_sha256
            raise GrootPolicyContractError("reset attestation callback is required before inference")
        fingerprint = self.ensure_reset_attestation()
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise GrootPolicyContractError("reset attestation fingerprint is missing")
        return fingerprint

    def _action_step_start(self) -> int:
        if self.executed_action_count is None:
            return 0
        return int(self.executed_action_count())

    def _invoke_transport(self, request: dict[str, Any]) -> Mapping[str, Any]:
        if self.transport is None:
            raise GrootPolicyContractError("policy transport is not configured")
        try:
            response = self.transport(request)
        except PolicyInfraInvalidError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise GrootPolicyContractError(str(exc)) from exc
        if not isinstance(response, Mapping):
            raise GrootPolicyContractError("GR00T transport response must be an object")
        return response


def fake_groot_transport(policy_seed: int) -> Transport:
    """Deterministic fake transport for unit tests."""

    def _transport(request: Mapping[str, Any]) -> dict[str, Any]:
        index = int(request.get("request_index", 0))
        rows, cols = GROOT_ACTION_SHAPE
        base = float(policy_seed + index) * 1e-4
        action = [[base + step * 1e-5 + dim * 1e-6 for dim in range(cols)] for step in range(rows)]
        return {
            "action": action,
            "sampling_seed": request.get("sampling_seed"),
        }

    return _transport
