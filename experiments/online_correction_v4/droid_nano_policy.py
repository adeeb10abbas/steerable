"""Cosmos3 Nano policy adapter preserving the Phase-B request envelope."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from experiments.online_correction_v4.adapters import FutureArtifact, ObservationPacket, PolicyResponse
from experiments.online_correction_v4.attempts import InfraInvalidReason
from experiments.online_correction_v4.droid_contract import (
    ACTION_DIM,
    NANO_ACTION_CHUNK_STEPS,
    NANO_ACTION_SHAPE,
    NANO_POLICY_ID,
    PolicyRuntimeBinding,
    canonical_json_bytes,
    sha256_bytes,
)
from experiments.online_correction_v4.droid_policy_request import (
    PolicyInfraInvalidError,
    build_v4_request_envelope,
    missing_future_artifact,
    observation_packed_request,
    request_audit_projection,
)
from experiments.online_correction_v4.droid_transport import EpisodePolicyTransport


class NanoPolicyContractError(RuntimeError):
    """Raised when Nano request/response contracts diverge."""


Transport = Callable[[dict[str, Any]], Mapping[str, Any]]


def _normalize_action_chunk(raw: Any, expected_shape: tuple[int, int]) -> tuple[tuple[float, ...], ...]:
    tolist = getattr(raw, "tolist", None)
    if callable(tolist):
        raw = tolist()
    rows, cols = expected_shape
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise NanoPolicyContractError(f"action chunk must be a nested sequence {expected_shape}")
    if len(raw) != rows:
        raise NanoPolicyContractError(f"action chunk must be shape {expected_shape}, got {len(raw)} rows")
    normalized: list[tuple[float, ...]] = []
    for row in raw:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != cols:
            raise NanoPolicyContractError(f"action chunk must be shape {expected_shape}")
        values: list[float] = []
        for value in row:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise NanoPolicyContractError("action chunk must contain finite floats")
            values.append(float(value))
        normalized.append(tuple(values))
    return tuple(normalized)


def _hash_action_chunk(actions: tuple[tuple[float, ...], ...]) -> str:
    return sha256_bytes(canonical_json_bytes(actions))


def _encode_future_payload(normalized_future: Any) -> tuple[bytes, str, str]:
    try:
        import io

        import numpy as np

        array = np.asarray(normalized_future, dtype=np.uint8)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, rgb_future=array)
        payload = buffer.getvalue()
        return payload, sha256_bytes(payload), "decoded_rgb_future_npz"
    except ImportError:
        import zlib

        payload = zlib.compress(canonical_json_bytes(normalized_future), level=9)
        return payload, sha256_bytes(payload), "decoded_rgb_future_zjson"


def _normalize_future(raw: Any) -> Any:
    shape = getattr(raw, "shape", None)
    if shape is not None:
        observed_shape = tuple(int(size) for size in shape)
        if (
            len(observed_shape) != 4
            or observed_shape[0] != 33
            or observed_shape[-1] != 3
        ):
            raise NanoPolicyContractError(
                "every exposed Nano future must have shape (33, H, W, 3)"
            )
        if str(getattr(raw, "dtype", "")) != "uint8":
            raise NanoPolicyContractError(
                "array-backed Nano futures must use uint8 RGB values"
            )
        return raw
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise NanoPolicyContractError("future must be a 4D nested sequence")
    if len(raw) != 33:
        raise NanoPolicyContractError("every exposed Nano future must contain 33 frames")
    frames: list[Any] = []
    for frame in raw:
        if not isinstance(frame, Sequence) or isinstance(frame, (str, bytes)):
            raise NanoPolicyContractError("future frame must be a nested sequence")
        rows: list[tuple[tuple[int, ...], ...]] = []
        for row in frame:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
                raise NanoPolicyContractError("future row must be a nested sequence")
            pixels: list[tuple[int, ...]] = []
            for pixel in row:
                if not isinstance(pixel, Sequence) or isinstance(pixel, (str, bytes)) or len(pixel) != 3:
                    raise NanoPolicyContractError("future pixel must be RGB length 3")
                rgb = tuple(int(channel) for channel in pixel)
                pixels.append(rgb)
            rows.append(tuple(pixels))
        frames.append(tuple(rows))
    return tuple(frames)


@dataclass
class NanoRequestRecord:
    request_index: int
    observation_id: str
    observation_sha256: str
    action_step_start: int
    sampling_seed: int
    prompt_sha256: str
    reset_fingerprint_sha256: str
    runtime_identity_sha256: str
    action_sha256: str = ""
    future_sha256: str = ""
    wall_duration_s: float = 0.0
    wire_request_sha256: str = ""


@dataclass
class DroidNanoPolicyAdapter:
    """V4 PolicyAdapter for Cosmos3 Nano with exact 32×8 chunks and content-addressed futures."""

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
    records: list[NanoRequestRecord] = field(default_factory=list)
    persisted_future_digests: list[str] = field(default_factory=list)

    @property
    def policy_id(self) -> str:
        return NANO_POLICY_ID

    def reset(self, *, policy_seed: int, prompt_text: str) -> None:
        if policy_seed != self.policy_seed:
            raise NanoPolicyContractError("policy_seed changed across session reset")
        if prompt_text != self.prompt_text:
            raise NanoPolicyContractError("static episode prompt changed across session reset")
        self.request_count = 0
        self.records.clear()
        self.persisted_future_digests.clear()
        if isinstance(self.transport, EpisodePolicyTransport):
            self.transport.begin_episode()

    def infer(self, observation: ObservationPacket) -> PolicyResponse:
        if observation.payload_sha256 != sha256_bytes(observation.payload):
            raise NanoPolicyContractError("observation payload hash mismatch")
        reset_hash = self._ensure_reset()
        action_step_start = self._action_step_start()
        packed = observation_packed_request(observation)
        audit = {
            "schema_version": "v4-droid-nano-request-v1",
            "policy_id": NANO_POLICY_ID,
            "episode_id": self.episode_id,
            "observation_id": observation.observation_id,
            "observation_capture_time_s": observation.capture_time_s,
            "observation_sha256": observation.payload_sha256,
            "instruction": self.prompt_text,
            "prompt_sha256": self.prompt_sha256,
            "sampling_seed": self.policy_seed,
            "request_index": self.request_count,
            "action_step_start": action_step_start,
            "reset_fingerprint_sha256": reset_hash,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "checkpoint_sha256": self.binding.checkpoint_sha256,
        }
        wire, full_request = build_v4_request_envelope(
            policy_id=NANO_POLICY_ID,
            packed=packed,
            audit=audit,
        )
        started = time.monotonic()
        response = self._invoke_transport(full_request)
        wall_duration_s = time.monotonic() - started
        actions = _normalize_action_chunk(response.get("action"), NANO_ACTION_SHAPE)
        if response.get("sampling_seed") != self.policy_seed:
            raise NanoPolicyContractError("Nano server did not echo the released sampling seed")
        future_sha = ""
        future = response.get("video")
        future_artifact: FutureArtifact | None = None
        if future is None:
            future_artifact = missing_future_artifact(
                request_index=self.request_count,
                observation_id=observation.observation_id,
            )
            self.persisted_future_digests.append(future_artifact.payload_sha256)
            raise PolicyInfraInvalidError(
                InfraInvalidReason.MISSING_MANDATORY_STREAM,
                "Nano server returned no decodable future on the exposed interface",
                future_artifact=future_artifact,
            )
        normalized_future = _normalize_future(future)
        future_payload, future_sha, future_kind = _encode_future_payload(normalized_future)
        self.persisted_future_digests.append(future_sha)
        action_sha = _hash_action_chunk(actions)
        record = NanoRequestRecord(
            request_index=self.request_count,
            observation_id=observation.observation_id,
            observation_sha256=observation.payload_sha256,
            action_step_start=action_step_start,
            sampling_seed=self.policy_seed,
            prompt_sha256=self.prompt_sha256,
            reset_fingerprint_sha256=reset_hash,
            runtime_identity_sha256=self.runtime_identity_sha256,
            action_sha256=action_sha,
            future_sha256=future_sha,
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
                    "action_sha256": action_sha,
                    "future_sha256": future_sha,
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
            generated_horizon=NANO_ACTION_CHUNK_STEPS,
            future_artifact=FutureArtifact(
                kind=future_kind,
                payload=future_payload,
                payload_sha256=future_sha,
            ),
        )

    def _action_step_start(self) -> int:
        if self.executed_action_count is not None:
            return int(self.executed_action_count())
        return self.request_count * NANO_ACTION_CHUNK_STEPS

    def _ensure_reset(self) -> str:
        if self.ensure_reset_attestation is not None:
            produced = self.ensure_reset_attestation()
            if self.reset_fingerprint_sha256 and produced != self.reset_fingerprint_sha256:
                raise NanoPolicyContractError("reset attestation fingerprint mismatch")
            if not self.reset_fingerprint_sha256:
                self.reset_fingerprint_sha256 = produced
        if not self.reset_fingerprint_sha256:
            raise NanoPolicyContractError("reset attestation is required before request zero")
        return self.reset_fingerprint_sha256

    def _invoke_transport(self, request: dict[str, Any]) -> Mapping[str, Any]:
        if self.transport is None:
            raise NanoPolicyContractError(
                "Nano transport is not bound; live inference is blocked in this checkout"
            )
        response = self.transport(request)
        if not isinstance(response, Mapping):
            raise NanoPolicyContractError("Nano transport response must be an object")
        return response

    def close(self) -> None:
        if isinstance(self.transport, EpisodePolicyTransport):
            self.transport.close()


def fake_nano_transport(seed: int) -> Transport:
    """Deterministic transport for contract tests."""

    def _transport(request: Mapping[str, Any]) -> dict[str, Any]:
        index = int(request["request_index"])
        action = [[0.01 * (index + 1) if col == 0 else 0.0 for col in range(ACTION_DIM)] for _ in range(NANO_ACTION_CHUNK_STEPS)]
        future = [
            [
                [((seed + index) % 256, 0, 0) for _ in range(64)]
                for _ in range(64)
            ]
            for _ in range(33)
        ]
        return {"action": action, "video": future, "sampling_seed": seed}

    return _transport


def executed_action_dim() -> int:
    return ACTION_DIM
