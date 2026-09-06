"""Shared helpers for V4 DROID policy request packing and server wire envelopes."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from experiments.online_correction_v4.adapters import FutureArtifact, ObservationPacket
from experiments.online_correction_v4.attempts import InfraInvalidReason
from experiments.online_correction_v4.droid_contract import NANO_POLICY_ID, PI05_POLICY_ID, sha256_bytes


class PolicyInfraInvalidError(Exception):
    """Policy transport or mandatory decode surface failed with inputs intact."""

    def __init__(
        self,
        reason: InfraInvalidReason,
        message: str,
        *,
        future_artifact: FutureArtifact | None = None,
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.future_artifact = future_artifact


class ServerEnvelopeGateError(PolicyInfraInvalidError):
    """Pinned Phase-B server rejected the V4 wire envelope."""

    def __init__(self, message: str, *, server_detail: str | None = None) -> None:
        super().__init__(
            InfraInvalidReason.MALFORMED_ACTION_INTERFACE,
            message,
        )
        self.server_detail = server_detail


# Keys produced by the official Cosmos / π0.5 pack helpers.
_OBSERVATION_KEY_PREFIX = "observation/"
_NANO_WIRE_KEYS = frozenset(
    {
        "prompt",
        "sampling_seed",
        "action_step_start",
    }
)
_PI05_WIRE_KEYS = frozenset(
    {
        "prompt",
        "sampling_seed",
        "action_step_start",
    }
)

# V4 audit-only metadata never sent to released servers as V3 cell identity.
V4_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "episode_id",
        "observation_id",
        "observation_capture_time_s",
        "observation_sha256",
        "instruction",
        "prompt_sha256",
        "request_index",
        "reset_fingerprint_sha256",
        "runtime_identity_sha256",
        "checkpoint_sha256",
        "future_interface",
        "missing_future_policy",
    }
)


def normalize_pi05_response(response: Mapping[str, Any]) -> dict[str, Any]:
    """π0.5 servers may return ``actions`` instead of the adapter's ``action`` key."""
    normalized = dict(response)
    if "action" not in normalized and "actions" in normalized:
        normalized["action"] = normalized["actions"]
    return normalized


def observation_packed_request(observation: ObservationPacket) -> dict[str, Any]:
    """Recover the server-ready observation subtree from a captured packet."""
    native = observation.native_input
    if isinstance(native, Mapping):
        packed = native.get("packed_request")
        if isinstance(packed, Mapping):
            return dict(packed)
    try:
        decoded = json.loads(observation.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(decoded, Mapping):
        return {}
    packed = decoded.get("packed_request")
    if isinstance(packed, Mapping):
        return dict(packed)
    return {}


def request_audit_projection(value: Any) -> Any:
    """Project array-bearing policy inputs into compact, canonical JSON evidence."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): request_audit_projection(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return [request_audit_projection(item) for item in value]
    host = value
    if hasattr(host, "detach"):
        host = host.detach()
    if hasattr(host, "cpu"):
        host = host.cpu()
    if hasattr(host, "numpy"):
        host = host.numpy()
    shape = getattr(host, "shape", None)
    tobytes = getattr(host, "tobytes", None)
    if shape is not None and callable(tobytes):
        payload = tobytes()
        return {
            "encoding": "array_sha256",
            "shape": [int(size) for size in shape],
            "dtype": str(getattr(host, "dtype", "unknown")),
            "sha256": sha256_bytes(payload),
            "size_bytes": len(payload),
        }
    item = getattr(host, "item", None)
    if callable(item):
        return request_audit_projection(item())
    raise TypeError(
        f"policy request value is not audit-projectable: {type(value).__name__}"
    )


def build_v4_request_envelope(
    *,
    policy_id: str,
    packed: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge packed obs/proprio with V4 audit metadata while preserving wire split."""
    wire = extract_server_wire_request({**packed, **audit}, policy_id=policy_id)
    full = dict(packed)
    full.update(audit)
    return wire, full


def extract_server_wire_request(request: Mapping[str, Any], *, policy_id: str) -> dict[str, Any]:
    """Return only the keys accepted by released Cosmos3 / π0.5 websocket servers."""
    allowed = _NANO_WIRE_KEYS if policy_id == NANO_POLICY_ID else _PI05_WIRE_KEYS
    wire: dict[str, Any] = {}
    for key, value in request.items():
        if key.startswith(_OBSERVATION_KEY_PREFIX) or key in allowed:
            wire[key] = value
    if "prompt" not in wire:
        instruction = request.get("instruction")
        if isinstance(instruction, str):
            wire["prompt"] = instruction
    return wire


def missing_future_artifact(*, request_index: int, observation_id: str) -> FutureArtifact:
    payload = json.dumps(
        {
            "schema_version": "v4-droid-missing-nano-future-v1",
            "request_index": request_index,
            "observation_id": observation_id,
            "future_evidence_status": "missing_on_exposed_interface",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return FutureArtifact(
        kind="missing_future",
        payload=payload,
        payload_sha256=sha256_bytes(payload),
    )
