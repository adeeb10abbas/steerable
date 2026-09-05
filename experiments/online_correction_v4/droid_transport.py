"""Direct websocket transport for V4 DROID policy servers."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from experiments.online_correction_v4.droid_contract import NANO_POLICY_ID, PI05_POLICY_ID
from experiments.online_correction_v4.droid_policy_request import (
    ServerEnvelopeGateError,
    extract_server_wire_request,
    normalize_pi05_response,
)


class TransportError(RuntimeError):
    """Raised when live policy transport cannot be constructed or used."""


class EpisodePolicyTransport:
    """One fresh client/session per episode because released servers expose no reset RPC."""

    def __init__(self, *, policy_id: str, host: str, port: int) -> None:
        self.policy_id = policy_id
        self.host = host
        self.port = port
        self._client: Any = None

    def begin_episode(self) -> None:
        self.close()
        if self.policy_id == NANO_POLICY_ID:
            self._client = _create_nano_client(host=self.host, port=self.port)
        else:
            self._client = _create_pi05_client(host=self.host, port=self.port)

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def __call__(self, request: dict[str, Any]) -> Mapping[str, Any]:
        if self._client is None:
            raise TransportError("policy transport session is not open; call begin_episode() after reset")
        wire = extract_server_wire_request(request, policy_id=self.policy_id)
        try:
            response = self._client._query_server(wire)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if _looks_like_envelope_rejection(message):
                raise ServerEnvelopeGateError(
                    "released policy server rejected the V4 wire envelope; "
                    "cluster qualification must attest V4 metadata support",
                    server_detail=message,
                ) from exc
            raise
        if not isinstance(response, Mapping):
            raise TransportError("policy server response must be an object")
        if self.policy_id == PI05_POLICY_ID:
            return normalize_pi05_response(response)
        return response


def build_live_transport(*, policy_id: str, host: str, port: int) -> EpisodePolicyTransport:
    if policy_id not in {NANO_POLICY_ID, PI05_POLICY_ID}:
        raise TransportError(f"unsupported live transport policy_id: {policy_id}")
    try:
        if policy_id == NANO_POLICY_ID:
            _create_nano_client(host=host, port=port)
        else:
            _create_pi05_client(host=host, port=port)
    except ImportError as exc:
        raise TransportError(
            f"policy client for {policy_id} is unavailable in this checkout"
        ) from exc
    return EpisodePolicyTransport(policy_id=policy_id, host=host, port=port)


def _create_nano_client(*, host: str, port: int) -> Any:
    from policies.cosmos3.client import Cosmos3Client

    return Cosmos3Client(remote_host=host, remote_port=port)


def _create_pi05_client(*, host: str, port: int) -> Any:
    from policies.pi0_family.client import Pi0DroidJointposClient

    return Pi0DroidJointposClient(
        remote_host=host,
        remote_port=port,
        policy_variant="pi05",
    )


def _looks_like_envelope_rejection(message: str) -> bool:
    lowered = message.lower()
    needles = (
        "unknown field",
        "unexpected key",
        "invalid request",
        "unrecognized",
        "schema",
        "amendment_id",
        "registered_cell_id",
    )
    return any(item in lowered for item in needles)
