"""Pack RoboLab observations into pinned Cosmos3 / π0.5 request envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from experiments.online_correction_v4.droid_contract import NANO_POLICY_ID, PI05_POLICY_ID


class ObservationPackError(RuntimeError):
    """Raised when native RoboLab observations cannot be packed for a policy."""


@runtime_checkable
class ObservationPacker(Protocol):
    def pack(self, native_obs: Any, instruction: str) -> dict[str, Any]:
        """Return the server-ready observation subtree for one policy request."""


@dataclass
class LazyPolicyObservationPacker:
    """Lazy-import policy clients and reuse their exact extract/pack helpers."""

    policy_id: str
    _packer: ObservationPacker | None = None

    def pack(self, native_obs: Any, instruction: str) -> dict[str, Any]:
        if self._packer is None:
            self._packer = _build_packer(self.policy_id)
        return self._packer.pack(native_obs, instruction)


def pack_policy_request(
    *,
    policy_id: str,
    native_obs: Any,
    instruction: str,
    packer: ObservationPacker | None = None,
) -> dict[str, Any]:
    if native_obs is None:
        raise ObservationPackError("native observation input is required for live policy packing")
    backend = packer or LazyPolicyObservationPacker(policy_id=policy_id)
    packed = backend.pack(native_obs, instruction)
    if not isinstance(packed, Mapping):
        raise ObservationPackError("packed observation must be a mapping")
    return dict(packed)


def _build_packer(policy_id: str) -> ObservationPacker:
    if policy_id == NANO_POLICY_ID:
        return _NanoClientPacker()
    if policy_id == PI05_POLICY_ID:
        return _Pi05ClientPacker()
    raise ObservationPackError(f"unsupported policy for observation packing: {policy_id}")


class _NanoClientPacker:
    def __init__(self) -> None:
        try:
            from policies.cosmos3.client import Cosmos3Client
        except ImportError as exc:  # pragma: no cover - cluster runtime owns deps
            raise ObservationPackError(
                "Cosmos3Client is unavailable; live Nano observation packing is blocked"
            ) from exc
        self._client_cls = Cosmos3Client
        # Match pinned V3: bind image geometry without opening a websocket session.
        packer = object.__new__(Cosmos3Client)
        packer._image_w = Cosmos3Client.IMAGE_W
        packer._image_h = Cosmos3Client.IMAGE_H
        self._packer = packer

    def pack(self, native_obs: Any, instruction: str) -> dict[str, Any]:
        extracted = self._client_cls._extract_observation(self._packer, native_obs)
        return dict(self._client_cls._pack_request(self._packer, extracted, instruction))


class _Pi05ClientPacker:
    def __init__(self) -> None:
        try:
            from policies.pi0_family.client import Pi0DroidJointposClient
        except ImportError as exc:  # pragma: no cover - cluster runtime owns deps
            raise ObservationPackError(
                "Pi0DroidJointposClient is unavailable; live π0.5 observation packing is blocked"
            ) from exc
        self._client_cls = Pi0DroidJointposClient
        self._client = Pi0DroidJointposClient(
            remote_host="127.0.0.1",
            remote_port=0,
            policy_variant="pi05",
        )

    def pack(self, native_obs: Any, instruction: str) -> dict[str, Any]:
        extracted = self._client_cls._extract_observation(self._client, native_obs, env_id=0)
        return dict(self._client_cls._pack_request(self._client, extracted, instruction))


@dataclass
class FakeObservationPacker:
    """Deterministic packer for contract tests without policy dependencies."""

    keys: tuple[str, ...] = (
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    )
    builder: Callable[[Any, str], dict[str, Any]] | None = None

    def pack(self, native_obs: Any, instruction: str) -> dict[str, Any]:
        if self.builder is not None:
            return dict(self.builder(native_obs, instruction))
        tick = native_obs.get("tick") if isinstance(native_obs, Mapping) else None
        return {
            "observation/exterior_image_1_left": [[0, 0, 0]],
            "observation/wrist_image_left": [[0, 0, 0]],
            "observation/joint_position": [0.0] * 7,
            "observation/gripper_position": [0.0],
            "prompt": instruction,
            "fixture_tick": tick,
        }
