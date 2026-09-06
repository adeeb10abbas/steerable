"""Hash-pinned reset registry loader for V4 DROID fixtures."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from experiments.online_correction_v4.droid_task_files.binding import sha256_file
from experiments.online_correction_v4.droid_task_files.constants import (
    ENV_RESET_REGISTRY,
    ENV_RESET_REGISTRY_SHA256,
    FixtureObjectSpec,
    fixture_object_spec,
)


class ResetRegistryError(ValueError):
    """Raised when the reset registry is missing, stale, or incomplete."""


MODEL_BLIND_CANDIDATE_STATUS = "model_blind_candidate_not_released_for_inference"
RELEASED_FOR_POLICY_STATUS = "released_for_policy_inference"
KNOWN_REGISTRY_STATUSES = frozenset(
    {MODEL_BLIND_CANDIDATE_STATUS, RELEASED_FOR_POLICY_STATUS}
)


@dataclass(frozen=True)
class ObjectRoleBinding:
    role: str
    scene_object: str
    asset_identity: str


@dataclass(frozen=True)
class ResetRegistry:
    schema_version: str
    fixture_id: str
    status: str
    model_request_count: int
    behavioral_episode_count: int
    scene_asset: str
    scene_metadata_sha256: str
    contact_objects: tuple[str, ...]
    object_roles: dict[str, ObjectRoleBinding]
    positions_by_env_seed: dict[int, dict[str, tuple[float, float, float]]]
    registry_path: str
    registry_sha256: str


def _fail(message: str) -> None:
    raise ResetRegistryError(message)


def _asset_identity(spec: FixtureObjectSpec, object_name: str) -> str:
    return f"{spec.scene_asset}::{object_name}@{spec.scene_metadata_sha256}"


def _parse_positions(
    raw: Mapping[str, Any],
    *,
    env_seed: int,
    movable_objects: tuple[str, ...],
) -> dict[str, tuple[float, float, float]]:
    positions = raw.get("positions_robot_base_m")
    if not isinstance(positions, dict):
        _fail(f"reset {env_seed} lacks positions_robot_base_m")
    if set(positions) != set(movable_objects):
        _fail(f"reset {env_seed} movable-object inventory mismatch")
    parsed: dict[str, tuple[float, float, float]] = {}
    for name in movable_objects:
        item = positions[name]
        if not isinstance(item, list) or len(item) != 3:
            _fail(f"reset {env_seed} position for {name} must be a 3-vector")
        parsed[name] = tuple(float(value) for value in item)
    return parsed


def load_reset_registry(
    *,
    registry_path: str | None = None,
    registry_sha256: str | None = None,
    required_status: str | None = None,
    expected_fixture_id: str = "horizontal",
) -> ResetRegistry:
    try:
        spec = fixture_object_spec(expected_fixture_id)
    except ValueError as exc:
        raise ResetRegistryError(str(exc)) from exc
    raw_path = registry_path or os.environ.get(ENV_RESET_REGISTRY)
    expected_hash = registry_sha256 or os.environ.get(ENV_RESET_REGISTRY_SHA256)
    if not raw_path or not expected_hash:
        _fail(f"{ENV_RESET_REGISTRY} and {ENV_RESET_REGISTRY_SHA256} are required")
    path = Path(raw_path).resolve()
    if not path.is_file():
        _fail(f"reset registry path does not exist: {path}")
    digest = sha256_file(path)
    if digest != expected_hash:
        _fail("reset registry digest mismatch")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResetRegistryError(f"cannot read reset registry: {exc}") from exc
    if not isinstance(payload, dict):
        _fail("reset registry must be a JSON object")
    if payload.get("schema_version") != spec.reset_registry_schema:
        _fail("reset registry schema mismatch")
    if payload.get("fixture_id") != expected_fixture_id:
        _fail(f"reset registry fixture_id must be {expected_fixture_id}")
    status = payload.get("status")
    if status not in KNOWN_REGISTRY_STATUSES:
        _fail("reset registry status is missing or unrecognized")
    if required_status is not None and status != required_status:
        _fail(
            f"reset registry status {status!r} differs from required "
            f"{required_status!r}"
        )
    if payload.get("model_request_count") != 0:
        _fail("reset registry must be built without model requests")
    if payload.get("behavioral_episode_count") != 0:
        _fail("reset registry must not contain behavioral episodes")
    if payload.get("scene_asset") != spec.scene_asset:
        _fail("reset registry scene_asset mismatch")
    if payload.get("scene_metadata_sha256") != spec.scene_metadata_sha256:
        _fail("reset registry scene_metadata_sha256 mismatch")

    contact_objects = payload.get("contact_objects")
    if contact_objects != list(spec.contact_objects):
        _fail("reset registry contact_objects must match the frozen fixture inventory")

    roles_raw = payload.get("object_roles")
    if not isinstance(roles_raw, dict):
        _fail("reset registry object_roles must be an object")
    expected_roles = {
        "target": spec.target_object,
        "reference": spec.reference_object,
    }
    if spec.distractor_object is not None:
        expected_roles["distractor"] = spec.distractor_object
    if set(roles_raw) != set(expected_roles):
        _fail("reset registry object_roles differ from the frozen fixture roles")
    object_roles: dict[str, ObjectRoleBinding] = {}
    for role, scene_object in expected_roles.items():
        binding = roles_raw.get(role)
        if not isinstance(binding, dict):
            _fail(f"reset registry object_roles.{role} is incomplete")
        asset = binding.get("scene_object")
        asset_identity = binding.get("asset_identity")
        if asset != scene_object:
            _fail(f"reset registry object_roles.{role}.scene_object mismatch")
        if asset_identity != _asset_identity(spec, scene_object):
            _fail(f"reset registry object_roles.{role}.asset_identity mismatch")
        object_roles[role] = ObjectRoleBinding(
            role=role,
            scene_object=scene_object,
            asset_identity=asset_identity,
        )

    resets_raw = payload.get("resets_by_env_seed")
    if not isinstance(resets_raw, dict) or not resets_raw:
        _fail("reset registry resets_by_env_seed must be a nonempty object")
    positions_by_env_seed: dict[int, dict[str, tuple[float, float, float]]] = {}
    for key, value in resets_raw.items():
        try:
            env_seed = int(key)
        except (TypeError, ValueError) as exc:
            raise ResetRegistryError(f"reset registry env_seed key is not an integer: {key!r}") from exc
        if not isinstance(value, dict):
            _fail(f"reset registry entry for env_seed {env_seed} must be an object")
        positions_by_env_seed[env_seed] = _parse_positions(
            value,
            env_seed=env_seed,
            movable_objects=spec.movable_objects,
        )

    return ResetRegistry(
        schema_version=str(payload["schema_version"]),
        fixture_id=str(payload["fixture_id"]),
        status=str(status),
        model_request_count=0,
        behavioral_episode_count=0,
        scene_asset=str(payload["scene_asset"]),
        scene_metadata_sha256=str(payload["scene_metadata_sha256"]),
        contact_objects=tuple(str(item) for item in contact_objects),
        object_roles=object_roles,
        positions_by_env_seed=positions_by_env_seed,
        registry_path=str(path),
        registry_sha256=digest,
    )
