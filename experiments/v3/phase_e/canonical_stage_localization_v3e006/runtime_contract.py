"""Fail-closed exact π0.5 runtime contract for V3-E006."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "vla-wam-shared-v3e006-exact-pi05-runtime-contract-v1"
STATUS = "prospectively_frozen_before_v3e006_registration_or_model_request"


class RuntimeContractError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _verify_bindings(value: Any, *, study_root: Path, external_roots: tuple[Path, ...]) -> None:
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"} <= set(value):
            raw = Path(str(value["path"]))
            candidate = raw if raw.is_absolute() else study_root / raw
            resolved = candidate.resolve()
            permitted = resolved.is_relative_to(study_root) or any(resolved.is_relative_to(root) for root in external_roots)
            if not permitted or not resolved.is_file():
                raise RuntimeContractError(f"runtime source path is missing/out of scope: {resolved}")
            if resolved.stat().st_size != value["bytes"] or hashlib.sha256(resolved.read_bytes()).hexdigest() != value["sha256"]:
                raise RuntimeContractError(f"runtime source bytes/hash differ: {resolved}")
        for item in value.values():
            _verify_bindings(item, study_root=study_root, external_roots=external_roots)
    elif isinstance(value, list):
        for item in value:
            _verify_bindings(item, study_root=study_root, external_roots=external_roots)


def load_runtime_contract(
    path: Path,
    expected_file_sha256: str,
    *,
    study_root: Path | None = None,
    external_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected_file_sha256:
        raise RuntimeContractError("runtime contract file is missing or changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA or value.get("status") != STATUS:
        raise RuntimeContractError("runtime contract schema/status differs")
    claimed = value.get("canonical_contract_sha256")
    payload = {key: item for key, item in value.items() if key != "canonical_contract_sha256"}
    if claimed != canonical_sha256(payload):
        raise RuntimeContractError("canonical runtime-contract digest differs")
    required = {
        "checkpoint",
        "policy_server",
        "policy_client",
        "action_controller",
        "cameras",
        "success_predicate_and_scorer",
        "layout",
        "renderer",
        "raw_writer",
    }
    components = value.get("components")
    if not isinstance(components, Mapping) or set(components) != required:
        raise RuntimeContractError("runtime component set differs")
    for name, component in components.items():
        if component.get("canonical_sha256") != canonical_sha256(component.get("contract")):
            raise RuntimeContractError(f"runtime component digest differs: {name}")
    if study_root is not None:
        _verify_bindings(
            value,
            study_root=Path(study_root).resolve(),
            external_roots=tuple(Path(root).resolve() for root in external_roots),
        )
    return value


def assert_observed_runtime(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    required = expected_observation(expected)
    keys = (
        "canonical_contract_sha256",
        "model_id",
        "checkpoint_manifest_sha256",
        "checkpoint_payload_sha256",
        "openpi_commit",
        "robolab_commit",
        "action_chunk_shape",
        "open_loop_horizon",
        "action_cap",
        "policy_id",
        "renderer_contract",
        "component_canonical_sha256",
        "component_source_bindings",
        "layout_candidate_sha256",
        "camera_environment_set",
        "camera_policy_feeds",
        "success_predicate_ids",
    )
    for key in keys:
        if observed.get(key) != required.get(key):
            raise RuntimeContractError(f"observed runtime differs at {key}")


def expected_observation(value: Mapping[str, Any]) -> dict[str, Any]:
    components = value["components"]
    return {
        "canonical_contract_sha256": value["canonical_contract_sha256"],
        "model_id": value["model_id"],
        "checkpoint_manifest_sha256": value["checkpoint_manifest_sha256"],
        "checkpoint_payload_sha256": value["checkpoint_payload_sha256"],
        "openpi_commit": value["openpi_commit"],
        "robolab_commit": value["robolab_commit"],
        "action_chunk_shape": value["action_chunk_shape"],
        "open_loop_horizon": value["open_loop_horizon"],
        "action_cap": value["action_cap"],
        "policy_id": value["policy_id"],
        "renderer_contract": value["renderer_contract"],
        "component_canonical_sha256": {
            name: component["canonical_sha256"] for name, component in sorted(components.items())
        },
        "component_source_bindings": {
            name: component["contract"] for name, component in sorted(components.items())
        },
        "layout_candidate_sha256": components["layout"]["contract"]["candidate"]["sha256"],
        "camera_environment_set": components["cameras"]["contract"]["environment_preset_order"],
        "camera_policy_feeds": components["cameras"]["contract"]["policy_feeds"],
        "success_predicate_ids": components["success_predicate_and_scorer"]["contract"]["success_predicate_ids"],
    }
