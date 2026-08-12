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


def load_runtime_contract(path: Path, expected_file_sha256: str) -> dict[str, Any]:
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
    return value


def assert_observed_runtime(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
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
    )
    for key in keys:
        if observed.get(key) != expected.get(key):
            raise RuntimeContractError(f"observed runtime differs at {key}")

