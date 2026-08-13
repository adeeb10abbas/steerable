"""Universal A004 admission for original, replacement, and continuation lanes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    load_cells,
    read_finite_json,
    require,
    sha256_file,
    validate_file_binding,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.activation_v3_lane_replacement_runner import (
    SCHEMA as A003_SCHEMA,
    require_released_replacement_gate,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import (
    LANE_SCHEMA,
    RELEASE_SCHEMA as ORIGINAL_RELEASE_SCHEMA,
    require_released_gate as require_original_released_gate,
    validate_assignment,
)


CONTINUATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v4-continuation-gate-v1"
ORIGINAL_RELEASE_SHA = "28ee96e3fcda4637302aadef6e90233574101a75e5c7af75fa1d9e4b1c060a7d"
A003_SHA = "7b0835c2bb76631add47f5e13c6db4d5be40379d234e90c4b401a5214ec2463d"
ALL_SLOTS = tuple(f"repair-lane-{index:02d}" for index in range(8))
ORIGINAL_SLOTS = frozenset(ALL_SLOTS[2:])
REPLACEMENT_SLOTS = frozenset(ALL_SLOTS[:2])


def _gate(path: Path) -> dict[str, Any]:
    value = read_finite_json(path)
    require(isinstance(value, dict), "A004 gate is not an object")
    return value


def gate_kind(path: Path) -> str:
    """Classify only the three frozen A004 admission families."""
    value = _gate(path)
    digest = sha256_file(path)
    if digest == ORIGINAL_RELEASE_SHA:
        require(value.get("schema_version") == ORIGINAL_RELEASE_SCHEMA and value.get("status") == "passed_homogeneous_block_local_behavioral_release" and value.get("passed") is True, "original release28ee content changed")
        return "original"
    if digest == A003_SHA:
        require(value.get("schema_version") == A003_SCHEMA and value.get("status") == "passed_activation_v3_cluster_termination_lane_replacement" and value.get("passed") is True, "A003 release7b08 content changed")
        return "a003"
    require(value.get("schema_version") == CONTINUATION_SCHEMA and value.get("status") == "passed_outcome_blind_a004_continuation_release" and value.get("passed") is True, "gate is not exact original release28ee, exact A003 release7b08, or A004 continuation")
    return "continuation"


def admitted_slots(path: Path) -> frozenset[str]:
    kind = gate_kind(path)
    return ORIGINAL_SLOTS if kind == "original" else REPLACEMENT_SLOTS if kind == "a003" else frozenset(ALL_SLOTS)


def lane_records(gate: Mapping[str, Any]) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    records = gate.get("lane_manifests")
    require(isinstance(records, list) and len(records) == 8, "A004 gate lacks eight lane manifests")
    result: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for record in records:
        binding = validate_file_binding(record, "A004 lane manifest")
        value = read_finite_json(Path(binding["path"]))
        require(isinstance(value, dict) and value.get("schema_version") == LANE_SCHEMA and value.get("status") == "passed_single_lane_release" and value.get("passed") is True, "A004 lane manifest is not released")
        slot = value.get("lane_slot")
        require(isinstance(slot, str) and slot in ALL_SLOTS and slot not in result, "A004 lane slot set is invalid")
        result[slot] = (binding, value)
    require(tuple(sorted(result)) == ALL_SLOTS, "A004 lane manifests do not cover all eight slots")
    return result


def _verify_continuation(gate: Mapping[str, Any], *, queue_path: Path) -> None:
    a003 = validate_file_binding(gate.get("a003_release"), "A004 continuation A003 release")
    original = validate_file_binding(gate.get("original_release"), "A004 continuation original release")
    require(a003["sha256"] == A003_SHA and original["sha256"] == ORIGINAL_RELEASE_SHA, "A004 continuation changed a frozen predecessor gate")
    a003_gate, original_gate = _gate(Path(a003["path"])), _gate(Path(original["path"]))
    require(gate.get("queue", {}).get("sha256") == sha256_file(queue_path), "A004 continuation queue changed")
    require(gate.get("outcome_fields_read") is False and gate.get("no_cross_lane_failover") is True and gate.get("completed_blocks_never_rerun") is True and gate.get("science_unchanged") is True, "A004 continuation execution contract changed")
    expected: dict[str, str] = {}
    for slot, (binding, _) in lane_records(a003_gate).items():
        if slot in REPLACEMENT_SLOTS:
            expected[slot] = binding["sha256"]
    for slot, (binding, _) in lane_records(original_gate).items():
        if slot in ORIGINAL_SLOTS:
            expected[slot] = binding["sha256"]
    actual = {slot: binding["sha256"] for slot, (binding, _) in lane_records(gate).items()}
    require(actual == expected, "A004 continuation lane identity is not A003[00,01] plus release28ee[02..07]")


def require_a004_gate(*, registration_path: Path, queue_path: Path, release_gate_path: Path):
    """Admission used by the child adapter before every model request."""
    path = Path(release_gate_path)
    kind = gate_kind(path)
    if kind == "original":
        return require_original_released_gate(
            registration_path=registration_path,
            queue_path=queue_path,
            release_gate_path=path,
        )
    if kind == "a003":
        return require_released_replacement_gate(
            registration_path=registration_path,
            queue_path=queue_path,
            release_gate_path=path,
        )
    gate = _gate(path)
    parent, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    _verify_continuation(gate, queue_path=queue_path)
    validate_assignment(gate.get("assignment_manifest"))
    return parent, cells, gate
