"""Admission for A003 retry and outcome-blind A004 continuation gates."""

from __future__ import annotations

from pathlib import Path

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
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


CONTINUATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v4-continuation-gate-v1"
A003_SHA = "7b0835c2bb76631add47f5e13c6db4d5be40379d234e90c4b401a5214ec2463d"


def require_a004_gate(*, registration_path: Path, queue_path: Path, release_gate_path: Path):
    gate = read_finite_json(release_gate_path)
    require(isinstance(gate, dict), "A004 gate is not an object")
    if gate.get("schema_version") == A003_SCHEMA:
        require(sha256_file(release_gate_path) == A003_SHA, "A003 retry gate bytes changed")
        return require_released_replacement_gate(
            registration_path=registration_path,
            queue_path=queue_path,
            release_gate_path=release_gate_path,
        )
    require(
        gate.get("schema_version") == CONTINUATION_SCHEMA
        and gate.get("status") == "passed_outcome_blind_a004_continuation_release"
        and gate.get("passed") is True,
        "gate is neither exact A003 nor passed A004 continuation",
    )
    parent, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    a003 = validate_file_binding(gate.get("a003_release"), "A004 continuation A003 release")
    require(a003["sha256"] == A003_SHA, "A004 continuation changed A003")
    require(gate.get("queue", {}).get("sha256") == sha256_file(queue_path), "A004 continuation queue changed")
    require(gate.get("outcome_fields_read") is False and gate.get("no_cross_lane_failover") is True, "A004 continuation is not outcome blind/isolated")
    require(gate.get("completed_blocks_never_rerun") is True, "A004 continuation permits reruns")
    return parent, cells, gate
