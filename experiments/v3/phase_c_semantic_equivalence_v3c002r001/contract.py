"""Fail-closed V3-C002-R001 operational-repair contract.

The parent C002 scientific registration and queue remain the estimand source.
R001 changes only the pre-release execution topology after the original
cross-server exact-equality gate failed before behavioral execution.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
from typing import Any, Mapping

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    load_cells,
    read_finite_json,
    require,
    sha256_file,
    validate_file_binding,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
REPAIR_ID = "V3-C002-R001"
REGISTRATION_SCHEMA = "vla-wam-shared-v3c002r001-repair-registration-v1"
SOURCE_GATE_SCHEMA = "vla-wam-shared-v3c002r001-source-push-gate-v1"
PHYSICAL_GATE_SCHEMA = "vla-wam-shared-v3c002r001-model-blind-physical-gate-v1"
SMOKE_AUTH_SCHEMA = "vla-wam-shared-v3c002r001-smoke-authorization-v1"
SMOKE_GATE_SCHEMA = "vla-wam-shared-v3c002r001-excluded-smoke-gate-v1"
REPEAT_GATE_SCHEMA = "vla-wam-shared-v3c002r001-single-server-repeat-gate-v1"
LANE_SCHEMA = "vla-wam-shared-v3c002r001-single-lane-manifest-v1"
RELEASE_SCHEMA = "vla-wam-shared-v3c002r001-release-gate-v1"


def repo_binding(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file(), f"missing repository artifact: {path}")
    try:
        relative = path.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ContractError(f"repair artifact is outside repository: {path}") from exc
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def resolve(record: Mapping[str, Any]) -> Path:
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else REPO_ROOT / path


def bound_json(record: Any, label: str, *, schema: str, status: str) -> dict[str, Any]:
    binding = validate_file_binding(record, label)
    value = read_finite_json(Path(binding["path"]))
    require(isinstance(value, dict), f"{label} is not an object")
    require(value.get("schema_version") == schema, f"{label} schema changed")
    require(value.get("status") == status and value.get("passed") is True, f"{label} did not pass")
    return value


def load_repair(*, registration_path: Path, queue_path: Path) -> tuple[dict[str, Any], list[Any]]:
    repair = read_finite_json(registration_path)
    require(isinstance(repair, dict) and repair.get("schema_version") == REGISTRATION_SCHEMA, "repair registration schema changed")
    require(repair.get("repair_id") == REPAIR_ID and repair.get("status") == "registered_prospective_post_gate_repair", "repair registration is not active")
    require(repair.get("behavioral_episodes_before_registration") == 0, "repair was not registered before behavior")
    require(repair.get("user_authorized_protocol_override") is True, "repair lacks user override authorization")
    parent_registration = validate_file_binding(repair.get("parent_registration"), "parent C002 registration")
    parent_queue = validate_file_binding(repair.get("parent_queue"), "parent C002 queue")
    require(sha256_file(queue_path) == parent_queue["sha256"], "repair queue is not byte-identical to parent queue")
    parent, cells = load_cells(registration_path=Path(parent_registration["path"]), queue_path=queue_path)
    require(len(cells) == 1364, "repair queue is not the complete registered cohort")
    closure = repair.get("original_failed_isolation_closure")
    require(isinstance(closure, dict), "original closure binding is absent")
    for key in ("evidence_manifest", "failure_report", "target_raw_rehash_receipt"):
        validate_file_binding(closure.get(key), f"original closure {key}")
    require(repair.get("analysis_plan") == parent.get("analysis_plan"), "repair changed the registered analysis plan")
    sources = repair.get("source_bindings")
    require(isinstance(sources, dict) and len(sources) >= 18, "repair source bindings are incomplete")
    for name, binding in sources.items():
        require(isinstance(name, str) and name and not Path(str(binding.get("path", ""))).is_absolute(), f"repair source binding is not portable: {name}")
        validate_file_binding(binding, f"repair source {name}")
    validate_assignment(repair.get("assignment_manifest"))
    topology = repair.get("execution_topology")
    require(
        isinstance(topology, dict)
        and topology.get("policy") == "eight_homogeneous_block_local_lanes_with_exact_within_server_repeat"
        and topology.get("lane_count") == 8
        and topology.get("cross_server_tolerance") is None
        and topology.get("no_failover_within_block") is True
        and topology.get("incomplete_block_retry_same_lane_only") is True,
        "repair execution topology changed",
    )
    technical = repair.get("technical_gate_plan")
    require(isinstance(technical, dict) and technical.get("global_excluded_smoke_seed") == 12000 and technical.get("global_excluded_smoke_lane_slot") == "repair-lane-00" and technical.get("per_lane_physical_gate_count") == 8 and technical.get("per_lane_repeat_gate_count") == 8 and technical.get("repeat_sequence") == ["canonical_left", "canonical_right", "canonical_left"], "repair technical gate plan changed")
    return repair, cells


def validate_assignment(record: Any) -> list[dict[str, Any]]:
    binding = validate_file_binding(record, "repair assignment manifest")
    rows = [json.loads(line) for line in Path(binding["path"]).read_text(encoding="utf-8").splitlines()]
    require(len(rows) == 341, "repair assignment must contain 341 seed blocks")
    require({row.get("episode_seed") for row in rows} == set(range(12000, 12341)), "repair assignment seed set changed")
    require(len({row.get("seed_block_id") for row in rows}) == 341, "repair assignment block IDs are not unique")
    slots = [f"repair-lane-{index:02d}" for index in range(8)]
    ranked = sorted(range(12000, 12341), key=lambda seed: hashlib.sha256(f"V3-C002-R001|balanced-lane-rank-v1|{seed}".encode()).hexdigest())
    expected = {seed: (rank, slots[rank % 8]) for rank, seed in enumerate(ranked)}
    counts = {slot: 0 for slot in slots}
    for row in rows:
        seed = row.get("episode_seed")
        rank, slot = expected[seed]
        require(row.get("schema_version") == "vla-wam-shared-v3c002r001-block-assignment-v1", "repair assignment schema changed")
        require(row.get("repair_id") == REPAIR_ID and row.get("rank") == rank and row.get("lane_slot") == slot, "repair SHA-rank assignment changed")
        require(row.get("block_indivisible") is True and row.get("within_block_request0_bytes_matched") is True and row.get("incomplete_block_retry_same_lane_only") is True, "repair block-local contract changed")
        require(set(row.get("conditions", [])) == {"canonical_left", "inverse_reference_left", "canonical_right", "inverse_reference_right"}, "repair assignment omits a condition")
        counts[slot] += 1
    require(set(counts.values()) == {42, 43}, "repair assignment is not balanced within one block")
    return rows


def verify_pushed_gate(source_gate: Mapping[str, Any], repair: Mapping[str, Any]) -> None:
    require(source_gate.get("schema_version") == SOURCE_GATE_SCHEMA and source_gate.get("status") == "passed_repair_source_and_registration_pushed", "repair source gate did not pass")
    require(source_gate.get("passed") is True and source_gate.get("pushed") is True, "repair source is not pushed")
    require(source_gate.get("repair_registration_sha256") == sha256_file(resolve(source_gate["repair_registration"])), "repair source gate registration changed")
    require(source_gate.get("implementation_commit") == repair.get("runtime", {}).get("repair_wrapper_implementation_commit"), "repair implementation/source gate lineage changed")
    require(source_gate.get("queue", {}).get("sha256") == repair.get("queue", {}).get("sha256") and source_gate.get("assignment_manifest", {}).get("sha256") == repair.get("assignment_manifest", {}).get("sha256"), "repair source gate queue/assignment changed")
    remote, branch = source_gate.get("remote"), source_gate.get("branch")
    require(isinstance(remote, str) and isinstance(branch, str), "repair source remote/branch missing")
    rows = subprocess.run(["git", "ls-remote", "--heads", remote, branch], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.splitlines()
    heads = [row.split()[0] for row in rows if row.split()]
    require(len(heads) == 1, "repair remote branch missing or ambiguous")
    for key in ("implementation_commit", "registration_commit"):
        commit = source_gate.get(key)
        require(isinstance(commit, str) and len(commit) == 40, f"repair {key} invalid")
        require(subprocess.run(["git", "merge-base", "--is-ancestor", commit, heads[0]], cwd=REPO_ROOT).returncode == 0, f"repair {key} is not pushed")


def _repair_from_gate(gate: Mapping[str, Any]) -> tuple[dict[str, Any], list[Any]]:
    registration = validate_file_binding(gate.get("repair_registration"), "repair registration")
    queue = validate_file_binding(gate.get("queue"), "repair queue")
    repair, cells = load_repair(registration_path=Path(registration["path"]), queue_path=Path(queue["path"]))
    return repair, cells


def require_model_blind_preflight_authorization(*, registration_path: Path, queue_path: Path, source_push_gate_path: Path):
    parent, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    gate = read_finite_json(source_push_gate_path)
    require(isinstance(gate, dict), "repair source gate invalid")
    repair, repair_cells = _repair_from_gate(gate)
    require([cell.cell_id for cell in repair_cells] == [cell.cell_id for cell in cells], "repair and adapter queues differ")
    verify_pushed_gate(gate, repair)
    return parent, cells, gate


def require_smoke_authorization(*, registration_path: Path, queue_path: Path, authorization_path: Path):
    parent, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    gate = read_finite_json(authorization_path)
    require(isinstance(gate, dict) and gate.get("schema_version") == SMOKE_AUTH_SCHEMA and gate.get("status") == "passed_repair_excluded_smoke_authorization" and gate.get("passed") is True, "repair smoke is not authorized")
    repair, repair_cells = _repair_from_gate(gate)
    source = bound_json(gate.get("source_push_gate"), "repair smoke source", schema=SOURCE_GATE_SCHEMA, status="passed_repair_source_and_registration_pushed")
    verify_pushed_gate(source, repair)
    physical = bound_json(gate.get("physical_gate"), "repair physical gate", schema=PHYSICAL_GATE_SCHEMA, status="passed_repair_same_process_zero_request_preflight")
    require(all(physical.get(key) == 0 for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "query_server_entry_count")), "repair physical gate is not zero request")
    technical = repair["technical_gate_plan"]
    require(gate.get("lane_slot") == technical["global_excluded_smoke_lane_slot"] and physical.get("lane_slot") == gate.get("lane_slot"), "repair smoke lane/physical binding changed")
    require(gate.get("assignment_manifest", {}).get("sha256") == repair["assignment_manifest"]["sha256"], "repair smoke assignment changed")
    require(gate.get("excluded_smoke_seed") == technical["global_excluded_smoke_seed"] and gate.get("excluded_from_behavioral_denominators") is True, "repair smoke scope changed")
    block = [cell for cell in repair_cells if cell.seed == 12000]
    require(len(block) == 4 and gate.get("ordered_cell_ids") == [cell.cell_id for cell in sorted(block, key=lambda cell: cell.row["execution_order_index"])], "repair smoke block changed")
    return parent, block, gate


def require_released_gate(*, registration_path: Path, queue_path: Path, release_gate_path: Path):
    parent, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    gate = read_finite_json(release_gate_path)
    require(isinstance(gate, dict) and gate.get("schema_version") == RELEASE_SCHEMA and gate.get("status") == "passed_homogeneous_block_local_behavioral_release" and gate.get("passed") is True, "repair behavioral release did not pass")
    repair, repair_cells = _repair_from_gate(gate)
    require([cell.cell_id for cell in repair_cells] == [cell.cell_id for cell in cells], "repair release queue changed")
    source = bound_json(gate.get("source_push_gate"), "repair release source", schema=SOURCE_GATE_SCHEMA, status="passed_repair_source_and_registration_pushed")
    verify_pushed_gate(source, repair)
    physical_bindings = gate.get("physical_gates")
    require(isinstance(physical_bindings, list) and len(physical_bindings) == 8, "repair release lacks eight physical gates")
    physicals = [bound_json(record, f"repair physical gate {index}", schema=PHYSICAL_GATE_SCHEMA, status="passed_repair_same_process_zero_request_preflight") for index, record in enumerate(physical_bindings)]
    smoke = bound_json(gate.get("excluded_smoke_gate"), "repair excluded smoke", schema=SMOKE_GATE_SCHEMA, status="passed_repair_excluded_four_cell_smoke")
    repeat_bindings = gate.get("single_server_repeat_gates")
    require(isinstance(repeat_bindings, list) and len(repeat_bindings) == 8, "repair release lacks eight repeat gates")
    repeats = [bound_json(record, f"repair repeat gate {index}", schema=REPEAT_GATE_SCHEMA, status="passed_single_server_interleaved_exact_repeat") for index, record in enumerate(repeat_bindings)]
    require(all(all(physical.get(key) == 0 for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "query_server_entry_count")) for physical in physicals), "repair physical gate contains behavior")
    require(smoke.get("completed_cells") == 4 and smoke.get("behavioral_episode_count") == 0 and smoke.get("excluded_from_behavioral_denominators") is True and smoke.get("lane_slot") == "repair-lane-00", "repair smoke changed")
    smoke_fixture = validate_file_binding(smoke.get("repeat_fixture"), "repair smoke repeat fixture")
    require(all(repeat.get("model_request_count") == 3 and repeat.get("behavioral_episode_count") == 0 and repeat.get("first_final_repeat_exact") is True and repeat.get("prompt_sensitivity_distinct") is True for repeat in repeats), "repair repeat gate changed")
    require(all(repeat.get("fixture_sha256") == smoke_fixture["sha256"] and repeat.get("fixture", {}).get("sha256") == smoke_fixture["sha256"] for repeat in repeats), "repair lanes did not share the retained smoke fixture")
    repeat_manifest_shas = [repeat.get("fixture_manifest_sha256") for repeat in repeats]
    require(
        len(set(repeat_manifest_shas)) == 1
        and all(repeat.get("fixture_manifest", {}).get("sha256") == repeat_manifest_shas[0] for repeat in repeats),
        "repair lanes did not share one hash-bound smoke fixture manifest",
    )
    lanes = gate.get("lane_manifests")
    require(isinstance(lanes, list) and len(lanes) == 8, "repair release must have exactly eight lanes")
    lane_values = [bound_json(record, f"repair lane {index}", schema=LANE_SCHEMA, status="passed_single_lane_release") for index, record in enumerate(lanes)]
    server_keys = ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity")
    repeats_by_slot = {repeat.get("lane_slot"): repeat for repeat in repeats}
    physicals_by_slot = {physical.get("lane_slot"): physical for physical in physicals}
    require(len(repeats_by_slot) == 8, "repair repeat lane slots are not unique")
    require(len(physicals_by_slot) == 8, "repair physical lane slots are not unique")
    for lane in lane_values:
        repeat = repeats_by_slot.get(lane.get("lane_slot"))
        physical = physicals_by_slot.get(lane.get("lane_slot"))
        require(isinstance(repeat, dict) and all(lane.get(key) == repeat.get(key) for key in server_keys), "repair repeat and behavioral server identities differ")
        identity_keys = ("simulator_pod_uid", "simulator_gpu_uuid", *server_keys)
        require(isinstance(physical, dict) and all(lane.get(key) == physical.get(key) for key in identity_keys), "repair physical and behavioral lane identities differ")
        require(lane.get("no_failover_within_block") is True and lane.get("incomplete_block_retry_same_lane_only") is True, "repair lane retry policy changed")
        assigned = [row["episode_seed"] for row in validate_assignment(gate.get("assignment_manifest")) if row["lane_slot"] == lane.get("lane_slot")]
        require(lane.get("assigned_seed_blocks") == assigned and lane.get("assigned_seed_block_count") == len(assigned), "repair lane assigned blocks changed")
    require(len({lane.get("lane_slot") for lane in lane_values}) == 8, "repair lane slots are not unique")
    require(len({lane.get("simulator_pod_uid") for lane in lane_values}) == 8, "repair simulator pods are not isolated")
    require(len({lane.get("simulator_gpu_uuid") for lane in lane_values}) == 8, "repair simulator GPUs are not isolated")
    require(len({lane.get("policy_server_pod_uid") for lane in lane_values}) == 8, "repair policy pods are not isolated")
    require(len({lane.get("policy_server_gpu_uuid") for lane in lane_values}) == 8, "repair policy GPUs are not isolated")
    for key in ("lane_id", "server_port", "raw_root", "server_process_identity", "server_lock_identity"):
        require(len({lane.get(key) for lane in lane_values}) == 8 and all(lane.get(key) not in (None, "") for lane in lane_values), f"repair lanes are not isolated for {key}")
    homogeneous_keys = ("simulator_gpu_model", "simulator_driver", "policy_gpu_model", "policy_driver", "runtime_stack_sha256", "container_image_digest", "checkpoint_digest", "renderer_backend")
    for key in homogeneous_keys:
        values = [lane.get(key) for lane in lane_values]
        require(all(isinstance(value, str) and value for value in values), f"repair lane lacks typed {key}")
        require(len(set(values)) == 1, f"repair lanes are heterogeneous for {key}")
    expected = repair.get("homogeneity_contract", {}).get("expected", {})
    for key in ("simulator_gpu_model", "simulator_driver", "policy_gpu_model", "policy_driver", "checkpoint_digest", "renderer_backend"):
        require(isinstance(expected.get(key), str) and expected[key] and lane_values[0].get(key) == expected[key], f"repair lane differs from registered {key}")
    require(all(len(str(lane.get("runtime_stack_sha256"))) == 64 for lane in lane_values), "repair runtime stack digest is invalid")
    require(all(str(lane.get("container_image_digest")).startswith("sha256:") and len(str(lane.get("container_image_digest"))) == 71 for lane in lane_values), "repair container image digest is invalid")
    assignment = validate_file_binding(gate.get("assignment_manifest"), "repair assignment manifest")
    require(assignment["sha256"] == repair["assignment_manifest"]["sha256"], "repair assignment changed at release")
    validate_assignment(gate.get("assignment_manifest"))
    return parent, cells, gate
