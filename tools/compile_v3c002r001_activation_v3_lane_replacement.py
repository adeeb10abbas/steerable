#!/usr/bin/env python3
"""Fail-closed release of the two A100-killed activation-v3 lane replacements.

This is deliberately an additive, prospective gate.  It never edits the
activation-v3 registration, queue, original lane manifests, or released gate.
The gate authorizes exactly one four-cell retry in each of lanes 00 and 01
after independently hash-bound termination, zombie, partial-ledger, raw-rehash,
fresh-server preflight, and L-R-L exact-repeat evidence is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    file_binding,
    read_finite_json,
    require,
    sha256_file,
    validate_file_binding,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import (
    LANE_SCHEMA,
    PHYSICAL_GATE_SCHEMA,
    REPEAT_GATE_SCHEMA,
    validate_assignment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V3 = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-gate-v1"
REPLACEMENT_REGISTRATION_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-registration-v1"
REPLACEMENT_SOURCE_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-source-gate-v1"
TERMINATION_SCHEMA = "vla-wam-shared-v3c002r001-deleted-policy-pod-evidence-v1"
ZOMBIE_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-zombie-runner-evidence-v1"
PARTIAL_REHASH_SCHEMA = "vla-wam-shared-v3c002r001-lane-replacement-partial-rehash-v1"
COMPLETION_INDEX_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-completion-index-v1"
SLOTS = {"repair-lane-00": 12060, "repair-lane-01": 12101}
SERVER_KEYS = ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity")
SIM_KEYS = ("simulator_pod_uid", "simulator_gpu_uuid", "raw_root")
HOMOGENEOUS_KEYS = (
    "simulator_gpu_model", "simulator_driver", "policy_gpu_model", "policy_driver",
    "runtime_stack_sha256", "container_image_digest", "checkpoint_digest", "renderer_backend",
)


def _obj(path: Path, label: str) -> dict[str, Any]:
    value = read_finite_json(path)
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def _bound_obj(record: Any, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = validate_file_binding(record, label)
    return binding, _obj(Path(binding["path"]), label)


def _same_binding(a: Mapping[str, Any], b: Mapping[str, Any], label: str) -> None:
    require(a.get("sha256") == b.get("sha256") and a.get("bytes") == b.get("bytes"), f"{label} binding differs")


def _utc(value: Any, label: str) -> datetime:
    require(isinstance(value, str) and value.endswith("Z"), f"{label} must be UTC Z time")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ContractError(f"{label} is not an ISO timestamp") from exc
    require(parsed.tzinfo == timezone.utc, f"{label} is not UTC")
    return parsed


def _parse_slot_paths(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        slot, separator, raw_path = item.partition("=")
        require(separator == "=" and slot in SLOTS and raw_path, f"{label} must be SLOT=PATH for lanes 00 and 01")
        require(slot not in result, f"duplicate {label} for {slot}")
        result[slot] = Path(raw_path).resolve()
    require(set(result) == set(SLOTS), f"{label} must cover exactly killed lanes 00 and 01")
    return result


def _released_lanes(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    gate_path = root / "release_gate.released.json"
    gate = _obj(gate_path, "activation-v3 release")
    require(gate.get("schema_version") == "vla-wam-shared-v3c002r001-release-gate-v1" and gate.get("status") == "passed_homogeneous_block_local_behavioral_release" and gate.get("passed") is True, "activation-v3 release is not passed")
    lanes: dict[str, dict[str, Any]] = {}
    bindings: dict[str, dict[str, Any]] = {}
    records = gate.get("lane_manifests")
    require(isinstance(records, list) and len(records) == 8, "activation-v3 release lacks eight old lane manifests")
    for record in records:
        binding, value = _bound_obj(record, "released old lane manifest")
        require(value.get("schema_version") == LANE_SCHEMA and value.get("status") == "passed_single_lane_release" and value.get("passed") is True, "old lane manifest is not passed")
        slot = value.get("lane_slot")
        require(isinstance(slot, str) and slot not in lanes, "old lane slots are not unique")
        lanes[slot], bindings[slot] = value, binding
    require(set(lanes) == {f"repair-lane-{n:02d}" for n in range(8)}, "old lane set changed")
    return gate, lanes, bindings


def _require_pushed_source(path: Path, registration: Mapping[str, Any]) -> dict[str, Any]:
    gate = _obj(path, "replacement source gate")
    require(gate.get("schema_version") == REPLACEMENT_SOURCE_SCHEMA and gate.get("status") == "passed_activation_v3_lane_replacement_source_and_registration_pushed" and gate.get("passed") is True and gate.get("pushed") is True, "replacement source is not pushed")
    bound = validate_file_binding(gate.get("replacement_registration"), "replacement source registration")
    require(bound["sha256"] == registration.get("registration_sha256"), "replacement source/registration digest differs")
    implementation = gate.get("source_bindings")
    require(isinstance(implementation, dict) and implementation, "replacement source binding is missing")
    for name, record in implementation.items():
        require(isinstance(name, str) and name, "replacement source name invalid")
        validate_file_binding(record, f"replacement source {name}")
    return gate


def _require_termination(value: Mapping[str, Any], old_binding: Mapping[str, Any], old: Mapping[str, Any], slot: str) -> None:
    """Correlate the preserved combined Kubernetes capture to one old lane.

    The incident capture predates this additive gate and intentionally has no
    newly minted per-lane wrapper.  Binding its bytes plus the frozen old lane
    identity prevents a later narrative-only attribution of the kill.
    """
    require(value.get("schema_version") == TERMINATION_SCHEMA and all(value.get(key) == 0 for key in ("model_request_count", "behavioral_action_count", "behavioral_episode_count")), f"{slot} deleted-pod evidence is not zero-request")
    _utc(value.get("captured_at_utc"), f"{slot} deleted-pod capture time")
    records = value.get("records")
    require(isinstance(records, list) and records, f"{slot} deleted-pod evidence has no records")
    matches = [record for record in records if isinstance(record, dict) and record.get("policy_server_pod_uid") == old.get("policy_server_pod_uid")]
    require(len(matches) == 1, f"{slot} deleted-pod evidence does not uniquely bind old server UID")
    record = matches[0]
    require(record.get("current_pod_lookup") == "NotFound" and record.get("old_pod_uid_absent_from_current_cluster") is True, f"{slot} old policy pod is not confirmed deleted")
    events = record.get("events")
    require(isinstance(events, list) and events, f"{slot} deleted-pod evidence has no Kubernetes events")
    require(any(isinstance(event, dict) and event.get("reason") == "Killing" and isinstance(event.get("involvedObject"), dict) and event["involvedObject"].get("uid") == old.get("policy_server_pod_uid") for event in events), f"{slot} deleted-pod evidence lacks UID-bound Killing event")
    # The separate old-manifest binding is emitted in the final gate.
    require(old_binding.get("sha256"), f"{slot} old manifest binding is absent")


def _require_zombie(value: Mapping[str, Any], term_binding: Mapping[str, Any], old_binding: Mapping[str, Any], old: Mapping[str, Any], slot: str) -> None:
    require(value.get("schema_version") == ZOMBIE_SCHEMA and value.get("status") == "confirmed_no_live_zombie_runner" and value.get("passed") is True, f"{slot} zombie evidence did not pass")
    require(value.get("lane_slot") == slot and value.get("runner_process_not_live") is True and value.get("no_live_policy_server_process") is True and value.get("stale_lock_not_reused") is True, f"{slot} zombie evidence unsafe")
    _same_binding(validate_file_binding(value.get("termination_evidence"), f"{slot} zombie termination"), term_binding, f"{slot} zombie termination")
    _same_binding(validate_file_binding(value.get("old_lane_manifest"), f"{slot} zombie old lane"), old_binding, f"{slot} zombie old lane")
    for key in SERVER_KEYS:
        require(value.get(f"old_{key}") == old.get(key), f"{slot} zombie old {key} differs")
    _utc(value.get("observed_at_utc"), f"{slot} zombie observation time")
    for key in ("process_table", "pod_exec_probe", "capture_invocation", "environment"):
        validate_file_binding(value.get(key), f"{slot} zombie {key}")


def _expected_cell_ids(queue_cells: list[Any], seed: int) -> set[str]:
    result = {cell.cell_id for cell in queue_cells if cell.seed == seed}
    require(len(result) == 4, f"seed {seed} no longer has exactly four registered cells")
    return result


def _require_partial_ledger(path: Path, slot: str, seed: int, expected_cells: set[str]) -> dict[str, Any]:
    require(path.is_file() and path.stat().st_size > 0, f"{slot} partial ledger is missing")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        require(isinstance(value, dict), f"{slot} partial ledger row is not an object")
        require(value.get("schema_version") == "vla-wam-shared-v3c002-infrastructure-attempt-v1" and value.get("record_type") == "infrastructure_attempt" and value.get("infrastructure_status") == "infrastructure_invalid_excluded", f"{slot} partial ledger is not infrastructure-invalid")
        require(value.get("denominator_eligible") is False and value.get("authorization_mode") == "behavioral" and value.get("seed_block_id") == f"v3c002:seed{seed}" and value.get("cell_id") in expected_cells and value.get("entire_partial_block_invalidated") is True, f"{slot} partial ledger scope changed")
        completed = value.get("completed_cell_ids_before_failure")
        require(isinstance(completed, list) and len(completed) == len(set(completed)) and set(completed) <= expected_cells, f"{slot} partial completed cells changed")
        require(isinstance(value.get("attempt_root"), str) and Path(value["attempt_root"]).is_absolute(), f"{slot} partial attempt root invalid")
        rows.append(value)
    require(rows, f"{slot} partial ledger is empty")
    return file_binding(path)


def _require_partial_receipt(value: Mapping[str, Any], ledger_binding: Mapping[str, Any], slot: str, seed: int) -> None:
    """Validate the retained incident receipt, never recasting it as behavior."""
    require(value.get("schema_version") == PARTIAL_REHASH_SCHEMA and value.get("status") == "passed_retained_infrastructure_invalid_partial_block_rehash" and value.get("passed") is True, f"{slot} partial raw rehash did not pass")
    require(value.get("lane_slot") == slot and value.get("seed_block") == seed and value.get("partial_block_denominator_eligible") is False, f"{slot} partial receipt claims behavioral eligibility")
    _same_binding(validate_file_binding(value.get("infrastructure_ledger"), f"{slot} partial receipt ledger"), ledger_binding, f"{slot} partial receipt ledger")
    raw = value.get("attempt_files")
    require(isinstance(raw, list) and raw, f"{slot} partial receipt lacks retained attempt bindings")
    for record in raw:
        validate_file_binding(record, f"{slot} retained partial raw")


def _require_completion_index(value: Mapping[str, Any], old_binding: Mapping[str, Any], old: Mapping[str, Any], slot: str, seed: int, assigned: list[int]) -> list[int]:
    require(value.get("schema_version") == COMPLETION_INDEX_SCHEMA and value.get("status") == "captured_complete_blocks_before_replacement" and value.get("passed") is True and value.get("lane_slot") == slot, f"{slot} completion index is invalid")
    _same_binding(validate_file_binding(value.get("old_lane_manifest"), f"{slot} completion old lane"), old_binding, f"{slot} completion old lane")
    completed = value.get("completed_seed_blocks")
    unstarted = value.get("unstarted_seed_blocks")
    require(isinstance(completed, list) and len(completed) == len(set(completed)) and isinstance(unstarted, list) and len(unstarted) == len(set(unstarted)) and value.get("incomplete_seed_blocks") == [seed], f"{slot} completion partition is malformed")
    completed_set, unstarted_set = set(completed), set(unstarted)
    require(seed not in completed_set and seed not in unstarted_set and completed_set.isdisjoint(unstarted_set) and completed_set | unstarted_set | {seed} == set(assigned), f"{slot} completion partition does not cover the frozen assignment")
    markers = value.get("completed_block_markers")
    require(isinstance(markers, list) and len(markers) == len(completed_set), f"{slot} completion marker index incomplete")
    marker_seeds = set()
    for record in markers:
        require(isinstance(record, dict) and record.get("episode_seed") in completed_set, f"{slot} completion marker seed invalid")
        marker_seeds.add(record["episode_seed"])
        _, marker = _bound_obj(record.get("marker"), f"{slot} completion marker")
        require(marker.get("schema_version") == "vla-wam-shared-v3c002-completed-block-v1" and marker.get("status") == "completed_behavioral_block" and marker.get("authorization_mode") == "behavioral" and marker.get("episode_seed") == record["episode_seed"], f"{slot} completion marker changed")
        raws = marker.get("raw_episodes")
        require(isinstance(raws, list) and len(raws) == 4, f"{slot} completion marker does not bind four raw cells")
        for raw in raws:
            validate_file_binding(raw, f"{slot} completed block raw")
    require(marker_seeds == completed_set, f"{slot} completion marker set differs")
    require(old.get("assigned_seed_blocks") == assigned, f"{slot} old assignment changed")
    return sorted(completed_set)


def _require_physical(value: Mapping[str, Any], lane: Mapping[str, Any], slot: str) -> None:
    require(value.get("schema_version") == PHYSICAL_GATE_SCHEMA and value.get("status") == "passed_repair_same_process_zero_request_preflight" and value.get("passed") is True and value.get("lane_slot") == slot, f"{slot} replacement physical gate failed")
    require(all(value.get(key) == 0 for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "query_server_entry_count")) and value.get("same_process_gate_must_repeat_before_request_zero") is True, f"{slot} replacement physical gate is not zero request/same-process")
    for key in (*SIM_KEYS[:2], *SERVER_KEYS):
        require(value.get(key) == lane.get(key), f"{slot} replacement physical {key} differs")
    _, same = _bound_obj(value.get("same_process_report"), f"{slot} replacement same-process report")
    require(same.get("schema_version") == "vla-wam-shared-v3c002-same-process-model-blind-adapter-gate-v1" and same.get("status") == "passed_same_process_gate_stopped_before_query_server" and same.get("passed") is True and all(same.get(key) == 0 for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "query_server_entry_count")), f"{slot} replacement same-process report changed")
    _, receipt = _bound_obj(value.get("target_raw_rehash_receipt"), f"{slot} replacement physical rehash")
    require(receipt.get("schema_version") == "vla-wam-shared-v3c002-target-raw-rehash-receipt-v1" and receipt.get("status") == "passed_target_side_raw_rehash" and receipt.get("passed") is True and all(receipt.get(key) == 0 for key in ("model_requests", "behavioral_episodes", "behavioral_actions")), f"{slot} replacement physical raw rehash changed")


def _require_repeat(value: Mapping[str, Any], receipt: Mapping[str, Any], physical_binding: Mapping[str, Any], lane: Mapping[str, Any], fixture: tuple[str, str, str], slot: str) -> None:
    require(value.get("schema_version") == REPEAT_GATE_SCHEMA and value.get("status") == "passed_single_server_interleaved_exact_repeat" and value.get("passed") is True and value.get("lane_slot") == slot, f"{slot} replacement repeat failed")
    require(value.get("model_request_count") == 3 and value.get("behavioral_episode_count") == 0 and value.get("excluded_from_behavioral_denominators") is True and value.get("first_final_repeat_exact") is True and value.get("prompt_sensitivity_distinct") is True, f"{slot} replacement L-R-L receipt changed")
    require(tuple(value.get(key) for key in ("fixture_sha256", "fixture_manifest_sha256", "fixture_observation_payload_sha256")) == fixture, f"{slot} replacement repeat fixture changed")
    _same_binding(validate_file_binding(value.get("physical_gate"), f"{slot} repeat physical"), physical_binding, f"{slot} repeat physical")
    for key in SERVER_KEYS:
        require(value.get(key) == lane.get(key), f"{slot} replacement repeat {key} differs")
    require(receipt.get("schema_version") == "vla-wam-shared-v3c002r001-repeat-target-raw-rehash-v1" and receipt.get("status") == "passed_corrected_repeat_target_raw_rehash" and receipt.get("passed") is True and all(receipt.get(key) == expected for key, expected in (("model_request_count", 3), ("successful_response_count", 3), ("action_array_count", 3), ("behavioral_action_count", 0), ("behavioral_episode_count", 0))), f"{slot} replacement repeat raw receipt changed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path, default=V3)
    parser.add_argument("--replacement-registration", type=Path, required=True)
    parser.add_argument("--source-push-gate", type=Path, required=True)
    parser.add_argument("--termination-evidence", action="append", default=[], required=True)
    parser.add_argument("--zombie-evidence", action="append", default=[], required=True)
    parser.add_argument("--partial-ledger", action="append", default=[], required=True)
    parser.add_argument("--partial-rehash-receipt", action="append", default=[], required=True)
    parser.add_argument("--completion-index", action="append", default=[], required=True)
    parser.add_argument("--replacement-lane-manifest", action="append", default=[], required=True)
    parser.add_argument("--replacement-physical-gate", action="append", default=[], required=True)
    parser.add_argument("--replacement-repeat-gate", action="append", default=[], required=True)
    parser.add_argument("--replacement-repeat-receipt", action="append", default=[], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite replacement gate: {args.output}")
    root = args.activation_root.resolve()
    release, old_lanes, old_bindings = _released_lanes(root)
    registration = _obj(args.replacement_registration, "replacement registration")
    require(registration.get("schema_version") == REPLACEMENT_REGISTRATION_SCHEMA and registration.get("status") == "registered_prospective_activation_v3_lane_replacement" and registration.get("replacement_gate_model_requests_before_registration") == 0 and registration.get("replacement_gate_behavioral_episodes_before_registration") == 0, "replacement was not prospectively registered before new gate requests")
    registration = {**registration, "registration_sha256": sha256_file(args.replacement_registration)}
    source = _require_pushed_source(args.source_push_gate, registration)
    paths = {name: _parse_slot_paths(getattr(args, name), name.replace("_", " ")) for name in ("termination_evidence", "zombie_evidence", "partial_ledger", "partial_rehash_receipt", "completion_index", "replacement_lane_manifest", "replacement_physical_gate", "replacement_repeat_gate", "replacement_repeat_receipt")}
    activation_registration = validate_file_binding(release.get("repair_registration"), "activation-v3 registration")
    activation_queue = validate_file_binding(release.get("queue"), "activation-v3 queue")
    assignments = validate_assignment(release.get("assignment_manifest"))
    # The original release proves every unaffected lane used this exact retained fixture.
    old_repeat = _obj(Path(validate_file_binding(release["single_server_repeat_gates"][0], "retained repeat gate")["path"]), "retained repeat gate")
    fixture = tuple(old_repeat.get(key) for key in ("fixture_sha256", "fixture_manifest_sha256", "fixture_observation_payload_sha256"))
    require(all(isinstance(value, str) and value for value in fixture), "retained fixture identity missing")
    lane_bindings: list[dict[str, Any]] = []
    output: dict[str, Any] = {"schema_version": SCHEMA, "repair_id": "V3-C002-R001", "activation_id": registration.get("activation_id"), "status": "passed_activation_v3_cluster_termination_lane_replacement", "passed": True, "replacement_registration": file_binding(args.replacement_registration), "source_push_gate": file_binding(args.source_push_gate), "activation_v3_registration": activation_registration, "repair_registration": activation_registration, "queue": activation_queue, "assignment_manifest": validate_file_binding(release.get("assignment_manifest"), "activation-v3 assignment"), "prior_release_gate": file_binding(root / "release_gate.released.json"), "queue_science_unchanged": True, "same_simulator_lanes": True, "no_cross_lane_failover": True, "completed_blocks_never_rerun": True, "retry_seed_by_lane": dict(SLOTS), "replacements": {}, "unaffected_lane_manifests": {slot: old_bindings[slot] for slot in sorted(old_bindings) if slot not in SLOTS}}
    used_servers = {tuple(old_lanes[slot].get(key) for key in SERVER_KEYS) for slot in old_lanes if slot not in SLOTS}
    for slot, seed in SLOTS.items():
        term_binding, term = _bound_obj(file_binding(paths["termination_evidence"][slot]), f"{slot} termination")
        _require_termination(term, old_bindings[slot], old_lanes[slot], slot)
        zombie_binding, zombie = _bound_obj(file_binding(paths["zombie_evidence"][slot]), f"{slot} zombie")
        _require_zombie(zombie, term_binding, old_bindings[slot], old_lanes[slot], slot)
        # C002 cells are reconstructed from the frozen queue only for target-cell validation.
        queue_rows = [json.loads(line) for line in Path(activation_queue["path"]).read_text(encoding="utf-8").splitlines()]
        expected_cells = {str(row["cell_id"]) for row in queue_rows if int(row["episode_seed"]) == seed}
        require(len(expected_cells) == 4, f"{slot} frozen queue lost seed {seed}")
        ledger_binding = _require_partial_ledger(paths["partial_ledger"][slot], slot, seed, expected_cells)
        _, partial = _bound_obj(file_binding(paths["partial_rehash_receipt"][slot]), f"{slot} partial receipt")
        _require_partial_receipt(partial, ledger_binding, slot, seed)
        _, completion = _bound_obj(file_binding(paths["completion_index"][slot]), f"{slot} completion index")
        assigned = [int(row["episode_seed"]) for row in assignments if row["lane_slot"] == slot]
        completed = _require_completion_index(completion, old_bindings[slot], old_lanes[slot], slot, seed, assigned)
        lane_binding, lane = _bound_obj(file_binding(paths["replacement_lane_manifest"][slot]), f"{slot} replacement lane")
        require(lane.get("schema_version") == LANE_SCHEMA and lane.get("status") == "passed_single_lane_release" and lane.get("passed") is True and lane.get("lane_slot") == slot, f"{slot} replacement lane failed")
        require(lane.get("assigned_seed_blocks") == assigned and lane.get("assigned_seed_block_count") == len(assigned) and lane.get("no_failover_within_block") is True and lane.get("incomplete_block_retry_same_lane_only") is True and lane.get("completed_block_rerun_prohibited") is True, f"{slot} replacement lane assignment/retry policy changed")
        require(all(lane.get(key) == old_lanes[slot].get(key) for key in SIM_KEYS), f"{slot} simulator lane or raw root changed")
        require(all(lane.get(key) == old_lanes["repair-lane-02"].get(key) for key in HOMOGENEOUS_KEYS), f"{slot} replacement no longer homogeneous with unaffected lanes")
        server = tuple(lane.get(key) for key in SERVER_KEYS)
        # A new pod may legitimately land on the old physical A100.  Freshness
        # is an execution identity property (pod, port, process, lock), while
        # GPU uniqueness is required only against simultaneously live lanes.
        freshness_keys = ("policy_server_pod_uid", "server_port", "server_process_identity", "server_lock_identity")
        live_gpu_uuids = {old_lanes[other].get("policy_server_gpu_uuid") for other in old_lanes if other not in SLOTS} | {item[1] for item in used_servers}
        require(all(value not in (None, "") for value in server) and server not in used_servers and all(lane.get(key) != old_lanes[slot].get(key) for key in freshness_keys) and lane.get("policy_server_gpu_uuid") not in live_gpu_uuids, f"{slot} replacement server is not fresh/isolated")
        used_servers.add(server)
        physical_binding, physical = _bound_obj(file_binding(paths["replacement_physical_gate"][slot]), f"{slot} replacement physical")
        _require_physical(physical, lane, slot)
        repeat_binding, repeat = _bound_obj(file_binding(paths["replacement_repeat_gate"][slot]), f"{slot} replacement repeat")
        _, repeat_receipt = _bound_obj(file_binding(paths["replacement_repeat_receipt"][slot]), f"{slot} replacement repeat receipt")
        _require_repeat(repeat, repeat_receipt, physical_binding, lane, fixture, slot)
        _same_binding(validate_file_binding(lane.get("physical_gate"), f"{slot} lane physical"), physical_binding, f"{slot} lane physical")
        _same_binding(validate_file_binding(lane.get("repeat_gate"), f"{slot} lane repeat"), repeat_binding, f"{slot} lane repeat")
        output["replacements"][slot] = {"old_lane_manifest": old_bindings[slot], "termination_evidence": term_binding, "zombie_runner_evidence": zombie_binding, "partial_infrastructure_ledger": ledger_binding, "partial_target_raw_rehash": file_binding(paths["partial_rehash_receipt"][slot]), "completion_index": file_binding(paths["completion_index"][slot]), "replacement_lane_manifest": lane_binding, "replacement_physical_gate": physical_binding, "replacement_repeat_gate": repeat_binding, "replacement_repeat_target_receipt": file_binding(paths["replacement_repeat_receipt"][slot]), "authorized_retry_seed": seed, "authorized_retry_cell_count": 4, "replacement_attempt_name": "attempt002", "completed_seed_blocks_forbidden": completed, "completed_seed_blocks_never_rerun": True}
        lane_bindings.append(lane_binding)
    output["lane_manifests"] = lane_bindings + [old_bindings[slot] for slot in sorted(old_bindings) if slot not in SLOTS]
    output["source_gate_status"] = source["status"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "sha256": sha256_file(args.output), "retry_seed_by_lane": SLOTS}, sort_keys=True))


if __name__ == "__main__":
    main()
