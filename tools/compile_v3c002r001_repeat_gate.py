#!/usr/bin/env python3
"""Compile one repair server's retained interleaved exact-repeat response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, sha256_file, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError, repo_binding


def validate_repeat_evidence(response: object, physical: object, lane_slot: str) -> list[np.ndarray]:
    if not isinstance(response, dict) or response.get("schema_version") != "vla-wam-shared-v3c002r001-single-server-repeat-response-v1" or response.get("status") != "completed_excluded_single_server_interleaved_repeat" or response.get("passed") is not True or response.get("model_request_count") != 3 or response.get("behavioral_episode_count") != 0:
        raise ContractError("single-server repeat response failed")
    if not isinstance(physical, dict) or physical.get("schema_version") != "vla-wam-shared-v3c002r001-model-blind-physical-gate-v1" or physical.get("status") != "passed_repair_same_process_zero_request_preflight" or physical.get("passed") is not True or physical.get("lane_slot") != lane_slot:
        raise ContractError("repeat physical gate differs")
    if response.get("lane_slot") != lane_slot or response.get("successful_response_count") != 3:
        raise ContractError("single-server repeat lane/count changed")
    records = response.get("records")
    if not isinstance(records, list) or len(records) != 3 or response.get("sequence") != ["canonical_left", "canonical_right", "canonical_left"] or [record.get("ordinal") for record in records] != [0, 1, 2] or [record.get("condition") for record in records] != response["sequence"]:
        raise ContractError("single-server repeat sequence changed")
    actions = []
    for record in records:
        bound = validate_file_binding(record.get("actions"), "repeat action")
        array = np.load(Path(bound["path"]), allow_pickle=False)
        if array.shape != (15, 8) or not np.isfinite(array).all() or record.get("seed_echo") != response.get("probe_seed"):
            raise ContractError("repeat action array/seed is invalid")
        actions.append(array)
    exact = bool(np.array_equal(actions[0], actions[2])); sensitive = bool(not np.array_equal(actions[0], actions[1]))
    if not exact or not sensitive or response.get("first_final_repeat_exact") is not exact or response.get("prompt_sensitivity_distinct") is not sensitive:
        raise ContractError("single-server repeat/sensitivity recomputation failed")
    fixture = validate_file_binding(response.get("fixture"), "repeat fixture")
    if response.get("fixture_sha256") != fixture["sha256"]:
        raise ContractError("repeat fixture digest claim changed")
    server_keys = ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity")
    if any(response.get(key) != physical.get(key) for key in server_keys):
        raise ContractError("repeat response used a different policy server")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--lane-slot", required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--repeat-response", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite repeat gate: {args.output}")
    response = read_finite_json(args.repeat_response)
    physical = read_finite_json(args.physical_gate)
    validate_repeat_evidence(response, physical, args.lane_slot)
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-single-server-repeat-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_single_server_interleaved_exact_repeat",
        "passed": True,
        "lane_slot": args.lane_slot,
        "repair_registration": repo_binding(args.repair_root / "registration.json"),
        "assignment_manifest": repo_binding(args.repair_root / "assignment.jsonl"),
        "physical_gate": repo_binding(args.physical_gate),
        "repeat_response": {"path": str(args.repeat_response.resolve()), "bytes": args.repeat_response.stat().st_size, "sha256": sha256_file(args.repeat_response)},
        "model_request_count": 3,
        "behavioral_episode_count": 0,
        "excluded_from_behavioral_denominators": True,
        "first_final_repeat_exact": True,
        "prompt_sensitivity_distinct": True,
        "probe_seed": response["probe_seed"],
        "fixture_sha256": response["fixture_sha256"],
        "fixture": response["fixture"],
    }
    for key in ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
        value[key] = response[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
