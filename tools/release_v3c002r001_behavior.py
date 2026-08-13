#!/usr/bin/env python3
"""Compile the prospective eight-lane R001 behavioral release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError, repo_binding, require_released_gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--source-push-gate", type=Path, required=True)
    parser.add_argument("--physical-gate", type=Path, action="append", required=True)
    parser.add_argument("--excluded-smoke-gate", type=Path, required=True)
    parser.add_argument("--repeat-gate", type=Path, action="append", required=True)
    parser.add_argument("--repeat-target-receipt", type=Path, action="append", required=True)
    parser.add_argument("--failed-repeat-attempt001-receipt", type=Path, required=True)
    parser.add_argument("--lane-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.physical_gate) != 8 or len(args.repeat_gate) != 8 or len(args.repeat_target_receipt) != 8 or len(args.lane_manifest) != 8:
        raise ContractError("repair release requires one new output and exactly eight lane gate sets")
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-release-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_homogeneous_block_local_behavioral_release",
        "passed": True,
        "repair_registration": repo_binding(args.repair_root / "registration.json"),
        "queue": repo_binding(args.queue),
        "assignment_manifest": repo_binding(args.repair_root / "assignment.jsonl"),
        "source_push_gate": repo_binding(args.source_push_gate),
        "physical_gates": [repo_binding(path) for path in args.physical_gate],
        "excluded_smoke_gate": repo_binding(args.excluded_smoke_gate),
        "single_server_repeat_gates": [repo_binding(path) for path in args.repeat_gate],
        "single_server_repeat_target_receipts": [repo_binding(path) for path in args.repeat_target_receipt],
        "failed_repeat_attempt001_target_receipt": repo_binding(args.failed_repeat_attempt001_receipt),
        "lane_manifests": [repo_binding(path) for path in args.lane_manifest],
        "behavioral_episode_count_before_release": 0,
        "original_c002_excluded_request_count": 30,
        "repair_excluded_request_count_before_release": None,
        "cross_lane_action_equality_required": False,
        "cross_server_numerical_tolerance": None,
        "all_confirmatory_estimands_block_local": True,
    }
    smoke = json.loads(args.excluded_smoke_gate.read_text(encoding="utf-8"))
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in args.repeat_target_receipt]
    failed = json.loads(args.failed_repeat_attempt001_receipt.read_text(encoding="utf-8"))
    if failed.get("schema_version") != "vla-wam-shared-v3c002r001-failed-repeat-target-rehash-v1" or failed.get("status") != "retained_eight_infrastructure_invalid_flat_cache_requests" or failed.get("passed") is not True or failed.get("model_request_count") != 8 or failed.get("successful_response_count") != 0 or failed.get("action_array_count") != 0 or failed.get("behavioral_episode_count") != 0:
        raise ContractError("retained invalid repeat receipt counts changed")
    if any(
        receipt.get("schema_version") != "vla-wam-shared-v3c002r001-repeat-target-raw-rehash-v1"
        or receipt.get("status") != "passed_corrected_repeat_target_raw_rehash"
        or receipt.get("passed") is not True
        or receipt.get("model_request_count") != 3
        or receipt.get("successful_response_count") != 3
        or receipt.get("action_array_count") != 3
        or receipt.get("behavioral_action_count") != 0
        or receipt.get("behavioral_episode_count") != 0
        for receipt in receipts
    ):
        raise ContractError("repair repeat target receipts did not pass")
    value["repair_excluded_request_count_before_release"] = int(smoke["model_request_count"]) + 8 + sum(int(receipt["model_request_count"]) for receipt in receipts)
    value["repair_excluded_request_breakdown"] = {"global_smoke": 70, "retained_invalid_attempt001": 8, "corrected_repeat_attempt002": 24}
    value["behavioral_episodes_authorized"] = True
    value["full_queue_launched"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_released_gate(registration_path=args.parent_registration, queue_path=args.queue, release_gate_path=args.output)
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
