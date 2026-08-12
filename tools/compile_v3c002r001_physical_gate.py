#!/usr/bin/env python3
"""Bind one repair lane's same-process zero-request physical gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import validate_runtime_manifest
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import (
    ContractError,
    load_repair,
    repo_binding,
    verify_pushed_gate,
)


def raw_binding(path: Path) -> dict:
    path = path.resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise ContractError(f"missing physical evidence: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--lane-slot", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--standalone-report", type=Path, required=True)
    parser.add_argument("--same-process-report", type=Path, required=True)
    parser.add_argument("--target-raw-rehash-receipt", type=Path, required=True)
    parser.add_argument("--invocation", type=Path, required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite repair physical gate: {args.output}")
    if args.lane_slot not in {f"repair-lane-{index:02d}" for index in range(8)}:
        raise ContractError("repair physical lane slot is invalid")
    repair, _ = load_repair(registration_path=args.repair_root / "registration.json", queue_path=args.queue)
    source_gate = read_finite_json(args.repair_root / "source_push_gate.released.json")
    if not isinstance(source_gate, dict):
        raise ContractError("repair pushed source gate is invalid")
    verify_pushed_gate(source_gate, repair)
    runtime = validate_runtime_manifest(args.runtime_manifest, args.runtime_manifest_sha256, registration_path=args.parent_registration, queue_path=args.queue, pod_uid=args.lane_pod_uid, gpu_uuid=args.lane_gpu_uuid)["runtime_identity"]
    parent_registration = read_finite_json(args.parent_registration)
    standalone = read_finite_json(args.standalone_report)
    same = read_finite_json(args.same_process_report)
    receipt = read_finite_json(args.target_raw_rehash_receipt)
    expected_study_commit = parent_registration["source_lineage"]["replacement_commit"]
    if not isinstance(standalone, dict) or standalone.get("schema_version") != "vla-wam-shared-v3e004-standalone-model-blind-droid-gate-v2" or standalone.get("status") != "passed_model_blind_preflight_not_a_behavioral_release" or standalone.get("passed") is not True or any(standalone.get(key) != 0 for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count")) or standalone.get("model_id") != "pi05_current_stack_droid" or standalone.get("study_commit") != expected_study_commit or standalone.get("candidate_sha256") != parent_registration["e004_s1_layout"]["candidate_sha256"]:
        raise ContractError("repair standalone gate is not zero request")
    for key in ("live_scene_gate", "viewport_video"):
        record = standalone.get(key)
        path = Path(str(record.get("path", ""))) if isinstance(record, dict) else Path("")
        if not path.is_file() or record.get("bytes") != path.stat().st_size or record.get("sha256") != sha256_file(path):
            raise ContractError(f"repair standalone {key} binding changed")
    if not isinstance(standalone.get("request0_replay"), dict) or len(str(standalone["request0_replay"].get("pair_identity_sha256", ""))) != 64:
        raise ContractError("repair standalone request-zero evidence is missing")
    if not isinstance(same, dict) or same.get("schema_version") != "vla-wam-shared-v3c002-same-process-model-blind-adapter-gate-v1" or same.get("status") != "passed_same_process_gate_stopped_before_query_server" or same.get("passed") is not True or same.get("same_process_gate_completed_before_query_server") is not True or any(same.get(key) != 0 for key in ("model_request_count", "behavioral_episode_count", "behavioral_action_count", "query_server_entry_count")) or same.get("source_commit") != expected_study_commit or same.get("registration_sha256") != sha256_file(args.parent_registration) or same.get("queue_sha256") != sha256_file(args.queue):
        raise ContractError("repair same-process gate is not zero request")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != "vla-wam-shared-v3c002-target-raw-rehash-receipt-v1" or receipt.get("status") != "passed_target_side_raw_rehash" or receipt.get("passed") is not True or receipt.get("same_process_adapter_report_sha256") != sha256_file(args.same_process_report) or receipt.get("standalone_report_sha256") != sha256_file(args.standalone_report) or any(receipt.get(key) != 0 for key in ("model_requests", "behavioral_episodes", "behavioral_actions")):
        raise ContractError("repair target raw rehash did not pass")
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-model-blind-physical-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_repair_same_process_zero_request_preflight",
        "passed": True,
        "lane_slot": args.lane_slot,
        "repair_registration": repo_binding(args.repair_root / "registration.json"),
        "queue": repo_binding(args.queue),
        "source_push_gate": repo_binding(args.repair_root / "source_push_gate.released.json"),
        "runtime_manifest": raw_binding(args.runtime_manifest),
        "standalone_report": raw_binding(args.standalone_report),
        "same_process_report": raw_binding(args.same_process_report),
        "target_raw_rehash_receipt": raw_binding(args.target_raw_rehash_receipt),
        "invocation": raw_binding(args.invocation),
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "behavioral_action_count": 0,
        "query_server_entry_count": 0,
        "physical_scene": True,
        "full_reset": True,
        "policy_cameras": True,
        "raw_writer": True,
        "renderer": True,
        "same_process_gate_must_repeat_before_request_zero": True,
    }
    for key in ("simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity", "checkpoint_digest", "renderer_backend"):
        value[key] = runtime[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
