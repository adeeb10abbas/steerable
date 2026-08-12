#!/usr/bin/env python3
"""Bind one homogeneous repair lane after physical and exact-repeat gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import read_finite_json, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import validate_runtime_manifest
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError, repo_binding, validate_assignment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--lane-slot", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--repeat-gate", type=Path, required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--homogeneity-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite lane manifest: {args.output}")
    runtime = validate_runtime_manifest(args.runtime_manifest, args.runtime_manifest_sha256, registration_path=args.parent_registration, queue_path=args.queue, pod_uid=args.lane_pod_uid, gpu_uuid=args.lane_gpu_uuid)["runtime_identity"]
    physical = read_finite_json(args.physical_gate); repeat = read_finite_json(args.repeat_gate)
    if not isinstance(physical, dict) or physical.get("schema_version") != "vla-wam-shared-v3c002r001-model-blind-physical-gate-v1" or physical.get("status") != "passed_repair_same_process_zero_request_preflight" or physical.get("passed") is not True or physical.get("lane_slot") != args.lane_slot:
        raise ContractError("lane physical gate changed")
    if not isinstance(repeat, dict) or repeat.get("schema_version") != "vla-wam-shared-v3c002r001-single-server-repeat-gate-v1" or repeat.get("status") != "passed_single_server_interleaved_exact_repeat" or repeat.get("passed") is not True or repeat.get("lane_slot") != args.lane_slot:
        raise ContractError("lane repeat gate changed")
    identity_keys = ("simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity")
    if any(physical.get(key) != runtime.get(key) for key in identity_keys) or any(repeat.get(key) != runtime.get(key) for key in identity_keys[2:]):
        raise ContractError("lane physical/repeat/runtime identity differs")
    assignment = repo_binding(args.repair_root / "assignment.jsonl")
    rows = validate_assignment(assignment)
    assigned = [row for row in rows if row["lane_slot"] == args.lane_slot]
    observed = read_finite_json(args.homogeneity_observation)
    if not isinstance(observed, dict) or observed.get("schema_version") != "vla-wam-shared-v3c002r001-lane-homogeneity-observation-v1" or observed.get("status") != "passed_observed_lane_homogeneity_identity" or observed.get("passed") is not True or observed.get("lane_slot") != args.lane_slot:
        raise ContractError("lane homogeneity observation is invalid")
    observed_identity = ("simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity")
    if any(observed.get(key) != runtime.get(key) for key in observed_identity):
        raise ContractError("lane homogeneity observation/runtime identity differs")
    homogeneous = ("simulator_gpu_model", "simulator_driver", "policy_gpu_model", "policy_driver", "runtime_stack_sha256", "container_image_digest")
    if not all(isinstance(observed.get(key), str) and observed[key] for key in homogeneous):
        raise ContractError("lane homogeneity observation lacks typed values")
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-single-lane-manifest-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_single_lane_release",
        "passed": True,
        "lane_slot": args.lane_slot,
        "repair_registration": repo_binding(args.repair_root / "registration.json"),
        "queue": repo_binding(args.queue),
        "assignment_manifest": assignment,
        "runtime_manifest": {"path": str(args.runtime_manifest.resolve()), "bytes": args.runtime_manifest.stat().st_size, "sha256": sha256_file(args.runtime_manifest)},
        "physical_gate": repo_binding(args.physical_gate),
        "repeat_gate": repo_binding(args.repeat_gate),
        "homogeneity_observation": {"path": str(args.homogeneity_observation.resolve()), "bytes": args.homogeneity_observation.stat().st_size, "sha256": sha256_file(args.homogeneity_observation)},
        "assigned_seed_block_count": len(assigned),
        "assigned_seed_blocks": [row["episode_seed"] for row in assigned],
        "no_failover_within_block": True,
        "incomplete_block_retry_same_lane_only": True,
        "completed_block_rerun_prohibited": True,
        **{key: observed[key] for key in homogeneous},
        "checkpoint_digest": runtime["checkpoint_digest"],
        "renderer_backend": runtime["renderer_backend"],
    }
    for key in ("lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "raw_root", "container_identity", "runtime_identity", "server_process_identity", "server_lock_identity"):
        value[key] = runtime[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "lane_slot": args.lane_slot, "blocks": len(assigned), "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
