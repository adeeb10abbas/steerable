#!/usr/bin/env python3
"""Register the prospective activation-v3 two-lane operational replacement.

The registration is a new amendment record, not a mutation of activation-v3.
It records the already-running cohort's structural progress separately from
the replacement gate's own zero-request starting counter.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, file_binding, read_finite_json, require, sha256_file, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import repo_binding


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/activation_v3"
SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-lane-replacement-registration-v1"
SNAPSHOT_SCHEMA = "vla-wam-shared-v3c002r001-activation-v3-pre-replacement-progress-snapshot-v1"
SLOTS = {"repair-lane-00": 12060, "repair-lane-01": 12101}
REQUIRED_SOURCE_PATHS = {
    "experiments/v3/phase_c_semantic_equivalence_v3c002/runner.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/contract.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/runner.py",
    "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v3_lane_replacement_runner.py",
    "tools/snapshot_v3c002r001_activation_v3_pre_replacement_progress.py",
    "tools/build_v3c002r001_activation_v3_lane_replacement_registration.py",
    "tools/compile_v3c002r001_activation_v3_lane_replacement_source_gate.py",
    "tools/compile_v3c002r001_activation_v3_lane_replacement.py",
    "tests/test_v3c002r001_activation_v3_lane_replacement.py",
}


def _source_bindings(raw_values: list[str]) -> dict[str, dict]:
    require(set(raw_values) == REQUIRED_SOURCE_PATHS and len(raw_values) == len(REQUIRED_SOURCE_PATHS), "replacement registration requires the exact frozen source set")
    sources: dict[str, dict] = {}
    for raw in raw_values:
        path = (ROOT / raw).resolve()
        require(path.is_file() and path.is_relative_to(ROOT), f"replacement source is not a repository file: {raw}")
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = repo_binding(path)
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", type=Path, default=V3)
    parser.add_argument("--progress-snapshot", type=Path, required=True)
    parser.add_argument("--deleted-policy-pod-evidence", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, help="Committed repository-relative implementation path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"refusing to overwrite replacement registration: {args.output}")
    activation = args.activation_root.resolve()
    release = read_finite_json(activation / "release_gate.released.json")
    require(isinstance(release, dict) and release.get("passed") is True and release.get("behavioral_episodes_authorized") is True, "activation-v3 is not the released parent")
    snapshot_binding = file_binding(args.progress_snapshot)
    snapshot = read_finite_json(args.progress_snapshot)
    require(isinstance(snapshot, dict) and snapshot.get("schema_version") == SNAPSHOT_SCHEMA and snapshot.get("status") == "captured_pre_replacement_activation_progress" and snapshot.get("passed") is True, "pre-replacement progress snapshot is invalid")
    require(isinstance(snapshot.get("completed_behavioral_block_count"), int) and snapshot["completed_behavioral_block_count"] >= 0, "progress snapshot lacks a completed-block count")
    require(isinstance(snapshot.get("captured_at_utc"), str) and snapshot["captured_at_utc"].endswith("Z") and all(snapshot.get(key) == 0 for key in ("replacement_gate_model_request_count", "replacement_gate_behavioral_action_count", "replacement_gate_behavioral_episode_count")), "progress snapshot replacement gate counters are not zero")
    lanes = snapshot.get("lanes")
    require(isinstance(lanes, dict) and set(lanes) == {f"repair-lane-{n:02d}" for n in range(8)}, "progress snapshot does not cover exactly eight lanes")
    counted = 0
    for slot, lane in lanes.items():
        require(isinstance(lane, dict), f"progress snapshot lane is invalid: {slot}")
        assigned = lane.get("assigned_seed_blocks"); completed = lane.get("completed_seed_blocks"); unstarted = lane.get("unstarted_seed_blocks"); partial = lane.get("partial_seed_blocks")
        require(all(isinstance(value, list) and len(value) == len(set(value)) for value in (assigned, completed, unstarted, partial)), f"progress snapshot partition is invalid: {slot}")
        require(set(completed).isdisjoint(unstarted) and set(completed).isdisjoint(partial) and set(unstarted).isdisjoint(partial) and set(completed) | set(unstarted) | set(partial) == set(assigned), f"progress snapshot partition does not cover assignment: {slot}")
        require(partial == ([SLOTS[slot]] if slot in SLOTS else []), f"progress snapshot partial scope changed: {slot}")
        markers = lane.get("completed_block_markers")
        require(isinstance(markers, list) and len(markers) == len(completed), f"progress snapshot markers are incomplete: {slot}")
        marker_seeds = set()
        for record in markers:
            require(isinstance(record, dict) and record.get("episode_seed") in set(completed), f"progress snapshot marker seed invalid: {slot}")
            marker_seeds.add(record["episode_seed"])
            marker = read_finite_json(Path(validate_file_binding(record.get("marker"), f"progress snapshot marker {slot}")["path"]))
            require(isinstance(marker, dict) and marker.get("schema_version") == "vla-wam-shared-v3c002-completed-block-v1" and marker.get("status") == "completed_behavioral_block" and marker.get("authorization_mode") == "behavioral" and marker.get("episode_seed") == record["episode_seed"], f"progress snapshot marker changed: {slot}")
            raws = marker.get("raw_episodes")
            require(isinstance(raws, list) and len(raws) == 4, f"progress snapshot marker raw count changed: {slot}")
            for raw in raws:
                validate_file_binding(raw, f"progress snapshot raw {slot}")
        require(marker_seeds == set(completed), f"progress snapshot marker coverage changed: {slot}")
        counted += len(completed)
    require(counted == snapshot["completed_behavioral_block_count"], "progress snapshot completed count is inconsistent")
    deleted_binding = file_binding(args.deleted_policy_pod_evidence)
    deleted = read_finite_json(args.deleted_policy_pod_evidence)
    require(isinstance(deleted, dict) and deleted.get("schema_version") == "vla-wam-shared-v3c002r001-deleted-policy-pod-evidence-v1", "deleted policy pod evidence schema changed")
    sources = _source_bindings(args.source)
    value = {
        "schema_version": SCHEMA,
        "repair_id": "V3-C002-R001",
        "activation_id": "V3-C002-R001-A003",
        "status": "registered_prospective_activation_v3_lane_replacement",
        "registered_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "parent_activation_v3_registration": repo_binding(activation / "registration.json"),
        "parent_release_gate": repo_binding(activation / "release_gate.released.json"),
        "queue": repo_binding(Path(validate_file_binding(release.get("queue"), "activation-v3 queue")["path"])),
        "assignment_manifest": repo_binding(Path(validate_file_binding(release.get("assignment_manifest"), "activation-v3 assignment")["path"])),
        "pre_replacement_activation_progress": snapshot_binding,
        "existing_activation_completed_behavioral_block_count": snapshot["completed_behavioral_block_count"],
        "replacement_gate_model_requests_before_registration": 0,
        "replacement_gate_behavioral_episodes_before_registration": 0,
        "retained_deleted_policy_pod_evidence": deleted_binding,
        "replacement_scope": {"lane_slots": list(SLOTS), "retry_seed_by_lane": SLOTS, "retry_only_incomplete_seed": True, "completed_blocks_never_rerun": True, "same_simulator_lanes": True, "fresh_policy_servers_only": True, "no_cross_lane_failover": True},
        "science_unchanged": {"queue": True, "assignment": True, "analysis": True, "predicates": True, "frozen_definitions": True},
        "source_bindings": sources,
        "release_status": "blocked_pending_committed_pushed_source_and_replacement_evidence",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output), "existing_completed_blocks": snapshot["completed_behavioral_block_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
