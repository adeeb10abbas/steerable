#!/usr/bin/env python3
"""Compile the unchanged C002 analysis with R001 assignment provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import compiler as parent
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import load_cells, repo_file_binding, require, sha256_file, validate_file_binding
from .contract import load_repair, require_released_gate, validate_assignment


def read_jsonl(path: Path) -> list[dict]:
    return parent._read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-registration", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--raw-episodes", type=Path, required=True)
    parser.add_argument("--episodes-output", type=Path, required=True)
    parser.add_argument("--pairs-output", type=Path, required=True)
    parser.add_argument("--results-output", type=Path, required=True)
    parser.add_argument("--decision-memo-output", type=Path, required=True)
    parser.add_argument("--evidence-manifest-output", type=Path, required=True)
    parser.add_argument("--manuscript-insert-output", type=Path, required=True)
    parser.add_argument("--infrastructure-attempts", type=Path, required=True)
    args = parser.parse_args()
    repair, cells = load_repair(registration_path=args.repair_registration, queue_path=args.queue)
    registration, parent_cells = load_cells(registration_path=args.parent_registration, queue_path=args.queue)
    require_released_gate(registration_path=args.parent_registration, queue_path=args.queue, release_gate_path=args.release_gate)
    require([cell.cell_id for cell in cells] == [cell.cell_id for cell in parent_cells], "repair compiler queue changed")
    assignment = {row["episode_seed"]: row["lane_slot"] for row in validate_assignment(repair["assignment_manifest"])}
    release = json.loads(args.release_gate.read_text(encoding="utf-8"))
    lane_values = []
    lane_binding_by_slot = {}
    for binding in release["lane_manifests"]:
        path = Path(binding["path"])
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[3] / path
        value = json.loads(path.read_text(encoding="utf-8"))
        lane_values.append(value)
        lane_binding_by_slot[value["lane_slot"]] = binding
    lane_by_slot = {lane["lane_slot"]: lane for lane in lane_values}
    raw_rows = read_jsonl(args.raw_episodes)
    require(len(raw_rows) == 1364 and len({row.get("cell_id") for row in raw_rows}) == 1364, "repair raw source is not 1,364 unique cells")
    cells_by_id = {cell.cell_id: cell for cell in cells}
    provenance_bindings = []
    for row in raw_rows:
        cell = cells_by_id.get(row.get("cell_id")); require(cell is not None, "repair raw cell is unregistered")
        request_events = validate_file_binding(row.get("raw_artifacts", {}).get("raw_episode_jsonl"), "repair request event stream")
        sidecar = Path(request_events["path"]).with_name("r001_provenance.json")
        require(sidecar.is_file(), f"repair raw provenance missing: {cell.cell_id}")
        provenance = json.loads(sidecar.read_text(encoding="utf-8"))
        slot = assignment[cell.seed]; lane = lane_by_slot[slot]
        require(provenance.get("schema_version") == "vla-wam-shared-v3c002r001-raw-provenance-v1" and provenance.get("cell_id") == cell.cell_id and provenance.get("lane_slot") == slot, "repair raw provenance identity changed")
        parent_raw = validate_file_binding(provenance.get("parent_raw_episode"), "repair parent raw episode")
        raw_parent_path = Path(parent_raw["path"])
        source_lines = raw_parent_path.read_text(encoding="utf-8").splitlines()
        parent_raw_row = json.loads(source_lines[0]) if len(source_lines) == 1 else None
        require(parent_raw_row == row and parent_raw_row.get("cell_id") == cell.cell_id, "repair parent raw episode/cell changed")
        require(provenance.get("repair_registration", {}).get("sha256") == sha256_file(args.repair_registration), "repair provenance registration changed")
        require(provenance.get("assignment_manifest", {}).get("sha256") == repair["assignment_manifest"]["sha256"], "repair provenance assignment changed")
        require(provenance.get("authorization_gate", {}).get("sha256") == sha256_file(args.release_gate), "repair provenance release changed")
        require(provenance.get("released_lane_manifest", {}).get("sha256") == lane_binding_by_slot[slot]["sha256"], "repair provenance lane manifest changed")
        require(provenance.get("lane_id") == lane.get("lane_id"), "repair provenance lane ID changed")
        runtime = row.get("runtime_identity", {})
        for key in ("lane_id", "simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
            require(runtime.get(key) == lane.get(key), f"repair episode differs from assigned lane for {key}")
        provenance_bindings.append({"path": str(sidecar.resolve()), "bytes": sidecar.stat().st_size, "sha256": sha256_file(sidecar)})
    registration_sha = sha256_file(args.parent_registration); queue_sha = sha256_file(args.queue)
    episodes = [parent.compile_episode(row, cell=cells_by_id[row["cell_id"]], registration_sha256=registration_sha, queue_sha256=queue_sha, exact_runtime_contract=registration["exact_e004_pi05_runtime"]) for row in raw_rows]
    pairs, results = parent.compile_results(episodes, registration_sha256=registration_sha, queue_sha256=queue_sha)
    for episode in episodes:
        episode["repair_id"] = "V3-C002-R001"
        episode["repair_lane_slot"] = assignment[episode["episode_seed"]]
    lane_diagnostics = {}
    for slot in sorted(lane_by_slot):
        slot_pairs = [pair for pair in pairs if assignment[pair["episode_seed"]] == slot]
        lane_diagnostics[slot] = {
            "seed_blocks": len({pair["episode_seed"] for pair in slot_pairs}),
            "left_depth_inverse_minus_canonical_mean_m": fmean(pair["depth_inverse_minus_canonical_m"] for pair in slot_pairs if pair["physical_goal"] == "left"),
            "right_depth_inverse_minus_canonical_mean_m": fmean(pair["depth_inverse_minus_canonical_m"] for pair in slot_pairs if pair["physical_goal"] == "right"),
        }
    leave_one_out = {}
    for omitted in sorted(lane_by_slot):
        subset = [pair for pair in pairs if assignment[pair["episode_seed"]] != omitted]
        leave_one_out[omitted] = {
            goal: fmean(pair["depth_inverse_minus_canonical_m"] for pair in subset if pair["physical_goal"] == goal)
            for goal in ("left", "right")
        }
    results["repair_id"] = "V3-C002-R001"
    results["original_cross_lane_gate_remains_failed"] = True
    results["cross_lane_numerical_tolerance_used"] = False
    results["lane_diagnostics_descriptive_only"] = lane_diagnostics
    results["leave_one_lane_out_diagnostics_descriptive_only"] = leave_one_out
    infrastructure_rows = read_jsonl(args.infrastructure_attempts) if args.infrastructure_attempts.stat().st_size else []
    require(all(row.get("infrastructure_status") == "infrastructure_invalid_excluded" for row in infrastructure_rows), "repair infrastructure ledger includes denominator evidence")
    parent._write_jsonl(args.episodes_output, episodes)
    parent._write_jsonl(args.pairs_output, pairs)
    parent._write_json(args.results_output, results)
    disclosure = "\n\nR001 disclosure: this is a prospective post-gate operational repair. Original C002 remains closed after failed cross-server isolation; R001 uses exact within-server repeatability and block-local homogeneous lanes without a numerical tolerance.\n"
    parent._write_text(args.decision_memo_output, parent.decision_memo(results) + disclosure)
    parent._write_text(args.manuscript_insert_output, parent.manuscript_insert(results) + disclosure)
    raw_bindings = [record for episode in episodes for record in list(episode["raw_artifacts"].values()) + list(episode["policy_camera_image_artifacts"].values())]
    manifest = {
        "schema_version": "vla-wam-shared-v3c002r001-evidence-manifest-v1",
        "repair_id": "V3-C002-R001",
        "status": "complete_hash_bound_repair_results",
        "repair_registration": repo_file_binding(args.repair_registration),
        "parent_registration": repo_file_binding(args.parent_registration),
        "queue": repo_file_binding(args.queue),
        "assignment_manifest": repair["assignment_manifest"],
        "release_gate": repo_file_binding(args.release_gate),
        "raw_episodes": {"path": str(args.raw_episodes.resolve()), "bytes": args.raw_episodes.stat().st_size, "sha256": sha256_file(args.raw_episodes)},
        "infrastructure_attempts": {"path": str(args.infrastructure_attempts.resolve()), "bytes": args.infrastructure_attempts.stat().st_size, "sha256": sha256_file(args.infrastructure_attempts)},
        "repair_provenance_sidecars": provenance_bindings,
        "compiled_outputs": {name: repo_file_binding(path) for name, path in (("episodes", args.episodes_output), ("pairs", args.pairs_output), ("results", args.results_output), ("decision_memo", args.decision_memo_output), ("manuscript_insert", args.manuscript_insert_output))},
        "valid_behavioral_episode_count": 1364,
        "complete_seed_block_count": 341,
        "raw_source_artifact_count_rehashed": len(raw_bindings),
        "raw_source_bytes_rehashed": sum(record["bytes"] for record in raw_bindings),
        "infrastructure_attempt_count_excluded": len(infrastructure_rows),
        "original_cross_lane_gate_remains_failed": True,
        "cross_lane_numerical_tolerance_used": False,
        "compiler": repo_file_binding(Path(__file__)),
        "invocation": [sys.executable, *sys.argv],
    }
    parent._write_json(args.evidence_manifest_output, manifest)


if __name__ == "__main__":
    main()
