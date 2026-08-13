#!/usr/bin/env python3
"""Frozen eight-lane block-local runner for the prospective C002 repair."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from experiments.v3.phase_c_semantic_equivalence_v3c002 import runner as parent_runner
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import require
from .contract import require_released_gate, require_smoke_authorization


parent_runner.require_released_gate = require_released_gate
parent_runner.require_smoke_authorization = require_smoke_authorization


def _argument(name: str) -> str:
    index = sys.argv.index(name)
    return sys.argv[index + 1]


def _assigned(cells, *, shard_index: int, shard_count: int):
    require(shard_count == 8 and 0 <= shard_index < 8, "repair requires exactly eight frozen lane slots")
    gate = json.loads(Path(_argument("--authorization-gate")).read_text(encoding="utf-8"))
    assignment_record = gate.get("assignment_manifest")
    require(isinstance(assignment_record, dict), "repair release lacks assignment manifest")
    assignment_path = Path(str(assignment_record["path"]))
    if not assignment_path.is_absolute():
        assignment_path = Path(__file__).resolve().parents[3] / assignment_path
    rows = [json.loads(line) for line in assignment_path.read_text(encoding="utf-8").splitlines()]
    slot = f"repair-lane-{shard_index:02d}"
    seeds = {int(row["episode_seed"]) for row in rows if row["lane_slot"] == slot}
    selected = [cell for cell in cells if cell.seed in seeds]
    require(len(selected) == len(seeds) * 4, "repair assignment split or omitted a block")
    return selected


parent_runner.grouped_shard = _assigned


_parent_dispatch = parent_runner._dispatch_block


def _dispatch_with_provenance(*, block, args, registration_sha, queue_sha):
    _parent_dispatch(block=block, args=args, registration_sha=registration_sha, queue_sha=queue_sha)
    marker = parent_runner._completed_marker(Path(args.raw_root) / args.authorization_mode, block[0].seed)
    completed = json.loads(marker.read_text(encoding="utf-8"))
    gate_path = Path(args.authorization_gate).resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    repair_registration = gate["repair_registration"]
    repair_path = Path(str(repair_registration["path"]))
    if not repair_path.is_absolute():
        repair_path = Path(__file__).resolve().parents[3] / repair_path
    assignment = gate["assignment_manifest"]
    assignment_path = Path(str(assignment["path"]))
    if not assignment_path.is_absolute():
        assignment_path = Path(__file__).resolve().parents[3] / assignment_path
    assignment_rows = [json.loads(line) for line in assignment_path.read_text(encoding="utf-8").splitlines()]
    assigned = [row for row in assignment_rows if row["episode_seed"] == block[0].seed]
    if args.authorization_mode == "behavioral":
        require(len(assigned) == 1 and assigned[0]["lane_slot"] == f"repair-lane-{args.shard_index:02d}", "completed block differs from repair assignment")
        lane_bindings = gate.get("lane_manifests", [])
        lane_values = []
        for binding in lane_bindings:
            path = Path(str(binding["path"])); path = path if path.is_absolute() else Path(__file__).resolve().parents[3] / path
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("lane_id") == args.lane_id:
                lane_values.append((binding, value))
        require(len(lane_values) == 1 and lane_values[0][1].get("lane_slot") == assigned[0]["lane_slot"], "repair runner lane ID/slot changed")
        lane_manifest_binding = lane_values[0][0]
        lane_slot = assigned[0]["lane_slot"]
    else:
        require(gate.get("lane_slot") == "repair-lane-00" and args.shard_index == 0, "repair smoke must use lane slot zero")
        lane_manifest_binding = None
        lane_slot = "repair-lane-00"
    for record in completed["raw_episodes"]:
        raw_path = Path(record["path"])
        sidecar = raw_path.with_name("r001_provenance.json")
        value = {
            "schema_version": "vla-wam-shared-v3c002r001-raw-provenance-v1",
            "repair_id": "V3-C002-R001",
            "cell_id": record["cell_id"],
            "episode_seed": block[0].seed,
            "lane_slot": lane_slot,
            "lane_id": args.lane_id,
            "parent_raw_episode": {"path": str(raw_path.resolve()), "bytes": raw_path.stat().st_size, "sha256": parent_runner.sha256_file(raw_path)},
            "repair_registration": repair_registration,
            "assignment_manifest": assignment,
            "authorization_gate": {"path": str(gate_path), "bytes": gate_path.stat().st_size, "sha256": parent_runner.sha256_file(gate_path)},
            "released_lane_manifest": lane_manifest_binding,
            "wrapper_source": {"path": str(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size, "sha256": parent_runner.sha256_file(Path(__file__))},
            "block_indivisible": True,
        }
        encoded = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
        if sidecar.exists():
            require(sidecar.read_text(encoding="utf-8") == encoded, "repair provenance sidecar changed")
        else:
            sidecar.write_text(encoded, encoding="utf-8")


parent_runner._dispatch_block = _dispatch_with_provenance


def main() -> None:
    parent_runner.main()


if __name__ == "__main__":
    main()
