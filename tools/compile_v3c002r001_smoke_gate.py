#!/usr/bin/env python3
"""Compile the global excluded R001 four-cell smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import load_cells, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002.runner import _raw_row
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError, repo_binding, require_smoke_authorization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--smoke-authorization", type=Path, required=True)
    parser.add_argument("--completed-block", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite repair smoke gate: {args.output}")
    _, cells = load_cells(registration_path=args.parent_registration, queue_path=args.queue)
    _, authorized_block, authorization = require_smoke_authorization(registration_path=args.parent_registration, queue_path=args.queue, authorization_path=args.smoke_authorization)
    block = {cell.cell_id: cell for cell in cells if cell.seed == 12000}
    marker = json.loads(args.completed_block.read_text(encoding="utf-8"))
    authorization = json.loads(args.smoke_authorization.read_text(encoding="utf-8"))
    if authorization.get("schema_version") != "vla-wam-shared-v3c002r001-smoke-authorization-v1" or authorization.get("status") != "passed_repair_excluded_smoke_authorization" or authorization.get("passed") is not True or authorization.get("lane_slot") != "repair-lane-00":
        raise ContractError("repair smoke authorization changed")
    records = marker.get("raw_episodes")
    if marker.get("schema_version") != "vla-wam-shared-v3c002-completed-block-v1" or marker.get("status") != "completed_excluded_smoke_block" or marker.get("authorization_mode") != "excluded_smoke" or marker.get("episode_seed") != 12000 or not isinstance(records, list) or len(records) != 4:
        raise ContractError("repair smoke block incomplete")
    if [record.get("cell_id") for record in records] != authorization.get("ordered_cell_ids"):
        raise ContractError("repair smoke execution order changed")
    rows = []
    for record in records:
        cell = block.get(record.get("cell_id")); path = Path(str(record.get("path", "")))
        if cell is None or not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ContractError("repair smoke raw binding changed")
        rows.append(_raw_row(path, cell=cell, mode="excluded_smoke"))
        runtime = rows[-1].get("runtime_identity", {})
        physical = json.loads(Path(authorization["physical_gate"]["path"] if Path(str(authorization["physical_gate"]["path"])).is_absolute() else Path(__file__).resolve().parents[1] / authorization["physical_gate"]["path"]).read_text(encoding="utf-8"))
        for key in ("simulator_pod_uid", "simulator_gpu_uuid", "policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
            if runtime.get(key) != physical.get(key):
                raise ContractError(f"repair smoke runtime differs from physical gate for {key}")
    if len({row["initial_state_sha256"] for row in rows}) != 1 or len({row["request0_pair_identity_sha256"] for row in rows}) != 1:
        raise ContractError("repair smoke is not state/request0 matched")
    repeat_fixture = Path(str(marker.get("attempt_root", ""))) / "request0" / "observation_cache.npz"
    if not repeat_fixture.is_file():
        raise ContractError("repair smoke retained request-zero fixture is missing")
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-excluded-smoke-gate-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_repair_excluded_four_cell_smoke",
        "passed": True,
        "repair_registration": repo_binding(args.repair_root / "registration.json"),
        "queue": repo_binding(args.queue),
        "assignment_manifest": repo_binding(args.repair_root / "assignment.jsonl"),
        "smoke_authorization": repo_binding(args.smoke_authorization),
        "completed_block": {"path": str(args.completed_block.resolve()), "bytes": args.completed_block.stat().st_size, "sha256": sha256_file(args.completed_block)},
        "completed_cells": 4,
        "lane_slot": "repair-lane-00",
        "behavioral_episode_count": 0,
        "model_request_count": sum(int(row["model_request_count"]) for row in rows),
        "excluded_from_behavioral_denominators": True,
        "initial_state_sha256": rows[0]["initial_state_sha256"],
        "request0_pair_identity_sha256": rows[0]["request0_pair_identity_sha256"],
        "repeat_fixture": {"path": str(repeat_fixture.resolve()), "bytes": repeat_fixture.stat().st_size, "sha256": sha256_file(repeat_fixture)},
        "raw_episode_bindings": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
