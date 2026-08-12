#!/usr/bin/env python3
"""Authorize one global excluded four-cell smoke on one repair lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import load_cells, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError, repo_binding, require_smoke_authorization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--lane-slot", default="repair-lane-00")
    parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite repair smoke authorization: {args.output}")
    _, cells = load_cells(registration_path=args.parent_registration, queue_path=args.queue)
    block = sorted([cell for cell in cells if cell.seed == 12000], key=lambda cell: cell.row["execution_order_index"])
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-smoke-authorization-v1",
        "repair_id": "V3-C002-R001",
        "status": "passed_repair_excluded_smoke_authorization",
        "passed": True,
        "repair_registration": repo_binding(args.repair_root / "registration.json"),
        "queue": repo_binding(args.queue),
        "assignment_manifest": repo_binding(args.repair_root / "assignment.jsonl"),
        "source_push_gate": repo_binding(args.repair_root / "source_push_gate.released.json"),
        "physical_gate": repo_binding(args.physical_gate),
        "lane_slot": args.lane_slot,
        "excluded_smoke_seed": 12000,
        "ordered_cell_ids": [cell.cell_id for cell in block],
        "excluded_from_behavioral_denominators": True,
        "model_requests_before_smoke": 0,
        "behavioral_episodes_before_smoke": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidate = args.output.with_name(args.output.name + ".candidate")
    candidate.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_smoke_authorization(registration_path=args.parent_registration, queue_path=args.queue, authorization_path=candidate)
    candidate.replace(args.output)
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
