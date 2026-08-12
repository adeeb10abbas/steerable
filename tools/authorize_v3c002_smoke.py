#!/usr/bin/env python3
"""Authorize exactly one excluded V3-C002 four-cell smoke block."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, SEED_START, load_cells, repo_file_binding, require_smoke_authorization, sha256_file

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--source-push-gate", type=Path, required=True); parser.add_argument("--physical-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    if args.output.exists(): raise ContractError(f"refusing to overwrite smoke authorization: {args.output}")
    _, cells = load_cells(registration_path=args.registration, queue_path=args.queue)
    block = sorted([cell for cell in cells if cell.seed == SEED_START], key=lambda cell: int(cell.row["execution_order_index"]))
    value = {
        "schema_version": "vla-wam-shared-v3c002-smoke-authorization-v2", "status": "passed_pre_request_excluded_smoke_authorization", "passed": True,
        "registration": repo_file_binding(args.registration), "queue": repo_file_binding(args.queue),
        "source_push_gate": repo_file_binding(args.source_push_gate), "physical_gate": repo_file_binding(args.physical_gate),
        "excluded_smoke_seed": SEED_START, "ordered_cell_ids": [cell.cell_id for cell in block],
        "excluded_from_behavioral_denominators": True, "model_requests_before_smoke": 0, "behavioral_episodes_before_smoke": 0,
        "authorization_scope": "exactly these four cells, serially on one isolated lane; never part of the 1,364 behavioral denominator",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    require_smoke_authorization(registration_path=args.registration, queue_path=args.queue, authorization_path=args.output)
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))
if __name__ == "__main__": main()
