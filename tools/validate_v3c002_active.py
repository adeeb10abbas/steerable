#!/usr/bin/env python3
"""Validate an active V3-C002 registration before or after behavioral release."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    RELEASE_GATE_SCHEMA, WORDING_GATE_SCHEMA, ContractError, load_cells, read_finite_json,
    require_released_gate, sha256_file, validate_file_binding,
)
ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active"
FINAL_RELATIVE = (
    "results/episodes.jsonl", "results/pairs.jsonl", "results/results.json",
    "results/DECISION_MEMO.md", "results/evidence_manifest.json", "MANUSCRIPT_INSERT.md",
)
def validate(root: Path = ROOT) -> dict:
    root = Path(root).resolve(); registration_path = root / "registration.json"; queue_path = root / "queue.jsonl"
    registration, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    if registration.get("registration_status") != "registered_after_two_human_wording_agreements": raise ContractError("active C002 registration status changed")
    wording = read_finite_json(root / "wording_gate.json")
    if not isinstance(wording, dict) or wording.get("schema_version") != WORDING_GATE_SCHEMA or wording.get("passed") is not True: raise ContractError("active wording gate did not pass")
    readers = wording.get("reader_attestations")
    if not isinstance(readers, list) or len(readers) != 2 or len({row.get("reader_id") for row in readers}) != 2: raise ContractError("active wording gate lacks two independent readers")
    receipt = read_finite_json(root / "attestation_receipt_order.json")
    if not isinstance(receipt, dict) or receipt.get("received_before_registration") is not True or receipt.get("receipt_source") != "Codex task response": raise ContractError("attestation receipt order is missing")
    if "does not invent reader attestation times" not in str(receipt.get("timestamp_limitation")): raise ContractError("date-only attestation limitation is not disclosed")
    for record in receipt.get("attestations", []): validate_file_binding(record, "received human attestation")
    release_path = root / "release_gate.released.json" if (root / "release_gate.released.json").is_file() else root / "release_gate.json"
    release = read_finite_json(release_path)
    if not isinstance(release, dict) or release.get("schema_version") != RELEASE_GATE_SCHEMA: raise ContractError("active release schema changed")
    for label, source in (("registration", registration_path), ("queue", queue_path), ("wording_gate", root / "wording_gate.json"), ("attestation_receipt_order", root / "attestation_receipt_order.json")):
        record = release.get(label); validate_file_binding(record, f"active release {label}")
        if record.get("sha256") != sha256_file(source): raise ContractError(f"active release {label} digest changed")
    infra = root / "infrastructure_attempts.jsonl"
    if not infra.is_file(): raise ContractError("active infrastructure ledger is missing")
    present = [relative for relative in FINAL_RELATIVE if (root / relative).exists()]
    if present and len(present) != len(FINAL_RELATIVE): raise ContractError(f"partial C002 final result bundle: {present}")
    if release.get("passed") is not True:
        if present: raise ContractError("results exist before behavioral release")
        if registration.get("model_requests_authorized") is not False or registration.get("behavioral_episodes_authorized") is not False: raise ContractError("blocked active registration authorizes behavior")
        return {"status": "valid_registered_pending_preflight_release", "queue_cells": len(cells), "result_artifacts_present": 0, "infrastructure_ledger_bytes": infra.stat().st_size}
    require_released_gate(registration_path=registration_path, queue_path=queue_path, release_gate_path=release_path)
    return {"status": "valid_registered_released", "queue_cells": len(cells), "result_artifacts_present": len(present), "infrastructure_ledger_bytes": infra.stat().st_size}
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT); args = parser.parse_args()
    try: print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except ContractError as exc: raise SystemExit(f"active V3-C002 validation failed: {exc}") from exc
if __name__ == "__main__": main()
