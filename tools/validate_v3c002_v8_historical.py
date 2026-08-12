#!/usr/bin/env python3
"""Audit the immutable, unreleased V8 activation and its non-operative smoke authorization."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, sha256_file  # noqa: E402


ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/superseded_activation_v8"
EXPECTED = {
    "registration.json": "d79d57f8f15f7f41400c38c96aa9d07737adf0aa1a38a15db560ab4856211066",
    "queue.jsonl": "a3cca2f97f22f82935434d4200947a2a7647b3bead8dad6cbba95ab55437561e",
    "release_gate.json": "27f04da11dc6a2aa75c9eb34a5f934ef23472f672963e289ed5f8723f7bae4d9",
    "source_push_gate.json": "5020769419f189073521f6cac5b12dd4176402d21ea707c5c8df71b258f5edf6",
    "wording_gate.json": "1dd4b963a8ef1656fc766493dc81c9c2362b01387fc230cc7fc3a4549b5ee8a8",
    "attestation_receipt_order.json": "a5adbb0c4260616bf03addc3428c0f3f856c789ea429d3cdb0bfd2da734368b8",
    "gates/model_blind_physical_gate.json": "64c1a1cafc2d7903567d9d281a5ef79a1e67204a4eae53b434de2aee87a4efe7",
    "gates/excluded_smoke_authorization.json": "8028dea7b2c6e4e8216bc1e4d8d8fe163465efa28cd558d0977c4828df02d2ab",
}


def validate() -> dict[str, object]:
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ContractError(f"historical V8 artifact changed: {relative}")
    registration = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
    release = json.loads((ROOT / "release_gate.json").read_text(encoding="utf-8"))
    if release.get("passed") is not False or registration.get("model_requests_authorized") is not False or registration.get("behavioral_episodes_authorized") is not False:
        raise ContractError("historical V8 activation was not fail closed")
    if (ROOT / "infrastructure_attempts.jsonl").read_bytes() != b"":
        raise ContractError("historical V8 contains an execution ledger")
    with (ROOT / "queue.jsonl").open(encoding="utf-8") as handle:
        queue_rows = sum(1 for _ in handle)
    return {
        "status": "valid_immutable_unexecuted_superseded_v8_activation",
        "queue_rows": queue_rows,
        "model_requests": 0,
        "behavioral_episodes": 0,
        "provisional_smoke_authorization_nonoperative": True,
        "supersession_reason": "V8 authorization preceded the required exact C002 same-process zero-request proof, target-side raw rehash receipt, and explicit same-process flag enforcement; it was never executed.",
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
