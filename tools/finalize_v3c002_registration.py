#!/usr/bin/env python3
"""Activate V3-C002 only after two real independent wording attestations.

This is intentionally a *registration* finalizer, not an inference launcher.
It leaves the fail-closed draft untouched and writes a separate immutable
activation package.  Runtime/preflight/isolation gates still remain required.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    AMENDMENT_ID,
    ContractError,
    REGISTRATION_SCHEMA,
    RELEASE_GATE_SCHEMA,
    file_binding,
    read_finite_json,
    repo_file_binding,
    sha256_file,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.wording_gate import validate_attestations  # noqa: E402


def _write_new(path: Path, value: object) -> None:
    if path.exists():
        raise ContractError(f"refusing to overwrite activation evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-root", type=Path, required=True)
    parser.add_argument("--reader-attestation", action="append", type=Path, required=True)
    parser.add_argument("--registered-at-utc", required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    args = parser.parse_args()
    draft_root = args.draft_root.resolve()
    activation_root = args.activation_root.resolve()
    draft_registration = draft_root / "registration.json"
    queue = draft_root / "queue.jsonl"
    sheet = draft_root / "prompt_comprehension_sheet.json"
    draft = read_finite_json(draft_registration)
    if not isinstance(draft, dict) or draft.get("schema_version") != REGISTRATION_SCHEMA or draft.get("registration_status") != "pre_registration_draft_pending_two_human_wording_agreements":
        raise ContractError("only the untouched fail-closed C002 draft may be activated")
    supersession = read_finite_json(draft_root / "supersession.json")
    if not isinstance(supersession, dict) or supersession.get("status") != "prospective_v8_supersedes_unexecuted_v1_v2_v3_v4_v5_v6_v7_packages" or supersession.get("v1_v2_v3_v4_v5_v6_v7_must_never_be_activated") is not True:
        raise ContractError("only the disclosed V8 superseding draft may be activated")
    active_queue = activation_root / "queue.jsonl"
    infrastructure = activation_root / "infrastructure_attempts.jsonl"
    for path in (active_queue, infrastructure):
        if path.exists():
            raise ContractError(f"refusing to overwrite activation evidence: {path}")
    wording_gate = validate_attestations(sheet_path=sheet, attestation_paths=args.reader_attestation)
    wording_gate["sheet"] = repo_file_binding(sheet)
    for record, source in zip(wording_gate["reader_attestations"], args.reader_attestation):
        record["attestation"] = repo_file_binding(source)
    wording_gate["draft_registration"] = repo_file_binding(draft_registration)
    receipt_order = {
        "schema_version": "vla-wam-shared-v3c002-attestation-receipt-order-v1",
        "status": "recorded_before_registration_activation",
        "received_before_registration": True,
        "receipt_source": "Codex task response",
        "registration_activation_at_utc": args.registered_at_utc,
        "attestations": [repo_file_binding(source) for source in args.reader_attestation],
        "timestamp_limitation": "Reader attestations supplied calendar dates but no time-of-day. Their bytes are preserved exactly; this record establishes receipt/validation order only and does not invent reader attestation times.",
        "model_requests_before_receipt_record": 0,
        "behavioral_episodes_before_receipt_record": 0,
    }
    activation_root.mkdir(parents=True, exist_ok=True)
    active_queue.write_bytes(queue.read_bytes())
    infrastructure.write_bytes(b"")
    wording_gate["queue"] = repo_file_binding(active_queue)
    _write_new(activation_root / "attestation_receipt_order.json", receipt_order)
    _write_new(activation_root / "wording_gate.json", wording_gate)
    active = dict(draft)
    active.update(
        {
            "registration_status": "registered_after_two_human_wording_agreements",
            "registered_at_utc": args.registered_at_utc,
            "model_requests_authorized": False,
            "behavioral_episodes_authorized": False,
            "pre_registration_draft": repo_file_binding(draft_registration),
            "wording_gate": repo_file_binding(activation_root / "wording_gate.json"),
            "attestation_receipt_order": repo_file_binding(activation_root / "attestation_receipt_order.json"),
            "queue": repo_file_binding(active_queue),
            "release_boundary": "Wording gate passed and queue is registered, but no model request or behavioral episode is authorized until the source commit is pushed and exact model-blind physical/runtime/raw-writer/renderer/fixed-observation/two-lane-isolation gates are passed and bound in a new release_gate.json.",
        }
    )
    _write_new(activation_root / "registration.json", active)
    release = {
        "schema_version": RELEASE_GATE_SCHEMA,
        "study_id": active["study_id"],
        "amendment_id": AMENDMENT_ID,
        "status": "blocked_pending_committed_source_and_runtime_preflight_gates",
        "passed": False,
        "registration": repo_file_binding(activation_root / "registration.json"),
        "queue": repo_file_binding(active_queue),
        "wording_gate": repo_file_binding(activation_root / "wording_gate.json"),
        "attestation_receipt_order": repo_file_binding(activation_root / "attestation_receipt_order.json"),
        "required_remaining_gates": [
            "committed_and_pushed_source_commit_bound_to_release",
            "model_blind_exact_e004_s1_physical_reset_camera_raw_writer_renderer_gate",
            "excluded_single_lane_smoke_seed",
            "excluded_two_lane_fixed-observation_isolation_test",
            "exact_pi05_runtime_identity_per_lane",
        ],
    }
    _write_new(activation_root / "release_gate.json", release)
    print(json.dumps({"status": release["status"], "activation_registration_sha256": sha256_file(activation_root / "registration.json")}, sort_keys=True))


if __name__ == "__main__":
    main()
