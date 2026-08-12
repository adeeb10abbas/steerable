"""Fail-closed independent-reader wording gate for V3-C002."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .contract import (
    AMENDMENT_ID,
    STUDY_ID,
    WORDING_GATE_SCHEMA,
    ContractError,
    canonical_json_sha256,
    file_binding,
    read_finite_json,
    registered_prompts,
    require,
    sha256_file,
)


SHEET_SCHEMA = "vla-wam-shared-v3c002-blinded-prompt-comprehension-sheet-v1"
ATTESTATION_SCHEMA = "vla-wam-shared-v3c002-human-prompt-attestation-v1"


def build_blinded_sheet() -> dict[str, Any]:
    prompts = registered_prompts()
    # Pair identifiers and presentation order are blinded to form/goal labels.
    pairs = (("pair_kestrel", "inverse_reference_right", "canonical_right"), ("pair_orchid", "canonical_left", "inverse_reference_left"))
    return {
        "schema_version": SHEET_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "purpose": "Independent wording-validity check only; this sheet is not a behavioral experiment.",
        "blinding": {
            "withheld_from_readers": ["canonical_or_inverse_label", "registered_physical_goal", "surface_direction_word", "expected_answer"],
            "reader_task": "For each pair, decide whether the two sentences specify the same physical final placement of the Rubik's cube relative to the bowl.",
            "allowed_decisions": ["same_physical_endpoint", "different_physical_endpoint", "unclear"],
        },
        "pairs": [
            {
                "reader_pair_id": pair_id,
                "sentence_a": prompts[first]["prompt"],
                "sentence_a_utf8_hex": prompts[first]["prompt_utf8_hex"],
                "sentence_a_sha256": prompts[first]["prompt_sha256"],
                "sentence_b": prompts[second]["prompt"],
                "sentence_b_utf8_hex": prompts[second]["prompt_utf8_hex"],
                "sentence_b_sha256": prompts[second]["prompt_sha256"],
            }
            for pair_id, first, second in pairs
        ],
        "response_contract": {
            "reader_identity_policy": "Record a reviewer-provided pseudonymous reader ID and a separate authorization reference; do not invent either.",
            "independence_requirement": "Two distinct authorized readers must attest independently, with no shared response file.",
            "required_fields_per_response": ["reader_id", "authorization_reference", "attested_at_utc", "responses", "signature_or_record_reference"],
        },
    }


def validate_attestations(*, sheet_path: Path, attestation_paths: list[Path]) -> dict[str, Any]:
    sheet = read_finite_json(sheet_path)
    require(isinstance(sheet, dict) and sheet.get("schema_version") == SHEET_SCHEMA, "blinded wording sheet schema changed")
    require(len(attestation_paths) == 2, "exactly two independent reader attestations are required")
    pair_ids = {str(pair["reader_pair_id"]) for pair in sheet["pairs"]}
    reader_ids: set[str] = set()
    records = []
    for path in attestation_paths:
        value = read_finite_json(path)
        require(isinstance(value, dict) and value.get("schema_version") == ATTESTATION_SCHEMA, "reader attestation schema changed")
        require(value.get("sheet_sha256") == sha256_file(sheet_path), "reader attestation addresses a different sheet")
        reader_id = value.get("reader_id")
        require(isinstance(reader_id, str) and reader_id.strip(), "reader ID is absent")
        require(reader_id not in reader_ids, "reader IDs are not independent")
        reader_ids.add(reader_id)
        require(isinstance(value.get("authorization_reference"), str) and value["authorization_reference"].strip(), "reader authorization evidence is absent")
        require(isinstance(value.get("attested_at_utc"), str) and value["attested_at_utc"].strip(), "reader attestation time is absent")
        require(isinstance(value.get("signature_or_record_reference"), str) and value["signature_or_record_reference"].strip(), "reader signature/record reference is absent")
        responses = value.get("responses")
        require(isinstance(responses, list) and len(responses) == len(pair_ids), "reader response count differs")
        result = {}
        for response in responses:
            require(isinstance(response, dict) and response.get("reader_pair_id") in pair_ids, "reader pair is invalid")
            decision = response.get("decision")
            require(decision in {"same_physical_endpoint", "different_physical_endpoint", "unclear"}, "reader decision is invalid")
            result[str(response["reader_pair_id"])] = decision
        require(set(result) == pair_ids, "reader did not answer every blinded pair")
        require(all(decision == "same_physical_endpoint" for decision in result.values()), "wording gate did not receive agreement")
        records.append({"reader_id": reader_id, "responses": result, "attestation": file_binding(path)})
    return {
        "schema_version": WORDING_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "passed": True,
        "status": "passed_two_authorized_independent_human_readers_agree_same_endpoint",
        "sheet": file_binding(sheet_path),
        "reader_attestations": records,
        "paired_endpoint_interpretation": "Both blinded pairs were independently judged to specify the same physical endpoint.",
    }


def pending_gate(*, sheet_path: Path, reason: str) -> dict[str, Any]:
    return {
        "schema_version": WORDING_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "passed": False,
        "status": "blocked_pending_two_authorized_independent_human_reader_attestations",
        "sheet": file_binding(sheet_path),
        "reason": reason,
        "behavioral_release_authorized": False,
        "model_requests_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", type=Path, required=True)
    parser.add_argument("--attestation", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.attestation:
        value = validate_attestations(sheet_path=args.sheet, attestation_paths=args.attestation)
    else:
        value = pending_gate(
            sheet_path=args.sheet,
            reason="No existing authorized two-reader evidence was located; external independent human attestations are required before behavioral registration.",
        )
    if args.output.exists():
        raise ContractError(f"refusing to overwrite wording gate: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
