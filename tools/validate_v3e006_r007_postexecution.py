#!/usr/bin/env python3
"""Validate retained R007 evidence after a validator-only schema correction.

The frozen R007 runtime copies a registered stage schedule and then annotates the
retained copy with ``candidate_rank``.  The pre-execution validator compared that
retained mapping to the unannotated frozen mapping.  This wrapper requires the
runtime's exact annotation, removes only that redundant annotation in a copy,
and delegates every scientific/evidence check to the frozen validator.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from tools import validate_v3e006_r007 as frozen


ROOT = Path(__file__).resolve().parents[1]
AMENDMENT = (
    ROOT
    / "artifacts/vla_wam_shared_v3/phase_e/canonical_stage_localization_v3e006_r007"
    / "postexecution_validator_amendment_v1.json"
)
SCHEMA = "vla-wam-shared-v3e006-r007-postexecution-validator-amendment-v1"
STATUS = "validator_only_correction_after_execution_before_r007_closure_or_behavior"
FROZEN_SOURCE_GATE_V2_SHA256 = (
    "386cc701d3a184988b3fd42cbd83ade886b9f89b7aad49a773ef18c80f4a1307"
)
RAW_RESULT_SHA256 = "f20314dcd32d9d6503dd5c2bd3777a08563fea3ca2a47c5f91bc26f4483a5cd6"
_FROZEN_VALIDATE_OPEN_CONTACT_STATE = frozen.validate_open_contact_state


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_binding(root: Path, row: Mapping[str, Any], label: str) -> Path:
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = root / path
    frozen.require(
        path.is_file()
        and path.stat().st_size == row.get("bytes")
        and sha256(path) == row.get("sha256"),
        f"{label} binding differs: {path}",
    )
    return path


def validate_open_contact_state(
    state: Mapping[str, Any],
    label: str,
    expected_stage: Mapping[str, Any],
    *,
    candidate_rank: int,
) -> None:
    """Require the exact retained rank annotation, then run frozen checks."""

    construction = state.get("construction")
    frozen.require(isinstance(construction, Mapping), f"{label} construction absent")
    expected_retained = dict(expected_stage)
    expected_retained["candidate_rank"] = candidate_rank
    frozen.require(
        construction.get("registered_stage_schedule") == expected_retained,
        f"{label} retained registered stage schedule differs",
    )
    normalized = deepcopy(state)
    normalized["construction"]["registered_stage_schedule"] = deepcopy(expected_stage)
    _FROZEN_VALIDATE_OPEN_CONTACT_STATE(
        normalized, label, expected_stage, candidate_rank=candidate_rank
    )


def validate_amendment(root: Path) -> dict[str, Any]:
    value = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    frozen.require(value.get("schema_version") == SCHEMA, "validator amendment schema differs")
    frozen.require(value.get("status") == STATUS, "validator amendment status differs")
    frozen.require(value.get("model_request_count") == 0, "validator amendment model count differs")
    frozen.require(value.get("behavioral_episode_count") == 0, "validator amendment behavior count differs")
    frozen.require(value.get("accepted_state_candidate_count") == 0, "validator amendment accepted count differs")
    frozen.require(value.get("raw_result", {}).get("sha256") == RAW_RESULT_SHA256, "raw result digest differs")
    frozen.require(
        value.get("correction") == {
            "retained_runtime_field": "construction.registered_stage_schedule.candidate_rank",
            "runtime_semantics": "exact redundant copy of enclosing candidate_rank",
            "frozen_validator_defect": "compared annotated retained mapping to unannotated frozen stage mapping",
            "corrected_expectation": "retained_stage_schedule == {**frozen_stage_schedule, candidate_rank: enclosing_rank}",
            "scientific_execution_change": False,
            "raw_evidence_change": False,
        },
        "validator correction contract differs",
    )
    source_gate_path = verify_binding(root, value["frozen_source_push_gate_v2"], "frozen source gate v2")
    frozen.require(sha256(source_gate_path) == FROZEN_SOURCE_GATE_V2_SHA256, "frozen source gate digest differs")
    gate = json.loads(source_gate_path.read_text(encoding="utf-8"))
    inventory = {row["path"]: row for row in gate["implementation_files"]}
    for relative in (
        "experiments/v3/phase_e/canonical_stage_localization_v3e006_r007/state_repair_gate.py",
        "tools/validate_v3e006_r007.py",
        "tests/test_v3e006_r007.py",
    ):
        frozen.require(value["frozen_execution_inventory"][relative] == inventory[relative], f"frozen inventory differs: {relative}")
        verify_binding(root, inventory[relative], f"frozen execution inventory {relative}")
    for label, row in value["corrected_validator_inventory"].items():
        verify_binding(root, row, f"corrected validator inventory {label}")
    for label in ("raw_result", "raw_harness", "raw_launch", "raw_runtime_log"):
        verify_binding(root, value[label], label)
    implementation = str(value.get("correction_implementation_commit", ""))
    frozen.require(
        subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{implementation}^{{commit}}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0,
        "validator correction implementation commit absent",
    )
    frozen.require(
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", implementation, "HEAD"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0,
        "validator correction implementation is not an ancestor",
    )
    return {"passed": True, "amendment": binding(AMENDMENT)}


def validate_candidate_root(root: Path, candidate_root: Path) -> dict[str, Any]:
    original = frozen.validate_open_contact_state
    frozen.validate_open_contact_state = validate_open_contact_state
    try:
        return frozen.validate_candidate_root(root, candidate_root)
    finally:
        frozen.validate_open_contact_state = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--verify-raw", action="store_true", required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    result = frozen.validate_static(root, require_source_gate=True, verify_retry_history=True)
    result["postexecution_validator_amendment"] = validate_amendment(root)
    result["candidate_evidence"] = validate_candidate_root(root, args.candidate_root)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
