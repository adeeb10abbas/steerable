#!/usr/bin/env python3
"""Build and verify the immutable C01--C24 fixture cohort freeze.

The input inventory contains only model-blind fixture evidence paths and exact
hashes.  ``freeze`` deeply revalidates every accepted gate, pose manifest and
fixed observation, then writes one hash-signed 24-layout artifact.  Behavioral
confirmation launchers require this artifact and bind their selected layout to
its exact row; this program never launches a model or simulator action.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

import confirmation_runtime_common as common


INVENTORY_SCHEMA = "wmf-confirmation-fixture-inventory-v1"
FREEZE_SCHEMA = "wmf-confirmation-fixture-freeze-v1"
FREEZE_STATUS = "frozen_for_confirmation"
INPUT_ROW_KEYS = {
    "layout_pair_id",
    "candidate_id",
    "gate_receipt_path",
    "gate_receipt_sha256",
    "pose_manifest_path",
    "pose_manifest_sha256",
    "capture_receipt_path",
    "capture_receipt_sha256",
}
FROZEN_ROW_KEYS = {
    "layout_pair_id",
    "environment_seed",
    "candidate_id",
    "candidate_payload_sha256",
    "accepted_gate_record_sha256",
    "gate_receipt",
    "pose_manifest",
    "gate_ledger",
    "gate_attempt_receipt",
    "capture_receipt",
    "raw_capture_receipt",
    "n3_fixed_observation",
    "d1_fixed_observation",
    "model_request_count",
    "behavioral_action_count",
}
TOP_LEVEL_KEYS = {
    "schema_version",
    "study_id",
    "namespace",
    "status",
    "created_from_study_commit",
    "selection_uses_target_model_outcomes",
    "candidate_selection_rule",
    "layout_count",
    "layouts",
    "model_request_count",
    "behavioral_action_count",
    "claim_boundary",
    "payload_sha256",
}
CANDIDATE_RE = re.compile(r"C(?:0[1-9]|1[0-9]|2[0-4])__candidate_[0-9]{2}\Z")
EXPECTED_ACCEPTED_CANDIDATES = {
    layout: f"{layout}__candidate_{1 if layout in {'C01', 'C17', 'C24'} else 0:02d}"
    for layout in common.CONFIRMATION_LAYOUT_IDS
}


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    common.require(set(value) == expected, "confirmation_fixture_keys_changed", label)


def _payload_hash(value: Mapping[str, Any]) -> str:
    unsigned = dict(value)
    unsigned.pop("payload_sha256", None)
    import hashlib

    return hashlib.sha256(common.canonical_bytes(unsigned)).hexdigest()


def _verify_descriptor(value: Any, label: str) -> dict[str, Any]:
    common.require(isinstance(value, Mapping), "confirmation_fixture_descriptor_invalid", label)
    _exact_keys(value, {"path", "bytes", "sha256"}, label)
    identity = common.verify_exact_file(Path(str(value.get("path", ""))), str(value.get("sha256", "")), label)
    common.require(identity == dict(value), "confirmation_fixture_descriptor_changed", label)
    return identity


def _load_runtime_modules(source_root: Path):
    forecast = (
        Path(source_root).resolve()
        / "workshops/corl2026_world_models/experiments/forecast_layout"
    )
    if str(forecast) not in sys.path:
        sys.path.insert(0, str(forecast))
    import d1_development_block_jobs as d1_development
    import fixed_observation_job as fixed

    return d1_development, fixed


def validate_one_fixture(
    *,
    source_root: Path,
    layout_pair_id: str,
    candidate_id: str,
    gate_receipt_path: Path,
    gate_receipt_sha256: str,
    pose_manifest_path: Path,
    pose_manifest_sha256: str,
    capture_receipt_path: Path,
    capture_receipt_sha256: str,
) -> dict[str, Any]:
    """Deeply authenticate one selected model-blind C-layout fixture."""

    common.require(layout_pair_id in common.CONFIRMATION_LAYOUT_IDS, "confirmation_fixture_layout_invalid", layout_pair_id)
    common.require(CANDIDATE_RE.fullmatch(candidate_id) is not None, "confirmation_fixture_candidate_invalid", candidate_id)
    common.require(candidate_id.startswith(f"{layout_pair_id}__candidate_"), "confirmation_fixture_candidate_layout_mismatch")
    common.require(
        candidate_id == EXPECTED_ACCEPTED_CANDIDATES[layout_pair_id],
        "confirmation_fixture_accepted_candidate_changed",
        layout_pair_id,
    )
    d1_development, fixed = _load_runtime_modules(source_root)
    try:
        release = fixed.verify_gate_and_pose_manifest(
            gate_receipt_path=Path(gate_receipt_path),
            gate_receipt_sha256=gate_receipt_sha256,
            pose_manifest_path=Path(pose_manifest_path),
            pose_manifest_sha256=pose_manifest_sha256,
            layout_pair_id=layout_pair_id,
            candidate_id=candidate_id,
        )
    except BaseException as error:
        raise common.ConfirmationRuntimeError("confirmation_fixture_gate_pose_invalid", str(error)) from error
    environment_seed = 2026091200 + int(layout_pair_id[1:])
    block = SimpleNamespace(
        layout_pair_id=layout_pair_id,
        environment_seed=environment_seed,
    )
    try:
        capture = d1_development.verify_development_capture(
            Path(capture_receipt_path),
            capture_receipt_sha256,
            block=block,
            release=release,
        )
    except BaseException as error:
        raise common.ConfirmationRuntimeError("confirmation_fixture_capture_invalid", str(error)) from error
    raw = d1_development.pilot.load_json(
        Path(capture["raw_capture_receipt"]["path"]),
        "confirmation_raw_capture_unreadable",
    )
    fixtures = raw.get("model_fixtures")
    common.require(
        isinstance(fixtures, Mapping) and set(fixtures) == {"N3", "D1"},
        "confirmation_capture_model_fixtures_invalid",
    )
    n3 = fixtures.get("N3")
    d1 = fixtures.get("D1")
    common.require(isinstance(n3, Mapping) and isinstance(d1, Mapping), "confirmation_capture_model_fixture_missing")
    n3_identity = d1_development._verify_descriptor(
        n3.get("fixture"), "confirmation_n3_fixed_observation"
    )
    d1_identity = d1_development._verify_descriptor(
        d1.get("fixture"), "confirmation_d1_fixed_observation"
    )
    common.require(
        d1_identity == capture["d1_fixed_observation"],
        "confirmation_d1_fixed_observation_changed",
    )
    return {
        "layout_pair_id": layout_pair_id,
        "environment_seed": environment_seed,
        "candidate_id": release["candidate_id"],
        "candidate_payload_sha256": release["candidate_payload_sha256"],
        "accepted_gate_record_sha256": release["accepted_gate_record_sha256"],
        "gate_receipt": release["gate_receipt"],
        "pose_manifest": release["pose_manifest"],
        "gate_ledger": release["gate_ledger"],
        "gate_attempt_receipt": release["gate_attempt_receipt"],
        "capture_receipt": capture["capture_receipt"],
        "raw_capture_receipt": capture["raw_capture_receipt"],
        "n3_fixed_observation": n3_identity,
        "d1_fixed_observation": d1_identity,
        "model_request_count": 0,
        "behavioral_action_count": 0,
    }


def build_fixture_freeze(
    *, source_root: Path, inventory_path: Path, study_commit: str
) -> dict[str, Any]:
    common.require(common.COMMIT_RE.fullmatch(study_commit) is not None, "confirmation_fixture_study_commit_invalid")
    inventory = common.load_json(Path(inventory_path), "confirmation_fixture_inventory")
    _exact_keys(
        inventory,
        {
            "schema_version",
            "study_id",
            "namespace",
            "selection_uses_target_model_outcomes",
            "candidate_selection_rule",
            "layouts",
        },
        "fixture inventory",
    )
    common.require(inventory.get("schema_version") == INVENTORY_SCHEMA, "confirmation_fixture_inventory_schema_changed")
    common.require(inventory.get("study_id") == common.STUDY_ID, "confirmation_fixture_inventory_study_changed")
    common.require(inventory.get("namespace") == common.NAMESPACE, "confirmation_fixture_inventory_namespace_changed")
    common.require(
        inventory.get("selection_uses_target_model_outcomes") is False,
        "confirmation_fixture_selection_not_model_blind",
    )
    rule = inventory.get("candidate_selection_rule")
    common.require(isinstance(rule, str) and bool(rule.strip()), "confirmation_fixture_selection_rule_missing")
    rows = inventory.get("layouts")
    common.require(isinstance(rows, list) and len(rows) == 24, "confirmation_fixture_inventory_count_changed")
    by_layout: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        common.require(isinstance(row, Mapping), "confirmation_fixture_inventory_row_invalid")
        _exact_keys(row, INPUT_ROW_KEYS, "fixture inventory row")
        layout = row.get("layout_pair_id")
        common.require(isinstance(layout, str) and layout in common.CONFIRMATION_LAYOUT_IDS, "confirmation_fixture_layout_invalid", str(layout))
        common.require(layout not in by_layout, "confirmation_fixture_layout_duplicate", layout)
        common.require(
            row.get("candidate_id") == EXPECTED_ACCEPTED_CANDIDATES[layout],
            "confirmation_fixture_accepted_candidate_changed",
            layout,
        )
        by_layout[layout] = row
    common.require(set(by_layout) == set(common.CONFIRMATION_LAYOUT_IDS), "confirmation_fixture_layout_inventory_incomplete")
    frozen_rows = []
    for layout in common.CONFIRMATION_LAYOUT_IDS:
        row = by_layout[layout]
        frozen_rows.append(
            validate_one_fixture(
                source_root=source_root,
                layout_pair_id=layout,
                candidate_id=str(row["candidate_id"]),
                gate_receipt_path=Path(str(row["gate_receipt_path"])),
                gate_receipt_sha256=str(row["gate_receipt_sha256"]),
                pose_manifest_path=Path(str(row["pose_manifest_path"])),
                pose_manifest_sha256=str(row["pose_manifest_sha256"]),
                capture_receipt_path=Path(str(row["capture_receipt_path"])),
                capture_receipt_sha256=str(row["capture_receipt_sha256"]),
            )
        )
    document = {
        "schema_version": FREEZE_SCHEMA,
        "study_id": common.STUDY_ID,
        "namespace": common.NAMESPACE,
        "status": FREEZE_STATUS,
        "created_from_study_commit": study_commit,
        "selection_uses_target_model_outcomes": False,
        "candidate_selection_rule": rule,
        "layout_count": 24,
        "layouts": frozen_rows,
        "model_request_count": 0,
        "behavioral_action_count": 0,
        "claim_boundary": (
            "This freeze authenticates 24 model-blind physical fixtures and fixed observations. "
            "It contains zero learned-policy requests or actions and does not release confirmation "
            "without the separate post-development confirmation freeze."
        ),
    }
    return {**document, "payload_sha256": _payload_hash(document)}


def validate_fixture_freeze(
    path: Path,
    expected_sha256: str,
    *,
    source_root: Path,
    expected_layout_pair_id: str | None = None,
    expected_study_commit: str | None = None,
    deep_validate_selected: bool = True,
) -> dict[str, Any]:
    """Verify all 24 frozen identities and optionally re-run one deep check."""

    identity = common.verify_exact_file(path, expected_sha256, "confirmation_fixture_freeze")
    value = common.load_json(Path(identity["path"]), "confirmation_fixture_freeze")
    _exact_keys(value, TOP_LEVEL_KEYS, "fixture freeze")
    expected = {
        "schema_version": FREEZE_SCHEMA,
        "study_id": common.STUDY_ID,
        "namespace": common.NAMESPACE,
        "status": FREEZE_STATUS,
        "selection_uses_target_model_outcomes": False,
        "layout_count": 24,
        "model_request_count": 0,
        "behavioral_action_count": 0,
    }
    for key, wanted in expected.items():
        common.require(value.get(key) == wanted, "confirmation_fixture_freeze_mismatch", key)
    common.require(
        isinstance(value.get("created_from_study_commit"), str)
        and common.COMMIT_RE.fullmatch(value["created_from_study_commit"]) is not None,
        "confirmation_fixture_study_commit_invalid",
    )
    if expected_study_commit is not None:
        common.require(
            common.COMMIT_RE.fullmatch(expected_study_commit) is not None
            and value["created_from_study_commit"] == expected_study_commit,
            "confirmation_fixture_study_commit_changed",
        )
    common.require(isinstance(value.get("candidate_selection_rule"), str) and bool(value["candidate_selection_rule"].strip()), "confirmation_fixture_selection_rule_missing")
    common.require(value.get("payload_sha256") == _payload_hash(value), "confirmation_fixture_payload_hash_mismatch")
    rows = value.get("layouts")
    common.require(isinstance(rows, list) and len(rows) == 24, "confirmation_fixture_freeze_count_changed")
    by_layout: dict[str, dict[str, Any]] = {}
    for expected_index, row in enumerate(rows, start=1):
        common.require(isinstance(row, Mapping), "confirmation_fixture_frozen_row_invalid")
        _exact_keys(row, FROZEN_ROW_KEYS, "frozen fixture row")
        layout = f"C{expected_index:02d}"
        common.require(row.get("layout_pair_id") == layout, "confirmation_fixture_row_order_changed", layout)
        common.require(row.get("environment_seed") == 2026091200 + expected_index, "confirmation_fixture_seed_changed", layout)
        candidate = row.get("candidate_id")
        common.require(isinstance(candidate, str) and CANDIDATE_RE.fullmatch(candidate) is not None and candidate.startswith(f"{layout}__candidate_"), "confirmation_fixture_candidate_invalid", layout)
        common.require(
            candidate == EXPECTED_ACCEPTED_CANDIDATES[layout],
            "confirmation_fixture_accepted_candidate_changed",
            layout,
        )
        for key in ("candidate_payload_sha256", "accepted_gate_record_sha256"):
            common.require(isinstance(row.get(key), str) and common.SHA256_RE.fullmatch(row[key]) is not None, "confirmation_fixture_hash_invalid", f"{layout}:{key}")
        for key in (
            "gate_receipt", "pose_manifest", "gate_ledger", "gate_attempt_receipt",
            "capture_receipt", "raw_capture_receipt", "n3_fixed_observation",
            "d1_fixed_observation",
        ):
            _verify_descriptor(row.get(key), f"{layout}_{key}")
        common.require(row.get("model_request_count") == 0 and row.get("behavioral_action_count") == 0, "confirmation_fixture_contains_behavioral_work", layout)
        by_layout[layout] = dict(row)
    common.require(set(by_layout) == set(common.CONFIRMATION_LAYOUT_IDS), "confirmation_fixture_layout_inventory_incomplete")
    selected = None
    if expected_layout_pair_id is not None:
        common.require(expected_layout_pair_id in by_layout, "confirmation_fixture_selected_layout_missing", str(expected_layout_pair_id))
        selected = by_layout[expected_layout_pair_id]
        if deep_validate_selected:
            observed = validate_one_fixture(
                source_root=source_root,
                layout_pair_id=expected_layout_pair_id,
                candidate_id=selected["candidate_id"],
                gate_receipt_path=Path(selected["gate_receipt"]["path"]),
                gate_receipt_sha256=selected["gate_receipt"]["sha256"],
                pose_manifest_path=Path(selected["pose_manifest"]["path"]),
                pose_manifest_sha256=selected["pose_manifest"]["sha256"],
                capture_receipt_path=Path(selected["capture_receipt"]["path"]),
                capture_receipt_sha256=selected["capture_receipt"]["sha256"],
            )
            common.require(observed == selected, "confirmation_fixture_selected_row_drifted", expected_layout_pair_id)
    return {
        "fixture_freeze": identity,
        "created_from_study_commit": value["created_from_study_commit"],
        "candidate_selection_rule": value["candidate_selection_rule"],
        "selected_layout": selected,
        "layout_count": 24,
        "model_request_count": 0,
        "behavioral_action_count": 0,
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(common.canonical_bytes(value))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            target.unlink()
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--source-root", type=Path, required=True)
    freeze.add_argument("--inventory", type=Path, required=True)
    freeze.add_argument("--study-commit", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--source-root", type=Path, required=True)
    validate.add_argument("--freeze", type=Path, required=True)
    validate.add_argument("--freeze-sha256", required=True)
    validate.add_argument("--layout-pair-id", choices=common.CONFIRMATION_LAYOUT_IDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "freeze":
        value = build_fixture_freeze(
            source_root=args.source_root,
            inventory_path=args.inventory,
            study_commit=args.study_commit,
        )
        _write_immutable_json(args.output, value)
        print(json.dumps(common.file_identity(args.output), sort_keys=True), flush=True)
        return 0
    result = validate_fixture_freeze(
        args.freeze,
        args.freeze_sha256,
        source_root=args.source_root,
        expected_layout_pair_id=args.layout_pair_id,
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "technical_failure",
                    "error_type": type(error).__name__,
                    "reason": getattr(error, "reason", None),
                    "detail": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise
