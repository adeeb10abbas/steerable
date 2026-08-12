#!/usr/bin/env python3
"""Validate V3-C002 registration/queue integrity without running inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    AMENDMENT_ID,
    MODEL_ID,
    SEEDS,
    ContractError,
    load_cells,
    read_finite_json,
    sha256_file,
    validate_exact_runtime_contract,
)
from tools.validate_v3c002_v1_historical import validate as validate_v1_historical  # noqa: E402
from tools.validate_v3c002_v2_historical import validate as validate_v2_historical  # noqa: E402
from tools.validate_v3c002_v3_historical import validate as validate_v3_historical  # noqa: E402
from tools.validate_v3c002_v4_historical import validate as validate_v4_historical  # noqa: E402
from tools.validate_v3c002_v5_historical import validate as validate_v5_historical  # noqa: E402
from tools.validate_v3c002_v6_historical import validate as validate_v6_historical  # noqa: E402
from tools.validate_v3c002_v7_historical import validate as validate_v7_historical  # noqa: E402


ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/draft_v8"


def _resolve(binding: dict, *, root: Path) -> Path:
    path = Path(str(binding.get("path", "")))
    return path if path.is_absolute() else (root / path)


def _check_binding(binding: object, *, root: Path, label: str, checks: list[str]) -> None:
    if not isinstance(binding, dict):
        raise ContractError(f"{label} binding is missing")
    if Path(str(binding.get("path", ""))).is_absolute():
        raise ContractError(f"{label} committed binding is not checkout-portable")
    path = _resolve(binding, root=root)
    if not path.is_file():
        raise ContractError(f"{label} file is missing: {path}")
    if binding.get("bytes") != path.stat().st_size or binding.get("sha256") != sha256_file(path):
        raise ContractError(f"{label} binding differs")
    checks.append(f"{label} hash and bytes match")


def validate(root: Path = ROOT) -> dict:
    root = Path(root).resolve()
    registration_path = root / "registration.json"
    queue_path = root / "queue.jsonl"
    gate_path = root / "wording_gate.json"
    release_path = root / "release_gate.json"
    checks: list[str] = []
    registration, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    checks.append("1364 unique queue cells and every four-cell seed block are complete")
    if len(cells) != 1364 or len({cell.seed for cell in cells}) != len(SEEDS):
        raise ContractError("C002 queue arithmetic differs")
    if any(cell.row["runtime_identity_requirement"].get("checkpoint") != "pi05_droid_jointpos_polaris" for cell in cells):
        raise ContractError("C002 did not bind the exact E004 π0.5 checkpoint")
    if any(cell.row["runtime_identity_requirement"].get("action_horizon") != 15 for cell in cells):
        raise ContractError("C002 action horizon changed")
    if any(cell.row["runtime_identity_requirement"].get("action_dim") != 8 for cell in cells):
        raise ContractError("C002 action interface changed")
    checks.append("π0.5 E004 checkpoint and 8D/15-step action contract are exact")
    runtime_sha = validate_exact_runtime_contract(registration.get("exact_e004_pi05_runtime"))
    if any(cell.row.get("exact_runtime_contract_sha256") != runtime_sha for cell in cells):
        raise ContractError("a queue cell differs from the exact E004 runtime contract")
    checks.append("controller, cameras, horizon, scorer, raw writer, renderer, checkpoint, runtime, and policy server are path/byte/SHA bound and equality checked")
    layout = registration.get("e004_s1_layout")
    if not isinstance(layout, dict) or layout.get("symmetry_level_s") != 1.0:
        raise ContractError("C002 does not require E004 s=1 layout")
    _check_binding(layout.get("candidate"), root=REPO_ROOT, label="E004 candidate", checks=checks)
    if layout.get("candidate_sha256") != layout["candidate"]["sha256"]:
        raise ContractError("E004 candidate digest changed")
    for source, binding in sorted(registration.get("source_bindings", {}).items()):
        _check_binding(binding, root=REPO_ROOT, label=f"source {source}", checks=checks)
    supersession_path = root / "supersession.json"
    supersession = read_finite_json(supersession_path)
    if not isinstance(supersession, dict) or supersession.get("status") != "prospective_v8_supersedes_unexecuted_v1_v2_v3_v4_v5_v6_v7_packages":
        raise ContractError("V8 supersession disclosure is missing")
    if any(supersession.get(key) != 0 for key in ("v1_model_requests", "v1_behavioral_episodes", "v2_model_requests", "v2_behavioral_episodes", "v3_model_requests", "v3_behavioral_episodes", "v4_model_requests", "v4_behavioral_episodes", "v5_model_requests", "v5_behavioral_episodes", "v6_model_requests", "v6_behavioral_episodes", "v7_model_requests", "v7_behavioral_episodes")) or supersession.get("v1_v2_v3_v4_v5_v6_v7_must_never_be_activated") is not True:
        raise ContractError("V1 through V7 were not immutably retired before inference")
    for revision, records in supersession.get("superseded_draft_bindings", {}).items():
        for name, binding in records.items():
            _check_binding(binding, root=REPO_ROOT, label=f"superseded {revision} {name}", checks=checks)
    checks.append("V1 through V7 packages remain immutable, unexecuted, and permanently superseded")
    gate = read_finite_json(gate_path)
    if not isinstance(gate, dict) or gate.get("amendment_id") != AMENDMENT_ID:
        raise ContractError("wording gate is invalid")
    _check_binding(gate.get("sheet"), root=REPO_ROOT, label="blinded wording sheet", checks=checks)
    release = read_finite_json(release_path)
    if not isinstance(release, dict) or release.get("passed") is not False:
        raise ContractError("pre-registration release gate must be explicitly failed closed")
    _check_binding(release.get("registration"), root=REPO_ROOT, label="draft registration", checks=checks)
    _check_binding(release.get("queue"), root=REPO_ROOT, label="draft queue", checks=checks)
    _check_binding(release.get("wording_gate"), root=REPO_ROOT, label="pending wording gate", checks=checks)
    required = set(release.get("required_bound_artifacts", []))
    if required != {"source_push_gate", "physical_gate", "excluded_smoke_gate", "two_lane_isolation_gate", "lane_manifests"}:
        raise ContractError("blocked release gate does not enumerate every hard release artifact")
    status = registration.get("registration_status")
    if status != "pre_registration_draft_pending_two_human_wording_agreements":
        raise ContractError("no unverified registration status is permitted")
    if registration.get("model_requests_authorized") is not False or registration.get("behavioral_episodes_authorized") is not False:
        raise ContractError("draft must not authorize inference")
    if gate.get("passed") is not False or gate.get("model_requests_authorized") is not False:
        raise ContractError("missing human evidence must fail closed")
    if (root / "infrastructure_attempts.jsonl").read_text(encoding="utf-8") != "":
        raise ContractError("no infrastructure attempt may be recorded before release")
    checks.append("no fabricated human agreement, model request, or behavioral authorization")
    historical_v1 = validate_v1_historical()
    historical_v2 = validate_v2_historical()
    historical_v3 = validate_v3_historical()
    historical_v4 = validate_v4_historical()
    historical_v5 = validate_v5_historical()
    historical_v6 = validate_v6_historical()
    historical_v7 = validate_v7_historical()
    checks.append("independent historical validators confirm immutable unexecuted fail-closed V1 through V7 packages")
    return {"status": "valid_superseding_v8_pre_registration_draft_blocked_pending_activation", "check_count": len(checks), "checks": checks, "queue_sha256": sha256_file(queue_path), "historical_v1": historical_v1, "historical_v2": historical_v2, "historical_v3": historical_v3, "historical_v4": historical_v4, "historical_v5": historical_v5, "historical_v6": historical_v6, "historical_v7": historical_v7}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(f"V3-C002 validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
