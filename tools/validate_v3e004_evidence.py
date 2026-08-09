#!/usr/bin/env python3
"""Validate partial or complete V3-E004 compact evidence and claim gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"


class Invalid(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Invalid(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve(path: str, *, base: Path) -> Path:
    value = Path(path)
    if value.is_absolute():
        return value
    repository = ROOT / value
    return repository if repository.exists() else base / value


def validate(base: Path, *, require_complete: bool, verify_raw_sources: bool) -> dict[str, Any]:
    base = Path(base).resolve()
    results_path = base / "results/results.json"
    episodes_path = base / "results/episodes.jsonl"
    pairs_path = base / "results/pairs.jsonl"
    invalid_path = base / "results/infrastructure_invalid.jsonl"
    discovery_path = base / "results/discovery_only.jsonl"
    source_ledger_path = base / "results/source_ledger.jsonl"
    manifest_path = base / "evidence_manifest.json"
    for path in (
        results_path,
        episodes_path,
        pairs_path,
        invalid_path,
        discovery_path,
        source_ledger_path,
        manifest_path,
        base / "DECISION_MEMO.md",
    ):
        require(path.is_file(), f"missing compact E004 evidence: {path}")
    results = load_json(results_path)
    episodes = load_jsonl(episodes_path)
    pairs = load_jsonl(pairs_path)
    invalid = load_jsonl(invalid_path)
    discovery = load_jsonl(discovery_path)
    ledger = load_jsonl(source_ledger_path)
    manifest = load_json(manifest_path)
    require(results.get("amendment_id") == "V3-E004", "wrong result amendment")
    require(len(episodes) == results.get("valid_behavioral_episodes"), "episode/result count differs")
    require(len({row.get("cell_id") for row in episodes}) == len(episodes), "duplicate compact episode cell")
    require(len({row.get("matched_pair_id") for row in pairs}) == len(pairs), "duplicate compact matched pair")
    require(len(pairs) == sum(item["complete_pairs"] for item in results["checkpoints"].values()), "pair/result count differs")
    require(len(invalid) == results.get("infrastructure_invalid_attempts"), "invalid-attempt count differs")
    require(
        len(discovery)
        == results.get("discovery_only_behavioral_artifacts_excluded_from_denominators"),
        "discovery-only count differs",
    )
    for row in discovery:
        require(
            row.get("disposition") == "discovery_only_excluded_from_behavioral_denominator"
            and row.get("behavioral_denominator_included") is False,
            "discovery-only row is denominator-eligible",
        )
        require(
            row.get("reason") == "pre_r002_s0_missing_prospective_attestation",
            "unknown discovery-only exclusion reason",
        )
    require(results.get("coverage", {}).get("valid_cells") == len(episodes), "coverage count differs")
    complete = results.get("coverage", {}).get("complete") is True
    require(
        results.get("publication_claim_status")
        == (
            "enabled_subject_to_per-estimand_power_and_positive-control_gates"
            if complete
            else "withheld_until_all_registered_cells_are_valid"
        ),
        "publication claim gate differs from completeness",
    )
    if require_complete:
        require(complete, "registered E004 cohort is incomplete")
        require(results.get("status") == "complete_hash_closed", "complete result status missing")
        require(len(episodes) == results.get("registered_behavioral_cells") == 4096, "complete E004 cell count differs")
    else:
        require(
            results.get("status") in {"complete_hash_closed", "partial_progress_no_publication_claims"},
            "unknown result status",
        )
    for row in episodes:
        require(row.get("missing_measurement_policy") == "NR remains null and is never converted to zero", "NR policy missing")
        require(row.get("arena") in {"droid_robolab", "robotwin"}, "unknown arena")
        require(row.get("relation") in {"left", "right"}, "unknown relation")
        require(type(row.get("success")) is bool, "success is not boolean")
        for field in ("signed_final_lateral_offset", "requested_side_depth", "asymmetry_metric_A"):
            value = row.get(field)
            require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)), f"invalid {field}")
        source = row.get("source_raw_episode", {})
        require(isinstance(source.get("sha256"), str) and len(source["sha256"]) == 64, "source digest missing")
        if row.get("arena") == "droid_robolab":
            for field in (
                "request0_pair_identity_sha256",
                "request0_observation_payload_sha256",
                "request0_reset_contract_sha256",
            ):
                require(isinstance(row.get(field), str) and len(row[field]) == 64, f"missing R001 {field}")
            require(row.get("request0_replay_mode") in {"capture_left", "replay_right"}, "invalid R001 replay mode")
        if verify_raw_sources:
            path = Path(source["path"])
            require(path.is_file(), f"raw source unavailable: {path}")
            require(path.stat().st_size == source["bytes"] and sha256(path) == source["sha256"], f"raw source changed: {path}")
        if row.get("pair_fields_status") == "derived_after_both_hash_bound_directions_exist":
            require(type(row.get("action_distinct")) is bool and isinstance(row.get("endpoint_shift"), (int, float)), "materialized pair fields invalid")
    for row in pairs:
        if row.get("arena") == "droid_robolab":
            require(
                row.get("identical_reset_definition")
                == "R001 identical request0 observation bytes and reset-contract payload",
                "DROID pair uses an obsolete native-reset identity definition",
            )
    for model, checkpoint in results["checkpoints"].items():
        gate = checkpoint["claim_gate"]
        if not complete:
            require(gate.get("publication_claims_enabled") is False, f"partial checkpoint claim enabled: {model}")
            continue
        require(gate.get("publication_claims_enabled") is True, f"complete checkpoint claim gate disabled: {model}")
        for estimand, item in gate["equivalence_claims"].items():
            if item["publication_equivalence_claim_allowed"]:
                require(item["registered_power_gate_passed"] and item["margin_defined"] and item["interval_within_registered_margin"], f"invalid equivalence claim: {model}/{estimand}")
            if model in {"dreamzero_droid_action_cfg", "cosmos3_edge_policy_droid", "fastwam_robotwin"}:
                require(item["publication_equivalence_claim_allowed"] is False, f"underpowered equivalence claim: {model}/{estimand}")
        for level, positive in gate["H4_endpoint_positive_control_by_level"].items():
            if not positive:
                require(gate["equalisation_interpretation_allowed_by_level"][level] is False, f"failed positive control did not close claim: {model}/{level}")
    require(manifest.get("registration_sha256") == sha256(base / "registration.json"), "manifest registration hash differs")
    require(manifest.get("results_sha256") == sha256(results_path), "manifest results hash differs")
    require(manifest.get("episodes_sha256") == sha256(episodes_path), "manifest episodes hash differs")
    require(manifest.get("pairs_sha256") == sha256(pairs_path), "manifest pairs hash differs")
    require(
        manifest.get("discovery_only_behavioral_artifacts") == len(discovery),
        "manifest discovery-only count differs",
    )
    require(
        manifest.get("status")
        == ("hash_closed_compact_evidence" if complete else "partial_progress_not_publication_evidence"),
        "manifest status differs",
    )
    for item in manifest.get("compact_files", []) + manifest.get("implementation_files", []):
        path = resolve(item["path"], base=base)
        require(path.is_file(), f"manifested file missing: {path}")
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"], f"manifested file changed: {path}")
    require(results["source_summary"]["source_ledger_rows"] == len(ledger), "source ledger count differs")
    return {
        "status": "valid_complete" if complete else "valid_partial_no_publication_claims",
        "valid_behavioral_episodes": len(episodes),
        "infrastructure_invalid_attempts": len(invalid),
        "discovery_only_behavioral_artifacts": len(discovery),
        "source_ledger_rows": len(ledger),
        "raw_sources_verified": verify_raw_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--verify-raw-sources", action="store_true")
    args = parser.parse_args()
    try:
        report = validate(args.base, require_complete=args.require_complete, verify_raw_sources=args.verify_raw_sources)
    except (Invalid, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, indent=2))
        raise SystemExit(1)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
