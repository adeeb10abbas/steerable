#!/usr/bin/env python3
"""Build the compact hash ledger for V3-E004 post-processing outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/vla_wam_shared_v3/phase_e/symmetric_layout_cohort_v3e004"
IMPLEMENTATION = ROOT / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        display = str(resolved.relative_to(ROOT))
    except ValueError:
        display = str(resolved)
    return {"path": display, "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def build(base: Path, output: Path) -> dict[str, Any]:
    base = Path(base).resolve()
    results_path = base / "results/results.json"
    episodes_path = base / "results/episodes.jsonl"
    pairs_path = base / "results/pairs.jsonl"
    invalid_path = base / "results/infrastructure_invalid.jsonl"
    discovery_path = base / "results/discovery_only.jsonl"
    ledger_path = base / "results/source_ledger.jsonl"
    memo_path = base / "DECISION_MEMO.md"
    required = (
        base / "registration.json",
        base / "queue.jsonl",
        base / "layout/candidate.json",
        base / "gates/static_layout_gate.json",
        results_path,
        episodes_path,
        pairs_path,
        invalid_path,
        discovery_path,
        ledger_path,
        memo_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing E004 compact evidence: " + ", ".join(missing))
    results = json.loads(results_path.read_text(encoding="utf-8"))
    complete = results.get("status") == "complete_hash_closed" and results.get("coverage", {}).get("complete") is True
    figure_manifest_path = base / "results/figures/figure_manifest.json"
    if complete and not figure_manifest_path.is_file():
        raise FileNotFoundError("complete E004 evidence requires rendered figures")
    compact_paths = list(required)
    if figure_manifest_path.is_file():
        figure_manifest = json.loads(figure_manifest_path.read_text(encoding="utf-8"))
        compact_paths.append(figure_manifest_path)
        for item in figure_manifest.get("figures", []):
            path = Path(item["path"])
            if not path.is_absolute():
                path = ROOT / path
            if path.is_file():
                compact_paths.append(path)
            elif complete:
                raise FileNotFoundError(f"missing rendered E004 figure: {path}")

    code_paths = sorted(IMPLEMENTATION.glob("*.py"))
    code_paths.extend(
        path
        for path in (
            ROOT / "tools/build_v3e004_registration.py",
            ROOT / "tools/validate_v3e004.py",
            ROOT / "tools/compile_v3e004_results.py",
            ROOT / "tools/render_v3e004_results.py",
            ROOT / "tools/build_v3e004_evidence_manifest.py",
            ROOT / "tools/validate_v3e004_evidence.py",
        )
        if path.is_file()
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e004-evidence-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "status": "hash_closed_compact_evidence" if complete else "partial_progress_not_publication_evidence",
        "registration_sha256": sha256(base / "registration.json"),
        "queue_sha256": sha256(base / "queue.jsonl"),
        "candidate_sha256": sha256(base / "layout/candidate.json"),
        "results_sha256": sha256(results_path),
        "episodes_sha256": sha256(episodes_path),
        "pairs_sha256": sha256(pairs_path),
        "valid_behavioral_episodes": results["valid_behavioral_episodes"],
        "registered_behavioral_cells": results["registered_behavioral_cells"],
        "infrastructure_invalid_attempts": results["infrastructure_invalid_attempts"],
        "discovery_only_behavioral_artifacts": results[
            "discovery_only_behavioral_artifacts_excluded_from_denominators"
        ],
        "publication_claim_status": results["publication_claim_status"],
        "compact_files": [record(path) for path in sorted(set(compact_paths))],
        "implementation_files": [record(path) for path in sorted(set(code_paths))],
        "raw_evidence_policy": "Full simulator videos, action traces, states, and source episode files remain on the ali-owned PVC. Compact episodes retain source SHA-256 and byte bindings; no checkpoint, environment, or unbounded rollout media is committed.",
        "arena_boundary": "DROID/RoboLab and RoboTwin remain separate and are never pooled.",
        "claim_boundary": "Partial compilation carries no publication claim. Equivalence requires both complete registered evidence and the preregistered per-estimand power/interval gate.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE / "evidence_manifest.json")
    args = parser.parse_args()
    value = build(args.base, args.output)
    print(json.dumps({"status": value["status"], "compact_files": len(value["compact_files"])}, indent=2))


if __name__ == "__main__":
    main()
