#!/usr/bin/env python3
"""Independently reconstruct and hash-audit a completed V3-C002 cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.compiler import (  # noqa: E402
    _read_jsonl, compile_episode, compile_results, decision_memo, manuscript_insert,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    ContractError, load_cells, read_finite_json, require_released_gate, sha256_file, validate_file_binding,
)


ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active"


def validate(root: Path = ROOT) -> dict:
    root = Path(root).resolve(); results_root = root / "results"
    registration_path, queue_path = root / "registration.json", root / "queue.jsonl"
    release_path = root / "release_gate.released.json"
    raw_path, infra_path = root / "raw/episodes.jsonl", root / "infrastructure_attempts.jsonl"
    episodes_path, pairs_path, results_path = results_root / "episodes.jsonl", results_root / "pairs.jsonl", results_root / "results.json"
    memo_path, evidence_path, insert_path = results_root / "DECISION_MEMO.md", results_root / "evidence_manifest.json", root / "MANUSCRIPT_INSERT.md"
    for path in (registration_path, queue_path, release_path, raw_path, infra_path, episodes_path, pairs_path, results_path, memo_path, evidence_path, insert_path):
        if not path.is_file(): raise ContractError(f"required final C002 artifact is missing: {path}")
    registration, cells = load_cells(registration_path=registration_path, queue_path=queue_path)
    require_released_gate(registration_path=registration_path, queue_path=queue_path, release_gate_path=release_path)
    if registration.get("registration_status") != "registered_after_two_human_wording_agreements": raise ContractError("final registration is inactive")
    raw_rows = _read_jsonl(raw_path); cell_map = {cell.cell_id: cell for cell in cells}
    if len(raw_rows) != 1364 or {str(row.get("cell_id")) for row in raw_rows} != set(cell_map): raise ContractError("raw behavioral coverage differs from queue")
    registration_sha, queue_sha = sha256_file(registration_path), sha256_file(queue_path)
    regenerated = [compile_episode(row, cell=cell_map[str(row["cell_id"])], registration_sha256=registration_sha, queue_sha256=queue_sha, exact_runtime_contract=registration["exact_e004_pi05_runtime"]) for row in raw_rows]
    committed_episodes = _read_jsonl(episodes_path)
    if regenerated != committed_episodes: raise ContractError("episodes.jsonl is not regenerated exactly from raw episodes")
    pairs, results = compile_results(regenerated, registration_sha256=registration_sha, queue_sha256=queue_sha)
    if pairs != _read_jsonl(pairs_path): raise ContractError("pairs.jsonl is not regenerated exactly")
    if results != read_finite_json(results_path): raise ContractError("results.json is not regenerated exactly")
    infra = _read_jsonl(infra_path) if infra_path.stat().st_size else []
    if any(row.get("infrastructure_status") != "infrastructure_invalid_excluded" for row in infra): raise ContractError("infrastructure ledger contains behavioral data")
    if decision_memo(results) != memo_path.read_text(encoding="utf-8"): raise ContractError("decision memo differs from results")
    if manuscript_insert(results) != insert_path.read_text(encoding="utf-8"): raise ContractError("manuscript insert differs from results")
    manifest = read_finite_json(evidence_path)
    if not isinstance(manifest, dict) or manifest.get("status") != "complete_hash_bound_results": raise ContractError("evidence manifest is incomplete")
    for label, record in (("registration", manifest.get("registration")), ("queue", manifest.get("queue")), ("release gate", manifest.get("release_gate")), ("raw episodes", manifest.get("raw_episodes")), ("infrastructure attempts", manifest.get("infrastructure_attempts"))):
        validate_file_binding(record, label)
    outputs = manifest.get("compiled_outputs")
    if not isinstance(outputs, dict): raise ContractError("compiled output bindings are missing")
    for label, record in outputs.items(): validate_file_binding(record, f"compiled {label}")
    raw_records = [record for episode in regenerated for record in list(episode["raw_artifacts"].values()) + list(episode["policy_camera_image_artifacts"].values())]
    if manifest.get("raw_source_artifact_count_rehashed") != len(raw_records) or manifest.get("raw_source_bytes_rehashed") != sum(record["bytes"] for record in raw_records): raise ContractError("raw rehash totals differ")
    if manifest.get("infrastructure_attempt_count_excluded") != len(infra): raise ContractError("infrastructure exclusion count differs")
    return {"status": "valid_complete_v3c002_results", "episodes": len(regenerated), "pairs": len(pairs), "infrastructure_attempts_excluded": len(infra), "raw_artifacts_rehashed": len(raw_records), "results_sha256": sha256_file(results_path), "evidence_manifest_sha256": sha256_file(evidence_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT); args = parser.parse_args()
    try: print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except ContractError as exc: raise SystemExit(f"V3-C002 final validation failed: {exc}") from exc


if __name__ == "__main__": main()
