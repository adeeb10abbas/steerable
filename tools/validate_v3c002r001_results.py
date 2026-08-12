#!/usr/bin/env python3
"""Hash- and assignment-audit a completed V3-C002-R001 result bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.compiler import _read_jsonl, compile_episode, compile_results
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import ContractError, load_cells, read_finite_json, sha256_file, validate_file_binding
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import REPO_ROOT, load_repair, require, require_released_gate, validate_assignment


ROOT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002r001/active"
PARENT = REPO_ROOT / "artifacts/vla_wam_shared_v3/phase_c/semantic_equivalence_v3c002/active"


def validate(root: Path = ROOT) -> dict:
    root = Path(root).resolve(); results_root = root / "results"
    paths = {
        "episodes": results_root / "episodes.jsonl", "pairs": results_root / "pairs.jsonl",
        "results": results_root / "results.json", "memo": results_root / "DECISION_MEMO.md",
        "manifest": results_root / "evidence_manifest.json", "insert": root / "MANUSCRIPT_INSERT.md",
        "raw": root / "raw/episodes.jsonl", "infra": root / "infrastructure_attempts.jsonl",
    }
    for label, path in paths.items():
        require(path.is_file(), f"repair final {label} is missing")
    repair, cells = load_repair(registration_path=root / "registration.json", queue_path=root / "queue.jsonl")
    require_released_gate(registration_path=PARENT / "registration.json", queue_path=root / "queue.jsonl", release_gate_path=root / "release_gate.released.json")
    assignment = {row["episode_seed"]: row["lane_slot"] for row in validate_assignment(repair["assignment_manifest"])}
    episodes = _read_jsonl(paths["episodes"]); raw = _read_jsonl(paths["raw"]); pairs = _read_jsonl(paths["pairs"])
    require(len(episodes) == len(raw) == 1364 and len({row.get("cell_id") for row in episodes}) == 1364, "repair episode coverage changed")
    require(len({cell.cell_id for cell in cells}) == 1364, "repair queue coverage changed")
    for episode in episodes:
        require(episode.get("repair_id") == "V3-C002-R001" and episode.get("repair_lane_slot") == assignment.get(episode.get("episode_seed")), "repair compiled episode assignment changed")
    parent_registration, parent_cells = load_cells(registration_path=PARENT / "registration.json", queue_path=root / "queue.jsonl")
    cell_map = {cell.cell_id: cell for cell in parent_cells}
    regenerated = [compile_episode(row, cell=cell_map[str(row["cell_id"])], registration_sha256=sha256_file(PARENT / "registration.json"), queue_sha256=sha256_file(root / "queue.jsonl"), exact_runtime_contract=parent_registration["exact_e004_pi05_runtime"]) for row in raw]
    stripped = [{key: value for key, value in episode.items() if key not in ("repair_id", "repair_lane_slot")} for episode in episodes]
    require(regenerated == stripped, "repair episodes are not regenerated exactly from raw")
    regenerated_pairs, regenerated_results = compile_results(regenerated, registration_sha256=sha256_file(PARENT / "registration.json"), queue_sha256=sha256_file(root / "queue.jsonl"))
    require(pairs == regenerated_pairs and len(pairs) == 682 and all(pair.get("episode_seed") in assignment for pair in pairs), "repair pair coverage/regeneration changed")
    results = read_finite_json(paths["results"])
    require(isinstance(results, dict) and results.get("repair_id") == "V3-C002-R001" and results.get("original_cross_lane_gate_remains_failed") is True and results.get("cross_lane_numerical_tolerance_used") is False, "repair result routing changed")
    require(set(results.get("lane_diagnostics_descriptive_only", {})) == {f"repair-lane-{index:02d}" for index in range(8)}, "repair lane diagnostics incomplete")
    require(set(results.get("leave_one_lane_out_diagnostics_descriptive_only", {})) == {f"repair-lane-{index:02d}" for index in range(8)}, "repair leave-one-lane-out diagnostics incomplete")
    repair_only_result_keys = {"repair_id", "original_cross_lane_gate_remains_failed", "cross_lane_numerical_tolerance_used", "lane_diagnostics_descriptive_only", "leave_one_lane_out_diagnostics_descriptive_only"}
    require({key: value for key, value in results.items() if key not in repair_only_result_keys} == regenerated_results, "repair results changed the parent registered analysis")
    infra = _read_jsonl(paths["infra"]) if paths["infra"].stat().st_size else []
    require(all(row.get("infrastructure_status") == "infrastructure_invalid_excluded" for row in infra), "repair infrastructure ledger changed")
    manifest = read_finite_json(paths["manifest"])
    require(isinstance(manifest, dict) and manifest.get("schema_version") == "vla-wam-shared-v3c002r001-evidence-manifest-v1" and manifest.get("status") == "complete_hash_bound_repair_results", "repair evidence manifest incomplete")
    for label in ("repair_registration", "parent_registration", "queue", "assignment_manifest", "release_gate", "raw_episodes", "infrastructure_attempts"):
        validate_file_binding(manifest.get(label), f"repair manifest {label}")
    for label, binding in manifest.get("compiled_outputs", {}).items():
        validate_file_binding(binding, f"repair compiled {label}")
    for binding in manifest.get("repair_provenance_sidecars", []):
        validate_file_binding(binding, "repair raw provenance")
    raw_bindings = [binding for episode in regenerated for binding in list(episode["raw_artifacts"].values()) + list(episode["policy_camera_image_artifacts"].values())]
    for binding in raw_bindings:
        validate_file_binding(binding, "repair retained raw artifact")
    require(manifest.get("raw_source_artifact_count_rehashed") == len(raw_bindings) and manifest.get("raw_source_bytes_rehashed") == sum(binding["bytes"] for binding in raw_bindings), "repair raw rehash totals changed")
    require(manifest.get("valid_behavioral_episode_count") == 1364 and manifest.get("complete_seed_block_count") == 341, "repair evidence counts changed")
    return {"status": "valid_complete_v3c002r001_results", "episodes": len(episodes), "pairs": len(pairs), "infrastructure_attempts_excluded": len(infra), "results_sha256": sha256_file(paths["results"]), "manifest_sha256": sha256_file(paths["manifest"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", type=Path, default=ROOT); args = parser.parse_args()
    try:
        print(json.dumps(validate(args.root), indent=2, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(f"V3-C002-R001 final validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
