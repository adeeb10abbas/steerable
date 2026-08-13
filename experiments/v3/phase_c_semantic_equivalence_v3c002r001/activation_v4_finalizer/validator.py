#!/usr/bin/env python3
"""Independent hash and regeneration validator for an A004 finalizer bundle.

The validator deliberately recompiles the 1,364 raw rows through the frozen
parent compiler by calling the finalizer's pure compilation routine.  It never
uses the published episode/result values as inputs to analysis.  Consequently,
changing epoch routing, an infrastructure count, a sidecar, a pair row, a
decision memo, or any hash-bound output fails closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.v3.phase_c_semantic_equivalence_v3c002 import compiler as parent
from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    file_binding,
    load_cells,
    read_finite_json,
    require,
    sha256_file,
    validate_file_binding,
)
from .finalizer import (
    A003_RELEASE_SHA,
    FINALIZER_SCHEMA,
    INFRA_COUNT,
    ORIGINAL_RELEASE_SHA,
    PAIR_COUNT,
    RAW_COUNT,
    SEED_BLOCK_COUNT,
    RUNTIME_KEYS,
    _continuation_lanes,
    _released_lanes,
    _validate_checkout_binding,
    add_r001_diagnostics,
    build_exact_routing,
    decision_memo,
    identity_normalized_copy,
    manuscript_insert,
    validate_aggregation_receipt,
    validate_finalization_admission,
    validate_infrastructure,
    validate_mixed_provenance,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import load_repair, validate_assignment


def _read_text(path: Path) -> str:
    require(path.is_file(), f"A004 final output is missing: {path}")
    return path.read_text(encoding="utf-8")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return parent._read_jsonl(path)


def validate_bundle(
    *,
    output_dir: Path,
    parent_registration: Path,
    queue: Path,
    raw_episodes: Path,
    infrastructure_attempts: Path,
    aggregation_receipt: Path,
    finalization_registration: Path,
    finalization_source_gate: Path,
    original_release: Path,
    a003_release: Path,
    continuation_gate: Path,
    v11_registration: Path,
    v11_source_gate: Path,
) -> dict[str, Any]:
    """Rebuild and hash-check a completed A004 finalization directory."""
    output_dir = Path(output_dir).resolve()
    paths = {
        "episodes": output_dir / "episodes.jsonl",
        "pairs": output_dir / "pairs.jsonl",
        "results": output_dir / "results.json",
        "diagnostics": output_dir / "epoch_diagnostics.json",
        "memo": output_dir / "DECISION_MEMO.md",
        "insert": output_dir / "MANUSCRIPT_INSERT.md",
        "manifest": output_dir / "evidence_manifest.json",
    }
    for path in paths.values():
        _read_text(path)
    # Reconstruct admission/routing separately from compile_final.  This path
    # intentionally does not invoke compile_final; it owns a second complete
    # parent compile_episode -> _pair_rows -> compile_results regeneration.
    a003_gate, a003_lanes = _released_lanes(a003_release, expected_sha=A003_RELEASE_SHA, label="validator A0037b08")
    original_gate, original_lanes = _released_lanes(original_release, expected_sha=ORIGINAL_RELEASE_SHA, label="validator original28ee")
    a003_gate = {**a003_gate, "_path": str(a003_release.resolve())}
    original_gate = {**original_gate, "_path": str(original_release.resolve())}
    repair_registration = _validate_checkout_binding(a003_gate.get("repair_registration"), "validator A003 repair registration")
    assignment_binding = _validate_checkout_binding(a003_gate.get("assignment_manifest"), "validator A003 assignment")
    admission = validate_finalization_admission(
        finalization_registration=finalization_registration,
        finalization_source_gate=finalization_source_gate,
        parent_registration=parent_registration,
        repair_registration=repair_registration,
        queue=queue,
        original_release=original_release,
        a003_release=a003_release,
        continuation_gate=continuation_gate,
        v11_registration=v11_registration,
        v11_source_gate=v11_source_gate,
    )
    registration, parent_cells = load_cells(registration_path=parent_registration, queue_path=queue)
    _, repair_cells = load_repair(registration_path=Path(repair_registration["path"]), queue_path=queue)
    require([cell.cell_id for cell in repair_cells] == [cell.cell_id for cell in parent_cells], "validator repair queue differs from frozen parent queue")
    assignment = {int(row["episode_seed"]): str(row["lane_slot"]) for row in validate_assignment(assignment_binding)}
    continuation, continuation_lanes, remaining = _continuation_lanes(
        continuation_gate, assignment=assignment, original_release=original_release, original=original_lanes,
        a003_release=a003_release, replacement=a003_lanes,
    )
    continuation = {**continuation, "_path": str(continuation_gate.resolve())}
    routing = build_exact_routing(
        assignment=assignment, original_gate=original_gate, original_lanes=original_lanes,
        a003_gate=a003_gate, a003_lanes=a003_lanes, continuation_gate=continuation,
        continuation_lanes=continuation_lanes, continuation_remaining=remaining,
    )
    independent_epoch_counts = {epoch: sum(1 for route in routing.values() if route["epoch"] == epoch) for epoch in ("original_release", "a003_replacement_retry", "continuation")}
    require(independent_epoch_counts == {"original_release": 130, "a003_replacement_retry": 2, "continuation": 209}, "validator exact 130/2/209 epoch map changed")
    cells_by_id = {cell.cell_id: cell for cell in parent_cells}
    infra = validate_infrastructure(_jsonl(infrastructure_attempts) if infrastructure_attempts.stat().st_size else [], cells_by_id=cells_by_id, assignment=assignment)
    validate_aggregation_receipt(
        receipt_path=aggregation_receipt, raw_episodes=raw_episodes, infrastructure_attempts=infrastructure_attempts,
        repair_registration=repair_registration, queue=queue, assignment_binding=assignment_binding,
        original_release=original_release, a003_release=a003_release, continuation_gate=continuation_gate,
        routing=routing, infrastructure_rows=infra, cells_by_id=cells_by_id, assignment=assignment,
    )
    raw = _jsonl(raw_episodes)
    require(len(raw) == RAW_COUNT and len({row.get("cell_id") for row in raw}) == RAW_COUNT and set(row.get("cell_id") for row in raw) == set(cells_by_id), "validator raw cohort coverage changed")
    sidecars, regenerated_diagnostics = validate_mixed_provenance(
        raw, cells_by_id=cells_by_id, routing=routing, repair_registration=repair_registration, assignment_binding=assignment_binding,
    )
    regenerated_episodes = [
        parent.compile_episode(
            row, cell=cells_by_id[str(row["cell_id"])], registration_sha256=sha256_file(parent_registration),
            queue_sha256=sha256_file(queue), exact_runtime_contract=registration["exact_e004_pi05_runtime"],
        )
        for row in raw
    ]
    for episode in regenerated_episodes:
        route = routing[int(episode["episode_seed"])]
        episode["repair_id"] = "V3-C002-R001"
        episode["repair_lane_slot"] = route["lane_slot"]
        episode["authorization_epoch"] = route["epoch"]
        episode["authorization_gate"] = route["authorization_gate"]
        episode["released_lane_manifest"] = route["lane_manifest"]
    actual_pairs = parent._pair_rows(regenerated_episodes)
    normalized = identity_normalized_copy(regenerated_episodes)
    require(parent._pair_rows(normalized) == actual_pairs, "validator private identity copy changes frozen pair rows")
    regenerated_pairs, regenerated_results = parent.compile_results(
        normalized, registration_sha256=sha256_file(parent_registration), queue_sha256=sha256_file(queue),
    )
    require(regenerated_pairs == actual_pairs, "validator parent compile_results changed actual pair rows")
    add_r001_diagnostics(regenerated_results, regenerated_pairs, assignment)
    regenerated_diagnostics.update({
        "identity_normalization_analysis_local_only": True,
        "pair_rows_equal_before_after_normalization": True,
        "infrastructure_attempt_count_excluded": len(infra),
        "routing_seed_blocks": [
            {"episode_seed": seed, "lane_slot": route["lane_slot"], "epoch": route["epoch"], "authorization_gate_sha256": route["authorization_gate"]["sha256"], "lane_manifest_sha256": route["lane_manifest"]["sha256"]}
            for seed, route in sorted(routing.items())
        ],
        "repair_registration": repair_registration,
        "assignment_manifest": assignment_binding,
        "aggregation_receipt": file_binding(aggregation_receipt),
        **admission,
    })
    episodes = _jsonl(paths["episodes"])
    pairs = _jsonl(paths["pairs"])
    results = read_finite_json(paths["results"])
    diagnostics = read_finite_json(paths["diagnostics"])
    require(episodes == regenerated_episodes, "A004 public episodes are not an exact actual-identity regeneration")
    require(pairs == regenerated_pairs, "A004 pair rows are not an exact regeneration")
    require(results == regenerated_results, "A004 results are not an exact parent-analysis regeneration")
    require(diagnostics == regenerated_diagnostics, "A004 actual epoch diagnostics are not an exact regeneration")
    require(
        len(episodes) == RAW_COUNT
        and len({row.get("cell_id") for row in episodes}) == RAW_COUNT
        and len({row.get("episode_seed") for row in episodes}) == SEED_BLOCK_COUNT,
        "A004 public episode coverage changed",
    )
    require(len(pairs) == PAIR_COUNT, "A004 public pair count changed")
    require(
        parent.BOOTSTRAP_RESAMPLES == 20_000
        and results.get("valid_behavioral_episodes") == RAW_COUNT
        and results.get("complete_seed_blocks") == SEED_BLOCK_COUNT
        and results.get("prompt_form_pairs") == PAIR_COUNT,
        "A004 parent analysis constants/counts changed",
    )
    for goal in ("left", "right"):
        depth = results["primary_requested_side_depth_equivalence"][goal]["depth_inverse_minus_canonical_m"]
        require(depth.get("margin_m") == parent.DEPTH_MARGIN_M, "A004 depth equivalence margin changed")
        require(depth.get("tost", {}).get("margin") == parent.DEPTH_MARGIN_M, "A004 TOST margin changed")
        require(results["primary_requested_side_depth_equivalence"][goal]["pair_count"] == SEED_BLOCK_COUNT, "A004 directional pair count changed")
    controls = results.get("positive_controls")
    claim = results.get("claim_gate_components")
    require(
        isinstance(controls, dict)
        and set(controls) == {"canonical", "inverse_reference"}
        and all(isinstance(controls[key].get("positive_with_ci_excluding_zero"), bool) for key in controls)
        and isinstance(claim, dict)
        and results.get("semantic_redirection_supported") == claim.get("inverse_reference_endpoint_positive_control")
        and results.get("model_level_semantic_depth_equivalence_claim_authorized") == bool(
            claim.get("directional_depth_tost_conjunction") and claim.get("inverse_reference_endpoint_positive_control")
        ),
        "A004 positive-control keys or semantic claim conjunction changed",
    )
    require(
        _read_text(paths["memo"]) == decision_memo(regenerated_results)
        and _read_text(paths["insert"]) == manuscript_insert(regenerated_results),
        "A004 decision memo or manuscript insert is inconsistent with regenerated results",
    )
    manifest = read_finite_json(paths["manifest"])
    require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == FINALIZER_SCHEMA
        and manifest.get("repair_id") == "V3-C002-R001"
        and manifest.get("status") == "complete_hash_bound_mixed_epoch_finalization"
        and manifest.get("actual_identity_preserved") is True
        and manifest.get("identity_normalized_copy_not_published_as_episode_data") is True
        and manifest.get("pair_rows_equal_before_after_normalization") is True
        and manifest.get("infrastructure_attempt_count_excluded") == INFRA_COUNT,
        "A004 final evidence manifest contract changed",
    )
    for label in (
        "parent_registration", "queue", "raw_episodes", "infrastructure_attempts", "aggregation_receipt",
        "finalization_registration", "finalization_source_gate", "original_release28ee", "a003_release7b08",
        "continuation_gate", "v11_registration", "v11_source_gate", "compiler",
    ):
        validate_file_binding(manifest.get(label), f"A004 manifest {label}")
    expected_inputs = {
        "parent_registration": parent_registration,
        "queue": queue,
        "raw_episodes": raw_episodes,
        "infrastructure_attempts": infrastructure_attempts,
        "aggregation_receipt": aggregation_receipt,
        "finalization_registration": finalization_registration,
        "finalization_source_gate": finalization_source_gate,
        "original_release28ee": original_release,
        "a003_release7b08": a003_release,
        "continuation_gate": continuation_gate,
        "v11_registration": v11_registration,
        "v11_source_gate": v11_source_gate,
    }
    for label, supplied in expected_inputs.items():
        require(manifest[label]["sha256"] == sha256_file(supplied), f"A004 manifest supplied {label} hash changed")
    outputs = manifest.get("compiled_outputs")
    require(isinstance(outputs, dict) and set(outputs) == {"episodes.jsonl", "pairs.jsonl", "results.json", "epoch_diagnostics.json", "DECISION_MEMO.md", "MANUSCRIPT_INSERT.md"}, "A004 manifest output set changed")
    for name, binding in outputs.items():
        validate_file_binding(binding, f"A004 manifest compiled {name}")
        require(binding["sha256"] == sha256_file(output_dir / name), f"A004 manifest compiled {name} hash changed")
    raw_bindings = [
        binding
        for episode in regenerated_episodes
        for binding in [*episode["raw_artifacts"].values(), *episode["policy_camera_image_artifacts"].values()]
    ]
    for binding in raw_bindings:
        validate_file_binding(binding, "A004 retained raw artifact")
    require(
        manifest.get("raw_source_artifact_count_rehashed") == len(raw_bindings)
        and manifest.get("raw_source_bytes_rehashed") == sum(binding["bytes"] for binding in raw_bindings)
        and manifest.get("raw_source_unique_sha256_count") == len({binding["sha256"] for binding in raw_bindings}),
        "A004 raw artifact rehash totals changed",
    )
    for label, expected in (("repair_provenance_sidecars", regenerated_diagnostics["provenance_sidecars"]), ("parent_raw_episodes", regenerated_diagnostics["parent_raw_episodes"])):
        bindings = manifest.get(label)
        require(bindings == expected and isinstance(bindings, list) and len(bindings) == RAW_COUNT, f"A004 manifest {label} coverage changed")
        for binding in bindings:
            validate_file_binding(binding, f"A004 manifest {label}")
    require(
        diagnostics.get("actual_episode_identity_preserved") is True
        and diagnostics.get("identity_normalization_analysis_local_only") is True
        and diagnostics.get("pair_rows_equal_before_after_normalization") is True
        and diagnostics.get("infrastructure_attempt_count_excluded") == INFRA_COUNT
        and len(diagnostics.get("routing_seed_blocks", [])) == SEED_BLOCK_COUNT,
        "A004 epoch diagnostics contract changed",
    )
    return {
        "status": "valid_complete_v3c002r001_a004_results",
        "episodes": RAW_COUNT,
        "pairs": PAIR_COUNT,
        "infrastructure_attempts_excluded": INFRA_COUNT,
        "results_sha256": sha256_file(paths["results"]),
        "manifest_sha256": sha256_file(paths["manifest"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--raw-episodes", type=Path, required=True)
    parser.add_argument("--infrastructure-attempts", type=Path, required=True)
    parser.add_argument("--aggregation-receipt", type=Path, required=True)
    parser.add_argument("--finalization-registration", type=Path, required=True)
    parser.add_argument("--finalization-source-gate", type=Path, required=True)
    parser.add_argument("--original-release", type=Path, required=True)
    parser.add_argument("--a003-release", type=Path, required=True)
    parser.add_argument("--continuation-gate", type=Path, required=True)
    parser.add_argument("--v11-registration", type=Path, required=True)
    parser.add_argument("--v11-source-gate", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(validate_bundle(**vars(args)), indent=2, sort_keys=True))
    except ContractError as exc:
        raise SystemExit(f"V3-C002-R001 A004 final validation failed: {exc}") from exc


if __name__ == "__main__":
    main()
