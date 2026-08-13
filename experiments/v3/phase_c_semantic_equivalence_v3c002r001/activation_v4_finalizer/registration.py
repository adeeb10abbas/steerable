#!/usr/bin/env python3
"""Write the prospective A004 final-analysis registration before raw aggregation.

This tool only hashes source and predecessor gate files.  It deliberately has
no arguments for raw behavioral JSONL, result files, or outcome values, so a
committed registration cannot depend on those data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    repo_file_binding,
    require,
    sha256_file,
    validate_file_binding,
)
from .finalizer import A003_RELEASE_SHA, ORIGINAL_RELEASE_SHA


SCHEMA = "vla-wam-shared-v3c002r001-a004-final-analysis-registration-v2"
SOURCE_SCHEMA = "vla-wam-shared-v3c002r001-a004-final-analysis-source-gate-v2"
V10_CONTINUATION_SHA = "f898a52148fd39f6b5178aa7200d3539ec243ce2ed412356e2bf62e3e28139a8"


def _write_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite final-analysis registration: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registered-at-utc", required=True)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--repair-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--original-release", type=Path, required=True)
    parser.add_argument("--a003-release", type=Path, required=True)
    parser.add_argument("--v10-continuation-gate", type=Path, required=True)
    parser.add_argument("--v11-registration", type=Path, required=True)
    parser.add_argument("--v11-source-gate", type=Path, required=True)
    parser.add_argument("--superseded-registration", type=Path, required=True)
    parser.add_argument("--superseded-source-gate", type=Path, required=True)
    args = parser.parse_args()
    require(sha256_file(args.original_release) == ORIGINAL_RELEASE_SHA, "original release28ee bytes changed")
    require(sha256_file(args.a003_release) == A003_RELEASE_SHA, "A003 release7b08 bytes changed")
    require(sha256_file(args.v10_continuation_gate) == V10_CONTINUATION_SHA, "v10 continuation gate bytes changed")
    v11_source = validate_file_binding(repo_file_binding(args.v11_source_gate), "A004 v11 source gate")
    # The finalizer itself is source-controlled; raw aggregation remains a
    # source binding, never an outcome file input to registration.
    root = Path(__file__).resolve().parents[4]
    inventory_paths = (
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002/compiler.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002/contract.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/compiler.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/contract.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_finalizer/finalizer.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_finalizer/validator.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_finalizer/registration.py",
        root / "experiments/v3/phase_c_semantic_equivalence_v3c002r001/activation_v4_finalizer/source_gate.py",
        root / "tools/aggregate_v3c002r001_activation_v4_raw.py",
        root / "tests/test_aggregate_v3c002r001_activation_v4_raw.py",
        root / "tests/test_v3c002r001_activation_v4_finalizer.py",
    )
    inventory = {str(path.relative_to(root)): repo_file_binding(path) for path in inventory_paths}
    value = {
        "schema_version": SCHEMA,
        "status": "registered_prospective_corrected_final_analysis_before_raw_aggregation_or_result_read",
        "study_id": "vla_wam_language_steerability_v3",
        "repair_id": "V3-C002-R001",
        "activation_id": "V3-C002-R001-A004-final-analysis",
        "registered_at_utc": args.registered_at_utc,
        "final_analysis_raw_behavioral_rows_read_before_registration": 0,
        "final_analysis_result_compilations_before_registration": 0,
        "final_analysis_output_files_before_registration": 0,
        "superseded_v1_final_analysis": {
            "registration": repo_file_binding(args.superseded_registration),
            "source_gate": repo_file_binding(args.superseded_source_gate),
            "status": "superseded_unexecuted_before_any_outcome_aggregation_or_result_read",
            "raw_behavioral_rows_read": 0,
            "result_compilations": 0,
            "output_files": 0,
            "reason": "The additive lane diagnostics referenced a nonexistent pair-row field name. The frozen parent emits depth_difference_inverse_minus_canonical_m; this v2 prospectively corrects only that additive diagnostic reader and adds a real parent-pair regression.",
        },
        "prospective_order": [
            "commit_and_push_this_registration_and_its_source_gate",
            "perform_outcome_blind_mixed_epoch_raw_aggregation",
            "compile_actual_identity_episodes_with_frozen_parent_compiler",
            "analyze_only_a_deep_identity_normalized_copy_after_pair_row_equality",
            "validate_the_complete_bundle_by_regeneration",
        ],
        "frozen_analysis_contract": {
            "parent_compile_episode": "experiments/v3/phase_c_semantic_equivalence_v3c002/compiler.py:compile_episode",
            "parent_pair_rows": "experiments/v3/phase_c_semantic_equivalence_v3c002/compiler.py:_pair_rows",
            "parent_compile_results": "experiments/v3/phase_c_semantic_equivalence_v3c002/compiler.py:compile_results",
            "bootstrap_resamples": 20_000,
            "infrastructure_attempt_count_excluded": 14,
            "actual_identity_episode_output": True,
            "analysis_only_deep_identity_normalization": True,
            "pair_rows_must_equal_before_after_normalization": True,
            "r001_lane_and_leave_one_out_diagnostics_preserved": True,
        },
        "parent_registration": repo_file_binding(args.parent_registration),
        "repair_registration": repo_file_binding(args.repair_registration),
        "queue": repo_file_binding(args.queue),
        "original_release28ee": repo_file_binding(args.original_release),
        "a003_release7b08": repo_file_binding(args.a003_release),
        "v10_continuation_gate": repo_file_binding(args.v10_continuation_gate),
        "v11_registration": repo_file_binding(args.v11_registration),
        "v11_source_gate": v11_source,
        "source_inventory": inventory,
        "source_gate_schema_required": SOURCE_SCHEMA,
    }
    _write_new(args.output, value)
    print(json.dumps(repo_file_binding(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except ContractError as exc:
        raise SystemExit(f"A004 final-analysis registration failed: {exc}") from exc
