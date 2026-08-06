#!/usr/bin/env python3
"""Compile the V3 DROID direction-by-failure-mode split.

This is a read-only retrospective analysis of the committed 54-episode
Phase-A summaries. It does not relabel an episode or modify the frozen V3
failure taxonomy. The failure-only test is the probability-ordered,
two-sided Fisher-Freeman-Halton exact test for a 2 x K table, conditional on
the observed row and column margins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts/vla_wam_shared_v3/phase_b/pi05_mirror_v3b002/analysis"
)

TAXONOMY_PATH = REPO_ROOT / "artifacts/vla_wam_shared_v3/failure_taxonomy.json"
CATEGORIES = (
    "correct",
    "pick_failed",
    "transport_failed",
    "wrong_side",
    "release_failed",
)
FAILURE_CATEGORIES = CATEGORIES[1:]
DISPLAY_ALIASES = {
    "correct": "correct",
    "pick_failed": "pick",
    "transport_failed": "transport",
    "wrong_side": "wrong_side",
    "release_failed": "release",
}

SOURCES = (
    {
        "model_id": "pi05_current_stack_droid",
        "display_name": "pi0.5 current stack DROID",
        "summary": "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_summary.json",
        "evidence_manifest": "artifacts/vla_wam_shared_v3/results/pi05_current_stack_droid_phase_a_evidence_hash_manifest.json",
    },
    {
        "model_id": "dreamzero_droid_action_cfg",
        "display_name": "DreamZero DROID action guidance s=2",
        "summary": "artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_summary.json",
        "evidence_manifest": "artifacts/vla_wam_shared_v3/results/dreamzero_droid_action_cfg_phase_a_evidence_hash_manifest.json",
    },
    {
        "model_id": "cosmos3_edge_policy_droid",
        "display_name": "Cosmos3 Edge Policy DROID",
        "summary": "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_summary.json",
        "evidence_manifest": "artifacts/vla_wam_shared_v3/results/cosmos3_edge_policy_droid_phase_a_evidence_hash_manifest.json",
    },
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write_bytes(path: Path, content: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": str(path.relative_to(REPO_ROOT)),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def success_value(cell: dict[str, Any]) -> bool:
    values = [cell[key] for key in ("success", "requested_success") if key in cell]
    if len(values) != 1 or not isinstance(values[0], bool):
        raise ValueError(
            f"{cell.get('registered_cell_id')}: expected one Boolean success field"
        )
    return values[0]


def raw_behavioral_hash(cell: dict[str, Any]) -> str | None:
    if "raw_behavioral_pair_jsonl_sha256" in cell:
        return cell["raw_behavioral_pair_jsonl_sha256"]
    hashes = cell.get("hashes", {})
    return hashes.get("compiled_jsonl_sha256")


def allocations(column_totals: tuple[int, ...], row_total: int) -> Iterable[tuple[int, ...]]:
    """Yield every possible first row with the fixed 2 x K margins."""

    def visit(index: int, remaining: int, prefix: tuple[int, ...]):
        if index == len(column_totals) - 1:
            if 0 <= remaining <= column_totals[index]:
                yield prefix + (remaining,)
            return
        tail_capacity = sum(column_totals[index + 1 :])
        lower = max(0, remaining - tail_capacity)
        upper = min(column_totals[index], remaining)
        for value in range(lower, upper + 1):
            yield from visit(index + 1, remaining - value, prefix + (value,))

    yield from visit(0, row_total, ())


def fisher_freeman_halton_two_sided(
    left: tuple[int, ...], right: tuple[int, ...]
) -> dict[str, Any]:
    """Exact conditional 2 x K test with probability-ordered two-sided p."""

    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Fisher-Freeman-Halton requires matched K >= 2 columns")
    if any(value < 0 for value in left + right):
        raise ValueError("counts must be nonnegative")

    retained = tuple(
        index for index, total in enumerate(a + b for a, b in zip(left, right)) if total
    )
    reduced_left = tuple(left[index] for index in retained)
    reduced_right = tuple(right[index] for index in retained)
    column_totals = tuple(a + b for a, b in zip(reduced_left, reduced_right))
    left_total = sum(reduced_left)
    grand_total = sum(column_totals)
    denominator = math.comb(grand_total, left_total)

    def probability_numerator(first_row: tuple[int, ...]) -> int:
        return math.prod(
            math.comb(column_total, value)
            for column_total, value in zip(column_totals, first_row)
        )

    observed_numerator = probability_numerator(reduced_left)
    two_sided_numerator = 0
    enumerated_table_count = 0
    equally_or_less_probable_table_count = 0
    for candidate in allocations(column_totals, left_total):
        numerator = probability_numerator(candidate)
        enumerated_table_count += 1
        if numerator <= observed_numerator:
            two_sided_numerator += numerator
            equally_or_less_probable_table_count += 1

    observed_probability = Fraction(observed_numerator, denominator)
    p_value = Fraction(two_sided_numerator, denominator)
    return {
        "test": "Fisher-Freeman-Halton exact test for a 2xK contingency table",
        "alternative": "two_sided_probability_ordering",
        "conditioning": "fixed row totals and fixed failure-category column totals",
        "zero_total_columns_dropped_for_enumeration": [
            FAILURE_CATEGORIES[index]
            for index in range(len(FAILURE_CATEGORIES))
            if index not in retained
        ],
        "enumerated_table_count": enumerated_table_count,
        "equally_or_less_probable_table_count": equally_or_less_probable_table_count,
        "observed_table_probability": float(observed_probability),
        "observed_table_probability_exact": (
            f"{observed_probability.numerator}/{observed_probability.denominator}"
        ),
        "p_value": float(p_value),
        "p_value_exact": f"{p_value.numerator}/{p_value.denominator}",
    }


def normalized(counts: dict[str, int], denominator: int) -> dict[str, float]:
    if denominator <= 0:
        raise ValueError("normalization denominator must be positive")
    return {category: counts[category] / denominator for category in CATEGORIES}


def compile_model(source: dict[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = REPO_ROOT / source["summary"]
    evidence_path = REPO_ROOT / source["evidence_manifest"]
    summary = json.loads(summary_path.read_text())
    if summary.get("model_id") != source["model_id"]:
        raise ValueError(f"model mismatch in {summary_path}")
    cells = summary.get("cells")
    if not isinstance(cells, list) or len(cells) != 54:
        raise ValueError(f"{source['model_id']}: expected exactly 54 committed cells")

    per_direction: dict[str, Counter[str]] = {
        "left": Counter(),
        "right": Counter(),
    }
    derived_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for cell in cells:
        cell_id = cell.get("registered_cell_id")
        direction = cell.get("relation")
        category = cell.get("failure_taxonomy")
        if not isinstance(cell_id, str) or cell_id in seen_ids:
            raise ValueError(f"{source['model_id']}: missing or duplicate cell ID {cell_id}")
        if direction not in per_direction:
            raise ValueError(f"{cell_id}: invalid direction {direction}")
        if category not in CATEGORIES:
            raise ValueError(f"{cell_id}: taxonomy value {category} is not frozen")
        success = success_value(cell)
        if success != (category == "correct"):
            raise ValueError(f"{cell_id}: success/taxonomy mismatch")
        seen_ids.add(cell_id)
        per_direction[direction][category] += 1
        derived_rows.append(
            {
                "schema_version": "vla-wam-shared-v3-failure-mode-episode-view-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "arena": "droid_robolab",
                "cohort": "phase_a_expanded_54_episode",
                "model_id": source["model_id"],
                "registered_cell_id": cell_id,
                "seed": cell.get("seed"),
                "direction": direction,
                "success": success,
                "failure_category": category,
                "failure_category_display_alias": DISPLAY_ALIASES[category],
                "source_summary": source["summary"],
                "source_summary_sha256": sha256_path(summary_path),
                "source_raw_behavioral_jsonl_sha256": raw_behavioral_hash(cell),
            }
        )

    for direction in ("left", "right"):
        if sum(per_direction[direction].values()) != 27:
            raise ValueError(f"{source['model_id']}: {direction} denominator is not 27")
    overall = Counter(cell["failure_taxonomy"] for cell in cells)
    if dict(overall) != summary.get("overall_failure_taxonomy_counts"):
        raise ValueError(f"{source['model_id']}: per-cell counts disagree with summary")

    tables: dict[str, Any] = {}
    failure_rows: dict[str, tuple[int, ...]] = {}
    for direction in ("left", "right"):
        counts = {category: per_direction[direction][category] for category in CATEGORIES}
        failure_count = 27 - counts["correct"]
        failure_counts = {category: counts[category] for category in FAILURE_CATEGORIES}
        failure_rows[direction] = tuple(failure_counts[category] for category in FAILURE_CATEGORIES)
        tables[direction] = {
            "raw_counts": counts,
            "row_normalized_proportions_all_episodes": normalized(counts, 27),
            "failure_count": failure_count,
            "failure_only_raw_counts": failure_counts,
            "failure_only_row_normalized_proportions": {
                category: failure_counts[category] / failure_count
                for category in FAILURE_CATEGORIES
            },
            "success_rate": counts["correct"] / 27,
            "failure_rate": failure_count / 27,
        }

    exact = fisher_freeman_halton_two_sided(
        failure_rows["left"], failure_rows["right"]
    )
    alpha = 0.05
    exact["alpha_descriptive_threshold"] = alpha
    exact["failure_shape_difference_detected_at_alpha"] = exact["p_value"] < alpha
    if exact["p_value"] < alpha:
        interpretation = (
            "Failure-mode shape differs by direction in this cohort; the observed "
            "asymmetry is not described by a failure-rate change alone."
        )
    else:
        interpretation = (
            "No failure-mode shape difference is detected in this cohort. This is "
            "compatible with, but does not prove, a same-shape/different-rate account; "
            "interpretation is limited by the smaller direction-specific failure row."
        )

    result = {
        "model_id": source["model_id"],
        "display_name": source["display_name"],
        "arena": "droid_robolab",
        "cohort": "phase_a_expanded_54_episode",
        "directions": tables,
        "table_2x5": {
            "columns": list(CATEGORIES),
            "column_display_aliases": [DISPLAY_ALIASES[value] for value in CATEGORIES],
            "rows": {
                direction: [tables[direction]["raw_counts"][category] for category in CATEGORIES]
                for direction in ("left", "right")
            },
        },
        "failure_only_table_2x4": {
            "columns": list(FAILURE_CATEGORIES),
            "column_display_aliases": [
                DISPLAY_ALIASES[value] for value in FAILURE_CATEGORIES
            ],
            "rows": {
                direction: list(failure_rows[direction])
                for direction in ("left", "right")
            },
        },
        "failure_only_exact_test": exact,
        "interpretation": interpretation,
        "source_summary": {
            "path": source["summary"],
            "bytes": summary_path.stat().st_size,
            "sha256": sha256_path(summary_path),
        },
        "source_evidence_manifest": {
            "path": source["evidence_manifest"],
            "bytes": evidence_path.stat().st_size,
            "sha256": sha256_path(evidence_path),
        },
    }
    return result, derived_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    taxonomy = json.loads(TAXONOMY_PATH.read_text())
    if tuple(taxonomy["primary_precedence"]) != (
        "correct",
        "pick_failed",
        "wrong_side",
        "release_failed",
        "transport_failed",
    ):
        raise ValueError("frozen V3 taxonomy precedence changed unexpectedly")

    model_results = []
    episode_rows = []
    for source in SOURCES:
        result, rows = compile_model(source)
        model_results.append(result)
        episode_rows.extend(rows)
    episode_rows.sort(key=lambda row: (row["model_id"], row["seed"], row["direction"]))
    if len(episode_rows) != 162:
        raise ValueError("expected 162 derived episode records")

    jsonl_bytes = b"".join(
        (json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in episode_rows
    )
    jsonl_meta = write_bytes(output_dir / "failure_mode_split_episodes.jsonl", jsonl_bytes)

    report = {
        "schema_version": "vla-wam-shared-v3-failure-mode-split-report-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_retrospective_analysis_no_new_inference",
        "analysis_date": "2026-08-06",
        "analysis_git_head": git_head(),
        "scope": {
            "arena": "droid_robolab",
            "cohort": "three separate V3 Phase-A expanded 54-episode cohorts",
            "models": [source["model_id"] for source in SOURCES],
            "pooling": "No model cohorts are pooled.",
            "new_model_compute": False,
        },
        "hypothesis": (
            "If leftward placement is only harder in rate, LEFT and RIGHT failures "
            "should have the same failure-mode shape. A direction-dependent shape "
            "supports a direction-specific mechanism within the tested cohort."
        ),
        "method": {
            "table": "direction x frozen V3 failure taxonomy",
            "all_episode_table": "2x5 raw counts and within-direction proportions",
            "exact_test_subset": "failures only; correct dropped before testing",
            "exact_test": (
                "Probability-ordered two-sided Fisher-Freeman-Halton test, "
                "enumerating every table conditional on observed row and column margins."
            ),
            "pairing_note": (
                "The test requested here compares marginal failure-mode distributions. "
                "Matched seeds remain identifiable in the episode JSONL but are not "
                "used as independent failure observations in this contingency test."
            ),
            "inference_note": (
                "A nonsignificant test does not establish equal shapes; sparse failure "
                "rows can have low power."
            ),
        },
        "frozen_taxonomy": {
            "path": str(TAXONOMY_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256_path(TAXONOMY_PATH),
            "categories_in_report_order": list(CATEGORIES),
            "primary_precedence_unchanged": taxonomy["primary_precedence"],
            "display_aliases_only": DISPLAY_ALIASES,
        },
        "derived_episode_view": jsonl_meta,
        "results": model_results,
    }
    report_meta = write_bytes(
        output_dir / "failure_mode_split_report.json", canonical_json_bytes(report)
    )

    manifest = {
        "schema_version": "vla-wam-shared-v3-failure-mode-split-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "analysis_git_head": report["analysis_git_head"],
        "analysis_script": {
            "path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "bytes": Path(__file__).stat().st_size,
            "sha256": sha256_path(Path(__file__)),
        },
        "inputs": [
            item
            for result in model_results
            for item in (result["source_summary"], result["source_evidence_manifest"])
        ]
        + [report["frozen_taxonomy"] | {"bytes": TAXONOMY_PATH.stat().st_size}],
        "outputs": [jsonl_meta, report_meta],
        "integrity": {
            "episode_rows": len(episode_rows),
            "model_cohorts": len(model_results),
            "per_model_direction_denominator": 27,
            "frozen_definitions_modified": False,
        },
    }
    manifest_meta = write_bytes(
        output_dir / "failure_mode_split_manifest.json", canonical_json_bytes(manifest)
    )
    print(json.dumps({"outputs": [jsonl_meta, report_meta, manifest_meta]}, indent=2))


if __name__ == "__main__":
    main()
