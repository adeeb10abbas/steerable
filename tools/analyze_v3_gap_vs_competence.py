#!/usr/bin/env python3
"""Analyze directional success gap versus overall DROID competence.

This is a retrospective, model-separated analysis of the five completed V3
Phase-A DROID cohorts.  It derives every value from the committed 54-episode
summaries, makes the success-rate boundary on an observable directional gap
explicit, and uses exact permutation tests for the descriptive rank and
linear correlations (five checkpoints, so all 5! labelings are enumerable).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "artifacts/vla_wam_shared_v3/results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/vla_wam_shared_v3/analysis/mechanism"

SOURCES = (
    ("pi05_current_stack_droid_phase_a_summary.json", "pi05_current_stack_droid", "π0.5"),
    ("groot_n17_droid_phase_a_summary.json", "groot_n17_droid_vla", "GR00T N1.7"),
    ("cosmos3_edge_policy_droid_phase_a_summary.json", "cosmos3_edge_policy_droid", "Cosmos3 Edge"),
    ("cosmos3_nano_policy_droid_phase_a_summary.json", "cosmos3_nano_policy_droid", "Cosmos3 Nano"),
    ("dreamzero_droid_action_cfg_phase_a_summary.json", "dreamzero_droid_action_cfg", "DreamZero"),
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()


def trials(direction: dict[str, Any]) -> int:
    value = direction.get("valid_denominator", direction.get("trials"))
    if type(value) is not int:
        raise ValueError("directional trial denominator is missing")
    return value


def ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Return average ranks, using 1-based ranks for tied values."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            result[order[position]] = average
        start = end
    return tuple(result)


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation inputs must have equal length >= 2")
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_ss = sum((a - x_mean) ** 2 for a in x)
    y_ss = sum((b - y_mean) ** 2 for b in y)
    if x_ss == 0.0 or y_ss == 0.0:
        raise ValueError("correlation is undefined for a constant input")
    return numerator / math.sqrt(x_ss * y_ss)


def exact_permutation_correlation(x: Sequence[float], y: Sequence[float], *, rank: bool) -> dict[str, Any]:
    x_values = ranks(x) if rank else tuple(x)
    y_values = ranks(y) if rank else tuple(y)
    observed = pearson(x_values, y_values)
    permuted = [pearson(x_values, candidate) for candidate in itertools.permutations(y_values)]
    extreme = sum(abs(value) >= abs(observed) - 1e-15 for value in permuted)
    return {
        "coefficient": observed,
        "two_sided_exact_permutation_p": extreme / len(permuted),
        "permutations_enumerated": len(permuted),
    }


def compile_row(filename: str, expected_model_id: str, display_name: str) -> dict[str, Any]:
    path = RESULTS_DIR / filename
    data = json.loads(path.read_text())
    if data.get("model_id") != expected_model_id:
        raise ValueError(f"unexpected model_id in {path}")
    if data.get("arena") != "droid_robolab":
        raise ValueError(f"unexpected arena in {path}")
    if len(data.get("cells", [])) != 54 or len(data.get("pairs", [])) != 27:
        raise ValueError(f"{display_name}: expected 54 cells and 27 matched pairs")

    left = data["directional"]["left"]
    right = data["directional"]["right"]
    left_n = trials(left)
    right_n = trials(right)
    left_successes = int(left["successes"])
    right_successes = int(right["successes"])
    if (left_n, right_n) != (27, 27):
        raise ValueError(f"{display_name}: expected 27 episodes per direction")

    left_rate = left_successes / left_n
    right_rate = right_successes / right_n
    overall_rate = (left_successes + right_successes) / (left_n + right_n)
    gap = right_rate - left_rate
    maximum_gap_magnitude = 2.0 * min(overall_rate, 1.0 - overall_rate)
    if abs(gap) > maximum_gap_magnitude + 1e-12:
        raise ValueError(f"{display_name}: directional gap violates its success-rate envelope")

    return {
        "model_id": expected_model_id,
        "display_name": display_name,
        "left_successes": left_successes,
        "left_trials": left_n,
        "left_success_rate": left_rate,
        "right_successes": right_successes,
        "right_trials": right_n,
        "right_success_rate": right_rate,
        "overall_successes": left_successes + right_successes,
        "overall_trials": left_n + right_n,
        "overall_success_rate": overall_rate,
        "directional_gap_right_minus_left": gap,
        "absolute_directional_gap": abs(gap),
        "maximum_observable_gap_magnitude_at_this_success_rate": maximum_gap_magnitude,
        "fraction_of_available_gap_magnitude": (
            abs(gap) / maximum_gap_magnitude if maximum_gap_magnitude else None
        ),
        "source_summary": {
            "path": str(path.relative_to(REPO_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": sha256_path(path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("output directory must be inside the repository") from exc
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [compile_row(*source) for source in SOURCES]
    competence = [row["overall_success_rate"] for row in rows]
    signed_gap = [row["directional_gap_right_minus_left"] for row in rows]
    absolute_gap = [row["absolute_directional_gap"] for row in rows]

    report = {
        "schema_version": "vla-wam-shared-v3-gap-versus-competence-report-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "status": "complete_retrospective_analysis_no_new_inference",
        "analysis_date": "2026-08-06",
        "analysis_git_head": git_head(),
        "scope": {
            "arena": "droid_robolab",
            "cohorts": "five separate V3 Phase-A expanded cohorts",
            "checkpoint_count": 5,
            "episodes_per_checkpoint": 54,
            "matched_pairs_per_checkpoint": 27,
            "pooling": "No checkpoint cohorts are pooled.",
            "new_model_compute": False,
        },
        "estimands": {
            "competence": "(LEFT successes + RIGHT successes) / 54",
            "signed_directional_gap": "RIGHT success rate minus LEFT success rate",
            "absolute_directional_gap": "absolute value of the signed directional gap",
            "mechanical_gap_envelope": "2 * min(overall success rate, 1 - overall success rate)",
        },
        "results": rows,
        "descriptive_associations": {
            "competence_vs_signed_gap": {
                "pearson": exact_permutation_correlation(competence, signed_gap, rank=False),
                "spearman": exact_permutation_correlation(competence, signed_gap, rank=True),
            },
            "competence_vs_absolute_gap": {
                "pearson": exact_permutation_correlation(competence, absolute_gap, rank=False),
                "spearman": exact_permutation_correlation(competence, absolute_gap, rank=True),
            },
            "multiplicity_note": "Descriptive five-checkpoint tests; no multiplicity adjustment and no population-level model claim.",
        },
        "interpretation": {
            "monotonic_difficulty_competence_account_supported": False,
            "statement": (
                "The five checkpoints do not show a monotonic relation between overall success and either the signed or absolute directional gap. "
                "Near-floor and near-ceiling competence mechanically compress the maximum observable binary gap; intermediate competence creates opportunity for a large gap but does not determine its sign or mechanism."
            ),
            "next_experiment": (
                "Retain the lateral-position dose-response and DreamZero mirror experiments: they manipulate geometry directly and can distinguish a geometric mechanism from this descriptive success-rate envelope."
            ),
        },
    }

    output_path = output_dir / "gap_vs_competence_report.json"
    output_path.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "path": str(output_path.relative_to(REPO_ROOT)),
        "sha256": sha256_path(output_path),
        "bytes": output_path.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
