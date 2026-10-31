#!/usr/bin/env python3
"""Exact finite-population label-deletion sensitivity for retained H02.

The sealed H02 capped-index decomposition defines U=450(1-E)+Z and exports
27 registered environment-label blocks.  This post-snapshot diagnostic deletes
each registered label once and recomputes the same LEFT, RIGHT, block-balanced
and command-by-layout arithmetic means over the remaining 26 labels.

This is a deterministic sensitivity description of one fixed finite population,
not jackknife inference, a standard error, confidence interval, p-value,
bootstrap, influence-function estimate, independent-scene generalization or a
new estimand.  It launches no model, retrieval, annotation, remote job or
release workflow.
"""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
UPSTREAM_ANALYZER = PACKAGE / "analysis/analyze_h02_entry_status_contributions.py"
UPSTREAM_JSON = PACKAGE / "paper/data/h02_entry_status_contributions.json"
UPSTREAM_CSV = PACKAGE / "paper/data/h02_entry_status_contributions.csv"
DEFAULT_JSON = PACKAGE / "paper/data/h02_label_deletion_sensitivity.json"
DEFAULT_CSV = PACKAGE / "paper/data/h02_label_deletion_sensitivity.csv"
SCHEMA = "wmf-retained-h02-label-deletion-sensitivity-v1"
SEEDS = tuple(range(9400, 9427))
COMPONENTS = (
    "entry_incidence_at_cap_term",
    "zero_masked_recorded_entry_index_term",
    "cap_restricted_waiting_action_index_contrast",
)
SCOPES = (
    ("left", "left"),
    ("right", "right"),
    ("block_balanced", "block_balanced"),
    ("command_by_layout_interaction", "difference_in_differences"),
)
UPSTREAM_IDENTITIES = {
    "analyzer": {
        "path": UPSTREAM_ANALYZER,
        "sha256": "6aa3d0b1f6276ac57b8b3838f2ac22bae5929a5d764f57a5370cb75ba04e8a5e",
        "bytes": 21_753,
        "lines": 483,
    },
    "json": {
        "path": UPSTREAM_JSON,
        "sha256": "9bc4c8ade7cdcea065a7eee53d74059c0098ca79a55ddf9f930f29c9d771d9d9",
        "bytes": 15_060,
        "lines": 386,
    },
    "csv": {
        "path": UPSTREAM_CSV,
        "sha256": "5a48495fb332ca7ec764df36f69e064899797cea562913f1add5840b3923e1ba",
        "bytes": 1_639,
        "lines": 9,
    },
}


def _identity(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "lines": len(raw.splitlines()),
    }


def _load_upstream_module():
    """Authenticate the latest sealed H02 analyzer before executing its code."""
    expected = UPSTREAM_IDENTITIES["analyzer"]
    actual = _identity(expected["path"])
    wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
    if actual != wanted:
        raise ValueError("sealed entry-status analyzer identity changed before import")
    spec = importlib.util.spec_from_file_location(
        "sealed_h02_entry_status_contributions", UPSTREAM_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("sealed entry-status analyzer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream_module()


def load_authenticated_base() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the latest H02 package and recover its capped block rows."""
    observed: dict[str, Any] = {}
    for label, expected in UPSTREAM_IDENTITIES.items():
        actual = _identity(expected["path"])
        wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
        if actual != wanted:
            raise ValueError(f"sealed entry-status {label} identity changed")
        observed[label] = {
            "path": str(expected["path"].relative_to(ROOT)),
            **actual,
        }

    capped, inherited = UPSTREAM.load_authenticated_base()
    ledger = UPSTREAM.analyze_base(capped, inherited)
    if ledger.get("schema_version") != "wmf-retained-h02-entry-status-contributions-v1":
        raise ValueError("sealed entry-status schema changed")
    if UPSTREAM.render_json(ledger) != UPSTREAM_JSON.read_text(encoding="utf-8"):
        raise ValueError("sealed entry-status JSON does not regenerate exactly")
    if UPSTREAM.render_csv(ledger) != UPSTREAM_CSV.read_text(encoding="utf-8"):
        raise ValueError("sealed entry-status CSV does not regenerate exactly")
    provenance = ledger.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("sealed entry-status provenance changed")
    return capped, {
        "sealed_entry_status_files": observed,
        "historical_source_commit": provenance["historical_source_commit"],
        "historical_authenticated_file_count": provenance[
            "historical_authenticated_file_count"
        ],
        "historical_authenticated_bytes": provenance[
            "historical_authenticated_bytes"
        ],
        "historical_authenticated_lines": provenance[
            "historical_authenticated_lines"
        ],
        "upstream_sealed_capped_decomposition_files": provenance[
            "sealed_capped_decomposition_files"
        ],
        "upstream_sealed_matched_entry_files": provenance[
            "upstream_sealed_matched_entry_files"
        ],
        "upstream_sealed_h02_package_files": provenance[
            "upstream_sealed_h02_package_files"
        ],
    }


def _number(row: Mapping[str, Any], field: str) -> float:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must remain numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must remain finite")
    return 0.0 if result == 0.0 else result


def _fraction(value: float | int) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("exact sensitivity value must remain numeric")
    if not math.isfinite(float(value)):
        raise ValueError("exact sensitivity value must remain finite")
    return Fraction(str(value))


def _measure(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact_fraction": f"{value.numerator}/{value.denominator}",
        "decimal": float(value),
    }


def _sign(value: Fraction) -> str:
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def _scope_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[Fraction]]]:
    values: dict[str, dict[str, list[Fraction]]] = {
        public: {component: [] for component in COMPONENTS}
        for public, _ in SCOPES
    }
    for row in rows:
        for public, field_prefix in SCOPES:
            for component in COMPONENTS:
                values[public][component].append(
                    _fraction(_number(row, f"{field_prefix}_{component}"))
                )

        for public in ("left", "right", "block_balanced", "command_by_layout_interaction"):
            parts = values[public]
            if parts[COMPONENTS[0]][-1] + parts[COMPONENTS[1]][-1] != parts[COMPONENTS[2]][-1]:
                raise ValueError("per-label capped-index component identity changed")
        for component in COMPONENTS:
            left = values["left"][component][-1]
            right = values["right"][component][-1]
            if values["block_balanced"][component][-1] != (left + right) / 2:
                raise ValueError("per-label block-balanced identity changed")
            if values["command_by_layout_interaction"][component][-1] != right - left:
                raise ValueError("per-label command-interaction identity changed")
    return values


def _summary_row(
    scope: str,
    component: str,
    values: Sequence[Fraction],
) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError("label-deletion sensitivity requires 27 values per row")
    full_sum = sum(values, Fraction())
    full_mean = full_sum / len(values)
    deletions = []
    for seed, omitted in zip(SEEDS, values):
        remaining_sum = full_sum - omitted
        mean = remaining_sum / (len(values) - 1)
        deletions.append(
            {
                "omitted_seed": seed,
                "remaining_labels": len(values) - 1,
                "omitted_contribution": _measure(omitted),
                "remaining_sum": _measure(remaining_sum),
                "mean": _measure(mean),
                "shift_from_full_mean": _measure(mean - full_mean),
            }
        )

    means = [Fraction(item["mean"]["numerator"], item["mean"]["denominator"]) for item in deletions]
    shifts = [mean - full_mean for mean in means]
    minimum = min(means)
    maximum = max(means)
    maximum_absolute_shift = max(abs(shift) for shift in shifts)
    sign_counts = {
        sign: sum(_sign(mean) == sign for mean in means)
        for sign in ("positive", "zero", "negative")
    }
    return {
        "scope": scope,
        "component": component,
        "full_population": {
            "labels": len(values),
            "sum": _measure(full_sum),
            "mean": _measure(full_mean),
            "sign": _sign(full_mean),
        },
        "leave_one_registered_label_out": {
            "estimates": len(deletions),
            "remaining_labels_per_estimate": len(values) - 1,
            "minimum_mean": _measure(minimum),
            "minimum_mean_omitted_seeds": [
                item["omitted_seed"]
                for item, mean in zip(deletions, means)
                if mean == minimum
            ],
            "maximum_mean": _measure(maximum),
            "maximum_mean_omitted_seeds": [
                item["omitted_seed"]
                for item, mean in zip(deletions, means)
                if mean == maximum
            ],
            "maximum_absolute_shift_from_full_mean": _measure(maximum_absolute_shift),
            "maximum_absolute_shift_omitted_seeds": [
                item["omitted_seed"]
                for item, shift in zip(deletions, shifts)
                if abs(shift) == maximum_absolute_shift
            ],
            "mean_sign_counts": sign_counts,
            "all_means_retain_full_population_sign": sign_counts[_sign(full_mean)] == len(deletions),
        },
        "deletion_estimates": deletions,
    }


def analyze_base(base: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    if base.get("schema_version") != "wmf-retained-h02-capped-index-decomposition-v1":
        raise ValueError("label-deletion analysis requires the sealed capped decomposition")
    population = base.get("population")
    if not isinstance(population, Mapping) or population.get(
        "matched_environment_label_blocks"
    ) != 27 or population.get("seed_labels") != list(SEEDS):
        raise ValueError("label-deletion analysis requires the exact 27-label population")
    rows = base.get("per_block")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != list(SEEDS):
        raise ValueError("label-deletion analysis requires ordered registered labels")

    values = _scope_values(rows)
    summaries = base.get("summaries")
    if not isinstance(summaries, Mapping):
        raise ValueError("capped-decomposition summaries changed")
    upstream_groups = {
        "left": summaries["command_specific_position_mirrored_minus_control"]["left"],
        "right": summaries["command_specific_position_mirrored_minus_control"]["right"],
        "block_balanced": summaries["block_balanced_position_mirrored_minus_control"],
        "command_by_layout_interaction": summaries["command_by_layout_difference_in_differences"],
    }

    sensitivity_rows = []
    for scope, _ in SCOPES:
        for component in COMPONENTS:
            row = _summary_row(scope, component, values[scope][component])
            upstream = upstream_groups[scope][component]
            if row["full_population"]["sum"]["decimal"] != _number(upstream, "sum"):
                raise ValueError("full-population sum changed from capped decomposition")
            if row["full_population"]["mean"]["decimal"] != _number(upstream, "mean"):
                raise ValueError("full-population mean changed from capped decomposition")
            sensitivity_rows.append(row)

    by_key = {(row["scope"], row["component"]): row for row in sensitivity_rows}
    for seed_index in range(len(SEEDS)):
        for scope in ("left", "right", "block_balanced", "command_by_layout_interaction"):
            incidence = by_key[(scope, COMPONENTS[0])]["deletion_estimates"][seed_index]["mean"]
            index_mass = by_key[(scope, COMPONENTS[1])]["deletion_estimates"][seed_index]["mean"]
            total = by_key[(scope, COMPONENTS[2])]["deletion_estimates"][seed_index]["mean"]
            if Fraction(incidence["numerator"], incidence["denominator"]) + Fraction(
                index_mass["numerator"], index_mass["denominator"]
            ) != Fraction(total["numerator"], total["denominator"]):
                raise ValueError("deletion-level capped-index identity changed")

    new_activity = dict(base.get("new_activity", {}))
    gate_state = dict(base.get("gate_state", {}))
    if len(new_activity) != 13 or any(type(value) is not int or value != 0 for value in new_activity.values()):
        raise ValueError("label-deletion analysis must retain 13 zero activity counters")
    if len(gate_state) != 9 or any(type(value) is not bool or value is not False for value in gate_state.values()):
        raise ValueError("label-deletion analysis must retain nine closed gates")

    return {
        "schema_version": SCHEMA,
        "status": "post_snapshot_retained_h02_exact_label_deletion_sensitivity_no_stage_advance",
        "provenance": dict(provenance),
        "configuration": dict(base["configuration"]),
        "population": {
            "base_behavioral_episodes": 108,
            "matched_environment_label_blocks": 27,
            "conditions_per_block": 4,
            "seed_labels": list(SEEDS),
            "sensitivity_rows": len(sensitivity_rows),
            "deletion_estimates_per_row": len(SEEDS),
            "total_deletion_means": len(sensitivity_rows) * len(SEEDS),
            "new_cohort_created": False,
        },
        "identity": dict(base["identity"]),
        "scope_order": [scope for scope, _ in SCOPES],
        "component_order": list(COMPONENTS),
        "sensitivity_rows": sensitivity_rows,
        "definitions": {
            "finite_population_deletion": (
                "Delete each of the 27 registered environment labels once and recompute the unchanged arithmetic mean over the remaining 26 labels."
            ),
            "sign_retention": (
                "All 27 deletion means share the full-population arithmetic sign. This says no single registered label changes that sign; it is not a robustness proof for unobserved labels."
            ),
            "exact_arithmetic": (
                "Every sum, mean and shift is serialized as an exact rational number plus its decimal rendering; U=450(1-E)+Z closes for every scope and deletion."
            ),
            "inference": (
                "Deterministic post-result sensitivity for this fixed 27-label population only; not jackknife inference, a standard error, confidence interval, p-value, bootstrap, influence function, resampling distribution, equivalence test or registered analysis."
            ),
            "independence": (
                "Registered environment labels are matched labels on one scene and its reflection, not independent physical scenes or independent model-noise draws; deletion results do not establish generalization."
            ),
        },
        "new_activity": new_activity,
        "gate_state": gate_state,
        "claim_boundary": {
            "retained_post_result_finite_population_sensitivity_only": True,
            "exact_leave_one_registered_label_out_reported": True,
            "new_underlying_behavioral_evidence": False,
            "new_cohort_or_registered_estimand": False,
            "jackknife_inference_or_standard_error_reported": False,
            "confidence_interval_or_p_value_reported": False,
            "bootstrap_or_resampling_distribution_reported": False,
            "influence_function_estimated": False,
            "robustness_to_unobserved_labels_established": False,
            "independent_scene_or_model_noise_generalization": False,
            "equivalence_or_causal_effect_established": False,
            "physical_time_or_latency_estimated": False,
            "model_based_survival_curve_or_hazard_estimated": False,
            "forecast_accuracy_or_generated_execution_comparison": False,
            "official_conditional_D1_evidence": False,
            "default_DreamZero_evidence": False,
            "A3_A9_completed": False,
            "new_inference_episode_request_or_label": False,
            "current_GM_PVC_state_reconciled": False,
            "changes_232_cell_core_bound": False,
            "advances_any_gate": False,
        },
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


CSV_FIELDS = (
    "scope",
    "component",
    "omitted_seed",
    "remaining_labels",
    "full_sum_exact",
    "full_mean_exact",
    "omitted_contribution_exact",
    "remaining_sum_exact",
    "leave_one_mean_exact",
    "shift_from_full_mean_exact",
    "full_mean_decimal",
    "leave_one_mean_decimal",
    "shift_from_full_mean_decimal",
)


def render_csv(report: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in report["sensitivity_rows"]:
        for deletion in row["deletion_estimates"]:
            writer.writerow(
                {
                    "scope": row["scope"],
                    "component": row["component"],
                    "omitted_seed": deletion["omitted_seed"],
                    "remaining_labels": deletion["remaining_labels"],
                    "full_sum_exact": row["full_population"]["sum"]["exact_fraction"],
                    "full_mean_exact": row["full_population"]["mean"]["exact_fraction"],
                    "omitted_contribution_exact": deletion["omitted_contribution"]["exact_fraction"],
                    "remaining_sum_exact": deletion["remaining_sum"]["exact_fraction"],
                    "leave_one_mean_exact": deletion["mean"]["exact_fraction"],
                    "shift_from_full_mean_exact": deletion["shift_from_full_mean"]["exact_fraction"],
                    "full_mean_decimal": row["full_population"]["mean"]["decimal"],
                    "leave_one_mean_decimal": deletion["mean"]["decimal"],
                    "shift_from_full_mean_decimal": deletion["shift_from_full_mean"]["decimal"],
                }
            )
    return output.getvalue()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args(argv)
    base, provenance = load_authenticated_base()
    report = analyze_base(base, provenance)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(render_json(report), encoding="utf-8")
    args.csv_output.write_text(render_csv(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "json": str(args.json_output),
                "csv": str(args.csv_output),
                "sensitivity_rows": report["population"]["sensitivity_rows"],
                "total_deletion_means": report["population"]["total_deletion_means"],
                "new_activity": report["new_activity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
