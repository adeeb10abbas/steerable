#!/usr/bin/env python3
"""Partition the retained H02 capped-index contrast by entry-status pair.

The sealed capped-index decomposition defines, for administrative cap C=450,
entry indicator E, and zero-masked recorded entry index Z,

    U = C(1-E) + Z,       delta U = -C delta E + delta Z.

This post-snapshot supplement authenticates and exactly regenerates that sealed
decomposition before partitioning its already-exported LEFT and RIGHT component
sums across the four control/reflected entry-status pairs.  The partition is an
accounting identity for the same capped composite, not a conditional effect,
entrant-only timing estimand, transition, principal stratum or causal analysis.
It runs no model, retrieval, annotation, remote job or release workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
UPSTREAM_ANALYZER = PACKAGE / "analysis/analyze_h02_capped_index_decomposition.py"
UPSTREAM_JSON = PACKAGE / "paper/data/h02_capped_index_decomposition.json"
UPSTREAM_CSV = PACKAGE / "paper/data/h02_capped_index_decomposition.csv"
DEFAULT_JSON = PACKAGE / "paper/data/h02_entry_status_contributions.json"
DEFAULT_CSV = PACKAGE / "paper/data/h02_entry_status_contributions.csv"
SCHEMA = "wmf-retained-h02-entry-status-contributions-v1"
CAP = 450.0
SEEDS = tuple(range(9400, 9427))
RELATIONS = ("left", "right")
STATUS_PAIRS = (
    "control_no_entry__reflected_no_entry",
    "control_no_entry__reflected_entry",
    "control_entry__reflected_no_entry",
    "control_entry__reflected_entry",
)
STATUS_BITS = {
    "control_no_entry__reflected_no_entry": (0, 0),
    "control_no_entry__reflected_entry": (0, 1),
    "control_entry__reflected_no_entry": (1, 0),
    "control_entry__reflected_entry": (1, 1),
}
COMPONENTS = (
    "entry_incidence_at_cap_term",
    "zero_masked_recorded_entry_index_term",
    "cap_restricted_waiting_action_index_contrast",
)
UPSTREAM_IDENTITIES = {
    "analyzer": {
        "path": UPSTREAM_ANALYZER,
        "sha256": "bbfe43b7dde0ec4f30e7a7f2fb40c11389c8ced5175d6196dae512f7c363c11a",
        "bytes": 20_089,
        "lines": 469,
    },
    "json": {
        "path": UPSTREAM_JSON,
        "sha256": "33d66ed365d2fe330f2a4988a578a6fdbdc83c335184fb9098aeaa0f0c676f09",
        "bytes": 37_325,
        "lines": 808,
    },
    "csv": {
        "path": UPSTREAM_CSV,
        "sha256": "11fd4944b7cb5f9d3509bda6d9c0d1764d5d077461a05b027a13f980b16e3cb4",
        "bytes": 4_581,
        "lines": 28,
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
    """Authenticate the upstream analyzer before executing any of its code."""
    expected = UPSTREAM_IDENTITIES["analyzer"]
    actual = _identity(expected["path"])
    wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
    if actual != wanted:
        raise ValueError("sealed capped-decomposition analyzer identity changed before import")
    spec = importlib.util.spec_from_file_location(
        "sealed_h02_capped_index_decomposition", UPSTREAM_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("sealed capped-decomposition analyzer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = _load_upstream_module()


def load_authenticated_base() -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate and byte-regenerate the immutable capped decomposition."""
    observed: dict[str, Any] = {}
    for label, expected in UPSTREAM_IDENTITIES.items():
        actual = _identity(expected["path"])
        wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
        if actual != wanted:
            raise ValueError(f"sealed capped-decomposition {label} identity changed")
        observed[label] = {
            "path": str(expected["path"].relative_to(ROOT)),
            **actual,
        }

    matched, inherited = UPSTREAM.load_authenticated_base()
    report = UPSTREAM.analyze_base(matched, inherited)
    if report.get("schema_version") != "wmf-retained-h02-capped-index-decomposition-v1":
        raise ValueError("sealed capped-decomposition schema changed")
    if UPSTREAM.render_json(report) != UPSTREAM_JSON.read_text(encoding="utf-8"):
        raise ValueError("sealed capped-decomposition JSON does not regenerate exactly")
    if UPSTREAM.render_csv(report) != UPSTREAM_CSV.read_text(encoding="utf-8"):
        raise ValueError("sealed capped-decomposition CSV does not regenerate exactly")
    upstream_provenance = report.get("provenance")
    if not isinstance(upstream_provenance, Mapping):
        raise ValueError("sealed capped-decomposition provenance changed")
    return report, {
        "sealed_capped_decomposition_files": observed,
        "historical_source_commit": upstream_provenance["historical_source_commit"],
        "historical_authenticated_file_count": upstream_provenance[
            "historical_authenticated_file_count"
        ],
        "historical_authenticated_bytes": upstream_provenance[
            "historical_authenticated_bytes"
        ],
        "historical_authenticated_lines": upstream_provenance[
            "historical_authenticated_lines"
        ],
        "upstream_sealed_matched_entry_files": upstream_provenance[
            "sealed_matched_entry_files"
        ],
        "upstream_sealed_h02_package_files": upstream_provenance[
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
    return result


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def _expected_incidence(status: str) -> float:
    control, reflected = STATUS_BITS[status]
    return -CAP * (reflected - control)


def _validate_status_component_semantics(
    status: str, incidence: float, zero_masked: float, total: float
) -> None:
    control, reflected = STATUS_BITS[status]
    if incidence != _expected_incidence(status):
        raise ValueError("entry-status incidence term changed")
    if incidence + zero_masked != total:
        raise ValueError("entry-status row decomposition does not close")
    if (control, reflected) == (0, 0) and (zero_masked != 0.0 or total != 0.0):
        raise ValueError("neither-entry row must retain zero algebraic mass")
    if (control, reflected) == (0, 1) and not 0.0 <= zero_masked <= CAP:
        raise ValueError("no-entry/entry zero-masked term is outside cap bounds")
    if (control, reflected) == (1, 0) and not -CAP <= zero_masked <= 0.0:
        raise ValueError("entry/no-entry zero-masked term is outside cap bounds")
    if (control, reflected) == (1, 1) and not -CAP <= zero_masked <= CAP:
        raise ValueError("both-entry zero-masked contrast is outside cap bounds")


def analyze_base(base: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    if base.get("schema_version") != "wmf-retained-h02-capped-index-decomposition-v1":
        raise ValueError("contribution ledger requires the sealed capped-decomposition schema")
    identity = base.get("identity")
    if not isinstance(identity, Mapping) or identity.get("cap_action_index") != CAP:
        raise ValueError("capped-decomposition cap identity changed")
    if identity.get("cell_identity") != "U = C * (1 - E) + Z" or identity.get(
        "contrast_identity"
    ) != "delta U = -C * delta E + delta Z":
        raise ValueError("capped-decomposition algebra identity changed")

    population = base.get("population")
    if not isinstance(population, Mapping):
        raise ValueError("capped-decomposition population changed")
    if (
        population.get("base_behavioral_episodes") != 108
        or population.get("matched_environment_label_blocks") != 27
        or population.get("conditions_per_block") != 4
        or population.get("seed_labels") != list(SEEDS)
        or population.get("new_cohort_created") is not False
    ):
        raise ValueError("contribution ledger requires the exact retained H02 population")
    rows = base.get("per_block")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != list(SEEDS):
        raise ValueError("contribution ledger requires the ordered H02 seed-label population")

    grouped: dict[str, dict[str, list[dict[str, float]]]] = {
        relation: {status: [] for status in STATUS_PAIRS} for relation in RELATIONS
    }
    derived_counts = {
        relation: {status: 0 for status in STATUS_PAIRS} for relation in RELATIONS
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("capped-decomposition per-block row changed")
        for relation in RELATIONS:
            status = row.get(f"{relation}_entry_status_pair")
            if status not in STATUS_BITS:
                raise ValueError("unknown cross-layout entry-status pair")
            component = {
                name: _clean_zero(_number(row, f"{relation}_{name}"))
                for name in COMPONENTS
            }
            _validate_status_component_semantics(
                status,
                component["entry_incidence_at_cap_term"],
                component["zero_masked_recorded_entry_index_term"],
                component["cap_restricted_waiting_action_index_contrast"],
            )
            grouped[relation][status].append(component)
            derived_counts[relation][status] += 1

    published_counts = base.get("cross_layout_entry_status_pair_counts")
    if derived_counts != published_counts:
        raise ValueError("entry-status partition counts do not close upstream")

    ledger: list[dict[str, Any]] = []
    relation_sums: dict[str, dict[str, float]] = {}
    for relation in RELATIONS:
        relation_sums[relation] = {name: 0.0 for name in COMPONENTS}
        for status in STATUS_PAIRS:
            items = grouped[relation][status]
            count = len(items)
            sums = {
                name: _clean_zero(sum(item[name] for item in items))
                for name in COMPONENTS
            }
            if sums["entry_incidence_at_cap_term"] != _expected_incidence(status) * count:
                raise ValueError("status-level incidence sum does not close")
            if sums["entry_incidence_at_cap_term"] + sums[
                "zero_masked_recorded_entry_index_term"
            ] != sums["cap_restricted_waiting_action_index_contrast"]:
                raise ValueError("status-level capped-index decomposition does not close")
            for name in COMPONENTS:
                relation_sums[relation][name] += sums[name]
            control, reflected = STATUS_BITS[status]
            ledger.append(
                {
                    "relation": relation,
                    "entry_status_pair": status,
                    "control_entry_observed": bool(control),
                    "reflected_entry_observed": bool(reflected),
                    "matched_blocks": count,
                    "empty_stratum": count == 0,
                    "component_sums": sums,
                    "conditional_mean": (
                        None
                        if count == 0
                        else {name: sums[name] / count for name in COMPONENTS}
                    ),
                    "contribution_to_command_mean": {
                        name: sums[name] / len(SEEDS) for name in COMPONENTS
                    },
                }
            )

    summaries = base.get("summaries")
    if not isinstance(summaries, Mapping):
        raise ValueError("capped-decomposition summaries changed")
    command = summaries.get("command_specific_position_mirrored_minus_control")
    if not isinstance(command, Mapping):
        raise ValueError("capped-decomposition command summaries changed")
    for relation in RELATIONS:
        if sum(derived_counts[relation].values()) != len(SEEDS):
            raise ValueError("entry-status counts do not cover every registered label")
        for name in COMPONENTS:
            if relation_sums[relation][name] != _number(command[relation][name], "sum"):
                raise ValueError("entry-status component partition does not close command sum")
            if relation_sums[relation][name] / len(SEEDS) != _number(
                command[relation][name], "mean"
            ):
                raise ValueError("entry-status component partition does not close command mean")

    balanced = {
        name: _clean_zero(0.5 * (relation_sums["left"][name] + relation_sums["right"][name]))
        for name in COMPONENTS
    }
    interaction = {
        name: _clean_zero(relation_sums["right"][name] - relation_sums["left"][name])
        for name in COMPONENTS
    }
    upstream_balanced = summaries.get("block_balanced_position_mirrored_minus_control")
    upstream_interaction = summaries.get("command_by_layout_difference_in_differences")
    if not isinstance(upstream_balanced, Mapping) or not isinstance(
        upstream_interaction, Mapping
    ):
        raise ValueError("capped-decomposition aggregate summaries changed")
    for name in COMPONENTS:
        if balanced[name] != _number(upstream_balanced[name], "sum"):
            raise ValueError("entry-status partition does not close block-balanced sum")
        if interaction[name] != _number(upstream_interaction[name], "sum"):
            raise ValueError("entry-status partition does not close interaction sum")
    for values in (*relation_sums.values(), balanced, interaction):
        if values["entry_incidence_at_cap_term"] + values[
            "zero_masked_recorded_entry_index_term"
        ] != values["cap_restricted_waiting_action_index_contrast"]:
            raise ValueError("contribution-ledger aggregate decomposition does not close")

    new_activity = dict(base.get("new_activity", {}))
    gate_state = dict(base.get("gate_state", {}))
    if len(new_activity) != 13 or any(
        type(value) is not int or value != 0 for value in new_activity.values()
    ):
        raise ValueError("contribution ledger must retain 13 zero activity counters")
    if len(gate_state) != 9 or any(
        type(value) is not bool or value is not False for value in gate_state.values()
    ):
        raise ValueError("contribution ledger must retain nine closed gates")

    return {
        "schema_version": SCHEMA,
        "status": "post_snapshot_retained_h02_entry_status_contribution_ledger_no_stage_advance",
        "provenance": dict(provenance),
        "configuration": dict(base["configuration"]),
        "population": {
            "base_behavioral_episodes": 108,
            "matched_environment_label_blocks": 27,
            "conditions_per_block": 4,
            "command_label_comparisons": 54,
            "ledger_rows": 8,
            "nonempty_ledger_rows": sum(row["matched_blocks"] > 0 for row in ledger),
            "seed_labels": list(SEEDS),
            "new_cohort_created": False,
        },
        "identity": dict(identity),
        "entry_status_contribution_ledger": ledger,
        "closures": {
            "command_specific_component_sums": relation_sums,
            "block_balanced_component_sums": balanced,
            "command_by_layout_interaction_component_sums": interaction,
        },
        "definitions": {
            "ledger_scope": (
                "An exact accounting partition of the already-reported command-specific capped-index composite U sums across observed control/reflected entry-status pairs; it does not change the estimand."
            ),
            "conditional_mean": (
                "For a nonempty row, component sum divided by the number of matched labels in that outcome-defined category; null for an empty category. It is descriptive arithmetic, not an independently identified conditional effect or entrant-only timing estimand."
            ),
            "empty_stratum": (
                "An empty category has matched_blocks zero, algebraic empty sums zero and conditional_mean null; those zero sums are not observed zero effects."
            ),
            "contribution_to_command_mean": (
                "A category component sum divided by all 27 registered labels, so categories add exactly to the existing command-specific mean rather than defining a new conditional estimand."
            ),
            "entry_status_pairs": (
                "Outcome-defined comparison categories across different layouts on a matched registered label, not temporal transitions, within-episode conversions, principal strata, mediators, counterfactual changes or causal effects."
            ),
            "inference": (
                "Post-result descriptive algebra only; no p-value, confidence interval, multiplicity-adjusted test, survival curve, hazard, censoring-independence, physical-time, equivalence, generalization or causal claim."
            ),
        },
        "new_activity": new_activity,
        "gate_state": gate_state,
        "claim_boundary": {
            "retained_post_result_algebra_only": True,
            "exact_status_pair_contribution_ledger_reported": True,
            "new_underlying_behavioral_evidence": False,
            "registered_estimand": False,
            "status_pair_conditioned_effect_identified": False,
            "principal_stratum_or_mediator_identified": False,
            "causal_mediation_or_effect_decomposition_claimed": False,
            "entrant_only_timing_effect_identified": False,
            "A3_A9_completed": False,
            "default_DreamZero_evidence": False,
            "official_conditional_D1_evidence": False,
            "entry_is_behavioral_success": False,
            "equivalence_or_causal_effect_established": False,
            "raw_XYZ_recovered_or_rederived": False,
            "physical_time_or_latency_estimated": False,
            "model_based_survival_curve_or_hazard_estimated": False,
            "post_cap_or_uncensored_event_time_inferred": False,
            "censoring_independence_claimed": False,
            "independent_scene_or_model_noise_generalization": False,
            "forecast_accuracy_or_generated_execution_comparison": False,
            "new_inference_episode_request_or_label": False,
            "current_GM_PVC_state_reconciled": False,
            "changes_232_cell_core_bound": False,
            "advances_any_gate": False,
        },
    }


def render_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


CSV_FIELDS = (
    "relation",
    "entry_status_pair",
    "control_entry_observed",
    "reflected_entry_observed",
    "matched_blocks",
    "empty_stratum",
    *(f"{name}_sum" for name in COMPONENTS),
    *(f"{name}_conditional_mean" for name in COMPONENTS),
    *(f"{name}_contribution_to_command_mean" for name in COMPONENTS),
)


def render_csv(report: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for item in report["entry_status_contribution_ledger"]:
        conditional = item["conditional_mean"]
        row = {
            "relation": item["relation"],
            "entry_status_pair": item["entry_status_pair"],
            "control_entry_observed": item["control_entry_observed"],
            "reflected_entry_observed": item["reflected_entry_observed"],
            "matched_blocks": item["matched_blocks"],
            "empty_stratum": item["empty_stratum"],
        }
        for name in COMPONENTS:
            row[f"{name}_sum"] = item["component_sums"][name]
            row[f"{name}_conditional_mean"] = (
                "" if conditional is None else conditional[name]
            )
            row[f"{name}_contribution_to_command_mean"] = item[
                "contribution_to_command_mean"
            ][name]
        writer.writerow(row)
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
                "ledger_rows": report["population"]["ledger_rows"],
                "nonempty_ledger_rows": report["population"]["nonempty_ledger_rows"],
                "closures": report["closures"],
                "new_activity": report["new_activity"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
