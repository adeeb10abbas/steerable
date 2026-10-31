#!/usr/bin/env python3
"""Exactly decompose the retained H02 capped first-entry-index contrast.

For entry indicator E, recorded entry index T, and administrative cap C=450,
define zero-masked recorded-index term Z=T when E=1 and Z=0 when E=0.  The
off-entry zero is bookkeeping, not an imputed event or index.  The
already-published capped variable is U=C(1-E)+Z.  Therefore every
reflected-minus-control contrast obeys

    delta U = -C delta E + delta Z.

This post-snapshot supplement authenticates and exactly regenerates the sealed
matched-entry supplement before applying that identity to its 27 exported
blocks.  It runs no model, retrieval, annotation, remote job, release workflow
or new statistical inference.
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
import statistics
from typing import Any, Mapping, Sequence


PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
MATCHED_ANALYZER = PACKAGE / "analysis/analyze_h02_matched_entry.py"
MATCHED_JSON = PACKAGE / "paper/data/h02_matched_entry.json"
MATCHED_CSV = PACKAGE / "paper/data/h02_matched_entry.csv"
DEFAULT_JSON = PACKAGE / "paper/data/h02_capped_index_decomposition.json"
DEFAULT_CSV = PACKAGE / "paper/data/h02_capped_index_decomposition.csv"
SCHEMA = "wmf-retained-h02-capped-index-decomposition-v1"
CAP = 450.0
SEEDS = tuple(range(9400, 9427))
RELATIONS = ("left", "right")
COMPONENTS = (
    "entry_incidence_at_cap_term",
    "zero_masked_recorded_entry_index_term",
    "cap_restricted_waiting_action_index_contrast",
)
MATCHED_IDENTITIES = {
    "analyzer": {
        "path": MATCHED_ANALYZER,
        "sha256": "25160e55c9f0ba016728e4c1f5b4bc1a32322921eade121f27cf1e0719d47530",
        "bytes": 19_729,
        "lines": 446,
    },
    "json": {
        "path": MATCHED_JSON,
        "sha256": "95ad90a69aa6e0b5b2a135de6eb93a6a8d6290ec792bb19ba9d05e9864f96cb0",
        "bytes": 48_292,
        "lines": 947,
    },
    "csv": {
        "path": MATCHED_CSV,
        "sha256": "1af3d5d2276f116bccda0a9ace3cbc657222bd98775be9ba39503a964346270a",
        "bytes": 3_934,
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


def _load_matched_module():
    """Authenticate the upstream analyzer before executing any of its code."""
    expected = MATCHED_IDENTITIES["analyzer"]
    actual = _identity(expected["path"])
    wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
    if actual != wanted:
        raise ValueError("sealed matched-entry analyzer identity changed before import")
    spec = importlib.util.spec_from_file_location(
        "sealed_h02_matched_entry", MATCHED_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("sealed matched-entry analyzer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MATCHED = _load_matched_module()


def load_authenticated_base() -> tuple[dict[str, Any], dict[str, Any]]:
    observed: dict[str, Any] = {}
    for label, expected in MATCHED_IDENTITIES.items():
        actual = _identity(expected["path"])
        wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
        if actual != wanted:
            raise ValueError(f"sealed matched-entry {label} identity changed")
        observed[label] = {
            "path": str(expected["path"].relative_to(ROOT)),
            **actual,
        }

    sealed_h02, inherited = MATCHED.load_authenticated_base()
    report = MATCHED.analyze_base(sealed_h02, inherited)
    if report.get("schema_version") != "wmf-retained-h02-matched-entry-supplement-v1":
        raise ValueError("sealed matched-entry schema changed")
    if MATCHED.render_json(report) != MATCHED_JSON.read_text(encoding="utf-8"):
        raise ValueError("sealed matched-entry JSON does not regenerate exactly")
    if MATCHED.render_csv(report) != MATCHED_CSV.read_text(encoding="utf-8"):
        raise ValueError("sealed matched-entry CSV does not regenerate exactly")
    return report, {
        "sealed_matched_entry_files": observed,
        "historical_source_commit": report["provenance"]["historical_source_commit"],
        "historical_authenticated_file_count": report["provenance"][
            "historical_authenticated_file_count"
        ],
        "historical_authenticated_bytes": report["provenance"][
            "historical_authenticated_bytes"
        ],
        "historical_authenticated_lines": report["provenance"][
            "historical_authenticated_lines"
        ],
        "upstream_sealed_h02_package_files": report["provenance"][
            "sealed_package_files"
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
    """Canonicalize IEEE signed zero without changing any arithmetic value."""
    return 0.0 if value == 0.0 else value


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("decomposition values must be finite and nonempty")
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    return {
        "n": len(values),
        "sum": sum(values),
        "mean": sum(values) / len(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "sign_counts": {
            "positive": positive,
            "zero": len(values) - positive - negative,
            "negative": negative,
        },
    }


def _status_pair(control: float, reflected: float) -> str:
    if control not in (0.0, 1.0) or reflected not in (0.0, 1.0):
        raise ValueError("entry indicators must remain binary")
    labels = {
        (0.0, 0.0): "control_no_entry__reflected_no_entry",
        (0.0, 1.0): "control_no_entry__reflected_entry",
        (1.0, 0.0): "control_entry__reflected_no_entry",
        (1.0, 1.0): "control_entry__reflected_entry",
    }
    return labels[(control, reflected)]


def analyze_base(base: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    if base.get("schema_version") != "wmf-retained-h02-matched-entry-supplement-v1":
        raise ValueError("decomposition requires the sealed matched-entry schema")
    population = base.get("population")
    if not isinstance(population, Mapping) or population.get(
        "matched_environment_label_blocks"
    ) != 27:
        raise ValueError("decomposition requires exactly 27 matched blocks")
    rows = base.get("per_block")
    if not isinstance(rows, list) or [row.get("seed") for row in rows] != list(SEEDS):
        raise ValueError("decomposition requires the ordered H02 seed-label population")

    values: dict[str, dict[str, list[float]]] = {
        name: {component: [] for component in COMPONENTS}
        for name in (*RELATIONS, "block_balanced", "difference_in_differences")
    }
    status_pairs = {
        relation: {
            "control_no_entry__reflected_no_entry": 0,
            "control_no_entry__reflected_entry": 0,
            "control_entry__reflected_no_entry": 0,
            "control_entry__reflected_entry": 0,
        }
        for relation in RELATIONS
    }
    per_block: list[dict[str, Any]] = []

    for row in rows:
        block: dict[str, Any] = {"seed": row["seed"]}
        relation_components: dict[str, dict[str, float]] = {}
        entry_deltas: dict[str, float] = {}
        for relation in RELATIONS:
            control_entry = _number(row, f"control_{relation}_entry_observed")
            reflected_entry = _number(
                row, f"position_mirrored_{relation}_entry_observed"
            )
            control_index = _number(
                row, f"control_{relation}_cap_restricted_waiting_action_index"
            )
            reflected_index = _number(
                row,
                f"position_mirrored_{relation}_cap_restricted_waiting_action_index",
            )
            status_pair = _status_pair(control_entry, reflected_entry)
            status_pairs[relation][status_pair] += 1
            entry_delta = reflected_entry - control_entry
            entry_deltas[relation] = entry_delta
            if entry_delta != _number(
                row, f"entry_observed_position_mirrored_minus_control_{relation}"
            ):
                raise ValueError("upstream matched-entry indicator contrast changed")
            for indicator, index in (
                (control_entry, control_index),
                (reflected_entry, reflected_index),
            ):
                if not 0.0 <= index <= CAP:
                    raise ValueError("cap-restricted indices must remain in [0,450]")
                if indicator == 0.0 and index != CAP:
                    raise ValueError("no-entry rows must retain the administrative cap")

            incidence = _clean_zero(-CAP * (reflected_entry - control_entry))
            reflected_mass = reflected_index if reflected_entry == 1.0 else 0.0
            control_mass = control_index if control_entry == 1.0 else 0.0
            index_mass = _clean_zero(reflected_mass - control_mass)
            total = _clean_zero(reflected_index - control_index)
            if incidence + index_mass != total:
                raise ValueError("row-level capped-index decomposition does not close")
            upstream_total = _number(
                row,
                "cap_restricted_waiting_action_index_"
                f"position_mirrored_minus_control_{relation}",
            )
            if upstream_total != total:
                raise ValueError("upstream matched-entry contrast identity changed")

            components = {
                "entry_incidence_at_cap_term": incidence,
                "zero_masked_recorded_entry_index_term": index_mass,
                "cap_restricted_waiting_action_index_contrast": total,
            }
            relation_components[relation] = components
            block[f"{relation}_entry_status_pair"] = status_pair
            for name, value in components.items():
                block[f"{relation}_{name}"] = value
                values[relation][name].append(value)

        for aggregate, multiplier in (
            ("block_balanced", 0.5),
            ("difference_in_differences", 1.0),
        ):
            for component in COMPONENTS:
                left = relation_components["left"][component]
                right = relation_components["right"][component]
                value = _clean_zero(
                    multiplier * (left + right)
                    if aggregate == "block_balanced"
                    else right - left
                )
                block[f"{aggregate}_{component}"] = value
                values[aggregate][component].append(value)

        if 0.5 * (entry_deltas["left"] + entry_deltas["right"]) != _number(
            row, "entry_observed_block_balanced_position_mirrored_minus_control"
        ):
            raise ValueError("upstream block-balanced entry-indicator identity changed")
        if entry_deltas["right"] - entry_deltas["left"] != _number(
            row, "entry_observed_difference_in_differences"
        ):
            raise ValueError("upstream entry-indicator difference-in-differences changed")

        if block["block_balanced_cap_restricted_waiting_action_index_contrast"] != _number(
            row,
            "cap_restricted_waiting_action_index_"
            "block_balanced_position_mirrored_minus_control",
        ):
            raise ValueError("block-balanced capped-index identity changed")
        if block["difference_in_differences_cap_restricted_waiting_action_index_contrast"] != _number(
            row, "cap_restricted_waiting_action_index_difference_in_differences"
        ):
            raise ValueError("capped-index difference-in-differences identity changed")
        per_block.append(block)

    summaries = {
        "command_specific_position_mirrored_minus_control": {
            relation: {
                component: _summary(values[relation][component])
                for component in COMPONENTS
            }
            for relation in RELATIONS
        },
        "block_balanced_position_mirrored_minus_control": {
            component: _summary(values["block_balanced"][component])
            for component in COMPONENTS
        },
        "command_by_layout_difference_in_differences": {
            component: _summary(values["difference_in_differences"][component])
            for component in COMPONENTS
        },
    }
    for group in (
        *summaries["command_specific_position_mirrored_minus_control"].values(),
        summaries["block_balanced_position_mirrored_minus_control"],
        summaries["command_by_layout_difference_in_differences"],
    ):
        if group["entry_incidence_at_cap_term"]["sum"] + group[
            "zero_masked_recorded_entry_index_term"
        ]["sum"] != group["cap_restricted_waiting_action_index_contrast"]["sum"]:
            raise ValueError("aggregate capped-index decomposition does not close")

    new_activity = dict(base["new_activity"])
    gate_state = dict(base["gate_state"])
    if len(new_activity) != 13 or any(
        type(value) is not int or value != 0 for value in new_activity.values()
    ):
        raise ValueError("decomposition must retain zero activity")
    if len(gate_state) != 9 or any(
        type(value) is not bool or value is not False for value in gate_state.values()
    ):
        raise ValueError("decomposition must retain closed gates")

    return {
        "schema_version": SCHEMA,
        "status": "post_snapshot_retained_h02_exact_cap_composite_decomposition_no_stage_advance",
        "provenance": dict(provenance),
        "configuration": dict(base["configuration"]),
        "population": {
            "base_behavioral_episodes": 108,
            "matched_environment_label_blocks": 27,
            "conditions_per_block": 4,
            "seed_labels": list(SEEDS),
            "new_cohort_created": False,
        },
        "identity": {
            "cap_action_index": CAP,
            "cell_identity": "U = C * (1 - E) + Z",
            "contrast_identity": "delta U = -C * delta E + delta Z",
            "entry_indicator_symbol": "E",
            "recorded_entry_index_symbol": "T (defined only for recorded entries)",
            "zero_masked_recorded_index_symbol": "Z = T when E=1; Z = 0 when E=0",
            "cap_restricted_index_symbol": "U",
        },
        "cross_layout_entry_status_pair_counts": status_pairs,
        "summaries": summaries,
        "per_block": per_block,
        "definitions": {
            "entry_incidence_at_cap_term": (
                "The exact -450 times reflected-minus-control change in the recorded entry-by-cap indicator E."
            ),
            "zero_masked_recorded_entry_index_term": (
                "The exact reflected-minus-control change in zero-masked recorded-index term Z. Z is the recorded entry index T when entry is observed and algebraic zero when no entry is observed; zero is bookkeeping, not an imputed entry index or event time."
            ),
            "decomposition_scope": (
                "A deterministic identity for the already-reported capped-index composite U, not a separation into independently identified incidence and entrant-only timing estimands."
            ),
            "block_balancing": (
                "Compute each component within LEFT and RIGHT for every registered environment label, then weight LEFT and RIGHT equally and summarize the 27 labels equally."
            ),
            "cross_layout_entry_status_pairs": (
                "Control/reflected entry-status pairs are comparison-direction categories across different layouts on a matched registered label; they are not temporal transitions, within-episode conversions, counterfactual changes or mediators."
            ),
            "inference": (
                "Post-result descriptive algebra and exact sign counts only; no p-value, confidence interval, survival curve, hazard, censoring-independence, physical-time, equivalence or causal claim."
            ),
        },
        "new_activity": new_activity,
        "gate_state": gate_state,
        "claim_boundary": {
            "retained_post_result_algebra_only": True,
            "exact_cap_composite_decomposition_reported": True,
            "new_underlying_behavioral_evidence": False,
            "registered_estimand": False,
            "A3_A9_completed": False,
            "default_DreamZero_evidence": False,
            "entry_is_behavioral_success": False,
            "equivalence_or_causal_effect_established": False,
            "entrant_only_timing_effect_identified": False,
            "language_or_model_diagnosis": False,
            "raw_XYZ_recovered_or_rederived": False,
            "zero_mass_is_imputed_entry_index": False,
            "physical_time_or_latency_estimated": False,
            "model_based_survival_curve_or_hazard_estimated": False,
            "post_cap_or_uncensored_event_time_inferred": False,
            "censoring_independence_claimed": False,
            "causal_mediation_or_effect_decomposition_claimed": False,
            "independent_scene_or_model_noise_generalization": False,
            "official_conditional_D1_evidence": False,
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
    "seed",
    "left_entry_status_pair",
    "right_entry_status_pair",
    *(
        f"{group}_{component}"
        for group in (*RELATIONS, "block_balanced", "difference_in_differences")
        for component in COMPONENTS
    ),
)


def render_csv(report: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["per_block"])
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
    print(json.dumps({
        "json": str(args.json_output),
        "csv": str(args.csv_output),
        "matched_blocks": report["population"]["matched_environment_label_blocks"],
        "block_balanced_components": {
            name: summary["mean"]
            for name, summary in report["summaries"][
                "block_balanced_position_mirrored_minus_control"
            ].items()
        },
        "new_activity": report["new_activity"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
