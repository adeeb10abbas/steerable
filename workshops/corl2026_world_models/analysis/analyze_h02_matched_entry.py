#!/usr/bin/env python3
"""Aggregate retained H02 requested-cone entry rows without changing H02.

The sealed H02 v1 package already authenticates 108 custom DreamZero-s2
episodes and exports one entry/censor row for each of 27 registered four-cell
environment labels.  This post-snapshot supplement authenticates that base
package, recomputes every row-level contrast from its four cells, and reports
block-balanced descriptive aggregates.  It runs no model, retrieval,
annotation, remote job or release workflow.
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
BASE_ANALYZER = PACKAGE / "analysis/analyze_h02_position_reflection.py"
BASE_JSON = PACKAGE / "paper/data/h02_position_reflection.json"
BASE_CSV = PACKAGE / "paper/data/h02_position_reflection.csv"
DEFAULT_JSON = PACKAGE / "paper/data/h02_matched_entry.json"
DEFAULT_CSV = PACKAGE / "paper/data/h02_matched_entry.csv"
SCHEMA = "wmf-retained-h02-matched-entry-supplement-v1"
BASE_IDENTITIES = {
    "analyzer": {
        "path": BASE_ANALYZER,
        "sha256": "44b941be8d362a9498798051301a52da99b2679d14f1571f049122fded59f43e",
        "bytes": 42_021,
        "lines": 877,
    },
    "json": {
        "path": BASE_JSON,
        "sha256": "9e39905c58cbd674bf41e342cadfd3e97f0c9e618bde8870e1785ed16af6adf1",
        "bytes": 103_418,
        "lines": 1_986,
    },
    "csv": {
        "path": BASE_CSV,
        "sha256": "8f1a9dd252c80d57fe543707820d3eec0bbd1580bacf61a597d33adc85b9579b",
        "bytes": 13_277,
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


def _load_h02_module():
    """Authenticate the sealed analyzer before executing any of its code."""
    expected = BASE_IDENTITIES["analyzer"]
    actual = _identity(expected["path"])
    wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
    if actual != wanted:
        raise ValueError("sealed H02 base analyzer identity changed before import")
    spec = importlib.util.spec_from_file_location(
        "sealed_h02_position_reflection", BASE_ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("sealed H02 base analyzer could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H02 = _load_h02_module()


def load_authenticated_base() -> tuple[dict[str, Any], dict[str, Any]]:
    observed: dict[str, Any] = {}
    for label, expected in BASE_IDENTITIES.items():
        actual = _identity(expected["path"])
        wanted = {key: expected[key] for key in ("sha256", "bytes", "lines")}
        if actual != wanted:
            raise ValueError(f"sealed H02 base {label} identity changed")
        observed[label] = {"path": str(expected["path"].relative_to(ROOT)), **actual}

    documents, source_provenance = H02.read_pinned_sources(ROOT)
    report = H02.analyze_documents(documents, source_provenance)
    if report.get("schema_version") != "wmf-retained-h02-position-reflection-v1":
        raise ValueError("sealed H02 base schema changed")
    if H02.render_json(report) != BASE_JSON.read_text(encoding="utf-8"):
        raise ValueError("sealed H02 base JSON does not regenerate exactly")
    if H02.render_csv(report) != BASE_CSV.read_text(encoding="utf-8"):
        raise ValueError("sealed H02 base CSV does not regenerate exactly")
    return report, {
        "sealed_package_files": observed,
        "historical_source_commit": source_provenance["commit"],
        "historical_authenticated_file_count": source_provenance[
            "authenticated_file_count"
        ],
        "historical_authenticated_bytes": source_provenance["authenticated_bytes"],
        "historical_authenticated_lines": source_provenance["authenticated_lines"],
        "episode_source": {
            "git_blob": source_provenance["files"][H02.EPISODES]["git_blob"],
            "sha256": source_provenance["files"][H02.EPISODES]["sha256"],
            "bytes": source_provenance["files"][H02.EPISODES]["bytes"],
            "lines": source_provenance["files"][H02.EPISODES]["lines"],
        },
        "matched_pair_source": {
            "git_blob": source_provenance["files"][H02.PAIRS]["git_blob"],
            "sha256": source_provenance["files"][H02.PAIRS]["sha256"],
            "bytes": source_provenance["files"][H02.PAIRS]["bytes"],
            "lines": source_provenance["files"][H02.PAIRS]["lines"],
        },
        "entry_semantics_source": {
            "git_blob": source_provenance["files"][H02.ROBOT_BRIDGE]["git_blob"],
            "sha256": source_provenance["files"][H02.ROBOT_BRIDGE]["sha256"],
            "bytes": source_provenance["files"][H02.ROBOT_BRIDGE]["bytes"],
            "lines": source_provenance["files"][H02.ROBOT_BRIDGE]["lines"],
        },
    }


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("matched descriptive values must be finite and nonempty")
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


def _condition_values(row: Mapping[str, Any], label: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for arm in H02.ARMS:
        for relation in H02.RELATIONS:
            prefix = f"{arm}_{relation}"
            observed = row.get(f"{prefix}_entry_observed")
            action_index = row.get(f"{prefix}_entry_action_index")
            waiting = row.get(f"{prefix}_restricted_waiting_action_index")
            actions = row.get(f"{prefix}_actions_executed")
            right_censored = row.get(f"{prefix}_right_censored")
            if type(observed) is not bool or type(right_censored) is not bool:
                raise ValueError("matched entry/censor flags must remain boolean")
            if isinstance(waiting, bool) or not isinstance(waiting, int) or not 0 <= waiting <= 450:
                raise ValueError("cap-restricted waiting index must be an integer in [0,450]")
            if observed:
                if (
                    isinstance(action_index, bool)
                    or not isinstance(action_index, int)
                    or action_index != waiting
                ):
                    raise ValueError("observed entry must retain its recorded action index")
            elif not (
                action_index is None
                and waiting == 450
                and actions == 450
                and right_censored is True
            ):
                raise ValueError("no-entry trace must remain censored at the action-450 cap")
            values[(arm, relation)] = float(int(observed) if label == "entry" else waiting)
    return values


def _contrasts(values: Mapping[tuple[str, str], float]) -> dict[str, float]:
    result = {
        "control_right_minus_left": (
            values[("control", "right")] - values[("control", "left")]
        ),
        "position_mirrored_right_minus_left": (
            values[("position_mirrored", "right")]
            - values[("position_mirrored", "left")]
        ),
        "position_mirrored_minus_control_left": (
            values[("position_mirrored", "left")] - values[("control", "left")]
        ),
        "position_mirrored_minus_control_right": (
            values[("position_mirrored", "right")]
            - values[("control", "right")]
        ),
    }
    result["difference_in_differences"] = (
        result["position_mirrored_right_minus_left"]
        - result["control_right_minus_left"]
    )
    if result["difference_in_differences"] != (
        result["position_mirrored_minus_control_right"]
        - result["position_mirrored_minus_control_left"]
    ):
        raise ValueError("difference-in-differences identity changed")
    result["block_balanced_position_mirrored_minus_control"] = 0.5 * (
        result["position_mirrored_minus_control_left"]
        + result["position_mirrored_minus_control_right"]
    )
    return result


def analyze_base(base: Mapping[str, Any], provenance: Mapping[str, Any]) -> dict[str, Any]:
    blocks = base.get("matched_entry_censor_blocks")
    if not isinstance(blocks, list) or len(blocks) != len(H02.SEEDS):
        raise ValueError("supplement requires exactly 27 matched H02 blocks")
    if [row.get("seed") for row in blocks] != list(H02.SEEDS):
        raise ValueError("supplement requires the ordered H02 seed-label population")

    per_block: list[dict[str, Any]] = []
    aggregate_values: dict[str, dict[str, list[float]]] = {
        "entry_observed": {},
        "cap_restricted_waiting_action_index": {},
    }
    for row in blocks:
        entry_cells = _condition_values(row, "entry")
        waiting_cells = _condition_values(row, "waiting")
        entry = _contrasts(entry_cells)
        waiting = _contrasts(waiting_cells)
        for label, values in (
            ("entry_observed", entry),
            ("cap_restricted_waiting_action_index", waiting),
        ):
            for name, value in values.items():
                aggregate_values[label].setdefault(name, []).append(value)

        # The base CSV already exports these ten non-balanced contrasts.  An
        # identity mismatch here fails closed rather than silently replacing it.
        for base_label, derived in (
            ("entry_observed", entry),
            ("restricted_waiting_action_index", waiting),
        ):
            for name, value in derived.items():
                if name == "block_balanced_position_mirrored_minus_control":
                    continue
                field = (
                    f"{base_label}_difference_in_differences"
                    if name == "difference_in_differences"
                    else f"{name}_{base_label}"
                )
                observed = row.get(field)
                if isinstance(observed, bool) or not isinstance(observed, (int, float)):
                    raise ValueError("sealed H02 contrast field has an invalid type")
                if float(observed) != value:
                    raise ValueError("sealed H02 row-level contrast identity changed")

        block = {"seed": row["seed"]}
        for label, cells, derived in (
            ("entry_observed", entry_cells, entry),
            ("cap_restricted_waiting_action_index", waiting_cells, waiting),
        ):
            for (arm, relation), value in cells.items():
                block[f"{arm}_{relation}_{label}"] = value
            for name, value in derived.items():
                block[f"{label}_{name}"] = value
        per_block.append(block)

    summaries = {
        label: {
            "block_balanced_position_mirrored_minus_control": _summary(
                values["block_balanced_position_mirrored_minus_control"]
            ),
            "command_specific_position_mirrored_minus_control": {
                relation: _summary(
                    values[f"position_mirrored_minus_control_{relation}"]
                )
                for relation in H02.RELATIONS
            },
            "within_layout_right_minus_left": {
                arm: _summary(values[f"{arm}_right_minus_left"])
                for arm in H02.ARMS
            },
            "command_by_layout_difference_in_differences": _summary(
                values["difference_in_differences"]
            ),
        }
        for label, values in aggregate_values.items()
    }

    condition = base["recorded_requested_region_entry"]["conditions"]
    marginal_means = {
        "entry_observed": {
            (arm, relation): condition[f"{arm}:{relation}"]["recorded_entry_events"] / len(H02.SEEDS)
            for arm in H02.ARMS for relation in H02.RELATIONS
        },
        "cap_restricted_waiting_action_index": {
            (arm, relation): condition[f"{arm}:{relation}"]["restricted_waiting_action_index"]["mean"]
            for arm in H02.ARMS for relation in H02.RELATIONS
        },
    }
    for label, marginal in marginal_means.items():
        left = marginal[("position_mirrored", "left")] - marginal[("control", "left")]
        right = marginal[("position_mirrored", "right")] - marginal[("control", "right")]
        observed = summaries[label]
        checks = (
            (observed["command_specific_position_mirrored_minus_control"]["left"]["mean"], left),
            (observed["command_specific_position_mirrored_minus_control"]["right"]["mean"], right),
            (observed["block_balanced_position_mirrored_minus_control"]["mean"], 0.5 * (left + right)),
            (observed["command_by_layout_difference_in_differences"]["mean"], right - left),
        )
        if any(not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12) for actual, expected in checks):
            raise ValueError("matched aggregate does not close against H02 condition marginals")

    new_activity = dict(base["new_activity"])
    gate_state = dict(base["gate_state"])
    if len(new_activity) != 13 or any(new_activity.values()):
        raise ValueError("H02 supplement must retain zero activity")
    if len(gate_state) != 9 or any(gate_state.values()):
        raise ValueError("H02 supplement must retain closed gates")
    return {
        "schema_version": SCHEMA,
        "status": "post_snapshot_retained_h02_matched_entry_aggregate_no_stage_advance",
        "provenance": dict(provenance),
        "configuration": dict(base["configuration"]),
        "population": {
            "base_behavioral_episodes": 108,
            "matched_environment_label_blocks": 27,
            "seed_labels": list(H02.SEEDS),
            "conditions_per_block": 4,
            "layout_matching_basis": "registered environment-seed label",
            "within_layout_reset_identity_authenticated": True,
            "cross_layout_reset_hash_equality_claimed": False,
            "independent_scene_or_model_noise_replicates_claimed": False,
        },
        "summaries": summaries,
        "per_block": per_block,
        "definitions": {
            "entry_observed": (
                "One only when the authenticated compact episode records a requested-cone entry action index; zero otherwise."
            ),
            "cap_restricted_waiting_action_index": (
                "Use the recorded entry action index when observed and the common administrative observation cap 450 otherwise. "
                "A value of 450 means no recorded entry by the cap, not an event at 450 or an imputed event. This descriptive "
                "capped-index composite mixes entry-by-cap incidence with first-entry index among entrants; it is not a pure timing "
                "or physical-latency contrast, a model-based or extrapolated survival curve or hazard, an uncensored or post-cap "
                "event-time inference, or a censoring-independence claim."
            ),
            "block_balanced_layout_contrast": (
                "Within each registered label, average LEFT and RIGHT position-mirrored-minus-control contrasts; then summarize labels equally."
            ),
            "signs": (
                "For the recorded requested-cone-entry indicator, positive means more recorded entries after reflection. "
                "For restricted waiting, negative means "
                "the reflected trace enters at an earlier action index or avoids a control no-entry cap."
            ),
            "inference": (
                "Post-result descriptive summaries and exact sign counts only; no p-value, confidence interval, equivalence or causal claim."
            ),
        },
        "new_activity": new_activity,
        "gate_state": gate_state,
        "claim_boundary": {
            "retained_post_result_algebra_only": True,
            "new_underlying_behavioral_evidence": False,
            "registered_estimand": False,
            "raw_XYZ_recovered_or_rederived": False,
            "entry_is_behavioral_success": False,
            "cap_value_is_imputed_entry_event": False,
            "physical_time_or_latency_estimated": False,
            "descriptive_capped_index_contrast_reported": True,
            "model_based_survival_curve_or_hazard_estimated": False,
            "post_cap_or_uncensored_event_time_inferred": False,
            "censoring_independence_claimed": False,
            "independent_scene_or_model_noise_generalization": False,
            "official_conditional_D1_evidence": False,
            "default_DreamZero_evidence": False,
            "forecast_accuracy_or_generated_execution_comparison": False,
            "A3_A9_completed": False,
            "equivalence_or_causal_effect_established": False,
            "language_or_model_diagnosis": False,
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
    *(
        f"{arm}_{relation}_{label}"
        for label in ("entry_observed", "cap_restricted_waiting_action_index")
        for arm in H02.ARMS
        for relation in H02.RELATIONS
    ),
    *(
        f"{label}_{name}"
        for label in ("entry_observed", "cap_restricted_waiting_action_index")
        for name in (
            "block_balanced_position_mirrored_minus_control",
            "position_mirrored_minus_control_left",
            "position_mirrored_minus_control_right",
            "control_right_minus_left",
            "position_mirrored_right_minus_left",
            "difference_in_differences",
        )
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
        "new_activity": report["new_activity"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
