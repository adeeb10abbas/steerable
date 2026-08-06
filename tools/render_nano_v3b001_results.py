#!/usr/bin/env python3
"""Render the completed Nano V3-B001 result as publication-ready evidence.

The renderer consumes only the fail-closed aggregate summary and its exact
108-row behavioral JSONL.  It validates the frozen prompts, cell grid,
condition counts, seed-level estimands, uncertainty records, and source hash
before drawing anything.  Outputs are written only to a new or empty
directory; there is intentionally no overwrite switch.

Two figures are emitted:

* a portrait primary figure explaining the frozen full-sample estimands and
  showing every matched-seed value, paired-bootstrap intervals, and exact sign
  tests; and
* a compact failure-taxonomy panel that retains every valid behavioral
  failure and displays zero-count categories rather than dropping them.

The media manifest deterministically selects all four arm-by-direction cells
from the lowest released seed, irrespective of outcome, and lists every
decoded local prediction exposed for those cells.  Optional local media paths
may be supplied for byte/hash verification, but media are never copied into
ordinary Git by this tool.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import tempfile
import textwrap
from typing import Any, Mapping, Sequence


SUMMARY_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-results-v1"
MANIFEST_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-results-media-manifest-v1"
FIGURE_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-scientific-figure-v1"
TAXONOMY_FIGURE_SCHEMA = "vla-wam-shared-v3b-nano-v3b001-failure-taxonomy-figure-v1"
BEHAVIORAL_SCHEMA = "vla-wam-shared-v3-raw-episode-v1"
STUDY_ID = "vla_wam_language_steerability_v3"
AMENDMENT_ID = "V3-B001"
MODEL_ID = "cosmos3_nano_policy_droid"
ARENA = "droid_robolab"
SEEDS = tuple(range(9400, 9427))
ARMS = ("control", "position_mirrored")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
FAILURE_CLASSES = (
    "correct",
    "pick_failed",
    "transport_failed",
    "wrong_side",
    "release_failed",
)
FORMULAS = {
    "s": "signed_final_lateral_offset_m; positive is robot LEFT",
    "D[a,i]": "s[a,i,left] - s[a,i,right]",
    "B[a,i]": "(-s[a,i,right]) - s[a,i,left]",
    "I[i]": "B[position_mirrored,i] - B[control,i]",
    "J[i]": "D[position_mirrored,i] - D[control,i]",
}
INTERPRETATIONS = {
    "positive_D": "the prompt change ordered endpoints LEFT-to-RIGHT",
    "positive_B": "requested-side depth is greater for RIGHT than LEFT",
    "positive_I": "the RIGHT-over-LEFT requested-depth contrast is larger after position reflection",
    "positive_J": "LEFT-to-RIGHT endpoint separation is larger after position reflection",
}

PRIMARY_WIDTH = 1440
PRIMARY_HEIGHT = 1920
TAXONOMY_WIDTH = 1440
TAXONOMY_HEIGHT = 1080
PNG_WIDTH = 2160

OUTPUT_NAMES = {
    "primary_svg": "nano_v3b001_primary.svg",
    "primary_png": "nano_v3b001_primary.png",
    "taxonomy_svg": "nano_v3b001_failure_taxonomy.svg",
    "taxonomy_png": "nano_v3b001_failure_taxonomy.png",
    "manifest": "nano_v3b001_media_manifest.json",
}

COLORS = {
    "background": "#F5F2EC",
    "card": "#FFFDF9",
    "ink": "#1E2A33",
    "muted": "#5D6972",
    "hairline": "#D7D9D3",
    "left": "#B8642B",
    "right": "#2F6F9F",
    "control": "#68737C",
    "reflected": "#377A68",
    "interaction": "#72568A",
    "positive_band": "#EAF2ED",
    "negative_band": "#F6ECE7",
    "correct": "#2F8068",
    "pick_failed": "#D1A04C",
    "transport_failed": "#B86A53",
    "wrong_side": "#875B8F",
    "release_failed": "#6D7885",
}

_SHA256_HEX = frozenset("0123456789abcdef")


class NanoResultRenderError(RuntimeError):
    """Raised when evidence or publication output fails closed."""


def _fail(message: str) -> None:
    raise NanoResultRenderError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_HEX)
    )


def _reject_constant(value: str) -> None:
    _fail(f"non-finite JSON constant is prohibited: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key is prohibited: {key}")
        result[key] = value
    return result


def _parse_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NanoResultRenderError(f"cannot parse {label}: {exc}") from exc


def _canonical_json(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise NanoResultRenderError(f"cannot serialize manifest: {exc}") from exc


def _finite(value: Any, label: str) -> float:
    _require(
        type(value) in {int, float} and math.isfinite(float(value)),
        f"{label} must be a finite number",
    )
    return float(value)


def _close(observed: Any, expected: float, label: str, tolerance: float = 1e-10) -> None:
    value = _finite(observed, label)
    _require(
        math.isclose(value, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{label} disagrees with the exact episode aggregate",
    )


def _file_record(path: Path, *, relative_path: str | None = None) -> dict[str, Any]:
    path = Path(path)
    _require(path.is_file() and path.stat().st_size > 0, f"missing or empty file: {path}")
    return {
        "path": relative_path if relative_path is not None else str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _artifact_record(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} must be an artifact object")
    _require(isinstance(value.get("path"), str) and value["path"], f"{label} path is missing")
    _require(_is_sha256(value.get("sha256")), f"{label} SHA-256 is invalid")
    _require(type(value.get("bytes")) is int and value["bytes"] > 0, f"{label} byte count is invalid")
    return {"path": value["path"], "sha256": value["sha256"], "bytes": value["bytes"]}


def _exact_sign_test(values: Sequence[float]) -> dict[str, Any]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    ties = len(values) - positive - negative
    effective = positive + negative
    if effective == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(effective, index) for index in range(min(positive, negative) + 1))
        p_value = min(1.0, 2.0 * tail / (2**effective))
    return {
        "positive": positive,
        "negative": negative,
        "ties_excluded": ties,
        "effective_n": effective,
        "p_value": p_value,
    }


@dataclass(frozen=True)
class Metric:
    key: str
    values_m: tuple[float, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class Evidence:
    summary_path: Path
    episodes_path: Path
    summary: dict[str, Any]
    ordered_rows: tuple[dict[str, Any], ...]
    rows_by_id: dict[str, dict[str, Any]]
    metrics: dict[str, Metric]
    condition_outcomes: dict[str, dict[str, Any]]
    taxonomy_counts: dict[str, int]
    source_sha256: dict[str, str]

    @property
    def selected_seed(self) -> int:
        return min(SEEDS)

    @property
    def selected_rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            row for row in self.ordered_rows if row["environment_seed"] == self.selected_seed
        )


def _expected_cell_id(seed: int, arm: str, relation: str) -> str:
    return f"v3b001:nano:seed{seed}:{arm}:{relation}"


def _validate_episode_row(row: Any, index: int) -> dict[str, Any]:
    label = f"episode row {index}"
    _require(isinstance(row, dict), f"{label} must be an object")
    expected_scalar = {
        "schema_version": BEHAVIORAL_SCHEMA,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "amendment_id": AMENDMENT_ID,
        "prompt_family": "direct_command",
        "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
        "missing_future_policy": "infrastructure_invalid_never_zero",
    }
    for key, expected in expected_scalar.items():
        _require(row.get(key) == expected, f"{label} has unexpected {key}")
    seed = row.get("environment_seed")
    arm = row.get("phase_b_arm")
    relation = row.get("requested_relation")
    _require(type(seed) is int and seed in SEEDS, f"{label} has an unauthorized seed")
    _require(row.get("policy_seed") == seed, f"{label} policy seed is not matched")
    _require(arm in ARMS, f"{label} has an invalid Phase-B arm")
    _require(relation in RELATIONS, f"{label} has an invalid requested relation")
    cell_id = _expected_cell_id(seed, arm, relation)
    _require(row.get("registered_cell_id") == cell_id, f"{label} cell identity changed")
    _require(row.get("pair_id") == f"v3b001:nano:seed{seed}", f"{label} pair identity changed")
    _require(row.get("prompt") == PROMPTS[relation], f"{cell_id} exact prompt bytes changed")
    _require(type(row.get("requested_success")) is bool, f"{cell_id} success must be boolean")
    taxonomy = row.get("failure_taxonomy")
    _require(taxonomy in FAILURE_CLASSES, f"{cell_id} has an unknown failure taxonomy")
    _require(
        (taxonomy == "correct") == row["requested_success"],
        f"{cell_id} success and failure taxonomy disagree",
    )
    measurements = row.get("measurements")
    _require(isinstance(measurements, dict), f"{cell_id} measurements are missing")
    signed = _finite(measurements.get("signed_final_lateral_offset_m"), f"{cell_id} signed offset")
    margin = _finite(measurements.get("final_requested_signed_margin_m"), f"{cell_id} requested margin")
    expected_margin = signed if relation == "left" else -signed
    _require(
        math.isclose(margin, expected_margin, rel_tol=0.0, abs_tol=1e-12),
        f"{cell_id} requested margin sign is inconsistent",
    )
    artifacts = row.get("artifacts")
    _require(isinstance(artifacts, dict), f"{cell_id} artifacts are missing")
    _artifact_record(artifacts.get("viewport_video"), f"{cell_id} actual viewport video")
    future_requests = row.get("future_requests")
    _require(isinstance(future_requests, list) and future_requests, f"{cell_id} decoded futures are missing")
    observed_indices: list[int] = []
    for request in future_requests:
        _require(isinstance(request, dict), f"{cell_id} future request must be an object")
        request_index = request.get("request_index")
        _require(type(request_index) is int and request_index >= 0, f"{cell_id} future request index is invalid")
        observed_indices.append(request_index)
        _artifact_record(
            request.get("decoded_future"),
            f"{cell_id} decoded local prediction request {request_index}",
        )
        shape = request.get("decoded_future_shape")
        _require(
            isinstance(shape, list)
            and len(shape) == 4
            and shape[0] == 33
            and shape[-1] == 3
            and all(type(value) is int and value > 0 for value in shape),
            f"{cell_id} decoded local prediction shape changed",
        )
        _require(
            request.get("future_evidence_status") == "exposed_and_retained",
            f"{cell_id} future evidence status changed",
        )
    _require(
        observed_indices == list(range(len(future_requests))),
        f"{cell_id} decoded local prediction request order is not contiguous",
    )
    return row


def _validate_metric(record: Any, values: Sequence[float], label: str, bootstrap_replicates: int) -> dict[str, Any]:
    _require(isinstance(record, dict), f"summary metric {label} is missing")
    _require(record.get("n") == len(values), f"summary metric {label} n changed")
    expected = {
        "mean_m": statistics.fmean(values),
        "median_m": float(statistics.median(values)),
        "minimum_m": min(values),
        "maximum_m": max(values),
    }
    for key, value in expected.items():
        _close(record.get(key), value, f"summary metric {label} {key}")
    if len(values) > 1:
        _close(
            record.get("sample_standard_deviation_m"),
            statistics.stdev(values),
            f"summary metric {label} sample standard deviation",
        )
    interval = record.get("mean_bootstrap_95")
    _require(isinstance(interval, dict), f"summary metric {label} mean CI is missing")
    expected_interval = {
        "method": "matched_seed_nonparametric_percentile_bootstrap",
        "unit_of_resampling": "matched_seed",
        "statistic": "mean",
        "confidence": 0.95,
        "replicates": bootstrap_replicates,
    }
    for key, value in expected_interval.items():
        _require(interval.get(key) == value, f"summary metric {label} CI {key} changed")
    lower = _finite(interval.get("lower"), f"summary metric {label} CI lower")
    upper = _finite(interval.get("upper"), f"summary metric {label} CI upper")
    _require(lower <= upper, f"summary metric {label} CI is reversed")
    test = record.get("paired_sign_test")
    _require(isinstance(test, dict), f"summary metric {label} sign test is missing")
    _require(
        test.get("method") == "exact_two_sided_paired_sign_test",
        f"summary metric {label} sign-test method changed",
    )
    exact = _exact_sign_test(values)
    for key in ("positive", "negative", "ties_excluded", "effective_n"):
        _require(test.get(key) == exact[key], f"summary metric {label} sign-test {key} disagrees")
    _close(test.get("p_value"), exact["p_value"], f"summary metric {label} sign-test p")
    return record


def load_evidence(summary_path: Path, episodes_path: Path) -> Evidence:
    """Load and cross-validate the exact completed aggregate."""

    summary_path = Path(summary_path).resolve()
    episodes_path = Path(episodes_path).resolve()
    _require(summary_path.is_file(), f"summary does not exist: {summary_path}")
    _require(episodes_path.is_file(), f"episode aggregate does not exist: {episodes_path}")
    summary = _parse_json(summary_path.read_bytes(), f"summary {summary_path}")
    _require(isinstance(summary, dict), "summary must be an object")
    expected_summary = {
        "schema_version": SUMMARY_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "exact_prompts": PROMPTS,
    }
    for key, expected in expected_summary.items():
        _require(summary.get(key) == expected, f"summary {key} changed")
    _require(isinstance(summary.get("claim_boundary"), str) and summary["claim_boundary"].strip(), "summary claim boundary is missing")
    evidence_record = summary.get("behavioral_evidence")
    _require(isinstance(evidence_record, dict), "summary behavioral evidence is missing")
    _require(evidence_record.get("valid_episode_count") == 108, "summary behavioral denominator changed")
    _require(evidence_record.get("matched_seed_count") == 27, "summary matched-seed count changed")
    aggregate = evidence_record.get("aggregate_jsonl")
    _require(isinstance(aggregate, dict), "summary aggregate JSONL record is missing")
    _require(Path(str(aggregate.get("path"))).name == episodes_path.name, "summary aggregate path does not bind the supplied JSONL")
    _require(aggregate.get("bytes") == episodes_path.stat().st_size, "episode aggregate byte count disagrees with summary")
    _require(aggregate.get("sha256") == sha256_file(episodes_path), "episode aggregate SHA-256 disagrees with summary")

    lines = episodes_path.read_bytes().splitlines()
    _require(len(lines) == 108 and all(line.strip() for line in lines), "episode aggregate must contain exactly 108 non-empty rows")
    ordered_rows: list[dict[str, Any]] = []
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, line in enumerate(lines, 1):
        row = _validate_episode_row(_parse_json(line, f"{episodes_path}:{index}"), index)
        cell_id = row["registered_cell_id"]
        _require(cell_id not in rows_by_id, f"duplicate behavioral cell: {cell_id}")
        rows_by_id[cell_id] = row
        ordered_rows.append(row)
    expected_ids = {
        _expected_cell_id(seed, arm, relation)
        for seed in SEEDS
        for arm in ARMS
        for relation in RELATIONS
    }
    _require(set(rows_by_id) == expected_ids, "episode aggregate does not contain the exact 27×2×2 released grid")

    condition_outcomes: dict[str, dict[str, Any]] = {}
    total_taxonomy: Counter[str] = Counter()
    signed: dict[tuple[int, str, str], float] = {}
    for seed in SEEDS:
        for arm in ARMS:
            for relation in RELATIONS:
                row = rows_by_id[_expected_cell_id(seed, arm, relation)]
                signed[(seed, arm, relation)] = float(row["measurements"]["signed_final_lateral_offset_m"])
                total_taxonomy[row["failure_taxonomy"]] += 1
    for arm in ARMS:
        for relation in RELATIONS:
            rows = [rows_by_id[_expected_cell_id(seed, arm, relation)] for seed in SEEDS]
            counts = dict(sorted(Counter(row["failure_taxonomy"] for row in rows).items()))
            condition_outcomes[f"{arm}:{relation}"] = {
                "episodes": 27,
                "successes": sum(row["requested_success"] for row in rows),
                "failure_taxonomy_counts": counts,
            }
    _require(summary.get("condition_outcomes") == condition_outcomes, "summary condition outcomes disagree with episode aggregate")
    expected_taxonomy = dict(sorted(total_taxonomy.items()))
    _require(summary.get("failure_taxonomy_counts") == expected_taxonomy, "summary failure taxonomy disagrees with episode aggregate")

    seed_level = summary.get("seed_level")
    _require(isinstance(seed_level, list) and len(seed_level) == 27, "summary seed-level result grid changed")
    seed_records: dict[int, dict[str, Any]] = {}
    vectors: dict[str, list[float]] = {
        "D_control": [],
        "D_position_mirrored": [],
        "B_control": [],
        "B_position_mirrored": [],
        "I": [],
        "J": [],
    }
    complete_case_seeds: list[int] = []
    for entry in seed_level:
        _require(isinstance(entry, dict), "summary seed-level row must be an object")
        seed = entry.get("seed")
        _require(type(seed) is int and seed in SEEDS and seed not in seed_records, "summary seed-level identity changed")
        seed_records[seed] = entry
        d_control = signed[(seed, "control", "left")] - signed[(seed, "control", "right")]
        d_reflected = signed[(seed, "position_mirrored", "left")] - signed[(seed, "position_mirrored", "right")]
        b_control = -signed[(seed, "control", "right")] - signed[(seed, "control", "left")]
        b_reflected = -signed[(seed, "position_mirrored", "right")] - signed[(seed, "position_mirrored", "left")]
        i_value = b_reflected - b_control
        j_value = d_reflected - d_control
        expected_values = {
            "D_control_m": d_control,
            "D_position_mirrored_m": d_reflected,
            "B_control_m": b_control,
            "B_position_mirrored_m": b_reflected,
            "I_position_reflection_interaction_m": i_value,
            "J_redirection_interaction_m": j_value,
        }
        full = entry.get("full_sample")
        _require(isinstance(full, dict), f"summary seed {seed} full-sample values are missing")
        for key, value in expected_values.items():
            _close(full.get(key), value, f"summary seed {seed} {key}")
        vectors["D_control"].append(d_control)
        vectors["D_position_mirrored"].append(d_reflected)
        vectors["B_control"].append(b_control)
        vectors["B_position_mirrored"].append(b_reflected)
        vectors["I"].append(i_value)
        vectors["J"].append(j_value)
        if all(
            rows_by_id[_expected_cell_id(seed, arm, relation)]["requested_success"]
            for arm in ARMS
            for relation in RELATIONS
        ):
            complete_case_seeds.append(seed)
    _require(set(seed_records) == set(SEEDS), "summary seed-level result grid is incomplete")

    full_sample = summary.get("full_sample_primary")
    _require(isinstance(full_sample, dict), "summary full-sample primary analysis is missing")
    population = full_sample.get("population")
    _require(
        population == {
            "matched_seed_count": 27,
            "behavioral_episode_count": 108,
            "valid_failures_included": True,
            "infrastructure_attempts_included": False,
            "missing_value_imputation": "none",
        },
        "summary full-sample population contract changed",
    )
    _require(full_sample.get("formulas") == FORMULAS, "summary frozen estimand formulas changed")
    _require(full_sample.get("interpretation") == INTERPRETATIONS, "summary estimand interpretations changed")
    uncertainty = summary.get("uncertainty_contract")
    _require(isinstance(uncertainty, dict), "summary uncertainty contract is missing")
    _require(uncertainty.get("unit") == "matched_seed", "summary uncertainty unit changed")
    bootstrap_replicates = uncertainty.get("bootstrap_replicates")
    _require(type(bootstrap_replicates) is int and bootstrap_replicates > 0, "summary bootstrap replicate count is invalid")

    metric_sources = {
        "D_control": full_sample.get("D_by_arm", {}).get("control"),
        "D_position_mirrored": full_sample.get("D_by_arm", {}).get("position_mirrored"),
        "B_control": full_sample.get("B_by_arm", {}).get("control"),
        "B_position_mirrored": full_sample.get("B_by_arm", {}).get("position_mirrored"),
        "I": full_sample.get("I_position_reflection_interaction"),
        "J": full_sample.get("J_redirection_interaction"),
    }
    metrics: dict[str, Metric] = {}
    for key, values in vectors.items():
        record = _validate_metric(metric_sources[key], values, key, bootstrap_replicates)
        metrics[key] = Metric(key=key, values_m=tuple(values), summary=record)

    secondary = summary.get("success_conditional_secondary")
    _require(isinstance(secondary, dict), "summary success-conditional secondary analysis is missing")
    _require(
        secondary.get("subset_id") == "nano_v3b001_all_four_cells_correct",
        "summary success-complete subset identity changed",
    )
    _require(
        secondary.get("realized_matched_seed_count") == len(complete_case_seeds)
        and secondary.get("included_seeds") == complete_case_seeds,
        "summary success-complete subset disagrees with episode outcomes",
    )
    _require(secondary.get("failures_as_zero") is False, "summary converted failures to zeros")
    _require(secondary.get("unmatched_successful_cells_used") is False, "summary mixed unmatched successes")

    return Evidence(
        summary_path=summary_path,
        episodes_path=episodes_path,
        summary=summary,
        ordered_rows=tuple(ordered_rows),
        rows_by_id=rows_by_id,
        metrics=metrics,
        condition_outcomes=condition_outcomes,
        taxonomy_counts={name: total_taxonomy.get(name, 0) for name in FAILURE_CLASSES},
        source_sha256={
            "summary": sha256_file(summary_path),
            "episodes": sha256_file(episodes_path),
        },
    )


class Svg:
    def __init__(self) -> None:
        self.parts: list[str] = []

    def raw(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: float, y: float, width: float, height: float, **attrs: Any) -> None:
        values = {"x": x, "y": y, "width": width, "height": height, **attrs}
        self.raw("<rect " + _attrs(values) + "/>")

    def line(self, x1: float, y1: float, x2: float, y2: float, **attrs: Any) -> None:
        values = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, **attrs}
        self.raw("<line " + _attrs(values) + "/>")

    def circle(self, cx: float, cy: float, r: float, **attrs: Any) -> None:
        values = {"cx": cx, "cy": cy, "r": r, **attrs}
        self.raw("<circle " + _attrs(values) + "/>")

    def polygon(self, points: Sequence[tuple[float, float]], **attrs: Any) -> None:
        values = {"points": " ".join(f"{x:g},{y:g}" for x, y in points), **attrs}
        self.raw("<polygon " + _attrs(values) + "/>")

    def text(self, x: float, y: float, value: str, class_: str, **attrs: Any) -> None:
        values = {"x": x, "y": y, "class": class_, **attrs}
        self.raw(f"<text {_attrs(values)}>{html.escape(value)}</text>")

    def multiline(
        self,
        x: float,
        y: float,
        lines: Sequence[str],
        class_: str,
        *,
        line_height: float,
        **attrs: Any,
    ) -> None:
        values = {"x": x, "y": y, "class": class_, **attrs}
        tspans = []
        for index, line in enumerate(lines):
            dy = 0 if index == 0 else line_height
            tspans.append(
                f'<tspan x="{x:g}" dy="{dy:g}">{html.escape(line)}</tspan>'
            )
        self.raw(f"<text {_attrs(values)}>{''.join(tspans)}</text>")

    def build(self) -> str:
        return "".join(self.parts)


def _attrs(values: Mapping[str, Any]) -> str:
    rendered = []
    for key, value in values.items():
        key = key.replace("_", "-")
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        rendered.append(f'{key}="{html.escape(text, quote=True)}"')
    return " ".join(rendered)


def _svg_open(
    *,
    width: int,
    height: int,
    title_id: str,
    desc_id: str,
    title: str,
    description: str,
    metadata: Mapping[str, Any],
) -> Svg:
    svg = Svg()
    svg.raw(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" '
        'style="max-width:100%;height:auto;display:block" role="img" '
        f'aria-labelledby="{title_id} {desc_id}">'
    )
    svg.raw(f'<title id="{title_id}">{html.escape(title)}</title>')
    svg.raw(f'<desc id="{desc_id}">{html.escape(description)}</desc>')
    svg.raw(
        "<metadata>"
        + html.escape(json.dumps(metadata, allow_nan=False, sort_keys=True, separators=(",", ":")))
        + "</metadata>"
    )
    svg.raw(
        "<style>"
        ".all{font-family:'Source Sans 3','Aptos','Segoe UI',Arial,sans-serif;fill:#1E2A33}"
        ".serif{font-family:'Source Serif 4','Iowan Old Style',Georgia,serif}"
        ".overline{font-size:14px;font-weight:700;letter-spacing:1.4px}"
        ".title{font-size:45px;font-weight:600;letter-spacing:-.6px}"
        ".subtitle{font-size:19px;fill:#5D6972}"
        ".prompt-label{font-size:13px;font-weight:700;letter-spacing:1px}"
        ".prompt{font-size:21px;font-weight:600}"
        ".section{font-size:25px;font-weight:650}"
        ".card-title{font-size:17px;font-weight:700}"
        ".body{font-size:16px}.small{font-size:13px;fill:#5D6972}"
        ".metric-name{font-size:15px;font-weight:700}.metric-value{font-size:14px;font-weight:650}"
        ".tick{font-size:12px;fill:#5D6972;font-variant-numeric:tabular-nums}"
        ".axis{font-size:13px;font-weight:600}.count{font-size:27px;font-weight:700}"
        ".footer{font-size:13px;fill:#5D6972}.footer-strong{font-size:13px;font-weight:700}"
        ".taxonomy-label{font-size:17px;font-weight:650}.taxonomy-count{font-size:15px;font-weight:700}"
        "</style>"
    )
    svg.rect(0, 0, width, height, fill=COLORS["background"])
    svg.raw('<g class="all">')
    return svg


def _prompt_card(svg: Svg, relation: str, x: float, y: float, width: float) -> None:
    color = COLORS[relation]
    svg.rect(x, y, width, 108, rx=16, fill=COLORS["card"], stroke=color, stroke_width=1.6)
    svg.text(x + 24, y + 31, f"{relation.upper()} CONDITION · EXACT EPISODE-STATIC PROMPT", "prompt-label", fill=color)
    svg.text(x + 24, y + 71, PROMPTS[relation], "prompt")


def _metric_domain(metrics: Sequence[Metric]) -> float:
    values_cm: list[float] = []
    for metric in metrics:
        values_cm.extend(value * 100.0 for value in metric.values_m)
        interval = metric.summary["mean_bootstrap_95"]
        values_cm.extend([float(interval["lower"]) * 100.0, float(interval["upper"]) * 100.0])
    maximum = max((abs(value) for value in values_cm), default=1.0)
    return max(5.0, math.ceil((maximum * 1.08) / 5.0) * 5.0)


def _x(value_cm: float, limit: float, x0: float, x1: float) -> float:
    return x0 + (value_cm + limit) / (2.0 * limit) * (x1 - x0)


def _format_p(value: float) -> str:
    if value < 0.0001:
        return f"{value:.2e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def _metric_stats(metric: Metric) -> tuple[float, float, float, float, dict[str, Any]]:
    summary = metric.summary
    interval = summary["mean_bootstrap_95"]
    return (
        float(summary["mean_m"]) * 100.0,
        float(interval["lower"]) * 100.0,
        float(interval["upper"]) * 100.0,
        float(summary["median_m"]) * 100.0,
        summary["paired_sign_test"],
    )


def _draw_metric_panel(
    svg: Svg,
    *,
    x: float,
    y: float,
    width: float,
    title: str,
    subtitle: str,
    metrics: Sequence[tuple[str, Metric, str]],
) -> None:
    height = 310
    svg.rect(x, y, width, height, rx=18, fill=COLORS["card"], stroke=COLORS["hairline"], stroke_width=1.4)
    svg.text(x + 26, y + 37, title, "section")
    svg.text(x + 26, y + 64, subtitle, "small")
    plot_x0, plot_x1 = x + 310, x + 900
    stats_x = x + 930
    plot_top, plot_bottom = y + 82, y + 244
    limit = _metric_domain([metric for _, metric, _ in metrics])
    zero = _x(0.0, limit, plot_x0, plot_x1)
    svg.rect(plot_x0, plot_top, zero - plot_x0, plot_bottom - plot_top, fill=COLORS["negative_band"])
    svg.rect(zero, plot_top, plot_x1 - zero, plot_bottom - plot_top, fill=COLORS["positive_band"])
    for fraction in (-1.0, -0.5, 0.0, 0.5, 1.0):
        value = limit * fraction
        tick_x = _x(value, limit, plot_x0, plot_x1)
        svg.line(
            tick_x,
            plot_top,
            tick_x,
            plot_bottom,
            stroke=COLORS["ink"] if fraction == 0 else COLORS["hairline"],
            stroke_width=1.7 if fraction == 0 else 1.0,
        )
        svg.text(tick_x, y + 268, f"{value:+.0f}" if value else "0", "tick", text_anchor="middle")
    row_ys = (y + 112, y + 165, y + 218)
    for (label, metric, color), row_y in zip(metrics, row_ys, strict=True):
        svg.text(x + 28, row_y + 5, label, "metric-name")
        for index, value in enumerate(metric.values_m):
            dot_x = _x(value * 100.0, limit, plot_x0, plot_x1)
            dot_y = row_y + ((index * 5) % 9 - 4) * 1.3
            svg.circle(dot_x, dot_y, 3.2, fill=color, fill_opacity=0.38)
        mean, lower, upper, median, test = _metric_stats(metric)
        svg.line(
            _x(lower, limit, plot_x0, plot_x1),
            row_y,
            _x(upper, limit, plot_x0, plot_x1),
            row_y,
            stroke=color,
            stroke_width=5.2,
            stroke_linecap="round",
        )
        svg.circle(_x(mean, limit, plot_x0, plot_x1), row_y, 7.2, fill=color, stroke=COLORS["card"], stroke_width=2)
        median_x = _x(median, limit, plot_x0, plot_x1)
        svg.polygon(
            ((median_x, row_y - 8), (median_x + 6, row_y), (median_x, row_y + 8), (median_x - 6, row_y)),
            fill=COLORS["card"],
            stroke=color,
            stroke_width=2,
        )
        svg.text(stats_x, row_y - 3, f"mean {mean:+.1f} cm · 95% CI [{lower:+.1f}, {upper:+.1f}]", "metric-value")
        svg.text(
            stats_x,
            row_y + 17,
            f"median {median:+.1f} · sign p={_format_p(float(test['p_value']))} · +/−/tie {test['positive']}/{test['negative']}/{test['ties_excluded']}",
            "small",
        )
    svg.text((plot_x0 + plot_x1) / 2, y + 292, "estimand value (cm) · green side follows the positive definition", "axis", text_anchor="middle")
    svg.circle(x + width - 255, y + 285, 5, fill=COLORS["control"])
    svg.text(x + width - 243, y + 290, "dots = all 27 seeds", "small")


def _wrap(value: str, width: int) -> list[str]:
    return textwrap.wrap(value, width=width, break_long_words=False, break_on_hyphens=False)


def render_primary_svg(evidence: Evidence) -> str:
    metadata = {
        "schema_version": FIGURE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "exact_prompts": PROMPTS,
        "source_sha256": evidence.source_sha256,
        "estimands": FORMULAS,
        "uncertainty_contract": evidence.summary["uncertainty_contract"],
        "claim_boundary": evidence.summary["claim_boundary"],
    }
    svg = _svg_open(
        width=PRIMARY_WIDTH,
        height=PRIMARY_HEIGHT,
        title_id="nano-v3b001-title",
        desc_id="nano-v3b001-desc",
        title="Nano V3-B001 paired position-reflection result",
        description="Every one of 27 matched seeds contributes control and position-reflected LEFT and RIGHT episodes. The figure reports the frozen full-sample endpoint-redirection and requested-depth estimands with deterministic paired-bootstrap intervals and exact paired sign tests.",
        metadata=metadata,
    )
    svg.text(60, 55, "NANO V3-B001 · DROID/ROBOLAB · 27 MATCHED SEEDS", "overline", fill=COLORS["reflected"])
    svg.text(60, 112, "Did reflecting object positions change", "title serif")
    svg.text(60, 163, "the directional contrast?", "title serif")
    svg.text(60, 198, "Positions-only movable-object reflection · 108 valid episodes · all valid failures retained", "subtitle")
    _prompt_card(svg, "left", 60, 224, 645)
    _prompt_card(svg, "right", 735, 224, 645)

    svg.rect(60, 354, 1320, 102, rx=16, fill=COLORS["card"], stroke=COLORS["hairline"], stroke_width=1.4)
    svg.text(86, 387, "MATCHED DESIGN", "prompt-label", fill=COLORS["reflected"])
    svg.text(86, 418, "Each seed contributes control LEFT, control RIGHT, position-reflected LEFT, and position-reflected RIGHT.", "body")
    svg.text(86, 443, "The primary full-sample analysis includes successes and behavioral failures; infrastructure attempts and imputation are excluded.", "small")

    svg.text(60, 501, "What the four frozen estimands measure", "section")
    cards = (
        ("D · endpoint redirection", "sLEFT − sRIGHT", "Positive: endpoints follow the requested LEFT-to-RIGHT ordering.", COLORS["control"]),
        ("B · requested-depth contrast", "marginRIGHT − marginLEFT", "Positive: RIGHT finishes deeper in its requested side than LEFT.", COLORS["right"]),
        ("J · reflection interaction on D", "Dreflected − Dcontrol", "Positive: endpoint separation is larger after position reflection.", COLORS["interaction"]),
        ("I · reflection interaction on B", "Breflected − Bcontrol", "Positive: the RIGHT-over-LEFT depth contrast is larger after reflection.", COLORS["interaction"]),
    )
    for index, (heading, formula, body, color) in enumerate(cards):
        col = index % 2
        row = index // 2
        card_x = 60 + col * 675
        card_y = 526 + row * 112
        svg.rect(card_x, card_y, 645, 96, rx=14, fill=COLORS["card"], stroke=color, stroke_width=1.3)
        svg.text(card_x + 20, card_y + 27, heading, "card-title", fill=color)
        svg.text(card_x + 20, card_y + 51, formula, "body")
        svg.text(card_x + 20, card_y + 76, body, "small")

    svg.text(60, 776, "Observed paired full-sample estimates", "section")
    svg.text(60, 804, "Dots are matched seeds; thick lines are deterministic paired-bootstrap 95% CIs; circles are means and diamonds are medians.", "small")
    _draw_metric_panel(
        svg,
        x=60,
        y=826,
        width=1320,
        title="Endpoint redirection",
        subtitle="D is evaluated within each layout; J is the position-reflected minus control change in D.",
        metrics=(
            ("Control D", evidence.metrics["D_control"], COLORS["control"]),
            ("Position-reflected D", evidence.metrics["D_position_mirrored"], COLORS["reflected"]),
            ("J: reflected − control", evidence.metrics["J"], COLORS["interaction"]),
        ),
    )
    _draw_metric_panel(
        svg,
        x=60,
        y=1158,
        width=1320,
        title="Requested-side depth contrast",
        subtitle="B compares RIGHT and LEFT requested margins within each layout; I is the reflected minus control change in B.",
        metrics=(
            ("Control B", evidence.metrics["B_control"], COLORS["control"]),
            ("Position-reflected B", evidence.metrics["B_position_mirrored"], COLORS["reflected"]),
            ("I: reflected − control", evidence.metrics["I"], COLORS["interaction"]),
        ),
    )

    svg.text(60, 1516, "Task outcomes remain a separate diagnostic", "section")
    outcome_labels = (
        ("control:left", "Control · LEFT", COLORS["left"]),
        ("control:right", "Control · RIGHT", COLORS["right"]),
        ("position_mirrored:left", "Reflected · LEFT", COLORS["left"]),
        ("position_mirrored:right", "Reflected · RIGHT", COLORS["right"]),
    )
    for index, (key, label, color) in enumerate(outcome_labels):
        card_x = 60 + index * 330
        record = evidence.condition_outcomes[key]
        svg.rect(card_x, 1542, 300, 104, rx=14, fill=COLORS["card"], stroke=color, stroke_width=1.25)
        svg.text(card_x + 18, 1571, label, "card-title", fill=color)
        svg.text(card_x + 18, 1612, f"{record['successes']}/27", "count", fill=color)
        svg.text(card_x + 102, 1611, "frozen task successes", "small")
    secondary = evidence.summary["success_conditional_secondary"]
    svg.text(60, 1676, "Success-conditional secondary subset", "footer-strong")
    svg.text(
        355,
        1676,
        f"{secondary['realized_matched_seed_count']}/27 seeds had all four cells correct; unmatched successful cells were not mixed and failures were not encoded as zero.",
        "footer",
    )

    svg.line(60, 1710, 1380, 1710, stroke=COLORS["hairline"], stroke_width=1.4)
    svg.text(60, 1740, "MEDIA LABELS", "prompt-label", fill=COLORS["reflected"])
    svg.text(60, 1770, "ACTUAL SIMULATOR ROLLOUT", "footer-strong")
    svg.text(280, 1770, "Viewport video of executed robot behavior; this is the behavioral episode.", "footer")
    svg.text(60, 1798, "DECODED LOCAL PREDICTION", "footer-strong")
    svg.text(280, 1798, "A 33-frame model future exposed at one policy request; not execution and not an additional episode.", "footer")
    svg.text(60, 1832, "UNCERTAINTY", "footer-strong")
    svg.text(180, 1832, "Mean CIs use matched-seed percentile bootstrap; p is the exact two-sided paired sign test with zero ties excluded. No multiplicity adjustment.", "footer")
    claim_lines = _wrap(evidence.summary["claim_boundary"], 145)[:2]
    svg.text(60, 1862, "FROZEN CLAIM BOUNDARY", "footer-strong")
    svg.multiline(245, 1862, claim_lines, "footer", line_height=22)
    svg.raw("</g></svg>")
    return svg.build()


def render_taxonomy_svg(evidence: Evidence) -> str:
    metadata = {
        "schema_version": TAXONOMY_FIGURE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "exact_prompts": PROMPTS,
        "source_sha256": evidence.source_sha256,
        "failure_classes": list(FAILURE_CLASSES),
    }
    svg = _svg_open(
        width=TAXONOMY_WIDTH,
        height=TAXONOMY_HEIGHT,
        title_id="nano-v3b001-taxonomy-title",
        desc_id="nano-v3b001-taxonomy-desc",
        title="Nano V3-B001 failure taxonomy",
        description="Stacked condition bars and full-cohort counts show every mutually exclusive outcome among 108 valid behavioral episodes, including zero-count categories.",
        metadata=metadata,
    )
    svg.text(60, 55, "NANO V3-B001 · FAILURE DECOMPOSITION", "overline", fill=COLORS["reflected"])
    svg.text(60, 112, "What happened in the 108 valid episodes?", "title serif")
    svg.text(60, 148, "Mutually exclusive frozen taxonomy · infrastructure attempts remain outside the behavioral denominator", "subtitle")
    svg.rect(60, 172, 1320, 64, rx=13, fill=COLORS["card"], stroke=COLORS["left"], stroke_width=1.2)
    svg.text(82, 197, "LEFT PROMPT", "prompt-label", fill=COLORS["left"])
    svg.text(214, 207, PROMPTS["left"], "prompt")
    svg.rect(60, 248, 1320, 64, rx=13, fill=COLORS["card"], stroke=COLORS["right"], stroke_width=1.2)
    svg.text(82, 273, "RIGHT PROMPT", "prompt-label", fill=COLORS["right"])
    svg.text(214, 283, PROMPTS["right"], "prompt")

    legend_x = 60
    for name in FAILURE_CLASSES:
        svg.rect(legend_x, 340, 16, 16, rx=3, fill=COLORS[name])
        svg.text(legend_x + 23, 354, name.replace("_", " "), "small")
        legend_x += 250

    rows = (
        ("control:left", "Control · LEFT", COLORS["left"]),
        ("control:right", "Control · RIGHT", COLORS["right"]),
        ("position_mirrored:left", "Position-reflected · LEFT", COLORS["left"]),
        ("position_mirrored:right", "Position-reflected · RIGHT", COLORS["right"]),
    )
    bar_x0, bar_x1 = 390.0, 1330.0
    for index, (key, label, label_color) in enumerate(rows):
        row_y = 412 + index * 92
        record = evidence.condition_outcomes[key]
        svg.text(60, row_y + 16, label, "taxonomy-label", fill=label_color)
        svg.text(60, row_y + 41, f"{record['successes']}/27 correct", "small")
        counts = record["failure_taxonomy_counts"]
        cursor = bar_x0
        for name in FAILURE_CLASSES:
            count = int(counts.get(name, 0))
            segment = (bar_x1 - bar_x0) * count / 27.0
            if segment > 0:
                svg.rect(cursor, row_y - 8, segment, 48, fill=COLORS[name])
                if segment >= 28:
                    svg.text(cursor + segment / 2, row_y + 23, str(count), "taxonomy-count", fill="#FFFFFF", text_anchor="middle")
            cursor += segment
        svg.rect(bar_x0, row_y - 8, bar_x1 - bar_x0, 48, rx=6, fill="none", stroke=COLORS["hairline"], stroke_width=1.2)

    svg.text(60, 790, "Full-cohort counts", "section")
    for index, name in enumerate(FAILURE_CLASSES):
        card_x = 60 + index * 264
        svg.rect(card_x, 814, 240, 104, rx=14, fill=COLORS["card"], stroke=COLORS[name], stroke_width=1.3)
        svg.text(card_x + 18, 845, name.replace("_", " "), "card-title", fill=COLORS[name])
        svg.text(card_x + 18, 890, str(evidence.taxonomy_counts[name]), "count", fill=COLORS[name])
        svg.text(card_x + 68, 889, "of 108", "small")

    svg.line(60, 950, 1380, 950, stroke=COLORS["hairline"], stroke_width=1.4)
    svg.text(60, 980, "HOW TO READ", "footer-strong")
    svg.text(175, 980, "‘correct’ is the frozen requested-success outcome. Each valid episode appears exactly once; zero-count classes remain visible.", "footer")
    svg.text(60, 1008, "MEDIA LABELS", "footer-strong")
    svg.text(175, 1008, "Actual simulator rollout is executed behavior. A decoded local prediction is a per-request model future, not execution or an additional outcome.", "footer")
    svg.text(60, 1036, "BOUNDARY", "footer-strong")
    svg.text(175, 1036, "DROID/RoboLab only. No infrastructure attempt enters these 108 outcomes and no failure is converted to missing data or zero.", "footer")
    svg.raw("</g></svg>")
    return svg.build()


def _write_exclusive(path: Path, payload: bytes) -> None:
    try:
        with Path(path).open("xb") as handle:
            handle.write(payload)
    except FileExistsError as exc:
        raise NanoResultRenderError(f"refusing to overwrite output: {path}") from exc


def _render_png(svg_path: Path, png_path: Path, width: int = PNG_WIDTH) -> str:
    renderer = shutil.which("rsvg-convert")
    _require(renderer is not None, "rsvg-convert is required for the mandatory high-resolution PNG export")
    completed = subprocess.run(
        [renderer, "--format=png", f"--width={width}", "--output", str(png_path), str(svg_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(completed.returncode == 0, f"rsvg-convert failed: {completed.stderr.strip()}")
    _require(png_path.is_file() and png_path.stat().st_size > 0, "PNG renderer produced no output")
    return "rsvg-convert"


def _png_dimensions(path: Path) -> tuple[int, int]:
    payload = Path(path).read_bytes()[:24]
    _require(payload.startswith(b"\x89PNG\r\n\x1a\n") and len(payload) >= 24, f"invalid PNG output: {path}")
    return struct.unpack(">II", payload[16:24])


def _verify_local_asset(path: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(path.is_file() and path.stat().st_size > 0, f"{label} local asset is missing or empty: {path}")
    _require(path.stat().st_size == expected["bytes"], f"{label} local asset byte count disagrees with aggregate")
    digest = sha256_file(path)
    _require(digest == expected["sha256"], f"{label} local asset SHA-256 disagrees with aggregate")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def _media_manifest(
    evidence: Evidence,
    *,
    generated_files: Mapping[str, Mapping[str, Any]],
    actual_rollout_assets: Mapping[str, Path],
    decoded_prediction_assets: Mapping[tuple[str, int], Path],
    png_renderer: str,
) -> dict[str, Any]:
    selected_rows = evidence.selected_rows
    _require(len(selected_rows) == 4, "deterministic lowest-seed selection did not produce four cells")
    expected_actual = {row["registered_cell_id"] for row in selected_rows}
    expected_predictions = {
        (row["registered_cell_id"], request["request_index"])
        for row in selected_rows
        for request in row["future_requests"]
    }
    if actual_rollout_assets:
        _require(
            set(actual_rollout_assets) == expected_actual,
            "optional actual-rollout assets must provide all four deterministically selected cells",
        )
    if decoded_prediction_assets:
        _require(
            set(decoded_prediction_assets) == expected_predictions,
            "optional decoded-prediction assets must provide every exposed request for the selected cells",
        )

    selected: list[dict[str, Any]] = []
    for row in selected_rows:
        cell_id = row["registered_cell_id"]
        actual_source = _artifact_record(row["artifacts"]["viewport_video"], f"{cell_id} viewport")
        actual = {
            "media_kind": "actual_simulator_rollout",
            "label": "ACTUAL SIMULATOR ROLLOUT — executed robot behavior",
            "source": actual_source,
            "local_verification": (
                {
                    "status": "verified_local_asset",
                    "file": _verify_local_asset(actual_rollout_assets[cell_id], actual_source, f"{cell_id} actual rollout"),
                }
                if actual_rollout_assets
                else {"status": "aggregate_hash_reference_only_source_not_opened"}
            ),
        }
        predictions: list[dict[str, Any]] = []
        for request in row["future_requests"]:
            request_index = request["request_index"]
            source = _artifact_record(request["decoded_future"], f"{cell_id} future {request_index}")
            key = (cell_id, request_index)
            predictions.append(
                {
                    "request_index": request_index,
                    "action_step_start": request.get("action_step_start"),
                    "media_kind": "decoded_local_prediction_not_execution",
                    "label": "DECODED LOCAL PREDICTION — model future, not execution",
                    "shape": request["decoded_future_shape"],
                    "source": source,
                    "local_verification": (
                        {
                            "status": "verified_local_asset",
                            "file": _verify_local_asset(
                                decoded_prediction_assets[key],
                                source,
                                f"{cell_id} decoded prediction request {request_index}",
                            ),
                        }
                        if decoded_prediction_assets
                        else {"status": "aggregate_hash_reference_only_source_not_opened"}
                    ),
                }
            )
        selected.append(
            {
                "registered_cell_id": cell_id,
                "source_aggregate_row_index": evidence.ordered_rows.index(row),
                "environment_seed": row["environment_seed"],
                "arm": row["phase_b_arm"],
                "requested_relation": row["requested_relation"],
                "exact_prompt": row["prompt"],
                "requested_success": row["requested_success"],
                "failure_taxonomy": row["failure_taxonomy"],
                "actual_rollout": actual,
                "decoded_local_predictions": predictions,
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "complete",
        "exact_prompts": PROMPTS,
        "claim_boundary": evidence.summary["claim_boundary"],
        "sources": {
            "summary": _file_record(evidence.summary_path),
            "episode_aggregate": _file_record(evidence.episodes_path),
        },
        "generated_files": dict(generated_files),
        "png_renderer": png_renderer,
        "selection": {
            "rule_id": "lowest_released_seed_all_four_cells_all_exposed_predictions_v1",
            "rule": "Select the minimum released environment seed; include all four control/position-reflected × LEFT/RIGHT cells in aggregate order irrespective of outcome; include each selected cell's actual viewport rollout and every exposed decoded local prediction in request order.",
            "selected_seed": evidence.selected_seed,
            "outcome_used_for_selection": False,
            "no_substitution_for_missing_media": True,
            "statistics_affected_by_media_selection": False,
            "selected_cell_count": 4,
            "selected_decoded_prediction_count": sum(len(row["future_requests"]) for row in selected_rows),
            "no_outcome_hiding_audit": {
                "selected_successes": sum(row["requested_success"] for row in selected_rows),
                "selected_failures": sum(not row["requested_success"] for row in selected_rows),
                "full_cohort_condition_outcomes": evidence.condition_outcomes,
                "full_cohort_failure_taxonomy_counts": evidence.taxonomy_counts,
            },
        },
        "media_semantics": {
            "actual_simulator_rollout": "Executed robot behavior and the source behavioral episode.",
            "decoded_local_prediction_not_execution": "A retained 33-frame future exposed at one policy request; not execution, not a task outcome, and not an additional behavioral episode.",
            "missing_future_policy": "Missing or unavailable future evidence is never converted to zero.",
        },
        "selected_media": selected,
    }


def _ensure_output_target(output_directory: Path) -> Path:
    output_directory = Path(output_directory).resolve()
    if output_directory.exists():
        _require(output_directory.is_dir(), f"output path is not a directory: {output_directory}")
        _require(not any(output_directory.iterdir()), f"output directory must be empty: {output_directory}")
    else:
        output_directory.parent.mkdir(parents=True, exist_ok=True)
    return output_directory


def _publish_stage(stage: Path, output_directory: Path) -> None:
    if output_directory.exists():
        _require(not any(output_directory.iterdir()), f"output directory became non-empty: {output_directory}")
        for source in sorted(stage.iterdir(), key=lambda path: path.name):
            target = output_directory / source.name
            _require(not target.exists(), f"refusing to overwrite output: {target}")
            os.replace(source, target)
        stage.rmdir()
    else:
        os.replace(stage, output_directory)


def render_nano_v3b001_results(
    *,
    summary_path: Path,
    episodes_path: Path,
    output_directory: Path,
    actual_rollout_assets: Mapping[str, Path] | None = None,
    decoded_prediction_assets: Mapping[tuple[str, int], Path] | None = None,
) -> dict[str, Path]:
    """Validate evidence and write the complete deterministic publication slice."""

    evidence = load_evidence(summary_path, episodes_path)
    output_directory = _ensure_output_target(output_directory)
    actual_rollout_assets = dict(actual_rollout_assets or {})
    decoded_prediction_assets = dict(decoded_prediction_assets or {})
    parent = output_directory.parent
    stage = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.render-", dir=parent))
    try:
        primary_svg = stage / OUTPUT_NAMES["primary_svg"]
        primary_png = stage / OUTPUT_NAMES["primary_png"]
        taxonomy_svg = stage / OUTPUT_NAMES["taxonomy_svg"]
        taxonomy_png = stage / OUTPUT_NAMES["taxonomy_png"]
        manifest_path = stage / OUTPUT_NAMES["manifest"]
        _write_exclusive(primary_svg, render_primary_svg(evidence).encode("utf-8"))
        _write_exclusive(taxonomy_svg, render_taxonomy_svg(evidence).encode("utf-8"))
        renderer = _render_png(primary_svg, primary_png)
        _render_png(taxonomy_svg, taxonomy_png)
        primary_dimensions = _png_dimensions(primary_png)
        taxonomy_dimensions = _png_dimensions(taxonomy_png)
        _require(primary_dimensions[0] == PNG_WIDTH, "primary PNG width is not high-resolution")
        _require(taxonomy_dimensions[0] == PNG_WIDTH, "taxonomy PNG width is not high-resolution")
        generated = {
            "primary_svg": {
                **_file_record(primary_svg, relative_path=primary_svg.name),
                "view_box": [0, 0, PRIMARY_WIDTH, PRIMARY_HEIGHT],
                "responsive": True,
            },
            "primary_png": {
                **_file_record(primary_png, relative_path=primary_png.name),
                "pixel_dimensions": list(primary_dimensions),
            },
            "failure_taxonomy_svg": {
                **_file_record(taxonomy_svg, relative_path=taxonomy_svg.name),
                "view_box": [0, 0, TAXONOMY_WIDTH, TAXONOMY_HEIGHT],
                "responsive": True,
            },
            "failure_taxonomy_png": {
                **_file_record(taxonomy_png, relative_path=taxonomy_png.name),
                "pixel_dimensions": list(taxonomy_dimensions),
            },
        }
        manifest = _media_manifest(
            evidence,
            generated_files=generated,
            actual_rollout_assets=actual_rollout_assets,
            decoded_prediction_assets=decoded_prediction_assets,
            png_renderer=renderer,
        )
        _write_exclusive(manifest_path, _canonical_json(manifest))
        _publish_stage(stage, output_directory)
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise
    return {key: output_directory / filename for key, filename in OUTPUT_NAMES.items()}


def _parse_actual_specs(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        _require("=" in value, "--actual-rollout-asset requires CELL_ID=PATH")
        cell_id, raw_path = value.split("=", 1)
        _require(cell_id and raw_path and cell_id not in result, "invalid or duplicate actual-rollout asset mapping")
        result[cell_id] = Path(raw_path).expanduser().resolve()
    return result


def _parse_prediction_specs(values: Sequence[str]) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    for value in values:
        _require("=" in value, "--decoded-prediction-asset requires CELL_ID@REQUEST_INDEX=PATH")
        key_text, raw_path = value.split("=", 1)
        _require("@" in key_text and raw_path, "--decoded-prediction-asset key is invalid")
        cell_id, index_text = key_text.rsplit("@", 1)
        try:
            request_index = int(index_text)
        except ValueError as exc:
            raise NanoResultRenderError("decoded-prediction request index must be an integer") from exc
        key = (cell_id, request_index)
        _require(cell_id and request_index >= 0 and key not in result, "invalid or duplicate decoded-prediction mapping")
        result[key] = Path(raw_path).expanduser().resolve()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True, help="Completed nano_v3b001_summary.json")
    parser.add_argument("--episodes", type=Path, required=True, help="Exact nano_v3b001_episodes.jsonl bound by the summary")
    parser.add_argument("--output-directory", type=Path, required=True, help="New or empty output directory; overwrite is prohibited")
    parser.add_argument(
        "--actual-rollout-asset",
        action="append",
        default=[],
        metavar="CELL_ID=PATH",
        help="Optional local copy for every selected actual rollout; if used, all four selected cells are required",
    )
    parser.add_argument(
        "--decoded-prediction-asset",
        action="append",
        default=[],
        metavar="CELL_ID@REQUEST_INDEX=PATH",
        help="Optional local copy for every selected decoded future; if used, the complete selected request set is required",
    )
    args = parser.parse_args()
    outputs = render_nano_v3b001_results(
        summary_path=args.summary,
        episodes_path=args.episodes,
        output_directory=args.output_directory,
        actual_rollout_assets=_parse_actual_specs(args.actual_rollout_asset),
        decoded_prediction_assets=_parse_prediction_specs(args.decoded_prediction_asset),
    )
    result = {
        "status": "complete",
        "schema_version": MANIFEST_SCHEMA,
        "outputs": {
            key: _file_record(path)
            for key, path in sorted(outputs.items())
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
