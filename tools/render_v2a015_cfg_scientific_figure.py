#!/usr/bin/env python3
"""Render the V2-A015 guidance ablation as a reader-first scientific SVG.

The renderer is deliberately JSON-only: it consumes the four compact compiled
results plus their compact paired comparison and never reads raw rollouts.  It
fails closed when model identities, schemas, prompt bytes, seeds, source hashes,
or source-to-comparison values disagree.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


FIGURE_SCHEMA = "vla-wam-shared-v2-cfg-ablation-v2a015-figure-v1"
COMPARISON_SCHEMA = "vla-wam-shared-v2-cfg-ablation-v2a015-comparison-v1"
ARENA = "droid_robolab"
SEEDS = (8300, 8301, 8302)
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}

WIDTH = 1600
HEIGHT = 1530

COLORS = {
    "background": "#F7F4EE",
    "card": "#FFFDF9",
    "ink": "#23313B",
    "muted": "#65727B",
    "line": "#D8DDD9",
    "left": "#C87932",
    "right": "#4D7698",
    "success": "#3D8067",
    "negative_band": "#F6ECE9",
    "positive_band": "#EDF3EE",
    "accent": "#71647F",
}


@dataclass(frozen=True)
class ResultSpec:
    key: str
    schema: str
    model_id: str
    amendment_id: str


RESULT_SPECS = {
    "cosmos_baseline": ResultSpec(
        key="cosmos_baseline",
        schema="vla-wam-shared-v2-cosmos3-nano-policy-droid-result-v1",
        model_id="cosmos3_nano_policy_droid",
        amendment_id="V2-A011",
    ),
    "cosmos_intervention": ResultSpec(
        key="cosmos_intervention",
        schema="vla-wam-shared-v2-cosmos3-nano-v2a015-g1-result-v1",
        model_id="cosmos3_nano_policy_droid",
        amendment_id="V2-A015",
    ),
    "dreamzero_baseline": ResultSpec(
        key="dreamzero_baseline",
        schema="vla-wam-shared-v2-dreamzero-droid-direct-gate-v1",
        model_id="dreamzero_droid",
        amendment_id="V2-A007",
    ),
    "dreamzero_intervention": ResultSpec(
        key="dreamzero_intervention",
        schema="vla-wam-shared-v2-dreamzero-v2a015-s2-result-v1",
        model_id="dreamzero_droid_action_cfg",
        amendment_id="V2-A015",
    ),
}


@dataclass(frozen=True)
class Episode:
    seed: int
    relation: str
    prompt: str
    success: bool
    margin_m: float


@dataclass(frozen=True)
class Configuration:
    label: str
    short_label: str
    episodes: dict[tuple[int, str], Episode]

    def successes(self, relation: str) -> int:
        return sum(
            episode.success
            for episode in self.episodes.values()
            if episode.relation == relation
        )

    def total_successes(self) -> int:
        return sum(episode.success for episode in self.episodes.values())

    def mean_margin(self, relation: str) -> float:
        values = [
            episode.margin_m
            for episode in self.episodes.values()
            if episode.relation == relation
        ]
        return sum(values) / len(values)

    def signed_direction_gap(self) -> float:
        """Return mean RIGHT requested margin minus mean LEFT margin."""

        return self.mean_margin("right") - self.mean_margin("left")


@dataclass(frozen=True)
class ModelComparison:
    key: str
    display_name: str
    technical_description: str
    baseline: Configuration
    intervention: Configuration


@dataclass(frozen=True)
class FigureEvidence:
    models: tuple[ModelComparison, ModelComparison]
    source_hashes: dict[str, str]


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _require(condition: bool, message: str) -> None:
    if not condition:
        _fail(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"Could not load {label} JSON {path}: {exc}")
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _finite_number(value: Any, label: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{label} must be finite")
    return result


def _margin_from_episode(row: dict[str, Any], relation: str, label: str) -> float:
    final_lateral = _finite_number(
        row.get("final_lateral_display_m"), f"{label} final_lateral_display_m"
    )
    derived = -final_lateral if relation == "left" else final_lateral
    explicit_values = []
    for key in ("requested_signed_final_margin_m", "requested_signed_final_offset_m"):
        if row.get(key) is not None:
            explicit_values.append(_finite_number(row[key], f"{label} {key}"))
    for explicit in explicit_values:
        _require(
            math.isclose(explicit, derived, rel_tol=0.0, abs_tol=1e-9),
            f"{label} requested margin disagrees with final lateral endpoint",
        )
    return explicit_values[0] if explicit_values else derived


def _expected_grid() -> set[tuple[int, str]]:
    return {(seed, relation) for seed in SEEDS for relation in RELATIONS}


def _validate_result_summary(
    payload: dict[str, Any], spec: ResultSpec, episodes: dict[tuple[int, str], Episode]
) -> None:
    successes = {
        relation: sum(row.success for row in episodes.values() if row.relation == relation)
        for relation in RELATIONS
    }
    total = sum(successes.values())
    if spec.key == "dreamzero_baseline":
        _require(payload.get("valid_episode_count") == 6, "DreamZero baseline denominator changed")
        _require(payload.get("requested_success_count") == total, "DreamZero baseline success total disagrees with episodes")
        by_relation = payload.get("success_by_relation")
        _require(isinstance(by_relation, dict), "DreamZero baseline success_by_relation is missing")
        for relation in RELATIONS:
            record = by_relation.get(relation)
            _require(isinstance(record, dict), f"DreamZero baseline {relation} summary is missing")
            _require(record.get("successes") == successes[relation], f"DreamZero baseline {relation} success count disagrees")
            _require(record.get("trials") == 3, f"DreamZero baseline {relation} trial count changed")
        return

    summary = payload.get("summary")
    _require(isinstance(summary, dict), f"{spec.key} summary is missing")
    by_direction = summary.get("by_direction")
    _require(isinstance(by_direction, dict), f"{spec.key} by_direction summary is missing")
    for relation in RELATIONS:
        record = by_direction.get(relation)
        _require(isinstance(record, dict), f"{spec.key} {relation} summary is missing")
        _require(record.get("successes") == successes[relation], f"{spec.key} {relation} success count disagrees")
        _require(record.get("episodes") == 3, f"{spec.key} {relation} denominator changed")
        if "prompt" in record:
            _require(record["prompt"] == PROMPTS[relation], f"{spec.key} {relation} summary prompt changed")
    observed_n = summary.get("valid_episode_count", summary.get("episode_count"))
    observed_successes = summary.get("requested_success_count", summary.get("successes"))
    _require(observed_n == 6, f"{spec.key} total denominator changed")
    _require(observed_successes == total, f"{spec.key} total success count disagrees")


def _validate_result(path: Path, spec: ResultSpec) -> tuple[dict[str, Any], dict[tuple[int, str], Episode]]:
    payload = _load_json(path, spec.key)
    _require(payload.get("schema_version") == spec.schema, f"{spec.key} schema changed")
    _require(payload.get("model_id") == spec.model_id, f"{spec.key} model identity changed")
    _require(payload.get("amendment_id") == spec.amendment_id, f"{spec.key} amendment identity changed")
    _require(payload.get("status") == "complete", f"{spec.key} is not complete")
    if "arena" in payload:
        _require(payload["arena"] == ARENA, f"{spec.key} is not DROID/RoboLab evidence")
    if "exact_prompts" in payload:
        _require(payload["exact_prompts"] == PROMPTS, f"{spec.key} exact prompt registry changed")

    rows = payload.get("episodes")
    _require(isinstance(rows, list) and len(rows) == 6, f"{spec.key} must contain exactly six episodes")
    episodes: dict[tuple[int, str], Episode] = {}
    for index, row in enumerate(rows):
        _require(isinstance(row, dict), f"{spec.key} episode {index} must be an object")
        label = f"{spec.key} episode {index}"
        _require(type(row.get("environment_seed")) is int, f"{label} environment_seed must be an integer")
        seed = row["environment_seed"]
        relation = row.get("requested_relation")
        _require(seed in SEEDS, f"{label} has unauthorized seed {seed}")
        _require(relation in RELATIONS, f"{label} has invalid requested_relation")
        _require(row.get("sampling_seed") == seed, f"{label} sampling seed must match environment seed")
        _require(row.get("prompt") == PROMPTS[relation], f"{label} exact prompt bytes changed")
        _require(type(row.get("requested_success")) is bool, f"{label} requested_success must be boolean")
        key = (seed, relation)
        _require(key not in episodes, f"{spec.key} duplicates seed/relation {key}")
        episodes[key] = Episode(
            seed=seed,
            relation=relation,
            prompt=PROMPTS[relation],
            success=row["requested_success"],
            margin_m=_margin_from_episode(row, relation, label),
        )
    _require(set(episodes) == _expected_grid(), f"{spec.key} does not contain the exact seed/relation grid")
    _validate_result_summary(payload, spec, episodes)
    return payload, episodes


def _validate_source_record(record: Any, source: Path, label: str) -> None:
    _require(isinstance(record, dict), f"Comparison provenance is missing {label}")
    _require(record.get("bytes") == source.stat().st_size, f"Comparison {label} byte count does not bind the supplied source")
    _require(record.get("sha256") == sha256(source), f"Comparison {label} hash does not bind the supplied source")


def _counter_as_plain(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "unchanged_success"
    if before and not after:
        return "regressed_success_to_failure"
    if not before and after:
        return "improved_failure_to_success"
    return "unchanged_failure"


def _validate_configuration_summary(
    summary: Any,
    episodes: dict[tuple[int, str], Episode],
    label: str,
) -> None:
    _require(isinstance(summary, dict), f"Comparison {label} configuration summary is missing")
    _require(summary.get("valid_episode_count") == 6, f"Comparison {label} denominator changed")
    total = sum(row.success for row in episodes.values())
    _require(summary.get("requested_success_count") == total, f"Comparison {label} success total disagrees")
    by_direction = summary.get("by_direction")
    _require(isinstance(by_direction, dict), f"Comparison {label} by_direction is missing")
    means: dict[str, float] = {}
    for relation in RELATIONS:
        record = by_direction.get(relation)
        _require(isinstance(record, dict), f"Comparison {label} {relation} summary is missing")
        rows = [row for row in episodes.values() if row.relation == relation]
        expected_successes = sum(row.success for row in rows)
        expected_mean = sum(row.margin_m for row in rows) / 3
        _require(record.get("prompt") == PROMPTS[relation], f"Comparison {label} {relation} summary prompt changed")
        _require(record.get("episodes") == 3, f"Comparison {label} {relation} denominator changed")
        _require(record.get("successes") == expected_successes, f"Comparison {label} {relation} successes disagree")
        margins = record.get("requested_margin_m")
        _require(isinstance(margins, dict), f"Comparison {label} {relation} margins are missing")
        _require(math.isclose(_finite_number(margins.get("mean"), f"Comparison {label} {relation} mean margin"), expected_mean, rel_tol=0.0, abs_tol=1e-9), f"Comparison {label} {relation} mean margin disagrees")
        values = margins.get("values_by_seed")
        _require(isinstance(values, list) and len(values) == 3, f"Comparison {label} {relation} seed margins changed")
        observed = {}
        for value in values:
            _require(isinstance(value, dict), f"Comparison {label} {relation} seed margin must be an object")
            seed = value.get("environment_seed")
            _require(type(seed) is int and seed in SEEDS and seed not in observed, f"Comparison {label} {relation} seed margin grid changed")
            observed[seed] = _finite_number(value.get("value"), f"Comparison {label} {relation} seed {seed} margin")
        for row in rows:
            _require(math.isclose(observed[row.seed], row.margin_m, rel_tol=0.0, abs_tol=1e-9), f"Comparison {label} {relation} seed {row.seed} margin disagrees")
        means[relation] = expected_mean
    balance = summary.get("mean_margin_balance")
    _require(isinstance(balance, dict), f"Comparison {label} mean-margin balance is missing")
    gap = means["right"] - means["left"]
    observed_gap = _finite_number(balance.get("right_minus_left_m"), f"Comparison {label} direction gap")
    observed_abs = _finite_number(balance.get("absolute_direction_imbalance_m"), f"Comparison {label} absolute imbalance")
    _require(math.isclose(observed_gap, gap, rel_tol=0.0, abs_tol=1e-9), f"Comparison {label} direction gap disagrees")
    _require(math.isclose(observed_abs, abs(gap), rel_tol=0.0, abs_tol=1e-9), f"Comparison {label} absolute imbalance disagrees")


def _validate_comparison_model(
    record: Any,
    *,
    key: str,
    expected_model: str,
    expected_baseline_label: str,
    expected_intervention_label: str,
    baseline: dict[tuple[int, str], Episode],
    intervention: dict[tuple[int, str], Episode],
) -> None:
    _require(isinstance(record, dict), f"Comparison entry {key} is missing")
    _require(record.get("model") == expected_model, f"Comparison {key} model label changed")
    _require(record.get("baseline_label") == expected_baseline_label, f"Comparison {key} baseline label changed")
    _require(record.get("intervention_label") == expected_intervention_label, f"Comparison {key} intervention label changed")
    _require(record.get("exact_prompts") == PROMPTS, f"Comparison {key} exact prompts changed")
    cells = record.get("cells")
    _require(isinstance(cells, list) and len(cells) == 6, f"Comparison {key} must contain six cells")
    observed_grid = set()
    transitions: Counter[str] = Counter()
    transitions_by_relation = {relation: Counter() for relation in RELATIONS}
    for index, cell in enumerate(cells):
        _require(isinstance(cell, dict), f"Comparison {key} cell {index} must be an object")
        seed = cell.get("environment_seed")
        relation = cell.get("requested_relation")
        _require(type(seed) is int and seed in SEEDS, f"Comparison {key} cell {index} has invalid seed")
        _require(relation in RELATIONS, f"Comparison {key} cell {index} has invalid relation")
        grid_key = (seed, relation)
        _require(grid_key not in observed_grid, f"Comparison {key} duplicates {grid_key}")
        observed_grid.add(grid_key)
        _require(cell.get("cell_id") == f"seed{seed}_{relation}", f"Comparison {key} cell_id changed")
        _require(cell.get("prompt") == PROMPTS[relation], f"Comparison {key} exact prompt bytes changed")
        success = cell.get("success")
        margins = cell.get("requested_signed_final_margin_m")
        _require(isinstance(success, dict) and isinstance(margins, dict), f"Comparison {key} cell evidence is incomplete")
        before = success.get(expected_baseline_label)
        after = success.get(expected_intervention_label)
        _require(type(before) is bool and type(after) is bool, f"Comparison {key} success values must be boolean")
        _require(before == baseline[grid_key].success, f"Comparison {key} baseline success disagrees with source")
        _require(after == intervention[grid_key].success, f"Comparison {key} intervention success disagrees with source")
        expected_transition = _transition(before, after)
        _require(success.get("transition") == expected_transition, f"Comparison {key} transition disagrees")
        transitions[expected_transition] += 1
        transitions_by_relation[relation][expected_transition] += 1
        for label, source in (
            (expected_baseline_label, baseline),
            (expected_intervention_label, intervention),
        ):
            value = _finite_number(margins.get(label), f"Comparison {key} {label} margin")
            _require(math.isclose(value, source[grid_key].margin_m, rel_tol=0.0, abs_tol=1e-9), f"Comparison {key} {label} margin disagrees with source")
        delta = _finite_number(margins.get("intervention_minus_baseline"), f"Comparison {key} paired margin effect")
        expected_delta = intervention[grid_key].margin_m - baseline[grid_key].margin_m
        _require(math.isclose(delta, expected_delta, rel_tol=0.0, abs_tol=1e-9), f"Comparison {key} paired margin effect disagrees")
    _require(observed_grid == _expected_grid(), f"Comparison {key} seed/relation grid changed")

    success_summary = record.get("success")
    _require(isinstance(success_summary, dict), f"Comparison {key} success summary is missing")
    before_total = sum(row.success for row in baseline.values())
    after_total = sum(row.success for row in intervention.values())
    _require(success_summary.get("baseline_total") == before_total, f"Comparison {key} baseline total disagrees")
    _require(success_summary.get("intervention_total") == after_total, f"Comparison {key} intervention total disagrees")
    _require(success_summary.get("net_success_change") == after_total - before_total, f"Comparison {key} net success change disagrees")
    _require(success_summary.get("exact_paired_transitions") == _counter_as_plain(transitions), f"Comparison {key} paired transitions disagree")
    observed_by_relation = success_summary.get("exact_paired_transitions_by_relation")
    expected_by_relation = {
        relation: _counter_as_plain(transitions_by_relation[relation])
        for relation in RELATIONS
    }
    _require(observed_by_relation == expected_by_relation, f"Comparison {key} relation transitions disagree")
    _validate_configuration_summary(record.get("baseline_configuration_summary"), baseline, f"{key} baseline")
    _validate_configuration_summary(record.get("intervention_configuration_summary"), intervention, f"{key} intervention")

    paired = record.get("paired_seed_diagnostics")
    _require(isinstance(paired, list) and len(paired) == 3, f"Comparison {key} paired seed diagnostics changed")
    paired_seeds = set()
    for row in paired:
        _require(isinstance(row, dict), f"Comparison {key} paired diagnostic must be an object")
        seed = row.get("environment_seed")
        _require(type(seed) is int and seed in SEEDS and seed not in paired_seeds, f"Comparison {key} paired seed grid changed")
        paired_seeds.add(seed)
        _require(row.get("left_prompt") == PROMPTS["left"] and row.get("right_prompt") == PROMPTS["right"], f"Comparison {key} paired prompt bytes changed")
    _require(paired_seeds == set(SEEDS), f"Comparison {key} paired seed grid is incomplete")


def load_evidence(
    *,
    cosmos_baseline: Path,
    cosmos_intervention: Path,
    dreamzero_baseline: Path,
    dreamzero_intervention: Path,
    comparison_path: Path,
) -> FigureEvidence:
    source_paths = {
        "cosmos_baseline": cosmos_baseline,
        "cosmos_intervention": cosmos_intervention,
        "dreamzero_baseline": dreamzero_baseline,
        "dreamzero_intervention": dreamzero_intervention,
    }
    payloads: dict[str, dict[str, Any]] = {}
    episodes: dict[str, dict[tuple[int, str], Episode]] = {}
    for key, source in source_paths.items():
        payloads[key], episodes[key] = _validate_result(source, RESULT_SPECS[key])

    comparison = _load_json(comparison_path, "V2-A015 comparison")
    _require(comparison.get("schema_version") == COMPARISON_SCHEMA, "V2-A015 comparison schema changed")
    _require(comparison.get("status") == "complete", "V2-A015 comparison is not complete")
    _require(comparison.get("amendment_id") == "V2-A015", "V2-A015 comparison amendment changed")
    _require(comparison.get("arena") == ARENA, "V2-A015 comparison is not DROID/RoboLab")
    _require(comparison.get("exact_prompts") == PROMPTS, "V2-A015 comparison exact prompts changed")

    provenance = comparison.get("provenance")
    _require(isinstance(provenance, dict), "V2-A015 comparison provenance is missing")
    provenance_keys = {
        "cosmos_baseline": "cosmos3_nano_baseline",
        "cosmos_intervention": "cosmos3_nano_intervention",
        "dreamzero_baseline": "dreamzero_baseline",
        "dreamzero_intervention": "dreamzero_intervention",
    }
    for source_key, provenance_key in provenance_keys.items():
        _validate_source_record(provenance.get(provenance_key), source_paths[source_key], provenance_key)

    comparisons = comparison.get("comparisons")
    _require(isinstance(comparisons, dict) and set(comparisons) == {"cosmos3_nano", "dreamzero"}, "V2-A015 comparison model set changed")
    _validate_comparison_model(
        comparisons["cosmos3_nano"],
        key="cosmos3_nano",
        expected_model="Cosmos3 Nano Policy DROID",
        expected_baseline_label="g=3 baseline",
        expected_intervention_label="g=1 intervention",
        baseline=episodes["cosmos_baseline"],
        intervention=episodes["cosmos_intervention"],
    )
    _validate_comparison_model(
        comparisons["dreamzero"],
        key="dreamzero",
        expected_model="DreamZero DROID",
        expected_baseline_label="s=1 conditional-action equivalent",
        expected_intervention_label="s=2 CFG-style negative-branch action guidance",
        baseline=episodes["dreamzero_baseline"],
        intervention=episodes["dreamzero_intervention"],
    )

    models = (
        ModelComparison(
            key="cosmos3_nano",
            display_name="Cosmos3 Nano Policy DROID",
            technical_description="Joint action–video guidance reduced: g = 3 baseline → g = 1 (no joint CFG blend)",
            baseline=Configuration(
                label="g=3 baseline",
                short_label="g = 3 baseline",
                episodes=episodes["cosmos_baseline"],
            ),
            intervention=Configuration(
                label="g=1 intervention",
                short_label="g = 1 intervention",
                episodes=episodes["cosmos_intervention"],
            ),
        ),
        ModelComparison(
            key="dreamzero",
            display_name="DreamZero DROID",
            technical_description="CFG-style negative-branch action guidance increased: s = 1 → s = 2; video CFG fixed at 5",
            baseline=Configuration(
                label="s=1 conditional-action equivalent",
                short_label="s = 1 baseline",
                episodes=episodes["dreamzero_baseline"],
            ),
            intervention=Configuration(
                label="s=2 CFG-style negative-branch action guidance",
                short_label="s = 2 intervention",
                episodes=episodes["dreamzero_intervention"],
            ),
        ),
    )
    hashes = {key: sha256(path) for key, path in source_paths.items()}
    hashes["comparison"] = sha256(comparison_path)
    return FigureEvidence(models=models, source_hashes=hashes)


class SvgBuilder:
    def __init__(self) -> None:
        self.parts: list[str] = []

    @staticmethod
    def _attrs(attrs: dict[str, Any]) -> str:
        def normalize(key: str) -> str:
            if key.endswith("_"):
                key = key[:-1]
            return key.replace("_", "-")

        return " ".join(
            f'{normalize(key)}="{html.escape(str(value), quote=True)}"'
            for key, value in attrs.items()
            if value is not None
        )

    def raw(self, value: str) -> None:
        self.parts.append(value)

    def rect(self, x: float, y: float, width: float, height: float, **attrs: Any) -> None:
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" {self._attrs(attrs)}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, **attrs: Any) -> None:
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" {self._attrs(attrs)}/>'
        )

    def circle(self, cx: float, cy: float, radius: float, **attrs: Any) -> None:
        self.parts.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{radius:.2f}" {self._attrs(attrs)}/>'
        )

    def diamond(self, cx: float, cy: float, radius: float, **attrs: Any) -> None:
        points = " ".join(
            f"{x:.2f},{y:.2f}"
            for x, y in (
                (cx, cy - radius),
                (cx + radius, cy),
                (cx, cy + radius),
                (cx - radius, cy),
            )
        )
        self.parts.append(f'<polygon points="{points}" {self._attrs(attrs)}/>')

    def text(self, x: float, y: float, value: str, css_class: str, **attrs: Any) -> None:
        attributes = {"class": css_class, **attrs}
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" {self._attrs(attributes)}>{html.escape(value)}</text>'
        )

    def group_start(self, **attrs: Any) -> None:
        self.parts.append(f'<g {self._attrs(attrs)}>')

    def group_end(self) -> None:
        self.parts.append("</g>")

    def build(self) -> str:
        return "\n".join(self.parts) + "\n"


def _nice_scale(values: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(values)
    _require(values, "Cannot render an empty requested-margin scale")
    raw_min = min(min(values), 0.0)
    raw_max = max(max(values), 0.0)
    span = max(raw_max - raw_min, 0.1)
    target = span / 6.0
    magnitude = 10 ** math.floor(math.log10(target))
    normalized = target / magnitude
    if normalized <= 1:
        step = magnitude
    elif normalized <= 2:
        step = 2 * magnitude
    elif normalized <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude
    low = math.floor((raw_min - 0.12 * span) / step) * step
    high = math.ceil((raw_max + 0.12 * span) / step) * step
    low = min(low, -step)
    high = max(high, step)
    return low, high, step


def _format_signed_m(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:+.3f} m"


def _favored_direction(gap: float) -> str:
    if gap > 0.005:
        return "RIGHT larger"
    if gap < -0.005:
        return "LEFT larger"
    return "approximately balanced"


def _observed_story(model: ModelComparison) -> str:
    before = model.baseline.signed_direction_gap()
    after = model.intervention.signed_direction_gap()
    before_side = _favored_direction(before)
    after_side = _favored_direction(after)
    if model.key == "cosmos3_nano":
        direction_story = "mean-margin gap narrowed because RIGHT fell more"
    elif before_side != after_side and "balanced" not in before_side + after_side:
        direction_story = "favored direction reversed"
    elif abs(after) > abs(before) + 0.01:
        direction_story = "absolute mean-margin gap widened"
    elif abs(after) + 0.01 < abs(before):
        direction_story = "absolute mean-margin gap narrowed"
    else:
        direction_story = "absolute mean-margin gap was similar"
    return (
        f"Observed: {direction_story}; total success "
        f"{model.baseline.total_successes()}/6 → {model.intervention.total_successes()}/6."
    )


def _x_for(value: float, low: float, high: float, x0: float, x1: float) -> float:
    return x0 + (value - low) / (high - low) * (x1 - x0)


def _draw_marker(
    svg: SvgBuilder,
    *,
    x: float,
    y: float,
    relation: str,
    success: bool,
    intervention: bool,
) -> None:
    color = COLORS[relation]
    fill = color if success else COLORS["card"]
    common = {
        "fill": fill,
        "stroke": color,
        "stroke_width": 2.5,
    }
    if intervention:
        svg.diamond(x, y, 8.0, **common)
    else:
        svg.circle(x, y, 7.0, **common)


def _draw_prompt_card(svg: SvgBuilder, relation: str, x: float, y: float, width: float) -> None:
    color = COLORS[relation]
    svg.rect(x, y, width, 96, rx=16, fill=COLORS["card"], stroke=COLORS["line"], stroke_width=1.2)
    svg.rect(x, y, 8, 96, rx=4, fill=color)
    svg.text(x + 28, y + 31, f"{relation.upper()} CONDITION · EXACT EPISODE-STATIC PROMPT", "overline", fill=color)
    svg.text(x + 28, y + 68, f'“{PROMPTS[relation]}”', "prompt")


def _draw_success_table(svg: SvgBuilder, model: ModelComparison, y: float) -> None:
    x = 82
    svg.text(x, y + 112, "Full task success", "section")
    svg.text(x, y + 136, "Official release-inside-cone predicate · raw counts", "small")
    svg.text(346, y + 165, "LEFT", "column", text_anchor="middle", fill=COLORS["left"])
    svg.text(438, y + 165, "RIGHT", "column", text_anchor="middle", fill=COLORS["right"])
    for row_index, configuration in enumerate((model.baseline, model.intervention)):
        row_y = y + 202 + row_index * 54
        svg.text(x, row_y + 5, configuration.short_label, "body")
        for relation, center_x in (("left", 346), ("right", 438)):
            color = COLORS[relation]
            svg.rect(center_x - 35, row_y - 22, 70, 38, rx=12, fill=COLORS["background"], stroke=color, stroke_width=1.6)
            svg.text(center_x, row_y + 4, f"{configuration.successes(relation)}/3", "count", text_anchor="middle", fill=color)
    svg.text(x, y + 297, "n = 3 matched seeds per direction and setting", "small")


def _draw_imbalance(svg: SvgBuilder, model: ModelComparison, y: float) -> None:
    x = 82
    before = model.baseline.signed_direction_gap()
    after = model.intervention.signed_direction_gap()
    svg.text(x, y + 343, "Mean directional margin gap", "section")
    svg.text(x, y + 368, "RIGHT mean − LEFT mean; sign identifies the larger margin", "small")
    svg.text(x, y + 404, model.baseline.short_label, "body")
    svg.text(238, y + 404, _format_signed_m(before), "metric", fill=COLORS["ink"])
    svg.text(356, y + 404, _favored_direction(before), "small")
    svg.text(x, y + 438, model.intervention.short_label, "body")
    svg.text(238, y + 438, _format_signed_m(after), "metric", fill=COLORS["ink"])
    svg.text(356, y + 438, _favored_direction(after), "small")
    svg.text(x, y + 474, f"Absolute mean-margin gap: {abs(before):.3f} → {abs(after):.3f} m", "small-emphasis")
    svg.text(x, y + 503, _observed_story(model), "story")


def _draw_margin_plot(
    svg: SvgBuilder,
    model: ModelComparison,
    panel_y: float,
    low: float,
    high: float,
    step: float,
) -> None:
    label_x = 530
    seed_x = 610
    plot_x0, plot_x1 = 672.0, 1328.0
    value_x = 1352
    plot_top, plot_bottom = panel_y + 148, panel_y + 402
    zero_x = _x_for(0.0, low, high, plot_x0, plot_x1)
    svg.text(label_x, panel_y + 112, "Requested final margin by matched seed", "section")
    svg.text(value_x, panel_y + 112, "baseline → intervention", "small", text_anchor="start")
    svg.rect(plot_x0, plot_top, zero_x - plot_x0, plot_bottom - plot_top, fill=COLORS["negative_band"])
    svg.rect(zero_x, plot_top, plot_x1 - zero_x, plot_bottom - plot_top, fill=COLORS["positive_band"])
    svg.line(zero_x, plot_top, zero_x, plot_bottom, stroke=COLORS["ink"], stroke_width=2.0)

    tick = math.ceil(low / step - 1e-9) * step
    ticks = []
    while tick <= high + 1e-9:
        ticks.append(0.0 if abs(tick) < 1e-12 else tick)
        tick += step
    for value in ticks:
        x = _x_for(value, low, high, plot_x0, plot_x1)
        if abs(value) > 1e-12:
            svg.line(x, plot_top, x, plot_bottom, stroke=COLORS["line"], stroke_width=1.0)
        svg.text(x, panel_y + 430, f"{value:+.2f}" if value else "0", "tick", text_anchor="middle")

    relation_layout = {
        "left": (panel_y + 169, (panel_y + 194, panel_y + 224, panel_y + 254)),
        "right": (panel_y + 303, (panel_y + 328, panel_y + 358, panel_y + 388)),
    }
    for relation in RELATIONS:
        title_y, row_ys = relation_layout[relation]
        svg.text(label_x, title_y, f"{relation.upper()} prompt", "relation", fill=COLORS[relation])
        svg.text(label_x + 108, title_y, f"mean {model.intervention.mean_margin(relation):+.3f} m after intervention", "small")
        for seed, row_y in zip(SEEDS, row_ys, strict=True):
            before = model.baseline.episodes[(seed, relation)]
            after = model.intervention.episodes[(seed, relation)]
            before_x = _x_for(before.margin_m, low, high, plot_x0, plot_x1)
            after_x = _x_for(after.margin_m, low, high, plot_x0, plot_x1)
            svg.text(seed_x, row_y + 5, str(seed), "seed", text_anchor="end")
            svg.group_start(role="img", aria_label=f"Seed {seed}, {relation}: baseline {before.margin_m:+.3f} metres, intervention {after.margin_m:+.3f} metres")
            svg.raw(f"<title>Seed {seed} {relation.upper()}: baseline {before.margin_m:+.3f} m; intervention {after.margin_m:+.3f} m</title>")
            svg.line(before_x, row_y - 3, after_x, row_y + 3, stroke=COLORS["muted"], stroke_width=2.0, stroke_opacity=0.65)
            _draw_marker(svg, x=before_x, y=row_y - 3, relation=relation, success=before.success, intervention=False)
            _draw_marker(svg, x=after_x, y=row_y + 3, relation=relation, success=after.success, intervention=True)
            svg.group_end()
            svg.text(value_x, row_y + 5, f"{before.margin_m:+.3f} → {after.margin_m:+.3f} m", "value")

    svg.text((plot_x0 + plot_x1) / 2, panel_y + 459, "Requested endpoint margin (m) · positive is farther into the requested side", "axis", text_anchor="middle")
    svg.text(plot_x0, panel_y + 482, "opposite / short of requested side", "zone", text_anchor="start")
    svg.text(plot_x1, panel_y + 482, "farther into requested side", "zone", text_anchor="end")


def render_svg(evidence: FigureEvidence) -> str:
    all_margins = [
        episode.margin_m
        for model in evidence.models
        for configuration in (model.baseline, model.intervention)
        for episode in configuration.episodes.values()
    ]
    low, high, step = _nice_scale(all_margins)
    svg = SvgBuilder()
    metadata = {
        "schema_version": FIGURE_SCHEMA,
        "arena": ARENA,
        "seeds": list(SEEDS),
        "exact_prompts": PROMPTS,
        "source_sha256": evidence.source_hashes,
        "statistical_scope": "descriptive_post_result_n3_per_direction_no_powered_significance_claim",
    }
    svg.raw(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" preserveAspectRatio="xMidYMid meet" '
        'style="max-width:100%;height:auto;display:block" role="img" '
        'aria-labelledby="v2a015-title v2a015-desc">'
    )
    svg.raw('<title id="v2a015-title">Guidance can redistribute directional performance without uniformly improving control</title>')
    svg.raw('<desc id="v2a015-desc">Two DROID model panels compare baseline and guidance intervention success counts, exact-seed requested endpoint margins, and the mean RIGHT-minus-LEFT margin gap. A smaller gap is not interpreted as improved balance when both margins fall.</desc>')
    svg.raw(f"<metadata>{html.escape(json.dumps(metadata, sort_keys=True, separators=(',', ':')))}</metadata>")
    svg.raw(
        "<style>"
        ".academic{font-family:'Source Sans 3','Aptos','Segoe UI',Arial,sans-serif;fill:%s}"
        ".title{font-size:38px;font-weight:600;letter-spacing:-.5px}.subtitle{font-size:18px;fill:%s}"
        ".overline{font-size:13px;font-weight:600;letter-spacing:1px}.prompt{font-size:19px;font-weight:500}"
        ".model{font-size:29px;font-weight:600}.technical{font-size:17px;fill:%s}.story{font-size:16px;font-weight:600;fill:%s}"
        ".section{font-size:17px;font-weight:600}.body{font-size:16px}.small{font-size:13px;fill:%s}.small-emphasis{font-size:14px;font-weight:600}"
        ".column{font-size:13px;font-weight:600;letter-spacing:.8px}.count{font-size:18px;font-weight:600}.metric{font-size:17px;font-weight:600}"
        ".relation{font-size:14px;font-weight:600;letter-spacing:.5px}.seed{font-size:13px;fill:%s}.value{font-size:13px;font-variant-numeric:tabular-nums}"
        ".tick{font-size:12px;fill:%s;font-variant-numeric:tabular-nums}.axis{font-size:14px;font-weight:500}.zone{font-size:12px;fill:%s}"
        ".legend{font-size:14px;fill:%s}.footer{font-size:13px;fill:%s}.footer-strong{font-size:13px;font-weight:600}"
        "</style>"
        % (
            COLORS["ink"],
            COLORS["muted"],
            COLORS["muted"],
            COLORS["accent"],
            COLORS["muted"],
            COLORS["muted"],
            COLORS["muted"],
            COLORS["muted"],
            COLORS["muted"],
            COLORS["muted"],
        )
    )
    svg.rect(0, 0, WIDTH, HEIGHT, fill=COLORS["background"])
    svg.group_start(class_="academic")
    svg.text(64, 64, "Guidance can redistribute directional performance", "title")
    svg.text(64, 108, "without uniformly improving control", "title")
    svg.text(64, 140, "V2-A015 exploratory DROID/RoboLab ablation · same reset, task predicate, static prompt, and seeds 8300–8302 within each model", "subtitle")
    _draw_prompt_card(svg, "left", 64, 164, 724)
    _draw_prompt_card(svg, "right", 812, 164, 724)

    legend_y = 290
    _draw_marker(svg, x=74, y=legend_y, relation="right", success=True, intervention=False)
    svg.text(91, legend_y + 5, "circle = baseline", "legend")
    _draw_marker(svg, x=235, y=legend_y, relation="right", success=True, intervention=True)
    svg.text(252, legend_y + 5, "diamond = intervention", "legend")
    _draw_marker(svg, x=459, y=legend_y, relation="left", success=True, intervention=False)
    _draw_marker(svg, x=484, y=legend_y, relation="left", success=False, intervention=False)
    svg.text(501, legend_y + 5, "filled = full task success · hollow = failure", "legend")
    svg.line(846, legend_y, 886, legend_y, stroke=COLORS["muted"], stroke_width=2.0)
    svg.text(898, legend_y + 5, "line = same seed", "legend")
    svg.text(1130, legend_y + 5, "All counts: n = 3 per direction", "legend")

    for model, panel_y in zip(evidence.models, (326.0, 866.0), strict=True):
        svg.rect(48, panel_y, 1504, 516, rx=20, fill=COLORS["card"], stroke=COLORS["line"], stroke_width=1.4)
        svg.text(80, panel_y + 42, model.display_name, "model")
        svg.text(80, panel_y + 69, model.technical_description, "technical")
        svg.text(1518, panel_y + 42, _observed_story(model), "story", text_anchor="end")
        svg.line(80, panel_y + 84, 1520, panel_y + 84, stroke=COLORS["line"], stroke_width=1.2)
        svg.line(504, panel_y + 104, 504, panel_y + 488, stroke=COLORS["line"], stroke_width=1.2)
        _draw_success_table(svg, model, panel_y)
        _draw_imbalance(svg, model, panel_y)
        _draw_margin_plot(svg, model, panel_y, low, high, step)

    footer_y = 1406
    svg.line(64, footer_y, 1536, footer_y, stroke=COLORS["line"], stroke_width=1.4)
    svg.text(64, footer_y + 30, "How to read the margin", "footer-strong")
    svg.text(240, footer_y + 30, "LEFT uses −final lateral coordinate; RIGHT uses +final lateral coordinate. Positive means the endpoint lies on the requested side.", "footer")
    svg.text(64, footer_y + 54, "Success remains stricter", "footer-strong")
    svg.text(240, footer_y + 54, "A positive endpoint margin is not automatically a success; the full task also requires the official release-inside-45°-cone predicate.", "footer")
    svg.text(64, footer_y + 78, "Inference boundary", "footer-strong")
    svg.text(240, footer_y + 78, "Post-result descriptive pilot: three matched seeds per direction and setting. Raw counts and paired effects are shown without CI, p-value, or powered significance claim.", "footer")
    svg.text(64, footer_y + 102, "DreamZero naming", "footer-strong")
    svg.text(240, footer_y + 102, "s=2 is derived CFG-style negative-branch action guidance using the released visual-quality negative prompt—not an official action-CFG feature. DROID only; no RoboTwin pooling.", "footer")
    svg.group_end()
    svg.raw("</svg>")
    return svg.build()


def _write_text(path: Path, content: str, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        _fail(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_png_if_supported(svg_path: Path, png_path: Path, *, overwrite: bool) -> str | None:
    if png_path.exists() and not overwrite:
        _fail(f"Refusing to overwrite existing output: {png_path}")
    renderer = shutil.which("rsvg-convert")
    if renderer is None:
        return None
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{png_path.name}.", suffix=".tmp.png", dir=png_path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        completed = subprocess.run(
            [renderer, "--format=png", f"--width={WIDTH}", "--output", str(temporary), str(svg_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            _fail(f"rsvg-convert failed: {completed.stderr.strip()}")
        os.replace(temporary, png_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "rsvg-convert"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cosmos-baseline", type=Path, required=True)
    parser.add_argument("--cosmos-intervention", type=Path, required=True)
    parser.add_argument("--dreamzero-baseline", type=Path, required=True)
    parser.add_argument("--dreamzero-intervention", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--svg-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for key in (
        "cosmos_baseline",
        "cosmos_intervention",
        "dreamzero_baseline",
        "dreamzero_intervention",
        "comparison",
        "svg_output",
        "png_output",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())

    evidence = load_evidence(
        cosmos_baseline=args.cosmos_baseline,
        cosmos_intervention=args.cosmos_intervention,
        dreamzero_baseline=args.dreamzero_baseline,
        dreamzero_intervention=args.dreamzero_intervention,
        comparison_path=args.comparison,
    )
    _write_text(args.svg_output, render_svg(evidence), overwrite=args.overwrite)
    png_renderer = None
    if args.png_output is not None:
        png_renderer = render_png_if_supported(args.svg_output, args.png_output, overwrite=args.overwrite)
    result = {
        "status": "complete",
        "schema_version": FIGURE_SCHEMA,
        "svg": {
            "path": str(args.svg_output),
            "bytes": args.svg_output.stat().st_size,
            "sha256": sha256(args.svg_output),
        },
        "png": (
            {
                "status": "complete",
                "renderer": png_renderer,
                "path": str(args.png_output),
                "bytes": args.png_output.stat().st_size,
                "sha256": sha256(args.png_output),
            }
            if args.png_output is not None and png_renderer is not None
            else {
                "status": "not_requested" if args.png_output is None else "renderer_unavailable"
            }
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
