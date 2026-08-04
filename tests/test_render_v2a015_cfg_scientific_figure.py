from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import render_v2a015_cfg_scientific_figure as figure  # noqa: E402


MARGINS = {
    "cosmos_baseline": {
        "left": (0.10, 0.20, 0.15),
        "right": (0.30, 0.25, 0.35),
    },
    "cosmos_intervention": {
        "left": (0.12, 0.10, 0.14),
        "right": (0.22, 0.20, 0.24),
    },
    "dreamzero_baseline": {
        "left": (0.10, 0.08, 0.12),
        "right": (0.04, -0.02, 0.01),
    },
    "dreamzero_intervention": {
        "left": (-0.01, 0.01, 0.15),
        "right": (0.12, 0.31, 0.22),
    },
}

SUCCESSES = {
    "cosmos_baseline": {
        "left": (True, True, True),
        "right": (True, True, True),
    },
    "cosmos_intervention": {
        "left": (True, False, True),
        "right": (True, True, True),
    },
    "dreamzero_baseline": {
        "left": (True, False, True),
        "right": (True, False, False),
    },
    "dreamzero_intervention": {
        "left": (False, False, True),
        "right": (True, True, True),
    },
}


def _episodes(key: str) -> list[dict[str, object]]:
    rows = []
    for relation in figure.RELATIONS:
        for seed, margin, success in zip(
            figure.SEEDS,
            MARGINS[key][relation],
            SUCCESSES[key][relation],
            strict=True,
        ):
            final = -margin if relation == "left" else margin
            row: dict[str, object] = {
                "environment_seed": seed,
                "sampling_seed": seed,
                "requested_relation": relation,
                "prompt": figure.PROMPTS[relation],
                "requested_success": success,
                "final_lateral_display_m": final,
            }
            if key.endswith("intervention"):
                row["requested_signed_final_margin_m"] = margin
            elif key == "cosmos_baseline":
                row["requested_signed_final_offset_m"] = margin
            rows.append(row)
    return sorted(rows, key=lambda row: (int(row["environment_seed"]), str(row["requested_relation"])))


def _source_payload(key: str) -> dict[str, object]:
    spec = figure.RESULT_SPECS[key]
    rows = _episodes(key)
    by_direction = {}
    for relation in figure.RELATIONS:
        selected = [row for row in rows if row["requested_relation"] == relation]
        by_direction[relation] = {
            "prompt": figure.PROMPTS[relation],
            "episodes": 3,
            "successes": sum(bool(row["requested_success"]) for row in selected),
        }
    total = sum(record["successes"] for record in by_direction.values())
    payload: dict[str, object] = {
        "schema_version": spec.schema,
        "model_id": spec.model_id,
        "amendment_id": spec.amendment_id,
        "status": "complete",
        "episodes": rows,
    }
    if key == "dreamzero_baseline":
        payload.update(
            {
                "valid_episode_count": 6,
                "requested_success_count": total,
                "success_by_relation": {
                    relation: {
                        "successes": by_direction[relation]["successes"],
                        "trials": 3,
                    }
                    for relation in figure.RELATIONS
                },
            }
        )
    elif key == "cosmos_baseline":
        payload["summary"] = {
            "episode_count": 6,
            "successes": total,
            "by_direction": by_direction,
        }
    else:
        payload.update(
            {
                "arena": figure.ARENA,
                "exact_prompts": figure.PROMPTS,
                "summary": {
                    "valid_episode_count": 6,
                    "requested_success_count": total,
                    "by_direction": by_direction,
                },
            }
        )
    return payload


def _configuration_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    by_direction = {}
    means = {}
    for relation in figure.RELATIONS:
        selected = [row for row in rows if row["requested_relation"] == relation]
        margins = [
            -float(row["final_lateral_display_m"])
            if relation == "left"
            else float(row["final_lateral_display_m"])
            for row in selected
        ]
        mean = sum(margins) / 3
        means[relation] = mean
        by_direction[relation] = {
            "prompt": figure.PROMPTS[relation],
            "episodes": 3,
            "successes": sum(bool(row["requested_success"]) for row in selected),
            "requested_margin_m": {
                "values_by_seed": [
                    {"environment_seed": row["environment_seed"], "value": margin}
                    for row, margin in zip(selected, margins, strict=True)
                ],
                "mean": mean,
                "median": sorted(margins)[1],
                "minimum": min(margins),
                "maximum": max(margins),
            },
        }
    gap = means["right"] - means["left"]
    return {
        "valid_episode_count": 6,
        "valid_failure_count": sum(not bool(row["requested_success"]) for row in rows),
        "requested_success_count": sum(bool(row["requested_success"]) for row in rows),
        "by_direction": by_direction,
        "mean_margin_balance": {
            "right_minus_left_m": gap,
            "absolute_direction_imbalance_m": abs(gap),
            "weaker_direction_mean_margin_m": min(means.values()),
        },
    }


def _transition(before: bool, after: bool) -> str:
    if before and after:
        return "unchanged_success"
    if before and not after:
        return "regressed_success_to_failure"
    if not before and after:
        return "improved_failure_to_success"
    return "unchanged_failure"


def _comparison_model(
    *,
    model: str,
    baseline_label: str,
    intervention_label: str,
    baseline_key: str,
    intervention_key: str,
) -> dict[str, object]:
    before_rows = _episodes(baseline_key)
    after_rows = _episodes(intervention_key)
    before = {(int(row["environment_seed"]), str(row["requested_relation"])): row for row in before_rows}
    after = {(int(row["environment_seed"]), str(row["requested_relation"])): row for row in after_rows}
    cells = []
    transitions: Counter[str] = Counter()
    by_relation = {relation: Counter() for relation in figure.RELATIONS}
    for seed in figure.SEEDS:
        for relation in figure.RELATIONS:
            old = before[(seed, relation)]
            new = after[(seed, relation)]
            old_margin = -float(old["final_lateral_display_m"]) if relation == "left" else float(old["final_lateral_display_m"])
            new_margin = -float(new["final_lateral_display_m"]) if relation == "left" else float(new["final_lateral_display_m"])
            transition = _transition(bool(old["requested_success"]), bool(new["requested_success"]))
            transitions[transition] += 1
            by_relation[relation][transition] += 1
            cells.append(
                {
                    "cell_id": f"seed{seed}_{relation}",
                    "environment_seed": seed,
                    "requested_relation": relation,
                    "prompt": figure.PROMPTS[relation],
                    "success": {
                        baseline_label: old["requested_success"],
                        intervention_label: new["requested_success"],
                        "transition": transition,
                        "numeric_delta": int(bool(new["requested_success"])) - int(bool(old["requested_success"])),
                    },
                    "requested_signed_final_margin_m": {
                        baseline_label: old_margin,
                        intervention_label: new_margin,
                        "intervention_minus_baseline": new_margin - old_margin,
                    },
                }
            )
    before_total = sum(bool(row["requested_success"]) for row in before_rows)
    after_total = sum(bool(row["requested_success"]) for row in after_rows)
    return {
        "model": model,
        "baseline_label": baseline_label,
        "intervention_label": intervention_label,
        "exact_prompts": figure.PROMPTS,
        "success": {
            "baseline_total": before_total,
            "intervention_total": after_total,
            "net_success_change": after_total - before_total,
            "exact_paired_transitions": dict(sorted(transitions.items())),
            "exact_paired_transitions_by_relation": {
                relation: dict(sorted(by_relation[relation].items()))
                for relation in figure.RELATIONS
            },
        },
        "baseline_configuration_summary": _configuration_summary(before_rows),
        "intervention_configuration_summary": _configuration_summary(after_rows),
        "paired_seed_diagnostics": [
            {
                "pair_id": f"droid_pair_seed_{seed}",
                "environment_seed": seed,
                "left_prompt": figure.PROMPTS["left"],
                "right_prompt": figure.PROMPTS["right"],
            }
            for seed in figure.SEEDS
        ],
        "cells": cells,
    }


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": figure.sha256(path),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(root: Path) -> dict[str, Path]:
    paths = {
        key: root / f"{key}.json"
        for key in figure.RESULT_SPECS
    }
    for key, path in paths.items():
        _write_json(path, _source_payload(key))
    comparison_path = root / "comparison.json"
    comparison = {
        "schema_version": figure.COMPARISON_SCHEMA,
        "status": "complete",
        "amendment_id": "V2-A015",
        "arena": figure.ARENA,
        "exact_prompts": figure.PROMPTS,
        "comparisons": {
            "cosmos3_nano": _comparison_model(
                model="Cosmos3 Nano Policy DROID",
                baseline_label="g=3 baseline",
                intervention_label="g=1 intervention",
                baseline_key="cosmos_baseline",
                intervention_key="cosmos_intervention",
            ),
            "dreamzero": _comparison_model(
                model="DreamZero DROID",
                baseline_label="s=1 conditional-action equivalent",
                intervention_label="s=2 CFG-style negative-branch action guidance",
                baseline_key="dreamzero_baseline",
                intervention_key="dreamzero_intervention",
            ),
        },
        "provenance": {
            "cosmos3_nano_baseline": _file_record(paths["cosmos_baseline"]),
            "cosmos3_nano_intervention": _file_record(paths["cosmos_intervention"]),
            "dreamzero_baseline": _file_record(paths["dreamzero_baseline"]),
            "dreamzero_intervention": _file_record(paths["dreamzero_intervention"]),
        },
    }
    _write_json(comparison_path, comparison)
    paths["comparison"] = comparison_path
    return paths


def _load(paths: dict[str, Path]) -> figure.FigureEvidence:
    return figure.load_evidence(
        cosmos_baseline=paths["cosmos_baseline"],
        cosmos_intervention=paths["cosmos_intervention"],
        dreamzero_baseline=paths["dreamzero_baseline"],
        dreamzero_intervention=paths["dreamzero_intervention"],
        comparison_path=paths["comparison"],
    )


class V2A015FigureTest(unittest.TestCase):
    def test_deterministic_responsive_svg_contains_scientific_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            evidence = _load(paths)
            first = figure.render_svg(evidence)
            second = figure.render_svg(evidence)
            self.assertEqual(first, second)
            root = ET.fromstring(first)
            self.assertEqual(root.attrib["viewBox"], f"0 0 {figure.WIDTH} {figure.HEIGHT}")
            self.assertIn("max-width:100%;height:auto", root.attrib["style"])
            rendered_text = " ".join(root.itertext())
            self.assertIn(figure.PROMPTS["left"], rendered_text)
            self.assertIn(figure.PROMPTS["right"], rendered_text)
            self.assertIn("n = 3 matched seeds per direction and setting", first)
            self.assertIn("without CI, p-value, or powered significance claim", first)
            self.assertIn("not automatically a success", first)
            self.assertIn("favored direction reversed", first)
            self.assertNotIn("foreignObject", first)

    def test_prompt_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            payload = json.loads(paths["dreamzero_intervention"].read_text(encoding="utf-8"))
            payload["episodes"][0]["prompt"] = "Put it left."
            _write_json(paths["dreamzero_intervention"], payload)
            with self.assertRaisesRegex(RuntimeError, "exact prompt bytes changed"):
                _load(paths)

    def test_seed_drift_and_schema_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            payload = json.loads(paths["cosmos_baseline"].read_text(encoding="utf-8"))
            payload["episodes"][0]["environment_seed"] = 9999
            _write_json(paths["cosmos_baseline"], payload)
            with self.assertRaisesRegex(RuntimeError, "unauthorized seed"):
                _load(paths)
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            payload = json.loads(paths["comparison"].read_text(encoding="utf-8"))
            payload["schema_version"] = "unexpected"
            _write_json(paths["comparison"], payload)
            with self.assertRaisesRegex(RuntimeError, "comparison schema changed"):
                _load(paths)

    def test_comparison_source_hash_binding_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _fixture(Path(temporary))
            comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
            comparison["provenance"]["dreamzero_baseline"]["sha256"] = "0" * 64
            _write_json(paths["comparison"], comparison)
            with self.assertRaisesRegex(RuntimeError, "hash does not bind"):
                _load(paths)

    @unittest.skipUnless(shutil.which("rsvg-convert"), "rsvg-convert is unavailable")
    def test_optional_png_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _fixture(root)
            svg_path = root / "figure.svg"
            png_path = root / "figure.png"
            figure._write_text(svg_path, figure.render_svg(_load(paths)), overwrite=False)
            renderer = figure.render_png_if_supported(svg_path, png_path, overwrite=False)
            self.assertEqual(renderer, "rsvg-convert")
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
