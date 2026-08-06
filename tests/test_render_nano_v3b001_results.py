import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import render_nano_v3b001_results as renderer  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _sign_test(values: list[float]) -> dict[str, object]:
    positive = sum(value > 0.0 for value in values)
    negative = sum(value < 0.0 for value in values)
    ties = len(values) - positive - negative
    effective = positive + negative
    if effective == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(effective, index)
            for index in range(min(positive, negative) + 1)
        )
        p_value = min(1.0, 2.0 * tail / (2**effective))
    return {
        "method": "exact_two_sided_paired_sign_test",
        "null": "positive and negative paired differences are equally probable",
        "positive": positive,
        "negative": negative,
        "ties_excluded": ties,
        "effective_n": effective,
        "p_value": p_value,
    }


def _metric(values: list[float], replicates: int = 1000) -> dict[str, object]:
    ordered = sorted(values)
    mean = statistics.fmean(values)
    median = float(statistics.median(values))
    # The renderer validates the frozen interval contract and bounds. Synthetic
    # fixtures use the observed range as deterministic, conservative bounds.
    interval_common = {
        "method": "matched_seed_nonparametric_percentile_bootstrap",
        "unit_of_resampling": "matched_seed",
        "confidence": 0.95,
        "replicates": replicates,
        "seed": 314159,
    }
    return {
        "n": len(values),
        "mean_m": mean,
        "median_m": median,
        "sample_standard_deviation_m": statistics.stdev(values),
        "minimum_m": min(values),
        "maximum_m": max(values),
        "mean_bootstrap_95": {
            **interval_common,
            "statistic": "mean",
            "lower": min(values),
            "upper": max(values),
        },
        "median_bootstrap_95": {
            **interval_common,
            "statistic": "median",
            "lower": ordered[1],
            "upper": ordered[-2],
        },
        "median_exact_interval": {
            "method": "exact_distribution_free_order_statistic_interval",
            "requested_confidence": 0.95,
            "achieved_confidence": 0.95,
            "lower_order_statistic": 2,
            "upper_order_statistic": len(values) - 1,
            "lower": ordered[1],
            "upper": ordered[-2],
        },
        "paired_sign_test": _sign_test(values),
    }


def _offset(seed: int, arm: str, relation: str) -> float:
    delta = (seed - 9413) * 0.001
    values = {
        ("control", "left"): 0.100 + delta,
        ("control", "right"): -(0.160 + 0.5 * delta),
        ("position_mirrored", "left"): 0.120 + 0.8 * delta,
        ("position_mirrored", "right"): -(0.240 + 0.4 * delta),
    }
    return values[(arm, relation)]


def _taxonomy(seed_index: int, arm: str, relation: str) -> str:
    # Force one failure into the deterministically selected lowest-seed block,
    # so the fixture proves that media selection is not outcome-filtered.
    if seed_index == 0 and arm == "control" and relation == "left":
        return "wrong_side"
    success_threshold = {
        ("control", "left"): 20,
        ("control", "right"): 24,
        ("position_mirrored", "left"): 21,
        ("position_mirrored", "right"): 25,
    }[(arm, relation)]
    if seed_index < success_threshold:
        return "correct"
    failures = (
        "pick_failed",
        "transport_failed",
        "wrong_side",
        "release_failed",
    )
    return failures[(seed_index + renderer.ARMS.index(arm) + renderer.RELATIONS.index(relation)) % 4]


class SyntheticEvidence:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.assets = root / "assets"
        self.assets.mkdir(parents=True)
        self.actual_assets: dict[str, Path] = {}
        self.prediction_assets: dict[tuple[str, int], Path] = {}
        self.episodes_path = root / "nano_v3b001_episodes.jsonl"
        self.summary_path = root / "nano_v3b001_summary.json"
        self.rows = self._rows()
        self._write_episodes(self.rows)
        self.summary = self._summary(self.rows)
        _write_json(self.summary_path, self.summary)

    def _selected_file_record(
        self,
        *,
        seed: int,
        cell_id: str,
        media_kind: str,
    ) -> dict[str, object]:
        if seed != min(renderer.SEEDS):
            return {
                "path": f"pvc://synthetic/{cell_id}/{media_kind}.mp4",
                "sha256": hashlib.sha256(f"{cell_id}:{media_kind}".encode()).hexdigest(),
                "bytes": 113,
            }
        path = self.assets / f"{cell_id.replace(':', '_')}-{media_kind}.bin"
        path.write_bytes(f"synthetic {cell_id} {media_kind}\n".encode())
        if media_kind == "actual":
            self.actual_assets[cell_id] = path
        else:
            self.prediction_assets[(cell_id, 0)] = path
        return _file_record(path)

    def _rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for seed_index, seed in enumerate(renderer.SEEDS):
            for arm in renderer.ARMS:
                for relation in renderer.RELATIONS:
                    cell_id = renderer._expected_cell_id(seed, arm, relation)
                    signed = _offset(seed, arm, relation)
                    margin = signed if relation == "left" else -signed
                    taxonomy = _taxonomy(seed_index, arm, relation)
                    rows.append(
                        {
                            "schema_version": renderer.BEHAVIORAL_SCHEMA,
                            "record_type": "behavioral_episode",
                            "behavioral_result_valid": True,
                            "study_id": renderer.STUDY_ID,
                            "model_id": renderer.MODEL_ID,
                            "arena": renderer.ARENA,
                            "amendment_id": renderer.AMENDMENT_ID,
                            "registered_cell_id": cell_id,
                            "pair_id": f"v3b001:nano:seed{seed}",
                            "environment_seed": seed,
                            "policy_seed": seed,
                            "phase_b_arm": arm,
                            "requested_relation": relation,
                            "prompt_family": "direct_command",
                            "prompt": renderer.PROMPTS[relation],
                            "requested_success": taxonomy == "correct",
                            "failure_taxonomy": taxonomy,
                            "future_interface": "decoded_rgb_uint8_33_frames_per_policy_request",
                            "missing_future_policy": "infrastructure_invalid_never_zero",
                            "measurements": {
                                "signed_final_lateral_offset_m": signed,
                                "final_requested_signed_margin_m": margin,
                            },
                            "artifacts": {
                                "viewport_video": self._selected_file_record(
                                    seed=seed,
                                    cell_id=cell_id,
                                    media_kind="actual",
                                )
                            },
                            "future_requests": [
                                {
                                    "request_index": 0,
                                    "action_step_start": 0,
                                    "decoded_future": self._selected_file_record(
                                        seed=seed,
                                        cell_id=cell_id,
                                        media_kind="prediction",
                                    ),
                                    "decoded_future_shape": [33, 8, 8, 3],
                                    "future_evidence_status": "exposed_and_retained",
                                }
                            ],
                        }
                    )
        return rows

    def _write_episodes(self, rows: list[dict[str, object]]) -> None:
        payload = b"".join(
            (
                json.dumps(
                    row,
                    allow_nan=False,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8")
            for row in rows
        )
        self.episodes_path.write_bytes(payload)

    def _summary(self, rows: list[dict[str, object]]) -> dict[str, object]:
        by_id = {str(row["registered_cell_id"]): row for row in rows}
        condition_outcomes: dict[str, object] = {}
        all_taxonomy: Counter[str] = Counter()
        for arm in renderer.ARMS:
            for relation in renderer.RELATIONS:
                cells = [
                    by_id[renderer._expected_cell_id(seed, arm, relation)]
                    for seed in renderer.SEEDS
                ]
                counts = Counter(str(row["failure_taxonomy"]) for row in cells)
                all_taxonomy.update(counts)
                condition_outcomes[f"{arm}:{relation}"] = {
                    "episodes": 27,
                    "successes": sum(bool(row["requested_success"]) for row in cells),
                    "failure_taxonomy_counts": dict(sorted(counts.items())),
                }

        vectors = {
            "D_control": [],
            "D_position_mirrored": [],
            "B_control": [],
            "B_position_mirrored": [],
            "I": [],
            "J": [],
        }
        seed_level: list[dict[str, object]] = []
        complete: list[dict[str, float | int]] = []
        for seed in renderer.SEEDS:
            s = {
                (arm, relation): float(
                    by_id[renderer._expected_cell_id(seed, arm, relation)]["measurements"][  # type: ignore[index]
                        "signed_final_lateral_offset_m"
                    ]
                )
                for arm in renderer.ARMS
                for relation in renderer.RELATIONS
            }
            d_control = s[("control", "left")] - s[("control", "right")]
            d_reflected = s[("position_mirrored", "left")] - s[("position_mirrored", "right")]
            b_control = -s[("control", "right")] - s[("control", "left")]
            b_reflected = -s[("position_mirrored", "right")] - s[("position_mirrored", "left")]
            i_value = b_reflected - b_control
            j_value = d_reflected - d_control
            values = {
                "D_control_m": d_control,
                "D_position_mirrored_m": d_reflected,
                "B_control_m": b_control,
                "B_position_mirrored_m": b_reflected,
                "I_position_reflection_interaction_m": i_value,
                "J_redirection_interaction_m": j_value,
            }
            vectors["D_control"].append(d_control)
            vectors["D_position_mirrored"].append(d_reflected)
            vectors["B_control"].append(b_control)
            vectors["B_position_mirrored"].append(b_reflected)
            vectors["I"].append(i_value)
            vectors["J"].append(j_value)
            seed_row: dict[str, object] = {
                "seed": seed,
                "matched_block_id": f"v3b001:nano:seed{seed}",
                "full_sample": values,
            }
            cells = [
                by_id[renderer._expected_cell_id(seed, arm, relation)]
                for arm in renderer.ARMS
                for relation in renderer.RELATIONS
            ]
            if all(bool(cell["requested_success"]) for cell in cells):
                margins = {
                    (arm, relation): float(
                        by_id[renderer._expected_cell_id(seed, arm, relation)]["measurements"][  # type: ignore[index]
                            "final_requested_signed_margin_m"
                        ]
                    )
                    for arm in renderer.ARMS
                    for relation in renderer.RELATIONS
                }
                g_control = margins[("control", "right")] - margins[("control", "left")]
                g_reflected = margins[("position_mirrored", "right")] - margins[("position_mirrored", "left")]
                success_row = {
                    "seed": seed,
                    "G_control_m": g_control,
                    "G_position_mirrored_m": g_reflected,
                    "G_position_reflection_interaction_m": g_reflected - g_control,
                }
                seed_row["success_conditional_secondary"] = success_row
                complete.append(success_row)
            seed_level.append(seed_row)

        return {
            "schema_version": renderer.SUMMARY_SCHEMA,
            "study_id": renderer.STUDY_ID,
            "amendment_id": renderer.AMENDMENT_ID,
            "model_id": renderer.MODEL_ID,
            "arena": renderer.ARENA,
            "claim_boundary": (
                "Prespecified DROID/RoboLab position-reflection ablation only; "
                "the result does not identify training-distribution causality."
            ),
            "exact_prompts": renderer.PROMPTS,
            "behavioral_evidence": {
                "valid_episode_count": 108,
                "matched_seed_count": 27,
                "aggregate_jsonl": {
                    "path": self.episodes_path.name,
                    "sha256": _sha256(self.episodes_path),
                    "bytes": self.episodes_path.stat().st_size,
                },
            },
            "uncertainty_contract": {
                "unit": "matched_seed",
                "bootstrap": "deterministic paired nonparametric percentile",
                "bootstrap_replicates": 1000,
                "bootstrap_master_seed": 20260806,
                "robust_test": "exact two-sided paired sign test with zero ties excluded",
                "median_interval": "exact distribution-free order-statistic interval",
                "multiplicity_adjustment": "none; estimands are prespecified and reported separately",
            },
            "condition_outcomes": condition_outcomes,
            "failure_taxonomy_counts": dict(sorted(all_taxonomy.items())),
            "full_sample_primary": {
                "population": {
                    "matched_seed_count": 27,
                    "behavioral_episode_count": 108,
                    "valid_failures_included": True,
                    "infrastructure_attempts_included": False,
                    "missing_value_imputation": "none",
                },
                "formulas": renderer.FORMULAS,
                "interpretation": renderer.INTERPRETATIONS,
                "D_by_arm": {
                    "control": _metric(vectors["D_control"]),
                    "position_mirrored": _metric(vectors["D_position_mirrored"]),
                },
                "B_by_arm": {
                    "control": _metric(vectors["B_control"]),
                    "position_mirrored": _metric(vectors["B_position_mirrored"]),
                },
                "I_position_reflection_interaction": _metric(vectors["I"]),
                "J_redirection_interaction": _metric(vectors["J"]),
            },
            "success_conditional_secondary": {
                "subset_id": "nano_v3b001_all_four_cells_correct",
                "inclusion_rule": "all four cells satisfy the frozen requested-success predicate",
                "realized_matched_seed_count": len(complete),
                "included_seeds": [int(row["seed"]) for row in complete],
                "failures_as_zero": False,
                "unmatched_successful_cells_used": False,
                "seed_level": complete,
            },
            "seed_level": seed_level,
        }

    def rewrite_summary_binding(self) -> None:
        summary = copy.deepcopy(self.summary)
        aggregate = summary["behavioral_evidence"]["aggregate_jsonl"]  # type: ignore[index]
        aggregate["sha256"] = _sha256(self.episodes_path)
        aggregate["bytes"] = self.episodes_path.stat().st_size
        self.summary = summary
        _write_json(self.summary_path, summary)


class NanoV3B001RendererTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.fixture = SyntheticEvidence(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_load_evidence_recomputes_exact_grid_and_estimands(self) -> None:
        evidence = renderer.load_evidence(
            self.fixture.summary_path,
            self.fixture.episodes_path,
        )
        self.assertEqual(len(evidence.ordered_rows), 108)
        self.assertEqual(len(evidence.rows_by_id), 108)
        self.assertEqual(evidence.selected_seed, 9400)
        self.assertEqual(len(evidence.selected_rows), 4)
        self.assertEqual(evidence.metrics["I"].summary["n"], 27)
        self.assertEqual(evidence.metrics["J"].summary["n"], 27)
        self.assertEqual(sum(evidence.taxonomy_counts.values()), 108)

    @unittest.skipUnless(shutil.which("rsvg-convert"), "rsvg-convert is required")
    def test_render_is_deterministic_readable_and_hash_bearing(self) -> None:
        first = renderer.render_nano_v3b001_results(
            summary_path=self.fixture.summary_path,
            episodes_path=self.fixture.episodes_path,
            output_directory=self.root / "first",
        )
        second = renderer.render_nano_v3b001_results(
            summary_path=self.fixture.summary_path,
            episodes_path=self.fixture.episodes_path,
            output_directory=self.root / "second",
        )
        self.assertEqual(set(first), set(renderer.OUTPUT_NAMES))
        for key in renderer.OUTPUT_NAMES:
            self.assertEqual(first[key].read_bytes(), second[key].read_bytes(), key)

        primary_root = ET.fromstring(first["primary_svg"].read_text(encoding="utf-8"))
        taxonomy_root = ET.fromstring(first["taxonomy_svg"].read_text(encoding="utf-8"))
        self.assertEqual(primary_root.attrib["viewBox"], "0 0 1440 1920")
        self.assertIn("max-width:100%", primary_root.attrib["style"])
        self.assertEqual(taxonomy_root.attrib["viewBox"], "0 0 1440 1080")
        primary_text = " ".join(primary_root.itertext())
        taxonomy_text = " ".join(taxonomy_root.itertext())
        for prompt in renderer.PROMPTS.values():
            self.assertIn(prompt, primary_text)
            self.assertIn(prompt, taxonomy_text)
        for label in (
            "D · endpoint redirection",
            "B · requested-depth contrast",
            "J · reflection interaction on D",
            "I · reflection interaction on B",
            "ACTUAL SIMULATOR ROLLOUT",
            "DECODED LOCAL PREDICTION",
        ):
            self.assertIn(label, primary_text)
        for taxonomy in renderer.FAILURE_CLASSES:
            self.assertIn(taxonomy.replace("_", " "), taxonomy_text)

        for key in ("primary_png", "taxonomy_png"):
            payload = first[key].read_bytes()[:24]
            self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
            width, height = struct.unpack(">II", payload[16:24])
            self.assertEqual(width, renderer.PNG_WIDTH)
            self.assertGreater(height, 1000)

        manifest = json.loads(first["manifest"].read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], renderer.MANIFEST_SCHEMA)
        self.assertEqual(manifest["selection"]["selected_seed"], 9400)
        self.assertFalse(manifest["selection"]["outcome_used_for_selection"])
        self.assertEqual(manifest["selection"]["selected_cell_count"], 4)
        self.assertEqual(manifest["selection"]["selected_decoded_prediction_count"], 4)
        self.assertEqual(
            manifest["selection"]["no_outcome_hiding_audit"]["selected_failures"],
            1,
        )
        self.assertEqual(len(manifest["selected_media"]), 4)
        for record in manifest["generated_files"].values():
            generated = first["primary_svg"].parent / record["path"]
            self.assertEqual(record["sha256"], _sha256(generated))
            self.assertEqual(record["bytes"], generated.stat().st_size)
        for selected in manifest["selected_media"]:
            self.assertIn(selected["exact_prompt"], renderer.PROMPTS.values())
            self.assertEqual(
                selected["actual_rollout"]["label"],
                "ACTUAL SIMULATOR ROLLOUT — executed robot behavior",
            )
            self.assertEqual(len(selected["decoded_local_predictions"]), 1)
            self.assertIn("not execution", selected["decoded_local_predictions"][0]["label"])

    def test_source_hash_and_exact_prompt_drift_fail_closed(self) -> None:
        bad_summary = copy.deepcopy(self.fixture.summary)
        bad_summary["behavioral_evidence"]["aggregate_jsonl"]["sha256"] = "0" * 64
        _write_json(self.fixture.summary_path, bad_summary)
        with self.assertRaisesRegex(renderer.NanoResultRenderError, "SHA-256"):
            renderer.load_evidence(self.fixture.summary_path, self.fixture.episodes_path)

        _write_json(self.fixture.summary_path, self.fixture.summary)
        changed_rows = copy.deepcopy(self.fixture.rows)
        changed_rows[0]["prompt"] = "Put the Rubik's cube somewhere else."
        self.fixture._write_episodes(changed_rows)
        self.fixture.rewrite_summary_binding()
        with self.assertRaisesRegex(renderer.NanoResultRenderError, "exact prompt bytes"):
            renderer.load_evidence(self.fixture.summary_path, self.fixture.episodes_path)

    @unittest.skipUnless(shutil.which("rsvg-convert"), "rsvg-convert is required")
    def test_nonempty_output_is_never_overwritten(self) -> None:
        output = self.root / "occupied"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("user data\n", encoding="utf-8")
        with self.assertRaisesRegex(renderer.NanoResultRenderError, "must be empty"):
            renderer.render_nano_v3b001_results(
                summary_path=self.fixture.summary_path,
                episodes_path=self.fixture.episodes_path,
                output_directory=output,
            )
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "user data\n")
        self.assertEqual(list(output.iterdir()), [sentinel])

    @unittest.skipUnless(shutil.which("rsvg-convert"), "rsvg-convert is required")
    def test_optional_media_requires_complete_hash_verified_selection(self) -> None:
        output = self.root / "verified"
        results = renderer.render_nano_v3b001_results(
            summary_path=self.fixture.summary_path,
            episodes_path=self.fixture.episodes_path,
            output_directory=output,
            actual_rollout_assets=self.fixture.actual_assets,
            decoded_prediction_assets=self.fixture.prediction_assets,
        )
        manifest = json.loads(results["manifest"].read_text(encoding="utf-8"))
        for selected in manifest["selected_media"]:
            self.assertEqual(
                selected["actual_rollout"]["local_verification"]["status"],
                "verified_local_asset",
            )
            self.assertEqual(
                selected["decoded_local_predictions"][0]["local_verification"]["status"],
                "verified_local_asset",
            )

        one_actual = dict(self.fixture.actual_assets)
        one_actual.pop(next(iter(one_actual)))
        with self.assertRaisesRegex(renderer.NanoResultRenderError, "all four"):
            renderer.render_nano_v3b001_results(
                summary_path=self.fixture.summary_path,
                episodes_path=self.fixture.episodes_path,
                output_directory=self.root / "partial",
                actual_rollout_assets=one_actual,
            )
        self.assertFalse((self.root / "partial").exists())

        first_path = next(iter(self.fixture.prediction_assets.values()))
        payload = bytearray(first_path.read_bytes())
        payload[0] ^= 1
        first_path.write_bytes(bytes(payload))
        with self.assertRaisesRegex(renderer.NanoResultRenderError, "SHA-256"):
            renderer.render_nano_v3b001_results(
                summary_path=self.fixture.summary_path,
                episodes_path=self.fixture.episodes_path,
                output_directory=self.root / "tampered",
                actual_rollout_assets=self.fixture.actual_assets,
                decoded_prediction_assets=self.fixture.prediction_assets,
            )
        self.assertFalse((self.root / "tampered").exists())


if __name__ == "__main__":
    unittest.main()
