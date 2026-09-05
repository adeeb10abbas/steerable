"""Synthetic tests for the V4 offline analysis compiler."""

from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.analysis import (
    DEFAULT_ANALYSIS_SEED,
    DEFAULT_BOOTSTRAP_RESAMPLES,
    AnalysisError,
    InferenceResult,
    c1_block_interaction,
    c1_goal_interaction,
    c2_goal_selectivity,
    compile_analysis,
    compile_primary_inference,
    export_analysis_tables,
    holm_adjust_primary_tests,
    studentized_reset_block_inference,
    validate_accepted_ledger,
    _cell_lookup,
    _c1_estimator,
    _c2_estimator,
)


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("online_correction_v4", ROOT / "tools/online_correction_v4.py")
v4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v4)


GOAL_SET_HASH = "1" * 64
PREFIX_RECEIPT_SHA = "2" * 64
PREFIX_IDENTITY_SHA = "3" * 64
RESPONSE_SCORER_SHA = "4" * 64


def _c2_response_outcome(response_m: float, *, goal_hash: str = GOAL_SET_HASH) -> dict:
    return {
        "goal_violation_capped_m": response_m,
        "response_goal_violation_capped_m": response_m,
        "response_horizon_s": 2.0,
        "response_anchor": "t_event_planned+2.0s",
        "response_goal_set_branch": "move",
        "response_goal_set_hash_sha256": goal_hash,
        "response_projection": "planar",
        "response_scorer_sha256": RESPONSE_SCORER_SHA,
        "goal_set_empty": False,
        "goal_violation_cap_applied": False,
        "failure_stage": "pickup",
    }


def _result(
    row: dict,
    *,
    success: bool = False,
    trigger_eligible: bool = True,
    response_m: float = 0.10,
    c2: bool | None = None,
    goal_hash: str = GOAL_SET_HASH,
) -> dict:
    is_c2 = row["family"] == "C2" if c2 is None else c2
    capped = 0.0 if success else response_m
    outcome = _c2_response_outcome(capped if is_c2 else response_m, goal_hash=goal_hash) if is_c2 else {
        "goal_violation_capped_m": capped,
        "response_goal_violation_capped_m": capped,
        "goal_set_empty": False,
        "goal_violation_cap_applied": False,
        "failure_stage": "none" if success else "pickup",
    }
    record = {
        "episode_id": row["episode_id"],
        "attempt_id": "attempt-1",
        "status": "valid",
        "config_sha256": row["config_sha256"],
        "prefix_group_id": row["prefix_group_id"],
        "success": success,
        "trigger_eligible": trigger_eligible,
        "event_delivered": trigger_eligible,
        "event_observed": False,
        "outcome": outcome,
        "trace_uri": "s3://data/trace",
        "video_uri": "s3://data/video",
        "trace_sha256": "a" * 64,
        "video_sha256": "b" * 64,
        "scorer_sha256": "c" * 64,
        "protocol_sha256": "d" * 64,
        "response_latency_s": None,
    }
    if is_c2:
        record.update(
            {
                "common_prefix_verification_mode": "deterministic_fresh_session_replay",
                "common_prefix_verification_receipt_sha256": PREFIX_RECEIPT_SHA,
                "common_prefix_identity_hash_sha256": PREFIX_IDENTITY_SHA,
            }
        )
    return record


def _subset_manifest(config: dict, config_sha: str, *, families: tuple[str, ...], blocks: range) -> list[dict]:
    rows = v4.build_manifest(config, config_sha)
    selected = {families} if isinstance(families, str) else set(families)
    return [row for row in rows if row["family"] in selected and row["block_id"] in blocks]


def _fill_c1_block(
    manifest: list[dict],
    *,
    policy: str,
    block_id: int,
    direct_move: bool,
    direct_sham: bool,
    inverse_move: bool,
    inverse_sham: bool,
) -> list[dict]:
    rows = []
    for row in manifest:
        if row["family"] != "C1" or row["block_id"] != block_id or row["factors"]["policy"] != policy:
            continue
        if row["factors"]["scenario"] == "destination_static":
            continue
        key = (row["factors"]["wording"], row["factors"]["scenario"])
        success = {
            ("direct", "move_stop"): direct_move,
            ("direct", "original_sham"): direct_sham,
            ("inverse", "move_stop"): inverse_move,
            ("inverse", "original_sham"): inverse_sham,
        }[key]
        rows.append(_result(row, success=success))
    return rows


def _complete_results(manifest: list[dict], partial: list[dict]) -> list[dict]:
    by_id = {row["episode_id"]: row for row in partial}
    completed = []
    for row in manifest:
        if row["episode_id"] in by_id:
            completed.append(by_id[row["episode_id"]])
            continue
        trigger = row["family"] == "C2"
        completed.append(_result(row, success=False, trigger_eligible=trigger, response_m=0.10))
    return completed


def _fill_c2_block(
    manifest: list[dict],
    *,
    policy: str,
    block_id: int,
    sham_a: float,
    move_a: float,
    sham_b: float,
    move_b: float,
    eligible: bool = True,
) -> list[dict]:
    rows = []
    for row in manifest:
        if row["family"] != "C2" or row["block_id"] != block_id or row["factors"]["policy"] != policy:
            continue
        named = row["factors"]["named_reference"]
        scenario = row["factors"]["scenario"]
        if scenario == "original_sham":
            response = {"A": sham_a, "B": sham_b}[named]
            rows.append(_result(row, success=False, trigger_eligible=eligible, response_m=response))
        else:
            move = {"A": move_a, "B": move_b}[named]
            rows.append(_result(row, success=False, trigger_eligible=eligible, response_m=move))
    return rows


class LedgerValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.manifest = _subset_manifest(cls.config, cls.config_sha, families=("C1", "C2"), blocks=range(2))

    def test_accepted_ledger_validation_requires_c2_response_field(self):
        c2_row = next(row for row in self.manifest if row["family"] == "C2")
        broken = _result(c2_row)
        del broken["outcome"]["response_goal_violation_capped_m"]
        report = validate_accepted_ledger([c2_row], [broken], config=self.config)
        self.assertFalse(report["ok"])
        self.assertTrue(any("explicit outcome.response_goal_violation_capped_m" in error for error in report["errors"]))

    def test_terminal_only_goal_violation_is_rejected_for_c2(self):
        c2_row = next(row for row in self.manifest if row["family"] == "C2")
        broken = _result(c2_row)
        del broken["outcome"]["response_goal_violation_capped_m"]
        broken["outcome"]["goal_violation_capped_m"] = 0.05
        report = validate_accepted_ledger([c2_row], [broken], config=self.config)
        self.assertFalse(report["ok"])
        self.assertTrue(any("terminal goal_violation_capped_m is not accepted" in error for error in report["errors"]))

    def test_coverage_reconciliation_counts_missing_cells(self):
        valid = _result(next(row for row in self.manifest if row["family"] == "C1"))
        report = validate_accepted_ledger(self.manifest, [valid], config=self.config)
        self.assertFalse(report["ok"])
        self.assertGreater(len(report["missing_episode_ids"]), 0)


class C1EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.manifest = _subset_manifest(cls.config, cls.config_sha, families=("C1",), blocks=range(4))

    def test_exact_c1_block_estimator_matches_prespecified_interaction(self):
        results = []
        for block_id in range(4):
            results.extend(
                _fill_c1_block(
                    self.manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=False,
                    direct_sham=False,
                    inverse_move=True,
                    inverse_sham=False,
                )
            )
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(self.manifest, accepted)
        for block_id in range(4):
            for goal in ("left", "right", "front", "behind"):
                self.assertAlmostEqual(
                    c1_goal_interaction(lookup, policy="cosmos3_nano_droid", block_id=block_id, goal=goal),
                    1.0,
                )
            self.assertAlmostEqual(c1_block_interaction(lookup, policy="cosmos3_nano_droid", block_id=block_id), 1.0)
        point, complete = _c1_estimator(lookup, policy="cosmos3_nano_droid", block_ids=list(range(4)))
        self.assertEqual(complete, 4)
        self.assertAlmostEqual(point, 1.0)


class C2EstimatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")
        cls.manifest = _subset_manifest(cls.config, cls.config_sha, families=("C2",), blocks=range(3))

    def test_exact_c2_equal_goal_estimator_uses_eligibility_masks(self):
        results = []
        for block_id in range(3):
            results.extend(
                _fill_c2_block(
                    self.manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    sham_a=0.20,
                    move_a=0.10,
                    sham_b=0.30,
                    move_b=0.30,
                )
            )
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(self.manifest, accepted)
        for block_id in range(3):
            for goal in ("left", "right", "front", "behind"):
                self.assertAlmostEqual(
                    c2_goal_selectivity(
                        lookup, policy="cosmos3_nano_droid", block_id=block_id, goal=goal, config=self.config
                    ),
                    0.10,
                )
        aggregate, details = _c2_estimator(
            lookup, policy="cosmos3_nano_droid", block_ids=list(range(3)), config=self.config
        )
        self.assertAlmostEqual(aggregate, 0.10)
        self.assertEqual(details["goal_eligible_counts"]["left"], 3)

    def test_sparse_goal_makes_c2_aggregate_unestimable(self):
        manifest = copy.deepcopy(self.manifest)
        results = []
        for block_id in range(3):
            results.extend(
                _fill_c2_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    sham_a=0.20,
                    move_a=0.10,
                    sham_b=0.30,
                    move_b=0.30,
                    eligible=True,
                )
            )
        for row in results:
            manifest_row = next(item for item in manifest if item["episode_id"] == row["episode_id"])
            if manifest_row["factors"]["goal"] == "behind":
                row["trigger_eligible"] = False
                row["event_delivered"] = False
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(manifest, accepted)
        self.assertIsNone(
            c2_goal_selectivity(
                lookup, policy="cosmos3_nano_droid", block_id=0, goal="behind", config=self.config
            )
        )
        aggregate, details = _c2_estimator(
            lookup, policy="cosmos3_nano_droid", block_ids=list(range(3)), config=self.config
        )
        self.assertIsNone(aggregate)
        self.assertIsNone(details["goal_means"]["behind"])


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")

    def test_known_effect_produces_estimable_positive_c1_inference(self):
        manifest = _subset_manifest(self.config, self.config_sha, families=("C1",), blocks=range(32))
        results = []
        patterns = [
            (False, False, True, False),
            (False, False, True, True),
            (False, True, True, False),
            (False, True, True, True),
        ]
        for block_id in range(32):
            direct_move, direct_sham, inverse_move, inverse_sham = patterns[block_id % len(patterns)]
            results.extend(
                _fill_c1_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=direct_move,
                    direct_sham=direct_sham,
                    inverse_move=inverse_move,
                    inverse_sham=inverse_sham,
                )
            )
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(manifest, accepted)
        inference = studentized_reset_block_inference(
            contrast_key="C1_wording_x_motion_success_per_main_policy",
            policy_id="cosmos3_nano_droid",
            robot_stack="robolab_droid",
            estimand="wording_x_motion_success_interaction_pp",
            block_ids=list(range(32)),
            estimator=lambda sample: _c1_estimator(lookup, policy="cosmos3_nano_droid", block_ids=sample),
            bootstrap_resamples=1000,
            seed=DEFAULT_ANALYSIS_SEED,
        )
        self.assertEqual(inference.test_status, "estimable")
        assert inference.point_estimate is not None
        self.assertGreater(inference.point_estimate, 0.0)

    def test_null_effect_ci_includes_zero(self):
        manifest = _subset_manifest(self.config, self.config_sha, families=("C1",), blocks=range(32))
        results = []
        patterns = [
            (False, False, True, False),
            (False, False, False, True),
        ]
        for block_id in range(32):
            direct_move, direct_sham, inverse_move, inverse_sham = patterns[block_id % len(patterns)]
            results.extend(
                _fill_c1_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=direct_move,
                    direct_sham=direct_sham,
                    inverse_move=inverse_move,
                    inverse_sham=inverse_sham,
                )
            )
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(manifest, accepted)
        inference = studentized_reset_block_inference(
            contrast_key="C1_wording_x_motion_success_per_main_policy",
            policy_id="cosmos3_nano_droid",
            robot_stack="robolab_droid",
            estimand="wording_x_motion_success_interaction_pp",
            block_ids=list(range(32)),
            estimator=lambda sample: _c1_estimator(lookup, policy="cosmos3_nano_droid", block_ids=sample),
            bootstrap_resamples=1000,
            seed=DEFAULT_ANALYSIS_SEED,
        )
        self.assertEqual(inference.test_status, "estimable")
        self.assertAlmostEqual(inference.point_estimate, 0.0)
        assert inference.ci_low is not None and inference.ci_high is not None
        self.assertLessEqual(inference.ci_low, 0.0)
        self.assertGreaterEqual(inference.ci_high, 0.0)

    def test_zero_se_marks_test_not_estimable(self):
        manifest = _subset_manifest(self.config, self.config_sha, families=("C1",), blocks=range(4))
        results = []
        for block_id in range(4):
            results.extend(
                _fill_c1_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=True,
                    direct_sham=False,
                    inverse_move=True,
                    inverse_sham=False,
                )
            )
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(manifest, accepted)
        inference = studentized_reset_block_inference(
            contrast_key="C1_wording_x_motion_success_per_main_policy",
            policy_id="cosmos3_nano_droid",
            robot_stack="robolab_droid",
            estimand="wording_x_motion_success_interaction_pp",
            block_ids=list(range(4)),
            estimator=lambda sample: _c1_estimator(lookup, policy="cosmos3_nano_droid", block_ids=sample),
            bootstrap_resamples=200,
            seed=DEFAULT_ANALYSIS_SEED,
        )
        self.assertEqual(inference.test_status, "not_estimable")
        self.assertIsNone(inference.p_value)
        self.assertIn("zero", inference.not_estimable_reason or "")

    def test_studentized_inference_is_deterministic_for_seed_20260905(self):
        manifest = _subset_manifest(self.config, self.config_sha, families=("C1",), blocks=range(6))
        results = []
        for block_id in range(6):
            results.extend(
                _fill_c1_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=block_id % 2 == 0,
                    direct_sham=False,
                    inverse_move=True,
                    inverse_sham=block_id % 2 == 1,
                )
            )
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(manifest, accepted)

        def run_once() -> tuple[float | None, float | None]:
            item = studentized_reset_block_inference(
                contrast_key="C1_wording_x_motion_success_per_main_policy",
                policy_id="cosmos3_nano_droid",
                robot_stack="robolab_droid",
                estimand="wording_x_motion_success_interaction_pp",
                block_ids=list(range(6)),
                estimator=lambda sample: _c1_estimator(lookup, policy="cosmos3_nano_droid", block_ids=sample),
                bootstrap_resamples=DEFAULT_BOOTSTRAP_RESAMPLES,
                seed=DEFAULT_ANALYSIS_SEED,
            )
            return item.point_estimate, item.p_value

        first = run_once()
        second = run_once()
        self.assertEqual(first, second)
        self.assertEqual(DEFAULT_ANALYSIS_SEED, 20260905)

    def test_holm_uses_bookkeeping_p_one_for_unestimable_slot(self):
        base = InferenceResult(
            contrast_key="C1_wording_x_motion_success_per_main_policy",
            policy_id="cosmos3_nano_droid",
            robot_stack="robolab_droid",
            estimand="wording_x_motion_success_interaction_pp",
            point_estimate=0.5,
            ci_low=0.1,
            ci_high=0.9,
            standard_error=0.1,
            p_value=0.01,
            test_status="estimable",
            not_estimable_reason=None,
            n_blocks=8,
            n_effective_blocks=8,
            bootstrap_resamples=1000,
            bootstrap_seed=DEFAULT_ANALYSIS_SEED,
            undefined_bootstrap_resamples=0,
            zero_or_undefined_se_resamples=0,
            holm_adjusted_p=None,
            holm_rejected=False,
            descriptive={},
        )
        null_slot = InferenceResult(
            **{
                **base.__dict__,
                "contrast_key": "C2_reference_x_motion_goal_improvement_per_main_policy",
                "p_value": None,
                "test_status": "not_estimable",
                "not_estimable_reason": "observed jackknife standard error is zero or undefined",
            }
        )
        adjusted = holm_adjust_primary_tests([base, null_slot, null_slot, null_slot])
        self.assertIsNone(adjusted[1].p_value)
        self.assertIsNone(adjusted[1].holm_adjusted_p)
        self.assertFalse(adjusted[1].holm_rejected)
        self.assertIsNotNone(adjusted[0].holm_adjusted_p)


class NoPoolingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")

    def test_policies_are_analyzed_separately_without_cross_stack_pooling(self):
        manifest = _subset_manifest(self.config, self.config_sha, families=("C1",), blocks=range(6))
        results = []
        for block_id in range(6):
            partial = []
            partial.extend(
                _fill_c1_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=False,
                    direct_sham=False,
                    inverse_move=True,
                    inverse_sham=False,
                )
            )
            partial.extend(
                _fill_c1_block(
                    manifest,
                    policy="pi05_droid",
                    block_id=block_id,
                    direct_move=False,
                    direct_sham=False,
                    inverse_move=False,
                    inverse_sham=False,
                )
            )
            results.extend(partial)
        results = _complete_results(manifest, results)
        accepted = {row["episode_id"]: row for row in results}
        lookup = _cell_lookup(manifest, accepted)
        primary = compile_primary_inference(lookup, self.config, bootstrap_resamples=300, seed=DEFAULT_ANALYSIS_SEED)
        c1_rows = [row for row in primary if row.contrast_key.startswith("C1_")]
        self.assertEqual(len(c1_rows), 2)
        by_policy = {row.policy_id: row for row in c1_rows}
        self.assertAlmostEqual(by_policy["cosmos3_nano_droid"].point_estimate, 1.0)
        self.assertAlmostEqual(by_policy["pi05_droid"].point_estimate, 0.0)
        stacks = {row.robot_stack for row in c1_rows}
        self.assertEqual(stacks, {"robolab_droid"})


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.config_sha = v4.load_json(ROOT / "docs/online_correction_v4/campaign.json")

    def test_export_builds_compact_tables_without_fabricated_figures(self):
        manifest = _subset_manifest(self.config, self.config_sha, families=("C1", "C2"), blocks=range(2))
        partial = []
        for block_id in range(2):
            partial.extend(
                _fill_c1_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    direct_move=False,
                    direct_sham=False,
                    inverse_move=True,
                    inverse_sham=False,
                )
            )
            partial.extend(
                _fill_c2_block(
                    manifest,
                    policy="cosmos3_nano_droid",
                    block_id=block_id,
                    sham_a=0.20,
                    move_a=0.10,
                    sham_b=0.20,
                    move_b=0.20,
                )
            )
        results = _complete_results(manifest, partial)
        compiled = compile_analysis(
            manifest=manifest,
            results=results,
            config=self.config,
            bootstrap_resamples=200,
            seed=DEFAULT_ANALYSIS_SEED,
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest_path = out / "manifest.jsonl"
            results_path = out / "results.jsonl"
            config_path = out / "campaign.json"
            manifest_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in manifest) + "\n")
            results_path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in results) + "\n")
            config_path.write_text(json.dumps(self.config))
            export = export_analysis_tables(
                compiled,
                out / "tables",
                manifest_path=manifest_path,
                results_path=results_path,
                config_path=config_path,
            )
            for name in (
                "coverage_by_cell.csv",
                "primary_results.csv",
                "paired_contrasts.csv",
                "wording_results.csv",
                "scope_replications.csv",
                "failure_composition.csv",
                "timing_and_motion.csv",
                "audit_report.json",
                "results_manifest.json",
            ):
                self.assertTrue(Path(export["tables"][name]).exists(), name)
            audit = json.loads((out / "tables" / "audit_report.json").read_text())
            self.assertIn("parquet", audit["limitations"][0].lower())
            primary = (out / "tables" / "primary_results.csv").read_text()
            self.assertIn("contrast_registry_key", primary)
            self.assertNotIn(".png", primary)


if __name__ == "__main__":
    unittest.main()
