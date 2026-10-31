"""Checks on the behavioral-paper extraction, independent of any new model run."""
import copy
import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "analysis" / "extract_paper_results.py"
SPEC = importlib.util.spec_from_file_location("paper_results", MODULE_PATH)
PAPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PAPER)


def complete_fixture():
    rows = []
    for seed in PAPER.SEEDS:
        for arm in PAPER.ARMS:
            for side in ("left", "right"):
                y = {("control", "left"): 1, ("control", "right"): -2,
                     ("position_mirrored", "left"): 2, ("position_mirrored", "right"): -1}[(arm, side)]
                rows.append({"seed": seed, "arm": arm, "direction": side, "y_m": y,
                             "success": side == "left", "failure_category": "correct" if side == "left" else "transport_failed",
                             "initial_state_sha256": arm, "pair_identity_sha256": arm,
                             "executed_action_sha256": side})
    return rows


def dreamzero_runtime_fixture():
    return {
        "checkpoint": {"id": "checkpoint"},
        "action_cap": 450,
        "open_loop_horizon": 8,
        "effective_official_model_noise_seed": 1140,
        "registered_seed_semantics": "matched labels; fixed model noise",
        "instruction_controller": "static_episode_prompt",
        "frozen_predicate_id": "predicate",
        **copy.deepcopy(PAPER.DREAMZERO_S2_RUNTIME_EXACT),
    }


class PaperResultsTests(unittest.TestCase):
    def test_dreamzero_s2_runtime_contract_is_preserved(self):
        contract = dreamzero_runtime_fixture()
        runtime = PAPER.dreamzero_s2_runtime(contract)
        self.assertEqual(set(runtime), set(PAPER.DREAMZERO_RUNTIME_REPORT_FIELDS))
        for field, expected in PAPER.DREAMZERO_S2_RUNTIME_EXACT.items():
            self.assertEqual(runtime[field], expected)
            self.assertIs(type(runtime[field]), type(expected))

    def test_dreamzero_s2_runtime_contract_mutations_fail(self):
        mutations = {
            "action_cfg_style_scale": 3,
            "baseline_action_cfg_equivalent": 2,
            "baseline_s1_used": True,
            "identity_binding": "other",
            "video_cfg_scale": 4,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                contract = dreamzero_runtime_fixture()
                contract[field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    PAPER.dreamzero_s2_runtime(contract)
        for field in ("id", "sha256"):
            with self.subTest(runtime_identity_field=field):
                contract = dreamzero_runtime_fixture()
                contract["runtime_identity"][field] = "changed"
                with self.assertRaisesRegex(ValueError, "runtime_identity"):
                    PAPER.dreamzero_s2_runtime(contract)

    def test_dreamzero_s2_runtime_contract_missing_fields_fail(self):
        for field in PAPER.DREAMZERO_S2_RUNTIME_EXACT:
            with self.subTest(field=field):
                contract = dreamzero_runtime_fixture()
                del contract[field]
                with self.assertRaisesRegex(ValueError, field):
                    PAPER.dreamzero_s2_runtime(contract)
        for field in ("id", "sha256"):
            with self.subTest(runtime_identity_field=field):
                contract = dreamzero_runtime_fixture()
                del contract["runtime_identity"][field]
                with self.assertRaisesRegex(ValueError, "runtime_identity"):
                    PAPER.dreamzero_s2_runtime(contract)

    def test_dreamzero_s2_runtime_contract_types_are_exact(self):
        for field, replacement in (
            ("action_cfg_style_scale", 2.0),
            ("baseline_action_cfg_equivalent", 1.0),
            ("baseline_s1_used", 0),
            ("video_cfg_scale", 5.0),
        ):
            with self.subTest(field=field):
                contract = dreamzero_runtime_fixture()
                contract[field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    PAPER.dreamzero_s2_runtime(contract)

    def test_signed_depth_is_separate_from_endpoint_ordering(self):
        values, conditions, pairs = PAPER.block_contrasts(complete_fixture(), PAPER.ARMS)
        self.assertEqual(values["control"]["endpoint_response_m"], [3] * 27)
        self.assertEqual(values["position_mirrored"]["endpoint_response_m"], [3] * 27)
        self.assertEqual(values["control"]["placement_depth_gap_m"], [1] * 27)
        self.assertEqual(values["position_mirrored"]["placement_depth_gap_m"], [-1] * 27)
        self.assertEqual(conditions["control:left"]["successes"], 27)
        self.assertEqual(len(pairs), 54)

    def test_missing_or_duplicate_cell_fails(self):
        rows = complete_fixture()
        for malformed in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                PAPER.block_contrasts(malformed, PAPER.ARMS)

    def test_mismatched_pair_identity_fails(self):
        rows = complete_fixture()
        rows[0]["pair_identity_sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Pair identity"):
            PAPER.block_contrasts(rows, PAPER.ARMS)

    def test_exact_sign_test_excludes_ties(self):
        result = PAPER.sign_test([1, 1, 1, 0, 0])
        self.assertEqual(result["p_value"], 0.25)
        self.assertEqual(result["ties"], 2)

    def test_exact_permutation_retains_zero_blocks(self):
        result = PAPER.sign_flip([1, 1, 1, 0, 0])
        self.assertEqual(result, {"p_value": 0.25, "extreme": 8, "permutations": 32})
        self.assertEqual(PAPER.sign_flip([1, -1])["p_value"], 1.0)

    def test_reported_point_estimate_mismatch_fails(self):
        with self.assertRaisesRegex(ValueError, "mean"):
            PAPER.reconcile_continuous([1.0, 2.0], {"mean_m": 3.0}, engine="python_random")


if __name__ == "__main__":
    unittest.main()
