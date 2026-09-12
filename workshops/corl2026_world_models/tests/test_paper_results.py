"""Checks on the behavioral-paper extraction, independent of any new model run."""
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


class PaperResultsTests(unittest.TestCase):
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
