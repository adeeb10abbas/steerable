"""Focused statistical behavior tests for V4 analysis helpers."""

from __future__ import annotations

import unittest

from experiments.online_correction_v4.analysis import (
    DEFAULT_ANALYSIS_SEED,
    InferenceResult,
    holm_adjust_primary_tests,
)


class HolmAdjustmentTests(unittest.TestCase):
    def test_four_slot_holm_keeps_unestimable_display_null(self):
        slots = []
        for index, p_value in enumerate((0.001, 0.02, 0.03, 0.04)):
            slots.append(
                InferenceResult(
                    contrast_key=f"contrast_{index}",
                    policy_id="cosmos3_nano_droid",
                    robot_stack="robolab_droid",
                    estimand="synthetic",
                    point_estimate=float(index),
                    ci_low=-1.0,
                    ci_high=1.0,
                    standard_error=0.1,
                    p_value=p_value,
                    test_status="estimable",
                    not_estimable_reason=None,
                    n_blocks=8,
                    n_effective_blocks=8,
                    bootstrap_resamples=1000,
                    bootstrap_seed=DEFAULT_ANALYSIS_SEED + index,
                    undefined_bootstrap_resamples=0,
                    zero_or_undefined_se_resamples=0,
                    holm_adjusted_p=None,
                    holm_rejected=False,
                    descriptive={},
                )
            )
        slots[2] = InferenceResult(
            **{
                **slots[2].__dict__,
                "p_value": None,
                "test_status": "not_estimable",
                "not_estimable_reason": "synthetic unestimable slot",
            }
        )
        adjusted = holm_adjust_primary_tests(slots)
        self.assertEqual(len(adjusted), 4)
        self.assertIsNone(adjusted[2].p_value)
        self.assertIsNone(adjusted[2].holm_adjusted_p)
        self.assertFalse(adjusted[2].holm_rejected)
        self.assertIsNotNone(adjusted[0].holm_adjusted_p)
        self.assertTrue(adjusted[0].holm_rejected)


if __name__ == "__main__":
    unittest.main()
