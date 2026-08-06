from __future__ import annotations

import json
import unittest

from tools import audit_vla_wam_measurement_coverage as audit


class MeasurementCoverageAuditTest(unittest.TestCase):
    def test_committed_report_reproduces_exactly(self) -> None:
        committed = json.loads(
            (
                audit.V3 / "measurement_coverage_audit.json"
            ).read_text()
        )
        reproduced = audit.build_report(committed["recorded_at_utc"])
        self.assertEqual(reproduced, committed)

    def test_all_behavioral_episodes_retain_both_measurements(self) -> None:
        report = audit.build_report("2026-08-05T23:00:00Z")
        self.assertEqual(report["study_id"], "vla_wam_language_steerability_v3")
        self.assertEqual(report["scope"]["unique_behavioral_episode_count"], 982)
        self.assertEqual(report["scope"]["droid_robolab_episode_count"], 532)
        self.assertEqual(report["scope"]["robotwin_episode_count"], 450)
        self.assertEqual(
            report["coverage"],
            {
                "requested_side_margin_available": "982/982",
                "signed_final_lateral_offset_available": "982/982",
                "values_imputed_from_success_labels": 0,
                "measurement_coverage_rerun_required": False,
            },
        )
        cohorts = report["cohorts"]
        self.assertEqual(len(cohorts), 27)
        self.assertEqual(len({row["cohort_id"] for row in cohorts}), len(cohorts))
        self.assertTrue(
            all(not row["rerun_required_for_these_two_measurements"] for row in cohorts)
        )

    def test_nano_margin_signal_and_groot_reconciliation(self) -> None:
        report = audit.build_report("2026-08-05T23:00:00Z")
        nano = report["nano_phase_a_margin_sensitivity_reproduction"]
        self.assertEqual(nano["matched_pair_count"], 27)
        self.assertAlmostEqual(nano["right_minus_left_mean_margin_gap_m"], 0.12360139639565239)
        self.assertEqual(nano["positive_zero_negative_pair_counts"], [23, 0, 4])
        self.assertAlmostEqual(
            nano["exact_two_sided_sign_test_p_excluding_ties"],
            0.000310748815536499,
        )
        self.assertEqual(
            report["groot_phase_a_reconciliation"]["status"],
            "already_complete_do_not_rerun",
        )


if __name__ == "__main__":
    unittest.main()
