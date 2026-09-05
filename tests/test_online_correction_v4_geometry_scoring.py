"""Hand-checked terminal scoring tests for online correction V4."""

from __future__ import annotations

import math
import unittest

from experiments.online_correction_v4 import geometry as geom
from experiments.online_correction_v4 import scoring


class TerminalScorerTests(unittest.TestCase):
    def setUp(self):
        self.frame = geom.TaskFrame.identity()
        self.workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        self.foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        self.ref = (0.0, 0.0, 0.04)
        self.planar = geom.PlanarRelationSpec(
            relation="left",
            clearance_m=0.01,
            workspace=self.workspace,
            object_footprint=self.foot,
            reference_footprint=self.foot,
        )
        self.ctx = scoring.ScoringContext(frame=self.frame, d_cap_m=0.75, planar_spec=self.planar)

    def _evidence(self, **kwargs) -> scoring.TerminalEvidence:
        base = dict(
            p_obj_world=(0.12, 0.0, 0.04),
            p_named_ref_world=self.ref,
            relation="left",
            grasp_occurred=True,
            carry_verified=True,
            released=True,
            stable_for_dwell=True,
            allowed_support=True,
        )
        base.update(kwargs)
        return scoring.TerminalEvidence(**base)

    def test_success_in_any_valid_region(self):
        score = scoring.score_terminal_first_placement(self.ctx, self._evidence(p_obj_world=(0.20, 0.05, 0.04)))
        self.assertTrue(score.success)
        self.assertEqual(score.failure_label, "success")
        self.assertEqual(score.failure_stage, "none")
        self.assertAlmostEqual(score.goal_violation_m, 0.0)
        self.assertFalse(score.goal_violation_cap_applied)

    def test_wrong_side_release_is_wrong_goal_region(self):
        score = scoring.score_terminal_first_placement(
            self.ctx,
            self._evidence(p_obj_world=(-0.20, 0.0, 0.04)),
        )
        self.assertFalse(score.success)
        self.assertEqual(score.failure_label, "wrong_goal_region")
        self.assertEqual(score.failure_stage, "wrong_relation")
        self.assertGreater(score.goal_violation_m, 0.0)

    def test_correct_side_but_not_released(self):
        score = scoring.score_terminal_first_placement(
            self.ctx,
            self._evidence(p_obj_world=(0.20, 0.0, 0.04), released=False, allowed_support=False),
        )
        self.assertFalse(score.success)
        self.assertEqual(score.failure_label, "release_failed")
        self.assertEqual(score.failure_stage, "release")

    def test_no_grasp_precedes_wrong_region(self):
        score = scoring.score_terminal_first_placement(
            self.ctx,
            self._evidence(grasp_occurred=False, carry_verified=False, released=False),
        )
        self.assertEqual(score.failure_label, "no_grasp")
        self.assertEqual(score.failure_stage, "pickup")

    def test_empty_goal_set_forces_failure_and_cap(self):
        tiny = geom.AxisAlignedBox(0.40, 0.45, -0.1, 0.1, 0.0, 0.08)
        ctx = scoring.ScoringContext(
            frame=self.frame,
            d_cap_m=0.5,
            planar_spec=geom.PlanarRelationSpec(
                relation="left",
                clearance_m=0.01,
                workspace=tiny,
                object_footprint=self.foot,
                reference_footprint=self.foot,
            ),
        )
        score = scoring.score_terminal_first_placement(
            ctx,
            self._evidence(p_named_ref_world=(0.44, 0.0, 0.04)),
        )
        self.assertFalse(score.success)
        self.assertTrue(score.goal_set_empty)
        self.assertTrue(score.goal_violation_cap_applied)
        self.assertEqual(score.goal_violation_capped_m, 0.5)
        self.assertFalse(math.isfinite(score.goal_violation_m))

    def test_all_predicates_logged(self):
        score = scoring.score_terminal_first_placement(self.ctx, self._evidence())
        expected = {
            "geometric_relation_correct",
            "required_manipulation_occurred",
            "released",
            "allowed_support_or_containment",
            "stable_for_registered_dwell",
            "no_registered_terminal_violation",
            "success",
        }
        self.assertEqual(set(score.predicates), expected)

    def test_failure_mapping_table(self):
        mappings = {
            "no_grasp": "pickup",
            "grasp_lost": "transport",
            "transport_incomplete": "transport",
            "wrong_goal_region": "wrong_relation",
            "release_failed": "release",
            "support_or_containment_failed": "release",
            "timeout_without_completion": "timeout",
            "collision_caused_terminal_failure": "collision",
            "model_output_invalid": "other",
            "unresolved_behavioral_failure": "other",
            "success": "none",
        }
        for label, stage in mappings.items():
            self.assertEqual(scoring.failure_stage_for_label(label), stage)


class VerticalAndContainmentScorerTests(unittest.TestCase):
    def setUp(self):
        self.frame = geom.TaskFrame.identity()

    def test_hovering_above_shelf_fails_support_predicate(self):
        top = geom.AxisAlignedBox(-0.2, 0.2, -0.2, 0.2, 0.62, 0.68)
        bottom = geom.AxisAlignedBox(-0.2, 0.2, -0.2, 0.2, 0.12, 0.18)
        shelf = geom.ShelfRelationSpec(
            relation="above",
            top_shelf=top,
            bottom_shelf=bottom,
            reference_footprint=geom.ObjectFootprint(0.04, 0.04, 0.03),
            object_footprint=geom.ObjectFootprint(0.02, 0.02, 0.02),
            horizontal_overlap_min_m=0.001,
        )
        ctx = scoring.ScoringContext(frame=self.frame, d_cap_m=1.0, shelf_spec=shelf)
        evidence = scoring.TerminalEvidence(
            p_obj_world=(0.0, 0.0, 0.65),
            p_named_ref_world=(0.0, 0.0, 0.50),
            relation="above",
            grasp_occurred=True,
            carry_verified=True,
            released=True,
            stable_for_dwell=True,
            allowed_support=False,
        )
        score = scoring.score_terminal_first_placement(ctx, evidence)
        self.assertFalse(score.success)
        self.assertEqual(score.failure_label, "support_or_containment_failed")

    def test_containment_success_requires_full_interior(self):
        spec = geom.ContainmentSpec(
            interior_reference_local=geom.AxisAlignedBox(-0.08, 0.08, -0.08, 0.08, 0.0, 0.10),
            object_footprint=geom.ObjectFootprint(0.02, 0.02, 0.02),
            wall_clearance_m=0.005,
        )
        ctx = scoring.ScoringContext(frame=self.frame, d_cap_m=1.0, containment_spec=spec)
        good = scoring.TerminalEvidence(
            p_obj_world=(0.0, 0.0, 0.025),
            p_named_ref_world=(0.0, 0.0, 0.0),
            relation="inside",
            grasp_occurred=True,
            carry_verified=True,
            released=True,
            stable_for_dwell=True,
            allowed_containment=True,
        )
        bad = scoring.TerminalEvidence(
            p_obj_world=(0.07, 0.0, 0.05),
            p_named_ref_world=(0.0, 0.0, 0.0),
            relation="contains",
            grasp_occurred=True,
            carry_verified=True,
            released=True,
            stable_for_dwell=True,
            allowed_containment=True,
        )
        self.assertTrue(scoring.score_terminal_first_placement(ctx, good).success)
        self.assertEqual(
            scoring.score_terminal_first_placement(ctx, bad).failure_label,
            "wrong_goal_region",
        )


class C2MembershipScorerTests(unittest.TestCase):
    def test_named_vs_other_membership_reported(self):
        frame = geom.TaskFrame.identity()
        workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        ref_a = (-0.15, 0.0, 0.04)
        ref_b = (0.15, 0.0, 0.04)
        ctx = scoring.ScoringContext(
            frame=frame,
            d_cap_m=1.0,
            planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
            other_planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
        )
        evidence = scoring.TerminalEvidence(
            p_obj_world=(-0.05, 0.0, 0.04),
            p_named_ref_world=ref_a,
            p_other_ref_world=ref_b,
            relation="left",
            grasp_occurred=True,
            carry_verified=True,
            released=True,
            stable_for_dwell=True,
            allowed_support=True,
        )
        score = scoring.score_terminal_first_placement(ctx, evidence)
        self.assertEqual(score.reference_membership, "named")


class ResponseHorizonScorerTests(unittest.TestCase):
    def test_response_projection_distance_matches_geometry_helper(self):
        frame = geom.TaskFrame.identity()
        workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.30)
        planar = geom.PlanarRelationSpec(
            relation="left",
            clearance_m=0.01,
            workspace=workspace,
            object_footprint=geom.ObjectFootprint(0.02, 0.02, 0.02),
            reference_footprint=geom.ObjectFootprint(0.05, 0.05, 0.02),
        )
        ctx = scoring.ScoringContext(frame=frame, d_cap_m=0.8, planar_spec=planar)
        evidence = scoring.TerminalEvidence(
            p_obj_world=(0.05, 0.0, 0.25),
            p_named_ref_world=(0.0, 0.0, 0.04),
            relation="left",
        )
        score = scoring.score_terminal_first_placement(ctx, evidence, include_response_projection=True)
        direct = scoring.score_response_horizon(ctx, evidence, p_obj_world=(0.05, 0.0, 0.25))
        self.assertIsNotNone(score.response_projection_distance_m)
        self.assertAlmostEqual(score.response_projection_distance_m, direct.distance_m)
        self.assertAlmostEqual(score.response_projection_capped_m, direct.capped_distance_m)

    def test_horizon_pair_uses_moving_branch_shared_goal_set(self):
        frame = geom.TaskFrame.identity()
        workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        ctx = scoring.ScoringContext(
            frame=frame,
            d_cap_m=1.0,
            planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
        )
        sham = geom.TrajectorySeries(
            samples=(
                geom.TrajectorySample(0.0, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
                geom.TrajectorySample(2.0, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
            )
        )
        move = geom.TrajectorySeries(
            samples=(
                geom.TrajectorySample(0.0, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
                geom.TrajectorySample(2.0, (0.20, 0.0, 0.04), (0.12, 0.0, 0.04)),
            )
        )
        result = scoring.score_response_horizon_pair(
            ctx,
            scoring.ResponseHorizonRequest(
                t_event_planned_s=0.0,
                horizon_s=2.0,
                sham_trajectory=sham,
                move_trajectory=move,
            ),
        )
        self.assertAlmostEqual(result.d_cap_move_m, 0.0, places=6)
        self.assertGreater(result.d_cap_sham_m, 0.0)
        self.assertGreater(result.h_response_m, 0.0)
        self.assertEqual(result.shared_p_ref_world, (0.12, 0.0, 0.04))

    def test_horizon_pair_applies_terminal_extension_after_early_release(self):
        frame = geom.TaskFrame.identity()
        workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        ctx = scoring.ScoringContext(
            frame=frame,
            d_cap_m=1.0,
            planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
        )
        sham = geom.TrajectorySeries(
            samples=(
                geom.TrajectorySample(0.0, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
                geom.TrajectorySample(0.5, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
            )
        )
        move = geom.TrajectorySeries(
            samples=(
                geom.TrajectorySample(0.0, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
                geom.TrajectorySample(0.5, (0.18, 0.0, 0.04), (0.12, 0.0, 0.04)),
            )
        )
        move_terminal = geom.TrajectorySample(0.5, (0.18, 0.0, 0.04), (0.12, 0.0, 0.04))
        result = scoring.score_response_horizon_pair(
            ctx,
            scoring.ResponseHorizonRequest(
                t_event_planned_s=0.0,
                horizon_s=2.0,
                sham_trajectory=sham,
                move_trajectory=move,
                move_terminal_extension=move_terminal,
            ),
        )
        self.assertTrue(result.move_terminal_extension_applied)
        self.assertAlmostEqual(result.move_p_obj_world, (0.18, 0.0, 0.04))
        self.assertAlmostEqual(result.d_cap_move_m, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
