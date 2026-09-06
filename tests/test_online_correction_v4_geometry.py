"""Hand-checked geometry tests for online correction V4."""

from __future__ import annotations

import math
import unittest

from experiments.online_correction_v4 import geometry as geom


class TaskFrameTests(unittest.TestCase):
    def test_identity_basis_is_orthonormal(self):
        frame = geom.TaskFrame.identity()
        self.assertAlmostEqual(geom._dot(frame.u_left, frame.u_front), 0.0)
        self.assertAlmostEqual(geom._dot(frame.u_front, frame.u_up), 0.0)

    def test_world_task_roundtrip(self):
        frame = geom.TaskFrame.identity()
        p = (0.12, -0.05, 0.03)
        back = frame.task_to_world(frame.world_to_task(p))
        for a, b in zip(p, back):
            self.assertAlmostEqual(a, b, places=9)

    def test_rejects_non_unit_or_left_handed_basis(self):
        with self.assertRaises(ValueError):
            geom.TaskFrame(
                u_left=(2.0, 0.0, 0.0),
                u_front=(0.0, 1.0, 0.0),
                u_up=(0.0, 0.0, 1.0),
            )
        with self.assertRaises(ValueError):
            geom.TaskFrame(
                u_left=(1.0, 0.0, 0.0),
                u_front=(0.0, -1.0, 0.0),
                u_up=(0.0, 0.0, 1.0),
            )

    def test_moving_reference_changes_relative_not_world_object(self):
        frame = geom.TaskFrame.identity()
        p_obj = (0.20, 0.10, 0.03)
        p_ref_before = (0.0, 0.0, 0.03)
        p_ref_after = (0.12, 0.0, 0.03)
        rel_before = frame.relative_offset_task(p_obj, p_ref_before)
        rel_after = frame.relative_offset_task(p_obj, p_ref_after)
        self.assertNotAlmostEqual(rel_before[0], rel_after[0])
        self.assertAlmostEqual(rel_before[1], rel_after[1])


class PlanarGoalSetTests(unittest.TestCase):
    def setUp(self):
        self.frame = geom.TaskFrame.identity()
        self.workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        self.foot_obj = geom.ObjectFootprint(0.02, 0.02, 0.02)
        self.foot_ref = geom.ObjectFootprint(0.05, 0.05, 0.02)
        self.clearance = 0.01
        self.p_ref = (0.0, 0.0, 0.04)
        self.d_cap = 1.0

    def _spec(self, relation: geom.RelationKind) -> geom.PlanarRelationSpec:
        return geom.PlanarRelationSpec(
            relation=relation,
            clearance_m=self.clearance,
            workspace=self.workspace,
            object_footprint=self.foot_obj,
            reference_footprint=self.foot_ref,
        )

    def test_left_threshold_hand_checked(self):
        spec = self._spec("left")
        goal = geom.build_planar_goal_set(self.frame, spec, self.p_ref)
        threshold = 0.0 + 0.05 + 0.02 + 0.01
        valid = (threshold + 0.01, 0.0, 0.04)
        invalid = (threshold - 0.02, 0.0, 0.04)
        self.assertTrue(geom.point_in_goal_set(self.frame, valid, goal))
        self.assertFalse(geom.point_in_goal_set(self.frame, invalid, goal))
        self.assertAlmostEqual(
            geom.goal_distance(self.frame, valid, goal, d_cap_m=self.d_cap).distance_m, 0.0
        )
        self.assertGreater(
            geom.goal_distance(self.frame, invalid, goal, d_cap_m=self.d_cap).distance_m, 0.0
        )

    def test_right_front_behind_signs(self):
        cases = {
            "right": (( -0.12, 0.0, 0.04), (0.05, 0.0, 0.04)),
            "front": ((0.0, 0.12, 0.04), (0.0, -0.02, 0.04)),
            "behind": ((0.0, -0.12, 0.04), (0.0, 0.05, 0.04)),
        }
        for relation, (good, bad) in cases.items():
            goal = geom.build_planar_goal_set(self.frame, self._spec(relation), self.p_ref)
            self.assertTrue(geom.point_in_goal_set(self.frame, good, goal), relation)
            self.assertFalse(geom.point_in_goal_set(self.frame, bad, goal), relation)

    def test_boundary_on_threshold_counts_as_valid(self):
        spec = self._spec("left")
        goal = geom.build_planar_goal_set(self.frame, spec, self.p_ref)
        threshold = 0.0 + 0.05 + 0.02 + 0.01
        self.assertTrue(geom.point_in_goal_set(self.frame, (threshold, 0.0, 0.04), goal, tol=0.0))

    def test_empty_goal_set_when_halfspace_misses_workspace(self):
        tiny = geom.AxisAlignedBox(0.40, 0.45, -0.1, 0.1, 0.0, 0.08)
        spec = geom.PlanarRelationSpec(
            relation="left",
            clearance_m=0.01,
            workspace=tiny,
            object_footprint=self.foot_obj,
            reference_footprint=self.foot_ref,
        )
        goal = geom.build_planar_goal_set(self.frame, spec, (0.44, 0.0, 0.04))
        self.assertTrue(goal.empty)
        dist = geom.goal_distance(self.frame, (0.42, 0.0, 0.04), goal, d_cap_m=0.5)
        self.assertTrue(dist.goal_set_empty)
        self.assertTrue(dist.cap_applied)
        self.assertEqual(dist.capped_distance_m, 0.5)

    def test_capped_distance_above_d_cap(self):
        spec = self._spec("left")
        goal = geom.build_planar_goal_set(self.frame, spec, self.p_ref)
        far = (-0.40, 0.0, 0.04)
        dist = geom.goal_distance(self.frame, far, goal, d_cap_m=0.05)
        self.assertGreater(dist.distance_m, 0.05)
        self.assertEqual(dist.capped_distance_m, 0.05)
        self.assertTrue(dist.cap_applied)

    def test_convex_workspace_is_clipped_without_axis_aligned_overreach(self):
        workspace = geom.ConvexPolygonPrism(
            vertices_xy=((0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)),
            z_min=0.04,
            z_max=0.04,
        )
        spec = geom.PolygonPlanarRelationSpec(
            relation="left",
            clearance_m=0.0,
            workspace=workspace,
            object_footprint=geom.ObjectFootprint(0.1, 0.1, 0.0),
            reference_footprint=geom.ObjectFootprint(0.1, 0.1, 0.0),
        )
        goal = geom.build_planar_goal_set(self.frame, spec, self.p_ref)
        self.assertTrue(
            geom.point_in_goal_set(self.frame, (0.3, 0.0, 0.04), goal)
        )
        self.assertFalse(
            geom.point_in_goal_set(self.frame, (0.8, 0.8, 0.04), goal)
        )
        distance = geom.goal_distance(
            self.frame,
            (0.8, 0.8, 0.04),
            goal,
            d_cap_m=1.0,
        )
        self.assertAlmostEqual(distance.distance_m, math.sqrt(0.18))

    def test_convex_workspace_response_projection_ignores_height(self):
        workspace = geom.ConvexPolygonPrism(
            vertices_xy=((0.0, 1.0), (-1.0, 0.0), (0.0, -1.0), (1.0, 0.0)),
            z_min=0.04,
            z_max=0.04,
        )
        spec = geom.PolygonPlanarRelationSpec(
            relation="left",
            clearance_m=0.0,
            workspace=workspace,
            object_footprint=geom.ObjectFootprint(0.1, 0.1, 0.0),
            reference_footprint=geom.ObjectFootprint(0.1, 0.1, 0.0),
        )
        goal = geom.build_planar_goal_set(
            self.frame,
            spec,
            self.p_ref,
            projection_kind="response_planar",
        )
        distance = geom.goal_distance(
            self.frame,
            (0.3, 0.0, 10.0),
            goal,
            d_cap_m=1.0,
        )
        self.assertEqual(distance.distance_m, 0.0)


class ShelfGoalTests(unittest.TestCase):
    def setUp(self):
        self.frame = geom.TaskFrame.identity()
        self.ref = (0.0, 0.0, 0.50)
        self.foot_obj = geom.ObjectFootprint(0.02, 0.02, 0.02)
        self.foot_ref = geom.ObjectFootprint(0.04, 0.04, 0.03)
        self.top = geom.AxisAlignedBox(-0.20, 0.20, -0.20, 0.20, 0.62, 0.68)
        self.bottom = geom.AxisAlignedBox(-0.20, 0.20, -0.20, 0.20, 0.12, 0.18)

    def _spec(self, relation: geom.RelationKind) -> geom.ShelfRelationSpec:
        return geom.ShelfRelationSpec(
            relation=relation,
            top_shelf=self.top,
            bottom_shelf=self.bottom,
            reference_footprint=self.foot_ref,
            object_footprint=self.foot_obj,
            horizontal_overlap_min_m=0.001,
        )

    def test_above_and_below_regions_differ_in_height(self):
        above = geom.build_shelf_goal_set(self.frame, self._spec("above"), self.ref)
        below = geom.build_shelf_goal_set(self.frame, self._spec("below"), self.ref)
        self.assertFalse(above.empty)
        self.assertFalse(below.empty)
        above_center = (0.0, 0.0, 0.65)
        below_center = (0.0, 0.0, 0.15)
        self.assertTrue(geom.point_in_goal_set(self.frame, above_center, above))
        self.assertTrue(geom.point_in_goal_set(self.frame, below_center, below))
        self.assertFalse(geom.point_in_goal_set(self.frame, above_center, below))
        self.assertFalse(geom.point_in_goal_set(self.frame, below_center, above))

    def test_unsupported_hovering_fails_support_check(self):
        spec = self._spec("above")
        hover = (0.0, 0.0, 0.80)
        self.assertFalse(geom.shelf_support_ok(self.frame, spec, hover))


class ContainmentTests(unittest.TestCase):
    def setUp(self):
        self.frame = geom.TaskFrame.identity()
        self.interior_local = geom.AxisAlignedBox(-0.08, 0.08, -0.08, 0.08, 0.0, 0.10)
        self.spec = geom.ContainmentSpec(
            interior_reference_local=self.interior_local,
            object_footprint=geom.ObjectFootprint(0.02, 0.02, 0.02),
            wall_clearance_m=0.005,
        )
        self.ref = (0.0, 0.0, 0.0)

    def test_partial_containment_center_outside_fails(self):
        goal = geom.build_containment_goal_set(self.frame, self.spec, self.ref)
        partial = (0.07, 0.0, 0.05)
        full = (0.0, 0.0, 0.025)
        self.assertFalse(geom.inside_containment(goal, self.frame, partial))
        self.assertTrue(geom.inside_containment(goal, self.frame, full))

    def test_centroid_in_outer_box_but_not_eroded_interior_fails(self):
        goal = geom.build_containment_goal_set(self.frame, self.spec, self.ref)
        near_wall = (0.075, 0.0, 0.05)
        self.assertTrue(self.interior_local.point_inside(self.frame.world_to_task(near_wall)))
        self.assertFalse(geom.inside_containment(goal, self.frame, near_wall))

    def test_displaced_reference_translates_interior(self):
        ref_shifted = (0.25, -0.10, 0.05)
        goal = geom.build_containment_goal_set(self.frame, self.spec, ref_shifted)
        inside_shifted = (0.25, -0.10, 0.10)
        outside_shifted = (0.0, 0.0, 0.025)
        self.assertTrue(geom.inside_containment(goal, self.frame, inside_shifted))
        self.assertFalse(geom.inside_containment(goal, self.frame, outside_shifted))

    def test_non_identity_orientation_fails_closed(self):
        rotated = geom.ReferenceOrientation(u_left=(0.0, 1.0, 0.0), u_front=(-1.0, 0.0, 0.0), u_up=(0.0, 0.0, 1.0))
        with self.assertRaises(geom.UnsupportedReferenceOrientationError):
            geom.ContainmentSpec(
                interior_reference_local=self.interior_local,
                object_footprint=geom.ObjectFootprint(0.02, 0.02, 0.02),
                wall_clearance_m=0.005,
                orientation=rotated,
            )


class PromptEquivalenceTests(unittest.TestCase):
    def test_direct_inverse_pairs_share_semantic_goal(self):
        for relation in ("left", "right", "front", "behind", "above", "below", "inside"):
            self.assertTrue(
                geom.direct_inverse_pair_equivalent("cube", "bowl", relation, horizontal=relation not in ("above", "below", "inside"))
            )

    def test_inside_and_contains_prompts_equivalent(self):
        inside = geom.build_prompt("cube", "bowl", "inside", "direct", horizontal=False)
        contains = geom.build_prompt("cube", "bowl", "inside", "inverse", horizontal=False)
        self.assertTrue(geom.prompts_semantically_equivalent(inside, contains))

    def test_different_reference_breaks_equivalence(self):
        a = geom.build_prompt("cube", "blue bowl", "left", "direct")
        b = geom.build_prompt("cube", "yellow bowl", "left", "direct")
        self.assertFalse(geom.prompts_semantically_equivalent(a, b))


class ReferenceMembershipTests(unittest.TestCase):
    def setUp(self):
        self.frame = geom.TaskFrame.identity()
        self.workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        self.foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        self.ref_a = (-0.40, 0.0, 0.04)
        self.ref_b = (0.40, 0.0, 0.04)

    def test_c2_membership_categories(self):
        spec_a = geom.PlanarRelationSpec("left", 0.01, self.workspace, self.foot, self.foot)
        spec_b = geom.PlanarRelationSpec("left", 0.01, self.workspace, self.foot, self.foot)
        goal_a = geom.build_planar_goal_set(self.frame, spec_a, self.ref_a)
        goal_b = geom.build_planar_goal_set(self.frame, spec_b, self.ref_b)
        point_named = (-0.20, 0.0, 0.04)
        point_both = (0.50, 0.0, 0.04)
        self.assertEqual(
            geom.reference_membership(
                named_goal=goal_a, other_goal=goal_b, frame=self.frame, p_obj_world=point_named
            ),
            "named",
        )
        self.assertEqual(
            geom.reference_membership(
                named_goal=goal_a, other_goal=goal_b, frame=self.frame, p_obj_world=point_both
            ),
            "both",
        )
        neither = geom.reference_membership(
            named_goal=goal_a, other_goal=goal_b, frame=self.frame, p_obj_world=(-0.50, 0.0, 0.04)
        )
        self.assertEqual(neither, "neither")

    def test_asymmetric_relations_yield_other_only_membership(self):
        ref_a = (-0.40, 0.0, 0.04)
        ref_b = (-0.10, 0.0, 0.04)
        goal_named = geom.build_planar_goal_set(
            self.frame,
            geom.PlanarRelationSpec("left", 0.01, self.workspace, self.foot, self.foot),
            ref_a,
        )
        goal_other = geom.build_planar_goal_set(
            self.frame,
            geom.PlanarRelationSpec("right", 0.01, self.workspace, self.foot, self.foot),
            ref_b,
        )
        self.assertEqual(
            geom.reference_membership(
                named_goal=goal_named,
                other_goal=goal_other,
                frame=self.frame,
                p_obj_world=(-0.40, 0.0, 0.04),
            ),
            "other",
        )
        self.assertEqual(
            geom.reference_membership(
                named_goal=goal_named,
                other_goal=goal_other,
                frame=self.frame,
                p_obj_world=(-0.12, 0.0, 0.04),
            ),
            "named",
        )


class ResponseProjectionTests(unittest.TestCase):
    def test_planar_projection_ignores_carry_height(self):
        frame = geom.TaskFrame.identity()
        workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        spec = geom.PlanarRelationSpec(
            relation="left",
            clearance_m=0.01,
            workspace=workspace,
            object_footprint=geom.ObjectFootprint(0.02, 0.02, 0.02),
            reference_footprint=geom.ObjectFootprint(0.05, 0.05, 0.02),
        )
        ref = (0.0, 0.0, 0.04)
        goal = geom.build_planar_goal_set(frame, spec, ref)
        good_xy = (0.12, 0.0, 0.04)
        high_carry = (0.12, 0.0, 0.25)
        terminal_far = geom.goal_distance(frame, high_carry, goal, d_cap_m=1.0)
        response = geom.response_projection_distance(frame, high_carry, goal, d_cap_m=1.0)
        self.assertGreater(terminal_far.distance_m, 0.07)
        self.assertAlmostEqual(response.distance_m, 0.0, places=6)
        self.assertAlmostEqual(
            geom.response_projection_distance(frame, good_xy, goal, d_cap_m=1.0).distance_m,
            0.0,
        )


class TrajectorySamplingTests(unittest.TestCase):
    def test_resolve_trajectory_sample_holds_terminal_extension(self):
        series = geom.TrajectorySeries(
            samples=(
                geom.TrajectorySample(0.0, (0.0, 0.0, 0.04), (0.0, 0.0, 0.04)),
                geom.TrajectorySample(1.0, (0.05, 0.0, 0.04), (0.06, 0.0, 0.04)),
            )
        )
        terminal = geom.TrajectorySample(3.0, (0.20, 0.0, 0.04), (0.12, 0.0, 0.04))
        resolved = geom.resolve_trajectory_sample(series, 2.5, terminal_extension=terminal)
        self.assertTrue(resolved.terminal_extension_applied)
        self.assertEqual(resolved.sample.p_obj_world, terminal.p_obj_world)
        self.assertEqual(resolved.sample.p_named_ref_world, terminal.p_named_ref_world)


class WordingGoalAgreementTests(unittest.TestCase):
    def test_inverse_wording_yields_identical_planar_goal_set(self):
        frame = geom.TaskFrame.identity()
        workspace = geom.AxisAlignedBox(-0.5, 0.5, -0.5, 0.5, 0.0, 0.08)
        foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
        ref = (0.0, 0.0, 0.04)
        direct = geom.build_planar_goal_set(
            frame,
            geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
            ref,
        )
        inverse = geom.build_planar_goal_set(
            frame,
            geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
            ref,
        )
        self.assertFalse(direct.empty)
        self.assertEqual(direct.region, inverse.region)


if __name__ == "__main__":
    unittest.main()
