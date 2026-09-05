"""Focused tests for live motion derivation and terminal predicate scoring."""

from __future__ import annotations

import math
import unittest

from experiments.online_correction_v4.adapters import TerminalPhysicalPredicates
from experiments.online_correction_v4.contracts import EpisodeManifestRow, EpisodeRuntimeFlags, TimingConfig
from experiments.online_correction_v4.detectors import GraspDetectorConfig, NaturalGraspDetector, ObjectKinematicState
from experiments.online_correction_v4.droid_bindings import resolve_motion_direction
from experiments.online_correction_v4.droid_scorer import HorizontalDroidTerminalScorer, aggregate_settling_predicates
from experiments.online_correction_v4.droid_simulator import FakeRoboLabEnv
from experiments.online_correction_v4.motion import MotionDirectionError, ReferenceMotionController
from experiments.online_correction_v4 import geometry as geom
from experiments.online_correction_v4.adapters import SimulatorSnapshot
from experiments.online_correction_v4.scoring import ScoringContext


def _manifest(**overrides) -> EpisodeManifestRow:
    base = {
        "schema_version": 1,
        "manifest_type": "planning_manifest",
        "runtime_bound": False,
        "episode_id": "test-episode",
        "campaign": "online_correction_v4",
        "family": "C1",
        "fixture": "horizontal",
        "block_id": 0,
        "block_key": "k",
        "env_seed": 1,
        "policy_seed": 2,
        "cohort": "confirmatory",
        "priority": "primary",
        "factors": {"goal": "left"},
        "prefix_group_id": "prefix",
        "execution_group": "g",
        "execution_order_key": "order",
        "config_sha256": "abc",
        "reuse_episode_ids": [],
        "counterbalance": {"physical_translation_sign": 1},
        "prompt_recipe": {},
    }
    base.update(overrides)
    return EpisodeManifestRow.from_manifest_dict(base)


def _scoring_context() -> ScoringContext:
    workspace = geom.AxisAlignedBox(-1.0, 1.0, -1.0, 1.0, 0.0, 0.2)
    foot = geom.ObjectFootprint(0.02, 0.02, 0.02)
    return ScoringContext(
        frame=geom.TaskFrame.identity(),
        d_cap_m=0.12,
        planar_spec=geom.PlanarRelationSpec("left", 0.01, workspace, foot, foot),
    )


class MotionDirectionTests(unittest.TestCase):
    def test_wrong_sign_inverts_horizontal_axis(self) -> None:
        positive = ReferenceMotionController.resolve_live_direction(
            fixture="horizontal",
            goal="left",
            counterbalance={"physical_translation_sign": 1},
        )
        negative = ReferenceMotionController.resolve_live_direction(
            fixture="horizontal",
            goal="left",
            counterbalance={"physical_translation_sign": -1},
        )
        self.assertAlmostEqual(positive[0], -1.0 / math.sqrt(1.0))
        self.assertAlmostEqual(negative[0], 1.0 / math.sqrt(1.0))
        self.assertNotEqual(positive, negative)

    def test_diagonal_reference_binding_uses_counterbalance_signs(self) -> None:
        direction = ReferenceMotionController.resolve_live_direction(
            fixture="reference_binding",
            goal="left",
            counterbalance={
                "physical_translation_sign": 1,
                "physical_A_diagonal_signs": [-1, 1],
            },
            supported_fixtures=("horizontal", "reference_binding"),
        )
        norm = math.sqrt(2.0)
        self.assertAlmostEqual(direction[0], -1.0 / norm)
        self.assertAlmostEqual(direction[1], 1.0 / norm)

    def test_unsupported_fixture_fails_closed(self) -> None:
        manifest = _manifest(fixture="reference_binding", factors={"goal": "left"})
        with self.assertRaises(MotionDirectionError):
            resolve_motion_direction(manifest)

    def test_missing_physical_sign_fails_closed(self) -> None:
        manifest = _manifest(counterbalance={})
        with self.assertRaises(MotionDirectionError):
            resolve_motion_direction(manifest)


class TerminalPredicateTests(unittest.TestCase):
    def test_unsupported_drop_does_not_count_as_success(self) -> None:
        env = FakeRoboLabEnv()
        env.support_contacts = ()
        env.object_state = ObjectKinematicState(
            sim_time=1.0,
            control_tick=20,
            object_z=0.5,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.2,
            object_x=0.0,
            object_y=0.0,
            object_z_pos=0.5,
            contact=False,
            detached=True,
        )
        predicates = env.sample_terminal_predicates()
        self.assertFalse(predicates.allowed_support)

        scorer = HorizontalDroidTerminalScorer(
            relation="left",
            ctx=_scoring_context(),
            timing=TimingConfig(),
        )
        snapshot = SimulatorSnapshot(
            sim_time=1.0,
            control_tick=20,
            object_state=env.object_state,
            reference_position_world=(0.0, 0.0, 0.0),
            terminal_predicates=predicates,
        )
        evidence = scorer.score_terminal(
            snapshot=snapshot,
            runtime_flags=EpisodeRuntimeFlags(trigger_eligible=True, event_delivered=True),
            passive_settling_reason="release",
            grasp_occurred=True,
            carry_verified=True,
            settling_predicates=(predicates, predicates),
        )
        self.assertFalse(evidence.success)
        self.assertFalse(evidence.allowed_support)

    def test_spinning_object_fails_terminal_stability(self) -> None:
        env = FakeRoboLabEnv()
        env.object_state = ObjectKinematicState(
            sim_time=1.0,
            control_tick=20,
            object_z=0.0,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.2,
            object_x=0.0,
            object_y=0.0,
            object_z_pos=0.0,
            contact=False,
            detached=True,
        )
        env.object_linear_speed_m_s = 0.0
        env.object_angular_speed_rad_s = 0.5
        env.anchor_passive_settling_baseline()
        predicates = env.sample_terminal_predicates()
        self.assertFalse(predicates.stable_for_dwell)
        self.assertGreater(predicates.angular_speed_rad_s, 0.20)

    def test_missing_support_contact_evidence_fails_closed(self) -> None:
        env = FakeRoboLabEnv()
        env.support_contacts = None
        env.object_state = ObjectKinematicState(
            sim_time=1.0,
            control_tick=20,
            object_z=0.0,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.2,
            object_x=0.0,
            object_y=0.0,
            object_z_pos=0.0,
            contact=False,
            detached=True,
        )
        predicates = env.sample_terminal_predicates()
        self.assertFalse(predicates.allowed_support)
        self.assertFalse(predicates.support_evidence_available)
        self.assertIn("support_contact_evidence", predicates.missing_fields)
        stable = TerminalPhysicalPredicates(
            available=True,
            allowed_support=True,
            stable_for_dwell=True,
        )
        unstable = TerminalPhysicalPredicates(
            available=True,
            allowed_support=True,
            stable_for_dwell=False,
        )
        aggregated = aggregate_settling_predicates((stable, unstable), dwell_ticks=2)
        self.assertFalse(aggregated.stable_for_dwell)

    def test_missing_predicate_fails_closed(self) -> None:
        scorer = HorizontalDroidTerminalScorer(
            relation="left",
            ctx=_scoring_context(),
            timing=TimingConfig(),
        )
        state = ObjectKinematicState(
            sim_time=1.0,
            control_tick=20,
            object_z=0.0,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.2,
            object_x=-0.2,
            object_y=0.0,
            object_z_pos=0.0,
            contact=False,
            detached=True,
        )
        snapshot = SimulatorSnapshot(
            sim_time=1.0,
            control_tick=20,
            object_state=state,
            reference_position_world=(0.0, 0.0, 0.0),
        )
        evidence = scorer.score_terminal(
            snapshot=snapshot,
            runtime_flags=EpisodeRuntimeFlags(trigger_eligible=True, event_delivered=True),
            passive_settling_reason="release",
            grasp_occurred=True,
            carry_verified=True,
            settling_predicates=(),
        )
        self.assertTrue(evidence.unresolved_behavioral_failure)
        self.assertFalse(evidence.predicates_available)


class GraspFlagSeparationTests(unittest.TestCase):
    def test_grasp_occurred_before_carry_verified(self) -> None:
        cfg = GraspDetectorConfig(dwell_s=0.20)
        detector = NaturalGraspDetector(config=cfg, control_dt_s=0.05)
        early = ObjectKinematicState(
            sim_time=0.05,
            control_tick=1,
            object_z=0.0,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.10,
            object_x=0.0,
            object_y=0.0,
            object_z_pos=0.095,
            contact=True,
        )
        self.assertIsNone(detector.update(early))
        self.assertTrue(detector.grasp_occurred)
        self.assertFalse(detector.carry_verified)
        self.assertFalse(detector.eligible)


if __name__ == "__main__":
    unittest.main()
