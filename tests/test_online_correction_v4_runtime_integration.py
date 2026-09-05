"""Integration-style deterministic tests for episode runtime orchestration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.online_correction_v4.clock import ActionQueue, ControlledSimulationClock, QuerySchedule
from experiments.online_correction_v4.contracts import PolicyTimingAchieved, TimingConfig
from experiments.online_correction_v4.detectors import (
    DetachmentDetector,
    DetachmentDetectorConfig,
    GraspDetectorConfig,
    NaturalGraspDetector,
    ObjectKinematicState,
)
from experiments.online_correction_v4.motion import ReferenceMotionController


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "docs/online_correction_v4/campaign.json").read_text())


class EpisodeOrchestrationTests(unittest.TestCase):
    def test_release_freezes_motion_and_runs_settling(self):
        timing = TimingConfig.from_mapping(CONFIG["timing"])
        native_dt = 0.05
        achieved = PolicyTimingAchieved.from_requested(native_dt, timing, 16)
        clock = ControlledSimulationClock(
            timing=timing,
            achieved=achieved,
            schedule=QuerySchedule.STANDARD,
            action_queue=ActionQueue(native_control_dt_s=native_dt),
        )
        reference = ReferenceMotionController.from_scenario(
            "move_stop",
            displacement_m=0.12,
            motion_config=CONFIG["motion"],
        )
        grasp = NaturalGraspDetector(
            config=GraspDetectorConfig(
                min_lift_m=timing.natural_grasp_min_lift_m,
                dwell_s=timing.natural_grasp_dwell_s,
                relative_drift_max_m=timing.kinematic_grasp_relative_drift_max_m,
                trigger_deadline_s=timing.trigger_deadline_s,
            ),
            control_dt_s=native_dt,
        )
        detach = DetachmentDetector(
            config=DetachmentDetectorConfig(dwell_ticks=timing.release_detection_dwell_ticks)
        )
        detach.arm_after_verified_carry()

        # Establish natural grasp on discrete ticks before any policy query.
        for tick in range(6):
            sim_time = tick * native_dt
            state = ObjectKinematicState(
                sim_time=sim_time,
                control_tick=tick,
                object_z=0.095,
                initial_supported_z=0.0,
                gripper_x=0.0,
                gripper_y=0.0,
                gripper_z=0.10,
                object_x=0.0,
                object_y=0.0,
                object_z_pos=0.095,
                contact=True,
                detached=False,
            )
            grasp.update(state)
        self.assertTrue(grasp.eligible)
        event = grasp.event
        assert event is not None
        clock.register_natural_grasp(event.t_eligible)
        clock.event_phase_fraction = 0.0
        onset = clock.plan_event_onset_after_grasp()
        reference.schedule_event(onset)

        # Two-tick detachment well after motion begins.
        detachment_time = onset + 0.15
        for offset in (0, 1):
            tick = int(detachment_time / native_dt) + offset
            sim_time = tick * native_dt
            detach.update(
                ObjectKinematicState(
                    sim_time=sim_time,
                    control_tick=tick,
                    object_z=0.05,
                    initial_supported_z=0.0,
                    gripper_x=0.0,
                    gripper_y=0.0,
                    gripper_z=0.10,
                    object_x=0.0,
                    object_y=0.0,
                    object_z_pos=0.05,
                    contact=False,
                    detached=True,
                )
            )
        self.assertTrue(detach.detected)
        detachment = detach.event
        assert detachment is not None
        moving_at_onset = reference.pose_at(detachment.t_onset + 0.05).displacement_m
        reference.freeze_at(detachment.t_detected, reason="release_detected")
        frozen = reference.pose_at(detachment.t_detected).displacement_m
        clock.start_passive_settling(detachment.t_detected)
        while clock.passive_settling_active:
            clock._advance_one_control_tick()

        self.assertGreater(moving_at_onset, 0.0)
        self.assertAlmostEqual(
            reference.pose_at(clock.sim_time).displacement_m,
            frozen,
            places=9,
        )
        self.assertAlmostEqual(
            (clock.episode_end_time or -1) - detachment.t_detected,
            timing.release_settling_s,
            places=9,
        )
        self.assertFalse(clock.policy_phase_active)


if __name__ == "__main__":
    unittest.main()
