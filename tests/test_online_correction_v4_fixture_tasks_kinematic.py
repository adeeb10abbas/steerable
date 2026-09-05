"""Tests for V4 kinematic reference motion and timeout-only task termination metadata."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from experiments.online_correction_v4.droid_task_files.horizontal_core import termination_field_names
from experiments.online_correction_v4.droid_task_files.kinematic_reference import KinematicReferenceMotion
from experiments.online_correction_v4.motion import ReferenceMotionController


@dataclass
class FakeRootAsset:
    pose: tuple[float, float, float, float, float, float, float] = (
        0.55,
        0.12,
        0.82,
        1.0,
        0.0,
        0.0,
        0.0,
    )
    write_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = field(default_factory=list)

    def write_root_pose_to_sim(self, pose) -> None:
        self.pose = tuple(float(item) for item in pose)
        if self.write_calls:
            last_velocity = self.write_calls[-1][1]
        else:
            last_velocity = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.write_calls.append((self.pose, last_velocity))

    def write_root_velocity_to_sim(self, velocity) -> None:
        zero = tuple(float(item) for item in velocity)
        if self.write_calls:
            pose, _ = self.write_calls[-1]
            self.write_calls[-1] = (pose, zero)
        else:
            self.write_calls.append((self.pose, zero))


class TimeoutTerminationTests(unittest.TestCase):
    def test_timeout_only_termination_has_no_success_terms(self) -> None:
        fields = termination_field_names()
        self.assertEqual(fields, frozenset({"time_out"}))
        self.assertNotIn("success", fields)


class KinematicReferenceMotionTests(unittest.TestCase):
    def test_initial_pose_anchoring_prevents_cumulative_drift(self) -> None:
        asset = FakeRootAsset()
        controller = ReferenceMotionController.from_scenario(
            "move_stop",
            displacement_m=0.12,
            motion_config={
                "move_stop_duration_s": 0.5,
                "slow_drift_duration_s": 4.0,
                "fast_drift_duration_s": 1.0,
                "reversal_waypoints": [
                    {"time_s": 0.0, "displacement_units": 0.0},
                    {"time_s": 2.0, "displacement_units": 1.0},
                    {"time_s": 4.0, "displacement_units": -0.5},
                ],
            },
        )
        controller.schedule_event(1.0)
        motion = KinematicReferenceMotion(
            writer=asset,
            motion_controller=controller,
            direction=(1.0, 0.0),
        )
        initial = asset.pose
        motion.anchor_initial_pose(initial)

        first = motion.apply_at(1.1)
        second = motion.apply_at(1.2)

        self.assertGreater(second[0] - initial[0], first[0] - initial[0])

        # Re-applying an earlier time must reproduce the anchored pose, not accumulate drift.
        replay_first = motion.pose_at(1.1)
        self.assertEqual(replay_first, first)
        self.assertEqual(motion.pose_at(1.0)[:3], initial[:3])

    def test_apply_writes_zero_root_velocity(self) -> None:
        asset = FakeRootAsset()
        controller = ReferenceMotionController.from_scenario(
            "original_sham",
            displacement_m=0.12,
            motion_config={
                "move_stop_duration_s": 0.5,
                "slow_drift_duration_s": 4.0,
                "fast_drift_duration_s": 1.0,
                "reversal_waypoints": [
                    {"time_s": 0.0, "displacement_units": 0.0},
                    {"time_s": 2.0, "displacement_units": 1.0},
                    {"time_s": 4.0, "displacement_units": -0.5},
                ],
            },
        )
        motion = KinematicReferenceMotion(
            writer=asset,
            motion_controller=controller,
            direction=(0.0, 1.0),
        )
        motion.anchor_initial_pose(asset.pose)
        controller.schedule_event(0.0)
        motion.apply_at(0.1)
        self.assertEqual(asset.write_calls[-1][1], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_unanchored_motion_fails_closed(self) -> None:
        asset = FakeRootAsset()
        controller = ReferenceMotionController.from_scenario(
            "original_sham",
            displacement_m=0.12,
            motion_config={
                "move_stop_duration_s": 0.5,
                "slow_drift_duration_s": 4.0,
                "fast_drift_duration_s": 1.0,
                "reversal_waypoints": [
                    {"time_s": 0.0, "displacement_units": 0.0},
                    {"time_s": 2.0, "displacement_units": 1.0},
                    {"time_s": 4.0, "displacement_units": -0.5},
                ],
            },
        )
        motion = KinematicReferenceMotion(
            writer=asset,
            motion_controller=controller,
            direction=(1.0, 0.0),
        )
        with self.assertRaises(RuntimeError):
            motion.apply_at(0.0)

    def test_freeze_holds_displacement_at_detection_time(self) -> None:
        asset = FakeRootAsset()
        controller = ReferenceMotionController.from_scenario(
            "move_stop",
            displacement_m=0.12,
            motion_config={
                "move_stop_duration_s": 0.5,
                "slow_drift_duration_s": 4.0,
                "fast_drift_duration_s": 1.0,
                "reversal_waypoints": [
                    {"time_s": 0.0, "displacement_units": 0.0},
                    {"time_s": 2.0, "displacement_units": 1.0},
                    {"time_s": 4.0, "displacement_units": -0.5},
                ],
            },
        )
        controller.schedule_event(0.0)
        motion = KinematicReferenceMotion(
            writer=asset,
            motion_controller=controller,
            direction=(1.0, 0.0),
        )
        motion.anchor_initial_pose(asset.pose)
        frozen = motion.freeze_at(0.25, reason="first_placement_detected")
        later = motion.apply_at(0.50)
        self.assertEqual(frozen[:3], later[:3])


if __name__ == "__main__":
    unittest.main()
