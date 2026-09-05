"""Tests for V4 P1 protocol gaps: terminal stability and event-observed visibility."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.adapters import CapturedObservation
from experiments.online_correction_v4.clock import ActionQueue, ControlledSimulationClock, QuerySchedule
from experiments.online_correction_v4.contracts import EpisodeManifestRow, PolicyTimingAchieved, TimingConfig
from experiments.online_correction_v4.observation_audit import (
    ObservationAuditEvidence,
    evaluate_changed_observation_visibility,
)
from experiments.online_correction_v4.preparation import FrozenChangedObservationDetector
from experiments.online_correction_v4.recorder import EpisodeEvidenceRecorder
from experiments.online_correction_v4.runner import EpisodeRunConfig, EpisodeRunner
from experiments.online_correction_v4.leases import AttemptFinalizer
from experiments.online_correction_v4.testing import FakeViewportVideoWriter, ScriptedPolicy, ScriptedSimulator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "docs/online_correction_v4/campaign.json").read_text())


def _manifest_row(**overrides) -> EpisodeManifestRow:
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
        "factors": {},
        "prefix_group_id": "prefix",
        "execution_group": "cosmos3_nano_droid:horizontal",
        "execution_order_key": "order",
        "config_sha256": "abc",
        "reuse_episode_ids": [],
        "counterbalance": {"event_phase_fraction": 0.0},
        "prompt_recipe": {},
    }
    base.update(overrides)
    return EpisodeManifestRow.from_manifest_dict(base)


def _runner(*, simulator: ScriptedSimulator) -> tuple[EpisodeRunner, tempfile.TemporaryDirectory[str]]:
    timing = TimingConfig.from_mapping(CONFIG["timing"])
    native_dt = 0.05
    achieved = PolicyTimingAchieved.from_requested(native_dt, timing, 16)
    clock = ControlledSimulationClock(
        timing=timing,
        achieved=achieved,
        schedule=QuerySchedule.STANDARD,
        action_queue=ActionQueue(native_control_dt_s=native_dt),
    )
    tmp = tempfile.TemporaryDirectory()
    finalizer = AttemptFinalizer(Path(tmp.name))
    recorder = EpisodeEvidenceRecorder.open(
        finalizer=finalizer,
        episode_id="test-episode",
        attempt_id="a001",
        metadata={"episode_id": "test-episode", "attempt_id": "a001"},
    )
    runner = EpisodeRunner(
        manifest_row=_manifest_row(),
        timing=timing,
        clock=clock,
        simulator=simulator,
        policy=ScriptedPolicy(),
        recorder=recorder,
        run_config=EpisodeRunConfig(
            displacement_m=0.12,
            motion_direction=(1.0, 0.0),
            prompt_text="Place the cube.",
            scenario="move_stop",
            motion_config=CONFIG["motion"],
            viewport_video_required=False,
            trajectory_flush_interval=5,
        ),
        viewport_writer=FakeViewportVideoWriter(),
    )
    return runner, tmp


class ChangedObservationVisibilityTests(unittest.TestCase):
    def test_valid_camera_mask_qualifies(self) -> None:
        audit = ObservationAuditEvidence(
            reference_displacement_m=0.002,
            camera_ids=("head", "wrist"),
            moved_object_mask_pixels_by_camera={"head": 12, "wrist": 0},
        )
        result = evaluate_changed_observation_visibility(
            audit,
            displacement_threshold_m=0.001,
            policy_camera_ids=("head", "wrist"),
        )
        self.assertTrue(result.qualified)
        self.assertEqual(result.qualifying_camera_id, "head")

    def test_wrong_camera_mask_does_not_qualify(self) -> None:
        audit = ObservationAuditEvidence(
            reference_displacement_m=0.002,
            camera_ids=("head", "wrist"),
            moved_object_mask_pixels_by_camera={"external_only": 50},
        )
        result = evaluate_changed_observation_visibility(
            audit,
            displacement_threshold_m=0.001,
            policy_camera_ids=("head", "wrist"),
        )
        self.assertFalse(result.qualified)
        self.assertEqual(result.reason, "no_nonempty_mask_in_policy_cameras")

    def test_absent_mask_fails_closed_without_displacement_fallback(self) -> None:
        audit = ObservationAuditEvidence(
            reference_displacement_m=0.05,
            camera_ids=("head", "wrist"),
            moved_object_mask_pixels_by_camera={},
        )
        result = evaluate_changed_observation_visibility(
            audit,
            displacement_threshold_m=0.001,
            policy_camera_ids=("head", "wrist"),
        )
        self.assertFalse(result.qualified)
        self.assertEqual(result.reason, "missing_visibility_evidence")

    def test_frozen_visibility_detector_requires_policy_camera_mask(self) -> None:
        detector = FrozenChangedObservationDetector(displacement_threshold_m=0.001)
        self.assertFalse(
            detector.qualifies_changed_observation(
                observation_id="obs-1",
                reference_displacement_m=0.01,
                moved_object_mask_pixels=0,
                camera_ids=("head", "wrist"),
                moved_object_mask_pixels_by_camera={},
            )
        )
        self.assertTrue(
            detector.qualifies_changed_observation(
                observation_id="obs-2",
                reference_displacement_m=0.01,
                moved_object_mask_pixels=0,
                camera_ids=("head", "wrist"),
                moved_object_mask_pixels_by_camera={"wrist": 3},
            )
        )


class EventObservedRunnerTests(unittest.TestCase):
    def _carry_script(self, sim_time: float, tick: int) -> tuple[bool, bool, float]:
        contact = tick >= 2 and tick < 30
        detached = tick >= 30
        z = 0.095 if contact else 0.0
        return contact, detached, z

    def _captured(
        self,
        *,
        displacement_m: float,
        mask_by_camera: dict[str, int] | None,
        camera_ids: tuple[str, ...] = ("head", "wrist"),
    ) -> CapturedObservation:
        sim = ScriptedSimulator(
            reference_displacement_m=displacement_m,
            camera_ids=camera_ids,
            moved_object_mask_pixels_by_camera=mask_by_camera or {},
        )
        return sim.capture_observation()

    def test_valid_camera_mask_marks_event_observed(self) -> None:
        runner, tmp = _runner(simulator=ScriptedSimulator(carry_script=self._carry_script))
        runner.flags.event_delivered = True
        captured = self._captured(
            displacement_m=0.002,
            mask_by_camera={"head": 4},
            camera_ids=("head", "wrist"),
        )
        with tmp:
            runner._maybe_mark_event_observed(captured, request_id="req-1")
            self.assertTrue(runner.flags.event_observed)
            observed = [event for event in runner.recorder.events if event.get("kind") == "event_observed"]
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["qualifying_camera_id"], "head")

    def test_absent_mask_records_event_not_observed(self) -> None:
        runner, tmp = _runner(simulator=ScriptedSimulator(carry_script=self._carry_script))
        runner.flags.event_delivered = True
        captured = self._captured(displacement_m=0.01, mask_by_camera={})
        with tmp:
            runner._maybe_mark_event_observed(captured, request_id="req-1")
            self.assertFalse(runner.flags.event_observed)
            not_observed = [
                event for event in runner.recorder.events if event.get("kind") == "event_not_observed"
            ]
            self.assertEqual(len(not_observed), 1)
            self.assertEqual(not_observed[0]["reason"], "missing_visibility_evidence")

    def test_wrong_camera_mask_does_not_count_observed(self) -> None:
        runner, tmp = _runner(simulator=ScriptedSimulator(carry_script=self._carry_script))
        runner.flags.event_delivered = True
        captured = self._captured(
            displacement_m=0.01,
            mask_by_camera={"external_only": 99},
            camera_ids=("head", "wrist"),
        )
        with tmp:
            runner._maybe_mark_event_observed(captured, request_id="req-1")
            self.assertFalse(runner.flags.event_observed)
            not_observed = [
                event for event in runner.recorder.events if event.get("kind") == "event_not_observed"
            ]
            self.assertEqual(len(not_observed), 1)
            self.assertEqual(not_observed[0]["reason"], "no_nonempty_mask_in_policy_cameras")


if __name__ == "__main__":
    unittest.main()
