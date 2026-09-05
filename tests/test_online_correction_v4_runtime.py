"""Deterministic unit tests for the V4 online-correction runtime core."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.attempts import (
    AttemptClassifier,
    InfraInvalidReason,
    RetryPolicy,
    TerminalEvidenceFlags,
    classify_terminal_outcome,
)
from experiments.online_correction_v4.clock import (
    ActionQueue,
    ControlledSimulationClock,
    PolicyRequest,
    QuerySchedule,
    quantize_upward,
)
from experiments.online_correction_v4.contracts import (
    AttemptStatus,
    CampaignPhase,
    FailureLabel,
    FailureStage,
    PolicyTimingAchieved,
    TimingConfig,
)
from experiments.online_correction_v4.detectors import (
    DetachmentDetector,
    DetachmentDetectorConfig,
    GraspDetectorConfig,
    NaturalGraspDetector,
    ObjectKinematicState,
)
from experiments.online_correction_v4.leases import (
    AttemptFinalizer,
    GroupLeaseStore,
    LeaseConflict,
    WriteOnceViolation,
)
from experiments.online_correction_v4.motion import (
    MotionProfileKind,
    ReferenceMotionController,
    minimum_jerk_scalar,
    peak_speed,
)
from experiments.online_correction_v4.registry import (
    CampaignRegistry,
    load_manifest_from_config,
)
from experiments.online_correction_v4.scheduler import CampaignScheduler, GateStatus


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "docs/online_correction_v4/campaign.json"


def _timing_config() -> TimingConfig:
    timing = json.loads(CONFIG_PATH.read_text())["timing"]
    return TimingConfig.from_mapping(timing)


def _achieved(native_dt: float = 0.05, horizon: int = 16) -> PolicyTimingAchieved:
    return PolicyTimingAchieved.from_requested(native_dt, _timing_config(), horizon)


def _clock(
    *,
    schedule: QuerySchedule = QuerySchedule.STANDARD,
    native_dt: float = 0.05,
) -> ControlledSimulationClock:
    timing = _timing_config()
    achieved = PolicyTimingAchieved.from_requested(native_dt, timing, 16)
    return ControlledSimulationClock(
        timing=timing,
        achieved=achieved,
        schedule=schedule,
        action_queue=ActionQueue(native_control_dt_s=native_dt),
    )


class MotionProfileTests(unittest.TestCase):
    def test_minimum_jerk_endpoints_and_peak_speed(self):
        self.assertAlmostEqual(minimum_jerk_scalar(0.0), 0.0)
        self.assertAlmostEqual(minimum_jerk_scalar(1.0), 1.0)
        self.assertTrue(0.0 < minimum_jerk_scalar(0.5) < 1.0)
        delta = 0.12
        duration = 0.5
        self.assertAlmostEqual(peak_speed(delta, duration), 1.875 * delta / duration)

    def test_profiles_match_registered_durations(self):
        motion_cfg = json.loads(CONFIG_PATH.read_text())["motion"]
        controller = ReferenceMotionController.from_scenario(
            "move_stop", displacement_m=0.12, motion_config=motion_cfg
        )
        controller.schedule_event(1.0)
        at_end = controller.pose_at(1.0 + float(motion_cfg["move_stop_duration_s"]))
        self.assertAlmostEqual(at_end.displacement_m, 0.12, places=9)
        self.assertAlmostEqual(at_end.velocity_m_s, 0.0, places=9)

    def test_displacement_vector_respects_goal_and_sign(self):
        left_pos = ReferenceMotionController.displacement_vector(
            goal="left", fixture="horizontal", physical_sign=1
        )
        left_neg = ReferenceMotionController.displacement_vector(
            goal="left", fixture="horizontal", physical_sign=-1
        )
        self.assertLess(left_pos[0], 0.0)
        self.assertGreater(left_neg[0], 0.0)

    def test_reversal_reaches_negative_half_displacement(self):
        motion_cfg = json.loads(CONFIG_PATH.read_text())["motion"]
        controller = ReferenceMotionController.from_scenario(
            "reversal", displacement_m=0.12, motion_config=motion_cfg
        )
        controller.schedule_event(0.0)
        final = controller.pose_at(4.0)
        self.assertAlmostEqual(final.displacement_m, -0.06, places=9)

    def test_motion_freezes_only_at_detection_tick(self):
        motion_cfg = json.loads(CONFIG_PATH.read_text())["motion"]
        controller = ReferenceMotionController.from_scenario(
            "move_stop", displacement_m=0.12, motion_config=motion_cfg
        )
        controller.schedule_event(0.0)
        before = controller.pose_at(0.20).displacement_m
        controller.freeze_at(0.25, reason="release_detected")
        after = controller.pose_at(0.50).displacement_m
        frozen = controller.pose_at(0.25).displacement_m
        self.assertGreater(before, 0.0)
        self.assertAlmostEqual(after, frozen)
        self.assertGreater(frozen, before)


class DetectorTests(unittest.TestCase):
    def test_natural_grasp_requires_lift_dwell_and_low_drift(self):
        cfg = GraspDetectorConfig()
        detector = NaturalGraspDetector(config=cfg, control_dt_s=0.05)
        base = ObjectKinematicState(
            sim_time=0.0,
            control_tick=0,
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
        self.assertIsNone(detector.update(base))
        eligible = ObjectKinematicState(
            **{**base.__dict__, "sim_time": 0.25, "control_tick": 5, "object_z_pos": 0.095}
        )
        event = detector.update(eligible)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertGreaterEqual(event.lift_m, cfg.min_lift_m)
        self.assertGreaterEqual(event.dwell_s, cfg.dwell_s)

    def test_first_detachment_requires_two_ticks(self):
        detector = DetachmentDetector(config=DetachmentDetectorConfig(dwell_ticks=2))
        detector.arm_after_verified_carry()
        tick0 = ObjectKinematicState(
            sim_time=0.0,
            control_tick=0,
            object_z=0.0,
            initial_supported_z=0.0,
            gripper_x=0.0,
            gripper_y=0.0,
            gripper_z=0.0,
            object_x=0.0,
            object_y=0.0,
            object_z_pos=0.0,
            detached=True,
        )
        self.assertIsNone(detector.update(tick0))
        tick1 = ObjectKinematicState(
            **{**tick0.__dict__, "sim_time": 0.05, "control_tick": 1}
        )
        event = detector.update(tick1)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertAlmostEqual(event.t_onset, 0.0)
        self.assertAlmostEqual(event.t_detected, 0.05)
        self.assertEqual(event.onset_tick, tick0.control_tick)
        self.assertEqual(event.detected_tick, 1)


class ClockInvariantTests(unittest.TestCase):
    """Runbook section 7 timing invariants."""

    def test_quantization_matches_release_check_rule(self):
        native_dt = 0.04
        timing = _timing_config()
        achieved = PolicyTimingAchieved.from_requested(native_dt, timing, 16)
        self.assertAlmostEqual(achieved.achieved_delay_s, 0.12)
        self.assertAlmostEqual(achieved.achieved_standard_query_period_s, 0.52)

    def test_invariant_a_no_response_before_availability(self):
        clock = _clock()
        events = clock.advance_to_next_query()
        request = clock.pending_requests[-1]
        clock.complete_inference(request, chunk_id="c1", actions=[(1.0,)], wall_duration_s=0.2)
        clock.advance_for_delay_window(request)
        self.assertFalse(request.applied)
        self.assertEqual(clock.action_queue.pending_count, 0)
        applied = clock.apply_due_responses()
        self.assertEqual(len(applied), 1)
        self.assertGreaterEqual(clock.sim_time, request.response_available_time or -1)

    def test_invariant_b_delay_window_uses_old_queue(self):
        clock = _clock()
        clock.action_queue.enqueue_chunk(
            chunk_id="old", request_id="old-req", actions=[(0.0,), (0.0,), (0.0,)]
        )
        events = clock.advance_to_next_query()
        request = clock.pending_requests[-1]
        clock.complete_inference(request, chunk_id="new", actions=[(9.0,)], wall_duration_s=0.01)
        executed = clock.advance_for_delay_window(request)
        values = [cmd.values[0] for cmd in executed if cmd is not None]
        self.assertEqual(values, [0.0, 0.0])
        clock.apply_due_responses()
        cmd = clock.action_queue.pop_for_tick()
        assert cmd is not None
        self.assertEqual(cmd.values[0], 9.0)

    def test_chunk_replacement_action_index_continues_executed_count(self):
        clock = _clock()
        clock.action_queue.enqueue_chunk(
            chunk_id="first",
            request_id="req-1",
            actions=[(1.0,), (2.0,), (3.0,)],
            start_index=0,
        )
        for _ in range(3):
            clock._advance_one_control_tick()
        self.assertEqual([cmd.action_index for cmd in clock.action_queue.executed], [0, 1, 2])
        events = clock.advance_to_next_query()
        request = clock.pending_requests[-1]
        clock.finish_query_cycle(
            request,
            chunk_id="second",
            actions=[(4.0,), (5.0,)],
            wall_duration_s=0.01,
        )
        while clock.action_queue.pending_count:
            clock._advance_one_control_tick()
        indices = [cmd.action_index for cmd in clock.action_queue.executed]
        self.assertEqual(indices, [0, 1, 2, 3, 4])

    def test_invariant_c_reference_moves_during_delay(self):
        motion_cfg = json.loads(CONFIG_PATH.read_text())["motion"]
        reference = ReferenceMotionController.from_scenario(
            "move_stop", displacement_m=0.12, motion_config=motion_cfg
        )
        reference.schedule_event(0.0)
        clock = _clock()
        seen: list[float] = []

        def advance_ref(t: float) -> None:
            seen.append(reference.pose_at(t).displacement_m)

        events = clock.advance_to_next_query()
        request = clock.pending_requests[-1]
        clock.finish_query_cycle(
            request,
            chunk_id="c1",
            actions=[(0.0,)],
            wall_duration_s=0.01,
            advance_reference=advance_ref,
        )
        self.assertTrue(any(displacement > 0.0 for displacement in seen))

    def test_invariant_d_response_not_from_future_observation(self):
        clock = _clock()
        clock.advance_to_next_query()
        request = clock.pending_requests[-1]
        obs_time = request.observation_capture_time
        clock.finish_query_cycle(
            request,
            chunk_id="c1",
            actions=[(1.0,)],
            wall_duration_s=0.5,
        )
        self.assertEqual(request.observation_capture_time, obs_time)
        self.assertLess(request.observation_capture_time, request.response_available_time or 0.0)

    def test_invariant_e_schedule_independent_of_wall_inference_duration(self):
        traces: list[list[float]] = []
        for wall in (0.01, 0.5, 2.0):
            clock = _clock()
            query_times: list[float] = []
            for _ in range(3):
                clock.advance_to_next_query()
                request = clock.pending_requests[-1]
                query_times.append(request.observation_capture_time)
                clock.finish_query_cycle(
                    request,
                    chunk_id=f"c-{wall}",
                    actions=[(1.0,)],
                    wall_duration_s=wall,
                )
            traces.append(query_times)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(traces[1], traces[2])

    def test_invariant_f_c4_pre_trigger_matches_standard_schedule(self):
        standard = _clock(schedule=QuerySchedule.STANDARD)
        fast = _clock(schedule=QuerySchedule.FAST_AFTER_GRASP)
        std_times: list[float] = []
        fast_times: list[float] = []
        for _ in range(4):
            standard.advance_to_next_query()
            req = standard.pending_requests[-1]
            std_times.append(req.observation_capture_time)
            standard.finish_query_cycle(req, chunk_id="s", actions=[(0.0,)], wall_duration_s=0.01)
            fast.advance_to_next_query()
            req = fast.pending_requests[-1]
            fast_times.append(req.observation_capture_time)
            fast.finish_query_cycle(req, chunk_id="f", actions=[(0.0,)], wall_duration_s=0.01)
        self.assertEqual(std_times, fast_times)
        standard.register_natural_grasp(1.0)
        fast.register_natural_grasp(1.0)
        self.assertTrue(fast.fast_schedule_active)
        self.assertLess(fast.query_period, standard.query_period)

    def test_action_buffer_crosses_query_boundary(self):
        clock = _clock()
        clock.action_queue.enqueue_chunk(
            chunk_id="seed", request_id="seed", actions=[(0.0,)] * 10
        )
        clock.advance_to_next_query()
        request = clock.pending_requests[-1]
        result = clock.finish_query_cycle(
            request, chunk_id="next", actions=[(1.0,)] * 4, wall_duration_s=0.01
        )
        self.assertGreater(len(result["delay_actions"]), 0)

    def test_event_phase_uses_standard_period_even_for_fast_schedule(self):
        clock = _clock(schedule=QuerySchedule.FAST_AFTER_GRASP)
        clock.event_phase_fraction = 0.25
        clock.register_natural_grasp(0.0)
        onset = clock.plan_event_onset_after_grasp()
        self.assertAlmostEqual(
            onset,
            quantize_upward(
                0.25 * clock.achieved.achieved_standard_query_period_s,
                clock.achieved.native_control_dt_s,
            ),
        )

    def test_passive_settling_lasts_exactly_one_second(self):
        clock = _clock()
        detection_time = 2.0
        clock.sim_time = detection_time
        clock.control_tick = int(detection_time / clock.achieved.native_control_dt_s)
        clock.start_passive_settling(detection_time)
        end_time = None
        while clock.passive_settling_active:
            clock._advance_one_control_tick()
            if clock.episode_end_time is not None:
                end_time = clock.episode_end_time
        self.assertAlmostEqual(end_time or -1, detection_time + 1.0, places=9)

    def test_first_detachment_stops_policy_requests(self):
        clock = _clock()
        clock.register_natural_grasp(0.0)
        clock.start_passive_settling(0.25)
        self.assertFalse(clock.policy_phase_active)
        pending_before = len(clock.pending_requests)
        events = clock.advance_to_next_query()
        self.assertEqual(events, [])
        self.assertEqual(len(clock.pending_requests), pending_before)


class LeaseAndAttemptTests(unittest.TestCase):
    def test_group_lease_is_exclusive_and_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GroupLeaseStore(Path(tmp))
            lease = store.acquire(
                group_id="cosmos3_nano_droid:horizontal",
                owner_lane="lane-a",
                attempt_id="attempt-1",
                manifest_sha256="abc",
            )
            self.assertTrue(store.verify(lease))
            with self.assertRaises(LeaseConflict):
                store.acquire(
                    group_id="cosmos3_nano_droid:horizontal",
                    owner_lane="lane-b",
                    attempt_id="attempt-2",
                    manifest_sha256="abc",
                )
            store.release(lease)
            lease2 = store.acquire(
                group_id="cosmos3_nano_droid:horizontal",
                owner_lane="lane-b",
                attempt_id="attempt-2",
                manifest_sha256="abc",
            )
            self.assertEqual(lease2.owner_lane, "lane-b")

    def test_attempt_finalization_is_write_once(self):
        finalizer, tmp = AttemptFinalizer.with_temp_store()
        with tmp:
            path = finalizer.begin_attempt(
                episode_id="ep-1",
                attempt_id="a001",
                metadata={"episode_id": "ep-1"},
            )
            finalizer.write_incremental(path, "trace.json", {"steps": 1})
            receipt = {"status": "valid", "episode_id": "ep-1"}
            finalizer.finalize(path, receipt)
            self.assertTrue(finalizer.is_finalized(path))
            with self.assertRaises(WriteOnceViolation):
                finalizer.finalize(path, receipt)
            with self.assertRaises(WriteOnceViolation):
                finalizer.begin_attempt(
                    episode_id="ep-1",
                    attempt_id="a001",
                    metadata={"episode_id": "ep-1"},
                )


class AttemptClassificationTests(unittest.TestCase):
    def test_infra_invalid_is_retryable_but_behavioral_failure_is_not(self):
        classifier = AttemptClassifier(retry_policy=RetryPolicy(max_retries=2))
        infra = classifier.classify_infra_invalid(
            episode_id="ep",
            attempt_id="a1",
            reason=InfraInvalidReason.SIMULATOR_CRASH,
            prior_infra_attempts=1,
        )
        self.assertTrue(classifier.authorize_retry(infra))
        behavioral = classifier.classify_behavioral_failure(failure_label=FailureLabel.NO_GRASP)
        self.assertFalse(classifier.authorize_retry(behavioral))

    def test_classify_terminal_outcome_preserves_valid_no_grasp(self):
        flags = TerminalEvidenceFlags(
            grasp_occurred=False,
            timeout_without_completion=True,
            timeout_after_no_grasp=True,
        )
        status, label, stage, meta = classify_terminal_outcome(flags)
        self.assertEqual(status, AttemptStatus.VALID)
        self.assertEqual(label.value, "no_grasp")
        self.assertEqual(stage, FailureStage.PICKUP)


class RegistryAndSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = CampaignRegistry.from_rows(
            load_manifest_from_config(CONFIG_PATH),
            config_sha256="test",
        )

    def test_registry_loads_full_manifest(self):
        self.assertEqual(len(self.registry.rows), 17664)
        self.assertEqual(len(self.registry.by_episode_id), 17664)

    def test_scheduler_enforces_c3_c4_dependencies(self):
        scheduler = CampaignScheduler(registry=self.registry)
        errors = scheduler.set_release_state(
            released={"C3"},
            blocked={fid: "blocked" for fid in sorted(self.registry.families_present() - {"C3"})},
        )
        self.assertTrue(any("C1" in err for err in errors))
        all_families = self.registry.families_present()
        released = all_families - {"C8"}
        blocked = {"C8": "bridge unavailable"}
        scheduler = CampaignScheduler(registry=self.registry)
        errors = scheduler.set_release_state(released=released, blocked=blocked)
        self.assertEqual(errors, [])
        self.assertTrue(scheduler.family_dispatchable("C3"))
        self.assertTrue(scheduler.family_dispatchable("C4"))

    def test_scheduler_group_completion_requires_all_cells(self):
        scheduler = CampaignScheduler(registry=self.registry)
        scheduler.set_release_state(
            released={"C1"},
            blocked={fid: "blocked" for fid in sorted(self.registry.families_present() - {"C1"})},
        )
        for gate in ("G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"):
            scheduler.record_gate(gate, GateStatus.PASSED)
        scheduler.advance_to_piloting()
        scheduler.advance_to_frozen()
        scheduler.advance_to_running()
        group = next(
            g
            for g in scheduler.registry.iter_groups()
            if g.policy == "cosmos3_nano_droid" and g.fixture == "horizontal"
        )
        c1_episodes = [
            scheduler.registry.get(episode_id).episode_id
            for episode_id in group.episode_ids
            if scheduler.registry.get(episode_id).family == "C1"
        ]
        with tempfile.TemporaryDirectory() as tmp:
            scheduler.lease_store = GroupLeaseStore(Path(tmp))
            scheduler.acquire_group_lease(
                group_id=group.group_id,
                owner_lane="lane-1",
                attempt_id="attempt-1",
                manifest_sha256="manifest",
            )
        first = c1_episodes[0]
        scheduler.record_valid_episode(first)
        self.assertNotEqual(
            scheduler.group_states[group.group_id].value,
            "complete",
        )
        for episode_id in c1_episodes:
            scheduler.record_valid_episode(episode_id)
        self.assertEqual(scheduler.group_states[group.group_id].value, "complete")


if __name__ == "__main__":
    unittest.main()
