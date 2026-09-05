"""Extended runtime tests for runner, recorder, preparation gates, and leases."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.online_correction_v4.attempts import (
    FailureLabel,
    TerminalEvidenceFlags,
    classify_terminal_outcome,
    derive_failure_label,
)
from experiments.online_correction_v4.clock import ActionQueue, ControlledSimulationClock, QuerySchedule
from experiments.online_correction_v4.contracts import (
    EpisodeManifestRow,
    PolicyTimingAchieved,
    TimingConfig,
    failure_stage_for_label,
)
from experiments.online_correction_v4.detectors import DetachmentDetector, DetachmentDetectorConfig, ObjectKinematicState
from experiments.online_correction_v4.leases import (
    DeadOwnerVerificationReceipt,
    GroupLeaseStore,
    LeaseConflict,
    StaleLeaseTakeoverDenied,
)
from experiments.online_correction_v4.preparation import (
    GateOutcome,
    GoalAreaGateCase,
    PreparationGateFramework,
    SCRIPTED_TOTAL_PER_GEOMETRY,
)
from experiments.online_correction_v4.recorder import EpisodeEvidenceRecorder
from experiments.online_correction_v4.runner import EpisodeEndReason, EpisodeRunConfig, EpisodeRunner
from experiments.online_correction_v4.leases import AttemptFinalizer
from experiments.online_correction_v4.scheduler import CampaignScheduler, GateStatus
from experiments.online_correction_v4.registry import CampaignRegistry, load_manifest_from_config
from experiments.online_correction_v4.testing import (
    FakeViewportVideoWriter,
    ScriptedPolicy,
    ScriptedSimulator,
    ScriptedTerminalScorer,
)


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


def _runner(
    *,
    simulator: ScriptedSimulator,
    policy: ScriptedPolicy | None = None,
    scenario: str = "move_stop",
    terminal_scorer: ScriptedTerminalScorer | None = None,
    viewport_writer: FakeViewportVideoWriter | None = None,
    viewport_video_required: bool = True,
) -> tuple[EpisodeRunner, AttemptFinalizer, tempfile.TemporaryDirectory[str]]:
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
        policy=policy or ScriptedPolicy(),
        recorder=recorder,
        run_config=EpisodeRunConfig(
            displacement_m=0.12,
            motion_direction=(1.0, 0.0),
            prompt_text="Place the cube.",
            scenario=scenario,
            motion_config=CONFIG["motion"],
            viewport_video_required=viewport_video_required,
            trajectory_flush_interval=5,
        ),
        terminal_scorer=terminal_scorer,
        viewport_writer=viewport_writer or FakeViewportVideoWriter(),
    )
    return runner, finalizer, tmp


class FailureTaxonomyTests(unittest.TestCase):
    def test_timeout_after_no_grasp_stays_no_grasp(self):
        flags = TerminalEvidenceFlags(
            grasp_occurred=False,
            timeout_without_completion=True,
            timeout_after_no_grasp=True,
        )
        label = derive_failure_label(flags)
        self.assertEqual(label, FailureLabel.NO_GRASP)
        _, classified, stage, meta = classify_terminal_outcome(flags)
        self.assertEqual(classified, FailureLabel.NO_GRASP)
        self.assertEqual(stage, failure_stage_for_label(FailureLabel.NO_GRASP))
        self.assertTrue(meta["timeout_after_no_grasp"])

    def test_queue_coverage_uses_largest_period_plus_delay(self):
        timing = TimingConfig.from_mapping(CONFIG["timing"])
        achieved = PolicyTimingAchieved.from_requested(0.05, timing, 16)
        required = achieved.required_queue_coverage_s(timing)
        self.assertGreater(required, achieved.achieved_standard_query_period_s)
        self.assertTrue(achieved.covers_registered_interval(timing))


class DetachmentCarryGateTests(unittest.TestCase):
    def test_detachment_ignored_before_carry(self):
        detector = DetachmentDetector(config=DetachmentDetectorConfig(dwell_ticks=2))
        state = ObjectKinematicState(
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
        self.assertIsNone(detector.update(state))
        detector.arm_after_verified_carry()
        self.assertIsNone(detector.update(state))
        state2 = ObjectKinematicState(**{**state.__dict__, "sim_time": 0.05, "control_tick": 1})
        self.assertIsNotNone(detector.update(state2))


class LeaseTakeoverTests(unittest.TestCase):
    def test_ttl_alone_cannot_take_over(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GroupLeaseStore(Path(tmp), liveness_probe=lambda _: False)
            store.acquire(
                group_id="g1",
                owner_lane="lane-a",
                attempt_id="a1",
                manifest_sha256="m",
            )
            with self.assertRaises(StaleLeaseTakeoverDenied):
                store.acquire(
                    group_id="g1",
                    owner_lane="lane-b",
                    attempt_id="a2",
                    manifest_sha256="m",
                )

    def test_dead_owner_receipt_allows_takeover(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GroupLeaseStore(Path(tmp))
            store.acquire(
                group_id="g1",
                owner_lane="lane-a",
                attempt_id="a1",
                manifest_sha256="m",
            )
            receipt = DeadOwnerVerificationReceipt(
                group_id="g1",
                prior_owner_lane="lane-a",
                prior_attempt_id="a1",
                verified_by="audit-agent",
                verification_method="process_exit",
                verified_at_unix=1.0,
                process_exit_observed=True,
                heartbeat_absent=True,
                evidence_sha256="deadbeef" * 8,
            )
            lease = store.acquire(
                group_id="g1",
                owner_lane="lane-b",
                attempt_id="a2",
                manifest_sha256="m",
                dead_owner_receipt=receipt,
            )
            self.assertEqual(lease.owner_lane, "lane-b")

    def test_concurrent_acquire_race_only_one_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = GroupLeaseStore(Path(tmp))
            store.acquire(
                group_id="race-group",
                owner_lane="lane-a",
                attempt_id="a1",
                manifest_sha256="m",
            )
            with self.assertRaises(LeaseConflict):
                store.acquire(
                    group_id="race-group",
                    owner_lane="lane-b",
                    attempt_id="a2",
                    manifest_sha256="m",
                )

    def test_scheduler_lease_integration_and_resume(self):
        registry = CampaignRegistry.from_rows(
            load_manifest_from_config(ROOT / "docs/online_correction_v4/campaign.json")[:1],
            config_sha256="x",
        )
        with tempfile.TemporaryDirectory() as tmp:
            store = GroupLeaseStore(Path(tmp))
            scheduler = CampaignScheduler(registry=registry, lease_store=store)
            scheduler.set_release_state(
                released={"C1"},
                blocked={fid: "b" for fid in sorted(registry.families_present() - {"C1"})},
            )
            group = next(scheduler.registry.iter_groups())
            lease = scheduler.acquire_group_lease(
                group_id=group.group_id,
                owner_lane="lane-1",
                attempt_id="attempt-1",
                manifest_sha256="manifest",
            )
            self.assertTrue(store.verify(lease))
            scheduler.mark_group_interrupted(group.group_id)
            self.assertNotIn(group.group_id, scheduler.active_leases)
            self.assertIn(group.group_id, scheduler.interrupted_groups)


class PreparationGateTests(unittest.TestCase):
    def test_scale_ladder_and_area_gate_hooks(self):
        class StubGeometry:
            def evaluate_scale(self, fixture_id: str, scale: float):
                from experiments.online_correction_v4.preparation import ScaleLadderCandidate

                return ScaleLadderCandidate(scale=scale, fixture_id=fixture_id, jointly_feasible=scale <= 1.0)

            def goal_area_cases(self, fixture_id: str, scale: float):
                yield GoalAreaGateCase(
                    fixture_id=fixture_id,
                    reset_id="r0",
                    relation="left",
                    motion_sign=1,
                    shrinking_direction=True,
                    original_area_m2=1.0,
                    destination_area_m2=0.7,
                    overlap_fraction=0.7,
                    passes_information_gate=True,
                )

        framework = PreparationGateFramework(
            scale_candidates=(2.0, 1.0, 0.5),
            geometry=StubGeometry(),
        )
        outcome, scale, _ = framework.select_fixture_scale("horizontal")
        self.assertEqual(outcome, GateOutcome.PASSED)
        self.assertEqual(scale, 1.0)
        summary = framework.account_scripted_checks(
            geometry_candidate_id="geom-1",
            run_check=lambda spec: __import__(
                "experiments.online_correction_v4.preparation", fromlist=["ScriptedCheckReceipt"]
            ).ScriptedCheckReceipt(
                check_kind="stationary",
                fixture_id="horizontal",
                goal="left",
                reference_case="bowl",
                reset_case="canonical",
                reference_position="original",
                passed=True,
            ),
            check_specs=[{"id": i} for i in range(SCRIPTED_TOTAL_PER_GEOMETRY)],
        )
        self.assertEqual(summary["observed_total"], SCRIPTED_TOTAL_PER_GEOMETRY)
        self.assertEqual(summary["outcome"], GateOutcome.PASSED.value)


class EpisodeRunnerScenarioTests(unittest.TestCase):
    def _carry_script(self, sim_time: float, tick: int) -> tuple[bool, bool, float]:
        z = 0.095 if tick >= 2 else 0.0
        contact = tick >= 2 and tick < 30
        detached = tick >= 30
        if detached:
            z = 0.0
        return contact, detached, z

    def test_motion_advances_on_normal_ticks(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        runner, finalizer, tmp = _runner(simulator=sim)
        with tmp:
            result = runner.run()
            displacements = [row["reference_displacement_m"] for row in runner.recorder.trajectory]
            self.assertTrue(any(d > 0.0 for d in displacements))
            self.assertTrue(runner.flags.trigger_eligible)

    def test_early_release_and_settling(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        runner, _, tmp = _runner(simulator=sim)
        with tmp:
            result = runner.run()
            self.assertEqual(result.end_reason, EpisodeEndReason.RELEASE_CONFIRMED)
            self.assertTrue(result.flags.motion_truncated_by_release)
            self.assertIsNotNone(result.timing["t_release_detected"])

    def test_no_trigger_timeout(self):
        sim = ScriptedSimulator()
        timing = TimingConfig.from_mapping(CONFIG["timing"])
        timing = TimingConfig(
            emulated_observation_action_delay_s=timing.emulated_observation_action_delay_s,
            standard_query_period_s=timing.standard_query_period_s,
            fast_query_period_s=timing.fast_query_period_s,
            episode_cap_s=2.0,
            trigger_deadline_s=timing.trigger_deadline_s,
            post_event_cap_s=timing.post_event_cap_s,
            release_detection_dwell_ticks=timing.release_detection_dwell_ticks,
            release_settling_s=timing.release_settling_s,
            natural_grasp_min_lift_m=timing.natural_grasp_min_lift_m,
            natural_grasp_dwell_s=timing.natural_grasp_dwell_s,
            kinematic_grasp_relative_drift_max_m=timing.kinematic_grasp_relative_drift_max_m,
            event_phase_fractions=timing.event_phase_fractions,
        )
        runner, _, tmp = _runner(simulator=sim, scenario="original_sham")
        runner.timing = timing
        runner.clock.timing = timing
        with tmp:
            result = runner.run()
            self.assertFalse(result.flags.trigger_eligible)
            self.assertEqual(result.failure_label, FailureLabel.NO_GRASP.value)
            self.assertTrue(result.metadata.get("timeout_after_no_grasp"))

    def test_infra_failure_is_not_behavioral(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        policy = ScriptedPolicy(fail_on_request=1)
        runner, _, tmp = _runner(simulator=sim, policy=policy)
        with tmp:
            result = runner.run()
            self.assertEqual(result.end_reason, EpisodeEndReason.INFRA_INVALID)
            self.assertEqual(result.attempt_status, "infra_invalid")

    def test_recorder_write_once_finalization(self):
        sim = ScriptedSimulator()
        runner, finalizer, tmp = _runner(simulator=sim)
        with tmp:
            runner.run()
            path = runner.recorder.attempt_path
            self.assertTrue(finalizer.is_finalized(path))
            manifest = json.loads((path / "evidence_manifest.json").read_text())
            self.assertIn("episode_sha256", manifest)
            self.assertGreater(manifest["blob_count"], 0)

    def test_observations_use_simulator_capture_not_fabricated(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        policy = ScriptedPolicy()
        runner, _, tmp = _runner(simulator=sim, policy=policy)
        with tmp:
            runner.run()
            self.assertGreater(len(runner.recorder.observations), 0)
            for obs in runner.recorder.observations:
                self.assertIn("payload_uri", obs)
                self.assertIn("camera_ids", obs)
                self.assertTrue(obs["camera_ids"])
                self.assertEqual(len(obs["state_hash"]), 64)
                self.assertTrue(obs["native_input_present"])
                blob = (runner.recorder.attempt_path / obs["payload_uri"]).read_bytes()
                self.assertIn(b"reference_displacement_m", blob)
                self.assertNotIn(b"observation_id:", blob)
            self.assertIsNotNone(policy.last_native_input)
            self.assertIn("tick", policy.last_native_input)

    def test_executed_action_count_exposed_to_policy_and_requests(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        policy = ScriptedPolicy()
        runner, _, tmp = _runner(simulator=sim, policy=policy)
        with tmp:
            runner.run()
            self.assertIsNotNone(policy.last_executed_action_count)
            response_rows = [row for row in runner.recorder.requests if "action_sha256" in row]
            self.assertGreater(len(response_rows), 0)
            last = response_rows[-1]
            self.assertEqual(last["executed_action_count"], policy.last_executed_action_count)

    def test_viewport_frames_and_video_artifact_recorded(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        writer = FakeViewportVideoWriter(fps=20.0)
        runner, _, tmp = _runner(simulator=sim, viewport_writer=writer)
        with tmp:
            result = runner.run()
            self.assertGreater(len(runner.recorder.viewport_frames), 0)
            self.assertGreater(result.timing["viewport_frame_count"], 0)
            for row in runner.recorder.viewport_frames:
                self.assertEqual(row["evidence_mode"], "video_index")
                self.assertNotIn("payload_uri", row)
                self.assertEqual(len(row["payload_sha256"]), 64)
            video = runner.recorder.episode_record.get("viewport_video")
            self.assertIsNotNone(video)
            self.assertEqual(video["codec"], "fake-v4-test-encoder-v1")
            self.assertTrue((runner.recorder.attempt_path / video["video_uri"]).exists())

    def test_close_hooks_run_without_deleting_evidence(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        policy = ScriptedPolicy()
        writer = FakeViewportVideoWriter()
        runner, finalizer, tmp = _runner(simulator=sim, policy=policy, viewport_writer=writer)
        with tmp:
            runner.run()
            path = runner.recorder.attempt_path
            self.assertTrue(sim.closed)
            self.assertTrue(policy.closed)
            self.assertTrue(writer._closed)
            self.assertTrue(finalizer.is_finalized(path))
            self.assertTrue((path / "trajectory.json").exists())

    def test_partial_flush_preserved_on_exception(self):
        sim = ScriptedSimulator(carry_script=self._carry_script, crash_at_tick=5)
        runner, finalizer, tmp = _runner(simulator=sim)
        with tmp:
            with self.assertRaises(RuntimeError):
                runner.run()
            path = runner.recorder.attempt_path
            self.assertFalse(finalizer.is_finalized(path))
            self.assertTrue((path / "trajectory.json").exists())
            episode = json.loads((path / "episode.json").read_text())
            self.assertIn("partial_flush_reasons", episode)

    def test_missing_viewport_video_fail_closed_when_required(self):
        sim = ScriptedSimulator(viewport_unavailable=True)
        runner, finalizer, tmp = _runner(simulator=sim, viewport_writer=None, viewport_video_required=True)
        with tmp:
            result = runner.run()
            self.assertEqual(result.end_reason, EpisodeEndReason.INFRA_INVALID)
            self.assertEqual(result.attempt_status, "infra_invalid")

    def test_clock_schedule_must_match_manifest(self):
        sim = ScriptedSimulator()
        timing = TimingConfig.from_mapping(CONFIG["timing"])
        achieved = PolicyTimingAchieved.from_requested(0.05, timing, 16)
        clock = ControlledSimulationClock(
            timing=timing,
            achieved=achieved,
            schedule=QuerySchedule.FAST_AFTER_GRASP,
            action_queue=ActionQueue(native_control_dt_s=0.05),
        )
        tmp = tempfile.TemporaryDirectory()
        finalizer = AttemptFinalizer(Path(tmp.name))
        recorder = EpisodeEvidenceRecorder.open(
            finalizer=finalizer,
            episode_id="test-episode",
            attempt_id="a001",
            metadata={"episode_id": "test-episode", "attempt_id": "a001"},
        )
        with tmp:
            with self.assertRaises(RuntimeError):
                EpisodeRunner(
                    manifest_row=_manifest_row(),
                    timing=timing,
                    clock=clock,
                    simulator=sim,
                    policy=ScriptedPolicy(),
                    recorder=recorder,
                    run_config=EpisodeRunConfig(
                        displacement_m=0.12,
                        motion_direction=(1.0, 0.0),
                        prompt_text="Place the cube.",
                        scenario="move_stop",
                        motion_config=CONFIG["motion"],
                        schedule=QuerySchedule.FAST_AFTER_GRASP,
                    ),
                    viewport_writer=FakeViewportVideoWriter(),
                )

    def test_event_delivered_only_after_motion_onset(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        runner, _, tmp = _runner(simulator=sim)
        with tmp:
            runner.run()
            self.assertTrue(runner.flags.trigger_eligible)
            self.assertTrue(runner.flags.event_delivered)
            delivered = [e for e in runner.recorder.events if e.get("kind") == "event_delivered"]
            self.assertEqual(len(delivered), 1)
            planned = next(e for e in runner.recorder.events if e.get("kind") == "event_planned")
            self.assertGreaterEqual(delivered[0]["sim_time"], planned["sim_time"])

    def test_destination_static_applies_initial_reference_before_observations(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        runner, _, tmp = _runner(simulator=sim, scenario="destination_static")
        with tmp:
            runner.run()
            first_row = runner.recorder.trajectory[0]
            self.assertAlmostEqual(first_row["reference_displacement_m"], 0.12, places=6)
            self.assertFalse(runner.flags.event_delivered)

    def test_terminal_scorer_can_mark_success(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        scorer = ScriptedTerminalScorer(success=True, geometric_relation_correct=True)
        runner, _, tmp = _runner(simulator=sim, terminal_scorer=scorer)
        with tmp:
            result = runner.run()
            self.assertEqual(result.end_reason, EpisodeEndReason.RELEASE_CONFIRMED)
            self.assertTrue(result.terminal.success)
            self.assertTrue(result.terminal.geometric_relation_correct)

    def test_timeout_passive_adjudication_distinct_from_release(self):
        sim = ScriptedSimulator()
        timing = TimingConfig.from_mapping(CONFIG["timing"])
        timing = TimingConfig(
            emulated_observation_action_delay_s=timing.emulated_observation_action_delay_s,
            standard_query_period_s=timing.standard_query_period_s,
            fast_query_period_s=timing.fast_query_period_s,
            episode_cap_s=1.5,
            trigger_deadline_s=timing.trigger_deadline_s,
            post_event_cap_s=timing.post_event_cap_s,
            release_detection_dwell_ticks=timing.release_detection_dwell_ticks,
            release_settling_s=timing.release_settling_s,
            natural_grasp_min_lift_m=timing.natural_grasp_min_lift_m,
            natural_grasp_dwell_s=timing.natural_grasp_dwell_s,
            kinematic_grasp_relative_drift_max_m=timing.kinematic_grasp_relative_drift_max_m,
            event_phase_fractions=timing.event_phase_fractions,
        )
        runner, _, tmp = _runner(simulator=sim, scenario="original_sham")
        runner.timing = timing
        runner.clock.timing = timing
        with tmp:
            result = runner.run()
            self.assertEqual(result.end_reason, EpisodeEndReason.TIMEOUT)
            self.assertEqual(result.timing["passive_settling_reason"], "timeout")
            self.assertFalse(result.terminal.released)

    def test_model_invalid_is_distinct_from_infra_failure(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        policy = ScriptedPolicy(invalid_on_request=1)
        runner, _, tmp = _runner(simulator=sim, policy=policy)
        with tmp:
            result = runner.run()
            self.assertEqual(result.end_reason, EpisodeEndReason.MODEL_OUTPUT_INVALID)
            self.assertEqual(result.attempt_status, "valid")
            self.assertEqual(result.failure_label, FailureLabel.MODEL_OUTPUT_INVALID.value)
            infra = [e for e in runner.recorder.events if e.get("kind") == "infra_invalid"]
            self.assertEqual(infra, [])

    def test_future_artifact_recorded_when_exposed(self):
        sim = ScriptedSimulator(carry_script=self._carry_script)
        policy = ScriptedPolicy(future_on_request=b"future-bytes")
        runner, _, tmp = _runner(simulator=sim, policy=policy)
        with tmp:
            runner.run()
            self.assertGreater(len(runner.recorder.future_artifacts), 0)
            artifact = runner.recorder.future_artifacts[0]
            self.assertEqual(artifact["kind"], "nano_decoded_future")
            req = next(row for row in runner.recorder.requests if "action_sha256" in row)
            self.assertEqual(len(req["action_sha256"]), 64)
            self.assertIn("timing", runner.recorder.episode_record)
            self.assertIn("viewport_video", runner.recorder.episode_record)


if __name__ == "__main__":
    unittest.main()
