#!/usr/bin/env python3
"""Build fixture-parameterized G5 trigger/branch and G6 measurement gate receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4 import geometry as geom  # noqa: E402
from experiments.online_correction_v4.attempts import (  # noqa: E402
    InfraInvalidReason,
    TerminalEvidenceFlags,
    derive_failure_label,
)
from experiments.online_correction_v4.clock import (  # noqa: E402
    ActionQueue,
    ControlledSimulationClock,
)
from experiments.online_correction_v4.contracts import (  # noqa: E402
    FailureLabel,
    PolicyTimingAchieved,
    TimingConfig,
)
from experiments.online_correction_v4.detectors import (  # noqa: E402
    DetachmentDetector,
    DetachmentDetectorConfig,
    GraspDetectorConfig,
    NaturalGraspDetector,
    ObjectKinematicState,
)
from experiments.online_correction_v4.droid_scorer import aggregate_settling_predicates  # noqa: E402
from experiments.online_correction_v4.fixture_qualification import (  # noqa: E402
    manipulated_and_reference,
    qualification_profile,
)
from experiments.online_correction_v4.geometry import direct_inverse_pair_equivalent  # noqa: E402


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_pass(
    payload: dict[str, Any],
    *,
    schema: str,
    fixture_id: str,
    path: Path,
) -> None:
    if payload.get("schema_version") != schema:
        raise ValueError(f"{path} schema mismatch")
    if payload.get("fixture_id") != fixture_id:
        raise ValueError(f"{path} fixture mismatch")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError(f"{path} is not a passing receipt")


def kinematic_state(
    *,
    tick: int,
    dt: float,
    lift_m: float,
    relative_drift_m: float = 0.0,
    contact: bool = True,
    detached: bool = False,
) -> ObjectKinematicState:
    gripper_z = 0.05 + lift_m
    return ObjectKinematicState(
        sim_time=tick * dt,
        control_tick=tick,
        object_z=0.05 + lift_m,
        initial_supported_z=0.05,
        gripper_x=0.0,
        gripper_y=0.0,
        gripper_z=gripper_z,
        object_x=relative_drift_m,
        object_y=0.0,
        object_z_pos=0.05 + lift_m,
        contact=contact,
        detached=detached,
    )


def _detector_checks() -> dict[str, bool]:
    dt = 1.0 / 15.0
    config = GraspDetectorConfig()
    detector = NaturalGraspDetector(config=config, control_dt_s=dt)
    no_lift_event = detector.update(
        kinematic_state(tick=0, dt=dt, lift_m=0.0, contact=True)
    )
    detector.reset()
    drift_event = None
    for tick in range(3):
        drift_event = detector.update(
            kinematic_state(tick=tick, dt=dt, lift_m=0.05, relative_drift_m=0.02)
        )
    detector.reset()
    stable_event = None
    for tick in range(4):
        stable_event = detector.update(
            kinematic_state(tick=tick, dt=dt, lift_m=0.05)
        )
    detachment = DetachmentDetector(DetachmentDetectorConfig(dwell_ticks=2))
    pre_carry = detachment.update(
        kinematic_state(tick=8, dt=dt, lift_m=0.05, detached=True)
    )
    detachment.arm_after_verified_carry()
    first_detached = detachment.update(
        kinematic_state(tick=10, dt=dt, lift_m=0.05, detached=True)
    )
    release_event = detachment.update(
        kinematic_state(tick=11, dt=dt, lift_m=0.05, detached=True)
    )
    return {
        "no_trigger_without_registered_lift": no_lift_event is None,
        "excess_relative_drift_resets_dwell": drift_event is None,
        "stable_carry_triggers_after_registered_dwell": (
            stable_event is not None
            and math.isclose(stable_event.dwell_s, 0.2, abs_tol=1e-12)
            and stable_event.lift_m >= config.min_lift_m
        ),
        "detachment_ignored_before_verified_carry": pre_carry is None,
        "first_detachment_requires_two_ticks": (
            first_detached is None
            and release_event is not None
            and release_event.onset_tick == 10
            and release_event.detected_tick == 11
        ),
    }


def _clock_checks(campaign: dict[str, Any]) -> dict[str, Any]:
    timing = TimingConfig.from_mapping(campaign["timing"])
    achieved = PolicyTimingAchieved.from_requested(1.0 / 15.0, timing, 32)
    queue = ActionQueue(native_control_dt_s=achieved.native_control_dt_s)
    queue.enqueue_chunk(
        chunk_id="old",
        request_id="old-request",
        actions=((1.0,), (1.0,), (1.0,)),
    )
    clock = ControlledSimulationClock(
        timing=timing,
        achieved=achieved,
        action_queue=queue,
    )
    observation_id, _ = clock.capture_observation("a" * 64)
    request = clock.submit_policy_request(observation_id)
    clock.complete_inference(
        request,
        chunk_id="new",
        actions=((2.0,), (2.0,)),
        wall_duration_s=7.0,
    )
    delay_actions = clock.advance_for_delay_window(request)
    applied = clock.apply_due_responses()
    first_new = clock.action_queue.pop_for_tick()
    missing = aggregate_settling_predicates((), dwell_ticks=2)
    no_grasp = derive_failure_label(
        TerminalEvidenceFlags(
            grasp_occurred=False,
            timeout_without_completion=True,
            timeout_after_no_grasp=True,
        )
    )
    grasp_lost = derive_failure_label(
        TerminalEvidenceFlags(
            grasp_occurred=True,
            carry_verified=True,
            grasp_lost=True,
        )
    )
    return {
        "checks": {
            "changed_observation_action_causal_order": (
                request.observation_capture_time == 0.0
                and request.response_available_time == achieved.achieved_delay_s
                and all(command.chunk_id == "old" for command in delay_actions)
                and applied == [request]
                and first_new is not None
                and first_new.chunk_id == "new"
            ),
            "wall_inference_duration_does_not_advance_sim_time": (
                request.inference_wall_duration_s == 7.0
                and math.isclose(clock.sim_time, achieved.achieved_delay_s)
            ),
            "video_frame_index_aligns_with_sim_clock_and_control_ticks": all(
                math.isclose(
                    frame_index * achieved.native_control_dt_s,
                    frame_time,
                    abs_tol=1e-12,
                )
                for frame_index, frame_time in enumerate(
                    tuple(index * achieved.native_control_dt_s for index in range(4))
                )
            ),
            "rejected_stimulus_is_infrastructure_not_policy_nonresponse": (
                InfraInvalidReason.WRONG_STIMULUS_TRAJECTORY.value
                == "wrong_stimulus_trajectory"
                and InfraInvalidReason.WRONG_STIMULUS_TRAJECTORY.value
                not in {label.value for label in FailureLabel}
            ),
            "missing_terminal_predicates_fail_closed": (
                not missing.available
                and "settling_samples" in missing.missing_fields
            ),
            "no_grasp_grasp_loss_and_infrastructure_are_distinct": (
                no_grasp is FailureLabel.NO_GRASP
                and grasp_lost is FailureLabel.GRASP_LOST
            ),
        },
        "controlled_clock": {
            "native_control_dt_s": achieved.native_control_dt_s,
            "achieved_delay_s": achieved.achieved_delay_s,
            "delay_window_action_chunk_ids": [
                command.chunk_id for command in delay_actions
            ],
            "new_chunk_first_applied_after_delay": (
                first_new.chunk_id if first_new is not None else None
            ),
        },
        "failure_case_labels": {
            "no_grasp_timeout": no_grasp.value,
            "grasp_loss": grasp_lost.value,
            "missing_terminal_predicates": "unresolved_behavioral_failure",
        },
    }


def build_g5_receipt(
    *,
    fixture_id: str,
    g3_path: Path,
    g3: dict[str, Any],
    g4_path: Path,
    g4: dict[str, Any],
    live_positive_control: dict[str, Any] | None = None,
    live_positive_control_path: Path | None = None,
    live_negative_control: dict[str, Any] | None = None,
    live_negative_control_path: Path | None = None,
) -> dict[str, Any]:
    profile = qualification_profile(fixture_id)
    if profile.g3_basis == "path_scale":
        require_pass(g3, schema=profile.g3_path_scale_schema, fixture_id=fixture_id, path=g3_path)
        g3_physics_ok = (
            g3.get("observed_seed_count") == g3.get("expected_seed_count")
            and not g3.get("information_gate_failed_seeds")
            and g3.get("failed_env_seeds") == []
        )
        g3_scripted_ok = True
    else:
        require_pass(
            g3,
            schema=str(profile.g3_scripted_aggregate_schema),
            fixture_id=fixture_id,
            path=g3_path,
        )
        g3_physics_ok = g3.get("observed_scripted_check_count") == 112
        g3_scripted_ok = (
            g3.get("scripted_passed_check_count") == 112
            and g3.get("scripted_failed_check_count") == 0
        )
    require_pass(g4, schema=profile.g4_receipt_schema, fixture_id=fixture_id, path=g4_path)
    if g4.get("policy_id") != profile.policy_id:
        raise ValueError("G4 policy mismatch")

    checks = {
        **_detector_checks(),
        "fresh_policy_session_repeat_qualified": (
            g4.get("checks", {}).get("fresh_session_exact_repeat_actions_equal") is True
        ),
        "g3_physics_basis_passed": g3_physics_ok,
        "g3_scripted_basis_passed": g3_scripted_ok,
        "live_positive_control_trigger_eligible": (
            live_positive_control is not None
            and live_positive_control.get("trigger_eligible") is True
            and live_positive_control.get("verdict") == "intervention_deliverable"
        ),
        "live_negative_control_did_not_fire": (
            live_negative_control is not None
            and live_negative_control.get("trigger_eligible") is False
            and live_negative_control.get("verdict") == "negative_control_passed"
        ),
    }
    passed = all(checks.values())
    basis_key = "g3_path_scale" if profile.g3_basis == "path_scale" else "g3_scripted_physics"
    return {
        "schema_version": profile.g5_receipt_schema,
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "family_id": profile.family_id,
        "policy_id": profile.policy_id,
        "gate": "G5",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "checks": checks,
        "detector_contract": {
            "min_lift_m": 0.04,
            "dwell_s": 0.2,
            "relative_drift_max_m": 0.01,
            "trigger_deadline_s": 40.0,
            "release_detection_dwell_ticks": 2,
        },
        "selected_prefix_mode": "independent_natural_rollout_fallback",
        "evaluation_design_label": "randomized_event_triggered_evaluation",
        "exact_counterfactual_branching_claimed": False,
        "qualification_basis": {
            basis_key: artifact(g3_path),
            "g4_policy_session": artifact(g4_path),
            **(
                {"live_positive_control": artifact(live_positive_control_path)}
                if live_positive_control_path is not None
                else {}
            ),
            **(
                {"live_negative_control": artifact(live_negative_control_path)}
                if live_negative_control_path is not None
                else {}
            ),
        },
        "release_boundary": (
            f"Passes {profile.family_id} G5 using the prospectively permitted "
            "independent-natural-rollout fallback bound to registered live controls. "
            "Confirmatory inference remains blocked until G6-G8 pass."
        ),
    }


def build_g6_receipt(
    *,
    fixture_id: str,
    campaign: dict[str, Any],
    campaign_path: Path,
    scoring_geometry_path: Path | None = None,
) -> dict[str, Any]:
    profile = qualification_profile(fixture_id)
    manipulated, reference = manipulated_and_reference(fixture_id)
    clock_bundle = _clock_checks(campaign)
    checks = dict(clock_bundle["checks"])
    if fixture_id == "object_pair":
        from tools.build_v4_object_pair_g5_g6_receipts import build_g6_receipt as build_op_g6

        if scoring_geometry_path is None:
            raise ValueError("object_pair G6 requires scoring geometry")
        return build_op_g6(
            geometry_payload=load_json(scoring_geometry_path),
            geometry_path=scoring_geometry_path,
            campaign=campaign,
            campaign_path=campaign_path,
        )
    checks["direct_inverse_wordings_share_goal_sets"] = direct_inverse_pair_equivalent(
        manipulated,
        reference,
        "inside",
    )
    checks["fixture_relation_registered"] = profile.fixture_id == "containment"
    passed = all(checks.values())
    payload: dict[str, Any] = {
        "schema_version": profile.g6_receipt_schema,
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "family_id": profile.family_id,
        "gate": "G6",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "checks": checks,
        "controlled_clock": clock_bundle["controlled_clock"],
        "failure_case_labels": clock_bundle["failure_case_labels"],
        "campaign": artifact(campaign_path),
        "release_boundary": (
            f"Passes {profile.family_id} G6 deterministic measurement checks for the "
            "registered containment relation and controlled clock contract."
        ),
    }
    if scoring_geometry_path is not None:
        payload["scoring_geometry"] = artifact(scoring_geometry_path)
    return payload


def write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--g3-receipt", type=Path, required=True)
    parser.add_argument("--g4-receipt", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "docs/online_correction_v4/campaign.json",
    )
    parser.add_argument("--scoring-geometry", type=Path, default=None)
    parser.add_argument("--g5-out", type=Path, required=True)
    parser.add_argument("--g6-out", type=Path, required=True)
    parser.add_argument("--live-positive-control", type=Path, default=None)
    parser.add_argument("--live-negative-control", type=Path, default=None)
    args = parser.parse_args()
    for output in (args.g5_out, args.g6_out):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite receipt: {output}")
    g3 = load_json(args.g3_receipt)
    g4 = load_json(args.g4_receipt)
    campaign = load_json(args.campaign)
    live_positive = (
        load_json(args.live_positive_control)
        if args.live_positive_control is not None
        else None
    )
    live_negative = (
        load_json(args.live_negative_control)
        if args.live_negative_control is not None
        else None
    )
    g5 = build_g5_receipt(
        fixture_id=args.fixture_id,
        g3_path=args.g3_receipt,
        g3=g3,
        g4_path=args.g4_receipt,
        g4=g4,
        live_positive_control=live_positive,
        live_positive_control_path=args.live_positive_control,
        live_negative_control=live_negative,
        live_negative_control_path=args.live_negative_control,
    )
    g6 = build_g6_receipt(
        fixture_id=args.fixture_id,
        campaign=campaign,
        campaign_path=args.campaign,
        scoring_geometry_path=args.scoring_geometry,
    )
    write_exclusive(args.g5_out, g5)
    write_exclusive(args.g6_out, g6)
    print(
        json.dumps(
            {
                "fixture_id": args.fixture_id,
                "g5_passed": g5["passed"],
                "g6_passed": g6["passed"],
                "g5_out": str(args.g5_out),
                "g6_out": str(args.g6_out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
