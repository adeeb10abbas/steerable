#!/usr/bin/env python3
"""Build the C8 natural-trigger and branch-mode qualification receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.detectors import (
    DetachmentDetector,
    DetachmentDetectorConfig,
    GraspDetectorConfig,
    NaturalGraspDetector,
    ObjectKinematicState,
)


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


def kinematic_state(
    *,
    tick: int,
    dt: float,
    lift_m: float,
    relative_drift_m: float = 0.0,
    contact: bool = True,
    detached: bool = False,
) -> ObjectKinematicState:
    return ObjectKinematicState(
        sim_time=tick * dt,
        control_tick=tick,
        object_z=0.05 + lift_m,
        initial_supported_z=0.05,
        gripper_x=0.0,
        gripper_y=0.0,
        gripper_z=0.05 + lift_m,
        object_x=relative_drift_m,
        object_y=0.0,
        object_z_pos=0.05 + lift_m,
        contact=contact,
        detached=detached,
    )


def build_receipt(*, g3_path: Path, g4_path: Path) -> dict[str, Any]:
    g3 = load_json(g3_path)
    g4 = load_json(g4_path)
    if (
        g3.get("schema_version")
        != "v4-second-stack-g3-scripted-aggregate-v1"
        or g3.get("fixture_id") != "second_stack"
        or g3.get("passed") is not True
        or g3.get("observed_check_count") != 112
        or g3.get("passed_check_count") != 112
    ):
        raise ValueError("C8 G3 scripted receipt is not a complete pass")
    if (
        g4.get("schema_version")
        != "v4-second-stack-g4-policy-session-receipt-v1"
        or g4.get("fixture_id") != "second_stack"
        or g4.get("passed") is not True
        or g4.get("policy_id") != "groot_n1_7_simplerenv_bridge"
    ):
        raise ValueError("C8 G4 policy-session receipt is not a pass")

    dt = 0.2
    config = GraspDetectorConfig()
    detector = NaturalGraspDetector(config=config, control_dt_s=dt)
    no_lift_event = detector.update(
        kinematic_state(tick=0, dt=dt, lift_m=0.0)
    )
    detector.reset()
    drift_event = None
    for tick in range(3):
        drift_event = detector.update(
            kinematic_state(
                tick=tick,
                dt=dt,
                lift_m=0.05,
                relative_drift_m=0.02,
            )
        )
    detector.reset()
    stable_event = None
    for tick in range(2):
        stable_event = detector.update(
            kinematic_state(tick=tick, dt=dt, lift_m=0.05)
        )

    detachment = DetachmentDetector(DetachmentDetectorConfig(dwell_ticks=2))
    before_arm = detachment.update(
        kinematic_state(tick=8, dt=dt, lift_m=0.05, detached=True)
    )
    detachment.arm_after_verified_carry()
    first_detached = detachment.update(
        kinematic_state(tick=10, dt=dt, lift_m=0.05, detached=True)
    )
    release_event = detachment.update(
        kinematic_state(tick=11, dt=dt, lift_m=0.05, detached=True)
    )
    checks = {
        "no_trigger_without_registered_lift": no_lift_event is None,
        "excess_relative_drift_resets_dwell": drift_event is None,
        "stable_carry_triggers_after_registered_dwell": (
            stable_event is not None
            and math.isclose(stable_event.dwell_s, 0.2, abs_tol=1e-12)
            and stable_event.lift_m >= config.min_lift_m
        ),
        "detachment_ignored_before_verified_carry": before_arm is None,
        "first_detachment_requires_two_ticks": (
            first_detached is None
            and release_event is not None
            and release_event.onset_tick == 10
            and release_event.detected_tick == 11
        ),
        "fresh_policy_session_repeat_qualified": (
            g4["checks"]["fresh_reset_exact_repeat_actions_equal"] is True
        ),
        "live_scripted_grasp_transport_release_basis": (
            g3["passed_check_count"] == 112
        ),
    }
    passed = all(checks.values())
    return {
        "schema_version": "v4-second-stack-g5-trigger-branch-receipt-v1",
        "campaign_id": "online_correction_v4",
        "family_id": "C8",
        "fixture_id": "second_stack",
        "policy_id": "groot_n1_7_simplerenv_bridge",
        "gate": "G5",
        "status": "passed" if passed else "blocked",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "checks": checks,
        "detector_contract": {
            "native_control_dt_s": dt,
            "min_lift_m": config.min_lift_m,
            "dwell_s": config.dwell_s,
            "relative_drift_max_m": config.relative_drift_max_m,
            "trigger_deadline_s": config.trigger_deadline_s,
            "release_detection_dwell_ticks": 2,
        },
        "selected_prefix_mode": "independent_natural_rollout_fallback",
        "evaluation_design_label": "randomized_event_triggered_evaluation",
        "exact_counterfactual_branching_claimed": False,
        "qualification_basis": {
            "g3_scripted_physics": artifact(g3_path),
            "g4_policy_session": artifact(g4_path),
        },
        "release_boundary": (
            "A pass qualifies the prospectively permitted independent-natural-"
            "rollout mode for C8. It does not authorize exact-prefix claims, C2, "
            "or C8 behavioral inference before G6-G8."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g3", type=Path, required=True)
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_receipt(
        g3_path=args.g3.resolve(),
        g4_path=args.g4.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(
            json.dumps(
                payload,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
    print(json.dumps({"status": payload["status"], "path": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
