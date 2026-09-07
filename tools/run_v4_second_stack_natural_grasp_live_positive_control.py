#!/usr/bin/env python3
"""Live SimplerEnv/WidowX natural-grasp positive and negative controls for C8."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT_SCHEMA = "v4-second-stack-natural-grasp-live-positive-control-v1"
FIXTURE_ID = "second_stack"
CANONICAL_ENV_SEED = 2100050000
DEFAULT_SCALE = 0.5
DEFAULT_GOAL = "left"
NATIVE_CONTROL_DT_S = 0.2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


class InstrumentedSecondStackEnv:
    def __init__(self, env: Any, adapter: Any, detector: Any) -> None:
        self._env = env
        self._adapter = adapter
        self._detector = detector
        self.detector_samples: list[dict[str, Any]] = []
        self._step_hook_installed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self._env.reset(*args, **kwargs)

    def object_kinematic_state(self) -> Any:
        return self._adapter.object_kinematic_state()

    def _install_step_hook(self) -> None:
        from experiments.online_correction_v4.second_stack import unwrap_simpler_env

        raw = unwrap_simpler_env(self._env)
        if getattr(raw, "_v4_detector_after_step_hook", False):
            return
        self._adapter.bind_sim_dt(float(raw._scene.get_timestep()))
        steps_per_control = max(
            1,
            round(self._adapter.control_dt_s / self._adapter._sim_dt_s),
        )
        original_after = raw._after_simulation_step

        def after_and_record() -> None:
            original_after()
            self._adapter.on_physics_step()
            if self._adapter._physics_steps % steps_per_control != 0:
                return
            self._adapter.on_control_boundary()
            self.record_detector(phase="scripted")

        raw._after_simulation_step = after_and_record  # type: ignore[method-assign]
        raw._v4_detector_after_step_hook = True  # type: ignore[attr-defined]
        self._step_hook_installed = True

    def record_detector(self, *, phase: str = "live") -> None:
        state = self._adapter.object_kinematic_state()
        event = self._detector.update(state)
        self.detector_samples.append(
            {
                "phase": phase,
                "control_tick": state.control_tick,
                "sim_time_s": state.sim_time,
                "contact": state.contact,
                "lift_m": state.object_z_pos - state.initial_supported_z,
                "relative_drift_m": (
                    (state.object_x - state.gripper_x) ** 2
                    + (state.object_y - state.gripper_y) ** 2
                    + (state.object_z_pos - state.gripper_z) ** 2
                )
                ** 0.5,
                "trigger_eligible": self._detector.eligible,
                "grasp_occurred": self._detector.grasp_occurred,
                "event": None if event is None else event.__dict__,
            }
        )

    def simulate_hold(self, steps: int) -> None:
        from experiments.online_correction_v4.second_stack import unwrap_simpler_env

        self._install_step_hook()
        raw = unwrap_simpler_env(self._env)
        for _ in range(steps):
            raw.agent.before_simulation_step()
            raw._scene.step()
            raw._after_simulation_step()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--pilot-reset-registry", type=Path, required=True)
    parser.add_argument("--pilot-reset-registry-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--path-receipt", type=Path, required=True)
    parser.add_argument("--path-receipt-sha256", required=True)
    parser.add_argument("--environment-seed", type=int, default=CANONICAL_ENV_SEED)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--control-mode",
        choices=("scripted_grasp", "hold_only"),
        default="scripted_grasp",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_root = args.study_root.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True)

    from experiments.online_correction_v4.contracts import TimingConfig
    from experiments.online_correction_v4.detectors import GraspDetectorConfig, NaturalGraspDetector
    from experiments.online_correction_v4.second_stack import apply_registered_reset, ensure_registered_support
    from experiments.online_correction_v4.second_stack_kinematic import SecondStackKinematicAdapter
    from tools.run_v4_second_stack_g2 import load_json, sha256_file as g2_sha256, verify_external_stack
    from tools.run_v4_second_stack_g3_scripted import _run_check

    for path, expected, label in (
        (args.pilot_reset_registry, args.pilot_reset_registry_sha256, "reset registry"),
        (args.plan, args.plan_sha256, "plan"),
        (args.path_receipt, args.path_receipt_sha256, "path receipt"),
    ):
        if g2_sha256(path) != expected.lower():
            raise RuntimeError(f"{label} sha256 mismatch")

    registry = load_json(args.pilot_reset_registry)
    plan = load_json(args.plan)
    path_receipt = load_json(args.path_receipt)
    if str(args.environment_seed) not in registry["resets_by_env_seed"]:
        raise RuntimeError("environment seed missing from pilot reset registry")
    if path_receipt.get("passed") is not True:
        raise RuntimeError("path receipt is not a pass")
    integration_root = args.integration_root.resolve()
    verify_external_stack(integration_root=integration_root, registry=registry)

    campaign = load_json(study_root / "docs/online_correction_v4/campaign.json")
    timing = TimingConfig.from_mapping(campaign["timing"])
    detector = NaturalGraspDetector(
        config=GraspDetectorConfig(
            min_lift_m=timing.natural_grasp_min_lift_m,
            dwell_s=timing.natural_grasp_dwell_s,
            relative_drift_max_m=timing.kinematic_grasp_relative_drift_max_m,
            trigger_deadline_s=timing.trigger_deadline_s,
        ),
        control_dt_s=NATIVE_CONTROL_DT_S,
    )

    sys.path.insert(0, str(integration_root))
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

    register_simpler_envs()
    import gymnasium as gym

    from experiments.online_correction_v4.second_stack import ENV_NAME

    env = gym.make(ENV_NAME)
    adapter = SecondStackKinematicAdapter(env, control_dt_s=NATIVE_CONTROL_DT_S)
    instrumented = InstrumentedSecondStackEnv(env, adapter, detector)
    reset_row = registry["resets_by_env_seed"][str(args.environment_seed)]
    instrumented._install_step_hook()
    instrumented.reset(seed=args.environment_seed)
    apply_registered_reset(env, reset_row, settle_steps=30)
    ensure_registered_support(env)
    instrumented.record_detector(phase="post_reset")
    path_audit = adapter.observation_path_audit()
    defect_state = adapter.object_kinematic_state_robot_base_defect()
    robot_base_would_fire = (
        defect_state.object_z_pos - defect_state.initial_supported_z >= timing.natural_grasp_min_lift_m
        and (
            (defect_state.object_x - defect_state.gripper_x) ** 2
            + (defect_state.object_y - defect_state.gripper_y) ** 2
            + (defect_state.object_z_pos - defect_state.gripper_z) ** 2
        )
        ** 0.5
        <= timing.kinematic_grasp_relative_drift_max_m
    )

    selected_scale = float(path_receipt["scale"])
    plan_row = next(
        check
        for row in plan["scales"]
        if float(row["scale"]) == selected_scale
        for check in row["checks"]
        if int(check["environment_seed"]) == args.environment_seed
        and str(check["relation"]) == args.goal
    )
    trajectory_result: dict[str, Any]
    if args.control_mode == "hold_only":
        hold_ticks = max(int(round(timing.trigger_deadline_s / NATIVE_CONTROL_DT_S)) + 2, 4)
        instrumented.simulate_hold(hold_ticks)
        trajectory_result = {
            "passed": True,
            "mode": "hold_only",
            "tick_count": hold_ticks,
            "stages": {},
        }
    else:
        check_result = _run_check(
            env=instrumented,
            env_seed=args.environment_seed,
            reset_row=reset_row,
            plan_row=plan_row,
            position_label="original",
            moving_reference=False,
            skip_initial_reset=True,
        )
        for _ in range(max(int(round(timing.natural_grasp_dwell_s / NATIVE_CONTROL_DT_S)) + 2, 4)):
            if detector.eligible:
                break
            instrumented.simulate_hold(1)
        trajectory_result = {
            "passed": bool(check_result.get("passed")),
            "mode": "scripted_grasp",
            "tick_count": len(instrumented.detector_samples),
            "stages": {
                "grasped": bool(check_result.get("grasp_contact_count", 0)),
                "transported": check_result.get("lift_height_m", 0.0) >= timing.natural_grasp_min_lift_m,
                "released": check_result.get("post_release_stability_drift_m") is not None,
                "stably_placed": bool(check_result.get("passed")),
                "goal_satisfied": bool(check_result.get("passed")),
            },
            "check_record": check_result,
        }

    if args.control_mode == "hold_only":
        verdict = "negative_control_passed" if not detector.eligible else "blocking_false_positive"
        passed = not detector.eligible
        status = "passed" if passed else "blocking_false_positive"
    else:
        verdict = "intervention_deliverable" if detector.eligible else "blocking_setup_defect"
        passed = detector.eligible
        status = "passed" if passed else "blocking_setup_defect"

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "platform": "simplerenv",
        "attempt_id": args.attempt_id,
        "control_mode": args.control_mode,
        "environment_seed": args.environment_seed,
        "scale": float(args.scale),
        "goal": args.goal,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "detector_class": "NaturalGraspDetector",
        "detector_config": {
            "min_lift_m": timing.natural_grasp_min_lift_m,
            "dwell_s": timing.natural_grasp_dwell_s,
            "relative_drift_max_m": timing.kinematic_grasp_relative_drift_max_m,
            "trigger_deadline_s": timing.trigger_deadline_s,
        },
        "native_control_dt_s": NATIVE_CONTROL_DT_S,
        "scripted_trajectory_passed": bool(trajectory_result.get("passed")),
        "scripted_grasp_stages": trajectory_result.get("stages", {}),
        "scripted_check_record": trajectory_result.get("check_record"),
        "trigger_eligible": detector.eligible,
        "grasp_occurred": detector.grasp_occurred,
        "trigger_event": None if detector.event is None else detector.event.__dict__,
        "verdict": verdict,
        "passed": passed,
        "status": status,
        "observation_path_audit": path_audit,
        "robot_base_defect_audit": {
            "robot_base_to_target_m": path_audit["robot_base_to_target_m"],
            "finger_to_target_m": path_audit["finger_to_target_m"],
            "would_use_robot_base_for_gripper": True,
            "equivalent_isaac_robot_base_defect_present": path_audit["robot_base_to_target_m"]
            > 0.15,
            "robot_base_reference_would_support_spurious_trigger": robot_base_would_fire,
            "production_gripper_reference": "finger_midpoint",
        },
        "integration_root": str(integration_root),
        "integration_commit": registry["external_stack_identity"]["gr00t_commit"],
        "detector_sample_count": len(instrumented.detector_samples),
        "detector_samples_tail": instrumented.detector_samples[-8:],
    }
    receipt_path = output_dir / "natural_grasp_positive_control.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    env.close()
    print(json.dumps({"passed": passed, "verdict": verdict, "receipt": str(receipt_path)}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
