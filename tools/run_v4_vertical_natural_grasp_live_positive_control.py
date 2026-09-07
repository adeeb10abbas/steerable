#!/usr/bin/env python3
"""Live Isaac positive control for vertical NaturalGraspDetector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RECEIPT_SCHEMA = "v4-vertical-natural-grasp-live-positive-control-v1"
FIXTURE_ID = "vertical"
CANONICAL_ENV_SEED = 2100020000
DEFAULT_SCALE = 0.5
DEFAULT_GOAL = "above"
ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class GraspDetectorInstrumentedEnv:
    """Proxy env.step that feeds NaturalGraspDetector from object_kinematic_state()."""

    def __init__(self, env: Any, detector: Any) -> None:
        self._env = env
        self._detector = detector
        self.detector_samples: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self._env.step(action)
        state = self._env.object_kinematic_state()
        event = self._detector.update(state)
        self.detector_samples.append(
            {
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
        return observation, info


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT)
    parser.add_argument("--robolab-root", type=Path)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--reset-registry", type=Path, required=True)
    parser.add_argument("--reset-registry-sha256", required=True)
    parser.add_argument("--environment-seed", type=int, default=CANONICAL_ENV_SEED)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-study-commit")
    parser.add_argument("--expected-robolab-commit", default=ROBOLAB_COMMIT)
    parser.add_argument("--expected-driver-version")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--pod")
    parser.add_argument("--pod-uid")
    parser.add_argument("--native-control-dt-s", type=float, required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument(
        "--control-mode",
        choices=("scripted_grasp", "hold_only"),
        default="scripted_grasp",
        help="scripted_grasp=live positive control; hold_only=live negative control",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    study_root = args.study_root.resolve()
    tools_root = study_root / "tools"
    for path_entry in (study_root, tools_root):
        if str(path_entry) not in sys.path:
            sys.path.insert(0, str(path_entry))

    output_raw = args.output_dir or (
        Path(os.environ["EPISODE_OUTPUT_DIR"]) if os.environ.get("EPISODE_OUTPUT_DIR") else None
    )
    if output_raw is None:
        raise RuntimeError("--output-dir or EPISODE_OUTPUT_DIR is required")
    output_dir = output_raw.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite positive-control output: {output_dir}")
    output_dir.mkdir(parents=True)

    from experiments.online_correction_v4.contracts import TimingConfig
    from experiments.online_correction_v4.detectors import GraspDetectorConfig, NaturalGraspDetector
    from experiments.online_correction_v4.droid_g3 import (
        fixture_object_spec,
        geometry_from_scene_for_fixture,
        goal_set_for_reference,
        task_frame_from_evidence,
    )
    from experiments.online_correction_v4.droid_g3_scripted import run_scripted_check
    from experiments.online_correction_v4.droid_robolab import (
        ResetFixtureBinding,
        RoboLabSession,
        build_live_robolab_env,
        close_live_droid_stack,
        write_queue_row,
    )
    from experiments.online_correction_v4.droid_task_files.binding import sha256_file as registry_sha256
    from experiments.online_correction_v4.droid_task_files.reset_registry import (
        MODEL_BLIND_CANDIDATE_STATUS,
        load_reset_registry,
    )
    from experiments.online_correction_v4.model_blind_g2 import task_frame_evidence
    from run_v4_horizontal_g3_scripted_seed import (
        FIXTURE_PROMPTS,
        FROZEN_SCRIPTED_GEOMETRIC_TOLERANCE_M,
        _git_identity,
        _gpu_identity,
        frozen_scripted_controller_config,
        validate_scripted_seed_gate_inputs,
    )

    campaign_path = args.campaign.resolve()
    plan_path = args.plan.resolve()
    registry_path = args.reset_registry.resolve()
    robolab_root = (args.robolab_root or Path(os.environ["ROBOLAB_ROOT"])).resolve()
    plan = json.loads(plan_path.read_bytes())
    campaign = json.loads(campaign_path.read_bytes())
    validate_scripted_seed_gate_inputs(
        plan=plan,
        campaign=campaign,
        campaign_path=campaign_path,
        campaign_sha256=args.campaign_sha256,
        plan_path=plan_path,
        plan_sha256=args.plan_sha256,
        reset_registry_path=registry_path,
        reset_registry_sha256=args.reset_registry_sha256,
        environment_seed=args.environment_seed,
        scale=args.scale,
        mode="stationary",
        expected_fixture_id=FIXTURE_ID,
        sha256_file=registry_sha256,
    )
    if not args.expected_driver_version:
        raise RuntimeError("--expected-driver-version is required for live execution")
    pod_name = args.pod or os.environ.get("POD_NAME")
    pod_uid = args.pod_uid or os.environ.get("POD_UID")
    if not pod_name or not pod_uid:
        raise RuntimeError("--pod/--pod-uid or POD_NAME/POD_UID are required")

    timing = TimingConfig.from_mapping(campaign["timing"])
    detector = NaturalGraspDetector(
        config=GraspDetectorConfig(
            min_lift_m=timing.natural_grasp_min_lift_m,
            dwell_s=timing.natural_grasp_dwell_s,
            relative_drift_max_m=timing.kinematic_grasp_relative_drift_max_m,
            trigger_deadline_s=timing.trigger_deadline_s,
        ),
        control_dt_s=args.native_control_dt_s,
    )

    registry = load_reset_registry(
        registry_path=str(registry_path),
        registry_sha256=args.reset_registry_sha256,
        required_status=MODEL_BLIND_CANDIDATE_STATUS,
        expected_fixture_id=FIXTURE_ID,
    )
    registry_payload = json.loads(registry_path.read_bytes())
    from run_v4_horizontal_g3_path_seed import _fixture_geometry_from_registry

    fixture_geometry = _fixture_geometry_from_registry(registry_payload, FIXTURE_ID)
    if args.environment_seed not in registry.positions_by_env_seed:
        raise RuntimeError("environment seed is absent from reset registry")

    fixture_spec = fixture_object_spec(FIXTURE_ID)
    prompt = FIXTURE_PROMPTS[FIXTURE_ID]
    goal = args.goal.strip().lower()
    episode_id = (
        f"online-correction-v4-natural-grasp-positive-control-{FIXTURE_ID.replace('_', '-')}-"
        f"{args.environment_seed}-scale-{args.scale:g}-{goal}"
    )
    from experiments.online_correction_v4.droid_contract import sha256_bytes

    prompt_sha256 = sha256_bytes(prompt.encode("utf-8"))
    queue_row, queue_row_sha256 = write_queue_row(
        output_dir=output_dir,
        episode_id=episode_id,
        fixture_id=FIXTURE_ID,
        prompt_text=prompt,
        prompt_sha256=prompt_sha256,
        env_seed=args.environment_seed,
        goal=goal,
    )
    runtime_identity = {
        "study_checkout": _git_identity(study_root, expected_commit=args.expected_study_commit),
        "robolab_checkout": _git_identity(robolab_root, expected_commit=args.expected_robolab_commit),
        "gpu": _gpu_identity(
            expected_driver=args.expected_driver_version,
            gpu_uuid=args.gpu_uuid,
        ),
        "pod": pod_name,
        "pod_uid": pod_uid,
        "positive_control_runner_sha256": sha256_file(Path(__file__).resolve()),
        "natural_grasp_detector_module_sha256": sha256_file(
            study_root / "experiments/online_correction_v4/detectors.py"
        ),
        "droid_robolab_sha256": sha256_file(
            study_root / "experiments/online_correction_v4/droid_robolab.py"
        ),
    }
    runtime_identity_sha256 = sha256_bytes(canonical_json_bytes(runtime_identity))

    env = None
    receipt: dict[str, Any] | None = None
    infra: dict[str, Any] | None = None
    try:
        fixture = ResetFixtureBinding(
            fixture_id=FIXTURE_ID,
            reset_registry_sha256=args.reset_registry_sha256,
            reset_registry_uri=f"file://{registry_path}",
        )
        os.environ["V4_DROID_RENDERER"] = "realtime"
        os.environ["V4_DROID_RENDERING_MODE"] = "balanced"
        os.environ["ONLINE_CORRECTION_V4_OUTPUT_DIR"] = str((output_dir / "robolab_native").resolve())
        RoboLabSession.begin_episode(episode_id)
        env = build_live_robolab_env(
            fixture=fixture,
            env_seed=args.environment_seed,
            episode_id=episode_id,
            goal=goal,
            prompt_text=prompt,
            prompt_sha256=prompt_sha256,
            policy_id="model_blind_no_policy",
            queue_row_path=queue_row,
            queue_row_sha256=queue_row_sha256,
            output_dir=output_dir / "robolab_native",
            locked_native_control_dt_s=args.native_control_dt_s,
            g3_contact_probe=True,
            action_mode="absolute_ik",
        )
        reset_check_id = f"{episode_id}-registered-reset"
        _reset_attestation, physical_reset = env.reset_for_model_blind_g3(
            check_id=reset_check_id,
            prompt_sha256=prompt_sha256,
            runtime_identity_sha256=runtime_identity_sha256,
        )
        task_frame_dict = task_frame_evidence(physical_reset)
        task_frame_from_evidence(task_frame_dict)
        initial_scene = env.backend.g3_scene_state()
        geometry_contract = dict(plan["geometry_contract"])
        geometry = geometry_from_scene_for_fixture(
            fixture_id=FIXTURE_ID,
            task_frame_evidence=task_frame_dict,
            scene_state=initial_scene,
            support_edge_margin_m=float(geometry_contract["support_edge_margin_m"]),
            fixture_geometry=fixture_geometry,
        )
        controller_config = frozen_scripted_controller_config(FIXTURE_ID)
        table_bounds = geometry["table_bounds_task"]
        target_footprint = geometry["target_footprint"]
        reference_position = tuple(
            float(value)
            for value in physical_reset["objects"][fixture_spec.reference_object][
                "position_world_xyz_m"
            ]
        )
        goal_set = goal_set_for_reference(
            geometry=geometry,
            relation=goal,
            reference_position_world=reference_position,
            clearance_m=float(geometry_contract["relation_clearance_m"]),
        )
        instrumented = GraspDetectorInstrumentedEnv(env, detector)
        control_mode = args.control_mode
        if control_mode == "hold_only":
            observation_ticks = max(
                int(round(timing.trigger_deadline_s / args.native_control_dt_s)) + 2,
                4,
            )
            for _ in range(observation_ticks):
                instrumented.step(env.hold_action())
            trajectory_result = {
                "passed": True,
                "mode": "hold_only",
                "tick_count": observation_ticks,
                "stages": {},
            }
        else:
            trajectory_result = run_scripted_check(
                instrumented,
                target_object=fixture_spec.target_object,
                reference_object=fixture_spec.reference_object,
                relation=goal,
                goal=goal_set,
                frame=geometry["frame"],
                config=controller_config,
                table_top_z_task=float(table_bounds.z_max),
                object_half_up=float(target_footprint.half_up),
                geometric_tol_m=FROZEN_SCRIPTED_GEOMETRIC_TOLERANCE_M,
                fixture_id=FIXTURE_ID,
            )
            hold_ticks = max(
                int(round(timing.natural_grasp_dwell_s / args.native_control_dt_s)) + 2, 4
            )
            for _ in range(hold_ticks):
                if detector.eligible:
                    break
                state = env.object_kinematic_state()
                detector.update(state)
                instrumented.detector_samples.append(
                    {
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
                        "trigger_eligible": detector.eligible,
                        "grasp_occurred": detector.grasp_occurred,
                        "event": None
                        if detector.event is None
                        else detector.event.__dict__,
                        "phase": "post_scripted_hold",
                    }
                )

        scripted_grasp_stages = trajectory_result.get("stages", {})
        trigger_event = detector.event.__dict__ if detector.event is not None else None
        if control_mode == "hold_only":
            verdict = (
                "negative_control_passed"
                if not detector.eligible
                else "blocking_false_positive"
            )
        else:
            verdict = (
                "intervention_deliverable" if detector.eligible else "blocking_setup_defect"
            )
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "fixture_id": FIXTURE_ID,
            "attempt_id": args.attempt_id,
            "control_mode": control_mode,
            "environment_seed": args.environment_seed,
            "scale": float(args.scale),
            "goal": goal,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "detector_class": "NaturalGraspDetector",
            "detector_config": {
                "min_lift_m": timing.natural_grasp_min_lift_m,
                "dwell_s": timing.natural_grasp_dwell_s,
                "relative_drift_max_m": timing.kinematic_grasp_relative_drift_max_m,
                "trigger_deadline_s": timing.trigger_deadline_s,
            },
            "native_control_dt_s": args.native_control_dt_s,
            "scripted_trajectory_passed": bool(trajectory_result.get("passed")),
            "scripted_grasp_stages": scripted_grasp_stages,
            "trigger_eligible": detector.eligible,
            "grasp_occurred": detector.grasp_occurred,
            "trigger_event": trigger_event,
            "verdict": verdict,
            "passed": (
                detector.eligible
                if control_mode == "scripted_grasp"
                else not detector.eligible
            ),
            "status": (
                "passed"
                if (
                    detector.eligible
                    if control_mode == "scripted_grasp"
                    else not detector.eligible
                )
                else (
                    "blocking_setup_defect"
                    if control_mode == "scripted_grasp"
                    else "blocking_false_positive"
                )
            ),
            "runtime_identity": runtime_identity,
            "detector_sample_count": len(instrumented.detector_samples),
            "detector_samples_tail": instrumented.detector_samples[-8:],
            "trajectory_tick_count": trajectory_result.get("tick_count"),
        }
        receipt_identity = _write_json_exclusive(output_dir / "natural_grasp_positive_control.json", receipt)
        _write_json_exclusive(
            output_dir / "scripted_trajectory.json",
            trajectory_result,
        )
        _write_json_exclusive(
            output_dir / "detector_samples.json",
            {
                "schema_version": "v4-natural-grasp-detector-live-samples-v1",
                "samples": instrumented.detector_samples,
            },
        )
        print(json.dumps({"receipt": receipt_identity, "verdict": verdict, "trigger_eligible": detector.eligible}))
        passed = receipt["passed"]
        return 0 if passed else 2
    except Exception as exc:
        infra = {
            "schema_version": "v4-vertical-natural-grasp-live-positive-control-infra-v1",
            "attempt_id": args.attempt_id,
            "fixture_id": FIXTURE_ID,
            "class": "infrastructure_failure",
            "detail": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json_exclusive(output_dir / "infrastructure_failure.json", infra)
        raise
    finally:
        if env is not None:
            close_live_droid_stack()


if __name__ == "__main__":
    raise SystemExit(main())
