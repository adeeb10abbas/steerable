"""Execute six scripted SGW-01 checks per candidate without model requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from typing import Any, Iterable, Mapping

from .fixtures import (
    ACTION_CAP,
    GOAL_MARGIN_M,
    REFERENCE_MOTION_LIMIT_M,
    FixtureCandidate,
    FixtureError,
    validate_reset,
    write_json,
)
from .simulator_bridge import Environment, ObjectState, ScriptedController, SimulatorBridge, load_candidate_file, load_factory
from .task_definitions import build_task_definition


class QualificationError(RuntimeError):
    """A physical candidate did not satisfy the frozen acceptance contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relation(candidate: FixtureCandidate, objects: Mapping[str, ObjectState]) -> float:
    return candidate.relation_m({name: state.pose for name, state in objects.items()})


def _validate_final(candidate: FixtureCandidate, initial: Mapping[str, ObjectState], trace: list[Any], goal_sign: int) -> dict[str, Any]:
    if not trace:
        raise QualificationError("scripted controller emitted no actions")
    final = trace[-1].objects
    if len(trace) > ACTION_CAP:
        raise QualificationError("scripted check exceeded the 450 action cap")
    requested_margin = goal_sign * _relation(candidate, final)
    if requested_margin < GOAL_MARGIN_M:
        raise QualificationError(f"goal margin {requested_margin:.6f} m is below 30 mm")
    for reference in ("bowl", "plate"):
        if reference not in final:
            continue
        displacement = max(
            abs(a - b)
            for a, b in zip(final[reference].pose.position_m, initial[reference].pose.position_m, strict=True)
        )
        if displacement > REFERENCE_MOTION_LIMIT_M:
            raise QualificationError(f"{reference} moved {displacement:.6f} m, exceeding 5 mm")
    cube = final["rubiks_cube"]
    initial_cube = initial["rubiks_cube"]
    lifted = sum(
        snapshot.objects["rubiks_cube"].pose.position_m[2] - initial_cube.pose.position_m[2] >= 0.03
        for snapshot in trace
    ) >= 3
    if not lifted:
        raise QualificationError("cube was not raised 30 mm for three recorded control steps")
    if cube.attached_to_gripper or not cube.supported:
        raise QualificationError("final cube is not detached and supported")
    stable_start: float | None = None
    for snapshot in trace:
        current = snapshot.objects["rubiks_cube"]
        stable_now = (
            current.supported
            and not current.attached_to_gripper
            and current.linear_speed_m_s < 0.02
            and current.angular_speed_rad_s < 0.2
        )
        if stable_now and stable_start is None:
            stable_start = snapshot.simulated_time_s
        elif not stable_now:
            stable_start = None
    if stable_start is None:
        raise QualificationError("final support state is not stable")
    stable_for_seconds = trace[-1].simulated_time_s - stable_start
    if stable_for_seconds < 0.5:
        raise QualificationError("final detached supported state did not persist for 0.5 simulated seconds")
    return {
        "actions_executed": len(trace),
        "requested_margin_m": requested_margin,
        "cube_supported": cube.supported,
        "cube_detached": not cube.attached_to_gripper,
        "final_detached_release": not cube.attached_to_gripper,
        "stable_for_seconds": stable_for_seconds,
        "per_step_states": [{
                "cube_xyz_m": list(initial["rubiks_cube"].pose.position_m),
                "bowl_xyz_m": list(initial["bowl"].pose.position_m),
                "plate_xyz_m": list(initial["plate"].pose.position_m) if "plate" in initial else None,
                "gripper_holding": initial["rubiks_cube"].attached_to_gripper,
                "cube_height_lift_m": 0.0,
                "simulated_time_s": 0.0,
        }, *[
            {
                "cube_xyz_m": list(snapshot.objects["rubiks_cube"].pose.position_m),
                "bowl_xyz_m": list(snapshot.objects["bowl"].pose.position_m),
                "plate_xyz_m": list(snapshot.objects["plate"].pose.position_m) if "plate" in snapshot.objects else None,
                "gripper_holding": snapshot.objects["rubiks_cube"].attached_to_gripper,
                "cube_height_lift_m": snapshot.objects["rubiks_cube"].pose.position_m[2] - initial_cube.pose.position_m[2],
                "simulated_time_s": snapshot.simulated_time_s,
            }
            for snapshot in trace
        ]],
    }


def qualify_candidate(
    candidate: FixtureCandidate, bridge: SimulatorBridge, controller: ScriptedController, *, seed: int,
    video_root: Path | None = None,
) -> dict[str, Any]:
    """Run exactly three resets for each sign and one scripted path per reset."""

    task = build_task_definition(candidate)
    checks: list[dict[str, Any]] = []
    for goal_sign in (1, -1):
        environment = bridge.create_environment(task, seed)
        try:
            sign_checks: list[dict[str, Any]] = []
            reset_results = []
            for reset_index in range(3):
                # A fresh physical reset begins every scripted check.  Collect
                # all three reset identities for the deterministic comparison
                # only after no action can alter another reset sample.
                reset_result = environment.reset()
                reset_result.validate_for_policy()
                reset_results.append(reset_result)
                initial = reset_result.snapshot.objects
                actions = controller.actions_for_goal(environment, candidate, goal_sign)
                frames = [_viewport(environment)]
                trace = []
                for action in actions:
                    trace.append(environment.step(action))
                    frames.append(_viewport(environment))
                result = _validate_final(candidate, initial, trace, goal_sign)
                video = None
                if video_root is not None:
                    video = video_root / candidate.candidate_id / f"{goal_sign:+d}-{reset_index}.mp4"
                    _encode_video(frames, video)
                sign_checks.append({
                    "goal_sign": goal_sign,
                    "reset_index": reset_index,
                    "reset_receipt": dict(reset_result.receipt),
                    "viewport_video": str(video) if video else None,
                    **result,
                })
            reset_rows = validate_reset(
                candidate, [result.snapshot.reset_snapshot() for result in reset_results]
            )
            for check in sign_checks:
                check["reset_validation"] = [
                    row for row in reset_rows if row["repeat"] == check["reset_index"]
                ]
            checks.extend(sign_checks)
        finally:
            environment.close()
    if len(checks) != 6:
        raise QualificationError("each candidate must retain exactly six scripted checks")
    return {
        "schema_version": "sgw-01-model-blind-fixture-qualification-v1",
        "status": "accepted_model_blind_fixture_candidate",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_id": candidate.candidate_id,
        "candidate_sha256": hashlib.sha256(json.dumps(candidate.task_payload(), sort_keys=True).encode()).hexdigest(),
        "family": candidate.family,
        "seed": seed,
        "action_cap": ACTION_CAP,
        "checks": checks,
    }


def _viewport(environment: Environment) -> np.ndarray:
    data = environment.render_viewport()
    frame = np.asarray(data)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise QualificationError("environment did not provide an RGB viewport frame")
    return frame.astype(np.uint8, copy=False)


def _encode_video(frames: list[np.ndarray], path: Path) -> None:
    try:
        import cv2
    except ImportError as error:
        raise QualificationError("pinned qualification runtime lacks OpenCV video encoding") from error
    if not frames:
        raise QualificationError("cannot encode an empty viewport recording")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (width, height))
    if not writer.isOpened():
        raise QualificationError("failed to open viewport video writer")
    try:
        for frame in frames:
            if frame.shape != frames[0].shape:
                raise QualificationError("viewport frame shape changed during trial")
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _candidate_paths(root: Path, family: str) -> Iterable[Path]:
    paths = sorted(root.glob(f"{family}/*.json"))
    if not paths:
        raise QualificationError(f"no {family} candidates exist under {root}")
    if len(paths) > 100:
        raise QualificationError("candidate root exceeds the frozen 100-candidate family cap")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=("LAT", "HEIGHT", "DIST"))
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bridge-factory", required=True, help="Pinned module:factory yielding SimulatorBridge")
    parser.add_argument("--controller-factory", required=True, help="Pinned module:factory yielding ScriptedController")
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--robolab-root", type=Path, required=True)
    parser.add_argument("--assets-manifest", type=Path, required=True)
    parser.add_argument("--renderer", default="realtime")
    parser.add_argument("--rendering-type", default="balanced")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.headless or args.renderer != "realtime" or args.rendering_type != "balanced":
        raise QualificationError("physical qualification requires --headless realtime/balanced RTX")
    if not (args.robolab_root / ".git").exists():
        raise QualificationError("RoboLab root is not a pinned checkout")
    if not args.assets_manifest.is_file():
        raise QualificationError("verified actual asset manifest is required")
    bridge = load_factory(args.bridge_factory)(
        robolab_root=args.robolab_root, assets_manifest=args.assets_manifest, device=args.device,
        renderer=args.renderer, rendering_type=args.rendering_type,
    )
    controller = load_factory(args.controller_factory)(
        robolab_root=args.robolab_root, assets_manifest=args.assets_manifest, device=args.device,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    for path in _candidate_paths(args.candidate_root, args.family):
        candidate = load_candidate_file(path)
        output = args.output_root / candidate.family / f"{candidate.candidate_id}.json"
        if output.exists():
            raise QualificationError(f"refusing to overwrite receipt {output}")
        try:
            receipt = qualify_candidate(candidate, bridge, controller, seed=args.seed, video_root=args.output_root / "videos")
        except Exception as error:
            receipt = {
                "schema_version": "sgw-01-model-blind-fixture-qualification-v1",
                "status": "rejected_model_blind_fixture_candidate",
                "model_request_count": 0,
                "behavioral_episode_count": 0,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "candidate_file": str(path.resolve()),
                "candidate_file_sha256": _sha256(path),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        write_json(output, receipt)


if __name__ == "__main__":
    main()
