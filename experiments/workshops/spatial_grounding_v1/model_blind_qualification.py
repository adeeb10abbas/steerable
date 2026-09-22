"""Run six recorded, model-blind physical checks in one candidate's fresh process."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

import numpy as np

from .fixtures import ACTION_CAP, FixtureCandidate, FixtureError, validate_reset
from .recorder import atomic_json, encode_viewport_video
from .scoring import GoalSpec, OutcomeStatus, score_episode
from .simulator_bridge import SimulatorBridge, ScriptedController, SimulatorSnapshot, load_candidate_file, load_factory
from .task_definitions import build_task_definition


class QualificationError(RuntimeError):
    """The physical qualification cannot supply complete accepted evidence."""


def _state(snapshot: SimulatorSnapshot, index: int) -> dict[str, Any]:
    cube = snapshot.objects["rubiks_cube"]
    return {
        "action_step": index, "sim_time_s": snapshot.simulated_time_s,
        "cube_xyz_m": cube.pose.position_m,
        "bowl_xyz_m": snapshot.objects["bowl"].pose.position_m,
        "plate_xyz_m": snapshot.objects["plate"].pose.position_m if "plate" in snapshot.objects else None,
        "supported": cube.supported,
        "final_detached_release": not cube.attached_to_gripper,
        "gripper_holding": cube.attached_to_gripper,
        "linear_speed_m_s": cube.linear_speed_m_s,
        "angular_speed_rad_s": cube.angular_speed_rad_s,
    }


class _TrialEvidence:
    def __init__(self, path: Path, dt: float) -> None:
        if not math.isfinite(dt) or dt <= 0:
            raise QualificationError("qualification requires measured positive control_step_dt_s")
        path.mkdir(parents=True, exist_ok=False)
        self.path = path
        self.dt = dt
        self.frames: list[Path] = []
        self.states: list[dict[str, Any]] = []

    def _array(self, name: str, value: Any) -> str:
        array = np.asarray(value)
        if array.dtype == object or not np.isfinite(array).all():
            raise QualificationError("trial array is nonfinite or nonnumeric")
        path = self.path / name
        with path.open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def command(self, index: int, action: Any) -> None:
        digest = self._array(f"action-{index:04d}.npy", action)
        atomic_json(self.path / f"command-{index:04d}.json",
                    {"action_step": index, "sha256": digest, "status": "issued_not_yet_observed"})

    def observe(self, snapshot: SimulatorSnapshot, frame: Any) -> None:
        index = len(self.frames)
        array = np.asarray(frame)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3 or not np.ptp(array):
            raise QualificationError("viewport must be nonblank HWC uint8 RGB")
        name = f"frame-{index:04d}.npy"
        digest = self._array(name, array)
        state = _state(snapshot, index)
        atomic_json(self.path / f"state-{index:04d}.json", {
            "state": state, "raw_snapshot": asdict(snapshot),
            "viewport_path": name, "viewport_sha256": digest,
        })
        self.frames.append(self.path / name)
        self.states.append(state)

    def finish(self, result: Mapping[str, Any]) -> dict[str, Any]:
        video = encode_viewport_video(self.frames, self.path / "viewport.mp4", fps=1 / self.dt) if self.frames else None
        receipt = {
            **dict(result), "model_request_count": 0, "behavioral_episode_count": 0,
            "observed_actions": max(0, len(self.states) - 1),
            "viewport_video": video,
            "per_step_states": self.states,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                for path in sorted(self.path.iterdir()) if path.is_file()
            },
        }
        atomic_json(self.path / "trial.json", receipt)
        return receipt


def qualify_candidate(
    candidate: FixtureCandidate, bridge: SimulatorBridge, controller: ScriptedController, *,
    seed: int, evidence_root: Path,
) -> dict[str, Any]:
    """Retain rejected and interrupted trials; accept only all six full traces."""
    task = build_task_definition(candidate)
    checks: list[dict[str, Any]] = []
    environment = bridge.create_environment(task, seed)
    try:
        for goal_sign in (1, -1):
            resets = []
            sign_checks = []
            for reset_index in range(3):
                reset = environment.reset()
                reset.validate_for_policy()
                resets.append(reset.snapshot.reset_snapshot())
                evidence = _TrialEvidence(
                    evidence_root / f"goal-{goal_sign:+d}" / f"reset-{reset_index}",
                    float(reset.receipt["control_step_dt_s"]),
                )
                atomic_json(evidence.path / "reset.json", dict(reset.receipt))
                try:
                    evidence.observe(reset.snapshot, environment.render_viewport())
                    actions = list(controller.actions_for_goal(environment, candidate, goal_sign))
                    if len(actions) != ACTION_CAP:
                        raise QualificationError("scripted plan must cover exactly 450 controller actions")
                    for index, action in enumerate(actions, 1):
                        evidence.command(index, action)
                        snapshot = environment.step(action)
                        evidence.observe(snapshot, environment.render_viewport())
                        if snapshot.termination_reason and index < ACTION_CAP:
                            raise QualificationError(f"native environment ended at action {index}: {snapshot.termination_reason}")
                    score = score_episode(
                        {"states": evidence.states, "termination_reason": "action_cap"},
                        GoalSpec(candidate.family, goal_sign),
                    )
                    if score.status is not OutcomeStatus.VALID_MODEL:
                        raise QualificationError(f"incomplete physical trace: {score.infrastructure_reason}")
                except Exception:
                    evidence.finish({
                        "status": "infrastructure_invalid_qualification",
                        "error": traceback.format_exc(), "goal_sign": goal_sign, "reset_index": reset_index,
                    })
                    raise
                result = evidence.finish({
                    "status": "passed_scripted_goal" if score.requested_success else "rejected_scripted_goal",
                    "passed": score.requested_success, "goal_sign": goal_sign, "reset_index": reset_index,
                    "reset_receipt": dict(reset.receipt), "actions_executed": len(actions),
                    "requested_margin_m": score.terminal_margin_m, "score": asdict(score),
                })
                sign_checks.append(result)
            try:
                reset_rows = validate_reset(candidate, resets)
            except FixtureError as error:
                for check in sign_checks:
                    check.update(passed=False, reset_error=str(error))
            else:
                for check in sign_checks:
                    check["reset_validation"] = [row for row in reset_rows if row["repeat"] == check["reset_index"]]
            checks.extend(sign_checks)
    finally:
        environment.close()
    accepted = len(checks) == 6 and all(check["passed"] for check in checks)
    return {
        "schema_version": "sgw-01-model-blind-fixture-qualification-v1",
        "status": "accepted_model_blind_fixture_candidate" if accepted else "rejected_model_blind_fixture_candidate",
        "model_request_count": 0, "behavioral_episode_count": 0,
        "candidate_id": candidate.candidate_id, "family": candidate.family, "seed": seed,
        "candidate_sha256": hashlib.sha256(json.dumps(asdict(candidate), sort_keys=True).encode()).hexdigest(),
        "action_cap": ACTION_CAP, "checks": checks,
    }


def _load_selected_candidate(args: argparse.Namespace) -> FixtureCandidate:
    if args.proposal_file:
        value = json.loads(args.proposal_file.read_text())
        if (value.get("status") != "proposed_unqualified" or value.get("model_request_count") != 0
                or value.get("behavioral_episode_count") != 0):
            raise QualificationError("proposal file lacks explicit unqualified/model-blind status")
        rows = value.get("candidates", [])
        if not 1 <= len(rows) <= 100:
            raise QualificationError("proposal file exceeds the family candidate cap")
        matches = [row for row in rows if row["candidate_id"] == args.candidate_id]
        if len(matches) != 1:
            raise QualificationError("select exactly one candidate from the frozen proposal file")
        if matches[0].get("metadata", {}).get("geometric_screen_status") != "passed":
            raise QualificationError("candidate did not pass the recorded geometric screen")
        return FixtureCandidate.from_json(matches[0])
    paths = list(args.candidate_root.glob(f"{args.family}/*.json"))
    if len(paths) != 1:
        raise QualificationError("run exactly one candidate per fresh Isaac process")
    return load_candidate_file(paths[0])


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--family", required=True, choices=("LAT", "HEIGHT", "DIST"))
    source = bootstrap.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-root", type=Path)
    source.add_argument("--proposal-file", type=Path)
    bootstrap.add_argument("--candidate-id")
    bootstrap.add_argument("--output-root", type=Path, required=True)
    bootstrap.add_argument("--bridge-factory", required=True)
    bootstrap.add_argument("--controller-factory", required=True)
    bootstrap.add_argument("--seed", type=int, default=20260922)
    bootstrap.add_argument("--robolab-root", type=Path, required=True)
    bootstrap.add_argument("--assets-manifest", type=Path, required=True)
    known, _ = bootstrap.parse_known_args()
    if known.output_root.exists():
        raise FileExistsError(f"refusing to overwrite qualification output: {known.output_root}")
    from isaaclab.app import AppLauncher
    from robolab.eval.runner import add_common_eval_args
    parser = argparse.ArgumentParser(parents=[bootstrap], allow_abbrev=False)
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.headless or args.renderer != "realtime" or args.rendering_type != "balanced":
        raise QualificationError("physical qualification requires headless realtime/balanced RTX")
    candidate = _load_selected_candidate(args)
    if candidate.family != args.family:
        raise QualificationError("candidate family differs from selected family")
    if hashlib.sha256(args.assets_manifest.read_bytes()).hexdigest() != candidate.asset_manifest_sha256:
        raise QualificationError("actual asset manifest differs from candidate binding")
    import imageio_ffmpeg
    imageio_ffmpeg.get_ffmpeg_exe()
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    args.output_root.mkdir(parents=True, exist_ok=False)
    app = AppLauncher(args).app
    try:
        import robolab
        import robolab.constants
        from robolab.constants import set_output_dir
        if not Path(robolab.__file__).resolve().is_relative_to(args.robolab_root.resolve()):
            raise QualificationError("RoboLab import is outside the pinned checkout")
        set_output_dir(str(args.output_root / "native"))
        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False
        robolab.constants.RECORD_IMAGE_DATA = False
        bridge = load_factory(args.bridge_factory)(
            robolab_root=args.robolab_root, assets_manifest=args.assets_manifest,
            device=args.device, renderer=args.renderer, rendering_type=args.rendering_type,
            evidence_root=args.output_root / "reset_warmup",
        )
        controller = load_factory(args.controller_factory)(
            robolab_root=args.robolab_root, assets_manifest=args.assets_manifest, device=args.device,
        )
        receipt = qualify_candidate(candidate, bridge, controller, seed=args.seed, evidence_root=args.output_root / "trials")
        atomic_json(args.output_root / "qualification.json", receipt)
        print(json.dumps({"candidate_id": candidate.candidate_id, "status": receipt["status"]}))
    except Exception:
        atomic_json(args.output_root / "qualification.json", {
            "status": "infrastructure_invalid_qualification", "candidate_id": candidate.candidate_id,
            "model_request_count": 0, "behavioral_episode_count": 0, "error": traceback.format_exc(),
        })
        raise
    finally:
        app.close()


if __name__ == "__main__":
    main()
