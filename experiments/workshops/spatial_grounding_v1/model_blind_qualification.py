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

from .fixtures import ACTION_CAP, FixtureCandidate, FixtureError, Pose, pose_error, validate_reset
from .prospective_family_designs import require_design_capture
from .recorder import atomic_json, encode_viewport_video
from .scoring import GoalSpec, OutcomeStatus, score_episode
from .simulator_bridge import (
    PhysicalGeometryRejection, SimulatorBridge, ScriptedController, SimulatorSnapshot, load_candidate_file, load_factory,
)
from .task_definitions import build_task_definition


class QualificationError(RuntimeError):
    """The physical qualification cannot supply complete accepted evidence."""


def _state(snapshot: SimulatorSnapshot, index: int) -> dict[str, Any]:
    return snapshot.scoring_state(index)


def validate_preaction_reset_geometry(snapshot: SimulatorSnapshot, candidate: FixtureCandidate) -> PhysicalGeometryRejection | None:
    """Screen measured reset geometry before requesting a controller plan."""

    if candidate.family not in {"HEIGHT", "DIST"}:
        return None
    context = snapshot.context_measurements
    expected = candidate.metadata.get("baseline_banana_pose")
    supports = candidate.metadata.get("geometry_guard_support_ids")
    if not isinstance(context, Mapping) or not isinstance(expected, Mapping) or not isinstance(supports, list):
        raise QualificationError("family reset lacks measured banana geometry guard inputs")
    required = {"banana", "table", *supports}
    if not required.issubset(context):
        raise QualificationError("family reset lacks measured banana/table/support context")
    try:
        observed = Pose.from_json({
            "position_m": context["banana"]["root_position_env_local_xyz_m"],
            "quaternion_wxyz": context["banana"]["root_quaternion_world_wxyz"],
        })
        reference = Pose.from_json(expected)
    except (KeyError, TypeError, ValueError) as error:
        raise QualificationError("family reset banana pose is malformed") from error
    position_error, angle_error = pose_error(observed, reference)
    if position_error > .003 or angle_error > 2.0:
        return PhysicalGeometryRejection(scope="reset", reason="banana_pose_differs_from_fixed_baseline")
    table_minimum, table_maximum = _context_bounds(context["table"], "table")
    banana_minimum, banana_maximum = _context_bounds(context["banana"], "banana")
    if (
        banana_minimum[0] < table_minimum[0] or banana_maximum[0] > table_maximum[0]
        or banana_minimum[1] < table_minimum[1] or banana_maximum[1] > table_maximum[1]
    ):
        return PhysicalGeometryRejection(scope="reset", reason="banana_leaves_measured_table_bounds")
    for name in supports:
        minimum, maximum = _context_bounds(context[name], name)
        dx = max(minimum[0] - banana_maximum[0], banana_minimum[0] - maximum[0], 0.0)
        dy = max(minimum[1] - banana_maximum[1], banana_minimum[1] - maximum[1], 0.0)
        if math.hypot(dx, dy) < .02:
            return PhysicalGeometryRejection(scope="reset", reason=f"banana_clearance_below_20mm:{name}")
    return None


def _context_bounds(row: Any, name: str) -> tuple[list[float], list[float]]:
    if not isinstance(row, Mapping):
        raise QualificationError(f"family reset lacks {name} geometry")
    minimum, maximum = row.get("bbox_env_local_min_xyz_m"), row.get("bbox_env_local_max_xyz_m")
    if (
        not isinstance(minimum, (list, tuple)) or not isinstance(maximum, (list, tuple))
        or len(minimum) != 3 or len(maximum) != 3
        or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in (*minimum, *maximum))
        or any(low > high for low, high in zip(minimum, maximum, strict=True))
    ):
        raise QualificationError(f"family reset has malformed {name} bounds")
    return list(minimum), list(maximum)


def _write_preaction_geometry_guard(
    trial: Path, candidate: FixtureCandidate, goal_sign: int, reset_index: int, *,
    physical_geometry_rejection: Mapping[str, str] | None,
) -> None:
    """Bind state-0 to the trial before the first controller command is issued."""

    state = trial / "state-0000.json"
    if not state.is_file():
        raise QualificationError("preaction guard requires retained raw reset state")
    candidate_capture_sha256 = candidate.metadata.get("candidate_capture_sha256")
    design_id = candidate.metadata.get("prospective_design_id")
    if not isinstance(candidate_capture_sha256, str) or len(candidate_capture_sha256) != 64 or not isinstance(design_id, str):
        # LAT has no prospective capture chain and intentionally does not emit
        # the family-worker guard.
        if candidate.family == "LAT":
            return
        raise QualificationError("family candidate lacks prospective capture/design binding")
    value: dict[str, Any] = {
        "schema_version": "sgw-01-family-preaction-geometry-guard-v1",
        "design_id": design_id,
        "candidate_sha256": hashlib.sha256(
            json.dumps(asdict(candidate), sort_keys=True).encode()
        ).hexdigest(),
        "candidate_capture_sha256": candidate_capture_sha256,
        "goal_sign": goal_sign,
        "reset_index": reset_index,
        "raw_reset": {
            "path": str(state.resolve()), "sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
            "bytes": state.stat().st_size,
        },
        "status": "measured_banana_geometry_valid_before_actions",
        "controller_actions_executed": 0,
    }
    if physical_geometry_rejection is not None:
        value.update({
            "status": "physical_geometry_rejection_before_actions",
            "rejection_scope": physical_geometry_rejection["scope"],
            "reason": physical_geometry_rejection["reason"],
        })
    atomic_json(trial / "preaction-geometry-guard.json", value)


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
                    rejection = validate_preaction_reset_geometry(reset.snapshot, candidate)
                    if rejection is not None:
                        raise rejection
                    _write_preaction_geometry_guard(
                        evidence.path, candidate, goal_sign, reset_index,
                        physical_geometry_rejection=None,
                    )
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
                except PhysicalGeometryRejection as rejection:
                    _write_preaction_geometry_guard(
                        evidence.path, candidate, goal_sign, reset_index,
                        physical_geometry_rejection={"scope": rejection.scope, "reason": rejection.reason},
                    )
                    result = evidence.finish({
                        "status": "physical_geometry_rejection_before_actions", "passed": False,
                        "goal_sign": goal_sign, "reset_index": reset_index, "reset_receipt": dict(reset.receipt),
                        "actions_executed": 0, "requested_margin_m": None,
                        "physical_geometry_rejection": {"rejection_scope": rejection.scope, "reason": rejection.reason},
                    })
                    sign_checks.append(result)
                    checks.extend(sign_checks)
                    return {
                        "schema_version": "sgw-01-model-blind-fixture-qualification-v1",
                        "status": "rejected_model_blind_fixture_candidate",
                        "model_request_count": 0, "behavioral_episode_count": 0,
                        "candidate_id": candidate.candidate_id, "family": candidate.family, "seed": seed,
                        "candidate_sha256": hashlib.sha256(json.dumps(asdict(candidate), sort_keys=True).encode()).hexdigest(),
                        "controller_identity": getattr(controller, "identity", {"recipe": "unattested_controller"}),
                        "action_cap": ACTION_CAP, "checks": checks,
                        "physical_geometry_rejection_before_actions": {
                            "goal_sign": goal_sign, "reset_index": reset_index,
                            "rejection_scope": rejection.scope, "reason": rejection.reason,
                            "controller_actions_executed": 0,
                        },
                    }
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
        "controller_identity": getattr(controller, "identity", {"recipe": "unattested_controller"}),
        "action_cap": ACTION_CAP, "checks": checks,
    }


def _load_selected_candidate(args: argparse.Namespace) -> FixtureCandidate:
    if getattr(args, "candidate_file", None):
        candidate = load_candidate_file(args.candidate_file)
        if args.candidate_id is not None and candidate.candidate_id != args.candidate_id:
            raise QualificationError("explicit candidate file differs from requested candidate id")
        return candidate
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


def _preflight_output_root(args: argparse.Namespace) -> None:
    """Allow the executor's prepared family root without overwriting any evidence."""
    if not args.candidate_file:
        if args.output_root.exists():
            raise FileExistsError(f"refusing to overwrite qualification output: {args.output_root}")
        return
    root = args.output_root.resolve()
    if args.candidate_file.resolve() != root / "candidate.json":
        raise QualificationError("explicit family candidate must be output-root/candidate.json")
    if args.candidate_manifest is None or args.candidate_capture is None:
        raise QualificationError("explicit family candidate requires manifest and fresh capture bindings")
    if (args.candidate_manifest.resolve() != root / "candidate_manifest.json"
            or args.candidate_capture.resolve() != root / "candidate_capture.json"):
        raise QualificationError("explicit family bindings must use canonical output-root paths")
    if not root.is_dir() or not all(path.is_file() for path in (
        args.candidate_file, args.candidate_manifest, args.candidate_capture,
    )):
        raise QualificationError("explicit family candidate root lacks canonical immutable inputs")
    candidate = _load_selected_candidate(args)
    try:
        capture = require_design_capture(
            candidate_manifest_path=args.candidate_manifest, capture_path=args.candidate_capture,
        )
    except Exception as error:
        raise QualificationError(f"explicit family capture binding is invalid: {error}") from error
    if (
        candidate.family not in {"HEIGHT", "DIST"} or capture.get("family") != candidate.family
        or candidate.metadata.get("candidate_capture_sha256") != hashlib.sha256(args.candidate_capture.read_bytes()).hexdigest()
        or candidate.metadata.get("prospective_design_id")
        != json.loads(args.candidate_manifest.read_text(encoding="utf-8")).get("design_id")
    ):
        raise QualificationError("explicit family candidate differs from its manifest/capture binding")
    for name in ("qualification.json", "controller.json", "trials", "reset_warmup", "native"):
        if (root / name).exists():
            raise FileExistsError(f"refusing to overwrite existing qualification evidence: {root / name}")


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    bootstrap.add_argument("--family", required=True, choices=("LAT", "HEIGHT", "DIST"))
    source = bootstrap.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-root", type=Path)
    source.add_argument("--proposal-file", type=Path)
    source.add_argument("--candidate-file", type=Path)
    bootstrap.add_argument("--candidate-id")
    bootstrap.add_argument("--candidate-manifest", type=Path)
    bootstrap.add_argument("--candidate-capture", type=Path)
    bootstrap.add_argument("--output-root", type=Path, required=True)
    bootstrap.add_argument("--bridge-factory", required=True)
    bootstrap.add_argument("--controller-factory", required=True)
    bootstrap.add_argument("--controller-calibration", type=Path)
    bootstrap.add_argument("--seed", type=int, default=20260922)
    bootstrap.add_argument("--robolab-root", type=Path, required=True)
    bootstrap.add_argument("--assets-manifest", type=Path, required=True)
    known, _ = bootstrap.parse_known_args()
    _preflight_output_root(known)
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
    if candidate.family in {"HEIGHT", "DIST"}:
        from .robolab_height_dist_qualification import validate_candidate_inputs

        validate_candidate_inputs(candidate)
    import imageio_ffmpeg
    imageio_ffmpeg.get_ffmpeg_exe()
    # Keep candidate/calibration validation CPU-only: AppLauncher may reserve a
    # renderer before the bridge has a chance to reject malformed captures.
    bridge = load_factory(args.bridge_factory)(
        robolab_root=args.robolab_root, assets_manifest=args.assets_manifest,
        device=args.device, renderer=args.renderer, rendering_type=args.rendering_type,
        evidence_root=args.output_root / "reset_warmup",
    )
    controller = load_factory(args.controller_factory)(
        robolab_root=args.robolab_root, assets_manifest=args.assets_manifest, device=args.device,
        controller_calibration=args.controller_calibration,
    )
    from isaaclab.app import AppLauncher
    args.enable_cameras = True
    args.output_root.mkdir(parents=True, exist_ok=args.candidate_file is not None)
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
        atomic_json(args.output_root / "controller.json", getattr(controller, "identity", {"recipe": "unattested_controller"}))
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
