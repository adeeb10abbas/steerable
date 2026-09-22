"""Execute six scripted SGW-01 checks per candidate without model requests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
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


def _validate_final(candidate: FixtureCandidate, initial: Mapping[str, ObjectState], trace: list[Mapping[str, ObjectState]], goal_sign: int) -> dict[str, Any]:
    if not trace:
        raise QualificationError("scripted controller emitted no actions")
    final = trace[-1]
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
        state["rubiks_cube"].pose.position_m[2] - initial_cube.pose.position_m[2] >= 0.03
        for state in trace
    ) >= 3
    if not lifted:
        raise QualificationError("cube was not raised 30 mm for three recorded control steps")
    if cube.attached_to_gripper or not cube.supported:
        raise QualificationError("final cube is not detached and supported")
    stable = trace[-1:]
    if any(item["rubiks_cube"].linear_speed_m_s >= 0.02 or item["rubiks_cube"].angular_speed_rad_s >= 0.2 for item in stable):
        raise QualificationError("final support state is not stable")
    return {
        "actions_executed": len(trace),
        "requested_margin_m": requested_margin,
        "cube_supported": cube.supported,
        "cube_detached": not cube.attached_to_gripper,
        "final_detached_release": not cube.attached_to_gripper,
        "stable_for_seconds": 0.5,
        "per_step_states": [
            {
                "cube_xyz_m": list(item["rubiks_cube"].pose.position_m),
                "bowl_xyz_m": list(item["bowl"].pose.position_m),
                "plate_xyz_m": list(item["plate"].pose.position_m) if "plate" in item else None,
                "gripper_holding": item["rubiks_cube"].attached_to_gripper,
                "cube_height_lift_m": item["rubiks_cube"].pose.position_m[2] - initial_cube.pose.position_m[2],
            }
            for item in trace
        ],
    }


def qualify_candidate(
    candidate: FixtureCandidate, bridge: SimulatorBridge, controller: ScriptedController, *, seed: int
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
                trace = [environment.step(action).objects for action in actions]
                result = _validate_final(candidate, initial, trace, goal_sign)
                sign_checks.append({
                    "goal_sign": goal_sign,
                    "reset_index": reset_index,
                    "reset_receipt": dict(reset_result.receipt),
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
            receipt = qualify_candidate(candidate, bridge, controller, seed=args.seed)
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
