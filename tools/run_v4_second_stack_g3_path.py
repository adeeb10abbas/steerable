#!/usr/bin/env python3
"""Run the live model-blind C8 swept-reference path gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.second_stack import (  # noqa: E402
    ENV_NAME,
    REFERENCE_OBJECT,
    SOURCE_OBJECT,
    active_contact_pairs,
    apply_registered_reset,
    ensure_registered_support,
    set_reference_xy,
    unwrap_simpler_env,
)
from tools.run_v4_second_stack_g2 import (  # noqa: E402
    canonical_json_bytes,
    load_json,
    sha256_file,
    verify_external_stack,
)


class SecondStackG3PathError(RuntimeError):
    pass


def minimum_jerk_fraction(fraction: float) -> float:
    clamped = min(max(float(fraction), 0.0), 1.0)
    return 10.0 * clamped**3 - 15.0 * clamped**4 + 6.0 * clamped**5


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _forbidden_contacts(contacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {"v4_second_stack_support"}
    return [
        row
        for row in contacts
        if REFERENCE_OBJECT in {row["actor0"], row["actor1"]}
        and next(
            name
            for name in (row["actor0"], row["actor1"])
            if name != REFERENCE_OBJECT
        )
        not in allowed
    ]


def _run_check(
    *,
    env: Any,
    env_seed: int,
    reset_row: Mapping[str, Any],
    plan_row: Mapping[str, Any],
    path_steps: int,
) -> dict[str, Any]:
    env.reset(seed=env_seed)
    apply_registered_reset(env, reset_row, settle_steps=50)
    raw = unwrap_simpler_env(env)
    source_initial = [float(value) for value in raw.episode_source_obj.pose.p]
    reference_initial = [float(value) for value in raw.episode_target_obj.pose.p]
    endpoint = [
        float(value)
        for value in plan_row["endpoint_reference_scene_xy_m"]
    ]
    maximum_tracking_error = 0.0
    minimum_source_reference_distance = float("inf")
    forbidden: list[dict[str, Any]] = []
    for index in range(path_steps + 1):
        progress = minimum_jerk_fraction(index / path_steps)
        command = [
            reference_initial[0]
            + progress * (endpoint[0] - reference_initial[0]),
            reference_initial[1]
            + progress * (endpoint[1] - reference_initial[1]),
        ]
        set_reference_xy(env, command)
        raw._scene.step()
        observed_reference = [
            float(value) for value in raw.episode_target_obj.pose.p
        ]
        observed_source = [float(value) for value in raw.episode_source_obj.pose.p]
        maximum_tracking_error = max(
            maximum_tracking_error,
            math.hypot(
                observed_reference[0] - command[0],
                observed_reference[1] - command[1],
            ),
        )
        minimum_source_reference_distance = min(
            minimum_source_reference_distance,
            math.hypot(
                observed_reference[0] - observed_source[0],
                observed_reference[1] - observed_source[1],
            ),
        )
        forbidden.extend(_forbidden_contacts(active_contact_pairs(env)))
    source_final = [float(value) for value in raw.episode_source_obj.pose.p]
    reference_final = [float(value) for value in raw.episode_target_obj.pose.p]
    source_drift = _distance(source_initial, source_final)
    endpoint_error = math.hypot(
        reference_final[0] - endpoint[0],
        reference_final[1] - endpoint[1],
    )
    passed = (
        maximum_tracking_error <= 0.002
        and endpoint_error <= 0.002
        and source_drift <= 0.005
        and minimum_source_reference_distance >= 0.035
        and not forbidden
    )
    return {
        "environment_seed": env_seed,
        "relation": plan_row["relation"],
        "physical_translation_sign": plan_row["physical_translation_sign"],
        "path_steps": path_steps,
        "initial_source_scene_m": source_initial,
        "final_source_scene_m": source_final,
        "initial_reference_scene_m": reference_initial,
        "commanded_endpoint_reference_scene_xy_m": endpoint,
        "final_reference_scene_m": reference_final,
        "maximum_reference_tracking_error_m": maximum_tracking_error,
        "endpoint_reference_error_m": endpoint_error,
        "stationary_source_drift_m": source_drift,
        "minimum_source_reference_center_distance_m": (
            minimum_source_reference_distance
        ),
        "forbidden_contacts": forbidden,
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def run_path_gate(
    *,
    registry_path: Path,
    plan_path: Path,
    integration_root: Path,
    path_steps: int,
    scale: float | None = None,
) -> dict[str, Any]:
    if path_steps < 25:
        raise SecondStackG3PathError("path_steps must sample at least 50 Hz")
    registry = load_json(registry_path)
    plan = load_json(plan_path)
    if (
        registry.get("fixture_id") != "second_stack"
        or plan.get("fixture_id") != "second_stack"
        or plan.get("reset_registry", {}).get("sha256")
        != sha256_file(registry_path)
    ):
        raise SecondStackG3PathError("C8 registry and G3 plan binding differs")
    analytical_scale = plan.get("selected_analytical_scale")
    selected_scale = analytical_scale if scale is None else float(scale)
    if float(selected_scale) > float(analytical_scale):
        raise SecondStackG3PathError(
            "C8 path scale cannot exceed the analytical selection"
        )
    selected_rows = [
        row
        for row in plan.get("scales", [])
        if row.get("scale") == selected_scale
    ]
    if len(selected_rows) != 1 or selected_rows[0].get("passed") is not True:
        raise SecondStackG3PathError("C8 analytical scale is not passing")
    checks_by_key = {
        (int(row["environment_seed"]), str(row["relation"])): row
        for row in selected_rows[0]["checks"]
    }
    if len(checks_by_key) != 256:
        raise SecondStackG3PathError("C8 analytical plan lacks complete checks")
    stack_receipt = verify_external_stack(
        integration_root=integration_root,
        registry=registry,
    )
    sys.path.insert(0, str(integration_root))
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

    register_simpler_envs()
    import gymnasium as gym

    resets = registry["resets_by_env_seed"]
    env = gym.make(ENV_NAME)
    records: list[dict[str, Any]] = []
    try:
        env.reset(seed=min(map(int, resets)))
        ensure_registered_support(env)
        for env_seed in sorted(map(int, resets)):
            for relation in ("left", "right", "front", "behind"):
                records.append(
                    _run_check(
                        env=env,
                        env_seed=env_seed,
                        reset_row=resets[str(env_seed)],
                        plan_row=checks_by_key[(env_seed, relation)],
                        path_steps=path_steps,
                    )
                )
    finally:
        env.close()
    passed = len(records) == 256 and all(row["passed"] for row in records)
    return {
        "schema_version": "v4-second-stack-g3-path-aggregate-v1",
        "campaign_id": "online_correction_v4",
        "family_ids": ["C8"],
        "fixture_id": "second_stack",
        "gate": "G3",
        "qualification_scope": "model_blind_no_policy",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "scale": selected_scale,
        "analytical_selected_scale": analytical_scale,
        "scale_selection_mode": (
            "analytical_selection"
            if scale is None
            else "controller_feasibility_fallback_candidate"
        ),
        "displacement_m": selected_rows[0]["displacement_m"],
        "expected_check_count": 256,
        "observed_check_count": len(records),
        "path_duration_s": 0.5,
        "path_steps": path_steps,
        "stack_receipt": stack_receipt,
        "reset_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "geometry_plan": {
            "path": str(plan_path),
            "sha256": sha256_file(plan_path),
        },
        "records": records,
        "release_boundary": (
            "Passing C8 live swept paths complete only the path portion of G3. "
            "Privileged grasp/transport/release checks remain required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--path-steps", type=int, default=250)
    parser.add_argument("--scale", type=float)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite C8 G3 path receipt: {args.output}")
    payload = run_path_gate(
        registry_path=args.registry.resolve(),
        plan_path=args.plan.resolve(),
        integration_root=args.integration_root.resolve(),
        path_steps=args.path_steps,
        scale=args.scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": sha256_file(args.output),
                "status": payload["status"],
                "observed_check_count": payload["observed_check_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
