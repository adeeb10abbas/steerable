#!/usr/bin/env python3
"""Run model-blind reset, camera, and frame qualification for C8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.online_correction_v4.second_stack import (  # noqa: E402
    ENV_NAME,
    REFERENCE_OBJECT,
    RELATION_AXES_SCENE_XY,
    SOURCE_OBJECT,
    active_contact_pairs,
    apply_registered_reset,
    task_axes_from_camera_extrinsic,
    unwrap_simpler_env,
)


class SecondStackG2Error(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
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
        raise SecondStackG2Error(f"{path} must contain a JSON object")
    return value


def _git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def verify_external_stack(
    *,
    integration_root: Path,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    identity = registry.get("external_stack_identity")
    if not isinstance(identity, Mapping):
        raise SecondStackG2Error("reset registry lacks external_stack_identity")
    simpler_root = integration_root / "external_dependencies/SimplerEnv"
    maniskill_root = simpler_root / "ManiSkill2_real2sim"
    observed = {
        "gr00t_commit": _git_commit(integration_root),
        "simpler_env_commit": _git_commit(simpler_root),
        "maniskill2_real2sim_commit": _git_commit(maniskill_root),
    }
    for key, actual in observed.items():
        if actual != identity.get(key):
            raise SecondStackG2Error(f"external stack {key} differs")
    fixture_files = identity.get("fixture_files")
    if not isinstance(fixture_files, Mapping):
        raise SecondStackG2Error("reset registry lacks fixture file identities")
    verified_files: dict[str, dict[str, Any]] = {}
    for label, record in fixture_files.items():
        if not isinstance(record, Mapping):
            raise SecondStackG2Error(f"fixture file {label} is invalid")
        relative = record.get("path")
        expected = record.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise SecondStackG2Error(f"fixture file {label} binding is invalid")
        path = simpler_root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise SecondStackG2Error(f"fixture file {label} digest differs")
        verified_files[str(label)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    return {**observed, "fixture_files": verified_files}


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _image_record(image: Any) -> dict[str, Any]:
    shape = [int(value) for value in image.shape]
    body = image.tobytes()
    return {
        "shape": shape,
        "dtype": str(image.dtype),
        "sha256": hashlib.sha256(body).hexdigest(),
        "minimum": int(image.min()),
        "maximum": int(image.max()),
    }


def _capture_processed_observation(env: Any) -> Mapping[str, Any]:
    outer = getattr(env, "unwrapped", env)
    inner = getattr(outer, "env", None)
    raw = unwrap_simpler_env(env)
    if inner is None or not hasattr(raw, "get_obs") or not hasattr(
        outer, "_process_observation"
    ):
        raise SecondStackG2Error("C8 observation bindings are unavailable")
    raw_obs = raw.get_obs()
    processed = outer._process_observation(raw_obs)
    if not isinstance(processed, Mapping):
        raise SecondStackG2Error("C8 processed observation is not a mapping")
    return processed


def _reset_once(
    *,
    env: Any,
    env_seed: int,
    row: Mapping[str, Any],
    settle_steps: int,
    drift_steps: int,
) -> dict[str, Any]:
    _observation, info = env.reset(seed=env_seed)
    if (
        info.get("episode_source_obj_name") != SOURCE_OBJECT
        or info.get("episode_target_obj_name") != REFERENCE_OBJECT
    ):
        raise SecondStackG2Error("C8 reset returned different object names")
    settled = apply_registered_reset(env, row, settle_steps=settle_steps)
    raw = unwrap_simpler_env(env)
    before_drift = {
        name: [float(value) for value in position]
        for name, position in settled.items()
    }
    for _ in range(drift_steps):
        raw._scene.step()
    raw._scene.update_render()
    after_drift = {
        SOURCE_OBJECT: [
            float(value) for value in raw.episode_source_obj.pose.p
        ],
        REFERENCE_OBJECT: [
            float(value) for value in raw.episode_target_obj.pose.p
        ],
    }
    requested = row["positions_scene_xy_m"]
    xy_errors = {
        name: math.hypot(
            after_drift[name][0] - float(requested[name][0]),
            after_drift[name][1] - float(requested[name][1]),
        )
        for name in (SOURCE_OBJECT, REFERENCE_OBJECT)
    }
    drift = {
        name: _distance(before_drift[name], after_drift[name])
        for name in (SOURCE_OBJECT, REFERENCE_OBJECT)
    }
    contacts = active_contact_pairs(env)
    forbidden_pair_contact = any(
        {record["actor0"], record["actor1"]}
        == {SOURCE_OBJECT, REFERENCE_OBJECT}
        for record in contacts
    )
    processed = _capture_processed_observation(env)
    image = processed.get("video.image_0")
    if image is None:
        raise SecondStackG2Error("C8 observation lacks video.image_0")
    image_record = _image_record(image)
    passed = (
        image_record["shape"] == [256, 256, 3]
        and image_record["minimum"] < image_record["maximum"]
        and max(xy_errors.values()) <= 0.002
        and max(drift.values()) <= 0.005
        and not forbidden_pair_contact
    )
    return {
        "passed": passed,
        "environment_seed": env_seed,
        "requested_scene_xy_m": requested,
        "settled_positions_scene_m": before_drift,
        "final_positions_scene_m": after_drift,
        "xy_error_m": xy_errors,
        "post_settle_drift_m": drift,
        "forbidden_source_reference_contact": forbidden_pair_contact,
        "active_contacts": contacts,
        "observation": image_record,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }


def run_g2(
    *,
    registry_path: Path,
    integration_root: Path,
    settle_steps: int,
    drift_steps: int,
    max_seeds: int | None,
) -> dict[str, Any]:
    registry = load_json(registry_path)
    if (
        registry.get("fixture_id") != "second_stack"
        or registry.get("status")
        != "model_blind_candidate_not_released_for_inference"
        or registry.get("environment_name") != ENV_NAME
    ):
        raise SecondStackG2Error("C8 reset registry is not an unreleased candidate")
    stack_receipt = verify_external_stack(
        integration_root=integration_root,
        registry=registry,
    )
    sys.path.insert(0, str(integration_root))
    from gr00t.eval.sim.SimplerEnv.simpler_env import register_simpler_envs

    register_simpler_envs()
    import gymnasium as gym

    resets = registry.get("resets_by_env_seed")
    if not isinstance(resets, Mapping):
        raise SecondStackG2Error("C8 reset registry lacks reset rows")
    seed_values = sorted(int(seed) for seed in resets)
    if max_seeds is not None:
        seed_values = seed_values[:max_seeds]
    env = gym.make(ENV_NAME)
    try:
        env.reset(seed=seed_values[0])
        raw = unwrap_simpler_env(env)
        camera = raw._cameras["3rd_view_camera"].camera
        extrinsic = camera.get_extrinsic_matrix().tolist()
        model = camera.get_model_matrix().tolist()
        intrinsic = camera.get_intrinsic_matrix().tolist()
        observed_axes = task_axes_from_camera_extrinsic(extrinsic)
        axis_error = max(
            abs(observed_axes[relation][axis] - expected)
            for relation, vector in RELATION_AXES_SCENE_XY.items()
            for axis, expected in enumerate(vector)
        )
        rows: list[dict[str, Any]] = []
        for env_seed in seed_values:
            row = resets[str(env_seed)]
            first = _reset_once(
                env=env,
                env_seed=env_seed,
                row=row,
                settle_steps=settle_steps,
                drift_steps=drift_steps,
            )
            repeated = _reset_once(
                env=env,
                env_seed=env_seed,
                row=row,
                settle_steps=settle_steps,
                drift_steps=drift_steps,
            )
            repeat_position_error = max(
                _distance(
                    first["final_positions_scene_m"][name],
                    repeated["final_positions_scene_m"][name],
                )
                for name in (SOURCE_OBJECT, REFERENCE_OBJECT)
            )
            rows.append(
                {
                    "environment_seed": env_seed,
                    "first": first,
                    "repeated": repeated,
                    "repeat_position_error_m": repeat_position_error,
                    "repeat_observation_exact": (
                        first["observation"]["sha256"]
                        == repeated["observation"]["sha256"]
                    ),
                    "passed": (
                        first["passed"]
                        and repeated["passed"]
                        and repeat_position_error <= 0.002
                    ),
                }
            )
    finally:
        env.close()
    expected_count = int(registry["registered_env_seed_count"])
    complete = len(rows) == expected_count
    passed = (
        complete
        and axis_error <= 1e-6
        and all(row["passed"] for row in rows)
    )
    return {
        "schema_version": "v4-second-stack-g2-aggregate-v1",
        "campaign_id": "online_correction_v4",
        "family_ids": ["C8"],
        "fixture_id": "second_stack",
        "gate": "G2",
        "qualification_scope": "model_blind_no_policy",
        "status": "passed" if passed else ("partial" if not complete else "failed"),
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "expected_seed_count": expected_count,
        "observed_seed_count": len(rows),
        "stack_receipt": stack_receipt,
        "reset_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
        },
        "camera_and_task_frame": {
            "camera": "3rd_view_camera",
            "model_matrix": model,
            "extrinsic_matrix": extrinsic,
            "intrinsic_matrix": intrinsic,
            "relation_axes_scene_xy": {
                key: list(value) for key, value in observed_axes.items()
            },
            "registered_axis_max_abs_error": axis_error,
        },
        "settle_steps": settle_steps,
        "drift_steps": drift_steps,
        "records": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--integration-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--settle-steps", type=int, default=250)
    parser.add_argument("--drift-steps", type=int, default=250)
    parser.add_argument("--max-seeds", type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite G2 receipt: {args.output}")
    payload = run_g2(
        registry_path=args.registry.resolve(),
        integration_root=args.integration_root.resolve(),
        settle_steps=args.settle_steps,
        drift_steps=args.drift_steps,
        max_seeds=args.max_seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(payload))
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": sha256_file(args.output),
                "status": payload["status"],
                "observed_seed_count": payload["observed_seed_count"],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
