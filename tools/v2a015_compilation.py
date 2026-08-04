#!/usr/bin/env python3
"""Shared, fail-closed compilation helpers for V2-A015.

The behavioral compilers deliberately consume hash-bearing manifests.  They
never discover a "best" run by globbing an output directory: a cell is in the
denominator only when a supplied manifest names its exact simulator, action,
video, and future evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_ID = "V2-A015"
SEEDS = (8300, 8301, 8302)
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
ACTION_CAP = 450
INTERACTION_THRESHOLD_M = 0.01
PICKUP_LIFT_M = 0.03
RELATION_COSINE = math.cos(math.radians(45.0))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    path = path.resolve()
    if path.exists() and not overwrite:
        raise RuntimeError(f"Refusing to overwrite compiled evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".staging")
    if temporary.exists():
        raise RuntimeError(f"Refusing to overwrite stale staging evidence: {temporary}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Missing JSON evidence: {path}")
    return json.loads(path.read_text())


def resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def file_record(path: Path, *, relative_to_repo: bool = False) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"Missing evidence file: {path}")
    display = str(path)
    if relative_to_repo:
        display = str(path.relative_to(REPO_ROOT))
    return {"path": display, "bytes": path.stat().st_size, "sha256": sha256(path)}


def validate_file_record(
    record: dict[str, Any], base: Path, label: str, *, require_bytes: bool = False
) -> Path:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise RuntimeError(f"{label} is not a file record")
    if not isinstance(record.get("sha256"), str) or len(record["sha256"]) != 64:
        raise RuntimeError(f"{label} lacks a SHA-256")
    path = resolve_path(record["path"], base)
    if not path.is_file():
        raise RuntimeError(f"Missing {label}: {path}")
    if "bytes" in record and int(record["bytes"]) != path.stat().st_size:
        raise RuntimeError(f"Byte mismatch for {label}: {path}")
    if require_bytes and "bytes" not in record:
        raise RuntimeError(f"{label} lacks a byte count")
    if sha256(path) != record["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch for {label}: {path}")
    return path


def validate_exact_file(
    path: Path, *, expected_sha256: str, label: str
) -> dict[str, Any]:
    path = path.resolve()
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise RuntimeError(
            f"{label} hash changed: expected {expected_sha256}, observed {record['sha256']}"
        )
    return record


def ledger_rows(
    path: Path,
    *,
    invalid: bool,
    expected_model_id: str | None = None,
    expected_arm_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, dict):
        declared_model = payload.get("model_id")
        declared_arm = payload.get("arm_id")
        if expected_model_id and declared_model not in (None, expected_model_id):
            raise RuntimeError(
                f"Ledger belongs to model {declared_model!r}, not {expected_model_id!r}: {path}"
            )
        if expected_arm_id and declared_arm not in (None, expected_arm_id):
            raise RuntimeError(
                f"Ledger belongs to arm {declared_arm!r}, not {expected_arm_id!r}: {path}"
            )
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = next(
            (
                payload[key]
                for key in (
                    "attempts",
                    "invalid_attempts",
                    "setup_invalid_attempts",
                    "events",
                    "interventions",
                    "rows",
                )
                if isinstance(payload.get(key), list)
            ),
            None,
        )
    else:
        rows = None
    if rows is None:
        raise RuntimeError(f"Cannot identify ledger rows: {path}")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"Ledger row {index} is not an object: {path}")
        if expected_model_id and row.get("model_id") not in (None, expected_model_id):
            raise RuntimeError(f"Ledger mixes model identities: {path}:{index}")
        if expected_arm_id and row.get("arm_id") not in (None, expected_arm_id):
            raise RuntimeError(f"Ledger mixes arm identities: {path}:{index}")
        if invalid:
            classification = str(row.get("classification", "")).lower()
            result = str(row.get("result", "")).lower()
            effect = str(row.get("effect", "")).lower()
            explicitly_invalid = row.get("behavioral_result_valid") is False
            named_invalid = any(
                token in classification
                for token in ("invalid", "partial", "infrastructure", "setup")
            )
            result_invalid = any(
                token in result for token in ("invalid", "partial", "infrastructure")
            )
            explicitly_excluded = (
                row.get("denominator_status") == "excluded"
                or "excluded" in effect
            )
            if not (
                explicitly_invalid
                or named_invalid
                or result_invalid
                or explicitly_excluded
            ):
                raise RuntimeError(
                    f"Invalid-attempt row is not explicitly excluded from behavior: {path}:{index}"
                )
        elif row.get("behavioral_result_valid", True) is not True:
            raise RuntimeError(
                f"Runtime-intervention row invalidates behavior and belongs in the invalid ledger: {path}:{index}"
            )
    return rows, file_record(path)


def ledger_summary(
    paths: Iterable[Path],
    *,
    invalid: bool,
    expected_model_id: str | None = None,
    expected_arm_id: str | None = None,
) -> dict[str, Any]:
    records, rows = [], []
    for path in paths:
        selected, record = ledger_rows(
            path.resolve(),
            invalid=invalid,
            expected_model_id=expected_model_id,
            expected_arm_id=expected_arm_id,
        )
        rows.extend(selected)
        records.append({**record, "row_count": len(selected)})
    classifications = Counter(
        str(
            row.get("classification")
            or row.get("result")
            or row.get("stage")
            or "unspecified"
        )
        for row in rows
    )
    return {
        "row_count": len(rows),
        "by_classification": dict(sorted(classifications.items())),
        "sources": records,
        "behavioral_denominator_policy": (
            "excluded" if invalid else "behavior retained; affected wall latency excluded"
        ),
    }


def rotation_wxyz(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    if np.any(norm <= 1e-12):
        raise RuntimeError("Robot root pose contains a zero quaternion")
    w, x, y, z = (value / norm).T
    matrix = np.empty((len(value), 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix


def robot_frame_delta(cube: np.ndarray, bowl: np.ndarray, robot: np.ndarray) -> np.ndarray:
    return np.einsum(
        "tij,ti->tj",
        rotation_wxyz(robot[:, 3:7]),
        np.asarray(cube[:, :3] - bowl[:, :3], dtype=np.float64),
    )


def relation_mask(delta: np.ndarray, relation: str) -> np.ndarray:
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    sign = 1.0 if relation == "left" else -1.0
    cosine = np.divide(
        sign * delta[:, 1],
        horizontal,
        out=np.zeros_like(horizontal),
        where=horizontal > 1e-8,
    )
    return cosine >= RELATION_COSINE


def first_consecutive(mask: np.ndarray, count: int = 3) -> int | None:
    if len(mask) < count:
        return None
    hits = np.convolve(mask.astype(np.int8), np.ones(count, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(hits == count)
    return int(indices[0]) if len(indices) else None


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if len(indices) else None


def initial_fingerprint(group: Any) -> str:
    import h5py

    arrays: dict[str, np.ndarray] = {}

    def collect(name: str, item: Any) -> None:
        if isinstance(item, h5py.Dataset) and name.startswith(("articulation/", "rigid_object/")):
            arrays[name] = np.asarray(item)

    group.visititems(collect)
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def trajectory_quality(cube: np.ndarray, delta: np.ndarray, relation: str) -> dict[str, Any]:
    cube_xyz = np.asarray(cube[:, :3], dtype=np.float64)
    increments = np.linalg.norm(np.diff(cube_xyz, axis=0), axis=1)
    path_length = float(increments.sum())
    net = float(np.linalg.norm(cube_xyz[-1] - cube_xyz[0]))
    displacement = np.linalg.norm(cube_xyz - cube_xyz[0], axis=1)
    sign = 1.0 if relation == "left" else -1.0
    oriented_lateral = sign * np.asarray(delta[:, 1], dtype=np.float64)
    oriented_lateral_change = oriented_lateral - oriented_lateral[0]
    ratio = path_length / net if net >= INTERACTION_THRESHOLD_M else None
    return {
        "cube_path_length_3d_m": path_length,
        "cube_net_displacement_3d_m": net,
        "cube_excess_path_ratio": ratio,
        "cube_excess_path_ratio_null_reason": (
            None
            if ratio is not None
            else "net displacement below the frozen 0.01 m interaction threshold"
        ),
        "cube_max_excursion_from_initial_3d_m": float(displacement.max()),
        "cube_max_requested_lateral_excursion_from_initial_m": float(
            oriented_lateral_change.max()
        ),
        "cube_max_opposite_lateral_excursion_from_initial_m": float(
            (-oriented_lateral_change).max()
        ),
    }


def action_quality(actions: np.ndarray) -> dict[str, Any]:
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or len(values) == 0:
        raise RuntimeError(f"Executed action trace must have shape [T,8], got {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("Executed action trace contains a non-finite value")
    transition_norm = np.linalg.norm(np.diff(values[:, :7], axis=0), axis=1)
    gripper = values[:, 7]
    binary = np.isclose(gripper, 0.0, atol=1e-6) | np.isclose(gripper, 1.0, atol=1e-6)
    if not bool(np.all(binary)):
        raise RuntimeError("Executed gripper trace is not the frozen binary 0/1 command")
    switches = int(np.count_nonzero((gripper[1:] > 0.5) != (gripper[:-1] > 0.5)))
    return {
        "joint_action_total_variation_l2": float(transition_norm.sum()),
        "joint_action_mean_l2_per_transition": (
            float(transition_norm.mean()) if len(transition_norm) else 0.0
        ),
        "joint_action_transition_count": int(len(transition_norm)),
        "gripper_switch_count": switches,
        "gripper_switch_definition": "successive changes in the executed binary 0/1 gripper command",
    }


def first_chunk_pair_metrics(
    left: np.ndarray, right: np.ndarray, *, horizon: int
) -> dict[str, Any]:
    count = min(len(left), len(right), horizon)
    if count <= 0:
        raise RuntimeError("Cannot compare empty LEFT/RIGHT action traces")
    left_slice = np.asarray(left[:count], dtype=np.float64)
    right_slice = np.asarray(right[:count], dtype=np.float64)
    delta = left_slice - right_slice
    gripper_disagreement = (left_slice[:, 7] > 0.5) != (right_slice[:, 7] > 0.5)
    return {
        "executed_steps_compared": count,
        "requested_horizon": horizon,
        "joint_only_rms_7d": float(np.sqrt(np.mean(np.square(delta[:, :7])))),
        "legacy_all_8d_rms": float(np.sqrt(np.mean(np.square(delta)))),
        "gripper_disagreement_steps": int(np.count_nonzero(gripper_disagreement)),
        "gripper_disagreement_fraction": float(np.mean(gripper_disagreement)),
    }


def load_simulator_cell(
    cell: dict[str, Any], manifest_base: Path
) -> tuple[dict[str, Any], np.ndarray]:
    import h5py

    seed = int(cell["environment_seed"])
    relation = cell["requested_relation"]
    artifacts = cell.get("simulator_artifacts")
    if not isinstance(artifacts, dict):
        raise RuntimeError(f"Cell {seed}/{relation} lacks explicit simulator_artifacts")
    required = {"environment_config", "rollout_hdf5", "episode_log", "viewport_video"}
    if set(artifacts) != required:
        raise RuntimeError(
            f"Cell {seed}/{relation} simulator_artifacts must be exactly {sorted(required)}"
        )
    paths = {
        key: validate_file_record(value, manifest_base, f"{seed}/{relation} {key}", require_bytes=True)
        for key, value in artifacts.items()
    }
    task_dir = resolve_path(cell["simulator_task_dir"], manifest_base)
    if task_dir.name != TASKS[relation]:
        raise RuntimeError(f"Task/relation mismatch for {seed}/{relation}: {task_dir}")
    if any(path.parent != task_dir for path in paths.values()):
        raise RuntimeError(f"Simulator file record escapes the declared task directory: {seed}/{relation}")

    env = load_json(paths["environment_config"])
    log = load_json(paths["episode_log"])
    if env.get("instruction") != PROMPTS[relation] or int(env.get("seed", -1)) != seed:
        raise RuntimeError(f"Static prompt/environment seed mismatch: {seed}/{relation}")
    with h5py.File(paths["rollout_hdf5"], "r") as handle:
        demo = handle["data/demo_0"]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        cube = np.asarray(demo["states/rigid_object/rubiks_cube/root_pose"], dtype=np.float64)
        bowl = np.asarray(demo["states/rigid_object/bowl/root_pose"], dtype=np.float64)
        robot = np.asarray(demo["states/articulation/robot/root_pose"], dtype=np.float64)
        fingerprint = initial_fingerprint(demo["initial_state"])
    steps = int(log["final_step"])
    if not (len(actions) == len(cube) == len(bowl) == len(robot) == steps):
        raise RuntimeError(f"Simulator trajectory length mismatch: {seed}/{relation}")
    success = bool(log["success"])
    if success and not 0 < steps <= ACTION_CAP:
        raise RuntimeError(f"Invalid successful completion step: {seed}/{relation}/{steps}")
    if not success and steps != ACTION_CAP:
        raise RuntimeError(
            f"A valid failure must be right-censored at the {ACTION_CAP}-action cap: {seed}/{relation}/{steps}"
        )

    delta = robot_frame_delta(cube, bowl, robot)
    requested = relation_mask(delta, relation)
    opposite_relation = "right" if relation == "left" else "left"
    opposite = relation_mask(delta, opposite_relation)
    lift = cube[:, 2] - cube[0, 2]
    movement = np.linalg.norm(cube[:, :3] - cube[0, :3], axis=1)
    pickup = first_consecutive(lift >= PICKUP_LIFT_M)
    interaction = first_consecutive(movement >= INTERACTION_THRESHOLD_M)
    entered = first_true(requested)
    failure_stage = (
        "success"
        if success
        else "no_object_interaction"
        if interaction is None
        else "object_moved_no_verified_pickup"
        if pickup is None
        else "picked_never_entered_requested_region"
        if entered is None
        else "entered_requested_region_not_released"
    )
    final_display = float(-delta[-1, 1])
    requested_margin = -final_display if relation == "left" else final_display
    episode = {
        "environment_seed": seed,
        "sampling_seed": int(cell["sampling_seed"]),
        "pair_id": f"droid_pair_seed_{seed}",
        "requested_relation": relation,
        "prompt": PROMPTS[relation],
        "prompt_family": "direct_command",
        "prompt_controller": "episode_static",
        "oracle_actions": 0,
        "dynamic_prompt_switches": 0,
        "requested_success": success,
        "actions_executed": steps,
        "completion_actions_observed": steps if success else None,
        "completion_action_status": (
            "observed_success_event" if success else f"right_censored_at_{ACTION_CAP}_action_cap"
        ),
        "failure_stage": failure_stage,
        "verified_pickup_proxy": pickup is not None,
        "first_verified_pickup_proxy_step": pickup,
        "object_interaction_proxy": interaction is not None,
        "first_object_interaction_proxy_step": interaction,
        "ever_entered_requested_region": bool(np.any(requested)),
        "first_requested_region_step": entered,
        "final_requested_relation": bool(requested[-1]),
        "ever_entered_opposite_region": bool(np.any(opposite)),
        "final_opposite_relation": bool(opposite[-1]),
        "ever_released_in_requested_region": success,
        "initial_lateral_display_m": float(-delta[0, 1]),
        "final_lateral_display_m": final_display,
        "requested_signed_final_margin_m": requested_margin,
        "requested_margin_definition": (
            "-final_lateral_display_m for LEFT; +final_lateral_display_m for RIGHT"
        ),
        "max_object_lift_m": float(lift.max()),
        "physical_initial_state_sha256": fingerprint,
        "trajectory_quality": trajectory_quality(cube, delta, relation),
        "action_quality": action_quality(actions),
        "simulator_artifacts": {key: file_record(path) for key, path in paths.items()},
    }
    return episode, actions


def validate_cell_protocol(cell: dict[str, Any]) -> tuple[int, str]:
    seed = int(cell.get("environment_seed", -1))
    relation = cell.get("requested_relation")
    if seed not in SEEDS or relation not in PROMPTS:
        raise RuntimeError(f"Unauthorized cell: seed={seed}, relation={relation!r}")
    expected = {
        "sampling_seed": seed,
        "prompt": PROMPTS[relation],
        "prompt_family": "direct_command",
        "prompt_controller": "episode_static",
        "oracle_actions": 0,
        "dynamic_prompt_switches": 0,
    }
    for key, value in expected.items():
        if cell.get(key) != value:
            raise RuntimeError(
                f"Cell protocol mismatch {seed}/{relation} for {key}: expected={value!r}, observed={cell.get(key)!r}"
            )
    return seed, relation


def build_pairs(
    episodes: list[dict[str, Any]], actions: dict[tuple[int, str], np.ndarray], *, horizon: int
) -> list[dict[str, Any]]:
    pairs = []
    for seed in SEEDS:
        left = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "left")
        right = next(row for row in episodes if row["environment_seed"] == seed and row["requested_relation"] == "right")
        if left["physical_initial_state_sha256"] != right["physical_initial_state_sha256"]:
            raise RuntimeError(f"Matched physical initial state differs for seed {seed}")
        left_margin = float(left["requested_signed_final_margin_m"])
        right_margin = float(right["requested_signed_final_margin_m"])
        left_endpoint = float(left["final_lateral_display_m"])
        right_endpoint = float(right["final_lateral_display_m"])
        endpoint_separation = right_endpoint - left_endpoint
        pair_actions = first_chunk_pair_metrics(
            actions[(seed, "left")], actions[(seed, "right")], horizon=horizon
        )
        pairs.append(
            {
                "pair_id": f"droid_pair_seed_{seed}",
                "environment_seed": seed,
                "left_prompt": PROMPTS["left"],
                "right_prompt": PROMPTS["right"],
                "left_requested_success": left["requested_success"],
                "right_requested_success": right["requested_success"],
                "left_requested_margin_m": left_margin,
                "right_requested_margin_m": right_margin,
                "seed_balance_gap_right_minus_left_margin_m": right_margin - left_margin,
                "seed_absolute_direction_imbalance_m": abs(right_margin - left_margin),
                "seed_weaker_side_margin_m": min(left_margin, right_margin),
                "left_final_lateral_display_m": left_endpoint,
                "right_final_lateral_display_m": right_endpoint,
                "endpoint_separation_right_minus_left_m": endpoint_separation,
                "endpoint_ordering": (
                    "aligned" if endpoint_separation > 0 else "anti_aligned" if endpoint_separation < 0 else "tie"
                ),
                "first_chunk_prompt_response": pair_actions,
                "executed_actions_distinct": bool(
                    not np.array_equal(actions[(seed, "left")], actions[(seed, "right")])
                ),
                "physical_initial_state_sha256": left["physical_initial_state_sha256"],
            }
        )
    return pairs


def configuration_summary(
    episodes: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    by_direction: dict[str, Any] = {}
    for relation in PROMPTS:
        rows = [row for row in episodes if row["requested_relation"] == relation]
        margins = [float(row["requested_signed_final_margin_m"]) for row in rows]
        by_direction[relation] = {
            "prompt": PROMPTS[relation],
            "episodes": len(rows),
            "successes": sum(bool(row["requested_success"]) for row in rows),
            "verified_pickups": sum(bool(row["verified_pickup_proxy"]) for row in rows),
            "entered_requested_region": sum(bool(row["ever_entered_requested_region"]) for row in rows),
            "requested_margin_m": {
                "values_by_seed": [
                    {
                        "environment_seed": row["environment_seed"],
                        "value": row["requested_signed_final_margin_m"],
                    }
                    for row in sorted(rows, key=lambda value: value["environment_seed"])
                ],
                "mean": float(np.mean(margins)),
                "median": float(np.median(margins)),
                "minimum": min(margins),
                "maximum": max(margins),
            },
            "mean_cube_path_length_3d_m": float(
                np.mean([row["trajectory_quality"]["cube_path_length_3d_m"] for row in rows])
            ),
            "mean_joint_action_total_variation_l2": float(
                np.mean([row["action_quality"]["joint_action_total_variation_l2"] for row in rows])
            ),
        }
    left_successes = by_direction["left"]["successes"]
    right_successes = by_direction["right"]["successes"]
    left_margin = by_direction["left"]["requested_margin_m"]["mean"]
    right_margin = by_direction["right"]["requested_margin_m"]["mean"]
    gap = right_margin - left_margin
    return {
        "valid_episode_count": len(episodes),
        "valid_failure_count": sum(not row["requested_success"] for row in episodes),
        "requested_success_count": sum(row["requested_success"] for row in episodes),
        "by_direction": by_direction,
        "failure_stage_counts": dict(sorted(Counter(row["failure_stage"] for row in episodes).items())),
        "success_direction_gap_right_minus_left_count": right_successes - left_successes,
        "success_absolute_direction_imbalance_count": abs(right_successes - left_successes),
        "success_weaker_direction_count": min(left_successes, right_successes),
        "mean_margin_balance": {
            "right_minus_left_m": gap,
            "absolute_direction_imbalance_m": abs(gap),
            "weaker_direction_mean_margin_m": min(left_margin, right_margin),
        },
        "aligned_endpoint_pair_count": sum(pair["endpoint_ordering"] == "aligned" for pair in pairs),
        "distinct_executed_action_pair_count": sum(pair["executed_actions_distinct"] for pair in pairs),
    }


def validate_complete_grid(cells: list[dict[str, Any]]) -> None:
    expected = {(seed, relation) for seed in SEEDS for relation in PROMPTS}
    observed = [validate_cell_protocol(cell) for cell in cells]
    if len(observed) != 6 or set(observed) != expected or len(set(observed)) != 6:
        raise RuntimeError(f"Expected one exact six-cell V2-A015 grid, observed {observed}")


def metric_definitions() -> dict[str, Any]:
    return {
        "requested_signed_final_margin_m": (
            "-final_lateral_display_m for LEFT and +final_lateral_display_m for RIGHT; positive is farther into the requested side"
        ),
        "seed_balance_gap": "RIGHT requested margin minus LEFT requested margin for the matched seed",
        "seed_absolute_direction_imbalance": "absolute value of the matched seed balance gap",
        "seed_weaker_side_margin": "minimum of the matched LEFT and RIGHT requested margins",
        "endpoint_separation": "RIGHT final lateral display coordinate minus LEFT final lateral display coordinate",
        "cube_path_length_3d_m": "sum of successive Euclidean displacements of the cube root pose in world XYZ",
        "cube_excess_path_ratio": (
            "cube 3D path length divided by start-to-end 3D displacement; null when net displacement is below 0.01 m"
        ),
        "cube_max_excursion": (
            "maximum 3D distance from the initial cube pose, with maximum requested- and opposite-oriented lateral displacement from the initial lateral coordinate reported separately"
        ),
        "joint_action_total_variation_l2": (
            "sum of successive Euclidean L2 changes over the seven executed joint-position dimensions"
        ),
        "gripper_switch_count": "successive changes in the separate executed binary gripper command",
        "first_chunk_prompt_response": (
            "matched LEFT/RIGHT RMS over the first 32 executed steps for Nano or first 8 for DreamZero; joint-only 7D and legacy all-8D values plus gripper disagreement"
        ),
        "completion_actions": (
            f"reported only for observed successes; failures are right-censored at the {ACTION_CAP}-action cap and are never treated as completion times"
        ),
    }
