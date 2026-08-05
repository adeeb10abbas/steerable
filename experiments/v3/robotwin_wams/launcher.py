#!/usr/bin/env python3
"""Launch or compile one exact v3 RoboTwin WAM matched pair.

No broad queue mode is provided intentionally.  One invocation resolves one
registered model/pair/replicate, revalidates its runtime identity, and runs the
existing hash-pinned v2 entrypoint once.  That entrypoint resets the identical
scene and policy seed for LEFT and RIGHT and changes only the static prompt and
requested relation checker.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


STUDY_ROOT_DEFAULT = Path(__file__).resolve().parents[3]
if str(STUDY_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT_DEFAULT))

from experiments.v3.robotwin_wams.contract import (  # noqa: E402
    BEHAVIORAL_ARENA,
    MODEL_SPECS,
    STUDY_ID,
    AdapterError,
    AuthorizedPair,
    canonical_sha256,
    load_authorized_pair,
    prompt_for,
    sha256_file,
    transform_xyz,
    verify_runtime_identity,
)
from tools.vla_wam_v3_episode_schema import (  # noqa: E402
    BEHAVIORAL_SCHEMA_VERSION,
    INFRASTRUCTURE_SCHEMA_VERSION,
    MEASUREMENT_FRAME_DESCRIPTION,
    MEASUREMENT_FRAME_ID,
    derive_failure_taxonomy,
    derive_initial_state_sha256,
    derive_measurements,
    validate_behavioral_record,
    validate_infrastructure_record,
    write_jsonl,
)


ACTION_CAP = 400
PICKUP_LIFT_M = 0.03
SUSTAINED_SAMPLES = 3


def _fail(message: str) -> None:
    raise AdapterError(message)


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{path} must contain one JSON object")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        _fail(f"required raw artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _condition_dir(native_root: Path, pair: AuthorizedPair, relation: str) -> Path:
    return (
        Path(native_root)
        / pair.anchor_task
        / f"environment_seed_{pair.environment_seed}"
        / f"sampling_seed_{pair.policy_seed}"
        / f"direct_command__{relation}"
    )


def build_native_command(
    pair: AuthorizedPair,
    *,
    study_root: Path,
    external_repository: Path,
    native_output_dir: Path,
) -> list[str]:
    """Return the existing exact v2 entrypoint for one v3 r1..r9 pair."""

    if pair.replicate == 0:
        _fail("replicate r0 is preserved evidence and has no native command")
    spec = MODEL_SPECS[pair.model_id]
    wrapper = (Path(external_repository).resolve() / spec["wrapper_path"]).resolve()
    shared = [
        "--output-dir",
        str(Path(native_output_dir).resolve()),
    ]
    protocol = str((Path(study_root).resolve() / "artifacts/vla_wam_shared_v2/protocol.json"))
    if pair.model_id == "efficient_wam_rt_robotwin":
        return [
            str(wrapper),
            *shared,
            "--task",
            pair.anchor_task,
            "--seed",
            str(pair.environment_seed),
            "--sampling-seed",
            str(pair.policy_seed),
            "--task-config",
            "demo_clean",
            "--prompt-family",
            "direct_command",
            "--study-protocol",
            protocol,
            "--max-actions",
            str(ACTION_CAP),
            "--save-simulator-video",
            "--predicted-video-max-chunks",
            "1",
        ]
    if pair.model_id == "fastwam_robotwin":
        return [
            str(wrapper),
            *shared,
            "--cell",
            f"{pair.anchor_task}:{pair.environment_seed}:{pair.policy_seed}",
            "--task-config",
            "demo_clean",
            "--prompt-family",
            "direct_command",
            "--study-protocol",
            protocol,
            "--max-actions",
            str(ACTION_CAP),
            "--action-horizon",
            "32",
            "--replan-steps",
            "24",
            "--num-inference-steps",
            "10",
            "--text-cfg-scale",
            "2.0",
            "--save-simulator-video",
        ]
    if pair.model_id == "lingbot_va_robotwin":
        return [
            str(wrapper),
            *shared,
            "--task",
            pair.anchor_task,
            "--environment-seed",
            str(pair.environment_seed),
            "--sampling-seed",
            str(pair.policy_seed),
            "--prompt-family",
            "direct_command",
            "--study-protocol",
            protocol,
            "--condition",
            "correct",
            "--condition",
            "swapped",
            "--max-actions",
            str(ACTION_CAP),
            "--guidance-scale",
            "5.0",
            "--action-guidance-scale",
            "1.0",
            "--save-simulator-video",
            "--save-first-predicted-latent",
        ]
    _fail(f"unsupported model {pair.model_id}")


def build_guard_command(
    pair: AuthorizedPair,
    *,
    study_root: Path,
    attempt_dir: Path,
    gpu_index: int,
    native_command: list[str],
) -> list[str]:
    """Wrap the pair in the model-specific native process-group guard."""

    attempt = Path(attempt_dir).resolve()
    model = pair.model_id
    return [
        sys.executable,
        str(Path(study_root).resolve() / "tools/native_process_group_thermal_guard.py"),
        "--launch",
        "--gpu-index",
        str(gpu_index),
        "--output",
        str(attempt / "thermal_events.jsonl"),
        "--ledger-output",
        str(attempt / f"runtime_interventions_{model}.json"),
        "--invalid-attempts-output",
        str(attempt / f"invalid_attempts_{model}.json"),
        "--model-id",
        model,
        "--pair-id",
        pair.pair_id,
        "--environment-seed",
        str(pair.environment_seed),
        "--sampling-seed",
        str(pair.policy_seed),
        "--requested-relation",
        "left",
        "--requested-relation",
        "right",
        "--",
        *native_command,
    ]


def _native_region(step: dict[str, Any], relation: str) -> bool:
    try:
        object_xyz = [float(value) for value in step["object_xyz"]]
        reference_xyz = [float(value) for value in step["target_xyz"]]
    except (KeyError, TypeError, ValueError) as error:
        raise AdapterError(f"malformed native RoboTwin XYZ state: {error}") from error
    if len(object_xyz) != 3 or len(reference_xyz) != 3:
        _fail("native object/target XYZ must have three components")
    delta_x = object_xyz[0] - reference_xyz[0]
    delta_y = object_xyz[1] - reference_xyz[1]
    distance_xy = math.hypot(delta_x, delta_y)
    side = delta_x < 0.0 if relation == "left" else delta_x > 0.0
    return bool(0.08 < distance_xy < 0.2 and side and abs(delta_y) < 0.05)


def _first_sustained(mask: list[bool], count: int = SUSTAINED_SAMPLES) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _legacy_failure_stage(trajectory: list[dict[str, Any]], success: bool) -> str:
    initial_z = float(trajectory[0]["object_xyz"][2])
    pickup = _first_sustained(
        [
            float(step["object_xyz"][2]) - initial_z >= PICKUP_LIFT_M
            and not bool(step["grippers_open"])
            for step in trajectory
        ]
    ) is not None
    entered = any(bool(step["relation_region"]) for step in trajectory)
    closed = any(not bool(step["grippers_open"]) for step in trajectory)
    if success:
        return "success"
    if entered:
        return "entered_requested_region_without_verified_completion"
    if pickup:
        return "picked_never_entered_requested_region"
    if closed:
        return "closed_gripper_no_verified_pickup"
    return "no_verified_interaction"


def _event_timeline(
    *,
    requested: list[bool],
    opposite: list[bool],
    transformed_object_xyz: list[list[float]],
    actions_executed: int,
) -> list[dict[str, Any]]:
    baseline_z = transformed_object_xyz[0][2]
    pickup = _first_sustained(
        [point[2] - baseline_z >= PICKUP_LIFT_M for point in transformed_object_xyz]
    )
    requested_entry = next((i for i, value in enumerate(requested) if value), None)
    opposite_entry = next((i for i, value in enumerate(opposite) if value), None)
    middle = [
        ("verified_pickup", pickup),
        ("requested_region_entry", requested_entry),
        ("opposite_region_entry", opposite_entry),
    ]
    order = {"verified_pickup": 0, "requested_region_entry": 1, "opposite_region_entry": 2}
    retained = sorted(
        [(name, step) for name, step in middle if step is not None],
        key=lambda item: (int(item[1]), order[item[0]]),
    )
    return [
        {"event": "episode_start", "action_step": 0},
        *[
            {"event": name, "action_step": int(step)}
            for name, step in retained
        ],
        {"event": "episode_end", "action_step": actions_executed},
    ]


def _validate_action_trace(path: Path, result: dict[str, Any], actions: int) -> dict[str, Any]:
    declared = result.get("action_trace")
    if not isinstance(declared, dict):
        _fail("native result lacks action_trace provenance")
    resolved = Path(path).resolve()
    if Path(str(declared.get("path", ""))).expanduser().resolve() != resolved:
        _fail("native action_trace path does not match the exact condition directory")
    record = _file_record(resolved)
    if declared.get("sha256") != record["sha256"]:
        _fail("native action trace hash claim is false")
    if declared.get("count") != actions:
        _fail("native action trace count does not match actions_executed")
    shape = declared.get("shape")
    if not isinstance(shape, list) or not shape or shape[0] != actions:
        _fail("native action trace shape does not match actions_executed")
    observed_shape = _npy_shape_from_npz(resolved, "executed")
    if len(observed_shape) != 2 or observed_shape[0] != actions:
        _fail("executed action array shape does not match the native record")
    if list(observed_shape) != shape:
        _fail("executed action array shape disagrees with native provenance")
    return record


def _npy_shape_from_npz(path: Path, array_name: str) -> tuple[int, ...]:
    """Read one NPY header from NPZ without importing a model-stack NumPy.

    The launcher commonly runs from the compact study environment while NumPy
    lives only inside each incompatible model venv.  Parsing the documented NPY
    header is enough to verify the executed array name, rank, shape, dtype, and
    uncompressed payload length without importing either stack.
    """

    member = f"{array_name}.npy"
    try:
        with zipfile.ZipFile(path) as archive:
            if member not in archive.namelist():
                _fail(f"action_trace.npz lacks the {array_name} array")
            payload = archive.read(member)
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise AdapterError(f"cannot inspect action trace archive {path}: {error}") from error
    if len(payload) < 10 or payload[:6] != b"\x93NUMPY":
        _fail("executed action array has an invalid NPY header")
    major, minor = payload[6], payload[7]
    if (major, minor) == (1, 0):
        header_length = struct.unpack("<H", payload[8:10])[0]
        header_start = 10
    elif major in {2, 3}:
        if len(payload) < 12:
            _fail("executed action array has a truncated NPY header")
        header_length = struct.unpack("<I", payload[8:12])[0]
        header_start = 12
    else:
        _fail(f"unsupported NPY header version {major}.{minor}")
    header_end = header_start + header_length
    if header_end > len(payload):
        _fail("executed action array has a truncated NPY dictionary")
    try:
        header = ast.literal_eval(payload[header_start:header_end].decode("latin1").strip())
    except (SyntaxError, ValueError, UnicodeDecodeError) as error:
        raise AdapterError(f"invalid NPY header dictionary: {error}") from error
    if not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}:
        _fail("executed action NPY header fields are not canonical")
    shape = header["shape"]
    if (
        not isinstance(shape, tuple)
        or not shape
        or any(type(dimension) is not int or dimension < 0 for dimension in shape)
    ):
        _fail("executed action NPY shape is invalid")
    if type(header["fortran_order"]) is not bool:
        _fail("executed action NPY fortran_order is invalid")
    dtype = header["descr"]
    match = re.fullmatch(r"[<>=|]?[A-Za-z?](\d+)", dtype) if isinstance(dtype, str) else None
    if match is None:
        _fail("executed action NPY dtype is unsupported")
    item_size = int(match.group(1))
    expected_bytes = math.prod(shape) * item_size
    if len(payload) - header_end != expected_bytes:
        _fail("executed action NPY payload length disagrees with its shape and dtype")
    return shape


def _future_evidence(model_id: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    if model_id == "efficient_wam_rt_robotwin":
        directory = Path(str(result.get("predicted_video_dir", ""))).expanduser().resolve()
        if not directory.is_dir():
            _fail("Efficient-WAM-RT did not retain its exposed predicted-future directory")
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        if not files:
            _fail("Efficient-WAM-RT predicted-future directory is empty")
        return [{"kind": "decoded_future_file", **_file_record(path)} for path in files]
    if model_id == "lingbot_va_robotwin":
        path = Path(str(result.get("first_predicted_latent_path", ""))).expanduser().resolve()
        return [{"kind": "latent_tensor", **_file_record(path)}]
    if result.get("predicted_video_dir") or result.get("first_predicted_latent_path"):
        _fail("FastWAM action-only interface unexpectedly reported future evidence")
    return []


def _assert_native_state_consistency(
    trajectory: list[dict[str, Any]], result: dict[str, Any], relation: str
) -> tuple[list[bool], list[bool]]:
    requested: list[bool] = []
    opposite: list[bool] = []
    opposite_relation = "right" if relation == "left" else "left"
    for index, step in enumerate(trajectory):
        if not isinstance(step, dict) or step.get("action_step") != index:
            _fail("native trajectory action_step must be contiguous from zero")
        requested_value = _native_region(step, relation)
        opposite_value = _native_region(step, opposite_relation)
        if type(step.get("relation_region")) is not bool:
            _fail("native trajectory lacks relation_region boolean")
        if step["relation_region"] != requested_value:
            _fail("native relation_region disagrees with the frozen RoboTwin predicate")
        if type(step.get("grippers_open")) is not bool:
            _fail("native trajectory lacks grippers_open boolean")
        if type(step.get("success")) is not bool:
            _fail("native trajectory lacks success boolean")
        if step["success"] != (requested_value and step["grippers_open"]):
            _fail("native success disagrees with relation-region plus detached release")
        requested.append(requested_value)
        opposite.append(opposite_value)
    for label, expected_step in (("initial", trajectory[0]), ("final", trajectory[-1])):
        declared = result.get(label)
        if not isinstance(declared, dict):
            _fail(f"native result lacks {label} state")
        for key in (
            "success",
            "relation_region",
            "object_xyz",
            "target_xyz",
            "object_minus_target_x",
            "object_minus_target_y",
            "distance_xy",
            "grippers_open",
        ):
            if declared.get(key) != expected_step.get(key):
                _fail(f"native result {label}.{key} disagrees with trajectory")
    return requested, opposite


def _write_final_state(
    support_dir: Path,
    relation: str,
    source_final: dict[str, Any],
    transformed_object: list[float],
    transformed_reference: list[float],
) -> dict[str, Any]:
    support_dir.mkdir(parents=True, exist_ok=True)
    path = support_dir / f"{relation}_final_state.json"
    payload = {
        "schema_version": "vla-wam-shared-v3-robotwin-final-state-v1",
        "source_frame": "sapien_world_xyz_m",
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "source_state": source_final,
        "object_xyz": transformed_object,
        "reference_xyz": transformed_reference,
    }
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return _file_record(path)


def build_behavioral_record(
    pair: AuthorizedPair,
    relation: str,
    *,
    runtime: dict[str, Any],
    native_root: Path,
    attempt_id: str,
    behavioral_jsonl_path: Path,
    support_dir: Path,
    operational_wall_time_valid: bool,
) -> dict[str, Any]:
    """Compile one complete native condition into the shared raw v3 schema."""

    row = pair.cell(relation)
    condition_dir = _condition_dir(native_root, pair, relation)
    result_path = condition_dir / "result.json"
    result = _load_object(result_path)
    expected_scalars = {
        "task": pair.anchor_task,
        "environment_seed": pair.environment_seed,
        "sampling_seed": pair.policy_seed,
        "condition": f"direct_command__{relation}",
        "prompt_family": "direct_command",
        "requested_relation": relation,
        "prompt": prompt_for(pair.pair_number, relation),
    }
    for key, expected in expected_scalars.items():
        if result.get(key) != expected:
            _fail(f"native result mismatch for {row['cell_id']}.{key}")
    if pair.model_id == "fastwam_robotwin" and result.get("negative_prompt") not in {"", None}:
        _fail("FastWAM contrastive negative prompting is prohibited in Phase A")
    actions = result.get("actions_executed")
    if type(actions) is not int or not 0 <= actions <= ACTION_CAP:
        _fail("native actions_executed is outside the frozen action cap")
    trajectory_path = condition_dir / "trajectory.json"
    if Path(str(result.get("trajectory_path", ""))).expanduser().resolve() != trajectory_path.resolve():
        _fail("native trajectory_path leaves the exact condition directory")
    try:
        trajectory = json.loads(trajectory_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read native trajectory {trajectory_path}: {error}") from error
    if not isinstance(trajectory, list) or len(trajectory) != actions + 1:
        _fail("native trajectory must retain initial state plus one state per action")
    requested, opposite = _assert_native_state_consistency(trajectory, result, relation)
    requested_success = result.get("requested_success")
    if type(requested_success) is not bool or requested_success != trajectory[-1]["success"]:
        _fail("requested_success disagrees with the frozen final native scorer")
    action_trace_path = condition_dir / "action_trace.npz"
    action_trace = _validate_action_trace(action_trace_path, result, actions)
    simulator_video = condition_dir / "simulator.mp4"
    if Path(str(result.get("simulator_video", ""))).expanduser().resolve() != simulator_video.resolve():
        _fail("native simulator video path leaves the exact condition directory")
    video_record = _file_record(simulator_video)
    if video_record["bytes"] == 0:
        _fail("simulator viewport video is empty")
    future_evidence = _future_evidence(pair.model_id, result)
    transform = runtime["measurement_transform"]
    transformed_object: list[list[float]] = []
    steps: list[dict[str, Any]] = []
    for index, (source, requested_value, opposite_value) in enumerate(
        zip(trajectory, requested, opposite, strict=True)
    ):
        object_xyz = transform_xyz(transform, source["object_xyz"])
        reference_xyz = transform_xyz(transform, source["target_xyz"])
        transformed_object.append(object_xyz)
        steps.append(
            {
                "action_step": index,
                "object_xyz": object_xyz,
                "reference_xyz": reference_xyz,
                "grippers_open": bool(source["grippers_open"]),
                "requested_region": requested_value,
                "opposite_region": opposite_value,
                "source_world_object_xyz": source["object_xyz"],
                "source_world_reference_xyz": source["target_xyz"],
                "source_native_dx_m": float(source["object_minus_target_x"]),
                "source_native_dy_m": float(source["object_minus_target_y"]),
            }
        )
    transformed_reference_final = steps[-1]["reference_xyz"]
    final_state = _write_final_state(
        support_dir,
        relation,
        trajectory[-1],
        transformed_object[-1],
        transformed_reference_final,
    )
    legacy_stage = _legacy_failure_stage(trajectory, requested_success)
    wall_time = result.get("wall_seconds")
    if type(wall_time) not in {int, float} or not math.isfinite(float(wall_time)) or wall_time < 0:
        _fail("native wall_seconds must be finite and non-negative")
    checkpoint = runtime["checkpoint"]
    fixture_artifact = runtime["measurement_transform"]["fixture_validation_artifact"]
    record: dict[str, Any] = {
        "schema_version": BEHAVIORAL_SCHEMA_VERSION,
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": STUDY_ID,
        "arena": BEHAVIORAL_ARENA,
        "registered_cell_id": row["cell_id"],
        "attempt_id": f"{attempt_id}:{relation}",
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "requested_relation": relation,
        "prompt": row["prompt"],
        "prompt_family": row["prompt_family"],
        "predicate_id": row["success_predicate_id"],
        "reset_id": row["reset_identity"],
        "environment_seed": pair.environment_seed,
        "policy_seed": pair.policy_seed,
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": {"id": checkpoint["id"], "revision": checkpoint["revision"]},
        "runtime_identity": {
            "id": runtime["runtime_id"],
            "sha256": runtime["runtime_identity_sha256"],
        },
        "artifacts": {
            "viewport_video": video_record,
            "executed_action_trace": action_trace,
            "raw_result_jsonl": {
                "path": str(Path(behavioral_jsonl_path).resolve()),
                "integrity_scope": "batch_manifest_after_close",
            },
            "source_result": _file_record(result_path),
            "source_trajectory": _file_record(trajectory_path),
            "fixture_manifest": fixture_artifact,
            "final_state": final_state,
        },
        "future_interface": MODEL_SPECS[pair.model_id]["future_interface"],
        "future_evidence": future_evidence,
        "actions_executed": actions,
        "action_cap": ACTION_CAP,
        "right_censored": bool(not requested_success and actions == ACTION_CAP),
        "requested_success": requested_success,
        "final_detached_release": bool(trajectory[-1]["grippers_open"]),
        "failure_stage": legacy_stage,
        "frozen_failure_stage": legacy_stage,
        "failure_taxonomy": "correct" if requested_success else "pick_failed",
        "first_contact_step": None,
        "first_contact_unavailable_reason": (
            "The frozen RoboTwin v2 runner did not retain a contact stream."
        ),
        "wall_time_s": float(wall_time),
        "operational_wall_time_valid": operational_wall_time_valid,
        "steps": steps,
        "event_timeline": _event_timeline(
            requested=requested,
            opposite=opposite,
            transformed_object_xyz=transformed_object,
            actions_executed=actions,
        ),
        "source_frame": "sapien_world_xyz_m",
        "measurement_transform_sha256": transform["transform_sha256"],
        "native_object_name": result.get("object_name"),
        "native_reference_name": result.get("target_name"),
    }
    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    measurements = derive_measurements(record)
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    return validate_behavioral_record(record)


def _physical_initial_hash(record: dict[str, Any]) -> str:
    step = record["steps"][0]
    payload = {
        "object_xyz": step["object_xyz"],
        "reference_xyz": step["reference_xyz"],
        "source_world_object_xyz": step["source_world_object_xyz"],
        "source_world_reference_xyz": step["source_world_reference_xyz"],
        "native_object_name": record.get("native_object_name"),
        "native_reference_name": record.get("native_reference_name"),
    }
    return canonical_sha256(payload)


def _thermal_status(path: Path, guard_return_code: int | None) -> tuple[bool, bool]:
    """Return (runtime_intervention, operational_wall_time_valid)."""

    event_path = Path(path)
    if not event_path.is_file():
        return bool(guard_return_code not in {None, 0}), False
    intervention = False
    completed = False
    for line_number, line in enumerate(event_path.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise AdapterError(f"invalid thermal event {event_path}:{line_number}: {error}") from error
        name = event.get("event")
        if name in {"cooldown_started", "cooldown_completed", "emergency_hold"}:
            intervention = True
        if name == "monitor_completed" and event.get("worker_exit_code", 0) == 0:
            completed = True
        if name in {
            "temperature_query_failed",
            "worker_missing",
            "worker_exit_nonzero",
            "monitor_error",
        }:
            intervention = True
    valid_wall = guard_return_code in {None, 0} and completed and not intervention
    return intervention, valid_wall


def _log_hash(error: str, *paths: Path) -> str:
    digest = hashlib.sha256(error.encode())
    for path in paths:
        if Path(path).is_file():
            digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def build_infrastructure_record(
    pair: AuthorizedPair,
    relation: str,
    *,
    runtime: dict[str, Any],
    attempt_id: str,
    infrastructure_jsonl_path: Path,
    native_root: Path,
    error: str,
    runtime_intervention: bool,
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, Any]:
    row = pair.cell(relation)
    condition = _condition_dir(native_root, pair, relation)
    retained = list(condition.rglob("*")) if condition.exists() else []
    partial = any(path.is_file() for path in retained)
    artifacts: dict[str, Any] = {
        "raw_result_jsonl": {
            "path": str(Path(infrastructure_jsonl_path).resolve()),
            "integrity_scope": "batch_manifest_after_close",
        }
    }
    for label, path in (("stdout_log", stdout_path), ("stderr_log", stderr_path)):
        if Path(path).is_file():
            artifacts[label] = _file_record(path)
    checkpoint = runtime["checkpoint"]
    record = {
        "schema_version": INFRASTRUCTURE_SCHEMA_VERSION,
        "record_type": "infrastructure_attempt",
        "behavioral_result_valid": False,
        "classification": "partial" if partial else "technical_invalid",
        "study_id": STUDY_ID,
        "arena": BEHAVIORAL_ARENA,
        "registered_cell_id": row["cell_id"],
        "attempt_id": f"{attempt_id}:{relation}:infrastructure",
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "prompt": row["prompt"],
        "prompt_family": row["prompt_family"],
        "predicate_id": row["success_predicate_id"],
        "reset_id": row["reset_identity"],
        "environment_seed": pair.environment_seed,
        "policy_seed": pair.policy_seed,
        "measurement_frame": MEASUREMENT_FRAME_ID,
        "measurement_frame_description": MEASUREMENT_FRAME_DESCRIPTION,
        "checkpoint": {"id": checkpoint["id"], "revision": checkpoint["revision"]},
        "runtime_identity": {
            "id": runtime["runtime_id"],
            "sha256": runtime["runtime_identity_sha256"],
        },
        "artifacts": artifacts,
        "stage": "native_pair_execution_or_postprocess",
        "error": error,
        "log_hash": _log_hash(error, stdout_path, stderr_path),
        "runtime_intervention": runtime_intervention,
        "repair_attempt_id": None,
        "event_timeline": [
            {"sequence": 0, "stage": "registered_pair_validated"},
            {
                "sequence": 1,
                "stage": "partial_output_retained" if partial else "pre_behavior_failure_retained",
            },
        ],
    }
    return validate_infrastructure_record(record)


def compile_native_pair(
    pair: AuthorizedPair,
    *,
    runtime: dict[str, Any],
    attempt_dir: Path,
    attempt_id: str,
    guard_return_code: int | None = None,
) -> dict[str, Any]:
    """Compile complete cells and separately retain missing/invalid cells."""

    attempt = Path(attempt_dir).resolve()
    native_root = attempt / "native"
    behavioral_path = attempt / "behavioral_episodes.jsonl"
    infrastructure_path = attempt / "infrastructure_attempts.jsonl"
    derived_paths = (
        behavioral_path,
        behavioral_path.with_name(behavioral_path.name + ".manifest.json"),
        infrastructure_path,
        infrastructure_path.with_name(infrastructure_path.name + ".manifest.json"),
        attempt / "attempt_manifest.json",
    )
    if any(path.exists() for path in derived_paths) or any(
        path.is_file() for path in (attempt / "support").glob("*")
    ):
        _fail(
            "refusing to overwrite an existing compiled attempt; preserve it and use a new attempt_id"
        )
    stdout_path = attempt / "native.stdout.log"
    stderr_path = attempt / "native.stderr.log"
    runtime_intervention, valid_wall = _thermal_status(
        attempt / "thermal_events.jsonl", guard_return_code
    )
    behavioral: list[dict[str, Any]] = []
    infrastructure: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for relation in ("left", "right"):
        try:
            record = build_behavioral_record(
                pair,
                relation,
                runtime=runtime,
                native_root=native_root,
                attempt_id=attempt_id,
                behavioral_jsonl_path=behavioral_path,
                support_dir=attempt / "support",
                operational_wall_time_valid=valid_wall,
            )
            behavioral.append(record)
        except (AdapterError, OSError, ValueError) as error:
            message = str(error)
            errors[relation] = message
            infrastructure.append(
                build_infrastructure_record(
                    pair,
                    relation,
                    runtime=runtime,
                    attempt_id=attempt_id,
                    infrastructure_jsonl_path=infrastructure_path,
                    native_root=native_root,
                    error=message,
                    runtime_intervention=runtime_intervention,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            )
    if len(behavioral) == 2:
        initial_hashes = {_physical_initial_hash(record) for record in behavioral}
        if len(initial_hashes) != 1:
            # Neither cell may enter a matched-pair denominator when the reset
            # invariant is false.  Preserve both as infrastructure attempts.
            mismatch = "LEFT/RIGHT physical initial states are not byte-identical after exact reset"
            errors = {"left": mismatch, "right": mismatch}
            behavioral = []
            infrastructure = [
                build_infrastructure_record(
                    pair,
                    relation,
                    runtime=runtime,
                    attempt_id=attempt_id,
                    infrastructure_jsonl_path=infrastructure_path,
                    native_root=native_root,
                    error=mismatch,
                    runtime_intervention=runtime_intervention,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                for relation in ("left", "right")
            ]
    manifests: dict[str, Any] = {}
    if behavioral:
        manifests["behavioral"] = write_jsonl(behavioral_path, behavioral)
    if infrastructure:
        manifests["infrastructure"] = write_jsonl(infrastructure_path, infrastructure)
    summary = {
        "schema_version": "vla-wam-shared-v3-robotwin-pair-attempt-manifest-v1",
        "study_id": STUDY_ID,
        "model_id": pair.model_id,
        "pair_id": pair.pair_id,
        "pair_number": pair.pair_number,
        "replicate": pair.replicate,
        "environment_seed": pair.environment_seed,
        "policy_seed": pair.policy_seed,
        "attempt_id": attempt_id,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "queue_sha256": pair.queue_sha256,
        "guard_return_code": guard_return_code,
        "runtime_intervention": runtime_intervention,
        "operational_wall_time_valid": valid_wall,
        "behavioral_rows": len(behavioral),
        "infrastructure_rows": len(infrastructure),
        "errors": errors,
        "batch_manifests": manifests,
    }
    summary["manifest_sha256"] = canonical_sha256(summary)
    (attempt / "attempt_manifest.json").write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return summary


def execute_pair(
    pair: AuthorizedPair,
    *,
    runtime: dict[str, Any],
    study_root: Path,
    external_repository: Path,
    attempt_dir: Path,
    attempt_id: str,
    gpu_index: int,
) -> dict[str, Any]:
    """Execute one immutable attempt through the guard, then compile it."""

    attempt = Path(attempt_dir).resolve()
    if attempt.exists():
        _fail(f"refusing to execute into an existing attempt directory: {attempt}")
    attempt.mkdir(parents=True)
    native_root = attempt / "native"
    native_command = build_native_command(
        pair,
        study_root=study_root,
        external_repository=external_repository,
        native_output_dir=native_root,
    )
    guard_command = build_guard_command(
        pair,
        study_root=study_root,
        attempt_dir=attempt,
        gpu_index=gpu_index,
        native_command=native_command,
    )
    environment = dict(os.environ)
    environment[MODEL_SPECS[pair.model_id]["gpu_environment_variable"]] = str(gpu_index)
    environment["VLA_WAM_V2_STUDY_ROOT"] = str(Path(study_root).resolve())
    with (attempt / "native.stdout.log").open("wb") as stdout_handle, (
        attempt / "native.stderr.log"
    ).open("wb") as stderr_handle:
        completed = subprocess.run(
            guard_command,
            cwd=Path(study_root).resolve(),
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
        )
    return compile_native_pair(
        pair,
        runtime=runtime,
        attempt_dir=attempt,
        attempt_id=attempt_id,
        guard_return_code=completed.returncode,
    )


def _common_parser(parser: argparse.ArgumentParser, *, needs_gpu: bool) -> None:
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT_DEFAULT)
    parser.add_argument("--model-id", choices=sorted(MODEL_SPECS), required=True)
    parser.add_argument("--pair", type=int, required=True, help="Pair number 3..9")
    parser.add_argument("--replicate", type=int, required=True, help="New replicate 1..9; r0 is forbidden")
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--external-repository", type=Path, required=True)
    parser.add_argument("--simulator-repository", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--attempt-id", required=True)
    if needs_gpu:
        parser.add_argument("--gpu-index", type=int, required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    _common_parser(subparsers.add_parser("dry-run"), needs_gpu=True)
    _common_parser(subparsers.add_parser("execute"), needs_gpu=True)
    _common_parser(subparsers.add_parser("compile"), needs_gpu=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.study_root.expanduser().resolve()
    attempt = args.attempt_dir.expanduser().resolve()
    try:
        attempt.relative_to(root)
    except ValueError:
        pass
    else:
        _fail("raw attempts must live on the PVC outside the Git study repository")
    pair = load_authorized_pair(root, args.model_id, args.pair, args.replicate)
    runtime = verify_runtime_identity(
        root,
        args.model_id,
        args.runtime_manifest,
        external_repository=args.external_repository,
        simulator_repository=args.simulator_repository,
        verify_live_files=True,
    )
    if args.mode == "compile":
        summary = compile_native_pair(
            pair,
            runtime=runtime,
            attempt_dir=args.attempt_dir,
            attempt_id=args.attempt_id,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["infrastructure_rows"] == 0 else 2
    native = build_native_command(
        pair,
        study_root=root,
        external_repository=args.external_repository,
        native_output_dir=args.attempt_dir / "native",
    )
    guard = build_guard_command(
        pair,
        study_root=root,
        attempt_dir=args.attempt_dir,
        gpu_index=args.gpu_index,
        native_command=native,
    )
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "dry_run_only_no_model_or_simulator_loaded",
                    "model_id": pair.model_id,
                    "pair_id": pair.pair_id,
                    "replicate": pair.replicate,
                    "environment_seed": pair.environment_seed,
                    "policy_seed": pair.policy_seed,
                    "left_prompt": pair.left["prompt"],
                    "right_prompt": pair.right["prompt"],
                    "runtime_identity_sha256": runtime["runtime_identity_sha256"],
                    "native_command": native,
                    "guard_command": guard,
                    "environment": {
                        MODEL_SPECS[pair.model_id]["gpu_environment_variable"]: args.gpu_index,
                        "VLA_WAM_V2_STUDY_ROOT": str(root),
                    },
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary = execute_pair(
        pair,
        runtime=runtime,
        study_root=root,
        external_repository=args.external_repository,
        attempt_dir=args.attempt_dir,
        attempt_id=args.attempt_id,
        gpu_index=args.gpu_index,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["infrastructure_rows"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
