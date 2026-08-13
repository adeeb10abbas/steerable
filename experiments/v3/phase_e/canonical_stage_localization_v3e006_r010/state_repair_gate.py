#!/usr/bin/env python3
"""Evaluate the preregistered V3-E006-R010 contact-consistent repair schedule.

Historical sources came from E004 pi0.5, but this process makes no new model
request and selects only with the unchanged physical/OOD/camera gates.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import torch
from isaaclab.app import AppLauncher


BOOTSTRAP = argparse.ArgumentParser(add_help=False)
BOOTSTRAP.add_argument("--study-root", type=Path, required=True)
BOOTSTRAP.add_argument("--robolab-root", type=Path, required=True)
BOOTSTRAP.add_argument("--e004-candidate", type=Path, required=True)
BOOTSTRAP.add_argument("--e004-candidate-sha256", required=True)
BOOTSTRAP.add_argument("--ood-freeze", type=Path, required=True)
BOOTSTRAP.add_argument("--ood-freeze-sha256", required=True)
BOOTSTRAP.add_argument("--e004-reset-reference", type=Path, required=True)
BOOTSTRAP.add_argument("--e004-reset-reference-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-bindings", type=Path, required=True)
BOOTSTRAP.add_argument("--runtime-bindings-sha256", required=True)
BOOTSTRAP.add_argument("--repair-registration", type=Path, required=True)
BOOTSTRAP.add_argument("--repair-registration-sha256", required=True)
BOOTSTRAP.add_argument("--candidate-schedule", type=Path, required=True)
BOOTSTRAP.add_argument("--candidate-schedule-sha256", required=True)
BOOTSTRAP.add_argument("--original-closure-binding", type=Path, required=True)
BOOTSTRAP.add_argument("--original-closure-binding-sha256", required=True)
BOOTSTRAP.add_argument("--predecessor-closure-binding", type=Path, required=True)
BOOTSTRAP.add_argument("--predecessor-closure-binding-sha256", required=True)
BOOTSTRAP.add_argument("--source-push-gate", type=Path, required=True)
BOOTSTRAP.add_argument("--source-push-gate-sha256", required=True)
BOOTSTRAP.add_argument("--control-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--paired-scene-asset", type=Path, required=True)
BOOTSTRAP.add_argument("--output-dir", type=Path, required=True)
BOOTSTRAP.add_argument("--pod", required=True)
BOOTSTRAP.add_argument("--pod-uid", required=True)
BOOTSTRAP.add_argument("--gpu-uuid", required=True)
BOOTSTRAP.add_argument("--expected-study-commit", required=True)
BOOTSTRAP.add_argument("--expected-robolab-commit", default="0aef241fb088ca21bb4ebd24448940ed56620d17")
BOOTSTRAP.add_argument("--health-preflight-root", type=Path, required=True)
BOOTSTRAP.add_argument("--health-harness-sha256", required=True)
BOOTSTRAP.add_argument("--health-launch-sha256", required=True)
BOOTSTRAP.add_argument("--health-child-sha256", required=True)
BOOTSTRAP.add_argument("--health-runtime-log-sha256", required=True)
BOOTSTRAP.add_argument("--runtime-log", type=Path, required=True)
BOOTSTRAP.add_argument("--container-image", required=True)
BOOTSTRAP.add_argument("--container-id", required=True)
BOOTSTRAP.add_argument("--driver-version", required=True)
from robolab.eval.runner import add_common_eval_args

add_common_eval_args(BOOTSTRAP)
AppLauncher.add_app_launcher_args(BOOTSTRAP)
args, _ = BOOTSTRAP.parse_known_args()
args.enable_cameras = True
args.num_envs = 1
args.num_runs = 1

study_root = args.study_root.resolve()
robolab_root = args.robolab_root.resolve()
sys.path[:0] = [str(study_root), str(robolab_root)]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_passed_health_preflight(files: Mapping[str, tuple[Path, str]]) -> None:
    harness_path, _ = files["harness_result"]
    child_path, _ = files["preflight_result"]
    try:
        harness = json.loads(harness_path.read_text(encoding="utf-8"))
        child = json.loads(child_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("formal health preflight JSON is missing or invalid") from exc
    if harness.get("status") != "passed_generic_zero_model_health_preflight" or harness.get("passed") is not True:
        raise ValueError("formal health harness did not pass")
    if child.get("status") != "passed_generic_zero_model_cuda_vulkan_isaac_physics_render_health_preflight" or child.get("passed") is not True:
        raise ValueError("formal health child did not pass exact runtime checks")
    for label, payload in (("harness", harness), ("child", child)):
        for key in ("model_request_count", "behavioral_episode_count", "state_candidate_count"):
            if payload.get(key) != 0:
                raise ValueError(f"formal health {label} has nonzero {key}")
    binding = harness.get("child_report")
    if not isinstance(binding, Mapping) or binding != {
        "path": str(child_path.resolve()),
        "bytes": child_path.stat().st_size,
        "sha256": _sha(child_path),
    }:
        raise ValueError("formal health harness child binding differs")


health_files = {
    "harness_result": (args.health_preflight_root / "harness_result.json", args.health_harness_sha256),
    "preflight_launch": (args.health_preflight_root / "preflight_launch.json", args.health_launch_sha256),
    "preflight_result": (args.health_preflight_root / "preflight_result.json", args.health_child_sha256),
    "runtime_log": (args.health_preflight_root / "runtime.log", args.health_runtime_log_sha256),
}
for path, digest in (
    (args.e004_candidate, args.e004_candidate_sha256),
    (args.ood_freeze, args.ood_freeze_sha256),
    (args.e004_reset_reference, args.e004_reset_reference_sha256),
    (args.runtime_bindings, args.runtime_bindings_sha256),
    (args.repair_registration, args.repair_registration_sha256),
    (args.candidate_schedule, args.candidate_schedule_sha256),
    (args.original_closure_binding, args.original_closure_binding_sha256),
    (args.predecessor_closure_binding, args.predecessor_closure_binding_sha256),
    (args.source_push_gate, args.source_push_gate_sha256),
    *(health_files.values()),
):
    if not path.is_file() or _sha(path) != digest:
        BOOTSTRAP.error(f"hash-bound input is missing or changed: {path}")
try:
    _verify_passed_health_preflight(health_files)
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
for path in (args.control_scene_asset, args.paired_scene_asset):
    if not path.is_file():
        BOOTSTRAP.error(f"scene input is missing: {path}")
if args.output_dir.exists():
    BOOTSTRAP.error(f"refusing to overwrite state-construction evidence: {args.output_dir}")
if subprocess.check_output(["git", "-C", str(study_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_study_commit:
    BOOTSTRAP.error("study checkout differs from expected state-construction commit")
if subprocess.check_output(["git", "-C", str(robolab_root), "rev-parse", "HEAD"], text=True).strip() != args.expected_robolab_commit:
    BOOTSTRAP.error("RoboLab checkout differs from the pinned commit")

try:
    repair_registration = json.loads(args.repair_registration.read_text(encoding="utf-8"))
    candidate_schedule = json.loads(args.candidate_schedule.read_text(encoding="utf-8"))
    source_push_gate = json.loads(args.source_push_gate.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    BOOTSTRAP.error(f"repair control artifact is unreadable: {exc}")
if repair_registration.get("repair_amendment_id") != "V3-E006-R010":
    BOOTSTRAP.error("repair registration ID differs")
if repair_registration.get("status") != "prospectively_registered_before_any_r010_live_diagnostic_candidate_or_model_request":
    BOOTSTRAP.error("repair registration status differs")
if repair_registration.get("counts_at_registration") != {
    "r010_geometry_attachment_preflights": 0,
    "r010_live_diagnostics": 0,
    "r010_live_candidate_evaluations": 0,
    "model_requests": 0,
    "behavioral_episodes": 0,
}:
    BOOTSTRAP.error("repair registration does not bind zero counts")
if (
    candidate_schedule.get("repair_amendment_id") != "V3-E006-R010"
    or candidate_schedule.get("status")
    != "frozen_before_any_r010_live_diagnostic_candidate_or_model_request"
    or candidate_schedule.get("candidate_budget") != 4
    or candidate_schedule.get("diagnostic_budget") != 4
    or candidate_schedule.get("r010_geometry_attachment_preflight_count") != 0
    or candidate_schedule.get("r010_live_diagnostic_count") != 0
    or candidate_schedule.get("r010_live_candidate_evaluation_count") != 0
    or candidate_schedule.get("model_request_count") != 0
    or candidate_schedule.get("behavioral_episode_count") != 0
):
    BOOTSTRAP.error("candidate schedule identity/budget differs")
if (
    len(candidate_schedule.get("candidate_pairs", [])) != 4
    or [row.get("candidate_rank") for row in candidate_schedule["candidate_pairs"]]
    != list(range(1, 5))
    or len(candidate_schedule.get("known_reachable_diagnostics", [])) != 4
    or [row.get("diagnostic_index_one_based") for row in candidate_schedule["known_reachable_diagnostics"]]
    != list(range(1, 5))
):
    BOOTSTRAP.error("candidate schedule ranks differ")
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.residual_correction import (
    corrected_command,
    validate_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.pinch_geometry import (
    validate_attachment_preflight_contract,
    validate_contract as validate_pinch_geometry_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.predecessor_contract import (
    R009_RESULTS_SHA256,
    validate_r009_attachment_invalid_closure,
)

correction_contract = candidate_schedule.get("residual_correction_contract", {})
try:
    validate_contract(correction_contract)
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
correction_digest = hashlib.sha256(
    (json.dumps(correction_contract, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()
if correction_digest != candidate_schedule.get("residual_correction_contract_sha256"):
    BOOTSTRAP.error("R010 residual-correction contract digest differs")
pinch_geometry_contract = candidate_schedule.get("pinch_geometry_contract", {})
try:
    validate_pinch_geometry_contract(pinch_geometry_contract)
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
pinch_geometry_digest = hashlib.sha256(
    (json.dumps(pinch_geometry_contract, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()
if pinch_geometry_digest != candidate_schedule.get("pinch_geometry_contract_sha256"):
    BOOTSTRAP.error("R010 pinch-geometry contract digest differs")
geometry_attachment_preflight_contract = candidate_schedule.get(
    "geometry_attachment_preflight_contract", {}
)
try:
    validate_attachment_preflight_contract(geometry_attachment_preflight_contract)
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
geometry_attachment_preflight_digest = hashlib.sha256(
    (
        json.dumps(
            geometry_attachment_preflight_contract,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
).hexdigest()
if geometry_attachment_preflight_digest != candidate_schedule.get(
    "geometry_attachment_preflight_contract_sha256"
):
    BOOTSTRAP.error("R010 geometry-attachment preflight contract digest differs")
joint_handoff_contract = candidate_schedule.get("joint_handoff_contract", {})
joint_handoff_digest = hashlib.sha256(
    (json.dumps(joint_handoff_contract, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()
if (
    joint_handoff_digest != candidate_schedule.get("joint_handoff_contract_sha256")
    or joint_handoff_contract.get("settle_steps") != 600
    or joint_handoff_contract.get("joint_target_write_count") != 1
):
    BOOTSTRAP.error("R010 joint handoff contract differs")
lifecycle_contract = candidate_schedule.get("construction_lifecycle_contract", {})
lifecycle_digest = hashlib.sha256(
    (json.dumps(lifecycle_contract, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
).hexdigest()
if (
    lifecycle_digest != candidate_schedule.get("construction_lifecycle_contract_sha256")
    or lifecycle_contract.get("worst_case_steps") != 1695
    or lifecycle_contract.get("registered_max_episode_length_steps") != 1800
    or lifecycle_contract.get("registered_margin_steps") != 105
):
    BOOTSTRAP.error("R010 construction lifecycle contract differs")
archived_predecessor_contracts = candidate_schedule.get("archived_predecessor_contracts", {})
archived_horizon = archived_predecessor_contracts.get("r005_construction_horizon_contract", {})
archived_open_contact = archived_predecessor_contracts.get(
    "r007_open_contact_construction_contract", {}
)
if (
    archived_predecessor_contracts.get("status")
    != "archived_lineage_only_not_active_r010_runtime_evidence"
    or "construction_horizon_contract" in candidate_schedule
    or "open_contact_construction_contract" in candidate_schedule
    or hashlib.sha256(
        (json.dumps(archived_horizon, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    != archived_predecessor_contracts.get("r005_construction_horizon_contract_sha256")
    or hashlib.sha256(
        (json.dumps(archived_open_contact, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()
    != archived_predecessor_contracts.get("r007_open_contact_construction_contract_sha256")
):
    BOOTSTRAP.error("R010 predecessor construction contracts are not explicitly archived")
construction_asset_bindings = candidate_schedule.get("construction_asset_bindings", {})
if set(construction_asset_bindings) != {"robot_usd", "cube_usd"}:
    BOOTSTRAP.error("R010 collision-geometry asset inventory differs")
for asset_label, expected in construction_asset_bindings.items():
    asset_path = Path(str(expected.get("path", "")))
    if (
        not asset_path.is_absolute()
        or not asset_path.is_file()
        or asset_path.stat().st_size != expected.get("bytes")
        or _sha(asset_path) != expected.get("sha256")
    ):
        BOOTSTRAP.error(
            f"R010 collision-geometry asset binding changed: {asset_label}: {asset_path}"
        )
geometry_oracle_source_bindings = candidate_schedule.get(
    "geometry_oracle_source_bindings", {}
)
if set(geometry_oracle_source_bindings) != {
    "physx_python_interface",
    "nvidia_physx_camera_sync_test",
    "isaac_simulation_manager",
    "isaaclab_simulation_context",
}:
    BOOTSTRAP.error("R010 geometry-oracle source inventory differs")
for source_label, expected in geometry_oracle_source_bindings.items():
    source_path = Path(str(expected.get("path", "")))
    if (
        not source_path.is_absolute()
        or not source_path.is_file()
        or source_path.stat().st_size != expected.get("bytes")
        or _sha(source_path) != expected.get("sha256")
    ):
        BOOTSTRAP.error(
            f"R010 geometry-oracle source binding changed: {source_label}: {source_path}"
        )
for diagnostic in candidate_schedule["known_reachable_diagnostics"]:
    if (
        diagnostic.get("r004_residual_correction_contract_sha256") != correction_digest
        or diagnostic.get("maximum_correction_rounds") != 3
    ):
        BOOTSTRAP.error("R010 diagnostic correction binding differs")
for pair in candidate_schedule["candidate_pairs"]:
    for stage_name in ("canonical_grasp", "canonical_carry"):
        stage_schedule = pair[stage_name]
        if (
            stage_schedule.get("r010_target_cube_pose")
            != stage_schedule.get("r009_target_cube_pose")
            or stage_schedule.get("r010_acquisition_base_quaternion_world_wxyz")
            != stage_schedule.get("r009_acquisition_base_quaternion_world_wxyz")
            or stage_schedule.get("r010_final_base_quaternion_world_wxyz")
            != stage_schedule.get("r009_final_base_quaternion_world_wxyz")
        ):
            BOOTSTRAP.error(
                "R010 runtime-driving target aliases differ from immutable R009"
            )
        initialization = stage_schedule.get("r004_solver_initialization", {})
        if initialization.get("residual_correction_contract_sha256") != correction_digest:
            BOOTSTRAP.error("R010 candidate correction binding differs")
        for waypoint in initialization.get("waypoints", []):
            if (
                waypoint.get("maximum_correction_rounds") != 3
                or waypoint.get("r004_residual_correction_contract_sha256") != correction_digest
            ):
                BOOTSTRAP.error("R010 waypoint correction binding differs")
schedule_registration = candidate_schedule.get("repair_registration", {})
if schedule_registration.get("bytes") != args.repair_registration.stat().st_size or schedule_registration.get("sha256") != args.repair_registration_sha256:
    BOOTSTRAP.error("candidate schedule does not bind this repair registration")
schedule_ood = candidate_schedule.get("unchanged_gate_bindings", {}).get("ood_freeze", {})
if schedule_ood.get("bytes") != args.ood_freeze.stat().st_size or schedule_ood.get("sha256") != args.ood_freeze_sha256:
    BOOTSTRAP.error("candidate schedule does not bind the frozen OOD reference")
schedule_closure = candidate_schedule.get("original_v3e006_closure_binding", {})
if schedule_closure.get("bytes") != args.original_closure_binding.stat().st_size or schedule_closure.get("sha256") != args.original_closure_binding_sha256:
    BOOTSTRAP.error("candidate schedule does not bind the original closure proof")
r009_predecessor = candidate_schedule.get("r009_predecessor", {})
predecessor_binding = r009_predecessor.get("results", {})
if (
    predecessor_binding.get("bytes") != args.predecessor_closure_binding.stat().st_size
    or predecessor_binding.get("sha256") != args.predecessor_closure_binding_sha256
    or predecessor_binding.get("sha256") != R009_RESULTS_SHA256
    or r009_predecessor.get("closure_commit")
    != "9000d2897e634eee9469d02c9449baf85fe15729"
    or r009_predecessor.get("raw_result", {}).get("sha256")
    != "0753b4fe44ed479d4575804f3b5f38e3244a1cea25de3f1a8564dca4cd26ab3c"
):
    BOOTSTRAP.error("candidate schedule does not bind the immutable R009 attachment-invalid closure")
try:
    predecessor_result = json.loads(args.predecessor_closure_binding.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    BOOTSTRAP.error(f"R009 predecessor result is unreadable: {exc}")
try:
    validate_r009_attachment_invalid_closure(predecessor_result)
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
for label in ("raw_result", "raw_harness", "raw_launch", "raw_runtime_log", "authoritative_target_validation_receipt"):
    expected = predecessor_result[label]
    raw_path = Path(str(expected["path"]))
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != expected["bytes"]
        or _sha(raw_path) != expected["sha256"]
    ):
        BOOTSTRAP.error(f"R009 predecessor raw evidence changed: {label}")
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.source_gate_contract import (
    validate_source_gate,
)

try:
    validate_source_gate(
        source_push_gate, study_root=study_root, verify_raw_history=True
    )
except ValueError as exc:
    BOOTSTRAP.error(str(exc))
implementation_commit = str(source_push_gate.get("implementation_commit", ""))
if not implementation_commit or subprocess.run(
    ["git", "-C", str(study_root), "merge-base", "--is-ancestor", implementation_commit, args.expected_study_commit],
    check=False,
).returncode:
    BOOTSTRAP.error("source-push implementation commit is not an ancestor of runtime checkout")
implementation_files = source_push_gate.get("implementation_files")
if not isinstance(implementation_files, list) or not implementation_files:
    BOOTSTRAP.error("source-push gate has no implementation-file inventory")
for row in implementation_files:
    relative = Path(str(row.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        BOOTSTRAP.error(f"source-push gate contains unsafe path: {relative}")
    actual = study_root / relative
    if not actual.is_file() or actual.stat().st_size != row.get("bytes") or _sha(actual) != row.get("sha256"):
        BOOTSTRAP.error(f"source-push implementation file changed: {relative}")


def _verify_historical_source_rows(schedule: Mapping[str, Any]) -> dict[str, Any]:
    """Rehash and re-read every selected historical row before AppLauncher."""

    verified: dict[str, Any] = {}
    sources: list[tuple[str, str, Mapping[str, Any]]] = []
    for diagnostic in schedule["known_reachable_diagnostics"]:
        sources.append(
            (
                f"diagnostic{int(diagnostic['diagnostic_index_one_based']):02d}",
                str(diagnostic["source_side"]),
                diagnostic["source"],
            )
        )
    for pair in schedule["candidate_pairs"]:
        for stage in ("canonical_grasp", "canonical_carry"):
            side = str(pair[stage]["r004_solver_initialization"]["source_side"])
            sources.append(
                (
                    f"rank{int(pair['candidate_rank']):02d}/{stage}",
                    side,
                    pair[stage]["both_direction_sources"][side],
                )
            )
    for source_label, side, source in sources:
        stage = "canonical_grasp" if "grasp" in source_label else "canonical_carry"
        provenance = source["provenance"]
        retained: dict[str, Any] = {}
        for name in ("raw_episode", "state_capture", "hdf5_trace"):
            expected = provenance[name]
            path = Path(str(expected["path"])).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != expected["bytes"]
                or _sha(path) != expected["sha256"]
            ):
                BOOTSTRAP.error(f"historical {stage}/{side} {name} binding changed: {path}")
            retained[name] = {
                "path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)
            }
        state_capture = json.loads(Path(retained["state_capture"]["path"]).read_text(encoding="utf-8"))
        state_index = int(source["state_capture_index"])
        hdf5_index = int(source["hdf5_index"])
        if hdf5_index - state_index != int(provenance["hdf5_to_state_capture_offset"]):
            BOOTSTRAP.error(f"historical {stage}/{side} HDF5 alignment changed")
        state_row = state_capture["steps"][state_index]
        if state_row.get("object_grabbed") is not True:
            BOOTSTRAP.error(f"historical {stage}/{side} source was not grabbed")
        with h5py.File(retained["hdf5_trace"]["path"], "r") as handle:
            root = handle["data/demo_0"]
            action = np.asarray(root["actions"][hdf5_index], dtype=np.float64)
            joints = np.asarray(
                root["states/articulation/robot/joint_position"][hdf5_index], dtype=np.float64
            )
            cube_pose = np.asarray(
                root["states/rigid_object/rubiks_cube/root_pose"][hdf5_index], dtype=np.float64
            )
            base_link_position = np.asarray(root["ee_pose/position"][hdf5_index], dtype=np.float64)
            base_link_quaternion = np.asarray(root["ee_pose/orientation"][hdf5_index], dtype=np.float64)
        if action.shape[0] < 1 or float(action[-1]) != 1.0:
            BOOTSTRAP.error(f"historical {stage}/{side} close action is not exactly 1.0")
        comparisons = (
            (joints, source["joint_position_rad"], "joint position"),
            (cube_pose, source["cube_pose_world_wxyz"], "cube pose"),
            (
                base_link_position,
                source.get("base_link_position_world_m", source.get("eef_position_world_m")),
                "base_link position",
            ),
            (
                base_link_quaternion,
                source.get(
                    "base_link_quaternion_world_wxyz",
                    source.get("eef_quaternion_world_wxyz"),
                ),
                "base_link quaternion",
            ),
        )
        for actual, expected, field_label in comparisons:
            if actual.shape != np.asarray(expected).shape or not np.array_equal(
                actual.astype(np.float64), np.asarray(expected, dtype=np.float64)
            ):
                BOOTSTRAP.error(
                    f"historical {stage}/{side} {field_label} differs from frozen schedule"
                )
        verified[f"{source_label}/{side}"] = {
            "bindings": retained,
            "state_capture_index": state_index,
            "hdf5_index": hdf5_index,
            "object_grabbed": True,
            "action_last_component": float(action[-1]),
            "joint_position_rad": joints.tolist(),
            "cube_pose_world_wxyz": cube_pose.tolist(),
            "hdf5_ee_pose_semantic_frame": "robot/base_link",
            "base_link_position_world_m": base_link_position.tolist(),
            "base_link_quaternion_world_wxyz": base_link_quaternion.tolist(),
        }
    return verified


HISTORICAL_SOURCE_VERIFICATION = _verify_historical_source_rows(candidate_schedule)
CONTROLLER_SOURCE_VERIFICATION: dict[str, Any] = {}
for name, expected in candidate_schedule.get("controller_source_bindings", {}).items():
    controller_path = Path(str(expected.get("path", ""))).resolve()
    if (
        not controller_path.is_file()
        or controller_path.stat().st_size != expected.get("bytes")
        or _sha(controller_path) != expected.get("sha256")
    ):
        BOOTSTRAP.error(f"registered controller source binding changed: {name}: {controller_path}")
    CONTROLLER_SOURCE_VERIFICATION[str(name)] = {
        "path": str(controller_path),
        "bytes": controller_path.stat().st_size,
        "sha256": _sha(controller_path),
    }

mapping = {name: name for name in ("banana", "banana_right", "bowl", "rubiks_cube")}
os.environ.update(
    {
        "VLA_WAM_V3E004_FIXTURE_CANDIDATE": str(args.e004_candidate.resolve()),
        "VLA_WAM_V3E004_FIXTURE_SHA256": args.e004_candidate_sha256,
        "VLA_WAM_V3E004_SYMMETRY_LEVEL_S": "1.0",
        "VLA_WAM_V3E004_CONTROL_SCENE_ASSET": str(args.control_scene_asset.resolve()),
        "VLA_WAM_V3E004_PAIRED_SCENE_ASSET": str(args.paired_scene_asset.resolve()),
        "VLA_WAM_V3E004_SCENE_OBJECT_MAPPING": json.dumps(mapping, sort_keys=True),
    }
)

simulation_app = AppLauncher(args).app
CURRENT_STAGE = "app_launcher_started"
LAST_REFERENCE_BOUNDS_EVIDENCE: dict[str, Any] | None = None
LAST_PARTIAL_STAGES: dict[str, Any] = {}
LAST_TERMINATION_EVIDENCE: dict[str, Any] | None = None
ENVIRONMENT_LIFECYCLE: list[dict[str, Any]] = []
CANDIDATE_EVALUATION_COUNT = 0
DIAGNOSTIC_EVALUATION_COUNT = 0
COMPLETED_DIAGNOSTICS: list[dict[str, Any]] = []
COMPLETED_ATTEMPTS: list[dict[str, Any]] = []
GEOMETRY_ATTACHMENT_PREFLIGHT: dict[str, Any] | None = None

import carb  # noqa: E402
import omni.physx  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx.bindings._physx import SETTING_UPDATE_TO_USD  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from robolab.constants import set_output_dir  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.sensors.contact_sensor_utils import get_contact_sensors  # noqa: E402
from robolab.core.task.conditionals import object_grabbed  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD  # noqa: E402
from robolab.robots.droid import EEF_OFFSET_POS, EEF_OFFSET_ROT  # noqa: E402

from experiments.v3.phase_e.canonical_stage_localization_v3e006.ood_reference import (  # noqa: E402
    _quat_inverse_wxyz,
    _quat_multiply_wxyz,
    _quat_normalize_wxyz,
    _quat_rotate_inverse_wxyz,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006.runtime_contract import (  # noqa: E402
    load_runtime_contract,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006.state_contract import (  # noqa: E402
    canonical_bytes,
    compare_full_reset_to_e004,
    normalized_state_sha256,
    settled_gate,
    stage_ood,
)
from experiments.v3.phase_e.canonical_stage_localization_v3e006_r010.pinch_geometry import (  # noqa: E402
    pinch_alignment_command,
    reconstruct_collision_bounds_env_local,
    transform_collision_corners_env_local,
)
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.layout_contract import load_candidate  # noqa: E402
from experiments.v3.phase_e.symmetric_layout_cohort_v3e004.occlusion import (  # noqa: E402
    CameraEvidence,
    YawOrientedBox,
    evaluate_all_cameras,
    project_world_target_to_pixel,
)


CAMERAS = ("head_camera", "over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
MOVABLE = ("banana", "banana_right", "bowl", "rubiks_cube")
PERMITTED_CONTACTS = {
    "gripper__rubiks_cube",
    "banana__table",
    "banana_right__table",
    "bowl__table",
}
EXPECTED_CONTACT_SENSORS = {
    "gripper__rubiks_cube",
    "gripper__banana",
    "gripper__banana_right",
    "gripper__bowl",
    "gripper__table",
    "banana__rubiks_cube",
    "banana_right__rubiks_cube",
    "bowl__rubiks_cube",
    "rubiks_cube__table",
    "banana__banana_right",
    "banana__bowl",
    "banana__table",
    "banana_right__bowl",
    "banana_right__table",
    "bowl__table",
}


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha(resolved)}


def _retained_environment() -> dict[str, str | None]:
    return {
        key: os.environ.get(key)
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "VK_ICD_FILENAMES",
            "LD_LIBRARY_PATH",
            "HOME",
            "XDG_CACHE_HOME",
            "WARP_CACHE_PATH",
            "MPLCONFIGDIR",
            "TMPDIR",
            "PYTHONPATH",
        )
    }


def _base_evidence(*, candidate_gate_passed: bool = False, state_candidate_count: int = 0) -> dict[str, Any]:
    source = Path(__file__).resolve()
    return {
        "schema_version": "vla-wam-shared-v3e006-r010-state-repair-attempt-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E006-R010",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "state_candidate_count": state_candidate_count,
        "geometry_attachment_preflight_count": (
            1 if GEOMETRY_ATTACHMENT_PREFLIGHT is not None else 0
        ),
        "r010_live_diagnostic_count": DIAGNOSTIC_EVALUATION_COUNT,
        "repair_candidate_evaluation_count": CANDIDATE_EVALUATION_COUNT,
        "behavioral_denominator_included": False,
        "candidate_gate_passed": candidate_gate_passed,
        "construction_lifecycle_contract": lifecycle_contract,
        "construction_lifecycle_contract_sha256": lifecycle_digest,
        "geometry_attachment_preflight_contract": (
            geometry_attachment_preflight_contract
        ),
        "geometry_attachment_preflight_contract_sha256": (
            geometry_attachment_preflight_digest
        ),
        "failure_stage": CURRENT_STAGE,
        "invocation": sys.argv,
        "construction_source": {**_binding(source), "study_commit": args.expected_study_commit},
        "input_bindings": {
            "e004_candidate": _binding(args.e004_candidate),
            "ood_freeze": _binding(args.ood_freeze),
            "e004_full_reset_reference": _binding(args.e004_reset_reference),
            "runtime_contract": _binding(args.runtime_bindings),
            "repair_registration": _binding(args.repair_registration),
            "candidate_schedule": _binding(args.candidate_schedule),
            "original_v3e006_closure_binding": _binding(args.original_closure_binding),
            "r009_predecessor_results": _binding(args.predecessor_closure_binding),
            "source_push_gate": _binding(args.source_push_gate),
            "residual_correction_source": _binding(
                study_root
                / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r010/residual_correction.py"
            ),
            "pinch_geometry_source": _binding(
                study_root
                / "experiments/v3/phase_e/canonical_stage_localization_v3e006_r010/pinch_geometry.py"
            ),
            "robot_collision_usd": _binding(
                Path(construction_asset_bindings["robot_usd"]["path"])
            ),
            "cube_collision_usd": _binding(
                Path(construction_asset_bindings["cube_usd"]["path"])
            ),
            "geometry_oracle_sources": {
                name: _binding(Path(value["path"]))
                for name, value in geometry_oracle_source_bindings.items()
            },
            "control_scene_asset": _binding(args.control_scene_asset),
            "paired_scene_asset": _binding(args.paired_scene_asset),
            "frozen_e004_bounds_source": _binding(
                study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/model_blind_droid_gate.py"
            ),
        },
        "passed_health_preflight": {name: _binding(path) for name, (path, _digest) in health_files.items()},
        "historical_source_verification_before_AppLauncher": HISTORICAL_SOURCE_VERIFICATION,
        "controller_source_verification_before_AppLauncher": CONTROLLER_SOURCE_VERIFICATION,
        "runtime_log": {
            "path": str(args.runtime_log.resolve()),
            "binding_status": "rehash_after_process_exit_by_outer_ledger",
        },
        "environment": _retained_environment(),
        "lane": {
            "pod": args.pod,
            "pod_uid": args.pod_uid,
            "gpu_uuid": args.gpu_uuid,
            "container_image": args.container_image,
            "container_id": args.container_id,
            "driver_version": args.driver_version,
            "device": args.device,
            "python": sys.executable,
        },
        "last_reference_bounds_evidence": LAST_REFERENCE_BOUNDS_EVIDENCE,
        "partial_stage_evidence": LAST_PARTIAL_STAGES,
        "last_termination_evidence": LAST_TERMINATION_EVIDENCE,
        "environment_lifecycle": ENVIRONMENT_LIFECYCLE,
        "geometry_attachment_preflight": GEOMETRY_ATTACHMENT_PREFLIGHT,
    }


def _write_failure(exc: BaseException) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = {}
    for path in sorted(args.output_dir.rglob("*")):
        if path.is_file() and path.name != "state_construction_failure.json":
            available[str(path.relative_to(args.output_dir))] = _binding(path)
    report = {
        **_base_evidence(),
        "status": "infrastructure_invalid_r010_state_repair",
        "passed": False,
        "failure_exit_policy": (
            "after this report is atomically closed and streams are flushed, the failed child exits "
            "nonzero without SimulationApp.close; attempt01 proved that close can terminate Kit with "
            "status zero before Python propagates the retained exception"
        ),
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        },
        "available_raw_artifacts": available,
    }
    path = args.output_dir / "state_construction_failure.json"
    path.write_bytes(canonical_bytes(report))
    return path


def _persist_completed_evidence(
    *, diagnostics: list[dict[str, Any]], attempts: list[dict[str, Any]]
) -> Path:
    """Atomically retain completed zero-model work without changing selection."""

    path = args.output_dir / "completed_evidence.partial.json"
    temporary = path.with_suffix(".json.tmp")
    payload = {
        "schema_version": "vla-wam-shared-v3e006-r010-completed-evidence-v1",
        "amendment_id": "V3-E006-R010",
        "status": "diagnostic_only_incremental_retention_not_a_terminal_result",
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate_gate_passed": False,
        "selection_rule": candidate_schedule["selection_rule"],
        "completed_diagnostic_count": len(diagnostics),
        "completed_candidate_pair_count": len(attempts),
        "known_reachable_diagnostics": diagnostics,
        "attempts": attempts,
    }
    temporary.write_bytes(canonical_bytes(payload))
    os.replace(temporary, path)
    return path


def _host(value: Any) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    return [float(item) for item in array.reshape(-1)]


def _bool_host(value: Any) -> list[bool]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return [bool(item) for item in np.asarray(value).reshape(-1)]


def _termination_config_snapshot(env: Any, cfg: Any) -> dict[str, Any]:
    """Capture the complete termination contract without stepping the environment."""

    term_cfg = cfg.terminations.to_dict()
    if not isinstance(term_cfg, Mapping):
        raise RuntimeError("construction termination config is not a mapping")
    manager = getattr(env, "termination_manager", None)
    if manager is None:
        raise RuntimeError("construction environment has no termination manager")
    names = getattr(manager, "active_terms", None)
    if names is None:
        names = getattr(manager, "_term_names", None)
    if names is None:
        raise RuntimeError("construction termination manager exposes no active term names")
    active_terms = sorted(str(name) for name in names)
    if active_terms != ["success", "time_out"]:
        raise RuntimeError(f"construction termination terms differ: {active_terms}")
    serialized = json.dumps(
        term_cfg, allow_nan=False, default=str, sort_keys=True, separators=(",", ":")
    )
    return {
        "active_terms": active_terms,
        "termination_config": term_cfg,
        "termination_config_canonical_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }


def _activate_registered_construction_horizon(env: Any, cfg: Any) -> dict[str, Any]:
    """Apply R010's sole lifecycle change before any reset or physics step."""

    if cfg is not env.cfg:
        raise RuntimeError("returned construction config is not env.cfg")
    before = _termination_config_snapshot(env, cfg)
    step_dt = float(env.step_dt)
    original_episode_length_s = float(cfg.episode_length_s)
    original_max_episode_length = int(env.max_episode_length)
    episode_length_before = _host(env.episode_length_buf)
    common_step_before = int(env.common_step_counter)
    if (
        not math.isclose(step_dt, 1.0 / 15.0, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(original_episode_length_s, 30.0, rel_tol=0.0, abs_tol=1e-12)
        or original_max_episode_length != 450
        or common_step_before != 0
        or any(value != 0.0 for value in episode_length_before)
    ):
        raise RuntimeError("fresh construction environment does not have frozen E004 horizon/reset state")

    # ManagerBasedRLEnv.max_episode_length is derived dynamically as
    # ceil(cfg.episode_length_s / step_dt); do not touch its counter or terms.
    cfg.episode_length_s = 1800 * step_dt
    registered_episode_length_s = float(cfg.episode_length_s)
    registered_max_episode_length = int(env.max_episode_length)
    after = _termination_config_snapshot(env, cfg)
    if (
        registered_max_episode_length != 1800
        or not math.isclose(registered_episode_length_s, 120.0, rel_tol=0.0, abs_tol=1e-12)
        or before != after
        or int(env.common_step_counter) != common_step_before
        or _host(env.episode_length_buf) != episode_length_before
    ):
        raise RuntimeError("registered R010 construction horizon activation changed more than timeout length")
    return {
        "status": "registered_construction_timeout_extended_before_first_reset_or_step",
        "only_mutated_field": "env.cfg.episode_length_s",
        "step_dt_s": step_dt,
        "original_episode_length_s": original_episode_length_s,
        "registered_episode_length_s": registered_episode_length_s,
        "original_max_episode_length_steps": original_max_episode_length,
        "registered_max_episode_length_steps": registered_max_episode_length,
        "common_step_counter_before_and_after": common_step_before,
        "episode_length_buf_before_and_after": episode_length_before,
        "termination_contract_before": before,
        "termination_contract_after": after,
        "termination_config_byte_equal": before == after,
        "registered_worst_case_steps": 1695,
        "registered_margin_steps": 105,
        "behavioral_horizon_mutated": False,
    }


def _record_termination(
    env: Any,
    *,
    label: str,
    phase: str,
    phase_step_zero_based: int,
    terminated: Any,
    truncated: Any,
    extras: Any,
) -> dict[str, Any]:
    """Retain exact terminal signals before raising an infrastructure error."""

    global LAST_TERMINATION_EVIDENCE
    manager = getattr(env, "termination_manager", None)
    terms: dict[str, list[bool]] = {}
    if manager is not None:
        for name, values in manager.get_active_iterable_terms(env_idx=0):
            terms[str(name)] = [bool(value) for value in values]
    episode_length = getattr(env, "episode_length_buf", None)
    reset_buf = getattr(env, "reset_buf", None)
    snapshot = {
        "label": label,
        "phase": phase,
        "phase_step_zero_based": phase_step_zero_based,
        "phase_step_one_based": phase_step_zero_based + 1,
        "terminated_return": _bool_host(terminated),
        "truncated_return": _bool_host(truncated),
        "termination_terms": terms,
        "episode_length_buf_after_step_and_automatic_reset": (
            _host(episode_length) if episode_length is not None else None
        ),
        "reset_buf_after_step": _bool_host(reset_buf) if reset_buf is not None else None,
        "max_episode_length": int(getattr(env, "max_episode_length", -1)),
        "common_step_counter": int(getattr(env, "common_step_counter", -1)),
        "physics_dt_s": float(env.physics_dt) if hasattr(env, "physics_dt") else None,
        "step_dt_s": float(env.step_dt) if hasattr(env, "step_dt") else None,
        "extras_keys": sorted(str(key) for key in extras) if isinstance(extras, Mapping) else [],
    }
    LAST_TERMINATION_EVIDENCE = snapshot
    LAST_PARTIAL_STAGES["termination_event"] = snapshot
    return snapshot


def _qmul(left: Sequence[float], right: Sequence[float]) -> np.ndarray:
    return _quat_multiply_wxyz(left, right)


def _qrotate(quaternion: Sequence[float], vector: Sequence[float]) -> np.ndarray:
    q = _quat_normalize_wxyz(quaternion)
    w, xyz = float(q[0]), q[1:]
    v = np.asarray(vector, dtype=np.float64)
    return 2 * np.dot(xyz, v) * xyz + (w * w - np.dot(xyz, xyz)) * v + 2 * w * np.cross(xyz, v)


def _rotvec_to_quat(rotvec: Sequence[float]) -> np.ndarray:
    value = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(value))
    if angle <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0])
    return _quat_normalize_wxyz(np.concatenate(([math.cos(angle / 2)], value / angle * math.sin(angle / 2))))


def _command(position: Sequence[float], eef_quaternion: Sequence[float], grip: float, device: str) -> torch.Tensor:
    """Exact E004 eef_frame command used only for reset stabilization."""

    command_quaternion = _qmul(eef_quaternion, _quat_inverse_wxyz(np.asarray(EEF_OFFSET_ROT, dtype=np.float64)))
    action = np.concatenate((np.asarray(position), command_quaternion, [grip])).astype(np.float32)
    return torch.from_numpy(action).reshape(1, 8).to(device)


def _command_base_link(
    position: Sequence[float], base_link_quaternion: Sequence[float], grip: float, device: str
) -> torch.Tensor:
    """Command DroidIKActionCfg's controlled base_link pose directly in WXYZ."""

    action = np.concatenate(
        (
            np.asarray(position, dtype=np.float64),
            _quat_normalize_wxyz(base_link_quaternion),
            [float(grip)],
        )
    ).astype(np.float32)
    return torch.from_numpy(action).reshape(1, 8).to(device)


def _base_link_body_index(env: Any) -> int:
    names = [str(name) for name in env.scene["robot"].body_names]
    if names.count("base_link") != 1:
        raise RuntimeError(f"exactly one robot base_link body is required, found {names}")
    return names.index("base_link")


def _base_link_pose(env: Any) -> tuple[list[float], list[float]]:
    robot = env.scene["robot"].data
    index = _base_link_body_index(env)
    world_position = np.asarray(_host(robot.body_pos_w[0, index]), dtype=np.float64)
    env_origin = np.asarray(_host(env.scene.env_origins[0]), dtype=np.float64)
    return (world_position - env_origin).tolist(), _host(robot.body_quat_w[0, index])


def _cube_pose(env: Any) -> tuple[list[float], list[float]]:
    cube = env.scene["rubiks_cube"].data
    world_position = np.asarray(_host(cube.root_pos_w[0]), dtype=np.float64)
    env_origin = np.asarray(_host(env.scene.env_origins[0]), dtype=np.float64)
    return (world_position - env_origin).tolist(), _host(cube.root_quat_w[0])


def _frame_identity_evidence(env: Any, frames: Any, eef_index: int) -> dict[str, Any]:
    base_position, base_quaternion = _base_link_pose(env)
    live_eef_position_world = np.asarray(
        _host(frames.data.target_pos_w[0, eef_index]), dtype=np.float64
    )
    env_origin = np.asarray(_host(env.scene.env_origins[0]), dtype=np.float64)
    live_eef_position = live_eef_position_world - env_origin
    live_eef_quaternion = _host(frames.data.target_quat_w[0, eef_index])
    expected_eef_position = np.asarray(base_position, dtype=np.float64) + _qrotate(
        base_quaternion, EEF_OFFSET_POS
    )
    expected_eef_quaternion = _qmul(base_quaternion, EEF_OFFSET_ROT)
    position_residual = float(np.linalg.norm(live_eef_position - expected_eef_position))
    orientation_residual = _quaternion_geodesic_error_deg(
        live_eef_quaternion, expected_eef_quaternion
    )
    passed = bool(position_residual <= 1e-6 and orientation_residual <= 1e-4)
    return {
        "base_link_position_world_m": base_position,
        "base_link_quaternion_world_wxyz": base_quaternion,
        "live_eef_frame_position_world_m": live_eef_position.tolist(),
        "live_eef_frame_quaternion_world_wxyz": live_eef_quaternion,
        "registered_eef_offset_position_base_m": [float(value) for value in EEF_OFFSET_POS],
        "registered_eef_offset_quaternion_wxyz": [float(value) for value in EEF_OFFSET_ROT],
        "expected_eef_frame_position_world_m": expected_eef_position.tolist(),
        "expected_eef_frame_quaternion_world_wxyz": _host(expected_eef_quaternion),
        "position_composition_residual_m": position_residual,
        "orientation_composition_residual_deg": orientation_residual,
        "position_residual_m_inclusive": 1e-6,
        "orientation_residual_deg_inclusive": 1e-4,
        "passed": passed,
        "position_semantics": "env-local world-axis coordinates, matching RoboLab recorder subtraction of scene.env_origins",
        "scene_env_origin_world_m": env_origin.tolist(),
    }


def _quaternion_geodesic_error_deg(left: Sequence[float], right: Sequence[float]) -> float:
    a = _quat_normalize_wxyz(left)
    b = _quat_normalize_wxyz(right)
    dot = min(1.0, max(-1.0, abs(float(np.dot(a, b)))))
    return math.degrees(2.0 * math.acos(dot))


def _slerp_wxyz(left: Sequence[float], right: Sequence[float], fraction: float) -> np.ndarray:
    a = _quat_normalize_wxyz(left)
    b = _quat_normalize_wxyz(right)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 1.0 - 1e-12:
        return _quat_normalize_wxyz((1.0 - fraction) * a + fraction * b)
    theta = math.acos(dot)
    return _quat_normalize_wxyz(
        math.sin((1.0 - fraction) * theta) / math.sin(theta) * a
        + math.sin(fraction * theta) / math.sin(theta) * b
    )


def _construction_trace_row(
    env: Any,
    frames: Any,
    eef_index: int,
    *,
    phase: str,
    phase_step_one_based: int,
    action: torch.Tensor,
) -> dict[str, Any]:
    robot = env.scene["robot"].data
    cube = env.scene["rubiks_cube"].data
    base_position, base_quaternion = _base_link_pose(env)
    cube_position_env_local, _cube_quaternion = _cube_pose(env)
    frame_identity = _frame_identity_evidence(env, frames, eef_index)
    return {
        "phase": phase,
        "phase_step_one_based": phase_step_one_based,
        "command_action_8d": _host(action[0]),
        "eef_position_world_m": _host(frames.data.target_pos_w[0, eef_index]),
        "eef_quaternion_world_wxyz": _host(frames.data.target_quat_w[0, eef_index]),
        "base_link_position_world_m": base_position,
        "base_link_quaternion_world_wxyz": base_quaternion,
        "base_link_to_eef_frame_identity": frame_identity,
        "joint_position_rad": _host(robot.joint_pos[0]),
        "joint_velocity_rad_s": _host(robot.joint_vel[0]),
        "cube_position_world_m": _host(cube.root_pos_w[0]),
        "cube_position_env_local_m": cube_position_env_local,
        "cube_quaternion_world_wxyz": _host(cube.root_quat_w[0]),
        "cube_linear_velocity_m_s": _host(cube.root_lin_vel_w[0]),
        "cube_angular_velocity_rad_s": _host(cube.root_ang_vel_w[0]),
    }


def _joint_equilibrium_trace_row(
    env: Any,
    frames: Any,
    eef_index: int,
    *,
    phase: str,
    phase_step_one_based: int,
    joint_position_target: torch.Tensor,
) -> dict[str, Any]:
    """Retain one row from the registered normal-actuator equilibrium hold."""

    row = _construction_trace_row(
        env,
        frames,
        eef_index,
        phase=phase,
        phase_step_one_based=phase_step_one_based,
        action=torch.zeros((1, 8), dtype=torch.float32, device=env.device),
    )
    row.pop("command_action_8d")
    row["normal_joint_position_target_rad"] = _host(joint_position_target[0])
    row["cartesian_action_manager_applied"] = False
    return row


def _normal_joint_equilibrium_step(
    env: Any,
    *,
    joint_position_target: torch.Tensor,
    label: str,
    phase: str,
    phase_step_zero_based: int,
) -> tuple[Mapping[str, Any], dict[str, Any] | None]:
    """Advance one control interval under the exact fixed normal joint target.

    This mirrors ManagerBasedRLEnv's physics/termination/observation cadence but
    deliberately never processes or applies a Cartesian action.  It also never
    auto-resets a terminated construction environment.
    """

    is_rendering = env.sim.has_gui() or env.sim.has_rtx_sensors()
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        if env._sim_step_counter % env.cfg.sim.render_interval == 0 and is_rendering:
            env.sim.render()
        env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    reset = env.termination_manager.compute()
    terminated = env.termination_manager.terminated.clone()
    truncated = env.termination_manager.time_outs.clone()
    if bool(reset[0]):
        return env.observation_manager.compute(update_history=True), _record_termination(
            env,
            label=label,
            phase=phase,
            phase_step_zero_based=phase_step_zero_based,
            terminated=terminated,
            truncated=truncated,
            extras=env.extras,
        )
    return env.observation_manager.compute(update_history=True), None


def _step_checked(
    env: Any,
    action: torch.Tensor,
    *,
    label: str,
    phase: str,
    phase_step_zero_based: int,
) -> Mapping[str, Any]:
    obs, _, terminated, truncated, extras = env.step(action)
    if bool(terminated[0]) or bool(truncated[0]):
        evidence = _record_termination(
            env,
            label=label,
            phase=phase,
            phase_step_zero_based=phase_step_zero_based,
            terminated=terminated,
            truncated=truncated,
            extras=extras,
        )
        raise RuntimeError(
            f"environment terminated during {label}/{phase}; "
            f"terminal evidence retained at step {evidence['phase_step_one_based']}"
        )
    return obs


def _contact_forces(env: Any) -> dict[str, float]:
    rows: dict[str, float] = {}
    for name, sensor in sorted(get_contact_sensors(env.scene).items()):
        if name.endswith("__all_objs"):
            continue
        matrix = getattr(sensor.data, "force_matrix_w", None)
        raw = matrix if matrix is not None else getattr(sensor.data, "net_forces_w", None)
        if raw is None:
            raise RuntimeError(f"contact sensor {name} has no force stream")
        value = np.asarray(raw.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        rows[name] = float(np.max(np.linalg.norm(value, axis=1))) if value.size else 0.0
    return rows


def _contact_coverage(env: Any) -> dict[str, Any]:
    inventory = sorted(name for name in get_contact_sensors(env.scene) if not name.endswith("__all_objs"))
    missing = sorted(EXPECTED_CONTACT_SENSORS - set(inventory))
    extra = sorted(set(inventory) - EXPECTED_CONTACT_SENSORS)
    checks = {
        "complete_pairwise_sensor_inventory": not missing,
        "cube_gripper_sensor_present": "gripper__rubiks_cube" in inventory,
        "companion_table_sensors_present": {"banana__table", "banana_right__table"} <= set(inventory),
        "all_sensor_force_streams_live": set(_contact_forces(env)) == set(inventory),
    }
    if not all(checks.values()):
        raise RuntimeError(f"contact-sensor coverage failed closed: missing={missing}, extra={extra}")
    return {
        "inventory": inventory,
        "expected_inventory": sorted(EXPECTED_CONTACT_SENSORS),
        "missing": missing,
        "extra": extra,
        "checks": checks,
        "passed": True,
        "force_threshold_n": 1.0,
    }


def _sample(env: Any, frames: Any, eef_index: int) -> dict[str, Any]:
    cube = env.scene["rubiks_cube"].data
    robot = env.scene["robot"].data
    base_position, base_quaternion = _base_link_pose(env)
    return {
        "cube_position_world_m": _host(cube.root_pos_w[0]),
        "cube_linear_velocity_m_s": _host(cube.root_lin_vel_w[0]),
        "cube_angular_velocity_rad_s": _host(cube.root_ang_vel_w[0]),
        # The frozen OOD reference's HDF5 ee_pose field is robot/base_link.
        # Keep the historical feature key unchanged while supplying the same frame.
        "eef_position_world_m": base_position,
        "base_link_quaternion_world_wxyz": base_quaternion,
        "live_eef_frame_position_world_m": _host(frames.data.target_pos_w[0, eef_index]),
        "live_eef_frame_quaternion_world_wxyz": _host(frames.data.target_quat_w[0, eef_index]),
        "base_link_to_eef_frame_identity": _frame_identity_evidence(env, frames, eef_index),
        "arm_joint_velocity_rad_s": _host(robot.joint_vel[0])[:7],
        "object_grabbed": bool(object_grabbed(env, object="rubiks_cube", env_id=0)),
        "contact_force_n": _contact_forces(env),
    }


def _capture_state(
    env: Any,
    frames: Any,
    eef_index: int,
    *,
    gripper_command: float,
    contact_coverage: Mapping[str, Any],
    contact_samples: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    robot = env.scene["robot"].data
    objects: dict[str, Any] = {}
    for name in MOVABLE:
        data = env.scene[name].data
        objects[name] = {
            "position_world_m": _host(data.root_pos_w[0]),
            "quaternion_world_wxyz": _host(data.root_quat_w[0]),
            "linear_velocity_m_s": _host(data.root_lin_vel_w[0]),
            "angular_velocity_rad_s": _host(data.root_ang_vel_w[0]),
        }
    base_position, base_quaternion = _base_link_pose(env)
    frame_identity = _frame_identity_evidence(env, frames, eef_index)
    if not frame_identity["passed"]:
        raise RuntimeError("live base_link/eef_frame transform differs from frozen EEF offset")
    return {
        "robot": {
            "joint_names": [str(name) for name in env.scene["robot"].joint_names],
            "root_position_world_m": _host(robot.root_pos_w[0]),
            "root_quaternion_world_wxyz": _host(robot.root_quat_w[0]),
            "root_linear_velocity_m_s": _host(robot.root_lin_vel_w[0]),
            "root_angular_velocity_rad_s": _host(robot.root_ang_vel_w[0]),
            "joint_position_rad": _host(robot.joint_pos[0]),
            "joint_velocity_rad_s": _host(robot.joint_vel[0]),
            "gripper": {
                "joint_names": [str(name) for name in env.scene["robot"].joint_names[7:]],
                "joint_position_rad": _host(robot.joint_pos[0])[7:],
                "joint_velocity_rad_s": _host(robot.joint_vel[0])[7:],
                "normal_binary_command": float(gripper_command),
                "object_grabbed": bool(object_grabbed(env, object="rubiks_cube", env_id=0)),
            },
        },
        "objects": objects,
        "eef": {
            # Historical field name retained for byte-identical stage_ood math.
            # Its values now match the frozen reference recorder's base_link frame.
            "position_world_m": base_position,
            "quaternion_world_wxyz": base_quaternion,
            "semantic_frame": "robot/base_link (RoboLab HDF5 ee_pose recorder default)",
        },
        "eef_frame_diagnostic_only": {
            "position_world_m": _host(frames.data.target_pos_w[0, eef_index]),
            "quaternion_world_wxyz": _host(frames.data.target_quat_w[0, eef_index]),
        },
        "base_link_to_eef_frame_identity": frame_identity,
        "contact_evidence": {
            "coverage": dict(contact_coverage),
            "settled_force_snapshots_n": [dict(row["contact_force_n"]) for row in contact_samples],
            "object_grabbed_by_step": [bool(row["object_grabbed"]) for row in contact_samples],
        },
    }


def _physical_prim_path(asset: Any) -> str:
    for owner_name in ("root_physx_view", "_root_physx_view"):
        paths = getattr(getattr(asset, owner_name, None), "prim_paths", None)
        if paths:
            return str(paths[0])
    return str(asset.cfg.prim_path).replace("{ENV_REGEX_NS}", "/World/envs/env_0")


def _valid_descendants(root_path: str) -> list[Any]:
    """Return the valid USD prim subtree rooted at one materialized env_0 path."""

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"R010 geometry root prim is invalid: {root_path}")
    prefix = root_path.rstrip("/") + "/"
    rows = [root]
    rows.extend(
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix) and prim.IsValid()
    )
    return rows


def _resolve_unique_inner_finger_body(
    *, robot_root_path: str, suffix: str
) -> str:
    matches = sorted(
        str(prim.GetPath())
        for prim in _valid_descendants(robot_root_path)
        if str(prim.GetPath()).endswith("/" + suffix)
    )
    if len(matches) != 1:
        raise RuntimeError(
            f"R010 expected exactly one {suffix} body under {robot_root_path}; "
            f"found {matches}"
        )
    return matches[0]


def _enabled_collision_prim_paths(root_path: str) -> list[str]:
    """Resolve every enabled UsdPhysics collision shape below one bound body."""

    matches: list[str] = []
    for prim in _valid_descendants(root_path):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if enabled is False:
            continue
        matches.append(str(prim.GetPath()))
    matches.sort()
    if not matches:
        raise RuntimeError(f"R010 found no enabled collision prim under {root_path}")
    return matches


def _range_corners(minimum: np.ndarray, maximum: np.ndarray) -> list[Gf.Vec3d]:
    return [
        Gf.Vec3d(float(x), float(y), float(z))
        for x in (minimum[0], maximum[0])
        for y in (minimum[1], maximum[1])
        for z in (minimum[2], maximum[2])
    ]


def _collision_geometry_body_local(
    *, collision_prim_paths: Sequence[str], body_prim_path: str
) -> dict[str, Any]:
    """Resolve static collision corners once in the owning rigid body's frame.

    USD is used only here, before the registered action schedule begins.  Every
    dynamic bound used by the controller is later reconstructed from tensor
    poses and these retained body-local corners.
    """

    stage = omni.usd.get_context().get_stage()
    body = stage.GetPrimAtPath(body_prim_path)
    if not body or not body.IsValid():
        raise RuntimeError(f"R010 bound rigid-body prim changed: {body_prim_path}")
    if not body.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"R010 collision owner lacks RigidBodyAPI: {body_prim_path}")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    aggregate: list[list[float]] = []
    shapes: list[dict[str, Any]] = []
    for path in collision_prim_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            raise RuntimeError(f"R010 bound collision prim changed: {path}")
        cursor = prim
        owning_rigid_body_paths: list[str] = []
        while cursor and cursor.IsValid():
            if cursor.HasAPI(UsdPhysics.RigidBodyAPI):
                owning_rigid_body_paths.append(str(cursor.GetPath()))
            if str(cursor.GetPath()) == body_prim_path:
                break
            cursor = cursor.GetParent()
        if owning_rigid_body_paths != [body_prim_path]:
            raise RuntimeError(
                "R010 collision crosses a nested/different rigid-body boundary: "
                f"{path} owners={owning_rigid_body_paths} expected={[body_prim_path]}"
            )
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            raise RuntimeError(f"R010 bound collision prim became disabled: {path}")
        # ComputeRelativeBound already returns the collision bound in the
        # owning rigid body's coordinate system.  Its aligned range is the
        # only body-local geometry retained by R010.  In particular, do not
        # apply the collision prim transform a second time (the R009 defect).
        aligned = bbox_cache.ComputeRelativeBound(prim, body).ComputeAlignedRange()
        minimum = np.asarray(
            [float(aligned.GetMin()[axis]) for axis in range(3)], dtype=np.float64
        )
        maximum = np.asarray(
            [float(aligned.GetMax()[axis]) for axis in range(3)], dtype=np.float64
        )
        if (
            not np.all(np.isfinite(np.concatenate((minimum, maximum))))
            or np.any(maximum <= minimum)
        ):
            raise RuntimeError(f"R010 collision prim has invalid body-relative bounds: {path}")
        corners_body = []
        for corner in _range_corners(minimum, maximum):
            row = [float(corner[axis]) for axis in range(3)]
            if not np.all(np.isfinite(row)):
                raise RuntimeError(f"R010 collision corner is nonfinite: {path}")
            corners_body.append(row)
            aggregate.append(row)
        shapes.append(
            {
                "collision_prim_path": path,
                "compute_relative_bound_ancestor_prim_path": body_prim_path,
                "collision_prim_body_relative_aligned_minimum_m": minimum.tolist(),
                "collision_prim_body_relative_aligned_maximum_m": maximum.tolist(),
                "collision_corners_body_m": corners_body,
                "additional_prim_or_world_transform_after_relative_bound": False,
            }
        )
    corners = np.asarray(aggregate, dtype=np.float64)
    if corners.ndim != 2 or corners.shape[0] < 8 or corners.shape[1] != 3:
        raise RuntimeError("R010 aggregate body-local collision geometry is empty")
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    half = 0.5 * (maximum - minimum)
    if np.any(half <= 0.0):
        raise RuntimeError("R010 aggregate body-local collision geometry is empty")
    value = {
        "body_prim_path": body_prim_path,
        "body_has_rigid_body_api": True,
        "all_collision_prims_owned_by_exact_body_without_nested_boundary": True,
        "collision_prim_paths": list(collision_prim_paths),
        "shape_local_geometry": shapes,
        "collision_corners_body_m": corners.tolist(),
        "minimum_body_m": minimum.tolist(),
        "maximum_body_m": maximum.tolist(),
        "center_body_m": (0.5 * (minimum + maximum)).tolist(),
        "half_extents_body_m": half.tolist(),
        "extraction_api": (
            "UsdGeom.BBoxCache.ComputeRelativeBound(collision_prim, "
            "owning_rigid_body).ComputeAlignedRange"
        ),
        "additional_transform_after_compute_relative_bound": False,
        "derivation": (
            "one-time USD collision bounds computed directly relative to the exact "
            "owning rigid body before candidate actions; aligned-range corners are "
            "already body-local and receive no further prim/world transform"
        ),
    }
    value["canonical_sha256"] = hashlib.sha256(canonical_bytes(value)).hexdigest()
    return value


def _unique_robot_body_tensor_index(env: Any, body_name: str) -> int:
    names = [str(name) for name in env.scene["robot"].body_names]
    matches = [index for index, name in enumerate(names) if name == body_name]
    if len(matches) != 1:
        raise RuntimeError(
            f"R010 expected one tensor body named {body_name}; found {matches} in {names}"
        )
    return matches[0]


def _tensor_pose_evidence(
    *, raw_world_position: Sequence[float], quaternion: Sequence[float], env_origin: Sequence[float]
) -> dict[str, Any]:
    raw = np.asarray(raw_world_position, dtype=np.float64)
    origin = np.asarray(env_origin, dtype=np.float64)
    q = _quat_normalize_wxyz(quaternion)
    if raw.shape != (3,) or origin.shape != (3,) or not np.all(np.isfinite(np.concatenate((raw, origin, q)))):
        raise RuntimeError("R010 tensor pose is malformed")
    return {
        "position_tensor_world_m": raw.tolist(),
        "scene_env_origin_world_m": origin.tolist(),
        "position_env_local_m": (raw - origin).tolist(),
        "quaternion_world_wxyz": q.tolist(),
        "position_semantics": "env-local world-axis = tensor world position - scene env origin",
    }


def _resolve_pinch_scene_geometry(env: Any) -> dict[str, Any]:
    """Freeze the unique pad/cube collision inventories in this fresh environment."""

    robot_root = _materialize_single_env_prim_path(
        _physical_prim_path(env.scene["robot"]), num_envs=args.num_envs
    )
    cube_root = _materialize_single_env_prim_path(
        _physical_prim_path(env.scene["rubiks_cube"]), num_envs=args.num_envs
    )
    left_body = _resolve_unique_inner_finger_body(
        robot_root_path=robot_root,
        suffix=str(pinch_geometry_contract["left_inner_finger_body_suffix"]),
    )
    right_body = _resolve_unique_inner_finger_body(
        robot_root_path=robot_root,
        suffix=str(pinch_geometry_contract["right_inner_finger_body_suffix"]),
    )
    inventory = {
        "robot_root_prim_path": robot_root,
        "cube_root_prim_path": cube_root,
        "left_inner_finger_body_prim_path": left_body,
        "right_inner_finger_body_prim_path": right_body,
        "left_collision_prim_paths": _enabled_collision_prim_paths(left_body),
        "right_collision_prim_paths": _enabled_collision_prim_paths(right_body),
        "cube_collision_prim_paths": _enabled_collision_prim_paths(cube_root),
    }
    if left_body == right_body:
        raise RuntimeError("R010 inner-finger body resolution is not unique")
    left_name = left_body.rsplit("/", 1)[-1]
    right_name = right_body.rsplit("/", 1)[-1]
    inventory.update(
        {
            "left_robot_body_tensor_name": left_name,
            "left_robot_body_tensor_index": _unique_robot_body_tensor_index(env, left_name),
            "right_robot_body_tensor_name": right_name,
            "right_robot_body_tensor_index": _unique_robot_body_tensor_index(env, right_name),
            "cube_tensor_source": "rubiks_cube.data.root_pos_w/root_quat_w",
        }
    )
    local = {
        "left": _collision_geometry_body_local(
            collision_prim_paths=inventory["left_collision_prim_paths"],
            body_prim_path=left_body,
        ),
        "right": _collision_geometry_body_local(
            collision_prim_paths=inventory["right_collision_prim_paths"],
            body_prim_path=right_body,
        ),
        "cube": _collision_geometry_body_local(
            collision_prim_paths=inventory["cube_collision_prim_paths"],
            body_prim_path=cube_root,
        ),
    }
    value = {
        "inventory": inventory,
        "static_body_local_collision_geometry": local,
        "scene_env_origin_world_m_at_resolution": _host(env.scene.env_origins[0]),
        "dynamic_geometry_source": "IsaacLab tensor rigid-body/root poses minus explicit scene env origin",
        "dynamic_usd_world_bounds_used": False,
    }
    identity = {
        "inventory": inventory,
        "static_body_local_collision_geometry": local,
        "scene_env_origin_world_m_at_resolution": value[
            "scene_env_origin_world_m_at_resolution"
        ],
    }
    value["geometry_identity_sha256"] = hashlib.sha256(
        canonical_bytes(identity)
    ).hexdigest()
    return value


def _live_pinch_bounds(env: Any, geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct all live bounds from tensors; USD is never queried here."""

    inventory = geometry["inventory"]
    local = geometry["static_body_local_collision_geometry"]
    origin = _host(env.scene.env_origins[0])
    robot = env.scene["robot"].data
    cube = env.scene["rubiks_cube"].data
    poses = {
        "left": _tensor_pose_evidence(
            raw_world_position=_host(
                robot.body_pos_w[0, int(inventory["left_robot_body_tensor_index"])]
            ),
            quaternion=_host(
                robot.body_quat_w[0, int(inventory["left_robot_body_tensor_index"])]
            ),
            env_origin=origin,
        ),
        "right": _tensor_pose_evidence(
            raw_world_position=_host(
                robot.body_pos_w[0, int(inventory["right_robot_body_tensor_index"])]
            ),
            quaternion=_host(
                robot.body_quat_w[0, int(inventory["right_robot_body_tensor_index"])]
            ),
            env_origin=origin,
        ),
        "cube": _tensor_pose_evidence(
            raw_world_position=_host(cube.root_pos_w[0]),
            quaternion=_host(cube.root_quat_w[0]),
            env_origin=origin,
        ),
    }
    result: dict[str, Any] = {}
    for key in ("left", "right", "cube"):
        pose = poses[key]
        static = local[key]
        bounds = reconstruct_collision_bounds_env_local(
            body_position_env_local=pose["position_env_local_m"],
            body_quaternion_world_wxyz=pose["quaternion_world_wxyz"],
            collision_corners_body=static["collision_corners_body_m"],
            collision_center_body=static["center_body_m"],
        )
        result[key] = {
            "live_tensor_pose": pose,
            "static_body_local_geometry_sha256": static["canonical_sha256"],
            "reconstructed_bounds_env_local": bounds,
        }
    return result


def _run_geometry_attachment_preflight(
    env: Any,
    *,
    geometry: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    fresh_reset: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare tensor/body-relative reconstruction to a synchronized USD oracle.

    This dedicated preflight is the only R010 code path allowed to call
    ComputeWorldBound.  Its output is validation-only and is never supplied to
    the construction controller.
    """

    settings = carb.settings.get_settings()
    setting_path = str(
        geometry_attachment_preflight_contract["physics_to_usd_setting_path"]
    )
    if setting_path != str(SETTING_UPDATE_TO_USD):
        raise RuntimeError("R010 PhysX update-to-USD setting constant differs")
    if _host(env.episode_length_buf) != [75]:
        raise RuntimeError("R010 geometry oracle did not run after exact 75-step reset")
    origin = np.asarray(_host(env.scene.env_origins[0]), dtype=np.float64)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise RuntimeError("R010 geometry oracle env origin is malformed")
    # Snapshot every tensor pose before the one-shot writeback.  There is no
    # action or physics step between this snapshot and the oracle read.
    live = _live_pinch_bounds(env, geometry)
    setting_before = bool(settings.get_as_bool(setting_path))
    physx_interface = omni.physx.get_physx_interface()
    update_transformations = getattr(physx_interface, "update_transformations", None)
    if not callable(update_transformations):
        raise RuntimeError("R010 PhysX-to-USD synchronization API is unavailable")
    update_transformations(False, True, False, False)
    setting_after = bool(settings.get_as_bool(setting_path))
    if setting_after != setting_before:
        raise RuntimeError("R010 one-shot PhysX-to-USD sync mutated persistent setting")
    stage = omni.usd.get_context().get_stage()
    # Both caches are deliberately created after the one-shot writeback.
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_], useExtentsHint=False
    )
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    tolerance = float(
        geometry_attachment_preflight_contract["oracle_tolerance_m_inclusive"]
    )
    owner_rows: list[dict[str, Any]] = []
    owner_position_tolerance = float(
        geometry_attachment_preflight_contract[
            "owner_pose_tolerance_m_inclusive"
        ]
    )
    owner_orientation_tolerance = float(
        geometry_attachment_preflight_contract[
            "owner_orientation_tolerance_deg_inclusive"
        ]
    )
    for role in ("left", "right", "cube"):
        static = geometry["static_body_local_collision_geometry"][role]
        body_path = str(static["body_prim_path"])
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim or not body_prim.IsValid():
            raise RuntimeError(f"R010 geometry oracle body prim changed: {body_path}")
        matrix = xform_cache.GetLocalToWorldTransform(body_prim)
        usd_origin = np.asarray(
            [float(value) for value in matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))],
            dtype=np.float64,
        )
        tensor_pose = live[role]["live_tensor_pose"]
        tensor_origin = np.asarray(
            tensor_pose["position_tensor_world_m"], dtype=np.float64
        )
        position_error = float(np.max(np.abs(usd_origin - tensor_origin)))
        tensor_quaternion = tensor_pose["quaternion_world_wxyz"]
        axis_errors: list[float] = []
        usd_axes: list[list[float]] = []
        tensor_axes: list[list[float]] = []
        for axis in np.eye(3, dtype=np.float64):
            usd_axis = np.asarray(
                [float(value) for value in matrix.TransformDir(Gf.Vec3d(*axis))],
                dtype=np.float64,
            )
            usd_axis /= np.linalg.norm(usd_axis)
            tensor_axis = np.asarray(
                _qrotate(tensor_quaternion, axis), dtype=np.float64
            )
            tensor_axis /= np.linalg.norm(tensor_axis)
            axis_errors.append(
                math.degrees(
                    math.acos(float(np.clip(np.dot(usd_axis, tensor_axis), -1.0, 1.0)))
                )
            )
            usd_axes.append(usd_axis.tolist())
            tensor_axes.append(tensor_axis.tolist())
        orientation_error = max(axis_errors)
        owner_rows.append(
            {
                "role": role,
                "owning_rigid_body_prim_path": body_path,
                "live_tensor_body_pose": tensor_pose,
                "usd_world_origin_m": usd_origin.tolist(),
                "usd_world_axes": usd_axes,
                "tensor_world_axes": tensor_axes,
                "maximum_absolute_position_error_m": position_error,
                "maximum_axis_orientation_error_deg": orientation_error,
                "position_tolerance_m_inclusive": owner_position_tolerance,
                "orientation_tolerance_deg_inclusive": (
                    owner_orientation_tolerance
                ),
                "passed": (
                    position_error <= owner_position_tolerance
                    and orientation_error <= owner_orientation_tolerance
                ),
            }
        )
    rows: list[dict[str, Any]] = []
    for role in ("left", "right", "cube"):
        body_pose = live[role]["live_tensor_pose"]
        static = geometry["static_body_local_collision_geometry"][role]
        for shape in static["shape_local_geometry"]:
            collision_path = str(shape["collision_prim_path"])
            collision_prim = stage.GetPrimAtPath(collision_path)
            if not collision_prim or not collision_prim.IsValid():
                raise RuntimeError(
                    f"R010 geometry oracle collision prim changed: {collision_path}"
                )
            tensor_corners_env = transform_collision_corners_env_local(
                body_position_env_local=body_pose["position_env_local_m"],
                body_quaternion_world_wxyz=body_pose["quaternion_world_wxyz"],
                collision_corners_body=shape["collision_corners_body_m"],
            )
            tensor_min_world = np.min(tensor_corners_env, axis=0) + origin
            tensor_max_world = np.max(tensor_corners_env, axis=0) + origin
            oracle_range = bbox_cache.ComputeWorldBound(
                collision_prim
            ).ComputeAlignedRange()
            oracle_min_world = np.asarray(
                [float(oracle_range.GetMin()[axis]) for axis in range(3)],
                dtype=np.float64,
            )
            oracle_max_world = np.asarray(
                [float(oracle_range.GetMax()[axis]) for axis in range(3)],
                dtype=np.float64,
            )
            all_values = np.concatenate(
                (
                    tensor_min_world,
                    tensor_max_world,
                    oracle_min_world,
                    oracle_max_world,
                )
            )
            if not np.all(np.isfinite(all_values)):
                raise RuntimeError(
                    f"R010 geometry oracle produced nonfinite bounds: {collision_path}"
                )
            minimum_error = float(
                np.max(np.abs(tensor_min_world - oracle_min_world))
            )
            maximum_error = float(
                np.max(np.abs(tensor_max_world - oracle_max_world))
            )
            passed = minimum_error <= tolerance and maximum_error <= tolerance
            rows.append(
                {
                    "role": role,
                    "owning_rigid_body_prim_path": static["body_prim_path"],
                    "collision_prim_path": collision_path,
                    "body_relative_geometry": shape,
                    "live_tensor_body_pose": body_pose,
                    "tensor_reconstructed_corners_env_local_m": (
                        tensor_corners_env.tolist()
                    ),
                    "tensor_reconstructed_minimum_world_m": (
                        tensor_min_world.tolist()
                    ),
                    "tensor_reconstructed_maximum_world_m": (
                        tensor_max_world.tolist()
                    ),
                    "usd_compute_world_bound_minimum_world_m": (
                        oracle_min_world.tolist()
                    ),
                    "usd_compute_world_bound_maximum_world_m": (
                        oracle_max_world.tolist()
                    ),
                    "maximum_absolute_minimum_error_m": minimum_error,
                    "maximum_absolute_maximum_error_m": maximum_error,
                    "tolerance_m_inclusive": tolerance,
                    "passed": passed,
                }
            )
    expected_paths = [
        path
        for key in (
            "left_collision_prim_paths",
            "right_collision_prim_paths",
            "cube_collision_prim_paths",
        )
        for path in geometry["inventory"][key]
    ]
    observed_paths = [row["collision_prim_path"] for row in rows]
    passed = (
        observed_paths == expected_paths
        and bool(rows)
        and all(row["passed"] for row in rows)
        and len(owner_rows) == 3
        and all(row["passed"] for row in owner_rows)
    )
    return {
        "schema_version": (
            "vla-wam-shared-v3e006-r010-geometry-attachment-preflight-v1"
        ),
        "status": (
            "passed_r010_relative_bound_tensor_world_oracle_preflight"
            if passed
            else "failed_r010_relative_bound_tensor_world_oracle_preflight"
        ),
        "passed": passed,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "geometry_attachment_preflight_contract": (
            geometry_attachment_preflight_contract
        ),
        "geometry_attachment_preflight_contract_sha256": (
            geometry_attachment_preflight_digest
        ),
        "physics_to_usd_setting_path": setting_path,
        "physics_to_usd_setting_before_one_shot_sync": setting_before,
        "physics_to_usd_setting_after_one_shot_sync": setting_after,
        "physics_to_usd_setting_unchanged": setting_after == setting_before,
        "physics_to_usd_synchronization_call": (
            "omni.physx.get_physx_interface().update_transformations"
            "(False, True, False, False)"
        ),
        "physics_to_usd_synchronized_reset_steps": 75,
        "physics_or_action_steps_between_tensor_snapshot_and_oracle": 0,
        "usd_world_bounds_validation_only_not_controller_input": True,
        "environment_lifecycle": lifecycle,
        "fresh_reset": fresh_reset,
        "collision_geometry_resolution": geometry,
        "geometry_identity_sha256": geometry["geometry_identity_sha256"],
        "scene_env_origin_world_m": origin.tolist(),
        "collision_prim_count": len(rows),
        "expected_collision_prim_paths": expected_paths,
        "owner_pose_oracle_rows": owner_rows,
        "oracle_rows": rows,
        "all_inventory_paths_evaluated_once": observed_paths == expected_paths,
    }


def _materialize_single_env_prim_path(raw_path: str, *, num_envs: int) -> str:
    """Resolve RoboLab's single-environment regex without changing geometry."""
    if num_envs != 1:
        raise RuntimeError("bowl prim-path materialization is defined only for num_envs=1")
    if raw_path.count("env_.*/") > 1:
        raise RuntimeError(f"ambiguous environment regex in prim path: {raw_path}")
    if ".*" in raw_path and "/World/envs/env_.*/" not in raw_path:
        raise RuntimeError(f"unsupported environment regex in prim path: {raw_path}")
    resolved = raw_path.replace("/World/envs/env_.*/", "/World/envs/env_0/")
    if ".*" in resolved or "{" in resolved or "}" in resolved:
        raise RuntimeError(f"unresolved/ambiguous prim path: {raw_path}")
    return resolved


def _reference_bounds(env: Any) -> YawOrientedBox:
    global LAST_REFERENCE_BOUNDS_EVIDENCE
    asset = env.scene["bowl"]
    raw_path = _physical_prim_path(asset)
    prim_path = _materialize_single_env_prim_path(raw_path, num_envs=args.num_envs)
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    matches = [path for path in (prim_path,) if stage.GetPrimAtPath(path) and stage.GetPrimAtPath(path).IsValid()]
    if matches != [prim_path] or not prim or not prim.IsValid():
        LAST_REFERENCE_BOUNDS_EVIDENCE = {
            "raw_prim_path": raw_path,
            "resolved_prim_path": prim_path,
            "valid_matches": matches,
            "passed": False,
        }
        raise RuntimeError(f"expected exactly one valid bowl prim, found {matches}")
    local_bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeLocalBound(prim)
    local_range = local_bound.ComputeAlignedRange()
    minimum, maximum = local_range.GetMin(), local_range.GetMax()
    minimum_values = [float(minimum[index]) for index in range(3)]
    maximum_values = [float(maximum[index]) for index in range(3)]
    local_center = np.asarray(
        [(minimum_values[index] + maximum_values[index]) * 0.5 for index in range(3)], dtype=np.float64
    )
    half = tuple((maximum_values[index] - minimum_values[index]) * 0.5 for index in range(3))
    LAST_REFERENCE_BOUNDS_EVIDENCE = {
        "method": "frozen E004 _reference_bounds_world math after deterministic num_envs=1 regex materialization",
        "frozen_e004_source": _binding(
            study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/model_blind_droid_gate.py"
        ),
        "raw_prim_path": raw_path,
        "resolved_prim_path": prim_path,
        "valid_matches": matches,
        "local_minimum_m": minimum_values,
        "local_maximum_m": maximum_values,
        "local_center_m": local_center.tolist(),
        "half_extents_m": list(half),
        "passed": False,
    }
    if not all(math.isfinite(value) for value in (*minimum_values, *maximum_values, *local_center)):
        raise RuntimeError("bowl USD local range/center is nonfinite")
    if not all(math.isfinite(value) and value > 0 for value in half):
        raise RuntimeError("bowl USD local bound is invalid")
    position = np.asarray(_host(asset.data.root_pos_w[0]))
    quaternion = _host(asset.data.root_quat_w[0])
    center = position + _qrotate(quaternion, local_center)
    if not np.all(np.isfinite(center)):
        raise RuntimeError("bowl USD world center is nonfinite")
    w, x, y, z = _quat_normalize_wxyz(quaternion)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    LAST_REFERENCE_BOUNDS_EVIDENCE.update(
        {
            "center_world_m": center.tolist(),
            "yaw_world_rad": yaw,
            "passed": True,
        }
    )
    return YawOrientedBox(
        tuple(float(value) for value in center),
        tuple(float(value) for value in half),
        float(yaw),
    )


def _save_camera_evidence(env: Any, obs: Mapping[str, Any], stage: str, root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    cube = _host(env.scene["rubiks_cube"].data.root_pos_w[0])
    bounds = _reference_bounds(env)
    evidence: dict[str, CameraEvidence] = {}
    bindings: dict[str, Any] = {}
    for name in CAMERAS:
        image = np.asarray(obs["image_obs"][name][0].detach().cpu().numpy(), dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] != 3 or not np.ptp(image):
            raise RuntimeError(f"blank or malformed conditioning camera {name}")
        path = root / f"{stage}__{name}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"failed to retain camera image {path}")
        sensor = env.scene[name]
        center = _host(sensor.data.pos_w[0])
        quaternion = _host(sensor.data.quat_w_ros[0])
        intrinsic = np.asarray(sensor.data.intrinsic_matrices[0].detach().cpu().numpy(), dtype=np.float64).tolist()
        geometry = {
            "camera_center_world_m": center,
            "camera_quaternion_world_wxyz_ros": quaternion,
            "intrinsic_matrix_3x3": intrinsic,
            "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
            "extrinsics": {
                "translation_world_m": center,
                "quaternion_world_wxyz_ros": quaternion,
                "convention": "ROS camera axes: +X right, +Y down, +Z forward",
            },
        }
        geometry_sha = hashlib.sha256(canonical_bytes(geometry)).hexdigest()
        pixel = project_world_target_to_pixel(
            camera_center_world_m=center,
            camera_quaternion_world_wxyz_ros=quaternion,
            target_center_world_m=cube,
            intrinsic_matrix_3x3=intrinsic,
        )
        evidence[name] = CameraEvidence(
            camera_name=name,
            camera_center_world_m=tuple(center),
            target_center_world_m=tuple(cube),
            reference_bounds_world=bounds,
            target_instance_visible_pixels=None,
            segmentation_source_sha256=None,
            target_projected_pixel_uv=pixel,
            image_size_wh=(int(image.shape[1]), int(image.shape[0])),
            camera_geometry_source_sha256=geometry_sha,
        )
        bindings[name] = {
            **geometry,
            "target_projected_pixel_uv": list(pixel),
            "rgb": {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha(path)},
            "rgb_nonblank": True,
        }
    gate = evaluate_all_cameras(evidence, expected_cameras=CAMERAS, minimum_visible_target_pixels=32)
    return {
        "bindings": bindings,
        "reference_bounds_evidence": LAST_REFERENCE_BOUNDS_EVIDENCE,
        "gate": gate,
        "policy_conditioning_camera_feeds": {
            "observation/exterior_image_1_left": "over_shoulder_left_camera",
            "observation/wrist_image_left": "wrist_cam",
        },
        "all_policy_conditioning_feeds_retained": all(
            name in bindings for name in ("over_shoulder_left_camera", "wrist_cam")
        ),
        "passed": all(row["gate_passed"] for row in gate.values()),
    }


def _companion_gate(state: Mapping[str, Any], candidate: Any) -> dict[str, Any]:
    nominal = candidate.layout(1.0)
    observed: dict[str, Any] = {}
    passed = True
    for name in ("banana", "banana_right", "bowl"):
        actual = state["objects"][name]
        expected = nominal[name]
        position_error = float(np.linalg.norm(np.asarray(actual["position_world_m"]) - [expected.x_m, expected.y_m, expected.z_m]))
        q = _quat_normalize_wxyz(actual["quaternion_world_wxyz"])
        yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
        orientation_error = abs((yaw - expected.yaw_rad + math.pi) % (2 * math.pi) - math.pi)
        row_passed = position_error < candidate.realisation_position_tolerance_m and orientation_error < candidate.realisation_orientation_tolerance_rad
        observed[name] = {"position_error_m": position_error, "orientation_error_rad": orientation_error, "passed": row_passed}
        passed = passed and row_passed
    return {
        "observed": observed,
        "position_tolerance_m_strict": candidate.realisation_position_tolerance_m,
        "orientation_tolerance_rad_strict": candidate.realisation_orientation_tolerance_rad,
        "passed": passed,
    }


def _write_video(path: Path, frames: list[np.ndarray]) -> None:
    if not frames:
        raise RuntimeError("state construction video has no frames")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError("video writer failed to open")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _fresh_reset_and_gate(
    env: Any,
    *,
    candidate: Any,
    reset_reference: Mapping[str, Any],
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    global CURRENT_STAGE
    CURRENT_STAGE = f"{label}_fresh_e004_reset"
    obs, _ = env.reset()
    reset_position = _host(frames.data.target_pos_w[0, eef_index])
    reset_quaternion = _host(frames.data.target_quat_w[0, eef_index])
    reset_action = _command(reset_position, reset_quaternion, 0.0, env.device)
    reset_samples: list[dict[str, Any]] = []
    for step in range(75):
        obs, _, terminated, truncated, extras = env.step(reset_action)
        if bool(terminated[0]) or bool(truncated[0]):
            evidence = _record_termination(
                env,
                label=label,
                phase="fresh_e004_reset_settle",
                phase_step_zero_based=step,
                terminated=terminated,
                truncated=truncated,
                extras=extras,
            )
            raise RuntimeError(
                f"environment terminated during {label} E004 reset settle; "
                f"terminal evidence retained at step {evidence['phase_step_one_based']}"
            )
        if step % 12 == 0:
            video_frames.append(np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8))
        if step >= 65:
            reset_samples.append(_sample(env, frames, eef_index))
    reset_state = _capture_state(
        env,
        frames,
        eef_index,
        gripper_command=0.0,
        contact_coverage=contact_coverage,
        contact_samples=reset_samples,
    )
    reset_state["normalized_state_sha256"] = normalized_state_sha256(reset_state)
    reset_state["camera_evidence"] = _save_camera_evidence(
        env, obs, f"{label}__full_reset", args.output_dir / "cameras"
    )
    reset_state["companion_pose_gate"] = _companion_gate(reset_state, candidate)
    reset_state["e004_full_reset_comparison"] = compare_full_reset_to_e004(
        reset_state,
        reference=reset_reference,
        reference_file_sha256=args.e004_reset_reference_sha256,
    )
    reset_state["passed"] = all(
        (
            reset_state["camera_evidence"]["passed"],
            reset_state["companion_pose_gate"]["passed"],
            reset_state["e004_full_reset_comparison"]["passed"],
            contact_coverage["passed"],
        )
    )
    if not reset_state["passed"]:
        raise RuntimeError(f"{label} fresh reset differs from retained exact E004 s=1 reset")
    return obs, reset_state


def _write_registered_historical_state(
    env: Any,
    *,
    source: Mapping[str, Any],
    frames: Any,
    eef_index: int,
) -> dict[str, Any]:
    """Atomically restore a frozen historical contact state without integration."""

    joints = np.asarray(source["joint_position_rad"], dtype=np.float64)
    cube_pose_world = np.asarray(source["cube_pose_world_wxyz"], dtype=np.float64)
    position_key = (
        "base_link_position_world_m"
        if "base_link_position_world_m" in source
        else "eef_position_world_m"
    )
    quaternion_key = (
        "base_link_quaternion_world_wxyz"
        if "base_link_quaternion_world_wxyz" in source
        else "eef_quaternion_world_wxyz"
    )
    target_base_position = np.asarray(source[position_key], dtype=np.float64)
    target_base_quaternion = _quat_normalize_wxyz(source[quaternion_key])
    if position_key != "base_link_position_world_m" or quaternion_key != "base_link_quaternion_world_wxyz":
        raise RuntimeError("R010 historical state lacks explicit base_link pose semantics")
    if joints.shape != (13,) or cube_pose_world.shape != (7,):
        raise RuntimeError("registered historical state dimensions differ from exact E004")
    if not np.all(
        np.isfinite(
            np.concatenate((joints, cube_pose_world, target_base_position, target_base_quaternion))
        )
    ):
        raise RuntimeError("registered historical state contains a nonfinite value")

    robot = env.scene["robot"]
    cube = env.scene["rubiks_cube"]
    joint_position = torch.tensor(joints, dtype=torch.float32, device=env.device).reshape(1, -1)
    joint_velocity = torch.zeros_like(joint_position)
    cube_pose = torch.tensor(cube_pose_world, dtype=torch.float32, device=env.device).reshape(1, 7)
    cube_velocity = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.set_joint_position_target(joint_position)
    robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=env.device))
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(cube_velocity)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)

    actual_base_position_raw, actual_base_quaternion = _base_link_pose(env)
    actual_base_position = np.asarray(actual_base_position_raw, dtype=np.float64)
    actual_joints = np.asarray(_host(robot.data.joint_pos[0]), dtype=np.float64)
    actual_joint_velocity = np.asarray(_host(robot.data.joint_vel[0]), dtype=np.float64)
    actual_cube_pose = np.concatenate(
        (
            np.asarray(_host(cube.data.root_pos_w[0]), dtype=np.float64),
            np.asarray(_host(cube.data.root_quat_w[0]), dtype=np.float64),
        )
    )
    actual_cube_velocity = np.concatenate(
        (
            np.asarray(_host(cube.data.root_lin_vel_w[0]), dtype=np.float64),
            np.asarray(_host(cube.data.root_ang_vel_w[0]), dtype=np.float64),
        )
    )
    soft_limits = np.asarray(
        robot.data.soft_joint_pos_limits[0].detach().cpu().numpy(), dtype=np.float64
    )[:7]
    arm = actual_joints[:7]
    finite = bool(
        np.all(
            np.isfinite(
                np.concatenate(
                    (
                        actual_joints,
                        actual_joint_velocity,
                        actual_cube_pose,
                        actual_cube_velocity,
                        actual_base_position,
                        np.asarray(actual_base_quaternion, dtype=np.float64),
                    )
                )
            )
        )
    )
    inside_limits = bool(
        finite
        and soft_limits.shape == (7, 2)
        and np.all(arm >= soft_limits[:, 0])
        and np.all(arm <= soft_limits[:, 1])
    )
    evidence = {
        "registered_joint_position_rad": joints.tolist(),
        "registered_cube_pose_world_wxyz": cube_pose_world.tolist(),
        "registered_base_link_pose_world_wxyz": np.concatenate(
            (target_base_position, target_base_quaternion)
        ).tolist(),
        "registered_all_velocities": "exact_zero",
        "actual_joint_position_rad_after_forward": actual_joints.tolist(),
        "actual_joint_velocity_rad_s_after_forward": actual_joint_velocity.tolist(),
        "actual_cube_pose_world_wxyz_after_forward": actual_cube_pose.tolist(),
        "actual_cube_velocity_after_forward": actual_cube_velocity.tolist(),
        "actual_base_link_pose_world_wxyz_after_forward": np.concatenate(
            (actual_base_position, np.asarray(actual_base_quaternion, dtype=np.float64))
        ).tolist(),
        "base_link_position_error_m_after_forward": float(
            np.linalg.norm(actual_base_position - target_base_position)
        ),
        "base_link_orientation_geodesic_error_deg_after_forward": _quaternion_geodesic_error_deg(
            actual_base_quaternion, target_base_quaternion
        ),
        "base_link_to_eef_frame_identity": _frame_identity_evidence(env, frames, eef_index),
        "soft_joint_limits_rad": soft_limits.tolist(),
        "finite": finite,
        "arm_inside_soft_joint_limits": inside_limits,
        "physics_steps_after_atomic_write": 0,
    }
    if not evidence["base_link_to_eef_frame_identity"]["passed"]:
        raise RuntimeError("historical initialization base_link/eef_frame identity failed")
    if "expected_eef_frame_position_world_m" in source:
        expected_eef_position = np.asarray(
            source["expected_eef_frame_position_world_m"], dtype=np.float64
        )
        expected_eef_quaternion = source["expected_eef_frame_quaternion_world_wxyz"]
        observed_identity = evidence["base_link_to_eef_frame_identity"]
        evidence["registered_expected_eef_frame_comparison"] = {
            "position_residual_m": float(
                np.linalg.norm(
                    np.asarray(
                        observed_identity["expected_eef_frame_position_world_m"],
                        dtype=np.float64,
                    )
                    - expected_eef_position
                )
            ),
            "orientation_residual_deg": _quaternion_geodesic_error_deg(
                observed_identity["expected_eef_frame_quaternion_world_wxyz"],
                expected_eef_quaternion,
            ),
        }
        if (
            evidence["registered_expected_eef_frame_comparison"]["position_residual_m"]
            > 1e-12
            or evidence["registered_expected_eef_frame_comparison"][
                "orientation_residual_deg"
            ]
            > 1e-9
        ):
            raise RuntimeError("registered base_link-to-eef_frame composition changed")
    if not finite or not inside_limits:
        raise RuntimeError("historical initialization is nonfinite or outside live soft joint limits")
    return evidence


def _run_registered_pose_hold(
    env: Any,
    *,
    target_position: Sequence[float],
    target_quaternion: Sequence[float],
    hold_steps: int,
    required_final_steps: int,
    frames: Any,
    eef_index: int,
    label: str,
    phase: str,
    video_frames: list[np.ndarray],
) -> dict[str, Any]:
    """Apply the one registered symmetric measured-residual correction loop."""

    global CURRENT_STAGE
    desired_position = np.asarray(target_position, dtype=np.float64)
    desired_quaternion = _quat_normalize_wxyz(target_quaternion)
    if desired_position.shape != (3,) or not np.all(np.isfinite(desired_position)):
        raise RuntimeError("registered pose-hold target is malformed")
    if hold_steps != 30 or required_final_steps != 10:
        raise RuntimeError("R010 pose-hold step contract differs")
    contract = candidate_schedule["residual_correction_contract"]
    maximum_rounds = int(contract["maximum_correction_rounds"])
    command_position = desired_position.copy()
    command_quaternion = desired_quaternion.copy()
    errors: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    termination: dict[str, Any] | None = None
    all_finite = True
    all_inside_limits = True
    all_frame_identity = True
    soft_limits: np.ndarray | None = None
    final_window_passed = False
    CURRENT_STAGE = f"{label}_{phase}"
    for correction_round in range(1, maximum_rounds + 1):
        action = _command_base_link(command_position, command_quaternion, 1.0, env.device)
        round_errors: list[dict[str, Any]] = []
        round_trace: list[dict[str, Any]] = []
        for step in range(hold_steps):
            obs, _, terminated, truncated, extras = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                termination = _record_termination(
                    env,
                    label=label,
                    phase=f"{phase}_correction_round_{correction_round:02d}",
                    phase_step_zero_based=step,
                    terminated=terminated,
                    truncated=truncated,
                    extras=extras,
                )
                break
            actual_position_raw, actual_quaternion_raw = _base_link_pose(env)
            actual_position = np.asarray(actual_position_raw, dtype=np.float64)
            actual_quaternion = np.asarray(actual_quaternion_raw, dtype=np.float64)
            robot = env.scene["robot"]
            joints = np.asarray(_host(robot.data.joint_pos[0]), dtype=np.float64)
            soft_limits = np.asarray(
                robot.data.soft_joint_pos_limits[0].detach().cpu().numpy(), dtype=np.float64
            )[:7]
            frame_identity = _frame_identity_evidence(env, frames, eef_index)
            row_finite = bool(
                np.all(np.isfinite(np.concatenate((actual_position, actual_quaternion, joints))))
            )
            row_inside = bool(
                row_finite
                and soft_limits.shape == (7, 2)
                and np.all(joints[:7] >= soft_limits[:, 0])
                and np.all(joints[:7] <= soft_limits[:, 1])
            )
            all_finite = all_finite and row_finite
            all_inside_limits = all_inside_limits and row_inside
            all_frame_identity = all_frame_identity and bool(frame_identity["passed"])
            error_row = {
                "correction_round_one_based": correction_round,
                "round_step_one_based": step + 1,
                "step_one_based": len(errors) + 1,
                "desired_target_position_world_m": desired_position.tolist(),
                "desired_target_quaternion_world_wxyz": desired_quaternion.tolist(),
                "command_position_world_m": command_position.tolist(),
                "command_quaternion_world_wxyz": command_quaternion.tolist(),
                "measured_position_world_m": actual_position.tolist(),
                "measured_quaternion_world_wxyz": actual_quaternion.tolist(),
                "position_error_m": float(np.linalg.norm(actual_position - desired_position)),
                "orientation_geodesic_error_deg": _quaternion_geodesic_error_deg(
                    actual_quaternion, desired_quaternion
                ),
                "finite": row_finite,
                "arm_inside_soft_joint_limits": row_inside,
                "base_link_to_eef_frame_identity": frame_identity,
            }
            round_errors.append(error_row)
            errors.append(error_row)
            trace_row = _construction_trace_row(
                env,
                frames,
                eef_index,
                phase=f"{phase}_correction_round_{correction_round:02d}",
                phase_step_one_based=step + 1,
                action=action,
            )
            round_trace.append(trace_row)
            trace.append(trace_row)
            if step % 10 == 0:
                video_frames.append(
                    np.asarray(
                        obs["image_obs"]["head_camera"][0].detach().cpu().numpy(),
                        dtype=np.uint8,
                    )
                )
        final_window = round_errors[-required_final_steps:]
        final_window_passed = bool(
            len(final_window) == required_final_steps
            and all(
                row["position_error_m"] <= 0.001
                and row["orientation_geodesic_error_deg"] <= 1.0
                for row in final_window
            )
        )
        round_record: dict[str, Any] = {
            "round_one_based": correction_round,
            "desired_target_position_world_m": desired_position.tolist(),
            "desired_target_quaternion_world_wxyz": desired_quaternion.tolist(),
            "command_position_world_m": command_position.tolist(),
            "command_quaternion_world_wxyz": command_quaternion.tolist(),
            "command_action_8d": _host(action[0]),
            "completed_steps": len(round_errors),
            "final_window_passed": final_window_passed,
            "errors": round_errors,
            "construction_action_trace": round_trace,
            "termination": termination,
        }
        unsafe = bool(
            termination is not None
            or not all_finite
            or not all_inside_limits
            or not all_frame_identity
            or not round_errors
        )
        if not final_window_passed and not unsafe and correction_round < maximum_rounds:
            last = round_errors[-1]
            correction = corrected_command(
                desired_position=desired_position,
                desired_quaternion=desired_quaternion,
                measured_position=last["measured_position_world_m"],
                measured_quaternion=last["measured_quaternion_world_wxyz"],
                current_command_position=command_position,
                current_command_quaternion=command_quaternion,
                translation_gain=float(contract["translation_gain"]),
                rotation_gain=float(contract["rotation_gain"]),
            )
            round_record["measured_residual_correction"] = correction
            command_position = np.asarray(
                correction["next_command_position_world_m"], dtype=np.float64
            )
            command_quaternion = np.asarray(
                correction["next_command_quaternion_world_wxyz"], dtype=np.float64
            )
        else:
            round_record["measured_residual_correction"] = None
        rounds.append(round_record)
        if final_window_passed or unsafe:
            break
    passed = bool(
        termination is None
        and all_finite
        and all_inside_limits
        and all_frame_identity
        and final_window_passed
    )
    return {
        "passed": passed,
        "failure_reason": (
            None
            if passed
            else "environment_terminated"
            if termination is not None
            else "nonfinite_state"
            if not all_finite
            else "arm_outside_live_soft_joint_limits"
            if not all_inside_limits
            else "base_link_to_eef_frame_identity_failed"
            if not all_frame_identity
            else "final_ten_steps_outside_registered_pose_tolerance"
        ),
        "phase": phase,
        "target_base_link_pose_world_wxyz": np.concatenate(
            (desired_position, desired_quaternion)
        ).tolist(),
        "desired_target_invariant_across_rounds": all(
            row["desired_target_position_world_m"] == desired_position.tolist()
            and row["desired_target_quaternion_world_wxyz"] == desired_quaternion.tolist()
            for row in rounds
        ),
        "initial_command_action_8d": rounds[0]["command_action_8d"] if rounds else None,
        "final_command_action_8d": rounds[-1]["command_action_8d"] if rounds else None,
        "command_action_8d": rounds[-1]["command_action_8d"] if rounds else None,
        "command_frame_conversion": "commanded base_link quaternion is sent directly, WXYZ",
        "residual_correction_contract": contract,
        "residual_correction_contract_sha256": candidate_schedule[
            "residual_correction_contract_sha256"
        ],
        "maximum_correction_rounds": maximum_rounds,
        "completed_correction_rounds": len(rounds),
        "hold_steps": hold_steps,
        "hold_steps_per_round": hold_steps,
        "completed_steps": len(errors),
        "required_final_consecutive_steps": required_final_steps,
        "position_error_m_inclusive": 0.001,
        "orientation_geodesic_error_deg_inclusive": 1.0,
        "final_window_passed": final_window_passed,
        "all_states_finite": all_finite,
        "all_arm_states_inside_live_soft_joint_limits": all_inside_limits,
        "all_base_link_to_eef_frame_identity_checks_passed": all_frame_identity,
        "soft_joint_limits_rad": soft_limits.tolist() if soft_limits is not None else None,
        "errors": errors,
        "correction_rounds": rounds,
        "termination": termination,
        "construction_action_trace": trace,
    }


def _run_known_reachable_diagnostic(
    env: Any,
    *,
    diagnostic: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> dict[str, Any]:
    source = diagnostic["source"]
    initialized = _write_registered_historical_state(
        env, source=source, frames=frames, eef_index=eef_index
    )
    hold = _run_registered_pose_hold(
        env,
        target_position=source["base_link_position_world_m"],
        target_quaternion=source["base_link_quaternion_world_wxyz"],
        hold_steps=int(diagnostic["hold_steps"]),
        required_final_steps=int(diagnostic["required_final_consecutive_steps"]),
        frames=frames,
        eef_index=eef_index,
        label=label,
        phase="known_reachable_historical_pose_hold",
        video_frames=video_frames,
    )
    return {
        "passed": hold["passed"],
        "diagnostic_index_one_based": diagnostic["diagnostic_index_one_based"],
        "stage": diagnostic["stage"],
        "source_side": diagnostic["source_side"],
        "registered_diagnostic": diagnostic,
        "historical_state_initialization": initialized,
        "pose_hold": hold,
    }


def _solve_registered_ik(
    env: Any,
    *,
    schedule_stage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> dict[str, Any]:
    """Run the frozen historical-q initialization and eight registered waypoints."""

    initialization = schedule_stage["r004_solver_initialization"]
    source_side = str(initialization["source_side"])
    source = schedule_stage["both_direction_sources"][source_side]
    if (
        list(initialization["exact_historical_joint_position_rad"])
        != list(source["joint_position_rad"])
        or list(initialization["exact_historical_cube_pose_world_wxyz"])
        != list(source["cube_pose_world_wxyz"])
        or list(initialization["exact_historical_base_link_pose_world_wxyz"])
        != list(source["base_link_position_world_m"])
        + list(source["base_link_quaternion_world_wxyz"])
        or initialization.get("all_robot_and_cube_velocities") != "exact_zero"
        or initialization.get("gripper_command") != 1.0
        or initialization.get("residual_correction_contract_sha256")
        != candidate_schedule["residual_correction_contract_sha256"]
    ):
        raise RuntimeError("R010 solver initialization differs from its frozen historical source")
    initialized = _write_registered_historical_state(
        env, source=source, frames=frames, eef_index=eef_index
    )
    waypoint_results: list[dict[str, Any]] = []
    for waypoint in initialization["waypoints"]:
        expected_index = len(waypoint_results) + 1
        if (
            waypoint.get("waypoint_index_one_based") != expected_index
            or waypoint.get("fraction") != expected_index / 8.0
            or waypoint.get("hold_steps") != 30
            or waypoint.get("required_final_consecutive_steps") != 10
            or waypoint.get("position_error_m_inclusive") != 0.001
            or waypoint.get("orientation_geodesic_error_deg_inclusive") != 1.0
            or waypoint.get("maximum_correction_rounds") != 3
            or waypoint.get("r004_residual_correction_contract_sha256")
            != candidate_schedule["residual_correction_contract_sha256"]
        ):
            raise RuntimeError("R010 waypoint schedule differs from the registered finite solver")
        result = _run_registered_pose_hold(
            env,
            target_position=waypoint["position_world_m"],
            target_quaternion=waypoint["quaternion_world_wxyz"],
            hold_steps=30,
            required_final_steps=10,
            frames=frames,
            eef_index=eef_index,
            label=label,
            phase=f"registered_waypoint_{expected_index:02d}_of_08",
            video_frames=video_frames,
        )
        waypoint_results.append(result)
        if not result["passed"]:
            break

    robot = env.scene["robot"]
    arm_solution = np.asarray(_host(robot.data.joint_pos[0])[:7], dtype=np.float64)
    soft_limits = np.asarray(
        robot.data.soft_joint_pos_limits[0].detach().cpu().numpy(), dtype=np.float64
    )[:7]
    finite = bool(np.all(np.isfinite(arm_solution)))
    inside_limits = bool(
        finite
        and soft_limits.shape == (7, 2)
        and np.all(arm_solution >= soft_limits[:, 0])
        and np.all(arm_solution <= soft_limits[:, 1])
    )
    passed = bool(
        len(waypoint_results) == 8
        and all(result["passed"] for result in waypoint_results)
        and finite
        and inside_limits
    )
    return {
        "passed": passed,
        "failure_reason": (
            None
            if passed
            else next(
                (
                    f"waypoint_{index:02d}_{result['failure_reason']}"
                    for index, result in enumerate(waypoint_results, start=1)
                    if not result["passed"]
                ),
                "solver_did_not_complete_all_eight_waypoints",
            )
            if len(waypoint_results) < 8 or any(not row["passed"] for row in waypoint_results)
            else "ik_solution_nonfinite_or_outside_soft_joint_limits"
        ),
        "solver_topology": "historical_q_seed_then_eight_registered_abs_ik_waypoints_with_symmetric_measured_residual_correction",
        "historical_source_side": source_side,
        "historical_state_initialization": initialized,
        "registered_initialization": initialization,
        "target": schedule_stage["centerline_constrained_base_link_ik_target"],
        "waypoint_count": 8,
        "completed_waypoint_count": len(waypoint_results),
        "waypoint_results": waypoint_results,
        "maximum_steps": 720,
        "completed_steps": sum(int(row["completed_steps"]) for row in waypoint_results),
        "arm_joint_solution_rad": arm_solution.tolist(),
        "soft_joint_limits_rad": soft_limits.tolist(),
        "solution_finite": finite,
        "solution_inside_soft_joint_limits": inside_limits,
        "construction_action_trace": [
            trace
            for result in waypoint_results
            for trace in result["construction_action_trace"]
        ],
    }


def _target_pose_arrays(schedule_stage: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_eef = schedule_stage["centerline_constrained_base_link_ik_target"]
    target_cube = schedule_stage["target_cube_pose"]
    eef_position = np.asarray(target_eef["position_world_m"], dtype=np.float64)
    eef_quaternion = _quat_normalize_wxyz(target_eef["quaternion_world_wxyz"])
    cube_position = np.asarray(target_cube["position_world_m"], dtype=np.float64)
    cube_quaternion = _quat_normalize_wxyz(target_cube["quaternion_world_wxyz"])
    if not np.all(np.isfinite(np.concatenate((eef_position, eef_quaternion, cube_position, cube_quaternion)))):
        raise RuntimeError("registered target pose contains a nonfinite value")
    return eef_position, eef_quaternion, cube_position, cube_quaternion


def _finalize_unchanged_gates(
    env: Any,
    *,
    obs: Mapping[str, Any],
    stage_name: str,
    schedule_stage: Mapping[str, Any],
    stage_reference: Mapping[str, Any],
    candidate: Any,
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    settled: Sequence[Mapping[str, Any]],
    construction: Mapping[str, Any],
) -> dict[str, Any]:
    global CURRENT_STAGE
    if len(settled) != 10:
        raise RuntimeError("unchanged scientific gate requires exactly 10 settled samples")
    sensor_names = set(_contact_forces(env))
    unintended = sorted(sensor_names - PERMITTED_CONTACTS)
    CURRENT_STAGE = f"{label}_unchanged_gates"
    state = _capture_state(
        env,
        frames,
        eef_index,
        gripper_command=1.0,
        contact_coverage=contact_coverage,
        contact_samples=settled,
    )
    state["physics_gate"] = settled_gate(settled, unintended_contact_pairs=unintended)
    state["ood_gate"] = stage_ood(state, stage_reference=stage_reference)
    state["camera_evidence"] = _save_camera_evidence(env, obs, label, args.output_dir / "cameras")
    state["companion_pose_gate"] = _companion_gate(state, candidate)
    actual_base_p = np.asarray(state["eef"]["position_world_m"], dtype=np.float64)
    actual_base_q = _quat_normalize_wxyz(state["eef"]["quaternion_world_wxyz"])
    actual_cube_p = np.asarray(state["objects"]["rubiks_cube"]["position_world_m"], dtype=np.float64)
    actual_cube_q = _quat_normalize_wxyz(state["objects"]["rubiks_cube"]["quaternion_world_wxyz"])
    actual_relative_translation = _quat_rotate_inverse_wxyz(
        actual_base_q, actual_cube_p - actual_base_p
    )
    actual_relative_quaternion = _qmul(_quat_inverse_wxyz(actual_base_q), actual_cube_q)
    registered_relative = schedule_stage["selected_observed_cube_in_base_link_transform"]
    state["se3_contact_transform_evidence"] = {
        "frame": "cube_in_robot_base_link",
        "registered": registered_relative,
        "post_settle_observed": {
            "translation_m": _host(actual_relative_translation),
            "quaternion_wxyz": _host(actual_relative_quaternion),
        },
        "translation_residual_m": float(
            np.linalg.norm(
                actual_relative_translation
                - np.asarray(registered_relative["translation_m"], dtype=np.float64)
            )
        ),
        "orientation_geodesic_residual_deg": _quaternion_geodesic_error_deg(
            actual_relative_quaternion, registered_relative["quaternion_wxyz"]
        ),
        "post_settle_cube_midline_residual_m": abs(float(actual_cube_p[1])),
    }
    state["construction"] = dict(construction)
    state["normalized_state_sha256"] = normalized_state_sha256(state)
    state["passed"] = all(
        (
            state["physics_gate"]["passed"],
            state["ood_gate"]["passed"],
            state["camera_evidence"]["passed"],
            state["companion_pose_gate"]["passed"],
        )
    )
    return state


def _direct_materialize_and_gate(
    env: Any,
    *,
    ik: Mapping[str, Any],
    stage_name: str,
    schedule_stage: Mapping[str, Any],
    stage_reference: Mapping[str, Any],
    candidate: Any,
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> dict[str, Any]:
    global CURRENT_STAGE
    base_position, base_quaternion, cube_position, cube_quaternion = _target_pose_arrays(
        schedule_stage
    )
    arm = list(ik["arm_joint_solution_rad"])
    gripper = list(schedule_stage["selected_historical_gripper_joint_position_rad"])
    if len(arm) != 7 or len(gripper) != 6:
        raise RuntimeError("registered direct initialization joint dimensions differ")
    joint_values = arm + gripper
    if len(joint_values) != len(env.scene["robot"].joint_names):
        raise RuntimeError("direct initialization does not match exact E004 robot joints")

    CURRENT_STAGE = f"{label}_atomic_contact_consistent_write"
    robot = env.scene["robot"]
    cube = env.scene["rubiks_cube"]
    joint_position = torch.tensor(joint_values, dtype=torch.float32, device=env.device).reshape(1, -1)
    joint_velocity = torch.zeros_like(joint_position)
    cube_pose = torch.tensor(
        np.concatenate((cube_position, cube_quaternion)), dtype=torch.float32, device=env.device
    ).reshape(1, 7)
    cube_velocity = torch.zeros((1, 6), dtype=torch.float32, device=env.device)
    robot.write_joint_state_to_sim(joint_position, joint_velocity)
    robot.set_joint_position_target(joint_position)
    cube.write_root_pose_to_sim(cube_pose)
    cube.write_root_velocity_to_sim(cube_velocity)
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.scene.update(0.0)
    actual_position, actual_quaternion = _base_link_pose(env)
    post_write_fk = {
        "position_world_m": actual_position,
        "quaternion_world_wxyz": actual_quaternion,
        "frame": "robot/base_link",
        "position_error_m": float(np.linalg.norm(np.asarray(actual_position) - base_position)),
        "orientation_geodesic_error_deg": _quaternion_geodesic_error_deg(
            actual_quaternion, base_quaternion
        ),
        "base_link_to_eef_frame_identity": _frame_identity_evidence(env, frames, eef_index),
    }
    if post_write_fk["position_error_m"] > 0.001 or post_write_fk["orientation_geodesic_error_deg"] > 1.0:
        return {
            "passed": False,
            "candidate_rejection": "post_write_fk_outside_registered_ik_tolerance",
            "post_write_fk": post_write_fk,
            "construction_method": "direct_contact_initialization",
        }

    trace: list[dict[str, Any]] = []
    settled: list[dict[str, Any]] = []
    obs: Mapping[str, Any] | None = None
    termination: dict[str, Any] | None = None
    episode_length_before_equilibrium = _host(env.episode_length_buf)
    if episode_length_before_equilibrium != [75]:
        raise RuntimeError(
            "registered joint-equilibrium hold must start at episode step 75; "
            f"observed {episode_length_before_equilibrium}"
        )
    # The one normal-actuator target-buffer write occurred with the atomic
    # initialization above. Each later physics tick pushes that invariant
    # stored target through scene.write_data_to_sim.
    CURRENT_STAGE = f"{label}_normal_joint_equilibrium_hold_780"
    for step in range(780):
        obs, termination = _normal_joint_equilibrium_step(
            env,
            joint_position_target=joint_position,
            label=label,
            phase="normal_joint_equilibrium_hold_780",
            phase_step_zero_based=step,
        )
        if termination is not None:
            break
        trace.append(
            _joint_equilibrium_trace_row(
                env,
                frames,
                eef_index,
                phase="normal_joint_equilibrium_hold_780",
                phase_step_one_based=step + 1,
                joint_position_target=joint_position,
            )
        )
        if step % 20 == 0:
            video_frames.append(
                np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8)
            )
        if step >= 770:
            settled.append(_sample(env, frames, eef_index))
    if termination is not None or obs is None:
        return {
            "passed": False,
            "candidate_rejection": "environment_terminated_during_direct_normal_contact_settle",
            "termination": termination,
            "post_write_fk": post_write_fk,
            "construction_action_trace": trace,
            "construction_method": "direct_contact_initialization",
        }
    return _finalize_unchanged_gates(
        env,
        obs=obs,
        stage_name=stage_name,
        schedule_stage=schedule_stage,
        stage_reference=stage_reference,
        candidate=candidate,
        contact_coverage=contact_coverage,
        frames=frames,
        eef_index=eef_index,
        label=label,
        settled=settled,
        construction={
            "method": "direct_contact_initialization_then_uniform_normal_joint_equilibrium_hold",
            "stage": stage_name,
            "candidate_rank": schedule_stage["candidate_rank"],
            "registered_stage_schedule": schedule_stage,
            "ik_solution": ik,
            "atomic_write": {
                "arm_joint_solution_rad": arm,
                "historical_gripper_joint_position_rad": gripper,
                "all_joint_velocity_rad_s": [0.0] * 13,
                "cube_pose_world_wxyz": np.concatenate((cube_position, cube_quaternion)).tolist(),
                "cube_velocity_world": [0.0] * 6,
                "physics_steps_before_hold": 0,
                "post_write_fk": post_write_fk,
            },
            "closed_gripper_intent_provenance_only": 1.0,
            "authoritative_closed_gripper_joint_target_rad": gripper,
            "normal_joint_position_target_rad": joint_values,
            "joint_target_write_count_before_settle": 1,
            "joint_target_write_count_during_settle": 0,
            "cartesian_action_manager_apply_count_during_settle": 0,
            "joint_or_cube_state_write_count_during_settle": 0,
            "episode_length_buf_before_equilibrium": episode_length_before_equilibrium,
            "episode_length_buf_after_equilibrium": _host(env.episode_length_buf),
            "settle_steps": 780,
            "gate_window_final_steps": 10,
            "construction_action_trace": trace,
            "prohibitions_obeyed": {
                "no_weld_or_attachment": True,
                "no_collision_suppression": True,
                "no_force_injection": True,
                "no_model_request": True,
                "no_prompt_conditioned_action": True,
            },
        },
    )


def _archived_r008_open_contact_materialize_and_gate_never_called(
    env: Any,
    *,
    stage_name: str,
    schedule_stage: Mapping[str, Any],
    stage_reference: Mapping[str, Any],
    candidate: Any,
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> dict[str, Any]:
    """Archived predecessor topology; R010 dispatch cannot call this function."""

    raise RuntimeError("archived R008 object-servo construction is unreachable in R010")

    global CURRENT_STAGE
    servo_contract = candidate_schedule["object_space_servo_contract"]
    handoff_contract = candidate_schedule["joint_handoff_contract"]
    lifecycle_contract = candidate_schedule["construction_lifecycle_contract"]
    expected_steps = {
        "open_approach": 120,
        "open_descent": 120,
        "normal_close": 90,
        "closed_object_space_servo": 360,
        "captured_q_normal_joint_settle": 600,
    }
    if lifecycle_contract != {
        "fresh_reset_steps": 75,
        "open_approach_steps": 120,
        "open_descent_steps": 120,
        "normal_close_steps": 90,
        "object_space_servo_steps": 360,
        "joint_equilibrium_settle_steps": 600,
        "worst_case_steps": 1365,
        "registered_max_episode_length_steps": 1500,
        "registered_margin_steps": 135,
        "required_step_dt_s": 1.0 / 15.0,
        "registered_episode_length_s": 100.0,
        "only_construction_environment_timeout_changes": True,
        "behavioral_horizon_unchanged": True,
    }:
        raise RuntimeError("R010 lifecycle contract differs")
    if _host(env.episode_length_buf) != [75]:
        raise RuntimeError("R010 candidate actions must begin at exact reset step 75")
    precontact = schedule_stage["r008_precontact_targets"]
    target_cube = schedule_stage["r008_target_cube_pose"]
    approach = precontact["approach_base_link_pose"]
    contact = precontact["contact_base_link_pose_at_exact_reset_cube"]
    start_position_raw, start_quaternion_raw = _base_link_pose(env)
    start_position = np.asarray(start_position_raw, dtype=np.float64)
    start_quaternion = _quat_normalize_wxyz(start_quaternion_raw)
    approach_position = np.asarray(approach["position_world_m"], dtype=np.float64)
    approach_quaternion = _quat_normalize_wxyz(approach["quaternion_world_wxyz"])
    contact_position = np.asarray(contact["position_world_m"], dtype=np.float64)
    contact_quaternion = _quat_normalize_wxyz(contact["quaternion_world_wxyz"])
    target_cube_position = np.asarray(target_cube["position_world_m"], dtype=np.float64)
    target_cube_quaternion = _quat_normalize_wxyz(target_cube["quaternion_world_wxyz"])
    if not np.all(
        np.isfinite(
            np.concatenate(
                (
                    start_position, start_quaternion, approach_position,
                    approach_quaternion, contact_position, contact_quaternion,
                    target_cube_position, target_cube_quaternion,
                )
            )
        )
    ):
        raise RuntimeError("R010 open-contact target contains a nonfinite value")

    phases: list[tuple[str, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]] = [
        ("open_approach", 120, start_position, start_quaternion,
         approach_position, approach_quaternion, 0.0),
        ("open_descent", 120, approach_position, approach_quaternion,
         contact_position, contact_quaternion, 0.0),
        ("normal_close", 90, contact_position, contact_quaternion,
         contact_position, contact_quaternion, 1.0),
    ]
    trace: list[dict[str, Any]] = []
    servo_rows: list[dict[str, Any]] = []
    settled: list[dict[str, Any]] = []
    obs: Mapping[str, Any] | None = None
    termination: dict[str, Any] | None = None
    for phase, steps, from_p, from_q, to_p, to_q, grip in phases:
        CURRENT_STAGE = f"{label}_{phase}"
        for step in range(steps):
            fraction = float(step + 1) / float(steps)
            position = (1.0 - fraction) * from_p + fraction * to_p
            quaternion = _slerp_wxyz(from_q, to_q, fraction)
            action = _command_base_link(position, quaternion, grip, env.device)
            obs, _, terminated, truncated, extras = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                termination = _record_termination(
                    env,
                    label=label,
                    phase=phase,
                    phase_step_zero_based=step,
                    terminated=terminated,
                    truncated=truncated,
                    extras=extras,
                )
                break
            trace.append(
                _construction_trace_row(
                    env, frames, eef_index, phase=phase,
                    phase_step_one_based=step + 1, action=action,
                )
            )
            if step % 20 == 0:
                video_frames.append(
                    np.asarray(
                        obs["image_obs"]["head_camera"][0].detach().cpu().numpy(),
                        dtype=np.uint8,
                    )
                )
        if termination is not None:
            break
    if termination is None:
        CURRENT_STAGE = f"{label}_closed_object_space_servo_360"
        for step in range(360):
            live_base_p, live_base_q = _base_link_pose(env)
            live_cube_p, live_cube_q = _cube_pose(env)
            servo = object_space_servo_command(
                live_base_position=live_base_p,
                live_base_quaternion=live_base_q,
                live_cube_position=live_cube_p,
                live_cube_quaternion=live_cube_q,
                target_cube_position=target_cube_position,
                target_cube_quaternion=target_cube_quaternion,
                translation_gain=float(servo_contract["translation_gain"]),
                rotation_gain=float(servo_contract["rotation_gain"]),
                translation_cap_m_per_step=float(servo_contract["translation_cap_m_per_step"]),
                rotation_cap_deg_per_step=float(servo_contract["rotation_cap_deg_per_step"]),
            )
            action = _command_base_link(
                servo["command_base_position_world_m"],
                servo["command_base_quaternion_world_wxyz"],
                1.0,
                env.device,
            )
            obs, _, terminated, truncated, extras = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                termination = _record_termination(
                    env,
                    label=label,
                    phase="closed_object_space_servo",
                    phase_step_zero_based=step,
                    terminated=terminated,
                    truncated=truncated,
                    extras=extras,
                )
                break
            row = _construction_trace_row(
                env,
                frames,
                eef_index,
                phase="closed_object_space_servo",
                phase_step_one_based=step + 1,
                action=action,
            )
            row["pre_action_object_space_servo"] = servo
            trace.append(row)
            servo_rows.append(row)
            if step % 20 == 0:
                video_frames.append(
                    np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8)
                )
    captured_joint_target: torch.Tensor | None = None
    episode_before_handoff: list[Any] | None = None
    if termination is None:
        CURRENT_STAGE = f"{label}_capture_q_and_normal_joint_settle_600"
        captured = np.asarray(_host(env.scene["robot"].data.joint_pos[0]), dtype=np.float64)
        if captured.shape != (13,) or not np.all(np.isfinite(captured)):
            raise RuntimeError("R010 captured-q handoff is malformed")
        captured_joint_target = torch.tensor(
            captured, dtype=torch.float32, device=env.device
        ).reshape(1, 13)
        env.scene["robot"].set_joint_position_target(captured_joint_target)
        episode_before_handoff = _host(env.episode_length_buf)
        if episode_before_handoff != [765]:
            raise RuntimeError("R010 handoff did not begin at step 765")
        for step in range(600):
            obs, termination = _normal_joint_equilibrium_step(
                env,
                joint_position_target=captured_joint_target,
                label=label,
                phase="captured_q_normal_joint_settle",
                phase_step_zero_based=step,
            )
            if termination is not None:
                break
            trace_row = _joint_equilibrium_trace_row(
                env,
                frames,
                eef_index,
                phase="captured_q_normal_joint_settle",
                phase_step_one_based=step + 1,
                joint_position_target=captured_joint_target,
            )
            trace.append(trace_row)
            if step % 20 == 0:
                video_frames.append(
                    np.asarray(obs["image_obs"]["head_camera"][0].detach().cpu().numpy(), dtype=np.uint8)
                )
            if step >= 590:
                settled.append(_sample(env, frames, eef_index))
    if termination is not None or obs is None:
        return {
            "passed": False,
            "candidate_rejection": "environment_terminated_during_r008_object_servo_or_handoff",
            "termination": termination,
            "construction_method": "exact_reset_open_close_uniform_object_servo_q_handoff",
            "construction_action_trace": trace,
        }
    if len(trace) != 1290 or _host(env.episode_length_buf) != [1365]:
        raise RuntimeError("R010 construction action/counter total differs")
    if captured_joint_target is None or episode_before_handoff is None or len(servo_rows) != 360:
        raise RuntimeError("R010 object servo or q handoff evidence is incomplete")
    return _finalize_unchanged_gates(
        env,
        obs=obs,
        stage_name=stage_name,
        schedule_stage=schedule_stage,
        stage_reference=stage_reference,
        candidate=candidate,
        contact_coverage=contact_coverage,
        frames=frames,
        eef_index=eef_index,
        label=label,
        settled=settled,
        construction={
            "method": "exact_reset_open_close_uniform_object_servo_q_handoff",
            "stage": stage_name,
            "candidate_rank": schedule_stage["candidate_rank"],
            "registered_stage_schedule": schedule_stage,
            "object_space_servo_contract": servo_contract,
            "joint_handoff_contract": handoff_contract,
            "construction_lifecycle_contract": lifecycle_contract,
            "phase_steps": expected_steps,
            "start_base_link_pose_world_wxyz": [*start_position.tolist(), *start_quaternion.tolist()],
            "registered_precontact_targets": precontact,
            "registered_target_cube_pose": target_cube,
            "episode_length_buf_before_candidate_actions": [75],
            "episode_length_buf_before_handoff": episode_before_handoff,
            "episode_length_buf_after_candidate_actions": _host(env.episode_length_buf),
            "post_reset_joint_state_write_count": 0,
            "post_reset_object_state_write_count": 0,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "construction_action_trace": trace,
            "object_space_servo_trace": servo_rows,
            "captured_joint_position_target_rad": _host(captured_joint_target[0]),
            "captured_joint_target_write_count": 1,
            "cartesian_action_manager_apply_count_during_joint_settle": 0,
            "joint_or_object_state_write_count": 0,
            "settled_gate_samples": [dict(row) for row in settled],
            "gate_window_final_steps": 10,
            "prohibitions_obeyed": {
                "no_weld_or_attachment": True,
                "no_collision_suppression": True,
                "no_force_injection": True,
                "no_post_reset_state_write": True,
                "no_gate_read_or_early_stop_during_servo": True,
                "no_cartesian_action_during_joint_settle": True,
                "no_model_request": True,
                "no_prompt_or_requested_side_input": True,
            },
        },
    )


def _pinch_geometry_materialize_and_gate(
    env: Any,
    *,
    stage_name: str,
    schedule_stage: Mapping[str, Any],
    stage_reference: Mapping[str, Any],
    candidate: Any,
    contact_coverage: Mapping[str, Any],
    frames: Any,
    eef_index: int,
    label: str,
    video_frames: list[np.ndarray],
) -> dict[str, Any]:
    """Run the frozen nonadaptive collision-pinch acquisition and final gates."""

    global CURRENT_STAGE, LAST_PARTIAL_STAGES
    contract = candidate_schedule["pinch_geometry_contract"]
    handoff_contract = candidate_schedule["joint_handoff_contract"]
    registered_lifecycle = candidate_schedule["construction_lifecycle_contract"]
    expected_lifecycle = {
        "fresh_reset_steps": 75,
        "open_approach_steps": 180,
        "open_descent_steps": 180,
        "normal_close_steps": 120,
        "closed_vertical_lift_steps": 240,
        "closed_stage_transport_steps": 300,
        "joint_equilibrium_settle_steps": 600,
        "worst_case_steps": 1695,
        "registered_max_episode_length_steps": 1800,
        "registered_margin_steps": 105,
        "required_step_dt_s": 1.0 / 15.0,
        "registered_episode_length_s": 120.0,
        "only_construction_environment_timeout_changes": True,
        "behavioral_horizon_unchanged": True,
    }
    if registered_lifecycle != expected_lifecycle:
        raise RuntimeError("R010 construction lifecycle contract differs at materialization")
    if _host(env.episode_length_buf) != [75]:
        raise RuntimeError("R010 candidate actions must begin at exact reset step 75")

    target_cube = schedule_stage["r010_target_cube_pose"]
    target_cube_position = np.asarray(target_cube["position_world_m"], dtype=np.float64)
    target_cube_quaternion = _quat_normalize_wxyz(
        target_cube["quaternion_world_wxyz"]
    )
    acquisition_quaternion = _quat_normalize_wxyz(
        schedule_stage["r010_acquisition_base_quaternion_world_wxyz"]
    )
    final_quaternion = _quat_normalize_wxyz(
        schedule_stage["r010_final_base_quaternion_world_wxyz"]
    )
    if not np.all(
        np.isfinite(
            np.concatenate(
                (
                    target_cube_position,
                    target_cube_quaternion,
                    acquisition_quaternion,
                    final_quaternion,
                )
            )
        )
    ):
        raise RuntimeError("R010 frozen geometry target contains a nonfinite value")

    CURRENT_STAGE = f"{label}_resolve_unique_collision_geometry"
    geometry = _resolve_pinch_scene_geometry(env)
    if (
        GEOMETRY_ATTACHMENT_PREFLIGHT is None
        or GEOMETRY_ATTACHMENT_PREFLIGHT.get("passed") is not True
        or geometry.get("geometry_identity_sha256")
        != GEOMETRY_ATTACHMENT_PREFLIGHT.get("geometry_identity_sha256")
    ):
        raise RuntimeError(
            "R010 rank-stage collision inventory/body-relative geometry differs "
            "from the passed attachment preflight"
        )
    initial_bounds = _live_pinch_bounds(env, geometry)
    reset_cube_position_raw, reset_cube_quaternion_raw = _cube_pose(env)
    reset_cube_position = np.asarray(reset_cube_position_raw, dtype=np.float64)
    reset_cube_quaternion = _quat_normalize_wxyz(reset_cube_quaternion_raw)
    initial_cube_bounds = initial_bounds["cube"]["reconstructed_bounds_env_local"]
    reset_cube_collision_center = np.asarray(
        initial_cube_bounds["collision_center_env_local_m"], dtype=np.float64
    )
    reset_cube_half_extents = np.asarray(
        initial_cube_bounds["aabb_half_extents_env_local_m"], dtype=np.float64
    )
    cube_collision_center_in_cube = np.asarray(
        geometry["static_body_local_collision_geometry"]["cube"]["center_body_m"],
        dtype=np.float64,
    )
    reconstructed_reset_center = reset_cube_position + _qrotate(
        reset_cube_quaternion, cube_collision_center_in_cube
    )
    target_cube_collision_center = target_cube_position + _qrotate(
        target_cube_quaternion, cube_collision_center_in_cube
    )
    if (
        cube_collision_center_in_cube.shape != (3,)
        or target_cube_collision_center.shape != (3,)
        or not np.all(
            np.isfinite(
                np.concatenate(
                    (
                        reset_cube_collision_center,
                        reset_cube_half_extents,
                        cube_collision_center_in_cube,
                        reconstructed_reset_center,
                        target_cube_collision_center,
                    )
                )
            )
        )
        or np.any(reset_cube_half_extents <= 0.0)
        or not np.allclose(
            reconstructed_reset_center,
            reset_cube_collision_center,
            atol=1e-9,
            rtol=0.0,
        )
    ):
        raise RuntimeError("R010 cube collision-center reconstruction is invalid")
    lift_collision_center = reset_cube_collision_center.copy()
    lift_collision_center[2] = target_cube_collision_center[2]
    approach_clearance = 2.0 * float(reset_cube_half_extents[2])

    trace: list[dict[str, Any]] = []
    acquisition_trace: list[dict[str, Any]] = []
    settled: list[dict[str, Any]] = []
    contact_trace: list[dict[str, Any]] = []
    obs: Mapping[str, Any] | None = None
    termination: dict[str, Any] | None = None

    progress = {
        "method": "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff",
        "stage": stage_name,
        "candidate_rank": schedule_stage["candidate_rank"],
        "registered_stage_schedule": schedule_stage,
        "pinch_geometry_contract": contract,
        "collision_geometry_resolution": geometry,
        "geometry_attachment_preflight_identity": {
            "preflight_geometry_identity_sha256": (
                GEOMETRY_ATTACHMENT_PREFLIGHT["geometry_identity_sha256"]
            ),
            "rank_stage_geometry_identity_sha256": geometry[
                "geometry_identity_sha256"
            ],
            "passed": True,
        },
        "construction_action_trace": trace,
        "contact_and_grab_trace_diagnostic_only": contact_trace,
    }
    if stage_name in LAST_PARTIAL_STAGES:
        LAST_PARTIAL_STAGES[stage_name]["construction_progress"] = progress

    def run_cartesian_step(
        *,
        phase: str,
        phase_step_one_based: int,
        target_midpoint: np.ndarray,
        target_quaternion: np.ndarray,
        grip: float,
        command_uses_live_cube: bool,
        live_geometry: Mapping[str, Any],
    ) -> None:
        nonlocal obs, termination
        live_base_position, live_base_quaternion = _base_link_pose(env)
        live_bounds = live_geometry
        pre_cube_position, pre_cube_quaternion = _cube_pose(env)
        left_bounds = live_bounds["left"]["reconstructed_bounds_env_local"]
        right_bounds = live_bounds["right"]["reconstructed_bounds_env_local"]
        command = pinch_alignment_command(
            live_base_position_env_local=live_base_position,
            live_base_quaternion=live_base_quaternion,
            live_left_center_env_local=left_bounds["collision_center_env_local_m"],
            live_right_center_env_local=right_bounds["collision_center_env_local_m"],
            target_pinch_midpoint_env_local=target_midpoint,
            target_base_quaternion=target_quaternion,
            translation_gain=float(contract["translation_gain"]),
            rotation_gain=float(contract["rotation_gain"]),
            translation_cap_m_per_step=float(
                contract["translation_cap_m_per_step"]
            ),
            rotation_cap_deg_per_step=float(contract["rotation_cap_deg_per_step"]),
        )
        action = _command_base_link(
            command["command_base_position_env_local_m"],
            command["command_base_quaternion_world_wxyz"],
            grip,
            env.device,
        )
        pre_contact = {
            "contact_force_n": _contact_forces(env),
            "object_grabbed": bool(
                object_grabbed(env, object="rubiks_cube", env_id=0)
            ),
        }
        obs, _, terminated, truncated, extras = env.step(action)
        if bool(terminated[0]) or bool(truncated[0]):
            termination = _record_termination(
                env,
                label=label,
                phase=phase,
                phase_step_zero_based=phase_step_one_based - 1,
                terminated=terminated,
                truncated=truncated,
                extras=extras,
            )
            raise RuntimeError(
                "R010 construction environment terminated; registered candidate "
                "rejection is prohibited for infrastructure termination"
            )
        row = _construction_trace_row(
            env,
            frames,
            eef_index,
            phase=phase,
            phase_step_one_based=phase_step_one_based,
            action=action,
        )
        post_contact = {
            "contact_force_n": _contact_forces(env),
            "object_grabbed": bool(
                object_grabbed(env, object="rubiks_cube", env_id=0)
            ),
        }
        row["pre_action_pinch_geometry"] = {
            "live_base_position_env_local_m": live_base_position,
            "live_base_quaternion_world_wxyz": live_base_quaternion,
            "live_cube_position_env_local_m": pre_cube_position,
            "live_cube_quaternion_world_wxyz": pre_cube_quaternion,
            "live_tensor_collision_geometry": live_bounds,
            "coordinate_semantics": (
                "all controller positions are env-local world-axis coordinates; "
                "each retained tensor pose binds the subtracted scene env origin"
            ),
            "target_pinch_midpoint_env_local_m": target_midpoint.tolist(),
            "target_base_quaternion_world_wxyz": target_quaternion.tolist(),
            "command_uses_live_cube_collision_center": command_uses_live_cube,
            "pinch_alignment_command": command,
            "gripper_command": float(grip),
            "contact_and_grab_diagnostic_before_action": pre_contact,
        }
        row["contact_and_grab_diagnostic_after_action"] = post_contact
        trace.append(row)
        acquisition_trace.append(row)
        contact_trace.append(
            {
                "phase": phase,
                "phase_step_one_based": phase_step_one_based,
                "before": pre_contact,
                "after": post_contact,
            }
        )
        if (phase_step_one_based - 1) % 20 == 0:
            video_frames.append(
                np.asarray(
                    obs["image_obs"]["head_camera"][0].detach().cpu().numpy(),
                    dtype=np.uint8,
                )
            )

    phase_specs = (
        ("open_approach", int(contract["open_approach_steps"]), 0.0),
        ("open_descent", int(contract["open_descent_steps"]), 0.0),
        ("normal_close", int(contract["normal_close_steps"]), 1.0),
        ("closed_vertical_lift", int(contract["closed_vertical_lift_steps"]), 1.0),
        ("closed_stage_transport", int(contract["closed_stage_transport_steps"]), 1.0),
    )
    for phase, steps, grip in phase_specs:
        CURRENT_STAGE = f"{label}_{phase}"
        for step in range(steps):
            fraction = float(step + 1) / float(steps)
            live_geometry = _live_pinch_bounds(env, geometry)
            if phase in {"open_approach", "open_descent", "normal_close"}:
                live_cube = live_geometry["cube"]["reconstructed_bounds_env_local"]
                target_midpoint = np.asarray(
                    live_cube["collision_center_env_local_m"], dtype=np.float64
                )
                if phase == "open_approach":
                    target_midpoint = target_midpoint + np.asarray(
                        [
                            0.0,
                            0.0,
                            2.0 * float(live_cube["aabb_half_extents_env_local_m"][2]),
                        ],
                        dtype=np.float64,
                    )
                target_quaternion = acquisition_quaternion
                command_uses_live_cube = True
            elif phase == "closed_vertical_lift":
                target_midpoint = (
                    (1.0 - fraction) * reset_cube_collision_center
                    + fraction * lift_collision_center
                )
                target_quaternion = acquisition_quaternion
                command_uses_live_cube = False
            else:
                target_midpoint = (
                    (1.0 - fraction) * lift_collision_center
                    + fraction * target_cube_collision_center
                )
                target_quaternion = _slerp_wxyz(
                    acquisition_quaternion, final_quaternion, fraction
                )
                command_uses_live_cube = False
            run_cartesian_step(
                phase=phase,
                phase_step_one_based=step + 1,
                target_midpoint=target_midpoint,
                target_quaternion=target_quaternion,
                grip=grip,
                command_uses_live_cube=command_uses_live_cube,
                live_geometry=live_geometry,
            )

    captured = np.asarray(_host(env.scene["robot"].data.joint_pos[0]), dtype=np.float64)
    if captured.shape != (13,) or not np.all(np.isfinite(captured)):
        raise RuntimeError("R010 captured-q handoff is malformed")
    captured_joint_target = torch.tensor(
        captured, dtype=torch.float32, device=env.device
    ).reshape(1, 13)
    env.scene["robot"].set_joint_position_target(captured_joint_target)
    episode_before_handoff = _host(env.episode_length_buf)
    if episode_before_handoff != [1095]:
        raise RuntimeError("R010 captured-q handoff did not begin at registered step 1095")
    CURRENT_STAGE = f"{label}_captured_q_normal_joint_settle_600"
    for step in range(600):
        obs, termination = _normal_joint_equilibrium_step(
            env,
            joint_position_target=captured_joint_target,
            label=label,
            phase="captured_q_normal_joint_settle",
            phase_step_zero_based=step,
        )
        if termination is not None:
            raise RuntimeError(
                "R010 construction environment terminated during captured-q settle"
            )
        row = _joint_equilibrium_trace_row(
            env,
            frames,
            eef_index,
            phase="captured_q_normal_joint_settle",
            phase_step_one_based=step + 1,
            joint_position_target=captured_joint_target,
        )
        contact = {
            "contact_force_n": _contact_forces(env),
            "object_grabbed": bool(
                object_grabbed(env, object="rubiks_cube", env_id=0)
            ),
        }
        row["contact_and_grab_diagnostic_after_step"] = contact
        trace.append(row)
        contact_trace.append(
            {
                "phase": "captured_q_normal_joint_settle",
                "phase_step_one_based": step + 1,
                "after": contact,
            }
        )
        if step % 20 == 0:
            video_frames.append(
                np.asarray(
                    obs["image_obs"]["head_camera"][0].detach().cpu().numpy(),
                    dtype=np.uint8,
                )
            )
        if step >= 590:
            settled.append(_sample(env, frames, eef_index))

    if obs is None or len(trace) != 1620 or len(acquisition_trace) != 1020:
        raise RuntimeError("R010 retained action trace length differs")
    if len(contact_trace) != 1620 or _host(env.episode_length_buf) != [1695]:
        raise RuntimeError("R010 contact trace or lifecycle counter differs")
    if len(settled) != 10:
        raise RuntimeError("R010 final scientific gate window differs")

    return _finalize_unchanged_gates(
        env,
        obs=obs,
        stage_name=stage_name,
        schedule_stage=schedule_stage,
        stage_reference=stage_reference,
        candidate=candidate,
        contact_coverage=contact_coverage,
        frames=frames,
        eef_index=eef_index,
        label=label,
        settled=settled,
        construction={
            **progress,
            "joint_handoff_contract": handoff_contract,
            "construction_lifecycle_contract": registered_lifecycle,
            "phase_steps": {
                phase: steps for phase, steps, _grip in phase_specs
            }
            | {"captured_q_normal_joint_settle": 600},
            "reset_cube_pose_env_local_wxyz": [
                *reset_cube_position.tolist(),
                *reset_cube_quaternion.tolist(),
            ],
            "reset_cube_collision_center_env_local_m": reset_cube_collision_center.tolist(),
            "reset_cube_collision_half_extents_env_local_m": reset_cube_half_extents.tolist(),
            "cube_collision_center_in_cube_m": cube_collision_center_in_cube.tolist(),
            "target_cube_pose_env_local_wxyz": [
                *target_cube_position.tolist(),
                *target_cube_quaternion.tolist(),
            ],
            "target_cube_collision_center_env_local_m": target_cube_collision_center.tolist(),
            "lift_collision_center_env_local_m": lift_collision_center.tolist(),
            "registered_approach_clearance_m": approach_clearance,
            "episode_length_buf_before_candidate_actions": [75],
            "episode_length_buf_before_handoff": episode_before_handoff,
            "episode_length_buf_after_candidate_actions": _host(env.episode_length_buf),
            "post_reset_joint_state_write_count": 0,
            "post_reset_object_state_write_count": 0,
            "contact_or_grab_conditioned_branch_count": 0,
            "all_registered_phases_executed_unconditionally": True,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "acquisition_lift_transport_trace": acquisition_trace,
            "captured_joint_position_target_rad": _host(captured_joint_target[0]),
            "captured_joint_target_write_count": 1,
            "cartesian_action_manager_apply_count_during_joint_settle": 0,
            "joint_or_object_state_write_count": 0,
            "settled_gate_samples": [dict(row) for row in settled],
            "gate_window_final_steps": 10,
            "prohibitions_obeyed": {
                "no_weld_or_attachment": True,
                "no_collision_suppression": True,
                "no_force_injection": True,
                "no_post_reset_state_write": True,
                "no_contact_or_grab_conditioned_branch": True,
                "no_early_stop_or_phase_skip": True,
                "no_post_close_cube_base_rigid_servo": True,
                "no_cartesian_action_during_joint_settle": True,
                "no_model_request": True,
                "no_prompt_or_requested_side_input": True,
            },
        },
    )


def main() -> None:
    global CURRENT_STAGE, LAST_PARTIAL_STAGES, CANDIDATE_EVALUATION_COUNT
    global DIAGNOSTIC_EVALUATION_COUNT, COMPLETED_DIAGNOSTICS, COMPLETED_ATTEMPTS
    global GEOMETRY_ATTACHMENT_PREFLIGHT
    CURRENT_STAGE = "load_hash_bound_r010_contracts"
    args.output_dir.mkdir(parents=True)
    ood = json.loads(args.ood_freeze.read_text(encoding="utf-8"))
    reset_reference = json.loads(args.e004_reset_reference.read_text(encoding="utf-8"))
    runtime_bindings = load_runtime_contract(
        args.runtime_bindings,
        args.runtime_bindings_sha256,
        study_root=study_root,
        external_roots=(robolab_root,),
    )
    candidate = load_candidate(args.e004_candidate, args.e004_candidate_sha256)
    CURRENT_STAGE = "register_exact_e004_environment"
    task_file = study_root / "experiments/v3/phase_e/symmetric_layout_cohort_v3e004/task_files/left.py"
    auto_register_droid_abs_ik_envs(task=[str(task_file)], cameras=WRIST_LEFT_RIGHT_HEAD)
    started = time.time()
    video_frames: list[np.ndarray] = []
    attempts: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    task_prompt: str | None = None
    contact_coverage_by_stage: dict[str, Any] = {}

    def _open_exact_stage_environment(
        *,
        rank: int,
        stage_name: str,
        role: str,
    ) -> tuple[Any, Any, int, Mapping[str, Any], Mapping[str, Any], dict[str, Any]]:
        nonlocal task_prompt
        global CURRENT_STAGE
        label = f"rank{rank:02d}__{stage_name}__{role}"
        native_output = (args.output_dir / "native" / label).resolve()
        lifecycle: dict[str, Any] = {
            "environment_ordinal": len(ENVIRONMENT_LIFECYCLE) + 1,
            "candidate_rank": rank,
            "stage": stage_name,
            "role": role,
            "label": label,
            "construction": "fresh create_env instance dedicated to exactly one rank, stage, and construction role",
            "seed": 13000,
            "native_output_dir": str(native_output),
            "created": False,
            "contact_sensors_initialized_in_this_environment": False,
            "fresh_reset_completed_in_this_environment": False,
            "closed_before_next_environment": False,
        }
        ENVIRONMENT_LIFECYCLE.append(lifecycle)
        CURRENT_STAGE = f"{label}_create_new_environment"
        set_output_dir(str(native_output))
        env, cfg = create_env(
            "V3E004DroidLeftTask",
            device=args.device,
            seed=13000,
            num_envs=1,
            instruction_type="default",
            policy="v3e006_r010_zero_model_contact_consistent_state_repair",
            renderer=args.renderer,
            rendering_mode=args.rendering_type,
        )
        lifecycle["created"] = True
        lifecycle["python_object_id"] = id(env)
        lifecycle["construction_horizon_activation"] = {
            "status": "activation_started_before_first_reset_or_step"
        }
        lifecycle["construction_horizon_activation"] = _activate_registered_construction_horizon(
            env, cfg
        )
        if task_prompt is None:
            task_prompt = str(cfg.instruction)
        elif str(cfg.instruction) != task_prompt:
            raise RuntimeError("fresh construction environment task prompt differs from exact E004")
        frames = env.scene["frames"]
        eef_index = frames.data.target_frame_names.index("eef_frame")
        CURRENT_STAGE = f"{label}_contact_sensor_initialization_reset"
        env.reset()
        lifecycle["episode_length_after_sensor_initialization_reset"] = _host(env.episode_length_buf)
        CURRENT_STAGE = f"{label}_stage_local_contact_sensor_coverage"
        contact_coverage = _contact_coverage(env)
        lifecycle["contact_sensors_initialized_in_this_environment"] = True
        lifecycle["contact_sensor_coverage_passed"] = contact_coverage["passed"]
        contact_coverage_by_stage[label] = contact_coverage
        _obs, reset_state = _fresh_reset_and_gate(
            env,
            candidate=candidate,
            reset_reference=reset_reference,
            contact_coverage=contact_coverage,
            frames=frames,
            eef_index=eef_index,
            label=label,
            video_frames=video_frames,
        )
        lifecycle["fresh_reset_completed_in_this_environment"] = True
        return env, frames, eef_index, contact_coverage, reset_state, lifecycle

    def _close_stage_environment(env: Any, lifecycle: dict[str, Any], *, retained_failure: BaseException | None) -> None:
        global CURRENT_STAGE
        before = CURRENT_STAGE
        CURRENT_STAGE = f"{lifecycle['label']}_close_dedicated_environment"
        try:
            env.close()
            lifecycle["closed_before_next_environment"] = True
        except BaseException as close_error:
            lifecycle["close_error"] = {
                "type": type(close_error).__name__, "message": str(close_error)
            }
            if retained_failure is None:
                raise
            print(
                f"environment close raised after retained R010 failure: "
                f"{type(close_error).__name__}: {close_error}", file=sys.stderr,
            )
        finally:
            gc.collect()
            if retained_failure is not None:
                CURRENT_STAGE = before

    # R010 first validates that its one-time body-relative extraction attaches
    # to the live tensor bodies exactly as registered.  This environment is
    # closed before any diagnostic or candidate environment is created.
    preflight_env = None
    preflight_lifecycle: dict[str, Any] | None = None
    preflight_failure: BaseException | None = None
    try:
        (
            preflight_env,
            _preflight_frames,
            _preflight_eef_index,
            _preflight_contact_coverage,
            preflight_reset,
            preflight_lifecycle,
        ) = _open_exact_stage_environment(
            rank=0,
            stage_name="attachment_geometry",
            role="relative_bound_tensor_world_oracle_preflight",
        )
        CURRENT_STAGE = "r010_geometry_attachment_preflight_resolve_relative_bounds"
        preflight_geometry = _resolve_pinch_scene_geometry(preflight_env)
        CURRENT_STAGE = "r010_geometry_attachment_preflight_one_shot_usd_oracle"
        GEOMETRY_ATTACHMENT_PREFLIGHT = _run_geometry_attachment_preflight(
            preflight_env,
            geometry=preflight_geometry,
            lifecycle=preflight_lifecycle,
            fresh_reset=preflight_reset,
        )
    except BaseException as exc:
        preflight_failure = exc
        if preflight_lifecycle is not None:
            preflight_lifecycle["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
        raise
    finally:
        if preflight_env is not None and preflight_lifecycle is not None:
            _close_stage_environment(
                preflight_env,
                preflight_lifecycle,
                retained_failure=preflight_failure,
            )
            del preflight_env
    if GEOMETRY_ATTACHMENT_PREFLIGHT is None:
        raise RuntimeError("R010 geometry attachment preflight produced no receipt")
    preflight_receipt_path = args.output_dir / "geometry_attachment_preflight.json"
    preflight_receipt_path.write_bytes(
        canonical_bytes(GEOMETRY_ATTACHMENT_PREFLIGHT)
    )
    if GEOMETRY_ATTACHMENT_PREFLIGHT["passed"] is not True:
        CURRENT_STAGE = "r010_geometry_attachment_preflight_failed_blocking_candidates"
        video_path = (
            args.output_dir
            / "videos"
            / "v3e006_r010_geometry_attachment_preflight_failed.mp4"
        )
        video_path.parent.mkdir(parents=True)
        _write_video(video_path, video_frames)
        report = {
            "schema_version": "vla-wam-shared-v3e006-r010-state-repair-result-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": "V3-E006-R010",
            "status": "r010_geometry_attachment_preflight_failed_candidates_not_evaluated",
            "passed": False,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "geometry_attachment_preflight_count": 1,
            "r010_live_diagnostic_count": 0,
            "repair_candidate_evaluation_count": 0,
            "state_candidate_count": 0,
            "candidate_budget": 4,
            "diagnostic_budget": 4,
            "accepted_candidate_rank": None,
            "accepted_states": None,
            "geometry_attachment_preflight": GEOMETRY_ATTACHMENT_PREFLIGHT,
            "geometry_attachment_preflight_receipt": _binding(
                preflight_receipt_path
            ),
            "known_reachable_diagnostics": [],
            "attempts": [],
            "repair_registration": _binding(args.repair_registration),
            "candidate_schedule": _binding(args.candidate_schedule),
            "source_push_gate": _binding(args.source_push_gate),
            "original_v3e006_closure_binding": _binding(
                args.original_closure_binding
            ),
            "r009_predecessor_results": _binding(
                args.predecessor_closure_binding
            ),
            "ood_freeze": _binding(args.ood_freeze),
            "e004_full_reset_reference": _binding(args.e004_reset_reference),
            "e004_candidate": _binding(args.e004_candidate),
            "frozen_e004_runtime_bindings": {
                **_binding(args.runtime_bindings),
                "value": runtime_bindings,
            },
            "scene_assets": {
                "control": _binding(args.control_scene_asset),
                "paired": _binding(args.paired_scene_asset),
            },
            "construction_source": {
                **_binding(Path(__file__).resolve()),
                "study_commit": args.expected_study_commit,
            },
            "execution_evidence": _base_evidence(),
            "video": _binding(video_path),
            "release_boundary": (
                "the registered attachment oracle failed, so every diagnostic, "
                "candidate, model request, and behavioral episode remains blocked"
            ),
        }
        report_path = args.output_dir / "state_repair_result.json"
        report_path.write_bytes(canonical_bytes(report))
        print(
            json.dumps(
                {
                    "passed": False,
                    "output": str(report_path),
                    "sha256": _sha(report_path),
                    "geometry_attachment_preflight_failed": True,
                    "candidate_evaluations": 0,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    for diagnostic in candidate_schedule["known_reachable_diagnostics"]:
        diagnostic_index = int(diagnostic["diagnostic_index_one_based"])
        DIAGNOSTIC_EVALUATION_COUNT = diagnostic_index
        diagnostic_env = None
        diagnostic_lifecycle: dict[str, Any] | None = None
        diagnostic_failure: BaseException | None = None
        try:
            (
                diagnostic_env,
                diagnostic_frames,
                diagnostic_eef_index,
                _diagnostic_contact_coverage,
                diagnostic_reset,
                diagnostic_lifecycle,
            ) = _open_exact_stage_environment(
                rank=0,
                stage_name=str(diagnostic["stage"]),
                role=(
                    f"known_reachable_diagnostic_{diagnostic_index:02d}_"
                    f"{diagnostic['source_side']}"
                ),
            )
            result = _run_known_reachable_diagnostic(
                diagnostic_env,
                diagnostic=diagnostic,
                frames=diagnostic_frames,
                eef_index=diagnostic_eef_index,
                label=diagnostic_lifecycle["label"],
                video_frames=video_frames,
            )
            result["environment_lifecycle"] = diagnostic_lifecycle
            result["fresh_reset"] = diagnostic_reset
            diagnostics.append(result)
            LAST_PARTIAL_STAGES = {
                "r010_live_diagnostics": diagnostics,
                "candidate_evaluation_prohibited_until_all_four_pass": True,
            }
        except BaseException as exc:
            diagnostic_failure = exc
            if diagnostic_lifecycle is not None:
                diagnostic_lifecycle["failure"] = {
                    "type": type(exc).__name__, "message": str(exc)
                }
            raise
        finally:
            if diagnostic_env is not None and diagnostic_lifecycle is not None:
                _close_stage_environment(
                    diagnostic_env,
                    diagnostic_lifecycle,
                    retained_failure=diagnostic_failure,
                )
                del diagnostic_env
        COMPLETED_DIAGNOSTICS = diagnostics.copy()
        _persist_completed_evidence(diagnostics=diagnostics, attempts=attempts)
        if not diagnostics[-1]["passed"]:
            CURRENT_STAGE = "r010_known_reachable_diagnostic_failed_blocking_candidates"
            video_path = (
                args.output_dir
                / "videos"
                / "v3e006_r010_known_reachable_diagnostic_failed.mp4"
            )
            video_path.parent.mkdir(parents=True)
            _write_video(video_path, video_frames)
            report = {
                "schema_version": "vla-wam-shared-v3e006-r010-state-repair-result-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-E006-R010",
                "status": "r010_known_reachable_diagnostic_failed_candidates_not_evaluated",
                "passed": False,
                "model_request_count": 0,
                "behavioral_episode_count": 0,
                "geometry_attachment_preflight_count": 1,
                "r010_live_diagnostic_count": len(diagnostics),
                "repair_candidate_evaluation_count": 0,
                "state_candidate_count": 0,
                "candidate_budget": 4,
                "diagnostic_budget": 4,
                "accepted_candidate_rank": None,
                "accepted_states": None,
                "known_reachable_diagnostics": diagnostics,
                "geometry_attachment_preflight": GEOMETRY_ATTACHMENT_PREFLIGHT,
                "geometry_attachment_preflight_receipt": _binding(
                    preflight_receipt_path
                ),
                "attempts": [],
                "repair_registration": _binding(args.repair_registration),
                "candidate_schedule": _binding(args.candidate_schedule),
                "source_push_gate": _binding(args.source_push_gate),
                "original_v3e006_closure_binding": _binding(args.original_closure_binding),
                "r009_predecessor_results": _binding(args.predecessor_closure_binding),
                "ood_freeze": _binding(args.ood_freeze),
                "e004_full_reset_reference": _binding(args.e004_reset_reference),
                "e004_candidate": _binding(args.e004_candidate),
                "frozen_e004_runtime_bindings": {
                    **_binding(args.runtime_bindings),
                    "value": runtime_bindings,
                },
                "scene_assets": {
                    "control": _binding(args.control_scene_asset),
                    "paired": _binding(args.paired_scene_asset),
                },
                "construction_source": {
                    **_binding(Path(__file__).resolve()),
                    "study_commit": args.expected_study_commit,
                },
                "execution_evidence": _base_evidence(),
                "video": _binding(video_path),
                "release_boundary": (
                    "registered diagnostic failure blocks every R010 candidate and all behavior; "
                    "no adaptive solver edit is authorized under this registration"
                ),
            }
            report_path = args.output_dir / "state_repair_result.json"
            report_path.write_bytes(canonical_bytes(report))
            print(
                json.dumps(
                    {
                        "passed": False,
                        "output": str(report_path),
                        "sha256": _sha(report_path),
                        "diagnostic_failed": diagnostic_index,
                        "candidate_evaluations": 0,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return

    if len(diagnostics) != 4 or not all(row["passed"] for row in diagnostics):
        raise RuntimeError("R010 candidate evaluation reached without four passed diagnostics")

    for candidate_pair in candidate_schedule["candidate_pairs"]:
        rank = int(candidate_pair["candidate_rank"])
        CANDIDATE_EVALUATION_COUNT = rank
        rank_attempt: dict[str, Any] = {
            "candidate_rank": rank,
            "construction_method": candidate_pair["construction_method"],
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "geometry_attachment_preflight_count": 1,
            "stages": {},
        }
        LAST_PARTIAL_STAGES = {"candidate_rank": rank}
        for stage_name in ("canonical_grasp", "canonical_carry"):
            stage_entry: dict[str, Any] = {
                "stage": stage_name,
                "candidate_rank": rank,
                "construction_method": candidate_pair["construction_method"],
            }
            rank_attempt["stages"][stage_name] = stage_entry
            LAST_PARTIAL_STAGES[stage_name] = stage_entry
            schedule_stage = dict(candidate_pair[stage_name])
            schedule_stage["candidate_rank"] = rank

            material_env = None
            material_lifecycle: dict[str, Any] | None = None
            material_failure: BaseException | None = None
            try:
                (
                    material_env, material_frames, material_eef_index, material_contact_coverage,
                    material_reset, material_lifecycle,
                ) = _open_exact_stage_environment(
                    rank=rank,
                    stage_name=stage_name,
                    role="collision_pinch_acquire_lift_transport_q_handoff_materialization",
                )
                stage_entry["materialization_environment"] = {
                    "environment_lifecycle": material_lifecycle,
                    "fresh_reset": material_reset,
                }
                method = candidate_pair["construction_method"]
                if method != "exact_reset_uniform_collision_pinch_acquire_lift_transport_q_handoff":
                    raise RuntimeError(f"unregistered construction method: {method}")
                state = _pinch_geometry_materialize_and_gate(
                    material_env,
                    stage_name=stage_name,
                    schedule_stage=schedule_stage,
                    stage_reference=ood["stages"][stage_name],
                    candidate=candidate,
                    contact_coverage=material_contact_coverage,
                    frames=material_frames,
                    eef_index=material_eef_index,
                    label=material_lifecycle["label"],
                    video_frames=video_frames,
                )
                stage_entry["candidate_state"] = state
                LAST_PARTIAL_STAGES[stage_name] = stage_entry
            except BaseException as exc:
                material_failure = exc
                if material_lifecycle is not None:
                    material_lifecycle["failure"] = {
                        "type": type(exc).__name__, "message": str(exc)
                    }
                raise
            finally:
                if material_env is not None and material_lifecycle is not None:
                    _close_stage_environment(
                        material_env, material_lifecycle, retained_failure=material_failure
                    )
                    del material_env
        rank_attempt["passed"] = all(
            row["candidate_state"]["passed"] for row in rank_attempt["stages"].values()
        )
        attempts.append(rank_attempt)
        COMPLETED_ATTEMPTS = attempts.copy()
        _persist_completed_evidence(diagnostics=diagnostics, attempts=attempts)
        if rank_attempt["passed"]:
            accepted = rank_attempt
            break

    CURRENT_STAGE = "retain_r010_candidate_search"
    video_path = args.output_dir / "videos" / "v3e006_r010_state_repair_search.mp4"
    video_path.parent.mkdir(parents=True)
    _write_video(video_path, video_frames)
    passed = accepted is not None
    if task_prompt is None:
        raise RuntimeError("R010 candidate search created no exact E004 stage environment")
    report = {
            "schema_version": "vla-wam-shared-v3e006-r010-state-repair-result-v1",
            "study_id": "vla_wam_language_steerability_v3",
            "amendment_id": "V3-E006-R010",
            "status": (
                "passed_r010_state_repair_not_released_for_behavior"
                if passed
                else "r010_candidate_budget_exhausted_no_valid_state_pair"
            ),
            "passed": passed,
            "model_request_count": 0,
            "behavioral_episode_count": 0,
            "geometry_attachment_preflight_count": 1,
            "repair_candidate_evaluation_count": len(attempts),
            "accepted_candidate_rank": accepted["candidate_rank"] if accepted else None,
            "first_passing_rule_obeyed": accepted is None or all(not row["passed"] for row in attempts[:-1]),
            "candidate_budget": 4,
            "diagnostic_budget": 4,
            "r010_live_diagnostic_count": len(diagnostics),
            "known_reachable_diagnostics": diagnostics,
            "geometry_attachment_preflight": GEOMETRY_ATTACHMENT_PREFLIGHT,
            "geometry_attachment_preflight_receipt": _binding(
                preflight_receipt_path
            ),
            "repair_registration": _binding(args.repair_registration),
            "candidate_schedule": _binding(args.candidate_schedule),
            "source_push_gate": _binding(args.source_push_gate),
            "original_v3e006_closure_binding": _binding(args.original_closure_binding),
            "r009_predecessor_results": _binding(args.predecessor_closure_binding),
            "ood_freeze": _binding(args.ood_freeze),
            "construction_source": {
                **_binding(Path(__file__).resolve()),
                "study_commit": args.expected_study_commit,
            },
            "frozen_e004_runtime_bindings": {
                **_binding(args.runtime_bindings),
                "value": runtime_bindings,
            },
            "e004_full_reset_reference": _binding(args.e004_reset_reference),
            "e004_candidate": _binding(args.e004_candidate),
            "scene_assets": {
                "control": _binding(args.control_scene_asset),
                "paired": _binding(args.paired_scene_asset),
            },
            "historical_policy_provenance_disclosure": candidate_schedule[
                "historical_policy_provenance_disclosure"
            ],
            "construction_prompt_exposure": "the exact E004 environment task prompt exists in cfg but is never read by or supplied to the repair controller",
            "task_prompt_retained_for_audit": task_prompt,
            "selection_rule": candidate_schedule["selection_rule"],
            "environment_lifecycle": ENVIRONMENT_LIFECYCLE,
            "contact_sensor_coverage_by_stage": contact_coverage_by_stage,
            "attempts": attempts,
            "accepted_states": accepted["stages"] if accepted else None,
            "execution_evidence": _base_evidence(
                candidate_gate_passed=passed,
                state_candidate_count=1 if passed else 0,
            ),
            "construction_seconds": time.time() - started,
            "video": _binding(video_path),
            "release_boundary": "behavioral registration, queue, smoke/isolation, and release remain prohibited until this result is independently validated and committed",
    }
    report_path = args.output_dir / "state_repair_result.json"
    report_path.write_bytes(canonical_bytes(report))
    print(
        json.dumps(
            {
                "passed": passed,
                "output": str(report_path),
                "sha256": _sha(report_path),
                "bytes": report_path.stat().st_size,
                "candidate_evaluations": len(attempts),
                "accepted_candidate_rank": report["accepted_candidate_rank"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    construction_failure: BaseException | None = None
    try:
        main()
    except BaseException as exc:
        construction_failure = exc
        failure_path = _write_failure(exc)
        print(
            json.dumps(
                {"passed": False, "failure": str(failure_path), "sha256": _sha(failure_path)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        traceback.print_exc()
    if construction_failure is not None:
        print(
            "retained R010 failure; exiting child nonzero before SimulationApp.close to prevent "
            "Kit from replacing the failure status",
            file=sys.stderr,
        )
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        try:
            simulation_app.close()
        except BaseException as close_error:
            raise RuntimeError(
                f"SimulationApp.close raised after completed R010 search: "
                f"{type(close_error).__name__}: {close_error}"
            ) from close_error
