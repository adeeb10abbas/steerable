#!/usr/bin/env python3
"""Finalize a complete retained Cosmos V3-C001 seed without new inference.

This recovery path is deliberately narrow.  It is for a seed whose eight
RoboLab episodes, action traces, state traces, model request traces, decoded
futures, and closed viewport videos all exist, but whose normal bridge failed
before packaging the behavioral JSONL and whole-seed report.  The tool refuses
partial cells, validates every retained request artifact, and never connects to
a model server or simulator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"required retained artifact is missing or empty: {path}")
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def in_requested_region(sample: dict[str, Any], relation: str) -> bool:
    obj = np.asarray(sample["object_xyz"], dtype=np.float64)
    ref = np.asarray(sample["reference_xyz"], dtype=np.float64)
    forward, lateral = (obj - ref)[:2]
    distance = math.hypot(float(forward), float(lateral))
    margin = float(lateral) if relation == "left" else -float(lateral)
    return distance > 1e-8 and margin / distance >= math.cos(math.radians(45.0))


def event_timeline(capture: dict[str, Any]) -> list[dict[str, Any]]:
    relation = capture["requested_relation"]
    samples = capture["samples"]
    initial_z = float(samples[0]["object_xyz"][2])
    pickup = first_sustained(
        [float(row["object_xyz"][2]) - initial_z >= 0.03 for row in samples]
    )
    requested = next(
        (i for i, row in enumerate(samples) if in_requested_region(row, relation)), None
    )
    opposite = "right" if relation == "left" else "left"
    opposite_step = next(
        (i for i, row in enumerate(samples) if in_requested_region(row, opposite)), None
    )
    events = [{"event": "episode_start", "action_step": 0}]
    for name, step in (
        ("verified_pickup", pickup),
        ("requested_region_entry", requested),
        ("opposite_region_entry", opposite_step),
    ):
        if step is not None:
            events.append({"event": name, "action_step": int(step)})
    events.append({"event": "episode_end", "action_step": capture["actions_executed"]})
    rank = {
        "episode_start": 0,
        "verified_pickup": 1,
        "requested_region_entry": 2,
        "opposite_region_entry": 3,
        "episode_end": 4,
    }
    return sorted(events, key=lambda row: (row["action_step"], rank[row["event"]]))


def validate_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    opened = capture.isOpened()
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    decoded, _ = capture.read()
    capture.release()
    if not opened or not decoded or frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0:
        raise ValueError(f"viewport video is not decodable: {path}")
    return {"frame_count": frame_count, "width": width, "height": height, "fps": fps}


def validate_future_trace(path: Path, seed: int) -> int:
    trace = json.loads(path.read_text())
    requests = trace.get("requests")
    if not isinstance(requests, list) or len(requests) != trace.get("model_request_count"):
        raise ValueError(f"model request count mismatch: {path}")
    for request in requests:
        if (
            request.get("requested_sampling_seed") != seed
            or request.get("server_sampling_seed") != seed
        ):
            raise ValueError(f"sampling-seed echo mismatch: {path}")
        action_path = Path(request.get("action_path", ""))
        future_path = Path(request.get("future_path", ""))
        for artifact_path, digest_name, shape_name in (
            (action_path, "action_sha256", "action_shape"),
            (future_path, "future_sha256", "future_shape"),
        ):
            if not artifact_path.is_file() or sha256_file(artifact_path) != request.get(digest_name):
                raise ValueError(f"request artifact integrity mismatch: {artifact_path}")
            array = np.load(artifact_path, allow_pickle=False, mmap_mode="r")
            if list(array.shape) != request.get(shape_name) or not np.isfinite(array).all():
                raise ValueError(f"request artifact shape/content mismatch: {artifact_path}")
        if request.get("action_shape") != [32, 8]:
            raise ValueError(f"Cosmos action chunk changed shape: {action_path}")
        future_shape = request.get("future_shape")
        if (
            not isinstance(future_shape, list)
            or len(future_shape) != 4
            or future_shape[0] != 33
            or future_shape[-1] != 3
        ):
            raise ValueError(f"Cosmos decoded future changed shape: {future_path}")
    return len(requests)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        choices=("cosmos3_edge_policy_droid", "cosmos3_nano_policy_droid"),
        required=True,
    )
    parser.add_argument("--bridge-preflight", type=Path, required=True)
    parser.add_argument("--task-registration", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--registration-manifest", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--runner-output-root", type=Path, required=True)
    parser.add_argument("--launch-evidence", type=Path, required=True)
    parser.add_argument("--repair-evidence", type=Path, required=True)
    parser.add_argument("--source-stdout", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    sys.path.insert(0, str(study_root))
    sys.path.insert(0, str(study_root / "tools"))
    from experiments.v3.cosmos_droid.contract import verify_runtime_identity
    from experiments.v3.phase_c_four_phrasings.contract import (
        EXPERIMENT_ID,
        sha256_file as contract_sha256_file,
        validate_release_manifest,
    )
    from experiments.v3.phase_c_four_phrasings.cosmos_behavioral_contract import (
        validate_live_output_contract,
        validate_live_task_registration,
        validate_task_sources,
    )
    from vla_wam_v3_episode_schema import (
        derive_failure_taxonomy,
        derive_initial_state_sha256,
        derive_measurements,
        validate_behavioral_record,
        write_jsonl,
    )

    if args.launch_evidence.exists() or args.repair_evidence.exists():
        parser.error("refusing to overwrite retained finalization evidence")
    bridge, task_registration = validate_live_task_registration(
        bridge_preflight_path=args.bridge_preflight,
        task_registration_path=args.task_registration,
    )
    if bridge["model_id"] != args.model_id:
        parser.error("model ID differs from bridge preflight")
    if contract_sha256_file(args.execution_plan) != bridge.get("execution_plan_sha256"):
        parser.error("execution plan differs from bridge preflight")
    released = validate_release_manifest(
        args.release_manifest,
        model_id=args.model_id,
        registration_manifest_sha256=contract_sha256_file(args.registration_manifest),
    )
    if released.release_manifest_sha256 != bridge.get("release_manifest_sha256"):
        parser.error("release manifest differs from bridge preflight")
    release_payload = json.loads(args.release_manifest.read_text())
    runtime_identity_path = Path(release_payload["runtime_identity"]["path"])
    runtime_identity = verify_runtime_identity(study_root, args.model_id, runtime_identity_path)
    if released.runtime_identity_sha256 != runtime_identity["runtime_identity_sha256"]:
        parser.error("runtime identity differs from Phase-C release")
    if validate_task_sources(study_root) != bridge.get("task_source_sha256"):
        parser.error("task sources changed after bridge preflight")

    runner_rows = [
        json.loads(line)
        for line in (args.runner_output_root / "episode_results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(runner_rows) != 8:
        parser.error("retained runner output must contain exactly eight completed episodes")
    runner_by_task = {row.get("task_name"): row for row in runner_rows}
    if len(runner_by_task) != 8:
        parser.error("retained runner task identities are not unique")

    prepared: list[dict[str, Any]] = []
    for cell in bridge["cells"]:
        outputs = validate_live_output_contract(cell, fresh=False)
        raw_dir = Path(cell["raw_cell_directory"])
        if outputs["behavioral_jsonl"].exists() or outputs["simulator_viewport_video"].exists():
            parser.error(f"cell is already finalized: {cell['registered_cell_id']}")
        capture_path = raw_dir / "state_capture.json"
        capture = json.loads(capture_path.read_text())
        if not capture.get("behavioral_result_valid_candidate"):
            parser.error(f"partial cell cannot be repaired: {cell['registered_cell_id']}")
        if capture.get("requested_relation") != cell["relation"]:
            parser.error("capture requested relation changed")
        actions = np.load(outputs["executed_actions"], allow_pickle=False)
        if (
            actions.ndim != 2
            or actions.shape != (capture.get("actions_executed"), 8)
            or not np.isfinite(actions).all()
        ):
            parser.error(f"executed-action trace is invalid: {outputs['executed_actions']}")
        state_rows = sum(1 for line in outputs["state_trace"].open() if line.strip())
        if state_rows != actions.shape[0] + 1 or len(capture.get("samples", [])) != state_rows:
            parser.error(f"state trace is incomplete: {outputs['state_trace']}")
        trace_path = outputs["action_future_metadata"]
        request_count = validate_future_trace(trace_path, bridge["seed"])
        runner = runner_by_task.get(cell["task_name"])
        if (
            runner is None
            or runner.get("instruction") != cell["prompt"]
            or runner.get("success") != capture.get("requested_success")
            or runner.get("episode_step") != capture.get("actions_executed")
        ):
            parser.error(f"retained runner row differs from capture: {cell['registered_cell_id']}")
        candidates = sorted((args.runner_output_root / cell["task_name"]).glob("*_viewport.mp4"))
        if len(candidates) != 1:
            parser.error(f"expected one closed viewport video for {cell['registered_cell_id']}")
        video_metadata = validate_video(candidates[0])
        prepared.append(
            {
                "cell": cell,
                "outputs": outputs,
                "capture": capture,
                "source_video": candidates[0],
                "video_metadata": video_metadata,
                "request_count": request_count,
            }
        )

    rows = []
    for item in prepared:
        cell = item["cell"]
        outputs = item["outputs"]
        capture = item["capture"]
        target_video = outputs["simulator_viewport_video"]
        shutil.copy2(item["source_video"], target_video)
        if validate_video(target_video) != item["video_metadata"]:
            raise ValueError(f"copied viewport metadata changed: {target_video}")
        record = {
            "schema_version": "vla-wam-shared-v3-raw-episode-v1",
            "record_type": "behavioral_episode",
            "behavioral_result_valid": True,
            "study_id": "vla_wam_language_steerability_v3",
            "arena": "droid_robolab",
            "registered_cell_id": cell["registered_cell_id"],
            "attempt_id": capture["attempt_id"],
            "model_id": args.model_id,
            "pair_id": cell["seed_block_id"] + ":" + cell["prompt_family"],
            "prompt": cell["prompt"],
            "prompt_family": cell["prompt_family"],
            "predicate_id": "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
            "reset_id": (
                "v3c001:droid_robolab:neutral_reset:environment_seed_"
                f"{cell['environment_seed']}"
            ),
            "environment_seed": cell["environment_seed"],
            "policy_seed": cell["sampling_seed"],
            "requested_relation": cell["relation"],
            "requested_success": capture["requested_success"],
            "failure_stage": capture["frozen_failure_stage"],
            "frozen_failure_stage": capture["frozen_failure_stage"],
            "failure_taxonomy": "transport_failed",
            "measurement_frame": "robot_base_object_minus_reference_xyz_m",
            "measurement_frame_description": (
                "Object and reference XYZ samples are expressed in the frozen robot-base "
                "frame; forward is object-minus-reference x and lateral is "
                "object-minus-reference y, with positive lateral denoting robot LEFT."
            ),
            "checkpoint": {
                "id": release_payload["runtime_identity"]["checkpoint"],
                "revision": release_payload["runtime_identity"]["checkpoint_revision"],
            },
            "runtime_identity": {
                "id": runtime_identity["checkpoint_identifier"],
                "sha256": sha256_file(runtime_identity_path),
            },
            "artifacts": {
                "viewport_video": file_record(target_video),
                "executed_action_trace": file_record(outputs["executed_actions"]),
                "decoded_future_trace": file_record(outputs["action_future_metadata"]),
                "raw_result_jsonl": {
                    "path": str(outputs["behavioral_jsonl"]),
                    "integrity_scope": "batch_manifest_after_close",
                },
            },
            "steps": capture["samples"],
            "actions_executed": capture["actions_executed"],
            "action_cap": capture["action_cap"],
            "right_censored": capture["right_censored"],
            "first_contact_step": None,
            "first_contact_unavailable_reason": capture["first_contact_unavailable_reason"],
            "final_detached_release": capture["final_detached_release"],
            "wall_time_s": capture["wall_time_s"],
            "operational_wall_time_valid": capture["operational_wall_time_valid"],
            "event_timeline": event_timeline(capture),
        }
        record["initial_state_sha256"] = derive_initial_state_sha256(record)
        measurements = derive_measurements(record)
        record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
        record = validate_behavioral_record(record)
        jsonl_manifest = write_jsonl(outputs["behavioral_jsonl"], [record])
        rows.append(
            {
                "within_seed_execution_order": cell["within_seed_execution_order"],
                "registered_cell_id": cell["registered_cell_id"],
                "prompt_family": cell["prompt_family"],
                "relation": cell["relation"],
                "requested_success": record["requested_success"],
                "failure_taxonomy": record["failure_taxonomy"],
                "actions_executed": record["actions_executed"],
                "initial_state_sha256": record["initial_state_sha256"],
                "measurements": record["measurements"],
                "artifacts": {
                    "viewport_video": record["artifacts"]["viewport_video"],
                    "executed_actions": record["artifacts"]["executed_action_trace"],
                    "decoded_future_trace": record["artifacts"]["decoded_future_trace"],
                    "state_trace": file_record(outputs["state_trace"]),
                    "behavioral_jsonl": {
                        "path": str(outputs["behavioral_jsonl"]),
                        "sha256": jsonl_manifest["jsonl_sha256"],
                        "bytes": jsonl_manifest["jsonl_bytes"],
                    },
                },
            }
        )

    initial_hashes = {row["initial_state_sha256"] for row in rows}
    if len(initial_hashes) != 1:
        raise ValueError("retained whole-seed cells do not share an identical reset")
    report = {
        "schema_version": "vla-wam-shared-v3c-cosmos-whole-seed-smoke-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": args.model_id,
        "seed": bridge["seed"],
        "passed": True,
        "behavioral_episode_count": 8,
        "infrastructure_episode_count": 0,
        "model_request_count": sum(item["request_count"] for item in prepared),
        "release_manifest_sha256": sha256_file(args.release_manifest),
        "bridge_preflight_sha256": sha256_file(args.bridge_preflight),
        "task_registration_sha256": sha256_file(args.task_registration),
        "execution_plan_sha256": sha256_file(args.execution_plan),
        "runtime_identity_sha256": sha256_file(runtime_identity_path),
        "matched_initial_state_sha256": next(iter(initial_hashes)),
        "cells": rows,
        "finalization_repair": {
            "inference_requests_during_repair": 0,
            "actions_executed_during_repair": 0,
            "source_attempt_completed_behavioral_episodes": 8,
            "reason": (
                "The live bridge completed all eight trajectories and retained every raw "
                "artifact, then failed before packaging because the isolated deployment "
                "omitted the shared JSONL schema module."
            ),
        },
    }
    args.launch_evidence.write_bytes(canonical_json_bytes(report))
    repair = {
        "schema_version": "vla-wam-shared-v3c-offline-finalization-repair-v1",
        "experiment_id": EXPERIMENT_ID,
        "model_id": args.model_id,
        "seed": bridge["seed"],
        "status": "retained_complete_behavioral_seed_finalized_without_new_inference",
        "behavioral_episode_count": 8,
        "infrastructure_episode_count": 0,
        "inference_requests_during_repair": 0,
        "actions_executed_during_repair": 0,
        "source_stdout": file_record(args.source_stdout),
        "launch_evidence": file_record(args.launch_evidence),
        "interpretation": (
            "The original behavior was complete rather than censored; only deterministic "
            "post-run packaging was repaired. No model or simulator process was contacted."
        ),
    }
    args.repair_evidence.write_bytes(canonical_json_bytes(repair))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
