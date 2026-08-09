#!/usr/bin/env python3
"""Plan or execute the exact registered V3-E005 LingBot RoboTwin queue.

The supervisor assigns whole four-cell seed blocks to deterministic shards.
Every behavioral cell runs in a fresh subprocess through the existing native
process-group guard, matching the Phase-A LingBot isolation contract.  The
worker monkeypatches only the registered scene layout and exact static prompt;
it calls the hash-pinned Phase-A ``run_episode`` implementation unchanged.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from .evidence import (
    EPISODE_SCHEMA,
    PAIR_SCHEMA,
    build_provisional_episode,
    close_pair,
    infrastructure_record,
)
from .runtime_contract import (
    AMENDMENT_ID,
    ARENA,
    MODEL_ID,
    STUDY_ID,
    E005ContractError,
    RegisteredBundle,
    RegisteredCell,
    canonical_json_bytes,
    canonical_sha256,
    file_record,
    load_object,
    load_registered_bundle,
    require,
    sha256_file,
    shard_seed_blocks,
    validate_bound_artifact,
    validate_candidate_binding,
    validate_model_blind_gate_binding,
    verify_runtime_identity,
)


STUDY_ROOT_DEFAULT = Path(__file__).resolve().parents[4]
SCENE_CONTRACT_MODULE = (
    "experiments.v3.phase_e.cross_arena_geometry_v3e005.scene_contract"
)
MODEL_BLIND_GATE_MODULE = (
    "experiments.v3.phase_e.cross_arena_geometry_v3e005.model_blind_gate"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write(path: Path, payload: bytes) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(destination)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as handle:
        handle.write(canonical_json_bytes(row))
        handle.flush()
        os.fsync(handle.fileno())


def safe_cell_name(cell: RegisteredCell) -> str:
    return cell.cell_id.replace(":", "__")


def cell_root(output: Path, cell: RegisteredCell) -> Path:
    return Path(output) / "cells" / safe_cell_name(cell)


def _common(parser: argparse.ArgumentParser, *, output_required: bool) -> None:
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT_DEFAULT)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--layout-candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--model-blind-gate", type=Path, required=True)
    parser.add_argument("--model-blind-gate-sha256", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--external-repository", type=Path, required=True)
    parser.add_argument("--simulator-repository", type=Path, required=True)
    parser.add_argument("--expected-study-commit", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--attempt-id", required=True)
    if output_required:
        parser.add_argument("--output-dir", type=Path, required=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    plan = commands.add_parser("plan", help="Validate and print the exact shard; no model/simulator import.")
    _common(plan, output_required=False)
    execute = commands.add_parser("execute", help="Run one exact queue shard.")
    _common(execute, output_required=True)
    execute.add_argument("--gpu-index", type=int, required=True)
    execute.add_argument("--checkpoint", type=Path, required=True)
    execute.add_argument("--frozen-assets", type=Path, required=True)
    execute.add_argument("--pod", required=True)
    execute.add_argument("--pod-uid", required=True)
    execute.add_argument("--gpu-uuid", required=True)
    execute.add_argument("--limit-seed-blocks", type=int)
    execute.add_argument("--resume", action="store_true")

    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    _common(worker, output_required=True)
    worker.add_argument("--gpu-index", type=int, required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--frozen-assets", type=Path, required=True)
    worker.add_argument("--pod", required=True)
    worker.add_argument("--pod-uid", required=True)
    worker.add_argument("--gpu-uuid", required=True)
    worker.add_argument("--cell-id", required=True)
    return parser.parse_args(argv)


def _load_bound_inputs(args: argparse.Namespace, *, verify_live: bool) -> tuple[
    RegisteredBundle, dict[str, Any], dict[str, Any], dict[str, Any]
]:
    bundle = load_registered_bundle(
        args.study_root,
        registration_path=args.registration,
        queue_path=args.queue,
    )
    candidate = validate_bound_artifact(
        args.layout_candidate, args.candidate_sha256, "layout candidate"
    )
    validate_candidate_binding(
        candidate, bundle=bundle, candidate_sha256=args.candidate_sha256
    )
    gate = validate_bound_artifact(
        args.model_blind_gate,
        args.model_blind_gate_sha256,
        "model-blind gate",
    )
    validate_model_blind_gate_binding(
        gate,
        bundle=bundle,
        candidate_sha256=args.candidate_sha256,
    )
    runtime = verify_runtime_identity(
        bundle,
        args.runtime_manifest,
        external_repository=args.external_repository,
        simulator_repository=args.simulator_repository,
        expected_study_commit=args.expected_study_commit,
        verify_live_files=verify_live,
    )
    return bundle, candidate, gate, runtime


def _plan_payload(
    args: argparse.Namespace,
    bundle: RegisteredBundle,
    blocks: Sequence[Sequence[RegisteredCell]],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3e005-lingbot-shard-plan-v1",
        "status": "validated_plan_only_no_model_or_simulator_loaded",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "whole_seed_count": len(blocks),
        "behavioral_cell_count": sum(len(block) for block in blocks),
        "seeds": [block[0].environment_seed for block in blocks],
        "cell_ids": [cell.cell_id for block in blocks for cell in block],
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "layout_candidate_sha256": args.candidate_sha256,
        "model_blind_gate_sha256": args.model_blind_gate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "expected_study_commit": args.expected_study_commit,
        "sharding_rule": "position of registered seed in 9400..9426 modulo shard_count",
        "atomicity": "all s0/s1 LEFT/RIGHT cells for one seed stay on one lane",
        "denominator_boundary": "RoboTwin only; never pooled with DROID.",
    }


def _scene_runtime_modules() -> tuple[Any, Any]:
    try:
        scene = importlib.import_module(SCENE_CONTRACT_MODULE)
        geometry = importlib.import_module(MODEL_BLIND_GATE_MODULE)
    except ImportError as error:
        raise E005ContractError(
            "E005 scene contract is unavailable; build/hash-close the model-blind "
            "candidate and scene_contract.py before behavioral inference"
        ) from error
    for name in ("load_candidate", "validate_live_snapshot"):
        require(hasattr(scene, name), f"scene contract lacks required runtime API {name}")
    for name in (
        "actor_pose",
        "arm_reset_pose",
        "build_runtime_task_class",
        "camera_centers",
        "collision_bounding_radius",
        "reference_occludes_target",
        "rgb_views",
    ):
        require(hasattr(geometry, name), f"model-blind gate lacks required runtime API {name}")
    return scene, geometry


def _bound_gate_scene(gate: Mapping[str, Any], cell: RegisteredCell) -> dict[str, Any]:
    scenes = gate.get("scenes")
    require(isinstance(scenes, Mapping), "model-blind gate lacks its seven-scene inventory")
    row = scenes.get(cell.scene_id)
    require(isinstance(row, Mapping), f"model-blind gate lacks {cell.scene_id}")
    require(row.get("scene_id") == cell.scene_id, "model-blind gate scene id drift")
    return dict(row)


def _validate_execution_lane(args: argparse.Namespace, gate: Mapping[str, Any]) -> dict[str, str]:
    expected = {
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "gpu_uuid": args.gpu_uuid,
    }
    for key, value in expected.items():
        require(gate.get(key) == value, f"behavioral lane differs from model-blind gate for {key}")
    hostname = os.environ.get("HOSTNAME")
    if hostname:
        require(hostname == args.pod, "live pod hostname differs from the bound gate")
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    require(query.returncode == 0, "cannot verify the assigned E005 GPU")
    assigned = None
    for line in query.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) == 3 and fields[0] == str(args.gpu_index):
            assigned = fields
            break
    require(assigned is not None, "assigned E005 GPU index is not visible")
    require(assigned[1] == args.gpu_uuid, "assigned E005 GPU UUID drift")
    return {**expected, "gpu_index": str(args.gpu_index), "gpu_name": assigned[2]}


def _normalise_runtime_snapshot(
    *,
    scene: Any,
    candidate: Mapping[str, Any],
    gate_scene: Mapping[str, Any],
    cell: RegisteredCell,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and flatten one live reset without changing simulator state."""

    validation = scene.validate_live_snapshot(
        candidate,
        cell.scene_id,
        cell.symmetry_level,
        snapshot,
        gate_scene,
    )
    camera_checks = snapshot.get("occlusion_check")
    require(isinstance(camera_checks, Mapping), "runtime reset lacks per-camera occlusion checks")
    output = dict(snapshot)
    output["validation"] = validation
    output["asymmetry_metric_A"] = float(validation["asymmetry_metric_A"])
    output["position_residual_m"] = float(validation["position_residual_m"])
    output["orientation_residual_rad"] = float(validation["orientation_residual_rad"])
    output["midline_residual_m"] = float(validation["midline_residual_m"])
    output["all_camera_occlusion_checks"] = dict(camera_checks)
    output["occlusion_check"] = any(bool(value) for value in camera_checks.values())
    # validate_live_snapshot has already checked the exact candidate-bound
    # asset identity and quaternion for both roles.  Keep these explicit
    # booleans because they are registered per-episode fields.
    output["mirrored_asset_identity_verified"] = True
    output["mirrored_yaw_verified"] = True
    return output


def _runtime_task_with_snapshot(
    *,
    scene: Any,
    geometry: Any,
    candidate: Mapping[str, Any],
    gate_scene: Mapping[str, Any],
    cell: RegisteredCell,
    simulator: Path,
) -> type:
    """Wrap the gate-approved task class to expose the exact episode reset."""

    base = geometry.build_runtime_task_class(
        candidate,
        cell.scene_id,
        cell.symmetry_level,
        simulator,
        gate_scene,
    )
    asset_rows = candidate["scenes"][cell.scene_id]["assets"]
    identities = {
        role: tuple(asset_rows[role]["asset_identity"])
        for role in ("target", "reference")
    }

    class CapturingE005Task(base):
        latest_reset_snapshot: dict[str, Any] | None = None

        def setup_demo(self, *arguments, **keywords):
            value = super().setup_demo(*arguments, **keywords)
            poses = {
                "target": geometry.actor_pose(self.object, identities["target"]),
                "reference": geometry.actor_pose(
                    self.target_object, identities["reference"]
                ),
            }
            views = geometry.rgb_views(self.get_obs())
            centers = geometry.camera_centers(self)
            reference_radius = geometry.collision_bounding_radius(self.target_object)
            camera_checks = {
                camera: geometry.reference_occludes_target(
                    centers[camera],
                    poses["target"].position_xyz_m,
                    poses["reference"].position_xyz_m,
                    reference_radius,
                )
                for camera in scene.CAMERAS
            }
            raw_snapshot = {
                "scene_id": cell.scene_id,
                "symmetry_level_s": cell.symmetry_level,
                "realised_object_poses": {
                    role: pose.to_json() for role, pose in poses.items()
                },
                "asset_contract": asset_rows,
                "occlusion_check": camera_checks,
                "occlusion_method": (
                    "live_camera_center_to_target_segment_vs_reference_collision_bounding_sphere"
                ),
                "reference_collision_bounding_radius_m": reference_radius,
                "camera_centers_world_xyz_m": {
                    name: list(center) for name, center in centers.items()
                },
                "arm_reset_pose": geometry.arm_reset_pose(self),
                "views": {
                    name: {
                        "shape": list(image.shape),
                        "dtype": str(image.dtype),
                        "pixel_range": int(image.max()) - int(image.min()),
                    }
                    for name, image in views.items()
                },
            }
            snapshot = _normalise_runtime_snapshot(
                scene=scene,
                candidate=candidate,
                gate_scene=gate_scene,
                cell=cell,
                snapshot=raw_snapshot,
            )
            type(self).latest_reset_snapshot = snapshot
            return value

    CapturingE005Task.__name__ = f"{base.__name__}_RuntimeCapture"
    return CapturingE005Task


def _worker_native_paths(root: Path, task_name: str, cell: RegisteredCell) -> tuple[Path, Path]:
    condition = f"{cell.level_code}__{cell.relation}"
    directory = (
        root
        / "native"
        / task_name
        / f"environment_seed_{cell.environment_seed}"
        / f"sampling_seed_{cell.sampling_seed}"
        / condition
    )
    return directory / "result.json", directory / "live_reset_snapshot.json"


def run_worker(args: argparse.Namespace) -> int:
    """Run one cell in one model/simulator process, matching Phase-A isolation."""

    bundle, candidate_json, gate_json, runtime = _load_bound_inputs(args, verify_live=True)
    cell = bundle.cell(args.cell_id)
    output = Path(args.output_dir).expanduser().resolve()
    try:
        output.relative_to(bundle.study_root)
    except ValueError:
        pass
    else:
        raise E005ContractError("raw E005 output must stay on PVC outside ordinary Git")
    require(not output.exists(), f"refusing to overwrite existing cell attempt: {output}")
    output.mkdir(parents=True)
    execution_lane = _validate_execution_lane(args, gate_json)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    os.environ["LINGBOT_GPU"] = str(args.gpu_index)
    os.environ["VLA_WAM_V2_STUDY_ROOT"] = str(bundle.study_root)
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    require(
        os.environ.get("VK_ICD_FILENAMES") == "/etc/vulkan/icd.d/nvidia_icd.json",
        "E005 requires the gate-verified NVIDIA Vulkan ICD",
    )
    os.environ.pop("DISPLAY", None)

    external = Path(args.external_repository).expanduser().resolve()
    simulator = Path(args.simulator_repository).expanduser().resolve()
    runner_dir = external / "experiments/lingbot_language_gate"
    sys.path[:0] = [
        str(runner_dir),
        str(external / "src"),
        str(simulator),
        str(external / "third_party/curobo/src"),
        str(external),
    ]
    os.chdir(simulator)
    gate = importlib.import_module("closed_loop_language_gate")
    torch = importlib.import_module("torch")
    scene, geometry = _scene_runtime_modules()
    # The scene module performs semantic reconstruction as well as byte-hash
    # verification; runtime_contract already checked the external digest.
    candidate = scene.load_candidate(
        Path(args.layout_candidate).resolve(),
        args.candidate_sha256,
        bundle.registration_sha256,
        bundle.queue_sha256,
    )
    gate_scene = _bound_gate_scene(gate_json, cell)
    task_class = _runtime_task_with_snapshot(
        scene=scene,
        geometry=geometry,
        candidate=candidate,
        gate_scene=gate_scene,
        cell=cell,
        simulator=simulator,
    )
    task_name = f"v3e005_scene{cell.scene_number:02d}_{cell.level_code}"
    anchor_relation = "right" if cell.anchor_task.endswith("right") else "left"
    gate.RELATION_BY_TASK[task_name] = anchor_relation
    original_task_class = gate.task_class

    def patched_task_class(name: str):
        if name == task_name:
            return task_class
        return original_task_class(name)

    gate.task_class = patched_task_class
    original_prompt = gate.make_seen_prompt

    def exact_prompt(env, robotwin_root, relation, prompt_family, protocol):
        require(relation == cell.relation, "worker relation changed before prompt rendering")
        require(prompt_family == "direct_command", "worker prompt family is not direct_command")
        # Call the frozen renderer as an independent identity check, then use
        # the preregistered exact bytes.  Scene aliases may differ, but the
        # historical first-seen prompt bytes may not.
        rendered = original_prompt(env, robotwin_root, relation, prompt_family, protocol)
        require(rendered == cell.prompt, f"frozen LingBot prompt drift: {rendered!r}")
        return cell.prompt

    gate.make_seen_prompt = exact_prompt
    torch.cuda.reset_peak_memory_stats()
    policy, postprocessor = gate.load_policy(
        SimpleNamespace(
            checkpoint=Path(args.checkpoint).expanduser().resolve(),
            frozen_assets=Path(args.frozen_assets).expanduser().resolve(),
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            save_first_predicted_latent=True,
        )
    )
    protocol = gate.load_protocol(
        bundle.study_root / "artifacts/vla_wam_shared_v2/protocol.json"
    )
    result = gate.run_episode(
        policy=policy,
        postprocessor=postprocessor,
        prompt_cache={},
        negative_cache=[],
        robotwin_root=simulator,
        task_name=task_name,
        environment_seed=cell.environment_seed,
        sampling_seed=cell.sampling_seed,
        requested_relation=cell.relation,
        condition=f"{cell.level_code}__{cell.relation}",
        condition_alignment="correct" if cell.relation == anchor_relation else "swapped",
        prompt_family="direct_command",
        protocol=protocol,
        output_dir=output / "native",
        max_actions=400,
        save_simulator_video=True,
        save_first_predicted_latent=True,
    )
    snapshot = getattr(task_class, "latest_reset_snapshot", None)
    require(isinstance(snapshot, dict), "scene task did not expose a live reset snapshot")
    result_path, snapshot_path = _worker_native_paths(output, task_name, cell)
    require(result_path.is_file(), "frozen runner did not write result.json")
    atomic_write(snapshot_path, canonical_json_bytes(snapshot))
    provisional = build_provisional_episode(
        bundle=bundle,
        cell=cell,
        result_path=result_path,
        snapshot_path=snapshot_path,
        runtime=runtime,
        candidate_sha256=args.candidate_sha256,
        model_blind_gate_sha256=args.model_blind_gate_sha256,
        expected_study_commit=args.expected_study_commit,
        attempt_id=args.attempt_id,
        verify_video_decode=True,
    )
    provisional["execution_lane"] = execution_lane
    provisional_path = output / "provisional_episode.json"
    atomic_write(provisional_path, canonical_json_bytes(provisional))
    manifest = {
        "schema_version": "vla-wam-shared-v3e005-lingbot-cell-attempt-manifest-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "complete_behavioral_cell_awaiting_matched_pair_close",
        "cell_id": cell.cell_id,
        "attempt_id": args.attempt_id,
        "behavioral_episode_count": 1,
        "infrastructure_failure_count": 0,
        "model_action_request_count": int(result["actions_executed"]),
        "provisional_episode": file_record(provisional_path),
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "layout_candidate_sha256": args.candidate_sha256,
        "model_blind_gate_sha256": args.model_blind_gate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "study_commit": args.expected_study_commit,
        "execution_lane": execution_lane,
        "completed_at_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_write(output / "attempt_manifest.json", canonical_json_bytes(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _worker_command(args: argparse.Namespace, cell: RegisteredCell, destination: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "experiments.v3.phase_e.cross_arena_geometry_v3e005.run_lingbot_queue",
        "worker",
        "--study-root",
        str(Path(args.study_root).resolve()),
        "--registration",
        str(Path(args.registration).resolve()),
        "--queue",
        str(Path(args.queue).resolve()),
        "--layout-candidate",
        str(Path(args.layout_candidate).resolve()),
        "--candidate-sha256",
        args.candidate_sha256,
        "--model-blind-gate",
        str(Path(args.model_blind_gate).resolve()),
        "--model-blind-gate-sha256",
        args.model_blind_gate_sha256,
        "--runtime-manifest",
        str(Path(args.runtime_manifest).resolve()),
        "--external-repository",
        str(Path(args.external_repository).resolve()),
        "--simulator-repository",
        str(Path(args.simulator_repository).resolve()),
        "--expected-study-commit",
        args.expected_study_commit,
        "--shard-index",
        str(args.shard_index),
        "--shard-count",
        str(args.shard_count),
        "--attempt-id",
        args.attempt_id,
        "--output-dir",
        str(destination),
        "--gpu-index",
        str(args.gpu_index),
        "--checkpoint",
        str(Path(args.checkpoint).resolve()),
        "--frozen-assets",
        str(Path(args.frozen_assets).resolve()),
        "--pod",
        args.pod,
        "--pod-uid",
        args.pod_uid,
        "--gpu-uuid",
        args.gpu_uuid,
        "--cell-id",
        cell.cell_id,
    ]


def _guard_command(
    args: argparse.Namespace,
    cell: RegisteredCell,
    destination: Path,
    worker: Sequence[str],
) -> list[str]:
    thermal, interventions, invalid_attempts = _guard_artifact_paths(destination)
    return [
        sys.executable,
        str(Path(args.study_root).resolve() / "tools/native_process_group_thermal_guard.py"),
        "--launch",
        "--gpu-index",
        str(args.gpu_index),
        "--output",
        str(thermal),
        "--ledger-output",
        str(interventions),
        "--invalid-attempts-output",
        str(invalid_attempts),
        "--model-id",
        MODEL_ID,
        "--pair-id",
        cell.matched_layout_pair_id,
        "--environment-seed",
        str(cell.environment_seed),
        "--sampling-seed",
        str(cell.sampling_seed),
        "--requested-relation",
        cell.relation,
        "--",
        *worker,
    ]


def _guard_artifact_paths(destination: Path) -> tuple[Path, Path, Path]:
    """Keep guard logs beside the cell so the worker can create it atomically."""

    root = Path(destination)
    prefix = root.parent / root.name
    return (
        prefix.with_name(f"{prefix.name}.thermal_events.jsonl"),
        prefix.with_name(f"{prefix.name}.runtime_interventions_{MODEL_ID}.json"),
        prefix.with_name(f"{prefix.name}.invalid_attempts_{MODEL_ID}.json"),
    )


def _optional_artifact_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "path": str(resolved),
            "status": "not_emitted_by_guard",
            "sha256": None,
            "bytes": None,
        }
    return {"status": "retained", **file_record(resolved)}


def _validate_reusable_episode(
    path: Path,
    *,
    bundle: RegisteredBundle,
    cell: RegisteredCell,
    args: argparse.Namespace,
) -> dict[str, Any]:
    row = load_object(path)
    require(row.get("schema_version") == EPISODE_SCHEMA, "reused episode schema mismatch")
    checks = {
        "cell_id": cell.cell_id,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "layout_candidate_sha256": args.candidate_sha256,
        "model_blind_gate_sha256": args.model_blind_gate_sha256,
        "study_commit": args.expected_study_commit,
        "attempt_id": args.attempt_id,
    }
    for key, expected in checks.items():
        require(row.get(key) == expected, f"reused episode mismatch for {key}")
    expected_lane = {
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "gpu_uuid": args.gpu_uuid,
        "gpu_index": str(args.gpu_index),
    }
    lane = row.get("execution_lane")
    require(isinstance(lane, Mapping), "reused episode lacks its execution lane")
    for key, expected in expected_lane.items():
        require(lane.get(key) == expected, f"reused episode lane mismatch for {key}")
    for record in row.get("source_artifacts", {}).values():
        require(isinstance(record, dict), "reused episode artifact record is malformed")
        artifact = Path(str(record.get("path", ""))).expanduser().resolve()
        require(artifact.is_file(), f"reused source artifact is missing: {artifact}")
        require(sha256_file(artifact) == record.get("sha256"), "reused source artifact hash drift")
    return row


def _load_cell_after_worker(
    output: Path,
    cell: RegisteredCell,
    *,
    bundle: RegisteredBundle,
    args: argparse.Namespace,
) -> dict[str, Any]:
    root = cell_root(output, cell)
    final = root / "raw_episode.jsonl"
    provisional = root / "provisional_episode.json"
    if final.is_file():
        require(args.resume, f"cell is already complete; pass --resume: {cell.cell_id}")
        return _validate_reusable_episode(final, bundle=bundle, cell=cell, args=args)
    if provisional.is_file():
        require(args.resume, f"cell has prior provisional evidence; pass --resume: {cell.cell_id}")
        return _validate_reusable_episode(
            provisional, bundle=bundle, cell=cell, args=args
        )
    require(not root.exists(), f"partial cell attempt exists without valid evidence: {root}")
    worker = _worker_command(args, cell, root)
    guard = _guard_command(args, cell, root, worker)
    root.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = root.parent / f"{root.name}.stdout.log"
    stderr_path = root.parent / f"{root.name}.stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            guard,
            cwd=Path(args.study_root).resolve(),
            env=dict(os.environ),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0 or not provisional.is_file():
        retained = [stdout_path, stderr_path, *_guard_artifact_paths(root)]
        if root.exists():
            retained.extend(path for path in root.rglob("*") if path.is_file())
        infra = infrastructure_record(
            cell=cell,
            attempt_id=args.attempt_id,
            error=f"guarded LingBot worker exited {completed.returncode}",
            stage="guarded_native_cell_execution_or_postprocess",
            retained_paths=retained,
            bundle=bundle,
            candidate_sha256=args.candidate_sha256,
            model_blind_gate_sha256=args.model_blind_gate_sha256,
        )
        append_jsonl(output / "infrastructure_attempts.jsonl", infra)
        raise E005ContractError(
            f"cell {cell.cell_id} failed technically; partial output retained outside denominator"
        )
    return _validate_reusable_episode(provisional, bundle=bundle, cell=cell, args=args)


def _validate_seed_marker(
    marker_path: Path,
    block: Sequence[RegisteredCell],
    *,
    output: Path,
    bundle: RegisteredBundle,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    marker = load_object(marker_path)
    claimed_marker_sha = marker.get("marker_sha256")
    marker_body = dict(marker)
    marker_body.pop("marker_sha256", None)
    require(
        claimed_marker_sha == canonical_sha256(marker_body),
        "seed marker self hash is invalid",
    )
    require(marker.get("status") == "complete_four_valid_behavioral_cells", "seed marker is not complete")
    require(marker.get("behavioral_episode_count") == 4, "seed marker does not bind four cells")
    require(marker.get("matched_pair_count") == 2, "seed marker does not bind two pairs")
    expected_scalars = {
        "seed": block[0].environment_seed,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "layout_candidate_sha256": args.candidate_sha256,
        "model_blind_gate_sha256": args.model_blind_gate_sha256,
        "study_commit": args.expected_study_commit,
        "attempt_id": args.attempt_id,
    }
    for key, expected in expected_scalars.items():
        require(marker.get(key) == expected, f"seed marker mismatch for {key}")
    require(
        set(marker.get("cell_ids", [])) == {cell.cell_id for cell in block},
        "seed marker cell inventory drift",
    )
    episodes = [
        _validate_reusable_episode(
            cell_root(output, cell) / "raw_episode.jsonl",
            bundle=bundle,
            cell=cell,
            args=args,
        )
        for cell in block
    ]
    expected_episode_paths = {
        str((cell_root(output, cell) / "raw_episode.jsonl").resolve())
        for cell in block
    }
    require(
        set(marker.get("compact_episode_paths", [])) == expected_episode_paths,
        "seed marker compact episode paths drift",
    )
    for row in episodes:
        episode_path = cell_root(output, bundle.cell(row["cell_id"])) / "raw_episode.jsonl"
        require(
            marker.get("episode_sha256", {}).get(row["cell_id"])
            == sha256_file(episode_path),
            "seed marker episode hash drift",
        )
    pair_paths = [Path(path).expanduser().resolve() for path in marker.get("pair_paths", [])]
    pairs = [load_object(path) for path in pair_paths]
    require(len(pairs) == 2, "seed marker does not bind two matched pairs")
    require(all(pair.get("schema_version") == PAIR_SCHEMA for pair in pairs), "seed marker pair schema drift")
    require(
        {pair.get("matched_layout_pair_id") for pair in pairs}
        == {cell.matched_layout_pair_id for cell in block},
        "seed marker pair inventory drift",
    )
    for path, pair in zip(pair_paths, pairs, strict=True):
        pair_id = pair.get("matched_layout_pair_id")
        claimed_pair_sha = pair.get("pair_sha256")
        pair_body = dict(pair)
        pair_body.pop("pair_sha256", None)
        require(claimed_pair_sha == canonical_sha256(pair_body), "matched-pair self hash drift")
        require(
            marker.get("pair_sha256", {}).get(pair_id) == sha256_file(path),
            "seed marker pair hash drift",
        )
    return episodes, pairs


def run_execute(args: argparse.Namespace) -> int:
    bundle, _, _, runtime = _load_bound_inputs(args, verify_live=True)
    blocks = list(
        shard_seed_blocks(
            bundle, shard_index=args.shard_index, shard_count=args.shard_count
        )
    )
    if args.limit_seed_blocks is not None:
        require(args.limit_seed_blocks > 0, "limit_seed_blocks must be positive")
        blocks = blocks[: args.limit_seed_blocks]
    output = Path(args.output_dir).expanduser().resolve()
    try:
        output.relative_to(bundle.study_root)
    except ValueError:
        pass
    else:
        raise E005ContractError("raw E005 output must stay on PVC outside ordinary Git")
    if output.exists() and not args.resume:
        raise E005ContractError(f"refusing to overwrite existing shard output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "queue_progress.jsonl"
    all_episodes: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    for block in blocks:
        seed = block[0].environment_seed
        marker_path = output / "seeds" / f"seed_{seed}" / f"seed_{seed}_manifest.json"
        if marker_path.is_file():
            require(args.resume, f"seed {seed} already complete; pass --resume")
            episodes, pairs = _validate_seed_marker(
                marker_path, block, output=output, bundle=bundle, args=args
            )
            all_episodes.extend(episodes)
            all_pairs.extend(pairs)
            append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_reused", "seed": seed})
            continue
        append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_started", "seed": seed})
        provisional = {
            (cell.symmetry_level, cell.relation): _load_cell_after_worker(
                output, cell, bundle=bundle, args=args
            )
            for cell in block
        }
        seed_episodes: list[dict[str, Any]] = []
        seed_pairs: list[dict[str, Any]] = []
        pair_paths: list[str] = []
        for level in (0.0, 1.0):
            left, right, pair = close_pair(
                provisional[(level, "left")], provisional[(level, "right")]
            )
            for row, relation in ((left, "left"), (right, "right")):
                cell = next(
                    item
                    for item in block
                    if item.symmetry_level == level and item.relation == relation
                )
                episode_path = cell_root(output, cell) / "raw_episode.jsonl"
                atomic_write(episode_path, canonical_json_bytes(row))
                seed_episodes.append(row)
            pair_path = marker_path.parent / f"{pair['matched_layout_pair_id'].replace(':', '__')}.json"
            atomic_write(pair_path, canonical_json_bytes(pair))
            pair_paths.append(str(pair_path.resolve()))
            seed_pairs.append(pair)
        marker = {
            "schema_version": "vla-wam-shared-v3e005-lingbot-whole-seed-completion-v1",
            "study_id": STUDY_ID,
            "amendment_id": AMENDMENT_ID,
            "model_id": MODEL_ID,
            "arena": ARENA,
            "seed": seed,
            "status": "complete_four_valid_behavioral_cells",
            "behavioral_episode_count": 4,
            "matched_pair_count": 2,
            "infrastructure_failure_count": 0,
            "cell_ids": [row["cell_id"] for row in seed_episodes],
            "compact_episode_paths": [
                str(
                    (
                        cell_root(output, bundle.cell(row["cell_id"]))
                        / "raw_episode.jsonl"
                    ).resolve()
                )
                for row in seed_episodes
            ],
            "episode_sha256": {
                row["cell_id"]: sha256_file(
                    cell_root(output, bundle.cell(row["cell_id"])) / "raw_episode.jsonl"
                )
                for row in seed_episodes
            },
            "pair_paths": pair_paths,
            "pair_sha256": {
                pair["matched_layout_pair_id"]: sha256_file(Path(path))
                for pair, path in zip(seed_pairs, pair_paths, strict=True)
            },
            "registration_sha256": bundle.registration_sha256,
            "queue_sha256": bundle.queue_sha256,
            "layout_candidate_sha256": args.candidate_sha256,
            "model_blind_gate_sha256": args.model_blind_gate_sha256,
            "runtime_identity_sha256": runtime["runtime_identity_sha256"],
            "study_commit": args.expected_study_commit,
            "attempt_id": args.attempt_id,
            "completed_at_utc": utc_now(),
        }
        marker["marker_sha256"] = canonical_sha256(marker)
        atomic_write(marker_path, canonical_json_bytes(marker))
        append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_complete", "seed": seed})
        all_episodes.extend(seed_episodes)
        all_pairs.extend(seed_pairs)

    all_episodes.sort(
        key=lambda row: (
            int(row["environment_seed"]),
            float(row["symmetry_level_s"]),
            0 if row["relation"] == "left" else 1,
        )
    )
    all_pairs.sort(
        key=lambda row: (int(row["environment_seed"]), float(row["symmetry_level_s"]))
    )
    episodes_path = output / "behavioral_episodes.jsonl"
    pairs_path = output / "matched_pairs.jsonl"
    atomic_write(
        episodes_path,
        b"".join(canonical_json_bytes(row) for row in all_episodes),
    )
    atomic_write(pairs_path, b"".join(canonical_json_bytes(row) for row in all_pairs))
    status = (
        "smoke_slice_complete"
        if args.limit_seed_blocks is not None
        else "requested_shard_complete"
    )
    manifest = {
        "schema_version": "vla-wam-shared-v3e005-lingbot-shard-manifest-v1",
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": status,
        "attempt_id": args.attempt_id,
        "execution_lane": {
            "pod": args.pod,
            "pod_uid": args.pod_uid,
            "gpu_uuid": args.gpu_uuid,
            "gpu_index": str(args.gpu_index),
        },
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "whole_seed_count": len(blocks),
        "matched_pair_count": len(all_pairs),
        "behavioral_episode_count": len(all_episodes),
        "infrastructure_failure_count": 0,
        "seeds": [block[0].environment_seed for block in blocks],
        "behavioral_episodes": file_record(episodes_path),
        "matched_pairs": file_record(pairs_path),
        "whole_seed_manifests": [
            file_record(
                output
                / "seeds"
                / f"seed_{block[0].environment_seed}"
                / f"seed_{block[0].environment_seed}_manifest.json"
            )
            for block in blocks
        ],
        "process_guard_artifacts": {
            cell.cell_id: [
                _optional_artifact_record(path)
                for path in _guard_artifact_paths(cell_root(output, cell))
            ]
            for block in blocks
            for cell in block
        },
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "layout_candidate_sha256": args.candidate_sha256,
        "model_blind_gate_sha256": args.model_blind_gate_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "study_commit": args.expected_study_commit,
        "external_repository_commit": runtime["external_repository"]["commit"],
        "simulator_repository_commit": runtime["simulator_repository"]["commit"],
        "sharding_rule": "position of registered seed in 9400..9426 modulo shard_count",
        "whole_seed_atomic": True,
        "latent_future_policy": "retained as latent-only; never decoded or scored as video",
        "denominator_boundary": "RoboTwin only; never pooled with DROID.",
        "completed_at_utc": utc_now(),
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    manifest_name = (
        f"partial_shard_manifest_limit-seed-blocks-{args.limit_seed_blocks:04d}.json"
        if args.limit_seed_blocks is not None
        else "shard_manifest.json"
    )
    atomic_write(output / manifest_name, canonical_json_bytes(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "worker":
            return run_worker(args)
        bundle, _, _, runtime = _load_bound_inputs(args, verify_live=False)
        blocks = shard_seed_blocks(
            bundle, shard_index=args.shard_index, shard_count=args.shard_count
        )
        if args.mode == "plan":
            print(json.dumps(_plan_payload(args, bundle, blocks, runtime), indent=2, sort_keys=True))
            return 0
        return run_execute(args)
    except (E005ContractError, OSError, ValueError, KeyError) as error:
        print(f"V3-E005 fail-closed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
