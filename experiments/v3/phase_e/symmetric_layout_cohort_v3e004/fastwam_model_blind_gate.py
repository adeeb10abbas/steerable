#!/usr/bin/env python3
"""Live zero-model gate for the V3-E004 FastWAM RoboTwin stretch.

The gate validates the exact V3-B007 control and the new full-symmetry
endpoint on RoboTwin's native source-x lateral axis.  It loads no policy and
executes no action.  Occlusion is tested conservatively by intersecting each
camera-to-target-center segment with the reference actor's live collision
bounding sphere.
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
from typing import Any, Mapping, Sequence

import numpy as np

from .fastwam_robotwin import (
    ARENA,
    CAMERAS,
    CONTROL_FIXTURE,
    CORE_SEEDS,
    EXPECTED_OBJECT,
    EXPECTED_REFERENCE,
    FASTWAM_COMMIT,
    LIVE_ORIENTATION_TOLERANCE_RAD,
    LIVE_POSITION_TOLERANCE_M,
    MODEL_ID,
    ORIENTATION_TOLERANCE_RAD,
    POSITION_TOLERANCE_M,
    PROMPTS,
    SOURCE_FIXTURE_ENVIRONMENT_SEED,
    ActorPose,
    asymmetry_A,
    candidate_payload,
    canonical_json_bytes,
    layout_for_level,
    load_candidate,
    quaternion_yaw,
    residuals,
    validate_registered_queue,
    wrap_angle,
)


TASK = "place_a2b_right"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--external-repository", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pod", required=True)
    parser.add_argument("--pod-uid", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--repeat-resets", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=250)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


def load_task_args(robotwin_root: Path) -> dict[str, Any]:
    import yaml

    with (robotwin_root / "task_config/demo_clean.yml").open() as handle:
        task_args = yaml.safe_load(handle)
    with (robotwin_root / "task_config/_embodiment_config.yml").open() as handle:
        embodiments = yaml.safe_load(handle)
    with (robotwin_root / "task_config/_camera_config.yml").open() as handle:
        cameras = yaml.safe_load(handle)
    embodiment_name = task_args["embodiment"][0]
    robot_file = embodiments[embodiment_name]["file_path"]
    with (robotwin_root / robot_file / "config.yml").open() as handle:
        embodiment_config = yaml.safe_load(handle)
    head = cameras[task_args["camera"]["head_camera_type"]]
    task_args.update(
        task_name=TASK,
        task_config="demo_clean",
        eval_mode=True,
        save_data=False,
        collect_data=False,
        render_freq=0,
        head_camera_h=head["h"],
        head_camera_w=head["w"],
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=embodiment_config,
        right_embodiment_config=embodiment_config,
    )
    return task_args


def actor_pose(actor: Any, identity: tuple[str, int]) -> ActorPose:
    pose = actor.get_pose()
    return ActorPose(
        tuple(float(value) for value in pose.p),
        tuple(float(value) for value in pose.q),
        identity,
    )


def angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(float(x) * float(y) for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def validate_live_pose(observed: ActorPose, expected: ActorPose, label: str) -> None:
    if observed.asset_identity != expected.asset_identity:
        raise ValueError(f"{label}: asset identity drift")
    translation = max(abs(a - b) for a, b in zip(observed.position_xyz_m, expected.position_xyz_m))
    if translation > LIVE_POSITION_TOLERANCE_M:
        raise ValueError(f"{label}: translation differs from candidate by {translation:.6f} m")
    angle = angular_distance(observed.quaternion_wxyz, expected.quaternion_wxyz)
    if angle > LIVE_ORIENTATION_TOLERANCE_RAD:
        raise ValueError(f"{label}: orientation differs from candidate by {math.degrees(angle):.3f} deg")


def rgb_views(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    camera_rows = observation.get("observation", observation)
    output: dict[str, np.ndarray] = {}
    for name in CAMERAS:
        row = camera_rows.get(name)
        if not isinstance(row, Mapping) or "rgb" not in row:
            raise ValueError(f"required RoboTwin view missing: {name}")
        value = np.asarray(row["rgb"])
        if value.ndim != 3 or value.shape[-1] not in (3, 4):
            raise ValueError(f"malformed RoboTwin view: {name} {value.shape}")
        value = value[..., :3]
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if value.max() <= 1.0 else value, 0, 255).astype(np.uint8)
        if int(np.ptp(value)) == 0:
            raise ValueError(f"blank RoboTwin view: {name}")
        output[name] = value
    return output


def _entity_pose(entity: Any) -> tuple[float, float, float] | None:
    if entity is None:
        return None
    seen: set[int] = set()
    current = entity
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        pose = current.get_pose() if hasattr(current, "get_pose") else getattr(current, "pose", None)
        if pose is not None and hasattr(pose, "p"):
            values = tuple(float(value) for value in pose.p)
            return values if len(values) == 3 and all(math.isfinite(value) for value in values) else None
        current = getattr(current, "entity", None) or getattr(current, "actor", None)
    return None


def camera_centers(env: Any) -> dict[str, tuple[float, float, float]]:
    output: dict[str, tuple[float, float, float]] = {}
    entities = []
    rig = getattr(env, "cameras", None)
    if rig is not None:
        entities.extend(
            camera
            for camera in (getattr(rig, "left_camera", None), getattr(rig, "right_camera", None))
            if camera is not None
        )
        names = list(getattr(rig, "static_camera_name", []))
        static = list(getattr(rig, "static_camera_list", []))
        entities.extend(static)
        named_from_rig = dict(zip(names, static, strict=False))
    else:
        named_from_rig = {}
    scene = getattr(env, "scene", None)
    for getter in ("get_all_cameras", "get_cameras"):
        if scene is not None and hasattr(scene, getter):
            try:
                entities.extend(list(getattr(scene, getter)()))
            except TypeError:
                pass
    for name in CAMERAS:
        candidates = [getattr(env, name, None), named_from_rig.get(name)]
        if rig is not None:
            candidates.append(getattr(rig, name, None))
        candidates.extend(entity for entity in entities if getattr(entity, "name", None) == name)
        candidates.extend(
            entity for entity in entities if getattr(getattr(entity, "entity", None), "name", None) == name
        )
        center = next((value for value in (_entity_pose(item) for item in candidates) if value is not None), None)
        if center is None:
            raise ValueError(f"{name}: calibrated live camera pose unavailable")
        output[name] = center
    return output


def _shape_radius(shape: Any) -> float | None:
    geometry = getattr(shape, "geometry", None)
    if geometry is None and hasattr(shape, "get_geometry"):
        geometry = shape.get_geometry()
    if geometry is None:
        geometry = shape
    for field in ("half_lengths", "half_size", "half_extents"):
        value = getattr(geometry, field, None)
        if value is not None:
            row = np.asarray(value, dtype=np.float64).reshape(-1)
            if len(row) >= 3 and np.all(np.isfinite(row[:3])):
                return float(np.linalg.norm(row[:3]))
    radius = getattr(geometry, "radius", None)
    half_length = getattr(geometry, "half_length", 0.0)
    if radius is not None and math.isfinite(float(radius)) and math.isfinite(float(half_length)):
        return float(math.hypot(float(radius), float(half_length)))
    for field in ("vertices", "points"):
        value = getattr(shape, field, None)
        if value is None and hasattr(shape, f"get_{field}"):
            value = getattr(shape, f"get_{field}")()
        if value is not None:
            row = np.asarray(value, dtype=np.float64)
            if row.ndim == 2 and row.shape[1] >= 3 and row.size and np.all(np.isfinite(row[:, :3])):
                scale = np.asarray(getattr(shape, "scale", [1.0, 1.0, 1.0]), dtype=np.float64).reshape(-1)
                if scale.size == 1:
                    scale = np.repeat(scale, 3)
                return float(np.max(np.linalg.norm(row[:, :3] * scale[:3], axis=1)))
    return None


def collision_bounding_radius(actor: Any) -> float:
    shapes = []
    sources = [actor, getattr(actor, "actor", None), getattr(actor, "entity", None)]
    for source in tuple(item for item in sources if item is not None):
        if hasattr(source, "get_components"):
            sources.extend(list(source.get_components()))
    for source in (item for item in sources if item is not None):
        for getter in ("get_collision_shapes", "get_collision_shape"):
            if hasattr(source, getter):
                value = getattr(source, getter)()
                shapes.extend(value if isinstance(value, (list, tuple)) else [value])
        value = getattr(source, "collision_shapes", None)
        if value is not None:
            shapes.extend(value if isinstance(value, (list, tuple)) else [value])
    radii = [radius for radius in (_shape_radius(shape) for shape in shapes) if radius is not None]
    if not radii:
        raise ValueError("reference live collision geometry does not expose a bounding radius")
    radius = max(radii)
    if not 0.001 < radius < 0.5:
        raise ValueError(f"reference collision radius is implausible: {radius}")
    return radius


def reference_occludes_target(
    camera_xyz: Sequence[float],
    target_xyz: Sequence[float],
    reference_xyz: Sequence[float],
    reference_radius_m: float,
) -> bool:
    camera = np.asarray(camera_xyz, dtype=np.float64)
    target = np.asarray(target_xyz, dtype=np.float64)
    reference = np.asarray(reference_xyz, dtype=np.float64)
    segment = target - camera
    denominator = float(np.dot(segment, segment))
    if denominator <= 0.0:
        raise ValueError("camera and target centers coincide")
    fraction = float(np.dot(reference - camera, segment) / denominator)
    if not 0.0 < fraction < 1.0:
        return False
    closest = camera + fraction * segment
    return bool(float(np.linalg.norm(reference - closest)) <= float(reference_radius_m))


def arm_reset_pose(env: Any) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name in ("left_robot", "right_robot", "robot"):
        robot = getattr(env, name, None)
        if robot is None:
            continue
        qpos = robot.get_qpos() if hasattr(robot, "get_qpos") else None
        if qpos is not None:
            rows[name] = {"joint_positions_rad": [float(value) for value in np.asarray(qpos).reshape(-1)]}
        # RoboTwin's dual-arm wrapper exposes two SAPIEN articulations rather
        # than one combined get_qpos method.
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity", None)
            side_qpos = entity.get_qpos() if entity is not None and hasattr(entity, "get_qpos") else None
            if side_qpos is None and hasattr(robot, f"get_{side}_arm_jointState"):
                side_qpos = getattr(robot, f"get_{side}_arm_jointState")()
            if side_qpos is not None:
                rows[f"{name}.{side}"] = {
                    "joint_positions_rad": [float(value) for value in np.asarray(side_qpos).reshape(-1)]
                }
    return {
        "status": "available" if rows else "unavailable_runtime_does_not_expose_robot_articulation",
        "robots": rows,
        "object_layout_symmetric_not_embodiment": True,
    }


def relation_region(target: ActorPose, reference: ActorPose, relation: str) -> bool:
    dx = target.position_xyz_m[0] - reference.position_xyz_m[0]
    dy = target.position_xyz_m[1] - reference.position_xyz_m[1]
    distance = math.hypot(dx, dy)
    side = dx < 0.0 if relation == "left" else dx > 0.0
    return bool(0.08 < distance < 0.20 and side and abs(dy) < 0.05)


def write_video(path: Path, views: Mapping[str, np.ndarray], frames: int = 3) -> dict[str, Any]:
    # Keep the model-blind contract importable in the repository's lightweight
    # validation environment.  The live RoboTwin image installs imageio.
    import imageio.v2 as imageio

    montage = np.concatenate([views[name] for name in CAMERAS], axis=1)
    with imageio.get_writer(path, fps=2, codec="libx264", macro_block_size=1) as writer:
        for _ in range(frames):
            writer.append_data(montage)
    reader = imageio.get_reader(path)
    decoded = sum(1 for _ in reader)
    reader.close()
    if decoded != frames:
        raise ValueError("RoboTwin viewport write/decode gate failed")
    return {**record(path), "decoded_frame_count": decoded, "shape": list(montage.shape)}


def main() -> None:
    args = parse_args()
    study = args.study_root.resolve()
    external = args.external_repository.resolve()
    robotwin = external / "third_party/RoboTwin"
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.repeat_resets < 2 or args.settle_steps < 2:
        raise ValueError("gate requires repeated resets and a stability window")
    if git(study, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("study checkout has tracked changes")
    if git(external, "rev-parse", "HEAD") != FASTWAM_COMMIT:
        raise ValueError("FastWAM revision differs from V3-B007")
    if git(external, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("FastWAM checkout has tracked changes")
    registration = json.loads(args.registration.read_text())
    if (
        registration.get("amendment_id") != "V3-E004"
        or registration.get("status")
        != "prospectively_registered_zero_e004_model_requests_or_behavioral_episodes"
    ):
        raise ValueError("V3-E004 registration is not prospectively frozen")
    queue_rows = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    validate_registered_queue(queue_rows)
    load_candidate(args.candidate, args.candidate_sha256)
    gpu_line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    if args.gpu_uuid not in gpu_line:
        raise ValueError("assigned GPU UUID mismatch")
    if os.environ.get("VK_ICD_FILENAMES") != "/etc/vulkan/icd.d/nvidia_icd.json":
        raise ValueError("NVIDIA Vulkan ICD is not explicitly selected")
    output.mkdir(parents=True)

    os.chdir(robotwin)
    sys.path[:0] = [str(robotwin), str(robotwin / "script")]
    import sapien  # noqa: E402
    from envs.place_a2b_right import place_a2b_right  # noqa: E402
    from envs.utils import create_actor  # noqa: E402

    task_args = load_task_args(robotwin)
    source = place_a2b_right()
    try:
        source.setup_demo(now_ep_num=0, seed=SOURCE_FIXTURE_ENVIRONMENT_SEED, is_test=True, **task_args)
        identity = (
            str(source.selected_modelname_A), int(source.selected_model_id_A),
            str(source.selected_modelname_B), int(source.selected_model_id_B),
        )
        if identity != (*EXPECTED_OBJECT, *EXPECTED_REFERENCE):
            raise ValueError(f"pair03 identity drifted: {identity}")
        validate_live_pose(actor_pose(source.object, EXPECTED_OBJECT), CONTROL_FIXTURE["target"], "source target")
        validate_live_pose(actor_pose(source.target_object, EXPECTED_REFERENCE), CONTROL_FIXTURE["reference"], "source reference")
    finally:
        source.close_env(clear_cache=True)
        del source
        gc.collect()

    class FixedPair03(place_a2b_right):
        fixed_layout = CONTROL_FIXTURE

        def load_actors(self):
            self.selected_modelname_A, self.selected_model_id_A = EXPECTED_OBJECT
            self.selected_modelname_B, self.selected_model_id_B = EXPECTED_REFERENCE
            target = self.fixed_layout["target"]
            reference = self.fixed_layout["reference"]
            self.object = create_actor(
                scene=self,
                pose=sapien.Pose(p=target.position_xyz_m, q=target.quaternion_wxyz),
                modelname=self.selected_modelname_A,
                convex=True,
                model_id=self.selected_model_id_A,
            )
            self.target_object = create_actor(
                scene=self,
                pose=sapien.Pose(p=reference.position_xyz_m, q=reference.quaternion_wxyz),
                modelname=self.selected_modelname_B,
                convex=True,
                model_id=self.selected_model_id_B,
            )
            self.object.set_mass(0.05)
            self.target_object.set_mass(0.05)
            self.add_prohibit_area(self.object, padding=0.05)
            self.add_prohibit_area(self.target_object, padding=0.1)

    rows: list[dict[str, Any]] = []
    videos: dict[str, Any] = {}
    for level in (0.0, 1.0):
        expected_layout = layout_for_level(level)
        FixedPair03.fixed_layout = expected_layout
        for repeat_index in range(args.repeat_resets):
            for relation in ("left", "right"):
                env = FixedPair03()
                label = f"s{int(level * 100):03d}_{relation}_repeat{repeat_index:02d}"
                try:
                    env.setup_demo(now_ep_num=0, seed=CORE_SEEDS[0], is_test=True, **task_args)
                    before = {
                        "target": actor_pose(env.object, EXPECTED_OBJECT),
                        "reference": actor_pose(env.target_object, EXPECTED_REFERENCE),
                    }
                    for _ in range(args.settle_steps):
                        env.scene.step()
                    after = {
                        "target": actor_pose(env.object, EXPECTED_OBJECT),
                        "reference": actor_pose(env.target_object, EXPECTED_REFERENCE),
                    }
                    for name in ("target", "reference"):
                        validate_live_pose(after[name], expected_layout[name], f"{label}/{name}")
                        drift = max(abs(a - b) for a, b in zip(after[name].position_xyz_m, before[name].position_xyz_m))
                        if drift > LIVE_POSITION_TOLERANCE_M:
                            raise ValueError(f"{label}/{name}: translated during stability window")
                    if any(relation_region(after["target"], after["reference"], side) for side in RELATIONS):
                        raise ValueError(f"{label}: a requested relation is already true at reset")
                    observed_residuals = residuals(after)
                    if level == 1.0:
                        if observed_residuals["midline_residual_m"] >= POSITION_TOLERANCE_M:
                            raise ValueError(f"{label}: s1 midline residual failed")
                        if observed_residuals["orientation_residual_rad"] >= ORIENTATION_TOLERANCE_RAD:
                            raise ValueError(f"{label}: s1 orientation residual failed")
                    observation = env.get_obs()
                    views = rgb_views(observation)
                    centers = camera_centers(env)
                    reference_radius = collision_bounding_radius(env.target_object)
                    occlusion = {
                        name: reference_occludes_target(
                            centers[name],
                            after["target"].position_xyz_m,
                            after["reference"].position_xyz_m,
                            reference_radius,
                        )
                        for name in CAMERAS
                    }
                    if any(occlusion.values()):
                        raise ValueError(f"{label}: reference occludes target in {sorted(name for name, value in occlusion.items() if value)}")
                    if repeat_index == 0:
                        videos[f"s{int(level * 100):03d}_{relation}"] = write_video(output / f"s{int(level * 100):03d}_{relation}.mp4", views)
                    rows.append(
                        {
                            "label": label,
                            "symmetry_level_s": level,
                            "relation": relation,
                            "prompt": PROMPTS[relation],
                            "repeat_index": repeat_index,
                            "realised_object_poses": {name: pose.to_json() for name, pose in after.items()},
                            "asymmetry_metric_A": asymmetry_A(after),
                            **observed_residuals,
                            "occlusion_check": occlusion,
                            "occlusion_method": "live_camera_center_to_target_segment_vs_reference_collision_bounding_sphere",
                            "reference_collision_bounding_radius_m": reference_radius,
                            "camera_centers_world_xyz_m": {name: list(value) for name, value in centers.items()},
                            "arm_reset_pose": arm_reset_pose(env),
                            "views": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "pixel_range": int(np.ptp(value))} for name, value in views.items()},
                        }
                    )
                finally:
                    env.close_env(clear_cache=True)
                    del env
                    gc.collect()

    by_key: dict[tuple[float, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((row["symmetry_level_s"], row["repeat_index"]), []).append(row)
    for key, pair in by_key.items():
        if len(pair) != 2 or pair[0]["realised_object_poses"] != pair[1]["realised_object_poses"]:
            raise ValueError(f"{key}: LEFT/RIGHT reset fingerprints differ")

    report = {
        "schema_version": "vla-wam-shared-v3e004-fastwam-robotwin-model-blind-gate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "passed_model_blind_not_released_for_inference",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "candidate": record(args.candidate),
        "candidate_sha256": args.candidate_sha256,
        "registration": record(args.registration),
        "queue": record(args.queue),
        "fastwam_commit": FASTWAM_COMMIT,
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "gpu_uuid": args.gpu_uuid,
        "gpu_query": gpu_line,
        "source_identity": {"target": list(EXPECTED_OBJECT), "reference": list(EXPECTED_REFERENCE)},
        "source_fixture_environment_seed": SOURCE_FIXTURE_ENVIRONMENT_SEED,
        "levels": {
            f"{level:.2f}": {
                "fixture": {name: pose.to_json() for name, pose in layout_for_level(level).items()},
                "registered_asymmetry_metric_A": asymmetry_A(layout_for_level(level)),
                "registered_residuals": residuals(layout_for_level(level)),
            }
            for level in (0.0, 1.0)
        },
        "reset_gate": {
            "repeat_count_per_level_relation": args.repeat_resets,
            "settle_steps": args.settle_steps,
            "left_right_fingerprints_match": True,
            "single_target_and_single_reference": True,
            "native_lateral_axis": "source_x",
            "s1_source_x_centered": True,
            "s1_yaw_self_mirrored": True,
        },
        "tasks": rows,
        "viewport_write_gate": videos,
        "scope_caveat": "Object placement is symmetric about RoboTwin source x=0; robot joints, cameras, and embodiment remain asymmetric and the reset pose is reported rather than assumed symmetric.",
        "release_boundary": "Zero policy requests and zero behavioral episodes were made. A separately hash-bound runtime release is required.",
    }
    report_path = output / "model_blind_gate_report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    print(json.dumps({"passed": True, "report": record(report_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
