#!/usr/bin/env python3
"""Static and live zero-model geometry gate for V3-E005.

The live mode reconstructs each of the seven deterministic RoboTwin source
fixtures, captures their full poses, resolves the registered control and
symmetric-object layouts, and repeats LEFT/RIGHT resets without loading or
calling LingBot-VA.  It executes no policy action and no behavioral episode.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from .runtime_contract import (
    SIMULATOR_COMMIT,
    canonical_json_bytes,
    file_record,
    load_registered_bundle,
    require,
    verify_git_identity,
)
from .scene_contract import (
    CAMERAS,
    LIVE_ORIENTATION_TOLERANCE_RAD,
    LIVE_POSITION_TOLERANCE_M,
    ORIENTATION_TOLERANCE_RAD,
    POSITION_TOLERANCE_M,
    SCENE_IDS,
    SCENES,
    ActorPose,
    angular_distance,
    asymmetry_A,
    candidate_sha256,
    layout_for,
    load_candidate,
    residuals,
    symmetric_layout,
    validate_live_snapshot,
    verify_asset_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("static", "live"), default="static")
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--queue", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulator-repository", type=Path)
    parser.add_argument("--expected-study-commit")
    parser.add_argument("--pod")
    parser.add_argument("--pod-uid")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--repeat-resets", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=250)
    return parser.parse_args()


def record(path: Path) -> dict[str, Any]:
    return file_record(Path(path).resolve())


def load_task_args(robotwin_root: Path, task_name: str) -> dict[str, Any]:
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
        task_name=task_name,
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


def task_class(task_name: str) -> type:
    require(task_name in {"place_a2b_left", "place_a2b_right"}, "unsupported E005 task")
    module = importlib.import_module(f"envs.{task_name}")
    value = getattr(module, task_name)
    require(isinstance(value, type), f"RoboTwin task class unavailable: {task_name}")
    return value


def actor_pose(actor: Any, identity: tuple[str, int]) -> ActorPose:
    pose = actor.get_pose()
    return ActorPose(
        tuple(float(value) for value in pose.p),
        tuple(float(value) for value in pose.q),
        identity,
    )


def validate_source_scene(scene_id: str, env: Any) -> dict[str, ActorPose]:
    source = SCENES[scene_id]
    identities = {
        "target": tuple(source["target_asset"]),
        "reference": tuple(source["reference_asset"]),
    }
    observed_identity = {
        "target": (str(env.selected_modelname_A), int(env.selected_model_id_A)),
        "reference": (str(env.selected_modelname_B), int(env.selected_model_id_B)),
    }
    require(observed_identity == identities, f"{scene_id}: deterministic source asset identity drift")
    observed = {
        "target": actor_pose(env.object, identities["target"]),
        "reference": actor_pose(env.target_object, identities["reference"]),
    }
    for role in ("target", "reference"):
        expected = source["historical_settled_centers"][role]
        translation = max(
            abs(a - b)
            for a, b in zip(observed[role].position_xyz_m, expected, strict=True)
        )
        require(
            translation <= LIVE_POSITION_TOLERANCE_M,
            f"{scene_id}/{role}: deterministic source center differs from hash-bound historical evidence by {translation:.6f} m",
        )
    return observed


def build_runtime_task_class(
    candidate: Mapping[str, Any],
    scene_id: str,
    level: float,
    robotwin_root: Path,
    gate_scene: Mapping[str, Any],
) -> type:
    """Build the fixed-layout task class used by the gate and later runtime.

    A runtime must pass the hash-bound scene row from the live gate report;
    the prospective candidate alone intentionally cannot invent unmeasured
    source quaternions.
    """

    root = Path(robotwin_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import sapien
    from envs.utils import create_actor

    source = SCENES[scene_id]
    base = task_class(source["anchor_task"])
    concrete = layout_for(candidate, scene_id, level, gate_scene)
    poses = {role: ActorPose.from_json(concrete[role]) for role in ("target", "reference")}
    target_identity = tuple(source["target_asset"])
    reference_identity = tuple(source["reference_asset"])

    class FixedE005Scene(base):
        e005_scene_id = scene_id
        e005_symmetry_level_s = float(level)
        e005_resolved_layout = concrete

        def load_actors(self):
            self.selected_modelname_A, self.selected_model_id_A = target_identity
            self.selected_modelname_B, self.selected_model_id_B = reference_identity
            target = poses["target"]
            reference = poses["reference"]
            self.object = create_actor(
                scene=self,
                pose=sapien.Pose(
                    p=target.position_xyz_m, q=target.quaternion_wxyz
                ),
                modelname=self.selected_modelname_A,
                convex=True,
                model_id=self.selected_model_id_A,
            )
            self.target_object = create_actor(
                scene=self,
                pose=sapien.Pose(
                    p=reference.position_xyz_m, q=reference.quaternion_wxyz
                ),
                modelname=self.selected_modelname_B,
                convex=True,
                model_id=self.selected_model_id_B,
            )
            self.object.set_mass(0.05)
            self.target_object.set_mass(0.05)
            self.add_prohibit_area(self.object, padding=0.05)
            self.add_prohibit_area(self.target_object, padding=0.1)

    FixedE005Scene.__name__ = (
        f"V3E005_{scene_id}_s{int(round(float(level) * 100)):03d}"
    )
    return FixedE005Scene


def rgb_views(observation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = observation.get("observation", observation)
    output: dict[str, np.ndarray] = {}
    for name in CAMERAS:
        row = rows.get(name)
        require(isinstance(row, Mapping) and "rgb" in row, f"required RoboTwin view missing: {name}")
        value = np.asarray(row["rgb"])
        require(value.ndim == 3 and value.shape[-1] in (3, 4), f"malformed RoboTwin view: {name}")
        value = value[..., :3]
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if value.max() <= 1.0 else value, 0, 255).astype(np.uint8)
        require(int(np.ptp(value)) > 0, f"blank RoboTwin view: {name}")
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
            if len(values) == 3 and all(math.isfinite(value) for value in values):
                return values
        current = getattr(current, "entity", None) or getattr(current, "actor", None)
    return None


def camera_centers(env: Any) -> dict[str, tuple[float, float, float]]:
    output: dict[str, tuple[float, float, float]] = {}
    entities: list[Any] = []
    rig = getattr(env, "cameras", None)
    named_from_rig: dict[str, Any] = {}
    if rig is not None:
        entities.extend(
            camera
            for camera in (
                getattr(rig, "left_camera", None),
                getattr(rig, "right_camera", None),
            )
            if camera is not None
        )
        names = list(getattr(rig, "static_camera_name", []))
        static = list(getattr(rig, "static_camera_list", []))
        entities.extend(static)
        named_from_rig = dict(zip(names, static, strict=False))
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
        candidates.extend(item for item in entities if getattr(item, "name", None) == name)
        candidates.extend(
            item
            for item in entities
            if getattr(getattr(item, "entity", None), "name", None) == name
        )
        center = next(
            (value for value in (_entity_pose(item) for item in candidates) if value is not None),
            None,
        )
        require(center is not None, f"{name}: calibrated live camera pose unavailable")
        output[name] = center
    return output


def registered_camera_components(env: Any) -> dict[str, Any]:
    rig = getattr(env, "cameras", None)
    require(rig is not None, "RoboTwin camera rig unavailable")
    static = dict(
        zip(
            list(getattr(rig, "static_camera_name", [])),
            list(getattr(rig, "static_camera_list", [])),
            strict=False,
        )
    )
    output = {
        "head_camera": static.get("head_camera"),
        "left_camera": getattr(rig, "left_camera", None),
        "right_camera": getattr(rig, "right_camera", None),
    }
    for name, camera in output.items():
        require(camera is not None and hasattr(camera, "get_picture"), f"{name}: raw segmentation camera unavailable")
    return output


def _per_scene_id(actor: Any) -> int:
    candidates = [actor, getattr(actor, "actor", None), getattr(actor, "entity", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for name in ("get_per_scene_id", "per_scene_id"):
            value = getattr(candidate, name, None)
            if callable(value):
                value = value()
            if isinstance(value, (int, np.integer)) and int(value) >= 0:
                return int(value)
    raise ValueError("target actor per-scene segmentation id unavailable")


def target_visibility_pixels(env: Any, target_actor: Any) -> dict[str, int]:
    """Return target segmentation pixels as a diagnostic, not an occlusion gate.

    The registered check is specifically whether the reference object blocks
    the camera-to-target segment.  A zero segmentation count can instead mean
    that a wrist camera's current field of view does not contain the target,
    so treating it as reference occlusion would enforce a stronger,
    unregistered condition.
    """

    target_id = _per_scene_id(target_actor)
    output: dict[str, int] = {}
    for name, camera in registered_camera_components(env).items():
        segmentation = np.asarray(camera.get_picture("Segmentation"))
        require(
            segmentation.ndim == 3 and segmentation.shape[-1] >= 2,
            f"{name}: malformed actor segmentation buffer",
        )
        count = int(np.count_nonzero(segmentation[..., 1].astype(np.int64) == target_id))
        output[name] = count
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
                scale = np.asarray(
                    getattr(shape, "scale", [1.0, 1.0, 1.0]), dtype=np.float64
                ).reshape(-1)
                if scale.size == 1:
                    scale = np.repeat(scale, 3)
                return float(np.max(np.linalg.norm(row[:, :3] * scale[:3], axis=1)))
    return None


def collision_bounding_radius(actor: Any) -> float:
    shapes: list[Any] = []
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
    radii = [value for value in (_shape_radius(shape) for shape in shapes) if value is not None]
    require(bool(radii), "reference live collision geometry lacks a bounding radius")
    radius = max(radii)
    require(0.001 < radius < 0.5, f"reference collision radius is implausible: {radius}")
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
    require(denominator > 0.0, "camera and target centers coincide")
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
            rows[name] = {
                "joint_positions_rad": [float(value) for value in np.asarray(qpos).reshape(-1)]
            }
        for side in ("left", "right"):
            entity = getattr(robot, f"{side}_entity", None)
            side_qpos = (
                entity.get_qpos()
                if entity is not None and hasattr(entity, "get_qpos")
                else None
            )
            if side_qpos is None and hasattr(robot, f"get_{side}_arm_jointState"):
                side_qpos = getattr(robot, f"get_{side}_arm_jointState")()
            if side_qpos is not None:
                rows[f"{name}.{side}"] = {
                    "joint_positions_rad": [
                        float(value) for value in np.asarray(side_qpos).reshape(-1)
                    ]
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
    import imageio.v2 as imageio

    montage = np.concatenate([views[name] for name in CAMERAS], axis=1)
    with imageio.get_writer(path, fps=2, codec="libx264", macro_block_size=1) as writer:
        for _ in range(frames):
            writer.append_data(montage)
    reader = imageio.get_reader(path)
    decoded = sum(1 for _ in reader)
    reader.close()
    require(decoded == frames, "RoboTwin viewport write/decode gate failed")
    return {**record(path), "decoded_frame_count": decoded, "shape": list(montage.shape)}


def static_report(
    *,
    bundle: Any,
    candidate_path: Path,
    expected_candidate_sha256: str,
    candidate: Mapping[str, Any],
    asset_files: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "vla-wam-shared-v3e005-static-scene-contract-check-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "arena": "robotwin",
        "model_id": "lingbot_va_robotwin",
        "status": "static_contract_valid_live_gate_not_run",
        "passed": False,
        "static_contract_passed": True,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": expected_candidate_sha256,
        "candidate": record(candidate_path),
        "scene_count": len(candidate["scenes"]),
        "registered_cell_count": len(bundle.cells),
        "asset_file_checks": asset_files,
        "model_request_count": 0,
        "model_action_request_count": 0,
        "behavioral_episode_count": 0,
        "release_boundary": "This static check does not replace the live geometry/occlusion gate and cannot authorize inference.",
    }


def run_live(args: argparse.Namespace, bundle: Any, candidate: Mapping[str, Any]) -> dict[str, Any]:
    require(args.simulator_repository is not None, "live gate requires --simulator-repository")
    require(args.expected_study_commit, "live gate requires --expected-study-commit")
    require(args.pod and args.pod_uid and args.gpu_uuid, "live gate requires pod, pod UID, and GPU UUID")
    require(args.repeat_resets >= 2, "live gate requires at least two reset repeats")
    require(args.settle_steps >= 2, "live gate requires a stability window")
    study = args.study_root.resolve()
    robotwin = args.simulator_repository.resolve()
    verify_git_identity(study, str(args.expected_study_commit), require_untracked_clean=True)
    verify_git_identity(robotwin, SIMULATOR_COMMIT)
    assets = verify_asset_files(candidate, robotwin)
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    gpu_line = next((line for line in gpu_lines if str(args.gpu_uuid) in line), None)
    require(gpu_line is not None, "assigned GPU UUID mismatch")
    require(
        os.environ.get("VK_ICD_FILENAMES") == "/etc/vulkan/icd.d/nvidia_icd.json",
        "NVIDIA Vulkan ICD is not explicitly selected",
    )
    if str(robotwin) not in sys.path:
        sys.path.insert(0, str(robotwin))
    os.chdir(robotwin)

    output = args.output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    scenes_report: dict[str, Any] = {}
    videos: dict[str, Any] = {}

    for scene_id in SCENE_IDS:
        source_spec = SCENES[scene_id]
        base = task_class(source_spec["anchor_task"])
        source_env = base()
        try:
            task_args = load_task_args(robotwin, source_spec["anchor_task"])
            source_env.setup_demo(
                now_ep_num=0,
                seed=source_spec["source_fixture_environment_seed"],
                is_test=True,
                **task_args,
            )
            source_layout = validate_source_scene(scene_id, source_env)
        finally:
            source_env.close_env(clear_cache=True)
            del source_env
            gc.collect()

        symmetric = symmetric_layout(source_layout)
        scene_gate: dict[str, Any] = {
            "scene_id": scene_id,
            "anchor_task": source_spec["anchor_task"],
            "source_fixture_environment_seed": source_spec["source_fixture_environment_seed"],
            "source_control_layout": {
                role: pose.to_json() for role, pose in source_layout.items()
            },
            "source_control_layout_sha256": hashlib.sha256(
                canonical_json_bytes(
                    {role: pose.to_json() for role, pose in source_layout.items()}
                )
            ).hexdigest(),
            "resolved_layouts": {
                "0.00": {role: pose.to_json() for role, pose in source_layout.items()},
                "1.00": {role: pose.to_json() for role, pose in symmetric.items()},
            },
            "asset_contract": candidate["scenes"][scene_id]["assets"],
            "reset_evidence": [],
        }
        first_seed = min(
            cell.environment_seed for cell in bundle.cells if cell.scene_id == scene_id
        )
        for level in (0.0, 1.0):
            fixed = build_runtime_task_class(candidate, scene_id, level, robotwin, scene_gate)
            for repeat_index in range(args.repeat_resets):
                for relation in ("left", "right"):
                    env = fixed()
                    label = f"{scene_id}_s{int(level * 100):03d}_{relation}_repeat{repeat_index:02d}"
                    try:
                        env.setup_demo(
                            now_ep_num=0,
                            seed=first_seed,
                            is_test=True,
                            **load_task_args(robotwin, source_spec["anchor_task"]),
                        )
                        identities = {
                            role: tuple(source_spec[f"{role}_asset"])
                            for role in ("target", "reference")
                        }
                        before = {
                            "target": actor_pose(env.object, identities["target"]),
                            "reference": actor_pose(env.target_object, identities["reference"]),
                        }
                        for _ in range(args.settle_steps):
                            env.scene.step()
                        after = {
                            "target": actor_pose(env.object, identities["target"]),
                            "reference": actor_pose(env.target_object, identities["reference"]),
                        }
                        expected_json = layout_for(candidate, scene_id, level, scene_gate)
                        expected = {
                            role: ActorPose.from_json(expected_json[role])
                            for role in ("target", "reference")
                        }
                        for role in ("target", "reference"):
                            drift = max(
                                abs(a - b)
                                for a, b in zip(
                                    before[role].position_xyz_m,
                                    after[role].position_xyz_m,
                                    strict=True,
                                )
                            )
                            require(drift <= LIVE_POSITION_TOLERANCE_M, f"{label}/{role}: translated during stability window")
                            require(
                                angular_distance(after[role].quaternion_wxyz, before[role].quaternion_wxyz)
                                <= LIVE_ORIENTATION_TOLERANCE_RAD,
                                f"{label}/{role}: rotated during stability window",
                            )
                            require(
                                max(
                                    abs(a - b)
                                    for a, b in zip(
                                        after[role].position_xyz_m,
                                        expected[role].position_xyz_m,
                                        strict=True,
                                    )
                                )
                                <= LIVE_POSITION_TOLERANCE_M,
                                f"{label}/{role}: settled position differs from resolved layout",
                            )
                        require(
                            not any(
                                relation_region(after["target"], after["reference"], side)
                                for side in ("left", "right")
                            ),
                            f"{label}: requested relation is already true at reset",
                        )
                        views = rgb_views(env.get_obs())
                        visible_pixels = target_visibility_pixels(env, env.object)
                        centers = camera_centers(env)
                        reference_radius = collision_bounding_radius(env.target_object)
                        occlusion = {
                            camera: reference_occludes_target(
                                centers[camera],
                                after["target"].position_xyz_m,
                                after["reference"].position_xyz_m,
                                reference_radius,
                            )
                            for camera in CAMERAS
                        }
                        snapshot = {
                            "label": label,
                            "scene_id": scene_id,
                            "symmetry_level_s": level,
                            "relation": relation,
                            "prompt": source_spec["prompts"][relation],
                            "repeat_index": repeat_index,
                            "realised_object_poses": {
                                role: pose.to_json() for role, pose in after.items()
                            },
                            "asset_contract": candidate["scenes"][scene_id]["assets"],
                            "occlusion_check": occlusion,
                            "occlusion_method": "live_camera_center_to_target_segment_vs_reference_collision_bounding_sphere",
                            "reference_collision_bounding_radius_m": reference_radius,
                            "camera_centers_world_xyz_m": {
                                name: list(value) for name, value in centers.items()
                            },
                            "arm_reset_pose": arm_reset_pose(env),
                            "views": {
                                name: {
                                    "shape": list(value.shape),
                                    "dtype": str(value.dtype),
                                    "pixel_range": int(np.ptp(value)),
                                    "target_visible_pixels": visible_pixels[name],
                                }
                                for name, value in views.items()
                            },
                        }
                        snapshot["validation"] = validate_live_snapshot(
                            candidate, scene_id, level, snapshot, scene_gate
                        )
                        if repeat_index == 0:
                            video_name = f"{scene_id}_s{int(level * 100):03d}_{relation}.mp4"
                            videos[video_name] = write_video(output / video_name, views)
                        scene_gate["reset_evidence"].append(snapshot)
                    finally:
                        env.close_env(clear_cache=True)
                        del env
                        gc.collect()

        by_key: dict[tuple[float, int], list[dict[str, Any]]] = {}
        for row in scene_gate["reset_evidence"]:
            by_key.setdefault(
                (float(row["symmetry_level_s"]), int(row["repeat_index"])), []
            ).append(row)
        for key, pair in by_key.items():
            require(len(pair) == 2, f"{scene_id}/{key}: LEFT/RIGHT gate pair incomplete")
            left = next(row for row in pair if row["relation"] == "left")
            right = next(row for row in pair if row["relation"] == "right")
            require(
                left["realised_object_poses"] == right["realised_object_poses"],
                f"{scene_id}/{key}: LEFT/RIGHT non-language reset fingerprints differ",
            )
            require(
                left["arm_reset_pose"] == right["arm_reset_pose"],
                f"{scene_id}/{key}: LEFT/RIGHT arm reset fingerprints differ",
            )
        scenes_report[scene_id] = scene_gate

    return {
        "schema_version": "vla-wam-shared-v3e005-seven-scene-model-blind-gate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E005",
        "model_id": "lingbot_va_robotwin",
        "arena": "robotwin",
        "status": "passed_model_blind_not_released_for_inference",
        "passed": True,
        "registration_sha256": bundle.registration_sha256,
        "queue_sha256": bundle.queue_sha256,
        "candidate_sha256": args.candidate_sha256,
        "registration": record(bundle.registration_path),
        "queue": record(bundle.queue_path),
        "candidate": record(args.candidate),
        "simulator_repository_commit": SIMULATOR_COMMIT,
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "gpu_uuid": args.gpu_uuid,
        "gpu_query": gpu_line,
        "asset_file_checks": assets,
        "scene_count": len(scenes_report),
        "reset_count": sum(
            len(scene["reset_evidence"]) for scene in scenes_report.values()
        ),
        "scenes": scenes_report,
        "viewport_write_gate": videos,
        "reset_gate": {
            "repeat_count_per_scene_level_relation": args.repeat_resets,
            "settle_steps": args.settle_steps,
            "left_right_non_language_fingerprints_match": True,
            "single_target_and_single_reference_per_scene": True,
            "s1_native_x_centered": True,
            "s1_world_yaw_self_mirrored": True,
            "all_registered_cameras_checked": list(CAMERAS),
        },
        "model_request_count": 0,
        "model_action_request_count": 0,
        "behavioral_episode_count": 0,
        "scope_caveat": "Object placement is symmetric about the calibrated robot midline; robot joints, cameras, and embodiment remain asymmetric and are logged, not assumed symmetric.",
        "release_boundary": "Zero policy requests, zero policy actions, and zero behavioral episodes were made. A separate hash-bound runtime release remains required.",
    }


def main() -> None:
    args = parse_args()
    study = args.study_root.expanduser().resolve()
    bundle = load_registered_bundle(
        study,
        registration_path=args.registration,
        queue_path=args.queue,
    )
    candidate = load_candidate(
        args.candidate,
        args.candidate_sha256,
        bundle.registration_sha256,
        bundle.queue_sha256,
    )
    require(
        args.candidate_sha256 == candidate_sha256(),
        "candidate digest does not match reconstructed E005 contract",
    )
    output = args.output_dir.expanduser().resolve()
    require(not output.exists(), f"refusing to overwrite {output}")
    if args.mode == "static":
        asset_files = None
        if args.simulator_repository is not None:
            verify_git_identity(args.simulator_repository, SIMULATOR_COMMIT)
            asset_files = verify_asset_files(candidate, args.simulator_repository)
        output.mkdir(parents=True)
        report = static_report(
            bundle=bundle,
            candidate_path=args.candidate,
            expected_candidate_sha256=args.candidate_sha256,
            candidate=candidate,
            asset_files=asset_files,
        )
        report_path = output / "static_contract_report.json"
    else:
        report = run_live(args, bundle, candidate)
        report_path = output / "model_blind_gate_report.json"
    report_path.write_bytes(canonical_json_bytes(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": record(report_path),
                "model_request_count": 0,
                "model_action_request_count": 0,
                "behavioral_episode_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
