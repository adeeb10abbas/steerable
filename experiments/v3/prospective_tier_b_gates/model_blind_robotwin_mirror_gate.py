#!/usr/bin/env python3
"""Discover and validate the V3-B007 pair03 mirror with zero model requests."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
import yaml


FASTWAM_COMMIT = "068d3fd70c89df3726c09893f47b75a624b20c02"
ENVIRONMENT_SEED = 4_300_003
TASK = "place_a2b_right"
EXPECTED_OBJECT = ("086_woodenblock", 1)
EXPECTED_REFERENCE = ("081_playingcards", 1)
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--external-repository", type=Path, required=True)
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


def record(path: Path) -> dict:
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}


def git(repository: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def load_task_args(robotwin_root: Path) -> dict:
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
    head_camera = cameras[task_args["camera"]["head_camera_type"]]
    task_args.update(
        task_name=TASK,
        task_config="demo_clean",
        eval_mode=True,
        save_data=False,
        collect_data=False,
        render_freq=0,
        head_camera_h=head_camera["h"],
        head_camera_w=head_camera["w"],
        left_robot_file=robot_file,
        right_robot_file=robot_file,
        dual_arm_embodied=True,
        left_embodiment_config=embodiment_config,
        right_embodiment_config=embodiment_config,
    )
    return task_args


def pose_record(actor) -> dict:
    pose = actor.get_pose()
    return {
        "position_xyz_m": [float(value) for value in pose.p],
        "quaternion_wxyz": [float(value) for value in pose.q],
        "linear_velocity_m_s": [float(value) for value in actor.get_velocity()],
        "angular_velocity_rad_s": [float(value) for value in actor.get_angular_velocity()],
    }


def relation_region(object_position: list[float], reference_position: list[float], relation: str) -> bool:
    delta_x = object_position[0] - reference_position[0]
    delta_y = object_position[1] - reference_position[1]
    distance = float(np.hypot(delta_x, delta_y))
    side = delta_x < 0.0 if relation == "left" else delta_x > 0.0
    return bool(0.08 < distance < 0.20 and side and abs(delta_y) < 0.05)


def rgb_views(observation: dict) -> dict[str, np.ndarray]:
    camera_rows = observation.get("observation", {})
    output = {}
    for name in ("head_camera", "left_camera", "right_camera"):
        row = camera_rows.get(name)
        if not isinstance(row, dict) or "rgb" not in row:
            raise ValueError(f"required SAPIEN view missing: {name}")
        value = np.asarray(row["rgb"])
        if value.ndim != 3 or value.shape[-1] not in (3, 4):
            raise ValueError(f"malformed SAPIEN RGB view: {name} {value.shape}")
        value = value[..., :3]
        if value.dtype != np.uint8:
            value = np.clip(value * 255.0 if value.max() <= 1.0 else value, 0, 255).astype(np.uint8)
        if not np.ptp(value):
            raise ValueError(f"blank SAPIEN RGB view: {name}")
        output[name] = value
    return output


def write_video(path: Path, views: dict[str, np.ndarray], frames: int = 3) -> dict:
    montage = np.concatenate([views[name] for name in ("left_camera", "head_camera", "right_camera")], axis=1)
    with imageio.get_writer(path, fps=2, codec="libx264", macro_block_size=1) as writer:
        for _ in range(frames):
            writer.append_data(montage)
    reader = imageio.get_reader(path)
    decoded = sum(1 for _ in reader)
    reader.close()
    if decoded != frames:
        raise ValueError(f"viewport video decoded {decoded}, expected {frames}")
    return {**record(path), "decoded_frame_count": decoded, "shape": list(montage.shape)}


def main() -> None:
    args = parse_args()
    study_root = args.study_root.resolve()
    external = args.external_repository.resolve()
    robotwin_root = external / "third_party/RoboTwin"
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    if args.repeat_resets < 2 or args.settle_steps < 2:
        raise ValueError("gate requires repeated resets and a stability window")
    if git(study_root, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("study checkout has tracked changes")
    if git(external, "rev-parse", "HEAD") != FASTWAM_COMMIT:
        raise ValueError("FastWAM revision differs from frozen runtime")
    if git(external, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("FastWAM checkout has tracked changes")
    gpu_line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()[0]
    if args.gpu_uuid not in gpu_line:
        raise ValueError("assigned GPU UUID mismatch")
    if os.environ.get("VK_ICD_FILENAMES") != "/etc/vulkan/icd.d/nvidia_icd.json":
        raise ValueError("NVIDIA Vulkan ICD is not explicitly selected")

    output_dir.mkdir(parents=True)
    os.chdir(robotwin_root)
    sys.path[:0] = [str(robotwin_root), str(robotwin_root / "script")]
    import sapien  # noqa: E402
    from envs.place_a2b_right import place_a2b_right  # noqa: E402
    from envs.utils import create_actor  # noqa: E402

    task_args = load_task_args(robotwin_root)
    source_env = place_a2b_right()
    try:
        source_env.setup_demo(now_ep_num=0, seed=ENVIRONMENT_SEED, is_test=True, **task_args)
        identity = (
            str(source_env.selected_modelname_A),
            int(source_env.selected_model_id_A),
            str(source_env.selected_modelname_B),
            int(source_env.selected_model_id_B),
        )
        if identity != (*EXPECTED_OBJECT, *EXPECTED_REFERENCE):
            raise ValueError(f"pair03 identity drifted: {identity}")
        source_object = pose_record(source_env.object)
        source_reference = pose_record(source_env.target_object)
    finally:
        source_env.close_env(clear_cache=True)
        del source_env
        gc.collect()

    control = {
        "object": source_object,
        "reference": source_reference,
    }
    mirrored = json.loads(json.dumps(control))
    mirrored["object"]["position_xyz_m"][0] *= -1.0
    mirrored["reference"]["position_xyz_m"][0] *= -1.0

    class FixedPair03(place_a2b_right):
        fixed_layout = control

        def load_actors(self):
            self.selected_modelname_A, self.selected_model_id_A = EXPECTED_OBJECT
            self.selected_modelname_B, self.selected_model_id_B = EXPECTED_REFERENCE
            object_pose = self.fixed_layout["object"]
            reference_pose = self.fixed_layout["reference"]
            self.object = create_actor(
                scene=self,
                pose=sapien.Pose(p=object_pose["position_xyz_m"], q=object_pose["quaternion_wxyz"]),
                modelname=self.selected_modelname_A,
                convex=True,
                model_id=self.selected_model_id_A,
            )
            self.target_object = create_actor(
                scene=self,
                pose=sapien.Pose(p=reference_pose["position_xyz_m"], q=reference_pose["quaternion_wxyz"]),
                modelname=self.selected_modelname_B,
                convex=True,
                model_id=self.selected_model_id_B,
            )
            self.object.set_mass(0.05)
            self.target_object.set_mass(0.05)
            self.add_prohibit_area(self.object, padding=0.05)
            self.add_prohibit_area(self.target_object, padding=0.1)

    rows = []
    videos = {}
    for arm, layout in (("control", control), ("position_mirrored", mirrored)):
        FixedPair03.fixed_layout = layout
        for relation in ("left", "right"):
            env = FixedPair03()
            label = f"{arm}_{relation}"
            try:
                env.setup_demo(now_ep_num=0, seed=ENVIRONMENT_SEED, is_test=True, **task_args)
                before_object = pose_record(env.object)
                before_reference = pose_record(env.target_object)
                for _ in range(args.settle_steps):
                    env.scene.step()
                after_object = pose_record(env.object)
                after_reference = pose_record(env.target_object)
                for name, row in (("object", after_object), ("reference", after_reference)):
                    if max(abs(value) for value in row["linear_velocity_m_s"]) > 0.02:
                        raise ValueError(f"{label} {name} linear instability")
                    if max(abs(value) for value in row["angular_velocity_rad_s"]) > 0.20:
                        raise ValueError(f"{label} {name} angular instability")
                    expected = layout[name]["position_xyz_m"]
                    if max(abs(a - b) for a, b in zip(row["position_xyz_m"], expected)) > 0.003:
                        raise ValueError(f"{label} {name} missed mirrored position tolerance")
                object_position = after_object["position_xyz_m"]
                reference_position = after_reference["position_xyz_m"]
                predicates = {
                    side: relation_region(object_position, reference_position, side)
                    for side in ("left", "right")
                }
                if any(predicates.values()):
                    raise ValueError(f"{label} is not neutral at reset")
                observation = env.get_obs()
                views = rgb_views(observation)
                videos[label] = write_video(output_dir / f"{label}_repeated_frame.mp4", views)
                rows.append({
                    "label": label,
                    "arm": arm,
                    "relation": relation,
                    "prompt": PROMPTS[relation],
                    "identity": {"object": list(EXPECTED_OBJECT), "reference": list(EXPECTED_REFERENCE)},
                    "before_stability_window": {"object": before_object, "reference": before_reference},
                    "after_stability_window": {"object": after_object, "reference": after_reference},
                    "predicates_at_reset": predicates,
                    "views": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "pixel_range": int(np.ptp(value))} for name, value in views.items()},
                })
            finally:
                env.close_env(clear_cache=True)
                del env
                gc.collect()

    by_label = {row["label"]: row for row in rows}
    for arm in ("control", "position_mirrored"):
        left = by_label[f"{arm}_left"]["after_stability_window"]
        right = by_label[f"{arm}_right"]["after_stability_window"]
        if left != right:
            raise ValueError(f"{arm} LEFT/RIGHT physical fingerprints differ")
    for name in ("object", "reference"):
        control_position = by_label["control_left"]["after_stability_window"][name]["position_xyz_m"]
        mirror_position = by_label["position_mirrored_left"]["after_stability_window"][name]["position_xyz_m"]
        expected = [-control_position[0], control_position[1], control_position[2]]
        if max(abs(a - b) for a, b in zip(expected, mirror_position)) > 0.003:
            raise ValueError(f"live x reflection failed for {name}")

    report = {
        "schema_version": "vla-wam-shared-v3b-fastwam-robotwin-mirror-model-blind-gate-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "model_id": "fastwam_robotwin",
        "arena": "robotwin",
        "status": "passed_model_blind_not_released_for_behavior",
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "pod": args.pod,
        "pod_uid": args.pod_uid,
        "gpu_uuid": args.gpu_uuid,
        "gpu_query": gpu_line,
        "study_checkout": {"commit": git(study_root, "rev-parse", "HEAD"), "tracked_diff_empty": True},
        "fastwam_repository": {"commit": FASTWAM_COMMIT, "tracked_diff_empty": True, "path": str(external)},
        "simulator_repository": {"commit": FASTWAM_COMMIT, "path": str(robotwin_root)},
        "gate_source": record(Path(__file__)),
        "task_source": record(robotwin_root / "envs/place_a2b_right.py"),
        "task_config": record(robotwin_root / "task_config/demo_clean.yml"),
        "environment_seed": ENVIRONMENT_SEED,
        "source_identity": {"object": list(EXPECTED_OBJECT), "reference": list(EXPECTED_REFERENCE)},
        "derived_numeric_fixture": {"control": control, "position_mirrored": mirrored, "transform": "object and reference centers only: (x,y,z)->(-x,y,z); quaternions unchanged"},
        "reset_gate": {"repeat_count_per_arm": args.repeat_resets, "settle_steps_after_native_stability_gate": args.settle_steps, "left_right_physical_fingerprints_equal_within_arm": True, "neither_predicate_true_at_every_reset": True, "live_center_reflection_passed": True},
        "renderer": {"backend": "headless SAPIEN Vulkan ray tracing", "vulkan_icd": record(Path("/etc/vulkan/icd.d/nvidia_icd.json")), "all_required_rgb_views_nonblank": True},
        "tasks": rows,
        "viewport_write_gate": videos,
        "release_boundary": "This model-blind gate freezes the pair03 numeric mirror but releases zero behavioral cells. A separately validated queue/release amendment is required before inference.",
    }
    report_path = output_dir / "model_blind_gate_report.json"
    report_path.write_text(json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": True, "report": record(report_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
