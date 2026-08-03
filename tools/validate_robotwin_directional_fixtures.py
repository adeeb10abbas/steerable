#!/usr/bin/env python3
"""Initialize frozen RoboTwin expansion scenes without loading a policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_task_args(robotwin_root: Path, task_name: str, task_config: str) -> dict[str, Any]:
    with (robotwin_root / "task_config" / f"{task_config}.yml").open() as handle:
        task_args = yaml.safe_load(handle)
    with (robotwin_root / "task_config" / "_embodiment_config.yml").open() as handle:
        embodiments = yaml.safe_load(handle)
    with (robotwin_root / "task_config" / "_camera_config.yml").open() as handle:
        cameras = yaml.safe_load(handle)

    embodiment_name = task_args["embodiment"][0]
    robot_file = embodiments[embodiment_name]["file_path"]
    with (robotwin_root / robot_file / "config.yml").open() as handle:
        embodiment_config = yaml.safe_load(handle)
    head_camera = cameras[task_args["camera"]["head_camera_type"]]
    task_args.update(
        task_name=task_name,
        task_config=task_config,
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
        eval_video_save_dir=None,
    )
    return task_args


def relation_region(delta: np.ndarray, direction: str) -> bool:
    distance_xy = float(np.linalg.norm(delta[:2]))
    direction_ok = bool(delta[0] < 0) if direction == "left" else bool(delta[0] > 0)
    return bool(
        0.08 < distance_xy < 0.20
        and direction_ok
        and abs(float(delta[1])) < 0.05
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/directional_expansion.json"),
    )
    parser.add_argument(
        "--robotwin-root",
        type=Path,
        default=Path("/home/ali/projects/EfficientWAM-RoboTwin"),
    )
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/directional_fixture_validation.json"),
    )
    args = parser.parse_args()

    workspace = Path(__file__).resolve().parents[1]
    registry_path = args.registry if args.registry.is_absolute() else workspace / args.registry
    output_path = args.output if args.output.is_absolute() else workspace / args.output
    robotwin_root = args.robotwin_root.expanduser().resolve()
    registry = json.loads(registry_path.read_text())

    os.chdir(robotwin_root)
    sys.path[:0] = [str(robotwin_root), str(robotwin_root / "script")]
    if str(workspace / "tools") not in sys.path:
        sys.path.insert(0, str(workspace / "tools"))
    from vla_wam_v2_protocol import first_seen_object_description

    records: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(registry["scenes"]):
        if scene["phase"] != "new_expansion":
            continue
        task_name = scene["anchor_task"]
        module = __import__(f"envs.{task_name}", fromlist=[task_name])
        task_class = getattr(module, task_name)
        task_args = load_task_args(robotwin_root, task_name, args.task_config)
        env = task_class()
        record: dict[str, Any] = {
            "pair_id": scene["pair_id"],
            "anchor_task": task_name,
            "environment_seed": scene["environment_seed"],
            "sampling_seed": scene["sampling_seed"],
        }
        try:
            env.setup_demo(
                now_ep_num=scene_index,
                seed=scene["environment_seed"],
                is_test=True,
                **task_args,
            )
            object_pose = np.asarray(env.object.get_pose().p, dtype=np.float64)
            target_pose = np.asarray(env.target_object.get_pose().p, dtype=np.float64)
            delta = object_pose - target_pose
            movable_model = str(env.selected_modelname_A)
            reference_model = str(env.selected_modelname_B)
            record.update(
                status="valid",
                movable_model_name=movable_model,
                movable_model_id=int(env.selected_model_id_A),
                movable_description=first_seen_object_description(
                    robotwin_root, movable_model, int(env.selected_model_id_A)
                ),
                reference_model_name=reference_model,
                reference_model_id=int(env.selected_model_id_B),
                reference_description=first_seen_object_description(
                    robotwin_root, reference_model, int(env.selected_model_id_B)
                ),
                initial_object_xyz_m=object_pose.tolist(),
                initial_reference_xyz_m=target_pose.tolist(),
                initial_object_minus_reference_xyz_m=delta.tolist(),
                initial_distance_xy_m=float(np.linalg.norm(delta[:2])),
                initially_in_left_region=relation_region(delta, "left"),
                initially_in_right_region=relation_region(delta, "right"),
            )
            if record["initially_in_left_region"] or record["initially_in_right_region"]:
                record["status"] = "invalid_already_in_requested_region"
        except Exception as error:
            record.update(status="invalid_initialization", error=repr(error))
        finally:
            env.close_env(clear_cache=True)
        records.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    valid_count = sum(record["status"] == "valid" for record in records)
    report = {
        "schema_version": "vla-wam-shared-v2-directional-fixtures-v1",
        "status": "valid" if valid_count == len(records) else "invalid",
        "method": "Model-blind RoboTwin scene initialization; no policy checkpoint or action generator loaded.",
        "registry_path": str(registry_path.relative_to(workspace)),
        "registry_sha256": sha256(registry_path),
        "robotwin_root": str(robotwin_root),
        "task_config": args.task_config,
        "scene_count": len(records),
        "valid_scene_count": valid_count,
        "scenes": records,
        "claim_limit": "Fixture validation establishes technical initialization, object identity, and neutral starting geometry only. It contains no model behavior or success evidence.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "status": report["status"]}, indent=2))
    if report["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
