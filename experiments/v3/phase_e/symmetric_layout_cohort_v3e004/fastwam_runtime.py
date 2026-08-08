#!/usr/bin/env python3
"""Execute the registered V3-E004 FastWAM RoboTwin stretch queue.

The external FastWAM V3-B007 controller remains untouched.  This overlay
supplies the E004 s=0/s=1 task classes, enforces seeds 9400--9426 and exact
prompts, captures the live reset pose, and writes the E004 per-episode fields.
Infrastructure-invalid partial seed blocks are kept outside the behavioral
denominator.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .fastwam_model_blind_gate import actor_pose, arm_reset_pose
from .fastwam_robotwin import (
    ARENA,
    CHECKPOINT_SHA256,
    CONTROL_FIXTURE,
    CORE_SEEDS,
    DATASET_STATS_SHA256,
    EXPECTED_OBJECT,
    EXPECTED_REFERENCE,
    FASTWAM_COMMIT,
    LEVELS,
    MODEL_ID,
    PROMPTS,
    RELATIONS,
    SOURCE_RELEASE_QUEUE_SHA256,
    ActorPose,
    asymmetry_A,
    canonical_json_bytes,
    layout_for_level,
    load_candidate,
    residuals,
    validate_registered_queue,
)


MODEL_SPECIFIC_SCHEMA = "vla-wam-shared-v3-robotwin-fixed-observation-release-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--external-repository", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--model-blind-gate", type=Path, required=True)
    parser.add_argument("--model-blind-gate-sha256", required=True)
    parser.add_argument("--model-specific-gate", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(row))


def validate_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing simulator video: {path}")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def relation_region(step: Mapping[str, Any], relation: str) -> bool:
    dx = float(step["object_minus_target_x"])
    dy = float(step["object_minus_target_y"])
    distance = math.hypot(dx, dy)
    side = dx < 0.0 if relation == "left" else dx > 0.0
    return bool(0.08 < distance < 0.20 and side and abs(dy) < 0.05)


def first_sustained(values: list[bool], length: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(values):
        run = run + 1 if value else 0
        if run == length:
            return index - length + 1
    return None


def final_sustained(values: list[bool], length: int = 3) -> bool:
    return len(values) >= length and all(values[-length:])


def episode_measures(result: Mapping[str, Any], trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    relation = str(result["requested_relation"])
    opposite = "right" if relation == "left" else "left"
    initial_z = float(trajectory[0]["object_xyz"][2])
    pickup = [
        float(step["object_xyz"][2]) >= initial_z + 0.03 and not bool(step["grippers_open"])
        for step in trajectory
    ]
    requested = [relation_region(step, relation) for step in trajectory]
    opposite_mask = [relation_region(step, opposite) for step in trajectory]
    grasp_step = first_sustained(pickup)
    entry_step = first_sustained(requested)
    if bool(result["requested_success"]):
        category = "correct"
    elif grasp_step is None:
        category = "pick_failed"
    elif final_sustained(opposite_mask):
        category = "wrong_side"
    elif final_sustained(requested) and not bool(trajectory[-1]["grippers_open"]):
        category = "release_failed"
    else:
        category = "transport_failed"
    positions = np.asarray([step["object_xyz"] for step in trajectory], dtype=np.float64)
    native_x = positions[:, 0]
    signed_lateral = -native_x
    return {
        "success": bool(result["requested_success"]),
        "failure_category": category,
        "signed_final_lateral_offset": -float(trajectory[-1]["object_minus_target_x"]),
        "requested_side_depth": float(-float(trajectory[-1]["object_minus_target_x"])) * (1.0 if relation == "left" else -1.0),
        "cone_entry_step": entry_step,
        "cone_entry_sustained": entry_step is not None,
        "episode_length": int(result["actions_executed"]),
        # The released RoboTwin trajectory has no contact-force channel.  NR
        # remains null rather than being converted to zero.
        "time_to_first_contact": None,
        "grasp_step": grasp_step,
        "cumulative_lateral_path": float(np.abs(np.diff(signed_lateral)).sum()) if len(signed_lateral) > 1 else 0.0,
        "peak_lateral_excursion": float(np.max(np.abs(signed_lateral - signed_lateral[0]))),
    }


def action_pair(left_path: Path, right_path: Path) -> dict[str, Any]:
    with np.load(left_path) as payload:
        left = np.asarray(payload["executed"], dtype=np.float64)
    with np.load(right_path) as payload:
        right = np.asarray(payload["executed"], dtype=np.float64)
    count = min(10, len(left), len(right))
    rms = float(np.sqrt(np.mean(np.square(left[:count] - right[:count])))) if count else None
    return {
        "actions_compared": count,
        "first_10_action_rms": rms,
        "action_distinct": bool(rms is not None and rms > 0.0),
    }


def fingerprint(snapshot: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()


def task_classes(robotwin_root: Path):
    os.chdir(robotwin_root)
    sys.path[:0] = [str(robotwin_root), str(robotwin_root / "script"), str(robotwin_root / "description/utils")]
    import sapien
    from envs.place_a2b_right import place_a2b_right
    from envs.utils import create_actor

    def build(level: float):
        layout = layout_for_level(level)

        class FixedPair03(place_a2b_right):
            latest_reset_snapshot: dict[str, Any] | None = None

            def load_actors(self):
                self.selected_modelname_A, self.selected_model_id_A = EXPECTED_OBJECT
                self.selected_modelname_B, self.selected_model_id_B = EXPECTED_REFERENCE
                target = layout["target"]
                reference = layout["reference"]
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

            def setup_demo(self, *args, **kwargs):
                value = super().setup_demo(*args, **kwargs)
                poses = {
                    "target": actor_pose(self.object, EXPECTED_OBJECT),
                    "reference": actor_pose(self.target_object, EXPECTED_REFERENCE),
                }
                self.__class__.latest_reset_snapshot = {
                    "realised_object_poses": {name: pose.to_json() for name, pose in poses.items()},
                    "arm_reset_pose": arm_reset_pose(self),
                    "asymmetry_metric_A": asymmetry_A(poses),
                    **residuals(poses),
                }
                return value

        FixedPair03.__name__ = f"v3e004_fastwam_s{int(level * 100):03d}"
        return FixedPair03

    return {level: build(level) for level in LEVELS}


def main() -> None:
    args = parse_args()
    study = args.study_root.resolve()
    external = args.external_repository.resolve()
    robotwin = external / "third_party/RoboTwin"
    output = args.output_dir.resolve()
    if git(study, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("study checkout has tracked changes")
    if git(external, "rev-parse", "HEAD") != FASTWAM_COMMIT:
        raise ValueError("FastWAM revision drift")
    if git(external, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("FastWAM checkout has tracked changes")
    for path, expected in (
        (args.candidate.resolve(), args.candidate_sha256),
        (args.model_blind_gate.resolve(), args.model_blind_gate_sha256),
        (args.checkpoint.resolve(), CHECKPOINT_SHA256),
        (args.dataset_stats.resolve(), DATASET_STATS_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"hash mismatch: {path}")
    registration = json.loads(args.registration.read_text())
    if registration.get("amendment_id") != "V3-E004":
        raise ValueError("wrong registration")
    if registration.get("queue", {}).get("sha256") != sha256(args.queue):
        raise ValueError("queue is not bound by V3-E004 registration")
    queue_rows = [json.loads(line) for line in args.queue.read_text().splitlines() if line.strip()]
    rows = validate_registered_queue(queue_rows)
    load_candidate(args.candidate, args.candidate_sha256)
    blind = json.loads(args.model_blind_gate.read_text())
    if (
        blind.get("schema_version") != "vla-wam-shared-v3e004-fastwam-robotwin-model-blind-gate-v1"
        or blind.get("passed") is not True
        or blind.get("model_request_count") != 0
        or blind.get("behavioral_episode_count") != 0
        or blind.get("candidate_sha256") != args.candidate_sha256
    ):
        raise ValueError("FastWAM E004 model-blind gate did not pass")
    specific = json.loads(args.model_specific_gate.read_text())
    if (
        specific.get("schema_version") != MODEL_SPECIFIC_SCHEMA
        or specific.get("status") != "passed_exact_repeat_and_left_right_prompt_sensitivity"
        or specific.get("behavioral_episodes") != 0
        or specific.get("model_action_requests") != 3
        or not all(specific.get("requested_release_checks", {}).values())
    ):
        raise ValueError("FastWAM model-specific identity/sensitivity gate did not pass")
    requested_seeds = args.seeds or list(CORE_SEEDS)
    if args.smoke and requested_seeds != [CORE_SEEDS[0]]:
        raise ValueError("FastWAM E004 smoke is exactly seed 9400")
    if len(set(requested_seeds)) != len(requested_seeds) or any(seed not in CORE_SEEDS for seed in requested_seeds):
        raise ValueError("seed outside V3-E004 FastWAM queue")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)

    external_runner_dir = external / "experiments/robotwin_language_gate"
    sys.path[:0] = [str(external_runner_dir), str(external)]
    import closed_loop_language_gate as gate
    import torch

    classes = task_classes(robotwin)
    task_names = {level: f"v3e004_fastwam_s{int(level * 100):03d}" for level in LEVELS}
    gate.RELATION_BY_TASK.update({name: "right" for name in task_names.values()})
    original_task_class = gate.task_class

    def patched_task_class(name: str):
        for level, task_name in task_names.items():
            if name == task_name:
                return classes[level]
        return original_task_class(name)

    gate.task_class = patched_task_class
    original_prompt = gate.make_seen_prompt

    def checked_prompt(env, robotwin_root, relation, prompt_family, protocol):
        value = original_prompt(env, robotwin_root, relation, prompt_family, protocol)
        if prompt_family != "direct_command" or value != PROMPTS[relation]:
            raise ValueError(f"exact FastWAM prompt drift: {value!r}")
        return value

    gate.make_seen_prompt = checked_prompt
    policy = gate.build_policy(
        SimpleNamespace(
            checkpoint=args.checkpoint.resolve(),
            dataset_stats=args.dataset_stats.resolve(),
            action_horizon=32,
            replan_steps=24,
            num_inference_steps=10,
            text_cfg_scale=2.0,
        )
    )
    policy.text_cfg_scale = 2.0
    protocol = gate.load_protocol(study / "artifacts/vla_wam_shared_v2/protocol.json")
    infrastructure = output / "infrastructure_failures.jsonl"
    progress = output / "queue_progress.jsonl"
    completed: list[dict[str, Any]] = []

    blind_scene = {
        float(row["symmetry_level_s"]): row
        for row in blind["tasks"]
        if row["relation"] == "left" and row["repeat_index"] == 0
    }
    for seed in requested_seeds:
        seed_rows = sorted(
            (row for row in rows if int(row["environment_seed"]) == seed),
            key=lambda row: int(row["execution_order_index_within_model_seed"]),
        )
        expected = {(level, relation) for level in LEVELS for relation in RELATIONS}
        if len(seed_rows) != 4 or {(float(row["symmetry_level_s"]), row["relation"]) for row in seed_rows} != expected:
            raise ValueError(f"seed {seed} lacks its exact four-cell block")
        marker_path = output / f"seed_{seed}_complete.json"
        if marker_path.exists():
            if not args.resume:
                raise FileExistsError(marker_path)
            marker = json.loads(marker_path.read_text())
            completed.extend(json.loads(Path(path).read_text()) for path in marker["compact_episode_paths"])
            append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_reused", "seed": seed})
            continue
        append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_started", "seed": seed})
        seed_results: dict[tuple[float, str], dict[str, Any]] = {}
        try:
            for row in seed_rows:
                level = float(row["symmetry_level_s"])
                relation = str(row["relation"])
                task_name = task_names[level]
                condition = f"s{int(level * 100):03d}__{relation}"
                condition_dir = output / task_name / f"environment_seed_{seed}" / f"sampling_seed_{seed}" / condition
                result_path = condition_dir / "result.json"
                if result_path.exists():
                    if not args.resume:
                        raise FileExistsError(result_path)
                    result = json.loads(result_path.read_text())
                else:
                    result = gate.run_episode(
                        policy=policy,
                        robotwin_root=robotwin,
                        task_name=task_name,
                        task_args=gate.load_task_args(robotwin, task_name, "demo_clean"),
                        environment_seed=seed,
                        sampling_seed=seed,
                        requested_relation=relation,
                        condition=condition,
                        prompt_family="direct_command",
                        protocol=protocol,
                        output_dir=output,
                        max_actions=400,
                        save_simulator_video=True,
                        contrastive_negative=False,
                    )
                if result.get("prompt") != PROMPTS[relation] or result.get("requested_relation") != relation:
                    raise ValueError(f"prompt/relation drift: {row['cell_id']}")
                validate_video(Path(result["simulator_video"]))
                snapshot = classes[level].latest_reset_snapshot
                if snapshot is None:
                    raise ValueError("task class did not expose its live reset snapshot")
                result["v3e004"] = {
                    "cell_id": row["cell_id"],
                    "symmetry_level_s": level,
                    "candidate_sha256": args.candidate_sha256,
                    "registration_sha256": sha256(args.registration),
                    "queue_sha256": sha256(args.queue),
                    "model_blind_gate_sha256": args.model_blind_gate_sha256,
                    "model_specific_gate_sha256": sha256(args.model_specific_gate),
                    "source_release_queue_sha256": SOURCE_RELEASE_QUEUE_SHA256,
                    "initial_physical_fingerprint_sha256": fingerprint(snapshot),
                    **snapshot,
                }
                result_path.write_bytes(canonical_json_bytes(result))
                seed_results[(level, relation)] = {"result": result, "result_path": result_path}

            compact_paths: list[str] = []
            for level in LEVELS:
                left = seed_results[(level, "left")]
                right = seed_results[(level, "right")]
                if left["result"]["v3e004"]["initial_physical_fingerprint_sha256"] != right["result"]["v3e004"]["initial_physical_fingerprint_sha256"]:
                    raise ValueError(f"seed {seed} s={level}: LEFT/RIGHT reset mismatch")
                pair_action = action_pair(
                    Path(left["result"]["action_trace"]["path"]),
                    Path(right["result"]["action_trace"]["path"]),
                )
                provisional: dict[str, dict[str, Any]] = {}
                for relation, source in (("left", left), ("right", right)):
                    result = source["result"]
                    trajectory = json.loads(Path(result["trajectory_path"]).read_text())
                    measures = episode_measures(result, trajectory)
                    provisional[relation] = {
                        "schema_version": "vla-wam-shared-v3e004-fastwam-robotwin-episode-v1",
                        "study_id": "vla_wam_language_steerability_v3",
                        "amendment_id": "V3-E004",
                        "model_id": MODEL_ID,
                        "arena": ARENA,
                        "cell_id": result["v3e004"]["cell_id"],
                        "environment_seed": seed,
                        "sampling_seed": seed,
                        "relation": relation,
                        "prompt": PROMPTS[relation],
                        **measures,
                        "symmetry_level_s": level,
                        "asymmetry_metric_A": result["v3e004"]["asymmetry_metric_A"],
                        "position_residual": result["v3e004"]["position_residual_m"],
                        "orientation_residual": result["v3e004"]["orientation_residual_rad"],
                        "midline_residual": result["v3e004"]["midline_residual_m"],
                        "occlusion_check": blind_scene[level]["occlusion_check"],
                        "realised_object_poses": result["v3e004"]["realised_object_poses"],
                        "arm_reset_pose": result["v3e004"]["arm_reset_pose"],
                        "source_result_path": str(source["result_path"].resolve()),
                        "source_result_sha256": sha256(source["result_path"]),
                        "simulator_video": result["simulator_video"],
                        "executed_action_trace": result["action_trace"],
                        "future_evidence": [],
                        "future_interface": "action_only_no_decodable_future",
                    }
                endpoint_shift = provisional["right"]["signed_final_lateral_offset"] - provisional["left"]["signed_final_lateral_offset"]
                for relation in RELATIONS:
                    compact = provisional[relation]
                    compact["endpoint_shift"] = endpoint_shift
                    compact["action_distinct"] = pair_action["action_distinct"]
                    compact["action_pair"] = pair_action
                    compact_path = Path(seed_results[(level, relation)]["result_path"]).with_name("e004_episode.json")
                    compact_path.write_bytes(canonical_json_bytes(compact))
                    compact_paths.append(str(compact_path.resolve()))
                    completed.append(compact)
            marker = {
                "schema_version": "vla-wam-shared-v3e004-fastwam-whole-seed-completion-v1",
                "seed": seed,
                "status": "complete_four_valid_behavioral_cells",
                "behavioral_episode_count": 4,
                "infrastructure_failure_count": 0,
                "compact_episode_paths": compact_paths,
                "completed_at_utc": utc_now(),
            }
            marker_path.write_bytes(canonical_json_bytes(marker))
            append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_complete", "seed": seed})
        except Exception as error:
            append_jsonl(
                infrastructure,
                {
                    "timestamp_utc": utc_now(),
                    "seed": seed,
                    "status": "infrastructure_invalid_partial_seed_excluded_from_behavioral_denominator",
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            append_jsonl(progress, {"timestamp_utc": utc_now(), "event": "seed_failed_closed", "seed": seed})
            raise

    episodes_path = output / "behavioral_episodes.jsonl"
    episodes_path.write_bytes(b"".join(canonical_json_bytes(row) for row in completed))
    manifest = {
        "schema_version": "vla-wam-shared-v3e004-fastwam-runtime-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-E004",
        "model_id": MODEL_ID,
        "arena": ARENA,
        "status": "smoke_complete" if args.smoke else "requested_queue_slice_complete",
        "whole_seeds_complete": len(completed) // 4,
        "behavioral_episode_count": len(completed),
        "requested_seeds": requested_seeds,
        "candidate_sha256": args.candidate_sha256,
        "registration_sha256": sha256(args.registration),
        "queue_sha256": sha256(args.queue),
        "model_blind_gate_sha256": args.model_blind_gate_sha256,
        "model_specific_gate_sha256": sha256(args.model_specific_gate),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset_stats_sha256": DATASET_STATS_SHA256,
        "fastwam_commit": FASTWAM_COMMIT,
        "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "completed_at_utc": utc_now(),
        "denominator_boundary": "RoboTwin only; never pooled with DROID.",
    }
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
