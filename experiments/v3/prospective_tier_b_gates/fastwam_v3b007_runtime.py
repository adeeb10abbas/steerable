#!/usr/bin/env python3
"""Run the released V3-B007 FastWAM mirror queue in whole-seed blocks.

The external FastWAM runner remains the source of the controller and scoring
loop.  This bridge supplies only the two model-blind, hash-frozen pair03
fixtures and enforces the released four-cell order for every matched seed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

import numpy as np


FASTWAM_COMMIT = "068d3fd70c89df3726c09893f47b75a624b20c02"
CHECKPOINT_SHA256 = "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63"
STATS_SHA256 = "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095"
QUEUE_SHA256 = "2ffe2f99e4d6c4b3d80c24fab7276b21bb83de86d92b8a3438ce38a7ba9e1ae3"
MODEL_BLIND_GATE_SHA256 = "e092917893591490f1b1ee2ab2f9c6bd4cd9cc560fa5702d49dd6974a301d6ad"
MODEL_SPECIFIC_SCHEMA = "vla-wam-shared-v3-robotwin-fixed-observation-release-v1"
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
FIXTURES = {
    "control": {
        "object": {
            "position_xyz_m": [-0.047076620161533356, -0.030880313366651535, 0.7405446767807007],
            "quaternion_wxyz": [0.3491150736808777, 0.3625031113624573, 0.609960675239563, 0.6120932698249817],
        },
        "reference": {
            "position_xyz_m": [-0.21130692958831787, -0.1640346497297287, 0.7408550977706909],
            "quaternion_wxyz": [-0.34580183029174805, -0.3450961410999298, 0.6168054938316345, 0.6171554923057556],
        },
    },
    "position_mirrored": {
        "object": {
            "position_xyz_m": [0.047076620161533356, -0.030880313366651535, 0.7405446767807007],
            "quaternion_wxyz": [0.3491150736808777, 0.3625031113624573, 0.609960675239563, 0.6120932698249817],
        },
        "reference": {
            "position_xyz_m": [0.21130692958831787, -0.1640346497297287, 0.7408550977706909],
            "quaternion_wxyz": [-0.34580183029174805, -0.3450961410999298, 0.6168054938316345, 0.6171554923057556],
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--external-repository", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--model-blind-gate", type=Path, required=True)
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
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def git(repository: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(repository), *arguments], text=True).strip()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical(row))


def task_classes(robotwin_root: Path):
    os.chdir(robotwin_root)
    sys.path[:0] = [str(robotwin_root), str(robotwin_root / "script"), str(robotwin_root / "description/utils")]
    import sapien
    from envs.place_a2b_right import place_a2b_right
    from envs.utils import create_actor

    def build(arm: str):
        layout = FIXTURES[arm]

        class FixedPair03(place_a2b_right):
            def load_actors(self):
                self.selected_modelname_A, self.selected_model_id_A = "086_woodenblock", 1
                self.selected_modelname_B, self.selected_model_id_B = "081_playingcards", 1
                object_row = layout["object"]
                reference_row = layout["reference"]
                self.object = create_actor(
                    scene=self,
                    pose=sapien.Pose(
                        p=object_row["position_xyz_m"], q=object_row["quaternion_wxyz"]
                    ),
                    modelname=self.selected_modelname_A,
                    convex=True,
                    model_id=self.selected_model_id_A,
                )
                self.target_object = create_actor(
                    scene=self,
                    pose=sapien.Pose(
                        p=reference_row["position_xyz_m"], q=reference_row["quaternion_wxyz"]
                    ),
                    modelname=self.selected_modelname_B,
                    convex=True,
                    model_id=self.selected_model_id_B,
                )
                self.object.set_mass(0.05)
                self.target_object.set_mass(0.05)
                self.add_prohibit_area(self.object, padding=0.05)
                self.add_prohibit_area(self.target_object, padding=0.1)

        FixedPair03.__name__ = f"v3b007_{arm}"
        return FixedPair03

    return {arm: build(arm) for arm in FIXTURES}


def fingerprint(initial: dict) -> str:
    physical = {
        "object_xyz": initial["object_xyz"],
        "target_xyz": initial["target_xyz"],
        "distance_xy": initial["distance_xy"],
        "grippers_open": initial["grippers_open"],
    }
    return hashlib.sha256(canonical(physical)).hexdigest()


def validate_video(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing simulator video: {path}")
    value = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames", "-of", "default=nw=1:nk=1", str(path)],
        text=True,
    ).strip()
    if not value or value == "N/A" or int(value) < 1:
        raise ValueError(f"simulator video does not decode: {path}")


def main() -> None:
    args = parse_args()
    study = args.study_root.resolve()
    external = args.external_repository.resolve()
    robotwin = external / "third_party/RoboTwin"
    queue_path = args.queue.resolve()
    blind_gate_path = args.model_blind_gate.resolve()
    specific_gate_path = args.model_specific_gate.resolve()
    checkpoint = args.checkpoint.resolve()
    stats = args.dataset_stats.resolve()
    output = args.output_dir.resolve()

    if git(study, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("study checkout has tracked changes")
    if git(external, "rev-parse", "HEAD") != FASTWAM_COMMIT:
        raise ValueError("FastWAM revision drift")
    if git(external, "status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("FastWAM checkout has tracked changes")
    for path, expected in (
        (queue_path, QUEUE_SHA256),
        (blind_gate_path, MODEL_BLIND_GATE_SHA256),
        (checkpoint, CHECKPOINT_SHA256),
        (stats, STATS_SHA256),
    ):
        if sha256(path) != expected:
            raise ValueError(f"hash mismatch: {path}")
    blind_gate = json.loads(blind_gate_path.read_text())
    if not blind_gate.get("passed") or blind_gate.get("model_request_count") or blind_gate.get("behavioral_episode_count"):
        raise ValueError("model-blind gate is not a zero-request pass")
    specific_gate = json.loads(specific_gate_path.read_text())
    if (
        specific_gate.get("schema_version") != MODEL_SPECIFIC_SCHEMA
        or specific_gate.get("status") != "passed_exact_repeat_and_left_right_prompt_sensitivity"
        or specific_gate.get("behavioral_episodes") != 0
        or specific_gate.get("model_action_requests") != 3
        or not all(specific_gate.get("requested_release_checks", {}).values())
    ):
        raise ValueError("model-specific gate did not pass")

    rows = [json.loads(line) for line in queue_path.read_text().splitlines() if line.strip()]
    requested_seeds = args.seeds or list(range(9900, 9927))
    if args.smoke and requested_seeds != [9900]:
        raise ValueError("the released smoke block is exactly seed 9900")
    if any(seed not in range(9900, 9927) for seed in requested_seeds):
        raise ValueError("seed outside released V3-B007 queue")
    if output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite without --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)

    external_runner_dir = external / "experiments/robotwin_language_gate"
    sys.path[:0] = [str(external_runner_dir), str(external)]
    import closed_loop_language_gate as gate
    import torch

    classes = task_classes(robotwin)
    task_names = {arm: f"v3b007_{arm}" for arm in FIXTURES}
    gate.RELATION_BY_TASK.update({name: "right" for name in task_names.values()})
    original_task_class = gate.task_class

    def patched_task_class(name: str):
        for arm, task_name in task_names.items():
            if name == task_name:
                return classes[arm]
        return original_task_class(name)

    gate.task_class = patched_task_class
    original_prompt = gate.make_seen_prompt

    def checked_prompt(env, robotwin_root, relation, prompt_family, protocol):
        value = original_prompt(env, robotwin_root, relation, prompt_family, protocol)
        if prompt_family != "direct_command" or value != PROMPTS[relation]:
            raise ValueError(f"exact prompt drift: {prompt_family=} {relation=} {value!r}")
        return value

    gate.make_seen_prompt = checked_prompt
    model_args = SimpleNamespace(
        checkpoint=checkpoint,
        dataset_stats=stats,
        action_horizon=32,
        replan_steps=24,
        num_inference_steps=10,
        text_cfg_scale=2.0,
    )
    torch.cuda.reset_peak_memory_stats()
    policy = gate.build_policy(model_args)
    policy.text_cfg_scale = 2.0
    protocol = gate.load_protocol(study / "artifacts/vla_wam_shared_v2/protocol.json")
    infrastructure_stream = output / "infrastructure_failures.jsonl"
    progress_stream = output / "queue_progress.jsonl"
    completed_results: list[dict] = []

    for seed in requested_seeds:
        seed_rows = sorted(
            (row for row in rows if row["matched_seed"] == seed),
            key=lambda row: row["execution_order_index_within_seed"],
        )
        expected_cells = {(arm, relation) for arm in FIXTURES for relation in PROMPTS}
        if len(seed_rows) != 4 or {(row["arm"], row["relation"]) for row in seed_rows} != expected_cells:
            raise ValueError(f"seed {seed} lacks its exact four-cell block")
        seed_marker = output / f"seed_{seed}_complete.json"
        if seed_marker.exists():
            if not args.resume:
                raise FileExistsError(seed_marker)
            marker = json.loads(seed_marker.read_text())
            for result_path in marker["result_paths"]:
                completed_results.append(json.loads(Path(result_path).read_text()))
            append_jsonl(progress_stream, {"timestamp_utc": utc_now(), "event": "seed_reused", "seed": seed})
            continue

        seed_results = []
        append_jsonl(progress_stream, {"timestamp_utc": utc_now(), "event": "seed_started", "seed": seed})
        try:
            for row in seed_rows:
                arm = row["arm"]
                relation = row["relation"]
                if row["prompt"] != PROMPTS[relation] or row["fixture"] != FIXTURES[arm]:
                    raise ValueError(f"released row drift for {row['cell_id']}")
                task_name = task_names[arm]
                condition = f"{arm}__{relation}"
                result_path = output / task_name / f"environment_seed_{seed}" / f"sampling_seed_{seed}" / condition / "result.json"
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
                if result["prompt"] != PROMPTS[relation] or result["requested_relation"] != relation:
                    raise ValueError(f"result prompt/relation drift for {row['cell_id']}")
                validate_video(Path(result["simulator_video"]))
                result["v3b007"] = {
                    "cell_id": row["cell_id"],
                    "arm": arm,
                    "queue_sha256": QUEUE_SHA256,
                    "model_blind_gate_sha256": MODEL_BLIND_GATE_SHA256,
                    "model_specific_gate_sha256": sha256(specific_gate_path),
                    "initial_physical_fingerprint_sha256": fingerprint(result["initial"]),
                }
                result_path.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
                seed_results.append(result)

            for arm in FIXTURES:
                hashes = {
                    result["v3b007"]["initial_physical_fingerprint_sha256"]
                    for result in seed_results
                    if result["v3b007"]["arm"] == arm
                }
                if len(hashes) != 1:
                    raise ValueError(f"seed {seed} {arm} LEFT/RIGHT reset mismatch")
            marker = {
                "schema_version": "vla-wam-shared-v3b007-whole-seed-completion-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-B007",
                "arena": "robotwin",
                "seed": seed,
                "status": "complete_four_valid_behavioral_cells",
                "behavioral_episode_count": 4,
                "infrastructure_failure_count": 0,
                "result_paths": [result["trajectory_path"].replace("trajectory.json", "result.json") for result in seed_results],
                "requested_success": {
                    f"{result['v3b007']['arm']}:{result['requested_relation']}": result["requested_success"]
                    for result in seed_results
                },
                "completed_at_utc": utc_now(),
            }
            seed_marker.write_bytes(json.dumps(marker, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")
            completed_results.extend(seed_results)
            append_jsonl(progress_stream, {"timestamp_utc": utc_now(), "event": "seed_complete", "seed": seed})
        except Exception as error:
            append_jsonl(
                infrastructure_stream,
                {
                    "timestamp_utc": utc_now(),
                    "seed": seed,
                    "status": "infrastructure_invalid_partial_seed_excluded_from_behavioral_denominator",
                    "exception_type": type(error).__name__,
                    "exception": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            append_jsonl(progress_stream, {"timestamp_utc": utc_now(), "event": "seed_failed_closed", "seed": seed})
            raise

    with (output / "behavioral_episodes.jsonl").open("wb") as handle:
        for result in completed_results:
            handle.write(canonical(result))
    manifest = {
        "schema_version": "vla-wam-shared-v3b007-fastwam-runtime-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "arena": "robotwin",
        "status": "smoke_complete" if args.smoke else "requested_queue_slice_complete",
        "whole_seeds_complete": len(completed_results) // 4,
        "behavioral_episode_count": len(completed_results),
        "requested_seeds": requested_seeds,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "dataset_stats_sha256": STATS_SHA256,
        "queue_sha256": QUEUE_SHA256,
        "model_blind_gate_sha256": MODEL_BLIND_GATE_SHA256,
        "model_specific_gate_sha256": sha256(specific_gate_path),
        "fastwam_commit": FASTWAM_COMMIT,
        "gpu_peak_memory_mib": float(torch.cuda.max_memory_allocated() / (1024 * 1024)),
        "completed_at_utc": utc_now(),
        "denominator_boundary": "RoboTwin only; never pooled with DROID.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
