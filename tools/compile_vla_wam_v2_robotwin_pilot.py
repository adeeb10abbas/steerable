#!/usr/bin/env python3
"""Compile standardized RoboTwin v2 pilot evidence without an oracle policy.

The compiler treats simulator state as measurement, not as a source of actions or
prompts.  It preserves the raw result, trajectory, imagined future, and executed
video paths while adding stage-of-failure proxies that can be audited later.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_ID = "efficient_wam_rt_robotwin"
MODEL_LABELS = {
    "efficient_wam_rt_robotwin": "Efficient-WAM-RT",
    "fastwam_robotwin": "FastWAM",
    "lingbot_va_robotwin": "LingBot-VA",
}
PAIR_BY_ENVIRONMENT_SEED = {4300000: "pair00", 4300001: "pair01", 4300002: "pair02"}
PICKUP_LIFT_M = 0.03
PICKUP_CONSECUTIVE_STEPS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def predicted_video(result: dict[str, Any]) -> Path | None:
    directory_value = result.get("predicted_video_dir")
    if not directory_value:
        return None
    directory = Path(directory_value)
    videos = sorted(directory.glob("*.mp4"))
    if len(videos) != 1:
        return None
    return videos[0]


def predicted_artifact(
    result_path: Path, result: dict[str, Any], future_video: Path | None
) -> dict[str, Any] | None:
    if future_video is not None:
        record = file_record(future_video)
        return {"kind": "decoded_video", **record} if record else None
    latent_value = result.get("first_predicted_latent_path")
    latent_path = Path(latent_value) if latent_value else result_path.parent / "first_predicted_latent.pt"
    record = file_record(latent_path)
    return {"kind": "latent_tensor", **record} if record else None


def max_consecutive(values: list[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def classify_episode(
    result_path: Path,
    model_id: str,
    interventions: dict[tuple[str, int, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    result = json.loads(result_path.read_text())
    trajectory_path = Path(result["trajectory_path"])
    trajectory = json.loads(trajectory_path.read_text())
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"Invalid trajectory: {trajectory_path}")

    initial_z = float(trajectory[0]["object_xyz"][2])
    lift = [float(step["object_xyz"][2]) - initial_z for step in trajectory]
    closed = [not bool(step["grippers_open"]) for step in trajectory]
    pickup_mask = [
        step_lift >= PICKUP_LIFT_M and is_closed
        for step_lift, is_closed in zip(lift, closed, strict=True)
    ]
    verified_pickup = max_consecutive(pickup_mask) >= PICKUP_CONSECUTIVE_STEPS
    relation_indices = [
        int(step["action_step"]) for step in trajectory if step["relation_region"]
    ]
    released_relation_indices = [
        int(step["action_step"])
        for step in trajectory
        if step["relation_region"] and step["grippers_open"]
    ]

    requested_success = bool(result["requested_success"])
    if requested_success:
        failure_stage = "success"
    elif relation_indices:
        failure_stage = "entered_requested_region_without_verified_completion"
    elif verified_pickup:
        failure_stage = "picked_never_entered_requested_region"
    elif any(closed):
        failure_stage = "closed_gripper_no_verified_pickup"
    else:
        failure_stage = "no_verified_interaction"

    final_dx = float(result["final"]["object_minus_target_x"])
    final_dy = float(result["final"]["object_minus_target_y"])
    direction_sign = -1.0 if result["requested_relation"] == "left" else 1.0
    sim_video = Path(result["simulator_video"])
    future_video = predicted_video(result)
    future_artifact = predicted_artifact(result_path, result, future_video)
    cell_interventions = interventions.get(
        (model_id, int(result["environment_seed"]), result["requested_relation"]), []
    )

    pair_id = PAIR_BY_ENVIRONMENT_SEED.get(int(result["environment_seed"]))
    if pair_id is None:
        raise ValueError(f"Unregistered pilot environment seed: {result['environment_seed']}")
    prompt_body = result["prompt"].removeprefix("Put the ").removesuffix(".")
    prompt_separator = f" to the {result['requested_relation']} of the "
    if prompt_separator not in prompt_body:
        raise ValueError(f"Cannot recover direct-command object descriptions: {result['prompt']}")
    movable_description, reference_description = prompt_body.split(prompt_separator, 1)
    return {
        "model_id": model_id,
        "pair_id": pair_id,
        "task": result["task"],
        "environment_seed": int(result["environment_seed"]),
        "sampling_seed": int(result["sampling_seed"]),
        "prompt_family": result["prompt_family"],
        "requested_relation": result["requested_relation"],
        "prompt": result["prompt"],
        "movable_description": movable_description,
        "reference_description": reference_description,
        "movable_model_name": result["object_name"],
        "reference_model_name": result["target_name"],
        "requested_success": requested_success,
        "native_task_success": bool(result["original_task_success"]),
        "opposite_relation_success": bool(result["opposite_relation_success"]),
        "prompt_ignored_native_task_completed": bool(
            not requested_success
            and result["original_task_success"]
            and result["task_relation"] != result["requested_relation"]
        ),
        "actions_executed": int(result["actions_executed"]),
        "wall_seconds": float(result["wall_seconds"]),
        "operational_wall_latency_valid": all(
            event["wall_latency_valid"] for event in cell_interventions
        ),
        "runtime_intervention_ids": [event["id"] for event in cell_interventions],
        "initial_dx_m": float(result["initial"]["object_minus_target_x"]),
        "initial_dy_m": float(result["initial"]["object_minus_target_y"]),
        "final_dx_m": final_dx,
        "final_dy_m": final_dy,
        "command_alignment_margin_m": direction_sign * final_dx,
        "max_object_lift_m": max(lift),
        "ever_gripper_closed": any(closed),
        "verified_pickup_proxy": verified_pickup,
        "ever_entered_requested_region": bool(relation_indices),
        "first_requested_region_action": relation_indices[0] if relation_indices else None,
        "ever_released_in_requested_region": bool(released_relation_indices),
        "first_release_in_requested_region_action": (
            released_relation_indices[0] if released_relation_indices else None
        ),
        "failure_stage": failure_stage,
        "raw_result": file_record(result_path),
        "raw_trajectory": file_record(trajectory_path),
        "executed_video": file_record(sim_video),
        "imagined_future_video": file_record(future_video),
        "imagined_future_artifact": future_artifact,
    }


def flatten(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
    flat["runtime_intervention_ids"] = ";".join(row["runtime_intervention_ids"])
    for key in (
        "raw_result",
        "raw_trajectory",
        "executed_video",
        "imagined_future_video",
        "imagined_future_artifact",
    ):
        record = row.get(key)
        flat[f"{key}_path"] = record["path"] if record else ""
        flat[f"{key}_sha256"] = record["sha256"] if record else ""
        if key == "imagined_future_artifact":
            flat[f"{key}_kind"] = record["kind"] if record else ""
    return flat


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_direction: dict[str, dict[str, Any]] = {}
    for direction in ("left", "right"):
        rows = [row for row in episodes if row["requested_relation"] == direction]
        by_direction[direction] = {
            "episodes": len(rows),
            "successes": sum(row["requested_success"] for row in rows),
            "verified_pickups": sum(row["verified_pickup_proxy"] for row in rows),
            "entered_requested_region": sum(
                row["ever_entered_requested_region"] for row in rows
            ),
            "prompt_ignored_native_task_completed": sum(
                row["prompt_ignored_native_task_completed"] for row in rows
            ),
            "mean_command_alignment_margin_m": sum(
                row["command_alignment_margin_m"] for row in rows
            )
            / len(rows),
        }

    pairs = []
    for pair_id in sorted({row["pair_id"] for row in episodes}):
        rows = {row["requested_relation"]: row for row in episodes if row["pair_id"] == pair_id}
        if set(rows) != {"left", "right"}:
            raise ValueError(f"Incomplete directional pair: {pair_id}")
        endpoint_shift = rows["right"]["final_dx_m"] - rows["left"]["final_dx_m"]
        pairs.append(
            {
                "pair_id": pair_id,
                "left_success": rows["left"]["requested_success"],
                "right_success": rows["right"]["requested_success"],
                "left_final_dx_m": rows["left"]["final_dx_m"],
                "right_final_dx_m": rows["right"]["final_dx_m"],
                "right_minus_left_endpoint_dx_m": endpoint_shift,
                "endpoint_response_direction": (
                    "aligned" if endpoint_shift > 0 else "anti_directed" if endpoint_shift < 0 else "none"
                ),
            }
        )

    left_successes = by_direction["left"]["successes"]
    right_successes = by_direction["right"]["successes"]
    if left_successes and right_successes:
        gate_decision = "expand_frozen_four_prompt_grid"
        gate_reason = (
            "Direct-command success occurred in both directions, so the frozen gate "
            "permits the ten-seed four-prompt expansion."
        )
    elif left_successes or right_successes:
        gate_decision = "expand_direct_directional_bias_only"
        observed_direction = "LEFT" if left_successes else "RIGHT"
        gate_reason = (
            f"Direct-command success occurred for {observed_direction} only, so the frozen gate "
            "calls for a ten-scene direct-command directional-bias confirmation before any "
            "four-wording sweep."
        )
    else:
        gate_decision = "stop_base_competence_failure"
        gate_reason = (
            "No direct-command episode succeeded in either direction. The frozen gate retains "
            "all six attempts as a base-competence failure and stops this model's wording expansion."
        )

    return {
        "episode_count": len(episodes),
        "pair_count": len(pairs),
        "successes": sum(row["requested_success"] for row in episodes),
        "success_rate": sum(row["requested_success"] for row in episodes) / len(episodes),
        "by_direction": by_direction,
        "failure_stage_counts": dict(Counter(row["failure_stage"] for row in episodes)),
        "prompt_ignored_native_task_completed_count": sum(
            row["prompt_ignored_native_task_completed"] for row in episodes
        ),
        "operational_wall_latency_valid_episodes": sum(
            row["operational_wall_latency_valid"] for row in episodes
        ),
        "operational_wall_latency_excluded_episodes": sum(
            not row["operational_wall_latency_valid"] for row in episodes
        ),
        "paired_endpoint_responses": pairs,
        "pilot_gate_decision": gate_decision,
        "pilot_gate_reason": gate_reason,
    }


def write_report(path: Path, compiled: dict[str, Any]) -> None:
    summary = compiled["summary"]
    left = summary["by_direction"]["left"]
    right = summary["by_direction"]["right"]
    lines = [
        f"# {compiled['model_label']} standardized direct-command pilot",
        "",
        f"Compiled at `{compiled['compiled_at_utc']}` from {summary['episode_count']} executed episodes "
        f"in {summary['pair_count']} exact left/right scene pairs.",
        "",
        "## Result",
        "",
        f"- LEFT: **{left['successes']}/{left['episodes']}** requested-relation successes.",
        f"- RIGHT: **{right['successes']}/{right['episodes']}** requested-relation successes.",
        f"- Overall: **{summary['successes']}/{summary['episode_count']}**.",
        f"- Prompt-ignored/native-task-completed failures: **{summary['prompt_ignored_native_task_completed_count']}**.",
        f"- Wall-latency-valid episodes: **{summary['operational_wall_latency_valid_episodes']}/{summary['episode_count']}**.",
        "",
        compiled["summary"]["pilot_gate_reason"],
        "",
        "This six-episode pilot is a gate, not a stable rate estimate.",
        "",
        "## Exact command pairs and endpoints",
        "",
        "| Pair | LEFT | RIGHT | Final x: LEFT | Final x: RIGHT | RIGHT - LEFT | Response |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for pair in summary["paired_endpoint_responses"]:
        lines.append(
            f"| {pair['pair_id']} | {'success' if pair['left_success'] else 'failure'} | "
            f"{'success' if pair['right_success'] else 'failure'} | "
            f"{pair['left_final_dx_m']:+.3f} m | {pair['right_final_dx_m']:+.3f} m | "
            f"{pair['right_minus_left_endpoint_dx_m']:+.3f} m | "
            f"{pair['endpoint_response_direction'].replace('_', ' ')} |"
        )
    lines.extend(
        [
            "",
            "## Measurement boundary",
            "",
            "The model receives only the frozen language prompt and observation. Simulator state is "
            "used after each action solely for scoring and visualization; no oracle actions, subtask "
            "coach, dynamic prompt, or online correction is used.",
            "",
            f"`verified_pickup_proxy` means at least {PICKUP_CONSECUTIVE_STEPS} consecutive recorded "
            f"states with the movable object lifted at least {PICKUP_LIFT_M:.2f} m above its initial "
            "height while the gripper is reported closed. It is a transparent diagnostic proxy, not "
            "a learned semantic judgment.",
            "",
            (
                "Pixel differences between imagined futures are deliberately not used as semantic "
                "steerability evidence. The raw imagined artifacts are hash-locked here for later "
                "predicate scoring and imagination-versus-execution analysis."
                if any(episode["imagined_future_artifact"] for episode in compiled["episodes"])
                else "This inference interface emits actions but no test-time imagined-video artifact, "
                "so imagination-versus-execution metrics are recorded as not applicable rather than zero."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/home/ali/projects/Efficient-WAM/outputs/vla_wam_shared_v2/direct_gate"),
    )
    parser.add_argument("--model-id", choices=sorted(MODEL_LABELS), default=DEFAULT_MODEL_ID)
    parser.add_argument("--output-stem")
    parser.add_argument(
        "--runtime-interventions",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/runtime_interventions.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/results"),
    )
    args = parser.parse_args()

    paths = sorted(args.input_root.glob("pair*/**/result.json"))
    if len(paths) != 6:
        raise RuntimeError(f"Expected exactly six pilot results, found {len(paths)}")
    intervention_path = args.runtime_interventions
    if not intervention_path.is_absolute():
        intervention_path = Path(__file__).resolve().parents[1] / intervention_path
    intervention_map: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    if intervention_path.exists():
        intervention_payload = json.loads(intervention_path.read_text())
        for event in intervention_payload["events"]:
            key = (
                event["model_id"],
                int(event["environment_seed"]),
                event["requested_relation"],
            )
            intervention_map.setdefault(key, []).append(event)
    episodes = [
        classify_episode(path, args.model_id, intervention_map) for path in paths
    ]
    output_stem = args.output_stem or (
        "efficient_wam_rt_direct_gate"
        if args.model_id == DEFAULT_MODEL_ID
        else f"{args.model_id.removesuffix('_robotwin')}_direct_gate"
    )
    compiled = {
        "schema_version": "vla-wam-shared-v2-robotwin-pilot-v1",
        "compiled_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": args.model_id,
        "model_label": MODEL_LABELS[args.model_id],
        "measurement": {
            "oracle_actions": 0,
            "dynamic_prompts": 0,
            "pickup_lift_threshold_m": PICKUP_LIFT_M,
            "pickup_consecutive_steps": PICKUP_CONSECUTIVE_STEPS,
            "simulator_state_role": "post_action_scoring_and_visualization_only",
            "runtime_interventions_path": (
                str(intervention_path) if intervention_path.exists() else None
            ),
        },
        "summary": summarize(episodes),
        "episodes": episodes,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{output_stem}.json"
    csv_path = args.output_dir / f"{output_stem}.csv"
    report_path = args.output_dir / f"{output_stem}.md"
    json_path.write_text(json.dumps(compiled, indent=2) + "\n")
    flat = [flatten(row) for row in episodes]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(flat)
    write_report(report_path, compiled)
    print(json.dumps(compiled["summary"], indent=2))
    print(json_path)
    print(csv_path)
    print(report_path)


if __name__ == "__main__":
    main()
