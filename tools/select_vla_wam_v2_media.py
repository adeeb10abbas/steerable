#!/usr/bin/env python3
"""Apply the frozen v2 media rules to the complete v1 DROID population."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROMPT_ORDER = {
    "canonical": 0,
    "short_paraphrase": 1,
    "declarative_goal": 2,
    "contrastive_goal": 3,
}
DIRECTION_ORDER = {"left": 0, "right": 1}
MODEL_ORDER = {"pi05_droid_vla": 0, "cosmos3_edge_droid_wam": 1}
EXPECTED_MODEL_IDS = tuple(MODEL_ORDER)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: str) -> bool:
    if value not in {"True", "False"}:
        raise ValueError(f"Expected serialized bool, got {value!r}")
    return value == "True"


def episode_record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "episode_key": "__".join(
            [
                row["model_id"],
                row["wording"],
                row["direction"],
                f"seed-{row['episode_seed']}",
            ]
        ),
        "model_id": row["model_id"],
        "model_class": row["model_class"],
        "arena": "droid_robolab",
        "condition_id": row["condition_id"],
        "wording": row["wording"],
        "direction": row["direction"],
        "run": int(row["run"]),
        "episode_seed": int(row["episode_seed"]),
        "instruction": row["instruction"],
        "binary_success": truth(row["binary_success"]),
        "outcome_stage": row["outcome_stage"],
        "outcome_stage_label": row["outcome_stage_label"],
        "endpoint_class": row["endpoint_class"],
        "final_cube_minus_bowl_robot_y_m": float(
            row["final_cube_minus_bowl_robot_y_m"]
        ),
        "requested_signed_final_offset_m": float(
            row["requested_signed_final_offset_m"]
        ),
        "hdf5_path": row["hdf5_path"],
        "log_path": row["log_path"],
        "trajectory_figure_path": row["figure_path"],
        "source_video_status": "needs_validated_replay",
    }


def first_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return min(
        rows,
        key=lambda row: (
            PROMPT_ORDER[row["wording"]],
            int(row["episode_seed"]),
            DIRECTION_ORDER[row["direction"]],
        ),
    )


def missing(role: str, model_id: str, reason: str) -> dict[str, Any]:
    return {
        "role": role,
        "model_id": model_id,
        "status": "no_qualifying_example",
        "reason": reason,
        "episodes": [],
    }


def select_for_model(model_id: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for direction in ("left", "right"):
        role = f"first_success_{direction}"
        match = first_row(
            [
                row
                for row in rows
                if row["direction"] == direction and truth(row["binary_success"])
            ]
        )
        if match:
            records.append(
                {
                    "role": role,
                    "model_id": model_id,
                    "status": "selected",
                    "reason": "First valid success in frozen prompt order, then ascending seed.",
                    "episodes": [episode_record(match)],
                }
            )
        else:
            records.append(missing(role, model_id, f"No {direction.upper()} success in v1."))

    post_pick = first_row(
        [row for row in rows if row["outcome_stage"] == "picked_never_entered_goal"]
    )
    if post_pick:
        records.append(
            {
                "role": "first_post_pick_placement_failure",
                "model_id": model_id,
                "status": "selected",
                "reason": "First pickup-without-requested-placement failure in frozen prompt, seed, and direction order.",
                "episodes": [episode_record(post_pick)],
            }
        )
    else:
        records.append(
            missing(
                "first_post_pick_placement_failure",
                model_id,
                "No pickup-without-requested-placement failure in v1.",
            )
        )

    by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for row in rows:
        by_key[(row["wording"], row["direction"], int(row["episode_seed"]))] = row
    pairs: list[tuple[int, int, dict[str, str], dict[str, str]]] = []
    for direction in ("left", "right"):
        direct_seeds = {
            seed
            for wording, row_direction, seed in by_key
            if wording == "canonical" and row_direction == direction
        }
        contrastive_seeds = {
            seed
            for wording, row_direction, seed in by_key
            if wording == "contrastive_goal" and row_direction == direction
        }
        for seed in direct_seeds & contrastive_seeds:
            direct = by_key[("canonical", direction, seed)]
            contrastive = by_key[("contrastive_goal", direction, seed)]
            if truth(direct["binary_success"]) != truth(contrastive["binary_success"]):
                pairs.append((seed, DIRECTION_ORDER[direction], direct, contrastive))
    if pairs:
        _, _, direct, contrastive = min(pairs, key=lambda item: item[:2])
        records.append(
            {
                "role": "first_direct_to_contrastive_reversal",
                "model_id": model_id,
                "status": "selected",
                "reason": "First exact-seed direct/contrastive outcome reversal, then LEFT before RIGHT.",
                "episodes": [episode_record(direct), episode_record(contrastive)],
            }
        )
    else:
        records.append(
            missing(
                "first_direct_to_contrastive_reversal",
                model_id,
                "v1 has no exact-seed direct/contrastive pair: direct used 6100-6109 while contrastive used 7200-7209.",
            )
        )
    return records


def write_csv(path: Path, selections: list[dict[str, Any]]) -> None:
    fields = [
        "role",
        "selection_status",
        "selection_reason",
        "episode_index_within_role",
        "episode_key",
        "model_id",
        "model_class",
        "arena",
        "condition_id",
        "wording",
        "direction",
        "run",
        "episode_seed",
        "instruction",
        "binary_success",
        "outcome_stage",
        "outcome_stage_label",
        "endpoint_class",
        "final_cube_minus_bowl_robot_y_m",
        "requested_signed_final_offset_m",
        "hdf5_path",
        "log_path",
        "trajectory_figure_path",
        "source_video_status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for selection in selections:
            episodes = selection["episodes"] or [{}]
            for index, episode in enumerate(episodes):
                row = {field: "" for field in fields}
                row.update(episode)
                row.update(
                    {
                        "role": selection["role"],
                        "selection_status": selection["status"],
                        "selection_reason": selection["reason"],
                        "episode_index_within_role": index if episode else "",
                    }
                )
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/media"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    source_path = (
        workspace
        / "artifacts/vla_wam_shared_v1/trajectory_evidence/trajectory_index.csv"
    )
    media_plan_path = workspace / "artifacts/vla_wam_shared_v2/media_selection_plan.json"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with source_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 160:
        raise RuntimeError(f"Expected the complete 160-episode v1 population, got {len(rows)}")
    by_model: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_model[row["model_id"]].append(row)
    if set(by_model) != set(EXPECTED_MODEL_IDS):
        raise RuntimeError(f"Unexpected v1 models: {sorted(by_model)}")

    selections: list[dict[str, Any]] = []
    for model_id in sorted(by_model, key=MODEL_ORDER.__getitem__):
        selections.extend(select_for_model(model_id, by_model[model_id]))

    selected_episodes: dict[str, dict[str, Any]] = {}
    roles_by_episode: dict[str, list[str]] = defaultdict(list)
    for selection in selections:
        for episode in selection["episodes"]:
            selected_episodes[episode["episode_key"]] = episode
            roles_by_episode[episode["episode_key"]].append(selection["role"])
    replay_queue = []
    for key in sorted(
        selected_episodes,
        key=lambda item: (
            MODEL_ORDER[selected_episodes[item]["model_id"]],
            PROMPT_ORDER[selected_episodes[item]["wording"]],
            DIRECTION_ORDER[selected_episodes[item]["direction"]],
            selected_episodes[item]["episode_seed"],
        ),
    ):
        episode = dict(selected_episodes[key])
        episode["selection_roles"] = sorted(set(roles_by_episode[key]))
        replay_queue.append(episode)

    payload = {
        "schema_version": "1.0.0",
        "status": "selected_from_complete_v1_population",
        "selection_tier": "retrospective_deterministic",
        "source_path": str(source_path.relative_to(workspace)),
        "source_sha256": sha256(source_path),
        "source_episode_count": len(rows),
        "media_plan_path": str(media_plan_path.relative_to(workspace)),
        "media_plan_sha256": sha256(media_plan_path),
        "prompt_order": list(PROMPT_ORDER),
        "selections": selections,
        "selected_role_count": sum(item["status"] == "selected" for item in selections),
        "missing_role_count": sum(
            item["status"] == "no_qualifying_example" for item in selections
        ),
        "unique_episode_replay_count": len(replay_queue),
        "replay_queue": replay_queue,
    }
    json_path = output_dir / "v1_droid_selection.json"
    csv_path = output_dir / "v1_droid_selection.csv"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, selections)
    print(
        json.dumps(
            {
                "json": str(json_path.relative_to(workspace)),
                "csv": str(csv_path.relative_to(workspace)),
                "selected_roles": payload["selected_role_count"],
                "missing_roles": payload["missing_role_count"],
                "unique_episode_replays": payload["unique_episode_replay_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
