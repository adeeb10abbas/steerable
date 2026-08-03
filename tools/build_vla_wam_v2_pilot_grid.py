#!/usr/bin/env python3
"""Compile the frozen v2 protocol into an auditable 144-cell execution grid."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from vla_wam_v2_protocol import PROMPT_IDS, load_protocol, render_prompt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_rows(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    return [model for model in protocol["models"] if model["standardized_v2_expansion_required"]]


def grid(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    arenas = {arena["id"]: arena for arena in protocol["design"]["arenas"]}
    prompts = {family["id"]: family for family in protocol["prompt_families"]}
    rows: list[dict[str, Any]] = []
    for model in model_rows(protocol):
        arena_id = model["arena"]
        arena = arenas[arena_id]
        if arena_id == "droid_robolab":
            seeds = arena["episode_seeds"]["new_v2_paired"][: arena["pilot_seed_count"]]
            pairs = [
                {
                    "pair_id": f"droid_pair_seed_{seed}",
                    "environment_seed": seed,
                    "sampling_seed": seed,
                }
                for seed in seeds
            ]
        elif arena_id == "robotwin_place_a2b":
            pairs = arena["paired_scenes"]
        else:
            raise ValueError(arena_id)

        for pair in pairs:
            for family_id in PROMPT_IDS:
                family = prompts[family_id]
                for direction in ("left", "right"):
                    if arena_id == "droid_robolab":
                        task_name = arena["tasks"][direction]
                        rendered_prompt = render_prompt(
                            protocol,
                            family_id=family_id,
                            direction=direction,
                            movable="Rubik's cube",
                            reference="bowl",
                            movable_short="cube",
                            arena=arena_id,
                        )
                        rendering_status = "fully_rendered"
                    else:
                        task_name = pair["anchor_task"]
                        rendered_prompt = None
                        rendering_status = "requires_frozen_scene_object_metadata"
                    cell_id = "__".join(
                        [
                            model["id"],
                            pair["pair_id"],
                            family_id,
                            direction,
                        ]
                    )
                    rows.append(
                        {
                            "cell_id": cell_id,
                            "model_id": model["id"],
                            "model_class": model["class"],
                            "world_model_interface": model["world_model_interface"],
                            "arena": arena_id,
                            "pair_id": pair["pair_id"],
                            "anchor_task": task_name,
                            "environment_seed": pair["environment_seed"],
                            "sampling_seed": pair["sampling_seed"],
                            "prompt_family": family_id,
                            "legacy_v1_wording": family["legacy_v1_id"],
                            "requested_relation": direction,
                            "native_task_relation": (
                                "left" if task_name.endswith("left") else "right"
                            ),
                            "condition_role": (
                                "native_direction"
                                if task_name.endswith(direction)
                                else "counterfactual_direction"
                            ),
                            "prompt_template": family[direction],
                            "rendered_prompt": rendered_prompt,
                            "prompt_rendering_status": rendering_status,
                            "execution_batch": (
                                "direct_base_competence_gate"
                                if family_id == "direct_command"
                                else "wording_followup_if_base_gate_passes"
                            ),
                            "video_required": True,
                            "valid_failure_retained": True,
                        }
                    )
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 144:
        raise RuntimeError(f"Expected 144 pilot cells, got {len(rows)}")
    cell_ids = {row["cell_id"] for row in rows}
    if len(cell_ids) != len(rows):
        raise RuntimeError("Pilot cell IDs are not unique")
    by_model = Counter(row["model_id"] for row in rows)
    if set(by_model.values()) != {24}:
        raise RuntimeError(f"Each model must have 24 cells: {by_model}")
    direct = [row for row in rows if row["prompt_family"] == "direct_command"]
    followup = [row for row in rows if row["prompt_family"] != "direct_command"]
    if len(direct) != 36 or len(followup) != 108:
        raise RuntimeError("Expected 36 direct-gate and 108 conditional wording cells")

    by_pair: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pair[(row["model_id"], row["pair_id"], row["prompt_family"], row["arena"])].append(row)
    for key, pair in by_pair.items():
        if len(pair) != 2 or {row["requested_relation"] for row in pair} != {"left", "right"}:
            raise RuntimeError(f"Incomplete LEFT/RIGHT pair: {key}")
        invariant_fields = ("environment_seed", "sampling_seed")
        if key[3] == "robotwin_place_a2b":
            invariant_fields = ("anchor_task", *invariant_fields)
        for field in invariant_fields:
            if len({row[field] for row in pair}) != 1:
                raise RuntimeError(f"Pair changes {field}: {key}")
    return {
        "episode_count": len(rows),
        "direct_base_competence_gate_count": len(direct),
        "conditional_wording_followup_count": len(followup),
        "model_cell_counts": dict(sorted(by_model.items())),
        "exact_left_right_pair_count": len(by_pair),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot"),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    protocol_path = workspace / "artifacts/vla_wam_shared_v2/protocol.json"
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = workspace / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol(protocol_path)
    rows = grid(protocol)
    summary = validate(rows)
    csv_path = output_dir / "pilot_grid.csv"
    json_path = output_dir / "pilot_grid.json"
    write_csv(csv_path, rows)
    payload = {
        "schema_version": "1.0.0",
        "status": "compiled_before_standardized_v2_inference",
        "protocol_path": str(protocol_path.relative_to(workspace)),
        "protocol_sha256": sha256(protocol_path),
        "summary": summary,
        "execution_order": [
            "Run each model's six direct-command cells first.",
            "Apply the frozen base-competence gate per model.",
            "Run the remaining 18 wording cells only when the gate permits.",
        ],
        "cells": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"csv": str(csv_path), "json": str(json_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
