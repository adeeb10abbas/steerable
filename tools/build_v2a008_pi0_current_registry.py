#!/usr/bin/env python3
"""Build the frozen 60-cell V2-A008 current-stack pi0-FAST registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from vla_wam_v2_protocol import load_protocol, render_prompt


AMENDMENT_ID = "V2-A008"
FAMILIES = (
    "short_command",
    "goal_as_outcome",
    "desired_plus_negated_opposite",
)
DIRECTIONS = ("left", "right")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(workspace: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(workspace)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def build_cells(
    protocol: dict[str, Any], amendment: dict[str, Any]
) -> list[dict[str, Any]]:
    grid = amendment["behavioral_grid"]
    if amendment["amendment_id"] != AMENDMENT_ID:
        raise ValueError("Registry builder requires the V2-A008 amendment")
    if tuple(grid["prompt_families"]) != FAMILIES:
        raise ValueError("V2-A008 prompt-family order changed")
    if tuple(grid["requested_relations"]) != DIRECTIONS:
        raise ValueError("V2-A008 requested-relation order changed")
    environment_seeds = tuple(int(seed) for seed in grid["environment_seeds"])
    sampling_seeds = tuple(int(seed) for seed in grid["sampling_seeds"])
    if environment_seeds != tuple(range(8300, 8310)):
        raise ValueError("V2-A008 environment seeds must be 8300 through 8309")
    if sampling_seeds != environment_seeds:
        raise ValueError("V2-A008 environment and sampling seed roots must match")

    prompt_by_id = {row["id"]: row for row in protocol["prompt_families"]}
    cells: list[dict[str, Any]] = []
    for environment_seed, sampling_seed in zip(environment_seeds, sampling_seeds):
        for family in FAMILIES:
            for relation in DIRECTIONS:
                prompt = render_prompt(
                    protocol,
                    family_id=family,
                    direction=relation,
                    movable="Rubik's cube",
                    movable_short="cube",
                    reference="bowl",
                    arena="droid_robolab",
                )
                pair_id = f"droid_pair_seed_{environment_seed}"
                output_folder = (
                    f"v2a008_pi0_current_seed{environment_seed}_{family}_{relation}"
                )
                cells.append(
                    {
                        "cell_id": "__".join(
                            ["pi0_fast_current_stack_droid_vla", pair_id, family, relation]
                        ),
                        "experiment_id": amendment["replication_identity"]["experiment_id"],
                        "model_id": "pi0_fast_current_stack_droid_vla",
                        "arena": "droid_robolab",
                        "pair_id": pair_id,
                        "anchor_task": (
                            "RubiksCubeLeftOfBowlMatchedTask"
                            if relation == "left"
                            else "RubiksCubeRightOfBowlMatchedTask"
                        ),
                        "environment_seed": environment_seed,
                        "sampling_seed_base": sampling_seed,
                        "first_policy_request_sampling_seed": sampling_seed * 1000,
                        "prompt_family": family,
                        "legacy_v1_wording": prompt_by_id[family]["legacy_v1_id"],
                        "requested_relation": relation,
                        "rendered_prompt": prompt,
                        "instruction_controller": "static",
                        "oracle_or_subtask_coach": False,
                        "dynamic_prompt_switches": 0,
                        "open_loop_horizon": 10,
                        "video_mode": "viewport",
                        "executed_action_trace_required": True,
                        "valid_failure_retained": True,
                        "output_folder_name": output_folder,
                        "action_trace_stem": (
                            f"seed{sampling_seed}_{family}_{relation}"
                        ),
                    }
                )
    if len(cells) != grid["episode_count"]:
        raise ValueError(f"Expected {grid['episode_count']} cells, built {len(cells)}")
    return cells


def validate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cells) != 60 or len({row["cell_id"] for row in cells}) != 60:
        raise ValueError("V2-A008 registry requires 60 unique cells")
    by_family = Counter(row["prompt_family"] for row in cells)
    by_relation = Counter(row["requested_relation"] for row in cells)
    if by_family != Counter({family: 20 for family in FAMILIES}):
        raise ValueError(f"Unexpected prompt-family counts: {by_family}")
    if by_relation != Counter({relation: 30 for relation in DIRECTIONS}):
        raise ValueError(f"Unexpected relation counts: {by_relation}")
    pair_keys = {
        (row["environment_seed"], row["prompt_family"])
        for row in cells
    }
    for pair_key in pair_keys:
        pair = [
            row
            for row in cells
            if (row["environment_seed"], row["prompt_family"]) == pair_key
        ]
        if len(pair) != 2 or {row["requested_relation"] for row in pair} != set(DIRECTIONS):
            raise ValueError(f"Incomplete LEFT/RIGHT pair: {pair_key}")
        if len({row["sampling_seed_base"] for row in pair}) != 1:
            raise ValueError(f"Paired sampling seed changed: {pair_key}")
    return {
        "episode_count": len(cells),
        "left_right_pair_count": len(pair_keys),
        "prompt_family_cell_counts": dict(sorted(by_family.items())),
        "requested_relation_cell_counts": dict(sorted(by_relation.items())),
        "environment_seeds": sorted({row["environment_seed"] for row in cells}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/vla_wam_shared_v2/pilot/expansion/"
            "pi0_fast_current_stack_v2a008_registry.json"
        ),
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    protocol_path = workspace / "artifacts/vla_wam_shared_v2/protocol.json"
    amendment_path = workspace / (
        "artifacts/vla_wam_shared_v2/pilot/"
        "post_result_current_stack_replication_amendment.json"
    )
    output_path = args.output if args.output.is_absolute() else workspace / args.output
    protocol = load_protocol(protocol_path)
    amendment = load_object(amendment_path)
    cells = build_cells(protocol, amendment)
    summary = validate_cells(cells)
    source_paths = [
        workspace / "tools/build_v2a008_pi0_current_registry.py",
        workspace / "tools/compile_v2a008_pi0_current_wording.py",
        workspace / "experiments/pi0_current_stack/v2a008_serve_policy.py",
        workspace / "experiments/pi0_current_stack/v2a008_robolab_client.py",
        workspace / "experiments/pi0_current_stack/v2a008_robolab_gate.py",
        workspace / "experiments/pi0_current_stack/v2a008_capture_fixed_observation.py",
        workspace / "experiments/pi0_current_stack/v2a008_fixed_observation_probe.py",
        workspace
        / "experiments/pi0_current_stack/robolab_v2_tasks/"
        "rubiks_cube_left_of_bowl_matched.py",
        workspace
        / "experiments/pi0_current_stack/robolab_v2_tasks/"
        "rubiks_cube_right_of_bowl_matched.py",
    ]
    payload = {
        "schema_version": "vla-wam-v2a008-pi0-current-stack-registry-v1",
        "study_id": protocol["study_id"],
        "amendment_id": AMENDMENT_ID,
        "status": "frozen_before_current_stack_model_load_or_behavioral_inference",
        "protocol": {
            "path": str(protocol_path.relative_to(workspace)),
            "sha256": sha256(protocol_path),
        },
        "amendment": {
            "path": str(amendment_path.relative_to(workspace)),
            "sha256": sha256(amendment_path),
        },
        "replication_identity": amendment["replication_identity"],
        "claim_boundary": amendment["claim_boundary"],
        "adapter_sources": [file_record(workspace, path) for path in source_paths],
        "summary": summary,
        "cells": cells,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
