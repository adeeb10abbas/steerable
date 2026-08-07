#!/usr/bin/env python3
"""Independent fail-closed validator for the V3-B009 release bundle."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3b009"
PROMPTS = {
    ("cube_target_bowl_reference", "left"): "Put the Rubik's cube to the left of the bowl.",
    ("cube_target_bowl_reference", "right"): "Put the Rubik's cube to the right of the bowl.",
    ("bowl_target_cube_reference", "left"): "Put the bowl to the left of the Rubik's cube.",
    ("bowl_target_cube_reference", "right"): "Put the bowl to the right of the Rubik's cube.",
}
ARMS = {"cube_target_bowl_reference", "bowl_target_cube_reference"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def main() -> None:
    manifest = load(RELEASE / "release_manifest.json")
    amendment = load(RELEASE / "release_amendment.json")
    gate = load(ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b009/model_blind_gate_report.json")
    require(gate.get("passed") is True and gate.get("model_request_count") == 0 and gate.get("behavioral_episode_count") == 0, "gate is not model-blind pass")
    require(amendment.get("authorized_behavioral_cells") == 108 and amendment.get("launched_behavioral_cells_at_release") == 0, "release count/status drift")
    require(manifest.get("counts") == {"seeds": 27, "arms": 2, "directions": 2, "cells": 108, "launched": 0}, "manifest counts drift")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"hash mismatch: {path}")

    lines = (RELEASE / "v3b009_cells.jsonl").read_text().splitlines()
    require(len(lines) == 108 and all(line.strip() for line in lines), "queue must contain 108 nonblank rows")
    rows = [json.loads(line) for line in lines]
    require(len({row["cell_id"] for row in rows}) == 108, "cell IDs are not unique")
    by_seed = defaultdict(list)
    order_counts = Counter()
    for row in rows:
        key = (row["arm"], row["relation"])
        require(row["amendment_id"] == "V3-B009" and row["behavioral_status"] == "authorized_not_launched", "row release state drift")
        require(row["arm"] in ARMS and key in PROMPTS and row["prompt"] == PROMPTS[key], "arm/relation/prompt drift")
        require(row["prompt_mode"] == "static_episode_prompt", "prompt controller drift")
        if row["arm"] == "cube_target_bowl_reference":
            require((row["target_object"], row["reference_object"]) == ("rubiks_cube", "bowl"), "cube-target roles drift")
        else:
            require((row["target_object"], row["reference_object"]) == ("bowl", "rubiks_cube"), "bowl-target roles drift")
        by_seed[row["seed"]].append(row)
        order_counts[(row["arm"], row["relation"], row["execution_order_index_within_seed"])] += 1
    require(set(by_seed) == set(range(9800, 9827)), "seed list drift")
    expected_cells = set(PROMPTS)
    for seed, block in by_seed.items():
        require(len(block) == 4 and {(row["arm"], row["relation"]) for row in block} == expected_cells, f"seed {seed} does not have all four cells")
        require({row["execution_order_index_within_seed"] for row in block} == set(range(4)), f"seed {seed} order indices drift")
        require(len({canonical_positions(row["fixture_positions_robot_base_m"]) for row in block}) == 1, f"seed {seed} physical poses changed across role cells")
    require(all(count in (6, 7) for count in order_counts.values()) and len(order_counts) == 16, "order-position balance drift")
    print("V3-B009 release validation passed: 27 matched seeds, 108 exact cells, 0 launched")


def canonical_positions(value: dict) -> str:
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


if __name__ == "__main__":
    main()
