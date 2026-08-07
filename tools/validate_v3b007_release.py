#!/usr/bin/env python3
"""Independent fail-closed validator for the V3-B007 release."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3b007"
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
ARMS = {"control", "position_mirrored"}


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
    gate = load(ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b007/model_blind_gate_report.json")
    require(gate.get("passed") is True and gate.get("model_request_count") == 0 and gate.get("behavioral_episode_count") == 0, "gate is not a model-blind pass")
    require(gate.get("arena") == "robotwin" and gate["reset_gate"]["live_center_reflection_passed"] is True, "RoboTwin mirror gate drift")
    require(amendment.get("authorized_behavioral_cells") == 108 and amendment.get("launched_behavioral_cells_at_release") == 0, "release count/status drift")
    require(manifest.get("counts") == {"seeds": 27, "arms": 2, "directions": 2, "cells": 108, "launched": 0}, "manifest counts drift")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"hash mismatch: {path}")
    lines = (RELEASE / "v3b007_cells.jsonl").read_text().splitlines()
    require(len(lines) == 108 and all(line.strip() for line in lines), "queue must contain 108 nonblank rows")
    rows = [json.loads(line) for line in lines]
    require(len({row["cell_id"] for row in rows}) == 108, "cell IDs are not unique")
    by_seed = defaultdict(list)
    order_counts = Counter()
    for row in rows:
        require(row["arena"] == "robotwin" and row["anchor_task"] == "place_a2b_right", "arena/task drift")
        require(row["arm"] in ARMS and row["relation"] in PROMPTS and row["prompt"] == PROMPTS[row["relation"]], "arm/relation/prompt drift")
        require(row["prompt_mode"] == "static_episode_prompt" and row["behavioral_status"] == "authorized_not_launched", "row release state drift")
        require(row["environment_seed"] == row["sampling_seed"] == row["matched_seed"], "matched seed is not shared")
        by_seed[row["matched_seed"]].append(row)
        order_counts[(row["arm"], row["relation"], row["execution_order_index_within_seed"])] += 1
    require(set(by_seed) == set(range(9900, 9927)), "seed list drift")
    expected_cells = {(arm, relation) for arm in ARMS for relation in PROMPTS}
    for seed, block in by_seed.items():
        require(len(block) == 4 and {(row["arm"], row["relation"]) for row in block} == expected_cells, f"seed {seed} does not have all four cells")
        require({row["execution_order_index_within_seed"] for row in block} == set(range(4)), f"seed {seed} order indices drift")
    require(all(count in (6, 7) for count in order_counts.values()) and len(order_counts) == 16, "order-position balance drift")
    print("V3-B007 release validation passed: 27 RoboTwin matched seeds, 108 exact cells, 0 launched")


if __name__ == "__main__":
    main()
