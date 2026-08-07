#!/usr/bin/env python3
"""Independent fail-closed validator for the V3-B008 release bundle."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/releases/v3b008"
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
ARMS = {"target_start_right", "target_start_center", "target_start_left"}


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
    gate = load(ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b008/model_blind_gate_report.json")
    require(gate.get("passed") is True and gate.get("model_request_count") == 0 and gate.get("behavioral_episode_count") == 0, "gate is not model-blind pass")
    require(amendment.get("authorized_behavioral_cells") == 162 and amendment.get("launched_behavioral_cells_at_release") == 0, "release count/status drift")
    require(manifest.get("counts") == {"seeds": 27, "arms": 3, "directions": 2, "cells": 162, "launched": 0}, "manifest counts drift")
    for row in manifest["files"]:
        path = ROOT / row["path"]
        require(path.is_file() and path.stat().st_size == row["bytes"] and digest(path) == row["sha256"], f"hash mismatch: {path}")

    lines = (RELEASE / "v3b008_cells.jsonl").read_text().splitlines()
    require(len(lines) == 162 and all(line.strip() for line in lines), "queue must contain 162 nonblank rows")
    rows = [json.loads(line) for line in lines]
    require(len({row["cell_id"] for row in rows}) == 162, "cell IDs are not unique")
    by_seed = defaultdict(list)
    order_counts = Counter()
    for row in rows:
        require(row["amendment_id"] == "V3-B008" and row["behavioral_status"] == "authorized_not_launched", "row release state drift")
        require(row["arm"] in ARMS and row["relation"] in PROMPTS and row["prompt"] == PROMPTS[row["relation"]], "arm/relation/prompt drift")
        require(row["prompt_mode"] == "static_episode_prompt", "prompt controller drift")
        by_seed[row["seed"]].append(row)
        order_counts[(row["arm"], row["relation"], row["execution_order_index_within_seed"])] += 1
    require(set(by_seed) == set(range(9700, 9727)), "seed list drift")
    expected_cells = {(arm, relation) for arm in ARMS for relation in PROMPTS}
    for seed, block in by_seed.items():
        require(len(block) == 6 and {(row["arm"], row["relation"]) for row in block} == expected_cells, f"seed {seed} does not have all six cells")
        require({row["execution_order_index_within_seed"] for row in block} == set(range(6)), f"seed {seed} order indices drift")
    require(all(count in (4, 5) for count in order_counts.values()) and len(order_counts) == 36, "order-position balance drift")
    print("V3-B008 release validation passed: 27 matched seeds, 162 exact cells, 0 launched")


if __name__ == "__main__":
    main()
