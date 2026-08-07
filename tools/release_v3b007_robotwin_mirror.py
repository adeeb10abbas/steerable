#!/usr/bin/env python3
"""Freeze the exact V3-B007 FastWAM RoboTwin mirror queue."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/fastwam_robotwin_mirror_v3b007.json"
GATE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b007/model_blind_gate_report.json"
GATE_MANIFEST = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b007/evidence_manifest.json"
EXPECTED = {
    REGISTRATION: "84d14a5c6a02c5f6655384d2ed1ef6e3cdaab05341136d81a3b0e727268ecc8e",
    GATE: "e092917893591490f1b1ee2ab2f9c6bd4cd9cc560fa5702d49dd6974a301d6ad",
}
PROMPTS = {
    "left": "Put the small woodenblock to the left of the red playingcards box.",
    "right": "Put the small woodenblock to the right of the red playingcards box.",
}
BASE_ORDER = (
    ("control", "right"),
    ("position_mirrored", "left"),
    ("control", "left"),
    ("position_mirrored", "right"),
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(path: Path) -> dict:
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": digest(path)}


def canonical(value: dict) -> bytes:
    return (json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    for path, expected in EXPECTED.items():
        if digest(path) != expected:
            raise ValueError(f"input digest changed: {path}")
    registration = json.loads(REGISTRATION.read_text())
    gate = json.loads(GATE.read_text())
    if not gate.get("passed") or gate.get("model_request_count") or gate.get("behavioral_episode_count"):
        raise ValueError("V3-B007 did not pass a zero-model gate")
    if len(gate["tasks"]) != 4 or gate["source_identity"] != {"object": ["086_woodenblock", 1], "reference": ["081_playingcards", 1]}:
        raise ValueError("V3-B007 gate identity/coverage changed")
    fixtures = gate["derived_numeric_fixture"]
    seeds = registration["design"]["matched_seeds"]
    rows = []
    for seed_index, seed in enumerate(seeds):
        shift = seed_index % len(BASE_ORDER)
        order = BASE_ORDER[shift:] + BASE_ORDER[:shift]
        for order_index, (arm, relation) in enumerate(order):
            rows.append({
                "schema_version": "vla-wam-shared-v3b007-cell-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-B007",
                "cell_id": f"v3b007:fastwam:robotwin:seed{seed}:{arm}:{relation}",
                "model_id": "fastwam_robotwin",
                "arena": "robotwin",
                "anchor_task": "place_a2b_right",
                "source_fixture_environment_seed": 4300003,
                "matched_seed": seed,
                "environment_seed": seed,
                "sampling_seed": seed,
                "arm": arm,
                "relation": relation,
                "prompt": PROMPTS[relation],
                "prompt_mode": "static_episode_prompt",
                "execution_order_index_within_seed": order_index,
                "fixture": fixtures[arm],
                "registration_sha256": EXPECTED[REGISTRATION],
                "model_blind_gate_sha256": EXPECTED[GATE],
                "behavioral_status": "authorized_not_launched",
            })
    queue_path = output / "v3b007_cells.jsonl"
    write_new(queue_path, b"".join(canonical(row) for row in rows))

    amendment = {
        "schema_version": "vla-wam-shared-v3b007-release-amendment-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "recorded_at_utc": "2026-08-07T03:27:00Z",
        "status": "released_after_model_blind_gate_behavior_deferred_for_lane_scheduling",
        "model_id": "fastwam_robotwin",
        "arena": "robotwin",
        "authorized_behavioral_cells": 108,
        "launched_behavioral_cells_at_release": 0,
        "completed_behavioral_cells_at_release": 0,
        "matched_seeds": seeds,
        "arms": ["control", "position_mirrored"],
        "relations": ["left", "right"],
        "exact_prompts": PROMPTS,
        "queue": record(queue_path),
        "registration": record(REGISTRATION),
        "model_blind_gate": record(GATE),
        "model_blind_gate_manifest": record(GATE_MANIFEST),
        "runtime_boundary": "Use FastWAM@068d3fd70c89df3726c09893f47b75a624b20c02, its frozen checkpoint/runtime, SAPIEN renderer, and the architecture-compatible curobo-fastwam-rtxpro-sm120-torch27 build on an isolated ali lane. No behavior was launched by this amendment.",
        "arena_boundary": "RoboTwin results remain separate from every DROID denominator and success rate.",
        "frozen_rules": [
            "same environment and sampling seed for all four cells in a matched block",
            "static exact direct-command prompt only",
            "center-position x reflection only; quaternions, robot, cameras, and nonmovable geometry unchanged",
            "RoboTwin relation-aware predicate, controller, and 400-action cap unchanged",
            "simulator video, raw action trace, trajectory, and per-episode JSONL retained",
            "valid failures stay in denominators; infrastructure failures remain separate",
        ],
    }
    amendment_path = output / "release_amendment.json"
    write_new(amendment_path, json.dumps(amendment, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")
    manifest = {
        "schema_version": "vla-wam-shared-v3b007-release-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B007",
        "status": "exact_queue_released_zero_cells_launched",
        "counts": {"seeds": 27, "arms": 2, "directions": 2, "cells": 108, "launched": 0},
        "files": [record(path) for path in (REGISTRATION, GATE, GATE_MANIFEST, queue_path, amendment_path)],
        "order_design": "Fixed four-cell base permutation rotated by seed index modulo four; every seed receives all four cells and every cell occupies each order position six or seven times.",
    }
    manifest_path = output / "release_manifest.json"
    write_new(manifest_path, json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"queue": record(queue_path), "amendment": record(amendment_path), "manifest": record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
