#!/usr/bin/env python3
"""Freeze the exact V3-B008 queue after its independent model-blind gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/nano_start_side_v3b008.json"
CANDIDATE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/droid_gate_candidate_v3b008_v3b009.json"
GATE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b008/model_blind_gate_report.json"
GATE_MANIFEST = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b008/evidence_manifest.json"
EXPECTED = {
    REGISTRATION: "8cd7c3bda7db0c3b9097e72c54d74fe0b81fd8a3d1909b6ec4aea00748c854c2",
    CANDIDATE: "8820932c3768ca9324e1dce8b6e18c4c7847126cc4388aad6f305f7088b14f27",
    GATE: "a256113c8a53186ad2842555804a491a2bc759f22bdb14bc5c89a44b1d88cb46",
}
ARMS = ("target_start_right", "target_start_center", "target_start_left")
RELATIONS = ("left", "right")
PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
BASE_ORDER = (
    ("target_start_center", "right"),
    ("target_start_right", "left"),
    ("target_start_left", "right"),
    ("target_start_center", "left"),
    ("target_start_left", "left"),
    ("target_start_right", "right"),
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
    candidate = json.loads(CANDIDATE.read_text())
    gate = json.loads(GATE.read_text())
    if not gate.get("passed") or gate.get("model_request_count") or gate.get("behavioral_episode_count"):
        raise ValueError("V3-B008 did not pass a zero-model gate")
    if gate["reset_gate"]["repeat_count_per_task"] != 3 or len(gate["tasks"]) != 6:
        raise ValueError("V3-B008 gate coverage changed")

    seeds = registration["design"]["matched_seeds"]
    candidate_arms = candidate["studies"]["V3-B008"]["arms"]
    rows = []
    for seed_index, seed in enumerate(seeds):
        shift = seed_index % len(BASE_ORDER)
        order = BASE_ORDER[shift:] + BASE_ORDER[:shift]
        for order_index, (arm, relation) in enumerate(order):
            rows.append({
                "schema_version": "vla-wam-shared-v3b008-cell-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-B008",
                "cell_id": f"v3b008:nano:start_side:seed{seed}:{arm}:{relation}",
                "model_id": "cosmos3_nano_policy_droid",
                "arena": "droid_robolab",
                "seed": seed,
                "arm": arm,
                "relation": relation,
                "prompt": PROMPTS[relation],
                "prompt_mode": "static_episode_prompt",
                "execution_order_index_within_seed": order_index,
                "fixture_positions_robot_base_m": candidate_arms[arm]["positions_robot_base_m"],
                "registration_sha256": EXPECTED[REGISTRATION],
                "candidate_sha256": EXPECTED[CANDIDATE],
                "model_blind_gate_sha256": EXPECTED[GATE],
                "behavioral_status": "authorized_not_launched",
            })
    queue_path = output / "v3b008_cells.jsonl"
    write_new(queue_path, b"".join(canonical(row) for row in rows))

    amendment = {
        "schema_version": "vla-wam-shared-v3b008-release-amendment-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B008",
        "recorded_at_utc": "2026-08-07T03:23:00Z",
        "status": "released_after_model_blind_gate_behavior_deferred_for_server_scheduling",
        "model_id": "cosmos3_nano_policy_droid",
        "arena": "droid_robolab",
        "authorized_behavioral_cells": 162,
        "launched_behavioral_cells_at_release": 0,
        "completed_behavioral_cells_at_release": 0,
        "matched_seeds": seeds,
        "arms": list(ARMS),
        "relations": list(RELATIONS),
        "exact_prompts": PROMPTS,
        "selected_symmetric_cube_minus_bowl_lateral_offset_m": 0.1,
        "queue": record(queue_path),
        "registration": record(REGISTRATION),
        "fixture_candidate": record(CANDIDATE),
        "model_blind_gate": record(GATE),
        "model_blind_gate_manifest": record(GATE_MANIFEST),
        "runtime_boundary": "Reuse the exact hash-gated Nano policy/RoboLab stack only after the active V3-B005 and V3-C001 server schedules permit a new isolated serial owner. This amendment does not displace them or start a server.",
        "frozen_rules": [
            "one static direct-command prompt per episode",
            "same seed and reset fixture for all six cells in a matched block",
            "frozen DROID success predicate, 45-degree cone, controller, and action cap",
            "simulator viewport video and raw per-episode JSONL for every valid episode",
            "behavioral failures stay in denominators; infrastructure failures remain separate",
            "never pool DROID with RoboTwin",
        ],
    }
    amendment_path = output / "release_amendment.json"
    write_new(amendment_path, json.dumps(amendment, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")

    manifest = {
        "schema_version": "vla-wam-shared-v3b008-release-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B008",
        "status": "exact_queue_released_zero_cells_launched",
        "counts": {"seeds": 27, "arms": 3, "directions": 2, "cells": 162, "launched": 0},
        "files": [record(path) for path in (REGISTRATION, CANDIDATE, GATE, GATE_MANIFEST, queue_path, amendment_path)],
        "order_design": "Fixed six-cell base permutation rotated by seed index modulo six; every seed receives all six cells and every cell occupies each order position four or five times.",
    }
    manifest_path = output / "release_manifest.json"
    write_new(manifest_path, json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"queue": record(queue_path), "amendment": record(amendment_path), "manifest": record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
