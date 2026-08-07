#!/usr/bin/env python3
"""Freeze the exact V3-B009 queue after its model-blind role gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/nano_role_swap_v3b009.json"
CANDIDATE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/droid_gate_candidate_v3b008_v3b009.json"
GATE = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b009/model_blind_gate_report.json"
GATE_MANIFEST = ROOT / "artifacts/vla_wam_shared_v3/prospective_tier_b/gates/v3b009/evidence_manifest.json"
EXPECTED = {
    REGISTRATION: "b6128c0ace0982980f1e650186324644cd89b896c2c1bb01c807adba584c1108",
    CANDIDATE: "8820932c3768ca9324e1dce8b6e18c4c7847126cc4388aad6f305f7088b14f27",
    GATE: "d102e99cda8a38d1c936488c886fe2317cddf5cf13cf7eca149a010de97314b0",
}
ARMS = ("cube_target_bowl_reference", "bowl_target_cube_reference")
RELATIONS = ("left", "right")
PROMPTS = {
    ("cube_target_bowl_reference", "left"): "Put the Rubik's cube to the left of the bowl.",
    ("cube_target_bowl_reference", "right"): "Put the Rubik's cube to the right of the bowl.",
    ("bowl_target_cube_reference", "left"): "Put the bowl to the left of the Rubik's cube.",
    ("bowl_target_cube_reference", "right"): "Put the bowl to the right of the Rubik's cube.",
}
BASE_ORDER = (
    ("cube_target_bowl_reference", "right"),
    ("bowl_target_cube_reference", "left"),
    ("cube_target_bowl_reference", "left"),
    ("bowl_target_cube_reference", "right"),
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
        raise ValueError("V3-B009 did not pass a zero-model gate")
    if gate["reset_gate"]["repeat_count_per_task"] != 3 or len(gate["tasks"]) != 4:
        raise ValueError("V3-B009 gate coverage changed")

    seeds = registration["design"]["matched_seeds"]
    candidate_arms = candidate["studies"]["V3-B009"]["arms"]
    rows = []
    for seed_index, seed in enumerate(seeds):
        shift = seed_index % len(BASE_ORDER)
        order = BASE_ORDER[shift:] + BASE_ORDER[:shift]
        for order_index, (arm, relation) in enumerate(order):
            row = candidate_arms[arm]
            rows.append({
                "schema_version": "vla-wam-shared-v3b009-cell-v1",
                "study_id": "vla_wam_language_steerability_v3",
                "amendment_id": "V3-B009",
                "cell_id": f"v3b009:nano:role_swap:seed{seed}:{arm}:{relation}",
                "model_id": "cosmos3_nano_policy_droid",
                "arena": "droid_robolab",
                "seed": seed,
                "arm": arm,
                "target_object": row["target_object"],
                "reference_object": row["reference_object"],
                "relation": relation,
                "prompt": PROMPTS[(arm, relation)],
                "prompt_mode": "static_episode_prompt",
                "execution_order_index_within_seed": order_index,
                "fixture_positions_robot_base_m": row["positions_robot_base_m"],
                "registration_sha256": EXPECTED[REGISTRATION],
                "candidate_sha256": EXPECTED[CANDIDATE],
                "model_blind_gate_sha256": EXPECTED[GATE],
                "behavioral_status": "authorized_not_launched",
            })
    queue_path = output / "v3b009_cells.jsonl"
    write_new(queue_path, b"".join(canonical(row) for row in rows))

    amendment = {
        "schema_version": "vla-wam-shared-v3b009-release-amendment-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B009",
        "recorded_at_utc": "2026-08-07T03:38:00Z",
        "status": "released_after_model_blind_gate_behavior_deferred_for_server_scheduling",
        "model_id": "cosmos3_nano_policy_droid",
        "arena": "droid_robolab",
        "authorized_behavioral_cells": 108,
        "launched_behavioral_cells_at_release": 0,
        "completed_behavioral_cells_at_release": 0,
        "matched_seeds": seeds,
        "arms": list(ARMS),
        "relations": list(RELATIONS),
        "exact_prompts": {f"{arm}:{relation}": prompt for (arm, relation), prompt in PROMPTS.items()},
        "queue": record(queue_path),
        "registration": record(REGISTRATION),
        "fixture_candidate": record(CANDIDATE),
        "model_blind_gate": record(GATE),
        "model_blind_gate_manifest": record(GATE_MANIFEST),
        "runtime_boundary": "Reuse the exact hash-gated Nano policy/RoboLab stack only after active isolated Nano schedules permit it. This release starts no model server and launches no behavior.",
        "claim_boundary": "Role swap changes semantic/scoring role and object affordance together; a measured interaction is role-affordance evidence, not a pure language-only effect.",
        "frozen_rules": [
            "one static direct-command prompt per episode",
            "same seed and full physical reset for all four cells in a matched block",
            "pickup, transport, cone, and detached-release scoring follow the named target and reference",
            "frozen DROID controller, 45-degree cone, and action cap",
            "simulator viewport video and raw per-episode JSONL for every valid episode",
            "behavioral failures stay in denominators; infrastructure failures remain separate",
            "never pool DROID with RoboTwin",
        ],
    }
    amendment_path = output / "release_amendment.json"
    write_new(amendment_path, json.dumps(amendment, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")

    manifest = {
        "schema_version": "vla-wam-shared-v3b009-release-manifest-v1",
        "study_id": "vla_wam_language_steerability_v3",
        "amendment_id": "V3-B009",
        "status": "exact_queue_released_zero_cells_launched",
        "counts": {"seeds": 27, "arms": 2, "directions": 2, "cells": 108, "launched": 0},
        "files": [record(path) for path in (REGISTRATION, CANDIDATE, GATE, GATE_MANIFEST, queue_path, amendment_path)],
        "order_design": "Fixed four-cell base permutation rotated by seed index modulo four; every seed receives all four cells and each cell occupies every order position six or seven times.",
    }
    manifest_path = output / "release_manifest.json"
    write_new(manifest_path, json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps({"queue": record(queue_path), "amendment": record(amendment_path), "manifest": record(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
