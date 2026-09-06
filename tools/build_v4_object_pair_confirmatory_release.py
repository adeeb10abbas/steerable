#!/usr/bin/env python3
"""Promote qualified C7 from pilot-only to confirmatory-family release."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re

from tools.build_v4_object_pair_g7_pilot_release import (
    CHECKPOINT_REVISION,
    FIXTURE_ID,
    POLICY_ID,
    ROOT,
    artifact,
    canonical_json_bytes,
    load_json,
    sha256_file,
    write_exclusive,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def load_jsonl(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain JSON objects")
    return rows


def runtime_uri(path: Path, runtime_root: str) -> str:
    return f"{runtime_root}/{path.resolve().relative_to(ROOT)}"


def passing_receipt(path: Path, *, gate: str) -> dict:
    payload = load_json(path)
    if payload.get("gate") != gate:
        raise ValueError(f"{path} is not a {gate} receipt")
    if payload.get("passed") is not True or payload.get("status") != "passed":
        raise ValueError(f"{gate} receipt is not passing")
    return payload


def build_confirmatory_seed_registry(
    *,
    queue_path: Path,
    pilot_seed_registry_path: Path,
) -> dict:
    rows = [row for row in load_jsonl(queue_path) if row.get("family") == "C7"]
    if len(rows) != 768:
        raise ValueError("confirmatory C7 allocation must contain 768 rows")
    policy_seeds = list(dict.fromkeys(int(row["policy_seed"]) for row in rows))
    if len(policy_seeds) != 64:
        raise ValueError("confirmatory C7 allocation must contain 64 block seeds")
    pilot = load_json(pilot_seed_registry_path)
    pilot_seeds = {int(value) for value in pilot.get("allowed_sampling_seeds") or []}
    if set(policy_seeds) & pilot_seeds:
        raise ValueError("confirmatory C7 policy seeds collide with pilot seeds")
    return {
        "schema_version": "v4-nano-policy-seed-registry-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "policy_id": POLICY_ID,
        "scope": "c7_confirmatory_family",
        "checkpoint_revision": CHECKPOINT_REVISION,
        "allowed_sampling_seeds": policy_seeds,
        "source_queue": artifact(queue_path),
        "pilot_collision_audit": {
            "pilot_seed_registry": artifact(pilot_seed_registry_path),
            "pilot_seed_count": len(pilot_seeds),
            "collision_count": 0,
        },
        "behavioral_episode_count": 0,
        "release_boundary": (
            "Allows exactly the 64 frozen C7 confirmatory block sampling seeds "
            "covering 768 rows. Engineering-pilot seeds remain excluded."
        ),
    }


def build_runtime_lock(
    *,
    pilot_lock_path: Path,
    queue_path: Path,
    seed_registry_path: Path,
    main_reset_registry_path: Path,
    main_g2_path: Path,
    main_g3_path: Path,
    g7_path: Path,
    g8_path: Path,
    analysis_manifest_path: Path,
    source_commit: str,
    runtime_root: str,
) -> dict:
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    pilot_lock = load_json(pilot_lock_path)
    if (
        pilot_lock.get("release_status") != "PILOT_RELEASED"
        or pilot_lock.get("released_families") != ["C7"]
    ):
        raise ValueError("source lock is not the C7 pilot release")
    passing_receipt(g7_path, gate="G7")
    passing_receipt(g8_path, gate="G8")
    reset = load_json(main_reset_registry_path)
    if (
        reset.get("fixture_id") != FIXTURE_ID
        or reset.get("status") != "released_for_policy_inference"
        or reset.get("registered_env_seed_count") != 64
    ):
        raise ValueError("main C7 reset registry is not released over 64 seeds")
    g2 = load_json(main_g2_path)
    g3 = load_json(main_g3_path)
    if g2.get("passed") is not True or g3.get("passed") is not True:
        raise ValueError("main C7 G2/G3 qualification is not passing")
    lock = copy.deepcopy(pilot_lock)
    lock["manifest_sha256"] = sha256_file(queue_path)
    lock["source_commit"] = source_commit
    lock["release_status"] = "RELEASED"
    lock["runner"]["commit"] = source_commit
    lock["runner"]["entrypoint"] = f"{runtime_root}/tools/run_online_correction_v4.py"
    lock["policies"][POLICY_ID]["allowed_seed_registry_uri"] = runtime_uri(
        seed_registry_path,
        runtime_root,
    )
    lock["policies"][POLICY_ID]["allowed_seed_registry_sha256"] = sha256_file(
        seed_registry_path
    )
    fixture = lock["fixtures"][FIXTURE_ID]
    fixture["reset_registry_uri"] = runtime_uri(
        main_reset_registry_path,
        runtime_root,
    )
    fixture["reset_registry_sha256"] = sha256_file(main_reset_registry_path)
    fixture["intervention_trajectory_registry_uri"] = runtime_uri(
        main_g3_path,
        runtime_root,
    )
    receipts = lock["receipts"]

    def bind(path: Path, *, passed: bool = True) -> dict:
        return {
            "passed": passed,
            "family_ids": ["C7"],
            "uri": runtime_uri(path, runtime_root),
            "sha256": sha256_file(path),
        }

    receipts["prompt_and_frame_review"] = bind(main_g2_path)
    receipts["geometry_and_scripted_feasibility"] = bind(main_g3_path)
    receipts["historical_seed_collision_audit"] = bind(seed_registry_path)
    receipts["engineering_pilots_complete"] = bind(g7_path)
    receipts["full_miniature_campaign"] = bind(g8_path)
    receipts["frozen_analysis_and_inventory"] = bind(analysis_manifest_path)
    lock.pop("pilot_release_boundary", None)
    lock["release_boundary"] = (
        "RELEASED authorizes only frozen C7 confirmatory rows from the full "
        "17,664-row queue. Every other family remains blocked."
    )
    return lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--runtime-root",
        default="/data/users/ali/vla_wam/src/steerable-v4-c7-main",
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--pilot-runtime-lock", type=Path, required=True)
    parser.add_argument("--pilot-seed-registry", type=Path, required=True)
    parser.add_argument("--main-reset-registry", type=Path, required=True)
    parser.add_argument("--main-g2", type=Path, required=True)
    parser.add_argument("--main-g3", type=Path, required=True)
    parser.add_argument("--g7", type=Path, required=True)
    parser.add_argument("--g8", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--seed-registry-out", type=Path, required=True)
    parser.add_argument("--runtime-lock-out", type=Path, required=True)
    args = parser.parse_args()
    seed_registry = build_confirmatory_seed_registry(
        queue_path=args.queue.resolve(),
        pilot_seed_registry_path=args.pilot_seed_registry.resolve(),
    )
    write_exclusive(
        args.seed_registry_out.resolve(),
        canonical_json_bytes(seed_registry),
    )
    lock = build_runtime_lock(
        pilot_lock_path=args.pilot_runtime_lock.resolve(),
        queue_path=args.queue.resolve(),
        seed_registry_path=args.seed_registry_out.resolve(),
        main_reset_registry_path=args.main_reset_registry.resolve(),
        main_g2_path=args.main_g2.resolve(),
        main_g3_path=args.main_g3.resolve(),
        g7_path=args.g7.resolve(),
        g8_path=args.g8.resolve(),
        analysis_manifest_path=args.analysis_manifest.resolve(),
        source_commit=args.source_commit,
        runtime_root=args.runtime_root.rstrip("/"),
    )
    write_exclusive(
        args.runtime_lock_out.resolve(),
        canonical_json_bytes(lock),
    )
    print(
        json.dumps(
            {
                "confirmatory_episode_count": len(
                    [
                        row
                        for row in load_jsonl(args.queue.resolve())
                        if row.get("family") == "C7"
                    ]
                ),
                "allowed_block_seed_count": len(
                    seed_registry["allowed_sampling_seeds"]
                ),
                "seed_registry": artifact(args.seed_registry_out.resolve()),
                "runtime_lock": artifact(args.runtime_lock_out.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
