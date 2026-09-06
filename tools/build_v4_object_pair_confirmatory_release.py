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
from tools.derive_v4_lane_spec import derive_spec

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
    hardware_g4_path: Path,
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
    hardware_g4 = passing_receipt(hardware_g4_path, gate="G4")
    if hardware_g4.get("hardware_stratum") != "a10080-policy_a40-simulator":
        raise ValueError("C7 confirmatory hardware G4 receipt has wrong stratum")
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
    lock["policies"][POLICY_ID][
        "policy_reset_and_history_contract_uri"
    ] = runtime_uri(hardware_g4_path, runtime_root)
    fixture = lock["fixtures"][FIXTURE_ID]
    scorer_path = ROOT / "experiments/online_correction_v4/droid_scorer.py"
    fixture["scorer_uri"] = runtime_uri(scorer_path, runtime_root)
    fixture["scorer_sha256"] = sha256_file(scorer_path)
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
    receipts["cluster_lane_qualification"] = bind(hardware_g4_path)
    receipts["source_and_checkpoint_identity"] = bind(hardware_g4_path)
    receipts["historical_seed_collision_audit"] = bind(seed_registry_path)
    receipts["engineering_pilots_complete"] = bind(g7_path)
    receipts["terminal_metadata_writer_amendment"] = bind(g7_path)
    receipts["full_miniature_campaign"] = bind(g8_path)
    receipts["frozen_analysis_and_inventory"] = bind(analysis_manifest_path)
    lock.pop("pilot_release_boundary", None)
    lock["release_boundary"] = (
        "RELEASED authorizes only frozen C7 confirmatory rows from the full "
        "17,664-row queue. Every other family remains blocked."
    )
    return lock


def build_confirmatory_lane_spec(
    *,
    pilot_lane_spec_path: Path,
    pilot_seed_registry_path: Path,
    seed_registry_path: Path,
    runtime_root: str,
    output_parent: str,
) -> dict:
    pilot_spec = load_json(pilot_lane_spec_path)
    pilot_pythonpath = str(pilot_spec["runtime"]["policy"]["pythonpath"])
    pilot_runtime_root = pilot_pythonpath.split(":", 1)[0]
    return derive_spec(
        source_path=pilot_lane_spec_path,
        overrides=[
            'lane_id="c7m00"',
            'attempt_id="c7mainrelease20260906a"',
            "policy_port=18157",
            f"output_parent={json.dumps(output_parent)}",
            'policy.gpu_product="NVIDIA-A100-SXM4-80GB"',
            'policy.expected_gpu_name="NVIDIA A100-SXM4-80GB"',
        ],
        replacements=[
            f"{pilot_runtime_root}={runtime_root}",
            f"{pilot_seed_registry_path.name}={seed_registry_path.name}",
            (
                f"{sha256_file(pilot_seed_registry_path)}="
                f"{sha256_file(seed_registry_path)}"
            ),
            "g7-object-pair=c7-object-pair-main",
        ],
        absolutize_sources=False,
    )


def build_launch_matrix(
    *,
    lane_spec_path: Path,
    runtime_lock_path: Path,
    hardware_g4_path: Path,
    lane_count: int,
) -> dict:
    if lane_count < 1 or lane_count > 40:
        raise ValueError("C7 confirmatory lane count must be between 1 and 40")
    lock = load_json(runtime_lock_path)
    if (
        lock.get("release_status") != "RELEASED"
        or lock.get("released_families") != ["C7"]
    ):
        raise ValueError("C7 confirmatory runtime lock is not released")
    hardware = passing_receipt(hardware_g4_path, gate="G4")
    if hardware.get("hardware_stratum") != "a10080-policy_a40-simulator":
        raise ValueError("C7 launch matrix hardware stratum is not qualified")
    spec = load_json(lane_spec_path)
    if (
        spec.get("qualification_only") is not False
        or spec.get("policy", {}).get("gpu_product") != "NVIDIA-A100-SXM4-80GB"
        or spec.get("simulator", {}).get("gpu_product") != "NVIDIA-A40"
    ):
        raise ValueError("C7 confirmatory lane spec has wrong execution hardware")
    return {
        "schema_version": 1,
        "campaign_id": "online_correction_v4",
        "release_status": "RELEASED",
        "qualified_lanes": [
            {
                "lane_id": f"c7m{index:02d}",
                "hardware_stratum": "a10080-policy_a40-simulator",
                "lane_spec_template_path": str(
                    lane_spec_path.resolve().relative_to(ROOT)
                ),
            }
            for index in range(lane_count)
        ],
        "resource_budget": {
            "authorized_storage_bytes": 1000000000000,
            "estimated_bytes_per_episode": 500000000,
            "estimated_bytes_per_infra_retry": 500000000,
        },
        "dispatch": {
            "max_infra_retries_per_episode": 3,
            "lane_quarantine_threshold": 3,
        },
        "bindings": {
            "runtime_lock": artifact(runtime_lock_path),
            "hardware_g4_receipt": artifact(hardware_g4_path),
            "lane_spec": artifact(lane_spec_path),
        },
        "release_boundary": (
            f"{lane_count} qualified A100-80GB policy/A40 simulator lanes for "
            "the 768 frozen C7 confirmatory rows only."
        ),
    }


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
    parser.add_argument("--hardware-g4", type=Path, required=True)
    parser.add_argument("--g7", type=Path, required=True)
    parser.add_argument("--g8", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--pilot-lane-spec", type=Path, required=True)
    parser.add_argument("--lane-spec-out", type=Path, required=True)
    parser.add_argument("--launch-matrix-out", type=Path, required=True)
    parser.add_argument("--lane-count", type=int, default=40)
    parser.add_argument(
        "--output-parent",
        default="/data/users/ali/vla_wam/raw/v4/c7-object-pair-main",
    )
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
        hardware_g4_path=args.hardware_g4.resolve(),
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
    lane_spec = build_confirmatory_lane_spec(
        pilot_lane_spec_path=args.pilot_lane_spec.resolve(),
        pilot_seed_registry_path=args.pilot_seed_registry.resolve(),
        seed_registry_path=args.seed_registry_out.resolve(),
        runtime_root=args.runtime_root.rstrip("/"),
        output_parent=args.output_parent.rstrip("/"),
    )
    write_exclusive(
        args.lane_spec_out.resolve(),
        json.dumps(
            lane_spec,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        + b"\n",
    )
    launch_matrix = build_launch_matrix(
        lane_spec_path=args.lane_spec_out.resolve(),
        runtime_lock_path=args.runtime_lock_out.resolve(),
        hardware_g4_path=args.hardware_g4.resolve(),
        lane_count=args.lane_count,
    )
    write_exclusive(
        args.launch_matrix_out.resolve(),
        canonical_json_bytes(launch_matrix),
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
                "lane_spec": artifact(args.lane_spec_out.resolve()),
                "launch_matrix": artifact(args.launch_matrix_out.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
