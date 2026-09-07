#!/usr/bin/env python3
"""Promote qualified fixture families from pilot-only to confirmatory release."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.online_correction_v4.fixture_qualification import qualification_profile
from tools.build_v4_object_pair_g7_pilot_release import (  # noqa: E402
    CHECKPOINT_REVISION,
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
    if payload.get("passed") is not True and payload.get("status") != "passed":
        raise ValueError(f"{gate} receipt is not passing")
    return payload


def build_confirmatory_seed_registry(
    *,
    fixture_id: str,
    queue_path: Path,
    pilot_seed_registry_path: Path,
) -> dict:
    profile = qualification_profile(fixture_id)
    rows = [row for row in load_jsonl(queue_path) if row.get("family") == profile.family_id]
    if len(rows) != profile.confirmatory_episode_count:
        raise ValueError(
            f"confirmatory {profile.family_id} allocation must contain "
            f"{profile.confirmatory_episode_count} rows"
        )
    policy_seeds = list(dict.fromkeys(int(row["policy_seed"]) for row in rows))
    if len(policy_seeds) != profile.confirmatory_block_seed_count:
        raise ValueError(
            f"confirmatory {profile.family_id} allocation must contain "
            f"{profile.confirmatory_block_seed_count} block seeds"
        )
    pilot = load_json(pilot_seed_registry_path)
    pilot_seeds = {int(value) for value in pilot.get("allowed_sampling_seeds") or []}
    if set(policy_seeds) & pilot_seeds:
        raise ValueError(
            f"confirmatory {profile.family_id} policy seeds collide with pilot seeds"
        )
    return {
        "schema_version": profile.nano_seed_registry_schema,
        "campaign_id": "online_correction_v4",
        "fixture_id": fixture_id,
        "policy_id": profile.policy_id,
        "scope": f"released_{profile.family_id.lower()}",
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
            f"Allows exactly the {profile.confirmatory_block_seed_count} frozen "
            f"{profile.family_id} confirmatory block sampling seeds covering "
            f"{profile.confirmatory_episode_count} rows. Engineering-pilot seeds "
            "remain excluded."
        ),
    }


def build_runtime_lock(
    *,
    fixture_id: str,
    pilot_lock_path: Path,
    queue_path: Path,
    queue_manifest_path: Path,
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
    profile = qualification_profile(fixture_id)
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    pilot_lock = load_json(pilot_lock_path)
    if (
        pilot_lock.get("release_status") != "PILOT_RELEASED"
        or pilot_lock.get("released_families") != [profile.family_id]
    ):
        raise ValueError("source lock is not the pilot release")
    passing_receipt(g7_path, gate="G7")
    passing_receipt(g8_path, gate="G8")
    hardware_g4 = passing_receipt(hardware_g4_path, gate="G4")
    if profile.hardware_stratum is not None:
        observed = hardware_g4.get("hardware_stratum")
        if observed is not None and observed != profile.hardware_stratum:
            raise ValueError(f"{profile.family_id} confirmatory hardware G4 receipt has wrong stratum")
    reset = load_json(main_reset_registry_path)
    if (
        reset.get("fixture_id") != fixture_id
        or reset.get("status") != "released_for_policy_inference"
        or reset.get("registered_env_seed_count") != profile.confirmatory_block_seed_count
    ):
        raise ValueError(f"main {profile.family_id} reset registry is not released")
    g2 = load_json(main_g2_path)
    g3 = load_json(main_g3_path)
    if g2.get("passed") is not True or g3.get("passed") is not True:
        raise ValueError(f"main {profile.family_id} G2/G3 qualification is not passing")
    queue_manifest = load_json(queue_manifest_path)
    if queue_manifest.get("queue_sha256") != sha256_file(queue_path):
        raise ValueError("confirmatory queue hash differs from queue manifest")
    planning_manifest_sha256 = queue_manifest.get("planning_manifest_sha256")
    if (
        not isinstance(planning_manifest_sha256, str)
        or len(planning_manifest_sha256) != 64
    ):
        raise ValueError("confirmatory queue manifest lacks planning hash")
    lock = copy.deepcopy(pilot_lock)
    lock["manifest_sha256"] = planning_manifest_sha256
    lock["frozen_queue_sha256"] = sha256_file(queue_path)
    lock["source_commit"] = source_commit
    lock["release_status"] = "RELEASED"
    lock["runner"]["commit"] = source_commit
    lock["runner"]["entrypoint"] = f"{runtime_root}/tools/run_online_correction_v4.py"
    lock["runner"]["sha256"] = sha256_file(ROOT / "tools/run_online_correction_v4.py")
    lock["policies"][profile.policy_id]["allowed_seed_registry_uri"] = runtime_uri(
        seed_registry_path,
        runtime_root,
    )
    lock["policies"][profile.policy_id]["allowed_seed_registry_sha256"] = sha256_file(
        seed_registry_path
    )
    lock["policies"][profile.policy_id][
        "policy_reset_and_history_contract_uri"
    ] = runtime_uri(hardware_g4_path, runtime_root)
    fixture = lock["fixtures"][fixture_id]
    scorer_path = ROOT / "experiments/online_correction_v4/droid_scorer.py"
    fixture["scorer_uri"] = runtime_uri(scorer_path, runtime_root)
    fixture["scorer_sha256"] = sha256_file(scorer_path)
    fixture["reset_registry_uri"] = runtime_uri(main_reset_registry_path, runtime_root)
    fixture["reset_registry_sha256"] = sha256_file(main_reset_registry_path)
    fixture["intervention_trajectory_registry_uri"] = runtime_uri(main_g3_path, runtime_root)
    receipts = lock["receipts"]

    def bind(path: Path, *, passed: bool = True) -> dict:
        return {
            "passed": passed,
            "family_ids": [profile.family_id],
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
        f"RELEASED authorizes only frozen {profile.family_id} confirmatory rows from "
        "the full 17,664-row queue. Every other family remains blocked."
    )
    return lock


def build_confirmatory_lane_spec(
    *,
    fixture_id: str,
    pilot_lane_spec_path: Path,
    pilot_seed_registry_path: Path,
    seed_registry_path: Path,
    runtime_root: str,
    output_parent: str,
    lane_id: str,
    attempt_id: str,
    policy_port: int,
    policy_gpu: str,
    sim_gpu: str,
) -> dict:
    pilot_spec = load_json(pilot_lane_spec_path)
    pilot_pythonpath = str(pilot_spec["runtime"]["policy"]["pythonpath"])
    pilot_runtime_root = pilot_pythonpath.split(":", 1)[0]
    return derive_spec(
        source_path=pilot_lane_spec_path,
        overrides=[
            f'lane_id="{lane_id}"',
            f'attempt_id="{attempt_id}"',
            f"policy_port={policy_port}",
            f"output_parent={json.dumps(output_parent)}",
            f'policy.gpu_product="{policy_gpu}"',
            f'policy.expected_gpu_name="{policy_gpu.replace("-", " ")}"',
            f'simulator.gpu_product="{sim_gpu}"',
            f'simulator.expected_gpu_name="{sim_gpu.replace("-", " ")}"',
        ],
        replacements=[
            f"{pilot_runtime_root}={runtime_root}",
            f"{pilot_seed_registry_path.name}={seed_registry_path.name}",
            (
                f"{sha256_file(pilot_seed_registry_path)}="
                f"{sha256_file(seed_registry_path)}"
            ),
        ],
        absolutize_sources=False,
    )


def build_launch_matrix(
    *,
    fixture_id: str,
    lane_spec_path: Path,
    runtime_lock_path: Path,
    hardware_g4_path: Path,
    lane_count: int,
    lane_id_prefix: str,
) -> dict:
    profile = qualification_profile(fixture_id)
    if lane_count < 1 or lane_count > 40:
        raise ValueError("confirmatory lane count must be between 1 and 40")
    lock = load_json(runtime_lock_path)
    if (
        lock.get("release_status") != "RELEASED"
        or lock.get("released_families") != [profile.family_id]
    ):
        raise ValueError(f"{profile.family_id} confirmatory runtime lock is not released")
    hardware = passing_receipt(hardware_g4_path, gate="G4")
    if profile.hardware_stratum is not None:
        observed = hardware.get("hardware_stratum")
        if observed is not None and observed != profile.hardware_stratum:
            raise ValueError(f"{profile.family_id} launch matrix hardware stratum mismatch")
    return {
        "schema_version": 1,
        "campaign_id": "online_correction_v4",
        "release_status": "RELEASED",
        "qualified_lanes": [
            {
                "lane_id": f"{lane_id_prefix}{index:02d}",
                "hardware_stratum": profile.hardware_stratum,
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
            f"{lane_count} qualified lanes for the {profile.confirmatory_episode_count} "
            f"frozen {profile.family_id} confirmatory rows only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-manifest", type=Path, required=True)
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
    parser.add_argument("--lane-id-prefix", default=None)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--policy-port", type=int, default=18157)
    parser.add_argument("--output-parent", required=True)
    parser.add_argument("--policy-gpu", default="NVIDIA-B200")
    parser.add_argument("--sim-gpu", default="NVIDIA-A100-SXM4-40GB")
    parser.add_argument("--seed-registry-out", type=Path, required=True)
    parser.add_argument("--runtime-lock-out", type=Path, required=True)
    args = parser.parse_args()
    profile = qualification_profile(args.fixture_id)
    lane_prefix = args.lane_id_prefix or f"{profile.family_id.lower()}m"
    seed_registry = build_confirmatory_seed_registry(
        fixture_id=args.fixture_id,
        queue_path=args.queue.resolve(),
        pilot_seed_registry_path=args.pilot_seed_registry.resolve(),
    )
    write_exclusive(
        args.seed_registry_out.resolve(),
        canonical_json_bytes(seed_registry),
    )
    lock = build_runtime_lock(
        fixture_id=args.fixture_id,
        pilot_lock_path=args.pilot_runtime_lock.resolve(),
        queue_path=args.queue.resolve(),
        queue_manifest_path=args.queue_manifest.resolve(),
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
        fixture_id=args.fixture_id,
        pilot_lane_spec_path=args.pilot_lane_spec.resolve(),
        pilot_seed_registry_path=args.pilot_seed_registry.resolve(),
        seed_registry_path=args.seed_registry_out.resolve(),
        runtime_root=args.runtime_root.rstrip("/"),
        output_parent=args.output_parent.rstrip("/"),
        lane_id=f"{lane_prefix}00",
        attempt_id=args.attempt_id,
        policy_port=args.policy_port,
        policy_gpu=args.policy_gpu,
        sim_gpu=args.sim_gpu,
    )
    write_exclusive(
        args.lane_spec_out.resolve(),
        json.dumps(lane_spec, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n",
    )
    launch_matrix = build_launch_matrix(
        fixture_id=args.fixture_id,
        lane_spec_path=args.lane_spec_out.resolve(),
        runtime_lock_path=args.runtime_lock_out.resolve(),
        hardware_g4_path=args.hardware_g4.resolve(),
        lane_count=args.lane_count,
        lane_id_prefix=lane_prefix,
    )
    write_exclusive(
        args.launch_matrix_out.resolve(),
        canonical_json_bytes(launch_matrix),
    )
    print(
        json.dumps(
            {
                "fixture_id": args.fixture_id,
                "family_id": profile.family_id,
                "confirmatory_episode_count": profile.confirmatory_episode_count,
                "runtime_lock": artifact(args.runtime_lock_out.resolve()),
                "launch_matrix": artifact(args.launch_matrix_out.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
