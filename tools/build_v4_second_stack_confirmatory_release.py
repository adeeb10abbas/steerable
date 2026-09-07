#!/usr/bin/env python3
"""Promote qualified C8 from pilot-only to confirmatory-family release."""

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

from tools.build_v4_second_stack_g7_pilot_release import (  # noqa: E402
    CHECKPOINT_REVISION,
    FIXTURE_ID,
    POLICY_ID,
    ROOT,
    artifact,
    canonical_json_bytes,
    load_json,
    require_passing,
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
    if payload.get("passed") is not True and payload.get("status") != "passed":
        raise ValueError(f"{gate} receipt is not passing")
    return payload


def extract_confirmatory_rows(queue_path: Path) -> list[dict]:
    rows = [
        row
        for row in load_jsonl(queue_path)
        if row.get("family") == "C8" and row.get("cohort") == "confirmatory"
    ]
    if len(rows) != 768:
        raise ValueError("confirmatory C8 allocation must contain 768 rows")
    return rows


def build_confirmatory_seed_registry(
    *,
    rows: list[dict],
    pilot_seed_registry_path: Path,
    queue_path: Path,
) -> dict:
    policy_seeds = list(dict.fromkeys(int(row["policy_seed"]) for row in rows))
    if len(policy_seeds) != 64:
        raise ValueError("confirmatory C8 allocation must contain 64 block seeds")
    pilot = load_json(pilot_seed_registry_path)
    pilot_seeds = {int(value) for value in pilot.get("allowed_sampling_seeds") or []}
    if set(policy_seeds) & pilot_seeds:
        raise ValueError("confirmatory C8 policy seeds collide with pilot seeds")
    return {
        "schema_version": "v4-groot-bridge-policy-seed-registry-v1",
        "campaign_id": "online_correction_v4",
        "fixture_id": FIXTURE_ID,
        "policy_id": POLICY_ID,
        "scope": "released_c8",
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
            "Allows exactly the 64 frozen C8 confirmatory block sampling seeds "
            "covering 768 rows. Engineering-pilot seeds remain excluded."
        ),
    }


def release_confirmatory_resets(
    *,
    candidate: dict,
    candidate_path: Path,
    main_g2_path: Path,
    main_g3_path: Path,
) -> dict:
    if candidate.get("fixture_id") != FIXTURE_ID:
        raise ValueError("confirmatory reset fixture mismatch")
    if candidate.get("qualification_scope") != "confirmatory":
        raise ValueError("reset registry is not the confirmatory allocation")
    if candidate.get("registered_env_seed_count") != 64:
        raise ValueError("confirmatory reset registry must contain 64 seeds")
    if candidate.get("status") != "model_blind_candidate_not_released_for_inference":
        raise ValueError("confirmatory reset registry is not a model-blind candidate")
    require_passing(load_json(main_g2_path))
    g3 = load_json(main_g3_path)
    require_passing(g3)
    if g3.get("selected_scale") != 0.5:
        raise ValueError("confirmatory C8 G3 selected scale is not 0.5")
    return {
        **candidate,
        "status": "released_for_policy_inference",
        "qualification_release_basis": {
            "candidate": artifact(candidate_path),
            "main_g2": artifact(main_g2_path),
            "main_g3": artifact(main_g3_path),
        },
        "release_boundary": (
            "Released only for the 768 frozen C8 confirmatory episodes. "
            "Engineering-pilot resets remain disjoint."
        ),
    }


def build_runtime_lock(
    *,
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
    runtime_image_digest: str,
) -> dict:
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source commit must be a full lowercase Git SHA")
    pilot_lock = load_json(pilot_lock_path)
    if (
        pilot_lock.get("release_status") != "PILOT_RELEASED"
        or pilot_lock.get("released_families") != ["C8"]
    ):
        raise ValueError("source lock is not the C8 pilot release")
    passing_receipt(g7_path, gate="G7")
    passing_receipt(g8_path, gate="G8")
    passing_receipt(hardware_g4_path, gate="G4")
    reset = load_json(main_reset_registry_path)
    if (
        reset.get("fixture_id") != FIXTURE_ID
        or reset.get("status") != "released_for_policy_inference"
        or reset.get("registered_env_seed_count") != 64
    ):
        raise ValueError("main C8 reset registry is not released over 64 seeds")
    g2 = load_json(main_g2_path)
    g3 = load_json(main_g3_path)
    if g2.get("passed") is not True or g3.get("passed") is not True:
        raise ValueError("main C8 G2/G3 qualification is not passing")
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
    lock["policies"][POLICY_ID]["runtime_image_digest"] = runtime_image_digest
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
            "family_ids": ["C8"],
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
        "RELEASED authorizes only frozen C8 confirmatory rows from the full "
        "17,664-row queue. Every other family remains blocked."
    )
    return lock


def build_launch_matrix(
    *,
    lane_spec_path: Path,
    runtime_lock_path: Path,
    hardware_g4_path: Path,
    lane_count: int,
) -> dict:
    if lane_count < 1 or lane_count > 40:
        raise ValueError("C8 confirmatory lane count must be between 1 and 40")
    lock = load_json(runtime_lock_path)
    if (
        lock.get("release_status") != "RELEASED"
        or lock.get("released_families") != ["C8"]
    ):
        raise ValueError("C8 confirmatory runtime lock is not released")
    passing_receipt(hardware_g4_path, gate="G4")
    spec = load_json(lane_spec_path)
    if spec.get("qualification_only") is not False:
        raise ValueError("C8 confirmatory lane spec is qualification-only")
    if spec.get("policy", {}).get("gpu_product") != "NVIDIA-A40":
        raise ValueError("C8 confirmatory lane spec must use A40 single-container lane")
    return {
        "schema_version": 1,
        "campaign_id": "online_correction_v4",
        "release_status": "RELEASED",
        "qualified_lanes": [
            {
                "lane_id": f"c8m{index:02d}",
                "hardware_stratum": "a40-single-container-groot-bridge",
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
            f"{lane_count} qualified A40 GR00T Bridge lanes for the 768 frozen "
            "C8 confirmatory rows only."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument(
        "--runtime-root",
        default="/data/users/ali/vla_wam/src/steerable-v4-c8-main",
    )
    parser.add_argument(
        "--runtime-image-digest",
        required=True,
        help="Immutable sha256:… container digest for the GR00T Bridge policy lane.",
    )
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--pilot-runtime-lock", type=Path, required=True)
    parser.add_argument("--pilot-seed-registry", type=Path, required=True)
    parser.add_argument("--candidate-reset-registry", type=Path, required=True)
    parser.add_argument("--main-g2", type=Path, required=True)
    parser.add_argument("--main-g3", type=Path, required=True)
    parser.add_argument("--hardware-g4", type=Path, required=True)
    parser.add_argument("--g7", type=Path, required=True)
    parser.add_argument("--g8", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--lane-count", type=int, default=8)
    parser.add_argument(
        "--output-parent",
        default="/data/users/ali/vla_wam/raw/v4/c8-second-stack-main",
    )
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--queue-manifest-out", type=Path, required=True)
    parser.add_argument("--seed-registry-out", type=Path, required=True)
    parser.add_argument("--released-reset-out", type=Path, required=True)
    parser.add_argument("--runtime-lock-out", type=Path, required=True)
    default_lane_spec = (
        ROOT / "deploy/k8s/v4_lane_bundle/c8-second-stack-confirmatory-spec.json"
    )
    parser.add_argument("--lane-spec-out", type=Path, default=default_lane_spec)
    parser.add_argument("--launch-matrix-out", type=Path, required=True)
    args = parser.parse_args()
    rows = extract_confirmatory_rows(args.queue.resolve())
    manifest_body = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    write_exclusive(args.manifest_out.resolve(), manifest_body)
    manifest_sha256 = sha256_file(args.manifest_out.resolve())
    write_exclusive(
        args.queue_manifest_out.resolve(),
        canonical_json_bytes(
            {
                "schema_version": 1,
                "campaign_id": "online_correction_v4",
                "release_status": "RELEASED",
                "queue_path": str(args.manifest_out.resolve().relative_to(ROOT)),
                "row_count": 768,
                "expected_confirmatory_episodes": 768,
                "queue_sha256": manifest_sha256,
                "frozen_queue_sha256": manifest_sha256,
                "planning_manifest_sha256": manifest_sha256,
                "release_boundary": (
                    "Frozen C8 confirmatory queue only; engineering-pilot rows excluded."
                ),
            }
        ),
    )
    seed_registry = build_confirmatory_seed_registry(
        rows=rows,
        pilot_seed_registry_path=args.pilot_seed_registry.resolve(),
        queue_path=args.manifest_out.resolve(),
    )
    write_exclusive(
        args.seed_registry_out.resolve(),
        canonical_json_bytes(seed_registry),
    )
    released_resets = release_confirmatory_resets(
        candidate=load_json(args.candidate_reset_registry),
        candidate_path=args.candidate_reset_registry.resolve(),
        main_g2_path=args.main_g2.resolve(),
        main_g3_path=args.main_g3.resolve(),
    )
    write_exclusive(
        args.released_reset_out.resolve(),
        canonical_json_bytes(released_resets),
    )
    lock = build_runtime_lock(
        pilot_lock_path=args.pilot_runtime_lock.resolve(),
        queue_path=args.manifest_out.resolve(),
        queue_manifest_path=args.queue_manifest_out.resolve(),
        seed_registry_path=args.seed_registry_out.resolve(),
        main_reset_registry_path=args.released_reset_out.resolve(),
        main_g2_path=args.main_g2.resolve(),
        main_g3_path=args.main_g3.resolve(),
        hardware_g4_path=args.hardware_g4.resolve(),
        g7_path=args.g7.resolve(),
        g8_path=args.g8.resolve(),
        analysis_manifest_path=args.analysis_manifest.resolve(),
        source_commit=args.source_commit,
        runtime_root=args.runtime_root.rstrip("/"),
        runtime_image_digest=args.runtime_image_digest,
    )
    write_exclusive(
        args.runtime_lock_out.resolve(),
        canonical_json_bytes(lock),
    )
    runtime_root = args.runtime_root.rstrip("/")
    python_bin = (
        "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89/"
        "gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python"
    )
    simpler_env_root = (
        "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89/"
        "external_dependencies/SimplerEnv"
    )
    runtime_common = {
        "python_bin": python_bin,
        "ffmpeg_bin": "/data/users/ali/vla_wam/envs/lingbot-va-b200/bin/ffmpeg",
        "vk_icd_filenames": (
            "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89/"
            "external_dependencies/SimplerEnv/nvidia_icd.json"
        ),
        "ld_library_path": (
            "/data/users/ali/vla_wam/envs/groot-render-libs/lib:/usr/lib/x86_64-linux-gnu"
        ),
        "pythonpath": f"{runtime_root}:{simpler_env_root}",
    }
    lane_spec = {
        "schema_version": "vla-wam-v4-k8s-lane-render-spec-v1",
        "qualification_only": False,
        "kube_context": "prod-dcwi-warrenq1-vmkub007",
        "namespace": "211247-prod",
        "lane_id": "c8m00",
        "attempt_id": "c8mainrelease20260908a",
        "policy_port": 18200,
        "policy_wait_timeout_seconds": 2400,
        "expected_driver_version": "580.95.05",
        "image_repository": "artifactory-ci.gm.com/docker-approved/devcontainers/base",
        "image_sha256": args.runtime_image_digest.removeprefix("sha256:"),
        "image_pull_secret": "artifactory-ci-pull-secret",
        "pvc": "211247-prod-pvc",
        "output_parent": args.output_parent.rstrip("/"),
        "entrypoint": "/opt/v4-lane/scripts/lane_entrypoint.py",
        "prestop_wait_seconds": 120,
        "runtime": {
            "policy": dict(runtime_common),
            "simulator": dict(runtime_common),
        },
        "policy": {
            "gpu_product": "NVIDIA-A40",
            "expected_gpu_name": "NVIDIA A40",
            "experiment_argv": [
                python_bin,
                f"{runtime_root}/tools/run_v4_groot_bridge_policy_server.py",
                "--host",
                "0.0.0.0",
                "--port",
                "18200",
                "--checkpoint-path",
                "/data/users/ali/vla_wam/checkpoints/groot-n1.7-simplerenv-bridge",
                "--integration-root",
                "/data/users/ali/vla_wam/external/Isaac-GR00T-51d4c89",
            ],
            "checkpoint_path": (
                "/data/users/ali/vla_wam/checkpoints/groot-n1.7-simplerenv-bridge"
            ),
            "checkpoint_sha256": "ba6ebd8503df6950d18e416a9d4d1b945b02493af7fea5406bfe7f889acc9c4f",
            "checkpoint_registry_path": (
                f"{runtime_root}/artifacts/online_correction_v4/setup/"
                "second_stack_g4_checkpoint_registry.candidate.json"
            ),
            "nvidia_smi_bin": "/usr/bin/nvidia-smi",
            "python_imports": ["torch", "simpler_env"],
            "file_bindings": [
                {
                    "source": "scripts/lane_entrypoint.py",
                    "path": "/opt/v4-lane/scripts/lane_entrypoint.py",
                },
                {
                    "source": "scripts/startup_preflight.py",
                    "path": "/opt/v4-lane/scripts/startup_preflight.py",
                },
                {
                    "source": "scripts/check_policy_ready.py",
                    "path": "/opt/v4-lane/scripts/check_policy_ready.py",
                },
                {
                    "source": "../../../tools/run_v4_groot_bridge_policy_server.py",
                    "path": f"{runtime_root}/tools/run_v4_groot_bridge_policy_server.py",
                },
                {
                    "source": (
                        "../../../artifacts/online_correction_v4/setup/"
                        "second_stack_g4_checkpoint_registry.candidate.json"
                    ),
                    "path": (
                        f"{runtime_root}/artifacts/online_correction_v4/setup/"
                        "second_stack_g4_checkpoint_registry.candidate.json"
                    ),
                },
            ],
            "vulkan_contract": {"required": False},
            "readiness_interface": "groot_bridge_http",
        },
        "simulator": {
            "gpu_product": "NVIDIA-A40",
            "expected_gpu_name": "NVIDIA A40",
            "experiment_argv": [],
            "checkpoint_path": "",
            "checkpoint_sha256": "0" * 64,
            "nvidia_smi_bin": "/usr/bin/nvidia-smi",
            "python_imports": ["simpler_env"],
            "file_bindings": [
                {
                    "source": "scripts/lane_entrypoint.py",
                    "path": "/opt/v4-lane/scripts/lane_entrypoint.py",
                },
                {
                    "source": "scripts/startup_preflight.py",
                    "path": "/opt/v4-lane/scripts/startup_preflight.py",
                },
                {
                    "source": "scripts/run_online_correction_v4_lane_dispatch.py",
                    "path": (
                        f"{runtime_root}/deploy/k8s/v4_lane_bundle/scripts/"
                        "run_online_correction_v4_lane_dispatch.py"
                    ),
                },
            ],
            "vulkan_contract": {"required": True},
            "readiness_interface": "embedded_in_policy_container",
        },
    }
    write_exclusive(
        args.lane_spec_out.resolve(),
        json.dumps(lane_spec, allow_nan=False, indent=2, sort_keys=True).encode(
            "utf-8"
        )
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
                "confirmatory_episode_count": len(rows),
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
