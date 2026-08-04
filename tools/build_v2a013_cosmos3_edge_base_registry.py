#!/usr/bin/env python3
"""Build the V2-A013 Cosmos3-Edge base metadata registry without model download or inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


MODEL_ID = "nvidia/Cosmos3-Edge"
REVISION = "ff48d22144de52de296a7b4d3a78914831007212"
EXPECTED_FILES = 48
EXPECTED_BYTES = 9_173_855_122
EXPECTED_LFS_BYTES = 9_173_276_024
EXPECTED_PARAMETER_COUNT = 3_858_999_728
AMENDMENT = Path("artifacts/vla_wam_shared_v2/pilot/post_result_cosmos3_edge_base_amendment.json")
PROTOCOL = Path("artifacts/vla_wam_shared_v2/protocol.json")
EDGE_POLICY_RESULT = Path("artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_droid_direct_gate.json")
RUNBOOK = Path("experiments/cosmos/COSMOS3_EDGE_BASE_V2A013.md")
BUILDER = Path("tools/build_v2a013_cosmos3_edge_base_registry.py")

REMOTE_SOURCES = {
    "cosmos_framework_droid_dataset": {
        "url": "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/a904d2d36b774a51dd06ff9ff906816b1a04f579/cosmos_framework/data/generator/action/datasets/droid_lerobot_dataset.py",
        "sha256": "418fc30908ff485a0f1f23cf021339247b58c90b40790f0e53f21833c65ba5bb",
    },
    "cosmos_framework_pose_utils": {
        "url": "https://raw.githubusercontent.com/NVIDIA/cosmos-framework/a904d2d36b774a51dd06ff9ff906816b1a04f579/cosmos_framework/data/generator/action/pose_utils.py",
        "sha256": "c3039f9c5b7c13fc65ffbef67331ae2261f941686065073ef5eac28a6dac31d0",
    },
    "vllm_omni_edge_recipe": {
        "url": "https://raw.githubusercontent.com/vllm-project/vllm-omni/900a7f0813d0482811b0e4dfd3cf7deabbe2429f/recipes/cosmos3/Cosmos3-Edge.md",
        "sha256": "9f5ce0484f94285ea2673f54e15314181d6cbb5534acfe95b24fbfd7f05b5ba8",
    },
    "vllm_omni_action_recipe": {
        "url": "https://raw.githubusercontent.com/vllm-project/vllm-omni/900a7f0813d0482811b0e4dfd3cf7deabbe2429f/recipes/cosmos3/Cosmos3-Nano.md",
        "sha256": "b80c81a92267cdf3162c170635a2b55c447a48dd97dbb525d409b4f1563ee979",
    },
    "vllm_omni_pipeline": {
        "url": "https://raw.githubusercontent.com/vllm-project/vllm-omni/900a7f0813d0482811b0e4dfd3cf7deabbe2429f/vllm_omni/diffusion/models/cosmos3/pipeline_cosmos3.py",
        "sha256": "d6cff51acc9c56a84c56274ddfd1e2d9580f26c2e72e0d5a9b3f813cc232b2c0",
    },
    "vllm_omni_robolab_utils": {
        "url": "https://raw.githubusercontent.com/vllm-project/vllm-omni/900a7f0813d0482811b0e4dfd3cf7deabbe2429f/vllm_omni/diffusion/models/cosmos3/utils.py",
        "sha256": "cc5e7f7bb9eb1c41754c49ce1399aa5439a78c4c7a2ac1f5bd1739c9c49fef87",
    },
    "robolab_droid_robot": {
        "url": "https://raw.githubusercontent.com/NVlabs/RoboLab/0aef241fb088ca21bb4ebd24448940ed56620d17/robolab/robots/droid.py",
        "sha256": "3c43b562cc22476135b7cd82c9c4c01ed361350eda25f3ef6669408ffce53e5d",
    },
    "curobo_franka_urdf": {
        "url": "https://raw.githubusercontent.com/NVlabs/curobo/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/src/curobo/content/assets/robot/franka_description/franka_panda.urdf",
        "sha256": "6a0044e6e72ee667927f17d1871ec3e2615a8bc5fe978882fc909e4094667967",
    },
    "curobo_franka_config": {
        "url": "https://raw.githubusercontent.com/NVlabs/curobo/d64c4b005459db10c5dd867d8b30a87d5bda9bdb/src/curobo/content/configs/robot/franka.yml",
        "sha256": "c809e9c1d044c8cf98d888552fb55739077419fb3568d5e696242cdecb457430",
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "steerable-v2a013-freeze"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(), {key.lower(): value for key, value in response.headers.items()}


def fetch_json(url: str) -> tuple[Any, dict[str, str]]:
    payload, headers = fetch(url)
    return json.loads(payload), headers


def tree_rows() -> list[dict[str, Any]]:
    url = f"https://huggingface.co/api/models/{MODEL_ID}/tree/{REVISION}?recursive=true&expand=true"
    rows: list[dict[str, Any]] = []
    while url:
        page, headers = fetch_json(url)
        if not isinstance(page, list):
            raise RuntimeError("Hugging Face tree endpoint did not return a list")
        rows.extend(page)
        match = re.search(r'<([^>]+)>; rel="next"', headers.get("link", ""))
        url = match.group(1) if match else ""
    return rows


def local_record(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    return {"path": str(relative), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def build_cells() -> list[dict[str, Any]]:
    prompts = {
        "left": "Put the Rubik's cube to the left of the bowl.",
        "right": "Put the Rubik's cube to the right of the bowl.",
    }
    return [
        {
            "cell_id": f"cosmos3_edge_base_seed{seed}_{relation}",
            "environment_seed": seed,
            "sampling_seed": seed,
            "requested_relation": relation,
            "rendered_prompt": prompts[relation],
            "prompt_controller": "episode_static",
            "oracle_or_subtask_coach": False,
            "dynamic_prompt_switches": 0,
            "control_interface": "derived_control_curobo_ik_only_after_release",
            "viewport_video_required": True,
            "executed_action_trace_required": True,
            "model_action_trace_required": True,
            "exposed_generated_future_required": True,
            "status": "frozen_unreleased",
        }
        for seed in (8300, 8301, 8302)
        for relation in ("left", "right")
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/vla_wam_shared_v2/pilot/expansion/cosmos3_edge_base_v2a013_registry.json"),
    )
    args = parser.parse_args()
    root = args.workspace.resolve()

    amendment = json.loads((root / AMENDMENT).read_text())
    if amendment.get("amendment_id") != "V2-A013" or amendment.get("experiment_identity", {}).get("model_revision") != REVISION:
        raise RuntimeError("V2-A013 amendment identity mismatch")

    model_info, _ = fetch_json(f"https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}")
    if model_info.get("sha") != REVISION or model_info.get("private") is not False or model_info.get("gated") is not False:
        raise RuntimeError("Cosmos3-Edge did not resolve to the frozen public ungated revision")
    parameter_count = sum(model_info.get("safetensors", {}).get("parameters", {}).values())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(f"Unexpected safetensors parameter count: {parameter_count}")

    files = sorted((row for row in tree_rows() if row.get("type") == "file"), key=lambda row: row["path"])
    if len(files) != EXPECTED_FILES or len({row["path"] for row in files}) != EXPECTED_FILES:
        raise RuntimeError(f"Unexpected snapshot layout: {len(files)} files")
    snapshot_total = sum(int(row["size"]) for row in files)
    lfs_total = sum(int(row["size"]) for row in files if row.get("lfs"))
    if snapshot_total != EXPECTED_BYTES or lfs_total != EXPECTED_LFS_BYTES:
        raise RuntimeError(f"Unexpected snapshot bytes: total={snapshot_total} lfs={lfs_total}")
    snapshot_files = {
        row["path"]: {
            "bytes": int(row["size"]),
            "git_blob_oid": row["oid"],
            "lfs_sha256": row.get("lfs", {}).get("oid"),
        }
        for row in files
    }

    metadata_sha256: dict[str, str] = {}
    metadata_payloads: dict[str, bytes] = {}
    for relative in ("README.md", "config.json", "model.safetensors.index.json", "transformer/config.json"):
        payload, _ = fetch(f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{relative}?download=true")
        metadata_payloads[relative] = payload
        metadata_sha256[relative] = sha256_bytes(payload)
    transformer_config = json.loads(metadata_payloads["transformer/config.json"])
    if transformer_config.get("action_gen") is not True or transformer_config.get("action_dim") != 64:
        raise RuntimeError("Frozen Edge transformer no longer exposes the expected action head")

    remote_records: dict[str, dict[str, Any]] = {}
    remote_payloads: dict[str, bytes] = {}
    for source_id, spec in REMOTE_SOURCES.items():
        payload, _ = fetch(spec["url"])
        actual = sha256_bytes(payload)
        if actual != spec["sha256"]:
            raise RuntimeError(f"Official source hash changed for {source_id}: {actual}")
        remote_payloads[source_id] = payload
        remote_records[source_id] = {"url": spec["url"], "bytes": len(payload), "sha256": actual}
    required_anchors = {
        "cosmos_framework_droid_dataset": [
            b"10D ``[Pos, Rot6d, Gripper]``",
            b'pose_convention: PoseConvention = "backward_framewise"',
        ],
        "vllm_omni_edge_recipe": [b"nvidia/Cosmos3-Edge-Policy-DROID", b'`policy`'],
        "vllm_omni_action_recipe": [b"and a rollout video", b'"raw_action_dim":10'],
        "vllm_omni_pipeline": [b'"action_only_output": True', b"raw_action_dim"],
        "vllm_omni_robolab_utils": [b"ROBOLAB_MIDTRAIN_RAW_ACTION_DIM = 10", b'pose_convention="backward_framewise"'],
        "robolab_droid_robot": [b"franka_robotiq_2f_85_flattened.usd", b'joint_names=["panda_joint.*"]'],
        "curobo_franka_config": [b'ee_link: "panda_hand"', b"collision_link_names"],
    }
    for source_id, anchors in required_anchors.items():
        for anchor in anchors:
            if anchor not in remote_payloads[source_id]:
                raise RuntimeError(f"Missing frozen interface anchor {anchor!r} in {source_id}")

    edge_result = json.loads((root / EDGE_POLICY_RESULT).read_text())
    if edge_result.get("status") != "complete" or edge_result.get("summary", {}).get("episode_count") != 6:
        raise RuntimeError("Completed Edge-Policy-DROID evidence is not intact")

    registry = {
        "schema_version": "vla-wam-shared-v2-cosmos3-edge-base-v2a013-registry-v1",
        "amendment_id": "V2-A013",
        "status": "pre_inference_probe_frozen_behavioral_cells_unreleased",
        "amendment": local_record(root, AMENDMENT),
        "protocol": local_record(root, PROTOCOL),
        "checkpoint": {
            "id": MODEL_ID,
            "revision": REVISION,
            "public": True,
            "gated": False,
            "safetensors_parameter_count": parameter_count,
            "snapshot_file_count": len(snapshot_files),
            "snapshot_total_bytes": snapshot_total,
            "snapshot_lfs_payload_bytes": lfs_total,
            "metadata_sha256": metadata_sha256,
            "files": snapshot_files,
            "download_status": "not_started",
            "model_load_attempt_count": 0,
            "model_action_request_count": 0,
            "behavioral_episode_count": 0,
            "hash_gate_passed": False,
            "local_path": "/data/users/ali/vla_wam/checkpoints/cosmos3_edge_base_ff48d221",
        },
        "software": {
            "nvidia_cosmos_commit": "e494d734022ab0610061cdf57fa24c843e18767e",
            "nvidia_cosmos_framework_commit": "a904d2d36b774a51dd06ff9ff906816b1a04f579",
            "vllm_omni_commit": "900a7f0813d0482811b0e4dfd3cf7deabbe2429f",
            "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
            "curobo_commit": "d64c4b005459db10c5dd867d8b30a87d5bda9bdb",
            "curobo_version": "0.7.8",
            "official_source_records": remote_records,
            "runtime_status": "pinned_not_restored_or_import_verified_by_this_freeze",
        },
        "preserved_completed_edge_policy": {
            "checkpoint": "nvidia/Cosmos3-Edge-Policy-DROID",
            "result": local_record(root, EDGE_POLICY_RESULT),
            "valid_behavioral_episode_count": 6,
            "native_action_interface": "32x8 joint_pos",
            "do_not_rerun": True,
            "pool_with_edge_base": False,
        },
        "interface": amendment["frozen_interface_determination"],
        "fixed_observation_probe": {
            **amendment["fixed_observation_feasibility_probe"],
            "status": "frozen_not_run",
            "released_request_count": 0,
            "raw_output_root": "/data/users/ali/vla_wam/raw/cosmos3_edge_base/v2_a013/fixed_observation",
            "failure_policy": "Any repeat, sensitivity, action-layout, or exposed-future failure closes the probe without releasing behavior. Missing future is unavailable evidence, not zero.",
        },
        "behavioral_queue": {
            **amendment["conditional_behavioral_grid"],
            "status": "six_cells_frozen_zero_released",
            "cells": build_cells(),
        },
        "derived_control_curobo": amendment["derived_control_curobo_contract"],
        "release_gates": [
            "Verify every downloaded checkpoint file against this registry's revision, byte count, Git blob/LFS record, and local SHA-256 before model load.",
            "Restore and record the exact pinned source commits in isolated environments; pass import and action-interface smoke checks before the fixed-observation probe.",
            "Run exactly LEFT, LEFT exact repeat, RIGHT on the frozen RGB bytes through the generic async policy endpoint. Require [16,10] actions and decodable 17-frame future video for all three.",
            "Require bit-identical LEFT repeats and non-identical LEFT/RIGHT action and future outputs. A failure preserves the three requests as diagnostic evidence and releases zero behavior.",
            "Before behavior, replace neither RoboLab's Robotiq asset nor its base_link frame with CuRobo's Panda hand. Hash and verify an exact Franka+Robotiq URDF/collision model and transform equivalence to the frozen RoboLab USD.",
            "Before behavior, attest source position units as meters, exact 10D semantics, robot base/world frames, quaternion ordering, joint order/limits, gripper mapping, solver/collision configuration, and deterministic seeds.",
            "No rejected IK, collision, limit, stale-state, unit, frame, asset, or hash result sends an action; retain a controller-rejection ledger outside denominators.",
            "Every released behavioral cell retains actual viewport video, executed 8D derived actions, raw 10D model chunks, decoded imagined futures, simulator state, solver records, and all hashes.",
            "Report derived control separately from native policies. Preserve valid failures; exclude infrastructure failures and partial runs from behavioral denominators.",
        ],
        "evidence_paths": {
            "raw_root": "/data/users/ali/vla_wam/raw/cosmos3_edge_base/v2_a013",
            "invalid_attempts": "/data/users/ali/vla_wam/raw/cosmos3_edge_base/v2_a013/invalid_attempts.json",
            "controller_rejections": "/data/users/ali/vla_wam/raw/cosmos3_edge_base/v2_a013/controller_rejections.json",
            "runtime_interventions": "/data/users/ali/vla_wam/raw/cosmos3_edge_base/v2_a013/runtime_interventions.json",
        },
        "local_sources": [local_record(root, BUILDER), local_record(root, RUNBOOK)],
        "claim_boundary": amendment["claim_boundary"],
    }

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "files": len(snapshot_files), "bytes": snapshot_total, "cells": 6}, sort_keys=True))


if __name__ == "__main__":
    main()
