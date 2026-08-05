#!/usr/bin/env python3
"""Fail-closed V3-A002 π0-FAST old-name-config bridge behavioral adapter.

This module performs no inference.  It derives the 40 bridge cells from the
blocked Phase-A rows, validates the post-result amendment, exact sources,
checkpoint, live runtime identity, and three-request release gate, and compiles
retained simulator evidence into the shared v3 behavioral/infrastructure
schemas.  The bridge is always reported separately from historical π0-FAST.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "pi0_fast_old_name_config_v3a002"
SOURCE_MODEL_ID = "pi0_fast_droid_vla"
STUDY_ID = "vla_wam_language_steerability_v3"
PHASE = "A_direct_command_matched_pairs"
QUEUE_SCHEMA = "vla-wam-shared-v3-phase-a-cells-v1"
QUEUE_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")
QUEUE_MANIFEST_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells_manifest.json")
AMENDMENT_RELATIVE = Path(
    "artifacts/vla_wam_shared_v3/"
    "post_result_pi0_fast_old_name_config_amendment.json"
)
CHECKPOINT_MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "pi0_fast_wording_readiness.json"
)
AMENDMENT_SCHEMA = (
    "vla-wam-shared-v3-post-result-pi0-fast-old-name-config-amendment-v1"
)
RUNTIME_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-runtime-identity-v1"
GATE_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-gate-v1"
CAPTURE_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-state-capture-v1"
INFRA_CAPTURE_SCHEMA = "vla-wam-shared-v3-pi0-fast-old-name-config-infrastructure-capture-v1"
ACTION_TRACE_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-action-trace-v1"
)

PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
FROZEN_OPENPI_COMMIT = "235044ed8a1502c0a18338eedc5d7adfe705af05"
FROZEN_OPENPI_TREE = "03a4387bedbc0fa1467c367c60fc24e28b61ec6c"
FROZEN_OPENPI_PARENT = "a4808042a2cd75964167c4a747d2aabdf5fb9133"
FROZEN_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FROZEN_CONFIG = "pi0_fast_droid_jointpos"
FROZEN_CONFIG_SOURCE_SHA256 = (
    "96ddf85ff5903e68acca310d7af9d9d093373f7dc060fe94dbb379c6828481ad"
)
FROZEN_UV_LOCK_SHA256 = (
    "5e3a9a0a12d9a6048afea5591f4520c98585499cbd4a8343dcabfe2aaed94e3d"
)
FROZEN_CHECKPOINT_MANIFEST_SHA256 = (
    "47b38eb2f17be802c126ef0a7e93b16693823ee2df62b8007f51bb0514baf5c5"
)
FROZEN_QUEUE_SHA256 = (
    "8350b98f958424b56b66e67e8c70ec3951d27f4ae257476d6f08c0aaa873cb7c"
)
FROZEN_FIXTURE_SHA256 = (
    "ce8be012347718a162bf0d92ba2fb71a01c570a3462d72ef2c16a86082131778"
)
OPEN_LOOP_HORIZON = 10
ACTION_SHAPE = (10, 8)
ACTION_CAP = 450
FROZEN_CHECKPOINT = {
    "id": FROZEN_CONFIG,
    "revision": f"v3a002-manifest-{FROZEN_CHECKPOINT_MANIFEST_SHA256}",
}
FROZEN_REPOSITORY_SOURCES = {
    "experiments/v3/pi0_fast_old_name_config_bridge/serve_policy.py":
        "45d688adfb689350d19051782c541f98b34554095f6ea0019d83d02b8e669068",
    "experiments/v3/pi0_fast_old_name_config_bridge/fixed_observation_gate.py":
        "dc0ca2afab6ae93b11c649fdd4a7e1226e21be507c1cba5f2ab85d0a4ab7e14d",
    "experiments/v3/pi0_fast_old_name_config_bridge/model_blind_writer_gate.py":
        "7a4f7ff32c0a4e0e1672d48bc31ad6b5d539e4e89de10f45b812ee012a5a6d41",
    "experiments/v3/pi0_fast_old_name_config_bridge/model_blind_renderer_gate.py":
        "72ff267ef607ef3a58f730bef4a8702d807b3a029f321acbd0b8939a0eb0a678",
    "experiments/v3/pi0_fast_old_name_config_bridge/run_shard.py":
        "86f2b8bfe0cbd76717fe15c1b53e353af4dcc076c5adeb8ee199ec73c1114a19",
    "experiments/pi0_current_stack/v2a008_robolab_client.py":
        "60e70abde3642d58fdaa9101d633c62bdf33f07bd22e9683edd4f987c022c745",
    "experiments/pi0_current_stack/v2a008_robolab_gate.py":
        "4425bc10f381b44481df61c9cecc133fdfb4c0a6889ed7754b6b1706fe6e8150",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py":
        "9c4d90be770266bac3ba5242b743098348c565ee622179b5e88fa2af0c4891bc",
    "experiments/groot_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py":
        "ba0eb879590960c57976dd1b749c4ebbd3e86054e152ca5af014ac1bc2b6d02a",
}
ADAPTER_SOURCES = (
    "experiments/v3/pi0_fast_old_name_config_bridge/adapter.py",
    "experiments/v3/pi0_fast_old_name_config_bridge/client.py",
    "experiments/v3/pi0_fast_old_name_config_bridge/robolab_bridge.py",
)
HEX64 = set("0123456789abcdef")
POLICY_RUNTIME = {
    "pod": "lerobot-b200-4gpu-1-ali",
    "pod_uid": "1e0f438c-6041-4cc3-af32-0c118963e54c",
    "node": "dcwipphhgc225.edc.nam.gm.com",
    "pod_ip": "10.244.103.110",
    "container_image": "artifactory-ci.gm.com/docker-approved/devcontainers/base:ubuntu22.04",
    "gpu_index": 2,
    "gpu_uuid": "GPU-4ca76921-a7d2-e920-8555-47e0e8f105f7",
    "gpu_model": "NVIDIA B200",
    "driver": "580.95.05",
    "cuda_visible_devices": "2",
    "endpoint": "10.244.103.110:8011",
}
SIMULATOR_RUNTIMES = {
    "raytrace-rtxpro6000-ali": {
        "pod_uid": "d5b0405a-a9b1-4baa-a802-d5171e03c228",
        "pod_ip": "10.244.222.28",
        "gpu_uuid": "GPU-f28bd513-a38a-b768-7589-d2959f814ae8",
        "seed_range": [8310, 8316],
    },
    "vla-wam-rtx-cosmos-ali": {
        "pod_uid": "b7ec6369-2ab7-42d6-a4ef-3fc549ca652a",
        "pod_ip": "10.244.222.15",
        "gpu_uuid": "GPU-c2407ba3-cc0f-2456-5df4-8c968bfb8435",
        "seed_range": [8317, 8323],
    },
    "vla-wam-rtx-nano-ali": {
        "pod_uid": "30b51054-f277-480d-831d-337581e1cc49",
        "pod_ip": "10.244.222.14",
        "gpu_uuid": "GPU-37425604-6c72-a77a-f635-392439431707",
        "seed_range": [8324, 8329],
    },
}
SIMULATOR_SHARED = {
    "node": "dcwipphrtx0005.edc.nam.gm.com",
    "container_image": "artifactory-ci.gm.com/docker-approved/devcontainers/base:ubuntu22.04",
    "gpu_index": 0,
    "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
    "driver": "580.105.08",
}
FROZEN_PYTHONPATH = [
    "/data/users/ali/vla_wam/src/steerable",
    "/data/users/ali/vla_wam/external/RoboLab-pi0fast-bridge-0aef241-clean01",
    "/data/users/ali/vla_wam/external/openpi-235044ed/packages/openpi-client/src",
]
FROZEN_IMPORTS = {
    "openpi_client": (
        "/data/users/ali/vla_wam/external/openpi-235044ed/packages/openpi-client/"
        "src/openpi_client/__init__.py",
        "91447944015cec709e8aa7655f7e9d64e1e4508e7023a57fe3746911c0fc6fed",
    ),
    "openpi_client.websocket_client_policy": (
        "/data/users/ali/vla_wam/external/openpi-235044ed/packages/openpi-client/"
        "src/openpi_client/websocket_client_policy.py",
        "36557cb0b91ccf31cd4fb4b508306850d76ed0feb4028dac5182d0f5a5d88005",
    ),
    "openpi_client.msgpack_numpy": (
        "/data/users/ali/vla_wam/external/openpi-235044ed/packages/openpi-client/"
        "src/openpi_client/msgpack_numpy.py",
        "c04568948fcee52b691e3be4b6cffb759f7e79ad67530fcd5d23095a0d13c057",
    ),
    "robolab": (
        "/data/users/ali/vla_wam/external/RoboLab-pi0fast-bridge-0aef241-clean01/"
        "robolab/__init__.py",
        "d7912be543fa14354464740c9dd280e3530b38acbee28379b34a0ff825801d84",
    ),
    "policies.pi0_family.client": (
        "/data/users/ali/vla_wam/external/RoboLab-pi0fast-bridge-0aef241-clean01/"
        "policies/pi0_family/client.py",
        "2386e6230ca4e2bbf163159ff0692780f5027a6b158b510b206700d54a4a29a3",
    ),
}
WRITER_GATE_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-model-blind-writer-gate-manifest-v1"
)
RENDERER_GATE_SCHEMA = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-model-blind-renderer-gate-v1"
)


class AdapterError(ValueError):
    """The queue, runtime, release gate, or retained evidence is inconsistent."""


def _fail(message: str) -> None:
    raise AdapterError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX64


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{path} must contain one JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                _fail(f"blank line in queue at {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                _fail(f"queue row at {path}:{number} is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read queue {path}: {error}") from error
    return rows


def _observed_source_hashes(study_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_REPOSITORY_SOURCES.items():
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing frozen V3-A002 source: {path}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            _fail(f"frozen V3-A002 source hash mismatch: {relative}")
    return observed


def adapter_source_sha256(study_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ADAPTER_SOURCES:
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing v3 π0-FAST old-name-config bridge adapter source: {path}")
        result[relative] = sha256_file(path)
    return result


def checkpoint_contract_sha256(study_root: Path) -> str:
    path = study_root / CHECKPOINT_MANIFEST_RELATIVE
    if sha256_file(path) != FROZEN_CHECKPOINT_MANIFEST_SHA256:
        _fail("committed π0-FAST old-name-config bridge checkpoint manifest hash changed")
    manifest = _load_object(path)
    checkpoint = manifest.get("checkpoint", {})
    if (
        manifest.get("schema_version")
        != "vla-wam-shared-v2-pi0-fast-wording-readiness-v1"
        or manifest.get("status")
        != "blocked_before_model_load_and_behavioral_inference"
        or checkpoint.get("status") != "complete_and_sha256_hashed"
        or checkpoint.get("file_count") != 19
        or checkpoint.get("payload_bytes") != 10_844_314_410
        or len(checkpoint.get("files", [])) != 19
    ):
        _fail("committed π0-FAST checkpoint manifest contract changed")
    files = checkpoint["files"]
    if len({row.get("path") for row in files}) != 19:
        _fail("π0-FAST checkpoint manifest paths are not unique")
    norm = next(
        (row for row in files if row.get("path") == "assets/droid/norm_stats.json"),
        None,
    )
    if norm is None or norm.get("sha256") != (
        "b6ea29dd1cce48be4a57850e9118d7723145fb69d86168be8e86ac38fcac8c4f"
    ):
        _fail("frozen checkpoint-local DROID norm statistics changed")
    payload = [
        {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
        for row in files
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_amendment(study_root: Path) -> dict[str, Any]:
    """Validate the complete post-result V3-A002 authorization."""

    amendment = _load_object(study_root / AMENDMENT_RELATIVE)
    for key, wanted in {
        "schema_version": AMENDMENT_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": "V3-A002",
        "status": "frozen_before_v3a002_model_load_or_request",
    }.items():
        if amendment.get(key) != wanted:
            _fail(f"V3-A002 amendment mismatch for {key}")

    selection = amendment.get("candidate_selection", {})
    readiness_path = study_root / "artifacts/vla_wam_shared_v2/model_readiness.json"
    if (
        selection.get("artifact")
        != "artifacts/vla_wam_shared_v2/model_readiness.json"
        or selection.get("artifact_sha256")
        != "9013439f67a765b21358b17bdc4b6848a397c7f6c57bec4977cbb19f7013fae2"
        or not readiness_path.is_file()
        or sha256_file(readiness_path) != selection["artifact_sha256"]
    ):
        _fail("V3-A002 historical readiness selection evidence changed")

    bridge = amendment.get("bridge_identity", {})
    openpi = bridge.get("openpi", {})
    checkpoint = bridge.get("checkpoint", {})
    robolab = bridge.get("robolab", {})
    if bridge.get("model_id") != MODEL_ID:
        _fail("V3-A002 bridge model identity changed")
    for key, wanted in {
        "commit": FROZEN_OPENPI_COMMIT,
        "tree": FROZEN_OPENPI_TREE,
        "parent": FROZEN_OPENPI_PARENT,
        "config": FROZEN_CONFIG,
        "config_source_path": "src/openpi/training/misc/polaris_config.py",
        "config_source_sha256": FROZEN_CONFIG_SOURCE_SHA256,
        "action_shape": list(ACTION_SHAPE),
        "max_token_len": 250,
        "data_config": "SimpleDataConfig",
        "input_transform": "DroidInputs(ModelType.PI0_FAST)",
        "output_transforms": "AbsoluteActions(first_7_mask)+DroidOutputs",
        "uv_lock_sha256": FROZEN_UV_LOCK_SHA256,
    }.items():
        if openpi.get(key) != wanted:
            _fail(f"V3-A002 OpenPI amendment mismatch for {key}")
    for key, wanted in {
        "manifest_path": str(CHECKPOINT_MANIFEST_RELATIVE),
        "manifest_sha256": FROZEN_CHECKPOINT_MANIFEST_SHA256,
        "file_count": 19,
        "payload_bytes": 10_844_314_410,
    }.items():
        if checkpoint.get(key) != wanted:
            _fail(f"V3-A002 checkpoint amendment mismatch for {key}")
    for key, wanted in {
        "commit": FROZEN_ROBOLAB_COMMIT,
        "left_task_source_sha256": FROZEN_REPOSITORY_SOURCES[
            "experiments/groot_droid/robolab_v2_tasks/"
            "rubiks_cube_left_of_bowl_matched.py"
        ],
        "right_task_source_sha256": FROZEN_REPOSITORY_SOURCES[
            "experiments/groot_droid/robolab_v2_tasks/"
            "rubiks_cube_right_of_bowl_matched.py"
        ],
    }.items():
        if robolab.get(key) != wanted:
            _fail(f"V3-A002 RoboLab amendment mismatch for {key}")

    implementation = amendment.get("implementation", {})
    if (
        implementation.get("server_path")
        != "experiments/v3/pi0_fast_old_name_config_bridge/serve_policy.py"
        or implementation.get("server_sha256")
        != FROZEN_REPOSITORY_SOURCES[
            "experiments/v3/pi0_fast_old_name_config_bridge/serve_policy.py"
        ]
        or implementation.get("gate_path")
        != "experiments/v3/pi0_fast_old_name_config_bridge/fixed_observation_gate.py"
        or implementation.get("gate_sha256")
        != FROZEN_REPOSITORY_SOURCES[
            "experiments/v3/pi0_fast_old_name_config_bridge/"
            "fixed_observation_gate.py"
        ]
    ):
        _fail("V3-A002 implementation identity changed")

    gate = amendment.get("three_request_gate", {})
    if (
        gate.get("fixture_sha256") != FROZEN_FIXTURE_SHA256
        or gate.get("sampling_seed") != 8_310_000
        or gate.get("order") != ["left", "left_exact_repeat", "right"]
        or gate.get("prompts") != PROMPTS
    ):
        _fail("V3-A002 three-request gate contract changed")

    release = amendment.get("behavioral_release_if_gate_passes", {})
    source_filter = release.get("source_filter", {})
    if source_filter != {
        "model_id": SOURCE_MODEL_ID,
        "status": "blocked_pi0",
        "execution_status": (
            "blocked_pending_exact_historical_openpi_and_robolab_recovery"
        ),
        "phase": PHASE,
        "prompt_family": "direct_command",
        "environment_seeds_inclusive": [8310, 8329],
    }:
        _fail("V3-A002 source queue filter changed")
    for key, wanted in {
        "source_queue": str(QUEUE_RELATIVE),
        "source_queue_sha256": FROZEN_QUEUE_SHA256,
        "new_model_id": MODEL_ID,
        "matched_pairs": 20,
        "behavioral_cells": 40,
        "action_cap": ACTION_CAP,
        "open_loop_horizon": OPEN_LOOP_HORIZON,
        "success_predicate": (
            "v2_frozen_droid_robolab_release_inside_45deg_requested_relation"
        ),
    }.items():
        if release.get(key) != wanted:
            _fail(f"V3-A002 behavioral release mismatch for {key}")
    prohibited = amendment.get("reporting_boundary", {}).get("prohibited", [])
    if (
        "calling V3-A002 checkpoint-era code or recovered historical code"
        not in prohibited
        or "pooling DROID and RoboTwin" not in prohibited
    ):
        _fail("V3-A002 reporting boundary changed")

    _observed_source_hashes(study_root)
    checkpoint_contract_sha256(study_root)
    return amendment


def adapter_contract_sha256(study_root: Path) -> str:
    contract = {
        "model_id": MODEL_ID,
        "queue_sha256": sha256_file(study_root / QUEUE_RELATIVE),
        "amendment_sha256": sha256_file(study_root / AMENDMENT_RELATIVE),
        "checkpoint_contract_sha256": checkpoint_contract_sha256(study_root),
        "frozen_repository_sources": FROZEN_REPOSITORY_SOURCES,
        "adapter_sources": adapter_source_sha256(study_root),
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "openpi_tree": FROZEN_OPENPI_TREE,
        "openpi_parent": FROZEN_OPENPI_PARENT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "config": FROZEN_CONFIG,
        "max_token_len": 250,
        "uv_lock_sha256": FROZEN_UV_LOCK_SHA256,
        "open_loop_horizon": OPEN_LOOP_HORIZON,
        "action_cap": ACTION_CAP,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_authorized_pair(study_root: Path, seed: int) -> list[dict[str, Any]]:
    """Derive one amended bridge pair from two exact blocked queue rows."""

    if seed not in range(8310, 8330):
        _fail("V3-A002 bridge seeds are exactly 8310-8329")
    validate_amendment(study_root)
    queue_path = study_root / QUEUE_RELATIVE
    manifest = _load_object(study_root / QUEUE_MANIFEST_RELATIVE)
    queue_sha = sha256_file(queue_path)
    if queue_sha != FROZEN_QUEUE_SHA256 or (
        manifest.get("queue_file") != str(QUEUE_RELATIVE)
        or manifest.get("queue_sha256") != queue_sha
        or manifest.get("study_id") != STUDY_ID
    ):
        _fail("committed Phase-A queue/manifest identity mismatch")
    selected = [
        row for row in _load_jsonl(queue_path)
        if row.get("model_id") == SOURCE_MODEL_ID
        and row.get("environment_seed") == seed
    ]
    if len(selected) != 2:
        _fail(f"expected exactly two blocked π0-FAST rows for seed {seed}")
    by_relation = {row.get("relation"): row for row in selected}
    if set(by_relation) != {"left", "right"}:
        _fail("source pair must contain one LEFT and one RIGHT row")
    for relation, row in by_relation.items():
        expected = {
            "schema_version": QUEUE_SCHEMA,
            "study_id": STUDY_ID,
            "arena": "droid_robolab",
            "phase": PHASE,
            "model_id": SOURCE_MODEL_ID,
            "cell_id": f"v3:droid:{SOURCE_MODEL_ID}:seed{seed}:{relation}",
            "pair_id": f"v3:droid:{SOURCE_MODEL_ID}:seed{seed}",
            "environment_seed": seed,
            "sampling_seed": seed,
            "replicate": 0,
            "status": "blocked_pi0",
            "execution_status": (
                "blocked_pending_exact_historical_openpi_and_robolab_recovery"
            ),
            "prompt_family": "direct_command",
            "prompt": PROMPTS[relation],
            "reset_identity": f"v3:droid_robolab:neutral_reset:environment_seed_{seed}",
            "success_predicate_id":
                "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        }
        for key, wanted in expected.items():
            if row.get(key) != wanted:
                _fail(f"blocked source queue mismatch for seed {seed}/{relation}/{key}")
        if row.get("prompt_sha256") != hashlib.sha256(PROMPTS[relation].encode()).hexdigest():
            _fail(f"prompt hash mismatch for seed {seed}/{relation}")
        requirement = row.get("runtime_identity_requirement", {})
        if requirement.get("left_right_must_match") is not True:
            _fail("queue no longer requires identical LEFT/RIGHT runtime identity")
    result = []
    for relation in ("left", "right"):
        source = by_relation[relation]
        amended = dict(source)
        amended.update({
            "model_id": MODEL_ID,
            "cell_id": f"v3:droid:{MODEL_ID}:seed{seed}:{relation}",
            "pair_id": f"v3:droid:{MODEL_ID}:seed{seed}",
            "status": "authorized_by_post_result_amendment",
            "execution_status": "gated_by_v3a002_sequential_release",
            "source_cell_id": source["cell_id"],
            "source_pair_id": source["pair_id"],
            "source_status": source["status"],
            "amendment_id": "V3-A002",
        })
        result.append(amended)
    return result


def _git_status_sha256(directory: Path) -> tuple[str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain=v1"], text=True
    )
    return head, hashlib.sha256(status.encode()).hexdigest()


def _git_tree(directory: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()


def _validate_file_record(raw: Any, label: str) -> Path:
    if not isinstance(raw, dict):
        _fail(f"{label} must be a file record")
    path = Path(str(raw.get("path", ""))).resolve()
    if (
        not path.is_file()
        or type(raw.get("bytes")) is not int
        or raw["bytes"] <= 0
        or path.stat().st_size != raw["bytes"]
        or not _is_sha256(raw.get("sha256"))
        or sha256_file(path) != raw["sha256"]
    ):
        _fail(f"{label} file record does not reproduce")
    return path


def _validate_runtime_gates(identity: dict[str, Any]) -> str:
    target = identity.get("target_kubernetes", {})
    if not isinstance(target, dict):
        _fail("V3-A002 target_kubernetes must be an object")
    for key, wanted in {
        "context": "prod-dcwi-warrenq1-vmkub007",
        "namespace": "211247-prod",
        "pvc": "211247-prod-pvc",
        "pvc_mount": "/data",
    }.items():
        if target.get(key) != wanted:
            _fail(f"V3-A002 target Kubernetes mismatch for {key}")
    if target.get("policy") != POLICY_RUNTIME:
        _fail("V3-A002 policy pod/GPU/endpoint identity changed")
    simulator = target.get("simulator", {})
    if not isinstance(simulator, dict):
        _fail("V3-A002 simulator runtime identity must be an object")
    pod = simulator.get("pod")
    if pod not in SIMULATOR_RUNTIMES:
        _fail("V3-A002 simulator pod is outside the three frozen ali-owned shards")
    expected_simulator = {
        "pod": pod,
        **SIMULATOR_RUNTIMES[pod],
        **SIMULATOR_SHARED,
    }
    if simulator != expected_simulator:
        _fail("V3-A002 simulator pod/UID/GPU identity changed")

    writer_ref = identity.get("model_blind_writer_gate", {})
    writer_path = _validate_file_record(writer_ref, "V3-A002 writer-gate manifest")
    writer = _load_object(writer_path)
    if (
        writer.get("schema_version") != WRITER_GATE_SCHEMA
        or writer.get("passed") is not True
        or writer.get("pod") != pod
        or writer.get("model_request_count") != 0
    ):
        _fail("V3-A002 writer-gate manifest identity changed")
    for key in ("viewport_video", "neutral_hold_action", "writer_jsonl"):
        _validate_file_record(writer.get(key), f"V3-A002 writer gate {key}")
    writer_rows = _load_jsonl(Path(writer["writer_jsonl"]["path"]))
    if len(writer_rows) != 1:
        _fail("V3-A002 writer gate must contain one JSONL row")
    writer_row = writer_rows[0]
    if (
        writer_row.get("model_id") != MODEL_ID
        or writer_row.get("pod") != pod
        or writer_row.get("model_request_count") != 0
        or writer_row.get("neutral_reset_verified") is not True
        or writer_row.get("fixture", {}).get("sha256") != FROZEN_FIXTURE_SHA256
    ):
        _fail("V3-A002 writer-gate JSONL contract changed")

    renderer_ref = identity.get("live_renderer_gate", {})
    renderer_path = _validate_file_record(renderer_ref, "V3-A002 renderer-gate manifest")
    renderer = _load_object(renderer_path)
    if (
        renderer.get("schema_version") != RENDERER_GATE_SCHEMA
        or renderer.get("status") != "passed"
        or renderer.get("passed") is not True
        or renderer.get("pod") != pod
        or renderer.get("pod_uid") != simulator["pod_uid"]
        or renderer.get("gpu_uuid") != simulator["gpu_uuid"]
        or renderer.get("model_request_count") != 0
        or renderer.get("environment_seed") != 8310
        or renderer.get("prompt") != PROMPTS["left"]
        or renderer.get("policy_endpoint") != POLICY_RUNTIME["endpoint"]
        or renderer.get("neutral_reset_contract")
        != {"left_predicate_at_reset": False, "right_predicate_at_reset": False}
        or renderer.get("renderer")
        != {"backend": "realtime RTX Vulkan", "quality": "balanced"}
        or renderer.get("simulator_versions")
        != {"isaaclab": "2.2.0", "isaacsim": "5.0.0.0", "robolab": "0.2.1"}
    ):
        _fail("V3-A002 live Isaac/Vulkan renderer-gate identity changed")
    if renderer.get("nvidia_icd") != {
        "path": "/etc/vulkan/icd.d/nvidia_icd.json",
        "sha256": "7bdb6f27d35b66fc848df6f94b8773bba30ea3a7f06f114100d14154a235a34b",
    }:
        _fail("V3-A002 renderer gate NVIDIA Vulkan ICD changed")
    viewport = _validate_file_record(
        renderer.get("viewport_video"), "V3-A002 live renderer viewport"
    )
    if renderer.get("video_decode_frame_count", 0) < 1 or viewport.stat().st_size <= 0:
        _fail("V3-A002 renderer viewport did not decode")
    handshake = renderer.get("websocket_metadata_only_handshake", {})
    expected_server_metadata = {
        "pi0_fast_old_name_config_bridge": "v3a002",
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "openpi_tree": FROZEN_OPENPI_TREE,
        "openpi_config": FROZEN_CONFIG,
        "max_token_len": 250,
        "checkpoint_assets_rule": "checkpoint_local_assets_only",
        "sampling_contract": "required_request_field:sampling_seed",
    }
    if (
        handshake.get("passed") is not True
        or handshake.get("inference_requests_sent") != 0
        or handshake.get("server_metadata") != expected_server_metadata
    ):
        _fail("V3-A002 metadata-only WebSocket handshake changed")
    imports = renderer.get("effective_imports", {})
    if imports.get("pythonpath") != FROZEN_PYTHONPATH:
        _fail("V3-A002 effective PYTHONPATH changed")
    modules = imports.get("modules", {})
    if not isinstance(modules, dict) or set(modules) != set(FROZEN_IMPORTS):
        _fail("V3-A002 effective import module set changed")
    for module, (expected_path, expected_sha) in FROZEN_IMPORTS.items():
        path = _validate_file_record(modules[module], f"V3-A002 import {module}")
        if str(path) != expected_path or modules[module]["sha256"] != expected_sha:
            _fail(f"V3-A002 effective import changed for {module}")
    return str(pod)


def validate_runtime_identity(
    study_root: Path, path: Path, *, check_live_repositories: bool = False
) -> dict[str, Any]:
    identity = _load_object(path)
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": "V3-A002",
        "model_id": MODEL_ID,
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "openpi_tree": FROZEN_OPENPI_TREE,
        "openpi_parent": FROZEN_OPENPI_PARENT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "openpi_config": FROZEN_CONFIG,
        "openpi_config_source_sha256": FROZEN_CONFIG_SOURCE_SHA256,
        "openpi_uv_lock_sha256": FROZEN_UV_LOCK_SHA256,
        "max_token_len": 250,
        "data_config": "SimpleDataConfig",
        "checkpoint_identifier": FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_CHECKPOINT["revision"],
        "checkpoint_manifest_sha256": FROZEN_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_norm_stats_sha256": (
            "b6ea29dd1cce48be4a57850e9118d7723145fb69d86168be8e86ac38fcac8c4f"
        ),
        "open_loop_horizon": OPEN_LOOP_HORIZON,
        "action_chunk_shape": list(ACTION_SHAPE),
        "action_cap": ACTION_CAP,
        "fixed_observation_fixture_sha256": FROZEN_FIXTURE_SHA256,
        "repository_worktrees_clean": True,
        "model_blind_neutral_reset_passed": True,
        "pvc_persistence_and_capacity_passed": True,
        "raw_video_action_state_jsonl_write_passed": True,
    }
    for key, wanted in expected.items():
        if identity.get(key) != wanted:
            _fail(f"V3-A002 runtime identity mismatch for {key}")
    for key in (
        "checkpoint_sha256",
        "environment_lock_hash",
        "adapter_contract_hash",
        "external_repository_diff_hash",
        "openpi_dir_status_sha256",
        "robolab_dir_status_sha256",
        "amendment_sha256",
    ):
        if not _is_sha256(identity.get(key)):
            _fail(f"V3-A002 runtime identity requires lowercase SHA-256 {key}")
    for key in ("runtime_id", "simulator_version", "renderer_backend"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            _fail(f"V3-A002 runtime identity requires non-empty {key}")
    if identity.get("phase_a_queue_sha256") != FROZEN_QUEUE_SHA256:
        _fail("V3-A002 runtime identity queue hash mismatch")
    if identity["amendment_sha256"] != sha256_file(study_root / AMENDMENT_RELATIVE):
        _fail("V3-A002 runtime amendment hash mismatch")
    if identity["checkpoint_sha256"] != checkpoint_contract_sha256(study_root):
        _fail("V3-A002 checkpoint contract hash mismatch")
    if identity["adapter_contract_hash"] != adapter_contract_sha256(study_root):
        _fail("V3-A002 adapter contract hash mismatch")
    if (
        identity.get("frozen_repository_source_sha256")
        != _observed_source_hashes(study_root)
    ):
        _fail("V3-A002 frozen repository source hashes mismatch")
    if identity.get("adapter_source_sha256") != adapter_source_sha256(study_root):
        _fail("V3-A002 adapter source hashes mismatch")
    combined = hashlib.sha256(
        json.dumps(
            {
                "openpi": identity["openpi_dir_status_sha256"],
                "robolab": identity["robolab_dir_status_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if identity["external_repository_diff_hash"] != combined:
        _fail("V3-A002 combined external-repository diff hash mismatch")
    clean_sha = hashlib.sha256(b"").hexdigest()
    if (
        identity["openpi_dir_status_sha256"] != clean_sha
        or identity["robolab_dir_status_sha256"] != clean_sha
    ):
        _fail("V3-A002 requires clean detached OpenPI and RoboLab worktrees")
    simulator_pod = _validate_runtime_gates(identity)

    if check_live_repositories:
        if os.environ.get("HOSTNAME") != simulator_pod:
            _fail("live simulator pod differs from the bound V3-A002 runtime")
        if os.environ.get("PYTHONPATH", "").split(":") != FROZEN_PYTHONPATH:
            _fail("live V3-A002 PYTHONPATH differs from the import-bound runtime")
        for key, wanted in {
            "CUDA_VISIBLE_DEVICES": "0",
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "OMNI_KIT_ACCEPT_EULA": "YES",
        }.items():
            if os.environ.get(key) != wanted:
                _fail(f"live V3-A002 simulator environment mismatch for {key}")
        for label, dir_key, commit_key, status_key in (
            ("OpenPI", "openpi_dir", "openpi_commit", "openpi_dir_status_sha256"),
            ("RoboLab", "robolab_dir", "robolab_commit", "robolab_dir_status_sha256"),
        ):
            directory = Path(str(identity.get(dir_key, "")))
            if not directory.is_dir():
                _fail(f"live {label} directory is unavailable")
            head, status_sha = _git_status_sha256(directory)
            if head != identity[commit_key] or status_sha != identity[status_key]:
                _fail(f"live {label} revision/diff differs from runtime identity")
            if label == "OpenPI" and _git_tree(directory) != FROZEN_OPENPI_TREE:
                _fail("live OpenPI tree differs from the V3-A002 identity")

        openpi_root = Path(str(identity["openpi_dir"]))
        config_source = openpi_root / "src/openpi/training/misc/polaris_config.py"
        if (
            not config_source.is_file()
            or sha256_file(config_source) != FROZEN_CONFIG_SOURCE_SHA256
        ):
            _fail("live OpenPI checkpoint-matched config source changed")
        uv_lock = openpi_root / "uv.lock"
        if not uv_lock.is_file() or sha256_file(uv_lock) != FROZEN_UV_LOCK_SHA256:
            _fail("live V3-A002 OpenPI uv.lock changed")

        checkpoint_root = Path(str(identity.get("checkpoint_path", "")))
        manifest = _load_object(study_root / CHECKPOINT_MANIFEST_RELATIVE)
        if not checkpoint_root.is_dir():
            _fail("live π0-FAST checkpoint directory is unavailable")
        for row in manifest["checkpoint"]["files"]:
            file_path = checkpoint_root / row["path"]
            if (
                not file_path.is_file()
                or file_path.stat().st_size != row["bytes"]
                or sha256_file(file_path) != row["sha256"]
            ):
                _fail(f"live π0-FAST checkpoint payload mismatch: {row['path']}")

        lock = Path(str(identity.get("environment_lock_path", "")))
        if not lock.is_file() or sha256_file(lock) != identity["environment_lock_hash"]:
            _fail("live V3-A002 environment lock hash mismatch")
    return identity


def _resolve_artifact_path(raw: Any, manifest_path: Path) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        _fail("gate artifact path must be a non-empty string")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def validate_release_gate(
    path: Path,
    *,
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    gate = _load_object(path)
    expected = {
        "schema_version": GATE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "status": "passed",
        "behavioral_release": True,
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            _fail(f"V3-A002 release gate mismatch for {key}")

    import numpy as np

    fixture_path = _resolve_artifact_path(gate.get("fixture_path"), path)
    if (
        gate.get("fixture_sha256") != FROZEN_FIXTURE_SHA256
        or not fixture_path.is_file()
        or sha256_file(fixture_path) != FROZEN_FIXTURE_SHA256
    ):
        _fail("V3-A002 gate fixture bytes changed")

    metadata = gate.get("server_metadata", {})
    for key, wanted in {
        "pi0_fast_old_name_config_bridge": "v3a002",
        "openpi_commit": FROZEN_OPENPI_COMMIT,
        "openpi_tree": FROZEN_OPENPI_TREE,
        "openpi_config": FROZEN_CONFIG,
        "max_token_len": 250,
        "checkpoint_assets_rule": "checkpoint_local_assets_only",
        "sampling_contract": "required_request_field:sampling_seed",
    }.items():
        if metadata.get(key) != wanted:
            _fail(f"V3-A002 gate server metadata mismatch for {key}")
    checkpoint_assets = (
        Path(str(runtime_identity["checkpoint_path"])) / "assets"
    ).resolve()
    if (
        Path(str(metadata.get("checkpoint_assets_override", ""))).resolve()
        != checkpoint_assets
    ):
        _fail("V3-A002 gate used a different checkpoint-assets directory")

    records = gate.get("records")
    if not isinstance(records, dict) or set(records) != {"left_a", "left_b", "right"}:
        _fail("V3-A002 gate requires exactly LEFT, repeated LEFT, and RIGHT records")
    arrays: dict[str, np.ndarray] = {}
    for label, relation in (
        ("left_a", "left"),
        ("left_b", "left"),
        ("right", "right"),
    ):
        record = records[label]
        prompt = PROMPTS[relation]
        if (
            not isinstance(record, dict)
            or record.get("prompt") != prompt
            or record.get("prompt_sha256")
            != hashlib.sha256(prompt.encode()).hexdigest()
            or record.get("sampling_seed") != 8_310_000
            or record.get("shape") != list(ACTION_SHAPE)
            or record.get("dtype") != "float32"
            or not _is_sha256(record.get("tokenized_prompt_sha256"))
        ):
            _fail(f"V3-A002 gate record contract changed for {label}")
        action_path = _resolve_artifact_path(record.get("action_path"), path)
        if (
            not action_path.is_file()
            or record.get("action_sha256") != sha256_file(action_path)
        ):
            _fail(f"V3-A002 gate action artifact mismatch for {label}")
        array = np.load(action_path, allow_pickle=False)
        if (
            array.shape != ACTION_SHAPE
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            _fail(f"V3-A002 gate action tensor invalid for {label}")
        arrays[label] = array

    repeat_equal = bool(np.array_equal(arrays["left_a"], arrays["left_b"]))
    action_equal = bool(np.array_equal(arrays["left_a"], arrays["right"]))
    repeat_tokens_equal = (
        records["left_a"]["tokenized_prompt_sha256"]
        == records["left_b"]["tokenized_prompt_sha256"]
    )
    prompt_tokens_differ = (
        records["left_a"]["tokenized_prompt_sha256"]
        != records["right"]["tokenized_prompt_sha256"]
    )
    rms = float(np.sqrt(np.mean((arrays["left_a"] - arrays["right"]) ** 2)))
    metrics = gate.get("metrics", {})
    if (
        not repeat_equal
        or not repeat_tokens_equal
        or not prompt_tokens_differ
        or action_equal
        or not math.isfinite(rms)
        or rms <= 0
        or metrics.get("left_exact_repeat_bit_identical") is not True
        or metrics.get("left_exact_repeat_token_bytes_identical") is not True
        or metrics.get("left_right_token_bytes_differ") is not True
        or metrics.get("left_right_actions_bit_identical") is not False
        or not math.isclose(
            float(metrics.get("left_right_action_rms", -1.0)),
            rms,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        _fail("V3-A002 three-request release metrics do not reproduce")
    return gate


def preflight(
    study_root: Path,
    seed: int,
    runtime_identity_path: Path,
    release_gate_path: Path,
    *,
    check_live_repositories: bool = False,
) -> dict[str, Any]:
    cells = load_authorized_pair(study_root, seed)
    runtime = validate_runtime_identity(
        study_root,
        runtime_identity_path,
        check_live_repositories=check_live_repositories,
    )
    runtime_sha = sha256_file(runtime_identity_path)
    gate = validate_release_gate(
        release_gate_path,
        runtime_identity=runtime,
    )
    return {
        "status": "ready",
        "seed": seed,
        "pair_id": cells[0]["pair_id"],
        "cell_ids": [row["cell_id"] for row in cells],
        "source_cell_ids": [row["source_cell_id"] for row in cells],
        "runtime_identity": runtime,
        "runtime_identity_sha256": runtime_sha,
        "release_gate": gate,
        "release_gate_sha256": sha256_file(release_gate_path),
        "phase_a_queue_sha256": FROZEN_QUEUE_SHA256,
        "amendment_sha256": sha256_file(study_root / AMENDMENT_RELATIVE),
    }


def bridge_command(
    study_root: Path,
    seed: int,
    runtime_identity_path: Path,
    release_gate_path: Path,
    output_dir: Path,
    action_trace_dir: Path,
    remote_host: str,
    remote_port: int,
    *,
    condition: str = "both",
    attempt: int = 0,
) -> list[str]:
    if condition not in {"both", "left", "right"}:
        _fail("bridge condition must be both, left, or right")
    if type(attempt) is not int or attempt < 0:
        _fail("bridge attempt must be a non-negative integer")
    output_folder_name = (
        f"v3a002_pi0_fast_old_name_config_bridge_seed{seed}_{condition}_"
        f"attempt{attempt:02d}"
    )
    return [
        sys.executable,
        str(study_root / "experiments/v3/pi0_fast_old_name_config_bridge/robolab_bridge.py"),
        "--study-root", str(study_root),
        "--environment-seed", str(seed),
        "--sampling-seed-base", str(seed),
        "--runtime-identity", str(runtime_identity_path),
        "--release-gate", str(release_gate_path),
        "--state-capture-dir", str(output_dir / "state_capture"),
        "--action-trace-dir", str(action_trace_dir),
        "--remote-host", remote_host,
        "--remote-port", str(remote_port),
        "--open-loop-horizon", "10",
        "--instruction-controller", "static",
        "--condition", condition,
        "--output-folder-name", output_folder_name,
        "--num-envs", "1",
        "--num-runs", "1",
        "--video-mode", "viewport",
        "--instruction-type", "default",
        "--disable-subtask",
    ]


def _file_record(path: Path, *, nonempty: bool = True) -> dict[str, Any]:
    if not path.is_file() or (nonempty and path.stat().st_size <= 0):
        _fail(f"required retained artifact is absent/empty: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_action_trace(
    path: Path, cell: dict[str, Any], actions_executed: int
) -> tuple[dict[str, Any], Path, Path]:
    import numpy as np

    trace = _load_object(path)
    expected = {
        "schema_version": ACTION_TRACE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "bridge_id": "v3a002",
        "environment_seed": cell["environment_seed"],
        "sampling_seed_base": cell["sampling_seed"],
        "prompt": cell["prompt"],
        "prompt_sha256": hashlib.sha256(cell["prompt"].encode()).hexdigest(),
        "requested_relation": cell["relation"],
        "prompt_controller": "episode_static",
        "open_loop_execution_horizon": OPEN_LOOP_HORIZON,
    }
    for key, wanted in expected.items():
        if trace.get(key) != wanted:
            _fail(f"π0-FAST old-name-config bridge action trace mismatch for {key}")
    entry = trace.get("executed_actions", {})
    action_path = Path(str(entry.get("path", ""))).resolve()
    if (
        not action_path.is_file()
        or entry.get("sha256") != sha256_file(action_path)
        or entry.get("bytes") != action_path.stat().st_size
        or entry.get("count") != actions_executed
        or entry.get("shape") != [actions_executed, 8]
        or entry.get("dtype") != "float32"
    ):
        _fail("π0-FAST old-name-config bridge executed-action artifact/metadata mismatch")
    actions = np.load(action_path, allow_pickle=False)
    if (
        actions.shape != (actions_executed, 8)
        or actions.dtype != np.float32
        or not np.isfinite(actions).all()
    ):
        _fail("V3-A002 executed-action tensor is invalid")
    request_seeds = trace.get("request_sampling_seeds")
    if not isinstance(request_seeds, list) or not request_seeds or request_seeds != [
        cell["sampling_seed"] * 1000 + index for index in range(len(request_seeds))
    ]:
        _fail("π0-FAST old-name-config bridge per-request seed attestation sequence is invalid")
    expected_requests = math.ceil(actions_executed / OPEN_LOOP_HORIZON)
    if len(request_seeds) != expected_requests:
        _fail("V3-A002 request count does not match the ten-action controller")
    chunks_entry = trace.get("returned_action_chunks", {})
    chunks_path = Path(str(chunks_entry.get("path", ""))).resolve()
    if (
        not chunks_path.is_file()
        or chunks_entry.get("sha256") != sha256_file(chunks_path)
        or chunks_entry.get("bytes") != chunks_path.stat().st_size
        or chunks_entry.get("count") != len(request_seeds)
        or chunks_entry.get("shape") != [len(request_seeds), 10, 8]
        or chunks_entry.get("dtype") != "float32"
    ):
        _fail("π0-FAST old-name-config bridge returned action-chunk artifact/metadata mismatch")
    chunks = np.load(chunks_path, allow_pickle=False)
    if (
        chunks.shape != (len(request_seeds), *ACTION_SHAPE)
        or chunks.dtype != np.float32
        or not np.isfinite(chunks).all()
    ):
        _fail("V3-A002 returned action-chunk tensor is invalid")
    attestations = trace.get("request_attestations")
    if not isinstance(attestations, list) or len(attestations) != len(request_seeds):
        _fail("V3-A002 request attestations are incomplete")
    token_hashes = set()
    for index, attestation in enumerate(attestations):
        expected_attestation = {
            "request_index": index,
            "sampling_seed": request_seeds[index],
            "prompt_sha256": hashlib.sha256(cell["prompt"].encode()).hexdigest(),
            "action_chunk_payload_sha256": hashlib.sha256(
                chunks[index].tobytes(order="C")
            ).hexdigest(),
        }
        for key, wanted in expected_attestation.items():
            if attestation.get(key) != wanted:
                _fail(f"V3-A002 request attestation mismatch for {index}/{key}")
        token_sha = attestation.get("tokenized_prompt_sha256")
        if not _is_sha256(token_sha):
            _fail("V3-A002 request token hash is invalid")
        token_hashes.add(token_sha)
    if len(token_hashes) != 1:
        _fail("V3-A002 static prompt reached inconsistent token bytes within an episode")
    return trace, action_path, chunks_path


def _first_sustained(mask: list[bool], count: int = 3) -> int | None:
    run = 0
    for index, value in enumerate(mask):
        run = run + 1 if value else 0
        if run == count:
            return index - count + 1
    return None


def _in_cone(sample: dict[str, Any], relation: str) -> bool:
    obj, ref = sample["object_xyz"], sample["reference_xyz"]
    forward = float(obj[0]) - float(ref[0])
    lateral = float(obj[1]) - float(ref[1])
    margin = lateral if relation == "left" else -lateral
    distance = math.hypot(forward, lateral)
    return distance > 1e-8 and margin / distance >= math.cos(math.radians(45))


def _event_timeline(capture: dict[str, Any], relation: str) -> list[dict[str, Any]]:
    samples = capture["samples"]
    z0 = float(samples[0]["object_xyz"][2])
    pickup = _first_sustained([
        float(sample["object_xyz"][2]) - z0 >= 0.03 for sample in samples
    ])
    requested = next((i for i, sample in enumerate(samples) if _in_cone(sample, relation)), None)
    opposite_relation = "right" if relation == "left" else "left"
    opposite = next(
        (
            i
            for i, sample in enumerate(samples)
            if _in_cone(sample, opposite_relation)
        ),
        None,
    )
    events = [{"event": "episode_start", "action_step": 0}]
    for name, step in (
        ("first_contact", capture.get("first_contact_step")),
        ("verified_pickup", pickup),
        ("requested_region_entry", requested),
        ("opposite_region_entry", opposite),
    ):
        if step is not None:
            events.append({"event": name, "action_step": int(step)})
    events.append({"event": "episode_end", "action_step": int(capture["actions_executed"])})
    rank = {name: index for index, name in enumerate((
        "episode_start", "first_contact", "verified_pickup",
        "requested_region_entry", "opposite_region_entry", "episode_end",
    ))}
    return sorted(events, key=lambda event: (event["action_step"], rank[event["event"]]))


def _validate_state_score_trace(
    capture: dict[str, Any], relation: str
) -> Path:
    samples = capture["samples"]
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("action_step") != index:
            _fail("V3-A002 state samples are not contiguous")
        for key in ("object_xyz", "reference_xyz"):
            point = sample.get(key)
            if (
                not isinstance(point, list)
                or len(point) != 3
                or not all(
                    type(value) in {int, float} and math.isfinite(float(value))
                    for value in point
                )
            ):
                _fail(f"V3-A002 sample {index}/{key} is not finite XYZ")
        if type(sample.get("grippers_open")) is not bool:
            _fail(f"V3-A002 sample {index} has invalid gripper state")
        obj, ref = sample["object_xyz"], sample["reference_xyz"]
        forward = float(obj[0]) - float(ref[0])
        lateral = float(obj[1]) - float(ref[1])
        requested_margin = lateral if relation == "left" else -lateral
        distance = math.hypot(forward, lateral)
        cone_score = requested_margin / distance if distance > 1e-8 else None
        score = sample.get("score_trace")
        if not isinstance(score, dict):
            _fail(f"V3-A002 sample {index} is missing its score trace")
        expected_numbers = {
            "signed_lateral_offset_m": lateral,
            "forward_offset_m": forward,
            "requested_signed_margin_m": requested_margin,
        }
        for key, wanted in expected_numbers.items():
            observed = score.get(key)
            if (
                type(observed) not in {int, float}
                or not math.isclose(
                    float(observed), wanted, rel_tol=0.0, abs_tol=1e-12
                )
            ):
                _fail(f"V3-A002 sample {index} score mismatch for {key}")
        observed_cosine = score.get("requested_cone_cosine")
        if cone_score is None:
            if observed_cosine is not None:
                _fail("zero-distance V3-A002 cone score must be null")
        elif (
            type(observed_cosine) not in {int, float}
            or not math.isclose(
                float(observed_cosine), cone_score, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            _fail(f"V3-A002 sample {index} cone score mismatch")
        threshold = math.cos(math.radians(45.0))
        if score.get("requested_region") is not bool(
            cone_score is not None and cone_score >= threshold
        ):
            _fail(f"V3-A002 sample {index} requested-region score mismatch")
        if score.get("opposite_region") is not bool(
            cone_score is not None and -cone_score >= threshold
        ):
            _fail(f"V3-A002 sample {index} opposite-region score mismatch")

    contract = capture.get("capture_contract", {})
    partial_path = Path(str(contract.get("partial_state_stream", ""))).resolve()
    if not partial_path.is_file():
        _fail("V3-A002 partial state-score stream is absent")
    raw_samples = _load_jsonl(partial_path)
    if raw_samples != samples:
        _fail("V3-A002 partial state-score JSONL differs from final capture")
    if contract.get("failure_early_stopping") is not False:
        _fail("V3-A002 capture no longer prohibits failure early stopping")
    return partial_path


def build_behavioral_record(
    study_root: Path,
    cell: dict[str, Any],
    capture: dict[str, Any],
    capture_path: Path,
    runtime_identity_path: Path,
    release_gate_path: Path,
    video_path: Path,
    action_trace_metadata_path: Path,
    raw_jsonl_path: Path,
) -> dict[str, Any]:
    relation = cell["relation"]
    expected = {
        "schema_version": CAPTURE_SCHEMA,
        "registered_cell_id": cell["cell_id"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "prompt": cell["prompt"],
        "requested_relation": relation,
    }
    for key, wanted in expected.items():
        if capture.get(key) != wanted:
            _fail(f"π0-FAST old-name-config bridge state capture mismatch for {key}")
    samples = capture.get("samples")
    actions_executed = capture.get("actions_executed")
    if type(actions_executed) is not int or actions_executed < 0:
        _fail("π0-FAST old-name-config bridge capture requires a non-negative actions_executed")
    if not isinstance(samples, list) or len(samples) != actions_executed + 1:
        _fail("π0-FAST old-name-config bridge capture must retain initial plus every post-action state")
    if capture.get("behavioral_result_valid_candidate") is not True:
        _fail("partial π0-FAST old-name-config bridge capture cannot enter a behavioral denominator")
    if capture.get("action_cap") != ACTION_CAP:
        _fail("π0-FAST old-name-config bridge uses the frozen 450-action cap")
    if capture.get("requested_success") is True:
        if capture.get("right_censored") is not False:
            _fail("successful π0-FAST old-name-config bridge episode cannot be right-censored")
    elif capture.get("right_censored") is not True or actions_executed != ACTION_CAP:
        _fail("valid π0-FAST old-name-config bridge failure must run to the 450-action cap")
    state_trace_path = _validate_state_score_trace(capture, relation)
    authorization = preflight(
        study_root,
        cell["environment_seed"],
        runtime_identity_path,
        release_gate_path,
    )
    runtime = authorization["runtime_identity"]
    simulator_pod = runtime["target_kubernetes"]["simulator"]["pod"]
    if capture.get("simulator_pod") != simulator_pod:
        _fail("V3-A002 state capture simulator pod differs from runtime identity")
    _, actions_path, chunks_path = _validate_action_trace(
        action_trace_metadata_path, cell, actions_executed
    )
    steps = []
    for sample in samples:
        step = {
            "action_step": sample["action_step"],
            "object_xyz": sample["object_xyz"],
            "reference_xyz": sample["reference_xyz"],
            "grippers_open": sample["grippers_open"],
        }
        if "contact_detected" in sample:
            step["contact_detected"] = sample["contact_detected"]
        steps.append(step)
    record: dict[str, Any] = {
        "schema_version": "vla-wam-shared-v3-raw-episode-v1",
        "record_type": "behavioral_episode",
        "behavioral_result_valid": True,
        "study_id": STUDY_ID,
        "arena": "droid_robolab",
        "registered_cell_id": cell["cell_id"],
        "attempt_id": capture["attempt_id"],
        "model_id": MODEL_ID,
        "simulator_pod": simulator_pod,
        "pair_id": cell["pair_id"],
        "bridge_provenance": {
            "amendment_id": "V3-A002",
            "amendment_sha256": sha256_file(study_root / AMENDMENT_RELATIVE),
            "source_blocked_cell_id": cell["source_cell_id"],
            "source_blocked_pair_id": cell["source_pair_id"],
            "historical_pooling_prohibited": True,
        },
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": cell["success_predicate_id"],
        "reset_id": cell["reset_identity"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "requested_relation": relation,
        "requested_success": capture["requested_success"],
        "failure_stage": capture["frozen_failure_stage"],
        "frozen_failure_stage": capture["frozen_failure_stage"],
        "failure_taxonomy": "transport_failed",
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "measurement_frame_description": (
            "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
            "forward is object-minus-reference x and lateral is object-minus-reference y, "
            "with positive lateral denoting robot LEFT."
        ),
        "checkpoint": dict(FROZEN_CHECKPOINT),
        "runtime_identity": {
            "id": runtime["runtime_id"],
            "sha256": sha256_file(runtime_identity_path),
        },
        "release_authorization": {
            "amendment_id": "V3-A002",
            "release_gate_sha256": authorization["release_gate_sha256"],
            "three_request_gate_passed": True,
        },
        "artifacts": {
            "viewport_video": _file_record(video_path),
            "executed_action_trace": _file_record(actions_path),
            "returned_action_chunks": _file_record(chunks_path),
            "action_trace_manifest": _file_record(action_trace_metadata_path),
            "state_score_trace": _file_record(state_trace_path),
            "state_capture_manifest": _file_record(capture_path),
            "raw_result_jsonl": {
                "path": str(raw_jsonl_path),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "steps": steps,
        "actions_executed": actions_executed,
        "action_cap": ACTION_CAP,
        "right_censored": capture["right_censored"],
        "first_contact_step": capture.get("first_contact_step"),
        "first_contact_unavailable_reason": capture.get("first_contact_unavailable_reason"),
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": capture["operational_wall_time_valid"],
        "event_timeline": _event_timeline(capture, relation),
        "controller_contract": {
            "open_loop_horizon": OPEN_LOOP_HORIZON,
            "prompt_controller": "episode_static",
            "failure_early_stopping": False,
        },
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import (  # type: ignore
        derive_failure_taxonomy,
        derive_initial_state_sha256,
        derive_measurements,
        validate_behavioral_record,
    )
    record["initial_state_sha256"] = derive_initial_state_sha256(record)
    measurements = derive_measurements(record)
    record["failure_taxonomy"] = derive_failure_taxonomy(record, measurements)
    return validate_behavioral_record(record)


def build_infrastructure_record(
    study_root: Path,
    cell: dict[str, Any],
    capture: dict[str, Any],
    runtime_identity_path: Path,
    raw_jsonl_path: Path,
    *,
    video_path: Path | None = None,
    action_trace_path: Path | None = None,
) -> dict[str, Any]:
    if capture.get("schema_version") != INFRA_CAPTURE_SCHEMA:
        _fail("unexpected π0-FAST old-name-config bridge infrastructure capture schema")
    for key, wanted in (
        ("registered_cell_id", cell["cell_id"]),
        ("environment_seed", cell["environment_seed"]),
        ("policy_seed", cell["sampling_seed"]),
        ("prompt", cell["prompt"]),
        ("requested_relation", cell["relation"]),
    ):
        if capture.get(key) != wanted:
            _fail(f"π0-FAST old-name-config bridge infrastructure capture mismatch for {key}")
    runtime = validate_runtime_identity(study_root, runtime_identity_path)
    artifacts: dict[str, Any] = {
        "raw_result_jsonl": {
            "path": str(raw_jsonl_path),
            "integrity_scope": "batch_manifest_after_close",
        }
    }
    if video_path is not None:
        artifacts["viewport_video"] = _file_record(video_path)
    if action_trace_path is not None:
        artifacts["executed_action_trace"] = _file_record(action_trace_path)
    log_path_raw = capture.get("log_path")
    if not isinstance(log_path_raw, str) or not log_path_raw.strip():
        _fail("V3-A002 infrastructure capture requires a retained technical log")
    log_path = Path(log_path_raw).resolve()
    if capture.get("log_hash") != sha256_file(log_path):
        _fail("V3-A002 infrastructure log hash does not match the retained log")
    artifacts["technical_log"] = _file_record(log_path)
    record = {
        "schema_version": "vla-wam-shared-v3-infrastructure-attempt-v1",
        "record_type": "infrastructure_attempt",
        "behavioral_result_valid": False,
        "classification": capture["classification"],
        "study_id": STUDY_ID,
        "arena": "droid_robolab",
        "registered_cell_id": cell["cell_id"],
        "attempt_id": capture["attempt_id"],
        "model_id": MODEL_ID,
        "pair_id": cell["pair_id"],
        "bridge_provenance": {
            "amendment_id": "V3-A002",
            "amendment_sha256": sha256_file(study_root / AMENDMENT_RELATIVE),
            "source_blocked_cell_id": cell["source_cell_id"],
            "historical_pooling_prohibited": True,
        },
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": cell["success_predicate_id"],
        "reset_id": cell["reset_identity"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "measurement_frame": "robot_base_object_minus_reference_xyz_m",
        "measurement_frame_description": (
            "Object and reference XYZ samples are expressed in the frozen robot-base frame; "
            "forward is object-minus-reference x and lateral is object-minus-reference y, "
            "with positive lateral denoting robot LEFT."
        ),
        "checkpoint": dict(FROZEN_CHECKPOINT),
        "runtime_identity": {
            "id": runtime["runtime_id"],
            "sha256": sha256_file(runtime_identity_path),
        },
        "artifacts": artifacts,
        "stage": capture["stage"],
        "error": capture["error"],
        "log_hash": capture["log_hash"],
        "runtime_intervention": capture["runtime_intervention"],
        "repair_attempt_id": capture.get("repair_attempt_id"),
        "event_timeline": capture["event_timeline"],
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import validate_infrastructure_record  # type: ignore
    return validate_infrastructure_record(record)


def _output_paths_available(paths: list[Path]) -> None:
    for path in paths:
        manifest = path.with_name(path.name + ".manifest.json")
        if path.exists() or manifest.exists():
            _fail(f"refusing to overwrite retained V3-A002 output: {path}")


def _trace_token_hashes(path: Path) -> set[str]:
    trace = _load_object(path)
    attestations = trace.get("request_attestations")
    if not isinstance(attestations, list) or not attestations:
        _fail("V3-A002 action trace has no request attestations")
    hashes = {row.get("tokenized_prompt_sha256") for row in attestations}
    if len(hashes) != 1 or not all(_is_sha256(value) for value in hashes):
        _fail("V3-A002 episode token hashes are not one stable SHA-256")
    return hashes


def compile_behavioral_pair(
    *,
    study_root: Path,
    seed: int,
    runtime_identity_path: Path,
    release_gate_path: Path,
    left_capture_path: Path,
    left_video_path: Path,
    left_action_trace_path: Path,
    left_output_jsonl: Path,
    right_capture_path: Path,
    right_video_path: Path,
    right_action_trace_path: Path,
    right_output_jsonl: Path,
    pair_manifest_path: Path,
) -> dict[str, Any]:
    """Compile both cells atomically after proving identical reset state."""

    cells = {
        row["relation"]: row for row in load_authorized_pair(study_root, seed)
    }
    outputs = [left_output_jsonl, right_output_jsonl]
    if left_output_jsonl.resolve() == right_output_jsonl.resolve():
        _fail("LEFT and RIGHT require separate per-episode raw JSONL files")
    _output_paths_available(outputs)
    if pair_manifest_path.exists():
        _fail(f"refusing to overwrite V3-A002 pair manifest: {pair_manifest_path}")

    captures = {
        "left": _load_object(left_capture_path),
        "right": _load_object(right_capture_path),
    }
    records = {
        "left": build_behavioral_record(
            study_root,
            cells["left"],
            captures["left"],
            left_capture_path,
            runtime_identity_path,
            release_gate_path,
            left_video_path,
            left_action_trace_path,
            left_output_jsonl,
        ),
        "right": build_behavioral_record(
            study_root,
            cells["right"],
            captures["right"],
            right_capture_path,
            runtime_identity_path,
            release_gate_path,
            right_video_path,
            right_action_trace_path,
            right_output_jsonl,
        ),
    }
    if (
        records["left"]["initial_state_sha256"]
        != records["right"]["initial_state_sha256"]
    ):
        _fail("matched LEFT/RIGHT captures do not begin from an identical reset")
    if records["left"]["runtime_identity"] != records["right"]["runtime_identity"]:
        _fail("matched LEFT/RIGHT captures use different runtime identities")
    if _trace_token_hashes(left_action_trace_path) == _trace_token_hashes(
        right_action_trace_path
    ):
        _fail("LEFT and RIGHT episode prompts reached identical tokenizer bytes")

    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import write_jsonl  # type: ignore

    left_manifest = write_jsonl(left_output_jsonl, [records["left"]])
    right_manifest = write_jsonl(right_output_jsonl, [records["right"]])
    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pair_manifest = {
        "schema_version": (
            "vla-wam-shared-v3-pi0-fast-old-name-config-pair-manifest-v1"
        ),
        "study_id": STUDY_ID,
        "amendment_id": "V3-A002",
        "model_id": MODEL_ID,
        "pair_id": records["left"]["pair_id"],
        "environment_seed": seed,
        "initial_state_sha256": records["left"]["initial_state_sha256"],
        "runtime_identity": records["left"]["runtime_identity"],
        "release_gate_sha256": records["left"]["release_authorization"][
            "release_gate_sha256"
        ],
        "left": {
            "registered_cell_id": records["left"]["registered_cell_id"],
            "raw_jsonl": _file_record(left_output_jsonl),
            "raw_jsonl_manifest": _file_record(
                left_output_jsonl.with_name(
                    left_output_jsonl.name + ".manifest.json"
                )
            ),
        },
        "right": {
            "registered_cell_id": records["right"]["registered_cell_id"],
            "raw_jsonl": _file_record(right_output_jsonl),
            "raw_jsonl_manifest": _file_record(
                right_output_jsonl.with_name(
                    right_output_jsonl.name + ".manifest.json"
                )
            ),
        },
        "historical_pooling_prohibited": True,
    }
    pair_manifest_path.write_text(
        json.dumps(pair_manifest, indent=2, sort_keys=True) + "\n"
    )
    return {
        "status": "compiled",
        "pair_manifest": _file_record(pair_manifest_path),
        "left_jsonl_manifest": left_manifest,
        "right_jsonl_manifest": right_manifest,
        "initial_state_sha256": pair_manifest["initial_state_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)

    audit = commands.add_parser("audit-static")
    audit.add_argument("--study-root", type=Path, required=True)
    audit.add_argument("--seed", type=int, default=8310)

    for name in ("preflight", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--study-root", type=Path, required=True)
        command.add_argument("--seed", type=int, required=True)
        command.add_argument("--runtime-identity", type=Path, required=True)
        command.add_argument("--release-gate", type=Path, required=True)
        command.add_argument("--check-live-repositories", action="store_true")
    plan = commands.choices["plan"]
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--action-trace-dir", type=Path, required=True)
    plan.add_argument("--remote-host", required=True)
    plan.add_argument("--remote-port", type=int, default=8011)
    plan.add_argument(
        "--condition", choices=["both", "left", "right"], default="both"
    )
    plan.add_argument("--attempt", type=int, default=0)

    pair = commands.add_parser("compile-pair")
    for argument in (
        "study-root",
        "runtime-identity",
        "release-gate",
        "left-capture",
        "left-video",
        "left-action-trace",
        "left-output-jsonl",
        "right-capture",
        "right-video",
        "right-action-trace",
        "right-output-jsonl",
        "pair-manifest",
    ):
        pair.add_argument(f"--{argument}", type=Path, required=True)
    pair.add_argument("--seed", type=int, required=True)

    infrastructure = commands.add_parser("compile-infrastructure")
    for argument in (
        "study-root",
        "runtime-identity",
        "capture",
        "output-jsonl",
    ):
        infrastructure.add_argument(f"--{argument}", type=Path, required=True)
    infrastructure.add_argument("--video", type=Path)
    infrastructure.add_argument("--action-trace", type=Path)
    infrastructure.add_argument("--seed", type=int, required=True)
    infrastructure.add_argument(
        "--relation", choices=["left", "right"], required=True
    )

    record_infrastructure = commands.add_parser("record-infrastructure")
    for argument in (
        "study-root",
        "runtime-identity",
        "log",
        "output-jsonl",
    ):
        record_infrastructure.add_argument(
            f"--{argument}", type=Path, required=True
        )
    record_infrastructure.add_argument("--seed", type=int, required=True)
    record_infrastructure.add_argument(
        "--relation", choices=["left", "right"], required=True
    )
    record_infrastructure.add_argument("--attempt-id", required=True)
    record_infrastructure.add_argument(
        "--classification",
        choices=["technical_invalid", "partial"],
        required=True,
    )
    record_infrastructure.add_argument("--stage", required=True)
    record_infrastructure.add_argument("--error", required=True)
    record_infrastructure.add_argument(
        "--runtime-intervention", action="store_true"
    )
    record_infrastructure.add_argument("--repair-attempt-id")
    record_infrastructure.add_argument("--video", type=Path)
    record_infrastructure.add_argument("--action-trace", type=Path)

    args = parser.parse_args()
    root = args.study_root.resolve()

    if args.mode == "audit-static":
        cells = load_authorized_pair(root, args.seed)
        result = {
            "status": "static_contract_valid",
            "amendment_sha256": sha256_file(root / AMENDMENT_RELATIVE),
            "queue_sha256": sha256_file(root / QUEUE_RELATIVE),
            "checkpoint_contract_sha256": checkpoint_contract_sha256(root),
            "adapter_source_sha256": adapter_source_sha256(root),
            "cell_ids": [row["cell_id"] for row in cells],
            "source_cell_ids": [row["source_cell_id"] for row in cells],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.mode in {"preflight", "plan"}:
        result = preflight(
            root,
            args.seed,
            args.runtime_identity,
            args.release_gate,
            check_live_repositories=args.check_live_repositories,
        )
        if args.mode == "plan":
            result["command"] = bridge_command(
                root,
                args.seed,
                args.runtime_identity,
                args.release_gate,
                args.output_dir,
                args.action_trace_dir,
                args.remote_host,
                args.remote_port,
                condition=args.condition,
                attempt=args.attempt,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if args.mode == "compile-pair":
        result = compile_behavioral_pair(
            study_root=root,
            seed=args.seed,
            runtime_identity_path=args.runtime_identity,
            release_gate_path=args.release_gate,
            left_capture_path=args.left_capture,
            left_video_path=args.left_video,
            left_action_trace_path=args.left_action_trace,
            left_output_jsonl=args.left_output_jsonl,
            right_capture_path=args.right_capture,
            right_video_path=args.right_video,
            right_action_trace_path=args.right_action_trace,
            right_output_jsonl=args.right_output_jsonl,
            pair_manifest_path=args.pair_manifest,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    cell = {
        row["relation"]: row for row in load_authorized_pair(root, args.seed)
    }[args.relation]
    if args.mode == "compile-infrastructure":
        capture = _load_object(args.capture)
    else:
        if not args.log.is_file() or args.log.stat().st_size <= 0:
            _fail("record-infrastructure requires a retained non-empty log")
        capture = {
            "schema_version": INFRA_CAPTURE_SCHEMA,
            "registered_cell_id": cell["cell_id"],
            "attempt_id": args.attempt_id,
            "environment_seed": cell["environment_seed"],
            "policy_seed": cell["sampling_seed"],
            "prompt": cell["prompt"],
            "requested_relation": cell["relation"],
            "classification": args.classification,
            "stage": args.stage,
            "error": args.error,
            "log_path": str(args.log.resolve()),
            "log_hash": sha256_file(args.log),
            "runtime_intervention": args.runtime_intervention,
            "repair_attempt_id": args.repair_attempt_id,
            "event_timeline": [
                {"sequence": 0, "stage": "attempt_started"},
                {"sequence": 1, "stage": args.stage},
            ],
        }
    result = build_infrastructure_record(
        root,
        cell,
        capture,
        args.runtime_identity,
        args.output_jsonl,
        video_path=args.video,
        action_trace_path=args.action_trace,
    )
    sys.path.insert(0, str(root / "tools"))
    from vla_wam_v3_episode_schema import write_jsonl  # type: ignore

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            write_jsonl(args.output_jsonl, [result]),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
