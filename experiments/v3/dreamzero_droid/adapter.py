#!/usr/bin/env python3
"""Fail-closed powered-v3 adapter for the DreamZero s=2 DROID arm.

The committed v3 queue names ``dreamzero_droid_action_cfg``.  That exact model
ID occurs in the frozen V2-A015 ``dreamzero_action_cfg_s2`` arm; the original
V2-A007/s=1 checkpoint baseline instead uses ``dreamzero_droid``.  This module
therefore binds the prospective rows only to the derived s=2 negative-branch
action-guidance runtime and rejects baseline/s=1 identities.

No model inference is performed here.  The adapter validates the queue,
runtime/checkpoint/overlay hashes, and a fresh future-aware release gate before
emitting a full matched-pair bridge command or compiling retained evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any


MODEL_ID = "dreamzero_droid_action_cfg"
IDENTITY_BINDING = "V2-A015:dreamzero_action_cfg_s2"
BASELINE_MODEL_ID = "dreamzero_droid"
STUDY_ID = "vla_wam_language_steerability_v3"
PHASE = "A_direct_command_matched_pairs"
QUEUE_SCHEMA = "vla-wam-shared-v3-phase-a-cells-v1"
QUEUE_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells.jsonl")
QUEUE_MANIFEST_RELATIVE = Path("artifacts/vla_wam_shared_v3/phase_a_cells_manifest.json")
V2A015_AMENDMENT_RELATIVE = Path(
    "artifacts/vla_wam_shared_v2/pilot/post_result_cfg_ablation_v2a015_amendment.json"
)
CHECKPOINT_MANIFEST_RELATIVE = Path(
    "artifacts/vla_wam_shared_v2/pilot/expansion/"
    "dreamzero_official_source_checkpoint_manifest.json"
)
RUNTIME_SCHEMA = "vla-wam-shared-v3-dreamzero-s2-runtime-identity-v1"
GATE_SCHEMA = "vla-wam-shared-v3-dreamzero-s2-release-gate-v1"
CAPTURE_SCHEMA = "vla-wam-shared-v3-dreamzero-s2-state-capture-v1"
INFRA_CAPTURE_SCHEMA = "vla-wam-shared-v3-dreamzero-s2-infrastructure-capture-v1"

PROMPTS = {
    "left": "Put the Rubik's cube to the left of the bowl.",
    "right": "Put the Rubik's cube to the right of the bowl.",
}
TASKS = {
    "left": "RubiksCubeLeftOfBowlMatchedTask",
    "right": "RubiksCubeRightOfBowlMatchedTask",
}
FROZEN_SOURCE_COMMIT = "ab790c198fbce33503358efbbd4187ce9a89adf3"
FROZEN_ROBOLAB_COMMIT = "0aef241fb088ca21bb4ebd24448940ed56620d17"
FROZEN_CHECKPOINT = {
    "id": "GEAR-Dreams/DreamZero-DROID",
    "revision": "96ad344138c66e82536422432ad742f015784942",
}
FROZEN_CHECKPOINT_MANIFEST_SHA256 = (
    "75fd6c6b7601f5706eb70140519ee8d57b18fe79e49cc2792c30b0d9be016eeb"
)
FROZEN_CHECKPOINT_AGGREGATE_SHA256 = (
    "b4af0ac93474c3295c1ba841a34a8f2f91a5c3ec3c6aac1431b97689a6618c56"
)
FROZEN_TOKENIZER_AGGREGATE_SHA256 = (
    "00f9b974f8f0b33c5e284849a0507c308893d01d915377ba88c2f7084ea92434"
)
ACTION_CFG_STYLE_SCALE = 2.0
BASELINE_ACTION_CFG_EQUIVALENT = 1.0
VIDEO_CFG_SCALE = 5.0
OFFICIAL_NOISE_SEED = 1140
OPEN_LOOP_HORIZON = 8
ACTION_CHUNK_SHAPE = [24, 8]
RUNTIME_NUM_INFERENCE_STEPS = 16
EVALUATED_DIT_STEPS = 8
OVERLAY_TARGET_SHA256 = "65dc9873aef37563dedf3787fd7b59e0a6d50e575775e38b70edd9e38489f9b8"
OVERLAY_PATCH_SHA256 = "de2b82a1c9f81ee4751fb384158775865f5e5e8fb622134cae3b2c4e8b2a2cc0"
FROZEN_V2_SOURCES = {
    "experiments/dreamzero_droid/v2a015_instrumented_server.py":
        "5f9932acc77fe8fc80157622ef3e1db5b552bd31f322768330f2000f9af7d09d",
    "experiments/dreamzero_droid/v2a015_robolab_client.py":
        "4609a2e4cee618898c5037418d148453906a4fd2cffee40d6c5c1a7d13241da4",
    "experiments/dreamzero_droid/v2a015_robolab_gate.py":
        "b0c18d26e9dd4ab368d64a461c5e35e84e122162045aa3374ed75fd88a2e4b3d",
    "experiments/dreamzero_droid/v2_robolab_client.py":
        "3f3fe50f4b7747c52a995efb1e07bf00af8a5e38620829d8b0c991b43fd9a7ec",
    "experiments/dreamzero_droid/v2a015_action_cfg.patch": OVERLAY_PATCH_SHA256,
    "experiments/dreamzero_droid/robolab_v2_tasks/rubiks_cube_left_of_bowl_matched.py":
        "d9663bbcfbfe301af41c8551f6a878df13014312b9451466465b00623422a64c",
    "experiments/dreamzero_droid/robolab_v2_tasks/rubiks_cube_right_of_bowl_matched.py":
        "e1587bf5a57f58988a5fb1f85fc03d7ce788562096fdd142552bb4dd34aaefec",
}
ADAPTER_SOURCES = (
    "experiments/v3/dreamzero_droid/adapter.py",
    "experiments/v3/dreamzero_droid/client.py",
    "experiments/v3/dreamzero_droid/robolab_bridge.py",
)
HEX64 = set("0123456789abcdef")


class AdapterError(ValueError):
    """The DreamZero s=2 queue/runtime/release/evidence contract is invalid."""


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
                _fail(f"blank queue line at {path}:{number}")
            row = json.loads(line)
            if not isinstance(row, dict):
                _fail(f"queue row at {path}:{number} is not an object")
            rows.append(row)
    except (OSError, json.JSONDecodeError) as error:
        raise AdapterError(f"cannot read queue {path}: {error}") from error
    return rows


def _observed_source_hashes(study_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in FROZEN_V2_SOURCES.items():
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing frozen DreamZero s=2 source: {path}")
        observed[relative] = sha256_file(path)
        if observed[relative] != expected:
            _fail(f"frozen DreamZero s=2 source hash mismatch: {relative}")
    return observed


def adapter_source_sha256(study_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in ADAPTER_SOURCES:
        path = study_root / relative
        if not path.is_file():
            _fail(f"missing v3 DreamZero adapter source: {path}")
        result[relative] = sha256_file(path)
    return result


def _binding_arm(study_root: Path) -> dict[str, Any]:
    amendment_path = study_root / V2A015_AMENDMENT_RELATIVE
    amendment = _load_object(amendment_path)
    if (
        amendment.get("schema_version")
        != "vla-wam-shared-v2-post-result-cfg-ablation-v1"
        or amendment.get("amendment_id") != "V2-A015"
    ):
        _fail("DreamZero identity source is not the frozen V2-A015 amendment")
    arms = {row.get("model_id"): row for row in amendment.get("arms", [])}
    arm = arms.get(MODEL_ID)
    if not isinstance(arm, dict):
        _fail("V2-A015 has no exact dreamzero_droid_action_cfg arm")
    expected = {
        "arm_id": "dreamzero_action_cfg_s2",
        "model_id": MODEL_ID,
        "checkpoint": FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_CHECKPOINT["revision"],
        "source_commit": FROZEN_SOURCE_COMMIT,
        "action_guidance": ACTION_CFG_STYLE_SCALE,
        "baseline_action_guidance_equivalent": BASELINE_ACTION_CFG_EQUIVALENT,
        "video_guidance": VIDEO_CFG_SCALE,
        "runtime_num_inference_steps": RUNTIME_NUM_INFERENCE_STEPS,
        "dit_cache": True,
        "evaluated_dit_steps": EVALUATED_DIT_STEPS,
        "action_chunk_shape": ACTION_CHUNK_SHAPE,
        "executed_open_loop_horizon": OPEN_LOOP_HORIZON,
    }
    for key, wanted in expected.items():
        if arm.get(key) != wanted:
            _fail(f"DreamZero V2-A015 s=2 binding changed for {key}")
    if BASELINE_MODEL_ID in arms:
        _fail("baseline DreamZero model ID unexpectedly aliases the s=2 arm")
    return arm


def binding_contract_sha256(study_root: Path) -> str:
    arm = _binding_arm(study_root)
    contract = {
        "v3_model_id": MODEL_ID,
        "binding": IDENTITY_BINDING,
        "v2_arm": arm,
        "baseline_model_id_rejected": BASELINE_MODEL_ID,
        "amendment_sha256": sha256_file(study_root / V2A015_AMENDMENT_RELATIVE),
    }
    return hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def checkpoint_contract_sha256(study_root: Path) -> str:
    path = study_root / CHECKPOINT_MANIFEST_RELATIVE
    if sha256_file(path) != FROZEN_CHECKPOINT_MANIFEST_SHA256:
        _fail("committed DreamZero checkpoint manifest hash changed")
    manifest = _load_object(path)
    checkpoint = manifest.get("checkpoint", {})
    tokenizer = manifest.get("tokenizer", {})
    if (
        checkpoint.get("repository") != FROZEN_CHECKPOINT["id"]
        or checkpoint.get("revision") != FROZEN_CHECKPOINT["revision"]
        or checkpoint.get("payload_file_count") != 25
        or checkpoint.get("payload_bytes") != 64_789_159_581
        or checkpoint.get("aggregate_sha256") != FROZEN_CHECKPOINT_AGGREGATE_SHA256
        or tokenizer.get("repository") != "google/umt5-xxl"
        or tokenizer.get("revision") != "66cb9e7e85526fe440a945569e42c72fb6cbc0ad"
        or tokenizer.get("aggregate_sha256") != FROZEN_TOKENIZER_AGGREGATE_SHA256
    ):
        _fail("committed DreamZero checkpoint/tokenizer contract changed")
    payload = {
        "checkpoint": checkpoint,
        "tokenizer": tokenizer,
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def adapter_contract_sha256(study_root: Path) -> str:
    contract = {
        "model_id": MODEL_ID,
        "identity_binding": IDENTITY_BINDING,
        "binding_contract_sha256": binding_contract_sha256(study_root),
        "queue_sha256": sha256_file(study_root / QUEUE_RELATIVE),
        "checkpoint_contract_sha256": checkpoint_contract_sha256(study_root),
        "frozen_v2_sources": FROZEN_V2_SOURCES,
        "adapter_sources": adapter_source_sha256(study_root),
        "source_commit": FROZEN_SOURCE_COMMIT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "overlay_target_sha256": OVERLAY_TARGET_SHA256,
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": VIDEO_CFG_SCALE,
        "official_noise_seed": OFFICIAL_NOISE_SEED,
    }
    return hashlib.sha256(json.dumps(
        contract, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def load_authorized_pair(study_root: Path, seed: int) -> list[dict[str, Any]]:
    if seed not in range(8303, 8330):
        _fail("DreamZero v3 Phase-A new seeds are exactly 8303-8329")
    _binding_arm(study_root)
    queue_path = study_root / QUEUE_RELATIVE
    manifest = _load_object(study_root / QUEUE_MANIFEST_RELATIVE)
    queue_sha = sha256_file(queue_path)
    if (
        manifest.get("queue_file") != str(QUEUE_RELATIVE)
        or manifest.get("queue_sha256") != queue_sha
        or manifest.get("study_id") != STUDY_ID
    ):
        _fail("committed Phase-A queue/manifest identity mismatch")
    selected = [
        row for row in _load_jsonl(queue_path)
        if row.get("model_id") == MODEL_ID and row.get("environment_seed") == seed
    ]
    if len(selected) != 2:
        _fail(f"expected exactly two DreamZero s=2 rows for seed {seed}")
    by_relation = {row.get("relation"): row for row in selected}
    if set(by_relation) != {"left", "right"}:
        _fail("DreamZero registered pair must contain one LEFT and one RIGHT row")
    for relation, row in by_relation.items():
        expected = {
            "schema_version": QUEUE_SCHEMA,
            "study_id": STUDY_ID,
            "arena": "droid_robolab",
            "phase": PHASE,
            "model_id": MODEL_ID,
            "cell_id": f"v3:droid:{MODEL_ID}:seed{seed}:{relation}",
            "pair_id": f"v3:droid:{MODEL_ID}:seed{seed}",
            "environment_seed": seed,
            "sampling_seed": seed,
            "replicate": 0,
            "status": "authorized_new",
            "execution_status": "authorized_after_all_registered_release_gates",
            "prompt_family": "direct_command",
            "prompt": PROMPTS[relation],
            "reset_identity": f"v3:droid_robolab:neutral_reset:environment_seed_{seed}",
            "success_predicate_id":
                "v2_frozen_droid_robolab_release_inside_45deg_requested_relation",
        }
        for key, wanted in expected.items():
            if row.get(key) != wanted:
                _fail(f"DreamZero queue mismatch for seed {seed}/{relation}/{key}")
        if row.get("prompt_sha256") != hashlib.sha256(PROMPTS[relation].encode()).hexdigest():
            _fail(f"DreamZero prompt hash mismatch for seed {seed}/{relation}")
        if row.get("runtime_identity_requirement", {}).get("left_right_must_match") is not True:
            _fail("DreamZero queue no longer requires matched runtime identity")
    return [by_relation["left"], by_relation["right"]]


def _git_status_sha256(directory: Path) -> tuple[str, str]:
    head = subprocess.check_output(
        ["git", "-C", str(directory), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(directory), "status", "--porcelain=v1"], text=True
    )
    return head, hashlib.sha256(status.encode()).hexdigest()


def validate_runtime_identity(
    study_root: Path, path: Path, *, check_live_repositories: bool = False
) -> dict[str, Any]:
    identity = _load_object(path)
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "identity_binding": IDENTITY_BINDING,
        "baseline_model_id_rejected": BASELINE_MODEL_ID,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "robolab_commit": FROZEN_ROBOLAB_COMMIT,
        "checkpoint_identifier": FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_CHECKPOINT["revision"],
        "checkpoint_manifest_sha256": FROZEN_CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_payload_aggregate_sha256": FROZEN_CHECKPOINT_AGGREGATE_SHA256,
        "tokenizer_payload_aggregate_sha256": FROZEN_TOKENIZER_AGGREGATE_SHA256,
        "overlay_target_sha256": OVERLAY_TARGET_SHA256,
        "overlay_patch_sha256": OVERLAY_PATCH_SHA256,
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "baseline_action_cfg_equivalent": BASELINE_ACTION_CFG_EQUIVALENT,
        "video_cfg_scale": VIDEO_CFG_SCALE,
        "official_noise_seed": OFFICIAL_NOISE_SEED,
        "runtime_num_inference_steps": RUNTIME_NUM_INFERENCE_STEPS,
        "evaluated_dit_steps": EVALUATED_DIT_STEPS,
        "dit_cache": True,
        "open_loop_horizon": OPEN_LOOP_HORIZON,
        "action_chunk_shape": ACTION_CHUNK_SHAPE,
    }
    for key, wanted in expected.items():
        if identity.get(key) != wanted:
            _fail(f"DreamZero s=2 runtime identity mismatch for {key}")
    if identity.get("model_id") == BASELINE_MODEL_ID or identity.get("action_cfg_style_scale") == 1:
        _fail("DreamZero s=1 baseline cannot satisfy the v3 action_cfg identity")
    for key in (
        "checkpoint_sha256", "environment_lock_hash", "adapter_contract_hash",
        "binding_contract_sha256", "external_repository_diff_hash",
        "dreamzero_dir_status_sha256", "robolab_dir_status_sha256",
    ):
        if not _is_sha256(identity.get(key)):
            _fail(f"DreamZero runtime identity requires lowercase SHA-256 {key}")
    for key in ("runtime_id", "simulator_version", "renderer_backend"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            _fail(f"DreamZero runtime identity requires non-empty {key}")
    if identity.get("phase_a_queue_sha256") != sha256_file(study_root / QUEUE_RELATIVE):
        _fail("DreamZero runtime queue hash mismatch")
    if identity["checkpoint_sha256"] != checkpoint_contract_sha256(study_root):
        _fail("DreamZero checkpoint contract hash mismatch")
    if identity["binding_contract_sha256"] != binding_contract_sha256(study_root):
        _fail("DreamZero s=2 binding contract hash mismatch")
    if identity["adapter_contract_hash"] != adapter_contract_sha256(study_root):
        _fail("DreamZero adapter contract hash mismatch")
    if identity.get("frozen_v2_source_sha256") != _observed_source_hashes(study_root):
        _fail("DreamZero frozen V2 source hashes mismatch")
    if identity.get("adapter_source_sha256") != adapter_source_sha256(study_root):
        _fail("DreamZero v3 adapter source hashes mismatch")
    combined = hashlib.sha256(json.dumps(
        {
            "dreamzero": identity["dreamzero_dir_status_sha256"],
            "robolab": identity["robolab_dir_status_sha256"],
        }, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    if identity["external_repository_diff_hash"] != combined:
        _fail("DreamZero combined repository diff hash mismatch")
    if check_live_repositories:
        dreamzero_dir = Path(str(identity.get("dreamzero_dir", "")))
        robolab_dir = Path(str(identity.get("robolab_dir", "")))
        for label, directory, commit, status_key in (
            ("DreamZero", dreamzero_dir, FROZEN_SOURCE_COMMIT, "dreamzero_dir_status_sha256"),
            ("RoboLab", robolab_dir, FROZEN_ROBOLAB_COMMIT, "robolab_dir_status_sha256"),
        ):
            if not directory.is_dir():
                _fail(f"live {label} directory is unavailable")
            head, status_sha = _git_status_sha256(directory)
            if head != commit or status_sha != identity[status_key]:
                _fail(f"live {label} revision/diff differs from runtime identity")
        overlay = dreamzero_dir / "groot/vla/model/dreamzero/action_head/wan_flow_matching_action_tf.py"
        if not overlay.is_file() or sha256_file(overlay) != OVERLAY_TARGET_SHA256:
            _fail("live DreamZero checkout is not the exact V2-A015 s=2 overlay")
        manifest = _load_object(study_root / CHECKPOINT_MANIFEST_RELATIVE)
        for root_key, section_key in (
            ("checkpoint_path", "checkpoint"), ("tokenizer_path", "tokenizer")
        ):
            payload_root = Path(str(identity.get(root_key, "")))
            if not payload_root.is_dir():
                _fail(f"live DreamZero {root_key} is unavailable")
            for row in manifest[section_key]["files"]:
                file_path = payload_root / row["path"]
                if (
                    not file_path.is_file()
                    or file_path.stat().st_size != row["bytes"]
                    or sha256_file(file_path) != row["sha256"]
                ):
                    _fail(f"live DreamZero payload mismatch: {section_key}/{row['path']}")
        lock = Path(str(identity.get("environment_lock_path", "")))
        if not lock.is_file() or sha256_file(lock) != identity["environment_lock_hash"]:
            _fail("live DreamZero environment lock hash mismatch")
    return identity


def _validate_server_contract(path: Path) -> dict[str, Any]:
    contract = _load_object(path)
    expected = {
        "schema_version": "vla-wam-shared-v2-dreamzero-v2a015-server-contract-v1",
        "amendment_id": "V2-A015",
        "official_repository_commit": FROZEN_SOURCE_COMMIT,
        "world_size": 2,
        "official_noise_seed": OFFICIAL_NOISE_SEED,
        "enable_dit_cache": True,
        "runtime_num_inference_steps": RUNTIME_NUM_INFERENCE_STEPS,
        "evaluated_dit_steps_with_cache": EVALUATED_DIT_STEPS,
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": VIDEO_CFG_SCALE,
    }
    for key, wanted in expected.items():
        if contract.get(key) != wanted:
            _fail(f"DreamZero live server contract mismatch for {key}")
    port = contract.get("port")
    if type(port) is not int or port <= 0 or port == 5000:
        _fail("DreamZero s=2 server must use an isolated non-5000 port")
    future_root = contract.get("future_root")
    if not isinstance(future_root, str) or not future_root.strip():
        _fail("DreamZero server contract requires a future root")
    return contract


def validate_release_gate(
    path: Path, *, queue_sha256: str, runtime_identity_sha256: str
) -> dict[str, Any]:
    gate = _load_object(path)
    expected = {
        "schema_version": GATE_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "identity_binding": IDENTITY_BINDING,
        "phase": PHASE,
        "phase_a_queue_sha256": queue_sha256,
        "runtime_identity_sha256": runtime_identity_sha256,
        "left_prompt": PROMPTS["left"],
        "right_prompt": PROMPTS["right"],
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": VIDEO_CFG_SCALE,
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "model_blind_neutral_reset_fixture_passed": True,
        "raw_video_action_jsonl_write_passed": True,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "all_exposed_futures_retained": True,
        "official_full_reset_decode_passed": True,
        "behavioral_release": True,
        "action_shape": ACTION_CHUNK_SHAPE,
    }
    for key, wanted in expected.items():
        if gate.get(key) != wanted:
            _fail(f"DreamZero s=2 release gate mismatch for {key}")
    if gate.get("left_exact_repeat_action_bit_identical") is not True:
        _fail("DreamZero release requires bit-identical repeat LEFT actions")
    if gate.get("left_exact_repeat_latent_bit_identical") is not True:
        _fail("DreamZero release requires bit-identical repeat LEFT latent futures")
    for key in ("left_right_action_rms", "left_right_latent_rms"):
        value = gate.get(key)
        if type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) <= 0:
            _fail(f"DreamZero release gate requires positive {key}")
    for key in (
        "left_action_sha256", "right_action_sha256", "left_latent_sha256",
        "right_latent_sha256", "gate_artifact_sha256", "server_contract_sha256",
    ):
        if not _is_sha256(gate.get(key)):
            _fail(f"DreamZero release gate requires {key}")
    if gate["left_action_sha256"] == gate["right_action_sha256"]:
        _fail("DreamZero LEFT and RIGHT action gate hashes must differ")
    if gate["left_latent_sha256"] == gate["right_latent_sha256"]:
        _fail("DreamZero LEFT and RIGHT latent gate hashes must differ")
    server_path = Path(str(gate.get("server_contract_path", "")))
    if not server_path.is_file() or sha256_file(server_path) != gate["server_contract_sha256"]:
        _fail("DreamZero release gate server contract hash/path mismatch")
    contract = _validate_server_contract(server_path)
    if gate.get("future_root") != contract["future_root"]:
        _fail("DreamZero release gate and server future roots differ")
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
        study_root, runtime_identity_path,
        check_live_repositories=check_live_repositories,
    )
    queue_sha = sha256_file(study_root / QUEUE_RELATIVE)
    runtime_sha = sha256_file(runtime_identity_path)
    gate = validate_release_gate(
        release_gate_path,
        queue_sha256=queue_sha,
        runtime_identity_sha256=runtime_sha,
    )
    return {
        "status": "ready",
        "seed": seed,
        "pair_id": cells[0]["pair_id"],
        "cell_ids": [row["cell_id"] for row in cells],
        "identity_binding": IDENTITY_BINDING,
        "baseline_s1_rejected": True,
        "runtime_identity": runtime,
        "runtime_identity_sha256": runtime_sha,
        "release_gate": gate,
        "release_gate_sha256": sha256_file(release_gate_path),
        "phase_a_queue_sha256": queue_sha,
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
) -> list[str]:
    if remote_port == 5000:
        _fail("protected pre-existing DreamZero port 5000 is prohibited")
    return [
        sys.executable,
        str(study_root / "experiments/v3/dreamzero_droid/robolab_bridge.py"),
        "--study-root", str(study_root),
        "--environment-seed", str(seed),
        "--sampling-seed", str(seed),
        "--runtime-identity", str(runtime_identity_path),
        "--release-gate", str(release_gate_path),
        "--state-capture-dir", str(output_dir / "state_capture"),
        "--action-trace-dir", str(action_trace_dir),
        "--remote-host", remote_host,
        "--remote-port", str(remote_port),
        "--simulator-lane", "raytrace-rtxpro6000-ali",
        "--open-loop-horizon", "8",
        "--instruction-controller", "static",
        "--condition", "both",
        "--output-dir", str(output_dir),
        "--num-envs", "1",
        "--num-runs", "1",
        "--device", "cuda:0",
        "--video-mode", "viewport",
        "--disable-subtask",
    ]


def _file_record(path: Path, *, nonempty: bool = True) -> dict[str, Any]:
    if not path.is_file() or (nonempty and path.stat().st_size <= 0):
        _fail(f"required retained DreamZero artifact is absent/empty: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_file_entry(entry: Any, label: str) -> Path:
    if not isinstance(entry, dict):
        _fail(f"DreamZero {label} must be a file record")
    path = Path(str(entry.get("path", "")))
    if (
        not path.is_file()
        or entry.get("sha256") != sha256_file(path)
        or ("bytes" in entry and entry.get("bytes") != path.stat().st_size)
    ):
        _fail(f"DreamZero {label} file/hash record is invalid")
    return path


def _validate_future_manifest(
    path: Path, *, prompt: str, request_count: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_object(path)
    expected = {
        "schema_version": "vla-wam-shared-v2-dreamzero-v2a015-future-retention-v1",
        "amendment_id": "V2-A015",
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": VIDEO_CFG_SCALE,
    }
    for key, wanted in expected.items():
        if manifest.get(key) != wanted:
            _fail(f"DreamZero future manifest is not exact s=2 for {key}")
    requests = manifest.get("requests")
    if not isinstance(requests, list) or len(requests) != request_count:
        _fail("DreamZero future manifest request count mismatch")
    retained: list[dict[str, Any]] = []
    for index, request in enumerate(requests):
        if (
            not isinstance(request, dict)
            or request.get("prompt") != prompt
            or request.get("action_cfg_style_scale") != ACTION_CFG_STYLE_SCALE
        ):
            _fail(f"DreamZero future request {index} changed prompt/s=2 identity")
        action_path = _validate_file_entry(
            request.get("returned_action"), f"future request {index} returned action"
        )
        latent_path = _validate_file_entry(
            request.get("latent_video"), f"future request {index} latent video"
        )
        retained.append({
            "request_index": index,
            "returned_action": _file_record(action_path),
            "latent_video": _file_record(latent_path),
        })
    decoded = manifest.get("official_reset_decode")
    if not isinstance(decoded, list) or not decoded:
        _fail("DreamZero future manifest lacks the official full reset decode")
    for index, entry in enumerate(decoded):
        decoded_path = _validate_file_entry(entry, f"official reset decode {index}")
        retained.append({
            "official_reset_decode_index": index,
            "official_reset_decode": _file_record(decoded_path),
        })
    return manifest, retained


def _validate_action_and_future_trace(
    path: Path, cell: dict[str, Any], actions_executed: int
) -> tuple[dict[str, Any], Path, Path, Path, Path, list[dict[str, Any]]]:
    trace = _load_object(path)
    expected = {
        "schema_version": "vla-wam-shared-v3-dreamzero-s2-action-trace-v1",
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "identity_binding": IDENTITY_BINDING,
        "environment_seed": cell["environment_seed"],
        "sampling_seed_label": cell["sampling_seed"],
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "prompt": cell["prompt"],
        "requested_relation": cell["relation"],
        "prompt_controller": "episode_static",
        "checkpoint": FROZEN_CHECKPOINT["id"],
        "checkpoint_revision": FROZEN_CHECKPOINT["revision"],
        "official_repository_commit": FROZEN_SOURCE_COMMIT,
        "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        "video_cfg_scale": VIDEO_CFG_SCALE,
        "open_loop_execution_horizon": OPEN_LOOP_HORIZON,
    }
    for key, wanted in expected.items():
        if trace.get(key) != wanted:
            _fail(f"DreamZero s=2 action trace mismatch for {key}")
    if trace.get("sampling_seed_semantics") != (
        "registered matched-pair label; released checkpoint noise remains fixed at 1140"
    ):
        _fail("DreamZero registered/effective seed semantics are not disclosed")
    executed_entry = trace.get("executed_actions", {})
    action_path = _validate_file_entry(executed_entry, "executed actions")
    if (
        executed_entry.get("count") != actions_executed
        or executed_entry.get("shape") != [actions_executed, 8]
        or executed_entry.get("dtype") != "float32"
    ):
        _fail("DreamZero executed action count/shape/dtype mismatch")
    raw_entry = trace.get("returned_raw_chunks", {})
    raw_path = _validate_file_entry(raw_entry, "returned raw chunks")
    request_count = trace.get("request_count")
    if type(request_count) is not int or request_count <= 0:
        _fail("DreamZero trace requires a positive policy request count")
    if (
        raw_entry.get("count") != request_count
        or raw_entry.get("shape") != [request_count, 24, 8]
        or raw_entry.get("dtype") != "float32"
    ):
        _fail("DreamZero raw action chunks do not match the request count")
    executable_entry = trace.get("returned_executable_chunks", {})
    executable_path = _validate_file_entry(
        executable_entry, "returned executable chunks"
    )
    if (
        executable_entry.get("count") != request_count
        or executable_entry.get("shape") != [request_count, 24, 8]
        or executable_entry.get("dtype") != "float32"
    ):
        _fail("DreamZero executable action chunks do not match the request count")
    future_entry = trace.get("future_manifest", {})
    future_path = _validate_file_entry(future_entry, "future manifest")
    if (
        future_entry.get("request_count") != request_count
        or type(future_entry.get("official_decode_count")) is not int
        or future_entry["official_decode_count"] <= 0
    ):
        _fail("DreamZero action trace lacks complete future/decode counts")
    _, retained = _validate_future_manifest(
        future_path, prompt=cell["prompt"], request_count=request_count
    )
    observed_decode_count = sum(
        "official_reset_decode" in artifact for artifact in retained
    )
    if future_entry["official_decode_count"] != observed_decode_count:
        _fail("DreamZero declared and retained official decode counts differ")
    return (
        trace,
        action_path,
        raw_path,
        executable_path,
        future_path,
        retained,
    )


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
    opposite = next((i for i, sample in enumerate(samples) if _in_cone(sample, opposite_relation)), None)
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


def build_behavioral_record(
    study_root: Path,
    cell: dict[str, Any],
    capture: dict[str, Any],
    runtime_identity_path: Path,
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
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "identity_binding": IDENTITY_BINDING,
        "prompt": cell["prompt"],
        "requested_relation": relation,
    }
    for key, wanted in expected.items():
        if capture.get(key) != wanted:
            _fail(f"DreamZero s=2 state capture mismatch for {key}")
    samples = capture.get("samples")
    actions_executed = capture.get("actions_executed")
    if type(actions_executed) is not int or actions_executed < 0:
        _fail("DreamZero capture requires non-negative actions_executed")
    if not isinstance(samples, list) or len(samples) != actions_executed + 1:
        _fail("DreamZero capture must retain initial plus every post-action state")
    if capture.get("behavioral_result_valid_candidate") is not True:
        _fail("partial DreamZero capture cannot enter a behavioral denominator")
    if capture.get("action_cap") != 450:
        _fail("DreamZero uses the frozen 450-action cap")
    if capture.get("requested_success") is True:
        if capture.get("right_censored") is not False:
            _fail("successful DreamZero episode cannot be right-censored")
    elif capture.get("right_censored") is not True or actions_executed != 450:
        _fail("valid DreamZero failure must run to the 450-action cap")
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict) or sample.get("action_step") != index:
            _fail("DreamZero state samples are not contiguous")
    runtime = validate_runtime_identity(study_root, runtime_identity_path)
    (
        _,
        action_path,
        raw_chunks_path,
        executable_chunks_path,
        future_path,
        retained_futures,
    ) = _validate_action_and_future_trace(action_trace_metadata_path, cell, actions_executed)
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
        "pair_id": cell["pair_id"],
        "prompt": cell["prompt"],
        "prompt_family": cell["prompt_family"],
        "predicate_id": cell["success_predicate_id"],
        "reset_id": cell["reset_identity"],
        "environment_seed": cell["environment_seed"],
        "policy_seed": cell["sampling_seed"],
        "effective_official_model_noise_seed": OFFICIAL_NOISE_SEED,
        "policy_seed_semantics": (
            "registered matched-pair label; released checkpoint noise remains fixed at 1140"
        ),
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
        "artifacts": {
            "viewport_video": _file_record(video_path),
            "executed_action_trace": _file_record(action_path),
            "returned_raw_action_chunks": _file_record(raw_chunks_path),
            "returned_executable_action_chunks": _file_record(executable_chunks_path),
            "action_trace_manifest": _file_record(action_trace_metadata_path),
            "exposed_future_manifest": _file_record(future_path),
            "raw_result_jsonl": {
                "path": str(raw_jsonl_path),
                "integrity_scope": "batch_manifest_after_close",
            },
        },
        "steps": steps,
        "actions_executed": actions_executed,
        "action_cap": 450,
        "right_censored": capture["right_censored"],
        "first_contact_step": capture.get("first_contact_step"),
        "first_contact_unavailable_reason": capture.get("first_contact_unavailable_reason"),
        "final_detached_release": capture["final_detached_release"],
        "wall_time_s": capture["wall_time_s"],
        "operational_wall_time_valid": capture["operational_wall_time_valid"],
        "event_timeline": _event_timeline(capture, relation),
        "dreamzero_identity": {
            "binding": IDENTITY_BINDING,
            "baseline_s1_used": False,
            "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
            "video_cfg_scale": VIDEO_CFG_SCALE,
            "negative_branch_caveat": (
                "Derived CFG-style fixed-negative-prompt action guidance; not an "
                "official DreamZero action-CFG feature."
            ),
        },
        "future_evidence": {
            "interface": "joint action and latent video with official reset decode",
            "missing_futures_scored_as_zero": False,
            "retained_artifacts": retained_futures,
        },
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
        _fail("unexpected DreamZero infrastructure capture schema")
    for key, wanted in (
        ("registered_cell_id", cell["cell_id"]),
        ("environment_seed", cell["environment_seed"]),
        ("policy_seed", cell["sampling_seed"]),
        ("prompt", cell["prompt"]),
        ("requested_relation", cell["relation"]),
        ("identity_binding", IDENTITY_BINDING),
    ):
        if capture.get(key) != wanted:
            _fail(f"DreamZero infrastructure capture mismatch for {key}")
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
        "dreamzero_identity": {
            "binding": IDENTITY_BINDING,
            "baseline_s1_used": False,
            "action_cfg_style_scale": ACTION_CFG_STYLE_SCALE,
        },
    }
    sys.path.insert(0, str(study_root / "tools"))
    from vla_wam_v3_episode_schema import validate_infrastructure_record  # type: ignore
    return validate_infrastructure_record(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="mode", required=True)
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
    plan.add_argument("--remote-port", type=int, required=True)
    behavioral = commands.add_parser("compile-behavioral")
    for argument in ("study-root", "runtime-identity", "capture", "video", "action-trace", "output-jsonl"):
        behavioral.add_argument(f"--{argument}", type=Path, required=True)
    behavioral.add_argument("--seed", type=int, required=True)
    behavioral.add_argument("--relation", choices=["left", "right"], required=True)
    infrastructure = commands.add_parser("compile-infrastructure")
    for argument in ("study-root", "runtime-identity", "capture", "output-jsonl"):
        infrastructure.add_argument(f"--{argument}", type=Path, required=True)
    infrastructure.add_argument("--video", type=Path)
    infrastructure.add_argument("--action-trace", type=Path)
    infrastructure.add_argument("--seed", type=int, required=True)
    infrastructure.add_argument("--relation", choices=["left", "right"], required=True)
    args = parser.parse_args()
    root = args.study_root.resolve()
    if args.mode.startswith("compile-"):
        cell = {row["relation"]: row for row in load_authorized_pair(root, args.seed)}[args.relation]
        capture = _load_object(args.capture)
        if args.mode == "compile-behavioral":
            result = build_behavioral_record(
                root, cell, capture, args.runtime_identity, args.video,
                args.action_trace, args.output_jsonl,
            )
        else:
            result = build_infrastructure_record(
                root, cell, capture, args.runtime_identity, args.output_jsonl,
                video_path=args.video, action_trace_path=args.action_trace,
            )
        sys.path.insert(0, str(root / "tools"))
        from vla_wam_v3_episode_schema import write_jsonl  # type: ignore
        args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        print(json.dumps(write_jsonl(args.output_jsonl, [result]), indent=2, sort_keys=True))
        return
    result = preflight(
        root, args.seed, args.runtime_identity, args.release_gate,
        check_live_repositories=args.check_live_repositories,
    )
    if args.mode == "plan":
        result["command"] = bridge_command(
            root, args.seed, args.runtime_identity, args.release_gate,
            args.output_dir, args.action_trace_dir, args.remote_host, args.remote_port,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
