#!/usr/bin/env python3
"""Build and validate the V3-B002 pi0.5 runtime and release gates.

The fixture/reset/renderer/writer section is strictly model blind.  The fixed
observation repeat/sensitivity section is model-facing but executes no
behavioral episode.  Keeping these sections distinct prevents a successful
policy probe from laundering a failed physical preflight.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

from experiments.v3.pi05_phase_b.contract import (
    ACTION_CAP,
    ACTION_CHUNK_STEPS,
    ACTION_DIM,
    ACTION_SPACE,
    AMENDMENT_ID,
    CHECKPOINT_MANIFEST_SHA256,
    GATE_SCHEMA,
    MODEL_ID,
    OPENPI_COMMIT,
    OPENPI_CONFIG,
    PROMPTS,
    ROBOLAB_COMMIT,
    RUNTIME_SCHEMA,
    STUDY_ID,
    ContractError,
    ReleaseBundle,
    canonical_json_bytes,
    load_release_bundle,
    sha256_bytes,
    sha256_file,
)


EMPTY_SHA256 = sha256_bytes(b"")
BASE_RUNTIME_SCHEMA = "vla-wam-shared-v3-pi05-current-runtime-identity-v1"
MODEL_BLIND_SCHEMA = "vla-wam-shared-v3b-pi05-model-blind-preflight-v1"
FIXED_OBSERVATION_SCHEMA = "vla-wam-shared-v3b-pi05-fixed-observation-gate-v1"
ADAPTER_FILES = (
    "experiments/v3/pi05_phase_b/contract.py",
    "experiments/v3/pi05_phase_b/runtime.py",
    "experiments/v3/pi05_phase_b/diagnostics.py",
    "experiments/v3/pi05_phase_b/client.py",
    "experiments/v3/pi05_phase_b/robolab_bridge.py",
    "experiments/v3/pi05_phase_b/compile_cell.py",
    "experiments/v3/pi05_phase_b/compiler.py",
    "experiments/v3/pi05_phase_b/queue.py",
    "experiments/v3/pi05_phase_b/fixture_tasks.py",
    "experiments/v3/pi05_phase_b/fixed_observation_probe.py",
    "experiments/v3/pi05_phase_b/model_blind_preflight.py",
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ContractError(f"expected one JSON object: {path}")
    return value


def adapter_contract_sha256(repo_root: Path) -> str:
    root = Path(repo_root).resolve()
    rows = []
    for relative in ADAPTER_FILES:
        path = root / relative
        if not path.is_file():
            raise ContractError(f"missing V3-B002 source: {relative}")
        rows.append({"path": relative, "sha256": sha256_file(path)})
    wrappers = sorted((root / "experiments/v3/pi05_phase_b/task_files").glob("*.py"))
    if len(wrappers) != 5:  # package initializer plus four task wrappers
        raise ContractError("V3-B002 requires exactly four task wrappers")
    rows.extend(
        {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
        for path in wrappers
    )
    return sha256_bytes(canonical_json_bytes(rows))


def current_git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(repo_root), check=True,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    value = result.stdout.strip()
    if len(value) != 40:
        raise ContractError("could not bind the current study Git commit")
    return value


def require_clean_tracked_checkout(repo_root: Path) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=Path(repo_root), check=True, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout:
        raise ContractError("runtime requires a clean tracked study checkout")


def validate_live_topology(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a captured ali-owned policy/lane topology without guessing IDs."""

    if value.get("schema_version") != "vla-wam-shared-v3b-pi05-live-topology-v1":
        raise ContractError("unexpected live topology schema")
    policy = value.get("policy_server")
    lanes = value.get("simulator_lanes")
    if not isinstance(policy, dict) or not isinstance(lanes, list) or not lanes:
        raise ContractError("live topology requires one policy server and simulator lanes")
    for label, row in [("policy_server", policy), *[("simulator_lane", item) for item in lanes]]:
        if not isinstance(row, dict) or row.get("owner") != "ali":
            raise ContractError(f"{label} is not explicitly ali-owned")
        for key in ("pod", "pod_uid", "pod_ip", "gpu_uuid", "gpu_model", "driver_version"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ContractError(f"{label} lacks {key}")
        if not row["gpu_uuid"].startswith("GPU-"):
            raise ContractError(f"{label} GPU UUID is invalid")
    if policy.get("port") != 8001 or policy.get("gpu_index") != 2:
        raise ContractError("policy server must bind the released B200 GPU2 endpoint on port 8001")
    if policy.get("model_request_count_at_capture") != 0:
        raise ContractError("policy endpoint was not captured before request zero")
    lane_names = [row["pod"] for row in lanes]
    if len(lane_names) != len(set(lane_names)):
        raise ContractError("simulator lane pod identities must be unique")
    return dict(value)


def validate_base_runtime(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": BASE_RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "model_id": MODEL_ID,
        "openpi_commit": OPENPI_COMMIT,
        "robolab_commit": ROBOLAB_COMMIT,
        "openpi_config": OPENPI_CONFIG,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ContractError(f"base pi0.5 runtime mismatch for {key}")
    for key in (
        "checkpoint_sha256",
        "environment_lock_hash",
        "external_repository_diff_hash",
        "openpi_dir_status_sha256",
        "robolab_dir_status_sha256",
    ):
        digest = value.get(key)
        if not isinstance(digest, str) or len(digest) != 64:
            raise ContractError(f"base runtime lacks SHA-256 {key}")
    return dict(value)


def build_runtime_identity(
    *, repo_root: Path, release: ReleaseBundle, base_runtime: Mapping[str, Any],
    live_topology: Mapping[str, Any], study_git_commit: str,
) -> dict[str, Any]:
    base = validate_base_runtime(base_runtime)
    topology = validate_live_topology(live_topology)
    require_clean_tracked_checkout(repo_root)
    if study_git_commit != current_git_commit(repo_root):
        raise ContractError("requested runtime commit is not the checked-out study HEAD")
    payload = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "openpi_commit": OPENPI_COMMIT,
        "robolab_commit": ROBOLAB_COMMIT,
        "openpi_config": OPENPI_CONFIG,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "checkpoint_sha256": base["checkpoint_sha256"],
        "environment_lock_sha256": base["environment_lock_hash"],
        "base_runtime_identity_sha256": sha256_bytes(canonical_json_bytes(base)),
        "external_repository_diff_hash": base["external_repository_diff_hash"],
        "openpi_dir_status_sha256": base["openpi_dir_status_sha256"],
        "robolab_dir_status_sha256": base["robolab_dir_status_sha256"],
        "release_manifest_sha256": release.manifest_sha256,
        "phase_b_adapter_contract_sha256": adapter_contract_sha256(repo_root),
        "study_git_commit": study_git_commit,
        "live_topology": topology,
        "live_topology_sha256": sha256_bytes(canonical_json_bytes(topology)),
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static_episode_prompt",
        "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
        "renderer_backend": base.get("renderer_backend"),
        "simulator_version": base.get("simulator_version"),
    }
    payload["runtime_identity_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_runtime_identity(
    value: Mapping[str, Any], *, repo_root: Path, release: ReleaseBundle
) -> dict[str, Any]:
    require_clean_tracked_checkout(repo_root)
    expected = {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "openpi_commit": OPENPI_COMMIT,
        "robolab_commit": ROBOLAB_COMMIT,
        "openpi_config": OPENPI_CONFIG,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "release_manifest_sha256": release.manifest_sha256,
        "phase_b_adapter_contract_sha256": adapter_contract_sha256(repo_root),
        "action_space": ACTION_SPACE,
        "action_chunk_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
        "open_loop_horizon": ACTION_CHUNK_STEPS,
        "action_cap": ACTION_CAP,
        "instruction_controller": "static_episode_prompt",
        "future_interface": "actions_only",
        "missing_future_policy": "action_only_interface_not_applicable_never_zero",
        "study_git_commit": current_git_commit(repo_root),
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ContractError(f"V3-B002 runtime mismatch for {key}")
    topology = validate_live_topology(value.get("live_topology", {}))
    if value.get("live_topology_sha256") != sha256_bytes(canonical_json_bytes(topology)):
        raise ContractError("live topology hash binding changed")
    claimed = value.get("runtime_identity_sha256")
    body = {key: child for key, child in value.items() if key != "runtime_identity_sha256"}
    if claimed != sha256_bytes(canonical_json_bytes(body)):
        raise ContractError("runtime_identity_sha256 does not bind the runtime fields")
    return dict(value)


def validate_model_blind_preflight(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": MODEL_BLIND_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
        "renderer_backend": "realtime RTX Vulkan",
        "all_required_rgb_views_nonblank": True,
        "viewport_writer_passed": True,
        "raw_jsonl_writer_passed": True,
        "action_trace_writer_passed": True,
        "fixture_positions_match": True,
        "neutral_reset_passed": True,
        "settle_steps": 60,
        "stable_window_steps": 15,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ContractError(f"model-blind B002 gate mismatch for {key}")
    tasks = value.get("tasks")
    if not isinstance(tasks, list) or {
        (row.get("arm"), row.get("relation")) for row in tasks if isinstance(row, dict)
    } != {(arm, relation) for arm in ("control", "position_mirrored") for relation in ("left", "right")}:
        raise ContractError("model-blind gate must cover all four cells")
    return dict(value)


def validate_fixed_observation_gate(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version": FIXED_OBSERVATION_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "left_prompt": PROMPTS["left"],
        "right_prompt": PROMPTS["right"],
        "left_exact_repeat_bit_identical": True,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "action_shape": [ACTION_CHUNK_STEPS, ACTION_DIM],
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ContractError(f"fixed-observation B002 gate mismatch for {key}")
    rms = value.get("left_right_action_rms")
    if isinstance(rms, bool) or not isinstance(rms, (int, float)) or not math.isfinite(rms) or rms <= 0:
        raise ContractError("fixed-observation LEFT/RIGHT RMS must be positive and finite")
    if value.get("left_action_sha256") == value.get("right_action_sha256"):
        raise ContractError("fixed-observation LEFT/RIGHT action hashes must differ")
    return dict(value)


def validate_lane_preflights(
    values: list[Mapping[str, Any]], *, runtime: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Require one fresh real-Isaac zero-request preflight per runtime lane."""

    rows = [validate_model_blind_preflight(value) for value in values]
    expected = {
        (lane["pod_uid"], lane["gpu_uuid"])
        for lane in runtime["live_topology"]["simulator_lanes"]
    }
    observed = {(row.get("pod_uid"), row.get("gpu_uuid")) for row in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ContractError("model-blind preflights must cover every runtime simulator lane exactly once")
    return rows


def build_release_gate(
    *,
    release: ReleaseBundle,
    runtime: Mapping[str, Any],
    model_blind_preflights: list[Mapping[str, Any]],
    fixed_observation_gate: Mapping[str, Any],
) -> dict[str, Any]:
    blind_rows = validate_lane_preflights(model_blind_preflights, runtime=runtime)
    fixed = validate_fixed_observation_gate(fixed_observation_gate)
    payload = {
        "schema_version": GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "release_manifest_sha256": release.manifest_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "live_topology_sha256": runtime["live_topology_sha256"],
        "model_blind_lane_count": len(blind_rows),
        "model_blind_preflight_sha256": [
            sha256_bytes(canonical_json_bytes(row)) for row in blind_rows
        ],
        "model_blind_lane_bindings": [
            {"pod": row["pod"], "pod_uid": row["pod_uid"], "gpu_uuid": row["gpu_uuid"]}
            for row in blind_rows
        ],
        "fixed_observation_gate_sha256": sha256_bytes(canonical_json_bytes(fixed)),
        "model_blind_fixture_reset_renderer_writer_passed": True,
        "model_blind_model_request_count": 0,
        "model_blind_behavioral_episode_count": 0,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "left_exact_repeat_bit_identical": True,
        "left_right_action_rms": fixed["left_right_action_rms"],
        "left_action_sha256": fixed["left_action_sha256"],
        "right_action_sha256": fixed["right_action_sha256"],
        "behavioral_release": True,
    }
    payload["release_gate_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_release_gate(
    value: Mapping[str, Any], *, release: ReleaseBundle, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "schema_version": GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "release_manifest_sha256": release.manifest_sha256,
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "live_topology_sha256": runtime["live_topology_sha256"],
        "model_blind_lane_count": len(runtime["live_topology"]["simulator_lanes"]),
        "model_blind_fixture_reset_renderer_writer_passed": True,
        "model_blind_model_request_count": 0,
        "model_blind_behavioral_episode_count": 0,
        "fixed_observation_exact_repeat_passed": True,
        "fixed_observation_left_right_prompt_sensitivity_passed": True,
        "left_exact_repeat_bit_identical": True,
        "behavioral_release": True,
    }
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise ContractError(f"B002 release gate mismatch for {key}")
    expected_lanes = {
        (lane["pod"], lane["pod_uid"], lane["gpu_uuid"])
        for lane in runtime["live_topology"]["simulator_lanes"]
    }
    bindings = value.get("model_blind_lane_bindings")
    if not isinstance(bindings, list) or {
        (row.get("pod"), row.get("pod_uid"), row.get("gpu_uuid"))
        for row in bindings if isinstance(row, dict)
    } != expected_lanes:
        raise ContractError("release gate does not bind every simulator-lane preflight")
    digests = value.get("model_blind_preflight_sha256")
    if not isinstance(digests, list) or len(digests) != len(expected_lanes) or any(
        not isinstance(digest, str) or len(digest) != 64 for digest in digests
    ):
        raise ContractError("release gate lacks one preflight digest per simulator lane")
    if not isinstance(value.get("fixed_observation_gate_sha256"), str) or len(value["fixed_observation_gate_sha256"]) != 64:
        raise ContractError("release gate lacks the fixed-observation gate digest")
    claimed = value.get("release_gate_sha256")
    body = {key: child for key, child in value.items() if key != "release_gate_sha256"}
    if claimed != sha256_bytes(canonical_json_bytes(body)):
        raise ContractError("release_gate_sha256 does not bind the gate fields")
    return dict(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-manifest-sha256", required=True)
    commands = parser.add_subparsers(dest="mode", required=True)
    runtime_cmd = commands.add_parser("build-runtime")
    runtime_cmd.add_argument("--base-runtime", type=Path, required=True)
    runtime_cmd.add_argument("--live-topology", type=Path, required=True)
    runtime_cmd.add_argument("--study-git-commit", required=True)
    runtime_cmd.add_argument("--output", type=Path, required=True)
    gate_cmd = commands.add_parser("build-gate")
    gate_cmd.add_argument("--runtime", type=Path, required=True)
    gate_cmd.add_argument("--model-blind-preflight", type=Path, nargs="+", required=True)
    gate_cmd.add_argument("--fixed-observation-gate", type=Path, required=True)
    gate_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    release = load_release_bundle(
        args.repo_root, args.release_manifest,
        expected_manifest_sha256=args.release_manifest_sha256,
    )
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.mode == "build-runtime":
        result = build_runtime_identity(
            repo_root=args.repo_root, release=release, base_runtime=_object(args.base_runtime),
            live_topology=_object(args.live_topology), study_git_commit=args.study_git_commit,
        )
    else:
        runtime = validate_runtime_identity(
            _object(args.runtime), repo_root=args.repo_root, release=release
        )
        result = build_release_gate(
            release=release,
            runtime=runtime,
            model_blind_preflights=[_object(path) for path in args.model_blind_preflight],
            fixed_observation_gate=_object(args.fixed_observation_gate),
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
