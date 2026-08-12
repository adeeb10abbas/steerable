"""Exact-runtime identity binding for a V3-C002 π0.5 lane.

This module is intentionally model- and simulator-free.  It verifies an
operator-produced runtime observation before the queue runner can dispatch the
first request; it does not start a server or contact a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract import (
    AMENDMENT_ID, ARENA, MODEL_ID, STUDY_ID, ContractError, canonical_json_sha256,
    file_binding, read_finite_json, require, sha256_file, validate_exact_runtime_contract,
)


RUNTIME_SCHEMA = "vla-wam-shared-v3c002-pi05-lane-runtime-v2"


def bind_runtime(
    *,
    registration_path: Path,
    queue_path: Path,
    observed_runtime_path: Path,
    observed_runtime_sha256: str,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
    policy_server_pod_uid: str,
    policy_server_gpu_uuid: str,
    server_port: int,
    raw_root: str,
    container_identity: str,
    runtime_identity: str,
    lane_id: str,
    server_process_identity: str,
    server_lock_identity: str,
) -> dict[str, Any]:
    """Validate and bind a fresh lane observation to the frozen E004 π0.5 contract."""

    registration = read_finite_json(registration_path)
    require(isinstance(registration, dict), "registration is invalid")
    exact = registration.get("exact_e004_pi05_runtime")
    exact_sha = validate_exact_runtime_contract(exact)
    requirement = exact["identity_values"]
    observed_runtime_path = Path(observed_runtime_path).resolve()
    require(observed_runtime_path.is_file(), "observed runtime record is missing")
    require(sha256_file(observed_runtime_path) == observed_runtime_sha256, "observed runtime record changed")
    observed = read_finite_json(observed_runtime_path)
    require(isinstance(observed, dict), "observed runtime record must be an object")
    for key, expected in (
        ("model_id", MODEL_ID),
        ("arena", ARENA),
        ("checkpoint", requirement["checkpoint"]),
        ("checkpoint_manifest_sha256", requirement["checkpoint_manifest_sha256"]),
        ("checkpoint_digest", requirement["checkpoint_digest"]),
        ("openpi_commit", requirement["openpi_commit"]),
        ("robolab_commit", requirement["robolab_commit"]),
        ("action_dim", requirement["action_interface"]["action_dim"]),
        ("action_horizon", requirement["action_interface"]["action_horizon"]),
        ("action_cap", requirement["action_interface"]["action_cap"]),
        ("simulator_pod_uid", lane_pod_uid),
        ("simulator_gpu_uuid", lane_gpu_uuid),
        ("policy_server_pod_uid", policy_server_pod_uid),
        ("policy_server_gpu_uuid", policy_server_gpu_uuid),
        ("server_port", server_port),
        ("raw_root", raw_root),
        ("container_identity", container_identity),
        ("runtime_identity", runtime_identity),
        ("lane_id", lane_id),
        ("server_process_identity", server_process_identity),
        ("server_lock_identity", server_lock_identity),
        ("source_commit", requirement["source_commit"]),
        ("simulator_identity", requirement["simulator_identity"]),
        ("renderer_backend", requirement["renderer_backend"]),
        ("policy_cameras", requirement["policy_cameras"]),
        ("full_reset", True),
        ("stage_identifier", "full_reset"),
        ("exact_runtime_contract_sha256", exact_sha),
    ):
        require(observed.get(key) == expected, f"observed runtime differs for {key}")
    require(observed.get("component_digests") == exact["component_digests"], "observed component digests differ from exact E004 bindings")
    require(observed.get("dependency_bindings") == exact["dependency_bindings"], "observed source path/hash bindings differ from exact E004 bindings")
    camera_hashes = observed.get("policy_camera_image_artifact_hashes")
    require(isinstance(camera_hashes, dict) and set(camera_hashes) == set(requirement["policy_cameras"]), "policy camera/image hashes are incomplete")
    require(all(isinstance(value, str) and len(value) == 64 for value in camera_hashes.values()), "policy camera/image hashes are invalid")
    return {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "passed_exact_e004_pi05_runtime_identity_pre_request",
        "registration": file_binding(registration_path),
        "queue": file_binding(queue_path),
        "lane_pod_uid": lane_pod_uid,
        "lane_gpu_uuid": lane_gpu_uuid,
        "lane_id": lane_id,
        "policy_server_pod_uid": policy_server_pod_uid,
        "policy_server_gpu_uuid": policy_server_gpu_uuid,
        "server_port": server_port,
        "raw_root": raw_root,
        "container_identity": container_identity,
        "runtime_identity_label": runtime_identity,
        "server_process_identity": server_process_identity,
        "server_lock_identity": server_lock_identity,
        "exact_runtime_contract_sha256": exact_sha,
        "observed_runtime": file_binding(observed_runtime_path),
        "runtime_identity": observed,
        "no_model_request_or_behavioral_episode_in_this_runtime_binding": True,
    }


def validate_runtime_manifest(path: Path, expected_sha256: str, *, registration_path: Path, queue_path: Path, pod_uid: str, gpu_uuid: str) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file() and sha256_file(path) == expected_sha256, "lane runtime manifest changed")
    value = read_finite_json(path)
    require(isinstance(value, dict) and value.get("schema_version") == RUNTIME_SCHEMA, "lane runtime manifest schema changed")
    require(value.get("status") == "passed_exact_e004_pi05_runtime_identity_pre_request", "lane runtime manifest has not passed")
    require(value.get("lane_pod_uid") == pod_uid and value.get("lane_gpu_uuid") == gpu_uuid, "lane runtime identity differs")
    for name, source in (("registration", registration_path), ("queue", queue_path)):
        binding = value.get(name)
        require(isinstance(binding, Mapping) and binding.get("sha256") == sha256_file(source), f"runtime {name} binding differs")
    registration = read_finite_json(registration_path)
    exact_sha = validate_exact_runtime_contract(registration.get("exact_e004_pi05_runtime"))
    require(value.get("exact_runtime_contract_sha256") == exact_sha, "lane runtime exact contract differs")
    observed = value.get("runtime_identity")
    require(isinstance(observed, dict) and observed.get("exact_runtime_contract_sha256") == exact_sha, "observed lane runtime contract differs")
    return value


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--observed-runtime", type=Path, required=True)
    parser.add_argument("--observed-runtime-sha256", required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--policy-server-pod-uid", required=True)
    parser.add_argument("--policy-server-gpu-uuid", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--container-identity", required=True)
    parser.add_argument("--runtime-identity", required=True)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--server-process-identity", required=True)
    parser.add_argument("--server-lock-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite runtime manifest: {args.output}")
    value = bind_runtime(
        registration_path=args.registration,
        queue_path=args.queue,
        observed_runtime_path=args.observed_runtime,
        observed_runtime_sha256=args.observed_runtime_sha256,
        lane_pod_uid=args.lane_pod_uid,
        lane_gpu_uuid=args.lane_gpu_uuid,
        policy_server_pod_uid=args.policy_server_pod_uid,
        policy_server_gpu_uuid=args.policy_server_gpu_uuid,
        server_port=args.server_port,
        raw_root=args.raw_root,
        container_identity=args.container_identity,
        runtime_identity=args.runtime_identity,
        lane_id=args.lane_id,
        server_process_identity=args.server_process_identity,
        server_lock_identity=args.server_lock_identity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
