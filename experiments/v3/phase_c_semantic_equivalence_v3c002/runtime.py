"""Exact-runtime identity binding for a V3-C002 π0.5 lane.

This module is intentionally model- and simulator-free.  It verifies an
operator-produced runtime observation before the queue runner can dispatch the
first request; it does not start a server or contact a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .contract import AMENDMENT_ID, ARENA, MODEL_ID, STUDY_ID, ContractError, file_binding, read_finite_json, require, sha256_file


RUNTIME_SCHEMA = "vla-wam-shared-v3c002-pi05-lane-runtime-v1"


def bind_runtime(
    *,
    registration_path: Path,
    queue_path: Path,
    observed_runtime_path: Path,
    observed_runtime_sha256: str,
    lane_pod_uid: str,
    lane_gpu_uuid: str,
) -> dict[str, Any]:
    """Validate and bind a fresh lane observation to the frozen E004 π0.5 contract."""

    registration = read_finite_json(registration_path)
    require(isinstance(registration, dict), "registration is invalid")
    requirement = registration.get("runtime_identity_requirement")
    require(isinstance(requirement, dict), "registration runtime requirement is missing")
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
        ("openpi_commit", requirement["openpi_commit"]),
        ("robolab_commit", requirement["robolab_commit"]),
        ("action_dim", requirement["action_interface"]["action_dim"]),
        ("action_horizon", requirement["action_interface"]["action_horizon"]),
        ("action_cap", requirement["action_interface"]["action_cap"]),
        ("lane_pod_uid", lane_pod_uid),
        ("lane_gpu_uuid", lane_gpu_uuid),
    ):
        require(observed.get(key) == expected, f"observed runtime differs for {key}")
    for key in ("checkpoint_digest", "controller_digest", "action_interface_digest", "camera_configuration_digest", "horizon_digest", "scorer_digest"):
        value = observed.get(key)
        require(isinstance(value, str) and len(value) == 64, f"observed runtime lacks {key}")
    return {
        "schema_version": RUNTIME_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "status": "passed_exact_e004_pi05_runtime_identity_pre_request",
        "registration": file_binding(registration_path),
        "queue": file_binding(queue_path),
        "lane_pod_uid": lane_pod_uid,
        "lane_gpu_uuid": lane_gpu_uuid,
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
