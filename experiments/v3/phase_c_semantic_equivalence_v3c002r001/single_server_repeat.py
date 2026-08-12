#!/usr/bin/env python3
"""Collect one lane's excluded same-process L-R-L exact-repeat probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
from openpi_client import websocket_client_policy

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (
    ContractError,
    file_binding,
    registered_prompts,
    sha256_file,
)
from experiments.v3.phase_c_semantic_equivalence_v3c002.runtime import validate_runtime_manifest
from .contract import SMOKE_GATE_SCHEMA, read_finite_json


PROBE_SEED = 13_000_000
SEQUENCE = ("canonical_left", "canonical_right", "canonical_left")


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-registration", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--excluded-smoke-gate", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--runtime-manifest-sha256", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fixture-sha256", required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, required=True)
    parser.add_argument("--lane-slot", required=True)
    parser.add_argument("--lane-pod-uid", required=True)
    parser.add_argument("--lane-gpu-uuid", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ContractError(f"refusing to overwrite repeat probe: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "repeat_response.json"
    records: list[dict] = []
    dispatched = 0
    successful = 0
    runtime: dict = {}
    base = {
        "schema_version": "vla-wam-shared-v3c002r001-single-server-repeat-response-v1",
        "repair_id": "V3-C002-R001",
        "lane_slot": args.lane_slot,
        "behavioral_episode_count": 0,
        "excluded_from_behavioral_denominators": True,
        "probe_seed": PROBE_SEED,
        "sequence": list(SEQUENCE),
        "records": records,
    }
    try:
        smoke = read_finite_json(args.excluded_smoke_gate)
        if not isinstance(smoke, dict) or smoke.get("schema_version") != SMOKE_GATE_SCHEMA or smoke.get("status") != "passed_repair_excluded_four_cell_smoke" or smoke.get("passed") is not True:
            raise ContractError("repair excluded smoke did not pass")
        smoke_fixture = smoke.get("repeat_fixture", {})
        if smoke_fixture.get("sha256") != args.fixture_sha256 or smoke_fixture.get("bytes") != args.fixture.stat().st_size:
            raise ContractError("repeat fixture is not the retained global-smoke request-zero fixture")
        runtime = validate_runtime_manifest(
            args.runtime_manifest,
            args.runtime_manifest_sha256,
            registration_path=args.parent_registration,
            queue_path=args.queue,
            pod_uid=args.lane_pod_uid,
            gpu_uuid=args.lane_gpu_uuid,
        )["runtime_identity"]
        if runtime.get("server_port") != args.remote_port:
            raise ContractError("repeat endpoint differs from runtime")
        if not args.fixture.is_file() or sha256_file(args.fixture) != args.fixture_sha256:
            raise ContractError("repeat fixture changed")
        with np.load(args.fixture, allow_pickle=False) as loaded:
            observation = {key: loaded[key] for key in loaded.files}
        prompts = registered_prompts()
        client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)
        for ordinal, condition in enumerate(SEQUENCE):
            dispatched += 1
            response = client.infer({**observation, "prompt": prompts[condition]["prompt"], "sampling_seed": PROBE_SEED})
            successful += 1
            if response.get("v2a010_sampling_seed") != PROBE_SEED:
                raise ContractError("repeat seed echo changed")
            actions = np.asarray(response.get("actions"), dtype=np.float32)
            if actions.shape != (15, 8) or not np.isfinite(actions).all():
                raise ContractError("repeat response is not finite [15,8]")
            path = args.output_dir / f"request{ordinal}_{condition}.npy"
            np.save(path, actions, allow_pickle=False)
            records.append({"ordinal": ordinal, "condition": condition, "actions": file_binding(path), "seed_echo": PROBE_SEED})
        arrays = [np.load(record["actions"]["path"], allow_pickle=False) for record in records]
        exact = bool(np.array_equal(arrays[0], arrays[2]))
        sensitive = bool(not np.array_equal(arrays[0], arrays[1]))
        if not exact or not sensitive:
            raise ContractError("single-server repeat/sensitivity gate failed")
        value = {
            **base,
            "status": "completed_excluded_single_server_interleaved_repeat",
            "passed": True,
            "model_request_count": dispatched,
            "successful_response_count": successful,
            "fixture_sha256": args.fixture_sha256,
            "fixture": file_binding(args.fixture),
            "first_final_repeat_exact": exact,
            "prompt_sensitivity_distinct": sensitive,
            "runtime_manifest": file_binding(args.runtime_manifest),
            "excluded_smoke_gate": file_binding(args.excluded_smoke_gate),
        }
        for key in ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
            value[key] = runtime[key]
        _write(output, value)
        print(json.dumps({"status": value["status"], "sha256": sha256_file(output)}, sort_keys=True))
    except BaseException as exc:
        failure = {
            **base,
            "status": "failed_excluded_single_server_repeat_retained",
            "passed": False,
            "model_request_count": dispatched,
            "successful_response_count": successful,
            "first_final_repeat_exact": False,
            "prompt_sensitivity_distinct": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        for key in ("policy_server_pod_uid", "policy_server_gpu_uuid", "server_port", "server_process_identity", "server_lock_identity"):
            if key in runtime:
                failure[key] = runtime[key]
        _write(output, failure)
        raise


if __name__ == "__main__":
    main()
