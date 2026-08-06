#!/usr/bin/env python3
"""Build the V3-B003 behavioral release from fresh zero-behavior gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.v3.dreamzero_droid.adapter import validate_runtime_identity
from experiments.v3.dreamzero_phase_b.contract import (
    AMENDMENT_ID,
    EXPECTED_SHA256,
    IDENTITY_BINDING,
    MODEL_ID,
    RELEASE_GATE_SCHEMA,
    STUDY_ID,
    sha256_file,
)


def record(path: Path, **extra):
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--server-contract", type=Path, required=True)
    parser.add_argument("--fixed-observation-probe", type=Path, required=True)
    parser.add_argument("--model-blind-lane", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.model_blind_lane) != 4:
        raise ValueError("V3-B003 release requires exactly four independent RTX lane gates")
    runtime = validate_runtime_identity(args.repo_root, args.runtime_identity)
    if runtime.get("identity_binding") != IDENTITY_BINDING:
        raise ValueError("runtime is not the frozen DreamZero s=2 identity")
    contract = json.loads(args.server_contract.read_text())
    if (
        contract.get("schema_version")
        != "vla-wam-shared-v2-dreamzero-v2a015-server-contract-v1"
        or contract.get("action_cfg_style_scale") != 2
        or contract.get("video_cfg_scale") != 5.0
        or contract.get("world_size") != 2
        or contract.get("port") == 5000
    ):
        raise ValueError("fresh DreamZero server contract is not exact V2-A015 s=2")
    fixed = json.loads(args.fixed_observation_probe.read_text())
    metrics = fixed.get("metrics", {})
    if (
        fixed.get("schema_version")
        != "vla-wam-shared-v2-dreamzero-v2a015-fixed-observation-probe-v1"
        or fixed.get("status") != "passed"
        or fixed.get("release_gate_passed") is not True
        or fixed.get("action_cfg_style_scale") != 2.0
        or metrics.get("left_exact_repeat_action_array_equal") is not True
        or metrics.get("left_exact_repeat_latent_tensor_equal") is not True
        or float(metrics.get("left_vs_right_action_rms", 0)) <= 0
        or float(metrics.get("left_vs_right_latent_rms", 0)) <= 0
        or len(fixed.get("records", {})) != 3
    ):
        raise ValueError("fresh DreamZero fixed-observation gate did not pass")
    if fixed.get("server_contract", {}).get("sha256") != sha256_file(args.server_contract):
        raise ValueError("fixed probe and supplied server contract differ")
    lanes = []
    seen = set()
    for path in args.model_blind_lane:
        value = json.loads(path.read_text())
        identity = (value.get("pod_uid"), value.get("gpu_uuid"))
        if (
            value.get("schema_version")
            != "vla-wam-shared-v3b-dreamzero-model-blind-preflight-v1"
            or value.get("amendment_id") != AMENDMENT_ID
            or value.get("model_id") != MODEL_ID
            or value.get("passed") is not True
            or value.get("model_request_count") != 0
            or value.get("behavioral_episode_count") != 0
            or value.get("all_required_rgb_views_nonblank") is not True
            or value.get("viewport_writer_passed") is not True
            or value.get("raw_jsonl_writer_passed") is not True
            or value.get("action_trace_writer_passed") is not True
            or len(value.get("tasks", [])) != 4
            or value.get("fresh_process_count") != 12
            or identity in seen
        ):
            raise ValueError(f"invalid or duplicate DreamZero model-blind lane: {path}")
        seen.add(identity)
        lanes.append(record(
            path,
            passed=True,
            pod=value["pod"],
            pod_uid=value["pod_uid"],
            gpu_uuid=value["gpu_uuid"],
        ))
    output = {
        "schema_version": RELEASE_GATE_SCHEMA,
        "study_id": STUDY_ID,
        "amendment_id": AMENDMENT_ID,
        "model_id": MODEL_ID,
        "identity_binding": IDENTITY_BINDING,
        "manifest_sha256": EXPECTED_SHA256["manifest"],
        "cells_sha256": EXPECTED_SHA256["cells"],
        "runtime_identity_sha256": sha256_file(args.runtime_identity),
        "runtime_identity": record(args.runtime_identity),
        "server_contract": record(args.server_contract),
        "fixed_observation_probe": record(args.fixed_observation_probe),
        "fixed_observation_release_passed": True,
        "model_blind_lanes": lanes,
        "all_model_blind_lanes_passed": True,
        "model_request_count_before_release": 3,
        "behavioral_episode_count_before_release": 0,
        "future_root": contract["future_root"],
        "remote_port": contract["port"],
        "left_exact_repeat_action_bit_identical": True,
        "left_exact_repeat_latent_bit_identical": True,
        "left_right_action_rms": metrics["left_vs_right_action_rms"],
        "left_right_latent_rms": metrics["left_vs_right_latent_rms"],
        "behavioral_release": True,
        "release_boundary": (
            "Only the exact 108 registered V3-B003 cells may run, in their frozen "
            "whole-seed order, on the four hash-bound simulator lanes."
        ),
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(record(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
