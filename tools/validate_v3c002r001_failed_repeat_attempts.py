#!/usr/bin/env python3
"""Rehash the eight retained, infrastructure-invalid R001 repeat attempts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import file_binding, read_finite_json, sha256_file
from experiments.v3.phase_c_semantic_equivalence_v3c002r001.contract import ContractError


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or len(args.attempt_root) != 8:
        raise ContractError("failed-repeat receipt requires one new output and eight roots")
    attempts = []
    for index, root in enumerate(args.attempt_root):
        response_path = root / "response/repeat_response.json"
        response = read_finite_json(response_path)
        lane = f"repair-lane-{index:02d}"
        if (
            not isinstance(response, dict)
            or response.get("schema_version") != "vla-wam-shared-v3c002r001-single-server-repeat-response-v1"
            or response.get("status") != "failed_excluded_single_server_repeat_retained"
            or response.get("passed") is not False
            or response.get("lane_slot") != lane
            or response.get("model_request_count") != 1
            or response.get("successful_response_count") != 0
            or response.get("behavioral_episode_count") != 0
            or response.get("records") != []
            or "observation/gripper_position" not in str(response.get("error"))
        ):
            raise ContractError(f"failed repeat envelope changed for {lane}")
        files = [file_binding(path) for path in sorted(root.rglob("*")) if path.is_file()]
        if any(path.suffix == ".npy" for path in root.rglob("*")):
            raise ContractError(f"failed repeat unexpectedly retained an action array for {lane}")
        attempts.append({
            "lane_slot": lane,
            "root": str(root.resolve()),
            "response": file_binding(response_path),
            "files": files,
            "file_count": len(files),
            "bytes": sum(record["bytes"] for record in files),
            "model_request_count": 1,
            "successful_response_count": 0,
            "action_array_count": 0,
            "behavioral_episode_count": 0,
        })
    value = {
        "schema_version": "vla-wam-shared-v3c002r001-failed-repeat-target-rehash-v1",
        "status": "retained_eight_infrastructure_invalid_flat_cache_requests",
        "passed": True,
        "classification": "infrastructure_invalid_client_serialization_before_any_successful_response",
        "attempts": attempts,
        "model_request_count": 8,
        "successful_response_count": 0,
        "action_array_count": 0,
        "behavioral_action_count": 0,
        "behavioral_episode_count": 0,
        "retry_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
