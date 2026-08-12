#!/usr/bin/env python3
"""Target-side raw rehash for both V3-C002 zero-request physical proofs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.v3.phase_c_semantic_equivalence_v3c002.contract import (  # noqa: E402
    ContractError,
    file_binding,
    read_finite_json,
    sha256_file,
)


def _rehash_bindings(value: Any, *, label: str, seen: set[tuple[str, str]]) -> tuple[int, int]:
    count = 0
    byte_count = 0
    if isinstance(value, dict):
        if set(("path", "sha256", "bytes")).issubset(value):
            path = Path(str(value["path"]))
            if not path.is_absolute() or not path.is_file():
                raise ContractError(f"{label} target artifact is absent or not absolute: {path}")
            if value["bytes"] != path.stat().st_size or value["sha256"] != sha256_file(path):
                raise ContractError(f"{label} target artifact bytes/hash changed: {path}")
            key = (str(path.resolve()), str(value["sha256"]))
            if key not in seen:
                seen.add(key)
                count += 1
                byte_count += path.stat().st_size
            return count, byte_count
        for key, item in value.items():
            child_count, child_bytes = _rehash_bindings(item, label=f"{label}.{key}", seen=seen)
            count += child_count
            byte_count += child_bytes
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_count, child_bytes = _rehash_bindings(item, label=f"{label}[{index}]", seen=seen)
            count += child_count
            byte_count += child_bytes
    return count, byte_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone-report", type=Path, required=True)
    parser.add_argument("--standalone-invocation", type=Path, required=True)
    parser.add_argument("--same-process-report", type=Path, required=True)
    parser.add_argument("--same-process-invocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ContractError(f"refusing to overwrite target-side rehash receipt: {args.output}")
    standalone = read_finite_json(args.standalone_report)
    same_process = read_finite_json(args.same_process_report)
    if not isinstance(standalone, dict) or standalone.get("schema_version") != "vla-wam-shared-v3e004-standalone-model-blind-droid-gate-v2" or standalone.get("status") != "passed_model_blind_preflight_not_a_behavioral_release":
        raise ContractError("standalone E004 report is not the required passed zero-request gate")
    if not isinstance(same_process, dict) or same_process.get("schema_version") != "vla-wam-shared-v3c002-same-process-model-blind-adapter-gate-v1" or same_process.get("status") != "passed_same_process_gate_stopped_before_query_server":
        raise ContractError("same-process C002 adapter report is not the required passed zero-request gate")
    for label, value in (("standalone", standalone), ("same-process", same_process)):
        if value.get("passed") is not True:
            raise ContractError(f"{label} preflight did not pass")
        request_count = value.get("model_request_count", value.get("model_requests"))
        episode_count = value.get("behavioral_episode_count", value.get("behavioral_episodes"))
        action_count = value.get("behavioral_action_count", value.get("behavioral_actions"))
        if request_count != 0 or episode_count != 0 or action_count != 0:
            raise ContractError(f"{label} preflight was not model blind")
    if same_process.get("same_process_gate_completed_before_query_server") is not True or same_process.get("query_server_entry_count") != 0:
        raise ContractError("same-process report did not prove the pre-query stop")
    seen: set[tuple[str, str]] = set()
    count_a, bytes_a = _rehash_bindings(standalone, label="standalone", seen=seen)
    count_b, bytes_b = _rehash_bindings(same_process, label="same_process", seen=seen)
    invocation_records = []
    for label, path in (("standalone", args.standalone_invocation), ("same_process", args.same_process_invocation)):
        if not path.is_absolute() or not path.is_file() or path.stat().st_size <= 0:
            raise ContractError(f"{label} invocation is missing on the target")
        record = file_binding(path)
        invocation_records.append({"label": label, **record})
        key = (record["path"], record["sha256"])
        if key not in seen:
            seen.add(key)
            count_b += 1
            bytes_b += int(record["bytes"])
    value = {
        "schema_version": "vla-wam-shared-v3c002-target-raw-rehash-receipt-v1",
        "status": "passed_target_side_raw_rehash",
        "passed": True,
        "standalone_report": file_binding(args.standalone_report),
        "same_process_adapter_report": file_binding(args.same_process_report),
        "standalone_report_sha256": sha256_file(args.standalone_report),
        "same_process_adapter_report_sha256": sha256_file(args.same_process_report),
        "invocations": invocation_records,
        "unique_raw_bindings_rehashed": len(seen),
        "raw_bytes_rehashed": bytes_a + bytes_b,
        "model_requests": 0,
        "behavioral_episodes": 0,
        "behavioral_actions": 0,
        "validation_location": "target simulator PVC/runtime with direct access to absolute raw paths",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "sha256": sha256_file(args.output), "raw_bindings": len(seen)}, sort_keys=True))


if __name__ == "__main__":
    main()
