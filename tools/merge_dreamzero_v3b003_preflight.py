#!/usr/bin/env python3
"""Merge twelve single-reset V3-B003 fixture gates for one RTX lane."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "control:left", "control:right",
    "position_mirrored:left", "position_mirrored:right",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.input) != 12 or args.output.exists():
        raise ValueError("merge requires twelve inputs and a fresh output")
    values = [json.loads(path.read_text()) for path in args.input]
    scopes = {value.get("condition_scope") for value in values}
    identities = {(value.get("pod"), value.get("pod_uid"), value.get("gpu_uuid")) for value in values}
    repeats = {(value.get("condition_scope"), value.get("repeat_scope")) for value in values}
    expected_repeats = {(scope, repeat) for scope in EXPECTED for repeat in range(3)}
    if scopes != EXPECTED or repeats != expected_repeats or len(identities) != 1:
        raise ValueError("V3-B003 partial gates do not cover one complete lane")
    for value in values:
        if (
            value.get("schema_version")
            != "vla-wam-shared-v3b-dreamzero-model-blind-preflight-v1"
            or value.get("passed") is not True
            or value.get("model_request_count") != 0
            or value.get("behavioral_episode_count") != 0
            or len(value.get("tasks", [])) != 1
            or len(value["tasks"][0].get("repeat_resets", [])) != 1
            or value["tasks"][0]["repeat_resets"][0].get("repeat") != value.get("repeat_scope")
        ):
            raise ValueError("a V3-B003 partial physical gate did not pass")
    base = values[0]
    grouped = {}
    for value in values:
        row = value["tasks"][0]
        key = f"{row['arm']}:{row['relation']}"
        grouped.setdefault(key, {**row, "repeat_resets": []})["repeat_resets"].extend(
            row["repeat_resets"]
        )
    tasks = list(grouped.values())
    for row in tasks:
        row["repeat_resets"].sort(key=lambda item: item["repeat"])
    task_keys = set(grouped)
    if task_keys != EXPECTED:
        raise ValueError("V3-B003 task payloads are incomplete")
    output = {
        **base,
        "schema_version": "vla-wam-shared-v3b-dreamzero-model-blind-preflight-v1",
        "condition_scope": "all_merged_from_twelve_fresh_processes",
        "repeat_scope": "all_merged",
        "tasks": sorted(tasks, key=lambda row: (row["arm"], row["relation"])),
        "viewport_evidence": {
            f"{value['condition_scope']}:repeat{value['repeat_scope']}":
                next(iter(value["viewport_evidence"].values()))
            for value in values
        },
        "partial_process_gates": [
            {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in args.input
        ],
        "fresh_process_count": 12,
        "passed": True,
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"path": str(args.output.resolve()), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
