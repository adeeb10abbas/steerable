#!/usr/bin/env python3
"""Rebind one rendered V4 simulator lane to an attested shared policy Service."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def yaml_block(text: str, key: str) -> tuple[dict[str, Any], int, int]:
    marker = f"  {key}: |\n"
    start = text.find(marker)
    if start < 0:
        raise ValueError(f"YAML block is missing: {key}")
    payload_start = start + len(marker)
    cursor = payload_start
    payload_lines: list[str] = []
    for line in text[payload_start:].splitlines(keepends=True):
        if not line.startswith("    "):
            break
        payload_lines.append(line[4:])
        cursor += len(line)
    value = json.loads("".join(payload_lines))
    if not isinstance(value, dict):
        raise ValueError(f"YAML block is not a JSON object: {key}")
    if "".join(payload_lines) != canonical_json(value):
        raise ValueError(f"YAML block is not canonical JSON: {key}")
    return value, payload_start, cursor


def bind_shared_policy(
    *,
    donor_configmap: Path,
    target_configmap: Path,
    target_simulator_job: Path,
    target_scripts_configmap: Path,
    output_dir: Path,
) -> dict[str, Any]:
    donor_text = donor_configmap.read_text(encoding="utf-8")
    donor_simulator, _donor_start, _donor_end = yaml_block(
        donor_text,
        "simulator-launch.json",
    )
    donor_wait = donor_simulator.get("policy_wait")
    if not isinstance(donor_wait, dict):
        raise ValueError("donor simulator launch lacks policy_wait")
    donor_identity = donor_wait.get("service_identity")
    if not isinstance(donor_identity, dict) or not donor_identity:
        raise ValueError("donor policy Service identity is missing")

    target_text = target_configmap.read_text(encoding="utf-8")
    target_simulator, block_start, block_end = yaml_block(
        target_text,
        "simulator-launch.json",
    )
    target_wait = target_simulator.get("policy_wait")
    if not isinstance(target_wait, dict):
        raise ValueError("target simulator launch lacks policy_wait")
    original_host = target_wait.get("host")
    original_port = target_wait.get("port")
    if original_port != donor_wait.get("port"):
        raise ValueError("shared policy Service port differs")
    original_json = canonical_json(target_simulator)
    target_wait["host"] = donor_wait["host"]
    target_wait["port"] = donor_wait["port"]
    target_wait["service_identity"] = dict(donor_identity)
    rebound_json = canonical_json(target_simulator)
    rebound_text = (
        target_text[:block_start]
        + "".join(f"    {line}" for line in rebound_json.splitlines(keepends=True))
        + target_text[block_end:]
    )

    original_sha = sha256_bytes(original_json.encode("utf-8"))
    rebound_sha = sha256_bytes(rebound_json.encode("utf-8"))
    simulator_job_text = target_simulator_job.read_text(encoding="utf-8")
    old_value = f'value: "{original_sha}"'
    new_value = f'value: "{rebound_sha}"'
    if simulator_job_text.count(old_value) != 1:
        raise ValueError("target simulator launch hash binding is missing or repeated")
    rebound_job_text = simulator_job_text.replace(old_value, new_value)

    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rebound_configmap = output_dir / "configmap.yaml"
    rebound_job = output_dir / "simulator-job.yaml"
    rebound_scripts = output_dir / "scripts-configmap.yaml"
    rebound_configmap.write_text(rebound_text, encoding="utf-8")
    rebound_job.write_text(rebound_job_text, encoding="utf-8")
    shutil.copyfile(target_scripts_configmap, rebound_scripts)
    receipt = {
        "schema_version": "v4-shared-policy-lane-binding-v1",
        "policy_session_model": (
            "Concurrent fresh websocket clients use one stateless policy server; "
            "the server serializes seeded inference requests."
        ),
        "donor_policy_service": {
            "host": donor_wait["host"],
            "port": donor_wait["port"],
            "service_identity": donor_identity,
        },
        "target_original_policy_service": {
            "host": original_host,
            "port": original_port,
        },
        "target_simulator_launch_sha256_before": original_sha,
        "target_simulator_launch_sha256_after": rebound_sha,
        "source_files": {
            "donor_configmap": {
                "path": str(donor_configmap),
                "sha256": sha256_file(donor_configmap),
            },
            "target_configmap": {
                "path": str(target_configmap),
                "sha256": sha256_file(target_configmap),
            },
            "target_simulator_job": {
                "path": str(target_simulator_job),
                "sha256": sha256_file(target_simulator_job),
            },
            "target_scripts_configmap": {
                "path": str(target_scripts_configmap),
                "sha256": sha256_file(target_scripts_configmap),
            },
        },
        "rendered_files": {
            "configmap": {
                "path": str(rebound_configmap),
                "sha256": sha256_file(rebound_configmap),
            },
            "simulator_job": {
                "path": str(rebound_job),
                "sha256": sha256_file(rebound_job),
            },
            "scripts_configmap": {
                "path": str(rebound_scripts),
                "sha256": sha256_file(rebound_scripts),
            },
        },
        "model_request_count": 0,
        "behavioral_episode_count": 0,
    }
    receipt_path = output_dir / "shared-policy-binding.json"
    receipt_path.write_text(canonical_json(receipt), encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-configmap", type=Path, required=True)
    parser.add_argument("--target-configmap", type=Path, required=True)
    parser.add_argument("--target-simulator-job", type=Path, required=True)
    parser.add_argument("--target-scripts-configmap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = bind_shared_policy(
        donor_configmap=args.donor_configmap.resolve(),
        target_configmap=args.target_configmap.resolve(),
        target_simulator_job=args.target_simulator_job.resolve(),
        target_scripts_configmap=args.target_scripts_configmap.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
