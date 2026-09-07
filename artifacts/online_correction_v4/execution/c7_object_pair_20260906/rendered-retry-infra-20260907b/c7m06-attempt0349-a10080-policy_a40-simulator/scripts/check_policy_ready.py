#!/usr/bin/env python3
"""Policy readiness: immutable local proof plus a valid HTTP health check."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

from startup_preflight import PreflightError, load_launch_config, probe_http_healthz


def _runtime_report_path() -> Path:
    return (
        Path(os.environ["OUTPUT_PARENT"])
        / ".lane-runtime"
        / os.environ["POD_UID"]
        / f"lane-{os.environ['LANE_ID']}"
        / f"attempt-{os.environ['ATTEMPT_ID']}"
        / "policy"
        / "runtime-preflight.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-loaded",
        action="store_true",
        help="Required explicit assertion that this probe is the post-checkpoint-load readiness gate.",
    )
    parser.add_argument("--mode", choices=("http_healthz", "metadata_jsonl"), default="http_healthz")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--launch-config", type=Path)
    parser.add_argument("--expected-launch-sha256")
    parser.add_argument("--timeout-seconds", type=float, default=3.0)
    args = parser.parse_args()
    if not args.checkpoint_loaded:
        raise SystemExit("readiness probe requires explicit --checkpoint-loaded")

    config_path = args.launch_config or Path(os.environ["LANE_LAUNCH_CONFIG"])
    launch_sha256 = args.expected_launch_sha256 or os.environ["LANE_LAUNCH_CONFIG_SHA256"]
    config, _ = load_launch_config(config_path, launch_sha256)
    if config.get("role") != "policy":
        raise SystemExit("readiness launch config is not for a policy lane")
    if args.mode == "http_healthz" and config.get("readiness_contract") != "http_healthz_after_checkpoint_load":
        raise SystemExit("HTTP /healthz readiness is not authorized by the immutable launch contract")
    if args.mode == "metadata_jsonl" and config.get("readiness_contract") != "metadata_jsonl_after_checkpoint_load":
        raise SystemExit("metadata readiness is not authorized by the immutable launch contract")
    port = args.port or int(config.get("policy_port", 0))

    report = json.loads(_runtime_report_path().read_text(encoding="utf-8"))
    if report.get("passed") is not True:
        raise SystemExit("policy startup preflight has not passed")
    if report.get("checkpoint_sha256") != config.get("checkpoint_sha256"):
        raise SystemExit("checkpoint-loaded policy preflight evidence differs")
    if report.get("launch_config", {}).get("sha256") != launch_sha256:
        raise SystemExit("policy preflight launch-config digest differs")

    if args.mode == "http_healthz":
        try:
            probe_http_healthz(args.host, port, args.timeout_seconds)
        except PreflightError as exc:
            raise SystemExit(str(exc)) from exc
        return

    request = {"type": "lane_metadata"}
    with socket.create_connection((args.host, port), timeout=args.timeout_seconds) as connection:
        connection.sendall(json.dumps(request, sort_keys=True).encode("utf-8") + b"\n")
        connection.settimeout(args.timeout_seconds)
        payload = b""
        while not payload.endswith(b"\n") and len(payload) <= 65536:
            block = connection.recv(4096)
            if not block:
                break
            payload += block
    response = json.loads(payload.decode("utf-8"))
    if not isinstance(response, dict):
        raise SystemExit("policy metadata response is not an object")
    expected = {
        "checkpoint_loaded": True,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "launch_config_sha256": launch_sha256,
    }
    mismatches = {key: (value, response.get(key)) for key, value in expected.items() if response.get(key) != value}
    if mismatches:
        raise SystemExit(f"policy readiness metadata differs: {mismatches}")


if __name__ == "__main__":
    main()
