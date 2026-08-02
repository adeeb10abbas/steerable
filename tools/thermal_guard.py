#!/usr/bin/env python3
"""Pause a named Docker simulator before a GPU reaches its hard stop.

The guard records every state transition as JSONL. It does not change
simulated time, seeds, or policy configuration. Wall time does include cooling
pauses and must therefore be interpreted with the emitted event log.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def _container_status(name: str) -> str | None:
    result = _command(
        "docker", "inspect", "--format", "{{.State.Status}}", name, check=False
    )
    if result.returncode:
        return None
    return result.stdout.strip()


def _gpu_temperature(index: int) -> int:
    result = _command(
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=temperature.gpu",
        "--format=csv,noheader,nounits",
    )
    return int(result.stdout.strip())


class EventLog:
    def __init__(self, path: Path, context: dict[str, Any]) -> None:
        self.path = path
        self.context = context
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **values: Any) -> None:
        record = {
            "timestamp_utc": _utc_now(),
            "event": event,
            **self.context,
            **values,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--pause-temperature-c", type=int, default=87)
    parser.add_argument("--resume-temperature-c", type=int, default=80)
    parser.add_argument("--emergency-stop-temperature-c", type=int, default=90)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (
        args.resume_temperature_c
        < args.pause_temperature_c
        < args.emergency_stop_temperature_c
    ):
        raise ValueError("Require resume < pause < emergency-stop temperatures")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")

    context = {
        "container": args.container,
        "gpu_index": args.gpu_index,
        "pause_temperature_c": args.pause_temperature_c,
        "resume_temperature_c": args.resume_temperature_c,
        "emergency_stop_temperature_c": args.emergency_stop_temperature_c,
    }
    events = EventLog(args.output, context)
    status = _container_status(args.container)
    if status not in {"running", "paused"}:
        raise RuntimeError(f"Container {args.container!r} is not running: {status!r}")
    events.write("monitor_started", container_status=status)

    while True:
        status = _container_status(args.container)
        if status is None or status in {"exited", "dead", "removing"}:
            events.write("monitor_completed", container_status=status)
            return
        temperature = _gpu_temperature(args.gpu_index)
        if temperature >= args.emergency_stop_temperature_c:
            events.write(
                "emergency_stop",
                temperature_c=temperature,
                container_status=status,
            )
            if status == "paused":
                _command("docker", "unpause", args.container, check=False)
            _command("docker", "stop", "--timeout", "30", args.container, check=False)
            raise SystemExit(90)
        if status == "running" and temperature >= args.pause_temperature_c:
            pause_started = time.monotonic()
            _command("docker", "pause", args.container)
            peak = temperature
            events.write("cooldown_started", temperature_c=temperature)
            while True:
                status = _container_status(args.container)
                if status is None or status in {"exited", "dead", "removing"}:
                    events.write("monitor_completed_while_paused", container_status=status)
                    return
                temperature = _gpu_temperature(args.gpu_index)
                peak = max(peak, temperature)
                if temperature >= args.emergency_stop_temperature_c:
                    events.write(
                        "emergency_stop_while_paused",
                        temperature_c=temperature,
                        peak_temperature_c=peak,
                    )
                    _command("docker", "unpause", args.container, check=False)
                    _command("docker", "stop", "--timeout", "30", args.container, check=False)
                    raise SystemExit(90)
                if temperature <= args.resume_temperature_c:
                    _command("docker", "unpause", args.container)
                    events.write(
                        "cooldown_completed",
                        temperature_c=temperature,
                        peak_temperature_c=peak,
                        cooldown_seconds=round(time.monotonic() - pause_started, 3),
                    )
                    break
                time.sleep(args.poll_seconds)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
