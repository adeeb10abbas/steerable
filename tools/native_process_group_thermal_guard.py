#!/usr/bin/env python3
"""Launch or attach to one native process group and guard it thermally.

Launch mode is the safe default for WAM jobs: the exact worker command is
started in a new session/process group, its PID/PGID are recorded, and only
that group can receive POSIX STOP/CONT.  Attaching to an existing PID/PGID is
supported for recovery, but requires an explicit risk acknowledgement.

This tool is deliberately separate from ``thermal_guard.py`` and does not
change that Docker-specific guard's semantics.  An emergency temperature
leaves the native group stopped and records ``emergency_hold``; it never kills,
restarts, or labels a partial cell as a model failure.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_TEMPERATURE_COMMAND = (
    "nvidia-smi --id {gpu_index} --query-gpu=temperature.gpu "
    "--format=csv,noheader,nounits"
)
LEDGER_SCHEMA_VERSION = "vla-wam-shared-v2-native-thermal-interventions-v1"
INVALID_LEDGER_SCHEMA_VERSION = "vla-wam-shared-v2-native-thermal-invalid-attempts-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class EventLog:
    def __init__(self, path: Path, context: dict[str, Any]) -> None:
        self.path = path
        self.context = context
        self.records: list[dict[str, Any]] = []
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **values: Any) -> dict[str, Any]:
        record = {"timestamp_utc": utc_now(), "event": event, **self.context, **values}
        self.records.append(record)
        line = json.dumps(record, sort_keys=True, allow_nan=False)
        with self.path.open("a") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        return record


def process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError as error:
        raise RuntimeError(f"Cannot inspect process group {pgid}: {error}") from error
    return True


def read_temperature(command_template: str, gpu_index: int) -> int:
    command = shlex.split(command_template.format(gpu_index=gpu_index))
    if not command:
        raise ValueError("Temperature command must not be empty")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise RuntimeError(f"Temperature command failed: {detail}")
    try:
        return int(result.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"Temperature command returned non-integer output: {result.stdout!r}") from error


def signal_group(pgid: int, sig: signal.Signals, dry_run: bool) -> str:
    if dry_run:
        return f"would_{sig.name.lower()}"
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return "worker_exited"
    except PermissionError as error:
        raise RuntimeError(f"Cannot signal process group {pgid}: {error}") from error
    return sig.name.lower()


def validate_cell_metadata(args: argparse.Namespace) -> None:
    fields = (
        args.model_id,
        args.pair_id,
        args.environment_seed,
        args.sampling_seed,
        args.requested_relations,
    )
    has_ledger = args.ledger_output is not None or args.invalid_attempts_output is not None
    if has_ledger and any(value is None for value in fields):
        raise ValueError(
            "Thermal ledgers require --model-id, --pair-id, --environment-seed, "
            "--sampling-seed, and --requested-relation"
        )
    if args.requested_relations and len(args.requested_relations) != len(set(args.requested_relations)):
        raise ValueError("Each --requested-relation may be specified at most once")
    if (args.ledger_output is None) != (args.invalid_attempts_output is None):
        raise ValueError("Provide --ledger-output and --invalid-attempts-output together")
    if has_ledger:
        for ledger_path in (args.ledger_output, args.invalid_attempts_output):
            if args.model_id not in ledger_path.name:
                raise ValueError(
                    f"Model-specific ledger filename must include {args.model_id!r}: {ledger_path}"
                )


def write_cell_ledger(
    path: Path,
    args: argparse.Namespace,
    events: EventLog,
    *,
    status: str,
    worker_pid: int | None,
    worker_pgid: int | None,
    worker_exit_code: int | None,
) -> None:
    transition_events = [
        {
            key: record[key]
            for key in ("timestamp_utc", "event", "temperature_c", "worker_state", "action", "error")
            if key in record
        }
        for record in events.records
        if record["event"] != "temperature_sample"
    ]
    temperatures = [record["temperature_c"] for record in events.records if "temperature_c" in record]
    thermally_intervened = any(
        record["event"] in {"cooldown_started", "cooldown_completed", "emergency_hold"}
        for record in events.records
    )
    partial = status in {
        "emergency_hold",
        "temperature_query_failed",
        "worker_missing",
        "worker_exit_nonzero",
        "monitor_error",
    }
    started = events.records[0]["timestamp_utc"] if events.records else utc_now()
    completed = events.records[-1]["timestamp_utc"] if events.records else started
    if not thermally_intervened and not partial:
        return
    destination = args.invalid_attempts_output if partial else path
    schema_version = INVALID_LEDGER_SCHEMA_VERSION if partial else LEDGER_SCHEMA_VERSION
    if destination.exists():
        payload = json.loads(destination.read_text())
        if payload.get("schema_version") != schema_version or not isinstance(payload.get("events"), list):
            raise RuntimeError(f"Refusing to append to incompatible intervention ledger: {destination}")
    else:
        payload = {"schema_version": schema_version, "events": []}
    for requested_relation in args.requested_relations:
        event_id = (
            f"native-thermal-{args.model_id}-{args.pair_id}-{requested_relation}-"
            f"{started.replace(':', '').replace('-', '').replace('.', '')}"
        )
        payload["events"].append(
            {
                "id": event_id,
                "model_id": args.model_id,
                "pair_id": args.pair_id,
                "environment_seed": args.environment_seed,
                "sampling_seed": args.sampling_seed,
                "requested_relation": requested_relation,
                "started_at_utc": started,
                "completed_at_utc": completed,
                "status": status,
                "classification": (
                    "partial" if partial else "thermal_intervention" if thermally_intervened else "completed"
                ),
                "behavioral_result_valid": not partial,
                "wall_latency_valid": not partial and not thermally_intervened,
                "worker_pid": worker_pid,
                "worker_pgid": worker_pgid,
                "worker_exit_code": worker_exit_code,
                "gpu_index": args.gpu_index,
                "max_temperature_c": max(temperatures) if temperatures else None,
                "events": transition_events,
                "raw_event_log": str(events.path),
            }
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--launch", action="store_true", help="Start COMMAND in a new session/process group.")
    target.add_argument("--pgid", type=int, help="Attach to this native process group.")
    target.add_argument("--pid", type=int, help="Resolve and attach to this worker's process group.")
    parser.add_argument(
        "--acknowledge-attach-risk",
        action="store_true",
        help="Required with --pid/--pgid; confirms the resolved group was checked.",
    )
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Append-only raw JSONL event log.")
    parser.add_argument("--ledger-output", type=Path, help="Per-cell compiler-compatible JSON ledger.")
    parser.add_argument(
        "--invalid-attempts-output",
        type=Path,
        help="Compiler-compatible JSON ledger for emergency/partial cells.",
    )
    parser.add_argument("--model-id")
    parser.add_argument("--pair-id")
    parser.add_argument("--environment-seed", type=int)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument(
        "--requested-relation",
        action="append",
        choices=("left", "right"),
        dest="requested_relations",
        help="Affected cell relation; repeat for a paired LEFT/RIGHT invocation.",
    )
    parser.add_argument("--pause-temperature-c", type=int, default=87)
    parser.add_argument("--resume-temperature-c", type=int, default=80)
    parser.add_argument("--emergency-stop-temperature-c", type=int, default=90)
    parser.add_argument("--poll-seconds", type=float, default=0.5)
    parser.add_argument(
        "--temperature-command",
        default=DEFAULT_TEMPERATURE_COMMAND,
        help="Shell-style command with optional {gpu_index}; parsed without a shell.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Log signals but do not send them.")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="With --launch: exact worker command after --.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.resume_temperature_c < args.pause_temperature_c < args.emergency_stop_temperature_c):
        raise ValueError("Require resume < pause < emergency-stop temperatures")
    if args.poll_seconds <= 0:
        raise ValueError("--poll-seconds must be positive")
    validate_cell_metadata(args)
    if args.launch and args.acknowledge_attach_risk:
        raise ValueError("--acknowledge-attach-risk applies only to --pid/--pgid")
    if not args.launch and not args.acknowledge_attach_risk:
        raise ValueError("Direct --pid/--pgid attachment requires --acknowledge-attach-risk")

    worker: subprocess.Popen[Any] | None = None
    worker_pid: int | None = None
    if args.launch:
        command = list(args.command)
        if command and command[0] == "--":
            command.pop(0)
        if not command:
            raise ValueError("--launch requires an exact worker command after --")
        worker = subprocess.Popen(command, start_new_session=True)
        worker_pid = worker.pid
        pgid = os.getpgid(worker.pid)
        if pgid != worker.pid:
            raise RuntimeError(f"Launched worker did not receive a private process group: pid={worker.pid}, pgid={pgid}")
    elif args.pgid is not None:
        pgid = args.pgid
    else:
        worker_pid = args.pid
        try:
            pgid = os.getpgid(args.pid)
        except ProcessLookupError:
            pgid = None

    if pgid is not None and pgid <= 0:
        raise ValueError("Process group must be positive")
    if pgid == os.getpgrp():
        raise ValueError("Refusing to monitor or signal the guard's own process group")

    events = EventLog(
        args.output,
        {
            "pgid": pgid,
            "target_pid": worker_pid,
            "gpu_index": args.gpu_index,
            "pause_temperature_c": args.pause_temperature_c,
            "resume_temperature_c": args.resume_temperature_c,
            "emergency_stop_temperature_c": args.emergency_stop_temperature_c,
            "dry_run": args.dry_run,
        },
    )

    def finish(code: int, status: str, worker_exit_code: int | None = None) -> int:
        if args.ledger_output is not None:
            write_cell_ledger(
                args.ledger_output,
                args,
                events,
                status=status,
                worker_pid=worker_pid,
                worker_pgid=pgid,
                worker_exit_code=worker_exit_code,
            )
        return code

    if pgid is None or not process_group_exists(pgid):
        events.write("worker_missing", worker_state="missing")
        return finish(3, "worker_missing")

    worker_state = "running"
    if worker is not None:
        events.write("worker_started", worker_state=worker_state, command=command)
    events.write("monitor_started", worker_state=worker_state)
    while True:
        if worker is not None:
            worker_exit_code = worker.poll()
            if worker_exit_code is not None:
                status = "completed" if worker_exit_code == 0 else "worker_exit_nonzero"
                events.write("monitor_completed", worker_state=worker_state, worker_exit_code=worker_exit_code)
                return finish(0 if worker_exit_code == 0 else 4, status, worker_exit_code)
        elif not process_group_exists(pgid):
            events.write("monitor_completed", worker_state=worker_state)
            return finish(0, "completed")
        try:
            temperature = read_temperature(args.temperature_command, args.gpu_index)
        except (RuntimeError, ValueError) as error:
            action = signal_group(pgid, signal.SIGSTOP, args.dry_run)
            events.write(
                "temperature_query_failed",
                worker_state="emergency_hold",
                action=action,
                error=str(error),
            )
            return finish(2, "temperature_query_failed")
        events.write("temperature_sample", temperature_c=temperature, worker_state=worker_state)
        if temperature >= args.emergency_stop_temperature_c:
            action = signal_group(pgid, signal.SIGSTOP, args.dry_run)
            events.write(
                "emergency_hold",
                temperature_c=temperature,
                worker_state="emergency_hold",
                action=action,
            )
            return finish(90, "emergency_hold")
        if worker_state == "running" and temperature >= args.pause_temperature_c:
            action = signal_group(pgid, signal.SIGSTOP, args.dry_run)
            if action == "worker_exited":
                events.write("monitor_completed", worker_state=worker_state)
                return finish(0, "completed")
            worker_state = "paused"
            events.write("cooldown_started", temperature_c=temperature, worker_state=worker_state, action=action)
        elif worker_state == "paused" and temperature <= args.resume_temperature_c:
            action = signal_group(pgid, signal.SIGCONT, args.dry_run)
            if action == "worker_exited":
                events.write("monitor_completed", worker_state=worker_state)
                return finish(0, "completed")
            worker_state = "running"
            events.write("cooldown_completed", temperature_c=temperature, worker_state=worker_state, action=action)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
