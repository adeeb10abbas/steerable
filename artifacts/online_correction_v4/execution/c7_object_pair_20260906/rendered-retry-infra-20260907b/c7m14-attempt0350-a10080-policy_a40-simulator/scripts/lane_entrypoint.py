#!/usr/bin/env python3
"""Synchronous preflight followed by direct PID-1 experiment replacement."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import signal
import sys
import time
import traceback
from typing import Any

from startup_preflight import (
    PreflightError,
    acquire_attempt_lock,
    load_launch_config,
    prepare_runtime_directories,
    reserve_policy_port,
    require_absolute_executable,
    run_preflight,
    utc_now,
)


TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_ENV = (
    "LANE_ROLE",
    "POD_UID",
    "POD_NAME",
    "POD_NAMESPACE",
    "POD_IP",
    "LANE_ID",
    "ATTEMPT_ID",
    "OUTPUT_PARENT",
    "LANE_LAUNCH_CONFIG",
    "LANE_LAUNCH_CONFIG_SHA256",
    "IMAGE_DIGEST_EXPECTED",
)
LOCK_HANDLES: list[Any] = []


def record_outer_failure(exc: BaseException) -> Path | None:
    """Best-effort immutable evidence for failures before the full preflight."""

    required = {
        key: os.environ.get(key)
        for key in (
            "LANE_ROLE",
            "POD_UID",
            "POD_NAME",
            "POD_NAMESPACE",
            "POD_IP",
            "LANE_ID",
            "ATTEMPT_ID",
            "OUTPUT_PARENT",
            "LANE_LAUNCH_CONFIG",
            "LANE_LAUNCH_CONFIG_SHA256",
            "IMAGE_DIGEST_EXPECTED",
        )
    }
    output_raw = required.get("OUTPUT_PARENT")
    role = required.get("LANE_ROLE")
    pod_uid = required.get("POD_UID")
    lane_id = required.get("LANE_ID")
    attempt_id = required.get("ATTEMPT_ID")
    if (
        not output_raw
        or not role
        or not pod_uid
        or not lane_id
        or not attempt_id
        or not TOKEN_RE.fullmatch(lane_id)
        or not TOKEN_RE.fullmatch(attempt_id)
    ):
        return None
    output_parent = Path(output_raw)
    if not output_parent.is_absolute() or not output_parent.is_dir() or not os.access(output_parent, os.W_OK):
        return None
    evidence_dir = (
        output_parent
        / ".lane-runtime"
        / pod_uid
        / f"lane-{lane_id}"
        / f"attempt-{attempt_id}"
        / role
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    path = evidence_dir / "entrypoint-failure.json"
    payload = {
        "schema_version": "vla-wam-v4-k8s-lane-entrypoint-failure-v1",
        "created_at_utc": utc_now(),
        "status": "infrastructure_invalid_before_experiment_exec",
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "selected_environment": required,
        "episode_output_dir_precreated": False,
        "error": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
    }
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        return path
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


def required_environment() -> dict[str, str]:
    result: dict[str, str] = {}
    for key in REQUIRED_ENV:
        value = os.environ.get(key)
        if not value:
            raise PreflightError(f"required lane environment is missing: {key}")
        result[key] = value
    if result["LANE_ROLE"] not in {"simulator", "policy"}:
        raise PreflightError("LANE_ROLE must be simulator or policy")
    for key in ("LANE_ID", "ATTEMPT_ID"):
        if not TOKEN_RE.fullmatch(result[key]):
            raise PreflightError(f"unsafe {key}: {result[key]!r}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", result["IMAGE_DIGEST_EXPECTED"]):
        raise PreflightError("IMAGE_DIGEST_EXPECTED must be a literal sha256: digest")
    return result


def pre_stop() -> int:
    env = required_environment()
    wait_seconds_raw = os.environ.get("PRESTOP_WAIT_SECONDS", "120")
    try:
        wait_seconds = float(wait_seconds_raw)
    except ValueError as exc:
        raise PreflightError("PRESTOP_WAIT_SECONDS must be numeric") from exc
    if not 0 < wait_seconds <= 240:
        raise PreflightError("PRESTOP_WAIT_SECONDS must be in (0, 240]")
    marker = os.environ.get("PRESTOP_MARKER")
    if not marker:
        marker = str(
            Path(env["OUTPUT_PARENT"])
            / ".lane-runtime"
            / env["POD_UID"]
            / f"lane-{env['LANE_ID']}"
            / f"attempt-{env['ATTEMPT_ID']}"
            / env["LANE_ROLE"]
            / "prestop-received.json"
        )
    path = Path(marker)
    path.parent.mkdir(parents=True, exist_ok=True)
    policy_signal = env["LANE_ROLE"] == "policy"
    payload = json.dumps(
        {
            "schema_version": "vla-wam-v4-k8s-lane-prestop-v1",
            "received_at_utc": utc_now(),
            "hook_pid": os.getpid(),
            "lane_role": env["LANE_ROLE"],
            "signal_after_fsync": "SIGINT" if policy_signal else None,
            "signal_target_pid": 1 if policy_signal else None,
            "post_signal_wait_seconds": wait_seconds if policy_signal else None,
        },
        sort_keys=True,
    ) + "\n"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    if policy_signal:
        if os.getpid() == 1:
            raise PreflightError("policy preStop hook must execute separately from PID 1")
        # The marker is durable before the signal. The frozen asyncio WebSocket
        # server handles SIGINT cleanly, whereas SIGTERM left JAX alive until
        # Kubernetes exhausted the grace period and issued SIGKILL.
        os.kill(1, signal.SIGINT)
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            try:
                os.kill(1, 0)
            except ProcessLookupError:
                return 0
            time.sleep(0.25)
    return 0


def main() -> None:
    if sys.argv[1:] == ["--prestop"]:
        raise SystemExit(pre_stop())
    if len(sys.argv) != 1:
        raise PreflightError("lane_entrypoint.py accepts no launch arguments; use immutable launch.json")
    if os.getpid() != 1:
        raise PreflightError(f"lane entrypoint must be PID 1 in launch mode, got PID {os.getpid()}")

    env = required_environment()
    role = env["LANE_ROLE"]
    output_parent = Path(env["OUTPUT_PARENT"])
    if not output_parent.is_absolute() or not output_parent.is_dir():
        raise PreflightError("OUTPUT_PARENT must already exist and be absolute")
    prepare_runtime_directories()
    config_path = Path(env["LANE_LAUNCH_CONFIG"])
    config, config_sha256 = load_launch_config(config_path, env["LANE_LAUNCH_CONFIG_SHA256"])
    if config.get("schema_version") != "vla-wam-v4-k8s-lane-launch-v1":
        raise PreflightError("launch config schema differs")
    if config.get("role") != role:
        raise PreflightError("launch config role differs from LANE_ROLE")
    experiment_argv = config.get("experiment_argv")
    if not isinstance(experiment_argv, list) or not experiment_argv or not all(
        isinstance(item, str) and item for item in experiment_argv
    ):
        raise PreflightError("experiment_argv must be a nonempty string array")
    require_absolute_executable(experiment_argv[0], "experiment_argv[0]")

    # Only the hidden runtime-evidence tree is created.  The runner receives an
    # absent EPISODE_OUTPUT_DIR and must atomically create it write-once.
    pod_uid = env["POD_UID"]
    identity_root = output_parent / ".lane-runtime" / pod_uid / f"lane-{env['LANE_ID']}" / f"attempt-{env['ATTEMPT_ID']}"
    evidence_dir = identity_root / role
    evidence_dir.mkdir(parents=True, exist_ok=False)
    episode_output_dir = (
        output_parent
        / pod_uid
        / f"lane-{env['LANE_ID']}"
        / f"attempt-{env['ATTEMPT_ID']}"
        / "episodes"
    )
    if episode_output_dir.exists():
        raise PreflightError(f"write-once episode output already exists: {episode_output_dir}")

    attempt_handle, attempt_lock_path = acquire_attempt_lock(
        output_parent, role, env["LANE_ID"], env["ATTEMPT_ID"]
    )
    LOCK_HANDLES.extend([attempt_handle])
    port_lock_path: Path | None = None
    if role == "policy":
        port = int(config.get("policy_port", 0))
        port_handle, port_lock_path = reserve_policy_port(port)
        LOCK_HANDLES.append(port_handle)

    run_preflight(
        config=config,
        config_path=config_path,
        config_sha256=config_sha256,
        evidence_dir=evidence_dir,
        output_parent=output_parent,
        role=role,
        lane_id=env["LANE_ID"],
        attempt_id=env["ATTEMPT_ID"],
        pod_identity={
            "pod_uid": pod_uid,
            "pod_name": env["POD_NAME"],
            "pod_namespace": env["POD_NAMESPACE"],
            "pod_ip": env["POD_IP"],
        },
        experiment_argv=experiment_argv,
        image_digest=env["IMAGE_DIGEST_EXPECTED"],
    )

    os.environ["EPISODE_OUTPUT_DIR"] = str(episode_output_dir)
    os.environ["LANE_RUNTIME_EVIDENCE_DIR"] = str(evidence_dir)
    os.environ["LANE_ATTEMPT_LOCK_PATH"] = str(attempt_lock_path)
    if port_lock_path is not None:
        os.environ["LANE_PORT_LOCK_PATH"] = str(port_lock_path)
    for handle in LOCK_HANDLES:
        os.set_inheritable(handle.fileno(), True)

    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    os.execvpe(experiment_argv[0], experiment_argv, os.environ)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        try:
            record_outer_failure(exc)
        except Exception:
            pass
        raise
