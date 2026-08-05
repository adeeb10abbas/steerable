#!/usr/bin/env python3
"""Run one non-overwriting shard of V3-A002 matched pi0-FAST pairs.

The shard unit is an environment seed, and every seed always launches the
frozen LEFT and RIGHT cells together.  Contract or endpoint failures stop
before any worker is launched.  Once a shard starts, pair-local technical
failures are retained in a machine-readable ledger and do not prevent later
registered pairs from being attempted.

This launcher does not compile behavioral evidence.  A zero guard exit means
only that the paired worker exited cleanly; the retained captures, actions,
videos, and JSONL still require the fail-closed adapter compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from adapter import (  # noqa: E402
    MODEL_ID,
    bridge_command,
    preflight,
    sha256_file,
)


SCHEMA_VERSION = (
    "vla-wam-shared-v3-pi0-fast-old-name-config-shard-ledger-v1"
)
MIN_SEED = 8310
MAX_SEED = 8329
RELATIONS = ("left", "right")
SAFE_SHARD_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
ALLOWED_SIMULATOR_PODS = {
    "raytrace-rtxpro6000-ali",
    "vla-wam-rtx-cosmos-ali",
    "vla-wam-rtx-nano-ali",
}
SIMULATOR_PYTHON = Path(
    "/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python"
)
OPENPI_ROOT = Path("/data/users/ali/vla_wam/external/openpi-235044ed")
OPENPI_CLIENT_SOURCE = OPENPI_ROOT / "packages/openpi-client/src"
ROBOLAB_ROOT = Path(
    "/data/users/ali/vla_wam/external/RoboLab-pi0fast-bridge-0aef241-clean01"
)
FROZEN_PATH = (
    "/data/users/ali/vla_wam/tools/git-lfs:/usr/local/sbin:/usr/local/bin:"
    "/usr/sbin:/usr/bin:/sbin:/bin"
)
FROZEN_LD_LIBRARY_PATH = (
    "/data/users/ali/vla_wam/envs/groot-render-libs/lib:"
    "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/"
    "usr/lib/x86_64-linux-gnu:"
    "/data/users/ali/vla_wam/envs/isaac-system-libs/lib:"
    "/usr/lib/x86_64-linux-gnu"
)
FROZEN_VK_ICD = "/etc/vulkan/icd.d/nvidia_icd.json"
RENDERER_ARGUMENTS = (
    "--headless",
    "--device",
    "cuda:0",
    "--renderer",
    "realtime",
    "--rendering-type",
    "balanced",
    "--kit_args=--/rtx/verifyDriverVersion/enabled=false",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resolve_existing_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ValueError(f"{label} is not a file: {resolved}")
    return resolved


def _require_raw_outside_git(study_root: Path, raw_root: Path) -> None:
    try:
        raw_root.relative_to(study_root)
    except ValueError:
        return
    raise ValueError(
        f"--raw-root must remain outside the study Git worktree: {raw_root}"
    )


def _probe_endpoint(
    host: str, port: int, timeout_s: float, environment: dict[str, str]
) -> dict[str, Any]:
    """Complete the real WebSocket metadata handshake without inference."""

    snippet = (
        "import json; "
        "from openpi_client.websocket_client_policy import WebsocketClientPolicy; "
        f"c=WebsocketClientPolicy(host={host!r},port={port}); "
        "m=c.get_server_metadata(); c._ws.close(); print(json.dumps(m))"
    )
    try:
        completed = subprocess.run(
            [str(SIMULATOR_PYTHON), "-c", snippet],
            env=environment,
            stdin=subprocess.DEVNULL,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"policy metadata handshake timed out for {host}:{port}"
        ) from error
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"policy metadata handshake failed: {detail}")
    try:
        metadata = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("policy metadata handshake returned malformed JSON") from error
    expected = {
        "pi0_fast_old_name_config_bridge": "v3a002",
        "openpi_commit": "235044ed8a1502c0a18338eedc5d7adfe705af05",
        "openpi_tree": "03a4387bedbc0fa1467c367c60fc24e28b61ec6c",
        "openpi_config": "pi0_fast_droid_jointpos",
        "max_token_len": 250,
        "checkpoint_assets_rule": "checkpoint_local_assets_only",
        "sampling_contract": "required_request_field:sampling_seed",
    }
    for key, wanted in expected.items():
        if metadata.get(key) != wanted:
            raise RuntimeError(f"policy metadata mismatch for {key}")
    return expected


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _base_worker_environment(study_root: Path) -> dict[str, str]:
    """Return the exact last-known-good RTX simulator environment prefix."""

    expected_study = Path("/data/users/ali/vla_wam/src/steerable")
    if study_root != expected_study:
        raise ValueError(
            f"V3-A002 RTX execution requires study root {expected_study}, got "
            f"{study_root}"
        )
    if not SIMULATOR_PYTHON.is_file():
        raise ValueError(f"frozen simulator interpreter is missing: {SIMULATOR_PYTHON}")
    if not os.path.samefile(sys.executable, SIMULATOR_PYTHON):
        raise ValueError(
            "run_shard.py must be invoked with the frozen RoboLab simulator "
            f"interpreter {SIMULATOR_PYTHON}; got {sys.executable}"
        )
    required_directories = (
        OPENPI_CLIENT_SOURCE,
        ROBOLAB_ROOT,
        Path("/data/users/ali/vla_wam/envs/groot-render-libs/lib"),
        Path(
            "/data/users/ali/vla_wam/envs/robolab-native-libs-ubuntu2204/"
            "usr/lib/x86_64-linux-gnu"
        ),
        Path("/data/users/ali/vla_wam/envs/isaac-system-libs/lib"),
    )
    missing = [str(path) for path in required_directories if not path.is_dir()]
    if missing:
        raise ValueError(f"frozen RTX environment paths are missing: {missing}")
    if not Path(FROZEN_VK_ICD).is_file():
        raise ValueError(f"frozen NVIDIA Vulkan ICD is missing: {FROZEN_VK_ICD}")

    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_ICD_FILENAMES": FROZEN_VK_ICD,
            "LD_LIBRARY_PATH": FROZEN_LD_LIBRARY_PATH,
            "PATH": FROZEN_PATH,
            "PYTHONPATH": ":".join(
                (str(study_root), str(ROBOLAB_ROOT), str(OPENPI_CLIENT_SOURCE))
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    return environment


def _attest_openpi_client(environment: dict[str, str]) -> dict[str, Any]:
    """Fail unless the simulator imports the client from OpenPI 235044ed."""

    snippet = (
        "import json, pathlib, openpi_client; "
        "print(json.dumps({'file': str(pathlib.Path(openpi_client.__file__).resolve())}))"
    )
    completed = subprocess.run(
        [str(SIMULATOR_PYTHON), "-c", snippet],
        env=environment,
        stdin=subprocess.DEVNULL,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"openpi_client import attestation failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
        imported = Path(payload["file"]).resolve()
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise RuntimeError(
            "openpi_client import attestation returned malformed output: "
            f"{completed.stdout!r}"
        ) from error
    if not _path_is_within(imported, OPENPI_CLIENT_SOURCE.resolve()):
        raise RuntimeError(
            "openpi_client resolved outside the frozen OpenPI 235044ed client "
            f"source: {imported}"
        )
    return {
        "module_file": str(imported),
        "expected_source_root": str(OPENPI_CLIENT_SOURCE.resolve()),
        "passed": True,
    }


def _pair_worker_environment(
    base: dict[str, str], *, pod_tag: str
) -> tuple[dict[str, str], dict[str, str]]:
    """Add the exact per-pod writable caches used by the RTX runtime."""

    environment = dict(base)
    cache_root = Path("/data/users/ali/vla_wam/cache/pi0fast-v3a002") / pod_tag
    cache_paths = {
        "XDG_CACHE_HOME": cache_root / "xdg",
        "WARP_CACHE_PATH": cache_root / "warp",
        "MPLCONFIGDIR": cache_root / "matplotlib",
        "TMPDIR": cache_root / "tmp",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    environment.update({key: str(path) for key, path in cache_paths.items()})
    return environment, {key: str(path) for key, path in cache_paths.items()}


def _guard_command(
    *,
    study_root: Path,
    pair_dir: Path,
    gpu_index: int,
    pair_id: str,
    seed: int,
    worker_command: list[str],
) -> list[str]:
    thermal_guard = study_root / "tools/native_process_group_thermal_guard.py"
    if not thermal_guard.is_file():
        raise ValueError(f"thermal guard is missing: {thermal_guard}")
    return [
        sys.executable,
        str(thermal_guard),
        "--launch",
        "--gpu-index",
        str(gpu_index),
        "--output",
        str(pair_dir / "thermal_events.jsonl"),
        "--ledger-output",
        str(pair_dir / f"runtime_interventions_{MODEL_ID}.json"),
        "--invalid-attempts-output",
        str(pair_dir / f"invalid_attempts_{MODEL_ID}.json"),
        "--model-id",
        MODEL_ID,
        "--pair-id",
        pair_id,
        "--environment-seed",
        str(seed),
        "--sampling-seed",
        str(seed),
        "--requested-relation",
        "left",
        "--requested-relation",
        "right",
        "--",
        *worker_command,
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-end", type=int, required=True)
    parser.add_argument("--runtime-identity", type=Path, required=True)
    parser.add_argument("--release-gate", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=8011)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument(
        "--pod-tag",
        required=True,
        help="Exact ali-owned simulator pod name used to isolate writable caches.",
    )
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--gpu-index", type=int, required=True)
    parser.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="Positive attempt number embedded in every frozen output name.",
    )
    parser.add_argument("--endpoint-timeout-s", type=float, default=10.0)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    study_root = args.study_root.expanduser().resolve()
    if not (study_root / ".git").exists():
        raise ValueError(f"--study-root is not a Git worktree: {study_root}")
    if not (MIN_SEED <= args.seed_start <= args.seed_end <= MAX_SEED):
        raise ValueError(
            f"seed shard must be inclusive and contained in {MIN_SEED}-{MAX_SEED}"
        )
    if not SAFE_SHARD_ID.fullmatch(args.shard_id):
        raise ValueError(
            "--shard-id must contain only letters, digits, dot, underscore, or "
            "hyphen and be at most 128 characters"
        )
    if not SAFE_SHARD_ID.fullmatch(args.pod_tag):
        raise ValueError(
            "--pod-tag must contain only letters, digits, dot, underscore, or "
            "hyphen and be at most 128 characters"
        )
    if args.pod_tag not in ALLOWED_SIMULATOR_PODS:
        raise ValueError(
            "--pod-tag must identify one of the three explicitly selected "
            f"ali-owned RTX pods: {sorted(ALLOWED_SIMULATOR_PODS)}"
        )
    if args.attempt <= 0:
        raise ValueError("--attempt must be positive")
    if args.gpu_index != 0:
        raise ValueError(
            "the frozen single-GPU RTX simulator prefix requires --gpu-index 0"
        )
    if not (1 <= args.remote_port <= 65535):
        raise ValueError("--remote-port must be in 1..65535")
    if not args.remote_host.strip() or any(ch.isspace() for ch in args.remote_host):
        raise ValueError("--remote-host must be a non-empty hostname or address")
    if args.endpoint_timeout_s <= 0:
        raise ValueError("--endpoint-timeout-s must be positive")

    runtime_identity = _resolve_existing_file(
        args.runtime_identity, "runtime identity"
    )
    release_gate = _resolve_existing_file(args.release_gate, "release gate")
    raw_root = args.raw_root.expanduser().resolve()
    _require_raw_outside_git(study_root, raw_root)
    return study_root, runtime_identity, release_gate, raw_root


def _authorization_snapshot(
    *,
    study_root: Path,
    seeds: list[int],
    runtime_identity: Path,
    release_gate: Path,
) -> dict[int, dict[str, Any]]:
    """Authorize the complete shard before launching its first pair."""

    authorized: dict[int, dict[str, Any]] = {}
    for index, seed in enumerate(seeds):
        result = preflight(
            study_root,
            seed,
            runtime_identity,
            release_gate,
            # Rehash the live external repositories, checkpoint payload, and
            # environment lock once.  Every later pair still reruns the exact
            # bridge preflight in robolab_bridge.py before Isaac starts.
            check_live_repositories=index == 0,
        )
        expected_pair_id = f"v3:droid:{MODEL_ID}:seed{seed}"
        expected_cells = [
            f"{expected_pair_id}:left",
            f"{expected_pair_id}:right",
        ]
        if (
            result.get("status") != "ready"
            or result.get("pair_id") != expected_pair_id
            or result.get("cell_ids") != expected_cells
        ):
            raise ValueError(f"preflight returned an unexpected pair for seed {seed}")
        authorized[seed] = result
    return authorized


def main() -> int:
    args = _parse_args()
    study_root, runtime_identity, release_gate, raw_root = _validate_args(args)
    seeds = list(range(args.seed_start, args.seed_end + 1))

    # Both gates occur before creation of the unique shard directory, so a
    # corrected invocation can reuse the same shard id without ambiguity.
    authorized = _authorization_snapshot(
        study_root=study_root,
        seeds=seeds,
        runtime_identity=runtime_identity,
        release_gate=release_gate,
    )
    first = authorized[seeds[0]]
    runtime = first["runtime_identity"]
    if Path(str(runtime.get("openpi_dir", ""))).resolve() != OPENPI_ROOT:
        raise ValueError(
            f"runtime identity must use exact OpenPI 235044ed worktree {OPENPI_ROOT}"
        )
    if Path(str(runtime.get("robolab_dir", ""))).resolve() != ROBOLAB_ROOT:
        raise ValueError(
            f"runtime identity must use exact clean RoboLab worktree {ROBOLAB_ROOT}"
        )
    target = runtime.get("target_kubernetes")
    expected_endpoint = f"{args.remote_host}:{args.remote_port}"
    if not isinstance(target, dict):
        raise ValueError("per-pod runtime identity lacks target_kubernetes")
    simulator = target.get("simulator")
    policy = target.get("policy")
    if not isinstance(simulator, dict) or not isinstance(policy, dict):
        raise ValueError(
            "per-pod runtime identity requires nested target_kubernetes "
            "simulator and policy objects"
        )
    for label, observed, wanted in (
        ("simulator.pod", simulator.get("pod"), args.pod_tag),
        ("simulator.gpu_index", simulator.get("gpu_index"), 0),
        ("policy.endpoint", policy.get("endpoint"), expected_endpoint),
    ):
        if observed != wanted:
            raise ValueError(
                f"per-pod runtime identity target mismatch for {label}: "
                f"expected {wanted!r}, got {observed!r}"
            )
    base_worker_environment = _base_worker_environment(study_root)
    openpi_client_attestation = _attest_openpi_client(base_worker_environment)
    endpoint_metadata = _probe_endpoint(
        args.remote_host,
        args.remote_port,
        args.endpoint_timeout_s,
        base_worker_environment,
    )

    shards_root = raw_root / "shards"
    shards_root.mkdir(parents=True, exist_ok=True)
    shard_dir = shards_root / args.shard_id
    shard_dir.mkdir(exist_ok=False)
    ledger_path = shard_dir / "shard_ledger.json"

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "study_id": "vla_wam_language_steerability_v3",
        "model_id": MODEL_ID,
        "shard_id": args.shard_id,
        "simulator_pod_tag": args.pod_tag,
        "status": "running",
        "started_at_utc": utc_now(),
        "completed_at_utc": None,
        "seed_range_inclusive": [args.seed_start, args.seed_end],
        "seeds": seeds,
        "matched_pair_contract": {
            "relations": list(RELATIONS),
            "condition": "both",
            "left_right_may_be_split": False,
            "environment_and_sampling_seed_equal": True,
        },
        "attempt": args.attempt,
        "gpu_index": args.gpu_index,
        "python_executable": sys.executable,
        "openpi_client_import_attestation": openpi_client_attestation,
        "study_root": str(study_root),
        "raw_root": str(raw_root),
        "shard_dir": str(shard_dir),
        "policy_endpoint": {
            "host": args.remote_host,
            "port": args.remote_port,
            "websocket_metadata_handshake_passed": True,
            "metadata": endpoint_metadata,
            "inference_requests_sent": 0,
        },
        "runtime_identity": {
            "path": str(runtime_identity),
            "sha256": first["runtime_identity_sha256"],
        },
        "release_gate": {
            "path": str(release_gate),
            "sha256": first["release_gate_sha256"],
            "behavioral_release": True,
        },
        "phase_a_queue_sha256": first["phase_a_queue_sha256"],
        "amendment_sha256": first["amendment_sha256"],
        "frozen_simulator_environment": {
            "DISPLAY": "unset",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "VK_ICD_FILENAMES": FROZEN_VK_ICD,
            "LD_LIBRARY_PATH": FROZEN_LD_LIBRARY_PATH,
            "PATH": FROZEN_PATH,
            "PYTHONPATH": base_worker_environment["PYTHONPATH"],
            "renderer_arguments": list(RENDERER_ARGUMENTS),
            "cache_scope": (
                "/data/users/ali/vla_wam/cache/pi0fast-v3a002/<pod-tag>"
            ),
        },
        "pair_count_planned": len(seeds),
        "pair_count_guard_exit_zero": 0,
        "pair_count_technical_failure": 0,
        "pairs": [],
        "interpretation": (
            "Guard exit zero is launch completion only. Behavioral validity and "
            "outcome require the fail-closed pair compiler."
        ),
    }
    _write_json_atomic(ledger_path, ledger)

    for seed in seeds:
        pair_id = authorized[seed]["pair_id"]
        pair_dir = shard_dir / f"seed{seed}_attempt{args.attempt:02d}"
        pair_dir.mkdir(exist_ok=False)
        action_trace_dir = pair_dir / "action_trace"
        worker_command = bridge_command(
            study_root,
            seed,
            runtime_identity,
            release_gate,
            pair_dir,
            action_trace_dir,
            args.remote_host,
            args.remote_port,
            condition="both",
            attempt=args.attempt,
        )
        worker_command.extend(RENDERER_ARGUMENTS)
        guard_command = _guard_command(
            study_root=study_root,
            pair_dir=pair_dir,
            gpu_index=args.gpu_index,
            pair_id=pair_id,
            seed=seed,
            worker_command=worker_command,
        )
        stdout_log = pair_dir / "pair_stdout_stderr.log"
        pair_record: dict[str, Any] = {
            "seed": seed,
            "pair_id": pair_id,
            "cell_ids": authorized[seed]["cell_ids"],
            "relations": list(RELATIONS),
            "condition": "both",
            "attempt_id": f"{args.shard_id}:seed{seed}:attempt{args.attempt:02d}",
            "status": "launching",
            "started_at_utc": utc_now(),
            "completed_at_utc": None,
            "pair_dir": str(pair_dir),
            "stdout_stderr_log": str(stdout_log),
            "worker_command": worker_command,
            "worker_command_sha256": canonical_sha256(worker_command),
            "guard_command": guard_command,
            "guard_command_sha256": canonical_sha256(guard_command),
            "guard_exit_code": None,
            "error": None,
            "retained_paths": {
                "state_capture_dir": str(pair_dir / "state_capture"),
                "action_trace_dir": str(action_trace_dir),
                "thermal_events": str(pair_dir / "thermal_events.jsonl"),
                "runtime_interventions": str(
                    pair_dir / f"runtime_interventions_{MODEL_ID}.json"
                ),
                "invalid_attempts": str(
                    pair_dir / f"invalid_attempts_{MODEL_ID}.json"
                ),
            },
        }
        worker_environment, cache_paths = _pair_worker_environment(
            base_worker_environment,
            pod_tag=args.pod_tag,
        )
        pair_record["writable_cache_paths"] = cache_paths
        ledger["pairs"].append(pair_record)
        _write_json_atomic(ledger_path, ledger)
        print(
            json.dumps(
                {
                    "event": "pair_launching",
                    "seed": seed,
                    "pair_id": pair_id,
                    "pair_dir": str(pair_dir),
                },
                sort_keys=True,
            ),
            flush=True,
        )

        try:
            with stdout_log.open("xb") as log_handle:
                completed = subprocess.run(
                    guard_command,
                    cwd=pair_dir,
                    env=worker_environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            pair_record["guard_exit_code"] = completed.returncode
            if completed.returncode == 0:
                pair_record["status"] = (
                    "guard_exit_zero_pending_fail_closed_pair_compilation"
                )
                ledger["pair_count_guard_exit_zero"] += 1
            else:
                pair_record["status"] = "technical_failure_guard_exit_nonzero"
                pair_record["error"] = (
                    f"native process-group thermal guard exited {completed.returncode}"
                )
                ledger["pair_count_technical_failure"] += 1
        except Exception as error:  # continue to the next registered pair
            pair_record["status"] = "technical_failure_launcher_exception"
            pair_record["error"] = f"{type(error).__name__}: {error}"
            pair_record["launcher_traceback"] = traceback.format_exc()
            ledger["pair_count_technical_failure"] += 1

        pair_record["completed_at_utc"] = utc_now()
        if stdout_log.is_file():
            pair_record["stdout_stderr_log_sha256"] = sha256_file(stdout_log)
            pair_record["stdout_stderr_log_bytes"] = stdout_log.stat().st_size
        pair_record["retained_path_presence_after_launch"] = {
            name: Path(path).exists()
            for name, path in pair_record["retained_paths"].items()
        }
        _write_json_atomic(ledger_path, ledger)
        print(
            json.dumps(
                {
                    "event": "pair_finished",
                    "seed": seed,
                    "pair_id": pair_id,
                    "status": pair_record["status"],
                    "guard_exit_code": pair_record["guard_exit_code"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    ledger["completed_at_utc"] = utc_now()
    if ledger["pair_count_technical_failure"]:
        ledger["status"] = "completed_with_technical_failures"
    else:
        ledger["status"] = "all_guard_launches_exit_zero_pending_compilation"
    _write_json_atomic(ledger_path, ledger)
    print(
        json.dumps(
            {
                "event": "shard_finished",
                "status": ledger["status"],
                "ledger": str(ledger_path),
                "guard_exit_zero": ledger["pair_count_guard_exit_zero"],
                "technical_failures": ledger["pair_count_technical_failure"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 2 if ledger["pair_count_technical_failure"] else 0


if __name__ == "__main__":
    sys.exit(main())
