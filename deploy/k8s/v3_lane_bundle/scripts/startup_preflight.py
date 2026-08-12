#!/usr/bin/env python3
"""Fail-closed startup qualification for a V3 Kubernetes lane.

The launch ConfigMap supplies the exact commands and hashes; Kubernetes' Downward
API supplies the live pod identity.  This module writes only runtime evidence.
It deliberately does not create the runner's write-once episode directory.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "vla-wam-v3-k8s-lane-startup-preflight-v1"
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "NVIDIA_DRIVER_CAPABILITIES",
    "VK_ICD_FILENAMES",
    "VK_DRIVER_FILES",
    "DISPLAY",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PYTHONNOUSERSITE",
    "PYTHONUNBUFFERED",
    "OMNI_KIT_ACCEPT_EULA",
    "HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "WARP_CACHE_PATH",
    "MPLCONFIGDIR",
    "TMPDIR",
    "FFMPEG_BIN",
    "PYTHON_BIN",
)
RUNTIME_DIRECTORY_KEYS = (
    "HOME",
    "XDG_CACHE_HOME",
    "XDG_CONFIG_HOME",
    "XDG_RUNTIME_DIR",
    "WARP_CACHE_PATH",
    "MPLCONFIGDIR",
    "TMPDIR",
)
HASH_BUFFER_BYTES = 8 * 1024 * 1024


class PreflightError(RuntimeError):
    """The lane is infrastructure-invalid and must not launch behavior."""


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(HASH_BUFFER_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash a directory without following links or depending on mtime/uid."""

    root = path.resolve()
    digest = hashlib.sha256()
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        if entry.is_symlink():
            raise PreflightError(f"checkpoint contains a forbidden symlink: {entry}")
        if entry.is_dir():
            digest.update(b"D\0" + relative + b"\0")
        elif entry.is_file():
            digest.update(b"F\0" + relative + b"\0")
            digest.update(str(entry.stat().st_size).encode("ascii") + b"\0")
            digest.update(sha256_file(entry).encode("ascii") + b"\0")
        else:
            raise PreflightError(f"checkpoint contains a non-file entry: {entry}")
    return digest.hexdigest()


def hash_path(path: Path) -> tuple[str, str]:
    if path.is_file():
        return "file", sha256_file(path)
    if path.is_dir():
        return "tree-v1", sha256_tree(path)
    raise PreflightError(f"hash-bound path does not exist: {path}")


def _require_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PreflightError(f"launch config requires nonempty string {key!r}")
    return value


def _require_argv(mapping: Mapping[str, Any], key: str) -> list[str]:
    value = mapping.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise PreflightError(f"launch config requires nonempty string array {key!r}")
    return list(value)


def require_absolute_executable(raw: str, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
        raise PreflightError(f"{label} must be an existing executable absolute path: {raw!r}")
    return path


def load_launch_config(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise PreflightError(f"launch config is missing: {path}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256.lower():
        raise PreflightError(
            f"launch config digest changed: expected {expected_sha256.lower()}, got {actual_sha256}"
        )
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"launch config is not readable JSON: {path}") from exc
    if not isinstance(config, dict):
        raise PreflightError("launch config must be a JSON object")
    return config, actual_sha256


def acquire_attempt_lock(output_parent: Path, role: str, lane_id: str, attempt_id: str):
    lock_path = output_parent / f".{role}-lane-{lane_id}-attempt-{attempt_id}.lock"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(lock_path, flags, 0o640)
    except FileExistsError as exc:
        raise PreflightError(f"attempt lock already exists; use a fresh attempt identity: {lock_path}") from exc
    handle = os.fdopen(fd, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise PreflightError(f"attempt lock is already owned: {lock_path}") from exc
    return handle, lock_path


def reserve_policy_port(port: int):
    """Acquire an O_EXCL attempt lock and prove that the TCP port is bindable."""

    if not 1 <= port <= 65535:
        raise PreflightError(f"invalid policy port: {port}")
    path = Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")) / f"v3-lane-policy-port-{port}.lock"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PreflightError(f"policy port lock already exists: {path}") from exc
    handle = os.fdopen(fd, "w", encoding="utf-8")
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    port_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        port_probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        port_probe.bind(("0.0.0.0", port))
        port_probe.listen(1)
    except BaseException:
        port_probe.close()
        handle.close()
        path.unlink(missing_ok=True)
        raise
    # The frozen policy server cannot accept a pre-opened socket.  Close this
    # probe before exec and retain the O_EXCL lock FD.  The unique per-attempt
    # Job/Service makes the remaining bind race small; the server fails closed
    # if another process nevertheless intervenes.
    port_probe.close()
    return handle, path


def _run(argv: Sequence[str], *, timeout: float = 120.0, input_bytes: bytes | None = None) -> dict[str, Any]:
    started = time.monotonic()
    completed = subprocess.run(
        list(argv),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )
    result = {
        "argv": list(argv),
        "returncode": completed.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout_tail": completed.stdout.decode("utf-8", errors="replace")[-8192:],
        "stderr_tail": completed.stderr.decode("utf-8", errors="replace")[-8192:],
    }
    if completed.returncode != 0:
        raise PreflightError(f"command failed ({completed.returncode}): {list(argv)!r}; {result['stderr_tail']}")
    return result


def prepare_runtime_directories(runtime_root: Path = Path("/lane-runtime")) -> dict[str, Any]:
    """Create only the isolated emptyDir-backed cache tree, never episode output."""

    if not runtime_root.is_absolute() or runtime_root.is_symlink() or not runtime_root.is_dir():
        raise PreflightError(f"runtime root must be an existing nonsymlink directory: {runtime_root}")
    rows: dict[str, Any] = {}
    for key in RUNTIME_DIRECTORY_KEYS:
        raw = os.environ.get(key)
        if not raw:
            raise PreflightError(f"isolated runtime environment lacks {key}")
        path = Path(raw)
        if not path.is_absolute():
            raise PreflightError(f"isolated runtime path is not absolute: {key}={path}")
        try:
            relative = path.relative_to(runtime_root)
        except ValueError as exc:
            raise PreflightError(f"isolated runtime path escapes {runtime_root}: {key}={path}") from exc
        if not relative.parts:
            raise PreflightError(f"isolated runtime path cannot equal its root: {key}={path}")
        current = runtime_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise PreflightError(f"isolated runtime path contains a symlink: {key}={current}")
            if current.exists():
                if not current.is_dir():
                    raise PreflightError(f"isolated runtime component is not a directory: {key}={current}")
                continue
            current.mkdir(mode=0o700)
            current.chmod(0o700)
        rows[key] = {"path": str(path), "created_or_verified": True, "under_runtime_root": True}
    try:
        descriptor = os.open(runtime_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise PreflightError(f"could not fsync isolated runtime root: {runtime_root}") from exc
    return rows


def verify_runtime_directories() -> dict[str, Any]:
    for key in ("DISPLAY", "LD_PRELOAD"):
        if os.environ.get(key) not in (None, ""):
            raise PreflightError(f"headless runtime requires empty {key}")
    rows: dict[str, Any] = {}
    for key in RUNTIME_DIRECTORY_KEYS:
        raw = os.environ.get(key)
        if not raw:
            raise PreflightError(f"isolated runtime environment lacks {key}")
        path = Path(raw)
        if not path.is_dir() or not os.access(path, os.W_OK | os.X_OK):
            raise PreflightError(f"isolated runtime directory is not writable: {key}={path}")
        rows[key] = {"path": str(path), "writable": True}
    return rows


def verify_writable_parent(output_parent: Path) -> dict[str, Any]:
    if not output_parent.is_absolute() or not output_parent.is_dir():
        raise PreflightError(f"OUTPUT_PARENT must be an existing absolute directory: {output_parent}")
    if not os.access(output_parent, os.W_OK | os.X_OK):
        raise PreflightError(f"OUTPUT_PARENT is not writable: {output_parent}")
    try:
        fd, name = tempfile.mkstemp(prefix=".v3-lane-write-probe-", dir=output_parent)
        os.write(fd, b"v3-lane\n")
        os.fsync(fd)
        os.close(fd)
        Path(name).unlink()
    except BaseException as exc:
        raise PreflightError(f"writable-parent create/fsync/unlink probe failed: {output_parent}") from exc
    return {"path": str(output_parent), "writable_parent": True, "create_fsync_unlink": True}


def verify_cuda(config: Mapping[str, Any]) -> dict[str, Any]:
    custom = config.get("cuda_probe_argv")
    if custom is not None:
        if not isinstance(custom, list) or not custom or not all(isinstance(item, str) and item for item in custom):
            raise PreflightError("cuda_probe_argv must be a nonempty string array")
        require_absolute_executable(custom[0], "cuda_probe_argv[0]")
        result = _run(custom, timeout=180.0)
        return {"device": "cuda:0", "custom_kernel_probe": result}

    python_bin = str(require_absolute_executable(os.environ.get("PYTHON_BIN", ""), "PYTHON_BIN"))
    source = (
        "import json, torch\n"
        "assert torch.cuda.is_available(), 'torch CUDA unavailable'\n"
        "x=torch.arange(4096,device='cuda:0',dtype=torch.float32)\n"
        "y=(x*x+3*x).sum()\n"
        "torch.cuda.synchronize()\n"
        "expected=sum(float(i*i+3*i) for i in range(4096))\n"
        "assert abs(float(y.cpu())-expected) <= max(1.0,abs(expected)*1e-6)\n"
        "print(json.dumps({'available':True,'device':'cuda:0','device_name':torch.cuda.get_device_name(0),'kernel_result':float(y.cpu())}))\n"
    )
    result = _run([python_bin, "-c", source], timeout=180.0)
    try:
        payload = json.loads(result["stdout_tail"].splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise PreflightError("CUDA kernel probe did not emit valid JSON") from exc
    return {**payload, "probe": result}


def verify_vulkan_and_render(config: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    contract = _require_string(config, "vulkan_contract")
    if contract != "isaac_app_launcher_rtx_frame_under_bound_vk_icd":
        raise PreflightError("vulkan_contract differs from the supported live-operable proof")
    probe_argv = _require_argv(config, "render_probe_argv")
    require_absolute_executable(probe_argv[0], "render_probe_argv[0]")
    rendered_frame = evidence_dir / "rendered-frame.png"
    if rendered_frame.exists():
        raise PreflightError(f"rendered-frame evidence unexpectedly exists: {rendered_frame}")
    expanded = [item.replace("{rendered_frame}", str(rendered_frame)) for item in probe_argv]
    if expanded == probe_argv:
        raise PreflightError("render_probe_argv must contain the {rendered_frame} output placeholder")
    probe = _run(expanded, timeout=float(config.get("render_probe_timeout_seconds", 300)))
    if not rendered_frame.is_file() or rendered_frame.stat().st_size <= 0:
        raise PreflightError("render probe did not create a nonempty rendered frame")
    return {
        "vulkan_contract": contract,
        "contract_explanation": "successful Isaac AppLauncher realtime RTX camera capture under the exact NVIDIA Vulkan ICD proves Vulkan initialization and one rendered frame",
        "render_probe": probe,
        "rendered_frame": {
            "path": str(rendered_frame),
            "bytes": rendered_frame.stat().st_size,
            "sha256": sha256_file(rendered_frame),
        },
    }


def verify_imports(config: Mapping[str, Any], *, require_curobo: bool = True) -> dict[str, Any]:
    imports = config.get("python_imports")
    if not isinstance(imports, list) or not imports or not all(isinstance(item, str) and item for item in imports):
        raise PreflightError("python_imports must be a nonempty string array")
    if require_curobo and not any(name == "curobo" or name.startswith("curobo.") for name in imports):
        raise PreflightError("python_imports must include the exact CuRobo import")
    python_bin = str(require_absolute_executable(os.environ.get("PYTHON_BIN", ""), "PYTHON_BIN"))
    source = (
        "import importlib,json,pathlib,sys\n"
        "names=json.loads(sys.argv[1])\n"
        "rows=[]\n"
        "for name in names:\n"
        " m=importlib.import_module(name)\n"
        " raw=getattr(m,'__file__',None)\n"
        " rows.append({'module':name,'module_file':str(pathlib.Path(raw).resolve()) if raw else None,'version':str(getattr(m,'__version__','not_exposed'))})\n"
        "print(json.dumps(rows,sort_keys=True))\n"
    )
    probe = _run([python_bin, "-c", source, json.dumps(imports)], timeout=180.0)
    try:
        rows = json.loads(probe["stdout_tail"].splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise PreflightError("exact Python/CuRobo import probe did not emit valid JSON") from exc
    return {
        "python_bin": python_bin,
        "exact_imports": imports,
        "curobo_required": require_curobo,
        "results": rows,
        "probe": probe,
    }


def verify_policy_readiness_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = _require_string(config, "readiness_contract")
    allowed = {"tcp_bind_after_checkpoint_load", "metadata_jsonl_after_checkpoint_load"}
    if contract not in allowed:
        raise PreflightError(f"unsupported policy readiness contract: {contract!r}")
    return {"contract": contract, "checkpoint_loaded_before_ready": True}


def verify_ffmpeg(evidence_dir: Path) -> dict[str, Any]:
    ffmpeg = str(require_absolute_executable(os.environ.get("FFMPEG_BIN", ""), "FFMPEG_BIN"))
    encoded = evidence_dir / "ffmpeg-encode.mp4"
    decoded = evidence_dir / "ffmpeg-decode.raw"
    encode = _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=64x64:r=2:d=1",
            "-frames:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(encoded),
        ]
    )
    if not encoded.is_file() or encoded.stat().st_size <= 0:
        raise PreflightError("ffmpeg encode probe produced no video")
    decode = _run(
        [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(encoded),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-y",
            str(decoded),
        ]
    )
    expected_bytes = 64 * 64 * 3
    if not decoded.is_file() or decoded.stat().st_size != expected_bytes:
        raise PreflightError(
            f"ffmpeg decode probe differs: expected {expected_bytes} bytes, got "
            f"{decoded.stat().st_size if decoded.exists() else 'missing'}"
        )
    return {
        "ffmpeg_encode": encode,
        "ffmpeg_decode": decode,
        "encoded_mp4": {"path": str(encoded), "bytes": encoded.stat().st_size, "sha256": sha256_file(encoded)},
        "decoded_raw": {"path": str(decoded), "bytes": decoded.stat().st_size, "sha256": sha256_file(decoded)},
    }


def verify_checkpoint(config: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(_require_string(config, "checkpoint_path"))
    if not path.is_absolute():
        raise PreflightError("checkpoint_path must be absolute")
    expected = _require_string(config, "checkpoint_sha256").lower()
    kind, actual = hash_path(path)
    if actual != expected:
        raise PreflightError(f"checkpoint digest changed: expected {expected}, got {actual}")
    return {"path": str(path.resolve()), "hash_kind": kind, "checkpoint_sha256": actual}


def verify_file_bindings(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    bindings = config.get("file_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise PreflightError("file_bindings must be a nonempty array")
    checked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(bindings):
        if not isinstance(raw, dict):
            raise PreflightError(f"file_bindings[{index}] must be an object")
        path = Path(_require_string(raw, "path"))
        if not path.is_absolute():
            raise PreflightError(f"file_bindings[{index}].path must be absolute")
        expected_sha256 = _require_string(raw, "sha256").lower()
        expected_bytes = raw.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise PreflightError(f"file_bindings[{index}].bytes must be a nonnegative integer")
        resolved = str(path.resolve())
        if resolved in seen:
            raise PreflightError(f"file binding repeats path: {resolved}")
        seen.add(resolved)
        if not path.is_file():
            raise PreflightError(f"bound launch file is missing: {path}")
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
            raise PreflightError(
                f"bound launch file changed: {path}; expected {expected_bytes}/{expected_sha256}, "
                f"got {actual_bytes}/{actual_sha256}"
            )
        checked.append({"path": resolved, "bytes": actual_bytes, "sha256": actual_sha256})
    return checked


def query_gpu_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    nvidia_smi = str(
        require_absolute_executable(_require_string(config, "nvidia_smi_bin"), "nvidia_smi_bin")
    )
    result = _run(
        [
            nvidia_smi,
            "--query-gpu=uuid,name,driver_version",
            "--format=csv,noheader,nounits",
        ],
        timeout=30.0,
    )
    rows = [line.strip() for line in result["stdout_tail"].splitlines() if line.strip()]
    if len(rows) != 1:
        raise PreflightError(f"one-GPU lane exposed {len(rows)} physical GPUs, expected exactly one")
    parts = [item.strip() for item in rows[0].split(",")]
    if len(parts) != 3 or not parts[0].startswith("GPU-"):
        raise PreflightError(f"unexpected nvidia-smi identity row: {rows[0]!r}")
    expected_name = _require_string(config, "expected_gpu_name")
    expected_driver = _require_string(config, "expected_driver_version")
    if parts[1] != expected_name:
        raise PreflightError(f"GPU name differs: expected {expected_name!r}, got {parts[1]!r}")
    if parts[2] != expected_driver:
        raise PreflightError(f"GPU driver differs: expected {expected_driver!r}, got {parts[2]!r}")
    return {"gpu_uuid": parts[0], "gpu_name": parts[1], "driver_version": parts[2]}


def wait_for_policy(config: Mapping[str, Any]) -> dict[str, Any] | None:
    wait = config.get("policy_wait")
    if wait is None:
        return None
    if not isinstance(wait, dict):
        raise PreflightError("policy_wait must be an object")
    host = _require_string(wait, "host")
    port = int(wait.get("port", 0))
    mode = wait.get("mode", "tcp")
    if mode not in {"tcp", "metadata_jsonl"}:
        raise PreflightError("policy_wait.mode must be tcp or metadata_jsonl")
    service_identity = wait.get("service_identity")
    if not isinstance(service_identity, dict) or not service_identity:
        raise PreflightError("policy_wait.service_identity must bind the unique Service selector")
    expected = wait.get("expected_response")
    if mode == "metadata_jsonl" and (not isinstance(expected, dict) or not expected):
        raise PreflightError("metadata_jsonl policy wait requires expected_response")
    request = wait.get("request", {"type": "lane_metadata"})
    if mode == "metadata_jsonl" and not isinstance(request, dict):
        raise PreflightError("policy_wait.request must be an object")
    deadline = time.monotonic() + float(wait.get("timeout_seconds", 900))
    last_error = "not attempted"
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=5.0) as connection:
                if mode == "tcp":
                    return {
                        "mode": mode,
                        "host": host,
                        "port": port,
                        "tcp_connected": True,
                        "service_identity": service_identity,
                    }
                connection.sendall(json.dumps(request, sort_keys=True).encode("utf-8") + b"\n")
                connection.settimeout(5.0)
                payload = b""
                while not payload.endswith(b"\n") and len(payload) <= 65536:
                    block = connection.recv(4096)
                    if not block:
                        break
                    payload += block
            response = json.loads(payload.decode("utf-8"))
            if not isinstance(response, dict):
                raise ValueError("metadata response is not an object")
            mismatches = {key: {"expected": value, "actual": response.get(key)} for key, value in expected.items() if response.get(key) != value}
            if mismatches:
                raise ValueError(f"policy metadata mismatch: {mismatches}")
            return {
                "mode": mode,
                "host": host,
                "port": port,
                "request": request,
                "response": response,
                "matched": True,
                "service_identity": service_identity,
            }
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(float(wait.get("poll_seconds", 2)))
    raise PreflightError(f"policy readiness/metadata handshake timed out: {last_error}")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o640)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_preflight(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    config_sha256: str,
    evidence_dir: Path,
    output_parent: Path,
    role: str,
    lane_id: str,
    attempt_id: str,
    pod_identity: Mapping[str, str],
    experiment_argv: Sequence[str],
    image_digest: str,
) -> dict[str, Any]:
    """Run all gates and retain a failure report before raising."""

    report_path = evidence_dir / "runtime-preflight.json"
    marker_path = evidence_dir / "runtime-preflight.passed"
    startup_path = evidence_dir / "runtime-startup.json"
    effective_environment = {key: os.environ.get(key) for key in SAFE_ENVIRONMENT_KEYS}
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "status": "running",
        "passed": False,
        "role": role,
        "lane_id": lane_id,
        "attempt_id": attempt_id,
        "pod_identity": dict(pod_identity),
        "image_digest": image_digest,
        "pid": os.getpid(),
        "argv": list(experiment_argv),
        "launch_config": {"path": str(config_path), "sha256": config_sha256},
        "effective_environment": effective_environment,
        "output_parent": str(output_parent),
        "episode_output_dir_precreated": False,
    }
    try:
        checks: dict[str, Any] = {}
        checks["runtime_directories"] = verify_runtime_directories()
        checks["writable_parent"] = verify_writable_parent(output_parent)
        checks["cuda_kernel"] = verify_cuda(config)
        if role == "simulator":
            checks["vulkan_and_rendered_frame"] = verify_vulkan_and_render(config, evidence_dir)
            checks["python_imports"] = verify_imports(config, require_curobo=True)
        elif role == "policy":
            checks["python_imports"] = verify_imports(config, require_curobo=False)
            checks["policy_readiness_contract"] = verify_policy_readiness_contract(config)
        else:
            raise PreflightError(f"unsupported lane role: {role!r}")
        checks["ffmpeg_encode_decode"] = verify_ffmpeg(evidence_dir)
        checks["file_bindings"] = verify_file_bindings(config)
        checkpoint = verify_checkpoint(config)
        checks["checkpoint_digest"] = checkpoint
        gpu = query_gpu_identity(config)
        checks["policy_readiness"] = wait_for_policy(config) if role == "simulator" else None
        report = {
            **base,
            "status": "passed_startup_preflight",
            "passed": True,
            "gpu_uuid": gpu["gpu_uuid"],
            "gpu": gpu,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "checks": checks,
            "completed_at_utc": utc_now(),
        }
        _write_json_exclusive(report_path, report)
        _write_json_exclusive(
            startup_path,
            {
                "schema_version": "vla-wam-v3-k8s-lane-runtime-startup-v1",
                "created_at_utc": utc_now(),
                "pod_identity": dict(pod_identity),
                "role": role,
                "lane_id": lane_id,
                "attempt_id": attempt_id,
                "effective_environment": effective_environment,
                "gpu_uuid": gpu["gpu_uuid"],
                "image_digest": image_digest,
                "pid": os.getpid(),
                "argv": list(experiment_argv),
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "preflight": {"path": str(report_path), "sha256": sha256_file(report_path)},
            },
        )
        marker_payload = json.dumps(
            {"status": "passed", "preflight_sha256": sha256_file(report_path), "startup_sha256": sha256_file(startup_path)},
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        fd = os.open(marker_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(fd, "wb") as handle:
            handle.write(marker_payload)
            handle.flush()
            os.fsync(handle.fileno())
        return report
    except BaseException as exc:
        failure = {
            **base,
            "status": "infrastructure_invalid_startup_preflight",
            "passed": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "completed_at_utc": utc_now(),
        }
        if not report_path.exists():
            _write_json_exclusive(report_path, failure)
        raise
