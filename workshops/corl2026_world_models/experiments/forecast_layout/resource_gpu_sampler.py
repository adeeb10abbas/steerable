#!/usr/bin/env python3
"""Tamper-evident, task-owned ``nvidia-smi`` sampling for GM resource probes.

This module deliberately measures both whole-device memory and the sum of all
driver-reported compute processes.  PyTorch allocator records remain separate.
It has no model, simulator, queue, annotation, or confirmation authority.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import time
from typing import Any, Mapping, Sequence


SAMPLE_SCHEMA = "wmf-gm-gpu-sample-v1"
SUMMARY_SCHEMA = "wmf-gm-gpu-sample-summary-v1"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
MIB = 1024 * 1024


class GpuSampleError(RuntimeError):
    """The resource sampler or its evidence failed closed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GpuSampleError(message)


def compact_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GpuSampleError("sample contains noncanonical JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    supplied = Path(path)
    require(not supplied.is_symlink(), f"sample artifact is a symlink: {supplied}")
    resolved = supplied.resolve()
    require(resolved.is_file(), f"sample artifact is missing: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def signed_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(compact_bytes(result))
    return result


def verify_signed_document(value: Mapping[str, Any], label: str) -> None:
    observed = value.get("payload_sha256")
    require(
        isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None,
        f"{label} signature is invalid",
    )
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(compact_bytes(unsigned)) == observed, f"{label} signature changed")


def _csv_rows(payload: str, columns: Sequence[str], label: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for values in csv.reader(io.StringIO(payload)):
        if not values or not any(value.strip() for value in values):
            continue
        require(len(values) == len(columns), f"{label} CSV shape changed")
        rows.append({key: value.strip() for key, value in zip(columns, values)})
    return rows


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, str) and re.fullmatch(r"[0-9]+", value) is not None,
            f"{label} is not an integer")
    result = int(value)
    require(result >= minimum, f"{label} is below its minimum")
    return result


def query_nvidia_smi(binary: Path | str = "nvidia-smi") -> dict[str, Any]:
    """Return one bounded, normalized device/process snapshot."""

    device_columns = (
        "index", "uuid", "name", "driver_version", "memory.total", "memory.used",
        "utilization.gpu",
    )
    # Exact compute-apps field already exercised by the workshop's qualified
    # N3/D1 topology probes on GM.
    process_columns = ("gpu_uuid", "pid", "process_name", "used_memory")

    def run(prefix: str, columns: Sequence[str]) -> str:
        try:
            completed = subprocess.run(
                [
                    str(binary),
                    prefix + ",".join(columns),
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GpuSampleError("nvidia-smi query unavailable") from error
        require(completed.returncode == 0, "nvidia-smi query failed")
        require(len(completed.stdout.encode("utf-8")) <= 1024 * 1024,
                "nvidia-smi output exceeded bound")
        return completed.stdout

    started_wall = time.time_ns()
    started_monotonic = time.monotonic_ns()
    raw_devices = _csv_rows(run("--query-gpu=", device_columns), device_columns, "GPU")
    raw_processes = _csv_rows(
        run("--query-compute-apps=", process_columns), process_columns, "compute process"
    )
    finished_monotonic = time.monotonic_ns()
    finished_wall = time.time_ns()

    devices: list[dict[str, Any]] = []
    for row in raw_devices:
        uuid = row["uuid"]
        require(uuid.startswith("GPU-") and len(uuid) > 8, "GPU UUID is invalid")
        total = _integer(row["memory.total"], "GPU total MiB", minimum=1)
        used = _integer(row["memory.used"], "GPU used MiB")
        require(used <= total, "GPU used memory exceeds total")
        devices.append({
            "index": _integer(row["index"], "GPU index"),
            "uuid": uuid,
            "name": row["name"],
            "driver_version": row["driver_version"],
            "memory_total_mib": total,
            "memory_used_mib": used,
            "utilization_gpu_percent": _integer(
                row["utilization.gpu"], "GPU utilization", minimum=0
            ),
        })
    require(len({row["uuid"] for row in devices}) == len(devices), "GPU UUID duplicated")

    processes: list[dict[str, Any]] = []
    visible = {row["uuid"] for row in devices}
    for row in raw_processes:
        # Some driver builds spell unavailable process memory as N/A.  That
        # cannot support a whole-process peak and is therefore not coerced.
        used = _integer(row["used_memory"], "compute-process used MiB")
        require(row["gpu_uuid"] in visible, "compute process escaped visible GPU inventory")
        processes.append({
            "gpu_uuid": row["gpu_uuid"],
            "pid": _integer(row["pid"], "compute-process PID", minimum=1),
            "process_name": row["process_name"],
            "used_gpu_memory_mib": used,
        })
    return {
        "wall_started_ns": started_wall,
        "wall_finished_ns": finished_wall,
        "monotonic_started_ns": started_monotonic,
        "monotonic_finished_ns": finished_monotonic,
        "devices": devices,
        "compute_processes": processes,
    }


def _parent_pid(pid: int) -> int | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        tail = payload[payload.rfind(")") + 2 :].split()
        # tail[0] is state (field 3), tail[1] is PPID (field 4).
        return int(tail[1])
    except (OSError, ValueError, IndexError):
        return None


def classify_pid(pid: int, roots: Mapping[str, int]) -> str | None:
    """Classify a live PID by walking its Linux parent chain."""

    wanted = {int(value): str(key) for key, value in roots.items()}
    current = int(pid)
    visited: set[int] = set()
    for _ in range(128):
        if current in wanted:
            return wanted[current]
        if current <= 1 or current in visited:
            return None
        visited.add(current)
        parent = _parent_pid(current)
        if parent is None:
            return None
        current = parent
    return None


class _SampleJournal:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        require(not self.path.exists() and not self.path.is_symlink(),
                "GPU sample journal already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        require(not self.path.parent.is_symlink(), "GPU sample parent is a symlink")
        self.sequence = 0
        self.tail: str | None = None

    def append(self, value: Mapping[str, Any]) -> None:
        base = {
            "schema_version": SAMPLE_SCHEMA,
            "sequence": self.sequence,
            "previous_event_sha256": self.tail,
            **dict(value),
        }
        require(len(base) == len(value) + 3, "GPU sample reserved fields collided")
        digest = sha256_bytes(compact_bytes(base))
        row = dict(base, event_sha256=digest)
        with self.path.open("ab") as stream:
            stream.write(compact_bytes(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.sequence += 1
        self.tail = digest


class GpuSampler:
    """Background sampler with mutable phase and task process roots."""

    def __init__(
        self,
        path: Path,
        *,
        interval_seconds: float,
        nvidia_smi: Path | str = "nvidia-smi",
    ) -> None:
        require(math.isfinite(interval_seconds) and interval_seconds > 0,
                "GPU sample interval is invalid")
        self.journal = _SampleJournal(path)
        self.interval_seconds = float(interval_seconds)
        self.nvidia_smi = nvidia_smi
        self._phase = "preflight"
        self._roots: dict[str, int] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None

    def set_phase(self, phase: str) -> None:
        require(isinstance(phase, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", phase),
                "GPU sampler phase is invalid")
        with self._lock:
            self._phase = phase

    def set_root(self, name: str, pid: int) -> None:
        require(re.fullmatch(r"[a-z][a-z0-9_]{0,31}", name) is not None,
                "GPU sampler root name is invalid")
        require(type(pid) is int and pid > 1, "GPU sampler root PID is invalid")
        with self._lock:
            require(name not in self._roots, "GPU sampler root was replaced")
            self._roots[name] = pid

    def _sample_once(self) -> None:
        with self._lock:
            phase = self._phase
            roots = dict(self._roots)
        snapshot = query_nvidia_smi(self.nvidia_smi)
        processes = []
        for row in snapshot.pop("compute_processes"):
            owner = classify_pid(row["pid"], roots)
            processes.append({**row, "task_process_owner": owner})
        self.journal.append({
            "phase": phase,
            "task_process_roots": roots,
            **snapshot,
            "compute_processes": processes,
        })

    def start(self) -> None:
        require(self._thread is None, "GPU sampler already started")

        def target() -> None:
            try:
                while not self._stop.is_set():
                    started = time.monotonic()
                    self._sample_once()
                    remaining = self.interval_seconds - (time.monotonic() - started)
                    if remaining > 0:
                        self._stop.wait(remaining)
            except BaseException as error:  # propagated by stop()
                self._error = error

        self._thread = threading.Thread(target=target, name="wmf-gpu-sampler", daemon=False)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        require(self._thread is not None, "GPU sampler was not started")
        self._stop.set()
        self._thread.join(timeout=max(30.0, self.interval_seconds * 10))
        require(not self._thread.is_alive(), "GPU sampler did not stop")
        if self._error is not None:
            raise GpuSampleError(f"GPU sampler failed: {type(self._error).__name__}") from self._error
        require(self.journal.sequence > 0, "GPU sampler produced no samples")
        return {
            "artifact": file_identity(self.journal.path),
            "sample_count": self.journal.sequence,
            "tail_event_sha256": self.journal.tail,
            "configured_interval_seconds": self.interval_seconds,
        }


def load_samples(path: Path) -> list[dict[str, Any]]:
    supplied = Path(path)
    require(not supplied.is_symlink() and supplied.is_file(), "GPU sample journal is invalid")
    rows: list[dict[str, Any]] = []
    previous: str | None = None
    for sequence, raw in enumerate(supplied.read_bytes().splitlines()):
        require(bool(raw), "GPU sample journal contains blank row")
        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GpuSampleError("GPU sample journal contains invalid JSON") from error
        require(isinstance(row, dict), "GPU sample row is not an object")
        require(row.get("schema_version") == SAMPLE_SCHEMA, "GPU sample schema changed")
        require(row.get("sequence") == sequence, "GPU sample sequence changed")
        require(row.get("previous_event_sha256") == previous, "GPU sample chain changed")
        observed = row.get("event_sha256")
        require(isinstance(observed, str) and SHA256_RE.fullmatch(observed) is not None,
                "GPU sample signature is invalid")
        unsigned = dict(row)
        unsigned.pop("event_sha256")
        require(sha256_bytes(compact_bytes(unsigned)) == observed, "GPU sample signature changed")
        previous = observed
        rows.append(row)
    require(bool(rows), "GPU sample journal is empty")
    return rows


def _interval_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    start_wall_ns: int,
    end_wall_ns: int,
) -> dict[str, Any]:
    require(type(start_wall_ns) is int and type(end_wall_ns) is int and end_wall_ns > start_wall_ns,
            "GPU summary interval is invalid")
    selected = [
        row for row in rows
        if start_wall_ns
        <= (int(row["wall_started_ns"]) + int(row["wall_finished_ns"])) // 2
        <= end_wall_ns
    ]
    require(bool(selected), "GPU summary interval has no samples")
    uuids = [row["uuid"] for row in selected[0]["devices"]]
    result_devices = []
    aggregate_device_peak = 0
    aggregate_process_peak = 0
    for uuid in uuids:
        device_used = []
        process_used = []
        owner_peaks: dict[str, int] = {}
        for sample in selected:
            device_rows = [row for row in sample["devices"] if row["uuid"] == uuid]
            require(len(device_rows) == 1, "GPU disappeared within summary interval")
            device_used.append(int(device_rows[0]["memory_used_mib"]) * MIB)
            process_used.append(sum(
                int(row["used_gpu_memory_mib"]) * MIB
                for row in sample["compute_processes"] if row["gpu_uuid"] == uuid
            ))
            owners = {
                str(row["task_process_owner"])
                for row in sample["compute_processes"]
                if row["gpu_uuid"] == uuid
            }
            for owner in owners:
                value = sum(
                    int(row["used_gpu_memory_mib"]) * MIB
                    for row in sample["compute_processes"]
                    if row["gpu_uuid"] == uuid
                    and str(row["task_process_owner"]) == owner
                )
                owner_peaks[owner] = max(owner_peaks.get(owner, 0), value)
        result_devices.append({
            "gpu_uuid": uuid,
            "whole_device_peak_bytes": max(device_used),
            "whole_process_peak_bytes": max(process_used),
            "task_owner_process_peak_bytes": owner_peaks,
        })
    for sample in selected:
        aggregate_device_peak = max(
            aggregate_device_peak,
            sum(int(row["memory_used_mib"]) * MIB for row in sample["devices"]),
        )
        aggregate_process_peak = max(
            aggregate_process_peak,
            sum(int(row["used_gpu_memory_mib"]) * MIB for row in sample["compute_processes"]),
        )
    return {
        "start_wall_time_ns": start_wall_ns,
        "end_wall_time_ns": end_wall_ns,
        "sample_count": len(selected),
        "devices": result_devices,
        "co_sampled_whole_device_peak_bytes": aggregate_device_peak,
        "co_sampled_whole_process_peak_bytes": aggregate_process_peak,
    }


def summarize_samples(
    path: Path,
    *,
    expected_gpu_count: int,
    required_gpu_name: str,
    maximum_gap_seconds: float,
    intervals: Mapping[str, tuple[int, int]],
    minimum_samples_by_interval: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Validate a journal and calculate exact per-interval driver peaks."""

    rows = load_samples(path)
    require(type(expected_gpu_count) is int and expected_gpu_count > 0,
            "expected GPU count is invalid")
    expected_uuids: tuple[str, ...] | None = None
    total_by_uuid: dict[str, int] = {}
    index_by_uuid: dict[str, int] = {}
    unknown_process_rows = 0
    midpoints: list[int] = []
    for row in rows:
        devices = row.get("devices")
        processes = row.get("compute_processes")
        require(isinstance(devices, list) and len(devices) == expected_gpu_count,
                "GPU sample device count changed")
        require(isinstance(processes, list), "GPU sample process inventory changed")
        uuids = tuple(sorted(str(device.get("uuid")) for device in devices))
        if expected_uuids is None:
            expected_uuids = uuids
        require(uuids == expected_uuids, "GPU identity changed during sampling")
        for device in devices:
            require(device.get("name") == required_gpu_name, "sampled GPU model changed")
            total = int(device["memory_total_mib"]) * MIB
            uuid = str(device["uuid"])
            if uuid in total_by_uuid:
                require(total_by_uuid[uuid] == total, "sampled GPU total memory changed")
                require(index_by_uuid[uuid] == int(device["index"]),
                        "sampled GPU logical index changed")
            total_by_uuid[uuid] = total
            index_by_uuid[uuid] = int(device["index"])
        for process in processes:
            if process.get("task_process_owner") is None:
                unknown_process_rows += 1
        midpoints.append((int(row["wall_started_ns"]) + int(row["wall_finished_ns"])) // 2)
    gaps = [right - left for left, right in zip(midpoints, midpoints[1:])]
    maximum_gap_ns = max(gaps, default=0)
    require(
        maximum_gap_ns <= int(float(maximum_gap_seconds) * 1e9),
        "GPU sample coverage gap exceeded contract",
    )
    require(unknown_process_rows == 0, "unowned compute process appeared during resource probe")
    minimum = dict(minimum_samples_by_interval or {})
    summaries = {
        name: _interval_summary(rows, start_wall_ns=start, end_wall_ns=end)
        for name, (start, end) in intervals.items()
    }
    for name, count in minimum.items():
        require(name in summaries, f"minimum sample interval is absent: {name}")
        require(type(count) is int and count > 0, f"minimum sample count is invalid: {name}")
        require(summaries[name]["sample_count"] >= count,
                f"GPU sample interval is undersampled: {name}")
    return signed_document({
        "schema_version": SUMMARY_SCHEMA,
        "status": "passed",
        "sample_journal": file_identity(path),
        "sample_count": len(rows),
        "gpu_count": expected_gpu_count,
        "gpu_uuids": list(expected_uuids or ()),
        "gpu_total_memory_bytes_by_uuid": total_by_uuid,
        "gpu_logical_index_by_uuid": index_by_uuid,
        "unknown_compute_process_rows": 0,
        "maximum_observed_sample_gap_ns": maximum_gap_ns,
        "intervals": summaries,
        "measurement_boundary": (
            "nvidia-smi whole-device/process sampling; PyTorch allocator metrics are separate"
        ),
    })
