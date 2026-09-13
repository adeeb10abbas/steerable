#!/usr/bin/env python3
"""Pinned N3 fixed-observation qualification for WMF-ABLATION-001.

This program deliberately uses the official Cosmos server's ``infer`` method.
It wraps the model object only at the call boundary so it can retain the exact
transformed batch and the jointly generated ``samples["vision"]`` tensor.  The
official server is configured with ``decode_video=False`` for every request;
therefore a no-decode request skips only VAE rendering, never joint prediction.
For a decode request the exact retained latent object is rendered *after* the
returned action has been copied and hashed.

The command runs six generation requests and no robot episode.  Large tensors
stay in the raw attempt directory; the publish directory receives only a small
receipt.  Any mismatch writes a failure receipt and exits nonzero.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Sequence
import uuid

import numpy as np


MODULE_PATH = Path(__file__).resolve()
CONTRACT_PATH = MODULE_PATH.with_name("n3_first_live_contract.json")
OBSERVATION_SCHEMA = "wmf-n3-fixed-observation-v1"
RECEIPT_SCHEMA = "wmf-n3-runtime-qualification-v1"
FAILURE_SCHEMA = "wmf-n3-runtime-qualification-failure-v1"
CONTEXT_RESET_SCOPE = "one_saved_observation_generation_request"
REQUIRED_WIRE_KEYS = (
    "observation/image",
    "observation/joint_position",
    "observation/gripper_position",
)
IMAGE_WIRE_KEYS = REQUIRED_WIRE_KEYS[:1]
SHA256_LENGTH = 64


class N3QualificationError(RuntimeError):
    """The attempt cannot occupy the N3 qualification slot."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise N3QualificationError(f"value is not finite canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "wmf-n3-qualification-contract-v1":
        raise N3QualificationError("unexpected N3 qualification contract schema")
    if value.get("study_id") != "WMF-ABLATION-001" or value.get("model_config") != "N3":
        raise N3QualificationError("N3 qualification contract belongs to another study/model")
    protocol = value.get("fixed_observation_protocol")
    expected = [
        (0, "left_no_decode", "left", False, 3),
        (1, "left_exact_repeat_no_decode", "left", False, 4),
        (2, "right_no_decode", "right", False, 5),
        (3, "left_decode", "left", True, 0),
        (4, "left_exact_repeat_decode", "left", True, 1),
        (5, "right_decode", "right", True, 2),
    ]
    observed = [
        (
            row.get("request_index"),
            row.get("request_id"),
            row.get("prompt_key"),
            row.get("decode"),
            row.get("pair_index"),
        )
        for row in protocol or []
    ]
    if observed != expected:
        raise N3QualificationError("the exact six-request N3 protocol changed")
    runtime = value.get("runtime", {})
    required_runtime = {
        "guidance": 3.0,
        "denoising_steps": 4,
        "shift": 5.0,
        "history_length": 1,
        "conditioning_fps": 15.0,
        "resolution_setting": "480",
        "use_state": True,
        "action_space": "joint_pos",
        "action_dimensions": 8,
        "returned_actions": 32,
        "decode_video_during_official_infer": False,
    }
    if any(runtime.get(key) != wanted for key, wanted in required_runtime.items()):
        raise N3QualificationError("the frozen N3 runtime constants changed")
    return value


class _EventJournal:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"refusing to overwrite N3 event journal: {self.path}")
        self.sequence = 0
        self.tail_sha256: str | None = None
        self._lock = threading.Lock()

    def append(self, kind: str, payload: Mapping[str, Any]) -> None:
        with self._lock:
            base = {
                "sequence": self.sequence,
                "kind": kind,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                "previous_event_sha256": self.tail_sha256,
                "payload": dict(payload),
            }
            digest = _sha256_bytes(_canonical_bytes(base))
            record = dict(base, event_sha256=digest)
            with self.path.open("ab") as handle:
                handle.write(_canonical_bytes(record) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.sequence == 0:
                _fsync_directory(self.path.parent)
            self.sequence += 1
            self.tail_sha256 = digest


def _array_value_identity(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    header = {"kind": "numpy", "dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    digest.update(contiguous.tobytes(order="C"))
    return dict(header, value_sha256=digest.hexdigest())


def _is_torch_tensor(value: Any) -> bool:
    return all(hasattr(value, name) for name in ("detach", "cpu", "contiguous", "element_size"))


def _tensor_value_identity(tensor: Any) -> dict[str, Any]:
    cpu = tensor.detach().cpu().contiguous()
    header = {"kind": "torch", "dtype": str(cpu.dtype), "shape": list(cpu.shape)}
    try:
        raw = cpu.view(np.uint8).numpy().tobytes(order="C")
    except (TypeError, RuntimeError, AttributeError):
        # ``torch.uint8`` is required for bfloat16, which NumPy cannot represent.
        torch = importlib.import_module("torch")
        raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(_canonical_bytes(header))
    digest.update(raw)
    return dict(header, value_sha256=digest.hexdigest())


def value_identity(value: Any) -> dict[str, Any]:
    if _is_torch_tensor(value):
        return _tensor_value_identity(value)
    return _array_value_identity(np.asarray(value))


class _ArtifactWriter:
    """Write a lossless nested payload and a path-independent logical digest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self._counter = 0

    def _write_numpy(self, array: np.ndarray, role: str) -> dict[str, Any]:
        contiguous = np.ascontiguousarray(array)
        if contiguous.dtype.hasobject:
            raise N3QualificationError(f"object array is prohibited in {role}")
        name = f"{self._counter:04d}_{role}.npy"
        self._counter += 1
        path = self.root / name
        with path.open("xb") as handle:
            np.save(handle, contiguous, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        identity = _array_value_identity(contiguous)
        return {
            "__type__": "numpy",
            "artifact": {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)},
            **identity,
        }

    def _write_tensor(self, tensor: Any, role: str) -> dict[str, Any]:
        torch = importlib.import_module("torch")
        cpu = tensor.detach().cpu().contiguous()
        name = f"{self._counter:04d}_{role}.pt"
        self._counter += 1
        path = self.root / name
        with path.open("xb") as handle:
            torch.save(cpu, handle)
            handle.flush()
            os.fsync(handle.fileno())
        identity = _tensor_value_identity(cpu)
        return {
            "__type__": "torch",
            "artifact": {"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)},
            **identity,
        }

    def _freeze(self, value: Any, role: str) -> Any:
        safe_role = "".join(character if character.isalnum() else "_" for character in role)[-80:]
        if _is_torch_tensor(value):
            return self._write_tensor(value, safe_role)
        if isinstance(value, np.ndarray):
            return self._write_numpy(value, safe_role)
        if isinstance(value, np.generic):
            return self._write_numpy(np.asarray(value), safe_role)
        if isinstance(value, Mapping):
            items: dict[str, Any] = {}
            for key, child in value.items():
                if not isinstance(key, (str, int, float, bool)):
                    raise N3QualificationError(f"unsupported mapping key in {role}: {type(key)}")
                encoded = str(key)
                if encoded in items:
                    raise N3QualificationError(f"mapping key collision in {role}: {encoded}")
                items[encoded] = self._freeze(child, f"{role}_{encoded}")
            return {"__type__": "mapping", "items": items}
        if isinstance(value, tuple):
            return {
                "__type__": "tuple",
                "items": [self._freeze(child, f"{role}_{index}") for index, child in enumerate(value)],
            }
        if isinstance(value, list):
            return {
                "__type__": "list",
                "items": [self._freeze(child, f"{role}_{index}") for index, child in enumerate(value)],
            }
        if isinstance(value, Path):
            return {"__type__": "path", "value": str(value)}
        if isinstance(value, bytes):
            return self._write_numpy(np.frombuffer(value, dtype=np.uint8).copy(), safe_role) | {
                "logical_type": "bytes"
            }
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise N3QualificationError(f"non-finite float in {role}")
            return value
        raise N3QualificationError(f"unsupported exact payload value in {role}: {type(value)}")

    def _logical(self, value: Any) -> Any:
        if isinstance(value, list):
            return [self._logical(item) for item in value]
        if not isinstance(value, Mapping):
            return value
        if value.get("__type__") in {"numpy", "torch"}:
            keys = ("__type__", "kind", "dtype", "shape", "value_sha256", "logical_type")
            return {key: value[key] for key in keys if key in value}
        return {key: self._logical(child) for key, child in value.items()}

    def write(self, role: str, value: Any) -> dict[str, Any]:
        structure = self._freeze(value, role)
        manifest = {
            "schema_version": "wmf-lossless-nested-payload-v1",
            "role": role,
            "structure": structure,
            "logical_sha256": _sha256_bytes(_canonical_bytes(self._logical(structure))),
        }
        path = self.root / f"{role}.manifest.json"
        _atomic_json(path, manifest)
        return {
            "manifest_path": str(path),
            "manifest_sha256": sha256_file(path),
            "logical_sha256": manifest["logical_sha256"],
        }


def _run_git(source_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source_root), *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise N3QualificationError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout.decode("utf-8", "surrogateescape")


def inspect_official_server_ast(path: Path) -> dict[str, Any]:
    """Confirm the pinned source still exposes the measurement-only seam."""

    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    service: ast.ClassDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RobolabPolicyService":
            service = node
            break
    if service is None:
        raise N3QualificationError("official source lacks RobolabPolicyService")
    methods = {node.name: node for node in service.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for required in ("__init__", "_build_sample", "_next_seed", "infer"):
        if required not in methods:
            raise N3QualificationError(f"official source lacks {required}")
    infer = methods["infer"]
    calls = [
        node.func.attr
        for node in ast.walk(infer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    string_subscripts = {
        node.slice.value
        for node in ast.walk(infer)
        if isinstance(node, ast.Subscript)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    decode_guards = [
        node
        for node in ast.walk(infer)
        if isinstance(node, ast.If)
        and any(
            isinstance(child, ast.Attribute) and child.attr == "decode_video"
            for child in ast.walk(node.test)
        )
    ]
    if calls.count("generate_samples_from_batch") != 1:
        raise N3QualificationError("official infer must invoke joint generation exactly once")
    if calls.count("decode") != 1 or not decode_guards:
        raise N3QualificationError("official infer decode seam changed")
    if not {"action", "vision"}.issubset(string_subscripts):
        raise N3QualificationError("official infer no longer exposes action and vision samples")
    return {
        "class": service.name,
        "generate_samples_from_batch_calls": calls.count("generate_samples_from_batch"),
        "decode_calls": calls.count("decode"),
        "decode_guarded_by_cfg_decode_video": True,
        "sample_keys": sorted({"action", "vision"} & string_subscripts),
    }


def _tracked_inventory(source_root: Path) -> dict[str, Any]:
    raw = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        timeout=60,
    ).stdout
    relative_paths = sorted(item for item in raw.split(b"\0") if item)
    aggregate = hashlib.sha256()
    total_bytes = 0
    for encoded in relative_paths:
        relative = os.fsdecode(encoded)
        path = source_root / relative
        if not path.is_file():
            raise N3QualificationError(f"tracked source path is missing: {relative}")
        digest = sha256_file(path)
        total_bytes += path.stat().st_size
        aggregate.update(f"{digest}  {relative}\n".encode("utf-8", "surrogateescape"))
    return {
        "tracked_file_count": len(relative_paths),
        "tracked_bytes": total_bytes,
        "tracked_aggregate_sha256": aggregate.hexdigest(),
    }


def validate_external_source(source_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    source_root = Path(source_root).resolve()
    expected = contract["external_source"]
    if not (source_root / ".git").exists():
        raise N3QualificationError(f"N3 external source is not a git checkout: {source_root}")
    commit = _run_git(source_root, "rev-parse", "HEAD").strip()
    tree = _run_git(source_root, "rev-parse", "HEAD^{tree}").strip()
    tracked_status = _run_git(source_root, "status", "--porcelain=v1", "--untracked-files=no")
    untracked_raw = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "--others", "--exclude-standard", "-z"],
        check=True,
        stdout=subprocess.PIPE,
        timeout=60,
    ).stdout
    untracked = sorted(os.fsdecode(item) for item in untracked_raw.split(b"\0") if item)
    import_shadowing = [
        item
        for item in untracked
        if item.startswith("cosmos_framework/") and "__pycache__" not in Path(item).parts
    ]
    if commit != expected["commit"] or tree != expected["git_tree"]:
        raise N3QualificationError("N3 external source commit/tree differs from the pinned runtime")
    if tracked_status:
        raise N3QualificationError("N3 external source has tracked changes")
    if import_shadowing:
        raise N3QualificationError(
            f"N3 external source has untracked import-shadowing paths: {import_shadowing[:10]}"
        )
    inventory = _tracked_inventory(source_root)
    for key in ("tracked_file_count", "tracked_bytes", "tracked_aggregate_sha256"):
        if inventory[key] != expected[key]:
            raise N3QualificationError(f"N3 external tracked inventory mismatch for {key}")
    server_path = source_root / expected["server_relative_path"]
    if (
        not server_path.is_file()
        or server_path.stat().st_size != expected["server_bytes"]
        or sha256_file(server_path) != expected["server_sha256"]
    ):
        raise N3QualificationError("official N3 server file identity mismatch")
    ast_receipt = inspect_official_server_ast(server_path)
    return {
        "path": str(source_root),
        "commit": commit,
        "git_tree": tree,
        "tracked_status_clean_before_import": True,
        "untracked_path_count": len(untracked),
        "untracked_import_shadowing_path_count": 0,
        **inventory,
        "server_file": {
            "path": expected["server_relative_path"],
            "bytes": server_path.stat().st_size,
            "sha256": sha256_file(server_path),
            "ast_contract": ast_receipt,
        },
    }


def checkpoint_inventory(checkpoint_root: Path) -> dict[str, Any]:
    root = Path(checkpoint_root).resolve()
    if not root.is_dir():
        raise N3QualificationError(f"N3 checkpoint directory is missing: {root}")
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    )
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        total_bytes += path.stat().st_size
        aggregate.update(f"{digest}  {relative}\n".encode("utf-8"))
    return {
        "path": str(root),
        "payload_file_count": len(files),
        "payload_bytes": total_bytes,
        "payload_aggregate_sha256": aggregate.hexdigest(),
    }


def validate_checkpoint(checkpoint_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    observed = checkpoint_inventory(checkpoint_root)
    expected = contract["checkpoint"]
    for key in ("payload_file_count", "payload_bytes", "payload_aggregate_sha256"):
        if observed[key] != expected[key]:
            raise N3QualificationError(f"N3 checkpoint byte inventory mismatch for {key}")
    return {
        **observed,
        "repository": expected["repository"],
        "revision": expected["revision"],
        "full_payload_rehash_performed": True,
        "aggregate_definition": expected["aggregate_definition"],
    }


def _resolve_sibling(manifest_path: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise N3QualificationError(f"{label} must be a relative path")
    root = manifest_path.parent.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise N3QualificationError(f"{label} escapes the observation bundle")
    return path


def write_fixed_observation_bundle(
    destination: Path,
    observation: Mapping[str, Any],
    *,
    observation_id: str,
    source_capture: Mapping[str, Any],
) -> Path:
    """Package an exact official-client wire observation for this qualification.

    The physical recorder remains authoritative for original camera/state data;
    ``source_capture.capture_receipt`` must point back to that hash-checked
    evidence.  This helper only freezes the three arrays that the official N3
    server actually receives, avoiding an ad-hoc format in the cluster runner.
    """

    root = _prepare_directory(destination, "N3 fixed-observation bundle")
    if not isinstance(observation_id, str) or not observation_id:
        raise N3QualificationError("fixed-observation bundle needs a nonempty observation_id")
    if set(observation) != set(REQUIRED_WIRE_KEYS):
        raise N3QualificationError("fixed-observation bundle requires exactly three official wire arrays")
    frozen = {key: np.ascontiguousarray(np.asarray(observation[key])).copy() for key in REQUIRED_WIRE_KEYS}
    archive_keys = {key: f"wire_{index:02d}" for index, key in enumerate(REQUIRED_WIRE_KEYS)}
    archive_path = root / "observation.npz"
    temporary = root / f".observation.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **{archive_keys[key]: value for key, value in frozen.items()})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, archive_path)
        _fsync_directory(root)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": OBSERVATION_SCHEMA,
        "observation_id": observation_id,
        "payload": {
            "path": archive_path.name,
            "sha256": sha256_file(archive_path),
            "arrays": {
                key: {
                    "archive_key": archive_keys[key],
                    **{
                        field: _array_value_identity(frozen[key])[field]
                        for field in ("dtype", "shape", "value_sha256")
                    },
                }
                for key in REQUIRED_WIRE_KEYS
            },
        },
        "source_capture": copy.deepcopy(dict(source_capture)),
    }
    manifest_path = root / "n3_fixed_observation.json"
    _atomic_json(manifest_path, manifest)
    # Re-open from disk so the producer and consumer share exactly one gate.
    load_fixed_observation(manifest_path)
    return manifest_path


def load_fixed_observation(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != OBSERVATION_SCHEMA:
        raise N3QualificationError("unexpected fixed-observation schema")
    if not isinstance(manifest.get("observation_id"), str) or not manifest["observation_id"]:
        raise N3QualificationError("fixed observation lacks observation_id")
    payload = manifest.get("payload")
    if not isinstance(payload, Mapping):
        raise N3QualificationError("fixed observation lacks payload descriptor")
    archive_path = _resolve_sibling(path, payload.get("path"), "payload.path")
    if not archive_path.is_file() or sha256_file(archive_path) != payload.get("sha256"):
        raise N3QualificationError("fixed-observation NPZ path/hash mismatch")
    arrays = payload.get("arrays")
    if not isinstance(arrays, Mapping) or set(arrays) != set(REQUIRED_WIRE_KEYS):
        raise N3QualificationError("fixed observation must expose exactly the three official N3 wire arrays")
    observation: dict[str, Any] = {}
    with np.load(archive_path, allow_pickle=False) as archive:
        for wire_key in REQUIRED_WIRE_KEYS:
            descriptor = arrays[wire_key]
            if not isinstance(descriptor, Mapping):
                raise N3QualificationError(f"invalid array descriptor for {wire_key}")
            archive_key = descriptor.get("archive_key")
            if not isinstance(archive_key, str) or archive_key not in archive.files:
                raise N3QualificationError(f"missing archive key for {wire_key}")
            array = np.ascontiguousarray(archive[archive_key]).copy()
            identity = _array_value_identity(array)
            for identity_key in ("dtype", "shape", "value_sha256"):
                if descriptor.get(identity_key) != identity[identity_key]:
                    raise N3QualificationError(
                        f"fixed-observation array identity mismatch for {wire_key}.{identity_key}"
                    )
            observation[wire_key] = array
    for key in IMAGE_WIRE_KEYS:
        image = observation[key]
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise N3QualificationError(f"{key} must be one uint8 HxWx3 image")
    joint = observation["observation/joint_position"]
    gripper = observation["observation/gripper_position"]
    if joint.ndim not in {1, 2} or joint.shape[-1] != 7 or not np.isfinite(joint).all():
        raise N3QualificationError("joint_position must be finite [7] or [T,7]")
    if gripper.ndim > 2 or gripper.size == 0 or not np.isfinite(gripper).all():
        raise N3QualificationError("gripper_position must be a finite nonempty scalar/vector")
    capture = manifest.get("source_capture")
    historical = isinstance(capture, Mapping) and capture.get("provenance_kind") == "archived_policy_input"
    required_capture = ("capture_id", "capture_receipt") if historical else (
        "capture_id",
        "settled_reset_identity",
        "simulator_observation_id",
        "camera_frame_ids",
        "camera_capture_time_ns",
        "capture_receipt",
    )
    if not isinstance(capture, Mapping) or any(not capture.get(key) for key in required_capture):
        raise N3QualificationError("fixed observation lacks source-capture/reset/timing provenance")
    if historical and (capture.get("historical_timing_unavailable") is not True
                       or capture.get("physical_time_mapping_qualified") is not False):
        raise N3QualificationError("historical generation-only input must disclose unavailable physical timing")
    for field in (() if historical else ("camera_frame_ids", "camera_capture_time_ns")):
        values = capture[field]
        if not isinstance(values, Mapping) or set(values) != set(IMAGE_WIRE_KEYS):
            raise N3QualificationError(f"source_capture.{field} must bind the packed official image key")
    if not historical and any(type(value) is not int or value < 0 for value in capture["camera_capture_time_ns"].values()):
        raise N3QualificationError("camera capture times must be nonnegative integer nanoseconds")
    receipt = capture["capture_receipt"]
    if not isinstance(receipt, Mapping):
        raise N3QualificationError("source capture receipt descriptor is invalid")
    receipt_path_value = receipt.get("path")
    if not isinstance(receipt_path_value, str) or not receipt_path_value:
        raise N3QualificationError("source capture receipt path is missing")
    receipt_path = Path(receipt_path_value)
    if not receipt_path.is_absolute():
        receipt_path = _resolve_sibling(path, receipt_path_value, "source_capture.capture_receipt.path")
    else:
        receipt_path = receipt_path.resolve()
    if not receipt_path.is_file() or sha256_file(receipt_path) != receipt.get("sha256"):
        raise N3QualificationError("source capture receipt path/hash mismatch")
    observation_identity = _sha256_bytes(
        _canonical_bytes({key: _array_value_identity(value) for key, value in observation.items()})
    )
    manifest_receipt = {
        "path": str(path),
        "sha256": sha256_file(path),
        "observation_id": manifest["observation_id"],
        "payload_path": str(archive_path),
        "payload_sha256": sha256_file(archive_path),
        "wire_observation_sha256": observation_identity,
        "source_capture": copy.deepcopy(dict(capture)),
    }
    return observation, manifest_receipt


def _copy_wire_observation(observation: Mapping[str, Any], prompt: str) -> dict[str, Any]:
    output = {key: np.ascontiguousarray(observation[key]).copy() for key in REQUIRED_WIRE_KEYS}
    output["prompt"] = prompt
    return output


def _wire_arrays_sha256(observation: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        _canonical_bytes({key: _array_value_identity(np.asarray(observation[key])) for key in REQUIRED_WIRE_KEYS})
    )


def _collect_gpu_evidence(torch: Any, required_name: str) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise N3QualificationError("N3 qualification requires a visible CUDA GPU")
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "visible_index": index,
                "name": str(properties.name),
                "total_memory_bytes": int(properties.total_memory),
                "major": int(properties.major),
                "minor": int(properties.minor),
                "uuid": str(getattr(properties, "uuid", "")) or None,
            }
        )
    current = int(torch.cuda.current_device())
    if required_name not in devices[current]["name"]:
        raise N3QualificationError(
            f"N3 qualification device is {devices[current]['name']!r}, expected name containing {required_name!r}"
        )
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    if smi.returncode:
        raise N3QualificationError(
            f"nvidia-smi identity query failed: {smi.stderr.decode('utf-8', 'replace').strip()}"
        )
    smi_rows = [line.strip() for line in smi.stdout.decode("utf-8", "replace").splitlines() if line.strip()]
    if not smi_rows:
        raise N3QualificationError("nvidia-smi returned no visible GPU identities")
    return {
        "cuda_available": True,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "visible_device_count": len(devices),
        "current_device": current,
        "devices": devices,
        "nvidia_smi_rows": smi_rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "required_current_device_name_substring": required_name,
        "requirement_satisfied": True,
    }


def _synchronize(torch: Any) -> None:
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def _memory_snapshot(torch: Any) -> dict[str, int]:
    if torch is None or not torch.cuda.is_available():
        return {}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _validate_runtime_config(service: Any, checkpoint_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    cfg = service.cfg
    expected = contract["runtime"]
    observed = {
        "checkpoint_path": str(Path(cfg.checkpoint_path).resolve()),
        "domain_name": cfg.domain_name,
        "decode_video": bool(cfg.decode_video),
        "guidance": float(cfg.guidance),
        "denoising_steps": int(cfg.num_steps),
        "shift": float(cfg.shift),
        "conditioning_fps": float(cfg.conditioning_fps),
        "resolution_setting": str(cfg.resolution),
        "action_chunk_size": int(cfg.action_chunk_size),
        "action_dimensions": int(cfg.action_dim),
        "action_space": cfg.action_space,
        "use_state": bool(cfg.use_state),
        "history_length": int(cfg.history_length),
        "deterministic_seed": bool(cfg.deterministic_seed),
    }
    wanted = {
        "checkpoint_path": str(Path(checkpoint_root).resolve()),
        "domain_name": expected["domain_name"],
        "decode_video": False,
        "guidance": expected["guidance"],
        "denoising_steps": expected["denoising_steps"],
        "shift": expected["shift"],
        "conditioning_fps": expected["conditioning_fps"],
        "resolution_setting": expected["resolution_setting"],
        "action_chunk_size": expected["returned_actions"],
        "action_dimensions": expected["action_dimensions"],
        "action_space": expected["action_space"],
        "use_state": expected["use_state"],
        "history_length": expected["history_length"],
        "deterministic_seed": True,
    }
    changed = [key for key, value in wanted.items() if observed.get(key) != value]
    if changed:
        raise N3QualificationError(f"constructed N3 runtime mismatch for {', '.join(changed)}")
    return observed


def _import_official_server(source_root: Path, *, pycache_root: Path) -> Any:
    root = Path(source_root).resolve()
    for name, module in list(sys.modules.items()):
        if name == "cosmos_framework" or name.startswith("cosmos_framework."):
            module_file = getattr(module, "__file__", None)
            if module_file is not None:
                loaded = Path(module_file).resolve()
                if root != loaded and root not in loaded.parents:
                    raise N3QualificationError("cosmos_framework was already imported from another checkout")
    Path(pycache_root).mkdir(parents=True, exist_ok=True)
    sys.pycache_prefix = str(Path(pycache_root).resolve())
    sys.path.insert(0, str(root))
    module = importlib.import_module("cosmos_framework.scripts.action_policy_server_robolab")
    expected_path = root / "cosmos_framework/scripts/action_policy_server_robolab.py"
    if Path(module.__file__).resolve() != expected_path.resolve():
        raise N3QualificationError("official N3 server imported from an unexpected path")
    return module


def _build_official_service(
    server: Any,
    *,
    checkpoint_root: Path,
    model_output_dir: Path,
    effective_seed: int,
    contract: Mapping[str, Any],
) -> Any:
    runtime = contract["runtime"]

    class MeasurementOnlyService(server.RobolabPolicyService):
        _wmf_effective_seed: int | None = None

        def _build_setup_args(self, args: Any) -> Any:
            setup_args = super()._build_setup_args(args)
            # This is the historical, separately gated no-download compatibility
            # change. It does not alter model transforms, denoising, actions or VAE.
            return setup_args.model_copy(update={"guardrails": False})

        def _next_seed(self) -> int:
            if self._wmf_effective_seed is None:
                raise N3QualificationError("official generation attempted without a request-bound seed")
            return int(self._wmf_effective_seed)

    args = server.RobolabServerArgs(
        checkpoint_path=str(Path(checkpoint_root).resolve()),
        hf_revision=contract["checkpoint"]["revision"],
        allow_dcp_checkpoint=False,
        output_dir=Path(model_output_dir).resolve(),
        domain_name=runtime["domain_name"],
        decode_video=False,
        seed=effective_seed,
        deterministic_seed=True,
        guidance=runtime["guidance"],
        num_steps=runtime["denoising_steps"],
        shift=runtime["shift"],
        resolution=runtime["resolution_setting"],
        conditioning_fps=runtime["conditioning_fps"],
        action_chunk_size=runtime["returned_actions"],
        action_dim=runtime["action_dimensions"],
        action_space=runtime["action_space"],
        use_state=runtime["use_state"],
        history_length=runtime["history_length"],
    )
    service = MeasurementOnlyService(args)
    service._wmf_effective_seed = None
    return service


@dataclass
class _CaptureSession:
    writer: _ArtifactWriter
    generator_calls: int = 0
    decode_calls: int = 0
    model_input_artifact: dict[str, Any] | None = None
    generated_action_artifact: dict[str, Any] | None = None
    latent_artifact: dict[str, Any] | None = None
    latent_identity: dict[str, Any] | None = None
    latent_object: Any = None


class _CapturingModelProxy:
    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.active: _CaptureSession | None = None
        self.offline_decode_allowed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    def generate_samples_from_batch(self, data_batch: Any, *args: Any, **kwargs: Any) -> Any:
        capture = self.active
        if capture is None:
            raise N3QualificationError("joint generation escaped the active measurement request")
        capture.generator_calls += 1
        if capture.generator_calls != 1:
            raise N3QualificationError("one N3 request invoked joint generation more than once")
        capture.model_input_artifact = capture.writer.write("exact_transformed_model_input", data_batch)
        samples = self._delegate.generate_samples_from_batch(data_batch, *args, **kwargs)
        if not isinstance(samples, Mapping) or "action" not in samples or "vision" not in samples:
            raise N3QualificationError("official joint generation lacks action or vision output")
        action = samples["action"][0]
        latent = samples["vision"][0]
        capture.generated_action_artifact = capture.writer.write("raw_generated_action", action)
        capture.latent_artifact = capture.writer.write("retained_vision_latent", latent)
        capture.latent_identity = value_identity(latent)
        capture.latent_object = latent
        return samples

    def decode(self, latent: Any) -> Any:
        capture = self.active
        if capture is None or not self.offline_decode_allowed:
            raise N3QualificationError("official infer attempted latent rendering in the no-decode path")
        if latent is not capture.latent_object:
            raise N3QualificationError("offline decode did not receive the exact retained latent object")
        capture.decode_calls += 1
        if capture.decode_calls != 1:
            raise N3QualificationError("one N3 request decoded its latent more than once")
        return self._delegate.decode(latent)


def _official_uint8_video(decoded: Any) -> Any:
    """Apply only the official server's post-VAE uint8 rendering transform."""

    return (
        ((decoded[0].clamp(-1.0, 1.0) + 1.0) * 127.5)
        .to(importlib.import_module("torch").uint8)
        .permute(1, 2, 3, 0)
        .detach()
        .cpu()
        .numpy()
    )


class N3InstrumentedRuntime:
    """Measurement wrapper reusable by the qualification job and block runner."""

    def __init__(
        self,
        service: Any,
        *,
        contract: Mapping[str, Any],
        checkpoint_root: Path,
        torch_module: Any | None,
        render_decoded: Callable[[Any], Any] = _official_uint8_video,
    ) -> None:
        self.service = service
        self.contract = contract
        self.torch = torch_module
        self.render_decoded = render_decoded
        self.config_receipt = _validate_runtime_config(service, checkpoint_root, contract)
        self._model_proxy = _CapturingModelProxy(service.model)
        service.model = self._model_proxy
        self._lock = threading.Lock()
        self.server_context_id = f"n3:{socket.gethostname()}:{os.getpid()}:{id(service)}"
        self.request_counter = 0

    def reset_request_context(self, effective_seed: int, request_id: str) -> dict[str, Any]:
        if self._model_proxy.active is not None or self._model_proxy.offline_decode_allowed:
            raise N3QualificationError("previous N3 capture context was not cleared")
        if type(effective_seed) is not int or not 0 <= effective_seed < 2**31:
            raise N3QualificationError("effective N3 seed is outside the official signed-31-bit range")
        self.service._wmf_effective_seed = effective_seed
        if hasattr(self.service, "_rng"):
            self.service._rng = np.random.default_rng(effective_seed)
        if self.torch is not None:
            self.torch.manual_seed(effective_seed)
            if self.torch.cuda.is_available():
                self.torch.cuda.manual_seed_all(effective_seed)
        stateful_names = sorted(
            name
            for name in vars(self.service)
            if any(marker in name.lower() for marker in ("cache", "history", "buffer"))
            and name not in {"cfg"}
        )
        if stateful_names:
            raise N3QualificationError(
                f"official N3 service unexpectedly exposes temporal state fields: {stateful_names}"
            )
        return {
            "passed": True,
            "reset_scope": CONTEXT_RESET_SCOPE,
            "server_context_id": self.server_context_id,
            "request_id": request_id,
            "effective_seed": effective_seed,
            "wrapper_capture_cleared_before_request": True,
            "official_request_rng_reinitialized": True,
            "torch_cpu_rng_reinitialized": self.torch is not None,
            "torch_cuda_rngs_reinitialized": bool(self.torch is not None and self.torch.cuda.is_available()),
            "request_bound_official_next_seed": True,
            "official_service_temporal_state_fields": stateful_names,
            "history_length": int(self.service.cfg.history_length),
            "single_saved_observation_reloaded": True,
        }

    def run_request(
        self,
        *,
        observation: Mapping[str, Any],
        prompt: str,
        effective_seed: int,
        request_id: str,
        request_index: int,
        decode: bool,
        request_root: Path,
    ) -> dict[str, Any]:
        if request_index != self.request_counter:
            raise N3QualificationError("N3 request order is not contiguous")
        with self._lock:
            writer = _ArtifactWriter(request_root)
            reset = self.reset_request_context(effective_seed, request_id)
            _atomic_json(request_root / "context_reset.json", reset)
            reset_descriptor = {
                "path": str(request_root / "context_reset.json"),
                "sha256": sha256_file(request_root / "context_reset.json"),
            }
            wire = _copy_wire_observation(observation, prompt)
            wire_hash_before = _wire_arrays_sha256(wire)
            wire_artifact = writer.write("exact_wire_request", wire)
            capture = _CaptureSession(writer=writer)
            self._model_proxy.active = capture
            self._model_proxy.offline_decode_allowed = False
            if self.torch is not None and self.torch.cuda.is_available():
                self.torch.cuda.reset_peak_memory_stats()
            _synchronize(self.torch)
            generation_start_wall_ns = time.time_ns()
            generation_start_ns = time.monotonic_ns()
            memory_before = _memory_snapshot(self.torch)
            try:
                response = self.service.infer(wire)
                _synchronize(self.torch)
                action_captured_ns = time.monotonic_ns()
                memory_after_generation = _memory_snapshot(self.torch)
                if not isinstance(response, Mapping) or set(response) != {"action"}:
                    raise N3QualificationError("official no-decode response must contain only action")
                action = np.ascontiguousarray(np.asarray(response["action"]))
                expected_shape = (
                    self.contract["runtime"]["returned_actions"],
                    self.contract["runtime"]["action_dimensions"],
                )
                if action.shape != expected_shape or not np.isfinite(action).all():
                    raise N3QualificationError(
                        f"official N3 action must be finite {expected_shape}, got {action.shape}"
                    )
                action_artifact = writer.write("returned_action", action)
                action_identity = _array_value_identity(action)
                if capture.generator_calls != 1 or capture.latent_object is None:
                    raise N3QualificationError("official request did not retain exactly one joint generation")
                if capture.decode_calls != 0:
                    raise N3QualificationError("official infer rendered despite decode_video=False")
                latent_before_decode = value_identity(capture.latent_object)
                if latent_before_decode != capture.latent_identity:
                    raise N3QualificationError("retained latent changed before optional offline decode")
                decoded_artifact = None
                decoded_identity = None
                decode_start_ns = None
                decode_end_ns = None
                if decode:
                    decode_start_ns = time.monotonic_ns()
                    if decode_start_ns < action_captured_ns:
                        raise N3QualificationError("offline decode began before action capture")
                    self._model_proxy.offline_decode_allowed = True
                    decoded_raw = self._model_proxy.decode(capture.latent_object)
                    _synchronize(self.torch)
                    decoded = self.render_decoded(decoded_raw)
                    decode_end_ns = time.monotonic_ns()
                    decoded_artifact = writer.write("offline_decoded_future", decoded)
                    decoded_identity = value_identity(decoded)
                    if value_identity(capture.latent_object) != latent_before_decode:
                        raise N3QualificationError("offline decode mutated the retained latent")
                if capture.decode_calls != (1 if decode else 0):
                    raise N3QualificationError("N3 decode call count differs from the request mode")
                wire_hash_after = _wire_arrays_sha256(wire)
                if wire_hash_after != wire_hash_before:
                    raise N3QualificationError("official inference mutated the saved wire observation")
                _synchronize(self.torch)
                end_ns = time.monotonic_ns()
                record = {
                    "request_index": request_index,
                    "request_id": request_id,
                    "prompt": prompt,
                    "decode_requested": decode,
                    "decode_mode": "offline_exact_retained_latent_after_action_capture" if decode else "none",
                    "effective_seed": effective_seed,
                    "reset_receipt": reset_descriptor,
                    "wire_observation_sha256": wire_hash_before,
                    "wire_request_artifact": wire_artifact,
                    "exact_transformed_model_input_artifact": capture.model_input_artifact,
                    "raw_generated_action_artifact": capture.generated_action_artifact,
                    "returned_action_artifact": action_artifact,
                    "returned_action_identity": action_identity,
                    "retained_latent_artifact": capture.latent_artifact,
                    "retained_latent_identity": latent_before_decode,
                    "decoded_future_artifact": decoded_artifact,
                    "decoded_future_identity": decoded_identity,
                    "official_joint_generation_calls": capture.generator_calls,
                    "official_infer_decode_calls": 0,
                    "offline_decode_calls": capture.decode_calls,
                    "timing": {
                        "generation_start_wall_time_ns": generation_start_wall_ns,
                        "generation_start_monotonic_ns": generation_start_ns,
                        "action_captured_monotonic_ns": action_captured_ns,
                        "offline_decode_start_monotonic_ns": decode_start_ns,
                        "offline_decode_end_monotonic_ns": decode_end_ns,
                        "request_end_monotonic_ns": end_ns,
                        "generation_through_action_capture_seconds": (action_captured_ns - generation_start_ns) / 1e9,
                        "offline_decode_seconds": (
                            None if decode_start_ns is None else (decode_end_ns - decode_start_ns) / 1e9
                        ),
                    },
                    "gpu_memory": {
                        "before_generation": memory_before,
                        "after_generation": memory_after_generation,
                        "after_request": _memory_snapshot(self.torch),
                    },
                    "action_captured_before_offline_decode": (
                        not decode or (decode_start_ns is not None and action_captured_ns <= decode_start_ns)
                    ),
                }
                _atomic_json(request_root / "request_receipt.json", record)
                self.request_counter += 1
                return record
            finally:
                self._model_proxy.offline_decode_allowed = False
                self._model_proxy.active = None
                self.service._wmf_effective_seed = None


def evaluate_protocol(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != 6 or [row.get("request_index") for row in records] != list(range(6)):
        raise N3QualificationError("N3 fixed-observation protocol did not produce exactly six ordered records")
    pairs = []
    for no_decode_index, decode_index in ((0, 3), (1, 4), (2, 5)):
        first, second = records[no_decode_index], records[decode_index]
        checks = {
            "wire_observation_equal": first["wire_observation_sha256"] == second["wire_observation_sha256"],
            "model_input_equal": (
                first["exact_transformed_model_input_artifact"]["logical_sha256"]
                == second["exact_transformed_model_input_artifact"]["logical_sha256"]
            ),
            "action_equal": (
                first["returned_action_identity"]["value_sha256"]
                == second["returned_action_identity"]["value_sha256"]
            ),
            "latent_equal": (
                first["retained_latent_identity"]["value_sha256"]
                == second["retained_latent_identity"]["value_sha256"]
            ),
            "effective_seed_equal": first["effective_seed"] == second["effective_seed"],
            "decoded_only_in_decode_request": (
                first["decoded_future_artifact"] is None and second["decoded_future_artifact"] is not None
            ),
        }
        if not all(checks.values()):
            raise N3QualificationError(
                f"decode/no-decode pair {no_decode_index}/{decode_index} changed generation: {checks}"
            )
        pairs.append({"no_decode_index": no_decode_index, "decode_index": decode_index, "checks": checks})
    repeats = []
    for first_index, second_index in ((0, 1), (3, 4)):
        first, second = records[first_index], records[second_index]
        checks = {
            "wire_observation_equal": first["wire_observation_sha256"] == second["wire_observation_sha256"],
            "model_input_equal": (
                first["exact_transformed_model_input_artifact"]["logical_sha256"]
                == second["exact_transformed_model_input_artifact"]["logical_sha256"]
            ),
            "action_equal": (
                first["returned_action_identity"]["value_sha256"]
                == second["returned_action_identity"]["value_sha256"]
            ),
            "latent_equal": (
                first["retained_latent_identity"]["value_sha256"]
                == second["retained_latent_identity"]["value_sha256"]
            ),
        }
        if second["decode_requested"]:
            checks["decoded_future_equal"] = (
                first["decoded_future_identity"]["value_sha256"]
                == second["decoded_future_identity"]["value_sha256"]
            )
        if not all(checks.values()):
            raise N3QualificationError(f"fixed-input LEFT repeat was nondeterministic: {checks}")
        repeats.append({"first_index": first_index, "second_index": second_index, "checks": checks})
    left, right = records[0], records[2]
    prompt_sensitivity = {
        "same_saved_observation": left["wire_observation_sha256"] == right["wire_observation_sha256"],
        "different_exact_prompt": left["prompt"] != right["prompt"],
        "different_transformed_model_input": (
            left["exact_transformed_model_input_artifact"]["logical_sha256"]
            != right["exact_transformed_model_input_artifact"]["logical_sha256"]
        ),
        "action_differs": (
            left["returned_action_identity"]["value_sha256"]
            != right["returned_action_identity"]["value_sha256"]
        ),
        "latent_differs": (
            left["retained_latent_identity"]["value_sha256"]
            != right["retained_latent_identity"]["value_sha256"]
        ),
    }
    if not (
        prompt_sensitivity["same_saved_observation"]
        and prompt_sensitivity["different_exact_prompt"]
        and prompt_sensitivity["different_transformed_model_input"]
        and (prompt_sensitivity["action_differs"] or prompt_sensitivity["latent_differs"])
    ):
        raise N3QualificationError(f"N3 prompt-sensitivity gate failed: {prompt_sensitivity}")
    return {
        "decode_no_decode_pairs": pairs,
        "fixed_input_repeat_checks": repeats,
        "prompt_sensitivity": prompt_sensitivity,
        "all_decode_pairs_action_and_latent_exact": True,
        "fixed_input_repeat_deterministic": True,
    }


def execute_six_request_protocol(
    runtime: N3InstrumentedRuntime,
    *,
    observation: Mapping[str, Any],
    effective_seed: int,
    requests_root: Path,
    contract: Mapping[str, Any],
    journal: _EventJournal | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates = contract["candidate_effective_seeds"]
    if effective_seed not in candidates:
        raise N3QualificationError("N3 qualification seed is outside the frozen candidate seed set")
    requests_root = Path(requests_root)
    requests_root.mkdir(parents=True, exist_ok=False)
    records = []
    for row in contract["fixed_observation_protocol"]:
        if journal is not None:
            journal.append(
                "generation_request_started",
                {
                    "request_index": row["request_index"],
                    "request_id": row["request_id"],
                    "decode": row["decode"],
                    "effective_seed": effective_seed,
                },
            )
        record = runtime.run_request(
            observation=observation,
            prompt=contract["prompts"][row["prompt_key"]],
            effective_seed=effective_seed,
            request_id=row["request_id"],
            request_index=row["request_index"],
            decode=row["decode"],
            request_root=requests_root / f"{row['request_index']:02d}_{row['request_id']}",
        )
        records.append(record)
        if journal is not None:
            journal.append(
                "generation_request_completed",
                {
                    "request_index": row["request_index"],
                    "request_id": row["request_id"],
                    "action_sha256": record["returned_action_identity"]["value_sha256"],
                    "latent_sha256": record["retained_latent_identity"]["value_sha256"],
                },
            )
    evaluation = evaluate_protocol(records)
    return records, evaluation


def _prepare_directory(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise FileExistsError(f"refusing to reuse nonempty {label}: {resolved}")
    else:
        resolved.mkdir(parents=True)
    return resolved


def run_qualification(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = _prepare_directory(args.output_dir, "N3 raw attempt directory")
    publish_root = _prepare_directory(args.publish_dir or raw_root / "publish", "N3 publish directory")
    journal = _EventJournal(raw_root / "events.partial.jsonl")
    started_wall_ns = time.time_ns()
    started_monotonic_ns = time.monotonic_ns()
    process = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.version,
        "started_wall_time_ns": started_wall_ns,
        "started_monotonic_ns": started_monotonic_ns,
    }
    journal.append("qualification_started", {"process": process, "effective_seed": args.effective_seed})
    completed_requests = 0
    try:
        contract = load_contract()
        if args.effective_seed not in contract["candidate_effective_seeds"]:
            raise N3QualificationError("effective seed is not one of the 29 frozen N3 block seeds")
        journal.append("source_validation_started", {"source_root": str(Path(args.source_root).resolve())})
        source = validate_external_source(args.source_root, contract)
        journal.append("source_validation_completed", source)
        journal.append("checkpoint_full_rehash_started", {"checkpoint_root": str(Path(args.checkpoint_root).resolve())})
        checkpoint = validate_checkpoint(args.checkpoint_root, contract)
        journal.append("checkpoint_full_rehash_completed", checkpoint)
        observation, observation_receipt = load_fixed_observation(args.observation_manifest)
        journal.append("fixed_observation_validated", observation_receipt)
        torch = importlib.import_module("torch")
        gpu = _collect_gpu_evidence(torch, contract["runtime"]["required_gpu_name_substring"])
        journal.append("gpu_validated", gpu)
        server = _import_official_server(
            args.source_root,
            pycache_root=raw_root / "python_bytecode_cache",
        )
        model_output_dir = raw_root / "model_runtime"
        model_output_dir.mkdir()
        load_started_ns = time.monotonic_ns()
        service = _build_official_service(
            server,
            checkpoint_root=args.checkpoint_root,
            model_output_dir=model_output_dir,
            effective_seed=args.effective_seed,
            contract=contract,
        )
        _synchronize(torch)
        model_loaded_ns = time.monotonic_ns()
        runtime = N3InstrumentedRuntime(
            service,
            contract=contract,
            checkpoint_root=args.checkpoint_root,
            torch_module=torch,
        )
        journal.append(
            "official_model_loaded",
            {
                "load_seconds": (model_loaded_ns - load_started_ns) / 1e9,
                "config": runtime.config_receipt,
                "gpu_memory": _memory_snapshot(torch),
            },
        )
        records, evaluation = execute_six_request_protocol(
            runtime,
            observation=observation,
            effective_seed=args.effective_seed,
            requests_root=raw_root / "requests",
            contract=contract,
            journal=journal,
        )
        completed_requests = len(records)
        journal.append("protocol_validated", evaluation)
        ended_monotonic_ns = time.monotonic_ns()
        receipt = {
            "schema_version": RECEIPT_SCHEMA,
            "study_id": contract["study_id"],
            "model_config": "N3",
            "status": "passed",
            "qualified": True,
            "generation_request_count": 6,
            "robot_episode_count": 0,
            "effective_seed": args.effective_seed,
            "effective_seed_semantics": {
                "mode": "request_bound_override_of_official_RobolabPolicyService._next_seed",
                "same_effective_seed_for_all_six_requests": True,
                "fixed_input_repeat_resets_rng_and_uses_same_seed": True,
                "candidate_is_matched_within_one_future_layout_block": True,
                "candidate_labels_are_not_independent_noise_draws": True,
                "accepted_signed_31_bit_integer": 0 <= args.effective_seed < 2**31,
            },
            "decode_semantics": {
                "official_decode_video": False,
                "joint_video_prediction_executed_per_request": True,
                "no_decode_skips_only_latent_rendering": True,
                "decode_requests_render_exact_retained_latent_after_action_capture": True,
                "decode_toggle_changes_action_or_latent": False,
                "official_interface_toggle": "used_false_for_all_generation_then_measurement_only_offline_decode",
            },
            "source": source,
            "checkpoint": checkpoint,
            "runtime_config": runtime.config_receipt,
            "wrapper": {
                "path": str(MODULE_PATH),
                "sha256": sha256_file(MODULE_PATH),
                "contract_path": str(CONTRACT_PATH),
                "contract_sha256": sha256_file(CONTRACT_PATH),
                "guardrail_compatibility_change": "setup_args.guardrails_false_only",
            },
            "fixed_observation": observation_receipt,
            "gpu": gpu,
            "process": process,
            "timing": {
                "model_load_seconds": (model_loaded_ns - load_started_ns) / 1e9,
                "qualification_wall_seconds": (ended_monotonic_ns - started_monotonic_ns) / 1e9,
                "ended_wall_time_ns": time.time_ns(),
            },
            "requests": records,
            "evaluation": evaluation,
            "raw_root": str(raw_root),
            "journal": {
                "path": str(journal.path),
                "event_count_before_completion": journal.sequence,
                "tail_sha256_before_completion": journal.tail_sha256,
            },
            "claim_boundary": contract["claim_boundary"],
            "physical_time_mapping_qualified": False,
            "behavioral_policy_skill_evaluated": False,
        }
        journal.append(
            "qualification_completed",
            {
                "status": "passed",
                "generation_request_count": 6,
                "robot_episode_count": 0,
            },
        )
        receipt["journal"]["event_count"] = journal.sequence
        receipt["journal"]["tail_sha256"] = journal.tail_sha256
        raw_receipt_path = raw_root / "n3_qualification.json"
        _atomic_json(raw_receipt_path, receipt)
        compact = {
            "schema_version": RECEIPT_SCHEMA,
            "study_id": contract["study_id"],
            "model_config": "N3",
            "status": "passed",
            "qualified": True,
            "generation_request_count": 6,
            "robot_episode_count": 0,
            "effective_seed": args.effective_seed,
            "source_commit": source["commit"],
            "source_tree": source["git_tree"],
            "server_sha256": source["server_file"]["sha256"],
            "checkpoint_revision": checkpoint["revision"],
            "checkpoint_aggregate_sha256": checkpoint["payload_aggregate_sha256"],
            "fixed_observation_sha256": observation_receipt["wire_observation_sha256"],
            "decode_no_decode_pairs": evaluation["decode_no_decode_pairs"],
            "fixed_input_repeat_checks": evaluation["fixed_input_repeat_checks"],
            "prompt_sensitivity": evaluation["prompt_sensitivity"],
            "raw_receipt": {
                "path": str(raw_receipt_path),
                "bytes": raw_receipt_path.stat().st_size,
                "sha256": sha256_file(raw_receipt_path),
            },
            "claim_boundary": contract["claim_boundary"],
        }
        _atomic_json(publish_root / "n3_qualification.json", compact)
        return compact
    except BaseException as exc:
        if "runtime" in locals():
            completed_requests = max(completed_requests, int(runtime.request_counter))
        ended_ns = time.monotonic_ns()
        failure = {
            "schema_version": FAILURE_SCHEMA,
            "study_id": "WMF-ABLATION-001",
            "model_config": "N3",
            "status": "failed",
            "qualified": False,
            "generation_requests_completed_before_failure": completed_requests,
            "robot_episode_count": 0,
            "effective_seed": args.effective_seed,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "process": process,
            "raw_root": str(raw_root),
            "wall_seconds_before_failure": (ended_ns - started_monotonic_ns) / 1e9,
            "live_model_process_continues": False,
            "claim_boundary": "Failed qualification evidence only; not policy skill or a robot episode.",
        }
        with contextlib.suppress(Exception):
            journal.append("qualification_failed", failure)
            failure["journal"] = {
                "path": str(journal.path),
                "event_count": journal.sequence,
                "tail_sha256": journal.tail_sha256,
            }
        with contextlib.suppress(Exception):
            _atomic_json(raw_root / "n3_qualification_failure.json", failure)
        with contextlib.suppress(Exception):
            _atomic_json(publish_root / "n3_qualification_failure.json", failure)
        raise


def build_parser() -> argparse.ArgumentParser:
    contract = load_contract()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, default=Path(contract["checkpoint"]["default_path"]))
    parser.add_argument("--observation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--publish-dir", type=Path)
    parser.add_argument("--effective-seed", type=int, default=2026091000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        compact = run_qualification(args)
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "model_config": "N3",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    print(json.dumps(compact, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
