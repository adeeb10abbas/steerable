#!/usr/bin/env python3
"""Compile immutable WMF development recordings into downstream evidence.

This is a CPU-only, source-free adapter.  It does not import either model or a
simulator and it never executes an action.  It authenticates the retained
aggregate -> cell -> recording completion -> journal -> transported official
request chain, then emits:

* one timing-request inventory per model;
* one signed action manifest and recording receipt per valid cell; and
* one signed request-provenance inventory per model.

Formal mode requires all 32 selected development cells.  Diagnostic mode may
inspect an incomplete D1 cohort, but its timing inventories deliberately use a
non-release schema that ``qualify_forecast_timing.py`` rejects.
"""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pickle
import pickletools
import re
import shutil
import struct
import tempfile
from typing import Any, Mapping, Sequence
import zipfile


STUDY_ID = "WMF-ABLATION-001"
INPUT_SCHEMA = "wmf-development-evidence-compiler-input-v1"
COMPILER_SCHEMA = "wmf-development-evidence-compiler-receipt-v1"
PROVENANCE_SCHEMA = "wmf-development-request-provenance-v1"
TIMING_INVENTORY_SCHEMA = "wmf-development-timing-request-inventory-v1"
DIAGNOSTIC_TIMING_INVENTORY_SCHEMA = (
    "wmf-development-timing-request-inventory-diagnostic-v1"
)
ACTION_MANIFEST_SCHEMA = "wmf-forecast-action-manifest-v1"
RECORDING_RECEIPT_SCHEMA = "wmf-forecast-recording-receipt-v1"
FREEZE_FRAGMENT_SCHEMA = "wmf-development-freeze-cell-evidence-fragment-v1"
PLANNED_CELLS_SHA256 = (
    "7d06120a56419877d1acdfdc498c6dc054bf6860ce6bda5b2a25f55ae3b4166e"
)
PRIMARY_CAMERA_CHOICES = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
MODELS = ("N3", "D1")
LAYOUT_IDS = ("D01", "D02", "D03", "D04")
CONDITIONS = ("original_left", "original_right", "reflected_left", "reflected_right")
DEVELOPMENT_CONDITION_ORDERS: dict[str, tuple[str, ...]] = {
    "D01": ("original-left", "reflected-right", "original-right", "reflected-left"),
    "D02": ("reflected-right", "original-left", "reflected-left", "original-right"),
    "D03": ("reflected-right", "original-right", "original-left", "reflected-left"),
    "D04": ("original-left", "reflected-left", "reflected-right", "original-right"),
}
DEVELOPMENT_ENVIRONMENT_SEEDS = {
    "D01": 2026091101,
    "D02": 2026091102,
    "D03": 2026091103,
    "D04": 2026091104,
}
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

MODEL_PINS: dict[str, dict[str, str]] = {
    "N3": {
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "cosmos_commit": "411d25b2e35bc441126f48c44a4b93e1c0564274",
        "checkpoint_revision": "6706d7680581c255ff61e0f3bb49d90eac55c79e",
        "checkpoint_aggregate_sha256": (
            "55895125805b12635ece5dd2e88453a1aca6a684bac556f114337ae448ffdd83"
        ),
    },
    "D1": {
        "robolab_commit": "0aef241fb088ca21bb4ebd24448940ed56620d17",
        "dreamzero_commit": "ab790c198fbce33503358efbbd4187ce9a89adf3",
        "dreamzero_tree": "6b7ba27f1af81e963a6507f1204c05c65a94098c",
        "dreamzero_aggregate_sha256": (
            "a39e1ef8b7d8668caf15914637446fe58acb512e6672413d3af8b4b6442eca7f"
        ),
        "checkpoint_revision": "96ad344138c66e82536422432ad742f015784942",
        "checkpoint_aggregate_sha256": (
            "b4af0ac93474c3295c1ba841a34a8f2f91a5c3ec3c6aac1431b97689a6618c56"
        ),
        "tokenizer_revision": "66cb9e7e85526fe440a945569e42c72fb6cbc0ad",
        "tokenizer_aggregate_sha256": (
            "00f9b974f8f0b33c5e284849a0507c308893d01d915377ba88c2f7084ea92434"
        ),
    },
}

D1_MAPPING_CONTENT_HASH_DEFINITION = (
    "SHA-256 of canonical JSON over sorted keys, value kinds, shapes, dtypes, "
    "and exact tensor/array data hashes; artifact paths are excluded"
)
D1_RESET_FIELDS_TO_NONE = (
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
    "clip_feas",
    "ys",
    "language",
)
D1_CACHE_FIELDS = (
    "kv_cache1",
    "kv_cache_neg",
    "crossattn_cache",
    "crossattn_cache_neg",
)
D1_RESET_SCHEMA = "wmf-d1-two-rank-reset-receipt-v1"
D1_EPISODE_SCHEMA = "wmf-d1-episode-manifest-v1"
D1_READY_SCHEMA = "wmf-d1-behavioral-server-ready-v1"
D1_CLAIM_SCHEMA = "wmf-d1-behavioral-simulator-claim-v1"
D1_SERVER_CONTRACT_SCHEMA = "wmf-d1-instrumented-server-v1"
D1_ALLOWED_SIMULATOR_ROLES = {
    "wmf-forecast-0912-worker-00",
    "wmf-forecast-0912-worker-05",
    "wmf-forecast-0912-worker-06",
    "wmf-forecast-0912-worker-09",
}
CONTEXT_RESET_SCOPE = "full_episode_temporal_and_cache_context"

MODEL_LIMITS: dict[str, dict[str, Any]] = {
    "N3": {
        "aggregate_schema": "wmf-n3-behavioral-development-job-v1",
        "cell_schema": "wmf-n3-behavioral-development-cell-v1",
        "request_schema": "wmf-n3-behavioral-server-request-v1",
        "action_cap": 450,
        "observation_count": 451,
        "request_count": 15,
        "executed_prefix": 32,
        "returned_actions": 32,
        "decoded_frames": 33,
        "future_kinds": ["decoded"],
        "required_future_evidence": ["decoded"],
    },
    "D1": {
        "aggregate_schema": "wmf-d1-behavioral-development-simulator-job-v1",
        "cell_schema": "wmf-d1-behavioral-development-cell-v1",
        "request_schema": "wmf-d1-request-receipt-v1",
        "action_cap": 450,
        "observation_count": 451,
        "request_count": 57,
        "executed_prefix": 8,
        "returned_actions": 24,
        "decoded_frames": 9,
        # These are retained *per-request measurement* artifacts, not a
        # universal frame-to-action mapping.  Pinned DreamZero emits the
        # conditioning latent only on cache-boundary requests.  Consequently
        # a fresh offline VAE decode has nine RGB frames for request indices
        # divisible by four and five for the intervening two-latent chunks.
        # The timing sidecar, not this structural compiler, decides which
        # decoded outputs have source-proven timing semantics.
        "development_decode_shapes": {
            "full_conditioning_origin": {
                "latent": [1, 16, 3, 44, 80],
                "tensor": [1, 3, 9, 352, 640],
                "rgb": [9, 352, 640, 3],
            },
            "incremental_standalone": {
                "latent": [1, 16, 2, 44, 80],
                "tensor": [1, 3, 5, 352, 640],
                "rgb": [5, 352, 640, 3],
            },
        },
        "future_kinds": ["decoded", "latent"],
        "required_future_evidence": ["latent", "decoded"],
    },
}

FREEZE_PATH = Path(__file__).with_name("freeze_development_release.py")


def _load_freeze_module() -> Any:
    specification = importlib.util.spec_from_file_location(
        "wmf_development_compiler_freeze", FREEZE_PATH
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("development release validator cannot be loaded")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


freeze = _load_freeze_module()


class CompilerError(RuntimeError):
    """The supplied retained evidence cannot be compiled safely."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompilerError(message)


def canonical_bytes(value: Any, *, ensure_ascii: bool = False) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=ensure_ascii,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CompilerError("value is not finite canonical JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CompilerError("value is not finite JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise CompilerError(f"cannot hash file: {path}") from error
    return digest.hexdigest()


def sign_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    output = dict(value)
    output["payload_sha256"] = sha256_bytes(canonical_bytes(output))
    return output


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompilerError(f"{label} is not readable JSON: {path}") from error
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute lexical path without following any symlink."""

    candidate = Path(path)
    # Do not use abspath/resolve here: lexical ``..`` components after a
    # symlink are filesystem-significant, and normalizing them first would
    # erase the very ancestor that this gate must reject.
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_symlink_components(path: Path, label: str, *, stop: Path | None = None) -> None:
    """Reject a symlink in ``path`` or any inspected ancestor before resolve.

    ``Path.resolve`` erases the evidence that an ancestor was a symlink.  Walk
    the lexical path first, including ``stop`` when supplied.  Broken symlinks
    are rejected as well because ``is_symlink`` uses lstat semantics.
    """

    lexical = _lexical_absolute(path)
    boundary = None if stop is None else _lexical_absolute(stop)
    if boundary is not None:
        require(lexical.is_relative_to(boundary), f"{label} lexically escapes raw root: {lexical}")
    cursor = lexical
    while True:
        require(not cursor.is_symlink(), f"{label} path contains a symlink: {cursor}")
        if boundary is not None and cursor == boundary:
            return
        parent = cursor.parent
        if parent == cursor:
            require(boundary is None, f"{label} did not reach its lexical raw root")
            return
        cursor = parent


def _under(path: Path, root: Path, label: str) -> Path:
    lexical_root = _lexical_absolute(root)
    lexical = _lexical_absolute(path)
    # Validate both the configured root's ancestry and every target component
    # before either value is resolved.
    _reject_symlink_components(lexical_root, f"{label} raw root")
    _reject_symlink_components(lexical, label, stop=lexical_root)
    root_resolved = lexical_root.resolve()
    resolved = lexical.resolve()
    require(resolved.is_relative_to(root_resolved), f"{label} escapes raw root: {resolved}")
    return resolved


def _descriptor(
    value: Any,
    *,
    base: Path,
    label: str,
    raw_root: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    require(isinstance(value, Mapping), f"{label} descriptor is missing")
    raw_path = value.get("path")
    digest = value.get("sha256")
    size = value.get("bytes")
    require(isinstance(raw_path, str) and raw_path, f"{label} path is missing")
    require(_valid_sha(digest), f"{label} SHA-256 is invalid")
    require(type(size) is int and size >= 0, f"{label} byte count is invalid")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    if raw_root is not None:
        path = _under(candidate, raw_root, label)
    else:
        _reject_symlink_components(candidate, label)
        path = candidate.resolve()
    require(path.is_file(), f"{label} file is missing: {path}")
    require(path.stat().st_size == size, f"{label} byte count changed")
    require(sha256_file(path) == digest, f"{label} file hash changed")
    return {"path": str(path), "sha256": digest, "bytes": size}, path


def file_descriptor(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    _reject_symlink_components(Path(path), "file descriptor")
    resolved = Path(path).resolve()
    require(resolved.is_file(), f"cannot describe missing file: {resolved}")
    return {
        "path": str(resolved) if display_path is None else display_path,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def bytes_descriptor(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "sha256": sha256_bytes(payload), "bytes": len(payload)}


def _mapping_item(value: Any, key: str, label: str) -> Any:
    require(
        isinstance(value, Mapping) and value.get("__type__") == "mapping",
        f"{label} is not a stored mapping",
    )
    items = value.get("items")
    require(isinstance(items, Mapping) and key in items, f"{label} lacks {key}")
    return items[key]


def _thaw_scalar(value: Any, label: str) -> Any:
    if not isinstance(value, Mapping) or "__type__" not in value:
        return value
    kind = value.get("__type__")
    require(kind != "ndarray", f"{label} unexpectedly contains an array")
    if kind == "mapping":
        items = value.get("items")
        require(isinstance(items, Mapping), f"{label} stored mapping is invalid")
        return {str(key): _thaw_scalar(child, label) for key, child in items.items()}
    if kind in {"list", "tuple"}:
        items = value.get("items")
        require(isinstance(items, list), f"{label} stored sequence is invalid")
        output = [_thaw_scalar(child, label) for child in items]
        return output if kind == "list" else tuple(output)
    if kind == "path":
        require(isinstance(value.get("value"), str), f"{label} stored path is invalid")
        return value["value"]
    raise CompilerError(f"{label} has unsupported stored type: {kind}")


def _array_nodes(value: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if value.get("__type__") == "ndarray":
            output.append(dict(value))
        for child in value.values():
            output.extend(_array_nodes(child))
    elif isinstance(value, list):
        for child in value:
            output.extend(_array_nodes(child))
    return output


def _read_npy_header(handle: Any, label: str) -> dict[str, Any]:
    require(handle.read(6) == b"\x93NUMPY", f"{label} lacks NPY magic")
    version = handle.read(2)
    require(len(version) == 2 and version[0] in (1, 2, 3), f"{label} NPY version is invalid")
    length_bytes = 2 if version[0] == 1 else 4
    packed = handle.read(length_bytes)
    require(len(packed) == length_bytes, f"{label} NPY header length is truncated")
    header_length = struct.unpack("<H" if length_bytes == 2 else "<I", packed)[0]
    require(0 < header_length <= 1024 * 1024, f"{label} NPY header length is unsafe")
    raw_header = handle.read(header_length)
    require(len(raw_header) == header_length, f"{label} NPY header is truncated")
    try:
        header = ast.literal_eval(raw_header.decode("latin1").strip())
    except (UnicodeDecodeError, SyntaxError, ValueError) as error:
        raise CompilerError(f"{label} NPY header is invalid") from error
    require(
        isinstance(header, dict)
        and set(header) == {"descr", "fortran_order", "shape"},
        f"{label} NPY header fields changed",
    )
    require(isinstance(header["descr"], str), f"{label} NPY dtype is invalid")
    require(header["fortran_order"] is False, f"{label} NPY array is not C-order")
    require(
        isinstance(header["shape"], tuple)
        and all(type(item) is int and item >= 0 for item in header["shape"]),
        f"{label} NPY shape is invalid",
    )
    require("O" not in header["descr"], f"{label} NPY object dtype is prohibited")
    return header


def _numpy_itemsize(dtype: str, label: str) -> int:
    """Return the byte width of a simple ``numpy.dtype.str`` value.

    Both workshop payload writers deliberately prohibit object arrays and emit
    ``dtype.str`` rather than arbitrary structured dtype descriptions.  Keeping
    this parser narrow lets the compiler authenticate raw array bytes without
    importing NumPy (and without accepting pickle-backed object arrays).
    """

    match = re.fullmatch(r"[<>=|]([?bBiufcSUV])([0-9]+)?", dtype)
    require(match is not None, f"{label} NPY dtype is outside the safe scalar subset")
    kind, width_text = match.groups()
    if kind == "?":
        require(width_text in (None, "1"), f"{label} boolean width is invalid")
        return 1
    require(width_text is not None and int(width_text) > 0, f"{label} dtype width is invalid")
    width = int(width_text)
    return width * 4 if kind == "U" else width


def _expected_array_bytes(header: Mapping[str, Any], label: str) -> int:
    count = math.prod(header["shape"])
    return count * _numpy_itemsize(header["descr"], label)


def _consume_exact_array_data(
    handle: Any,
    expected_bytes: int,
    label: str,
    *,
    digest_prefix: bytes | None = None,
) -> str | None:
    require(expected_bytes <= 16 * 1024**3, f"{label} array is implausibly large")
    digest = hashlib.sha256(digest_prefix) if digest_prefix is not None else None
    remaining = expected_bytes
    while remaining:
        block = handle.read(min(1024 * 1024, remaining))
        require(bool(block), f"{label} array data is truncated")
        if digest is not None:
            digest.update(block)
        remaining -= len(block)
    require(handle.read(1) == b"", f"{label} array has trailing bytes")
    return None if digest is None else digest.hexdigest()


def _verify_npy(path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        with Path(path).open("rb") as handle:
            header = _read_npy_header(handle, label)
            identity_header = {
                "kind": "numpy",
                "dtype": header["descr"],
                "shape": list(header["shape"]),
            }
            value_sha256 = _consume_exact_array_data(
                handle,
                _expected_array_bytes(header, label),
                label,
                digest_prefix=canonical_bytes(identity_header, ensure_ascii=True),
            )
    except OSError as error:
        raise CompilerError(f"{label} is not a readable NPY: {path}") from error
    require(value_sha256 is not None, f"{label} value identity was not computed")
    return header, value_sha256


def _npy_data_sha256(path: Path, label: str) -> tuple[dict[str, Any], str]:
    """Return safe NPY header plus SHA-256 of the exact C-order data bytes."""

    try:
        with Path(path).open("rb") as handle:
            header = _read_npy_header(handle, label)
            data_sha256 = _consume_exact_array_data(
                handle,
                _expected_array_bytes(header, label),
                label,
                digest_prefix=b"",
            )
    except OSError as error:
        raise CompilerError(f"{label} is not a readable NPY: {path}") from error
    require(data_sha256 is not None, f"{label} data identity was not computed")
    return header, data_sha256


def _numpy_dtype_name(dtype: str, label: str) -> str:
    """Translate the safe NPY scalar subset to ``str(numpy.dtype)`` names."""

    match = re.fullmatch(r"([<>=|])([?bBiufc])([0-9]+)?", dtype)
    require(match is not None, f"{label} NPY dtype has no canonical scalar name")
    byte_order, kind, width_text = match.groups()
    width = 1 if width_text is None else int(width_text)
    require(byte_order in {"<", "=", "|"}, f"{label} uses a non-native big-endian dtype")
    names = {
        "?": "bool",
        "b": "bool",
        "B": "uint8",
        "i": f"int{width * 8}",
        "u": f"uint{width * 8}",
        "f": f"float{width * 8}",
        "c": f"complex{width * 8}",
    }
    return names[kind]


def _verify_npz(path: Path, nodes: Sequence[Mapping[str, Any]], label: str) -> None:
    expected = {f"{node['key']}.npy": node for node in nodes}
    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), f"{label} NPZ duplicates a member")
            require(set(names) == set(expected), f"{label} NPZ members changed")
            for name in sorted(names):
                require(Path(name).name == name, f"{label} NPZ member path is unsafe")
                require(
                    archive.getinfo(name).compress_type == zipfile.ZIP_STORED,
                    f"{label} NPZ member is unexpectedly compressed",
                )
                with archive.open(name, "r") as handle:
                    header = _read_npy_header(handle, f"{label}:{name}")
                    node = expected[name]
                    require(list(header["shape"]) == node.get("shape"), f"{label} NPZ shape changed")
                    require(header["descr"] == node.get("dtype"), f"{label} NPZ dtype changed")
                    _consume_exact_array_data(
                        handle,
                        _expected_array_bytes(header, f"{label}:{name}"),
                        f"{label}:{name}",
                    )
            require(archive.testzip() is None, f"{label} NPZ CRC check failed")
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, CompilerError):
            raise
        raise CompilerError(f"{label} is not a safe readable NPZ: {path}") from error


def _n3_logical_structure(value: Any) -> Any:
    if isinstance(value, list):
        return [_n3_logical_structure(child) for child in value]
    if not isinstance(value, Mapping):
        return value
    if value.get("__type__") in {"numpy", "torch"}:
        fields = ("__type__", "kind", "dtype", "shape", "value_sha256", "logical_type")
        return {key: value[key] for key in fields if key in value}
    return {str(key): _n3_logical_structure(child) for key, child in value.items()}


def _torch_archive_data_sha256(
    path: Path, *, dtype: Any, shape: Any, label: str
) -> str:
    """Authenticate one tensor artifact with a restricted metadata decoder.

    The official writer saves one detached, CPU-contiguous tensor per archive.
    Its restricted protocol-2 metadata and single raw storage jointly bind the
    declared dtype/shape to the writer's path-independent value identity.  No
    artifact-selected Python global is imported or executed.
    """

    dtype_sizes = {
        "torch.bool": 1,
        "torch.uint8": 1,
        "torch.int8": 1,
        "torch.int16": 2,
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.int32": 4,
        "torch.float32": 4,
        "torch.int64": 8,
        "torch.float64": 8,
        "torch.complex64": 8,
        "torch.complex128": 16,
    }
    require(dtype in dtype_sizes, f"{label} tensor dtype is unsupported")
    require(
        isinstance(shape, list) and all(type(item) is int and item >= 0 for item in shape),
        f"{label} tensor shape is invalid",
    )
    expected_bytes = math.prod(shape) * dtype_sizes[dtype]
    require(expected_bytes <= 16 * 1024**3, f"{label} tensor is implausibly large")

    storage_globals = {
        "torch.bool": "torch BoolStorage",
        "torch.uint8": "torch ByteStorage",
        "torch.int8": "torch CharStorage",
        "torch.int16": "torch ShortStorage",
        "torch.float16": "torch HalfStorage",
        "torch.bfloat16": "torch BFloat16Storage",
        "torch.int32": "torch IntStorage",
        "torch.float32": "torch FloatStorage",
        "torch.int64": "torch LongStorage",
        "torch.float64": "torch DoubleStorage",
        "torch.complex64": "torch ComplexFloatStorage",
        "torch.complex128": "torch ComplexDoubleStorage",
    }

    class RestrictedMetadataUnpickler(pickle.Unpickler):
        """Decode only the inert structure emitted by ``torch.save(tensor)``."""

        def __init__(self, handle: io.BytesIO) -> None:
            super().__init__(handle)
            self.persistent_storage_count = 0

        @staticmethod
        def _ordered_dict(*args: Any) -> dict[str, Any]:
            require(args == (), f"{label} tensor metadata has nonempty hooks")
            return {"kind": "empty_ordered_dict"}

        @staticmethod
        def _rebuild_tensor_v2(*args: Any) -> dict[str, Any]:
            require(len(args) == 6, f"{label} tensor rebuild arity changed")
            storage, offset, tensor_shape, stride, requires_grad, hooks = args
            require(isinstance(storage, Mapping), f"{label} tensor storage metadata is invalid")
            require(
                isinstance(tensor_shape, tuple)
                and all(type(item) is int and item >= 0 for item in tensor_shape),
                f"{label} serialized tensor shape is invalid",
            )
            require(
                isinstance(stride, tuple)
                and all(type(item) is int and item >= 0 for item in stride),
                f"{label} serialized tensor stride is invalid",
            )
            require(
                type(offset) is int and offset >= 0 and type(requires_grad) is bool,
                f"{label} serialized tensor flags are invalid",
            )
            require(
                hooks == {"kind": "empty_ordered_dict"},
                f"{label} serialized tensor hooks changed",
            )
            return {
                "storage": dict(storage),
                "offset": offset,
                "shape": list(tensor_shape),
                "stride": list(stride),
                "requires_grad": requires_grad,
            }

        def find_class(self, module: str, name: str) -> Any:
            identity = f"{module} {name}"
            if identity == "torch._utils _rebuild_tensor_v2":
                return self._rebuild_tensor_v2
            if identity == "collections OrderedDict":
                return self._ordered_dict
            if identity in storage_globals.values():
                return {"kind": "storage_type", "global": identity}
            raise pickle.UnpicklingError(
                f"{label} tensor metadata names unsupported global {identity!r}"
            )

        def persistent_load(self, persistent_id: Any) -> dict[str, Any]:
            require(
                isinstance(persistent_id, tuple) and len(persistent_id) == 5,
                f"{label} tensor persistent storage identity changed",
            )
            kind, storage_type, key, location, element_count = persistent_id
            require(
                kind == "storage"
                and isinstance(storage_type, Mapping)
                and storage_type.get("kind") == "storage_type"
                and isinstance(key, str)
                and key
                and location == "cpu"
                and type(element_count) is int
                and element_count >= 0,
                f"{label} tensor persistent storage metadata is invalid",
            )
            self.persistent_storage_count += 1
            return {
                "kind": "storage",
                "storage_global": storage_type["global"],
                "key": key,
                "location": location,
                "element_count": element_count,
            }

    def verify_metadata(payload: bytes, *, storage_key: str) -> None:
        require(0 < len(payload) <= 64 * 1024, f"{label} tensor metadata size is invalid")
        try:
            operations = list(pickletools.genops(payload))
        except (ValueError, pickle.UnpicklingError) as error:
            raise CompilerError(f"{label} tensor metadata opcode stream is invalid") from error
        require(len(operations) <= 256, f"{label} tensor metadata has too many opcodes")
        require(
            bool(operations)
            and operations[0][0].name == "PROTO"
            and operations[0][1] == 2
            and operations[-1][0].name == "STOP"
            and operations[-1][2] == len(payload) - 1,
            f"{label} tensor metadata is not the official protocol-2 envelope",
        )
        allowed_opcodes = {
            "PROTO", "GLOBAL", "BINPUT", "LONG_BINPUT", "BINGET", "LONG_BINGET",
            "MARK", "BINUNICODE", "BININT", "BININT1", "BININT2", "LONG1",
            "LONG4", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3", "BINPERSID",
            "NEWFALSE", "NEWTRUE", "EMPTY_TUPLE", "REDUCE", "STOP",
        }
        require(
            all(operation.name in allowed_opcodes for operation, _, _ in operations),
            f"{label} tensor metadata contains an unsupported opcode",
        )
        globals_seen = [
            argument for operation, argument, _ in operations if operation.name == "GLOBAL"
        ]
        require(
            globals_seen
            == [
                "torch._utils _rebuild_tensor_v2",
                storage_globals[dtype],
                "collections OrderedDict",
            ],
            f"{label} tensor metadata global sequence changed",
        )
        restricted = RestrictedMetadataUnpickler(io.BytesIO(payload))
        try:
            decoded = restricted.load()
        except CompilerError:
            raise
        except (EOFError, OverflowError, pickle.UnpicklingError, TypeError, ValueError) as error:
            raise CompilerError(f"{label} tensor metadata cannot be safely decoded") from error
        require(
            isinstance(decoded, Mapping)
            and restricted.persistent_storage_count == 1,
            f"{label} tensor metadata does not contain one tensor/storage",
        )
        storage = decoded.get("storage")
        require(
            isinstance(storage, Mapping)
            and storage.get("storage_global") == storage_globals[dtype]
            and storage.get("key") == storage_key
            and storage.get("location") == "cpu"
            and storage.get("element_count") == math.prod(shape),
            f"{label} tensor serialized storage identity changed",
        )
        expected_stride: list[int] = []
        running = 1
        for dimension in reversed(shape):
            expected_stride.insert(0, running)
            running *= max(dimension, 1)
        require(
            decoded.get("offset") == 0
            and decoded.get("shape") == shape
            and decoded.get("stride") == expected_stride
            and decoded.get("requires_grad") is False,
            f"{label} tensor serialized shape/dtype layout changed",
        )

    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
            require(len(names) == len(set(names)), f"{label} tensor archive duplicates a member")
            require(
                all(not name.startswith("/") and ".." not in Path(name).parts for name in names),
                f"{label} tensor archive contains an unsafe path",
            )
            storage_names = [
                name for name in names
                if "/data/" in name and not name.endswith("/")
            ]
            require(len(storage_names) == 1, f"{label} tensor archive storage inventory changed")
            metadata_names = [name for name in names if name.endswith("/data.pkl")]
            require(len(metadata_names) == 1, f"{label} tensor metadata inventory changed")
            metadata_name = metadata_names[0]
            archive_prefix = metadata_name[: -len("data.pkl")]
            require(
                storage_names[0].startswith(f"{archive_prefix}data/")
                and Path(storage_names[0]).name,
                f"{label} tensor storage/metadata archive roots differ",
            )
            metadata_info = archive.getinfo(metadata_name)
            require(
                0 < metadata_info.file_size <= 64 * 1024,
                f"{label} tensor metadata size is invalid",
            )
            verify_metadata(
                archive.read(metadata_info), storage_key=Path(storage_names[0]).name
            )
            info = archive.getinfo(storage_names[0])
            require(info.compress_type == zipfile.ZIP_STORED,
                    f"{label} tensor storage is unexpectedly compressed")
            require(info.file_size == expected_bytes, f"{label} tensor storage byte count changed")
            with archive.open(info, "r") as handle:
                data_sha256 = _consume_exact_array_data(
                    handle,
                    expected_bytes,
                    label,
                    digest_prefix=b"",
                )
            require(archive.testzip() is None, f"{label} tensor archive CRC check failed")
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, CompilerError):
            raise
        raise CompilerError(f"{label} is not a safe readable tensor archive") from error
    require(data_sha256 is not None, f"{label} tensor data identity was not computed")
    return data_sha256


def _verify_torch_archive(
    path: Path, *, node: Mapping[str, Any], label: str
) -> None:
    _torch_archive_data_sha256(
        path, dtype=node.get("dtype"), shape=node.get("shape"), label=label
    )
    # The N3 payload identity hashes the canonical header followed by the raw
    # storage bytes, not the digest of those bytes.  Stream it a second time in
    # the narrow N3 verifier so no tensor metadata is ever unpickled.
    dtype = node.get("dtype")
    shape = node.get("shape")
    prefix = canonical_bytes({"kind": "torch", "dtype": dtype, "shape": shape}, ensure_ascii=True)
    try:
        with zipfile.ZipFile(path, "r") as archive:
            storage_names = [name for name in archive.namelist() if "/data/" in name and not name.endswith("/")]
            require(len(storage_names) == 1, f"{label} tensor archive storage inventory changed")
            with archive.open(storage_names[0], "r") as handle:
                value_sha256 = _consume_exact_array_data(
                    handle,
                    archive.getinfo(storage_names[0]).file_size,
                    label,
                    digest_prefix=prefix,
                )
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, CompilerError):
            raise
        raise CompilerError(f"{label} is not a safe readable tensor archive") from error
    require(value_sha256 == node.get("value_sha256"), f"{label} tensor value identity changed")


def _verify_n3_payload_node(
    value: Any, *, root: Path, raw_root: Path, label: str
) -> None:
    if isinstance(value, list):
        for index, child in enumerate(value):
            _verify_n3_payload_node(
                child, root=root, raw_root=raw_root, label=f"{label}[{index}]"
            )
        return
    if not isinstance(value, Mapping):
        return
    kind = value.get("__type__")
    if kind in {"numpy", "torch"}:
        require(value.get("kind") == kind, f"{label} stored payload kind changed")
        require(_valid_sha(value.get("value_sha256")), f"{label} value identity is invalid")
        artifact, artifact_path = _descriptor(
            value.get("artifact"), base=root, raw_root=raw_root, label=f"{label} artifact"
        )
        del artifact
        require(artifact_path.parent == root.resolve(), f"{label} artifact escaped its manifest directory")
        if kind == "numpy":
            header, value_sha256 = _verify_npy(artifact_path, label)
            require(header["descr"] == value.get("dtype"), f"{label} numpy dtype changed")
            require(list(header["shape"]) == value.get("shape"), f"{label} numpy shape changed")
            require(
                value_sha256 == value.get("value_sha256"),
                f"{label} numpy value identity changed",
            )
        else:
            _verify_torch_archive(artifact_path, node=value, label=label)
    for key, child in value.items():
        if key != "artifact":
            _verify_n3_payload_node(
                child, root=root, raw_root=raw_root, label=f"{label}.{key}"
            )


def _verify_n3_payload_reference(
    value: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
    required_role: str,
) -> dict[str, Any]:
    require(
        isinstance(value, Mapping)
        and set(value) == {"manifest_path", "manifest_sha256", "logical_sha256"},
        f"{label} nested-payload reference fields changed",
    )
    require(_valid_sha(value.get("manifest_sha256")), f"{label} manifest SHA-256 is invalid")
    require(_valid_sha(value.get("logical_sha256")), f"{label} logical SHA-256 is invalid")
    raw_path = value.get("manifest_path")
    require(isinstance(raw_path, str) and raw_path, f"{label} manifest path is missing")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = base / candidate
    manifest_path = _under(candidate, raw_root, label)
    require(not candidate.is_symlink() and manifest_path.is_file(), f"{label} manifest is missing")
    require(sha256_file(manifest_path) == value["manifest_sha256"], f"{label} manifest hash changed")
    manifest = load_json(manifest_path, f"{label} manifest")
    require(
        set(manifest) == {"schema_version", "role", "structure", "logical_sha256"}
        and manifest.get("schema_version") == "wmf-lossless-nested-payload-v1",
        f"{label} manifest schema/fields changed",
    )
    require(manifest.get("role") == required_role, f"{label} payload role changed")
    logical = sha256_bytes(
        canonical_bytes(_n3_logical_structure(manifest.get("structure")), ensure_ascii=True)
    )
    require(
        logical == manifest.get("logical_sha256") == value.get("logical_sha256"),
        f"{label} logical identity changed",
    )
    _verify_n3_payload_node(
        manifest.get("structure"), root=manifest_path.parent, raw_root=raw_root, label=label
    )
    return dict(value)


def _verify_payload_descriptor(
    value: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
    required_role: str | None = None,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} payload descriptor is missing")
    observed = value.get("payload_sha256")
    require(_valid_sha(observed), f"{label} payload SHA-256 is invalid")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(
        sha256_bytes(canonical_bytes(unsigned, ensure_ascii=True)) == observed,
        f"{label} payload descriptor hash changed",
    )
    role = value.get("role")
    require(isinstance(role, str) and role, f"{label} payload role is missing")
    if required_role is not None:
        require(role == required_role, f"{label} payload role changed")
    structure = value.get("structure")
    require(isinstance(structure, Mapping), f"{label} payload structure is missing")
    nodes = _array_nodes(structure)
    count = value.get("array_count")
    require(type(count) is int and count >= 0, f"{label} payload array count is invalid")
    keys = [node.get("key") for node in nodes]
    require(
        len(nodes) == count
        and len(set(keys)) == len(keys)
        and all(isinstance(key, str) and key for key in keys),
        f"{label} payload array manifest is incomplete",
    )
    for node in nodes:
        require(
            isinstance(node.get("shape"), list)
            and all(type(item) is int and item >= 0 for item in node["shape"])
            and isinstance(node.get("dtype"), str)
            and node["dtype"],
            f"{label} payload array metadata is invalid",
        )
    if count:
        _, artifact_path = _descriptor(
            value.get("artifact"), base=base, label=f"{label} NPZ", raw_root=raw_root
        )
        _verify_npz(artifact_path, nodes, label)
    else:
        require(value.get("artifact") is None, f"{label} array-free payload names an artifact")
    return dict(value)


def _payload_binding(
    value: Mapping[str, Any], *, base: Path, raw_root: Path, label: str
) -> dict[str, Any]:
    """Emit a compact binding to a fully verified recorder payload."""

    raw_artifact = value.get("artifact")
    artifact = None
    if raw_artifact is not None:
        artifact, _ = _descriptor(
            raw_artifact, base=base, raw_root=raw_root, label=f"{label} artifact"
        )
    return {
        "role": value["role"],
        "payload_sha256": value["payload_sha256"],
        "array_count": value["array_count"],
        "structure_sha256": sha256_bytes(
            canonical_bytes(value["structure"], ensure_ascii=True)
        ),
        "artifact": artifact,
        "artifact_base": str(base.resolve()),
    }


def _small_payload_array(
    descriptor: Mapping[str, Any],
    *,
    member_key: str,
    base: Path,
    raw_root: Path,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    """Read one already-authenticated small recorder array without NumPy."""

    structure = descriptor.get("structure")
    node = _mapping_item(structure, member_key, label)
    require(
        isinstance(node, Mapping) and node.get("__type__") == "ndarray",
        f"{label} is not a retained ndarray",
    )
    _, artifact_path = _descriptor(
        descriptor.get("artifact"), base=base, raw_root=raw_root, label=f"{label} NPZ"
    )
    member = f"{node.get('key')}.npy"
    try:
        with zipfile.ZipFile(artifact_path, "r") as archive:
            require(member in archive.namelist(), f"{label} NPZ member is missing")
            require(archive.getinfo(member).compress_type == zipfile.ZIP_STORED,
                    f"{label} NPZ member is unexpectedly compressed")
            with archive.open(member, "r") as handle:
                header = _read_npy_header(handle, label)
                expected = _expected_array_bytes(header, label)
                require(expected <= 1024 * 1024, f"{label} unexpectedly exceeds one MiB")
                raw = handle.read(expected)
                require(len(raw) == expected and handle.read(1) == b"", f"{label} bytes changed")
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise CompilerError(f"{label} is not readable from its NPZ") from error
    require(
        list(header["shape"]) == node.get("shape") and header["descr"] == node.get("dtype"),
        f"{label} structure/array metadata changed",
    )
    return header, raw


def _recorder_numpy_value_identity(
    descriptor: Mapping[str, Any],
    node: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
) -> str:
    require(
        isinstance(node, Mapping) and node.get("__type__") == "ndarray",
        f"{label} is not a retained recorder ndarray",
    )
    _, artifact_path = _descriptor(
        descriptor.get("artifact"), base=base, raw_root=raw_root, label=f"{label} NPZ"
    )
    member = f"{node.get('key')}.npy"
    try:
        with zipfile.ZipFile(artifact_path, "r") as archive:
            require(member in archive.namelist(), f"{label} NPZ member is missing")
            require(archive.getinfo(member).compress_type == zipfile.ZIP_STORED,
                    f"{label} NPZ member is unexpectedly compressed")
            with archive.open(member, "r") as handle:
                header = _read_npy_header(handle, label)
                require(
                    list(header["shape"]) == node.get("shape")
                    and header["descr"] == node.get("dtype"),
                    f"{label} structure/array metadata changed",
                )
                prefix = canonical_bytes(
                    {
                        "kind": "numpy",
                        "dtype": header["descr"],
                        "shape": list(header["shape"]),
                    },
                    ensure_ascii=True,
                )
                value_sha256 = _consume_exact_array_data(
                    handle,
                    _expected_array_bytes(header, label),
                    label,
                    digest_prefix=prefix,
                )
    except (OSError, zipfile.BadZipFile, KeyError) as error:
        raise CompilerError(f"{label} is not readable from its NPZ") from error
    require(value_sha256 is not None, f"{label} value identity was not computed")
    return value_sha256


def _action_hashes_from_chunk_payload(
    descriptor: Mapping[str, Any],
    *,
    base: Path,
    raw_root: Path,
    model: str,
    label: str,
) -> list[str]:
    header, raw = _small_payload_array(
        descriptor,
        member_key="executable_action_chunk",
        base=base,
        raw_root=raw_root,
        label=f"{label} executable action chunk",
    )
    shape = list(header["shape"])
    require(
        len(shape) == 2 and shape[0] >= MODEL_LIMITS[model]["executed_prefix"]
        and shape[1] == 8,
        f"{label} executable action shape changed",
    )
    row_bytes = _numpy_itemsize(header["descr"], label) * shape[1]
    require(len(raw) == row_bytes * shape[0], f"{label} action chunk byte count changed")
    identity_header = {"dtype": header["descr"], "shape": [8], "order": "C"}
    prefix = canonical_bytes(identity_header, ensure_ascii=True)
    return [
        sha256_bytes(prefix + raw[index * row_bytes:(index + 1) * row_bytes])
        for index in range(shape[0])
    ]


def _d1_artifact_descriptor(
    entry: Mapping[str, Any], *, base: Path, raw_root: Path, label: str
) -> tuple[dict[str, Any], Path]:
    return _descriptor(
        {
            "path": entry.get("path"),
            "sha256": entry.get("file_sha256"),
            "bytes": entry.get("bytes"),
        },
        base=base,
        raw_root=raw_root,
        label=label,
    )


def _verify_d1_exact_mapping(
    value: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
    prefix: str,
) -> dict[str, Any]:
    """Deeply authenticate one ``save_exact_mapping`` result.

    The mapping-level digest intentionally excludes paths, while each entry's
    descriptor authenticates its concrete artifact.  Both layers are required.
    Tensor archives are inspected without unpickling executable metadata.
    """

    require(
        isinstance(value, Mapping)
        and set(value)
        == {"entries", "entry_count", "content_sha256", "content_hash_definition"},
        f"{label} mapping fields changed",
    )
    entries = value.get("entries")
    require(isinstance(entries, list) and bool(entries), f"{label} mapping is empty")
    require(value.get("entry_count") == len(entries), f"{label} entry count changed")
    require(
        value.get("content_hash_definition") == D1_MAPPING_CONTENT_HASH_DEFINITION,
        f"{label} content-hash definition changed",
    )
    require(_valid_sha(value.get("content_sha256")), f"{label} content SHA-256 is invalid")
    identities: list[dict[str, Any]] = []
    observed_keys: list[str] = []
    observed_paths: set[Path] = set()
    common_parent: Path | None = None
    for index, raw_entry in enumerate(entries):
        require(isinstance(raw_entry, Mapping), f"{label} entry {index} is invalid")
        entry = dict(raw_entry)
        key = entry.get("key")
        kind = entry.get("kind")
        require(isinstance(key, str) and key, f"{label} entry {index} key is invalid")
        require(kind in {"numpy_array", "torch_tensor", "json_value"},
                f"{label} entry {index} kind is invalid")
        array_fields = {
            "key", "kind", "path", "file_sha256", "bytes", "shape", "dtype", "data_sha256"
        }
        json_fields = {"key", "kind", "path", "file_sha256", "bytes", "json_sha256"}
        require(
            set(entry) == (json_fields if kind == "json_value" else array_fields),
            f"{label} entry {index} fields changed",
        )
        artifact, artifact_path = _d1_artifact_descriptor(
            entry, base=base, raw_root=raw_root, label=f"{label} entry {index} artifact"
        )
        del artifact
        require(artifact_path not in observed_paths, f"{label} repeats an artifact path")
        observed_paths.add(artifact_path)
        if common_parent is None:
            common_parent = artifact_path.parent
        require(artifact_path.parent == common_parent, f"{label} artifacts span directories")
        expected_suffix = ".json" if kind == "json_value" else (".npy" if kind == "numpy_array" else ".pt")
        require(
            artifact_path.name == f"{prefix}_{index:03d}{expected_suffix}",
            f"{label} artifact ordering/name changed",
        )
        if kind == "numpy_array":
            header, data_sha256 = _npy_data_sha256(artifact_path, f"{label} entry {index}")
            require(list(header["shape"]) == entry.get("shape"), f"{label} entry {index} shape changed")
            require(_numpy_dtype_name(header["descr"], label) == entry.get("dtype"),
                    f"{label} entry {index} dtype changed")
            require(data_sha256 == entry.get("data_sha256"),
                    f"{label} entry {index} data hash changed")
        elif kind == "torch_tensor":
            require(_valid_sha(entry.get("data_sha256")), f"{label} entry {index} data hash is invalid")
            require(
                _torch_archive_data_sha256(
                    artifact_path,
                    dtype=entry.get("dtype"),
                    shape=entry.get("shape"),
                    label=f"{label} entry {index}",
                )
                == entry["data_sha256"],
                f"{label} entry {index} data hash changed",
            )
        else:
            require(_valid_sha(entry.get("json_sha256")), f"{label} entry {index} JSON hash is invalid")
            try:
                decoded = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CompilerError(f"{label} entry {index} JSON artifact is unreadable") from error
            require(
                sha256_bytes(canonical_bytes(decoded)) == entry["json_sha256"],
                f"{label} entry {index} JSON content hash changed",
            )
        identity_fields = ("key", "kind", "shape", "dtype", "data_sha256", "json_sha256")
        identities.append({field: entry[field] for field in identity_fields if field in entry})
        observed_keys.append(key)
    require(observed_keys == sorted(observed_keys), f"{label} keys are not sorted")
    require(len(set(observed_keys)) == len(observed_keys), f"{label} keys are duplicated")
    require(
        sha256_bytes(canonical_bytes(identities)) == value["content_sha256"],
        f"{label} mapping content hash changed",
    )
    return dict(value)


def _verify_d1_array_artifact(
    value: Any,
    *,
    base: Path,
    raw_root: Path,
    label: str,
    tensor: bool,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} is missing")
    required = {"path", "file_sha256", "bytes", "shape", "dtype", "data_sha256"}
    require(required <= set(value), f"{label} fields are incomplete")
    require(_valid_sha(value.get("data_sha256")), f"{label} data hash is invalid")
    _, path = _d1_artifact_descriptor(value, base=base, raw_root=raw_root, label=label)
    if tensor:
        observed = _torch_archive_data_sha256(
            path, dtype=value.get("dtype"), shape=value.get("shape"), label=label
        )
    else:
        header, observed = _npy_data_sha256(path, label)
        require(list(header["shape"]) == value.get("shape"), f"{label} shape changed")
        require(_numpy_dtype_name(header["descr"], label) == value.get("dtype"),
                f"{label} dtype changed")
    require(observed == value["data_sha256"], f"{label} data hash changed")
    return dict(value)


def _event_rows(rows: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [dict(row) for row in rows if row.get("kind") == kind]


def _event_payload(row: Mapping[str, Any], label: str) -> dict[str, Any]:
    payload = row.get("payload")
    require(isinstance(payload, Mapping), f"{label} payload is invalid")
    return dict(payload)


def _rfc3339_ns(value: Any, label: str) -> str:
    require(type(value) is int and value >= 0, f"{label} wall clock is invalid")
    seconds, nanoseconds = divmod(value, 1_000_000_000)
    require(seconds <= 253_402_300_799, f"{label} wall clock is outside RFC3339 range")
    try:
        stamp = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    except (OverflowError, OSError, ValueError) as error:
        raise CompilerError(f"{label} wall clock is outside RFC3339 range") from error
    # The frozen annotation validator uses ``datetime.fromisoformat``, whose
    # supported precision is microseconds.  The immutable journal retains the
    # exact nanosecond integer; this interoperable display field truncates only
    # its sub-microsecond remainder.
    return f"{stamp}.{nanoseconds // 1_000:06d}Z"


def _load_planned_cells(path: Path) -> dict[tuple[str, str], dict[str, dict[str, str]]]:
    output: dict[tuple[str, str], dict[str, dict[str, str]]] = {
        (model, layout): {} for model in MODELS for layout in LAYOUT_IDS
    }
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("phase") != "development" or row.get("selected") != "true":
                    continue
                model = row.get("model_config")
                layout = row.get("layout_pair_id")
                if (model, layout) not in output:
                    continue
                condition = f"{row.get('layout_arm')}_{row.get('command')}"
                cell_id = row.get("cell_id")
                require(condition in CONDITIONS, f"planned cell has invalid condition: {cell_id}")
                require(isinstance(cell_id, str) and cell_id, "planned cell ID is missing")
                require(cell_id not in output[(model, layout)], f"planned cell is duplicated: {cell_id}")
                output[(model, layout)][cell_id] = dict(row)
    except OSError as error:
        raise CompilerError("planned-cell CSV is unreadable") from error
    for key, rows in output.items():
        require(len(rows) == len(CONDITIONS), f"planned-cell CSV is incomplete for {key}")
        require(
            {f"{row['layout_arm']}_{row['command']}" for row in rows.values()}
            == set(CONDITIONS),
            f"planned-cell condition coverage changed for {key}",
        )
    return output


def _verify_journal(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        rows, tail = freeze._verify_journal(path)
    except Exception as error:
        raise CompilerError(f"recording journal failed deep validation: {path}: {error}") from error
    prior_monotonic: int | None = None
    prior_wall: int | None = None
    for row in rows:
        monotonic = row.get("monotonic_ns")
        wall = row.get("wall_time_ns")
        require(type(monotonic) is int and monotonic >= 0, "journal monotonic clock is invalid")
        require(type(wall) is int and wall >= 0, "journal wall clock is invalid")
        if prior_monotonic is not None:
            require(monotonic >= prior_monotonic, "journal monotonic clock moved backward")
            require(wall >= prior_wall, "journal wall clock moved backward")
        prior_monotonic = monotonic
        prior_wall = wall
    return rows, tail


def _verify_d1_first_request_temporal_metrics(value: Any, *, label: str) -> None:
    """Replay the producer's first-request cache reinitialization gate."""

    require(
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(row, Mapping) for row in value)
        and sorted(row.get("rank") for row in value) == [0, 1],
        f"{label} temporal/cache rank inventory changed",
    )
    for row in value:
        rank = row.get("rank")
        before = row.get("temporal_before")
        after = row.get("temporal_after")
        require(
            isinstance(before, Mapping) and isinstance(after, Mapping),
            f"{label} rank {rank} temporal scan is missing",
        )
        require(
            before.get("current_start_frame") == 0
            and type(after.get("current_start_frame")) is int
            and after["current_start_frame"] > 0,
            f"{label} rank {rank} did not start cleanly and advance",
        )
        before_fields = before.get("fields")
        after_fields = after.get("fields")
        require(
            isinstance(before_fields, Mapping) and isinstance(after_fields, Mapping),
            f"{label} rank {rank} cache scan is missing",
        )
        require(
            all(
                isinstance(before_fields.get(field), Mapping)
                and before_fields[field].get("is_none") is True
                and isinstance(after_fields.get(field), Mapping)
                and after_fields[field].get("is_none") is False
                for field in D1_CACHE_FIELDS
            ),
            f"{label} rank {rank} cache did not transition from empty to initialized",
        )
        events = row.get("cache_reinitialization")
        require(
            isinstance(events, Mapping)
            and isinstance(events.get("_create_kv_caches"), list)
            and len(events["_create_kv_caches"]) == 1
            and isinstance(events.get("_create_crossattn_caches"), list)
            and len(events["_create_crossattn_caches"]) == 1,
            f"{label} rank {rank} cache constructors were not called exactly once",
        )


def _verify_d1_development_decode_shapes(
    request: Mapping[str, Any],
    *,
    request_index: int,
    limits: Mapping[str, Any],
    label: str,
) -> None:
    """Authenticate the retained per-request D1 decode shape schedule.

    This is intentionally a structural check only.  Accepting the five-frame
    shape on a cache-active request does not give those frame indices the
    conditioning-origin timing semantics of a nine-frame decode.  Eligibility
    is established separately by the signed timing sidecar.
    """

    require(type(request_index) is int and request_index >= 0,
            f"{label} request index is invalid")
    schedule = limits.get("development_decode_shapes")
    require(
        isinstance(schedule, Mapping)
        and set(schedule) == {"full_conditioning_origin", "incremental_standalone"},
        f"{label} compiler decode schedule is invalid",
    )
    schedule_key = (
        "full_conditioning_origin"
        if request_index % 4 == 0
        else "incremental_standalone"
    )
    expected = schedule[schedule_key]
    require(
        isinstance(expected, Mapping)
        and set(expected) == {"latent", "tensor", "rgb"}
        and all(
            isinstance(expected.get(key), list)
            and all(type(item) is int and item > 0 for item in expected[key])
            for key in ("latent", "tensor", "rgb")
        ),
        f"{label} compiler decode schedule entry is invalid",
    )
    latent = request.get("latent_video")
    decode = request.get("offline_decode")
    decoded = decode.get("decoded_rgb") if isinstance(decode, Mapping) else None
    decoded_tensor = (
        decode.get("decoded_tensor") if isinstance(decode, Mapping) else None
    )
    require(
        isinstance(latent, Mapping)
        and latent.get("shape") == expected["latent"],
        f"{label} retained latent shape/schedule changed",
    )
    require(
        isinstance(decoded, Mapping)
        and decoded.get("shape") == expected["rgb"],
        f"{label} decoded RGB shape/schedule changed",
    )
    require(
        isinstance(decoded_tensor, Mapping)
        and decoded_tensor.get("shape") == expected["tensor"],
        f"{label} decoded tensor shape/schedule changed",
    )


def _verify_request_receipt(
    *,
    descriptor: Mapping[str, Any],
    request_path: Path,
    model: str,
    cell: Mapping[str, Any],
    request_index: int,
    raw_root: Path,
) -> dict[str, Any]:
    try:
        _, request = freeze._validate_server_request(
            request_path,
            descriptor=descriptor,
            model=model,
            cell=cell,
            request_index=request_index,
        )
    except Exception as error:
        raise CompilerError(
            f"{model} request {request_index} failed official-receipt validation: {error}"
        ) from error
    limits = MODEL_LIMITS[model]
    if model == "N3":
        require(request.get("generation_qualification_request") is False,
                "N3 development request is marked as generation qualification")
        require(
            request.get("sampling_seed") == cell.get("effective_seed")
            and request.get("returned_action_shape") == [limits["returned_actions"], 8]
            and request.get("joint_generation_calls") == 1
            and request.get("decode_calls") == 1,
            "N3 official behavioral generation contract changed",
        )
        require(
            type(request.get("started_wall_time_ns")) is int
            and type(request.get("completed_wall_time_ns")) is int
            and request["completed_wall_time_ns"] >= request["started_wall_time_ns"]
            and type(request.get("started_monotonic_ns")) is int
            and type(request.get("completed_monotonic_ns")) is int
            and request["completed_monotonic_ns"] >= request["started_monotonic_ns"],
            "N3 official request clocks changed",
        )
        require(request.get("decoded_future_shape", [None])[0] == limits["decoded_frames"],
                "N3 decoded frame count changed")
        required_payloads = {
            "wire_request": "exact_wire_request",
            "exact_transformed_model_input": "exact_transformed_model_input",
            "raw_generated_action": "raw_generated_action",
            "retained_vision_latent": "retained_vision_latent",
            "exact_decoder_input_latent": "exact_decoder_input_latent",
            "raw_decoder_output": "raw_decoder_output",
            "official_returned_response": "official_returned_response",
        }
        for key, role in required_payloads.items():
            _verify_n3_payload_reference(
                request.get(key), base=request_path.parent, raw_root=raw_root,
                label=f"N3 request {request_index} {key}", required_role=role,
            )
    else:
        returned = request.get("official_returned_action")
        require(isinstance(returned, Mapping)
                and returned.get("shape") == [limits["returned_actions"], 8]
                and returned.get("dtype") == "float32"
                and _valid_sha(returned.get("data_sha256")),
                "D1 returned action shape changed")
        decode = request.get("offline_decode")
        decoded = decode.get("decoded_rgb") if isinstance(decode, Mapping) else None
        decoded_tensor = (
            decode.get("decoded_tensor") if isinstance(decode, Mapping) else None
        )
        latent = request.get("latent_video")
        decoded_shape = decoded.get("shape") if isinstance(decoded, Mapping) else None
        decoded_tensor_shape = (
            decoded_tensor.get("shape") if isinstance(decoded_tensor, Mapping) else None
        )
        _verify_d1_development_decode_shapes(
            request,
            request_index=request_index,
            limits=limits,
            label=f"D1 request {request_index}",
        )
        require(isinstance(decode, Mapping)
                and decode.get("requested") is True and decode.get("performed") is True
                and isinstance(latent, Mapping) and _valid_sha(latent.get("data_sha256"))
                and decode.get("latent_data_sha256_before") == latent["data_sha256"]
                and decode.get("latent_data_sha256_after") == latent["data_sha256"]
                and isinstance(decoded, Mapping)
                and isinstance(decoded_shape, list)
                and len(decoded_shape) == 4
                and decoded_shape[-1] == 3
                and decoded.get("dtype") == "uint8"
                and isinstance(decoded_tensor, Mapping)
                and isinstance(decoded_tensor_shape, list)
                and len(decoded_tensor_shape) == 5
                and decoded_tensor_shape[:2] == [1, 3]
                and decoded_tensor_shape[3:] == decoded_shape[1:3],
                "D1 decoded RGB/tensor frame schedule changed")
        metrics = request.get("temporal_and_cache_rank_metrics")
        require(
            request.get("probe_id") is None
            and request.get("official_forward_call_count") == 1
            and isinstance(metrics, list)
            and len(metrics) == 2
            and all(isinstance(row, Mapping) for row in metrics)
            and sorted(row.get("rank") for row in metrics) == [0, 1],
            "D1 official conditional request/rank contract changed",
        )
        if request_index == 0:
            _verify_d1_first_request_temporal_metrics(
                metrics, label=f"D1 request {request_index}"
            )
        for mapping_key, prefix in (
            ("raw_inputs", "raw"),
            ("converted_inputs", "converted"),
            ("normalized_model_inputs", "normalized"),
        ):
            _verify_d1_exact_mapping(
                request.get(mapping_key),
                base=request_path.parent,
                raw_root=raw_root,
                label=f"D1 request {request_index} {mapping_key}",
                prefix=prefix,
            )
        _verify_d1_array_artifact(
            returned,
            base=request_path.parent,
            raw_root=raw_root,
            label=f"D1 request {request_index} official returned action",
            tensor=False,
        )
        _verify_d1_array_artifact(
            latent,
            base=request_path.parent,
            raw_root=raw_root,
            label=f"D1 request {request_index} latent video",
            tensor=True,
        )
        _verify_d1_array_artifact(
            decoded_tensor,
            base=request_path.parent,
            raw_root=raw_root,
            label=f"D1 request {request_index} decoded tensor",
            tensor=True,
        )
        _verify_d1_array_artifact(
            decoded,
            base=request_path.parent,
            raw_root=raw_root,
            label=f"D1 request {request_index} decoded RGB",
            tensor=False,
        )
    return request


def _discover_requests(
    *,
    rows: Sequence[Mapping[str, Any]],
    completion_path: Path,
    completion: Mapping[str, Any],
    cell: Mapping[str, Any],
    model: str,
    raw_root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[list[str]],
    list[dict[str, Any]],
]:
    limits = MODEL_LIMITS[model]
    packed = _event_rows(rows, "model_request_packed")
    sent = _event_rows(rows, "model_request_sent")
    responses = _event_rows(rows, "model_response_received")
    completed = _event_rows(rows, "model_request_completed")
    require(
        len(packed) == len(sent) == len(responses) == len(completed)
        == limits["request_count"],
        f"{cell['cell_id']} request event count changed",
    )
    request_descriptors: list[dict[str, Any]] = []
    request_values: list[dict[str, Any]] = []
    response_descriptors: list[dict[str, Any]] = []
    executable_action_hashes: list[list[str]] = []
    packed_request_bindings: list[dict[str, Any]] = []
    execution = completion.get("request_execution")
    require(
        isinstance(execution, list) and len(execution) == limits["request_count"],
        f"{cell['cell_id']} completion request inventory changed",
    )
    seen_paths: set[Path] = set()
    for request_index, response_row in enumerate(responses):
        packed_row = packed[request_index]
        sent_row = sent[request_index]
        completed_row = completed[request_index]
        packed_payload = _event_payload(packed_row, f"packed request {request_index}")
        sent_payload = _event_payload(sent_row, f"sent request {request_index}")
        payload = _event_payload(response_row, f"response {request_index}")
        completed_payload = _event_payload(completed_row, f"completed request {request_index}")
        require(
            packed_payload.get("request_index") == sent_payload.get("request_index")
            == payload.get("request_index") == completed_payload.get("request_index")
            == request_index,
            f"{cell['cell_id']} request event identity changed",
        )
        require(
            packed_row["sequence"] < sent_row["sequence"] < response_row["sequence"]
            < completed_row["sequence"],
            f"{cell['cell_id']} request event order changed",
        )
        packed_descriptor = _verify_payload_descriptor(
            packed_payload.get("model_request_artifact"),
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} packed model request {request_index}",
            required_role="model_request",
        )
        _mapping_item(
            packed_descriptor["structure"],
            "extracted_preprocessing_output",
            f"{cell['cell_id']} packed model request {request_index}",
        )
        _mapping_item(
            packed_descriptor["structure"],
            "wire_request",
            f"{cell['cell_id']} packed model request {request_index}",
        )
        require(
            type(packed_payload.get("pack_monotonic_ns")) is int
            and packed_payload["pack_monotonic_ns"] >= 0
            and type(sent_payload.get("send_monotonic_ns")) is int
            and sent_payload["send_monotonic_ns"] >= packed_payload["pack_monotonic_ns"]
            and type(payload.get("receive_monotonic_ns")) is int
            and payload["receive_monotonic_ns"] >= sent_payload["send_monotonic_ns"],
            f"{cell['cell_id']} request monotonic clocks changed",
        )
        require(
            payload.get("future_kinds") == limits["future_kinds"],
            f"{cell['cell_id']} response future evidence inventory changed",
        )
        require(payload.get("request_index") == request_index,
                f"{cell['cell_id']} response order changed")
        response_descriptor = _verify_payload_descriptor(
            payload.get("response_artifact"),
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} response {request_index}",
            required_role="model_response",
        )
        raw_response = _mapping_item(
            response_descriptor["structure"], "raw_response",
            f"{cell['cell_id']} response {request_index}",
        )
        transported = _thaw_scalar(
            _mapping_item(
                raw_response,
                "wmf_server_request_receipt",
                f"{cell['cell_id']} response {request_index} raw response",
            ),
            f"{cell['cell_id']} transported request descriptor",
        )
        request_descriptor, request_path = _descriptor(
            transported,
            base=completion_path.parent,
            label=f"{cell['cell_id']} official request {request_index}",
            raw_root=raw_root,
        )
        require(request_path not in seen_paths, f"{cell['cell_id']} repeats a request receipt")
        seen_paths.add(request_path)
        response_index = _thaw_scalar(
            _mapping_item(raw_response, "wmf_request_index", "raw response"),
            "transported request index",
        )
        require(response_index == request_index, "transported request index changed")
        if model == "N3":
            require(
                _thaw_scalar(_mapping_item(raw_response, "wmf_cell_id", "raw response"), "cell")
                == cell["cell_id"],
                "N3 transported cell identity changed",
            )
            require(
                _thaw_scalar(
                    _mapping_item(raw_response, "wmf_action_step_start", "raw response"),
                    "action start",
                )
                == request_index * limits["executed_prefix"],
                "N3 transported action start changed",
            )
        request = _verify_request_receipt(
            descriptor=request_descriptor,
            request_path=request_path,
            model=model,
            cell=cell,
            request_index=request_index,
            raw_root=raw_root,
        )
        if model == "N3":
            recorder_action_node = _mapping_item(
                raw_response, "action", "N3 recorder raw response"
            )
            recorder_video_node = _mapping_item(
                raw_response, "video", "N3 recorder raw response"
            )
            recorder_future = _mapping_item(
                response_descriptor["structure"],
                "future_evidence",
                "N3 recorder response",
            )
            recorder_decoded_node = _mapping_item(
                recorder_future, "decoded", "N3 recorder future evidence"
            )
            recorder_action_identity = _recorder_numpy_value_identity(
                response_descriptor,
                recorder_action_node,
                base=completion_path.parent,
                raw_root=raw_root,
                label=f"{cell['cell_id']} request {request_index} recorder N3 action",
            )
            recorder_video_identity = _recorder_numpy_value_identity(
                response_descriptor,
                recorder_video_node,
                base=completion_path.parent,
                raw_root=raw_root,
                label=f"{cell['cell_id']} request {request_index} recorder N3 video",
            )
            require(
                _recorder_numpy_value_identity(
                    response_descriptor,
                    recorder_decoded_node,
                    base=completion_path.parent,
                    raw_root=raw_root,
                    label=f"{cell['cell_id']} request {request_index} recorder decoded future",
                ) == recorder_video_identity,
                "N3 raw-response/recorder decoded future differs",
            )
            official_response_ref = request.get("official_returned_response")
            official_response_candidate = Path(str(official_response_ref["manifest_path"]))
            if not official_response_candidate.is_absolute():
                official_response_candidate = request_path.parent / official_response_candidate
            official_response_path = _under(
                official_response_candidate,
                raw_root,
                f"{cell['cell_id']} request {request_index} official N3 response",
            )
            official_response = load_json(
                official_response_path,
                f"{cell['cell_id']} request {request_index} official N3 response",
            )
            official_action_node = _mapping_item(
                official_response.get("structure"), "action", "official N3 response"
            )
            official_video_node = _mapping_item(
                official_response.get("structure"), "video", "official N3 response"
            )
            require(
                isinstance(official_action_node, Mapping)
                and official_action_node.get("value_sha256") == recorder_action_identity
                and isinstance(official_video_node, Mapping)
                and official_video_node.get("value_sha256") == recorder_video_identity,
                "N3 recorder raw response differs from official returned response",
            )
        else:
            episode_id = _thaw_scalar(
                _mapping_item(raw_response, "wmf_episode_context_id", "raw response"),
                "D1 episode context",
            )
            require(isinstance(episode_id, str) and episode_id == request.get("episode_id"),
                    "D1 transported episode context changed")
            raw_future = _thaw_scalar(
                _mapping_item(raw_response, "future_evidence", "D1 raw response"),
                "D1 raw-response future evidence",
            )
            outer_future = _thaw_scalar(
                _mapping_item(
                    response_descriptor["structure"],
                    "future_evidence",
                    "D1 recorder response",
                ),
                "D1 recorder future evidence",
            )
            require(
                isinstance(raw_future, Mapping) and raw_future == outer_future,
                "D1 raw-response/recorder future evidence differs",
            )
            expected_future = {
                "latent": request.get("latent_video"),
                "decoded.tensor": request.get("offline_decode", {}).get("decoded_tensor"),
                "decoded.rgb": request.get("offline_decode", {}).get("decoded_rgb"),
            }
            observed_future = {
                "latent": raw_future.get("latent"),
                "decoded.tensor": raw_future.get("decoded", {}).get("tensor")
                if isinstance(raw_future.get("decoded"), Mapping) else None,
                "decoded.rgb": raw_future.get("decoded", {}).get("rgb")
                if isinstance(raw_future.get("decoded"), Mapping) else None,
            }
            for future_key, official in expected_future.items():
                observed = observed_future[future_key]
                require(
                    isinstance(official, Mapping) and isinstance(observed, Mapping),
                    f"D1 {future_key} future descriptor is missing",
                )
                official_descriptor, _ = _descriptor(
                    {
                        "path": official.get("path"),
                        "sha256": official.get("file_sha256"),
                        "bytes": official.get("bytes"),
                    },
                    base=request_path.parent,
                    raw_root=raw_root,
                    label=f"D1 official {future_key}",
                )
                observed_descriptor, _ = _descriptor(
                    observed,
                    base=completion_path.parent,
                    raw_root=raw_root,
                    label=f"D1 recorder {future_key}",
                )
                require(
                    observed_descriptor == official_descriptor,
                    f"D1 recorder/official {future_key} future differs",
                )
        request_descriptors.append(request_descriptor)
        request_values.append(request)
        response_descriptors.append(response_descriptor)
        chunk_descriptor = _verify_payload_descriptor(
            completed_payload.get("action_chunks_artifact"),
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} request {request_index} action chunks",
            required_role="action_chunks",
        )
        execution_row = execution[request_index]
        expected_start = request_index * limits["executed_prefix"]
        expected_current = f"obs_{expected_start:06d}"
        expected_preceding = None if expected_start == 0 else f"obs_{expected_start - 1:06d}"
        require(
            isinstance(execution_row, Mapping)
            and execution_row.get("request_index") == request_index
            and execution_row.get("action_step_start") == expected_start
            and execution_row.get("current_observation_id") == expected_current
            and execution_row.get("preceding_observation_id") == expected_preceding
            and execution_row.get("action_chunks_artifact") == chunk_descriptor
            and execution_row.get("returned_actions") == limits["returned_actions"]
            and execution_row.get("eligible_executable_prefix_actions")
            == limits["executed_prefix"]
            and execution_row.get("future_kinds") == limits["future_kinds"],
            f"{cell['cell_id']} request {request_index} completion/journal binding changed",
        )
        require(
            packed_payload.get("action_step_start") == execution_row.get("action_step_start")
            and packed_payload.get("current_observation_id")
            == execution_row.get("current_observation_id")
            and packed_payload.get("preceding_observation_id")
            == execution_row.get("preceding_observation_id")
            and packed_payload.get("returned_action_horizon") == limits["returned_actions"]
            and packed_payload.get("executed_prefix_horizon") == limits["executed_prefix"]
            and packed_payload.get("required_future_evidence")
            == limits["required_future_evidence"],
            f"{cell['cell_id']} packed/completion observation or horizon binding changed",
        )
        owned_proposals = [
            row
            for row in _event_rows(rows, "policy_action_returned")
            if _event_payload(row, "owned action proposal").get("request_index")
            == request_index
        ]
        require(bool(owned_proposals), f"{cell['cell_id']} request {request_index} owns no action")
        require(
            completed_row["sequence"] < owned_proposals[0]["sequence"],
            f"{cell['cell_id']} request {request_index} completed after its first owned action",
        )
        require(
            completed_payload.get("returned_action_shape")
            == [limits["returned_actions"], 8]
            and isinstance(completed_payload.get("executable_action_shape"), list)
            and completed_payload["executable_action_shape"][:1]
            and completed_payload["executable_action_shape"][0]
            >= limits["executed_prefix"]
            and completed_payload["executable_action_shape"][1:] == [8]
            and completed_payload.get("missing_future_evidence") == [],
            f"{cell['cell_id']} request {request_index} action/future completion changed",
        )
        executable_action_hashes.append(_action_hashes_from_chunk_payload(
            chunk_descriptor,
            base=completion_path.parent,
            raw_root=raw_root,
            model=model,
            label=f"{cell['cell_id']} request {request_index}",
        ))
        packed_request_bindings.append(_payload_binding(
            packed_descriptor,
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} packed model request {request_index}",
        ))
        returned_header, returned_raw = _small_payload_array(
            chunk_descriptor,
            member_key="returned_action_chunk",
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} request {request_index} returned action chunk",
        )
        require(
            list(returned_header["shape"]) == [limits["returned_actions"], 8],
            f"{cell['cell_id']} request {request_index} retained returned-action shape changed",
        )
        if model == "D1":
            official_action = request.get("official_returned_action")
            require(
                isinstance(official_action, Mapping)
                and sha256_bytes(returned_raw) == official_action.get("data_sha256"),
                f"{cell['cell_id']} request {request_index} recorder/official D1 action differs",
            )
        else:
            response_reference = request.get("official_returned_response")
            response_candidate = Path(str(response_reference["manifest_path"]))
            if not response_candidate.is_absolute():
                response_candidate = request_path.parent / response_candidate
            manifest_path = _under(
                response_candidate,
                raw_root,
                f"{cell['cell_id']} request {request_index} official N3 response",
            )
            response_manifest = load_json(
                manifest_path,
                f"{cell['cell_id']} request {request_index} official N3 response",
            )
            official_node = _mapping_item(
                response_manifest.get("structure"),
                "action",
                f"{cell['cell_id']} request {request_index} official N3 response",
            )
            returned_identity = sha256_bytes(
                canonical_bytes(
                    {
                        "kind": "numpy",
                        "dtype": returned_header["descr"],
                        "shape": list(returned_header["shape"]),
                    },
                    ensure_ascii=True,
                ) + returned_raw
            )
            require(
                isinstance(official_node, Mapping)
                and official_node.get("__type__") == "numpy"
                and official_node.get("value_sha256") == returned_identity,
                f"{cell['cell_id']} request {request_index} recorder/official N3 action differs",
            )
    if model == "D1":
        listed = cell.get("server_request_receipts")
        require(isinstance(listed, list) and len(listed) == len(request_descriptors),
                "D1 cell request descriptor inventory changed")
        normalized = [
            _descriptor(
                item, base=completion_path.parent, label=f"D1 listed request {index}",
                raw_root=raw_root,
            )[0]
            for index, item in enumerate(listed)
        ]
        require(normalized == request_descriptors,
                "D1 listed request receipts differ from transported recorder receipts")
    return (
        request_descriptors,
        request_values,
        response_descriptors,
        executable_action_hashes,
        packed_request_bindings,
    )


def _validate_context_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    completion: Mapping[str, Any],
    completion_path: Path,
    raw_root: Path,
    expected_receipt: Mapping[str, Any],
    cell_id: str,
) -> dict[str, Any]:
    events = _event_rows(rows, "model_context_reset")
    require(len(events) == 1, f"{cell_id} model-context reset event count changed")
    event = events[0]
    payload = _event_payload(event, f"{cell_id} model-context reset")
    raw_descriptor = payload.get("artifact")
    require(
        raw_descriptor == completion.get("context_reset_artifact"),
        f"{cell_id} context-reset event/completion descriptor differs",
    )
    descriptor = _verify_payload_descriptor(
        raw_descriptor,
        base=completion_path.parent,
        raw_root=raw_root,
        label=f"{cell_id} context-reset payload",
        required_role="context_reset",
    )
    recorded = _thaw_scalar(
        descriptor["structure"], f"{cell_id} context-reset structure"
    )
    require(
        isinstance(recorded, Mapping),
        f"{cell_id} recorder context-reset payload is not a mapping",
    )
    recorded = dict(recorded)
    client_state_before = recorded.pop("client_state_before", None)
    client_state_after = recorded.pop("client_state_after", None)
    require(
        recorded == dict(expected_receipt),
        f"{cell_id} recorder context-reset payload differs from server attestation",
    )
    empty_client_state = {
        "chunk_env_ids": [],
        "counter_env_ids": [],
        "session_ids": [],
    }
    require(
        client_state_before == empty_client_state
        and client_state_after == empty_client_state,
        f"{cell_id} recorder client reset state changed",
    )
    later = _event_rows(rows, "model_request_packed") + _event_rows(
        rows, "policy_action_returned"
    )
    require(
        bool(later) and event["sequence"] < min(row["sequence"] for row in later),
        f"{cell_id} context reset did not precede requests/actions",
    )
    return _payload_binding(
        descriptor,
        base=completion_path.parent,
        raw_root=raw_root,
        label=f"{cell_id} context-reset payload",
    )


def _validate_source_and_checkpoint_identity(
    *, cell: Mapping[str, Any], completion: Mapping[str, Any], model: str
) -> dict[str, Any]:
    cell_id = str(cell.get("cell_id"))
    pins = MODEL_PINS[model]
    source = cell.get("source_pins")
    require(isinstance(source, Mapping), f"{cell_id} source pins are missing")
    study_commit = source.get("study_commit")
    require(
        isinstance(study_commit, str) and COMMIT_RE.fullmatch(study_commit) is not None,
        f"{cell_id} study source commit is invalid",
    )
    if model == "N3":
        expected_source = {
            "study_commit": study_commit,
            "robolab_commit": pins["robolab_commit"],
            "cosmos_commit": pins["cosmos_commit"],
        }
        source_fragment = f"cosmos:{pins['cosmos_commit']}"
    else:
        expected_source = {
            "study_commit": study_commit,
            "robolab_commit": pins["robolab_commit"],
            "dreamzero_commit": pins["dreamzero_commit"],
            "dreamzero_tree": pins["dreamzero_tree"],
        }
        source_fragment = f"dreamzero:{pins['dreamzero_commit']}"
    require(dict(source) == expected_source, f"{cell_id} model source pins changed")
    checkpoint = {
        "revision": pins["checkpoint_revision"],
        "aggregate_sha256": pins["checkpoint_aggregate_sha256"],
    }
    require(cell.get("checkpoint_pin") == checkpoint, f"{cell_id} checkpoint pin changed")
    identity = completion.get("identity")
    require(isinstance(identity, Mapping), f"{cell_id} completion identity is missing")
    source_identity = identity.get("source_identity")
    require(
        isinstance(source_identity, str) and source_identity.count(";pose:") == 1,
        f"{cell_id} source identity is invalid",
    )
    pose_sha = source_identity.rsplit(";pose:", 1)[1]
    require(_valid_sha(pose_sha), f"{cell_id} pose identity is invalid")
    expected_identity = (
        f"study:{study_commit};robolab:{pins['robolab_commit']};"
        f"{source_fragment};pose:{pose_sha}"
    )
    require(source_identity == expected_identity, f"{cell_id} completion source identity changed")
    require(
        identity.get("checkpoint_identity")
        == f"revision:{pins['checkpoint_revision']};aggregate:{pins['checkpoint_aggregate_sha256']}",
        f"{cell_id} completion checkpoint identity changed",
    )
    return {
        "source_pins": dict(source),
        "checkpoint_pin": checkpoint,
        "source_identity": source_identity,
        "checkpoint_identity": identity["checkpoint_identity"],
        "pose_manifest_sha256": pose_sha,
    }


def _validate_n3_context_chain(
    *,
    cell: Mapping[str, Any],
    completion: Mapping[str, Any],
    completion_path: Path,
    rows: Sequence[Mapping[str, Any]],
    request_values: Sequence[Mapping[str, Any]],
    raw_root: Path,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    begin = cell.get("server_begin_receipt")
    require(isinstance(begin, Mapping), f"{cell_id} N3 begin attestation is missing")
    condition_index = cell.get("condition_index")
    effective_seed = cell.get("effective_seed")
    block_id = cell.get("block_id")
    require(
        block_id == f"wmf_ablation_001_20260912__development__{cell.get('layout_pair_id')}__N3",
        f"{cell_id} N3 development block identity changed",
    )
    require(type(condition_index) is int and condition_index >= 0,
            f"{cell_id} N3 condition index is invalid")
    require(type(effective_seed) is int, f"{cell_id} N3 effective seed is invalid")
    context_id = begin.get("server_context_id")
    require(isinstance(context_id, str) and context_id, f"{cell_id} N3 context ID is missing")
    require(
        begin.get("passed") is True
        and begin.get("reset_scope") == CONTEXT_RESET_SCOPE
        and begin.get("cell_id") == cell_id
        and begin.get("condition_index") == condition_index
        and begin.get("effective_seed") == effective_seed,
        f"{cell_id} N3 begin/reset identity changed",
    )
    reset = begin.get("cache_reset_evidence")
    require(isinstance(reset, Mapping), f"{cell_id} N3 reset evidence is missing")
    require(
        reset.get("passed") is True
        and reset.get("episode_context_id") == context_id
        and reset.get("exclusive_active_episode") == cell_id
        and reset.get("wrapper_request_index_reset_to_zero") is True
        and reset.get("official_history_length") == 1
        and reset.get("request_bound_seed") == effective_seed
        and reset.get("model_process_reused_but_episode_state_not_reused") is True
        and reset.get("unresolved_mutable_temporal_fields") == [],
        f"{cell_id} N3 temporal/cache reset attestation changed",
    )
    end = cell.get("server_end_receipt")
    limits = MODEL_LIMITS["N3"]
    require(
        isinstance(end, Mapping)
        and end.get("passed") is True
        and end.get("status") == "completed"
        and end.get("cell_id") == cell_id
        and end.get("condition_index") == condition_index
        and end.get("server_context_id") == context_id
        and end.get("server_request_count") == limits["request_count"]
        and end.get("client_request_count") == limits["request_count"]
        and end.get("actions_executed") == limits["action_cap"],
        f"{cell_id} N3 end attestation changed",
    )
    require(
        all(
            request.get("server_context_id") == context_id
            and request.get("block_id") == block_id
            and request.get("condition_index") == condition_index
            and request.get("sampling_seed") == effective_seed
            for request in request_values
        ),
        f"{cell_id} N3 request context differs from begin/end attestations",
    )
    context_payload = _validate_context_payload(
        rows=rows,
        completion=completion,
        completion_path=completion_path,
        raw_root=raw_root,
        expected_receipt=begin,
        cell_id=cell_id,
    )
    return {
        "server_context_id": context_id,
        "server_begin_receipt": dict(begin),
        "server_end_receipt": dict(end),
        "recorder_context_reset": context_payload,
    }


def _validate_d1_temporal_reset(
    reset: Mapping[str, Any], *, expected_control: Mapping[str, Any], label: str
) -> dict[str, Any]:
    require(
        reset.get("schema_version") == D1_RESET_SCHEMA
        and reset.get("status") == "passed"
        and reset.get("world_size") == 2
        and reset.get("control") == dict(expected_control),
        f"{label} reset header/control changed",
    )
    reset_id = reset.get("reset_id")
    require(isinstance(reset_id, str) and reset_id, f"{label} reset ID is invalid")
    ranks = reset.get("rank_receipts")
    require(
        isinstance(ranks, list)
        and len(ranks) == 2
        and all(isinstance(row, Mapping) for row in ranks)
        and [row.get("rank") for row in ranks] == [0, 1],
        f"{label} reset rank inventory changed",
    )
    scan: list[dict[str, Any]] = []
    for rank, row in enumerate(ranks):
        require(
            row.get("schema_version") == D1_RESET_SCHEMA
            and row.get("status") == "passed"
            and row.get("rank") == rank
            and row.get("reset_id") == reset_id
            and row.get("failures") == [],
            f"{label} rank {rank} reset failed or changed",
        )
        before = row.get("before")
        after = row.get("after")
        require(isinstance(before, Mapping) and isinstance(after, Mapping),
                f"{label} rank {rank} temporal scan is missing")
        fields = after.get("fields")
        require(
            after.get("current_start_frame") == 0
            and isinstance(fields, Mapping)
            and all(
                isinstance(fields.get(field), Mapping)
                and fields[field].get("is_none") is True
                for field in D1_RESET_FIELDS_TO_NONE
            ),
            f"{label} rank {rank} temporal fields were not cleared",
        )
        if rank == 0:
            wrapper = row.get("wrapper_after")
            lengths = wrapper.get("frame_buffer_lengths") if isinstance(wrapper, Mapping) else None
            require(
                isinstance(wrapper, Mapping)
                and isinstance(lengths, Mapping)
                and all(value == 0 for value in lengths.values())
                and wrapper.get("call_count") == 0
                and wrapper.get("is_first_call") is True
                and wrapper.get("video_across_time_count") == 0
                and wrapper.get("current_session_id") is None,
                f"{label} rank-0 wrapper state was not cleared",
            )
        scan.append({
            "rank": rank,
            "before": before,
            "after": after,
            "fields_cleared": row.get("fields_cleared"),
        })
    return {
        "passed": True,
        "reset_id": reset_id,
        "world_size": 2,
        "rank_temporal_state_scan": scan,
        "unresolved_mutable_temporal_fields": [],
    }


def _validate_d1_runtime_identity(value: Mapping[str, Any], label: str) -> None:
    pins = MODEL_PINS["D1"]
    source = value.get("source")
    checkpoint = value.get("checkpoint")
    tokenizer = value.get("tokenizer")
    require(value.get("status") == "passed", f"{label} did not pass")
    require(
        isinstance(source, Mapping)
        and source.get("commit") == pins["dreamzero_commit"]
        and source.get("git_tree") == pins["dreamzero_tree"]
        and source.get("aggregate_sha256") == pins["dreamzero_aggregate_sha256"],
        f"{label} DreamZero source identity changed",
    )
    require(
        isinstance(checkpoint, Mapping)
        and checkpoint.get("revision") == pins["checkpoint_revision"]
        and checkpoint.get("aggregate_sha256") == pins["checkpoint_aggregate_sha256"],
        f"{label} checkpoint identity changed",
    )
    require(
        isinstance(tokenizer, Mapping)
        and tokenizer.get("revision") == pins["tokenizer_revision"]
        and tokenizer.get("aggregate_sha256") == pins["tokenizer_aggregate_sha256"],
        f"{label} tokenizer identity changed",
    )


def _validate_d1_context_chain(
    *,
    cell: Mapping[str, Any],
    completion: Mapping[str, Any],
    completion_path: Path,
    rows: Sequence[Mapping[str, Any]],
    request_values: Sequence[Mapping[str, Any]],
    request_descriptors: Sequence[Mapping[str, Any]],
    cell_path: Path,
    raw_root: Path,
) -> dict[str, Any]:
    cell_id = str(cell["cell_id"])
    limits = MODEL_LIMITS["D1"]
    block_id = cell.get("block_id")
    condition_index = cell.get("condition_index")
    contract_sha = cell.get("development_contract_sha256")
    source_pins = cell["source_pins"]
    require(isinstance(block_id, str) and block_id, f"{cell_id} D1 block ID is invalid")
    require(
        block_id == f"wmf_ablation_001_20260912__development__{cell.get('layout_pair_id')}__D1",
        f"{cell_id} D1 development block identity changed",
    )
    require(type(condition_index) is int and condition_index >= 0,
            f"{cell_id} D1 condition index is invalid")
    require(_valid_sha(contract_sha), f"{cell_id} D1 development contract hash is invalid")

    ready_descriptor, ready_path = _descriptor(
        cell.get("server_ready"), base=cell_path.parent, raw_root=raw_root,
        label=f"{cell_id} D1 server ready",
    )
    ready = load_json(ready_path, f"{cell_id} D1 server ready")
    future_raw = ready.get("future_root")
    require(isinstance(future_raw, str) and Path(future_raw).is_absolute(),
            f"{cell_id} D1 future root is invalid")
    future_root = _under(Path(future_raw), raw_root, f"{cell_id} D1 future root")
    require(future_root.is_dir(), f"{cell_id} D1 future root is missing")
    require(
        ready.get("schema_version") == D1_READY_SCHEMA
        and ready.get("status") == "ready"
        and ready.get("study_commit") == source_pins["study_commit"]
        and ready.get("study_id") == STUDY_ID
        and ready.get("block_id") == block_id
        and ready.get("model_config") == "D1"
        and ready.get("service_host") == "wmf-forecast-0912-d1"
        and ready.get("service_port") == 18021
        and ready.get("pilot_contract_sha256") == contract_sha
        and ready.get("returned_action_shape") == [limits["returned_actions"], 8]
        and ready.get("executed_prefix_horizon") == limits["executed_prefix"]
        and ready.get("effective_model_noise_seed") == 1140
        and ready.get("global_state_noninterleaving") is True,
        f"{cell_id} D1 server-ready contract changed",
    )
    for identity_key in ("run_id", "server_job_id", "paired_simulator_job_id"):
        identity_value = ready.get(identity_key)
        require(
            isinstance(identity_value, str)
            and SAFE_ID_RE.fullmatch(identity_value) is not None,
            f"{cell_id} D1 server-ready {identity_key} is invalid",
        )
    expected_cell_ids = ready.get("expected_cell_ids")
    condition_order = DEVELOPMENT_CONDITION_ORDERS.get(str(cell.get("layout_pair_id")))
    require(
        isinstance(condition_order, tuple)
        and len(condition_order) == len(CONDITIONS),
        f"{cell_id} D1 frozen condition order is missing",
    )
    authoritative_cell_ids = [
        f"wmf1__development__{cell.get('layout_pair_id')}__D1__{condition.replace('-', '__')}"
        for condition in condition_order
    ]
    require(
        expected_cell_ids == authoritative_cell_ids,
        f"{cell_id} D1 server-ready cohort changed",
    )

    runtime_descriptor, runtime_path = _descriptor(
        ready.get("runtime_identity"), base=ready_path.parent, raw_root=raw_root,
        label=f"{cell_id} D1 runtime identity",
    )
    runtime_identity = load_json(runtime_path, f"{cell_id} D1 runtime identity")
    _validate_d1_runtime_identity(runtime_identity, f"{cell_id} D1 runtime identity")
    contract_descriptor, contract_path = _descriptor(
        ready.get("server_contract"), base=ready_path.parent, raw_root=raw_root,
        label=f"{cell_id} D1 server contract",
    )
    require(
        ready.get("server_contract_sha256") == contract_descriptor["sha256"],
        f"{cell_id} D1 ready/contract hash binding changed",
    )
    contract = load_json(contract_path, f"{cell_id} D1 server contract")
    pins = MODEL_PINS["D1"]
    require(
        contract.get("schema_version") == D1_SERVER_CONTRACT_SCHEMA
        and contract.get("status") == "passed"
        and contract.get("configuration_id") == "D1"
        and contract.get("official_repository_commit") == pins["dreamzero_commit"]
        and contract.get("official_repository_tree") == pins["dreamzero_tree"]
        and contract.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal"
        and contract.get("custom_s2_used") is False
        and contract.get("patched_s1_used") is False
        and contract.get("world_size") == 2
        and contract.get("port") == 18021
        and contract.get("future_root") == str(future_root)
        and contract.get("returned_action_shape") == [limits["returned_actions"], 8]
        and contract.get("executed_action_prefix") == limits["executed_prefix"]
        and contract.get("effective_official_model_noise_seed") == 1140
        and contract.get("noise_semantics") == "fixed; no request is an independent noise draw"
        and contract.get("dynamic_cache_schedule") is False
        and contract.get("tensorrt_engine_active") is False,
        f"{cell_id} official D1 server contract changed",
    )
    overlay = contract.get("instrumentation_overlay")
    require(
        isinstance(overlay, Mapping) and overlay.get("returned_action_modified") is False,
        f"{cell_id} D1 instrumentation modified returned actions",
    )
    topology = contract.get("topology")
    heads = contract.get("head_contracts")
    loaders = contract.get("bounded_loader_receipts")
    require(
        isinstance(topology, list)
        and sorted(row.get("rank") for row in topology if isinstance(row, Mapping)) == [0, 1]
        and len({row.get("cuda_device_index") for row in topology if isinstance(row, Mapping)}) == 2
        and all(row.get("cuda_device_name") == "NVIDIA B200" for row in topology),
        f"{cell_id} D1 topology changed",
    )
    require(
        isinstance(heads, list)
        and sorted(row.get("rank") for row in heads if isinstance(row, Mapping)) == [0, 1]
        and all(row.get("status") == "passed" for row in heads),
        f"{cell_id} D1 head contract changed",
    )
    require(
        isinstance(loaders, list)
        and sorted(row.get("rank") for row in loaders if isinstance(row, Mapping)) == [0, 1]
        and all(
            isinstance(row.get("receipt"), Mapping)
            and row["receipt"].get("passed") is True
            and row["receipt"].get("forward_path_modified") is False
            for row in loaders
        ),
        f"{cell_id} D1 bounded loader evidence changed",
    )
    require(
        contract.get("identity_receipt") == str(runtime_path)
        and contract.get("identity_receipt_sha256") == runtime_descriptor["sha256"],
        f"{cell_id} D1 contract/runtime identity binding changed",
    )

    claim_descriptor, claim_path = _descriptor(
        cell.get("simulator_claim"), base=cell_path.parent, raw_root=raw_root,
        label=f"{cell_id} D1 simulator claim",
    )
    claim = load_json(claim_path, f"{cell_id} D1 simulator claim")
    lease_token = claim.get("lease_token")
    require(
        claim.get("schema_version") == D1_CLAIM_SCHEMA
        and claim.get("status") == "claimed"
        and claim.get("run_id") == ready.get("run_id")
        and claim.get("simulator_job_id") == ready.get("paired_simulator_job_id")
        and claim.get("server_job_id") == ready.get("server_job_id")
        and claim.get("server_ready_sha256") == ready_descriptor["sha256"]
        and claim.get("study_commit") == source_pins["study_commit"]
        and claim.get("block_id") == block_id
        and claim.get("pilot_contract_sha256") == contract_sha
        and claim.get("worker_role") in D1_ALLOWED_SIMULATOR_ROLES
        and isinstance(lease_token, str)
        and SAFE_ID_RE.fullmatch(lease_token) is not None
        and type(claim.get("start_cell_index")) is int
        and 0 <= claim["start_cell_index"] < len(CONDITIONS),
        f"{cell_id} D1 simulator claim changed",
    )

    begin = cell.get("server_begin_receipt")
    require(isinstance(begin, Mapping), f"{cell_id} D1 begin/reset attestation is missing")
    episode_id = begin.get("episode_context_id")
    session_id = begin.get("client_session_id")
    require(
        isinstance(episode_id, str)
        and SAFE_ID_RE.fullmatch(episode_id) is not None
        and isinstance(session_id, str)
        and SAFE_ID_RE.fullmatch(session_id) is not None
        and session_id != episode_id,
        f"{cell_id} D1 episode/session identity is invalid",
    )
    require(
        begin.get("passed") is True
        and begin.get("reset_scope") == CONTEXT_RESET_SCOPE
        and begin.get("server_context_id") == episode_id
        and begin.get("service_route_proved_by_reset_artifact") is True
        and begin.get("service_host") == "wmf-forecast-0912-d1"
        and begin.get("service_port") == 18021
        and begin.get("server_ready_sha256") == ready_descriptor["sha256"]
        and begin.get("simulator_claim_sha256") == claim_descriptor["sha256"],
        f"{cell_id} D1 begin/reset binding changed",
    )
    expected_control = {
        "episode_id": episode_id,
        "expected_session_id": session_id,
        "purpose": "d1_behavioral_development",
        "study_id": STUDY_ID,
        "block_id": block_id,
        "cell_id": cell_id,
        "condition_index": condition_index,
        "layout_arm": cell.get("layout_arm"),
        "command": cell.get("command"),
        "server_ready_sha256": ready_descriptor["sha256"],
        "simulator_claim_sha256": claim_descriptor["sha256"],
        "simulator_lease_token": lease_token,
        "pilot_contract_sha256": contract_sha,
    }
    reset_descriptor, reset_path = _descriptor(
        cell.get("server_reset_receipt"), base=cell_path.parent, raw_root=raw_root,
        label=f"{cell_id} D1 reset receipt",
    )
    require(
        reset_path == future_root / "episodes" / episode_id / "reset_receipt.json",
        f"{cell_id} D1 reset path changed",
    )
    reset = load_json(reset_path, f"{cell_id} D1 reset receipt")
    reset_scan = _validate_d1_temporal_reset(
        reset, expected_control=expected_control, label=f"{cell_id} D1"
    )
    require(cell.get("server_temporal_reset_scan") == reset_scan,
            f"{cell_id} D1 temporal-reset scan changed")
    require(begin.get("server_reset_receipt") == reset_descriptor,
            f"{cell_id} D1 begin/reset descriptor differs")
    require(begin.get("temporal_state_scan") == reset_scan,
            f"{cell_id} D1 begin/reset scan differs")
    expected_cache = {
        "source": "validated_official_d1_two_rank_reset_receipt",
        "server_reset_receipt": reset_descriptor,
        "reset_id": reset["reset_id"],
        "world_size": 2,
        "rank_temporal_state_scan": reset_scan["rank_temporal_state_scan"],
        "unresolved_mutable_temporal_fields": [],
    }
    require(begin.get("cache_reset_evidence") == expected_cache,
            f"{cell_id} D1 cache-reset attestation changed")
    context_payload = _validate_context_payload(
        rows=rows,
        completion=completion,
        completion_path=completion_path,
        raw_root=raw_root,
        expected_receipt=begin,
        cell_id=cell_id,
    )

    manifest_descriptor, manifest_path = _descriptor(
        cell.get("server_episode_manifest"), base=cell_path.parent, raw_root=raw_root,
        label=f"{cell_id} D1 episode manifest",
    )
    require(
        manifest_path == future_root / "episodes" / episode_id / "episode_manifest.json",
        f"{cell_id} D1 episode-manifest path changed",
    )
    manifest = load_json(manifest_path, f"{cell_id} D1 episode manifest")
    require(
        manifest.get("schema_version") == D1_EPISODE_SCHEMA
        and manifest.get("configuration_id") == "D1"
        and manifest.get("episode_id") == episode_id
        and manifest.get("status") == "complete"
        and manifest.get("official_repository_commit") == pins["dreamzero_commit"]
        and manifest.get("official_action_path") == "GrootSimPolicy.lazy_joint_forward_causal"
        and manifest.get("custom_s2_used") is False
        and manifest.get("patched_s1_used") is False
        and manifest.get("effective_official_model_noise_seed") == 1140
        and manifest.get("noise_semantics") == "fixed; not an independent draw"
        and manifest.get("request_count") == limits["request_count"]
        and manifest.get("server_contract_sha256") == contract_descriptor["sha256"]
        and manifest.get("two_rank_reset") == reset,
        f"{cell_id} D1 episode manifest changed",
    )
    embedded = manifest.get("requests")
    require(
        isinstance(embedded, list) and embedded == list(request_values),
        f"{cell_id} D1 embedded request inventory changed",
    )
    for index, (request, descriptor) in enumerate(zip(request_values, request_descriptors)):
        require(
            request.get("episode_id") == episode_id
            and request.get("session_id") == session_id
            and request.get("prompt") == cell.get("prompt")
            and request.get("measurement_control", {}).get("probe_plan_sha256") == contract_sha,
            f"{cell_id} D1 request {index} episode/session/contract binding changed",
        )
        require(
            Path(descriptor["path"])
            == future_root / "episodes" / episode_id / f"request_{index:04d}" / "request_receipt.json",
            f"{cell_id} D1 request {index} path changed",
        )
    return {
        "server_ready": ready_descriptor,
        "runtime_identity": runtime_descriptor,
        "server_contract": contract_descriptor,
        "simulator_claim": claim_descriptor,
        "server_context_id": episode_id,
        "server_reset_receipt": reset_descriptor,
        "server_episode_manifest": manifest_descriptor,
        "recorder_context_reset": context_payload,
    }


def _observation_and_action_evidence(
    *,
    rows: Sequence[Mapping[str, Any]],
    completion_path: Path,
    completion: Mapping[str, Any],
    cell: Mapping[str, Any],
    model: str,
    camera_id: str,
    raw_root: Path,
    executable_action_hashes: Sequence[Sequence[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    limits = MODEL_LIMITS[model]
    observations = _event_rows(rows, "observation_captured")
    proposals = _event_rows(rows, "policy_action_returned")
    starts = _event_rows(rows, "environment_step_started")
    finishes = _event_rows(rows, "environment_step_completed")
    require(len(observations) == limits["observation_count"],
            f"{cell['cell_id']} observation event count changed")
    require(len(proposals) == len(starts) == len(finishes) == limits["action_cap"],
            f"{cell['cell_id']} action event count changed")

    observation_evidence: list[dict[str, Any]] = []
    observation_by_step: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    prior_physics: int | None = None
    prior_camera: Any = None
    for expected_step, row in enumerate(observations):
        payload = _event_payload(row, f"observation {expected_step}")
        require(payload.get("control_step") == expected_step
                and payload.get("observation_id") == f"obs_{expected_step:06d}",
                f"{cell['cell_id']} observation schedule changed")
        clock = payload.get("clock")
        require(isinstance(clock, Mapping) and clock.get("control_step") == expected_step,
                f"{cell['cell_id']} observation clock changed")
        physics_step = clock.get("physics_step")
        physics_time = clock.get("physics_time_s")
        cameras = clock.get("cameras")
        camera = cameras.get(camera_id) if isinstance(cameras, Mapping) else None
        require(type(physics_step) is int and physics_step >= 0,
                f"{cell['cell_id']} physics-step identity is missing")
        require(type(physics_time) in (int, float) and math.isfinite(float(physics_time)),
                f"{cell['cell_id']} physics time is missing or non-finite")
        require(isinstance(camera, Mapping)
                and ((type(camera.get("frame_id")) is int and camera["frame_id"] >= 0)
                     or (isinstance(camera.get("frame_id"), str) and camera["frame_id"]))
                and type(camera.get("capture_time_ns")) is int
                and camera["capture_time_ns"] >= 0
                and isinstance(camera.get("timestamp_source"), str)
                and camera["timestamp_source"],
                f"{cell['cell_id']} original-camera identity is missing")
        if prior_physics is not None:
            require(physics_step > prior_physics,
                    f"{cell['cell_id']} physics-step identity did not advance")
            require(camera["frame_id"] != prior_camera,
                    f"{cell['cell_id']} original-camera frame repeated")
        prior_physics = physics_step
        prior_camera = camera["frame_id"]
        artifact = _verify_payload_descriptor(
            payload.get("artifact"),
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} observation {expected_step}",
            required_role="observation",
        )
        stored_clock = _thaw_scalar(
            _mapping_item(artifact["structure"], "clock", "observation payload"),
            "stored observation clock",
        )
        require(stored_clock == dict(clock),
                f"{cell['cell_id']} observation payload clock changed")
        images = _mapping_item(artifact["structure"], "image_obs", "observation payload")
        camera_node = _mapping_item(images, camera_id, "stored original cameras")
        require(isinstance(camera_node, Mapping)
                and camera_node.get("__type__") == "ndarray"
                and isinstance(camera_node.get("shape"), list)
                and len(camera_node["shape"]) == 3
                and camera_node["shape"][-1] in (3, 4),
                f"{cell['cell_id']} original camera pixels are unavailable")
        pixel_artifact, _ = _descriptor(
            artifact["artifact"],
            base=completion_path.parent,
            raw_root=raw_root,
            label=f"{cell['cell_id']} observation {expected_step} pixels",
        )
        compact = {
            "observation_id": payload["observation_id"],
            "control_step": expected_step,
            "physics_step": physics_step,
            "physics_time_s": float(physics_time),
            "camera_frame_native_id": camera["frame_id"],
            "camera_frame_id": (
                f"{camera_id}:native-int:{camera['frame_id']}"
                if type(camera["frame_id"]) is int
                else f"{camera_id}:native-str:{camera['frame_id']}"
            ),
            "camera_capture_time_ns": camera["capture_time_ns"],
            "camera_timestamp_source": camera["timestamp_source"],
            "payload_sha256": artifact["payload_sha256"],
            "payload_artifact": pixel_artifact,
        }
        observation_evidence.append(compact)
        observation_by_step[expected_step] = (payload, compact)

    execution = completion.get("request_execution")
    require(isinstance(execution, list) and len(execution) == limits["request_count"],
            f"{cell['cell_id']} completion request inventory changed")
    expected_owner: list[tuple[int, int]] = []
    for request in execution:
        require(isinstance(request, Mapping), "request execution row is invalid")
        request_index = request.get("request_index")
        executed = request.get("executed_actions")
        require(type(request_index) is int and type(executed) is int and executed > 0,
                "request execution ownership is invalid")
        expected_owner.extend((request_index, offset) for offset in range(executed))
    require(len(expected_owner) == limits["action_cap"],
            f"{cell['cell_id']} request prefixes do not cover all actions")

    action_rows: list[dict[str, Any]] = []
    for action_index, (proposal_row, start_row, finish_row) in enumerate(
        zip(proposals, starts, finishes)
    ):
        action_step = action_index + 1
        proposal = _event_payload(proposal_row, f"action proposal {action_step}")
        start = _event_payload(start_row, f"action start {action_step}")
        finish = _event_payload(finish_row, f"action finish {action_step}")
        owner, offset = expected_owner[action_index]
        require(
            proposal.get("action_step") == start.get("action_step")
            == finish.get("action_step") == action_step
            and proposal.get("request_index") == start.get("request_index")
            == finish.get("request_index") == owner
            and proposal.get("chunk_offset") == start.get("chunk_offset")
            == finish.get("chunk_offset") == offset,
            f"{cell['cell_id']} action/request ownership changed at {action_step}",
        )
        identity = proposal.get("action_identity")
        require(isinstance(identity, Mapping)
                and identity.get("shape") == [8]
                and identity.get("order") == "C"
                and isinstance(identity.get("dtype"), str)
                and _valid_sha(identity.get("value_sha256")),
                f"{cell['cell_id']} action identity is invalid at {action_step}")
        require(start.get("executed_action_identity") == identity,
                f"{cell['cell_id']} returned/executed action bytes differ at {action_step}")
        require(
            0 <= owner < len(executable_action_hashes)
            and 0 <= offset < len(executable_action_hashes[owner])
            and identity["value_sha256"] == executable_action_hashes[owner][offset],
            f"{cell['cell_id']} action does not match retained executable chunk at {action_step}",
        )
        require(start.get("env_step_start_monotonic_ns")
                == finish.get("env_step_start_monotonic_ns")
                and type(finish.get("env_step_end_monotonic_ns")) is int
                and finish["env_step_end_monotonic_ns"]
                >= finish["env_step_start_monotonic_ns"],
                f"{cell['cell_id']} environment-step clocks changed at {action_step}")
        require(type(finish.get("terminated")) is bool
                and type(finish.get("truncated")) is bool,
                f"{cell['cell_id']} environment-step flags are invalid")
        require(
            proposal_row["sequence"] < start_row["sequence"] < finish_row["sequence"]
            < observations[action_step]["sequence"],
            f"{cell['cell_id']} action/observation event order changed at {action_step}",
        )
        target = observation_by_step[action_step][1]
        action_rows.append({
            "action_index": action_index,
            "request_index": owner,
            "executed_action_sha256": identity["value_sha256"],
            "control_timestamp": _rfc3339_ns(
                start_row.get("wall_time_ns"), f"{cell['cell_id']} action {action_step}"
            ),
            "physics_step_id": f"physics-step:{target['physics_step']}",
            "camera_frame_id": target["camera_frame_id"],
        })
    return observation_evidence, action_rows


def _compile_cell(
    *,
    cell_descriptor: Mapping[str, Any],
    cell_path: Path,
    model: str,
    planned: Mapping[str, str],
    camera_id: str,
    raw_root: Path,
) -> dict[str, Any]:
    limits = MODEL_LIMITS[model]
    cell = load_json(cell_path, f"{model} development cell")
    cell_id = planned["cell_id"]
    require(cell.get("schema_version") == limits["cell_schema"], f"{cell_id} cell schema changed")
    require(cell.get("status") == "passed" and cell.get("study_id") == STUDY_ID,
            f"{cell_id} cell did not pass")
    require(cell.get("cell_id") == cell_id and cell.get("model_config") == model,
            f"{cell_id} cell identity changed")
    require(cell.get("layout_pair_id") == planned["layout_pair_id"]
            and cell.get("layout_arm") == planned["layout_arm"]
            and cell.get("command") == planned["command"],
            f"{cell_id} planned condition binding changed")
    layout = planned["layout_pair_id"]
    environment_seed = DEVELOPMENT_ENVIRONMENT_SEEDS.get(layout)
    raw_planned_seed = planned.get("candidate_effective_policy_seed")
    require(
        isinstance(raw_planned_seed, str)
        and re.fullmatch(r"[0-9]+", raw_planned_seed) is not None,
        f"{cell_id} planned effective policy seed is invalid",
    )
    planned_seed = int(raw_planned_seed)
    expected_effective_seed = environment_seed if model == "N3" else 1140
    require(
        type(environment_seed) is int
        and planned_seed == expected_effective_seed,
        f"{cell_id} planned effective policy seed changed",
    )
    if model == "N3":
        require(
            cell.get("effective_seed") == expected_effective_seed,
            f"{cell_id} N3 effective seed changed",
        )
    else:
        require(
            cell.get("effective_model_noise_seed") == expected_effective_seed
            and cell.get("environment_seed") == environment_seed
            and cell.get("noise_semantics")
            == "fixed; this cell is not an independent noise draw",
            f"{cell_id} D1 model/environment seed contract changed",
        )
    require(
        cell.get("actions_executed") == limits["action_cap"]
        and cell.get("observation_count") == limits["observation_count"]
        and cell.get("behavioral_model_request_count") == limits["request_count"]
        and cell.get("behavioral_episode_count") == 1
        and cell.get("generation_qualification_request_count") == 0,
        f"{cell_id} behavioral counts changed",
    )

    completion_descriptor, completion_path = _descriptor(
        cell.get("adapter_completion"), base=cell_path.parent,
        label=f"{cell_id} adapter completion", raw_root=raw_root,
    )
    completion = load_json(completion_path, f"{cell_id} adapter completion")
    journal_descriptor, journal_path = _descriptor(
        cell.get("adapter_journal"), base=cell_path.parent,
        label=f"{cell_id} adapter journal", raw_root=raw_root,
    )
    rows, tail = _verify_journal(journal_path)
    require(completion.get("event_count") == len(rows)
            and completion.get("journal_tail_sha256") == tail,
            f"{cell_id} completion does not bind the journal")
    require(cell.get("adapter_journal", {}).get("event_count") == len(rows)
            and cell.get("adapter_journal", {}).get("tail_sha256") == tail,
            f"{cell_id} cell does not bind the full journal")
    journal_descriptor.update(event_count=len(rows), tail_sha256=tail)
    require(
        rows[0].get("kind") == "attempt_started"
        and rows[-1].get("kind") == "attempt_finalized"
        and len(_event_rows(rows, "attempt_started")) == 1
        and len(_event_rows(rows, "attempt_finalized")) == 1,
        f"{cell_id} journal attempt boundary changed",
    )
    require(
        completion.get("event_count_before_final") == len(rows) - 1
        and completion.get("journal_tail_sha256_before_final")
        == (rows[-2].get("event_sha256") if len(rows) > 1 else None),
        f"{cell_id} final receipt does not bind its pre-final journal",
    )
    identity = completion.get("identity")
    require(
        isinstance(identity, Mapping)
        and identity.get("cell_id") == cell_id
        and identity.get("stage") == "development"
        and identity.get("model_config") == model
        and identity.get("layout_pair_id") == planned["layout_pair_id"]
        and identity.get("layout_arm") == planned["layout_arm"]
        and identity.get("command") == planned["command"]
        and identity.get("prompt") == cell.get("prompt"),
        f"{cell_id} completion identity changed",
    )
    require(
        identity.get("effective_seed") == expected_effective_seed,
        f"{cell_id} completion effective seed changed",
    )
    source_and_checkpoint = _validate_source_and_checkpoint_identity(
        cell=cell, completion=completion, model=model
    )
    finalized_payload = _event_payload(rows[-1], f"{cell_id} finalized event")
    for key in (
        "schema_version", "study_id", "identity", "stop_reason",
        "behavioral_result_valid", "technical_invalid", "right_censored",
        "actions_executed", "observation_count", "request_count",
        "request_execution", "validation_errors",
    ):
        require(
            finalized_payload.get(key) == completion.get(key),
            f"{cell_id} finalized-event/completion field differs: {key}",
        )
    viewport_descriptor, viewport_path = _descriptor(
        cell.get("viewport_video"), base=cell_path.parent,
        label=f"{cell_id} viewport video", raw_root=raw_root,
    )
    (
        request_descriptors,
        request_values,
        response_descriptors,
        executable_action_hashes,
        packed_request_bindings,
    ) = _discover_requests(
        rows=rows,
        completion_path=completion_path,
        completion=completion,
        cell=cell,
        model=model,
        raw_root=raw_root,
    )
    if model == "N3":
        model_chain = _validate_n3_context_chain(
            cell=cell,
            completion=completion,
            completion_path=completion_path,
            rows=rows,
            request_values=request_values,
            raw_root=raw_root,
        )
    else:
        model_chain = _validate_d1_context_chain(
            cell=cell,
            completion=completion,
            completion_path=completion_path,
            rows=rows,
            request_values=request_values,
            request_descriptors=request_descriptors,
            cell_path=cell_path,
            raw_root=raw_root,
        )
    evidence_entry = {
        "cell_receipt": dict(cell_descriptor),
        "server_request_receipts": request_descriptors,
        "resource_receipt": None,
    }
    try:
        validated = freeze._validate_cell(
            evidence_entry,
            model=model,
            expected_cell_id=cell_id,
            evidence_base=cell_path.parent,
            require_resource=False,
        )
    except Exception as error:
        raise CompilerError(f"{cell_id} failed release-validator replay: {error}") from error
    observation_evidence, action_rows = _observation_and_action_evidence(
        rows=rows,
        completion_path=completion_path,
        completion=completion,
        cell=cell,
        model=model,
        camera_id=camera_id,
        raw_root=raw_root,
        executable_action_hashes=executable_action_hashes,
    )
    attempt_id = identity.get("attempt_id")
    require(isinstance(attempt_id, str) and attempt_id, f"{cell_id} attempt identity is missing")
    condition_id = f"{planned['layout_arm']}_{planned['command']}"
    # The ID denotes this episode's source-video role, while the full adjacent
    # SHA authenticates the MP4 bytes.  Deriving the ID from the unique cell
    # receipt avoids aliasing two byte-identical videos from distinct episodes.
    source_video_id = "video_" + cell_descriptor["sha256"][:24]
    action_manifest = sign_document({
        "schema_version": ACTION_MANIFEST_SCHEMA,
        "study_id": STUDY_ID,
        "cell_id": cell_id,
        "recording_id": attempt_id,
        "model_id": model,
        "executed_action_count": limits["action_cap"],
        "actions": action_rows,
    })
    recording_receipt = sign_document({
        "schema_version": RECORDING_RECEIPT_SCHEMA,
        "study_id": STUDY_ID,
        "receipt_id": "recording_" + cell_descriptor["sha256"][:24],
        "stage": "development",
        "cell_id": cell_id,
        "recording_id": attempt_id,
        "model_id": model,
        "layout_pair_id": planned["layout_pair_id"],
        "condition_id": condition_id,
        "recording_status": "valid_complete",
        "executed_action_count": limits["action_cap"],
        "censor_reason": None,
        "source_video_id": source_video_id,
        "source_video_sha256": viewport_descriptor["sha256"],
        "action_manifest_path": "action_manifest.json",
        # Filled after the deterministic action-manifest bytes are known.
        "action_manifest_sha256": None,
    })
    requests = []
    execution = completion["request_execution"]
    observations_by_id = {row["observation_id"]: row for row in observation_evidence}
    for request_index, (request_descriptor, response_descriptor, packed_binding) in enumerate(
        zip(request_descriptors, response_descriptors, packed_request_bindings)
    ):
        executed = execution[request_index]
        current_id = executed.get("current_observation_id")
        preceding_id = executed.get("preceding_observation_id")
        require(current_id in observations_by_id, f"{cell_id} request current observation is missing")
        require(preceding_id is None or preceding_id in observations_by_id,
                f"{cell_id} request preceding observation is missing")
        requests.append({
            "source_request_id": "request_" + request_descriptor["sha256"][:32],
            "cell_id": cell_id,
            "recording_id": attempt_id,
            "model_id": model,
            "layout_pair_id": planned["layout_pair_id"],
            "condition_id": condition_id,
            "request_index": request_index,
            "action_step_start": executed["action_step_start"],
            "executed_prefix_actions": executed["executed_actions"],
            "current_observation_id": current_id,
            "preceding_observation_id": preceding_id,
            "history_mode": (
                "preceding_observation" if preceding_id is not None
                else "persistence_at_initial_request"
            ),
            "current_observation": observations_by_id[current_id],
            "preceding_observation": (
                None if preceding_id is None else observations_by_id[preceding_id]
            ),
            "official_request_receipt": request_descriptor,
            "recorder_model_request": packed_binding,
            "recorder_response_payload_sha256": response_descriptor["payload_sha256"],
            "adapter_completion": completion_descriptor,
            "adapter_journal": journal_descriptor,
            "source_video_id": source_video_id,
            "source_video": viewport_descriptor,
            "model_output_or_action_modified": False,
            "model_identity": source_and_checkpoint,
            "model_context": model_chain,
        })
    require([row["request_index"] for row in requests] == list(range(limits["request_count"])),
            f"{cell_id} compiled request order changed")
    require(validated["request_receipt_sha256s"]
            == [row["official_request_receipt"]["sha256"] for row in requests],
            f"{cell_id} release/compiler request order differs")
    return {
        "cell": cell,
        "cell_descriptor": dict(cell_descriptor),
        "completion_descriptor": completion_descriptor,
        "journal_descriptor": journal_descriptor,
        "viewport_descriptor": viewport_descriptor,
        "viewport_path": viewport_path,
        "request_descriptors": request_descriptors,
        "action_manifest": action_manifest,
        "recording_receipt": recording_receipt,
        "requests": requests,
        "condition_id": condition_id,
        "recording_id": attempt_id,
        "source_video_id": source_video_id,
        "model_identity": source_and_checkpoint,
        "model_context": model_chain,
    }


def _verify_block_context_uniqueness(
    compiled: Sequence[Mapping[str, Any]], *, model: str, layout: str
) -> None:
    """Replay the per-block context non-reuse gates from each producer."""

    if model == "N3":
        context_ids = [item["model_context"]["server_context_id"] for item in compiled]
        require(
            len(context_ids) == len(set(context_ids)),
            f"{model} {layout} server context ID was reused",
        )
        return
    episode_ids = [
        item["cell"]["server_begin_receipt"].get("episode_context_id")
        for item in compiled
    ]
    session_ids = [
        item["cell"]["server_begin_receipt"].get("client_session_id")
        for item in compiled
    ]
    require(
        len(episode_ids) == len(set(episode_ids))
        and len(session_ids) == len(set(session_ids))
        and not set(episode_ids).intersection(session_ids),
        f"{model} {layout} episode/session context identity was reused or aliased",
    )


def _validate_aggregate(
    *,
    descriptor: Mapping[str, Any],
    path: Path,
    model: str,
    layout: str,
    expected_cells: Mapping[str, Mapping[str, str]],
    camera_id: str,
    raw_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = load_json(path, f"{model} {layout} aggregate")
    limits = MODEL_LIMITS[model]
    exact = {
        "schema_version": limits["aggregate_schema"],
        "status": "passed",
        "exit_code": 0,
        "study_id": STUDY_ID,
        "phase": "development",
        "layout_pair_id": layout,
        "model_config": model,
    }
    for key, wanted in exact.items():
        require(receipt.get(key) == wanted, f"{model} {layout} aggregate {key} changed")
    # The two production launchers intentionally use different names for the
    # top-level study-source pin.  N3's single-process aggregate predates the
    # paired D1 launcher and records ``source_commit``; the paired D1 aggregate
    # records the same semantic identity as ``study_commit``.  Do not accept a
    # fallback field: bind the exact model-specific signed aggregate contract.
    aggregate_commit_field = "study_commit" if model == "D1" else "source_commit"
    source_commit = receipt.get(aggregate_commit_field)
    require(
        isinstance(source_commit, str) and COMMIT_RE.fullmatch(source_commit) is not None,
        f"{model} {layout} aggregate {aggregate_commit_field} is invalid",
    )
    counts = receipt.get("counts")
    expected_count_values = {
        "planned_behavioral_cells": len(CONDITIONS),
        "launched_behavioral_cells": len(CONDITIONS),
        "completed_valid_behavioral_cells": len(CONDITIONS),
        "technically_invalid_behavioral_cells": 0,
        "right_censored_behavioral_cells": 0,
        "unrun_behavioral_cells": 0,
        "actual_behavioral_actions": len(CONDITIONS) * limits["action_cap"],
        "actual_behavioral_model_requests": len(CONDITIONS) * limits["request_count"],
        "new_generation_qualification_requests": 0,
    }
    require(isinstance(counts, Mapping), f"{model} {layout} aggregate counts are missing")
    for key, wanted in expected_count_values.items():
        require(counts.get(key) == wanted, f"{model} {layout} aggregate count {key} changed")
    require(receipt.get("failure") is None, f"{model} {layout} aggregate records a failure")
    if model == "D1":
        require(receipt.get("all_simulator_children_reaped") is True,
                f"{model} {layout} simulator children are not reaped")
    else:
        require(receipt.get("raw_attempt_recoverable_on_gm_pvc") is True,
                f"{model} {layout} raw attempt is not marked recoverable")
    raw_attempt = receipt.get("raw_attempt_root")
    require(isinstance(raw_attempt, str) and raw_attempt, f"{model} {layout} raw attempt is missing")
    _under(Path(raw_attempt), raw_root, f"{model} {layout} raw attempt")
    cell_ids = receipt.get("cell_ids")
    cells = receipt.get("cell_receipts")
    require(isinstance(cell_ids, list) and isinstance(cells, list)
            and len(cell_ids) == len(cells) == len(CONDITIONS),
            f"{model} {layout} aggregate cell inventory changed")
    condition_order = DEVELOPMENT_CONDITION_ORDERS.get(layout)
    require(
        isinstance(condition_order, tuple)
        and len(condition_order) == len(CONDITIONS),
        f"{model} {layout} frozen condition order is missing",
    )
    expected_ordered_ids = [
        f"wmf1__development__{layout}__{model}__{condition.replace('-', '__')}"
        for condition in condition_order
    ]
    require(
        cell_ids == expected_ordered_ids
        and set(cell_ids) == set(expected_cells),
        f"{model} {layout} aggregate cohort/order changed",
    )
    compiled: list[dict[str, Any]] = []
    for index, raw_cell_descriptor in enumerate(cells):
        cell_descriptor, cell_path = _descriptor(
            raw_cell_descriptor,
            base=path.parent,
            label=f"{model} {layout} cell {index}",
            raw_root=raw_root,
        )
        cell_id = load_json(cell_path, f"{model} {layout} cell identity").get("cell_id")
        require(cell_id == cell_ids[index], f"{model} {layout} aggregate cell order changed")
        require(cell_id in expected_cells, f"{model} {layout} aggregate has an unplanned cell")
        compiled.append(_compile_cell(
            cell_descriptor=cell_descriptor,
            cell_path=cell_path,
            model=model,
            planned=expected_cells[cell_id],
            camera_id=camera_id,
            raw_root=raw_root,
        ))
        require(
            compiled[-1]["cell"]["source_pins"]["study_commit"] == source_commit,
            f"{model} {layout} aggregate/cell study source differs",
        )
        require(
            compiled[-1]["cell"].get("condition_index") == index,
            f"{model} {layout} cell condition index/order changed",
        )
    _verify_block_context_uniqueness(compiled, model=model, layout=layout)
    return dict(descriptor), compiled


def _safe_cell_component(cell_id: str) -> str:
    require(re.fullmatch(r"[A-Za-z0-9_-]+", cell_id) is not None,
            f"unsafe cell ID: {cell_id}")
    return cell_id


def compile_manifest(
    manifest_path: Path,
    manifest_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    supplied_manifest = Path(manifest_path)
    _reject_symlink_components(supplied_manifest, "compiler input manifest")
    manifest_path = supplied_manifest.resolve()
    require(manifest_path.is_file(), "compiler input manifest is missing")
    require(_valid_sha(manifest_sha256), "compiler input manifest SHA-256 is invalid")
    require(sha256_file(manifest_path) == manifest_sha256,
            "compiler input manifest file hash changed")
    manifest = load_json(manifest_path, "compiler input manifest")
    require(set(manifest) == {
        "schema_version", "study_id", "mode", "raw_root", "camera_id",
        "planned_cells", "aggregate_receipts",
    }, "compiler input manifest fields changed")
    require(manifest.get("schema_version") == INPUT_SCHEMA, "compiler input schema changed")
    require(manifest.get("study_id") == STUDY_ID, "compiler input study changed")
    mode = manifest.get("mode")
    require(mode in {"formal_full", "diagnostic_partial"}, "compiler mode is invalid")
    camera_id = manifest.get("camera_id")
    require(camera_id in PRIMARY_CAMERA_CHOICES, "compiler camera is not an original camera")
    raw_root_raw = manifest.get("raw_root")
    require(isinstance(raw_root_raw, str) and Path(raw_root_raw).is_absolute(),
            "compiler raw root must be absolute")
    raw_root_supplied = Path(raw_root_raw)
    _reject_symlink_components(raw_root_supplied, "compiler raw root")
    raw_root = raw_root_supplied.resolve()
    require(raw_root.is_dir(), "compiler raw root is unavailable")
    output_dir = Path(output_dir)
    require(not output_dir.exists(), f"refusing to overwrite compiler output: {output_dir}")
    require(_under(output_dir.parent, raw_root_supplied, "compiler output parent").is_dir(),
            "compiler output parent is unavailable")

    planned_descriptor, planned_path = _descriptor(
        manifest.get("planned_cells"), base=manifest_path.parent, label="planned-cell CSV"
    )
    require(planned_descriptor["sha256"] == PLANNED_CELLS_SHA256,
            "planned-cell CSV is not the authoritative study file")
    planned = _load_planned_cells(planned_path)
    raw_aggregates = manifest.get("aggregate_receipts")
    require(isinstance(raw_aggregates, list), "aggregate receipt inventory must be a list")
    aggregates: dict[tuple[str, str], tuple[dict[str, Any], Path]] = {}
    for index, item in enumerate(raw_aggregates):
        require(isinstance(item, Mapping) and set(item) == {
            "model_id", "layout_pair_id", "receipt"
        }, f"aggregate input row {index} fields changed")
        model = item.get("model_id")
        layout = item.get("layout_pair_id")
        require(model in MODELS and layout in LAYOUT_IDS, f"aggregate input row {index} identity is invalid")
        key = (str(model), str(layout))
        require(key not in aggregates, f"aggregate input duplicates {model} {layout}")
        descriptor, path = _descriptor(
            item.get("receipt"), base=manifest_path.parent,
            label=f"{model} {layout} aggregate", raw_root=raw_root,
        )
        aggregates[key] = (descriptor, path)
    n3_keys = {key for key in aggregates if key[0] == "N3"}
    d1_keys = {key for key in aggregates if key[0] == "D1"}
    expected_n3 = {("N3", layout) for layout in LAYOUT_IDS}
    expected_d1 = {("D1", layout) for layout in LAYOUT_IDS}
    require(n3_keys == expected_n3, "compiler requires all four passed N3 development blocks")
    if mode == "formal_full":
        require(d1_keys == expected_d1,
                "formal compiler mode requires all four passed D1 development blocks")
    else:
        require(bool(d1_keys) and d1_keys <= expected_d1,
                "diagnostic mode requires one or more whole passed D1 blocks")

    compiled_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    aggregate_descriptors: dict[str, list[dict[str, Any]]] = {model: [] for model in MODELS}
    for model in MODELS:
        for layout in LAYOUT_IDS:
            key = (model, layout)
            if key not in aggregates:
                continue
            aggregate_descriptor, aggregate_path = aggregates[key]
            verified_descriptor, cells = _validate_aggregate(
                descriptor=aggregate_descriptor,
                path=aggregate_path,
                model=model,
                layout=layout,
                expected_cells=planned[key],
                camera_id=camera_id,
                raw_root=raw_root,
            )
            aggregate_descriptors[model].append(verified_descriptor)
            compiled_by_model[model].extend(cells)

    files: dict[str, bytes] = {}
    model_outputs: dict[str, Any] = {}
    total_requests = 0
    total_actions = 0
    all_source_request_ids: set[str] = set()
    all_recording_ids: set[str] = set()
    for model in MODELS:
        cells = sorted(compiled_by_model[model], key=lambda item: item["cell"]["cell_id"])
        requests: list[dict[str, Any]] = []
        timing_entries: list[dict[str, Any]] = []
        freeze_entries: list[dict[str, Any]] = []
        roster: list[dict[str, Any]] = []
        cell_outputs: list[dict[str, Any]] = []
        for cell in cells:
            cell_id = cell["cell"]["cell_id"]
            require(
                cell["recording_id"] not in all_recording_ids,
                f"recording identity is duplicated: {cell['recording_id']}",
            )
            all_recording_ids.add(cell["recording_id"])
            cell_rel = Path("cells") / model.lower() / _safe_cell_component(cell_id)
            action_rel = cell_rel / "action_manifest.json"
            recording_rel = cell_rel / "recording_receipt.json"
            action_payload = pretty_bytes(cell["action_manifest"])
            action_sha = sha256_bytes(action_payload)
            recording_unsigned = dict(cell["recording_receipt"])
            recording_unsigned.pop("payload_sha256")
            recording_unsigned["action_manifest_sha256"] = action_sha
            recording_receipt = sign_document(recording_unsigned)
            recording_payload = pretty_bytes(recording_receipt)
            files[str(action_rel)] = action_payload
            files[str(recording_rel)] = recording_payload
            action_descriptor = bytes_descriptor(str(action_rel), action_payload)
            recording_descriptor = bytes_descriptor(str(recording_rel), recording_payload)
            for request in cell["requests"]:
                enriched = dict(request)
                enriched["action_manifest"] = action_descriptor
                enriched["recording_receipt"] = recording_descriptor
                require(
                    enriched["source_request_id"] not in all_source_request_ids,
                    f"source request identity is duplicated: {enriched['source_request_id']}",
                )
                all_source_request_ids.add(enriched["source_request_id"])
                requests.append(enriched)
                timing_entries.append({
                    "cell_id": cell_id,
                    "request_index": request["request_index"],
                    "request_receipt": request["official_request_receipt"],
                    "adapter_completion": request["adapter_completion"],
                    "adapter_journal": request["adapter_journal"],
                })
            freeze_entries.append({
                "cell_receipt": cell["cell_descriptor"],
                "server_request_receipts": cell["request_descriptors"],
                "resource_receipt": None,
            })
            roster.append({
                "cell_id": cell_id,
                "recording_id": cell["recording_id"],
                "model_id": model,
                "layout_pair_id": cell["cell"]["layout_pair_id"],
                "condition_id": cell["condition_id"],
                "recording_status": "valid_complete",
                "executed_action_count": MODEL_LIMITS[model]["action_cap"],
                "censor_reason": None,
                "recording_receipt_path": recording_descriptor["path"],
                "recording_receipt_sha256": recording_descriptor["sha256"],
                "action_manifest_path": action_descriptor["path"],
                "action_manifest_sha256": action_descriptor["sha256"],
                "source_video_id": cell["source_video_id"],
                "source_video_sha256": cell["viewport_descriptor"]["sha256"],
            })
            cell_outputs.append({
                "cell_id": cell_id,
                "source_cell_receipt": cell["cell_descriptor"],
                "action_manifest": action_descriptor,
                "recording_receipt": recording_descriptor,
                "official_request_count": len(cell["requests"]),
                "source_video": cell["viewport_descriptor"],
            })
        expected_cell_count = len(LAYOUT_IDS) * len(CONDITIONS)
        complete = len(cells) == expected_cell_count
        expected_request_count = expected_cell_count * MODEL_LIMITS[model]["request_count"]
        if mode == "formal_full":
            require(complete and len(requests) == expected_request_count,
                    f"formal {model} compiler output is incomplete")
        timing_schema = (
            TIMING_INVENTORY_SCHEMA
            if mode == "formal_full"
            else DIAGNOSTIC_TIMING_INVENTORY_SCHEMA
        )
        suffix = "" if mode == "formal_full" else ".diagnostic"
        timing_name = f"{model.lower()}_development_timing_request_inventory{suffix}.json"
        provenance_name = f"{model.lower()}_development_request_provenance{suffix}.json"
        freeze_name = f"{model.lower()}_development_freeze_cells{suffix}.json"
        provenance = sign_document({
            "schema_version": PROVENANCE_SCHEMA,
            "study_id": STUDY_ID,
            "mode": mode,
            "model_id": model,
            "stage": "development",
            "status": "complete" if complete else "diagnostic_partial_only",
            "safe_for_formal_release": mode == "formal_full" and complete,
            "annotation_state": "not_started",
            "camera_id": camera_id,
            "camera_crop_contract": None,
            "resource_measurements": None,
            "labels": None,
            "episode_roster": roster,
            "requests": requests,
        })
        timing_inventory: dict[str, Any] = {
            "schema_version": timing_schema,
            "study_id": STUDY_ID,
            "model_id": model,
            "request_receipts": timing_entries,
        }
        if mode != "formal_full":
            timing_inventory.update({
                "status": "diagnostic_partial_only",
                "safe_for_timing_binding": False,
                "missing_layout_pair_ids": [
                    layout for layout in LAYOUT_IDS
                    if (model, layout) not in aggregates
                ],
            })
        freeze_fragment = sign_document({
            "schema_version": FREEZE_FRAGMENT_SCHEMA,
            "study_id": STUDY_ID,
            "mode": mode,
            "model_id": model,
            "status": "complete" if complete else "diagnostic_partial_only",
            "safe_for_alignment_input": mode == "formal_full" and complete,
            "resource_receipts_synthesized": False,
            "development_cells": freeze_entries,
        })
        timing_payload = pretty_bytes(timing_inventory)
        provenance_payload = pretty_bytes(provenance)
        freeze_payload = pretty_bytes(freeze_fragment)
        files[timing_name] = timing_payload
        files[provenance_name] = provenance_payload
        files[freeze_name] = freeze_payload
        model_outputs[model] = {
            "complete": complete,
            "cell_count": len(cells),
            "request_count": len(requests),
            "action_count": len(cells) * MODEL_LIMITS[model]["action_cap"],
            "aggregate_receipts": aggregate_descriptors[model],
            "timing_request_inventory": bytes_descriptor(timing_name, timing_payload),
            "request_provenance": bytes_descriptor(provenance_name, provenance_payload),
            "freeze_cell_evidence": bytes_descriptor(freeze_name, freeze_payload),
            "cells": cell_outputs,
        }
        total_requests += len(requests)
        total_actions += len(cells) * MODEL_LIMITS[model]["action_cap"]

    formal_complete = mode == "formal_full" and all(
        model_outputs[model]["complete"] for model in MODELS
    )
    compiler_source = {
        "path": "workshops/corl2026_world_models/analysis/compile_development_evidence.py",
        "sha256": sha256_file(Path(__file__).resolve()),
        "bytes": Path(__file__).resolve().stat().st_size,
    }
    freeze_validator_dependency = {
        "path": "workshops/corl2026_world_models/analysis/freeze_development_release.py",
        "sha256": sha256_file(FREEZE_PATH.resolve()),
        "bytes": FREEZE_PATH.resolve().stat().st_size,
    }
    receipt = sign_document({
        "schema_version": COMPILER_SCHEMA,
        "study_id": STUDY_ID,
        "mode": mode,
        "status": "compiled_complete" if formal_complete else "diagnostic_partial_only",
        "formal_cohort_complete": formal_complete,
        "safe_for_timing_binding": formal_complete,
        "safe_for_confirmation_release": False,
        "input_manifest": file_descriptor(manifest_path),
        "planned_cells": planned_descriptor,
        "compiler_source": compiler_source,
        "freeze_validator_dependency": freeze_validator_dependency,
        "raw_root": str(raw_root),
        "camera_id": camera_id,
        "models": model_outputs,
        "counts": {
            "compiled_cells": sum(item["cell_count"] for item in model_outputs.values()),
            "compiled_source_behavioral_requests": total_requests,
            "compiled_source_behavioral_actions": total_actions,
        },
        "compiler_science_activity": {
            "model_loads": 0,
            "model_requests_issued": 0,
            "simulator_processes_started": 0,
            "physical_resets": 0,
            "behavioral_actions_executed": 0,
            "labels_created": 0,
        },
        "unsupported_outputs": {
            "resource_metrics": "not_synthesized",
            "camera_crop_contract": "not_created",
            "human_pixel_blindness_receipt": "not_created",
            "labels": "not_created",
            "movement_threshold": "not_created",
            "confirmation_release": "not_created",
        },
        "claim_boundary": (
            "CPU-only compilation of retained immutable development evidence. "
            "A complete formal receipt permits timing-sidecar binding only; camera crop, "
            "annotation, measured resources, and confirmation remain separately gated."
        ),
    })
    files["compiler_receipt.json"] = pretty_bytes(receipt)
    _write_atomic_directory(output_dir, files)
    return receipt


def _write_atomic_directory(target: Path, files: Mapping[str, bytes]) -> None:
    target = Path(target)
    require(not target.exists(), f"refusing to overwrite output directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        for relative, payload in sorted(files.items()):
            path = temporary / relative
            require(path.resolve().is_relative_to(temporary.resolve()), "output path escaped bundle")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = compile_manifest(args.manifest, args.manifest_sha256, args.output_dir)
    except CompilerError as error:
        print(json.dumps({"status": "blocked", "reason": str(error)}, sort_keys=True))
        return 2
    output = Path(args.output_dir).resolve() / "compiler_receipt.json"
    print(json.dumps({
        "status": result["status"],
        "formal_cohort_complete": result["formal_cohort_complete"],
        "safe_for_timing_binding": result["safe_for_timing_binding"],
        "compiled_cells": result["counts"]["compiled_cells"],
        "compiled_source_behavioral_requests": result["counts"]["compiled_source_behavioral_requests"],
        "compiler_science_activity": result["compiler_science_activity"],
        "output": str(output),
        "sha256": sha256_file(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
