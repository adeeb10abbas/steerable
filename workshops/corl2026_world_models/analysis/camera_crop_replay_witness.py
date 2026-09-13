#!/usr/bin/env python3
"""Fail-closed pixel replay for the WMF selected-camera crop contract.

This module has two deliberately separate uses:

* ``witness`` reopens the retained P00 capture and one decoded request, hashes
  the active source/checkpoint/runtime chain, and emits one signed per-model
  ``wmf-camera-crop-contract-v1`` document; and
* :func:`replay_original_camera_frame` and :func:`extract_generated_crop`
  execute the signed transforms for downstream blind-media packaging.

It is CPU-only.  It never imports a model, opens a simulator, issues a model
request, creates a label, or authorizes confirmation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

SCHEMA = "wmf-camera-crop-contract-v1"
RUNTIME_SCHEMA = "wmf-camera-crop-witness-runtime-contract-v1"
STUDY_ID = "WMF-ABLATION-001"
CAMERA_ID = "over_shoulder_left_camera"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
ZERO_SCIENCE_COUNTS = {
    "model_runtime_loads": 0,
    "model_servers_started": 0,
    "model_requests_issued_by_job": 0,
    "simulator_processes_started": 0,
    "physical_resets": 0,
    "robot_episodes": 0,
    "behavioral_actions_executed_by_job": 0,
    "behavioral_cells_launched_by_job": 0,
    "labels_created_by_job": 0,
}


class CameraCropWitnessError(RuntimeError):
    """A retained byte, executable transform, or authority gate changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CameraCropWitnessError(message)


def compact_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CameraCropWitnessError("value is not canonical finite JSON") from error


def pretty_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CameraCropWitnessError("value is not canonical finite JSON") from error


def n3_canonical_bytes(value: Any) -> bytes:
    """Match n3_first_live._canonical_bytes byte-for-byte."""

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CameraCropWitnessError("value is not finite N3 canonical JSON") from error


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_document(value: Mapping[str, Any]) -> dict[str, Any]:
    require("payload_sha256" not in value, "document is already signed")
    result = dict(value)
    result["payload_sha256"] = sha256_bytes(compact_bytes(result))
    return result


def verify_signed_document(value: Mapping[str, Any], label: str) -> None:
    digest = value.get("payload_sha256")
    require(isinstance(digest, str) and SHA_RE.fullmatch(digest) is not None,
            f"{label} signature is missing")
    unsigned = dict(value)
    unsigned.pop("payload_sha256")
    require(sha256_bytes(compact_bytes(unsigned)) == digest, f"{label} signature changed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CameraCropWitnessError(f"non-finite JSON token: {value}")


def load_json(path: Path, label: str) -> dict[str, Any]:
    secure_path(path, label=label, kind="file")
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except CameraCropWitnessError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CameraCropWitnessError(f"{label} is unreadable JSON") from error
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def _components(path: Path) -> list[Path]:
    absolute = Path(path)
    require(absolute.is_absolute(), f"path is not absolute: {absolute}")
    result = [absolute]
    result.extend(absolute.parents)
    return list(reversed(result))


def secure_path(
    path: Path,
    *,
    label: str,
    kind: str,
    expected: Path | None = None,
    within: Path | None = None,
) -> Path:
    """Return one lexical absolute path only if no component is a symlink."""

    supplied = Path(path)
    require(supplied.is_absolute(), f"{label} is not absolute")
    if expected is not None:
        require(supplied == Path(expected), f"{label} path changed")
    for component in _components(supplied):
        if component == Path("/"):
            continue
        require(not component.is_symlink(), f"{label} contains a symlink: {component}")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise CameraCropWitnessError(f"{label} is missing") from error
    require(resolved == supplied, f"{label} has lexical/resolved path drift")
    if within is not None:
        root = secure_path(Path(within), label=f"{label} root", kind="dir")
        require(resolved == root or root in resolved.parents, f"{label} escapes its root")
    if kind == "file":
        require(resolved.is_file(), f"{label} is not a file")
    elif kind == "dir":
        require(resolved.is_dir(), f"{label} is not a directory")
    else:
        raise CameraCropWitnessError(f"unknown secure-path kind: {kind}")
    return resolved


def file_identity(
    path: Path,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
    expected_bytes: int | None = None,
    within: Path | None = None,
) -> dict[str, Any]:
    resolved = secure_path(
        Path(path), label=label, kind="file", expected=expected_path, within=within
    )
    identity = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }
    if expected_sha256 is not None:
        require(identity["sha256"] == expected_sha256, f"{label} SHA-256 changed")
    if expected_bytes is not None:
        require(identity["bytes"] == expected_bytes, f"{label} byte count changed")
    return identity


def array_data_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    require(not array.dtype.hasobject, "object arrays are prohibited")
    return sha256_bytes(array.tobytes(order="C"))


def n3_value_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    if all(hasattr(value, name) for name in ("detach", "cpu", "contiguous")):
        import torch

        tensor = value.detach().cpu().contiguous()
        header = {"kind": "torch", "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
        raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    else:
        array = np.ascontiguousarray(np.asarray(value))
        header = {"kind": "numpy", "dtype": array.dtype.str, "shape": list(array.shape)}
        raw = array.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(n3_canonical_bytes(header))
    digest.update(raw)
    return {**header, "value_sha256": digest.hexdigest()}


def capture_value_identity(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = {"dtype": array.dtype.str, "shape": list(array.shape), "order": "C"}
    digest = hashlib.sha256()
    digest.update(compact_bytes(header))
    digest.update(array.tobytes(order="C"))
    return {**header, "value_sha256": digest.hexdigest()}


def _same_array(left: Any, right: Any, label: str) -> None:
    import numpy as np

    a = np.asarray(left)
    b = np.asarray(right)
    require(a.shape == b.shape and a.dtype == b.dtype, f"{label} shape/dtype changed")
    require(bool(np.array_equal(a, b)), f"{label} bytes changed")


def _load_capture_npz(
    descriptor: Mapping[str, Any], *, label: str, expected: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    require(isinstance(descriptor, Mapping), f"{label} descriptor is missing")
    identity = file_identity(
        Path(str(descriptor.get("path", ""))),
        label=label,
        expected_path=Path(str(descriptor.get("path", ""))),
        expected_sha256=str(expected["sha256"]),
        expected_bytes=int(expected["bytes"]),
    )
    require(descriptor.get("sha256") == identity["sha256"], f"{label} receipt hash changed")
    require(descriptor.get("bytes") == identity["bytes"], f"{label} receipt bytes changed")
    described = descriptor.get("arrays")
    require(isinstance(described, Mapping) and described, f"{label} array inventory is missing")
    arrays: dict[str, Any] = {}
    try:
        with np.load(Path(identity["path"]), allow_pickle=False) as archive:
            require(set(archive.files) == set(described), f"{label} NPZ keys changed")
            for key in sorted(archive.files):
                array = np.ascontiguousarray(archive[key]).copy()
                observed = capture_value_identity(array)
                require(described[key] == observed, f"{label} array identity changed: {key}")
                arrays[key] = array
    except (OSError, ValueError) as error:
        raise CameraCropWitnessError(f"{label} NPZ is unreadable") from error
    return arrays, identity


def _logical_nested(value: Any) -> Any:
    if isinstance(value, list):
        return [_logical_nested(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    if value.get("__type__") in {"numpy", "torch"}:
        keys = ("__type__", "kind", "dtype", "shape", "value_sha256", "logical_type")
        return {key: value[key] for key in keys if key in value}
    return {key: _logical_nested(child) for key, child in value.items()}


def _load_nested_node(node: Any, root: Path, retained: list[dict[str, Any]], role: str) -> Any:
    import numpy as np

    if not isinstance(node, Mapping) or "__type__" not in node:
        if isinstance(node, list):
            return [_load_nested_node(item, root, retained, role) for item in node]
        return node
    node_type = node.get("__type__")
    if node_type in {"numpy", "torch"}:
        artifact = node.get("artifact")
        require(isinstance(artifact, Mapping), f"{role} nested artifact descriptor is missing")
        relative = artifact.get("path")
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
                f"{role} nested artifact path is invalid")
        path = root / relative
        identity = file_identity(
            path,
            label=f"{role} nested artifact",
            expected_path=path,
            expected_sha256=str(artifact.get("sha256")),
            expected_bytes=int(artifact.get("bytes", -1)),
            within=root,
        )
        retained.append(identity)
        if node_type == "numpy":
            with path.open("rb") as stream:
                value = np.load(stream, allow_pickle=False)
        else:
            import torch

            with path.open("rb") as stream:
                value = torch.load(stream, map_location="cpu", weights_only=True)
        observed = n3_value_identity(value)
        for key in ("kind", "dtype", "shape", "value_sha256"):
            require(node.get(key) == observed.get(key), f"{role} nested {key} changed")
        return value
    if node_type in {"list", "tuple"}:
        items = node.get("items")
        require(isinstance(items, list), f"{role} nested sequence is invalid")
        values = [_load_nested_node(item, root, retained, role) for item in items]
        return tuple(values) if node_type == "tuple" else values
    if node_type in {"mapping", "dataclass"}:
        key = "items" if node_type == "mapping" else "fields"
        items = node.get(key)
        require(isinstance(items, Mapping), f"{role} nested mapping is invalid")
        return {
            name: _load_nested_node(child, root, retained, role)
            for name, child in items.items()
        }
    if node_type == "path":
        require(isinstance(node.get("value"), str), f"{role} nested path is invalid")
        return Path(node["value"])
    raise CameraCropWitnessError(f"{role} unsupported nested type: {node_type}")


def load_n3_artifact(
    descriptor: Mapping[str, Any], *, role: str, allowed_root: Path
) -> tuple[Any, list[dict[str, Any]]]:
    require(isinstance(descriptor, Mapping), f"{role} descriptor is missing")
    manifest_path = Path(str(descriptor.get("manifest_path", "")))
    manifest_identity = file_identity(
        manifest_path,
        label=f"{role} manifest",
        expected_path=manifest_path,
        expected_sha256=str(descriptor.get("manifest_sha256")),
        within=allowed_root,
    )
    manifest = load_json(manifest_path, f"{role} manifest")
    require(manifest.get("schema_version") == "wmf-lossless-nested-payload-v1",
            f"{role} manifest schema changed")
    require(manifest.get("role") == role, f"{role} manifest role changed")
    structure = manifest.get("structure")
    logical = sha256_bytes(n3_canonical_bytes(_logical_nested(structure)))
    require(manifest.get("logical_sha256") == logical, f"{role} logical hash changed")
    require(descriptor.get("logical_sha256") == logical, f"{role} descriptor logical hash changed")
    retained = [manifest_identity]
    return _load_nested_node(structure, manifest_path.parent, retained, role), retained


def _load_d1_mapping(
    descriptor: Mapping[str, Any], *, role: str, allowed_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np
    import torch

    require(isinstance(descriptor, Mapping), f"{role} mapping descriptor is missing")
    entries = descriptor.get("entries")
    require(isinstance(entries, list) and descriptor.get("entry_count") == len(entries),
            f"{role} mapping inventory changed")
    result: dict[str, Any] = {}
    identities: list[dict[str, Any]] = []
    logical: list[dict[str, Any]] = []
    for entry in entries:
        require(isinstance(entry, Mapping), f"{role} mapping entry is invalid")
        key = entry.get("key")
        require(isinstance(key, str) and key not in result, f"{role} mapping key changed")
        path = Path(str(entry.get("path", "")))
        identity = file_identity(
            path,
            label=f"{role} mapping artifact {key}",
            expected_path=path,
            expected_sha256=str(entry.get("file_sha256")),
            expected_bytes=int(entry.get("bytes", -1)),
            within=allowed_root,
        )
        identities.append(identity)
        kind = entry.get("kind")
        if kind == "numpy_array":
            with path.open("rb") as stream:
                value = np.load(stream, allow_pickle=False)
            observed = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "data_sha256": array_data_sha256(value),
            }
        elif kind == "torch_tensor":
            with path.open("rb") as stream:
                value = torch.load(stream, map_location="cpu", weights_only=True)
            contiguous = value.detach().cpu().contiguous()
            observed = {
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
                "data_sha256": sha256_bytes(
                    contiguous.view(torch.uint8).numpy().tobytes(order="C")
                ),
            }
        elif kind == "json_value":
            value = load_json(path, f"{role} JSON value") if path.read_text().lstrip().startswith("{") else json.loads(path.read_text())
            observed = {"json_sha256": sha256_bytes(compact_bytes(value))}
        else:
            raise CameraCropWitnessError(f"{role} mapping kind changed: {kind}")
        for name, wanted in observed.items():
            require(entry.get(name) == wanted, f"{role} {key} {name} changed")
        result[key] = value
        logical.append(
            {
                field: entry[field]
                for field in ("key", "kind", "shape", "dtype", "data_sha256", "json_sha256")
                if field in entry
            }
        )
    require(descriptor.get("content_sha256") == sha256_bytes(compact_bytes(logical)),
            f"{role} mapping content hash changed")
    return result, identities


def _tracked_source_identity(root: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    source = secure_path(root, label=label, kind="dir", expected=root)
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True, timeout=60
        ).strip()
        tree = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD^{tree}"], text=True, timeout=60
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=no"],
            text=True,
            timeout=60,
        )
        raw = subprocess.check_output(
            ["git", "-C", str(source), "ls-files", "-z"], timeout=60
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CameraCropWitnessError(f"{label} Git identity unavailable") from error
    require(commit == expected["commit"] and tree == expected["git_tree"],
            f"{label} commit/tree changed")
    require(not status, f"{label} tracked working copy changed")
    paths = sorted(os.fsdecode(value) for value in raw.split(b"\0") if value)
    aggregate = hashlib.sha256()
    total = 0
    for relative in paths:
        path = source / relative
        identity = file_identity(path, label=f"{label} tracked file", within=source)
        total += identity["bytes"]
        aggregate.update(f"{identity['sha256']}  {relative}\n".encode("utf-8", "surrogateescape"))
    require(len(paths) == expected["tracked_file_count"], f"{label} tracked count changed")
    require(total == expected["tracked_bytes"], f"{label} tracked bytes changed")
    require(aggregate.hexdigest() == expected["tracked_aggregate_sha256"],
            f"{label} tracked aggregate changed")
    required = []
    for relative, digest in sorted(expected["required_files"].items()):
        required.append(
            file_identity(
                source / relative,
                label=f"{label} required file {relative}",
                expected_sha256=digest,
                within=source,
            )
        )
    return {
        "path": str(source),
        "commit": commit,
        "git_tree": tree,
        "tracked_file_count": len(paths),
        "tracked_bytes": total,
        "tracked_aggregate_sha256": aggregate.hexdigest(),
        "required_files": required,
    }


def _checkpoint_identity(root: Path, expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    checkpoint = secure_path(root, label=label, kind="dir", expected=root)
    paths = sorted(
        path
        for path in checkpoint.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(checkpoint).parts
    )
    aggregate = hashlib.sha256()
    total = 0
    for path in paths:
        identity = file_identity(path, label=f"{label} payload", within=checkpoint)
        relative = path.relative_to(checkpoint).as_posix()
        total += identity["bytes"]
        aggregate.update(f"{identity['sha256']}  {relative}\n".encode("utf-8"))
    require(len(paths) == expected["checkpoint_file_count"], f"{label} file count changed")
    require(total == expected["checkpoint_bytes"], f"{label} bytes changed")
    require(aggregate.hexdigest() == expected["checkpoint_aggregate_sha256"],
            f"{label} aggregate changed")
    return {
        "path": str(checkpoint),
        "revision": expected["checkpoint_revision"],
        "payload_file_count": len(paths),
        "payload_bytes": total,
        "payload_aggregate_sha256": aggregate.hexdigest(),
        "full_payload_rehash_performed": True,
    }


def _module_file(value: Any, label: str) -> dict[str, Any]:
    path = inspect.getsourcefile(value)
    require(path is not None, f"{label} source is unavailable")
    return file_identity(Path(path), label=label)


def _runtime_dependencies(model: str, runtime: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import PIL
    from PIL import Image
    import torch

    result: dict[str, Any] = {
        "python": file_identity(Path(sys.executable).resolve(), label="Python executable"),
        "numpy": {
            "version": np.__version__,
            "module": file_identity(Path(np.__file__), label="NumPy module"),
        },
        "torch": {
            "version": torch.__version__,
            "module": file_identity(Path(torch.__file__), label="Torch module"),
            "interpolate_source": _module_file(torch.nn.functional.interpolate, "torch interpolate source"),
        },
        "pillow": {
            "version": PIL.__version__,
            "module": file_identity(Path(PIL.__file__), label="Pillow module"),
            "image_source": _module_file(Image.Image.resize, "Pillow resize source"),
        },
    }
    if model == "N3":
        from openpi_client import image_tools

        source = _module_file(image_tools.resize_with_pad, "openpi resize source")
        require(
            source["sha256"] == runtime["dependencies"]["n3"]["openpi_image_tools_sha256"],
            "active openpi image_tools source changed",
        )
        result["openpi_client.image_tools"] = source
    else:
        import torchvision
        from torchvision.transforms import v2

        expected = runtime["dependencies"]["d1"]["required_python_packages"]
        require(np.__version__ == expected["numpy"], "active D1 NumPy version changed")
        require(torch.__version__.split("+")[0] == expected["torch"], "active D1 torch version changed")
        require(torchvision.__version__.split("+")[0] == expected["torchvision"],
                "active D1 torchvision version changed")
        result["torchvision"] = {
            "version": torchvision.__version__,
            "module": file_identity(Path(torchvision.__file__), label="torchvision module"),
            "center_crop_source": _module_file(v2.CenterCrop, "torchvision CenterCrop source"),
            "resize_source": _module_file(v2.Resize, "torchvision Resize source"),
        }
    return result


def _load_resize_with_pad(path: Path):
    source = secure_path(path, label="RoboLab resize source", kind="file", expected=path)
    specification = importlib.util.spec_from_file_location("wmf_camera_robolab_image_utils", source)
    require(specification is not None and specification.loader is not None,
            "cannot load RoboLab resize source")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.resize_with_pad


def _n3_replay_stages(image: Any) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from openpi_client import image_tools

    raw = np.ascontiguousarray(np.asarray(image))
    require(raw.shape == (720, 1280, 3) and raw.dtype == np.uint8,
            "N3 original camera frame shape/dtype changed")
    resized = np.ascontiguousarray(image_tools.resize_with_pad(raw, 360, 640))
    require(resized.shape == (360, 640, 3) and resized.dtype == np.uint8,
            "N3 openpi resize output changed")
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float()
    down = functional.interpolate(tensor, size=(180, 320), mode="bilinear")
    down = np.ascontiguousarray(down.squeeze(0).permute(1, 2, 0).numpy().astype(resized.dtype))
    require(down.shape == (180, 320, 3) and down.dtype == np.uint8,
            "N3 torch resize output changed")
    return {"openpi_resized": resized, "torch_downsampled": down}


def _d1_replay_stages(image: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms import v2

    raw = np.ascontiguousarray(np.asarray(image))
    require(raw.shape == (720, 1280, 3) and raw.dtype == np.uint8,
            "D1 original camera frame shape/dtype changed")
    dependency = contract.get("dependency_chain", {}).get("robolab_resize_with_pad_source")
    require(isinstance(dependency, Mapping), "D1 resize dependency is missing")
    source = file_identity(
        Path(str(dependency.get("path", ""))),
        label="D1 signed RoboLab resize source",
        expected_sha256=str(dependency.get("sha256")),
        expected_bytes=int(dependency.get("bytes", -1)),
    )
    resize_with_pad = _load_resize_with_pad(Path(source["path"]))
    wire = np.ascontiguousarray(resize_with_pad(raw, 180, 320))
    require(wire.shape == (180, 320, 3) and wire.dtype == np.uint8,
            "D1 RoboLab resize output changed")
    converted = wire[None, ...]
    tensor = torch.from_numpy(converted).to(torch.float32) / 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    cropped = v2.CenterCrop((171, 304))(tensor)
    require(list(cropped.shape) == [1, 3, 171, 304], "D1 center crop shape changed")
    resized = v2.Resize(
        (176, 320), interpolation=InterpolationMode.BILINEAR, antialias=True
    )(cropped)
    normalized = (resized.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
    require(normalized.shape == (1, 176, 320, 3) and normalized.dtype == np.uint8,
            "D1 normalized view changed")
    return {
        "wire": wire,
        "converted": converted,
        "tensor": tensor,
        "center_crop": cropped,
        "resized": resized,
        "normalized": normalized,
    }


def replay_original_camera_frame(image: Any, contract: Mapping[str, Any]) -> Any:
    """Execute the signed raw-camera replay and return its selected HWC crop.

    Runtime dependency identities are checked by default; callers therefore
    cannot silently approximate Pillow, torch, torchvision, or openpi rules.
    """

    validate_camera_crop_contract(contract)
    model = str(contract["model_id"])
    _verify_runtime_dependency_chain(model, contract)
    if model == "N3":
        stages = _n3_replay_stages(image)
        output = stages["torch_downsampled"][:168, :, :]
    else:
        stages = _d1_replay_stages(image, contract)
        output = stages["normalized"][0]
    import numpy as np

    output = np.ascontiguousarray(output)
    require(list(output.shape) == contract["original_camera_replay"]["output_shape"],
            "original replay output shape changed")
    witness_input = contract["original_camera_replay"].get("witness_input_value_sha256")
    if witness_input == array_data_sha256(image):
        require(
            array_data_sha256(output)
            == contract["original_camera_replay"]["output_value_sha256"],
            "original witness output bytes changed",
        )
    return output


def extract_generated_crop(frames: Any, contract: Mapping[str, Any]) -> Any:
    """Apply the signed half-open THWC crop to exact decoded frames."""

    validate_camera_crop_contract(contract)
    import numpy as np

    value = np.asarray(frames)
    crop = contract["generated_decoded_crop"]
    require(list(value.shape) == crop["canvas_shape"], "generated decoded canvas shape changed")
    require(value.dtype == np.uint8, "generated decoded canvas dtype changed")
    y0, y1 = crop["y"]
    x0, x1 = crop["x"]
    result = np.ascontiguousarray(value[:, y0:y1, x0:x1, :])
    require(list(result.shape) == crop["output_shape"], "generated crop output shape changed")
    return result


def _verify_runtime_dependency_chain(model: str, contract: Mapping[str, Any]) -> None:
    current = _runtime_dependencies(model, {"dependencies": contract["pinned_dependencies"]})
    signed = contract.get("runtime_dependencies")
    require(isinstance(signed, Mapping), "runtime dependency chain is missing")
    require(current == signed, "active runtime dependency chain changed")


def _expected_transform_chain(model: str) -> list[dict[str, Any]]:
    if model == "N3":
        return [
            {
                "operation": "select_camera_hwc",
                "camera_id": CAMERA_ID,
                "environment_index": 0,
                "input_shape": [1, 720, 1280, 3],
                "output_shape": [720, 1280, 3],
            },
            {
                "operation": "openpi_resize_with_pad_hwc",
                "output_shape": [360, 640, 3],
                "interpolation": "PIL.Image.Resampling.BILINEAR",
                "rounding": "exact_active_openpi_source",
            },
            {
                "operation": "torch_interpolate_hwc",
                "output_shape": [180, 320, 3],
                "mode": "bilinear",
                "align_corners": False,
                "input_conversion": "uint8 HWC -> float32 NCHW",
                "output_conversion": "float32 NCHW -> uint8 HWC by NumPy astype",
            },
            {
                "operation": "compose_mosaic_hwc",
                "canvas_shape": [540, 640, 3],
                "placement": {"y": [360, 540], "x": [0, 320]},
            },
            {
                "operation": "crop_half_open_hwc",
                "y": [360, 528],
                "x": [0, 320],
                "output_shape": [168, 320, 3],
            },
        ]
    require(model == "D1", "unknown crop model")
    return [
        {
            "operation": "select_camera_hwc",
            "camera_id": CAMERA_ID,
            "environment_index": 0,
            "input_shape": [1, 720, 1280, 3],
            "output_shape": [720, 1280, 3],
        },
        {
            "operation": "robolab_resize_with_pad_hwc",
            "output_shape": [180, 320, 3],
            "interpolation": "PIL.Image.Resampling.BILINEAR",
            "rounding": "exact_active_robolab_source",
        },
        {
            "operation": "add_time_axis_and_map_key",
            "key": "video.exterior_image_1_left",
            "output_shape": [1, 180, 320, 3],
        },
        {
            "operation": "video_to_tensor",
            "input_layout": "THWC uint8",
            "output_layout": "TCHW float32",
            "scale_divisor": 255,
        },
        {
            "operation": "torchvision_v2_center_crop",
            "size": [171, 304],
            "observed_bounds": {"y": [4, 175], "x": [8, 312]},
        },
        {
            "operation": "torchvision_v2_resize",
            "output_shape": [1, 3, 176, 320],
            "interpolation": "InterpolationMode.BILINEAR",
            "antialias": True,
        },
        {"operation": "video_color_jitter_eval", "effect": "identity"},
        {
            "operation": "video_to_numpy",
            "input_layout": "TCHW float32",
            "output_layout": "THWC uint8",
            "conversion": "permute, multiply 255, torch.to(uint8), cpu numpy",
        },
        {
            "operation": "dreamzero_droid_mosaic_hwc",
            "canvas_shape": [352, 640, 3],
            "placement": {"y": [176, 352], "x": [0, 320]},
        },
        {
            "operation": "crop_half_open_hwc",
            "y": [176, 352],
            "x": [0, 320],
            "output_shape": [176, 320, 3],
        },
    ]


def _expected_crop_operation(model: str) -> str:
    require(model in {"N3", "D1"}, "unknown crop model")
    if model == "N3":
        return "raw_selected_camera_replay_and_generated_uint8_thwc_crop_half_open_y360_528_x0_320"
    return "raw_selected_camera_replay_and_generated_uint8_thwc_crop_half_open_y176_352_x0_320"


def validate_camera_crop_contract(
    value: Mapping[str, Any], expected_model: str | None = None
) -> None:
    require(isinstance(value, Mapping), "camera crop contract is not an object")
    verify_signed_document(value, "camera crop contract")
    model = value.get("model_id")
    require(model in {"N3", "D1"}, "camera crop contract model changed")
    if expected_model is not None:
        require(model == expected_model, "camera crop contract belongs to another model")
    require(value.get("schema_version") == SCHEMA and value.get("study_id") == STUDY_ID,
            "camera crop contract schema/study changed")
    require(value.get("status") == "qualified_from_original_camera_pixels",
            "camera crop contract did not pass")
    require(value.get("camera_id") == CAMERA_ID, "camera crop contract selected another camera")
    expected_height = 168 if model == "N3" else 176
    require(value.get("crop_operation") == _expected_crop_operation(str(model)),
            "camera crop operation changed")
    require(value.get("image_width_px") == 320
            and value.get("image_height_px") == expected_height,
            "camera crop image dimensions changed")
    expected_generated = {
        "N3": ([33, 528, 640, 3], [360, 528], [0, 320], [33, 168, 320, 3]),
        "D1": ([9, 352, 640, 3], [176, 352], [0, 320], [9, 176, 320, 3]),
    }[str(model)]
    generated = value.get("generated_decoded_crop")
    require(isinstance(generated, Mapping), "generated crop is missing")
    require(
        (
            generated.get("operation"),
            generated.get("canvas_shape"),
            generated.get("y"),
            generated.get("x"),
            generated.get("output_shape"),
        )
        == ("crop_half_open_thwc", *expected_generated),
        "generated crop geometry changed",
    )
    original = value.get("original_camera_replay")
    require(isinstance(original, Mapping), "original camera replay is missing")
    expected_output = [168, 320, 3] if model == "N3" else [176, 320, 3]
    require(original.get("source_camera_shape") == [720, 1280, 3],
            "original source shape changed")
    require(original.get("output_shape") == expected_output, "original crop shape changed")
    require(original.get("transform_chain") == _expected_transform_chain(str(model)),
            "original transform chain changed")
    for digest_name in ("witness_input_value_sha256", "output_value_sha256"):
        digest = original.get(digest_name)
        require(isinstance(digest, str) and SHA_RE.fullmatch(digest) is not None,
                f"original replay {digest_name} is invalid")
    require(value.get("science_counts") == ZERO_SCIENCE_COUNTS,
            "camera witness science counts are nonzero")
    for field in (
        "simulator_state_render_used",
        "whole_frame_identity",
        "safe_to_release_confirmation",
        "confirmation_released",
        "behavioral_policy_skill_evaluated",
    ):
        require(value.get(field) is False, f"camera crop contract incorrectly sets {field}")
    checks = value.get("replay_checks")
    require(isinstance(checks, Mapping) and checks and all(item is True for item in checks.values()),
            "camera crop replay checks did not all pass")


def _capture_bundle(runtime: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    capture_path = Path(runtime["paths"]["capture_receipt"])
    expected_capture = runtime["retained_inputs"]["capture_receipt"]
    capture_identity = file_identity(
        capture_path,
        label="fixed capture receipt",
        expected_path=capture_path,
        expected_sha256=expected_capture["sha256"],
        expected_bytes=expected_capture["bytes"],
    )
    capture = load_json(capture_path, "fixed capture receipt")
    require(capture.get("schema_version") == "wmf-forecast-layout-fixed-observation-capture-v1"
            and capture.get("status") == "passed", "fixed capture did not pass")
    require(capture.get("model_request_count") == 0 and capture.get("behavioral_action_count") == 0,
            "fixed capture is not zero-policy preprocessing evidence")
    artifacts = capture.get("artifacts")
    require(isinstance(artifacts, Mapping), "fixed capture artifacts are missing")
    raw, raw_identity = _load_capture_npz(
        artifacts.get("raw_settled_observation", {}),
        label="raw settled observation",
        expected=runtime["retained_inputs"]["raw_settled_observation"],
    )
    intermediate, intermediate_identity = _load_capture_npz(
        artifacts.get("preprocessing_intermediates", {}),
        label="preprocessing intermediates",
        expected=runtime["retained_inputs"]["preprocessing_intermediates"],
    )
    return (
        {"capture": capture, "raw": raw, "intermediate": intermediate},
        {
            "capture_receipt": capture_identity,
            "raw_settled_observation": raw_identity,
            "preprocessing_intermediates": intermediate_identity,
        },
        artifacts,
    )


def _source_dependencies(runtime: Mapping[str, Any], model: str) -> dict[str, Any]:
    paths = runtime["paths"]
    robolab = secure_path(Path(paths["robolab_root"]), label="RoboLab root", kind="dir",
                          expected=Path(paths["robolab_root"]))
    try:
        robolab_commit = subprocess.check_output(
            ["git", "-C", str(robolab), "rev-parse", "HEAD"], text=True, timeout=60
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CameraCropWitnessError("RoboLab Git identity unavailable") from error
    require(robolab_commit == paths["robolab_commit"], "RoboLab commit changed")
    robolab_files = []
    for relative, digest in sorted(runtime["dependencies"]["robolab"].items()):
        robolab_files.append(
            file_identity(
                robolab / relative,
                label=f"RoboLab dependency {relative}",
                expected_sha256=digest,
                within=robolab,
            )
        )
    external_key = "n3" if model == "N3" else "d1"
    source_key = "n3_source" if model == "N3" else "d1_source"
    source = _tracked_source_identity(
        Path(paths[source_key]), runtime["dependencies"][external_key], f"{model} source"
    )
    checkpoint_key = "n3_checkpoint" if model == "N3" else "d1_checkpoint"
    checkpoint = _checkpoint_identity(
        Path(paths[checkpoint_key]), runtime["dependencies"][external_key], f"{model} checkpoint"
    )
    if model == "D1":
        config_path = Path(paths["d1_checkpoint"]) / runtime["dependencies"]["d1"]["checkpoint_config_relative"]
        file_identity(
            config_path,
            label="D1 checkpoint transform configuration",
            expected_sha256=runtime["dependencies"]["d1"]["checkpoint_config_sha256"],
            within=Path(paths["d1_checkpoint"]),
        )
    result = {
        "robolab": {
            "path": str(robolab),
            "commit": robolab_commit,
            "required_files": robolab_files,
        },
        "model_source": source,
        "checkpoint": checkpoint,
    }
    if model == "D1":
        result["capture_overlay"] = file_identity(
            Path(paths["d1_capture_overlay"]),
            label="D1 fixed-capture overlay",
            expected_path=Path(paths["d1_capture_overlay"]),
            expected_sha256=runtime["dependencies"]["d1_capture_overlay_sha256"],
        )
    return result


def _n3_witness(runtime: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    import numpy as np
    import torch
    from torchvision.transforms import functional as tv_functional

    bundle, retained, capture_artifacts = _capture_bundle(runtime)
    raw = bundle["raw"]
    intermediate = bundle["intermediate"]
    camera_keys = {
        "left": "over_shoulder_left_camera",
        "right": "over_shoulder_right_camera",
        "wrist": "wrist_cam",
    }
    stages = {
        role: _n3_replay_stages(raw[f"image_obs/{camera}"])
        for role, camera in camera_keys.items()
    }
    for role, key in (
        ("left", "N3/left_image"),
        ("right", "N3/right_image"),
        ("wrist", "N3/wrist_image"),
    ):
        require(key in intermediate, f"retained N3 intermediate is missing: {key}")
        _same_array(stages[role]["openpi_resized"], intermediate[key], f"N3 {role} intermediate")
    mosaic = np.concatenate(
        (
            stages["wrist"]["openpi_resized"],
            np.concatenate(
                (stages["left"]["torch_downsampled"], stages["right"]["torch_downsampled"]),
                axis=1,
            ),
        ),
        axis=0,
    )
    wire, wire_identity = _load_capture_npz(
        capture_artifacts.get("N3", {}),
        label="N3 official wire observation",
        expected=runtime["retained_inputs"]["n3_wire"],
    )
    _same_array(mosaic, wire["observation/image"], "N3 raw-to-wire replay")
    retained["n3_wire"] = wire_identity
    qualification_path = Path(runtime["paths"]["n3_qualification"])
    expected_q = runtime["retained_inputs"]["n3_qualification"]
    q_identity = file_identity(
        qualification_path,
        label="N3 retained qualification",
        expected_path=qualification_path,
        expected_sha256=expected_q["sha256"],
        expected_bytes=expected_q["bytes"],
    )
    q = load_json(qualification_path, "N3 retained qualification")
    require(q.get("schema_version") == "wmf-n3-runtime-qualification-v1"
            and q.get("status") == "passed" and q.get("qualified") is True,
            "N3 retained qualification did not pass")
    requests = q.get("requests")
    require(isinstance(requests, list) and len(requests) == 6, "N3 request roster changed")
    request = requests[3]
    require(request.get("request_index") == 3 and request.get("request_id") == "left_decode"
            and request.get("decode_requested") is True, "N3 selected decode request changed")
    request_root = qualification_path.parent / "requests" / "03_left_decode"
    secure_path(request_root, label="N3 selected request root", kind="dir", expected=request_root)
    exact_wire, exact_wire_files = load_n3_artifact(
        request["wire_request_artifact"], role="exact_wire_request", allowed_root=request_root
    )
    _same_array(exact_wire["observation/image"], mosaic, "N3 retained request wire")
    transformed, transformed_files = load_n3_artifact(
        request["exact_transformed_model_input_artifact"],
        role="exact_transformed_model_input",
        allowed_root=request_root,
    )
    require(isinstance(transformed, Mapping), "N3 transformed input is not a mapping")
    require(isinstance(transformed.get("video"), list) and len(transformed["video"]) == 1,
            "N3 transformed video structure changed")
    require(isinstance(transformed.get("image_size"), list) and len(transformed["image_size"]) == 1,
            "N3 transformed image_size structure changed")
    observed_video = transformed["video"][0]
    observed_size = transformed["image_size"][0]
    require(list(observed_video.shape) == runtime["models"]["N3"]["transformed_video_shape"]
            and observed_video.dtype == torch.uint8, "N3 transformed video shape/dtype changed")
    require(list(observed_size.shape) == [1, 4]
            and observed_size.dtype == torch.float32
            and observed_size[0].tolist() == runtime["models"]["N3"]["transformed_image_size"],
            "N3 image_size changed")
    base = torch.zeros((3, 33, 540, 640), dtype=torch.uint8)
    base[:, 0] = torch.from_numpy(mosaic.copy()).permute(2, 0, 1)
    expected_video = tv_functional.pad(base, [0, 0, 96, 4], padding_mode="reflect")[None]
    require(torch.equal(observed_video, expected_video), "N3 transformed model pixels changed")
    latent, latent_files = load_n3_artifact(
        request["retained_latent_artifact"], role="retained_vision_latent", allowed_root=request_root
    )
    require(list(latent.shape) == runtime["models"]["N3"]["latent_shape"],
            "N3 latent geometry changed")
    require(runtime["models"]["N3"]["spatial_factor"] == 16
            and list(latent.shape[-2:]) == [33, 40], "N3 latent crop/factor changed")
    decoded, decoded_files = load_n3_artifact(
        request["decoded_future_artifact"], role="offline_decoded_future", allowed_root=request_root
    )
    decoded = np.asarray(decoded)
    require(list(decoded.shape) == runtime["models"]["N3"]["decoded_shape"]
            and decoded.dtype == np.uint8, "N3 decoded geometry changed")
    generated = decoded[:, 360:528, 0:320, :].copy()
    original = mosaic[360:528, 0:320, :].copy()
    require(generated.shape == (33, 168, 320, 3) and original.shape == (168, 320, 3),
            "N3 selected crop geometry changed")
    dependencies = _source_dependencies(runtime, "N3")
    runtime_dependencies = _runtime_dependencies("N3", runtime)
    retained.update({
        "n3_qualification": q_identity,
        "n3_exact_wire_artifacts": exact_wire_files,
        "n3_transformed_input_artifacts": transformed_files,
        "n3_latent_artifacts": latent_files,
        "n3_decoded_artifacts": decoded_files,
    })
    return _final_contract(
        model="N3",
        runtime=runtime,
        original_frame=raw["image_obs/over_shoulder_left_camera"],
        original_crop=original,
        generated_crop=generated,
        output_root=output_root,
        retained=retained,
        dependencies=dependencies,
        runtime_dependencies=runtime_dependencies,
        replay_checks={
            "raw_to_openpi_intermediates_byte_exact": True,
            "raw_to_wire_byte_exact": True,
            "wire_to_transformed_model_input_byte_exact": True,
            "transformed_image_size_exact": True,
            "spatial_factor_and_latent_crop_exact": True,
            "decoded_canvas_exact": True,
            "selected_half_open_crop_exact": True,
            "checkpoint_full_payload_rehash_exact": True,
        },
    )


def _d1_witness(runtime: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    bundle, retained, capture_artifacts = _capture_bundle(runtime)
    raw = bundle["raw"]
    intermediate = bundle["intermediate"]
    dependency_source = (
        Path(runtime["paths"]["robolab_root"]) / "robolab/core/utils/image_utils.py"
    )
    dependency_identity = file_identity(
        dependency_source,
        label="RoboLab resize dependency",
        expected_sha256=runtime["dependencies"]["robolab"]["robolab/core/utils/image_utils.py"],
        within=Path(runtime["paths"]["robolab_root"]),
    )
    provisional = {"dependency_chain": {"robolab_resize_with_pad_source": dependency_identity}}
    camera_keys = {
        "left": "over_shoulder_left_camera",
        "right": "over_shoulder_right_camera",
        "wrist": "wrist_cam",
    }
    stages = {
        role: _d1_replay_stages(raw[f"image_obs/{camera}"], provisional)
        for role, camera in camera_keys.items()
    }
    for role, key in (
        ("left", "D1/right_image"),
        ("right", "D1/right_image_2"),
        ("wrist", "D1/wrist_image"),
    ):
        require(key in intermediate, f"retained D1 intermediate is missing: {key}")
        _same_array(raw[f"image_obs/{camera_keys[role]}"], intermediate[key],
                    f"D1 {role} extraction intermediate")
    wire, wire_identity = _load_capture_npz(
        capture_artifacts.get("D1", {}),
        label="D1 official wire observation",
        expected=runtime["retained_inputs"]["d1_wire"],
    )
    wire_keys = {
        "left": "observation/exterior_image_0_left",
        "right": "observation/exterior_image_1_left",
        "wrist": "observation/wrist_image_left",
    }
    for role, key in wire_keys.items():
        _same_array(stages[role]["wire"], wire[key], f"D1 {role} raw-to-wire replay")
    retained["d1_wire"] = wire_identity
    episode_path = Path(runtime["paths"]["d1_episode_manifest"])
    expected_episode = runtime["retained_inputs"]["d1_episode_manifest"]
    episode_identity = file_identity(
        episode_path,
        label="D1 selected episode manifest",
        expected_path=episode_path,
        expected_sha256=expected_episode["sha256"],
        expected_bytes=expected_episode["bytes"],
    )
    episode = load_json(episode_path, "D1 selected episode manifest")
    require(episode.get("schema_version") == "wmf-d1-episode-manifest-v1"
            and episode.get("status") == "complete"
            and episode.get("episode_id") == "left_decode"
            and episode.get("request_count") == 1,
            "D1 selected episode changed")
    request = episode["requests"][0]
    require(request.get("schema_version") == "wmf-d1-request-receipt-v1"
            and request.get("offline_decode", {}).get("performed") is True,
            "D1 selected request/decode changed")
    request_root = episode_path.parent / "request_0000"
    secure_path(request_root, label="D1 selected request root", kind="dir", expected=request_root)
    raw_map, raw_files = _load_d1_mapping(
        request["raw_inputs"], role="D1 raw inputs", allowed_root=request_root
    )
    converted, converted_files = _load_d1_mapping(
        request["converted_inputs"], role="D1 converted inputs", allowed_root=request_root
    )
    normalized, normalized_files = _load_d1_mapping(
        request["normalized_model_inputs"], role="D1 normalized inputs", allowed_root=request_root
    )
    for role, wire_key in wire_keys.items():
        _same_array(raw_map[wire_key], wire[wire_key], f"D1 retained raw request {role}")
    converted_keys = {
        "left": "video.exterior_image_1_left",
        "right": "video.exterior_image_2_left",
        "wrist": "video.wrist_image_left",
    }
    for role, key in converted_keys.items():
        _same_array(stages[role]["converted"], converted[key], f"D1 converted {role}")
    expected_canvas = np.zeros((1, 1, 352, 640, 3), dtype=np.uint8)
    expected_canvas[0, :, :176, :, :] = np.repeat(
        stages["wrist"]["normalized"], 2, axis=2
    )
    expected_canvas[0, :, 176:, :320, :] = stages["left"]["normalized"]
    expected_canvas[0, :, 176:, 320:, :] = stages["right"]["normalized"]
    images = normalized.get("images")
    require(torch.is_tensor(images) and list(images.shape) == runtime["models"]["D1"]["normalized_images_shape"]
            and images.dtype == torch.uint8, "D1 normalized image structure changed")
    _same_array(images.cpu().numpy(), expected_canvas, "D1 normalized model pixels")
    latent = request.get("latent_video")
    require(isinstance(latent, Mapping) and latent.get("shape") == runtime["models"]["D1"]["latent_shape"],
            "D1 latent geometry changed")
    latent_path = Path(str(latent.get("path", "")))
    latent_identity = file_identity(
        latent_path,
        label="D1 retained latent",
        expected_path=latent_path,
        expected_sha256=str(latent.get("file_sha256")),
        expected_bytes=int(latent.get("bytes", -1)),
        within=request_root,
    )
    with latent_path.open("rb") as stream:
        latent_value = torch.load(stream, map_location="cpu", weights_only=True)
    require(list(latent_value.shape) == runtime["models"]["D1"]["latent_shape"],
            "D1 retained latent file shape changed")
    decoded_descriptor = request["offline_decode"]["decoded_rgb"]
    decoded_path = Path(str(decoded_descriptor.get("path", "")))
    decoded_identity = file_identity(
        decoded_path,
        label="D1 decoded RGB",
        expected_path=decoded_path,
        expected_sha256=str(decoded_descriptor.get("file_sha256")),
        expected_bytes=int(decoded_descriptor.get("bytes", -1)),
        within=request_root,
    )
    with decoded_path.open("rb") as stream:
        decoded = np.load(stream, allow_pickle=False)
    require(array_data_sha256(decoded) == decoded_descriptor.get("data_sha256"),
            "D1 decoded RGB data hash changed")
    require(list(decoded.shape) == runtime["models"]["D1"]["decoded_shape"]
            and decoded.dtype == np.uint8, "D1 decoded geometry changed")
    generated = decoded[:, 176:352, 0:320, :].copy()
    original = stages["left"]["normalized"][0].copy()
    dependencies = _source_dependencies(runtime, "D1")
    dependencies["robolab_resize_with_pad_source"] = dependency_identity
    runtime_dependencies = _runtime_dependencies("D1", runtime)
    retained.update({
        "d1_episode_manifest": episode_identity,
        "d1_raw_input_artifacts": raw_files,
        "d1_converted_input_artifacts": converted_files,
        "d1_normalized_input_artifacts": normalized_files,
        "d1_latent_artifact": latent_identity,
        "d1_decoded_artifact": decoded_identity,
    })
    return _final_contract(
        model="D1",
        runtime=runtime,
        original_frame=raw["image_obs/over_shoulder_left_camera"],
        original_crop=original,
        generated_crop=generated,
        output_root=output_root,
        retained=retained,
        dependencies=dependencies,
        runtime_dependencies=runtime_dependencies,
        replay_checks={
            "raw_to_extraction_intermediates_byte_exact": True,
            "raw_to_wire_byte_exact": True,
            "wire_to_converted_input_byte_exact": True,
            "converted_through_eval_transform_byte_exact": True,
            "normalized_dreamzero_mosaic_byte_exact": True,
            "torchvision_center_crop_rounding_exact": True,
            "latent_geometry_exact": True,
            "decoded_canvas_exact": True,
            "selected_half_open_crop_exact": True,
            "checkpoint_full_payload_rehash_exact": True,
        },
    )


def _immutable_numpy(path: Path, value: Any) -> dict[str, Any]:
    import numpy as np

    target = Path(path)
    require(not target.exists() and not target.is_symlink(), f"refusing to replace output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    require(not target.parent.is_symlink(), "output parent is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.save(stream, np.ascontiguousarray(value), allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        **file_identity(target, label=f"output {target.name}"),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "data_sha256": array_data_sha256(value),
    }


def _final_contract(
    *,
    model: str,
    runtime: Mapping[str, Any],
    original_frame: Any,
    original_crop: Any,
    generated_crop: Any,
    output_root: Path,
    retained: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    runtime_dependencies: Mapping[str, Any],
    replay_checks: Mapping[str, bool],
) -> dict[str, Any]:
    model_contract = runtime["models"][model]
    root = Path(output_root)
    require(root.is_absolute(), "witness output root is not absolute")
    root.mkdir(parents=True, exist_ok=False)
    original_artifact = _immutable_numpy(root / "original_camera_crop.npy", original_crop)
    generated_artifact = _immutable_numpy(root / "generated_decoded_crop.npy", generated_crop)
    generated_spec = model_contract["generated_crop"]
    result = signed_document(
        {
            "schema_version": SCHEMA,
            "study_id": STUDY_ID,
            "contract_id": model_contract["camera_crop_id"],
            "model_id": model,
            "status": "qualified_from_original_camera_pixels",
            "camera_id": CAMERA_ID,
            "camera_crop_id": model_contract["camera_crop_id"],
            "crop_operation": _expected_crop_operation(model),
            "image_width_px": 320,
            "image_height_px": model_contract["original_output_shape"][0],
            "generated_decoded_crop": {
                **generated_spec,
                "selected_camera": CAMERA_ID,
                "output_value_sha256": array_data_sha256(generated_crop),
                "witness_artifact": generated_artifact,
            },
            "original_camera_replay": {
                "source_camera_shape": model_contract["source_camera_shape"],
                "transform_chain": _expected_transform_chain(model),
                "output_shape": model_contract["original_output_shape"],
                "witness_input_value_sha256": array_data_sha256(original_frame),
                "output_value_sha256": array_data_sha256(original_crop),
                "witness_artifact": original_artifact,
            },
            "dependency_chain": dependencies,
            "runtime_dependencies": runtime_dependencies,
            "pinned_dependencies": runtime["dependencies"],
            "retained_artifacts": dict(retained),
            "replay_checks": dict(replay_checks),
            "science_counts": dict(ZERO_SCIENCE_COUNTS),
            "simulator_state_render_used": False,
            "whole_frame_identity": False,
            "safe_to_release_confirmation": False,
            "confirmation_released": False,
            "behavioral_policy_skill_evaluated": False,
            "claim_boundary": (
                "CPU-only byte replay of one retained original camera frame through the exact "
                "active preprocessing and one retained decode crop. No model request, simulator "
                "render, behavior, label, physical-time claim, or confirmation release."
            ),
        }
    )
    validate_camera_crop_contract(result, expected_model=model)
    return result


def _safe_path_list(value: str | None) -> list[dict[str, Any]]:
    """Describe local import roots without echoing non-path environment values."""

    if value is None:
        return []
    rows: list[dict[str, Any]] = []
    for item in value.split(os.pathsep):
        if item and Path(item).is_absolute() and not any(char in item for char in "\r\n\0"):
            rows.append({"absolute_path": item})
        else:
            payload = item.encode("utf-8", errors="surrogateescape")
            rows.append({
                "redacted_nonabsolute_value_bytes": len(payload),
                "redacted_nonabsolute_value_sha256": sha256_bytes(payload),
            })
    return rows


def emit_runtime_preflight(model: str, *, phase: str = "before_replay") -> None:
    """Emit one publish-safe diagnostic line before any heavyweight replay import."""

    require(model in {"N3", "D1"}, "model must be N3 or D1")
    require(phase in {"before_replay", "after_replay_failure"},
            "runtime diagnostic phase changed")
    modules = ["numpy", "torch", "torchvision", "PIL"]
    if model == "N3":
        modules.append("openpi_client.image_tools")
    specs: dict[str, Any] = {}
    for name in modules:
        try:
            specification = importlib.util.find_spec(name)
            specs[name] = {
                "found": specification is not None,
                "origin": None if specification is None else specification.origin,
                "search_locations": (
                    []
                    if specification is None or specification.submodule_search_locations is None
                    else list(specification.submodule_search_locations)
                ),
            }
        except BaseException as error:
            specs[name] = {
                "found": False,
                "lookup_error_type": type(error).__name__,
                "lookup_error_detail": str(error)[:500],
            }
    executable = Path(sys.executable)
    event = {
        "schema_version": "wmf-camera-crop-child-runtime-preflight-v1",
        "event": "runtime_preflight",
        "phase": phase,
        "model_id": model,
        "python": {
            "lexical_path": str(executable),
            "resolved_path": str(executable.resolve()),
        },
        "module_specs": specs,
        "path_environment": {
            "PYTHONPATH": _safe_path_list(os.environ.get("PYTHONPATH")),
            "PYTHONHOME": _safe_path_list(os.environ.get("PYTHONHOME")),
            "VIRTUAL_ENV": _safe_path_list(os.environ.get("VIRTUAL_ENV")),
        },
        "cuda_visible_devices_empty": os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "argv_published": False,
        "secret_environment_published": False,
    }
    sys.stdout.buffer.write(compact_bytes(event) + b"\n")
    sys.stdout.buffer.flush()


def run_witness(model: str, runtime_contract_path: Path, output_root: Path) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "witness is not explicitly CPU-only")
    runtime = load_json(Path(runtime_contract_path), "camera witness runtime contract")
    require(runtime.get("schema_version") == RUNTIME_SCHEMA
            and runtime.get("study_id") == STUDY_ID, "runtime contract schema/study changed")
    require(runtime.get("science_counts") == ZERO_SCIENCE_COUNTS,
            "runtime contract science counts changed")
    require(runtime.get("simulator_state_render_used") is False
            and runtime.get("whole_frame_identity") is False
            and runtime.get("safe_to_release_confirmation") is False
            and runtime.get("confirmation_released") is False,
            "runtime contract authority boundary changed")
    require(model in {"N3", "D1"}, "model must be N3 or D1")
    return _n3_witness(runtime, output_root) if model == "N3" else _d1_witness(runtime, output_root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    witness = sub.add_parser("witness", help="replay one retained per-model witness")
    witness.add_argument("--model", choices=("N3", "D1"), required=True)
    witness.add_argument("--runtime-contract", type=Path, required=True)
    witness.add_argument("--output-root", type=Path, required=True)
    witness.add_argument("--output-contract", type=Path, required=True)
    validate = sub.add_parser("validate", help="validate one emitted signed contract")
    validate.add_argument("--contract", type=Path, required=True)
    validate.add_argument("--model", choices=("N3", "D1"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        value = load_json(args.contract, "camera crop contract")
        validate_camera_crop_contract(value, args.model)
        return 0
    emit_runtime_preflight(args.model)
    try:
        result = run_witness(args.model, args.runtime_contract, args.output_root)
    except BaseException:
        emit_runtime_preflight(args.model, phase="after_replay_failure")
        raise
    target = Path(args.output_contract)
    require(target.parent == Path(args.output_root).parent, "contract output parent changed")
    require(not target.exists() and not target.is_symlink(), "refusing to replace contract output")
    with target.open("xb") as stream:
        stream.write(pretty_bytes(result))
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
