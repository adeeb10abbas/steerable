"""Bounded, hash-verified extraction for large E006 state-repair JSON files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import ijson
from ijson.common import ObjectBuilder


DEFAULT_LIMITS = {
    "max_object_bytes": 128 * 1024,
    "max_comparison_bytes": 256 * 1024,
    "max_bounds_bytes": 256 * 1024,
    "max_geometry_identity_scalars": 64,
}


def _pointer(path: str, index: int | None = None) -> str:
    return "/" + "/".join(path.split(".")) + (f"/{index}" if index is not None else "")


def _items(path: Path, prefix: str) -> Iterable[Any]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, prefix)


def _bounded_json(value: Any, limit: int, label: str) -> Any:
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if len(encoded) > limit:
        raise ValueError(f"{label} exceeds retained-byte limit")
    return value


def verify_source(path: Path, expected_sha256: str, expected_bytes: int) -> str:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    actual = digest.hexdigest()
    if size != expected_bytes:
        raise ValueError(f"source size mismatch: expected {expected_bytes}, got {size}")
    if actual != expected_sha256:
        raise ValueError(f"source sha256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def _stream_selected(path: Path, limits: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    """Use ijson's tokenizer/builder while retaining only selected subtrees."""

    targets = {
        "fresh_reset_objects": (".fresh_reset.objects.", limits["max_object_bytes"]),
        "candidate_state_objects": (".candidate_state.objects.", limits["max_object_bytes"]),
        "full_reset_comparisons": (".fresh_reset.e004_full_reset_comparison", limits["max_comparison_bytes"]),
        "reference_bounds": (".last_reference_bounds_evidence", limits["max_bounds_bytes"]),
    }
    result = {name: [] for name in targets}
    active: dict[str, tuple[ObjectBuilder, int, str, str]] | None = None
    attempt_index = -1
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle):
            if prefix == "attempts.item" and event == "start_map":
                attempt_index += 1
            if active is None and event in {"start_map", "start_array"}:
                for name, (suffix, limit) in targets.items():
                    if (suffix in prefix if name.endswith("objects") else prefix.endswith(suffix)) and (
                        name.endswith("objects") or prefix.endswith("e004_full_reset_comparison")
                        or prefix.endswith("last_reference_bounds_evidence")
                    ):
                        builder = ObjectBuilder()
                        builder.event(event, value)
                        active = (builder, 1, name, prefix)
                        break
            elif active is not None:
                builder, depth, name, root_prefix = active
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    depth += 1
                elif event in {"end_map", "end_array"}:
                    depth -= 1
                if depth == 0:
                    item = builder.value
                    encoded = json.dumps(item, separators=(",", ":"), allow_nan=False).encode()
                    limit = targets[name][1]
                    if len(encoded) > limit:
                        raise ValueError(f"{name.replace('_', ' ')} exceeds retained-byte limit")
                    pointer = "/" + root_prefix.replace(".item", "").replace(".", "/")
                    pointer = pointer.replace("/attempts/stages/", f"/attempts/{attempt_index}/stages/")
                    result[name].append({"json_pointer": pointer, "value": item})
                    active = None
    return result


def extract_state_payload(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Hash the entire source, then retain only bounded producer-defined fields."""

    active = {**DEFAULT_LIMITS, **(limits or {})}
    verify_source(path, expected_sha256, expected_bytes)
    selected = _stream_selected(path, active)
    fresh = selected["fresh_reset_objects"]
    candidates = selected["candidate_state_objects"]
    comparisons = selected["full_reset_comparisons"]
    bounds = selected["reference_bounds"]
    identity = []
    for prefix in (
        "geometry_attachment_preflight.geometry_attachment_preflight_contract_sha256",
        "geometry_attachment_preflight.geometry_identity_sha256",
        "geometry_attachment_preflight.collision_geometry_resolution.geometry_identity_sha256",
    ):
        for value in _items(path, prefix):
            if len(identity) >= active["max_geometry_identity_scalars"]:
                raise ValueError("geometry identity scalar limit exceeded")
            identity.append({"json_pointer": _pointer(prefix), "value": value})
    missing = []
    for label, rows in (
        ("fresh_reset_objects", fresh),
        ("candidate_state_objects", candidates),
        ("full_reset_comparisons", comparisons),
        ("reference_bounds", bounds),
    ):
        if not rows:
            missing.append(label)
    return {
        "source_sha256": expected_sha256,
        "source_bytes": expected_bytes,
        "fresh_reset_objects": fresh,
        "candidate_state_objects": candidates,
        "full_reset_comparisons": comparisons,
        "reference_bounds": bounds,
        "geometry_preflight_identity": identity,
        "missing_pointer_groups": missing,
        "comparable_geometry_rows": [],
        "semantic_status": "root_only_or_preflight_evidence_not_comparable_geometry",
    }
