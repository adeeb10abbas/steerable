"""Bounded extraction of exact fields from hash-verified E006 state JSON."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import ijson
from ijson.common import ObjectBuilder


DEFAULT_LIMITS = {
    "max_object_bytes": 128 * 1024,
    "max_comparison_bytes": 256 * 1024,
    "max_bounds_bytes": 256 * 1024,
    "max_records": 4096,
    "max_total_bytes": 8 * 1024 * 1024,
    "max_geometry_identity_scalars": 64,
}
OBJECT_FIELDS = {
    "position_world_m", "quaternion_world_wxyz",
    "linear_velocity_m_s", "angular_velocity_rad_s",
}
IDENTITY_PATHS = {
    ("geometry_attachment_preflight", "geometry_attachment_preflight_contract_sha256"),
    ("geometry_attachment_preflight", "geometry_identity_sha256"),
    ("geometry_attachment_preflight", "collision_geometry_resolution", "geometry_identity_sha256"),
}
PathTokens = tuple[str | int, ...]


class _HashingReader:
    def __init__(self, stream: BinaryIO, expected_bytes: int) -> None:
        self.stream = stream
        self.expected_bytes = expected_bytes
        self.size = 0
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        raw = self.stream.read(size)
        self.size += len(raw)
        self.digest.update(raw)
        if self.size > self.expected_bytes:
            raise ValueError("source size mismatch: parsed stream exceeds expected bytes")
        return raw

    def readinto(self, buffer: bytearray) -> int:
        raw = self.read(len(buffer))
        buffer[:len(raw)] = raw
        return len(raw)

    def verify(self, expected_sha256: str) -> None:
        if self.size != self.expected_bytes:
            raise ValueError(f"source size mismatch: expected {self.expected_bytes}, got {self.size}")
        if self.digest.hexdigest() != expected_sha256:
            raise ValueError("source sha256 mismatch for the actual parsed stream")


def verify_source(path: Path, expected_sha256: str, expected_bytes: int) -> str:
    with path.open("rb") as stream:
        reader = _HashingReader(stream, expected_bytes)
        while reader.read(1024 * 1024):
            pass
        reader.verify(expected_sha256)
        return reader.digest.hexdigest()


@dataclass
class _Container:
    path: PathTokens
    kind: str
    key: str | None = None
    next_index: int = 0


def _path_events(reader: _HashingReader) -> Iterator[tuple[PathTokens, str, Any]]:
    stack: list[_Container] = []
    started = False
    for event, value in ijson.basic_parse(reader, use_float=True):
        if not started:
            if event != "start_map":
                raise ValueError("state payload must be a JSON object")
            started = True
        if event == "map_key":
            stack[-1].key = value
            yield stack[-1].path, event, value
            continue
        if event in {"end_map", "end_array"}:
            yield stack.pop().path, event, value
            continue
        if not stack:
            path = ()
        elif stack[-1].kind == "map":
            parent = stack[-1]
            if parent.key is None:
                raise ValueError("object value has no JSON key")
            path = parent.path + (parent.key,)
            parent.key = None
        else:
            parent = stack[-1]
            path = parent.path + (parent.next_index,)
            parent.next_index += 1
        if event in {"start_map", "start_array"}:
            if len(stack) >= 128:
                raise ValueError("source nesting exceeds the bounded parser depth")
            stack.append(_Container(path, "map" if event == "start_map" else "array"))
        yield path, event, value


def _pointer(path: PathTokens) -> str:
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in path)


def _selection(path: PathTokens) -> tuple[str, str] | None:
    relative = ()
    if (
        len(path) >= 6 and path[0] == "attempts" and isinstance(path[1], int)
        and path[2] == "stages" and isinstance(path[3], str)
    ):
        relative = path[4:]
        if relative[0] == "ik_solve_environment":
            if relative[1] != "fresh_reset":
                return None
            relative = relative[1:]
    elif len(path) >= 4 and path[0] == "known_reachable_diagnostics" and isinstance(path[1], int):
        relative = path[2:]
        if relative[0] != "fresh_reset":
            return None
    if (
        len(relative) == 4 and relative[1] == "objects"
        and isinstance(relative[2], str) and relative[3] in OBJECT_FIELDS
    ):
        if relative[0] == "fresh_reset":
            return "fresh_reset_objects", "max_object_bytes"
        if relative[0] == "candidate_state":
            return "candidate_state_objects", "max_object_bytes"
    if relative == ("fresh_reset", "e004_full_reset_comparison"):
        return "full_reset_comparisons", "max_comparison_bytes"
    if path == ("execution_evidence", "last_reference_bounds_evidence"):
        return "reference_bounds", "max_bounds_bytes"
    if path in IDENTITY_PATHS:
        return "geometry_preflight_identity", "max_object_bytes"
    return None


@dataclass
class _Selected:
    path: PathTokens
    kind: str
    limit: str
    builder: ObjectBuilder
    charged_bytes: int = 0


def extract_state_payload(
    path: Path, *, expected_sha256: str, expected_bytes: int,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Publish only after EOF verifies the exact bytes consumed by the parser."""
    active_limits = {**DEFAULT_LIMITS, **(limits or {})}
    if set(active_limits) != set(DEFAULT_LIMITS) or any(
        type(value) is not int or value <= 0 for value in active_limits.values()
    ):
        raise ValueError("retention limits must be known positive integers")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("expected source size must be a positive integer")
    result: dict[str, list[dict[str, Any]]] = {
        key: [] for key in ("fresh_reset_objects", "candidate_state_objects",
                           "full_reset_comparisons", "reference_bounds", "geometry_preflight_identity")
    }
    objects: dict[tuple[str, PathTokens], dict[str, Any]] = {}
    object_bytes: dict[tuple[str, PathTokens], int] = {}
    selected: _Selected | None = None
    record_count = total_bytes = 0
    with path.open("rb") as stream:
        reader = _HashingReader(stream, expected_bytes)
        for location, event, value in _path_events(reader):
            if selected is None:
                match = None if event in {"map_key", "end_map", "end_array"} else _selection(location)
                if match is None:
                    continue
                kind, limit = match
                if kind == "geometry_preflight_identity" and event in {"start_map", "start_array"}:
                    raise ValueError("geometry identity must be a bounded scalar")
                if kind == "geometry_preflight_identity" and len(result[kind]) >= active_limits["max_geometry_identity_scalars"]:
                    raise ValueError("geometry identity scalar limit exceeded")
                if not kind.endswith("objects") or (kind, location[:-1]) not in objects:
                    record_count += 1
                    if record_count > active_limits["max_records"]:
                        raise ValueError("retained record limit exceeded")
                selected = _Selected(location, kind, limit, ObjectBuilder())
                total_bytes += len(_pointer(location).encode()) + 64
            cost = len(json.dumps(value, allow_nan=False, ensure_ascii=False).encode()) + 2
            selected.charged_bytes += cost
            total_bytes += cost
            charge = selected.charged_bytes
            if selected.kind.endswith("objects"):
                charge += object_bytes.get((selected.kind, selected.path[:-1]), 0)
            if charge > active_limits[selected.limit]:
                raise ValueError(f"{selected.kind.replace('_', ' ')} exceeds retained-byte limit")
            if total_bytes > active_limits["max_total_bytes"]:
                raise ValueError("retained aggregate-byte limit exceeded")
            selected.builder.event(event, value)
            if location == selected.path and event not in {"start_map", "start_array", "map_key"}:
                if selected.kind.endswith("objects"):
                    key = selected.kind, selected.path[:-1]
                    if key not in objects:
                        objects[key] = {"json_pointer": _pointer(key[1]), "value": {}}
                        result[selected.kind].append(objects[key])
                    field = selected.path[-1]
                    if field in objects[key]["value"]:
                        raise ValueError("duplicate selected object field in source JSON")
                    objects[key]["value"][field] = selected.builder.value
                    object_bytes[key] = charge
                else:
                    result[selected.kind].append({
                        "json_pointer": _pointer(selected.path), "value": selected.builder.value,
                    })
                selected = None
        reader.verify(expected_sha256)
    return {
        "source_sha256": expected_sha256, "source_bytes": expected_bytes,
        "selection_contract": "exact_producer_paths_ijson_same_stream_hash_v1",
        "retention_limits": active_limits,
        **result,
        "missing_pointer_groups": [
            kind for kind in result if kind != "geometry_preflight_identity" and not result[kind]
        ],
        "comparable_geometry_rows": [],
        "semantic_status": "root_only_or_preflight_evidence_not_comparable_geometry",
    }
