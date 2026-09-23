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
    "max_records": 4096,
    "max_total_bytes": 8 * 1024 * 1024,
    "max_geometry_identity_scalars": 64,
}
OBJECT_FIELDS = {
    "position_world_m",
    "quaternion_world_wxyz",
    "linear_velocity_m_s",
    "angular_velocity_rad_s",
}


def _items(path: Path, prefix: str) -> Iterable[Any]:
    with path.open("rb") as handle:
        yield from ijson.items(handle, prefix, use_float=True)


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


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _pointer(prefix: str, indices: dict[str, int]) -> str:
    parts = prefix.split(".")
    out = []
    for i, part in enumerate(parts):
        if part == "item":
            out.append(str(indices[".".join(parts[:i])]))
        else:
            out.append(_escape(part))
    return "/" + "/".join(out)


def _item_index(prefix: str, event: str, indices: dict[str, int]) -> None:
    if event in {"map_key", "end_map", "end_array"} or not prefix.endswith(".item"):
        return
    base = prefix.rsplit(".item", 1)[0]
    indices[base] = indices.get(base, -1) + 1


def _match_object(parts: list[str]) -> bool:
    return (
        len(parts) == 7
        and parts[0:3] == ["attempts", "item", "stages"]
        and parts[4:6] in (["fresh_reset", "objects"], ["candidate_state", "objects"])
        and parts[2] == "stages"
    )


def _object_kind(parts: list[str]) -> str | None:
    if len(parts) != 7 or parts[0:3] != ["attempts", "item", "stages"]:
        return None
    if parts[4:6] == ["fresh_reset", "objects"]:
        return "fresh_reset_objects"
    if parts[4:6] == ["candidate_state", "objects"]:
        return "candidate_state_objects"
    return None


def _stream_selected(path: Path, limits: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    result = {
        "fresh_reset_objects": [],
        "candidate_state_objects": [],
        "full_reset_comparisons": [],
        "reference_bounds": [],
    }
    indices: dict[str, int] = {}
    active: dict[str, Any] | None = None
    total_bytes = 0
    identity: list[dict[str, Any]] = []
    with path.open("rb") as handle:
        for prefix, event, value in ijson.parse(handle, use_float=True):
            _item_index(prefix, event, indices)
            parts = prefix.split(".")
            kind = _object_kind(parts) if event == "start_map" else None
            if active is None and kind:
                active = {
                    "kind": kind,
                    "pointer": _pointer(prefix, indices),
                    "fields": {},
                    "field_values": None,
                    "bytes": 0,
                }
                continue
            if active is None and event == "start_map" and (
                prefix.endswith(".fresh_reset.e004_full_reset_comparison")
                or prefix.endswith(".last_reference_bounds_evidence")
            ):
                kind = "full_reset_comparisons" if prefix.endswith("comparison") else "reference_bounds"
                active = {
                    "kind": kind,
                    "pointer": _pointer(prefix, indices),
                    "builder": ObjectBuilder(),
                    "depth": 1,
                    "bytes": 0,
                }
                active["builder"].event(event, value)
                continue
            if active is not None:
                if active["kind"].endswith("objects"):
                    if event == "map_key":
                        active["field"] = value if value in OBJECT_FIELDS else None
                        active["field_values"] = [] if active["field"] is not None else None
                    elif active.get("field") is not None:
                        active["bytes"] += len(repr(value).encode())
                        if active["bytes"] > limits["max_object_bytes"]:
                            raise ValueError(f"{active['kind'].replace('_', ' ')} exceeds retained-byte limit")
                        if event in {"number", "string", "boolean", "null"}:
                            if active["field_values"] is not None:
                                active["field_values"].append(value)
                            else:
                                active["fields"][active["field"]] = value
                        elif event == "end_array":
                            active["fields"][active["field"]] = active["field_values"]
                            active["field_values"] = None
                            active["field"] = None
                    if event == "end_map":
                        value_out = active["fields"]
                        if len(result[active["kind"]]) >= limits["max_records"]:
                            raise ValueError("retained record limit exceeded")
                        total_bytes += len(json.dumps(value_out, allow_nan=False).encode())
                        if total_bytes > limits["max_total_bytes"]:
                            raise ValueError("retained aggregate-byte limit exceeded")
                        result[active["kind"]].append(
                            {"json_pointer": active["pointer"], "value": value_out}
                        )
                        active = None
                else:
                    encoded_size = len(repr(value).encode())
                    active["bytes"] += encoded_size
                    if active["bytes"] > limits[
                        "max_comparison_bytes" if active["kind"] == "full_reset_comparisons" else "max_bounds_bytes"
                    ]:
                        raise ValueError(f"{active['kind'].replace('_', ' ')} exceeds retained-byte limit")
                    active["builder"].event(event, value)
                    if event in {"start_map", "start_array"}:
                        active["depth"] += 1
                    elif event in {"end_map", "end_array"}:
                        active["depth"] -= 1
                    if active["depth"] == 0:
                        value_out = active["builder"].value
                        total_bytes += len(json.dumps(value_out, allow_nan=False).encode())
                        if total_bytes > limits["max_total_bytes"]:
                            raise ValueError("retained aggregate-byte limit exceeded")
                        result[active["kind"]].append(
                            {"json_pointer": active["pointer"], "value": value_out}
                        )
                        active = None
                continue
            if event in {"string", "number", "boolean"} and (
                prefix.endswith("geometry_identity_sha256")
                or prefix.endswith("geometry_attachment_preflight_contract_sha256")
            ):
                if len(identity) >= limits["max_geometry_identity_scalars"]:
                    raise ValueError("geometry identity scalar limit exceeded")
                identity.append({"json_pointer": _pointer(prefix, indices), "value": value})
    if active is not None:
        raise ValueError("unterminated selected subtree")
    result["geometry_preflight_identity"] = identity
    return result


def extract_state_payload(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Stream/hash the complete source and publish only bounded selected fields."""

    active = {**DEFAULT_LIMITS, **(limits or {})}
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    if size != expected_bytes:
        raise ValueError(f"source size mismatch: expected {expected_bytes}, got {size}")
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f"source sha256 mismatch: expected {expected_sha256}, got {digest.hexdigest()}")
    selected = _stream_selected(path, active)
    missing = [
        name for name in (
            "fresh_reset_objects",
            "candidate_state_objects",
            "full_reset_comparisons",
            "reference_bounds",
        ) if not selected[name]
    ]
    return {
        "source_sha256": expected_sha256,
        "source_bytes": expected_bytes,
        **selected,
        "missing_pointer_groups": missing,
        "comparable_geometry_rows": [],
        "semantic_status": "root_only_or_preflight_evidence_not_comparable_geometry",
    }
