import hashlib
import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.historical_layout_streaming import (
    extract_state_payload,
    verify_source,
)


def _write(path: Path, payload: dict) -> tuple[str, int]:
    data = json.dumps(payload, separators=(",", ":")).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def _payload():
    return {
        "attempts": [
            {"stages": {"canonical_carry": {"fresh_reset": {"objects": {
                "rubiks_cube": {"position_world_m": [1, 2, 3]},
            }, "e004_full_reset_comparison": {"passed": True}},
            "candidate_state": {"objects": {
                "rubiks_cube": {"position_world_m": [4, 5, 6]},
            }}}}},
        ],
        "execution_evidence": {
            "last_reference_bounds_evidence": {"center_world_m": [0, 0, 0]},
        },
        "geometry_attachment_preflight": {
            "geometry_identity_sha256": "a" * 64,
        },
    }


def test_nested_arrays_and_pointer_groups(tmp_path):
    path = tmp_path / "state.json"
    digest, size = _write(path, _payload())
    result = extract_state_payload(path, expected_sha256=digest, expected_bytes=size)
    assert result["fresh_reset_objects"][0]["json_pointer"].startswith(
        "/attempts/0/stages/canonical_carry/fresh_reset/objects/"
    )
    assert result["full_reset_comparisons"][0]["value"]["passed"] is True
    assert result["reference_bounds"][0]["value"]["center_world_m"] == [0, 0, 0]
    assert result["missing_pointer_groups"] == []


def test_tail_mutation_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    digest, size = _write(path, _payload())
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_source(path, digest, size)


def test_wrong_source_and_limits_fail_closed(tmp_path):
    path = tmp_path / "state.json"
    digest, size = _write(path, _payload())
    with pytest.raises(ValueError, match="source sha256 mismatch"):
        extract_state_payload(path, expected_sha256="0" * 64, expected_bytes=size)
    with pytest.raises(ValueError, match="fresh reset objects"):
        extract_state_payload(path, expected_sha256=digest, expected_bytes=size,
                              limits={"max_object_bytes": 1})


def test_missing_pointer_accounting(tmp_path):
    path = tmp_path / "state.json"
    digest, size = _write(path, {"attempts": [], "execution_evidence": {}})
    result = extract_state_payload(path, expected_sha256=digest, expected_bytes=size)
    assert "fresh_reset_objects" in result["missing_pointer_groups"]
    assert "reference_bounds" in result["missing_pointer_groups"]
    assert result["comparable_geometry_rows"] == []
