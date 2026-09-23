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
                "a/b~c": {"position_world_m": [1.25, 2.5, 3.75]},
            }, "e004_full_reset_comparison": {"passed": True}},
            "candidate_state": {"objects": {
                "rubiks_cube": {"position_world_m": [4, 5, 6]},
            }}}}},
            {"stages": {"canonical_carry": {"fresh_reset": {"objects": {
                "second": {"position_world_m": [7.125, 8.25, 9.5]},
            }, "e004_full_reset_comparison": {"passed": True}},
            "candidate_state": {"objects": {}}}}},
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
    assert any(
        row["json_pointer"].endswith("/a~1b~0c")
        for row in result["fresh_reset_objects"]
    )
    assert any(
        row["json_pointer"].startswith("/attempts/1/")
        for row in result["fresh_reset_objects"]
    )
    assert result["full_reset_comparisons"][0]["value"]["passed"] is True
    assert result["reference_bounds"][0]["value"]["center_world_m"] == [0, 0, 0]
    assert result["missing_pointer_groups"] == []


def test_tail_mutation_is_rejected(tmp_path):
    path = tmp_path / "state.json"
    digest, size = _write(path, _payload())
    mutated = bytearray(path.read_bytes())
    mutated[-1] = ord(" ")
    path.write_bytes(mutated)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        extract_state_payload(path, expected_sha256=digest, expected_bytes=size)


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
