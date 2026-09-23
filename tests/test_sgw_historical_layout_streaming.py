import hashlib
import json
from pathlib import Path

import pytest

from experiments.workshops.spatial_grounding_v1.historical_layout_streaming import (
    extract_state_payload,
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
    payload = {**_payload(), "unselected_tail": "a" * 150000}
    digest, size = _write(path, payload)
    mutated = bytearray(path.read_bytes())
    mutated[mutated.rfind(b"a")] = ord("b")
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


def test_exact_paths_nested_maps_and_dotted_keys_are_preserved(tmp_path):
    payload = _payload()
    stage = payload["attempts"][0]["stages"].pop("canonical_carry")
    payload["attempts"][0]["stages"]["a.b/~"] = stage
    obj = stage["fresh_reset"]["objects"]["a/b~c"]
    obj["unselected"] = {"position_world_m": [99, 99, 99], "deep": {"quaternion_world_wxyz": [9]}}
    obj["quaternion_world_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    obj["linear_velocity_m_s"] = [[0.1, 0.2], [0.3, 0.4]]
    payload["unrelated"] = {
        "fresh_reset": {"e004_full_reset_comparison": {"smuggled": True}},
        "last_reference_bounds_evidence": {"smuggled": True},
        "geometry_identity_sha256": "not a preflight identity",
    }
    path = tmp_path / "state.json"
    digest, size = _write(path, payload)
    result = extract_state_payload(path, expected_sha256=digest, expected_bytes=size)
    row = result["fresh_reset_objects"][0]
    assert row["json_pointer"] == "/attempts/0/stages/a.b~1~0/fresh_reset/objects/a~1b~0c"
    assert row["value"] == {key: value for key, value in obj.items() if key != "unselected"}
    assert len(result["full_reset_comparisons"]) == 2
    assert len(result["reference_bounds"]) == len(result["geometry_preflight_identity"]) == 1


def test_only_the_parsed_stream_is_opened_and_hashed(tmp_path, monkeypatch):
    path = tmp_path / "state.json"
    digest, size = _write(path, _payload())
    original_open = Path.open
    opens = []

    def observed_open(self, *args, **kwargs):
        if self == path:
            opens.append(args)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observed_open)
    result = extract_state_payload(path, expected_sha256=digest, expected_bytes=size)
    assert result["source_sha256"] == digest
    assert opens == [("rb",)]


def test_source_defined_solve_environment_and_diagnostic_resets(tmp_path):
    fresh = _payload()["attempts"][0]["stages"]["canonical_carry"]["fresh_reset"]
    payload = {
        "attempts": [{"stages": {"carry": {"ik_solve_environment": {"fresh_reset": fresh}}}}],
        "known_reachable_diagnostics": [{"fresh_reset": fresh}, {"fresh_reset": fresh}],
    }
    path = tmp_path / "state.json"
    digest, size = _write(path, payload)
    result = extract_state_payload(path, expected_sha256=digest, expected_bytes=size)
    assert [row["json_pointer"] for row in result["fresh_reset_objects"]] == [
        "/attempts/0/stages/carry/ik_solve_environment/fresh_reset/objects/a~1b~0c",
        "/known_reachable_diagnostics/0/fresh_reset/objects/a~1b~0c",
        "/known_reachable_diagnostics/1/fresh_reset/objects/a~1b~0c",
    ]
    assert len(result["full_reset_comparisons"]) == 3


@pytest.mark.parametrize("limits,reason", [
    ({"max_total_bytes": 1}, "aggregate-byte"),
    ({"max_records": 1}, "record limit"),
    ({"max_geometry_identity_scalars": 1}, "identity scalar"),
])
def test_all_retained_groups_share_global_limits(tmp_path, limits, reason):
    payload = _payload()
    payload["geometry_attachment_preflight"]["geometry_attachment_preflight_contract_sha256"] = "b" * 64
    path = tmp_path / "state.json"
    digest, size = _write(path, payload)
    with pytest.raises(ValueError, match=reason):
        extract_state_payload(path, expected_sha256=digest, expected_bytes=size, limits=limits)


def test_reject_before_building_large_selected_array(tmp_path, monkeypatch):
    from experiments.workshops.spatial_grounding_v1 import historical_layout_streaming as module

    payload = _payload()
    payload["attempts"][0]["stages"]["canonical_carry"]["fresh_reset"]["objects"]["a/b~c"]["position_world_m"] = list(range(10000))
    path = tmp_path / "state.json"
    digest, size = _write(path, payload)
    events = []
    builder = module.ObjectBuilder

    class TrackedBuilder(builder):
        def event(self, event, value):
            events.append(event)
            return super().event(event, value)

    monkeypatch.setattr(module, "ObjectBuilder", TrackedBuilder)
    with pytest.raises(ValueError, match="retained-byte"):
        extract_state_payload(path, expected_sha256=digest, expected_bytes=size, limits={"max_object_bytes": 32})
    assert len(events) < 12
