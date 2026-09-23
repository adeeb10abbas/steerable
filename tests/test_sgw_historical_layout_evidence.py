from experiments.workshops.spatial_grounding_v1.historical_layout_evidence import (
    PVC_POSE_MANIFESTS,
    SOURCE_SPECS,
    compile_registry,
)
import hashlib
import json
import os
from pathlib import Path
import pytest

from experiments.workshops.spatial_grounding_v1 import historical_layout_evidence as evidence


def test_registry_is_hash_bound_incomplete_and_excludes_wrong_arenas():
    registry = compile_registry()
    assert registry["release_authorization"] is False
    assert registry["coverage_status"].startswith("incomplete_")
    assert "V3-E007" not in {row["source_id"] for row in registry["comparable_pose_rows"]}
    assert any(row["source_id"] == "v4-c8-widowx" for row in registry["sources"])
    assert "SimplerEnv/WidowX" in registry["excluded_arenas"]
    assert len(registry["sources"]) == len(SOURCE_SPECS)
    assert registry["raw_pvc_paths_still_needed"]
    assert len(registry["raw_pvc_paths_still_needed"]) == 24
    assert [row["expected_sha256"] for row in registry["raw_pvc_paths_still_needed"]] == [
        digest for _, _, digest in PVC_POSE_MANIFESTS
    ]


def test_duplicate_semantics_are_source_identity_not_path_only():
    registry = compile_registry()
    keys = [
        (row["commit"], row["path"], row["git_blob_sha1"], row["expected_sha256"])
        for row in registry["sources"]
    ]
    assert len(keys) == len(set(keys))
    assert all(row["purpose_and_limit"] for row in registry["sources"])


def test_available_git_objects_are_verified_without_checkout():
    registry = compile_registry()
    resolved = [row for row in registry["sources"] if row.get("git_object_sha256")]
    assert resolved
    assert all(row["git_object_blob_match"] for row in resolved)
    assert all(row["git_object_sha256"] == row["expected_sha256"] for row in resolved)


@pytest.mark.parametrize("mutation", (None, "rehash", "quaternion", "bounds"))
def test_hash_linked_gate_chain_uses_aabb_and_separates_reset_displacement(tmp_path, monkeypatch, mutation):
    gate_path = "/raw/gate.json"
    pose_doc = {"layout_pairs": {"C01": {"accepted_gate_attempt_receipt": {
        "path": gate_path, "sha256": "PLACEHOLDER"
    }}}}
    capture = {
        "layout_pair_id": "C01",
        "layout_arm": "original",
        "configured_poses": {"rubiks_cube": {
            "position_robot_base_m": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }},
        "settled_poses": {"rubiks_cube": {
            "position_robot_base_m": [0.000001, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        }},
        "cameras": {"head_camera": {"geometry": {
            "object_aabbs_robot_base_m": {
                "rubiks_cube": {"lower": [0.09, -0.01, -0.01], "upper": [0.11, 0.01, 0.01]}
            }
        }}},
    }
    gate_doc = {"evaluation": {"capture_evidence": [capture]}}
    if mutation == "quaternion":
        capture["settled_poses"]["rubiks_cube"]["quaternion_wxyz"] = [2, 0, 0, 0]
    elif mutation == "bounds":
        capture["cameras"]["head_camera"]["geometry"]["object_aabbs_robot_base_m"]["rubiks_cube"]["lower"][0] = 0.2
    gate_text = json.dumps(gate_doc)
    gate_sha = hashlib.sha256(gate_text.encode()).hexdigest()
    pose_doc["layout_pairs"]["C01"]["accepted_gate_attempt_receipt"]["sha256"] = gate_sha
    pose_doc["layout_pairs"]["C01"]["accepted_gate_attempt_receipt"]["bytes"] = len(gate_text.encode())
    pose_text = json.dumps(pose_doc)
    monkeypatch.setattr(evidence, "PVC_POSE_MANIFESTS", (
        ("C01", "pose.json", hashlib.sha256(pose_text.encode()).hexdigest()),
    ))
    if mutation == "rehash":
        pose_text += "\n"
    pose_export = tmp_path / "poses.json"
    gate_export = tmp_path / "gates.json"
    pose_export.write_text(json.dumps({"files": {"pose.json": {
        "bytes": len(pose_text.encode()), "sha256": hashlib.sha256(pose_text.encode()).hexdigest(),
        "text": pose_text
    }}}))
    gate_export.write_text(json.dumps({"files": {gate_path: {
        "bytes": len(gate_text.encode()), "sha256": gate_sha, "text": gate_text
    }}}))
    registry = compile_registry(pose_export=pose_export, gate_export=gate_export)
    if mutation:
        assert registry["comparable_pose_rows"] == []
        assert registry["structured_blockers"]
        return
    assert len(registry["comparable_pose_rows"]) == 1
    row = registry["comparable_pose_rows"][0]
    assert row["pose_manifest"]["receipt_sha256"] == row["gate_receipt"]["sha256"]
    assert row["reset_displacement_m"][0] == 0.000001
    assert row["root_local_geometric_offset_m"][0] > 0.09
    assert "AABB" in row["semantic_basis"]


def test_real_exports_are_optional_and_hash_checked():
    pose = os.environ.get("SGW_HISTORICAL_POSE_EXPORT")
    gate = os.environ.get("SGW_HISTORICAL_GATE_EXPORT")
    if not pose or not gate:
        pytest.skip("requires retained hash-bound historical pose and gate exports")
    registry = compile_registry(pose_export=Path(pose), gate_export=Path(gate))
    assert len(registry["comparable_pose_rows"]) == 144


def test_export_mutation_is_rejected(tmp_path):
    export = tmp_path / "mutated.json"
    export.write_text(json.dumps({"files": {"x": {
        "bytes": 3, "sha256": hashlib.sha256(b"abc").hexdigest(), "text": "abd"
    }}}))
    registry = compile_registry(pose_export=export, gate_export=export)
    assert registry["comparable_pose_rows"] == []
    assert any("failed its declared" in item["reason"] for item in registry["structured_blockers"])
