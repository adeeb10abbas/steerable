import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from tools.audit_sgw_historical_lineage import audit


ROOT = Path(__file__).resolve().parents[1]
INFRA = ROOT / "artifacts/workshops/spatial_grounding_v1/infrastructure"
EXPORT = INFRA / "historical-lineage-20260923bv"
PRIOR = INFRA / "historical-state-fields-20260923bl"
INPUTS = ROOT / "handoff/k8s/sgw01-ali-historical-lineage-inputs-20260923bv.json"


@pytest.fixture(scope="module", autouse=True)
def require_historical_git_objects():
    for path in sorted((EXPORT / "selected").glob("*-state-fields.json")):
        value = json.loads(path.read_text())
        source = next(item["value"] for item in value["source_lineage"]
                      if item["json_pointer"] == "/execution_evidence/construction_source")
        result = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-e", source["study_commit"] + "^{commit}"],
            capture_output=True,
        )
        if result.returncode:
            pytest.skip("Source-backed audit requires the recorded historical Git objects, not a shallow checkout")


def test_real_lineage_matches_historical_git_without_releasing_coverage():
    result = audit(ROOT, EXPORT, PRIOR, INPUTS)
    assert len(result["records"]) == 7
    assert sum(row["frame_record_count"] for row in result["records"]) == 100
    assert all(row["prior_selected_fields_unchanged"] for row in result["records"])
    assert all(row["all_frame_identity_checks_passed"] for row in result["records"])
    assert all(row["observed_scene_environment_origins_world_m"] == [(0, 0, 0)]
               for row in result["records"])
    assert result["historical_population_coverage_complete"] is False
    assert result["release_permitted"] is False


@pytest.mark.parametrize("mutation,message", [
    ("old_field", "prior fresh_reset_objects changed"),
    ("producer", "construction_source differs"),
    ("path", "export path escapes"),
])
def test_mutated_evidence_is_rejected(tmp_path, mutation, message):
    target = tmp_path / "export"
    shutil.copytree(EXPORT, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    row = manifest["records"][0]
    selected_path = target / row["export_path"]
    value = json.loads(selected_path.read_text())
    if mutation == "old_field":
        value["fresh_reset_objects"][0]["value"]["position_world_m"][0] += 1
    elif mutation == "producer":
        source = next(item["value"] for item in value["source_lineage"]
                      if item["json_pointer"] == "/execution_evidence/construction_source")
        source["sha256"] = "0" * 64
    else:
        row["export_path"] = "../outside.json"
    raw = (json.dumps(value, sort_keys=True) + "\n").encode()
    selected_path.write_bytes(raw)
    row["export_bytes"] = len(raw)
    row["export_sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        audit(ROOT, target, PRIOR, INPUTS)


def test_named_root_proof_covers_every_pair_without_claiming_population_coverage():
    from tools.prove_sgw_observed_root_nonmatches import compile_proof

    result = compile_proof()
    assert result["historical_snapshot_count"] == 100
    assert len(result["candidates"]) == 4
    for candidate in result["candidates"]:
        assert candidate["comparison_count"] == candidate["nonmatch_count"] == 100
        assert candidate["unresolved_count"] == 0
        assert all(max(row["max_component_differences_m"].values()) > 0.003
                   for row in candidate["comparisons"])
    assert result["historical_population_coverage_complete"] is False
    assert result["release_permitted"] is False


@pytest.mark.parametrize("mutation,message", [
    ("frame", "native frame differs"),
    ("duplicate", "candidate inventory differs"),
])
def test_named_root_proof_rejects_native_frame_drift(tmp_path, monkeypatch, mutation, message):
    from tools import prove_sgw_observed_root_nonmatches as module

    target = tmp_path / "frames"
    shutil.copytree(module.FRAMES, target)
    path = target / "manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "frame":
        manifest["common_frame_id"] = "unverified_robot_base"
    else:
        manifest["records"][0] = manifest["records"][1]
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(module, "FRAMES", target)
    with pytest.raises(ValueError, match=message):
        module.compile_proof()
