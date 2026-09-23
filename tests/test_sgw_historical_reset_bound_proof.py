import json
import shutil
import subprocess

import pytest

from tools import prove_sgw_r005_reset_bounds as module


@pytest.fixture(scope="module", autouse=True)
def require_historical_git_objects():
    revisions = {module.REVISION}
    for path in (module.INFRA / "historical-lineage-20260923bv/selected").glob("*-state-fields.json"):
        payload = json.loads(path.read_text())
        source = next(row["value"] for row in payload["source_lineage"]
                      if row["json_pointer"] == "/execution_evidence/construction_source")
        revisions.add(source["study_commit"])
    for revision in revisions:
        if subprocess.run(
            ["git", "-C", str(module.ROOT), "cat-file", "-e", revision + "^{commit}"],
            capture_output=True,
        ).returncode:
            pytest.skip("Source-bound proof requires the recorded historical Git objects")


def test_source_bound_reset_exclusions_are_not_measured_points_or_coverage():
    report = module.compile_proof()
    assert report["historical_euclidean_position_bound_m"] == 0.005
    assert report["sgw_componentwise_reset_tolerance_m"] == 0.003
    assert report["comparison_count"] == report["nonmatch_count"] == 80
    assert len(report["completed_fresh_reset_lifecycles"]) == 20
    assert len(report["candidates"]) == 4
    assert all(min(row["minimum_component_separations_m"].values()) > 0.1
               for row in report["candidates"])
    assert report["release_permitted"] is False
    assert report["historical_population_coverage_complete"] is False


@pytest.mark.parametrize("name", ["launch.json", "target-validation.json", "controller-verification.json"])
def test_changed_bound_receipt_is_rejected(tmp_path, monkeypatch, name):
    target = tmp_path / "export"
    shutil.copytree(module.EXPORT, target)
    path = target / name
    path.write_bytes(path.read_bytes() + b"\n")
    monkeypatch.setattr(module, "EXPORT", target)
    with pytest.raises(ValueError, match="retained input byte/hash mismatch"):
        module.compile_proof()


@pytest.mark.parametrize("mutation", ["missing_marker", "false_marker", "duplicate_ordinal", "contradictory_observation"])
def test_source_implication_rejects_incomplete_or_contradictory_evidence(monkeypatch, mutation):
    read_bound = module.read_bound

    def changed_inventory(path, binding):
        value = read_bound(path, binding)
        if path == module.INVENTORY:
            if mutation == "missing_marker":
                del value["environment_lifecycle"][0]["fresh_reset_completed_in_this_environment"]
            elif mutation == "false_marker":
                value["environment_lifecycle"][0]["fresh_reset_completed_in_this_environment"] = False
            elif mutation == "duplicate_ordinal":
                value["environment_lifecycle"][1]["environment_ordinal"] = 1
            else:
                snapshot = next(row for row in value["partial_snapshots"]
                                if row["json_pointer"].endswith("/fresh_reset"))
                snapshot["objects"]["rubiks_cube"]["position_world_m"][0] += 0.1
        return value

    monkeypatch.setattr(module, "read_bound", changed_inventory)
    with pytest.raises(ValueError, match="lifecycle inventory|completion marker|contradicts"):
        module.compile_proof()
