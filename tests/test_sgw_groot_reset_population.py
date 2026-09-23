import shutil
import subprocess

import pytest

from tools import audit_sgw_groot_reset_population as module


@pytest.fixture(autouse=True)
def historical_source():
    if subprocess.run(["git", "cat-file", "-e", module.REVISION + "^{commit}"], capture_output=True).returncode:
        pytest.skip("Reset population audit requires the recorded historical Git revision")


def test_complete_final_cell_selection_is_not_global_coverage():
    result = module.compile_audit()
    assert result["final_manifest_episode_count"] == 54
    assert result["retained_behavioral_initial_resets"] == result["retained_preinference_warmup_resets"] == 54
    assert result["retained_snapshot_count"] == 108
    assert result["distinct_declared_cube_bowl_root_pairs"] == 1
    assert result["historical_runtime_patch_independently_hash_anchored"] is False
    assert result["proved_historical_nonmatches_added"] == 0
    assert result["historical_population_coverage_complete"] is result["release_permitted"] is False


@pytest.mark.parametrize("name", ["resets.json", "runtime/robolab_bridge_runtime.py", "runtime/launcher_manifest.json"])
def test_changed_recovered_artifacts_fail_closed(tmp_path, monkeypatch, name):
    shutil.copytree(module.EXPORT, tmp_path / "export")
    monkeypatch.setattr(module, "EXPORT", tmp_path / "export")
    path = module.EXPORT / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mismatch|differs"):
        module.compile_audit()


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "source", "postaction", "frame", "vector", "attestation"])
def test_recovery_rejects_incomplete_or_reinterpreted_records(monkeypatch, mutation):
    read = module.read_bound

    def altered(path, binding):
        value = read(path, binding)
        if path == module.EXPORT / "resets.json":
            if mutation == "missing":
                value["records"].pop()
            elif mutation == "duplicate":
                value["records"][1] = value["records"][0]
            elif mutation == "source":
                value["records"][0]["state_trace"]["sha256"] = "0" * 64
            elif mutation == "postaction":
                value["records"][0]["episode_initial"]["action_step"] = 1
            elif mutation == "frame":
                value["records"][0]["capture_contract"]["coordinates"] = "world"
            elif mutation == "vector":
                value["records"][0]["episode_initial"]["object_xyz"] = [1, 2]
            else:
                value["retained_runtime_sources"][0]["historical_hash_anchor_in_final_manifest"] = True
        return value

    monkeypatch.setattr(module, "read_bound", altered)
    with pytest.raises(ValueError):
        module.compile_audit()
