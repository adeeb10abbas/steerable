import copy
import json
import shutil
import subprocess

import pytest

from tools import audit_sgw_r005_object_inventory as module


@pytest.fixture
def inventory():
    return json.loads((module.EXPORT / "inventory.json").read_bytes())


@pytest.fixture
def historical_source():
    if subprocess.run(["git", "cat-file", "-e", module.REVISION + "^{commit}"], capture_output=True).returncode:
        pytest.skip("Source-bound inventory audit requires the historical Git revision")


def test_all_twenty_configs_reconstruct_exactly(inventory):
    base = (module.EXPORT / "base_env_cfg.json").read_bytes()
    configs = [module.reconstruct_config(base, row) for row in inventory["records"]]
    assert len(configs) == 20
    assert len({config["recorders"]["dataset_export_dir_path"] for config in configs}) == 20
    assert all(len(config["scene"]) == 37 for config in configs)


@pytest.mark.parametrize("start,end", [(-1, 2), (True, 2), (5, 2), (0, 999999)])
def test_invalid_delta_bounds_rejected(inventory, start, end):
    row = copy.deepcopy(inventory["records"][0])
    row["config_byte_delta"].update(base_start_inclusive=start, base_end_exclusive=end)
    with pytest.raises(ValueError, match="byte-delta bounds"):
        module.reconstruct_config((module.EXPORT / "base_env_cfg.json").read_bytes(), row)


@pytest.mark.parametrize("mutation", ["replacement", "hash", "bytes"])
def test_modified_delta_or_binding_rejected(inventory, mutation):
    row = copy.deepcopy(inventory["records"][0])
    if mutation == "replacement":
        row["config_byte_delta"]["replacement_base64"] = "eA=="
    elif mutation == "hash":
        row["source_binding"]["sha256"] = "0" * 64
    else:
        row["source_binding"]["bytes"] += 1
    with pytest.raises(ValueError, match="config byte/hash mismatch"):
        module.reconstruct_config((module.EXPORT / "base_env_cfg.json").read_bytes(), row)


def test_complete_config_recovery_does_not_release_exclusion_or_poses(historical_source):
    report = module.compile_audit()
    assert report["hash_verified_native_configs"] == report["unique_complete_configs"] == 20
    assert report["source_config_bytes_verified"] == 1031942
    assert report["registered_rigid_objects"] == list(module.ACTORS)
    assert report["native_config_has_registered_plate"] is False
    assert report["dist_no_plate_exclusion_status"] == "unresolved_not_excluded"
    assert report["observed_pose_rows_added"] == 0
    assert report["missing_materialization_states_recovered"] is False
    assert report["release_permitted"] is report["historical_population_coverage_complete"] is False


@pytest.mark.parametrize("name", ["inventory.json", "base_env_cfg.json"])
def test_changed_retained_artifact_rejected(tmp_path, monkeypatch, historical_source, name):
    shutil.copytree(module.EXPORT, tmp_path / "export")
    monkeypatch.setattr(module, "EXPORT", tmp_path / "export")
    path = module.EXPORT / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte/hash mismatch"):
        module.compile_audit()


@pytest.mark.parametrize("mutation", ["missing", "ordinal", "lifecycle", "reset", "horizon", "reference"])
def test_lifecycle_completeness_and_scoring_identity_fail_closed(monkeypatch, historical_source, mutation):
    read = module.read_bound

    def altered(path, binding):
        value = read(path, binding)
        if path.name == "inventory.json" and path.parent == module.EXPORT:
            if mutation == "missing":
                value["records"].pop()
            elif mutation == "ordinal":
                value["records"][1]["environment_ordinal"] = 1
            elif mutation == "lifecycle":
                value["records"][0]["label"] = "different"
        elif path == module.INVENTORY:
            life = value["environment_lifecycle"][0]
            if mutation == "reset":
                life["fresh_reset_completed_in_this_environment"] = False
            elif mutation == "horizon":
                life["construction_horizon_activation"]["only_mutated_field"] = "scene"
            elif mutation == "reference":
                life["construction_horizon_activation"]["termination_contract_after"]["termination_config"] = {}
        return value

    monkeypatch.setattr(module, "read_bound", altered)
    with pytest.raises(ValueError):
        module.compile_audit()
