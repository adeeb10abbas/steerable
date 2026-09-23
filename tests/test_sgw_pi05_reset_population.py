from copy import deepcopy
import json

import pytest

from tools import audit_sgw_pi05_reset_population as audit


def test_final_resets_have_historically_anchored_source_without_global_release():
    report = audit.compile_audit()
    assert report["final_cell_count"] == report["original_reset_attestations_reconstructed"] == 108
    assert report["distinct_numerical_root_pairs"] == 2
    source = report["source_contract"]
    assert len(source["source_files"]) == 16
    assert source["historically_hash_bound"] is True
    assert source["adapter_contract_sha256"] == "03c051ed517e8d72660c77a8a0a536a423f6be76e99cba918c94ae43e754a932"
    assert report["historical_population_coverage_complete"] is False
    assert report["cross_frame_comparison_qualified"] is False
    assert report["proved_historical_nonmatches_added"] == 0
    assert report["release_permitted"] is False


def mutate_recovery(monkeypatch, change):
    value = json.loads((audit.EXPORT / "resets.json").read_bytes())
    change(value)
    monkeypatch.setattr(audit, "read_bound", lambda *args: deepcopy(value))


@pytest.mark.parametrize("change", [
    lambda v: v["entries"].pop(),
    lambda v: v["entries"].append(deepcopy(v["entries"][0])),
    lambda v: v["entries"][0].update(arm="unsupported"),
    lambda v: v["entries"][0].update(measurement_frame="world"),
    lambda v: v["entries"][0]["sample"].update(action_step=1),
    lambda v: v["entries"][0]["sample"].update(object_xyz=[0.0, 0.0, 0.0]),
    lambda v: v["entries"][0]["reset_attestation"]["value"].update(physical_reset_calls=2),
    lambda v: v["entries"][0]["reset_attestation"]["value"].update(passed=False),
    lambda v: v["entries"][0]["reset_attestation"]["binding"].update(sha256="0" * 64),
])
def test_mutated_population_points_and_attestations_fail_closed(monkeypatch, change):
    # The outer immutable export binding is bypassed only to exercise deeper checks.
    mutate_recovery(monkeypatch, change)
    with pytest.raises(ValueError):
        audit.compile_audit()


def test_changed_historical_producer_cannot_retain_attested_identity(monkeypatch):
    read = audit._git

    def changed(path):
        value = read(path)
        return value + b"\n" if path == audit.SOURCE else value

    monkeypatch.setattr(audit, "_git", changed)
    with pytest.raises(ValueError, match="adapter sources"):
        audit.compile_audit()
