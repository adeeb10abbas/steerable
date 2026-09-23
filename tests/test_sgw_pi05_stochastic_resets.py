from copy import deepcopy
import json

import pytest

from tools import audit_sgw_pi05_stochastic_resets as audit


def test_complete_final_reset_selection_does_not_invent_a_producer_anchor():
    result = audit.compile_audit()
    assert result["final_cell_count"] == 432
    assert result["registered_conditions"] == 54
    assert result["original_reset_attestations_reconstructed"] == 864
    assert result["distinct_numerical_root_pairs"] == result["distinct_initial_state_hashes"] == 1
    assert result["complete_final_manifest_cell_selection"] is True
    assert result["independent_historical_producer_anchor_established"] is False
    assert result["stochastic_bridge_in_recorded_runtime_source_maps"] is False
    assert result["cross_frame_comparison_qualified"] is False
    assert result["proved_historical_nonmatches_added"] == 0
    assert result["historical_population_coverage_complete"] is result["release_permitted"] is False
    assert result["new_model_requests"] == result["new_behavioral_episodes"] == 0


@pytest.mark.parametrize("change", [
    "duplicate_cell", "missing_reset", "altered_sample", "changed_frame", "swapped_binding", "invalid_raw_episode",
])
def test_recovery_projection_cannot_change_registered_history(monkeypatch, change):
    original = audit.read_bound

    def altered(path, binding):
        value = original(path, binding)
        if path.name != "resets.json":
            return value
        value = deepcopy(value)
        rows = value["entries"]
        if change == "duplicate_cell":
            rows[1] = rows[0]
        elif change == "missing_reset":
            rows[0]["reset_attestations"].pop()
        elif change == "altered_sample":
            rows[0]["reset_attestations"][0]["value"]["sample"]["object_xyz"][0] += 0.01
        elif change == "changed_frame":
            rows[0]["measurement_frame"] = "world"
        elif change == "invalid_raw_episode":
            rows[0]["raw_behavioral_result_valid"] = False
        else:
            rows[0]["reset_attestations"][0]["binding"] = rows[1]["reset_attestations"][0]["binding"]
        return value

    monkeypatch.setattr(audit, "read_bound", altered)
    with pytest.raises(ValueError):
        audit.compile_audit()


def test_committed_audit_reproduces_byte_identically():
    expected = (audit.EXPORT / "audit.json").read_bytes()
    actual = (json.dumps(audit.compile_audit(), indent=2, sort_keys=True) + "\n").encode()
    assert actual == expected
