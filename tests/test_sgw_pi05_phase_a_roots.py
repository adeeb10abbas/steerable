import ast
from copy import deepcopy
import json

import numpy as np
import pytest

import test_sgw_pi05_root_separation as numerical_tests
from tools import prove_sgw_pi05_phase_a_roots as proof


def test_actual_runtime_anchor_and_two_recorded_resets_remain_bounded():
    value = proof.compile_proof()
    assert value["source_contract"]["historically_hash_bound"] is True
    assert value["source_contract"]["runtime_instrumentation"]["sha256"] == proof.BRIDGE_SHA
    assert value["source_contract"]["root_getter_and_rotation_ast_match_registered_source"] is True
    assert value["final_cell_count"] == value["recorded_setup_reset_samples"] == 54
    assert value["recorded_behavioral_initial_samples"] == 54
    assert value["comparison_count"] == value["nonmatches_under_registered_api_contract"] == 108
    assert value["distinct_numerical_root_pairs"] == 1
    assert value["unresolved_count"] == 0
    assert value["historical_native_import_bytes_independently_attested"] is False
    assert value["historical_population_coverage_complete"] is value["release_permitted"] is False
    assert value["new_model_requests"] == value["new_behavioral_episodes"] == 0


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_actual_attested_phase_a_rotation_satisfies_decimal_certificate(monkeypatch, dtype):
    source = (proof.EXPORT / "runtime_bridge.py").read_bytes()
    assert proof.sha(source) == proof.BRIDGE_SHA
    node = proof.function(ast.parse(source), "_quat_inverse_rotate_wxyz")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<attested Phase-A rotation>", "exec"), namespace)
    monkeypatch.setattr(numerical_tests, "_historical_rotation", lambda: namespace["_quat_inverse_rotate_wxyz"])
    numerical_tests.test_actual_frozen_rotation_with_cancellation_and_scaled_quaternions(dtype)


@pytest.mark.parametrize("change", ["duplicate", "setup", "initial", "launch", "frame"])
def test_source_or_reset_projection_changes_fail_closed(monkeypatch, change):
    read = proof.read_bound

    def altered(path, binding):
        value = read(path, binding)
        if path.name != "resets.json":
            return value
        value = deepcopy(value)
        pair = value["pairs"][0]
        row = pair["entries"][0]
        if change == "duplicate":
            pair["entries"][1] = row
        elif change == "setup":
            row["setup_sample"]["object_xyz"][0] += 0.01
        elif change == "initial":
            row["initial_sample"]["object_xyz"][0] += 0.01
        elif change == "launch":
            command = pair["launch_event"]["command"]
            command[command.index("/data/users/ali/vla_wam/envs/robolab-v2-isaac50/bin/python") + 1] = "unbound.py"
        else:
            row["measurement_frame"] = "world"
        return value

    monkeypatch.setattr(proof, "read_bound", altered)
    with pytest.raises(ValueError):
        proof.compile_proof()


def test_committed_phase_a_proof_reproduces():
    assert (proof.EXPORT / "proof.json").read_bytes() == (
        json.dumps(proof.compile_proof(), indent=2, sort_keys=True) + "\n"
    ).encode()
