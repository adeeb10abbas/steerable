import ast
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import subprocess

import numpy as np
import pytest

from tools import prove_sgw_pi05_root_separation as proof


def test_named_root_separations_keep_native_and_population_limitations():
    result = proof.compile_proof()
    assert result["comparison_count"] == result["nonmatches_under_registered_api_contract"] == 108
    assert result["unresolved_count"] == 0
    assert result["distinct_historical_numerical_root_pairs"] == 2
    assert result["release_permitted"] is False
    assert result["historical_population_coverage_complete"] is False
    assert result["historical_native_import_bytes_independently_attested"] is False
    assert 0 < result["frames"]["historical_root_error_m"] < 0.001
    assert result["frames"]["prospective_root_error_m"] == 0
    for row in result["comparisons"]:
        assert row["separation_difference_lower_bound_m"] > row["necessary_match_distance_bound_m"]


@pytest.mark.parametrize("value", [[], [[True, 0, 0]], [[math.nan, 0, 0]], [[math.inf, 0, 0]], [[1, 0.1, 0]], [[0, 0]]])
def test_roundoff_certificate_rejects_invalid_or_out_of_domain_roots(value):
    with pytest.raises(ValueError):
        proof.rotation_roundoff_bound(value)


def test_certificate_is_outward_rounded_and_includes_subnormal_flush():
    result = proof.rotation_roundoff_bound([[1, 0, 0]])
    exact = Fraction(*(int(x) for x in result["exact_error_bound_fraction"]))
    observed = Fraction.from_float(result["per_root_euclidean_error_upper_bound_m"])
    assert observed >= exact
    assert 0.00067 < float(exact) < 0.00068
    zero = proof.rotation_roundoff_bound([[0, 0, 0]])
    assert zero["per_root_euclidean_error_upper_bound_m"] >= 3 * 2**-14


def _historical_rotation():
    source = subprocess.check_output([
        "git", "-C", str(proof.ROOT), "show",
        "636a33eedb9e4a9920f4ff357388099fba78a108:experiments/v3/pi05_phase_b/robolab_bridge.py",
    ])
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "_inverse_rotate")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(module, "<hash-bound inverse rotation>", "exec"), namespace)
    return namespace["_inverse_rotate"]


def _decimal_rotation(q, vector):
    q = [Decimal.from_float(float(x)) for x in q]
    vector = [Decimal.from_float(float(x)) for x in vector]
    norm = sum(x * x for x in q).sqrt()
    q = [x / norm for x in q]
    w, inverse = q[0], [-x for x in q[1:]]
    dot = sum(x * y for x, y in zip(inverse, vector, strict=True))
    square = sum(x * x for x in inverse)
    cross = [inverse[1] * vector[2] - inverse[2] * vector[1],
             inverse[2] * vector[0] - inverse[0] * vector[2],
             inverse[0] * vector[1] - inverse[1] * vector[0]]
    return [2 * dot * inverse[i] + (w * w - square) * vector[i] + 2 * w * cross[i] for i in range(3)]


@pytest.mark.parametrize("dtype", [np.float16, np.float32])
def test_actual_frozen_rotation_with_cancellation_and_scaled_quaternions(dtype):
    rotate = _historical_rotation()
    cases = [
        ([0.303, 0.124, 0.081], [0, 0, 0]),
        ([2048.25, -2047.75, 2048.125], [2048, -2048, 2048]),
        ([0.25, 0.125, 0.0625], [-0.125, 0, -0.03125]),
        ([2**-20, -(2**-20), 0], [0, 0, 0]),
    ]
    quaternions = (
        [1, 0, 0, 0], [0.5, 0.5, 0.5, 0.5], [0, 1, 1, -1],
        [1e-30, -2e-30, 3e-30, 4e-30], [1e30, 2e30, -3e30, 4e30],
        [2**-149, 0, 0, 0],
        [np.finfo(np.float32).max, 2**-149, -np.finfo(np.float32).max, 0],
    )
    with localcontext() as context:
        context.prec = 100
        for position, origin in cases:
            p, o = np.asarray(position, dtype=dtype), np.asarray(origin, dtype=dtype)
            true_difference = p.astype(np.float64) - o.astype(np.float64)
            rounded = p - o
            for values in quaternions:
                q = np.asarray(values, dtype=np.float32)
                observed = rotate(q, rounded)
                certificate = proof.rotation_roundoff_bound([observed.tolist()])
                reference = _decimal_rotation(q, true_difference)
                error = sum((Decimal.from_float(float(x)) - y) ** 2 for x, y in zip(observed, reference, strict=True)).sqrt()
                assert error <= Decimal.from_float(certificate["per_root_euclidean_error_upper_bound_m"])
                rounded_reference = _decimal_rotation(q, rounded)
                rotation_error = sum(
                    (Decimal.from_float(float(x)) - y) ** 2
                    for x, y in zip(observed, rounded_reference, strict=True)
                ).sqrt()
                norm = sum(Decimal.from_float(float(x)) ** 2 for x in rounded).sqrt()
                assert rotation_error <= Decimal.from_float(
                    certificate["rotation_relative_error_upper_bound"]
                ) * norm


def test_changed_native_contract_cannot_silently_qualify(monkeypatch):
    original = proof.read_bound

    def altered(path, binding):
        value = original(path, binding)
        if path.name == "native_contract.json":
            value["stage_units_in_meters"] = 0.01
        return value

    monkeypatch.setattr(proof, "read_bound", altered)
    with pytest.raises(ValueError, match="metre contract"):
        proof.compile_proof()
