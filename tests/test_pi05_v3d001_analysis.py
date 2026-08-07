from experiments.v3.pi05_stochastic_v3d001.analyze_results import (
    BOOTSTRAP_REPLICATES,
    _bootstrap_mean,
    _exact_sign_test,
    _wilson,
)


def test_cluster_bootstrap_is_deterministic_and_bounded() -> None:
    first = _bootstrap_mean([0.0, 0.5, 1.0], label="unit-test")
    second = _bootstrap_mean([0.0, 0.5, 1.0], label="unit-test")
    assert first == second
    assert first["replicates"] == BOOTSTRAP_REPLICATES
    assert first["unit_of_resampling"] == "environment_seed"
    assert first["lower"] <= 0.5 <= first["upper"]


def test_exact_sign_test_excludes_ties() -> None:
    result = _exact_sign_test([1.0, 2.0, 0.0, -1.0])
    assert result == {
        "effective_n": 3,
        "method": "exact_two_sided_paired_sign_test_zero_ties_excluded",
        "negative": 1,
        "p_value": 1.0,
        "positive": 2,
        "ties": 1,
    }


def test_wilson_zero_successes_remains_in_unit_interval() -> None:
    result = _wilson(0, 27)
    assert result["lower"] == 0.0
    assert 0.0 < result["upper"] < 1.0
