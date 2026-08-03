from __future__ import annotations

import numpy as np
import pytest

from steerable_bridge.cmi import (
    categorical_mutual_information_bits,
    estimate_gaussian_cmi,
    estimate_kde_cmi,
    gaussian_entropy_nats,
)


def test_shifted_prompt_distributions_clear_permutation_null() -> None:
    rng = np.random.default_rng(7)
    samples = np.stack(
        [rng.normal(loc=mean, scale=0.25, size=(80, 3)) for mean in (-2.0, 0.0, 2.0)]
    )
    result = estimate_kde_cmi(
        samples,
        permutations=99,
        bootstrap_samples=30,
        seed=11,
    )

    assert result.cmi_bits > result.null_p95_bits
    assert result.null_centered_score_bits > 1.0
    assert result.permutation_p_value <= 0.02
    assert result.cmi_bits <= np.log2(samples.shape[0])
    assert result.null_centered_score_bits > result.cmi_bits


def test_prompt_independent_samples_do_not_clear_null() -> None:
    rng = np.random.default_rng(13)
    samples = rng.normal(size=(4, 100, 2))
    result = estimate_kde_cmi(
        samples,
        permutations=199,
        bootstrap_samples=0,
        seed=17,
    )

    assert result.permutation_p_value > 0.05
    assert result.cmi_bits < result.null_p95_bits


def test_cmi_is_invariant_to_common_per_dimension_units() -> None:
    rng = np.random.default_rng(23)
    samples = np.stack([rng.normal(loc=mean, size=(60, 2)) for mean in (-1.0, 1.0)])
    scaled = samples * np.array([1000.0, 0.001]) + np.array([20.0, -30.0])

    original = estimate_kde_cmi(samples, permutations=0, bootstrap_samples=0, seed=3)
    transformed = estimate_kde_cmi(scaled, permutations=0, bootstrap_samples=0, seed=3)

    assert transformed.cmi_bits == pytest.approx(original.cmi_bits, abs=1e-10)


def test_discrete_gripper_mutual_information() -> None:
    dependent = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
    independent = np.array([[0, 1, 0, 1], [0, 1, 0, 1]])

    assert categorical_mutual_information_bits(dependent) == pytest.approx(1.0)
    assert categorical_mutual_information_bits(independent) == pytest.approx(0.0)


def test_entropy_requires_positive_ridge_for_constant_samples() -> None:
    with pytest.raises(ValueError, match="positive ridge"):
        gaussian_entropy_nats(np.ones((4, 2)), ridge=0.0)


def test_action_sample_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="prompts"):
        estimate_gaussian_cmi(np.zeros((10, 7)), permutations=0, bootstrap_samples=0)
