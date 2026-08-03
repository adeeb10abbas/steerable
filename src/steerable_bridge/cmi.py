from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import Literal

import numpy as np


CovarianceKind = Literal["diagonal", "full"]


@dataclass(frozen=True)
class CMIResult:
    """Finite-sample estimate of I(A; L | S=s) for one fixed state.

    ``null_centered_score_bits`` subtracts the permutation-null median from the
    raw estimate. It is an effect-size diagnostic, not mutual information, and
    therefore does not inherit the prompt-entropy bound.
    """

    cmi_bits: float
    null_centered_score_bits: float
    pooled_entropy_bits: float
    conditional_entropy_bits: float
    null_mean_bits: float
    null_median_bits: float
    null_p95_bits: float
    permutation_p_value: float
    bootstrap_low_bits: float
    bootstrap_high_bits: float
    n_prompts: int
    samples_per_prompt: int
    action_dim: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cmi_bits": self.cmi_bits,
            "null_centered_score_bits": self.null_centered_score_bits,
            "pooled_entropy_bits": self.pooled_entropy_bits,
            "conditional_entropy_bits": self.conditional_entropy_bits,
            "null_mean_bits": self.null_mean_bits,
            "null_median_bits": self.null_median_bits,
            "null_p95_bits": self.null_p95_bits,
            "permutation_p_value": self.permutation_p_value,
            "bootstrap_low_bits": self.bootstrap_low_bits,
            "bootstrap_high_bits": self.bootstrap_high_bits,
            "n_prompts": self.n_prompts,
            "samples_per_prompt": self.samples_per_prompt,
            "action_dim": self.action_dim,
        }


def _validate_action_samples(action_samples: np.ndarray) -> np.ndarray:
    values = np.asarray(action_samples, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError(
            "action_samples must have shape (prompts, samples_per_prompt, action_dim)"
        )
    n_prompts, samples_per_prompt, action_dim = values.shape
    if n_prompts < 2:
        raise ValueError("at least two prompts are required")
    if samples_per_prompt < 2:
        raise ValueError("at least two action samples per prompt are required")
    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if not np.isfinite(values).all():
        raise ValueError("action_samples contains NaN or infinity")
    return values


def _standardize_with_pooled_scale(values: np.ndarray) -> np.ndarray:
    """Apply one invertible, prompt-independent transform for numerical stability."""

    pooled = values.reshape(-1, values.shape[-1])
    center = pooled.mean(axis=0)
    scale = pooled.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - center) / scale


def gaussian_entropy_nats(
    samples: np.ndarray,
    *,
    covariance: CovarianceKind = "diagonal",
    ridge: float = 1e-6,
) -> float:
    """Entropy of a Gaussian fitted to rows in ``samples``.

    ``diagonal`` is the stable default for the small Monte Carlo sample sizes used
    in VLA probing. ``full`` retains action correlations but needs substantially
    more samples than action dimensions.
    """

    rows = np.asarray(samples, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or rows.shape[1] < 1:
        raise ValueError("samples must have shape (n >= 2, d >= 1)")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")

    d = rows.shape[1]
    if covariance == "diagonal":
        variance = np.var(rows, axis=0, ddof=1) + ridge
        if np.any(variance <= 0):
            raise ValueError("zero variance requires a positive ridge")
        log_determinant = float(np.log(variance).sum())
    elif covariance == "full":
        covariance_matrix = np.atleast_2d(np.cov(rows, rowvar=False, ddof=1))
        covariance_matrix = covariance_matrix + ridge * np.eye(d)
        sign, log_determinant = np.linalg.slogdet(covariance_matrix)
        if sign <= 0:
            raise ValueError("covariance is not positive definite; increase ridge")
    else:
        raise ValueError(f"unsupported covariance estimator: {covariance}")

    return 0.5 * (d * log(2.0 * np.pi * np.e) + log_determinant)


def _gaussian_cmi_nats(
    standardized_samples: np.ndarray,
    *,
    covariance: CovarianceKind,
    ridge: float,
) -> tuple[float, float, float]:
    pooled = standardized_samples.reshape(-1, standardized_samples.shape[-1])
    pooled_entropy = gaussian_entropy_nats(pooled, covariance=covariance, ridge=ridge)
    conditional_entropies = [
        gaussian_entropy_nats(prompt_samples, covariance=covariance, ridge=ridge)
        for prompt_samples in standardized_samples
    ]
    conditional_entropy = float(np.mean(conditional_entropies))
    return pooled_entropy - conditional_entropy, pooled_entropy, conditional_entropy


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def _kde_cmi_nats(
    standardized_samples: np.ndarray,
    *,
    bandwidth: float,
) -> tuple[float, float, float]:
    """Leave-one-out KDE estimate via I(A;L)=E[log p(L|A)-log p(L)]."""

    n_prompts, samples_per_prompt, action_dim = standardized_samples.shape
    log_normalizer = -action_dim * log(bandwidth) - 0.5 * action_dim * log(2.0 * np.pi)
    flat = standardized_samples.reshape(-1, action_dim)
    prompt_labels = np.repeat(np.arange(n_prompts), samples_per_prompt)
    squared_distance = np.square(flat[:, None, :] - flat[None, :, :]).sum(axis=-1)
    np.fill_diagonal(squared_distance, np.inf)
    log_kernels = log_normalizer - 0.5 * squared_distance / bandwidth**2
    prompt_log_densities = np.empty((len(flat), n_prompts), dtype=np.float64)
    for candidate_prompt in range(n_prompts):
        start = candidate_prompt * samples_per_prompt
        stop = start + samples_per_prompt
        log_sums = _logsumexp(log_kernels[:, start:stop], axis=1)
        counts = np.where(
            prompt_labels == candidate_prompt,
            samples_per_prompt - 1,
            samples_per_prompt,
        )
        prompt_log_densities[:, candidate_prompt] = log_sums - np.log(counts)
    conditional_log_densities = prompt_log_densities[
        np.arange(len(flat)), prompt_labels
    ]
    pooled_log_densities = _logsumexp(prompt_log_densities, axis=1) - log(n_prompts)
    return (
        float(np.mean(conditional_log_densities - pooled_log_densities)),
        float(-np.mean(pooled_log_densities)),
        float(-np.mean(conditional_log_densities)),
    )


def estimate_kde_cmi(
    action_samples: np.ndarray,
    *,
    bandwidth: float | None = None,
    permutations: int = 200,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> CMIResult:
    """Estimate CMI with shared-bandwidth, leave-one-out Gaussian KDE.

    Expressing the estimate through the KDE prompt posterior ensures the result
    cannot exceed the entropy of the uniform prompt prior (``log2(n_prompts)``).
    The default bandwidth is Scott's rule using the per-prompt sample count after
    a common pooled standardization.
    """

    values = _validate_action_samples(action_samples)
    if permutations < 0 or bootstrap_samples < 0:
        raise ValueError("permutations and bootstrap_samples must be non-negative")
    standardized = _standardize_with_pooled_scale(values)
    shape = standardized.shape
    if bandwidth is None:
        bandwidth = float(shape[1] ** (-1.0 / (shape[2] + 4.0)))
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be finite and positive")

    observed, pooled_entropy, conditional_entropy = _kde_cmi_nats(
        standardized, bandwidth=bandwidth
    )
    prompt_entropy = log(shape[0])
    if observed > prompt_entropy + 1e-12:
        raise AssertionError("KDE CMI estimate exceeded the prompt-entropy ceiling")
    rng = np.random.default_rng(seed)
    flat = standardized.reshape(-1, shape[-1])
    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = flat[rng.permutation(len(flat))].reshape(shape)
        null[index] = _kde_cmi_nats(shuffled, bandwidth=bandwidth)[0]

    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample_indices = rng.integers(0, shape[1], size=(shape[0], shape[1]))
        resampled = np.stack(
            [standardized[prompt, sample_indices[prompt]] for prompt in range(shape[0])]
        )
        bootstrap[index] = _kde_cmi_nats(resampled, bandwidth=bandwidth)[0]

    nats_to_bits = 1.0 / log(2.0)
    if permutations:
        null_mean = float(null.mean())
        null_median = float(np.median(null))
        null_p95 = float(np.quantile(null, 0.95))
        p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    else:
        null_mean = null_median = null_p95 = p_value = float("nan")
    if bootstrap_samples:
        bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.025, 0.975])
    else:
        bootstrap_low = bootstrap_high = float("nan")

    return CMIResult(
        cmi_bits=float(observed * nats_to_bits),
        null_centered_score_bits=float((observed - null_median) * nats_to_bits),
        pooled_entropy_bits=float(pooled_entropy * nats_to_bits),
        conditional_entropy_bits=float(conditional_entropy * nats_to_bits),
        null_mean_bits=float(null_mean * nats_to_bits),
        null_median_bits=float(null_median * nats_to_bits),
        null_p95_bits=float(null_p95 * nats_to_bits),
        permutation_p_value=p_value,
        bootstrap_low_bits=float(bootstrap_low * nats_to_bits),
        bootstrap_high_bits=float(bootstrap_high * nats_to_bits),
        n_prompts=shape[0],
        samples_per_prompt=shape[1],
        action_dim=shape[2],
    )


def estimate_gaussian_cmi(
    action_samples: np.ndarray,
    *,
    covariance: CovarianceKind = "diagonal",
    ridge: float = 1e-6,
    permutations: int = 200,
    bootstrap_samples: int = 200,
    seed: int = 0,
) -> CMIResult:
    """Estimate CMI and a prompt-label permutation null for one fixed state.

    The prompt prior is uniform. Samples are standardized with one pooled affine
    transform before entropy estimation; exact CMI is invariant to this common
    transform, while the scaling makes the ridge dimensionless and reproducible.
    """

    values = _validate_action_samples(action_samples)
    if permutations < 0 or bootstrap_samples < 0:
        raise ValueError("permutations and bootstrap_samples must be non-negative")
    standardized = _standardize_with_pooled_scale(values)
    observed, pooled_entropy, conditional_entropy = _gaussian_cmi_nats(
        standardized, covariance=covariance, ridge=ridge
    )
    rng = np.random.default_rng(seed)
    shape = standardized.shape
    flat = standardized.reshape(-1, shape[-1])

    null = np.empty(permutations, dtype=np.float64)
    for index in range(permutations):
        shuffled = flat[rng.permutation(len(flat))].reshape(shape)
        null[index] = _gaussian_cmi_nats(shuffled, covariance=covariance, ridge=ridge)[
            0
        ]

    bootstrap = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sample_indices = rng.integers(0, shape[1], size=(shape[0], shape[1]))
        resampled = np.stack(
            [standardized[prompt, sample_indices[prompt]] for prompt in range(shape[0])]
        )
        bootstrap[index] = _gaussian_cmi_nats(
            resampled, covariance=covariance, ridge=ridge
        )[0]

    nats_to_bits = 1.0 / log(2.0)
    if permutations:
        null_mean = float(null.mean())
        null_median = float(np.median(null))
        null_p95 = float(np.quantile(null, 0.95))
        p_value = float((1 + np.count_nonzero(null >= observed)) / (permutations + 1))
    else:
        null_mean = null_median = null_p95 = p_value = float("nan")
    if bootstrap_samples:
        bootstrap_low, bootstrap_high = np.quantile(bootstrap, [0.025, 0.975])
    else:
        bootstrap_low = bootstrap_high = float("nan")

    return CMIResult(
        cmi_bits=float(observed * nats_to_bits),
        null_centered_score_bits=float((observed - null_median) * nats_to_bits),
        pooled_entropy_bits=float(pooled_entropy * nats_to_bits),
        conditional_entropy_bits=float(conditional_entropy * nats_to_bits),
        null_mean_bits=float(null_mean * nats_to_bits),
        null_median_bits=float(null_median * nats_to_bits),
        null_p95_bits=float(null_p95 * nats_to_bits),
        permutation_p_value=p_value,
        bootstrap_low_bits=float(bootstrap_low * nats_to_bits),
        bootstrap_high_bits=float(bootstrap_high * nats_to_bits),
        n_prompts=shape[0],
        samples_per_prompt=shape[1],
        action_dim=shape[2],
    )


def categorical_mutual_information_bits(labels: np.ndarray) -> float:
    """Plug-in I(G; L | S=s) for categorical actions such as a gripper command."""

    values = np.asarray(labels)
    if values.ndim != 2:
        raise ValueError("labels must have shape (prompts, samples_per_prompt)")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("at least two prompts and two samples per prompt are required")

    _, encoded = np.unique(values.reshape(-1), return_inverse=True)
    encoded = encoded.reshape(values.shape)

    def entropy_bits(group: np.ndarray) -> float:
        counts = np.bincount(group, minlength=int(encoded.max()) + 1)
        probabilities = counts[counts > 0] / counts.sum()
        return float(-(probabilities * np.log2(probabilities)).sum())

    pooled_entropy = entropy_bits(encoded.reshape(-1))
    conditional_entropy = float(np.mean([entropy_bits(group) for group in encoded]))
    return pooled_entropy - conditional_entropy
