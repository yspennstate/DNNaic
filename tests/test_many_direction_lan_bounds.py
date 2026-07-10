"""Numerical checks of the Gaussian identities used in the many-direction theorem."""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def bayes_scores(X: np.ndarray, means: np.ndarray, inverse_covariance: np.ndarray) -> np.ndarray:
    return (
        X @ inverse_covariance @ means.T
        - 0.5 * np.einsum("ki,ij,kj->k", means, inverse_covariance, means)
    )


def test_pairwise_competitor_event_has_q_of_half_mahalanobis_distance():
    rng = np.random.default_rng(20260710)
    base = rng.normal(size=(5, 5))
    covariance = base @ base.T + np.eye(5)
    inverse = np.linalg.inv(covariance)
    means = rng.normal(size=(4, 5))
    true_class = 0
    n = 250_000
    X = rng.multivariate_normal(means[true_class], covariance, size=n)
    scores = bayes_scores(X, means, inverse)
    for competitor in range(1, len(means)):
        delta = means[competitor] - means[true_class]
        distance = float(np.sqrt(delta @ inverse @ delta))
        analytic = float(ndtr(-distance / 2.0))
        empirical = float((scores[:, competitor] >= scores[:, true_class]).mean())
        assert abs(empirical - analytic) < 0.004


def test_multiclass_error_is_between_largest_pair_event_and_their_sum():
    rng = np.random.default_rng(17)
    covariance = np.array([
        [1.0, 0.2, 0.0],
        [0.2, 1.4, 0.1],
        [0.0, 0.1, 0.8],
    ])
    inverse = np.linalg.inv(covariance)
    means = rng.normal(scale=1.2, size=(6, 3))
    true_class = 2
    X = rng.multivariate_normal(means[true_class], covariance, size=300_000)
    scores = bayes_scores(X, means, inverse)
    error = float((scores.argmax(axis=1) != true_class).mean())
    pair_errors = []
    for competitor in range(len(means)):
        if competitor == true_class:
            continue
        delta = means[competitor] - means[true_class]
        distance = float(np.sqrt(delta @ inverse @ delta))
        pair_errors.append(float(ndtr(-distance / 2.0)))
    assert max(pair_errors) - 0.004 <= error
    assert error <= min(1.0, sum(pair_errors)) + 0.004
