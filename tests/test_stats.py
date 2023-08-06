import numpy as np
import pytest
from scipy.stats import gaussian_kde
from snippets.stats import GaussianKernelDensity, evaluate_bounded_kde_logpdf


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("bounds", [False, True])
def test_bounded_kernel_density(n_features: int, bounds: bool) -> None:
    x = np.random.normal(0, 1, (100, n_features))
    bounds = [(0, 1) for _ in range(n_features)] if bounds else None
    estimator = GaussianKernelDensity(bounds=bounds)
    estimator.fit(x)
    scores = estimator.score_samples(x)
    assert scores.shape == (100,)
    assert np.isscalar(estimator.score(x))
    assert estimator.bandwidth_factor_ > 0


@pytest.mark.parametrize("n_features", [1, 2, 3])
def test_evaluate_bounded_kde_logpdf(n_features: int) -> None:
    x = np.random.normal(0, 1, (n_features, 100))
    estimator = gaussian_kde(x)
    bounds = [(0, 1) for _ in range(n_features)]
    scores = evaluate_bounded_kde_logpdf(estimator, x, bounds)
    assert scores.shape == (100,)

    # Test single sample evaluation.
    x = np.random.uniform(0, 1, n_features)
    assert evaluate_bounded_kde_logpdf(estimator, x, bounds).shape == (1,)
