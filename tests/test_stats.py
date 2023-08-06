import numpy as np
import pytest
from sklearn.neighbors import KernelDensity
from snippets.stats import BoundedKernelDensity


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("bounds", [False, True])
def test_bounded_kernel_density(n_features: int, bounds: bool) -> None:
    x = np.random.normal(0, 1, (100, n_features))
    bounds = [(0, 1) for _ in range(n_features)] if bounds else None
    estimator = BoundedKernelDensity(bounds=bounds)
    estimator.fit(x)
    scores = estimator.score_samples(x)
    assert scores.shape == (100,)

    if not bounds:
        np.testing.assert_allclose(scores, KernelDensity().fit(x).score_samples(x))
