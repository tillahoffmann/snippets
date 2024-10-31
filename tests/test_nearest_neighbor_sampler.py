import numpy as np
import pytest
from scipy import stats
from sklearn.exceptions import NotFittedError
from snippets.nearest_neighbor_sampler import NearestNeighborSampler


def sample_params(n: int, p: int) -> np.ndarray:
    return np.random.normal(0, 1, (n, p))


def sample_data(params: np.ndarray) -> np.ndarray:
    n = params.shape[0]
    x = params + np.random.normal(0, 0.1, (n, 2))
    return np.hstack([x, np.random.normal(0, 10, (n, 1))])


@pytest.fixture(params=[1, 2])
def n_params(request: pytest.FixtureRequest) -> int:
    return request.param


@pytest.fixture
def simulated_params(n_params: int) -> np.ndarray:
    return sample_params(100_000, n_params)


@pytest.fixture
def simulated_data(simulated_params: np.ndarray) -> np.ndarray:
    return sample_data(simulated_params)


@pytest.fixture
def latent_params(n_params: int) -> np.ndarray:
    return sample_params(100, n_params)


@pytest.fixture
def observed_data(latent_params: np.ndarray) -> np.ndarray:
    return sample_data(latent_params)


@pytest.mark.parametrize("multi_output", [False, True])
def test_posterior_mean_correlation(
    simulated_data: np.ndarray,
    simulated_params: np.ndarray,
    observed_data: np.ndarray,
    latent_params: np.ndarray,
    multi_output: bool,
) -> None:
    if not multi_output:
        simulated_params = np.squeeze(simulated_params)

    sampler = NearestNeighborSampler(frac=0.0017).fit(simulated_data, simulated_params)
    samples = sampler.predict(observed_data)
    posterior_mean = samples.mean(axis=1)
    pearsonr = stats.pearsonr(posterior_mean.ravel(), latent_params.ravel())
    assert pearsonr.statistic > 0.8 and pearsonr.pvalue < 0.01

    if multi_output or simulated_params.ndim > 1:
        expected_shape = (
            observed_data.shape[0],
            sampler.n_samples,
            simulated_params.shape[1],
        )
    else:
        expected_shape = (observed_data.shape[0], sampler.n_samples)
    assert samples.shape == expected_shape


def test_nearest_neighbor_not_fitted() -> None:
    with pytest.raises(NotFittedError):
        NearestNeighborSampler(frac=0.01).predict(None)


@pytest.mark.parametrize("n_samples", [1, 2])
@pytest.mark.parametrize("batch_size", [1, 7])
def test_nearest_neighbor_single_sample(n_samples: int, batch_size: int) -> None:
    sampler = NearestNeighborSampler(n_samples=n_samples)
    sampler.fit(np.random.normal(0, 1, (100, 3)), np.random.normal(0, 1, (100, 2)))
    samples = sampler.predict(np.random.normal(0, 1, (batch_size, 3)))
    assert samples.shape == (batch_size, n_samples, 2)


def test_mutually_exclusive_kwargs() -> None:
    with pytest.raises(ValueError, match="Exactly one of"):
        NearestNeighborSampler()
    with pytest.raises(ValueError, match="Exactly one of"):
        NearestNeighborSampler(frac=0.3, n_samples=3)
