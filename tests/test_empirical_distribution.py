import numpy as np
import pytest
from scipy import stats
from snippets.empirical_distribution import normalize_cdf, sample_empirical_pdf
from typing import Optional, Tuple, Union


@pytest.fixture
def x():
    return np.linspace(-5, 5, 1001)


@pytest.fixture
def pdf(x: np.ndarray):
    return stats.norm.pdf(x)


@pytest.mark.parametrize("kind", ["linear", "quadratic", "cubic"])
def test_sample_empirical_pdf(x: np.ndarray, pdf: np.ndarray, kind: str) -> None:
    y = sample_empirical_pdf(x, pdf, (100, 17), kind, tol=1e-5)
    assert y.shape == (100, 17)
    assert stats.normaltest(y.ravel()).pvalue > 0.01
    assert np.unique(y).size == 1_700


def test_sample_empirical_pdf_without_shape(x: np.ndarray, pdf: np.ndarray) -> None:
    assert sample_empirical_pdf(x, pdf, tol=1e-5).shape == ()


@pytest.mark.parametrize("shape", [
    None,
    17,
    (4, 5),
    (3, 4, 5),
])
def test_sample_empirical_pdf_shapes(x: np.ndarray, pdf: np.ndarray,
                                     shape: Optional[Union[Tuple[int], int]]) -> None:
    y = sample_empirical_pdf(x, pdf, shape, tol=1e-5)
    expected_shape = () if shape is None else ((shape,) if isinstance(shape, int) else shape)
    assert y.shape == expected_shape


def test_rv_non_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        sample_empirical_pdf(np.zeros(3), np.zeros(3) * np.nan)


def test_normalize_cdf() -> None:
    cdf = np.linspace(0, 1)
    np.testing.assert_allclose(normalize_cdf(cdf), cdf)
    # Normalize without validation.
    np.testing.assert_allclose(normalize_cdf(1 + cdf), cdf)

    # Complain about non-monotone cdf.
    with pytest.raises(ValueError, match="must be non-decreasing"):
        normalize_cdf(cdf[::-1], tol=0)

    # Complain about the initial value.
    with pytest.raises(ValueError, match="must start with 0"):
        normalize_cdf(1 + cdf, tol=0)
    with pytest.raises(ValueError, match="must start with 0"):
        normalize_cdf(cdf - 1, tol=0)

    # Complain about the final value.
    with pytest.raises(ValueError, match="must end with 1"):
        normalize_cdf(cdf / 2, tol=0)
    with pytest.raises(ValueError, match="must end with 1"):
        normalize_cdf(2 * cdf, tol=0)
