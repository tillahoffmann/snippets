import numpy as np
import pytest
from scipy import stats
from snippets.empirical_distribution import sample_empirical_pdf
from typing import Optional, Tuple, Union


@pytest.fixture
def x():
    return np.linspace(-5, 5, 1001)


@pytest.fixture
def pdf(x: np.ndarray):
    return stats.norm.pdf(x)


def test_sample_empirical_pdf(x: np.ndarray, pdf: np.ndarray) -> None:
    y = sample_empirical_pdf(x, pdf, (100, 17))
    assert y.shape == (100, 17)
    assert stats.normaltest(y.ravel()).pvalue > 0.01


def test_sample_empirical_pdf_without_shape(x: np.ndarray, pdf: np.ndarray) -> None:
    assert sample_empirical_pdf(x, pdf).shape == ()


@pytest.mark.parametrize("shape", [
    None,
    17,
    (4, 5),
    (3, 4, 5),
])
def test_sample_empirical_pdf_shapes(x: np.ndarray, pdf: np.ndarray,
                                     shape: Optional[Union[Tuple[int], int]]) -> None:
    y = sample_empirical_pdf(x, pdf, shape)
    expected_shape = () if shape is None else ((shape,) if isinstance(shape, int) else shape)
    assert y.shape == expected_shape
