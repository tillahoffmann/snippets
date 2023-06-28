import numpy as np
from scipy import stats
from snippets.empirical_distribution import sample_empirical_pdf


def test_sample_empirical_cdf() -> None:
    x = np.linspace(-5, 5, 1001)
    pdf = stats.norm.pdf(x)
    y = sample_empirical_pdf(x, pdf, (100, 17))
    assert y.shape == (100, 17)
    assert stats.normaltest(y.ravel()).pvalue > 0.01
