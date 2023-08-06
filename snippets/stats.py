from __future__ import annotations
from itertools import product
from typing import Optional, Union

from .util import raise_for_missing_modules


with raise_for_missing_modules():
    import numpy as np
    from scipy.special import logsumexp
    from scipy.stats import gaussian_kde
    from sklearn.base import BaseEstimator
    from sklearn.exceptions import NotFittedError
    from sklearn.utils import check_array


def evaluate_bounded_kde_logpdf(kde: gaussian_kde, X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Evaluated the log probability of a kernel density estimate with bounded support.

    Args:
        kde: Kernel density estimate to evaluate.
        X: Points at which to evaluate the kernel density estimate with shape
            :code:`(n_features, n_samples)`.
        bounds: Sequence of :code:`(lower, upper)` bounds, i.e., a matrix with shape
            :code:`(n_features, 2)`.

    Returns:
        Log probability evaluated at :code:`X`.

    Example:

        .. plot::
            :include-source:

            from matplotlib import pyplot as plt
            import numpy as np
            from scipy.stats import gaussian_kde
            from snippets.stats import evaluate_bounded_kde_logpdf

            fig, ax = plt.subplots()
            x = np.random.uniform(0, 1, 1000)
            kde = gaussian_kde(x)
            lin = np.linspace(0, 1)
            ax.plot(lin, kde.evaluate(lin))
            ax.plot(lin, np.exp(evaluate_bounded_kde_logpdf(kde, lin, (0, 1))))
    """
    bounds = np.atleast_2d(bounds)
    assert bounds.shape == (kde.d, 2), \
        f"Expected shape (n_features, 2) for `bounds` but got {bounds.shape}."

    # Handle sample shapes just like scipy.stats.gaussian_kde does (see source of `logpdf`).
    X = np.atleast_2d(X)
    n_features, n_samples = X.shape
    if n_features != kde.d and n_features == 1 and n_samples == kde.d:
        X = X.T
    assert X.shape[0] == (kde.d), \
        f"Expected shape (n_features={kde.d}, n_samples=...) for `X` but got {X.shape}."

    reflections = [(lower, None, upper) for (lower, upper) in bounds]
    scores = []
    for reflection in product(*reflections):
        reflected = []
        for x, bound in zip(X, reflection):
            # Apply the reflection.
            if bound is not None and np.isfinite(bound):
                x = 2 * bound - x
            reflected.append(x)

        scores.append(kde.logpdf(reflected))
    return logsumexp(scores, axis=0)


class GaussianKernelDensity(BaseEstimator):
    """
    Gaussian kernel density estimator using :class:`~scipy.stats.gaussian_kde` as the base
    implementation. This facilitates more sophisticated kernel covariances because
    :class:`~sklearn.neighbors.KernelDensity` only allows isotropic kernel bandwidths.

    Args:
        bandwidth: Bandwidth selection method or scalar factor (see
            :class:`~scipy.stats.gaussian_kde` for details).
        bounds: Array of lower and upper bounds for each dimension (use :code:`nan` for unbounded or
            semi-bounded domains).

    Example:

        .. plot::
            :include-source:

            from matplotlib import pyplot as plt
            import numpy as np
            from sklearn.neighbors import KernelDensity
            from snippets.stats import GaussianKernelDensity

            fig, axes = plt.subplots(2, 2, sharex="row", sharey="row")

            # One-dimensional bounds.
            x = np.random.uniform(0, 1, (10_000, 1))
            kdes = [
                GaussianKernelDensity(),
                GaussianKernelDensity(bounds=[0, 1]),
            ]
            lin = np.linspace(0, 1)
            for ax, kde in zip(axes[0], kdes):
                kde.fit(x)
                ax.plot(lin, np.exp(kde.score_samples(lin[:, None])))

            # Two-dimensional bounds.
            x = np.random.uniform(0, 1, (10_000, 2))
            kdes = [
                GaussianKernelDensity(),
                GaussianKernelDensity(bounds=[[0, 1], [0, 1]]),
            ]
            lin = np.linspace(0, 1)
            xx, yy = np.meshgrid(lin, lin)
            xy = np.stack([xx, yy], axis=-1).reshape((-1, 2))
            for ax, kde in zip(axes[1], kdes):
                kde.fit(x)
                score = kde.score_samples(xy).reshape(xx.shape)
                im = ax.imshow(np.exp(score), extent=(0, 1, 0, 1))
                fig.colorbar(im, ax=ax)
    """
    def __init__(self, bandwidth: Optional[Union[str, float]] = None,
                 bounds: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.kde_: Optional[gaussian_kde] = None
        self.bandwidth = bandwidth
        self.bounds = bounds
        if self.bounds is not None:
            self.bounds = np.atleast_2d(bounds)
            assert self.bounds.ndim == 2 and self.bounds.shape[1] == 2, \
                f"Expected shape `(n_features, 2)` for `bounds`; got {self.bounds.shape}."

    def fit(self, X: np.ndarray, y: None = None, sample_weight: Optional[np.ndarray] = None) \
            -> GaussianKernelDensity:
        X = check_array(X)
        assert self.bounds is None or X.shape[1] == self.bounds.shape[0], \
            f"Bounds have {self.bounds.shape[0]} dimensions, but data has {X.shape[1]} dimensions."
        self.kde_ = gaussian_kde(X.T, self.bandwidth, sample_weight)
        return self

    def _ensure_fitted(self) -> None:  # pragma: no cover
        if self.kde_ is None:
            raise NotFittedError

    @property
    def n_features_in_(self) -> int:
        """
        Number of features the density estimator was fit to.
        """
        self._ensure_fitted()
        return self.kde_.d

    @property
    def bandwidth_factor_(self) -> float:
        """
        Multiplicative bandwidth factor for the density estimator.
        """
        self._ensure_fitted()
        return self.kde_.factor

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the log-likelihood of each sample.

        Args:
            X : Samples whose log-likelihood to evaluate.

        Returns:
            Log-likelihood of each sample.
        """
        X = check_array(X)
        assert X.shape[1] == self.n_features_in_, \
            f"Estimator was fit to {self.n_features_in_} features but samples have {X.shape[1]}."

        # Return the estimate as-is if no bounds are specified.
        if self.bounds is None:
            return self.kde_.logpdf(X.T)
        return evaluate_bounded_kde_logpdf(self.kde_, X.T, self.bounds)

    def score(self, X: np.ndarray, y: None = None) -> float:
        """
        Compute the total log-likelihood.

        Args:
            X : Samples whose log-likelihood to evaluate.
            y: Ignored.

        Returns:
            Total log-likelihood of the data.
        """
        return self.score_samples(X).sum()
