from __future__ import annotations
from itertools import product
from typing import Dict, Optional

from .util import raise_for_missing_modules


with raise_for_missing_modules():
    import numpy as np
    from scipy.special import logsumexp
    from sklearn.neighbors import KernelDensity
    from sklearn.utils import check_array


class BoundedKernelDensity(KernelDensity):
    """
    Kernel density estimator with reflection at the boundary.

    Args:
        bounds: Array of lower and upper bounds for each dimension (use :code:`nan` for unbounded or
            semi-bounded domains).
        **kwargs: Keyword arguments passed to :class:`~sklearn.neighbors.KernelDensity`.

    Example:

        .. plot::
            :include-source:

            from matplotlib import pyplot as plt
            import numpy as np
            from sklearn.neighbors import KernelDensity
            from snippets.stats import BoundedKernelDensity

            fig, axes = plt.subplots(2, 2, sharex="row", sharey="row")
            bandwidth = "scott"

            # One-dimensional bounds.
            x = np.random.uniform(0, 1, (1000, 1))
            kdes = [
                KernelDensity(bandwidth=bandwidth),
                BoundedKernelDensity(bandwidth=bandwidth, bounds=[0, 1]),
            ]
            lin = np.linspace(0, 1)
            for ax, kde in zip(axes[0], kdes):
                kde.fit(x)
                ax.plot(lin, np.exp(kde.score_samples(lin[:, None])))

            # Two-dimensional bounds.
            x = np.random.uniform(0, 1, (1000, 2))
            kdes = [
                KernelDensity(bandwidth=bandwidth),
                BoundedKernelDensity(bandwidth=bandwidth, bounds=[[0, 1], [0, 1]]),
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
    def __init__(self, *, bandwidth: float = 1.0, algorithm: str = "auto", kernel: str = "gaussian",
                 metric: str = "euclidean", atol: float = 0, rtol: float = 0,
                 breadth_first: bool = True, leaf_size: int = 40,
                 metric_params: Optional[Dict] = None, bounds: Optional[np.ndarray] = None) -> None:
        super().__init__(
            bandwidth=bandwidth, algorithm=algorithm, kernel=kernel, metric=metric, atol=atol,
            rtol=rtol, breadth_first=breadth_first, leaf_size=leaf_size, metric_params=metric_params
        )
        if bounds is None:
            self.bounds = None
        else:
            bounds = np.atleast_2d(bounds)
            assert bounds.ndim == 2 and bounds.shape[1] == 2, \
                f"Expected shape `(n_features, 2)` for `bounds`; got {bounds.shape}."
            self.bounds = bounds

    def fit(self, X: np.ndarray, y: None = None, sample_weight: Optional[np.ndarray] = None) \
            -> BoundedKernelDensity:
        X = check_array(X)
        assert self.bounds is None or X.shape[1] == self.bounds.shape[0], \
            "Bounds have {self.bounds.shape[0]} dimensions, but data has {X.shape[1]} dimensions."
        return super().fit(X, y, sample_weight)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return super().score_samples(X)

        # Obtain all reflections along the boundaries.
        X = check_array(X)
        scores = []
        reflections = [(lower, None, upper) for (lower, upper) in self.bounds]
        for reflection in product(*reflections):
            reflected = []
            for x, bound in zip(X.T, reflection):
                # Apply the reflection.
                if bound is not None and np.isfinite(bound):
                    x = 2 * bound - x
                reflected.append(x)

            scores.append(super().score_samples(np.transpose(reflected)))
        return logsumexp(scores, axis=0)
