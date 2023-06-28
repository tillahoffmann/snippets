from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .util import TensorLike

from .util import raise_for_missing_modules

with raise_for_missing_modules():
    import numpy as np
    from scipy import integrate, interpolate


def sample_empirical_cdf(x: "TensorLike", cdf: "TensorLike", size: int | Tuple[int] | None,
                         kind: str = "linear", random_state: np.random.RandomState | None = None) \
        -> np.ndarray:
    """
    Sample from a univariate empirical cumulative distribution function using interpolation of the
    inverse cumulative distribution function.

    Args:
        x: Ordered vector of random variable values corresponding to `cdf` values.
        cdf: Cumulative distribution function values corresponding to `x` values.
        size: Sample size to draw.
        kind: Interpolation method to use (see :class:`scipy.interpolate.inter1pd` for details).
        random_state: Random number generator state.

    Returns:
        Sample the desired size.
    """
    random_state = random_state or np.random
    interpolated = interpolate.interp1d(cdf, x, kind=kind, assume_sorted=True)
    return interpolated(random_state.uniform(size=size))


def sample_empirical_pdf(x: "TensorLike", pdf: "TensorLike", size: int | Tuple[int] | None,
                         kind: str = "linear", random_state: np.random.RandomState | None = None) \
        -> np.ndarray:
    """
    Sample from a univariate empirical probability distribution function using interpolation of the
    inverse cumulative distribution function.

    Args:
        x: Ordered vector of random variable values corresponding to `pdf` values.
        pdf: Cumulative distribution function values corresponding to `x` values.
        size: Sample size to draw.
        kind: Interpolation method to use (see :class:`scipy.interpolate.inter1pd` for details).
        random_state: Random number generator state.

    Returns:
        Sample the desired size.
    """
    # Obtain the empirical CDF; we need to append a zero at the front because `cumulative_trapezoid`
    # returns `n - 1` elements for `n`-length input.
    cdf = np.zeros(x.size)
    cdf[1:] = integrate.cumulative_trapezoid(pdf, x)
    return sample_empirical_cdf(x, cdf, size, kind, random_state)
