from typing import Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .util import TensorLike

from .util import raise_for_missing_modules

with raise_for_missing_modules():
    import numpy as np
    from scipy import integrate, interpolate


def normalize_cdf(cdf: "TensorLike", tol: Optional[float] = None) -> np.ndarray:
    """
    Normalize a cumulative distribution function between 0 and 1 after validation.

    Args:
        cdf: Cumulative distribution function to normalize.
        tol: If given, the maximum acceptable difference for the first value of `cdf` to differ from
            0 and the last value to differ from 1. Discrepancies may arise, for example, due to
            numerical errors incurred integrating a probability distribution function to obtain
            `cdf`.

    Returns:
        Normalized cumulative distribution function.
    """
    if np.diff(cdf).min() < 0:
        raise ValueError("`cdf` must be non-decreasing.")

    if tol is not None:
        if abs(cdf[0]) > tol:
            raise ValueError(f"`cdf` must start with 0 (currently using tolerance {tol}); got "
                             f"{cdf[0]}.")
        if abs(cdf[-1] - 1) > tol:
            raise ValueError(f"`cdf` must end with 1 (currently using tolerance {tol}); got "
                             f"{cdf[-1]}.")

    cdfmin = cdf.min()
    return (cdf - cdfmin) / (cdf.max() - cdfmin)


def sample_empirical_cdf(x: "TensorLike", cdf: "TensorLike",
                         size: Optional[Union[int, Tuple[int]]] = None, kind: str = "linear",
                         random_state: Optional[np.random.RandomState] = None, tol: float = 1e-9) \
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
        tol: Tolerance for normalizing `cdf` (see :func:`.normalize_cdf` for details).

    Returns:
        Sample the desired size.
    """
    if np.diff(x).min() <= 0:
        raise ValueError("`x` must be strictly increasing.")

    random_state = random_state or np.random
    interpolated = interpolate.interp1d(normalize_cdf(cdf, tol), x, kind=kind, assume_sorted=True)
    return interpolated(random_state.uniform(size=size))


def sample_empirical_pdf(x: "TensorLike", pdf: "TensorLike",
                         size: Optional[Union[int, Tuple[int]]] = None, kind: str = "linear",
                         random_state: Optional[np.random.RandomState] = None, tol: float = 0) \
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
        tol: Maximum acceptable difference for the integrated cumulative distribution function to
            differ from 0 and the last value to differ from 1. Discrepancies may arise, for example,
            due to numerical errors incurred integrating `pdf`.

    Returns:
        Sample the desired size.
    """
    # Obtain the empirical CDF; we need to append a zero at the front because `cumulative_trapezoid`
    # returns `n - 1` elements for `n`-length input.
    cdf = np.zeros(x.size)
    cdf[1:] = integrate.cumulative_trapezoid(pdf, x)
    return sample_empirical_cdf(x, cdf, size, kind, random_state, tol)
