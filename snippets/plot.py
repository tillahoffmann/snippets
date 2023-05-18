from typing import Optional, Tuple
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib import pyplot as plt
    import numpy as np


def plot_band(x: np.ndarray, ys: np.ndarray, *, ax: Optional[Axes] = None, ralpha: float = 0.2,
              lower: float = 0.05, center: float = 0.5, upper: float = 0.95, **kwargs) \
        -> Tuple[Line2D, PolyCollection]:
    """
    Plot a central line and shaded band for samples.

    Args:
        x: :math:`n`-vector of horizontal coordinates.
        ys: :math:`m \\times n`-matrix comprising :math:`m` samples each at :math:`n` horizontal
            coordinates.
        ralpha: Alpha for the shaded band relative to the central line.
        lower: Quantile of the lower edge of the shaded band.
        center: Quantile of the central line.
        upper: Quantile of the upper edge of the shaded band.
        ax: Axes to use for plotting (defaults to :func:`matplotlib.pyplot.gca`).
        **kwargs: Keyword arguments passed to :func:`matplotlib.axes.Axes.plot` for plotting the
            central line.

    Example:

        >>> import numpy as np
        >>> from snippets.plot import plot_band
        >>> x = np.linspace(0, 2 * np.pi, 20)
        >>> ys = np.sin(x) + np.random.normal(0, .25, (100, x.size))
        >>> plot_band(x, ys)
        (<matplotlib.lines.Line2D object ...>, <matplotlib.collections.PolyCollection object ...>)
    """
    ax = ax or plt.gca()
    lower_, center_, upper_ = np.quantile(ys, [lower, center, upper], axis=0)
    line, = ax.plot(x, center_, **kwargs)
    fill = ax.fill_between(x, lower_, upper_, alpha=(line.get_alpha() or 1.0) * ralpha)
    return line, fill
