from typing import Optional, Tuple
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.path import Path
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

        .. doctest::

            >>> import numpy as np
            >>> from snippets.plot import plot_band

            >>> x = np.linspace(0, 2 * np.pi, 20)
            >>> ys = np.sin(x) + np.random.normal(0, .25, (100, x.size))
            >>> plot_band(x, ys)
            (<matplotlib.lines.Line2D ...>, <matplotlib.collections.PolyCollection ...>)
    """
    ax = ax or plt.gca()
    lower_, center_, upper_ = np.quantile(ys, [lower, center, upper], axis=0)
    line, = ax.plot(x, center_, **kwargs)
    fill = ax.fill_between(x, lower_, upper_, alpha=(line.get_alpha() or 1.0) * ralpha)
    return line, fill


def rounded_path(vertices: np.ndarray, radius: float, shrink: float = 0, closed: bool = False,
                 readonly: bool = False) -> Path:
    """
    Create a path with rounded corners.

    Args:
        vertices: Vertices comprising the path.
        radius: Radius of rounded corners.
        shrink: Amount to shrink the beginning and end of the path by.
        closed: Close the path.
        readonly: Make the path readonly.

    Returns:
        Path with rounded corners.

    Example:

        .. plot::
            :include-source:

            >>> import matplotlib as mpl
            >>> from matplotlib import pyplot as plt
            >>> import numpy as np
            >>> from snippets.plot import rounded_path
            >>>
            >>> fig, ax = plt.subplots()
            >>> vertices = np.asarray([(0, 0), (0, 1), (1, 1), (2, 0), (1.9, 0), (1, 0.5)])
            >>> ax.plot(*vertices.T, marker=".", ls=":")
            [<matplotlib.lines.Line2D object at 0x...>]
            >>> path = rounded_path(vertices, 0.2, 0.1)
            >>> ax.add_patch(mpl.patches.PathPatch(path, lw=3, fc="none", ec="C1", alpha=0.8))
            <matplotlib.patches.PathPatch object at 0x...>
            >>> ax.set_aspect("equal")
            >>> fig.tight_layout()
    """
    vertices = np.asarray(vertices, dtype=float)
    parts = []
    for i, vertex in enumerate(vertices):
        # Calculate the unit vector from the current to the next point.
        next = vertices[i + 1]
        forward = next - vertex
        distance = np.linalg.norm(forward)
        forward /= distance
        # Move to starting point.
        if i == 0:
            parts.append((vertex + shrink * forward, Path.MOVETO))
        # Add the control point and beginning of the next straight segment.
        if i > 0:
            parts.append((vertex, Path.CURVE3))
            parts.append((vertex + min(radius, distance / 2) * forward, Path.CURVE3))
        # Until the last point, add the start of the next curve.
        if i < len(vertices) - 2:
            parts.append((next - min(radius, distance / 2) * forward, Path.LINETO))
        # Line to the end point.
        if i == len(vertices) - 2:
            parts.append((next - shrink * forward, Path.LINETO))
            break

    return Path(*zip(*parts), closed=closed, readonly=readonly)
