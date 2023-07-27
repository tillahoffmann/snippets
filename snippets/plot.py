import string
from typing import Iterable, List, Optional, Tuple, Union
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    from matplotlib.axes import Axes
    from matplotlib.collections import PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.path import Path
    from matplotlib import pyplot as plt
    from matplotlib.text import Text
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

        .. plot::
            :include-source:

            import numpy as np
            from snippets.plot import plot_band

            x = np.linspace(0, 2 * np.pi, 20)
            ys = np.sin(x) + np.random.normal(0, .25, (100, x.size))
            plot_band(x, ys)
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

            import matplotlib as mpl
            from matplotlib.patches import PathPatch
            from matplotlib import pyplot as plt
            import numpy as np
            from snippets.plot import rounded_path

            fig, ax = plt.subplots()
            lines = [
                [(0, 0), (0, 1), (1, 1), (2, 0), (1.9, 0), (1, 0.5)],
                [(1.5, 1), (2, 1)],
                [(0, 1.5), (1, 2), (2, 1.5)],
            ]
            for i, line in enumerate(lines):
                color = f"C{i}"
                ax.plot(*np.transpose(line), marker="o", ls="--", color=color)
                path = rounded_path(line, 0.2, 0.1)
                patch = PathPatch(path, lw=5, fc="none", ec=color, alpha=0.5)
                ax.add_patch(patch)
            ax.set_aspect("equal")
            fig.tight_layout()
    """
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 2:
        raise ValueError(f"at least two vertices are required; got {len(vertices)}")
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


def label_axes(axes: Union[Iterable[Axes], Axes],
               labels: Optional[Union[Iterable[str], str]] = None, loc: str = 'top left',
               offset: float = 0.05, label_offset: int = 0, **kwargs) -> List[Text]:
    """
    Add labels to axes.

    Args:
        axes: Iterable of matplotlib axes.
        labels: Iterable of labels (defaults to lowercase letters in parentheses).
        loc: Location of the label as a string (defaults to top left).
        offset: Offset for positioning labels in axes coordinates.
        label_offset: Index by which to offset labels.

    Returns:
        List of text labels.

    Example:

        .. plot::
            :include-source:

            from matplotlib import pyplot as plt
            from snippets.plot import label_axes

            fig, axes = plt.subplots(2, 2)
            label_axes(axes[0])
            label_axes(axes[1], label_offset=2)
    """
    if isinstance(axes, Axes):
        axes = [axes]
    if labels is None:
        labels = [f'({x})' for x in string.ascii_lowercase]
    elif isinstance(labels, str):
        labels = [labels]
    if label_offset is not None:
        labels = labels[label_offset:]
    if isinstance(offset, float):
        xfactor = yfactor = offset
    else:
        xfactor, yfactor = offset
    y, x = loc.split()
    kwargs = {'ha': x, 'va': y, **kwargs}
    xloc = xfactor if x == 'left' else (1 - xfactor)
    yloc = yfactor if y == 'bottom' else (1 - yfactor)
    elements = []
    for ax, label in zip(axes, labels):
        elements.append(ax.text(xloc, yloc, label, transform=ax.transAxes, **kwargs))
    return elements
