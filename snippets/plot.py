import collections
import string
from typing import Iterable, List, Optional, Tuple, Union
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    from matplotlib.artist import Artist
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


Point = collections.namedtuple("Point", ["x", "y"])
Point.__doc__ = "Point in two dimensions."
Point.x.__doc__ = "Horizontal coordinate."
Point.y.__doc__ = "Vertical coordinate."


def get_anchor(artist: Union[Artist, Text], hour: float) -> Point:
    """
    Get an anchor on the boundary of an artist at the given "hour".

    Args:
        artist: Artist on whose boundary to get an anchor. If a :class:`~matplotlib.text.Text`
            instance and it has a bounding box patch, the bounding box patch is used.
        hour: Direction of the anchor as the hour on a 12-hour clock.

    Returns:
        Location of the anchor.

    .. note::

        :meth:`matplotlib.Figure.draw_without_rendering` may need to be called for extents of
        artists to be calculated correctly.

    Example:

        .. plot::
            :include-source:

            from matplotlib import pyplot as plt
            import numpy as np
            from snippets.plot import get_anchor

            # Add some text to the plot.
            fig, ax = plt.subplots()
            texts = [
                ax.text(0.1, 0.5, "hello", fontsize=40),
                ax.text(0.6, 0.5, "world", fontsize=40, bbox={"boxstyle": "round,pad=0.5"})
            ]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            # Draw without rendering ensures the extent of all artists is computed.
            fig.draw_without_rendering()

            # Find the anchors at different positions and plot them.
            hours = np.arange(12)
            for text in texts:
                anchors = [get_anchor(text, hour) for hour in hours]
                ax.scatter(*np.transpose(anchors), c=hours, zorder=9)
    """
    ax = artist.axes
    if isinstance(artist, Text):
        artist = artist.get_bbox_patch() or artist

    # Get the extent of the artist and find its center.
    xmin, ymin, width, height = artist.get_window_extent().bounds
    x = xmin + width / 2
    y = ymin + height / 2

    # Determine the displacement vector.
    angle = 2 * np.pi * hour / 12
    dx = np.sin(angle)
    dy = np.cos(angle)
    scale = 1 / (2 * max(abs(dy) / height, abs(dx) / width))
    point = x + scale * dx, y + scale * dy

    # Transform to the data coordinate system.
    point = ax.transData.inverted().transform(point)
    return Point(*point)


def arrow_path(path: Path, length: float, width: Optional[float] = None) -> Path:
    """
    Create an arrow at the end of a path.

    Args:
        path: Path to create an arrow for.
        length: Length of the arrow.
        width: Width of the arrow (defaults to an equilateral triangle).

    Returns:
        Path representing an arrow.

    Example:

        .. plot::
            :include-source:

            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            from matplotlib import pyplot as plt
            from snippets.plot import arrow_path

            fig, ax = plt.subplots()
            path = Path([(0.2, 0.4), (0.9, 0.7)], [Path.MOVETO, Path.LINETO])
            arrow = arrow_path(path, 0.1)
            ax.add_patch(PathPatch(path, fc="none"))
            ax.add_patch(PathPatch(arrow))
            ax.set_aspect("equal")
    """
    width = 2 * length / np.sqrt(3) if width is None else width
    *_, a, b = path.vertices
    delta = b - a
    delta /= np.linalg.norm(delta)
    orth = delta[::-1] * [-1, 1]
    vertices = [b, b - length * delta + width / 2 * orth, b - length * delta - width / 2 * orth, b]
    return Path(vertices, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY], closed=True)
