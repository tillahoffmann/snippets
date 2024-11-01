import collections
import string
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
from .util import raise_for_missing_modules


with raise_for_missing_modules():
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection, PolyCollection
    from matplotlib.lines import Line2D
    from matplotlib.image import AxesImage
    from matplotlib.path import Path
    from matplotlib import pyplot as plt
    from matplotlib.text import Text
    import numpy as np


def plot_band(
    x: np.ndarray,
    ys: np.ndarray,
    *,
    ax: Optional[Axes] = None,
    ralpha: float = 0.2,
    lower: float = 0.05,
    center: float = 0.5,
    upper: float = 0.95,
    **kwargs,
) -> Tuple[Line2D, PolyCollection]:
    """
    Plot a central line and shaded band for samples.

    Args:
        x: :math:`n`-vector of horizontal coordinates.
        ys: :math:`m \\times n`-matrix comprising :math:`m` samples each at :math:`n`
            horizontal coordinates.
        ralpha: Alpha for the shaded band relative to the central line.
        lower: Quantile of the lower edge of the shaded band.
        center: Quantile of the central line.
        upper: Quantile of the upper edge of the shaded band.
        ax: Axes to use for plotting (defaults to :func:`matplotlib.pyplot.gca`).
        **kwargs: Keyword arguments passed to :func:`matplotlib.axes.Axes.plot` for
            plotting the central line.

    Example:

        .. plot::

            import numpy as np
            from snippets.plot import plot_band

            x = np.linspace(0, 2 * np.pi, 20)
            ys = np.sin(x) + np.random.normal(0, .25, (100, x.size))
            plot_band(x, ys)
    """
    ax = ax or plt.gca()
    lower_, center_, upper_ = np.quantile(ys, [lower, center, upper], axis=0)
    (line,) = ax.plot(x, center_, **kwargs)
    fill = ax.fill_between(x, lower_, upper_, alpha=(line.get_alpha() or 1.0) * ralpha)
    return line, fill


def rounded_path(
    vertices: np.ndarray,
    radius: float,
    shrink: float = 0,
    closed: bool = False,
    readonly: bool = False,
) -> Path:
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


def label_axes(
    axes: Union[Iterable[Axes], Axes],
    labels: Optional[Union[Iterable[str], str]] = None,
    loc: str = "top left",
    offset: float = 0.05,
    label_offset: int = 0,
    **kwargs,
) -> List[Text]:
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

            from matplotlib import pyplot as plt
            from snippets.plot import label_axes

            fig, axes = plt.subplots(2, 2)
            label_axes(axes[0])
            label_axes(axes[1], label_offset=2)
    """
    if isinstance(axes, Axes):
        axes = [axes]
    if labels is None:
        labels = [f"({x})" for x in string.ascii_lowercase]
    elif isinstance(labels, str):
        labels = [labels]
    if label_offset is not None:
        labels = labels[label_offset:]
    if isinstance(offset, float):
        xfactor = yfactor = offset
    else:
        xfactor, yfactor = offset
    y, x = loc.split()
    kwargs = {"ha": x, "va": y, **kwargs}
    xloc = xfactor if x == "left" else (1 - xfactor)
    yloc = yfactor if y == "bottom" else (1 - yfactor)
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
        artist: Artist on whose boundary to get an anchor. If a
            :class:`~matplotlib.text.Text` instance and it has a bounding box patch, the
            bounding box patch is used.
        hour: Direction of the anchor as the hour on a 12-hour clock.

    Returns:
        Location of the anchor.

    .. note::

        :meth:`matplotlib.Figure.draw_without_rendering` may need to be called for
        extents of artists to be calculated correctly.

    Example:

        .. plot::

            from matplotlib import pyplot as plt
            import numpy as np
            from snippets.plot import get_anchor

            # Add some text to the plot.
            fig, ax = plt.subplots()
            texts = [
                ax.text(0.1, 0.5, "hello", fontsize=40),
                ax.text(
                    0.6, 0.5, "world", fontsize=40, bbox={"boxstyle": "round,pad=0.5"}
                ),
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


def arrow_path(
    path: Path, length: float, width: Optional[float] = None, backward: bool = False
) -> Path:
    """
    Create an arrow at the end of a path.

    Args:
        path: Path to create an arrow for.
        length: Length of the arrow.
        width: Width of the arrow (defaults to an equilateral triangle).
        backward: Create an arrow at the start of the path.

    Returns:
        Path representing an arrow.

    Example:

        .. plot::

            from matplotlib.patches import PathPatch
            from matplotlib.path import Path
            from matplotlib import pyplot as plt
            from snippets.plot import arrow_path

            fig, ax = plt.subplots()
            path = Path([(0.2, 0.4), (0.9, 0.7)], [Path.MOVETO, Path.LINETO])
            ax.add_patch(PathPatch(path, fc="none"))

            arrow = arrow_path(path, 0.1)
            ax.add_patch(PathPatch(arrow))
            arrow = arrow_path(path, 0.1, backward=True)
            ax.add_patch(PathPatch(arrow, fc="C1"))
            ax.set_aspect("equal")
    """
    width = 2 * length / np.sqrt(3) if width is None else width
    *_, a, b = reversed(path.vertices) if backward else path.vertices
    delta = b - a
    delta /= np.linalg.norm(delta)
    orth = delta[::-1] * [-1, 1]
    vertices = [
        b,
        b - length * delta + width / 2 * orth,
        b - length * delta - width / 2 * orth,
        b,
    ]
    return Path(
        vertices, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY], closed=True
    )


def dependence_heatmap(
    samples: Dict[str, np.ndarray],
    method: Literal["corrcoef", "nmi"] = "corrcoef",
    ax: Optional[Axes] = None,
    labels: bool = True,
    lines: bool = True,
    xlabel_rotation: float = -90,
    ylabel_rotation: float = 0,
    **kwargs,
) -> AxesImage:
    """
    Show the dependence between parameters as a heatmap.

    Args:
        fit: Named parameter samples.
        method: Method to estimate dependence between variables.
        ax: Axes to use for plotting.
        labels: Show parameter labels.
        lines: Show lines between blocks of parameters.
        xlabel_rotation: Rotation of x-axis labels for parameter names.
        ylabel_rotation: Rotation of y-axis labels for parameter names.
        **kwargs: Keyword arguments passed to :class:`.Axes.imshow`.

    Example:

        .. plot::

            import numpy as np
            from snippets.plot import dependence_heatmap

            # Draw some correlated samples.
            n = 25
            cov = np.cov(np.random.normal(0, 1, (n, 100)))
            samples = np.random.multivariate_normal(np.zeros(n), cov, 100)
            samples = {
                "a": samples[:, :10],
                "b": samples[:, 10:19],
                "c": samples[:, 19:],
            }

            # Show the dependence.
            fig, ax = plt.subplots()
            dependence_heatmap(samples)
    """
    ax = ax or plt.gca()

    # Compute the correlation coefficient and mask the diagonal.
    stacked = np.hstack([x.reshape((x.shape[0], -1)) for x in samples.values()])

    # Estimate dependence and limits for the colormap.
    if method == "corrcoef":
        dependence = np.corrcoef(stacked.T)
        np.fill_diagonal(dependence, np.nan)

        vmax = kwargs.setdefault("vmax", np.nanmax(np.abs(dependence)))
        kwargs.setdefault("vmin", -vmax)
        kwargs.setdefault("cmap", "coolwarm")
    elif method == "nmi":
        with raise_for_missing_modules():
            from sklearn.feature_selection import mutual_info_regression
        dependence = np.asarray([mutual_info_regression(stacked, x) for x in stacked.T])
        diag = np.diag(dependence)
        dependence /= (diag[:, None] + diag) / 2
        np.fill_diagonal(dependence, np.nan)
    else:  # pragma: no cover
        raise ValueError(method)

    # Plot the heatmap.
    im = ax.imshow(dependence, **kwargs)

    # Add lines and labels.
    sizes = np.asarray([value[0].size for value in samples.values()])
    locs = np.cumsum(sizes) - (sizes + 1) / 2

    if lines:
        for axxline in [ax.axhline, ax.axvline]:
            for loc in np.cumsum(sizes):
                axxline(loc - 1 / 2, color="gray", ls=":")

    if labels:
        for axis, rotation in [
            (ax.xaxis, xlabel_rotation),
            (ax.yaxis, ylabel_rotation),
        ]:
            axis.set_ticks(locs)
            axis.set_ticklabels(samples, rotation=rotation)
    return im


def parameterization_mutual_info(
    x: np.ndarray,
    scale: np.ndarray,
    ax: Optional[Axes] = None,
    labels: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, PathCollection]:
    """
    Scatter plot of the mutual information between scale and location parameters for
    centered and non-centered parameterizations.

    Args:
        x: Centered parameter of interest.
        scale: Scale parameter of the prior on :code:`x`.
        ax: Axes to use for plotting.
        labels: Add axis labels.
        **kwargs: Keyword arguments passed to :class:`.Axes.scatter`.

    Returns:
        Tuple of mutual information between scale and centered parameter, mutual
        information between scale and non-centered parameter, and points.

    Notes:

        Standard *centered* parameterizations in hierarchical models often exhibit
        "funnels" if the data do not strongly inform each parameter individually (see
        `here <https://mc-stan.org/docs/stan-users-guide/reparameterization.html>`__ for
        details). Choosing between the *centered* and *non-centered* parameterizations
        is often challenging without inspecting scatter plots between parameters and the
        scale parameter of the corresponding prior. This function estimates the mutual
        information between the scale :math:`\\sigma` of the prior and both the
        *centered* parameterization
        :math:`x \\sim \\mathsf{Normal}\\left(0, \\sigma^2\\right)` and the
        *non-centered* parameterization :math:`x = \\sigma z` with
        :math:`z \\sim \\mathsf{Normal}\\left(0, 1\\right)`. The parameterization with
        the lower mutual information is generally preferable because it decouples the
        parameters under the posterior.

    Example:

        .. plot::

            from matplotlib import pyplot as plt
            import numpy as np
            from snippets.plot import parameterization_mutual_info


            fig, axes = plt.subplots(2, 2)

            n = 1000
            p = 10
            scale = np.exp(np.random.normal(0, 1, n))

            # Example parameters dominated by the data, i.e., `x` is independent of
            # `scale`.
            x1 = np.random.normal(0, 1, (n, p))
            # Example parameters dominated by the prior, i.e., `x` is strongly informed
            # by `scale`.
            x2 = x1 * scale[:, None]

            # Scatter the mutual information, lower is better.
            for (ax1, ax2), x in zip(axes.T, [x1, x2]):
                assert x.shape == (n, p)
                assert scale.shape == (n,)
                ax1.scatter(x[:, 0], scale)
                ax1.set_yscale("log")
                parameterization_mutual_info(x, scale, ax=ax2)
            fig.tight_layout()
    """
    with raise_for_missing_modules():
        from sklearn.feature_selection import mutual_info_regression

    ax = ax or plt.gca()

    x = np.asarray(x).reshape((len(x), -1))
    # Evaluate the non-centered parameter.
    z = x / np.asarray(scale)[:, None]

    mix = mutual_info_regression(x, scale)
    miz = mutual_info_regression(z, scale)
    pts = ax.scatter(mix, miz, **kwargs)
    mm = mix.min(), mix.max()
    ax.plot(mm, mm, color="k", ls=":")

    if labels:
        ax.set_xlabel("$MI(x, \\sigma)$")
        ax.set_ylabel("$MI(z = x / \\sigma, \\sigma)$")

    return mix, miz, pts
