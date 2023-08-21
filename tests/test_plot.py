from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib import pyplot as plt
import numpy as np
import pytest
from snippets.plot import arrow_path, dependence_heatmap, get_anchor, label_axes, plot_band, \
    rounded_path
from typing import Iterable, Optional, Union


def test_plot_band() -> None:
    x = np.linspace(0, 2 * np.pi, 20)
    ys = np.sin(x) + np.random.normal(0, .25, (100, x.size))
    line, band = plot_band(x, ys)
    assert isinstance(line, Line2D)
    assert isinstance(band, PolyCollection)


def test_rounded_path() -> None:
    vertices = [(0, 0), (1, 0), (1, 1)]
    assert isinstance(rounded_path(vertices, 0.2, 0.1), Path)

    with pytest.raises(ValueError, match="at least two"):
        rounded_path([], 0, 0)


@pytest.mark.parametrize("labels", ["a", ("a", "b"), None])
def test_label_axes(labels: Optional[Union[str, Iterable[str]]]) -> None:
    # Try all sorts of different options to ensure full coverage.
    _, axes = plt.subplots(2, 2)
    labels = label_axes(axes.ravel(), labels, label_offset=2 if labels is None else None,
                        offset=(0.05, 0.06))
    assert len(labels) == (4 if labels is None else len(labels))

    # Add a label to a single axes.
    label_axes(axes[0, 0], "foo", loc="bottom right")


def test_get_anchor() -> None:
    fig, ax = plt.subplots()
    text = ax.text(0.5, 0.5, "hello")
    padded = ax.text(0.5, 0.5, "hello", bbox={"boxstyle": "round,pad=0.5"})
    fig.draw_without_rendering()

    # We are using bottom alignment so three o'clock should be above the specified y coordinate.
    three = get_anchor(text, 3)
    assert three.x > 0.5
    assert three.y > 0.5

    # Ensure periodicity.
    np.testing.assert_allclose(three, get_anchor(text, 15))

    # Ensure the anchor is always further away if we have a bounding box.
    assert get_anchor(text, 3).x < get_anchor(padded, 3).x
    np.testing.assert_allclose(get_anchor(text, 3).y, get_anchor(padded, 3).y)


def test_arrow_path() -> None:
    path = rounded_path([(0, 0), (0, 1)], 0.2)
    arrow = arrow_path(path, 0.2)
    # Check for three tips plus the closing vertex.
    assert len(arrow.vertices) == 4


@pytest.mark.parametrize("method", ["corrcoef", "nmi"])
def test_dependence_heatmap(method: str) -> None:
    n = 23
    cov = np.cov(np.random.normal(0, 1, (n, 100)))
    samples = np.random.multivariate_normal(np.zeros(n), cov, 200)
    a, b, c = np.array_split(samples, 3, axis=1)
    im = dependence_heatmap({"a": a, "b": b, "c": c}, method=method)
    assert im.get_array().shape == (n, n)
