from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
from matplotlib import pyplot as plt
import numpy as np
import pytest
from snippets.plot import label_axes, plot_band, rounded_path
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
