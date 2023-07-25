from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.path import Path
import numpy as np
import pytest
from snippets.plot import plot_band, rounded_path


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
