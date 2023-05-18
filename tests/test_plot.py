from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
import numpy as np
from snippets.plot import plot_band


def test_plot_band() -> None:
    x = np.linspace(0, 2 * np.pi, 20)
    ys = np.sin(x) + np.random.normal(0, .25, (100, x.size))
    line, band = plot_band(x, ys)
    assert isinstance(line, Line2D)
    assert isinstance(band, PolyCollection)
