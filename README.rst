✂️ Snippets
===========

.. image:: https://github.com/tillahoffmann/snippets/actions/workflows/main.yml/badge.svg
    :target: https://github.com/tillahoffmann/snippets/
.. image:: https://readthedocs.org/projects/scientific-snippets/badge/?version=latest
    :target: https://scientific-snippets.readthedocs.io/en/latest/?badge=latest


This repository contains useful, tested code snippets and command line tools.

.. code-block:: bash

    pip install git+https://github.com/tillahoffmann/snippets.git[@optional commit hash]

✂️ Code Snippets
----------------

- :docitem:`snippets.call_with_timeout.call_with_timeout`
- :docitem:`snippets.empirical_distribution.sample_empirical_cdf`
- :docitem:`snippets.empirical_distribution.sample_empirical_pdf`
- :docitem:`snippets.nearest_neighbor_sampler.NearestNeighborSampler`
- :docitem:`snippets.nn.Affine`
- :docitem:`snippets.nn.StopOnPlateau`
- :docitem:`snippets.param_dict.from_param_dict`
- :docitem:`snippets.param_dict.to_param_dict`
- :docitem:`snippets.plot.arrow_path`
- :docitem:`snippets.plot.dependence_heatmap`
- :docitem:`snippets.plot.get_anchor`
- :docitem:`snippets.plot.label_axes`
- :docitem:`snippets.plot.parameterization_mutual_info`
- :docitem:`snippets.plot.plot_band`
- :docitem:`snippets.plot.rounded_path`
- :docitem:`snippets.stats.GaussianKernelDensity`
- :docitem:`snippets.tensor_data_loader.TensorDataLoader`
- :docitem:`snippets.timer.Timer`

🧑‍💻 Command Line Tools
------------------------

- :docitem:`snippets.check_references.CheckReferences`

.. toctree::
    :maxdepth: 1
    :hidden:

    docs/call_with_timeout
    docs/check_references
    docs/empirical_distribution
    docs/nearest_neighbor_sampler
    docs/nn
    docs/param_dict
    docs/plot
    docs/stats
    docs/tensor_data_loader
    docs/timer
    docs/util
