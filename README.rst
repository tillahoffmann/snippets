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

- :docline:`snippets.call_with_timeout.call_with_timeout`
- :docline:`snippets.empirical_distribution.sample_empirical_cdf`
- :docline:`snippets.empirical_distribution.sample_empirical_pdf`
- :docline:`snippets.nearest_neighbor_sampler.NearestNeighborSampler`
- :docline:`snippets.param_dict.from_param_dict`
- :docline:`snippets.param_dict.to_param_dict`
- :docline:`snippets.plot.label_axes`
- :docline:`snippets.plot.plot_band`
- :docline:`snippets.plot.rounded_path`
- :docline:`snippets.tensor_data_loader.TensorDataLoader`
- :docline:`snippets.timer.Timer`

🧑‍💻 Command Line Tools
------------------------

.. sh:: python -m snippets.check_references --help

.. toctree::
    :maxdepth: 1
    :hidden:

    docs/call_with_timeout
    docs/empirical_distribution
    docs/nearest_neighbor_sampler
    docs/param_dict
    docs/plot
    docs/tensor_data_loader
    docs/timer
    docs/util
