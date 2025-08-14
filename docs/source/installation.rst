
============================
Downloading and Installation
============================

Prerequisites
~~~~~~~~~~~~~

The Raysect package requires Python 3.9+, numpy and matplotlib.
IPython is recommended for interactive use.

Installation
~~~~~~~~~~~~

Raysect is available from the python package repository `pypi <https://pypi.python.org/pypi/raysect>`_. The easiest way to install Raysect is using `pip <https://pip.pypa.io/en/stable/>`_::

    python -m pip install raysect

If you prefer to install Raysect from source, the source files can be downloaded from `pypi <https://pypi.python.org/pypi/raysect>`_ or from our `development repository <https://github.com/raysect/source>`_. Once you have the source files, locate the folder containing the ``pyproject.toml`` file and run::

    python -m pip install .

This should start the Raysect compilation and installation process.
All required dependencies are automatically installed.

If you would like to play with the bleeding-edge code or contribute to development, please see the Raysect development repository on `github <https://github.com/raysect/source>`_.


Testing
~~~~~~~

A selection of test scripts can be run using Python's built-in `unittest <https://docs.python.org/3/library/unittest.html>`_ framework.
These tests are included in the source code and can be executed by running the following command in the source directory::

    python -m unittest -v

This will automatically discover and run all tests without errors or failures.

Many of the demos used throughout the Raysect documentation are distributed with the source code in
the ``demos`` folder.
