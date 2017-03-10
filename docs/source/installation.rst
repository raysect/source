
============================
Downloading and Installation
============================

.. _raysect github repository:   https://github.com/raysect
.. _Python Setup Tools:        http://pypi.python.org/pypi/setuptools

Prerequisites
~~~~~~~~~~~~~

The Raysect package requires Python 3.3+, numpy, scipy and matplotlib. Scipy version 0.13 or
higher is recommended. Raysect has not been tested on Python 2, currently support for Python
2 is not planned. IPython is recommended for interactive use.

Installation
~~~~~~~~~~~~

Raysect is available from the python package repository `pypi <https://pypi.python.org/pypi/raysect>`_. The easiest way to install Raysect is using `pip <https://pip.pypa.io/en/stable/>`_::

    pip install raysect

If pip is not available, the source files can be downloaded from `pypi <https://pypi.python.org/pypi/raysect>`_ or from our `development repository <https://github.com/raysect/source>`_. Once you have the source files, locate the folder containing setup.py and run::

    python setup.py install

If all the required dependencies are present (cython, numpy, scipy and matplotlib), this should start the Raysect compilation and installation process.

If you would like to play with the bleeding-edge code or contribute to development, please see the Raysect development repository on `github <https://github.com/raysect/source>`_.


Testing
~~~~~~~

A selection of test scripts can be run with the `nose` testing framework. These are routinely
run on the development version.  Running ``nosetests`` at the terminal in the source directory
should run all of these tests to completion without errors or failures.

Many of the demos used throughout the Raysect documentation are distributed with the source code in
the ``demo`` folder.
