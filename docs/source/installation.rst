
============================
Downloading and Installation
============================

.. _raysect github repository:   https://github.com/raysect
.. _Python Setup Tools:        http://pypi.python.org/pypi/setuptools

Prerequisites
~~~~~~~~~~~~~

The Raysect package requires Python 3, Numpy, Scipy and Matplotlib. Scipy version 0.13 or
higher is recommended. Raysect has not been tested on Python 2, currently support for Python
2 is not planned. IPython is recommended for interactive use.

Downloads
~~~~~~~~~

The latest stable version of Raysect is tagged as release v0.1 and available from
`github <https://github.com/raysect/>`_.

Installation
~~~~~~~~~~~~

Download the source, unpack it and install with::

    python setup.py install

Installation with pip and setup tools is not currently supported but will be with a future version.

Testing
~~~~~~~

A selection of test scripts can be run with the `nose` testing framework. These are routinely
run on the development version.  Running ``nosetests`` at the terminal in the source directory
should run all of these tests to completion without errors or failures.

Many of the demos used throughout the Raysect documentation are distributed with the source code in
the ``demo`` folder.

License
=======

.. literalinclude:: license.rst
