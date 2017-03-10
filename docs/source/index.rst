.. title:: Home

|

.. image:: RaysectLogo_small.png
   :align: center

|
|

Welcome
=======

Welcome to Raysect, an OOP ray-tracing framework for Python. Raysect has been built with scientific ray-tracing in mind. Some of its features include:

* Fully spectral, high precision. Supports scientific ray-tracing of spectra from physical light sources such as plasmas.
* All core loops are written in cython for speed.
* Easily extensible, written with user customisation of materials and emissive sources in mind.
* Different observer types supported such as Pinhole cameras and optical fibres.

Installation
------------

Raysect is available from the python package repository `pypi <https://pypi.python.org/pypi/raysect>`_. The easiest way to install Raysect is using `pip <https://pip.pypa.io/en/stable/>`_::

    pip install raysect

If pip is not available, the source files can be downloaded from `pypi <https://pypi.python.org/pypi/raysect>`_ or from our `development repository <https://github.com/raysect/source>`_. Once you have the source files, locate the folder containing setup.py and run::

    python setup.py install

If all the required dependencies are present (cython, numpy, scipy and matplotlib), this should start the Raysect compilation and installation process.

If you would like to play with the bleeding-edge code or contribute to development, please see the Raysect development repository on `github <https://github.com/raysect/source>`_.

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Table of Contents
   :name: mastertoc

   introduction
   installation
   how_it_works
   quickstart_guide
   primitives
   materials
   conventions
   gallery
   references
   license
   help


.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: Demonstrations
   :name: demonstrations

   demonstrations/demonstrations


.. toctree::
   :maxdepth: 3
   :numbered:
   :caption: API Reference
   :name: apireferenceto

   api_reference/core/core
   api_reference/primitives/primitives
   api_reference/optical/optical

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

