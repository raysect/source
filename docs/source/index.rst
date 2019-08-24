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

.. image:: demonstrations/optics/prism_720x405.png
   :align: center


The Raysect Development Team:
-----------------------------

* Dr Alex Meakins
* Dr Matthew Carr


Quick Installation
------------------

The easiest way to install Raysect is using `pip <https://pip.pypa.io/en/stable/>`_::

    pip install raysect

For more detailed installation instructions see :any:`installation`.

Need Help?
----------

Please post a question on the `github issue queue <https://github.com/raysect/source/issues>`_.

.. toctree::
   :maxdepth: 2
   :numbered:
   :caption: Table of Contents
   :name: mastertoc

   introduction
   license
   installation
   how_it_works
   quickstart_guide
   observers
   primitives
   materials
   conventions
   references
   glossary


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

