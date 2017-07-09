
Monte-Carlo Sampling
====================

The coordinate transformation is applied by multiplying the column vector for
the desired Point3D/Vector3D against the transformation matrix. For example,
if the original vector :math:`\\vec{V_a}` is in space A and the transformation matrix
:math:`\\mathbf{T_{AB}}` describes the position and orientation of Space A relative
to Space B, then the multiplication


General Monte-carlo integration
-------------------------------

.. math::

   Q_{N} = \\frac{1}{N} \\sum_{i=1}^{N} \frac{f(x_0) \\cos(\\theta)}{p(x_0)}


Uniform Sampling
----------------

.. math::

   p(x_0) = \\frac{1}{2 \\pi}


.. math::

   Q_{N} = \\frac{2 \\pi}{N} \\sum_{i=1}^{N} f(x_0) \\cos(\\theta)


Cosine Weighted
---------------

.. math::

   p(x_0) = \\frac{\\cos(\\theta)}{\\pi}

.. math::

   Q_{N} = \\frac{\\pi}{N \\cos(\\theta)} \\sum_{i=1}^{N} f(x_0) \\cos(\\theta) = \\frac{\\pi}{N} \\sum_{i=1}^{N} f(x_0)


Irradiance vs Radiance
----------------------

Irradiance at surface.

.. math::

   I = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{f(x_0) \\cos(\\theta)}{p(x)}

Conversion to radiance:

.. math::

   radiance = I / (2 \\pi)


Implementation
--------------

Implementation in base observer:

.. math::

   radiance = \\frac{1}{2 \\pi} \\frac{1}{N} \\sum_{i=1}^{N} w f(x)

- f(x) is obtained from ray-tracing in base class
- w is cos(theta)/p(x).
- for an observer its this.
- or a surface its this.


