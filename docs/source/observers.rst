
**********************
Sampling and Observers
**********************

=================
Observer Equation
=================

The total intensity measured by an observing pixel is given be the integral of
the incident emission over the collecting solid angle :math:`\Omega` and surface
area :math:`A`.

.. math::

   I = \int_{A} \int_{\Omega} L_{i}(p, \omega_i) \times \cos (\theta_i) d\omega_i dA

Here, :math:`L_{i}(p, \omega_i)` is the incident emission at a given point :math:`p`
and incident angle :math:`\omega_i` on the observing surface. :math:`\cos (\theta_i)`
is a geometry factor describing the increase in effective observing area as the incident
rays become increasingly parallel to the surface.

.. figure:: images/pixel_sampling.png
   :align: center

   **Caption:** Example sampling geometry for a pixel in a camera. (Image credit: Carr, M.,
   Meakins, A., et. al. (2017))

=======================
Monte-Carlo Integration
=======================

The integral in the observer equation is exact but difficult to evaluate analytically
for most physically relevant scenes. Monte-Carlo integration is a technique for
approximating the value of an integral by simulating a random process with random
numbers.

The simplest form of Monte-Carlo integration is the average value method, which starts by
considering the integral

.. math::

   I = \int_{a}^{b} f(x) dx

The average value of the function :math:`f(x)` over the domain can be expressed in terms
of the integral over the same domain.

.. math::

   <f> = \frac{1}{b-a} \int_{a}^{b} f(x) dx = \frac{I}{b-a}

Therefore, through manipulation we can express our integral in terms of the average of
:math:`f(x)`. We can measure :math:`<f(x)>` by sampling :math:`f(x)` at :math:`N` points
:math:`x_1`, :math:`x_2`, ..., :math:`x_N` chosen randomly between :math:`a` and
:math:`b`.

.. math::

   I \approx \frac{b-a}{N} \sum_{i=1}^{N} f(x_i) = \frac{1}{N} \sum_{i=1}^{N} \frac{f(x_i)}{p(x_i)}

:math:`p(x_i) = 1/(a-b)` is the uniform distribution we are sampling from. For a rectangular
pixel in a camera the sampling distributions would be uniformly distributed 2D points in the
square area :math:`A` and 3D vectors in the hemisphere :math:`\Omega`.

In general, the standard error on the approximation scales as :math:`1/\sqrt{N}` with
:math:`N` samples.

===================
Importance Sampling
===================

The average value formula works well for smoothly varying functions but becomes inefficient when
the integral contains steep gradients or a divergence, for example, a bright point light source.
The standard error on the integral estimator can become large in such circumstances.

We can get around these problems by drawing the sample points :math:`x_i` from a non-uniform
probability density function (i.e. higher density for regions with stronger emission). :math:`p(x_i)`
now becomes

.. math::

   p(x_i) = \frac{w(x_i)}{\int_{a}^{b} w(x) dx}

where :math:`w(x)` is the weight function and has been normalised so that its integral is 1. The
estimator for the integral is now

.. math::

   I \approx \frac{1}{N} \sum_{i=1}^{N} \frac{f(x_i)}{w(x_i)} \int_{a}^{b} w(x) dx

This is the fundamental formula of importance sampling and a generalisation of the average
value method (try :math:`w(x) = 1`). It allows estimation of the integral :math:`I` by performing
a weighted sum, which can be weighted to have higher density in regions of interest. The price we
pay is that the random samples are being drawn from a more complicated distribution.

Importance sampling exploits the fact that the Monte-Carlo estimator converges fastest when samples
are taken from a distribution p(x) that is similar to the function f(x) in the integrand
(i.e. concentrate the calculations where the integrand is high).

Cosine distribution
-------------------

Sampling the lights
-------------------

Sampling the BRDF
-----------------

============================
Multiple Importance Sampling
============================




