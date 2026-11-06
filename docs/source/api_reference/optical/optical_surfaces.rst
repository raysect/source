Optical Surface Materials
=========================

Absorber
--------

.. autoclass:: raysect.optical.material.absorber.AbsorbingSurface
   :show-inheritance:


Emitters
--------

.. autoclass:: raysect.optical.material.emitter.UnitySurfaceEmitter
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.UniformSurfaceEmitter
   :show-inheritance:

.. autoclass:: raysect.optical.material.AnisotropicSurfaceEmitter
   :show-inheritance:

.. autoclass:: raysect.optical.material.Checkerboard
   :show-inheritance:

Lambertian
----------

.. autoclass:: raysect.optical.material.lambert.Lambert
   :show-inheritance:


Tabulated BRDF
--------------

.. autoclass:: raysect.optical.material.custom.CustomBRDF
   :show-inheritance:

A ``CustomBRDF`` can describe an isotropic surface with a user-supplied BRDF
sampled over wavelength and direction. The array axes are ordered as
``(wavelength, incidence polar angle, relative azimuth, outgoing polar angle)``.
Wavelengths are specified in nanometres and may be unevenly spaced. The
angular axes are evenly spaced in radians: both polar axes include 0 and
:math:`\pi/2`, while relative azimuth covers :math:`[0, 2\pi)` without a
duplicate endpoint. Relative azimuth is the outgoing azimuth minus the
incident azimuth, modulo :math:`2\pi`; both directions point away from the
surface. BRDF values have units of sr\ :sup:`-1`.

For example, an isotropic Lambertian table with 40% reflectance can be made as
follows::

    import numpy as np
    from raysect.optical.material import CustomBRDF

    wavelengths = [500.0, 700.0]  # nm
    table = np.full((2, 19, 36, 19), 0.4 / np.pi)
    material = CustomBRDF(table, wavelengths)

The material linearly interpolates between samples. For wavelengths outside
the table, it uses the nearest endpoint. A single-wavelength table is treated
as constant over wavelength; a three-dimensional angular table is also
accepted as wavelength independent. Sampling uses a wavelength-independent
cosine-weighted hemisphere, which may converge slowly for sharp peaks.
The constructor copies the table into contiguous float64 storage, so account
for the input and the copy when planning memory use. The caller is responsible
for supplying non-negative, reciprocal and energy-conserving BRDF data.

For a runnable wavelength-dependent example, see
``demos/materials/custom_brdf.py`` in the source repository.


Conductors
----------

.. autoclass:: raysect.optical.material.conductor.Conductor
   :show-inheritance:

.. autoclass:: raysect.optical.material.conductor.RoughConductor
   :show-inheritance:


Dielectrics
-----------

.. autoclass:: raysect.optical.material.dielectric.Dielectric
   :show-inheritance:

.. autoclass:: raysect.optical.material.dielectric.Sellmeier
   :show-inheritance:


Modifiers
---------

.. autoclass:: raysect.optical.material.modifiers.Blend
   :show-inheritance:

.. autoclass:: raysect.optical.material.modifiers.Add
   :show-inheritance:

.. autoclass:: raysect.optical.material.modifiers.VolumeTransform
   :show-inheritance:

.. autoclass:: raysect.optical.material.modifiers.Roughen
   :show-inheritance:


Debugging
---------

This module contains materials to aid with debugging and creating test scenes.

.. autoclass:: raysect.optical.material.emitter.Checkerboard
   :show-inheritance:

.. autoclass:: raysect.optical.material.debug.Light
   :show-inheritance:

.. autoclass:: raysect.optical.material.debug.PerfectReflectingSurface
   :show-inheritance:
