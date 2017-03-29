
Optical Materials
=================

Optical Material Base Classes
-----------------------------

The Bidirectional Scattering Distribution Function (BSDF) function describes how
incoming light is split into the reflected (bidirectional reflectance distribution
function BRDF) and transmitted (bidirectional transmittance distribution function
BTDF) components. See
`wikipedia <https://en.wikipedia.org/wiki/Bidirectional_scattering_distribution_function>`_.


.. autoclass:: raysect.optical.material.material.Material
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.material.NullSurface
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.material.NullVolume
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.material.DiscreteBSDF
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.material.ContinuousBSDF
   :members:
   :show-inheritance:


Absorber
--------

.. autoclass:: raysect.optical.material.absorber.AbsorbingSurface
   :members:
   :show-inheritance:


Emitter
-------

.. autoclass:: raysect.optical.material.emitter.UniformSurfaceEmitter
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.VolumeEmitterHomogeneous
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.VolumeEmitterInhomogeneous
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.UnityVolumeEmitter
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.UniformVolumeEmitter
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.Checkerboard
   :members:
   :show-inheritance:


Lambertian
----------

.. autoclass:: raysect.optical.material.lambert.Lambert
   :members:
   :show-inheritance:


Conductors
----------

.. autoclass:: raysect.optical.material.conductor.Conductor
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.conductor.RoughConductor
   :members:
   :show-inheritance:


Dielectrics
-----------

.. autoclass:: raysect.optical.material.dielectric.Sellmeier
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.dielectric.Dielectric
   :members:
   :show-inheritance:


Modifiers
---------

.. autoclass:: raysect.optical.material.modifiers.Roughen
   :members:
   :show-inheritance:


Debugging
---------

This module contains materials to aid with debugging.

.. autoclass:: raysect.optical.material.modifiers.Light
   :members:
   :show-inheritance:

.. autoclass:: raysect.optical.material.modifiers.PerfectReflectingSurface
   :members:
   :show-inheritance:

.. automodule:: raysect.optical.material.debug
   :members:


