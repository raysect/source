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
