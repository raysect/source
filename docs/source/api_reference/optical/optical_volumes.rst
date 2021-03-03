

Optical Volume Materials
========================

Basic Emitters
--------------

.. autoclass:: raysect.optical.material.emitter.UnityVolumeEmitter
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.UniformVolumeEmitter
   :show-inheritance:


.. autoclass:: raysect.optical.material.emitter.HomogeneousVolumeEmitter
   :members: emission_function
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.InhomogeneousVolumeEmitter
   :members: emission_function
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.inhomogeneous.VolumeIntegrator
   :members:

.. autoclass:: raysect.optical.material.emitter.inhomogeneous.NumericalIntegrator
   :members:
   :show-inheritance:

Special Emitters
----------------

.. autoclass:: raysect.optical.material.emitter.RegularGridEmitter
   :members: voxel_index, cache_valid, cache_empty, cache_override, cache_build, integrate, add_emission_to_array
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.CartesianRegularEmitter
   :members: emission_function
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.CylindricalRegularEmitter
   :members: emission_function
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.RegularGridIntegrator
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.CartesianRegularIntegrator
   :members: integrate
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.CylindricalRegularIntegrator
   :members: integrate
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.RegularGridVolume

.. autoclass:: raysect.optical.material.emitter.RegularGridBox
   :show-inheritance:

.. autoclass:: raysect.optical.material.emitter.RegularGridCylinder
   :show-inheritance:

