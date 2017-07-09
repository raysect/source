
Optical Material Base Classes
=============================

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
