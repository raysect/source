
Random
======

Random samplers
---------------

.. autofunction:: raysect.core.math.random.seed

.. autofunction:: raysect.core.math.random.uniform

.. autofunction:: raysect.core.math.random.normal

.. autofunction:: raysect.core.math.random.probability


3D Surface Samplers
-------------------

.. autoclass:: raysect.core.math.sampler.surface3d.SurfaceSampler3D
   :members:
   :special-members: __call__

.. autoclass:: raysect.core.math.sampler.surface3d.DiskSampler3D
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.surface3d.RectangleSampler3D
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.surface3d.TriangleSampler3D
   :members:
   :show-inheritance:


Solid Angle Samplers
--------------------

.. autoclass:: raysect.core.math.sampler.solidangle.SolidAngleSampler
   :members:
   :special-members: __call__

.. autoclass:: raysect.core.math.sampler.solidangle.SphereSampler
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.solidangle.HemisphereUniformSampler
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.solidangle.HemisphereCosineSampler
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.solidangle.ConeUniformSampler
   :show-inheritance:
