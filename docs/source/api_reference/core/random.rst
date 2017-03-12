
Random
======

Random samplers
---------------

.. autofunction:: raysect.core.math.random.seed

.. autofunction:: raysect.core.math.random.uniform

.. autofunction:: raysect.core.math.random.normal

.. autofunction:: raysect.core.math.random.probability

.. autofunction:: raysect.core.math.random.point_disk

.. autofunction:: raysect.core.math.random.point_square

.. autofunction:: raysect.core.math.random.vector_sphere

.. autofunction:: raysect.core.math.random.vector_hemisphere_uniform

.. autofunction:: raysect.core.math.random.vector_hemisphere_cosine

.. autofunction:: raysect.core.math.random.vector_cone


Bulk sampling
-------------

.. autoclass:: raysect.core.math.sampler.PointSampler
   :members:
   :special-members: __call__

.. autoclass:: raysect.core.math.sampler.DiskSampler
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.RectangleSampler
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.VectorSampler
   :members:
   :special-members: __call__

.. autoclass:: raysect.core.math.sampler.ConeSampler
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.SphereSampler
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.HemisphereUniformSampler
   :members:
   :show-inheritance:

.. autoclass:: raysect.core.math.sampler.HemisphereCosineSampler
   :members:
   :show-inheritance:
