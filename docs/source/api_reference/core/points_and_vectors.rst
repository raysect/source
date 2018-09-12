
Points and Vectors
==================

Two Dimensions
--------------

.. autoclass:: raysect.core.math.point.Point2D
   :members:
   :special-members: __getitem__, __setitem__, __iter__, __add__, __sub__

.. autoclass:: raysect.core.math.vector.Vector2D
   :members:
   :special-members: __getitem__, __setitem__, __iter__, __neg__, __add__, __sub__, __mul__, __truediv__

Three Dimensions
----------------

.. autoclass:: raysect.core.math.point.Point3D
   :members:
   :special-members: __getitem__, __setitem__, __iter__, __add__, __sub__, __mul__


.. autoclass:: raysect.core.math.vector.Vector3D
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __iter__, __neg__, __add__, __sub__, __mul__, __truediv__

.. autoclass:: raysect.core.math.normal.Normal3D
   :members: as_vector, transform_with_inverse
