
.. _demo_point_inside_material:

Point Inside A Material
=======================

This demo shows how the world.contains() method can be used to query the world for
all primitives that intersect the test point. This simple scene contains a Sphere
at the origin of radius 0.5m. A grid of test points is generated in the x-y plane.
Each point is tested to see if it lies inside the sphere. The test points are
plotted in the figure, coloured by their true (inside)/false (outside) categorisation.


.. literalinclude:: ../../../../demos/core/world_contains_point.py

.. image:: test_point_inside_material.png
   :align: center
