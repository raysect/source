.. _demo_mesh_observers:

Mesh Observers
==============

In this example we demonstrate two different types of observations
on a mesh surface. The first case measures the total power arriving
on a mesh surface. The second case treats each triangle in a mesh as
an observing pixel and saves the resulting power on the mesh to a
vtk data file for later visualisation in paraview.


.. literalinclude:: ../../../../demos/observers/mesh_power.py

.. figure:: mesh_observers.jpg

   a) The position of the emitting sphere inside the bunny mesh. b) A
   visualisation of the resulting power measured on the mesh surface.

This figure and calculation has been reproduced from Carr, M.,
Meakins, A., et al. “Description of complex viewing geometries of
fusion tomography diagnostics by ray-tracing.” Review of Scientific
Instruments 89.8 (2018): 083506.
