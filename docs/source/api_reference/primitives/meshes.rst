
Meshes
======

.. autoclass:: raysect.primitive.mesh.mesh.Mesh
   :members: from_file, load, save
   :show-inheritance:

.. warning::
   Some engineering meshes are exported in different units (mm for example)
   whereas Raysect units are in m. Therefore, all import functions have an
   optional scaling factor argument that allows you to convert the mesh
   units on import. For example, if a given engineering mesh was in mm,
   applying a scale factor of 0.001 would convert the mesh into m for
   use in Raysect.

.. autofunction:: raysect.primitive.mesh.obj.import_obj

.. autofunction:: raysect.primitive.mesh.obj.export_obj

.. autofunction:: raysect.primitive.mesh.stl.import_stl

.. autofunction:: raysect.primitive.mesh.stl.export_stl

.. autofunction:: raysect.primitive.mesh.ply.import_ply

.. autofunction:: raysect.primitive.mesh.ply.export_ply

.. autofunction:: raysect.primitive.mesh.vtk.import_vtk

.. autofunction:: raysect.primitive.mesh.vtk.export_vtk
