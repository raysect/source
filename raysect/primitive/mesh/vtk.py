# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


import re
import numpy as np

from raysect.primitive.mesh import Mesh


class VTKHandler:

    @classmethod
    def import_vtk(cls, filename, scaling=1.0, mode="AUTOMATIC", **kwargs):
        """
        Create a mesh instance from a VTK mesh data file (.vtk).

        Some engineering meshes are exported in different units (mm for example)
        whereas Raysect units are in m. Applying a scale factor of 0.001 would
        convert the mesh into m for use in Raysect.

        :param str filename: Mesh file path.
        :param double scaling: Scale the mesh by this factor (default=1.0).
        :param **kwargs: Accepts optional keyword arguments from the Mesh class.
        :rtype: Mesh
        """

        if mode == "ASCII":
            vertices, triangles, mesh_name = cls._load_ascii(filename, scaling)
        elif mode == "BINARY":
            raise NotImplementedError('The BINARY .vtk loading routine has not been implemented yet.')
        elif mode == "AUTOMATIC":
            try:
                vertices, triangles, mesh_name = cls._load_ascii(filename, scaling)
            except ValueError:
                # vertices, triangles, mesh_name = cls._load_binary(filename, scaling)
                raise NotImplementedError('The BINARY .vtk loading routine has not been implemented yet.')
        else:
            raise ValueError("The import_vtk() mode argument must be one of ['ASCII', 'BINARY', 'AUTOMATIC'].")

        if 'name' not in kwargs.keys():
            kwargs['name'] = mesh_name or "VTKMesh"

        return Mesh(vertices, triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        with open(filename, 'r') as fh:

            # parse the file header
            assert fh.readline().strip() == "# vtk DataFile Version 2.0"
            mesh_name = fh.readline().strip()
            assert fh.readline().strip() == "ASCII"

            vertices = cls._ascii_read_vertices(fh, scaling)

            triangles = cls._ascii_read_triangles(fh)

            return vertices, triangles, mesh_name

    @classmethod
    def _ascii_read_vertices(cls, fh, scaling):

        if not fh.readline().strip() == "DATASET UNSTRUCTURED_GRID":
            raise RuntimeError("Unrecognised dataset encountered in vtk file.")
        match = re.match("POINTS  ([0-9]*)  float", fh.readline().strip())
        if not match:
            raise RuntimeError("Unrecognised dataset encountered in vtk file.")
        num_points = int(match.group(1))

        vertices = np.empty((num_points, 3))

        for i in range(num_points):
            coordinates = fh.readline().split()
            vertices[i, 0] = float(coordinates[0]) * scaling
            vertices[i, 1] = float(coordinates[1]) * scaling
            vertices[i, 2] = float(coordinates[2]) * scaling

        return vertices

    @classmethod
    def _ascii_read_triangles(cls, fh):

        match = re.match("CELLS\s*([0-9]*)\s*([0-9]*)", fh.readline())
        if not match:
            raise RuntimeError("Unrecognised dataset encountered in vtk file.")
        num_triangles = int(match.group(1))
        triangles = np.empty((num_triangles, 3), dtype=np.int32)
        for i in range(num_triangles):
            triangle_specification = fh.readline().split()
            triangles[i, 0] = int(triangle_specification[1])
            triangles[i, 1] = int(triangle_specification[2])
            triangles[i, 2] = int(triangle_specification[3])

        assert fh.readline().split()[0] == 'CELL_TYPES'
        for i in range(num_triangles):
            assert int(fh.readline().strip()) == 5

        return triangles

    @classmethod
    def write_vtk(cls, mesh, filename, triangle_data=None, vertex_data=None, mode="ASCII"):
        """
        Write a mesh instance to a vtk mesh file (.vtk) with optional cell and point data.

        :param Mesh mesh: The Raysect mesh instance to write as VTK.
        :param str filename: Mesh file path.
        :param dict triangle_data: A dictionary of triangle face datasets to be saved along with the
          mesh. The dictionary keys will be the variable names. Each array must be 1D with length
          equal to the number of triangles in the mesh.
        :param dict vertex_data: A dictionary of vertex datasets to be saved along with the
          mesh. The dictionary keys will be the variable names. Each array must be 1D with length
          equal to the number of vertices in the mesh.
        """

        if not isinstance(mesh, Mesh):
            raise ValueError("The mesh argument to write_vtk() must be a valid Raysect Mesh primitive object.")

        if mode == "ASCII":
            cls._write_ascii(mesh, filename, triangle_data=triangle_data, vertex_data=vertex_data)
        elif mode == "BINARY":
            raise NotImplementedError("A Binary VTK writer has not been implemented yet.")
        else:
            raise ValueError("mode argument for write_vtk() must be ['BINARY', 'ASCII']")

    @classmethod
    def _write_ascii(cls, mesh, filename, triangle_data=None, vertex_data=None):

        num_triangles = mesh.data.triangles.shape[0]

        filehandle = open(filename, 'w')

        # # vtk DataFile Version 2.0
        # My Raysect mesh data
        # ASCII
        filehandle.write('# vtk DataFile Version 2.0\n')
        filehandle.write('{}\n'.format(mesh.name.replace(" ", "_") or 'RaysectMesh'))
        filehandle.write('ASCII\n')

        cls._ascii_write_geometry(mesh, filehandle)

        if vertex_data:
            cls._ascii_write_vertex_data(mesh, filehandle, vertex_data)

        if triangle_data:
            cls._ascii_write_triangle_data(mesh, filehandle, triangle_data)

        filehandle.close()

    @classmethod
    def _ascii_write_geometry(cls, mesh, filehandle):

        triangles = mesh.data.triangles
        vertices = mesh.data.vertices
        num_triangles = mesh.data.triangles.shape[0]
        num_verticies = mesh.data.vertices.shape[0]

        # DATASET UNSTRUCTURED_GRID
        # POINTS  5081  float
        # 5.12135678592 3.59400404579 5.20377763887
        # 5.07735666785 3.40460816029 5.27386350545
        # ...
        filehandle.write('DATASET UNSTRUCTURED_GRID\n')
        filehandle.write('POINTS {} float\n'.format(num_verticies))
        for i in range(num_verticies):
            filehandle.write('{} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        # CELLS  9804 39216
        # 3 447 4361 446
        # 3 444 4248 445
        # ...
        filehandle.write('CELLS {} {}\n'.format(num_triangles, 4 * num_triangles))
        for i in range(num_triangles):
            filehandle.write('3 {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2]))

        # CELL_TYPES  9804
        # 5
        # 5
        # ...
        filehandle.write('CELL_TYPES {}\n'.format(num_triangles))
        for i in range(num_triangles):
            filehandle.write('5\n')

    @classmethod
    def _ascii_write_vertex_data(cls, mesh, filehandle, vertex_data):
        raise NotImplementedError("write_vtk() does not currently support mesh vertex data.")

    # TODO - support more VTK data types
    # TODO - add better input data validation
    @classmethod
    def _ascii_write_triangle_data(cls, mesh, filehandle, triangle_data):

        # CELL_DATA 9804
        # SCALARS cell_scalars FLOAT
        # LOOKUP_TABLE default
        # 0
        # 1
        # ...

        num_triangles = mesh.data.triangles.shape[0]
        filehandle.write('CELL_DATA {}\n'.format(num_triangles))

        error_msg = "The triangle_data argument in write_vtk() must be a dictionary or arrays/lists " \
                    "with length equal to the number of triangles."

        if not isinstance(triangle_data, dict):
            raise ValueError(error_msg)

        for var_name, values in triangle_data.items():

            try:
                if not len(values) == num_triangles:
                    raise ValueError(error_msg)
            except TypeError:
                raise ValueError(error_msg)

            filehandle.write('SCALARS {} FLOAT\n'.format(var_name.replace(" ", "_")))
            filehandle.write('LOOKUP_TABLE default\n')
            for value in values:
                filehandle.write('{}\n'.format(value))


import_vtk = VTKHandler.import_vtk
write_vtk = VTKHandler.write_vtk
