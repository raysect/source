# cython: language_level=3

# Copyright (c) 2014-2023, Dr Alex Meakins, Raysect Project
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

VTK_AUTOMATIC = 'auto'
VTK_ASCII = 'ascii'
VTK_BINARY = 'binary'


# todo: very rigid, needs to be made more flexible
# todo: add support for different file versions
# todo: add support for binary files
class VTKHandler:

    @classmethod
    def import_vtk(cls, filename, scaling=1.0, mode=VTK_AUTOMATIC, **kwargs):
        """
        Create a mesh instance from a VTK mesh data file (.vtk).

        .. warning ::
           Currently only supports VTK DataFile v2.0 and unstructured grid data with
           3 element (triangular) cells.

        :param str filename: Mesh file path.
        :param double scaling: Scale the mesh by this factor (default=1.0).
        :param str mode: The file format to load: 'ascii', 'binary', 'auto' (default='auto').
        :param kwargs: Accepts optional keyword arguments from the Mesh class.
        :rtype: Mesh
        """

        mode = mode.lower()
        if mode == VTK_ASCII:
            vertices, triangles, mesh_name = cls._load_ascii(filename, scaling)
        elif mode == VTK_BINARY:
            raise NotImplementedError('The binary .vtk loading routine has not been implemented yet.')
        elif mode == VTK_AUTOMATIC:
            try:
                vertices, triangles, mesh_name = cls._load_ascii(filename, scaling)
            except ValueError:
                # vertices, triangles, mesh_name = cls._load_binary(filename, scaling)
                raise NotImplementedError('The binary .vtk loading routine has not been implemented yet.')
        else:
            modes = (VTK_ASCII, VTK_BINARY)
            raise ValueError('Unrecognised import mode, valid values are: {}'.format(modes))

        if 'name' not in kwargs.keys():
            kwargs['name'] = mesh_name or "VTKMesh"

        return Mesh(vertices, triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        with open(filename, 'r') as f:

            # parse the file header
            assert f.readline().strip() == "# vtk DataFile Version 2.0"
            mesh_name = f.readline().strip()
            assert f.readline().strip() == "ASCII"

            if not f.readline().strip() == "DATASET UNSTRUCTURED_GRID":
                raise RuntimeError("Unrecognised dataset encountered in vtk file.")

            vertices = cls._ascii_read_vertices(f, scaling)
            triangles = cls._ascii_read_triangles(f)

            return vertices, triangles, mesh_name

    @classmethod
    def _ascii_read_vertices(cls, f, scaling):

        match = re.match("POINTS\s*([0-9]*)\s*float", f.readline().strip())
        if not match:
            raise RuntimeError("Unrecognised dataset encountered in vtk file.")
        num_points = int(match.group(1))

        vertices = np.empty((num_points, 3))

        for i in range(num_points):
            coordinates = f.readline().split()
            vertices[i, 0] = float(coordinates[0]) * scaling
            vertices[i, 1] = float(coordinates[1]) * scaling
            vertices[i, 2] = float(coordinates[2]) * scaling

        return vertices

    @classmethod
    def _ascii_read_triangles(cls, f):

        match = re.match("CELLS\s*([0-9]*)\s*([0-9]*)", f.readline())
        if not match:
            raise RuntimeError("Unrecognised dataset encountered in vtk file.")
        num_triangles = int(match.group(1))
        triangles = np.empty((num_triangles, 3), dtype=np.int32)
        for i in range(num_triangles):
            triangle_specification = f.readline().split()
            triangles[i, 0] = int(triangle_specification[1])
            triangles[i, 1] = int(triangle_specification[2])
            triangles[i, 2] = int(triangle_specification[3])

        assert f.readline().split()[0] == 'CELL_TYPES'
        for i in range(num_triangles):
            assert int(f.readline().strip()) == 5

        return triangles

    @classmethod
    def export_vtk(cls, mesh, filename, triangle_data=None, vertex_data=None, mode=VTK_ASCII):
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
        :param str mode: The file format to write: 'ascii' or 'binary' (default='ascii').
        """

        if not isinstance(mesh, Mesh):
            raise ValueError("The mesh argument to write_vtk() must be a valid Raysect Mesh primitive object.")

        mode = mode.lower()
        if mode == VTK_ASCII:
            cls._write_ascii(mesh, filename, triangle_data=triangle_data, vertex_data=vertex_data)
        elif mode == VTK_BINARY:
            raise NotImplementedError("A binary VTK writer has not been implemented yet.")
        else:
            modes = (VTK_ASCII, VTK_BINARY)
            raise ValueError('Unrecognised export mode, valid values are: {}'.format(modes))

    @classmethod
    def _write_ascii(cls, mesh, filename, triangle_data=None, vertex_data=None):

        with open(filename, 'w') as f:

            # # vtk DataFile Version 2.0
            # My Raysect mesh data
            # ASCII
            mesh_name = (mesh.name or 'RaysectMesh').replace(" ", "_")
            f.write('# vtk DataFile Version 2.0\n')
            f.write('{}\n'.format(mesh_name))
            f.write('ASCII\n')

            cls._ascii_write_geometry(f, mesh)

            if vertex_data:
                cls._ascii_write_vertex_data(f, mesh, vertex_data)

            if triangle_data:
                cls._ascii_write_triangle_data(f, mesh, triangle_data)

    @classmethod
    def _ascii_write_geometry(cls, f, mesh):

        triangles = mesh.data.triangles
        vertices = mesh.data.vertices
        num_triangles = mesh.data.triangles.shape[0]
        num_vertices = mesh.data.vertices.shape[0]

        # DATASET UNSTRUCTURED_GRID
        # POINTS  5081  float
        # 5.12135678592 3.59400404579 5.20377763887
        # 5.07735666785 3.40460816029 5.27386350545
        # ...
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('POINTS {} float\n'.format(num_vertices))
        for i in range(num_vertices):
            f.write('{} {} {}\n'.format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        # CELLS  9804 39216
        # 3 447 4361 446
        # 3 444 4248 445
        # ...
        f.write('CELLS {} {}\n'.format(num_triangles, 4 * num_triangles))
        for i in range(num_triangles):
            f.write('3 {} {} {}\n'.format(triangles[i, 0], triangles[i, 1], triangles[i, 2]))

        # CELL_TYPES  9804
        # 5
        # 5
        # ...
        f.write('CELL_TYPES {}\n'.format(num_triangles))
        for i in range(num_triangles):
            f.write('5\n')

    @classmethod
    def _ascii_write_vertex_data(cls, f, mesh, vertex_data):
        raise NotImplementedError("write_vtk() does not currently support mesh vertex data.")

    # TODO - support more VTK data types
    # TODO - add better input data validation
    @classmethod
    def _ascii_write_triangle_data(cls, f, mesh, triangle_data):

        # CELL_DATA 9804
        # SCALARS cell_scalars FLOAT
        # LOOKUP_TABLE default
        # 0
        # 1
        # ...

        num_triangles = mesh.data.triangles.shape[0]
        f.write('CELL_DATA {}\n'.format(num_triangles))

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

            f.write('SCALARS {} FLOAT\n'.format(var_name.replace(" ", "_")))
            f.write('LOOKUP_TABLE default\n')
            for value in values:
                f.write('{}\n'.format(value))


import_vtk = VTKHandler.import_vtk
export_vtk = VTKHandler.export_vtk
