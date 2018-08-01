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

from raysect.primitive.mesh import Mesh
import struct
import os

STL_AUTOMATIC = 0
STL_ASCII = 1
STL_BINARY = 2


class STLHandler:

    @classmethod
    def import_stl(cls, filename, scaling=1.0, mode=STL_AUTOMATIC, **kwargs):
        """
        Create a mesh instance from a STereoLithography (STL) mesh file (.stl).

        Some engineering meshes are exported in different units (mm for example)
        whereas Raysect units are in m. Applying a scale factor of 0.001 would
        convert the mesh into m for use in Raysect.

        :param str filename: Mesh file path.
        :param double scaling: Scale the mesh by this factor (default=1.0).
        :param **kwargs: Accepts optional keyword arguments from the Mesh class.
        :rtype: Mesh
        """

        if mode == STL_ASCII:
            vertices, triangles = cls._load_ascii(filename, scaling)
        elif mode == STL_BINARY:
            vertices, triangles = cls._load_binary(filename, scaling)
        else:
            try:
                vertices, triangles = cls._load_ascii(filename, scaling)
            except ValueError:
                vertices, triangles = cls._load_binary(filename, scaling)

        return Mesh(vertices, triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        with open(filename, 'r') as fh:
            line = cls._get_ascii_line(fh)
            if not line.startswith('solid'):
                raise ValueError('ASCII STL files should start with a solid definition. The application that produced this STL '
                                 'file may be faulty, please report this error. The erroneous line: {}'.format(line))
            return cls._ascii_read_triangle(fh, scaling)

    @classmethod
    def _ascii_read_triangle(cls, fh, scaling):

        vertices = []
        triangles = []

        while True:
            try:
                # read
                _ = cls._get_ascii_line(fh, 'facet normal')
                assert cls._get_ascii_line(fh) == 'outer loop'
                v1 = cls._get_ascii_line(fh, 'vertex')
                v2 = cls._get_ascii_line(fh, 'vertex')
                v3 = cls._get_ascii_line(fh, 'vertex')
                assert cls._get_ascii_line(fh) == 'endloop'
                assert cls._get_ascii_line(fh) == 'endfacet'

                # store
                vertices.append([scaling * v1[0], scaling * v1[1], scaling * v1[2]])
                vertices.append([scaling * v2[0], scaling * v2[1], scaling * v2[2]])
                vertices.append([scaling * v3[0], scaling * v3[1], scaling * v3[2]])
                triangles.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])

            except AssertionError as e:
                raise RuntimeError(e)
            except StopIteration:
                break

        return vertices, triangles

    @classmethod
    def _get_ascii_line(cls, fh, prefix=''):

        # get next line, remove whitespace i.e. indentation, newline character, etc
        line = fh.readline().lower().strip()
        if prefix:
            if line.startswith(prefix):
                values = line.replace(prefix, '', 1).strip().split()
            elif line.startswith('endsolid'):
                raise StopIteration()
            else:
                raise RuntimeError('{} should start with {}'.format(line, prefix))

            if len(values) == 3:
                return [float(v) for v in values]
            else:
                raise RuntimeError('Incorrect number of values in line: {}'.format(line))
        else:
            return line

    @classmethod
    def _load_binary(cls, filename, scaling):

        #: Amount of bytes to read while using buffered reading
        BUFFER_SIZE = 4096

        #: The amount of bytes in the header field
        HEADER_SIZE = 80

        #: The amount of bytes in the count field
        COUNT_SIZE = 4

        triangles = []

        with open(filename, 'rb') as fh:

            header = fh.read(HEADER_SIZE).lower()
            count, = struct.unpack('@i', fh.read(COUNT_SIZE))

            # if check_size:
            #     # Check the size of the file
            #     fh.seek(0, os.SEEK_END)
            #     raw_size = fh.tell() - HEADER_SIZE - COUNT_SIZE
            #     expected_count = raw_size / cls.dtype.itemsize
            #     if expected_count != count:
            #         raise ValueError('Expected {} vectors but header indicates {}. '
            #                          'Is file invalid?'.format(expected_count, count))
            #     #
            #     fh.seek(HEADER_SIZE + COUNT_SIZE)

            vertices = []
            triangles = []

            # Read the rest of the binary data
            while count > 0:

                # stored normal is not used, recalculated in Triangle
                nx, ny, nz, = struct.unpack('<3f', fh.read(12))

                # triangle vertices
                v1x, v1y, v1z, = struct.unpack('<3f', fh.read(12))
                v2x, v2y, v2z, = struct.unpack('<3f', fh.read(12))
                v3x, v3y, v3z, = struct.unpack('<3f', fh.read(12))

                # unused attribute
                attrib = struct.unpack('<H', fh.read(2))

                # store
                vertices.append([scaling * v1x, scaling * v1y, scaling * v1z])
                vertices.append([scaling * v2x, scaling * v2y, scaling * v2z])
                vertices.append([scaling * v3x, scaling * v3y, scaling * v3z])
                triangles.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])
                count -= 1

            return vertices, triangles

    @classmethod
    def write_stl(cls, mesh, filename, mode="BINARY"):
        """
        Write a mesh instance to a STereoLithography (STL) mesh file (.stl).

        :param str filename: Mesh file path.
        """

        if not isinstance(mesh, Mesh):
            raise ValueError("The mesh argument to write_stl() must be a valid Raysect Mesh primitive object.")

        if mode == "ASCII":
            cls._write_ascii(mesh, filename)
        elif mode == "BINARY":
            cls._write_binary(mesh, filename)
            # raise NotImplementedError("A Binary STL writer has not been implemented yet.")
        else:
            raise ValueError("mode argument for write_stl() must be ['BINARY', 'ASCII']")

    @classmethod
    def _write_ascii(cls, mesh, filename):

        triangles = mesh.data.triangles
        normals = mesh.data.face_normals
        vertices = mesh.data.vertices
        num_triangles = triangles.shape[0]

        filehandle = open(filename, 'w')
        mesh_name = mesh.name.replace(" ", "_") or 'RaysectMesh'
        filehandle.write('solid {}\n'.format(mesh_name))

        for i in range(num_triangles):
            v1, v2, v3 = triangles[i, 0:3]
            filehandle.write('  facet normal {:6e} {:6e} {:6e}\n'.format(normals[i, 0], normals[i, 1], normals[i, 2]))
            filehandle.write('    outer loop\n')
            filehandle.write('      vertex  {:6e} {:6e} {:6e}\n'.format(vertices[v1, 0], vertices[v1, 1], vertices[v1, 2]))
            filehandle.write('      vertex  {:6e} {:6e} {:6e}\n'.format(vertices[v2, 0], vertices[v2, 1], vertices[v2, 2]))
            filehandle.write('      vertex  {:6e} {:6e} {:6e}\n'.format(vertices[v3, 0], vertices[v3, 1], vertices[v3, 2]))
            filehandle.write('    endloop\n')
            filehandle.write('  endfacet\n')

            if i == 0:
                print(v1, v2, v3)
                print(vertices[v1, 0], vertices[v1, 1], vertices[v1, 2])
                print(vertices[v2, 0], vertices[v2, 1], vertices[v2, 2])
                print(vertices[v3, 0], vertices[v3, 1], vertices[v3, 2])

        filehandle.write('endsolid {}'.format(mesh_name))
        filehandle.close()

    @classmethod
    def _write_binary(cls, mesh, filename):

        triangles = mesh.data.triangles
        normals = mesh.data.face_normals
        vertices = mesh.data.vertices
        num_triangles = triangles.shape[0]

        filehandle = open(filename, 'wb')
        mesh_name = mesh.name.replace(" ", "") or 'RaysectMesh'
        if len(mesh_name) < 80:
            mesh_name = mesh_name.ljust(80)
        else:
            mesh_name = mesh_name[0:80]
        filehandle.write(mesh_name.encode('utf-8'))

        # filehandle.write(num_triangles.to_bytes(4, byteorder='little', signed=False))
        filehandle.write(struct.pack('<I', num_triangles))

        for i in range(num_triangles):

            # write the face normal coordinates
            filehandle.write(struct.pack('<f', normals[i, 0]))
            filehandle.write(struct.pack('<f', normals[i, 1]))
            filehandle.write(struct.pack('<f', normals[i, 2]))

            # write the three vertex coordinates
            v1, v2, v3 = triangles[i, 0:3]
            for vj in (v1, v2, v3):
                filehandle.write(struct.pack('<f', vertices[vj, 0]))
                filehandle.write(struct.pack('<f', vertices[vj, 1]))
                filehandle.write(struct.pack('<f', vertices[vj, 2]))

            # UINT16 – Attribute byte count
            filehandle.write(struct.pack('<H', 0))

        filehandle.close()


import_stl = STLHandler.import_stl
write_stl = STLHandler.write_stl

