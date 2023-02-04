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

from raysect.primitive.mesh import Mesh
import struct

STL_AUTOMATIC = 'auto'
STL_ASCII = 'ascii'
STL_BINARY = 'binary'


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
        :param str mode: The file format to load: 'ascii', 'binary', 'auto' (default='auto').
        :param kwargs: Accepts optional keyword arguments from the Mesh class.
        :rtype: Mesh

        .. code-block:: pycon

            >>> from raysect.optical import World, translate, rotate, ConstantSF, Sellmeier, Dielectric
            >>> from raysect.primitive import import_stl
            >>>
            >>> world = World()
            >>>
            >>> diamond = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0),
            >>>                      ConstantSF(1.0))
            >>>
            >>> mesh = import_stl("my_mesh.stl", scaling=1, mode='binary', parent=world,
            >>>                   transform=translate(0, 0, 0)*rotate(165, 0, 0), material=diamond)
        """

        mode = mode.lower()
        if mode == STL_ASCII:
            vertices, triangles = cls._load_ascii(filename, scaling)
        elif mode == STL_BINARY:
            vertices, triangles = cls._load_binary(filename, scaling)
        elif mode == STL_AUTOMATIC:
            try:
                vertices, triangles = cls._load_ascii(filename, scaling)
            except ValueError:
                vertices, triangles = cls._load_binary(filename, scaling)
        else:
            modes = (STL_AUTOMATIC, STL_ASCII, STL_BINARY)
            raise ValueError('Unrecognised import mode, valid values are: {}'.format(modes))

        return Mesh(vertices, triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        with open(filename, 'r') as f:
            line = cls._get_ascii_line(f)
            if not line.startswith('solid'):
                raise ValueError('ASCII STL files should start with a solid definition. The application that produced this STL '
                                 'file may be faulty, please report this error. The erroneous line: {}'.format(line))
            return cls._ascii_read_triangle(f, scaling)

    @classmethod
    def _ascii_read_triangle(cls, f, scaling):

        vertices = []
        triangles = []

        while True:
            try:
                # read
                _ = cls._get_ascii_line(f, 'facet normal')
                assert cls._get_ascii_line(f) == 'outer loop'
                v1 = cls._get_ascii_line(f, 'vertex')
                v2 = cls._get_ascii_line(f, 'vertex')
                v3 = cls._get_ascii_line(f, 'vertex')
                assert cls._get_ascii_line(f) == 'endloop'
                assert cls._get_ascii_line(f) == 'endfacet'

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
    def _get_ascii_line(cls, f, prefix=''):

        # get next line, remove whitespace i.e. indentation, newline character, etc
        line = f.readline().lower().strip()
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

        # amount of bytes to read while using buffered reading
        BUFFER_SIZE = 4096

        # the amount of bytes in the header field
        HEADER_SIZE = 80

        # the amount of bytes in the count field
        COUNT_SIZE = 4

        triangles = []

        with open(filename, 'rb') as f:

            header = f.read(HEADER_SIZE).lower()
            count, = struct.unpack('@i', f.read(COUNT_SIZE))

            # if check_size:
            #     # Check the size of the file
            #     f.seek(0, os.SEEK_END)
            #     raw_size = f.tell() - HEADER_SIZE - COUNT_SIZE
            #     expected_count = raw_size / cls.dtype.itemsize
            #     if expected_count != count:
            #         raise ValueError('Expected {} vectors but header indicates {}. '
            #                          'Is file invalid?'.format(expected_count, count))
            #     #
            #     f.seek(HEADER_SIZE + COUNT_SIZE)

            vertices = []
            triangles = []

            # Read the rest of the binary data
            while count > 0:

                # stored normal is not used, recalculated by Mesh
                nx, ny, nz, = struct.unpack('<3f', f.read(12))

                # triangle vertices
                v1x, v1y, v1z, = struct.unpack('<3f', f.read(12))
                v2x, v2y, v2z, = struct.unpack('<3f', f.read(12))
                v3x, v3y, v3z, = struct.unpack('<3f', f.read(12))

                # unused attribute
                attrib = struct.unpack('<H', f.read(2))

                # store
                vertices.append([scaling * v1x, scaling * v1y, scaling * v1z])
                vertices.append([scaling * v2x, scaling * v2y, scaling * v2z])
                vertices.append([scaling * v3x, scaling * v3y, scaling * v3z])
                triangles.append([len(vertices) - 3, len(vertices) - 2, len(vertices) - 1])
                count -= 1

            return vertices, triangles

    @classmethod
    def export_stl(cls, mesh, filename, mode=STL_BINARY):
        """
        Write a mesh instance to a STereoLithography (STL) mesh file (.stl).

        :param Mesh mesh: The Raysect mesh instance to write to STL.
        :param str filename: Mesh file path.
        :param str mode: The file format to write: 'ascii' or 'binary' (default='ascii').

        .. code-block:: pycon

            >>> mesh
            <raysect.primitive.mesh.mesh.Mesh at 0x7f2c09eac2e8>
            >>> from raysect.primitive import export_stl
            >>> export_stl(mesh, 'my_mesh.stl', mode='ascii')

        """

        if not isinstance(mesh, Mesh):
            raise ValueError('The mesh argument to write_stl() must be a valid Raysect Mesh primitive object.')

        mode = mode.lower()
        if mode == STL_ASCII:
            cls._write_ascii(mesh, filename)
        elif mode == STL_BINARY:
            cls._write_binary(mesh, filename)
        else:
            modes = (STL_ASCII, STL_BINARY)
            raise ValueError('Unrecognised export mode, valid values are: {}'.format(modes))

    @classmethod
    def _write_ascii(cls, mesh, filename):

        triangles = mesh.data.triangles
        normals = mesh.data.face_normals
        vertices = mesh.data.vertices
        num_triangles = triangles.shape[0]

        with open(filename, 'w') as f:

            mesh_name = mesh.name or 'RaysectMesh'
            mesh_name = mesh_name.replace(" ", "_")

            f.write('solid {}\n'.format(mesh_name))

            for i in range(num_triangles):
                v1, v2, v3 = triangles[i, 0:3]
                f.write('  facet normal {:6e} {:6e} {:6e}\n'.format(normals[i, 0], normals[i, 1], normals[i, 2]))
                f.write('    outer loop\n')
                f.write('      vertex  {:6e} {:6e} {:6e}\n'.format(vertices[v1, 0], vertices[v1, 1], vertices[v1, 2]))
                f.write('      vertex  {:6e} {:6e} {:6e}\n'.format(vertices[v2, 0], vertices[v2, 1], vertices[v2, 2]))
                f.write('      vertex  {:6e} {:6e} {:6e}\n'.format(vertices[v3, 0], vertices[v3, 1], vertices[v3, 2]))
                f.write('    endloop\n')
                f.write('  endfacet\n')

            f.write('endsolid {}'.format(mesh_name))

    @classmethod
    def _write_binary(cls, mesh, filename):

        triangles = mesh.data.triangles
        normals = mesh.data.face_normals
        vertices = mesh.data.vertices
        num_triangles = triangles.shape[0]

        mesh_name = mesh.name or 'RaysectMesh'
        mesh_name = mesh_name.replace(" ", "_")

        with open(filename, 'wb') as f:

            f.write(struct.pack('80s', mesh_name.encode('utf-8')))
            f.write(struct.pack('<I', num_triangles))

            for i in range(num_triangles):

                # write the face normal coordinates
                f.write(struct.pack('<f', normals[i, 0]))
                f.write(struct.pack('<f', normals[i, 1]))
                f.write(struct.pack('<f', normals[i, 2]))

                # write the three vertex coordinates
                v1, v2, v3 = triangles[i, 0:3]
                for vj in (v1, v2, v3):
                    f.write(struct.pack('<f', vertices[vj, 0]))
                    f.write(struct.pack('<f', vertices[vj, 1]))
                    f.write(struct.pack('<f', vertices[vj, 2]))

                # UINT16 – Attribute byte count
                f.write(struct.pack('<H', 0))


import_stl = STLHandler.import_stl
export_stl = STLHandler.export_stl

