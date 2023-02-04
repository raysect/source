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

import struct
import numpy as np
from raysect.primitive.mesh import Mesh

PLY_AUTOMATIC = 'auto'
PLY_ASCII = 'ascii'
PLY_BINARY = 'binary'


# TODO: missing vertex normal support
# TODO: add support for other data types, e.g. face colours and other arbitrary data
# TODO: this implementation is currently very rigid. Some valid PLY files might fail unnecessarily at present.
class PLYHandler:

    @classmethod
    def import_ply(cls, filename, scaling=1.0, mode=PLY_AUTOMATIC, **kwargs):
        """
        Create a mesh instance from a Polygon File Format (PLY) mesh file (.ply).
        Note PLY is also known as the Stanford Triangle Format.

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
            >>> from raysect.primitive import import_ply
            >>>
            >>> world = World()
            >>>
            >>> diamond = Dielectric(Sellmeier(0.3306, 4.3356, 0.0, 0.1750**2, 0.1060**2, 0.0),
            >>>                      ConstantSF(1.0))
            >>>
            >>> mesh = import_ply("your_mesh.ply", scaling=1, mode='binary', parent=world,
            >>>                   transform=translate(0, 0, 0)*rotate(165, 0, 0), material=diamond)
        """

        mode = mode.lower()
        if mode == PLY_ASCII:
            vertices, triangles = cls._load_ascii(filename, scaling)

        elif mode == PLY_BINARY:
            vertices, triangles = cls._load_binary(filename, scaling)

        elif mode == PLY_AUTOMATIC:
            try:
                vertices, triangles = cls._load_ascii(filename, scaling)
            except ValueError:
                vertices, triangles = cls._load_binary(filename, scaling)

        else:
            modes = (PLY_AUTOMATIC, PLY_ASCII, PLY_BINARY)
            raise ValueError('Unrecognised import mode, valid values are: {}'.format(modes))

        return Mesh(vertices, triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        # parse the file header
        try:

            with open(filename, 'r') as f:

                assert f.readline().strip() == "ply"
                assert f.readline().strip() == "format ascii 1.0"

                # skip over comments
                while True:
                    line = f.readline().strip()
                    if line[0] == "comment":
                        continue
                    break

                # read out vertex specification
                assert (line[0] == 'element' and line[1] == 'vertex')
                num_vertices = int(line[2])
                assert f.readline().strip() == 'property float x'
                assert f.readline().strip() == 'property float y'
                assert f.readline().strip() == 'property float z'

                line = f.readline().strip().split()
                assert (line[0] == 'element' and line[1] == 'face')
                num_triangles = int(line[2])
                assert f.readline().strip() == 'property list uchar int vertex_indices'

                # go to end of header
                while True:
                    line = f.readline().strip().split()
                    if line[0] == "end_header":
                        break

                # read in the vertices
                vertices = np.empty((num_vertices, 3))
                for i in range(num_vertices):
                    line = f.readline().strip().split()
                    vertices[i, :] = float(line[0])*scaling, float(line[1])*scaling, float(line[2])*scaling

                # read in the triangles
                triangles = np.empty((num_triangles, 3), dtype=np.int32)
                for i in range(num_triangles):
                    line = f.readline().strip().split()
                    if not int(line[0]) == 3:
                        raise ValueError("Raysect meshes can only handle triangles.")
                    triangles[i, :] = float(line[1])*scaling, float(line[2])*scaling, float(line[3])*scaling

            return vertices, triangles

        except AssertionError:
            raise ValueError("This file is not a valid PLY file.")

    @classmethod
    def _load_binary(cls, filename, scaling):

        with open(filename, 'rb') as f:

            assert f.readline().decode().strip() == "ply"
            assert f.readline().decode().strip() == "format binary_little_endian 1.0"

            # skip over comments
            while True:
                line = f.readline().decode().strip().split()
                if line[0] == "comment":
                    continue
                break

            # read out vertex specification
            assert (line[0] == 'element' and line[1] == 'vertex')
            num_vertices = int(line[2])
            assert f.readline().decode().strip() == 'property float x'
            assert f.readline().decode().strip() == 'property float y'
            assert f.readline().decode().strip() == 'property float z'

            line = f.readline().decode().strip().split()
            assert (line[0] == 'element' and line[1] == 'face')
            num_triangles = int(line[2])
            assert f.readline().decode().strip() == 'property list uchar int vertex_indices'

            # go to end of header
            while True:
                line = f.readline().decode().strip().split()
                if line[0] == "end_header":
                    break

            # read in the vertices
            vertices = np.empty((num_vertices, 3))
            for i in range(num_vertices):
                next_line = f.read(12)
                vertices[i, :] = struct.unpack('<fff', next_line)
            vertices *= scaling

            # read in the triangles
            triangles = np.empty((num_triangles, 3), dtype=np.int32)
            for i in range(num_triangles):
                next_line = f.read(13)
                data = struct.unpack('<BIII', next_line)
                if not data[0] == 3:
                    raise ValueError("Raysect meshes can only handle triangles.")
                triangles[i, :] = data[1:]

        return vertices, triangles

    @classmethod
    def export_ply(cls, mesh, filename, mode=PLY_BINARY, comment="Generated by Raysect"):
        """
        Write a mesh instance to a Polygon File Format (PLY) mesh file (.ply).
        Note PLY is also known as the Stanford Triangle Format.

        :param Mesh mesh: The Raysect mesh instance to write as PLY.
        :param str filename: Mesh file path.
        :param str mode: The file format to write: 'ascii', 'binary' (default='ascii').
        :param str comment: An optional string comment to include in the output file,
          can be multiple lines.

        .. code-block:: pycon

            >>> mesh
            <raysect.primitive.mesh.mesh.Mesh at 0x7f2c09eac2e8>
            >>> from raysect.primitive import export_ply
            >>> export_ply(mesh, 'my_mesh.ply', mode='ascii', comment='My mesh')
        """

        if not isinstance(mesh, Mesh):
            raise ValueError("The mesh argument to write_ply() must be a valid Raysect Mesh primitive object.")

        mode = mode.lower()
        if mode == PLY_ASCII:
            cls._write_ascii(mesh, filename, comment=comment)
        elif mode == PLY_BINARY:
            cls._write_binary(mesh, filename, comment=comment)
        else:
            modes = (PLY_ASCII, PLY_BINARY)
            raise ValueError('Unrecognised export mode, valid values are: {}'.format(modes))

    @classmethod
    def _write_ascii(cls, mesh, filename, comment=None):

        with open(filename, "w") as f:

            vertices = mesh.data.vertices
            triangles = mesh.data.triangles

            num_vertices = vertices.shape[0]
            num_triangles = triangles.shape[0]

            # write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            comment_lines = comment.splitlines()
            for comment in comment_lines:
                f.write("comment " + comment + "\n")
            f.write("element vertex {}\n".format(num_vertices))
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("element face {}\n".format(num_triangles))
            f.write("property list uchar int vertex_index\n")
            f.write("end_header\n")

            # write vertices
            for vertex in vertices:
                f.write("{:.6e} {:6e} {:6e}\n".format(vertex[0], vertex[1], vertex[2]))

            # TODO: handle vertex normals

            # write triangles
            for triangle in triangles:
                f.write("3 {:d} {:d} {:d}\n".format(triangle[0], triangle[1], triangle[2]))

    @classmethod
    def _write_binary(cls, mesh, filename, comment=None):

        with open(filename, "wb") as f:

            vertices = mesh.data.vertices
            triangles = mesh.data.triangles

            num_vertices = vertices.shape[0]
            num_triangles = triangles.shape[0]

            # write header
            f.write("ply\n".encode())
            f.write("format binary_little_endian 1.0\n".encode())
            comment_lines = comment.splitlines()
            for comment in comment_lines:
                f.write(("comment " + comment + "\n").encode())
            f.write("element vertex {}\n".format(num_vertices).encode())
            f.write("property float x\n".encode())
            f.write("property float y\n".encode())
            f.write("property float z\n".encode())
            f.write("element face {}\n".format(num_triangles).encode())
            f.write("property list uchar int vertex_index\n".encode())
            f.write("end_header\n".encode())

            # write vertices
            for vertex in vertices:
                f.write(struct.pack('<fff', vertex[0], vertex[1], vertex[2]))

            # TODO: handle vertex normals

            # write triangles
            for triangle in triangles:
                f.write(struct.pack('<BIII', 3, triangle[0], triangle[1], triangle[2]))


import_ply = PLYHandler.import_ply
export_ply = PLYHandler.export_ply
