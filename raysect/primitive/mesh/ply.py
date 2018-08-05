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

import struct
import numpy as np
from raysect.primitive.mesh import Mesh


# TODO - add support for other data types, e.g. face colours and other arbitrary data
class PLYHandler:

    @classmethod
    def import_ply(cls, filename, scaling=1.0, mode="AUTOMATIC", **kwargs):
        """
        Create a mesh instance from a Polygon File Format (PLY) mesh file (.ply).
        Note PLY is also known as the Stanford Triangle Format.

        Some engineering meshes are exported in different units (mm for example)
        whereas Raysect units are in m. Applying a scale factor of 0.001 would
        convert the mesh into m for use in Raysect.

        :param str filename: Mesh file path.
        :param double scaling: Scale the mesh by this factor (default=1.0).
        :param str mode: The file format to load ['ASCII', 'BINARY', 'AUTOMATIC'].
        :param **kwargs: Accepts optional keyword arguments from the Mesh class.
        :rtype: Mesh
        """

        if mode == "ASCII":
            vertices, triangles = cls._load_ascii(filename, scaling)

        elif mode == "BINARY":
            vertices, triangles = cls._load_binary(filename, scaling)

        else:
            try:
                vertices, triangles = cls._load_ascii(filename, scaling)
            except ValueError:
                vertices, triangles = cls._load_binary(filename, scaling)

        return Mesh(vertices, triangles, smoothing=False, **kwargs)

    @classmethod
    def _load_ascii(cls, filename, scaling):

        # parse the file header
        try:

            filehandle = open(filename, 'r')

            assert filehandle.readline().strip() == "ply"
            assert filehandle.readline().strip() == "format ascii 1.0"

            # skip over comments
            while True:
                line = filehandle.readline().strip()
                if line[0] == "comment":
                    continue
                break

            # read out vertex specification
            assert (line[0] == 'element' and line[1] == 'vertex')
            num_vertices = int(line[2])
            assert filehandle.readline().strip() == 'property float x'
            assert filehandle.readline().strip() == 'property float y'
            assert filehandle.readline().strip() == 'property float z'

            line = filehandle.readline().strip().split()
            assert (line[0] == 'element' and line[1] == 'face')
            num_triangles = int(line[2])
            assert filehandle.readline().strip() == 'property list uchar int vertex_indices'

            # go to end of header
            while True:
                line = filehandle.readline().strip().split()
                if line[0] == "end_header":
                    break

            # read in the vertices
            vertices = np.empty((num_vertices, 3))
            for i in range(num_vertices):
                line = filehandle.readline().strip().split()
                vertices[i, :] = float(line[0])*scaling, float(line[1])*scaling, float(line[2])*scaling

            # read in the triangles
            triangles = np.empty((num_triangles, 3), dtype=np.int32)
            for i in range(num_triangles):
                line = filehandle.readline().strip().split()
                if not int(line[0]) == 3:
                    raise ValueError("This PLY file contains polygons, Raysect can only handle triangles.")
                triangles[i, :] = float(line[1])*scaling, float(line[2])*scaling, float(line[3])*scaling

            filehandle.close()

            return vertices, triangles

        except AssertionError:
            raise ValueError("This file is not a valid PLY file.")

    @classmethod
    def _load_binary(cls, filename, scaling):

        filehandle = open(filename, 'rb')

        assert filehandle.readline().decode().strip() == "ply"
        assert filehandle.readline().decode().strip() == "format binary_little_endian 1.0"

        # skip over comments
        while True:
            line = filehandle.readline().decode().strip().split()
            if line[0] == "comment":
                continue
            break

        # read out vertex specification
        assert (line[0] == 'element' and line[1] == 'vertex')
        num_vertices = int(line[2])
        assert filehandle.readline().decode().strip() == 'property float x'
        assert filehandle.readline().decode().strip() == 'property float y'
        assert filehandle.readline().decode().strip() == 'property float z'

        line = filehandle.readline().decode().strip().split()
        assert (line[0] == 'element' and line[1] == 'face')
        num_triangles = int(line[2])
        assert filehandle.readline().decode().strip() == 'property list uchar int vertex_indices'

        # go to end of header
        while True:
            line = filehandle.readline().decode().strip().split()
            if line[0] == "end_header":
                break

        # read in the vertices
        vertices = np.empty((num_vertices, 3))
        for i in range(num_vertices):
            next_line = filehandle.read(12)
            vertices[i, :] = struct.unpack('<fff', next_line)
        vertices *= scaling

        # read in the triangles
        triangles = np.empty((num_triangles, 3), dtype=np.int32)
        for i in range(num_triangles):
            next_line = filehandle.read(13)
            data = struct.unpack('<BIII', next_line)
            if not data[0] == 3:
                raise ValueError("This PLY file contains polygons, Raysect meshes can only handle triangles.")
            triangles[i, :] = data[1:]

        filehandle.close()

        return vertices, triangles

    @classmethod
    def write_ply(cls, mesh, filename, mode="BINARY", comment="Generated by Raysect"):
        """
        Write a mesh instance to a Polygon File Format (PLY) mesh file (.ply).
        Note PLY is also known as the Stanford Triangle Format.

        :param Mesh mesh: The Raysect mesh instance to write as PLY.
        :param str filename: Mesh file path.
        :param str mode: The file format to write ['ASCII', 'BINARY'].
        :param str comment: An optional string comment to include in the output file,
          can be multiple lines.
        """

        if not isinstance(mesh, Mesh):
            raise ValueError("The mesh argument to write_ply() must be a valid Raysect Mesh primitive object.")

        if mode == "ASCII":
            cls._write_ascii(mesh, filename, comment=comment)

        elif mode == "BINARY":
            cls._write_binary(mesh, filename, comment=comment)

        else:
            raise ValueError("The mode argument for write_ply() must be on of ['BINARY', 'ASCII'].")

    @classmethod
    def _write_ascii(cls, mesh, filename, comment=None):

        filehandle = open(filename, "w")

        vertices = mesh.data.vertices
        num_vertices = vertices.shape[0]
        triangles = mesh.data.triangles
        num_triangles = triangles.shape[0]

        # write header
        filehandle.write("ply\n")
        filehandle.write("format ascii 1.0\n")
        comment_lines = comment.splitlines()
        for comment in comment_lines:
            filehandle.write("comment " + comment + "\n")
        filehandle.write("element vertex {}\n".format(num_vertices))
        filehandle.write("property float x\n")
        filehandle.write("property float y\n")
        filehandle.write("property float z\n")
        filehandle.write("element face {}\n".format(num_triangles))
        filehandle.write("property list uchar int vertex_index\n")
        filehandle.write("end_header\n")

        # write vertices
        for i in range(num_vertices):
            filehandle.write("{} {} {}\n".format(vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        # write triangles
        for i in range(num_triangles):
            filehandle.write("3 {} {} {}\n".format(triangles[i, 0], triangles[i, 1], triangles[i, 2]))

        filehandle.close()

    @classmethod
    def _write_binary(cls, mesh, filename, comment=None):

        filehandle = open(filename, "wb")

        vertices = mesh.data.vertices
        num_vertices = vertices.shape[0]
        triangles = mesh.data.triangles
        num_triangles = triangles.shape[0]

        # write header
        filehandle.write("ply\n".encode())
        filehandle.write("format binary_little_endian 1.0\n".encode())
        comment_lines = comment.splitlines()
        for comment in comment_lines:
            filehandle.write(("comment " + comment + "\n").encode())
        filehandle.write("element vertex {}\n".format(num_vertices).encode())
        filehandle.write("property float x\n".encode())
        filehandle.write("property float y\n".encode())
        filehandle.write("property float z\n".encode())
        filehandle.write("element face {}\n".format(num_triangles).encode())
        filehandle.write("property list uchar int vertex_index\n".encode())
        filehandle.write("end_header\n".encode())

        # write vertices
        for i in range(num_vertices):
            filehandle.write(struct.pack('<fff', vertices[i, 0], vertices[i, 1], vertices[i, 2]))

        # write triangles
        for i in range(num_triangles):
            filehandle.write(struct.pack('<BIII', 3, triangles[i, 0], triangles[i, 1], triangles[i, 2]))

        filehandle.close()


import_ply = PLYHandler.import_ply
write_ply = PLYHandler.write_ply
