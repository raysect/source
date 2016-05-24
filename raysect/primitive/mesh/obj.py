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

from raysect.core import Normal3D
from raysect.primitive.mesh import Mesh


class OBJHandler:

    @classmethod
    def import_obj(cls, filename, scaling=1.0, **kwargs):

        vertices = []
        normals = []
        triangles = []

        with open(filename) as f:
            for line in f:

                # skip comments
                if line[0] == "#":
                    continue

                # clean up and tokenise
                tokens = line.strip().split(" ")

                # parse tokens
                cmd = tokens[0]
                tokens = tokens[1:]

                if cmd == "v":
                    vertex = cls._to_point(tokens, scaling)
                    vertices.append(vertex)

                elif cmd == "vt":
                    # texture coordinates not currently supported
                    continue

                elif cmd == "vn":
                    normal = cls._to_normal(tokens)
                    normals.append(normal)

                elif cmd == "f":
                    triangle = cls._to_triangle(tokens)
                    triangles.append(triangle)

        if normals:
            return Mesh(vertices, triangles, normals, **kwargs)
        return Mesh(vertices, triangles, **kwargs)

    @classmethod
    def _to_point(cls, tokens, scaling):

        x, y, z = tokens
        return [scaling * float(x), scaling * float(y), scaling * float(z)]

    @classmethod
    def _to_normal(cls, tokens):

        x, y, z = tokens
        n = Normal3D(float(x), float(y), float(z)).normalise()
        return [n.x, n.y, n.z]

    @classmethod
    def _to_triangle(cls, tokens):

        if len(tokens) != 3:
            raise ValueError("The .obj importer only support meshes containing 3 sided faces (triangles).")

        v1, n1 = cls._parse_face_token(tokens[0])
        v2, n2 = cls._parse_face_token(tokens[1])
        v3, n3 = cls._parse_face_token(tokens[2])

        # do we have a full set of normals?
        if n1 is None or n2 is None or n3 is None:
            return [v1, v2, v3]
        return [v1, v2, v3, n1, n2, n3]

    @classmethod
    def _parse_face_token(cls, token):

        # texture coordinates not currently supported
        # note indexing in obj format is 1 based, Python is 0 based
        indices = token.split("/")
        v, vn = None, None
        if len(indices) == 1 or len(indices) == 2:
            v = int(indices[0]) - 1
        elif len(indices) == 3:
            v = int(indices[0]) - 1
            vn = int(indices[2]) - 1
        else:
            raise ValueError("The .obj contains an invalid face definition.")

        return v, vn


import_obj = OBJHandler.import_obj
