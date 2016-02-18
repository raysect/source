# Copyright (c) 2014-2016, Dr Alex Meakins, Raysect Project
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

import unittest
from raysect.core.math import translate, rotate_x, rotate_y, rotate_z, rotate_vector, rotate, rotate_basis, Vector3D
from math import sin, cos, pi, sqrt

# TODO: Port to Cython to allow testing of the Cython API


class TestTransform(unittest.TestCase):

    def test_translate(self):
        """Translation matrix factory function."""

        m = translate(1.3, 4.5, 2.2)

        r = [[1, 0, 0, 1.3],
             [0, 1, 0, 4.5],
             [0, 0, 1, 2.2],
             [0, 0, 0, 1]]

        for i, row in enumerate(r):
            for j, v in enumerate(row):
                self.assertAlmostEqual(m[i, j], v, places=14, msg="Transform matrix generation failed (R"+str(i)+", C"+str(j)+").")

    def test_rotate_x(self):
        """Rotation about x-axis matrix factory function."""

        m = rotate_x(67)

        a = pi * 67 / 180

        r = [[1, 0, 0, 0],
             [0, cos(a), -sin(a), 0],
             [0, sin(a), cos(a), 0],
             [0, 0, 0, 1]]

        for i, row in enumerate(r):
            for j, v in enumerate(row):
                self.assertAlmostEqual(m[i, j], v, places=14, msg="Rotate_x matrix generation failed (R"+str(i)+", C"+str(j)+").")

    def test_rotate_y(self):
        """Rotation about y-axis matrix factory function."""

        m = rotate_y(-7.3)

        a = pi * -7.3 / 180

        r = [[cos(a), 0, sin(a), 0],
             [0, 1, 0, 0],
             [-sin(a), 0, cos(a), 0],
             [0, 0, 0, 1]]

        for i, row in enumerate(r):
            for j, v in enumerate(row):
                self.assertAlmostEqual(m[i, j], v, places=14, msg="Rotate_y matrix generation failed (R"+str(i)+", C"+str(j)+").")

    def test_rotate_z(self):
        """Rotation about z-axis matrix factory function."""

        m = rotate_z(23)

        a = pi * 23 / 180

        r = [[cos(a), -sin(a), 0, 0],
             [sin(a), cos(a), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]]

        for i, row in enumerate(r):
            for j, v in enumerate(row):
                self.assertAlmostEqual(m[i, j], v, places=14, msg="Rotate_z matrix generation failed (R"+str(i)+", C"+str(j)+").")

    def test_rotate_vector(self):
        """Rotation about vector matrix factory function."""

        m = rotate_vector(54, Vector3D(1.0, 0.22, 0.34))

        s = sin(pi*54/180)
        c = cos(pi*54/180)

        x = 1.0
        y = 0.22
        z = 0.34

        # normalise
        l = sqrt(x * x + y * y + z * z)
        x = x / l
        y = y / l
        z = z / l

        r = [[x*x+(1-x*x)*c, x*y*(1-c)-z*s, x*z*(1-c)+y*s, 0],
             [x*y*(1-c)+z*s, y*y+(1-y*y)*c, y*z*(1-c)-x*s, 0],
             [x*z*(1-c)-y*s, y*z*(1-c)+x*s, z*z+(1-z*z)*c, 0],
             [0, 0, 0, 1]]

        for i, row in enumerate(r):
            for j, v in enumerate(row):
                self.assertAlmostEqual(m[i, j], v, places=14, msg="Rotate_vector matrix generation failed (R"+str(i)+", C"+str(j)+").")

    def test_rotate(self):
        """Rotation by yaw, pitch and roll factory function."""

        m = rotate(63, -40, 12)
        r = rotate_y(-63) * rotate_x(40) * rotate_z(12)

        for i in range(0, 4):
            for j in range(0, 4):
                self.assertAlmostEqual(m[i, j], r[i, j], places=14, msg="Rotate matrix generation failed (R"+str(i)+", C"+str(j)+").")

    def test_rotate_basis(self):
        """Rotation specified by a pair of basis vectors."""

        # valid vectors
        m = rotate_basis(Vector3D(1.0, 0.0, 0.0), Vector3D(0.0, -1.0, 0.0))
        r = [[0, 0, 1, 0],
             [0, -1, 0, 0],
             [1, 0, 0, 0],
             [0, 0, 0, 1]]

        for i, row in enumerate(r):
            for j, v in enumerate(row):
                self.assertAlmostEqual(m[i, j], v, places=14, msg="Rotate_basis matrix generation failed (R"+str(i)+", C"+str(j)+").")

        # invalid, coincident vectors
        with self.assertRaises(ValueError, msg="Coincident forward and up vectors did not raise a ValueError."):
            rotate_basis(Vector3D(1, 2, 3), Vector3D(1, 2, 3))
