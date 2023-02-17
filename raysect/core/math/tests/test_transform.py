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

import unittest
from raysect.core.math import Point3D, Vector3D
from raysect.core.math.transform import *
from math import sin, cos, pi, sqrt
import numpy as np

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

    def test_to_cylindrical(self):
        """Test the to_cylindrical() math utility function."""

        origin = Point3D(0, 0, 0)
        r, z, phi = to_cylindrical(origin)
        self.assertEqual(r, 0.0, "Origin in cartesian (0, 0, 0) did not map to the origin in cylindrical coordinates.")
        self.assertEqual(z, 0.0, "Origin in cartesian (0, 0, 0) did not map to the origin in cylindrical coordinates.")
        self.assertEqual(phi, 0.0, "Origin in cartesian (0, 0, 0) did not map to the origin in cylindrical coordinates.")

        point = Point3D(1, 1, 1)
        r_test = np.sqrt(point.x**2 + point.y**2)
        z_test = point.z
        phi_test = np.rad2deg(np.arctan2(point.y, point.x))
        r, z, phi = to_cylindrical(point)
        self.assertEqual(r, r_test, "R coordinate did not map successfully.")
        self.assertEqual(z, z_test, "Z coordinate did not map successfully.")
        self.assertEqual(phi, phi_test, "Phi coordinate did not map successfully.")

    def test_from_cylindrical(self):
        """Test the from_cartesian() math utility function."""

        r, z, phi = 0, 0, 0
        cartesian_point = from_cylindrical(r, z, phi)
        self.assertEqual(cartesian_point.x, 0.0, "Origin in cylindrical coordinates did not map to the origin in cartesian coordinates.")
        self.assertEqual(cartesian_point.y, 0.0, "Origin in cylindrical coordinates did not map to the origin in cartesian coordinates.")
        self.assertEqual(cartesian_point.z, 0.0, "Origin in cylindrical coordinates did not map to the origin in cartesian coordinates.")

        # check invalid radial coordinate
        with self.assertRaises(ValueError, msg="Invalid radial coordinate was not detected."):
            cartesian_point = from_cylindrical(-1, 1, 1)

        r, z, phi = 1, 1, 45
        x_test = r * np.cos(np.deg2rad(phi))
        y_test = r * np.sin(np.deg2rad(phi))
        z_test = z
        cartesian_point = from_cylindrical(r, z, phi)
        self.assertEqual(cartesian_point.x, x_test, "X coordinate did not map successfully.")
        self.assertEqual(cartesian_point.y, y_test, "Y coordinate did not map successfully.")
        self.assertEqual(cartesian_point.z, z_test, "Z coordinate did not map successfully.")

    def test_extract_rotation(self):

        r = (-60, 23, -9)

        # test z axis forward convention
        mf = rotate(*r)
        vf = extract_rotation(mf)
        self.assertAlmostEqual(vf[0], r[0], delta=1e-10, msg="Failed to extract yaw from affine matrix (z-axis forward).")
        self.assertAlmostEqual(vf[1], r[1], delta=1e-10, msg="Failed to extract pitch from affine matrix (z-axis forward).")
        self.assertAlmostEqual(vf[2], r[2], delta=1e-10, msg="Failed to extract roll from affine matrix (z-axis forward).")

        # test z axis up convention
        mu = rotate_z(-r[0]) * rotate_y(-r[1]) * rotate_x(r[2])
        vu = extract_rotation(mu, z_up=True)
        self.assertAlmostEqual(vu[0], r[0], delta=1e-10, msg="Failed to extract yaw from affine matrix (z-axis up).")
        self.assertAlmostEqual(vu[1], r[1], delta=1e-10, msg="Failed to extract pitch from affine matrix (z-axis up).")
        self.assertAlmostEqual(vu[2], r[2], delta=1e-10, msg="Failed to extract roll from affine matrix (z-axis up).")

    def test_extract_translation(self):

        r = (10, -20, 40)
        m = translate(*r)
        v = extract_translation(m)
        self.assertAlmostEqual(v, r, delta=1e-10, msg="Failed to extract translation from affine matrix.")

