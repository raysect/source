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

"""
Interaction tests.

Checks that the mathematical classes interact correctly.
"""

import unittest
from raysect.core.math import Vector3D, Normal3D, Point3D, AffineMatrix3D
from math import sqrt

# TODO: Port to Cython to allow testing of the Cython API

class TestInteraction3D(unittest.TestCase):

    def test_vector_initialise(self):
        """Initialisation."""

        v = Vector3D(3, -4, 5)
        n = Normal3D(6, 3, -9)
        p = Point3D(-5, -2, 10)

        # check Vector3D can be initialise by other vector objects
        r = Vector3D(*n)
        self.assertEqual(r.x, 6, "Vector3D initialisation failed [X].")
        self.assertEqual(r.y, 3, "Vector3D initialisation failed [Y].")
        self.assertEqual(r.z, -9, "Vector3D initialisation failed [Z].")

        r = Vector3D(*p)
        self.assertEqual(r.x, -5, "Vector3D initialisation failed [X].")
        self.assertEqual(r.y, -2, "Vector3D initialisation failed [Y].")
        self.assertEqual(r.z, 10, "Vector3D initialisation failed [Z].")

        # check Normal3D can be initialise by other vector objects
        r = Normal3D(*v)
        self.assertEqual(r.x, 3, "Normal3D initialisation failed [X].")
        self.assertEqual(r.y, -4, "Normal3D initialisation failed [Y].")
        self.assertEqual(r.z, 5, "Normal3D initialisation failed [Z].")

        r = Normal3D(*p)
        self.assertEqual(r.x, -5, "Normal3D initialisation failed [X].")
        self.assertEqual(r.y, -2, "Normal3D initialisation failed [Y].")
        self.assertEqual(r.z, 10, "Normal3D initialisation failed [Z].")

        # check Point3D can be initialise by other vector objects
        r = Point3D(*v)
        self.assertEqual(r.x, 3, "Point3D initialisation failed [X].")
        self.assertEqual(r.y, -4, "Point3D initialisation failed [Y].")
        self.assertEqual(r.z, 5, "Point3D initialisation failed [Z].")

        r = Point3D(*n)
        self.assertEqual(r.x, 6, "Point3D initialisation failed [X].")
        self.assertEqual(r.y, 3, "Point3D initialisation failed [Y].")
        self.assertEqual(r.z, -9, "Point3D initialisation failed [Z].")

    def test_vector_add(self):
        """Add operator."""

        v = Vector3D(3, -4, 5)
        n = Normal3D(6, 3, -9)
        p = Point3D(-5, -2, 10)

        # Vector3D and Normal3D
        r = v + n
        self.assertTrue(isinstance(r, Vector3D), "Vector3D addition did not return a Vector3D.")
        self.assertEqual(r.x, 3 + 6, "Vector3D + Normal3D failed [X].")
        self.assertEqual(r.y, -4 + 3, "Vector3D + Normal3D failed [X].")
        self.assertEqual(r.z, 5 - 9, "Vector3D + Normal3D failed [X].")

        r = n + v
        self.assertTrue(isinstance(r, Normal3D), "Vector3D addition did not return a Normal3D.")
        self.assertEqual(r.x, 3 + 6, "Normal3D + Vector3D failed [X].")
        self.assertEqual(r.y, -4 + 3, "Normal3D + Vector3D failed [X].")
        self.assertEqual(r.z, 5 - 9, "Normal3D + Vector3D failed [X].")

        # Point3D and Vector3D
        r = p + v
        self.assertTrue(isinstance(r, Point3D), "Vector3D addition did not return a Point3D.")
        self.assertEqual(r.x, -5 + 3 , "Point3D + Vector3D failed [X].")
        self.assertEqual(r.y, -2 - 4 , "Point3D + Vector3D failed [X].")
        self.assertEqual(r.z, 10 + 5, "Point3D + Vector3D failed [X].")

        with self.assertRaises(TypeError, msg = "Vector3D + Point3D should have raise a TypeError."):

            r = v + p

        # Point3D and Normal3D
        r = p + n
        self.assertTrue(isinstance(r, Point3D), "Vector3D addition did not return a Point3D.")
        self.assertEqual(r.x, -5 + 6 , "Point3D + Normal3D failed [X].")
        self.assertEqual(r.y, -2 + 3 , "Point3D + Normal3D failed [X].")
        self.assertEqual(r.z, 10 - 9, "Point3D + Normal3D failed [X].")

        with self.assertRaises(TypeError, msg = "Normal3D + Point3D should have raise a TypeError."):

            r = n + p

    def test_vector_subtract(self):
        """Subtract operator."""

        v = Vector3D(3, -4, 5)
        n = Normal3D(6, 3, -9)
        p = Point3D(-5, -2, 10)

        # Vector3D and Normal3D
        r = v - n
        self.assertTrue(isinstance(r, Vector3D), "Vector3D addition did not return a Vector3D.")
        self.assertEqual(r.x, 3 - 6, "Vector3D - Normal3D failed [X].")
        self.assertEqual(r.y, -4 - 3, "Vector3D - Normal3D failed [X].")
        self.assertEqual(r.z, 5 + 9, "Vector3D - Normal3D failed [X].")

        r = n - v
        self.assertTrue(isinstance(r, Normal3D), "Vector3D addition did not return a Normal3D.")
        self.assertEqual(r.x, -3 + 6, "Normal3D - Vector3D failed [X].")
        self.assertEqual(r.y, 4 + 3, "Normal3D - Vector3D failed [X].")
        self.assertEqual(r.z, -5 - 9, "Normal3D - Vector3D failed [X].")

        # Point3D and Vector3D
        r = p - v
        self.assertTrue(isinstance(r, Point3D), "Vector3D addition did not return a Point3D.")
        self.assertEqual(r.x, -5 - 3 , "Point3D - Vector3D failed [X].")
        self.assertEqual(r.y, -2 + 4 , "Point3D - Vector3D failed [X].")
        self.assertEqual(r.z, 10 - 5, "Point3D - Vector3D failed [X].")

        with self.assertRaises(TypeError, msg = "Vector3D - Point3D should have raise a TypeError."):

            r = v - p

        # Point3D and Normal3D
        r = p - n
        self.assertTrue(isinstance(r, Point3D), "Vector3D addition did not return a Point3D.")
        self.assertEqual(r.x, -5 - 6 , "Point3D - Normal3D failed [X].")
        self.assertEqual(r.y, -2 - 3 , "Point3D - Normal3D failed [X].")
        self.assertEqual(r.z, 10 + 9, "Point3D - Normal3D failed [X].")

        with self.assertRaises(TypeError, msg = "Normal3D - Point3D should have raise a TypeError."):

            r = n - p

    def test_vector_cross(self):
        """Cross product."""

        a = Vector3D(3, -4, 5)
        b = Normal3D(6, 3, -9)

        # Vector3D x Normal3D
        r = a.cross(b)
        self.assertTrue(isinstance(r, Vector3D), "Cross did not return a Vector3D.")
        self.assertEqual(r.x, a.y * b.z - b.y * a.z, "Cross product failed [X].")
        self.assertEqual(r.y, b.x * a.z - a.x * b.z, "Cross product failed [Y].")
        self.assertEqual(r.z, a.x * b.y - b.x * a.y, "Cross product failed [Z].")

        # Normal3D x Vector3D
        r = b.cross(a)
        self.assertTrue(isinstance(r, Vector3D), "Cross did not return a Vector3D.")
        self.assertEqual(r.x, b.y * a.z - a.y * b.z, "Cross product failed [X].")
        self.assertEqual(r.y, a.x * b.z - b.x * a.z, "Cross product failed [Y].")
        self.assertEqual(r.z, b.x * a.y - a.x * b.y, "Cross product failed [Z].")

    def test_affine_vector_multiply(self):
        """Matrix-Vector3D multiply."""

        m = AffineMatrix3D([[1, 2, 3, 4],
                            [5,6,2,8],
                            [9,10,4,9],
                            [4,14,15,16]])

        v = Vector3D(-1, 2, 6)
        n = Normal3D(-1, 2, 6)
        p = Point3D(-1, 2, 6)

        # AffineMatrix3D * Vector3D
        r = m * v
        self.assertTrue(isinstance(r, Vector3D), "AffineMatrix3D * Vector3D did not return a Vector3D.")
        self.assertEqual(r.x, 1 * -1 +  2 * 2 + 3 * 6, "AffineMatrix3D * Vector3D failed [X].")
        self.assertEqual(r.y, 5 * -1 +  6 * 2 + 2 * 6, "AffineMatrix3D * Vector3D failed [Y].")
        self.assertEqual(r.z, 9 * -1 + 10 * 2 + 4 * 6, "AffineMatrix3D * Vector3D failed [Z].")

        # Vector3D * AffineMatrix3D
        with self.assertRaises(TypeError, msg = "Vector3D * AffineMatrix3D should have raise a TypeError."):

            r = v * m

        # AffineMatrix3D * Normal3D
        r = m * n
        self.assertTrue(isinstance(r, Normal3D), "AffineMatrix3D * Normal3D did not return a Normal3D.")
        self.assertAlmostEqual(r.x,  258/414 * -1 +  -381/414 * 2 +  210/414 * 6, places = 14, msg = "AffineMatrix3D * Normal3D failed [X].")
        self.assertAlmostEqual(r.y, -132/414 * -1 +    81/414 * 2 + -162/414 * 6, places = 14, msg = "AffineMatrix3D * Normal3D failed [Y].")
        self.assertAlmostEqual(r.z,  120/414 * -1 +   -36/414 * 2 +   72/414 * 6, places = 14, msg = "AffineMatrix3D * Normal3D failed [Z].")

        # Normal3D * AffineMatrix3D
        with self.assertRaises(TypeError, msg = "Normal3D * AffineMatrix3D should have raise a TypeError."):

            r = n * m

        # AffineMatrix3D * Point3D
        r = m * p
        self.assertTrue(isinstance(r, Point3D), "AffineMatrix3D * Point3D did not return a Point3D.")
        w = (4 * -1 + 14 * 2 + 15 * 6 + 16)
        self.assertEqual(r.x, (1 * -1 +  2 * 2 + 3 * 6 + 4) / w, "AffineMatrix3D * Point3D failed [X].")
        self.assertEqual(r.y, (5 * -1 +  6 * 2 + 2 * 6 + 8) / w, "AffineMatrix3D * Point3D failed [Y].")
        self.assertEqual(r.z, (9 * -1 + 10 * 2 + 4 * 6 + 9) / w, "AffineMatrix3D * Point3D failed [Z].")

        # Point3D * AffineMatrix3D
        with self.assertRaises(TypeError, msg = "Point3D * AffineMatrix3D should have raise a TypeError."):

            p = p * m


if __name__ == "__main__":
    unittest.main()
