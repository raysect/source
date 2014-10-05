# Copyright (c) 2014, Dr Alex Meakins, Raysect Project
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
from raysect.core.math import Vector, Normal, Point, AffineMatrix
from math import sqrt

# TODO: Port to Cython to allow testing of the Cython API

class TestInteraction(unittest.TestCase):

    def test_vector_initialise(self):
        """Initialisation."""

        v = Vector(3, -4, 5)
        n = Normal(6, 3, -9)
        p = Point(-5, -2, 10)

        # check Vector can be initialise by other vector objects
        r = Vector(*n)
        self.assertEqual(r.x, 6, "Vector initialisation failed [X].")
        self.assertEqual(r.y, 3, "Vector initialisation failed [Y].")
        self.assertEqual(r.z, -9, "Vector initialisation failed [Z].")

        r = Vector(*p)
        self.assertEqual(r.x, -5, "Vector initialisation failed [X].")
        self.assertEqual(r.y, -2, "Vector initialisation failed [Y].")
        self.assertEqual(r.z, 10, "Vector initialisation failed [Z].")

        # check Normal can be initialise by other vector objects
        r = Normal(*v)
        self.assertEqual(r.x, 3, "Normal initialisation failed [X].")
        self.assertEqual(r.y, -4, "Normal initialisation failed [Y].")
        self.assertEqual(r.z, 5, "Normal initialisation failed [Z].")

        r = Normal(*p)
        self.assertEqual(r.x, -5, "Normal initialisation failed [X].")
        self.assertEqual(r.y, -2, "Normal initialisation failed [Y].")
        self.assertEqual(r.z, 10, "Normal initialisation failed [Z].")

        # check Point can be initialise by other vector objects
        r = Point(*v)
        self.assertEqual(r.x, 3, "Point initialisation failed [X].")
        self.assertEqual(r.y, -4, "Point initialisation failed [Y].")
        self.assertEqual(r.z, 5, "Point initialisation failed [Z].")

        r = Point(*n)
        self.assertEqual(r.x, 6, "Point initialisation failed [X].")
        self.assertEqual(r.y, 3, "Point initialisation failed [Y].")
        self.assertEqual(r.z, -9, "Point initialisation failed [Z].")

    def test_vector_add(self):
        """Add operator."""

        v = Vector(3, -4, 5)
        n = Normal(6, 3, -9)
        p = Point(-5, -2, 10)

        # Vector and Normal
        r = v + n
        self.assertTrue(isinstance(r, Vector), "Vector addition did not return a Vector.")
        self.assertEqual(r.x, 3 + 6, "Vector + Normal failed [X].")
        self.assertEqual(r.y, -4 + 3, "Vector + Normal failed [X].")
        self.assertEqual(r.z, 5 - 9, "Vector + Normal failed [X].")

        r = n + v
        self.assertTrue(isinstance(r, Normal), "Vector addition did not return a Normal.")
        self.assertEqual(r.x, 3 + 6, "Normal + Vector failed [X].")
        self.assertEqual(r.y, -4 + 3, "Normal + Vector failed [X].")
        self.assertEqual(r.z, 5 - 9, "Normal + Vector failed [X].")

        # Point and Vector
        r = p + v
        self.assertTrue(isinstance(r, Point), "Vector addition did not return a Point.")
        self.assertEqual(r.x, -5 + 3 , "Point + Vector failed [X].")
        self.assertEqual(r.y, -2 - 4 , "Point + Vector failed [X].")
        self.assertEqual(r.z, 10 + 5, "Point + Vector failed [X].")

        with self.assertRaises(TypeError, msg = "Vector + Point should have raise a TypeError."):

            r = v + p

        # Point and Normal
        r = p + n
        self.assertTrue(isinstance(r, Point), "Vector addition did not return a Point.")
        self.assertEqual(r.x, -5 + 6 , "Point + Normal failed [X].")
        self.assertEqual(r.y, -2 + 3 , "Point + Normal failed [X].")
        self.assertEqual(r.z, 10 - 9, "Point + Normal failed [X].")

        with self.assertRaises(TypeError, msg = "Normal + Point should have raise a TypeError."):

            r = n + p

    def test_vector_subtract(self):
        """Subtract operator."""

        v = Vector(3, -4, 5)
        n = Normal(6, 3, -9)
        p = Point(-5, -2, 10)

        # Vector and Normal
        r = v - n
        self.assertTrue(isinstance(r, Vector), "Vector addition did not return a Vector.")
        self.assertEqual(r.x, 3 - 6, "Vector - Normal failed [X].")
        self.assertEqual(r.y, -4 - 3, "Vector - Normal failed [X].")
        self.assertEqual(r.z, 5 + 9, "Vector - Normal failed [X].")

        r = n - v
        self.assertTrue(isinstance(r, Normal), "Vector addition did not return a Normal.")
        self.assertEqual(r.x, -3 + 6, "Normal - Vector failed [X].")
        self.assertEqual(r.y, 4 + 3, "Normal - Vector failed [X].")
        self.assertEqual(r.z, -5 - 9, "Normal - Vector failed [X].")

        # Point and Vector
        r = p - v
        self.assertTrue(isinstance(r, Point), "Vector addition did not return a Point.")
        self.assertEqual(r.x, -5 - 3 , "Point - Vector failed [X].")
        self.assertEqual(r.y, -2 + 4 , "Point - Vector failed [X].")
        self.assertEqual(r.z, 10 - 5, "Point - Vector failed [X].")

        with self.assertRaises(TypeError, msg = "Vector - Point should have raise a TypeError."):

            r = v - p

        # Point and Normal
        r = p - n
        self.assertTrue(isinstance(r, Point), "Vector addition did not return a Point.")
        self.assertEqual(r.x, -5 - 6 , "Point - Normal failed [X].")
        self.assertEqual(r.y, -2 - 3 , "Point - Normal failed [X].")
        self.assertEqual(r.z, 10 + 9, "Point - Normal failed [X].")

        with self.assertRaises(TypeError, msg = "Normal - Point should have raise a TypeError."):

            r = n - p

    def test_vector_cross(self):
        """Cross product."""

        a = Vector(3, -4, 5)
        b = Normal(6, 3, -9)

        # Vector x Normal
        r = a.cross(b)
        self.assertTrue(isinstance(r, Vector), "Cross did not return a Vector.")
        self.assertEqual(r.x, a.y * b.z - b.y * a.z, "Cross product failed [X].")
        self.assertEqual(r.y, b.x * a.z - a.x * b.z, "Cross product failed [Y].")
        self.assertEqual(r.z, a.x * b.y - b.x * a.y, "Cross product failed [Z].")

        # Normal x Vector
        r = b.cross(a)
        self.assertTrue(isinstance(r, Vector), "Cross did not return a Vector.")
        self.assertEqual(r.x, b.y * a.z - a.y * b.z, "Cross product failed [X].")
        self.assertEqual(r.y, a.x * b.z - b.x * a.z, "Cross product failed [Y].")
        self.assertEqual(r.z, b.x * a.y - a.x * b.y, "Cross product failed [Z].")

    def test_affine_vector_multiply(self):
        """Matrix-Vector multiply."""

        m = AffineMatrix([[1,2,3,4],
                          [5,6,2,8],
                          [9,10,4,9],
                          [4,14,15,16]])

        v = Vector(-1, 2, 6)
        n = Normal(-1, 2, 6)
        p = Point(-1, 2, 6)

        # AffineMatrix * Vector
        r = m * v
        self.assertTrue(isinstance(r, Vector), "AffineMatrix * Vector did not return a Vector.")
        self.assertEqual(r.x, 1 * -1 +  2 * 2 + 3 * 6, "AffineMatrix * Vector failed [X].")
        self.assertEqual(r.y, 5 * -1 +  6 * 2 + 2 * 6, "AffineMatrix * Vector failed [Y].")
        self.assertEqual(r.z, 9 * -1 + 10 * 2 + 4 * 6, "AffineMatrix * Vector failed [Z].")

        # Vector * AffineMatrix
        with self.assertRaises(TypeError, msg = "Vector * AffineMatrix should have raise a TypeError."):

            r = v * m

        # AffineMatrix * Normal
        r = m * n
        self.assertTrue(isinstance(r, Normal), "AffineMatrix * Normal did not return a Normal.")
        self.assertAlmostEqual(r.x,  258/414 * -1 +  -381/414 * 2 +  210/414 * 6, places = 14, msg = "AffineMatrix * Normal failed [X].")
        self.assertAlmostEqual(r.y, -132/414 * -1 +    81/414 * 2 + -162/414 * 6, places = 14, msg = "AffineMatrix * Normal failed [Y].")
        self.assertAlmostEqual(r.z,  120/414 * -1 +   -36/414 * 2 +   72/414 * 6, places = 14, msg = "AffineMatrix * Normal failed [Z].")

        # Normal * AffineMatrix
        with self.assertRaises(TypeError, msg = "Normal * AffineMatrix should have raise a TypeError."):

            r = n * m

        # AffineMatrix * Point
        r = m * p
        self.assertTrue(isinstance(r, Point), "AffineMatrix * Point did not return a Point.")
        w = (4 * -1 + 14 * 2 + 15 * 6 + 16)
        self.assertEqual(r.x, (1 * -1 +  2 * 2 + 3 * 6 + 4) / w, "AffineMatrix * Point failed [X].")
        self.assertEqual(r.y, (5 * -1 +  6 * 2 + 2 * 6 + 8) / w, "AffineMatrix * Point failed [Y].")
        self.assertEqual(r.z, (9 * -1 + 10 * 2 + 4 * 6 + 9) / w, "AffineMatrix * Point failed [Z].")

        # Point * AffineMatrix
        with self.assertRaises(TypeError, msg = "Point * AffineMatrix should have raise a TypeError."):

            p = p * m


if __name__ == "__main__":
    unittest.main()
