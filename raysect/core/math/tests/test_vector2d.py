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
Unit tests for the Vector2D object.
"""

import unittest
from raysect.core.math import Vector2D
from math import sqrt

# TODO: Port to Cython to allow testing of the Cython API

class TestVector2D(unittest.TestCase):

    def test_initialise_default(self):
        """Default initialisation, unit vector pointing along x-axis."""

        v = Vector2D()
        self.assertEqual(v.x, 1.0, "Default initialisation is not (1,0) [X].")
        self.assertEqual(v.y, 0.0, "Default initialisation is not (0,0) [Y].")

    def test_initialise_indexable(self):
        """Initialisation with an indexable object."""

        v = Vector2D(1.0, 2.0)
        self.assertEqual(v.x, 1.0, "Initialisation with indexable failed [X].")
        self.assertEqual(v.y, 2.0, "Initialisation with indexable failed [Y].")

    def test_initialise_invalid(self):
        """Initialisation with an invalid type should raise a TypeError."""

        with self.assertRaises(TypeError, msg="Initialised with a string."):
            Vector2D("spoon")

    def test_x(self):
        """Get/set x co-ordinate."""

        v = Vector2D(2.5, 6.7)

        # get x attribute
        self.assertEqual(v.x, 2.5, "Getting x attribute failed.")

        # set x attribute
        v.x = 10.0
        self.assertEqual(v.x, 10.0, "Setting x attribute failed.")

    def test_y(self):
        """Get/set y co-ordinate."""

        v = Vector2D(2.5, 6.7)

        # get y attribute
        self.assertEqual(v.y, 6.7, "Getting y attribute failed.")

        # set y attribute
        v.y = -7.1
        self.assertEqual(v.y, -7.1, "Setting y attribute failed.")

    def test_indexing(self):
        """Getting/setting components by indexing."""

        v = Vector2D(2.5, 6.7)

        v[0] = 1.0
        v[1] = 2.0

        # check getting/setting via valid indexes
        self.assertEqual(v[0], 1.0, "Indexing failed [X].")
        self.assertEqual(v[1], 2.0, "Indexing failed [Y].")

        # check invalid indexes
        with self.assertRaises(IndexError, msg="Invalid positive index did not raise IndexError."):

            r = v[3]

        with self.assertRaises(IndexError, msg="Invalid negative index did not raise IndexError."):

            r = v[-1]

    def test_iter(self):
        """Obtain values by iteration."""

        v = Vector2D(2.5, 6.7)
        l = list(v)
        self.assertEqual(len(l), 2, "Iteration failed to return the correct number of items.")
        self.assertEqual(l[0], 2.5, "Iteration failed [X].")
        self.assertEqual(l[1], 6.7, "Iteration failed [Y].")

    def test_length(self):
        """Get/set the vector length."""

        v = Vector2D(1.2, -3)

        # get length
        r = sqrt(1.2 * 1.2 + 3 * 3)
        self.assertAlmostEqual(v.length, r, places = 14, msg="Vector2D returned incorrect length.")

        # set length
        v.length = 10.0
        rx = 1.2 / sqrt(1.2 * 1.2 + 3 * 3) * 10
        ry = -3 / sqrt(1.2 * 1.2 + 3 * 3) * 10
        self.assertAlmostEqual(v.x, rx, 14, "Vector2D length was not set correctly [X].")
        self.assertAlmostEqual(v.y, ry, 14, "Vector2D length was not set correctly [Y].")

        # trying to rescale a zero length vector should raise a ZeroDivisionError
        v = Vector2D(0, 0)
        with self.assertRaises(ZeroDivisionError, msg="Adjusting length of zero length vector did not raise a ZeroDivisionError."):

            v.length = 10.0

    def test_equal(self):
        """Equality operator."""

        self.assertTrue(Vector2D(1, 2) == Vector2D(1, 2), "Equality operator returned false for equal vectors.")
        self.assertFalse(Vector2D(5, 2) == Vector2D(1, 2), "Equality operator returned true for a vector with non-equal x components.")
        self.assertFalse(Vector2D(1, 5) == Vector2D(1, 2), "Equality operator returned true for a vector with non-equal y components.")

    def test_not_equal(self):
        """Inequality operator."""

        self.assertFalse(Vector2D(1, 2) != Vector2D(1, 2), "Inequality operator returned true for equal vectors.")
        self.assertTrue(Vector2D(5, 2) != Vector2D(1, 2), "Inequality operator returned false for a vector with non-equal x components.")
        self.assertTrue(Vector2D(1, 5) != Vector2D(1, 2), "Inequality operator returned false for a vector with non-equal y components.")

    def test_negate(self):
        """Negate operator."""

        r = -Vector2D(2.5, 6.7)
        self.assertEqual(r.x, -2.5, "Negation failed [X].")
        self.assertEqual(r.y, -6.7, "Negation failed [Y].")

    def test_add(self):
        """Add operator."""

        # Vector2D + Vector2D, returns Vector2D
        a = Vector2D(-1.4, 0.2)
        b = Vector2D(0.7, -64.0)
        r = a + b
        self.assertTrue(isinstance(r, Vector2D), "Vector2D + Vector2D did not return a Vector2D.")
        self.assertEqual(r.x, -1.4 + 0.7, "Vector2D + Vector2D failed [X].")
        self.assertEqual(r.y, 0.2 - 64.0, "Vector2D + Vector2D failed [Y].")

    def test_subtract(self):
        """Subtract operator."""

        # Vector2D - Vector2D, returns Vector2D
        a = Vector2D(-1.4, 0.2)
        b = Vector2D(0.7, -64.0)
        r = a - b
        self.assertTrue(isinstance(r, Vector2D), "Vector2D - Vector2D did not return a Vector2D.")
        self.assertEqual(r.x, -1.4 - 0.7, "Vector2D - Vector2D failed [X].")
        self.assertEqual(r.y, 0.2 + 64.0, "Vector2D - Vector2D failed [Y].")

    def test_multiply(self):
        """Multiply operator."""

        v = Vector2D(-1.4, 0.2)

        # c * Vector2D, returns Vector2D
        r = 0.23 * v
        self.assertTrue(isinstance(r, Vector2D), "c * Vector2D did not return a Vector2D.")
        self.assertEqual(r.x, 0.23 * -1.4, "c * Vector2D failed [X].")
        self.assertEqual(r.y, 0.23 * 0.20, "c * Vector2D failed [Y].")

        # Vector2D * c, returns Vector2D
        r = v * -2.6
        self.assertTrue(isinstance(r, Vector2D), "Vector2D * c did not return a Vector2D.")
        self.assertEqual(r.x, -2.6 * -1.4, "Vector2D * c failed [X].")
        self.assertEqual(r.y, -2.6 * 0.20, "Vector2D * c failed [Y].")

    def test_divide(self):
        """Division operator."""

        v = Vector2D(-1.4, 0.2)

        # Vector2D / c, returns Vector2D
        r = v / 5.3
        self.assertTrue(isinstance(r, Vector2D), "Vector2D * c did not return a Vector2D.")
        self.assertEqual(r.x, -1.4 / 5.3, "Vector2D * c failed [X].")
        self.assertEqual(r.y, 0.20 / 5.3, "Vector2D * c failed [Y].")

        # dividing by zero should raise a ZeroDivisionError
        with self.assertRaises(ZeroDivisionError, msg="Dividing by zero did not raise a ZeroDivisionError."):

            r = v / 0.0

        # any other division operations should raise TypeError
        with self.assertRaises(TypeError, msg="Undefined division did not raised a TypeError."):

            r = 54.2 / v

        with self.assertRaises(TypeError, msg="Undefined division did not raised a TypeError."):

            r = v / v

    def test_normalise(self):
        """Testing normalise() method."""

        # normalise
        v = Vector2D(23.2, 0.12)
        r = v.normalise()
        l = v.length
        self.assertTrue(isinstance(r, Vector2D), "Normalise did not return a Vector2D.")
        self.assertAlmostEqual(r.x, 23.2 / l, 14, "Normalise failed [X].")
        self.assertAlmostEqual(r.y, 0.12 / l, 14, "Normalise failed [Y].")

        # attempting to normalise a zero length vector should raise a ZeroDivisionError
        v = Vector2D(0.0, 0.0)
        with self.assertRaises(ZeroDivisionError, msg="Normalising a zero length vector did not raise a ZeroDivisionError."):

            r = v.normalise()

    def test_dot_product(self):
        """Testing dot product."""

        x = Vector2D(1, 0)
        y = Vector2D(0, 1)


        # orthogonal
        self.assertEqual(x.dot(y), 0.0, "Dot product of orthogonal vectors does not equal 0.0.")

        # orthonormal
        self.assertEqual(x.dot(x), 1.0, "Dot product of orthonormal vectors does not equal 1.0.")
        self.assertEqual(y.dot(y), 1.0, "Dot product of orthonormal vectors does not equal 1.0.")

        # arbitrary
        a = Vector2D(4, 2)
        b = Vector2D(-1, 2)
        self.assertEqual(a.dot(b), 4*-1 + 2*2, "Dot product of two arbitrary vectors gives the wrong value.")
        self.assertEqual(a.dot(b), b.dot(a), "a.b does not equal b.a.")

    def test_cross_product(self):
        """Testing cross product."""

        # raysect uses a right handed coordinate system
        x = Vector2D(1, 0)
        y = Vector2D(0, 1)

        # orthogonal
        r = x.cross(y)
        self.assertEqual(r, 1.0, "Cross product failed.")

        # orthonormal
        r = x.cross(x)
        self.assertEqual(r, 0.0, "Cross product failed.")

        # arbitrary Vector2D x Vector2D
        a = Vector2D(4, 2)
        b = Vector2D(-1, 2)

        r1 = a.cross(b)
        r2 = b.cross(a)

        self.assertEqual(r1, 4*2 - 2*-1, "Cross product failed.")

        self.assertEqual(r2, 2*-1 - 4*2, "Cross product failed.")

    def test_copy(self):
        """Testing method copy()."""

        v = Vector2D(1.0, 2.0)
        r = v.copy()

        # check a new instance has been created by modifying the original
        v.x = 5.0
        v.y = 6.0

        self.assertEqual(r.x, 1.0, "Copy failed [X].")
        self.assertEqual(r.y, 2.0, "Copy failed [Y].")

if __name__ == "__main__":
    unittest.main()
