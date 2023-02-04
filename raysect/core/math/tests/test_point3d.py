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
Unit tests for the Point3D object.
"""

import unittest
from raysect.core.math import Point3D, Vector3D, AffineMatrix3D
from math import sqrt

# TODO: Port to Cython to allow testing of the Cython API

class TestPoint3D(unittest.TestCase):

    def test_initialise_default(self):
        """Default initialisation, point at local origin."""

        v = Point3D()
        self.assertEqual(v.x, 0.0, "Default initialisation is not (0,0,0) [X].")
        self.assertEqual(v.y, 0.0, "Default initialisation is not (0,0,0) [Y].")
        self.assertEqual(v.z, 0.0, "Default initialisation is not (0,0,0) [Z].")

    def test_initialise_indexable(self):
        """Initialisation with an indexable object."""

        v = Point3D(1.0, 2.0, 3.0)
        self.assertEqual(v.x, 1.0, "Initialisation with indexable failed [X].")
        self.assertEqual(v.y, 2.0, "Initialisation with indexable failed [Y].")
        self.assertEqual(v.z, 3.0, "Initialisation with indexable failed [Z].")

    def test_initialise_invalid(self):
        """Initialisation with invalid types should raise a TypeError."""

        with self.assertRaises(TypeError, msg="Initialised with a string."):
            Point3D("spoon")

    def test_x(self):
        """Get/set x co-ordinate."""

        v = Point3D(2.5, 6.7, -4.6)

        # get x attribute
        self.assertEqual(v.x, 2.5, "Getting x attribute failed.")

        # set x attribute
        v.x = 10.0
        self.assertEqual(v.x, 10.0, "Setting x attribute failed.")

    def test_y(self):
        """Get/set y co-ordinate."""

        v = Point3D(2.5, 6.7, -4.6)

        # get y attribute
        self.assertEqual(v.y, 6.7, "Getting y attribute failed.")

        # set y attribute
        v.y = -7.1
        self.assertEqual(v.y, -7.1, "Setting y attribute failed.")

    def test_z(self):
        """Get/set z co-ordinate."""

        v = Point3D(2.5, 6.7, -4.6)

        # get z attribute
        self.assertEqual(v.z, -4.6, "Getting z attribute failed.")

        # set z attribute
        v.z = 157.3
        self.assertEqual(v.z, 157.3, "Setting z attribute failed.")

    def test_indexing(self):
        """Getting/setting components by indexing."""

        v = Point3D(2.5, 6.7, -4.6)

        v[0] = 1.0
        v[1] = 2.0
        v[2] = 7.0

        # check getting/setting via valid indexes
        self.assertEqual(v[0], 1.0, "Indexing failed [X].")
        self.assertEqual(v[1], 2.0, "Indexing failed [Y].")
        self.assertEqual(v[2], 7.0, "Indexing failed [Z].")

        # check invalid indexes
        with self.assertRaises(IndexError, msg="Invalid positive index did not raise IndexError."):

            r = v[4]

        with self.assertRaises(IndexError, msg="Invalid negative index did not raise IndexError."):

            r = v[-1]

    def test_equal(self):
        """Equality operator."""

        self.assertTrue(Point3D(1, 2, 3) == Point3D(1, 2, 3), "Equality operator returned false for equal points.")
        self.assertFalse(Point3D(5, 2, 3) == Point3D(1, 2, 3), "Equality operator returned true for a point with non-equal x components.")
        self.assertFalse(Point3D(1, 5, 3) == Point3D(1, 2, 3), "Equality operator returned true for a point with non-equal y components.")
        self.assertFalse(Point3D(1, 2, 5) == Point3D(1, 2, 3), "Equality operator returned true for a point with non-equal z components.")

    def test_not_equal(self):
        """Inequality operator."""

        self.assertFalse(Point3D(1, 2, 3) != Point3D(1, 2, 3), "Inequality operator returned true for equal points.")
        self.assertTrue(Point3D(5, 2, 3) != Point3D(1, 2, 3), "Inequality operator returned false for a point with non-equal x components.")
        self.assertTrue(Point3D(1, 5, 3) != Point3D(1, 2, 3), "Inequality operator returned false for a point with non-equal y components.")
        self.assertTrue(Point3D(1, 2, 5) != Point3D(1, 2, 3), "Inequality operator returned false for a point with non-equal z components.")

    def test_iter(self):
        """Obtain values by iteration."""

        p = Point3D(2.5, 6.7, -4.6)
        l = list(p)
        self.assertEqual(len(l), 3, "Iteration failed to return the correct number of items.")
        self.assertEqual(l[0], 2.5, "Iteration failed [X].")
        self.assertEqual(l[1], 6.7, "Iteration failed [Y].")
        self.assertEqual(l[2], -4.6, "Iteration failed [Z].")

    def test_add(self):
        """Addition operator."""

        # adding points is undefined
        with self.assertRaises(TypeError, msg="Point3D addition did not raise a TypeError."):

            Point3D() + Point3D()

    def test_subtract(self):
        """Subtraction operator."""

        # subtracting points is undefined
        with self.assertRaises(TypeError, msg="Point3D subtraction did not raise a TypeError."):

            Point3D() - Point3D()

    def test_distance_to(self):
        """Testing method distance_to()."""

        a = Point3D(-1, 5, 26)
        b = Point3D(9, 4, -1)
        v = a.distance_to(b)
        r = sqrt((9 + 1)**2 + (4 - 5)**2 + (-1 - 26)**2)
        self.assertEqual(v, r, "Point3D to Point3D distance is incorrect.")

    def test_vector_to(self):
        """Testing method vector_to()."""

        a = Point3D(-1, 5, 26)
        b = Point3D(9, 4, -1)
        v = a.vector_to(b)
        self.assertTrue(isinstance(v, Vector3D), "Vector_to did not return a Vector3D.")
        self.assertEqual(v.x, 9 + 1, "Vector_to failed [X].")
        self.assertEqual(v.y, 4 - 5, "Vector_to failed [Y].")
        self.assertEqual(v.z, -1 - 26, "Vector_to failed [Z].")

    def test_transform(self):
        """Testing method transform()."""

        m = AffineMatrix3D([[1, 2, 3, 4],
                            [5,6,2,8],
                            [9,10,4,9],
                            [4,14,15,16]])

        v = Point3D(-1, 2, 6)

        r = v.transform(m)

        self.assertTrue(isinstance(r, Point3D), "Transform did not return a Point3D.")

        w = (4 * -1 + 14 * 2 + 15 * 6 + 16)
        self.assertEqual(r.x, (1 * -1 +  2 * 2 + 3 * 6 + 4) / w, "Transform failed [X].")
        self.assertEqual(r.y, (5 * -1 +  6 * 2 + 2 * 6 + 8) / w, "Transform failed [Y].")
        self.assertEqual(r.z, (9 * -1 + 10 * 2 + 4 * 6 + 9) / w, "Transform failed [Z].")

    def test_copy(self):
        """Testing method copy()."""

        v = Point3D(1.0, 2.0, 3.0)
        r = v.copy()

        # check a new instance has been created by modifying the original
        v.x = 5.0
        v.y = 6.0
        v.z = 7.0

        self.assertEqual(r.x, 1.0, "Copy failed [X].")
        self.assertEqual(r.y, 2.0, "Copy failed [Y].")
        self.assertEqual(r.z, 3.0, "Copy failed [Z].")

    def test_orthogonal(self):

        v = Vector3D(1.0, 2.0, 3.0)
        r = v.orthogonal()

        self.assertAlmostEqual(v.dot(r), 0.0, delta=1e-10, msg="Orthogonal did not return an orthogonal vector.")


if __name__ == "__main__":
    unittest.main()