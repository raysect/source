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
Unit tests for the Normal object.
"""

import unittest
from ..normal import Normal
from ..vector import Vector
from ..affinematrix import AffineMatrix
from math import sqrt

# TODO: Port to Cython to allow testing of the Cython API

class TestNormal(unittest.TestCase):

    def test_initialise_default(self):
        """Default initialisation, unit normal pointing along z-axis."""

        v = Normal()
        self.assertEqual(v.x, 0.0, "Default initialisation is not (0,0,1) [X].")
        self.assertEqual(v.y, 0.0, "Default initialisation is not (0,0,1) [Y].")
        self.assertEqual(v.z, 1.0, "Default initialisation is not (0,0,1) [Z].")

    def test_initialise_indexable(self):
        """Initialisation with an indexable object."""

        v = Normal([1.0, 2.0, 3.0])
        self.assertEqual(v.x, 1.0, "Initialisation with indexable failed [X].")
        self.assertEqual(v.y, 2.0, "Initialisation with indexable failed [Y].")
        self.assertEqual(v.z, 3.0, "Initialisation with indexable failed [Z].")

    def test_initialise_invalid(self):
        """Initialisation with an invalid type should raise a TypeError."""

        with self.assertRaises(TypeError, msg="Initialised with a string."):
            Normal("spoon")

        with self.assertRaises(TypeError, msg="Initialised with a list containing too few items."):
            Normal([1.0, 2.0])

    def test_x(self):
        """Get/set x co-ordinate."""

        v = Normal([2.5, 6.7, -4.6])

        # get x attribute
        self.assertEqual(v.x, 2.5, "Getting x attribute failed.")

        # set x attribute
        v.x = 10.0
        self.assertEqual(v.x, 10.0, "Setting x attribute failed.")

    def test_y(self):
        """Get/set y co-ordinate."""

        v = Normal([2.5, 6.7, -4.6])

        # get y attribute
        self.assertEqual(v.y, 6.7, "Getting y attribute failed.")

        # set y attribute
        v.y = -7.1
        self.assertEqual(v.y, -7.1, "Setting y attribute failed.")

    def test_z(self):
        """Get/set z co-ordinate."""

        v = Normal([2.5, 6.7, -4.6])

        # get z attribute
        self.assertEqual(v.z, -4.6, "Getting z attribute failed.")

        # set z attribute
        v.z = 157.3
        self.assertEqual(v.z, 157.3, "Setting z attribute failed.")

    def test_indexing(self):
        """Getting/setting components by indexing."""

        v = Normal([2.5, 6.7, -4.6])

        # check valid indexes
        self.assertEqual(v[0], 2.50, "Indexing failed [X].")
        self.assertEqual(v[1], 6.70, "Indexing failed [Y].")
        self.assertEqual(v[2], -4.6, "Indexing failed [Z].")

        # check invalid indexes
        with self.assertRaises(IndexError, msg="Invalid positive index did not raise IndexError."):

            r = v[4]

        with self.assertRaises(IndexError, msg="Invalid negative index did not raise IndexError."):

            r = v[-1]

    def test_length(self):
        """Get/set the normal length."""

        v = Normal([1.2, -3, 9.8])

        # get length
        r = sqrt(1.2 * 1.2 + 3 * 3 + 9.8 * 9.8)
        self.assertAlmostEqual(v.length, r, places = 14, msg = "Normal returned incorrect length.")

        # set length
        v.length = 10.0
        rx = 1.2 / sqrt(1.2 * 1.2 + 3 * 3 + 9.8 * 9.8) * 10
        ry = -3 / sqrt(1.2 * 1.2 + 3 * 3 + 9.8 * 9.8) * 10
        rz = 9.8 / sqrt(1.2 * 1.2 + 3 * 3 + 9.8 * 9.8) * 10
        self.assertAlmostEqual(v.x, rx, 14, "Normal length was not set correctly [X].")
        self.assertAlmostEqual(v.y, ry, 14, "Normal length was not set correctly [Y].")
        self.assertAlmostEqual(v.z, rz, 14, "Normal length was not set correctly [Z].")

        # trying to rescale a zero length normal should raise a ZeroDivisionError
        v = Normal([0,0,0])
        with self.assertRaises(ZeroDivisionError, msg="Adjusting length of zero length normal did not raise a ZeroDivisionError."):

            v.length = 10.0

    def test_negate(self):
        """Negate operator."""

        r = -Normal([2.5, 6.7, -4.6])
        self.assertTrue(isinstance(r, Normal), "Normal negation did not return a Normal.")
        self.assertEqual(r.x, -2.5, "Negation failed [X].")
        self.assertEqual(r.y, -6.7, "Negation failed [Y].")
        self.assertEqual(r.z, 4.60, "Negation failed [Z].")

    def test_add(self):
        """Add operator."""

        # Normal + Normal, returns Normal
        a = Normal([-1.4, 0.2, 99.1])
        b = Normal([0.7, -64.0, -0.1])
        r = a + b
        self.assertTrue(isinstance(r, Normal), "Normal + Normal did not return a Normal.")
        self.assertEqual(r.x, -1.4 + 0.7, "Normal + Normal failed [X].")
        self.assertEqual(r.y, 0.2 - 64.0, "Normal + Normal failed [Y].")
        self.assertEqual(r.z, 99.1 - 0.1, "Normal + Normal failed [Z].")

    def test_subtract(self):
        """Subtract operator."""

        # Normal - Normal, returns Normal
        a = Normal([-1.4, 0.2, 99.1])
        b = Normal([0.7, -64.0, -0.1])
        r = a - b
        self.assertTrue(isinstance(r, Normal), "Normal - Normal did not return a Normal.")
        self.assertEqual(r.x, -1.4 - 0.7, "Normal - Normal failed [X].")
        self.assertEqual(r.y, 0.2 + 64.0, "Normal - Normal failed [Y].")
        self.assertEqual(r.z, 99.1 + 0.1, "Normal - Normal failed [Z].")

    def test_multiply(self):
        """Multiply operator."""

        v = Normal([-1.4, 0.2, 99.1])

        # c * Normal, returns Normal
        r = 0.23 * v
        self.assertTrue(isinstance(r, Normal), "c * Normal did not return a Normal.")
        self.assertEqual(r.x, 0.23 * -1.4, "c * Normal failed [X].")
        self.assertEqual(r.y, 0.23 * 0.20, "c * Normal failed [Y].")
        self.assertEqual(r.z, 0.23 * 99.1, "c * Normal failed [Z].")

        # Normal * c, returns Normal
        r = v * -2.6
        self.assertTrue(isinstance(r, Normal), "Normal * c did not return a Normal.")
        self.assertEqual(r.x, -2.6 * -1.4, "Normal * c failed [X].")
        self.assertEqual(r.y, -2.6 * 0.20, "Normal * c failed [Y].")
        self.assertEqual(r.z, -2.6 * 99.1, "Normal * c failed [Z].")

    def test_divide(self):
        """Division operator."""

        v = Normal([-1.4, 0.2, 99.1])

        # Normal / c, returns Normal
        r = v / 5.3
        self.assertTrue(isinstance(r, Normal), "Normal * c did not return a Normal.")
        self.assertEqual(r.x, -1.4 / 5.3, "Normal * c failed [X].")
        self.assertEqual(r.y, 0.20 / 5.3, "Normal * c failed [Y].")
        self.assertEqual(r.z, 99.1 / 5.3, "Normal * c failed [Z].")

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
        v = Normal([23.2, 0.12, -5.0])
        r = v.normalise()
        l = v.length
        self.assertTrue(isinstance(r, Normal), "Normalise did not return a Normal.")
        self.assertAlmostEqual(r.x, 23.2 / l, 14, "Normalise failed [X].")
        self.assertAlmostEqual(r.y, 0.12 / l, 14, "Normalise failed [Y].")
        self.assertAlmostEqual(r.z, -5.0 / l, 14, "Normalise failed [Z].")

        # attempting to normalise a zero length normal should raise a ZeroDivisionError
        v = Normal([0.0, 0.0, 0.0])
        with self.assertRaises(ZeroDivisionError, msg="Normalising a zero length normal did not raise a ZeroDivisionError."):

            r = v.normalise()

    def test_dot_product(self):
        """Testing dot product."""

        x = Normal([1,0,0])
        y = Normal([0,1,0])
        z = Normal([0,0,1])

        # orthogonal
        self.assertEqual(x.dot(y), 0.0, "Dot product of orthogonal normals does not equal 0.0.")
        self.assertEqual(x.dot(z), 0.0, "Dot product of orthogonal normals does not equal 0.0.")
        self.assertEqual(y.dot(z), 0.0, "Dot product of orthogonal normals does not equal 0.0.")

        # orthonormal
        self.assertEqual(x.dot(x), 1.0, "Dot product of orthonormal normals does not equal 1.0.")
        self.assertEqual(y.dot(y), 1.0, "Dot product of orthonormal normals does not equal 1.0.")
        self.assertEqual(z.dot(z), 1.0, "Dot product of orthonormal normals does not equal 1.0.")

        # arbitrary
        a = Normal([4, 2, 3])
        b = Normal([-1, 2, 6])
        self.assertEqual(a.dot(b), 4*-1 + 2*2 + 3*6, "Dot product of two arbitrary normals gives the wrong value.")
        self.assertEqual(a.dot(b), b.dot(a), "a.b does not equal b.a.")

    def test_cross_product(self):
        """Testing cross product."""

        # raysect uses a right handed coordinate system
        x = Normal([1,0,0])
        y = Normal([0,1,0])
        z = Normal([0,0,1])

        # orthogonal
        r = x.cross(y)
        self.assertEqual(r.x, 0.0, "Cross product failed [X].")
        self.assertEqual(r.y, 0.0, "Cross product failed [Y].")
        self.assertEqual(r.z, 1.0, "Cross product failed [Z].")

        r = x.cross(z)
        self.assertEqual(r.x, 0.0, "Cross product failed [X].")
        self.assertEqual(r.y, -1.0, "Cross product failed [Y].")
        self.assertEqual(r.z, 0.0, "Cross product failed [Z].")

        r = y.cross(z)
        self.assertEqual(r.x, 1.0, "Cross product failed [X].")
        self.assertEqual(r.y, 0.0, "Cross product failed [Y].")
        self.assertEqual(r.z, 0.0, "Cross product failed [Z].")

        # orthonormal
        r = x.cross(x)
        self.assertEqual(r.x, 0.0, "Cross product failed [X].")
        self.assertEqual(r.y, 0.0, "Cross product failed [Y].")
        self.assertEqual(r.z, 0.0, "Cross product failed [Z].")

        r = y.cross(y)
        self.assertEqual(r.x, 0.0, "Cross product failed [X].")
        self.assertEqual(r.y, 0.0, "Cross product failed [Y].")
        self.assertEqual(r.z, 0.0, "Cross product failed [Z].")

        r = z.cross(z)
        self.assertEqual(r.x, 0.0, "Cross product failed [X].")
        self.assertEqual(r.y, 0.0, "Cross product failed [Y].")
        self.assertEqual(r.z, 0.0, "Cross product failed [Z].")

        # arbitrary Normal x Normal
        a = Normal([4, 2, 3])
        b = Normal([-1, 2, 6])

        r1 = a.cross(b)
        r2 = b.cross(a)

        self.assertTrue(isinstance(r, Vector), "Cross did not return a Vector.")

        self.assertEqual(r1.x, a.y * b.z - b.y * a.z, "Cross product failed [X].")
        self.assertEqual(r1.y, b.x * a.z - a.x * b.z, "Cross product failed [Y].")
        self.assertEqual(r1.z, a.x * b.y - b.x * a.y, "Cross product failed [Z].")

        self.assertEqual(r2.x, b.y * a.z - a.y * b.z, "Cross product failed [X].")
        self.assertEqual(r2.y, a.x * b.z - b.x * a.z, "Cross product failed [Y].")
        self.assertEqual(r2.z, b.x * a.y - a.x * b.y, "Cross product failed [Z].")

    def test_transform(self):
        """Testing transform() method."""

        m = AffineMatrix([[1,2,3,4],
                          [5,6,2,8],
                          [9,10,4,9],
                          [4,14,15,16]])

        v = Normal([-1, 2, 6])

        r = v.transform(m)
        self.assertTrue(isinstance(r, Normal), "Transform did not return a Normal.")
        self.assertAlmostEqual(r.x,  258/414 * -1 +  -381/414 * 2 +  210/414 * 6, places = 14, msg = "Transform failed [X].")
        self.assertAlmostEqual(r.y, -132/414 * -1 +    81/414 * 2 + -162/414 * 6, places = 14, msg = "Transform failed [Y].")
        self.assertAlmostEqual(r.z,  120/414 * -1 +   -36/414 * 2 +   72/414 * 6, places = 14, msg = "Transform failed [Z].")

    def test_transform_with_inverse(self):
        """Testing transform_with_inverse() method."""

        minv = AffineMatrix([[258/414, -132/414, 120/414, -66/414],
                             [-381/414, 81/414, -36/414, 75/414],
                             [210/414, -162/414, 72/414, -12/414],
                             [72/414, 114/414, -66/414, -12/414]])

        v = Normal([-1, 2, 6])

        r = v.transform_with_inverse(minv)
        self.assertTrue(isinstance(r, Normal), "Transform did not return a Normal.")
        self.assertAlmostEqual(r.x,  258/414 * -1 +  -381/414 * 2 +  210/414 * 6, places = 14, msg = "Transform failed [X].")
        self.assertAlmostEqual(r.y, -132/414 * -1 +    81/414 * 2 + -162/414 * 6, places = 14, msg = "Transform failed [Y].")
        self.assertAlmostEqual(r.z,  120/414 * -1 +   -36/414 * 2 +   72/414 * 6, places = 14, msg = "Transform failed [Z].")


if __name__ == "__main__":
    unittest.main()

