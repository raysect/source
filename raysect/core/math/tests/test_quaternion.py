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
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS'
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
Unit tests for the Quaternion object.
"""

import unittest
from math import sqrt
from raysect.core.math import Quaternion, AffineMatrix3D, Vector3D, rotate_x, rotate_z


class TestQuaternion(unittest.TestCase):

    def test_initialise_default(self):
        """Default initialisation, identity quaternion."""

        q = Quaternion()
        self.assertEqual(q.x, 0.0, 'Default initialisation is not (<0,0,0>,1) [X].')
        self.assertEqual(q.y, 0.0, 'Default initialisation is not (<0,0,0>,1) [Y].')
        self.assertEqual(q.z, 0.0, 'Default initialisation is not (<0,0,0>,1) [Z].')
        self.assertEqual(q.s, 1.0, 'Default initialisation is not (<0,0,0>,1) [S].')

    def test_initialise_indexable(self):
        """Initialisation with an indexable object."""

        q = Quaternion(1.0, 2.0, 3.0, 4.0)
        self.assertEqual(q.x, 1.0, 'Initialisation with indexable failed [X].')
        self.assertEqual(q.y, 2.0, 'Initialisation with indexable failed [Y].')
        self.assertEqual(q.z, 3.0, 'Initialisation with indexable failed [Z].')
        self.assertEqual(q.s, 4.0, 'Initialisation with indexable failed [S].')

    def test_initialise_invalid(self):
        """Initialisation with invalid types should raise a TypeError."""

        with self.assertRaises(TypeError, msg='Initialised with a string.'):
            Quaternion('spoon')

    def test_x(self):
        """Get/set x co-ordinate."""

        q = Quaternion(2.5, 6.7, -4.6, 1.0)

        # get x attribute
        self.assertEqual(q.x, 2.5, 'Getting x attribute failed.')

        # set x attribute
        q.x = 10.0
        self.assertEqual(q.x, 10.0, 'Setting x attribute failed.')

    def test_y(self):
        """Get/set y co-ordinate."""

        q = Quaternion(2.5, 6.7, -4.6, 1.0)

        # get y attribute
        self.assertEqual(q.y, 6.7, 'Getting y attribute failed.')

        # set y attribute
        q.y = -7.1
        self.assertEqual(q.y, -7.1, 'Setting y attribute failed.')

    def test_z(self):
        """Get/set z co-ordinate."""

        q = Quaternion(2.5, 6.7, -4.6, 1.0)

        # get z attribute
        self.assertEqual(q.z, -4.6, 'Getting z attribute failed.')

        # set z attribute
        q.z = 157.3
        self.assertEqual(q.z, 157.3, 'Setting z attribute failed.')

    def test_s(self):
        """Get/set s co-ordinate."""

        q = Quaternion(2.5, 6.7, -4.6, 1.0)

        # get x attribute
        self.assertEqual(q.s, 1.0, 'Getting s attribute failed.')

        # set x attribute
        q.s = 10.0
        self.assertEqual(q.s, 10.0, 'Setting s attribute failed.')

    def test_indexing(self):
        """Getting/setting components by indexing."""

        q = Quaternion(2.5, 6.7, -4.6, 1.0)

        q[0] = 2.0
        q[1] = 7.0
        q[2] = 10.0
        q[3] = 1.5

        # check getting/setting via valid indexes
        self.assertEqual(q[0], 2.0, 'Indexing failed [X].')
        self.assertEqual(q[1], 7.0, 'Indexing failed [Y].')
        self.assertEqual(q[2], 10.0, 'Indexing failed [Z].')
        self.assertEqual(q[3], 1.5, 'Indexing failed [S].')

        # check invalid indexes
        with self.assertRaises(IndexError, msg='Invalid positive index did not raise IndexError.'):
            r = q[4]

        with self.assertRaises(IndexError, msg='Invalid negative index did not raise IndexError.'):
            r = q[-1]

    def test_equal(self):
        """Equality operator."""

        self.assertTrue(Quaternion(1, 2, 3, 4) == Quaternion(1, 2, 3, 4),
                        'Equality operator returned false for equal quaternions.')
        self.assertFalse(Quaternion(5, 2, 3, 4) == Quaternion(1, 2, 3, 4),
                         'Equality operator returned true for quaternions with non-equal x components.')
        self.assertFalse(Quaternion(1, 5, 3, 4) == Quaternion(1, 2, 3, 4),
                         'Equality operator returned true for quaternions with non-equal y components.')
        self.assertFalse(Quaternion(1, 2, 5, 4) == Quaternion(1, 2, 3, 4),
                         'Equality operator returned true for quaternions with non-equal z components.')
        self.assertFalse(Quaternion(1, 2, 3, 5) == Quaternion(1, 2, 3, 4),
                         'Equality operator returned true for quaternions with non-equal s components.')

    def test_not_equal(self):
        """Inequality operator."""

        self.assertFalse(Quaternion(1, 2, 3, 4) != Quaternion(1, 2, 3, 4),
                         'Inequality operator returned true for equal quaternions.')
        self.assertTrue(Quaternion(5, 1, 3, 4) != Quaternion(1, 2, 3, 4),
                        'Inequality operator returned false for quaternions with non-equal x components.')
        self.assertTrue(Quaternion(1, 5, 3, 4) != Quaternion(1, 2, 3, 4),
                        'Inequality operator returned false for quaternions with non-equal y components.')
        self.assertTrue(Quaternion(1, 2, 5, 4) != Quaternion(1, 2, 3, 4),
                        'Inequality operator returned false for quaternions with non-equal z components.')
        self.assertTrue(Quaternion(1, 2, 3, 5) != Quaternion(1, 2, 3, 4),
                        'Inequality operator returned false for quaternions with non-equal s components.')

    def test_iter(self):
        """Obtain values by iteration."""

        q = Quaternion(2.5, 6.7, -4.6, 1.0)
        l = list(q)
        self.assertEqual(len(l), 4, 'Iteration failed to return the correct number of items.')
        self.assertEqual(l[0], 2.5, 'Iteration failed [X].')
        self.assertEqual(l[1], 6.7, 'Iteration failed [Y].')
        self.assertEqual(l[2], -4.6, 'Iteration failed [Z].')
        self.assertEqual(l[3], 1.0, 'Iteration failed [S].')

    def test_add(self):
        """Addition operator."""

        q1 = Quaternion(2, 3, 4, 1)
        q2 = Quaternion(2, -3, 10, 0)

        q_theory = Quaternion(4, 0, 14, 1)
        q_result = q1 + q2

        self.assertEqual(q_result.x, q_theory.x, 'Addition of two quaternions failed [X].')
        self.assertEqual(q_result.y, q_theory.y, 'Addition of two quaternions failed [Y].')
        self.assertEqual(q_result.z, q_theory.z, 'Addition of two quaternions failed [Z].')
        self.assertEqual(q_result.s, q_theory.s, 'Addition of two quaternions failed [S].')

    def test_subtract(self):
        """Subtraction operator."""

        q1 = Quaternion(2, 3, 4, 1)
        q2 = Quaternion(2, -3, 10, 0)

        # desired result for subtraction of q1 and q2
        q_theory = Quaternion(0, 6, -6, 1)
        q_result = q1 - q2

        self.assertEqual(q_result.x, q_theory.x, 'Subtraction of two quaternions failed [X].')
        self.assertEqual(q_result.y, q_theory.y, 'Subtraction of two quaternions failed [Y].')
        self.assertEqual(q_result.z, q_theory.z, 'Subtraction of two quaternions failed [Z].')
        self.assertEqual(q_result.s, q_theory.s, 'Subtraction of two quaternions failed [S].')

    def test_negation(self):
        """Negation operation"""

        q = Quaternion(1, 2, 3, 4)
        q_result = Quaternion(-1, -2, -3, -4)

        self.assertTrue(-q == q_result, 'Negation of a quaternion failed to produce the correct result.')

    def test_multiplication(self):
        """Multiplication operation"""

        q1 = Quaternion(0, 1, 0, 1)
        q2 = Quaternion(0.5, 0.5, 0.75, 1)

        q_result = Quaternion(0, 3, 0, 3)
        self.assertTrue(q1 * 3 == q_result,
                        'Multiplication of a quaternion and a scalar failed to produce the correct result.')
        self.assertTrue(3 * q1 == q_result,
                        'Multiplication of a quaternion and a scalar failed to produce the correct result.')

        q_result = Quaternion(1.25, 1.5, 0.25, 0.5)
        self.assertTrue(q1 * q2 == q_result, 'Multiplication of two quaternions failed to produce the correct result.')

    def test_division(self):
        """Division operation"""

        q1 = Quaternion(0, 1, 0, 1)
        q2 = Quaternion(0.5, 0.5, 0.75, 1)

        q_theory = Quaternion(-0.60606060606, 0.242424242424, -0.12121212121212, 0.7272727272)
        q_result = q1 / q2

        self.assertAlmostEqual(q_result.x, q_theory.x, delta=1e-10,
                               msg='Division of two quaternions failed to produce the correct result [X].')
        self.assertAlmostEqual(q_result.y, q_theory.y, delta=1e-10,
                               msg='Division of two quaternions failed to produce the correct result [Y].')
        self.assertAlmostEqual(q_result.z, q_theory.z, delta=1e-10,
                               msg='Division of two quaternions failed to produce the correct result [Z].')
        self.assertAlmostEqual(q_result.s, q_theory.s, delta=1e-10,
                               msg='Division of two quaternions failed to produce the correct result [S].')

        q_result = Quaternion(0.25, 0.25, 0.375, 0.5)
        self.assertTrue(q2 / 2 == q_result,
                        'Division of a quaternion and a scalar failed to produce the correct result.')

    def test_conjugate(self):
        """Test conjugation operation"""

        q = Quaternion(2, -3, 0, 1)
        result = q.conjugate()
        answer = Quaternion(-2, 3, 0, 1)

        self.assertEqual(result.x, answer.x,
                         msg='Conjugation of a Quaternion failed to produce the correct result [X].')
        self.assertEqual(result.y, answer.y,
                         msg='Conjugation of a Quaternion failed to produce the correct result [Y].')
        self.assertEqual(result.z, answer.z,
                         msg='Conjugation of a Quaternion failed to produce the correct result [Z].')

    def test_inverse(self):
        """Inverse operation"""

        q = Quaternion(0, 1, 0, 1)
        q_result = q.inverse()

        self.assertAlmostEqual(q_result.x, 0, delta=1e-10,
                               msg='Inverse of a quaternion failed to produce the correct result [X].')
        self.assertAlmostEqual(q_result.y, -0.5, delta=1e-10,
                               msg='Inverse of a quaternion failed to produce the correct result [Y].')
        self.assertAlmostEqual(q_result.z, 0, delta=1e-10,
                               msg='Inverse of a quaternion failed to produce the correct result [Z].')
        self.assertAlmostEqual(q_result.s, 0.5, delta=1e-10,
                               msg='Inverse of a quaternion failed to produce the correct result [S].')

    def test_length(self):
        """Length (norm) of a quaternion"""

        q = Quaternion(2, 3, 4, 1)

        self.assertAlmostEqual(q.length, 5.477225575051661, delta=1e-10,
                               msg='Length of a quaternion failed to produce the correct result.')

    def test_normalise(self):
        """Normalising a quaternion"""

        q = Quaternion(2, 3, 4, 1)
        q_result = q.normalise()

        self.assertAlmostEqual(q_result.x, 0.3651483716701107, delta=1e-10,
                               msg='Norm of a quaternion failed to produce the correct result.')
        self.assertAlmostEqual(q_result.y, 0.5477225575051661, delta=1e-10,
                               msg='Norm of a quaternion failed to produce the correct result.')
        self.assertAlmostEqual(q_result.z, 0.7302967433402214, delta=1e-10,
                               msg='Norm of a quaternion failed to produce the correct result.')
        self.assertAlmostEqual(q_result.s, 0.18257418583505536, delta=1e-10,
                               msg='Norm of a quaternion failed to produce the correct result.')

    def test_is_unit(self):
        """Testing method is_unit()"""

        self.assertTrue(Quaternion(1, 0, 0, 0).is_unit(),
                        'Quaternion is_unit() test failed to produce the correct result.')

        self.assertTrue(Quaternion(0, 1, 0, 0).is_unit(),
                        'Quaternion is_unit() test failed to produce the correct result.')

        self.assertTrue(Quaternion(0, 0, 1, 0).is_unit(),
                        'Quaternion is_unit() test failed to produce the correct result.')

        self.assertTrue(Quaternion(0, 0, 0, 1).is_unit(),
                        'Quaternion is_unit() test failed to produce the correct result.')

        self.assertFalse(Quaternion(1, 2, 3, 4).is_unit(),
                         'Quaternion is_unit() test failed to produce the correct result.')

        self.assertFalse(Quaternion(-0.777777, 3.1214352, -0.9483527, 3.1415).is_unit(),
                         'Quaternion is_unit() test failed to produce the correct result.')

    def test_axis_angle_decomposition(self):

        q = Quaternion(1, 1, 1, 1)
        axis = q.axis
        angle = q.angle

        self.assertAlmostEqual(axis.x, 0.5773502691896258, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(axis.y, 0.5773502691896258, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(axis.z, 0.5773502691896258, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(angle, 120, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')

        q = Quaternion(2, 3, 4, 1)
        axis = q.axis
        angle = q.angle

        self.assertAlmostEqual(axis.x, 0.3713906763541038, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(axis.y, 0.5570860145311557, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(axis.z, 0.7427813527082076, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(angle, 158.96053021868278, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')

        q = Quaternion(0, 0, 0, 1)
        axis = q.axis
        angle = q.angle

        self.assertAlmostEqual(axis.x, 0, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(axis.y, 0, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(axis.z, 0, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')
        self.assertAlmostEqual(angle, 0, delta=1e-10,
                               msg='Decomposition of a quaternion into axis angle representation '
                                   'failed to produce the correct result.')

    def test_copy(self):
        """Testing method copy()"""

        q = Quaternion(1, 2, 3, 4)
        r = q.copy()

        # check a new instance has been created by modifying the original
        q.x = 5.0
        q.y = 6.0
        q.z = 7.0
        q.s = 8.0

        self.assertEqual(r.x, 1.0, 'Copy failed [X].')
        self.assertEqual(r.y, 2.0, 'Copy failed [Y].')
        self.assertEqual(r.z, 3.0, 'Copy failed [Z].')
        self.assertEqual(r.s, 4.0, 'Copy failed [S].')

    def test_transform(self):

        # transform between space 1 and space 2
        m = rotate_z(47)

        # define in coordinate space 1
        v1 = Vector3D(1, 0.3, -0.2)
        q1 = Quaternion.from_axis_angle(Vector3D(0.1, -0.7, 0.2), 57)

        # transform to coordinate space 2
        v2 = v1.transform(m)
        q2 = q1.transform(m)

        # use quaternion to rotate vector in each space
        r1 = v1.transform(q1.as_matrix())
        r2 = v2.transform(q2.as_matrix())

        # convert result in space 2 to space 1 for comparison
        r2_1 = r2.transform(m.inverse())

        self.assertAlmostEqual(r1.x, r2_1.x, delta=1e-10, msg='Transform failed [X]')
        self.assertAlmostEqual(r1.y, r2_1.y, delta=1e-10, msg='Transform failed [Y]')
        self.assertAlmostEqual(r1.z, r2_1.z, delta=1e-10, msg='Transform failed [Z]')

    def test_as_matrix(self):
        """Test AffineMatrix3D generation from a quaternion"""

        message = 'Conversion of a Quaternion to AffineMatrix3D failed to produce the correct result.'

        matrix = Quaternion(0.5, 0, 0, 0.5).as_matrix()
        answer = rotate_x(90)

        # TODO - replace this with a utility function e.g. _assert_matrix()
        self.assertAlmostEqual(matrix[0, 0], answer[0, 0], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[0, 1], answer[0, 1], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[0, 2], answer[0, 2], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[0, 3], answer[0, 3], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[1, 0], answer[1, 0], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[1, 1], answer[1, 1], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[1, 2], answer[1, 2], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[1, 3], answer[1, 3], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[2, 0], answer[2, 0], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[2, 1], answer[2, 1], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[2, 2], answer[2, 2], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[2, 3], answer[2, 3], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[3, 0], answer[3, 0], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[3, 1], answer[3, 1], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[3, 2], answer[3, 2], delta=1e-10, msg=message)
        self.assertAlmostEqual(matrix[3, 3], answer[3, 3], delta=1e-10, msg=message)

        # TODO - increase the resolution of this test by calculating quaternion more accurately
        matrix = Quaternion(0.3826834, 0, 0, 0.923879).as_matrix()
        answer = rotate_x(45)

        self.assertAlmostEqual(matrix[0, 0], answer[0, 0], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[0, 1], answer[0, 1], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[0, 2], answer[0, 2], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[0, 3], answer[0, 3], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[1, 0], answer[1, 0], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[1, 1], answer[1, 1], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[1, 2], answer[1, 2], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[1, 3], answer[1, 3], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[2, 0], answer[2, 0], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[2, 1], answer[2, 1], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[2, 2], answer[2, 2], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[2, 3], answer[2, 3], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[3, 0], answer[3, 0], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[3, 1], answer[3, 1], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[3, 2], answer[3, 2], delta=1e-6, msg=message)
        self.assertAlmostEqual(matrix[3, 3], answer[3, 3], delta=1e-6, msg=message)

    def test_from_axis_angle(self):
        """Test generation of a quaternion from axis angle specification"""

        result = Quaternion.from_axis_angle(Vector3D(1, 0, 0), 45)
        answer = Quaternion(0.3826834323650898, 0.0, 0.0, 0.9238795325112867)

        self.assertAlmostEqual(answer.x, result.x, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [X].')
        self.assertAlmostEqual(answer.y, result.y, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [Y].')
        self.assertAlmostEqual(answer.z, result.z, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [Z].')
        self.assertAlmostEqual(answer.s, result.s, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [S].')

        result = Quaternion.from_axis_angle(Vector3D(0.5, 0.5, 0), -30)
        answer = Quaternion(-0.1830127018922193, -0.1830127018922193, -0.0, 0.9659258262890683)

        self.assertAlmostEqual(answer.y, result.y, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [Y].')
        self.assertAlmostEqual(answer.z, result.z, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [Z].')
        self.assertAlmostEqual(answer.x, result.x, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [X].')
        self.assertAlmostEqual(answer.s, result.s, delta=1e-10,
                               msg='Converting axis angle to quaternion produced wrong result [S].')

    def test_from_matrix(self):
        """Tests the extraction of a rotation quaternion from an AffineMatrix3D"""

        result = Quaternion.from_matrix(rotate_x(90))
        answer = Quaternion(0.7071067811865475, 0.0, 0.0, 0.7071067811865476)

        self.assertAlmostEqual(answer.x, result.x, delta=1e-10,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [X].')
        self.assertAlmostEqual(answer.y, result.y, delta=1e-10,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [Y].')
        self.assertAlmostEqual(answer.z, result.z, delta=1e-10,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [Z].')
        self.assertAlmostEqual(answer.s, result.s, delta=1e-10,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [S].')

        matrix = AffineMatrix3D(((0.9330127, 0.0669873, -0.3535534, 0.0),
                                 (0.0669873, 0.9330127, 0.3535534, 0.0),
                                 (0.3535534, -0.3535534, 0.8660254, 0.0),
                                 (0.0, 0.0, 0.0, 1.0)))

        result = Quaternion.from_matrix(matrix)
        answer = Quaternion(-0.1830127018922193, -0.1830127018922193, -0.0, 0.9659258262890683)

        self.assertAlmostEqual(answer.x, result.x, delta=1e-6,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [X].')
        self.assertAlmostEqual(answer.y, result.y, delta=1e-6,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [Y].')
        self.assertAlmostEqual(answer.z, result.z, delta=1e-6,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [Z].')
        self.assertAlmostEqual(answer.s, result.s, delta=1e-6,
                               msg='Extracting quaternion from AffineMatrix3D produced wrong result [S].')

    def test_quaternion_to(self):
        """Tests the calculation of a quaternion between two quaternions."""

        q1 = Quaternion.from_axis_angle(Vector3D(1.0, 0.3, -0.1), 10)
        q2 = Quaternion.from_axis_angle(Vector3D(0.2, 1.0, 1.0), 25)
        d = q1.quaternion_to(q2)

        # transform a vector with q1 + d and q2, both should result in the same answer
        v = Vector3D(1, 3, 4)
        r1 = v.transform(q1.as_matrix()).transform(d.as_matrix())
        r2 = v.transform(q2.as_matrix())

        self.assertAlmostEqual(r1.x, r2.x, delta=1e-10, msg='Rotation_to failed [X]')
        self.assertAlmostEqual(r1.y, r2.y, delta=1e-10, msg='Rotation_to failed [Y]')
        self.assertAlmostEqual(r1.z, r2.z, delta=1e-10, msg='Rotation_to failed [Z]')


if __name__ == '__main__':
    unittest.main()
