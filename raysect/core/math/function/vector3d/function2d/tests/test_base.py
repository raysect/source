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
Unit tests for the vector3d.Function2D class.
"""

import unittest
from raysect.core.math import Vector3D
from raysect.core.math.function.float.function2d.autowrap import PythonFunction2D as PythonFloatFunction2D
from raysect.core.math.function.vector3d.function2d.autowrap import PythonFunction2D as PythonVector3DFunction2D

# TODO: expand tests to cover the cython interface
class TestFunction2D(unittest.TestCase):

    def setUp(self):

        self.refv1 = lambda x, y: Vector3D(10 * x + 5 * y, 10 * x - 5 * y, x + y)
        self.refv2 = lambda x, y: Vector3D(x * x + y * y, x * x - y * y, x * y)

        self.vf1 = PythonVector3DFunction2D(self.refv1)
        self.vf2 = PythonVector3DFunction2D(self.refv2)

        self.ref1 = lambda x, y: 10 * x + 5 * y
        self.ref2 = lambda x, y: x * x + y * y

        self.f1 = PythonFloatFunction2D(self.ref1)
        self.f2 = PythonFloatFunction2D(self.ref2)

    def test_call(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                self.assertEqual(self.vf1(x, y), self.refv1(x, y), "vector3d.Function2D call did not match reference function value.")

    def test_negate(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = -self.vf1
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), -self.refv1(x, y), "vector3d.Function2D negate did not match reference function value.")

    def test_add_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        c1 = Vector3D(3, 4, 5)
        c2 = Vector3D(5, 12, 13)
        r1 = c1 + self.vf1
        r2 = self.vf1 + c2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), c1 + self.refv1(x, y), "vector3d.Function2D add Vector3D (V + vf()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.refv1(x, y) + c2, "vector3d.Function2D add Vector3D (vf() + V) did not match reference function value.")

    def test_sub_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        c1 = Vector3D(3, 4, 5)
        c2 = Vector3D(5, 12, 13)
        r1 = c1 - self.vf1
        r2 = self.vf1 - c2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), c1 - self.refv1(x, y), "vector3d.Function2D subtract Vector3D (V - vf()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.refv1(x, y) - c2, "vector3d.Function2D subtract Vector3D (vf() - V) did not match reference function value.")

    def test_mul_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.vf1 * -7.8
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.refv1(x, y) * -7.8, "vector3d.Function2D multiply Vector3D (vf() * V) did not match reference function value.")

    def test_div_vector(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r = self.vf1 / -7.8
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.refv1(x, y) / -7.8, "vector3d.Function2D divide Vector3D (vf() / V) did not match reference function value.")

        with self.assertRaises(TypeError, msg="TypeError not raised when dividing Vector3D by function."):
            r = 5 / self.vf1

    def test_richcmp_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.refv1(x, y)
                not_ref_value = ref_value + Vector3D(*[ref + abs(ref) + 1 for ref in ref_value])
                self.assertEqual(
                    (self.vf1 == ref_value)(x, y), 1.0,
                    msg="vector3d.Function2D equals Vector3D (vf() == V) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value == self.vf1)(x, y), 1.0,
                    msg="Vector3D equals vector3d.Function2D(V == vf()) did not return true when it should."
                )
                self.assertEqual(
                    (self.vf1 == not_ref_value)(x, y), 0.0,
                    msg="vector3d.Function2D equals Vector3D (vf() == V) did not return false when it should."
                )
                self.assertEqual(
                    (not_ref_value == self.vf1)(x, y), 0.0,
                    msg="Vector3D equals vector3d.Function2D(V == vf()) did not return false when it should."
                )
                self.assertEqual(
                    (self.vf1 != not_ref_value)(x, y), 1.0,
                    msg="vector3d.Function2D not equals Vector3D (vf() != V) did not return true when it should."
                )
                self.assertEqual(
                    (not_ref_value != self.vf1)(x, y), 1.0,
                    msg="Vector3D not equals vector3d.Function2D(V != vf()) did not return true when it should."
                )
                self.assertEqual(
                    (self.vf1 != ref_value)(x, y), 0.0,
                    msg="vector3d.Function2D not equals Vector3D (vf() != V) did not return false when it should."
                )
                self.assertEqual(
                    (ref_value != self.vf1)(x, y), 0.0,
                    msg="Vector3D not equals vector3d.Function2D(V != vf()) did not return false when it should."
                )

    def test_add_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.vf1 + self.vf2
        r2 = self.refv1 + self.vf2
        r3 = self.vf1 + self.refv2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), self.refv1(x, y) + self.refv2(x, y), "vector3d.Function2D add function (vf1() + vf2()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.refv1(x, y) + self.refv2(x, y), "vector3d.Function2D add function (p1() + vf2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.refv1(x, y) + self.refv2(x, y), "vector3d.Function2D add function (vf1() + p2()) did not match reference function value.")

    def test_sub_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.vf1 - self.vf2
        r2 = self.refv1 - self.vf2
        r3 = self.vf1 - self.refv2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.refv1(x, y) - self.refv2(x, y), "vector3d.Function2D subtract function (vf1() - vf2()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.refv1(x, y) - self.refv2(x, y), "vector3d.Function2D subtract function (p1() - vf2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.refv1(x, y) - self.refv2(x, y), "vector3d.Function2D subtract function (vf1() - p2()) did not match reference function value.")

    def test_mul_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.vf1 * self.f2
        # No r2 = self.refv1 * self.f2, as p1() * f2() is treated as PythonFunction2D * Function2D
        r2 = self.f1 * self.vf2
        r3 = self.vf1 * self.ref2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), self.refv1(x, y) * self.ref2(x, y), "vector3d.Function2D multiply function (vf1() * f2()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) * self.refv2(x, y), "vector3d.Function2D multiply function (f1() * vf2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.refv1(x, y) * self.ref2(x, y), "vector3d.Function2D multiply function (vf1() * p2()) did not match reference function value.")

    def test_div_function2d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = self.vf1 / self.f2
        # No r2 = self.refv1 / self.f2, as p1() / f2() is treated as PythonFunction2D / Function2D
        r3 = self.vf1 / self.ref2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), self.refv1(x, y) / self.ref2(x, y), msg="vector3d.Function2D divide function (vf1() / f2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.refv1(x, y) / self.ref2(x, y), msg="vector3d.Function2D divide function (vf1() / p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0, 0)

    def test_richcmp_function2d_callable(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.refv1

                def not_ref_value(x, y):
                    ref = self.refv1(x, y)
                    return ref + Vector3D(*[abs(r) + 1 for r in ref])

                self.assertEqual(
                    (self.vf1 == ref_value)(x, y), 1.0,
                    msg="vector3d.Function2D equals callable (vf1() == f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.vf1 == not_ref_value)(x, y), 0.0,
                    msg="vector3d.Function2D equals callable (vf1() == f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.vf1 != not_ref_value)(x, y), 1.0,
                    msg="vector3d.Function2D not equals callable (vf1() != f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.vf1 != ref_value)(x, y), 0.0,
                    msg="vector3d.Function2D not equals callable (vf1() != f2()) did not return false when it should."
                )

    def test_richcmp_callable_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.refv1

                def not_ref_value(x, y):
                    ref = self.refv1(x, y)
                    return ref + Vector3D(*[abs(r) + 1 for r in ref])

                self.assertEqual(
                    (ref_value == self.vf1)(x, y), 1.0,
                    msg="Callable equals vector3d.Function2D(f1() == vf2()) did not return true when it should."
                )
                self.assertEqual(
                    (not_ref_value == self.vf1)(x, y), 0.0,
                    msg="Callable equals vector3d.Function2D(f1() == vf2()) did not return false when it should."
                )
                self.assertEqual(
                    (not_ref_value != self.vf1)(x, y), 1.0,
                    msg="Callable not equals vector3d.Function2D(f1() != vf2()) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value != self.vf1)(x, y), 0.0,
                    msg="Callable not equals vector3d.Function2D(f1() != vf2()) did not return false when it should."
                )

    def test_richcmp_function2d_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.vf1
                shift = Vector3D(*[abs(r) + 1 for r in self.refv1(x, y)])
                not_ref_value = self.vf1 + shift
                self.assertEqual(
                    (self.vf1 == ref_value)(x, y), 1.0,
                    msg="vector3d.Function2D equals vector3d.Function2D (vf1() == vf2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.vf1 == not_ref_value)(x, y), 0.0,
                    msg="vector3d.Function2D equals vector3d.Function2D (vf1() == vf2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.vf1 != not_ref_value)(x, y), 1.0,
                    msg="vector3d.Function2D not equals vector3d.Function2D (vf1() != vf2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.vf1 != ref_value)(x, y), 0.0,
                    msg="vector3d.Function2D not equals vector3d.Function2D (vf1() != vf2()) did not return false when it should."
                )
