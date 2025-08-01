# Copyright (c) 2014-2025, Dr Alex Meakins, Raysect Project
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
Unit tests for the vector3d.Function1D class.
"""

import unittest
from raysect.core.math import Vector3D
from raysect.core.math.function.float.function1d.autowrap import PythonFunction1D as PythonFloatFunction1D
from raysect.core.math.function.vector3d.function1d.autowrap import PythonFunction1D as PythonVector3DFunction1D

# TODO: expand tests to cover the cython interface
class TestFunction1D(unittest.TestCase):

    def setUp(self):

        self.refv1 = lambda x: Vector3D(10 * x + 5, 10 * x - 5 , x)
        self.refv2 = lambda x: Vector3D(x * x + 25, x * x - 25, x * 5)

        self.vf1 = PythonVector3DFunction1D(self.refv1)
        self.vf2 = PythonVector3DFunction1D(self.refv2)

        self.ref1 = lambda x: 10 * x + 5
        self.ref2 = lambda x: x * x

        self.f1 = PythonFloatFunction1D(self.ref1)
        self.f2 = PythonFloatFunction1D(self.ref2)

    def test_call(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            self.assertEqual(self.vf1(x), self.refv1(x), "vector3d.Function1D call did not match reference function value.")

    def test_negate(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = -self.vf1
        for x in v:
            self.assertEqual(r(x), -self.refv1(x), "vector3d.Function1D negate did not match reference function value.")

    def test_add_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        c1 = Vector3D(3, 4, 5)
        c2 = Vector3D(5, 12, 13)
        r1 = c1 + self.vf1
        r2 = self.vf1 + c2
        for x in v:
            self.assertEqual(r1(x), c1 + self.refv1(x), "vector3d.Function1D add Vector3D (V + vf()) did not match reference function value.")
            self.assertEqual(r2(x), self.refv1(x) + c2, "vector3d.Function1D add Vector3D (vf() + V) did not match reference function value.")

    def test_sub_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        c1 = Vector3D(3, 4, 5)
        c2 = Vector3D(5, 12, 13)
        r1 = c1 - self.vf1
        r2 = self.vf1 - c2
        for x in v:
            self.assertEqual(r1(x), c1 - self.refv1(x), "vector3d.Function1D subtract Vector3D (V - vf()) did not match reference function value.")
            self.assertEqual(r2(x), self.refv1(x) - c2, "vector3d.Function1D subtract Vector3D (vf() - V) did not match reference function value.")

    def test_mul_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.vf1 * -7.8
        for x in v:
            self.assertEqual(r(x), self.refv1(x) * -7.8, "vector3d.Function1D multiply Vector3D (vf() * V) did not match reference function value.")

    def test_div_vector(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r = self.vf1 / -7.8
        for x in v:
            self.assertEqual(r(x), self.refv1(x) / -7.8, "vector3d.Function1D divide Vector3D (vf() / V) did not match reference function value.")

        with self.assertRaises(TypeError, msg="TypeError not raised when dividing Vector3D by function."):
            r = 5 / self.vf1

    def test_richcmp_vector(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.refv1(x)
            not_ref_value = ref_value + Vector3D(*[ref + abs(ref) + 1 for ref in ref_value])
            self.assertEqual(
                (self.vf1 == ref_value)(x), 1.0,
                msg="vector3d.Function1D equals Vector3D (vf() == V) did not return true when it should."
            )
            self.assertEqual(
                (ref_value == self.vf1)(x), 1.0,
                msg="Vector3D equals vector3d.Function1D(V == vf()) did not return true when it should."
            )
            self.assertEqual(
                (self.vf1 == not_ref_value)(x), 0.0,
                msg="vector3d.Function1D equals Vector3D (vf() == V) did not return false when it should."
            )
            self.assertEqual(
                (not_ref_value == self.vf1)(x), 0.0,
                msg="Vector3D equals vector3d.Function1D(V == vf()) did not return false when it should."
            )
            self.assertEqual(
                (self.vf1 != not_ref_value)(x), 1.0,
                msg="vector3d.Function1D not equals Vector3D (vf() != V) did not return true when it should."
            )
            self.assertEqual(
                (not_ref_value != self.vf1)(x), 1.0,
                msg="Vector3D not equals vector3d.Function1D(V != vf()) did not return true when it should."
            )
            self.assertEqual(
                (self.vf1 != ref_value)(x), 0.0,
                msg="vector3d.Function1D not equals Vector3D (vf() != V) did not return false when it should."
            )
            self.assertEqual(
                (ref_value != self.vf1)(x), 0.0,
                msg="Vector3D not equals vector3d.Function1D(V != vf()) did not return false when it should."
            )

    def test_add_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.vf1 + self.vf2
        r2 = self.refv1 + self.vf2
        r3 = self.vf1 + self.refv2
        for x in v:
            self.assertEqual(r1(x), self.refv1(x) + self.refv2(x), "vector3d.Function1D add function (vf1() + vf2()) did not match reference function value.")
            self.assertEqual(r2(x), self.refv1(x) + self.refv2(x), "vector3d.Function1D add function (p1() + vf2()) did not match reference function value.")
            self.assertEqual(r3(x), self.refv1(x) + self.refv2(x), "vector3d.Function1D add function (vf1() + p2()) did not match reference function value.")

    def test_sub_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.vf1 - self.vf2
        r2 = self.refv1 - self.vf2
        r3 = self.vf1 - self.refv2
        for x in v:
            self.assertEqual(r(x), self.refv1(x) - self.refv2(x), "vector3d.Function1D subtract function (vf1() - vf2()) did not match reference function value.")
            self.assertEqual(r2(x), self.refv1(x) - self.refv2(x), "vector3d.Function1D subtract function (p1() - vf2()) did not match reference function value.")
            self.assertEqual(r3(x), self.refv1(x) - self.refv2(x), "vector3d.Function1D subtract function (vf1() - p2()) did not match reference function value.")

    def test_mul_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.vf1 * self.f2
        # No r2 = self.refv1 * self.f2, as p1() * f2() is treated as PythonFunction1D * Function1D
        r2 = self.f1 * self.vf2
        r3 = self.vf1 * self.ref2
        for x in v:
            self.assertEqual(r1(x), self.refv1(x) * self.ref2(x), "vector3d.Function1D multiply function (vf1() * f2()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) * self.refv2(x), "vector3d.Function1D multiply function (f1() * vf2()) did not match reference function value.")
            self.assertEqual(r3(x), self.refv1(x) * self.ref2(x), "vector3d.Function1D multiply function (vf1() * p2()) did not match reference function value.")

    def test_div_function1d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = self.vf1 / self.f2
        # No r2 = self.refv1 / self.f2, as p1() / f2() is treated as PythonFunction1D / Function1D
        r3 = self.vf1 / self.ref2
        for x in v:
            self.assertEqual(r1(x), self.refv1(x) / self.ref2(x), msg="vector3d.Function1D divide function (vf1() / f2()) did not match reference function value.")
            self.assertEqual(r3(x), self.refv1(x) / self.ref2(x), msg="vector3d.Function1D divide function (vf1() / p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0)

    def test_richcmp_function1d_callable(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.refv1

            def not_ref_value(x):
                ref = self.refv1(x)
                return ref + Vector3D(*[abs(r) + 1 for r in ref])

            self.assertEqual(
                (self.vf1 == ref_value)(x), 1.0,
                msg="vector3d.Function1D equals callable (vf1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.vf1 == not_ref_value)(x), 0.0,
                msg="vector3d.Function1D equals callable (vf1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.vf1 != not_ref_value)(x), 1.0,
                msg="vector3d.Function1D not equals callable (vf1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.vf1 != ref_value)(x), 0.0,
                msg="vector3d.Function1D not equals callable (vf1() != f2()) did not return false when it should."
            )

    def test_richcmp_callable_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.refv1

            def not_ref_value(x):
                ref = self.refv1(x)
                return ref + Vector3D(*[abs(r) + 1 for r in ref])

            self.assertEqual(
                (ref_value == self.vf1)(x), 1.0,
                msg="Callable equals vector3d.Function1D(f1() == vf2()) did not return true when it should."
            )
            self.assertEqual(
                (not_ref_value == self.vf1)(x), 0.0,
                msg="Callable equals vector3d.Function1D(f1() == vf2()) did not return false when it should."
            )
            self.assertEqual(
                (not_ref_value != self.vf1)(x), 1.0,
                msg="Callable not equals vector3d.Function1D(f1() != vf2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value != self.vf1)(x), 0.0,
                msg="Callable not equals vector3d.Function1D(f1() != vf2()) did not return false when it should."
            )

    def test_richcmp_function1d_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.vf1
            shift = Vector3D(*[abs(r) + 1 for r in self.refv1(x)])
            not_ref_value = self.vf1 + shift
            self.assertEqual(
                (self.vf1 == ref_value)(x), 1.0,
                msg="vector3d.Function1D equals vector3d.Function1D (vf1() == vf2()) did not return true when it should."
            )
            self.assertEqual(
                (self.vf1 == not_ref_value)(x), 0.0,
                msg="vector3d.Function1D equals vector3d.Function1D (vf1() == vf2()) did not return false when it should."
            )
            self.assertEqual(
                (self.vf1 != not_ref_value)(x), 1.0,
                msg="vector3d.Function1D not equals vector3d.Function1D (vf1() != vf2()) did not return true when it should."
            )
            self.assertEqual(
                (self.vf1 != ref_value)(x), 0.0,
                msg="vector3d.Function1D not equals vector3d.Function1D (vf1() != vf2()) did not return false when it should."
            )
