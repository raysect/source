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
Unit tests for the Function2D class.
"""

import math
import unittest
from raysect.core.math.function.float.function2d.autowrap import PythonFunction2D

# TODO: expand tests to cover the cython interface
class TestFunction2D(unittest.TestCase):

    def setUp(self):

        self.ref1 = lambda x, y: 10 * x + 5 * y
        self.ref2 = lambda x, y: x * x + y * y

        self.f1 = PythonFunction2D(self.ref1)
        self.f2 = PythonFunction2D(self.ref2)

    def test_call(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                self.assertEqual(self.f1(x, y), self.ref1(x, y), "Function2D call did not match reference function value.")

    def test_negate(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = -self.f1
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), -self.ref1(x, y), "Function2D negate did not match reference function value.")

    def test_add_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 + self.f1
        r2 = self.f1 + 65
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 8 + self.ref1(x, y), "Function2D add scalar (K + f()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) + 65, "Function2D add scalar (f() + K) did not match reference function value.")

    def test_sub_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 - self.f1
        r2 = self.f1 - 65
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 8 - self.ref1(x, y), "Function2D subtract scalar (K - f()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) - 65, "Function2D subtract scalar (f() - K) did not match reference function value.")

    def test_mul_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 5 * self.f1
        r2 = self.f1 * -7.8
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 5 * self.ref1(x, y), "Function2D multiply scalar (K * f()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) * -7.8, "Function2D multiply scalar (f() * K) did not match reference function value.")

    def test_div_scalar(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = 5 / self.f1
        r2 = self.f1 / -7.8
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 5 / self.ref1(x, y), "Function2D divide scalar (K / f()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y), self.ref1(x, y) / -7.8, delta=abs(r2(x, y)) * 1e-12, msg="Function2D divide scalar (f() / K) did not match reference function value.")

        r = 5 / self.f1
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r(0, 0)

    def test_mod_function2d_scalar(self):
        # Note that Function2D objects work with doubles, so the floating modulo
        # operator is used rather than the integer one. For accurate testing we
        # therefore need to use the math.fmod operator rather than % in Python.
        v = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5 % self.f1
        r2 = self.f1 % -7.8
        for x in v:
            for y in v:
                self.assertAlmostEqual(r1(x, y), math.fmod(5, self.ref1(x, y)), 15, "Function2D modulo scalar (K % f()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y), math.fmod(self.ref1(x, y), -7.8), 15, "Function2D modulo scalar (f() % K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns 0."):
            r1(0, 0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when modulo scalar is 0."):
            self.f1 % 0

    def test_pow_function2d_scalar(self):
        v = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5 ** self.f1
        r2 = self.f1 ** -7.8
        r3 = (-5) ** self.f1
        for x in v:
            for y in v:
                self.assertAlmostEqual(r1(x, y), 5 ** self.ref1(x, y), 15, "Function2D power scalar (K ** f()) did not match reference function value.")
                if self.ref1(x, y) < 0:
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral."):
                        r2(x, y)
                elif not float(self.ref1(x, y)).is_integer():
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral."):
                        r3(x, y)
                else:
                    self.assertAlmostEqual(r2(x, y), self.ref1(x, y) ** -7.8, 15, "Function2D power scalar (f() ** K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative."):
            r2(0, 0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative."):
            r4 = 0 ** self.f1
            r4(-1, 0)

    def test_richcmp_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.ref1(x, y)
                higher_value = ref_value + abs(ref_value) + 1
                lower_value = ref_value - abs(ref_value) - 1
                self.assertEqual(
                    (self.f1 == ref_value)(x, y), 1.0,
                    msg="Function2D equals scalar (f() == K) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value == self.f1)(x, y), 1.0,
                    msg="Scalar equals Function2D (K == f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 == higher_value)(x, y), 0.0,
                    msg="Function2D equals scalar (f() == K) did not return false when it should."
                )
                self.assertEqual(
                    (higher_value == self.f1)(x, y), 0.0,
                    msg="Scalar equals Function2D (K == f()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 != higher_value)(x, y), 1.0,
                    msg="Function2D not equals scalar (f() != K) did not return true when it should."
                )
                self.assertEqual(
                    (higher_value != self.f1)(x, y), 1.0,
                    msg="Scalar not equals Function2D (K != f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 != ref_value)(x, y), 0.0,
                    msg="Function2D not equals scalar (f() != K) did not return false when it should."
                )
                self.assertEqual(
                    (ref_value != self.f1)(x, y), 0.0,
                    msg="Scalar not equals Function2D (K != f()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 < higher_value)(x, y), 1.0,
                    msg="Function2D less than scalar (f() < K) did not return true when it should."
                )
                self.assertEqual(
                    (lower_value < self.f1)(x, y), 1.0,
                    msg="Scalar less than Function2D (K < f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 < lower_value)(x, y), 0.0,
                    msg="Function2D less than scalar (f() < K) did not return false when it should."
                )
                self.assertEqual(
                    (higher_value < self.f1)(x, y), 0.0,
                    msg="Scalar less than Function2D (K < f()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 > lower_value)(x, y), 1.0,
                    msg="Function2D greater than scalar (f() > K) did not return true when it should."
                )
                self.assertEqual(
                    (higher_value > self.f1)(x, y), 1.0,
                    msg="Scalar greater than Function2D (K > f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 > higher_value)(x, y), 0.0,
                    msg="Function2D greater than scalar (f() > K) did not return false when it should."
                )
                self.assertEqual(
                    (lower_value > self.f1)(x, y), 0.0,
                    msg="Scalar greater than Function2D (K > f()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 <= higher_value)(x, y), 1.0,
                    msg="Function2D less equals scalar (f() <= K) did not return true when it should."
                )
                self.assertEqual(
                    (lower_value <= self.f1)(x, y), 1.0,
                    msg="Scalar less equals Function2D (K <= f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 <= ref_value)(x, y), 1.0,
                    msg="Function2D less equals scalar (f() <= K) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value <= self.f1)(x, y), 1.0,
                    msg="Scalar less equals Function2D (K <= f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 <= lower_value)(x, y), 0.0,
                    msg="Function2D less equals scalar (f() <= K) did not return false when it should."
                )
                self.assertEqual(
                    (higher_value <= self.f1)(x, y), 0.0,
                    msg="Scalar less equals Function2D (K <= f()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 >= lower_value)(x, y), 1.0,
                    msg="Function2D greater equals scalar (f() >= K) did not return true when it should."
                )
                self.assertEqual(
                    (higher_value >= self.f1)(x, y), 1.0,
                    msg="Scalar greater equals Function2D (K >= f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 >= ref_value)(x, y), 1.0,
                    msg="Function2D greater equals scalar (f() >= K) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value >= self.f1)(x, y), 1.0,
                    msg="Scalar greater equals Function2D (K >= f()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 >= higher_value)(x, y), 0.0,
                    msg="Function2D greater equals scalar (f() >= K) did not return false when it should."
                )
                self.assertEqual(
                    (lower_value >= self.f1)(x, y), 0.0,
                    msg="Scalar greater equals Function2D (K >= f()) did not return false when it should."
                )

    def test_add_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 + self.f2
        r2 = self.ref1 + self.f2
        r3 = self.f1 + self.ref2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), self.ref1(x, y) + self.ref2(x, y), "Function2D add function (f1() + f2()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) + self.ref2(x, y), "Function2D add function (p1() + f2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.ref1(x, y) + self.ref2(x, y), "Function2D add function (f1() + p2()) did not match reference function value.")

    def test_sub_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 - self.f2
        r2 = self.ref1 - self.f2
        r3 = self.f1 - self.ref2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.ref1(x, y) - self.ref2(x, y), "Function2D subtract function (f1() - f2()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) - self.ref2(x, y), "Function2D subtract function (p1() - f2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.ref1(x, y) - self.ref2(x, y), "Function2D subtract function (f1() - p2()) did not match reference function value.")

    def test_mul_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 * self.f2
        r2 = self.ref1 * self.f2
        r3 = self.f1 * self.ref2
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), self.ref1(x, y) * self.ref2(x, y), "Function2D multiply function (f1() * f2()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.ref1(x, y) * self.ref2(x, y), "Function2D multiply function (p1() * f2()) did not match reference function value.")
                self.assertEqual(r3(x, y), self.ref1(x, y) * self.ref2(x, y), "Function2D multiply function (f1() * p2()) did not match reference function value.")

    def test_div_function2d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = self.f1 / self.f2
        r2 = self.ref1 / self.f2
        r3 = self.f1 / self.ref2
        for x in v:
            for y in v:
                self.assertAlmostEqual(r1(x, y), self.ref1(x, y) / self.ref2(x, y), delta=abs(r1(x, y)) * 1e-12, msg="Function2D divide function (f1() / f2()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y), self.ref1(x, y) / self.ref2(x, y), delta=abs(r2(x, y)) * 1e-12, msg="Function2D divide function (p1() / f2()) did not match reference function value.")
                self.assertAlmostEqual(r3(x, y), self.ref1(x, y) / self.ref2(x, y), delta=abs(r3(x, y)) * 1e-12, msg="Function2D divide function (f1() / p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0, 0)

    def test_mod_function2d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e9]
        r1 = self.f1 % self.f2
        r2 = self.ref1 % self.f2
        r3 = self.f1 % self.ref2
        for x in v:
            for y in v:
                self.assertAlmostEqual(r1(x, y), math.fmod(self.ref1(x, y), self.ref2(x, y)), delta=abs(r1(x, y)) * 1e-12, msg="Function2D modulo function (f1() % f2()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y), math.fmod(self.ref1(x, y), self.ref2(x, y)), delta=abs(r2(x, y)) * 1e-12, msg="Function2D modulo function (p1() % f2()) did not match reference function value.")
                self.assertAlmostEqual(r3(x, y), math.fmod(self.ref1(x, y), self.ref2(x, y)), delta=abs(r3(x, y)) * 1e-12, msg="Function2D modulo function (f1() % p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0, 0)

    def test_pow_function2d_function2d(self):
        v = [-3.0, -0.7, -0.001, 0.00003, 2]
        r1 = self.f1 ** self.f2
        r2 = self.ref1 ** self.f2
        r3 = self.f1 ** self.ref2
        for x in v:
            for y in v:
                if self.ref1(x, y) < 0 and not float(self.ref2(x, y)).is_integer():
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (1/3)."):
                        r1(x, y)
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (2/3)."):
                        r2(x, y)
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (3/3)."):
                        r3(x, y)
                else:
                    self.assertAlmostEqual(r1(x, y), self.ref1(x, y) ** self.ref2(x, y), 15, "Function2D power function (f1() ** f2()) did not match reference function value.")
                    self.assertAlmostEqual(r2(x, y), self.ref1(x, y) ** self.ref2(x, y), 15, "Function2D power function (p1() ** f2()) did not match reference function value.")
                    self.assertAlmostEqual(r3(x, y), self.ref1(x, y) ** self.ref2(x, y), 15, "Function2D power function (f1() ** p2()) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when f1() == 0 and f2() is negative."):
            r4 = PythonFunction2D(lambda x, y: 0) ** self.f1
            r4(-1, 0)

    def test_pow_3_arguments(self):
        v = [-10, -7, -0.001, 0.00003, 2]
        r1 = pow(self.f1, 5, 3)
        r2 = pow(5, self.f1, 3)
        r3 = pow(5, self.f1, self.f2)
        r4 = pow(self.f2, self.f1, self.f2)
        r5 = pow(self.f2, self.ref1, self.ref2)
        r6 = pow(self.ref2, self.f1, self.f2)
        # Can't use 3 argument pow() if all arguments aren't integers, so
        # use fmod(a, b) % c instead
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), math.fmod(self.ref1(x, y) ** 5, 3), "Function2D 3 argument pow(f1(), A, B) did not match reference value.")
                self.assertEqual(r2(x, y), math.fmod(5 ** self.ref1(x, y), 3), "Function2D 3 argument pow(A, f1(), B) did not match reference value.")
                self.assertEqual(r3(x, y), math.fmod(5 ** self.ref1(x, y), self.ref2(x, y)), "Function2D 3 argument pow(A, f1(), f2()) did not match reference value.")
                self.assertEqual(r4(x, y), math.fmod(self.ref2(x, y) ** self.ref1(x, y), self.ref2(x, y)), "Function2D 3 argument pow(f2(), f1(), f2()) did not match reference value.")
                self.assertEqual(r5(x, y), math.fmod(self.ref2(x, y) ** self.ref1(x, y), self.ref2(x, y)), "Function2D 3 argument pow(f2(), p1(), p2()) did not match reference value.")
                self.assertEqual(r6(x, y), math.fmod(self.ref2(x, y) ** self.ref1(x, y), self.ref2(x, y)), "Function2D 3 argument pow(p2(), f1(), f2()) did not match reference value.")

    def test_abs(self):
        v = [-1e10, -7, -0.001, 0.0, 0.0003, 10, 2.3e49]
        for x in v:
            for y in v:
                self.assertEqual(abs(self.f1)(x, y), abs(self.ref1(x, y)),
                                 msg="abs(Function2D) did not match reference value")

    def test_richcmp_function_callable(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.ref1
                higher_value = lambda x, y: self.ref1(x, y) + abs(self.ref1(x, y)) + 1
                lower_value = lambda x, y: self.ref1(x, y) - abs(self.ref1(x, y)) - 1
                self.assertEqual(
                    (self.f1 == ref_value)(x, y), 1.0,
                    msg="Function2D equals callable (f1() == f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 == higher_value)(x, y), 0.0,
                    msg="Function2D equals callable (f1() == f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 != higher_value)(x, y), 1.0,
                    msg="Function2D not equals callable (f1() != f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 != ref_value)(x, y), 0.0,
                    msg="Function2D not equals callable (f1() != f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 < higher_value)(x, y), 1.0,
                    msg="Function2D less than callable (f1() < f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 < lower_value)(x, y), 0.0,
                    msg="Function2D less than callable (f1() < f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 > lower_value)(x, y), 1.0,
                    msg="Function2D greater than callable (f1() > f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 > higher_value)(x, y), 0.0,
                    msg="Function2D greater than callable (f1() > f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 <= higher_value)(x, y), 1.0,
                    msg="Function2D less equals callable (f1() <= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 <= ref_value)(x, y), 1.0,
                    msg="Function2D less equals callable (f1() <= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 <= lower_value)(x, y), 0.0,
                    msg="Function2D less equals callable (f1() <= f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 >= lower_value)(x, y), 1.0,
                    msg="Function2D equals callable (f1() >= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 >= ref_value)(x, y), 1.0,
                    msg="Function2D greater equals callable (f1() >= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 >= higher_value)(x, y), 0.0,
                    msg="Function2D equals callable (f1() >= f2()) did not return false when it should."
                )

    def test_richcmp_callable_function(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.ref1
                higher_value = lambda x, y: self.ref1(x, y) + abs(self.ref1(x, y)) + 1
                lower_value = lambda x, y: self.ref1(x, y) - abs(self.ref1(x, y)) - 1
                self.assertEqual(
                    (ref_value == self.f1)(x, y), 1.0,
                    msg="Callable equals Function2D (f1() == f2()) did not return true when it should."
                )
                self.assertEqual(
                    (higher_value == self.f1)(x, y), 0.0,
                    msg="Callable equals Function2D (f1() == f2()) did not return false when it should."
                )
                self.assertEqual(
                    (higher_value != self.f1)(x, y), 1.0,
                    msg="Callable not equals Function2D (f1() != f2()) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value != self.f1)(x, y), 0.0,
                    msg="Callable not equals Function2D (f1() != f2()) did not return false when it should."
                )
                self.assertEqual(
                    (lower_value < self.f1)(x, y), 1.0,
                    msg="Callable less than Function2D (f1() < f2()) did not return true when it should."
                )
                self.assertEqual(
                    (higher_value < self.f1)(x, y), 0.0,
                    msg="Callable less than Function2D (f1() < f2()) did not return false when it should."
                )
                self.assertEqual(
                    (higher_value > self.f1)(x, y), 1.0,
                    msg="Callable greater than Function2D (f1() > f2()) did not return true when it should."
                )
                self.assertEqual(
                    (lower_value > self.f1)(x, y), 0.0,
                    msg="Callable greater than Function2D (f1() > f2()) did not return false when it should."
                )
                self.assertEqual(
                    (lower_value <= self.f1)(x, y), 1.0,
                    msg="Callable less equals Function2D (f1() <= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value <= self.f1)(x, y), 1.0,
                    msg="Callable less equals Function2D (f1() <= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (higher_value <= self.f1)(x, y), 0.0,
                    msg="Callable less equals Function2D (f1() <= f2()) did not return false when it should."
                )
                self.assertEqual(
                    (higher_value >= self.f1)(x, y), 1.0,
                    msg="Callable equals Function2D (f1() >= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (ref_value >= self.f1)(x, y), 1.0,
                    msg="Callable greater equals Function2D (f1() >= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (lower_value >= self.f1)(x, y), 0.0,
                    msg="Callable equals Function2D (f1() >= f2()) did not return false when it should."
                )

    def test_richcmp_function_function(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                ref_value = self.f1
                higher_value = self.f1 + abs(self.f1) + 1
                lower_value = self.f1 - abs(self.f1) - 1
                self.assertEqual(
                    (self.f1 == ref_value)(x, y), 1.0,
                    msg="Function2D equals Function2D (f1() == f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 == higher_value)(x, y), 0.0,
                    msg="Function2D equals Function2D (f1() == f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 != higher_value)(x, y), 1.0,
                    msg="Function2D not equals Function2D (f1() != f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 != ref_value)(x, y), 0.0,
                    msg="Function2D not equals Function2D (f1() != f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 < higher_value)(x, y), 1.0,
                    msg="Function2D less than Function2D (f1() < f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 < lower_value)(x, y), 0.0,
                    msg="Function2D less than Function2D (f1() < f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 > lower_value)(x, y), 1.0,
                    msg="Function2D greater than Function2D (f1() > f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 > higher_value)(x, y), 0.0,
                    msg="Function2D greater than Function2D (f1() > f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 <= higher_value)(x, y), 1.0,
                    msg="Function2D less equals Function2D (f1() <= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 <= ref_value)(x, y), 1.0,
                    msg="Function2D less equals Function2D (f1() <= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 <= lower_value)(x, y), 0.0,
                    msg="Function2D less equals Function2D (f1() <= f2()) did not return false when it should."
                )
                self.assertEqual(
                    (self.f1 >= lower_value)(x, y), 1.0,
                    msg="Function2D equals Function2D (f1() >= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 >= ref_value)(x, y), 1.0,
                    msg="Function2D greater equals Function2D (f1() >= f2()) did not return true when it should."
                )
                self.assertEqual(
                    (self.f1 >= higher_value)(x, y), 0.0,
                    msg="Function2D equals Function2D (f1() >= f2()) did not return false when it should."
                )
