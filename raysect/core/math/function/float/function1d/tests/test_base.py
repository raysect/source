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
Unit tests for the Function1D class.
"""

import math
import unittest
from raysect.core.math.function.float.function1d.autowrap import PythonFunction1D

# TODO: expand tests to cover the cython interface
class TestFunction1D(unittest.TestCase):

    def setUp(self):

        self.ref1 = lambda x: 10 * x
        self.ref2 = lambda x: x * x

        self.f1 = PythonFunction1D(self.ref1)
        self.f2 = PythonFunction1D(self.ref2)

    def test_call(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            self.assertEqual(self.f1(x), self.ref1(x), "Function1D call did not match reference function value.")

    def test_negate(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = -self.f1
        for x in v:
            self.assertEqual(r(x), -self.ref1(x), "Function1D negate did not match reference function value.")

    def test_add_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 + self.f1
        r2 = self.f1 + 65
        for x in v:
            self.assertEqual(r1(x), 8 + self.ref1(x), "Function1D add scalar (K + f()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) + 65, "Function1D add scalar (f() + K) did not match reference function value.")

    def test_sub_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 - self.f1
        r2 = self.f1 - 65
        for x in v:
            self.assertEqual(r1(x), 8 - self.ref1(x), "Function1D subtract scalar (K - f()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) - 65, "Function1D subtract scalar (f() - K) did not match reference function value.")

    def test_mul_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 5 * self.f1
        r2 = self.f1 * -7.8
        for x in v:
            self.assertEqual(r1(x), 5 * self.ref1(x), "Function1D multiply scalar (K * f()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) * -7.8, "Function1D multiply scalar (f() * K) did not match reference function value.")

    def test_div_scalar(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = 5 / self.f1
        r2 = self.f1 / -7.8
        for x in v:
            self.assertEqual(r1(x), 5 / self.ref1(x), "Function1D divide scalar (K / f()) did not match reference function value.")
            self.assertAlmostEqual(r2(x), self.ref1(x) / -7.8, delta=abs(r2(x)) * 1e-12, msg="Function1D divide scalar (f() / K) did not match reference function value.")

        r = 5 / self.f1
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r(0)

    def test_mod_function1d_scalar(self):
        # Note that Function1D objects work with doubles, so the floating modulo
        # operator is used rather than the integer one. For accurate testing we
        # therefore need to use the math.fmod operator rather than % in Python.
        v = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5 % self.f1
        r2 = self.f1 % -7.8
        for x in v:
            self.assertAlmostEqual(r1(x), math.fmod(5, self.ref1(x)), 15, "Function1D modulo scalar (K % f()) did not match reference function value.")
            self.assertAlmostEqual(r2(x), math.fmod(self.ref1(x), -7.8), 15, "Function1D modulo scalar (f() % K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns 0."):
            r1(0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when modulo scalar is 0."):
            self.f1 % 0

    def test_pow_function1d_scalar(self):
        v = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5 ** self.f1
        r2 = self.f1 ** -7.8
        r3 = (-5) ** self.f1
        for x in v:
            self.assertAlmostEqual(r1(x), 5 ** self.ref1(x), 15, "Function1D power scalar (K ** f()) did not match reference function value.")
            if self.ref1(x) < 0:
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral."):
                    r2(x)
            elif not float(self.ref1(x)).is_integer():
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral."):
                    r3(x)
            else:
                self.assertAlmostEqual(r2(x), self.ref1(x) ** -7.8, 15, "Function1D power scalar (f() ** K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative."):
            r2(0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative."):
            r4 = 0 ** self.f1
            r4(-1)

    def test_richcmp_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.ref1(x)
            higher_value = ref_value + abs(ref_value) + 1
            lower_value = ref_value - abs(ref_value) - 1
            self.assertEqual(
                (self.f1 == ref_value)(x), 1.0,
                msg="Function1D equals scalar (f() == K) did not return true when it should."
            )
            self.assertEqual(
                (ref_value == self.f1)(x), 1.0,
                msg="Scalar equals Function1D (K == f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 == higher_value)(x), 0.0,
                msg="Function1D equals scalar (f() == K) did not return false when it should."
            )
            self.assertEqual(
                (higher_value == self.f1)(x), 0.0,
                msg="Scalar equals Function1D (K == f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 != higher_value)(x), 1.0,
                msg="Function1D not equals scalar (f() != K) did not return true when it should."
            )
            self.assertEqual(
                (higher_value != self.f1)(x), 1.0,
                msg="Scalar not equals Function1D (K != f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 != ref_value)(x), 0.0,
                msg="Function1D not equals scalar (f() != K) did not return false when it should."
            )
            self.assertEqual(
                (ref_value != self.f1)(x), 0.0,
                msg="Scalar not equals Function1D (K != f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 < higher_value)(x), 1.0,
                msg="Function1D less than scalar (f() < K) did not return true when it should."
            )
            self.assertEqual(
                (lower_value < self.f1)(x), 1.0,
                msg="Scalar less than Function1D (K < f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 < lower_value)(x), 0.0,
                msg="Function1D less than scalar (f() < K) did not return false when it should."
            )
            self.assertEqual(
                (higher_value < self.f1)(x), 0.0,
                msg="Scalar less than Function1D (K < f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 > lower_value)(x), 1.0,
                msg="Function1D greater than scalar (f() > K) did not return true when it should."
            )
            self.assertEqual(
                (higher_value > self.f1)(x), 1.0,
                msg="Scalar greater than Function1D (K > f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 > higher_value)(x), 0.0,
                msg="Function1D greater than scalar (f() > K) did not return false when it should."
            )
            self.assertEqual(
                (lower_value > self.f1)(x), 0.0,
                msg="Scalar greater than Function1D (K > f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 <= higher_value)(x), 1.0,
                msg="Function1D less equals scalar (f() <= K) did not return true when it should."
            )
            self.assertEqual(
                (lower_value <= self.f1)(x), 1.0,
                msg="Scalar less equals Function1D (K <= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= ref_value)(x), 1.0,
                msg="Function1D less equals scalar (f() <= K) did not return true when it should."
            )
            self.assertEqual(
                (ref_value <= self.f1)(x), 1.0,
                msg="Scalar less equals Function1D (K <= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= lower_value)(x), 0.0,
                msg="Function1D less equals scalar (f() <= K) did not return false when it should."
            )
            self.assertEqual(
                (higher_value <= self.f1)(x), 0.0,
                msg="Scalar less equals Function1D (K <= f()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 >= lower_value)(x), 1.0,
                msg="Function1D greater equals scalar (f() >= K) did not return true when it should."
            )
            self.assertEqual(
                (higher_value >= self.f1)(x), 1.0,
                msg="Scalar greater equals Function1D (K >= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= ref_value)(x), 1.0,
                msg="Function1D greater equals scalar (f() >= K) did not return true when it should."
            )
            self.assertEqual(
                (ref_value >= self.f1)(x), 1.0,
                msg="Scalar greater equals Function1D (K >= f()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= higher_value)(x), 0.0,
                msg="Function1D greater equals scalar (f() >= K) did not return false when it should."
            )
            self.assertEqual(
                (lower_value >= self.f1)(x), 0.0,
                msg="Scalar greater equals Function1D (K >= f()) did not return false when it should."
            )

    def test_add_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 + self.f2
        r2 = self.ref1 + self.f2
        r3 = self.f1 + self.ref2
        for x in v:
            self.assertEqual(r1(x), self.ref1(x) + self.ref2(x), "Function1D add function (f1() + f2()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) + self.ref2(x), "Function1D add function (p1() + f2()) did not match reference function value.")
            self.assertEqual(r3(x), self.ref1(x) + self.ref2(x), "Function1D add function (f1() + p2()) did not match reference function value.")

    def test_sub_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 - self.f2
        r2 = self.ref1 - self.f2
        r3 = self.f1 - self.ref2
        for x in v:
            self.assertEqual(r1(x), self.ref1(x) - self.ref2(x), "Function1D subtract function (f1() - f2()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) - self.ref2(x), "Function1D subtract function (p1() - f2()) did not match reference function value.")
            self.assertEqual(r3(x), self.ref1(x) - self.ref2(x), "Function1D subtract function (f1() - p2()) did not match reference function value.")

    def test_mul_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = self.f1 * self.f2
        r2 = self.ref1 * self.f2
        r3 = self.f1 * self.ref2
        for x in v:
            self.assertEqual(r1(x), self.ref1(x) * self.ref2(x), "Function1D multiply function (f1() * f2()) did not match reference function value.")
            self.assertEqual(r2(x), self.ref1(x) * self.ref2(x), "Function1D multiply function (p1() * f2()) did not match reference function value.")
            self.assertEqual(r3(x), self.ref1(x) * self.ref2(x), "Function1D multiply function (f1() * p2()) did not match reference function value.")

    def test_div_function1d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = self.f1 / self.f2
        r2 = self.ref1 / self.f2
        r3 = self.f1 / self.ref2
        for x in v:
            self.assertAlmostEqual(r1(x), self.ref1(x) / self.ref2(x), delta=abs(r1(x)) * 1e-12, msg="Function1D divide function (f1() / f2()) did not match reference function value.")
            self.assertAlmostEqual(r2(x), self.ref1(x) / self.ref2(x), delta=abs(r2(x)) * 1e-12, msg="Function1D divide function (p1() / f2()) did not match reference function value.")
            self.assertAlmostEqual(r3(x), self.ref1(x) / self.ref2(x), delta=abs(r3(x)) * 1e-12, msg="Function1D divide function (f1() / p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0)

    def test_mod_function1d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e9]
        r1 = self.f1 % self.f2
        r2 = self.ref1 % self.f2
        r3 = self.f1 % self.ref2
        for x in v:
            self.assertAlmostEqual(r1(x), math.fmod(self.ref1(x), self.ref2(x)), delta=abs(r1(x)) * 1e-12, msg="Function1D modulo function (f1() % f2()) did not match reference function value.")
            self.assertAlmostEqual(r2(x), math.fmod(self.ref1(x), self.ref2(x)), delta=abs(r2(x)) * 1e-12, msg="Function1D modulo function (p1() % f2()) did not match reference function value.")
            self.assertAlmostEqual(r3(x), math.fmod(self.ref1(x), self.ref2(x)), delta=abs(r3(x)) * 1e-12, msg="Function1D modulo function (f1() % p2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r1(0)

    def test_pow_function1d_function1d(self):
        v = [-10, -7, -0.001, 0.00003, 2]
        r1 = self.f1 ** self.f2
        r2 = self.ref1 ** self.f2
        r3 = self.f1 ** self.ref2
        for x in v:
            if self.ref1(x) < 0 and not float(self.ref2(x)).is_integer():
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (1/3)."):
                    r1(x)
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (2/3)."):
                    r2(x)
                with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral (3/3)."):
                    r3(x)
            else:
                self.assertAlmostEqual(r1(x), self.ref1(x) ** self.ref2(x), 15, "Function1D power function (f1() ** f2()) did not match reference function value.")
                self.assertAlmostEqual(r2(x), self.ref1(x) ** self.ref2(x), 15, "Function1D power function (p1() ** f2()) did not match reference function value.")
                self.assertAlmostEqual(r3(x), self.ref1(x) ** self.ref2(x), 15, "Function1D power function (f1() ** p2()) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when f1() == 0 and f2() is negative."):
            r4 = PythonFunction1D(lambda x: 0) ** self.f1
            r4(-1)

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
            self.assertEqual(r1(x), math.fmod(self.ref1(x) ** 5, 3), "Function1D 3 argument pow(f1(), A, B) did not match reference value.")
            self.assertEqual(r2(x), math.fmod(5 ** self.ref1(x), 3), "Function1D 3 argument pow(A, f1(), B) did not match reference value.")
            self.assertEqual(r3(x), math.fmod(5 ** self.ref1(x), self.ref2(x)), "Function1D 3 argument pow(A, f1(), f2()) did not match reference value.")
            self.assertEqual(r4(x), math.fmod(self.ref2(x) ** self.ref1(x), self.ref2(x)), "Function1D 3 argument pow(f2(), f1(), f2()) did not match reference value.")
            self.assertEqual(r5(x), math.fmod(self.ref2(x) ** self.ref1(x), self.ref2(x)), "Function1D 3 argument pow(f2(), p1(), p2()) did not match reference value.")
            self.assertEqual(r6(x), math.fmod(self.ref2(x) ** self.ref1(x), self.ref2(x)), "Function1D 3 argument pow(p2(), f1(), f2()) did not match reference value.")

    def test_abs(self):
        v = [-1e10, -7, -0.001, 0.0, 0.0003, 10, 2.3e49]
        for x in v:
            self.assertEqual(abs(self.f1)(x), abs(self.ref1(x)),
                             msg="abs(Function1D) did not match reference value")

    def test_richcmp_function_callable(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.ref1
            higher_value = lambda x: self.ref1(x) + abs(self.ref1(x)) + 1
            lower_value = lambda x: self.ref1(x) - abs(self.ref1(x)) - 1
            self.assertEqual(
                (self.f1 == ref_value)(x), 1.0,
                msg="Function1D equals callable (f1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 == higher_value)(x), 0.0,
                msg="Function1D equals callable (f1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 != higher_value)(x), 1.0,
                msg="Function1D not equals callable (f1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 != ref_value)(x), 0.0,
                msg="Function1D not equals callable (f1() != f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 < higher_value)(x), 1.0,
                msg="Function1D less than callable (f1() < f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 < lower_value)(x), 0.0,
                msg="Function1D less than callable (f1() < f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 > lower_value)(x), 1.0,
                msg="Function1D greater than callable (f1() > f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 > higher_value)(x), 0.0,
                msg="Function1D greater than callable (f1() > f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 <= higher_value)(x), 1.0,
                msg="Function1D less equals callable (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= ref_value)(x), 1.0,
                msg="Function1D less equals callable (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= lower_value)(x), 0.0,
                msg="Function1D less equals callable (f1() <= f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 >= lower_value)(x), 1.0,
                msg="Function1D equals callable (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= ref_value)(x), 1.0,
                msg="Function1D greater equals callable (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= higher_value)(x), 0.0,
                msg="Function1D equals callable (f1() >= f2()) did not return false when it should."
            )

    def test_richcmp_callable_function(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.ref1
            higher_value = lambda x: self.ref1(x) + abs(self.ref1(x)) + 1
            lower_value = lambda x: self.ref1(x) - abs(self.ref1(x)) - 1
            self.assertEqual(
                (ref_value == self.f1)(x), 1.0,
                msg="Callable equals Function1D (f1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (higher_value == self.f1)(x), 0.0,
                msg="Callable equals Function1D (f1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (higher_value != self.f1)(x), 1.0,
                msg="Callable not equals Function1D (f1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value != self.f1)(x), 0.0,
                msg="Callable not equals Function1D (f1() != f2()) did not return false when it should."
            )
            self.assertEqual(
                (lower_value < self.f1)(x), 1.0,
                msg="Callable less than Function1D (f1() < f2()) did not return true when it should."
            )
            self.assertEqual(
                (higher_value < self.f1)(x), 0.0,
                msg="Callable less than Function1D (f1() < f2()) did not return false when it should."
            )
            self.assertEqual(
                (higher_value > self.f1)(x), 1.0,
                msg="Callable greater than Function1D (f1() > f2()) did not return true when it should."
            )
            self.assertEqual(
                (lower_value > self.f1)(x), 0.0,
                msg="Callable greater than Function1D (f1() > f2()) did not return false when it should."
            )
            self.assertEqual(
                (lower_value <= self.f1)(x), 1.0,
                msg="Callable less equals Function1D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value <= self.f1)(x), 1.0,
                msg="Callable less equals Function1D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (higher_value <= self.f1)(x), 0.0,
                msg="Callable less equals Function1D (f1() <= f2()) did not return false when it should."
            )
            self.assertEqual(
                (higher_value >= self.f1)(x), 1.0,
                msg="Callable equals Function1D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (ref_value >= self.f1)(x), 1.0,
                msg="Callable greater equals Function1D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (lower_value >= self.f1)(x), 0.0,
                msg="Callable equals Function1D (f1() >= f2()) did not return false when it should."
            )

    def test_richcmp_function_function(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            ref_value = self.f1
            higher_value = self.f1 + abs(self.f1) + 1
            lower_value = self.f1 - abs(self.f1) - 1
            self.assertEqual(
                (self.f1 == ref_value)(x), 1.0,
                msg="Function1D equals Function1D (f1() == f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 == higher_value)(x), 0.0,
                msg="Function1D equals Function1D (f1() == f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 != higher_value)(x), 1.0,
                msg="Function1D not equals Function1D (f1() != f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 != ref_value)(x), 0.0,
                msg="Function1D not equals Function1D (f1() != f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 < higher_value)(x), 1.0,
                msg="Function1D less than Function1D (f1() < f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 < lower_value)(x), 0.0,
                msg="Function1D less than Function1D (f1() < f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 > lower_value)(x), 1.0,
                msg="Function1D greater than Function1D (f1() > f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 > higher_value)(x), 0.0,
                msg="Function1D greater than Function1D (f1() > f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 <= higher_value)(x), 1.0,
                msg="Function1D less equals Function1D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= ref_value)(x), 1.0,
                msg="Function1D less equals Function1D (f1() <= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 <= lower_value)(x), 0.0,
                msg="Function1D less equals Function1D (f1() <= f2()) did not return false when it should."
            )
            self.assertEqual(
                (self.f1 >= lower_value)(x), 1.0,
                msg="Function1D equals Function1D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= ref_value)(x), 1.0,
                msg="Function1D greater equals Function1D (f1() >= f2()) did not return true when it should."
            )
            self.assertEqual(
                (self.f1 >= higher_value)(x), 0.0,
                msg="Function1D equals Function1D (f1() >= f2()) did not return false when it should."
            )
