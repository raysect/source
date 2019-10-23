# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
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
from raysect.core.math.function.function2d.base import PythonFunction2D

# TODO: expand tests to cover the cython interface
class TestFunction2D(unittest.TestCase):

    def setUp(self):

        self.f1 = PythonFunction2D(lambda x, y: 10*x + 5*y)
        self.f2 = PythonFunction2D(lambda x, y: x*x + y*y)

    def f1_ref(self, x, y):
        return 10*x + 5*y

    def f2_ref(self, x, y):
        return x*x + y*y

    def test_call(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        for x in v:
            for y in v:
                self.assertEqual(self.f1(x, y), self.f1_ref(x, y), "Function2D call did not match reference function value.")

    def test_negate(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = -self.f1
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), -self.f1_ref(x, y), "Function2D negate did not match reference function value.")

    def test_add_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 + self.f1
        r2 = self.f1 + 65
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 8 + self.f1_ref(x, y), "Function2D add scalar (K + f()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.f1_ref(x, y) + 65, "Function2D add scalar (f() + K) did not match reference function value.")

    def test_sub_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 8 - self.f1
        r2 = self.f1 - 65
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 8 - self.f1_ref(x, y), "Function2D subtract scalar (K - f()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.f1_ref(x, y) - 65, "Function2D subtract scalar (f() - K) did not match reference function value.")

    def test_mul_scalar(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r1 = 5 * self.f1
        r2 = self.f1 * -7.8
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 5 * self.f1_ref(x, y), "Function2D multiply scalar (K * f()) did not match reference function value.")
                self.assertEqual(r2(x, y), self.f1_ref(x, y) * -7.8, "Function2D multiply scalar (f() * K) did not match reference function value.")

    def test_div_scalar(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r1 = 5 / self.f1
        r2 = self.f1 / -7.8
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), 5 / self.f1_ref(x, y), "Function2D divide scalar (K / f()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y), self.f1_ref(x, y) / -7.8, delta=abs(r2(x, y)) * 1e-12, msg="Function2D divide scalar (f() / K) did not match reference function value.")

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
                self.assertAlmostEqual(r1(x, y), math.fmod(5, self.f1_ref(x, y)), 15, "Function2D modulo scalar (K % f()) did not match reference function value.")
                self.assertAlmostEqual(r2(x, y), math.fmod(self.f1_ref(x, y), -7.8), 15, "Function2D modulo scalar (f() % K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns 0"):
            r1(0, 0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when modulo scalar is 0"):
            self.f1 % 0

    def test_pow_function2d_scalar(self):
        v = [-10, -7, -0.001, 0.00003, 10, 12.3]
        r1 = 5 ** self.f1
        r2 = self.f1 ** -7.8
        r3 = (-5) ** self.f1
        for x in v:
            for y in v:
                self.assertAlmostEqual(r1(x, y), 5 ** self.f1_ref(x, y), 15, "Function2D power scalar (K ** f()) did not match reference function value.")
                if self.f1_ref(x, y) < 0:
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral"):
                        r2(x, y)
                elif not float(self.f1_ref(x, y)).is_integer():
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral"):
                        r3(x, y)
                else:
                    self.assertAlmostEqual(r2(x, y), self.f1_ref(x, y) ** -7.8, 15, "Function2D power scalar (f() ** K) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative"):
            r2(0, 0)
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when base is 0 and exponent negative"):
            r4 = 0 ** self.f1
            r4(-1, 0)

    def test_add_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 + self.f2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.f1_ref(x, y) + self.f2_ref(x, y), "Function2D add function (f1() + f2()) did not match reference function value.")

    def test_sub_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 - self.f2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.f1_ref(x, y) - self.f2_ref(x, y), "Function2D subtract function (f1() - f2()) did not match reference function value.")

    def test_mul_function2d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 * self.f2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.f1_ref(x, y) * self.f2_ref(x, y), "Function2D multiply function (f1() * f2()) did not match reference function value.")

    def test_div_function2d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r = self.f1 / self.f2
        for x in v:
            for y in v:
                self.assertAlmostEqual(r(x, y), self.f1_ref(x, y) / self.f2_ref(x, y), delta=abs(r(x, y)) * 1e-12, msg="Function2D divide function (f1() / f2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r(0, 0)

    def test_mod_function2d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e9]
        r = self.f1 % self.f2
        for x in v:
            for y in v:
                self.assertAlmostEqual(r(x, y), math.fmod(self.f1_ref(x, y), self.f2_ref(x, y)), delta=abs(r(x, y)) * 1e-12, msg="Function2D modulo function (f1() % f2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r(0, 0)

    def test_pow_function2d_function2d(self):
        v = [-3.0, -0.7, -0.001, 0.00003, 2]
        r = self.f1 ** self.f2
        for x in v:
            for y in v:
                if self.f1_ref(x, y) < 0 and not float(self.f2_ref(x, y)).is_integer():
                    with self.assertRaises(ValueError, msg="ValueError not raised when base is negative and exponent non-integral"):
                        r(x, y)
                else:
                    self.assertAlmostEqual(r(x, y), self.f1_ref(x, y) ** self.f2_ref(x, y), 15, "Function2D power function (f1() ** f2()) did not match reference function value.")
        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when f1() == 0 and f2() is negative"):
            r = PythonFunction2D(lambda x, y: 0) ** self.f1
            r(-1, 0)

    def test_pow_3_arguments(self):
        v = [-10, -7, -0.001, 0.00003, 2]
        r1 = pow(self.f1, 5, 3)
        r2 = pow(5, self.f1, 3)
        r3 = pow(5, self.f1, self.f2)
        # Can't use 3 argument pow() if all arguments aren't integers, so
        # use fmod(a, b) % c instead
        for x in v:
            for y in v:
                self.assertEqual(r1(x, y), math.fmod(self.f1_ref(x, y) ** 5, 3), "Function2D 3 argument pow(f1(), A, B) did not match reference value")
                self.assertEqual(r2(x, y), math.fmod(5 ** self.f1_ref(x, y), 3), "Function2D 3 argument pow(A, f1(), B) did not match reference value")
                self.assertEqual(r3(x, y), math.fmod(5 ** self.f1_ref(x, y), self.f2_ref(x, y)), "Function2D 3 argument pow(A, f1(), f2()) did not match reference value")
