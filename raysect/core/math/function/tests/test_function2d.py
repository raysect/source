# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

import unittest
from raysect.core.math.function.function2d import PythonFunction2D

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

    def test_add_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 + self.f2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.f1_ref(x, y) + self.f2_ref(x, y), "Function2D add function (f1() + f2()) did not match reference function value.")

    def test_sub_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 - self.f2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.f1_ref(x, y) - self.f2_ref(x, y), "Function2D subtract function (f1() - f2()) did not match reference function value.")

    def test_mul_function1d(self):
        v = [-1e10, -7, -0.001, 0.0, 0.00003, 10, 2.3e49]
        r = self.f1 * self.f2
        for x in v:
            for y in v:
                self.assertEqual(r(x, y), self.f1_ref(x, y) * self.f2_ref(x, y), "Function2D multiply function (f1() * f2()) did not match reference function value.")

    def test_div_function1d(self):
        v = [-1e10, -7, -0.001, 0.00003, 10, 2.3e49]
        r = self.f1 / self.f2
        for x in v:
            for y in v:
                self.assertAlmostEqual(r(x, y), self.f1_ref(x, y) / self.f2_ref(x, y), delta=abs(r(x, y)) * 1e-12, msg="Function2D divide function (f1() / f2()) did not match reference function value.")

        with self.assertRaises(ZeroDivisionError, msg="ZeroDivisionError not raised when function returns zero."):
            r(0, 0)

