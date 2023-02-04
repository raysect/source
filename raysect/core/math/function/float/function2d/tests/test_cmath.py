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
Unit tests for the cmath wrapper classes.
"""

import math
import unittest
import raysect.core.math.function.float.function2d.cmath as cmath2d
from raysect.core.math.function.float.function2d.autowrap import PythonFunction2D

# TODO: expand tests to cover the cython interface
class TestCmath2D(unittest.TestCase):

    def setUp(self):
        self.f1 = PythonFunction2D(lambda x, y: x / 10 + y)
        self.f2 = PythonFunction2D(lambda x, y: x * x + y * y)

    def test_exp(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                function = cmath2d.Exp2D(self.f1)
                expected = math.exp(self.f1(x, y))
                self.assertEqual(function(x, y), expected, "Exp2D call did not match reference value.")

    def test_sin(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                function = cmath2d.Sin2D(self.f1)
                expected = math.sin(self.f1(x, y))
                self.assertEqual(function(x, y), expected, "Sin2D call did not match reference value.")

    def test_cos(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                function = cmath2d.Cos2D(self.f1)
                expected = math.cos(self.f1(x, y))
                self.assertEqual(function(x, y), expected, "Cos2D call did not match reference value.")

    def test_tan(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                function = cmath2d.Tan2D(self.f1)
                expected = math.tan(self.f1(x, y))
                self.assertEqual(function(x, y), expected, "Tan2D call did not match reference value.")

    def test_asin(self):
        v = [-10, -6, -2, -0.001, 0, 0.001, 2, 6, 10]
        function = cmath2d.Asin2D(self.f1)
        for x in v:
            expected = math.asin(self.f1(x, 0))
            self.assertEqual(function(x, 0), expected, "Asin2D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Asin2D did not raise a ValueError with value outside domain."):
            function(100, 0)

    def test_acos(self):
        v = [-10, -6, -2, -0.001, 0, 0.001, 2, 6, 10]
        function = cmath2d.Acos2D(self.f1)
        for x in v:
            expected = math.acos(self.f1(x, 0))
            self.assertEqual(function(x, 0), expected, "Acos2D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Acos2D did not raise a ValueError with value outside domain."):
            function(100, 0)

    def test_atan(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                function = cmath2d.Atan2D(self.f1)
                expected = math.atan(self.f1(x, y))
                self.assertEqual(function(x, y), expected, "Atan2D call did not match reference value.")

    def test_atan2(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                function = cmath2d.Atan4Q2D(self.f1, self.f2)
                expected = math.atan2(self.f1(x, y), self.f2(x, y))
                self.assertEqual(function(x, y), expected, "Atan4Q2D call did not match reference value.")

    def test_erf(self):
        v = [-1e5, -7, -0.001, 0.0, 0.00003, 10, 23.4, 1e5]
        function = cmath2d.Erf2D(self.f1)
        for x in v:
            for y in v:
                expected = math.erf(self.f1(x, y))
                self.assertAlmostEqual(function(x, y), expected, 10, "Erf2D call did not match reference value.")

    def test_sqrt(self):
        v = [0.0, 0.00003, 10, 23.4, 1e5]
        function = cmath2d.Sqrt2D(self.f1)
        for x in v:
            for y in v:
                expected = math.sqrt(self.f1(x, y))
                self.assertEqual(function(x, y), expected, "Sqrt2D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Sqrt2D did not raise a ValueError with value outside domain."):
            function(-0.1, -0.1)
