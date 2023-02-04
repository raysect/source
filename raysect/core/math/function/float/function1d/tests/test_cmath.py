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
import raysect.core.math.function.float.function1d.cmath as cmath1d
from raysect.core.math.function.float.function1d.autowrap import PythonFunction1D

# TODO: expand tests to cover the cython interface
class TestCmath1D(unittest.TestCase):

    def setUp(self):
        self.f1 = PythonFunction1D(lambda x: x / 10)
        self.f2 = PythonFunction1D(lambda x: x * x)

    def test_exp(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        function = cmath1d.Exp1D(self.f1)
        for x in v:
            expected = math.exp(self.f1(x))
            self.assertEqual(function(x), expected, "Exp1D call did not match reference value.")

    def test_sin(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        function = cmath1d.Sin1D(self.f1)
        for x in v:
            expected = math.sin(self.f1(x))
            self.assertEqual(function(x), expected, "Sin1D call did not match reference value.")

    def test_cos(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        function = cmath1d.Cos1D(self.f1)
        for x in v:
            expected = math.cos(self.f1(x))
            self.assertEqual(function(x), expected, "Cos1D call did not match reference value.")

    def test_tan(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        function = cmath1d.Tan1D(self.f1)
        for x in v:
            expected = math.tan(self.f1(x))
            self.assertEqual(function(x), expected, "Tan1D call did not match reference value.")

    def test_asin(self):
        v = [-10, -6, -2, -0.001, 0, 0.001, 2, 6, 10]
        function = cmath1d.Asin1D(self.f1)
        for x in v:
            expected = math.asin(self.f1(x))
            self.assertEqual(function(x), expected, "Asin1D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Asin1D did not raise a ValueError with value outside domain."):
            function(100)

    def test_acos(self):
        v = [-10, -6, -2, -0.001, 0, 0.001, 2, 6, 10]
        function = cmath1d.Acos1D(self.f1)
        for x in v:
            expected = math.acos(self.f1(x))
            self.assertEqual(function(x), expected, "Acos1D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Acos1D did not raise a ValueError with value outside domain."):
            function(100)

    def test_atan(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        function = cmath1d.Atan1D(self.f1)
        for x in v:
            expected = math.atan(self.f1(x))
            self.assertEqual(function(x), expected, "Atan1D call did not match reference value.")

    def test_atan2(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        function = cmath1d.Atan4Q1D(self.f1, self.f2)
        for x in v:
            expected = math.atan2(self.f1(x), self.f2(x))
            self.assertEqual(function(x), expected, "Atan4Q1D call did not match reference value.")

    def test_erf(self):
        v = [-1e5, -7, -0.001, 0.0, 0.00003, 10, 23.4, 1e5]
        function = cmath1d.Erf1D(self.f1)
        for x in v:
            expected = math.erf(self.f1(x))
            self.assertAlmostEqual(function(x), expected, 10, "Erf1D call did not match reference value.")

    def test_sqrt(self):
        v = [0.0, 0.00003, 10, 23.4, 1e5]
        function = cmath1d.Sqrt1D(self.f1)
        for x in v:
            expected = math.sqrt(self.f1(x))
            self.assertEqual(function(x), expected, "Sqrt1D call did not match reference value.")

        with self.assertRaises(ValueError, msg="Sqrt1D did not raise a ValueError with value outside domain."):
            function(-0.1)
