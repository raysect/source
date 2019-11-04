# Copyright (c) 2014-2019, Dr Alex Meakins, Raysect Project
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
import raysect.core.math.function.function3d.cmath as cmath3d
from raysect.core.math.function.function3d.autowrap import PythonFunction3D

# TODO: expand tests to cover the cython interface
class TestCmath3D(unittest.TestCase):

    def setUp(self):
        self.f1 = PythonFunction3D(lambda x, y, z: x / 10 + y - z)
        self.f2 = PythonFunction3D(lambda x, y, z: x * x + y * y - z * z)

    def test_exp(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                for z in v:
                    function = cmath3d.Exp3D(self.f1)
                    expected = math.exp(self.f1(x, y, z))
                    self.assertEqual(function(x, y, z), expected, "Exp3D call did not match reference value")

    def test_sin(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                for z in v:
                    function = cmath3d.Sin3D(self.f1)
                    expected = math.sin(self.f1(x, y, z))
                    self.assertEqual(function(x, y, z), expected, "Sin3D call did not match reference value")

    def test_cos(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                for z in v:
                    function = cmath3d.Cos3D(self.f1)
                    expected = math.cos(self.f1(x, y, z))
                    self.assertEqual(function(x, y, z), expected, "Cos3D call did not match reference value")

    def test_atan(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                for z in v:
                    function = cmath3d.Atan3D(self.f1)
                    expected = math.atan(self.f1(x, y, z))
                    self.assertEqual(function(x, y, z), expected, "Atan3D call did not match reference value")

    def test_atan2(self):
        v = [-10.0, -7, -0.001, 0.0, 0.00003, 10, 23.4]
        for x in v:
            for y in v:
                for z in v:
                    function = cmath3d.Atan4Q3D(self.f1, self.f2)
                    expected = math.atan2(self.f1(x, y, z), self.f2(x, y, z))
                    self.assertEqual(function(x, y, z), expected, "Atan4Q3D call did not match reference value")
