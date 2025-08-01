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
Unit tests for the autowrap_2d function
"""

import unittest
from raysect.core.math import Vector3D
from raysect.core.math.function.vector3d.function2d.autowrap import _autowrap_function2d
from raysect.core.math.function.vector3d.function2d.autowrap import PythonFunction2D
from raysect.core.math.function.vector3d.function2d.constant import Constant2D

class TestAutowrap2D(unittest.TestCase):

    def test_constant_vector(self):
        function = _autowrap_function2d(Vector3D(3.0, 4.0, 5.0))
        self.assertIsInstance(function, Constant2D,
                              "Autowrapped Vector3D is not a vector3d.Constant2D.")

    def test_constant_iterable(self):
        function = _autowrap_function2d([3, 4, 5.0])
        self.assertIsInstance(function, Constant2D,
                              "Autowrapped iterable is not a vector3d.Constant2D.")

    def test_python_function(self):
        function = _autowrap_function2d(lambda x, y: Vector3D(x, y, x + y))
        self.assertIsInstance(function, PythonFunction2D,
                              "Autowrapped function is not a vector3d.PythonFunction2D.")
