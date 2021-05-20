
# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
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
Unit tests for the Constant1D class.
"""

import unittest
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, InterpType, ExtrapType
import numpy as np


class TestConstant1D(unittest.TestCase):  # TODO: expand tests to cover the cython interface

    def test_interpolator_1d_cubic(self):
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)

        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )

        # Test between spline points (midpoint gradient comparison)
        for i in range(len(x_in) - 1):
            expected_grad = ((y_in[i+1] - y_in[i]) / (x_in[i+1] - x_in[i]))
            midpoint = (x_in[i+1] - x_in[i])/2. + x_in[i]
            self.assertAlmostEqual(expected_grad, (interp_linear_extrap_nearest(midpoint) - y_in[i]) / (midpoint - x_in[i]))
        # Interpolating at the spline points should return the exact value (the end spline point is an extrapolation)
        for i in range(len(x_in)):
            self.assertEqual(y_in[i], interp_linear_extrap_nearest(x_in[i]),
                             msg="Constant1D call did not match reference value.")
        # Nearest neighbour extrapolation test
        self.assertEqual(y_in[0], interp_linear_extrap_nearest(-1.8))
        self.assertEqual(y_in[-1], interp_linear_extrap_nearest(-1.0))
        # Outside extrapolation range, there should be an error raised
        self.assertRaises(ValueError, interp_linear_extrap_nearest, -3.74)
        self.assertRaises(ValueError, interp_linear_extrap_nearest, 1.0)

        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )

        # Interpolating at the spline points should return the exact value (the end spline point is an extrapolation)
        for i in range(len(x_in)):
            self.assertEqual(y_in[i], interp_linear_extrap_linear(x_in[i]),
                             msg="Constant1D call did not match reference value.")

        # Linear extrapolation test
        expected_start_grad = ((y_in[1]-y_in[0])/(x_in[1]-x_in[0]))
        expected_end_grad = ((y_in[-2]-y_in[-1])/(x_in[-2]-x_in[-1]))
        self.assertAlmostEqual(expected_start_grad, (y_in[0] - interp_linear_extrap_linear(-1.8))/(x_in[0]--1.8))
        self.assertAlmostEqual(expected_end_grad, (interp_linear_extrap_linear(-1.0)-y_in[-1])/(-1.0-x_in[-1]))

        # Outside extrapolation range, there should be an error raised
        self.assertRaises(ValueError, interp_linear_extrap_linear, -3.74)
        self.assertRaises(ValueError, interp_linear_extrap_linear, 1.0)
