
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
Unit tests for the Interpolator1DCubic class from within Interpolate1D,
including interaction with Extrapolator1DLinear and Extrapolator1DNearest.
"""

import unittest
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, InterpType, ExtrapType
import numpy as np


class TestInterpolator1DCubic(unittest.TestCase):  # TODO: expand tests to cover the cython interface

    def test_interpolator_1d_cubic_at_a_minimum(self):
        """Tests that the 1D cubic spline around a minimum point. The interpolation at the known minimum should
        be lower than the spline points surrounding it """
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)

        interp_cubic_extrap_nearest = Interpolate1D(x_in, y_in, InterpType.CubicInt, ExtrapType.NearestExt,
                                                    extrapolation_range=2.0)

        # Test between spline points (minimum at pi/2)
        for i in range(len(x_in)):
            self.assertGreaterEqual(y_in[i], interp_cubic_extrap_nearest(-np.pi/2.))

    def test_nearest_neighbour_1d_extrapolation(self):
        """Tests for the nearest neighbour extrapolator returns the edge spline points in the lower and upper
        extrapolation range"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_cubic_extrap_nearest = Interpolate1D(x_in, y_in, InterpType.CubicInt, ExtrapType.NearestExt,
                                                    extrapolation_range=2.0)
        # Nearest neighbour extrapolation test
        self.assertEqual(y_in[0], interp_cubic_extrap_nearest(-1.8))
        self.assertEqual(y_in[-1], interp_cubic_extrap_nearest(-1.0))

    def test_cubic_1d_at_spline_points(self):
        """Tests that the cubic interpolator returns the value of the function at the spline points
        The value at the last spline point is returned from the linear extrapolator"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_cubic_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.CubicInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        interp_cubic_extrap_nearest = Interpolate1D(x_in, y_in, InterpType.CubicInt, ExtrapType.NearestExt,
                                                    extrapolation_range=2.0)
        # Interpolating at the spline points should return the exact value (the end spline point is an extrapolation)
        for i in range(len(x_in)):
            self.assertEqual(y_in[i], interp_cubic_extrap_linear(x_in[i]),
                             msg="Constant1D call did not match reference value.")
            self.assertEqual(y_in[i], interp_cubic_extrap_nearest(x_in[i]),
                             msg="Constant1D call did not match reference value.")

    def test_linear_1d_extrapolation_gradient(self):
        """Tests that the linear extrapolator calculates a similar gradient to expected"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_cubic_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.CubicInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        # Linear extrapolation test
        expected_start_grad = ((y_in[1]-y_in[0])/(x_in[1]-x_in[0]))
        expected_end_grad = ((y_in[-2]-y_in[-1])/(x_in[-2]-x_in[-1]))
        self.assertAlmostEqual(expected_start_grad, (y_in[0] - interp_cubic_extrap_linear(-1.8))/(x_in[0]--1.8))
        self.assertAlmostEqual(expected_end_grad, (interp_cubic_extrap_linear(-1.0)-y_in[-1])/(-1.0-x_in[-1]))

    def test_1d_extrapolation_range_out_of_bounds(self):
        """Tests for the linear and nearest neighbour extrapolator for inputs outside the extrapolation range,
        an error is raised"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_cubic_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.CubicInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )

        # Out of bounds test
        self.assertRaises(ValueError, interp_cubic_extrap_linear, -3.74)
        self.assertRaises(ValueError, interp_cubic_extrap_linear, 1.0)

        interp_cubic_extrap_nearest = Interpolate1D(x_in, y_in, InterpType.CubicInt, ExtrapType.NearestExt,
                                                    extrapolation_range=2.0)

        # Outside extrapolation range, there should be an error raised
        self.assertRaises(ValueError, interp_cubic_extrap_nearest, -3.74)
        self.assertRaises(ValueError, interp_cubic_extrap_nearest, 1.0)
