
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
Unit tests for the Interpolator1DLinear class from within Interpolate1D,
including interaction with Extrapolator1DLinear and Extrapolator1DNearest."""

import unittest
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, InterpType, ExtrapType
import numpy as np


class TestInterpolator1DLinear(unittest.TestCase):  # TODO: expand tests to cover the cython interface

    def test_interpolator_1d_linear(self):
        """
        Tests that the 1D linear spline interpolated at the midpoint between each spline
        point are close to the returned value
        """
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

    def test_nearest_neighbour_1d_extrapolation(self):
        """
        Tests for the nearest neighbour extrapolator returns the edge spline points in the lower and upper
        extrapolation range
        """
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)

        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        # Nearest neighbour extrapolation test
        self.assertEqual(y_in[0], interp_linear_extrap_nearest(-1.8))
        self.assertEqual(interp_linear_extrap_nearest(-1.75), interp_linear_extrap_nearest(-1.8))
        self.assertEqual(y_in[-1], interp_linear_extrap_nearest(-1.0))
        self.assertEqual(interp_linear_extrap_nearest(0.0), interp_linear_extrap_nearest(-1.0))


    def test_linear_1d_at_spline_points(self):
        """
        Tests that the linear interpolator returns the value of the function at the spline points
        The value at the last spline point is returned from the linear extrapolator
        """
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        # Interpolating at the spline points should return the exact value (the end spline point is an extrapolation)
        for i in range(len(x_in)):
            self.assertEqual(y_in[i], interp_linear_extrap_linear(x_in[i]),
                             msg="Constant1D call did not match reference value.")
            self.assertEqual(y_in[i], interp_linear_extrap_nearest(x_in[i]),
                             msg="Constant1D call did not match reference value.")

    def test_linear_1d_extrapolation_gradient(self):
        """Tests that the linear extrapolator calculates a similar gradient to expected"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        # Linear extrapolation test
        expected_start_grad = ((y_in[1]-y_in[0])/(x_in[1]-x_in[0]))
        expected_end_grad = ((y_in[-2]-y_in[-1])/(x_in[-2]-x_in[-1]))
        self.assertAlmostEqual(expected_start_grad, (y_in[0] - interp_linear_extrap_linear(-1.8))/(x_in[0]--1.8))
        self.assertAlmostEqual(expected_end_grad, (interp_linear_extrap_linear(-1.0)-y_in[-1])/(-1.0-x_in[-1]))

    def test_1d_extrapolation_range_out_of_bounds(self):
        """
        Tests for the linear and nearest neighbour extrapolator for inputs outside the extrapolation range,
        an error is raised
        """
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        # Outside extrapolation range, there should be an error raised
        self.assertRaises(ValueError, interp_linear_extrap_linear, -3.74)
        self.assertRaises(ValueError, interp_linear_extrap_linear, 1.0)

        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        # Outside extrapolation range, there should be an error raised
        self.assertRaises(ValueError, interp_linear_extrap_nearest, -3.74)
        self.assertRaises(ValueError, interp_linear_extrap_nearest, 1.0)

    def test_infinity_handling(self):
        """Test extrapolating at infinite values are outside the extrapolation range"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        self.assertRaises(ValueError, interp_linear_extrap_nearest, np.inf)
        self.assertRaises(ValueError, interp_linear_extrap_linear, np.inf)
        self.assertRaises(ValueError, interp_linear_extrap_nearest, -np.inf)
        self.assertRaises(ValueError, interp_linear_extrap_linear, -np.inf)

    def test_infinity_as_a_spline_point(self):
        """If one of the spline points is infinite, returns nan"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        y_in[0] = np.inf
        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        # Extrapolations return infinity, interpolations return nan for now
        self.assertTrue(np.isnan(interp_linear_extrap_nearest(-1.73)))
        self.assertTrue(np.isinf(interp_linear_extrap_nearest(-1.8)))
        self.assertTrue(np.isnan(interp_linear_extrap_linear(-1.73)))
        self.assertTrue(np.isinf(interp_linear_extrap_linear(-1.8)))

    def test_nan_as_a_spline_point(self):
        """If one of the spline points is nan, returns nan"""
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        y_in[0] = np.nan
        interp_linear_extrap_nearest = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        interp_linear_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.LinearInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        # Extrapolations and interpolations return nan
        self.assertTrue(np.isnan(interp_linear_extrap_nearest(-1.73)))
        self.assertTrue(np.isnan(interp_linear_extrap_nearest(-1.8)))
        self.assertTrue(np.isnan(interp_linear_extrap_linear(-1.73)))
        self.assertTrue(np.isnan(interp_linear_extrap_linear(-1.8)))

    def test_enforce_monotonicity(self):
        """The range of x values must be ordered from lowest to highest"""
        x_in = np.arange(-1.4, -1.73, -0.1)
        y_in = np.sin(x_in)
        dict_kwargs_linear_interp_nearest_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0}

        dict_kwargs_linear_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}

        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_linear_extrap)

    def test_equal_spline_array_length(self):
        """The input spline arrays must be the same length"""
        x_in = np.arange(-1.4, -1.73, -0.1)
        y_in = np.sin(x_in)
        y_in = y_in[:-1]
        dict_kwargs_linear_interp_nearest_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0}

        dict_kwargs_linear_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_linear_extrap)

    def test_not_length_1_array_linear_extrapolation(self):
        """At least 2 spline points are needed (for extrapolation type linear)"""
        x_in = np.array([0.1])
        y_in = np.sin(x_in)

        dict_kwargs_linear_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_linear_extrap)

    def test_array_dimension_checks(self):
        """The interpolator should check the dimensions of the spline points are correct"""
        x = np.arange(-1.4, -1.73, -0.1)
        x_in = np.meshgrid(x, x)
        x_in = np.array(x_in)
        y_in = np.sin(x_in)

        dict_kwargs_linear_interp_nearest_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0}

        dict_kwargs_linear_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.LinearInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_linear_interp_linear_extrap)