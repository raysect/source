
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
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, InterpType, ExtrapType, _Interpolator1DCubic
import numpy as np


class TestInterpolator1DCubic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        x_in_cubic = np.arange(-2, 5., 1.)
        y_cubic = lambda x: x**3 - 10.*x**2 +3.*x + 10.
        y_in_cubic = y_cubic(x_in_cubic)
        cls._x = x_in_cubic
        cls._y = y_in_cubic
        dy_dx_cubic = lambda x: 3.*x**2 - 2.*10.*x + 3.
        cls._dy_dx = dy_dx_cubic(x_in_cubic)
        cls._cubic_1d_cubic_function = Interpolate1D(
            x_in_cubic, y_in_cubic, InterpType.CubicConstrainedInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        cls._interpolator_cubic = _Interpolator1DCubic(cls._x, cls._y)

        cls._cubic_interp_range_x = np.arange(-2, 5., .07)
        cls._expected_y = y_cubic(cls._cubic_interp_range_x)

        cls._x_in_sin = np.arange(-1.73, -1.4, 0.1)
        cls._y_in_sin = np.sin(cls._x_in_sin)
        cls._dy_dx_in_sin = np.cos(cls._x_in_sin)
        cls._interp_cubic_extrap_linear = Interpolate1D(
            cls._x_in_sin, cls._y_in_sin, InterpType.CubicInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        cls._interp_cubic_extrap_nearest = Interpolate1D(
            cls._x_in_sin, cls._y_in_sin, InterpType.CubicInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        cls._interpolator_sin = _Interpolator1DCubic(cls._x_in_sin, cls._y_in_sin)
        r_approx = np.zeros((len(cls._x_in_sin)-1))
        # Choose the function for the differential
        diff_function = [lambda x: np.sin(x), lambda x: np.cos(x), lambda x: -np.sin(x), lambda x: -np.cos(x)]
        x_diff = np.diff(cls._x_in_sin)

        for i in range(3, 10):
            # Calculate the taylor approximations after n=3. This is max difference to the function value (not deriv)
            factorial = np.prod(np.arange(1, i+1))
            choose_diff_function = diff_function[i % len(diff_function)]
            r_approx = r_approx + np.power(x_diff, i) * choose_diff_function(cls._x_in_sin[:-1])/factorial
        # The taylor expansion higher order terms (sum from 3 to 10) for each point up to n-1
        cls.r_approx = np.abs(r_approx)

        # Linear, nearest extrpolation
        cls._interp_linear_extrap_nearest = Interpolate1D(
            cls._x_in_sin, cls._y_in_sin, InterpType.LinearInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )
        # Cubic Constrained, nearest extrpolation
        cls._interp_cubic_constrained_extrap_nearest = Interpolate1D(
            cls._x_in_sin, cls._y_in_sin, InterpType.CubicConstrainedInt, ExtrapType.NearestExt, extrapolation_range=2.0
        )

    @classmethod
    def tearDownClass(cls):
        cls._cubic_1d = None
        cls._interpolator_cubic = None
        cls._cubic_1d_cubic_function = None
        cls._x = None
        cls._y = None
        cls._dy_dx = None
        cls._cubic_interp_range_x = None
        cls._expected_y = None
        cls.r_approx = None
        cls._x_in_sin = None
        cls._y_in_sin = None
        cls._dy_dx_in_sin = None
        cls._interp_cubic_extrap_linear = None
        cls._interp_cubic_extrap_nearest = None
        cls._interpolator_sin = None
        cls._interp_linear_extrap_nearest = None
        cls._interp_cubic_constrained_extrap_nearest = None

    def test_polynomial_cubic_constrained_spline(self):
        """
        The constrained spline interpolated values must not be out of bounds of the spline knot point values
        """

        x_range = np.linspace(self._x_in_sin[0], self._x_in_sin[-1], 100)
        max_y = np.max(self._y_in_sin)
        min_y = np.min(self._y_in_sin)
        for i in range(len(x_range)):
            self.assertGreaterEqual(max_y, self._interp_cubic_constrained_extrap_nearest(x_range[i]))
            self.assertGreaterEqual(self._interp_cubic_constrained_extrap_nearest(x_range[i]), min_y)

    def test_polynomial(self):
        """
        Pick two spline points and test over the scaled range 0,1 that interpolated points match
        between the calculated cubic coefficients and the evaluated interpolator value
        """
        index_1 = 0
        a = self._interpolator_sin.test_return_polynormial_coefficients(index_1)
        x_range = np.linspace(self._x_in_sin[index_1], self._x_in_sin[index_1+1], 20)
        x_range_scal = (x_range-self._x_in_sin[index_1])/(self._x_in_sin[index_1+1]-self._x_in_sin[index_1])

        y_out = np.zeros((len(x_range_scal)))
        y_out2 = np.zeros((len(x_range_scal)))

        for i in range(len(x_range_scal)):
            y_out[i] = a[0]*x_range_scal[i]**3 + a[1]*x_range_scal[i]**2 + a[2]*x_range_scal[i] + a[3]
            y_out2[i] = self._interp_cubic_extrap_nearest(x_range[i])
            self.assertAlmostEqual(
                y_out[i], y_out2[i]
            )

    def test_gradient(self):
        """
        The gradient is approximated up to the 3rd term in a taylor expansion.

        The central difference doesn't have 3rd order terms included (second order are cancelled out), so this is
        approximately the order that the gradient should be accurate to.
        """

        for i in range(1, len(self._x_in_sin)-1):
            index = i
            m = self._interpolator_sin.test_get_gradient(index)

            self.assertAlmostEqual(
                m, self._dy_dx_in_sin[index]*(self._x_in_sin[index+1]-self._x_in_sin[index]), delta=2.*self.r_approx[i]
            )

    def test_interpolator_1d_cubic_at_a_minimum(self):
        """
        Tests the 1D cubic spline around a minimum point. The interpolation at the known minimum should
        be lower than the spline points surrounding it
        """
        # Test between spline points (minimum at -pi/2)
        for i in range(len(self._y_in_sin)):
            self.assertGreaterEqual(self._y_in_sin[i], self._interp_cubic_extrap_nearest(-np.pi/2.))

    def test_nearest_neighbour_1d_extrapolation(self):
        """
        Tests for the nearest neighbour extrapolator returns the edge spline points in the lower and upper
        extrapolation range
        """
        # Nearest neighbour extrapolation test
        self.assertEqual(self._y_in_sin[0], self._interp_cubic_extrap_nearest(-1.8))
        self.assertEqual(self._interp_cubic_extrap_nearest(-1.75), self._interp_cubic_extrap_nearest(-1.8))
        self.assertEqual(self._y_in_sin[-1], self._interp_cubic_extrap_nearest(-1.0))
        self.assertEqual(self._interp_cubic_extrap_nearest(0.0), self._interp_cubic_extrap_nearest(-1.0))

    def test_cubic_1d_at_spline_points(self):
        """
        Tests that the cubic interpolator returns the value of the function at the spline points
        The value at the last spline point is returned from the linear extrapolator
        """

        # Interpolating at the spline points should return the exact value (the end spline point is an extrapolation)
        for i in range(len(self._x_in_sin)):
            self.assertEqual(self._y_in_sin[i], self._interp_cubic_extrap_linear(self._x_in_sin[i]),
                             msg="Constant1D call did not match reference value.")
            self.assertEqual(self._y_in_sin[i], self._interp_cubic_extrap_nearest(self._x_in_sin[i]),
                             msg="Constant1D call did not match reference value.")

    def test_linear_1d_extrapolation_gradient(self):
        """
        Tests that the linear extrapolator calculates a similar gradient to expected
        """
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)
        interp_cubic_extrap_linear = Interpolate1D(
            x_in, y_in, InterpType.CubicInt, ExtrapType.LinearExt, extrapolation_range=2.0
        )
        # Linear extrapolation test
        expected_start_grad = ((self._y_in_sin[1]-self._y_in_sin[0])/(self._x_in_sin[1]-self._x_in_sin[0]))
        expected_end_grad = ((self._y_in_sin[-2]-self._y_in_sin[-1])/(self._x_in_sin[-2]-self._x_in_sin[-1]))
        self.assertAlmostEqual(
            expected_start_grad, (y_in[0] - self._interp_cubic_extrap_linear(-1.8))/(self._x_in_sin[0]--1.8)
        )
        self.assertAlmostEqual(
            expected_end_grad, (self._interp_cubic_extrap_linear(-1.0)-self._y_in_sin[-1])/(-1.0-self._x_in_sin[-1])
        )

    def test_1d_extrapolation_range_out_of_bounds(self):
        """
        Tests for the linear and nearest neighbour extrapolator for inputs outside the extrapolation range,
        an error is raised
        """

        # Out of bounds test
        self.assertRaises(ValueError, self._interp_cubic_extrap_linear, -3.74)
        self.assertRaises(ValueError, self._interp_cubic_extrap_linear, 1.0)

        # Outside extrapolation range, there should be an error raised
        self.assertRaises(ValueError, self._interp_cubic_extrap_nearest, -3.74)
        self.assertRaises(ValueError, self._interp_cubic_extrap_nearest, 1.0)

    def test_enforce_monotonicity(self):
        """
        The range of x values must be ordered from lowest to highest
        """
        x_in = np.arange(-1.4, -1.73, -0.1)
        y_in = np.sin(x_in)

        dict_kwargs_cubic_interp_nearest_extrap = {
            'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
            'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0
        }

        dict_kwargs_cubic_interp_linear_extrap = {
            'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0
        }

        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)

    def test_equal_spline_array_length(self):
        """
        The input spline arrays must be the same length
        """

        y_in = self._y_in_sin[:-1]
        dict_kwargs_cubic_interp_nearest_extrap = {
            'x': self._x_in_sin, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
            'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0
        }

        dict_kwargs_cubic_interp_linear_extrap = {
            'x': self._x_in_sin, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0
        }
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)

    def test_not_length_1_array_linear_extrapolation(self):
        """
        At least 2 spline points are needed (for extrapolation type linear)
        """
        x_in = np.array([0.1])
        y_in = np.sin(x_in)

        dict_kwargs_cubic_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)

    def test_array_dimension_checks(self):
        """
        The interpolator should check the dimensions of the spline points are correct
        """
        # x = np.arange(-1.4, -1.73, -0.1)
        x_in = np.meshgrid(self._x_in_sin, self._x_in_sin)
        x_in = np.array(x_in)
        y_in = np.sin(x_in)

        dict_kwargs_cubic_interp_nearest_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                                                   'extrapolation_type': ExtrapType.NearestExt,
                                                   'extrapolation_range': 2.0}

        dict_kwargs_cubic_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                                                  'extrapolation_type': ExtrapType.LinearExt,
                                                  'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)
