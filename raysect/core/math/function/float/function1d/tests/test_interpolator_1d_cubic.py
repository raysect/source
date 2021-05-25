
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

        x_in = np.arange(-2, 5., 1.)

        y_in = x_in**3 - 10.*x_in**2 +3.*x_in + 10.

        cls._connection = Interpolate1D(x_in, y_in, InterpType.CubicConstrainedInt, ExtrapType.NearestExt, extrapolation_range=2.0)
        cls._x = x_in
        cls._y = y_in
        cls._dy_dx = 3.*x_in**2 - 2.*10.*x_in + 3.
        cls._interpolator = _Interpolator1DCubic(cls._x, cls._y)

        cls._range_x = np.arange(-2, 5., .07)
        cls._expected_y = cls._range_x**3 - 10.*cls._range_x**2 + 3.*cls._range_x + 10.
        # Generated in scipy.__version__ 1.6.3 using scipy.interpolate.CubicSpline using the _x, _y spline points
        cls._range_y = np.array(
            [-370., -258.707, -169.856, -101.389, -51.248, -17.375, 2.288, 9.799, 7.216, -3.403, -20., -40.517,
             -62.896, -85.079, -105.008]
        )

    @classmethod
    def tearDownClass(cls):
        cls._connection = None
        cls._interpolator = None
        cls._x = None
        cls._y = None
        cls._dy_dx = None
        cls._range_x = None
        cls._range_y = None
        cls._expected_y = None

    # def test_interpolator_1d_cubic_constrained(self):
    #     """Tests that the 1D cubic spline around a minimum point. The interpolation at the known minimum should
    #     be lower than the spline points surrounding it """
    #
    #     import matplotlib.pyplot as plt
    #     fig, ax = plt.subplots()
    #     from scipy.interpolate import CubicSpline
    #     test_1d = CubicSpline(self._x, self._y)
    #     ysave = np.zeros(len(self._range_x))
    #     for i in range(len(self._range_x)):
    #         # ax.plot(self._range_x, test_1d(self._range_x), 'go')
    #         ax.plot(self._range_x[i], test_1d(self._range_x[i]), 'go')
    #         ax.plot(self._range_x[i], self._connection(self._range_x[i]), 'bx')
    #         ysave[i] = test_1d(self._range_x[i])
    #
    #     ax.plot(self._x, self._y, 'ro')
    #     ax.plot(self._range_x, self._expected_y, 'mo')
    #     # ax.plot(self._range_x, self._range_y, 'rx')
    #     index = 2
    #
    #     m = self._interpolator.test_get_gradient(self._x, self._y, index)
    #     c = self._y[index] - m * self._x[index]
    #     tangent_x = np.array([self._x[index-1], self._x[index+1]])
    #     tangent = np.array(m*tangent_x + c)
    #     ax.plot(tangent_x, tangent, '-r')
    #
    #     plt.show()
    #
    # def test_gradient(self):
    #     for i in range(1, len(self._x)-1):
    #         index = i
    #         m = self._interpolator.test_get_gradient(self._x, self._y, index)
    #         print(i, m, self._dy_dx[index], (self._y[index+1]-self._y[index-1])/2., self._x[i])

    def test_interpolator_1d_cubic_at_a_minimum(self):
        """Tests the 1D cubic spline around a minimum point. The interpolation at the known minimum should
        be lower than the spline points surrounding it """
        x_in = np.arange(-1.73, -1.4, 0.1)
        y_in = np.sin(x_in)

        interp_cubic_extrap_nearest = Interpolate1D(x_in, y_in, InterpType.CubicInt, ExtrapType.NearestExt,
                                                    extrapolation_range=2.0)

        # Test between spline points (minimum at -pi/2)
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
        self.assertEqual(interp_cubic_extrap_nearest(-1.75), interp_cubic_extrap_nearest(-1.8))
        self.assertEqual(y_in[-1], interp_cubic_extrap_nearest(-1.0))
        self.assertEqual(interp_cubic_extrap_nearest(0.0), interp_cubic_extrap_nearest(-1.0))

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

    def test_enforce_monotonicity(self):
        """The range of x values must be ordered from lowest to highest"""
        x_in = np.arange(-1.4, -1.73, -0.1)
        y_in = np.sin(x_in)
        dict_kwargs_cubic_interp_nearest_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0}

        dict_kwargs_cubic_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}

        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)

    def test_equal_spline_array_length(self):
        """The input spline arrays must be the same length"""
        x_in = np.arange(-1.4, -1.73, -0.1)
        y_in = np.sin(x_in)
        y_in = y_in[:-1]
        dict_kwargs_cubic_interp_nearest_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.NearestExt, 'extrapolation_range': 2.0}

        dict_kwargs_cubic_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_nearest_extrap)
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)

    def test_not_length_1_array_linear_extrapolation(self):
        """At least 2 spline points are needed (for extrapolation type linear)"""
        x_in = np.array([0.1])
        y_in = np.sin(x_in)

        dict_kwargs_cubic_interp_linear_extrap = {'x': x_in, 'f': y_in, 'interpolation_type': InterpType.CubicInt,
                 'extrapolation_type': ExtrapType.LinearExt, 'extrapolation_range': 2.0}
        self.assertRaises(ValueError, Interpolate1D, **dict_kwargs_cubic_interp_linear_extrap)

    def test_array_dimension_checks(self):
        """The interpolator should check the dimensions of the spline points are correct"""
        x = np.arange(-1.4, -1.73, -0.1)
        x_in = np.meshgrid(x, x)
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
