
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
import numpy as np
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D


X_LOWER = 0.0
X_UPPER = 1.0
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_XSAMPLES = 30


class TestInterpolators1D(unittest.TestCase):
    def setUp(self) -> None:

        # self.data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in self.data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER

        #: x values used to obtain self.data
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)

        #: data array from a function sampled on self.x. dtype should be np.float64
        self.data: np.array = np.array(
            [0., 0.11088263, 0.22039774, 0.3271947 , 0.42995636, 0.52741539, 0.6183698 , 0.70169788, 0.77637192,
             0.84147098]
        )

        #: precalculated result of sampling self.data on self.xsamples
        #   should be set in interpolator specific setup function.
        self.precalc_interpolation = None

        #: precalculated result of the function used to calculate self.data on self.xsamples
        self.precalc_function = np.array(
            [0., 0.03447593, 0.06891086, 0.10326387, 0.13749409, 0.17156085, 0.20542363, 0.23904218, 0.27237651,
             0.30538701, 0.33803442, 0.37027992, 0.40208519, 0.43341239, 0.4642243, 0.49448427, 0.52415632, 0.55320518,
             0.58159632, 0.60929597, 0.6362712, 0.66248994, 0.68792102, 0.7125342, 0.73630021, 0.75919081, 0.78117877,
             0.80223796, 0.82234333, 0.84147098]
        )

        #: x values on which self.precalc_interpolation was samples on
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

        #: array holding precalculated linear extrapolation data
        #   to be set in interpolator specific setup_ method
        self.precalc_extrapolation_linear: np.array = None

        #: array holding precalculated cubic extrapolation data
        #   to be set in interpolator specific setup_ method
        self.precalc_extrapolation_cubic: np.array = None

        #: x values on which self.precalc_extrapolation_ arrays were sampled on
        self.xsamples_extrap = np.array(
            [
                X_LOWER - X_EXTRAP_DELTA_MAX,
                X_LOWER - X_EXTRAP_DELTA_MIN,
                X_UPPER + X_EXTRAP_DELTA_MIN,
                X_UPPER + X_EXTRAP_DELTA_MAX,
            ],
            dtype=np.float64,
        )

        #: the interpolator object that is being tested. Set in setup_ method
        self.interpolator: Interpolate1D = None

    def setup_linear(self, extrapolator_type: str, extrapolation_range: float) -> None:
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1D object that is being tested.

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        """

        # set precalculated expected interpolation results
        # this is the result of sampling self.data on self.xsamples
        self.precalc_interpolation = None

        # set precalculated expected extrapolation results
        # this is the result of nearest neighbour extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_nearest = None

        # set precalculated expected extrapolation results
        # this is the result of linear extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_linear = None

        # set interpolator
        self.interpolator = Interpolate1D(self.x, self.data, "linear", extrapolator_type, extrapolation_range)

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1D object that is being tested.

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        """

        # set precalculated expected interpolation results
        # this is the result of sampling self.data on self.xsamples
        self.precalc_interpolation = np.array(
            [0., 0.03445727, 0.06892362, 0.10327643, 0.13747237, 0.17156602, 0.20544274, 0.23902114, 0.27237365,
             0.30540749, 0.33801842, 0.37026979, 0.40210303, 0.43340598, 0.46420873, 0.49449677, 0.52416232, 0.55318685,
             0.58160215, 0.609309, 0.63625324, 0.66248906, 0.68793574, 0.7125198, 0.73629368, 0.75920346, 0.78114771,
             0.80176427, 0.82158583, 0.84147098]
        )

        # set precalculated expected extrapolation results
        # this is the result of nearest neighbour extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_linear = None

        # set precalculated expected extrapolation results
        # this is the result of linear extrapolation on self.xsamples_extrap
        self.precalc_extrapolation_cubic = None

        # set interpolator
        self.interpolator = Interpolate1D(self.x, self.data, "linear", extrapolator_type, extrapolation_range)

    def test_linear_interpolation(self):
        self.setup_linear("none", 0.0)

    def test_linear_interpolation_extrapolators(self):
        self.setup_linear("nearest", 0.0)
        # test linear interpolation with 'nearest' extrapolator here

        self.setup_linear("linear", 0.0)
        # test linear interpolation with 'linear' extrapolator here

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against scipy.interpolate.CubicHermiteSpline with the same gradient calculations generated using scipy
        version 1.6.3
        """
        self.setup_cubic("nearest", 0.0)
        # test cubic interpolation with 'nearest' extrapolator here

        self.setup_cubic("linear", 0.0)
        # test cubic interpolation with 'linear' extrapolator here
