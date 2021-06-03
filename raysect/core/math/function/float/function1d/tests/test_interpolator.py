
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
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, id_to_extrapolator


X_LOWER = 0.0
X_UPPER = 1.0
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_XSAMPLES = 30

EXTRAPOLATION_RANGE = 0.06

PRECISION = 12


class TestInterpolators1D(unittest.TestCase):
    def setUp(self) -> None:

        # self.data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in self.data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER

        #: x values used to obtain self.data
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)

        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [0., 0.11088262851 , 0.220397743456, 0.327194696796, 0.429956363528, 0.527415385772, 0.61836980307,
             0.701697876147, 0.776371921301, 0.841470984808]#, dtype=np.float64
        )

        #: precalculated result of sampling self.data on self.xsamples
        #   should be set in interpolator specific setup function.
        self.precalc_interpolation = None

        #: precalculated result of the function used to calculate self.data on self.xsamples
        self.precalc_function = np.array(
            [0., 0.034475925345, 0.068910860786, 0.103263865154, 0.13749409469, 0.171560851609, 0.205423632484,
             0.239042176405, 0.272376512846, 0.305387009186, 0.338034417834, 0.370279922885, 0.402085186279,
             0.433412393376, 0.464224297918, 0.494484266311, 0.524156321185, 0.553205184161, 0.581596317803,
             0.609295966676, 0.636271197481, 0.662489938212, 0.687921016284, 0.7125341956, 0.736300212497,
             0.759190810539, 0.781178774115, 0.802237960792, 0.822343332402, 0.841470984808], dtype=np.float64
        )

        #: x values on which self.precalc_interpolation was samples on
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np. array(
            [0., 0., 0.841470984808, 0.841470984808], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [-0.079835492527, -0.039917746264,  0.864906647671,  0.888342310533], dtype=np.float64
        )

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

        #: set precalculated expected extrapolation results  Set in setup_ method
        self.precalc_extrapolation = None

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
        # this is the result of the type of extrapolation on self.xsamples_extrap
        self.setup_extrpolation_type(extrapolator_type)

        # set interpolator
        self.interpolator = Interpolate1D(self.x, self.data, 'linear', extrapolator_type, extrapolation_range)

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
            [0., 0.034457267669, 0.068923618826, 0.103276426379, 0.137472367276, 0.171566022088, 0.205442737431,
             0.239021136276, 0.27237364671, 0.305407493229, 0.338018423689, 0.37026978657, 0.40210303144, 0.43340598175,
             0.464208732427, 0.494496772755, 0.524162321048, 0.553186854304, 0.581602151176, 0.609309003405,
             0.636253243661, 0.662489056087, 0.687935735681, 0.712519804849, 0.736293678079, 0.759203461591,
             0.78114771311, 0.801764272516, 0.821585828526, 0.841470984808]#, dtype=np.float64
        )
        self.setup_extrpolation_type(extrapolator_type)
        # set interpolator
        self.interpolator = Interpolate1D(self.x, self.data, 'cubic', extrapolator_type, extrapolation_range)

    def setup_extrpolation_type(self, extrapolator_type: str):
        if extrapolator_type == 'linear':
            self.precalc_extrapolation = np.copy(self.precalc_extrapolation_linear)
        elif extrapolator_type == 'nearest':
            self.precalc_extrapolation = np.copy(self.precalc_extrapolation_nearest)
        elif extrapolator_type == 'none':
            self.precalc_extrapolation = None
        else:
            raise ValueError(
                f'Extrapolation type {extrapolator_type} not found or no test. options are {id_to_extrapolator.keys()}'
            )

    def test_linear_interpolation(self):
        self.setup_linear('none', 0.0)

    def test_linear_interpolation_extrapolators(self):
        self.setup_linear('nearest', EXTRAPOLATION_RANGE)

        # test linear interpolation with 'nearest' extrapolator here
        self.run_general_extrapolation_tests()

        self.setup_linear('linear', EXTRAPOLATION_RANGE)
        # test linear interpolation with 'linear' extrapolator here
        self.run_general_extrapolation_tests()

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against scipy.interpolate.CubicHermiteSpline with the same gradient calculations generated using scipy
        version 1.6.3
        """
        self.setup_cubic('nearest', EXTRAPOLATION_RANGE)
        # test cubic interpolation with 'nearest' extrapolator here
        self.run_general_extrapolation_tests()
        self.run_general_interpolation_tests()

        self.setup_cubic('linear', EXTRAPOLATION_RANGE)
        # test cubic interpolation with 'linear' extrapolator here
        self.run_general_extrapolation_tests()
        self.run_general_interpolation_tests()

    def run_general_extrapolation_tests(self):
        # Test extrapolator out of range, there should be an error raised
        self.assertRaises(ValueError, self.interpolator, self.xsamples_extrap[0])
        self.assertRaises(ValueError, self.interpolator, self.xsamples_extrap[-1])

        # Test extrapolation inside extrapolation range matches the predefined values
        self.assertAlmostEqual(
            self.interpolator(self.xsamples_extrap[1]), self.precalc_extrapolation[1], places=PRECISION - 1
        )

    def run_general_interpolation_tests(self):

        # Test continuity of smoothness

        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            self.assertAlmostEqual(
                self.interpolator(self.xsamples[i]), self.precalc_interpolation[i], places=PRECISION - 1
            )


