
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
Unit tests for the Interpolator1DCubic class from within Interpolate1DArray,
including interaction with Extrapolator1DLinear and Extrapolator1DNearest.
"""
import unittest
import numpy as np
from raysect.core.math.function.float.function1d.interpolate import Interpolate1DArray, id_to_extrapolator, \
    id_to_interpolator


X_LOWER = 0.0
X_UPPER = 1.0
X_EXTRAP_DELTA_MAX = 0.08
X_EXTRAP_DELTA_MIN = 0.04

NB_X = 10
NB_XSAMPLES = 30

EXTRAPOLATION_RANGE = 0.06

PRECISION = 12

BIG_VALUE_FACTOR = 20.
SMALL_VALUE_FACTOR = -20.


class TestInterpolatorLoadValues:
    def __init__(self):
        # Define in setup_cubic or setup_linear
        self.precalc_interpolation = None


class TestInterpolatorLoadNormalValues(TestInterpolatorLoadValues):
    """
    Loading values for the original np.sin(x) tests.

    These data are saved to 12 significant figures.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [0.000000000000E+00, 1.108826285100E-01, 2.203977434561E-01, 3.271946967962E-01, 4.299563635284E-01,
             5.274153857719E-01, 6.183698030697E-01, 7.016978761467E-01, 7.763719213007E-01, 8.414709848079E-01],
            dtype=np.float64
        )

        #: precalculated result of the function used to calculate self.data on self.xsamples
        self.precalc_function = np.array(
            [0.000000000000E+00, 3.447592534511E-02, 6.891086078616E-02, 1.032638651537E-01,
             1.374940946898E-01, 1.715608516085E-01, 2.054236324837E-01, 2.390421764051E-01,
             2.723765128461E-01, 3.053870091864E-01, 3.380344178335E-01, 3.702799228850E-01,
             4.020851862791E-01, 4.334123933760E-01, 4.642242979177E-01, 4.944842663114E-01,
             5.241563211847E-01, 5.532051841610E-01, 5.815963178030E-01, 6.092959666760E-01,
             6.362711974815E-01, 6.624899382121E-01, 6.879210162842E-01, 7.125341956002E-01,
             7.363002124969E-01, 7.591908105392E-01, 7.811787741148E-01, 8.022379607921E-01,
             8.223433324022E-01, 8.414709848079E-01], dtype=np.float64
        )
        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np.array(
            [0.000000000000E+00, 0.000000000000E+00, 8.414709848079E-01, 8.414709848079E-01], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [-7.983549252717E-02, -3.991774626358E-02, 8.649066476705E-01, 8.883423105331E-01], dtype=np.float64
        )
        #: array holding precalculated quadratic extrapolation data
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [-8.001272228503E-02, -3.996205370305E-02, 8.645964182651E-01, 8.871013929117E-01], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [0.000000000000E+00, 3.445726766897E-02, 6.892361882629E-02, 1.032764263792E-01,
             1.374723672762E-01, 1.715660220877E-01, 2.054427374305E-01, 2.390211362762E-01,
             2.723736467104E-01, 3.054074932289E-01, 3.380184236890E-01, 3.702697865700E-01,
             4.021030314403E-01, 4.334059817503E-01, 4.642087324270E-01, 4.944967727547E-01,
             5.241623210482E-01, 5.531868543036E-01, 5.816021511759E-01, 6.093090034050E-01,
             6.362532436612E-01, 6.624890560875E-01, 6.879357356812E-01, 7.125198048495E-01,
             7.362936780787E-01, 7.592034615909E-01, 7.811477131101E-01, 8.017642725164E-01,
             8.215858285263E-01, 8.414709848079E-01], dtype=np.float64
        )

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [0.000000000000E+00, 3.441185022723E-02, 6.882370045445E-02, 1.032355506817E-01,
             1.373173114280E-01, 1.713047608940E-01, 2.052922103601E-01, 2.388110112734E-01,
             2.719548933444E-01, 3.050987754155E-01, 3.378252140443E-01, 3.697167657888E-01,
             4.016083175333E-01, 4.333170194678E-01, 4.635629229227E-01, 4.938088263776E-01,
             5.240547298324E-01, 5.525062595092E-01, 5.807334924637E-01, 6.089607254182E-01,
             6.356100940512E-01, 6.614705305234E-01, 6.873309669955E-01, 7.119977444438E-01,
             7.351724481123E-01, 7.583471517807E-01, 7.808615118874E-01, 8.010646695275E-01,
             8.212678271677E-01, 8.414709848079E-01], dtype=np.float64
        )


class TestInterpolatorLoadBigValues(TestInterpolatorLoadValues):
    """
    Loading big values (10^20 times the original) instead of the original np.sin(x).

    These data are saved to 12 significant figures.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [0.000000000000E+00, 1.108826285100E+19, 2.203977434561E+19, 3.271946967962E+19,
             4.299563635284E+19, 5.274153857719E+19, 6.183698030697E+19, 7.016978761467E+19,
             7.763719213007E+19, 8.414709848079E+19], dtype=np.float64
        )
        #: precalculated result of the function used to calculate self.data on self.xsamples
        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np.array(
            [0.000000000000e+00, 0.000000000000e+00, 8.414709848079e+19, 8.414709848079e+19], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [-7.983549252717e+18, -3.991774626358e+18,  8.649066476705e+19,  8.883423105331e+19], dtype=np.float64
        )

        #: array holding precalculated quadratic extrapolation data
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [-8.001272228503E+18, -3.996205370305E+18, 8.645964182651E+19, 8.871013929117E+19], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.445726766897e+18, 6.892361882629e+18, 1.032764263792e+19,
             1.374723672762e+19, 1.715660220877e+19, 2.054427374305e+19, 2.390211362762e+19,
             2.723736467104e+19, 3.054074932289e+19, 3.380184236890e+19, 3.702697865700e+19,
             4.021030314403e+19, 4.334059817503e+19, 4.642087324270e+19, 4.944967727547e+19,
             5.241623210482e+19, 5.531868543036e+19, 5.816021511759e+19, 6.093090034050e+19,
             6.362532436612e+19, 6.624890560875e+19, 6.879357356812e+19, 7.125198048495e+19,
             7.362936780787e+19, 7.592034615909e+19, 7.811477131101e+19, 8.017642725164e+19,
             8.215858285263e+19, 8.414709848079e+19], dtype=np.float64)

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.441185022723e+18, 6.882370045445e+18, 1.032355506817e+19,
             1.373173114280e+19, 1.713047608940e+19, 2.052922103601e+19, 2.388110112734e+19,
             2.719548933444e+19, 3.050987754155e+19, 3.378252140443e+19, 3.697167657888e+19,
             4.016083175333e+19, 4.333170194678e+19, 4.635629229227e+19, 4.938088263776e+19,
             5.240547298324e+19, 5.525062595092e+19, 5.807334924637e+19, 6.089607254182e+19,
             6.356100940512e+19, 6.614705305234e+19, 6.873309669955e+19, 7.119977444438e+19,
             7.351724481123e+19, 7.583471517807e+19, 7.808615118874e+19, 8.010646695275e+19,
             8.212678271677e+19, 8.414709848079e+19], dtype=np.float64)


class TestInterpolatorLoadSmallValues(TestInterpolatorLoadValues):
    """
    Loading small values (10^-20 times the original) instead of the original np.sin(x)

    These data are saved to 12 significant figures.
    """
    def __init__(self):
        super().__init__()
        #: data array from a function sampled on self.x. dtype should be np.float64
        # self.data: np.array = np.sin(self.x)
        self.data: np.array = np.array(
            [0.000000000000E+00, 1.108826285100E-21, 2.203977434561E-21, 3.271946967962E-21,
             4.299563635284E-21, 5.274153857719E-21, 6.183698030697E-21, 7.016978761467E-21,
             7.763719213007E-21, 8.414709848079E-21], dtype=np.float64
        )

        #: precalculated result of the function used to calculate self.data on self.xsamples
        # self.precalc_function = np.array()
        #: array holding precalculated nearest neighbour extrapolation data
        self.precalc_extrapolation_nearest: np.array = np.array(
            [0.000000000000e+00, 0.000000000000e+00, 8.414709848079e-21, 8.414709848079e-21], dtype=np.float64
        )

        #: array holding precalculated linear extrapolation data
        self.precalc_extrapolation_linear: np.array = np.array(
            [-7.983549252717e-22, -3.991774626358e-22,  8.649066476705e-21,  8.883423105331e-21], dtype=np.float64
        )

        #: array holding precalculated quadratic extrapolation data
        self.precalc_extrapolation_quadratic: np.array = np.array(
            [-8.001272228503E-22, -3.996205370305E-22, 8.645964182651E-21, 8.871013929117E-21], dtype=np.float64
        )

    def setup_cubic(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.445726766897e-22, 6.892361882629e-22, 1.032764263792e-21,
             1.374723672762e-21, 1.715660220877e-21, 2.054427374305e-21, 2.390211362762e-21,
             2.723736467104e-21, 3.054074932289e-21, 3.380184236890e-21, 3.702697865700e-21,
             4.021030314403e-21, 4.334059817503e-21, 4.642087324270e-21, 4.944967727547e-21,
             5.241623210482e-21, 5.531868543036e-21, 5.816021511759e-21, 6.093090034050e-21,
             6.362532436612e-21, 6.624890560875e-21, 6.879357356812e-21, 7.125198048495e-21,
             7.362936780787e-21, 7.592034615909e-21, 7.811477131101e-21, 8.017642725164e-21,
             8.215858285263e-21, 8.414709848079e-21], dtype=np.float64
        )

    def setup_linear(self):
        self.precalc_interpolation = np.array(
            [0.000000000000e+00, 3.441185022723e-22, 6.882370045445e-22, 1.032355506817e-21,
             1.373173114280e-21, 1.713047608940e-21, 2.052922103601e-21, 2.388110112734e-21,
             2.719548933444e-21, 3.050987754155e-21, 3.378252140443e-21, 3.697167657888e-21,
             4.016083175333e-21, 4.333170194678e-21, 4.635629229227e-21, 4.938088263776e-21,
             5.240547298324e-21, 5.525062595092e-21, 5.807334924637e-21, 6.089607254182e-21,
             6.356100940512e-21, 6.614705305234e-21, 6.873309669955e-21, 7.119977444438e-21,
             7.351724481123e-21, 7.583471517807e-21, 7.808615118874e-21, 8.010646695275e-21,
             8.212678271677e-21, 8.414709848079e-21], dtype=np.float64
        )


class TestInterpolators1D(unittest.TestCase):
    def setUp(self) -> None:

        # self.data is a precalculated input values for testing. It's the result of applying function f on self.x
        # as in self.data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER

        #: x values used to obtain self.data
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)

        self.test_loaded_values = TestInterpolatorLoadNormalValues()
        self.test_loaded_big_values = TestInterpolatorLoadBigValues()
        self.test_loaded_small_values = TestInterpolatorLoadSmallValues()

        #: precalculated result of sampling self.data on self.xsamples
        #   should be set in interpolator specific setup function.
        self.precalc_interpolation = None

        #: x values on which self.precalc_interpolation was samples on
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

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
        self.interpolator: Interpolate1DArray = None

    def setup_linear(
            self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool) -> None:
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1DArray object that is being tested. Precalculated interpolation using
        scipy.interpolate.interp1d(kind=linear), generated using scipy version 1.6.3

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        :param big_values: For loading and testing big value saved data
        :param small_values: For loading and testing small value saved data
        """

        # set precalculated expected interpolation results  using scipy.interpolate.interp1d(kind=linear)
        # this is the result of sampling self.data on self.xsamples

        if big_values:
            self.value_storage_obj = self.test_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.test_loaded_small_values
        else:
            self.value_storage_obj = self.test_loaded_values

        self.value_storage_obj.setup_linear()
        self.data = self.value_storage_obj.data
        self.precalc_interpolation = self.value_storage_obj.precalc_interpolation
        # set precalculated expected extrapolation results
        # this is the result of the type of extrapolation on self.xsamples_extrap
        self.setup_extrpolation_type(extrapolator_type)

        # set interpolator
        self.interpolator = Interpolate1DArray(self.x, self.data, 'linear', extrapolator_type, extrapolation_range)

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        Once executed, self.precalc_NNN members variables will contain
        precalculated extrapolated / interpolated data. self.interpolator
        will hold Interpolate1DArray object that is being tested. Generated using scipy
        version 1.6.3 scipy.interpolate.CubicHermiteSpline, with input gradients.

        :param extrapolator_type: type of extrapolator 'none', 'linear' or 'cubic'
        :param extrapolation_range: padding around interpolation range where extrapolation is possible
        :param big_values: For loading and testing big value saved data
        :param small_values: For loading and testing small value saved data
        """

        # set precalculated expected interpolation results
        # this is the result of sampling self.data on self.xsamples
        if big_values:
            self.value_storage_obj = self.test_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.test_loaded_small_values
        else:
            self.value_storage_obj = self.test_loaded_values

        self.value_storage_obj.setup_cubic()
        self.data = self.value_storage_obj.data
        self.precalc_interpolation = self.value_storage_obj.precalc_interpolation

        self.setup_extrpolation_type(extrapolator_type)
        # set interpolator
        self.interpolator = Interpolate1DArray(self.x, self.data, 'cubic', extrapolator_type, extrapolation_range)

    def setup_extrpolation_type(self, extrapolator_type: str):
        if extrapolator_type == 'linear':
            self.precalc_extrapolation = np.copy(self.value_storage_obj.precalc_extrapolation_linear)
        elif extrapolator_type == 'nearest':
            self.precalc_extrapolation = np.copy(self.value_storage_obj.precalc_extrapolation_nearest)
        elif extrapolator_type == 'none':
            self.precalc_extrapolation = None
        elif extrapolator_type == 'quadratic':
            self.precalc_extrapolation = np.copy(self.value_storage_obj.precalc_extrapolation_quadratic)
        else:
            raise ValueError(
                f'Extrapolation type {extrapolator_type} not found or no test. options are {id_to_extrapolator.keys()}'
            )

    def test_extrapolation_none(self):
        self.setup_linear('none', EXTRAPOLATION_RANGE, big_values=False, small_values=False)
        self.assertRaises(ValueError, self.interpolator, self.xsamples_extrap[1])
        self.assertRaises(ValueError, self.interpolator, self.xsamples_extrap[2])

    def test_linear_interpolation_extrapolators(self):
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
            self.run_general_interpolation_tests()

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
            self.run_general_interpolation_tests()

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_linear(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)

            self.run_general_interpolation_tests()

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against scipy.interpolate.CubicHermiteSpline with the same gradient calculations
        """
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
            self.run_general_interpolation_tests()

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)
            self.run_general_interpolation_tests()

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            self.setup_cubic(extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True)
            if extrapolator_type != 'none':
                if extrapolator_type == 'nearest':
                    gradient_continuity = False
                else:
                    gradient_continuity = True
                self.run_general_extrapolation_tests(gradient_continuity=gradient_continuity)

            self.run_general_interpolation_tests()

    def run_general_extrapolation_tests(self, gradient_continuity=True):
        # Test extrapolator out of range, there should be an error raised
        self.assertRaises(ValueError, self.interpolator, self.xsamples_extrap[0])
        self.assertRaises(ValueError, self.interpolator, self.xsamples_extrap[-1])

        # Test extrapolation inside extrapolation range matches the predefined values
        delta_max = np.abs(self.precalc_extrapolation[1]/np.power(10., PRECISION - 1))
        self.assertAlmostEqual(
            self.interpolator(self.xsamples_extrap[1]), self.precalc_extrapolation[1], delta=delta_max
        )
        delta_max = np.abs(self.precalc_extrapolation[2]/np.power(10., PRECISION - 1))
        self.assertAlmostEqual(
            self.interpolator(self.xsamples_extrap[2]), self.precalc_extrapolation[2], delta=delta_max
        )

        # Test gradient continuity between interpolation and extrapolation
        delta_max_lower = np.abs(self.precalc_interpolation[0] / np.power(10., PRECISION - 1))
        delta_max_upper = np.abs(self.precalc_interpolation[-1] / np.power(10., PRECISION - 1))
        if gradient_continuity:
            gradients_lower, gradients_upper = self.interpolator.test_edge_gradients()
            self.assertAlmostEqual(gradients_lower[0], gradients_lower[1], delta=delta_max_lower)
            self.assertAlmostEqual(gradients_upper[0], gradients_upper[1], delta=delta_max_upper)

    def run_general_interpolation_tests(self):
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            delta_max = np.abs(self.precalc_interpolation[i] / np.power(10., PRECISION - 1))
            self.assertAlmostEqual(
                self.interpolator(self.xsamples[i]), self.precalc_interpolation[i], delta=delta_max
            )

    def initialise_tests_on_interpolators(self, x_values, f_values):
        # Test for all combinations
        for extrapolator_type in id_to_extrapolator.keys():
            for interpolator_type in id_to_interpolator.keys():
                dict_kwargs_interpolators = {
                    'x': x_values, 'f': f_values, 'interpolation_type': interpolator_type,
                    'extrapolation_type': extrapolator_type, 'extrapolation_range': 2.0
                }
                self.assertRaises(ValueError, Interpolate1DArray, **dict_kwargs_interpolators)

    def test_initialisation_errors(self):
        # monotonicity
        x_wrong = np.copy(self.x)
        x_wrong[0] = self.x[1]
        x_wrong[1] = self.x[0]
        self.initialise_tests_on_interpolators(x_wrong, self.test_loaded_values.data)

        # test repeated coordinate
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(x_wrong, self.test_loaded_values.data)

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(x_wrong, self.test_loaded_values.data)

        # Todo self._x_mv = x and self._f_mv = f need to be initialised after array checks
        # Test array length 1
        test_on = False
        if test_on:
            # Arrays are too short
            x_wrong = np.copy(self.x)
            f_wrong = np.copy(self.data)
            x_wrong = x_wrong[0]
            f_wrong = f_wrong[0]
            self.initialise_tests_on_interpolators(x_wrong, f_wrong)

        # Incorrect dimension (2D data)
        x_wrong = np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1))
        f_wrong = np.array(np.concatenate((
            np.copy(self.test_loaded_values.data)[:, np.newaxis], np.copy(self.test_loaded_values.data)[:, np.newaxis]
        ), axis=1)
        )
        self.initialise_tests_on_interpolators(x_wrong, f_wrong)
