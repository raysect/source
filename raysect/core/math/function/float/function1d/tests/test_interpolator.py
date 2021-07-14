
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
from raysect.core.math.function.float.function1d.tests.data_store.interpolator1d_test_data import \
    TestInterpolatorLoadBigValues, TestInterpolatorLoadNormalValues, TestInterpolatorLoadSmallValues

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


class TestInterpolators1D(unittest.TestCase):
    """
    Testing class for 1D interpolators and extrapolators.
    """
    def setUp(self) -> None:

        # data is a precalculated input values for testing. It's the result of applying function f on self.x.
        # as in data = f(self.x), where self.x is linearly spaced between X_LOWER and X_UPPER.

        #: x values used to obtain data.
        self.x = np.linspace(X_LOWER, X_UPPER, NB_X)

        self.reference_loaded_values = TestInterpolatorLoadNormalValues()
        self.reference_loaded_big_values = TestInterpolatorLoadBigValues()
        self.reference_loaded_small_values = TestInterpolatorLoadSmallValues()

        #: x values on which interpolation_data were sampled on.
        self.xsamples = np.linspace(X_LOWER, X_UPPER, NB_XSAMPLES)

        #: x values on which extrapolation_data arrays were sampled on.
        self.xsamples_extrap = np.array(
            [
                X_LOWER - X_EXTRAP_DELTA_MAX,
                X_LOWER - X_EXTRAP_DELTA_MIN,
                X_UPPER + X_EXTRAP_DELTA_MIN,
                X_UPPER + X_EXTRAP_DELTA_MAX,
            ],
            dtype=np.float64,
        )

    def setup_linear(
            self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool) -> None:
        """
        Sets precalculated values for linear interpolator.
        Called in every test method that addresses linear interpolation.

        interpolator will hold Interpolate1DArray object that is being tested. Precalculated interpolation using
        scipy.interpolate.interp1d(kind=linear), generated using scipy version 1.6.3.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        """

        # Set precalculated expected interpolation results  using scipy.interpolate.interp1d(kind=linear).
        # This is the result of sampling data on self.xsamples.

        if big_values:
            self.value_storage_obj = self.reference_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.reference_loaded_small_values
        else:
            self.value_storage_obj = self.reference_loaded_values

        self.value_storage_obj.setup_linear()
        data = self.value_storage_obj.data
        interpolation_data = self.value_storage_obj.precalc_interpolation
        # set precalculated expected extrapolation results.
        # this is the result of the type of extrapolation on self.xsamples_extrap.
        extrapolation_data = self.setup_extrpolation_type(extrapolator_type)

        #: The interpolator object that is being tested. Set in setup_ method.
        interpolator = Interpolate1DArray(self.x, data, 'linear', extrapolator_type, extrapolation_range)
        return interpolator, interpolation_data, extrapolation_data

    def setup_cubic(self, extrapolator_type: str, extrapolation_range: float, big_values: bool, small_values: bool):
        """
        Sets precalculated values for cubic interpolator.
        Called in every test method that addresses cubic interpolation.

        interpolator will hold Interpolate1DArray object that is being tested. Generated using scipy
        version 1.6.3 scipy.interpolate.CubicHermiteSpline, with input gradients at the spline knots.

        :param extrapolator_type: type of extrapolator 'none' or 'linear'.
        :param extrapolation_range: padding around interpolation range where extrapolation is possible.
        :param big_values: For loading and testing big value saved data.
        :param small_values: For loading and testing small value saved data.
        """

        # Set precalculated expected interpolation results.
        # This is the result of sampling data on self.xsamples.
        if big_values:
            self.value_storage_obj = self.reference_loaded_big_values
        elif small_values:
            self.value_storage_obj = self.reference_loaded_small_values
        else:
            self.value_storage_obj = self.reference_loaded_values

        self.value_storage_obj.setup_cubic()
        data = self.value_storage_obj.data
        interpolation_data = self.value_storage_obj.precalc_interpolation

        extrapolation_data = self.setup_extrpolation_type(extrapolator_type)
        #: The interpolator object that is being tested. Set in setup_ method.
        interpolator = Interpolate1DArray(self.x, data, 'cubic', extrapolator_type, extrapolation_range)
        return interpolator, interpolation_data, extrapolation_data

    def setup_extrpolation_type(self, extrapolator_type: str):
        """
        Moving data from the selected data class to the extrapolation variable to be tested.
        """
        if extrapolator_type == 'linear':
            extrapolation_data = np.copy(self.value_storage_obj.precalc_extrapolation_linear)
        elif extrapolator_type == 'nearest':
            extrapolation_data = np.copy(self.value_storage_obj.precalc_extrapolation_nearest)
        elif extrapolator_type == 'none':
            extrapolation_data = None
        elif extrapolator_type == 'quadratic':
            extrapolation_data = np.copy(self.value_storage_obj.precalc_extrapolation_quadratic)
        else:
            raise ValueError(
                f'Extrapolation type {extrapolator_type} not found or no test. options are {id_to_extrapolator.keys()}'
            )
        return extrapolation_data

    def test_extrapolation_none(self):
        """
        Testing that extrapolator_type 'none' returns a ValueError rather than data when attempting to extrapolate
        outside its extrapolation range.
        """
        interpolator, _, _ = self.setup_linear(
            'none', EXTRAPOLATION_RANGE, big_values=False, small_values=False
        )
        self.assertRaises(ValueError, interpolator, self.xsamples_extrap[1])
        self.assertRaises(ValueError, interpolator, self.xsamples_extrap[2])

    def test_linear_interpolation_extrapolators(self):
        """
        Testing against linear interpolator objects for interpolation and extrapolation agreement.

        Testing for linear and nearest extrapolation are compared with simple external functions in
        raysect.core.math.function.function1d.tests.scripts.generate_1d_splines. interp1d(kind='linear') was used
        for linear interpolation.
        """
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False
            )
            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(interpolator, extrapolation_data)
            self.run_general_interpolation_tests(interpolator, interpolation_data)

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False
            )
            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(interpolator, extrapolation_data)
            self.run_general_interpolation_tests(interpolator, interpolation_data)

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_linear(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True
            )
            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(interpolator, extrapolation_data)

            self.run_general_interpolation_tests(interpolator, interpolation_data)

    def test_cubic_interpolation_extrapolators(self):
        """
        Testing against scipy.interpolate.CubicHermiteSpline with the same gradient calculations
        """
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=False
            )
            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(interpolator, extrapolation_data)
            self.run_general_interpolation_tests(interpolator, interpolation_data)

        # Tests for big values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=True, small_values=False
            )
            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(interpolator, extrapolation_data)
            self.run_general_interpolation_tests(interpolator, interpolation_data)

        # Tests for small values
        for extrapolator_type in id_to_extrapolator.keys():
            interpolator, interpolation_data, extrapolation_data = self.setup_cubic(
                extrapolator_type, EXTRAPOLATION_RANGE, big_values=False, small_values=True
            )
            if extrapolator_type != 'none':
                self.run_general_extrapolation_tests(interpolator, extrapolation_data)
            self.run_general_interpolation_tests(interpolator, interpolation_data)

    def run_general_extrapolation_tests(self, interpolator, extrapolation_data):
        # Test extrapolator out of range, there should be an error raised
        self.assertRaises(ValueError, interpolator, self.xsamples_extrap[0])
        self.assertRaises(ValueError, interpolator, self.xsamples_extrap[-1])

        # Test extrapolation inside extrapolation range matches the predefined values
        delta_max = np.abs(extrapolation_data[1]/np.power(10., PRECISION - 1))
        self.assertAlmostEqual(
            interpolator(self.xsamples_extrap[1]), extrapolation_data[1], delta=delta_max
        )
        delta_max = np.abs(extrapolation_data[2]/np.power(10., PRECISION - 1))
        self.assertAlmostEqual(
            interpolator(self.xsamples_extrap[2]), extrapolation_data[2], delta=delta_max
        )

    def run_general_interpolation_tests(self, interpolator, interpolation_data):
        # Test interpolation against xsample
        for i in range(len(self.xsamples)):
            delta_max = np.abs(interpolation_data[i] / np.power(10., PRECISION - 1))
            self.assertAlmostEqual(
                interpolator(self.xsamples[i]), interpolation_data[i], delta=delta_max
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
        self.initialise_tests_on_interpolators(x_wrong, self.reference_loaded_values.data)

        # test repeated coordinate
        x_wrong = np.copy(self.x)
        x_wrong[0] = x_wrong[1]
        self.initialise_tests_on_interpolators(x_wrong, self.reference_loaded_values.data)

        # mismatch array size between x and data
        x_wrong = np.copy(self.x)
        x_wrong = x_wrong[:-1]
        self.initialise_tests_on_interpolators(x_wrong, self.reference_loaded_values.data)

        # Test array length 1
        # Arrays are too short
        x_wrong = np.copy(self.x)
        f_wrong = np.copy(self.reference_loaded_values.data)
        x_wrong = x_wrong[0]
        f_wrong = f_wrong[0]
        self.initialise_tests_on_interpolators(x_wrong, f_wrong)

        # Incorrect dimension (2D data)
        x_wrong = np.array(np.concatenate((np.copy(self.x)[:, np.newaxis], np.copy(self.x)[:, np.newaxis]), axis=1))
        f_wrong = np.array(np.concatenate((
            np.copy(self.reference_loaded_values.data)[:, np.newaxis],
            np.copy(self.reference_loaded_values.data)[:, np.newaxis]
        ), axis=1)
        )
        self.initialise_tests_on_interpolators(x_wrong, f_wrong)
